//! macOS push-to-talk: hold the Option (⌥) key to record, release to
//! transcribe + paste.
//!
//! A bare modifier can't be a normal global shortcut (the shortcut plugin needs
//! a real key), so we observe the keyboard directly with an `NSEvent` global
//! monitor. Two `flagsChanged`/`keyDown` monitors are installed on the main run
//! loop; both fire on the main thread, so they drive the recording session
//! directly.
//!
//! Option is also an ordinary modifier (⌥+arrow to jump words, accented
//! characters, ⌥+click), so we must not open the mic every time it's tapped as
//! part of a chord. Two guards keep it honest:
//!   * a short hold delay — Option must be held ALONE for `HOLD_DELAY` before a
//!     recording begins, so a quick ⌥+key chord never starts one;
//!   * a keystroke cancel — any other key pressed while a PTT recording is
//!     pending or live aborts it (the user was typing a shortcut, not talking).
//!
//! Global keyboard monitors require the Accessibility permission, which the app
//! already holds for pasting via enigo.

#[cfg(target_os = "macos")]
pub use imp::install;

/// No-op on non-macOS: push-to-talk is a native-macOS feature for now.
#[cfg(not(target_os = "macos"))]
pub fn install(_app: &tauri::AppHandle) {}

#[cfg(target_os = "macos")]
mod imp {
    use std::ptr::NonNull;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use block2::RcBlock;
    use objc2_app_kit::{NSEvent, NSEventMask, NSEventModifierFlags};
    use tauri::AppHandle;

    /// How long ⌥ must be held alone before a recording begins.
    const HOLD_DELAY: Duration = Duration::from_millis(180);

    /// Shared state between the two monitors and the scheduled-start timer.
    struct Ptt {
        /// Bumped on every ⌥-down, every ⌥-up, every other modifier change, and
        /// every keystroke. A scheduled start only fires if the generation it
        /// captured still holds — so a release or a chord key aborts it.
        generation: AtomicU64,
        /// True while a PTT recording is actually live (mic open).
        recording: AtomicBool,
    }

    pub fn install(app: &AppHandle) {
        let state = Arc::new(Ptt {
            generation: AtomicU64::new(0),
            recording: AtomicBool::new(false),
        });

        install_flags_monitor(app, &state);
        install_keydown_monitor(app, &state);
    }

    /// ⌥ down (alone) schedules a start; ⌥ up stops a live recording.
    fn install_flags_monitor(app: &AppHandle, state: &Arc<Ptt>) {
        let app = app.clone();
        let state = state.clone();
        let handler = RcBlock::new(move |event: NonNull<NSEvent>| {
            let flags = unsafe { event.as_ref().modifierFlags() };
            let option = flags.contains(NSEventModifierFlags::Option);
            let others = flags.intersects(
                NSEventModifierFlags::Command
                    | NSEventModifierFlags::Control
                    | NSEventModifierFlags::Shift
                    | NSEventModifierFlags::Function,
            );

            if option && !others {
                // ⌥ pressed alone — schedule a start unless one is already live.
                if !state.recording.load(Ordering::Acquire) {
                    let generation = state.generation.fetch_add(1, Ordering::AcqRel) + 1;
                    schedule_start(app.clone(), state.clone(), generation);
                }
            } else {
                // Any other transition (⌥ released, or a chord modifier added)
                // invalidates a pending start; a full ⌥ release also stops a
                // live recording.
                state.generation.fetch_add(1, Ordering::AcqRel);
                if !option && state.recording.swap(false, Ordering::AcqRel) {
                    crate::ptt_stop(&app);
                }
            }
        });

        let token = NSEvent::addGlobalMonitorForEventsMatchingMask_handler(
            NSEventMask::FlagsChanged,
            &handler,
        );
        keep_alive(token, handler);
    }

    /// Any key press while a PTT recording is pending or live aborts it.
    fn install_keydown_monitor(app: &AppHandle, state: &Arc<Ptt>) {
        let app = app.clone();
        let state = state.clone();
        let handler = RcBlock::new(move |_event: NonNull<NSEvent>| {
            // Cancel a pending scheduled start ...
            state.generation.fetch_add(1, Ordering::AcqRel);
            // ... and abort a recording already in flight.
            if state.recording.swap(false, Ordering::AcqRel) {
                crate::ptt_cancel(&app);
            }
        });

        let token =
            NSEvent::addGlobalMonitorForEventsMatchingMask_handler(NSEventMask::KeyDown, &handler);
        keep_alive(token, handler);
    }

    /// After `HOLD_DELAY`, start recording iff ⌥ is still held alone (the
    /// generation this start captured is unchanged) and nothing else has begun.
    /// The actual start is dispatched back onto the main thread, where AppKit
    /// and the recorder expect to run.
    fn schedule_start(app: AppHandle, state: Arc<Ptt>, generation: u64) {
        std::thread::spawn(move || {
            std::thread::sleep(HOLD_DELAY);
            if state.generation.load(Ordering::Acquire) != generation
                || state.recording.load(Ordering::Acquire)
            {
                return;
            }
            let _ = app.clone().run_on_main_thread(move || {
                // Re-check on the main thread: the release/keystroke may have
                // landed between the timer and this dispatch.
                if state.generation.load(Ordering::Acquire) != generation
                    || state.recording.swap(true, Ordering::AcqRel)
                {
                    return;
                }
                crate::ptt_start(&app);
            });
        });
    }

    /// A global monitor stays active only while its token is retained, and the
    /// block must outlive the monitor. Both live for the app's lifetime, so we
    /// leak them rather than thread a handle through app state.
    fn keep_alive(token: Option<objc2::rc::Retained<objc2::runtime::AnyObject>>, handler: RcBlock<dyn Fn(NonNull<NSEvent>)>) {
        if let Some(token) = token {
            std::mem::forget(token);
        } else {
            log::error!("Failed to install ⌥ push-to-talk monitor");
        }
        std::mem::forget(handler);
    }
}
