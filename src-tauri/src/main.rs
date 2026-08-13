#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod models;
#[cfg(target_os = "macos")]
mod nspanel;
mod overlay;
mod paste;
mod settings;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::menu::{MenuBuilder, MenuItemBuilder};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, State, WindowEvent};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

use paste::EnigoState;
use pie_engine::audio::{
    AudioRecorder, SileroVad, VadCache, VadPipeline, VadPolicy, VoiceActivityDetector,
    PIE_VAD_THRESHOLD, VAD_CONTEXT_FRAMES, VAD_HANGOVER_FRAMES, VAD_SPEECH_THRESHOLD_FRAMES,
    VAD_STREAM_HANGOVER_FRAMES,
};
use pie_engine::history::{HistoryStore, NewEntry};
use pie_engine::stt::{SttEngine, WhisperEngine};
use pie_engine::PieEngine;
use settings::Settings;

/// Everything the commands share. Locks are scoped and never held across
/// an await; heavy work (model load, transcription) runs on blocking threads.
struct AppState {
    settings: Mutex<Settings>,
    /// Open while a recording session is active.
    recorder: Mutex<Option<AudioRecorder>>,
    /// Cached Silero VAD session, reused across recordings so the ONNX model
    /// isn't rebuilt from disk on every start. Keyed by model path.
    vad_cache: Mutex<VadCache>,
    /// True from stop until processing finishes; the hotkey ignores presses
    /// while set so a double-tap can't start a session mid-decode.
    busy: AtomicBool,
    /// Loaded whisper engine, cached with the (path, language) it was built
    /// from so a settings change reloads it.
    whisper: Mutex<Option<(PathBuf, String, Arc<WhisperEngine>)>>,
    /// Text pipeline: intent -> memory -> optimizer -> LLM router.
    engine: tokio::sync::Mutex<PieEngine>,
    /// Local SQLite history of recordings.
    history: Mutex<HistoryStore>,
    /// Paste mode captured when the current recording started: "transcript" or
    /// "prompt". Set from the hotkey that fired (raw vs optimized), or from
    /// `settings.paste_output` for the UI record button. Read by the stop/paste
    /// path and the refine gate.
    pending_paste_mode: Mutex<String>,
}

/// Result payload for the frontend after a recording is processed.
#[derive(Clone, Serialize)]
struct Outcome {
    transcript: String,
    objective: String,
    conversation_type: String,
    confidence: String,
    optimized_prompt: String,
    estimated_tokens: usize,
    mode: String,
    applied: Vec<AppliedFixDto>,
}

/// A single pronunciation correction applied to the transcript, surfaced to
/// the frontend for transparency (e.g. "next jazz" -> "Next.js").
#[derive(Clone, Serialize)]
struct AppliedFixDto {
    from: String,
    to: String,
    tier: String,
}

/// Emit an event to the frontend, logging (rather than silently dropping) any
/// failure so a broken event channel is visible in the logs.
fn emit_event<S: Serialize + Clone>(app: &AppHandle, event: &str, payload: S) {
    if let Err(e) = app.emit(event, payload) {
        log::warn!("Failed to emit {event}: {e}");
    }
}

fn emit_state(app: &AppHandle, state: &str) {
    emit_event(app, "pie://state", state);
    // The floating overlay is only visible while a session is in flight.
    match state {
        "recording" | "decoding" => overlay::show_overlay(app),
        _ => overlay::hide_overlay(app),
    }
}

/// Bring the main window to the front (tray click / menu).
fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

/* ─── recording session (shared by UI commands and the global hotkey) ─── */

fn do_start_recording(app: &AppHandle) -> Result<(), String> {
    let state = app.try_state::<AppState>().ok_or("App state unavailable")?;
    let mut recorder_slot = state.recorder.lock().unwrap_or_else(|e| e.into_inner());
    if recorder_slot.is_some() {
        return Err("Already recording".to_string());
    }
    if state.busy.load(Ordering::Acquire) {
        return Err("Still processing the previous recording".to_string());
    }

    let settings = state
        .settings
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    // NOTE: the paste mode for this recording is set by the caller before
    // starting — on_hotkey_with_mode sets the hotkey's mode; the UI
    // `start_recording` command sets `settings.paste_output`. do_start_recording
    // must NOT touch pending_paste_mode or it would clobber the hotkey's choice.
    let (mut recorder, vad_active) = {
        let mut vad_cache = state.vad_cache.lock().unwrap_or_else(|e| e.into_inner());
        build_recorder(app, &settings, &mut vad_cache).map_err(|e| e.to_string())?
    };
    recorder.open(None).map_err(|e| e.to_string())?;
    recorder
        .start(if vad_active {
            VadPolicy::Offline
        } else {
            VadPolicy::Disabled
        })
        .map_err(|e| e.to_string())?;

    *recorder_slot = Some(recorder);
    emit_state(app, "recording");
    Ok(())
}

async fn do_stop_recording(app: AppHandle) -> Result<Outcome, String> {
    let state = app.try_state::<AppState>().ok_or("App state unavailable")?;

    // 1. Capture: take the recorder out and collect the session's samples.
    let samples = {
        let mut recorder_slot = state.recorder.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mut recorder) = recorder_slot.take() else {
            return Err("Not recording".to_string());
        };
        let samples = recorder.stop().map_err(|e| e.to_string())?;
        recorder.close().map_err(|e| e.to_string())?;
        samples
    };
    state.busy.store(true, Ordering::Release);
    emit_state(&app, "decoding");

    let result = transcribe_and_process(&app, samples).await;

    state.busy.store(false, Ordering::Release);
    emit_state(&app, "idle");
    result
}

/// Stop and discard an in-flight recording without transcribing, and reset the
/// UI. Shared by the Cancel button and the in-window Escape key. Escape is
/// handled in the webview only (never grabbed globally) so it can't interfere
/// with Escape in other apps.
fn do_cancel_recording(app: &AppHandle) {
    if let Some(state) = app.try_state::<AppState>() {
        let mut slot = state.recorder.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(mut recorder) = slot.take() {
            let _ = recorder.stop();
            let _ = recorder.close();
        }
    }
    emit_state(app, "idle");
}

async fn transcribe_and_process(app: &AppHandle, samples: Vec<f32>) -> Result<Outcome, String> {
    let state = app.try_state::<AppState>().ok_or("App state unavailable")?;
    if samples.len() < 1600 {
        return Err("Recording too short (or VAD detected no speech)".to_string());
    }

    let settings = state
        .settings
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();

    // 2. Transcribe on a blocking thread (Metal/CPU inference).
    let whisper = get_or_load_whisper(&state, &settings)?;
    let transcript = tauri::async_runtime::spawn_blocking(move || {
        whisper.transcribe(&samples).map(|t| t.trim().to_string())
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;

    if transcript.is_empty() {
        return Err("Transcription produced no text (silence?)".to_string());
    }

    // 3. Intent + optimization through the shared pipeline.
    let mut engine = state.engine.lock().await;
    let result = engine
        .process(&transcript, &settings.mode)
        .await
        .map_err(|e| e.to_string())?;

    // Opt-in deep correction (off the always-on path): re-run the LLM pass over
    // the deterministic result and fold in any extra fixes it finds. Never
    // fails the recording — an LLM error just falls back to the deterministic
    // transcript already computed above.
    //
    // Skip it in code mode: the transcript here already contains translated code
    // syntax ("console.log(", "==="), and deep-correct is a *pronunciation*
    // fixer — running it over syntax would likely undo the translation. The two
    // opt-ins don't compose, so code mode wins.
    let (final_transcript, applied_fixes) = if settings.deep_correct_ai && !settings.code_mode {
        match engine
            .deep_correct(
                &result.corrected_transcript,
                &settings.provider,
                model_opt(&settings),
            )
            .await
        {
            Ok(deep) => {
                let mut applied = result.applied.clone();
                applied.extend(deep.applied);
                (deep.text, applied)
            }
            Err(e) => {
                log::warn!("deep-correct failed, using deterministic result: {e}");
                (result.corrected_transcript.clone(), result.applied.clone())
            }
        }
    } else {
        (result.corrected_transcript.clone(), result.applied.clone())
    };
    drop(engine);

    let outcome = Outcome {
        transcript: final_transcript,
        objective: result.intent.objective,
        conversation_type: format!("{:?}", result.intent.conversation_type),
        confidence: format!("{:?}", result.intent.confidence),
        optimized_prompt: result.optimized_prompt,
        estimated_tokens: result.estimated_tokens,
        mode: format!("{:?}", result.mode),
        applied: applied_fixes
            .iter()
            .map(|f| AppliedFixDto {
                from: f.from.clone(),
                to: f.to.clone(),
                tier: format!("{:?}", f.tier),
            })
            .collect(),
    };

    // Best-effort history capture — never fail the recording over it.
    {
        let entry = NewEntry {
            transcript: outcome.transcript.clone(),
            objective: opt(&outcome.objective),
            conversation_type: opt(&outcome.conversation_type),
            confidence: opt(&outcome.confidence),
            optimized_prompt: opt(&outcome.optimized_prompt),
            estimated_tokens: Some(outcome.estimated_tokens as i64),
            mode: opt(&outcome.mode),
            language: opt(&settings.language),
        };
        let history = state.history.lock().unwrap_or_else(|e| e.into_inner());
        match history.add(entry, settings.history_limit) {
            Ok(_) => emit_event(app, "pie://history-changed", ()),
            Err(e) => log::warn!("Failed to record history: {e}"),
        }
    }

    Ok(outcome)
}

/* ─── global hotkey ─── */

/// Toggle handler carrying a paste mode: first press starts recording (and
/// records which output this hotkey wants — "transcript" or "prompt"), second
/// press stops, transcribes, and pastes that output into the focused app.
fn on_hotkey_with_mode(app: &AppHandle, mode: &str) {
    let Some(state) = app.try_state::<AppState>() else {
        log::error!("on_hotkey: app state unavailable");
        return;
    };
    let recording = state
        .recorder
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .is_some();
    log::debug!("on_hotkey: recording={recording} mode={mode}");

    if !recording {
        // Capture the paste mode for this recording BEFORE starting, so the
        // stop/paste path (and the refine gate) use the mode this hotkey wants.
        *state
            .pending_paste_mode
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = mode.to_string();
        if let Err(e) = do_start_recording(app) {
            log::warn!("Hotkey start failed: {e}");
            emit_event(app, "pie://error", e);
        }
        return;
    }

    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        match do_stop_recording(app.clone()).await {
            Ok(outcome) => {
                // Show the result in the window (if it's open) ...
                emit_event(&app, "pie://outcome", outcome.clone());

                // ... and paste into whichever app has focus.
                let Some(state) = app.try_state::<AppState>() else {
                    log::error!("hotkey paste: app state unavailable");
                    return;
                };
                let paste_mode = state
                    .pending_paste_mode
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                let text = if paste_mode == "prompt" {
                    outcome.optimized_prompt
                } else {
                    outcome.transcript
                };
                let result = tauri::async_runtime::spawn_blocking(move || {
                    let enigo = app
                        .try_state::<EnigoState>()
                        .ok_or("Keystroke engine unavailable")?;
                    paste::paste_text(&app, &enigo, &text)
                })
                .await;
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => log::error!("Paste failed: {e}"),
                    Err(e) => log::error!("Paste task failed: {e}"),
                }
            }
            Err(e) => {
                log::warn!("Hotkey stop failed: {e}");
                emit_event(&app, "pie://error", e);
            }
        }
    });
}

/// (Re-)register BOTH global hotkeys from settings. Clears all existing
/// bindings first, then registers each; an invalid or empty binding is logged
/// and skipped — never fatal, and one bad hotkey never blocks the other.
fn register_hotkeys(app: &AppHandle, settings: &Settings) -> Result<(), String> {
    let _ = app.global_shortcut().unregister_all();
    register_one(app, &settings.hotkey_raw, "transcript");
    // If both hotkeys resolve to the same combo (e.g. a legacy install whose
    // single hotkey migrated into `hotkey_raw` while `hotkey_optimized` kept its
    // identical default), register only the first — a double registration of one
    // shortcut is ambiguous. Raw wins, preserving the legacy transcript output.
    let opt = settings.hotkey_optimized.trim();
    if !opt.is_empty() && opt == settings.hotkey_raw.trim() {
        log::warn!("optimized hotkey equals the raw hotkey ('{opt}'); skipping the optimized binding so the raw/transcript hotkey wins — rebind one in Settings");
    } else {
        register_one(app, &settings.hotkey_optimized, "prompt");
    }
    Ok(())
}

/// Register one shortcut bound to a paste `mode`. Empty = disabled; a parse or
/// registration failure is logged and skipped (non-fatal).
fn register_one(app: &AppHandle, hotkey: &str, mode: &'static str) {
    let trimmed = hotkey.trim();
    if trimmed.is_empty() {
        log::info!("Hotkey ({mode}) disabled");
        return;
    }
    let shortcut: Shortcut = match trimmed.parse() {
        Ok(s) => s,
        Err(e) => {
            log::error!("Invalid hotkey '{hotkey}' ({mode}): {e}");
            return;
        }
    };
    let res = app
        .global_shortcut()
        .on_shortcut(shortcut, move |app, fired, event| {
            if event.state() == ShortcutState::Pressed {
                log::debug!("Hotkey fired ({mode}): {fired:?}");
                on_hotkey_with_mode(app, mode);
            }
        });
    match res {
        Ok(()) => log::info!("Hotkey registered ({mode}): {hotkey}"),
        Err(e) => log::error!("Failed to register hotkey '{hotkey}' ({mode}): {e}"),
    }
}

/* ─── commands ─── */

#[tauri::command]
fn get_settings(state: State<'_, AppState>) -> Settings {
    state
        .settings
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

#[tauri::command]
async fn update_settings(
    app: AppHandle,
    state: State<'_, AppState>,
    settings: Settings,
) -> Result<(), String> {
    let hotkeys_changed = {
        let current = state.settings.lock().unwrap_or_else(|e| e.into_inner());
        current.hotkey_raw != settings.hotkey_raw
            || current.hotkey_optimized != settings.hotkey_optimized
    };
    // Re-register both hotkeys when either changed. Registration is non-fatal
    // (a bad binding is logged and skipped), so this can't fail the save.
    if hotkeys_changed {
        register_hotkeys(&app, &settings)?;
    }
    let (llm_changed, code_mode_changed) = {
        let current = state.settings.lock().unwrap_or_else(|e| e.into_inner());
        let llm = current.llm_api_url != settings.llm_api_url
            || current.llm_api_key != settings.llm_api_key
            || current.llm_model != settings.llm_model;
        (llm, current.code_mode != settings.code_mode)
    };
    settings.save().map_err(|e| e.to_string())?;
    if llm_changed || code_mode_changed {
        let mut engine = state.engine.lock().await;
        if llm_changed {
            engine.set_llm_config(&llm_config(&settings));
        }
        if code_mode_changed {
            engine.set_code_mode(settings.code_mode);
        }
    }
    *state.settings.lock().unwrap_or_else(|e| e.into_inner()) = settings;
    Ok(())
}

/// Suspend (active=false) or restore (active=true) the global hotkey. The
/// Settings shortcut recorder suspends it while capturing so the current
/// binding doesn't fire on the keys being pressed to choose a new one.
#[tauri::command]
fn set_hotkey_active(
    app: AppHandle,
    state: State<'_, AppState>,
    active: bool,
) -> Result<(), String> {
    if active {
        let settings = state
            .settings
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        register_hotkeys(&app, &settings)
    } else {
        app.global_shortcut()
            .unregister_all()
            .map_err(|e| e.to_string())
    }
}

#[tauri::command]
fn start_recording(app: AppHandle) -> Result<(), String> {
    // UI record button: paste the configured default output.
    if let Some(state) = app.try_state::<AppState>() {
        let default_mode = state
            .settings
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .paste_output
            .clone();
        *state
            .pending_paste_mode
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = default_mode;
    }
    do_start_recording(&app)
}

#[tauri::command]
async fn stop_recording(app: AppHandle) -> Result<Outcome, String> {
    do_stop_recording(app).await
}

#[tauri::command]
fn cancel_recording(app: AppHandle) -> Result<(), String> {
    do_cancel_recording(&app);
    Ok(())
}

#[tauri::command]
fn list_models(state: State<'_, AppState>) -> Vec<models::ModelInfo> {
    let settings = state.settings.lock().unwrap_or_else(|e| e.into_inner());
    models::list_models(&settings)
}

/// Point the relevant setting at an already-downloaded catalog model.
#[tauri::command]
fn select_model(app: AppHandle, state: State<'_, AppState>, id: String) -> Result<(), String> {
    let (kind, path) = models::resolve(&id).ok_or("Unknown model")?;
    if !models::is_downloaded(&id) {
        return Err("Model isn't downloaded yet".to_string());
    }
    let path_str = path.to_string_lossy().into_owned();
    let settings = {
        let mut settings = state.settings.lock().unwrap_or_else(|e| e.into_inner());
        match kind {
            models::ModelKind::Whisper => settings.whisper_model = path_str,
            models::ModelKind::Vad => settings.silero_model = path_str,
        }
        settings.clone()
    };
    settings.save().map_err(|e| e.to_string())?;
    emit_event(&app, "pie://models-changed", ());
    // Warm the newly selected whisper model in the background so it's hot before
    // first use. VAD selection doesn't need this — the VAD cache loads on the
    // next recording start regardless.
    if matches!(kind, models::ModelKind::Whisper) {
        warm_whisper(&app);
    }
    Ok(())
}

/// Delete a downloaded model file. Refuses if the model is currently selected.
#[tauri::command]
fn delete_model(app: AppHandle, state: State<'_, AppState>, id: String) -> Result<(), String> {
    let (kind, path) = models::resolve(&id).ok_or("Unknown model")?;
    if !models::is_downloaded(&id) {
        return Err("Model isn't downloaded".to_string());
    }

    // Block deletion of the active model.
    let settings = state.settings.lock().unwrap_or_else(|e| e.into_inner());
    let selected_path = match kind {
        models::ModelKind::Whisper => Settings::expand(&settings.whisper_model),
        models::ModelKind::Vad => Settings::expand(&settings.silero_model),
    };
    if selected_path == path {
        return Err("Can't delete the model currently in use".to_string());
    }
    drop(settings);

    models::remove(&id).map_err(|e| format!("Failed to delete {}: {e}", path.display()))?;
    log::info!("Deleted model: {}", path.display());
    emit_event(&app, "pie://models-changed", ());
    Ok(())
}

/// Stream a catalog model to disk, emitting `pie://download` progress. On
/// success, auto-selects it so it's ready to use.
#[tauri::command]
async fn download_model(app: AppHandle, id: String) -> Result<(), String> {
    use serde::Serialize;
    use std::time::{Duration, Instant};

    #[derive(Clone, Serialize)]
    struct Progress {
        id: String,
        received: u64,
        total: u64,
        done: bool,
        error: Option<String>,
    }

    if models::resolve(&id).is_none() {
        return Err("Unknown model".to_string());
    }

    // Throttle progress events to ~10/s so the event channel isn't flooded.
    // Also track the latest counts (unthrottled) so the final "done" event
    // reports accurate totals across the model's whole fileset.
    let mut last_emit = Instant::now();
    let mut last_counts = (0u64, 0u64);
    let result = models::download_model(&id, |received, total| {
        last_counts = (received, total);
        if last_emit.elapsed() >= Duration::from_millis(100) {
            last_emit = Instant::now();
            emit_event(
                &app,
                "pie://download",
                Progress {
                    id: id.clone(),
                    received,
                    total,
                    done: false,
                    error: None,
                },
            );
        }
    })
    .await;

    match result {
        Ok(_path) => {
            let (received, total) = last_counts;
            emit_event(
                &app,
                "pie://download",
                Progress {
                    id: id.clone(),
                    received,
                    total: total.max(received),
                    done: true,
                    error: None,
                },
            );
            // Auto-select the freshly downloaded model (best-effort).
            if let Some(state) = app.try_state::<AppState>() {
                let _ = select_model(app.clone(), state, id);
            }
            Ok(())
        }
        Err(e) => {
            emit_event(
                &app,
                "pie://download",
                Progress {
                    id,
                    received: 0,
                    total: 0,
                    done: true,
                    error: Some(e.clone()),
                },
            );
            Err(e)
        }
    }
}

#[tauri::command]
async fn send_to_llm(state: State<'_, AppState>, prompt: String) -> Result<String, String> {
    let (provider, model) = {
        let settings = state.settings.lock().unwrap_or_else(|e| e.into_inner());
        (settings.provider.clone(), settings.llm_model.clone())
    };
    let model = (!model.is_empty()).then_some(model);

    let engine = state.engine.lock().await;
    engine
        .send_to_llm(&prompt, &provider, model.as_deref())
        .await
        .map_err(|e| e.to_string())
}

/// Verify typed-but-unsaved LLM credentials with a minimal round-trip.
/// Returns Ok("connected") on success, Err(message) otherwise.
#[tauri::command]
async fn test_llm_connection(url: String, key: String, model: String) -> Result<String, String> {
    if url.trim().is_empty() {
        return Err("API URL is required".to_string());
    }
    let client = pie_engine::llm::openai::OpenAiClient::new(&url, &key);
    let model = if model.trim().is_empty() {
        "gpt-4o-mini".to_string()
    } else {
        model
    };
    match client
        .chat_with_timeout("ping", &model, std::time::Duration::from_secs(10))
        .await
    {
        Ok(_) => Ok("connected".to_string()),
        Err(e) => Err(e.to_string()),
    }
}

#[tauri::command]
fn copy_to_clipboard(app: AppHandle, text: String) -> Result<(), String> {
    use tauri_plugin_clipboard_manager::ClipboardExt;
    app.clipboard()
        .write_text(text)
        .map_err(|e| format!("Failed to copy: {e}"))
}

#[tauri::command]
fn list_history(
    state: State<'_, AppState>,
    query: Option<String>,
) -> Result<Vec<pie_engine::history::HistoryEntry>, String> {
    let limit = state
        .settings
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .history_limit;
    let history = state.history.lock().unwrap_or_else(|e| e.into_inner());
    history
        .list(query.as_deref(), limit)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn delete_history_entry(app: AppHandle, state: State<'_, AppState>, id: i64) -> Result<(), String> {
    {
        let history = state.history.lock().unwrap_or_else(|e| e.into_inner());
        history.delete(id).map_err(|e| e.to_string())?;
    }
    emit_event(&app, "pie://history-changed", ());
    Ok(())
}

#[tauri::command]
fn clear_history(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    {
        let history = state.history.lock().unwrap_or_else(|e| e.into_inner());
        history.clear().map_err(|e| e.to_string())?;
    }
    emit_event(&app, "pie://history-changed", ());
    Ok(())
}

// Synchronous: it blocks with sleeps + keystroke simulation. Tauri runs sync
// commands on its own thread pool, so this won't stall the async runtime.
#[tauri::command]
fn paste_history_entry(
    app: AppHandle,
    state: State<'_, AppState>,
    enigo: State<'_, EnigoState>,
    id: i64,
) -> Result<(), String> {
    let text = {
        let history = state.history.lock().unwrap_or_else(|e| e.into_inner());
        history
            .get(id)
            .map_err(|e| e.to_string())?
            .map(|r| r.transcript)
            .ok_or_else(|| "History entry not found".to_string())?
    };

    // Hide the main window so focus returns to the previously active app,
    // then paste into it (same mechanism as the hotkey flow).
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.hide();
    }
    std::thread::sleep(std::time::Duration::from_millis(120));
    paste::paste_text(&app, &enigo, &text)
}

/// A user-defined heard->canonical pronunciation correction.
#[derive(Clone, Serialize)]
struct CorrectionDto {
    heard: String,
    canonical: String,
}

#[tauri::command]
async fn list_corrections(state: State<'_, AppState>) -> Result<Vec<CorrectionDto>, String> {
    let engine = state.engine.lock().await;
    Ok(engine
        .corrector_user_corrections()
        .into_iter()
        .map(|c| CorrectionDto {
            heard: c.heard,
            canonical: c.canonical,
        })
        .collect())
}

#[tauri::command]
async fn add_correction(
    state: State<'_, AppState>,
    heard: String,
    canonical: String,
) -> Result<(), String> {
    let mut engine = state.engine.lock().await;
    engine
        .corrector_add(&heard, &canonical)
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn delete_correction(state: State<'_, AppState>, heard: String) -> Result<(), String> {
    let mut engine = state.engine.lock().await;
    engine.corrector_remove(&heard).map_err(|e| e.to_string())
}

/// Number of learned (auto-mined + synced) corrections, for the vocab UI.
#[tauri::command]
async fn get_learned_vocab_count(state: State<'_, AppState>) -> Result<usize, String> {
    let engine = state.engine.lock().await;
    Ok(engine.corrector_learned_count())
}

/// Clear all learned/synced corrections (user corrections are untouched).
#[tauri::command]
async fn reset_learned_vocab(state: State<'_, AppState>) -> Result<(), String> {
    let mut engine = state.engine.lock().await;
    engine.corrector_reset_learned().map_err(|e| e.to_string())
}

/// On-demand deep-correct: run the LLM pass over an already-produced
/// transcript (e.g. from a past recording) and re-derive intent/optimization
/// from the corrected text, without re-recording.
#[tauri::command]
async fn recorrect_with_ai(
    state: State<'_, AppState>,
    transcript: String,
) -> Result<Outcome, String> {
    let settings = {
        state
            .settings
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    };
    let mut engine = state.engine.lock().await;
    let deep = engine
        .deep_correct(&transcript, &settings.provider, model_opt(&settings))
        .await
        .map_err(|e| e.to_string())?;
    // Re-run intent/optimize on the deep-corrected text for a fresh Outcome.
    let result = engine
        .process(&deep.text, &settings.mode)
        .await
        .map_err(|e| e.to_string())?;
    Ok(Outcome {
        transcript: deep.text,
        objective: result.intent.objective,
        conversation_type: format!("{:?}", result.intent.conversation_type),
        confidence: format!("{:?}", result.intent.confidence),
        optimized_prompt: result.optimized_prompt,
        estimated_tokens: result.estimated_tokens,
        mode: format!("{:?}", result.mode),
        applied: deep
            .applied
            .iter()
            .map(|f| AppliedFixDto {
                from: f.from.clone(),
                to: f.to.clone(),
                tier: format!("{:?}", f.tier),
            })
            .collect(),
    })
}

/// An auto-detected local source of past conversations, for the vocabulary
/// sync UI to offer alongside manual folder/file import.
#[derive(Clone, Serialize)]
struct SyncSourceDto {
    name: String,
    path: String,
}

/// Outcome of a vocabulary sync run, for the frontend summary.
#[derive(Clone, Serialize)]
struct SyncResultDto {
    conversations: usize,
    terms_added: usize,
    /// Number of LLM batches sent this run.
    batches_total: usize,
    /// Number of those batches that failed (LLM error or unparseable
    /// reply) — counts only, never error content.
    batches_failed: usize,
}

/// Persisted record of the last vocabulary sync, for the settings UI.
#[derive(Clone, Serialize)]
struct SyncStateDto {
    last_run_unix: u64,
    terms_added: usize,
}

/// Open a native folder or file picker for vocabulary import. Returns the
/// chosen path, or `None` if the user cancelled.
#[tauri::command]
async fn pick_import_target(folder: bool) -> Result<Option<String>, String> {
    let picked = if folder {
        rfd::FileDialog::new().pick_folder()
    } else {
        rfd::FileDialog::new()
            .add_filter("conversations", &["json", "md", "txt", "markdown"])
            .pick_file()
    };
    Ok(picked.map(|p| p.to_string_lossy().to_string()))
}

/// Auto-detect local sources of past conversations. Currently checks for
/// Cursor's `state.vscdb` (macOS/Linux paths). The manual import option is
/// always surfaced by the frontend regardless of this result.
#[tauri::command]
fn discover_sync_sources() -> Result<Vec<SyncSourceDto>, String> {
    let mut sources = Vec::new();

    let cursor_vscdb = if cfg!(target_os = "macos") {
        dirs::home_dir()
            .map(|h| h.join("Library/Application Support/Cursor/User/globalStorage/state.vscdb"))
    } else {
        dirs::home_dir().map(|h| h.join(".config/Cursor/User/globalStorage/state.vscdb"))
    };

    if let Some(path) = cursor_vscdb {
        if path.exists() {
            sources.push(SyncSourceDto {
                name: "Cursor".to_string(),
                path: path.to_string_lossy().to_string(),
            });
        }
    }

    Ok(sources)
}

/// Run a vocabulary sync over the given paths via the configured LLM,
/// emitting `sync-progress` `(done, total)` events as batches complete.
#[tauri::command]
async fn run_vocabulary_sync(
    app: AppHandle,
    state: State<'_, AppState>,
    paths: Vec<String>,
) -> Result<SyncResultDto, String> {
    let settings = {
        state
            .settings
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    };

    let paths: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();

    let mut engine = state.engine.lock().await;
    let result = engine
        .run_vocabulary_sync(
            paths,
            &settings.provider,
            model_opt(&settings),
            |done, total| {
                emit_event(&app, "sync-progress", (done, total));
            },
        )
        .await
        .map_err(|e| e.to_string())?;

    Ok(SyncResultDto {
        conversations: result.conversations,
        terms_added: result.terms_added,
        batches_total: result.batches_total,
        batches_failed: result.batches_failed,
    })
}

/// Read the persisted record of the last vocabulary sync.
#[tauri::command]
fn get_sync_state() -> Result<SyncStateDto, String> {
    let state = pie_engine::corrector::sync::SyncState::load(
        &pie_engine::corrector::sync::default_sync_state_path(),
    );
    Ok(SyncStateDto {
        last_run_unix: state.last_run_unix,
        terms_added: state.terms_added,
    })
}

/* ─── helpers ─── */

/// Map an empty string to `None` so blank optimizer fields aren't stored.
fn opt(s: &str) -> Option<String> {
    (!s.is_empty()).then(|| s.to_string())
}

/// Map an empty `llm_model` setting to `None` so the router falls back to the
/// provider's default model.
fn model_opt(s: &Settings) -> Option<&str> {
    (!s.llm_model.is_empty()).then_some(s.llm_model.as_str())
}

/// Build the engine's LLM connection config from settings (BYOK).
fn llm_config(s: &Settings) -> pie_engine::llm::LlmConfig {
    pie_engine::llm::LlmConfig {
        api_url: s.llm_api_url.clone(),
        api_key: s.llm_api_key.clone(),
        model: s.llm_model.clone(),
    }
}

/// Load (or reuse) the whisper engine for the configured model + language.
fn get_or_load_whisper(
    state: &State<'_, AppState>,
    settings: &Settings,
) -> Result<Arc<WhisperEngine>, String> {
    if settings.whisper_model.is_empty() {
        return Err("No whisper model configured. Set one in Settings (e.g. \
             ~/.cache/pie/models/ggml-tiny.en.bin)."
            .to_string());
    }
    let path = Settings::expand(&settings.whisper_model);

    let mut cache = state.whisper.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((cached_path, cached_lang, engine)) = cache.as_ref() {
        if *cached_path == path && *cached_lang == settings.language {
            return Ok(Arc::clone(engine));
        }
    }

    let engine =
        Arc::new(WhisperEngine::load(&path, &settings.language).map_err(|e| e.to_string())?);
    *cache = Some((path, settings.language.clone(), Arc::clone(&engine)));
    Ok(engine)
}

/// Load the configured whisper model into the cache on a background thread so
/// the first transcription of a session isn't cold. Non-blocking, silent, and
/// best-effort: a no-op when no model is configured, and a logged warning on
/// failure (the next real transcription falls back to the lazy load).
fn warm_whisper(app: &AppHandle) {
    let app = app.clone();
    std::thread::spawn(move || {
        let state = app.state::<AppState>();
        let settings = state
            .settings
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if settings.whisper_model.is_empty() {
            return;
        }
        match get_or_load_whisper(&state, &settings) {
            Ok(_) => log::info!("Whisper model warmed"),
            Err(e) => log::warn!("Whisper warm-up failed (will load on first use): {e}"),
        }
    });
}

/// Recorder with Silero VAD when configured; VAD-free otherwise. The Silero
/// session is loaded once and reused across recordings via `vad_cache`.
/// Attach the capture-level feed the overlay and record bar draw with.
///
/// The recorder hands us raw device-rate mono chunks on its capture thread, so
/// the callback stays cheap: one RMS pass, a perceptual curve, and an emit at
/// most every 50 ms. The UI degrades to a timed sweep if these never arrive,
/// so dropping frames here is harmless.
fn with_level_feed(recorder: AudioRecorder, app: &AppHandle) -> AudioRecorder {
    let app = app.clone();
    let last = Mutex::new(std::time::Instant::now() - LEVEL_INTERVAL);
    recorder.with_level_callback(move |chunk| {
        if chunk.is_empty() {
            return;
        }
        {
            let mut last = last.lock().unwrap_or_else(|e| e.into_inner());
            if last.elapsed() < LEVEL_INTERVAL {
                return;
            }
            *last = std::time::Instant::now();
        }
        let sum: f32 = chunk.iter().map(|s| s * s).sum();
        let rms = (sum / chunk.len() as f32).sqrt();
        // Speech RMS lands around 0.01–0.15; the square root opens up the
        // quiet end so normal talking moves the rule visibly.
        let level = (rms.sqrt() * 3.0).clamp(0.0, 1.0);
        emit_event(&app, "pie://level", level);
    })
}

/// Minimum gap between `pie://level` emissions.
const LEVEL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);

fn build_recorder(
    app: &AppHandle,
    settings: &Settings,
    vad_cache: &mut VadCache,
) -> anyhow::Result<(AudioRecorder, bool)> {
    if settings.silero_model.is_empty() {
        return Ok((with_level_feed(AudioRecorder::new()?, app), false));
    }
    let model_path = Settings::expand(&settings.silero_model);
    let detector = vad_cache.get_or_build(&model_path, || {
        let silero = SileroVad::new(&model_path, PIE_VAD_THRESHOLD)?;
        let smoothed = VadPipeline::new(
            Box::new(silero),
            VAD_CONTEXT_FRAMES,
            VAD_HANGOVER_FRAMES,
            VAD_SPEECH_THRESHOLD_FRAMES,
        );
        Ok(Box::new(smoothed) as Box<dyn VoiceActivityDetector>)
    })?;
    let recorder = AudioRecorder::new()?.with_vad_shared(
        detector,
        VAD_HANGOVER_FRAMES,
        VAD_STREAM_HANGOVER_FRAMES,
    );
    Ok((with_level_feed(recorder, app), true))
}

fn main() {
    env_logger::init();

    #[allow(unused_mut)]
    let mut builder = tauri::Builder::default()
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build());
    // macOS overlay is an NSPanel created directly in overlay.rs (see the
    // vendored nspanel module) — no external plugin needed.

    builder
        .setup(|app| {
            let settings = Settings::load();
            let mut engine =
                tauri::async_runtime::block_on(PieEngine::with_config(&llm_config(&settings)))?;
            engine.set_code_mode(settings.code_mode);
            let default_paste_mode = settings.paste_output.clone();
            let settings_for_hotkeys = settings.clone();
            let history_path = dirs::config_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("pie")
                .join("history.db");
            let history = HistoryStore::open(&history_path).unwrap_or_else(|e| {
                log::error!("Failed to open history DB ({e}); using in-memory");
                HistoryStore::open_in_memory().expect("in-memory history must open")
            });
            // Capture what the background learner needs before `settings` is
            // moved into `AppState` below.
            let mining = settings.background_mining;
            let learner_cfg = llm_config(&settings);
            let learner_provider = settings.provider.clone();
            let learner_model =
                (!settings.llm_model.is_empty()).then(|| settings.llm_model.clone());
            app.manage(AppState {
                settings: Mutex::new(settings),
                recorder: Mutex::new(None),
                vad_cache: Mutex::new(VadCache::new()),
                busy: AtomicBool::new(false),
                whisper: Mutex::new(None),
                engine: tokio::sync::Mutex::new(engine),
                history: Mutex::new(history),
                pending_paste_mode: Mutex::new(default_paste_mode),
            });
            app.manage(EnigoState::new());

            // Opt-in background learner: mines new pronunciation corrections
            // from transcripts via the configured LLM. OFF by default, so no
            // background LLM calls happen unless the user enables it.
            if mining {
                // Pipeline -> learner: transcripts to mine.
                let (tx, rx) =
                    tokio::sync::mpsc::channel::<pie_engine::pipeline::engine::LearnTask>(100);
                // Learner -> engine: mined corrections the engine applies (the
                // engine is the sole writer of learned_vocab.json — no race).
                let (mined_tx, mined_rx) = tokio::sync::mpsc::channel::<
                    pie_engine::corrector::learner::ExtractedCorrection,
                >(100);
                {
                    let state = app.state::<AppState>();
                    tauri::async_runtime::block_on(async {
                        let mut engine = state.engine.lock().await;
                        engine.set_learner_tx(tx); // process() fires transcripts
                        engine.set_mined_rx(mined_rx); // process() drains mined terms
                    });
                }
                let llm = pie_engine::llm::LlmRouter::from_config(&learner_cfg);
                let learner = pie_engine::corrector::learner::BackgroundLearner::new(
                    rx,
                    llm,
                    mined_tx,
                    learner_provider,
                    learner_model,
                    std::collections::HashSet::new(),
                );
                tauri::async_runtime::spawn(async move { learner.run().await });
            }

            if let Err(e) = register_hotkeys(app.handle(), &settings_for_hotkeys) {
                // A bad hotkey must not prevent the app from starting.
                log::error!("{e}");
            }

            // Floating indicator, created hidden and shown per recording state.
            // Creating the NSPanel overlay temporarily flips the app activation
            // policy to Prohibited (tauri-nspanel's no_activate), which orders
            // the main window out; restoring the policy doesn't bring it back,
            // so re-show the main window explicitly afterwards.
            overlay::create_overlay(app.handle());
            if let Some(main) = app.get_webview_window("main") {
                let _ = main.show();
                let _ = main.set_focus();
            }

            // System tray: PIE runs in the background so the hotkey works from
            // any app; the tray is how you reopen or quit it.
            let show_item = MenuItemBuilder::with_id("show", "Show PIE").build(app)?;
            let quit_item = MenuItemBuilder::with_id("quit", "Quit PIE").build(app)?;
            let menu = MenuBuilder::new(app)
                .items(&[&show_item, &quit_item])
                .build()?;
            // The menu bar wants a monochrome template image that macOS tints
            // for the current theme — the full-colour app icon shrunk to 18pt
            // reads as a muddy sticker and ignores light/dark. Other platforms
            // keep the colour icon, since a black-only glyph would disappear
            // against a dark Windows taskbar.
            #[cfg(target_os = "macos")]
            let tray_icon = tauri::include_image!("icons/tray-template.png");
            #[cfg(not(target_os = "macos"))]
            let tray_icon = tauri::include_image!("icons/32x32.png");

            TrayIconBuilder::with_id("main-tray")
                .icon(tray_icon)
                .icon_as_template(cfg!(target_os = "macos"))
                .tooltip("PIE — Personal Intent Engine")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "show" => show_main_window(app),
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        show_main_window(tray.app_handle());
                    }
                })
                .build(app)?;

            // Warm the whisper model off the launch path so the first
            // transcription isn't cold. Non-blocking.
            warm_whisper(app.handle());

            Ok(())
        })
        .on_window_event(|window, event| {
            // Closing the main window hides it to the tray instead of quitting,
            // so the global hotkey keeps working. Quit is via the tray menu.
            if window.label() == "main" {
                if let WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            get_settings,
            update_settings,
            set_hotkey_active,
            start_recording,
            stop_recording,
            cancel_recording,
            send_to_llm,
            test_llm_connection,
            copy_to_clipboard,
            list_history,
            delete_history_entry,
            clear_history,
            paste_history_entry,
            list_models,
            select_model,
            delete_model,
            download_model,
            list_corrections,
            add_correction,
            delete_correction,
            recorrect_with_ai,
            get_learned_vocab_count,
            reset_learned_vocab,
            pick_import_target,
            discover_sync_sources,
            run_vocabulary_sync,
            get_sync_state,
        ])
        .run(tauri::generate_context!())
        .expect("error while running PIE");
}

#[cfg(test)]
mod tests {
    use tauri_plugin_global_shortcut::Shortcut;

    /// The recorder builds accelerators from modifier names + `event.code`.
    /// Every shape it can produce must parse, or a captured hotkey would fail
    /// to register.
    #[test]
    fn recorder_accelerators_parse() {
        let cases = [
            "Command+Shift+Space",
            "Control+Alt+KeyK",
            "Command+KeyL",
            "Command+Digit1",
            "Shift+ArrowUp",
            "F5",
            "CmdOrCtrl+Shift+Space", // the default
        ];
        for a in cases {
            assert!(a.parse::<Shortcut>().is_ok(), "should parse: {a}");
        }
    }
}
