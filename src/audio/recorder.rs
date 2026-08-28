use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Sample, SizedSample};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use super::vad::{VadFrame, VadPolicy, VoiceActivityDetector};
use super::{AudioResampler, FRAME_DURATION_MS, FRAME_SAMPLES, STT_SAMPLE_RATE};

/// Commands for the audio worker thread
enum Cmd {
    /// Begin capturing. Carries the send timestamp so the consumer can log how
    /// long the command sat in the channel.
    Start(VadPolicy, Instant),
    Stop(mpsc::Sender<Vec<f32>>),
    Shutdown,
}

/// Audio chunk from the cpal callback. `EndOfStream` is the sentinel the
/// callback sends once after the stop flag is raised, guaranteeing the
/// consumer can drain every captured sample before finalizing a recording.
enum AudioChunk {
    Samples(Vec<f32>),
    EndOfStream,
}

/// A single VAD engine plus the two hangover-tail lengths its smoothing
/// wrapper should use. The offline and streaming policies are never active
/// concurrently, so one detector is reconfigured per session (see `Cmd::Start`)
/// rather than kept as two resident engines.
#[derive(Clone)]
struct VadConfig {
    detector: Arc<Mutex<Box<dyn VoiceActivityDetector>>>,
    offline_hangover_frames: usize,
    streaming_hangover_frames: usize,
}

impl VadConfig {
    /// Post-speech hangover tail (in 30 ms frames) for the given policy.
    /// `Disabled` never reaches the detector, so it maps to the offline value.
    fn hangover_for(&self, policy: VadPolicy) -> usize {
        match policy {
            VadPolicy::Streaming => self.streaming_hangover_frames,
            VadPolicy::Offline | VadPolicy::Disabled => self.offline_hangover_frames,
        }
    }
}

/// Callback invoked with each 16 kHz mono frame that passes the active capture
/// policy while recording. Used to feed live streaming transcription.
pub type AudioFrameCallback = Arc<dyn Fn(&[f32]) + Send + Sync + 'static>;

/// Cross-platform audio recorder using cpal.
///
/// Captures audio from an input device, downmixes to mono in the stream
/// callback, resamples to 16 kHz, applies VAD filtering, and provides both
/// buffered and streaming output.
///
/// Audio recorder: opens device, streams frames through VAD, collects samples.
/// - Dedicated worker thread; channel-based communication
/// - Stream plays continuously; a `recording` flag gates capture so no
///   audio is lost around start/stop transitions
/// - On stop, the consumer drains the channel until `EndOfStream` and then
///   flushes the resampler, so the tail of the recording is preserved
pub struct AudioRecorder {
    device: Option<Device>,
    cmd_tx: Option<mpsc::Sender<Cmd>>,
    worker_handle: Option<std::thread::JoinHandle<()>>,
    vad: Option<VadConfig>,
    level_cb: Option<Arc<dyn Fn(Vec<f32>) + Send + Sync + 'static>>,
    audio_cb: Option<AudioFrameCallback>,
    /// Preferred stream config cached per device name. The HAL property
    /// queries in `get_preferred_config` cost ~40-85ms per open (worse on
    /// USB/Bluetooth), which lands on the keypress->capture path. Keyed by
    /// name so a system-default change misses naturally; cleared whenever an
    /// open fails so a stale rate/format self-heals on the caller's retry.
    config_cache: Arc<Mutex<Option<(String, cpal::SupportedStreamConfig)>>>,
}

impl AudioRecorder {
    /// Create an idle recorder. No device is opened until [`AudioRecorder::open`].
    pub fn new() -> anyhow::Result<Self> {
        Ok(AudioRecorder {
            device: None,
            cmd_tx: None,
            worker_handle: None,
            vad: None,
            level_cb: None,
            audio_cb: None,
            config_cache: Arc::new(Mutex::new(None)),
        })
    }

    /// Attach a single VAD engine, reconfigured per session for the offline vs
    /// streaming hangover tail. The two policies are mutually exclusive within
    /// a recording, so one engine covers both instead of two resident instances.
    #[must_use]
    pub fn with_vad(
        self,
        detector: Box<dyn VoiceActivityDetector>,
        offline_hangover_frames: usize,
        streaming_hangover_frames: usize,
    ) -> Self {
        self.with_vad_shared(
            Arc::new(Mutex::new(detector)),
            offline_hangover_frames,
            streaming_hangover_frames,
        )
    }

    /// Attach a VAD engine via a *shared* handle so an expensive-to-load
    /// detector (e.g. the Silero ONNX session) can be reused across recordings
    /// instead of rebuilt each time. The recorder resets the detector's state
    /// at each session start, so the caller may keep its own clone of `detector`
    /// alive between recordings.
    #[must_use]
    pub fn with_vad_shared(
        mut self,
        detector: Arc<Mutex<Box<dyn VoiceActivityDetector>>>,
        offline_hangover_frames: usize,
        streaming_hangover_frames: usize,
    ) -> Self {
        self.vad = Some(VadConfig {
            detector,
            offline_hangover_frames,
            streaming_hangover_frames,
        });
        self
    }

    /// Register a callback for audio level visualization. Receives raw
    /// device-rate mono chunks; spectrum bucketing is a later phase.
    #[must_use]
    pub fn with_level_callback<F: Fn(Vec<f32>) + Send + Sync + 'static>(mut self, cb: F) -> Self {
        self.level_cb = Some(Arc::new(cb));
        self
    }

    /// Register a callback that receives real-time 16 kHz frames after the
    /// active VAD policy has been applied. Frames arrive in order on the
    /// recorder's consumer thread — keep the callback cheap (e.g. forward to
    /// a channel) so it never stalls capture.
    #[must_use]
    pub fn with_audio_callback<F: Fn(&[f32]) + Send + Sync + 'static>(mut self, cb: F) -> Self {
        self.audio_cb = Some(Arc::new(cb));
        self
    }

    /// Open the audio device and start the worker thread.
    pub fn open(&mut self, device: Option<Device>) -> anyhow::Result<()> {
        if self.worker_handle.is_some() {
            return Ok(()); // already open
        }

        let (sample_tx, sample_rx) = mpsc::channel::<AudioChunk>();
        let (cmd_tx, cmd_rx) = mpsc::channel::<Cmd>();
        let (init_tx, init_rx) = mpsc::sync_channel::<Result<(), String>>(1);

        let host = cpal::default_host();
        let device = match device {
            Some(dev) => dev,
            None => host
                .default_input_device()
                .ok_or_else(|| anyhow::anyhow!("No input device found"))?,
        };

        let thread_device = device.clone();
        let vad = self.vad.clone();
        let level_cb = self.level_cb.clone();
        let audio_cb = self.audio_cb.clone();
        let config_cache = Arc::clone(&self.config_cache);

        let worker = std::thread::spawn(move || {
            let stop_flag = Arc::new(AtomicBool::new(false));
            let stop_flag_for_stream = stop_flag.clone();

            let init_result = (|| -> Result<(cpal::Stream, u32), String> {
                let device_name = thread_device.name().unwrap_or_default();
                let cached_config = config_cache
                    .lock()
                    .expect("config cache poisoned")
                    .as_ref()
                    .filter(|(name, _)| !device_name.is_empty() && *name == device_name)
                    .map(|(_, cfg)| cfg.clone());
                let config_was_cached = cached_config.is_some();
                let config = match cached_config {
                    Some(cfg) => cfg,
                    None => Self::get_preferred_config(&thread_device)
                        .map_err(|e| format!("Failed to fetch preferred config: {e}"))?,
                };

                let sample_rate = config.sample_rate().0;
                let channels = config.channels() as usize;

                log::info!(
                    "Using device: {:?}, rate: {}, channels: {}, format: {:?}",
                    thread_device.name(),
                    sample_rate,
                    channels,
                    config.sample_format()
                );

                let stream = match config.sample_format() {
                    cpal::SampleFormat::U8 => Self::build_stream::<u8>(
                        &thread_device,
                        &config,
                        sample_tx,
                        channels,
                        stop_flag_for_stream,
                    ),
                    cpal::SampleFormat::I8 => Self::build_stream::<i8>(
                        &thread_device,
                        &config,
                        sample_tx,
                        channels,
                        stop_flag_for_stream,
                    ),
                    cpal::SampleFormat::I16 => Self::build_stream::<i16>(
                        &thread_device,
                        &config,
                        sample_tx,
                        channels,
                        stop_flag_for_stream,
                    ),
                    cpal::SampleFormat::I32 => Self::build_stream::<i32>(
                        &thread_device,
                        &config,
                        sample_tx,
                        channels,
                        stop_flag_for_stream,
                    ),
                    cpal::SampleFormat::F32 => Self::build_stream::<f32>(
                        &thread_device,
                        &config,
                        sample_tx,
                        channels,
                        stop_flag_for_stream,
                    ),
                    fmt => return Err(format!("Unsupported sample format: {fmt:?}")),
                }
                .map_err(|e| format!("Failed to build input stream: {e}"))?;

                stream
                    .play()
                    .map_err(|e| format!("Failed to start microphone stream: {e}"))?;

                // The device accepted this config; remember it so the next
                // open skips the HAL property queries entirely.
                if !config_was_cached && !device_name.is_empty() {
                    *config_cache.lock().expect("config cache poisoned") =
                        Some((device_name, config));
                }

                Ok((stream, sample_rate))
            })();

            match init_result {
                Ok((stream, sample_rate)) => {
                    let _ = init_tx.send(Ok(()));
                    run_consumer(
                        sample_rate,
                        vad,
                        sample_rx,
                        cmd_rx,
                        level_cb,
                        audio_cb,
                        stop_flag,
                    );
                    drop(stream);
                }
                Err(error_message) => {
                    // A failed open may mean the cached config went stale
                    // (device re-plugged, rate/format changed in the OS).
                    // Drop it so the next attempt re-queries the device.
                    *config_cache.lock().expect("config cache poisoned") = None;
                    log::error!("{error_message}");
                    let _ = init_tx.send(Err(error_message));
                }
            }
        });

        match init_rx.recv() {
            Ok(Ok(())) => {
                self.device = Some(device);
                self.cmd_tx = Some(cmd_tx);
                self.worker_handle = Some(worker);
                Ok(())
            }
            Ok(Err(e)) => {
                let _ = worker.join();
                anyhow::bail!("Audio init failed: {e}")
            }
            Err(e) => {
                let _ = worker.join();
                anyhow::bail!("Audio init channel error: {e}")
            }
        }
    }

    /// Start recording with the given VAD policy.
    pub fn start(&self, policy: VadPolicy) -> anyhow::Result<()> {
        if let Some(tx) = &self.cmd_tx {
            tx.send(Cmd::Start(policy, Instant::now()))
                .map_err(|_| anyhow::anyhow!("Audio worker disconnected"))?;
        }
        Ok(())
    }

    /// Stop recording and return the accumulated 16 kHz mono samples.
    pub fn stop(&self) -> anyhow::Result<Vec<f32>> {
        let (tx, rx) = mpsc::channel();
        if let Some(cmd_tx) = &self.cmd_tx {
            cmd_tx
                .send(Cmd::Stop(tx))
                .map_err(|_| anyhow::anyhow!("Audio worker disconnected"))?;
        }
        rx.recv()
            .map_err(|_| anyhow::anyhow!("Audio worker dropped result"))
    }

    /// Shut down the worker thread and release the device.
    pub fn close(&mut self) -> anyhow::Result<()> {
        if let Some(tx) = self.cmd_tx.take() {
            let _ = tx.send(Cmd::Shutdown);
        }
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
        self.device = None;
        Ok(())
    }

    /// Build a cpal input stream. Downmixes to mono inside the callback and
    /// sends `EndOfStream` exactly once after the stop flag is raised.
    fn build_stream<T>(
        device: &Device,
        config: &cpal::SupportedStreamConfig,
        tx: mpsc::Sender<AudioChunk>,
        channels: usize,
        stop_flag: Arc<AtomicBool>,
    ) -> Result<cpal::Stream, cpal::BuildStreamError>
    where
        T: Sample + SizedSample + Send + 'static,
        f32: cpal::FromSample<T>,
    {
        // Reused across callbacks: `mem::take` below hands the filled buffer to
        // the consumer without copying, then `reserve` re-grows the emptied
        // buffer in a single allocation on the next callback.
        let mut output_buffer: Vec<f32> = Vec::with_capacity(FRAME_SAMPLES * 2);
        let mut eos_sent = false;

        let stream_cb = move |data: &[T], _: &cpal::InputCallbackInfo| {
            if stop_flag.load(Ordering::Relaxed) {
                if !eos_sent {
                    let _ = tx.send(AudioChunk::EndOfStream);
                    eos_sent = true;
                }
                return;
            }
            eos_sent = false;

            let frame_count = if channels == 1 {
                data.len()
            } else {
                data.len() / channels
            };
            output_buffer.reserve(frame_count);

            if channels == 1 {
                output_buffer.extend(data.iter().map(|&s| s.to_sample::<f32>()));
            } else {
                for frame in data.chunks_exact(channels) {
                    let mono =
                        frame.iter().map(|&s| s.to_sample::<f32>()).sum::<f32>() / channels as f32;
                    output_buffer.push(mono);
                }
            }

            // Move the buffer into the message (no copy); the emptied
            // `output_buffer` is refilled from scratch on the next callback.
            if tx
                .send(AudioChunk::Samples(std::mem::take(&mut output_buffer)))
                .is_err()
            {
                log::error!("Failed to send samples");
            }
        };

        device.build_input_stream(
            &config.clone().into(),
            stream_cb,
            |err| log::error!("Audio stream error: {err}"),
            None,
        )
    }

    /// Pick the best stream config at the device's native rate. Forcing the
    /// hardware to 16 kHz can fail on Bluetooth codecs and some ALSA drivers,
    /// so we capture at the native rate and let `AudioResampler` downsample.
    fn get_preferred_config(device: &Device) -> anyhow::Result<cpal::SupportedStreamConfig> {
        let default_config = device
            .default_input_config()
            .map_err(|e| anyhow::anyhow!("No input config: {e}"))?;
        let target_rate = default_config.sample_rate();

        let supported_configs = match device.supported_input_configs() {
            Ok(configs) => configs,
            Err(e) => {
                log::warn!("Could not enumerate input configs ({e}), using device default");
                return Ok(default_config);
            }
        };

        let score = |fmt: cpal::SampleFormat| match fmt {
            cpal::SampleFormat::F32 => 4,
            cpal::SampleFormat::I16 => 3,
            cpal::SampleFormat::I32 => 2,
            _ => 1,
        };

        let best_config = supported_configs
            .filter(|range| {
                range.min_sample_rate() <= target_rate && range.max_sample_rate() >= target_rate
            })
            .max_by_key(|range| score(range.sample_format()));

        match best_config {
            Some(config) => Ok(config.with_sample_rate(target_rate)),
            None => {
                log::warn!(
                    "No supported config matched device default rate {target_rate:?}, using default"
                );
                Ok(default_config)
            }
        }
    }

    /// List available input devices
    pub fn list_devices() -> anyhow::Result<Vec<Device>> {
        let host = cpal::default_host();
        let devices: Vec<Device> = host
            .input_devices()
            .map_err(|e| anyhow::anyhow!("Device enumeration failed: {e}"))?
            .collect();
        Ok(devices)
    }
}

impl Drop for AudioRecorder {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Consumer loop: receives mono device-rate chunks, resamples to 16 kHz
/// frames, applies the active VAD policy, and accumulates recording output.
fn run_consumer(
    in_sample_rate: u32,
    vad: Option<VadConfig>,
    sample_rx: mpsc::Receiver<AudioChunk>,
    cmd_rx: mpsc::Receiver<Cmd>,
    level_cb: Option<Arc<dyn Fn(Vec<f32>) + Send + Sync + 'static>>,
    audio_cb: Option<AudioFrameCallback>,
    stop_flag: Arc<AtomicBool>,
) {
    let mut resampler = AudioResampler::new(
        in_sample_rate as usize,
        STT_SAMPLE_RATE,
        Duration::from_millis(FRAME_DURATION_MS as u64),
    );

    // Pre-size for a few seconds of speech so a typical recording accumulates
    // without repeated reallocation; it still grows for longer sessions.
    let mut processed_samples = Vec::<f32>::with_capacity(STT_SAMPLE_RATE * 4);
    let mut recording = false;
    let mut vad_policy = VadPolicy::Offline;

    fn handle_frame(
        samples: &[f32],
        recording: bool,
        vad_policy: VadPolicy,
        vad: &Option<VadConfig>,
        audio_cb: &Option<AudioFrameCallback>,
        out_buf: &mut Vec<f32>,
    ) {
        if !recording {
            return;
        }

        let mut emit = |buf: &[f32]| {
            out_buf.extend_from_slice(buf);
            if let Some(cb) = audio_cb {
                cb(buf);
            }
        };

        if vad_policy == VadPolicy::Disabled {
            emit(samples);
            return;
        }

        if let Some(cfg) = vad {
            let mut det = cfg.detector.lock().expect("VAD detector poisoned");
            // On VAD error, fail open: passing speech through beats dropping it.
            match det.push_frame(samples).unwrap_or(VadFrame::Speech(samples)) {
                VadFrame::Speech(buf) => emit(buf),
                VadFrame::Noise => {}
            }
        } else {
            emit(samples);
        }
    }

    // Runs until the stream closes and `recv` returns `Err`.
    while let Ok(chunk) = sample_rx.recv() {
        // Handle pending commands BEFORE the in-flight chunk so a Start
        // captures it. Polling commands after processing would silently drop
        // one buffer period of audio (~10ms built-in, up to ~100ms on
        // Bluetooth) at every recording start.
        let mut pending = Some(chunk);
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                Cmd::Start(policy, sent_at) => {
                    log::debug!(
                        "Cmd::Start processed {:?} after send; capture begins with the in-flight chunk",
                        sent_at.elapsed()
                    );
                    stop_flag.store(false, Ordering::Relaxed);
                    vad_policy = policy;
                    processed_samples.clear();
                    recording = true;
                    resampler.reset();
                    // Reconfigure the single VAD engine for this session's
                    // policy and clear its smoothing + recurrent state before
                    // it sees any frames.
                    if vad_policy != VadPolicy::Disabled {
                        if let Some(cfg) = &vad {
                            let mut det = cfg.detector.lock().expect("VAD detector poisoned");
                            det.set_hangover_frames(cfg.hangover_for(vad_policy));
                            det.reset();
                        }
                    }
                }
                Cmd::Stop(reply_tx) => {
                    recording = false;
                    stop_flag.store(true, Ordering::Relaxed);

                    // The chunk in hand arrived before the stop; it belongs to
                    // the recording, so feed it ahead of the drain below.
                    if let Some(AudioChunk::Samples(raw)) = pending.take() {
                        resampler.push(&raw, &mut |frame: &[f32]| {
                            handle_frame(
                                frame,
                                true,
                                vad_policy,
                                &vad,
                                &audio_cb,
                                &mut processed_samples,
                            )
                        });
                    }

                    // Drain all remaining audio until the producer confirms
                    // end-of-stream. The cpal callback sees the stop flag,
                    // sends EndOfStream, and goes silent — guaranteeing every
                    // captured sample is in the channel ahead of the sentinel.
                    loop {
                        match sample_rx.recv_timeout(Duration::from_secs(2)) {
                            Ok(AudioChunk::Samples(remaining)) => {
                                resampler.push(&remaining, &mut |frame: &[f32]| {
                                    handle_frame(
                                        frame,
                                        true,
                                        vad_policy,
                                        &vad,
                                        &audio_cb,
                                        &mut processed_samples,
                                    )
                                });
                            }
                            Ok(AudioChunk::EndOfStream) => break,
                            Err(_) => {
                                log::warn!("Timed out waiting for EndOfStream from audio callback");
                                break;
                            }
                        }
                    }

                    // Flush the resampler so the recording keeps its tail.
                    resampler.finish(&mut |frame: &[f32]| {
                        handle_frame(
                            frame,
                            true,
                            vad_policy,
                            &vad,
                            &audio_cb,
                            &mut processed_samples,
                        )
                    });

                    let _ = reply_tx.send(std::mem::take(&mut processed_samples));

                    // Resume the audio callback so the consumer loop keeps
                    // receiving chunks (always-on microphone mode).
                    stop_flag.store(false, Ordering::Relaxed);
                }
                Cmd::Shutdown => {
                    stop_flag.store(true, Ordering::Relaxed);
                    return;
                }
            }
        }

        let raw = match pending.take() {
            Some(AudioChunk::Samples(s)) => s,
            // EndOfStream, or the chunk was consumed by a Stop above.
            _ => continue,
        };

        if let Some(cb) = &level_cb {
            cb(raw.clone());
        }

        resampler.push(&raw, &mut |frame: &[f32]| {
            handle_frame(
                frame,
                recording,
                vad_policy,
                &vad,
                &audio_cb,
                &mut processed_samples,
            )
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::vad::PassthroughVad;

    #[test]
    fn with_vad_shared_reuses_the_passed_handle() {
        let detector: Arc<Mutex<Box<dyn VoiceActivityDetector>>> =
            Arc::new(Mutex::new(Box::new(PassthroughVad)));
        let recorder = AudioRecorder::new()
            .unwrap()
            .with_vad_shared(Arc::clone(&detector), 5, 10);
        assert!(recorder.vad.is_some(), "recorder should hold a VAD config");
        // Caller holds one ref; the recorder's VadConfig holds the second.
        assert_eq!(
            Arc::strong_count(&detector),
            2,
            "handle must be shared, not cloned into a new Arc"
        );
    }

    #[test]
    fn with_vad_wraps_detector_in_a_fresh_handle() {
        let recorder = AudioRecorder::new()
            .unwrap()
            .with_vad(Box::new(PassthroughVad), 5, 10);
        assert!(
            recorder.vad.is_some(),
            "with_vad should still populate the VAD config"
        );
    }
}
