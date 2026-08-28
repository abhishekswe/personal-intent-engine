pub mod recorder;
pub mod resampler;
#[cfg(feature = "vad")]
pub mod silero;
#[cfg(feature = "vad")]
pub mod silero_vad_engine;
pub mod traits;
pub mod vad;
pub mod vad_cache;

/// Audio capture configuration
pub const STT_SAMPLE_RATE: usize = 16000;
pub const FRAME_DURATION_MS: usize = 30;
pub const FRAME_SAMPLES: usize = STT_SAMPLE_RATE * FRAME_DURATION_MS / 1000; // 480

/// Re-exports
pub use recorder::{AudioFrameCallback, AudioRecorder};
pub use resampler::AudioResampler;
#[cfg(feature = "vad")]
pub use silero::{SileroVad, PIE_VAD_THRESHOLD};
pub use traits::AudioCapture;
pub use vad::{
    EnergyVad, VadFrame, VadPipeline, VadPolicy, VoiceActivityDetector, VAD_CONTEXT_FRAMES,
    VAD_HANGOVER_FRAMES, VAD_SPEECH_THRESHOLD_FRAMES, VAD_STREAM_HANGOVER_FRAMES,
};
pub use vad_cache::{SharedVad, VadCache};
