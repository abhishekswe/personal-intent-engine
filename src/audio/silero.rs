use anyhow::Result;
use std::path::Path;

use super::silero_vad_engine::SileroVadEngine;
use super::vad::{VadFrame, VoiceActivityDetector};
use super::{FRAME_SAMPLES, STT_SAMPLE_RATE};

/// Default speech probability threshold, empirically tuned for speech detection.
pub const PIE_VAD_THRESHOLD: f32 = 0.3;

/// Silero VAD (ONNX) — a small recurrent model that classifies 30 ms frames
/// as speech vs non-speech far more robustly than energy thresholding.
///
/// Wrap in [`super::VadPipeline`] for onset/hangover/prefill smoothing; this
/// type only scores individual frames.
pub struct SileroVad {
    engine: SileroVadEngine,
    threshold: f32,
}

impl SileroVad {
    /// Load the Silero ONNX model. `threshold` is the speech probability
    /// cutoff in `0.0..=1.0` (0.3 is a good default).
    pub fn new<P: AsRef<Path>>(model_path: P, threshold: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&threshold) {
            anyhow::bail!("threshold must be between 0.0 and 1.0");
        }

        Ok(Self {
            engine: SileroVadEngine::new(&model_path, STT_SAMPLE_RATE)?,
            threshold,
        })
    }
}

impl VoiceActivityDetector for SileroVad {
    fn push_frame<'a>(&'a mut self, frame: &'a [f32]) -> Result<VadFrame<'a>> {
        if self.is_voice(frame)? {
            Ok(VadFrame::Speech(frame))
        } else {
            Ok(VadFrame::Noise)
        }
    }

    fn is_voice(&mut self, frame: &[f32]) -> Result<bool> {
        if frame.len() != FRAME_SAMPLES {
            anyhow::bail!("expected {FRAME_SAMPLES} samples, got {}", frame.len());
        }

        let prob = self.engine.compute(frame)?;

        Ok(prob > self.threshold)
    }

    fn reset(&mut self) {
        // Clear the Silero LSTM hidden/cell state so a new session doesn't
        // inherit recurrent context from the previous recording.
        self.engine.reset();
    }

    fn set_hangover_frames(&mut self, _frames: usize) {}
}
