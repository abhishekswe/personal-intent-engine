#[cfg(feature = "vad")]
pub mod moonshine;
pub mod stream;
#[cfg(feature = "whisper")]
pub mod whisper;

#[cfg(feature = "vad")]
pub use moonshine::MoonshineModel;
pub use stream::{StreamCmd, TranscriptRouter};
#[cfg(feature = "whisper")]
pub use whisper::WhisperEngine;

/// Trait for speech-to-text engines
pub trait SttEngine: Send + Sync {
    /// Transcribe audio samples (16kHz mono f32) to text
    fn transcribe(&self, samples: &[f32]) -> anyhow::Result<String>;

    /// Check if the engine is ready (model loaded)
    fn is_ready(&self) -> bool;
}

/// Read a WAV file and convert it to 16 kHz mono f32, ready for STT.
/// Handles int/float encodings, multi-channel downmix, and resampling.
#[cfg(feature = "whisper")]
pub fn load_wav_as_16k_mono(path: &std::path::Path) -> anyhow::Result<Vec<f32>> {
    use crate::audio::{AudioResampler, FRAME_DURATION_MS, WHISPER_SAMPLE_RATE};
    use std::time::Duration;

    let mut reader = hound::WavReader::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to open WAV '{}': {e}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|e| anyhow::anyhow!("Failed to read WAV samples: {e}"))?,
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<Result<_, _>>()
                .map_err(|e| anyhow::anyhow!("Failed to read WAV samples: {e}"))?
        }
    };

    let mono: Vec<f32> = if channels == 1 {
        samples
    } else {
        samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    if spec.sample_rate as usize == WHISPER_SAMPLE_RATE {
        return Ok(mono);
    }

    let mut resampler = AudioResampler::new(
        spec.sample_rate as usize,
        WHISPER_SAMPLE_RATE,
        Duration::from_millis(FRAME_DURATION_MS as u64),
    );
    let mut out = Vec::with_capacity(mono.len() * WHISPER_SAMPLE_RATE / spec.sample_rate as usize);
    resampler.push(&mono, &mut |frame| out.extend_from_slice(frame));
    resampler.finish(&mut |frame| out.extend_from_slice(frame));
    Ok(out)
}
