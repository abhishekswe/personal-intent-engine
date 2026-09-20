//! Moonshine ONNX speech-to-text: raw-audio encoder + autoregressive KV-cache
//! decoder, run natively via `ort`. No mel-spectrogram.

use std::borrow::Cow;
use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::{builder::GraphOptimizationLevel, Session, SessionInputValue};
use ort::value::Value;
use tokenizers::Tokenizer;

use crate::stt::SttEngine;

/// Decoder start-of-sequence token id.
const DECODER_START: i64 = 1;
/// End-of-sequence token id.
const EOS: i64 = 2;

/// Collapse a borrowed `ort::Error<R>` into a plain Send/Sync `ort::Error`.
fn plain(e: impl Into<ort::Error>) -> ort::Error {
    e.into()
}

/// A loaded Moonshine model (encoder + decoder ONNX sessions), ready to
/// transcribe 16 kHz mono audio into token ids via greedy decoding.
pub struct MoonshineModel {
    encoder: Session,
    decoder: Session,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl MoonshineModel {
    /// Load the two ONNX sessions for `model_id` from `dir`.
    ///
    /// `model_id` must be `"moonshine-tiny"` or `"moonshine-base"`; `dir`
    /// must contain `encoder_model.onnx` and `decoder_model_merged.onnx`.
    pub fn load(model_id: &str, dir: &Path) -> anyhow::Result<Self> {
        let (num_layers, num_kv_heads, head_dim) = match model_id {
            "moonshine-tiny" => (6, 8, 36),
            "moonshine-base" => (8, 8, 52),
            other => anyhow::bail!("unknown moonshine model id: {other}"),
        };
        let encoder = Session::builder()
            .map_err(plain)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(plain)?
            .commit_from_file(dir.join("encoder_model.onnx"))
            .map_err(|e| anyhow::anyhow!("encoder load: {e}"))?;
        let decoder = Session::builder()
            .map_err(plain)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(plain)?
            .commit_from_file(dir.join("decoder_model_merged.onnx"))
            .map_err(|e| anyhow::anyhow!("decoder load: {e}"))?;
        Ok(Self {
            encoder,
            decoder,
            num_layers,
            num_kv_heads,
            head_dim,
        })
    }

    /// Transcribe raw 16 kHz mono samples to token ids via greedy decoding.
    ///
    /// Returns a sequence starting with `DECODER_START`; it ends with `EOS`
    /// if the decoder emits one before the internal step budget is spent.
    ///
    /// Takes `&mut self` because `ort::Session::run` requires a mutable
    /// borrow of the session (confirmed against this `ort` version by the
    /// working `SileroVadEngine::compute` pattern in
    /// `src/audio/silero_vad_engine.rs`).
    pub fn generate(&mut self, samples: &[f32]) -> anyhow::Result<Vec<i64>> {
        if samples.is_empty() {
            return Ok(vec![DECODER_START, EOS]);
        }

        // Encoder: [1, num_samples] -> last_hidden_state [1, seq, hidden].
        let audio = Array2::from_shape_vec((1, samples.len()), samples.to_vec())?;
        let enc_out = self
            .encoder
            .run(ort::inputs!["input_values" => Value::from_array(audio)?])
            .map_err(plain)?;
        let (hs_shape, hs_data) = enc_out["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(plain)?;
        let seq = hs_shape[hs_shape.len() - 2] as usize;
        let hidden = hs_shape[hs_shape.len() - 1] as usize;
        let encoder_hidden_states = Array3::from_shape_vec((1, seq, hidden), hs_data.to_vec())?;

        // Empty KV cache tensors, shape [1, num_kv_heads, 0, head_dim].
        let empty = || Array4::<f32>::zeros((1, self.num_kv_heads, 0, self.head_dim));
        let mut past: Vec<(String, Array4<f32>)> = Vec::new();
        for i in 0..self.num_layers {
            for a in ["decoder", "encoder"] {
                for b in ["key", "value"] {
                    past.push((format!("past_key_values.{i}.{a}.{b}"), empty()));
                }
            }
        }

        // Generation budget: ~6 tokens/sec of audio, plus one for headroom.
        let max_len = ((samples.len() as f32 / 16_000.0) * 6.0) as usize + 1;
        let mut tokens = vec![DECODER_START];
        let mut input_ids = Array2::from_shape_vec((1, 1), vec![DECODER_START])?;

        for step in 0..max_len {
            let use_cache = step > 0;

            let mut named: Vec<(Cow<str>, SessionInputValue)> = Vec::new();
            named.push((
                Cow::Borrowed("input_ids"),
                Value::from_array(input_ids.clone())?.into(),
            ));
            named.push((
                Cow::Borrowed("encoder_hidden_states"),
                Value::from_array(encoder_hidden_states.clone())?.into(),
            ));
            named.push((
                Cow::Borrowed("use_cache_branch"),
                Value::from_array(Array1::<bool>::from_vec(vec![use_cache]))?.into(),
            ));
            for (name, t) in &past {
                named.push((
                    Cow::Owned(name.clone()),
                    Value::from_array(t.clone())?.into(),
                ));
            }

            let dec_out = self.decoder.run(named).map_err(plain)?;

            // Next token = argmax over the vocab of the last logits row.
            let (lshape, logits) = dec_out["logits"]
                .try_extract_tensor::<f32>()
                .map_err(plain)?;
            let vocab = lshape[lshape.len() - 1] as usize;
            let last = &logits[logits.len() - vocab..];
            let mut next = last
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i64)
                .unwrap_or(EOS);

            // Guard against premature EOS on the very first token when audio has
            // significant duration (> 1.0s) and non-EOS candidates are competitive.
            if step == 0 && next == EOS && samples.len() >= MOONSHINE_SAMPLE_RATE {
                let alt = last
                    .iter()
                    .enumerate()
                    .filter(|&(idx, _)| idx as i64 != EOS && idx as i64 != DECODER_START)
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, &v)| (i as i64, v));
                if let Some((candidate, cand_logit)) = alt {
                    let eos_logit = last[EOS as usize];
                    if eos_logit - cand_logit < 3.0 {
                        next = candidate;
                    }
                }
            }

            tokens.push(next);
            if next == EOS {
                break;
            }
            input_ids = Array2::from_shape_vec((1, 1), vec![next])?;

            // Update the KV cache from `present.*` outputs. Encoder-cache
            // entries are only produced on the first (no-cache) step and are
            // frozen thereafter, per the Moonshine reference decoder.
            for (name, slot) in past.iter_mut() {
                if use_cache && name.contains(".encoder.") {
                    continue;
                }
                let present = name.replacen("past_key_values", "present", 1);
                let (sh, data) = dec_out[present.as_str()]
                    .try_extract_tensor::<f32>()
                    .map_err(plain)?;
                let dims: Vec<usize> = sh.iter().map(|&d| d as usize).collect();
                *slot =
                    Array4::from_shape_vec((dims[0], dims[1], dims[2], dims[3]), data.to_vec())?;
            }
        }
        Ok(tokens)
    }
}

/// Target sample rate for Moonshine STT (16 kHz).
pub const MOONSHINE_SAMPLE_RATE: usize = 16_000;
/// Maximum duration of an individual audio chunk passed to the decoder (10 seconds).
const MAX_CHUNK_SAMPLES: usize = MOONSHINE_SAMPLE_RATE * 10;
/// Minimum duration of an audio chunk before splitting on silence (1.5 seconds).
const MIN_CHUNK_SAMPLES: usize = (MOONSHINE_SAMPLE_RATE as f32 * 1.5) as usize;
/// Frame length for energy/silence analysis (30 ms = 480 samples).
const ANALYSIS_FRAME_SAMPLES: usize = 480;
/// Consecutive silent frames required to declare a phrase pause (300 ms = 10 frames).
const SILENCE_GAP_FRAMES: usize = 10;
/// Padding added to the start and end of detected speech (100 ms = 1600 samples).
const SPEECH_PADDING_SAMPLES: usize = 1600;

fn frame_rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = frame.iter().map(|&s| s * s).sum();
    (sum_sq / frame.len() as f32).sqrt()
}

fn total_audio_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Trim excessive leading silence from an audio slice, keeping a pre-roll buffer.
/// Does NOT aggressively truncate the tail to avoid clipping trailing consonants.
fn trim_leading_silence(samples: &[f32], threshold: f32) -> &[f32] {
    if samples.len() < ANALYSIS_FRAME_SAMPLES {
        return samples;
    }
    for (i, frame) in samples.chunks(ANALYSIS_FRAME_SAMPLES).enumerate() {
        if frame_rms(frame) >= threshold {
            let start = (i * ANALYSIS_FRAME_SAMPLES).saturating_sub(SPEECH_PADDING_SAMPLES);
            return &samples[start..];
        }
    }
    // Entirely below threshold
    &[]
}

/// Split long audio into natural utterances using silence gaps.
/// Avoids feeding excessively long monolithic audio to Moonshine's autoregressive decoder.
fn segment_audio(samples: &[f32]) -> Vec<&[f32]> {
    if samples.len() <= MAX_CHUNK_SAMPLES {
        return vec![samples];
    }

    let overall_rms = total_audio_rms(samples);
    let silence_threshold = (overall_rms * 0.25).clamp(0.005, 0.03);

    let mut segments = Vec::new();
    let mut seg_start = 0;
    let mut silence_streak = 0;

    let mut i = 0;
    while i + ANALYSIS_FRAME_SAMPLES <= samples.len() {
        let frame = &samples[i..i + ANALYSIS_FRAME_SAMPLES];
        let rms = frame_rms(frame);

        if rms < silence_threshold {
            silence_streak += 1;
        } else {
            silence_streak = 0;
        }

        let cur_len = (i + ANALYSIS_FRAME_SAMPLES).saturating_sub(seg_start);

        let natural_pause = cur_len >= MIN_CHUNK_SAMPLES && silence_streak >= SILENCE_GAP_FRAMES;
        let hard_limit = cur_len >= MAX_CHUNK_SAMPLES;

        if natural_pause || hard_limit {
            let cut = if natural_pause {
                let silence_backtrack = (silence_streak * ANALYSIS_FRAME_SAMPLES) / 2;
                (i + ANALYSIS_FRAME_SAMPLES).saturating_sub(silence_backtrack)
            } else {
                i + ANALYSIS_FRAME_SAMPLES
            };

            if cut > seg_start + (MOONSHINE_SAMPLE_RATE / 2) {
                segments.push(&samples[seg_start..cut]);
                seg_start = cut;
                silence_streak = 0;
            }
        }

        i += ANALYSIS_FRAME_SAMPLES;
    }

    if seg_start < samples.len() {
        let remaining = &samples[seg_start..];
        if !remaining.is_empty() {
            segments.push(remaining);
        }
    }

    segments
}

/// A ready-to-use Moonshine STT engine: an ONNX model plus its tokenizer.
///
/// The underlying `MoonshineModel::generate` requires `&mut self` (`ort`
/// sessions need a mutable borrow to run), but `SttEngine::transcribe` takes
/// `&self`. A `Mutex` reconciles the two: it also gives us `Sync` for free,
/// which `SttEngine: Send + Sync` requires.
pub struct MoonshineEngine {
    model: Mutex<MoonshineModel>,
    tokenizer: Tokenizer,
}

impl MoonshineEngine {
    /// Load the Moonshine model and its tokenizer from `dir`.
    ///
    /// `dir` must contain `encoder_model.onnx`, `decoder_model_merged.onnx`,
    /// and `tokenizer.json`.
    pub fn load(model_id: &str, dir: &Path) -> anyhow::Result<Self> {
        let model = MoonshineModel::load(model_id, dir)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
        Ok(Self {
            model: Mutex::new(model),
            tokenizer,
        })
    }

    /// Transcribe a single audio slice (single utterance).
    fn transcribe_slice(&self, slice: &[f32]) -> anyhow::Result<String> {
        if slice.len() < ANALYSIS_FRAME_SAMPLES {
            return Ok(String::new());
        }

        let rms = total_audio_rms(slice);
        let threshold = (rms * 0.25).clamp(0.005, 0.03);
        let active = trim_leading_silence(slice, threshold);
        if active.len() < ANALYSIS_FRAME_SAMPLES {
            return Ok(String::new());
        }

        let tokens = {
            let mut m = self
                .model
                .lock()
                .map_err(|_| anyhow::anyhow!("moonshine model lock poisoned"))?;
            m.generate(active)?
        };

        let ids: Vec<u32> = tokens
            .iter()
            .filter(|&&t| t != DECODER_START && t != EOS && t >= 0)
            .map(|&t| t as u32)
            .collect();

        if ids.is_empty() {
            return Ok(String::new());
        }

        let text = self
            .tokenizer
            .decode(&ids, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        Ok(text.trim().to_string())
    }
}

impl SttEngine for MoonshineEngine {
    fn transcribe(&self, samples: &[f32]) -> anyhow::Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        if samples.len() <= MAX_CHUNK_SAMPLES {
            return self.transcribe_slice(samples);
        }

        let segments = segment_audio(samples);
        let mut results = Vec::with_capacity(segments.len());

        for seg in segments {
            let part = self.transcribe_slice(seg)?;
            if !part.is_empty() {
                results.push(part);
            }
        }

        Ok(results.join(" "))
    }

    fn is_ready(&self) -> bool {
        true
    }
}

#[cfg(test)]
fn _assert_send_sync<T: Send + Sync>() {}
#[cfg(test)]
fn _moonshine_engine_is_send_sync() {
    _assert_send_sync::<MoonshineEngine>();
}

#[cfg(test)]
mod tests {
    use super::*;

    // Gated: only runs when the base model is present locally (CI has no model).
    fn base_dir() -> Option<std::path::PathBuf> {
        let d = dirs::home_dir()?.join(".cache/pie/models/moonshine-base");
        d.join("encoder_model.onnx").exists().then_some(d)
    }

    /// Read a 16 kHz mono 16-bit PCM fixture into f32 samples (the format the
    /// checked-in test WAVs use), so tests need no WAV loader in the shipped lib.
    fn load_fixture_16k(path: &str) -> Vec<f32> {
        let mut reader = hound::WavReader::open(path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1, "fixture must be mono");
        assert_eq!(spec.sample_rate, 16000, "fixture must be 16 kHz");
        let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
        reader
            .samples::<i32>()
            .map(|s| s.unwrap() as f32 / max)
            .collect()
    }

    #[test]
    fn generate_produces_tokens_ending_in_eos() {
        let Some(dir) = base_dir() else {
            eprintln!("skip: no local base model");
            return;
        };
        let mut m = MoonshineModel::load("moonshine-base", &dir).unwrap();
        let samples = load_fixture_16k("tests/fixtures/moonshine_hello.wav");
        let toks = m.generate(&samples).unwrap();
        assert!(toks.first() == Some(&DECODER_START));
        assert!(
            toks.last() == Some(&EOS) || toks.len() > 3,
            "should decode some tokens"
        );
    }

    #[test]
    fn transcribe_hello_fixture() {
        let Some(dir) = base_dir() else {
            eprintln!("skip: no local base model");
            return;
        };
        let eng = MoonshineEngine::load("moonshine-base", &dir).unwrap();
        assert!(eng.is_ready());
        let samples = load_fixture_16k("tests/fixtures/moonshine_hello.wav");

        let text = eng.transcribe(&samples).unwrap().to_lowercase();
        assert!(
            text.contains("hello") && (text.contains("test") || text.contains("task")),
            "got: {text}"
        );
    }

    #[test]
    fn trim_leading_silence_trims_leading_zeros() {
        let sample_rate = MOONSHINE_SAMPLE_RATE;
        let mut audio = vec![0.0f32; sample_rate * 2]; // 2s silence
        let speech = vec![0.2f32; sample_rate]; // 1s active
        audio.extend_from_slice(&speech);
        audio.extend(vec![0.0f32; sample_rate * 2]); // 2s trailing

        let trimmed = trim_leading_silence(&audio, 0.01);
        assert!(!trimmed.is_empty());
        // Leading 2s trimmed (leaving ~100ms padding) + 1s speech + 2s trailing
        assert!(trimmed.len() < sample_rate * 4);
        assert!(trimmed.len() >= sample_rate * 3);
    }

    #[test]
    fn segment_audio_short_returns_single_slice() {
        let short = vec![0.1f32; MOONSHINE_SAMPLE_RATE * 5];
        let segs = segment_audio(&short);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].len(), short.len());
    }

    #[test]
    fn segment_audio_splits_long_audio_with_pauses() {
        let sr = MOONSHINE_SAMPLE_RATE;
        // Build 15s audio: 4s speech, 1s silence, 4s speech, 1s silence, 5s speech
        let mut audio = Vec::new();
        audio.extend(vec![0.15f32; sr * 4]);
        audio.extend(vec![0.0f32; sr]);
        audio.extend(vec![0.15f32; sr * 4]);
        audio.extend(vec![0.0f32; sr]);
        audio.extend(vec![0.15f32; sr * 5]);

        let segs = segment_audio(&audio);
        assert!(
            segs.len() >= 2,
            "must segment long audio with pauses, got {}",
            segs.len()
        );
        for s in &segs {
            assert!(
                s.len() <= MAX_CHUNK_SAMPLES,
                "chunk duration must not exceed max"
            );
        }
    }
}
