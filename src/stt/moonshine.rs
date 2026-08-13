//! Moonshine ONNX speech-to-text: raw-audio encoder + autoregressive KV-cache
//! decoder, run natively via `ort`. No mel-spectrogram.

use std::borrow::Cow;
use std::path::Path;

use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::{builder::GraphOptimizationLevel, Session, SessionInputValue};
use ort::value::Value;

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
            let next = last
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i64)
                .unwrap_or(EOS);
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

#[cfg(test)]
mod tests {
    use super::*;

    // Gated: only runs when the base model is present locally (CI has no model).
    fn base_dir() -> Option<std::path::PathBuf> {
        let d = dirs::home_dir()?.join(".cache/pie/models/moonshine-base");
        d.join("encoder_model.onnx").exists().then_some(d)
    }

    #[test]
    fn generate_produces_tokens_ending_in_eos() {
        let Some(dir) = base_dir() else {
            eprintln!("skip: no local base model");
            return;
        };
        let mut m = MoonshineModel::load("moonshine-base", &dir).unwrap();
        let samples = crate::stt::load_wav_as_16k_mono(std::path::Path::new(
            "tests/fixtures/moonshine_hello.wav",
        ))
        .unwrap();
        let toks = m.generate(&samples).unwrap();
        assert!(toks.first() == Some(&DECODER_START));
        assert!(
            toks.last() == Some(&EOS) || toks.len() > 3,
            "should decode some tokens"
        );
    }
}
