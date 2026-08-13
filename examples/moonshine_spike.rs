//! Throwaway spike (Task 0 of the Moonshine STT migration plan): confirm
//! Moonshine's ONNX encoder/decoder load cleanly under `ort 2.0.0-rc.12`,
//! print their real input/output tensor names, and confirm the exact `ort`
//! mechanism for passing a *programmatically built* set of named inputs to
//! `Session::run` (needed for the decoder's `past_key_values.*` KV cache,
//! which the fixed-arity `ort::inputs!` macro can't express on its own).
//!
//! Not production code — deleted in Task 6 cleanup. Run with:
//!   cargo run --example moonshine_spike --features vad,whisper

use std::borrow::Cow;

use ort::session::{builder::GraphOptimizationLevel, Session, SessionInputValue};
use ort::value::Value;

/// Collapse a parameterized `ort::Error<R>` (borrows the builder/session and
/// so isn't `Send`/`Sync`) into the plain `ort::Error` `anyhow` can absorb.
/// Same workaround already used in `src/audio/silero_vad_engine.rs`.
fn plain(e: impl Into<ort::Error>) -> ort::Error {
    e.into()
}

fn main() -> anyhow::Result<()> {
    let dir = dirs::home_dir()
        .unwrap()
        .join(".cache/pie/models/moonshine-base");

    let mut enc = Session::builder()
        .map_err(plain)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(plain)?
        .commit_from_file(dir.join("encoder_model.onnx"))
        .map_err(|e| anyhow::anyhow!("encoder load: {e}"))?;
    let mut dec = Session::builder()
        .map_err(plain)?
        .commit_from_file(dir.join("decoder_model_merged.onnx"))
        .map_err(|e| anyhow::anyhow!("decoder load: {e}"))?;

    println!(
        "ENCODER inputs:  {:?}",
        enc.inputs().iter().map(|i| i.name()).collect::<Vec<_>>()
    );
    println!(
        "ENCODER outputs: {:?}",
        enc.outputs().iter().map(|o| o.name()).collect::<Vec<_>>()
    );
    println!(
        "DECODER inputs:  {:?}",
        dec.inputs().iter().map(|i| i.name()).collect::<Vec<_>>()
    );
    println!(
        "DECODER outputs: {:?}",
        dec.outputs().iter().map(|o| o.name()).collect::<Vec<_>>()
    );
    // Also dump dtypes so we can see rank/shape hints (e.g. symbolic dims on
    // the past_key_values.* tensors), not just names.
    for i in enc.inputs() {
        println!("  ENC IN  {} :: {:?}", i.name(), i.dtype());
    }
    for o in enc.outputs() {
        println!("  ENC OUT {} :: {:?}", o.name(), o.dtype());
    }
    for i in dec.inputs() {
        println!("  DEC IN  {} :: {:?}", i.name(), i.dtype());
    }
    for o in dec.outputs() {
        println!("  DEC OUT {} :: {:?}", o.name(), o.dtype());
    }

    // ---- Run the encoder on a real WAV -------------------------------
    let samples = pie_engine::stt::load_wav_as_16k_mono(&dir.join("hello.wav"))?;
    println!("wav samples: {}", samples.len());
    let arr = ndarray::Array2::from_shape_vec((1, samples.len()), samples)?;
    let enc_out = enc.run(ort::inputs!["input_values" => Value::from_array(arr)?])?;
    let (shape, data) = enc_out["last_hidden_state"].try_extract_tensor::<f32>()?;
    println!("encoder last_hidden_state shape: {shape:?}");
    println!("encoder last_hidden_state first 5 values: {:?}", &data[..5.min(data.len())]);

    // ---- Confirm the *programmatic* multi-input run API on the decoder ----
    // This is the crux of the spike: ort::inputs! only handles a fixed,
    // syntactically-known set of names. The decoder needs one entry per KV
    // cache layer (`past_key_values.{i}.decoder.key` etc.), a count only
    // known at runtime from `dec.inputs()`. Build a
    // `Vec<(Cow<str>, SessionInputValue)>` by hand and push into it in a
    // loop; `Session::run` accepts it because `ort` impls
    // `From<Vec<(K, V)>> for SessionInputs` for `K: Into<Cow<str>>`,
    // `V: Into<SessionInputValue>`.
    let last_hidden_state_shape: Vec<i64> = shape.iter().copied().collect();
    let encoder_hidden_states = Value::from_array((last_hidden_state_shape, data.to_vec()))?;

    let mut decoder_inputs: Vec<(Cow<str>, SessionInputValue)> = Vec::new();
    decoder_inputs.push((
        Cow::Borrowed("input_ids"),
        Value::from_array(ndarray::Array2::<i64>::from_shape_vec((1, 1), vec![1i64])?)?.into(),
    ));
    decoder_inputs.push((
        Cow::Borrowed("encoder_hidden_states"),
        encoder_hidden_states.into(),
    ));
    decoder_inputs.push((
        Cow::Borrowed("use_cache_branch"),
        Value::from_array(ndarray::Array1::<bool>::from_vec(vec![false]))?.into(),
    ));
    // Programmatically fill every past_key_values.* input the decoder
    // declares, with an empty (0-length sequence) placeholder tensor. Shapes
    // below are guesses at (batch, heads, seq=0, head_dim) refined from the
    // printed dtypes above; the goal here is only to prove the *plumbing*
    // compiles/runs, per the brief.
    for input in dec.inputs() {
        let name = input.name().to_string();
        if name.starts_with("past_key_values.") {
            // Confirmed via a first run's shape-mismatch error message:
            // (batch=1, heads=8, seq=0 for an empty cache, head_dim=52).
            let empty = ndarray::Array4::<f32>::zeros((1, 8, 0, 52));
            decoder_inputs.push((Cow::Owned(name), Value::from_array(empty)?.into()));
        }
    }

    match dec.run(decoder_inputs) {
        Ok(dec_out) => {
            println!(
                "DECODER run OK, outputs: {:?}",
                dec_out.keys().collect::<Vec<_>>()
            );
        }
        Err(e) => {
            // A shape mismatch here is still a useful finding (it tells us
            // the *real* head-count/head-dim), and the surrounding plumbing
            // (Vec<(Cow<str>, SessionInputValue)> -> session.run) is what we
            // actually need to confirm for Task 0.
            println!("DECODER run failed (expected if placeholder KV shapes are wrong): {e}");
        }
    }

    Ok(())
}
