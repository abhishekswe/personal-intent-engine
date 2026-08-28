//! Silero VAD integration test: a real ONNX model must classify synthesized
//! speech as speech and digital silence as noise, through the VadPipeline
//! wrapper exactly as the recorder uses it.
//!
//! Requires the `vad` feature, macOS (`say`/`afconvert`), and the Silero model
//! in the cache. Skips cleanly when those are missing.
//!
//! Run with: cargo test --features vad --test silero_vad

#![cfg(all(feature = "vad", target_os = "macos"))]

use std::path::PathBuf;
use std::process::Command;

use pie_engine::audio::{
    SileroVad, VadFrame, VadPipeline, VoiceActivityDetector, FRAME_SAMPLES, PIE_VAD_THRESHOLD,
    VAD_CONTEXT_FRAMES, VAD_HANGOVER_FRAMES, VAD_SPEECH_THRESHOLD_FRAMES,
};

/// Read a 16 kHz mono 16-bit PCM WAV (what `afconvert -d LEI16@16000 -c 1`
/// produces) into f32 samples. Test-only, via the `hound` dev-dependency —
/// the shipped library carries no WAV loader.
fn load_wav_16k_mono(path: &std::path::Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("wav open failed");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "expected mono");
    assert_eq!(spec.sample_rate, 16000, "expected 16 kHz");
    let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
    reader
        .samples::<i32>()
        .map(|s| s.expect("wav sample read failed") as f32 / max)
        .collect()
}

fn model_path() -> Option<PathBuf> {
    let path = std::env::var_os("PIE_SILERO_MODEL")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(|home| PathBuf::from(home).join(".cache/pie/models/silero_vad_v4.onnx"))
        })?;
    path.exists().then_some(path)
}

fn smoothed_silero(model: &PathBuf) -> VadPipeline {
    let silero = SileroVad::new(model, PIE_VAD_THRESHOLD).expect("silero load failed");
    VadPipeline::new(
        Box::new(silero),
        VAD_CONTEXT_FRAMES,
        VAD_HANGOVER_FRAMES,
        VAD_SPEECH_THRESHOLD_FRAMES,
    )
}

fn count_speech_frames(vad: &mut VadPipeline, samples: &[f32]) -> (usize, usize) {
    let mut speech = 0;
    let mut total = 0;
    for frame in samples.chunks_exact(FRAME_SAMPLES) {
        total += 1;
        if matches!(vad.push_frame(frame).unwrap(), VadFrame::Speech(_)) {
            speech += 1;
        }
    }
    (speech, total)
}

#[test]
fn detects_speech_and_rejects_silence() {
    let Some(model) = model_path() else {
        eprintln!("skipping: no silero model (set PIE_SILERO_MODEL)");
        return;
    };

    // --- Synthesized speech must be detected ---
    let dir = std::env::temp_dir().join("pie-vad-test");
    std::fs::create_dir_all(&dir).unwrap();
    let aiff = dir.join("speech.aiff");
    let wav = dir.join("speech.wav");

    let Ok(status) = Command::new("say")
        .arg("-o")
        .arg(&aiff)
        .arg("testing voice activity detection with real speech")
        .status()
    else {
        eprintln!("skipping: `say` unavailable");
        return;
    };
    assert!(status.success(), "say failed");
    let status = Command::new("afconvert")
        .args(["-f", "WAVE", "-d", "LEI16@16000", "-c", "1"])
        .arg(&aiff)
        .arg(&wav)
        .status()
        .expect("afconvert unavailable");
    assert!(status.success(), "afconvert failed");

    let speech_samples = load_wav_16k_mono(&wav);
    let mut vad = smoothed_silero(&model);
    let (speech, total) = count_speech_frames(&mut vad, &speech_samples);
    assert!(
        speech * 2 > total,
        "expected majority speech frames in spoken audio, got {speech}/{total}"
    );

    // --- Digital silence must be rejected (fresh detector state) ---
    let mut vad = smoothed_silero(&model);
    let silence = vec![0.0f32; FRAME_SAMPLES * 100]; // 3 seconds
    let (speech, total) = count_speech_frames(&mut vad, &silence);
    assert_eq!(
        speech, 0,
        "silence must produce zero speech frames, got {speech}/{total}"
    );

    // --- reset() clears LSTM state between sessions ---
    let mut vad = smoothed_silero(&model);
    let _ = count_speech_frames(&mut vad, &speech_samples);
    vad.reset();
    let (speech, _) = count_speech_frames(&mut vad, &silence);
    assert_eq!(
        speech, 0,
        "silence after reset must produce zero speech frames"
    );
}
