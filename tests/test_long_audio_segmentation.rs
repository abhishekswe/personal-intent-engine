#![cfg(all(feature = "vad", target_os = "macos"))]

use pie_engine::stt::{MoonshineEngine, SttEngine};
use std::process::Command;

fn load_wav_16k_mono(path: &std::path::Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("wav open failed");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1);
    assert_eq!(spec.sample_rate, 16000);
    let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
    reader
        .samples::<i32>()
        .map(|s| s.expect("sample read failed") as f32 / max)
        .collect()
}

fn synthesize_sentence(text: &str, out_wav: &std::path::Path) {
    let dir = out_wav.parent().unwrap();
    let aiff = dir.join("tmp.aiff");
    let status = Command::new("say")
        .arg("-o")
        .arg(&aiff)
        .arg(text)
        .status()
        .expect("say failed");
    assert!(status.success());
    let status = Command::new("afconvert")
        .args(["-f", "WAVE", "-d", "LEI16@16000", "-c", "1"])
        .arg(&aiff)
        .arg(out_wav)
        .status()
        .expect("afconvert failed");
    assert!(status.success());
    let _ = std::fs::remove_file(aiff);
}

fn local_model_dir() -> Option<std::path::PathBuf> {
    let home = dirs::home_dir()?;
    let tiny = home.join(".cache/pie/models/moonshine-tiny");
    if tiny.join("encoder_model.onnx").exists() {
        return Some(tiny);
    }
    let base = home.join(".cache/pie/models/moonshine-base");
    if base.join("encoder_model.onnx").exists() {
        return Some(base);
    }
    None
}

#[test]
fn long_audio_with_pauses_transcribes_completely() {
    let Some(model_dir) = local_model_dir() else {
        eprintln!("skipping: no local moonshine model");
        return;
    };
    let model_id = if model_dir.ends_with("moonshine-base") {
        "moonshine-base"
    } else {
        "moonshine-tiny"
    };
    let engine = MoonshineEngine::load(model_id, &model_dir).expect("load engine");

    let tmp = std::env::temp_dir().join("pie_long_stt_test");
    std::fs::create_dir_all(&tmp).unwrap();

    let sentences = [
        "First, we initialize the voice recognition system and verify that speech detection begins promptly.",
        "Second, we continue speaking for several sentences to ensure the transcription does not cut off prematurely.",
        "Third, after pausing between sentences, the engine must resume capturing all following words accurately.",
        "Finally, the complete transcript should contain all four sections rather than just one or two words.",
    ];

    let mut combined_audio = Vec::new();
    // Prepend 1.5s leading silence to test silence immunity
    combined_audio.extend(vec![0.0f32; 16000 * 3 / 2]);

    for (idx, text) in sentences.iter().enumerate() {
        let wav = tmp.join(format!("s_{idx}.wav"));
        synthesize_sentence(text, &wav);
        let samples = load_wav_16k_mono(&wav);
        combined_audio.extend_from_slice(&samples);
        // Add 1.2s silence pause between sentences
        combined_audio.extend(vec![0.0f32; (16000.0 * 1.2) as usize]);
    }

    let dur_secs = combined_audio.len() as f32 / 16000.0;
    assert!(dur_secs >= 20.0, "test audio should be substantial (>20s)");

    let transcript = engine.transcribe(&combined_audio).expect("transcribe");
    let lower = transcript.to_lowercase();

    // Verify key words from each of the 4 sentences are present (proves no premature truncation)
    assert!(
        lower.contains("first") || lower.contains("initialize"),
        "sentence 1 missing from transcript: {transcript}"
    );
    assert!(
        lower.contains("second") || lower.contains("continue"),
        "sentence 2 missing from transcript: {transcript}"
    );
    assert!(
        lower.contains("third") || lower.contains("pausing") || lower.contains("engine"),
        "sentence 3 missing from transcript: {transcript}"
    );
    assert!(
        lower.contains("finally") || lower.contains("complete") || lower.contains("four"),
        "sentence 4 missing from transcript: {transcript}"
    );

    // Total word count should be high (>35 words), never 1-2 words
    let word_count = transcript.split_whitespace().count();
    assert!(
        word_count >= 35,
        "expected >=35 words across long multi-sentence audio, got {word_count}: {transcript}"
    );
}

#[test]
fn pure_silence_returns_empty_string() {
    let Some(model_dir) = local_model_dir() else {
        return;
    };
    let model_id = if model_dir.ends_with("moonshine-base") {
        "moonshine-base"
    } else {
        "moonshine-tiny"
    };
    let engine = MoonshineEngine::load(model_id, &model_dir).expect("load engine");

    let silence = vec![0.0f32; 16000 * 3]; // 3s of pure silence
    let res = engine.transcribe(&silence).expect("transcribe silence");
    assert_eq!(res.trim(), "");
}
