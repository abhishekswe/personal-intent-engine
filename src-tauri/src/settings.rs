use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Desktop app settings, persisted as JSON at ~/.config/pie/settings.json
/// (next to the engine's memory.json).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Catalog id of the selected speech-to-text model (e.g.
    /// `"moonshine-base"`), not a path — the engine loads it by id from its
    /// directory under `~/.cache/pie/models/<id>/`. A stale `whisper_model`
    /// key from older installs is ignored (serde default).
    pub stt_model: String,
    /// Path to the Silero VAD ONNX model (empty = record without VAD)
    pub silero_model: String,
    /// Spoken language code or "auto"
    pub language: String,
    /// Prompt optimization mode
    pub mode: String,
    /// LLM provider ("echo", "openai", "openrouter")
    pub provider: String,
    /// LLM model name (empty = provider default)
    pub llm_model: String,
    /// OpenAI-compatible base URL for BYOK (empty = fall back to env vars).
    pub llm_api_url: String,
    /// API key / bearer token for BYOK (empty = none / local server).
    pub llm_api_key: String,
    /// Global shortcut that toggles dictation on and off.
    pub hotkey: String,
    /// Max number of recordings kept in the history store (hard cap).
    pub history_limit: usize,
    /// When true, run the opt-in LLM deep-correct pass on every transcript.
    pub deep_correct_ai: bool,
    /// When true, run the opt-in background learner that mines new
    /// pronunciation corrections from transcripts via the configured LLM.
    pub background_mining: bool,
    /// When true, translate spoken code patterns into syntax ("console dot log"
    /// -> "console.log(") after pronunciation correction. Off by default so
    /// ordinary dictation is never affected.
    pub code_mode: bool,
    /// When true, run the LLM enhancement pipeline (intent extraction +
    /// prompt optimization, and any opt-in deep-correct) after transcription.
    /// OFF by default: the default flow is pure voice-to-text — transcribe +
    /// dictionary correction + paste — with NO model/LLM call in the path, so
    /// it stays instant. Turning this on opts into the (slower) AI rewrite.
    pub enhance_with_ai: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            stt_model: "moonshine-base".to_string(),
            silero_model: existing_cache_model("silero_vad_v4.onnx"),
            language: "auto".to_string(),
            // "auto" = engine selects direct/enhanced from input complexity.
            // Legacy values ("balanced", "compact", ...) also map to auto.
            mode: "auto".to_string(),
            provider: "echo".to_string(),
            llm_model: String::new(),
            llm_api_url: String::new(),
            llm_api_key: String::new(),
            hotkey: default_hotkey().to_string(),
            history_limit: 10,
            deep_correct_ai: false,
            background_mining: false,
            code_mode: false,
            enhance_with_ai: false,
        }
    }
}

pub const fn default_hotkey() -> &'static str {
    "Control+Space"
}

/// Default to a model already present in ~/.cache/pie/models, else empty.
fn existing_cache_model(filename: &str) -> String {
    let Some(home) = dirs::home_dir() else {
        return String::new();
    };
    let path = home.join(".cache/pie/models").join(filename);
    if path.exists() {
        path.to_string_lossy().into_owned()
    } else {
        String::new()
    }
}

fn settings_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("pie")
        .join("settings.json")
}

impl Settings {
    pub fn load() -> Self {
        let path = settings_path();
        match std::fs::read_to_string(&path) {
            Ok(json) => Self::from_json_migrating(&json),
            Err(_) => Self::default(),
        }
    }

    /// Parse settings and collapse the former raw/optimized bindings into the
    /// single dictation toggle. Raw wins because voice-to-text is the default.
    pub fn from_json_migrating(json: &str) -> Self {
        let mut settings: Self = serde_json::from_str(json).unwrap_or_else(|e| {
            log::warn!("Failed to parse settings ({e}); using defaults");
            Self::default()
        });
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(json) {
            if value.get("hotkey").is_none() {
                let legacy = value
                    .get("hotkey_raw")
                    .and_then(|v| v.as_str())
                    .filter(|v| !v.trim().is_empty())
                    .or_else(|| {
                        value
                            .get("hotkey_optimized")
                            .and_then(|v| v.as_str())
                            .filter(|v| !v.trim().is_empty())
                    });
                if let Some(hotkey) = legacy {
                    settings.hotkey = hotkey.to_string();
                }
            }
        }
        settings
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let path = settings_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    /// Expand a leading `~/` so paths pasted from the shell work.
    pub fn expand(path: &str) -> PathBuf {
        if let Some(rest) = path.strip_prefix("~/") {
            if let Some(home) = dirs::home_dir() {
                return home.join(rest);
            }
        }
        PathBuf::from(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_roundtrip_json() {
        let settings = Settings {
            stt_model: "moonshine-tiny".into(),
            mode: "enhanced".into(),
            ..Settings::default()
        };
        let json = serde_json::to_string(&settings).unwrap();
        let loaded: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.stt_model, "moonshine-tiny");
        assert_eq!(loaded.mode, "enhanced");
    }

    #[test]
    fn stt_model_defaults_to_moonshine_base() {
        let s = Settings::default();
        assert_eq!(s.stt_model, "moonshine-base");
    }

    #[test]
    fn hotkey_uses_platform_default() {
        let s = Settings::default();
        assert_eq!(s.hotkey, "Control+Space");
    }

    #[test]
    fn hotkey_roundtrips() {
        let s = Settings {
            hotkey: "Control+Shift+Space".into(),
            ..Settings::default()
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.hotkey, "Control+Shift+Space");
    }

    #[test]
    fn migrates_legacy_raw_hotkey() {
        let loaded = Settings::from_json_migrating(
            r#"{"hotkey_raw":"Command+Shift+KeyV","hotkey_optimized":"Command+Shift+Space"}"#,
        );
        assert_eq!(loaded.hotkey, "Command+Shift+KeyV");
    }

    #[test]
    fn stale_whisper_model_key_is_ignored() {
        // Older installs persisted `whisper_model`; it's not a field anymore,
        // so it's silently dropped and `stt_model` falls back to default.
        let loaded: Settings =
            serde_json::from_str(r#"{"whisper_model":"/tmp/ggml-tiny.en.bin"}"#).unwrap();
        assert_eq!(loaded.stt_model, "moonshine-base");
    }

    #[test]
    fn partial_settings_fill_defaults() {
        let loaded: Settings = serde_json::from_str(r#"{"mode":"compact"}"#).unwrap();
        assert_eq!(loaded.mode, "compact");
        assert_eq!(loaded.language, "auto");
        assert!(!loaded.deep_correct_ai);
        assert!(!loaded.background_mining);
        assert!(!loaded.code_mode);
    }

    #[test]
    fn background_mining_roundtrips() {
        let s = Settings {
            background_mining: true,
            ..Settings::default()
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: Settings = serde_json::from_str(&json).unwrap();
        assert!(back.background_mining);
    }

    #[test]
    fn expand_tilde() {
        let expanded = Settings::expand("~/models/x.bin");
        assert!(!expanded.to_string_lossy().starts_with('~'));
        assert!(expanded.to_string_lossy().ends_with("models/x.bin"));
    }

    #[test]
    fn byok_fields_default_empty_and_roundtrip() {
        let loaded: Settings = serde_json::from_str(r#"{"mode":"compact"}"#).unwrap();
        assert_eq!(loaded.llm_api_url, "");
        assert_eq!(loaded.llm_api_key, "");

        let s = Settings {
            llm_api_url: "https://api.openai.com/v1".into(),
            llm_api_key: "sk-abc".into(),
            ..Settings::default()
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.llm_api_url, "https://api.openai.com/v1");
        assert_eq!(back.llm_api_key, "sk-abc");
    }
}
