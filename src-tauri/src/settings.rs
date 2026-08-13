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
    /// Global shortcut that pastes the raw transcript
    /// (tauri-plugin-global-shortcut syntax, e.g. "CmdOrCtrl+Shift+V").
    pub hotkey_raw: String,
    /// Global shortcut that pastes the PIE-optimized prompt
    /// (tauri-plugin-global-shortcut syntax, e.g. "CmdOrCtrl+Shift+Space").
    pub hotkey_optimized: String,
    /// Default output for the UI record button (and the fallback paste mode):
    /// "transcript" (raw speech-to-text) or "prompt" (PIE-optimized prompt).
    /// The two hotkeys override this per-press.
    pub paste_output: String,
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
            hotkey_raw: "CmdOrCtrl+Shift+V".to_string(),
            hotkey_optimized: "CmdOrCtrl+Shift+Space".to_string(),
            paste_output: "transcript".to_string(),
            history_limit: 10,
            deep_correct_ai: false,
            background_mining: false,
            code_mode: false,
        }
    }
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

    /// Parse settings JSON, migrating a legacy single `hotkey` into the dual
    /// hotkeys based on the legacy `paste_output` (`"prompt"` -> optimized,
    /// otherwise raw). New installs (no legacy `hotkey`) keep the defaults.
    pub fn from_json_migrating(json: &str) -> Self {
        let mut s: Settings = serde_json::from_str(json).unwrap_or_else(|e| {
            log::warn!("Failed to parse settings ({e}); using defaults");
            Self::default()
        });
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(json) {
            let has_new = v.get("hotkey_raw").is_some() || v.get("hotkey_optimized").is_some();
            if let (false, Some(legacy)) = (has_new, v.get("hotkey").and_then(|h| h.as_str())) {
                if !legacy.trim().is_empty() {
                    let to_prompt =
                        v.get("paste_output").and_then(|p| p.as_str()) == Some("prompt");
                    if to_prompt {
                        s.hotkey_optimized = legacy.to_string();
                    } else {
                        s.hotkey_raw = legacy.to_string();
                    }
                }
            }
        }
        s
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
    fn dual_hotkey_defaults() {
        let s = Settings::default();
        assert_eq!(s.hotkey_raw, "CmdOrCtrl+Shift+V");
        assert_eq!(s.hotkey_optimized, "CmdOrCtrl+Shift+Space");
    }

    #[test]
    fn migrates_legacy_hotkey_by_paste_output() {
        // Legacy install: single hotkey + paste_output=prompt -> optimized.
        let legacy = r#"{"hotkey":"CmdOrCtrl+Alt+P","paste_output":"prompt"}"#;
        let migrated = Settings::from_json_migrating(legacy);
        assert_eq!(migrated.hotkey_optimized, "CmdOrCtrl+Alt+P");
        assert_eq!(migrated.hotkey_raw, "CmdOrCtrl+Shift+V"); // default keeps

        // Legacy install: single hotkey + paste_output=transcript -> raw.
        let legacy2 = r#"{"hotkey":"CmdOrCtrl+Alt+R","paste_output":"transcript"}"#;
        let m2 = Settings::from_json_migrating(legacy2);
        assert_eq!(m2.hotkey_raw, "CmdOrCtrl+Alt+R");
        assert_eq!(m2.hotkey_optimized, "CmdOrCtrl+Shift+Space"); // default keeps
    }

    #[test]
    fn new_install_no_legacy_key_uses_defaults() {
        let s = Settings::from_json_migrating(r#"{"mode":"balanced"}"#);
        assert_eq!(s.hotkey_raw, "CmdOrCtrl+Shift+V");
        assert_eq!(s.hotkey_optimized, "CmdOrCtrl+Shift+Space");
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
