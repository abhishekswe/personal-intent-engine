use crate::corrector::llm_correct;
use crate::corrector::{AppliedFix, CorrectionOutcome, PronunciationCorrector};
use crate::intent::{Intent, IntentExtractor};
use crate::llm::{LlmRouter, RouterLlmClient};
use crate::memory::store::MemoryStore;
use crate::optimizer::{self, OptimizationMode};
use crate::stt::SttEngine;
use std::path::PathBuf;
use tokio::sync::mpsc;

/// Fire-and-forget signal from the pipeline to the background learner.
pub struct LearnTask {
    /// The raw (pre-correction) transcript, for the learner to mine.
    pub raw_transcript: String,
    /// The user's configured role, if any, for context-aware learning.
    pub role: Option<String>,
    /// The user's known technologies, for context-aware learning.
    pub technologies: Vec<String>,
}

/// Result of processing input through the PIE pipeline
#[derive(Debug)]
pub struct PieResult {
    /// Extracted intent
    pub intent: Intent,

    /// Optimized prompt
    pub optimized_prompt: String,

    /// Optimization mode used
    pub mode: OptimizationMode,

    /// Estimated token count
    pub estimated_tokens: usize,

    /// The transcript after correction (what intent/optimize actually saw).
    pub corrected_transcript: String,

    /// Corrections applied to the transcript, for UI transparency.
    pub applied: Vec<AppliedFix>,
}

/// Word count above which input is considered complex: `Enhanced` mode (with
/// LLM-backed intent extraction) instead of `Direct`.
pub const ENHANCED_WORD_THRESHOLD: usize = 20;

/// The main PIE engine that orchestrates the full pipeline.
///
/// Pipeline: Input -> Intent Extraction -> Memory Lookup -> Prompt Optimization -> LLM
pub struct PieEngine {
    memory: MemoryStore,
    extractor: IntentExtractor,
    llm: LlmRouter,
    stt: Option<Box<dyn SttEngine>>,
    corrector: PronunciationCorrector,
    learner_tx: Option<mpsc::Sender<LearnTask>>,
    /// Mined corrections coming back from the background learner. The engine is
    /// the SOLE writer of `learned_vocab.json`: it drains this in `process()`
    /// and applies each via `add_auto_correction`, so reinforcement, sync, and
    /// mining all write through one owner (no concurrent-writer race).
    mined_rx: Option<mpsc::Receiver<crate::corrector::learner::ExtractedCorrection>>,
    learned_vocab_path: Option<PathBuf>,
    learned_mtime: Option<std::time::SystemTime>,
    /// When true, `process()` translates spoken code phrases (e.g. "console
    /// dot log" -> "console.log(") after pronunciation correction and before
    /// intent extraction. Off by default.
    code_mode: bool,
}

/// Default on-disk location for the learned/synced vocabulary store. Mirrors
/// `corrector::default_learned_path` (kept private there) so the engine can
/// stat the file for reload-on-change without exposing the path externally.
fn default_learned_vocab_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("pie")
        .join("learned_vocab.json")
}

impl PieEngine {
    /// Initialize the PIE engine
    pub async fn new() -> anyhow::Result<Self> {
        let memory = MemoryStore::load();
        let extractor = IntentExtractor::new();
        let llm = LlmRouter::new();
        let corrector = PronunciationCorrector::new();

        Ok(Self {
            memory,
            extractor,
            llm,
            stt: None,
            corrector,
            learner_tx: None,
            mined_rx: None,
            learned_vocab_path: Some(default_learned_vocab_path()),
            learned_mtime: None,
            code_mode: false,
        })
    }

    /// Initialize the engine with a BYOK LLM config (falls back to env vars
    /// when the config URL is empty).
    pub async fn with_config(config: &crate::llm::LlmConfig) -> anyhow::Result<Self> {
        let mut engine = Self::new().await?;
        engine.set_llm_config(config);
        Ok(engine)
    }

    /// Rebuild the LLM router from a new config (e.g. after the user edits
    /// LLM settings). Does not touch memory, STT, or the corrector.
    pub fn set_llm_config(&mut self, config: &crate::llm::LlmConfig) {
        self.llm = crate::llm::LlmRouter::from_config(config);
    }

    /// Enable/disable code-aware post-processing (spoken code -> syntax).
    pub fn set_code_mode(&mut self, on: bool) {
        self.code_mode = on;
    }

    /// Test/ephemeral engine: performs NO disk persistence. Memory lives only
    /// in-process (never saved), and the corrector reads/writes an isolated
    /// `user_dict_path` instead of the real user config — so integration tests
    /// never touch or pollute real app data.
    #[doc(hidden)]
    pub fn new_ephemeral(user_dict_path: std::path::PathBuf) -> Self {
        Self {
            memory: MemoryStore::default(),
            extractor: IntentExtractor::new(),
            llm: LlmRouter::new(),
            stt: None,
            corrector: PronunciationCorrector::with_user_path(user_dict_path),
            learner_tx: None,
            mined_rx: None,
            learned_vocab_path: None,
            learned_mtime: None,
            code_mode: false,
        }
    }

    /// Attach a speech-to-text engine, enabling `process_audio`.
    pub fn with_stt(mut self, stt: Box<dyn SttEngine>) -> Self {
        self.stt = Some(stt);
        self
    }

    /// Transcribe 16 kHz mono samples and run them through the full pipeline.
    /// The transcript is available afterwards as `intent.raw_input`.
    pub async fn process_audio(
        &mut self,
        samples: &[f32],
        mode: &str,
    ) -> anyhow::Result<PieResult> {
        let stt = self
            .stt
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No STT engine configured. Use with_stt()."))?;

        let text = stt.transcribe(samples)?;
        let text = text.trim();
        log::info!("Transcribed {} samples: {text:?}", samples.len());
        if text.is_empty() {
            anyhow::bail!("Transcription produced no text (silence or unintelligible audio)");
        }

        self.process(text, mode).await
    }

    /// Drain corrections mined by the background learner and apply them via
    /// `add_auto_correction` (the engine owns the corrector and is the sole
    /// writer of learned_vocab.json). Non-blocking; errors on a single term
    /// are ignored rather than failing the pipeline.
    fn drain_mined(&mut self) {
        let Some(rx) = &mut self.mined_rx else {
            return;
        };
        let mut terms = Vec::new();
        while let Ok(term) = rx.try_recv() {
            terms.push(term);
        }
        for term in terms {
            let _ = self
                .corrector
                .add_auto_correction(&term.heard, &term.canonical);
        }
    }

    /// Reload learned vocab if the file changed since we last looked. Cheap
    /// stat; only rebuilds when mtime advanced.
    fn maybe_reload_learned(&mut self) {
        let Some(path) = &self.learned_vocab_path else {
            return;
        };
        let Ok(meta) = std::fs::metadata(path) else {
            return;
        };
        let Ok(mtime) = meta.modified() else {
            return;
        };
        if self.learned_mtime != Some(mtime) {
            self.learned_mtime = Some(mtime);
            let _ = self.corrector.reload_learned();
        }
    }

    /// Process text input through the full PIE pipeline.
    /// Returns the extracted intent and optimized prompt.
    pub async fn process(&mut self, input: &str, mode: &str) -> anyhow::Result<PieResult> {
        let raw = input.to_string();

        // Apply any corrections the background learner mined since last turn.
        // The engine is the sole writer of learned_vocab.json, so these land
        // through the same corrector as reinforcement and sync (no file race).
        self.drain_mined();

        // Reload learned vocab in case the file changed on disk (e.g. an
        // external edit or a sync run from another path), so this turn sees it.
        self.maybe_reload_learned();

        // Step 0: Correct speech-to-text jargon errors before anything else.
        // Allow-set: terms the user is known to use, so static phonetic entries
        // only fire for relevant terms. Derived from the profile's tech stack.
        let allowed: std::collections::HashSet<String> = self
            .memory
            .profile
            .technologies
            .iter()
            .map(|t| t.to_lowercase())
            .collect();
        let correction = self.corrector.correct(input, &allowed);
        let corrected_text = if self.code_mode {
            crate::corrector::code_phrases::apply_code_phrases(&correction.text)
        } else {
            correction.text.clone()
        };
        for fix in &correction.applied {
            // `from` is the lowercased heard phrase; reinforce if it's learned.
            // This write is safe: the engine is the SOLE writer of
            // learned_vocab.json — reinforcement, sync (add_synced_correction),
            // and mined terms (drain_mined) all go through this one
            // engine-owned corrector, serialized behind the app's engine mutex.
            // The background learner only sends candidates over a channel and
            // never touches the file, so there is no concurrent-writer race.
            let _ = self.corrector.reinforce_learned(&fix.from);
        }
        let input = corrected_text.as_str();

        // Step 1: Select mode. Explicit "direct"/"enhanced" are honored; any
        // other value (including legacy mode names from old settings) means
        // auto-select from input complexity.
        let optimization_mode = match mode {
            "direct" => OptimizationMode::Direct,
            "enhanced" => OptimizationMode::Enhanced,
            _ => {
                let complex = input.split_whitespace().count() > ENHANCED_WORD_THRESHOLD
                    || input.contains('?');
                if complex {
                    OptimizationMode::Enhanced
                } else {
                    OptimizationMode::Direct
                }
            }
        };

        // Step 2: Extract intent. Enhanced mode goes through the LLM (it
        // understands meaning in rambling speech); Direct mode and any LLM
        // failure use the deterministic rule-based extractor.
        let intent = match optimization_mode {
            OptimizationMode::Enhanced if self.llm.is_available("openai") => {
                // `None` model = use the router's configured default (the
                // user's BYOK model). Passing an explicit model here would
                // hardcode a provider-specific name and break every BYOK
                // setup whose provider doesn't serve it.
                let client = RouterLlmClient::new(&self.llm, "openai", None);
                let user_context = self.extraction_context();
                self.extractor
                    .extract_with_llm(input, &client, user_context.as_deref())
                    .await
            }
            _ => self.extractor.extract(input),
        };

        // Step 3: Record interaction in memory
        let conv_type = format!("{:?}", intent.conversation_type);
        self.memory.record_interaction(input, &conv_type);

        // Step 4: Optimize prompt based on mode
        let optimized_prompt = optimizer::optimize(&intent, optimization_mode);
        let estimated_tokens = optimized_prompt.len() / 4;

        // Step 5: Save memory
        let _ = self.memory.save();

        // Fire-and-forget: hand the raw transcript to the background learner,
        // if one is attached. Never blocks the pipeline.
        if let Some(tx) = &self.learner_tx {
            let _ = tx.try_send(LearnTask {
                raw_transcript: raw,
                role: self.memory.profile.role.clone(),
                technologies: self.memory.profile.technologies.clone(),
            });
        }

        Ok(PieResult {
            intent,
            optimized_prompt,
            mode: optimization_mode,
            estimated_tokens,
            corrected_transcript: corrected_text.clone(),
            applied: correction.applied,
        })
    }

    /// Fast voice-to-text path: apply ONLY pronunciation correction (dictionary
    /// + context-gated phonetic + learned vocab, plus code-phrase translation
    /// when code mode is on). No intent extraction, no optimization, no LLM
    /// call, and no memory writes — so it stays instant regardless of input
    /// length. Returns the corrected transcript and the fixes that were applied.
    /// Used as the default flow; `process` is the opt-in AI-enhanced path.
    pub fn correct_only(&mut self, input: &str) -> (String, Vec<AppliedFix>) {
        // Pick up anything the background learner mined and any on-disk change.
        self.drain_mined();
        self.maybe_reload_learned();

        let allowed: std::collections::HashSet<String> = self
            .memory
            .profile
            .technologies
            .iter()
            .map(|t| t.to_lowercase())
            .collect();
        let correction = self.corrector.correct(input, &allowed);
        let corrected = if self.code_mode {
            crate::corrector::code_phrases::apply_code_phrases(&correction.text)
        } else {
            correction.text.clone()
        };
        for fix in &correction.applied {
            let _ = self.corrector.reinforce_learned(&fix.from);
        }
        (corrected, correction.applied)
    }

    /// User context string for LLM intent extraction, from the memory profile.
    /// `None` when the profile has nothing useful to add.
    fn extraction_context(&self) -> Option<String> {
        let role = self.memory.profile.role.as_deref();
        let tech = &self.memory.profile.technologies;
        match (role, tech.is_empty()) {
            (None, true) => None,
            (Some(r), true) => Some(format!("role={r}")),
            (None, false) => Some(format!("tech={}", tech.join(", "))),
            (Some(r), false) => Some(format!("role={r}, tech={}", tech.join(", "))),
        }
    }

    /// Send optimized prompt to an LLM provider
    pub async fn send_to_llm(
        &self,
        prompt: &str,
        provider: &str,
        model: Option<&str>,
    ) -> anyhow::Result<String> {
        self.llm.send(prompt, provider, model).await
    }

    /// Get the current memory store (for inspection)
    pub fn memory(&self) -> &MemoryStore {
        &self.memory
    }

    /// Get a mutable reference to memory (for profile updates)
    pub fn memory_mut(&mut self) -> &mut MemoryStore {
        &mut self.memory
    }

    /// The user's own heard->canonical corrections (for UI listing/editing).
    pub fn corrector_user_corrections(&self) -> Vec<crate::corrector::Correction> {
        self.corrector.user_corrections()
    }

    /// Add or update a user correction.
    pub fn corrector_add(&mut self, heard: &str, canonical: &str) -> anyhow::Result<()> {
        self.corrector.add_user_correction(heard, canonical)
    }

    /// Remove a user correction.
    pub fn corrector_remove(&mut self, heard: &str) -> anyhow::Result<()> {
        self.corrector.remove_user_correction(heard)
    }

    /// Attach a background learner: `process` will forward a `LearnTask` for
    /// every turn (non-blocking; dropped if the channel is full or closed).
    pub fn set_learner_tx(&mut self, tx: mpsc::Sender<LearnTask>) {
        self.learner_tx = Some(tx);
    }

    /// Attach the channel on which the background learner returns mined
    /// corrections; `process()` drains and applies them.
    pub fn set_mined_rx(
        &mut self,
        rx: mpsc::Receiver<crate::corrector::learner::ExtractedCorrection>,
    ) {
        self.mined_rx = Some(rx);
    }

    /// Number of learned (auto + synced) corrections, for UI display.
    pub fn corrector_learned_count(&self) -> usize {
        self.corrector.learned_count()
    }

    /// Clear all learned/synced corrections (never touches the user dict).
    pub fn corrector_reset_learned(&mut self) -> anyhow::Result<()> {
        self.corrector.reset_learned()
    }

    /// Test-only ephemeral engine with an isolated learned-vocab path in
    /// addition to the isolated user dict, so tests can exercise learned
    /// corrections/reload without touching real app data.
    #[doc(hidden)]
    pub fn new_ephemeral_with_learned(user: PathBuf, learned: PathBuf) -> Self {
        let mut e = Self::new_ephemeral(user.clone());
        e.corrector = crate::corrector::PronunciationCorrector::with_paths(user, learned.clone());
        e.learned_vocab_path = Some(learned);
        e
    }

    /// Test-only: add an auto-learned correction directly to the corrector.
    #[doc(hidden)]
    pub fn corrector_add_auto(&mut self, heard: &str, canonical: &str) -> anyhow::Result<()> {
        self.corrector.add_auto_correction(heard, canonical)
    }

    /// Test-only: the `seen_count` of a learned entry, or 0 when absent.
    #[doc(hidden)]
    pub fn corrector_learned_seen(&self, heard: &str) -> u32 {
        self.corrector.learned_entries_seen(heard).unwrap_or(0)
    }

    /// Import vocabulary from the given paths via the configured LLM, storing
    /// terms as `Source::Synced`. User-initiated only.
    ///
    /// Borrow-safe by construction: `run_sync`'s `add` closure captures only a
    /// local `Vec` (not `self`), so it can run alongside the immutable borrow
    /// of `self.llm` for the duration of the `.await`. Once `run_sync`
    /// returns, that borrow ends and the collected pairs are applied via
    /// `self.corrector.add_synced_correction`, which needs `&mut self`.
    pub async fn run_vocabulary_sync(
        &mut self,
        paths: Vec<PathBuf>,
        provider: &str,
        model: Option<&str>,
        on_progress: impl Fn(usize, usize),
    ) -> anyhow::Result<crate::corrector::sync::SyncResult> {
        let mut pairs: Vec<(String, String)> = Vec::new();
        let result = crate::corrector::sync::run_sync(
            &paths,
            &self.llm,
            provider,
            model,
            |variant, canonical| {
                pairs.push((variant.to_string(), canonical.to_string()));
                Ok(())
            },
            on_progress,
        )
        .await?;
        for (variant, canonical) in &pairs {
            let _ = self.corrector.add_synced_correction(variant, canonical);
        }
        let _ = crate::corrector::sync::record_sync_state(result.terms_added);
        Ok(result)
    }

    /// Opt-in deep correction via the configured LLM. NOT on the always-on
    /// path — called only from the settings toggle or the on-demand command.
    /// Returns `Err` on any LLM failure; callers decide whether to fall back to
    /// the deterministic result (toggle path) or surface the error (on-demand).
    pub async fn deep_correct(
        &self,
        transcript: &str,
        provider: &str,
        model: Option<&str>,
    ) -> anyhow::Result<CorrectionOutcome> {
        let prompt = llm_correct::build_prompt(
            transcript,
            self.memory.profile.role.as_deref(),
            &self.memory.profile.technologies,
        );
        let corrected = self.llm.send(&prompt, provider, model).await?;
        let corrected = corrected.trim().to_string();
        let applied = llm_correct::diff_fixes(transcript, &corrected);
        Ok(CorrectionOutcome {
            text: corrected,
            applied,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LlmConfig;

    #[tokio::test]
    async fn with_config_uses_configured_client() {
        let cfg = LlmConfig {
            api_url: "https://api.example.com/v1".into(),
            api_key: "sk-x".into(),
            model: "gpt-4o-mini".into(),
        };
        let engine = PieEngine::with_config(&cfg).await.unwrap();
        assert!(engine.llm.is_available("openai"));
    }

    #[tokio::test]
    async fn set_llm_config_rebuilds_router() {
        std::env::remove_var("OPENAI_API_KEY");
        let mut engine = PieEngine::new().await.unwrap();
        assert!(!engine.llm.is_available("openai"));
        engine.set_llm_config(&LlmConfig {
            api_url: "https://api.example.com/v1".into(),
            api_key: "sk-x".into(),
            model: String::new(),
        });
        assert!(engine.llm.is_available("openai"));
    }

    #[tokio::test]
    async fn process_reinforces_a_firing_learned_correction() {
        // Ephemeral engine with injected corrector paths.
        let dir = std::env::temp_dir();
        let uid = format!("{}-{}", std::process::id(), line!());
        let cpath = dir.join(format!("pie-eng-user-{uid}.json"));
        let lpath = dir.join(format!("pie-eng-learned-{uid}.json"));
        let mut engine = PieEngine::new_ephemeral_with_learned(cpath.clone(), lpath.clone());
        engine
            .corrector_add_auto("terra form", "Terraform")
            .unwrap();
        let before = engine.corrector_learned_seen("terra form");
        let _ = engine
            .process("deploy with terra form now", "balanced")
            .await
            .unwrap();
        let after = engine.corrector_learned_seen("terra form");
        assert!(
            after > before,
            "a firing learned correction must be reinforced"
        );
        let _ = std::fs::remove_file(cpath);
        let _ = std::fs::remove_file(lpath);
    }

    #[tokio::test]
    async fn auto_mode_short_input_selects_direct() {
        let dir = std::env::temp_dir();
        let uid = format!("{}-{}", std::process::id(), line!());
        let cpath = dir.join(format!("pie-md-user-{uid}.json"));
        let lpath = dir.join(format!("pie-md-learned-{uid}.json"));
        let mut engine = PieEngine::new_ephemeral_with_learned(cpath.clone(), lpath.clone());
        let res = engine.process("build a rust cli", "auto").await.unwrap();
        assert_eq!(res.mode, OptimizationMode::Direct);
        assert!(!res.optimized_prompt.is_empty());
        let _ = std::fs::remove_file(cpath);
        let _ = std::fs::remove_file(lpath);
    }

    #[tokio::test]
    async fn auto_mode_long_input_selects_enhanced() {
        let dir = std::env::temp_dir();
        let uid = format!("{}-{}", std::process::id(), line!());
        let cpath = dir.join(format!("pie-md2-user-{uid}.json"));
        let lpath = dir.join(format!("pie-md2-learned-{uid}.json"));
        let mut engine = PieEngine::new_ephemeral_with_learned(cpath.clone(), lpath.clone());
        let long = "so ".to_string() + &"refactor the widget ".repeat(15); // > 20 words
        let res = engine.process(&long, "auto").await.unwrap();
        assert_eq!(res.mode, OptimizationMode::Enhanced);
        // No LLM configured in tests -> rule-based fallback still yields a prompt.
        assert!(!res.optimized_prompt.is_empty());
        let _ = std::fs::remove_file(cpath);
        let _ = std::fs::remove_file(lpath);
    }

    #[tokio::test]
    async fn legacy_mode_strings_map_to_auto_selection() {
        let dir = std::env::temp_dir();
        let uid = format!("{}-{}", std::process::id(), line!());
        let cpath = dir.join(format!("pie-md3-user-{uid}.json"));
        let lpath = dir.join(format!("pie-md3-learned-{uid}.json"));
        let mut engine = PieEngine::new_ephemeral_with_learned(cpath.clone(), lpath.clone());
        // Old saved settings may still say "balanced"/"compact"/"refine".
        let res = engine
            .process("build a rust cli", "balanced")
            .await
            .unwrap();
        assert_eq!(res.mode, OptimizationMode::Direct);
        let _ = std::fs::remove_file(cpath);
        let _ = std::fs::remove_file(lpath);
    }

    #[tokio::test]
    async fn explicit_enhanced_mode_is_honored_for_short_input() {
        let dir = std::env::temp_dir();
        let uid = format!("{}-{}", std::process::id(), line!());
        let cpath = dir.join(format!("pie-md4-user-{uid}.json"));
        let lpath = dir.join(format!("pie-md4-learned-{uid}.json"));
        let mut engine = PieEngine::new_ephemeral_with_learned(cpath.clone(), lpath.clone());
        let res = engine
            .process("build a rust cli", "enhanced")
            .await
            .unwrap();
        assert_eq!(res.mode, OptimizationMode::Enhanced);
        let _ = std::fs::remove_file(cpath);
        let _ = std::fs::remove_file(lpath);
    }

    #[tokio::test]
    async fn code_mode_translates_spoken_code() {
        let dir = std::env::temp_dir();
        let uid = format!("{}-{}", std::process::id(), line!());
        let cpath = dir.join(format!("pie-cm-user-{uid}.json"));
        let lpath = dir.join(format!("pie-cm-learned-{uid}.json"));
        let mut engine = PieEngine::new_ephemeral_with_learned(cpath.clone(), lpath.clone());
        engine.set_code_mode(true);
        let res = engine
            .process("console dot log hello", "compact")
            .await
            .unwrap();
        assert!(
            res.corrected_transcript.contains("console.log("),
            "got: {}",
            res.corrected_transcript
        );
        let _ = std::fs::remove_file(cpath);
        let _ = std::fs::remove_file(lpath);
    }

    #[tokio::test]
    async fn drains_mined_corrections_and_applies_them() {
        use crate::corrector::learner::ExtractedCorrection;
        let dir = std::env::temp_dir();
        let uid = format!("{}-{}", std::process::id(), line!());
        let cpath = dir.join(format!("pie-mined-user-{uid}.json"));
        let lpath = dir.join(format!("pie-mined-learned-{uid}.json"));
        let mut engine = PieEngine::new_ephemeral_with_learned(cpath.clone(), lpath.clone());
        let (tx, rx) = tokio::sync::mpsc::channel::<ExtractedCorrection>(10);
        engine.set_mined_rx(rx);
        tx.send(ExtractedCorrection {
            heard: "terra form".into(),
            canonical: "Terraform".into(),
        })
        .await
        .unwrap();
        // process() drains the channel and applies via add_auto_correction.
        let res = engine
            .process("i use terra form daily", "compact")
            .await
            .unwrap();
        assert_eq!(res.corrected_transcript, "i use Terraform daily");
        assert_eq!(engine.corrector_learned_count(), 1);
        let _ = std::fs::remove_file(cpath);
        let _ = std::fs::remove_file(lpath);
    }

    #[tokio::test]
    async fn code_mode_off_leaves_transcript_untouched() {
        let dir = std::env::temp_dir();
        let uid = format!("{}-{}", std::process::id(), line!());
        let cpath = dir.join(format!("pie-cm2-user-{uid}.json"));
        let lpath = dir.join(format!("pie-cm2-learned-{uid}.json"));
        let mut engine = PieEngine::new_ephemeral_with_learned(cpath.clone(), lpath.clone());
        // default: code_mode off
        let res = engine
            .process("open the bracket please", "compact")
            .await
            .unwrap();
        assert!(
            !res.corrected_transcript.contains('['),
            "off mode must not translate"
        );
        let _ = std::fs::remove_file(cpath);
        let _ = std::fs::remove_file(lpath);
    }

    #[tokio::test]
    async fn run_vocabulary_sync_counts_conversations_echo() {
        let dir = std::env::temp_dir();
        let uid = format!("{}-{}", std::process::id(), line!());
        let src = dir.join(format!("pie-syncsrc-{uid}"));
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("a.md"), "deploy nextjs on vercel").unwrap();
        let cpath = dir.join(format!("pie-eng-user-{uid}.json"));
        let lpath = dir.join(format!("pie-eng-learned-{uid}.json"));
        let mut engine = PieEngine::new_ephemeral_with_learned(cpath.clone(), lpath.clone());
        let res = engine
            .run_vocabulary_sync(vec![src.clone()], "echo", None, |_d, _t| {})
            .await
            .unwrap();
        assert_eq!(res.conversations, 1);
        let _ = std::fs::remove_dir_all(src);
        let _ = std::fs::remove_file(cpath);
        let _ = std::fs::remove_file(lpath);
    }
}
