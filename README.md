<p align="center">
  <img src="assets/icon.png" alt="PIE Logo" width="120" height="120" />
</p>

<h1 align="center">PIE — Personal Intent Engine</h1>

<p align="center">
  <strong>Transform spoken thought into structured, production-ready prompts delivered directly to your active text field.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Windows-lightgrey.svg" alt="Platform">
  <img src="https://img.shields.io/badge/Rust-stable-000000.svg?logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/Tauri-2-24C8DB.svg?logo=tauri" alt="Tauri">
  <img src="https://img.shields.io/badge/Svelte-5-FF3E00.svg?logo=svelte" alt="Svelte">
  <img src="https://img.shields.io/badge/privacy-100%25%20local%20speech-green.svg" alt="Privacy">
</p>

---

## Overview

Most voice-to-text tools dump raw, rambling transcripts into your focused application. **PIE** acts as intelligent middleware between your voice and AI models. Press a global shortcut anywhere, speak naturally, and PIE transcribes your speech **100% locally** (Metal-accelerated `whisper.cpp`), extracts your core intent, crafts a structured prompt, and pastes it into your focused text area — or routes it to your preferred LLM.

Your audio never leaves your machine.

---

## How It Works

```
 ┌────────────────┐      ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
 │ Spoken Thought │ ───► │  Local STT +   │ ───► │ Intent Engine  │ ───► │ Prompt Output  │
 │                │      │ Jargon Correct │      │                │      │                │
 └────────────────┘      └────────────────┘      └────────────────┘      └────────────────┘
   "set up a next          whisper.cpp +           "next jazz" → Next.js   Pasted at cursor
    jazz app with          Silero VAD, then        Extracts objective,     or routed to
    postgres, no ORM"      fix dev jargon          constraints, mode       OpenAI / LLM API
```

---

## Key Features

- 🎙️ **Universal Global Hotkey** — Trigger recording (`⌘⇧Space`) from any application; results paste directly at your cursor.
- 🔒 **100% On-Device Transcription** — High-speed, private speech-to-text powered by `whisper.cpp` with Apple Silicon Metal acceleration and Silero VAD.
- 🧠 **Intent Extraction & Optimization** — Automatically strips filler and converts conversational speech into structured prompts (`compact`, `balanced`, `enhanced`, `adaptive`).
- 🗣️ **Personal Pronunciation Corrector** — Fixes speech-to-text mangling of developer jargon (`next jazz` → `Next.js`, `coobernetes` → `Kubernetes`) using a built-in dictionary plus terms you teach it. Runs on-device; an optional AI pass handles novel garbling.
- ⚡ **Direct AI Integration** — Optionally route prompts directly to OpenAI or any OpenAI-compatible API endpoint.
- 🕘 **Local Recording History** — Every recording is saved to a local SQLite store you can revisit — nothing leaves your machine.
- 🖥️ **Lightweight Tray App** — Runs silently in the menu bar with a customizable floating overlay.
- 📦 **In-App Model Management** — Download and manage Whisper and Silero VAD models directly within the app.
- 🛠️ **Developer Friendly** — Available as a desktop application, a standalone CLI tool, or a reusable Rust crate.

---

## Personal Pronunciation Corrector

Generic dictation garbles technical terms and never learns your vocabulary. PIE corrects them in layers, all on-device by default:

1. **Built-in dictionary** — ships with common developer terms (`Next.js`, `Nginx`, `Kubernetes`, `PostgreSQL`, `kubectl`, …).
2. **Your vocabulary** — teach PIE any `heard → correct` mapping from Settings; it is saved to `pronunciation.json` and applied instantly on every future recording.
3. **AI deep-correct (opt-in)** — for novel mistakes the dictionary misses, an optional pass sends the transcript to your configured LLM (local or remote) to fix garbled terms only. Off by default; toggle it in Settings or hit **Re-correct with AI** on any result. One tap saves the fix into your vocabulary, so the next time is instant and offline.

A phonetic tier is **context-gated** — it only corrects *toward* terms you actually use, so an ordinary word like "next" is never turned into "Next.js" unless you've told PIE you work with it. Every correction is shown on the result (`heard → corrected`) so nothing changes silently.

---

## Quick Install

Builds are signed with a **stable self-signed certificate** (not an Apple Developer ID, so not notarized). macOS Gatekeeper still blocks un-notarized apps on first launch, so the commands below strip the quarantine attribute (`xattr -cr`) to open PIE cleanly. Because the certificate is stable across releases, your **Microphone and Accessibility permissions survive every update** rather than being re-prompted.

### macOS (Apple Silicon)

```bash
# One-line install: downloads the latest release, installs to /Applications, clears quarantine
curl -fsSL https://raw.githubusercontent.com/abhishekswe/personal-intent-engine/main/scripts/install.sh | bash
```

To review the script before running it, download it first with `-o install.sh`, read it, then `bash install.sh`.

Alternatively, via **Homebrew**:
```bash
brew tap abhishekswe/pie https://github.com/abhishekswe/homebrew-pie
brew install --cask abhishekswe/pie/pie
xattr -cr /Applications/PIE.app
```

> **Manual Install**: Download `PIE_<version>_aarch64.dmg` from [Releases](https://github.com/abhishekswe/personal-intent-engine/releases), move to `/Applications`, then run `xattr -cr /Applications/PIE.app` (or right-click → **Open** the first time).

### Windows

Download the latest `.exe` installer from [Releases](https://github.com/abhishekswe/personal-intent-engine/releases). SmartScreen may warn about an unrecognized app — click **More info → Run anyway**.

---

## Quick Start (Building from Source)

### Prerequisites

- **Rust** (stable) & **Node.js** (v18+)
- **CMake** (required for `whisper.cpp` compilation: `brew install cmake`)

> Developed and tested on **macOS 11+ (Apple Silicon)** with Metal acceleration. Windows and Linux code paths exist but are currently untested.

### Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/abhishekswe/personal-intent-engine.git
cd personal-intent-engine

# 2. Install Tauri CLI
cargo install tauri-cli --version "^2" --locked

# 3. Launch application in dev mode
cargo tauri dev
```

### Initial Run Setup

1. Open the **Models** tab in PIE to download a Whisper model (*Whisper Tiny* recommended) and *Silero VAD*.
2. Grant **Microphone** and **Accessibility** permissions when prompted by macOS.
3. Use the global shortcut `⌘⇧Space` to begin recording anywhere.

---

## CLI & Rust Library

### CLI Usage

```bash
# Process raw text into a structured prompt
cargo run -- --mode balanced --provider echo "help me set up docker with postgres for rust, no ORM"

# Send directly to an OpenAI model
OPENAI_API_KEY=sk-... cargo run -- --provider openai --model gpt-4o-mini "what is a lifetime in Rust?"

# Transcribe and process a WAV audio file
cargo run --features whisper -- \
  --audio-file input.wav \
  --whisper-model ~/.cache/pie/models/ggml-tiny.en.bin \
  --provider echo
```

Key flags: `--mode {compact|balanced|enhanced|adaptive}`, `--provider {echo|openai|openrouter}`, `--model`, `--language`, `--audio-file`, `--verbose`.

### Rust Library Usage

Add `pie-engine` to your `Cargo.toml`:

```rust
use pie_engine::PieEngine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut engine = PieEngine::new().await?;
    let result = engine.process("build a rest api in rust with postgres", "balanced").await?;

    println!("Optimized Prompt:\n{}", result.optimized_prompt);
    Ok(())
}
```

---

## Project Architecture

```
personal-intent-engine/
├── src/            # Core Rust library (pie-engine) & CLI (pie-cli)
│   ├── audio/      # Audio capture, resampling, Silero VAD
│   ├── stt/        # whisper.cpp STT integration
│   ├── corrector/  # Pronunciation corrector (dictionary + phonetic + LLM deep-correct)
│   ├── intent/     # Intent extraction & classification logic
│   ├── optimizer/  # Prompt rewriting engines
│   ├── memory/     # User profile & learned patterns
│   ├── history/    # Local SQLite recording history
│   └── llm/        # OpenAI-compatible API router
├── src-tauri/      # Tauri 2 Desktop container (pie-desktop)
└── ui/             # Svelte 5 frontend (Control Panel & Recording Overlay)
```

---

## Configuration

Settings live at `~/Library/Application Support/pie/settings.json` and are managed from the app's Settings panes:

| Setting | Description |
|---|---|
| Whisper / Silero model | Paths to the local models (set by the Models tab). |
| Language | Spoken-language ISO code, or `auto` to detect. |
| Optimization mode | How speech becomes a prompt (`compact` / `balanced` / `enhanced` / `adaptive`). |
| Provider / model | LLM target for "Send to LLM". `echo` reflects the prompt back for testing; `openai`/`openrouter` need `OPENAI_API_KEY`. |
| Hotkey | Global shortcut, rebindable by pressing a combo. |
| Paste output | Whether the hotkey pastes the raw transcript or the optimized prompt. |
| Deep-correct with AI | Opt-in LLM pass that fixes garbled terms the dictionary misses. Off by default; uses your configured provider (local or remote). |
| Vocabulary | Your personal `heard → correct` corrections (`pronunciation.json`), editable in Settings. |
| History | Number of past recordings kept in the local SQLite history. |

---

## Privacy & Security

PIE operates under a **strict local-first paradigm**:
- Audio streams and transcriptions remain **100% local on your machine**.
- Outbound network requests occur **only** if you configure an external LLM provider and explicitly request to route prompts to it.
- No analytics or telemetry tracking.

---

## Acknowledgements

PIE's audio-capture and desktop layers include work derived from two MIT-licensed projects:

- **[Handy](https://github.com/cjpais/handy)** — the worker-thread audio capture loop, VAD smoothing state machine, and cpal stream configuration in `src/audio/`.
- **[OpenSuperWhisper](https://github.com/starmel/OpenSuperWhisper)** — the recording indicator and clipboard paste flow in `src-tauri/`.

Thanks to both projects for doing the hard parts in the open. Their copyright notices are retained in [NOTICE](NOTICE).

---

## License

Distributed under the [Apache 2.0 License](LICENSE), with derived portions under MIT as recorded in [NOTICE](NOTICE).


---

Built by **Abhishek Kumar** ([@abhishekswe](https://github.com/abhishekswe)) · [abhishekswe.github.io](https://abhishekswe.github.io) · [LinkedIn](https://www.linkedin.com/in/abhishekkr-swe/) · [X](https://x.com/abhishekswe)
