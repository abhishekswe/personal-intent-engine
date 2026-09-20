# Developing PIE

PIE is a Rust library and CLI with a Tauri 2 desktop application and a Svelte 5 interface.

## Prerequisites

- Rust stable
- Node.js 18 or newer
- Platform dependencies required by [Tauri 2](https://v2.tauri.app/start/prerequisites/)

Clone the repository and install the frontend dependencies:

```bash
git clone https://github.com/abhishekswe/personal-intent-engine.git
cd personal-intent-engine
npm --prefix ui install
cargo install tauri-cli --version "^2" --locked
```

## Run the desktop app

```bash
cargo tauri dev
```

Use the Tauri CLI for desktop builds. A plain `cargo build` compiles a binary that expects the Vite development server and therefore opens a blank window when the server is absent.

## Build a standalone desktop app

```bash
cargo tauri build --no-bundle  # embedded frontend, no installer
cargo tauri build              # platform application and installer bundles
```

For a local macOS installation, use `./scripts/dev-install.sh`. It applies the stable local signing requirement needed to preserve Accessibility permission across rebuilds. Read [signing.md](signing.md) before changing signing configuration.

## CLI

Process text without the desktop application:

```bash
cargo run -- --mode direct --provider echo \
  "set up a Rust API with PostgreSQL and no ORM"
```

Use Enhanced mode with an OpenAI-compatible provider:

```bash
OPENAI_API_KEY=sk-... cargo run -- \
  --mode enhanced --provider openai --model gpt-4o-mini \
  "help me plan a migration, keep downtime below five minutes"
```

Run `cargo run -- --help` for the current flags.

## Rust library

The core pipeline is exported by `pie-engine`:

```rust
use pie_engine::PieEngine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut engine = PieEngine::new().await?;
    let result = engine.process("build a Rust API with PostgreSQL", "direct").await?;
    println!("{}", result.optimized_prompt);
    Ok(())
}
```

The repository currently consumes the library by path. Package registry distribution is not implied by this example.

## Repository map

```text
src/             Core library and CLI
  audio/         Capture, resampling, and voice activity detection
  stt/           Speech-to-text adapters, including Moonshine
  corrector/     Built-in, personal, learned, and AI-assisted corrections
  intent/        Intent schema, classification, and extraction
  optimizer/     Direct and Enhanced prompt output
  memory/        Local profile and communication patterns
  history/       Local SQLite recording history
  llm/           OpenAI-compatible routing
  pipeline/      End-to-end orchestration
src-tauri/       Desktop shell, commands, hotkey, paste, models, and signing config
ui/              Svelte interface and recording overlay
```

External capabilities sit behind traits (`AudioCapture`, `VoiceActivityDetector`, `SttEngine`, and `LlmClient`) so production adapters and tests use the same boundaries.

## Verification

```bash
npm --prefix ui run build
cargo fmt --all --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets
```

Tests must not call external LLM services; use the existing mock client.

## More technical documentation

- [Architecture](ARCHITECTURE.md)
- [Release process](RELEASING.md)
- [macOS signing](signing.md)
- [Intent extraction test results](INTENT_EXTRACTION_TEST_RESULTS.md)
- [Project contribution rules](../AGENTS.md)
