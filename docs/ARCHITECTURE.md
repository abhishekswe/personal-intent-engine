# PIE architecture

PIE is a Rust library and CLI with a Tauri desktop application. The desktop app captures speech, transcribes it locally, corrects vocabulary, optionally extracts intent with an LLM, and pastes the result into the active application.

## Data flow

```text
Microphone or text
        ↓
Audio capture → resampling → voice activity detection
        ↓
Local speech-to-text (Moonshine)
        ↓
Vocabulary correction
        ↓
Intent extraction and prompt optimization
        ↓
Paste at cursor, save to local history, or send to a configured LLM
```

Short, direct input stays on the deterministic fast path. Longer or question-shaped input can use LLM-backed extraction when Enhanced mode is selected or Auto mode chooses it. If the LLM is unavailable or returns invalid output, intent extraction falls back to the local rule-based path.

## Components

### Core library (`src/`)

- `audio/` captures audio with cpal, resamples to 16 kHz mono, and filters silence with Silero VAD.
- `stt/` exposes the `SttEngine` seam and the Moonshine adapter.
- `corrector/` applies built-in, personal, phonetic, learned, and optional LLM-assisted corrections.
- `intent/` classifies input and extracts objectives, context, constraints, and questions.
- `optimizer/` produces Direct or Enhanced output.
- `memory/` stores the local profile and communication patterns.
- `history/` stores recent recordings in SQLite.
- `llm/` routes requests to OpenAI-compatible providers behind `LlmClient`.
- `pipeline/` wires the full processing flow through `PieEngine`.

External capabilities are isolated behind traits:

| Capability | Trait | Production adapter |
|---|---|---|
| Audio capture | `AudioCapture` | `AudioRecorder` |
| Voice activity | `VoiceActivityDetector` | `SileroVad` or `EnergyVad` |
| Speech-to-text | `SttEngine` | `MoonshineEngine` |
| LLM completion | `LlmClient` | `RouterLlmClient` |

### Desktop (`src-tauri/`)

The Tauri shell owns the global toggle shortcut, recording lifecycle, model downloads, permissions, local settings, history commands, clipboard paste, and platform integration. The shortcut and UI call the same recording functions so behavior stays consistent.

### Interface (`ui/`)

The Svelte app provides the recording result, model management, history, vocabulary, transcription behavior, LLM configuration, and shortcut settings. A small overlay reflects recording and processing state.

## Storage and network boundaries

PIE stores settings and history in the operating system's application-data directory, vocabulary in local JSON files, and speech/VAD models under `~/.cache/pie/models`.

Audio and local transcription do not leave the machine. Network access occurs for model downloads and when the user invokes a feature backed by their configured LLM endpoint.

## Design constraints

- Core behavior belongs in the library; the CLI and desktop are adapters.
- Heavy or optional dependencies remain feature-gated where practical.
- External services and hardware integrations stay behind traits.
- Direct and Enhanced are the only optimization behaviors; Auto selects between them.
- Legacy mode strings remain accepted by the pipeline and map to automatic selection.
- Tests use fake adapters and never call an external LLM.

See [DEVELOPMENT.md](DEVELOPMENT.md) for build and verification commands.
