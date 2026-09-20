<p align="center">
  <img src="assets/icon.png" alt="PIE logo" width="112" height="112" />
</p>

<h1 align="center">PIE — Personal Intent Engine</h1>

<p align="center">
  An open-source desktop app that turns natural speech into clear text or a structured prompt and pastes it at your cursor.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="Apache 2.0 license"></a>
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Windows-lightgrey.svg" alt="macOS and Windows">
  <img src="https://img.shields.io/badge/transcription-local-2f855a.svg" alt="Local transcription">
</p>

## Demo

A short product walkthrough will be added here. Until then, the four-step flow below shows the complete interaction.

<!-- Replace the sentence above with a short product GIF or video when available. -->

## Why PIE

- **Dictate anywhere** — press one shortcut to start and press it again to finish and paste.
- **Keep audio private** — speech recognition runs locally on your computer.
- **Speak naturally** — PIE removes filler, preserves your intent, and can structure longer thoughts.
- **Teach your vocabulary** — save corrections for names, tools, and technical terms that transcription usually gets wrong.

## How it works

```text
Press shortcut → Speak → Transcribe and correct locally → Paste at your cursor
```

Use **Direct** for faithful voice-to-text. Use **Enhanced** when you want a configured LLM to turn longer, rambling speech into a structured prompt.

## Install

PIE is currently distributed as a prerelease for Apple Silicon Macs and Windows.

### macOS

```bash
curl -fsSL https://raw.githubusercontent.com/abhishekswe/personal-intent-engine/main/scripts/install.sh | bash
```

The script downloads the latest release, installs PIE in `/Applications`, and removes the quarantine attribute required for this self-signed build. To inspect it first, [read the script](scripts/install.sh) or download it before running.

You can also download the Apple Silicon `.dmg` from [Releases](https://github.com/abhishekswe/personal-intent-engine/releases). Move PIE to Applications, then run:

```bash
xattr -cr /Applications/PIE.app
```

### Windows

Download the latest `.exe` installer from [Releases](https://github.com/abhishekswe/personal-intent-engine/releases). If SmartScreen appears, choose **More info → Run anyway**.

## First use

1. Open **Models** and download Moonshine Base (recommended) or Moonshine Tiny, plus Silero VAD.
2. Grant microphone permission. On macOS, also grant Accessibility permission so PIE can paste.
3. Put your cursor in any text field and press `Ctrl+Space`.
4. Speak, then press `Ctrl+Space` again to transcribe and paste.

The shortcut is configurable under **Setup**. See the [usage guide](docs/USAGE.md) for models, vocabulary, AI enhancement, history, and troubleshooting.

## Privacy

Audio, transcription, vocabulary, settings, and history stay on your computer. PIE has no analytics or telemetry.

Optional AI features send relevant text—not audio—to the OpenAI-compatible endpoint you configure. They remain off until you configure or enable them.

## Documentation

- [Using PIE](docs/USAGE.md)
- [Developing and building from source](docs/DEVELOPMENT.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Release process](docs/RELEASING.md)
- [macOS signing](docs/signing.md)

## Acknowledgements

PIE's audio-capture and desktop layers include work derived from [Handy](https://github.com/cjpais/handy) and [OpenSuperWhisper](https://github.com/starmel/OpenSuperWhisper). Their MIT copyright notices are retained in [NOTICE](NOTICE).

## License

PIE is distributed under the [Apache 2.0 License](LICENSE). Derived portions retain their MIT notices as recorded in [NOTICE](NOTICE).

Built by **Abhishek Kumar** · [GitHub](https://github.com/abhishekswe) · [Website](https://abhishekswe.github.io) · [LinkedIn](https://www.linkedin.com/in/abhishekkr-swe/)
