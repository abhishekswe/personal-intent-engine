# Using PIE

PIE turns speech into clean text or a structured prompt and pastes it into the app you are already using.

## First run

1. Open **Models** and download Moonshine Base (recommended) or Moonshine Tiny, plus Silero VAD.
2. Allow microphone access when your operating system asks.
3. On macOS, also allow Accessibility access so PIE can paste at your cursor.
4. Put the cursor in any text field and press `Ctrl+Space`.

## Dictating anywhere

The default shortcut is `Ctrl+Space` on macOS and Windows:

1. Press once to begin recording.
2. Speak normally; you do not need to hold any key.
3. Press the shortcut again to stop.
4. PIE transcribes, corrects, and pastes the result into the active text field.

Change the shortcut under **Setup → Dictation shortcut**. If another application already owns a shortcut, PIE keeps the previous working shortcut and shows an error.

You can also start, stop, or cancel a recording from PIE's recording screen.

## Transcription models

PIE currently offers two local Moonshine models:

- **Moonshine Base** — better accuracy and the recommended default.
- **Moonshine Tiny** — smaller and faster, useful on constrained machines.

Silero VAD detects speech and trims silence. Models are downloaded from the **Models** screen and stored locally in `~/.cache/pie/models`.

## Direct, Enhanced, and Auto

- **Direct** keeps clear dictation close to what you said after transcription and vocabulary correction.
- **Enhanced** uses your configured LLM to identify the objective, constraints, and open questions in longer speech.
- **Auto** selects Direct for short commands and Enhanced for longer or question-shaped input.

Enhanced output requires a configured LLM. Direct transcription and correction continue to work without one.

## Personal vocabulary

Use **Lexicon → Your corrections** to teach PIE how a phrase should be written. For example:

```text
next jazz → Next.js
coobernetes → Kubernetes
```

Saved corrections are applied locally to later recordings. PIE also includes a small built-in technical vocabulary. Optional background learning and deep correction use your configured LLM and are off until enabled.

## Optional AI features

Under **LLM provider**, configure OpenAI, OpenRouter, or another OpenAI-compatible endpoint. The provider is used for:

- Enhanced intent extraction.
- Deep correction of unfamiliar transcription errors.
- Background vocabulary learning.
- Explicit **Send to LLM** actions.

Test the connection from the same screen before enabling AI-assisted features.

## History and privacy

Recording history, settings, vocabulary, and downloaded speech models stay on your computer. You can set the history limit, paste a previous result again, delete individual entries, or clear all history.

Audio and local transcription are not sent to an external service. When an AI feature is enabled, the relevant text is sent to the LLM endpoint you configured; review that provider's privacy policy before using it with sensitive content.

PIE has no analytics or telemetry.

## Permissions and troubleshooting

### The shortcut does nothing

- Open PIE from the tray or menu-bar icon and confirm the shortcut under **Setup**.
- Choose a different combination if another application owns `Ctrl+Space`.
- On macOS, verify PIE is enabled under **System Settings → Privacy & Security → Accessibility**.

### Recording does not start

- Confirm microphone permission is enabled for PIE.
- Confirm a Moonshine model and Silero VAD are installed in **Models**.
- Select the correct microphone in the app.

### Text is not pasted

The transcription should still appear in PIE. On macOS, recheck Accessibility permission. On Windows, make sure the destination text field is focused before the second shortcut press.

For installation problems or reproducible bugs, [open an issue](https://github.com/abhishekswe/personal-intent-engine/issues).
