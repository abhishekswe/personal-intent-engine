# PIE README Redesign

## Goal

Make the repository front page useful to people evaluating, installing, and using PIE. Keep it short enough to scan while preserving required attribution, privacy disclosure, and links to deeper technical material.

## Audience

The primary reader is an end user who wants to understand PIE, install it on macOS or Windows, and complete their first dictation. Contributors and library users are secondary audiences served by linked documents.

## README structure

1. Product identity: logo, plain-language description, essential badges.
2. Demo placeholder for a future short video or GIF.
3. Four concise product benefits.
4. A four-step description of the user workflow.
5. Installation instructions for macOS and Windows.
6. First-use steps, including the configurable `Ctrl+Space` default shortcut.
7. A brief, accurate privacy statement.
8. Links to usage, development, architecture, release, and signing documentation.
9. Required acknowledgements, NOTICE link, license, and author links.

The target is approximately 90–120 lines. The README will avoid deep implementation details, exhaustive settings tables, API examples, and contributor build instructions.

## Documentation split

- `docs/USAGE.md`: everyday operation, settings, vocabulary correction, optional AI enhancement, history, models, permissions, and common troubleshooting.
- `docs/DEVELOPMENT.md`: source setup, desktop build commands, CLI and Rust library examples, repository structure, and verification commands.
- `docs/ARCHITECTURE.md`: remains the technical architecture reference.
- `docs/signing.md` and `docs/RELEASING.md`: remain the signing and release references.

The repository's default-deny documentation ignore rules will explicitly allow the two new public documents.

## Accuracy requirements

- Describe the current local STT implementation without claiming Whisper is the only engine.
- Describe only the current `Direct` and `Enhanced` optimization behavior.
- Use the current configurable toggle shortcut, defaulting to `Ctrl+Space`.
- Do not claim Windows has been manually tested unless repository evidence supports it.
- State clearly that audio/transcription stay local while configured external LLM features may send text to that provider.
- Preserve `README.md` acknowledgements and the `NOTICE` reference as required by project policy.

## Validation

- Check every relative README link against a real tracked file.
- Search the rewritten public documentation for removed shortcut and optimizer-mode claims.
- Confirm installation URLs match the repository owner and current release process.
- Review the final rendered hierarchy for a clear install-to-first-use path.
