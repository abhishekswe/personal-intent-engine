//! Curated model catalog and downloads.
//!
//! Removes the manual `curl` step: the Models pane lists a small set of
//! speech-to-text and Silero VAD models, downloads them to
//! ~/.cache/pie/models with progress, and selects one into settings. Custom
//! paths still work — a selected catalog model just writes its path into the
//! same setting.
//!
//! Models come in two shapes:
//! - Single-file (e.g. Silero VAD): one file living directly in
//!   `models_dir()`, keyed by its own filename.
//! - Multi-file (e.g. Moonshine STT): a set of files (ONNX weights +
//!   tokenizer) that live together under `models_dir()/<id>/`. The model is
//!   only "present" once every file in the set exists, and the path handed
//!   to callers is the directory, not any single file.

use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::settings::Settings;

#[derive(Clone, Copy, PartialEq)]
pub enum ModelKind {
    /// Speech-to-text model. Historically Whisper GGML; now Moonshine ONNX
    /// filesets. Kept as `Whisper` to match the existing
    /// `Settings::whisper_model` field name that callers key off of.
    Whisper,
    Vad,
}

impl ModelKind {
    fn as_str(self) -> &'static str {
        match self {
            ModelKind::Whisper => "whisper",
            ModelKind::Vad => "vad",
        }
    }
}

/// One downloadable file within a catalog entry's fileset.
struct FileDef {
    url: &'static str,
    filename: &'static str,
    /// Pinned SHA-256 hex digest, checked after download. `None` means the
    /// hash isn't pinned yet and verification is skipped for that file.
    sha256: Option<&'static str>,
}

struct CatalogEntry {
    id: &'static str,
    name: &'static str,
    description: &'static str,
    kind: ModelKind,
    /// Files that make up this model.
    files: &'static [FileDef],
    /// When true, this entry's files live together under
    /// `models_dir()/<id>/` and the model's canonical path is that
    /// directory. When false, `files` has exactly one entry that lives
    /// directly in `models_dir()`.
    grouped: bool,
    /// Approximate total download size across all files, for the UI.
    size_mb: u32,
}

/// Moonshine ONNX models come from the `moonshine-ai/moonshine` HuggingFace
/// repo (renamed from `UsefulSensors/moonshine`; the old name redirects).
/// Each size ships `encoder_model.onnx` + `decoder_model_merged.onnx` under
/// `onnx/merged/<size>/float/`. The `tiny` variant's `float/` folder does not
/// ship its own `tokenizer.json` upstream (only the two ONNX weight files),
/// so tiny reuses base's `tokenizer.json` — Moonshine sizes share one
/// tokenizer/vocab. The Silero VAD ONNX model is from the official
/// snakers4/silero-vad repo, unchanged from before.
const CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        id: "moonshine-tiny",
        name: "Moonshine Tiny",
        description: "Fastest, lowest accuracy. Good for testing.",
        kind: ModelKind::Whisper,
        files: &[
            FileDef {
                url: "https://huggingface.co/moonshine-ai/moonshine/resolve/main/onnx/merged/tiny/float/encoder_model.onnx",
                filename: "encoder_model.onnx",
                sha256: Some("cbbf580f703b2af2137e0f6d14cd87f31cc67bd858bfd8715403a9489982d1a5"),
            },
            FileDef {
                url: "https://huggingface.co/moonshine-ai/moonshine/resolve/main/onnx/merged/tiny/float/decoder_model_merged.onnx",
                filename: "decoder_model_merged.onnx",
                sha256: Some("4131cef00b62942e9cdef691101f2cc7dbbcd828d71eee8c6c46c28fd051d6cb"),
            },
            FileDef {
                // tiny/float/ doesn't ship its own tokenizer.json upstream;
                // Moonshine tiny and base share the same tokenizer/vocab, so
                // this pulls base's copy (hash matches base's tokenizer.json
                // below byte-for-byte).
                url: "https://huggingface.co/moonshine-ai/moonshine/resolve/main/onnx/merged/base/float/tokenizer.json",
                filename: "tokenizer.json",
                sha256: Some("ad3a2ceb0e84e4da57451d86fa337f8116dcff5f5d106434f8aa0b0de89718b9"),
            },
        ],
        grouped: true,
        size_mb: 113,
    },
    CatalogEntry {
        id: "moonshine-base",
        name: "Moonshine Base",
        description: "Most accurate Moonshine model here; slower.",
        kind: ModelKind::Whisper,
        files: &[
            FileDef {
                url: "https://huggingface.co/moonshine-ai/moonshine/resolve/main/onnx/merged/base/float/encoder_model.onnx",
                filename: "encoder_model.onnx",
                sha256: Some("153e128e7abd64a74ee47f2c3f585c3171c4d46cbb368b032827934c4e01e779"),
            },
            FileDef {
                url: "https://huggingface.co/moonshine-ai/moonshine/resolve/main/onnx/merged/base/float/decoder_model_merged.onnx",
                filename: "decoder_model_merged.onnx",
                sha256: Some("58778763ca8438963190244d6b26572bdca2cedec56a4b91e828f3f2d69ef3c5"),
            },
            FileDef {
                url: "https://huggingface.co/moonshine-ai/moonshine/resolve/main/onnx/merged/base/float/tokenizer.json",
                filename: "tokenizer.json",
                sha256: Some("ad3a2ceb0e84e4da57451d86fa337f8116dcff5f5d106434f8aa0b0de89718b9"),
            },
        ],
        grouped: true,
        size_mb: 251,
    },
    CatalogEntry {
        id: "silero-vad",
        name: "Silero VAD v4",
        description: "Voice activity detection — trims silence.",
        kind: ModelKind::Vad,
        files: &[FileDef {
            url: "https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx",
            filename: "silero_vad_v4.onnx",
            sha256: None,
        }],
        grouped: false,
        size_mb: 2,
    },
];

/// Where downloaded models live. Matches the CLI/engine default cache.
pub fn models_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache/pie/models")
}

fn find(id: &str) -> Option<&'static CatalogEntry> {
    CATALOG.iter().find(|e| e.id == id)
}

/// The directory a model's files live in: `models_dir()/<id>/` for grouped
/// (multi-file) models, `models_dir()` itself for single-file models.
fn entry_root(entry: &CatalogEntry) -> PathBuf {
    if entry.grouped {
        models_dir().join(entry.id)
    } else {
        models_dir()
    }
}

/// The canonical path callers (settings, engine) use for this model: the
/// model's directory when it's a multi-file fileset, or its one file's path
/// otherwise.
fn entry_path(entry: &CatalogEntry) -> PathBuf {
    let root = entry_root(entry);
    if entry.grouped {
        root
    } else {
        // Single-file entries have exactly one FileDef.
        match entry.files.first() {
            Some(file) => root.join(file.filename),
            None => root,
        }
    }
}

/// Whether every file in a model's fileset exists on disk.
fn entry_is_downloaded(entry: &CatalogEntry) -> bool {
    let root = entry_root(entry);
    entry.files.iter().all(|f| root.join(f.filename).exists())
}

/// One catalog row as seen by the frontend.
#[derive(Serialize)]
pub struct ModelInfo {
    id: String,
    name: String,
    description: String,
    kind: String,
    size_mb: u32,
    downloaded: bool,
    selected: bool,
    path: String,
}

pub fn list_models(settings: &Settings) -> Vec<ModelInfo> {
    CATALOG
        .iter()
        .map(|e| {
            let path = entry_path(e);
            let path_str = path.to_string_lossy().into_owned();
            let selected_setting = match e.kind {
                ModelKind::Whisper => &settings.whisper_model,
                ModelKind::Vad => &settings.silero_model,
            };
            ModelInfo {
                id: e.id.to_string(),
                name: e.name.to_string(),
                description: e.description.to_string(),
                kind: e.kind.as_str().to_string(),
                size_mb: e.size_mb,
                downloaded: entry_is_downloaded(e),
                selected: Settings::expand(selected_setting) == path,
                path: path_str,
            }
        })
        .collect()
}

/// Resolve a catalog id to its kind and canonical path (a directory for
/// multi-file models, a file for single-file models).
pub fn resolve(id: &str) -> Option<(ModelKind, PathBuf)> {
    let entry = find(id)?;
    Some((entry.kind, entry_path(entry)))
}

/// Whether a catalog model's full fileset is present on disk.
pub fn is_downloaded(id: &str) -> bool {
    find(id).is_some_and(entry_is_downloaded)
}

/// Delete a catalog model from disk: a single file for single-file models, or
/// the whole per-model directory for multi-file models.
pub fn remove(id: &str) -> std::io::Result<()> {
    let Some(entry) = find(id) else {
        return Ok(());
    };
    if entry.grouped {
        std::fs::remove_dir_all(entry_root(entry))
    } else {
        std::fs::remove_file(entry_path(entry))
    }
}

/// Compute the SHA-256 hex digest of a file, streaming so large model files
/// never need to fit in memory at once.
pub fn sha256_file(path: &Path) -> anyhow::Result<String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Download every file in a catalog model's fileset into its destination
/// directory, verifying each against its pinned SHA-256 (when present) as it
/// lands. On a hash mismatch the bad file is deleted and an error is
/// returned.
///
/// `on_progress` is called with cumulative `(received, total)` bytes across
/// the whole fileset — `total` is an estimate (from the catalog's
/// `size_mb`) since HTTP `Content-Length` for files later in the set isn't
/// known until they start downloading.
///
/// Returns the model's canonical path (the per-model directory for
/// multi-file models, the file itself for single-file models) on success.
pub async fn download_model(
    id: &str,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<PathBuf, String> {
    let entry = find(id).ok_or_else(|| "Unknown model".to_string())?;
    let root = entry_root(entry);
    let approx_total = u64::from(entry.size_mb) * 1_000_000;

    let mut completed: u64 = 0;
    for file in entry.files {
        let dest = root.join(file.filename);
        let base = completed;
        let received = download_to(file.url, &dest, |r, _t| {
            on_progress(base + r, approx_total.max(base + r));
        })
        .await?;
        completed += received;

        if let Some(expected) = file.sha256 {
            let dest_for_hash = dest.clone();
            let actual = tokio::task::spawn_blocking(move || sha256_file(&dest_for_hash))
                .await
                .map_err(|e| format!("Hash check panicked: {e}"))?
                .map_err(|e| format!("Hash check failed for {}: {e}", file.filename))?;
            if actual != expected {
                let _ = std::fs::remove_file(&dest);
                return Err(format!(
                    "SHA-256 mismatch for {}: expected {expected}, got {actual}",
                    file.filename
                ));
            }
        }
    }

    Ok(entry_path(entry))
}

/// Stream `url` to `dest`, calling `on_progress(received, total)` as bytes
/// arrive. Downloads to a sibling `.part` file and renames on success, so an
/// interrupted download never leaves a truncated model that looks complete.
pub async fn download_to(
    url: &str,
    dest: &std::path::Path,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<u64, String> {
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;

    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("Can't create models folder: {e}"))?;
    }

    let response = reqwest::get(url)
        .await
        .map_err(|e| format!("Download failed: {e}"))?
        .error_for_status()
        .map_err(|e| format!("Download failed: {e}"))?;
    let total = response.content_length().unwrap_or(0);

    let tmp = dest.with_extension("part");
    let mut file = tokio::fs::File::create(&tmp)
        .await
        .map_err(|e| format!("Can't write file: {e}"))?;

    let mut stream = response.bytes_stream();
    let mut received: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download interrupted: {e}"))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("Write failed: {e}"))?;
        received += chunk.len() as u64;
        on_progress(received, total);
    }
    file.flush()
        .await
        .map_err(|e| format!("Flush failed: {e}"))?;
    drop(file);
    tokio::fs::rename(&tmp, dest)
        .await
        .map_err(|e| format!("Can't finalize download: {e}"))?;
    Ok(received)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_ids_are_unique_and_resolve() {
        let mut seen = std::collections::HashSet::new();
        for e in CATALOG {
            assert!(seen.insert(e.id), "duplicate catalog id: {}", e.id);
            assert!(resolve(e.id).is_some());
        }
    }

    #[test]
    fn sha256_file_matches_known_digest() {
        let dir = std::env::temp_dir().join("pie-sha256-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("abc.txt");
        std::fs::write(&path, b"abc").unwrap();

        let digest = sha256_file(&path).unwrap();
        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        let _ = std::fs::remove_file(&path);
    }

    // Real network download of the smallest catalog model (~2 MB). Ignored by
    // default so offline `cargo test` stays green; run explicitly to verify:
    //   cargo test -p pie-desktop --ignored download_streams_to_file
    #[test]
    #[ignore]
    fn download_streams_to_file() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let entry = find("silero-vad").unwrap();
            let url = entry.files[0].url;
            let dir = std::env::temp_dir().join("pie-model-test");
            let dest = dir.join("silero_vad_v4.onnx");
            let _ = std::fs::remove_file(&dest);

            let mut last = 0u64;
            let received = download_to(url, &dest, |r, _t| last = r).await.unwrap();

            assert!(received > 1_000_000, "expected >1MB, got {received}");
            assert_eq!(last, received, "final progress must equal total received");
            let on_disk = std::fs::metadata(&dest).unwrap().len();
            assert_eq!(on_disk, received, "file size must match downloaded bytes");
            let _ = std::fs::remove_file(&dest);
        });
    }
}
