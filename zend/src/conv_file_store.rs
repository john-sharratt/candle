//! Persistent conversation-files store (docs/zend_ui_redesign.md §2.5).
//!
//! Backs the GUI files pane: upload / list / get / delete of files attached to a
//! conversation. Stored under `<workspace>/.substrate/conv-files/` — a metadata
//! index (`index.json`) plus one blob per file — so it survives restarts and is
//! available **without the model loaded** (the store is independent of the
//! inference engine, so the routes and the harness exercise it model-less).
//!
//! Content is encoded via [`crate::conv_files`] (text verbatim, binaries as hex)
//! so every file kind round-trips byte-exact. Admitting a file into the model's
//! projection on reference is the engine-backed tier layered on top of this
//! store; the store itself is the durable source of the bytes.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::conv_files::{decode_from_storage, encode_for_storage, ext_badge, fmt_bytes, kind_for};

/// One stored file's durable record (persisted in `index.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileRecord {
    id: u64,
    conv: String,
    name: String,
    ext: String,
    kind: String,
    size_bytes: u64,
    created_ms: u64,
}

/// File metadata as the API/GUI consumes it (`size`/`added` are display strings).
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FileMeta {
    pub id: u64,
    pub name: String,
    pub ext: String,
    pub kind: String,
    pub size: String,
    pub added: String,
}

#[derive(Default, Serialize, Deserialize)]
struct Index {
    seq: u64,
    files: Vec<FileRecord>,
}

pub struct ConvFileStore {
    root: PathBuf,
    inner: Mutex<Index>,
}

impl ConvFileStore {
    /// Open (or create) the store under `<workspace>/.substrate/conv-files`.
    pub fn open(workspace: &Path) -> Self {
        let root = workspace.join(".substrate").join("conv-files");
        let inner = load_index(&root).unwrap_or_default();
        Self {
            root,
            inner: Mutex::new(inner),
        }
    }

    fn blob_path(&self, id: u64) -> PathBuf {
        self.root.join("blobs").join(id.to_string())
    }

    /// Store `bytes` as a new file on `conv`; returns its metadata.
    ///
    /// The blob write happens **outside** the index lock. Only the id
    /// reservation and the record commit need exclusion; writing the payload
    /// under the lock stalled every other store operation — `list`, metadata,
    /// other uploads — for the full duration of the disk write, which for a
    /// large upload is the whole transfer.
    ///
    /// Failure leaves the store consistent: a failed blob write consumes an id
    /// (a harmless gap in a monotonic counter) but commits no record, so nothing
    /// references the missing blob.
    pub fn upload(&self, conv: &str, name: &str, bytes: &[u8]) -> std::io::Result<FileMeta> {
        let kind = kind_for(name);
        let stored = encode_for_storage(bytes, kind);

        // Reserve the id, then release the lock for the slow part.
        let id = {
            let mut guard = self.inner.lock().unwrap();
            guard.seq += 1;
            guard.seq
        };

        fs::create_dir_all(self.root.join("blobs"))?;
        fs::write(self.blob_path(id), stored.as_bytes())?;

        let rec = FileRecord {
            id,
            conv: conv.to_string(),
            name: name.to_string(),
            ext: ext_badge(name),
            kind: kind.as_str().to_string(),
            size_bytes: bytes.len() as u64,
            created_ms: now_ms(),
        };
        // Re-take only to commit. `save_index` stays inside: it serialises the
        // whole index, so it needs a consistent snapshot, and it writes orders of
        // magnitude less than the blob.
        let mut guard = self.inner.lock().unwrap();
        guard.files.push(rec.clone());
        save_index(&self.root, &guard)?;
        Ok(meta_of(&rec))
    }

    /// Metadata for every file `conv` references, newest first.
    pub fn list(&self, conv: &str) -> Vec<FileMeta> {
        let guard = self.inner.lock().unwrap();
        let mut out: Vec<&FileRecord> = guard.files.iter().filter(|r| r.conv == conv).collect();
        out.sort_by(|a, b| b.created_ms.cmp(&a.created_ms));
        out.into_iter().map(meta_of).collect()
    }

    /// Reconstruct one file's original bytes.
    pub fn get_content(&self, conv: &str, id: u64) -> Option<Vec<u8>> {
        let kind = {
            let guard = self.inner.lock().unwrap();
            let rec = guard.files.iter().find(|r| r.conv == conv && r.id == id)?;
            kind_for(&rec.name)
        };
        let stored = fs::read_to_string(self.blob_path(id)).ok()?;
        Some(decode_from_storage(&stored, kind))
    }

    /// Drop a file's reference + blob. Returns whether it existed.
    pub fn delete(&self, conv: &str, id: u64) -> bool {
        let mut guard = self.inner.lock().unwrap();
        let before = guard.files.len();
        guard.files.retain(|r| !(r.conv == conv && r.id == id));
        let removed = guard.files.len() != before;
        if removed {
            let _ = fs::remove_file(self.blob_path(id));
            let _ = save_index(&self.root, &guard);
        }
        removed
    }
}

fn meta_of(rec: &FileRecord) -> FileMeta {
    FileMeta {
        id: rec.id,
        name: rec.name.clone(),
        ext: rec.ext.clone(),
        kind: rec.kind.clone(),
        size: fmt_bytes(rec.size_bytes),
        added: added_label(rec.created_ms, now_ms()),
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Relative "added" label, matching the GUI's vocabulary.
fn added_label(created_ms: u64, now: u64) -> String {
    let secs = now.saturating_sub(created_ms) / 1000;
    if secs < 60 {
        "just now".to_string()
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86_400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86_400)
    }
}

fn index_path(root: &Path) -> PathBuf {
    root.join("index.json")
}
fn load_index(root: &Path) -> Option<Index> {
    let data = fs::read(index_path(root)).ok()?;
    serde_json::from_slice(&data).ok()
}
fn save_index(root: &Path, index: &Index) -> std::io::Result<()> {
    fs::create_dir_all(root)?;
    let data = serde_json::to_vec_pretty(index).unwrap_or_default();
    fs::write(index_path(root), data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (ConvFileStore, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        (ConvFileStore::open(tmp.path()), tmp)
    }

    #[test]
    fn upload_list_get_delete_roundtrip() {
        let (s, _t) = store();
        let bytes = b"fn main() { println!(\"hi\"); }\n";
        let meta = s.upload("c1", "main.rs", bytes).unwrap();
        assert_eq!(meta.name, "main.rs");
        assert_eq!(meta.ext, "RS");
        assert_eq!(meta.kind, "code");

        let list = s.list("c1");
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, meta.id);

        // byte-exact reconstruction
        assert_eq!(s.get_content("c1", meta.id).unwrap(), bytes);

        assert!(s.delete("c1", meta.id));
        assert!(s.list("c1").is_empty());
        assert!(s.get_content("c1", meta.id).is_none());
    }

    #[test]
    fn binary_upload_reconstructs_byte_exact() {
        let (s, _t) = store();
        let bytes: Vec<u8> = vec![0x89, 0x50, 0x4e, 0x47, 0xff, 0x00, 0x0d];
        let meta = s.upload("c1", "logo.png", &bytes).unwrap();
        assert_eq!(meta.kind, "img");
        assert_eq!(s.get_content("c1", meta.id).unwrap(), bytes);
    }

    /// `upload` releases the index lock across the blob write, so concurrent
    /// uploads genuinely interleave. Each must still reserve a **distinct** id
    /// and land a byte-exact blob under it — id reservation is the only part
    /// that needs exclusion, and this pins that it actually does.
    #[test]
    fn concurrent_uploads_reserve_distinct_ids_and_keep_blobs_intact() {
        let (s, _t) = store();
        let s = std::sync::Arc::new(s);
        const N: usize = 16;
        let handles: Vec<_> = (0..N)
            .map(|i| {
                let s = s.clone();
                std::thread::spawn(move || {
                    // Large enough that the writes overlap in practice.
                    let body = vec![i as u8; 64 * 1024];
                    let m = s.upload("c1", &format!("f{i}.bin"), &body).unwrap();
                    (m.id, body)
                })
            })
            .collect();
        let done: Vec<(u64, Vec<u8>)> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        let mut ids: Vec<u64> = done.iter().map(|(id, _)| *id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), N, "every concurrent upload needs its own id");

        assert_eq!(s.list("c1").len(), N, "every record committed");
        for (id, body) in &done {
            assert_eq!(
                s.get_content("c1", *id).as_ref(),
                Some(body),
                "blob {id} did not survive concurrent upload"
            );
        }
    }

    #[test]
    fn files_are_scoped_per_conversation() {
        let (s, _t) = store();
        s.upload("a", "a.txt", b"a").unwrap();
        s.upload("b", "b.txt", b"b").unwrap();
        assert_eq!(s.list("a").len(), 1);
        assert_eq!(s.list("b").len(), 1);
        assert_eq!(s.list("a")[0].name, "a.txt");
    }

    #[test]
    fn survives_reopen() {
        let tmp = tempfile::tempdir().unwrap();
        let id = {
            let s = ConvFileStore::open(tmp.path());
            s.upload("c1", "note.md", b"# hi").unwrap().id
        };
        // reopen from disk
        let s2 = ConvFileStore::open(tmp.path());
        let list = s2.list("c1");
        assert_eq!(list.len(), 1);
        assert_eq!(s2.get_content("c1", id).unwrap(), b"# hi");
    }
}
