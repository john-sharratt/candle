//! Uploads kept before conversations held them (`docs/replicated_substrate.md`
//! §10.2, §15.1 step 2): `substrate/conv-files/`, an `index.json` naming each
//! file's conversation and one blob per file — text verbatim, a binary file
//! hex-encoded. Read once at startup, each file is put in its conversation's
//! uploads store, and the folder is removed.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use candle_conversation::persistence::SUBSTRATE_DIR;
use serde::Deserialize;

use crate::conv_files::{decode_from_storage, kind_for};

/// One file as the old index recorded it.
#[derive(Debug, Clone, Deserialize)]
struct Record {
    id: u64,
    conv: String,
    name: String,
}

#[derive(Debug, Default, Deserialize)]
struct Index {
    files: Vec<Record>,
}

/// One old upload: its old id, its name, its bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Legacy {
    pub id: u64,
    pub name: String,
    pub bytes: Vec<u8>,
}

/// The folder the old uploads were kept in.
pub fn folder(workspace: &Path) -> PathBuf {
    workspace.join(SUBSTRATE_DIR).join("conv-files")
}

/// Every old upload under `workspace`, by conversation, in the order the
/// index lists them; empty when there is no folder. A file whose blob is
/// missing is left out.
pub fn read(workspace: &Path) -> io::Result<BTreeMap<String, Vec<Legacy>>> {
    let root = folder(workspace);
    let index = match fs::read(root.join("index.json")) {
        Ok(bytes) => serde_json::from_slice::<Index>(&bytes).map_err(io::Error::other)?,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(e) => return Err(e),
    };
    let mut by_conv: BTreeMap<String, Vec<Legacy>> = BTreeMap::new();
    for r in index.files {
        let Ok(stored) = fs::read_to_string(root.join("blobs").join(r.id.to_string())) else {
            tracing::warn!(id = r.id, name = %r.name, "an old upload's blob is missing");
            continue;
        };
        let bytes = decode_from_storage(&stored, kind_for(&r.name));
        by_conv.entry(r.conv).or_default().push(Legacy {
            id: r.id,
            name: r.name,
            bytes,
        });
    }
    Ok(by_conv)
}

/// Remove the old folder, once every file in it is held elsewhere.
pub fn remove(workspace: &Path) -> io::Result<()> {
    match fs::remove_dir_all(folder(workspace)) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv_files::encode_for_storage;

    /// The old index and blobs read back byte for byte, by conversation, a
    /// binary file decoded from its hex; then the folder goes.
    #[test]
    fn old_uploads_read_back_by_conversation() {
        let ws = tempfile::tempdir().unwrap();
        let root = folder(ws.path());
        fs::create_dir_all(root.join("blobs")).unwrap();
        let png = vec![0x89, 0x50, 0x4e, 0x47, 0xff, 0x00];
        fs::write(
            root.join("blobs").join("1"),
            encode_for_storage(b"fn main() {}\n", kind_for("main.rs")),
        )
        .unwrap();
        fs::write(
            root.join("blobs").join("2"),
            encode_for_storage(&png, kind_for("logo.png")),
        )
        .unwrap();
        fs::write(
            root.join("index.json"),
            r#"{"seq":3,"files":[
                {"id":1,"conv":"c1","name":"main.rs","ext":"RS","kind":"code","size_bytes":13,"created_ms":5},
                {"id":2,"conv":"c1","name":"logo.png","ext":"PNG","kind":"img","size_bytes":6,"created_ms":6},
                {"id":3,"conv":"c2","name":"gone.txt","ext":"TXT","kind":"doc","size_bytes":1,"created_ms":7}
            ]}"#,
        )
        .unwrap();
        let read = read(ws.path()).unwrap();
        assert_eq!(
            read,
            BTreeMap::from([(
                "c1".to_string(),
                vec![
                    Legacy {
                        id: 1,
                        name: "main.rs".into(),
                        bytes: b"fn main() {}\n".to_vec(),
                    },
                    Legacy {
                        id: 2,
                        name: "logo.png".into(),
                        bytes: png,
                    },
                ],
            )])
        );
        remove(ws.path()).unwrap();
        assert!(!root.exists());
        assert!(super::read(ws.path()).unwrap().is_empty());
        remove(ws.path()).unwrap();
    }
}
