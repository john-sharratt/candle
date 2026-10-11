//! A substrate's node: the origin of every record it writes.
//!
//! 64 random bits minted when the substrate is first opened for writing, and
//! kept in `substrate/NODE` so they survive restarts (§3.7). Every record the
//! substrate writes names the node as its `origin`, and numbers itself in the
//! node's own sequence.
//!
//! **A copied substrate must not keep its node.** Two directories issuing
//! `oseq` 1, 2, 3… under one origin would each have the other's records dropped
//! as duplicates wherever both arrive. So the file stores, beside the node, a
//! fingerprint of where it was minted — the operating system install and the
//! directory's canonical path — and an open that finds a different fingerprint
//! mints a fresh node. Records the copy already holds keep the origin they
//! carry; nothing writes under the old node from the copy again.

use std::fs;
use std::io::{self, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::machine_id::machine_id;

/// The file the node is kept in, inside the substrate directory.
pub const NODE_FILE: &str = "NODE";

/// How a node came to be the one an open returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeSource {
    /// Read back from the file, minted here.
    Kept,
    /// The directory had none, and one was minted.
    Minted,
    /// The file was minted somewhere else — the directory was copied or moved
    /// — so a fresh node replaced `previous`.
    Reminted { previous: u64 },
}

#[derive(Serialize, Deserialize)]
struct NodeRecord {
    node: String,
    fingerprint: String,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The fingerprint of `dir` on this machine: SHA-256 over `tokera-node v1\0`,
/// the install identifier, `\0`, and the directory's canonical path.
pub fn fingerprint(dir: &Path) -> io::Result<[u8; 32]> {
    let canonical = fs::canonicalize(dir)?;
    let mut h = Sha256::new();
    h.update(b"tokera-node v1\0");
    h.update(machine_id()?.as_bytes());
    h.update(b"\0");
    h.update(canonical.to_string_lossy().as_bytes());
    Ok(h.finalize().into())
}

/// 64 random bits, never zero: zero is the origin of a record with no
/// identity.
fn fresh_node() -> u64 {
    loop {
        let n: u64 = rand::random();
        if n != 0 {
            return n;
        }
    }
}

fn read_record(dir: &Path) -> io::Result<Option<NodeRecord>> {
    match fs::read_to_string(dir.join(NODE_FILE)) {
        Ok(text) => serde_json::from_str(&text)
            .map(Some)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{NODE_FILE}: {e}"))),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

fn parse_node(text: &str) -> io::Result<u64> {
    u64::from_str_radix(text, 16)
        .ok()
        .filter(|&n| n != 0)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{NODE_FILE} names node {text:?}, which is not a nonzero 16-hex-digit id"),
            )
        })
}

/// Write the file through a temporary sibling and a rename, so a crash leaves
/// the old file or the new one and never a torn one.
fn write_record(dir: &Path, node: u64, fingerprint: &[u8; 32]) -> io::Result<()> {
    let record = NodeRecord {
        node: format!("{node:016x}"),
        fingerprint: hex(fingerprint),
    };
    let text = serde_json::to_string(&record).expect("a node record serializes");
    let tmp = dir.join(format!("{NODE_FILE}.tmp"));
    // Synced through the handle that wrote it: Windows refuses to flush a
    // handle opened for reading.
    let mut file = fs::File::create(&tmp)?;
    file.write_all(text.as_bytes())?;
    file.sync_all()?;
    drop(file);
    fs::rename(&tmp, dir.join(NODE_FILE))
}

/// The node of the substrate in `dir`, minting one when the directory has none
/// or was minted somewhere else. For a writable open.
pub fn load_or_mint(dir: &Path) -> io::Result<(u64, NodeSource)> {
    let here = fingerprint(dir)?;
    match read_record(dir)? {
        Some(r) if r.fingerprint == hex(&here) => Ok((parse_node(&r.node)?, NodeSource::Kept)),
        Some(r) => {
            let previous = parse_node(&r.node)?;
            let node = fresh_node();
            write_record(dir, node, &here)?;
            Ok((node, NodeSource::Reminted { previous }))
        }
        None => {
            let node = fresh_node();
            write_record(dir, node, &here)?;
            Ok((node, NodeSource::Minted))
        }
    }
}

/// The node the file in `dir` names, wherever it was minted, or `None` when
/// there is no file. For a read-only open, which writes nothing and needs the
/// node only to know which records are its own.
pub fn read(dir: &Path) -> io::Result<Option<u64>> {
    read_record(dir)?.map(|r| parse_node(&r.node)).transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_directory_mints_a_node_and_keeps_it() {
        let dir = tempfile::tempdir().unwrap();
        let (node, source) = load_or_mint(dir.path()).unwrap();
        assert_eq!(source, NodeSource::Minted);
        assert_ne!(node, 0);
        assert_eq!(load_or_mint(dir.path()).unwrap(), (node, NodeSource::Kept));
        assert_eq!(read(dir.path()).unwrap(), Some(node));
    }

    /// A directory copied elsewhere carries the original's file, and the copy
    /// mints a node of its own on first open; the original keeps its node.
    #[test]
    fn a_copied_directory_mints_a_fresh_node() {
        let root = tempfile::tempdir().unwrap();
        let original = root.path().join("a");
        let copy = root.path().join("b");
        fs::create_dir(&original).unwrap();
        fs::create_dir(&copy).unwrap();
        let (node, _) = load_or_mint(&original).unwrap();
        fs::copy(original.join(NODE_FILE), copy.join(NODE_FILE)).unwrap();

        let (copy_node, source) = load_or_mint(&copy).unwrap();
        assert_eq!(source, NodeSource::Reminted { previous: node });
        assert_ne!(copy_node, node);
        assert_eq!(load_or_mint(&copy).unwrap(), (copy_node, NodeSource::Kept));
        assert_eq!(load_or_mint(&original).unwrap(), (node, NodeSource::Kept));
    }

    /// The file is written whole: no temporary is left beside it.
    #[test]
    fn the_file_is_replaced_whole() {
        let dir = tempfile::tempdir().unwrap();
        load_or_mint(dir.path()).unwrap();
        let names: Vec<String> = fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec![NODE_FILE.to_string()]);
    }

    #[test]
    fn a_directory_with_no_file_has_no_node_to_read() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read(dir.path()).unwrap(), None);
    }

    /// A file naming node zero is refused, not taken as a node: zero is the
    /// origin of a record with no identity.
    #[test]
    fn a_zero_node_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let here = fingerprint(dir.path()).unwrap();
        fs::write(
            dir.path().join(NODE_FILE),
            format!(
                r#"{{"node":"0000000000000000","fingerprint":"{}"}}"#,
                hex(&here)
            ),
        )
        .unwrap();
        assert_eq!(
            load_or_mint(dir.path()).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }
}
