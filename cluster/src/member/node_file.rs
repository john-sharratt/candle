//! A member's node, kept in a file so it is the same member across restarts:
//! drawn at random the first time, and read back after.

use std::fs;
use std::io;
use std::path::Path;

use rand::random;

/// The node kept in `path`, drawn and written there when there is none. Never
/// 0, which no member is.
pub fn kept_node(path: &Path) -> io::Result<u64> {
    if let Ok(text) = fs::read_to_string(path) {
        return u64::from_str_radix(text.trim(), 16).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: not a node: {e}", path.display()),
            )
        });
    }
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let node = loop {
        let n: u64 = random();
        if n != 0 {
            break n;
        }
    };
    fs::write(path, format!("{node:016x}\n"))?;
    Ok(node)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A node is drawn once, written as sixteen hex digits, and read back the
    /// same after — its directory made on the way.
    #[test]
    fn a_node_is_drawn_once_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state").join("node");
        let first = kept_node(&path).unwrap();
        assert_ne!(first, 0);
        assert_eq!(kept_node(&path).unwrap(), first);
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            format!("{first:016x}\n")
        );
    }

    #[test]
    fn a_file_that_is_not_a_node_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("node");
        fs::write(&path, "not hex\n").unwrap();
        let err = kept_node(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert_eq!(fs::read_to_string(&path).unwrap(), "not hex\n");
    }

    #[test]
    fn a_kept_node_is_read_as_written() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("node");
        fs::write(&path, "00000000000000ff\n").unwrap();
        assert_eq!(kept_node(&path).unwrap(), 255);
    }
}
