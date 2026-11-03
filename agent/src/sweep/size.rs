//! How many bytes a folder's files hold, counted from their metadata.

use std::fs;
use std::path::Path;

/// The bytes every file under `dir` holds; 0 for a folder that is not there.
/// A file or folder that cannot be read is not counted.
pub fn dir_bytes(dir: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => dir_bytes(&e.path()),
            Ok(t) if t.is_file() => e.metadata().map_or(0, |m| m.len()),
            _ => 0,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_folders_bytes_are_its_files_at_every_depth() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("a/b")).unwrap();
        fs::write(root.path().join("one"), [0u8; 10]).unwrap();
        fs::write(root.path().join("a/b/two"), [0u8; 32]).unwrap();
        assert_eq!(dir_bytes(root.path()), 42);
        assert_eq!(dir_bytes(&root.path().join("missing")), 0);
    }
}
