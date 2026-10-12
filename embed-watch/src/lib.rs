//! What a build script prints so a crate that embeds a directory with
//! `include_dir!` is rebuilt when that directory changes.
//!
//! `include_dir!` reads the directory at compile time and tells cargo nothing
//! about it, so editing a file under it alone does not recompile the crate — the
//! binary keeps serving the copy it was built with. A `rerun-if-changed` for the
//! directory itself catches a file added or removed; one per file catches an
//! edit.

use std::fs::read_dir;
use std::path::{Path, PathBuf};

/// Print a `rerun-if-changed` for `dir` and everything under it.
pub fn watch(dir: impl AsRef<Path>) {
    for path in watched(dir.as_ref()) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

/// `dir`, then every directory and file under it, depth first.
pub fn watched(dir: &Path) -> Vec<PathBuf> {
    let mut out = vec![dir.to_path_buf()];
    walk(dir, &mut out);
    out
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        out.push(path.clone());
        if path.is_dir() {
            walk(&path, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs::{create_dir_all, write};

    use tempfile::TempDir;

    /// The directory, its subdirectories and every file, in order.
    #[test]
    fn every_directory_and_file_is_watched() {
        let t = TempDir::new().unwrap();
        create_dir_all(t.path().join("lib")).unwrap();
        write(t.path().join("index.html"), "").unwrap();
        write(t.path().join("lib").join("a.js"), "").unwrap();
        assert_eq!(
            watched(t.path()),
            vec![
                t.path().to_path_buf(),
                t.path().join("index.html"),
                t.path().join("lib"),
                t.path().join("lib").join("a.js"),
            ]
        );
    }
}
