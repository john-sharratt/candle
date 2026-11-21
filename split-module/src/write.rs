//! Where a split's files go, and putting them there without ever truncating
//! one.
//!
//! `foo.rs` becomes `foo/mod.rs` and `foo/<child>.rs`; a `mod.rs` is rewritten
//! in its own folder, its children beside it. The original is renamed to
//! `<name>.presplit` — kept, untouched, until the split is done and the build
//! repaired — never overwritten. Every new file is written whole under a
//! temporary name first and renamed into place only once all of them are, so
//! a split that stops part-way leaves the original as it was.

use std::fs;
use std::path::{Path, PathBuf};

use crate::assemble::Outputs;

/// The files a split of `original` writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Targets {
    pub dir: PathBuf,
    pub mod_rs: PathBuf,
    /// Where the original is kept.
    pub presplit: PathBuf,
}

/// Where a split of `original` writes.
pub fn targets(original: &Path) -> Result<Targets, String> {
    let name = original
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or("the file has no name")?;
    if name == "lib.rs" || name == "main.rs" {
        return Err("a crate root is not split: split a module of it".into());
    }
    let presplit = original.with_file_name(format!("{name}.presplit"));
    if name == "mod.rs" {
        let dir = original
            .parent()
            .ok_or("mod.rs has no folder")?
            .to_path_buf();
        return Ok(Targets {
            mod_rs: original.to_path_buf(),
            dir,
            presplit,
        });
    }
    let stem = name.strip_suffix(".rs").ok_or("not a .rs file")?;
    let dir = original.with_file_name(stem);
    Ok(Targets {
        mod_rs: dir.join("mod.rs"),
        dir,
        presplit,
    })
}

/// Write `outputs` for `original` at `t`. Returns each file written and its
/// size.
pub fn write(
    original: &Path,
    t: &Targets,
    outputs: &Outputs,
) -> Result<Vec<(PathBuf, usize)>, String> {
    let mut files: Vec<(PathBuf, &str)> = vec![(t.mod_rs.clone(), outputs.mod_rs.as_str())];
    for (name, text) in &outputs.children {
        files.push((t.dir.join(format!("{name}.rs")), text.as_str()));
    }
    if t.presplit.exists() {
        return Err(format!(
            "{} exists: finish or undo the last split first",
            t.presplit.display()
        ));
    }
    for (path, _) in &files {
        if path.exists() && path != original {
            return Err(format!("{} exists already", path.display()));
        }
    }
    fs::create_dir_all(&t.dir).map_err(|e| format!("{}: {e}", t.dir.display()))?;
    let tmp = |p: &Path| p.with_extension("rs.split-tmp");
    for (path, text) in &files {
        fs::write(tmp(path), text).map_err(|e| format!("{}: {e}", tmp(path).display()))?;
    }
    fs::rename(original, &t.presplit).map_err(|e| format!("keeping the original: {e}"))?;
    let mut written = Vec::with_capacity(files.len());
    for (path, text) in &files {
        fs::rename(tmp(path), path).map_err(|e| format!("{}: {e}", path.display()))?;
        written.push((path.clone(), text.len()));
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_becomes_a_folder_and_a_mod_rs_stays() {
        assert_eq!(
            targets(Path::new("src/pool.rs")).unwrap(),
            Targets {
                dir: PathBuf::from("src/pool"),
                mod_rs: PathBuf::from("src/pool/mod.rs"),
                presplit: PathBuf::from("src/pool.rs.presplit"),
            }
        );
        assert_eq!(
            targets(Path::new("src/sched/mod.rs")).unwrap(),
            Targets {
                dir: PathBuf::from("src/sched"),
                mod_rs: PathBuf::from("src/sched/mod.rs"),
                presplit: PathBuf::from("src/sched/mod.rs.presplit"),
            }
        );
        assert!(targets(Path::new("src/lib.rs")).is_err());
    }

    /// **The original is kept and every file lands**; a second split over a
    /// kept original, or onto a file that exists, is refused before anything
    /// is written.
    #[test]
    fn the_original_is_kept_and_nothing_is_overwritten() {
        let root = std::env::temp_dir().join(format!("split-module-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let original = root.join("pool.rs");
        fs::write(&original, "fn a() {}\n").unwrap();
        let t = targets(&original).unwrap();
        let out = Outputs {
            mod_rs: "mod a;\n".into(),
            children: vec![("a".into(), "fn a() {}\n".into())],
        };
        let written = write(&original, &t, &out).unwrap();
        assert_eq!(written.len(), 2);
        assert_eq!(fs::read_to_string(&t.presplit).unwrap(), "fn a() {}\n");
        assert_eq!(fs::read_to_string(&t.mod_rs).unwrap(), "mod a;\n");
        assert_eq!(
            fs::read_to_string(root.join("pool/a.rs")).unwrap(),
            "fn a() {}\n"
        );
        assert!(!original.exists());
        fs::write(&original, "fn b() {}\n").unwrap();
        assert!(write(&original, &t, &out).unwrap_err().contains("exists"));
        assert_eq!(fs::read_to_string(&original).unwrap(), "fn b() {}\n");
        let _ = fs::remove_dir_all(&root);
    }
}
