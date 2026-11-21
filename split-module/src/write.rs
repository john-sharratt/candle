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
use std::io::ErrorKind;
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

/// `mod_rs` with each `#[path = "…"]` re-based for the folder it now lies in.
/// A `#[path]` in `foo.rs` is read from `foo.rs`'s folder, in `foo/mod.rs`
/// from `foo/`: so the split of a `foo.rs` puts `../` before each path. A
/// `mod.rs` split stays in its folder and keeps its paths.
pub fn rebase_paths(original: &Path, mod_rs: &str) -> String {
    if original.file_name().and_then(|n| n.to_str()) == Some("mod.rs") {
        return mod_rs.to_string();
    }
    mod_rs
        .split_inclusive('\n')
        .map(|line| {
            let trimmed = line.trim_start();
            match trimmed.strip_prefix("#[path = \"") {
                Some(rest) => {
                    let indent = &line[..line.len() - trimmed.len()];
                    format!("{indent}#[path = \"../{rest}")
                }
                None => line.to_string(),
            }
        })
        .collect()
}

/// `text` with each relative `include_str!` / `include_bytes!` path re-based
/// for the folder it now lies in. Those paths are read from the file the
/// macro is in, and every output of a `foo.rs` split lies in `foo/`, one
/// folder down: so each gets `../`. A `mod.rs` split's outputs stay in its
/// folder.
pub fn rebase_includes(original: &Path, text: &str) -> String {
    if original.file_name().and_then(|n| n.to_str()) == Some("mod.rs") {
        return text.to_string();
    }
    let mut out = text.to_string();
    for mac in ["include_str!(\"", "include_bytes!(\""] {
        let mut at = 0;
        while let Some(found) = out[at..].find(mac) {
            let path_at = at + found + mac.len();
            let absolute = out[path_at..].starts_with('/') || out[path_at..].get(1..2) == Some(":");
            if !absolute {
                out.insert_str(path_at, "../");
            }
            at = path_at;
        }
    }
    out
}

/// The module names the split's folder already holds, sorted: a file there
/// other than `mod.rs`, without its `.rs`, or a folder. A new file may not take
/// one. A folder that does not exist yet holds none.
pub fn taken(t: &Targets) -> Result<Vec<String>, String> {
    let entries = match fs::read_dir(&t.dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("{}: {e}", t.dir.display())),
    };
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n != "mod.rs")
        .filter_map(|n| match n.strip_suffix(".rs") {
            Some(stem) => Some(stem.to_string()),
            None if !n.contains('.') => Some(n),
            None => None,
        })
        .collect();
    names.sort();
    names.dedup();
    Ok(names)
}

/// The names `source` brings in or reaches through: every word of its `use`
/// lines and every identifier a path starts from (`resume::unfinished`), not
/// counting `crate`, `self`, `super` and `Self`. A child module given one of
/// them would capture it: `mod resume;` beside `use crate::resume;` is the
/// child, for the moved code and for `mod.rs`.
pub fn used_names(source: &str) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    let words = |text: &str| -> Vec<String> {
        text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .filter(|w| {
                w.chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_lowercase() || c == '_')
            })
            .map(str::to_string)
            .collect()
    };
    for line in source.lines() {
        let l = line.trim_start();
        if let Some(rest) = l
            .strip_prefix("use ")
            .or_else(|| l.strip_prefix("pub use "))
        {
            names.extend(words(rest));
        }
    }
    let bytes = source.as_bytes();
    let mut at = 0;
    while let Some(found) = source[at..].find("::") {
        let end = at + found;
        let start = source[..end]
            .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .map_or(0, |p| p + 1);
        // A path's first segment: not after another `::`, and not a method
        // named with a turbofish (`.collect::<Vec<_>>()`).
        let before = source[..start].trim_end();
        let root = !before.ends_with("::") && !before.ends_with('.');
        if root && start < end && bytes[start].is_ascii_lowercase() {
            names.push(source[start..end].to_string());
        }
        at = end + 2;
    }
    names.retain(|n| {
        !matches!(
            n.as_str(),
            "crate" | "self" | "super" | "use" | "pub" | "as"
        )
    });
    names.sort();
    names.dedup();
    names
}

/// Refuse a split that would write over anything or capture a name: a kept
/// original from an earlier split, a child named for a module the folder
/// already holds, or one named for a name the file uses ([`used_names`]).
pub fn refuse_collisions(t: &Targets, outputs: &Outputs, source: &str) -> Result<(), String> {
    if t.presplit.exists() {
        return Err(format!(
            "{} exists: finish or undo the last split first",
            t.presplit.display()
        ));
    }
    let used = used_names(source);
    // A hoisted module keeps the name its users reach it by.
    let captured: Vec<&str> = outputs
        .children
        .iter()
        .map(|(name, _)| name.as_str())
        .filter(|name| !outputs.hoisted.iter().any(|h| h == name))
        .filter(|name| used.iter().any(|u| u == name))
        .collect();
    if !captured.is_empty() {
        return Err(format!(
            "these names are used by the file, so a module named for one captures it: {}",
            captured.join(", ")
        ));
    }
    let taken = taken(t)?;
    let clashes: Vec<&str> = outputs
        .children
        .iter()
        .map(|(name, _)| name.as_str())
        .filter(|name| taken.iter().any(|t| t == name))
        .collect();
    if !clashes.is_empty() {
        return Err(format!(
            "{} holds these modules already: {}",
            t.dir.display(),
            clashes.join(", ")
        ));
    }
    Ok(())
}

/// Write `outputs` for `original` at `t`. Returns each file written and its
/// size.
pub fn write(
    original: &Path,
    t: &Targets,
    outputs: &Outputs,
    source: &str,
) -> Result<Vec<(PathBuf, usize)>, String> {
    let mut files: Vec<(PathBuf, &str)> = vec![(t.mod_rs.clone(), outputs.mod_rs.as_str())];
    for (name, text) in &outputs.children {
        files.push((t.dir.join(format!("{name}.rs")), text.as_str()));
    }
    refuse_collisions(t, outputs, source)?;
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
            hoisted: Vec::new(),
        };
        let written = write(&original, &t, &out, "fn a() {}\n").unwrap();
        assert_eq!(written.len(), 2);
        assert_eq!(fs::read_to_string(&t.presplit).unwrap(), "fn a() {}\n");
        assert_eq!(fs::read_to_string(&t.mod_rs).unwrap(), "mod a;\n");
        assert_eq!(
            fs::read_to_string(root.join("pool/a.rs")).unwrap(),
            "fn a() {}\n"
        );
        assert!(!original.exists());
        fs::write(&original, "fn b() {}\n").unwrap();
        assert!(write(&original, &t, &out, "fn b() {}\n")
            .unwrap_err()
            .contains("exists"));
        assert_eq!(fs::read_to_string(&original).unwrap(), "fn b() {}\n");
        let _ = fs::remove_dir_all(&root);
    }

    /// **A `#[path]` follows the module file down a folder** when `foo.rs`
    /// becomes `foo/mod.rs`, and stays put when a `mod.rs` is split.
    #[test]
    fn path_attributes_are_rebased_for_the_new_folder() {
        let mod_rs =
            "#[cfg(test)]\n#[path = \"cuda_tests.rs\"]\nmod test;\n    #[path = \"a/b.rs\"]\n";
        assert_eq!(
            rebase_paths(Path::new("src/cuda.rs"), mod_rs),
            "#[cfg(test)]\n#[path = \"../cuda_tests.rs\"]\nmod test;\n    #[path = \"../a/b.rs\"]\n"
        );
        assert_eq!(rebase_paths(Path::new("src/cuda/mod.rs"), mod_rs), mod_rs);
    }

    /// **An included file is found from the file's new folder**: a relative
    /// path gains `../` in a `foo.rs` split, an absolute one is left, and a
    /// `mod.rs` split changes none.
    #[test]
    fn include_paths_are_rebased_for_the_new_folder() {
        let text = "let a = include_str!(\"t.json\");\nlet b = include_bytes!(\"../x.bin\");\n\
                    let c = include_str!(\"/abs\");\nlet d = include_str!(\"C:/abs\");\n";
        assert_eq!(
            rebase_includes(Path::new("src/llama.rs"), text),
            "let a = include_str!(\"../t.json\");\nlet b = include_bytes!(\"../../x.bin\");\n\
             let c = include_str!(\"/abs\");\nlet d = include_str!(\"C:/abs\");\n"
        );
        assert_eq!(rebase_includes(Path::new("src/llama/mod.rs"), text), text);
    }

    /// **The names a file uses** are its `use` lines' words and its paths'
    /// roots, lowercase, without `crate`, `self` or `super`.
    #[test]
    fn used_names_are_use_words_and_path_roots() {
        let src = "use crate::resume::{self, Unfinished};\nuse super::x as y;\n\
                   fn f() { passthrough::go(); a::b::c(); Self::new(); v.collect::<Vec<_>>(); }\n";
        assert_eq!(
            used_names(src),
            vec!["a", "passthrough", "resume", "x", "y"]
        );
    }

    /// **A folder's existing modules are taken** — `foo.rs` beside a `foo/`
    /// of children as much as a `mod.rs` — and a split naming one is refused.
    #[test]
    fn existing_modules_are_taken_and_a_clash_is_refused() {
        let root = std::env::temp_dir().join(format!("split-module-taken-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("pool/diag")).unwrap();
        fs::write(root.join("pool.rs"), "mod lock;\n").unwrap();
        fs::write(root.join("pool/lock.rs"), "").unwrap();
        fs::write(root.join("pool/notes.md"), "").unwrap();
        let t = targets(&root.join("pool.rs")).unwrap();
        assert_eq!(taken(&t).unwrap(), vec!["diag", "lock"]);
        let clash = Outputs {
            mod_rs: "mod lock;\n".into(),
            children: vec![("lock".into(), "".into()), ("rows".into(), "".into())],
            hoisted: Vec::new(),
        };
        assert!(refuse_collisions(&t, &clash, "")
            .unwrap_err()
            .ends_with("holds these modules already: lock"));
        let fine = Outputs {
            mod_rs: "".into(),
            children: vec![("rows".into(), "".into())],
            hoisted: Vec::new(),
        };
        assert!(refuse_collisions(&t, &fine, "").is_ok());
        // A name the file reaches through is captured by a module named for it.
        assert!(
            refuse_collisions(&t, &fine, "use crate::rows;\nfn f() { rows::g() }\n")
                .unwrap_err()
                .ends_with("captures it: rows")
        );
        assert!(taken(&targets(&root.join("none.rs")).unwrap())
            .unwrap()
            .is_empty());
        let _ = fs::remove_dir_all(&root);
    }
}
