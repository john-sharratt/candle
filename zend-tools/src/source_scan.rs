//! Test support: find tool source that names a forbidden primitive.
//!
//! Backs the tests that hold every tool to the capability-checked primitives in
//! [`crate::net`], [`crate::exec`], [`crate::disk`] and the file store. Comment
//! lines are skipped, so a doc comment explaining why a primitive is not used
//! does not trip it.

use std::fs;
use std::path::{Path, PathBuf};

/// `(file, line number, line)` for every non-comment line under `src/tools`
/// containing one of `needles`.
pub fn tool_sources_containing(needles: &[&str]) -> Vec<(String, usize, String)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("tools");
    let mut files = Vec::new();
    collect_rs(&root, &mut files);
    assert!(
        files.len() > 50,
        "only {} tool sources found under {}",
        files.len(),
        root.display()
    );
    let mut out = Vec::new();
    for file in files {
        let text = fs::read_to_string(&file).expect("tool source reads");
        for (i, line) in text.lines().enumerate() {
            let code = line.trim_start();
            if code.starts_with("//") {
                continue;
            }
            if needles.iter().any(|n| code.contains(n)) {
                out.push((
                    file.strip_prefix(&root)
                        .unwrap_or(&file)
                        .display()
                        .to_string(),
                    i + 1,
                    code.to_string(),
                ));
            }
        }
    }
    out
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("tools dir reads").flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}
