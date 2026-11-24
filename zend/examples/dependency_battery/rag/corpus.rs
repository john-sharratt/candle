//! The files R1 retrieves from: the repository's source and documents as
//! text. A file is admitted by its extension, when it is at most
//! [`MAX_BYTES`], outside `target/` and `.git/`, and when its bytes are UTF-8
//! text with no NUL in them. The walk honours the repository's ignore files,
//! as the grounding vocabulary's does, and leaves out what the caller excludes
//! — the battery itself, whose `expect` lists are the answers, and the folder
//! its results are written to.

use std::fs;
use std::path::{Component, Path, PathBuf};
use std::str;

use ignore::WalkBuilder;

/// The extensions of the files retrieved from.
pub const EXTENSIONS: [&str; 12] = [
    "rs", "cu", "cuh", "h", "md", "yaml", "toml", "py", "js", "ts", "css", "html",
];

/// The largest file retrieved from.
pub const MAX_BYTES: u64 = 1 << 20;

/// `path`, relative to the repository, with `/` between its parts.
pub fn relative(repo: &Path, path: &Path) -> Option<String> {
    let rel = path.strip_prefix(repo).ok()?;
    let parts: Vec<&str> = rel
        .components()
        .map(|c| match c {
            Component::Normal(s) => s.to_str(),
            _ => None,
        })
        .collect::<Option<_>>()?;
    Some(parts.join("/"))
}

/// Whether a file at `rel` of `bytes` bytes is one to retrieve from.
pub fn admitted(rel: &str, bytes: u64) -> bool {
    let ext = rel.rsplit_once('.').map(|(_, x)| x).unwrap_or("");
    bytes <= MAX_BYTES
        && EXTENSIONS.contains(&ext)
        && !rel.split('/').any(|p| p == "target" || p == ".git")
}

/// `bytes` as text: UTF-8 holding no NUL, which a binary file would.
pub fn text_of(bytes: &[u8]) -> Option<&str> {
    let text = str::from_utf8(bytes).ok()?;
    (!text.contains('\0')).then_some(text)
}

/// Call `each` with every admitted file of `repo` not under one of `exclude`,
/// as its repository-relative path and its text, in the walk's order.
pub fn walk(
    repo: &Path,
    exclude: &[PathBuf],
    mut each: impl FnMut(String, &str),
) -> anyhow::Result<()> {
    for entry in WalkBuilder::new(repo).build() {
        let entry = entry?;
        let p = entry.path();
        if !entry.file_type().is_some_and(|t| t.is_file())
            || exclude.iter().any(|x| p.starts_with(x))
        {
            continue;
        }
        let Some(rel) = relative(repo, p) else {
            continue;
        };
        let bytes = entry.metadata().map(|m| m.len()).unwrap_or(u64::MAX);
        if !admitted(&rel, bytes) {
            continue;
        }
        let Ok(raw) = fs::read(p) else { continue };
        if let Some(text) = text_of(&raw) {
            each(rel, text);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_is_admitted_by_its_extension_size_and_place() {
        assert!(admitted("candle-nn/src/kv_cache/mod.rs", 10));
        assert!(admitted("docs/unbounded_agents.md", MAX_BYTES));
        assert!(!admitted("docs/big.md", MAX_BYTES + 1));
        assert!(!admitted("candle-kernels/src/x.ptx", 10));
        assert!(!admitted("target/release/build/out.rs", 10));
        assert!(!admitted("a/.git/HEAD.md", 10));
        assert!(!admitted("Makefile", 10));
    }

    #[test]
    fn binary_bytes_are_no_text() {
        assert_eq!(text_of(b"fn main() {}"), Some("fn main() {}"));
        assert_eq!(text_of(b"a\0b"), None);
        assert_eq!(text_of(&[0xff, 0xfe]), None);
    }

    /// **The walk yields the admitted files and leaves the excluded out.**
    #[test]
    fn the_walk_reads_the_admitted_files_alone() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        for (rel, body) in [
            ("src/a.rs", &b"fn tier_fits() {}"[..]),
            ("docs/b.md", b"# Notes"),
            ("target/c.rs", b"fn built() {}"),
            ("src/d.bin.rs", b"\0\0"),
            ("src/e.png", b"png"),
            ("results/answers.md", b"the answers"),
            ("battery.yaml", b"expect: [x]"),
        ] {
            let p = root.join(rel);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(&p, body).unwrap();
        }
        let mut seen = Vec::new();
        walk(
            root,
            &[root.join("results"), root.join("battery.yaml")],
            |rel, text| seen.push((rel, text.to_string())),
        )
        .unwrap();
        seen.sort();
        assert_eq!(
            seen,
            vec![
                ("docs/b.md".to_string(), "# Notes".to_string()),
                ("src/a.rs".to_string(), "fn tier_fits() {}".to_string()),
            ]
        );
    }
}
