//! Sources the caller already has on disk.

use super::request::SourceRef;
use super::resolve::{Fetched, SourceFetch};
use candle::Result;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// The revision a caller's file is pinned at: its length and modification time.
///
/// A file supplied by path has no hub revision, and an empty one leaves a
/// request unpinned — so a checkpoint reconverted in place under the same name
/// matched the pack built from the old one and was served as the new model.
/// Pinned at its length and mtime, a rewritten file is another source and its
/// pack is rebuilt. Empty when the file is not there: the pack built from it
/// before still serves, which is how a gate runs from a pack whose source has
/// been moved off the machine.
pub fn local_rev(path: &Path) -> String {
    let Ok(meta) = std::fs::metadata(path) else {
        return String::new();
    };
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    format!("local-{:x}-{mtime:x}", meta.len())
}

/// The repo label a caller's checkpoint directory is packed under: the
/// directory's name, and a digest of its full path so two directories of one
/// name never share packs.
pub fn local_label(dir: &Path) -> String {
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "root".into());
    // FNV-1a: stable across builds and platforms, which a std hasher is not —
    // the label names packs on disk, so it must survive a rebuild.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in dir.to_string_lossy().bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("local/{name}-{:08x}", h as u32)
}

/// A [`SourceFetch`] over files a caller supplies by path: a source named under
/// the repo label `repo` is `dir/<file>`, never downloaded and never released;
/// every other source, and every tokenizer, is `other`'s.
///
/// The request still names a local source by a repo label, which is what keys
/// the pack's directory under the cache root and what its provenance records —
/// so a pack built from a local file lives in the model cache like any other,
/// and nothing is written beside the caller's file. A label rather than the
/// directory itself, because a path is not a name: it carries separators and a
/// drive the cache's directory scheme cannot hold.
pub struct LocalFetch<'a> {
    pub repo: String,
    pub dir: PathBuf,
    pub other: &'a dyn SourceFetch,
}

impl SourceFetch for LocalFetch<'_> {
    fn fetch(&self, source: &SourceRef) -> Result<Fetched> {
        if source.repo != self.repo {
            return self.other.fetch(source);
        }
        let path = self.dir.join(&source.file);
        if !path.is_file() {
            candle::bail!(
                "model pack: the {} source {} is not on disk",
                source.role,
                path.display()
            );
        }
        Ok(Fetched {
            path,
            cached: false,
        })
    }

    fn tokenizer_json(&self, repo: &str, rev: &str) -> Result<String> {
        self.other.tokenizer_json(repo, rev)
    }

    fn release(&self, fetched: &Fetched) -> Result<()> {
        if fetched.path.parent() == Some(self.dir.as_path()) {
            candle::bail!(
                "model pack: {} was supplied by path and is the caller's, not the cache's",
                fetched.path.display()
            );
        }
        self.other.release(fetched)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Records what it was asked for, and answers from a fixed file.
    struct Remote {
        answer: PathBuf,
        asked: RefCell<Vec<String>>,
        released: RefCell<Vec<PathBuf>>,
    }

    impl SourceFetch for Remote {
        fn fetch(&self, s: &SourceRef) -> Result<Fetched> {
            self.asked
                .borrow_mut()
                .push(format!("{}:{}", s.role, s.file));
            Ok(Fetched {
                path: self.answer.clone(),
                cached: true,
            })
        }
        fn tokenizer_json(&self, repo: &str, rev: &str) -> Result<String> {
            Ok(format!("{repo}@{rev}"))
        }
        fn release(&self, f: &Fetched) -> Result<()> {
            self.released.borrow_mut().push(f.path.clone());
            Ok(())
        }
    }

    fn remote(dir: &std::path::Path) -> Remote {
        Remote {
            answer: dir.join("remote").join("donor.gguf"),
            asked: RefCell::new(Vec::new()),
            released: RefCell::new(Vec::new()),
        }
    }

    /// A source under the label is answered in place and marked as the
    /// caller's, so the build never releases it.
    #[test]
    fn a_labelled_source_is_found_in_place_and_never_released() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("m.gguf"), b"x").unwrap();
        let other = remote(dir.path());
        let fetch = LocalFetch {
            repo: "local/m".into(),
            dir: dir.path().to_path_buf(),
            other: &other,
        };
        let got = fetch
            .fetch(&SourceRef::checkpoint("local/m", "", "m.gguf"))
            .unwrap();
        assert_eq!(got.path, dir.path().join("m.gguf"));
        assert!(!got.cached);
        assert!(fetch.release(&got).is_err());
        assert!(dir.path().join("m.gguf").exists());
        assert!(other.asked.borrow().is_empty());
    }

    /// Every other source and every tokenizer is the inner fetch's, and so is
    /// releasing what it fetched.
    #[test]
    fn other_sources_and_tokenizers_go_to_the_inner_fetch() {
        let dir = tempfile::tempdir().unwrap();
        let other = remote(dir.path());
        let fetch = LocalFetch {
            repo: "local/m".into(),
            dir: dir.path().to_path_buf(),
            other: &other,
        };
        let donor = SourceRef {
            role: "gate-donor".into(),
            repo: "org/base".into(),
            rev: "r".into(),
            file: "donor.gguf".into(),
        };
        let got = fetch.fetch(&donor).unwrap();
        assert!(got.cached);
        fetch.release(&got).unwrap();
        assert_eq!(*other.asked.borrow(), ["gate-donor:donor.gguf"]);
        assert_eq!(*other.released.borrow(), [got.path]);
        assert_eq!(fetch.tokenizer_json("org/t", "r").unwrap(), "org/t@r");
    }

    /// The label keeps the directory's name readable and tells two directories
    /// of one name apart; the same directory always gets the same label.
    #[test]
    fn a_local_label_is_the_name_and_a_stable_digest_of_the_path() {
        let a = local_label(Path::new("/models/a/qwen"));
        let b = local_label(Path::new("/models/b/qwen"));
        assert!(
            a.starts_with("local/qwen-") && a.len() == "local/qwen-".len() + 8,
            "{a}"
        );
        assert_ne!(a, b);
        assert_eq!(a, local_label(Path::new("/models/a/qwen")));
        // FNV-1a of the path's bytes, low 32 bits — pinned so a label, and the
        // packs filed under it, survive a rebuild.
        assert_eq!(local_label(Path::new("m")), "local/m-8601f358");
    }

    /// The same file reads the same revision; rewriting it in place moves it;
    /// an absent file is unpinned.
    #[test]
    fn a_local_rev_moves_when_the_file_is_rewritten() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.gguf");
        std::fs::write(&path, b"one").unwrap();
        let first = local_rev(&path);
        assert!(first.starts_with("local-3-"), "{first}");
        assert_eq!(local_rev(&path), first);
        std::fs::write(&path, b"three").unwrap();
        let second = local_rev(&path);
        assert!(second.starts_with("local-5-"), "{second}");
        assert_ne!(second, first);
        assert_eq!(local_rev(&dir.path().join("absent.gguf")), "");
    }

    #[test]
    fn a_missing_local_source_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let other = remote(dir.path());
        let fetch = LocalFetch {
            repo: "local/m".into(),
            dir: dir.path().to_path_buf(),
            other: &other,
        };
        let Err(e) = fetch.fetch(&SourceRef::checkpoint("local/m", "", "absent.gguf")) else {
            panic!("an absent source was answered");
        };
        let e = e.to_string();
        assert!(e.contains("not on disk"), "{e}");
    }
}
