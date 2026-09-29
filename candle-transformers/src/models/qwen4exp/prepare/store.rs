//! Where the build's source files come from, how they are checked, and how they
//! are released once the artifact that consumed them is complete.
//!
//! The sources are large — the `Q8_0` split alone is 188 GB — and the artifact
//! carries everything the engine reads, so once the artifact is built and
//! verified the sources are dead weight on the disk. [`release`] deletes every
//! cached copy of each one, and nothing else: a copy is identified by the pinned
//! location the store reports for it *and* its published length, so a file that
//! merely shares a directory is never touched.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use candle::Result;
use rayon::prelude::*;
use sha2::{Digest, Sha256};

use super::recipe::{hex, SourceFile};

/// A place pinned source files are fetched from and cached in.
pub trait SourceStore: Sync {
    /// A local path holding `file`'s bytes, downloading it if no cached copy
    /// exists.
    fn fetch(&self, file: &SourceFile) -> Result<PathBuf>;

    /// Every local path that may hold a cached copy of `file`, present or not.
    fn cached_copies(&self, file: &SourceFile) -> Vec<PathBuf>;
}

/// Read `path` and return its SHA-256, lowercase hex.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// Fetch every source and check each against its pin — length first (free),
/// then SHA-256. The hashes run in parallel, one file per thread; a mismatch
/// names the file and both values.
pub fn fetch_verified(store: &dyn SourceStore, sources: &[&SourceFile]) -> Result<Vec<PathBuf>> {
    let paths: Vec<PathBuf> = sources
        .iter()
        .map(|s| store.fetch(s))
        .collect::<Result<_>>()?;
    sources
        .par_iter()
        .zip(paths.par_iter())
        .try_for_each(|(s, p)| -> Result<()> {
            let len = std::fs::metadata(p)?.len();
            if len != s.bytes {
                candle::bail!(
                    "source {} at {p:?} is {len} bytes, pinned {}",
                    s.path,
                    s.bytes
                );
            }
            let got = sha256_file(p)?;
            if got != s.sha256 {
                candle::bail!(
                    "source {} at {p:?} hashes to {got}, pinned {}",
                    s.path,
                    s.sha256
                );
            }
            Ok(())
        })?;
    Ok(paths)
}

/// Delete every cached copy of `sources` whose length matches its pin.
///
/// A cache entry that is a link is removed, and so is the file it points at
/// when that file is the pinned object itself — a hub-cache blob, whose name
/// **is** the LFS SHA-256. A link to anything else (a copy kept elsewhere on the
/// machine) loses only the link: the pin identifies a cache's object, not every
/// file on the disk with the same bytes. Returns the bytes freed.
pub fn release(store: &dyn SourceStore, sources: &[&SourceFile]) -> Result<u64> {
    let mut freed = 0u64;
    for s in sources {
        for copy in store.cached_copies(s) {
            let Ok(link_meta) = std::fs::symlink_metadata(&copy) else {
                continue;
            };
            if !link_meta.file_type().is_symlink() {
                if link_meta.len() == s.bytes {
                    std::fs::remove_file(&copy)?;
                    freed += link_meta.len();
                }
                continue;
            }
            let target = std::fs::canonicalize(&copy).ok();
            std::fs::remove_file(&copy)?;
            if let Some(target) = target.filter(|t| is_pinned_blob(t, s)) {
                let len = std::fs::metadata(&target)?.len();
                if len == s.bytes {
                    std::fs::remove_file(&target)?;
                    freed += len;
                }
            }
        }
    }
    Ok(freed)
}

/// Whether `target` is `source`'s own cache object: a file named by its pinned
/// SHA-256, as the hub cache names every blob.
fn is_pinned_blob(target: &Path, source: &SourceFile) -> bool {
    target.file_name().and_then(|n| n.to_str()) == Some(source.sha256)
}

#[cfg(test)]
mod tests {
    use super::super::recipe::SourceRole;
    use super::*;

    /// A store over a directory: `fetch` answers the one pinned location,
    /// `cached_copies` reports it plus a second location, as the hub and the
    /// fallback cache do.
    struct DirStore {
        primary: PathBuf,
        secondary: PathBuf,
    }

    impl SourceStore for DirStore {
        fn fetch(&self, file: &SourceFile) -> Result<PathBuf> {
            Ok(self.primary.join(file.path))
        }
        fn cached_copies(&self, file: &SourceFile) -> Vec<PathBuf> {
            vec![self.primary.join(file.path), self.secondary.join(file.path)]
        }
    }

    fn pinned(path: &'static str, bytes: u64, sha256: &'static str) -> SourceFile {
        SourceFile {
            role: SourceRole::Trunk,
            repo: "org/model",
            revision: "abc",
            path,
            bytes,
            sha256,
        }
    }

    fn store(dir: &Path) -> DirStore {
        let s = DirStore {
            primary: dir.join("hub"),
            secondary: dir.join("fallback"),
        };
        std::fs::create_dir_all(&s.primary).unwrap();
        std::fs::create_dir_all(&s.secondary).unwrap();
        s
    }

    /// SHA-256("abc") — the FIPS 180-2 test vector.
    const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    #[test]
    fn sha256_file_matches_the_standard_vector() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f");
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(sha256_file(&p).unwrap(), ABC);
    }

    #[test]
    fn fetch_verified_accepts_a_matching_file() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());
        std::fs::write(s.primary.join("a"), b"abc").unwrap();
        let f = pinned("a", 3, ABC);
        let paths = fetch_verified(&s, &[&f]).unwrap();
        assert_eq!(paths, [s.primary.join("a")]);
    }

    #[test]
    fn fetch_verified_refuses_a_wrong_length_or_hash() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());
        std::fs::write(s.primary.join("a"), b"abd").unwrap();
        let wrong_hash = pinned("a", 3, ABC);
        let err = fetch_verified(&s, &[&wrong_hash]).unwrap_err().to_string();
        assert!(err.contains("hashes to"), "{err}");
        let wrong_len = pinned("a", 4, ABC);
        let err = fetch_verified(&s, &[&wrong_len]).unwrap_err().to_string();
        assert!(err.contains("is 3 bytes, pinned 4"), "{err}");
    }

    /// Both cached copies go; a neighbour in the same directory and a copy whose
    /// length does not match the pin stay.
    #[test]
    fn release_deletes_only_pinned_copies() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());
        std::fs::write(s.primary.join("a"), b"abc").unwrap();
        std::fs::write(s.secondary.join("a"), b"abc").unwrap();
        std::fs::write(s.primary.join("neighbour"), b"abc").unwrap();
        std::fs::write(s.primary.join("b"), b"abcd").unwrap();
        let a = pinned("a", 3, ABC);
        let b = pinned("b", 3, ABC);
        let freed = release(&s, &[&a, &b]).unwrap();
        assert_eq!(freed, 6);
        assert!(!s.primary.join("a").exists());
        assert!(!s.secondary.join("a").exists());
        assert!(s.primary.join("neighbour").exists());
        assert!(
            s.primary.join("b").exists(),
            "a length mismatch is not our file"
        );
    }

    /// Only a file named by the pin's hash is the cache's object.
    #[test]
    fn only_a_blob_named_by_the_pin_is_the_pinned_object() {
        let f = pinned("a", 3, ABC);
        assert!(is_pinned_blob(&Path::new("blobs").join(ABC), &f));
        assert!(!is_pinned_blob(Path::new("D:/models/a.gguf"), &f));
        assert!(!is_pinned_blob(&Path::new("blobs").join(&ABC[1..]), &f));
    }

    #[test]
    fn release_of_absent_copies_frees_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());
        let a = pinned("a", 3, ABC);
        assert_eq!(release(&s, &[&a]).unwrap(), 0);
    }
}
