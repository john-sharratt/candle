//! An artifact's download on disk (§3.9): chunks are appended to
//! `<data>/upgrade/<hash>/<sha256>.part`, and the part file *is* the resume
//! point — a dropped socket, a restarted member or a vanished source picks up
//! at its length. A finished artifact is checked against its SHA-256: a match
//! is renamed to `<sha256>` beside it — `<sha256>.exe` for a binary on
//! Windows, executable on Unix — verified; a mismatch is discarded and
//! fetched again from the start. A rejected build's directory is removed.

use std::env::consts::EXE_SUFFIX;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use super::sum::{file_sha256, hex};
use crate::frame::{Artifact, BuildId};

/// Where an artifact's download stands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Standing {
    /// Verified, at this path.
    Verified(PathBuf),
    /// Fetch from this offset: the part file's length.
    Resume(u64),
}

/// What writing a chunk came to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wrote {
    /// Appended: the part is this long now.
    Appended(u64),
    /// Not at the part's end — a chunk answering an older request: ignored.
    Stale,
}

/// What finishing a download came to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finished {
    Verified(PathBuf),
    /// Its SHA-256 was not the artifact's: the part is gone, and the fetch
    /// starts again from zero.
    Mismatch,
}

/// The downloads under `<data>/upgrade`.
#[derive(Debug, Clone)]
pub struct Parts {
    root: PathBuf,
}

impl Parts {
    pub fn new(root: impl Into<PathBuf>) -> Parts {
        Parts { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn dir(&self, build: &BuildId) -> PathBuf {
        self.root.join(&build.hash)
    }

    pub fn part(&self, build: &BuildId, a: &Artifact) -> PathBuf {
        self.dir(build).join(format!("{}.part", hex(&a.sha256)))
    }

    /// A verified artifact's file, named as this platform names an executable,
    /// so it runs — `--build-info` asks it what it opens before it is
    /// installed.
    pub fn verified(&self, build: &BuildId, a: &Artifact) -> PathBuf {
        self.dir(build)
            .join(format!("{}{EXE_SUFFIX}", hex(&a.sha256)))
    }

    pub fn standing(&self, build: &BuildId, a: &Artifact) -> Standing {
        let done = self.verified(build, a);
        if done.is_file() {
            return Standing::Verified(done);
        }
        Standing::Resume(fs::metadata(self.part(build, a)).map_or(0, |m| m.len()))
    }

    /// Append `bytes` at `offset`, which must be where the part ends.
    pub fn write(
        &self,
        build: &BuildId,
        a: &Artifact,
        offset: u64,
        bytes: &[u8],
    ) -> io::Result<Wrote> {
        let part = self.part(build, a);
        fs::create_dir_all(self.dir(build))?;
        let len = fs::metadata(&part).map_or(0, |m| m.len());
        if len != offset || len + bytes.len() as u64 > a.size {
            return Ok(Wrote::Stale);
        }
        let mut f = OpenOptions::new().create(true).append(true).open(&part)?;
        f.write_all(bytes)?;
        f.sync_data()?;
        Ok(Wrote::Appended(len + bytes.len() as u64))
    }

    /// Check a complete part against its SHA-256.
    pub fn finish(&self, build: &BuildId, a: &Artifact) -> io::Result<Finished> {
        let part = self.part(build, a);
        if file_sha256(&part)? == a.sha256 {
            let done = self.verified(build, a);
            fs::rename(&part, &done)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&done, fs::Permissions::from_mode(0o755))?;
            }
            Ok(Finished::Verified(done))
        } else {
            fs::remove_file(&part)?;
            Ok(Finished::Mismatch)
        }
    }

    /// Remove every download of `build`: it was rejected, or it is installed.
    pub fn remove(&self, build: &BuildId) -> io::Result<()> {
        match fs::remove_dir_all(self.dir(build)) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::TempDir;

    use crate::upgrade::sum::sha256;

    fn build() -> BuildId {
        BuildId {
            count: 7,
            hash: "abc1234".into(),
            local: String::new(),
        }
    }

    fn artifact(bytes: &[u8]) -> Artifact {
        Artifact {
            kind: "zend".into(),
            platform: "linux-x86_64".into(),
            size: bytes.len() as u64,
            sha256: sha256(bytes),
        }
    }

    /// A download cut at any offset resumes from the part's length, and a
    /// chunk not at the end is ignored; the whole verifies and is renamed.
    #[test]
    fn a_download_resumes_from_the_part_and_verifies() {
        let t = TempDir::new().unwrap();
        let parts = Parts::new(t.path().join("upgrade"));
        let bytes: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
        let a = artifact(&bytes);
        assert_eq!(parts.standing(&build(), &a), Standing::Resume(0));
        assert_eq!(
            parts.write(&build(), &a, 0, &bytes[..300]).unwrap(),
            Wrote::Appended(300)
        );
        // A restart: the offset is the part's length.
        let again = Parts::new(t.path().join("upgrade"));
        assert_eq!(again.standing(&build(), &a), Standing::Resume(300));
        assert_eq!(
            again.write(&build(), &a, 0, &bytes[..300]).unwrap(),
            Wrote::Stale
        );
        assert_eq!(
            again.write(&build(), &a, 300, &bytes[300..]).unwrap(),
            Wrote::Appended(1000)
        );
        let done = t
            .path()
            .join("upgrade/abc1234")
            .join(format!("{}{EXE_SUFFIX}", hex(&a.sha256)));
        assert_eq!(
            again.finish(&build(), &a).unwrap(),
            Finished::Verified(done.clone())
        );
        assert_eq!(
            again.standing(&build(), &a),
            Standing::Verified(done.clone())
        );
        assert_eq!(fs::read(done).unwrap(), bytes);
    }

    /// A hash mismatch discards the part, and the fetch starts over.
    #[test]
    fn a_mismatch_discards_and_restarts() {
        let t = TempDir::new().unwrap();
        let parts = Parts::new(t.path());
        let a = artifact(b"right");
        parts.write(&build(), &a, 0, b"wrong").unwrap();
        assert_eq!(parts.finish(&build(), &a).unwrap(), Finished::Mismatch);
        assert_eq!(parts.standing(&build(), &a), Standing::Resume(0));
        // More than the artifact's size is never written.
        assert_eq!(
            parts.write(&build(), &a, 0, b"too long!").unwrap(),
            Wrote::Stale
        );
    }

    #[test]
    fn a_rejected_builds_downloads_are_removed() {
        let t = TempDir::new().unwrap();
        let parts = Parts::new(t.path());
        let a = artifact(b"x");
        parts.write(&build(), &a, 0, b"x").unwrap();
        parts.remove(&build()).unwrap();
        assert!(!t.path().join("abc1234").exists());
        parts.remove(&build()).unwrap();
    }
}
