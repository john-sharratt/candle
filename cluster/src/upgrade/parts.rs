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

    /// Remove the downloads of every build but those `keep` names by hash —
    /// the ones still being fetched — once a build is installed. The files
    /// beside them (`ok`, `refused.json`, the plan) are not downloads and stay.
    /// Answers the builds removed.
    ///
    /// An installed build's downloads were never removed, so every build a
    /// member had installed stayed here: ~150 on WhiteWorld, 60.7 GB, on the
    /// drive its substrate shares, which then filled (2026-11-07).
    pub fn keep_only(&self, keep: &[&str]) -> io::Result<Vec<String>> {
        let entries = match fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let mut removed = Vec::new();
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if !entry.file_type()?.is_dir() || keep.contains(&name.as_str()) {
                continue;
            }
            fs::remove_dir_all(entry.path())?;
            removed.push(name);
        }
        removed.sort();
        Ok(removed)
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

    /// **Once a build is installed, only the downloads still being fetched
    /// stay**: every other build's folder goes, and the files beside them —
    /// the watchdog's `ok`, `refused.json` — are left.
    #[test]
    fn an_install_keeps_only_the_downloads_still_fetching() {
        let t = TempDir::new().unwrap();
        let parts = Parts::new(t.path());
        for hash in ["old1", "old2", "newer"] {
            fs::create_dir_all(t.path().join(hash)).unwrap();
            fs::write(t.path().join(hash).join("x.part"), b"x").unwrap();
        }
        fs::write(t.path().join("ok"), b"").unwrap();
        fs::write(t.path().join("refused.json"), b"[]").unwrap();
        assert_eq!(parts.keep_only(&["newer"]).unwrap(), ["old1", "old2"]);
        let mut left: Vec<String> = fs::read_dir(t.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        assert_eq!(left, ["newer", "ok", "refused.json"]);
        assert!(Parts::new(t.path().join("absent"))
            .keep_only(&[])
            .unwrap()
            .is_empty());
    }
}
