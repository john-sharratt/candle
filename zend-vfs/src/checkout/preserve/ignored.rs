//! The ignored files a checkout held when its own state was set aside.
//!
//! A snapshot holds every file git sees; an ignored one it does not, and
//! need not — ignored files are left where they are. But whether a file is
//! ignored depends on rules that live in the working tree: a run puts the
//! checkout on another branch, whose `.gitignore` may not ignore it, and an
//! uncommitted edit to `.gitignore` is gone the moment `HEAD` is checked out
//! again. Under the rules of the moment such a file reads as untracked, and
//! clearing a run's untracked files would take it.
//!
//! So what was ignored when the state was set aside is listed then, under
//! the checkout's own rules, and kept with the preservation: a folder
//! ignored whole as one entry, ending in `/`. Nothing it covers is ever
//! removed as a run's, and nothing it covers is read back as a change of the
//! conversation's — whatever the rules say by then.

use super::journal::Place;
use crate::checkout::CheckoutError;
use crate::Repo;

/// What was ignored when a checkout's state was set aside. Paths are git's
/// bytes, so a name that is not UTF-8 is covered like any other.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Ignored {
    /// Sorted. A folder ignored whole ends in `/`.
    entries: Vec<Vec<u8>>,
}

impl Ignored {
    /// Everything ignored in `repo`'s checkout now, under its own rules.
    pub(super) fn listed(repo: &Repo) -> Result<Self, CheckoutError> {
        let out = repo
            .git("ls-files")
            .args([
                "-z",
                "--others",
                "--ignored",
                "--exclude-standard",
                "--directory",
            ])
            .read_only()
            .run_ok()?;
        Ok(Self::from_bytes(&out))
    }

    /// Everything the checkout's preservations on disk list — every one, in
    /// this process or another, live or left by a crash: none of their
    /// files is anyone's to remove.
    pub(crate) fn kept_in(repo: &Repo) -> Result<Self, CheckoutError> {
        let mut entries = Vec::new();
        for place in Place::left_in(repo.git_dir())? {
            entries.extend(place.load_ignored()?.entries);
        }
        entries.sort();
        entries.dedup();
        Ok(Self { entries })
    }

    /// Whether `path` was ignored: listed itself, or inside a folder that
    /// was ignored whole.
    pub(crate) fn covers(&self, path: &[u8]) -> bool {
        self.entries
            .iter()
            .any(|entry| match entry.strip_suffix(b"/") {
                Some(folder) => {
                    path == folder
                        || (path.starts_with(folder) && path.get(folder.len()) == Some(&b'/'))
                }
                None => path == entry.as_slice(),
            })
    }

    /// The entries as saved: each followed by a NUL.
    pub(super) fn to_bytes(&self) -> Vec<u8> {
        self.entries
            .iter()
            .flat_map(|e| e.iter().copied().chain(std::iter::once(0)))
            .collect()
    }

    pub(super) fn from_bytes(bytes: &[u8]) -> Self {
        let mut entries: Vec<Vec<u8>> = bytes
            .split(|b| *b == 0)
            .filter(|e| !e.is_empty())
            .map(<[u8]>::to_vec)
            .collect();
        entries.sort();
        Self { entries }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A listed file and everything in a listed folder are covered**, and
    /// nothing that merely shares a prefix.
    #[test]
    fn files_and_folders_ignored_whole_are_covered() {
        let ignored = Ignored::from_bytes(b"target/\0.env\0dir/local.cfg\0");
        assert!(ignored.covers(b".env"));
        assert!(ignored.covers(b"dir/local.cfg"));
        assert!(ignored.covers(b"target"));
        assert!(ignored.covers(b"target/debug/app"));
        assert!(!ignored.covers(b"targets/x"));
        assert!(!ignored.covers(b".env.local"));
        assert!(!ignored.covers(b"dir"));
        assert!(!ignored.covers(b"src/main.rs"));
    }

    /// **The saved form reads back exactly**, a name that is not UTF-8
    /// included.
    #[test]
    fn the_list_reads_back_as_saved() {
        let ignored = Ignored::from_bytes(b"b\0a/\0\xff\xfe\0");
        assert_eq!(ignored.to_bytes(), b"a/\0b\0\xff\xfe\0");
        assert_eq!(Ignored::from_bytes(&ignored.to_bytes()), ignored);
        assert!(ignored.covers(b"\xff\xfe"));
    }
}
