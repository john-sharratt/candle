//! Cloning a remote.

use std::path::Path;

use crate::error::GitError;
use crate::runner::Invocation;
use crate::setup::require_empty_target;
use crate::types::{BranchName, RemoteName, RemoteUrl};
use crate::version;
use crate::Repo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloneOptions {
    /// The branch to check out; the remote's default when `None`.
    pub branch: Option<BranchName>,
    /// Fetch commits and trees only. File contents are fetched from the
    /// remote on first read — by a blob reader as much as by a checkout — so
    /// a large repository clones in a fraction of the data.
    pub partial: bool,
    /// Check the branch out into the working tree.
    pub checkout: bool,
}

impl Default for CloneOptions {
    fn default() -> Self {
        Self {
            branch: None,
            partial: false,
            checkout: true,
        }
    }
}

impl Repo {
    /// Clone `url` into `dir`, which must not exist or must be an empty
    /// folder. The remote is named `origin`.
    pub fn clone(url: &RemoteUrl, dir: &Path, options: &CloneOptions) -> Result<Repo, GitError> {
        version::installed()?;
        require_empty_target(dir)?;
        let parent = dir
            .parent()
            .ok_or_else(|| GitError::invalid(format!("{} has no parent", dir.display())))?;
        std::fs::create_dir_all(parent)?;
        // `--no-local`: a local path goes through the ordinary transport,
        // never the shortcut that copies or hardlinks the source's files —
        // the path of the local-clone fixes (CVE-2024-32004, -32020,
        // -32021). `--no-recurse-submodules`: submodules are never cloned —
        // the path of the clone-time code execution fixes (CVE-2024-32002,
        // CVE-2025-48384). Both hold on every supported git.
        let mut inv = Invocation::new(parent, "clone").args([
            "-q",
            "--no-local",
            "--no-hardlinks",
            "--no-recurse-submodules",
        ]);
        if options.partial {
            inv = inv.arg("--filter=blob:none");
        }
        if !options.checkout {
            inv = inv.arg("--no-checkout");
        }
        if let Some(branch) = &options.branch {
            inv = inv.arg("--branch").arg(branch.as_str());
        }
        let origin = RemoteName::parse("origin").expect("a valid remote name");
        let existed = dir.exists();
        let cloned = inv
            .arg("--end-of-options")
            .arg(url.as_str())
            .arg(dir)
            .transfer(&origin)
            .run_ok();
        if let Err(e) = cloned {
            discard_partial(dir, existed);
            return Err(e);
        }
        Repo::open(dir)
    }
}

/// Undo a clone that did not finish. Git cleans up after its own failures,
/// but not after a timeout kills it, and the half-written folder it leaves
/// would refuse every retry as occupied. The folder was empty or absent
/// before the clone, so everything in it is the clone's. Best effort: the
/// clone's own error is what the caller needs, and a folder that cannot be
/// cleared is reported by the retry.
fn discard_partial(dir: &Path, existed: bool) {
    if !existed {
        let _ = std::fs::remove_dir_all(dir);
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let _ = match entry.file_type() {
            Ok(t) if t.is_dir() => std::fs::remove_dir_all(&path),
            _ => std::fs::remove_file(&path),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read::head::Head;
    use crate::testing::{git_in, scratch, TestRepo};
    use crate::types::{Oid, RepoPath, Rev};

    fn origin_with_history() -> (TestRepo, String, Oid) {
        let origin = TestRepo::bare();
        origin.git(&["config", "uploadpack.allowFilter", "true"]);
        origin.git(&["config", "uploadpack.allowAnySHA1InWant", "true"]);
        let t = TestRepo::init();
        t.write("f.txt", b"first version\n");
        let first = t.commit_all("first");
        t.write("f.txt", b"second version\n");
        t.commit_all("second");
        t.git(&["checkout", "-q", "-b", "side"]);
        t.write("side.txt", b"side\n");
        t.commit_all("side");
        t.git(&["push", "-q", &origin.url(), "main", "side"]);
        let url = origin.url();
        (origin, url, first)
    }

    #[test]
    fn a_clone_checks_out_the_default_or_named_branch() {
        let (_origin, url, _) = origin_with_history();
        let dir = tempfile::Builder::new()
            .prefix("clone-")
            .tempdir_in(scratch())
            .unwrap();
        let url = RemoteUrl::parse(&url).unwrap();

        let a = Repo::clone(&url, &dir.path().join("a"), &CloneOptions::default()).unwrap();
        assert_eq!(a.head().unwrap().branch().unwrap().as_str(), "main");
        // The checkout follows the user's own line-ending config (Git for
        // Windows' portable build sets `core.autocrlf=true`), so compare
        // the text, not its line endings.
        let checked_out = std::fs::read_to_string(dir.path().join("a/f.txt")).unwrap();
        assert_eq!(checked_out.replace("\r\n", "\n"), "second version\n");
        assert_eq!(a.remotes().unwrap()[0].name.as_str(), "origin");

        let side = BranchName::parse("side").unwrap();
        let b = Repo::clone(
            &url,
            &dir.path().join("b"),
            &CloneOptions {
                branch: Some(side.clone()),
                ..CloneOptions::default()
            },
        )
        .unwrap();
        assert_eq!(b.head().unwrap().branch(), Some(&side));
        assert!(dir.path().join("b/side.txt").exists());

        let c = Repo::clone(
            &url,
            &dir.path().join("c"),
            &CloneOptions {
                checkout: false,
                ..CloneOptions::default()
            },
        )
        .unwrap();
        assert!(matches!(c.head().unwrap(), Head::Branch { .. }));
        assert!(!dir.path().join("c/f.txt").exists());
    }

    /// **A partial clone fetches file contents on first read.** The old
    /// version of a file is absent after the clone and arrives when the
    /// blob reader asks for it.
    #[test]
    fn a_partial_clone_fetches_contents_on_demand() {
        let (_origin, url, first) = origin_with_history();
        let dir = tempfile::Builder::new()
            .prefix("clone-")
            .tempdir_in(scratch())
            .unwrap();
        let target = dir.path().join("partial");
        let repo = Repo::clone(
            &RemoteUrl::parse(&url).unwrap(),
            &target,
            &CloneOptions {
                partial: true,
                ..CloneOptions::default()
            },
        )
        .unwrap();
        // Trees are present in a blob-less clone, so the old blob's id is
        // known without fetching it.
        let old_blob = git_in(&target, &["rev-parse", &format!("{first}:f.txt")]);
        let old_blob = old_blob.trim().to_string();
        let is_missing = || {
            git_in(
                &target,
                &["rev-list", "--objects", "--missing=print", "--all"],
            )
            .lines()
            .any(|l| l == format!("?{old_blob}"))
        };
        assert!(is_missing(), "the old blob was not fetched by the clone");

        let old = repo
            .blobs()
            .read_at(&Rev::Oid(first), &RepoPath::parse("f.txt").unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(old, b"first version\n");
        assert!(!is_missing(), "the read fetched the missing blob");
    }

    #[test]
    fn a_clone_into_an_occupied_folder_or_from_nowhere_fails() {
        let (_origin, url, _) = origin_with_history();
        let dir = tempfile::Builder::new()
            .prefix("clone-")
            .tempdir_in(scratch())
            .unwrap();
        std::fs::write(dir.path().join("occupied"), b"x").unwrap();
        let url = RemoteUrl::parse(&url).unwrap();
        assert!(matches!(
            Repo::clone(&url, dir.path(), &CloneOptions::default()),
            Err(GitError::InvalidInput(_))
        ));
        let nowhere =
            RemoteUrl::parse(dir.path().join("no-such-origin").to_str().unwrap()).unwrap();
        assert!(matches!(
            Repo::clone(&nowhere, &dir.path().join("x"), &CloneOptions::default()),
            Err(GitError::RemoteUnreachable { .. })
        ));
    }

    /// **A clone cut off part-way leaves nothing behind**, so a retry finds
    /// its target as the first attempt did: a folder that was created is
    /// removed, and one that was already there is emptied but kept.
    #[test]
    fn an_unfinished_clone_is_discarded() {
        let dir = tempfile::Builder::new()
            .prefix("clone-")
            .tempdir_in(scratch())
            .unwrap();
        let half_written = |at: &Path| {
            std::fs::create_dir_all(at.join(".git/objects/pack")).unwrap();
            std::fs::write(at.join(".git/objects/pack/tmp_pack_x"), b"partial").unwrap();
            std::fs::write(at.join("f.txt"), b"checked out").unwrap();
        };

        let created = dir.path().join("created");
        half_written(&created);
        discard_partial(&created, false);
        assert!(!created.exists());

        let kept = dir.path().join("kept");
        half_written(&kept);
        discard_partial(&kept, true);
        assert!(kept.is_dir());
        assert!(std::fs::read_dir(&kept).unwrap().next().is_none());
        require_empty_target(&kept).unwrap();
    }
}
