//! Applying a patch to a commit's tree, in the object store.
//!
//! `git apply --cached` needs an index to apply into. It gets a private one
//! — a temporary file named by `GIT_INDEX_FILE`, loaded from the base tree
//! and deleted afterwards — so the user's index is never read or written.
//! The file lives in the repository's own git folder, never a shared temp
//! folder where another user could create its path, or its lock, first.

use crate::error::GitError;
use crate::runner::utf8;
use crate::types::{FileMode, Oid, Rev};
use crate::write::scratch::PrivateIndex;
use crate::Repo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplyOutcome {
    /// The base tree with the patch applied.
    Applied { tree: Oid },
    /// The patch does not apply to the base; git's reason.
    Rejected { detail: String },
}

impl Repo {
    /// Apply `patch` (as [`Repo::patch_bytes`] produces, binary included) to
    /// `base`'s tree. The result is a tree for [`Repo::commit_tree`]; no
    /// commit is made and no ref moves.
    pub fn apply_patch(&self, base: &Oid, patch: &[u8]) -> Result<ApplyOutcome, GitError> {
        let _write = self.write_lock();
        let index = PrivateIndex::new(self.git_dir());
        self.git("read-tree")
            .arg("--end-of-options")
            .arg(base.as_str())
            .env("GIT_INDEX_FILE", &index.0)
            .about_rev(base.to_string())
            .run_ok()?;
        let applied = self
            .git("apply")
            .args(["--cached", "--whitespace=nowarn", "-"])
            .env("GIT_INDEX_FILE", &index.0)
            .stdin(patch.to_vec())
            .run_accepting(&[0, 1, 128])?;
        if applied.status != Some(0) {
            return Ok(ApplyOutcome::Rejected {
                detail: applied.stderr.trim().to_string(),
            });
        }
        let tree = self
            .git("write-tree")
            .env("GIT_INDEX_FILE", &index.0)
            .run_ok()?;
        let tree = Oid::parse(utf8("write-tree", tree)?.trim())?;
        self.refuse_unsafe_changes(base, &tree)?;
        Ok(ApplyOutcome::Applied { tree })
    }

    /// Hold a patch to the rules a [`ChangeSet`](crate::ChangeSet) enforces,
    /// judged by what it actually changed: nothing under a protected
    /// `secrets` folder, and no symlink or submodule created or changed — a
    /// symlink is only ever written through
    /// [`ChangeSet::symlink`](crate::ChangeSet::symlink), its own explicit
    /// call.
    fn refuse_unsafe_changes(&self, base: &Oid, tree: &Oid) -> Result<(), GitError> {
        for entry in self.diff(&Rev::Oid(base.clone()), &Rev::Oid(tree.clone()), &[])? {
            for side in [&entry.old, &entry.new].into_iter().flatten() {
                if side.path.is_protected() {
                    return Err(GitError::invalid(format!(
                        "the patch changes {}, which is under a protected `secrets` folder",
                        side.path
                    )));
                }
            }
            if let Some(new) = &entry.new {
                if matches!(new.mode, FileMode::Symlink | FileMode::Submodule) {
                    return Err(GitError::invalid(format!(
                        "the patch makes {} a {}; a patch may only write files",
                        new.path,
                        if new.mode == FileMode::Symlink {
                            "symlink"
                        } else {
                            "submodule"
                        }
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestRepo;

    /// **A patch taken between two commits, applied to the first, gives the
    /// second's tree exactly** — text, binary, added, deleted and renamed
    /// files included — in another repository that shares only the base.
    #[test]
    fn a_patch_round_trips_to_the_exact_tree() {
        let t = TestRepo::init();
        let body: String = (0..20).map(|i| format!("line {i}\n")).collect();
        t.write("edit.txt", b"one\ntwo\n");
        t.write("bin.dat", &[0, 1, 2, 3, 255]);
        t.write("gone.txt", b"gone\n");
        t.write("old name.rs", body.as_bytes());
        let base = t.commit_all("base");
        t.write("edit.txt", b"one\nTWO\n");
        t.write("bin.dat", &[9, 0, 9, 255, 1]);
        std::fs::remove_file(t.path.join("gone.txt")).unwrap();
        t.git(&["mv", "old name.rs", "new name.rs"]);
        t.write("added/ünï.txt", b"new\n");
        let next = t.commit_all("next");
        let repo = t.repo();
        let patch = repo
            .patch_bytes(&Rev::Oid(base.clone()), &Rev::Oid(next.clone()))
            .unwrap();

        // A second repository holding only the base commit, fetched by a
        // branch: git before 2.26 refuses to fetch an unadvertised id.
        t.git(&["branch", "base-only", base.as_str()]);
        let other = TestRepo::init();
        other.git(&["fetch", "-q", t.path.to_str().unwrap(), "base-only"]);
        let other_repo = other.repo();
        match other_repo.apply_patch(&base, &patch).unwrap() {
            ApplyOutcome::Applied { tree } => {
                assert_eq!(tree, t.oid(&format!("{next}^{{tree}}")));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_patch_that_does_not_fit_is_rejected_and_the_index_is_untouched() {
        let t = TestRepo::init();
        t.write("f.txt", b"one\n");
        let base = t.commit_all("base");
        t.write("f.txt", b"two\n");
        let next = t.commit_all("next");
        t.write("f.txt", b"three\n");
        let diverged = t.commit_all("diverged");
        let repo = t.repo();
        let patch = repo.patch_bytes(&Rev::Oid(base), &Rev::Oid(next)).unwrap();
        let index = std::fs::read(t.path.join(".git/index")).unwrap();

        match repo.apply_patch(&diverged, &patch).unwrap() {
            ApplyOutcome::Rejected { detail } => assert!(detail.contains("f.txt"), "{detail}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(std::fs::read(t.path.join(".git/index")).unwrap(), index);
    }

    /// The private index is made in the repository's own git folder and
    /// gone afterwards — nothing is left behind there, applied or rejected.
    #[test]
    fn nothing_is_left_in_the_git_folder() {
        let t = TestRepo::init();
        t.write("f.txt", b"one\n");
        let base = t.commit_all("base");
        t.write("f.txt", b"two\n");
        let next = t.commit_all("next");
        let repo = t.repo();
        let patch = repo
            .patch_bytes(&Rev::Oid(base.clone()), &Rev::Oid(next))
            .unwrap();
        let scratch = || {
            std::fs::read_dir(t.path.join(".git"))
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().starts_with("zen-"))
                .count()
        };
        repo.apply_patch(&base, &patch).unwrap();
        repo.apply_patch(&base, b"not a patch\n").unwrap();
        assert_eq!(scratch(), 0);
    }

    /// **A patch is held to the rules a change set is.** One that writes
    /// under `secrets`, or makes a symlink, is refused — `from: files` would
    /// refuse the same paths, and a patch is no way round it.
    #[test]
    fn a_patch_into_secrets_or_making_a_symlink_is_refused() {
        let t = TestRepo::init();
        t.write("f.txt", b"one\n");
        let base = t.commit_all("base");
        let repo = t.repo();

        let secret = "diff --git a/secrets/token.yaml b/secrets/token.yaml\n\
new file mode 100644\n\
--- /dev/null\n\
+++ b/secrets/token.yaml\n\
@@ -0,0 +1 @@\n\
+token: x\n";
        let link = "diff --git a/escape b/escape\n\
new file mode 120000\n\
--- /dev/null\n\
+++ b/escape\n\
@@ -0,0 +1 @@\n\
+../../outside\n\\ No newline at end of file\n";
        let fine = "diff --git a/g.txt b/g.txt\n\
new file mode 100644\n\
--- /dev/null\n\
+++ b/g.txt\n\
@@ -0,0 +1 @@\n\
+two\n";
        for (patch, why) in [(secret, "secrets"), (link, "symlink")] {
            match repo.apply_patch(&base, patch.as_bytes()) {
                Err(GitError::InvalidInput(detail)) => assert!(detail.contains(why), "{detail}"),
                other => panic!("{why}: {other:?}"),
            }
        }
        assert!(matches!(
            repo.apply_patch(&base, fine.as_bytes()).unwrap(),
            ApplyOutcome::Applied { .. }
        ));
    }

    /// An empty or garbage patch is a rejection, not an error.
    #[test]
    fn input_that_is_not_a_patch_is_rejected() {
        let t = TestRepo::init();
        t.write("f.txt", b"one\n");
        let base = t.commit_all("base");
        let repo = t.repo();
        for input in [&b""[..], b"not a patch\n"] {
            assert!(matches!(
                repo.apply_patch(&base, input).unwrap(),
                ApplyOutcome::Rejected { .. }
            ));
        }
    }
}
