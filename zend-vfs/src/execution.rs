//! The one place the layer changes a working tree: a checkout the daemon owns
//! and runs tools in, never a developer's.
//!
//! Every other operation writes objects and moves refs and leaves the working
//! tree, the index and `HEAD` alone. A tool run needs the opposite — a checkout
//! put at an exact state, a conversation's changes laid on top, the tool run
//! there, and what it changed read back — and that checkout is the daemon's to
//! overwrite. [`Repo::force_checkout_branch`] is its reset: `HEAD` on one
//! branch, and the tracked files and the index at that branch's commit,
//! whatever they held before. Untracked files are left to the caller, which
//! knows which of them it wrote. [`Repo::restore_paths`] is the narrow form,
//! for files a run is known to have changed, and [`Repo::read_committed_all`]
//! reads files as they are committed — the form a conversation's file store
//! reads them in, and so the form its changes are made against.

use std::collections::{BTreeMap, BTreeSet};

use crate::error::GitError;
use crate::types::{BranchName, Oid, RepoPath, Rev};
use crate::{RefOp, RefTransaction, Repo};

/// `paths` in runs short enough for one command line: well inside Windows'
/// 32 767 characters, with room for the fixed arguments and the repository's
/// path.
fn batches<'a>(paths: &'a [&'a RepoPath]) -> Vec<&'a [&'a RepoPath]> {
    const BATCH_CHARS: usize = 16_000;
    let mut out = Vec::new();
    let mut rest = paths;
    while !rest.is_empty() {
        let mut take = 0;
        let mut chars = 0;
        while take < rest.len() && (take == 0 || chars + rest[take].as_str().len() < BATCH_CHARS) {
            chars += rest[take].as_str().len() + 1;
            take += 1;
        }
        let (batch, tail) = rest.split_at(take);
        out.push(batch);
        rest = tail;
    }
    out
}

impl Repo {
    /// Put `HEAD` on `branch` and the working tree and the index at its
    /// commit — `git reset --hard` and a switch to `branch` in one step,
    /// moving no branch. Returns the commit.
    ///
    /// Every change to a tracked file is discarded. Git rewrites only the files
    /// whose content differs from the commit (it compares each against the
    /// index by its stat information first), so a file already at the commit's
    /// content keeps its timestamps — which is what keeps a build cache over
    /// the checkout valid. Untracked and ignored files are not touched.
    /// Submodules are not recursed into.
    ///
    /// **Discards work.** Only for a checkout the daemon owns.
    pub fn force_checkout_branch(&self, branch: &BranchName) -> Result<Oid, GitError> {
        // Resolved first, so a missing branch is a typed refusal rather than
        // git's "did not match any file(s) known to git".
        let commit = self.resolve(&Rev::Branch(branch.clone()))?;
        let _write = self.write_lock();
        self.git("checkout")
            .args([
                "--quiet",
                "--force",
                "--no-recurse-submodules",
                "--end-of-options",
            ])
            .arg(branch.as_str())
            // Nothing after the branch is a path.
            .arg("--")
            .about_rev(branch.to_string())
            .run_ok()?;
        Ok(commit)
    }

    /// Put `branch` at `to`, as a compare-and-swap against `from` — the commit
    /// it holds now, or `None` when it does not exist — whether or not it is
    /// checked out. The index and the working tree are not touched, so they
    /// read as changes against `to`.
    ///
    /// The one ref move that ignores a checkout: every other refuses to move
    /// a branch `HEAD` is on, because that changes a checkout under its user.
    /// **Only for a checkout the daemon owns**, taking back a branch a tool
    /// moved during a run.
    pub fn force_branch_tip(
        &self,
        branch: &BranchName,
        from: Option<&Oid>,
        to: &Oid,
    ) -> Result<(), GitError> {
        let op = match from {
            Some(old) => RefOp::Update {
                name: branch.to_ref(),
                new: to.clone(),
                old: old.clone(),
            },
            None => RefOp::Create {
                name: branch.to_ref(),
                new: to.clone(),
            },
        };
        let _write = self.write_lock();
        self.write_refs(&RefTransaction::new().push(op))
    }

    /// Point `HEAD` at `branch` — `git symbolic-ref HEAD refs/heads/<branch>` —
    /// leaving the index and the working tree exactly as they are, so they
    /// read as changes against the branch's commit.
    ///
    /// **Changes what the checkout is on.** Only for a checkout the daemon
    /// owns.
    pub fn attach_head(&self, branch: &BranchName) -> Result<(), GitError> {
        let _write = self.write_lock();
        self.git("symbolic-ref")
            .arg("HEAD")
            .arg(branch.to_ref().as_str())
            .about_rev(branch.to_string())
            .run_ok()?;
        Ok(())
    }

    /// Put each of `paths` back to its content at `commit` in the working
    /// tree — `git checkout <commit> -- <paths>` — leaving every other file as
    /// it is. Each path must be one `commit` holds: this restores tracked files
    /// a run changed or deleted, where a whole [`Self::force_checkout_branch`]
    /// would rewrite nothing more but costs a walk of the whole tree.
    ///
    /// Passed in batches, so any number of paths stays within the command-line
    /// limit.
    ///
    /// **Discards work.** Only for a checkout the daemon owns.
    pub fn restore_paths(&self, commit: &Oid, paths: &[&RepoPath]) -> Result<(), GitError> {
        let _write = self.write_lock();
        for batch in batches(paths) {
            self.git("checkout")
                .args(["--quiet", "--force", "--end-of-options"])
                .arg(commit.as_str())
                .arg("--")
                .args(batch.iter().map(|p| p.as_str()))
                .about_rev(commit.to_string())
                .run_ok()?;
        }
        Ok(())
    }

    /// Which of `paths` `rev` holds a file at — one `ls-tree` per batch of
    /// paths.
    pub fn files_at(&self, rev: &Rev, paths: &[&RepoPath]) -> Result<BTreeSet<String>, GitError> {
        let mut out = BTreeSet::new();
        for batch in batches(paths) {
            out.extend(
                self.tree_entries(rev, batch)?
                    .into_iter()
                    .filter(|e| e.mode.is_blob())
                    .map(|e| e.path.as_str().to_string()),
            );
        }
        Ok(out)
    }

    /// Each of `paths` at `rev` as committed: the blob's own bytes, with no
    /// conversion a checkout would apply — `core.autocrlf`, `.gitattributes`
    /// line endings, `smudge` filters. That is the form a conversation's file
    /// store reads a branch in ([`crate::GitSource`] reads blobs), so it is the
    /// text the conversation's changes were made against, and the base a chain
    /// of them replays onto and a tool's changes are measured from. Paths `rev`
    /// holds no file at are absent from the result. One reader for the call.
    pub fn read_committed_all(
        &self,
        rev: &Rev,
        paths: &[&RepoPath],
    ) -> Result<BTreeMap<String, Vec<u8>>, GitError> {
        let files = self.files_at(rev, paths)?;
        let blobs = self.blobs();
        let mut out = BTreeMap::new();
        for path in files {
            let repo_path = RepoPath::parse(&path)?;
            if let Some(content) = blobs.read_at(rev, &repo_path)? {
                out.insert(path, content);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read::head::Head;
    use crate::testing::TestRepo;

    fn branch(name: &str) -> BranchName {
        BranchName::parse(name).unwrap()
    }

    /// `first` and `second` on `main`, and a branch `old` left at `first`.
    fn two_commits(t: &TestRepo) -> (Oid, Oid) {
        t.write("a.txt", b"first\n");
        t.write("gone.txt", b"only in the first\n");
        let first = t.commit_all("first");
        t.git(&["branch", "old"]);
        t.write("a.txt", b"second\n");
        std::fs::remove_file(t.path.join("gone.txt")).unwrap();
        t.write("b.txt", b"added\n");
        let second = t.commit_all("second");
        (first, second)
    }

    /// **The checkout lands exactly on the branch** — `HEAD` on it, the
    /// tracked files and the index at its commit, local changes discarded —
    /// and no branch moves.
    #[test]
    fn the_checkout_lands_exactly_on_the_branch() {
        let t = TestRepo::init();
        let (first, second) = two_commits(&t);
        t.write("a.txt", b"local edit\n");
        t.git(&["add", "a.txt"]);
        t.write("b.txt", b"unstaged edit\n");

        let repo = t.repo();
        assert_eq!(repo.force_checkout_branch(&branch("old")).unwrap(), first);
        assert_eq!(
            repo.head().unwrap(),
            Head::Branch {
                branch: branch("old"),
                oid: first.clone()
            }
        );
        assert_eq!(t.read("a.txt"), b"first\n");
        assert_eq!(t.read("gone.txt"), b"only in the first\n");
        assert!(
            !t.path.join("b.txt").exists(),
            "a file the commit lacks is removed"
        );
        assert!(
            repo.status().unwrap().is_empty(),
            "index and tree are clean"
        );
        assert_eq!(t.oid("refs/heads/main"), second, "no branch moved");
        assert_eq!(t.oid("refs/heads/old"), first, "no branch moved");

        assert_eq!(repo.force_checkout_branch(&branch("main")).unwrap(), second);
        assert_eq!(t.read("a.txt"), b"second\n");
        assert!(!t.path.join("gone.txt").exists());
    }

    /// **Checking out the branch `HEAD` is already on is a hard reset** —
    /// local changes, staged and unstaged, discarded.
    #[test]
    fn the_same_branch_again_is_a_hard_reset() {
        let t = TestRepo::init();
        let (_, second) = two_commits(&t);
        t.write("a.txt", b"staged\n");
        t.git(&["add", "a.txt"]);
        t.write("b.txt", b"unstaged\n");
        let repo = t.repo();
        repo.force_checkout_branch(&branch("main")).unwrap();
        assert_eq!(t.read("a.txt"), b"second\n");
        assert_eq!(t.read("b.txt"), b"added\n");
        assert!(repo.status().unwrap().is_empty());
        assert_eq!(repo.head().unwrap().oid(), Some(&second));
    }

    /// **Untracked and ignored files are left alone** — the caller removes
    /// the untracked ones it wrote, and an ignored build output survives.
    #[test]
    fn untracked_and_ignored_files_survive() {
        let t = TestRepo::init();
        t.write(".gitignore", b"target/\n");
        two_commits(&t);
        t.write("scratch.txt", b"untracked\n");
        t.write("target/out.bin", b"build output\n");

        t.repo().force_checkout_branch(&branch("old")).unwrap();
        assert_eq!(t.read("scratch.txt"), b"untracked\n");
        assert_eq!(t.read("target/out.bin"), b"build output\n");
    }

    /// **A file already at the commit's content is not rewritten** — its
    /// modification time is what it was — while a changed one is.
    #[test]
    fn an_unchanged_file_keeps_its_timestamp() {
        let t = TestRepo::init();
        two_commits(&t);
        let repo = t.repo();
        repo.force_checkout_branch(&branch("main")).unwrap();
        let old = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000);
        for name in ["a.txt", "b.txt"] {
            std::fs::File::options()
                .write(true)
                .open(t.path.join(name))
                .unwrap()
                .set_modified(old)
                .unwrap();
        }
        // Refresh the index so git's stat information matches the new times,
        // as it would after any earlier checkout.
        t.git(&["update-index", "--refresh"]);
        t.write("a.txt", b"changed\n");

        repo.force_checkout_branch(&branch("main")).unwrap();
        let mtime = |name: &str| {
            std::fs::metadata(t.path.join(name))
                .unwrap()
                .modified()
                .unwrap()
        };
        assert_eq!(mtime("b.txt"), old, "an untouched file was rewritten");
        assert_ne!(mtime("a.txt"), old);
        assert_eq!(t.read("a.txt"), b"second\n");
    }

    /// **A committed read is the blob, whatever the checkout converts** — under
    /// `core.autocrlf=true` and a per-path `eol` rule, a text file reads with
    /// the LF it was committed with, a binary file untouched, awkward names
    /// whole, and a missing path and a folder absent — across more paths than
    /// one command line holds.
    #[test]
    fn a_committed_read_is_the_blob_whatever_the_checkout_converts() {
        let t = TestRepo::init();
        t.write(".gitattributes", b"*.lf eol=lf\n*.crlf eol=crlf\n");
        t.write("text.txt", b"one\ntwo\n");
        t.write("kept.lf", b"one\ntwo\n");
        t.write("made.crlf", b"one\ntwo\n");
        t.write("bin.dat", &[0, 1, b'\n', 2, 255]);
        t.write("dir with space/café ü.txt", b"x\n");
        t.write("empty.txt", b"");
        // 140 names of ~130 characters: more than one command line's batch.
        let long = "a-long-file-name-to-fill-the-command-line".repeat(3);
        let many: Vec<String> = (0..140)
            .map(|i| format!("many/{long}-{i:04}.txt"))
            .collect();
        for (i, name) in many.iter().enumerate() {
            t.write(name, format!("file {i}\n").as_bytes());
        }
        let commit = t.commit_all("base");
        t.git(&["config", "core.autocrlf", "true"]);
        for name in ["text.txt", "made.crlf"] {
            std::fs::remove_file(t.path.join(name)).unwrap();
        }
        t.git(&["checkout", "--", "."]);
        assert_eq!(t.read("text.txt"), b"one\r\ntwo\r\n", "the checkout converts");

        let repo = t.repo();
        let at = Rev::Oid(commit);
        let special: Vec<RepoPath> = [
            "text.txt",
            "kept.lf",
            "made.crlf",
            "bin.dat",
            "dir with space/café ü.txt",
            "empty.txt",
            "absent.txt",
            "dir with space",
        ]
        .iter()
        .map(|s| RepoPath::parse(s).unwrap())
        .collect();
        let bulk: Vec<RepoPath> = many.iter().map(|n| RepoPath::parse(n).unwrap()).collect();
        let refs: Vec<&RepoPath> = special.iter().chain(&bulk).collect();
        let all = repo.read_committed_all(&at, &refs).unwrap();

        for (i, path) in bulk.iter().enumerate() {
            assert_eq!(all[path.as_str()], format!("file {i}\n").into_bytes(), "{path}");
        }
        assert_eq!(all.len(), refs.len() - 2, "the missing path and the folder");
        for name in ["text.txt", "kept.lf", "made.crlf"] {
            assert_eq!(all[name], b"one\ntwo\n", "{name}: as committed");
        }
        assert_eq!(all["bin.dat"], [0, 1, b'\n', 2, 255]);
        assert_eq!(all["dir with space/café ü.txt"], b"x\n");
        assert_eq!(all["empty.txt"], b"");
        assert!(!all.contains_key("absent.txt") && !all.contains_key("dir with space"));
        assert!(repo.read_committed_all(&at, &[]).unwrap().is_empty());
    }

    /// **Only the named paths go back** — edited and deleted ones restored,
    /// every other change left standing, the untouched file's timestamp kept —
    /// and hundreds of paths go through in batches.
    #[test]
    fn restoring_paths_puts_back_only_those() {
        let t = TestRepo::init();
        for i in 0..400 {
            t.write(
                &format!("many/a-rather-long-file-name-number-{i:04}.txt"),
                b"base\n",
            );
        }
        t.write("keep.txt", b"base\n");
        t.write("edit.txt", b"base\n");
        t.write("gone.txt", b"base\n");
        let base = t.commit_all("base");
        let repo = t.repo();

        for i in 0..400 {
            t.write(
                &format!("many/a-rather-long-file-name-number-{i:04}.txt"),
                b"changed\n",
            );
        }
        t.write("keep.txt", b"left alone\n");
        t.write("edit.txt", b"changed\n");
        std::fs::remove_file(t.path.join("gone.txt")).unwrap();

        let mut names: Vec<String> = (0..400)
            .map(|i| format!("many/a-rather-long-file-name-number-{i:04}.txt"))
            .collect();
        names.extend(["edit.txt".to_string(), "gone.txt".to_string()]);
        let paths: Vec<RepoPath> = names.iter().map(|n| RepoPath::parse(n).unwrap()).collect();
        let refs: Vec<&RepoPath> = paths.iter().collect();
        repo.restore_paths(&base, &refs).unwrap();

        for name in &names {
            assert_eq!(t.read(name), b"base\n", "{name}");
        }
        assert_eq!(t.read("keep.txt"), b"left alone\n");
        assert_eq!(
            repo.head().unwrap(),
            Head::Branch {
                branch: branch("main"),
                oid: base
            }
        );
    }

    /// **The checked-out branch is moved, or made again, as a
    /// compare-and-swap** — the working tree untouched — where every other ref
    /// move refuses it.
    #[test]
    fn the_checked_out_branch_is_forced_as_a_compare_and_swap() {
        let t = TestRepo::init();
        let (first, second) = two_commits(&t);
        let repo = t.repo();
        assert!(matches!(
            repo.move_branch(&branch("main"), &second, &first),
            Err(GitError::CheckedOutBranch { .. })
        ));
        repo.force_branch_tip(&branch("main"), Some(&second), &first)
            .unwrap();
        assert_eq!(t.oid("refs/heads/main"), first);
        assert_eq!(
            t.read("a.txt"),
            b"second\n",
            "the working tree is untouched"
        );

        // A stale expectation changes nothing.
        assert!(repo
            .force_branch_tip(&branch("main"), Some(&second), &second)
            .is_err());
        assert_eq!(t.oid("refs/heads/main"), first);

        t.git(&["update-ref", "-d", "refs/heads/main"]);
        repo.force_branch_tip(&branch("main"), None, &second)
            .unwrap();
        assert_eq!(t.oid("refs/heads/main"), second);
        assert!(
            repo.force_branch_tip(&branch("main"), None, &first)
                .is_err(),
            "it exists now"
        );
    }

    /// **Attaching `HEAD` moves only `HEAD`**: the working tree and the index
    /// stay as they were, and read as changes against the new branch.
    #[test]
    fn attaching_head_moves_only_head() {
        let t = TestRepo::init();
        let (first, _) = two_commits(&t);
        t.git(&["checkout", "-q", "--detach"]);
        let repo = t.repo();
        repo.attach_head(&branch("old")).unwrap();
        assert_eq!(
            repo.head().unwrap(),
            Head::Branch {
                branch: branch("old"),
                oid: first
            }
        );
        assert_eq!(
            t.read("a.txt"),
            b"second\n",
            "the working tree is untouched"
        );
        assert!(
            !repo.status().unwrap().is_empty(),
            "second's files are changes"
        );
    }

    /// A branch that does not exist is a typed refusal, and the checkout is
    /// left as it was.
    #[test]
    fn an_unknown_branch_is_refused() {
        let t = TestRepo::init();
        two_commits(&t);
        t.write("a.txt", b"local edit\n");
        assert!(matches!(
            t.repo().force_checkout_branch(&branch("absent")),
            Err(GitError::UnknownRevision { .. })
        ));
        assert_eq!(t.read("a.txt"), b"local edit\n");
    }
}
