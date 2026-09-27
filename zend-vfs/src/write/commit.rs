//! Committing a [`ChangeSet`] onto a base commit, without the working tree
//! or the index.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::changeset::{Change, ChangeSet};
use crate::error::GitError;
use crate::read::tree::ObjectKind;
use crate::types::{FileMode, Oid, RefName, RepoPath, Rev, Signature};
use crate::write::fast_import::{CommitStream, FileOp};
use crate::write::ref_txn::{RefOp, RefTransaction};
use crate::Repo;

/// A ref name no other writer uses, for fast-import to write the commit to
/// before it is read back and deleted.
fn scratch_ref() -> RefName {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    RefName::parse(&format!(
        "refs/zen/scratch/{}-{nanos}-{n}",
        std::process::id()
    ))
    .expect("a scratch ref name is valid")
}

impl Repo {
    /// A new commit whose single parent is `base` and whose tree is `base`'s
    /// with `changes` applied. Moves no branch — publish it with
    /// [`Repo::update_refs`] or [`Repo::push`].
    ///
    /// 1. Each written file becomes a blob through the repository's filters
    ///    ([`Repo::write_blob`]), taking the base's mode when none is given.
    /// 2. One `fast-import` run writes the tree and commit onto a scratch ref.
    /// 3. The commit id is read back and the scratch ref deleted.
    ///
    /// Given the same base, changes, message and signatures, the commit id is
    /// always the same.
    pub fn commit_changes(
        &self,
        base: &Oid,
        changes: &ChangeSet,
        message: &str,
        author: &Signature,
        committer: &Signature,
    ) -> Result<Oid, GitError> {
        let _write = self.write_lock();

        let unmoded: Vec<&RepoPath> = changes
            .iter()
            .filter_map(|(path, change)| match change {
                Change::Write { mode: None, .. } => Some(path),
                _ => None,
            })
            .collect();
        let base_rev = Rev::Oid(base.clone());
        let existing = self.tree_entries(&base_rev, &unmoded)?;

        let mut ops = Vec::with_capacity(changes.len());
        for (path, change) in changes.iter() {
            match change {
                Change::Write { content, mode } => {
                    let inherited = existing
                        .iter()
                        .find(|e| &e.path == path && e.kind == ObjectKind::Blob)
                        .map(|e| e.mode);
                    // Inheriting a symlink's mode would make the written text
                    // the link's target — a link the caller never asked for.
                    if mode.is_none() && inherited == Some(FileMode::Symlink) {
                        return Err(GitError::invalid(format!(
                            "{path} is a symlink in the base; replace it with an explicit \
                             mode or ChangeSet::symlink"
                        )));
                    }
                    let mode = mode.or(inherited).unwrap_or(FileMode::Regular);
                    let oid = self.write_blob_locked(path, content.clone(), mode)?;
                    ops.push(FileOp::Modify {
                        mode,
                        oid,
                        path: path.clone(),
                    });
                }
                Change::Delete => ops.push(FileOp::Delete { path: path.clone() }),
            }
        }

        let target = scratch_ref();
        let stream = CommitStream {
            target: &target,
            author,
            committer,
            message,
            parent: base,
            ops: &ops,
        };
        self.git("fast-import")
            .args(["--quiet", "--done", "--date-format=raw"])
            .stdin(stream.to_bytes())
            .about_rev(base.to_string())
            .run_ok()?;
        let commit = self.ref_target(&target)?.ok_or_else(|| {
            GitError::malformed("fast-import", format!("{target} was not written"))
        })?;
        self.update_refs_locked(&RefTransaction::new().push(RefOp::Delete {
            name: target,
            old: commit.clone(),
        }))?;
        Ok(commit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read::log::LogRange;
    use crate::testing::TestRepo;
    use crate::types::{BranchName, GitTime};

    fn sig() -> Signature {
        Signature::new(
            "Ada Lovelace",
            "ada@example.com",
            GitTime::parse_raw("1700000000 +0100").unwrap(),
        )
        .unwrap()
    }

    fn p(s: &str) -> RepoPath {
        RepoPath::parse(s).unwrap()
    }

    fn base_repo() -> (TestRepo, Oid) {
        let t = TestRepo::init();
        t.write("keep.txt", b"keep\n");
        t.write("edit.txt", b"before\n");
        t.write("gone.txt", b"gone\n");
        t.write("run.sh", b"#!/bin/sh\n");
        t.git(&["add", "-A"]);
        t.git(&["update-index", "--chmod=+x", "run.sh"]);
        t.git(&["commit", "-q", "-m", "base"]);
        let base = t.oid("HEAD");
        (t, base)
    }

    fn changes() -> ChangeSet {
        let mut c = ChangeSet::new();
        c.write(p("edit.txt"), b"after\n".to_vec(), None).unwrap();
        c.write(p("new dir/ünï.txt"), b"new\n".to_vec(), None)
            .unwrap();
        c.write(p("run.sh"), b"#!/bin/sh\necho hi\n".to_vec(), None)
            .unwrap();
        c.delete(p("gone.txt")).unwrap();
        c
    }

    #[test]
    fn the_commit_has_the_base_tree_with_the_changes_applied() {
        let (t, base) = base_repo();
        let repo = t.repo();
        let commit = repo
            .commit_changes(&base, &changes(), "Apply", &sig(), &sig())
            .unwrap();

        let files: Vec<(String, FileMode)> = repo
            .ls_tree(&Rev::Oid(commit.clone()), None)
            .unwrap()
            .into_iter()
            .map(|e| (e.path.to_string(), e.mode))
            .collect();
        assert_eq!(
            files,
            vec![
                ("edit.txt".into(), FileMode::Regular),
                ("keep.txt".into(), FileMode::Regular),
                ("new dir".into(), FileMode::Tree),
                ("run.sh".into(), FileMode::Executable),
            ],
            "gone.txt deleted, run.sh keeps its executable bit"
        );
        let blobs = repo.blobs();
        let at = |path| {
            blobs
                .read_at(&Rev::Oid(commit.clone()), &p(path))
                .unwrap()
                .unwrap()
        };
        assert_eq!(at("edit.txt"), b"after\n");
        assert_eq!(at("new dir/ünï.txt"), b"new\n");
        assert_eq!(at("keep.txt"), b"keep\n");

        let info = &repo
            .log(&LogRange::of(Rev::Oid(commit.clone())), 1)
            .unwrap()[0];
        assert_eq!(info.parents, vec![base]);
        assert_eq!(info.message, "Apply\n");
        assert_eq!(info.author, sig());
        assert_eq!(info.committer, sig());
    }

    /// **The commit id is exactly what `git commit-tree` gives** for the same
    /// tree, parent, identities and message — an independent oracle that pins
    /// every byte of the commit object fast-import wrote.
    #[test]
    fn the_commit_id_matches_an_independent_commit_tree() {
        let (t, base) = base_repo();
        let repo = t.repo();
        let commit = repo
            .commit_changes(&base, &changes(), "Apply\n\nWith a body", &sig(), &sig())
            .unwrap();
        let tree = t.oid(&format!("{commit}^{{tree}}"));
        let oracle = repo
            .commit_tree(&tree, &[&base], "Apply\n\nWith a body", &sig(), &sig())
            .unwrap();
        assert_eq!(commit, oracle);

        let again = repo
            .commit_changes(&base, &changes(), "Apply\n\nWith a body", &sig(), &sig())
            .unwrap();
        assert_eq!(commit, again, "the same inputs give the same commit");
    }

    /// **The user's checkout is untouched**: working tree, index (bytes and
    /// modification time), `HEAD` and every branch, and no scratch ref is
    /// left behind.
    #[test]
    fn committing_leaves_the_working_tree_index_head_and_refs_alone() {
        let (t, base) = base_repo();
        t.write("edit.txt", b"user's uncommitted edit\n");
        t.write("untracked.txt", b"mine\n");
        let index = t.path.join(".git/index");
        let index_bytes = std::fs::read(&index).unwrap();
        let index_mtime = std::fs::metadata(&index).unwrap().modified().unwrap();
        let head_file = std::fs::read(t.path.join(".git/HEAD")).unwrap();
        let refs_before = t.git(&["for-each-ref"]);

        let repo = t.repo();
        repo.commit_changes(&base, &changes(), "Apply", &sig(), &sig())
            .unwrap();

        assert_eq!(t.read("edit.txt"), b"user's uncommitted edit\n");
        assert_eq!(t.read("untracked.txt"), b"mine\n");
        assert_eq!(t.read("gone.txt"), b"gone\n");
        assert!(!t.path.join("new dir").exists());
        assert_eq!(std::fs::read(&index).unwrap(), index_bytes);
        assert_eq!(
            std::fs::metadata(&index).unwrap().modified().unwrap(),
            index_mtime
        );
        assert_eq!(std::fs::read(t.path.join(".git/HEAD")).unwrap(), head_file);
        assert_eq!(t.git(&["for-each-ref"]), refs_before);
        assert_eq!(t.oid("HEAD"), base);
    }

    #[test]
    fn an_explicit_mode_overrides_the_base() {
        let (t, base) = base_repo();
        let repo = t.repo();
        let mut c = ChangeSet::new();
        c.write(p("run.sh"), b"x\n".to_vec(), Some(FileMode::Regular))
            .unwrap();
        c.write(p("tool"), b"x\n".to_vec(), Some(FileMode::Executable))
            .unwrap();
        let commit = repo
            .commit_changes(&base, &c, "modes", &sig(), &sig())
            .unwrap();
        let rev = Rev::Oid(commit);
        let (run, tool) = (p("run.sh"), p("tool"));
        let entries = repo.tree_entries(&rev, &[&run, &tool]).unwrap();
        assert_eq!(entries[0].mode, FileMode::Regular);
        assert_eq!(entries[1].mode, FileMode::Executable);
    }

    /// **A plain write over a base symlink is refused**, not committed as a
    /// link to whatever the written text says. An explicit mode replaces the
    /// link with a file; an explicit symlink call writes a link.
    #[test]
    fn writing_over_a_symlink_needs_an_explicit_mode() {
        let (t, base) = base_repo();
        let repo = t.repo();
        let mut link = ChangeSet::new();
        link.symlink(p("link"), "keep.txt").unwrap();
        let with_link = repo
            .commit_changes(&base, &link, "link", &sig(), &sig())
            .unwrap();
        let rev = Rev::Oid(with_link.clone());
        let l = p("link");
        assert_eq!(
            repo.tree_entries(&rev, &[&l]).unwrap()[0].mode,
            FileMode::Symlink
        );

        let mut plain = ChangeSet::new();
        plain
            .write(p("link"), b"../../.zend/secrets.yaml".to_vec(), None)
            .unwrap();
        assert!(matches!(
            repo.commit_changes(&with_link, &plain, "x", &sig(), &sig()),
            Err(GitError::InvalidInput(_))
        ));

        let mut replace = ChangeSet::new();
        replace
            .write(p("link"), b"now a file\n".to_vec(), Some(FileMode::Regular))
            .unwrap();
        let replaced = repo
            .commit_changes(&with_link, &replace, "x", &sig(), &sig())
            .unwrap();
        assert_eq!(
            repo.tree_entries(&Rev::Oid(replaced), &[&l]).unwrap()[0].mode,
            FileMode::Regular
        );
    }

    /// The commit can be published as a branch and read back through it.
    #[test]
    fn a_committed_change_set_publishes_as_a_branch() {
        let (t, base) = base_repo();
        let repo = t.repo();
        let commit = repo
            .commit_changes(&base, &changes(), "Apply", &sig(), &sig())
            .unwrap();
        let branch = BranchName::parse("zen/purge").unwrap();
        repo.create_branch(&branch, &commit).unwrap();
        assert_eq!(repo.resolve(&Rev::Branch(branch)).unwrap(), commit);
    }

    #[test]
    fn an_unknown_base_is_an_error() {
        let (t, _) = base_repo();
        let missing = Oid::parse(&"1".repeat(40)).unwrap();
        assert!(t
            .repo()
            .commit_changes(&missing, &changes(), "x", &sig(), &sig())
            .is_err());
    }
}
