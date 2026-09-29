//! Cherry-pick and revert, computed in the object store.
//!
//! Both are three-way merges ([`Repo::merge_trees`]): a cherry-pick applies
//! the change from a commit's parent to the commit onto another commit; a
//! revert applies the change from the commit back to its parent. Neither
//! touches the working tree, the index or any ref — the result is a commit
//! for the caller to publish.

use crate::error::GitError;
use crate::read::log::{CommitInfo, LogRange};
use crate::types::{Oid, RepoPath, Rev, Signature};
use crate::write::merge_tree::MergeOutcome;
use crate::Repo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PickOutcome {
    /// The new commit, on top of `onto`.
    Clean(Oid),
    /// The change does not apply cleanly; nothing was committed.
    Conflicted { paths: Vec<RepoPath> },
}

impl Repo {
    /// `commit` with its single parent, refusing root and merge commits —
    /// the change a merge introduces has no single base to replay from.
    fn single_parent(&self, commit: &Oid) -> Result<(CommitInfo, Oid), GitError> {
        let info = self
            .log(&LogRange::of(Rev::Oid(commit.clone())), 1)?
            .pop()
            .ok_or_else(|| GitError::UnknownRevision {
                rev: commit.to_string(),
            })?;
        match info.parents.as_slice() {
            [parent] => {
                let parent = parent.clone();
                Ok((info, parent))
            }
            [] => Err(GitError::invalid(format!(
                "{commit} is a root commit; it has no change to replay"
            ))),
            _ => Err(GitError::invalid(format!(
                "{commit} is a merge commit; it has no single change to replay"
            ))),
        }
    }

    /// Replay `commit`'s change onto `onto`, keeping its author and message
    /// as `git cherry-pick` does, with `committer` as the committer.
    pub fn cherry_pick(
        &self,
        commit: &Oid,
        onto: &Oid,
        committer: &Signature,
    ) -> Result<PickOutcome, GitError> {
        let (info, parent) = self.single_parent(commit)?;
        match self.merge_trees(&parent, onto, commit)? {
            MergeOutcome::Clean { tree } => Ok(PickOutcome::Clean(self.commit_tree(
                &tree,
                &[onto],
                &info.message,
                &info.author,
                committer,
            )?)),
            MergeOutcome::Conflicted { paths } => Ok(PickOutcome::Conflicted { paths }),
        }
    }

    /// Undo `commit`'s change on top of `onto`, with `git revert`'s message.
    pub fn revert(
        &self,
        commit: &Oid,
        onto: &Oid,
        author: &Signature,
        committer: &Signature,
    ) -> Result<PickOutcome, GitError> {
        let (info, parent) = self.single_parent(commit)?;
        let message = format!(
            "Revert \"{}\"\n\nThis reverts commit {commit}.\n",
            info.subject()
        );
        match self.merge_trees(commit, onto, &parent)? {
            MergeOutcome::Clean { tree } => Ok(PickOutcome::Clean(self.commit_tree(
                &tree,
                &[onto],
                &message,
                author,
                committer,
            )?)),
            MergeOutcome::Conflicted { paths } => Ok(PickOutcome::Conflicted { paths }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{TestRepo, SETUP_DATE};
    use crate::types::GitTime;

    fn setup_sig() -> Signature {
        Signature::new(
            "Setup",
            "setup@example.com",
            GitTime::parse_raw(SETUP_DATE).unwrap(),
        )
        .unwrap()
    }

    /// From a common base, `main` edits a.txt and a side branch edits b.txt.
    /// Returns the side commit to pick and main's head.
    fn history() -> (TestRepo, Oid, Oid) {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        t.write("b.txt", b"b\n");
        t.commit_all("base");
        t.git(&["checkout", "-q", "-b", "side"]);
        t.write("b.txt", b"side\n");
        let pick = t.commit_all("side change\n\nWith a body.");
        t.git(&["checkout", "-q", "main"]);
        t.write("a.txt", b"main\n");
        let main = t.commit_all("main change");
        (t, pick, main)
    }

    /// **A cherry-pick is the commit `git cherry-pick` makes** — same id —
    /// with the same committer and time.
    #[test]
    fn a_cherry_pick_matches_git_cherry_pick() {
        let (t, pick, main) = history();
        let repo = t.repo();
        let ours = match repo.cherry_pick(&pick, &main, &setup_sig()).unwrap() {
            PickOutcome::Clean(c) => c,
            other => panic!("{other:?}"),
        };
        // The oracle, on a throwaway branch at main.
        t.git(&["checkout", "-q", "-b", "oracle", main.as_str()]);
        t.git(&["cherry-pick", pick.as_str()]);
        assert_eq!(ours, t.oid("HEAD"));
    }

    /// **A revert is the commit `git revert` makes** — same id.
    #[test]
    fn a_revert_matches_git_revert() {
        let (t, _, main) = history();
        let repo = t.repo();
        let ours = match repo
            .revert(&main, &main, &setup_sig(), &setup_sig())
            .unwrap()
        {
            PickOutcome::Clean(c) => c,
            other => panic!("{other:?}"),
        };
        t.git(&["checkout", "-q", "-b", "oracle", main.as_str()]);
        t.git(&["revert", "--no-edit", main.as_str()]);
        assert_eq!(ours, t.oid("HEAD"));
    }

    #[test]
    fn picking_neither_moves_a_ref_nor_touches_the_checkout() {
        let (t, pick, main) = history();
        let refs = t.git(&["for-each-ref"]);
        let index = std::fs::read(t.path.join(".git/index")).unwrap();
        t.repo().cherry_pick(&pick, &main, &setup_sig()).unwrap();
        assert_eq!(t.git(&["for-each-ref"]), refs);
        assert_eq!(std::fs::read(t.path.join(".git/index")).unwrap(), index);
        assert_eq!(t.read("b.txt"), b"b\n");
    }

    #[test]
    fn a_conflicting_pick_reports_its_paths_and_commits_nothing() {
        let t = TestRepo::init();
        t.write("f.txt", b"base\n");
        t.commit_all("base");
        t.git(&["checkout", "-q", "-b", "side"]);
        t.write("f.txt", b"side\n");
        let pick = t.commit_all("side");
        t.git(&["checkout", "-q", "main"]);
        t.write("f.txt", b"main\n");
        let main = t.commit_all("main");
        assert_eq!(
            t.repo().cherry_pick(&pick, &main, &setup_sig()).unwrap(),
            PickOutcome::Conflicted {
                paths: vec![RepoPath::parse("f.txt").unwrap()]
            }
        );
    }

    #[test]
    fn root_and_merge_commits_are_refused() {
        let t = TestRepo::init();
        t.write("a", b"a\n");
        let root = t.commit_all("root");
        t.git(&["checkout", "-q", "-b", "side"]);
        t.write("s", b"s\n");
        t.commit_all("side");
        t.git(&["checkout", "-q", "main"]);
        t.write("m", b"m\n");
        t.commit_all("main");
        t.git(&["merge", "-q", "--no-edit", "side"]);
        let merge = t.oid("HEAD");
        let repo = t.repo();
        for c in [&root, &merge] {
            assert!(matches!(
                repo.cherry_pick(c, &merge, &setup_sig()),
                Err(GitError::InvalidInput(_))
            ));
        }
    }
}
