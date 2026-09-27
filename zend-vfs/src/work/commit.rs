//! Committing a conversation's work: one commit, published whole or not at
//! all.

use super::three_way::ThreeWay;
use super::{files, held};
use crate::origin::{Published, Pulled};
use crate::vfs::{Base, Carried};
use crate::write::merge_text::MergeLabels;
use crate::{BranchName, GitError, Oid, Rejection, Repo, Rev, Signature, VfsStore};

/// Why a commit of the conversation's work was not made. Nothing was
/// written to origin or to the conversation's copy; at most the local
/// branch, a cache of origin's, was brought up to what origin holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotCommitted {
    /// These paths are still in conflict from a merge.
    Conflicts(Vec<String>),
    /// The branch holds commits the conversation does not have — `record`,
    /// on origin or, with no origin, locally. They meet in a merge first.
    Behind { record: Oid },
    /// Origin refused the push.
    Refused(Rejection),
}

/// A commit that landed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Landed {
    pub commit: Oid,
    /// Its parents: the conversation's commit, and the one merged into it
    /// when the commit finished a merge.
    pub parents: Vec<Oid>,
    /// Where it is kept.
    pub published: Published,
    /// Why the conversation's files still read from the commit before it,
    /// when they could not be moved onto this one; the commit stands
    /// either way, and a merge brings the files up to it.
    pub behind: Option<String>,
}

/// A commit of the conversation's work, between checking its branch and
/// landing it.
///
/// [`Committing::begin`] checks what can be checked before anything is
/// built: no conflict left, and the branch — as origin holds it — holding
/// nothing the conversation's base lacks. The caller builds its commit on
/// [`Committing::onto`], however its content is sourced, and
/// [`Committing::land`] publishes it: pushed to origin under a lease on what
/// the check found, so a push that meets anything new is refused and nothing
/// moves. Only once origin has it do the local branch and the conversation's
/// base follow.
#[derive(Debug)]
pub struct Committing {
    pulled: Pulled,
    base: Base,
    onto: Oid,
}

impl Committing {
    /// Check that `store`'s work can be committed on `branch`, or say why
    /// not.
    pub fn begin(
        repo: &Repo,
        store: &VfsStore,
        branch: &BranchName,
    ) -> Result<Result<Self, NotCommitted>, GitError> {
        let conflicts = store.conflicts();
        if !conflicts.is_empty() {
            return Ok(Err(NotCommitted::Conflicts(conflicts)));
        }
        let base = store
            .base()
            .map_err(files)?
            .ok_or_else(|| GitError::invalid("these files are not read from a branch"))?;
        // A branch origin has let go of is published again, as `git push`
        // would: nothing it held is lost, and the commit brings it back.
        let pulled = repo.pull_branch(branch)?;
        if let Some(record) = pulled.record() {
            let r = Rev::Oid(record.clone());
            let mut has = false;
            for parent in &base.parents {
                if repo.is_ancestor(&r, &Rev::Oid(parent.clone()))? {
                    has = true;
                    break;
                }
            }
            if !has {
                return Ok(Err(NotCommitted::Behind {
                    record: record.clone(),
                }));
            }
        }
        // A merge being finished has no one commit holding its tree, and a
        // branch with no commit yet has none at all: the content is built on
        // a stand-in with that tree — the empty one for a first commit — and
        // committed with the right parents by `land`.
        let stand_in = |tree: &Oid, what: &str| -> Result<Oid, GitError> {
            let me = repo.identity()?;
            repo.commit_tree(tree, &[], what, &me, &me)
        };
        let onto = match (base.merging(), &base.tree, base.commit()) {
            (Some(_), Some(tree), _) => stand_in(tree, "merge")?,
            (_, _, Some(commit)) => commit.clone(),
            (_, _, None) => stand_in(&repo.format().empty_tree(), "first")?,
        };
        Ok(Ok(Self { pulled, base, onto }))
    }

    /// Whether this is the branch's first commit.
    fn first(&self) -> bool {
        self.base.commit().is_none()
    }

    /// The commit a new commit is built on: one whose tree is the
    /// conversation's base.
    pub fn onto(&self) -> &Oid {
        &self.onto
    }

    /// Whether this commit finishes a merge.
    pub fn finishes_merge(&self) -> bool {
        self.base.merging().is_some()
    }

    /// Publish `built` — a commit on [`Self::onto`] — as the branch's next
    /// commit, then carry `store` onto it, three ways: what it committed
    /// reads from the branch, whatever of the conversation's work it did not
    /// take stays, and where both changed a file — a patch over an edit —
    /// both are kept. Finishing a merge, the commit is written again with the
    /// merge's parents under `message`, `author` and `committer`.
    ///
    /// Once origin has the commit it has landed, whatever follows: a store
    /// that cannot be carried onto it is left where it was, and
    /// [`Landed::behind`] says why.
    pub fn land(
        self,
        repo: &Repo,
        store: &VfsStore,
        built: &Oid,
        message: &str,
        author: &Signature,
        committer: &Signature,
    ) -> Result<Result<Landed, NotCommitted>, GitError> {
        let blobs = repo.blobs();
        let tree_of = |commit: &Oid| -> Result<Oid, GitError> {
            blobs
                .commit_of(&Rev::Oid(commit.clone()))?
                .map(|(_, tree)| tree)
                .ok_or_else(|| GitError::invalid(format!("no commit {commit}")))
        };
        let commit = if self.finishes_merge() || self.first() {
            let parents: Vec<&Oid> = self.base.parents.iter().collect();
            repo.commit_tree(&tree_of(built)?, &parents, message, author, committer)?
        } else {
            built.clone()
        };
        let published = repo.publish_commit(&self.pulled, &commit)?;
        match published {
            Published::Behind => {
                let record = self.pulled.record().cloned().unwrap_or(commit);
                return Ok(Err(NotCommitted::Behind { record }));
            }
            Published::Refused(why) => return Ok(Err(NotCommitted::Refused(why))),
            Published::Pushed(_) | Published::Local => {}
        }
        let three = ThreeWay::new(
            repo,
            MergeLabels {
                ours: "yours",
                base: "before the commit",
                theirs: "committed",
            },
        );
        let behind = tree_of(&commit)
            .map_err(|e| e.to_string())
            .and_then(|tree| {
                store
                    .move_base(
                        None,
                        Base::at(commit.clone(), tree),
                        &[],
                        &mut |c: &Carried<'_>| three.carried(c),
                    )
                    .map_err(|e| e.to_string())
            })
            .err();
        if behind.is_none() {
            // The merge's tree is in the commit now.
            held::release(repo, &self.base);
        }
        let parents = if self.finishes_merge() || self.first() {
            self.base.parents
        } else {
            vec![self.onto]
        };
        Ok(Ok(Landed {
            commit,
            parents,
            published,
            behind,
        }))
    }
}
