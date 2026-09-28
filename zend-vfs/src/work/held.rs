//! Keeping a merge's settled tree alive while it is being finished.
//!
//! A merge that is being finished has a base whose tree no commit holds yet
//! — nothing in the repository reaches it, so git would prune it in time,
//! and the conversation could no longer read its own files. [`hold`] keeps
//! it reachable by a ref of the layer's own under
//! `refs/zend/merging/<tree>/`, to a commit of that tree with the merge's
//! parents; [`release`] lets one hold go once the merge is committed,
//! replaced, or dropped.
//!
//! Holds are counted, one ref each: two conversations that settle the same
//! tree each hold it, and the tree stays reachable until both have let go.
//! Every hold is balanced by one release; a release that finds nothing to
//! let go leaves everything as it is, so a miscount only ever keeps a tree
//! too long, never drops one still in use.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::vfs::Base;
use crate::{GitError, GitTime, Oid, RefName, RefOp, RefTransaction, Repo, Signature};

/// How many times a hold or release retries after another writer's ref
/// update raced it.
const ATTEMPTS: usize = 5;

/// The folder of refs holding `base`'s tree, while it is a merge being
/// finished.
fn folder(base: &Base) -> Option<(String, &Oid)> {
    match (&base.tree, base.merging()) {
        (Some(tree), Some(_)) => Some((format!("refs/zend/merging/{tree}/"), tree)),
        _ => None,
    }
}

/// Hold `base`'s tree, when it is a merge being finished: one more hold,
/// whatever holds it already.
pub fn hold(repo: &Repo, base: &Base) -> Result<(), GitError> {
    let Some((folder, tree)) = folder(base) else {
        return Ok(());
    };
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let me = Signature::new(
        "zend",
        "zend@localhost",
        GitTime {
            seconds,
            offset_minutes: 0,
        },
    )?;
    let parents: Vec<&Oid> = base.parents.iter().collect();
    let commit = repo.commit_tree(tree, &parents, "zend: a merge being finished", &me, &me)?;
    let mut last = None;
    for _ in 0..ATTEMPTS {
        let name = RefName::parse(&format!("{folder}{:016x}", rand::random::<u64>()))?;
        match repo.update_refs(&RefTransaction::new().push(RefOp::Create {
            name,
            new: commit.clone(),
        })) {
            Ok(()) => return Ok(()),
            Err(e) => last = Some(e),
        }
    }
    Err(last.expect("at least one attempt"))
}

/// Let go of one hold [`hold`] took for `base`, if it holds any.
///
/// Called once the base has already moved on, so a failure is reported and
/// never returned: the move stands, and a hold that could not be let go only
/// keeps a tree longer than it is needed.
pub fn release(repo: &Repo, base: &Base) {
    if let Err(e) = release_one(repo, base) {
        tracing::warn!(tree = ?base.tree, "a merge's tree could not be let go: {e}");
    }
}

fn release_one(repo: &Repo, base: &Base) -> Result<(), GitError> {
    let Some((folder, _)) = folder(base) else {
        return Ok(());
    };
    let mut last = None;
    for _ in 0..ATTEMPTS {
        let Some((name, old)) = repo.refs_under(&folder)?.into_iter().next() else {
            return Ok(());
        };
        match repo.update_refs(&RefTransaction::new().push(RefOp::Delete { name, old })) {
            Ok(()) => return Ok(()),
            // Another release took this one first: take the next.
            Err(e) => last = Some(e),
        }
    }
    Err(last.expect("at least one attempt"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestRepo;

    fn holds(t: &TestRepo, tree: &Oid) -> usize {
        t.git(&[
            "for-each-ref",
            "--format=%(refname)",
            &format!("refs/zend/merging/{tree}/"),
        ])
        .lines()
        .count()
    }

    /// **A merge's tree is held while it is being finished and let go
    /// after**; a base that is no merge holds nothing.
    #[test]
    fn a_merges_tree_is_held_until_let_go() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        let first = t.commit_all("first");
        t.write("a.txt", b"b\n");
        let second = t.commit_all("second");
        let tree = Oid::parse(t.git(&["rev-parse", "HEAD^{tree}"]).trim()).unwrap();
        let repo = t.repo();

        let plain = Base::at(second.clone(), tree.clone());
        hold(&repo, &plain).unwrap();
        assert_eq!(t.git(&["for-each-ref", "refs/zend/"]), "");

        let merging = Base {
            tree: Some(tree.clone()),
            parents: vec![second, first],
        };
        hold(&repo, &merging).unwrap();
        assert_eq!(holds(&t, &tree), 1);
        let held = t.git(&["for-each-ref", "--format=%(refname)", "refs/zend/"]);
        assert_eq!(
            t.git(&["rev-parse", &format!("{}^{{tree}}", held.trim())])
                .trim(),
            tree.as_str()
        );
        release(&repo, &merging);
        assert_eq!(t.git(&["for-each-ref", "refs/zend/"]), "");
        release(&repo, &merging);
        assert_eq!(t.git(&["for-each-ref", "refs/zend/"]), "");
    }

    /// **Two holds of one tree are two**: one conversation letting go of a
    /// tree another is still finishing a merge on leaves it held.
    #[test]
    fn a_tree_two_merges_settle_stays_held_until_both_let_go() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        let first = t.commit_all("first");
        t.write("a.txt", b"b\n");
        let second = t.commit_all("second");
        let tree = Oid::parse(t.git(&["rev-parse", "HEAD^{tree}"]).trim()).unwrap();
        let repo = t.repo();
        let merging = Base {
            tree: Some(tree.clone()),
            parents: vec![second, first],
        };
        hold(&repo, &merging).unwrap();
        hold(&repo, &merging).unwrap();
        assert_eq!(holds(&t, &tree), 2);
        release(&repo, &merging);
        assert_eq!(holds(&t, &tree), 1, "the other conversation's hold stays");
        release(&repo, &merging);
        assert_eq!(holds(&t, &tree), 0);
    }
}
