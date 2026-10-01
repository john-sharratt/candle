//! Taking a checkout's git state back from a tool that changed it.
//!
//! A run puts the checkout on a branch at a known commit. A tool that runs
//! git — through a shell, a build script, a hook of its own — can move that
//! state: commit (the branch moves), switch branch or detach (`HEAD` moves),
//! delete the branch. None of it touches what the tool changed in the files;
//! it changes what those changes are measured against. [`reclaim`] puts the
//! branch back at the commit and `HEAD` back on the branch, touching neither
//! the index nor the working tree, so that everything the tool did — its
//! commits included — shows as a difference from the commit the run started
//! at, and is read back like any other change.
//!
//! Other refs the tool made — a new branch, a tag — are left as they are:
//! they point at objects, and change nothing about the checkout.

use super::error::CheckoutError;
use crate::{BranchName, GitError, Head, Oid, Repo, Rev};

/// Put `branch` back at `base` and `HEAD` — which the caller has read as
/// `head` — back on `branch`, leaving the index and the working tree alone.
/// Each ref move is a compare-and-swap against what the tool left. Returns
/// whether anything moved.
pub(crate) fn reclaim(
    repo: &Repo,
    branch: &BranchName,
    base: &Oid,
    head: &Head,
) -> Result<bool, CheckoutError> {
    if matches!(head, Head::Branch { branch: on, oid } if on == branch && oid == base) {
        // Where the run left it: nothing moved.
        return Ok(false);
    }
    match repo.resolve(&Rev::Branch(branch.clone())) {
        Ok(tip) if &tip == base => {}
        Ok(tip) => repo.force_branch_tip(branch, Some(&tip), base)?,
        Err(GitError::UnknownRevision { .. }) => repo.force_branch_tip(branch, None, base)?,
        Err(e) => return Err(e.into()),
    }
    if head.branch() != Some(branch) {
        repo.attach_head(branch)?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestRepo;

    fn main() -> BranchName {
        BranchName::parse("main").unwrap()
    }

    /// Two commits on `main`, checked out at the first: the "tool" then moves
    /// things, and `reclaim` must put `main` and `HEAD` back at the first.
    fn repo() -> (TestRepo, Oid) {
        let t = TestRepo::init();
        t.write("a.txt", b"one\n");
        let base = t.commit_all("base");
        (t, base)
    }

    /// Reclaim with `HEAD` as it stands; whether anything moved.
    fn reclaim_now(t: &TestRepo, base: &Oid) -> bool {
        let repo = t.repo();
        let head = repo.head().unwrap();
        reclaim(&repo, &main(), base, &head).unwrap()
    }

    fn assert_on_base(t: &TestRepo, base: &Oid) {
        assert_eq!(
            t.repo().head().unwrap(),
            Head::Branch {
                branch: main(),
                oid: base.clone()
            }
        );
    }

    /// **A commit the tool made is taken off the branch**, and what it held
    /// is left in the index and the working tree as a change.
    #[test]
    fn a_commit_is_taken_back_off_the_branch() {
        let (t, base) = repo();
        t.write("a.txt", b"two\n");
        t.commit_all("tool");
        assert!(reclaim_now(&t, &base));
        assert_on_base(&t, &base);
        assert_eq!(t.read("a.txt"), b"two\n");
        assert_eq!(
            t.repo().status().unwrap().len(),
            1,
            "the commit is a change"
        );
    }

    /// **`HEAD` moved to another branch, or detached, is put back** — the
    /// other branch left where the tool put it.
    #[test]
    fn head_is_put_back_on_the_branch() {
        let (t, base) = repo();
        t.git(&["checkout", "-q", "-b", "other"]);
        t.write("a.txt", b"other\n");
        let other = t.commit_all("other");
        assert!(reclaim_now(&t, &base));
        assert_on_base(&t, &base);
        assert_eq!(t.oid("refs/heads/other"), other);

        t.git(&["checkout", "-q", "--detach"]);
        assert!(reclaim_now(&t, &base));
        assert_on_base(&t, &base);
    }

    /// **A deleted branch is made again** at the commit.
    #[test]
    fn a_deleted_branch_is_made_again() {
        let (t, base) = repo();
        t.git(&["update-ref", "-d", "refs/heads/main"]);
        assert!(reclaim_now(&t, &base));
        assert_on_base(&t, &base);
    }

    /// Nothing moved: nothing is written.
    #[test]
    fn an_untouched_checkout_is_left_alone() {
        let (t, base) = repo();
        assert!(!reclaim_now(&t, &base));
        assert_on_base(&t, &base);
        assert!(t.repo().status().unwrap().is_empty());
    }
}
