//! Bringing a job's local branch up to the commit its conversation pinned,
//! with the checkout locked and its owner's state set aside.

use std::collections::BTreeSet;

use crate::checkout::{CheckoutError, Preserved};
use crate::{BranchName, Followed, Oid, Repo, Rev};

/// Fast-forward `branch` to `to` where origin already holds it and the
/// branch is behind it ([`Repo::follow_record`]). A branch that has moved
/// off `to` since the job was checked fails it.
///
/// When the owner's checkout is on `branch` with changes of its own to
/// files the commits it would follow change too, nothing moves: put back
/// over the branch once it moved on, the owner's copies would undo what it
/// gained there — the case `git merge --ff-only` refuses too.
pub(super) fn follow(
    repo: &Repo,
    preserved: &Preserved,
    branch: &BranchName,
    to: &Oid,
) -> Result<(), CheckoutError> {
    if preserved.is_on(branch) {
        let in_the_way = own_work_in_the_way(repo, preserved, branch, to)?;
        if !in_the_way.is_empty() {
            return Err(CheckoutError::OwnWorkInTheWay {
                branch: branch.to_string(),
                paths: in_the_way,
            });
        }
    }
    match repo.follow_record(branch, to)? {
        Followed::Refused => Err(CheckoutError::BranchMoved {
            branch: branch.to_string(),
        }),
        Followed::Level | Followed::Moved { .. } => Ok(()),
    }
}

/// The paths the owner changed that the commits from `branch`'s tip to `to`
/// change too — none when the branch is not behind `to`.
fn own_work_in_the_way(
    repo: &Repo,
    preserved: &Preserved,
    branch: &BranchName,
    to: &Oid,
) -> Result<Vec<String>, CheckoutError> {
    let Some(tip) = repo.ref_target(&branch.to_ref())? else {
        return Ok(Vec::new());
    };
    let (from, to) = (Rev::Oid(tip), Rev::Oid(to.clone()));
    if from == to || !repo.is_ancestor(&from, &to)? {
        return Ok(Vec::new());
    }
    let gained: BTreeSet<String> = repo
        .diff(&from, &to, &[])?
        .iter()
        .flat_map(|e| e.old.iter().chain(&e.new))
        .map(|side| side.path.as_str().to_string())
        .collect();
    Ok(preserved
        .own_changes()?
        .intersection(&gained)
        .cloned()
        .collect())
}
