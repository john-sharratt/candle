//! Bringing a job's local branch onto the commit its conversation pinned,
//! with the checkout locked and its owner's state set aside.

use std::collections::BTreeSet;

use crate::checkout::{CheckoutError, Preserved};
use crate::read::record::LocalBranch;
use crate::{BranchName, Followed, Oid, Repo, Rev};

/// Put `branch` on `to`, the commit on origin's copy the conversation
/// pinned, for the job:
///
/// - a branch holding commits origin never had — its owner's, never pushed —
///   is set aside ([`Preserved::set_branch_aside`]): moved onto `to` for the
///   job and put back at its own commit with the rest of the checkout, so
///   the job never builds on them and nothing of them is lost or published;
/// - a branch behind `to` is fast-forwarded to it ([`Repo::follow_record`]),
///   and keeps what it gained.
///
/// A branch that has moved off what the job was checked against fails it.
/// When the owner's checkout is on a branch it would fast-forward, with
/// changes of its own to files the commits it would follow change too,
/// nothing moves: put back over the branch once it moved on, the owner's
/// copies would undo what it gained there — the case `git merge --ff-only`
/// refuses too.
pub(super) fn follow(
    repo: &Repo,
    preserved: &mut Preserved,
    branch: &BranchName,
    to: &Oid,
) -> Result<(), CheckoutError> {
    if let Some(local) = repo.local_branch(branch)?.filter(LocalBranch::unpushed) {
        return set_aside(repo, preserved, branch, &local, to);
    }
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
        Followed::Refused => Err(moved(branch)),
        Followed::Level | Followed::Moved { .. } => Ok(()),
    }
}

/// Set the owner's unpushed commits on `branch` aside for the job — only
/// onto a commit origin holds that what origin has of the branch leads to,
/// as the job was checked against.
fn set_aside(
    repo: &Repo,
    preserved: &mut Preserved,
    branch: &BranchName,
    local: &LocalBranch,
    to: &Oid,
) -> Result<(), CheckoutError> {
    let leads_to = match &local.on_record {
        Some(on_record) => {
            on_record == to
                || repo.is_ancestor(&Rev::Oid(on_record.clone()), &Rev::Oid(to.clone()))?
        }
        None => true,
    };
    if !leads_to || !repo.on_record(branch, to)? {
        return Err(moved(branch));
    }
    preserved.set_branch_aside(branch, &local.tip, to)
}

fn moved(branch: &BranchName) -> CheckoutError {
    CheckoutError::BranchMoved {
        branch: branch.to_string(),
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
