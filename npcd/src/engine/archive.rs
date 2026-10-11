//! The merge table's acts: `merge_status`, `merge_publish` and `merge_sync`
//! (`docs/replicated_substrate.md` §10.3).
//!
//! Each acts on the operation whose publish step the body carries — the step
//! that checks `published` — and on nothing else: a Maker on any other step,
//! or on none, is told there is nothing here for it. The git work itself is
//! [`crate::opstore::merge`]'s; this is what the Maker is told, and what the
//! operation records of it.
//!
//! **A publish that cannot reach git settles nothing.** `origin` down, or no
//! agent to push with, puts the Maker's mission down without a report and has
//! the table offer the step again later; the store keeps the work meanwhile.

use std::sync::Arc;

use tokio::runtime::{Handle, RuntimeFlavor};

use crate::api::now_ms;
use crate::engine::body::Outcome;
use crate::engine::workflow::Taker;
use crate::opstore::merge::follow::Followed;
use crate::opstore::merge::publish::Publish;
use crate::opstore::merge::sync::Synced;
use crate::opstore::owner::Owner;
use crate::opstore::MindStores;
use crate::sim::operations::Operation;
use crate::world::Hosted;

/// What a body is told at the merge table with no publish step to take.
pub const NO_PUBLISH: &str = "You carry no operation's publish step, so there is nothing of \
                              yours to land here. Work reaches the record through an \
                              operation's last step, once every check has passed.";

/// Perform the merge table's act `tool` for `body`.
pub fn perform(hosted: &Hosted, body: &str, tool: &str) -> Outcome {
    let Some((id, op)) = publish_step(hosted, body) else {
        return Outcome::Refused(NO_PUBLISH.into());
    };
    let Some(stores) = hosted.op_stores().cloned() else {
        return Outcome::Refused("There are no documents in this world to publish.".into());
    };
    let owner = Owner::operation(hosted.id(), id);
    match tool {
        "merge_status" => match blocking(|| stores.merge(&owner, |m| m.status())) {
            Some(Ok(Ok(status))) => Outcome::Did(status.render()),
            Some(Ok(Err(why)) | Err(why)) => Outcome::Refused(why),
            None => Outcome::Refused(NO_MIND.into()),
        },
        "merge_sync" => match blocking(|| stores.merge(&owner, |m| m.sync())) {
            Some(Ok(synced)) => synced_outcome(synced),
            Some(Err(why)) => Outcome::Refused(why),
            None => Outcome::Refused(NO_MIND.into()),
        },
        _ => publish(hosted, body, &stores, &op),
    }
}

/// The operation whose publish step `body` carries, if it carries one.
fn publish_step(hosted: &Hosted, body: &str) -> Option<(u64, Operation)> {
    hosted.sim(|s| {
        let (id, step) = s.missions.active(body)?.operation()?;
        let ops = s.missions.operations();
        let op = ops.get(id)?;
        ops.workflow_of(op)?
            .step(step)?
            .publishes()
            .then(|| (id, op.clone()))
    })
}

/// Why a daemon with no mind folder publishes nothing.
const NO_MIND: &str = "This daemon has no mind repository, so there is no record to publish to.";

fn publish(hosted: &Hosted, body: &str, stores: &Arc<MindStores>, op: &Operation) -> Outcome {
    let message = commit_message(hosted.id(), op);
    let owner = Owner::operation(hosted.id(), op.id);
    let published = match blocking(|| stores.merge(&owner, |m| (m.publish(&message), m.base()))) {
        Some(Ok(published)) => published,
        Some(Err(why)) => return Outcome::Refused(why),
        None => return Outcome::Refused(NO_MIND.into()),
    };
    match published {
        (Publish::Landed { commit, paths }, _) => {
            hosted.with_sim(|s| s.missions.published(op.id, &commit));
            tracing::info!(operation = %op.name, %commit, ?paths, "operation PUBLISHED");
            bring_up(stores);
            Outcome::Did(format!(
                "It is in the record now, as {}: {}. Report it done at the table.",
                short(&commit),
                paths.join(", ")
            ))
        }
        (Publish::Nothing, base) => {
            let at = base.map(|b| b.to_string()).unwrap_or_default();
            hosted.with_sim(|s| s.missions.published(op.id, &at));
            Outcome::Did(
                "Nothing in it differs from the record: the record says it already. Report it \
                 done at the table."
                    .into(),
            )
        }
        (Publish::Conflicts(paths), _) => Outcome::Refused(format!(
            "The record has moved under this work at the lines it changes, in {}. `merge_sync` \
             brings the record's text in with each overlap marked; settle every mark at the \
             bench, `bench_commit`, and publish again.",
            paths.join(", ")
        )),
        (Publish::Unsettled(paths), _) => Outcome::Refused(format!(
            "{} still hold the merge's marks. Settle each at the bench — keep what this work \
             was verified to say and what the record has gained — `bench_commit`, and publish \
             again.",
            paths.join(", ")
        )),
        (Publish::Unreachable(why), _) => {
            hosted.with_sim(|s| s.missions.publish_waits(body, &why, now_ms()));
            tracing::info!(operation = %op.name, %why, "operation publish waits: git unreachable");
            Outcome::Did(format!(
                "The record cannot be reached: {why}. Nothing is lost — the work stays as it \
                 is, and the table offers this publish again later. You are free for other work."
            ))
        }
        (Publish::Refused(why), _) => Outcome::Refused(format!("It cannot be published: {why}.")),
    }
}

fn synced_outcome(synced: Synced) -> Outcome {
    match synced {
        Synced::UpToDate => Outcome::Did(
            "The work already holds everything the record has: `merge_publish` it.".into(),
        ),
        Synced::Merged { conflicts } if conflicts.is_empty() => Outcome::Did(
            "The record's newer text is in the work now, and nothing overlaps: `merge_publish` \
             it."
            .into(),
        ),
        Synced::Merged { conflicts } => Outcome::Did(format!(
            "The record's newer text is in the work now. {} hold both sides between marks — \
             `<<<<<<< yours`, `=======`, `>>>>>>> the record`. Read each, keep what this work \
             was verified to say and what the record has gained, edit the document until no \
             mark is left, `bench_commit` it, and publish again.",
            conflicts.join(", ")
        )),
        Synced::Unreachable(why) => {
            Outcome::Refused(format!("The record cannot be reached: {why}."))
        }
        Synced::Refused(why) => Outcome::Refused(format!("It cannot be merged: {why}.")),
    }
}

/// The message a publish's commit carries: the operation, its document, and
/// every step's taker from its run history (§10.3).
pub fn commit_message(world: &str, op: &Operation) -> String {
    let steps: Vec<String> = op
        .run
        .history
        .iter()
        .map(|t| {
            let by = match &t.by {
                Taker::Table => "the table".to_string(),
                Taker::Actor(who) => who.clone(),
            };
            format!("- {}: {by}", t.step)
        })
        .collect();
    format!(
        "{}: {}\n\n{}\n\nSteps:\n{}\n\nOperation {} of {world}.\n",
        op.name,
        op.objective,
        op.document,
        steps.join("\n"),
        op.id
    )
}

/// Bring the mind folder up to the record a publish just moved, so the files
/// on disk — what the watcher ingests, and what the registries read — are what
/// was published.
pub fn bring_up(stores: &MindStores) {
    match blocking(|| stores.follow()) {
        Some(Ok(Some(Followed::Dirty(paths)))) => tracing::warn!(
            ?paths,
            "the mind folder has changes of its own on disk, and was not brought up to the record"
        ),
        Some(Ok(Some(Followed::Diverged))) => tracing::warn!(
            "the mind folder's branch holds commits the record does not, and was not brought up"
        ),
        Some(Err(e)) => {
            tracing::warn!("the mind folder could not be brought up to the record: {e}")
        }
        _ => {}
    }
}

/// The first ten characters of a commit id, as a Maker reads one.
fn short(commit: &str) -> &str {
    commit.get(..10).unwrap_or(commit)
}

/// Run git work from an act: off the runtime's worker, where there is one to
/// leave, so a push waiting on its agent's answer does not hold a worker the
/// answer needs.
fn blocking<T>(work: impl FnOnce() -> T) -> T {
    match Handle::try_current().map(|h| h.runtime_flavor()) {
        Ok(RuntimeFlavor::MultiThread) => tokio::task::block_in_place(work),
        _ => work(),
    }
}

#[cfg(test)]
mod tests;
