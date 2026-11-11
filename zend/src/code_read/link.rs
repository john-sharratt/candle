//! One file read as a link of the priming chain
//! (`crate::branch_ingest::prime`): the same per-file ingest a pool worker
//! runs, called for one file so the caller knows which conversation holds it
//! before the next link's reading starts.

use std::collections::HashSet;
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

use candle_conversation::projection::TimelineId;
use candle_conversation::{ConversationEngine, Sequence};

use super::{chain_finished, finished_content_keys, process_one_file, FileJob, Job};
use crate::branch_ingest::keys::CONTENT_KEY;
use crate::branch_ingest::prime::{repoint, ChainParent};
use crate::ingest_report::Failures;
use crate::priming_watch;
use crate::refresh_ctx::{RefreshContext, PRIMING_BAND};

/// Read `file` behind `parent`, the link before it, and return the
/// conversation that holds it — `None` when it could not be read (logged; the
/// chain continues from the link before it).
///
/// Its parent is recorded by `parent`'s content key ([`repoint`]). A file
/// already read is not read again: its conversation is re-pointed, since the
/// link before it may be another conversation now. Only a reading whose chain
/// finished counts as read ([`chain_finished`]); an unfinished one is read
/// again, as the pass would read it. `live` is every file key on any branch,
/// and `binary` the keys found to be binary, as a pool is given them.
pub(crate) fn ingest_link(
    ctx: &RefreshContext<'_>,
    base: &Sequence,
    file: &FileJob,
    live: &HashSet<String>,
    binary: &Mutex<HashSet<String>>,
    parent: Option<ChainParent<'_>>,
) -> anyhow::Result<Option<TimelineId>> {
    let key = file.key();
    let named = parent.map(|p| p.key);
    if let Some(held) = holder(ctx.engine, &key) {
        priming_watch::stage("re-pointing the held link");
        if repoint(ctx.engine, held, named)
            .map_err(|e| anyhow::anyhow!("priming chain: {} parent: {e}", file.path))?
        {
            return Ok(Some(held));
        }
    }
    let link_ctx = RefreshContext {
        link_parent: parent.map(|p| p.timeline),
        band: Some(PRIMING_BAND),
        ..ctx.clone()
    };
    // A link left to the member holding its lease has no holder here yet, so
    // the chain continues from the link before it, as for a failed read.
    let left = AtomicBool::new(false);
    let job = Job {
        file,
        key: &key,
        live,
        binary,
        left: &left,
        // Read ahead of the pass, outside its rounds: a link that found no ground
        // fails here, and the chain continues from the link before it.
        resubmit: None,
    };
    let failures = Failures::new();
    priming_watch::stage("reading the file");
    // A chain link is calibrated by the pass's catch-up round, not as it
    // commits: it is read ahead of the pass, outside its workers.
    process_one_file(
        &link_ctx,
        base,
        &job,
        &finished_content_keys(ctx.engine),
        &failures,
        None,
    )?;
    let report = failures.into_report(1);
    if let Some(failure) = report.failures.first() {
        tracing::warn!(
            target: "zend::priming_chain",
            file = %file.path,
            "a chain link could not be read — the chain continues from the link before \
             it: {}",
            failure.error,
        );
        return Ok(None);
    }
    // Its reading recorded the parent's timeline as it forked; the chain
    // names it by key.
    let held = holder(ctx.engine, &key);
    if let Some(tl) = held {
        repoint(ctx.engine, tl, named)
            .map_err(|e| anyhow::anyhow!("priming chain: {} parent: {e}", file.path))?;
    }
    Ok(held)
}

/// The conversation holding the file key `key` whose reading finished.
fn holder(engine: &ConversationEngine, key: &str) -> Option<TimelineId> {
    engine
        .find_conversations_by_metadata(CONTENT_KEY, key)
        .into_iter()
        .find(|&tl| chain_finished(engine, tl))
}
