//! One file read as a link of the priming chain
//! (`crate::branch_ingest::prime`): the same per-file ingest a pool worker
//! runs, called for one file so the caller knows which conversation holds it
//! before the next link's reading starts.

use std::collections::HashSet;
use std::sync::Mutex;

use candle_conversation::projection::TimelineId;
use candle_conversation::{ConversationEngine, Sequence};

use super::{chain_finished, finished_content_keys, process_one_file, FileJob, Job};
use crate::branch_ingest::keys::CONTENT_KEY;
use crate::ingest_report::Failures;
use crate::refresh_ctx::RefreshContext;

/// Read `file` with `parent` recorded as its parent before its reading
/// starts, and return the conversation that holds it — `None` when it could
/// not be read (logged; the chain continues from the link before it).
///
/// A file already read is not read again: its conversation is re-pointed at
/// `parent`, since the chain it hangs off is rebuilt every boot. Only a
/// reading whose chain finished counts as read ([`chain_finished`]); an
/// unfinished one is read again, as the pass would read it. `live` is every
/// file key on any branch, and `binary` the keys found to be binary, as a pool
/// is given them.
pub(crate) fn ingest_link(
    ctx: &RefreshContext<'_>,
    base: &Mutex<Sequence>,
    file: &FileJob,
    live: &HashSet<String>,
    binary: &Mutex<HashSet<String>>,
    parent: Option<TimelineId>,
) -> anyhow::Result<Option<TimelineId>> {
    let key = file.key();
    if let Some(held) = holder(ctx.engine, &key) {
        if let Some(parent) = parent {
            ctx.engine
                .lock()
                .unwrap()
                .set_forked_from(held, parent)
                .map_err(|e| anyhow::anyhow!("priming chain: {} parent: {e}", file.path))?;
        }
        return Ok(Some(held));
    }
    let link_ctx = RefreshContext {
        chain_end: parent,
        ..ctx.clone()
    };
    let job = Job {
        file,
        key: &key,
        live,
        binary,
    };
    let failures = Failures::new();
    process_one_file(
        &link_ctx,
        base,
        &job,
        &finished_content_keys(ctx.engine),
        &failures,
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
    Ok(holder(ctx.engine, &key))
}

/// The conversation holding the file key `key` whose reading finished.
fn holder(engine: &Mutex<ConversationEngine>, key: &str) -> Option<TimelineId> {
    let e = engine.lock().unwrap();
    e.find_conversations_by_metadata(CONTENT_KEY, key)
        .into_iter()
        .find(|&tl| chain_finished(&e, tl))
}
