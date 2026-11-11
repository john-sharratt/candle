//! One folder read as a link of the priming chain
//! (`crate::branch_ingest::prime`): the same per-unit ingest a pool worker
//! runs, called for one unit so the caller knows which conversation it is
//! before the next link's reading starts.

use std::collections::HashSet;

use candle_conversation::projection::TimelineId;
use candle_conversation::{ConversationEngine, Sequence};

use super::{process_one_dir, IngestPlan, UnitJob};
use crate::branch_ingest::keys::CONTENT_KEY;
use crate::branch_ingest::prime::{repoint, ChainParent};
use crate::ingest_report::Failures;
use crate::priming_watch;
use crate::refresh_ctx::{RefreshContext, PRIMING_BAND};

/// Ingest `job` behind `parent`, the link before it, and return the
/// conversation that holds it — `None` when its ingest failed (the failure is
/// logged; the chain continues from the link before it).
///
/// Its parent is recorded by `parent`'s content key ([`repoint`]). A unit
/// already committed is not read again: it is re-pointed, since the link
/// before it may be another conversation now. `live` is every unit key on any
/// branch, as a pool is given it, so a link's ingest retires only what no
/// branch holds.
pub(crate) fn ingest_link(
    ctx: &RefreshContext<'_>,
    base: &Sequence,
    layer_name: &str,
    job: &UnitJob,
    live: &HashSet<String>,
    parent: Option<ChainParent<'_>>,
) -> anyhow::Result<Option<TimelineId>> {
    let named = parent.map(|p| p.key);
    if let Some(held) = holder(ctx.engine, &job.unit.key) {
        priming_watch::stage("re-pointing the held link");
        if repoint(ctx.engine, held, named)
            .map_err(|e| anyhow::anyhow!("priming chain: {} parent: {e}", job.unit.dir))?
        {
            return Ok(Some(held));
        }
    }
    let plan = IngestPlan::new(ctx.engine, &ctx.proj_builder, &ctx.config, layer_name)?;
    let link_ctx = RefreshContext {
        link_parent: parent.map(|p| p.timeline),
        band: Some(PRIMING_BAND),
        ..ctx.clone()
    };
    let failures = Failures::new();
    process_one_dir(&link_ctx, base, &plan, job, live, &failures)?;
    let report = failures.into_report(1);
    if let Some(failure) = report.failures.first() {
        tracing::warn!(
            target: "zend::priming_chain",
            dir = %job.unit.dir,
            "a chain link could not be read — the chain continues from the link before \
             it: {}",
            failure.error,
        );
        return Ok(None);
    }
    // Its reading recorded the parent's timeline as it forked; the chain
    // names it by key.
    let held = holder(ctx.engine, &job.unit.key);
    if let Some(tl) = held {
        repoint(ctx.engine, tl, named)
            .map_err(|e| anyhow::anyhow!("priming chain: {} parent: {e}", job.unit.dir))?;
    }
    Ok(held)
}

/// The committed conversation holding the folder key `key`.
fn holder(engine: &ConversationEngine, key: &str) -> Option<TimelineId> {
    engine
        .find_conversations_by_metadata(CONTENT_KEY, key)
        .into_iter()
        .next()
}
