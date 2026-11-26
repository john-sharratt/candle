//! A pass's folders, sealed into the knowledge base as their listings.

use super::*;

/// Key this pass reports completeness under (see [`crate::ingest_report`]).
pub const PASS_NAME: &str = "repo_map";

/// Strip auto-summarization from a [`SequenceConfig`] before using it to
/// build an ingest base.
///
/// The legacy per-turn tree summarization (`summarize_every`) stays off, and
/// so does mid-decode reprojection: an ingest base and every conversation
/// forked from it is append-only. Ingest quantizes at C5, fully adaptive for
/// both K and V.
pub(crate) fn utility_config(mut config: SequenceConfig) -> SequenceConfig {
    config.tree.disable_summarization();
    config.disable_reprojection = true;
    config.kv_compression_level = Some(5);
    config
}

/// One folder to seal: the unit as a branch lists it, and the commit its
/// listing is read at.
#[derive(Debug, Clone)]
pub struct UnitJob {
    pub unit: FolderUnit,
    pub at: Oid,
    /// Every branch whose tip lists the folder this way, recorded on its
    /// record.
    pub branches: Vec<String>,
}

/// Seal `jobs` — the folders the pass queued (`docs/zend_branch_ingest.md`
/// §7.3). `live` is every folder key on any branch: when a folder's record
/// commits, each other record of the folder whose key is not in it is
/// retired, and one whose key is — the folder as another branch lists it —
/// is left.
pub fn ingest_units(
    ctx: &RefreshContext<'_>,
    jobs: &[UnitJob],
    live: &HashSet<String>,
    progress: &Arc<LoadProgress>,
    base: &Sequence,
) -> anyhow::Result<IngestReport> {
    tracing::info!(
        n_units = jobs.len(),
        sample = ?jobs.iter().take(5).map(|j| j.unit.dir.as_str()).collect::<Vec<_>>(),
        "repo map: sealing the listings of the folders no record holds yet",
    );
    let report = run_dir_pool(ctx, base, jobs, live, progress);
    crate::ingest_report::publish(PASS_NAME, report.clone());
    Ok(report)
}

/// A folder's path as a folder — closed by one `/` — which is how what a
/// drawing group draws is named. The workspace root, `""`, becomes `/`, which
/// nothing draws.
pub(super) fn as_folder(dir: &str) -> String {
    format!("{}/", dir.trim_end_matches('/'))
}
