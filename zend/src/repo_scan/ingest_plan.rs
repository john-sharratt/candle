//! The per-pass steering and tags that every folder's ingest conversation is
//! built from.
//!
//! Each folder's conversation is tagged with the layer and with its directory, so
//! a tag-scoped gallery admits exactly that folder's turns. The steering is held
//! once for the whole pass: its decodes bind the thinking trigger off and hold
//! every call to the chain's tools, because a folder conversation decoded free
//! wrote calls with no arguments and sealed an apology in place of a summary.
//! `calls_are_held` refuses steering that would leave a folder's calls free.
//! Utility ingests strip per-turn summarization and quantize at C5, since their
//! hundreds of small structured turns would otherwise queue a steady stream of
//! prefills behind the ingest. When a unit commits, the other conversations of its
//! folder whose keys no branch holds are tombstoned.

use super::*;

/// Key this pass reports completeness under (see [`crate::ingest_report`]).
pub const PASS_NAME: &str = "repo_map";

/// Gather-scope tags for a directory's turns: `["repo_map", <dir>]`. The second
/// tag is the unit's workspace-relative directory (`candle/` for a
/// repository's root), so a tag-scoped provenance gallery can admit exactly one
/// folder's turns.
pub(super) fn dir_tags(unit: &DirUnit) -> Vec<String> {
    vec!["repo_map".to_string(), unit.dir.clone()]
}

/// Strip auto-summarization from a [`SequenceConfig`] before using
/// it to mint a utility-layer conversation (repo_map, code_reading).
///
/// The legacy per-turn tree summarization (`summarize_every`) stays off here.
/// It does not block a turn — the task runs in the background and is applied
/// at a later turn boundary — but every one re-reads its whole window as a
/// fresh prefill on the scheduler. For repo_map / code_reading, which carry
/// hundreds of small structured turns (folder chains, scope reads), that would
/// queue a steady stream of those prefills behind ingest. The
/// async AVL summariser is separate and unaffected: it runs on its own thread
/// (wave-driven compression) and summarises every layer — including these —
/// without blocking ingest, and provenance scans expand the compressed nodes on
/// retrieval.
pub(crate) fn utility_config(mut config: SequenceConfig) -> SequenceConfig {
    config.tree.disable_summarization();
    // Utility ingests (repo_map, code_reading) are append-only: no mid-decode
    // reprojection, which would re-score and rebuild the context on every
    // wave and serialise the parallel workers behind the scheduler thread.
    // Each turn still assembles its slot from the substrate (a slot is emptied
    // at every seal), which is how a unit's priming chain — inherited through
    // `forked_from` — reaches its context.
    config.disable_reprojection = true;
    // Utility ingests quantize at C5, fully adaptive for both K and V (the
    // engine-wide uniform-K pin is off in this config). Both layers' ingest
    // bases are built from this config (`session.rs`), so every unit forked
    // from them carries it.
    config.kv_compression_level = Some(5);
    config
}

/// One folder to ingest: the unit as a branch lists it, and the commit its
/// listing and files are read from.
#[derive(Debug, Clone)]
pub struct UnitJob {
    pub unit: FolderUnit,
    pub at: Oid,
    /// Every branch whose tip lists the folder this way, recorded on the
    /// conversation ([`BRANCHES_KEY`]).
    pub branches: Vec<String>,
}

/// Ingest `jobs` — the units the pass queued (`docs/zend_branch_ingest.md`
/// §7.3). `live` is every unit key on any branch: when a unit commits, each
/// other conversation of its folder whose key is not in it is tombstoned,
/// and one whose key is — the folder as another branch lists it — is left.
///
/// Chat consumers keep running throughout: the pool's create/tombstone ops wait
/// on nothing a decode holds.
pub fn ingest_units(
    ctx: &RefreshContext<'_>,
    jobs: &[UnitJob],
    live: &HashSet<String>,
    progress: &Arc<LoadProgress>,
    layer_name: &str,
    base: &Sequence,
) -> anyhow::Result<IngestReport> {
    tracing::info!(
        n_units = jobs.len(),
        sample = ?jobs.iter().take(5).map(|j| j.unit.dir.as_str()).collect::<Vec<_>>(),
        "repo map: ingesting the folders no conversation holds yet",
    );
    let plan = IngestPlan::new(ctx, layer_name)?;
    let report = run_dir_pool(ctx, base, &plan, jobs, live, progress);
    crate::ingest_report::publish(PASS_NAME, report.clone());
    Ok(report)
}

/// The per-pass constants every worker needs to drive its unit's
/// conversation. Assembled once so the pool's per-unit signature stays
/// small.
pub(super) struct IngestPlan {
    /// The steering every folder's decodes run under: `<think>` bound to
    /// [`ThinkMode::Off`]'s tree, over the call grammar of the mode ingest's
    /// tools run in ([`ToolMode::Restricted`]), every call held to the chain's
    /// tools ([`render::CHAIN_TOOLS`]). A folder conversation that answers a
    /// listing with another call writes one of those tools, well-formed, with
    /// every required argument — decoded free, it wrote `file_list` with no
    /// arguments at all, three times in a row, and sealed an apology for a
    /// summary (`candle-conversation/`, 2026-11-14).
    ///
    /// Held ONCE here rather than per directory: holding decodes the
    /// catalog's name arms, identical for every folder.
    pub(super) triggers: Arc<TriggerRegistry>,
    /// The rules each folder's path holds are built into, kept across the
    /// pass's folders: a set built for one folder is built again only when the
    /// repositories' paths differ ([`HeldRules`]).
    pub(super) held_rules: Mutex<HeldRules>,
    /// The call syntax this checkpoint speaks, for the chain's PREFILLED
    /// `<tool_call>`s. Asked of the dialect so the prefills cannot teach a
    /// shape the chat stencil would refuse — see [`render::render_list_call`].
    pub(super) envelope: ToolCallEnvelope,
}

impl IngestPlan {
    pub(super) fn new(ctx: &RefreshContext<'_>, layer_name: &str) -> anyhow::Result<Self> {
        let layer = ctx
            .proj_builder
            .id_for_layer(layer_name)
            .ok_or_else(|| anyhow::anyhow!("projection schema missing '{layer_name}' layer"))?;
        let chain_tools: Vec<String> = render::CHAIN_TOOLS.iter().map(|t| t.to_string()).collect();
        let triggers = ctx
            .engine
            .holding_tools(
                &ctx.triggers_for(ThinkMode::Off, ToolMode::Restricted),
                &chain_tools,
                None,
            )
            .map_err(|e| {
                anyhow::anyhow!("repo map: holding its calls to the chain's tools: {e}")
            })?;
        calls_are_held(&triggers, ctx.engine.tokenizer().token_to_id(TOOL_CALL))?;
        // Append-only ingest layer (in-memory flag, re-applied every load): folder
        // summaries score self-local during ingest, so a summary is grounded in its
        // own folder rather than derailed by cross-directory retrieval.
        ctx.engine.mark_layer_append_only(layer);
        let envelope = ToolCallEnvelope::for_dialect(&ctx.config.dialect);
        Ok(Self {
            triggers,
            held_rules: Mutex::new(HeldRules::default()),
            envelope,
        })
    }

    /// The pass's steering with every call's paths held to what `files` — one
    /// folder's commit-pinned stores — holds (`docs/stencil_tree.md` §9.2): a
    /// listing names a directory that exists, a read a file that does.
    pub(super) fn unit_triggers(
        &self,
        engine: &ConversationEngine,
        files: &RepoFiles,
    ) -> Arc<TriggerRegistry> {
        let holds = ValueHolds {
            fields: held_paths::holds(files),
        };
        let mut rules = self
            .held_rules
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        engine.holding_values(&self.triggers, &holds, &mut rules)
    }
}

/// The token a model opens a call with.
pub(super) const TOOL_CALL: &str = "<tool_call>";

/// Refuse steering that would leave a folder's calls free: a vocabulary with a
/// `<tool_call>` token, and no call grammar bound to it. Holding passes such a
/// registry through unchanged, so a pass built on one decoded every call free
/// and said nothing — the repo map ran that way, on a registry holding only the
/// `<think>` trigger, until 2026-11-14.
pub(super) fn calls_are_held(triggers: &TriggerRegistry, call: Option<u32>) -> anyhow::Result<()> {
    match call {
        Some(token) if !triggers.is_trigger(token) => Err(anyhow::anyhow!(
            "repo map: no call grammar is bound to `{TOOL_CALL}`, so a folder's calls \
             would decode free"
        )),
        _ => Ok(()),
    }
}

/// The tools a unit's chain runs against: file stores reading the commit the
/// unit was found on, so the listing and the manifest hint its turns show are the
/// ones its key names. `None` when the unit's repository is not read through
/// git.
/// A folder's path as a folder — closed by one `/` — which is how what a
/// drawing group draws is named. The workspace root, `""`, becomes `/`, which
/// nothing draws.
pub(super) fn as_folder(dir: &str) -> String {
    format!("{}/", dir.trim_end_matches('/'))
}

pub(super) fn unit_tools(tools: &ToolContext, job: &UnitJob) -> Option<ToolContext> {
    let files = tools.files.fresh_at(&job.unit.repo, &job.at)?;
    Some(tools.with_files(Arc::new(files), ""))
}
