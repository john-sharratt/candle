//! The ingest of one directory into a fresh conversation, from its rendered
//! round-trip chain to its sealed turns.
//!
//! The tool responses are rendered before anything is minted, so a directory the
//! tools cannot read costs no conversation, and an error body is never prefilled
//! as if it were a tool result. The directory tag is written before the decode
//! that can fail, so a crash leaves a partial that names its folder, and the
//! content key is withheld until success. Reconciling the folder's older
//! generations is deferred: the prior one stays live until this ingest commits its
//! own key, so a failed attempt invalidates nothing. A cancelled shutdown is not
//! booked as a failure, and the conversation is flagged for eviction once it
//! commits, since nothing attends the folder again until a projection retrieves it.

use super::*;

/// Ingest one directory into a fresh conversation: render the folder's
/// round-trip chain, run it (the prefill + the summary decode), tag the
/// conversation, and free it.
///
/// The caller has already established this unit is not committed, and holds
/// its [`ScanSlot`] for the duration.
///
/// A per-directory ingest failure is TOLERATED up to [`MAX_DECODE_FAILURES`]:
/// the attempt's partial is tombstoned, the prior generation is left live, and
/// the unit is queued again next pass.
pub(super) fn process_one_dir(
    ctx: &RefreshContext<'_>,
    base: &Sequence,
    plan: &IngestPlan,
    job: &UnitJob,
    live: &HashSet<String>,
    failures: &Failures,
) -> anyhow::Result<()> {
    let engine = ctx.engine;
    let Some(tools) = unit_tools(&ctx.tool_ctx, job) else {
        failures.record(
            &job.unit.dir,
            "its repository is not read through git".to_string(),
        );
        return Ok(());
    };
    let unit = &DirUnit::of(&job.unit);
    // Render BEFORE minting anything: the tool responses come from actually
    // running the tools, so a directory the tools can't read is caught here and
    // costs no conversation. Prefilling an error body would be worse than
    // skipping — it teaches the model a tool interaction that failed.
    ctx.link_stage("listing the folder");
    let (prefilled, decode_user) = render::render_chain(&tools, unit, &plan.envelope);
    if let Some(detail) = render::chain_error(&prefilled, &decode_user) {
        let n = failures.record(&unit.dir, format!("file_list failed: {detail}"));
        tracing::warn!(
            target: "zend::repo_scan",
            dir = %unit.dir,
            "skip: file_list failed for this directory ({detail}); not prefilling an error body",
        );
        if n > MAX_DECODE_FAILURES {
            tracing::error!(
                target: "zend::repo_scan",
                n, cap = MAX_DECODE_FAILURES,
                "repo map ingest stopping early: failure cap reached (last: {detail})",
            );
            failures.set_abort();
        }
        return Ok(());
    }

    // Reconcile this directory's existing conversations WITHOUT invalidating
    // good content up front — a DEFERRED tombstone. A committed generation
    // whose key no branch holds any more is deferred into `superseded`,
    // staying live as the folder's content until this ingest commits its own
    // key. One whose key a branch still holds is the folder as that branch
    // lists it, and is left alone — and so is one with no key yet, another
    // worker's attempt at the folder as another branch lists it. An attempt
    // that fails retires its own; a crash's are retired at boot
    // ([`retire_crashed_partials`]).
    let (mut conv, superseded) = {
        let e = engine;
        let superseded: Vec<TimelineId> = e
            .find_conversations_by_metadata(DIR_KEY, &unit.dir)
            .into_iter()
            .filter(|&tl| {
                e.conversation_metadata_value(tl, CONTENT_KEY)
                    .is_some_and(|key| !live.contains(&key))
            })
            .collect();
        // Forks off `base` — this layer's prefilled template, `base_conv`'s
        // exact counterpart for ingestion (see `InferenceState::ingest_bases`)
        // — so this conversation shares the SAME already-computed prefix a
        // live dialogue turn forks from, rather than this pool's workers each
        // independently re-running the schema's "eager section ingestion"
        // under concurrent load.
        ctx.link_stage("forking its conversation");
        let conv = ctx
            .fork_unit(base)
            .map_err(|err| anyhow::anyhow!("repo_map conv create: {err}"))?;
        ctx.link_stage("recording its parent");
        // A priming-chain link records the link before it as its parent BEFORE
        // its own listing and summary run, so its turns are projected with the
        // chain so far in context (`Substrate::inherited_chain`). Every other
        // folder has no parent and is sealed behind the system prompt alone:
        // a turn's K/V carries the context it was computed in
        // (`RefreshContext::link_parent`).
        if let Some(parent) = ctx.link_parent {
            engine
                .set_forked_from(conv.timeline_id(), parent)
                .map_err(|err| anyhow::anyhow!("repo_map priming-chain parent: {err}"))?;
            // The fork seeded the recurrent slot before this pointer existed,
            // so ask again now that the lineage is on record.
            if let Err(err) = conv.seed_recurrent_from_lineage() {
                tracing::warn!(
                    target: "zend::repo_scan",
                    dir = %unit.dir,
                    "seeding recurrent memory from the priming chain failed: {err:#}",
                );
            }
        }
        let e = engine;
        // The folder's closing turn is its own decoded summary, so the AVL
        // summariser must not compress these turns into a second summary tree.
        e.set_timeline_summarize(conv.timeline_id(), false);
        // The conversation carries no conv_id (it is not a dialogue), so without a
        // label the substrate viewer renders it as "(untitled)".
        if let Err(err) = e.set_conversation_label(conv.timeline_id(), &unit.dir) {
            tracing::warn!(target: "zend::repo_scan", "repo_map label set failed: {err:#}");
        }
        (conv, superseded)
    };

    // Tag `dir` IMMEDIATELY — before the decode below that can fail. A partial
    // left by a crash then still names the directory it covers, so it (a) shows
    // in the substrate as that folder rather than "(untitled)", and (b) is found
    // by the next boot's [`retire_crashed_partials`]. The content key is withheld
    // until success, so a partial is never mistaken for a completed ingest and
    // skipped.
    {
        let mut early = BTreeMap::new();
        early.insert("kind".to_string(), "repo_map".to_string());
        early.insert(DIR_KEY.to_string(), unit.dir.clone());
        if let Err(e) = conv.set_metadata_many(&early) {
            tracing::warn!(
                target: "zend::repo_scan",
                dir = %unit.dir,
                "failed to tag dir metadata at conversation creation: {e:#}",
            );
        }
    }

    // One conversation, seeded with request → list → response and then driven as
    // a REAL tool loop until it answers — see `converse::run_folder_conversation`.
    // Any follow-up call runs against the same commit-pinned tools the seed
    // listing came from, so every turn shows what the unit's key names.
    // The conversation projects its own turns (`target_is_ingest_self`), so the
    // request is in every decode's context where it belongs, and there is no
    // throwaway intermediate decode to set the wrong style.
    ctx.link_stage("decoding its summary");
    let emit = converse::run_folder_conversation(
        &mut conv,
        unit,
        &prefilled,
        decode_user,
        plan.unit_triggers(engine, &tools.files),
        &tools,
    );

    ctx.link_stage("committing it");
    let summary = match emit {
        Ok(summary) => summary,
        Err(e) => {
            // The deferred tombstone is the safety net: `superseded` was never
            // tombstoned, so the prior generation stays live and committed —
            // this failed attempt invalidates nothing. Drop only THIS attempt's
            // partial; the retry re-mints cleanly.
            {
                let en = engine;
                if let Err(err) = en.tombstone_timeline(conv.timeline_id()) {
                    tracing::warn!(
                        target: "zend::repo_scan",
                        dir = %unit.dir,
                        "tombstone of failed-attempt partial failed: {err:#}",
                    );
                }
            }
            // If a graceful shutdown latched the cancel flag, this `Err` is the
            // interruptible decode-wait unwinding (`wait_cancellable` →
            // `IngestCancelled`), not a genuine ingest failure — the anyhow
            // layer has erased the variant, so the global flag is the source of
            // truth. Don't record it against the failure cap: a Ctrl-C with 24
            // workers in flight would otherwise book 24 failures at once, trip
            // the abort, and report the pass as incomplete. The partial was just
            // tombstoned, so the directory re-ingests next run.
            if crate::ingest_cancel::cancelled() {
                tracing::debug!(
                    target: "zend::repo_scan",
                    dir = %unit.dir,
                    "shutdown cancelled decode mid-directory — dropped partial; will re-ingest next run",
                );
                return Ok(());
            }
            // A turn the card did not move for a minute is no flood: a full
            // card stalls every folder in flight at once, and counted, they
            // would end the pass for every folder behind them.
            let msg = format!("{e:#}");
            if is_stall_message(&msg) {
                tracing::warn!(
                    target: "zend::repo_scan",
                    dir = %unit.dir,
                    superseded_kept = superseded.len(),
                    "directory ingest stalled (read again next pass; prior generation kept \
                     live; not counted against the pass's cap): {msg}",
                );
                ingest_holdoff::cool_down(&unit.dir, &msg);
                return Ok(());
            }
            let n = failures.record(&unit.dir, format!("{e:#}"));
            tracing::warn!(
                target: "zend::repo_scan",
                dir = %unit.dir,
                superseded_kept = superseded.len(),
                "directory ingest failed (will retry next run; prior generation kept live): {e:#}",
            );
            ingest_holdoff::cool_down(&unit.dir, &msg);
            if n > MAX_DECODE_FAILURES {
                // Carry the CAUSE, not just the count: the fatal-looking line
                // used to say only "N failed (cap = M)", so every diagnosis
                // began by scrolling back through N warnings to find the root.
                tracing::error!(
                    target: "zend::repo_scan",
                    n, cap = MAX_DECODE_FAILURES,
                    "repo map ingest stopping early: failure cap reached (last: {e:#})",
                );
                failures.set_abort();
            }
            return Ok(());
        }
    };

    let mut tags = BTreeMap::new();
    tags.insert("kind".to_string(), "repo_map".to_string());
    tags.insert(DIR_KEY.to_string(), unit.dir.clone());
    tags.insert(CONTENT_KEY.to_string(), unit.content_key.clone());
    if let Some(bits) = crate::branch_ingest::keys::model_bits() {
        tags.insert(
            crate::branch_ingest::keys::QUANT_KEY.to_string(),
            bits.to_string(),
        );
    }
    tags.insert("files".to_string(), unit.listed.len().to_string());
    if !job.branches.is_empty() {
        tags.insert(BRANCHES_KEY.to_string(), branches_value(&job.branches));
    }
    // The tag write is what commits the new generation. If it fails, this
    // attempt has to go: keeping the prior generation live is right, but keeping
    // BOTH is not — the untagged replacement is invisible to the pass's plan and
    // to every scope, yet until the next boot's sweep its turns stay visible to
    // provenance, so the folder would vote twice in every scan. Drop it and let
    // the unit retry, exactly like a decode failure.
    if let Err(e) = conv.set_metadata_many(&tags) {
        {
            let en = engine;
            if let Err(err) = en.tombstone_timeline(conv.timeline_id()) {
                tracing::warn!(
                    target: "zend::repo_scan",
                    dir = %unit.dir,
                    "tombstone of untagged replacement failed — TWO generations of this \
                     directory are now live: {err:#}",
                );
            }
        }
        let n = failures.record(&unit.dir, format!("metadata tag write failed: {e:#}"));
        tracing::warn!(
            target: "zend::repo_scan",
            dir = %unit.dir,
            superseded_kept = superseded.len(),
            "failed to tag conversation metadata (content key); dropped the replacement \
             and kept the prior generation: {e:#}",
        );
        if n > MAX_DECODE_FAILURES {
            tracing::error!(
                target: "zend::repo_scan",
                n, cap = MAX_DECODE_FAILURES,
                "repo map ingest stopping early: failure cap reached (last: {e:#})",
            );
            failures.set_abort();
        }
        return Ok(());
    }
    ctx.retrieval.mark_stale();

    // Deferred tombstone ACTIVATES, now that the new generation is truly
    // committed (its content key landed above). Until this instant a
    // projection saw the prior generation (stale but present); from here it
    // sees this one, and only this one.
    if !superseded.is_empty() {
        let e = engine;
        for tl in &superseded {
            if let Err(err) = e.tombstone_timeline(*tl) {
                tracing::warn!(
                    target: "zend::repo_scan",
                    dir = %unit.dir,
                    "deferred tombstone of superseded generation failed: {err:#}",
                );
            }
        }
    }

    // Nothing attends this folder again until a projection retrieves it. Flag it
    // for full KV eviction so the persistence pipeline offloads its turns to cold
    // and frees both the VRAM and RAM copies; `elevate_to_hot` pulls them back on
    // demand. `FreeSequence` on drop only releases the batch slot, not the sealed
    // KV, so this is what actually reclaims the space across a large scan.
    {
        let e = engine;
        // Read once, never continued: it keeps its turns and no recurrent
        // snapshot (`crate::ingest_snapshots`).
        finished(e, conv.timeline_id(), Unit::Folder(&unit.dir));
        e.evict_ingest_timeline(conv.timeline_id());
    }
    tracing::debug!(
        target: "zend::repo_scan",
        dir = %unit.dir,
        tokens = summary.tokens,
        tool_rounds = summary.tool_rounds,
        "directory ingested (chain prefilled + summary decoded)",
    );
    Ok(())
}
