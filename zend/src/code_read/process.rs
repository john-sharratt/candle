//! Reads one file into a fresh conversation, and commits its content key only
//! once that reading has succeeded.
//!
//! The file's earlier generation stays live until the new one commits: a failed,
//! cancelled or interrupted attempt leaves the earlier content in place, and its
//! own conversation is dropped. The path is tagged before the long decode, so a
//! crash leaves a partial that the next boot retires rather than an untitled
//! conversation. A reading that answered nothing, or read some other path, fails
//! like any other attempt, so an empty reading never replaces a good one. A file
//! the card had no room for is dropped whole and goes back through admission.

use super::*;

/// Ingest one file into a fresh per-file conversation: skip it when its key
/// is already committed or it is binary; otherwise mint the conversation, run
/// it as a real tool-using exchange ([`run_file_conversation`]), tag it with
/// its content key + metadata, calibrate it when `calibrate` names how — its
/// reading still resident — then drop it (freeing the GPU slot; the sealed
/// turns + tags persist in the substrate).
pub(super) fn process_one_file(
    ctx: &RefreshContext<'_>,
    base: &Sequence,
    job: &Job<'_>,
    present_keys: &HashSet<String>,
    failures: &Failures,
    calibrate: Option<&CalibrateSpec>,
) -> anyhow::Result<()> {
    let file = job.file;
    if present_keys.contains(job.key) {
        tracing::debug!(
            target: "zend::code_read::ingest",
            file = %file.path,
            "skip: file already committed",
        );
        return Ok(());
    }
    if job.binary.lock().unwrap().contains(job.key) {
        return Ok(());
    }
    // Another member that planned this same unread file holds its lease and is
    // the one that reads it. The conversation replicates, so this pass leaves
    // the file. A lease just handed back is read here only when that member's
    // writes have arrived and the key is still absent.
    let hold = match ingest_lease::share(job.key) {
        Share::Theirs => {
            tracing::info!(
                target: "zend::code_read::ingest",
                file = %file.path,
                "left to the member holding its ingest lease",
            );
            job.left.store(true, Ordering::Relaxed);
            return Ok(());
        }
        Share::Mine(hold) => hold,
    };
    if hold.recheck() && finished_content_keys(ctx.engine).contains(job.key) {
        return Ok(());
    }
    let Some(tools) = file_tools(&ctx.tool_ctx, file) else {
        failures.record(
            &file.path,
            "its repository is not read through git".to_string(),
        );
        return Ok(());
    };
    // Content guard: an allowlisted extension does NOT guarantee text. A
    // compiled fatbin / object dump committed as `*.txt` clears both the
    // extension gate and the size gate, and a hidden conversation told to
    // read it would read noise. Sniffed from the blob before any
    // conversation is minted, and remembered, so it is read once. The same
    // bytes give the line count a fast-path answer reports.
    let (repo, inner) = split(&file.path);
    let bytes = tools
        .files
        .repo(repo)
        .ok()
        .and_then(|store| store.read_bytes(inner).ok().flatten());
    let (lines, ranges) = match bytes {
        Some(bytes) if is_binary_sample(&bytes) => {
            tracing::debug!(
                target: "zend::code_read::ingest",
                file = %file.path,
                "skip: binary content behind a text extension",
            );
            job.binary.lock().unwrap().insert(job.key.to_string());
            // Recorded, so no later pass, restart or member plans it again.
            let conversation = ctx.engine.conversation();
            if let Err(e) = binary_mark::record(&conversation, job.key) {
                tracing::warn!(
                    target: "zend::code_read::ingest",
                    file = %file.path,
                    "could not record the file as binary; the next pass sniffs it again: {e:#}",
                );
            }
            return Ok(());
        }
        Some(bytes) => (
            line_count(&bytes),
            windows(inner, &String::from_utf8_lossy(&bytes)),
        ),
        None => {
            failures.record(
                &file.path,
                "the file could not be read at its commit".to_string(),
            );
            return Ok(());
        }
    };

    // Reconcile the existing conversations for this path WITHOUT invalidating
    // good content up front — a DEFERRED tombstone:
    //   * a committed generation whose key no branch holds any more is
    //     DEFERRED into `superseded`: it stays live as the file's fallback
    //     content, so a failed re-ingest below (e.g. a VRAM OOM) leaves it
    //     intact instead of destroying it. Its tombstone ACTIVATES only after
    //     this ingest commits its own key (see the success path below) — an
    //     atomic swap, "stale-but-present" over "gone";
    //   * a generation carrying THIS file's key is one whose chain never
    //     finished — a finished one would have been in `present_keys` and
    //     skipped above — so it is deferred the same way: the rebuild replaces
    //     it, and until the rebuild commits nothing is lost;
    //   * a committed generation whose key is still live is the file as
    //     another branch holds it, and is left alone;
    //   * one with no content key yet is another worker's, reading the file
    //     as another branch holds it, and is left alone too — an attempt
    //     that fails retires its own, and a crash's are retired at boot
    //     ([`retire_crashed_partials`]).
    let (mut conv, superseded) = {
        let e = ctx.engine;
        let superseded: Vec<TimelineId> = e
            .find_conversations_by_metadata(PATH_KEY, &file.path)
            .into_iter()
            .filter(|&tl| {
                e.conversation_metadata_value(tl, CONTENT_KEY)
                    .is_some_and(|key| key == job.key || !job.live.contains(&key))
            })
            .collect();
        // Forks off `base` — this layer's prefilled template, `base_conv`'s
        // exact counterpart for ingestion (see `InferenceState::ingest_bases`)
        // — so this conversation shares the SAME already-computed prefix a
        // live dialogue turn forks from, rather than this pool's workers each
        // independently re-running the schema's "eager section ingestion"
        // under concurrent load. That used to be this function's own
        // `new_conversation_with_projection` call, on the strength of a
        // comment claiming the two constructions were equivalent because they
        // shared the same prompt text and config — they were not: a live
        // conversation was measured losing track of its own turns after being
        // built this way, and stopped doing so once it forked `base_conv`
        // like every dialogue turn already did.
        let conv = ctx
            .fork_unit(base)
            .map_err(|err| anyhow::anyhow!("code_reading conv create: {err}"))?;
        // A priming-chain link records the link before it as its parent BEFORE
        // its own reading starts, so its turns are projected with the chain so
        // far in context (`Substrate::inherited_chain`). Every other file has
        // no parent: it is sealed behind the system prompt alone, from the
        // prompt's recurrent checkpoint the fork installed, because a turn's
        // K/V carries the context it was computed in — a file sealed behind
        // the chain is unreadable to a dialogue that does not hold the chain
        // directly ahead of it (`RefreshContext::link_parent`).
        {
            let e = ctx.engine;
            if let Some(parent) = ctx.link_parent {
                e.set_forked_from(conv.timeline_id(), parent)
                    .map_err(|err| anyhow::anyhow!("code_reading priming-chain parent: {err}"))?;
            }
            e.set_timeline_summarize(conv.timeline_id(), false);
        }
        // Now that a link's lineage is on record, take the recurrent memory of
        // the link it continues — the slot was seeded with the system prompt's
        // checkpoint when the fork returned.
        if ctx.link_parent.is_some() {
            if let Err(err) = conv.seed_recurrent_from_lineage() {
                tracing::warn!(
                    target: "zend::code_read::ingest",
                    file = %file.path,
                    "seeding recurrent memory from the priming chain failed: {err:#}",
                );
            }
        }
        (conv, superseded)
    };

    // Tag the `path` IMMEDIATELY — before the decode-heavy run below that can
    // fail (GPU OOM mid-decode, a decode error). A partial left by a crash then
    // still carries its path, so it (a) shows in the substrate as the file it
    // covers rather than "(untitled)", and (b) is found by the next boot's
    // [`retire_crashed_partials`]. The content key is deliberately withheld
    // until success (below), so a partial is never mistaken for a completed
    // ingest and skipped.
    {
        let mut early = BTreeMap::new();
        early.insert("kind".to_string(), "code_read".to_string());
        early.insert(PATH_KEY.to_string(), file.path.clone());
        if let Err(e) = conv.set_metadata_many(&early) {
            tracing::warn!(
                target: "zend::code_read::ingest",
                file = %file.path,
                "failed to tag path metadata at conversation creation: {e:#}",
            );
        }
    }

    // Each file's conversation is a conversation like any other, with file
    // stores of its own — reading the commit the file was found on: nothing
    // another unit, or a live dialogue, changed is what it reads.
    let unit_ctx = Arc::new(tools);
    // **A reading that answered nothing failed.** Its chain is one a later
    // pass finds broken (`ChainBreak::NoAnswer`) and reads again; committed,
    // it superseded — tombstoned — the generation that did read the file, and
    // every pass after replaced one empty reading with the next. Three files
    // on 2026-11-11 lost their readings that way. So it fails here, as any
    // failed attempt does: this attempt goes, and the prior generation stays.
    // **Every call of the reading names its own file**: `repo`, `path` and
    // `lines` held to the ones its prompt names, in the opening turn's grammar
    // and in every round's trigger after it.
    let mut kept = HeldRules::default();
    let held = ctx
        .engine
        .held_values(&reading_holds(&file.path, &ranges), &mut kept);
    let triggers = match &held {
        Some(held) => ctx
            .engine
            .triggers_holding(&ctx.think_triggers, Arc::clone(held)),
        None => Arc::clone(&ctx.think_triggers),
    };
    let read = run_file_conversation(
        &mut conv,
        &file.path,
        lines,
        &ranges,
        &triggers,
        ctx.read_opening
            .as_ref()
            .map(|opening| (opening, held.as_ref())),
        &unit_ctx,
    )
    .and_then(|text| match produced_no_answer(&text) {
        true => Err(anyhow::anyhow!(
            "{READ_NOTHING} ({} chars, none of them kept)",
            text.len()
        )),
        false => Ok(text),
    });
    let summary = match read {
        Ok(text) => text,
        Err(e) => {
            // The deferred tombstone is the safety net here: the prior good
            // generation in `superseded` was NEVER tombstoned, so it stays
            // live as the file's content — this failed attempt invalidates
            // nothing. Drop only THIS attempt's partial; the retry re-mints
            // cleanly.
            {
                let e2 = ctx.engine;
                if let Err(err) = e2.tombstone_timeline(conv.timeline_id()) {
                    tracing::warn!(
                        target: "zend::code_read::ingest",
                        file = %file.path,
                        "tombstone of failed-attempt partial failed: {err:#}",
                    );
                }
            }
            // If a graceful shutdown latched the cancel flag, this `Err` is the
            // interruptible decode-wait unwinding (`wait_cancellable` →
            // `IngestCancelled`), not a genuine decode failure — the anyhow layer
            // has erased the variant, so the global flag is the source of truth.
            // Don't record it against the failure cap; the partial was just
            // tombstoned, so the file re-ingests next run.
            if crate::ingest_cancel::cancelled() {
                tracing::debug!(
                    target: "zend::code_read::ingest",
                    file = %file.path,
                    "shutdown cancelled decode mid-file — dropped partial; will re-ingest next run",
                );
                return Ok(());
            }
            // **No ground is not a bad file.** The device refused the file's wave
            // for want of room the rest of the round is holding — and gives back as
            // it finishes. The attempt is dropped whole (its partial is tombstoned
            // above, its slot freed when `conv` drops) and the file goes back
            // through admission after the round, rather than being counted against
            // the cap and left for a pass that only a branch move or a layer clock
            // starts. On WhiteWorld (2026-11-08) every such failure was one region
            // short, and a pass gave up at its cap with 30 files refused this way.
            if let Some(resubmit) = job.resubmit {
                let msg = format!("{e:#}");
                if is_ground_refusal_message(&msg) {
                    tracing::info!(
                        target: "zend::code_read::ingest",
                        file = %file.path,
                        "file ingest found no ground; its attempt is dropped and the file \
                         is resubmitted after this round: {msg}",
                    );
                    resubmit.store(true, Ordering::Relaxed);
                    return Ok(());
                }
            }
            // **One reading's own failure is no flood.** A turn the card did not
            // move for a minute, a read that found some other path, an answer of
            // nothing: each is this file's attempt failing, which the next pass
            // reads again, not the systemic failure the cap stops a pass for. A
            // full card stalls every reading in flight at once, and counted,
            // sixteen of them would end the pass for every file behind them.
            let msg = format!("{e:#}");
            if is_stall_message(&msg) || is_reading_failure(&msg) {
                tracing::warn!(
                    target: "zend::code_read::ingest",
                    file = %file.path,
                    superseded_kept = superseded.len(),
                    "file ingest failed (read again next pass; prior generation kept live; \
                     not counted against the pass's cap): {msg}",
                );
                // A stall is the card's, and the next file would stall on it
                // too; a reading that wandered or answered nothing is this
                // file's own, and the next one may read well.
                if is_stall_message(&msg) {
                    ingest_holdoff::cool_down(&file.path, &msg);
                }
                return Ok(());
            }
            let n = failures.record(&file.path, format!("{e:#}"));
            tracing::warn!(
                target: "zend::code_read::ingest",
                file = %file.path,
                superseded_kept = superseded.len(),
                "file ingest failed (will retry next run; prior generation kept live): {e:#}",
            );
            ingest_holdoff::cool_down(&file.path, &msg);
            if n > MAX_DECODE_FAILURES {
                tracing::error!(
                    target: "zend::code_read::ingest",
                    n, cap = MAX_DECODE_FAILURES,
                    "code_read ingest stopping early: failure cap reached (last: {e:#})",
                );
                failures.set_abort();
            }
            return Ok(()); // conv drops → slot freed; superseded generation still live.
        }
    };
    tracing::debug!(
        target: "zend::code_read::ingest",
        file = %file.path,
        summary_chars = summary.len(),
        "file conversation answered",
    );

    // Tag the conversation: the content key commits it — what the pass plans
    // against, the fast path finds and retrieval scopes by — `path` and `blob`
    // are its parts, `lines` is what the working set checks the chain's reads
    // covered, the rest is diagnostic. One record, so the line count lands with
    // the key.
    let mut tags = BTreeMap::new();
    tags.insert("kind".to_string(), "code_read".to_string());
    tags.insert(PATH_KEY.to_string(), file.path.clone());
    tags.insert(BLOB_KEY.to_string(), file.blob.to_string());
    tags.insert(LINES_KEY.to_string(), lines.to_string());
    tags.insert(CONTENT_KEY.to_string(), job.key.to_string());
    if let Some(bits) = crate::branch_ingest::keys::model_bits() {
        tags.insert(
            crate::branch_ingest::keys::QUANT_KEY.to_string(),
            bits.to_string(),
        );
    }
    tags.insert("lang".to_string(), format!("{:?}", file.language));
    if !file.branches.is_empty() {
        tags.insert(BRANCHES_KEY.to_string(), branches_value(&file.branches));
    }
    if let Some(at) = &file.at {
        tags.insert(COMMIT_KEY.to_string(), at.to_string());
    }
    if let Err(err) = conv.set_metadata_many(&tags) {
        // Not committed: no pass, fast path or scope will find it. The prior
        // generation stays live, exactly as on the failure path, and this
        // attempt's conversation goes — nothing else retires it before the
        // next boot.
        tracing::warn!(
            target: "zend::code_read::ingest",
            file = %file.path,
            "failed to tag conversation metadata (content key): {err:#}",
        );
        if let Err(err) = ctx.engine.tombstone_timeline(conv.timeline_id()) {
            tracing::warn!(
                target: "zend::code_read::ingest",
                file = %file.path,
                "tombstone of the untagged attempt failed: {err:#}",
            );
        }
        failures.record(
            &file.path,
            format!("the content key was not written: {err:#}"),
        );
        return Ok(());
    }
    ctx.retrieval.mark_stale();
    ingest_rate::note_committed();

    // Deferred tombstone ACTIVATES, now that the new generation is committed.
    if !superseded.is_empty() {
        let e = ctx.engine;
        for tl in &superseded {
            if let Err(err) = e.tombstone_timeline(*tl) {
                tracing::warn!(
                    target: "zend::code_read::ingest",
                    file = %file.path,
                    "deferred tombstone of superseded generation failed: {err:#}",
                );
            }
        }
    }

    // Calibrate it now, while its reading is still resident: the draft forked
    // onto it projects the reading from the card, where a later pass would lift
    // it back from cold first. A failure is calibrated again by the catch-up
    // round ahead of the next pass.
    if let Some(spec) = calibrate {
        if let Err(e) =
            calibrate_committed(ctx, spec, base, conv.timeline_id(), &file.path, job.key)
        {
            if !crate::ingest_cancel::cancelled() {
                tracing::warn!(
                    target: "zend::calibrate",
                    layer = %spec.layer,
                    subject = %file.path,
                    "calibration failed (calibrated again next pass): {e:#}",
                );
            }
        }
    }

    // The file's conversation is now complete: nothing attends it again until a
    // projection retrieves it. Flag it for full KV eviction so the persistence
    // pipeline offloads its turns to cold (NVMe) and frees BOTH the VRAM and RAM
    // copies — otherwise the sealed turns linger hot and accumulate across a
    // large multi-file ingest until the card fills. `FreeSequence` on drop only
    // releases the batch slot, not the sealed KV, so this proactive flag is what
    // actually reclaims the space; `elevate_to_hot` pulls the file back from cold
    // on demand if reselected.
    let flagged = {
        let e = ctx.engine;
        // Read once, never continued: it keeps its turns and no recurrent
        // snapshot (`crate::ingest_snapshots`).
        finished(e, conv.timeline_id(), Unit::File(&file.path));
        e.evict_ingest_timeline(conv.timeline_id())
    };
    tracing::debug!(
        target: "zend::code_read::ingest",
        file = %file.path,
        turns = flagged,
        "flagged completed file conversation for full KV eviction to cold",
    );

    // `conv` drops here → FreeSequence releases the GPU slot; the sealed
    // turns and metadata remain in the substrate.
    Ok(())
}
