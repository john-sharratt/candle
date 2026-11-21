//! One persistence pass: copy hot turns to warm, write warm turns to the cold
//! log, purge warm RAM to its budget, and run segment maintenance.
//!
//! The hot-to-warm step copies rather than quantizes, because the scheduler has
//! already quantized each turn in the gap between forwards. The copy is grouped
//! and bounded, and each group's hold against chunk relocation is released only
//! after the group's device work is fenced, so a compaction cannot move chunks a
//! copy is still reading. A group that a forward turned away runs again at the
//! next gap within the same pass. Segment maintenance starts at most once per
//! interval and steps aside between batches when a flush is waiting, so a flush
//! never waits out a whole compaction.

use super::*;

/// A pass's segment maintenance: the op an earlier pass left in flight, and
/// whether this pass may start one (`MAINTENANCE_INTERVAL`).
pub(super) struct PassMaintenance<'a> {
    pub(super) op: &'a mut Option<MaintenanceOp>,
    pub(super) may_start: bool,
}

pub(super) fn run_pass(
    conversation: &Conversation,
    backings: &[ChunkedKvBacking],
    device: &Device,
    compression_policy: Option<&CompressionPolicy>,
    pinned_scratch: &mut Option<PinnedBuf>,
    maintenance: PassMaintenance<'_>,
    backlog: &BacklogGauge,
    warm: &WarmShare,
    warm_asks: &WarmAsks,
    flush: &mut PassFlush<'_>,
) {
    let t_pass = std::time::Instant::now();

    // ── Phase 1: hot → warm ─────────────────────────────────────────────
    //
    // A copy, not a quantize: a turn's seal quantized it in the gap
    // (`docs/kv_quantize_in_the_gap.md` §5), and a turn the seal gap left
    // native is the gap's to finish (`scheduler/gap_residence.rs`) — the
    // groups below leave it be until it is. One group per compression
    // override, as the residences are listed by it; each group's sequences
    // are cloned under its own migrate guard below, copied to the warm tier in
    // one batched DtoH, and installed under one substrate read.
    let policies = conversation.read().pending_warm_policies();
    let n_layers = backings.len();

    // Cross-thread write→read barrier. This pass reads each sealed turn's K/V
    // for the copy below, bytes the scheduler thread's forwards wrote — a
    // lossless turn's native K/V among them. The ordering between two host
    // threads issuing onto one stream is not guaranteed, so the device is
    // synchronised once, up front, so every prior write is retired before the
    // copy reads a source arena.
    //
    // **It fences the maintenance sweep as well.** The sweep after the copy
    // touches segment state an in-flight wave may still be reading, and this is
    // what retires it; removing the barrier took `CUDA_ERROR_ILLEGAL_ADDRESS`
    // from 0 to 199 in a run. The drain costs the depth of whatever inference is
    // in flight — a `residences=1 mib=0` pass measured 831 ms against 3 ms
    // uncontended — and is timed from here, so the figure means the wait.
    let t_sync = std::time::Instant::now();
    if let Err(e) = device.synchronize() {
        tracing::warn!(
            target: "candle_conversation::persistence::tier",
            "persist: pre-migrate device sync failed: {e:?}"
        );
    }
    // Phase-timing breakdown so we can see *where* the hot→warm drain spends its
    // time (the demote starves when this can't keep up with the ingest seal rate).
    let sync_pre_ms = t_sync.elapsed().as_millis() as u64;
    let t_migrate = std::time::Instant::now();

    let mut copy_ms = 0u64;
    // A single-latent backing (DeepSeek's K≡V window) compresses per 128-dim
    // band through its own compressor, and the gap takes no single-latent
    // backing: its turns are copied as they stand, never left for a gap.
    let single_latent = backings.first().is_some_and(|b| b.single_latent());
    // **Defer rather than race a compaction.** This marker is also the hold that
    // keeps a KV compaction from relocating the chunks whose addresses the migrate
    // below is about to capture; a pass in flight means those locations are moving
    // under us, and the copy would put another sequence's KV in the warm tier. Never
    // waits — see `migrate_flight` — so the pass ahead of us finishes and the next
    // persistence round picks this work up unchanged.
    // **Per group, not per pass, and each group's GPU work is fenced before the
    // guard goes back.**
    //
    // The guard is what keeps a KV compaction from relocating the chunks whose
    // addresses this migrate has captured — a pin keeps the arena alive and says
    // nothing about which slot of it the chunk is in, so a relocation underneath
    // makes the copy put another sequence's KV in the warm tier. Held across the
    // whole batch it starves compaction exactly when compaction is most needed: a
    // mass eviction is what produces both the fragmentation and the hot→warm work
    // that has to precede it. Measured: 50 of 65 passes refused, and the pools at
    // 72% through the burst.
    //
    // Releasing per group is only sound because the fence comes with it. The guard
    // protects captured *addresses*, and a group's addresses stay live until its
    // kernels retire, so handing the guard back while its copies are still queued
    // would hand a compaction ground it is still reading. The batch-wide sync below
    // stays as well — it is what `install_warm_and_hot` and the next projection
    // depend on, for reasons that have nothing to do with this guard.
    //
    // A group that cannot get the guard is skipped, not abandoned: it is still hot,
    // still flagged, and the next pass migrates it. Everything after this loop — the
    // cold writes, the maintenance sweep — reads the substrate rather than raw chunk
    // addresses, so it is unaffected by a relocation either way.
    //
    // **Nothing a group captured outlives its guard.** Its sequences are cloned
    // after the guard is taken, and its results are installed — and every clone
    // dropped — before it goes back. A compaction sweeps the holders it can reach,
    // and this thread's locals are not among them: a batch snapshotted up front and
    // held across the groups was a holder of every pending residence the sweep
    // never visited. It kept the relocated slots' sources alive, so their copies were
    // claimed and never used — 28,869 of 43,647 relocated slots in one pass, measured
    // when a section quantize swapped a section's hot form for its quantized one and
    // left the snapshot the only holder of the native chunks. And the install then
    // wrote gids captured before the relocation back into the substrate. Snapshotted
    // under the guard, a group names exactly what the substrate names at that
    // moment, and no compaction can run until it is installed.
    //
    // **And a group is bounded, not a whole compression class.** A class is every
    // pending residence that shares an override, which on a workload with one
    // override is the entire backlog — so "per group" was per pass in practice,
    // and the guard was held across the backlog's migrate, its device-wide fence
    // and its install exactly when a mass retirement had produced both the backlog
    // and the fragmentation. Measured on the Qwen3-30B engine probe: compaction
    // refused 32–63 times a run while the pools sat at 41–64% efficiency through
    // the bursts. `MAX_GROUP_RESIDENCES` per hold, and a refused pass — which sets
    // `COMPACTION_WAITING` — gets in before the next one.
    //
    // How often one pass waits for a gap and runs a turned-away group again,
    // and how long it waits for each: a gap is milliseconds under load, so the
    // bound is for a scheduler that has stopped looping, not a slow one.
    const MAX_GAP_RETRIES: usize = 16;
    const GAP_WAIT: Duration = Duration::from_millis(500);
    let mut gap_retries = 0usize;
    let mut deferred_groups = 0usize;
    let mut hot_to_warm_bytes: u64 = 0;
    let mut hot_to_warm_count: usize = 0;
    let mut install_ms = 0u64;
    'policies: for cc in policies {
        // A turn the gap still owes a quantize is left for it: copied now, its
        // warm and cold copies would keep the native bytes the gap replaces.
        let gap_owes = !single_latent && effective_turn_policy(compression_policy, cc).is_some();
        loop {
            let Some(mut group_guard) = try_migrate_flight() else {
                deferred_groups += 1;
                continue 'policies;
            };
            let mut read: Vec<(ResidenceIndex, Arc<HotKv>)> = conversation
                .read()
                .snapshot_pending_warm_for(cc)
                .into_iter()
                .filter(|(idx, hot)| {
                    let whole = hot.len() == n_layers;
                    if !whole {
                        tracing::warn!(
                            "persist: hot→warm layer-count mismatch for {idx:?} — skipping"
                        );
                    }
                    whole && !(gap_owes && holds_native(hot))
                })
                .collect();
            let more = read.len() > MAX_GROUP_RESIDENCES;
            read.truncate(MAX_GROUP_RESIDENCES);
            let group: Vec<(ResidenceIndex, Vec<SealedSequence>)> =
                read.iter().map(|(idx, hot)| (*idx, hot.to_vec())).collect();
            // The hot copies the copy reads, over which alone its installs land.
            let copied: HashMap<ResidenceIndex, Arc<HotKv>> = read.into_iter().collect();
            // Registered while the flight still holds chunk locations, so no
            // compaction can start between the snapshot and the registration and
            // spend claims on chunks this group keeps alive. Dropped last, once
            // every handle the group took is gone.
            let mut held = HeldChunks::new();
            held.hold(group.iter().flat_map(|(_, hot)| hot.iter()));
            let mut installs = Vec::new();
            // The device's stream, leased for this group's GPU work alone and
            // taken before any lock the group takes: a wave segment waits for
            // nothing while it records, and runs as issued while this is held.
            let Device::Cuda(cuda) = device else {
                unreachable!("substrate-persistence runs on a CUDA device")
            };
            let copy_stream = cuda.cuda_stream();
            let wave_deferred = copy_group_hot_to_warm(
                backings,
                device,
                &copy_stream,
                &group,
                &mut group_guard,
                pinned_scratch,
                n_layers,
                &mut installs,
                &mut copy_ms,
            );
            drop(copy_stream);
            let t_install = std::time::Instant::now();
            let (bytes, count) = install_hot_to_warm(conversation, &copied, installs);
            install_ms += t_install.elapsed().as_millis() as u64;
            hot_to_warm_bytes = hot_to_warm_bytes.saturating_add(bytes);
            hot_to_warm_count += count;
            // **The group's clones before the hot copies they share chunks with.**
            // A `HotKv` hands its chunks to the reclaim queue when it drops off the
            // device's thread; a bare clone frees them where it drops. Dropped
            // first, the clones are never the last handle, so the native chunks a
            // pass may be stepping around are freed on the thread that compacts.
            drop(group);
            drop(copied);
            drop(held);
            drop(group_guard);
            // A sized flush is answered the moment enough is warm — including
            // one sent while this pass was already running — and the rest of the
            // backlog moves on without a caller waiting on it.
            flush.at_group(hot_to_warm_bytes);
            // **A group a forward turned away runs again at the next gap, in
            // this pass.** It needed an arena that could only be made between
            // forwards; the scheduler makes every owed arena in its next gap,
            // which under a steady ingest is milliseconds away. Leaving it for
            // the next trigger instead cost about fifteen seconds a time — a
            // drain of a few MB/s against an ingest producing KV far faster,
            // with the backlog standing on the GPU and taking the expert zone's
            // ground until a lull let the passes through.
            if wave_deferred && count == 0 && gap_retries < MAX_GAP_RETRIES {
                gap_retries += 1;
                if await_owed_arenas(backings, GAP_WAIT) {
                    continue;
                }
            }
            // Nothing installed means the group could not move this pass, and
            // re-snapshotting would hand back the same residences.
            if !more || count == 0 {
                break;
            }
        }
    }
    if deferred_groups > 0 {
        tracing::debug!(
            target: "candle_conversation::persistence::tier",
            deferred_groups,
            "hot→warm groups deferred: a KV compaction is relocating chunks",
        );
    }
    let migrate_ms = t_migrate.elapsed().as_millis() as u64;
    let t_sync_post = std::time::Instant::now();
    // Every group's copy synchronised before installing; this sync runs on a fixed
    // cadence regardless of load (see the pass-timing log below, which fires
    // even with nothing to migrate), so it is the daemon's most reliable "the
    // device is still alive" heartbeat — closing any out-of-memory streak
    // `gpu_poison` is tracking before a resolved episode gets mistaken for a
    // still-open one much later.
    match device.synchronize() {
        Ok(()) => candle::gpu_poison::note_device_ok(),
        Err(e) => tracing::warn!(
            "cache: device sync after hot→warm pass failed: {e:?} (last CUDA kernel on this thread: {})",
            candle::last_cuda_kernel_launch()
        ),
    }
    let sync_post_ms = t_sync_post.elapsed().as_millis() as u64;
    // Log the breakdown when the pass did real work OR was slow even with nothing
    // to install (isolates a dominant `device.synchronize()` cost). This is the
    // number that tells us whether the drain is sync-bound, the batched copy,
    // or install-lock-bound.
    let total_ms = t_pass.elapsed().as_millis() as u64;
    if hot_to_warm_count > 0 || sync_pre_ms + migrate_ms + sync_post_ms > 50 {
        tracing::debug!(
            target: "candle_conversation::persistence::tier",
            residences = hot_to_warm_count,
            mib = hot_to_warm_bytes / (1 << 20),
            n_layers,
            sync_pre_ms,
            migrate_ms,
            copy_ms,
            sync_post_ms,
            install_ms,
            total_ms,
            "hot→warm pass timing"
        );
    }
    if hot_to_warm_count > 0 {
        relief_trace::note(
            "tier",
            "hot_to_warm",
            hot_to_warm_count as u64,
            hot_to_warm_bytes,
        );
    }
    // Feed the instrumented migration panel straight from the pass (no log
    // tail): a pass that moved something, and now and then one that found
    // nothing to move, so a quiet drain does not read as a stopped one.
    if hot_to_warm_count > 0 {
        push_migrate(migrate_sample(
            hot_to_warm_count as u64,
            hot_to_warm_bytes / (1 << 20),
            migrate_ms,
            copy_ms,
            total_ms,
        ));
    } else {
        push_idle_migrate(total_ms);
    }
    // Backlog split: how much of the hot-without-warm set was DEFERRED because it
    // is the in-flight decode's pinned working set (the tier-livelock fix) vs. a
    // genuine drain the migrate must still catch up on. Fires only when something
    // was pin-deferred, so a healthy idle pass stays quiet.
    // The resident KV by format, for the arena panel — on the pass the panel
    // is owed one, under its own short read. Keyed by this substrate's tenant,
    // since every tenant's thread takes its own and the panel shows their sum.
    let mix_key = conversation.tenant_rank().index();
    if mix_due(mix_key) {
        let rows = conversation
            .read()
            .hot_format_census()
            .rows()
            .into_iter()
            .map(|(tag, tokens)| (format!("{tag:?}"), tokens))
            .collect();
        push_mix(mix_key, rows);
    }
    {
        let (dc, db, pc, pb) = conversation.read().warm_backlog_split();
        // Stamp unconditionally so the scheduler's admission ceiling sees the
        // pin discount fall back to zero as soon as the working set releases —
        // a gauge that only ever rises would throttle admission forever.
        backlog.stamp(db, pb);
        // A waiting flush is answered here, once the hot→warm installs are
        // done: that is the work it waits for — the just-sealed turns now hold
        // a warm copy and may be evicted from VRAM. The cold writes and the
        // maintenance below are durability and housekeeping, and the scheduler,
        // blocked in relief, used to wait through them as well.
        flush.finish();
        if pc > 0 {
            tracing::trace!(
                target: "candle_conversation::persistence::tier",
                pinned_skipped = pc,
                pinned_mib = pb / (1 << 20),
                drainable = dc,
                drainable_mib = db / (1 << 20),
                "hot→warm backlog split (pinned working set deferred from drain)"
            );
        }
    }

    // ── Phase 2: warm → cold ────────────────────────────────────────────
    //
    // Same batching shape as phase 1: snapshot once, run all gathers +
    // chunk writes back-to-back with no substrate lock held, then take
    // **one** write lock and install every cold reference.
    //
    // The redo-log writes themselves share `SubstratePersistence`'s
    // internal `Mutex`, so they serialise within this loop — that's
    // intentional, every Chunk record gets a unique log offset.
    //
    // `snapshot_pending_cold` returns each residence's **warm** bytes —
    // the stored form, CPU arenas — so the gather here reads host memory,
    // and nothing it holds is ground a KV compaction relocates.
    //
    // A read-only substrate has no cold tier to write: its turns stay warm,
    // and Phases 2, 2.5, 3 and 4 have nothing to do. Phase 1 above and the
    // warm purge below are RAM/VRAM only and run as usual.
    let read_only = conversation.is_read_only_substrate();
    let pending_cold = if read_only {
        Vec::new()
    } else {
        conversation.read().snapshot_pending_cold()
    };
    let mut warm_to_cold_bytes: u64 = 0;
    let mut warm_to_cold_count: usize = 0;
    for (idx, stream_id, hot) in pending_cold {
        if backings.len() != hot.len() {
            tracing::warn!(
                "persist: warm→cold layer mismatch — {} backings vs {} sealed",
                backings.len(),
                hot.len()
            );
            continue;
        }
        let grid = match build_grid(&hot, backings, device) {
            Ok(g) => g,
            Err(e) => {
                tracing::warn!("persist: warm→cold gather failed for {idx:?}: {e}");
                continue;
            }
        };
        warm_to_cold_bytes = warm_to_cold_bytes.saturating_add(grid.bytes() as u64);
        warm_to_cold_count += 1;
        tracing::trace!(
            target: "candle_conversation::persistence::tier",
            residence = idx.0,
            stream_id = stream_id.0,
            bytes = grid.bytes(),
            "enqueue warm → cold (off-thread writer append + install_cold)"
        );
        // Hand the redo-log append + `install_cold` to the off-thread writer: the
        // GPU gather is done here, so the cold-write disk I/O no longer blocks this
        // thread's next hot→warm VRAM-relief pass, and `install_cold` runs
        // post-write so `slot.cold` is only ever set on durable data (the
        // `purge_warm` RAM-unload invariant). `snapshot_pending_cold` skips the
        // now `cold_pending` residence until the write lands, so no double-write;
        // the writer's dual-cap queue backpressures a sustained cold backlog.
        conversation.enqueue_kv_cold(idx, stream_id, grid);
    }

    // ── Phase 2.5: section persist ─────────────────────────────────────
    //
    // Sections are pinned and never enter `hot_lru` / `warm_lru`, so
    // the two-phase walk above misses them.  This sub-phase walks the
    // substrate's section map directly, finds any section whose
    // residence has hot bytes installed under a non-default stream id
    // and no cold copy yet, and writes its chunks to the redo log.
    //
    // **No quantize, no `replace_section_hot`.** Section bytes are
    // immutable once committed by the scheduler's `SealAction::Section`
    // handler: it applies the configured `compression_policy` inline
    // (synchronous, on the main thread) before installing the residence,
    // so `slot.hot` already holds the final (possibly quantized) form
    // by the time we read it here.  The persistence thread's job is
    // purely to gather those final bytes off GPU and append them to the
    // redo log — no mutation of substrate state.
    //
    // **Gathered a bounded group at a time under a chunk-location hold, and no clone
    // of a section's hot form outlives it** — the rule Phase 1 follows, for the same
    // reason. A section's hot bytes are device chunks a KV compaction relocates, and
    // this thread's locals are not among the holders a compaction sweeps. Snapshotted
    // up front and held across every section's gather and redo-log write, the clones
    // kept each relocated section's source slots alive after its residence had moved
    // on, so pass after pass planned them again and found nobody to move onto the copy:
    // ~180,000 of ~190,000 relocated slots `unwitnessed` per pass on the Qwen3-30B,
    // for the seconds the 109-section tool catalog took to reach the log after a base
    // conversation was built. Under the hold a group's gather reads exactly what the
    // residences name; the write that follows carries host bytes only.
    const MAX_GROUP_SECTIONS: usize = 8;
    let mut section_to_cold_bytes: u64 = 0;
    let mut section_to_cold_count: usize = 0;
    let section_drain = if read_only {
        HeldDrain::default()
    } else {
        drain_in_held_groups(
            MAX_GROUP_SECTIONS,
            try_hold_chunk_locations,
            || {
                conversation
                    .read()
                    .snapshot_pending_section_cold()
                    .into_iter()
                    .map(|(idx, stream_id, hot)| (idx, (stream_id, hot)))
                    .collect()
            },
            |idx, (stream_id, hot)| {
                if hot.len() != n_layers {
                    tracing::warn!(
                        "persist: section {idx:?} hot has {} layers, expected {n_layers} — \
                         skipping",
                        hot.len()
                    );
                    return None;
                }
                match build_grid(&hot, backings, device) {
                    Ok(grid) => Some((stream_id, grid)),
                    Err(e) => {
                        tracing::warn!("persist: section gather failed for {idx:?}: {e}");
                        None
                    }
                }
            },
            |gathered| {
                let (bytes, count) = persist_section_group(conversation, gathered);
                section_to_cold_bytes = section_to_cold_bytes.saturating_add(bytes);
                section_to_cold_count += count;
            },
        )
    };
    if section_drain.deferred {
        tracing::debug!(
            target: "candle_conversation::persistence::tier",
            groups = section_drain.groups,
            "section persist deferred: a KV compaction is relocating chunks",
        );
    }

    // ── Phase 2.6: warm-tier RAM purge ─────────────────────────────────
    //
    // The warm (RAM) copies produced by Phase 1 are durable the moment
    // their cold copy lands (Phase 2 / 2.5 above), but nothing else drops
    // them: `install_cold` frees hot, never warm, and this is the only
    // `purge_warm_to_budget` call site. During a bulk ingest (calibration,
    // repo scan) the warm tier would otherwise grow unbounded and exhaust
    // host RAM (the OOM that killed a full load). Run the LRU purge here,
    // where warm is actually produced, dropping the least-recently-used warm
    // copies until the OS holds at least `max(2 GiB, 5% × total)` free. Every
    // warm residence reachable here is already cold-backed (Phase 2 persisted
    // this pass's whole warm-without-cold set), so no un-persisted bytes are
    // dropped. `incoming = 0`: we bound the existing footprint, not reserve
    // for an upcoming allocation.
    //
    // Gated on there BEING warm to purge, not on this pass having produced it.
    // The migration above is not warm's only producer: the resume path installs
    // warm directly. A gate on "did this pass migrate?" therefore goes silent in
    // exactly the state that most needs the purge — a workspace whose hot→warm
    // and warm→cold sets have both drained to empty while warm stays resident in
    // RAM. That state pins free RAM under the floor indefinitely, which the
    // scheduler reads as permanent host-RAM pressure and answers by throttling
    // admission to its floor, forever, against a cause VRAM admission cannot
    // move. Gating on the warm population instead costs one OS query per pass
    // whenever anything is warm, and `purge_warm_to_budget` is a no-op while warm
    // is comfortable.
    let (warm_resident, warm_bytes, held) = {
        let read = conversation.read();
        let held = read.held_payload_bytes();
        (
            read.warm_resident_count(),
            read.warm_resident_bytes(),
            held.wide_q_sigs + held.projection_events,
        )
    };
    backlog.stamp_residents(warm_resident as u64, warm_bytes, held);
    warm.hold(warm_bytes);
    let asked = warm_asks.take();
    if warm_resident > 0 || !asked.is_empty() {
        let mut sys = System::new();
        sys.refresh_memory();
        // Budget-based purge: warm may hold up to what RAM affords after the OS
        // buffer, pinned pools, and fully-reserved weights. A zero budget means
        // this machine cannot afford a standing warm tier — everything
        // cold-backed drains straight out. The budget is the machine's, shared
        // by every substrate the engine holds, so this one purges to its share
        // of it (`warm_ledger`). `purge_warm_to_budget` logs its own batch
        // summary (see substrate.rs).
        let budget = candle::vram::host_ram_budget(sys.total_memory());
        let target = warm.target(budget.kv_warm_budget_bytes);
        if warm_resident > 0 {
            let _ = conversation.read().purge_warm_to_budget(target, 0);
        }
        // ── Phase 2.65: warm what was asked for, in the room the purge left ──
        //
        // After the purge, so an ask never pushes the tier past its share and
        // the next pass's purge never sheds what this one just warmed. Every
        // turn read here is cold-backed by construction, so a later purge may
        // drop it again without loss.
        if !asked.is_empty() {
            let room = target.saturating_sub(conversation.read().warm_resident_bytes());
            let warmed = conversation.warm_turns_from_cold(backings, &asked, room);
            if warmed.turns > 0 || warmed.failed > 0 || warmed.dropped > 0 {
                tracing::debug!(
                    target: "candle_conversation::persistence::tier",
                    asked = asked.len(),
                    warmed = warmed.turns,
                    mib = warmed.bytes / (1024 * 1024),
                    skipped = warmed.skipped,
                    dropped = warmed.dropped,
                    failed = warmed.failed,
                    "cold → warm on request"
                );
            }
        }
    }

    // ── Phase 2.7: the memory census's gauge ──────────────────────────────
    //
    // Once a minute, what this store's in-memory structures hold
    // (`docs/memory_census.md` §5). Here because this pass already holds the
    // store off every request path, and the gauge walks its maps. Built with
    // the `memory-census` feature.
    #[cfg(feature = "memory-census")]
    {
        let _scope = enter(Scope::Census);
        gauge_if_due(conversation, &conversation.substrate_dir());
    }

    // ── Phase 3: fsync ─────────────────────────────────────────────────
    //
    // Group-commit anything the previous phases staged. No-op when both
    // phases skipped — e.g. an idle workspace just ticked and found
    // nothing to do.
    if !read_only {
        if let Err(e) = conversation.commit_persistence_if_pending() {
            tracing::warn!("persist: fsync failed: {e}");
        }
    }

    // ── Phase 4: segment maintenance ───────────────────────────────────
    //
    // Reclaim dead weight incrementally — one segment drop / compact / combine
    // at a time (`persistence/maintenance.rs`). A new op starts at most every
    // `MAINTENANCE_INTERVAL` (the candidate scan is O(live records), so it's
    // decoupled from the trigger rate); one already in flight continues. The
    // relocation I/O runs under the persistence lock only (phased locking),
    // never the substrate write lock, so it can't stall decode.
    //
    // **A flush waiting steps the op aside.** The flush is this thread's own
    // work, so it would otherwise wait out every batch of the op — a 2.1 s
    // compaction was a 2.4 s relief wait inside a prefill. The op stops between
    // batches, the next pass serves the flush in its hot→warm phase, and the
    // op carries on after it ([`MaintenanceOp`]).
    let PassMaintenance { op, may_start } = maintenance;
    #[cfg(feature = "memory-census")]
    let maintenance_scope = enter(Scope::Maintenance);
    match if !read_only && (op.is_some() || may_start) {
        conversation.step_background_maintenance(op, may_start, &|| flush.waiting())
    } else {
        Ok(false)
    } {
        Ok(true) => {
            relief_trace::note("tier", "segment_reclaim", 0, 0);
            tracing::trace!(
                target: "candle_conversation::persistence::tier",
                "persist: substrate maintenance op applied (segment reclaim)"
            );
        }
        Ok(false) => {}
        Err(e) => tracing::warn!("persist: substrate maintenance failed: {e}"),
    }
    #[cfg(feature = "memory-census")]
    drop(maintenance_scope);

    // Per-pass aggregate. Only logged when something actually moved.
    if hot_to_warm_count > 0 || warm_to_cold_count > 0 || section_to_cold_count > 0 {
        tracing::trace!(
            target: "candle_conversation::persistence::tier",
            hot_to_warm_count,
            hot_to_warm_bytes,
            warm_to_cold_count,
            warm_to_cold_bytes,
            section_to_cold_count,
            section_to_cold_bytes,
            "persistence pass complete"
        );
    }
}

/// Gather every layer's `SealedSequence` into a [`TurnChunkGrid`]
/// (CPU-side). Each per-layer entry is a `Vec<ChunkImage>` produced by
/// the existing GPU gather helper.
/// Append one gathered group of sections to the redo log, commit each stream, and
/// install the cold references under one write lock. Answers the bytes and the count
/// installed.
///
/// Takes host bytes only: by the time a group reaches here its snapshots of the
/// sections' device K/V are gone, so a slow write holds nothing a KV compaction has to
/// account for.
pub(super) fn persist_section_group(
    conversation: &Conversation,
    gathered: Vec<(ResidenceIndex, (StreamId, TurnChunkGrid))>,
) -> (u64, usize) {
    let mut installs: Vec<(ResidenceIndex, Vec<StoredSequence>, u64)> =
        Vec::with_capacity(gathered.len());
    for (idx, (stream_id, grid)) in gathered {
        let stored = match conversation.persist_turn_chunks_capture(stream_id, &grid) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("persist: section write failed for {idx:?}: {e}");
                continue;
            }
        };
        if stored.is_empty() {
            continue;
        }
        let total_chunks: usize = stored.iter().map(|s| s.chunks.len()).sum();
        let through = (total_chunks.max(1) - 1) as u64;
        if let Err(e) = conversation.commit_stream_through(stream_id, through) {
            tracing::warn!("persist: section commit_stream for {idx:?} failed: {e}");
        }
        let bytes_for_item: u64 = stored
            .iter()
            .flat_map(|s| s.chunks.iter())
            .map(|c| c.record_len)
            .sum();
        tracing::trace!(
            target: "candle_conversation::persistence::tier",
            residence = idx.0,
            stream_id = stream_id.0,
            bytes = bytes_for_item,
            "persisted section (redo-log append)"
        );
        installs.push((idx, stored, bytes_for_item));
    }
    let mut bytes = 0u64;
    let count = installs.len();
    if !installs.is_empty() {
        let view = conversation.read();
        for (idx, stored, b) in installs {
            bytes = bytes.saturating_add(b);
            view.install_cold(idx, stored);
        }
    }
    (bytes, count)
}

pub(crate) fn build_grid(
    sealed_per_layer: &[SealedSequence],
    backings: &[ChunkedKvBacking],
    device: &Device,
) -> candle::Result<TurnChunkGrid> {
    let mut layers = Vec::with_capacity(sealed_per_layer.len());
    for (backing, seq) in backings.iter().zip(sealed_per_layer) {
        layers.push(seal_to_chunk_images(backing, device, seq)?);
    }
    Ok(TurnChunkGrid::new(layers))
}
