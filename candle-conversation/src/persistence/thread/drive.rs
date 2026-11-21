//! The persistence loop: each wake runs one pass that moves turns from hot to
//! warm and from warm to cold, so inference never waits on the disk.
//!
//! The thread binds to the device's context before it allocates pinned host
//! memory, because the context is current per thread and pinned allocation
//! fails silently without it. A flush request from a scheduler under pressure is
//! answered once the pass's hot-to-warm copies are installed. A shutdown runs a
//! full drain that repeats until nothing is left hot without warm, warm without
//! cold, or a section without its redo-log copy, and it stops after a pass that
//! did not shrink the backlog, so a turn that keeps failing cannot hold up exit.
//! Segment maintenance is skipped during that drain, since reclaim is
//! re-runnable and the drain must not queue behind it.

use super::*;

pub(super) fn run_loop(
    conversation: Conversation,
    backings: Arc<Vec<ChunkedKvBacking>>,
    device: Device,
    compression_policy: Option<CompressionPolicy>,
    trigger_rx: Receiver<()>,
    flush_rx: Receiver<FlushRequest>,
    shutdown_rx: Receiver<()>,
    backlog: BacklogGauge,
    warm: WarmShare,
    warm_asks: WarmAsks,
) {
    // Bind this thread to the device's CUDA context BEFORE any pinned
    // alloc — the CUDA context is per-thread on Windows and the
    // persistence thread was spawned without an active binding.
    // `cuMemHostAlloc` returns `CUDA_ERROR_NOT_INITIALIZED` otherwise,
    // silently falling back to non-pinned heap memory and crippling
    // the hot→warm DtoH leg.
    if let candle::Device::Cuda(d) = &device {
        let _ = d.cuda_context().bind_to_thread();
    }

    // Re-used pinned host scratch for the hot→warm DtoH. Eagerly allocated at
    // thread start — while the host heap is still clean and unfragmented — at
    // the full migration staging cap, so it holds the largest batch any pass can
    // produce and **never has to re-grow** later. A late re-grow to a large
    // contiguous size is exactly what fails once the warm tier has fragmented
    // the heap (the OOM that aborted a full overnight load); pre-sizing here
    // sidesteps it. Stays allocated across passes.
    let mut pinned_scratch: Option<PinnedBuf> = preallocate_pinned_scratch(
        candle_nn::kv_cache::MIGRATION_STAGING_CAP_BYTES,
        "persistence::pinned_scratch",
    );
    // Rate-limit the maintenance scan (see `MAINTENANCE_INTERVAL`). Start it a
    // full interval in the past so the first pass is eligible.
    let mut last_maintenance = Instant::now()
        .checked_sub(MAINTENANCE_INTERVAL)
        .unwrap_or_else(Instant::now);
    // A full drain the last pass took up mid-pass, owed to this one.
    let mut carried: Option<FlushRequest> = None;
    // A maintenance op an earlier pass stepped aside from for a flush.
    let mut maintenance_op: Option<MaintenanceOp> = None;
    loop {
        // Wait for trigger, shutdown, or the periodic tick — whichever
        // fires first. The `default` arm fires after `DEFAULT_TICK` if
        // nothing else has woken the thread. A carried drain runs at once.
        let mut shutting_down = false;
        // A blocking flush request (scheduler under VRAM pressure), answered by
        // this pass once its hot→warm migration has moved what it asked for.
        let mut flush_ack: Option<FlushRequest> = carried.take();
        if flush_ack.is_none() {
            crossbeam::channel::select! {
                recv(trigger_rx) -> _ => {}
                recv(flush_rx) -> ack => { flush_ack = ack.ok(); }
                recv(shutdown_rx) -> _ => { shutting_down = true; }
                default(DEFAULT_TICK) => {}
            }
        }

        // Stamp the backlog this pass is about to face (pre-drain). Held high
        // for the whole pass so the scheduler keeps ingest throttled while a
        // long drain is in flight — the conservative direction.
        backlog.stamp_warm(conversation.read().pending_warm_bytes());

        // On the shutdown iteration, SKIP maintenance. The full drain below needs
        // the persistence lock immediately; a segment-maintenance relocation holds
        // it ~10s each, and running one here first is the ~74s shutdown stall that
        // let an impatient second Ctrl-C / hard-kill abort the process BEFORE the
        // drain ran — losing the entire un-drained hot tail (the observed
        // repaired_integrity churn on the next reload). Reclaim is re-runnable
        // background work; it resumes on the next start. Durability at exit is the
        // drain's job, and it must not queue behind compaction.
        let run_maintenance = !shutting_down && last_maintenance.elapsed() >= MAINTENANCE_INTERVAL;
        if run_maintenance {
            last_maintenance = Instant::now();
        }
        // An op left in flight is let go for the same reason: its sources stay
        // until a later op reclaims them, and nothing it planned is lost.
        if shutting_down {
            if let Some(op) = maintenance_op.take() {
                conversation.abandon_maintenance(op);
            }
        }
        // Answered by the pass as soon as what it asks for is warm; a request sent
        // while the pass runs is taken up at its next group.
        let mut flush = PassFlush::new(flush_ack, Some(&flush_rx));
        run_pass(
            &conversation,
            &backings,
            &device,
            compression_policy.as_ref(),
            &mut pinned_scratch,
            PassMaintenance {
                op: &mut maintenance_op,
                may_start: run_maintenance,
            },
            &backlog,
            &warm,
            &warm_asks,
            &mut flush,
        );
        carried = flush.carried();

        // Re-stamp the residual backlog (post-drain) so the signal decays as
        // soon as the pass installs warm, letting admission reopen promptly.
        backlog.stamp_warm(conversation.read().pending_warm_bytes());

        if shutting_down {
            // A read-only substrate has no durable tier to drain into: its
            // turns were only ever going to live in RAM, and they end with the
            // process. Nothing to drain, nothing to commit.
            if conversation.is_read_only_substrate() {
                return;
            }
            // FULL drain before exit. The engine joins the scheduler BEFORE it
            // signals us (see `ConversationEngine::shutdown`), so no more turns
            // seal — the pending sets only shrink and this loop always converges.
            //
            // Clear the in-flight decode's now-stale working-set pins: normally
            // `snapshot_pending_warm` defers an already-persisted pinned
            // residence's warm copy (the decode is actively attending it), but
            // with the scheduler stopped nothing attends anything, so every hot
            // turn is drainable.
            // Shutdown drain: an empty keep-set stamps nothing, which is right —
            // nothing is attending, so every residence should age out.
            conversation.read().release_working_set_pins();
            // Loop hot→warm→cold (maintenance off) until nothing is left hot
            // without warm, warm without cold, or a sealed section without its
            // redo-log copy — the next boot restores sections from that copy. One `run_pass` moves a turn all
            // the way to enqueued-for-cold, so this normally converges in 1–2
            // passes; the cap is only a wedge-guard against a persistently-failing
            // turn (e.g. a gather error), whose leftover simply re-ingests on
            // reload. After this returns, the engine flushes the off-thread writer
            // so the enqueued warm→cold appends actually reach disk.
            let mut passes = 0usize;
            let mut prev_remaining = usize::MAX;
            loop {
                // Cheap COUNT poll (no work-list clone — see `pending_warm_count` /
                // `pending_cold_count`), matching `run_pass`'s own snapshot filters.
                let (warm_left, cold_left, sections_left) = {
                    let read = conversation.read();
                    (
                        read.pending_warm_count(),
                        read.pending_cold_count(),
                        read.pending_section_cold_count(),
                    )
                };
                let remaining = warm_left + cold_left + sections_left;
                if remaining == 0 {
                    break;
                }
                // No-progress guard: if a pass did not shrink the backlog, a turn is
                // persistently failing to drain (e.g. a repeatable GPU gather error).
                // Retrying the same failing gather only stalls exit, so stop now —
                // the leftover re-ingests on the next reload. This breaks after ONE
                // stuck pass instead of grinding through the whole absolute cap.
                if remaining >= prev_remaining {
                    tracing::warn!(
                        warm_left,
                        cold_left,
                        sections_left,
                        passes,
                        "persist: shutdown drain made no progress (a turn won't drain) \
                         — the leftover re-ingests on next reload",
                    );
                    break;
                }
                // Absolute backstop against a pathological oscillation that shrinks
                // then regrows without converging.
                if passes >= SHUTDOWN_DRAIN_MAX_PASSES {
                    tracing::warn!(
                        warm_left,
                        cold_left,
                        sections_left,
                        passes,
                        "persist: shutdown drain hit the pass cap with work remaining \
                         — the leftover turns re-ingest on next reload",
                    );
                    break;
                }
                tracing::info!(
                    pass = passes,
                    warm_left,
                    cold_left,
                    sections_left,
                    "persist: shutdown drain hot→warm→cold",
                );
                prev_remaining = remaining;
                run_pass(
                    &conversation,
                    &backings,
                    &device,
                    compression_policy.as_ref(),
                    &mut pinned_scratch,
                    // No maintenance during the shutdown drain; the op in
                    // flight was let go above.
                    PassMaintenance {
                        op: &mut None,
                        may_start: false,
                    },
                    // Still stamped: the drain is what finally empties the
                    // backlog, and leaving the gauge holding its pre-shutdown
                    // reading would publish a deficit that never clears.
                    &backlog,
                    &warm,
                    // A shutdown drain warms nothing: what it reads, it reads
                    // to put away.
                    &WarmAsks::default(),
                    // Nothing waits on the shutdown drain's passes.
                    &mut PassFlush::new(None, None),
                );
                passes += 1;
            }
            // Group-commit any records appended directly on this thread. The
            // authoritative final fsync of the warm→cold appends happens when the
            // engine flushes the off-thread writer after this thread joins.
            if let Err(e) = conversation.commit_persistence() {
                tracing::warn!("persist: shutdown commit failed: {e}");
            }
            return;
        }
    }
}

/// Ask the VRAM governor to escalate recovery after a copy failed to
/// allocate. Only fires for VRAM-exhaustion errors
/// (`candle_nn::kv_cache::is_device_oom`), the strongest "critically full"
/// signal; other error kinds won't be helped by evicting, so they don't
/// signal.
pub(super) fn signal_vram_starvation(device: &Device, err: &candle::Error) {
    if !candle_nn::kv_cache::is_device_oom(err) {
        return;
    }
    if let candle::DeviceLocation::Cuda { gpu_id } = device.location() {
        if let Some(gov) = candle::vram::get(gpu_id) {
            gov.signal_starvation();
        }
    }
}

/// Wait, up to `within`, for the scheduler's next gap between forwards to make
/// every arena a turned-away pass asked these backings for. `true` once none is
/// owed; `false` if the wait ran out with some still owed.
pub(super) fn await_owed_arenas(backings: &[ChunkedKvBacking], within: Duration) -> bool {
    let until = Instant::now() + within;
    loop {
        if !backings.iter().any(ChunkedKvBacking::arenas_owed) {
            return true;
        }
        if Instant::now() >= until {
            return false;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}
