//! The substrate persistence thread — the single owner of the redo-log
//! write path.
//!
//! One thread per workspace, spawned by the scheduler at engine start.
//! Wakes on a 5-second tick or on an external **trigger** (the scheduler
//! sends one after every turn-seal). On every wake it runs a single
//! pass over the substrate:
//!
//! ```text
//!  ┌──────────────────────────────────────────────────────────────────┐
//!  │ 1. Phase hot→warm                                                │
//!  │    Walk `hot_lru`, find slots with `warm = None`, run            │
//!  │    `migrate_sealed_to_cpu` per layer, install warm.              │
//!  │                                                                  │
//!  │ 2. Phase warm→cold                                               │
//!  │    Walk `warm_lru`, find slots with `cold = None`, gather GPU    │
//!  │    bytes to a `TurnChunkGrid`, append `Chunk` records to the     │
//!  │    redo log, capture the per-chunk log offsets, install cold.    │
//!  │                                                                  │
//!  │ 3. Phase fsync                                                   │
//!  │    `commit_if_pending` — group-fsyncs whatever the previous      │
//!  │    phases staged. No-op when nothing was written.                │
//!  └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! The thread holds a [`Conversation`] handle so it shares the same
//! `Arc<RwLock<Substrate>>` and `Arc<Mutex<SubstratePersistence>>` the
//! rest of the engine uses. Read locks are held only long enough to
//! snapshot the work list — the slow CUDA-side migrations and
//! gather/write work happen unlocked. Write locks are taken briefly
//! per residence to install the new tier.

use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use candle::cuda_backend::cudarc::driver::CudaStream;
use candle::quantized::pinned_staging::PinnedBuf;
use candle::Device;
use candle_nn::kv_cache::{ChunkedKvBacking, CompressionPolicy, SealedSequence};
use crossbeam::channel::{self, Receiver, Sender};

use super::cold_load::{preallocate_pinned_scratch, PINNED_PREALLOC_BYTES};
use super::resume::TurnChunkGrid;
use super::transfer::seal_to_chunk_images;
use crate::projection::Conversation;
use crate::substrate::{ConvCompression, ResidenceIndex, StoredSequence};
use std::collections::HashMap;

/// How often the loop wakes up on its own when no triggers arrive.
pub const DEFAULT_TICK: Duration = Duration::from_secs(5);

/// Clone-able trigger for the persistence thread. Held by anything that
/// wants to wake the loop early (e.g. the scheduler signalling "turn just
/// sealed"). Cheap to clone — wraps two `crossbeam::channel` senders: a
/// fire-and-forget wake and a blocking "drain hot→warm now" flush.
#[derive(Clone)]
pub struct PersistenceTrigger {
    tx: Sender<()>,
    /// Carries a one-shot ack sender into the loop: the next pass runs the
    /// hot→warm migration and replies on the ack once pending_warm is drained.
    flush_tx: Sender<Sender<()>>,
}

impl PersistenceTrigger {
    /// Wake the persistence thread now. No-op when the trigger queue is
    /// already full (one wake is as good as several).
    pub fn fire(&self) {
        let _ = self.tx.try_send(());
    }

    /// Block until the persistence thread completes a pass that drains the
    /// pending hot→warm migration, giving substrate offload **complete
    /// priority**. After this returns `true`, just-sealed turns hold a warm
    /// (RAM) copy and so become eligible for hot-tier eviction — the VRAM can
    /// actually move to the substrate. Used by the scheduler under VRAM
    /// pressure, before it evicts hot KV.
    ///
    /// Returns `false` if the request couldn't be queued (a flush is already
    /// in flight) or the pass didn't ack within `timeout` (the timeout is a
    /// safety cap against a wedged thread, not the expected path).
    pub fn flush_blocking(&self, timeout: Duration) -> bool {
        let (ack_tx, ack_rx) = channel::bounded::<()>(1);
        if self.flush_tx.try_send(ack_tx).is_err() {
            return false;
        }
        ack_rx.recv_timeout(timeout).is_ok()
    }

    /// Test-only no-op trigger. Holds senders whose receivers are dropped
    /// immediately, so [`Self::fire`]/[`Self::flush_blocking`] silently fail.
    /// For tests that construct a `Scheduler` without spawning a real
    /// persistence thread.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn noop() -> Self {
        let (tx, _rx) = channel::bounded(1);
        let (flush_tx, _flush_rx) = channel::bounded(1);
        Self { tx, flush_tx }
    }
}

/// Handle to a running persistence thread. Triggers wake the loop early;
/// [`Self::shutdown`] (or `Drop`) signals the loop, drains a final pass,
/// and joins the thread.
///
/// The `JoinHandle` lives behind `Mutex<Option<…>>` purely because
/// `JoinHandle::join` consumes itself — that lets both [`Self::shutdown`]
/// (via `&self`) and `Drop` reach in and take it. The choice is
/// **internal** to this type; owners hold a plain `PersistenceThread`.
pub struct PersistenceThread {
    handle: Mutex<Option<JoinHandle<()>>>,
    trigger_tx: Sender<()>,
    flush_tx: Sender<Sender<()>>,
    shutdown_tx: Sender<()>,
}

impl PersistenceThread {
    /// Spawn the thread. Takes the workspace [`Conversation`] handle and
    /// the per-layer GPU backings (used by the hot→warm migration and
    /// the warm→cold gather).
    ///
    /// The persistence thread owns a **dedicated CUDA copy stream**
    /// (forked off the device's default CUDA context) and a **pinned
    /// host scratch buffer** that grows on demand across passes. The
    /// hot→warm migration uses both: the gather kernel and the DtoH
    /// run on the copy stream so they overlap with main-stream
    /// compute, and the DtoH target is write-combined pinned memory
    /// (no driver-side pageable bounce).
    pub fn spawn(
        conversation: Conversation,
        backings: Arc<Vec<ChunkedKvBacking>>,
        device: Device,
        compression_policy: Option<CompressionPolicy>,
    ) -> Self {
        let (trigger_tx, trigger_rx) = channel::bounded::<()>(1);
        let (flush_tx, flush_rx) = channel::bounded::<Sender<()>>(1);
        let (shutdown_tx, shutdown_rx) = channel::bounded::<()>(1);

        // All persist-thread CUDA work runs on the device's primary
        // stream — selection, convert, gather, DtoH. Sharing the primary
        // stream with decode serialises persistence behind in-flight
        // inference work but eliminates the cross-stream coordination
        // bugs that arose when persist read shared mutable state
        // (per-head table, head_gids) on a side stream without proper
        // fences.
        //
        // Bound as `copy_stream` for compatibility with the migrate API's
        // parameter name (it was originally a dedicated DMA-only side
        // stream). It is now the primary stream.
        let copy_stream: Arc<CudaStream> = match &device {
            Device::Cuda(d) => d.cuda_stream(),
            _ => panic!("substrate-persistence: requires a CUDA device"),
        };

        let handle = std::thread::Builder::new()
            .name("substrate-persistence".into())
            .spawn(move || {
                run_loop(
                    conversation,
                    backings,
                    device,
                    copy_stream,
                    compression_policy,
                    trigger_rx,
                    flush_rx,
                    shutdown_rx,
                );
            })
            .expect("failed to spawn substrate-persistence thread");

        Self {
            handle: Mutex::new(Some(handle)),
            trigger_tx,
            flush_tx,
            shutdown_tx,
        }
    }

    /// Wake the thread now. No-op if a trigger is already pending or the
    /// thread is mid-pass — the next pass will see the latest state.
    pub fn trigger(&self) {
        let _ = self.trigger_tx.try_send(());
    }

    /// A clone-able trigger handle. Hand these out to anything that
    /// wants to fire the thread without holding the (single-owner)
    /// `PersistenceThread` itself.
    pub fn trigger_handle(&self) -> PersistenceTrigger {
        PersistenceTrigger {
            tx: self.trigger_tx.clone(),
            flush_tx: self.flush_tx.clone(),
        }
    }

    /// Stop the thread: send the shutdown signal, wait for the loop to
    /// drain any pending work, and join. Idempotent — second call is a
    /// no-op (the join handle has already been taken).
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.try_send(());
        if let Some(h) = self.handle.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = h.join();
        }
    }
}

impl Drop for PersistenceThread {
    fn drop(&mut self) {
        // Idempotent — if `shutdown` already joined, `handle.lock()` returns
        // a `None` and this is a no-op.
        self.shutdown();
    }
}

fn run_loop(
    conversation: Conversation,
    backings: Arc<Vec<ChunkedKvBacking>>,
    device: Device,
    copy_stream: Arc<CudaStream>,
    compression_policy: Option<CompressionPolicy>,
    trigger_rx: Receiver<()>,
    flush_rx: Receiver<Sender<()>>,
    shutdown_rx: Receiver<()>,
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

    // Re-used pinned host scratch for the hot→warm DtoH. Eagerly
    // allocated at thread start so the first migration doesn't pay
    // the `cuMemHostAlloc` cost on the hot path; stays allocated
    // across passes and grows in-place if a turn exceeds the
    // preallocation size.
    let mut pinned_scratch: Option<PinnedBuf> =
        preallocate_pinned_scratch(PINNED_PREALLOC_BYTES, "persistence::pinned_scratch");
    loop {
        // Wait for trigger, shutdown, or the periodic tick — whichever
        // fires first. The `default` arm fires after `DEFAULT_TICK` if
        // nothing else has woken the thread.
        let mut shutting_down = false;
        // One-shot ack for a blocking flush request (scheduler under VRAM
        // pressure): reply once this pass has drained the hot→warm migration.
        let mut flush_ack: Option<Sender<()>> = None;
        crossbeam::channel::select! {
            recv(trigger_rx) -> _ => {}
            recv(flush_rx) -> ack => { flush_ack = ack.ok(); }
            recv(shutdown_rx) -> _ => { shutting_down = true; }
            default(DEFAULT_TICK) => {}
        }

        run_pass(
            &conversation,
            &backings,
            &device,
            &copy_stream,
            compression_policy.as_ref(),
            &mut pinned_scratch,
        );

        // run_pass migrates all pending hot→warm before returning, so by here
        // the just-sealed turns hold a warm copy — signal the waiting flush.
        if let Some(ack) = flush_ack {
            let _ = ack.send(());
        }

        if shutting_down {
            // Final group-commit before exiting — any work staged but
            // not yet fsynced should be made durable.
            if let Err(e) = conversation.commit_persistence() {
                tracing::warn!("persist: shutdown commit failed: {e}");
            }
            return;
        }
    }
}

/// Resolve the effective hot→warm quantize policy for a residence group.
///
/// `base` is the engine-wide turn policy. `cc` is the group's
/// per-conversation override (from [`ConvCompression`]): when present it may
/// replace the compression level, drop the global K-format override so K is
/// adaptively quantized like V (`disable_k_override`), and/or pin K and V to
/// fixed uniform formats (`force_k` / `force_v`), bypassing adaptive selection.
/// Returns `None` (no quantization) only when the engine itself has no
/// policy — a per-conversation override never *introduces* quantization on
/// an otherwise-float engine.
fn effective_turn_policy(
    base: Option<&CompressionPolicy>,
    cc: Option<ConvCompression>,
) -> Option<CompressionPolicy> {
    match (base, cc) {
        // Lossless capture: skip the quantize pass so turns keep native R16/F16.
        (_, Some(c)) if c.lossless => None,
        (Some(b), Some(c)) => {
            let mut p = b.clone();
            if let Some(level) = c.level {
                p.compression_level = level;
            }
            if c.disable_k_override {
                p.override_k_quant = None;
            }
            // Forced formats win last: they pin the uniform dst format and
            // re-enable the K override path even when `disable_k_override` set it.
            if let Some(k) = c.force_k {
                p.override_k_quant = Some(k);
            }
            if let Some(v) = c.force_v {
                p.override_v_quant = Some(v);
            }
            Some(p)
        }
        (Some(b), None) => Some(b.clone()),
        (None, _) => None,
    }
}

/// Migrate one policy-group's residences hot→warm: per layer, quantize the
/// group's hot sequences in place (or pass them through when `policy` is
/// `None`) and DtoH-copy the result to a format-preserving CPU warm copy.
/// Successful residences (every layer migrated) are appended to `installs`
/// as `(idx, new_hot, new_warm)`. Queues all work on the primary stream;
/// the caller syncs once after every group.
#[allow(clippy::too_many_arguments)]
fn migrate_group_hot_to_warm(
    backings: &[ChunkedKvBacking],
    device: &Device,
    copy_stream: &Arc<CudaStream>,
    policy: Option<&CompressionPolicy>,
    group: &[(ResidenceIndex, Vec<SealedSequence>)],
    pinned_scratch: &mut Option<PinnedBuf>,
    n_layers: usize,
    installs: &mut Vec<(ResidenceIndex, Vec<SealedSequence>, Vec<SealedSequence>)>,
) {
    if group.is_empty() {
        return;
    }
    // Per-residence (new_hot, new_warm) accumulators, one slot per layer.
    let mut hot_per: Vec<Vec<SealedSequence>> = (0..group.len())
        .map(|_| Vec::with_capacity(n_layers))
        .collect();
    let mut warm_per: Vec<Vec<SealedSequence>> = (0..group.len())
        .map(|_| Vec::with_capacity(n_layers))
        .collect();
    let mut ok = vec![true; group.len()];
    for layer in 0..n_layers {
        let layer_inputs: Vec<&SealedSequence> = group.iter().map(|(_, hot)| &hot[layer]).collect();

        // Step 1: when a policy is configured, run the GPU-only
        // quantize-in-place pass — a fresh `Vec<SealedSequence>` whose full
        // chunks live in new GPU Q arenas (palette-4 per-(h, p) format).
        // Partial trailing chunks are left in their source float format
        // (`quantize_sealed_in_place` skips them): the selection kernel is
        // token-count-blind, and the active writer chunk's unused slots are
        // not guaranteed zero, so quantizing a partial would corrupt its
        // amax. See `compress.rs`.
        let gpu_hot_result: candle::Result<Vec<SealedSequence>> = match policy {
            Some(policy) => candle_nn::kv_cache::quantize_sealed_in_place(
                &backings[layer],
                &layer_inputs,
                policy,
                device,
                copy_stream,
                pinned_scratch,
            ),
            None => Ok(layer_inputs.iter().map(|s| (*s).clone()).collect()),
        };
        let gpu_hot = match gpu_hot_result {
            Ok(v) => v,
            Err(e) => {
                // CUDA errors are async — the error surfaces on the next
                // synchronous call, not at the kernel launch site. Use the
                // thread-local breadcrumb to identify the suspect kernel.
                tracing::warn!(
                    "cache: hot→warm layer {layer} quantize failed: {e} (last CUDA kernel on this thread: {})",
                    candle::last_cuda_kernel_launch()
                );
                ok.fill(false);
                break;
            }
        };

        // Step 2: format-preserving DtoH copy of the (possibly-quantized)
        // GPU sequences to CPU. Simple byte-scatter — no kernel work.
        let gpu_hot_refs: Vec<&SealedSequence> = gpu_hot.iter().collect();
        let cpu_warm_result = backings[layer].migrate_sealed_to_cpu_batch_async(
            device,
            copy_stream,
            pinned_scratch,
            &gpu_hot_refs,
        );
        let cpu_warm = match cpu_warm_result {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    "cache: hot→warm layer {layer} DtoH failed: {e} (last CUDA kernel on this thread: {})",
                    candle::last_cuda_kernel_launch()
                );
                ok.fill(false);
                break;
            }
        };

        if cpu_warm.len() != gpu_hot.len() {
            tracing::warn!(
                "cache: hot→warm layer {layer} mismatch — {} hot vs {} warm",
                gpu_hot.len(),
                cpu_warm.len()
            );
            ok.fill(false);
            break;
        }

        for (i, h) in gpu_hot.into_iter().enumerate() {
            hot_per[i].push(h);
        }
        for (i, w) in cpu_warm.into_iter().enumerate() {
            warm_per[i].push(w);
        }
    }

    for (i, (idx, _)) in group.iter().enumerate() {
        if ok[i] && hot_per[i].len() == n_layers && warm_per[i].len() == n_layers {
            let hot = std::mem::take(&mut hot_per[i]);
            let warm = std::mem::take(&mut warm_per[i]);
            installs.push((*idx, hot, warm));
        }
    }
}

fn run_pass(
    conversation: &Conversation,
    backings: &[ChunkedKvBacking],
    device: &Device,
    copy_stream: &Arc<CudaStream>,
    compression_policy: Option<&CompressionPolicy>,
    pinned_scratch: &mut Option<PinnedBuf>,
) {
    // Cross-thread write→read barrier. This persist pass runs on the background
    // persistence thread and reads each freshly-sealed turn's K/V for the
    // hot→warm migrate below. Those K/V bytes were WRITTEN by the scheduler
    // thread's decode. Both threads queue on the same primary CUDA stream, but
    // the ordering between two host threads issuing onto one stream is not
    // guaranteed — the migrate could begin reading a turn's arena before the
    // decode's writes to it have retired on the GPU, capturing half-written
    // K/V. Synchronize the device once, up front, so every prior decode write
    // is retired before we read any source arena. This is on the background
    // thread, off the decode hot path.
    if let Err(e) = device.synchronize() {
        tracing::warn!(
            target: "candle_conversation::persistence::tier",
            "persist: pre-migrate device sync failed: {e:?}"
        );
    }

    // ── Phase 1: hot → warm ─────────────────────────────────────────────
    //
    // Snapshot the work list under a brief read lock, then run the
    // fully-batched VRAM→RAM migration per layer on the primary CUDA
    // stream with no substrate lock held.
    //
    // For each layer we:
    //  - run `quantize_sealed_in_place` (selection + convert kernels,
    //    primary stream),
    //  - run `migrate_sealed_to_cpu_batch_async` (kv_migrate gather +
    //    DtoH into pinned host scratch, primary stream),
    //  - collect (hot, warm) pairs per residence.
    //
    // All work queues FIFO on the primary stream, so each step sees
    // the previous step's writes without explicit fences. After all
    // layers are migrated we take **one** substrate write lock and
    // install every warm copy at once.
    let pending_warm = conversation.read().snapshot_pending_warm();
    let n_layers = backings.len();
    for (idx, hot, _) in &pending_warm {
        if hot.len() != n_layers {
            tracing::warn!("persist: hot→warm layer-count mismatch for {idx:?} — skipping");
        }
    }

    // Group residences by their per-conversation compression override so a
    // single batched per-layer quantize call covers each policy. Most
    // residences share `None` (the engine-wide turn policy); utility layers
    // such as `code_reading` form their own group at a higher level with the
    // K override dropped (see `ConvCompression`). Every group's kernels
    // queue FIFO on the primary stream, so the one sync after this loop
    // covers all of them.
    let mut groups: HashMap<Option<ConvCompression>, Vec<(ResidenceIndex, Vec<SealedSequence>)>> =
        HashMap::new();
    for (idx, hot, cc) in pending_warm {
        if hot.len() != n_layers {
            continue;
        }
        groups.entry(cc).or_default().push((idx, hot));
    }

    let mut installs: Vec<(ResidenceIndex, Vec<SealedSequence>, Vec<SealedSequence>)> = Vec::new();
    for (cc, group) in groups {
        let effective = effective_turn_policy(compression_policy, cc);
        migrate_group_hot_to_warm(
            backings,
            device,
            copy_stream,
            effective.as_ref(),
            &group,
            pinned_scratch,
            n_layers,
            &mut installs,
        );
    }
    // Primary-stream sync after ALL groups/layers complete.
    //
    // `quantize_sealed_in_place` and the format-preserving DtoH leave
    // work in flight on the primary stream — selection kernel, convert
    // kernel, dst arena allocations, head-gid staging copies, kv_migrate
    // gather, the final DtoH itself. We need one explicit sync before
    // the substrate write — otherwise `install_warm_and_hot` (CPU
    // bookkeeping) can return and the next turn's `apply_projection`
    // can start before the GPU has finished writing the new Q-format
    // arenas the slot now references.
    //
    // Device-wide (not just primary-stream) sync: the reproject on the scheduler
    // thread reads these freshly-installed Q-arenas for the NEXT turn's context,
    // and if any of the convert's V work retires on a stream the primary-stream
    // sync doesn't cover, the reproject captures incomplete V (K, whose convert
    // retires earlier, is fine) — the V-only multi-turn duplication corruption.
    // `device.synchronize()` waits for every stream, closing that window.
    if let Err(e) = device.synchronize() {
        tracing::warn!(
            "cache: device sync after hot→warm batch failed: {e:?} (last CUDA kernel on this thread: {})",
            candle::last_cuda_kernel_launch()
        );
        // The whole batch's GPU work is suspect — don't install any of it.
        installs.clear();
    }
    let mut hot_to_warm_bytes: u64 = 0;
    let mut hot_to_warm_count: usize = 0;
    if !installs.is_empty() {
        let mut view = conversation.write();
        for (idx, hot, warm) in installs {
            let bytes: u64 = warm
                .iter()
                .flat_map(|s| s.chunks.iter())
                .map(|c| c.byte_size)
                .sum();
            tracing::trace!(
                target: "candle_conversation::persistence::tier",
                residence = idx.0,
                bytes,
                "cached hot → warm (with hot Q-format replace)"
            );
            hot_to_warm_bytes = hot_to_warm_bytes.saturating_add(bytes);
            hot_to_warm_count += 1;
            // Atomic dual install: replace residence.hot with the new
            // GPU Q-format sequences (drops the old R16/F16 Arcs from
            // record_turn), and install warm with the CPU copy. The
            // next turn's apply_projection injects residence.hot
            // directly — no warm→hot promotion needed, no kv_migrate
            // scatter into freshly-allocated arenas. Decode reads
            // exactly the bytes the convert kernel wrote.
            view.install_warm_and_hot(idx, hot, warm);
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
    // `snapshot_pending_cold` returns hot bytes (gather operates on
    // GPU sealed); the same payload as warm, only its device backing
    // differs.
    let pending_cold = conversation.read().snapshot_pending_cold();
    let mut cold_installs: Vec<(ResidenceIndex, Vec<StoredSequence>, u64)> =
        Vec::with_capacity(pending_cold.len());
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
        let stored = match conversation.persist_turn_chunks_capture(stream_id, &grid) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("persist: warm→cold write failed for {idx:?}: {e}");
                continue;
            }
        };
        if stored.is_empty() {
            continue;
        }
        // Mark the stream durable through the last-written chunk so
        // the recovery walker knows the turn's chunks landed.
        let total_chunks: usize = stored.iter().map(|s| s.chunks.len()).sum();
        let through = (total_chunks.max(1) - 1) as u64;
        if let Err(e) = conversation.commit_stream_through(stream_id, through) {
            tracing::warn!("persist: commit_stream for {idx:?} failed: {e}");
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
            "persisted warm → cold (redo-log append)"
        );
        cold_installs.push((idx, stored, bytes_for_item));
    }
    let mut warm_to_cold_bytes: u64 = 0;
    let mut warm_to_cold_count: usize = 0;
    if !cold_installs.is_empty() {
        let mut view = conversation.write();
        for (idx, stored, bytes) in cold_installs {
            warm_to_cold_bytes = warm_to_cold_bytes.saturating_add(bytes);
            warm_to_cold_count += 1;
            view.install_cold(idx, stored);
        }
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
    let pending_section_cold = conversation.read().snapshot_pending_section_cold();
    let mut section_to_cold_bytes: u64 = 0;
    let mut section_to_cold_count: usize = 0;
    if !pending_section_cold.is_empty() {
        let mut section_cold_installs: Vec<(ResidenceIndex, Vec<StoredSequence>, u64)> =
            Vec::with_capacity(pending_section_cold.len());
        for (idx, stream_id, hot) in pending_section_cold {
            if hot.len() != n_layers {
                tracing::warn!(
                    "persist: section {idx:?} hot has {} layers, expected {n_layers} — skipping",
                    hot.len()
                );
                continue;
            }
            let grid = match build_grid(&hot, backings, device) {
                Ok(g) => g,
                Err(e) => {
                    tracing::warn!("persist: section gather failed for {idx:?}: {e}");
                    continue;
                }
            };
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
            section_cold_installs.push((idx, stored, bytes_for_item));
        }
        if !section_cold_installs.is_empty() {
            let mut view = conversation.write();
            for (idx, stored, bytes) in section_cold_installs {
                section_to_cold_bytes = section_to_cold_bytes.saturating_add(bytes);
                section_to_cold_count += 1;
                view.install_cold(idx, stored);
            }
        }
    }

    // ── Phase 3: fsync ─────────────────────────────────────────────────
    //
    // Group-commit anything the previous phases staged. No-op when both
    // phases skipped — e.g. an idle workspace just ticked and found
    // nothing to do.
    if let Err(e) = conversation.commit_persistence_if_pending() {
        tracing::warn!("persist: fsync failed: {e}");
    }

    // ── Phase 4: compaction ────────────────────────────────────────────
    //
    // Reclaim the log's dead weight once it crosses the threshold. The
    // check is pure in-RAM arithmetic (the O(1) dead-byte counter plus
    // the tombstoned-stream sum), so polling it every pass is free; the
    // rewrite itself holds the persistence + substrate locks for its
    // duration, which is why it lives here on the background thread
    // rather than anywhere near the decode path.
    match conversation.compact_persistence_if_needed() {
        Ok(true) => {
            tracing::info!(
                target: "candle_conversation::persistence::tier",
                "persist: redo log compacted (dead-byte threshold crossed)"
            );
        }
        Ok(false) => {}
        Err(e) => tracing::warn!("persist: auto-compaction failed: {e}"),
    }

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
fn build_grid(
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
