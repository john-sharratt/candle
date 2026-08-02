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

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use candle::cuda_backend::cudarc::driver::CudaStream;
use candle::quantized::pinned_staging::PinnedBuf;
use candle::Device;
use candle_nn::kv_cache::{ChunkedKvBacking, CompressionPolicy, SealedSequence};
use crossbeam::channel::{self, Receiver, Sender};

use super::cold_load::preallocate_pinned_scratch;
use super::resume::TurnChunkGrid;
use super::transfer::seal_to_chunk_images;
use crate::projection::Conversation;
use crate::scheduler::relief_trace;
use crate::substrate::{ConvCompression, ResidenceIndex, StoredSequence};
use std::collections::HashMap;
use sysinfo::System;

/// How often the loop wakes up on its own when no triggers arrive.
pub const DEFAULT_TICK: Duration = Duration::from_secs(5);

/// Minimum wall-clock between background **segment-maintenance** scans. The
/// tiering phases run every pass (one per trigger / tick), but the maintenance
/// candidate scan is O(live segments' records), so it's decoupled from the
/// trigger rate and run at most this often — reclaim is a slow background job,
/// not something that needs to react to every turn-seal.
pub const MAINTENANCE_INTERVAL: Duration = Duration::from_secs(15);

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
    /// Live hot→warm drain backlog in bytes, stamped by the persistence loop
    /// each pass (pre- and post-drain). The scheduler polls this to throttle
    /// ingest admission off a leading signal. See [`Substrate::pending_warm_bytes`].
    pending_warm_bytes: Arc<AtomicU64>,
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

    /// The persistence thread's most recent hot→warm drain backlog, in bytes.
    /// A *leading* pressure signal (drain deficit) the scheduler reads every
    /// wave to size the ingest admission window — it rises as seals outrun the
    /// drain and falls as the drain catches up, before the lagging VRAM-pressure
    /// trip ever fires.
    pub fn pending_warm_bytes(&self) -> u64 {
        self.pending_warm_bytes.load(Ordering::Relaxed)
    }

    /// Test-only no-op trigger. Holds senders whose receivers are dropped
    /// immediately, so [`Self::fire`]/[`Self::flush_blocking`] silently fail.
    /// For tests that construct a `Scheduler` without spawning a real
    /// persistence thread.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn noop() -> Self {
        let (tx, _rx) = channel::bounded(1);
        let (flush_tx, _flush_rx) = channel::bounded(1);
        Self {
            tx,
            flush_tx,
            pending_warm_bytes: Arc::new(AtomicU64::new(0)),
        }
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
    /// Shared with the loop (which stamps it) and every [`PersistenceTrigger`]
    /// handed out (which read it) — the live hot→warm drain backlog in bytes.
    pending_warm_bytes: Arc<AtomicU64>,
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
        let pending_warm_bytes = Arc::new(AtomicU64::new(0));
        let loop_backlog = pending_warm_bytes.clone();

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
                    loop_backlog,
                );
            })
            .expect("failed to spawn substrate-persistence thread");

        Self {
            handle: Mutex::new(Some(handle)),
            trigger_tx,
            flush_tx,
            shutdown_tx,
            pending_warm_bytes,
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
            pending_warm_bytes: self.pending_warm_bytes.clone(),
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
    pending_warm_bytes: Arc<AtomicU64>,
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

        // Stamp the backlog this pass is about to face (pre-drain). Held high
        // for the whole pass so the scheduler keeps ingest throttled while a
        // long drain is in flight — the conservative direction.
        pending_warm_bytes.store(conversation.read().pending_warm_bytes(), Ordering::Relaxed);

        let run_maintenance = last_maintenance.elapsed() >= MAINTENANCE_INTERVAL;
        if run_maintenance {
            last_maintenance = Instant::now();
        }
        run_pass(
            &conversation,
            &backings,
            &device,
            &copy_stream,
            compression_policy.as_ref(),
            &mut pinned_scratch,
            run_maintenance,
        );

        // Re-stamp the residual backlog (post-drain) so the signal decays as
        // soon as the pass installs warm, letting admission reopen promptly.
        pending_warm_bytes.store(conversation.read().pending_warm_bytes(), Ordering::Relaxed);

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
pub(crate) fn effective_turn_policy(
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

/// Ask the VRAM governor to escalate recovery after a compress-to-free pass
/// failed to allocate. Only fires for VRAM-exhaustion errors ([`is_device_oom`])
/// — the compression couldn't fit even a small transient quant arena, the
/// strongest "critically full" signal — where the scheduler making room lets the
/// (untouched, still-consistent) turn's next compression attempt succeed. Other
/// error kinds won't be helped by evicting, so they don't signal.
///
/// [`is_device_oom`]: candle_nn::kv_cache::is_device_oom
fn signal_vram_starvation(device: &Device, err: &candle::Error) {
    if !candle_nn::kv_cache::is_device_oom(err) {
        return;
    }
    if let candle::DeviceLocation::Cuda { gpu_id } = device.location() {
        if let Some(gov) = candle::vram::get(gpu_id) {
            gov.signal_starvation();
        }
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
    quantize_ms: &mut u64,
    copy_ms: &mut u64,
    select_ms: &mut u64,
    alloc_ms: &mut u64,
    convert_ms: &mut u64,
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
    // K/V heads per layer (uniform across layers of one model) — the batched
    // convert reads the accumulated descriptors as `[jobs × n_kv_head]`.
    let n_kv_head = backings.first().map(|b| b.n_kv_head()).unwrap_or(0);
    // Per-(chunk, head) convert descriptors accumulated across EVERY layer, so a
    // single batched convert launch replaces the 48 per-layer launches.
    let mut all_descs: Vec<candle::quantized::cuda::PalHeadDesc> = Vec::new();
    // Each layer's quantized (but not-yet-converted) hot sequences, in layer
    // order, one Vec-per-residence entry each.
    let mut gpu_hot_per_layer: Vec<Vec<SealedSequence>> = Vec::with_capacity(n_layers);

    let success = 'work: {
        // ── Phase 1: quantize EVERY layer with ONE cross-layer selection,
        //    deferring the convert ─────────────────────────────────────────
        // `quantize_layers_deferred` runs the palette-4 format selection as a
        // SINGLE fused kernel + readback across all layers (not 48× per layer —
        // the term that dominated the drain), allocates each layer's dst GPU-Q
        // arenas, and appends every layer's per-(chunk, head) descriptors to
        // `all_descs` WITHOUT launching the convert. Returned sequences already
        // point at the (soon-to-be-filled) dst arenas. Partial trailing chunks
        // quantize too (dead slots are zero; the packed valid-range window keeps
        // amax correct); already-quant chunks pass through the preserve bucket.
        // Bit-identical to the per-layer selection — see the compress-test A/B.
        let t_q = std::time::Instant::now();
        match policy {
            Some(policy) => {
                let per_layer_inputs: Vec<Vec<&SealedSequence>> = (0..n_layers)
                    .map(|layer| group.iter().map(|(_, hot)| &hot[layer]).collect())
                    .collect();
                let backing_refs: Vec<&ChunkedKvBacking> = backings.iter().collect();
                match candle_nn::kv_cache::quantize_layers_deferred(
                    &backing_refs,
                    &per_layer_inputs,
                    policy,
                    device,
                    copy_stream,
                    pinned_scratch,
                    &mut all_descs,
                    select_ms,
                    alloc_ms,
                ) {
                    Ok(v) => gpu_hot_per_layer = v,
                    Err(e) => {
                        // CUDA errors are async — surfaced on the next sync call,
                        // not at the launch site; use the breadcrumb to identify it.
                        tracing::warn!(
                            "cache: hot→warm cross-layer quantize failed: {e} (last CUDA kernel on this thread: {})",
                            candle::last_cuda_kernel_launch()
                        );
                        // Non-destructive: the group stays hot-float + consistent
                        // and retries next pass. If it was VRAM exhaustion, the
                        // retry needs room — signal the scheduler to escalate.
                        signal_vram_starvation(device, &e);
                        *quantize_ms += t_q.elapsed().as_millis() as u64;
                        break 'work false;
                    }
                }
            }
            None => {
                // No policy: pass each layer's hot sequences through unchanged.
                for layer in 0..n_layers {
                    gpu_hot_per_layer
                        .push(group.iter().map(|(_, hot)| hot[layer].clone()).collect());
                }
            }
        }
        *quantize_ms += t_q.elapsed().as_millis() as u64;

        // ── Phase 2: ONE batched convert across every layer's descriptors ───
        // `n_layers × n_chunks × n_kv_head` blocks in a single launch (tiled at
        // the grid.y cap), filling all dst arenas before the migrate DtoH reads
        // them (same primary stream, FIFO). No-op when `policy` is None
        // (`all_descs` empty). Counts toward `quantize_ms`. Bit-identical to the
        // per-layer convert — proven by `quantize_layers_batched_matches_per_layer`.
        if !all_descs.is_empty() {
            let t_conv = std::time::Instant::now();
            let conv = candle_nn::kv_cache::convert_deferred_descs(
                &backings[0],
                &all_descs,
                n_kv_head,
                device,
            );
            let dt = t_conv.elapsed().as_millis() as u64;
            *quantize_ms += dt;
            *convert_ms += dt;
            if let Err(e) = conv {
                tracing::warn!(
                    "cache: hot→warm batched convert failed: {e} (last CUDA kernel on this thread: {})",
                    candle::last_cuda_kernel_launch()
                );
                signal_vram_starvation(device, &e);
                break 'work false;
            }
        }

        // ── Phase 3: ONE batched hot → warm DtoH across every layer ─────────
        // A single gather + DtoH + sync for all layers (vs 48 per-layer gathers +
        // syncs), which was the dominant `copy_ms` term. Bit-identical to the
        // per-layer migrate — proven by the compress-test A/B.
        let t_c = std::time::Instant::now();
        let per_layer_refs: Vec<Vec<&SealedSequence>> = gpu_hot_per_layer
            .iter()
            .map(|v| v.iter().collect())
            .collect();
        let backing_refs: Vec<&ChunkedKvBacking> = backings.iter().collect();
        let warm_result = ChunkedKvBacking::migrate_sealed_layers_to_cpu_batch(
            &backing_refs,
            device,
            copy_stream,
            pinned_scratch,
            &per_layer_refs,
        );
        *copy_ms += t_c.elapsed().as_millis() as u64;
        drop(per_layer_refs); // release the borrow of gpu_hot_per_layer
        let warm_per_layer = match warm_result {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    "cache: hot→warm batched DtoH failed: {e} (last CUDA kernel on this thread: {})",
                    candle::last_cuda_kernel_launch()
                );
                break 'work false;
            }
        };
        // Distribute warm (per layer, in layer order) into per-residence order.
        for (layer, layer_warm) in warm_per_layer.into_iter().enumerate() {
            if layer_warm.len() != gpu_hot_per_layer[layer].len() {
                tracing::warn!(
                    "cache: hot→warm layer {layer} mismatch — {} hot vs {} warm",
                    gpu_hot_per_layer[layer].len(),
                    layer_warm.len()
                );
                break 'work false;
            }
            for (i, w) in layer_warm.into_iter().enumerate() {
                warm_per[i].push(w);
            }
        }

        // Distribute the per-layer hot sequences into per-residence order.
        for layer_hot in gpu_hot_per_layer.into_iter() {
            for (i, h) in layer_hot.into_iter().enumerate() {
                hot_per[i].push(h);
            }
        }
        true
    };

    if !success {
        ok.fill(false);
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
    run_maintenance: bool,
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
    let t_pass = std::time::Instant::now();
    if let Err(e) = device.synchronize() {
        tracing::warn!(
            target: "candle_conversation::persistence::tier",
            "persist: pre-migrate device sync failed: {e:?}"
        );
    }
    // Phase-timing breakdown so we can see *where* the hot→warm drain spends its
    // time (the demote starves when this can't keep up with the ingest seal rate).
    let sync_pre_ms = t_pass.elapsed().as_millis() as u64;
    let t_migrate = std::time::Instant::now();

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
    let pending_warm: Vec<_> = conversation.read().snapshot_pending_warm();
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
    let mut quantize_ms = 0u64;
    let mut copy_ms = 0u64;
    let mut select_ms = 0u64;
    let mut alloc_ms = 0u64;
    let mut convert_ms = 0u64;
    // Hold the migration-in-flight guard across the whole hot→warm GPU window —
    // from before `per_head_table_host` captures arena base pointers (inside
    // `migrate_group_hot_to_warm`) through the post-migrate `device.synchronize()`
    // below. While held, the scheduler defers arena free / defrag / trim
    // (`BatchedInferenceSession::{release_empty_arenas,compact,defragment_bounded,
    // trim_kv_pool}`), so the `run_select_kv_format_palette4_paged` kernel can't
    // read a base pointer that a concurrent `cuMemPoolTrimTo` has unmapped.
    // Dropped right after the sync (below), before install — by then the kernel
    // has retired, so the install's own arena frees are safe.
    let migrate_guard = candle_nn::kv_cache::enter_migrate();
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
            &mut quantize_ms,
            &mut copy_ms,
            &mut select_ms,
            &mut alloc_ms,
            &mut convert_ms,
        );
    }
    let migrate_ms = t_migrate.elapsed().as_millis() as u64;
    let t_sync_post = std::time::Instant::now();
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
    let sync_post_ms = t_sync_post.elapsed().as_millis() as u64;
    // Migrate GPU work has retired (the sync above) — release the guard so the
    // scheduler's VRAM relief resumes. The install below only drops old hot Arcs
    // (its own arena frees), which no in-flight kernel is reading.
    drop(migrate_guard);
    let t_install = std::time::Instant::now();
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
            if view.residence_evict_when_cold(idx) {
                // Completed-ingest / collection-member residence flagged for full
                // eviction: don't keep the fresh GPU Q copy resident — install
                // warm-only and drop hot now, so the VRAM returns immediately.
                // The warm→cold write below (reading the CPU warm copy) then
                // frees warm too via `install_cold`, leaving it cold-only. The
                // just-produced `hot` Q sequences drop here → arena chunks return
                // to the pool.
                view.install_warm_and_evict_hot(idx, warm);
            } else {
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
    }
    let install_ms = t_install.elapsed().as_millis() as u64;
    // Log the breakdown when the pass did real work OR was slow even with nothing
    // to install (isolates a dominant `device.synchronize()` cost). This is the
    // number that tells us whether the drain is sync-bound, the serial 48-layer
    // migrate, or install-lock-bound.
    let total_ms = t_pass.elapsed().as_millis() as u64;
    if hot_to_warm_count > 0 || sync_pre_ms + migrate_ms + sync_post_ms > 50 {
        tracing::debug!(
            target: "candle_conversation::persistence::tier",
            residences = hot_to_warm_count,
            mib = hot_to_warm_bytes / (1 << 20),
            n_layers,
            sync_pre_ms,
            migrate_ms,
            quantize_ms,
            select_ms,
            alloc_ms,
            convert_ms,
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
    // Feed the instrumented migration panel straight from the pass (no log tail).
    if hot_to_warm_count > 0 {
        crate::scheduler::phase_ring::push_migrate(crate::scheduler::phase_ring::migrate_sample(
            hot_to_warm_count as u64,
            hot_to_warm_bytes / (1 << 20),
            migrate_ms,
            quantize_ms,
            copy_ms,
            total_ms,
        ));
    }
    // Backlog split: how much of the hot-without-warm set was DEFERRED because it
    // is the in-flight decode's pinned working set (the tier-livelock fix) vs. a
    // genuine drain the migrate must still catch up on. Fires only when something
    // was pin-deferred, so a healthy idle pass stays quiet.
    {
        let (dc, db, pc, pb) = conversation.read().warm_backlog_split();
        if pc > 0 {
            tracing::debug!(
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
    // `snapshot_pending_cold` returns hot bytes (gather operates on
    // GPU sealed); the same payload as warm, only its device backing
    // differs.
    let pending_cold = conversation.read().snapshot_pending_cold();
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

    // ── Phase 2.6: warm-tier RAM purge ─────────────────────────────────
    //
    // The warm (RAM) copies produced by Phase 1 are durable the moment
    // their cold copy lands (Phase 2 / 2.5 above), but nothing else drops
    // them: `install_cold` frees hot, never warm, and the only other
    // `purge_warm_to_target` call site is the cold→hot recall path in
    // `elevate_to_hot`. During a bulk ingest (calibration, repo scan)
    // there are almost no recalls — just hot→warm→cold migration — so the
    // warm tier would otherwise grow unbounded and exhaust host RAM (the
    // OOM that killed a full load). Run the same LRU purge here, where warm
    // is actually produced, dropping the least-recently-used warm copies
    // until the OS holds at least `max(2 GiB, 5% × total)` free. Every warm
    // residence reachable here is already cold-backed (Phase 2 persisted
    // this pass's whole warm-without-cold set), so no un-persisted bytes
    // are dropped. `incoming = 0`: we bound the existing footprint, not
    // reserve for an upcoming allocation.
    //
    // Gated on this pass having actually moved KV into warm — warm only
    // grows via the hot→warm migration above, and a pass that produced
    // none can't have pushed RAM up, so an idle tick skips the sysinfo
    // query entirely.
    if hot_to_warm_count > 0 || warm_to_cold_count > 0 {
        let mut sys = System::new();
        sys.refresh_memory();
        let total_ram = sys.total_memory();
        let available_ram = sys.available_memory();
        // `purge_warm_to_target` logs its own batch summary (see substrate.rs);
        // no extra log here.
        let _ = conversation
            .write()
            .purge_warm_to_target(0, available_ram, total_ram);
    }

    // ── Phase 3: fsync ─────────────────────────────────────────────────
    //
    // Group-commit anything the previous phases staged. No-op when both
    // phases skipped — e.g. an idle workspace just ticked and found
    // nothing to do.
    if let Err(e) = conversation.commit_persistence_if_pending() {
        tracing::warn!("persist: fsync failed: {e}");
    }

    // ── Phase 4: segment maintenance ───────────────────────────────────
    //
    // Reclaim dead weight incrementally — one segment drop / compact / combine
    // per invocation (`persistence/maintenance.rs`). Rate-limited to
    // `MAINTENANCE_INTERVAL` (the candidate scan is O(live records), so it's
    // decoupled from the trigger rate). The relocation I/O runs under the
    // persistence lock only (phased locking), never the substrate write lock,
    // so it can't stall decode. The specific op is logged at DEBUG by
    // `finish_maintenance`.
    match if run_maintenance {
        conversation.compact_persistence_if_needed()
    } else {
        Ok(false)
    } {
        Ok(true) => {
            relief_trace::note("tier", "segment_reclaim", 0, 0);
            tracing::info!(
                target: "candle_conversation::persistence::tier",
                "persist: substrate maintenance op applied (segment reclaim)"
            );
        }
        Ok(false) => {}
        Err(e) => tracing::warn!("persist: substrate maintenance failed: {e}"),
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
