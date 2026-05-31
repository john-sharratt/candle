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
use crate::substrate::ResidenceIndex;

/// How often the loop wakes up on its own when no triggers arrive.
pub const DEFAULT_TICK: Duration = Duration::from_secs(5);

/// Clone-able fire-and-forget trigger for the persistence thread. Held
/// by anything that wants to wake the loop early (e.g. the scheduler
/// signalling "turn just sealed"). Cheap to clone — wraps a
/// `crossbeam::channel::Sender<()>`.
#[derive(Clone)]
pub struct PersistenceTrigger {
    tx: Sender<()>,
}

impl PersistenceTrigger {
    /// Wake the persistence thread now. No-op when the trigger queue is
    /// already full (one wake is as good as several).
    pub fn fire(&self) {
        let _ = self.tx.try_send(());
    }

    /// Test-only no-op trigger. Holds a sender whose receiver is
    /// dropped immediately, so [`Self::fire`] silently fails. For
    /// tests that construct a `Scheduler` without spawning a real
    /// persistence thread.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn noop() -> Self {
        let (tx, _rx) = channel::bounded(1);
        Self { tx }
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
                    shutdown_rx,
                );
            })
            .expect("failed to spawn substrate-persistence thread");

        Self {
            handle: Mutex::new(Some(handle)),
            trigger_tx,
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
        crossbeam::channel::select! {
            recv(trigger_rx) -> _ => {}
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

fn run_pass(
    conversation: &Conversation,
    backings: &[ChunkedKvBacking],
    device: &Device,
    copy_stream: &Arc<CudaStream>,
    compression_policy: Option<&CompressionPolicy>,
    pinned_scratch: &mut Option<PinnedBuf>,
) {
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
    let layer_mismatched: Vec<ResidenceIndex> = pending_warm
        .iter()
        .filter(|(_, hot)| hot.len() != n_layers)
        .map(|(idx, _)| *idx)
        .collect();
    for idx in &layer_mismatched {
        tracing::warn!("persist: hot→warm layer-count mismatch for {idx:?} — skipping");
    }
    let pending_warm: Vec<_> = pending_warm
        .into_iter()
        .filter(|(_, hot)| hot.len() == n_layers)
        .collect();

    // Per-layer batched migrate. Each call submits one kernel + one
    // DtoH on the primary stream; layers queue sequentially on the
    // primary stream. PCIe is the bottleneck, not stream parallelism.
    // Per-residence (new_hot, new_warm) pairs. `new_hot` is the
    // policy-quantized GPU sequence (or the original R16/F16 sequence
    // if no policy). `new_warm` is the format-preserving CPU copy.
    let mut hot_per_residence: Vec<Vec<SealedSequence>> = (0..pending_warm.len())
        .map(|_| Vec::with_capacity(n_layers))
        .collect();
    let mut warm_per_residence: Vec<Vec<SealedSequence>> = (0..pending_warm.len())
        .map(|_| Vec::with_capacity(n_layers))
        .collect();
    let mut batch_ok = vec![true; pending_warm.len()];
    for layer in 0..n_layers {
        let layer_inputs: Vec<&SealedSequence> =
            pending_warm.iter().map(|(_, hot)| &hot[layer]).collect();

        // Step 1: when a `CompressionPolicy` is configured, run the
        // GPU-only quantize-in-place pass. Result: a fresh
        // `Vec<SealedSequence>` whose chunks live in new GPU Q arenas
        // (palette-4 per-(h, p) format). Partial chunks are quantized
        // alongside full chunks; the recycle hook on `ChunkGid` Drop
        // guarantees positions [token_count..32] are zero so the
        // convert kernel never quantizes random noise.
        let gpu_hot_result: candle::Result<Vec<SealedSequence>> = match compression_policy {
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
                // thread-local breadcrumb to identify which kernel was the
                // most recent launch on the persist thread (i.e. the
                // suspect).
                tracing::warn!(
                    "cache: hot→warm layer {layer} quantize failed: {e} (last CUDA kernel on this thread: {})",
                    candle::last_cuda_kernel_launch()
                );
                for ok in &mut batch_ok {
                    *ok = false;
                }
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
                for ok in &mut batch_ok {
                    *ok = false;
                }
                break;
            }
        };

        if cpu_warm.len() != gpu_hot.len() {
            tracing::warn!(
                "cache: hot→warm layer {layer} mismatch — {} hot vs {} warm",
                gpu_hot.len(),
                cpu_warm.len()
            );
            for ok in &mut batch_ok {
                *ok = false;
            }
            break;
        }

        for (i, h) in gpu_hot.into_iter().enumerate() {
            hot_per_residence[i].push(h);
        }
        for (i, w) in cpu_warm.into_iter().enumerate() {
            warm_per_residence[i].push(w);
        }
    }

    // Primary-stream sync after ALL layers complete.
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
    // Hoisted out of the per-layer loop because primary-stream
    // operations are FIFO: each layer's work queues behind the
    // previous layer's and executes in order on the GPU. A single
    // sync at the end covers everything.
    if let Device::Cuda(cuda_dev) = device {
        if let Err(e) = cuda_dev.cuda_stream().synchronize() {
            tracing::warn!(
                "cache: primary-stream sync after hot→warm batch failed: {e:?} (last CUDA kernel on this thread: {})",
                candle::last_cuda_kernel_launch()
            );
            for ok in &mut batch_ok {
                *ok = false;
            }
        }
    }

    let mut installs: Vec<(ResidenceIndex, Vec<SealedSequence>, Vec<SealedSequence>)> = Vec::new();
    let mut hot_to_warm_bytes: u64 = 0;
    let mut hot_to_warm_count: usize = 0;
    for (i, (idx, _)) in pending_warm.into_iter().enumerate() {
        if batch_ok[i]
            && hot_per_residence[i].len() == n_layers
            && warm_per_residence[i].len() == n_layers
        {
            let hot = std::mem::take(&mut hot_per_residence[i]);
            let warm = std::mem::take(&mut warm_per_residence[i]);
            let bytes: u64 = warm
                .iter()
                .flat_map(|s| s.chunks.iter())
                .map(|c| c.byte_size as u64)
                .sum();
            tracing::debug!(
                target: "candle_conversation::persistence::tier",
                residence = idx.0,
                bytes,
                "cached hot → warm (with hot Q-format replace)"
            );
            hot_to_warm_bytes = hot_to_warm_bytes.saturating_add(bytes);
            hot_to_warm_count += 1;
            installs.push((idx, hot, warm));
        }
    }
    if !installs.is_empty() {
        let mut view = conversation.write();
        for (idx, hot, warm) in installs {
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
    let mut cold_installs: Vec<(ResidenceIndex, Vec<crate::substrate::StoredSequence>, u64)> =
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
        tracing::debug!(
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

    // ── Phase 3: fsync ─────────────────────────────────────────────────
    //
    // Group-commit anything the previous phases staged. No-op when both
    // phases skipped — e.g. an idle workspace just ticked and found
    // nothing to do.
    if let Err(e) = conversation.commit_persistence_if_pending() {
        tracing::warn!("persist: fsync failed: {e}");
    }

    // Per-pass aggregate. Only logged when something actually moved.
    if hot_to_warm_count > 0 || warm_to_cold_count > 0 {
        tracing::info!(
            target: "candle_conversation::persistence::tier",
            hot_to_warm_count,
            hot_to_warm_bytes,
            warm_to_cold_count,
            warm_to_cold_bytes,
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
