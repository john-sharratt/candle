//! 1 MiB-unit pipelined cold-load.
//!
//! Drives one [`ChunkBatch`] through three concurrent actors so that
//! disk reads, host-to-device transfers, and the `kv_migrate` scatter
//! all run in parallel at the granularity of a 1 MiB sub-stripe
//! (= [`super::chunk_plan::UNIT_BYTES`]):
//!
//! - **Reader pool** (16 OS threads, each pinned to one DirectFile
//!   handle) — pulls [`ReadWork`] items off a shared queue and does a
//!   positioned direct-I/O read into the pinned scratch.
//! - **Allocator** (1 thread) — receives unit-completion events,
//!   decodes records whose bytes are now fully landed, batches their
//!   `BlockAllocSpec`s per layer for `alloc_sealed_blocks_bulk` and
//!   resolves their dest pointers via `resolve_block_ptrs_from_hgids`.
//! - **GPU dispatcher** (main thread, owns the CUDA context) — receives
//!   per-unit [`UnitWork`] messages, queues `memcpy_htod` for the
//!   unit on the CUDA stream, accumulates the migrate plan, and fires
//!   one `kv_migrate` at the end.
//!
//! ## Anchoring
//!
//! Each record has a `[first_unit, last_unit]` span pre-computed by
//! [`super::chunk_plan::plan_units`]. A record's CRC verify needs
//! every byte landed, so the allocator processes a record only when
//! **the last of its units** has been read. The metadata bytes are
//! always in `first_unit`, so by the time `last_unit` lands every
//! unit in `[first_unit, last_unit]` has also been read (unit
//! completions can arrive out of order, so the allocator decrements
//! per-record wait counts to find which records just unblocked).
//!
//! ## CUDA-stream ordering
//!
//! The dispatcher queues all per-unit `memcpy_htod`s on the device's
//! main inference stream in the order it receives `UnitWork`. The
//! single `kv_migrate` at the end therefore sees all bytes in
//! staging by stream-ordering: every HtoD that the migrate sources
//! from was queued earlier on the same stream.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use candle::cuda_backend::cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use candle::cuda_backend::WrapErr;
use candle::Device;
use candle_nn::kv_cache::{kv_migrate, BlockAllocSpec, ChunkedKvBacking, KvFormat, MigrationPlan};
use crossbeam::channel::{unbounded, Receiver, Sender};

use super::chunk_plan::{ChunkBatch, SourceLog, UnitPlan, UNIT_BYTES};
use super::cold_load::ColdLoadStager;
use super::direct_io::DirectFile;
use super::record::{decode_record, ChunkPayload};
use super::segment::SegmentId;
use super::SubstratePersistence;

/// Cross-thread raw pointer to the pinned scratch's base. Bundled into
/// a `Send + Sync` wrapper because the bare `*mut u8` would block the
/// pipeline's spawn'd closures from capturing it. Safety: the pipeline
/// protocol guarantees no two threads write the same byte range
/// concurrently — readers write disjoint unit regions, the allocator
/// only reads, the dispatcher only reads (and the device-side DMA is
/// queued by then).
///
/// All access to the underlying pointer goes through methods that take
/// `self` (by value) — accessing the raw `.0` field directly inside a
/// `move ||` closure triggers Rust 2021 disjoint-capture, which would
/// re-capture just the `*mut u8` field (bypassing the `Send` impl).
#[derive(Copy, Clone)]
struct PinnedPtr(*mut u8);
unsafe impl Send for PinnedPtr {}
unsafe impl Sync for PinnedPtr {}

impl PinnedPtr {
    /// Construct a `&mut [u8]` slice at `[offset, offset + length)`.
    /// Caller is responsible for non-overlapping access across threads.
    unsafe fn slice_mut<'a>(self, offset: usize, length: usize) -> &'a mut [u8] {
        std::slice::from_raw_parts_mut(self.0.add(offset), length)
    }

    /// Construct an immutable `&[u8]` slice at `[offset, offset + length)`.
    unsafe fn slice<'a>(self, offset: usize, length: usize) -> &'a [u8] {
        std::slice::from_raw_parts(self.0.add(offset), length)
    }
}

/// Work item dispatched to a reader thread.
struct ReadWork {
    unit_idx: u32,
    file_offset: u64,
    dest_offset: usize,
    length: usize,
    source: SourceLog,
}

/// Reader → allocator notification carrying the unit index and any I/O
/// error that occurred (so the allocator can propagate it).
struct UnitDone {
    unit_idx: u32,
    error: Option<candle::Error>,
}

/// Allocator → dispatcher message: a unit's bytes have landed and its
/// anchored records have been alloc'd.
struct UnitWork {
    unit_idx: u32,
    /// Records anchored at this unit (i.e. `last_unit == unit_idx`)
    /// — they're ready to migrate now.
    records: Vec<RecordMigrate>,
}

/// Sub-timer breakdown of the allocator thread's CPU time. Sum of
/// `decode_us + bulk_alloc_us + resolve_ptrs_us` should approximate
/// `alloc_ms * 1000`; the remainder is the per-unit accounting overhead
/// (per-layer Vec/clone construction, channel sends, etc.).
///
/// `bulk_alloc_us` is further split into:
///   - `pool_us`     – Phase 1: batched per-key gid alloc (no lock).
///   - `register_us` – Phase 2: per-spec slot mutation (state.write held).
///   - `gpu_push_us` – Phase 3: `resolve_arena_info` + per-layer
///     `update_gpu_chunks_bulk` (still inside state.write; the bulk
///     emits coalesced `memcpy_htod` runs onto the CUDA stream).
/// Sum of the three sub-sub-timers ≤ `bulk_alloc_us` (channel/hash
/// overhead in between accounts for the gap).
#[derive(Clone, Copy, Debug, Default)]
struct AllocBreakdown {
    alloc_ms: u64,
    decode_us: u64,
    bulk_alloc_us: u64,
    resolve_ptrs_us: u64,
    n_bulk_calls: u32,
    n_resolve_calls: u32,
    pool_us: u64,
    register_us: u64,
    gpu_push_us: u64,
}

/// One alloc'd record ready for the migrate plan.
struct RecordMigrate {
    /// Byte offset of the record's `kv_bytes` in the pinned scratch /
    /// GPU staging (same layout).
    src_offset: i64,
    /// Destination per-sub-band `(device_ptr, byte_len)` pairs in the
    /// freshly-allocated arena chunks.
    dst_ptrs: Vec<(i64, i64)>,
}

/// Per-stage wall-clock breakdown produced by [`run_pipeline`].
///
/// All times are in milliseconds, measured from the pipeline's start
/// instant. The metrics are designed for separate-axis profiling —
/// they are *not* additive, the stages overlap by design.
#[derive(Clone, Copy, Debug, Default)]
pub struct PipelineStats {
    /// Wall-clock from pipeline start to when **the last disk read
    /// finished** (i.e. when the slowest reader returned from
    /// `pread`/`ReadFile`). Captured on the reader side, so it's
    /// unaffected by allocator latency or channel buffering.
    /// Reader-bound floor: this is the I/O time.
    pub reads_ms: u64,
    /// Sum of CPU time spent inside the allocator thread doing
    /// per-unit decode + `alloc_sealed_blocks_bulk` +
    /// `resolve_block_ptrs_from_hgids`. Independent of when the
    /// dispatcher consumes the resulting `UnitWork`s. This is
    /// effectively the allocator's wall-clock since it works
    /// sequentially.
    pub alloc_ms: u64,
    /// Sum of time spent queuing `memcpy_htod` calls on the CUDA
    /// stream (each queue call returns ~immediately; this is just
    /// overhead bookkeeping).
    pub htod_ms: u64,
    /// Time the dispatcher spent in `kv_migrate` — which calls
    /// `stream.synchronize()` internally, so this captures the
    /// runtime of any GPU op still in flight at that point (the
    /// migrate kernel itself, plus whatever HtoDs haven't finished).
    pub migrate_ms: u64,
    /// Wall-clock for the whole pipeline call.
    pub total_ms: u64,
    /// Number of 1 MiB units in the chunk batch.
    pub n_units: u32,
    /// Per-record decode time, accumulated over the per-unit phase.
    /// Covers `decode_record` + `ChunkPayload::decode_with_kv_range`
    /// + `BlockAllocSpec` construction (Arc-wrap of pal/scale,
    /// Vec<KvFormat> build). Microseconds.
    pub decode_us: u64,
    /// Time inside `alloc_sealed_blocks_bulk` summed across every
    /// per-unit, per-layer dispatch. Microseconds.
    pub bulk_alloc_us: u64,
    /// Time inside `resolve_block_ptrs_from_hgids` summed similarly.
    /// Microseconds.
    pub resolve_ptrs_us: u64,
    /// Number of `alloc_sealed_blocks_bulk` invocations.
    pub n_bulk_calls: u32,
    /// Number of `resolve_block_ptrs_from_hgids` invocations.
    pub n_resolve_calls: u32,
    /// Sub-sub-timer of `bulk_alloc_us`: Phase 1, batched per-key gid
    /// alloc outside the state lock.
    pub pool_us: u64,
    /// Sub-sub-timer of `bulk_alloc_us`: Phase 2, per-spec slot
    /// mutation inside `state.write()`.
    pub register_us: u64,
    /// Sub-sub-timer of `bulk_alloc_us`: Phase 3, `resolve_arena_info`
    /// + per-layer `update_gpu_chunks_bulk` (coalesced `memcpy_htod`
    /// runs onto the CUDA stream).
    pub gpu_push_us: u64,
}

/// Run the pipelined cold-load for one [`ChunkBatch`]. The pinned
/// scratch (in `stager`) is filled by reader workers; the alloc'd
/// blocks land in the per-layer slots referenced by `slots`; the
/// kv_migrate scatter happens at the end of the pipeline.
///
/// Returns once `kv_migrate`'s drop syncs (i.e. all GPU work for this
/// chunk batch has completed and the pinned scratch is free for reuse
/// by the next batch).
pub fn run_pipeline(
    persistence: &SubstratePersistence,
    sealed: &std::collections::HashMap<SegmentId, DirectFile>,
    backings: &[ChunkedKvBacking],
    device: &Device,
    chunk_batch: &ChunkBatch,
    stager: &mut ColdLoadStager,
    slots: &[usize],
    chunks_per_layer: usize,
) -> candle::Result<PipelineStats> {
    if chunk_batch.total_bytes == 0 {
        return Ok(PipelineStats::default());
    }

    let unit_plan = super::chunk_plan::plan_units(chunk_batch, UNIT_BYTES);
    let n_units = unit_plan.units.len();

    let dev = match device {
        Device::Cuda(d) => d,
        _ => return Err(candle::Error::Msg("pipeline requires CUDA device".into())),
    };
    let stream: Arc<CudaStream> = dev.cuda_stream();

    // Ensure pinned scratch is sized; capture its raw base pointer
    // for cross-thread reader access.
    let _ = stager.buffer_mut(chunk_batch.total_bytes);
    let pinned_ptr = PinnedPtr(stager.buffer_ptr_mut());

    // One big GPU staging buffer mirroring the pinned scratch layout.
    // Per-unit memcpy_htod fills sub-ranges; the migrate plan sources
    // contiguous record ranges from staging_base + record.src_offset.
    let mut staging: CudaSlice<u8> = unsafe { stream.alloc::<u8>(chunk_batch.total_bytes).w()? };
    let staging_base: i64 = {
        let (p, _g) = staging.device_ptr(&stream);
        p as i64
    };

    // Per-unit list of records that contain this unit (used to
    // decrement record wait counts). Per-record wait count =
    // last_unit - first_unit + 1.
    let mut unit_to_records: Vec<Vec<usize>> = vec![Vec::new(); n_units];
    let mut record_wait_count: Vec<u32> = vec![0; chunk_batch.records.len()];
    for (r_idx, ru) in unit_plan.record_units.iter().enumerate() {
        record_wait_count[r_idx] = ru.last_unit - ru.first_unit + 1;
        for u in ru.first_unit..=ru.last_unit {
            unit_to_records[u as usize].push(r_idx);
        }
    }

    let (work_tx, work_rx) = unbounded::<ReadWork>();
    let (done_tx, done_rx) = unbounded::<UnitDone>();
    let (dispatch_tx, dispatch_rx) = unbounded::<UnitWork>();

    // Pre-load all read work. Reader pool drains the queue.
    for (idx, unit) in unit_plan.units.iter().enumerate() {
        let _ = work_tx.send(ReadWork {
            unit_idx: idx as u32,
            file_offset: unit.file_offset,
            dest_offset: unit.buf_offset,
            length: unit.length,
            source: chunk_batch.source,
        });
    }
    drop(work_tx);

    let t_total = Instant::now();
    let n_readers = persistence.active_direct_file().n_handles();

    // Shared "wall-clock time of the latest read completion across all
    // 16 reader threads". Each reader does `fetch_max` after sending
    // its UnitDone; after the scope, the final value is the moment
    // the last disk read returned. This decouples the read metric
    // from allocator/dispatcher latency.
    let reads_done_at_ms = Arc::new(AtomicU64::new(0));

    let ((htod_ms, migrate_ms), alloc_brk): ((u64, u64), AllocBreakdown) =
        std::thread::scope(|s| {
            // ── Reader pool ────────────────────────────────────────────
            for handle_idx in 0..n_readers {
                let work_rx = work_rx.clone();
                let done_tx = done_tx.clone();
                let reads_done_at_ms = Arc::clone(&reads_done_at_ms);
                s.spawn(move || {
                    while let Ok(work) = work_rx.recv() {
                        let direct = match work.source {
                            SourceLog::Active => persistence.active_direct_file(),
                            SourceLog::Sealed(id) => match sealed.get(&id) {
                                Some(d) => d,
                                // Every `Sealed(id)` in the plan gets a handle
                                // opened up front by `load_stream_into_hot`; a
                                // miss is a bug, but propagate it as a read
                                // error rather than panicking the reader thread.
                                None => {
                                    let _ = done_tx.send(UnitDone {
                                        unit_idx: work.unit_idx,
                                        error: Some(candle::Error::Msg(format!(
                                            "pipeline: no open handle for sealed segment {id}"
                                        ))),
                                    });
                                    continue;
                                }
                            },
                            SourceLog::Inherited(i) => persistence.inherited_direct_file(i),
                        };
                        // SAFETY: per-unit dest regions are disjoint
                        // (units are byte ranges, not overlapping). No
                        // other thread writes to this region until
                        // UnitDone is sent. Going through `slice_mut`
                        // (rather than `pinned_ptr.0.add(...)`) keeps the
                        // closure's capture as the whole PinnedPtr.
                        let dest = unsafe { pinned_ptr.slice_mut(work.dest_offset, work.length) };
                        let result = direct.read_at_with_handle(handle_idx, work.file_offset, dest);
                        // Record this reader's completion time. The
                        // pipeline-level `reads_ms` is the max of all
                        // readers' completion times — i.e. when the
                        // last disk read returned.
                        let now_ms = t_total.elapsed().as_millis() as u64;
                        reads_done_at_ms.fetch_max(now_ms, Ordering::Relaxed);
                        let _ = done_tx.send(UnitDone {
                            unit_idx: work.unit_idx,
                            error: result
                                .err()
                                .map(|e| candle::Error::Msg(format!("pipeline read: {e}"))),
                        });
                    }
                });
            }
            drop(done_tx);

            // Shared immutable references for both allocator & dispatcher.
            let unit_plan_ref: &UnitPlan = &unit_plan;

            // ── Allocator ──────────────────────────────────────────────
            let alloc_handle = s.spawn(move || -> candle::Result<AllocBreakdown> {
                allocator_worker(
                    backings,
                    chunk_batch,
                    unit_plan_ref,
                    pinned_ptr,
                    slots,
                    chunks_per_layer,
                    unit_to_records,
                    record_wait_count,
                    done_rx,
                    dispatch_tx,
                )
            });

            // ── Main thread: GPU dispatcher ────────────────────────────
            let dispatcher_result = dispatcher_main(
                device,
                &stream,
                pinned_ptr,
                unit_plan_ref,
                staging_base,
                &mut staging,
                dispatch_rx,
            );

            let alloc_result = alloc_handle
                .join()
                .unwrap_or_else(|_| Err(candle::Error::Msg("pipeline allocator panicked".into())));

            match (dispatcher_result, alloc_result) {
                (Ok(ms), Ok(brk)) => Ok((ms, brk)),
                (Err(e), _) | (_, Err(e)) => Err(e),
            }
        })?;

    // Drop staging here — drop syncs the stream so all GPU work has
    // completed before the pinned scratch is reused by the next
    // chunk batch.
    drop(staging);

    let total_ms = t_total.elapsed().as_millis() as u64;
    let reads_ms = reads_done_at_ms.load(Ordering::Relaxed);
    Ok(PipelineStats {
        reads_ms,
        alloc_ms: alloc_brk.alloc_ms,
        htod_ms,
        migrate_ms,
        total_ms,
        n_units: n_units as u32,
        decode_us: alloc_brk.decode_us,
        bulk_alloc_us: alloc_brk.bulk_alloc_us,
        resolve_ptrs_us: alloc_brk.resolve_ptrs_us,
        n_bulk_calls: alloc_brk.n_bulk_calls,
        n_resolve_calls: alloc_brk.n_resolve_calls,
        pool_us: alloc_brk.pool_us,
        register_us: alloc_brk.register_us,
        gpu_push_us: alloc_brk.gpu_push_us,
    })
}

#[allow(clippy::too_many_arguments)]
fn allocator_worker(
    backings: &[ChunkedKvBacking],
    chunk_batch: &ChunkBatch,
    unit_plan: &UnitPlan,
    pinned_ptr: PinnedPtr,
    slots: &[usize],
    chunks_per_layer: usize,
    unit_to_records: Vec<Vec<usize>>,
    mut record_wait_count: Vec<u32>,
    done_rx: Receiver<UnitDone>,
    dispatch_tx: Sender<UnitWork>,
) -> candle::Result<AllocBreakdown> {
    let n_layers = backings.len();
    // SAFETY: pinned scratch is alive for the whole pipeline run.
    let buf: &[u8] = unsafe { pinned_ptr.slice(0, chunk_batch.total_bytes) };

    // Pre-allocate each layer's slot block table + chunk vec **once**
    // for the whole cold-load. Per-unit `alloc_sealed_blocks_bulk`
    // calls then inherit this capacity and skip their own
    // `ensure_max_blocks` + `ensure_for_offset` — both of which
    // acquire `state.write()` and dominated the in-call gap (~10 ms
    // wall-clock on the 1824-chunk turn).
    for (li, backing) in backings.iter().enumerate() {
        backing.ensure_capacity_for_blocks(slots[li], chunks_per_layer)?;
    }

    let total_units = unit_plan.units.len() as u32;
    let mut units_received: u32 = 0;

    // Top-level allocator-thread CPU time and its sub-buckets.
    // Everything is accumulated in MICROSECONDS — `.as_millis()` would
    // truncate per-unit elapsed under 1 ms, losing up to `n_units` ms
    // of accumulated signal. We convert to ms only at the output
    // boundary.
    let mut alloc_us_accum: u64 = 0;
    let mut decode_us: u64 = 0;
    let mut bulk_alloc_us: u64 = 0;
    let mut resolve_ptrs_us: u64 = 0;
    let mut n_bulk_calls: u32 = 0;
    let mut n_resolve_calls: u32 = 0;
    let mut pool_us: u64 = 0;
    let mut register_us: u64 = 0;
    let mut gpu_push_us: u64 = 0;

    while units_received < total_units {
        let done = done_rx
            .recv()
            .map_err(|_| candle::Error::Msg("pipeline: read-done channel closed".into()))?;
        units_received += 1;
        if let Some(err) = done.error {
            return Err(err);
        }
        let w = done.unit_idx as usize;

        // Decrement wait counts for records that include this unit.
        // Collect records whose count just hit zero.
        let mut ready: Vec<usize> = Vec::new();
        for &r_idx in &unit_to_records[w] {
            record_wait_count[r_idx] -= 1;
            if record_wait_count[r_idx] == 0 {
                ready.push(r_idx);
            }
        }

        let t_alloc = Instant::now();
        let mut records_to_dispatch: Vec<RecordMigrate> = Vec::new();

        if !ready.is_empty() {
            // Group ready records by layer for batched alloc. Per-unit
            // dispatch keeps the alloc work overlapped with subsequent
            // reads on the allocator thread — measured empirically to
            // beat end-of-batch deferral (lost pipelining + per-spec
            // re-clones outweighed the per-call overhead saving).
            let mut per_layer: Vec<Vec<(usize, BlockAllocSpec, i64)>> =
                (0..n_layers).map(|_| Vec::new()).collect();

            // ── Sub-timer: decode + spec construction ────────────
            let t_decode = Instant::now();
            for &r_idx in &ready {
                let rec = &chunk_batch.records[r_idx];
                let record_bytes = &buf[rec.buf_offset..rec.buf_offset + rec.record_size as usize];
                // Single pass: one header parse + meta parse. `payload`
                // is a borrowed view into `record_bytes`, so we recover
                // the header byte count from the pointer offset instead
                // of re-decoding the header.
                let (header, payload, _total) = decode_record(record_bytes)
                    .map_err(|e| candle::Error::Msg(format!("pipeline decode_record: {e}")))?;
                let header_bytes = payload.as_ptr() as usize - record_bytes.as_ptr() as usize;
                let payload_buf_offset = rec.buf_offset + header_bytes;
                let (meta, kv_range) = ChunkPayload::decode_with_kv_range(payload)
                    .map_err(|e| candle::Error::Msg(format!("pipeline payload: {e}")))?;
                // Read-into-VRAM golden check at the consumption point: the KV
                // bytes were just read off disk and are about to scatter into GPU
                // arenas. Recompute the golden and compare to the stored value —
                // NON-fatal: a mismatch warns (possible on-disk / read-path
                // corruption) but still loads, so the latency-sensitive cold-load
                // never hard-fails on it.
                let recomputed = candle::fletcher::fletcher32(&payload[kv_range.start..kv_range.end]);
                if recomputed != header.crc {
                    tracing::warn!(
                        target: "candle_conversation::persistence::golden",
                        chunk_idx = rec.chunk_idx,
                        stored = header.crc,
                        recomputed,
                        "chunk golden mismatch on cold-load into VRAM — possible on-disk/read corruption"
                    );
                }
                let src_offset = (payload_buf_offset + kv_range.start) as i64;

                let layer = (rec.chunk_idx as usize) / chunks_per_layer;
                let block_idx = (rec.chunk_idx as usize) % chunks_per_layer;
                if layer >= n_layers {
                    return Err(candle::Error::Msg(format!(
                        "pipeline: chunk_idx {} routes to layer {} but only {} layers",
                        rec.chunk_idx, layer, n_layers
                    )));
                }

                let k_formats: candle::Result<Vec<KvFormat>> = meta
                    .k_formats
                    .iter()
                    .map(|&t| {
                        KvFormat::from_tag(t).ok_or_else(|| {
                            candle::Error::Msg(format!("pipeline: bad k format tag {t}"))
                        })
                    })
                    .collect();
                let v_formats: candle::Result<Vec<KvFormat>> = meta
                    .v_formats
                    .iter()
                    .map(|&t| {
                        KvFormat::from_tag(t).ok_or_else(|| {
                            candle::Error::Msg(format!("pipeline: bad v format tag {t}"))
                        })
                    })
                    .collect();

                let spec = BlockAllocSpec {
                    block_idx,
                    k_formats: k_formats?,
                    v_formats: v_formats?,
                    k_pal: Arc::new(meta.k_pal),
                    v_pal: Arc::new(meta.v_pal),
                    k_scale: Arc::new(meta.k_scale),
                    v_scale: Arc::new(meta.v_scale),
                    offset: meta.offset,
                    usage: rec.token_count as u32,
                };

                per_layer[layer].push((r_idx, spec, src_offset));
            }
            decode_us += t_decode.elapsed().as_micros() as u64;

            // ── Per-layer alloc + resolve dst ptrs ───────────────
            for (li, layer_recs) in per_layer.iter().enumerate() {
                if layer_recs.is_empty() {
                    continue;
                }
                let specs: Vec<BlockAllocSpec> =
                    layer_recs.iter().map(|(_, s, _)| s.clone()).collect();

                let t_bulk = Instant::now();
                let (hgids, arena_info) = backings[li].alloc_sealed_blocks_bulk(
                    slots[li],
                    &specs,
                    &mut pool_us,
                    &mut register_us,
                    &mut gpu_push_us,
                )?;
                bulk_alloc_us += t_bulk.elapsed().as_micros() as u64;
                n_bulk_calls += 1;

                // Resolve dst ptrs from the freshly-allocated hgids and
                // the arena_info snapshot — no `state.read()`, no per-
                // block dedup. Fresh CAS-claimed gids are unique by
                // construction.
                let t_resolve = Instant::now();
                let dst_ptrs_per_rec =
                    backings[li].resolve_block_ptrs_from_hgids(&hgids, &arena_info)?;
                resolve_ptrs_us += t_resolve.elapsed().as_micros() as u64;
                n_resolve_calls += 1;

                for ((_, _, src_offset), dst_ptrs) in
                    layer_recs.iter().zip(dst_ptrs_per_rec.into_iter())
                {
                    records_to_dispatch.push(RecordMigrate {
                        src_offset: *src_offset,
                        dst_ptrs,
                    });
                }
            }
        }
        alloc_us_accum += t_alloc.elapsed().as_micros() as u64;

        // Even if no records anchored here, send UnitWork so the
        // dispatcher knows this unit's bytes are ready for HtoD —
        // records passing through this unit but anchored later still
        // need the HtoD to happen.
        let _ = dispatch_tx.send(UnitWork {
            unit_idx: w as u32,
            records: records_to_dispatch,
        });
    }

    // Drop dispatch_tx (implicit at function return) signals the
    // dispatcher to exit its recv loop.
    Ok(AllocBreakdown {
        alloc_ms: alloc_us_accum / 1000,
        decode_us,
        bulk_alloc_us,
        resolve_ptrs_us,
        n_bulk_calls,
        n_resolve_calls,
        pool_us,
        register_us,
        gpu_push_us,
    })
}

/// Returns `(htod_ms, migrate_ms)` — the two stages this thread owns.
/// The other timing fields in [`PipelineStats`] are filled in by
/// [`run_pipeline`] from its own measurements.
#[allow(clippy::too_many_arguments)]
fn dispatcher_main(
    device: &Device,
    stream: &Arc<CudaStream>,
    pinned_ptr: PinnedPtr,
    unit_plan: &UnitPlan,
    staging_base: i64,
    staging: &mut CudaSlice<u8>,
    dispatch_rx: Receiver<UnitWork>,
) -> candle::Result<(u64, u64)> {
    let mut migrate_plan = MigrationPlan::new();
    let mut htod_ms_accum: u64 = 0;

    while let Ok(unit_work) = dispatch_rx.recv() {
        let unit = &unit_plan.units[unit_work.unit_idx as usize];
        // SAFETY: reader has finished writing this unit's bytes
        // (UnitDone → allocator → UnitWork ordering).
        let src: &[u8] = unsafe { pinned_ptr.slice(unit.buf_offset, unit.length) };
        let mut dst_view = staging.slice_mut(unit.buf_offset..unit.buf_offset + unit.length);
        let t_htod = Instant::now();
        stream.memcpy_htod(src, &mut dst_view).w()?;
        htod_ms_accum += t_htod.elapsed().as_millis() as u64;

        for rec in &unit_work.records {
            let mut cur_src = staging_base + rec.src_offset;
            for &(dst_ptr, dst_len) in &rec.dst_ptrs {
                migrate_plan.push(cur_src, dst_ptr, dst_len);
                cur_src += dst_len;
            }
        }
    }

    let t_migrate = Instant::now();
    kv_migrate(device, &migrate_plan)?;
    let migrate_ms = t_migrate.elapsed().as_millis() as u64;

    Ok((htod_ms_accum, migrate_ms))
}
