//! Carves the dense weight block from the left of the span before regions exist.
//!
//! The model loader bump-allocates its dense weights from the left of the span,
//! and the region grid is positioned from the block's right edge, so the block
//! may grow only until the first region is claimed. Each allocation is aligned
//! to the driver's guarantee for fresh memory, and its bytes arrive zeroed,
//! which the quantized matmul kernels rely on when they over-read a weight's
//! padding. `freeze_dense` closes the block when loading ends and retracts any
//! weight zone a previous model left behind, so a second model in one process
//! does not inherit the first one's floor.

use super::*;

/// Alignment every dense weight tensor is placed on.
///
/// What the driver guarantees a fresh `cuMemAlloc`, so a tensor carved out of
/// the span is aligned exactly as it would have been coming from the pool and no
/// kernel can tell the difference. The tensor-core loads assume it of a base
/// pointer.
pub(super) const DENSE_ALIGN: u64 = 256;

/// Carve `bytes` of the dense weight block, returning its device address.
///
/// The model loader's allocator: a bump pointer over `[dense_base, region_base)`
/// that grows as tensors arrive. No estimate of the model's size is needed
/// anywhere — the span is claimed before the load begins and the region count is
/// derived from whatever the weights leave behind.
///
/// # It may only run before the first region is claimed
///
/// Every region address is `region_base + i · REGION_BYTES`, and `region_base`
/// is this block's right edge. Growing the block after a region exists would
/// move the ground under a live arena, so the boundary is frozen by the first
/// claim rather than by a phase flag somebody has to remember to set. A dense
/// claim after that point is refused, not clamped: it means the loader is
/// running concurrently with inference, and there is no correct address to
/// return.
///
/// # Alignment
///
/// 256 bytes, which is what the driver guarantees a fresh `cuMemAlloc` and what
/// the tensor cores' vectorised loads assume of a base pointer. The block itself
/// is region-aligned on the right, so the KV grid is unaffected by whatever the
/// weights leave part-used.
///
/// # The bytes arrive zeroed, and a q-matmul operand depends on it
///
/// A quantized weight is allocated with `MATRIX_ROW_PADDING` past its data,
/// because the q-matmul kernels **over-read into that tail by design** —
/// `QCudaStorage::zeros` exists precisely to make that read a defined zero. A
/// caller placing a weight here must therefore claim the *padded* size, and the
/// tail must be zero.
///
/// It is, for a reason specific to this block: the span is freshly mapped and
/// the dense block is handed out exactly once, so nothing has ever written these
/// bytes. That is a stronger guarantee than the region free list's — which needs
/// its `dirt` record and a memset because a region can be recycled — and it holds
/// only as long as the block keeps its no-release discipline. If a dense free
/// path is ever added, the padding stops being defined and the q-matmul
/// over-read becomes a read of another weight's bytes.
pub fn claim_dense(stream: &std::sync::Arc<CudaStream>, bytes: usize) -> Result<u64> {
    with_pool(stream, |pool| {
        if !pool.dense_capable {
            candle::bail!(
                "dense weights: this span was sized from the fixed test constant rather \
                 than a measured card, so it holds only KV — a model placed here would \
                 take every region. The CUDA pool is the right home for these weights."
            )
        }
        if pool.dense_frozen {
            candle::bail!(
                "dense weights: the block was frozen at {} MiB when loading finished. \
                 A claim after that point is a weight arriving late — there is no \
                 address to give it, because everything above the block now belongs \
                 to the KV side.",
                pool.dense_bytes / (1024 * 1024),
            )
        }
        if pool.next > 0 || pool.live > 0 {
            candle::bail!(
                "dense weights: {} regions have already been claimed and every one of \
                 their addresses is measured from this block's right edge, so growing \
                 it now would move ground a live arena is standing on. A dense claim \
                 here means weights are loading while inference runs.",
                pool.next,
            )
        }
        let at = pool.dense_base + pool.dense_bytes as u64;
        let aligned = at.next_multiple_of(DENSE_ALIGN);
        let end = aligned
            .checked_add(bytes as u64)
            .ok_or_else(|| candle::Error::Msg("dense weights: address overflow".into()))?;
        // Against the weight floor, not the span end: the expert zone is placed
        // from the right edge and the dense block grows from the left, so the
        // floor is what they would collide at. At load the floor stands at the
        // load scratch's base, so the block never grows onto the transients
        // served there; past that it binds only once an expert cache has
        // installed a zone, and a loader that reaches that state has its
        // ordering wrong anyway.
        if end > pool.weight_floor {
            candle::bail!(
                "dense weights: {bytes} B does not fit — the block would end at \
                 {end:#x}, past the weight floor at {:#x}. The span is {} MiB and \
                 the weights have taken {} MiB of it.",
                pool.weight_floor,
                pool.span_bytes / (1024 * 1024),
                pool.dense_bytes / (1024 * 1024),
            )
        }
        pool.dense_bytes = (end - pool.dense_base) as usize;
        pool.relayout_for_dense();
        Ok(aligned)
    })
}

/// Bytes of the span the model's dense weights hold.
pub fn dense_bytes(stream: &std::sync::Arc<CudaStream>) -> Result<usize> {
    with_pool(stream, |pool| Ok(pool.dense_bytes))
}

/// End the load phase: lock the dense block's right edge and hand the rest of
/// the span to the runtime.
///
/// The boundary between the two ways this span is used. Before it, the dense
/// block bumps upward and no region exists. After it, the block is immovable and
/// the ground from `region_base` to the span's right edge is the runtime's: KV
/// regions growing up, the expert zone filling down.
///
/// Idempotent, and returns the dense block's final size so a caller can report
/// what the model actually took rather than what it predicted. Nothing needs to
/// have predicted anything: that is the point of loading into the span rather
/// than sizing the span around a load that already happened.
pub fn freeze_dense(stream: &std::sync::Arc<CudaStream>) -> Result<usize> {
    let ordinal = stream.context().ordinal();
    with_pool(stream, |pool| {
        // **The load's scratch closes first**, while its ground is still nobody
        // else's: the weight side is placed there next. `close` refuses while a
        // transient the load scoped is still live, and the load fails here rather
        // than leave it under the expert slots that would overwrite it. Raising
        // the floor to the span's end grows the KV side, so it never cuts a live
        // region.
        if let Some(report) = load_scratch::close(ordinal)? {
            let regions = pool.set_weight_floor(pool.span_end())?;
            tracing::info!(
                target: "candle_nn::kv_cache::region_pool",
                scratch_mib = report.bytes >> 20,
                peak_mib = report.peak >> 20,
                spilled_mib = report.spilled >> 20,
                regions,
                "close_load: the load's scratch at the span's top is closed and its ground \
                 returned; the weight side is placed there next"
            );
        }
        pool.dense_frozen = true;
        // **Retract a previous model's weight zone.**
        //
        // `POOLS` is process-global and never torn down, so `weight_floor`
        // outlives the model that installed it. The load path assumes that at
        // close_load no weight floor has been installed, and that holds only for
        // the FIRST load in a process. A second load of a model with no
        // weight side of its own installs nothing, so it silently inherits the
        // previous one's floor and runs with whatever KV area that left.
        //
        // Measured: in one test binary, DeepSeek-V4-Flash (284B, MXFP4 experts)
        // installs a floor leaving the weight side ~47 GiB; the dense Llama-2
        // that loads next then gets **81 KV regions of a 3,974-region span** and
        // dies in `place_transient` at 8 contexts — "this wave is too wide for a
        // partition that has nothing left to trade", against 47 GiB of weight
        // ground holding 0.1 MiB. Alone, the same model passes. The same shape
        // reaches production wherever one process loads a second checkpoint.
        //
        // Here is the one safe moment: the load phase has just closed, so
        // nothing has claimed a region and no zone is built yet — `build_experts`
        // and `build_layers` both run after this. A model that wants a weight
        // side installs it immediately below; one that does not now starts from
        // the documented default of the whole span.
        //
        // **Guarded on `live`, not on `next`.** `next` is the lowest index never
        // handed out — a monotonic bump that never returns to 0 once any region
        // has been claimed — so it is non-zero on *every* load after the first,
        // which is exactly when this retraction is needed. Guarding on it refuses
        // the case it exists for. (That is not a hypothetical either: the first
        // version of this guard did precisely that, and the suite failed
        // identically with the fix in place.)
        //
        // `live` is the real occupancy, and it is the only thing that could make
        // this unsafe. Note the direction as well: retracting toward `span_end`
        // GROWS the KV side and can never cut a live KV region — the hazard the
        // partition rules warn about is the floor moving *down* onto them.
        if pool.weight_floor != pool.span_end() {
            if pool.live == 0 {
                let span_end = pool.span_end();
                match pool.set_weight_floor(span_end) {
                    Ok(regions) => tracing::info!(
                        target: "candle_nn::kv_cache::region_pool",
                        regions,
                        "close_load: retracted a previous model's weight zone; the span is the \
                         KV side's until a zone is installed for this load"
                    ),
                    Err(e) => tracing::warn!(
                        target: "candle_nn::kv_cache::region_pool",
                        "close_load: could not retract the previous weight zone ({e}); this \
                         load runs against the KV area that zone left"
                    ),
                }
            } else {
                tracing::warn!(
                    target: "candle_nn::kv_cache::region_pool",
                    live = pool.live,
                    "close_load: a weight zone from an earlier load is installed and KV regions \
                     are still live, so it cannot be retracted safely; this load runs against \
                     the KV area that zone left"
                );
            }
        }
        Ok(pool.dense_bytes)
    })
}
