//! Holds the launch-time resources the latent kernels need: the split-count
//! policy and the shared partial workspace.
//!
//! The kernels combine split-KV partials through a workspace the launcher
//! cannot allocate for itself, so it is built once and shared by every launch
//! on a stream. Its capacity bounds every launch's rows times splits, and
//! prefill launches are chunked to fit it, with bit-identical results per row.
//! The split count is chosen so the device is filled, unless a test pins it
//! for determinism.

use super::*;

/// MMA M dimension: query heads per block (kernel HEADS_TILE).
pub const HEADS_TILE: usize = 16;

/// Split-KV factor policy for the decode kernel: enough blocks to fill the
/// device (~2 waves of 4 blocks/SM), pinned by `override_` for test
/// determinism. Resolved HERE because the caller owns the partial workspace —
/// the kernel launcher has no allocator and no shared state.
#[cfg(feature = "cuda")]
pub(super) fn decode_num_splits(num_slots: usize, n_q_head: usize, override_: usize) -> usize {
    if override_ > 0 {
        return override_.min(32);
    }
    let sm = unsafe { candle_kernels::paged_latent::run_latent_sm_count() }.max(1) as usize;
    let base_blocks = num_slots * n_q_head.div_ceil(HEADS_TILE);
    (sm * 8).div_ceil(base_blocks).clamp(1, 32)
}

/// Capacity of the shared partial workspace in `row × split` units — bounds
/// every launch's `rows·splits` product (`rows = queries-or-slots × heads`).
/// Prefill launches CHUNK their queries to fit (queries are mutually
/// independent, so chunked launches are bit-identical per row); the decode
/// split policy's product is bounded well under this for any wave size.
pub const WORKSPACE_CAP: usize = 32 * 1024;

/// Split-KV partial workspace for the latent kernels: the
/// `[rows, splits, HEAD_DIM]` accumulator + `[rows, splits, 2]` (m, l) pair
/// every decode/prefill launch hands to its combine. Built ONCE (fixed
/// `WORKSPACE_CAP`, ~64 MiB) and shared by every launch on a stream — peak
/// VRAM is a single buffer instead of per-launch allocations churning the
/// pool. It is IMMUTABLE on the host (only kernels write the device memory,
/// stream-ordered), so sharing is plain `&`/`Arc` — no device-side pool, no
/// static, no lock. The one ownership rule: host threads that launch
/// concurrently onto the SAME stream must not share one workspace (their
/// launch/combine pairs would interleave); the engine's single wave thread
/// and per-test instances satisfy this by construction.
#[cfg(feature = "cuda")]
pub struct LatentWorkspace {
    pub(super) acc: Tensor,
    pub(super) ml: Tensor,
}

#[cfg(feature = "cuda")]
impl LatentWorkspace {
    pub fn build(dev: &candle::Device) -> Result<Self> {
        Ok(Self {
            acc: Tensor::zeros(WORKSPACE_CAP * HEAD_DIM, DType::F32, dev)?,
            ml: Tensor::zeros(WORKSPACE_CAP * 2, DType::F32, dev)?,
        })
    }
}
