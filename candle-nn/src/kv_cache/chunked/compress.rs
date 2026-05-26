//! Decode-path quantization compression for KV cache chunks.
//!
//! This module contains the v2 reconcile path that is invoked during decode
//! (as opposed to prefill). The v2 path now derives the real per-palette K/V
//! layout from sampled selection and persists that routing through the palette
//! maps consumed by the decode and prefill kernels.

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use candle::quantized::cuda::{dtype_to_ggml_float, identity_pal_map_128, PalHeadDesc};
#[cfg(feature = "cuda")]
use candle::quantized::GgmlDType;
#[cfg(feature = "cuda")]
use candle::Device;
#[cfg(feature = "cuda")]
use candle::Result;

#[cfg(feature = "cuda")]
use super::head_gids::{HeadGids, GIDS_PER_HEAD};
#[cfg(feature = "cuda")]
use super::sampled_selection::{PagedSelectionGpuInputs, SampleFormat};
#[cfg(feature = "cuda")]
use super::{
    ArenaKey, ChunkedKvBacking, CompressionPolicy, CHUNK_SIZE, PRODUCTION_K_QREL_HIGH_THRESHOLDS,
    PRODUCTION_K_QREL_LOW_THRESHOLDS, PRODUCTION_V_QREL_HIGH_THRESHOLDS,
    PRODUCTION_V_QREL_LOW_THRESHOLDS,
};
#[cfg(feature = "cuda")]
use crate::kv_cache::arena_table::N_PALETTE;
#[cfg(feature = "cuda")]
use crate::kv_cache::chunked::backing::BackingInner;
#[cfg(feature = "cuda")]
use crate::kv_cache::{KvFormat, QuantFormat};
#[cfg(feature = "cuda")]
use crate::ChunkGid;
#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::CudaStream;
#[cfg(feature = "cuda")]
use candle::quantized::pinned_staging::Generation;

#[cfg(feature = "cuda")]
impl ChunkedKvBacking {
    fn pack_head_palette_maps(
        assignments: &[u8],
        n_batch: usize,
        n_head: usize,
        head_dim: usize,
    ) -> Result<Vec<Vec<u8>>> {
        let packed_bytes = head_dim / 4;
        let expected_packed_bytes = identity_pal_map_128().len();
        if head_dim % 4 != 0 || head_dim % N_PALETTE != 0 || packed_bytes != expected_packed_bytes {
            candle::bail!(
                "reconcile_batch_float_to_quant_v2: palette4 conversion requires head_dim={}, got {}",
                expected_packed_bytes * 4,
                head_dim
            );
        }

        let dims_per_head = head_dim;
        let expected = n_batch
            .checked_mul(n_head)
            .and_then(|v| v.checked_mul(dims_per_head))
            .ok_or_else(|| candle::Error::Msg("palette map shape overflow".into()))?;
        if assignments.len() != expected {
            candle::bail!(
                "palette map length mismatch: got {}, expected {} ({} dims/head × {} batch-heads)",
                assignments.len(),
                expected,
                dims_per_head,
                n_batch * n_head
            );
        }

        let target_per_palette = head_dim / N_PALETTE;
        let mut head_palette_maps = Vec::with_capacity(n_batch);
        for block_i in 0..n_batch {
            let mut block_map = vec![0u8; n_head * packed_bytes];
            for h in 0..n_head {
                let base = (block_i * n_head + h) * dims_per_head;
                let mut counts = [0usize; N_PALETTE];
                for (dim_offset, &raw_slot) in
                    assignments[base..base + dims_per_head].iter().enumerate()
                {
                    let slot = (raw_slot & 0x3) as usize;
                    counts[slot] += 1;
                    let byte_idx = h * packed_bytes + (dim_offset / 4);
                    block_map[byte_idx] |= (slot as u8) << (2 * (dim_offset % 4));
                }
                if counts.iter().any(|&count| count != target_per_palette) {
                    candle::bail!(
                        "invalid palette4 routing for block {} head {}: {:?}",
                        block_i,
                        h,
                        counts
                    );
                }
            }
            head_palette_maps.push(block_map);
        }
        Ok(head_palette_maps)
    }
}

