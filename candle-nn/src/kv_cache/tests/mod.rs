//! Unit tests for the KV cache types, covering forks, prefix sharing, quantized
//! writes and reads, and the format definitions.
//!
//! The shared helpers build a chunked backing with fixed geometry and read its block
//! ids back as a table, so each test can assert which blocks each sequence holds.
//! The test files are organised by the behaviour they pin.
//!
//! Tests:
//!
//! - `tests_concurrency`: shared and freed blocks under concurrent access.
//! - `tests_fork_and_scattered`: forks of chunked and scattered sequences.
//! - `tests_format_tags`: format definitions, byte sizes and tags.
//! - `tests_new_formats`: the newer quantized formats' write and read accuracy.
//! - `tests_prefix_share`: prefix sharing and block ownership.
//! - `tests_quantized_write`: quantized writes, reads and round-trip accuracy.

use super::*;
use candle::{DType, Device, IndexOp, Result, Tensor};

/// Test helper: snapshot K-side GIDs as `[batch][block]` with -1 for unallocated.
fn k_gid_snapshot(backing: &ChunkedKvBacking) -> Vec<Vec<i64>> {
    let state = backing.state.read().expect("lock");
    let mb = state.max_blocks;
    state
        .sequences
        .iter()
        .map(|slot| {
            let mut row = vec![-1i64; mb];
            if let Some(s) = slot {
                for (i, cw) in s.chunks_slice().iter().enumerate() {
                    row[i] = cw.gids.k_gid(0).raw();
                }
            }
            row
        })
        .collect()
}

// ==================== KvCache Fork Tests ====================

fn make_chunked_kvcache(backing: &ChunkedKvBacking) -> Result<KvCache> {
    let batch_idx = backing.alloc_sequence()?;
    let mut kv = KvCache::new(2, 256); // dim=2 (sequence dimension)
    kv.set_chunked_backing(backing, batch_idx, None)?;
    Ok(kv)
}

/// Read the head of one band's slot back as `f32`.
///
/// The arena is a run of untyped bytes, so the dtype is the caller's to
/// supply — these fixtures write `F32` through `write_contiguous`.
fn band_f32(backing: &ChunkedKvBacking, raw: usize, elems: usize) -> Result<Vec<f32>> {
    backing.with_arenas(|arenas| {
        arenas
            .get(&(raw / GID_STRIDE))
            .expect("gid must point at a live arena")
            .read_slot_typed(raw % GID_STRIDE, DType::F32, elems)?
            .to_vec1::<f32>()
    })?
}

// ==================== ChunkedKvBacking Prefix Sharing Tests ====================

fn make_test_backing(batch: usize) -> Result<ChunkedKvBacking> {
    ChunkedKvBacking::new(
        batch,
        4,  // n_kv_head
        64, // head_dim
        DType::F32,
        &Device::Cpu,
        256, // initial_max_seq_len
    )
}

mod tests_concurrency;
mod tests_fork_and_scattered;
mod tests_format_tags;
mod tests_new_formats;
mod tests_prefix_share;
mod tests_quantized_write;
