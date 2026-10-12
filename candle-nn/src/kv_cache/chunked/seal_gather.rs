//! Read sealed chunks off the device into host buffers of their own, each with
//! the Fletcher-32 golden of its bytes as the device produced them.
//!
//! The seal path's gather — warm eviction and the redo-log write both go
//! through it — took a staging buffer from the driver for the whole sealed
//! sequence, read it back with one synchronous copy into a pageable blob, and
//! then copied every chunk out of that blob into a buffer of its own. Here the
//! staging is the persistence domain's span, batched to what the span holds, and
//! each chunk is read straight into its own buffer: nothing is allocated on the
//! device, and every byte crosses host memory once.

use std::ops::Range;

#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_async;
#[cfg(feature = "cuda")]
use candle::cuda_backend::WrapErr;
#[cfg(feature = "cuda")]
use candle::Device;
use candle::Result;

#[cfg(feature = "cuda")]
use super::bump_arena::{persistence_domain, NOT_A_WAVE};
#[cfg(feature = "cuda")]
use super::chunk_ops::{staging_groups, MIGRATION_STAGING_CAP_BYTES};
#[cfg(feature = "cuda")]
use super::fletcher_golden::{fletcher32_golden, GoldenRecord};
#[cfg(feature = "cuda")]
use super::migrate::{kv_migrate, MigrationPlan};

/// One chunk read back: its bytes and their golden.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatheredChunk {
    pub bytes: Vec<u8>,
    pub golden: u32,
}

/// Which pieces make up each chunk: the pieces are a chunk's bands in order, so
/// consecutive pieces fill each chunk in turn, and every chunk must be filled
/// exactly — a piece straddling two chunks, or pieces left over, means the
/// addresses were resolved for a different sequence than the sizes.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn pieces_per_chunk(
    piece_lens: &[usize],
    chunk_sizes: &[usize],
) -> Result<Vec<Range<usize>>> {
    let mut out = Vec::with_capacity(chunk_sizes.len());
    let mut p = 0usize;
    for (c, &size) in chunk_sizes.iter().enumerate() {
        let start = p;
        let mut filled = 0usize;
        while filled < size {
            let Some(&len) = piece_lens.get(p) else {
                candle::bail!(
                    "seal gather: chunk {c} needs {size} B but the pieces ran out at {filled} B"
                );
            };
            filled += len;
            p += 1;
        }
        if filled != size {
            candle::bail!("seal gather: chunk {c} is {size} B but its pieces span {filled} B");
        }
        out.push(start..p);
    }
    if p != piece_lens.len() {
        candle::bail!(
            "seal gather: {} piece(s) left over after {} chunk(s)",
            piece_lens.len() - p,
            chunk_sizes.len()
        );
    }
    Ok(out)
}

/// Gather a sealed sequence to the host: `pieces` are its bands' `(device
/// address, byte length)` in chunk order, and `chunk_sizes` partitions them into
/// chunks. One [`GatheredChunk`] per chunk, in order.
///
/// Runs on the device's stream and synchronises it once per staging group,
/// before that group's range goes back to the domain.
#[cfg(feature = "cuda")]
pub fn gather_sealed_chunks(
    device: &Device,
    pieces: &[(i64, i64)],
    chunk_sizes: &[usize],
) -> Result<Vec<GatheredChunk>> {
    let Device::Cuda(dev) = device else {
        candle::bail!("gather_sealed_chunks requires a CUDA device");
    };
    let piece_lens: Vec<usize> = pieces.iter().map(|&(_, len)| len as usize).collect();
    let owned = pieces_per_chunk(&piece_lens, chunk_sizes)?;
    let stream = dev.cuda_stream();
    let domain = persistence_domain(&stream)?;
    let mut out = Vec::with_capacity(chunk_sizes.len());
    for (i, j, group_bytes) in staging_groups(chunk_sizes, MIGRATION_STAGING_CAP_BYTES) {
        // One generation per group: the cursor rewinds when it drops, and by
        // then the group's readback below has completed.
        let generation = domain.generation(NOT_A_WAVE, NOT_A_WAVE)?;
        let staging = generation.alloc(group_bytes.max(1), 256)?;
        let mut plan = MigrationPlan::new();
        let mut records = Vec::with_capacity(j - i);
        let mut at = staging.ptr;
        for c in i..j {
            records.push(GoldenRecord {
                src_ptr: at as i64,
                byte_len: chunk_sizes[c] as i64,
            });
            for &(ptr, len) in &pieces[owned[c].clone()] {
                plan.push(ptr, at as i64, len);
                at += len as u64;
            }
        }
        kv_migrate(device, &plan)?;
        // Over the staging bytes, before any of them crosses the bus — the
        // ground truth the reload checks the host copy against.
        let goldens = fletcher32_golden(device, &records)?;
        for (record, golden) in records.iter().zip(goldens) {
            let mut bytes = vec![0u8; record.byte_len as usize];
            if !bytes.is_empty() {
                // SAFETY: the range is `bytes.len()` of this group's staging,
                // held by `generation` until the synchronise below.
                unsafe { memcpy_dtoh_async(&mut bytes, record.src_ptr as u64, stream.cu_stream()) }
                    .w()?;
            }
            out.push(GatheredChunk { bytes, golden });
        }
        stream.synchronize().w()?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::pieces_per_chunk;

    /// Consecutive pieces fill each chunk in turn; an empty chunk owns none.
    #[test]
    fn pieces_fill_their_chunks_in_order() {
        let got = pieces_per_chunk(&[4, 4, 8, 2, 2, 2], &[8, 8, 0, 6]).unwrap();
        assert_eq!(got, vec![0..2, 2..3, 3..3, 3..6]);
    }

    /// A piece straddling a chunk boundary, too few pieces, and pieces left
    /// over are each refused.
    #[test]
    fn mismatched_pieces_are_refused() {
        assert!(pieces_per_chunk(&[6, 6], &[8, 4]).is_err());
        assert!(pieces_per_chunk(&[4], &[8]).is_err());
        assert!(pieces_per_chunk(&[4, 4, 4], &[8]).is_err());
    }
}
