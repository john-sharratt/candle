//! Where each band of a sequence's chunks lives, read from the host block table.
//!
//! The decode kernel finds a band through a chunk's serialised `KvHead` record.
//! This answers the same question from the block table that record was built
//! from, with the same arithmetic (`meta_pool::serialize_kv_heads`), so a
//! capture can read a band's bytes without a header table to walk — and reads
//! them raw, which matters because the active K format, `R16`, is one the
//! contiguous read path cannot decode.

use candle::Result;

use super::backing::ChunkedKvBacking;
use super::types::BlockTableMutation;
use crate::kv_cache::{ArenaFormatTag, ResolvedArenaInfo};

/// One band's device address and storage format tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BandAddr {
    pub ptr: u64,
    pub fmt: u8,
}

/// One chunk's token window and its bands, each side indexed
/// `head * n_palette + palette`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockBands {
    pub offset: u16,
    pub usage: u32,
    pub k: Vec<BandAddr>,
    pub v: Vec<BandAddr>,
}

/// Where a sequence's writes go, as the host holds it: the writer boundary
/// (chunks below it are shared and never written), the chunk the next write
/// lands in, and how many chunks the block table holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WriterIndices {
    pub start: usize,
    pub writer: usize,
    pub chunks: usize,
}

/// The address `serialize_kv_heads` gives a gid at `(arena_idx, chunk_idx)`:
/// 0 when the arena has no entry, which is where the empty gid lands.
fn band_ptr(info: &[ResolvedArenaInfo], arena_idx: usize, chunk_idx: usize) -> u64 {
    info.get(arena_idx).map_or(0, |a| {
        a.base_ptr + chunk_idx as u64 * a.chunk_byte_stride as u64
    })
}

impl ChunkedKvBacking {
    /// Every chunk of `batch_idx`'s block table, with each band's address and
    /// format as the decode record for that chunk would name them.
    pub fn band_map(&self, batch_idx: usize) -> Result<Vec<BlockBands>> {
        let info = self.resolve_arena_info()?;
        let nkv = self.n_kv_head();
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let Some(Some(seq)) = state.sequences.get(batch_idx) else {
            candle::bail!("band map: slot {batch_idx} is not allocated");
        };
        Ok(seq
            .chunks_slice()
            .iter()
            .map(|cw| {
                let gids = cw.gids.as_slice();
                // The record's own palette stride (`chunk_n_palette`).
                let np = (gids.len() / (nkv * 2).max(1)).max(1);
                let side = |is_v: usize, fmts: &[u8]| -> Vec<BandAddr> {
                    (0..nkv)
                        .flat_map(|h| (0..np).map(move |p| (h, p)))
                        .map(|(h, p)| BandAddr {
                            ptr: gids
                                .get(h * np * 2 + p * 2 + is_v)
                                .map_or(0, |g| band_ptr(&info, g.arena_idx(), g.chunk_idx())),
                            fmt: fmts
                                .get(h * np + p)
                                .copied()
                                .unwrap_or(ArenaFormatTag::Invalid.as_u8()),
                        })
                        .collect()
                };
                BlockBands {
                    offset: cw.offset,
                    usage: cw.usage,
                    k: side(0, cw.k_fmt.as_slice()),
                    v: side(1, cw.v_fmt.as_slice()),
                }
            })
            .collect())
    }

    /// `batch_idx`'s writer boundary and writer chunk — the two indices a
    /// write and its commit must agree on for the commit to count the slots
    /// the write filled.
    pub fn writer_indices(&self, batch_idx: usize) -> Result<WriterIndices> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let Some(Some(seq)) = state.sequences.get(batch_idx) else {
            candle::bail!("writer indices: slot {batch_idx} is not allocated");
        };
        Ok(WriterIndices {
            start: seq.writer_start_idx(),
            writer: seq.decode_write_chunk_idx(),
            chunks: seq.chunks_slice().len(),
        })
    }

    /// `batch_idx`'s most recent block-table mutations, oldest first — what
    /// last reshaped this layer's table, for a report that has found it wrong.
    pub fn block_table_mutations(&self, batch_idx: usize) -> Result<Vec<BlockTableMutation>> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let Some(Some(seq)) = state.sequences.get(batch_idx) else {
            candle::bail!("block table mutations: slot {batch_idx} is not allocated");
        };
        Ok(seq.mutations())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod writer_tests {
    use candle::{DType, Device};

    use super::WriterIndices;
    use crate::kv_cache::chunked::gpu_test_lock::gpu_serial;
    use crate::kv_cache::ChunkedKvBacking;

    /// With nothing shared the boundary is chunk 0, and the writer is the
    /// first chunk with room: 40 tokens fill chunk 0 and put 8 in chunk 1.
    #[test]
    fn the_writer_is_the_first_chunk_with_room_past_the_boundary() {
        let _gpu = gpu_serial();
        let dev = Device::new_cuda(0).unwrap();
        let backing = ChunkedKvBacking::new(1, 2, 32, DType::F16, &dev, 256).unwrap();
        let seq = backing.alloc_sequence().unwrap();
        backing.ensure_for_offset(seq, 0, 40).unwrap();
        backing.set_len(seq, 40);
        assert_eq!(
            backing.writer_indices(seq).unwrap(),
            WriterIndices {
                start: 0,
                writer: 1,
                chunks: 2
            }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::band_ptr;
    use crate::kv_cache::ResolvedArenaInfo;

    #[test]
    fn a_band_address_is_arena_base_plus_slot_stride() {
        let info = vec![
            ResolvedArenaInfo {
                base_ptr: 0x1000,
                chunk_byte_stride: 256,
                chunk_capacity: 64,
            },
            ResolvedArenaInfo {
                base_ptr: 0x9000,
                chunk_byte_stride: 128,
                chunk_capacity: 64,
            },
        ];
        assert_eq!(band_ptr(&info, 0, 1), 0x1100);
        assert_eq!(band_ptr(&info, 1, 2), 0x9100);
        // The empty gid's arena index is far past the table.
        assert_eq!(band_ptr(&info, usize::MAX / 1024, 3), 0);
    }
}
