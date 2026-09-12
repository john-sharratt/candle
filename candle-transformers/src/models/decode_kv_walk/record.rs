//! A chunk's `KvHead[n_kv_head]` record, read back the way the decode kernel
//! reads it, and the band addresses the host's block table says it should name.
//!
//! Per head (`slot_types.cuh`, `NP` palettes):
//! `k_pal[HD/4] v_pal[HD/4] k_ptr[NP]×u64 v_ptr[NP]×u64 k_fmt[NP] v_fmt[NP]
//! k_scale[NP]×f32 v_scale[NP]×f32` — `HD/2 + NP·26` bytes.

use candle::Result;
use candle_nn::kv_cache::ResolvedArenaInfo;

/// The pointer, format and scale block of one head's record.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HeadRecord {
    pub k_ptr: Vec<u64>,
    pub v_ptr: Vec<u64>,
    pub k_fmt: Vec<u8>,
    pub v_fmt: Vec<u8>,
    pub k_scale: Vec<f32>,
    pub v_scale: Vec<f32>,
}

/// Bytes of one head's record (`kv_head_byte_size<HD, NP>`).
pub(crate) fn head_record_bytes(head_dim: usize, n_palette: usize) -> usize {
    head_dim / 2 + n_palette * 26
}

/// Parse a whole chunk record of `n_kv_head` heads.
pub(crate) fn parse_record(
    bytes: &[u8],
    n_kv_head: usize,
    head_dim: usize,
    n_palette: usize,
) -> Result<Vec<HeadRecord>> {
    let per_head = head_record_bytes(head_dim, n_palette);
    if bytes.len() != per_head * n_kv_head {
        candle::bail!(
            "kv record: {} bytes, want {n_kv_head} heads × {per_head}",
            bytes.len()
        );
    }
    let np = n_palette;
    Ok(bytes
        .chunks_exact(per_head)
        .map(|h| {
            let base = head_dim / 2;
            let u64s = |at: usize| -> Vec<u64> {
                (0..np)
                    .map(|p| {
                        let o = at + p * 8;
                        u64::from_le_bytes(h[o..o + 8].try_into().expect("8 bytes"))
                    })
                    .collect()
            };
            let f32s = |at: usize| -> Vec<f32> {
                (0..np)
                    .map(|p| {
                        let o = at + p * 4;
                        f32::from_le_bytes(h[o..o + 4].try_into().expect("4 bytes"))
                    })
                    .collect()
            };
            HeadRecord {
                k_ptr: u64s(base),
                v_ptr: u64s(base + np * 8),
                k_fmt: h[base + np * 16..base + np * 17].to_vec(),
                v_fmt: h[base + np * 17..base + np * 18].to_vec(),
                k_scale: f32s(base + np * 18),
                v_scale: f32s(base + np * 22),
            }
        })
        .collect())
}

/// The band address the host's block table implies for a gid at
/// `(arena_idx, chunk_idx)` under `info` — computed exactly as
/// `meta_pool::serialize_kv_heads` computes it, so a disagreement with the
/// device record is a disagreement about *where the band is*, not about how the
/// address is derived. An arena index with no entry (the empty gid's is out of
/// range) serialises as 0, and so does it here.
pub(crate) fn expected_ptr(info: &[ResolvedArenaInfo], arena_idx: usize, chunk_idx: usize) -> u64 {
    info.get(arena_idx)
        .map_or(0, |a| a.base_ptr + chunk_idx as u64 * a.chunk_byte_stride as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The golden record `serialize_kv_heads` writes for head_dim 4, one head,
    /// four palettes: identity maps, K at arena 0 chunk 1 (0x1100), V at
    /// 0x2000.., K tag Q8_0 (7), V tag F16 (1), K scales 1.0, V scale 0.5.
    fn golden() -> Vec<u8> {
        let mut b = vec![0xE4u8, 0xE4];
        for p in 0..4u64 {
            b.extend_from_slice(&(0x1100 + p * 0x100).to_le_bytes());
        }
        for p in 0..4u64 {
            b.extend_from_slice(&(0x2000 + p * 0x40).to_le_bytes());
        }
        b.extend_from_slice(&[7, 7, 7, 7]);
        b.extend_from_slice(&[1, 1, 1, 1]);
        for _ in 0..4 {
            b.extend_from_slice(&1.0f32.to_le_bytes());
        }
        for _ in 0..4 {
            b.extend_from_slice(&0.5f32.to_le_bytes());
        }
        b
    }

    #[test]
    fn a_head_record_is_hd_half_plus_26_per_palette() {
        assert_eq!(head_record_bytes(4, 4), 106);
        assert_eq!(head_record_bytes(256, 4), 232);
    }

    #[test]
    fn the_golden_record_reads_back_field_for_field() {
        let heads = parse_record(&golden(), 1, 4, 4).unwrap();
        assert_eq!(
            heads,
            vec![HeadRecord {
                k_ptr: vec![0x1100, 0x1200, 0x1300, 0x1400],
                v_ptr: vec![0x2000, 0x2040, 0x2080, 0x20c0],
                k_fmt: vec![7; 4],
                v_fmt: vec![1; 4],
                k_scale: vec![1.0; 4],
                v_scale: vec![0.5; 4],
            }]
        );
    }

    #[test]
    fn heads_are_read_at_their_own_stride() {
        let mut two = golden();
        let mut second = golden();
        // Second head: K pointer 0 differs.
        second[2..10].copy_from_slice(&0xdead_0000u64.to_le_bytes());
        two.extend_from_slice(&second);
        let heads = parse_record(&two, 2, 4, 4).unwrap();
        assert_eq!(heads[0].k_ptr[0], 0x1100);
        assert_eq!(heads[1].k_ptr[0], 0xdead_0000);
    }

    #[test]
    fn a_record_of_the_wrong_length_is_refused() {
        assert!(parse_record(&golden()[..105], 1, 4, 4).is_err());
    }

    #[test]
    fn the_expected_address_mirrors_the_serialiser() {
        let info = vec![
            ResolvedArenaInfo { base_ptr: 0x1000, chunk_byte_stride: 256, chunk_capacity: 64 },
            ResolvedArenaInfo { base_ptr: 0x9000, chunk_byte_stride: 128, chunk_capacity: 64 },
        ];
        assert_eq!(expected_ptr(&info, 0, 1), 0x1100);
        assert_eq!(expected_ptr(&info, 1, 2), 0x9100);
        // The empty gid (-1) lands on an arena index far past the table.
        assert_eq!(expected_ptr(&info, usize::MAX / 1024, 3), 0);
    }
}
