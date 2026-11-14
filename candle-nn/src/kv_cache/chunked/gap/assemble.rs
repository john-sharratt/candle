//! A converted block, put back together on the host from its summary rows
//! (`docs/kv_quantize_in_the_gap.md` §3.5): its slots out of the lease, in the
//! order its gid grid holds them, and the palette maps, outer scales and format
//! tags its chunk carries — laid out exactly as today's seal lays them out, so
//! the chunk a gap installs is the chunk a seal would have built.

use std::vec::IntoIter;

use candle::Result;

use super::demand::PerClass;
use super::lease::Lease;
use super::summary::{HeadRow, BANDS_PER_HEAD};
use crate::kv_cache::arena_table::N_PALETTE;

/// What a block's chunk carries besides its gids, in its own layouts:
/// `n_kv_head × (head_dim / 4)` packed map bytes per side, `n_kv_head ×
/// N_PALETTE` outer scales per side (empty under that side's uniform
/// override: unity), and `n_kv_head × N_PALETTE` band format tags per side.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BlockState {
    pub k_pal: Vec<u8>,
    pub v_pal: Vec<u8>,
    pub k_scale: Vec<f32>,
    pub v_scale: Vec<f32>,
    pub k_fmt: Vec<u8>,
    pub v_fmt: Vec<u8>,
}

/// Each band of `rows` as `(class, lease index)`, in grid order: head, then
/// palette, K before V — `head * BANDS_PER_HEAD + palette * 2 + is_v`, the
/// order a block's gids are held in.
fn grid_bands(
    rows: &[HeadRow],
    class_of_fmt: &impl Fn(u8) -> Option<usize>,
) -> Result<Vec<(usize, u32)>> {
    let mut out = Vec::with_capacity(rows.len() * BANDS_PER_HEAD);
    for row in rows {
        for b in 0..BANDS_PER_HEAD {
            let (p, is_v) = (b / 2, b % 2 == 1);
            let fmt = if is_v { row.v_fmt[p] } else { row.k_fmt[p] };
            let class = class_of_fmt(fmt).ok_or_else(|| {
                candle::Error::Msg(format!("gap install: record format {fmt} has no class"))
            })?;
            out.push((class, row.lease_idx[b]));
        }
    }
    Ok(out)
}

/// A converted block's slots, taken out of `lease` by the indices its rows
/// name, in grid order. One take per class the block reaches, so a block whose
/// row names a slot already taken — or past the lease — takes nothing of that
/// class and refuses.
pub(crate) fn take_block<T>(
    lease: &mut Lease<T>,
    rows: &[HeadRow],
    class_of_fmt: impl Fn(u8) -> Option<usize>,
) -> Result<Vec<T>> {
    let bands = grid_bands(rows, &class_of_fmt)?;
    let mut by_class: PerClass<Vec<u32>> = std::array::from_fn(|_| Vec::new());
    for &(class, idx) in &bands {
        by_class
            .get_mut(class)
            .ok_or_else(|| candle::Error::Msg(format!("gap install: no size class {class}")))?
            .push(idx);
    }
    let mut taken: PerClass<IntoIter<T>> = std::array::from_fn(|_| Vec::new().into_iter());
    for (class, idx) in by_class.iter().enumerate() {
        if !idx.is_empty() {
            taken[class] = lease.take(class, idx)?.into_iter();
        }
    }
    Ok(bands
        .iter()
        .map(|&(class, _)| taken[class].next().expect("one slot per band of its class"))
        .collect())
}

/// The chunk state of a converted block from its rows, one per KV head.
/// `k_override` / `v_override`: whether that side sealed under a uniform
/// override, whose maps are the identity (as the claim wrote them) and whose
/// scales are unity, carried as an empty list as today's seal does.
/// `tag_of_fmt` turns a record format byte into the band tag the chunk holds.
pub(crate) fn block_state(
    rows: &[HeadRow],
    head_dim: usize,
    k_override: bool,
    v_override: bool,
    tag_of_fmt: impl Fn(u8) -> Option<u8>,
) -> Result<BlockState> {
    let pal_bytes = head_dim / 4;
    let n = rows.len();
    let mut s = BlockState {
        k_pal: Vec::with_capacity(n * pal_bytes),
        v_pal: Vec::with_capacity(n * pal_bytes),
        k_scale: Vec::with_capacity(if k_override { 0 } else { n * N_PALETTE }),
        v_scale: Vec::with_capacity(if v_override { 0 } else { n * N_PALETTE }),
        k_fmt: Vec::with_capacity(n * N_PALETTE),
        v_fmt: Vec::with_capacity(n * N_PALETTE),
    };
    let tag = |fmt: u8| {
        tag_of_fmt(fmt).ok_or_else(|| {
            candle::Error::Msg(format!("gap install: record format {fmt} has no band tag"))
        })
    };
    for row in rows {
        if row.k_pal.len() != pal_bytes || row.v_pal.len() != pal_bytes {
            candle::bail!(
                "gap install: a row's maps are {}/{} bytes, not {pal_bytes}",
                row.k_pal.len(),
                row.v_pal.len()
            );
        }
        s.k_pal.extend_from_slice(&row.k_pal);
        s.v_pal.extend_from_slice(&row.v_pal);
        if !k_override {
            s.k_scale.extend_from_slice(&row.k_scale);
        }
        if !v_override {
            s.v_scale.extend_from_slice(&row.v_scale);
        }
        for p in 0..N_PALETTE {
            s.k_fmt.push(tag(row.k_fmt[p])?);
            s.v_fmt.push(tag(row.v_fmt[p])?);
        }
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::super::super::size_class::SizeClass;
    use super::*;

    /// A row whose bands are K in class `k` and V in class `v`, at lease
    /// indices `idx`, with recognisable maps and scales.
    fn row(h: u8, k_fmt: u8, v_fmt: u8, idx: [u32; 8]) -> HeadRow {
        HeadRow {
            k_pal: vec![0x10 + h; 32],
            v_pal: vec![0x20 + h; 32],
            k_fmt: [k_fmt; 4],
            v_fmt: [v_fmt; 4],
            k_scale: [1.0 + f32::from(h), 2.0, 3.0, 4.0],
            v_scale: [5.0, 6.0, 7.0, 8.0 + f32::from(h)],
            k_ptr: [0; 4],
            v_ptr: [0; 4],
            lease_idx: idx,
        }
    }

    /// Format bytes 7 → class 1, 15 → class 3.
    fn class(f: u8) -> Option<usize> {
        match f {
            7 => Some(1),
            15 => Some(3),
            _ => None,
        }
    }

    fn lease(runs: &[(usize, &[u64])]) -> Lease<u64> {
        let mut l = Lease::default();
        for &(c, slots) in runs {
            l.add(c, slots.iter().map(|&s| (s, s)));
        }
        l
    }

    #[test]
    fn a_blocks_slots_come_back_in_grid_order() {
        // Head 0: K bands class 1 at 0..4, V bands class 3 at 0..4.
        // Head 1: K bands class 1 at 4..8, V bands class 3 at 4..8.
        let rows = [
            row(0, 7, 15, [0, 0, 1, 1, 2, 2, 3, 3]),
            row(1, 7, 15, [4, 4, 5, 5, 6, 6, 7, 7]),
        ];
        let mut l = lease(&[
            (1, &[100, 101, 102, 103, 104, 105, 106, 107]),
            (3, &[300, 301, 302, 303, 304, 305, 306, 307]),
        ]);
        let gids = take_block(&mut l, &rows, class).unwrap();
        // Grid order: h0 (p0 K, p0 V, p1 K, …), then h1.
        assert_eq!(
            gids,
            vec![100, 300, 101, 301, 102, 302, 103, 303, 104, 304, 105, 305, 106, 306, 107, 307]
        );
        assert_eq!(l.held(), [0; SizeClass::COUNT]);
    }

    #[test]
    fn blocks_taken_one_after_another_each_get_their_own_slots() {
        let a = [row(0, 7, 7, [0, 1, 2, 3, 4, 5, 6, 7])];
        let b = [row(0, 7, 7, [8, 9, 10, 11, 12, 13, 14, 15])];
        let slots: Vec<u64> = (0..16).collect();
        let mut l = lease(&[(1, &slots)]);
        assert_eq!(
            take_block(&mut l, &b, class).unwrap(),
            (8..16).collect::<Vec<_>>()
        );
        assert_eq!(
            take_block(&mut l, &a, class).unwrap(),
            (0..8).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_block_naming_a_slot_another_took_takes_nothing_of_that_class() {
        let a = [row(0, 7, 15, [0, 0, 1, 1, 2, 2, 3, 3])];
        let mut l = lease(&[(1, &[1, 2, 3, 4]), (3, &[5, 6, 7, 8])]);
        take_block(&mut l, &a, class).unwrap();
        assert!(take_block(&mut l, &a, class).is_err());
    }

    #[test]
    fn a_band_whose_format_has_no_class_refuses_the_block() {
        let a = [row(0, 7, 99, [0; 8])];
        let mut l = lease(&[(1, &[1, 2, 3, 4])]);
        assert!(take_block(&mut l, &a, class).is_err());
        assert_eq!(l.held()[1], 4, "nothing taken");
    }

    #[test]
    fn the_chunk_state_is_the_rows_laid_out_head_by_head() {
        let rows = [row(0, 7, 15, [0; 8]), row(1, 15, 7, [0; 8])];
        let tag = |f: u8| Some(f + 100);
        let s = block_state(&rows, 128, false, false, tag).unwrap();
        assert_eq!(s.k_pal, [vec![0x10; 32], vec![0x11; 32]].concat());
        assert_eq!(s.v_pal, [vec![0x20; 32], vec![0x21; 32]].concat());
        assert_eq!(s.k_scale, vec![1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 3.0, 4.0]);
        assert_eq!(s.v_scale, vec![5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 9.0]);
        assert_eq!(s.k_fmt, vec![107, 107, 107, 107, 115, 115, 115, 115]);
        assert_eq!(s.v_fmt, vec![115, 115, 115, 115, 107, 107, 107, 107]);
    }

    #[test]
    fn an_overridden_side_carries_unity_as_an_empty_scale_list() {
        let rows = [row(0, 7, 15, [0; 8])];
        let s = block_state(&rows, 128, true, false, Some).unwrap();
        assert!(s.k_scale.is_empty());
        assert_eq!(s.v_scale.len(), 4);
        let s = block_state(&rows, 128, false, true, Some).unwrap();
        assert!(s.v_scale.is_empty());
        assert_eq!(s.k_scale.len(), 4);
    }

    #[test]
    fn a_row_of_the_wrong_width_or_an_unknown_format_is_refused() {
        let rows = [row(0, 7, 15, [0; 8])];
        assert!(block_state(&rows, 256, false, false, Some).is_err());
        assert!(block_state(&rows, 128, false, false, |_| None).is_err());
    }
}
