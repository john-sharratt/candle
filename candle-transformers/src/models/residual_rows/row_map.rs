//! Where a residual's rows physically sit, and the copies between two layouts.
//!
//! A forward packs its rows in **internal** order
//! `[decode | single-token prefills | multi-token prefills | glue]`: a
//! one-token prefill is a decode and rides the decode group. Callers name rows
//! in **caller** order `[decode | prefill (caller order) | glue]`. The two differ
//! only by where the single-token prefills sit, so the mapping between them is a
//! handful of contiguous runs — each multi-token prefill moves as one block.
//!
//! Every copy into or out of a residual — a segment's input assembled from the
//! last segment's rows and a creep's, a creep's rows written to its arena, a
//! member's rows taken out — moves caller rows from one layout to another. It is
//! planned here, by composing the two layouts' runs into
//! `(source row, destination row, length)` copies, so no buffer is ever reordered
//! in place and nothing is built to reorder through (`docs/wave_step.md` §4).

use candle::Result;

/// `len` rows: caller rows `caller..caller + len` sit at physical rows
/// `phys..phys + len`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    pub caller: usize,
    pub phys: usize,
    pub len: usize,
}

/// A residual's layout: caller row → physical row, as runs that cover
/// `[0, rows)` in ascending caller order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowMap {
    runs: Vec<Run>,
}

/// One planned copy: `len` rows from physical row `src` of the source onto
/// physical row `dst` of the destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowCopy {
    pub src: usize,
    pub dst: usize,
    pub len: usize,
}

impl RowMap {
    /// `rows` rows laid out in caller order.
    pub fn identity(rows: usize) -> Self {
        let mut map = Self { runs: Vec::new() };
        map.push(0, 0, rows);
        map
    }

    /// A forward's internal layout for `n_decode` decode rows, prefills of
    /// `pre_lens` tokens each (caller order) and `glue_rows` glue rows.
    pub fn wave(n_decode: usize, pre_lens: &[usize], glue_rows: usize) -> Self {
        let singles = pre_lens.iter().filter(|&&l| l == 1).count();
        let mut map = Self { runs: Vec::new() };
        map.push(0, 0, n_decode);
        let mut caller = n_decode;
        let mut next_single = n_decode;
        let mut next_multi = n_decode + singles;
        for &l in pre_lens {
            if l == 1 {
                map.push(caller, next_single, 1);
                next_single += 1;
            } else {
                map.push(caller, next_multi, l);
                next_multi += l;
            }
            caller += l;
        }
        map.push(caller, next_multi, glue_rows);
        map
    }

    /// Rows the layout covers.
    pub fn rows(&self) -> usize {
        self.runs.last().map_or(0, |r| r.caller + r.len)
    }

    /// Whether every caller row sits at its own index.
    pub fn is_identity(&self) -> bool {
        self.runs.iter().all(|r| r.caller == r.phys)
    }

    /// The runs, ascending in caller order.
    pub fn runs(&self) -> &[Run] {
        &self.runs
    }

    /// Append `len` rows, merged into the last run where both orders continue it.
    fn push(&mut self, caller: usize, phys: usize, len: usize) {
        if len == 0 {
            return;
        }
        if let Some(last) = self.runs.last_mut() {
            if last.caller + last.len == caller && last.phys + last.len == phys {
                last.len += len;
                return;
            }
        }
        self.runs.push(Run { caller, phys, len });
    }

    /// The run holding caller row `row`, by index, and how many rows of it
    /// remain from `row` on.
    fn locate(&self, row: usize) -> Option<(usize, usize)> {
        let i = self.runs.partition_point(|r| r.caller + r.len <= row);
        let r = self.runs.get(i)?;
        (r.caller <= row).then(|| (r.phys + (row - r.caller), r.caller + r.len - row))
    }
}

/// The copies that move caller rows `src_start..src_start + len` of a residual
/// laid out by `src` onto caller rows `dst_start..dst_start + len` of one laid
/// out by `dst`.
///
/// Neighbouring copies that continue each other in both layouts are merged, so
/// the plan is one copy whenever the rows are contiguous on both sides. Refuses a
/// range either layout does not cover — a copy planned past a layout's end would
/// read or write another tenant's rows.
pub fn plan_copies(
    src: &RowMap,
    src_start: usize,
    dst: &RowMap,
    dst_start: usize,
    len: usize,
) -> Result<Vec<RowCopy>> {
    if src_start + len > src.rows() || dst_start + len > dst.rows() {
        candle::bail!(
            "row copy: {len} rows from {src_start} of a {}-row layout onto {dst_start} of a \
             {}-row layout",
            src.rows(),
            dst.rows()
        );
    }
    let mut out: Vec<RowCopy> = Vec::new();
    let mut k = 0;
    while k < len {
        let (s, s_left) = src.locate(src_start + k).expect("in range, checked above");
        let (d, d_left) = dst.locate(dst_start + k).expect("in range, checked above");
        let n = s_left.min(d_left).min(len - k);
        match out.last_mut() {
            Some(c) if c.src + c.len == s && c.dst + c.len == d => c.len += n,
            _ => out.push(RowCopy {
                src: s,
                dst: d,
                len: n,
            }),
        }
        k += n;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(caller: usize, phys: usize, len: usize) -> Run {
        Run { caller, phys, len }
    }

    fn copy(src: usize, dst: usize, len: usize) -> RowCopy {
        RowCopy { src, dst, len }
    }

    /// Two decode rows, prefills of 1, 3, 1 and 2 tokens, two glue rows: the
    /// one-token prefills move up beside the decode rows.
    #[test]
    fn single_prefills_move_up_beside_the_decode_rows() {
        let map = RowMap::wave(2, &[1, 3, 1, 2], 2);
        assert_eq!(
            map.runs(),
            &[run(0, 0, 3), run(3, 4, 3), run(6, 3, 1), run(7, 7, 4)]
        );
        assert_eq!(map.rows(), 11);
        assert!(!map.is_identity());
    }

    #[test]
    fn with_no_single_prefill_the_layout_is_the_identity() {
        let map = RowMap::wave(3, &[4, 2], 1);
        assert_eq!(map, RowMap::identity(10));
        assert!(map.is_identity());
    }

    #[test]
    fn an_empty_layout_has_no_rows() {
        assert_eq!(RowMap::identity(0).rows(), 0);
        assert_eq!(RowMap::wave(0, &[], 0).rows(), 0);
    }

    /// Between two identity layouts a range is one copy.
    #[test]
    fn identity_to_identity_is_one_copy() {
        let a = RowMap::identity(10);
        let b = RowMap::identity(20);
        assert_eq!(plan_copies(&a, 2, &b, 7, 5).unwrap(), vec![copy(2, 7, 5)]);
    }

    /// Every caller row of an internal layout lands at its own caller index, and
    /// back: the two plans are inverses, row for row.
    #[test]
    fn internal_to_caller_and_back_round_trips() {
        let internal = RowMap::wave(2, &[1, 3, 1, 2], 2);
        let caller = RowMap::identity(11);
        let out = plan_copies(&internal, 0, &caller, 0, 11).unwrap();
        assert_eq!(
            out,
            vec![copy(0, 0, 3), copy(4, 3, 3), copy(3, 6, 1), copy(7, 7, 4)]
        );
        let back = plan_copies(&caller, 0, &internal, 0, 11).unwrap();
        assert_eq!(
            back,
            vec![copy(0, 0, 3), copy(3, 4, 3), copy(6, 3, 1), copy(7, 7, 4)]
        );
        // Applied to the row indices themselves, each plan undoes the other.
        let mut phys = [usize::MAX; 11];
        for c in &back {
            for i in 0..c.len {
                phys[c.dst + i] = c.src + i;
            }
        }
        let mut again = [usize::MAX; 11];
        for c in &out {
            for i in 0..c.len {
                again[c.dst + i] = phys[c.src + i];
            }
        }
        assert_eq!(again, core::array::from_fn(|i| i));
    }

    /// A sub-range is split exactly where either layout's run boundary falls.
    #[test]
    fn a_sub_range_splits_at_both_layouts_boundaries() {
        let src = RowMap::wave(2, &[1, 3, 1, 2], 2);
        let dst = RowMap::wave(1, &[2, 1], 0);
        // Caller rows 2..6 of `src` onto caller rows 0..4 of `dst`.
        assert_eq!(
            plan_copies(&src, 2, &dst, 0, 4).unwrap(),
            vec![copy(2, 0, 1), copy(4, 2, 2), copy(6, 1, 1)]
        );
    }

    #[test]
    fn a_range_past_either_layout_is_refused() {
        let a = RowMap::identity(4);
        assert!(plan_copies(&a, 2, &a, 0, 3).is_err());
        assert!(plan_copies(&a, 0, &a, 2, 3).is_err());
        assert!(plan_copies(&a, 0, &a, 0, 4).is_ok());
    }

    #[test]
    fn a_zero_length_range_plans_nothing() {
        let a = RowMap::identity(4);
        assert!(plan_copies(&a, 4, &a, 4, 0).unwrap().is_empty());
    }
}
