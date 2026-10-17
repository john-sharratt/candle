//! A forward's resumed residual, and caller rows moved between residuals.
//!
//! A forward that resumes works in a residual of its own, on its forward span,
//! in its internal layout. What it resumes from is a run of **parts** in caller
//! order — the last segment's full-sweep rows, a creep's committed rows — each
//! held wherever it is held. [`ResidualIn::gather_into`] copies the parts into
//! the forward's buffer through the composed layouts, so the forward never
//! writes what it read and a failed forward leaves its inputs as they were.

use candle::{Result, Tensor};

use super::row_map::{plan_copies, RowMap};
use super::store::{copy_rows, RowStore, SpanRows};

/// `len` caller rows from row `start` of a residual held in `store`, laid out by
/// `layout`.
pub struct Part<'a> {
    pub store: &'a dyn RowStore,
    pub layout: RowMap,
    pub start: usize,
    pub len: usize,
}

/// A forward's input residual: parts concatenated in caller order, gathered
/// into the forward's own buffer in the forward's layout (`target`).
pub struct ResidualIn<'a> {
    parts: Vec<Part<'a>>,
    target: RowMap,
    width: usize,
}

impl<'a> ResidualIn<'a> {
    /// Refuses parts that do not add up to the target's rows, or that differ in
    /// row width or type — either would land rows at the wrong offsets.
    pub fn new(parts: Vec<Part<'a>>, target: RowMap) -> Result<Self> {
        let rows: usize = parts.iter().map(|p| p.len).sum();
        if rows != target.rows() {
            candle::bail!(
                "residual in: {rows} rows in parts for a {}-row forward",
                target.rows()
            );
        }
        let Some(first) = parts.first() else {
            candle::bail!("residual in: no parts");
        };
        let (width, dtype) = (first.store.width(), first.store.dtype());
        if parts
            .iter()
            .any(|p| p.store.width() != width || p.store.dtype() != dtype)
        {
            candle::bail!("residual in: parts of different row widths or types");
        }
        Ok(Self {
            parts,
            target,
            width,
        })
    }

    /// Rows the forward resumes.
    pub fn rows(&self) -> usize {
        self.target.rows()
    }

    /// Elements in one row.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Copy every part into `dst` — the forward's residual, any shape that is
    /// whole rows — at its rows in the forward's layout.
    pub fn gather_into(&self, dst: &Tensor) -> Result<()> {
        let rows = SpanRows::new(dst, self.width)?;
        if rows.capacity() != self.rows() {
            candle::bail!(
                "residual in: {} rows gathered into a {}-row buffer",
                self.rows(),
                rows.capacity()
            );
        }
        let mut at = 0;
        for p in &self.parts {
            let plan = plan_copies(&p.layout, p.start, &self.target, at, p.len)?;
            copy_rows(p.store, &rows, &plan)?;
            at += p.len;
        }
        Ok(())
    }
}

/// Move caller rows `src_start..src_start + len` of `src` (laid out by
/// `src_layout`) onto caller rows `dst_start..` of `dst` (laid out by
/// `dst_layout`).
pub fn move_rows(
    src: &dyn RowStore,
    src_layout: &RowMap,
    src_start: usize,
    dst: &dyn RowStore,
    dst_layout: &RowMap,
    dst_start: usize,
    len: usize,
) -> Result<()> {
    let plan = plan_copies(src_layout, src_start, dst_layout, dst_start, len)?;
    copy_rows(src, dst, &plan)
}

/// The previous segment of a step: its working residual and its layout.
pub struct HandedOn<'a> {
    pub store: &'a dyn RowStore,
    pub layout: &'a RowMap,
    /// The creep rode it, so its caller rows hold the creep between head and glue.
    pub carried_creep: bool,
}

/// A step segment's input parts, in its caller order `[head | creep? | glue]`
/// (`docs/wave_step.md` §2.2).
///
/// The first segment resumes from `creep` alone, or from nothing. Every later
/// one takes the head rows (`decode` + `verify`) from the front of the previous
/// segment's residual, the creep's committed rows when the creep rides this
/// segment, and the glue rows from behind the creep's in the previous
/// segment's caller order — or straight after the head when the creep did not
/// ride it. Empty parts are dropped.
pub fn segment_parts<'a>(
    prev: Option<HandedOn<'a>>,
    creep: Option<Part<'a>>,
    head: usize,
    creep_rows: usize,
    glue_rows: usize,
) -> Option<Vec<Part<'a>>> {
    let Some(prev) = prev else {
        return creep.map(|p| vec![p]);
    };
    let held = if prev.carried_creep { creep_rows } else { 0 };
    let mut parts = vec![Part {
        store: prev.store,
        layout: prev.layout.clone(),
        start: 0,
        len: head,
    }];
    parts.extend(creep);
    parts.push(Part {
        store: prev.store,
        layout: prev.layout.clone(),
        start: head + held,
        len: glue_rows,
    });
    parts.retain(|p| p.len > 0);
    Some(parts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    fn numbered(rows: usize, base: f32) -> Tensor {
        let v: Vec<f32> = (0..rows)
            .flat_map(|r| [base + r as f32, base + r as f32])
            .collect();
        Tensor::from_vec(v, (rows, 2), &Device::Cpu).unwrap()
    }

    fn firsts(t: &Tensor) -> Vec<f32> {
        t.to_vec2::<f32>().unwrap().iter().map(|r| r[0]).collect()
    }

    /// A segment's input assembled from the last segment's `[decode | glue]`
    /// rows and a creep's rows, landing in the forward's internal layout:
    /// caller order `[decode 2 | creep 3, 1 | glue 1]` moves the one-token creep
    /// member up beside the decode rows, ahead of the three-token one.
    #[test]
    fn parts_land_at_their_rows_in_the_forward_layout() {
        // Last segment: decode rows 0, 1 and one glue row 2 — identity layout.
        let last = SpanRows::new(&numbered(3, 0.), 2).unwrap();
        // The creep: a three-token member then a one-token one.
        let creep = SpanRows::new(&numbered(4, 10.), 2).unwrap();
        let parts = vec![
            Part {
                store: &last,
                layout: RowMap::identity(3),
                start: 0,
                len: 2,
            },
            Part {
                store: &creep,
                layout: RowMap::identity(4),
                start: 0,
                len: 4,
            },
            Part {
                store: &last,
                layout: RowMap::identity(3),
                start: 2,
                len: 1,
            },
        ];
        let target = RowMap::wave(2, &[3, 1], 1);
        assert!(!target.is_identity());
        let input = ResidualIn::new(parts, target).unwrap();
        let dst = Tensor::zeros((1, 7, 2), DType::F32, &Device::Cpu).unwrap();
        input.gather_into(&dst).unwrap();
        // Internal order: decode, the one-token member, the three-token one, glue.
        assert_eq!(
            firsts(&dst.squeeze(0).unwrap()),
            vec![0., 1., 13., 10., 11., 12., 2.]
        );
    }

    #[test]
    fn parts_that_do_not_fill_the_forward_are_refused() {
        let a = SpanRows::new(&numbered(2, 0.), 2).unwrap();
        let parts = vec![Part {
            store: &a,
            layout: RowMap::identity(2),
            start: 0,
            len: 2,
        }];
        assert!(ResidualIn::new(parts, RowMap::identity(3)).is_err());
    }

    fn gathered(parts: Vec<Part>, target: RowMap) -> Vec<f32> {
        let rows = target.rows();
        let input = ResidualIn::new(parts, target).unwrap();
        let dst = Tensor::zeros((rows, 2), DType::F32, &Device::Cpu).unwrap();
        input.gather_into(&dst).unwrap();
        firsts(&dst)
    }

    /// The middle segment of a three-segment step — decode 1, a creep of a
    /// three-token then a one-token member resuming mid-stack, glue 2 — takes
    /// decode and glue from the first segment and the creep from its arena, and
    /// lands them in its own layout: decode, the one-token member, the
    /// three-token one, glue.
    #[test]
    fn the_creeps_segment_takes_its_rows_from_both_sides() {
        // First segment: no creep, so its caller rows are `[decode | glue]`.
        let first = SpanRows::new(&numbered(3, 0.), 2).unwrap();
        let first_layout = RowMap::wave(1, &[], 2);
        // The creep's committed half, caller order.
        let creep = SpanRows::new(&numbered(4, 10.), 2).unwrap();
        let parts = segment_parts(
            Some(HandedOn {
                store: &first,
                layout: &first_layout,
                carried_creep: false,
            }),
            Some(Part {
                store: &creep,
                layout: RowMap::identity(4),
                start: 0,
                len: 4,
            }),
            1,
            4,
            2,
        )
        .unwrap();
        assert_eq!(
            gathered(parts, RowMap::wave(1, &[3, 1], 2)),
            vec![0., 13., 10., 11., 12., 1., 2.]
        );
    }

    /// The segment after the creep's window takes decode from the front of the
    /// creep's segment and glue from behind the creep — in that segment's
    /// caller order, not its physical rows, where the one-token member sits
    /// beside decode.
    #[test]
    fn the_segment_after_the_window_skips_the_creeps_rows() {
        // Physical rows of the creep's segment: decode 20, the one-token member
        // 21, the three-token one 22..25, glue 25, 26.
        let window = SpanRows::new(&numbered(7, 20.), 2).unwrap();
        let window_layout = RowMap::wave(1, &[3, 1], 2);
        let parts = segment_parts(
            Some(HandedOn {
                store: &window,
                layout: &window_layout,
                carried_creep: true,
            }),
            None,
            1,
            4,
            2,
        )
        .unwrap();
        assert_eq!(
            gathered(parts, RowMap::wave(1, &[], 2)),
            vec![20., 25., 26.]
        );
    }

    /// The first segment resumes from the creep alone, or starts fresh.
    #[test]
    fn the_first_segment_resumes_from_the_creep_or_nothing() {
        let creep = SpanRows::new(&numbered(2, 5.), 2).unwrap();
        let alone = segment_parts(
            None,
            Some(Part {
                store: &creep,
                layout: RowMap::identity(2),
                start: 0,
                len: 2,
            }),
            0,
            2,
            0,
        )
        .unwrap();
        assert_eq!(gathered(alone, RowMap::identity(2)), vec![5., 6.]);
        assert!(segment_parts(None, None, 1, 0, 0).is_none());
    }

    /// The creep's rows taken out of a forward's internal layout into caller
    /// order — the write a window makes to its creep's pending half.
    #[test]
    fn rows_move_out_of_a_forward_layout_into_caller_order() {
        // Each physical row holds its own index.
        let forward = SpanRows::new(&numbered(7, 0.), 2).unwrap();
        // Caller `[decode 2 | creep 3, 1 | glue 1]`: the one-token member sits at
        // physical row 2, the three-token one at 3..6.
        let layout = RowMap::wave(2, &[3, 1], 1);
        let out = SpanRows::new(&numbered(4, -1.), 2).unwrap();
        // Caller rows 2..6 are the creep, back in caller order.
        move_rows(&forward, &layout, 2, &out, &RowMap::identity(4), 0, 4).unwrap();
        assert_eq!(firsts(out.tensor()), vec![3., 4., 5., 2.]);
    }
}
