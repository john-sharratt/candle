//! Rows of a residual wherever they are held, and the copies between them.
//!
//! A residual is held in one of two shapes: one contiguous buffer — a forward's
//! working residual on its span, or any tensor viewed `[rows, width]` — or the
//! row blocks of a creep's arena, laid out over runs of span regions that are
//! not adjacent to one another (`super::arena`). A copy between two residuals is
//! planned in rows (`super::row_map`) and executed here, split wherever either
//! side's block ends.

use candle::{DType, Result, Tensor};

use super::row_map::RowCopy;

/// Somewhere a residual's rows are held, physically indexed.
pub trait RowStore {
    /// Elements in one row.
    fn width(&self) -> usize;
    fn dtype(&self) -> DType;
    /// Physical rows held.
    fn capacity(&self) -> usize;
    /// The block holding physical row `row`: the block as `[rows, width]`, the
    /// row's index in it, and the rows left in the block from there on.
    fn block_at(&self, row: usize) -> Result<(&Tensor, usize, usize)>;
}

/// A residual held in one contiguous buffer, viewed `[rows, width]`.
pub struct SpanRows {
    rows: Tensor,
}

impl SpanRows {
    /// View `t` — any shape whose trailing elements are whole rows of `width` —
    /// as `[rows, width]`. A view: nothing is copied.
    pub fn new(t: &Tensor, width: usize) -> Result<Self> {
        let elems = t.elem_count();
        if width == 0 || !elems.is_multiple_of(width) {
            candle::bail!(
                "residual rows: a {:?} tensor is not whole rows of {width}",
                t.dims()
            );
        }
        Ok(Self {
            rows: t.reshape((elems / width, width))?,
        })
    }

    /// The buffer, as `[rows, width]`.
    pub fn tensor(&self) -> &Tensor {
        &self.rows
    }
}

impl RowStore for SpanRows {
    fn width(&self) -> usize {
        self.rows.dims()[1]
    }

    fn dtype(&self) -> DType {
        self.rows.dtype()
    }

    fn capacity(&self) -> usize {
        self.rows.dims()[0]
    }

    fn block_at(&self, row: usize) -> Result<(&Tensor, usize, usize)> {
        let rows = self.capacity();
        if row >= rows {
            candle::bail!("residual rows: row {row} of a {rows}-row buffer");
        }
        Ok((&self.rows, row, rows - row))
    }
}

/// Execute `plan` from `src` onto `dst`: one copy per stretch that is
/// contiguous on both sides.
///
/// Refuses two stores whose rows differ in width or type — a copy between them
/// would land every row at the wrong offset, and nothing downstream could tell.
pub fn copy_rows(src: &dyn RowStore, dst: &dyn RowStore, plan: &[RowCopy]) -> Result<()> {
    if src.width() != dst.width() || src.dtype() != dst.dtype() {
        candle::bail!(
            "residual rows: copying {} × {:?} rows onto {} × {:?} rows",
            src.width(),
            src.dtype(),
            dst.width(),
            dst.dtype()
        );
    }
    for c in plan {
        let mut k = 0;
        while k < c.len {
            let (sb, so, s_left) = src.block_at(c.src + k)?;
            let (db, d_off, d_left) = dst.block_at(c.dst + k)?;
            let n = s_left.min(d_left).min(c.len - k);
            db.slice_set(&sb.narrow(0, so, n)?, 0, d_off)?;
            k += n;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    /// `rows` rows of `width`, row `r` holding `base + r` in every element.
    fn numbered(rows: usize, width: usize, base: f32) -> Tensor {
        let v: Vec<f32> = (0..rows)
            .flat_map(|r| std::iter::repeat_n(base + r as f32, width))
            .collect();
        Tensor::from_vec(v, (rows, width), &Device::Cpu).unwrap()
    }

    fn row_ids(t: &Tensor) -> Vec<f32> {
        t.to_vec2::<f32>().unwrap().iter().map(|r| r[0]).collect()
    }

    #[test]
    fn a_planned_copy_moves_exactly_its_rows() {
        let src = SpanRows::new(&numbered(6, 3, 10.), 3).unwrap();
        let dst = SpanRows::new(&numbered(5, 3, 0.), 3).unwrap();
        let plan = [
            RowCopy {
                src: 4,
                dst: 0,
                len: 2,
            },
            RowCopy {
                src: 0,
                dst: 3,
                len: 1,
            },
        ];
        copy_rows(&src, &dst, &plan).unwrap();
        assert_eq!(row_ids(dst.tensor()), vec![14., 15., 2., 10., 4.]);
    }

    /// A 3-D residual is viewed as its rows; the copy writes through the view
    /// into the tensor it was made from.
    #[test]
    fn a_view_writes_through_to_its_tensor() {
        let buf = numbered(4, 2, 0.).reshape((1, 4, 2)).unwrap();
        let dst = SpanRows::new(&buf, 2).unwrap();
        let src = SpanRows::new(&numbered(1, 2, 9.), 2).unwrap();
        copy_rows(
            &src,
            &dst,
            &[RowCopy {
                src: 0,
                dst: 2,
                len: 1,
            }],
        )
        .unwrap();
        assert_eq!(
            buf.squeeze(0).unwrap().to_vec2::<f32>().unwrap()[2],
            vec![9., 9.]
        );
    }

    #[test]
    fn rows_of_another_width_are_refused() {
        let a = SpanRows::new(&numbered(2, 3, 0.), 3).unwrap();
        let b = SpanRows::new(&numbered(3, 2, 0.), 2).unwrap();
        let plan = [RowCopy {
            src: 0,
            dst: 0,
            len: 1,
        }];
        assert!(copy_rows(&a, &b, &plan).is_err());
    }

    #[test]
    fn a_buffer_that_is_not_whole_rows_is_refused() {
        assert!(SpanRows::new(&numbered(1, 5, 0.), 2).is_err());
    }
}
