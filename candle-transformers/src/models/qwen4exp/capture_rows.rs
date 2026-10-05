//! The rows a verify wave keeps past itself for the rewind: the trunk residual
//! the head pass ran over, the PLE conv's normed input, and every KV layer's
//! raw index keys.
//!
//! Each is computed on the wave's own arena and read after the wave, by the
//! rewind at accept time, so it has to be copied off the arena. It is copied
//! into these buffers rather than into a fresh owned tensor per span per
//! layer: they are laid out like the rewind stash — one row per verify row of
//! the cohort, each sequence at its stash row — sized for the widest cohort
//! verified so far and reused across steps, so a verify wave allocates nothing
//! to keep them.

use candle::{DType, Device, Result, Tensor};

/// One cohort's kept rows. Uninitialised outside the rows a wave wrote: a
/// rewind reads a sequence's own rows and nothing else.
pub struct CaptureRows {
    /// `[cap, hc, n_embd]` — the head pass's trunk residual.
    pub head: Tensor,
    /// `[cap, hc · n_embd]` — the PLE conv's normed input.
    pub ple: Tensor,
    /// Per KV layer, `[cap, indexer_head_dim]` — the raw projected index keys.
    pub qsa: Vec<Tensor>,
}

/// The widths a cohort's rows are kept at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureWidths {
    pub hc: usize,
    pub n_embd: usize,
    pub kv_layers: usize,
    pub index_dim: usize,
}

impl CaptureRows {
    /// Buffers for `cap` verify rows. **Allocate outside a forward**, as the
    /// rewind stash is: the copies into them run inside the sweep.
    pub fn new(cap: usize, w: CaptureWidths, device: &Device) -> Result<Self> {
        Ok(Self {
            head: Tensor::empty((cap, w.hc, w.n_embd), DType::F32, device)?,
            ple: Tensor::empty((cap, w.hc * w.n_embd), DType::F32, device)?,
            qsa: (0..w.kv_layers)
                .map(|_| Tensor::empty((cap, w.index_dim), DType::F32, device))
                .collect::<Result<_>>()?,
        })
    }

    /// Verify rows these buffers hold.
    pub fn capacity(&self) -> Result<usize> {
        self.head.dim(0)
    }

    /// Copy `src`'s rows into rows `[row, row + src.dim(0))` of `dst` and
    /// return that view — what the stash keeps in place of an owned copy.
    pub fn keep(dst: &Tensor, row: usize, src: &Tensor) -> Result<Tensor> {
        let rows = src.dim(0)?;
        if row + rows > dst.dim(0)? {
            candle::bail!(
                "qwen4exp capture: rows {row}+{rows} past a {}-row buffer",
                dst.dim(0)?
            );
        }
        let view = dst.narrow(0, row, rows)?;
        view.slice_set(src, 0, 0)?;
        Ok(view)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn widths() -> CaptureWidths {
        CaptureWidths {
            hc: 2,
            n_embd: 3,
            kv_layers: 2,
            index_dim: 4,
        }
    }

    #[test]
    fn a_kept_span_is_its_rows_at_its_stash_row() -> Result<()> {
        let dev = Device::Cpu;
        let rows = CaptureRows::new(5, widths(), &dev)?;
        let src = Tensor::arange(0f32, 8.0, &dev)?.reshape((2, 4))?;
        let kept = CaptureRows::keep(&rows.qsa[1], 2, &src)?;
        assert_eq!(
            kept.to_vec2::<f32>()?,
            vec![vec![0., 1., 2., 3.], vec![4., 5., 6., 7.]]
        );
        let whole = rows.qsa[1].narrow(0, 2, 2)?.to_vec2::<f32>()?;
        assert_eq!(whole, vec![vec![0., 1., 2., 3.], vec![4., 5., 6., 7.]]);
        Ok(())
    }

    #[test]
    fn a_span_past_the_buffer_is_refused() -> Result<()> {
        let dev = Device::Cpu;
        let rows = CaptureRows::new(3, widths(), &dev)?;
        let src = Tensor::zeros((2, 6), DType::F32, &dev)?;
        assert!(CaptureRows::keep(&rows.ple, 2, &src).is_err());
        assert_eq!(rows.capacity()?, 3);
        Ok(())
    }
}
