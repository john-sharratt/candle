//! A short span's contribution to the recurrent state, stored as the
//! generators the mixer computes instead of as an advanced state
//! (`docs/deltanet_timeline.md`).
//!
//! One token advances a V head's state as `S ← S·e^g + v_new ⊗ k`: a scalar
//! decay and a rank-1 matrix. Every element moves, so a diff of the state is a
//! dense matrix with nothing to compress — but the move itself is `g`, `k` and
//! `v_new`, a few hundred numbers against the head's 16,384. A span of at most
//! [`TIMELINE_ROWS`] rows reads its state and writes those generators, one
//! **timeline row** per token; the **fold** applies the first `m` of them once
//! the step knows it keeps `m`:
//!
//! ```text
//!   S[i][j] ← S[i][j]·e^{G[m−1]} + Σ_{t<m} v_new[t][i] · (k[t][j]·e^{G[m−1]−G[t]})
//!   tail    ← the last K−1 of [tail | x[0] .. x[m−1]]
//! ```
//!
//! — the chunked scan's state carry over those `m` rows. Every row quantity
//! depends only on the rows before it, so a partial accept is a choice of `m`,
//! and nothing has to be replayed.
//!
//! The layout is field-major and mirrors `delta_net_timeline.cuh` exactly:
//! `g [R][h_v]`, `k [R][h_k][d]`, `v [R][h_v][d]`, `x [R][conv_dim]`.

use candle::{DType, Device, LiveTensor, Result, Tensor};

use super::mix::DeltaNetState;
use super::types::DeltaNetDims;

/// Rows a timeline holds: the longest span the short-span kernel runs, so every
/// span it takes keeps a timeline and every longer span is a prefill span.
pub const TIMELINE_ROWS: usize = 8;

#[cfg(feature = "cuda")]
const _: () = assert!(TIMELINE_ROWS == candle_kernels::delta_net::DELTA_NET_TIMELINE_ROWS);

/// Where each field of a timeline sits, in F32 elements from its base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineLayout {
    h_v: usize,
    h_k: usize,
    d: usize,
    conv_dim: usize,
}

impl TimelineLayout {
    pub fn of(dims: &DeltaNetDims) -> Self {
        Self {
            h_v: dims.n_v_heads,
            h_k: dims.n_k_heads,
            d: dims.head_dim,
            conv_dim: dims.conv_dim(),
        }
    }

    /// `G`, `[R, h_v]`.
    pub fn g_offset(&self) -> usize {
        0
    }

    /// `k`, `[R, h_k, d]`.
    pub fn k_offset(&self) -> usize {
        self.g_offset() + TIMELINE_ROWS * self.h_v
    }

    /// `v_new`, `[R, h_v, d]`.
    pub fn v_offset(&self) -> usize {
        self.k_offset() + TIMELINE_ROWS * self.h_k * self.d
    }

    /// The raw conv input, `[R, conv_dim]`.
    pub fn x_offset(&self) -> usize {
        self.v_offset() + TIMELINE_ROWS * self.h_v * self.d
    }

    /// One row's elements across every field.
    pub fn row_elems(&self) -> usize {
        self.h_v + self.h_k * self.d + self.h_v * self.d + self.conv_dim
    }

    /// The whole timeline's elements.
    pub fn elems(&self) -> usize {
        TIMELINE_ROWS * self.row_elems()
    }

    pub fn bytes(&self) -> usize {
        self.elems() * DType::F32.size_in_bytes()
    }

    /// A fresh timeline buffer, for a state that lives outside any store — the
    /// single-sequence entry points and the reference tests. Uninitialised: a
    /// fold reads only the rows a span wrote (hot-path invariant 6).
    pub fn alloc(&self, dev: &Device) -> Result<Tensor> {
        Tensor::empty(self.elems(), DType::F32, dev)
    }

    fn field(&self, tl: &Tensor, off: usize, cols: &[usize]) -> Result<Tensor> {
        if tl.dims1()? != self.elems() || tl.dtype() != DType::F32 {
            candle::bail!(
                "timeline: a {:?} {:?} buffer is not a {}-element F32 timeline",
                tl.dims(),
                tl.dtype(),
                self.elems()
            );
        }
        let per_row: usize = cols.iter().product();
        let mut shape = vec![TIMELINE_ROWS];
        shape.extend_from_slice(cols);
        tl.narrow(0, off, TIMELINE_ROWS * per_row)?.reshape(shape)
    }

    pub fn g(&self, tl: &Tensor) -> Result<Tensor> {
        self.field(tl, self.g_offset(), &[self.h_v])
    }

    pub fn k(&self, tl: &Tensor) -> Result<Tensor> {
        self.field(tl, self.k_offset(), &[self.h_k, self.d])
    }

    pub fn v(&self, tl: &Tensor) -> Result<Tensor> {
        self.field(tl, self.v_offset(), &[self.h_v, self.d])
    }

    pub fn x(&self, tl: &Tensor) -> Result<Tensor> {
        self.field(tl, self.x_offset(), &[self.conv_dim])
    }

    /// Write the first `len` rows of every field — the tensor-op mixer's form
    /// of what the kernels write. `g [len, h_v]`, `k [len, h_k, d]`,
    /// `v_new [len, h_v, d]`, `x [len, conv_dim]`.
    ///
    /// The rows may be in any float dtype: this is the reference path, which
    /// runs at whatever dtype the mixer was handed, and the timeline is F32 by
    /// definition — the narrowing (or widening) is the design's, not a guard.
    pub fn write_rows(
        &self,
        tl: &Tensor,
        g: &LiveTensor<'_>,
        k: &LiveTensor<'_>,
        v_new: &LiveTensor<'_>,
        x: &LiveTensor<'_>,
    ) -> Result<()> {
        let len = g.dim(0)?;
        if len == 0 || len > TIMELINE_ROWS {
            candle::bail!("timeline: {len} rows into a {TIMELINE_ROWS}-row timeline");
        }
        for (field, rows) in [
            (self.g(tl)?, g),
            (self.k(tl)?, k),
            (self.v(tl)?, v_new),
            (self.x(tl)?, x),
        ] {
            field
                .narrow(0, 0, len)?
                .slice_set(&rows.to_dtype(DType::F32)?.contiguous()?, 0, 0)?;
        }
        Ok(())
    }
}

/// The fold, in tensor ops: apply the first `m` rows of `tl` to `state` in
/// place — the arithmetic of `delta_net_fold_kernel.cuh` for the paths the
/// kernels do not serve (a CPU device, another head width), held to it by the
/// reference's parity tolerance.
///
/// Both buffers are written into, never replaced: a store's state keeps its
/// identity for the store's whole life.
pub fn fold_by_ops(
    state: &mut DeltaNetState,
    tl: &Tensor,
    m: usize,
    dims: &DeltaNetDims,
) -> Result<()> {
    if m == 0 || m > TIMELINE_ROWS {
        candle::bail!("timeline fold: {m} rows of a {TIMELINE_ROWS}-row timeline");
    }
    let lay = TimelineLayout::of(dims);
    let (h_v, h_k, d) = (dims.n_v_heads, dims.n_k_heads, dims.head_dim);
    // G over the kept rows `[m, h_v]`, its last row `[1, h_v]`, and each row's
    // decay to the end `[m, h_v]`.
    let g = lay.g(tl)?.narrow(0, 0, m)?;
    let g_last = g.narrow(0, m - 1, 1)?;
    let dec = g_last.exp()?.reshape((h_v, 1, 1))?;
    let e = g_last.broadcast_sub(&g)?.exp()?;

    // V head h reads K head h % h_k — the tiled broadcast every DeltaNet path
    // uses — and each row's key is decayed to the end: `[m, h_v, d]`.
    let k = lay
        .k(tl)?
        .narrow(0, 0, m)?
        .unsqueeze(1)?
        .expand((m, h_v / h_k, h_k, d))?
        .reshape((m, h_v, d))?;
    let kg = k.broadcast_mul(&e.unsqueeze(2)?)?;
    let v = lay.v(tl)?.narrow(0, 0, m)?;

    // Σ_t v[t,h,i]·kg[t,h,j] = (vᵀ kg) per head: `[h_v, d, d]`.
    let acc = v
        .permute((1, 2, 0))?
        .contiguous()?
        .matmul(&kg.permute((1, 0, 2))?.contiguous()?)?;
    let next = state.s.broadcast_mul(&dec)?.add(&acc)?;
    state.s.slice_set(&next, 0, 0)?;

    // The tail: the last K−1 raw inputs of [tail | x[0..m)], `x` read as
    // `[conv_dim, m]`.
    let tcols = dims.conv_kernel - 1;
    if tcols > 0 {
        let x = lay.x(tl)?.narrow(0, 0, m)?.t()?;
        let window = Tensor::cat(&[&state.conv_tail, &x], 1)?;
        let tail = window
            .narrow(1, window.dim(1)? - tcols, tcols)?
            .contiguous()?;
        state.conv_tail.slice_set(&tail, 0, 0)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims() -> DeltaNetDims {
        DeltaNetDims {
            head_dim: 4,
            n_k_heads: 2,
            n_v_heads: 4,
            conv_kernel: 4,
        }
    }

    /// The offsets are the kernel header's, element for element: G, then k
    /// once per K head, then v_new per V head, then the raw conv rows, each
    /// eight rows deep.
    #[test]
    fn the_layout_is_the_kernel_headers() {
        let lay = TimelineLayout::of(&dims());
        // conv_dim = (2·2 + 4)·4 = 32.
        assert_eq!(lay.g_offset(), 0);
        assert_eq!(lay.k_offset(), 32);
        assert_eq!(lay.v_offset(), 32 + 64);
        assert_eq!(lay.x_offset(), 32 + 64 + 128);
        assert_eq!(lay.row_elems(), 4 + 8 + 16 + 32);
        assert_eq!(lay.elems(), 480);
        assert_eq!(lay.bytes(), 1920);
        // Flash-Next: h_v 48, h_k 16, d 128, conv_dim 10,240.
        let flash = TimelineLayout::of(&DeltaNetDims {
            head_dim: 128,
            n_k_heads: 16,
            n_v_heads: 48,
            conv_kernel: 4,
        });
        assert_eq!(flash.row_elems(), 18_480);
        assert_eq!(flash.bytes(), 591_360);
        assert_eq!(flash.x_offset(), 384 + 16_384 + 49_152);
    }

    /// Rows written land at their fields and read back exactly.
    #[test]
    fn rows_read_back_from_their_fields() {
        let d = dims();
        let lay = TimelineLayout::of(&d);
        let dev = Device::Cpu;
        // Zeroed here, unlike `alloc`'s buffer, so a write past the span shows.
        let tl = Tensor::zeros(lay.elems(), DType::F32, &dev).unwrap();
        let ramp = |shape: &[usize], base: f32| {
            let n: usize = shape.iter().product();
            Tensor::from_vec(
                (0..n).map(|i| base + i as f32).collect::<Vec<_>>(),
                shape,
                &dev,
            )
            .unwrap()
        };
        let (g, k, v, x) = (
            ramp(&[2, 4], 1.0),
            ramp(&[2, 2, 4], 100.0),
            ramp(&[2, 4, 4], 1000.0),
            ramp(&[2, 32], 5000.0),
        );
        lay.write_rows(&tl, &g, &k, &v, &x).unwrap();
        let flat: Vec<f32> = tl.to_vec1().unwrap();
        assert_eq!(&flat[0..8], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        assert_eq!(flat[8], 0.0, "rows past the span are untouched");
        assert_eq!(flat[lay.k_offset()], 100.0);
        assert_eq!(flat[lay.k_offset() + 15], 115.0);
        assert_eq!(flat[lay.v_offset() + 31], 1031.0);
        assert_eq!(flat[lay.x_offset() + 63], 5063.0);
    }

    /// One row folded is the decode step's closed form: S·e^g + v_new ⊗ k,
    /// and the tail shifted by the row's raw input.
    #[test]
    fn one_row_folds_to_the_decode_steps_update() {
        let d = DeltaNetDims {
            head_dim: 2,
            n_k_heads: 1,
            n_v_heads: 1,
            conv_kernel: 3,
        };
        let lay = TimelineLayout::of(&d);
        let dev = Device::Cpu;
        let tl = lay.alloc(&dev).unwrap();
        // g = 0: e^g = 1 exactly, so the expected values are exact too.
        let g = Tensor::new(&[[0.0f32]], &dev).unwrap();
        let k = Tensor::new(&[[[1.0f32, 2.0]]], &dev).unwrap();
        let v = Tensor::new(&[[[3.0f32, -1.0]]], &dev).unwrap();
        // conv_dim = (2 + 1)·2 = 6.
        let x = Tensor::new(&[[10.0f32, 11., 12., 13., 14., 15.]], &dev).unwrap();
        lay.write_rows(&tl, &g, &k, &v, &x).unwrap();
        let mut st = DeltaNetState {
            s: Tensor::new(&[[[2.0f32, 4.0], [6.0, 8.0]]], &dev).unwrap(),
            conv_tail: Tensor::new(
                &[
                    [1.0f32, 2.0],
                    [3., 4.],
                    [5., 6.],
                    [7., 8.],
                    [9., 10.],
                    [11., 12.],
                ],
                &dev,
            )
            .unwrap(),
        };
        fold_by_ops(&mut st, &tl, 1, &d).unwrap();
        // S + [3, −1]ᵀ ⊗ [1, 2].
        assert_eq!(
            st.s.to_vec3::<f32>().unwrap(),
            vec![vec![vec![5.0, 10.0], vec![5.0, 6.0]]]
        );
        assert_eq!(
            st.conv_tail.to_vec2::<f32>().unwrap(),
            vec![
                vec![2.0, 10.0],
                vec![4.0, 11.0],
                vec![6.0, 12.0],
                vec![8.0, 13.0],
                vec![10.0, 14.0],
                vec![12.0, 15.0]
            ]
        );
    }

    #[test]
    fn a_fold_outside_the_timeline_is_refused() {
        let d = dims();
        let lay = TimelineLayout::of(&d);
        let tl = lay.alloc(&Device::Cpu).unwrap();
        let mut st = DeltaNetState::zeros(&d, &Device::Cpu).unwrap();
        assert!(fold_by_ops(&mut st, &tl, 0, &d).is_err());
        assert!(fold_by_ops(&mut st, &tl, TIMELINE_ROWS + 1, &d).is_err());
    }
}
