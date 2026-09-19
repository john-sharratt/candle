//! A hyper-connection module's projections KO-quantized for the engine.
//!
//! The F32 checkpoint form ([`super::HcWeights`]) is 26 MB a module — 99 of
//! them come to 2.6 GB of resident VRAM on a card whose decode rate is set by
//! how many experts fit — and its two projections run as F32 GEMMs, which
//! `ncu` puts on the SIMT pipe at ~50% SM. Held as Q8 KO they are ~1 byte a
//! weight in the span's dense block, and run on the int8 tensor cores every
//! other dense projection in this engine uses.
//!
//! # The padding, and why it is exact
//!
//! The int8 path tiles K at 128 and N at 32; the gate's bottleneck rank is 320,
//! which fits neither as K. So the rank is padded to the next multiple of 128
//! with **zero** weights, and the down projection's rows to a multiple of 32:
//!
//! ```text
//!   down  [gate 0..r | zero r..r' | inject r'..r'+hc | zero to a multiple of 32]
//!   up    [hc_dim, r'] — its columns r..r' zero
//! ```
//!
//! A zero gate row projects to exactly 0, `silu(0)` is exactly 0, and a zero
//! column of `up` multiplies it into nothing — so the padded columns contribute
//! nothing to the raw gate, and the inject columns are read from `r'` instead
//! of `r`. The only difference from the F32 form is the quantization itself.

use candle::quantized::cuda::to_dynamic;
use candle::quantized::{GgmlDType, Int8Mode, QTensor, SumScale};
use candle::{DType, LiveTensor, Result, Tensor};

use super::{HcProject, HcWeights};
use crate::models::quantized_matmul::QMatMul;

/// K tile of the int8 matmul; the padded gate rank is a multiple of it.
const K_TILE: usize = 128;
/// The int8 matmul's N tile — `ko_tileable`'s row rule; the down projection's
/// row count is padded to a multiple of it, or the weight would stay dense.
const ROW_GROUP: usize = 32;

/// One hyper-connection module with its projections KO-quantized.
pub struct HcWeightsKo {
    norm: Tensor,
    down: QMatMul,
    up: QMatMul,
    /// The gate rank padded to [`K_TILE`].
    gate_cols: usize,
    /// Where the inject columns start, on a module that injects.
    inject_col: Option<usize>,
    mode: Int8Mode,
}

impl HcWeightsKo {
    /// Quantize a checkpoint module (its `1/hc` already folded — see
    /// [`HcWeights::down`]) for `mode`.
    pub fn from_weights(w: &HcWeights, mode: Int8Mode) -> Result<Self> {
        let low_rank = w.low_rank()?;
        let (rows, hc_dim) = w.down.dims2()?;
        let streams = rows - low_rank;
        let gate_cols = low_rank.div_ceil(K_TILE) * K_TILE;
        let dev = w.down.device();

        let mut parts = vec![w.down.narrow(0, 0, low_rank)?];
        if gate_cols > low_rank {
            parts.push(Tensor::zeros(
                (gate_cols - low_rank, hc_dim),
                DType::F32,
                dev,
            )?);
        }
        let inject_col = (streams > 0).then_some(gate_cols);
        if streams > 0 {
            parts.push(w.down.narrow(0, low_rank, streams)?);
        }
        let used = gate_cols + streams;
        let down_rows = used.div_ceil(ROW_GROUP) * ROW_GROUP;
        if down_rows > used {
            parts.push(Tensor::zeros((down_rows - used, hc_dim), DType::F32, dev)?);
        }
        let down = Tensor::cat(&parts, 0)?;

        let up = if gate_cols > low_rank {
            Tensor::cat(
                &[
                    w.up.clone(),
                    Tensor::zeros((hc_dim, gate_cols - low_rank), DType::F32, dev)?,
                ],
                1,
            )?
        } else {
            w.up.clone()
        };

        let quantize = |t: &Tensor| -> Result<QMatMul> {
            QMatMul::from_qtensor_with_mode(QTensor::quantize(t, GgmlDType::Q8_0)?, mode)
        };
        Ok(Self {
            norm: w.norm.clone(),
            down: quantize(&down)?,
            up: quantize(&up)?,
            gate_cols,
            inject_col,
            mode,
        })
    }

    /// One operand for the matmul: q8a128 on an int8 session, the float
    /// tensor on a float one. Carved beside `x`, so it follows the phase.
    fn project<'w>(&self, w: &QMatMul, x: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
        let candle::Device::Cuda(dev) = x.device() else {
            candle::bail!("hc projections run on CUDA");
        };
        // Raw Σx — the mix's block sums stay far below f16's ceiling.
        let acts = to_dynamic(x, self.mode, dev, SumScale::Raw)?;
        w.forward_dynamic(acts.as_dynamic(), DType::F32)
    }
}

impl HcProject for HcWeightsKo {
    fn norm(&self) -> &Tensor {
        &self.norm
    }

    fn down<'w>(&self, xn_flat: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
        self.project(&self.down, xn_flat)
    }

    fn up<'w>(&self, lo: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
        self.project(&self.up, lo)
    }

    fn gate_cols(&self) -> usize {
        self.gate_cols
    }

    fn inject_col(&self) -> Option<usize> {
        self.inject_col
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen4exp::hyper::hc_mix;
    use candle::Device;

    fn lcg(shape: &[usize], seed: u64, scale: f32, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let v: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * scale
            })
            .collect();
        Tensor::from_vec(v, shape, dev).unwrap()
    }

    fn rel_gap(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let num: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = b.iter().map(|y| y * y).sum();
        (num / den).sqrt()
    }

    /// The KO module mixes what the F32 module mixes, to Q8 quantization error,
    /// at the checkpoint's own gate rank of 320 — so the zero padding to 384 is
    /// exercised and a misplaced inject column or a non-zero pad would show as
    /// a gap far above the quantization's.
    #[test]
    fn ko_module_matches_the_f32_module_to_quantization_error() {
        let Ok(dev) = Device::new_cuda(0) else { return };
        let (t, hc, d, lr) = (12usize, 4usize, 256usize, 320usize);
        for with_inject in [true, false] {
            let rows = lr + if with_inject { hc } else { 0 };
            let w = HcWeights::from_checkpoint(
                lcg(&[hc * d], 1, 0.4, &dev).affine(1.0, 1.0).unwrap(),
                lcg(&[rows, hc * d], 2, 0.1, &dev),
                lcg(&[hc * d, lr], 3, 0.1, &dev),
                hc,
            )
            .unwrap();
            let ko = HcWeightsKo::from_weights(&w, Int8Mode::auto(&dev)).unwrap();
            let x = lcg(&[t, hc, d], 4, 2.0, &dev);
            let (want, want_inj) = hc_mix(&x, &w, 1e-6, None).unwrap();
            let (got, got_inj) = hc_mix(&x, &ko, 1e-6, None).unwrap();
            let gap = rel_gap(&got, &want);
            assert!(gap < 0.02, "mix rel gap {gap} (inject={with_inject})");
            match (got_inj, want_inj) {
                (Some(g), Some(w)) => {
                    let gap = rel_gap(&g.contiguous().unwrap(), &w.contiguous().unwrap());
                    assert!(gap < 0.02, "inject rel gap {gap}");
                }
                (None, None) => {}
                _ => panic!("inject presence differs between the two forms"),
            }
        }
    }
}
