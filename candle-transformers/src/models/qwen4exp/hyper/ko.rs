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

use candle::quantized::cuda::{to_dynamic, DynamicTensor, Q8a128Operand};
use candle::quantized::{GgmlDType, Int8Mode, QTensor, SumScale};
use candle::{DType, LiveTensor, Result, Tensor};

use super::{HcProject, HcProjectQ8, HcWeights};
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

    /// The `down` and `up` weights and the mode they were built for — what the
    /// projection bench drives directly to sweep the K split.
    pub(super) fn matmuls(&self) -> (&QMatMul, &QMatMul, Int8Mode) {
        (&self.down, &self.up, self.mode)
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

    fn as_q8(&self) -> Option<&dyn HcProjectQ8> {
        self.mode.is_int8().then_some(self as &dyn HcProjectQ8)
    }
}

impl HcProjectQ8 for HcWeightsKo {
    /// The weights' own convention — one for both, since `new` builds them alike —
    /// so the producers' operands and `up_silu`'s in-loader quantize agree by
    /// construction rather than by two defaults happening to match.
    fn sum_scale(&self) -> SumScale {
        self.up.sum_scale()
    }

    fn down_q8<'w>(&self, xn: &Q8a128Operand<'w>) -> Result<LiveTensor<'w>> {
        self.down
            .forward_dynamic(DynamicTensor::Int8(xn), DType::F32)
    }

    fn up_q8<'w>(&self, lo: &Q8a128Operand<'w>) -> Result<LiveTensor<'w>> {
        self.up.forward_dynamic(DynamicTensor::Int8(lo), DType::F32)
    }

    fn up_silu<'w>(&self, proj: &LiveTensor<'w>, gate_cols: usize) -> Result<LiveTensor<'w>> {
        // Quantizes under `up`'s Σx convention, which is the module's (`sum_scale`).
        self.up.forward_silu_f32(proj, gate_cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::gpu_test_lock::gpu_serial as gpu_guard;
    use crate::models::qwen4exp::hyper::hc_mix;
    use candle::quantized::cuda::Q8a128Data;
    use candle::{CudaDevice, Device};

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

    /// The fused int8 mix is the unfused chain bit for bit: the norm emitting its
    /// own q8a128 operand and the SiLU emitting `up`'s quantize from the same
    /// floats, with the same mapping and order, as the standalone quantizes did.
    /// The unfused chain is spelled out longhand — norm, quantize, down, strided
    /// SiLU, quantize, up, collapse — so the claim is the property, not a snapshot.
    #[test]
    fn the_fused_int8_mix_is_the_unfused_chain_bit_for_bit() {
        use crate::models::qwen4exp::hyper::cuda_fused;
        let _gpu = gpu_guard();
        let Ok(dev) = Device::new_cuda(0) else { return };
        let (hc, d, lr) = (4usize, 2560usize, 320usize);
        let w = HcWeights::from_checkpoint(
            lcg(&[hc * d], 21, 0.4, &dev).affine(1.0, 1.0).unwrap(),
            lcg(&[lr + hc, hc * d], 22, 0.1, &dev),
            lcg(&[hc * d, lr], 23, 0.1, &dev),
            hc,
        )
        .unwrap();
        let ko = HcWeightsKo::from_weights(&w, Int8Mode::auto(&dev)).unwrap();
        assert!(
            ko.as_q8().is_some(),
            "an int8 KO module takes the fused path"
        );
        for t in [1usize, 3, 16] {
            let x = lcg(&[t, hc, d], 24 + t as u64, 2.0, &dev);
            let (got, got_inj) = hc_mix(&x, &ko, 1e-6, None).unwrap();

            let xn = cuda_fused::norm(&x, &w.norm, 1e-6, None).unwrap();
            let proj = ko.down(&xn.reshape((t, hc * d)).unwrap()).unwrap();
            let lo = proj.narrow(1, 0, ko.gate_cols()).unwrap().silu().unwrap();
            let gate_raw = ko.up(&lo).unwrap();
            let want = cuda_fused::mix(&xn, &gate_raw, hc, d, None).unwrap();
            let want_inj = proj.narrow(1, ko.inject_col().unwrap(), hc).unwrap();

            let bits = |a: &Tensor| a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(bits(&got), bits(&want), "mix at {t} rows");
            let inj = |a: &Tensor| a.to_vec2::<f32>().unwrap();
            assert_eq!(inj(&got_inj.unwrap()), inj(&want_inj), "inject at {t} rows");
        }
    }

    /// The bytes a q8a128 operand's tiles own, in tile order: each tile's 128 qs and
    /// the 4-byte `{scale, Σx}` half2 at the head of its 16-byte ds slot (blocks.cuh,
    /// `q8a1024_qs_off`/`q8a1024_ds_off`). The rest of each slot and the last
    /// block's unused tiles are never written, so they are not compared.
    fn tile_bytes(op: &Q8a128Operand<'_>, dev: &CudaDevice) -> Vec<u8> {
        const BLK: usize = 1152;
        const META: usize = 1024;
        let raw = match &op.data {
            Q8a128Data::Tensor(t) => t.flatten_all().unwrap().to_vec1::<u8>().unwrap(),
            Q8a128Data::Owned(s) => dev.memcpy_dtov(s).unwrap(),
        };
        let tiles = op.rows * op.cols / 128;
        let mut out = Vec::with_capacity(tiles * 132);
        for tile in 0..tiles {
            let base = (tile / 8) * BLK;
            let qs = base + (tile % 8) * 128;
            let ds = base + META + (tile % 8) * 16;
            out.extend_from_slice(&raw[qs..qs + 128]);
            out.extend_from_slice(&raw[ds..ds + 4]);
        }
        out
    }

    /// `got == want` over [`tile_bytes`] output, naming the first tile that differs
    /// and whether it is the int8 values or the `{scale, Σx}` header.
    fn assert_same_tiles(got: &[u8], want: &[u8], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: tile count");
        if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
            let (tile, at) = (i / 132, i % 132);
            let part = if at < 128 { "qs" } else { "{scale, Σx}" };
            let span = tile * 132..tile * 132 + 132;
            panic!(
                "{what}: tile {tile} differs in its {part} (byte {at}): got {:?} want {:?}",
                &got[span.clone()][128..],
                &want[span][128..]
            );
        }
    }

    /// The standalone quantize of `xs` — the operand each fused producer replaces.
    fn quantized(xs: &Tensor, dev: &CudaDevice, sum_scale: SumScale) -> Vec<u8> {
        let mode = Int8Mode::auto(&Device::Cuda(dev.clone()));
        let acts = to_dynamic(xs, mode, dev, sum_scale).unwrap();
        let DynamicTensor::Int8(op) = acts.as_dynamic() else {
            panic!("an int8 mode quantizes")
        };
        tile_bytes(op, dev)
    }

    /// `up` over the bottleneck's SiLU, quantized by the matmul's own loader, is the
    /// two-step chain bit for bit: `silu` of the gate columns, the standalone
    /// quantize, then the int8 GEMM over that operand. Across decode widths that run
    /// the mode-1 tile (1, 3, 16 and the 17–32 band) and prefill widths that run
    /// mode-2 (40, 96), a partial token tile, and the gate columns read through the
    /// padded projection's row stride.
    #[test]
    fn the_fused_up_is_the_silu_then_quantize_chain_bit_for_bit() {
        let _gpu = gpu_guard();
        let Ok(dev) = Device::new_cuda(0) else { return };
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let (hc, d, lr) = (4usize, 2560usize, 320usize);
        let w = HcWeights::from_checkpoint(
            lcg(&[hc * d], 71, 0.4, &dev).affine(1.0, 1.0).unwrap(),
            lcg(&[lr + hc, hc * d], 72, 0.1, &dev),
            lcg(&[hc * d, lr], 73, 0.1, &dev),
            hc,
        )
        .unwrap();
        let mode = Int8Mode::auto(&dev);
        let ko = HcWeightsKo::from_weights(&w, mode).unwrap();
        let q8 = ko.as_q8().expect("an int8 KO module");
        let (cols, width) = (ko.gate_cols(), ko.inject_col().unwrap() + hc);
        for t in [1usize, 3, 16, 17, 40, 96] {
            let proj = lcg(&[t, width.div_ceil(32) * 32], 74 + t as u64, 4.0, &dev);
            let got = q8.up_silu(&proj, cols).unwrap();
            let lo = proj.narrow(1, 0, cols).unwrap().silu().unwrap();
            let acts = to_dynamic(&lo, mode, cuda, SumScale::Raw).unwrap();
            let DynamicTensor::Int8(op) = acts.as_dynamic() else {
                panic!("an int8 mode quantizes")
            };
            let want = ko
                .matmuls()
                .1
                .forward_dynamic(DynamicTensor::Int8(op), DType::F32)
                .unwrap();
            let bits = |a: &Tensor| a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(bits(&got), bits(&want), "{t} rows");
        }
    }

    /// Each fused producer writes, byte for byte, the operand the standalone
    /// quantize writes from the floats it stores — the int8 values AND every
    /// tile's `{scale, Σx}`, under both `Σx` conventions. Compared as raw bytes:
    /// a GEMM against a symmetric weight never reads `Σx` (its `m` is zero), so
    /// it could not see a sum that disagreed.
    #[test]
    fn the_fused_producers_write_the_quantize_bytes() {
        use crate::models::qwen4exp::hyper::cuda_fused;
        let _gpu = gpu_guard();
        let Ok(dev) = Device::new_cuda(0) else { return };
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let (hc, d, width, cols) = (4usize, 2560usize, 416usize, 384usize);
        let gain = lcg(&[hc * d], 30, 0.4, &dev).affine(1.0, 1.0).unwrap();
        for sum_scale in [SumScale::Raw, SumScale::ByAmax] {
            for t in [1usize, 5, 16] {
                // The norm: its stored `xn`, quantized as the down projection's operand.
                let x = lcg(&[t, hc, d], 31 + t as u64, 2.0, &dev);
                let (xn, op) = cuda_fused::norm_q8(&x, &gain, 1e-6, None, sum_scale).unwrap();
                let want = quantized(&xn.reshape((t, hc * d)).unwrap(), cuda, sum_scale);
                assert_same_tiles(
                    &tile_bytes(&op, cuda),
                    &want,
                    &format!("norm {sum_scale:?} at {t} rows"),
                );

                // The SiLU: the gate columns of a padded projection, read through its
                // row stride.
                let proj = lcg(&[t, width], 32 + t as u64, 4.0, &dev);
                let op = cuda_fused::silu_q8(&proj, cols, sum_scale).unwrap();
                let lo = proj.narrow(1, 0, cols).unwrap().silu().unwrap();
                let want = quantized(&lo, cuda, sum_scale);
                assert_same_tiles(
                    &tile_bytes(&op, cuda),
                    &want,
                    &format!("silu {sum_scale:?} at {t} rows"),
                );

                // The collapse: the mixed block input it stores, bit for bit as the
                // plain mix stores it, and quantized as the block's operand.
                let xn = lcg(&[t, hc, d], 33 + t as u64, 2.0, &dev);
                let gate = lcg(&[t, hc * d], 40 + t as u64, 2.0, &dev);
                let (mixed, op) = cuda_fused::mix_q8(&xn, &gate, hc, d, None, sum_scale).unwrap();
                let want_mixed = cuda_fused::mix(&xn, &gate, hc, d, None).unwrap();
                let bits = |a: &Tensor| a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                assert_eq!(bits(&mixed), bits(&want_mixed), "mixed at {t} rows");
                let want = quantized(&want_mixed, cuda, sum_scale);
                assert_same_tiles(
                    &tile_bytes(&op, cuda),
                    &want,
                    &format!("mix {sum_scale:?} at {t} rows"),
                );
            }
        }
    }

    /// The gated combine is the eager sum it replaces bit for bit: the shared
    /// gate's sigmoid, the broadcast multiply, the add, then the combine.
    #[test]
    fn the_gated_combine_is_the_eager_sum_bit_for_bit() {
        use crate::models::qwen4exp::hyper::cuda_fused::{self, SharedGate};
        let _gpu = gpu_guard();
        let Ok(dev) = Device::new_cuda(0) else { return };
        let (hc, d) = (4usize, 2560usize);
        for t in [1usize, 7] {
            let res = lcg(&[t, hc, d], 50 + t as u64, 2.0, &dev);
            let routed = lcg(&[t, d], 51, 1.0, &dev);
            let shared = lcg(&[t, d], 52, 1.0, &dev);
            // The gate as the model hands it over: column 0 of a padded projection.
            let gate_full = lcg(&[t, 32], 53, 4.0, &dev);
            let gate = gate_full.narrow(1, 0, 1).unwrap();
            let inject = lcg(&[t, hc], 54, 1.0, &dev);

            let mut got = res.copy().unwrap();
            cuda_fused::combine_gated(
                &mut got,
                &routed,
                SharedGate {
                    shared: &shared,
                    gate: &gate,
                },
                &inject,
            )
            .unwrap();

            let gated = shared
                .broadcast_mul(&candle_nn::ops::sigmoid(&gate).unwrap())
                .unwrap();
            let block = (&routed + &gated).unwrap();
            let mut want = res.copy().unwrap();
            cuda_fused::combine(&mut want, &block, &inject).unwrap();

            let bits = |a: &Tensor| a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(bits(&got), bits(&want), "{t} rows");
        }
    }

    /// The KO module mixes what the F32 module mixes, to Q8 quantization error,
    /// at the checkpoint's own gate rank of 320 — so the zero padding to 384 is
    /// exercised and a misplaced inject column or a non-zero pad would show as
    /// a gap far above the quantization's.
    #[test]
    fn ko_module_matches_the_f32_module_to_quantization_error() {
        let _gpu = gpu_guard();
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
            assert_eq!(ko.gate_cols(), 384, "rank 320 pads to the next K tile");
            assert_eq!(ko.inject_col(), with_inject.then_some(384));
            let x = lcg(&[t, hc, d], 4, 2.0, &dev);
            let (want, want_inj) = hc_mix(&x, &w, 1e-6, None).unwrap();
            let (got, got_inj) = hc_mix(&x, &ko, 1e-6, None).unwrap();
            let gap = rel_gap(&got, &want);
            assert!(gap < 0.02, "mix rel gap {gap} (inject={with_inject})");
            match (got_inj, want_inj) {
                (Some(g), Some(w)) => {
                    let gap = rel_gap(&g, &w);
                    assert!(gap < 0.02, "inject rel gap {gap}");
                }
                (None, None) => {}
                _ => panic!("inject presence differs between the two forms"),
            }
        }
    }
}
