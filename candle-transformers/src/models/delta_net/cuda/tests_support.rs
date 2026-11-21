//! Fixtures for the DeltaNet kernel tests: synthetic mixer waves, span tables and
//! CPU reference operands.
//!
//! The fixtures build the fused layout and its tiled, scaled and gated operands
//! from the same random values. That pins the in-kernel grouped-query mapping,
//! the read scale and the gate arithmetic to the tensor-op definitions. The short
//! table is assembled as the production builder assembles it, and `tile_bytes`
//! gives each q8a128 tile as its 128 codes followed by its header.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::mix::l2_norm;

    use super::*;

    use candle::Device;

    /// One span as the kernel tests hold it: they drive the launches directly,
    /// without the mixer's [`DeltaNetSeq`] bookkeeping around them.
    pub(in super::super) struct TestSpan<'a> {
        /// The table is shared by the conv and the scan, and each reads only the
        /// rows it needs — the conv the tails, the scan the states. A case that
        /// drives one kernel names any live tensor in the rows that kernel does
        /// not dereference.
        pub(in super::super) tail: &'a Tensor,
        pub(in super::super) tail_out: &'a Tensor,
        pub(in super::super) state: &'a Tensor,
        pub(in super::super) state_out: &'a Tensor,
        pub(in super::super) start: usize,
        pub(in super::super) len: usize,
    }

    /// The table [`build_span_table`] produces, from tensors rather than from
    /// `DeltaNetSeq`s. Same layout, so a divergence between them is a
    /// compile-time break rather than a silent one.
    pub(in super::super) fn span_table(rows: &[TestSpan<'_>]) -> DeltaNetSpanTable<'static> {
        let n = rows.len();
        let mut ptrs: Vec<i64> = Vec::with_capacity(4 * n);
        for r in rows {
            ptrs.push(f32_ptr(r.tail, "tail").unwrap() as i64);
        }
        for r in rows {
            ptrs.push(f32_ptr(r.tail_out, "tail out").unwrap() as i64);
        }
        for r in rows {
            ptrs.push(f32_ptr(r.state, "state").unwrap() as i64);
        }
        for r in rows {
            ptrs.push(f32_ptr(r.state_out, "state out").unwrap() as i64);
        }
        let mut ext: Vec<u32> = Vec::with_capacity(2 * n);
        ext.extend(rows.iter().map(|r| r.start as u32));
        ext.extend(rows.iter().map(|r| r.len as u32));
        let max_len = rows.iter().map(|r| r.len).max().unwrap();
        let dev = rows[0].tail.device().clone();
        DeltaNetSpanTable {
            ptrs: Tensor::from_vec(ptrs, (4, n), &dev).unwrap(),
            spans: Tensor::from_vec(ext, (2, n), &dev).unwrap(),
            n,
            max_len,
        }
    }

    pub(in super::super) fn lcg_tensor(shape: &[usize], seed: u64, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let vals: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect();
        Tensor::from_vec(vals, shape, dev).unwrap()
    }

    pub(in super::super) fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        a.sub(b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// One synthetic mixer wave in the fused kernels' own layout — a single
    /// conv-output buffer whose Q|K columns are l2-normed and whose trailing
    /// columns are V, plus the raw gate projections — and the
    /// tiled/scaled/gated operands the sequential reference consumes.
    /// Building both from the same randomness is what pins the in-kernel GQA
    /// mapping, read scale, and gate arithmetic to the tensor-op forms.
    pub(in super::super) struct FusedCase {
        pub(in super::super) conved: Tensor,
        pub(in super::super) alpha: Tensor,
        pub(in super::super) blin: Tensor,
        pub(in super::super) dt_bias: Tensor,
        pub(in super::super) a: Tensor,
        pub(in super::super) q_scale: f32,
        // Reference operands (CPU): tiled to h_v, q scaled, gates applied.
        pub(in super::super) q_ref: Tensor,
        pub(in super::super) k_ref: Tensor,
        pub(in super::super) v_ref: Tensor,
        pub(in super::super) g_ref: Tensor,
        pub(in super::super) b_ref: Tensor,
    }

    impl FusedCase {
        pub(in super::super) fn build(
            t: usize,
            h_k: usize,
            h_v: usize,
            seed: u64,
            cpu: &Device,
            gpu: &Device,
        ) -> Self {
            use super::super::super::mix::softplus;
            let d = 128usize;
            let conv_dim = (2 * h_k + h_v) * d; // Q | K | V column layout

            // The scan kernels read a post-epilogue buffer: Q|K columns
            // l2-normed, V columns arbitrary (post-SiLU values — random works,
            // the reference reads the same buffer).
            let qk = l2_norm(&lcg_tensor(&[t, 2 * h_k, d], seed, cpu), 1e-6).unwrap();
            let v_cols = lcg_tensor(&[t, h_v * d], seed + 1, cpu);
            let conved =
                Tensor::cat(&[&qk.reshape((t, 2 * h_k * d)).unwrap(), &v_cols], 1).unwrap();
            let alpha = lcg_tensor(&[t, h_v], seed + 2, cpu);
            let blin = lcg_tensor(&[t, h_v], seed + 3, cpu);
            let dt_bias = lcg_tensor(&[h_v], seed + 4, cpu);
            // Negative per-head decay coefficients, as `−exp(A_log)` is.
            let a = (lcg_tensor(&[h_v], seed + 5, cpu).abs().unwrap() + 0.1)
                .unwrap()
                .neg()
                .unwrap();
            let q_scale = 1.0f32 / (d as f32).sqrt();
            assert_eq!(conved.dims2().unwrap(), (t, conv_dim));

            // The reference operands, by the tensor-op definitions.
            let tile = |x: &Tensor| -> Tensor {
                x.unsqueeze(1)
                    .unwrap()
                    .expand((t, h_v / h_k, h_k, d))
                    .unwrap()
                    .reshape((t, h_v, d))
                    .unwrap()
            };
            let q_ref = (tile(&qk.narrow(1, 0, h_k).unwrap().contiguous().unwrap())
                * q_scale as f64)
                .unwrap();
            let k_ref = tile(&qk.narrow(1, h_k, h_k).unwrap().contiguous().unwrap());
            let v_ref = v_cols.reshape((t, h_v, d)).unwrap();
            let g_ref = softplus(&alpha.broadcast_add(&dt_bias).unwrap())
                .unwrap()
                .broadcast_mul(&a)
                .unwrap();
            let b_ref = candle_nn::ops::sigmoid(&blin).unwrap();

            let to = |x: &Tensor| x.to_device(gpu).unwrap().contiguous().unwrap();
            Self {
                conved: to(&conved),
                alpha: to(&alpha),
                blin: to(&blin),
                dt_bias: to(&dt_bias),
                a: to(&a),
                q_scale,
                q_ref,
                k_ref,
                v_ref,
                g_ref,
                b_ref,
            }
        }

        pub(in super::super) fn fused<'a>(&'a self, o: &'a Tensor) -> DeltaNetFused<'a, 'static> {
            DeltaNetFused {
                conved: &self.conved,
                alpha: &self.alpha,
                blin: &self.blin,
                dt_bias: &self.dt_bias,
                a: &self.a,
                o,
                q_scale: self.q_scale,
            }
        }
    }

    /// The op-form conv epilogue: SiLU everywhere, then l2_norm over each
    /// 128-wide head of the leading `qk_channels` — exactly what the fallback
    /// computes with `silu` and `l2_norm`. `y` is channel-major `[C, T]`.
    pub(in super::super) fn conv_epilogue_ref(y: &Tensor, qk_channels: usize, eps: f64) -> Tensor {
        use super::super::super::mix::{l2_norm, silu};
        let (c, t) = y.dims2().unwrap();
        let s = silu(y).unwrap();
        let qk = s
            .narrow(0, 0, qk_channels)
            .unwrap()
            .t()
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((t, qk_channels / 128, 128))
            .unwrap();
        let qk = l2_norm(&qk, eps)
            .unwrap()
            .reshape((t, qk_channels))
            .unwrap()
            .t()
            .unwrap()
            .contiguous()
            .unwrap();
        Tensor::cat(
            &[&qk, &s.narrow(0, qk_channels, c - qk_channels).unwrap()],
            0,
        )
        .unwrap()
    }

    pub(in super::super) fn f32_bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// One mixer layer's raw wave operands for the short-span tests: the QKV
    /// projection rows the conv reads, the conv weights, and the gate
    /// projections and constants. GQA 3:1, Flash-Next's ratio (48 V heads over
    /// 16 K heads), at a width a test can afford.
    pub(in super::super) struct ShortCase {
        pub(in super::super) qkv: Tensor,
        pub(in super::super) kern: Tensor,
        pub(in super::super) alpha: Tensor,
        pub(in super::super) blin: Tensor,
        pub(in super::super) dt_bias: Tensor,
        pub(in super::super) a: Tensor,
        pub(in super::super) h_k: usize,
        pub(in super::super) h_v: usize,
        pub(in super::super) conv_dim: usize,
    }

    impl ShortCase {
        pub(in super::super) const D: usize = 128;

        pub(in super::super) fn build(t: usize, seed: u64, gpu: &Device) -> Self {
            let cpu = Device::Cpu;
            let (h_k, h_v) = (2usize, 6usize);
            let conv_dim = (2 * h_k + h_v) * Self::D;
            let to = |x: Tensor| x.to_device(gpu).unwrap().contiguous().unwrap();
            Self {
                qkv: to(lcg_tensor(&[t, conv_dim], seed, &cpu)
                    .affine(4.0, 0.0)
                    .unwrap()),
                kern: to(lcg_tensor(
                    &[conv_dim, DELTA_NET_SHORT_SPAN_CONV],
                    seed + 1,
                    &cpu,
                )),
                alpha: to(lcg_tensor(&[t, h_v], seed + 2, &cpu)
                    .affine(4.0, 0.0)
                    .unwrap()),
                blin: to(lcg_tensor(&[t, h_v], seed + 3, &cpu)
                    .affine(4.0, 0.0)
                    .unwrap()),
                dt_bias: to(lcg_tensor(&[h_v], seed + 4, &cpu)),
                a: to((lcg_tensor(&[h_v], seed + 5, &cpu).abs().unwrap() + 0.1)
                    .unwrap()
                    .neg()
                    .unwrap()),
                h_k,
                h_v,
                conv_dim,
            }
        }

        pub(in super::super) fn fused<'a>(
            &'a self,
            conved: &'a Tensor,
            o: &'a Tensor,
        ) -> DeltaNetFused<'a, 'static> {
            DeltaNetFused {
                conved,
                alpha: &self.alpha,
                blin: &self.blin,
                dt_bias: &self.dt_bias,
                a: &self.a,
                o,
                q_scale: 1.0 / (Self::D as f32).sqrt(),
            }
        }
    }

    /// A short-span table from tensors: each span's conv tail, state and
    /// timeline, then its extents — the layout [`build_span_tables`] writes for
    /// short spans.
    pub(in super::super) fn short_table(
        rows: &[(&Tensor, &Tensor, &Tensor, usize, usize)],
    ) -> DeltaNetSpanTable<'static> {
        let n = rows.len();
        let mut ptrs: Vec<i64> = Vec::with_capacity(3 * n);
        for field in 0..3 {
            for r in rows {
                let t = [r.0, r.1, r.2][field];
                ptrs.push(f32_ptr(t, "short span").unwrap() as i64);
            }
        }
        let mut ext: Vec<u32> = rows.iter().map(|r| r.3 as u32).collect();
        ext.extend(rows.iter().map(|r| r.4 as u32));
        let dev = rows[0].0.device().clone();
        DeltaNetSpanTable {
            ptrs: Tensor::from_vec(ptrs, (3, n), &dev).unwrap(),
            spans: Tensor::from_vec(ext, (2, n), &dev).unwrap(),
            n,
            max_len: rows.iter().map(|r| r.4).max().unwrap(),
        }
    }

    /// Each q8a128 tile of `op` as its 128 int8 codes then its 4-byte
    /// `{scale, Σx}` header, in tile order — the operand's bytes without the
    /// super-block padding, which no reader touches.
    pub(in super::super) fn tile_bytes(
        op: &Q8a128Operand<'_>,
        dev: &candle::cuda_backend::CudaDevice,
    ) -> Vec<u8> {
        use candle::quantized::cuda::Q8a128Data;
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
}
