//! Numerical comparison metrics for the A/B check.
//!
//! Both kernels emit `(num_slots, n_q_head, head_dim)` outputs. The reference
//! is the V2 paged-decode kernel; the candidate is fused-attn-v1. We report
//! aggregate parity plus a per-(q_head) breakdown so a divergence localizes to
//! the head whose read path differs.

use candle::{DType, Result, Tensor};

/// Aggregate + worst-head parity between two decode outputs.
#[derive(Clone, Debug)]
pub struct Metrics {
    /// Mean absolute error over all elements.
    pub mae: f32,
    /// Max absolute error over all elements.
    pub max_abs: f32,
    /// Cosine similarity over the flattened tensors (1.0 = identical direction).
    pub cosine: f32,
    /// Worst per-head MAE and which q_head it was (localizes a divergence).
    pub worst_head: usize,
    pub worst_head_mae: f32,
}

impl Metrics {
    /// `a` = reference (V2), `b` = candidate (fused). Tensors are converted to
    /// F32 on the host for an exact, dtype-independent comparison.
    pub fn compute(a: &Tensor, b: &Tensor, n_q_head: usize, head_dim: usize) -> Result<Metrics> {
        let dims = a.dims().to_vec();
        let num_slots = if dims.is_empty() { 0 } else { dims[0] };
        let va = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let vb = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        if va.len() != vb.len() {
            candle::bail!("metrics: shape mismatch a={:?} b={:?}", a.dims(), b.dims());
        }

        let n = va.len().max(1);
        let mut sum_abs = 0f64;
        let mut max_abs = 0f32;
        let mut dot = 0f64;
        let mut na = 0f64;
        let mut nb = 0f64;
        for (&x, &y) in va.iter().zip(vb.iter()) {
            let d = (x - y).abs();
            sum_abs += d as f64;
            if d > max_abs {
                max_abs = d;
            }
            dot += (x as f64) * (y as f64);
            na += (x as f64) * (x as f64);
            nb += (y as f64) * (y as f64);
        }
        let mae = (sum_abs / n as f64) as f32;
        let cosine = if na > 0.0 && nb > 0.0 {
            (dot / (na.sqrt() * nb.sqrt())) as f32
        } else if na == 0.0 && nb == 0.0 {
            1.0
        } else {
            0.0
        };

        // Per-head MAE: layout is [slot][q_head][dim], contiguous.
        let mut per_head_mae = vec![0f32; n_q_head];
        if n_q_head > 0 && head_dim > 0 && num_slots > 0 {
            let mut per_head_sum = vec![0f64; n_q_head];
            let stride_slot = n_q_head * head_dim;
            #[allow(clippy::needless_range_loop)]
            for s in 0..num_slots {
                for h in 0..n_q_head {
                    let base = s * stride_slot + h * head_dim;
                    let mut acc = 0f64;
                    for d in 0..head_dim {
                        acc += (va[base + d] - vb[base + d]).abs() as f64;
                    }
                    per_head_sum[h] += acc;
                }
            }
            let denom = (num_slots * head_dim) as f64;
            for h in 0..n_q_head {
                per_head_mae[h] = (per_head_sum[h] / denom) as f32;
            }
        }
        let (worst_head, worst_head_mae) =
            per_head_mae
                .iter()
                .enumerate()
                .fold(
                    (0usize, 0f32),
                    |(bi, bm), (i, &m)| {
                        if m > bm {
                            (i, m)
                        } else {
                            (bi, bm)
                        }
                    },
                );

        Ok(Metrics {
            mae,
            max_abs,
            cosine,
            worst_head,
            worst_head_mae,
        })
    }
}
