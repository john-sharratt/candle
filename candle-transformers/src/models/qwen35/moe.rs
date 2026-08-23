//! The MoE FFN block: softmax top-k routing over 256 experts plus a
//! sigmoid-gated shared expert — reference implementation.
//!
//! Semantics from `qwen35moe.cpp` `build_layer_ffn`:
//!
//! - Router logits → **softmax over all experts** → top-k → renormalize the
//!   selected weights to sum 1 (`norm_topk_prob`) → optional
//!   `expert_weights_scale`.
//! - Each expert is a SwiGLU FFN (`down(silu(gate(x)) ⊙ up(x))`).
//! - The shared expert is its own SwiGLU FFN, scaled by
//!   `sigmoid(w_shared_gate · x)` — a scalar per token — and **added** to the
//!   routed mixture.
//!
//! The reference computes the routed mixture with a dense per-token loop over
//! the selected experts: numerically transparent, trivially correct, and the
//! oracle for the grouped-GEMM production path (which reuses the deepseek4
//! host-dispatch contract).

use candle::{Result, Tensor};

use crate::models::delta_net::mix::silu;

/// One SwiGLU feed-forward: `down(silu(gate(x)) ⊙ up(x))`.
#[derive(Debug, Clone)]
pub struct FfnWeights {
    /// `[ffn, hidden]`.
    pub gate: Tensor,
    /// `[ffn, hidden]`.
    pub up: Tensor,
    /// `[hidden, ffn]`.
    pub down: Tensor,
}

impl FfnWeights {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = silu(&x.matmul(&self.gate.t()?)?)?;
        let u = x.matmul(&self.up.t()?)?;
        g.mul(&u)?.matmul(&self.down.t()?)
    }
}

/// Weights of one MoE block (reference path, F32).
#[derive(Debug, Clone)]
pub struct MoeWeights {
    /// `[n_experts, hidden]`.
    pub router: Tensor,
    /// One SwiGLU per expert.
    pub experts: Vec<FfnWeights>,
    /// The shared expert, always active.
    pub shared: FfnWeights,
    /// `[1, hidden]` — the shared expert's scalar gate.
    pub shared_gate: Tensor,
    pub n_experts_used: usize,
    pub norm_topk_prob: bool,
    /// `expert_weights_scale`; 1.0 when the checkpoint does not set it.
    pub weights_scale: f64,
}

/// Routing decision for one token: expert ids (descending weight) and their
/// mixing weights after renormalization and scaling.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenRoute {
    pub experts: Vec<usize>,
    pub weights: Vec<f32>,
}

/// Softmax → top-k → renorm → scale, per token. `logits [T, E]`.
pub fn route(
    logits: &Tensor,
    k: usize,
    norm_topk: bool,
    weights_scale: f64,
) -> Result<Vec<TokenRoute>> {
    let probs = candle_nn::ops::softmax_last_dim(logits)?;
    let rows = probs.to_vec2::<f32>()?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let mut idx: Vec<usize> = (0..row.len()).collect();
        // Descending by probability; ties broken by ascending expert id so
        // the selection is total-ordered and reproducible.
        idx.sort_by(|&a, &b| {
            row[b]
                .partial_cmp(&row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        idx.truncate(k);
        let mut w: Vec<f32> = idx.iter().map(|&i| row[i]).collect();
        if norm_topk {
            let sum: f32 = w.iter().sum();
            if sum > 0.0 {
                for wi in &mut w {
                    *wi /= sum;
                }
            }
        }
        for wi in &mut w {
            *wi *= weights_scale as f32;
        }
        out.push(TokenRoute {
            experts: idx,
            weights: w,
        });
    }
    Ok(out)
}

impl MoeWeights {
    /// The full block over `[T, hidden]`: routed mixture + gated shared expert.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (t, hidden) = x.dims2()?;
        let logits = x.matmul(&self.router.t()?)?;
        let routes = route(
            &logits,
            self.n_experts_used,
            self.norm_topk_prob,
            self.weights_scale,
        )?;

        let mut rows = Vec::with_capacity(t);
        for (ti, r) in routes.iter().enumerate() {
            let xt = x.narrow(0, ti, 1)?;
            let mut acc = Tensor::zeros((1, hidden), x.dtype(), x.device())?;
            for (&e, &w) in r.experts.iter().zip(r.weights.iter()) {
                let y = self.experts[e].forward(&xt)?;
                acc = acc.add(&y.affine(w as f64, 0.)?)?;
            }
            rows.push(acc);
        }
        let routed = Tensor::cat(&rows, 0)?;

        let shared = self.shared.forward(x)?;
        let gate = candle_nn::ops::sigmoid(&x.matmul(&self.shared_gate.t()?)?)?; // [T, 1]
        routed.add(&shared.broadcast_mul(&gate)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    fn dev() -> Device {
        Device::Cpu
    }

    fn lcg_tensor(shape: &[usize], seed: u64, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let vals: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect();
        Tensor::from_vec(vals, shape, dev).unwrap()
    }

    fn assert_close(a: &Tensor, b: &Tensor, tol: f32, what: &str) {
        let d = a
            .sub(b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(d <= tol, "{what}: max abs diff {d} > {tol}");
    }

    fn tiny_ffn(hidden: usize, ffn: usize, seed: u64, dev: &Device) -> FfnWeights {
        let scale = |t: Tensor| t.affine(0.3, 0.).unwrap();
        FfnWeights {
            gate: scale(lcg_tensor(&[ffn, hidden], seed, dev)),
            up: scale(lcg_tensor(&[ffn, hidden], seed + 1, dev)),
            down: scale(lcg_tensor(&[hidden, ffn], seed + 2, dev)),
        }
    }

    #[test]
    fn routing_selects_topk_renormalizes_and_scales() {
        let dev = dev();
        // One token, 4 experts, logits chosen so softmax order is 2 > 0 > 3 > 1.
        let logits =
            Tensor::from_vec(vec![1.0f32, -2.0, 3.0, 0.5], (1, 4), &dev).unwrap();
        let r = route(&logits, 2, true, 1.5).unwrap();
        assert_eq!(r[0].experts, vec![2, 0]);
        // Renormalized pair sums to 1, then scaled by 1.5.
        let sum: f32 = r[0].weights.iter().sum();
        assert!((sum - 1.5).abs() < 1e-6, "renorm+scale sum {sum}");
        assert!(r[0].weights[0] > r[0].weights[1]);
    }

    #[test]
    fn k_equals_e_without_renorm_is_the_full_softmax_mixture() {
        // With k = E, no renorm, scale 1: the block must equal the dense
        // softmax-weighted mixture of every expert — computed here explicitly.
        let dev = dev();
        let (hidden, ffn, e, t) = (4usize, 6usize, 3usize, 5usize);
        let experts: Vec<FfnWeights> =
            (0..e).map(|i| tiny_ffn(hidden, ffn, 100 + 10 * i as u64, &dev)).collect();
        let moe = MoeWeights {
            router: lcg_tensor(&[e, hidden], 71, &dev),
            experts: experts.clone(),
            shared: tiny_ffn(hidden, ffn, 200, &dev),
            shared_gate: lcg_tensor(&[1, hidden], 72, &dev),
            n_experts_used: e,
            norm_topk_prob: false,
            weights_scale: 1.0,
        };
        let x = lcg_tensor(&[t, hidden], 73, &dev);
        let got = moe.forward(&x).unwrap();

        let probs = candle_nn::ops::softmax_last_dim(
            &x.matmul(&moe.router.t().unwrap()).unwrap(),
        )
        .unwrap();
        let mut expect = Tensor::zeros((t, hidden), DType::F32, &dev).unwrap();
        for (i, ex) in experts.iter().enumerate() {
            let w = probs.narrow(1, i, 1).unwrap();
            expect = expect
                .add(&ex.forward(&x).unwrap().broadcast_mul(&w).unwrap())
                .unwrap();
        }
        let gate = candle_nn::ops::sigmoid(
            &x.matmul(&moe.shared_gate.t().unwrap()).unwrap(),
        )
        .unwrap();
        expect = expect
            .add(
                &moe.shared
                    .forward(&x)
                    .unwrap()
                    .broadcast_mul(&gate)
                    .unwrap(),
            )
            .unwrap();
        assert_close(&got, &expect, 1e-5, "k=E mixture");
    }

    #[test]
    fn shared_expert_is_always_present() {
        // Zero out the router path (k experts with zero weight scale): the
        // block must still emit the gated shared expert.
        let dev = dev();
        let (hidden, ffn) = (4usize, 6usize);
        let moe = MoeWeights {
            router: lcg_tensor(&[2, hidden], 81, &dev),
            experts: vec![
                tiny_ffn(hidden, ffn, 300, &dev),
                tiny_ffn(hidden, ffn, 310, &dev),
            ],
            shared: tiny_ffn(hidden, ffn, 320, &dev),
            shared_gate: lcg_tensor(&[1, hidden], 82, &dev),
            n_experts_used: 1,
            norm_topk_prob: true,
            weights_scale: 0.0,
        };
        let x = lcg_tensor(&[3, hidden], 83, &dev);
        let got = moe.forward(&x).unwrap();
        let gate = candle_nn::ops::sigmoid(
            &x.matmul(&moe.shared_gate.t().unwrap()).unwrap(),
        )
        .unwrap();
        let expect = moe
            .shared
            .forward(&x)
            .unwrap()
            .broadcast_mul(&gate)
            .unwrap();
        assert_close(&got, &expect, 1e-6, "shared-only output");
    }
}
