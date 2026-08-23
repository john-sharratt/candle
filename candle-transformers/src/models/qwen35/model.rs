//! Model assembly: the hybrid decoder stack, per-session state, and the
//! reference forward.
//!
//! Layer skeleton (per `qwen35.cpp`): the FFN's residual is taken **before**
//! `post_attention_norm` —
//!
//! ```text
//!   res  = x
//!   x    = mix(rms(x, attn_norm)) + res         mix ∈ {DeltaNet, Attention}
//!   res2 = x
//!   x    = ffn(rms(x, post_attention_norm)) + res2
//! ```
//!
//! The per-session state is one entry per layer — a recurrent
//! [`DeltaNetState`] or a KV [`AttentionState`] — and the whole-model
//! segmentation property (forward in segments ≡ forward one-shot) is what the
//! engine's turn sealing, snapshot, and resume semantics build on.

use candle::{Result, Tensor};

use super::attention::{attention_layer_forward, AttentionState, AttentionWeights, RopeTables};
use super::config::{LayerKind, Qwen35Config};
use crate::models::delta_net::{delta_net_layer_forward, DeltaNetState, DeltaNetWeights};
use super::moe::{FfnWeights, MoeWeights};

/// The token-mixing half of a layer.
#[derive(Debug, Clone)]
pub enum LayerMix {
    DeltaNet(DeltaNetWeights),
    Attention(AttentionWeights),
}

/// The FFN half of a layer.
#[derive(Debug, Clone)]
pub enum LayerFfn {
    Dense(FfnWeights),
    Moe(MoeWeights),
}

/// One decoder layer.
#[derive(Debug, Clone)]
pub struct Qwen35Layer {
    /// `[hidden]` RMS weights.
    pub attn_norm: Tensor,
    pub post_attn_norm: Tensor,
    pub mix: LayerMix,
    pub ffn: LayerFfn,
}

/// Per-layer carried state for one sequence.
#[derive(Debug)]
pub enum LayerState {
    DeltaNet(DeltaNetState),
    Attention(AttentionState),
}

impl LayerState {
    /// An independent copy — see [`DeltaNetState::snapshot`].
    pub fn snapshot(&self) -> Result<Self> {
        Ok(match self {
            Self::DeltaNet(s) => Self::DeltaNet(s.snapshot()?),
            // Attention state is a growing KV list that this path only ever
            // appends to, so its handles are never written through.
            Self::Attention(s) => Self::Attention(s.clone()),
        })
    }
}

/// All carried state for one sequence: exactly what a turn-seal snapshot
/// captures (the DeltaNet entries) alongside the sealed KV (the attention
/// entries, owned by the paged cache in production).
///
/// Not `Clone`, because half of it is written in place — see
/// [`DeltaNetState`]. [`Self::snapshot`] is what "take a copy of the session"
/// means.
#[derive(Debug)]
pub struct SessionState {
    pub layers: Vec<LayerState>,
}

impl SessionState {
    /// A copy that will not move when the original does.
    pub fn snapshot(&self) -> Result<Self> {
        Ok(Self {
            layers: self
                .layers
                .iter()
                .map(|l| l.snapshot())
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

/// The reference model.
#[derive(Debug, Clone)]
pub struct Qwen35Model {
    pub cfg: Qwen35Config,
    /// `[vocab, hidden]`.
    pub embed: Tensor,
    pub layers: Vec<Qwen35Layer>,
    /// `[hidden]`.
    pub final_norm: Tensor,
    /// `[vocab, hidden]` (tied to `embed` when the checkpoint has no
    /// `output.weight`).
    pub lm_head: Tensor,
    pub rope: RopeTables,
}

fn rms_norm_row(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let ms = x.sqr()?.mean_keepdim(candle::D::Minus1)?;
    let denom = (ms + eps)?.sqrt()?;
    x.broadcast_div(&denom)?.broadcast_mul(weight)
}

impl Qwen35Model {
    /// Fresh (sequence-start) state for every layer.
    pub fn new_session(&self) -> Result<SessionState> {
        let dev = self.embed.device();
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(match &layer.mix {
                LayerMix::DeltaNet(_) => {
                    LayerState::DeltaNet(DeltaNetState::zeros(&self.cfg.delta_net, dev)?)
                }
                LayerMix::Attention(_) => LayerState::Attention(AttentionState::empty()),
            });
        }
        Ok(SessionState { layers })
    }

    /// Forward a token segment, consuming and replacing the carried state.
    /// Returns logits `[T, vocab]`.
    pub fn forward(&self, tokens: &[u32], state: &mut SessionState) -> Result<Tensor> {
        let dev = self.embed.device();
        let ids = Tensor::from_vec(tokens.to_vec(), (tokens.len(),), dev)?;
        let mut x = self.embed.index_select(&ids, 0)?; // [T, hidden]
        let eps = self.cfg.rms_norm_eps;

        for (li, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            let h = rms_norm_row(&x, &layer.attn_norm, eps)?;
            let mixed = match (&layer.mix, &mut state.layers[li]) {
                (LayerMix::DeltaNet(w), LayerState::DeltaNet(s)) => {
                    // Written into, so there is nothing to take out and put
                    // back — the swap-with-zeros this replaced existed only to
                    // move a value the layer never needed to own.
                    delta_net_layer_forward(&h, w, &self.cfg.delta_net, s, eps)?
                }
                (LayerMix::Attention(w), LayerState::Attention(s)) => {
                    let taken = std::mem::replace(s, AttentionState::empty());
                    let (y, s_new) = attention_layer_forward(
                        &h,
                        w,
                        taken,
                        &self.rope,
                        self.cfg.num_attention_heads,
                        self.cfg.num_kv_heads,
                        self.cfg.attn_head_dim,
                        eps,
                    )?;
                    *s = s_new;
                    y
                }
                _ => candle::bail!(
                    "layer {li}: state kind does not match layer kind — the session \
                     was created for a different layer schedule"
                ),
            };
            x = mixed.add(&residual)?;

            let ffn_residual = x.clone();
            let h2 = rms_norm_row(&x, &layer.post_attn_norm, eps)?;
            let y = match &layer.ffn {
                LayerFfn::Dense(w) => w.forward(&h2)?,
                LayerFfn::Moe(w) => w.forward(&h2)?,
            };
            x = y.add(&ffn_residual)?;
        }

        let x = rms_norm_row(&x, &self.final_norm, eps)?;
        x.matmul(&self.lm_head.t()?)
    }

    /// The layer schedule this model was assembled with (for state-shape
    /// validation by callers).
    pub fn layer_kinds(&self) -> Vec<LayerKind> {
        self.layers
            .iter()
            .map(|l| match l.mix {
                LayerMix::DeltaNet(_) => LayerKind::DeltaNet,
                LayerMix::Attention(_) => LayerKind::Attention,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::delta_net::DeltaNetDims;
    use super::super::moe::MoeWeights;
    use candle::Device;

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

    /// A 4-layer tiny hybrid: DN, DN(+MoE), DN, Attention — every layer kind
    /// and both FFN kinds in one stack.
    fn tiny_model(dev: &Device) -> Qwen35Model {
        let hidden = 8usize;
        let vocab = 13usize;
        let ffn = 12usize;
        let dims = DeltaNetDims {
            head_dim: 4,
            n_k_heads: 2,
            n_v_heads: 4,
            conv_kernel: 3,
        };
        let sc = |t: Tensor| t.affine(0.15, 0.).unwrap();
        let norm1 = |seed: u64| lcg_tensor(&[hidden], seed, dev).affine(0.2, 1.0).unwrap();

        let dn = |seed: u64| {
            crate::models::delta_net::DeltaNetWeights {
                wqkv: sc(lcg_tensor(&[dims.conv_dim(), hidden], seed, dev)),
                wz: sc(lcg_tensor(&[dims.value_dim(), hidden], seed + 1, dev)),
                w_beta: sc(lcg_tensor(&[dims.n_v_heads, hidden], seed + 2, dev)),
                w_alpha: sc(lcg_tensor(&[dims.n_v_heads, hidden], seed + 3, dev)),
                dt_bias: sc(lcg_tensor(&[dims.n_v_heads], seed + 4, dev)),
                a: lcg_tensor(&[dims.n_v_heads], seed + 5, dev)
                    .abs()
                    .unwrap()
                    .affine(-1.0, -0.05)
                    .unwrap(),
                conv: sc(lcg_tensor(&[dims.conv_dim(), dims.conv_kernel], seed + 6, dev)),
                norm: lcg_tensor(&[dims.head_dim], seed + 7, dev)
                    .affine(0.2, 1.0)
                    .unwrap(),
                w_out: sc(lcg_tensor(&[hidden, dims.value_dim()], seed + 8, dev)),
            }
        };
        let ffn_w = |seed: u64| super::super::moe::FfnWeights {
            gate: sc(lcg_tensor(&[ffn, hidden], seed, dev)),
            up: sc(lcg_tensor(&[ffn, hidden], seed + 1, dev)),
            down: sc(lcg_tensor(&[hidden, ffn], seed + 2, dev)),
        };
        let (n_head, n_kv, d_attn) = (2usize, 1usize, 4usize);
        let attn = super::super::attention::AttentionWeights {
            wq: sc(lcg_tensor(&[2 * d_attn * n_head, hidden], 901, dev)),
            wk: sc(lcg_tensor(&[d_attn * n_kv, hidden], 902, dev)),
            wv: sc(lcg_tensor(&[d_attn * n_kv, hidden], 903, dev)),
            wo: sc(lcg_tensor(&[hidden, d_attn * n_head], 904, dev)),
            q_norm: lcg_tensor(&[d_attn], 905, dev).affine(0.2, 1.0).unwrap(),
            k_norm: lcg_tensor(&[d_attn], 906, dev).affine(0.2, 1.0).unwrap(),
        };
        let moe = MoeWeights {
            router: lcg_tensor(&[3, hidden], 950, dev),
            experts: vec![ffn_w(960), ffn_w(970), ffn_w(980)],
            shared: ffn_w(990),
            shared_gate: lcg_tensor(&[1, hidden], 951, dev),
            n_experts_used: 2,
            norm_topk_prob: true,
            weights_scale: 1.0,
        };

        let layer = |mix: LayerMix, ffn: LayerFfn, seed: u64| Qwen35Layer {
            attn_norm: norm1(seed),
            post_attn_norm: norm1(seed + 1),
            mix,
            ffn,
        };
        let cfg = Qwen35Config {
            vocab_size: vocab,
            hidden_size: hidden,
            intermediate_size: ffn,
            num_layers: 4,
            layer_kinds: vec![
                LayerKind::DeltaNet,
                LayerKind::DeltaNet,
                LayerKind::DeltaNet,
                LayerKind::Attention,
            ],
            num_attention_heads: n_head,
            num_kv_heads: n_kv,
            attn_head_dim: d_attn,
            // Partial rotary, like the published family: half the head
            // rotates, the rest passes through.
            rope_dim: d_attn / 2,
            rope_sections: [d_attn / 4, 0, 0, 0],
            rope_theta: 1e6,
            rms_norm_eps: 1e-6,
            delta_net: dims,
            moe: None,
            num_mtp_layers: 0,
            max_position_embeddings: 64,
        };
        let embed = lcg_tensor(&[vocab, hidden], 800, dev).affine(0.4, 0.).unwrap();
        Qwen35Model {
            layers: vec![
                layer(LayerMix::DeltaNet(dn(100)), LayerFfn::Dense(ffn_w(110)), 120),
                layer(LayerMix::DeltaNet(dn(200)), LayerFfn::Moe(moe), 220),
                layer(LayerMix::DeltaNet(dn(300)), LayerFfn::Dense(ffn_w(310)), 320),
                layer(LayerMix::Attention(attn), LayerFfn::Dense(ffn_w(410)), 420),
            ],
            final_norm: norm1(500),
            lm_head: lcg_tensor(&[vocab, hidden], 810, dev).affine(0.4, 0.).unwrap(),
            rope: RopeTables::new(d_attn / 2, 1e6, 64, dev).unwrap(),
            embed,
            cfg,
        }
    }

    #[test]
    fn whole_model_segments_equal_one_shot() {
        // The end-to-end contract: prefill-then-decode from carried state
        // reproduces the one-shot logits across a hybrid stack containing
        // DeltaNet, attention, dense-FFN and MoE layers. This is the model
        // spine of the turn-seal snapshot and resume design.
        let dev = dev();
        let model = tiny_model(&dev);
        let tokens: Vec<u32> = vec![3, 1, 7, 12, 5, 0, 9, 4, 2];

        let mut s_full = model.new_session().unwrap();
        let logits_full = model.forward(&tokens, &mut s_full).unwrap();

        let mut s_seg = model.new_session().unwrap();
        let l1 = model.forward(&tokens[..4], &mut s_seg).unwrap();
        let l2 = model.forward(&tokens[4..7], &mut s_seg).unwrap();
        let l3 = model.forward(&tokens[7..], &mut s_seg).unwrap();
        let logits_seg = Tensor::cat(&[l1, l2, l3], 0).unwrap();

        assert_close(&logits_full, &logits_seg, 5e-5, "model segmented logits");
    }

    #[test]
    fn a_cloned_state_resumes_identically() {
        // Snapshot-and-resume in miniature: clone the state mid-stream (the
        // snapshot), keep generating on the original, then resume from the
        // clone — both continuations must be identical.
        let dev = dev();
        let model = tiny_model(&dev);
        let mut state = model.new_session().unwrap();
        model.forward(&[3, 1, 7, 12], &mut state).unwrap();

        // `snapshot`, not `clone` — the state is written in place, so a shared
        // handle would follow the original forward instead of holding still.
        let snapshot = state.snapshot().unwrap();
        let cont: Vec<u32> = vec![5, 0, 9];
        let a = model.forward(&cont, &mut state).unwrap();
        let mut resumed = snapshot;
        let b = model.forward(&cont, &mut resumed).unwrap();
        assert_close(&a, &b, 0.0, "resumed continuation");
    }

    #[test]
    fn mismatched_state_schedule_is_refused() {
        let dev = dev();
        let model = tiny_model(&dev);
        let mut state = model.new_session().unwrap();
        // Swap a DeltaNet slot for an attention slot: the forward must refuse,
        // not corrupt.
        state.layers[0] = LayerState::Attention(AttentionState::empty());
        let err = model.forward(&[1, 2], &mut state).unwrap_err();
        assert!(err.to_string().contains("does not match layer kind"));
    }

    #[test]
    fn state_kinds_follow_the_schedule() {
        let dev = dev();
        let model = tiny_model(&dev);
        let state = model.new_session().unwrap();
        for (ls, kind) in state.layers.iter().zip(model.layer_kinds()) {
            match (ls, kind) {
                (LayerState::DeltaNet(_), LayerKind::DeltaNet) => {}
                (LayerState::Attention(_), LayerKind::Attention) => {}
                other => panic!("state/schedule mismatch: {other:?}"),
            }
        }
    }
}
