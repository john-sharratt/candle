//! Model assembly: the qwen4exp hybrid decoder stack, per-session state, and
//! the batched reference forward — the oracle of
//! `docs/qwen38_flash_next.md` §0.6.
//!
//! Layer skeleton (per `qwen4exp.cpp`; §12.3–§12.7 of the design doc): the
//! wide residual `[T, hc, n_embd]` replaces the plain stream *and* the layer
//! norms —
//!
//! ```text
//!   res = 4 copies of embed(x)
//!   per layer:
//!     res = ple(res)                      (layer 1 only, before its mixer)
//!     (h, inj) = hc_mix(res, hc_attn)     grouped norm + low-rank gate + mean
//!     y = mix(h)                          mix ∈ {GDN(sigmoid z), QSA attention}
//!     res = hc_combine(res, y, inj)
//!     (h, inj) = hc_mix(res, hc_ffn)
//!     y = moe(h)                          512 experts top-10 + gated shared
//!     res = hc_combine(res, y, inj)
//!   logits = lm_head · hc_mix(res, out_hc)   (the head mix IS the output norm)
//! ```
//!
//! `forward_batched` packs every sequence's tokens into one `[ΣT, …]` row
//! block: embeddings, HC mixes, routing, expert GEMMs and the head run over
//! the packed rows in one op each; only the three stateful mixers — the GDN
//! scan, attention + QSA selection, and the PLE conv — iterate per sequence.
//! That is the row-packed shape the production wave runs, which is what makes
//! the oracle and the engine comparable.

use std::collections::BTreeMap;

use candle::{Result, Tensor};

use super::config::Qwen4ExpConfig;
use super::hyper::{hc_combine, hc_mix, HcWeights};
use super::ple::{ple_apply, ple_row_ids, PleState, PleWeights};
use super::qsa::{qsa_selection_mask, IndexState, IndexerWeights};
use crate::models::delta_net::{delta_net_layer_forward, DeltaNetState, DeltaNetWeights, ZGate};
use crate::models::qwen35::attention::{
    gated_attention_core, AttentionState, AttentionWeights, RopeTables,
};
use crate::models::qwen35::moe::{route, FfnWeights};

/// Where a layer's routed experts come from: dequantized on demand, per
/// routed expert — the checkpoint's expert slabs never fit memory in F32.
/// The disk implementation lives in the loader; tests hold them in memory.
pub trait ExpertSource: Send + Sync {
    /// Expert `e` of layer `layer` as F32 [`FfnWeights`].
    fn expert(&self, layer: usize, e: usize) -> Result<FfnWeights>;
}

/// Where the 320M-row PLE table's rows come from. The disk implementation
/// reads 170-byte quantized records through the bounded RAM row cache;
/// tests hold a small table.
pub trait PleSource: Send + Sync {
    /// Gather rows `ids` as one F32 `[ids.len(), head_dim]` tensor.
    fn rows(&self, ids: &[u32]) -> Result<Tensor>;

    /// Cache instrumentation, where a cache exists (§8 item 4). `None` for
    /// sources with nothing between them and their table.
    fn cache_stats(&self) -> Option<super::ple_cache::PleCacheStats> {
        None
    }
}

/// The token-mixing half of a layer.
#[derive(Debug)]
pub enum LayerMix {
    DeltaNet(DeltaNetWeights),
    Attention {
        attn: AttentionWeights,
        indexer: IndexerWeights,
        /// QSA block-compression ratio; 0 attends densely.
        compress_ratio: usize,
    },
}

/// One decoder layer. The MoE weights that fit memory live here; the routed
/// expert slabs live behind [`ExpertSource`].
pub struct Qwen4ExpLayer {
    pub hc_attn: HcWeights,
    pub hc_ffn: HcWeights,
    pub mix: LayerMix,
    /// `[n_experts, hidden]`.
    pub router: Tensor,
    pub shared: FfnWeights,
    /// `[1, hidden]`.
    pub shared_gate: Tensor,
}

/// Per-layer carried state for one sequence.
#[derive(Debug)]
pub enum LayerState {
    DeltaNet(DeltaNetState),
    Attention { kv: AttentionState, idx: IndexState },
}

/// All carried state for one sequence: the GDN states, the attention KV +
/// index caches, and the PLE conv tail + hash window — the three state
/// classes the turn-seal snapshot must eventually cover (§6.3).
pub struct SessionState {
    pub layers: Vec<LayerState>,
    pub ple: PleState,
    /// Set while a forward is in flight over this session, cleared when it
    /// completes. A state that is still marked on entry belonged to a forward
    /// that returned an error partway through.
    ///
    /// The forward advances carried state **in place** as it walks the layers:
    /// `ple.prev` is advanced before the row gather that can fail, and each
    /// attention layer's history is `mem::replace`d out before
    /// `gated_attention_core` — which consumes it, so a failure there cannot
    /// put it back. The session is then internally inconsistent: some layers
    /// hold this call's state, one holds none, the rest hold the last call's.
    ///
    /// Nothing downstream would notice. A dense layer (`compress_ratio == 0`,
    /// so no selection) does not check its history's width, so the next call
    /// attends a zero-length history and returns plausible, silently wrong
    /// reference logits — from the oracle, which is what every other path is
    /// diffed against. A harness that logs an error and reuses the session (an
    /// oracle sweep over cases, a retry after a transient expert read) hits
    /// exactly this. So a poisoned session refuses to run again rather than
    /// pretending it recovered.
    poisoned: bool,
}

/// The reference model.
pub struct Qwen4ExpModel {
    pub cfg: Qwen4ExpConfig,
    /// `[vocab, hidden]` F32.
    pub embed: Tensor,
    pub layers: Vec<Qwen4ExpLayer>,
    /// The PLE layer's projection weights (at `cfg.ple.layer`).
    pub ple_w: PleWeights,
    /// The final hyper-connection mix — the output norm (no inject).
    pub out_hc: HcWeights,
    /// `[vocab, hidden]` F32.
    pub lm_head: Tensor,
    pub rope: RopeTables,
    pub experts: Box<dyn ExpertSource>,
    pub ple_table: Box<dyn PleSource>,
}

impl Qwen4ExpModel {
    /// Fresh (sequence-start) state.
    pub fn new_session(&self) -> Result<SessionState> {
        let dev = self.embed.device();
        let hc_dim = self.cfg.hc.dim(self.cfg.hidden_size);
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(match &layer.mix {
                LayerMix::DeltaNet(_) => {
                    LayerState::DeltaNet(DeltaNetState::zeros(&self.cfg.delta_net, dev)?)
                }
                LayerMix::Attention { .. } => LayerState::Attention {
                    kv: AttentionState::empty(),
                    idx: IndexState::empty(),
                },
            });
        }
        Ok(SessionState {
            layers,
            ple: PleState::zeros(&self.cfg.ple, hc_dim, dev)?,
            poisoned: false,
        })
    }

    /// The 512-expert MoE block over packed rows `[N, hidden]`: softmax
    /// routing, top-k renorm, on-demand expert dequant, gated shared expert.
    /// Semantics identical to `qwen35::moe::MoeWeights::forward`; the
    /// accumulation is expert-major (one dequant per routed expert per call)
    /// instead of token-major, because the experts live on disk.
    fn moe_forward(&self, li: usize, x: &Tensor) -> Result<Tensor> {
        let layer = &self.layers[li];
        let (n, hidden) = x.dims2()?;
        let logits = x.matmul(&layer.router.t()?)?;
        let routes = route(
            &logits,
            self.cfg.moe.n_experts_used,
            self.cfg.moe.norm_topk_prob,
            1.0,
        )?;

        // Expert-major grouping, BTreeMap so the accumulation order is a
        // deterministic function of the routing, not of a hash seed.
        let mut by_expert: BTreeMap<usize, Vec<(usize, f32)>> = BTreeMap::new();
        for (row, r) in routes.iter().enumerate() {
            for (&e, &w) in r.experts.iter().zip(r.weights.iter()) {
                by_expert.entry(e).or_default().push((row, w));
            }
        }

        let mut acc = Tensor::zeros((n, hidden), x.dtype(), x.device())?;
        for (e, rows) in by_expert {
            let w = self.experts.expert(li, e)?;
            let ids = Tensor::from_vec(
                rows.iter().map(|&(r, _)| r as u32).collect::<Vec<u32>>(),
                (rows.len(),),
                x.device(),
            )?;
            let xe = x.index_select(&ids, 0)?;
            let y = w.forward(&xe)?; // [n_e, hidden]
            let wts = Tensor::from_vec(
                rows.iter().map(|&(_, w)| w).collect::<Vec<f32>>(),
                (rows.len(), 1),
                x.device(),
            )?;
            acc = acc.index_add(&ids, &y.broadcast_mul(&wts)?, 0)?;
        }

        let shared = layer.shared.forward(x)?;
        let gate = candle_nn::ops::sigmoid(&x.matmul(&layer.shared_gate.t()?)?)?;
        acc.add(&shared.broadcast_mul(&gate)?)
    }

    /// Forward a batch of token segments, one per session, consuming and
    /// replacing each session's carried state. Returns per-session logits
    /// `[T_i, vocab]`.
    pub fn forward_batched(
        &self,
        tokens: &[&[u32]],
        states: &mut [SessionState],
    ) -> Result<Vec<Tensor>> {
        if tokens.len() != states.len() {
            candle::bail!(
                "qwen4exp: {} token segments against {} sessions",
                tokens.len(),
                states.len()
            );
        }
        if tokens.iter().any(|t| t.is_empty()) {
            candle::bail!("qwen4exp: an empty token segment");
        }
        // The argument checks above touch no state, so they leave every session
        // reusable. Past this point the walk advances carried state in place and
        // a mid-forward error cannot be unwound — see `SessionState::poisoned`.
        if let Some(i) = states.iter().position(|s| s.poisoned) {
            candle::bail!(
                "qwen4exp: session {i} is poisoned — an earlier forward_batched returned an \
                 error partway through the layer walk, leaving its PLE history and one \
                 layer's attention history inconsistent with the rest. Reusing it would \
                 attend a zero-length history and return plausible, silently wrong logits. \
                 Start a fresh session with `new_session`."
            );
        }
        for s in states.iter_mut() {
            s.poisoned = true;
        }
        let out = self.forward_batched_inner(tokens, states);
        if out.is_ok() {
            for s in states.iter_mut() {
                s.poisoned = false;
            }
        }
        out
    }

    fn forward_batched_inner(
        &self,
        tokens: &[&[u32]],
        states: &mut [SessionState],
    ) -> Result<Vec<Tensor>> {
        let dev = self.embed.device();
        let eps = self.cfg.rms_norm_eps;
        let hc = self.cfg.hc.count;
        let n_embd = self.cfg.hidden_size;

        // Row spans of the packed block, in session order.
        let spans: Vec<(usize, usize)> = {
            let mut off = 0usize;
            tokens
                .iter()
                .map(|t| {
                    let s = (off, t.len());
                    off += t.len();
                    s
                })
                .collect()
        };
        let total: usize = tokens.iter().map(|t| t.len()).sum();

        // Packed embedding: [ΣT, hidden] → wide residual [ΣT, hc, n_embd].
        let flat_ids: Vec<u32> = tokens.iter().flat_map(|t| t.iter().copied()).collect();
        let ids = Tensor::from_vec(flat_ids, (total,), dev)?;
        let x = self.embed.index_select(&ids, 0)?;
        let mut res_hc = x
            .reshape((total, 1, n_embd))?
            .broadcast_as((total, hc, n_embd))?
            .contiguous()?;

        for (li, layer) in self.layers.iter().enumerate() {
            if li == self.cfg.ple.layer {
                // PLE before this layer's mixer: hash + gather packed, the
                // keyed injection and conv per sequence (the conv carries).
                let mut parts = Vec::with_capacity(tokens.len());
                for ((seq, state), &(start, len)) in
                    tokens.iter().zip(states.iter_mut()).zip(&spans)
                {
                    let row_ids = ple_row_ids(&self.cfg.ple, seq, &mut state.ple.prev);
                    let flat: Vec<u32> = row_ids.into_iter().flatten().collect();
                    let emb = self
                        .ple_table
                        .rows(&flat)?
                        .reshape((len, self.cfg.hidden_size))?;
                    let rows = res_hc.narrow(0, start, len)?;
                    parts.push(ple_apply(
                        &rows,
                        &emb,
                        &self.ple_w,
                        &self.cfg.ple,
                        &mut state.ple,
                        eps,
                        // The oracle never speculates, so it has nothing to
                        // rewind and nothing to capture.
                        None,
                    )?);
                }
                res_hc = Tensor::cat(&parts, 0)?;
            }

            // ── Token mixer under the first HC module ────────────────────
            let (h, inject) = hc_mix(&res_hc, &layer.hc_attn, eps)?;
            let inject = inject.expect("layer HC modules carry an inject");

            let y = match &layer.mix {
                LayerMix::DeltaNet(w) => {
                    // Projections run packed inside the mixer per sequence —
                    // the reference single-span form, one call per session.
                    let mut parts = Vec::with_capacity(tokens.len());
                    for (state, &(start, len)) in states.iter_mut().zip(&spans) {
                        let LayerState::DeltaNet(s) = &mut state.layers[li] else {
                            candle::bail!("layer {li}: state kind mismatch");
                        };
                        let rows = h.narrow(0, start, len)?;
                        parts.push(delta_net_layer_forward(
                            &rows,
                            w,
                            &self.cfg.delta_net,
                            s,
                            eps,
                            ZGate::Sigmoid,
                        )?);
                    }
                    Tensor::cat(&parts, 0)?
                }
                LayerMix::Attention {
                    attn,
                    indexer,
                    compress_ratio,
                } => {
                    // Projections packed once over all rows; the stateful
                    // core (KV append, QSA selection, causal softmax) per
                    // sequence.
                    let qg = h.matmul(&attn.wq.t()?)?;
                    let k = h.matmul(&attn.wk.t()?)?;
                    let v = h.matmul(&attn.wv.t()?)?;
                    let mut parts = Vec::with_capacity(tokens.len());
                    for (state, &(start, len)) in states.iter_mut().zip(&spans) {
                        let LayerState::Attention { kv, idx } = &mut state.layers[li] else {
                            candle::bail!("layer {li}: state kind mismatch");
                        };
                        let rows = h.narrow(0, start, len)?;
                        let sel = if *compress_ratio > 0 {
                            qsa_selection_mask(
                                &rows,
                                indexer,
                                idx,
                                &self.rope,
                                *compress_ratio,
                                &self.cfg.indexer,
                                eps,
                            )?
                        } else {
                            None
                        };
                        let taken = std::mem::replace(kv, AttentionState::empty());
                        let (gated, kv_new) = gated_attention_core(
                            &qg.narrow(0, start, len)?,
                            &k.narrow(0, start, len)?,
                            &v.narrow(0, start, len)?,
                            &attn.q_norm,
                            &attn.k_norm,
                            taken,
                            &self.rope,
                            self.cfg.num_attention_heads,
                            self.cfg.num_kv_heads,
                            self.cfg.attn_head_dim,
                            eps,
                            sel.as_ref(),
                        )?;
                        *kv = kv_new;
                        parts.push(gated.reshape((
                            len,
                            self.cfg.num_attention_heads * self.cfg.attn_head_dim,
                        ))?);
                    }
                    // The output projection runs packed over every session.
                    Tensor::cat(&parts, 0)?.matmul(&attn.wo.t()?)?
                }
            };
            res_hc = hc_combine(&res_hc, &y, &inject)?;

            // ── MoE under the second HC module ───────────────────────────
            let (h2, inject2) = hc_mix(&res_hc, &layer.hc_ffn, eps)?;
            let inject2 = inject2.expect("layer HC modules carry an inject");
            let y2 = self.moe_forward(li, &h2)?;
            res_hc = hc_combine(&res_hc, &y2, &inject2)?;
        }

        // The head mix IS the output norm; logits over packed rows, then
        // split back per session.
        let (mixed, _) = hc_mix(&res_hc, &self.out_hc, eps)?;
        let logits = mixed.matmul(&self.lm_head.t()?)?;
        spans
            .iter()
            .map(|&(start, len)| logits.narrow(0, start, len))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::delta_net::{DeltaNetDims, LayerKind};
    use crate::models::qwen35::config::MoeConfig;
    use candle::Device;

    use super::super::config::{HcConfig, IndexerConfig, PleConfig};

    fn dev() -> Device {
        Device::Cpu
    }

    fn lcg_tensor(shape: &[usize], seed: u64, dev: &Device) -> Tensor {
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

    struct MemExperts {
        // [layer][expert]
        experts: Vec<Vec<FfnWeights>>,
    }
    impl ExpertSource for MemExperts {
        fn expert(&self, layer: usize, e: usize) -> Result<FfnWeights> {
            Ok(self.experts[layer][e].clone())
        }
    }

    struct MemPle {
        /// `[rows, head_dim]`.
        table: Tensor,
    }
    impl PleSource for MemPle {
        fn rows(&self, ids: &[u32]) -> Result<Tensor> {
            let idx = Tensor::from_vec(ids.to_vec(), (ids.len(),), self.table.device())?;
            self.table.index_select(&idx, 0)
        }
    }

    /// A 4-layer tiny qwen4exp: GDN, GDN(+PLE), GDN, Attention(+QSA) — every
    /// new component in one stack at toy width.
    fn tiny_model(dev: &Device) -> Qwen4ExpModel {
        let hidden = 8usize;
        let vocab = 13usize;
        let hcc = HcConfig {
            count: 2,
            low_rank: 3,
        };
        let hc_dim = hcc.dim(hidden);
        let dims = DeltaNetDims {
            head_dim: 4,
            n_k_heads: 2,
            n_v_heads: 4,
            conv_kernel: 3,
        };
        let (n_head, n_kv, d_attn) = (2usize, 1usize, 4usize);
        let idx_cfg = IndexerConfig {
            n_heads: 2,
            head_dim: 4,
            // Tiny budget so a modest context engages real selection.
            top_k: 4,
        };
        let ple_cfg = PleConfig {
            layer: 1,
            ngram_size: 3,
            heads_per_ngram: 2,
            conv_kernel: 3,
            eos_token_id: 12,
            multipliers: vec![
                0x9E37_79B9_7F4A_7C15,
                0xC2B2_AE3D_27D4_EB4F,
                0x1656_67B1_9E37_79F9,
            ],
            head_offsets: vec![0, 40, 80, 120],
            head_vocab_sizes: vec![40, 40, 40, 40],
            head_dim: 2, // 4 heads × 2 = hidden 8
        };
        let sc = |t: Tensor| t.affine(0.15, 0.).unwrap();
        let norm1 =
            |shape: usize, seed: u64| lcg_tensor(&[shape], seed, dev).affine(0.2, 1.0).unwrap();

        // `down` carries the inject rows stacked beneath the gate's, which is
        // the layout both loaders build — see `HcWeights::down`.
        let hcw = |seed: u64, inject: bool| HcWeights {
            norm: norm1(hc_dim, seed),
            down: sc(lcg_tensor(
                &[hcc.low_rank + if inject { hcc.count } else { 0 }, hc_dim],
                seed + 1,
                dev,
            )),
            up: sc(lcg_tensor(&[hc_dim, hcc.low_rank], seed + 2, dev)),
        };
        let dn = |seed: u64| DeltaNetWeights {
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
            conv: sc(lcg_tensor(
                &[dims.conv_dim(), dims.conv_kernel],
                seed + 6,
                dev,
            )),
            norm: norm1(dims.head_dim, seed + 7),
            w_out: sc(lcg_tensor(&[hidden, dims.value_dim()], seed + 8, dev)),
        };
        let ffn = |seed: u64, width: usize| FfnWeights {
            gate: sc(lcg_tensor(&[width, hidden], seed, dev)),
            up: sc(lcg_tensor(&[width, hidden], seed + 1, dev)),
            down: sc(lcg_tensor(&[hidden, width], seed + 2, dev)),
        };
        let n_experts = 3usize;
        let layer = |mix: LayerMix, seed: u64| Qwen4ExpLayer {
            hc_attn: hcw(seed, true),
            hc_ffn: hcw(seed + 10, true),
            mix,
            router: lcg_tensor(&[n_experts, hidden], seed + 20, dev),
            shared: ffn(seed + 30, 6),
            shared_gate: lcg_tensor(&[1, hidden], seed + 33, dev),
        };
        let attn = AttentionWeights {
            wq: sc(lcg_tensor(&[2 * d_attn * n_head, hidden], 901, dev)),
            wk: sc(lcg_tensor(&[d_attn * n_kv, hidden], 902, dev)),
            wv: sc(lcg_tensor(&[d_attn * n_kv, hidden], 903, dev)),
            wo: sc(lcg_tensor(&[hidden, d_attn * n_head], 904, dev)),
            q_norm: norm1(d_attn, 905),
            k_norm: norm1(d_attn, 906),
        };
        let indexer = IndexerWeights {
            q_proj: sc(lcg_tensor(
                &[idx_cfg.n_heads * idx_cfg.head_dim, hidden],
                911,
                dev,
            )),
            k_proj: sc(lcg_tensor(&[idx_cfg.head_dim, hidden], 912, dev)),
            q_norm: norm1(idx_cfg.head_dim, 913),
            k_norm: norm1(idx_cfg.head_dim, 914),
        };
        let ple_w = PleWeights {
            key: sc(lcg_tensor(&[hc_dim, hidden], 921, dev)),
            value: sc(lcg_tensor(&[hidden, hidden], 922, dev)),
            norm_key: norm1(hc_dim, 923),
            norm_query: norm1(hc_dim, 924),
            norm_conv: norm1(hc_dim, 925),
            conv: sc(lcg_tensor(&[hc_dim, ple_cfg.conv_kernel], 926, dev)),
        };
        let experts = MemExperts {
            experts: (0..4usize)
                .map(|l| {
                    (0..n_experts)
                        .map(|e| ffn((1000 + 100 * l + 10 * e) as u64, 6))
                        .collect()
                })
                .collect(),
        };
        let ple_table = MemPle {
            table: lcg_tensor(&[160, ple_cfg.head_dim], 930, dev)
                .affine(0.4, 0.)
                .unwrap(),
        };

        let cfg = Qwen4ExpConfig {
            vocab_size: vocab,
            hidden_size: hidden,
            num_layers: 4,
            // The oracle carries no draft head: this fixture is the reference
            // forward, and a head only ever proposes tokens the trunk then
            // scores — there is nothing for it to be the reference *of*.
            num_mtp_layers: 0,
            layer_kinds: vec![
                LayerKind::DeltaNet,
                LayerKind::DeltaNet,
                LayerKind::DeltaNet,
                LayerKind::Attention,
            ],
            num_attention_heads: n_head,
            num_kv_heads: n_kv,
            attn_head_dim: d_attn,
            rope_dim: d_attn / 2,
            rope_sections: [d_attn / 4, 0, 0, 0],
            rope_theta: 1e6,
            rms_norm_eps: 1e-6,
            delta_net: dims,
            moe: MoeConfig {
                n_experts,
                n_experts_used: 2,
                expert_ffn_size: 6,
                shared_expert_ffn_size: 6,
                norm_topk_prob: true,
            },
            hc: hcc,
            indexer: idx_cfg,
            compress_ratios: vec![0, 0, 0, 2],
            head_compress_ratio: None,
            ple: ple_cfg,
            max_position_embeddings: 64,
        };
        Qwen4ExpModel {
            embed: lcg_tensor(&[vocab, hidden], 800, dev)
                .affine(0.4, 0.)
                .unwrap(),
            layers: vec![
                layer(LayerMix::DeltaNet(dn(100)), 140),
                layer(LayerMix::DeltaNet(dn(200)), 240),
                layer(LayerMix::DeltaNet(dn(300)), 340),
                layer(
                    LayerMix::Attention {
                        attn,
                        indexer,
                        compress_ratio: cfg.compress_ratios[3],
                    },
                    440,
                ),
            ],
            ple_w,
            out_hc: hcw(950, false),
            lm_head: lcg_tensor(&[vocab, hidden], 810, dev)
                .affine(0.4, 0.)
                .unwrap(),
            rope: RopeTables::new(d_attn / 2, 1e6, 64, dev).unwrap(),
            experts: Box::new(experts),
            ple_table: Box::new(ple_table),
            cfg,
        }
    }

    #[test]
    fn whole_model_segments_equal_one_shot() {
        // Prefill-then-decode from carried state ≡ one-shot, across GDN,
        // PLE (hash window + conv tail), QSA (index cache) and the wide
        // residual. The spine of turn sealing for this model's THREE carried
        // state classes.
        let dev = dev();
        let model = tiny_model(&dev);
        let tokens: Vec<u32> = vec![3, 1, 7, 12, 5, 0, 9, 4, 2, 6, 11, 8, 10, 1, 3];

        let mut s_full = model.new_session().unwrap();
        let full = model
            .forward_batched(&[&tokens], std::slice::from_mut(&mut s_full))
            .unwrap()
            .remove(0);

        let mut s_seg = model.new_session().unwrap();
        let l1 = model
            .forward_batched(&[&tokens[..4]], std::slice::from_mut(&mut s_seg))
            .unwrap()
            .remove(0);
        let l2 = model
            .forward_batched(&[&tokens[4..9]], std::slice::from_mut(&mut s_seg))
            .unwrap()
            .remove(0);
        let l3 = model
            .forward_batched(&[&tokens[9..]], std::slice::from_mut(&mut s_seg))
            .unwrap()
            .remove(0);
        let seg = Tensor::cat(&[l1, l2, l3], 0).unwrap();
        assert_close(&full, &seg, 5e-5, "segmented ≡ one-shot");
    }

    #[test]
    fn batching_changes_nothing() {
        // A session's logits must not depend on what else is in the batch —
        // the ×1 / ×N rung of the gate, in miniature.
        let dev = dev();
        let model = tiny_model(&dev);
        let a: Vec<u32> = vec![3, 1, 7, 12, 5];
        let b: Vec<u32> = vec![9, 4, 2, 6, 11, 8, 10];

        let mut sa = model.new_session().unwrap();
        let solo_a = model
            .forward_batched(&[&a], std::slice::from_mut(&mut sa))
            .unwrap()
            .remove(0);
        let mut sb = model.new_session().unwrap();
        let solo_b = model
            .forward_batched(&[&b], std::slice::from_mut(&mut sb))
            .unwrap()
            .remove(0);

        let mut pair = [model.new_session().unwrap(), model.new_session().unwrap()];
        let both = model.forward_batched(&[&a, &b], &mut pair).unwrap();
        assert_close(&both[0], &solo_a, 1e-5, "session A under batching");
        assert_close(&both[1], &solo_b, 1e-5, "session B under batching");
    }

    #[test]
    fn qsa_selection_engages_at_depth_and_stays_causal() {
        // Run past the tiny top_k budget so the attention layer really
        // selects; the forward must stay finite and deterministic.
        let dev = dev();
        let model = tiny_model(&dev);
        let tokens: Vec<u32> = (0..24).map(|i| (i * 5 + 3) % 13).collect();

        let mut s1 = model.new_session().unwrap();
        let l1 = model
            .forward_batched(&[&tokens], std::slice::from_mut(&mut s1))
            .unwrap()
            .remove(0);
        let mut s2 = model.new_session().unwrap();
        let l2 = model
            .forward_batched(&[&tokens], std::slice::from_mut(&mut s2))
            .unwrap()
            .remove(0);
        assert_close(&l1, &l2, 0.0, "determinism under selection");
        let m = l1
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(m.is_finite(), "non-finite logits under QSA selection");
        // The index cache really accumulated.
        let LayerState::Attention { idx, .. } = &s1.layers[3] else {
            panic!("layer 3 should be attention");
        };
        assert_eq!(idx.seq_len(), tokens.len());
    }

    #[test]
    fn mismatched_state_schedule_is_refused() {
        let dev = dev();
        let model = tiny_model(&dev);
        let mut state = model.new_session().unwrap();
        state.layers[0] = LayerState::Attention {
            kv: AttentionState::empty(),
            idx: IndexState::empty(),
        };
        let err = model
            .forward_batched(&[&[1u32, 2][..]], std::slice::from_mut(&mut state))
            .unwrap_err();
        assert!(err.to_string().contains("state kind mismatch"));
    }
}
