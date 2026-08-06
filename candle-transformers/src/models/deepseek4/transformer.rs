//! Full DeepSeek-V4-Flash transformer: embedding → hyper-connection expansion → N
//! `Block`s (each an mHC-wrapped attention sub-block and MoE sub-block) → hyper-connection
//! head reduction → output norm → LM head. Mirrors `Transformer.forward` /
//! `Block.forward` in `inference/model.py`.
//!
//! This is the numerically-correct reference forward: it recomputes over the full prefix
//! each step (no incremental KV state), which matches the reference's prefill path.

use candle::{DType, IndexOp, Result, Tensor, D};

use super::attention::Attention;
use super::config::Config;
use super::hyper::{HyperConnection, HyperParams};
use super::linear::QLinear;
use super::moe::MoE;
use super::rope::RotaryCache;

/// One transformer block: mHC-wrapped attention followed by mHC-wrapped MoE.
pub struct Block {
    hc: HyperConnection,
    hc_attn: HyperParams,
    hc_ffn: HyperParams,
    attn_norm: Tensor,
    ffn_norm: Tensor,
    attn: Attention,
    moe: MoE,
    eps: f64,
}

impl Block {
    pub fn new(
        hc: HyperConnection,
        hc_attn: HyperParams,
        hc_ffn: HyperParams,
        attn_norm: Tensor,
        ffn_norm: Tensor,
        attn: Attention,
        moe: MoE,
        eps: f64,
    ) -> Self {
        Self {
            hc,
            hc_attn,
            hc_ffn,
            attn_norm,
            ffn_norm,
            attn,
            moe,
            eps,
        }
    }

    /// `h` is the `[b, s, hc_mult, dim]` residual stream; `input_ids` `[b, s]`.
    pub fn forward(&self, h: &Tensor, input_ids: &Tensor, rope: &RotaryCache) -> Result<Tensor> {
        // Attention sub-block.
        let residual = h;
        let (x, post, comb) = self.hc.pre(h, &self.hc_attn)?;
        let x = rms_norm(&x, &self.attn_norm, self.eps)?;
        let x = self.attn.forward(&x, rope)?;
        let h = self.hc.post(&x, residual, &post, &comb)?;

        // MoE sub-block.
        let residual = &h;
        let (x, post, comb) = self.hc.pre(&h, &self.hc_ffn)?;
        let x = rms_norm(&x, &self.ffn_norm, self.eps)?;
        let x = self.moe.forward(&x, input_ids)?;
        self.hc.post(&x, residual, &post, &comb)
    }

    /// Build the incremental (decode) form of this block: the mHC mix, norms, and MoE are all
    /// per-token stateless, so only the attention needs streaming KV state (see
    /// [`super::attention::IncrementalAttention`]).
    pub fn decoder(&self) -> Result<IncrementalBlock<'_>> {
        Ok(IncrementalBlock {
            b: self,
            attn: self.attn.decoder()?,
        })
    }
}

/// Streaming (decode-time) counterpart to [`Block`]: processes one token's `[1,1,hc_mult,dim]`
/// residual per `step`, running the same mHC-wrapped attention + MoE as prefill but with the
/// attention driven incrementally. Because every op except attention is per-token, and the
/// attention `step` matches prefill row-for-row, `step` reproduces `Block::forward` row `t`.
pub struct IncrementalBlock<'a> {
    b: &'a Block,
    attn: super::attention::IncrementalAttention<'a>,
}

impl IncrementalBlock<'_> {
    /// One decode step. `h` `[1,1,hc_mult,dim]` residual, `input_id` `[1,1]` u32 → next residual.
    pub fn step(&mut self, h: &Tensor, input_id: &Tensor, rope: &RotaryCache) -> Result<Tensor> {
        let b = self.b;
        // Attention sub-block (mHC pre → norm → incremental attention → mHC post).
        let residual = h;
        let (x, post, comb) = b.hc.pre(h, &b.hc_attn)?;
        let x = rms_norm(&x, &b.attn_norm, b.eps)?;
        let x = self.attn.step(&x, rope)?;
        let h = b.hc.post(&x, residual, &post, &comb)?;

        // MoE sub-block (per-token: identical to prefill).
        let residual = &h;
        let (x, post, comb) = b.hc.pre(&h, &b.hc_ffn)?;
        let x = rms_norm(&x, &b.ffn_norm, b.eps)?;
        let x = b.moe.forward(&x, input_id)?;
        b.hc.post(&x, residual, &post, &comb)
    }
}

/// The full model.
pub struct Transformer {
    embed: Tensor, // [vocab, dim]
    blocks: Vec<Block>,
    hc: HyperConnection,
    hc_head: HyperParams,
    output_norm: Tensor,
    lm_head: QLinear,
    rope_compress: RotaryCache,
    rope_swa: RotaryCache,
    cfg: Config,
}

impl Transformer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: Config,
        embed: Tensor,
        blocks: Vec<Block>,
        hc_head: HyperParams,
        output_norm: Tensor,
        lm_head: QLinear,
        rope_compress: RotaryCache,
        rope_swa: RotaryCache,
    ) -> Self {
        let hc = HyperConnection::new(cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps);
        Self {
            embed,
            blocks,
            hc,
            hc_head,
            output_norm,
            lm_head,
            rope_compress,
            rope_swa,
            cfg,
        }
    }

    fn rope_for(&self, layer: usize) -> &RotaryCache {
        if self.cfg.layer_kind(layer).compresses() {
            &self.rope_compress
        } else {
            &self.rope_swa
        }
    }

    /// Full-sequence forward. `input_ids` `[b, s]` (u32) → logits `[b, s, vocab]`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let ids = input_ids.to_dtype(DType::U32)?;
        // Embedding lookup → [b, s, dim].
        let flat = ids.reshape(b * s)?;
        let emb = self
            .embed
            .index_select(&flat, 0)?
            .reshape((b, s, self.cfg.dim))?;
        // Expand to the hc_mult-copy residual stream.
        let mut h = self.hc.expand(&emb.to_dtype(DType::F32)?)?;
        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, &ids, self.rope_for(i))?;
        }
        // Reduce copies → [b, s, dim], output norm, LM head.
        let h = self.hc.head_reduce(&h, &self.hc_head)?;
        let h = rms_norm(&h, &self.output_norm, self.cfg.norm_eps)?;
        self.lm_head.forward(&h)
    }

    /// Greedy decode of `max_new` tokens appended to `prompt` (a `[s]` u32 tensor of ids).
    /// Recomputes the full prefix each step (reference path). Returns the generated ids.
    pub fn generate(
        &self,
        prompt: &[u32],
        max_new: usize,
        device: &candle::Device,
    ) -> Result<Vec<u32>> {
        let mut ids: Vec<u32> = prompt.to_vec();
        let mut out = Vec::new();
        for _ in 0..max_new {
            let input = Tensor::from_vec(ids.clone(), (1, ids.len()), device)?;
            let logits = self.forward(&input)?; // [1, s, vocab]
            let last = logits.i((0, ids.len() - 1))?; // [vocab]
            let next = last.argmax(D::Minus1)?.to_scalar::<u32>()?;
            ids.push(next);
            out.push(next);
        }
        Ok(out)
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Build the incremental (decode) form of the whole model: one [`IncrementalBlock`] per
    /// layer, each holding its own streaming attention KV. `step` feeds one token at a time and
    /// returns that position's logits, equal to the corresponding row of `forward`.
    pub fn decoder(&self) -> Result<IncrementalTransformer<'_>> {
        let blocks = self
            .blocks
            .iter()
            .map(|b| b.decoder())
            .collect::<Result<Vec<_>>>()?;
        Ok(IncrementalTransformer { t: self, blocks })
    }
}

/// Streaming (decode-time) counterpart to [`Transformer`]: embed → hc-expand → per-block
/// incremental `step` → hc head-reduce → output norm → LM head, one token at a time. Because
/// every cross-token dependency lives in the per-block attention (now incremental and matched to
/// prefill) and everything else is per-token, streaming the prompt token-by-token reproduces the
/// full-prefix `forward` logits row-for-row (proven by `incremental_transformer_matches_prefill`).
pub struct IncrementalTransformer<'a> {
    t: &'a Transformer,
    blocks: Vec<IncrementalBlock<'a>>,
}

impl IncrementalTransformer<'_> {
    /// Feed the next token id and return its logits `[vocab]` — equal to row `pos` of `forward`.
    pub fn step(&mut self, id: u32) -> Result<Tensor> {
        let t = self.t;
        let dev = t.embed.device();
        let idt = Tensor::from_vec(vec![id], (1, 1), dev)?; // [1,1] u32
        let emb = t
            .embed
            .index_select(&idt.reshape(1)?, 0)?
            .reshape((1, 1, t.cfg.dim))?
            .to_dtype(DType::F32)?;
        let mut h = t.hc.expand(&emb)?; // [1,1,hc_mult,dim]
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            h = blk.step(&h, &idt, t.rope_for(i))?;
        }
        let h = t.hc.head_reduce(&h, &t.hc_head)?; // [1,1,dim]
        let h = rms_norm(&h, &t.output_norm, t.cfg.norm_eps)?;
        t.lm_head.forward(&h)?.reshape((t.cfg.vocab_size,))
    }
}

/// RMSNorm with a learned weight.
fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = x.broadcast_div(&(ms + eps)?.sqrt()?)?;
    normed.broadcast_mul(&w.to_dtype(DType::F32)?)
}

#[cfg(test)]
mod tests {
    use super::super::attention::AttentionParams;
    use super::super::compressor::Compressor;
    use super::super::indexer::Indexer;
    use super::super::moe::{Expert, Gate, ScoreFunc};
    use super::*;
    use candle::Device;

    fn dense(rows: usize, cols: usize, dev: &Device) -> Result<QLinear> {
        Ok(QLinear::from_weight(Tensor::randn(
            0f32,
            0.3,
            (rows, cols),
            dev,
        )?))
    }

    fn hc_params(mix: usize, hcdim: usize, dev: &Device) -> Result<HyperParams> {
        Ok(HyperParams {
            fn_w: Tensor::randn(0f32, 0.3, (mix, hcdim), dev)?.into(),
            base: Tensor::zeros(mix, DType::F32, dev)?,
            scale: Tensor::ones(3, DType::F32, dev)?,
        })
    }

    fn build_expert(cfg: &Config, dev: &Device) -> Result<Expert> {
        Ok(Expert::new(
            dense(cfg.moe_inter_dim, cfg.dim, dev)?,
            dense(cfg.dim, cfg.moe_inter_dim, dev)?,
            dense(cfg.moe_inter_dim, cfg.dim, dev)?,
            cfg.swiglu_limit,
        ))
    }

    fn build_moe(cfg: &Config, layer: usize, dev: &Device) -> Result<MoE> {
        let tid2eid = if cfg.is_hash_layer(layer) {
            // deterministic hash table
            let mut t = vec![0i64; cfg.vocab_size * cfg.n_activated_experts];
            for v in 0..cfg.vocab_size {
                for k in 0..cfg.n_activated_experts {
                    t[v * cfg.n_activated_experts + k] =
                        ((v * 7 + k * 3) % cfg.n_routed_experts) as i64;
                }
            }
            Some(Tensor::from_vec(
                t,
                (cfg.vocab_size, cfg.n_activated_experts),
                dev,
            )?)
        } else {
            None
        };
        let bias = if tid2eid.is_none() {
            Some(Tensor::zeros(cfg.n_routed_experts, DType::F32, dev)?)
        } else {
            None
        };
        let gate = Gate::new(
            Tensor::randn(0f32, 0.3, (cfg.n_routed_experts, cfg.dim), dev)?,
            bias,
            tid2eid,
            cfg.n_activated_experts,
            cfg.n_routed_experts,
            ScoreFunc::parse(&cfg.score_func),
            cfg.route_scale,
        );
        let experts: Vec<Expert> = (0..cfg.n_routed_experts)
            .map(|_| build_expert(cfg, dev).unwrap())
            .collect();
        let shared = build_expert(cfg, dev)?;
        Ok(MoE::new(gate, experts, shared, cfg.dim))
    }

    fn build_attention(cfg: &Config, layer: usize, dev: &Device) -> Result<Attention> {
        let (h, hd, ng, olr) = (cfg.n_heads, cfg.head_dim, cfg.o_groups, cfg.o_lora_rank);
        let ratio = cfg.compress_ratio(layer);
        let (compressor, indexer) = if cfg.layer_kind(layer).compresses() {
            let coff = if ratio == 4 { 2 } else { 1 };
            let comp = Compressor::new(
                Tensor::randn(0f32, 0.3, (coff * hd, cfg.dim), dev)?,
                Tensor::randn(0f32, 0.3, (coff * hd, cfg.dim), dev)?,
                Tensor::randn(0f32, 0.3, (ratio, coff * hd), dev)?,
                Tensor::ones(hd, DType::F32, dev)?,
                ratio,
                hd,
                cfg.rope_head_dim,
                cfg.norm_eps,
            );
            let indexer = if cfg.layer_kind(layer).has_indexer() {
                let ihd = cfg.index_head_dim;
                let icomp = Compressor::new(
                    Tensor::randn(0f32, 0.3, (2 * ihd, cfg.dim), dev)?,
                    Tensor::randn(0f32, 0.3, (2 * ihd, cfg.dim), dev)?,
                    Tensor::randn(0f32, 0.3, (ratio, 2 * ihd), dev)?,
                    Tensor::ones(ihd, DType::F32, dev)?,
                    ratio,
                    ihd,
                    cfg.rope_head_dim,
                    cfg.norm_eps,
                );
                Some(Indexer::new(
                    QLinear::from_weight(Tensor::randn(
                        0f32,
                        0.3,
                        (cfg.index_n_heads * ihd, cfg.q_lora_rank),
                        dev,
                    )?),
                    Tensor::randn(0f32, 0.3, (cfg.index_n_heads, cfg.dim), dev)?,
                    icomp,
                    cfg.index_n_heads,
                    ihd,
                    cfg.rope_head_dim,
                    cfg.index_topk,
                ))
            } else {
                None
            };
            (Some(comp), indexer)
        } else {
            (None, None)
        };
        let p = AttentionParams {
            wq_a: dense(cfg.q_lora_rank, cfg.dim, dev)?,
            q_norm: Tensor::ones(cfg.q_lora_rank, DType::F32, dev)?,
            wq_b: dense(h * hd, cfg.q_lora_rank, dev)?,
            wkv: dense(hd, cfg.dim, dev)?,
            kv_norm: Tensor::ones(hd, DType::F32, dev)?,
            wo_a: (0..ng)
                .map(|_| dense(olr, (h / ng) * hd, dev))
                .collect::<Result<Vec<_>>>()?,
            wo_b: dense(cfg.dim, ng * olr, dev)?,
            attn_sink: Tensor::randn(0f32, 1.0, h, dev)?,
            compressor,
            indexer,
        };
        Ok(Attention::new(cfg, layer, p))
    }

    fn build_model(cfg: &Config, dev: &Device) -> Result<Transformer> {
        let hc = HyperConnection::new(cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps);
        let mix = hc.mix_hc();
        let hcdim = cfg.hc_mult * cfg.dim;
        let mut blocks = Vec::new();
        for layer in 0..cfg.n_layers {
            blocks.push(Block::new(
                hc.clone(),
                hc_params(mix, hcdim, dev)?,
                hc_params(mix, hcdim, dev)?,
                Tensor::ones(cfg.dim, DType::F32, dev)?,
                Tensor::ones(cfg.dim, DType::F32, dev)?,
                build_attention(cfg, layer, dev)?,
                build_moe(cfg, layer, dev)?,
                cfg.norm_eps,
            ));
        }
        let hc_head = HyperParams {
            fn_w: Tensor::randn(0f32, 0.3, (cfg.hc_mult, hcdim), dev)?.into(),
            base: Tensor::zeros(cfg.hc_mult, DType::F32, dev)?,
            scale: Tensor::ones(1, DType::F32, dev)?,
        };
        let rope_c = RotaryCache::new(
            cfg.rope_head_dim,
            cfg.compress_rope_theta,
            cfg.original_seq_len,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
            dev,
        )?;
        let rope_s = RotaryCache::new(
            cfg.rope_head_dim,
            cfg.rope_theta,
            0,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
            dev,
        )?;
        Ok(Transformer::new(
            cfg.clone(),
            Tensor::randn(0f32, 0.3, (cfg.vocab_size, cfg.dim), dev)?,
            blocks,
            hc_head,
            Tensor::ones(cfg.dim, DType::F32, dev)?,
            dense(cfg.vocab_size, cfg.dim, dev)?,
            rope_c,
            rope_s,
        ))
    }

    /// The assembled model runs end-to-end at tiny config, producing finite logits of the
    /// right shape, deterministically, and greedy-generates without error.
    #[test]
    fn tiny_model_forward_and_generate() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = Config::tiny();
        let model = build_model(&cfg, &dev)?;
        let ids = Tensor::from_vec(vec![1u32, 5, 9, 3, 7, 2, 8, 4, 6, 0, 11, 12], (1, 12), &dev)?;
        let logits = model.forward(&ids)?;
        assert_eq!(logits.dims(), &[1, 12, cfg.vocab_size]);
        assert!(logits
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| v.is_finite()));

        // Determinism.
        let logits2 = model.forward(&ids)?;
        let d = (logits - logits2)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(d < 1e-5, "non-deterministic: {d}");

        // Greedy generate a few tokens.
        let gen = model.generate(&[1u32, 5, 9, 3], 4, &dev)?;
        assert_eq!(gen.len(), 4);
        assert!(gen.iter().all(|&t| (t as usize) < cfg.vocab_size));
        Ok(())
    }

    /// CAPSTONE: streaming the prompt token-by-token through the incremental decoder reproduces
    /// the full-prefix `forward` logits row-for-row — the end-to-end prefill/decode equivalence
    /// across every layer kind (SWA/CSA/HCA), the mHC residual stream, and the MoE. This is the
    /// correctness gate for the fast engine decode path: the streaming form the reference lacked.
    #[test]
    fn incremental_transformer_matches_prefill() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = Config::tiny();
        let model = build_model(&cfg, &dev)?;
        let toks: Vec<u32> = vec![1, 5, 9, 3, 7, 2, 8, 4, 6, 0, 11, 12];
        let s = toks.len();

        // Oracle: full-prefix prefill → [1, s, vocab].
        let prefill = model.forward(&Tensor::from_vec(toks.clone(), (1, s), &dev)?)?;

        // Stream token-by-token; collect each position's logits.
        let mut dec = model.decoder()?;
        let mut rows: Vec<Tensor> = Vec::with_capacity(s);
        for &id in &toks {
            rows.push(dec.step(id)?.reshape((1, 1, cfg.vocab_size))?);
        }
        let streamed = Tensor::cat(&rows, 1)?; // [1, s, vocab]

        let a = prefill.flatten_all()?.to_vec1::<f32>()?;
        let b = streamed.flatten_all()?.to_vec1::<f32>()?;
        let max_abs = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs < 1e-3,
            "incremental vs prefill logits diverge: max|Δ| = {max_abs}"
        );
        Ok(())
    }
}
