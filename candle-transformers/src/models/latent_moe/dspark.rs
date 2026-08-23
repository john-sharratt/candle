//! DSpark speculative-decode drafter (`dflash` architecture) for DeepSeek-V4-Flash.
//!
//! DSpark ("Confidence-Scheduled Speculative Decoding with Semi-Autoregressive
//! Generation", Cheng et al. 2026, arXiv:2607.05147) is the draft model bartowski ships
//! as a separate `dspark-*.gguf` alongside the main quant. It is **not** the V3-style
//! MTP `eh_proj` recurrence — it is a small standalone `dflash` model that:
//!
//! * has `block_count` transformer blocks structurally identical to a main DeepSeek-V4
//!   block (MLA + 256-expert MoE + hyper-connections) — so they load via the existing
//!   [`load_block`](super::loader::load_block);
//! * **shares** the target's frozen `token_embd` + output head (`output.weight`) — neither
//!   is present in the drafter file;
//! * adds a **low-rank Markov head** (`markov_w1`/`markov_w2`) for semi-autoregressive
//!   block drafting, an `output_norm`, and an optional **confidence head**;
//! * is **conditioned on the target** via feature extraction + KV injection from the
//!   `dflash.target_layers` of the main model.
//!
//! This module lands the parts that are self-contained and exactly testable: the config,
//! the weight loader, and the pure Markov-bias / confidence math (paper Eqs. 5 and 7).
//! The backbone forward with target-KV injection and the draft/verify loop are the
//! GPU-coupled integration built on top — see `docs/deepseek_v4_speculative_decode.md`.

use candle::{DType, Result, Tensor};
use candle_nn::ops::softmax_last_dim;

use super::arch::Arch;
use super::config::Config;
use super::linear::QLinear;
use super::loader::{config_from_gguf, dequant_f32, GgufModel};

// The drafter backbone + streaming-MoE integration is GPU-coupled (the streaming expert cache and
// the injected-context attention run on CUDA); the Markov/confidence math below is device-agnostic
// and stays available (and CPU-tested) without the `cuda` feature.
#[cfg(feature = "cuda")]
use super::attention::Attention;
#[cfg(feature = "cuda")]
use super::dspark_experts::{resident_slots_for_vram, DsparkStreamingMoe};
#[cfg(feature = "cuda")]
use super::hyper::{HyperConnection, HyperParams};
#[cfg(feature = "cuda")]
use super::loader::{hc_output, load_dspark_block, load_hc_params, qlinear};
#[cfg(feature = "cuda")]
use super::rope::RotaryCache;
#[cfg(feature = "cuda")]
use super::transformer::rms_norm as block_rms_norm;
#[cfg(feature = "cuda")]
use candle::quantized::{get_total_vram_device0, Int8Mode};
#[cfg(feature = "cuda")]
use candle::Device;
#[cfg(feature = "cuda")]
use candle::IndexOp;
#[cfg(feature = "cuda")]
use candle_nn::ops::rms_norm;

/// A DSpark drafter backbone block: the mHC-wrapped **injected-context** attention sub-block and
/// its norms, with the FFN sub-block's MoE supplied externally at forward time (the drafter's
/// experts live in the shared [`DsparkStreamingMoe`], not in the block). Structurally the injected
/// half of [`super::transformer::Block`] minus the eager `MoE` field — see [`load_dspark_block`].
#[cfg(feature = "cuda")]
pub struct DsparkBlock {
    hc: HyperConnection,
    hc_attn: HyperParams,
    hc_ffn: HyperParams,
    attn_norm: Tensor,
    ffn_norm: Tensor,
    attn: Attention,
    eps: f64,
}

#[cfg(feature = "cuda")]
impl DsparkBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hc: HyperConnection,
        hc_attn: HyperParams,
        hc_ffn: HyperParams,
        attn_norm: Tensor,
        ffn_norm: Tensor,
        attn: Attention,
        eps: f64,
    ) -> Self {
        Self {
            hc,
            hc_attn,
            hc_ffn,
            attn_norm,
            ffn_norm,
            attn,
            eps,
        }
    }

    /// Run the block over a draft `h` `[1, s, hc_mult, dim]` residual: the mHC-wrapped non-causal
    /// injected-context attention (attending over `[wkv(ctx) ‖ wkv(x)]`, `ctx` = `Hctx`), then the
    /// mHC-wrapped FFN whose MoE output is produced by `ffn` — a closure that takes the FFN-normed
    /// `[1, s, dim]` and returns the MoE result `[1, s, dim]` (the caller drives the streaming MoE
    /// for this block index). Identical mHC/norm math to [`super::transformer::Block::forward_injected`].
    pub fn forward_injected<F>(
        &self,
        h: &Tensor,
        ctx: &Tensor,
        rope: &RotaryCache,
        ctx_start: usize,
        q_start: usize,
        ffn: F,
    ) -> Result<Tensor>
    where
        F: FnOnce(&Tensor) -> Result<Tensor>,
    {
        // Attention sub-block (mHC pre → norm → injected non-causal attention → mHC post).
        let residual = h;
        let (x, post, comb) = self.hc.pre(h, &self.hc_attn)?;
        let x = block_rms_norm(&x, &self.attn_norm, self.eps)?;
        let x = self
            .attn
            .forward_injected(&x, ctx, rope, ctx_start, q_start)?;
        let h = self.hc.post(&x, residual, &post, &comb)?;

        // FFN sub-block (mHC pre → norm → streaming MoE → mHC post).
        let residual = &h;
        let (x, post, comb) = self.hc.pre(&h, &self.hc_ffn)?;
        let x = block_rms_norm(&x, &self.ffn_norm, self.eps)?;
        let x = ffn(&x)?;
        self.hc.post(&x, residual, &post, &comb)
    }
}

/// Draft-model configuration: the base [`Config`] parsed from the `dflash.*` metadata,
/// plus the two draft-specific fields that govern speculation.
#[derive(Debug, Clone)]
pub struct DsparkConfig {
    /// The `dflash` backbone config (blocks, heads, experts, hyper-connections). Parsed
    /// through the same [`config_from_gguf`] path as the main model — the metadata prefix
    /// is `dflash.` (from `general.architecture`), so `n_layers` = `dflash.block_count`.
    pub base: Config,
    /// The trained maximum draft block length (`dflash.block_size`, 5 for this checkpoint).
    /// `--spec-draft-n-max` is clamped to this — the backbone emits this many base logits
    /// per forward.
    pub block_size: usize,
    /// The target-model layer indices whose hidden states are extracted, projected, and
    /// injected into the draft's attention KV (`dflash.target_layers`, paper Eq. 2-3).
    pub target_layers: Vec<usize>,
    /// The block-diffusion mask token id (`tokenizer.ggml.mask_token_id`, 128799 for this
    /// checkpoint). The semi-autoregressive draft fills the block's not-yet-sampled positions with
    /// this token before the single backbone pass; feeding the wrong id (e.g. BOS) makes every
    /// masked position's base logit garbage and caps acceptance at one draft.
    pub mask_token: u32,
}

impl DsparkConfig {
    /// Parse the drafter's config from its GGUF metadata, under `arch`'s
    /// namespace — the drafter is its own architecture, obtained from the
    /// target's [`Arch::drafter`].
    pub fn from_gguf(m: &GgufModel, arch: &'static dyn Arch) -> Result<Self> {
        let base = config_from_gguf(m, arch)?;
        let block_size = m
            .metadata_u32("dflash.block_size")
            .map(|v| v as usize)
            .unwrap_or(5);
        let target_layers = u32_array(m, "dflash.target_layers");
        let mask_token = m.metadata_u32("tokenizer.ggml.mask_token_id")?;
        Ok(Self {
            base,
            block_size,
            target_layers,
            mask_token,
        })
    }
}

/// Low-rank first-order Markov head (paper Eq. 5): the per-block-position logit bias
/// `B(x_{k-1}) = W₁[x_{k-1}] · W₂`, keyed on the previously sampled token in the block.
///
/// `w1` is the `[vocab, rank]` lookup table (a row per token id); `w2` is the `[rank, vocab]`
/// projection back to the shared vocabulary. Stored orientation is normalised on load so
/// the bias is `row(w1, prev) @ w2`.
pub struct MarkovHead {
    /// `[vocab, rank]` — one `rank`-dim embedding per token id.
    w1: Tensor,
    /// `[rank, vocab]` — projects the looked-up embedding to a full-vocab logit bias.
    w2: Tensor,
    vocab: usize,
    rank: usize,
}

impl MarkovHead {
    /// Build from the two loaded weights, orienting each to the canonical shape from the
    /// known `vocab`. Either GGUF storage orientation is accepted; a weight that matches
    /// neither is an error (the file disagrees with the config — a real load fault).
    pub fn new(w1: Tensor, w2: Tensor, vocab: usize) -> Result<Self> {
        // w1 → [vocab, rank]: the lookup axis (vocab) must be dim 0.
        let w1 = orient_rows(w1, vocab, "markov_w1")?;
        let rank = w1.dim(1)?;
        // w2 → [rank, vocab]: the projection maps rank → vocab.
        let w2 = if w2.dim(0)? == rank && w2.dim(1)? == vocab {
            w2
        } else if w2.dim(0)? == vocab && w2.dim(1)? == rank {
            w2.t()?.contiguous()?
        } else {
            candle::bail!(
                "markov_w2 shape {:?} matches neither [rank={rank}, vocab={vocab}] nor its \
                 transpose",
                w2.dims()
            );
        };
        Ok(Self {
            w1,
            w2,
            vocab,
            rank,
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// The `rank`-dim Markov embedding of a previously sampled token — `W₁[prev]`, the term
    /// the confidence head also consumes (paper Eq. 7). Shape `[rank]`.
    pub fn embed(&self, prev_token: u32) -> Result<Tensor> {
        self.w1.narrow(0, prev_token as usize, 1)?.squeeze(0)
    }

    /// The full-vocab logit bias for the next block position given the previous token —
    /// `W₁[prev] · W₂` (paper Eq. 5). Shape `[vocab]`. Add this to the backbone base logit
    /// `Uₖ` before sampling position `k`.
    pub fn bias(&self, prev_token: u32) -> Result<Tensor> {
        // [1, rank] @ [rank, vocab] = [1, vocab] → [vocab].
        let e = self.w1.narrow(0, prev_token as usize, 1)?;
        e.matmul(&self.w2)?.squeeze(0)
    }

    /// [`Self::embed`] with the previous token as a DEVICE `[1]` u32 tensor —
    /// the sequential sampler's chained form, which never reads the token back
    /// to the host. Shape `[1, rank]`.
    pub fn embed_dev(&self, prev: &Tensor) -> Result<Tensor> {
        self.w1.index_select(prev, 0)
    }

    /// [`Self::bias`] from a device `[1, rank]` embedding (from
    /// [`Self::embed_dev`]). Shape `[vocab]`.
    pub fn bias_from_embed(&self, e: &Tensor) -> Result<Tensor> {
        e.matmul(&self.w2)?.squeeze(0)
    }
}

/// Confidence head (paper Eq. 7): `cₖ = σ(wᵀ · [hₖ ; W₁[x_{k-1}]])` — the survival
/// probability of the draft token at block position `k`, used by the confidence-scheduled
/// prefix selector (draft length = longest prefix whose cumulative product `∏ cᵢ` clears
/// the throughput-optimal threshold). Optional: some quantised checkpoints omit it.
pub struct ConfidenceHead {
    /// `[hidden + rank]` — the linear applied to `[hidden ; markov-embedding]`.
    w: Tensor,
}

impl ConfidenceHead {
    pub fn new(w: Tensor) -> Self {
        Self { w }
    }

    /// Length of the concatenated `[hidden ; markov-embedding]` feature the head consumes
    /// (= the weight's element count). `hidden` width is `feat_len − markov.rank()`.
    pub fn feat_len(&self) -> Result<usize> {
        Ok(self.w.elem_count())
    }

    /// `σ(wᵀ [h ; markov_embed])`. `h` is the block-position hidden `[hidden]`; `markov_embed`
    /// is `MarkovHead::embed(prev_token)` `[rank]`.
    pub fn confidence(&self, h: &Tensor, markov_embed: &Tensor) -> Result<f32> {
        let x = Tensor::cat(&[h.clone(), markov_embed.clone()], 0)?; // [hidden + rank]
        let z = x.broadcast_mul(&self.w)?.sum_all()?.to_scalar::<f32>()?;
        Ok(1.0 / (1.0 + (-z).exp()))
    }

    /// [`Self::confidence`] kept ON DEVICE — returns the survival probability as
    /// a `[1]` f32 tensor so the sequential sampler's chain never syncs the host.
    pub fn confidence_dev(&self, h: &Tensor, markov_embed: &Tensor) -> Result<Tensor> {
        let x = Tensor::cat(&[h.clone(), markov_embed.clone()], 0)?; // [hidden + rank]
        let z = x.broadcast_mul(&self.w)?.sum_all()?; // scalar
                                                      // σ(z) = 1 / (1 + e^(−z)), all tensor ops.
        ((z.neg()?.exp()? + 1.0)?.recip()?).reshape(1)
    }
}

/// The loaded DSpark drafter — every tensor in the `dflash` GGUF (verified against the real
/// file's tensor directory). The `token_embd` + output projection are shared with the target
/// and deliberately absent.
#[cfg(feature = "cuda")]
pub struct DsparkDrafter {
    pub cfg: DsparkConfig,
    /// `block_count` backbone blocks (mHC attention + norms; the MoE is the shared streaming cache).
    pub blocks: Vec<DsparkBlock>,
    /// The drafter's routed experts: host-resident, streamed into a small VRAM slot set on demand
    /// (one shared cache across all blocks). Replaces the per-block eager MoE the target uses —
    /// the drafter's 3×256 experts are far too large to keep VRAM-resident alongside the 284B target.
    moe: DsparkStreamingMoe,
    /// Target-feature encoder projection `Wc` (`fc.weight`): concatenated target-layer
    /// hidden states → draft hidden width (paper Eq. 2).
    pub fc: QLinear,
    /// RMSNorm over the encoder projection output, producing the injected context `Hctx`
    /// (`enc.output_norm.weight`, paper Eq. 2).
    pub enc_output_norm: Tensor,
    /// Low-rank semi-autoregressive Markov head (`markov_w1/w2`).
    pub markov: MarkovHead,
    /// Confidence head `conf_proj.weight` (paper Eq. 7), when the checkpoint ships one.
    pub confidence: Option<ConfidenceHead>,
    /// Output-side hyper-connection reducing the `hc_mult`-wide residual before the head
    /// (`output_hc_{base,fn,scale}`).
    pub output_hc: HyperParams,
    /// Final RMSNorm before the shared output head (`output_norm.weight`).
    pub output_norm: Tensor,
    /// mHC connection for the residual expand/head-reduce around the backbone blocks.
    hc: HyperConnection,
    /// RoPE cache for the backbone (SWA blocks → base theta, YaRN off).
    rope: RotaryCache,
}

#[cfg(feature = "cuda")]
impl DsparkDrafter {
    /// Load the drafter from its single-file GGUF. The backbone blocks reuse the main model's
    /// block loader; the encoder/Markov/confidence/output tensors are the draft-specific ones;
    /// the embedding and output projection are shared with the target and are NOT read here.
    ///
    /// `arch` is the drafter's own architecture — the target's
    /// [`Arch::drafter`], which is what makes its `dflash.*` metadata readable.
    pub fn load(
        model_path: &std::path::Path,
        arch: &'static dyn Arch,
        device: &Device,
    ) -> Result<Self> {
        let mut m = GgufModel::open(std::slice::from_ref(&model_path.to_path_buf()))?;
        let cfg = DsparkConfig::from_gguf(&m, arch)?;

        // VRAM-adaptive resident-slot count; below the smallest tier the drafter has no room and
        // speculative decode is disabled (the engine gates on this before ever calling `load`).
        let total_vram = get_total_vram_device0()?;
        // Total routed experts across the drafter's blocks — the pool the resident fraction is of.
        let total_experts = cfg.base.n_layers * cfg.base.n_routed_experts;
        let n_slots = resident_slots_for_vram(total_vram, total_experts).ok_or_else(|| {
            candle::Error::msg(format!(
                "DSpark drafter needs > 24 GiB VRAM ({:.1} GiB present) — speculative decode disabled",
                total_vram as f64 / (1u64 << 30) as f64
            ))
        })?;

        // Backbone blocks: attention + norms only (the MoE is the shared streaming cache below).
        let mut blocks = Vec::with_capacity(cfg.base.n_layers);
        for layer in 0..cfg.base.n_layers {
            blocks.push(load_dspark_block(&mut m, &cfg.base, layer, device)?);
        }
        // The drafter's routed experts → host RAM + `n_slots` VRAM slots streamed on demand.
        let moe = DsparkStreamingMoe::load(&mut m, &cfg.base, cfg.base.n_layers, n_slots, device)?;

        // Target-feature encoder: fc (Wc) then its RMSNorm → Hctx.
        let fc = qlinear(&mut m, "fc.weight", device)?;
        let enc_output_norm = dequant_f32(&mut m, "enc.output_norm.weight", device)?;

        let w1 = dequant_f32(&mut m, "markov_w1.weight", device)?;
        let w2 = dequant_f32(&mut m, "markov_w2.weight", device)?;
        let markov = MarkovHead::new(w1, w2, cfg.base.vocab_size)?;

        // Confidence head — `conf_proj.weight`, flattened to the `[hidden+rank]` weight vector.
        // Optional: a quantised checkpoint may drop it.
        let confidence = match m.info("conf_proj.weight") {
            Some(_) => Some(ConfidenceHead::new(
                dequant_f32(&mut m, "conf_proj.weight", device)?.flatten_all()?,
            )),
            None => None,
        };

        // Output-side hyper-connection + final norm before the shared head.
        let output_hc = load_hc_params(&mut m, hc_output(&cfg.base), device, Int8Mode::Off)?;
        let output_norm = dequant_f32(&mut m, "output_norm.weight", device)?;

        // mHC connection + RoPE cache for the backbone (its blocks are SWA → base theta).
        let hc = HyperConnection::new(
            cfg.base.hc_mult,
            cfg.base.hc_sinkhorn_iters,
            cfg.base.hc_eps,
        );
        let (theta, orig) = cfg.base.rope_params(0);
        let rope = RotaryCache::new(
            cfg.base.rope_head_dim,
            theta,
            orig,
            cfg.base.rope_factor,
            cfg.base.beta_fast,
            cfg.base.beta_slow,
            device,
        )?;

        Ok(Self {
            cfg,
            blocks,
            moe,
            fc,
            enc_output_norm,
            markov,
            confidence,
            output_hc,
            output_norm,
            hc,
            rope,
        })
    }

    /// The DSpark backbone (`graph_dsv4`) over a draft block: embed → mHC expand → the
    /// `block_count` injected non-causal blocks → mHC head-reduce → `output_norm`. Returns the
    /// per-position hidden `[s, dim]` (the shared target head turns it into base logits `U`, and
    /// the confidence head reads it). `embeds` `[s, dim]` are the block token embeddings (from the
    /// shared target embedding); `hctx` `[n_ctx, dim]` is `encode_context`'s output; `q_start`/
    /// `ctx_start` are the block's / context's absolute RoPE positions; `input_ids` `[1, s]` drives
    /// MoE routing.
    #[allow(clippy::too_many_arguments)]
    pub fn backbone(
        &mut self,
        embeds: &Tensor,
        hctx: &Tensor,
        input_ids: &Tensor,
        ctx_start: usize,
        q_start: usize,
    ) -> Result<Tensor> {
        let (s, dim) = embeds.dims2()?;
        let x = embeds.to_dtype(candle::DType::F32)?.reshape((1, s, dim))?;
        // Split-borrow so each block's FFN closure can drive the shared `&mut moe` (LRU residency)
        // while iterating `&blocks` immutably.
        let Self {
            blocks,
            moe,
            hc,
            output_hc,
            output_norm,
            rope,
            cfg,
            ..
        } = self;
        let mut h = hc.expand(&x)?; // [1, s, hc_mult, dim]
        for (bi, block) in blocks.iter().enumerate() {
            h = block.forward_injected(&h, hctx, rope, ctx_start, q_start, |xn| {
                moe.forward(bi, xn, input_ids)
            })?;
        }
        let h = hc.head_reduce(&h, output_hc)?; // [1, s, dim]
        let h = rms_norm(&h, output_norm, cfg.base.norm_eps as f32)?;
        h.reshape((s, dim))
    }

    /// One full DSpark draft cycle. Builds the block `[committed, MASK×(block_size−1)]`, embeds
    /// it with the shared target embedding, encodes the target features into `Hctx`, runs the
    /// backbone, turns the hidden into base logits with the shared `lm_head`, and returns the
    /// confidence-scheduled block from the Markov sampler. `target_features` is
    /// `[n_ctx, n_target_layers·dim]`; `q_start` is the accepted length (the block's first
    /// absolute position).
    #[allow(clippy::too_many_arguments)]
    pub fn draft(
        &mut self,
        target_features: &Tensor,
        committed: u32,
        mask_token: u32,
        embed: &Tensor,
        lm_head: &QLinear,
        q_start: usize,
        tau: f32,
    ) -> Result<Vec<u32>> {
        let bs = self.cfg.block_size;
        // Block token ids: position 0 is the last committed token, the rest are the mask token.
        let mut ids: Vec<u32> = vec![mask_token; bs];
        ids[0] = committed;
        // The shared embedding is host-resident; look up there, then move the block to the
        // drafter's device (where the backbone + MoE routing run).
        let dev = self.output_norm.device().clone();
        let ids_cpu = Tensor::from_vec(ids.clone(), bs, embed.device())?;
        let embeds = embed.index_select(&ids_cpu, 0)?.to_device(&dev)?; // [bs, dim] on drafter dev
        let ids_t = Tensor::from_vec(ids, (1, bs), &dev)?; // [1, bs] routing ids on drafter dev

        let hctx = self.encode_context(target_features)?; // [n_ctx, dim]
                                                          // The `n_ctx`-wide Hctx is the sliding WINDOW of the last target hiddens, consecutive and
                                                          // ending at `q_start-1`; RoPE it at its true absolute positions `q_start-n_ctx .. q_start-1`
                                                          // (matching DSparkAttention's main_kv ring, which sits just below the draft block). A single
                                                          // vector at position 0 would be a full q_start away and the block's attention barely reaches it.
        let n_ctx = hctx.dim(0)?;
        let ctx_start = q_start.saturating_sub(n_ctx);
        let hidden = self.backbone(&embeds, &hctx, &ids_t, ctx_start, q_start)?; // [bs, dim]
                                                                                 // The shared LM head may live on a different device than the drafter (GPU target head +
                                                                                 // CPU drafter): round-trip the small [bs,dim]→[bs,vocab] through the head's device.
        let head_dev = lm_head.device();
        let logits = lm_head
            .forward(&hidden.to_device(&head_dev)?)?
            .to_device(hidden.device())?; // [bs, vocab] base logits U on the drafter device

        let base: Vec<Tensor> = (0..bs).map(|i| logits.i(i)).collect::<Result<_>>()?;
        let hid: Vec<Tensor> = (0..bs).map(|i| hidden.i(i)).collect::<Result<_>>()?;
        self.sample_block(&base, &hid, committed, tau)
    }

    /// Build the injected context `Hctx = RMSNorm_enc( fc(target_features) )` (paper Eq. 2).
    /// `target_features` is the per-token concatenation of the target model's hidden states at
    /// `cfg.target_layers` — shape `[.., n_target_layers · dim]`; the result is `[.., dim]`,
    /// which each backbone block projects through its own `attn_kv` and prepends to the draft
    /// KV (non-causal), per Eq. 3.
    pub fn encode_context(&self, target_features: &Tensor) -> Result<Tensor> {
        let projected = self.fc.forward(target_features)?;
        rms_norm(
            &projected,
            &self.enc_output_norm,
            self.cfg.base.norm_eps as f32,
        )
    }

    /// DSpark's semi-autoregressive **sequential stage** (paper Eq. 4-5-7, Alg. 1): given the
    /// backbone's per-position base logits `U₁..U_γ` and hiddens `h₁..h_γ` (one draft forward),
    /// sample the block left-to-right with the Markov transition bias, stopping at the
    /// confidence-scheduled length. `committed` is the last accepted token (feeds position 0).
    /// Returns `0..=block_size` drafted tokens (0 ⇒ fall back to a plain decode).
    pub fn sample_block(
        &self,
        base_logits: &[Tensor],
        hidden: &[Tensor],
        committed: u32,
        tau: f32,
    ) -> Result<Vec<u32>> {
        sample_sequential_device(
            &self.markov,
            self.confidence.as_ref(),
            self.cfg.block_size,
            base_logits,
            hidden,
            committed,
            tau,
        )
    }
}

/// [`sample_sequential`] with the chain kept ON DEVICE: each position's Markov
/// bias, argmax, and confidence are tensor ops whose data-dependent inputs
/// (the previous token) stay device-resident, and the whole block's tokens +
/// confidences come back in ONE transfer at the end. Bit-identical tokens —
/// the math is the same ops in the same order — but the former per-position
/// `to_scalar` pair (argmax + confidence) cost TWO full WDDM pipeline drains
/// per draft position, which made drafting a 5-token block cost more than the
/// verify wave it fed (~200 ms of the ~330 ms drafted-step wall).
fn sample_sequential_device(
    markov: &MarkovHead,
    confidence: Option<&ConfidenceHead>,
    block_size: usize,
    base_logits: &[Tensor],
    hidden: &[Tensor],
    committed: u32,
    tau: f32,
) -> Result<Vec<u32>> {
    let gamma = base_logits.len().min(block_size).min(hidden.len());
    if gamma == 0 {
        return Ok(Vec::new());
    }
    let dev = base_logits[0].device().clone();
    let mut prev = Tensor::from_vec(vec![committed], 1, &dev)?;
    let mut toks: Vec<Tensor> = Vec::with_capacity(gamma);
    let mut confs: Vec<Tensor> = Vec::with_capacity(gamma);
    for k in 0..gamma {
        let e = markov.embed_dev(&prev)?; // [1, rank]
        let col = base_logits[k].broadcast_add(&markov.bias_from_embed(&e)?)?; // [vocab]
        let t = col.argmax(0)?.reshape(1)?; // device [1] u32
        let c = match confidence {
            Some(cf) => cf.confidence_dev(&hidden[k], &e.reshape(markov.rank())?)?,
            None => softmax_last_dim(&col)?.index_select(&t, 0)?, // [1]
        };
        toks.push(t.clone());
        confs.push(c);
        prev = t;
    }
    // The block's ONLY host syncs: γ tokens + γ confidences, two tiny reads.
    let toks_h: Vec<u32> = Tensor::cat(&toks, 0)?.to_vec1::<u32>()?;
    let confs_h: Vec<f32> = Tensor::cat(&confs, 0)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    // Alg. 1's cutoff, exactly as the host form applies it: accumulate the
    // survival product and keep the longest prefix that clears τ.
    let mut drafts = Vec::with_capacity(gamma);
    let mut cum = 1.0f32;
    for k in 0..gamma {
        cum *= confs_h[k];
        if cum < tau {
            break;
        }
        drafts.push(toks_h[k]);
    }
    Ok(drafts)
}

/// The DSpark sequential sampler (paper Eq. 4-5-7 + Alg. 1) in its scalar
/// host form — the readable reference [`sample_sequential_device`] is held
/// bit-identical to (`device_sampler_matches_host_reference`). At position
/// `k`: `col = U_k + W₁[prev]·W₂` (Eq. 5), `t = argmax(col)`, and the
/// survival probability `c_k` (confidence head Eq. 7, else the max-softmax
/// proxy) accumulates into `∏ c_i`; drafting stops the first time the
/// cumulative product drops below `tau`.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn sample_sequential(
    markov: &MarkovHead,
    confidence: Option<&ConfidenceHead>,
    block_size: usize,
    base_logits: &[Tensor],
    hidden: &[Tensor],
    committed: u32,
    tau: f32,
) -> Result<Vec<u32>> {
    let gamma = base_logits.len().min(block_size).min(hidden.len());
    let mut drafts = Vec::with_capacity(gamma);
    let mut prev = committed;
    let mut cum = 1.0f32;
    for k in 0..gamma {
        let col = base_logits[k].broadcast_add(&markov.bias(prev)?)?;
        let t = col.argmax(0)?.to_scalar::<u32>()?;
        let c = match confidence {
            Some(cf) => cf.confidence(&hidden[k], &markov.embed(prev)?)?,
            None => softmax_last_dim(&col)?
                .narrow(0, t as usize, 1)?
                .squeeze(0)?
                .to_scalar::<f32>()?,
        };
        cum *= c;
        if cum < tau {
            break; // Alg. 1: keep the longest prefix whose cumulative survival clears τ.
        }
        drafts.push(t);
        prev = t;
    }
    Ok(drafts)
}

/// Orient a 2-D weight so `want` rows are on dim 0 (transposing if it's stored the other
/// way), erroring if neither axis matches.
fn orient_rows(t: Tensor, want: usize, name: &str) -> Result<Tensor> {
    if t.dim(0)? == want {
        Ok(t)
    } else if t.dim(1)? == want {
        t.t()?.contiguous()
    } else {
        candle::bail!("{name} shape {:?} has no axis of size {want}", t.dims())
    }
}

/// Read a `u32`/`i32` metadata array (e.g. `dflash.target_layers`), empty when absent.
fn u32_array(m: &GgufModel, key: &str) -> Vec<usize> {
    match m.metadata.get(key).and_then(|v| v.to_vec().ok()) {
        Some(arr) => arr
            .iter()
            .filter_map(|x| {
                x.to_u32()
                    .ok()
                    .or_else(|| x.to_i32().ok().map(|i| i as u32))
            })
            .map(|x| x as usize)
            .collect(),
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // End-to-end gates against the real V4 target + its dflash drafter.
    #[cfg(feature = "cuda")]
    use crate::models::deepseek4::DEEPSEEK_V4;
    #[cfg(feature = "cuda")]
    use crate::models::deepseek4::DFLASH;
    use candle::DType;

    /// Markov bias equals `W₁[prev] · W₂` computed by hand, byte-for-byte (paper Eq. 5).
    #[test]
    fn markov_bias_matches_hand_computation() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, rank) = (4usize, 2usize);
        // W1 [vocab, rank], W2 [rank, vocab] — small, exact values.
        let w1 = Tensor::from_vec(
            vec![1f32, 2., 3., 4., 5., 6., 7., 8.], // rows: t0=[1,2] t1=[3,4] t2=[5,6] t3=[7,8]
            (vocab, rank),
            &dev,
        )?;
        let w2 = Tensor::from_vec(
            vec![1f32, 0., -1., 2., 0., 1., 3., -2.], // [2,4]
            (rank, vocab),
            &dev,
        )?;
        let head = MarkovHead::new(w1, w2, vocab)?;
        assert_eq!(head.rank(), rank);

        // prev = 1 → row [3,4]; bias = [3*1+4*0, 3*0+4*1, 3*(-1)+4*3, 3*2+4*(-2)]
        //                              = [3, 4, 9, -2]
        let bias = head.bias(1)?.to_vec1::<f32>()?;
        assert_eq!(bias, vec![3.0, 4.0, 9.0, -2.0]);

        // embed(1) is exactly the looked-up row.
        assert_eq!(head.embed(1)?.to_vec1::<f32>()?, vec![3.0, 4.0]);
        Ok(())
    }

    /// A `[rank, vocab]`-oriented W1 (the transposed storage) is normalised to `[vocab, rank]`
    /// and yields the same bias — the loader must accept either GGUF orientation.
    #[test]
    fn markov_accepts_transposed_w1() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, rank) = (4usize, 2usize);
        // Same W1 as above but stored transposed: [rank, vocab].
        let w1_t = Tensor::from_vec(vec![1f32, 3., 5., 7., 2., 4., 6., 8.], (rank, vocab), &dev)?;
        let w2 = Tensor::from_vec(
            vec![1f32, 0., -1., 2., 0., 1., 3., -2.],
            (rank, vocab),
            &dev,
        )?;
        let head = MarkovHead::new(w1_t, w2, vocab)?;
        assert_eq!(head.bias(1)?.to_vec1::<f32>()?, vec![3.0, 4.0, 9.0, -2.0]);
        Ok(())
    }

    /// A weight matching neither the vocab axis nor its transpose is a hard load error.
    #[test]
    fn markov_rejects_mismatched_shape() {
        let dev = Device::Cpu;
        let w1 = Tensor::from_vec(vec![0f32; 3 * 2], (3, 2), &dev).unwrap();
        let w2 = Tensor::from_vec(vec![0f32; 2 * 4], (2, 4), &dev).unwrap();
        assert!(
            MarkovHead::new(w1, w2, 4).is_err(),
            "vocab=4 not in w1 [3,2]"
        );
    }

    /// Build the tiny fixture Markov head used by the sequential-sampler tests:
    /// W1 rows t0=[1,2] t1=[3,4] t2=[5,6] t3=[7,8]; W2 = [[1,0,-1,2],[0,1,3,-2]].
    fn fixture_markov(dev: &Device) -> Result<MarkovHead> {
        let w1 = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6., 7., 8.], (4, 2), dev)?;
        let w2 = Tensor::from_vec(vec![1f32, 0., -1., 2., 0., 1., 3., -2.], (2, 4), dev)?;
        MarkovHead::new(w1, w2, 4)
    }

    /// The full sequential stage (Eq. 5 bias + argmax), with `tau = 0` so nothing is pruned:
    /// each position's token is `argmax(U_k + W₁[prev]·W₂)`, chained through `prev`.
    #[test]
    fn sample_sequential_applies_markov_bias_and_chains_prev() -> Result<()> {
        let dev = Device::Cpu;
        let markov = fixture_markov(&dev)?;
        let u = |v: Vec<f32>| Tensor::from_vec(v, 4, &dev).unwrap();
        // committed = 0 → bias(0) = [1,2,5,-2]; +U0=[0,0,0,10] → [1,2,5,8] → argmax 3.
        // prev = 3 → bias(3) = [7,8,17,-2]; +U1=0 → argmax 2.
        // prev = 2 → bias(2) = [5,6,13,-2]; +U2=[100,0,0,0] → [105,6,13,-2] → argmax 0.
        let base = vec![
            u(vec![0., 0., 0., 10.]),
            u(vec![0.; 4]),
            u(vec![100., 0., 0., 0.]),
        ];
        let hidden = vec![u(vec![0.; 4]); 3]; // unused when confidence is None + tau=0
        let drafts = sample_sequential(&markov, None, 3, &base, &hidden, 0, 0.0)?;
        assert_eq!(drafts, vec![3, 2, 0]);
        Ok(())
    }

    /// Confidence scheduling (Eq. 7 + Alg. 1): a head that emits `σ(0)=0.5` per position makes
    /// the cumulative product `0.5, 0.25, …`. τ selects the length: τ=0.3 keeps 1 (0.5≥0.3,
    /// 0.25<0.3), τ=0.6 keeps 0 (0.5<0.6 ⇒ a plain decode).
    #[test]
    fn confidence_schedule_picks_prefix_by_cumulative_product() -> Result<()> {
        let dev = Device::Cpu;
        let markov = fixture_markov(&dev)?;
        let conf = ConfidenceHead::new(Tensor::zeros(4, DType::F32, &dev)?); // hidden(2)+rank(2)
        let u = |v: Vec<f32>| Tensor::from_vec(v, 4, &dev).unwrap();
        let base = vec![
            u(vec![0., 0., 0., 10.]),
            u(vec![0.; 4]),
            u(vec![100., 0., 0., 0.]),
        ];
        let hidden = vec![Tensor::zeros(2, DType::F32, &dev)?; 3];

        let d_keep1 = sample_sequential(&markov, Some(&conf), 3, &base, &hidden, 0, 0.3)?;
        assert_eq!(d_keep1, vec![3], "0.5≥0.3 then 0.25<0.3 → length 1");

        let d_keep0 = sample_sequential(&markov, Some(&conf), 3, &base, &hidden, 0, 0.6)?;
        assert!(d_keep0.is_empty(), "0.5<0.6 → 0 drafts (plain decode)");
        Ok(())
    }

    /// The device-chained sampler (the runtime form: no per-position host sync)
    /// is token-identical to the scalar host reference across taus, with and
    /// without a confidence head.
    #[test]
    fn device_sampler_matches_host_reference() -> Result<()> {
        let dev = Device::Cpu;
        let markov = fixture_markov(&dev)?;
        let u = |v: Vec<f32>| Tensor::from_vec(v, 4, &dev).unwrap();
        let base = vec![
            u(vec![0., 0., 0., 10.]),
            u(vec![0.; 4]),
            u(vec![100., 0., 0., 0.]),
        ];
        let hidden = vec![Tensor::zeros(2, DType::F32, &dev)?; 3];
        let conf = ConfidenceHead::new(Tensor::from_vec(vec![0.3f32, -0.2, 1.0, 0.4], 4, &dev)?);
        for tau in [0.0f32, 0.3, 0.6, 0.9] {
            for c in [None, Some(&conf)] {
                let host = sample_sequential(&markov, c, 3, &base, &hidden, 0, tau)?;
                let devf = sample_sequential_device(&markov, c, 3, &base, &hidden, 0, tau)?;
                assert_eq!(host, devf, "tau={tau} conf={}", c.is_some());
            }
        }
        Ok(())
    }

    /// Confidence head is `σ(wᵀ[h; markov_embed])` (paper Eq. 7): a zero pre-activation → 0.5,
    /// and a known non-zero dot matches the sigmoid to f32 precision.
    #[test]
    fn confidence_is_sigmoid_of_concat_dot() -> Result<()> {
        let dev = Device::Cpu;
        // hidden = 2, rank = 2 → w over [h0,h1,e0,e1].
        let h = Tensor::from_vec(vec![1f32, -1.], 2, &dev)?;
        let e = Tensor::from_vec(vec![2f32, 0.], 2, &dev)?;

        // All-zero weight → z = 0 → σ(0) = 0.5.
        let c0 = ConfidenceHead::new(Tensor::zeros(4, DType::F32, &dev)?).confidence(&h, &e)?;
        assert!((c0 - 0.5).abs() < 1e-6);

        // w = [1, 1, 0.5, -3] → z = 1*1 + 1*(-1) + 0.5*2 + (-3)*0 = 1.0 → σ(1).
        let w = Tensor::from_vec(vec![1f32, 1., 0.5, -3.], 4, &dev)?;
        let c1 = ConfidenceHead::new(w).confidence(&h, &e)?;
        let want = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((c1 - want).abs() < 1e-6, "got {c1}, want {want}");
        Ok(())
    }

    /// Floor for an OPTIMIZED CPU drafter: time a single pre-dequantized f32 expert projection
    /// matmul (candle's gemm is already Rayon-threaded), at draft (5-token) and batched (40-token)
    /// widths, then scale to a full draft cycle (~30 active experts/block × 3 blocks × 3
    /// projections ≈ 270 matmuls). Shows what killing the per-cycle MXFP4 dequant + 256-loop buys.
    #[test]
    #[ignore]
    fn bench_cpu_expert_matmul_floor() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1.0, (2048usize, 4096usize), &dev)?; // one projection weight, f32
        let time_at = |n: usize| -> Result<f64> {
            let x = Tensor::randn(0f32, 1.0, (n, 4096usize), &dev)?;
            let _ = x.matmul(&w.t()?)?; // warm
            let iters = 100;
            let t = std::time::Instant::now();
            for _ in 0..iters {
                let _ = x.matmul(&w.t()?)?;
            }
            Ok(t.elapsed().as_secs_f64() * 1e6 / iters as f64) // µs/matmul
        };
        let per_matmuls = 270.0; // ~30 experts/block × 3 blocks × 3 projections
        for n in [5usize, 40] {
            let us = time_at(n)?;
            eprintln!(
                "[cpu-expert] N={n:2}: {:.1} µs/matmul → ~{:.1} ms/draft-cycle (×270 matmuls)",
                us,
                us * per_matmuls / 1000.0
            );
        }
        eprintln!(
            "[cpu-expert] (N=40 ≈ 8 sessions batched; per-token cost = cycle/(8·2.4 accepted))"
        );
        Ok(())
    }

    /// Measure JUST the MoE router (`Gate::route`) cost on CPU with the drafter's dims — the
    /// "which experts to stream" decision, isolated from the expensive expert matmuls. Confirms
    /// the routing/streaming decision is nearly free vs the 1 s full-cycle (which is the reference
    /// MoE's 256-expert dense loop + per-expert readback, not the router).
    #[test]
    #[ignore]
    fn bench_cpu_router() -> Result<()> {
        use super::super::moe::{Gate, ScoreFunc};
        let (n_exp, dim, k) = (256usize, 4096usize, 6usize);
        // Time the dequantized router (dense f32 weight) at N tokens on `dev`; `sync` synchronizes
        // (GPU is async — needed for honest wall-clock). Returns µs/block route.
        let bench = |dev: &Device, n_tok: usize, sync: bool| -> Result<f64> {
            let gate = Gate::new(
                QLinear::from_weight(Tensor::randn(0f32, 1.0, (n_exp, dim), dev)?),
                None,
                None,
                k,
                n_exp,
                ScoreFunc::SqrtSoftplus,
                1.5,
            );
            let x = Tensor::randn(0f32, 1.0, (n_tok, dim), dev)?;
            let ids = Tensor::zeros(n_tok, candle::DType::U32, dev)?;
            let _ = gate.route(&x, &ids)?;
            if sync {
                dev.synchronize()?;
            }
            let n = 200;
            let t = std::time::Instant::now();
            for _ in 0..n {
                let _ = gate.route(&x, &ids)?;
            }
            if sync {
                dev.synchronize()?;
            }
            Ok(t.elapsed().as_secs_f64() * 1e6 / n as f64)
        };
        let cpu = Device::Cpu;
        for n_tok in [5usize, 40, 320] {
            let c = bench(&cpu, n_tok, false)?;
            eprintln!(
                "[router] CPU  N={n_tok:3}: {c:8.1} µs/block  ({:.1} µs/cycle ×3)",
                c * 3.0
            );
            #[cfg(feature = "cuda")]
            {
                let gpu = Device::new_cuda(0)?;
                let g = bench(&gpu, n_tok, true)?;
                eprintln!(
                    "[router] CUDA N={n_tok:3}: {g:8.1} µs/block  ({:.1} µs/cycle ×3)",
                    g * 3.0
                );
            }
        }
        Ok(())
    }

    /// Inspect the real DSpark GGUF: total size, expert count, and a per-category byte breakdown
    /// (experts vs attention vs markov/fc/heads) — to see exactly how big the drafter actually is
    /// and whether anything is redundant with the target. Header-only (no GPU, no full load).
    #[test]
    #[ignore]
    fn inspect_real_dspark_sizes() -> Result<()> {
        use candle::quantized::GgmlDType;
        let path = std::path::PathBuf::from(
            r"D:\models\deepseek-v4-flash-mxfp4\dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf",
        );
        if !path.exists() {
            eprintln!("[skip] DSpark drafter absent");
            return Ok(());
        }
        let m = GgufModel::open(std::slice::from_ref(&path))?;
        // Metadata of interest.
        for k in [
            "dflash.block_count",
            "dflash.expert_count",
            "dflash.expert_used_count",
            "dflash.expert_feed_forward_length",
            "dflash.embedding_length",
            "dflash.block_size",
        ] {
            eprintln!("  {k} = {:?}", m.metadata.get(k).map(|v| format!("{v:?}")));
        }
        eprintln!(
            "  tokenizer.ggml.mask_token_id = {:?}",
            m.metadata_u32("tokenizer.ggml.mask_token_id")
        );
        let names = m.tensor_names();
        let bytes = |dt: GgmlDType, elems: usize| elems / dt.block_size() * dt.type_size();
        let mut cat: std::collections::BTreeMap<&str, (usize, u64)> = Default::default();
        let mut total = 0u64;
        for n in &names {
            let info = m.info(n).unwrap();
            let elems: usize = info.shape.elem_count();
            let b = bytes(info.ggml_dtype, elems) as u64;
            total += b;
            let c = if n.contains("_exps.weight") {
                "routed_experts"
            } else if n.contains("_shexp") {
                "shared_experts"
            } else if n.contains("attn") {
                "attention"
            } else if n.contains("markov") {
                "markov"
            } else if n.contains("fc") || n.contains("enc.") || n.contains("conf") {
                "encoder/conf"
            } else if n.contains("hc_") || n.contains("norm") {
                "norms/hc"
            } else {
                "other"
            };
            let e = cat.entry(c).or_default();
            e.0 += 1;
            e.1 += b;
        }
        let gb = |b: u64| b as f64 / (1u64 << 30) as f64;
        eprintln!(
            "=== DSpark tensor breakdown ({} tensors, {:.2} GB total) ===",
            names.len(),
            gb(total)
        );
        for (c, (n, b)) in &cat {
            eprintln!("  {c:16} {n:3} tensors  {:.2} GB", gb(*b));
        }
        // Print the shape+dtype of one routed-expert tensor.
        if let Some(en) = names.iter().find(|n| n.contains("ffn_gate_exps.weight")) {
            let i = m.info(en).unwrap();
            eprintln!("  e.g. {en}: {:?} {:?}", i.ggml_dtype, i.shape.dims());
        }
        Ok(())
    }

    /// Are DSpark's routed experts the SAME bytes as the target's (⇒ shareable, no load), or
    /// distinct draft-trained weights (⇒ must be streamed)? Compare expert-0 of each DSpark block
    /// against every target layer's expert-0. Header + one-expert reads only.
    #[test]
    #[ignore]
    fn compare_dspark_vs_target_experts() -> Result<()> {
        let dpath = std::path::PathBuf::from(
            r"D:\models\deepseek-v4-flash-mxfp4\dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf",
        );
        let tpath = std::path::PathBuf::from(
            r"D:\models\deepseek-v4-flash-mxfp4\DeepSeek-V4-Flash-0731-MXFP4-merged.gguf",
        );
        if !dpath.exists() || !tpath.exists() {
            eprintln!("[skip] dspark or merged target absent");
            return Ok(());
        }
        // Read one expert's raw bytes (expert 0 of `name`) from a GgufModel.
        fn expert0(m: &mut GgufModel, name: &str) -> Result<Vec<u8>> {
            let info = m.info(name).ok_or_else(|| candle::Error::msg("missing"))?;
            let dt = info.ggml_dtype;
            let per =
                info.shape.dims()[1..].iter().product::<usize>() / dt.block_size() * dt.type_size();
            let q = m.qtensor(name, &Device::Cpu)?;
            Ok(q.data_range(0..per)?.into_owned())
        }
        let mut dm = GgufModel::open(std::slice::from_ref(&dpath))?;
        let mut tm = GgufModel::open(std::slice::from_ref(&tpath))?;
        let n_tgt = super::super::loader::config_from_gguf(&tm, &DEEPSEEK_V4)?.n_layers;
        for db in 0..3usize {
            let de = expert0(&mut dm, &format!("blk.{db}.ffn_gate_exps.weight"))?;
            let mut matched = None;
            for tl in 0..n_tgt {
                let te = expert0(&mut tm, &format!("blk.{tl}.ffn_gate_exps.weight"))?;
                if te == de {
                    matched = Some(tl);
                    break;
                }
            }
            match matched {
                Some(tl) => eprintln!("  DSpark blk.{db} expert0 == target blk.{tl} (SHARED!)"),
                None => eprintln!(
                    "  DSpark blk.{db} expert0 matches NO target layer (distinct draft weights)"
                ),
            }
        }
        Ok(())
    }

    /// On-device validation against the REAL DSpark drafter weights: the loader parses the
    /// `dflash` config, loads every tensor, and the encoder + sequential sampler run on the GPU
    /// with the actual `fc`/`markov`/`conf_proj` shapes. Skips when the drafter GGUF is absent
    /// (fetch it with `zend --download-deepseek`). This is the real-weight, on-GPU counterpart
    /// to the synthetic unit tests above.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn real_dspark_loads_and_runs_on_gpu() -> Result<()> {
        let path = std::path::PathBuf::from(
            r"D:\models\deepseek-v4-flash-mxfp4\dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf",
        );
        if !path.exists() {
            eprintln!("[skip] DSpark drafter absent — run `zend --download-deepseek`");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let mut drafter = DsparkDrafter::load(&path, &DFLASH, &device)?;
        eprintln!(
            "[dspark] blocks={} block_size={} target_layers={:?} markov(vocab={},rank={}) conf={}",
            drafter.blocks.len(),
            drafter.cfg.block_size,
            drafter.cfg.target_layers,
            drafter.markov.vocab(),
            drafter.markov.rank(),
            drafter.confidence.is_some(),
        );
        assert_eq!(drafter.blocks.len(), drafter.cfg.base.n_layers);
        assert!(
            !drafter.blocks.is_empty(),
            "drafter must have backbone blocks"
        );
        assert_eq!(drafter.markov.vocab(), drafter.cfg.base.vocab_size);

        // Encoder Hctx = enc_norm(fc(target_features)): synthetic [seq, m·dim] → [seq, dim].
        let dim = drafter.cfg.base.dim;
        let m = drafter.cfg.target_layers.len().max(1);
        let feats = Tensor::randn(0f32, 1.0, (2, m * dim), &device)?;
        let hctx = drafter.encode_context(&feats)?;
        assert_eq!(hctx.dims2()?.1, dim, "Hctx last dim == model dim");
        assert!(hctx
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| v.is_finite()));

        // Sequential sampler on synthetic backbone outputs, using the REAL markov + confidence
        // heads (so conf_proj's real shape must match [hidden ; markov-embedding]).
        let vocab = drafter.markov.vocab();
        let hdim = match &drafter.confidence {
            Some(cf) => cf.feat_len()? - drafter.markov.rank(),
            None => dim,
        };
        let base: Vec<Tensor> = (0..drafter.cfg.block_size)
            .map(|_| Tensor::randn(0f32, 1.0, vocab, &device))
            .collect::<Result<_>>()?;
        let hidden: Vec<Tensor> = (0..drafter.cfg.block_size)
            .map(|_| Tensor::randn(0f32, 1.0, hdim, &device))
            .collect::<Result<_>>()?;
        let drafts = drafter.sample_block(&base, &hidden, 0, 0.0)?; // τ=0 → full block
        assert_eq!(drafts.len(), drafter.cfg.block_size);
        assert!(drafts.iter().all(|&t| (t as usize) < vocab));
        eprintln!("[dspark] on-GPU drafted block (τ=0): {drafts:?}");

        // Backbone forward: run the block `[committed, MASK×(bs-1)]` through the 3 REAL
        // injected-KV non-causal blocks on the GPU (Hctx from the encoder above), producing the
        // per-position hidden the shared head + sampler consume. Validates `forward_injected` +
        // the mHC expand/head-reduce + MoE end-to-end on real weights (finite, right shape).
        let bs = drafter.cfg.block_size;
        let embeds = Tensor::randn(0f32, 1.0, (bs, dim), &device)?;
        let ids = Tensor::from_vec(vec![0u32; bs], (1, bs), &device)?;
        let bb = drafter.backbone(&embeds, &hctx, &ids, 0, bs)?;
        assert_eq!(bb.dims2()?, (bs, dim));
        assert!(bb
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| v.is_finite()));
        eprintln!("[dspark] on-GPU backbone hidden finite [{bs},{dim}]");
        Ok(())
    }
}
