//! The NextN / MTP draft head this checkpoint's release ships.
//!
//! Speculative decode needs something that proposes tokens more cheaply than the
//! trunk scores them. The released `Qwen3.8-Flash-Next` carries a one-block
//! head for exactly that (`mtp_num_hidden_layers: 1`,
//! `mtp_use_dedicated_embeddings: false`), and it is folded into the engine
//! artifact as `blk.{num_layers}` — see `docs/qwen38_flash_next.md` §14 for why
//! the GGUF lineage lacked one and where these weights come from.
//!
//! # It is a layer of the model, not a sidecar
//!
//! The head loads from `blk.{num_layers}` through the same tensor names every
//! trunk block uses, and [`MtpHead::block`] is literally a [`GpuLayer`] — same
//! hyper-connections, same 512-expert MoE, same QSA indexer, same shapes. So it
//! runs the production path a trunk attention layer runs, holds its KV in the
//! same paged cache as an ordinary layer, and — the part that matters most here
//! — its experts join the same grid, streaming over PCIe and offloading through
//! the same three tiers rather than sitting resident. Prefill never touches
//! them, so they cost nothing exactly when the trunk needs the room most.
//!
//! That uniformity is the point. A head carrying private weights would be a
//! special case in every session-wide operation — fork, view, prefix injection,
//! turn sealing — each of which assumes a sequence's layers describe one stream
//! at one length.
//!
//! # What the head adds beyond a trunk block
//!
//! The input assembly, and it is the only genuinely new arithmetic:
//!
//! ```text
//!   enorm(embed(token))              [n_embd]     the token being followed
//!   hnorm(residual)                  [hc_dim]     the trunk's carried state
//!   mixer(hnorm(residual))           [n_embd]     collapsed to the block width
//!   eh_proj([enorm ; mixed])         [n_embd]     the block's input
//! ```
//!
//! The `mixer` step is why this is not qwen35's head verbatim: there, `hnorm`
//! is over the narrow hidden and `eh_proj` takes their plain concat. Here
//! `hnorm` is `[hc_dim]` — 10240 — while `eh_proj` is `[n_embd, 2·n_embd]`, so
//! something has to collapse the wide residual first. That something is the
//! head's own hyper-connection mixer, which the converter emitted under the
//! trunk's `output_hc_*` names because a standalone head has only one mixer.
//! Merged, that name is taken, so it is renamed into the head's own block
//! (`convert_mtp_sidecar`). It is **not** a copy of the trunk's output mix:
//! compared against the pinned source it differs by 9.8 / 2.6 / 14.2, while
//! `token_embd` and `output` are bit-identical — which is what
//! `mtp_use_dedicated_embeddings: false` promises and what makes the shared
//! embedding and LM head safe to drop.

use candle::{Device, Result, Tensor};

use super::config::Qwen4ExpConfig;
use super::engine::GpuLayer;
use super::hyper::{hc_grouped_norm, hc_mix, HcWeights};
use crate::models::latent_moe::GgufModel;
use crate::models::quantized_matmul::QMatMul;
use crate::quantized_nn::RmsNorm;

/// The `[enorm ; mixer(hnorm)] → eh_proj` input assembly.
pub struct MtpInput {
    /// RMSNorm over the token embedding — a norm, not a bare gain, for the
    /// same reason every trunk norm is one: the scale of what reaches
    /// `eh_proj` must not depend on the magnitude of the row it came from.
    pub enorm: RmsNorm,
    /// `[hc_dim]` — the gain of a **grouped** norm over the carried wide
    /// residual, applied through [`hc_grouped_norm`] like every other `[hc_dim]`
    /// norm weight in this stack.
    ///
    /// Not an [`RmsNorm`]: the width is what tells them apart. A grouped norm
    /// reduces per stream, over `n_embd`, and then applies the `[hc_dim]` gain
    /// flat — so a plain RMSNorm here would reduce over all 10240 and hand the
    /// head a differently-scaled input than the one it was trained on. Every
    /// `[hc_dim]` weight in the checkpoint (`hc_attn_norm`, `hc_ffn_norm`,
    /// `hc_mixer_norm`, and this) is the grouped kind; `enorm` is `[n_embd]`
    /// and is the plain kind.
    pub hnorm: Tensor,
    /// `[n_embd, 2·n_embd]` over the concat.
    pub eh_proj: QMatMul,
}

/// One NextN draft head.
pub struct MtpHead {
    pub input: MtpInput,
    /// The head's block — the same type, and the same production path, as a
    /// trunk attention layer.
    pub block: GpuLayer,
    /// The head's own output hyper-connection mix, collapsing its block's wide
    /// residual to `n_embd` for the shared LM head — structurally identical to
    /// the trunk's `out_hc` (`{norm, down, up}`, no inject), which is what it
    /// is: the converter emitted it as `output_hc_*`, and a standalone head
    /// file carries exactly one because it has exactly one output.
    pub mixer: HcWeights,
    /// `[hc_dim]` — grouped-norm gain before that mix. Named for what follows
    /// it: the **shared** head, whose LM projection this checkpoint shares with
    /// the trunk along with the embedding table.
    pub head_norm: Tensor,
    /// Trunk block index the head sits at, which is also its tensor prefix.
    pub layer_index: usize,
}

/// The head's own dense weights, before its block is attached.
///
/// These exist apart from [`MtpHead`] because of **when** they must be read.
/// The engine's load order is load-bearing: every dense tensor resident first,
/// then the expert cache sized from a live measurement of what they left behind
/// (`docs/archived/elastic_vram_partition.md` §4). The head's dense side is ~30 MB —
/// `hc_mixer_down` and `hc_mixer_up` are `[320, 10240]` and `[10240, 320]` F32,
/// 13.1 MB each, plus `eh_proj` and three norms, each with a transient F32
/// dequant buffer on top — so reading it after that measurement takes ground
/// the expert zone has already been told it owns. Loading it with the rest of
/// the dense stack keeps the measurement honest.
pub struct MtpDense {
    input: MtpInput,
    mixer: HcWeights,
    head_norm: Tensor,
    layer_index: usize,
}

impl MtpDense {
    /// Attach the block the engine's per-layer path built for `blk.{num_layers}`.
    pub fn with_block(self, block: GpuLayer) -> MtpHead {
        MtpHead {
            input: self.input,
            block,
            mixer: self.mixer,
            head_norm: self.head_norm,
            layer_index: self.layer_index,
        }
    }

    /// Load the head's own tensors.
    ///
    /// `eps` is the model's RMS epsilon: the head norms in the same arithmetic
    /// the trunk does, because a drafted position and the wave position that
    /// later replaces it must agree — or the K/V the draft attended over and the
    /// K/V the verify wrote disagree on a token both accepted.
    pub fn load(
        g: &mut GgufModel,
        cfg: &Qwen4ExpConfig,
        eps: f64,
        device: &Device,
    ) -> Result<Self> {
        let li = cfg.num_layers;
        let p = format!("blk.{li}");
        let f32t = |g: &mut GgufModel, name: &str| -> Result<Tensor> {
            g.qtensor(name, device)?.dequantize(device)
        };
        let input = MtpInput {
            enorm: RmsNorm::from_qtensor(
                g.qtensor(&format!("{p}.nextn.enorm.weight"), device)?,
                eps,
            )?,
            hnorm: f32t(g, &format!("{p}.nextn.hnorm.weight"))?,
            eh_proj: QMatMul::from_qtensor(
                g.qtensor(&format!("{p}.nextn.eh_proj.weight"), device)?,
            )?,
        };
        let mixer = HcWeights {
            norm: f32t(g, &format!("{p}.hc_mixer_norm.weight"))?,
            down: f32t(g, &format!("{p}.hc_mixer_down.weight"))?,
            up: f32t(g, &format!("{p}.hc_mixer_up.weight"))?,
        };
        let head_norm = f32t(g, &format!("{p}.nextn.shared_head_norm.weight"))?;
        Ok(Self {
            input,
            mixer,
            head_norm,
            layer_index: li,
        })
    }
}

impl MtpHead {
    /// The head's block input for one row per sequence.
    ///
    /// `embed` is `[n, n_embd]` — the embedding of the token each sequence is
    /// following — and `residual` the trunk's carried **wide** state
    /// `[n, hc, n_embd]`. Returns `[n, n_embd]`, ready for the block.
    ///
    /// Expressed through the model's own [`hc_mix`] rather than a second
    /// transcription of it. The head collapses the wide residual with exactly
    /// the arithmetic every trunk pre-mix uses — same gate, same mean, same
    /// kernel path — because a divergence here is not an error, it is a drafter
    /// proposing from a slightly different model than the one verifying, which
    /// shows up only as an accept rate that never justifies the head.
    ///
    /// `hc_mix` reads [`HcWeights::injects`] off the weight's own shape, and
    /// the head's mixer has no inject rows, so the same call that returns a
    /// trunk layer's `(mixed, Some(inject))` returns `(mixed, None)` here.
    pub fn assemble(&self, embed: &Tensor, residual: &Tensor, eps: f64) -> Result<Tensor> {
        let (n, hc, n_embd) = residual.dims3()?;
        // **Per hyper-connection stream, on the wide hidden — never a mean
        // first.** `eh_proj` is applied once per stream, so its output is
        // already the wide residual the head's block runs on: the embedding is
        // *folded into* the trunk's wide residual rather than replacing it.
        //
        // Collapsing to `n_embd` first and lifting the result back is the one
        // mistake this assembly invites, because it type-checks, runs, and
        // produces fluent proposals. The reference calls it out directly — "the
        // combiner must be run per hyper-connection stream on the wide hidden
        // state; if you do mean pooling first, the acceptance rate drops
        // catastrophically" (llama.cpp#27836) — and measured here it did
        // exactly that: plausible tokens that were never the trunk's.
        let hn = hc_grouped_norm(residual, &self.input.hnorm, eps)?.reshape((n * hc, n_embd))?;
        // **Embedding first, hidden second**: `eh_proj` fuses the checkpoint's
        // `fc_embedding` and `fc_hidden` side by side, so the one matmul
        // computes `fc_embedding @ e + fc_hidden @ h`. One `[n_embd, 2·n_embd]`
        // weight spans both halves, so nothing in the shapes can check the
        // order; measured the other way round the proposals got strictly worse.
        let en = self
            .input
            .enorm
            .forward_live(embed)?
            .reshape((n, 1, n_embd))?
            .broadcast_as((n, hc, n_embd))?
            .reshape((n * hc, n_embd))?;
        let cat = Tensor::cat(&[&en, &hn], 1)?;
        self.input
            .eh_proj
            .forward_live(&cat)?
            .reshape((n, hc, n_embd))
    }

    /// Collapse the head's block output to the width the **shared** LM head
    /// takes.
    ///
    /// `shared_head_norm` is `[hc_dim]`, so it is a grouped norm over the wide
    /// residual like every other `[hc_dim]` weight here — and its name says
    /// what it precedes: the shared head, which on this stack is the trunk's
    /// own output mix followed by the shared `lm_head`. The head carries the
    /// norm and borrows the rest, exactly as it borrows the embedding table.
    pub fn to_shared_head(&self, block_out: &Tensor, eps: f64) -> Result<Tensor> {
        let normed = hc_grouped_norm(block_out, &self.head_norm, eps)?;
        let (narrow, _) = hc_mix(&normed, &self.mixer, eps)?;
        Ok(narrow)
    }

    /// The head's block input — `[n, hc, n_embd]`, the wide residual its layer
    /// runs on.
    ///
    /// No lift: [`Self::assemble`] already works per stream, so what it returns
    /// *is* the wide residual. A trunk block reaches that shape by broadcasting
    /// a narrow embedding across the streams, and this deliberately does not —
    /// the head folds its embedding into the trunk's carried streams instead of
    /// starting fresh ones, which is what makes its proposals conditional on
    /// the state the trunk actually built.
    pub fn block_input(&self, embed: &Tensor, residual: &Tensor, eps: f64) -> Result<Tensor> {
        self.assemble(embed, residual, eps)?.contiguous()
    }
}
