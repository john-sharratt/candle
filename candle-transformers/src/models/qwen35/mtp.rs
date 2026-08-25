//! The NextN / MTP draft head these checkpoints ship.
//!
//! Qwen3.5/3.6 are trained with multi-token prediction: past the trunk's
//! `num_hidden_layers` blocks the checkpoint carries `nextn_predict_layers`
//! more, and each is a **draft head** that predicts one token further ahead
//! than the trunk did. SGLang serves it as the NEXTN speculative algorithm;
//! here it is the drafter behind
//! [`ManagedBatchedModel::speculative_draft`](crate::models::batched_inference::ManagedBatchedModel::speculative_draft).
//! A checkpoint without one simply does not speculate — the engine decodes it
//! a token at a time, which is what the whole lineage did before the head
//! landed. See [`super::draft`] for how the engine runs it.
//!
//! # What the head is
//!
//! **Structurally a trunk layer.** The GGUF gives `blk.N` (N = the trunk's
//! depth) exactly the tensor set of a trunk attention layer — `attn_q`,
//! `attn_k`, `attn_v`, `attn_output`, the Q/K norms, `attn_norm`,
//! `post_attention_norm`, and the layer's FFN — so the block loads and runs
//! through the same code as any other layer rather than a second transcription
//! of it. Four tensors are its own:
//!
//! ```text
//!   blk.N.nextn.enorm.weight             RMSNorm over the token embedding
//!   blk.N.nextn.hnorm.weight             RMSNorm over the carried hidden
//!   blk.N.nextn.eh_proj.weight           [hidden, 2·hidden] over their concat
//!   blk.N.nextn.shared_head_norm.weight  final norm before the SHARED lm_head
//! ```
//!
//! It **shares** `token_embd` and `output.weight` with the target, so the head
//! costs one block of weights rather than a second model. (`nextn.embed_tokens`
//! and `nextn.shared_head_head` exist in the format for checkpoints trained
//! with dedicated ones; these are not — `mtp_use_dedicated_embeddings: false` —
//! and the loader refuses rather than guesses if they ever appear.)
//!
//! # The recurrence
//!
//! Given the target's hidden `h` at the position whose argmax produced token
//! `t`, one step is
//!
//! ```text
//!   x      = eh_proj( [ enorm(embed(t)) ; hnorm(h) ] )
//!   h'     = block(x)                       // attention + FFN, both residuals
//!   logits = lm_head( head_norm(h') )
//!   t'     = argmax(logits)
//! ```
//!
//! and the next step feeds `(t', h')` back in. So a `k`-token draft is `k`
//! one-block forwards seeded once from the trunk — which is why the verify wave
//! has to surface the target's hidden per scored row, not just its logits.
//!
//! # The concat order is `[embedding ; hidden]`, and it is not guessable
//!
//! `eh_proj` is a single `[hidden, 2·hidden]` weight spanning both halves, so
//! swapping them multiplies each input by the other's block. The head still
//! returns a finite, plausible token — and speculation is lossless, so nothing
//! downstream ever disagrees. A mis-wired head shows up only as acceptance
//! decaying toward 1.00, which is far too weak a signal to debug
//! from. The order is taken from the reference consumer of these tensor names,
//! llama.cpp `src/models/qwen35moe.cpp`:
//! `ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0)` — ggml's dim 0 is the
//! embedding axis, this file's last axis. [`MtpInput`]'s test pins it.

use candle::cuda_backend::Backing;
use candle::{DType, Device, Result, Tensor};

use super::embedding::EmbeddingTable;
use super::quantized_attention::Qwen35AttentionLayer;
use super::quantized_weights::QuantLayer;
use crate::models::batched_layer::{
    forward_layer_batched_mixed, BatchedAttentionParams, WaveAttnGroup,
};
use crate::models::quantized_matmul::QMatMul;
use crate::models::rotary_layout::RotaryLayout;
use crate::models::tensor_cat::TensorCat;
use crate::quantized_nn::RmsNorm;
use candle_nn::kv_cache::KvCache;

/// The most tokens this head is ever asked to propose in one step.
///
/// Two, and the binding reason is the **expert cache**, not the head. A verify
/// wave carries `d + 1` rows per session where a decode wave carries one, and
/// every extra row routes independently — so draft width is a direct multiplier
/// on the routed-expert union each MoE layer must have resident. On a card where
/// the expert working set is already the scarce thing, a wider block buys
/// speculative reach with expert DMA, which is the trade that loses.
///
/// The head's own acceptance says the same thing from the other side. Measured
/// on the 9B over 256 tokens: 1.96 accepted per step at budget 1, 2.55 at 2,
/// 2.52 at 3, 2.80 at 4 — the first proposal lands ~96% of the time, the second
/// ~59%, and the third essentially never. A one-block NextN head applied
/// recurrently has that reach and no more, so budget 3 pays a full extra draft
/// pass for nothing and measured *slower* end to end (77.5 t/s against 83.6).
pub const MTP_MAX_DRAFT: usize = 2;

// How far ahead it is worth drafting is a separate question from how far the
// head can reach, and it lives in `crate::models::draft_ladder`: reach is a
// property of this block, worth is a property of the wave it rides on.

/// The head's input assembly: the two norms and the projection over their
/// concatenation.
///
/// Split from [`MtpHead`] so the ordering above can be tested without standing
/// up a transformer block — the block is ordinary and covered by the trunk's
/// own gates, while this is the one piece with no other consumer to check it.
pub struct MtpInput {
    /// `[hidden]` RMS gain over the token embedding, before the concat.
    ///
    /// [`RmsNorm`], not a bare gain, for the same reason every trunk norm is
    /// one: the head runs in the wave's activation dtype, and `RmsNorm` is what
    /// re-materialises a gain into that dtype when a session picks it
    /// (`maybe_change_dtype`). A plain tensor would sit at whatever the loader
    /// read and refuse the first BF16 row.
    pub enorm: RmsNorm,
    /// `[hidden]` RMS gain over the carried hidden.
    pub hnorm: RmsNorm,
    /// `[hidden, 2·hidden]` over `[enorm(e) ; hnorm(h)]`.
    pub eh_proj: QMatMul,
}

impl MtpInput {
    /// `eh_proj([enorm(e) ; hnorm(h)])`.
    ///
    /// Both `[n, hidden]`; result `[n, hidden]`. `n` is one per drafting
    /// sequence in a draft step, and the whole wave's row count in the head's
    /// wave pass.
    pub fn forward(&self, embed: &Tensor, hidden: &Tensor) -> Result<Tensor> {
        let (n, d) = embed.dims2()?;
        if hidden.dims2()? != (n, d) {
            candle::bail!(
                "mtp: {n}×{d} embeddings against {:?} hiddens — the head pairs \
                 one embedding with one hidden",
                hidden.dims()
            );
        }
        // Embedding first. See the module note: the halves are not
        // interchangeable and the order is llama.cpp's.
        let e = self.enorm.forward_live(embed)?;
        let h = self.hnorm.forward_live(hidden)?;
        let cat = Tensor::cat(&[&e, &h], 1)?;
        self.eh_proj.forward_live(&cat)
    }
}

// `MtpAttention` was a second spelling of `QuantAttentionWeights` — the same
// weights, the same `[q|gate]` interleave, differing only in holding the Q/K
// gains as plain tensors because the REFERENCE attention took them that way.
// The head now runs the production path, so it carries the production type.

/// Everything the head needs from the model around it. Borrowed per call
/// rather than cloned into the head: the embedding table and the LM head are
/// the **target's**, which is the whole reason a NextN head costs one block.
pub struct MtpContext<'a> {
    /// The target's `[vocab, hidden]` table, shared — never a copy of it.
    pub embed: &'a EmbeddingTable,
    /// The target's LM head, shared.
    pub lm_head: &'a QMatMul,
    /// The rotary reordering the production attention path applies to Q and K —
    /// the model's own, shared, because the head ropes on the same geometry.
    pub rotary: &'a RotaryLayout,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    /// The wave's activation dtype.
    ///
    /// The head runs in it for the same reason every trunk layer does — a
    /// drafted position and the wave position that later replaces it must be
    /// the same arithmetic, or the K/V the draft attended over and the K/V the
    /// verify wrote disagree on a token both of them accepted.
    pub act_dtype: DType,
    pub device: &'a Device,
}

impl MtpContext<'_> {
    /// Gather the embedding rows for ids that live on the **device**.
    ///
    /// The drafting path, and the reason it exists is the residency: under
    /// [`EmbeddingTable::HostMapped`] the GPU reads the ids where they already
    /// are, so an argmax can become the next step's embedding without ever
    /// reaching the host.
    pub fn embed_ids(&self, ids: &Tensor) -> Result<Tensor> {
        self.embed
            .rows(ids, self.device, Backing::Owned, self.act_dtype)
    }
}

/// One NextN draft head: a transformer block that attends, plus the input
/// assembly that feeds it and the norm that closes it.
///
/// # It is a layer of the model, not a sidecar
///
/// The checkpoint says so: the head loads from `blk.{num_layers}` through the
/// same tensor names every trunk block uses, and [`Self::block`] is literally a
/// [`QuantLayer`] — same weights, same `[q|gate]` interleave, same shapes. So it
/// runs through the same production path a trunk attention layer does
/// (`Qwen35AttentionLayer` → `forward_layer_batched_mixed` → the paged batched
/// decode kernel), holds its KV in the same paged cache as an ordinary layer,
/// and prefills, decodes and glues alongside them at the same length over the
/// same token stream.
///
/// That uniformity is the point. The head carrying a private dense KV made it a
/// special case in every session-wide operation — fork, view, prefix injection,
/// turn sealing — each of which assumes a sequence's layers describe one stream
/// at one length. As a layer, it needs none of them to know it exists.
///
/// What it does **not** share is weights: nothing here is inherited from a trunk
/// layer. The only coupling is the input, which is derived from the trunk's
/// OUTPUT rather than from the residual stream — so the head runs as a one-layer
/// pass after the trunk's sweep, over the same rows, in the same wave.
pub struct MtpHead {
    /// The `[enorm ; hnorm] → eh_proj` input assembly.
    pub input: MtpInput,
    /// The head's transformer block: ln1, gated attention, ln2, dense FFN —
    /// the same type, and the same production path, as a trunk attention layer.
    pub block: QuantLayer,
    /// Final norm before the shared LM head — the head's own, not the model's
    /// `output_norm`.
    pub head_norm: RmsNorm,
    /// Trunk block index this head sits at (`cfg.num_layers`), which is also its
    /// tensor prefix in the checkpoint.
    pub layer_index: usize,
}

impl MtpHead {
    /// One draft step: the head's block over **one row per sequence**, through
    /// the production attention path.
    ///
    /// `forward_layer_batched_mixed` over a single decode group — the paged
    /// batched kernel every trunk attention layer runs. One launch for the
    /// cohort, per-sequence KV lengths carried by the slot descriptor table,
    /// RoPE applied in-kernel at each sequence's own position, and the output
    /// gate folded into the combine pass.
    ///
    /// Only attention is per-sequence: a row attends to its own history at its
    /// own length. Every other step — the input assembly, the four projections,
    /// the FFN, the norms — is row-independent and runs as ONE launch over all
    /// the rows. Those steps are weight-bound, so a launch over `R` rows costs
    /// what a launch over one row costs, and drafting the cohort together reads
    /// the block's weights once for the step rather than once per session.
    ///
    /// `caches` are this layer's KV — the head's own layer of the session's
    /// paged cache, one entry per row — and `offsets` each sequence's length in
    /// it. `params` carries slot headers built for that layer **alone** (see
    /// [`build_decode_metadata_at`](crate::models::batched_inference::BatchedInferenceSession::build_decode_metadata_at)),
    /// because a drafting head stands one position ahead of the trunk and a
    /// whole-stack build would read that as divergence — so the head's headers
    /// are the group's only entry, at index 0.
    ///
    /// Returns the **post-`head_norm`** hidden `[n, hidden]`: both what the LM
    /// head scores and what seeds the next step, because llama.cpp takes
    /// `t_h_nextn` after the final norm on the trunk and on this block alike,
    /// so the two ends of the recurrence speak the same normalised space.
    pub fn step(
        &self,
        embed: &Tensor,
        hidden: &Tensor,
        caches: &mut [&mut KvCache],
        offsets: &[usize],
        params: &BatchedAttentionParams<'_>,
        ctx: &MtpContext<'_>,
    ) -> Result<Tensor> {
        let rows = embed.dim(0)?;
        if caches.len() != rows || offsets.len() != rows {
            candle::bail!(
                "mtp step: {rows} rows against {} caches and {} offsets — the head \
                 takes one row per sequence",
                caches.len(),
                offsets.len()
            );
        }
        let x = self.input.forward(embed, hidden)?;
        let hidden_dim = x.dim(1)?;
        // The decode entry wants `[b, 1, hidden]`; `forward_layer_batched_mixed`
        // reshapes it to the kernel's `[rows, 1, hidden]` itself via
        // `decode_layout`.
        let mut xt = TensorCat::from_cat_tensor(x.reshape((1, rows, hidden_dim))?, 0)?;
        let layer = Qwen35AttentionLayer {
            layer: &self.block,
            n_head: ctx.n_head,
            n_kv_head: ctx.n_kv_head,
            head_dim: ctx.head_dim,
            rotary: ctx.rotary,
        };
        let mut groups = [WaveAttnGroup {
            caches,
            offsets,
            params,
            rows,
            decode_layout: true,
        }];
        // Runs BOTH halves — attention + residual, then ln2 + FFN + residual —
        // which is exactly the head's block, because the head's block is a
        // trunk attention layer.
        // Layer index 0: the headers describe the head's KV layer and nothing
        // else, so its stride-indexed slot is the buffer's first.
        forward_layer_batched_mixed(&layer, &mut groups, &mut xt, x.dtype(), 0)?;
        let out = xt.to_tensor().reshape((rows, hidden_dim))?;
        self.head_norm.forward_live(&out)
    }

    // `extend` / `extend_cohort` are gone. They existed to catch the head's
    // private KV up over the tokens the target accepted — a second, after-the-
    // fact pass because the head's history lived outside the engine. The head's
    // KV is a layer of the paged cache now, so it advances inside the wave with
    // every other layer, and a rejected block truncates away with them.
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::{GgmlDType, QTensor};
    use candle::{DType, Device};

    fn qmm(w: &Tensor) -> QMatMul {
        QMatMul::from_qtensor(QTensor::quantize(w, GgmlDType::F32).unwrap()).unwrap()
    }

    fn unit_norm(d: usize, dev: &Device) -> RmsNorm {
        let ones = Tensor::ones(d, DType::F32, dev).unwrap();
        RmsNorm::from_qtensor(QTensor::quantize(&ones, GgmlDType::F32).unwrap(), 1e-6).unwrap()
    }

    /// `[hidden, 2·hidden]` that passes ONE half through and zeroes the other.
    /// `half = 0` keeps the embedding columns, `half = 1` the hidden columns.
    fn half_selector(d: usize, half: usize, dev: &Device) -> Tensor {
        let mut w = vec![0f32; d * 2 * d];
        for row in 0..d {
            w[row * 2 * d + half * d + row] = 1.0;
        }
        Tensor::from_vec(w, (d, 2 * d), dev).unwrap()
    }

    fn row(vals: &[f32], dev: &Device) -> Tensor {
        Tensor::from_vec(vals.to_vec(), (1, vals.len()), dev).unwrap()
    }

    /// **The embedding is the FIRST half of the concat.**
    ///
    /// Proved by which input the output responds to rather than by a
    /// hand-computed norm: with `eh_proj` selecting only the first half, the
    /// result must move when the embedding moves and must NOT move when the
    /// hidden does. Swap the concat and both assertions invert, so this cannot
    /// pass on the wrong order — which the end-to-end acceptance number could,
    /// since a mis-wired head is still lossless.
    #[test]
    fn the_first_half_of_the_concat_is_the_embedding() {
        let dev = Device::Cpu;
        let d = 4usize;
        let e1 = row(&[1.0, 2.0, 3.0, 4.0], &dev);
        let e2 = row(&[4.0, 3.0, 2.0, 1.0], &dev);
        let h1 = row(&[1.0, 0.0, 0.0, 0.0], &dev);
        let h2 = row(&[0.0, 0.0, 0.0, 9.0], &dev);
        let vals = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let first_half = MtpInput {
            enorm: unit_norm(d, &dev),
            hnorm: unit_norm(d, &dev),
            eh_proj: qmm(&half_selector(d, 0, &dev)),
        };
        // Selecting the first half: the hidden is invisible, the embedding is not.
        assert_eq!(
            vals(&first_half.forward(&e1, &h1).unwrap()),
            vals(&first_half.forward(&e1, &h2).unwrap()),
            "the first half of the concat responded to the HIDDEN — the two are \
             the wrong way round"
        );
        assert_ne!(
            vals(&first_half.forward(&e1, &h1).unwrap()),
            vals(&first_half.forward(&e2, &h1).unwrap()),
            "the first half of the concat ignored the EMBEDDING"
        );

        // And the mirror, so the selector itself is not what is being tested.
        let second_half = MtpInput {
            enorm: unit_norm(d, &dev),
            hnorm: unit_norm(d, &dev),
            eh_proj: qmm(&half_selector(d, 1, &dev)),
        };
        assert_eq!(
            vals(&second_half.forward(&e1, &h1).unwrap()),
            vals(&second_half.forward(&e2, &h1).unwrap()),
            "the second half of the concat responded to the EMBEDDING"
        );
        assert_ne!(
            vals(&second_half.forward(&e1, &h1).unwrap()),
            vals(&second_half.forward(&e1, &h2).unwrap()),
            "the second half of the concat ignored the HIDDEN"
        );
    }

    /// **The head is really in the pinned checkpoint, with the geometry the
    /// recurrence assumes.**
    ///
    /// This is the gate that would have caught the original mistake. An earlier
    /// pass concluded the architecture had no MTP head at all, from a *plain*
    /// GGUF conversion that drops the tensors — so the thing worth asserting is
    /// not "the loader can parse a head" but "the checkpoint this repo pins
    /// carries one". A pin moved back to a non-MTP repo fails here.
    #[test]
    #[ignore = "reads the pinned Qwen3.5-9B MTP GGUF (7.5 GB) and needs a GPU. Run with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                qwen35::mtp::tests::the_pinned_checkpoint_carries_a_draft_head \
                -- --ignored --nocapture"]
    fn the_pinned_checkpoint_carries_a_draft_head() -> candle::Result<()> {
        use crate::models::batch_test::test_helpers::hf_get;
        use crate::models::qwen35::quantized_weights::load_quantized_model;
        use candle::quantized::{gguf_file::Content, Int8Mode};
        use std::io::{BufReader, Seek, SeekFrom};

        let spec = crate::models::quantized_qwen35::QWEN35_9B;
        let path = hf_get(spec.0, hf_hub::RepoType::Model, spec.1, spec.2)?;
        let device = Device::new_cuda(0)?;
        let mut reader = BufReader::new(std::fs::File::open(&path)?);
        let content = Content::read(&mut reader)?;
        reader.seek(SeekFrom::Start(0))?;
        let model = load_quantized_model(
            &content,
            &mut reader,
            &device,
            Int8Mode::Off,
            None,
            |_, _| Ok(None),
        )?;

        assert_eq!(
            model.cfg.num_mtp_layers, 1,
            "the pinned 9B declares no MTP block — the pin is on a plain GGUF \
             conversion, which drops the head"
        );
        let head = model
            .mtp
            .as_ref()
            .expect("nextn_predict_layers = 1 but no head was loaded");
        assert_eq!(
            head.layer_index, model.cfg.num_layers,
            "the head sits immediately past the trunk"
        );

        // `eh_proj` spans BOTH halves of the concat: a `[hidden, 2·hidden]`
        // weight is what makes the ordering in this module load-bearing.
        let hidden = model.cfg.hidden_size;
        let w = head.input.eh_proj.weight_dims();
        assert_eq!(
            w,
            vec![hidden, 2 * hidden],
            "eh_proj must map the concatenated [embedding ; hidden] pair back to \
             one hidden width"
        );

        // And it runs: one step's input assembly on real weights, finite.
        let e = Tensor::randn(0f32, 1.0, (1, hidden), &device)?;
        let h = Tensor::randn(0f32, 1.0, (1, hidden), &device)?;
        let x = head.input.forward(&e, &h)?;
        assert_eq!(x.dims2()?, (1, hidden));
        let m = x.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(m.is_finite() && m > 0.0, "eh_proj produced {m}");
        println!(
            "MTP head: block {}, eh_proj {w:?}, |x|max {m:.4}",
            head.layer_index
        );
        Ok(())
    }

    /// The head pairs one embedding with one hidden; a mismatched batch is a
    /// caller bug that would otherwise broadcast into silent nonsense.
    #[test]
    fn mismatched_batches_are_refused() {
        let dev = Device::Cpu;
        let d = 4usize;
        let inp = MtpInput {
            enorm: unit_norm(d, &dev),
            hnorm: unit_norm(d, &dev),
            eh_proj: qmm(&half_selector(d, 0, &dev)),
        };
        let e = Tensor::zeros((2, d), DType::F32, &dev).unwrap();
        let h = Tensor::zeros((1, d), DType::F32, &dev).unwrap();
        let err = inp.forward(&e, &h).unwrap_err().to_string();
        assert!(err.contains("one embedding with one hidden"), "{err}");
    }
}
