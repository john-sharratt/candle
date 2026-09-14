//! Driving the NextN/MTP head through the engine: its wave pass, and the
//! speculative draft loop.
//!
//! [`super::mtp`] is the head itself — what it is made of and what one step of
//! its recurrence computes. This is the other half: where its rows come from,
//! where its KV lives, and how a proposal is proposed and then taken back. The
//! split is the same one the rest of the lineage keeps, between a layer's
//! algebra and the wave that runs it.
//!
//! # The head is a layer, so it prefills, decodes and glues like one
//!
//! Its KV is the LAST layer of the session's paged cache — one past every trunk
//! attention layer, which is exactly why nothing else can reach it:
//! [`KvLayerMap`](crate::models::delta_net::KvLayerMap) only ever yields
//! `0..num_kv_layers()`, so the sweep cannot name it, and the wave driver's
//! per-decode-row `set_current_seq_len` walks *every* cache a sequence has, so
//! it advances with the trunk's without being told about.
//!
//! [`head_wave_pass`] is what fills it: one attention pass over the same rows
//! the trunk just swept, at the same positions, in the same wave. That
//! uniformity is the whole design. A layer that stood at a different length
//! from its siblings would have needed a range parameter on every session-wide
//! operation that assumes "a sequence's layers describe one stream at one
//! length" — fork, view, prefix injection, turn sealing, truncation. At the
//! same length it needs none of them to know it exists.
//!
//! # Its input is the trunk's output, not the residual stream
//!
//! The head at position `t` reads `eh_proj([enorm(embed(t)) ; hnorm(h(t-1))])`,
//! where `h` is the trunk's post-`final_norm` hidden. So it is not step 33 of
//! the sweep — it is a one-layer pass *after* it, over the same rows, and its
//! hidden input is the wave's own output shifted one position right.
//!
//! Position 0 of a sequence has no `h(t-1)` and takes zeros. That costs
//! nothing real: a sequence always begins with a prefill, so position 0 is
//! inside the prompt and is never a draft seed — the zeros are there to keep
//! head row `i` aligned with trunk row `i` so RoPE agrees, nothing more.
//!
//! # A draft's positions are real, and then they are not
//!
//! [`draft_cohort`] appends one position per sequence per step to the head's
//! layer, exactly as a decode row does, and truncates the whole run away before
//! it returns. What the target then accepts is written again — properly, by the
//! next wave — so a rejected proposal leaves no trace and an accepted one is
//! not carried over from the draft.

use candle::quantized::pinned_staging::{Generation, GpuBuf};
use candle::{DType, Device, Result, Tensor};

use crate::models::draft_walk::{draft_reserve, draft_rope_depth, draft_walk};
use candle_nn::kv_cache::{begin_wave, KvCache, LayerPhase};

use std::cell::RefCell;

use super::batched::HybridBatched;
use super::mtp::{MtpContext, MtpHead};
use super::quantized_attention::Qwen35AttentionLayer;
use crate::models::batched_inference::BatchedInferenceSession;
use crate::models::batched_layer::{
    forward_attn_batched, BatchedAttentionParams, DecodeHeaders, WaveAttnGroup,
};
use crate::models::delta_net::SeqSpan;
use crate::models::kv_cache_utils::SequenceContext;
use crate::models::lora::LayerLora;
use crate::models::operand_guard::expect_dtype;
use crate::models::prefill_utils::SharedPm;
use crate::models::tensor_cat::TensorCat;
use crate::models::wave_buffers::wave_root;

/// Everything the head's wave pass needs from the sweep that just ran.
///
/// Borrowed rather than rebuilt: the groups, offsets and attention parameters
/// are the wave's own, and the head reuses them unchanged — the same rows at
/// the same positions, only against a different KV layer.
pub struct HeadWave<'a> {
    /// Decode rows, which lead the packed buffer.
    pub n_decode: usize,
    /// Prefill rows, which follow them.
    pub pre_rows: usize,
    /// Each decode row's sequence position.
    pub dec_off: &'a [usize],
    /// Each prefill span's starting sequence position.
    pub pre_off: &'a [usize],
    pub dec_params: &'a BatchedAttentionParams<'a>,
    pub pre_params: &'a BatchedAttentionParams<'a>,
    /// Where each sequence's rows sit in the packed buffer.
    pub spans: &'a [SeqSpan],
    /// The head's KV layer, which is also its slot-header index — the wave's
    /// metadata covers every layer of the session, this one included.
    pub kv_layer: usize,
    /// The wave's activation dtype.
    pub act_dtype: DType,
}

/// Run the head over this wave's rows, filling its KV layer, and hand back each
/// sequence's post-`final_norm` hiddens for the accept to seed from.
///
/// **Attention only.** All this pass owes is the head's K/V, and both are
/// projections of `attn_norm(x)` — the block's FFN changes only the block's
/// *output*, which nothing reads until a draft asks for logits. Running it
/// would be a second full read of the head's FFN weights per wave for a value
/// that is discarded, so the o_proj'd context is dropped where it lands.
///
/// The captured hiddens go to the sequences the caller armed
/// ([`HybridBatched::arm_hidden_capture`]), which is every sequence that could
/// draft next step. They are read back at the accept, not here: the seed is the
/// hidden at the last ACCEPTED position, and which position that is is not
/// known until the target's argmaxes have been compared.
pub fn head_wave_pass(
    model: &HybridBatched,
    head: &MtpHead,
    contexts: &mut [SequenceContext<'_>],
    x_flat: &Tensor,
    ids: &Tensor,
    w: &HeadWave<'_>,
) -> Result<()> {
    let rows = w.n_decode + w.pre_rows;
    if rows == 0 {
        return Ok(());
    }
    let q = model.model();
    let hidden = q.cfg.hidden_size;
    let dev = &q.device;
    let capture = model.hidden_capture_seqs()?;

    // The trunk's output for EVERY row, not just the scored ones: the head's
    // row `i` is conditioned on the hidden of row `i - 1`, so a wave that
    // normed only the rows the LM head scores would have nothing to feed the
    // interior of a prefill span.
    //
    // Off the pool rather than a wave span, like the residual stream it is
    // computed from: what consumes it is `forward_attn_batched`, whose input is
    // a [`TensorCat`], and a `TensorCat` holds an owned tensor. The head's own
    // transients — its projections, its context — do run on the span the pass
    // opens below.
    let h_all = q.final_norm.forward_live(x_flat)?;

    // Hand the armed sequences their rows before the shift consumes `h_all`.
    for span in w.spans {
        if !capture.contains(&span.seq) {
            continue;
        }
        let Some(dst) = model.hidden_buffer(span.seq)? else {
            continue;
        };
        // Sized by `arm_hidden_capture` from the same block length this span
        // carries, so a short buffer is a caller bug — and a silent clamp would
        // seed the next draft from a stale row rather than the accepted one.
        if dst.dim(0)? < span.len {
            candle::bail!(
                "qwen35 mtp: sequence {} captures {} rows into a {}-row buffer",
                span.seq,
                span.len,
                dst.dim(0)?
            )
        }
        // Validated, not converted: `arm_hidden_capture` allocates the buffer in
        // the wave's activation dtype, which is what `h_all` already is.
        expect_dtype(&dst, h_all.dtype(), "mtp capture buffer")?;
        let src = h_all.narrow(0, span.start, span.len)?;
        dst.narrow(0, 0, span.len)?.slice_set(&src, 0, 0)?;
    }

    // The shift. Row `start` of a span takes the seed the last accept left —
    // the trunk's hidden at the position before this wave's first — and rows
    // `start+1 ..` take the row before them, which this wave just produced.
    //
    // **This cat is one launch per span** (`cat0` allocates once and issues a
    // `copy_strided_src` per argument), so a decode wave over `n` sessions
    // spends `n` launches assembling a buffer — hot-path invariant 2, and the
    // one place this pass pays it. Removing it needs the seeds to already be
    // contiguous, which means a slab indexed by slot rather than a tensor per
    // sequence, or a gather kernel over a descriptor table (invariant 2b).
    // Neither is worth doing from this comment: the cost is bounded by the
    // cohort width, not the token count, and it has never been measured against
    // a wide wave. Instrument the span first — the last two refactors proposed
    // here from structural reasoning alone were both aimed at the wrong thing.
    let mut parts: Vec<Tensor> = Vec::with_capacity(w.spans.len() * 2);
    let mut zero: Option<Tensor> = None;
    for span in w.spans {
        let seed = match model.draft_seed(span.seq)? {
            // Already the wave's dtype — it is a row of a capture buffer that
            // was allocated in it.
            Some(s) => s,
            // No seed yet: the sequence's first wave, whose first row is
            // position 0 and has no predecessor. A fork inherits its parent's
            // KV but not its seed and lands here too — one row of the head's
            // history is then conditioned on zeros instead of the true hidden,
            // which costs a little draft quality on that step and nothing else,
            // because a proposal is only ever checked against the target.
            None => zero
                .get_or_insert(Tensor::zeros((1, hidden), w.act_dtype, dev)?)
                .clone(),
        };
        parts.push(seed);
        if span.len > 1 {
            parts.push(h_all.narrow(0, span.start, span.len - 1)?);
        }
    }
    let shifted = Tensor::cat(&parts, 0)?;
    drop(parts);

    let wave = match dev {
        Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Attention)?),
        _ => None,
    };
    let embed = q
        .embed
        .rows(ids, dev, wave_root(wave.as_ref()), w.act_dtype)?;
    let x = head.input.forward(&embed, &shifted)?;
    let xt = TensorCat::from_cat_tensor(x.reshape((1, rows, hidden))?, 0)?;

    let layer = Qwen35AttentionLayer {
        layer: &head.block,
        n_head: q.cfg.num_attention_heads,
        n_kv_head: q.cfg.num_kv_heads,
        head_dim: q.cfg.attn_head_dim,
        rotary: model.rotary(),
        // The drafter proposes; the adapted trunk verifies. See the same field
        // in `mtp.rs` for why leaving the head unadapted costs acceptance rate
        // and never correctness.
        lora: LayerLora::default(),
    };
    let mut cache_refs: Vec<&mut KvCache> = contexts
        .iter_mut()
        .map(|c| &mut c.kv_caches.caches[w.kv_layer])
        .collect();
    let (dec_c, pre_c) = cache_refs.split_at_mut(w.n_decode);
    let mut groups: Vec<WaveAttnGroup> = Vec::with_capacity(2);
    if w.n_decode > 0 {
        groups.push(WaveAttnGroup {
            caches: dec_c,
            offsets: w.dec_off,
            params: w.dec_params,
            rows: w.n_decode,
            decode_layout: true,
            // The drafter's attention is dense — no indexer on this lineage.
            qsa: None,
        });
    }
    if w.pre_rows > 0 {
        groups.push(WaveAttnGroup {
            caches: pre_c,
            offsets: w.pre_off,
            params: w.pre_params,
            rows: w.pre_rows,
            decode_layout: false,
            qsa: None,
        });
    }
    let mut row0 = 0usize;
    for g in groups.iter_mut() {
        let slice = xt.as_cat_tensor().narrow(1, row0, g.rows)?;
        let x_g = if g.decode_layout {
            TensorCat::from_cat_tensor(slice.reshape((g.rows, 1, hidden))?.contiguous()?, 0)?
        } else {
            TensorCat::from_cat_tensor(slice.contiguous()?, 0)?
        };
        // Dropped where it lands: see the note above on why the FFN half of the
        // block does not run here.
        forward_attn_batched(
            &layer,
            g.caches,
            &x_g,
            g.offsets,
            g.params,
            w.kv_layer,
            g.qsa,
            wave.as_ref(),
        )?;
        row0 += g.rows;
    }
    Ok(())
}

/// Propose up to `max_len` tokens per sequence, seeded by the target's hidden
/// at the position that produced each sequence's committed token.
///
/// **The cohort drafts together.** Step `j` of one sequence needs
/// `embed(argmax(...))` of its own step `j-1` — a serial dependency no batching
/// removes — but it does not depend on any *other* sequence's step `j`. So the
/// loop runs over draft depth, not over sequences, and each iteration is one
/// batched pass over every drafting session at once.
///
/// That is worth doing because the drafter is weight-bound, not
/// arithmetic-bound. On the 9B a drafted token costs ~2.7 ms, of which ~1.5 ms
/// is one full read of the ~795 MiB output projection to score a single row —
/// and four rows measured 1.69 ms against one row's 1.66. Per sequence, an
/// `n`-session step reads that weight `n` times; this way it reads it once.
///
/// **The loop never synchronises.** The argmax stays a device tensor and
/// [`EmbeddingTable::HostMapped`](super::embedding::EmbeddingTable) gathers its
/// row from where it is, so nothing drains the pipeline mid-block. The ids come
/// back **once**, at the end, because the caller needs them as tokens — one
/// readback per step for the whole cohort, rather than one per drafted token
/// per session.
///
/// **The head's layer runs ahead of the trunk's, and only here.** Each step
/// leaves it one position longer, which is precisely what a speculative
/// position is — so its slot headers are built for that layer ALONE. A
/// whole-stack build would compare it against the trunk's length, read the
/// difference as corruption, and "heal" it by truncating the proposal away.
pub fn draft_cohort(
    model: &HybridBatched,
    session: &mut BatchedInferenceSession,
    seqs: &[usize],
    committed: &[u32],
    seeds: &[Tensor],
    max_len: usize,
) -> Result<Vec<Vec<u32>>> {
    let n = seqs.len();
    if committed.len() != n || seeds.len() != n {
        candle::bail!(
            "qwen35 mtp draft: {n} sequences, {} committed tokens, {} seeds",
            committed.len(),
            seeds.len()
        );
    }
    let q = model.model();
    let Some(head) = q.mtp.as_ref() else {
        return Ok(vec![Vec::new(); n]);
    };
    let Some(kv_layer) = model.mtp_kv_layer() else {
        return Ok(vec![Vec::new(); n]);
    };
    if n == 0 || max_len == 0 {
        return Ok(vec![Vec::new(); n]);
    }

    let dev = &q.device;
    let act_dtype = session.activation_dtype();
    // Allocate every drafted position's write chunk before the walk begins —
    // `draft_walk`'s module docs carry why that is load-bearing rather than
    // tidy, and the walk does it itself. It is forced here because `max_blocks`
    // below must be read after it.
    draft_reserve(session, seqs, kv_layer, max_len)?;

    // The rope table spans the arena's whole addressable context, so it is read
    // from the head's own layer rather than assumed — and only after the
    // reserve above, which can grow it.
    let max_blocks = draft_rope_depth(session, seqs, kv_layer)?;
    let rope_cs = model.rope_cs(max_blocks)?;
    let inv_freq = model.inv_freq_device().clone();
    let theta = q.cfg.rope_theta;
    let rope_dtype = if act_dtype == DType::F8E4M3 {
        DType::BF16
    } else {
        act_dtype
    };
    let ctx = MtpContext {
        embed: &q.embed,
        lm_head: &q.lm_head,
        rotary: model.rotary(),
        n_head: q.cfg.num_attention_heads,
        n_kv_head: q.cfg.num_kv_heads,
        head_dim: q.cfg.attn_head_dim,
        act_dtype,
        device: dev,
    };

    // Validated, not converted. The seeds are rows of capture buffers that were
    // allocated in this dtype, so a mismatch is a producer bug and a cast here
    // would hide it — and hide it as draft quality, not as an error, because
    // verify keeps only the target's argmaxes whatever the head proposes.
    for s in seeds {
        expect_dtype(s, act_dtype, "mtp draft seed")?;
    }
    let seed_refs: Vec<&Tensor> = seeds.iter().collect();
    let seed_block = Tensor::cat(&seed_refs, 0)?;
    let q_lens = vec![1usize; n];

    // One position of the walk. Everything around it — the pre-ensure, the
    // per-step metadata, the cache advance, the rollback, the single readback —
    // is [`draft_walk`], which is the same for every drafter. What is the
    // model's own is the rope tables and the head's block, and that is all this
    // closure holds.
    let mut step = |ids: &Tensor,
                    h: &Tensor,
                    caches: &mut [&mut KvCache],
                    at: &[usize],
                    headers: (&GpuBuf, u64),
                    generation: &Generation|
     -> Result<(Tensor, Tensor)> {
        let pos: Vec<u32> = at.iter().map(|&p| p as u32).collect();
        let (cos, sin) = model.rotary().rope_cos_sin(&pos, theta, rope_dtype, dev)?;
        let pm: RefCell<Option<SharedPm>> = RefCell::new(None);
        let params = BatchedAttentionParams::new(
            &cos,
            &sin,
            false,
            &inv_freq,
            &rope_cs,
            DecodeHeaders::Decode {
                buf: Some(headers.0.clone()),
                stride: headers.1,
            },
            &q_lens,
            generation,
            &pm,
        );
        let embed = ctx.embed_ids(ids)?;
        let h_next = head.step(&embed, h, caches, at, &params, &ctx)?;
        let logits = ctx.lm_head.forward_live(&h_next)?;
        Ok((h_next, logits))
    };
    draft_walk(
        session,
        seqs,
        kv_layer,
        committed,
        &seed_block,
        max_len,
        // Nothing to open: this head's block runs through
        // `forward_layer_batched_mixed`, which lays its spans out in whatever
        // tier is already placed rather than wanting one of its own.
        || Ok(()),
        &mut step,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};

    /// Every layer's live token count for a sequence, the head's included.
    fn layer_lengths(session: &BatchedInferenceSession, seq: usize) -> Vec<usize> {
        session
            .sequence_caches(seq)
            .expect("live slot")
            .caches
            .iter()
            .map(|c| {
                let mut n = 0usize;
                c.k_cache().chunked_visit_live_chunks(|it| {
                    for ch in it {
                        n += ch.token_count as usize;
                    }
                });
                n
            })
            .collect()
    }

    /// **The head's layer keeps the trunk's length, and a draft leaves it
    /// there.**
    ///
    /// Two invariants in one run, and they are the two the whole design rests
    /// on. After the wave, the head's KV must hold exactly what every trunk
    /// attention layer holds — that uniformity is what lets fork, seal, view
    /// and truncate stay ignorant of it. After drafting, it must hold that
    /// again: the proposal's positions are appended like decode rows and rolled
    /// back before the verify wave writes them properly, and a leak here is not
    /// visible as a wrong answer. It is visible as a length skew that the next
    /// wave's `unify_decode_layout` silently "heals" by truncating — which is a
    /// dropped token, discovered much later and somewhere else.
    #[test]
    #[ignore = "reads the pinned Qwen3.5-9B MTP GGUF (7.5 GB) and needs a GPU. Run with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                qwen35::draft::tests::a_draft_leaves_the_head_at_the_trunk_s_length \
                -- --ignored --nocapture"]
    fn a_draft_leaves_the_head_at_the_trunk_s_length() -> Result<()> {
        use super::super::quantized_loader::Qwen35LoadOptions;
        use crate::models::batch_test::test_helpers::hf_get;
        use crate::models::quantized_qwen35::from_gguf_path;
        use candle::quantized::Int8Mode;

        let spec = crate::models::quantized_qwen35::QWEN35_9B;
        let path = hf_get(spec.0, hf_hub::RepoType::Model, spec.1, spec.2)?;
        let device = Device::new_cuda(0)?;
        let model = from_gguf_path(
            &path,
            &device,
            Qwen35LoadOptions {
                int8mode: Some(Int8Mode::Off),
                expert_pack_dir: None,
                mtp_path: None,
                gate_donor_path: None,
                tensor_overrides: Vec::new(),
            },
        )?;
        assert!(model.has_drafter(), "the pinned 9B carries an MTP head");
        let head_kv = model.mtp_kv_layer().expect("a head means a head KV layer");

        let mut session =
            model.create_batched_session(BatchedConfig::default().with_dtype(DType::BF16))?;
        let seq = session.create_sequence()?;
        let prompt: Vec<u32> = (0..24u32).map(|i| 1000 + i).collect();
        let n = prompt.len();
        let ids = Tensor::from_vec(prompt.clone(), (1, n), &device)?;
        model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            &[ids],
            &[],
            &[],
            0,
            model.num_layers(),
            None,
        )?;
        session.advance_sequence(seq, n)?;

        let after_wave = layer_lengths(&session, seq);
        assert_eq!(
            after_wave.len(),
            head_kv + 1,
            "the session allocates one KV layer per attention layer plus the head's"
        );
        assert!(
            after_wave.iter().all(|&l| l == n),
            "the head's layer must prefill alongside the trunk's: {after_wave:?} for a \
             {n}-token prompt"
        );

        // Seed the sequence the way an accept does, then draft off it.
        model.arm_hidden_capture(&[(seq, 1)], session.activation_dtype())?;
        model.mtp_take_seeds(&[(seq, None)])?;
        model.disarm_hidden_capture();
        let drafts = draft_cohort(
            &model,
            &mut session,
            &[seq],
            &[prompt[n - 1]],
            &[model.draft_seed(seq)?.expect("just seeded")],
            2,
        )?;

        let after_draft = layer_lengths(&session, seq);
        assert_eq!(
            after_draft, after_wave,
            "drafting left the head's layer at a different length from the trunk's — \
             the speculative positions were not rolled back"
        );
        let proposed = &drafts[0];
        assert_eq!(proposed.len(), 2);
        assert!(
            proposed
                .iter()
                .all(|&t| (t as usize) < model.model().cfg.vocab_size),
            "a draft is not a token id: {proposed:?}"
        );
        println!("drafts {proposed:?}, layer lengths {after_draft:?}");
        Ok(())
    }
}
