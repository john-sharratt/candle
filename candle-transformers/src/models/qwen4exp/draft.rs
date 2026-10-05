//! Running the draft head: the per-wave pass that keeps its KV in lockstep
//! with the trunk's.
//!
//! [`super::mtp`] is the head itself — its input assembly and its block. This
//! is how the engine runs it inside a wave.
//!
//! # Why the head runs on every wave, not only when drafting
//!
//! The head is a full-attention block with K/V of its own, and a proposal at
//! position `p` attends over the head's keys for every position before `p`.
//! Those keys exist only if the head ran at each of those positions. So the
//! head's layer has to advance with the trunk's whether or not anything drafts
//! this step — a head that ran only while speculating would attend over a
//! history full of holes and propose from a model that never existed.
//!
//! Running it in the same wave, over the same rows, at the same positions, is
//! also what keeps it invisible. Its layer stands at the same length as every
//! trunk attention layer, so it is a **stream** layer in the session's sense
//! ([`crate::models::batched_inference::KvLayers`]) and every session-wide
//! operation that assumes "a sequence's layers describe one stream at one
//! length" — fork, view, prefix injection, turn sealing, truncation — keeps
//! working without being told the head exists.
//!
//! # Attention only
//!
//! All this pass owes is the head's K/V, and both K and V are projections of
//! the block's attention pre-mix. The block's MoE changes only the block's
//! *output*, which nothing reads until a draft asks for logits — so running it
//! here would be a second full pass over a 512-expert layer per wave for a
//! value that is discarded. The o_proj'd context is dropped where it lands.
//!
//! # The input is the trunk's output, shifted one row right
//!
//! The head at position `t` reads `eh_proj([enorm(embed(t)) ; hnorm(h(t-1))])`
//! — its own token's embedding against the trunk's carried state from the row
//! *before* it. So it is not another layer of the sweep; it is a one-block pass
//! after it, whose hidden input is the wave's own output shifted right by one.
//!
//! Row 0 of a sequence has no `h(t-1)`. Within a wave the shift is internal,
//! but the first row of each span reaches back to the previous wave, so the
//! last row's residual is carried across as that sequence's **seed**
//! ([`SeedStore`]). At the very start of a sequence there is no previous wave
//! either, and the seed is zeros — which costs nothing real, because a sequence
//! always begins with a prefill and position 0 is inside the prompt, never a
//! draft seed. The zeros are there so head row `i` stays aligned with trunk row
//! `i` and RoPE agrees, nothing more.

use std::cell::RefCell;
use std::collections::HashMap;

use candle::quantized::pinned_staging::{Generation, GpuBuf};
use candle::{DType, Result, Tensor};

use super::batched_attention::Qwen4ExpAttentionLayer;
use super::capture_rows::CaptureRows;
use super::engine::GpuLayerMix;
use super::hyper::{hc_combine, hc_combine_gated, hc_mix};
use super::indexer::{IndexCache, IndexSnapshot};
use super::mtp::MtpHead;
use super::spec::SpecCapture;
use super::wave::Qwen4ExpBatched;
use crate::models::batched_inference::BatchedInferenceSession;
use crate::models::batched_inference::ManagedBatchedModel;
use crate::models::batched_layer::{forward_attn_batched, BatchedAttentionParams, DecodeHeaders};
use crate::models::delta_net::SeqSpan;
use crate::models::draft_walk::{draft_reserve, draft_walk};
use crate::models::kv_cache_utils::SequenceContext;
use crate::models::latent_moe::scatter::{rows_scatter, RowRun};
use crate::models::lazy_rope::LazyRope;
use crate::models::prefill_utils::SharedPm;
use crate::models::profile::gpu_span;
use crate::models::rope_schedule::FactoredRope;
use crate::models::tensor_cat::TensorCat;
use crate::models::wave_buffers::wave_empty_ticketed;
use candle::quantized::cuda::to_dynamic;
use candle_nn::kv_cache::{
    begin_wave, cover_wave_transient, KvCache, LayerPhase, WavePlan, WaveWidth,
};

/// Each sequence's last wide residual, carried between waves so the head's
/// first row has the `h(t-1)` the wave before it produced.
///
/// Keyed by slot id, and slot ids are **recycled**: a stale entry inherited by
/// the next conversation to land on the id would seed the head from a history
/// this sequence never had. Cleared through the engine's `release_sequence`
/// alongside the recurrent and index state, and reset whenever a sequence
/// starts over at offset 0.
///
/// The model holds two of these: the seeds, and each sequence's **spare** — the
/// buffer the next carry writes before it becomes the seed (see
/// [`Qwen4ExpBatched::head_wave_pass`]). A spare must never be a buffer the seed
/// store also holds, so every path that puts a seed back or takes one away
/// drops the sequence's spare with it.
pub type SeedStore = HashMap<usize, Tensor>;

/// Everything the head pass needs that the wave already computed — the same
/// rows at the same positions, only against a different KV layer.
pub struct HeadWave<'a> {
    /// Decode rows, which lead the packed buffer.
    pub n_decode: usize,
    /// Prefill rows, which follow them.
    pub pre_rows: usize,
    pub dec_off: &'a [usize],
    pub pre_off: &'a [usize],
    pub dec_params: &'a BatchedAttentionParams<'a>,
    pub pre_params: &'a BatchedAttentionParams<'a>,
    /// Where each sequence's rows sit in the packed buffer.
    pub spans: &'a [SeqSpan],
    /// Every row's sequence position, decode rows then prefill rows.
    pub offsets_all: &'a [usize],
    /// The indexer's factored RoPE table.
    pub index_rope: &'a FactoredRope,
    /// The head's KV layer, which is also its index into a sequence's index
    /// caches — past every trunk attention layer.
    pub kv_layer: usize,
    /// The wave's pinned staging generation, which the shift's and the
    /// carry's row-copy descriptors are staged in.
    pub generation: &'a Generation,
}

impl Qwen4ExpBatched {
    /// Run the head over this wave's rows, filling its KV layer, and carry each
    /// sequence's last residual forward as the next wave's seed.
    ///
    /// `res` is the trunk's wide residual after the sweep — `[total_rows, hc,
    /// n_embd]` — and `embeds` this wave's own token embeddings,
    /// `[total_rows, n_embd]`, the same rows the trunk entered on.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn head_wave_pass(
        &self,
        head: &MtpHead,
        contexts: &mut [SequenceContext],
        idx_map: &mut HashMap<usize, Vec<IndexCache>>,
        mut capture: Option<&mut SpecCapture>,
        seeds: &mut SeedStore,
        spares: &mut SeedStore,
        res: &Tensor,
        embeds: &Tensor,
        w: &HeadWave<'_>,
        eps: f64,
    ) -> Result<()> {
        let cfg = &self.model.cfg;
        let dev = &self.model.device;
        let (total_rows, hc, n_embd) = res.dims3()?;
        let g = gpu_span("q4e:mtp_head", dev);

        // **The head's own phase.** It runs one layer's worth of work after the
        // trunk's loop has closed its phases, so it opens the attention phase
        // for itself, and every transient below is rooted on it: the shift, the
        // input assembly, the pre-mix, the selection tables and the projections.
        //
        // Safe because what outlives the pass is written outside the arena: the
        // seeds are each sequence's own buffers (see the carry below) and the
        // verify's head rows are owned copies. Everything else dies with the
        // pass.
        let head_wave = match dev {
            candle::Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Attention)?),
            _ => None,
        };
        let head_ticket = head_wave.as_ref().map(|g| g.ticket());

        // ── The shift, as one scatter onto the head's phase. ──
        //
        // Row `i` of the head reads the trunk residual of row `i-1`, and the
        // first row of each span reaches back to the previous wave's seed. Each
        // span is therefore two runs — its seed into its first row, and its own
        // rows but the last into the rows after — and every span's runs go in
        // one launch. The rows are written where the head reads them, with no
        // `[seeds ; res]` block assembled first to gather from.
        //
        // A sequence the head has never run behind is at position 0, where the
        // reference takes zeros — which is what admission made its seed.
        let width = hc * n_embd;
        let shifted = wave_empty_ticketed((total_rows, hc, n_embd), DType::F32, dev, head_ticket)?;
        let shifted_rows = shifted.reshape((total_rows, width))?;
        let res_rows = res.reshape((total_rows, width))?;
        let mut shift_runs: Vec<RowRun<'_>> = Vec::with_capacity(2 * w.spans.len());
        for span in w.spans {
            let seed = seeds.get(&span.seq).cloned().ok_or_else(|| {
                candle::Error::Msg(format!(
                    "qwen4exp head: sequence {} entered the forward with no seed — \
                     admission makes it, so the forward never allocates one",
                    span.seq
                ))
            })?;
            shift_runs.push(RowRun::new(
                seed.reshape((1, width))?,
                &shifted_rows,
                span.start,
            ));
            if span.len > 1 {
                shift_runs.push(RowRun::new(
                    res_rows.narrow(0, span.start, span.len - 1)?,
                    &shifted_rows,
                    span.start + 1,
                ));
            }
        }
        rows_scatter(&shift_runs, w.generation)?;
        drop(shift_runs);

        // ── The head's block input, entering the hyper-connection stream the
        // way a token embedding does. ──
        let x_head = head.block_input(embeds, &shifted, eps, head_ticket)?;

        let GpuLayerMix::Attention {
            w: aw,
            indexer,
            compress_ratio,
        } = &head.block.mix
        else {
            candle::bail!(
                "qwen4exp draft head: block {} is not full-attention, but the head is \
                 declared `full_attention` and needs a KV layer",
                head.layer_index
            );
        };

        let (h, _) = hc_mix(&x_head, &head.block.hc_attn, eps, head_ticket)?;

        // QSA over the head's OWN cache. The head selects exactly as a trunk
        // attention layer does — same indexer weights, same budget — because a
        // proposal has to be scored against the context the verify will score
        // it against.
        let qsa = self.layer_selection(
            w.kv_layer,
            *compress_ratio,
            indexer,
            w.index_rope,
            &h,
            w.spans,
            w.offsets_all,
            idx_map,
            // The head's cache advances over a verify block exactly as the
            // trunk's do — it runs the same rows in the same wave — so it has
            // to be captured, or a partial accept would rewind twelve caches
            // and leave the thirteenth holding the rejected tokens' keys.
            capture.as_deref_mut(),
            total_rows,
            // The head's own phase: its tables are read by this pass's
            // attention and nothing after it.
            head_ticket,
        )?;
        let dec_sel = match &qsa {
            Some(s) if w.n_decode > 0 => Some(s.rows_slice(0, w.n_decode)?),
            _ => None,
        };
        let pre_sel = match &qsa {
            Some(s) if w.pre_rows > 0 => Some(s.rows_slice(w.n_decode, w.pre_rows)?),
            _ => None,
        };

        let alayer = Qwen4ExpAttentionLayer {
            w: aw,
            n_head: cfg.num_attention_heads,
            n_kv_head: cfg.num_kv_heads,
            head_dim: cfg.attn_head_dim,
            rotary: &self.model.rotary,
        };
        let mut cache_refs: Vec<&mut KvCache> = contexts
            .iter_mut()
            .map(|c| &mut c.kv_caches.caches[w.kv_layer])
            .collect();
        let (dec_c, pre_c) = cache_refs.split_at_mut(w.n_decode);
        // The output is dropped on both arms: this pass owes the K/V it just
        // wrote and nothing else (see the module docs).
        if w.n_decode > 0 {
            let x_g = TensorCat::from_cat_tensor(
                h.narrow(0, 0, w.n_decode)?
                    .reshape((w.n_decode, 1, n_embd))?
                    .contiguous()?,
                0,
            )?;
            forward_attn_batched(
                &alayer,
                dec_c,
                &x_g,
                w.dec_off,
                w.dec_params,
                w.kv_layer,
                dec_sel.as_ref(),
                head_wave.as_ref(),
            )?;
        }
        if w.pre_rows > 0 {
            let x_g = TensorCat::from_cat_tensor(
                h.narrow(0, w.n_decode, w.pre_rows)?
                    .reshape((1, w.pre_rows, n_embd))?
                    .contiguous()?,
                0,
            )?;
            forward_attn_batched(
                &alayer,
                pre_c,
                &x_g,
                w.pre_off,
                w.pre_params,
                w.kv_layer,
                pre_sel.as_ref(),
                head_wave.as_ref(),
            )?;
        }

        // ── Carry each sequence's rows forward. ──
        //
        // **Into the sequence's spare seed buffer, which then becomes its seed.**
        // Each sequence holds two `[1, hc, n_embd]` buffers for its whole life
        // and the carry alternates between them: the seed this wave read stays
        // exactly as it entered, which is what the failure bracket restores and
        // what a verify's rewind starts from, while the next wave's seed is
        // written beside it. Every sequence's row goes in one launch, and no
        // seed is allocated after the second wave a sequence runs in — where a
        // fresh owned copy per sequence per wave used to be made.
        //
        // **One row, and the block's rows only where a rewind can reach them.**
        // A verify wave's accept walk may keep just a prefix of its block, so
        // the position the next draft follows is `kept − 1` rather than the
        // span's last row — but that is only ever true of a *verifying*
        // sequence, and only its block is short. Keeping every span's rows here
        // instead would copy `[len, hc_dim]` per sequence per wave, and a
        // prefill span is the whole prompt chunk: a new allocate-plus-copy
        // scaling with rows, on the wave path, which is exactly what hot-path
        // invariant 2 forbids. It measured ~5% of wide-rung decode. So the
        // block rows go to the verify capture, which exists only for the
        // sequences that can be rewound, and `rewind_cohort` reseeds from them.
        //
        // The capture copies the block's rows into the cohort's kept-row
        // buffer at the sequence's stash row — buffers sized once for the
        // cohort and reused across steps, so keeping them allocates nothing.
        let mut carry_runs: Vec<RowRun<'_>> = Vec::with_capacity(w.spans.len());
        let mut flips: Vec<(usize, Tensor)> = Vec::with_capacity(w.spans.len());
        for span in w.spans {
            if let Some(cap) = capture.as_deref_mut() {
                let SpecCapture { seqs, rows, .. } = cap;
                if let Some(stash) = seqs.get_mut(&span.seq) {
                    stash.head_rows = Some(CaptureRows::keep(
                        &rows.head,
                        stash.row,
                        &res.narrow(0, span.start, span.len)?,
                    )?);
                }
            }
            let spare = spares.remove(&span.seq).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "qwen4exp head: sequence {} entered the forward with no spare seed — \
                     admission makes it, so the forward never allocates one",
                    span.seq
                ))
            })?;
            carry_runs.push(RowRun::new(
                res_rows.narrow(0, span.start + span.len - 1, 1)?,
                &spare.reshape((1, width))?,
                0,
            ));
            flips.push((span.seq, spare));
        }
        rows_scatter(&carry_runs, w.generation)?;
        drop(carry_runs);
        for (seq, written) in flips {
            if let Some(read) = seeds.insert(seq, written) {
                spares.insert(seq, read);
            }
        }
        g.end();
        Ok(())
    }

    /// One drafted position for the whole cohort: the head's block over one row
    /// per sequence, writing its K/V at `at`.
    ///
    /// Returns `(wide residual out, logits)` — the residual because it is what
    /// the next drafted position reads as its `h(t-1)`, and the logits because
    /// the walk takes their argmax as the proposal. The residual is written into
    /// `out`, one of the walk's two carry buffers (see [`Self::draft_carry`]);
    /// `prev_wide` is the other.
    ///
    /// **The whole block, not attention only.** The lockstep pass
    /// ([`Self::head_wave_pass`]) skips the MoE because it owes only K/V; a
    /// proposal is the block's *output*, so here the FFN half runs. Both halves
    /// go through the same `hc_mix`/`hc_combine` pair every trunk layer uses.
    ///
    /// **Every transient is on the tier's three spans**, which `mtp_draft`
    /// covers for the cohort before the walk: the embeddings and the embedding
    /// half of the input on the forward span, opened for the whole step; the
    /// hidden half, the pre-mix, the selection and the attention on the
    /// attention span; the FFN pre-mix and the MoE on the FFN span; and the
    /// shared head's mix and the logits back on the forward span. The attention
    /// and FFN phases close in turn, as a trunk layer closes them. The logits
    /// are returned past the forward phase's close:
    /// the walk's argmax is the next launch on this stream, so it reads them
    /// before the next step's forward phase carves over them, and its own
    /// output — inheriting a closed generation's ticket — lands on the pool.
    #[allow(clippy::too_many_arguments)]
    fn head_draft_step(
        &self,
        head: &MtpHead,
        ids: &Tensor,
        prev_wide: &Tensor,
        out: &Tensor,
        caches: &mut [&mut KvCache],
        seqs: &[usize],
        at: &[usize],
        params: &BatchedAttentionParams<'_>,
        idx_map: &mut HashMap<usize, Vec<IndexCache>>,
        index_rope: &FactoredRope,
        kv_layer: usize,
        eps: f64,
    ) -> Result<(Tensor, Tensor)> {
        let m = &self.model;
        let cfg = &m.cfg;
        let dev = &m.device;
        let n = at.len();
        let n_embd = cfg.hidden_size;
        let candle::Device::Cuda(cuda) = dev else {
            candle::bail!("qwen4exp draft runs on CUDA");
        };
        let stream = cuda.cuda_stream();

        let GpuLayerMix::Attention {
            w: aw,
            indexer,
            compress_ratio,
        } = &head.block.mix
        else {
            candle::bail!("qwen4exp draft: the head's block is not full-attention");
        };

        // The forward phase holds the step's embeddings — and with them the
        // embedding half of the head's input, which a norm keeps beside the rows
        // it reads — and, at the end, the shared head's logits.
        let fwd_wave = begin_wave(&stream, LayerPhase::Forward)?;
        let fwd_ticket = Some(fwd_wave.ticket());
        // Fully written by the lookup (invariant 6), in the head's F32.
        let embeds = wave_empty_ticketed((n, m.embed.ncols()), DType::F32, dev, fwd_ticket)?;
        m.embed.gather_into(ids, None, Some(&embeds))?;

        // ── Attention half. ──
        let attn_wave = begin_wave(&stream, LayerPhase::Attention)?;
        let attn_ticket = Some(attn_wave.ticket());
        // Assembled on the span, then laid into the carry buffer the next step
        // reads: the residual outlives this phase and the walk's step.
        let assembled = head.block_input(&embeds, prev_wide, eps, attn_ticket)?;
        out.slice_set(&assembled, 0, 0)?;
        let mut res = out.clone();
        let (h, inject) = hc_mix(&res, &head.block.hc_attn, eps, attn_ticket)?;
        let inject = inject.expect("a block's HC modules carry an inject");
        // One decode row per sequence, so the spans are one row each. The span
        // names the SEQUENCE, not the row: `layer_selection` keys each
        // sequence's index cache by it, and a row ordinal would read another
        // sequence's cache as soon as the cohort is wider than one.
        let spans: Vec<SeqSpan> = seqs
            .iter()
            .enumerate()
            .map(|(i, &seq)| SeqSpan {
                seq,
                start: i,
                len: 1,
            })
            .collect();
        let g_sel = gpu_span("q4e:draft:select", dev);
        let sel = self.layer_selection(
            kv_layer,
            *compress_ratio,
            indexer,
            index_rope,
            &h,
            &spans,
            at,
            idx_map,
            // A draft walk is rolled back whole, so it has nothing to rewind
            // partially and nothing to capture.
            None,
            n,
            // Read by this step's attention and nothing after it.
            attn_ticket,
        )?;
        g_sel.end();
        let g_attn = gpu_span("q4e:draft:attn", dev);
        let alayer = Qwen4ExpAttentionLayer {
            w: aw,
            n_head: cfg.num_attention_heads,
            n_kv_head: cfg.num_kv_heads,
            head_dim: cfg.attn_head_dim,
            rotary: &m.rotary,
        };
        let x_g = TensorCat::from_cat_tensor(h.reshape((n, 1, n_embd))?.contiguous()?, 0)?;
        // **Layer index `0`, not `kv_layer` — the two indices mean different
        // things and only coincide when the group starts at layer 0.**
        //
        // This one addresses the slot-header buffer, as `ptr + idx * stride`,
        // and the walk built that buffer for the group `kv_layer..kv_layer + 1`
        // — one entry, at index 0. Passing the absolute layer would read twelve
        // strides past a one-layer buffer: an illegal access reported at
        // whatever launch comes next, which is the out-projection, nowhere near
        // the cause. `kv_layer` stays absolute where it indexes a *sequence's*
        // caches, which is why both appear here.
        //
        // [`Self::head_wave_pass`] passes the absolute index for the opposite
        // reason: it rides the wave's own metadata, which covers every layer.
        //
        // The projection's output borrows the attention phase, so the combine
        // reads it before the phase closes — the compiler refuses the order
        // the other way round.
        let y = forward_attn_batched(
            &alayer,
            caches,
            &x_g,
            at,
            params,
            0,
            sel.as_ref(),
            Some(&attn_wave),
        )?
        .reshape((n, n_embd))?;
        hc_combine(&mut res, &y, &inject)?;
        drop(y);
        drop(sel);
        drop(attn_wave);
        g_attn.end();

        // ── MoE half. ──
        let g_moe = gpu_span("q4e:draft:moe", dev);
        let ffn_wave = begin_wave(&stream, LayerPhase::Ffn)?;
        let (h2, inject2) = hc_mix(&res, &head.block.hc_ffn, eps, Some(ffn_wave.ticket()))?;
        let inject2 = inject2.expect("a block's HC modules carry an inject");
        // Quantized once in the session's mode, as the trunk's MoE input is:
        // the router, the shared expert and the routed experts' tile gather all
        // read the one operand.
        let acts = to_dynamic(
            &h2.reshape((1, n, n_embd))?,
            m.lm_head.int8mode(),
            cuda,
            // Raw Σx — a language model's block sums stay far below f16's
            // ceiling.
            candle::quantized::SumScale::Raw,
        )?;
        // The layer's output in its three parts, assembled by the combine as
        // the trunk's are. A draft head only ever runs behind a decode step —
        // there is no prefill/prompt traffic through one — so all `n` rows are
        // decode-attributed.
        let parts = head
            .block
            .moe
            .forward_parts(acts, DType::F32, n, Some(&ffn_wave))?;
        let routed = parts.routed.reshape((n, n_embd))?;
        hc_combine_gated(&mut res, &routed, &parts.shared, &inject2)?;
        drop(routed);
        drop(parts);
        drop(ffn_wave);
        g_moe.end();

        // ── The shared head. ──
        let g_head = gpu_span("q4e:draft:lm_head", dev);
        let narrow = head.to_shared_head(&res, eps, fwd_ticket)?;
        let acts = to_dynamic(
            &narrow,
            m.lm_head.int8mode(),
            cuda,
            candle::quantized::SumScale::Raw,
        )?;
        let logits = m
            .lm_head
            .forward_dynamic(acts.as_dynamic(), DType::F32)?
            .reshape((n, cfg.vocab_size))?;
        drop(acts);
        drop(fwd_wave);
        g_head.end();
        Ok((res, logits))
    }

    /// The draft walk's two carried-residual buffers, `[n, hc, n_embd]` each.
    ///
    /// A step reads the residual the step before it wrote and writes its own
    /// beside it, so the walk alternates between two buffers: the seeds are
    /// laid into the first, and step `j` writes whichever one step `j − 1` did
    /// not. Both are held by the engine and grow only when a cohort is wider
    /// than any before it, so a walk allocates nothing for its carry.
    fn draft_carry(&self, n: usize) -> Result<(Tensor, Tensor)> {
        let cfg = &self.model.cfg;
        let (hc, n_embd) = (cfg.hc.count, cfg.hidden_size);
        let mut pair = self
            .draft_carry
            .lock()
            .map_err(|_| candle::Error::Msg("draft carry lock poisoned".into()))?;
        let held = match pair.as_ref() {
            Some((a, _)) => a.dim(0)?,
            None => 0,
        };
        if held < n {
            // Fully written before it is read: the seeds go into the first, and
            // each step writes the buffer it then hands on (invariant 6). Grown
            // by doubling, so a cohort ramping up settles in a few steps.
            let dims = ((held * 2).max(n), hc, n_embd);
            *pair = Some((
                Tensor::empty(dims, DType::F32, &self.model.device)?,
                Tensor::empty(dims, DType::F32, &self.model.device)?,
            ));
        }
        let (a, b) = pair.as_ref().expect("sized above");
        Ok((a.narrow(0, 0, n)?, b.narrow(0, 0, n)?))
    }

    /// Draft up to `max_len` proposals for the whole cohort with the head.
    ///
    /// The loop itself — the pre-ensure, the per-step slot headers, the cache
    /// advance, the rollback, the single readback — is
    /// [`draft_walk`](crate::models::draft_walk::draft_walk), shared with every
    /// other drafter in the tree. All that is this model's own is the rope
    /// tables and [`Self::head_draft_step`].
    ///
    /// Empty — a plain decode step — when the artifact carries no head, or
    /// before a sequence has a seed. A sequence always acquires one from the
    /// wave that prefilled it ([`Self::head_wave_pass`]), so the empty case is
    /// the first step of a fresh cohort and nothing else.
    pub(super) fn mtp_draft(
        &self,
        session: &mut BatchedInferenceSession,
        seqs: &[usize],
        committed: &[u32],
        max_len: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let n = seqs.len();
        let m = &self.model;
        let (Some(head), Some(kv_layer)) = (m.mtp.as_ref(), m.cfg.mtp_kv_layer()) else {
            return Ok(vec![Vec::new(); n]);
        };
        if n == 0 || max_len == 0 {
            return Ok(vec![Vec::new(); n]);
        }
        let eps = m.cfg.rms_norm_eps;
        let dev = &m.device;

        // Every sequence needs a seed. One that has none has not been through a
        // wave yet, so it takes a plain decode row and drafts from the next step.
        let seed_rows: Vec<Tensor> = {
            let seeds = self
                .seeds
                .read()
                .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?;
            let mut rows: Vec<Tensor> = Vec::with_capacity(n);
            for &s in seqs {
                match seeds.get(&s) {
                    Some(seed) => rows.push(seed.clone()),
                    None => return Ok(vec![Vec::new(); n]),
                }
            }
            rows
        };

        // **The walk runs on the previous forward's tier, priced for that
        // forward's width rather than this cohort's.** Each step is one decode
        // row per sequence through the head's assembly, its attention, its FFN
        // and the shared head, so the tier must hold the plan for `n` decode
        // rows; covering it is a no-op when it already does.
        if let candle::Device::Cuda(d) = dev {
            let plan = WavePlan::new(self.wave_geometry(DType::F32));
            // The head's selection at the walk's deepest position, which is the
            // widest any of its steps scores.
            let deepest: Vec<usize> = seqs
                .iter()
                .map(|&s| session.sequence_offset(s).unwrap_or(0) + max_len - 1)
                .collect();
            let num_layers = m.cfg.num_layers;
            let width = WaveWidth {
                qsa_bytes: self.wave_qsa_bytes(
                    seqs,
                    &vec![1; n],
                    &deepest,
                    num_layers,
                    num_layers,
                )?,
                ..WaveWidth::decode(n)
            };
            cover_wave_transient(
                &d.cuda_stream(),
                [
                    plan.phase_bytes(LayerPhase::Attention, width),
                    plan.phase_bytes(LayerPhase::Ffn, width),
                    plan.phase_bytes(LayerPhase::Forward, width),
                ],
            )?;
        }

        // The walk reserves these itself, but the rope depth below has to be
        // read after the reservation — it can grow the backing's block count.
        draft_reserve(session, seqs, kv_layer, max_len)?;
        let theta = m.cfg.rope_theta;
        let q_lens = vec![1usize; n];
        let mut idx_map = self
            .index
            .write()
            .map_err(|_| candle::Error::Msg("index lock poisoned".into()))?;

        // **Grow the head's index cache for the whole walk, before it starts.**
        //
        // The same hazard the KV pre-ensure exists for, one buffer over: the
        // cache is sized at wave entry for the tokens that wave lands, and a
        // walk runs `max_len` positions PAST that. Growing it reallocates
        // `keys`, and the walk never synchronises — so a step that grew the
        // cache would free the buffer an earlier step's selection kernel is
        // still reading, and the address is reissued immediately by a pool
        // being churned constantly. That is an illegal access, not an error
        // return, and it poisons the context for every later CUDA call.
        let ratios = self.attention_ratios();
        let ratio = ratios.get(kv_layer).copied().unwrap_or(0);
        if ratio > 0 {
            for &s in seqs {
                let base = session.sequence_offset(s).unwrap_or(0);
                let cache = idx_map
                    .get_mut(&s)
                    .and_then(|c| c.get_mut(kv_layer))
                    .ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "qwen4exp draft: seq {s} has no head index cache"
                        ))
                    })?;
                cache.ensure_capacity(base + max_len, ratio)?;
            }
        }

        // **The walk opens no forward of its own, deliberately.**
        //
        // A forward's transient tier is not released when the forward ends — it
        // stands until the next forward's phase 0 returns it — so a caller
        // arriving BETWEEN forwards, which is what a draft walk is, already has
        // ground for the head's attention to lay its spans in.
        //
        // Opening one here is worse than unnecessary: a forward that owns the
        // partition refuses every arena created inside it, and a walk cannot
        // avoid creating them. Its steps write, a filling chunk gets sealed into
        // a policy-chosen format, and the next step's
        // `build_decode_metadata_at` asks for a chunk of a key that did not
        // exist when the walk began. The key is not knowable in advance —
        // it depends on the data just written — so no amount of pre-ensuring
        // reaches it. Uncompressed runs survive only because every key they
        // touch already exists; C8 does not.
        //
        // A bracket here was originally added to fix a
        // `CUDA_ERROR_ILLEGAL_ADDRESS` in the head's out-projection. It did not:
        // the run after adding it failed identically. What fixed that was
        // passing the GROUP-RELATIVE slot-header index (`0`, not the absolute KV
        // layer) — see `head_draft_step`. The bracket was credited for a fix it
        // had no part in, and then cost every compressed rung.

        // **The walk rolls back the KV; the index cache is ours to roll back.**
        //
        // `draft_walk` truncates the head's paged K/V to where it found it, but
        // the head also appends to its QSA index cache on every drafted
        // position, and nothing down there knows that cache exists. Left
        // standing, the drafted keys are scored by every later selection as if
        // they were real context — a proposal that was rejected still steering
        // attention, silently and for the rest of the sequence. Taken here,
        // before the step closure borrows the map.
        let snaps: Vec<IndexSnapshot> = seqs
            .iter()
            .map(|s| {
                idx_map
                    .get(s)
                    .and_then(|c| c.get(kv_layer))
                    .ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "qwen4exp draft: seq {s} has no head index cache"
                        ))
                    })
                    .and_then(IndexCache::snapshot)
            })
            .collect::<Result<_>>()?;

        let index_rope = self.index_rope().clone();

        // The seeds, laid into the first carry buffer in one launch. Taken
        // under the index lock, which every walk holds for its whole length,
        // so no other walk can be writing the same pair. The staging
        // generation stays open across the walk: the last one to close
        // synchronises the stream, and the walk's own readback does that.
        let (carry_a, carry_b) = self.draft_carry(n)?;
        let seed_generation = session.begin_stager_generation();
        {
            let width = m.cfg.hc.count * m.cfg.hidden_size;
            let a_rows = carry_a.reshape((n, width))?;
            let runs: Vec<RowRun<'_>> = seed_rows
                .iter()
                .enumerate()
                .map(|(i, s)| Ok(RowRun::new(s.reshape((1, width))?, &a_rows, i)))
                .collect::<Result<_>>()?;
            rows_scatter(&runs, &seed_generation)?;
        }
        drop(seed_rows);

        let mut step = |ids: &Tensor,
                        h: &Tensor,
                        caches: &mut [&mut KvCache],
                        at: &[usize],
                        headers: (&GpuBuf, u64),
                        generation: &Generation|
         -> Result<(Tensor, Tensor)> {
            let pos: Vec<u32> = at.iter().map(|&p| p as u32).collect();
            let model_rope =
                LazyRope::new(|| m.rotary.rope_cos_sin(&pos, theta, DType::F32, dev, None));
            let pm: RefCell<Option<SharedPm>> = RefCell::new(None);
            let params = BatchedAttentionParams::new(
                &model_rope,
                false,
                &self.rope,
                DecodeHeaders::Decode {
                    buf: Some(headers.0.clone()),
                    stride: headers.1,
                },
                &q_lens,
                generation,
                &pm,
            );
            // Whichever carry buffer the previous step did not write.
            let out = if h.same_storage(&carry_a) {
                &carry_b
            } else {
                &carry_a
            };
            self.head_draft_step(
                head,
                ids,
                h,
                out,
                caches,
                seqs,
                at,
                &params,
                &mut idx_map,
                &index_rope,
                kv_layer,
                eps,
            )
        };
        let walked = draft_walk(
            session, seqs, kv_layer, committed, &carry_a, max_len, &mut step,
        );
        drop(seed_generation);
        for (&s, snap) in seqs.iter().zip(&snaps) {
            if let Some(cache) = idx_map.get_mut(&s).and_then(|c| c.get_mut(kv_layer)) {
                cache.restore(snap)?;
            }
        }
        walked
    }
}
