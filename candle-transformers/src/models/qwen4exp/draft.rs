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
use super::engine::GpuLayerMix;
use super::hyper::{hc_combine, hc_mix};
use super::indexer::{IndexCache, IndexSnapshot};
use super::mtp::MtpHead;
use super::spec::SpecCapture;
use super::wave::Qwen4ExpBatched;
use crate::models::batched_inference::BatchedInferenceSession;
use crate::models::batched_layer::{forward_attn_batched, BatchedAttentionParams, DecodeHeaders};
use crate::models::delta_net::SeqSpan;
use crate::models::draft_walk::{draft_reserve, draft_rope_depth, draft_walk};
use crate::models::kv_cache_utils::SequenceContext;
use crate::models::prefill_utils::SharedPm;
use crate::models::qwen35::attention::RopeTables;
use crate::models::tensor_cat::TensorCat;
use candle::quantized::cuda::to_dynamic;
use candle_nn::kv_cache::KvCache;

/// Each sequence's last wide residual, carried between waves so the head's
/// first row has the `h(t-1)` the wave before it produced.
///
/// Keyed by slot id, and slot ids are **recycled**: a stale entry inherited by
/// the next conversation to land on the id would seed the head from a history
/// this sequence never had. Cleared through the engine's `release_sequence`
/// alongside the recurrent and index state, and reset whenever a sequence
/// starts over at offset 0.
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
    /// The indexer's RoPE tables, built for this wave's depth.
    pub index_rope: &'a RopeTables,
    /// The head's KV layer, which is also its index into a sequence's index
    /// caches — past every trunk attention layer.
    pub kv_layer: usize,
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
        res: &Tensor,
        embeds: &Tensor,
        w: &HeadWave<'_>,
        eps: f64,
    ) -> Result<()> {
        let cfg = &self.model.cfg;
        let dev = &self.model.device;
        let (total_rows, hc, n_embd) = res.dims3()?;
        let g = crate::models::profile::gpu_span("q4e:mtp_head", dev);

        // ── The shift, as one gather. ──
        //
        // Row `i` of the head reads the trunk residual of row `i-1`, and the
        // first row of each span reaches back to the previous wave's seed. Both
        // sources are addressed by ONE index list over a pool of
        // `[seeds ; res]`, so the shift costs a single `index_select` rather
        // than a per-span splice — the rows are not contiguous in either
        // source, and stitching them per sequence is the copy storm invariant 2
        // exists to refuse.
        let mut seed_rows: Vec<Tensor> = Vec::with_capacity(w.spans.len());
        for span in w.spans {
            let seed = match seeds.get(&span.seq) {
                Some(t) => t.clone(),
                // No previous wave: position 0 of the sequence, where the head
                // has no carried state and the reference takes zeros.
                None => Tensor::zeros((1, hc, n_embd), DType::F32, dev)?,
            };
            seed_rows.push(seed);
        }
        let seed_block = if seed_rows.len() == 1 {
            seed_rows.pop().expect("one span")
        } else {
            Tensor::cat(&seed_rows, 0)?
        };
        let pool = Tensor::cat(&[&seed_block, res], 0)?;
        let n_seed = w.spans.len();
        let mut shift_idx: Vec<u32> = vec![0; total_rows];
        for (s, span) in w.spans.iter().enumerate() {
            for j in 0..span.len {
                shift_idx[span.start + j] = if j == 0 {
                    s as u32
                } else {
                    (n_seed + span.start + j - 1) as u32
                };
            }
        }
        let shift_idx = Tensor::from_vec(shift_idx, (total_rows,), dev)?;
        let shifted = pool.index_select(&shift_idx, 0)?;

        // ── The head's block input, entering the hyper-connection stream the
        // way a token embedding does. ──
        let x_head = head.block_input(embeds, &shifted, eps)?;

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

        let (h, _) = hc_mix(&x_head, &head.block.hc_attn, eps)?;

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
                None,
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
                None,
            )?;
        }

        // ── Carry each sequence's rows forward. ──
        //
        // **`to_owned_tensor`, not `contiguous`.** `res` is a wave-arena tensor
        // and the generation reset reclaims it, so a seed holding a view into
        // it reads another wave's activations by the time the next wave reads
        // the seed.
        //
        // `contiguous()` cannot get us out of the arena, in either of its
        // branches: on an already-contiguous tensor — which a dim-0 `narrow` of
        // one is — it returns `self.clone()`, sharing the storage outright; and
        // when it does copy it allocates with `self.wave_ticket()`, so the copy
        // is arena-leased too. Only `to_owned_tensor` clones the storage into a
        // fresh `Arc` that no generation owns.
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
        for span in w.spans {
            if let Some(cap) = capture.as_deref_mut() {
                if let Some(stash) = cap.seqs.get_mut(&span.seq) {
                    stash.head_rows = Some(res.narrow(0, span.start, span.len)?.to_owned_tensor()?);
                }
            }
            let last = res
                .narrow(0, span.start + span.len - 1, 1)?
                .to_owned_tensor()?;
            seeds.insert(span.seq, last);
        }
        g.end();
        Ok(())
    }

    /// One drafted position for the whole cohort: the head's block over one row
    /// per sequence, writing its K/V at `at`.
    ///
    /// Returns `(wide residual out, logits)` — the residual because it is what
    /// the next drafted position reads as its `h(t-1)`, and the logits because
    /// the walk takes their argmax as the proposal.
    ///
    /// **The whole block, not attention only.** The lockstep pass
    /// ([`Self::head_wave_pass`]) skips the MoE because it owes only K/V; a
    /// proposal is the block's *output*, so here the FFN half runs. Both halves
    /// go through the same `hc_mix`/`hc_combine` pair every trunk layer uses.
    #[allow(clippy::too_many_arguments)]
    fn head_draft_step(
        &self,
        head: &MtpHead,
        embeds: &Tensor,
        prev_wide: &Tensor,
        caches: &mut [&mut KvCache],
        seqs: &[usize],
        at: &[usize],
        params: &BatchedAttentionParams<'_>,
        idx_map: &mut HashMap<usize, Vec<IndexCache>>,
        index_rope: &RopeTables,
        kv_layer: usize,
        eps: f64,
    ) -> Result<(Tensor, Tensor)> {
        let m = &self.model;
        let cfg = &m.cfg;
        let dev = &m.device;
        let n = at.len();
        let n_embd = cfg.hidden_size;

        let GpuLayerMix::Attention {
            w: aw,
            indexer,
            compress_ratio,
        } = &head.block.mix
        else {
            candle::bail!("qwen4exp draft: the head's block is not full-attention");
        };

        let mut res = head.block_input(embeds, prev_wide, eps)?;

        // ── Attention half. ──
        let (h, inject) = hc_mix(&res, &head.block.hc_attn, eps)?;
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
        )?;
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
        let y = forward_attn_batched(&alayer, caches, &x_g, at, params, 0, sel.as_ref(), None)?
            .to_owned_tensor()?
            .reshape((n, n_embd))?;
        res = hc_combine(&res, &y, &inject)?;

        // ── MoE half. ──
        let (h2, inject2) = hc_mix(&res, &head.block.hc_ffn, eps)?;
        let inject2 = inject2.expect("a block's HC modules carry an inject");
        let candle::Device::Cuda(cuda) = dev else {
            candle::bail!("qwen4exp draft runs on CUDA");
        };
        // Float activations for the same reason the trunk's MoE uses them: the
        // int8 expert gather tiles at 1024 and this stack's hidden is 2560.
        let acts = to_dynamic(
            &h2.reshape((1, n, n_embd))?,
            candle::quantized::Int8Mode::Off,
            cuda,
        )?;
        let y2 = head
            .block
            .moe
            .forward_dynamic(acts, DType::F32, None)?
            .to_owned_tensor()?
            .reshape((n, n_embd))?;
        res = hc_combine(&res, &y2, &inject2)?;

        // ── The shared head. ──
        let narrow = head.to_shared_head(&res, eps)?;
        let acts = to_dynamic(&narrow, m.lm_head.int8mode(), cuda)?;
        let logits = m
            .lm_head
            .forward_dynamic(acts.as_dynamic(), DType::F32)?
            .to_owned_tensor()?
            .reshape((n, cfg.vocab_size))?;
        Ok((res, logits))
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
        let seed_block = {
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
            if rows.len() == 1 {
                rows.pop().expect("one row")
            } else {
                Tensor::cat(&rows, 0)?
            }
        };

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

        // **A draft walk is a forward, and has to be bracketed like one.**
        //
        // `forward_attn_batched` opens a wave per phase, but a wave only *lays
        // out* spans inside a tier someone else placed — `plan_wave_transient`
        // is what buys that ground, and `begin_forward` is what freezes the
        // partition while the walk runs on it. Skip the bracket and the head's
        // attention writes into whatever tenant owns the address instead, which
        // is hot-path invariant 7 and surfaces as an illegal access inside the
        // out-projection rather than anywhere near the cause.
        //
        // Priced for this walk: one decode row per sequence, the same way the
        // forward prices its own rows.
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
        // This bracket was originally added to fix a `CUDA_ERROR_ILLEGAL_ADDRESS`
        // in the head's out-projection. It did not: the run after adding it
        // failed identically. What fixed that was passing the GROUP-RELATIVE
        // slot-header index (`0`, not the absolute KV layer) — see
        // `head_draft_step`. The bracket was credited for a fix it had no part
        // in, and then cost every compressed rung.
        let open_forward = || Ok(());

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

        let depth = draft_rope_depth(session, seqs, kv_layer)?;
        let rope_cs = self.rope_cs_for(depth)?;
        // Hoisted: the indexer's tables are wave-invariant, and building them
        // per drafted position would take a lock and rebuild a table per token.
        let index_rope = self.index_rope_for(depth)?;

        let mut step = |ids: &Tensor,
                        h: &Tensor,
                        caches: &mut [&mut KvCache],
                        at: &[usize],
                        headers: (&GpuBuf, u64),
                        generation: &Generation|
         -> Result<(Tensor, Tensor)> {
            let pos: Vec<u32> = at.iter().map(|&p| p as u32).collect();
            let (cos, sin) = m.rotary.rope_cos_sin(&pos, theta, DType::F32, dev)?;
            let pm: RefCell<Option<SharedPm>> = RefCell::new(None);
            let params = BatchedAttentionParams::new(
                &cos,
                &sin,
                false,
                &self.inv_freq,
                &rope_cs,
                DecodeHeaders::Decode {
                    buf: Some(headers.0.clone()),
                    stride: headers.1,
                },
                &q_lens,
                generation,
                &pm,
            );
            let embeds = m.embed.index_select(ids, 0)?.to_dtype(DType::F32)?;
            self.head_draft_step(
                head,
                &embeds,
                h,
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
            session,
            seqs,
            kv_layer,
            committed,
            &seed_block,
            max_len,
            open_forward,
            &mut step,
        );
        for (&s, snap) in seqs.iter().zip(&snaps) {
            if let Some(cache) = idx_map.get_mut(&s).and_then(|c| c.get_mut(kv_layer)) {
                cache.restore(snap)?;
            }
        }
        walked
    }
}
