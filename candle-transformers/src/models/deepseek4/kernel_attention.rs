//! Per-layer paged-kernel attention state for the engine: the projections
//! (`wq_a`/`wq_b`/`wkv`, norms) stay host-side int8, while the attention math
//! itself — FP8 window walk, compressed top-k walk, sink softmax, RoPE and
//! de-rotation — runs in the `paged-latent` decode kernel over the
//! production chunked-arena slot. The compressed corpus lives in a
//! [`FloatGallery`] (attended entries pre-RoPE + Indexer keys), fed by the
//! host streaming compressors on group completion, and CSA selection is the
//! two-stage BDP-recall → Indexer-precision pass, fully on-device.
//!
//! [`super::attention::IncrementalAttention`] remains the frozen CPU
//! reference; this is the product path it validates.

use candle::{DType, Result, Tensor};

use crate::models::profile::{pipeline_record, profile_now};

use super::attention::{rms_norm, Attention};
use super::compressor::IncrementalCompressor;
use super::config::LayerKind;
use super::gallery::{CorpusSnapshot, FloatGallery};
use super::linear::shared_int8_pair;
use std::sync::Arc;

use super::paged::{
    build_rope_table, paged_latent_decode_raw, CorpusCache, LatentWorkspace, HEAD_DIM, NOPE_BANDS,
    NOPE_DIM, ROPE_DIM,
};
use super::rope::{yarn_freqs, RotaryCache};

/// Recall shortlist width for the two-stage selection: 2× the Indexer's
/// top-k, capped at the device argsort's 1024-column rescore limit.
pub fn shortlist_m(top_k: usize) -> usize {
    (2 * top_k).clamp(64, 1024)
}

/// Per-layer kernel-attention constants shared by every sequence: sink
/// logits, the factored RoPE cos/sin table, and the zero-entry placeholders.
/// One per layer, built once with the model.
pub struct KernelLayerStatic {
    /// Per-head sink logits `[n_heads]` f32, device.
    sinks: Tensor,
    /// Factored RoPE cos/sin table `[ROPE_TAB_LEN]` f32, device — built from
    /// the layer-kind YaRN frequencies by `build_rope_table` at load.
    rope_tab: Tensor,
    /// Split-KV partial workspace, ONE per model shared by every layer (the
    /// wave thread launches sequentially on one stream). Host-immutable —
    /// only kernels write it, stream-ordered — so plain `Arc` sharing, no
    /// locks.
    ws: Arc<LatentWorkspace>,
    /// Zero-entry placeholders (the two-region cache) so the kernel always gets
    /// valid pointers when a slot selects nothing.
    empty_nope_i8: Tensor,
    empty_nope_scale: Tensor,
    empty_rope_bf: Tensor,
    empty_pos: Tensor,
    empty_idx: Tensor,
    empty_cnt: Tensor,
}

/// Per-(layer, sequence) corpus state: the streaming compressors and the
/// compressed-corpus gallery. One per sequence per compression layer.
pub struct KernelLayerSeqState {
    /// Attention-side streaming compressor (pre-RoPE entries), compression
    /// layers only.
    pub(crate) comp: Option<IncrementalCompressor>,
    /// Indexer-side streaming compressor (roped keys), CSA layers only.
    pub(crate) icomp: Option<IncrementalCompressor>,
    /// The compressed corpus pair + sign index. `None` on SWA layers.
    pub(crate) gallery: Option<FloatGallery>,
}

/// One layer's kernel-path attention state for a single sequence — the
/// engine session's composition of the shared statics and one sequence's
/// corpus state.
pub struct KernelAttnLayer {
    st: KernelLayerStatic,
    seq: KernelLayerSeqState,
}

impl KernelLayerStatic {
    pub fn sinks(&self) -> &Tensor {
        &self.sinks
    }
    pub fn rope_tab(&self) -> &Tensor {
        &self.rope_tab
    }
    pub fn ws(&self) -> &LatentWorkspace {
        &self.ws
    }
    /// An empty two-region `CorpusCache` (valid pointers, nothing selectable) —
    /// used when a slot/wave selects no compressed entries.
    pub fn empty_corpus_cache(&self) -> Result<CorpusCache> {
        CorpusCache::from_gathered(
            self.empty_nope_i8.clone(),
            self.empty_nope_scale.clone(),
            self.empty_rope_bf.clone(),
            self.empty_pos.clone(),
            0, // one-row placeholder shape, but zero real entries
        )
    }

    pub fn new(
        a: &Attention,
        theta: f64,
        original_seq_len: usize,
        rope_factor: f64,
        beta_fast: f64,
        beta_slow: f64,
        ws: Arc<LatentWorkspace>,
        device: &candle::Device,
    ) -> Result<Self> {
        let sinks = a.attn_sink().to_dtype(DType::F32)?.to_device(device)?;
        let freqs_v: Vec<f32> = yarn_freqs(
            ROPE_DIM,
            theta,
            original_seq_len,
            rope_factor,
            beta_fast,
            beta_slow,
        )
        .into_iter()
        .map(|f| f as f32)
        .collect();
        let freqs = Tensor::from_vec(freqs_v, ROPE_DIM / 2, device)?;
        let rope_tab = build_rope_table(&freqs)?;
        Ok(Self {
            sinks,
            rope_tab,
            ws,
            empty_nope_i8: Tensor::zeros((1, NOPE_DIM), DType::U8, device)?,
            empty_nope_scale: Tensor::zeros((1, NOPE_BANDS), DType::F32, device)?,
            empty_rope_bf: Tensor::zeros((1, ROPE_DIM), DType::BF16, device)?,
            empty_pos: Tensor::zeros(1, DType::U32, device)?,
            empty_idx: Tensor::zeros((1, 1), DType::U32, device)?,
            empty_cnt: Tensor::zeros(1, DType::U32, device)?,
        })
    }
}

impl KernelLayerSeqState {
    pub fn new(a: &Attention, index_head_dim: usize, device: &candle::Device) -> Result<Self> {
        let comp = a.compressor().map(|c| c.incremental());
        let icomp = a.indexer().map(|ix| ix.incremental_compressor());
        let gallery = if a.compressor().is_some() {
            // HCA layers have no Indexer: key rows are a 1-wide placeholder
            // (selection is "all entries"; the scoring side is never touched).
            let key_dim = if a.indexer().is_some() {
                index_head_dim
            } else {
                1
            };
            Some(FloatGallery::new(device, HEAD_DIM, key_dim, 64)?)
        } else {
            None
        };
        Ok(Self {
            comp,
            icomp,
            gallery,
        })
    }

    /// Turn-seal **close** (Artifact B → C): finalize this layer's trailing
    /// partial compressor groups into corpus entries and append them to the
    /// gallery. After this the live window tail is fully represented in the
    /// compressed corpus, so the turn is persistable and resumable from the
    /// corpus alone (docs/deepseek_turn_seal_persistence.md §2). The attention
    /// and indexer compressors share group boundaries, so they close in lockstep
    /// (`comp` yields the attended entry, `icomp` the scoring key; HCA layers
    /// have no indexer and store a 1-wide placeholder key). A no-op on SWA
    /// layers (no compressor/gallery) and when the buffers are already empty (the
    /// group boundary fell exactly on the seal).
    pub fn seal_close(&mut self) -> Result<()> {
        let (Some(comp), Some(gallery)) = (self.comp.as_mut(), self.gallery.as_mut()) else {
            return Ok(());
        };
        if let Some((entry, gpos)) = comp.close()? {
            let (_, _, hd) = entry.dims3()?;
            let key = match self.icomp.as_mut() {
                Some(ic) => ic
                    .close()?
                    .expect("indexer compressor shares group boundaries")
                    .0
                    .reshape((1, ()))?,
                None => Tensor::zeros((1, 1), DType::F32, entry.device())?,
            };
            gallery.append_batch(&entry.reshape((1, hd))?, &key, &[gpos])?;
        } else if let Some(ic) = self.icomp.as_mut() {
            let none = ic.close()?;
            debug_assert!(none.is_none(), "compressor group boundaries diverged");
        }
        Ok(())
    }

    /// Snapshot this layer's compressed corpus in native durable form (Artifact
    /// C). `None` on SWA layers (no gallery). Call after [`Self::seal_close`] so
    /// the trailing partial is included.
    pub fn snapshot_gallery(&self) -> Result<Option<CorpusSnapshot>> {
        match &self.gallery {
            Some(g) => Ok(Some(g.snapshot()?)),
            None => Ok(None),
        }
    }

    /// Inject a persisted corpus snapshot into this layer's gallery on resume
    /// (Artifact C). Replaces the (fresh, empty) gallery with the restored one.
    /// `positions` are the RECONSTRUCTED group-start positions (`base + i·ratio`)
    /// the entries take in the resumed context — computed by the caller from the
    /// reconstruction layout, not read from the snapshot (which stores none).
    pub fn restore_gallery(
        &mut self,
        device: &candle::Device,
        snap: &CorpusSnapshot,
        positions: &[u32],
    ) -> Result<()> {
        self.gallery = Some(FloatGallery::from_snapshot(device, snap, positions)?);
        Ok(())
    }
}

impl KernelAttnLayer {
    pub fn new(
        a: &Attention,
        theta: f64,
        original_seq_len: usize,
        rope_factor: f64,
        beta_fast: f64,
        beta_slow: f64,
        index_head_dim: usize,
        ws: Arc<LatentWorkspace>,
        device: &candle::Device,
    ) -> Result<Self> {
        Ok(Self {
            st: KernelLayerStatic::new(
                a,
                theta,
                original_seq_len,
                rope_factor,
                beta_fast,
                beta_slow,
                ws,
                device,
            )?,
            seq: KernelLayerSeqState::new(a, index_head_dim, device)?,
        })
    }

    /// One decode step for this layer: the token's normalized hidden state
    /// `x`, its position, and this layer's `SlotHeader` device address →
    /// the attention output `[1, 1, dim]` (reference-row semantics).
    pub fn step(
        &mut self,
        a: &Attention,
        x: &Tensor,
        rope: &RotaryCache,
        pos: usize,
        headers_ptr: u64,
    ) -> Result<Tensor> {
        // The engine session's slot is host-authored FP8 identity (no seal-
        // time regrouping runs on this rung-3 harness path).
        kernel_attn_decode_step(a, &self.st, &mut self.seq, x, rope, pos, headers_ptr)
    }
}

/// One decode step through the paged kernel — the shared building block for
/// the engine session (one sequence) and the batched wave model (many).
#[allow(clippy::too_many_arguments)]
pub fn kernel_attn_decode_step(
    a: &Attention,
    st: &KernelLayerStatic,
    seq: &mut KernelLayerSeqState,
    x: &Tensor,
    rope: &RotaryCache,
    pos: usize,
    headers_ptr: u64,
) -> Result<Tensor> {
    let (h, hd) = (a.n_heads(), a.head_dim());
    let din = x.elem_count();
    let x = x.reshape((1, 1, din))?.to_dtype(DType::F32)?;

    // Host projections (int8-KO): query + latent, both PRE-RoPE — the
    // kernel rotates at the position it derives from the slot state.
    let qr = rms_norm(&a.wq_a().forward(&x)?, a.q_norm(), a.eps())?;
    let q = a.wq_b().forward(&qr)?.reshape((1, 1, h, hd))?;
    let q = a.rms_scale(&q)?;
    let q_bf = q.reshape((1, h, hd))?.to_dtype(DType::BF16)?;
    let kv = rms_norm(&a.wkv().forward(&x)?, a.kv_norm(), a.eps())?;
    let kv_bf = kv.reshape((1, hd))?.to_dtype(DType::BF16)?;

    // Corpus maintenance: when this token completes a group, append the
    // pre-RoPE attended entry + the (roped) Indexer key to the gallery.
    if let (Some(comp), Some(gallery)) = (seq.comp.as_mut(), seq.gallery.as_mut()) {
        if let Some((entry, gpos)) = comp.push_raw(&x)? {
            let key = match seq.icomp.as_mut() {
                Some(ic) => ic
                    .push(&x, rope)?
                    .expect("indexer compressor shares group boundaries")
                    .reshape((1, ()))?,
                None => Tensor::zeros((1, 1), DType::F32, x.device())?,
            };
            gallery.append_batch(&entry.reshape((1, hd))?, &key, &[gpos])?;
        } else if let Some(ic) = seq.icomp.as_mut() {
            let none = ic.push(&x, rope)?;
            debug_assert!(none.is_none(), "compressor group boundaries diverged");
        }
    }

    // Selection: SWA none; HCA all causal entries; CSA two-stage top-k.
    let n_entries = seq.gallery.as_ref().map_or(0, |g| g.len());
    let (cache, comp_idx, comp_cnt) = if n_entries == 0 {
        (
            st.empty_corpus_cache()?,
            st.empty_idx.clone(),
            st.empty_cnt.clone(),
        )
    } else {
        let gallery = seq.gallery.as_ref().unwrap();
        // Absolute entry ids the query attends: CSA two-stage top-k, HCA all.
        let (gids, k) = match a.kind() {
            LayerKind::Csa => {
                let ix = a.indexer().expect("CSA layer has an indexer");
                let (qi, w) = ix.query_space(&x, &qr, rope, pos)?;
                gallery.two_stage_select(&qi, &w, shortlist_m(ix.top_k()), ix.top_k())?
            }
            LayerKind::Hca => {
                let ids = Tensor::arange(0u32, n_entries as u32, x.device())?;
                (ids, n_entries)
            }
            LayerKind::SlidingWindow => (st.empty_idx.clone().reshape(1)?, 0),
        };
        if k == 0 {
            (
                st.empty_corpus_cache()?,
                st.empty_idx.clone(),
                st.empty_cnt.clone(),
            )
        } else {
            // Gather the k selected entries' HOT two-region cache (tier-aware) and
            // walk them densely — `comp_idx = 0..k`. The gallery pre-built the
            // int8 cache on append, so there is no per-step rebuild; the gathered
            // working set stays O(1) at any depth.
            let (ni8, nsc, rbf, cpos) = gallery.gather_corpus(&gids)?;
            let cache = CorpusCache::from_gathered(ni8, nsc, rbf, cpos, k)?;
            let comp_idx = Tensor::arange(0u32, k as u32, x.device())?.reshape((1, k))?;
            let cnt = Tensor::from_vec(vec![k as u32], 1, x.device())?;
            (cache, comp_idx, cnt)
        }
    };

    // The kernel: hybrid window+compressed attention, sink fold, de-rotation —
    // returns `[1, n_heads, 512]` bf16, reading the gathered two-region cache
    // directly (no per-step quant).
    let q_pos_t = Tensor::from_vec(vec![pos as u32], 1, x.device())?;
    let out = paged_latent_decode_raw(
        &q_bf,
        headers_ptr,
        &kv_bf,
        &cache,
        &comp_idx,
        &comp_cnt,
        &q_pos_t,
        &st.sinks,
        &st.rope_tab,
        a.softmax_scale() as f32,
        a.window_size(),
        0,     // auto split factor
        false, // wave hands a private per-token snapshot with the write-len
        // already patched host-side — the on-device commit would only touch a
        // throwaway copy, so skip it.
        &st.ws,
        None,
    )?;

    let o = out.to_dtype(DType::F32)?.reshape((1, h, 1, hd))?;
    a.output_proj(&o, 1, 1)
}

/// Whole-prompt batched prefill preparation. Every stateless projection is
/// hoisted OUT of the token loop into one GEMM over the full prompt `xs`
/// `[1, s, dim]`: the attention `wq_a`/`wq_b`/`wkv` (+ norms) and BOTH streaming
/// compressors' `wkv`/`wgate`. The token loop then only does the cheap stateful
/// work — buffering the pre-projected compressor rows, emitting a group when one
/// completes (`push_projected*`), and the per-token causal selection — so token
/// `t` still selects exactly over the entries completed by `t`. The projection
/// is the only thing moved out of the loop; the pooling/emit/selection semantics
/// (and their outputs) are unchanged. Returns the batched pre-RoPE bf16 query
/// `[s, n_heads, HEAD_DIM]`, latent `[s, HEAD_DIM]`, and each token's absolute
/// selected compressed GIDs.
#[allow(clippy::type_complexity)]
pub fn kernel_attn_prefill_prepare_batched(
    a: &Attention,
    seq: &mut KernelLayerSeqState,
    xs: &Tensor,
    rope: &RotaryCache,
    base: usize,
) -> Result<(Tensor, Tensor, Vec<Vec<u32>>)> {
    let (h, hd) = (a.n_heads(), a.head_dim());
    let s = xs.dim(1)?;
    let xs = xs.to_dtype(DType::F32)?;

    // ── Batched attention projections: ONE GEMM each over all s tokens ──
    // (`pprep:proj` — the stateless GEMMs, the prefill counterpart of decode's
    // `dprep:proj`.)
    let t_proj = profile_now();
    // wq_a and wkv both project `xs`; quantize the activation once and share it.
    let (qa_raw, kv_raw) = shared_int8_pair(&xs, a.wq_a(), a.wkv())?;
    let qr_all = rms_norm(&qa_raw, a.q_norm(), a.eps())?; // [1,s,qa]
    let q_all = a.wq_b().forward(&qr_all)?.reshape((1, s, h, hd))?;
    let q_all = a.rms_scale(&q_all)?;
    let q_bf_all = q_all.reshape((s, h, hd))?.to_dtype(DType::BF16)?; // [s,h,hd]
    let kv_all = rms_norm(&kv_raw, a.kv_norm(), a.eps())?; // [1,s,hd]
    let kv_bf_all = kv_all.reshape((s, hd))?.to_dtype(DType::BF16)?; // [s,hd]

    // ── Batched compressor projections (attention-comp + indexer-comp) ──
    let comp_proj = match seq.comp.as_ref() {
        Some(c) => Some(c.project_rows(&xs)?), // (kv [s,cd], score [s,cd])
        None => None,
    };
    let icomp_proj = match seq.icomp.as_ref() {
        Some(ic) => Some(ic.project_rows(&xs)?),
        None => None,
    };
    pipeline_record("pprep:proj", t_proj);

    // ── pprep:push — build the WHOLE prompt's compressed corpus in ONE batched
    // emit + ONE append, the carried-state-aware batch form of the per-token
    // push loop (bit-identical by `emit_groups_batched_matches_streamed`),
    // instead of `s` per-token pool/append launches. The per-token select below
    // reads a causal PREFIX of it. `comp`/`icomp` share group boundaries, so one
    // emit drives the attn entries (pre-RoPE) and the indexer keys (roped). ──
    let t_push = profile_now();
    let ratio_comp = seq.comp.as_ref().map_or(1, |c| c.ratio());
    let l0 = seq.comp.as_ref().map_or(0, |c| c.buffered_len()); // carried partial group
    let base_entries = seq.gallery.as_ref().map_or(0, |g| g.len()); // corpus before this prefill
    let mut g_total = 0usize;
    if let (Some(comp), Some(gallery), Some((ck, cs))) =
        (seq.comp.as_mut(), seq.gallery.as_mut(), comp_proj.as_ref())
    {
        if let Some((attn_entries, positions)) = comp.emit_groups_projected(ck, cs, None)? {
            g_total = positions.len();
            let key_entries = match (seq.icomp.as_mut(), icomp_proj.as_ref()) {
                (Some(ic), Some((ik, is))) => {
                    let (keys, kpos) = ic
                        .emit_groups_projected(ik, is, Some(rope))?
                        .expect("indexer compressor shares group boundaries");
                    debug_assert_eq!(kpos, positions, "comp/icomp group boundaries diverged");
                    keys // [g_total, index_head_dim]
                }
                _ => Tensor::zeros((g_total, 1), DType::F32, xs.device())?,
            };
            gallery.append_batch(&attn_entries, &key_entries, &positions)?;
        } else if let (Some(ic), Some((ik, is))) = (seq.icomp.as_mut(), icomp_proj.as_ref()) {
            let none = ic.emit_groups_projected(ik, is, Some(rope))?;
            debug_assert!(none.is_none(), "compressor group boundaries diverged");
        }
    }
    pipeline_record("pprep:push", t_push);

    // ── pprep:select — causal select over the up-front corpus. Each token sees
    // the entries present before this prefill plus the groups completed through
    // it (`(l0 + t + 1) / ratio`); bounding the select to that prefix reproduces
    // the per-token incremental gallery exactly. ──
    let t_sel = profile_now();
    let n_visible: Vec<usize> = (0..s)
        .map(|t| base_entries + ((l0 + t + 1) / ratio_comp).min(g_total))
        .collect();
    let idx_rows: Vec<Vec<u32>> = match a.kind() {
        LayerKind::Csa => {
            let ix = a.indexer().expect("CSA layer has an indexer");
            let m = shortlist_m(ix.top_k());
            let max_nv = base_entries + g_total; // widest window (the last token)
            if max_nv == 0 {
                vec![Vec::new(); s]
            } else if max_nv <= m.min(1024) {
                // In-regime: the shortlist covers every token's window, so
                // two-stage recall degenerates to the exact full Indexer top-k —
                // do the whole prompt in one batched query GEMM + one rescore +
                // one argsort (bit-identical to the per-token loop by
                // `batched_causal_select_matches_per_token`). Kills the per-token
                // select launches that dominated `pprep:select`.
                let (q_raw, weights) =
                    ix.query_gemm_batched(&xs.reshape((s, ()))?, &qr_all.reshape((s, ()))?)?;
                let q_idx = ix.rope_query_batched(&q_raw, rope, base)?; // [s,h,ih]
                let gallery = seq.gallery.as_ref().unwrap();
                gallery.batched_causal_select(&q_idx, &weights, &n_visible, ix.top_k())?
            } else {
                // Out-of-regime (corpus wider than the shortlist): the exact
                // batched top-k would diverge from the per-token recall
                // approximation the decode path uses, so keep the per-token
                // recall select to preserve prefill≡decode parity on deep prompts.
                let gallery = seq.gallery.as_ref().unwrap();
                let mut rows = Vec::with_capacity(s);
                for (t, &n_vis) in n_visible.iter().enumerate() {
                    let gids = if n_vis == 0 {
                        Vec::new()
                    } else {
                        let x_t = xs.narrow(1, t, 1)?;
                        let qr_t = qr_all.narrow(1, t, 1)?;
                        let (qi, w) = ix.query_space(&x_t, &qr_t, rope, base + t)?;
                        let (g, k) =
                            gallery.two_stage_select_causal(&qi, &w, m, ix.top_k(), n_vis)?;
                        if k == 0 {
                            Vec::new()
                        } else {
                            g.to_vec1::<u32>()?
                        }
                    };
                    rows.push(gids);
                }
                rows
            }
        }
        LayerKind::Hca => n_visible.iter().map(|&n| (0..n as u32).collect()).collect(),
        LayerKind::SlidingWindow => vec![Vec::new(); s],
    };
    pipeline_record("pprep:select", t_sel);
    Ok((q_bf_all, kv_bf_all, idx_rows))
}

/// What a decode session's corpus selection will be, captured BEFORE the select
/// runs so the wave can batch the (per-session) two-stage selection across the
/// whole decode set in one launch per Stage-1 kernel.
pub enum DecodeSel {
    /// CSA layer: two-stage BDP-recall → Indexer-precision top-k, using this
    /// session's per-head Indexer query `[n_idx_heads, index_head_dim]` and gate
    /// weights `[n_idx_heads]`. The gallery it selects over lives in the session
    /// state the wave holds.
    TwoStage { q_idx: Tensor, weights: Tensor },
    /// HCA layer: attend ALL `n` causal compressed entries.
    AllEntries(usize),
    /// SWA layer or an empty corpus: no compressed selection.
    None,
}

/// Corpus-push + selection-capture for ONE decode session's token, with the
/// attention projections ALREADY batched across sessions by the caller — the
/// counterpart to the batched prefill prep. `x` is the token's raw hidden
/// `[1, 1, dim]` (f32) and `qr` its pre-projected, q-normed low-rank query
/// `[1, 1, q_lora_rank]` (f32); the caller computed both in one GEMM over all
/// decode rows and hands each session its slice. This runs only the STATEFUL
/// per-session work: corpus maintenance (`push_raw`) and capturing the
/// [`DecodeSel`] (the Indexer query space for a CSA layer). The two-stage
/// selection itself is then batched over all sessions'
/// galleries ([`super::gallery::two_stage_select_batched`]).
#[allow(clippy::too_many_arguments)]
pub fn kernel_attn_decode_capture(
    a: &Attention,
    seq: &mut KernelLayerSeqState,
    x: &Tensor,
    comp_row: Option<(&Tensor, &Tensor)>,
    icomp_row: Option<(&Tensor, &Tensor)>,
    q_idx: Option<Tensor>,
    weights: Option<Tensor>,
    rope: &RotaryCache,
) -> Result<DecodeSel> {
    // Corpus maintenance — identical to the decode step, but fed the compressor
    // rows already projected in a batched GEMM by the caller (bit-identical to
    // `push_raw`/`push`, which differ only by the per-row projection).
    let t_push = profile_now();
    if let (Some(comp), Some(gallery), Some((ck, cs))) =
        (seq.comp.as_mut(), seq.gallery.as_mut(), comp_row)
    {
        if let Some((entry, gpos)) = comp.push_projected(ck, cs)? {
            let key = match (seq.icomp.as_mut(), icomp_row) {
                (Some(ic), Some((ik, is))) => ic
                    .push_projected_roped(ik, is, rope)?
                    .expect("indexer compressor shares group boundaries")
                    .reshape((1, ()))?,
                _ => Tensor::zeros((1, 1), DType::F32, x.device())?,
            };
            gallery.append_batch(&entry.reshape((1, a.head_dim()))?, &key, &[gpos])?;
        } else if let (Some(ic), Some((ik, is))) = (seq.icomp.as_mut(), icomp_row) {
            let none = ic.push_projected_roped(ik, is, rope)?;
            debug_assert!(none.is_none(), "compressor group boundaries diverged");
        }
    }
    pipeline_record("dprep:push", t_push);

    // Capture the selection intent (no gallery read here — the select is
    // batched). The CSA query/weights were projected in a batched GEMM by the
    // caller (`Indexer::query_gemm_batched` + per-session `rope_query`); this
    // just tags the intent with the gallery's live size.
    let n_entries = seq.gallery.as_ref().map_or(0, |g| g.len());
    let sel = if n_entries == 0 {
        DecodeSel::None
    } else {
        match a.kind() {
            LayerKind::Csa => DecodeSel::TwoStage {
                q_idx: q_idx.expect("CSA layer supplies a batched query"),
                weights: weights.expect("CSA layer supplies batched gate weights"),
            },
            LayerKind::Hca => DecodeSel::AllEntries(n_entries),
            LayerKind::SlidingWindow => DecodeSel::None,
        }
    };
    Ok(sel)
}
