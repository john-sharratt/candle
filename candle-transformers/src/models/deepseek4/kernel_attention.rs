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

use super::attention::{rms_norm, rms_scale, Attention};
use super::compressor::IncrementalCompressor;
use super::config::LayerKind;
use super::gallery::FloatGallery;
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
    let q = rms_scale(&q, a.eps())?;
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
        (st.empty_corpus_cache()?, st.empty_idx.clone(), st.empty_cnt.clone())
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
            (st.empty_corpus_cache()?, st.empty_idx.clone(), st.empty_cnt.clone())
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

/// Batched-prefill preparation for ONE prompt token: the SAME host projections
/// and corpus-push + causal `two_stage_select` as [`kernel_attn_decode_step`],
/// but it does NOT launch the attention — it returns the token's pre-RoPE bf16
/// query `[1, n_heads, HEAD_DIM]`, its pre-RoPE bf16 latent `[1, HEAD_DIM]` (the
/// batched `kv_fresh` diagonal source), and its absolute selected compressed
/// GIDs. Looping this in token order builds the corpus causally, so token `t`
/// selects only over entries completed by `t` — bit-identical selection to the
/// per-token decode path, launched once for the whole prompt.
pub fn kernel_attn_prefill_prepare(
    a: &Attention,
    seq: &mut KernelLayerSeqState,
    x: &Tensor,
    rope: &RotaryCache,
    pos: usize,
) -> Result<(Tensor, Tensor, Vec<u32>)> {
    let (h, hd) = (a.n_heads(), a.head_dim());
    let din = x.elem_count();
    let x = x.reshape((1, 1, din))?.to_dtype(DType::F32)?;

    let qr = rms_norm(&a.wq_a().forward(&x)?, a.q_norm(), a.eps())?;
    let q = a.wq_b().forward(&qr)?.reshape((1, 1, h, hd))?;
    let q = rms_scale(&q, a.eps())?;
    let q_bf = q.reshape((1, h, hd))?.to_dtype(DType::BF16)?;
    let kv = rms_norm(&a.wkv().forward(&x)?, a.kv_norm(), a.eps())?;
    let kv_bf = kv.reshape((1, hd))?.to_dtype(DType::BF16)?;

    // Corpus maintenance — identical to the decode step.
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

    // Absolute selected GIDs (the per-query causal top-k), read to host so the
    // wave can pack them into the `[s, max_sel]` selection the batched prefill
    // launch expects (prefill is amortized — this is not the decode hot path).
    let n_entries = seq.gallery.as_ref().map_or(0, |g| g.len());
    let gids: Vec<u32> = if n_entries == 0 {
        Vec::new()
    } else {
        let gallery = seq.gallery.as_ref().unwrap();
        match a.kind() {
            LayerKind::Csa => {
                let ix = a.indexer().expect("CSA layer has an indexer");
                let (qi, w) = ix.query_space(&x, &qr, rope, pos)?;
                let (g, k) =
                    gallery.two_stage_select(&qi, &w, shortlist_m(ix.top_k()), ix.top_k())?;
                if k == 0 {
                    Vec::new()
                } else {
                    g.to_vec1::<u32>()?
                }
            }
            LayerKind::Hca => (0..n_entries as u32).collect(),
            LayerKind::SlidingWindow => Vec::new(),
        }
    };
    Ok((q_bf, kv_bf, gids))
}

/// Batched-decode preparation for ONE sequence's token: the SAME host
/// projections + corpus-push + on-device select/gather as
/// [`kernel_attn_decode_step`], but WITHOUT the attention launch. Returns the
/// token's pre-RoPE bf16 query `[1, n_heads, HEAD_DIM]`, its pre-RoPE bf16
/// latent `[1, HEAD_DIM]` (the fused-scatter source), and the ON-DEVICE gathered
/// compressed block `[k, HEAD_DIM]` + positions `[k]` + count `k`. Selection and
/// gather stay on-device (no GID readback), so the wave can concatenate these
/// across sessions and attend ALL decode slots in ONE `paged_latent_decode`
/// launch — the per-slot selection is offset into the concatenated block.
#[allow(clippy::type_complexity)]
pub fn kernel_attn_decode_prepare(
    a: &Attention,
    seq: &mut KernelLayerSeqState,
    x: &Tensor,
    rope: &RotaryCache,
    pos: usize,
) -> Result<(Tensor, Tensor, Option<(Tensor, Tensor, Tensor, Tensor)>, usize)> {
    let (h, hd) = (a.n_heads(), a.head_dim());
    let din = x.elem_count();
    let x = x.reshape((1, 1, din))?.to_dtype(DType::F32)?;

    let qr = rms_norm(&a.wq_a().forward(&x)?, a.q_norm(), a.eps())?;
    let q = a.wq_b().forward(&qr)?.reshape((1, 1, h, hd))?;
    let q = rms_scale(&q, a.eps())?;
    let q_bf = q.reshape((1, h, hd))?.to_dtype(DType::BF16)?;
    let kv = rms_norm(&a.wkv().forward(&x)?, a.kv_norm(), a.eps())?;
    let kv_bf = kv.reshape((1, hd))?.to_dtype(DType::BF16)?;

    // Corpus maintenance — identical to the decode step.
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

    // Selection → on-device gather (compacted block); no host readback.
    let n_entries = seq.gallery.as_ref().map_or(0, |g| g.len());
    if n_entries == 0 {
        return Ok((q_bf, kv_bf, None, 0));
    }
    let gallery = seq.gallery.as_ref().unwrap();
    let (gids, k) = match a.kind() {
        LayerKind::Csa => {
            let ix = a.indexer().expect("CSA layer has an indexer");
            let (qi, w) = ix.query_space(&x, &qr, rope, pos)?;
            gallery.two_stage_select(&qi, &w, shortlist_m(ix.top_k()), ix.top_k())?
        }
        LayerKind::Hca => (
            Tensor::arange(0u32, n_entries as u32, x.device())?,
            n_entries,
        ),
        LayerKind::SlidingWindow => return Ok((q_bf, kv_bf, None, 0)),
    };
    if k == 0 {
        return Ok((q_bf, kv_bf, None, 0));
    }
    // The HOT two-region cache for the selection (gallery pre-built on append);
    // the wave concatenates these across sessions into one launch.
    let (ni8, nsc, rbf, cpos) = gallery.gather_corpus(&gids)?;
    Ok((q_bf, kv_bf, Some((ni8, nsc, rbf, cpos)), k))
}
