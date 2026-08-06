//! Per-layer paged-kernel attention state for the engine: the projections
//! (`wq_a`/`wq_b`/`wkv`, norms) stay host-side int8, while the attention math
//! itself — FP8 window walk, compressed top-k walk, sink softmax, RoPE and
//! de-rotation — runs in the `paged-deepseek` decode kernel over the
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
use super::paged::{paged_deepseek_decode_raw, HEAD_DIM, ROPE_DIM};
use super::rope::{yarn_freqs, RotaryCache};

/// Recall shortlist width for the two-stage selection: 2× the Indexer's
/// top-k, capped at the device argsort's 1024-column rescore limit.
pub fn shortlist_m(top_k: usize) -> usize {
    (2 * top_k).clamp(64, 1024)
}

/// Per-layer kernel-attention constants shared by every sequence: sink
/// logits, YaRN frequencies, and the zero-entry placeholders. One per layer,
/// built once with the model.
pub struct KernelLayerStatic {
    /// Per-head sink logits `[n_heads]` f32, device.
    sinks: Tensor,
    /// YaRN inverse frequencies `[ROPE_DIM/2]` f32, device (layer-kind theta).
    freqs: Tensor,
    /// Zero-entry placeholders so the kernel always gets valid pointers.
    empty_comp: Tensor,
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
    pub fn freqs(&self) -> &Tensor {
        &self.freqs
    }
    pub fn empty_comp(&self) -> Tensor {
        self.empty_comp.clone()
    }
    pub fn empty_pos(&self) -> Tensor {
        self.empty_pos.clone()
    }

    pub fn new(
        a: &Attention,
        theta: f64,
        original_seq_len: usize,
        rope_factor: f64,
        beta_fast: f64,
        beta_slow: f64,
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
        Ok(Self {
            sinks,
            freqs,
            empty_comp: Tensor::zeros((1, HEAD_DIM), DType::F32, device)?,
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
        kernel_attn_decode_step(a, &self.st, &mut self.seq, x, rope, pos, headers_ptr)
    }
}

/// One decode step through the paged kernel — the shared building block for
/// the engine session (one sequence) and the batched wave model (many).
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
    let (comp, comp_pos, comp_idx, comp_cnt) = if n_entries == 0 {
        (
            st.empty_comp.clone(),
            st.empty_pos.clone(),
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
                st.empty_comp.clone(),
                st.empty_pos.clone(),
                st.empty_idx.clone(),
                st.empty_cnt.clone(),
            )
        } else {
            // Gather the k selected entries into a COMPACTED GPU pair (tier-
            // aware: from CPU RAM when the corpus has spilled) and walk them
            // densely — `comp_idx = 0..k`. The kernel never touches the full
            // corpus, only the bounded selection, so the resident/gathered
            // working set stays O(1) at any depth.
            let (comp, comp_pos) = gallery.gather_selected(&gids)?;
            let comp_idx = Tensor::arange(0u32, k as u32, x.device())?.reshape((1, k))?;
            let cnt = Tensor::from_vec(vec![k as u32], 1, x.device())?;
            (comp, comp_pos, comp_idx, cnt)
        }
    };

    // The kernel: hybrid window+compressed attention, sink fold,
    // de-rotation — returns `[1, n_heads, 512]` bf16.
    let out = paged_deepseek_decode_raw(
        &q_bf,
        headers_ptr,
        &kv_bf,
        &comp,
        &comp_pos,
        &comp_idx,
        &comp_cnt,
        &st.sinks,
        &st.freqs,
        a.softmax_scale() as f32,
        a.window_size(),
        0,     // auto split factor
        false, // wave hands a private per-token snapshot with the write-len
        // already patched host-side — the on-device commit would only touch a
        // throwaway copy, so skip it.
        None,
    )?;

    let o = out.to_dtype(DType::F32)?.reshape((1, h, 1, hd))?;
    a.output_proj(&o, 1, 1)
}
