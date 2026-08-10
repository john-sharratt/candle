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
use super::compressor::{GroupPool, IncrementalCompressor};
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

    // Kernel output is F32, token-major [1, h, hd]; `output_proj` takes [b,s,h,hd].
    let o = out.reshape((1, 1, h, hd))?;
    a.output_proj(&o, 1, 1)
}

/// A prefill wave's corpus selection, produced by
/// [`kernel_attn_prefill_select`].
pub enum PrefillSel {
    /// Fully on-device (in-regime CSA, or the empty-corpus case): `comp_idx`
    /// `[s, kmax]` holds each token's selected **absolute** entry ids ascending
    /// (`u32::MAX`-padded past `comp_cnt`), `comp_cnt` `[s]` the per-token counts.
    /// The ids index a gather of the whole visible corpus `0..n_corpus`, so the
    /// caller gathers that range and uses `comp_idx`/`comp_cnt` directly — no
    /// host readback, union, or remap.
    Device {
        comp_idx: Tensor,
        comp_cnt: Tensor,
        n_corpus: usize,
    },
    /// Host per-token absolute GIDs (out-of-regime CSA recall, HCA attend-all,
    /// SWA none) — the caller unions, gathers, and remaps them on the host.
    Host(Vec<Vec<u32>>),
}

/// The WHOLE prefill span's stateless projections, computed ONCE over every
/// prompt sequence's rows ([`kernel_attn_prefill_project_batched`]). Every
/// projection is row-independent, so this is bit-identical to projecting each
/// sequence separately — the same batching decode's `dprep` already does — but
/// one GEMM/quantize instead of one per sequence. [`kernel_attn_prefill_assemble`]
/// slices each sequence's rows back out.
pub struct PrefillProj {
    /// Pre-RoPE bf16 query `[total, n_heads, HEAD_DIM]`.
    pub q_bf: Tensor,
    /// Pre-RoPE bf16 latent `[total, HEAD_DIM]` (also the arena writeback source).
    pub kv_bf: Tensor,
    /// q-normed low-rank query `[1, total, q_lora_rank]` — the indexer query source.
    pub qr_all: Tensor,
    /// F32 hidden `[1, total, dim]` (a view of `x`) — the indexer `query_space` source.
    pub xs: Tensor,
    /// Attention compressor rows `(kv, score)` `[total, cd]` (`None` on SWA layers).
    pub comp_proj: Option<(Tensor, Tensor)>,
    /// Indexer compressor rows `(kv, score)` `[total, cd]` (`None` when no indexer).
    pub icomp_proj: Option<(Tensor, Tensor)>,
}

/// The projected + assembled state for ONE prefill sequence
/// ([`kernel_attn_prefill_assemble`]), consumed by the wave's batched pool +
/// [`kernel_attn_prefill_select`]. `kv_bf`/`qr_all`/`xs` are slices (views)
/// of the batched [`PrefillProj`]. (The query goes to the batched kernel straight
/// from `PrefillProj.q_bf`, so no per-seq query slice is kept here.)
pub struct PrefillPrep {
    /// Pre-RoPE bf16 latent slice `[s, HEAD_DIM]` (the arena writeback source).
    pub kv_bf: Tensor,
    /// q-normed low-rank query slice `[1, s, q_lora_rank]` — the indexer query source.
    pub qr_all: Tensor,
    /// F32 hidden slice `[1, s, dim]` — the indexer `query_space` source (out-of-regime).
    pub xs: Tensor,
    /// Attention compressor's completed group pool (deferred; `None` if no group
    /// completed or the layer has no compressor).
    pub comp_gp: Option<GroupPool>,
    /// Indexer compressor's completed group pool (shares boundaries with `comp_gp`).
    pub icomp_gp: Option<GroupPool>,
    /// Per-token causally-visible entry counts, for the select.
    pub n_visible: Vec<usize>,
    /// Corpus entries present BEFORE this prefill (the select's absolute index base).
    pub base_entries: usize,
    /// Groups this prefill completes (`== comp_gp.positions.len()`, 0 if none).
    pub g_total: usize,
}

/// Batched prefill projections over the WHOLE prompt span `xs` `[1, total, dim]`
/// (all sequences' rows concatenated) — one `shared_int8_pair`/`wq_b`/norms for
/// the attention query+latent and one `project_rows` for each streaming
/// compressor, instead of one set per sequence. Uses the layer's shared compressor
/// weights (`a.compressor()`/`a.indexer().compressor()`); the per-sequence
/// streaming STATE lives in [`kernel_attn_prefill_assemble`]. Row-independent ⇒
/// bit-identical to the former per-seq projection.
pub fn kernel_attn_prefill_project_batched(a: &Attention, xs: &Tensor) -> Result<PrefillProj> {
    let (h, hd) = (a.n_heads(), a.head_dim());
    let s = xs.dim(1)?;
    let xs = xs.to_dtype(DType::F32)?; // no-op: hc.pre already yields F32 (a view of x)

    // wq_a and wkv both project `xs`; quantize the activation once and share it.
    let (qa_raw, kv_raw) = shared_int8_pair(&xs, a.wq_a(), a.wkv())?;
    let qr_all = rms_norm(&qa_raw, a.q_norm(), a.eps())?; // [1,total,qa]
    let q_all = a.wq_b().forward(&qr_all)?.reshape((1, s, h, hd))?;
    let q_all = a.rms_scale(&q_all)?;
    let q_bf = q_all.reshape((s, h, hd))?.to_dtype(DType::BF16)?; // [total,h,hd]
    let kv_all = rms_norm(&kv_raw, a.kv_norm(), a.eps())?; // [1,total,hd]
    let kv_bf = kv_all.reshape((s, hd))?.to_dtype(DType::BF16)?; // [total,hd]

    let comp_proj = match a.compressor() {
        Some(c) => Some(c.project_rows(&xs)?), // (kv [total,cd], score [total,cd])
        None => None,
    };
    let icomp_proj = match a.indexer() {
        Some(ix) => Some(ix.compressor().project_rows(&xs)?),
        None => None,
    };

    Ok(PrefillProj {
        q_bf,
        kv_bf,
        qr_all,
        xs,
        comp_proj,
        icomp_proj,
    })
}

/// Slice ONE sequence's rows `[off, off+s)` out of the batched [`PrefillProj`] and
/// run its stateful compressor ASSEMBLE (state advance + deferred group pool),
/// WITHOUT pooling — the wave pools every sequence together in one launch
/// ([`Compressor::pool_and_norm`]) before appending + selecting per sequence.
/// `comp`/`icomp` share group boundaries, so both assemble in lockstep; the state
/// advances every call (partial rows buffered) even when no group completes.
/// Bit-identical corpus to the streamed per-token push
/// (`emit_groups_batched_matches_streamed`).
pub fn kernel_attn_prefill_assemble(
    seq: &mut KernelLayerSeqState,
    proj: &PrefillProj,
    off: usize,
    s: usize,
) -> Result<PrefillPrep> {
    // This sequence's rows (views of the batched projections). The query is not
    // sliced — the batched kernel consumes `PrefillProj.q_bf` whole.
    let kv_bf = proj.kv_bf.narrow(0, off, s)?;
    let qr_all = proj.qr_all.narrow(1, off, s)?;
    let xs = proj.xs.narrow(1, off, s)?;
    let comp_slice = match &proj.comp_proj {
        Some((k, sc)) => Some((k.narrow(0, off, s)?, sc.narrow(0, off, s)?)),
        None => None,
    };
    let icomp_slice = match &proj.icomp_proj {
        Some((k, sc)) => Some((k.narrow(0, off, s)?, sc.narrow(0, off, s)?)),
        None => None,
    };

    let t_asm = profile_now();
    let ratio_comp = seq.comp.as_ref().map_or(1, |c| c.ratio());
    let l0 = seq.comp.as_ref().map_or(0, |c| c.buffered_len()); // carried partial group
    let base_entries = seq.gallery.as_ref().map_or(0, |g| g.len()); // corpus before this prefill
    let comp_gp = match (seq.comp.as_mut(), comp_slice.as_ref()) {
        (Some(comp), Some((ck, cs))) => comp.assemble_groups(ck, cs)?,
        _ => None,
    };
    let icomp_gp = match (seq.icomp.as_mut(), icomp_slice.as_ref()) {
        (Some(ic), Some((ik, is))) => ic.assemble_groups(ik, is)?,
        _ => None,
    };
    debug_assert_eq!(
        comp_gp.as_ref().map(|g| g.positions.clone()),
        icomp_gp.as_ref().map(|g| g.positions.clone()),
        "comp/icomp group boundaries diverged"
    );
    let g_total = comp_gp.as_ref().map_or(0, |g| g.positions.len());
    pipeline_record("pprep:assemble", t_asm);

    // Each token sees the entries present before this prefill plus the groups
    // completed through it (`(l0 + t + 1) / ratio`); bounding the select to that
    // prefix reproduces the per-token incremental gallery exactly.
    let n_visible: Vec<usize> = (0..s)
        .map(|t| base_entries + ((l0 + t + 1) / ratio_comp).min(g_total))
        .collect();

    Ok(PrefillPrep {
        kv_bf,
        qr_all,
        xs,
        comp_gp,
        icomp_gp,
        n_visible,
        base_entries,
        g_total,
    })
}

/// Whole-prompt corpus SELECT for one prefill sequence, run AFTER its pooled
/// entries are appended to `gallery`. Returns the [`PrefillSel`] (fully on-device
/// for the in-regime CSA path; host per-token GIDs otherwise). Split out of the
/// prep so the compressor pool + gallery append can batch across all prompt
/// sequences in between (the pool is the prefill hot spot).
pub fn kernel_attn_prefill_select(
    a: &Attention,
    gallery: Option<&FloatGallery>,
    prep: &PrefillPrep,
    rope: &RotaryCache,
    base: usize,
) -> Result<PrefillSel> {
    let t_sel = profile_now();
    let s = prep.xs.dim(1)?;
    let n_visible = &prep.n_visible;
    let (base_entries, g_total) = (prep.base_entries, prep.g_total);
    let sel: PrefillSel = match a.kind() {
        LayerKind::Csa => {
            let ix = a.indexer().expect("CSA layer has an indexer");
            let m = shortlist_m(ix.top_k());
            let max_nv = base_entries + g_total; // widest window (the last token)
            if max_nv == 0 {
                PrefillSel::Host(vec![Vec::new(); s])
            } else if max_nv <= m.min(1024) {
                // In-regime: the shortlist covers every token's window, so
                // two-stage recall degenerates to the exact full Indexer top-k —
                // one batched query GEMM + one rescore + one argsort, kept FULLY
                // ON-DEVICE (bit-identical selection to the per-token loop by
                // `batched_causal_select_matches_per_token`).
                let t_q = profile_now();
                let (q_raw, weights) = ix.query_gemm_batched(
                    &prep.xs.reshape((s, ()))?,
                    &prep.qr_all.reshape((s, ()))?,
                )?;
                let q_idx = ix.rope_query_batched(&q_raw, rope, base)?; // [s,h,ih]
                pipeline_record("psel:query", t_q);
                let gallery = gallery.expect("CSA layer has a gallery");
                let t_bcs = profile_now();
                let (comp_idx, comp_cnt, n_corpus) = gallery.batched_causal_select_device(
                    &q_idx,
                    &weights,
                    n_visible,
                    ix.top_k(),
                )?;
                pipeline_record("psel:bcs", t_bcs);
                PrefillSel::Device {
                    comp_idx,
                    comp_cnt,
                    n_corpus,
                }
            } else {
                // Out-of-regime (corpus wider than the shortlist): keep the
                // per-token recall select to preserve prefill≡decode parity on
                // deep prompts.
                let gallery = gallery.expect("CSA layer has a gallery");
                let mut rows = Vec::with_capacity(s);
                for (t, &n_vis) in n_visible.iter().enumerate() {
                    let gids = if n_vis == 0 {
                        Vec::new()
                    } else {
                        let x_t = prep.xs.narrow(1, t, 1)?;
                        let qr_t = prep.qr_all.narrow(1, t, 1)?;
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
                PrefillSel::Host(rows)
            }
        }
        LayerKind::Hca => {
            PrefillSel::Host(n_visible.iter().map(|&n| (0..n as u32).collect()).collect())
        }
        LayerKind::SlidingWindow => PrefillSel::Host(vec![Vec::new(); s]),
    };
    pipeline_record("pprep:select", t_sel);
    Ok(sel)
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
