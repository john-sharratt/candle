//! The hybrid wave: one forward, two kinds of layer.
//!
//! This is [`WaveSweep`] for the hybrid — the *only* half of a wave that a
//! Gated-DeltaNet stack does differently. Everything around it (bounding a
//! forward's token count, routing 1-token prefills to the decode kernel,
//! assembling the groups, permuting tokens between caller and internal order,
//! rolling KV back on failure, advancing the decode rows) is
//! [`crate::models::wave_driver`], shared with every other model.
//!
//! Four things separate the sweep here from a uniform transformer's:
//!
//! * **Caches are indexed by KV layer.** A DeltaNet layer owns no paged KV, so
//!   the session allocates one cache per *attention* layer and
//!   [`KvLayerMap`](crate::models::delta_net::KvLayerMap) is the only thing allowed to
//!   translate. That applies to the admit phase's layer range too — admitting
//!   over trunk indices would claim four times the chunks the wave writes and
//!   walk off the end of the cache vector.
//! * **Recurrent state is per sequence and advances in place.** The mixer
//!   carries `S` and a conv tail across a sequence's tokens, so the state is
//!   lifted out of the model's map for the sweep and every wave that begins
//!   must commit or roll back — "did not commit" is not "did not happen".
//! * **The rotary table is partial.** Only `rope_dim` of `head_dim` dims
//!   rotate, so `rope_cs` comes from [`RotaryLayout`] rather than the uniform
//!   `compute_rope_cs`, which would rotate the whole head.
//! * **Glue is refused.** The gap-fill kernel is compiled for `head_dim 128`
//!   and this family attends at 256, so a wave carrying glue rows fails at the
//!   top rather than silently running them as ordinary prefill against the
//!   wrong mask.

use std::cell::RefCell;

use candle::{DType, Device, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{
    begin_forward, begin_wave, end_wave_transient, plan_wave_transient, LayerPhase, WavePlan,
    REGION_BYTES, WAVE_FORWARD_BYTES,
};
use candle_nn::kv_cache::KvCache;

use super::batched::HybridBatched;
use super::quantized_attention::Qwen35AttentionLayer;
use super::quantized_delta_net::quantized_delta_net_ffn;
use super::quantized_weights::{QuantLayerMix, QuantModel};
use crate::models::delta_net::RecurrentStateStore;
use super::wave::delta_net_mix_wave;
use crate::models::delta_net::seq_spans;
use candle_nn::kv_cache::ModelGeometry;

use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ManagedBatchedModel, ModelCoreProperties, WaveResult,
};
use crate::models::batched_layer::{
    forward_layer_batched_mixed, BatchedAttentionParams, WaveAttnGroup,
};
use crate::models::expert_lre::{PipelineStats, ProfileSnapshot};
use crate::models::batched_model::{activation_dtype, WaveGuard, WavePhase};
use crate::models::kv_cache_utils::SequenceContext;
use crate::models::prefill_utils::SharedPm;
use crate::models::tensor_cat::TensorCat;
use crate::models::wave_admit::admit_wave_kv;
use crate::models::wave_buffers::wave_root;
use crate::models::wave_driver::{drive_wave, WaveGroups, WaveSweep};

/// The hybrid as the scheduler drives it.
///
/// Every method here either answers from the hybrid's own geometry (depth is
/// the trunk, KV heads are the attention layers') or hands the question to the
/// shared machinery — the wave driver, the expert cache. `forward_wave` is the
/// second kind: the sweep above is the model-specific half, and everything
/// around it is [`drive_wave`].
impl ManagedBatchedModel for HybridBatched {
    fn wave_geometry(&self, act_dtype: DType) -> ModelGeometry {
        HybridBatched::wave_geometry(self, act_dtype)
    }

    /// Rows the KV side can admit, priced against **attention** layers.
    ///
    /// The trait's default multiplies the per-row cost by trunk depth, which on
    /// a 3:1 hybrid over-charges by 4× and refuses four times more prefill than
    /// the cache can hold.
    fn kv_width_cap(&self, act_dtype: DType) -> Option<usize> {
        HybridBatched::kv_width_cap(self, act_dtype)
    }

    /// Weight ground the elastic boundary would cede to a stuck KV claim.
    ///
    /// Counted by `kv_width_cap` when sizing a prefill wave, so the wave is
    /// sliced against what the partition *can* admit rather than against what
    /// happens to stand free — on a streamed-expert model those differ by tens
    /// of gigabytes, and the difference is the whole ingest sweep.
    fn reclaimable_kv_bytes(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.model()
                .experts
                .as_ref()
                .map_or(0, |cache| cache.cedeable_span_bytes())
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    fn maybe_change_dtype(&self, dtype: DType) -> Result<()> {
        HybridBatched::maybe_change_dtype(self, dtype)
    }

    fn num_layers(&self) -> usize {
        HybridBatched::num_layers(self)
    }

    fn n_kv_head(&self) -> usize {
        HybridBatched::n_kv_head(self)
    }

    fn head_dim(&self) -> usize {
        HybridBatched::head_dim(self)
    }

    fn device(&self) -> &Device {
        HybridBatched::device(self)
    }

    fn model_core_properties(&self) -> ModelCoreProperties {
        HybridBatched::model_core_properties(self)
    }

    fn create_batched_session(&self, config: BatchedConfig) -> Result<BatchedInferenceSession> {
        HybridBatched::create_batched_session(self, config)
    }

    fn forward_wave(
        &self,
        session: &mut BatchedInferenceSession,
        decode_seqs: &[usize],
        decode_inputs: &[Tensor],
        prefill_seqs: &[usize],
        prefill_inputs: &[Tensor],
        glue_seqs: &[usize],
        glue_inputs: &[Tensor],
        layer_start: usize,
        layer_end: usize,
        residual_in: Option<Tensor>,
    ) -> Result<WaveResult> {
        drive_wave(
            self,
            session,
            decode_seqs,
            decode_inputs,
            prefill_seqs,
            prefill_inputs,
            glue_seqs,
            glue_inputs,
            layer_start,
            layer_end,
            residual_in,
        )
    }

    /// A sequence ends: its recurrent state must go with its KV slots, or the
    /// map grows for the life of the process.
    ///
    /// **Truncation to a non-zero offset is refused.** A recurrent state
    /// cannot be truncated — `S` is a running sum with no per-token
    /// decomposition, so there is no suffix to remove — and KV rewound under
    /// a state that still holds the un-truncated history is silent
    /// corruption: the model answers as though it remembers tokens the cache
    /// no longer has (measured: re-prefilling a truncated prompt diverges by
    /// ~9.5 in the logits). Truncating to zero is the one case with an
    /// answer — the state returns to its sequence-start value, which is what
    /// a fresh store holds. Partial truncation (speculative rejection,
    /// scheduler forks) needs recurrent-state checkpoints at the rewind
    /// offsets; until those exist, refusing loudly here is the difference
    /// between an error and a model that quietly hallucinates its history.
    fn truncate_sequence(
        &self,
        session: &mut BatchedInferenceSession,
        seq: usize,
        tokens: usize,
    ) -> Result<()> {
        if tokens != 0 {
            candle::bail!(
                "qwen35: cannot truncate sequence {seq} to {tokens} tokens — the \
                 DeltaNet recurrent state has no per-token decomposition to rewind \
                 to a non-zero offset. Truncate to 0 (full reset) or keep the \
                 sequence intact; partial rewind needs recurrent-state checkpoints."
            );
        }
        session.truncate_sequence_to_tokens(seq, tokens)?;
        self.release_recurrent(seq)?;
        Ok(())
    }

    fn prune(&self) -> Result<()> {
        Ok(())
    }

    fn expert_stats(&self) -> Option<PipelineStats> {
        #[cfg(feature = "cuda")]
        {
            self.model().experts.as_ref().map(|c| c.expert_stats())
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    fn request_kv_ground(&self, regions: usize) -> u64 {
        #[cfg(feature = "cuda")]
        {
            self.model()
                .experts
                .as_ref()
                .map_or(0, |c| c.request_kv_ground(regions))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = regions;
            0
        }
    }

    fn resident_weight_bytes(&self) -> Option<usize> {
        #[cfg(feature = "cuda")]
        {
            self.model()
                .experts
                .as_ref()
                .map(|c| c.resident_vram_bytes())
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    fn reset_expert_stats(&self) {
        #[cfg(feature = "cuda")]
        if let Some(c) = self.model().experts.as_ref() {
            c.reset_expert_stats();
        }
    }

    fn snapshot_profiles(&self) -> ProfileSnapshot {
        #[cfg(feature = "cuda")]
        {
            self.model()
                .experts
                .as_ref()
                .map_or_else(ProfileSnapshot::default, |c| c.snapshot_profiles())
        }
        #[cfg(not(feature = "cuda"))]
        {
            ProfileSnapshot::default()
        }
    }
}

impl WaveSweep for HybridBatched {
    fn device(&self) -> &Device {
        HybridBatched::device(self)
    }

    fn num_layers(&self) -> usize {
        HybridBatched::num_layers(self)
    }

    fn prefill_width_cap(&self, act_dtype: DType) -> usize {
        <Self as ManagedBatchedModel>::prefill_width_cap(self, act_dtype)
    }

    fn kv_layer_range(&self, layer_start: usize, layer_end: usize) -> (usize, usize) {
        self.kv_map().kv_range(layer_start, layer_end)
    }

    /// Open the wave's recurrent state, sweep, then commit or roll back.
    ///
    /// The state is lifted out of the model's map for the duration and put back
    /// unconditionally: a sweep that failed with the stores still lifted would
    /// leave those sequences stateless for every later wave, which is a worse
    /// failure than the one that caused it.
    fn sweep(
        &self,
        contexts: &mut [SequenceContext],
        groups: WaveGroups<'_>,
    ) -> Result<(WavePhase, Option<WaveGuard>)> {
        let seqs = groups.seq_ids.to_vec();
        // Offsets in context order, which is the order `seq_ids` is in — a
        // sequence standing at zero gets its recurrent state reset, not just
        // created (see `ensure_recurrent`).
        let offsets: Vec<usize> = contexts.iter().map(|c| c.offset).collect();
        self.ensure_recurrent(&seqs, &offsets)?;
        self.begin_recurrent_wave(&seqs)?;
        let mut stores = match self.take_recurrent(&seqs) {
            Ok(s) => s,
            Err(e) => {
                // Nothing was lifted, so there is nothing to put back — but the
                // wave IS open and must be closed or the next one is refused.
                let _ = self.rollback_recurrent_wave(&seqs);
                return Err(e);
            }
        };

        let swept = {
            let mut refs: Vec<&mut RecurrentStateStore> = stores.iter_mut().collect();
            sweep_layers(self, contexts, groups, &mut refs)
        };

        self.put_recurrent(&seqs, stores)?;
        match &swept {
            Ok(_) => self.commit_recurrent_wave(&seqs)?,
            Err(_) => self.rollback_recurrent_wave(&seqs)?,
        }
        swept
    }
}

/// The layer sweep proper.
fn sweep_layers(
    model: &HybridBatched,
    contexts: &mut [SequenceContext],
    groups: WaveGroups<'_>,
    stores: &mut [&mut RecurrentStateStore],
) -> Result<(WavePhase, Option<WaveGuard>)> {
    let WaveGroups {
        n_decode,
        n_prefill,
        seq_ids,
        decode_headers,
        prefill_headers,
        glue_headers,
        generation,
        layer_start,
        layer_end,
        x_in,
    } = groups;
    // Refused below, before it can be read — named here so the destructuring
    // stays exhaustive and a new group cannot be added without this seeing it.
    drop(glue_headers);
    if contexts.is_empty() {
        candle::bail!("qwen35 wave: empty batch");
    }
    let q = model.model();
    let num_layers = q.cfg.num_layers;
    if layer_start > layer_end || layer_end > num_layers {
        candle::bail!("qwen35 wave: bad layer range [{layer_start}, {layer_end}) over {num_layers} layers");
    }
    let n_glue = contexts
        .len()
        .checked_sub(n_decode + n_prefill)
        .ok_or_else(|| candle::Error::Msg("qwen35 wave: group bounds exceed batch".into()))?;
    // The gap-fill kernel is `head_dim 128` only and the float prefill fallback
    // carries no glue masking, so a glue row would be attended as an ordinary
    // prefill token — a wrong answer, not a slow one.
    if n_glue > 0 {
        candle::bail!(
            "qwen35 wave: {n_glue} glue rows — reprojection glue is not implemented at \
             head_dim {}; this stack must recompute rather than gap-fill",
            q.cfg.attn_head_dim
        );
    }

    let offsets: Vec<usize> = contexts.iter().map(|c| c.offset).collect();
    let q_lens: Vec<usize> = contexts.iter().map(|c| c.input_len).collect();
    let (dec_off, pre_off) = offsets.split_at(n_decode);
    let (dec_q, pre_q) = q_lens.split_at(n_decode);
    let pre_rows: usize = pre_q.iter().sum();
    let total_rows = n_decode + pre_rows;

    let cache_dtype = contexts
        .first()
        .map(|c| c.kv_caches.dtype())
        .unwrap_or(DType::F32);
    let embed_dtype = activation_dtype(cache_dtype);
    let dev = model.device();

    // **Phase 0: hand back the previous forward's tier**, then let the elastic
    // boundary grow in the one gap it is legal in — every guard from the last
    // forward is dropped and this one has opened none. Admit runs next and
    // claims against a pool the old tier would otherwise still be capping.
    #[cfg(feature = "cuda")]
    if let Device::Cuda(d) = dev {
        end_wave_transient(&d.cuda_stream());
        model.reclaim_spare_ground();
    }

    // **Phase 1: admit** — claim every KV chunk this wave will write before a
    // byte of it computes, so the arena frontier is final when the transient
    // tier is placed against it. Over the **KV** range: three quarters of the
    // trunk range owns no cache at all.
    let (kv_start, kv_end) = model.kv_map().kv_range(layer_start, layer_end);
    admit_wave_kv(contexts, n_decode, n_prefill, kv_start, kv_end)?;

    // **Phase 2: price and reserve this wave's transient tier**, sized to this
    // wave rather than to the widest one the engine can run.
    #[cfg(feature = "cuda")]
    if total_rows > 0 {
        if let Device::Cuda(d) = dev {
            let plan = WavePlan::new(model.wave_geometry(embed_dtype));
            let pad = |b: usize| b + REGION_BYTES;
            let per_phase = [
                pad(plan.phase_bytes(LayerPhase::Attention, total_rows)),
                pad(plan.phase_bytes(LayerPhase::Ffn, total_rows)),
                WAVE_FORWARD_BYTES,
            ];
            plan_wave_transient(&d.cuda_stream(), per_phase)?;
        }
    }

    // From here to the end of this function the forward owns the partition —
    // after phase 2 in both directions: admit creates arenas through the same
    // gate the sealing thread uses (opening earlier had the forward waiting on
    // itself), and the tier placement may buy ground from the weight side,
    // whose `set_weight_floor` refuses while a forward is open.
    #[cfg(feature = "cuda")]
    let _forward_open = match dev {
        Device::Cuda(d) => Some(begin_forward(&d.cuda_stream())),
        _ => None,
    };

    // Combined residual: embed every row flat `[1, total, hidden]`, or resume a
    // paused wave from its persisted stream.
    let mut x = match x_in {
        Some(resume) => resume,
        None => {
            let ids: Vec<Tensor> = contexts.iter().map(|c| c.input_ids.clone()).collect();
            let packed = TensorCat::from_tensors(1, ids.into_iter())?;
            TensorCat::from_cat_tensor(embed_rows(q, &packed.to_tensor(), embed_dtype)?, 0)?
        }
    };

    // The interleaved `(cos, sin)` table the paged kernels index by position.
    // Partial rotary, so it is the model's own table — `compute_rope_cs` would
    // rotate all 256 dims where only 64 turn.
    let max_blocks = contexts
        .first()
        .and_then(|c| c.kv_caches.caches.first().map(|k| k.k_cache().chunked_max_blocks()))
        .unwrap_or(0);
    let rope_cs = model.rope_cs(max_blocks)?;

    // Per-group split cos/sin over this wave's own positions.
    let theta = q.cfg.rope_theta as f32;
    let rot = model.rotary();
    let dec_pos: Vec<u32> = dec_off.iter().map(|&o| o as u32).collect();
    let mut pre_pos: Vec<u32> = Vec::with_capacity(pre_rows);
    for (&o, &l) in pre_off.iter().zip(pre_q) {
        for i in 0..l {
            pre_pos.push((o + i) as u32);
        }
    }
    let rope_dtype = if embed_dtype == DType::F8E4M3 {
        DType::BF16
    } else {
        embed_dtype
    };
    let dec_rope = rot.rope_cos_sin(&dec_pos, theta, rope_dtype, dev)?;
    let (pre_cos, pre_sin) = rot.rope_cos_sin(&pre_pos, theta, rope_dtype, dev)?;
    // Prefill's activation is the flat batch-of-one `[1, total, …]`, so its
    // tables carry the same leading axis.
    let half = q.cfg.attn_head_dim / 2;
    let pre_rope = (
        pre_cos.reshape((1, pre_rows, half))?,
        pre_sin.reshape((1, pre_rows, half))?,
    );

    let dec_pm: RefCell<Option<SharedPm>> = RefCell::new(None);
    let pre_pm: RefCell<Option<SharedPm>> = RefCell::new(None);
    // NeoX half-split within the rotary width — the layout `RotaryLayout`
    // permutes the head dims into, never the interleaved GPT-J form.
    let interleaved = false;
    let inv_freq = model.inv_freq_device();
    let dec_params = BatchedAttentionParams::new(
        &dec_rope.0,
        &dec_rope.1,
        interleaved,
        inv_freq,
        &rope_cs,
        decode_headers,
        dec_q,
        generation,
        &dec_pm,
    );
    let pre_params = BatchedAttentionParams::new(
        &pre_rope.0,
        &pre_rope.1,
        interleaved,
        inv_freq,
        &rope_cs,
        prefill_headers,
        pre_q,
        generation,
        &pre_pm,
    );

    // Where each sequence's rows sit in the packed buffer — what a recurrent
    // mixer needs to run one sequence at a time against its own state.
    let spans = seq_spans(seq_ids, &q_lens)?;
    let eps = q.cfg.rms_norm_eps;

    // The decode pointer table for the whole sweep — every DeltaNet layer's
    // state/tail addresses for every decode sequence, ONE host upload, built
    // here where the launch queue is still empty. Each layer takes its slice;
    // a per-layer upload would sync the stream mid-sweep and serialise the
    // pipeline. `None` when the wave carries no decode span.
    #[cfg(feature = "cuda")]
    let dn_table = crate::models::delta_net::cuda::build_wave_table(&spans, stores)?;

    for li in layer_start..layer_end {
        match &q.layers[li].mix {
            QuantLayerMix::Attention(_) => {
                let kv = model.kv_map().kv_index(li).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "qwen35 wave: layer {li} attends but has no KV index"
                    ))
                })?;
                let mut cache_refs: Vec<&mut KvCache> = contexts
                    .iter_mut()
                    .map(|c| &mut c.kv_caches.caches[kv])
                    .collect();
                let (dec_c, pre_c) = cache_refs.split_at_mut(n_decode);
                let mut wave_groups: Vec<WaveAttnGroup> = Vec::with_capacity(2);
                if n_decode > 0 {
                    wave_groups.push(WaveAttnGroup {
                        caches: dec_c,
                        offsets: dec_off,
                        params: &dec_params,
                        rows: n_decode,
                        decode_layout: true,
                    });
                }
                if n_prefill > 0 {
                    wave_groups.push(WaveAttnGroup {
                        caches: pre_c,
                        offsets: pre_off,
                        params: &pre_params,
                        rows: pre_rows,
                        decode_layout: false,
                    });
                }
                let layer = Qwen35AttentionLayer {
                    layer: &q.layers[li],
                    n_head: q.cfg.num_attention_heads,
                    n_kv_head: q.cfg.num_kv_heads,
                    head_dim: q.cfg.attn_head_dim,
                    rotary: model.rotary(),
                };
                // `layer_idx` names the KV layer, not the trunk layer: it is
                // what the per-layer arena bookkeeping inside the mixed
                // dispatch indexes by, and that bookkeeping is per cache.
                forward_layer_batched_mixed(&layer, &mut wave_groups, &mut x, embed_dtype, kv)?;
            }
            QuantLayerMix::DeltaNet(_) => {
                let orig = x.dtype();
                #[cfg(feature = "cuda")]
                let layer_table = match &dn_table {
                    Some(t) => Some(t.layer_slice(li)?),
                    None => None,
                };
                #[cfg(not(feature = "cuda"))]
                let layer_table = None;
                delta_net_mix_wave(q, li, &spans, &mut x, stores, eps, layer_table.as_ref())?;
                quantized_delta_net_ffn(&q.layers[li], &mut x, embed_dtype, orig)?;
            }
        }
    }

    if layer_end < num_layers {
        return Ok((WavePhase::Residual(x), None));
    }

    // Head over the rows that need logits: every decode row (one token each,
    // flat positions `0..n_decode`) and the last token of every prefill row.
    let xt = x.to_tensor();
    let hidden = xt.dim(2)?;
    let x_flat = xt.reshape((xt.dim(1)?, hidden))?;
    let mut idx: Vec<u32> = Vec::with_capacity(n_decode + n_prefill);
    for d in 0..n_decode {
        idx.push(d as u32);
    }
    let mut acc = n_decode as u32;
    for &l in pre_q {
        acc += l as u32;
        idx.push(acc - 1);
    }
    if idx.is_empty() {
        return Ok((WavePhase::Residual(x), None));
    }
    let pre_norm = {
        let sel = Tensor::from_vec(idx, n_decode + n_prefill, x_flat.device())?;
        x_flat.index_select(&sel, 0)?.contiguous()?
    };
    // The head's span, reset per forward — the lifetime the norm and the logits
    // actually have. Seeded from `wave_root`, which yields a ticket rather than
    // a borrow, so the logits stay `'static`-typed and physically on the span;
    // what makes that sound is handing the guard back with them.
    #[cfg(feature = "cuda")]
    let head_span = match dev {
        Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Forward)?),
        _ => None,
    };
    #[cfg(not(feature = "cuda"))]
    let head_span: Option<WaveGuard> = None;
    let logits = {
        #[cfg(feature = "cuda")]
        {
            let acts = q.final_norm.forward_dynamic(
                &pre_norm,
                q.lm_head.int8mode(),
                wave_root(head_span.as_ref()),
            )?;
            q.lm_head.forward_dynamic(acts.as_dynamic(), pre_norm.dtype())?
        }
        #[cfg(not(feature = "cuda"))]
        {
            use candle_nn::Module;
            q.lm_head.forward(&q.final_norm.forward(&pre_norm)?)?
        }
    };
    Ok((
        WavePhase::Logits(TensorCat::from_cat_tensor(logits, 0)?),
        head_span,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::batch_test::test_helpers::hf_get;
    use crate::models::batch_test::utils::TestParams;
    use crate::models::batched_inference::BatchedConfig;
    use crate::models::dialect::Dialect;
    use crate::models::qwen35::loader::load_reference_model;
    use crate::models::quantized_qwen35::from_gguf_path;
    use crate::models::qwen35::quantized_loader::Qwen35LoadOptions;
    use candle::quantized::gguf_file::Content;
    use candle::quantized::Int8Mode;
    use hf_hub::RepoType;
    use std::io::{BufReader, Seek, SeekFrom};

    /// **The wave's oracle.** Prefill the same tokens through the F32
    /// reference — which is token-identical to llama.cpp — and through
    /// `forward_wave`, and require the same next token.
    ///
    /// This is the check that separates "the engine is wrong" from "the model
    /// answered differently than the fixture expects". A gate that generates
    /// fluent but off-task text tells you nothing about which of the two you
    /// are looking at; this does, because the reference has no wave, no paged
    /// KV, no quantized projections and no recurrent bookkeeping — it is the
    /// arithmetic alone.
    ///
    /// The 0.8B on purpose: it is the largest of the family that dequantizes
    /// to F32 in host RAM (the 9B would need ~36 GB), and a wave bug is
    /// geometry, not scale — the same 3:1 schedule, the same partial rotary,
    /// the same head_dim 256.
    #[test]
    #[ignore = "reads the pinned Qwen3.5-0.8B GGUF and needs a GPU. Run with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                qwen35::forward::tests::wave_matches_the_reference -- --ignored --nocapture"]
    fn wave_matches_the_reference() -> Result<()> {
        let path = hf_get(
            "unsloth/Qwen3.5-0.8B-GGUF",
            RepoType::Model,
            "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
            "Qwen3.5-0.8B-BF16.gguf",
        )?;
        let tok_path = hf_get(
            "Qwen/Qwen3.5-0.8B",
            RepoType::Model,
            "2fc06364715b967f1860aea9cf38778875588b17",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::Msg(format!("load tokenizer: {e}")))?;

        // A prompt with the family's real turn structure, so a mis-encoded
        // special token would show up here rather than only in the gate.
        let prompt_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
                           <|im_start|>user\nThe capital of France is<|im_end|>\n\
                           <|im_start|>assistant\n<think>\n\n</think>\n\n";
        let tokens: Vec<u32> = tokenizer
            .encode(prompt_text, false)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        println!("prompt: {} tokens", tokens.len());
        // Every marker must be one id, not spelled out — the gate's symptom
        // (the model emitting "user") is what a split marker looks like.
        let decoded = tokenizer
            .decode(&tokens, false)
            .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
        assert_eq!(decoded, prompt_text, "prompt did not round-trip");

        // ── The reference, on CPU, in F32 ──
        let file = std::fs::File::open(&path)?;
        let mut reader = BufReader::new(file);
        let content = Content::read(&mut reader)?;
        reader.seek(SeekFrom::Start(0))?;
        let reference = load_reference_model(&content, &mut reader, &Device::Cpu)?;

        // Prompt length is the first thing to rule out when a row goes
        // non-finite: the chunked scan carries state across 64-token chunks,
        // and a short probe would never reach the second one.
        let finite = |t: &Tensor| -> Result<bool> {
            let m = t.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            Ok(m.is_finite())
        };
        let plain: Vec<u32> = tokenizer
            .encode(
                "The quick brown fox jumps over the lazy dog. Pack my box with five \
                 dozen liquor jugs. How vexingly quick daft zebras jump. The five \
                 boxing wizards jump quickly at dawn every single morning.",
                false,
            )
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        for (label, ids) in [("plain", &plain), ("chat", &tokens)] {
            // One full-length forward answers the question in the passing
            // case; the per-prefix sweep — a session and a forward per length,
            // O(n²) token-work — runs only on failure, where locating the
            // first bad length is exactly what it is for. Sweeping
            // unconditionally made this the slowest part of the suite, for a
            // diagnostic no passing run reads.
            let mut st = reference.new_session()?;
            let full = reference.forward(ids, &mut st)?;
            if finite(&full)? {
                println!("  reference [{label}]: finite over all {} tokens", ids.len());
                continue;
            }
            let mut first_bad = None;
            for n in 1..=ids.len() {
                let mut st = reference.new_session()?;
                let l = reference.forward(&ids[..n], &mut st)?;
                if !finite(&l)? {
                    first_bad = Some(n);
                    break;
                }
            }
            let n = first_bad.expect("the full forward was non-finite, so some prefix is");
            println!(
                "  reference [{label}]: finite up to {} tokens, non-finite at {n} \
                 (token id {})",
                n - 1,
                ids[n - 1]
            );
        }

        let mut ref_state = reference.new_session()?;
        let ref_logits = reference.forward(&tokens, &mut ref_state)?;
        let ref_last = ref_logits.narrow(0, tokens.len() - 1, 1)?.squeeze(0)?;
        assert!(
            finite(&ref_last)?,
            "the reference itself produced a non-finite logit row — fix the \
             oracle before comparing anything to it"
        );
        let ref_argmax = ref_last.argmax(0)?.to_scalar::<u32>()?;

        // ── The production wave, on the GPU ──
        let device = Device::new_cuda(0)?;
        let model = from_gguf_path(
            &path,
            &device,
            Qwen35LoadOptions {
                int8mode: Some(Int8Mode::Off),
                expert_pack_dir: None,
            },
        )?;
        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let ids = Tensor::from_vec(tokens.clone(), (1, tokens.len()), &device)?;
        let step = model.forward_wave(
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
        let logits = step
            .logits
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("wave returned no logits".into()))?;
        let wave_last = logits[0].flatten_all()?.to_device(&Device::Cpu)?;
        let wave_argmax = wave_last.argmax(0)?.to_scalar::<u32>()?;
        // The logits sit on the forward span and `WaveResult` holds its guard,
        // which is what stops the span being reclaimed underneath them. That
        // guard also *is* the live wave as far as the partition is concerned,
        // so anything that would create an arena — the second session below —
        // has to wait for it. `wave_last` is already a host copy.
        drop(step);
        drop(session);

        let name = |id: u32| {
            tokenizer
                .decode(&[id], false)
                .unwrap_or_else(|_| format!("<{id}>"))
        };
        println!(
            "reference → {ref_argmax} {:?}   wave → {wave_argmax} {:?}",
            name(ref_argmax),
            name(wave_argmax)
        );

        // Cosine over the whole row, reported before the assert so a near-miss
        // (quantization) reads differently from a structural break.
        let a = ref_last.to_dtype(DType::F32)?;
        let b = wave_last.to_dtype(DType::F32)?;
        let dot = a.mul(&b)?.sum_all()?.to_scalar::<f32>()?;
        let na = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let nb = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let cos = dot / (na * nb).max(1e-6);
        println!("logit cosine {cos:.6}");

        assert_eq!(
            ref_argmax, wave_argmax,
            "the wave and the reference disagree on the next token \
             ({ref_argmax} {:?} vs {wave_argmax} {:?}, cosine {cos:.4})",
            name(ref_argmax),
            name(wave_argmax)
        );
        assert!(cos > 0.99, "logit rows diverged: cosine {cos}");

        // ── The same comparison on the gate's own prompt ──
        //
        // ~800 tokens of system instruction plus the story, which is the shape
        // the story-rewrite gate actually runs. A short prompt exercises one
        // chunk of the recurrent scan and one page of KV; this one crosses
        // several of both, so it is where a carry between chunks or a stale
        // page shows up. Agreement here says the gate's answer — whatever it
        // is — is the model's, not the engine's.
        let params = TestParams::new(4, &std::fs::read_to_string(&tok_path)?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true);
        let mut gate_tokens = params.system_prompt_tokens(0);
        gate_tokens.extend(params.user_prompt_tokens(0));
        println!("gate prompt: {} tokens", gate_tokens.len());

        // Swept by prefix length rather than measured once at the end. A single
        // number cannot tell precision loss from a structural fault; the shape
        // of the curve can. Smooth decay is accumulation (the wave computes in
        // the session's activation dtype, the reference in F32); a step at a
        // boundary is a carry between recurrent chunks or a KV page.
        // Two KV dtypes, because they answer different questions. The
        // reference stores nothing — it keeps every layer's whole history in
        // F32 — so an F32 cache is the closest the engine gets to it and any
        // gap there is the engine's. BF16 is what a session actually runs, and
        // the difference between the two curves is the price of the cache.
        let mut worst = 1.0f32;
        for kv_dtype in [DType::F32, DType::BF16] {
            println!("  ── KV {kv_dtype:?} ──");
            worst = worst.min(sweep(
                &reference,
                &model,
                &device,
                &gate_tokens,
                kv_dtype,
                &name,
            )?);
        }
        assert!(
            worst > 0.99,
            "the wave drifts from the reference as context grows (worst cosine \
             {worst:.4}) — accumulation would be gradual and small; this is not"
        );

        // ── Decode ──
        //
        // Prefill and decode share almost nothing: different attention kernel,
        // and on the recurrent side a fused one-token step that updates the
        // state in place rather than the chunked scan. A prefill that matches
        // the reference exactly says nothing about either. Greedy on both
        // sides, so a divergence shows up as a different token and not as a
        // sampling difference.
        let mut ref_state = reference.new_session()?;
        let ref_logits = reference.forward(&gate_tokens, &mut ref_state)?;
        let mut ref_tok = ref_logits
            .narrow(0, gate_tokens.len() - 1, 1)?
            .squeeze(0)?
            .argmax(0)?
            .to_scalar::<u32>()?;

        let mut session =
            model.create_batched_session(BatchedConfig::default().with_dtype(DType::BF16))?;
        let seq = session.create_sequence()?;
        let ids = Tensor::from_vec(gate_tokens.clone(), (1, gate_tokens.len()), &device)?;
        let step = model.forward_wave(
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
        let mut wave_tok = step.logits.as_ref().expect("logits")[0]
            .flatten_all()?
            .argmax(0)?
            .to_scalar::<u32>()?;
        drop(step);
        assert_eq!(
            ref_tok, wave_tok,
            "first sampled token already differs ({:?} vs {:?})",
            name(ref_tok), name(wave_tok)
        );

        for i in 0..8 {
            let rl = reference.forward(&[ref_tok], &mut ref_state)?;
            ref_tok = rl.get(0)?.argmax(0)?.to_scalar::<u32>()?;

            let ids = Tensor::from_vec(vec![wave_tok], (1, 1), &device)?;
            let step = model.forward_wave(
                &mut session,
                &[seq],
                &[ids],
                &[],
                &[],
                &[],
                &[],
                0,
                model.num_layers(),
                None,
            )?;
            wave_tok = step.logits.as_ref().expect("logits")[0]
                .flatten_all()?
                .argmax(0)?
                .to_scalar::<u32>()?;
            drop(step);

            println!(
                "  decode {i}: reference {:?}   wave {:?}{}",
                name(ref_tok),
                name(wave_tok),
                if ref_tok == wave_tok { "" } else { "   ← DIVERGED" }
            );
            assert_eq!(
                ref_tok, wave_tok,
                "decode step {i} diverged: reference {:?}, wave {:?}",
                name(ref_tok),
                name(wave_tok)
            );
        }
        Ok(())
    }

    /// **The wave against itself.** Prefilling `[0, n)` in one forward and in
    /// two must land on the same logits.
    ///
    /// This needs no oracle, which is what makes it worth having: the
    /// reference is a second implementation and can be wrong (it was — a
    /// chunked-scan overflow put NaN in its logits at 23 tokens), so a
    /// disagreement between the two says only that they differ. Splitting the
    /// *same* implementation says something sharper. Everything that carries
    /// across the split — the recurrent `S` and conv tail, the paged K/V, the
    /// offsets — has to compose, and a carry that is dropped, double-applied
    /// or written at the wrong position shows up here and nowhere in a
    /// single-shot run.
    ///
    /// Splits are chosen against `CHUNK_SIZE` (32): inside the first chunk, on
    /// its boundary, and past it.
    #[test]
    #[ignore = "reads the pinned Qwen3.5-0.8B GGUF and needs a GPU. Run with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                qwen35::forward::tests::wave_one_shot_equals_segmented -- --ignored --nocapture"]
    fn wave_one_shot_equals_segmented() -> Result<()> {
        let path = hf_get(
            "unsloth/Qwen3.5-0.8B-GGUF",
            RepoType::Model,
            "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
            "Qwen3.5-0.8B-BF16.gguf",
        )?;
        let tok_path = hf_get(
            "Qwen/Qwen3.5-0.8B",
            RepoType::Model,
            "2fc06364715b967f1860aea9cf38778875588b17",
            "tokenizer.json",
        )?;
        let params = TestParams::new(4, &std::fs::read_to_string(&tok_path)?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true);
        let mut tokens = params.system_prompt_tokens(0);
        tokens.extend(params.user_prompt_tokens(0));
        tokens.truncate(200);

        let device = Device::new_cuda(0)?;
        let model = from_gguf_path(
            &path,
            &device,
            Qwen35LoadOptions {
                int8mode: Some(Int8Mode::Off),
                expert_pack_dir: None,
            },
        )?;

        // One forward over `[0, n)`, optionally split at `cut`.
        let run = |cut: Option<usize>| -> Result<Tensor> {
            let mut session =
                model.create_batched_session(BatchedConfig::default().with_dtype(DType::F32))?;
            let seq = session.create_sequence()?;
            let bounds: Vec<(usize, usize)> = match cut {
                None => vec![(0, tokens.len())],
                Some(c) => vec![(0, c), (c, tokens.len())],
            };
            let mut last = None;
            for (a, b) in bounds {
                let ids = Tensor::from_vec(tokens[a..b].to_vec(), (1, b - a), &device)?;
                let step = model.forward_wave(
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
                let row = step
                    .logits
                    .as_ref()
                    .ok_or_else(|| candle::Error::Msg("no logits".into()))?[0]
                    .flatten_all()?
                    .to_device(&Device::Cpu)?;
                drop(step);
                // What the next segment will build its RoPE positions, causal
                // mask and admission truncation from. All three read the same
                // number, so if it is wrong the segment is not merely
                // imprecise — it attends the wrong span at the wrong offsets.
                // The session's own offset is advanced lazily — the next
                // forward's entry reconciliation adopts the backing — so the
                // backing is the number that has to be right here.
                let backing = session.sequence_backing_tokens(seq).unwrap_or(usize::MAX);
                assert_eq!(
                    backing, b,
                    "after prefilling [{a}, {b}) the KV should hold {b} tokens"
                );
                last = Some(row);
            }
            last.ok_or_else(|| candle::Error::Msg("no segments".into()))
        };

        let whole = run(None)?;
        let cos = |a: &Tensor, b: &Tensor| -> Result<f32> {
            let a = a.to_dtype(DType::F32)?;
            let b = b.to_dtype(DType::F32)?;
            let dot = a.mul(&b)?.sum_all()?.to_scalar::<f32>()?;
            let na = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            let nb = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            Ok(dot / (na * nb).max(1e-6))
        };

        // The same split through the F32 reference. It shares the mixer core
        // (`delta_net_mix`) with the production path but none of the paged KV,
        // so a split it survives and the wave does not is attention-side, and
        // one they both fail is the recurrence composing wrongly.
        let file = std::fs::File::open(&path)?;
        let mut reader = BufReader::new(file);
        let content = Content::read(&mut reader)?;
        reader.seek(SeekFrom::Start(0))?;
        let reference = load_reference_model(&content, &mut reader, &Device::Cpu)?;
        let ref_run = |cut: Option<usize>| -> Result<Tensor> {
            let mut st = reference.new_session()?;
            let bounds: Vec<(usize, usize)> = match cut {
                None => vec![(0, tokens.len())],
                Some(c) => vec![(0, c), (c, tokens.len())],
            };
            let mut last = None;
            for (a, b) in bounds {
                let l = reference.forward(&tokens[a..b], &mut st)?;
                last = Some(l.narrow(0, b - a - 1, 1)?.squeeze(0)?);
            }
            last.ok_or_else(|| candle::Error::Msg("no segments".into()))
        };
        let ref_whole = ref_run(None)?;

        let mut worst = 1.0f32;
        for cut in [16usize, 32, 33, 64, 128] {
            let split = run(Some(cut))?;
            let c = cos(&whole, &split)?;
            let rc = cos(&ref_whole, &ref_run(Some(cut))?)?;
            worst = worst.min(c);
            println!(
                "  split at {cut:>3}: wave {c:.6} (argmax {} vs {})   reference {rc:.6}",
                whole.argmax(0)?.to_scalar::<u32>()?,
                split.argmax(0)?.to_scalar::<u32>()?
            );
        }
        assert!(
            worst > 0.999,
            "prefilling in two forwards does not equal prefilling in one \
             (worst cosine {worst:.5}) — something that must carry across the \
             split is not carrying"
        );
        Ok(())
    }

    /// Cosine between the reference's and the wave's final logit row at a
    /// range of prefix lengths. Returns the worst.
    fn sweep(
        reference: &crate::models::qwen35::model::Qwen35Model,
        model: &HybridBatched,
        device: &Device,
        gate_tokens: &[u32],
        kv_dtype: DType,
        name: &dyn Fn(u32) -> String,
    ) -> Result<f32> {
        let finite = |t: &Tensor| -> Result<bool> {
            let m = t.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            Ok(m.is_finite())
        };
        let mut worst = 1.0f32;
        for n in [32usize, 64, 65, 128, 256, 512, gate_tokens.len()] {
            let n = n.min(gate_tokens.len());
            let ids_host = &gate_tokens[..n];

            let mut ref_state = reference.new_session()?;
            let ref_logits = reference.forward(ids_host, &mut ref_state)?;
            let ref_last = ref_logits.narrow(0, n - 1, 1)?.squeeze(0)?;
            assert!(finite(&ref_last)?, "reference non-finite at {n} tokens");
            let ref_argmax = ref_last.argmax(0)?.to_scalar::<u32>()?;

            let mut session =
                model.create_batched_session(BatchedConfig::default().with_dtype(kv_dtype))?;
            let seq = session.create_sequence()?;
            let ids = Tensor::from_vec(ids_host.to_vec(), (1, n), &device)?;
            let step = model.forward_wave(
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
            let wave_last = step
                .logits
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("wave returned no logits".into()))?[0]
                .flatten_all()?
                .to_device(&Device::Cpu)?;
            let wave_argmax = wave_last.argmax(0)?.to_scalar::<u32>()?;
            drop(step);
            drop(session);

            let a = ref_last.to_dtype(DType::F32)?;
            let b = wave_last.to_dtype(DType::F32)?;
            let dot = a.mul(&b)?.sum_all()?.to_scalar::<f32>()?;
            let na = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            let nb = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            let c = dot / (na * nb).max(1e-6);
            worst = worst.min(c);
            println!(
                "    {n:>4} tokens: cosine {c:.6}   ref {:?} / wave {:?}{}",
                name(ref_argmax),
                name(wave_argmax),
                if ref_argmax == wave_argmax {
                    ""
                } else {
                    "   ← ARGMAX DIVERGED"
                }
            );
        }
        Ok(worst)
    }
}

/// Embed token ids through the host-resident table.
///
/// One of the two sanctioned GPU→CPU touches on the hot path (CLAUDE.md
/// invariant 3): the ids come back, a CPU `index_select` gathers the rows, and
/// one upload carries them in — which keeps a `vocab × hidden` table (4 GB at
/// the 9B's geometry) out of VRAM entirely.
fn embed_rows(model: &QuantModel, ids: &Tensor, dtype: DType) -> Result<Tensor> {
    let flat = ids.flatten_all()?;
    let n = flat.elem_count();
    let host_ids = flat.to_dtype(DType::U32)?.to_device(&Device::Cpu)?;
    let rows = model.embed.index_select(&host_ids, 0)?;
    let hidden = model.cfg.hidden_size;
    rows.reshape((1, n, hidden))?
        .to_dtype(dtype)?
        .to_device(&model.device)
}
