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
use candle_nn::kv_cache::KvCache;
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{
    begin_forward, begin_wave, end_wave_transient, plan_wave_transient, LayerPhase, WavePlan,
    REGION_BYTES, WAVE_FORWARD_BYTES,
};

use super::batched::HybridBatched;
use super::draft::{head_wave_pass, HeadWave};
use super::mtp::MTP_MAX_DRAFT;
use super::quantized_attention::Qwen35AttentionLayer;
use super::quantized_delta_net::quantized_delta_net_ffn;
use super::quantized_weights::{QuantLayerMix, QuantModel};
use super::spec::{split_block_rows, StashSpan, VerifyStash};
use super::wave::delta_net_mix_wave;
use crate::models::delta_net::seq_spans;
use crate::models::delta_net::{RecurrentStateStore, StashSlot};
use candle_nn::kv_cache::ModelGeometry;

use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ManagedBatchedModel, ModelCoreProperties, WaveResult,
};
use crate::models::batched_layer::{
    forward_layer_batched_mixed, BatchedAttentionParams, WaveAttnGroup,
};
use crate::models::batched_model::{activation_dtype, WaveGuard, WavePhase};
use crate::models::expert_lre::{PipelineStats, ProfileSnapshot};
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

    /// Draft with the checkpoint's own NextN/MTP head ([`super::mtp`]), for the
    /// whole cohort in one batched pass.
    ///
    /// Empty — a plain decode step — when the conversion carried no head, or
    /// before a sequence has a seed. Lossless either way: the verify pass keeps
    /// only this model's own argmaxes.
    ///
    /// The budget is capped at [`MTP_MAX_DRAFT`] whatever the caller asks for.
    fn speculative_draft(
        &self,
        session: &mut BatchedInferenceSession,
        seqs: &[usize],
        committed: &[u32],
        max_len: usize,
    ) -> Result<Vec<Vec<u32>>> {
        self.mtp_draft(session, seqs, committed, max_len.min(MTP_MAX_DRAFT))
    }

    /// Verify every drafted block in ONE wave, alongside the plain cohort.
    ///
    /// Each block is a **prefill span**, not a run of decode rows: the
    /// recurrence is sequential within a sequence, so two rows of one sequence
    /// cannot decode in parallel against a single carried state — the prefill
    /// scan is the form that walks them in order. Plain rows lead as ordinary
    /// decode rows in the same wave, so the whole step pays one launch floor
    /// rather than two, and the MoE's per-layer expert traffic amortizes across
    /// both cohorts.
    ///
    /// Naming the verifying sequences does two things inside the sweep: the
    /// head scores every one of their rows (a block position's prediction is
    /// what a proposal is checked against), and each DeltaNet layer stashes
    /// their recurrence operands so [`Self::truncate_sequence`] can replay the
    /// accepted prefix. Cleared on the way out, error or not.
    fn verify_blocks(
        &self,
        session: &mut BatchedInferenceSession,
        plain: &[(usize, u32)],
        seqs: &[usize],
        blocks: &[Vec<u32>],
        layer_end: usize,
    ) -> Result<(Vec<Tensor>, Vec<Vec<Tensor>>)> {
        if plain.is_empty() && seqs.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        if seqs.len() != blocks.len() {
            candle::bail!(
                "qwen35 verify: {} sequences against {} blocks",
                seqs.len(),
                blocks.len()
            );
        }
        // A one-token block is a decode step; the driver routes those to
        // `plain` and never here, and a 1-row prefill would be folded into the
        // decode group by the driver — where the head scores it as one row
        // anyway, so the row split below would still hold. Refuse it rather
        // than rely on that: the stash's `len` would disagree with the block.
        if let Some(b) = blocks.iter().find(|b| b.len() < 2) {
            candle::bail!(
                "qwen35 verify: a {}-token block is a plain decode step, not a verify",
                b.len()
            );
        }
        let dseqs: Vec<usize> = plain.iter().map(|&(s, _)| s).collect();
        let dinputs: Vec<Tensor> = plain
            .iter()
            .map(|&(_, t)| Tensor::from_vec(vec![t], (1, 1), &Device::Cpu))
            .collect::<Result<_>>()?;
        let pinputs: Vec<Tensor> = blocks
            .iter()
            .map(|b| Tensor::from_vec(b.clone(), (1, b.len()), &Device::Cpu))
            .collect::<Result<_>>()?;

        // **Size every verifying sequence's stash before the forward opens.**
        // A wave's storage is claimed by `admit_wave_kv` and the transient tier
        // is placed against that claim, so the arena refuses a device
        // allocation from inside the forward — which is what a stash that
        // allocated as the sweep reached each layer would be. (It is also not
        // hypothetical: the 9B at four contexts is where the pool first had no
        // room to spare and the wave failed outright. That failure looked like a
        // VRAM ceiling and was recorded as one; it was this, and the width-4 row
        // of `speculative_decode_9b` is what holds the fix down.)
        let cohort: Vec<(usize, usize)> = seqs
            .iter()
            .enumerate()
            .map(|(i, &seq)| (seq, blocks[i].len()))
            .collect();
        self.begin_verify_stash(&cohort)?;
        // Arm the MTP seed capture over BOTH cohorts: a plain row's hidden is
        // what lets a sequence that drafted nothing this step draft on the
        // next, and it is the only way the very first step after prefill ever
        // acquires a seed.
        if self.has_drafter() {
            let mut want: Vec<(usize, usize)> = plain.iter().map(|&(s, _)| (s, 1)).collect();
            want.extend(seqs.iter().enumerate().map(|(i, &s)| (s, blocks[i].len())));
            self.arm_hidden_capture(&want, session.activation_dtype())?;
        }
        self.set_verify_row_seqs(seqs)?;
        let step = self.forward_wave(
            session,
            &dseqs,
            &dinputs,
            seqs,
            &pinputs,
            &[],
            &[],
            0,
            layer_end,
            None,
        );
        self.set_verify_row_seqs(&[])?;
        self.disarm_hidden_capture();
        let step = match step {
            Ok(s) => s,
            Err(e) => {
                // The wave rolled its recurrent state back, so the stash names
                // a rewind point that no longer exists.
                self.drop_verify_stashes(seqs);
                return Err(e);
            }
        };
        for &(seq, _) in plain {
            session.advance_sequence(seq, 1)?;
        }
        for (i, &seq) in seqs.iter().enumerate() {
            session.advance_sequence(seq, blocks[i].len())?;
        }
        // Copied off the wave's span: the driver compares argmaxes and may run
        // another forward before it is done with these rows.
        let logits = step.logits_owned()?;
        let lens: Vec<usize> = blocks.iter().map(|b| b.len()).collect();
        split_block_rows(&logits, plain.len(), &lens)
    }

    /// Roll `seq` back to `tokens`, recurrent state included.
    ///
    /// Three cases, and only the middle one is new:
    ///
    /// * `tokens == 0` — a full reset. The state returns to its sequence-start
    ///   value, which is what a fresh store holds, so the store is dropped.
    /// * `tokens` inside a block this model just verified — the speculative
    ///   rewind. `S` is a running sum with no suffix to remove, so the state is
    ///   re-advanced *forward* over the accepted prefix from the wave's
    ///   entering state, which the store's ping-pong still holds
    ///   ([`super::spec`]). Exact, not approximate: the replay runs the same
    ///   mixer over the same operands.
    /// * `tokens` equal to the sequence's current length — a no-op, which is
    ///   what the driver's uniform truncate is for a plain decode row.
    ///
    /// Anything else has no answer and is refused. KV rewound under a state
    /// that still holds the un-truncated history is silent corruption: the
    /// model answers as though it remembers tokens the cache no longer has
    /// (measured: re-prefilling a truncated prompt diverges by ~9.5 in the
    /// logits).
    fn truncate_sequence(
        &self,
        session: &mut BatchedInferenceSession,
        seq: usize,
        tokens: usize,
    ) -> Result<()> {
        self.truncate_sequences(session, &[(seq, tokens)])
    }

    /// The whole step's truncates in one call, so the cohort's recurrent
    /// rewinds batch: every rewinding sequence's replay runs as ONE launch pair
    /// per DeltaNet layer through the shared stash, instead of one per layer
    /// per sequence.
    ///
    /// Validation runs for every target before anything mutates — a target with
    /// no rewind point aborts the call with nothing absorbed and nothing
    /// replayed, and the stash goes back untouched.
    fn truncate_sequences(
        &self,
        session: &mut BatchedInferenceSession,
        targets: &[(usize, usize)],
    ) -> Result<()> {
        if targets.is_empty() {
            return Ok(());
        }
        let mut stash = self.take_verify_stash()?;

        // What each target needs, decided before anything is touched.
        enum Plan {
            Reset,
            NoOp,
            Rewind { span: StashSpan, kept: usize },
        }
        let mut plans: Vec<Plan> = Vec::with_capacity(targets.len());
        for &(seq, tokens) in targets {
            if tokens == 0 {
                plans.push(Plan::Reset);
                continue;
            }
            let span = stash.as_ref().and_then(|st| st.span_of(seq));
            let current = session.sequence_offset(seq).unwrap_or(0);
            match span {
                Some(sp) if tokens >= sp.start && tokens <= sp.start + sp.len => {
                    plans.push(Plan::Rewind {
                        span: sp,
                        kept: tokens - sp.start,
                    });
                }
                _ if tokens == current => plans.push(Plan::NoOp),
                _ => {
                    // Nothing has mutated yet; hand the buffers back before
                    // aborting the step.
                    if let Some(st) = stash {
                        self.put_verify_stash(st)?;
                    }
                    candle::bail!(
                        "qwen35: cannot truncate sequence {seq} to {tokens} tokens — it \
                         stands at {current} and no verified block covers that offset, so \
                         the DeltaNet recurrent state has no rewind point. Truncate to 0 \
                         (full reset), or rewind inside a block this model just verified."
                    )
                }
            }
        }

        // Every target's span is consumed now, whatever its plan: a stash span
        // is good for exactly one step, and a failed replay's must not survive
        // to rewind a later one.
        if let Some(st) = stash.as_mut() {
            for &(seq, _) in targets {
                st.remove(seq);
            }
        }

        let mut jobs: Vec<(StashSpan, usize)> = Vec::new();
        let result = (|| -> Result<()> {
            // Take each sequence's next draft seed from the hiddens this wave
            // captured — the row at the last position the accept kept. This
            // runs for no-ops exactly as for rewinds, because a PLAIN row's
            // one-token block is how a sequence that drafted nothing acquires
            // the seed to draft next step; only a reset skips it, and its
            // sequence has no history left to seed from.
            //
            // A rewind carries its own row count; a **no-op does not**, and
            // must not be given one. It means the accept kept everything the
            // wave ran, which is a count only the capture knows — `None` asks
            // it. Hardcoding 1 here would be right for every plain decode row
            // and silently wrong for anything else that ever classifies as a
            // no-op, seeding from the block's first hidden instead of its last.
            //
            // The head's KV needs nothing here. It took every one of these
            // positions inside the wave, as a layer, and the truncate below
            // rolls its rejected tail back with every other layer's.
            let seeds: Vec<(usize, Option<usize>)> = targets
                .iter()
                .zip(&plans)
                .filter_map(|(&(seq, _), plan)| match plan {
                    Plan::Reset => None,
                    Plan::NoOp => Some((seq, None)),
                    Plan::Rewind { kept, .. } => Some((seq, Some(*kept))),
                })
                .collect();
            self.mtp_take_seeds(&seeds)?;

            for (&(seq, tokens), plan) in targets.iter().zip(&plans) {
                match plan {
                    Plan::Reset => {
                        session.truncate_sequence_to_tokens(seq, tokens)?;
                        self.release_recurrent(seq)?;
                    }
                    Plan::NoOp => {}
                    Plan::Rewind { span, kept } => {
                        jobs.push((*span, *kept));
                        session.truncate_sequence_to_tokens(seq, tokens)?;
                    }
                }
            }
            match stash.as_ref() {
                Some(st) => self.replay_recurrent(st, &jobs),
                None => Ok(()),
            }
        })();
        if let Some(st) = stash {
            self.put_verify_stash(st)?;
        }
        result
    }

    /// **This lineage cannot be rewound.** The DeltaNet state is a running sum
    /// with no per-token decomposition, so there is no suffix to remove and no
    /// inverse to apply, and K/V rewound under a state that still holds the
    /// un-truncated history is silent corruption — the model answers as though
    /// it remembers tokens the cache no longer has (measured: re-prefilling a
    /// truncated prompt diverges by ~9.5 in the logits).
    ///
    /// Declaring it here refuses the one path that rewinds — speculative
    /// decode — at its entry point. That replaces a bail *inside* the rewind,
    /// which fired only after the driver had drafted and verified, and which
    /// site 8 (the `<think>` re-prefill) bypassed entirely by reaching the
    /// session directly.
    fn carries_recurrent_state(&self) -> bool {
        true
    }

    /// The recurrent map is keyed by sequence id and owned by the *model*, so
    /// the scheduler freeing the KV slot does not touch it. This is the hook
    /// that ties the two lifetimes together.
    fn release_sequence(&self, seq: usize) -> Result<()> {
        self.release_recurrent(seq)
    }

    fn recurrent_memory_count(&self) -> usize {
        HybridBatched::recurrent_len(self).unwrap_or(0)
    }

    /// A view carve. The child borrows the parent's KV; its recurrent state has
    /// to be copied, because it is about to advance it.
    ///
    /// Tolerant of a parent with no state yet: a view can be carved before the
    /// parent has ever run a wave (a brand-new conversation's first turn), and
    /// there the child correctly starts from the sequence-start value — a fresh
    /// store holds exactly that, so the child's own `ensure_recurrent` does the
    /// right thing and there is nothing to copy.
    fn fork_recurrent(&self, parent: usize, child: usize) -> Result<()> {
        if !self.has_recurrent(parent)? {
            return Ok(());
        }
        HybridBatched::fork_recurrent(self, parent, child)
    }

    /// A view finalizes: its decoded blocks transfer to the parent, and its
    /// state goes with them.
    ///
    /// Tolerant in the same direction and for the same reason — a view that
    /// never ran a wave has nothing to move, and the parent keeps what it had.
    fn move_recurrent(&self, child: usize, parent: usize) -> Result<()> {
        if !self.has_recurrent(child)? {
            return Ok(());
        }
        HybridBatched::move_recurrent(self, child, parent)
    }

    fn export_recurrent(
        &self,
        seq: usize,
    ) -> Result<Option<(u64, Vec<crate::models::delta_net::ExportedLayerState>)>> {
        HybridBatched::export_recurrent(self, seq)
    }

    fn restore_recurrent(
        &self,
        seq: usize,
        schedule_hash: u64,
        layers: &[crate::models::delta_net::ExportedLayerState],
    ) -> Result<bool> {
        HybridBatched::restore_recurrent(self, seq, schedule_hash, layers)?;
        Ok(true)
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
        HybridBatched::kv_layer_range(self, layer_start, layer_end)
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
        candle::bail!(
            "qwen35 wave: bad layer range [{layer_start}, {layer_end}) over {num_layers} layers"
        );
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
    // trunk range owns no cache at all, and the draft head's layer is one past
    // the end of it (see [`HybridBatched::kv_layer_range`]).
    let (kv_start, kv_end) = model.kv_layer_range(layer_start, layer_end);
    let head_kv = model.mtp_kv_layer().filter(|_| layer_end == num_layers);
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
            let packed = TensorCat::from_tensors(1, ids)?;
            TensorCat::from_cat_tensor(embed_rows(q, &packed.to_tensor(), embed_dtype)?, 0)?
        }
    };

    // The interleaved `(cos, sin)` table the paged kernels index by position.
    // Partial rotary, so it is the model's own table — `compute_rope_cs` would
    // rotate all 256 dims where only 64 turn.
    let max_blocks = contexts
        .first()
        .and_then(|c| {
            c.kv_caches
                .caches
                .first()
                .map(|k| k.k_cache().chunked_max_blocks())
        })
        .unwrap_or(0);
    let rope_cs = model.rope_cs(max_blocks)?;

    // Per-group split cos/sin over this wave's own positions.
    let theta = q.cfg.rope_theta;
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

    // Spans a speculative verify will have to rewind stash each DeltaNet
    // layer's recurrence operands as the sweep passes through it — every
    // verifying span into its own rows of ONE shared set of buffers, which is
    // what lets the replay advance the whole cohort per layer in one launch.
    // The buffers were sized by `verify_blocks` BEFORE this forward opened —
    // nothing here allocates. `None` on every ordinary wave, so nothing is
    // copied either.
    let verify_seqs = model.verify_row_seqs()?;
    let cohort_stash: Option<VerifyStash> = if verify_seqs.is_empty() {
        None
    } else {
        model.take_verify_stash()?
    };
    // Each span's row in the shared buffers, resolved once rather than per
    // layer.
    let stash_rows: Vec<Option<usize>> = spans
        .iter()
        .map(|s| {
            cohort_stash
                .as_ref()
                .and_then(|st| st.span_of(s.seq))
                .map(|sp| sp.row)
        })
        .collect();
    // Which recurrent layer each DeltaNet index is, counted the way
    // `RecurrentStateStore::recurrent_layer_indices` counts — the order the
    // replay walks the stash in.
    let mut dn_ord = q.layers[..layer_start]
        .iter()
        .filter(|l| matches!(l.mix, QuantLayerMix::DeltaNet(_)))
        .count();

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
                let slots: Vec<Option<StashSlot<'_>>> = stash_rows
                    .iter()
                    .map(|row| {
                        row.and_then(|r| {
                            cohort_stash.as_ref().map(|st| StashSlot {
                                ops: &st.layers[dn_ord],
                                row: r,
                            })
                        })
                    })
                    .collect();
                delta_net_mix_wave(
                    q,
                    li,
                    &spans,
                    &mut x,
                    stores,
                    eps,
                    layer_table.as_ref(),
                    &slots,
                )?;
                dn_ord += 1;
                quantized_delta_net_ffn(&q.layers[li], &mut x, embed_dtype, orig)?;
            }
        }
    }

    if layer_end < num_layers {
        if !verify_seqs.is_empty() {
            candle::bail!(
                "qwen35 wave: a speculative verify was split across layer windows — its \
                 stash would cover only this segment's DeltaNet layers, and a rewind from \
                 a partial stash advances some layers and not others"
            );
        }
        return Ok((WavePhase::Residual(x), None));
    }

    // The stash is complete: every DeltaNet layer of a full sweep has captured
    // its operands. Stamp each span's ABSOLUTE start — not `span.start`, which
    // is its row in the packed wave buffer, a different number entirely for
    // anything but the first span; the driver rewinds to
    // `sequence_offset + accepted` — and file it back where the truncate will
    // find it if the accept comes back short.
    if let Some(mut st) = cohort_stash {
        for sp in st.spans.iter_mut() {
            let at = spans.iter().position(|s| s.seq == sp.seq).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "qwen35 wave: stash span for sequence {} has no wave span",
                    sp.seq
                ))
            })?;
            sp.start = offsets[at];
        }
        model.put_verify_stash(st)?;
    }

    let xt = x.to_tensor();
    let hidden = xt.dim(2)?;
    let x_flat = xt.reshape((xt.dim(1)?, hidden))?;

    // **The draft head's layer.** One more attention pass over the same rows at
    // the same positions, against the KV layer past every trunk one, so the
    // head's history stays exactly as long as the sequence it drafts for. It
    // runs here rather than inside the sweep because its input is the trunk's
    // OUTPUT, not the residual stream. See [`super::draft`].
    if let (Some(head), Some(kv_layer)) = (q.mtp.as_ref(), head_kv) {
        let packed: Vec<Tensor> = contexts.iter().map(|c| c.input_ids.clone()).collect();
        let ids = TensorCat::from_tensors(1, packed)?;
        head_wave_pass(
            model,
            head,
            contexts,
            &x_flat,
            &ids.to_tensor(),
            &HeadWave {
                n_decode,
                pre_rows,
                dec_off,
                pre_off,
                dec_params: &dec_params,
                pre_params: &pre_params,
                spans: &spans,
                kv_layer,
                act_dtype: embed_dtype,
            },
        )?;
    }

    // Head over the rows that need logits: every decode row (one token each,
    // flat positions `0..n_decode`) and the last token of every prefill row —
    // except a **verifying** span, where every row is a prediction to compare a
    // proposal against, so all of them are scored.
    let mut idx: Vec<u32> = Vec::with_capacity(n_decode + pre_rows);
    for (d, _) in seq_ids.iter().take(n_decode).enumerate() {
        idx.push(d as u32);
    }
    let mut acc = n_decode as u32;
    for (k, &l) in pre_q.iter().enumerate() {
        let seq = seq_ids[n_decode + k];
        if verify_seqs.contains(&seq) {
            for t in 0..l as u32 {
                idx.push(acc + t);
            }
        } else {
            idx.push(acc + l as u32 - 1);
        }
        acc += l as u32;
    }
    if idx.is_empty() {
        return Ok((WavePhase::Residual(x), None));
    }
    let pre_norm = {
        let n_sel = idx.len();
        let sel = Tensor::from_vec(idx, n_sel, x_flat.device())?;
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
            q.lm_head
                .forward_dynamic(acts.as_dynamic(), pre_norm.dtype())?
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

/// Embed token ids through the off-card table, as `[1, n, hidden]` of `dtype`.
///
/// The table is never resident — a `vocab × hidden` tensor is 4 GB at the 9B's
/// geometry for one row read per token — so this is where the forward reaches
/// off the device for it.
///
/// Under [`EmbeddingTable::HostMapped`] that reach is not a *synchronisation*:
/// the GPU gathers the quantized rows over PCIe from the ids where they already
/// are. The staging span is opened just for the gather, before the layer loop
/// claims the same arena for real work — the bytes are reserved whether or not
/// anything uses them, so they are free here, and the gathered bytes are dead
/// the moment the dequantize on the next line has read them.
fn embed_rows(model: &QuantModel, ids: &Tensor, dtype: DType) -> Result<Tensor> {
    let n = ids.elem_count();
    #[cfg(feature = "cuda")]
    let staging = match &model.device {
        Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Attention)?),
        _ => None,
    };
    #[cfg(not(feature = "cuda"))]
    let staging: Option<WaveGuard> = None;
    let rows = model
        .embed
        .rows(ids, &model.device, wave_root(staging.as_ref()), dtype)?;
    drop(staging);
    rows.reshape((1, n, model.cfg.hidden_size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::batch_test::test_helpers::hf_get;
    use crate::models::batch_test::utils::TestParams;
    use crate::models::batched_inference::BatchedConfig;
    use crate::models::dialect::Dialect;
    use crate::models::quantized_qwen35::from_gguf_path;
    use crate::models::qwen35::loader::load_reference_model;
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
                println!(
                    "  reference [{label}]: finite over all {} tokens",
                    ids.len()
                );
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
            ref_argmax,
            wave_argmax,
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
            ref_tok,
            wave_tok,
            "first sampled token already differs ({:?} vs {:?})",
            name(ref_tok),
            name(wave_tok)
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
                if ref_tok == wave_tok {
                    ""
                } else {
                    "   ← DIVERGED"
                }
            );
            assert_eq!(
                ref_tok,
                wave_tok,
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
            let ids = Tensor::from_vec(ids_host.to_vec(), (1, n), device)?;
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
