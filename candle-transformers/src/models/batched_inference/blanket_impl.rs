//! Lets any `BatchedInference<M>` serve as a `ManagedBatchedModel` and as a wave
//! sweep without per-model glue.
//!
//! A uniform transformer runs the same layer body at every index, and every
//! layer attends, so the layer-dependent answers are the uniform ones. Group
//! assembly, token permutation, KV rollback and the decode advance are left to
//! `drive_wave`, which every wave model shares. The row-cache answer stays
//! `None`: no model reached through this impl serves an embedding tier from
//! disk, and the one that does implements the trait directly.

use super::*;

/// Blanket implementation of `ManagedBatchedModel` for `BatchedInference<M>`.
///
/// This allows models using the new `BatchedModelCore` + `BatchedInference` pattern
/// to work with `BatchedInferenceSession` without implementing `BatchedModel`.
impl<M: BatchedModelCore> ManagedBatchedModel for BatchedInference<M> {
    fn wave_geometry(&self, act_dtype: DType) -> ModelGeometry {
        self.model().wave_geometry(act_dtype)
    }

    fn rope_select(&self) -> RungSelect {
        self.rope().select().clone()
    }

    fn maybe_change_dtype(&self, dtype: DType) -> Result<()> {
        self.model().maybe_change_dtype(dtype)
    }

    fn num_layers(&self) -> usize {
        self.model().num_layers()
    }

    fn n_kv_head(&self) -> usize {
        self.model().n_kv_head()
    }

    fn head_dim(&self) -> usize {
        self.model().head_dim()
    }

    fn device(&self) -> &Device {
        self.model().device()
    }

    fn model_core_properties(&self) -> ModelCoreProperties {
        let n = self.model().num_layers();
        let provenance_layer_indices = if n == 0 {
            ProvenanceLayerIndices {
                syn_l0: 0,
                syn_l4: 0,
                sem_l0: 0,
                sem_l4: 0,
                prag_l0: 0,
                prag_l4: 0,
            }
        } else {
            let syn = (n * 15 / 100).max(1);
            let sem = n / 2;
            let prag = (n * 85 / 100).min(n - 1);
            ProvenanceLayerIndices {
                syn_l0: syn.saturating_sub(4),
                syn_l4: syn,
                sem_l0: sem.saturating_sub(4),
                sem_l4: sem,
                prag_l0: prag.saturating_sub(4),
                prag_l4: prag,
            }
        };
        ModelCoreProperties {
            num_layers: n,
            n_kv_heads: self.model().n_kv_head(),
            head_dim: self.model().head_dim(),
            provenance_layer_indices,
            k_hi_error_threshold_factor: self.model().k_hi_error_threshold_factor(),
            k_low_error_threshold_factor: self.model().k_low_error_threshold_factor(),
            v_hi_error_threshold_factor: self.model().v_hi_error_threshold_factor(),
            v_low_error_threshold_factor: self.model().v_low_error_threshold_factor(),
            // Uniform transformer: every layer attends, so every layer has a Q.
            provenance_capture_layers: ManagedBatchedModel::num_layers(self),
            can_gap_fill: !self.carries_recurrent_state(),
            carries_recurrent_state: self.carries_recurrent_state(),
        }
    }

    #[allow(clippy::too_many_arguments)]
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
        residual: Option<&mut CreepResidual>,
    ) -> Result<WaveResult> {
        session.expect_rope_select(self.rope().select())?;
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
            residual,
        )
    }

    fn forward_step(
        &self,
        session: &mut BatchedInferenceSession,
        spec: StepSpec<'_>,
    ) -> Result<StepOutcome> {
        session.expect_rope_select(self.rope().select())?;
        drive_step(self, session, spec)
    }

    fn claim_residual(
        &self,
        session: &BatchedInferenceSession,
        rows: usize,
    ) -> Result<CreepResidual> {
        let (width, dtype) = WaveSweep::residual_row(self, session)?;
        CreepResidual::claim(WaveSweep::device(self), rows, width, dtype)
    }

    fn prune(&self) -> Result<()> {
        self.model().prune()
    }

    fn expert_stats(&self) -> Option<PipelineStats> {
        self.model().expert_stats()
    }

    fn expert_pipeline_dead(&self) -> bool {
        self.model().expert_pipeline_dead()
    }

    fn expert_stats_shared(&self) -> Option<Arc<Mutex<PipelineStats>>> {
        self.model().expert_stats_shared()
    }

    fn expert_hit_counts(&self) -> Option<(usize, usize)> {
        self.model().expert_hit_counts()
    }

    fn expert_hit_references(&self) -> Option<(f64, f64)> {
        self.model().expert_hit_references()
    }

    // `row_cache_stats` keeps the trait's `None`: no model reached through
    // `BatchedInference<M>` serves an embedding tier from disk. The one that
    // does (Qwen3.8-Flash-Next's PLE table) implements `ManagedBatchedModel`
    // directly.

    fn layer_stream_stats(&self) -> Option<[usize; 7]> {
        self.model().layer_stream_stats()
    }

    fn request_kv_ground(&self, regions: usize) -> u64 {
        self.model().request_kv_ground(regions)
    }

    fn reclaim_spare_ground(&self) {
        self.model().reclaim_spare_ground()
    }

    fn concede_for_guest(&self, regions: usize) -> u64 {
        self.model().concede_for_guest(regions)
    }

    fn stash_pinned_for_guest(&self) -> u64 {
        self.model().stash_pinned_for_guest()
    }

    fn restore_serving_floor(&self) -> bool {
        self.model().restore_serving_floor()
    }

    fn resident_weight_bytes(&self) -> Option<usize> {
        self.model().resident_weight_bytes()
    }

    fn weight_plan(&self) -> WeightPlanning {
        self.model().weight_plan()
    }

    fn recurrent_reserved_bytes(&self) -> usize {
        self.model().recurrent_reserved_bytes()
    }

    fn recurrent_store_bytes(&self) -> usize {
        self.model().recurrent_store_bytes()
    }

    fn compact_recurrent(&self, max_moves: usize) -> Result<RecurrentCompaction> {
        self.model().compact_recurrent(max_moves)
    }

    fn reset_expert_stats(&self) {
        self.model().reset_expert_stats()
    }

    fn snapshot_profiles(&self) -> ProfileSnapshot {
        self.model().snapshot_profiles()
    }
}

/// A uniform transformer's half of a wave: the same layer body at every index.
///
/// Everything around it — group assembly, the 1-token reroute, the token
/// permutation, KV rollback, the decode advance — is [`drive_wave`], shared with
/// every other model that runs waves.
impl<M: BatchedModelCore> WaveSweep for BatchedInference<M> {
    fn device(&self) -> &Device {
        self.model().device()
    }

    fn take_device_fault(&self) -> Result<()> {
        self.model().take_device_fault()
    }

    fn num_layers(&self) -> usize {
        self.model().num_layers()
    }

    fn prefill_width_cap(&self, act_dtype: DType) -> usize {
        <Self as ManagedBatchedModel>::prefill_width_cap(self, act_dtype)
    }

    /// The single-stream residual: `hidden` wide, in the width the sweep embeds
    /// in — the activation width of the live KV's dtype.
    fn residual_row(&self, session: &BatchedInferenceSession) -> Result<(usize, DType)> {
        let dtype = activation_dtype(session.kv_live_dtype());
        Ok((self.model().wave_geometry(dtype).hidden, dtype))
    }

    fn sweep(
        &self,
        session: &mut BatchedInferenceSession,
        wave: WaveGroups<'_>,
    ) -> Result<(WavePhase, Option<WaveGuard>)> {
        self.forward_wave_contexts(session, wave)
    }
}
