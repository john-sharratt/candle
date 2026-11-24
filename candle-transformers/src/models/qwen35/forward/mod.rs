//! The hybrid wave: the layer sweep for a Gated-DeltaNet stack, which mixes
//! recurrent layers and attention layers.
//!
//! Everything around the sweep is the shared wave driver: bounding a forward's
//! token count, routing one-token prefills, permuting rows, and rolling KV back
//! on failure. This module holds only what a hybrid does differently. Its KV is
//! indexed by attention layer rather than by trunk layer. Its recurrent state is
//! kept per sequence and advances in place, so every wave must either commit or
//! roll it back. Only part of each head rotates. Glue rows are refused, because
//! the gap-fill kernel is built for head dimension 128 and this family attends at
//! 256.
//!
//! - `images`: a wave's image rows, gathered as placeholders and copied over them.
//! - `layer_sweep`: the sweep over one layer range, with its checkpoints.
//! - `managed_model`: the scheduler-facing hooks for pricing, drafting and rewinding.
//! - `tests_wave`: oracle tests against the F32 reference and split-forward tests.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use candle::quantized::cuda::MOE_PREDICT_VOTES_MAX_TOKENS;
use candle::wave_provenance::WaveTicket;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::KvCache;
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{
    begin_forward, begin_wave, end_wave_transient, plan_wave_transient, LayerPhase,
    PrefillKvStageLayout, PrefillLaunchBounds, WavePlan, WaveWidth,
};

use super::batched::HybridBatched;
use super::draft::{head_wave_pass, HeadWave};
use super::mtp::MTP_MAX_DRAFT;
use super::quantized_attention::Qwen35AttentionLayer;
use super::quantized_delta_net::quantized_delta_net_ffn;
use super::quantized_weights::QuantModel;
use super::spec::split_block_rows;
use super::wave::delta_net_mix_wave;

#[cfg(feature = "cuda")]
use crate::models::delta_net::cuda::build_wave_table;
use crate::models::delta_net::seq_spans;
use crate::models::delta_net::ExportedLayerState;
use crate::models::delta_net::LayerKind;
use crate::models::delta_net::{RecurrentCompaction, RecurrentStateStore, TIMELINE_ROWS};
use crate::models::profile::pipeline_record_duration;
use crate::models::verify_wave::{upload_plan_rows, VerifyPlan};
use candle_nn::kv_cache::ModelGeometry;

use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ManagedBatchedModel, ModelCoreProperties, WaveResult,
};
use crate::models::batched_layer::{
    forward_layer_batched_mixed, BatchedAttentionParams, BatchedPrefillMeta, DecodeHeaders,
    WaveAttnGroup,
};
use crate::models::batched_model::{WaveGuard, WavePhase};
use crate::models::expert_lre::{PipelineStats, ProfileSnapshot, WeightPlan, WeightPlanning};
use crate::models::head_rows::select_head_rows;
use crate::models::lazy_rope::LazyRope;
use crate::models::prefill_utils::SharedPm;
use crate::models::qwen_vision::wave_images::{
    image_rotation, place_image_rows, upload_image_rows,
};
use crate::models::residency_rows::residency_decode_rows;
use crate::models::residual_rows::CreepResidual;
use crate::models::rope_schedule::RungSelect;
use crate::models::tensor_cat::TensorCat;
use crate::models::wave_admit::admit_wave_kv;
use crate::models::wave_buffers::{wave_empty_ticketed, wave_root, wave_root_ticketed};
use crate::models::wave_driver::{
    assemble_wave_contexts, drive_step, drive_wave, StepOutcome, StepSpec, SweepMode, WaveGroups,
    WaveSweep,
};
use crate::models::wave_token_ids::device_token_ids;
use images::wave_image_inputs;

// A verify block is the drafts plus the row they follow, and it must fit the
// DeltaNet timeline to be partially acceptable.
const _: () = assert!(MTP_MAX_DRAFT < TIMELINE_ROWS);

impl HybridBatched {
    /// Everything a forward does before its recurrent bracket opens: hand back
    /// the last tier, move the boundary, claim the state its sequences need —
    /// the prefill scratch too, for each whose span (`q_lens`) is a prefill one.
    fn prologue(
        &self,
        session: &BatchedInferenceSession,
        seqs: &[usize],
        q_lens: &[usize],
    ) -> Result<()> {
        // Hand back the previous wave's transient tier, and let the elastic
        // boundary grow in the one gap it is legal in — every guard from that
        // wave is dropped and this one has opened none.
        //
        // The release belongs HERE and not at the end of the wave that placed
        // it. `end_wave_transient` gates on `live_generations`, a host-side
        // count: by the time a sweep returns, its generations are dropped but
        // its kernels are still in flight, so releasing there hands ground back
        // to the persistence thread while the GPU is still reading it. Moving it
        // there was measured — the hot→warm migrate path took an illegal address
        // within seconds, 208 faults against a handful. What makes this point
        // safe is the work in between, which drains the wave that placed it.
        //
        // It runs before `ensure_recurrent` because that claims span regions for
        // any store it creates, and the ground those come from belongs to the wave
        // once a tier is placed: claiming under one is refused outright, since no
        // weight-side concession reaches above it.
        #[cfg(feature = "cuda")]
        if let Device::Cuda(d) = HybridBatched::device(self) {
            end_wave_transient(&d.cuda_stream());
            self.reclaim_spare_ground();
        }

        // Offsets in `seq_ids` order, read from the session — a sequence
        // standing at zero gets its recurrent state reset, not just created
        // (see `ensure_recurrent`).
        let offsets: Vec<usize> = seqs
            .iter()
            .map(|&s| session.sequence_offset(s).unwrap_or(0))
            .collect();
        self.ensure_recurrent(seqs, &offsets, q_lens)
    }
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
///
/// The dequantized rows are the residual, written where `residual` names —
/// the forward's span (`WaveBuffer::ForwardResidual`), or an owned buffer when
/// no forward span stands, as on the CPU device.
fn embed_rows(
    model: &QuantModel,
    ids: &Tensor,
    dtype: DType,
    residual: Option<WaveTicket>,
) -> Result<Tensor> {
    let n = ids.elem_count();
    #[cfg(feature = "cuda")]
    let staging = match &model.device {
        Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Attention)?),
        _ => None,
    };
    #[cfg(not(feature = "cuda"))]
    let staging: Option<WaveGuard> = None;
    let rows = model.embed.rows(
        ids,
        &model.device,
        wave_root(staging.as_ref()),
        dtype,
        residual,
    )?;
    drop(staging);
    rows.reshape((1, n, model.cfg.hidden_size))
}

mod images;
mod layer_sweep;
mod managed_model;
#[cfg(test)]
mod tests_wave;
