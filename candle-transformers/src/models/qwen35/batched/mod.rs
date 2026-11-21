//! The hybrid model as the scheduler holds it: one type that answers the engine's
//! questions for a model whose layers mix recurrence and attention.
//!
//! The questions a hybrid answers differently from a plain transformer are
//! answered here and delegated to the engine, KV layout, rotary and wave modules.
//! Recurrent state lives on this type, keyed by sequence id, because a DeltaNet
//! layer has no paged KV to hold it. The verify cohort, the hidden buffers and
//! the row settings are set around each forward and cleared on the way out, error
//! or not, so one wave cannot inherit another's settings. LoRA adapters are loaded
//! with the model, since a single adapter is shared by every conversation that
//! opts into it.
//!
//! - `hidden_capture`: arms and reads the hiddens a speculative wave hands on.
//! - `recurrent_state`: keeps each sequence's DeltaNet state in step with its KV.
//! - `session_geometry`: geometry, session creation and the KV width cap.
//! - `verify_and_draft`: the verify cohort, the draft seed and the draft head.
//! - Tests: `tests_state`.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use candle::quantized::Int8Mode;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{KvErrorThresholdFactors, ModelGeometry};

use super::draft::draft_cohort;
use super::engine::{
    create_session, mtp_kv_layer, provenance_layer_indices, session_kv_layers, wave_geometry,
    wave_kv_range,
};
use super::quantized_weights::QuantModel;
use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ModelCoreProperties, ProvenanceLayerIndices,
};
use crate::models::delta_net::ExportedLayerState;
use crate::models::delta_net::KvLayerMap;
use crate::models::delta_net::LayerKind;
use crate::models::delta_net::RecurrentStateStore;
use crate::models::delta_net::StagedSnapshot;
use crate::models::delta_net::{
    commit_stores, compact_stores, fold_stores, RecurrentCompaction, TIMELINE_ROWS,
};
use crate::models::draft_ladder::DraftLadder;
use crate::models::lora::Adapter;
use crate::models::rope_schedule::{RopeRungs, RopeSchedule};
use crate::models::rotary_layout::RotaryLayout;

use super::rope::lineage_schedule;

/// A loaded hybrid model of this lineage, ready to be driven by the scheduler.
///
/// Generic across the family: the per-model files
/// (`models/quantized_qwen35.rs`, `models/quantized_qwen35_moe.rs`,
/// `models/quantized_qwen36_moe.rs`, `models/quantized_qwen38.rs`) each
/// construct one of these around their pinned checkpoint, supplying the
/// model's own derived KV threshold factors.
pub struct HybridBatched {
    model: QuantModel,
    /// LoRA adapters loaded alongside the base weights, by name.
    ///
    /// Loaded up front — with the model, not with a conversation — because an
    /// adapter is a few hundred MB of weights that every adapted conversation
    /// shares. Nothing is merged into the base: `Wx` is still the quantized
    /// base projection, and the adapter adds a rank-`r` term to its result, so
    /// one resident checkpoint serves adapted and unadapted conversations at
    /// once. A wave names the one it wants; [`Self::adapter`] resolves it.
    adapters: HashMap<String, Arc<Adapter>>,
    /// The concrete model's derived KV error-threshold factor row
    /// (`candle_nn::kv_cache::QWEN35_0_8B_KV_FACTORS` and siblings) —
    /// supplied at construction because thresholds are model-specific by
    /// standing rule and this struct serves the whole lineage.
    kv_factors: KvErrorThresholdFactors,
    /// The concrete model's draft-budget ladder
    /// (`crate::models::draft_ladder::QWEN35_9B_DRAFT` and siblings). Supplied
    /// at construction for the same reason as the factor row above: how far
    /// ahead it pays to draft is measured per checkpoint, and this struct serves
    /// the whole lineage. A checkpoint with no NextN head carries
    /// [`DraftLadder::NONE`].
    draft: DraftLadder,
    kv_map: KvLayerMap,
    rotary: RotaryLayout,
    /// The lineage's RoPE rungs (`super::rope`) over the **rotary** width: the
    /// kernels read frequency `f < rope_dim / 2` from a slot's rung and treat
    /// every later one as pass-through, which is where [`RotaryLayout`] puts
    /// the head's non-rotary dims.
    rope: RopeRungs,
    /// Provenance capture depths, snapped onto layers that actually attend.
    provenance: ProvenanceLayerIndices,
    /// Per-sequence recurrent state, keyed by the scheduler's sequence id.
    ///
    /// Behind a lock because the scheduler holds the model behind a shared
    /// reference while a wave mutates the state of the sequences in it.
    recurrent: Mutex<HashMap<usize, RecurrentStateStore>>,
    /// Snapshots held for sequences that have no store yet
    /// ([`Self::stage_recurrent`]), the layers shared by every slot opened on one
    /// prompt branch. Built into a store when the sequence's state is first
    /// wanted — a fork from it, or a wave on it — so a slot that has not run a
    /// turn holds shared host bytes rather than a store of its own. Locked after
    /// `recurrent` wherever both are held.
    staged: Mutex<HashMap<usize, StagedSnapshot>>,
    /// Sequences whose prefill rows this wave must score **every** one of.
    ///
    /// An ordinary wave wants one logits row per prefill span (the last token's
    /// — the only prediction a prompt asks for). A speculative verify wants all
    /// of them: each row of the block is a prediction to compare a proposal
    /// against. Set for the duration of one forward by
    /// [`Self::verify_blocks`](crate::models::batched_inference::ManagedBatchedModel::verify_blocks)
    /// and cleared on the way out, error or not.
    verify_rows: Mutex<Vec<usize>>,
    /// The **armed verify cohort**: each sequence whose block this step verifies,
    /// and the block's length. Their DeltaNet layers stay pending past the
    /// forward's commit — every other short span folds there — until the
    /// reconcile folds the rows the accept kept ([`super::spec`]), or an abort
    /// discards them.
    armed: Mutex<HashMap<usize, usize>>,
    /// Sequences whose scored rows this wave must also hand back as
    /// **post-final-norm hiddens** — the MTP head's seed. Set for the duration
    /// of one forward, alongside `verify_rows`.
    hidden_seqs: Mutex<Vec<usize>>,
    /// Per-sequence buffers the sweep writes those hiddens into, `[rows, hidden]`
    /// in the wave's activation dtype. Sized before the forward opens: the arena
    /// refuses a device allocation from inside a wave.
    verify_hidden: Mutex<HashMap<usize, Tensor>>,
    /// How many rows of each buffer the last armed wave actually filled.
    ///
    /// The buffers only ever grow, so their `dim(0)` is a high-water mark and
    /// not an answer to "how much did this wave write". The accept needs the
    /// real count to find the last accepted row, and it runs after the capture
    /// is disarmed — so this outlives the active set deliberately.
    capture_rows: Mutex<HashMap<usize, usize>>,
    /// Per-sequence MTP seed: the trunk's post-`final_norm` hidden at the last
    /// ACCEPTED position.
    ///
    /// One value, two jobs, and they are the same value because the head's
    /// recurrence and its wave pass read the position the same way: it is what
    /// the next draft step is conditioned on, and it is the hidden input of the
    /// next wave's first row for that sequence (which is the position right
    /// after it). `None` until a wave has scored a row for the sequence — its
    /// first prefill, whose row 0 has no predecessor and takes zeros.
    ///
    /// The head's KV is NOT here: it is a layer of the session's paged cache.
    /// See [`super::draft`].
    seed: Mutex<HashMap<usize, Tensor>>,
    /// The draft walk's two carried-hidden buffers — see [`Self::draft_carry`].
    draft_carry: Mutex<Option<(Tensor, Tensor)>>,
}

impl HybridBatched {
    /// Wrap a loaded model with the two rows derived per checkpoint: its KV
    /// threshold factors and its draft-budget ladder.
    pub fn new(
        model: QuantModel,
        kv_factors: KvErrorThresholdFactors,
        draft: DraftLadder,
    ) -> Result<Self> {
        let kv_map = KvLayerMap::new(&model.cfg.layer_kinds);
        if kv_map.num_kv_layers() == 0 {
            candle::bail!(
                "qwen35: a stack with no attention layers has no KV to page — \
                 the engine has nothing to schedule against"
            );
        }
        let rotary = RotaryLayout::new(model.cfg.attn_head_dim, model.cfg.rope_dim, &model.device)?;
        let rope = RopeRungs::new(&lineage_schedule(&model.cfg)?, &model.device)?;
        let provenance = provenance_layer_indices(&model.cfg, &kv_map).ok_or_else(|| {
            candle::Error::Msg(
                "qwen35: no attention layers, so no provenance can be captured".into(),
            )
        })?;
        Ok(Self {
            model,
            adapters: HashMap::new(),
            kv_factors,
            draft,
            kv_map,
            rotary,
            rope,
            provenance,
            recurrent: Mutex::new(HashMap::new()),
            staged: Mutex::new(HashMap::new()),
            verify_rows: Mutex::new(Vec::new()),
            armed: Mutex::new(HashMap::new()),
            hidden_seqs: Mutex::new(Vec::new()),
            verify_hidden: Mutex::new(HashMap::new()),
            capture_rows: Mutex::new(HashMap::new()),
            seed: Mutex::new(HashMap::new()),
            draft_carry: Mutex::new(None),
        })
    }

    /// Load a LoRA adapter alongside the base weights, under `name`.
    ///
    /// Called during construction, before the model is handed to the scheduler
    /// — an adapter is shared by every conversation that opts into it, so it is
    /// loaded once with the model rather than per conversation.
    ///
    /// The adapter is checked against the checkpoint here, where a mismatch is a
    /// load error with both sets of numbers in it. Left unchecked, a wrong
    /// adapter is not a crash: the shapes that do line up still multiply, and
    /// the model simply answers slightly wrongly.
    /// `dtype` is the width activations arrive in — the same one
    /// `set_activation_dtype` materialises the norms in. PEFT stores adapters
    /// F32; converting once here rather than per projection is the hot-path
    /// rule against `to_dtype` in the loop.
    pub fn load_adapter(&mut self, name: &str, dir: &std::path::Path, dtype: DType) -> Result<()> {
        let adapter = Adapter::load(dir, name, dtype, &self.model.device)
            .map_err(|e| candle::Error::Msg(format!("LoRA `{name}` from {dir:?}: {e}")))?;

        // The adapter's attention pairs must land on layers that actually
        // attend. On a 3:1 hybrid they should be exactly the attention layers;
        // an adapter trained against a differently-mixed sibling would put them
        // on DeltaNet layers, where there is no q/k/v/o to add to and the pairs
        // would be silently ignored.
        for li in adapter.attention_layers() {
            match self.model.cfg.layer_kinds.get(li) {
                Some(LayerKind::Attention) => {}
                Some(LayerKind::DeltaNet) => candle::bail!(
                    "LoRA `{name}` has attention pairs on layer {li}, which is a DeltaNet \
                     layer in this checkpoint — the adapter was trained against a \
                     differently-mixed model and its attention half would be ignored"
                ),
                None => candle::bail!(
                    "LoRA `{name}` has pairs on layer {li} but this checkpoint has {} layers",
                    self.model.cfg.layer_kinds.len()
                ),
            }
        }
        tracing::info!(
            "LoRA `{name}`: {} pairs, rank {}, scale {:.4}, attention layers {:?}",
            adapter.len(),
            adapter.rank,
            adapter.scale(),
            adapter.attention_layers()
        );
        self.adapters.insert(name.to_owned(), Arc::new(adapter));
        Ok(())
    }

    /// Resolve a wave's adapter name against what is loaded.
    ///
    /// An unknown name is an error, not a fallback to the base model: a
    /// conversation that asked for an adapter and silently got the base one
    /// reads as the fine-tune not working, and there is nothing in the output
    /// to point at the name that failed to match.
    pub fn adapter(&self, name: Option<&str>) -> Result<Option<&Adapter>> {
        match name {
            None => Ok(None),
            Some(n) => match self.adapters.get(n) {
                Some(a) => Ok(Some(a.as_ref())),
                None => candle::bail!(
                    "wave asked for LoRA `{n}`, which is not loaded (have: {:?})",
                    self.adapters.keys().collect::<Vec<_>>()
                ),
            },
        }
    }

    /// The lineage's RoPE rungs, which every paged kernel rotates from.
    pub fn rope(&self) -> &RopeRungs {
        &self.rope
    }

    /// Run `schedule` in place of the lineage's — the control a long-context
    /// gate measures the lineage's rungs against (`docs/progressive_yarn.md`
    /// §10). Nothing stored depends on a rung (I1), so a session opened after
    /// this simply reads the new tables. The rotary width must be the model's.
    pub fn set_rope_schedule(&mut self, schedule: &RopeSchedule) -> Result<()> {
        if schedule.pairs() != self.rope.pairs() {
            candle::bail!(
                "qwen35: a {}-pair schedule on a {}-pair rotary width",
                schedule.pairs(),
                self.rope.pairs()
            );
        }
        self.rope = RopeRungs::new(schedule, &self.model.device)?;
        Ok(())
    }

    /// Let the elastic boundary grow into ground the weight side is no longer
    /// using. Legal only between forwards, which is where phase 0 calls it.
    pub fn reclaim_spare_ground(&self) {
        #[cfg(feature = "cuda")]
        {
            if let Some(cache) = self.model.experts.as_ref() {
                cache.reclaim_spare_ground();
            }
            // The dense counterpart: a streamed model's layer slots are the same
            // kind of tradeable ground its experts would be, and this is the
            // direction that takes ground back once the KV side's peak has
            // passed. A resident store returns without asking.
            self.model.layers.reclaim_spare_ground();
        }
    }

    pub fn model(&self) -> &QuantModel {
        &self.model
    }

    pub fn kv_map(&self) -> &KvLayerMap {
        &self.kv_map
    }

    pub fn rotary(&self) -> &RotaryLayout {
        &self.rotary
    }

    /// The numeric mode this model's projections were loaded for, read off a
    /// weight rather than carried separately so it cannot drift from what the
    /// kernels see.
    pub fn int8mode(&self) -> Int8Mode {
        self.model.lm_head.int8mode()
    }
}

mod hidden_capture;
mod recurrent_state;
mod session_geometry;
#[cfg(test)]
mod tests_state;
mod verify_and_draft;
