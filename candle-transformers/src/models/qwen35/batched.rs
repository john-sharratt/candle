//! The hybrid model as the scheduler holds it.
//!
//! Everything the engine asks that a hybrid answers *differently* is
//! collected here and delegates to the pieces that were built and tested
//! separately: [`super::engine`] for geometry, sessions and provenance,
//! [`super::kv_layout`] for the layer↔KV translation, [`super::rotary`] for
//! the rotary reordering, and [`super::wave`] for the layer sweep.
//!
//! The recurrent state lives here rather than in the session, because the
//! session's per-sequence storage is the paged KV cache and a DeltaNet layer
//! has none. One [`RecurrentStateStore`] per sequence, keyed by sequence id,
//! mirroring how `deepseek4` keeps its per-sequence streaming state.

use std::collections::HashMap;
use std::sync::Mutex;

use candle::quantized::Int8Mode;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{KvErrorThresholdFactors, ModelGeometry};

use super::engine::{create_session, provenance_layer_indices, wave_geometry};
use crate::models::delta_net::KvLayerMap;
use super::quantized_weights::{QuantLayerMix, QuantModel};
use crate::models::delta_net::RecurrentStateStore;
use crate::models::rotary_layout::RotaryLayout;
use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ModelCoreProperties, ProvenanceLayerIndices,
};

/// A loaded hybrid model of this lineage, ready to be driven by the scheduler.
///
/// Generic across the family: the per-model files
/// (`models/quantized_qwen35.rs`, `models/quantized_qwen35_moe.rs`,
/// `models/quantized_qwen36_moe.rs`, `models/quantized_qwen38.rs`) each
/// construct one of these around their pinned checkpoint, supplying the
/// model's own derived KV threshold factors.
pub struct HybridBatched {
    model: QuantModel,
    /// The concrete model's derived KV error-threshold factor row
    /// (`candle_nn::kv_cache::QWEN35_0_8B_KV_FACTORS` and siblings) —
    /// supplied at construction because thresholds are model-specific by
    /// standing rule and this struct serves the whole lineage.
    kv_factors: KvErrorThresholdFactors,
    kv_map: KvLayerMap,
    rotary: RotaryLayout,
    /// Inverse frequencies over the **rotary** width, not the head width.
    ///
    /// Carried because the attention parameters take one; the CUDA paged path
    /// reads the interleaved table instead and never touches it, but a table
    /// sized for the whole head would be a standing invitation to rotate 256
    /// dims where only 64 turn.
    inv_freq: Tensor,
    /// The interleaved `[pos, head_dim]` `(cos, sin)` table the paged kernels
    /// index, keyed by the arena's block count.
    ///
    /// Built once per geometry rather than per wave: it spans the whole
    /// addressable context, so rebuilding it every forward would cost more
    /// than the attention it feeds.
    rope_cs: Mutex<Option<(usize, Tensor)>>,
    /// Provenance capture depths, snapped onto layers that actually attend.
    provenance: ProvenanceLayerIndices,
    /// Per-sequence recurrent state, keyed by the scheduler's sequence id.
    ///
    /// Behind a lock because the scheduler holds the model behind a shared
    /// reference while a wave mutates the state of the sequences in it.
    recurrent: Mutex<HashMap<usize, RecurrentStateStore>>,
}

impl HybridBatched {
    /// Wrap a loaded model with its derived KV threshold factor row.
    pub fn new(model: QuantModel, kv_factors: KvErrorThresholdFactors) -> Result<Self> {
        let kv_map = KvLayerMap::new(&model.cfg.layer_kinds);
        if kv_map.num_kv_layers() == 0 {
            candle::bail!(
                "qwen35: a stack with no attention layers has no KV to page — \
                 the engine has nothing to schedule against"
            );
        }
        let rotary = RotaryLayout::new(
            model.cfg.attn_head_dim,
            model.cfg.rope_dim,
            &model.device,
        )?;
        let theta = model.cfg.rope_theta as f32;
        let rope_dim = model.cfg.rope_dim;
        let inv: Vec<f32> = (0..rope_dim / 2)
            .map(|j| 1f32 / theta.powf(2.0 * j as f32 / rope_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv, (rope_dim / 2,), &model.device)?;
        let provenance = provenance_layer_indices(&model.cfg, &kv_map).ok_or_else(|| {
            candle::Error::Msg(
                "qwen35: no attention layers, so no provenance can be captured".into(),
            )
        })?;
        Ok(Self {
            model,
            kv_factors,
            kv_map,
            rotary,
            inv_freq,
            rope_cs: Mutex::new(None),
            provenance,
            recurrent: Mutex::new(HashMap::new()),
        })
    }

    /// Inverse frequencies over the rotary width.
    pub fn inv_freq_device(&self) -> &Tensor {
        &self.inv_freq
    }

    /// The interleaved `(cos, sin)` table covering `max_blocks × CHUNK_SIZE`
    /// positions, built once and reused while the geometry holds.
    pub fn rope_cs(&self, max_blocks: usize) -> Result<Tensor> {
        let mut slot = self
            .rope_cs
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: rope_cs lock poisoned".into()))?;
        if let Some((blocks, table)) = slot.as_ref() {
            if *blocks == max_blocks {
                return Ok(table.clone());
            }
        }
        let table = self.rotary.rope_table(
            max_blocks * candle_nn::CHUNK_SIZE,
            self.model.cfg.rope_theta as f32,
            DType::F32,
            &self.model.device,
        )?;
        *slot = Some((max_blocks, table.clone()));
        Ok(table)
    }

    /// Let the elastic boundary grow into ground the weight side is no longer
    /// using. Legal only between forwards, which is where phase 0 calls it.
    pub fn reclaim_spare_ground(&self) {
        #[cfg(feature = "cuda")]
        if let Some(cache) = self.model.experts.as_ref() {
            cache.reclaim_spare_ground();
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

    /// Ensure every sequence in `seqs` has recurrent state consistent with how
    /// many tokens its KV holds.
    ///
    /// **A sequence standing at offset 0 has no history, so its recurrent state
    /// must be the sequence-start value** — and that is a reset, not merely a
    /// create. The paged KV is owned by the session and disappears with it;
    /// this map is owned by the *model* and outlives every session, because a
    /// DeltaNet layer has no per-session storage to put it in. So a new
    /// session, a fork, or a truncation back to nothing all hand a sequence id
    /// back with an empty cache while the entry here still holds whatever the
    /// last conversation left in it. Keying the reset off the offset ties the
    /// two together: the recurrence follows the KV, which is the thing the
    /// scheduler actually manages.
    ///
    /// Missed, this is close to invisible. The cache is empty so attention is
    /// correct, the shapes all match, nothing errors — the model simply
    /// answers as though it remembers a conversation the prompt never had.
    pub fn ensure_recurrent(&self, seqs: &[usize], offsets: &[usize]) -> Result<()> {
        if seqs.len() != offsets.len() {
            candle::bail!(
                "ensure_recurrent: {} sequences against {} offsets",
                seqs.len(),
                offsets.len()
            );
        }
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        for (&seq, &offset) in seqs.iter().zip(offsets) {
            let fresh = || {
                RecurrentStateStore::new(
                    &self.model.cfg.layer_kinds,
                    &self.model.cfg.delta_net,
                    &self.model.device,
                )
            };
            match map.entry(seq) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(fresh()?);
                }
                std::collections::hash_map::Entry::Occupied(mut slot) => {
                    if offset == 0 {
                        slot.insert(fresh()?);
                    }
                }
            }
        }
        Ok(())
    }

    /// Drop a sequence's recurrent state.
    ///
    /// The scheduler releases KV slots when a sequence ends; the recurrent
    /// half has to be released with it or the map grows for the life of the
    /// process.
    pub fn release_recurrent(&self, seq: usize) -> Result<()> {
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        map.remove(&seq);
        Ok(())
    }

    /// How many sequences currently carry recurrent state.
    pub fn recurrent_len(&self) -> Result<usize> {
        Ok(self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?
            .len())
    }

    /// Open a wave over `seqs`' recurrent state, stashing each entry state so
    /// a failed wave can be rolled back.
    ///
    /// The wave-atomicity contract: the mixer advances state in place as it
    /// runs, so "did not commit" is not the same as "did not happen". Every
    /// wave that begins must either commit or roll back.
    pub fn begin_recurrent_wave(&self, seqs: &[usize]) -> Result<()> {
        self.for_each_store(seqs, |s| s.begin_wave())
    }

    pub fn commit_recurrent_wave(&self, seqs: &[usize]) -> Result<()> {
        self.for_each_store(seqs, |s| {
            s.commit_wave();
            Ok(())
        })
    }

    pub fn rollback_recurrent_wave(&self, seqs: &[usize]) -> Result<()> {
        self.for_each_store(seqs, |s| s.rollback_wave())
    }

    /// Lift the wave's sequences' state out of the map, in `seqs` order.
    ///
    /// The layer sweep mixes every sequence against its own state in one pass,
    /// so it needs `&mut` to all of them at once — which a `HashMap` cannot
    /// hand out. Taking them out and putting them back is the borrow-checkable
    /// form of that, and it is why [`Self::put_recurrent`] must run on the
    /// failure path too: a sequence whose state stayed lifted would be missing
    /// from the map for every later wave.
    pub fn take_recurrent(&self, seqs: &[usize]) -> Result<Vec<RecurrentStateStore>> {
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        let mut out = Vec::with_capacity(seqs.len());
        for &seq in seqs {
            out.push(map.remove(&seq).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "qwen35: sequence {seq} has no recurrent state — \
                     `ensure_recurrent` was not called for this wave"
                ))
            })?);
        }
        Ok(out)
    }

    /// Put lifted state back. Runs on both the success and the failure path.
    pub fn put_recurrent(&self, seqs: &[usize], stores: Vec<RecurrentStateStore>) -> Result<()> {
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        for (&seq, store) in seqs.iter().zip(stores) {
            map.insert(seq, store);
        }
        Ok(())
    }

    fn for_each_store(
        &self,
        seqs: &[usize],
        mut f: impl FnMut(&mut RecurrentStateStore) -> Result<()>,
    ) -> Result<()> {
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        for &seq in seqs {
            let store = map.get_mut(&seq).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "qwen35: sequence {seq} has no recurrent state — \
                     `ensure_recurrent` was not called for this wave"
                ))
            })?;
            f(store)?;
        }
        Ok(())
    }

    // ── The engine's questions, answered for a hybrid ──────────────────────

    /// Transformer depth — what bounds a wave's layer range.
    pub fn num_layers(&self) -> usize {
        self.model.cfg.num_layers
    }

    /// KV heads per *attention* layer.
    pub fn n_kv_head(&self) -> usize {
        self.model.cfg.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.model.cfg.attn_head_dim
    }

    pub fn device(&self) -> &Device {
        &self.model.device
    }

    pub fn wave_geometry(&self, act_dtype: DType) -> ModelGeometry {
        wave_geometry(&self.model.cfg, act_dtype)
    }

    /// Re-materialise every norm weight in the session's activation dtype.
    ///
    /// The forward *refuses* a dtype it was not prepared for, so this is the
    /// single place the conversion happens — at session creation, never
    /// inside a wave.
    pub fn maybe_change_dtype(&self, dtype: DType) -> Result<()> {
        for layer in &self.model.layers {
            layer.attn_norm.maybe_change_dtype(dtype)?;
            layer.post_attn_norm.maybe_change_dtype(dtype)?;
            if let QuantLayerMix::Attention(a) = &layer.mix {
                a.q_norm.maybe_change_dtype(dtype)?;
                a.k_norm.maybe_change_dtype(dtype)?;
            }
        }
        self.model.final_norm.maybe_change_dtype(dtype)
    }

    /// A session whose KV is allocated per attention layer, with the norms
    /// materialised for its activation dtype.
    ///
    /// The model's KV threshold factor row is folded into the session config
    /// here, exactly as the `ManagedBatchedModel` default does — this
    /// override replaces that default (the KV layer count differs from the
    /// transformer depth on a hybrid), so it must also replace the factor
    /// fold, or the per-model calibration silently never reaches the
    /// compression policy.
    pub fn create_batched_session(
        &self,
        config: BatchedConfig,
    ) -> Result<BatchedInferenceSession> {
        let mut config = config;
        config.k_hi_error_threshold_factor *= self.kv_factors.k_hi;
        config.k_low_error_threshold_factor *= self.kv_factors.k_low;
        config.v_hi_error_threshold_factor *= self.kv_factors.v_hi;
        config.v_low_error_threshold_factor *= self.kv_factors.v_low;
        let session = create_session(&self.model.cfg, &self.model.device, config)?;
        self.maybe_change_dtype(session.activation_dtype())?;
        Ok(session)
    }

    /// Static properties, with the provenance depths snapped onto layers that
    /// actually attend.
    ///
    /// Infallible: [`Self::new`] refuses a stack with no attention layers, so
    /// the snap always has somewhere to land.
    pub fn model_core_properties(&self) -> ModelCoreProperties {
        let provenance_layer_indices = self.provenance;
        ModelCoreProperties {
            num_layers: self.model.cfg.num_layers,
            n_kv_heads: self.model.cfg.num_kv_heads,
            head_dim: self.model.cfg.attn_head_dim,
            provenance_layer_indices,
            // The concrete model's named factor row, supplied at construction —
            // one source of truth shared with the offline report, never inline
            // literals.
            k_hi_error_threshold_factor: self.kv_factors.k_hi,
            k_low_error_threshold_factor: self.kv_factors.k_low,
            v_hi_error_threshold_factor: self.kv_factors.v_hi,
            v_low_error_threshold_factor: self.kv_factors.v_low,
        }
    }

    /// Rows the KV side can admit, priced against **attention** layers.
    ///
    /// The trait's default multiplies the per-row cost by transformer depth,
    /// which on a 3:1 hybrid over-charges by 4× and refuses four times more
    /// prefill than the cache can hold.
    pub fn kv_width_cap(&self, act_dtype: DType) -> Option<usize> {
        let stats = candle_nn::kv_cache::region_stats(0)?;
        let free = (stats.free + stats.blocked).saturating_sub(1);
        let per_row = 2 * self.n_kv_head() * self.head_dim() * act_dtype.size_in_bytes();
        let per_row_all_kv_layers = per_row.checked_mul(self.kv_map.num_kv_layers())?;
        if per_row_all_kv_layers == 0 {
            return None;
        }
        let kv_bytes = free.saturating_mul(candle_nn::kv_cache::REGION_BYTES);
        // Never zero: a cap of nought is not a narrow wave but no wave, and
        // once the KV side is full it would be permanent.
        Some((kv_bytes / per_row_all_kv_layers).max(1))
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::{DeltaNetDims, Qwen35Config};
    use super::*;

    /// The 9B's geometry, without weights — enough to exercise the
    /// bookkeeping that does not touch tensors.
    fn cfg() -> Qwen35Config {
        Qwen35Config {
            vocab_size: 248_320,
            hidden_size: 4096,
            intermediate_size: 12_288,
            num_layers: 32,
            layer_kinds: Qwen35Config::schedule_from_interval(32, 4),
            num_attention_heads: 16,
            num_kv_heads: 4,
            attn_head_dim: 256,
            rope_dim: 64,
            rope_sections: [11, 11, 10, 0],
            rope_theta: 1e7,
            rms_norm_eps: 1e-6,
            delta_net: DeltaNetDims {
                head_dim: 128,
                n_k_heads: 16,
                n_v_heads: 32,
                conv_kernel: 4,
            },
            moe: None,
            num_mtp_layers: 0,
            max_position_embeddings: 262_144,
        }
    }

    /// A wave that advances state in place must be able to undo it — the
    /// store is opened, written, and rolled back, and the entry state has to
    /// come back exactly.
    #[test]
    fn recurrent_state_rolls_back_to_the_entry_value() -> Result<()> {
        let c = cfg();
        let dev = Device::Cpu;
        let mut store = RecurrentStateStore::new(&c.layer_kinds, &c.delta_net, &dev)?;
        // Layer 0 is DeltaNet under the 3:1 schedule.
        // `copy`, not `clone`: a clone shares storage, and against a state that
        // is now written in place it would track the mutation instead of
        // recording what preceded it.
        let entry = store.layer_state(0)?.s.copy()?;
        store.begin_wave()?;
        {
            // The mutating form the mixer uses.
            let live = store.layer_state_mut(0)?;
            let bump = Tensor::full(3f32, live.s.shape(), &dev)?;
            live.s.add_mut(&bump)?;
        }
        let during = store.layer_state(0)?.s.copy()?;
        store.rollback_wave()?;
        let after = store.layer_state(0)?.s.copy()?;

        let max = |t: &candle::Tensor| -> Result<f32> {
            t.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
        };
        assert!(max(&during.sub(&entry)?)? > 1.0, "the wave did write");
        assert_eq!(
            max(&after.sub(&entry)?)?,
            0.0,
            "rollback must restore the entry state exactly"
        );
        Ok(())
    }

    /// An attention layer has no recurrent slot, and asking for one is an
    /// error rather than a silently-zero state.
    #[test]
    fn attention_layers_have_no_recurrent_slot() -> Result<()> {
        let c = cfg();
        let store = RecurrentStateStore::new(&c.layer_kinds, &c.delta_net, &Device::Cpu)?;
        assert_eq!(store.n_recurrent_layers(), 24, "32 layers, 8 attend");
        assert!(store.layer_state(0).is_ok(), "layer 0 is DeltaNet");
        assert!(
            store.layer_state(3).is_err(),
            "layer 3 attends and must not answer with a state"
        );
        Ok(())
    }
}
