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

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle::quantized::Int8Mode;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{KvErrorThresholdFactors, ModelGeometry};

use super::draft::draft_cohort;
use super::engine::{
    create_session, mtp_kv_layer, provenance_layer_indices, session_kv_layers, wave_geometry,
    wave_kv_range,
};
use super::quantized_weights::QuantModel;
use super::spec::{replay_accepted_prefixes, ReplayLayer, StashSpan, VerifyStash};
use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ModelCoreProperties, ProvenanceLayerIndices,
};
use crate::models::delta_net::validate_snapshot;
use crate::models::delta_net::DeltaNetConstants;
use crate::models::delta_net::ExportedLayerState;
use crate::models::delta_net::KvLayerMap;
use crate::models::delta_net::LayerKind;
use crate::models::delta_net::RecurrentStateStore;
use crate::models::draft_ladder::DraftLadder;
use crate::models::lora::Adapter;
use crate::models::rotary_layout::RotaryLayout;

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
    /// Sequences whose prefill rows this wave must score **every** one of.
    ///
    /// An ordinary wave wants one logits row per prefill span (the last token's
    /// — the only prediction a prompt asks for). A speculative verify wants all
    /// of them: each row of the block is a prediction to compare a proposal
    /// against. Set for the duration of one forward by
    /// [`Self::verify_blocks`](crate::models::batched_inference::ManagedBatchedModel::verify_blocks)
    /// and cleared on the way out, error or not.
    verify_rows: Mutex<Vec<usize>>,
    /// Per-sequence recurrence operands of the block currently in flight, so a
    /// partial accept can replay the prefix it kept. See [`super::spec`].
    verify_stash: Mutex<Option<VerifyStash>>,
    /// Sequences whose scored rows this wave must also hand back as
    /// **post-final-norm hiddens** — the MTP head's seed. Set for the duration
    /// of one forward, alongside `verify_rows`.
    hidden_seqs: Mutex<Vec<usize>>,
    /// Per-sequence buffers the sweep writes those hiddens into, `[rows, hidden]`
    /// in the wave's activation dtype. Sized before the forward opens, for the
    /// same reason the verify stash is.
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
        let theta = model.cfg.rope_theta;
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
            adapters: HashMap::new(),
            kv_factors,
            draft,
            kv_map,
            rotary,
            inv_freq,
            rope_cs: Mutex::new(None),
            provenance,
            recurrent: Mutex::new(HashMap::new()),
            verify_rows: Mutex::new(Vec::new()),
            verify_stash: Mutex::new(None),
            hidden_seqs: Mutex::new(Vec::new()),
            verify_hidden: Mutex::new(HashMap::new()),
            capture_rows: Mutex::new(HashMap::new()),
            seed: Mutex::new(HashMap::new()),
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

    /// The sequences this wave must score every prefill row of. Empty on every
    /// wave but a speculative verify.
    pub fn verify_row_seqs(&self) -> Result<Vec<usize>> {
        Ok(self
            .verify_rows
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: verify_rows lock poisoned".into()))?
            .clone())
    }

    /// Name the verifying sequences for the next forward.
    pub fn set_verify_row_seqs(&self, seqs: &[usize]) -> Result<()> {
        *self
            .verify_rows
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: verify_rows lock poisoned".into()))? =
            seqs.to_vec();
        Ok(())
    }

    /// Lay out this step's verify cohort: size the shared stash buffers for the
    /// cohort's total rows and record each sequence's span of them.
    ///
    /// **Called before the forward opens**, never inside it: the stash is
    /// device memory and a wave's storage is claimed up front, so an allocation
    /// from inside the sweep is refused by the arena. Buffers are reused across
    /// steps and only reallocated when a wider cohort arrives.
    pub fn begin_verify_stash(&self, blocks: &[(usize, usize)]) -> Result<()> {
        let total: usize = blocks.iter().map(|&(_, len)| len).sum();
        let mut slot = self
            .verify_stash
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: verify_stash lock poisoned".into()))?;
        let grow = match slot.as_ref() {
            Some(s) => s.capacity()? < total,
            None => true,
        };
        if grow {
            // **Release the old stash before claiming the new one.** Both are
            // carved from the reservation now, not the driver, so building the
            // replacement in place would hold two cohorts' regions at once at
            // exactly the moment the wider one is hardest to satisfy — ~302 MiB
            // asked for while ~241 MiB is still held, at 20 contexts and budget
            // 4 on the 27B. Dropping first costs nothing: `grow` has already
            // decided these buffers are too small to keep, and `begin` below
            // rebuilds every span.
            drop(slot.take());
            *slot = Some(VerifyStash::new(
                &self.model.cfg.layer_kinds,
                &self.model.cfg.delta_net,
                total,
                &self.model.device,
            )?);
        }
        slot.as_mut().expect("just ensured").begin(blocks)
    }

    /// Take the cohort stash for the sweep or the replay. Taking rather than
    /// borrowing: a stash span is good for exactly one rewind, and a second use
    /// would replay from a state two waves old — the taker removes the spans it
    /// consumed before putting the buffers back.
    pub fn take_verify_stash(&self) -> Result<Option<VerifyStash>> {
        Ok(self
            .verify_stash
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: verify_stash lock poisoned".into()))?
            .take())
    }

    /// Return the cohort stash — buffers always, spans as the taker left them.
    pub fn put_verify_stash(&self, stash: VerifyStash) -> Result<()> {
        *self
            .verify_stash
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: verify_stash lock poisoned".into()))? =
            Some(stash);
        Ok(())
    }

    /// Pack recurrent stores toward the low end of the span so the region
    /// watermark falls, and report how many moved.
    ///
    /// **Highest first.** A store is relocated into the lowest ground standing
    /// free, and `relocate_down` refuses a move that would not be leftward — so
    /// taking the stores in descending address order gives the ground to the one
    /// stranding the most behind it. Ascending order would let a store already
    /// near the bottom take the low ground and leave the high one exactly where
    /// it was.
    ///
    /// Runs between forwards: a region may only be claimed there, and
    /// `RegionBump` refuses to open its window inside one.
    #[cfg(feature = "cuda")]
    pub fn compact_recurrent(&self) -> Result<usize> {
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        let mut order: Vec<(u64, usize)> = map
            .iter()
            .filter_map(|(seq, store)| store.top_base().map(|top| (top, *seq)))
            .collect();
        order.sort_unstable_by_key(|(top, _)| std::cmp::Reverse(*top));
        // **Report the candidates, not just the moves.** A pass that considered
        // stores and moved none looks identical from the caller to a pass that
        // never ran, and the two have opposite fixes — the first is a refusal
        // worth explaining, the second is a trigger that never fires. The first
        // build of this shipped without the distinction and cost a run to spot.
        let considered = order.len();
        // Read once for the pass — see `relocate_down` for why not per store.
        let free_regions = candle_nn::kv_cache::region_stats(0).map_or(0, |s| s.free);
        let mut moved = 0usize;
        for (_, seq) in order {
            let Some(store) = map.get_mut(&seq) else {
                continue;
            };
            if store.relocate_down(free_regions)? {
                moved += 1;
            }
        }
        if considered > 0 {
            tracing::debug!(
                target: "candle_transformers::qwen35",
                considered,
                moved,
                "span compaction: recurrent stores packed",
            );
        }
        Ok(moved)
    }

    /// Drop the stash spans of `seqs` without replaying — for a verify forward
    /// that failed, whose spans would otherwise rewind a wave that never
    /// committed. The buffers stay for reuse.
    pub fn drop_verify_stashes(&self, seqs: &[usize]) {
        if let Ok(mut slot) = self.verify_stash.lock() {
            if let Some(st) = slot.as_mut() {
                for s in seqs {
                    st.remove(*s);
                }
                // **A stash with no spans left is holding ground for nobody.**
                // The buffers are meant to outlive a span — they are kept across
                // steps and reallocated only for a wider cohort — but not to
                // outlive every span. They are carved from the reservation, so
                // once the last sequence goes they pin regions the weight side
                // could be using, for the life of the process, and no arena
                // sweep can see them because they are not KV. Measured on the
                // 27B: one region left behind by every config, on top of the
                // twenty the recurrent stores held.
                if st.is_unused() {
                    *slot = None;
                }
            }
        }
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
            self.model.cfg.rope_theta,
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
            // Loud: every sequence here was admitted against the span by
            // `reserve_recurrent`, so a refusal now is a fault, not an answer.
            if !self.materialise_recurrent(&mut map, seq, offset, false)? {
                candle::bail!(
                    "qwen35: sequence {seq} was admitted to the wave but the span refused \
                     its recurrent state between admission and the forward"
                );
            }
        }

        // **A store keeps its write buffers for as long as its sequence is
        // admitted.** They used to be given back when the sequence sat out a
        // wave and re-taken here when it next rode one — a region claim inside
        // the forward, before the transient tier is placed, for a sequence the
        // scheduler had already priced at the full store. Those claims came off
        // the top of the free list, the frontier rose under the tier the fill
        // had just sized the wave against, and the placement was refused by
        // the regions they took: run 9, five regions in one millisecond, no
        // forward. Admission prices every sequence at `recurrent_store_bytes`
        // — both halves — and a sequence that is admitted has not lost its
        // admittance by sitting out one wave, so the buffers stay. What bounds
        // residency is the turn seal: `evict_recurrent` drops the whole store
        // once the substrate snapshot is durable, and the next turn restores
        // it at its own admission.
        Ok(())
    }

    /// Put `seq`'s recurrent state on the device, complete with its write
    /// buffers, ready for a wave standing at `offset`.
    ///
    /// See [`needs_reset`] for the offset-0 rule.
    ///
    /// The one place a store is created, reset, unparked or given its write
    /// half — shared by the admission probe ([`Self::reserve_recurrent`]) and
    /// the wave path ([`Self::ensure_recurrent`]), so the two cannot disagree
    /// about what "resident" means.
    ///
    /// `quiet` selects how a span refusal comes back: `Ok(false)` for the
    /// probe, where "no room" is the expected answer to "one more?", or an
    /// error with the partition dumped for the wave path, where the sequence
    /// was already admitted and a refusal is a fault.
    ///
    /// A refusal leaves the map consistent. A store that could not be created
    /// is not inserted; one that could not complete its write buffers keeps
    /// the layers it did fill and resumes from there next time.
    fn materialise_recurrent(
        &self,
        map: &mut HashMap<usize, RecurrentStateStore>,
        seq: usize,
        offset: usize,
        quiet: bool,
    ) -> Result<bool> {
        let fresh = || -> Result<Option<RecurrentStateStore>> {
            let kinds = &self.model.cfg.layer_kinds;
            let dims = &self.model.cfg.delta_net;
            let device = &self.model.device;
            if quiet {
                RecurrentStateStore::try_new(kinds, dims, device)
            } else {
                RecurrentStateStore::new(kinds, dims, device).map(Some)
            }
        };
        let backups = |store: &mut RecurrentStateStore| -> Result<bool> {
            if quiet {
                store.try_ensure_backups()
            } else {
                store.ensure_backups().map(|()| true)
            }
        };
        match map.entry(seq) {
            Entry::Vacant(slot) => {
                let Some(mut store) = fresh()? else {
                    return Ok(false);
                };
                // A vacant slot is a sequence with no state anywhere — either it
                // has never run a wave, or its state was evicted at the last
                // turn's seal. Both start from the sequence-start value here;
                // the second is carried across by `restore_recurrent`, which the
                // scheduler runs from the substrate snapshot before this.
                //
                // **A fresh store already holds the sequence-start value, so it
                // is seeded.** Without the flag the wave's own `ensure_recurrent`
                // found a store standing at offset 0, judged it unseeded, and
                // remade it — a second 160 MiB store claimed *inside the
                // forward*, before the transient tier is placed, with the first
                // freed only afterwards. Those regions came off the top of the
                // free list, moved the arena frontier up, and the tier the fill
                // had just measured the gap for was then two to four regions
                // short: twenty-one refused placements in eight minutes of run
                // 7, every one a lost wave. The flag is consumed by that first
                // wave exactly as a restore's is, so a later genuine offset-0
                // reset on a recycled slot still happens.
                store.mark_seeded();
                let ready = backups(&mut store)?;
                slot.insert(store);
                Ok(ready)
            }
            Entry::Occupied(mut slot) => {
                if needs_reset(slot.get_mut(), offset, quiet) {
                    let Some(store) = fresh()? else {
                        return Ok(false);
                    };
                    slot.insert(store);
                }
                // Every sequence of the coming wave takes its write buffer
                // here — this is the gap between forwards, the only window
                // in which a region may be claimed. Idempotent, so a store
                // that has already run costs one check per layer.
                backups(slot.into_mut())
            }
        }
    }

    /// The admission probe: make `seq`'s recurrent state resident for a wave
    /// standing at `offset`, or report that the span has no room for it.
    ///
    /// **This is the real allocation, asked one sequence at a time.** The
    /// scheduler's wave fill calls it for each decode and prefill it would
    /// admit; `false` is the span saying the wave is as wide as this model can
    /// carry, and the sequence waits for the next one. Runs between forwards,
    /// which is the only window a region may be claimed in.
    ///
    /// Idempotent and cheap for a sequence already resident with its write
    /// buffers in place — one map lookup and a per-layer check.
    pub fn reserve_recurrent(&self, seq: usize, offset: usize) -> Result<bool> {
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        self.materialise_recurrent(&mut map, seq, offset, true)
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
        // A stash names a rewind point inside the state that just went away,
        // and the draft head's history belongs to the same sequence.
        self.drop_verify_stashes(&[seq]);
        self.release_draft(seq);
        Ok(())
    }

    /// Give `child` a copy of `parent`'s recurrent state.
    ///
    /// Reservation bytes every live sequence's recurrent state holds together.
    ///
    /// Summed over the map rather than derived from a per-sequence constant:
    /// a store's region count depends on how its buffers packed, and a forked
    /// child's need not match its parent's. A poisoned lock reports zero rather
    /// than failing — this is a report, and a wrong number in it is preferable
    /// to a scheduler that cannot answer how much memory it is using.
    pub fn recurrent_reserved_bytes(&self) -> usize {
        self.recurrent
            .lock()
            .map(|m| m.values().map(|s| s.reserved_bytes()).sum())
            .unwrap_or(0)
    }

    /// What one sequence's state costs — the widest store standing, or zero
    /// when none is.
    ///
    /// Every store has the same geometry, so any of them answers for all; the
    /// widest is taken because a store that has released its write buffers
    /// (`release_backups`, after sitting out enough waves) reports the read
    /// half only, and admission is pricing a store that will need both.
    ///
    /// Zero before the first store exists is not a gap: with nothing resident
    /// the engine has nothing active, and admission at that point is
    /// unconditional by design — see `admit::gate::may_admit`.
    pub fn recurrent_store_bytes(&self) -> usize {
        self.recurrent
            .lock()
            .map(|m| m.values().map(|s| s.reserved_bytes()).max().unwrap_or(0))
            .unwrap_or(0)
    }

    /// The turn loop carves a child slot per turn and decodes on it, borrowing
    /// the parent's KV blocks zero-copy. State cannot be borrowed the same way
    /// — the child advances it — so it is copied device-to-device
    /// ([`RecurrentStateStore::fork_from`]), and the copy is seeded so the
    /// child's first wave does not reset it back to zero.
    ///
    /// Errors when the parent carries no state: a fork of nothing is a caller
    /// bug, and returning quietly would hand the child zeros — which is the
    /// defect this whole path exists to remove, reintroduced as an error path.
    ///
    pub fn fork_recurrent(&self, parent: usize, child: usize) -> Result<()> {
        // The child's state is written from scratch, so any span standing
        // against that id belongs to whatever last used it — slot ids are
        // recycled — and would rewind the new sequence into a stranger's state.
        // The parent is untouched here and keeps its span.
        self.drop_verify_stashes(&[child]);
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        if let Some(store) = map.get(&parent) {
            let forked = store.fork_from()?;
            map.insert(child, forked);
            return Ok(());
        }
        Err(candle::Error::Msg(format!(
            "qwen35: fork_recurrent from sequence {parent}, which carries no \
             recurrent state — handing {child} zeros here is the amnesia this \
             path exists to prevent"
        )))
    }

    /// Move `child`'s state onto `parent` — the linear join at `finalize_view`.
    ///
    /// A move, so the child's entry is gone afterwards and the parent's old
    /// state is dropped. That is correct precisely because a view is a linear
    /// continuation: it entered with a copy of the parent's state and advanced
    /// it over the turn's tokens, so what it holds now is what the parent's
    /// state becomes.
    ///
    /// Errors when the child carries none, for the same reason as
    /// [`Self::fork_recurrent`]: silently leaving the parent's stale state in
    /// place would lose the turn without saying so.
    ///
    pub fn move_recurrent(&self, child: usize, parent: usize) -> Result<()> {
        // Both sequences' rewind points die here: the parent's state is replaced
        // by the child's, and the child's entry ceases to exist. A span left on
        // either names a state that is gone — see `restore_recurrent` for what a
        // surviving span costs the waves that follow.
        self.drop_verify_stashes(&[child, parent]);
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        map.remove(&parent);
        if let Some(mut store) = map.remove(&child) {
            // A moved state was put there deliberately, which is precisely what
            // `seeded` records — the same mark `fork_from` set when the view's
            // state was a copy. Without it a destination standing at offset 0
            // would read `needs_reset` as true on its first wave and wipe the
            // state this move just handed it; the flag protects exactly that one
            // wave and is consumed by it.
            store.mark_seeded();
            map.insert(parent, store);
            return Ok(());
        }
        Err(candle::Error::Msg(format!(
            "qwen35: move_recurrent from sequence {child}, which carries no \
             recurrent state — the turn's decode would be silently lost"
        )))
    }

    /// Read a sequence's state back as the snapshot record's layer rows.
    ///
    /// `None` when the sequence carries no state — a slot that has never run a
    /// wave has nothing worth persisting, and writing a zero snapshot would be
    /// worse than writing none: resume would install it and report success.
    ///
    pub fn export_recurrent(&self, seq: usize) -> Result<Option<(u64, Vec<ExportedLayerState>)>> {
        let map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        if let Some(store) = map.get(&seq) {
            // `export` refuses mid-wave itself; the seal runs outside the wave,
            // so reaching that error means the ordering broke, not that the
            // caller needs a retry.
            let layers = store.export()?;
            // **Bracket the substrate round trip.** The exported rows are host
            // bytes, so checking them is free — no readback, no sync — and it
            // separates "we persisted garbage" from "the round trip made
            // garbage", which a downstream NaN cannot distinguish.
            if let Some((layer, n)) = first_non_finite_layer(&layers) {
                tracing::error!(
                    target: "candle_transformers::qwen35",
                    seq, layer, non_finite = n,
                    "recurrent EXPORT is non-finite — the state was already bad on \
                     the device, so the corruption happened during the wave and \
                     the substrate is about to be given garbage",
                );
            }
            return Ok(Some((store.schedule_hash(), layers)));
        }
        Ok(None)
    }


    /// Scatter a snapshot into a sequence's state — the resume path.
    ///
    /// Validates the schedule hash and every layer's geometry first
    /// ([`validate_snapshot`]), so a foreign or torn snapshot is refused here,
    /// loudly, and never installed.
    ///
    /// **Restores land on the device.** A sequence already resident imports in
    /// place; one whose state was evicted at its last seal gets a store created
    /// here and the snapshot scattered into it — seeded, so the first wave's
    /// `offset == 0` reset does not undo it.
    ///
    /// Resume runs at `create_sequence`, which is outside a wave and therefore
    /// the window in which the span may be claimed from. A refusal is an error
    /// rather than a deferral: there is nowhere else for the state to wait, and
    /// a restore that quietly did not happen is a conversation that resumes
    /// fluent and having forgotten.
    pub fn restore_recurrent(
        &self,
        seq: usize,
        schedule_hash: u64,
        layers: &[ExportedLayerState],
    ) -> Result<()> {
        validate_snapshot(
            &self.model.cfg.layer_kinds,
            &self.model.cfg.delta_net,
            schedule_hash,
            layers,
        )?;
        // **The state a stash span rewinds into is about to be overwritten.**
        // A span names a rewind point *inside* this sequence's recurrent state;
        // importing a snapshot replaces that state wholesale, so the point it
        // names is gone. Left behind, the span outlives its cohort, and the next
        // wave that takes the stash finds a span for a sequence it does not
        // carry — which fails that whole wave, not just this sequence, for as
        // long as the span survives.
        //
        // After `validate_snapshot`, so a rejected snapshot leaves a valid stash
        // alone, and before the locks below, because `drop_verify_stashes` takes
        // the stash lock and the two must never be held in both orders.
        self.drop_verify_stashes(&[seq]);
        // The other half of the bracket in `export_recurrent`: if the bytes
        // coming back from the substrate are already non-finite then the round
        // trip is faithful and the fault is upstream of the seal.
        if let Some((layer, n)) = first_non_finite_layer(layers) {
            tracing::error!(
                target: "candle_transformers::qwen35",
                seq, layer, non_finite = n,
                "recurrent IMPORT is non-finite — the substrate handed back \
                 garbage, so what was persisted was already bad",
            );
        }
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        if let Some(store) = map.get_mut(&seq) {
            return store.import(schedule_hash, layers);
        }
        let mut store = RecurrentStateStore::new(
            &self.model.cfg.layer_kinds,
            &self.model.cfg.delta_net,
            &self.model.device,
        )?;
        // Import before insert: a store that could not take the snapshot must
        // not be left in the map, where the next wave would advance it from
        // zeros and call that the conversation's history.
        store.import(schedule_hash, layers)?;
        map.insert(seq, store);
        Ok(())
    }

    /// Whether a sequence's recurrent state stands on the device right now.
    ///
    /// Answers both questions a caller can ask, because state exists only on
    /// the device: the admission probe's ("a resident sequence costs the wave
    /// nothing to carry, a new one costs a store") and the fork/move/restore
    /// wiring's ("does this slot carry state at all", which must not fork from
    /// a slot the model has never seen, nor restore over state that exists).
    /// Those were two predicates while an idle sequence's state could sit on
    /// the host; with nothing parked they cannot disagree.
    pub fn recurrent_resident(&self, seq: usize) -> Result<bool> {
        Ok(self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?
            .contains_key(&seq))
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

    // ── The MTP draft head ────────────────────────────────────────────────
    //
    // `super::mtp` is the head itself and `super::draft` is how the engine runs
    // it; this is the per-sequence bookkeeping between them — one seed hidden
    // per sequence, and the KV layer the head writes.

    /// Whether this model carries a draft head at all.
    pub fn has_drafter(&self) -> bool {
        self.model.mtp.is_some()
    }

    /// This checkpoint's draft budget for a wave of `width` sequences.
    ///
    /// Gated on the head actually being loaded, not just on the ladder being
    /// non-empty: a checkpoint converted without the NextN tensors would
    /// otherwise have every wave pay a drafting call that can only return
    /// nothing.
    pub fn draft_budget_for(&self, width: usize) -> usize {
        if !self.has_drafter() {
            return 0;
        }
        // **Bounded by what its own rewind machinery costs.**
        //
        // A verify block of `k` proposals makes each sequence stash `k + 1` rows
        // of post-projection operands for every DeltaNet layer, because a
        // recurrence cannot be truncated and has to be replayed (`super::spec`).
        // So the stash is `width × (k + 1)` rows, and the ladder alone prices
        // only the *throughput* side of `k` — it has no term for the memory the
        // rewind needs.
        //
        // Measured on the 27B at 20 contexts: 60.4 MiB of stash at budget 0
        // against 301.8 MiB at budget 4. Widening the ladder to reach width 20
        // therefore took a config that fit and made it not fit, and the failure
        // arrived as a device OOM rather than as a refusal — the stash is
        // allocated between forwards, where nothing is left to concede to.
        //
        // Clamping here rather than refusing later is what lets the ladder ask
        // for depth at *any* width: a narrow cohort still gets the full budget,
        // and a wide one gets the deepest budget its stash can afford instead of
        // the whole wave failing. Speculation is lossless, so a shallower budget
        // costs throughput and never a token.
        self.draft
            .budget(width)
            .min(self.affordable_draft_budget(width))
    }

    /// The deepest budget whose rewind stash the KV side can currently hold.
    ///
    /// `usize::MAX` when the question does not arise — no reservation to measure
    /// (a CPU device or a test), or a model with no recurrent layers to stash.
    /// The caller `min`s with the ladder, so an unmeasurable bound never *raises*
    /// a budget.
    fn affordable_draft_budget(&self, width: usize) -> usize {
        let dims = &self.model.cfg.delta_net;
        let layers = self
            .model
            .cfg
            .layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::DeltaNet)
            .count();
        // The four operands `SpanOperands` holds, per row, per DeltaNet layer.
        let per_row = (dims.conv_dim() + dims.value_dim() + 2 * dims.n_v_heads)
            * std::mem::size_of::<f32>()
            * layers;
        if per_row == 0 || width == 0 {
            return usize::MAX;
        }
        let Some(budget) = candle_nn::kv_cache::vram_budget_available(&self.model.device) else {
            return usize::MAX;
        };
        // Half of what is claimable, not all of it: the stash is one tenant among
        // several and a wave that spends every free region on its own rewind
        // buffer has nothing left to decode into.
        let rows = (budget / 2) / per_row;
        (rows / width).saturating_sub(1)
    }

    /// The KV layer the draft head writes, past every trunk attention layer.
    /// `None` on a checkpoint without a head, where no such layer is allocated.
    pub fn mtp_kv_layer(&self) -> Option<usize> {
        mtp_kv_layer(&self.model.cfg)
    }

    /// The KV layers a wave over `[layer_start, layer_end)` touches — see
    /// [`wave_kv_range`], which is where the rule and its test live.
    pub fn kv_layer_range(&self, layer_start: usize, layer_end: usize) -> (usize, usize) {
        wave_kv_range(&self.model.cfg, layer_start, layer_end)
    }

    /// Sequences the current forward must hand back post-final-norm hiddens
    /// for. Empty on every wave but a speculative one.
    pub fn hidden_capture_seqs(&self) -> Result<Vec<usize>> {
        Ok(self
            .hidden_seqs
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: hidden_seqs lock poisoned".into()))?
            .clone())
    }

    /// Name them for the next forward, and size the buffers it will write.
    ///
    /// **Sized here, before the forward opens** — same rule as the verify
    /// stash: a wave's storage is claimed up front and the arena refuses a
    /// device allocation from inside the sweep.
    ///
    /// `act_dtype` is the wave's activation dtype, and the buffers are
    /// allocated in it rather than in F32. Everything on both sides of these
    /// rows already speaks that type — the trunk's `final_norm` output that
    /// fills them, and the head's input assembly that reads them back as a
    /// seed — so an F32 buffer bought nothing but a conversion at each end,
    /// one launch per sequence per wave in each direction.
    pub fn arm_hidden_capture(&self, seqs: &[(usize, usize)], act_dtype: DType) -> Result<()> {
        let hidden = self.model.cfg.hidden_size;
        {
            let mut buf = self
                .verify_hidden
                .lock()
                .map_err(|_| candle::Error::Msg("qwen35: verify_hidden lock poisoned".into()))?;
            for &(seq, rows) in seqs {
                // Width AND type: a buffer carried over from a session at a
                // different activation dtype is the wrong shape for this wave
                // however many rows it has.
                let fits = buf
                    .get(&seq)
                    .is_some_and(|t| t.dtype() == act_dtype && t.dim(0).is_ok_and(|n| n >= rows));
                if !fits {
                    buf.insert(
                        seq,
                        Tensor::zeros((rows, hidden), act_dtype, &self.model.device)?,
                    );
                }
            }
        }
        {
            let mut rows = self
                .capture_rows
                .lock()
                .map_err(|_| candle::Error::Msg("qwen35: capture_rows lock poisoned".into()))?;
            for &(seq, n) in seqs {
                rows.insert(seq, n);
            }
        }
        *self
            .hidden_seqs
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: hidden_seqs lock poisoned".into()))? =
            seqs.iter().map(|&(s, _)| s).collect();
        Ok(())
    }

    /// Clear the ACTIVE capture set — the buffers stay, and so does the record
    /// of how many rows each of them holds.
    ///
    /// The two are cleared at different times on purpose. The active set gates
    /// the sweep's capture and must be empty the moment the forward ends, or an
    /// ordinary prefill would write hiddens into buffers sized for someone
    /// else's block. The row counts are read by the accept, which runs *after*
    /// the forward, so clearing them here would take away the one number that
    /// says how much of the buffer the wave actually filled.
    pub fn disarm_hidden_capture(&self) {
        if let Ok(mut g) = self.hidden_seqs.lock() {
            g.clear();
        }
    }

    /// How many rows the last armed wave captured for `seq`.
    ///
    /// `None` when no wave has captured for it. Survives
    /// [`Self::disarm_hidden_capture`] — see there for why.
    pub fn captured_rows(&self, seq: usize) -> Result<Option<usize>> {
        Ok(self
            .capture_rows
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: capture_rows lock poisoned".into()))?
            .get(&seq)
            .copied())
    }

    /// The buffer the sweep writes `seq`'s hiddens into.
    pub fn hidden_buffer(&self, seq: usize) -> Result<Option<Tensor>> {
        Ok(self
            .verify_hidden
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: verify_hidden lock poisoned".into()))?
            .get(&seq)
            .cloned())
    }

    /// A sequence's seed hidden — the trunk's post-`final_norm` output at its
    /// last accepted position. `None` before its first wave.
    pub fn draft_seed(&self, seq: usize) -> Result<Option<Tensor>> {
        Ok(self
            .seed
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: seed lock poisoned".into()))?
            .get(&seq)
            .cloned())
    }

    /// Propose up to `max_len` tokens after `committed` for each sequence.
    ///
    /// Empty — a plain decode step — when there is no head, or when the
    /// sequence has no seed yet (the first step after prefill, before any wave
    /// has scored a row for it). A seedless sequence is left out rather than
    /// drafted from zeros, which would propose noise and waste the verify row
    /// it costs.
    pub fn mtp_draft(
        &self,
        session: &mut BatchedInferenceSession,
        seqs: &[usize],
        committed: &[u32],
        max_len: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let empty = || vec![Vec::new(); seqs.len()];
        if !self.has_drafter() || max_len == 0 || seqs.is_empty() {
            return Ok(empty());
        }
        if committed.len() != seqs.len() {
            candle::bail!(
                "qwen35 mtp: {} committed tokens for {} sequences",
                committed.len(),
                seqs.len()
            );
        }

        let want = session.activation_dtype();
        let (draftable, seeds): (Vec<usize>, Vec<Tensor>) = {
            let mut map = self
                .seed
                .lock()
                .map_err(|_| candle::Error::Msg("qwen35: seed lock poisoned".into()))?;
            let mut idx = Vec::with_capacity(seqs.len());
            let mut seeds = Vec::with_capacity(seqs.len());
            for (i, seq) in seqs.iter().enumerate() {
                match map.get(seq) {
                    // A seed captured under a DIFFERENT activation dtype belongs
                    // to a session that no longer exists — the harness builds one
                    // session per config and a sequence index is reused across
                    // them, so the seed left behind is a previous run's hidden.
                    // Concatenating it with this run's embedding is a dtype
                    // mismatch, and the sequence simply drafts nothing until the
                    // next wave captures a fresh one.
                    Some(s) if s.dtype() != want => {
                        map.remove(seq);
                    }
                    Some(s) => {
                        idx.push(i);
                        seeds.push(s.clone());
                    }
                    None => {}
                }
            }
            (idx, seeds)
        };
        if draftable.is_empty() {
            return Ok(empty());
        }

        let cohort: Vec<usize> = draftable.iter().map(|&i| seqs[i]).collect();
        let toks: Vec<u32> = draftable.iter().map(|&i| committed[i]).collect();
        let drafted = draft_cohort(self, session, &cohort, &toks, &seeds, max_len)?;

        let mut out = empty();
        for (k, &i) in draftable.iter().enumerate() {
            out[i] = drafted[k].clone();
        }
        Ok(out)
    }

    /// Take each sequence's next seed from the wave that just ran: the trunk's
    /// hidden at the LAST ACCEPTED position of its block.
    ///
    /// `jobs` pairs a sequence with how many of its rows the accept kept, so
    /// row `kept - 1` of the buffer the wave captured is the one. Which row
    /// that is cannot be known during the wave — it is decided by comparing the
    /// target's argmaxes against the proposal — which is why the wave captures
    /// the whole block and this picks from it.
    ///
    /// **`None` means "every row this wave captured", and it is not a synonym
    /// for one.** A caller that kept the whole block knows it kept the whole
    /// block; it does not necessarily know how long the block was, because the
    /// accept classifies by *offset* and only a rewind carries a row count.
    /// Asking here rather than assuming is the difference between seeding from
    /// the last accepted row and seeding from the block's first — an error that
    /// changes no token, because verify keeps only the target's own argmaxes,
    /// and shows up solely as acceptance decaying toward 1.00.
    ///
    /// Nothing else is caught up here. The head's KV took every one of those
    /// positions inside the wave, as a layer, and the rejected tail truncates
    /// away with every other layer's.
    pub fn mtp_take_seeds(&self, jobs: &[(usize, Option<usize>)]) -> Result<()> {
        if !self.has_drafter() {
            return Ok(());
        }
        let mut picked: Vec<(usize, Tensor)> = Vec::with_capacity(jobs.len());
        for &(seq, kept) in jobs {
            let Some(rows) = self.hidden_buffer(seq)? else {
                continue;
            };
            // The wave's own row count, never the buffer's: buffers only grow,
            // so `dim(0)` is a high-water mark from whichever earlier block was
            // widest.
            let Some(filled) = self.captured_rows(seq)? else {
                continue;
            };
            let kept = kept.unwrap_or(filled);
            if kept == 0 {
                continue;
            }
            if kept > filled || filled > rows.dim(0)? {
                candle::bail!(
                    "qwen35 mtp: sequence {seq} accepted {kept} of {filled} captured rows \
                     in a {}-row buffer",
                    rows.dim(0)?
                )
            }
            picked.push((seq, rows.narrow(0, kept - 1, 1)?.contiguous()?));
        }
        let mut map = self
            .seed
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: seed lock poisoned".into()))?;
        for (seq, seed) in picked {
            map.insert(seq, seed);
        }
        Ok(())
    }

    /// Drop a sequence's draft state.
    pub fn release_draft(&self, seq: usize) {
        if let Ok(mut m) = self.seed.lock() {
            m.remove(&seq);
        }
        if let Ok(mut m) = self.verify_hidden.lock() {
            m.remove(&seq);
        }
        if let Ok(mut m) = self.capture_rows.lock() {
            m.remove(&seq);
        }
    }

    /// Re-advance `seq`'s recurrent state over the first `kept` tokens of the
    /// block it just verified — the speculative rewind ([`super::spec`]).
    ///
    /// Runs against the store in place: the replay is a handful of small
    /// launches per recurrent layer, nothing like the whole-wave `&mut` a
    /// sweep needs, so there is no reason to lift the store out.
    /// Replay every job's accepted prefix — the whole cohort's recurrent
    /// rewinds in one batched launch pair per DeltaNet layer.
    ///
    /// `jobs` pairs each stash span with how many of its rows the accept kept.
    /// They must ascend by stash row, which they do when taken in the order the
    /// stash recorded them.
    pub fn replay_recurrent(&self, stash: &VerifyStash, jobs: &[(StashSpan, usize)]) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        // One pass over the map collects a distinct `&mut` per job's store —
        // the borrows are disjoint because the sequences are.
        let mut stores: HashMap<usize, &mut RecurrentStateStore> = map
            .iter_mut()
            .filter(|(seq, _)| jobs.iter().any(|(sp, _)| sp.seq == **seq))
            .map(|(seq, st)| (*seq, st))
            .collect();
        let mut full: Vec<(StashSpan, usize, &mut RecurrentStateStore)> =
            Vec::with_capacity(jobs.len());
        for &(span, kept) in jobs {
            let store = stores.remove(&span.seq).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "qwen35: sequence {} has no recurrent state — \
                     `ensure_recurrent` was not called for this wave",
                    span.seq
                ))
            })?;
            full.push((span, kept, store));
        }
        // The **residues**, not the layers. The replay runs at accept time,
        // well after the sweep that captured the stash, so on a streamed
        // checkpoint the layer's image may long since have been evicted — and
        // `ensure`ing it would pull ~240 MB over PCIe to read four small
        // constants that never left VRAM. The residue holds exactly those four.
        let recurrent: Vec<usize> = match full.first() {
            Some((_, _, store)) => store.recurrent_layer_indices().collect(),
            None => return Ok(()),
        };
        let mut residues = Vec::with_capacity(recurrent.len());
        for &li in &recurrent {
            residues.push((li, self.model.layers.residue(li)?));
        }
        let layers: Vec<ReplayLayer<'_>> = residues
            .iter()
            .map(|(li, r)| {
                let w = r.delta_net().map_err(|_| {
                    candle::Error::Msg(format!(
                        "qwen35 verify replay: layer {li} carries recurrent state but is \
                         not DeltaNet"
                    ))
                })?;
                Ok(ReplayLayer {
                    layer_index: *li,
                    consts: DeltaNetConstants {
                        dt_bias: &w.dt_bias,
                        a: &w.a,
                        conv: &w.conv,
                        norm: &w.norm,
                    },
                })
            })
            .collect::<Result<_>>()?;
        replay_accepted_prefixes(
            &layers,
            &self.model.cfg.delta_net,
            self.model.cfg.rms_norm_eps,
            &self.model.device,
            stash,
            &mut full,
        )
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
    pub fn maybe_change_dtype(&self, dtype: DType, kv_dtype: DType) -> Result<()> {
        // A dtype change means a NEW session — this is the one place that
        // happens, and only a new session can change it. Every piece of draft
        // state is an activation captured under the old one, belonging to a
        // sequence numbering the new session will reuse for something else, so
        // all of it dies here. Nothing is dropped when the dtype is unchanged,
        // which is what a sibling session sharing the backings gets.
        //
        // Not hypothetical: a gate that runs F16 and then BF16 against one
        // loaded model would otherwise hand the draft head an F16 hidden to
        // concatenate with a BF16 embedding.
        let stale = self
            .verify_hidden
            .lock()
            .map(|m| m.values().any(|t| t.dtype() != dtype))
            .unwrap_or(false)
            || self
                .seed
                .lock()
                .map(|m| m.values().any(|t| t.dtype() != dtype))
                .unwrap_or(false);
        if stale {
            if let Ok(mut m) = self.verify_hidden.lock() {
                m.clear();
            }
            if let Ok(mut m) = self.seed.lock() {
                m.clear();
            }
            if let Ok(mut m) = self.capture_rows.lock() {
                m.clear();
            }
        }
        // Through the residue, which is where every norm of every layer lives —
        // resident in both stores, so this reaches a streamed checkpoint's
        // layers without pulling one of them over PCIe. It carries the two
        // widths main introduced here: the residual-stream norms take the
        // activation dtype, Q/K's take the KV arena's.
        for li in 0..self.model.layers.len() {
            self.model
                .layers
                .residue(li)?
                .set_activation_dtype(dtype, kv_dtype)?;
        }
        // The draft head's block runs in the wave's dtype like any other, so
        // its norms are materialised with the trunk's — the head is a layer of
        // this model, not a sidecar that got to keep the loader's dtype.
        if let Some(head) = &self.model.mtp {
            head.block.residue().set_activation_dtype(dtype, kv_dtype)?;
            head.input.enorm.maybe_change_dtype(dtype)?;
            head.input.hnorm.maybe_change_dtype(dtype)?;
            head.head_norm.maybe_change_dtype(dtype)?;
        }
        // LoRA adapters, for exactly the reason the norms are here: their
        // weights are operands of matmuls whose other operand is an activation,
        // and a matmul refuses mismatched dtypes outright.
        //
        // **Three widths, not one**, because a layer does not use a single one:
        //
        // * `dtype` — the residual stream, which is what `gate`/`up` read and
        //   what `down` and the output projection write back into it.
        // * `kv_dtype` — what Q/K/V are projected in, because they become the
        //   arena's contents. Equal to `dtype` unless the model computes wider
        //   than it stores.
        // * the **SwiGLU promotion** — an F16 session runs its MLP intermediates
        //   in BF16, since they can exceed F16's range. That makes `gate`'s
        //   adapter write BF16 from an F16 input, and `down`'s read BF16 and
        //   write F16, inside the same layer.
        //
        // Every one of them is materialised here, at session creation, from the
        // host's F32 master — so nothing converts a weight inside a wave, and a
        // width the forward asks for is always already there. Missing one is not
        // a silent wrong answer: the matmul refuses it by name.
        //
        // Idempotent. A session whose widths match the last moves no bytes; in
        // production, where the stream and the arena agree and nothing is
        // promoted, this is a single resident copy.
        let mut widths = vec![dtype, kv_dtype];
        if dtype == DType::F16 {
            widths.push(DType::BF16);
        }
        widths.dedup();
        for adapter in self.adapters.values() {
            adapter.maybe_change_dtype(&widths, &self.model.device)?;
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
    pub fn create_batched_session(&self, config: BatchedConfig) -> Result<BatchedInferenceSession> {
        let mut config = config;
        config.k_hi_error_threshold_factor *= self.kv_factors.k_hi;
        config.k_low_error_threshold_factor *= self.kv_factors.k_low;
        config.v_hi_error_threshold_factor *= self.kv_factors.v_hi;
        config.v_low_error_threshold_factor *= self.kv_factors.v_low;
        // **This duplicates the generic `create_batched_session`.** It reads
        // `kv_factors` directly rather than `model_core_properties()`, so any
        // per-model property added to that struct lands there and is silently
        // dropped here — a new field looks wired, builds clean, and simply never
        // reaches this model. Extend both when adding one.
        let session = create_session(&self.model.cfg, &self.model.device, config)?;
        self.maybe_change_dtype(session.activation_dtype(), session.kv_live_dtype())?;
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
            // **No.** Three quarters of this stack mixes tokens through a
            // recurrence, and a recurrence cannot compute the output of a token
            // inserted mid-sequence: its state has already accumulated past the
            // hole and there is no way to re-enter it there. The planner has to
            // know before it plans, not discover it from a bail.
            // **Attention layers only.** Three quarters of this stack is
            // recurrent and has no Q in a KV cache to capture, so the fold's
            // `[n − 2, 1, 1]` groups over 10 layers here, not 40.
            provenance_capture_layers: self.kv_map.num_kv_layers(),
            can_gap_fill: false,
            carries_recurrent_state: true,
        }
    }

    /// Rows the KV side can admit, priced against the layers a row actually
    /// occupies.
    ///
    /// The trait's default multiplies the per-row cost by transformer depth,
    /// which on a 3:1 hybrid over-charges by 4× and refuses four times more
    /// prefill than the cache can hold. The count that is right is the
    /// session's, not the layer map's: a checkpoint with an MTP head pages one
    /// layer MORE than the map describes ([`session_kv_layers`]), because the
    /// head's own KV is a layer past every trunk one. Pricing on the map's
    /// eight while a row costs nine admitted 12.5% more prefill than the cache
    /// could hold, and the shortfall did not surface as a refusal — it surfaced
    /// as a chunk that could not be claimed from inside the forward that needed
    /// it, three layers into the sweep, on the widest configuration only.
    pub fn kv_width_cap(&self, act_dtype: DType) -> Option<usize> {
        let stats = candle_nn::kv_cache::region_stats(0)?;
        let free = (stats.free + stats.blocked).saturating_sub(1);
        let per_row = 2 * self.n_kv_head() * self.head_dim() * act_dtype.size_in_bytes();
        let per_row_all_kv_layers =
            per_row.checked_mul(session_kv_layers(&self.model.cfg).ok()?)?;
        if per_row_all_kv_layers == 0 {
            return None;
        }
        let kv_bytes = free.saturating_mul(candle_nn::kv_cache::REGION_BYTES);
        // Never zero: a cap of nought is not a narrow wave but no wave, and
        // once the KV side is full it would be permanent.
        Some((kv_bytes / per_row_all_kv_layers).max(1))
    }
}

/// Whether a store standing at `offset` must be thrown away and remade at the
/// sequence-start value.
///
/// A sequence at offset 0 has no history, so its state must be the
/// sequence-start value — except when the state was put there deliberately by a
/// fork or a restore, which is what the store's seeded flag records. The flag
/// protects exactly **one** offset-0 wave: one that outlived that wave would go
/// on to suppress a later, genuine reset, which is the recycled-slot defect
/// wearing the fix's own clothes.
///
/// `quiet` is what makes "one wave" true. It is set by the admission probe
/// ([`ManagedQwen35::reserve_recurrent`]), which is asked once per sequence per
/// fill and answers `false` whenever the span has no room — so a sequence can be
/// probed many times before it ever rides a wave. The probe therefore *reads*
/// the flag and the wave path *consumes* it. Consuming it in the probe spent it
/// on a question rather than on the wave it guards: a restored store imports
/// (seeded), its write buffers are refused part-way so the probe returns `false`
/// twice, and the third probe finds the flag gone and remakes the state as
/// zeros — after `materialise_recurrent`'s vacant arm has already dropped the
/// host copy. The conversation comes back fluent and amnesiac, which is the
/// exact failure resume exists to remove.
fn needs_reset(store: &mut RecurrentStateStore, offset: usize, quiet: bool) -> bool {
    let seeded = if quiet {
        store.is_seeded()
    } else {
        store.take_seeded()
    };
    offset == 0 && !seeded
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

    /// A restored store, standing at offset 0, is probed as many times as the
    /// span refuses it and keeps its state every time — then rides its wave.
    ///
    /// The probe is not the wave. It answers "is there room for one more?" and
    /// says `false` whenever there is not, so it runs repeatedly for a sequence
    /// that has not run at all; a flag spent there is spent on a question.
    #[test]
    fn the_probe_reads_the_seeded_flag_and_the_wave_consumes_it() -> Result<()> {
        let c = cfg();
        let dev = Device::Cpu;
        let mut store = RecurrentStateStore::new(&c.layer_kinds, &c.delta_net, &dev)?;
        store.mark_seeded();

        for probe in 0..5 {
            assert!(
                !needs_reset(&mut store, 0, true),
                "probe {probe} would have thrown away restored state",
            );
        }
        assert!(
            store.is_seeded(),
            "five probes must leave the flag for the wave that follows",
        );

        // The wave it was guarding: no reset, and the flag is spent.
        assert!(!needs_reset(&mut store, 0, false));
        assert!(!store.is_seeded(), "the wave consumes it");
        // A later offset-0 wave on the same slot is a genuine reset — this is
        // the recycled-slot case the flag must not go on suppressing.
        assert!(needs_reset(&mut store, 0, false));
        Ok(())
    }

    /// Away from offset 0 there is history to keep, so nothing is reset and
    /// the flag is irrelevant — but the wave path still spends it, because the
    /// wave it was reserved for is the one now running.
    #[test]
    fn a_store_with_history_is_never_reset() -> Result<()> {
        let c = cfg();
        let dev = Device::Cpu;
        let mut store = RecurrentStateStore::new(&c.layer_kinds, &c.delta_net, &dev)?;
        assert!(!needs_reset(&mut store, 1, true));
        assert!(!needs_reset(&mut store, 4096, false));
        // An unseeded store at offset 0 is the ordinary case: reset it.
        assert!(needs_reset(&mut store, 0, true));
        Ok(())
    }

    /// A wave writes the buffer it is NOT reading, so the entry state is intact
    /// whether the wave commits or fails — and a commit installs the wave's
    /// output.
    #[test]
    fn recurrent_state_rolls_back_to_the_entry_value() -> Result<()> {
        let c = cfg();
        let dev = Device::Cpu;
        let mut store = RecurrentStateStore::new(&c.layer_kinds, &c.delta_net, &dev)?;
        // The write half is taken between forwards, never inside one, and a
        // store that sits out enough waves gives it back — so every wave path
        // takes it first (`materialise_recurrent`). Without this the store has
        // only the buffer it reads and `layer_state_pair_mut` refuses.
        store.ensure_backups()?;
        // Layer 0 is DeltaNet under the 3:1 schedule.
        let entry = store.layer_state(0)?.s.copy()?;

        // The form the mixer uses: read `live`, write the other buffer.
        let advance = |store: &mut RecurrentStateStore| -> Result<()> {
            let (live, out) = store.layer_state_pair_mut(0)?;
            let bump = Tensor::full(3f32, live.s.shape(), &dev)?;
            out.s.slice_set(&live.s.add(&bump)?, 0, 0)?;
            Ok(())
        };

        store.begin_wave()?;
        advance(&mut store)?;
        let during = store.layer_state(0)?.s.copy()?;
        store.rollback_wave()?;
        let after = store.layer_state(0)?.s.copy()?;

        let max = |t: &candle::Tensor| -> Result<f32> {
            t.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
        };
        assert_eq!(
            max(&during.sub(&entry)?)?,
            0.0,
            "a wave must not write the buffer it is reading"
        );
        assert_eq!(
            max(&after.sub(&entry)?)?,
            0.0,
            "rollback must leave the entry state exactly as it was"
        );

        store.begin_wave()?;
        advance(&mut store)?;
        store.commit_wave();
        let committed = store.layer_state(0)?.s.copy()?;
        assert!(
            max(&committed.sub(&entry)?)? > 1.0,
            "commit must install the wave's output"
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

/// First layer of a snapshot carrying a non-finite value, with how many that
/// layer holds.
///
/// Host-side over the exported LE F32 bytes, so it costs no readback and no
/// sync — which is what makes it usable on both sides of the substrate round
/// trip. A downstream all-NaN logits row cannot say whether the state was
/// already bad when persisted or became bad in transit; checking the same bytes
/// going out and coming back does.
fn first_non_finite_layer(layers: &[ExportedLayerState]) -> Option<(u32, usize)> {
    let count = |bytes: &[u8]| -> usize {
        bytes
            .chunks_exact(4)
            .filter(|b| !f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite())
            .count()
    };
    layers.iter().find_map(|l| {
        let n = count(&l.state) + count(&l.conv_tail);
        (n > 0).then_some((l.layer_index, n))
    })
}
