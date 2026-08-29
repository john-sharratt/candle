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

use super::draft::draft_cohort;
use super::engine::{
    create_session, mtp_kv_layer, provenance_layer_indices, session_kv_layers, wave_geometry,
    wave_kv_range,
};
use super::quantized_weights::{QuantLayer, QuantLayerMix, QuantModel};
use super::spec::{replay_accepted_prefixes, StashSpan, VerifyStash};
use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ModelCoreProperties, ProvenanceLayerIndices,
};
use crate::models::delta_net::ExportedLayerState;
use crate::models::delta_net::KvLayerMap;
use crate::models::delta_net::RecurrentStateStore;
use crate::models::draft_ladder::DraftLadder;
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

    /// Drop the stash spans of `seqs` without replaying — for a verify forward
    /// that failed, whose spans would otherwise rewind a wave that never
    /// committed. The buffers stay for reuse.
    pub fn drop_verify_stashes(&self, seqs: &[usize]) {
        if let Ok(mut slot) = self.verify_stash.lock() {
            if let Some(st) = slot.as_mut() {
                for s in seqs {
                    st.remove(*s);
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
                    // The seeded flag is consumed on the FIRST wave either
                    // way. It protects the one wave that follows a fork or a
                    // restore — the wave whose slot legitimately stands at
                    // offset 0 while holding state that was put there on
                    // purpose. A flag that outlived that wave would go on to
                    // suppress a later, genuine reset, which is the recycled
                    // slot defect wearing the fix's own clothes.
                    let seeded = slot.get_mut().take_seeded();
                    if offset == 0 && !seeded {
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

    /// The turn loop carves a child slot per turn and decodes on it, borrowing
    /// the parent's KV blocks zero-copy. State cannot be borrowed the same way
    /// — the child advances it — so it is copied device-to-device
    /// ([`RecurrentStateStore::fork_from`]), and the copy is seeded so the
    /// child's first wave does not reset it back to zero.
    ///
    /// Errors when the parent carries no state: a fork of nothing is a caller
    /// bug, and returning quietly would hand the child zeros — which is the
    /// defect this whole path exists to remove, reintroduced as an error path.
    pub fn fork_recurrent(&self, parent: usize, child: usize) -> Result<()> {
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        let forked = map
            .get(&parent)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "qwen35: fork_recurrent from sequence {parent}, which carries no \
                     recurrent state — handing {child} zeros here is the amnesia this \
                     path exists to prevent"
                ))
            })?
            .fork_from()?;
        map.insert(child, forked);
        Ok(())
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
    pub fn move_recurrent(&self, child: usize, parent: usize) -> Result<()> {
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        let store = map.remove(&child).ok_or_else(|| {
            candle::Error::Msg(format!(
                "qwen35: move_recurrent from sequence {child}, which carries no \
                 recurrent state — the turn's decode would be silently lost"
            ))
        })?;
        map.insert(parent, store);
        Ok(())
    }

    /// Read a sequence's state back as the snapshot record's layer rows.
    ///
    /// `None` when the sequence carries no state — a slot that has never run a
    /// wave has nothing worth persisting, and writing a zero snapshot would be
    /// worse than writing none: resume would install it and report success.
    pub fn export_recurrent(&self, seq: usize) -> Result<Option<(u64, Vec<ExportedLayerState>)>> {
        let map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        let Some(store) = map.get(&seq) else {
            return Ok(None);
        };
        // `export` refuses mid-wave itself; the seal runs outside the wave, so
        // reaching that error means the ordering broke, not that the caller
        // needs a retry.
        let layers = store.export()?;
        Ok(Some((store.schedule_hash(), layers)))
    }

    /// Scatter a snapshot into a sequence's state — the resume path.
    ///
    /// Creates the store if the slot has none yet (the normal case: resume runs
    /// at `create_sequence`, before any wave). `import` validates the schedule
    /// hash and every layer's geometry before touching a tensor, and marks the
    /// store seeded so the first wave's `offset == 0` reset does not undo it.
    pub fn restore_recurrent(
        &self,
        seq: usize,
        schedule_hash: u64,
        layers: &[ExportedLayerState],
    ) -> Result<()> {
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        let store = match map.entry(seq) {
            std::collections::hash_map::Entry::Occupied(slot) => slot.into_mut(),
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(RecurrentStateStore::new(
                    &self.model.cfg.layer_kinds,
                    &self.model.cfg.delta_net,
                    &self.model.device,
                )?)
            }
        };
        store.import(schedule_hash, layers)
    }

    /// Whether a sequence currently carries recurrent state — for the
    /// scheduler's fork/move wiring, which must not call either on a slot the
    /// model has never seen (a view carved before the parent's first wave).
    pub fn has_recurrent(&self, seq: usize) -> Result<bool> {
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
        if self.has_drafter() {
            self.draft.budget(width)
        } else {
            0
        }
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
        replay_accepted_prefixes(&self.model, stash, &mut full)
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
        let block = |l: &QuantLayer| -> Result<()> {
            l.attn_norm.maybe_change_dtype(dtype)?;
            l.post_attn_norm.maybe_change_dtype(dtype)?;
            if let QuantLayerMix::Attention(a) = &l.mix {
                a.q_norm.maybe_change_dtype(dtype)?;
                a.k_norm.maybe_change_dtype(dtype)?;
            }
            Ok(())
        };
        for layer in &self.model.layers {
            block(layer)?;
        }
        // The draft head's block runs in the wave's dtype like any other, so
        // its norms are materialised with the trunk's — the head is a layer of
        // this model, not a sidecar that got to keep the loader's dtype.
        if let Some(head) = &self.model.mtp {
            block(&head.block)?;
            head.input.enorm.maybe_change_dtype(dtype)?;
            head.input.hnorm.maybe_change_dtype(dtype)?;
            head.head_norm.maybe_change_dtype(dtype)?;
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

    /// A wave writes the buffer it is NOT reading, so the entry state is intact
    /// whether the wave commits or fails — and a commit installs the wave's
    /// output.
    #[test]
    fn recurrent_state_rolls_back_to_the_entry_value() -> Result<()> {
        let c = cfg();
        let dev = Device::Cpu;
        let mut store = RecurrentStateStore::new(&c.layer_kinds, &c.delta_net, &dev)?;
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
