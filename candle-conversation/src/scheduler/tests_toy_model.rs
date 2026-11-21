//! A toy recurrent model for the scheduler tests, whose exact arithmetic makes a
//! lost or duplicated token visible.
//!
//! The toy state is a small matrix advanced by an order-dependent affine update
//! that is exact in f32, so assertions need no tolerance and a plumbing defect
//! shows up as a wrong value. The model carries an auxiliary token count as opaque
//! bytes, and a per-position page list whose boundaries the scheduler decides. Those
//! are the three kinds of state a real stack keeps. Its forks, restores and
//! releases copy and drop both classes of state, and its reset rule matches the
//! real one, including the single reset that a seeded fork suppresses.

use super::*;

// •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
// Scheduler unit tests
// •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
//
// The `DummyModel` is a minimal `ManagedBatchedModel` that returns dummy CPU
// tensors so no GPU is needed. `BatchedInferenceSession` is constructed on
// `Device::Cpu` with tiny arena dimensions so these tests run without CUDA.

#[cfg(test)]
pub(super) mod tests {
    // Expected token totals are written as the sum of the per-item terms they
    // came from, including the zero ones, so the arithmetic under test is
    // visible in the assertion.
    #![allow(clippy::identity_op)]

    use super::*;

    use candle::{DType, Tensor};

    use std::sync::Mutex;

    use candle_transformers::models::batched_inference::{
        BatchedInferenceSession, ManagedBatchedModel,
    };

    use candle_transformers::models::positional_piece::PositionalPiece;

    use candle_transformers::models::wave_driver::{StepOutcome, StepSpec};

    use super::super::tests::positional_ledger::PositionalLedger;

    use super::super::tests_fixtures::tests::{stub_residual, stub_step, DummyModel};

    // —— Recurrent dummy model ————————————————————————————————————————————————

    /// One sequence's toy recurrent state: 2 layers × a flattened 2×2 matrix.
    ///
    /// Deliberately tiny. Every assertion against it is an **exact** comparison
    /// rather than a tolerance — a state that should be `[2.0, 4.0, 6.0, 8.0]`
    /// either is or is not — which is what lets a tier-1 test fail for a
    /// plumbing defect that the real model would only ever express as
    /// slightly different, perfectly fluent prose.
    pub(in super::super) type ToyState = [[f32; 4]; 2];

    /// Ratio of the toy index: tokens per whole row, so a turn boundary that is
    /// not a multiple of it leaves a ragged open block — the case that makes the
    /// index worth persisting separately rather than deriving on restore.
    pub(in super::super) const TOY_INDEX_RATIO: u32 = 4;

    /// The toy model's auxiliary recurrence — a running token count.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(in super::super) struct ToyIndex {
        pub(in super::super) tokens: u32,
    }

    impl ToyIndex {
        /// Magic + version, so a blob from another model is refused rather than
        /// read as a plausible count. Mirrors the real aux container.
        pub(in super::super) const MAGIC: &'static [u8; 4] = b"TIDX";

        pub(in super::super) fn rows(&self) -> u32 {
            self.tokens / TOY_INDEX_RATIO
        }

        /// Tokens in the trailing partial row — 0 when the count lands on a
        /// boundary.
        pub(in super::super) fn open(&self) -> u32 {
            self.tokens % TOY_INDEX_RATIO
        }

        pub(in super::super) fn encode(&self) -> Vec<u8> {
            let mut out = Self::MAGIC.to_vec();
            out.extend_from_slice(&self.tokens.to_le_bytes());
            out
        }

        pub(in super::super) fn decode(blob: &[u8]) -> candle::Result<Self> {
            if blob.len() != 8 || &blob[..4] != Self::MAGIC {
                candle::bail!("not a toy index blob ({} bytes)", blob.len());
            }
            let mut w = [0u8; 4];
            w.copy_from_slice(&blob[4..8]);
            Ok(Self {
                tokens: u32::from_le_bytes(w),
            })
        }
    }

    /// The toy model's **per-position** state, standing in for the QSA index's
    /// page list — a third thing again, and modelled separately for the reason
    /// it exists separately in the real stack.
    ///
    /// `ToyState` is a recurrence (rebuildable by replaying tokens in order) and
    /// `ToyIndex` is an aux blob that travels whole. This is neither: it is a
    /// *list* whose boundaries are decided by the scheduler, at the instant a
    /// unit's K/V anchor is taken. Everything a test can get wrong about that
    /// timing is visible here as a wrong page width.
    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    pub(in super::super) struct ToyPages {
        /// Widths of the pages closed so far, oldest first.
        pub(in super::super) closed: Vec<u32>,
        /// Tokens forwarded since the last close — the live tail, which a unit
        /// beginning on this slot must not be allowed to extend.
        pub(in super::super) tail: u32,
    }

    impl ToyPages {
        pub(in super::super) fn indexed(&self) -> u32 {
            self.closed.iter().sum::<u32>() + self.tail
        }
    }

    pub(in super::super) const ZERO_STATE: ToyState = [[0.0; 4]; 2];

    /// The toy recurrence: `s ← 2s + t·k`, per element, with `k` the element's
    /// 1-based index.
    ///
    /// Three properties earn their place. It is **order-dependent** (advancing
    /// by 3 then 7 lands somewhere else than 7 then 3), so a test can tell a
    /// lost turn from a duplicated one. It is **exact in f32** for the token
    /// counts these tests use, so assertions need no epsilon. And it is
    /// **non-composable** the way the real delta rule is — you cannot recover
    /// it from a per-token record, which is the whole reason the state has to
    /// be carried rather than replayed.
    pub(in super::super) fn toy_advance(state: &mut ToyState, tokens: usize) {
        for (l, layer) in state.iter_mut().enumerate() {
            for (j, cell) in layer.iter_mut().enumerate() {
                *cell = *cell * 2.0 + tokens as f32 * (l * 4 + j + 1) as f32;
            }
        }
    }

    /// A handle on the recurrent map, cloned out before the model is boxed into
    /// the scheduler.
    ///
    /// The scheduler owns its model as `Box<dyn ManagedBatchedModel>`, which a
    /// test cannot downcast back to ask what the state is. Sharing the map
    /// itself is what makes the state observable, and it mirrors the real
    /// arrangement: `HybridBatched` also keeps this map behind a lock because
    /// the scheduler holds the model by shared reference while a wave mutates
    /// the sequences in it.
    #[derive(Clone, Default)]
    pub(in super::super) struct RecurrentProbe {
        pub(in super::super) states: Arc<Mutex<HashMap<usize, ToyState>>>,
        /// The model's *other* recurrence, standing in for Flash-Next's QSA
        /// index: a token count, from which whole rows and a ragged open block
        /// are derived. It travels as opaque bytes and is shaped like nothing in
        /// `ToyState`, which is the entire reason the aux blob exists.
        pub(in super::super) index: Arc<Mutex<HashMap<usize, ToyIndex>>>,
        /// The per-position pages, advanced by the same forward as the other
        /// two and cut by [`Scheduler::begin_unit`].
        pub(in super::super) pages: Arc<Mutex<HashMap<usize, ToyPages>>>,
        /// Sequences whose state arrived by fork or restore, and must therefore
        /// survive exactly one `offset == 0` reset. The store-level `seeded`
        /// flag of §10 decision 2, modelled per sequence.
        pub(in super::super) seeded: Arc<Mutex<HashSet<usize>>>,
        /// Sequences whose state was parked in its record and not yet restored.
        pub(in super::super) parked: Arc<Mutex<HashSet<usize>>>,
        /// What the scheduler installed ahead of each sequence's borrowed K/V,
        /// in order.
        pub(in super::super) pushed: Arc<Mutex<HashMap<usize, Vec<Pushed>>>>,
        /// Each sequence's per-position state as it now stands — pushed, cut,
        /// reset and spliced — for comparing what two slots built.
        pub(in super::super) ledger: Arc<Mutex<HashMap<usize, PositionalLedger>>>,
    }

    /// One piece installed ahead of borrowed K/V.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(in super::super) enum Pushed {
        /// A page the model took.
        Page(Vec<u8>),
        /// Tokens advanced over unindexed.
        Gap(usize),
    }

    /// The prefix of a page in the double's own format — the stand-in for a
    /// container version the build reads.
    pub(in super::super) const TOY_PAGE: &[u8] = b"TOYPAGE";

    impl RecurrentProbe {
        pub(in super::super) fn get(&self, seq: usize) -> Option<ToyState> {
            self.states.lock().unwrap().get(&seq).copied()
        }

        pub(in super::super) fn get_index(&self, seq: usize) -> Option<ToyIndex> {
            self.index.lock().unwrap().get(&seq).copied()
        }

        pub(in super::super) fn advance_index(&self, seq: usize, tokens: usize) {
            let mut idx = self.index.lock().unwrap();
            idx.entry(seq).or_default().tokens += tokens as u32;
            self.pages.lock().unwrap().entry(seq).or_default().tail += tokens as u32;
        }

        pub(in super::super) fn get_pages(&self, seq: usize) -> ToyPages {
            self.pages
                .lock()
                .unwrap()
                .get(&seq)
                .cloned()
                .unwrap_or_default()
        }

        pub(in super::super) fn len(&self) -> usize {
            self.states.lock().unwrap().len()
        }

        pub(in super::super) fn set(&self, seq: usize, state: ToyState) {
            self.states.lock().unwrap().insert(seq, state);
        }

        pub(in super::super) fn is_seeded(&self, seq: usize) -> bool {
            self.seeded.lock().unwrap().contains(&seq)
        }

        /// `ensure_recurrent`'s contract, modelled exactly: a vacant slot gets
        /// zeros; an occupied slot standing at offset 0 is **reset**, because a
        /// sequence with no history must hold the sequence-start value — unless
        /// it was seeded, which suppresses the reset once and once only.
        pub(in super::super) fn ensure(&self, seq: usize, offset: usize) {
            let mut states = self.states.lock().unwrap();
            match states.entry(seq) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(ZERO_STATE);
                }
                std::collections::hash_map::Entry::Occupied(mut slot) => {
                    // Consumed on the FIRST wave either way — the same rule as
                    // the real `ensure_recurrent`. The flag protects the one
                    // wave that follows a fork or a restore; one that outlived
                    // it would suppress a later, genuine reset.
                    let seeded = self.seeded.lock().unwrap().remove(&seq);
                    if offset == 0 && !seeded {
                        slot.insert(ZERO_STATE);
                    }
                }
            }
        }
    }

    /// `DummyModel` plus the per-sequence recurrent state the hybrid lineage
    /// carries — the tier-1 stand-in for `HybridBatched`.
    #[derive(Clone)]
    pub(in super::super) struct DummyRecurrentModel {
        pub(in super::super) inner: DummyModel,
        pub(in super::super) probe: RecurrentProbe,
    }

    impl DummyRecurrentModel {
        /// This double's layer-schedule fingerprint. Arbitrary, but fixed: what
        /// matters is that a restore under a *different* one is refused.
        pub(in super::super) const SCHEDULE_HASH: u64 = 0xD0D0_1234_5678_9ABC;

        /// One sequence's recurrent store, the size a hybrid stack really runs.
        pub(in super::super) const STORE_BYTES: usize = 126 * 1024 * 1024;

        /// What the process holds with ten conversations' stores standing —
        /// the figure admission must **not** reach for, whole or divided.
        pub(in super::super) const RESERVED_TOTAL: usize = 10 * Self::STORE_BYTES;

        pub(in super::super) fn new() -> Self {
            Self {
                inner: DummyModel::new(),
                probe: RecurrentProbe::default(),
            }
        }
    }

    impl ManagedBatchedModel for DummyRecurrentModel {
        fn maybe_change_dtype(&self, dtype: DType) -> candle::Result<()> {
            self.inner.maybe_change_dtype(dtype)
        }
        fn num_layers(&self) -> usize {
            self.inner.num_layers()
        }
        fn n_kv_head(&self) -> usize {
            self.inner.n_kv_head()
        }
        fn head_dim(&self) -> usize {
            self.inner.head_dim()
        }
        fn wave_geometry(&self, act_dtype: DType) -> candle_nn::kv_cache::ModelGeometry {
            self.inner.wave_geometry(act_dtype)
        }
        fn device(&self) -> &candle::Device {
            self.inner.device()
        }

        /// Advances every participating sequence's state by the tokens it
        /// carries, after applying `ensure_recurrent`'s reset rule — the same
        /// order the real wave runs them in, because the reset keying off the
        /// offset is exactly what a restore has to survive.
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
        ) -> candle::Result<candle_transformers::models::batched_inference::WaveResult> {
            for (seqs, inputs) in [(decode_seqs, decode_inputs), (prefill_seqs, prefill_inputs)] {
                for (i, &seq) in seqs.iter().enumerate() {
                    let offset = session.sequence_offset(seq).unwrap_or(0);
                    self.probe.ensure(seq, offset);
                    let tokens = inputs.get(i).map(|t| t.elem_count()).unwrap_or(0);
                    if let Some(state) = self.probe.states.lock().unwrap().get_mut(&seq) {
                        toy_advance(state, tokens);
                    }
                    // The second recurrence advances on the same forward as the
                    // first. That is not decoration: it is what makes a seal
                    // that exports one and not the other observable, because
                    // the two then describe different token counts.
                    self.probe.advance_index(seq, tokens);
                }
            }
            self.inner.forward_wave(
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
        ) -> candle::Result<StepOutcome> {
            stub_step(self, session, spec)
        }

        fn claim_residual(
            &self,
            _session: &BatchedInferenceSession,
            rows: usize,
        ) -> candle::Result<CreepResidual> {
            stub_residual(rows)
        }

        fn prune(&self) -> candle::Result<()> {
            self.inner.prune()
        }

        fn release_sequence(&self, seq: usize) -> candle::Result<()> {
            self.probe.states.lock().unwrap().remove(&seq);
            self.probe.parked.lock().unwrap().remove(&seq);
            self.probe.index.lock().unwrap().remove(&seq);
            self.probe.pages.lock().unwrap().remove(&seq);
            self.probe.seeded.lock().unwrap().remove(&seq);
            self.probe.ledger.lock().unwrap().remove(&seq);
            Ok(())
        }

        /// Copy, and seed the copy. Tolerant of a parent with no state — a view
        /// can be carved before the parent has ever run a wave, and there the
        /// child correctly starts from the sequence-start value.
        /// **The property the double used to get wrong.** It forks and moves
        /// recurrent state, so reporting `false` here — the trait default — made
        /// it a model that demonstrably carries state while claiming it carries
        /// none, which also made `can_gap_fill` true for it. Anything asserting
        /// on either property passed for the wrong reason, and the branch
        /// checkpoint path (gated on exactly this) was never entered by a CPU
        /// test at all.
        fn carries_recurrent_state(&self) -> bool {
            true
        }

        fn fork_recurrent(&self, parent: usize, child: usize) -> candle::Result<()> {
            let parent_state = self.probe.get(parent);
            if let Some(state) = parent_state {
                self.probe.set(child, state);
                self.probe.seeded.lock().unwrap().insert(child);
            }
            // Both classes fork, or the child decodes with one half of its
            // memory at the parent's depth and the other at zero.
            if let Some(idx) = self.probe.get_index(parent) {
                self.probe.index.lock().unwrap().insert(child, idx);
            }
            // **Wholesale, live tail included** — exactly what the real
            // `fork_recurrent` does with the QSA index cache, and the reason the
            // parent's boundary has to be closed BEFORE the carve rather than on
            // the view afterwards.
            let mut pages = self.probe.pages.lock().unwrap();
            if let Some(p) = pages.get(&parent).cloned() {
                pages.insert(child, p);
            }
            Ok(())
        }

        /// The double keeps per-position state, so the paths guarded on this
        /// answer are actually entered by a CPU test.
        fn carries_positional_state(&self) -> bool {
            true
        }

        /// Takes a page only in its own format, and refuses anything else the
        /// way the real model refuses a malformed page.
        fn push_positional_state(
            &self,
            seq: usize,
            blob: &[u8],
            key: &PieceKey,
        ) -> candle::Result<bool> {
            if !blob.starts_with(TOY_PAGE) {
                candle::bail!("toy page: a format this double does not read");
            }
            // The key must be the page's own — the caller takes it once and
            // hands it over with every push.
            assert_eq!(
                *key,
                PieceKey::of(blob),
                "a page pushed under another page's key"
            );
            let mut pushed = self.probe.pushed.lock().unwrap();
            pushed
                .entry(seq)
                .or_default()
                .push(Pushed::Page(blob.to_vec()));
            self.probe
                .ledger
                .lock()
                .unwrap()
                .entry(seq)
                .or_default()
                .push_page(blob);
            Ok(true)
        }

        fn push_positional_gap(&self, seq: usize, tokens: usize) -> candle::Result<()> {
            let mut pushed = self.probe.pushed.lock().unwrap();
            pushed.entry(seq).or_default().push(Pushed::Gap(tokens));
            self.probe
                .ledger
                .lock()
                .unwrap()
                .entry(seq)
                .or_default()
                .push_gap(tokens);
            Ok(())
        }

        /// Reads only its own format, as a push does.
        fn positional_lift_bytes(&self, page: &[u8], _key: &PieceKey) -> candle::Result<usize> {
            if !page.starts_with(TOY_PAGE) {
                candle::bail!("toy page: a format this double does not read");
            }
            Ok(0)
        }

        fn splice_positional_state(
            &self,
            seq: usize,
            from: usize,
            to: usize,
            pieces: &[PositionalPiece],
        ) -> candle::Result<()> {
            self.probe
                .ledger
                .lock()
                .unwrap()
                .entry(seq)
                .or_default()
                .splice(from, to, pieces)
        }

        /// Close the live tail into a page, returning the width it covered.
        ///
        /// `0` for an already-empty tail, which is the legitimate no-op every
        /// caller relies on — and is indistinguishable from a cut that landed
        /// after the tokens it was meant to separate.
        fn close_positional_page(&self, seq: usize) -> candle::Result<usize> {
            let mut pages = self.probe.pages.lock().unwrap();
            let entry = pages.entry(seq).or_default();
            let width = take(&mut entry.tail);
            if width > 0 {
                entry.closed.push(width);
            }
            Ok(width as usize)
        }

        /// What one store costs, and what the process is holding — deliberately
        /// far apart, because the difference is what admission used to divide
        /// by an unrelated count.
        fn recurrent_store_bytes(&self) -> usize {
            DummyRecurrentModel::STORE_BYTES
        }

        fn recurrent_reserved_bytes(&self) -> usize {
            DummyRecurrentModel::RESERVED_TOTAL
        }

        fn positional_coverage(&self, seq: usize) -> Option<usize> {
            Some(self.probe.get_pages(seq).indexed() as usize)
        }

        fn reset_positional_state(&self, seq: usize) -> candle::Result<()> {
            self.probe.pages.lock().unwrap().remove(&seq);
            self.probe.ledger.lock().unwrap().remove(&seq);
            Ok(())
        }

        /// Keep the closed pages that fit within `tokens` and drop the live
        /// tail — the shape the real cut has.
        fn truncate_positional_state(&self, seq: usize, tokens: usize) -> candle::Result<()> {
            self.probe
                .ledger
                .lock()
                .unwrap()
                .entry(seq)
                .or_default()
                .truncate(tokens)?;
            let mut pages = self.probe.pages.lock().unwrap();
            let entry = pages.entry(seq).or_default();
            let mut covered = 0usize;
            let keep = entry
                .closed
                .iter()
                .take_while(|&&w| {
                    covered += w as usize;
                    covered <= tokens
                })
                .count();
            entry.closed.truncate(keep);
            entry.tail = 0;
            Ok(())
        }

        /// The toy state as snapshot rows. One "layer" per `ToyState` row, with
        /// the row's four floats standing in for the delta-rule matrix and a
        /// zero-length conv tail — enough shape for the record round-trip and
        /// the geometry validation `restore_recurrent` performs.
        fn export_recurrent(
            &self,
            seq: usize,
        ) -> candle::Result<Option<(u64, Vec<ExportedLayerState>)>> {
            let Some(state) = self.probe.get(seq) else {
                return Ok(None);
            };
            let layers = state
                .iter()
                .enumerate()
                .map(|(i, row)| ExportedLayerState {
                    layer_index: i as u32,
                    n_v_heads: 1,
                    d_v: 1,
                    d_k: 4,
                    state: row.iter().flat_map(|f| f.to_le_bytes()).collect(),
                    conv_channels: 0,
                    conv_tail_cols: 0,
                    conv_tail: Vec::new(),
                })
                .collect();
            Ok(Some((Self::SCHEDULE_HASH, layers)))
        }

        /// Refuses a foreign schedule hash rather than scattering it, exactly as
        /// `RecurrentStateStore::import` does — a restore that silently accepted
        /// another model's layout is the fluent-amnesia failure this path exists
        /// to remove.
        fn restore_recurrent(
            &self,
            seq: usize,
            schedule_hash: u64,
            layers: &[ExportedLayerState],
        ) -> candle::Result<bool> {
            self.probe.parked.lock().unwrap().remove(&seq);
            if schedule_hash != Self::SCHEDULE_HASH {
                candle::bail!(
                    "schedule hash mismatch: {schedule_hash:#x} is not this model's \
                     {:#x}",
                    Self::SCHEDULE_HASH
                );
            }
            if layers.len() != 2 {
                candle::bail!("expected 2 layers, got {}", layers.len());
            }
            let mut state = ZERO_STATE;
            for (row, l) in state.iter_mut().zip(layers) {
                if l.state.len() != 16 {
                    candle::bail!(
                        "layer {}: state is {} bytes, want 16",
                        l.layer_index,
                        l.state.len()
                    );
                }
                for (slot, chunk) in row.iter_mut().zip(l.state.as_chunks::<4>().0) {
                    *slot = f32::from_le_bytes(*chunk);
                }
            }
            self.probe.set(seq, state);
            self.probe.seeded.lock().unwrap().insert(seq);
            Ok(true)
        }

        /// Move: the child's entry is gone afterwards and the parent's old
        /// state is replaced. Tolerant of a child that never ran a wave.
        fn park_recurrent(&self, seq: usize) -> candle::Result<bool> {
            let dropped = self.probe.states.lock().unwrap().remove(&seq).is_some();
            if dropped {
                self.probe.parked.lock().unwrap().insert(seq);
            }
            Ok(dropped)
        }

        fn recurrent_parked(&self, seq: usize) -> bool {
            self.probe.parked.lock().unwrap().contains(&seq)
        }

        fn move_recurrent(&self, child: usize, parent: usize) -> candle::Result<()> {
            let taken = self.probe.states.lock().unwrap().remove(&child);
            self.probe.seeded.lock().unwrap().remove(&child);
            if let Some(state) = taken {
                self.probe.set(parent, state);
            }
            let taken_idx = self.probe.index.lock().unwrap().remove(&child);
            if let Some(idx) = taken_idx {
                self.probe.index.lock().unwrap().insert(parent, idx);
            }
            let taken_pages = self.probe.pages.lock().unwrap().remove(&child);
            if let Some(p) = taken_pages {
                self.probe.pages.lock().unwrap().insert(parent, p);
            }
            Ok(())
        }

        /// The toy index in its own encoding — bytes only this model reads.
        fn export_aux_state(&self, seq: usize) -> candle::Result<Option<Vec<u8>>> {
            Ok(self.probe.get_index(seq).map(|i| i.encode()))
        }

        /// Installs the index and seeds the slot for the same reason
        /// `restore_recurrent` does: a restored sequence's first wave stands at
        /// offset 0, and an unseeded slot would have the state it was just given
        /// reset out from under it.
        fn restore_aux_state(&self, seq: usize, blob: &[u8]) -> candle::Result<bool> {
            let idx = ToyIndex::decode(blob)?;
            self.probe.index.lock().unwrap().insert(seq, idx);
            self.probe.seeded.lock().unwrap().insert(seq);
            Ok(true)
        }
    }
}
