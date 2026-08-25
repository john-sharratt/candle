//! The recurrent state store: one sequence's DeltaNet memory across all
//! recurrent layers, with wave-atomic advance/rollback and the export/import
//! bridge the turn-seal snapshot record is built from.
//!
//! # Wave atomicity
//!
//! The engine's relief design fails waves on purpose, and a failed wave must
//! leave no trace (`rollback_wave_kv` truncates KV; this store is the
//! recurrent analogue). The contract:
//!
//! ```text
//!   begin_wave()      nothing on the device — mark the slots un-advanced
//!   … the wave READS each layer's live `s` and WRITES the other buffer …
//!   commit_wave()     swap the two buffers of every layer that advanced
//!   rollback_wave()   nothing — the entering state was never written
//! ```
//!
//! A second `begin_wave` without a commit/rollback is refused — an overlapping
//! wave on one session is the bug wave atomicity exists to catch.
//!
//! # Why there is no snapshot
//!
//! The KV side gets its rollback free by being append-only: the pre-wave bytes
//! are still there, below the offset, so undoing a wave is `truncate_to_offset`
//! and costs nothing. The recurrent state has no such structure — `s` is a
//! fixed-size accumulator every token rewrites — so the first implementation
//! took the instruction "the same rollback discipline as KV" to mean copying the
//! entering state aside: ~2 MB and two `slice_set` launches per layer per wave,
//! on every wave, to insure against a rollback that almost never fires.
//!
//! Copying is not what makes KV's rollback free, though; *not destroying the old
//! value* is. So each slot holds two `s` buffers and the wave writes the one it
//! is not reading — the ping-pong `TableRing` and the expert staging ring
//! already use in this tree. Commit is a host `mem::swap`, rollback is nothing
//! at all, and a wave that fails at layer 7 leaves layers 0–6 correct because
//! their entering buffers were never written.
//!
//! Two consequences worth stating:
//!
//! - **`advanced` is per slot**, not per store. A sweep may cover part of the
//!   stack, and swapping a layer the wave never ran would install whatever its
//!   write buffer held two waves ago.
//! - **Both buffers ping-pong.** `s` and the conv tail are one state and swap
//!   together, because the conv kernels take the entering and advanced tails as
//!   two pointers: the decode kernel shifts one into the other and the prefill
//!   kernel writes the advance where the copy-back used to land. That copy-back
//!   was the last `slice_set` on this path — one launch per prefill span per
//!   layer, and the largest single source of `copy2d_f32` in the engine.
//!
//! A slot's buffers are still allocated once for its whole life; what a commit
//! changes is which of the two is live, so a device address resolved from the
//! store is good for the wave that resolved it. That is already how the engine
//! works — `build_wave_table` resolves the pointer table once per forward.
//!
//! # Export / import
//!
//! [`RecurrentStateStore::export`] reads every layer back as LE F32 bytes in
//! [`ExportedLayerState`] rows — field-for-field what the persistence layer's
//! `SnapshotLayer` carries (candle-conversation depends on this crate, not
//! the reverse, so the byte-layout contract lives here and the record
//! assembly there). [`RecurrentStateStore::import`] is the resume path and
//! validates dims + [`schedule_hash`] before touching any tensor.

use candle::{Device, Result, Tensor};

use super::mix::{DeltaNetOut, DeltaNetState};
use super::types::{DeltaNetDims, LayerKind};

/// One recurrent layer's state, exported as LE F32 bytes. Field-for-field the
/// persistence `SnapshotLayer` payload row.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportedLayerState {
    pub layer_index: u32,
    pub n_v_heads: u32,
    pub d_v: u32,
    pub d_k: u32,
    pub state: Vec<u8>,
    pub conv_channels: u32,
    pub conv_tail_cols: u32,
    pub conv_tail: Vec<u8>,
}

/// Fingerprint of a model's recurrent layout: the layer schedule plus the
/// DeltaNet dims. A snapshot taken under one hash must never be scattered
/// into a store built under another — resume recomputes instead.
pub fn schedule_hash(layer_kinds: &[LayerKind], dims: &DeltaNetDims) -> u64 {
    // FNV-1a: stable, dependency-free, and this is an identity check, not
    // crypto.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |b: u64| {
        for byte in b.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    mix(layer_kinds.len() as u64);
    for (i, k) in layer_kinds.iter().enumerate() {
        mix(i as u64);
        mix(match k {
            LayerKind::DeltaNet => 1,
            LayerKind::Attention => 2,
        });
    }
    mix(dims.head_dim as u64);
    mix(dims.n_k_heads as u64);
    mix(dims.n_v_heads as u64);
    mix(dims.conv_kernel as u64);
    h
}

/// Per-layer slot: the two halves of the state's ping-pong.
struct LayerSlot {
    /// Trunk layer index (recurrent layers only — attention layers have no
    /// slot here).
    layer_index: usize,
    /// The state as it stands. A wave READS this and never writes it.
    live: DeltaNetState,
    /// Where a wave WRITES the advanced state. Fully overwritten by the
    /// kernels, so it carries nothing forward from whatever it last held.
    backup: DeltaNetState,
    /// Whether this wave handed the layer its write buffer, i.e. whether
    /// `backup` holds an advanced state that commit should install.
    ///
    /// Per slot, not per store, because a sweep may cover only part of the
    /// stack: swapping a layer the wave never ran would install whatever its
    /// write buffer held two waves ago.
    advanced: bool,
}

/// One sequence's recurrent memory across every DeltaNet layer.
pub struct RecurrentStateStore {
    dims: DeltaNetDims,
    hash: u64,
    slots: Vec<LayerSlot>,
    /// Whether a wave is open, i.e. whether the backups hold an entry copy.
    ///
    /// One flag for the store rather than one per slot: the three wave
    /// operations act on every slot together, so a per-slot answer could only
    /// ever disagree with its neighbours by being wrong.
    open: bool,
    device: Device,
}

impl RecurrentStateStore {
    /// Fresh zeros for every recurrent layer in `layer_kinds`.
    pub fn new(layer_kinds: &[LayerKind], dims: &DeltaNetDims, device: &Device) -> Result<Self> {
        let mut slots = Vec::new();
        for (i, k) in layer_kinds.iter().enumerate() {
            if *k == LayerKind::DeltaNet {
                slots.push(LayerSlot {
                    layer_index: i,
                    live: DeltaNetState::zeros(dims, device)?,
                    backup: DeltaNetState::zeros(dims, device)?,
                    advanced: false,
                });
            }
        }
        Ok(Self {
            dims: *dims,
            hash: schedule_hash(layer_kinds, dims),
            slots,
            open: false,
            device: device.clone(),
        })
    }

    pub fn schedule_hash(&self) -> u64 {
        self.hash
    }

    /// Trunk layer indices of the recurrent layers, in slot order — what the
    /// decode pointer table iterates to collect every layer's state address.
    pub fn recurrent_layer_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots.iter().map(|s| s.layer_index)
    }

    pub fn n_recurrent_layers(&self) -> usize {
        self.slots.len()
    }

    /// The live state of trunk layer `layer_index`, for the layer forward /
    /// decode kernel. Errors on an attention layer's index.
    pub fn layer_state(&self, layer_index: usize) -> Result<&DeltaNetState> {
        self.slots
            .iter()
            .find(|s| s.layer_index == layer_index)
            .map(|s| &s.live)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "recurrent store: layer {layer_index} holds no recurrent state"
                ))
            })
    }

    /// Trunk layer `layer_index`'s live state, to be **written into** —
    /// **outside a wave only**.
    ///
    /// This is the in-place form, and a wave must not use it: a wave advances a
    /// layer by writing the buffer it is *not* reading
    /// ([`Self::layer_state_pair_mut`]), and writing `live` instead destroys the
    /// entering state that a rollback returns to, while `commit_wave` then swaps
    /// the untouched other buffer in and discards the work. Both failures are
    /// silent. What legitimately uses this is code holding a store no wave is
    /// open on — the verification path builds a fresh single-sequence store per
    /// block and advances it directly.
    ///
    /// There is deliberately no setter: a store that could be handed a
    /// *different* tensor is one where prefill and decode end up advancing the
    /// state two different ways.
    pub fn layer_state_mut(&mut self, layer_index: usize) -> Result<&mut DeltaNetState> {
        self.slots
            .iter_mut()
            .find(|s| s.layer_index == layer_index)
            .map(|s| &mut s.live)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "recurrent store: layer {layer_index} holds no recurrent state"
                ))
            })
    }

    /// The layer's `(entering, advanced)` buffers **without** recording that it
    /// advanced.
    ///
    /// For resolving addresses ahead of the work: the decode pointer table is
    /// built once per forward over every recurrent layer, including ones a
    /// partial sweep will never reach, so building it must not be what decides
    /// a layer gets swapped at commit. The layer records itself when it runs,
    /// through [`Self::layer_state_pair_mut`].
    pub fn layer_state_pair(&self, layer_index: usize) -> Result<(&DeltaNetState, DeltaNetOut)> {
        let slot = self
            .slots
            .iter()
            .find(|s| s.layer_index == layer_index)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "recurrent store: layer {layer_index} holds no recurrent state"
                ))
            })?;
        Ok((&slot.live, slot.backup.write_half()))
    }

    /// The layer's `(entering, advanced)` buffers — what a wave reads and what
    /// it writes — and the record that this layer advanced.
    ///
    /// Taking this pair is what marks the slot for the swap at
    /// [`Self::commit_wave`], so a caller asks for it exactly when it is about
    /// to run the layer, never to peek.
    pub fn layer_state_pair_mut(
        &mut self,
        layer_index: usize,
    ) -> Result<(&mut DeltaNetState, DeltaNetOut)> {
        let slot = self
            .slots
            .iter_mut()
            .find(|s| s.layer_index == layer_index)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "recurrent store: layer {layer_index} holds no recurrent state"
                ))
            })?;
        slot.advanced = true;
        let out = slot.backup.write_half();
        Ok((&mut slot.live, out))
    }

    /// The layer's halves **the other way round**: the state the last committed
    /// wave *entered* with, and the live buffer to write a corrected advance
    /// into.
    ///
    /// This is the rewind primitive. `commit_wave` exchanges a slot's two
    /// buffers, so immediately afterwards the half that is no longer live still
    /// holds the pre-wave state — untouched, because a wave writes only the
    /// buffer it is not reading. Re-running a *prefix* of the wave's tokens
    /// from there lands the correct shorter advance in the live buffer, which
    /// is how a speculative block keeps the accepted tokens and drops the rest;
    /// `S` is a running sum with no suffix to subtract, so replaying forward is
    /// the only exact answer.
    ///
    /// **Valid only between the commit and the next `begin_wave`.** After
    /// another wave has run, the non-live half holds *that* wave's entry state
    /// and this returns a rewind to the wrong point. Refused while a wave is
    /// open, which is the half of that the store can see.
    pub fn layer_state_rewind(
        &mut self,
        layer_index: usize,
    ) -> Result<(&mut DeltaNetState, DeltaNetOut)> {
        if self.open {
            candle::bail!(
                "recurrent store: layer_state_rewind mid-wave — the entering state \
                 to rewind to is the buffer the open wave is writing"
            );
        }
        let slot = self
            .slots
            .iter_mut()
            .find(|s| s.layer_index == layer_index)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "recurrent store: layer {layer_index} holds no recurrent state"
                ))
            })?;
        let out = slot.live.write_half();
        Ok((&mut slot.backup, out))
    }

    /// Open a wave. Refuses while one is already open.
    ///
    /// **Costs nothing on the device.** The entering state is preserved by not
    /// being written: a wave reads `live` and writes `backup`, so opening a wave
    /// is bookkeeping and rolling one back is doing nothing at all. This is the
    /// same trick the KV side gets for free by being append-only — its rollback
    /// is `truncate_to_offset`, because the pre-wave bytes were never touched.
    ///
    /// It replaces a copy of every layer's state into its backup: ~2 MB per
    /// layer per wave, two `slice_set` launches each, paid on every wave to
    /// insure against a rollback that almost never happens.
    pub fn begin_wave(&mut self) -> Result<()> {
        if self.open {
            candle::bail!(
                "recurrent store: begin_wave with a wave already open — overlapping \
                 waves on one session are exactly what atomicity forbids"
            );
        }
        for slot in &mut self.slots {
            slot.advanced = false;
        }
        self.open = true;
        Ok(())
    }

    /// The wave's writes stand: every layer the wave advanced exchanges its two
    /// buffers, so what the wave wrote becomes the state and what the state was
    /// becomes the next wave's write buffer.
    ///
    /// A host pointer swap per advanced layer, and no device work at all. Layers
    /// the sweep did not reach keep their buffers as they are — their write
    /// buffer holds an older wave's output, which is exactly why the flag is per
    /// slot.
    pub fn commit_wave(&mut self) {
        for slot in &mut self.slots {
            if slot.advanced {
                // The whole state: `s` and the conv tail are both written into
                // the backup half by the wave's kernels — the conv kernels take
                // the entering and advanced tails as two pointers — so they are
                // installed together.
                std::mem::swap(&mut slot.live, &mut slot.backup);
                slot.advanced = false;
            }
        }
        self.open = false;
    }

    /// The wave never happened.
    ///
    /// Nothing to undo: a wave writes only into the buffers `commit_wave` would
    /// have swapped in, so declining to swap *is* the rollback. Refuses when no
    /// wave is open (a rollback with nothing to roll back to is a sequencing
    /// bug, not a no-op).
    pub fn rollback_wave(&mut self) -> Result<()> {
        if !self.open {
            candle::bail!("recurrent store: rollback_wave with no wave open");
        }
        for slot in &mut self.slots {
            slot.advanced = false;
        }
        self.open = false;
        Ok(())
    }

    /// Read every layer back as LE F32 bytes — the turn-seal snapshot body.
    /// Refused mid-wave: a snapshot must capture a sealed boundary, never a
    /// wave in flight.
    pub fn export(&self) -> Result<Vec<ExportedLayerState>> {
        if self.open {
            candle::bail!("recurrent store: export mid-wave — seal, then snapshot");
        }
        let d = &self.dims;
        let mut out = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            let state_v: Vec<f32> = slot.live.s.flatten_all()?.to_vec1()?;
            let tail_v: Vec<f32> = slot.live.conv_tail.flatten_all()?.to_vec1()?;
            out.push(ExportedLayerState {
                layer_index: slot.layer_index as u32,
                n_v_heads: d.n_v_heads as u32,
                d_v: d.head_dim as u32,
                d_k: d.head_dim as u32,
                state: state_v.iter().flat_map(|f| f.to_le_bytes()).collect(),
                conv_channels: d.conv_dim() as u32,
                conv_tail_cols: (d.conv_kernel - 1) as u32,
                conv_tail: tail_v.iter().flat_map(|f| f.to_le_bytes()).collect(),
            });
        }
        Ok(out)
    }

    /// Scatter a snapshot back into the store — the resume path. Validates
    /// the schedule hash and every layer's dims before touching any tensor;
    /// on any mismatch the store is left untouched and the caller recomputes.
    pub fn import(&mut self, snapshot_hash: u64, layers: &[ExportedLayerState]) -> Result<()> {
        if snapshot_hash != self.hash {
            candle::bail!(
                "recurrent store: snapshot schedule hash {snapshot_hash:#x} does not match \
                 this model's {:#x} — recompute the state instead of scattering a foreign \
                 layout",
                self.hash
            );
        }
        if self.open {
            candle::bail!("recurrent store: import mid-wave");
        }
        let d = &self.dims;
        if layers.len() != self.slots.len() {
            candle::bail!(
                "recurrent store: snapshot has {} layers, store has {}",
                layers.len(),
                self.slots.len()
            );
        }
        // Validate everything first — import is all-or-nothing.
        for (slot, l) in self.slots.iter().zip(layers) {
            if l.layer_index as usize != slot.layer_index
                || l.n_v_heads as usize != d.n_v_heads
                || l.d_v as usize != d.head_dim
                || l.d_k as usize != d.head_dim
                || l.conv_channels as usize != d.conv_dim()
                || l.conv_tail_cols as usize != d.conv_kernel - 1
                || l.state.len() != d.state_elems() * 4
                || l.conv_tail.len() != d.conv_state_elems() * 4
            {
                candle::bail!(
                    "recurrent store: snapshot layer {} does not match the store's \
                     geometry",
                    l.layer_index
                );
            }
        }
        for (slot, l) in self.slots.iter_mut().zip(layers) {
            let state_f: Vec<f32> = l
                .state
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let tail_f: Vec<f32> = l
                .conv_tail
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            // Written into the slot's buffers rather than replacing them: the
            // slot's tensors keep their identity for the store's whole life, and
            // the fused decode kernels rely on that.
            slot.live.copy_from(&DeltaNetState {
                s: Tensor::from_vec(state_f, (d.n_v_heads, d.head_dim, d.head_dim), &self.device)?,
                conv_tail: Tensor::from_vec(
                    tail_f,
                    (d.conv_dim(), d.conv_kernel - 1),
                    &self.device,
                )?,
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims() -> DeltaNetDims {
        DeltaNetDims {
            head_dim: 4,
            n_k_heads: 2,
            n_v_heads: 4,
            conv_kernel: 3,
        }
    }

    fn kinds() -> Vec<LayerKind> {
        vec![
            LayerKind::DeltaNet,
            LayerKind::DeltaNet,
            LayerKind::Attention,
            LayerKind::DeltaNet,
        ]
    }

    /// One wave's worth of writes into a layer's destination half: `live + 1`
    /// in both buffers, which is what the kernels do to their two pointers.
    fn bump_into(live: &DeltaNetState, out: &DeltaNetOut) {
        let one = |src: &Tensor, dst: &Tensor| {
            let ones = Tensor::ones(src.shape(), src.dtype(), &Device::Cpu).unwrap();
            dst.slice_set(&src.add(&ones).unwrap(), 0, 0).unwrap();
        };
        one(&live.s, &out.s);
        one(&live.conv_tail, &out.conv_tail);
    }

    fn filled_store() -> RecurrentStateStore {
        let dev = Device::Cpu;
        let d = dims();
        let mut store = RecurrentStateStore::new(&kinds(), &d, &dev).unwrap();
        for (i, li) in [0usize, 1, 3].iter().enumerate() {
            let n = d.state_elems();
            let s: Vec<f32> = (0..n).map(|j| (i * 1000 + j) as f32 * 0.01).collect();
            let tn = d.conv_state_elems();
            let t: Vec<f32> = (0..tn).map(|j| (i * 100 + j) as f32 * 0.1).collect();
            let live = store.layer_state_mut(*li).unwrap();
            live.copy_from(&DeltaNetState {
                s: Tensor::from_vec(s, (d.n_v_heads, d.head_dim, d.head_dim), &dev).unwrap(),
                conv_tail: Tensor::from_vec(t, (d.conv_dim(), d.conv_kernel - 1), &dev).unwrap(),
            })
            .unwrap();
        }
        store
    }

    #[test]
    fn export_import_roundtrips_exactly() {
        let store = filled_store();
        let hash = store.schedule_hash();
        let exported = store.export().unwrap();
        assert_eq!(exported.len(), 3);
        assert_eq!(exported[2].layer_index, 3);

        let mut fresh = RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap();
        fresh.import(hash, &exported).unwrap();
        let re = fresh.export().unwrap();
        assert_eq!(exported, re, "export→import→export must be byte-identical");
    }

    #[test]
    fn import_refuses_wrong_hash_and_wrong_geometry() {
        let store = filled_store();
        let exported = store.export().unwrap();

        let mut fresh = RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap();
        let before = fresh.export().unwrap();
        let err = fresh
            .import(store.schedule_hash() ^ 1, &exported)
            .unwrap_err();
        assert!(err.to_string().contains("schedule hash"));
        assert_eq!(
            fresh.export().unwrap(),
            before,
            "refusal must not touch state"
        );

        let mut bad = exported.clone();
        bad[0].d_k = 5;
        let err = fresh.import(store.schedule_hash(), &bad).unwrap_err();
        assert!(err.to_string().contains("geometry"));
        assert_eq!(fresh.export().unwrap(), before);
    }

    #[test]
    fn wave_rollback_restores_entry_state_and_commit_keeps_writes() {
        let mut store = filled_store();
        let entry = store.export().unwrap();

        // A wave writes into the slot's OTHER buffer — the half `commit_wave`
        // swaps in — so the entering state survives by never being written.
        let bump = |store: &mut RecurrentStateStore| {
            let (live, out) = store.layer_state_pair_mut(0).unwrap();
            // Stands in for the kernels' writes into the destination buffers —
            // both of them, because commit installs the whole state.
            bump_into(live, &out);
        };
        store.begin_wave().unwrap();
        bump(&mut store);
        store.rollback_wave().unwrap();
        assert_eq!(
            store.export().unwrap(),
            entry,
            "rollback must restore the wave-entry state exactly"
        );

        // A successful wave: mutate, commit — the write stands, in BOTH
        // buffers. Asserting only on `s` would pass while the conv tail was
        // left behind in the half the swap filed away, which is precisely the
        // failure a partial swap produces: a state one wave ahead of its tail.
        store.begin_wave().unwrap();
        bump(&mut store);
        store.commit_wave();
        let committed = store.export().unwrap();
        assert_ne!(
            committed[0].state, entry[0].state,
            "commit must install `s`"
        );
        assert_ne!(
            committed[0].conv_tail, entry[0].conv_tail,
            "commit must install the advanced conv tail, not just `s`"
        );
        // Layer 1 never ran, so its slot keeps both buffers as they were.
        assert_eq!(committed[1], entry[1], "an unrun layer must not be swapped");
    }

    /// **A wave never writes the buffer it read, so an entering alias is never
    /// disturbed by a wave that fails.**
    ///
    /// This replaces the inverse contract — that rollback must copy the entry
    /// values back into the same allocation, because an alias resolved before
    /// the wave would otherwise still see the failed wave's writes. Under the
    /// ping-pong there are no writes to undo: the wave's output went to the
    /// other buffer, so the alias holds the entry values throughout and a
    /// rollback is doing nothing.
    ///
    /// The price is that `commit_wave` DOES change which tensor is live, so a
    /// resolved address is valid for one wave only. That is what the engine
    /// already does — `build_wave_table` resolves the pointers once per forward
    /// (`qwen35/forward.rs`), inside the wave that uses them.
    #[test]
    fn a_wave_leaves_the_entering_buffer_untouched() {
        let mut store = filled_store();
        // Shares storage with the slot's entering state — the same view the
        // decode kernel's pointer table holds for this wave.
        let alias = store.layer_state(0).unwrap().s.clone();
        let entry: Vec<f32> = alias.flatten_all().unwrap().to_vec1().unwrap();

        store.begin_wave().unwrap();
        {
            let (live, out) = store.layer_state_pair_mut(0).unwrap();
            bump_into(live, &out);
        }
        let during: Vec<f32> = alias.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            during, entry,
            "the wave wrote into the buffer it was reading — the entering state \
             is gone and a rollback has nothing to return to"
        );

        store.rollback_wave().unwrap();
        let after: Vec<f32> = alias.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(after, entry, "rollback must leave the entering state alone");

        // And on the committing path the swap installs the wave's output.
        store.begin_wave().unwrap();
        {
            let (live, out) = store.layer_state_pair_mut(0).unwrap();
            bump_into(live, &out);
        }
        store.commit_wave();
        let committed: Vec<f32> = store
            .layer_state(0)
            .unwrap()
            .s
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_ne!(committed, entry, "commit must install the wave's output");
    }

    #[test]
    fn wave_sequencing_is_enforced() {
        let mut store = filled_store();
        assert!(store.rollback_wave().is_err(), "rollback with no wave open");
        store.begin_wave().unwrap();
        assert!(store.begin_wave().is_err(), "overlapping wave");
        assert!(store.export().is_err(), "export mid-wave");
        store.commit_wave();
        assert!(store.export().is_ok());
    }

    #[test]
    fn schedule_hash_pins_layout() {
        let h = schedule_hash(&kinds(), &dims());
        assert_eq!(h, schedule_hash(&kinds(), &dims()), "deterministic");
        let mut other = kinds();
        other[2] = LayerKind::DeltaNet;
        assert_ne!(h, schedule_hash(&other, &dims()), "schedule change");
        let mut d2 = dims();
        d2.conv_kernel = 4;
        assert_ne!(h, schedule_hash(&kinds(), &d2), "dims change");
    }
}
