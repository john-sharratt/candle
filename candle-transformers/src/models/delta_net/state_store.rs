//! The recurrent state store: one sequence's DeltaNet memory across all
//! recurrent layers, with wave-atomic snapshot/rollback and the export/import
//! bridge the turn-seal snapshot record is built from.
//!
//! # Wave atomicity
//!
//! The engine's relief design fails waves on purpose, and a failed wave must
//! leave no trace (`rollback_wave_kv` truncates KV; this store is the
//! recurrent analogue). The contract:
//!
//! ```text
//!   begin_wave()      device copy of every layer's state + conv tail
//!   … decode steps mutate the live tensors in place …
//!   commit_wave()     the wave's writes stand
//!   rollback_wave()   restore the entry state — the wave never happened
//! ```
//!
//! A second `begin_wave` without a commit/rollback is refused — an overlapping
//! wave on one session is the bug wave atomicity exists to catch.
//!
//! # The backup is a buffer, not a value
//!
//! Every slot owns its entry copy for the store's whole life, and `begin_wave`
//! *writes into* it. The obvious implementation — allocate a copy per wave,
//! drop it on commit — costs an allocate/free pair per layer per session per
//! wave, and at decode a wave is one token: it was the single largest device
//! allocator in the loop, 679 MB across one measured window on the 0.8B at
//! batch 4, more than half of everything the decode loop allocated. Holding the
//! buffer costs nothing extra at the peak either, since every session in a wave
//! has a backup live at once regardless.
//!
//! Committing therefore drops nothing, and rollback copies back rather than
//! swapping tensors in. Both follow from the same rule the live state obeys:
//! **a slot's buffers are allocated once and keep their identity**, because the
//! fused decode kernels write them in place.
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

use super::mix::DeltaNetState;
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

/// Per-layer slot: the live state and the buffer its entry copy is written to.
struct LayerSlot {
    /// Trunk layer index (recurrent layers only — attention layers have no
    /// slot here).
    layer_index: usize,
    live: DeltaNetState,
    /// The wave-entry copy. Meaningful only while [`RecurrentStateStore::open`]
    /// is set; allocated with the slot either way, so a wave boundary never
    /// reaches the allocator.
    backup: DeltaNetState,
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

    /// Trunk layer `layer_index`'s live state, to be **written into**.
    ///
    /// The state is a fixed-size buffer owned by the sequence for its whole
    /// life, not a value the layer returns a replacement for — so the mixer
    /// takes this and mutates it. There is deliberately no setter: a store that
    /// could be handed a *different* tensor is one where the buffer identity
    /// can change under the wave snapshot, and where prefill and decode end up
    /// advancing the state two different ways.
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

    /// Copy every layer's entry state aside. Refuses while a wave is open.
    ///
    /// The copies go into buffers the slots already hold, so this is device
    /// copies and nothing else — see the module header.
    pub fn begin_wave(&mut self) -> Result<()> {
        if self.open {
            candle::bail!(
                "recurrent store: begin_wave with a wave already open — overlapping \
                 waves on one session are exactly what atomicity forbids"
            );
        }
        for slot in &mut self.slots {
            let LayerSlot { live, backup, .. } = slot;
            backup.copy_from(live)?;
        }
        self.open = true;
        Ok(())
    }

    /// The wave's writes stand.
    ///
    /// Nothing is freed: the backups are the slots' own buffers, and what makes
    /// them stale is the flag, not their contents.
    pub fn commit_wave(&mut self) {
        self.open = false;
    }

    /// The wave never happened: every layer's state reverts to its entry copy.
    /// Refuses when no wave is open (a rollback with nothing to roll back to
    /// is a sequencing bug, not a no-op).
    pub fn rollback_wave(&mut self) -> Result<()> {
        if !self.open {
            candle::bail!("recurrent store: rollback_wave with no wave open");
        }
        for slot in &mut self.slots {
            let LayerSlot { live, backup, .. } = slot;
            live.copy_from(backup)?;
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

        // A failed wave: write *into* the buffer, then roll back. The write is
        // in place, which is exactly why the snapshot has to exist — nothing
        // else holds the entry value once the mixer has run.
        let bump = |store: &mut RecurrentStateStore| {
            let live = store.layer_state_mut(0).unwrap();
            let ones = Tensor::ones(live.s.shape(), live.s.dtype(), &Device::Cpu).unwrap();
            live.s.add_mut(&ones).unwrap();
        };
        store.begin_wave().unwrap();
        bump(&mut store);
        store.rollback_wave().unwrap();
        assert_eq!(
            store.export().unwrap(),
            entry,
            "rollback must restore the wave-entry state exactly"
        );

        // A successful wave: mutate, commit — the write stands.
        store.begin_wave().unwrap();
        bump(&mut store);
        store.commit_wave();
        assert_ne!(store.export().unwrap(), entry);
    }

    /// **Rollback must restore the buffer, not replace it.**
    ///
    /// The mixer and the fused decode kernels write `s` and the conv tail in
    /// place, so the store's tensors are addresses other code has already
    /// resolved. Rolling back by swapping a *different* tensor into the slot
    /// reads correct through `layer_state`, and leaves everything holding the
    /// old buffer looking at the wave's writes — the failure this pins.
    ///
    /// Checked through a shallow `clone`, which shares storage: after a
    /// rollback it must show the entry values, which is only true if the entry
    /// values were copied back into the same allocation.
    #[test]
    fn rollback_writes_back_into_the_live_buffer() {
        let mut store = filled_store();
        // Shares storage with the slot's live state — the same view the decode
        // kernel holds.
        let alias = store.layer_state(0).unwrap().s.clone();
        let entry: Vec<f32> = alias.flatten_all().unwrap().to_vec1().unwrap();

        store.begin_wave().unwrap();
        {
            let live = store.layer_state_mut(0).unwrap();
            let ones = Tensor::ones(live.s.shape(), live.s.dtype(), &Device::Cpu).unwrap();
            live.s.add_mut(&ones).unwrap();
        }
        let during: Vec<f32> = alias.flatten_all().unwrap().to_vec1().unwrap();
        assert_ne!(
            during, entry,
            "the alias must see the wave's in-place writes"
        );

        store.rollback_wave().unwrap();
        let after: Vec<f32> = alias.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            after, entry,
            "rollback replaced the slot's tensor instead of writing into it — \
             everything already holding the old buffer still sees the failed \
             wave's state"
        );
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
