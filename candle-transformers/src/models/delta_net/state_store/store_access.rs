//! Creates a sequence's recurrent store, gives access to its layers, and resets
//! it to the sequence-start state.
//!
//! A new store holds zeros in every layer, because a slot is recycled from
//! whatever sequence held it last. A reset zeroes the state where it stands rather
//! than building a second store, and a store left open by a failed wave is rolled
//! back first. The live state is written only outside a wave, since a wave must
//! leave the state it entered untouched.

use super::*;

impl RecurrentStateStore {
    /// Fresh zeros for every recurrent layer in `layer_kinds`.
    pub fn new(layer_kinds: &[LayerKind], dims: &DeltaNetDims, device: &Device) -> Result<Self> {
        let mut slots = Vec::new();
        // On CUDA every layer is a state-arena slot inside the device reservation,
        // its state zeroed — it is genuinely READ at zero, being the
        // sequence-start state, and a slot is recycled from whatever sequence held
        // it last. The cfg gates COMPILATION; the device gates behaviour. A CUDA
        // build still runs on a CPU device — every unit test here does — and there
        // is no reservation there to carve from.
        #[cfg(feature = "cuda")]
        let mut claimed = claim_layer_states(
            dims,
            device,
            layer_kinds
                .iter()
                .filter(|k| **k == LayerKind::DeltaNet)
                .count(),
        )?
        .map(Vec::into_iter);
        for (i, k) in layer_kinds.iter().enumerate() {
            if *k != LayerKind::DeltaNet {
                continue;
            }
            #[cfg(feature = "cuda")]
            let live = match claimed.as_mut() {
                Some(it) => {
                    let slot = it.next().expect("one slot was claimed per recurrent layer");
                    zero_state(dims, device, &slot)?;
                    Half::in_slot(dims, device, slot)?
                }
                None => Half::host(dims, device, true)?,
            };
            #[cfg(not(feature = "cuda"))]
            let live = Half::host(dims, device, true)?;
            slots.push(LayerSlot {
                layer_index: i,
                live,
                scratch: None,
                pending: None,
                advanced: false,
            });
        }
        Ok(Self {
            dims: *dims,
            hash: schedule_hash(layer_kinds, dims),
            slots,
            open: false,
            ran_this_wave: false,
            long_this_wave: false,
            // A fresh store already holds the sequence-start value, so there is
            // nothing for a reset to destroy.
            seeded: false,
            step: None,
            device: device.clone(),
        })
    }

    /// Back to the sequence-start state, on the slots this store already holds.
    ///
    /// Each layer's live state is zeroed where it stands. This is how a sequence
    /// starting over gets its zeros — not by dropping the store and building
    /// another. A store's slots go back to their arena only once every tensor
    /// viewing them has gone, so a rebuild can ask for a second store's worth of
    /// slots while the first is still held; on a span with no room to spare that
    /// claim fails part-way, and the turn that was only restarting fails with it.
    ///
    /// Called before a wave opens, so a store still marked open here is one
    /// whose last wave was never closed — a failure that skipped its rollback.
    /// That wave is rolled back first, every pending timeline is dropped, and the
    /// prefill scratch goes: the restart goes ahead on a store that was otherwise
    /// left unusable.
    pub fn reset(&mut self) -> Result<()> {
        if self.open {
            self.rollback_wave()?;
        }
        self.break_step();
        self.discard_pending();
        self.release_scratch();
        for slot in &mut self.slots {
            slot.advanced = false;
            #[cfg(feature = "cuda")]
            if let Some(held) = slot.live.held.as_ref() {
                zero_state(&self.dims, &self.device, held)?;
                continue;
            }
            slot.live.state = DeltaNetState::zeros(&self.dims, &self.device)?;
        }
        self.seeded = false;
        Ok(())
    }

    pub fn schedule_hash(&self) -> u64 {
        self.hash
    }

    /// Whether a wave has this store open: its state is mid-forward, and
    /// nothing may take it away.
    pub fn wave_open(&self) -> bool {
        self.open
    }

    pub fn dims(&self) -> &DeltaNetDims {
        &self.dims
    }

    /// Trunk layer indices of the recurrent layers, in slot order — what the
    /// decode pointer table iterates to collect every layer's state address.
    pub fn recurrent_layer_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots.iter().map(|s| s.layer_index)
    }

    pub fn n_recurrent_layers(&self) -> usize {
        self.slots.len()
    }

    pub(super) fn slot(&self, layer_index: usize) -> Result<&LayerSlot> {
        self.slots
            .iter()
            .find(|s| s.layer_index == layer_index)
            .ok_or_else(|| no_slot(layer_index))
    }

    pub(super) fn slot_mut(&mut self, layer_index: usize) -> Result<&mut LayerSlot> {
        self.slots
            .iter_mut()
            .find(|s| s.layer_index == layer_index)
            .ok_or_else(|| no_slot(layer_index))
    }

    /// The live state of trunk layer `layer_index`. Errors on an attention
    /// layer's index.
    pub fn layer_state(&self, layer_index: usize) -> Result<&DeltaNetState> {
        Ok(&self.slot(layer_index)?.live.state)
    }

    /// Trunk layer `layer_index`'s live state, to be **written into** —
    /// **outside a wave only**.
    ///
    /// This is the in-place form, and a wave must not use it: a wave advances a
    /// layer through [`Self::layer_span_mut`], which leaves the live state as it
    /// entered until a fold or a commit installs the advance; writing it directly
    /// destroys the entering state a failed wave returns to. What legitimately
    /// uses this is code holding a store no wave is open on — the verification
    /// path builds a fresh single-sequence store per block and advances it
    /// directly.
    ///
    /// There is deliberately no setter: a store that could be handed a
    /// *different* tensor is one where prefill and decode end up advancing the
    /// state two different ways.
    pub fn layer_state_mut(&mut self, layer_index: usize) -> Result<&mut DeltaNetState> {
        self.break_step();
        Ok(&mut self.slot_mut(layer_index)?.live.state)
    }
}
