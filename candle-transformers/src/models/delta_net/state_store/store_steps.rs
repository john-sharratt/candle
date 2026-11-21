//! Opens and closes steps, which are several waves that stand or fall together,
//! and forks a store from another.
//!
//! A decode step is swept in segments that each commit, so a refusal in a later
//! segment must be undoable. Inside a step a commit folds nothing, which makes the
//! timeline the undo: dropping its rows returns each layer to the state the step
//! entered with. A fork copies the parent's live state into the child's own slot.
//! A fork or a relocation refuses a store with rows pending, since the child would
//! otherwise start a span behind its parent.

use super::*;

impl RecurrentStateStore {
    /// Open a step: several waves that stand or fall together.
    ///
    /// **A decode step is swept in segments**, each its own wave, and each
    /// commits as it ends — so a refused second segment would roll back only its
    /// own layers and leave the first segment's advanced. The scheduler retries
    /// the step from layer 0, which would run those layers' tokens into their
    /// state a second time: a recurrent state that has absorbed the same token
    /// twice is wrong in a way nothing downstream can see, until it is far
    /// enough wrong to be NaN.
    ///
    /// **Free: the timeline is the undo.** Inside a step a commit folds nothing
    /// ([`commit_stores`]): every short span's rows stay pending, so each layer's
    /// live state is still the one the step entered with. [`end_steps`] folds
    /// them; [`Self::undo_step`] drops them. The segments of a step cover
    /// disjoint layers, and a pending layer refuses a second span, so no layer
    /// is advanced twice within one. Refused mid-wave.
    pub fn begin_step(&mut self) -> Result<()> {
        if self.open {
            candle::bail!("recurrent store: begin_step mid-wave");
        }
        self.step = Some(StepRecord {
            entered_pending: self.slots.iter().map(|s| s.pending.is_some()).collect(),
            fold_owed: false,
            broken: false,
        });
        Ok(())
    }

    /// Close the step and keep its waves — [`end_steps`] for one store.
    pub fn end_step(&mut self) -> Result<()> {
        end_steps(&mut [self])
    }

    /// The step never happened: every row a wave of the step left pending is
    /// dropped, so each layer stands at the state the step entered with.
    /// Refuses mid-wave, with no step open, and when the record cannot vouch
    /// for the undo — the state moved inside the step other than by a short
    /// span — closing the step either way.
    pub fn undo_step(&mut self) -> Result<()> {
        if self.open {
            candle::bail!("recurrent store: undo_step mid-wave");
        }
        let Some(step) = self.step.take() else {
            candle::bail!("recurrent store: undo_step with no step open");
        };
        if step.broken {
            candle::bail!(
                "recurrent store: the step cannot be undone — its state moved by an exchange, \
                 a fold or a write outside a wave"
            );
        }
        for (slot, entered) in self.slots.iter_mut().zip(step.entered_pending) {
            if !entered {
                slot.pending = None;
            }
        }
        Ok(())
    }

    /// Whatever just happened to the live state, the open step (if any) can no
    /// longer be undone by dropping its rows.
    pub(super) fn break_step(&mut self) {
        if let Some(step) = self.step.as_mut() {
            step.broken = true;
        }
    }

    /// An independent store carrying this one's state — the fork primitive.
    ///
    /// Device-to-device: each layer's live `s` and conv tail are copied into the
    /// child's own slot. The child holds no scratch and nothing pending.
    ///
    /// Refused mid-wave and with a layer pending: either way the state this
    /// store will stand at is not yet the one in its live buffers, and the child
    /// would come up a span behind its parent with every shape correct.
    pub fn fork_from(&self) -> Result<Self> {
        if self.open {
            candle::bail!(
                "recurrent store: fork_from mid-wave — the wave's advance is not yet in \
                 the live state, so the child would come up stale. Fork at a wave boundary."
            );
        }
        if self.has_pending() {
            candle::bail!(
                "recurrent store: fork_from with timeline rows pending — fold or discard \
                 them first, or the child comes up a span behind its parent"
            );
        }
        let mut slots = Vec::with_capacity(self.slots.len());
        // The child's memory comes from the state arenas for the same reason the
        // parent's does — a fork is another sequence, and at ~3 forks per turn
        // this was ~126 MiB of pool traffic each.
        #[cfg(feature = "cuda")]
        let mut claimed =
            claim_layer_states(&self.dims, &self.device, self.slots.len())?.map(Vec::into_iter);
        for slot in &self.slots {
            #[cfg(feature = "cuda")]
            let live = match claimed.as_mut() {
                Some(it) => {
                    let half = Half::in_slot(
                        &self.dims,
                        &self.device,
                        it.next().expect("one slot was claimed per recurrent layer"),
                    )?;
                    // The fork's whole point: the child starts from the parent's
                    // state. A device-to-device copy into the child's own slot,
                    // rather than `snapshot()`, which would allocate a fresh pool
                    // buffer and hand back a tensor pointing outside the span.
                    copy_state_into(&self.device, &slot.live.state, &half.state)?;
                    half
                }
                None => Half {
                    state: slot.live.state.snapshot()?,
                    timeline: TimelineLayout::of(&self.dims).alloc(&self.device)?,
                    held: None,
                },
            };
            #[cfg(not(feature = "cuda"))]
            let live = Half {
                state: slot.live.state.snapshot()?,
                timeline: TimelineLayout::of(&self.dims).alloc(&self.device)?,
            };
            slots.push(LayerSlot {
                layer_index: slot.layer_index,
                live,
                scratch: None,
                pending: None,
                advanced: false,
            });
        }
        Ok(Self {
            dims: self.dims,
            hash: self.hash,
            slots,
            open: false,
            ran_this_wave: false,
            long_this_wave: false,
            seeded: true,
            step: None,
            device: self.device.clone(),
        })
    }

    /// Move every half of this store whose slot is the source of a planned move onto
    /// that move's destination, taking the destination out of `moves`; answers how
    /// many halves moved.
    ///
    /// Per half: the half's tensors rebuilt on the destination slot, the source slot
    /// dropped. A live half's state and conv tail are copied first — one device copy
    /// on the primary stream — and its timeline with them when the layer holds rows
    /// nothing has folded yet: a decoding sequence carries them from step to step
    /// until a fold, so between steps is not between folds, and a store that refused
    /// to move while any were pending pinned the frontier under every running decode
    /// (WhiteWorld, f66456bc2, 2026-10-19: recurrent state on the top region over 13
    /// to 18 holes, every lowering gaining nothing). A scratch is never copied: a
    /// long span overwrites it whole before reading. The drop is on the host while the copy may still be queued,
    /// which is sound for the reason `ArenaSlot` gives — the source's next tenant is
    /// ordered behind the copy on the same stream, or, if it empties its arena,
    /// behind the region pool's fence. Anything else still holding the old half
    /// keeps the old slot alive until it lets go.
    #[cfg(feature = "cuda")]
    pub fn relocate(&mut self, moves: &mut HashMap<u64, ArenaSlot>) -> Result<usize> {
        if self.open {
            candle::bail!(
                "recurrent store: relocate mid-wave — the wave has resolved the store's \
                 addresses"
            );
        }
        let Device::Cuda(cuda) = &self.device else {
            return Ok(0);
        };
        let stream = cuda.cuda_stream();
        let block = state_block(&self.dims);
        let mut moved = 0usize;
        for slot in &mut self.slots {
            // The state and its tail, and the whole block when timeline rows wait
            // in it to be folded.
            let copy = if slot.pending.is_some() {
                block.bytes
            } else {
                block.state_end
            };
            let halves = std::iter::once((&mut slot.live, true))
                .chain(slot.scratch.as_mut().map(|s| (s, false)));
            for (half, carries_state) in halves {
                let Some(at) = half.held.as_ref() else {
                    continue;
                };
                let Some(dst) = moves.remove(&at.ptr()) else {
                    continue;
                };
                if carries_state {
                    // SAFETY: both ranges are state-arena slots of at least `copy`
                    // bytes — the source held by this store, the destination claimed
                    // for this move — and they are distinct slots, so they do not
                    // overlap.
                    unsafe { memcpy_dtod_async(dst.ptr(), at.ptr(), copy, stream.cu_stream()) }
                        .map_err(|e| Error::Msg(format!("relocating recurrent state: {e}")))?;
                }
                *half = Half::in_slot(&self.dims, &self.device, Arc::new(dst))?;
                moved += 1;
            }
        }
        Ok(moved)
    }

    /// Reservation bytes this sequence's recurrent memory holds — its state-arena
    /// slots, live and scratch, at their stride. Zero off CUDA, where the buffers
    /// are ordinary allocations.
    ///
    /// A per-sequence figure, so it leaves out what no one sequence holds: an arena's
    /// free slots and unused tail, and slots kept by a handle that outlived its
    /// store. [`Self::arena_reserved_bytes`] is the whole-card figure that includes
    /// them.
    pub fn reserved_bytes(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.slots
                .iter()
                .flat_map(|s| std::iter::once(&s.live).chain(s.scratch.as_ref()))
                .filter_map(|h| h.held.as_ref())
                .map(|s| s.stride())
                .sum()
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    /// Reservation bytes every recurrent-state arena on `device` holds, as whole
    /// regions, whatever is in them.
    ///
    /// **The accounting figure.** It is what recurrent state denies the rest of the
    /// span: the slots every store holds, the free slots and unused tails of their
    /// arenas, and any slot a handle kept after its store went. Memory nothing can
    /// total is memory that goes missing (`AccountingSection`), and summing stores
    /// would miss all but the first. Zero off CUDA.
    pub fn arena_reserved_bytes(device: &Device) -> usize {
        #[cfg(feature = "cuda")]
        {
            arena_regions(device, SlotTenant::RecurrentState) * SpanRegion::bytes()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = device;
            0
        }
    }

    /// What one decoding sequence's store **will** reserve, from the geometry
    /// alone — [`Self::reserved_bytes`] for a store that does not exist yet and
    /// holds no scratch.
    ///
    /// **This is the figure admission needs, and it is the one a live store
    /// cannot give.** A store is priced before it is built, and at that moment
    /// there may be none in the process to measure: the first sequence of a
    /// session, or the first after a seal evicted every store.
    ///
    /// Deriving it from residency instead — summing the live stores and
    /// dividing by what the scheduler has in flight — is not a per-sequence
    /// figure at all, because the two counts range over different populations.
    /// The sum covers every store the process holds, parked conversations
    /// included; the divisor covers only what is in flight, and admission runs
    /// *between* forwards, where that is 0 or 1. The quotient therefore rises
    /// with the number of idle conversations and peaks when the engine is
    /// quiet, which is precisely when admission should be cheapest. Measured on
    /// a 72 GB card: a 41-row turn priced at 4,450 MiB and refused as
    /// throughput-worse on fifteen consecutive passes, seven turns queued
    /// behind it and 20 GiB standing free above the floor.
    ///
    /// The price is the store's slots at what they claim: one per DeltaNet
    /// layer, each the layer's block (`s`, its conv tail and its timeline, each
    /// on a 256-byte boundary) at its [`slot_share_bytes`] — its share of the
    /// region its arena stands on, the region's unused tail included. A
    /// prefilling sequence holds [`Self::scratch_bytes_for`] more. Priced at the
    /// stride instead, as [`Self::reserved_bytes`] reports a standing store, the
    /// tails were ground every store claimed and no price carried: 8.0 % of each
    /// recurrent-state region on Flash-Next, a cohort's two regions short at its
    /// second window (see [`slot_share_bytes`]). Nothing here reads the device,
    /// so it answers on any backend and at any moment.
    pub fn reserved_bytes_for(layer_kinds: &[LayerKind], dims: &DeltaNetDims) -> usize {
        #[cfg(feature = "cuda")]
        {
            let layers = layer_kinds
                .iter()
                .filter(|k| **k == LayerKind::DeltaNet)
                .count();
            layers * slot_share_bytes(state_block(dims).bytes)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (layer_kinds, dims);
            0
        }
    }

    /// What the prefill scratch adds to a sequence's store while it prefills —
    /// one more slot per DeltaNet layer.
    pub fn scratch_bytes_for(layer_kinds: &[LayerKind], dims: &DeltaNetDims) -> usize {
        Self::reserved_bytes_for(layer_kinds, dims)
    }

    /// Whether this store's state arrived by fork or restore and must survive
    /// its first `offset == 0` reset. Consumed by that reset — see
    /// [`Self::take_seeded`].
    pub fn is_seeded(&self) -> bool {
        self.seeded
    }

    /// Mark the state as externally seeded (a restore).
    pub fn mark_seeded(&mut self) {
        self.seeded = true;
    }

    /// Read and clear the seeded flag: `true` exactly once after a fork or a
    /// restore, and the caller must then not reset the store.
    ///
    /// The flag exists because "was this slot's state put here deliberately?"
    /// has no other answer. `ensure_recurrent` resets on `offset == 0` because
    /// a sequence with no history must hold the sequence-start value, and a
    /// freshly restored slot standing at offset 0 before its first wave looks
    /// exactly like one. Relying on the projection to have moved the offset
    /// first is correct today by ordering nothing asserts; this makes it
    /// explicit.
    pub fn take_seeded(&mut self) -> bool {
        std::mem::take(&mut self.seeded)
    }
}
