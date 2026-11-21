//! Runs a wave over a store's layers: opens it, advances each layer into its
//! timeline or its scratch, and then commits or rolls it back.
//!
//! A wave writes timelines and scratch but never a live state, so rolling a wave
//! back drops only what it wrote. Committing exchanges the scratch of each advanced
//! long span into the live state, while short spans stay pending until a fold.
//! The state assertions run asynchronously, so they do not suppress the race they
//! are there to catch.

use super::*;

impl RecurrentStateStore {
    /// The layer's live state, to be read by a span of `len` rows, and where the
    /// span's work lands — and the record that this layer ran.
    ///
    /// - A span of at most [`TIMELINE_ROWS`] rows writes the layer's timeline,
    ///   which leaves the layer pending until a fold or a discard. A layer
    ///   already pending refuses: its rows have not been applied, and a second
    ///   span would read a state they have not reached.
    /// - A longer span writes the prefill scratch, which must already be held
    ///   ([`Self::ensure_scratch`], between forwards); `commit_wave` installs it.
    ///
    /// Taking this is what marks the layer, so a caller asks for it exactly when
    /// it is about to run the layer, never to peek.
    pub fn layer_span_mut(
        &mut self,
        layer_index: usize,
        len: usize,
    ) -> Result<(&mut DeltaNetState, SpanWrite)> {
        if !self.open {
            candle::bail!(
                "recurrent store: a span for layer {layer_index} with no wave open — a span's \
                 writes belong to a wave, which is what can roll them back"
            );
        }
        if len == 0 {
            candle::bail!("recurrent store: a span of no rows for layer {layer_index}");
        }
        let long = len > TIMELINE_ROWS;
        let at = self
            .slots
            .iter()
            .position(|s| s.layer_index == layer_index)
            .ok_or_else(|| no_slot(layer_index))?;
        let slot = &mut self.slots[at];
        if let Some(p) = slot.pending {
            candle::bail!(
                "recurrent store: layer {layer_index} still holds {} pending timeline rows — \
                 a {len}-row span would read a state they have not reached. Fold or discard \
                 them first.",
                p.len
            );
        }
        let write = if long {
            let Some(scratch) = slot.scratch.as_ref() else {
                candle::bail!(
                    "recurrent store: a {len}-row span for layer {layer_index} needs the \
                     prefill scratch, which the store does not hold — claim it before the \
                     forward (`ensure_scratch`)"
                );
            };
            slot.advanced = true;
            SpanWrite::State(scratch.state.write_half())
        } else {
            slot.pending = Some(Pending {
                len,
                this_wave: true,
            });
            SpanWrite::Timeline(slot.live.timeline.clone())
        };
        self.ran_this_wave = true;
        self.long_this_wave |= long;
        Ok((&mut self.slots[at].live.state, write))
    }

    /// Open a wave. Refuses while one is already open.
    ///
    /// **Costs nothing on the device.** The entering state is preserved by not
    /// being written: a short span writes its timeline and a long span the
    /// scratch, so opening a wave is bookkeeping and rolling one back is
    /// dropping that bookkeeping.
    pub fn begin_wave(&mut self) -> Result<()> {
        if self.open {
            candle::bail!(
                "recurrent store: begin_wave with a wave already open — overlapping \
                 waves on one session are exactly what atomicity forbids"
            );
        }
        for slot in &mut self.slots {
            slot.advanced = false;
            if let Some(p) = slot.pending.as_mut() {
                p.this_wave = false;
            }
        }
        self.ran_this_wave = false;
        self.long_this_wave = false;
        self.open = true;
        self.assert_state("gdn.entry.s.L", "gdn.entry.tail.L");
        Ok(())
    }

    /// Fold every layer's live state into the assert slots under `prefix`.
    ///
    /// **Entry and exit, because the pair is what says who broke it.** A wave
    /// writes nothing live, so the state at entry is whatever the last fold or
    /// commit left plus anything that has happened to the memory since. If entry
    /// is bad while the previous exit was good, nothing computed it — the bytes
    /// changed while the store sat idle, which is a foreign write into the span.
    /// If exit is the first bad site, the wave computed it, and the cause is
    /// upstream in the same forward.
    ///
    /// Asynchronous: one reduction kernel per buffer, no readback and no fence. The
    /// per-wave drain in `wave_driver` reports the slots and names the first bad
    /// site by the kernel's own ticket, so the ordering this depends on is the
    /// device's, not the host's. That matters more than it looks — these faults stop
    /// reproducing in a fenced build, so an instrument that synchronised here would
    /// suppress the thing it is watching for.
    #[cfg(feature = "tensor-assert")]
    pub(super) fn assert_state(&self, s_prefix: &'static str, tail_prefix: &'static str) {
        use candle::tensor_assert::names::site;
        for slot in &self.slots {
            let l = slot.layer_index;
            slot.live.state.s.assert(site(s_prefix, l));
            slot.live.state.conv_tail.assert(site(tail_prefix, l));
        }
    }

    #[cfg(not(feature = "tensor-assert"))]
    #[inline]
    pub(super) fn assert_state(&self, _s_prefix: &'static str, _tail_prefix: &'static str) {}

    /// The wave's writes stand: every layer a long span advanced exchanges its
    /// live half with the scratch, so what the span wrote becomes the state and
    /// what the state was becomes the next long span's scratch. Layers a short
    /// span ran stay pending — their timelines wait for a fold.
    ///
    /// A host exchange per advanced layer, and no device work at all. A wave that
    /// ran layers and handed none of them to a long span ends the sequence's
    /// prefill, and the scratch goes back; one that ran no layer — a segment of
    /// attention layers only — leaves it.
    pub fn commit_wave(&mut self) {
        // An exchange inside a step is not one an undo can take back.
        if self.slots.iter().any(|s| s.advanced) {
            self.break_step();
        }
        for slot in &mut self.slots {
            if let Some(p) = slot.pending.as_mut() {
                p.this_wave = false;
            }
            if slot.advanced {
                let scratch = slot
                    .scratch
                    .as_mut()
                    .expect("a layer advanced by a long span holds the scratch it wrote");
                // The whole half: the state, its tail and the timeline region of
                // the slot move together, so the live slot always carries a
                // timeline beside its state.
                std::mem::swap(&mut slot.live, scratch);
                slot.advanced = false;
            }
        }
        self.open = false;
        if self.ran_this_wave && !self.long_this_wave {
            self.release_scratch();
        }
        // After the exchange, so this is the state the NEXT wave will read — the
        // same buffers `gdn.entry` will fold on the way in. A pair that disagrees
        // across the gap between two waves is a write nothing in the forward
        // performed.
        self.assert_state("gdn.exit.s.L", "gdn.exit.tail.L");
    }

    /// The wave never happened.
    ///
    /// Nothing to undo on the device: a wave writes only timelines and the
    /// scratch, never a live state. The timelines this wave wrote are dropped;
    /// rows an earlier, committed segment left pending stay. Refuses when no
    /// wave is open (a rollback with nothing to roll back to is a sequencing
    /// bug, not a no-op).
    pub fn rollback_wave(&mut self) -> Result<()> {
        if !self.open {
            candle::bail!("recurrent store: rollback_wave with no wave open");
        }
        for slot in &mut self.slots {
            slot.advanced = false;
            if slot.pending.is_some_and(|p| p.this_wave) {
                slot.pending = None;
            }
        }
        self.open = false;
        Ok(())
    }

    /// Whether any layer holds timeline rows nothing has applied yet.
    pub fn has_pending(&self) -> bool {
        self.slots.iter().any(|s| s.pending.is_some())
    }

    /// Every pending layer and its span's rows, in slot order.
    pub fn pending_layers(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.slots
            .iter()
            .filter_map(|s| s.pending.map(|p| (s.layer_index, p.len)))
    }

    /// The fold entries for every pending layer, keeping the first `kept` rows
    /// of each timeline (every row for `None`). A layer keeping none has no
    /// entry.
    #[cfg(feature = "cuda")]
    pub(super) fn fold_entries(&self, kept: Option<usize>) -> Vec<FoldEntry> {
        let block = state_block(&self.dims);
        self.slots
            .iter()
            .filter_map(|slot| {
                let p = slot.pending?;
                let m = kept.map_or(p.len, |k| k.min(p.len));
                let held = slot.live.held.as_ref()?;
                (m > 0).then(|| FoldEntry {
                    tail: held.ptr() + block.conv_off as u64,
                    state: held.ptr(),
                    timeline: held.ptr() + block.timeline_off as u64,
                    m,
                })
            })
            .collect()
    }

    /// Fold this store's pending layers — [`fold_stores`] for one store.
    pub fn fold(&mut self, kept: Option<usize>) -> Result<()> {
        fold_stores(&mut [(self, kept)])
    }

    /// Drop every pending timeline: the spans that wrote them are not kept, and
    /// the state stands as they found it.
    pub fn discard_pending(&mut self) {
        for slot in &mut self.slots {
            slot.pending = None;
        }
    }

    /// Hold the prefill scratch: a second slot per layer, where a long span
    /// writes its advanced state. **Between forwards** — a claim from inside a
    /// forward is refused by the arena — so a step's prepare pass takes it for
    /// every sequence it hands a long span. Held until a wave runs the sequence
    /// with no long span, or [`Self::release_scratch`].
    ///
    /// Not zeroed: a long span overwrites the scratch whole before anything
    /// reads it (hot-path invariant 6).
    pub fn ensure_scratch(&mut self) -> Result<()> {
        if self.open {
            candle::bail!("recurrent store: claiming the prefill scratch mid-wave");
        }
        let missing = self.slots.iter().filter(|s| s.scratch.is_none()).count();
        if missing == 0 {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        let mut claimed =
            claim_layer_states(&self.dims, &self.device, missing)?.map(Vec::into_iter);
        for slot in self.slots.iter_mut().filter(|s| s.scratch.is_none()) {
            #[cfg(feature = "cuda")]
            let half = match claimed.as_mut() {
                Some(it) => Half::in_slot(
                    &self.dims,
                    &self.device,
                    it.next().expect("one slot was claimed per missing scratch"),
                )?,
                None => Half::host(&self.dims, &self.device, false)?,
            };
            #[cfg(not(feature = "cuda"))]
            let half = Half::host(&self.dims, &self.device, false)?;
            slot.scratch = Some(half);
        }
        Ok(())
    }

    /// Give the prefill scratch back. Its slots return to their arena once no
    /// view of them remains.
    pub fn release_scratch(&mut self) {
        for slot in &mut self.slots {
            slot.scratch = None;
        }
    }

    /// Whether the store holds the prefill scratch.
    pub fn holds_scratch(&self) -> bool {
        self.slots.iter().any(|s| s.scratch.is_some())
    }
}
