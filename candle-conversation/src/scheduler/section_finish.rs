//! Finishes a section ingest: it seals the section's K/V with its index page,
//! applies the compression policy for its kind, and quantizes pending sections at
//! a safe boundary.
//!
//! Boundary sections, which every later token attends back over, take an 8-bit
//! policy because any noise in them is amplified through every layer. Collection
//! members take the adaptive policy at their collection's level, since the
//! projection's selection already masks their per-member precision
//! ([`SectionSeal`]). Pending sections are quantized only at a turn-seal
//! boundary, after every reader of their native form has finished, and the pass
//! is deferred while a hot-to-warm migrate is converting the same residences. A
//! section whose seal fails on a per-layer window divergence is tombstoned only
//! when its own persisted data confirms the divergence.

use std::collections::BTreeMap;

use super::*;

impl Scheduler {
    /// Seal and write a completed section ingest. Called after the batched
    /// forward pass has consumed all of the section's tokens.
    pub(in super::super) fn finalize_section_ingest(
        &mut self,
        sequence_id: SequenceId,
        section_id: SectionId,
        seal_block_from: usize,
        tokens: Arc<Vec<u32>>,
        address: ContentAddress,
        debug_name: String,
        seal: SectionSeal,
        export_end: bool,
    ) -> Result<SealResult, ConversationError> {
        #[cfg(feature = "context-dump")]
        if tracing::enabled!(
            target: "candle_conversation::scheduler::section_dump",
            tracing::Level::INFO,
        ) {
            let slot_chunk_count_at_seal = self
                .session
                .sequence_block_count(sequence_id.0)
                .unwrap_or(0);
            let slot_offset_at_seal = self.session.sequence_offset(sequence_id.0).unwrap_or(0);
            tracing::info!(
                target: "candle_conversation::scheduler::section_dump",
                section_id = ?section_id,
                seal_block_from,
                slot_chunk_count_at_seal,
                slot_offset_at_seal,
                "section seal: pre-seal slot stats"
            );
        }
        // Read before the move — the warn below needs the width, and the seal
        // takes ownership of the buffer.
        let n_tokens = tokens.len();
        let seal = self.perform_seal_and_write(
            sequence_id,
            sequence_id,
            seal_block_from,
            &SealAction::Section {
                section_id,
                tokens,
                address,
                debug_name,
                seal,
            },
            None,
        )?;
        let seal = seal.ok_or_else(|| {
            ConversationError::Channel(
                "ingest_section: seal returned None (slot had no content?)".into(),
            )
        })?;

        // The page that goes with the chunks just sealed. Taken here, at the
        // seal, because this is where the section's own forward has finished
        // and before the slot is reused — and closed on a block boundary, so
        // the piece covers its own tokens and no others.
        match self.model.seal_positional_state(sequence_id.0) {
            Ok(Some(blob)) => {
                // Persisted beside the section's chunks, under the same stream
                // id, so a restart that restores the section from the log can
                // hand a projection its index as well as its K/V.
                match self.slot_conversations.get(&sequence_id) {
                    Some(conversation) => {
                        let stream = section_stream_id(address);
                        conversation.enqueue_index_page(stream, blob.clone());
                        self.section_positional.insert(
                            stream,
                            conversation,
                            Arc::new(StoredPages::new(blob)),
                        );
                    }
                    None => tracing::warn!(
                        section = section_id.raw(),
                        seq = sequence_id.0,
                        "section index page neither kept nor persisted: the slot has no \
                         conversation, so nothing that borrows this section is indexed and a \
                         restart prefills it again instead of restoring it",
                    ),
                }
            }
            // **`None` from a model that keeps per-position state is a section
            // with no index.** It used to be swallowed, and that silence is what
            // let an entire system prompt reach the model unindexed: every
            // section returned `None`, the assembler had no blob to push, and
            // the shortfall surfaced only when the prefix grew past the QSA
            // identity threshold and a select refused — thousands of tokens and
            // one conversation later, naming the slot rather than the seal.
            Ok(None) if self.model.carries_positional_state() => tracing::warn!(
                section = section_id.raw(),
                seq = sequence_id.0,
                n_tokens,
                "section sealed with NO index page — the model holds no per-position \
                 state for this slot, so anything borrowing its tokens selects against \
                 a prefix it never indexed",
            ),
            // A model that keeps no per-position state seals no page, and
            // nothing is missing.
            Ok(None) => {}
            Err(e) => {
                // Not fatal to the seal — the K/V is committed — but the next
                // ingest to borrow this section will have to say so.
                tracing::warn!(
                    "section {:?}: index page not sealed ({e}); an ingest borrowing it will \
                     select against a prefix it never indexed",
                    section_id
                );
            }
        }

        // A section that began from its prefix's state hands back the state it
        // ended on, before the slot is reused: the start of a section that
        // continues it.
        let mut seal = seal;
        if export_end {
            seal.end_state = self.export_state_of(sequence_id.0)?;
        }
        Ok(seal)
    }

    /// Drain [`Self::pending_section_quantize`]: re-read each section's
    /// current native hot bytes from the substrate, quantize them
    /// per-layer under each section's own seal ([`SectionSeal::policy`]), and
    /// atomically replace the residence's hot with the quantized form.
    ///
    /// Called from the `SealAction::Turn` handler, on the main scheduler
    /// thread, *after* the turn record has been committed.  At that
    /// instant every in-flight reader of the native sections has
    /// finished — no other thread (including the persistence thread)
    /// holds a stale assumption about the section's contents — so the
    /// hot-replace is safe.  See the call site for the full ordering
    /// argument.
    pub(super) fn quantize_pending_sections(
        &mut self,
        conversation: &Conversation,
    ) -> Result<(), ConversationError> {
        // Defer while a hot→warm migrate is in flight: it is converting the
        // same residences float→quant, so quantizing them here would redo work
        // already under way. Leave the sections pending; they drain next wave.
        //
        // This was once a safety deferral — the migrate's per-head table was
        // dense over storage, so freeing any arena under it left the persist
        // convert kernel reading a stale base pointer (illegal address, layer 2
        // @42553ca3). The table is sized from the job list now, so every
        // pointer in it comes from a gid the migrate has pinned, and a pinned
        // arena cannot be freed. What remains is the duplicated work.
        if candle_nn::kv_cache::migrate_in_flight() {
            return Ok(());
        }
        let (drained, keep): (Vec<_>, Vec<_>) = take(&mut self.pending_section_quantize)
            .into_iter()
            .partition(|p| p.substrate.is(conversation));
        self.pending_section_quantize = keep;
        self.quantize_section_batch(conversation, drained, false)
    }

    /// Quantize + offload ONLY the collection-member sections currently
    /// pending, leaving boundary sections pending for the turn-seal boundary.
    ///
    /// Collection members are prefix-transparent — nothing attends back over
    /// them during the build (priming uses the collections-empty projection),
    /// so it is safe to quantize them mid-build the moment their ingest wave
    /// has sealed.  This is what lets a large catalog (e.g. the tool list,
    /// sealed ×no_think) be shrunk and offloaded as the build runs instead of
    /// piling up native until the end.  Each member's residence is flagged
    /// `evict_when_cold` so the persistence thread frees its VRAM the instant
    /// the redo-log write lands.
    pub(super) fn quantize_pending_collection_members(
        &mut self,
        conversation: &Conversation,
    ) -> Result<(), ConversationError> {
        // See `quantize_pending_sections` — deferred during a hot→warm migrate
        // so the member quantize doesn't redo the conversion it is performing.
        if candle_nn::kv_cache::migrate_in_flight() {
            return Ok(());
        }
        let mut members: Vec<PendingSectionQuantize> = Vec::new();
        let mut keep: Vec<PendingSectionQuantize> = Vec::new();
        for p in self.pending_section_quantize.drain(..) {
            if p.seal.is_member() && p.substrate.is(conversation) {
                members.push(p);
            } else {
                keep.push(p);
            }
        }
        self.pending_section_quantize = keep;
        self.quantize_section_batch(conversation, members, true)
    }

    /// Quantize an already-drained pending list and atomically swap each
    /// residence's hot to its quantized form.  `mark_evict` additionally flags
    /// every section's residence for offload-on-persist (collection-member path).
    pub(super) fn quantize_section_batch(
        &mut self,
        conversation: &Conversation,
        pending_list: Vec<PendingSectionQuantize>,
        mark_evict: bool,
    ) -> Result<(), ConversationError> {
        // Snapshot per-section native hot + residence + its seal under a
        // brief read lock; the heavy GPU work runs unlocked. The per-layer
        // loop below groups by seal so each policy gets its own batched
        // launch. Each section's native hot copy as the drain reads it: what
        // it quantizes, and what its install replaces only while still there.
        let pending: Vec<(SectionId, ResidenceIndex, Arc<HotKv>, SectionSeal)> = {
            let view = conversation.read();
            pending_list
                .into_iter()
                .filter_map(|p| {
                    let residence = view.section_residence(p.section_id)?;
                    let native = view.residence(residence).hot()?;
                    Some((p.section_id, residence, native, p.seal))
                })
                .collect()
        };
        if pending.is_empty() {
            return Ok(());
        }

        let n_layers = self.session.num_layers();
        let backings = self.session.backings();
        let device = self.session.device().clone();
        let copy_stream = match &device {
            Device::Cuda(d) => d.cuda_stream(),
            _ => {
                return Err(ConversationError::Channel(
                    "section quantize requires a CUDA device".into(),
                ))
            }
        };

        // Split pending into two groups so each gets its own batched
        // per-layer launch under its own policy.  `indices_*` records
        // the position of each section in the original `pending` list
        // so we can stitch the per-group results back together in the
        // install loop without losing the original order.
        let mut by_seal: BTreeMap<SectionSeal, Vec<usize>> = BTreeMap::new();
        for (i, (_, _, _, seal)) in pending.iter().enumerate() {
            by_seal.entry(*seal).or_default().push(i);
        }
        tracing::debug!(
            target: "candle_conversation::scheduler::section_quantize",
            groups = ?by_seal.iter().map(|(s, v)| (*s, v.len())).collect::<Vec<_>>(),
            "quantize drain: sections by seal"
        );

        // Batch across sections per layer for each group.
        // `quantize_sealed_in_place` returns one output SealedSequence
        // per input SealedSequence in the same order, so the per-group
        // result slot is filled positionally before being scattered
        // back into the global `quantized_per_section` Vec.
        let mut quantized_per_section: Vec<Vec<SealedSequence>> = (0..pending.len())
            .map(|_| Vec::with_capacity(n_layers))
            .collect();
        for (seal, group_indices) in &by_seal {
            let group_policy = &seal.policy();
            for layer in 0..n_layers {
                let inputs: Vec<&SealedSequence> = group_indices
                    .iter()
                    .map(|&idx| {
                        let hot = &pending[idx].2;
                        debug_assert_eq!(
                            hot.len(),
                            n_layers,
                            "pending section hot must have one SealedSequence per layer",
                        );
                        &hot[layer]
                    })
                    .collect();
                let out = quantize_sealed_in_place(
                    &backings[layer],
                    &inputs,
                    group_policy,
                    &device,
                    &copy_stream,
                    &mut self.quantize_pinned_scratch,
                )
                .map_err(ConversationError::Model)?;
                for (slot_idx, qi) in out.into_iter().enumerate() {
                    let global_idx = group_indices[slot_idx];
                    quantized_per_section[global_idx].push(qi);
                }
            }
        }

        copy_stream
            .synchronize()
            .map_err(|e| ConversationError::Channel(format!("section quantize sync: {e}")))?;

        // Replace every pending residence's hot with its quantized form —
        // each only over the native copy it was made from — and clear the
        // pending-quantize flag, so the persistence thread can now gather and
        // write the final bytes. Dropping the native copies afterwards
        // decrements the source chunks' refcounts to zero; their arena slots
        // return to the pool for reuse by the next prefill.
        {
            let view = conversation.read();
            for ((_, residence, native, _seal), q_per_layer) in
                pending.into_iter().zip(quantized_per_section)
            {
                if q_per_layer.len() != n_layers {
                    tracing::warn!(
                        "quantize_pending_sections: layer-count mismatch ({} vs {}), skipping section {:?}",
                        q_per_layer.len(),
                        n_layers,
                        residence
                    );
                    continue;
                }
                if !view.replace_section_hot(residence, &native, q_per_layer) {
                    // Retired or replaced while it quantized: the quantized
                    // copy is of bytes the section no longer holds.
                    continue;
                }
                view.clear_section_pending_quantize(residence);
                // Collection-member path: free this section's VRAM as soon as
                // the persistence thread lands its cold copy.
                if mark_evict {
                    view.mark_section_evict_when_cold(residence);
                }
            }
        }
        // Now that the substrate holds the final (quantized) form for
        // every drained section, wake the persistence thread so it
        // gathers them and appends to the redo log without waiting for
        // its 5 s tick.
        self.persist_triggers.fire();
        Ok(())
    }

    /// React to a `snapshot_sequence_per_layer` seal failure that names a
    /// per-layer token-window divergence — `assert_sealed_layers_aligned`'s
    /// "refusing to seal … different token windows" family
    /// (`candle-transformers/src/models/batched_inference.rs`). This is the
    /// production symptom of a shared base-conversation section whose
    /// persisted chunks disagree across layers at the same chunk index: the
    /// forward fails, the sequence is dropped, and every conversation
    /// forking that section hits the same wall on every attempt until
    /// something removes the bad data.
    ///
    /// **Narrow by design**, per the standing decision to fix the error
    /// handling rather than build a scanner: this fires on exactly this
    /// error family, and only when the failed seal was for a section
    /// (`SealAction::Section`) — a dialogue-turn seal failure is the
    /// existing `CorruptTurnPolicy` / `reconstruct_from_log` machinery's
    /// job and is untouched here.
    ///
    /// A live failure here is not by itself proof that the SEALING
    /// section's own persisted data is bad — the same divergence can
    /// originate in an already-resident PREFIX section this ingest
    /// borrowed as context, in which case the section actually being
    /// sealed has no persisted stream yet to be wrong. So rather than trust
    /// the live failure, this re-reads the sealing section's own persisted
    /// chunks off disk (`Conversation::check_section_window_integrity`,
    /// built on the generic, reusable
    /// `persistence::chunk_window_integrity::first_divergent_chunk`) and
    /// only tombstones on a CONFIRMED divergence in that data. An
    /// unconfirmed case (this section's own persisted data is clean, or it
    /// has none yet) is logged and left alone — the check this calls is
    /// generic precisely so a later, broader pass can reuse it against the
    /// prefix chain without this narrow hook growing into one.
    pub(super) fn repair_section_if_window_divergence_confirmed(
        &self,
        seal_slot: SequenceId,
        seal_action: &SealAction,
        seal_error: &candle::Error,
    ) {
        let SealAction::Section {
            section_id,
            address,
            ..
        } = seal_action
        else {
            return;
        };
        // Matched on `assert_sealed_layers_aligned`'s own shared marker
        // constant (not a duplicated literal) so an unrelated
        // `snapshot_sequence_per_layer` failure (e.g. a slot that was never
        // allocated) is left alone instead of treated as corruption — the
        // trigger stays exactly this one error class, and the two sides
        // can't silently drift apart if the message is ever reworded.
        if !seal_error.to_string().contains(WINDOW_DIVERGENCE_MARKER) {
            return;
        }
        let Some(conversation) = self.slot_conversations.get(&seal_slot).cloned() else {
            tracing::warn!(
                section_id = ?section_id,
                seal_slot = seal_slot.0,
                "section window-divergence repair: no conversation registered for the failed slot",
            );
            return;
        };
        let stream_id = section_stream_id(*address);
        let n_layers = self.session.num_layers();
        match conversation.check_section_window_integrity(stream_id, n_layers) {
            Ok(Some(divergence)) => {
                let reason = divergence.describe();
                tracing::error!(
                    section_id = ?section_id,
                    stream_id = stream_id.0,
                    seal_slot = seal_slot.0,
                    divergence = %reason,
                    "section window-divergence CONFIRMED on persisted data — tombstoning so \
                     the next request re-prefills fresh instead of restoring the corrupted \
                     chunks",
                );
                if let Err(e) = conversation.tombstone_section(stream_id, Some(&reason)) {
                    tracing::error!(
                        section_id = ?section_id,
                        stream_id = stream_id.0,
                        "section window-divergence repair: tombstone write failed: {e}",
                    );
                }
            }
            Ok(None) => {
                tracing::warn!(
                    section_id = ?section_id,
                    stream_id = stream_id.0,
                    seal_slot = seal_slot.0,
                    "seal failed on a per-layer window divergence, but this section's own \
                     persisted chunks read back clean (or it has none yet) — the divergence's \
                     root chunk is most likely in an already-resident prefix section this \
                     ingest borrowed, which this narrow hook does not trace; left unrepaired",
                );
            }
            Err(e) => {
                tracing::error!(
                    section_id = ?section_id,
                    stream_id = stream_id.0,
                    "section window-divergence repair: integrity re-read failed: {e}",
                );
            }
        }
    }
}
