//! Builds and commits the chunks that ingest sections, each section advancing
//! by its own remaining tokens within one forward budget.
//!
//! Sections are packed into one ragged forward per chunk. Each advances by the
//! smaller of its remaining tokens and its cap, so one nearly finished section
//! does not hold the others back to its length. The total is bounded by the
//! per-forward budget so the activation peak stays bounded, and sections that
//! have not started claim their recurrent store only as the ground pays for it.
//! A section chunk yields to a waiting dialogue prompt the way a creep does, and
//! a section whose chunks are all committed is sealed and hands its scratch
//! slot back.

use super::*;

impl Scheduler {
    /// Build one ragged section-ingest chunk: for each active section, its next
    /// `min(remaining, cap)` tokens, packed until the per-forward token budget.
    /// Returns `(seq_ids, inputs, group_idxs, advances)`, or `None` when nothing
    /// is pending. Shared by the standalone pass and the co-batched decode wave.
    ///
    /// Ragged batch: each section advances by its OWN min(remaining, cap). The
    /// varlen forward packs the heterogeneous lengths flat, so one near-finished
    /// section no longer collapses the whole wave to the batch minimum — the bug
    /// that dragged a 93-wide tool-catalog ingest down to ~1 token/seq/forward.
    ///
    /// Bound the TOTAL tokens to the same per-forward budget a normal prefill
    /// targets ([`Self::prefill_pass_budget`]). Without this the whole active set
    /// coalesces into one forward: the 93-section tool catalog (~21k tokens)
    /// packed into a single pass whose transient activation spiked VRAM to the
    /// card ceiling and paged. Sections beyond the budget ride the next chunk —
    /// the wave loop pumps until every section seals — so throughput is unchanged
    /// (each forward still fills to the expert-amortization target) while the peak
    /// stays bounded. At least one section is always admitted so the wave makes
    /// progress.
    ///
    /// Bound the sections that have not started, too: each claims a recurrent
    /// store as it enters, which tokens do not measure (`section_pack`).
    #[allow(clippy::type_complexity)]
    pub(in super::super) fn build_section_batch(
        &mut self,
    ) -> Option<(Vec<usize>, Vec<Tensor>, Vec<usize>, Vec<usize>)> {
        // Sections already creeping inside the wave group are excluded — their
        // offset isn't advanced until that group's head, so picking them here would
        // ingest the same chunk twice.
        let band = self.section_batch_band()?;
        let in_flight = self.wave_group_section_seqs();
        let active: Vec<usize> = (0..self.active_section_ingests.len())
            .filter(|&i| {
                let s = &self.active_section_ingests[i];
                s.error.is_none()
                    && s.offset < s.tokens.len()
                    && !in_flight.contains(&s.sequence_id.0)
                    && (!band.is_live() || s.band.is_live())
            })
            .collect();
        if active.is_empty() {
            return None;
        }
        let cap = self.prefill_pass_budget();
        // A section not yet under way claims its recurrent store as it enters
        // the forward, so the forward takes only as many as the ground above
        // the weight floor pays for — see `section_pack`.
        let candidates: Vec<section_pack::Candidate> = active
            .iter()
            .map(|&i| {
                let s = &self.active_section_ingests[i];
                section_pack::Candidate {
                    remaining: s.tokens.len() - s.offset,
                    fresh: s.offset == 0,
                }
            })
            .collect();
        let room = self.admit_headroom(self.standing_ground_bytes()).free_kv;
        let fresh_allowed = section_pack::fresh_allowance(room, self.recurrent_cost());
        let packed = section_pack::pack(&candidates, cap, fresh_allowed);
        let mut seq_ids: Vec<usize> = Vec::with_capacity(packed.len());
        let mut inputs: Vec<Tensor> = Vec::with_capacity(packed.len());
        let mut group_idxs: Vec<usize> = Vec::with_capacity(packed.len());
        let mut advances: Vec<usize> = Vec::with_capacity(packed.len());
        for (k, advance) in packed {
            let i = active[k];
            let s = &mut self.active_section_ingests[i];
            let off = s.offset;
            let tokens = &s.tokens[off..off + advance];
            // On the host: the forward lays every row's id on its own span in
            // one upload (`wave_token_ids`).
            match Tensor::new(tokens, &Device::Cpu).and_then(|t| t.unsqueeze(0)) {
                Ok(t) => {
                    seq_ids.push(s.sequence_id.0);
                    inputs.push(t);
                    group_idxs.push(i);
                    advances.push(advance);
                }
                Err(e) => {
                    s.error = Some(ConversationError::Model(e));
                }
            }
        }
        if seq_ids.is_empty() {
            return None;
        }
        Some((seq_ids, inputs, group_idxs, advances))
    }

    /// The band the next section chunk rides at ([`Self::build_section_batch`]),
    /// `None` with no section to ride: `High` while a dialogue's own section
    /// waits — it rides apart from ingest rows, as in a creep group
    /// (`form_wave_group`) — and otherwise the highest band waiting, since below
    /// `High` the bands mix. `None` too while a prompt of a higher band can
    /// advance.
    ///
    /// **Background yields to a waiting prompt here as a creep does**
    /// (`creep_outranked_by_a_waiting_prompt`). A chunk riding a step beside
    /// decode owes decode its share times the ratio its band yields at — 64 for
    /// a schema's `Low` ingest beside `High` decodes — and that debt holds every
    /// creep, whatever its band, until it is paid: on WhiteWorld (2026-10-26)
    /// one ingest chunk owed ~90 s, and four Makers' `High` prompts sat frozen at
    /// an 8,275-row backlog behind it while their decodes ran.
    pub(in super::super) fn section_batch_band(&self) -> Option<DecodePriority> {
        let in_flight = self.wave_group_section_seqs();
        let band = self
            .active_section_ingests
            .iter()
            .filter(|s| {
                s.error.is_none()
                    && s.offset < s.tokens.len()
                    && !in_flight.contains(&s.sequence_id.0)
            })
            .map(|s| s.band)
            .max_by_key(|&b| rank(b))?;
        (!self.waiting_above(band, |_| false)).then_some(band)
    }

    /// Commit one section-ingest chunk after its forward (standalone or
    /// co-batched): advance each section by its own `advance`, record its slot
    /// tokens, and bump its offset. Section logits are never used (no decode).
    pub(in super::super) fn complete_section_chunk(
        &mut self,
        group_idxs: &[usize],
        advances: &[usize],
    ) {
        for (&i, &advance) in group_idxs.iter().zip(advances.iter()) {
            let s = &mut self.active_section_ingests[i];
            if let Err(e) = self.session.advance_sequence(s.sequence_id.0, advance) {
                s.error = Some(ConversationError::Model(e));
                continue;
            }
            let seq_id = s.sequence_id;
            let off = s.offset;
            let chunk_tokens = s.tokens[off..off + advance].to_vec();
            super::super::Scheduler::record_slot_tokens(
                &mut self.slot_tokens,
                seq_id,
                &chunk_tokens,
            );
            s.offset += advance;
        }
    }

    /// Drain completed or errored section ingest entries. Errored entries send
    /// `Err`; finished entries call `finalize_section_ingest` (seal + write)
    /// and send the `SealResult`.
    pub(in super::super) fn finalize_done_section_ingests(&mut self) {
        let mut i = 0;
        while i < self.active_section_ingests.len() {
            let done = {
                let s = &self.active_section_ingests[i];
                s.error.is_some() || s.offset >= s.tokens.len()
            };
            if !done {
                i += 1;
                continue;
            }
            let s = self.active_section_ingests.swap_remove(i);
            // A sealed section hands its scratch slot back — a completion like
            // any other, and an admission opportunity.
            self.settled_since_admit = true;
            if let Some(e) = s.error {
                let _ = s.response_tx.send(Err(e));
                continue;
            }
            let sealing = Instant::now();
            let tokens = s.tokens.len();
            let name = s.debug_name.clone();
            let result = self.finalize_section_ingest(
                s.sequence_id,
                s.section_id,
                s.seal_block_from,
                Arc::new(s.tokens.to_vec()),
                s.address,
                s.debug_name,
                s.in_collection,
                s.export_end,
            );
            let split = s.timing.split(sealing, Instant::now());
            tracing::info!(
                target: "candle_conversation::scheduler::section",
                section = %name,
                tokens,
                band = ?s.band,
                queue_us = split.queue_us,
                prepare_us = split.prepare_us,
                forward_us = split.forward_us,
                seal_us = split.seal_us,
                "section ingest split"
            );
            let _ = s.response_tx.send(result);
            // swap_remove pulled the last element into i; don't increment.
        }
    }
}
