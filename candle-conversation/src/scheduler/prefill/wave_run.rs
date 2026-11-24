//! Completes, fails and reconciles a wave-group forward, so each member gets its
//! result or the error it earned.
//!
//! A group that reaches its final layer commits each member's chunk, and samples
//! a finished prefill's first token from its row while the forward still holds
//! it. A failure is routed by its kind: a device out-of-memory requeues the
//! prefills in the batch and cuts the admission budget, an expert stall keeps
//! every member for the next wave, and a refused tier retries in a narrower
//! group. Once a group has taken rows into carried state it is not retried from
//! its first layer, since a repeat would run those rows in a second time. Before
//! each forward a slot's logical offset is brought into line with the blocks it
//! physically holds, because attention reads every position through those blocks.

use super::*;

impl Scheduler {
    /// Fail every member of the paused creep and drop it — for a sweep that can
    /// neither continue nor be replayed.
    pub(in super::super) fn fail_paused_creep(&mut self, why: &str) {
        if let Some(p) = self.paused_creep.take() {
            self.fail_members(p.members, why);
        }
    }

    /// Fail every member of the creep in flight and end it — for a sweep that
    /// can neither continue nor be replayed.
    pub(in super::super) fn fail_wave_members(&mut self, why: &str) {
        let members = std::mem::take(&mut self.wave_prefill_members);
        self.fail_members(members, why);
        self.reset_wave_prefill();
    }

    /// Mark each of `members`' prefill or section ingest failed with `why`.
    pub(super) fn fail_members(&mut self, members: Vec<WaveMember>, why: &str) {
        for m in members {
            match m {
                WaveMember::Prefill { seq_id, .. } => {
                    if let Some(p) = self
                        .active_prefills
                        .iter_mut()
                        .find(|p| p.work.sequence_id.0 == seq_id)
                    {
                        p.error = Some(ConversationError::Other(why.to_string()));
                    }
                }
                WaveMember::Section { seq_id, .. } => {
                    if let Some(s) = self
                        .active_section_ingests
                        .iter_mut()
                        .find(|s| s.sequence_id.0 == seq_id)
                    {
                        s.error = Some(ConversationError::Other(why.to_string()));
                    }
                }
            }
        }
    }

    /// Resume the held wave group: rebuild each member's `(seq_id, input tensor)`
    /// from its live backing — both kinds feed their stable
    /// `[offset, offset+advance)` chunk — dropping members that errored/completed.
    /// Returns the kept members (aligned with `seq_ids`/`inputs`) plus the
    /// `active_prefills` positions of the prefill members (for OOM/error routing).
    #[allow(clippy::type_complexity)]
    pub(super) fn build_wave_group_inputs(
        &mut self,
    ) -> (Vec<WaveMember>, Vec<usize>, Vec<Tensor>, Vec<usize>) {
        let members = self.wave_prefill_members.clone();
        let mut kept: Vec<WaveMember> = Vec::with_capacity(members.len());
        let mut seq_ids: Vec<usize> = Vec::with_capacity(members.len());
        let mut inputs: Vec<Tensor> = Vec::with_capacity(members.len());
        let mut prefill_gidxs: Vec<usize> = Vec::new();
        for m in members {
            match m {
                WaveMember::Prefill { seq_id, advance } => {
                    let Some(i) = self
                        .active_prefills
                        .iter()
                        .position(|p| p.work.sequence_id.0 == seq_id)
                    else {
                        continue;
                    };
                    if self.active_prefills[i].error.is_some()
                        || self.active_prefills[i].finished.is_some()
                    {
                        continue;
                    }
                    if self.active_prefills[i].prefill_start.is_none() {
                        self.active_prefills[i].prefill_start = Some(Instant::now());
                    }
                    let off = self.active_prefills[i].offset;
                    let end = (off + advance).min(self.active_prefills[i].work.tokens.len());
                    let work = &self.active_prefills[i].work;
                    let toks = engine_ids(&work.tokens[off..end], off, &work.image_spans);
                    // On the host: the forward lays every row's id on its own
                    // span in one upload (`wave_token_ids`), an image's pads
                    // named by its rows' virtual ids.
                    match Tensor::new(toks.as_slice(), &Device::Cpu).and_then(|t| t.unsqueeze(0)) {
                        Ok(t) => {
                            kept.push(m);
                            seq_ids.push(seq_id);
                            inputs.push(t);
                            prefill_gidxs.push(i);
                        }
                        Err(e) => self.active_prefills[i].error = Some(ConversationError::Model(e)),
                    }
                }
                WaveMember::Section { seq_id, advance } => {
                    let Some(i) = self
                        .active_section_ingests
                        .iter()
                        .position(|s| s.sequence_id.0 == seq_id)
                    else {
                        continue;
                    };
                    if self.active_section_ingests[i].error.is_some() {
                        continue;
                    }
                    let off = self.active_section_ingests[i].offset;
                    let end = (off + advance).min(self.active_section_ingests[i].tokens.len());
                    let toks: Vec<u32> = self.active_section_ingests[i].tokens[off..end].to_vec();
                    match Tensor::new(toks.as_slice(), &Device::Cpu).and_then(|t| t.unsqueeze(0)) {
                        Ok(t) => {
                            kept.push(m);
                            seq_ids.push(seq_id);
                            inputs.push(t);
                        }
                        Err(e) => {
                            self.active_section_ingests[i].error = Some(ConversationError::Model(e))
                        }
                    }
                }
            }
        }
        (kept, seq_ids, inputs, prefill_gidxs)
    }

    /// Finish a wave group that reached the final layer: `members`/`member_logits`
    /// are aligned in caller order. Prefill members commit their chunk and emit a
    /// progress event; the FINAL chunk also emits the staged events and samples
    /// the turn's first token from its row for promotion to decode, while an
    /// earlier one leaves the prefill active for the next group. Section members
    /// advance their chunk + record slot tokens (sealed later by
    /// `finalize_done_section_ingests`). Clears the group.
    ///
    /// `member_logits` must be read before the forward that wrote them is
    /// dropped — the caller holds it across this call.
    pub(super) fn complete_wave_group(&mut self, members: &[WaveMember], member_logits: &[Tensor]) {
        // Prefills whose last chunk landed in this group, with their rows.
        let mut finishing: Vec<(usize, Tensor)> = Vec::new();
        for (k, m) in members.iter().enumerate() {
            match *m {
                WaveMember::Prefill {
                    seq_id: sid,
                    advance,
                } => {
                    let Some(i) = self
                        .active_prefills
                        .iter()
                        .position(|p| p.work.sequence_id.0 == sid)
                    else {
                        continue;
                    };
                    let total = self.active_prefills[i].work.tokens.len();
                    let seq_id = self.active_prefills[i].work.sequence_id;
                    let off = self.active_prefills[i].offset;
                    let end = (off + advance).min(total);
                    if let Err(e) = self.session.advance_sequence(seq_id.0, end - off) {
                        self.active_prefills[i].error = Some(ConversationError::Model(e));
                        continue;
                    }
                    let chunk_tokens: Vec<u32> =
                        self.active_prefills[i].work.tokens[off..end].to_vec();
                    super::super::Scheduler::record_slot_tokens(
                        &mut self.slot_tokens,
                        seq_id,
                        &chunk_tokens,
                    );
                    self.active_prefills[i].offset = end;
                    // A chunk `cut_at_page_breaks` ended on a break closes its
                    // page now, between this sweep's forward and the next.
                    if chunk_closes_page(&chunk_tokens, end < total, self.prefill_breaks()) {
                        self.close_page_at_break(seq_id, &chunk_tokens);
                    }
                    if end < total {
                        // More to prefill: report progress and ride the next group.
                        let _ = self.active_prefills[i].work.event_tx.send(
                            TurnEvent::PrefillProgress {
                                tokens_done: end,
                                tokens_total: total,
                            },
                        );
                        // At a boundary the opening waits while the slot is
                        // chosen again from what its rows hit (§7.4): the rest
                        // prefills against that slot, on the view the
                        // reprojection carves.
                        if self.active_prefills[i].at_boundary() {
                            self.active_prefills[i].awaiting = true;
                            Self::queue_reprojection(&mut self.pending_reprojections, seq_id);
                        }
                        continue;
                    }
                    // Staged calibration prefill: emit every segment's pinned
                    // projection here, once the last chunk has landed.
                    if let Some(comp) = self.active_prefills[i].work.staged_composition.clone() {
                        let gen_start = self.active_prefills[i].work.assistant_content_start;
                        let offs = self.active_prefills[i].work.projection_offsets.clone();
                        for seg in 0..offs.len() {
                            let prev_off = if seg == 0 { gen_start } else { offs[seg - 1] };
                            let mut ev = comp.clone();
                            ev.start_token = prev_off.saturating_sub(gen_start);
                            let _ = self.active_prefills[i]
                                .work
                                .event_tx
                                .send(TurnEvent::Projection(ev));
                        }
                        self.active_prefills[i].next_projection = offs.len();
                    }
                    // A captured span ran in an earlier chunk of this prefill, so
                    // by its last every recurrent layer has run it.
                    let captures = self.active_prefills[i]
                        .work
                        .recurrent_spans
                        .iter()
                        .any(|s| matches!(s.kind, ArmKind::Capture));
                    if captures {
                        let event = match self.model.take_recurrent_capture(sid) {
                            Ok(Some(rows)) => TurnEvent::RecurrentCapture(Arc::new(rows)),
                            Ok(None) => TurnEvent::Error(ConversationError::Other(
                                "the turn's captured span never ran every recurrent layer".into(),
                            )),
                            Err(e) => TurnEvent::Error(ConversationError::Model(e)),
                        };
                        let _ = self.active_prefills[i].work.event_tx.send(event);
                    }
                    let _ =
                        self.active_prefills[i]
                            .work
                            .event_tx
                            .send(TurnEvent::PrefillProgress {
                                tokens_done: total,
                                tokens_total: total,
                            });
                    // The turn's first token comes from this row, and the row
                    // is only valid while this wave holds its span: the next
                    // forward reuses it, and a row sampled after that reads a
                    // LATER step's logits for some other slot (the stored CJK
                    // drift, 0.007%→0.135% at 42553ca3). So it is sampled below,
                    // before this returns, rather than copied off the span to be
                    // sampled at promotion.
                    if let Some(l) = member_logits.get(k) {
                        finishing.push((i, l.clone()));
                    }
                }
                WaveMember::Section {
                    seq_id: sid,
                    advance,
                } => {
                    let Some(i) = self
                        .active_section_ingests
                        .iter()
                        .position(|s| s.sequence_id.0 == sid)
                    else {
                        continue;
                    };
                    if let Err(e) = self.session.advance_sequence(sid, advance) {
                        self.active_section_ingests[i].error = Some(ConversationError::Model(e));
                        continue;
                    }
                    let seq_id = self.active_section_ingests[i].sequence_id;
                    let off = self.active_section_ingests[i].offset;
                    let end = (off + advance).min(self.active_section_ingests[i].tokens.len());
                    let chunk_tokens = self.active_section_ingests[i].tokens[off..end].to_vec();
                    super::super::Scheduler::record_slot_tokens(
                        &mut self.slot_tokens,
                        seq_id,
                        &chunk_tokens,
                    );
                    self.active_section_ingests[i].offset = end;
                }
            }
        }
        self.sample_first_rows(finishing);
        self.reset_wave_prefill();
    }

    /// Route the failure of a forward that carried a section batch outside any
    /// wave group (`build_section_batch`).
    ///
    /// **Its sections are narrowed or failed, never left to repeat.** The batch
    /// is rebuilt from the same sections every loop, so an error left on no
    /// member rebuilt the same batch and failed it the same way, forever: the
    /// stuck Qwen3.8-Flash-Next calibration was this, about 1,500 refused
    /// placements a second. A refused tier narrows the next batch while there
    /// is anything left to narrow ([`Self::note_refused_width`]). A device OOM,
    /// or an expert stall — a cold expert whose pack read ran past the workers'
    /// wait ([`is_expert_stall`]) — leaves the sections as they are, to be
    /// retried on the next wave exactly as [`Self::fail_wave_group`] retries its
    /// section members: the two paths carry the same sections, so they answer
    /// the same error the same way, and a one-off costs a retry rather than
    /// every section in the batch. A stall failed them outright, and one slow
    /// pack read under disk contention failed zend's load with them. Any other
    /// error, or a refusal of the least chunk, fails the batch's sections so
    /// they answer their callers and give their slots back.
    pub(in super::super) fn fail_section_batch(
        &mut self,
        group_idxs: &[usize],
        advances: &[usize],
        err: &candle::Error,
    ) {
        if group_idxs.is_empty() || is_device_oom(err) || is_expert_stall(err) {
            return;
        }
        let rows: usize = advances.iter().sum();
        if tier_refusal_shortfall(err).is_some_and(|short| self.note_refused_width(rows, short)) {
            return;
        }
        let msg = format!("section batch forward failed: {err}");
        for &i in group_idxs {
            self.active_section_ingests[i].error = Some(ConversationError::Channel(msg.clone()));
        }
    }

    /// Route a wave-group forward failure. On device-OOM, requeue the prefill
    /// members' scope prefills ([`Self::handle_prefill_oom`]) — section members and
    /// dialogue turns just retry next wave once the group is dropped. On an
    /// expert stall ([`is_expert_stall`]) every member keeps its place and
    /// retries next wave: nothing was short, so there is nothing to requeue or
    /// narrow. On a refused
    /// tier, every member retries in a group composed narrower than the
    /// `creep_rows` refused ([`Self::note_refused_width`]) — the same members at
    /// the same width would be refused again. On any other error, or a refusal
    /// of a group already at the least chunk, surface it on each member's backing
    /// entry. Always resets the group — so only for a failure in a sweep's first
    /// window, before any layer has taken the group's rows into its state.
    pub(super) fn fail_wave_group(
        &mut self,
        members: &[WaveMember],
        prefill_gidxs: &[usize],
        creep_rows: usize,
        err: &candle::Error,
    ) {
        if is_device_oom(err) {
            self.handle_prefill_oom(prefill_gidxs, err);
        } else if is_expert_stall(err) || is_span_full(err) {
            // Every member keeps its place and its progress; the same group
            // re-forms on the next wave — after a full span, once turns
            // finishing and the KV side's relief have given ground back.
        } else if let Some(short) = tier_refusal_shortfall(err) {
            // Every member keeps its place and its progress; the group re-forms
            // narrower on the next wave — or, already at the least chunk, at the
            // same width once ground is back. **A refusal at the least width is a
            // full span, not a bad turn**, and is waited out like one: failed, it
            // ended passthrough turns on whiteworld (2026-10-18) with the zone at
            // its floor and every region live, ground that finishing turns and
            // relief hand back moments later.
            self.note_refused_width(creep_rows, short);
        } else {
            let msg = format!("wave group forward failed: {err}");
            for m in members {
                match *m {
                    WaveMember::Prefill { seq_id, .. } => {
                        if let Some(i) = self
                            .active_prefills
                            .iter()
                            .position(|p| p.work.sequence_id.0 == seq_id)
                        {
                            self.active_prefills[i].error =
                                Some(ConversationError::Channel(msg.clone()));
                        }
                    }
                    WaveMember::Section { seq_id, .. } => {
                        if let Some(i) = self
                            .active_section_ingests
                            .iter()
                            .position(|s| s.sequence_id.0 == seq_id)
                        {
                            self.active_section_ingests[i].error =
                                Some(ConversationError::Channel(msg.clone()));
                        }
                    }
                }
            }
        }
        self.reset_wave_prefill();
    }

    /// Consume this wave's deferred gap-fill plans into a co-batchable glue group
    /// `(parent slot ids, glue-token input tensors, per-slot scatter descriptors)`.
    ///
    /// Deferred glue is ingest / compression gap-fill — a pure K/V scatter whose
    /// content prefills through a *separate* unit later (`apply_segments`), so it
    /// has no same-wave, same-slot consumer and can ride the wave as a full-sweep
    /// member alongside decode rather than a separate drain forward. `mem::take`
    /// consumes it once; later decode steps this wave see an empty queue. Returns
    /// `None` when nothing was deferred (or every plan was empty).
    pub(super) fn take_wave_glue(&mut self) -> Option<(Vec<usize>, Vec<Tensor>, Vec<PendingGlue>)> {
        if self.deferred_glue_fires.is_empty() {
            return None;
        }
        let plans = std::mem::take(&mut self.deferred_glue_fires);
        let mut ids: Vec<usize> = Vec::with_capacity(plans.len());
        let mut inputs: Vec<Tensor> = Vec::with_capacity(plans.len());
        let mut pending: Vec<PendingGlue> = Vec::with_capacity(plans.len());
        for p in &plans {
            if p.glue_tokens.is_empty() {
                continue;
            }
            let input = match Tensor::new(p.glue_tokens.as_slice(), &Device::Cpu)
                .and_then(|t| t.unsqueeze(0))
            {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!("wave glue input build failed: {e}");
                    // Dropped unfilled: the slot's assembly recorded these glue
                    // pieces as placed, and no later rebuild may keep them.
                    if let Some(state) = self.slot_projection_state.get_mut(&p.parent_id) {
                        state.placed_pieces.clear();
                    }
                    continue;
                }
            };
            ids.push(p.parent_id.0);
            inputs.push(input);
            pending.push(PendingGlue {
                write_slice: p.glue_write_slice.clone(),
                write_in_blk: p.glue_write_in_blk.clone(),
                fwd_ahead: p.fwd_ahead.clone(),
            });
        }
        if ids.is_empty() {
            None
        } else {
            Some((ids, inputs, pending))
        }
    }

    /// Reconcile each slot's logical offset with its physical backing length —
    /// the wave-boundary invariant every member must satisfy: the varlen
    /// metadata (`cu_seqlens` / `kv_lens`, built from `session.offset`) and the
    /// slot headers (built from the live block table) describe the SAME slot,
    /// and the attention kernels resolve every `[0, kv_len)` position through
    /// the table. Any divergence sends the kernel past the slot's staged state
    /// into neighboring uploads (garbage slice indices → wild record pointers
    /// → CUDA_ERROR_ILLEGAL_ADDRESS, or silent cross-slot attention reads).
    ///
    /// Two producers, one per direction:
    /// - backing > offset: the co-batched glue scatter reserved gap chunk
    ///   space the unified wave didn't reflect in the slot's logical offset.
    ///   Left as-is, the NEXT prefill computes its write region from the
    ///   stale, shorter offset and clobbers the occupied `[offset, backing)`
    ///   span. Advance the offset up to the backing. (Previously a hard
    ///   assert that aborted the whole wave — the crash root at 42553ca3.)
    /// - offset > backing: a projection injected FEWER tokens than the
    ///   planner counted (`select-promote` drops sections it cannot lift to
    ///   hot under VRAM pressure), leaving the offset counting KV that never
    ///   landed. Clamp the offset down to the backing — positions are
    ///   slot-relative (slice ropes), so the clamped value is also the
    ///   correct RoPE base for the new tokens.
    pub(super) fn reconcile_wave_offsets(&mut self, ids: &[usize]) -> candle::Result<()> {
        for &id in ids {
            let session_off = self.session.sequence_offset(id).unwrap_or(0);
            // Physical ground truth: the token count the live block table
            // actually covers (the same walk the slot-header build performs).
            // NOT `current_seq_len` — that is the write cursor and reads 0 for
            // freshly injected slots whose tables already hold sealed tokens.
            let backing_len = self
                .session
                .sequence_backing_tokens(id)
                .unwrap_or(session_off);
            if backing_len > session_off {
                self.session
                    .advance_sequence(id, backing_len - session_off)
                    .map_err(|e| candle::Error::Msg(format!("reconcile_wave_offsets: {e}")))?;
                tracing::debug!(
                    slot = id,
                    from = session_off,
                    to = backing_len,
                    "slot offset reconciled up to backing length"
                );
            } else if backing_len < session_off {
                self.session
                    .set_sequence_offset(id, backing_len)
                    .map_err(|e| candle::Error::Msg(format!("reconcile_wave_offsets: {e}")))?;
                tracing::warn!(
                    slot = id,
                    offset = session_off,
                    backing = backing_len,
                    "slot offset AHEAD of backing — clamped down (projection dropped \
                     sections it could not lift; kv metadata must describe the \
                     physical backing)"
                );
            }
        }
        Ok(())
    }

    /// Handle a device-OOM from the ragged prefill forward: the batch was too
    /// wide for the card. Cut the admission budget (so subsequent waves admit
    /// less) and surface the error on each in-batch prefill's caller channel.
    ///
    /// The hardest evidence the controller gets — a forward that actually failed
    /// — so it acts immediately here rather than waiting for the setpoint loop.
    ///
    /// `group_idxs` are the `active_prefills` positions that were in this forward;
    /// they're still valid because nothing mutates `active_prefills` between the
    /// forward returning and this call.
    pub(super) fn handle_prefill_oom(&mut self, group_idxs: &[usize], err: &candle::Error) {
        let in_batch: HashSet<usize> = group_idxs.iter().copied().collect();
        self.cut_admit_budget(ThrottleReason::DeviceOom);
        let msg = format!("batched prefill forward failed: {err}");
        for (i, p) in self.active_prefills.iter_mut().enumerate() {
            if in_batch.contains(&i) {
                p.error = Some(ConversationError::Channel(msg.clone()));
            }
        }
    }
}
