//! Forms, pauses, resumes and drops the wave's prefill group: the prompts and
//! ingest chunks that sweep the layers together.
//!
//! A group carries one adapter and one band, and a `High` group carries `High`
//! prompts only, because every row rides every layer and rows of a lower band
//! would hold a person's prompt for the whole sweep. The group is filled in
//! order to a per-forward budget, nearest prompt first, and a background group
//! prefills its long prompts as a gang. A creep mid-sweep is never restarted,
//! since its finished layers already hold its rows in carried state. It is
//! paused at the layer it reached and resumed there, and a member that leaves
//! takes only its own rows.

use super::*;

impl Scheduler {
    /// Clear the in-flight continuous-fair-wave prefill group (residual, cursor,
    /// members) so the next wave forms a fresh one.
    ///
    /// **Only between sweeps.** A creep mid-sweep has run layers `[0, cursor)`
    /// for its members, and those layers' carried state — the QSA index, the
    /// DeltaNet state, the PLE history — has taken its rows. Nothing rolls that
    /// back, so a sweep restarted from layer 0 runs those rows into it a second
    /// time: the index ends up ahead of the K/V it describes (measured on `.5`,
    /// 2026-10-16: three section slots 2,545 tokens ahead after two restarts,
    /// whose next append then needed a page inside the forward and failed the
    /// wave), and the recurrent state has seen the prompt twice. A member that
    /// must leave mid-sweep leaves through [`Self::drop_wave_members`].
    pub(in super::super) fn reset_wave_prefill(&mut self) {
        self.held_fit_claims = None;
        self.sweep_forecast.sweep_abandoned();
        let held = self.held_arena_bytes();
        self.sweep_forecast.held_ended(held);
        self.wave_prefill_residual = None;
        self.wave_prefill_cursor = 0;
        self.wave_prefill_members.clear();
        self.wave_prefill_rows.clear();
    }

    /// Take the members `keep` marks `false` out of the creep in flight,
    /// keeping its sweep: their rows leave the held residual and the survivors
    /// carry on from the layer they reached (`paused_creep::drop_rows`). With
    /// no survivor the group is simply over.
    pub(in super::super) fn drop_wave_members(&mut self, keep: &[bool]) -> candle::Result<()> {
        if !keep.iter().any(|&k| k) {
            self.reset_wave_prefill();
            return Ok(());
        }
        drop_rows(
            &mut self.wave_prefill_residual,
            &mut self.wave_prefill_members,
            &mut self.wave_prefill_rows,
            keep,
        )
    }

    /// Take `seq` out of the paused creep, as [`Self::drop_wave_members`] does
    /// for the one in flight. With no member left it is gone.
    pub(in super::super) fn drop_paused_member(&mut self, seq: usize) -> candle::Result<()> {
        let Some(paused) = self.paused_creep.as_mut() else {
            return Ok(());
        };
        let keep: Vec<bool> = paused.members.iter().map(|m| m.seq_id() != seq).collect();
        if !keep.iter().any(|&k| k) {
            self.paused_creep = None;
            return Ok(());
        }
        drop_rows(
            &mut paused.residual,
            &mut paused.members,
            &mut paused.rows,
            &keep,
        )
    }

    /// Set the creep in flight aside mid-sweep, so a higher band's prompt runs
    /// first ([`PausedCreep`]).
    pub(super) fn pause_creep(&mut self) {
        let band = self.creep_priority(&self.wave_prefill_members);
        tracing::debug!(
            target: "candle_conversation::scheduler::creep",
            band = ?band,
            cursor = self.wave_prefill_cursor,
            members = self.wave_prefill_members.len(),
            "creep paused for a higher band's prompt"
        );
        self.paused_creep = Some(PausedCreep {
            residual: self.wave_prefill_residual.take(),
            cursor: self.wave_prefill_cursor,
            members: std::mem::take(&mut self.wave_prefill_members),
            rows: std::mem::take(&mut self.wave_prefill_rows),
            band,
        });
        self.reset_wave_prefill();
    }

    /// Put the paused creep back in flight when no creep is and nothing above
    /// its band is waiting. Returns whether it did.
    pub(super) fn resume_creep(&mut self) -> bool {
        let Some(paused) = self.paused_creep.as_ref() else {
            return false;
        };
        if self.waiting_above(paused.band, |seq| paused.holds(seq)) {
            return false;
        }
        let Some(p) = self.paused_creep.take() else {
            return false;
        };
        tracing::debug!(
            target: "candle_conversation::scheduler::creep",
            band = ?p.band,
            cursor = p.cursor,
            members = p.members.len(),
            "paused creep resumed"
        );
        self.wave_prefill_residual = p.residual;
        self.wave_prefill_cursor = p.cursor;
        self.wave_prefill_members = p.members;
        self.wave_prefill_rows = p.rows;
        true
    }

    /// Set of section-ingest `seq_id`s currently in flight in the wave group or
    /// the paused one, so the standalone section pass and a fresh group
    /// formation don't double-admit a chunk that is already creeping (its
    /// offset isn't advanced until the head).
    pub(in super::super) fn wave_group_section_seqs(&self) -> std::collections::HashSet<usize> {
        let paused = self.paused_creep.iter().flat_map(|p| p.members.iter());
        self.wave_prefill_members
            .iter()
            .chain(paused)
            .filter_map(|m| match m {
                WaveMember::Section { seq_id, .. } => Some(*seq_id),
                WaveMember::Prefill { .. } => None,
            })
            .collect()
    }

    /// Form a FRESH wave group into `wave_prefill_members`: ready dialogue
    /// prefills in queue order up to [`Self::prefill_pass_budget`], plus — when
    /// `include_sections` and at least one prefill is present — section chunks in
    /// whatever of that budget the prefills left. Section chunks join
    /// only alongside a cohort (so they co-batch a creep that is happening anyway);
    /// with no cohort the caller uses the faster full-sweep section path instead.
    /// Members are ordered prefills-then-sections and this order is then fixed for
    /// the group's life (the held residual depends on a stable input order).
    pub(super) fn form_wave_group(&mut self, include_sections: bool) {
        // ── Dialogue prefills creep in bounded chunks ────────────────────────
        //
        // Each takes at most `max_prefill_pass_tokens` rows, packed under that
        // same per-forward budget (at least one always admitted), exactly like
        // the section chunks below; its offset advances at the head and the rest
        // rides the next group. A prefill used to enter whole: a 15.2k-token
        // Cline turn asked this wave for a 13.2 GiB transient tier — more than
        // all the ground below the weight floor on a 16 GB card — so it could
        // never be placed and failed identically on every retry, after first
        // stripping the expert cache to its floor trying.
        let cap = self.max_prefill_pass_tokens;
        // ── A prompt past the model's RoPE reach fails alone ─────────────────
        //
        // Its headers would be refused, and the refusal would fail every other
        // member of the forward with it. Refused here instead, before it joins.
        let reach = self.session.rope_reach();
        for p in self.active_prefills.iter_mut() {
            if p.error.is_some() || p.finished.is_some() {
                continue;
            }
            let at = self
                .session
                .sequence_offset(p.work.sequence_id.0)
                .unwrap_or(0);
            let end = at + p.work.tokens.len().saturating_sub(p.offset);
            if end > reach {
                p.error = Some(ConversationError::Other(format!(
                    "this turn's prompt reaches position {end}, past the {reach} the model's \
                     RoPE schedule supports"
                )));
            }
        }
        let mut members: Vec<WaveMember> = Vec::new();
        // **The highest band leads the group.** The group's adapter is its
        // first member's, so leading with a lower band could leave a waiting
        // dialogue prompt out of the group re-formed to carry it
        // (`creep_outranked_by_a_waiting_prompt`) — and that restart would then
        // fire again every step, the creep never completing. Stable, so a band
        // keeps its queue order.
        // A paused creep's members are mid-sweep in their own group, so they
        // join no other (`PausedCreep`).
        let paused = self
            .paused_creep
            .as_ref()
            .map(PausedCreep::seq_ids)
            .unwrap_or_default();
        let mut candidates: Vec<&ActivePrefill> = self
            .active_prefills
            .iter()
            .filter(|p| !paused.contains(&p.work.sequence_id.0))
            .collect();
        // **Within a band, the prompt nearest its first token leads**, so the
        // budget filled in order (`gang::gang_shares`) finishes it and
        // it decodes while the next ones prefill.
        candidates.sort_by_key(|p| {
            (
                std::cmp::Reverse(rank(self.decode_priority_or_high(p.work.sequence_id))),
                p.work.tokens.len().saturating_sub(p.offset),
            )
        });
        // **A `High` group carries `High` prompts only.** Every row in a group
        // rides every layer of its sweep, so a person's prompt beside eight
        // characters' and ingest chunks waits for all of their rows — measured
        // on `.5`, 2026-10-17, a 10-member, 8,190-row group creeping at one
        // layer a step held a dialogue's first token ~24 s. Below `High`
        // throughput matters more than latency, and the bands mix.
        //
        // A `High` section counts: it is a dialogue's own (a journal entry its
        // next turn reads), and a group of ingest rows would hold it as long.
        // With no `High` prompt to ride beside, it runs in the standalone
        // section pass while this group stays empty.
        //
        // **An `Urgent` group carries `Urgent` prompts only**, for the same
        // reason one band down: the check an image draw waits on, beside four
        // Makers' table prompts, would wait for all of their rows.
        let lead = candidates
            .first()
            .map(|p| self.decode_priority_or_high(p.work.sequence_id));
        let high_section = include_sections
            && self.active_section_ingests.iter().any(|s| {
                s.band.is_live()
                    && s.error.is_none()
                    && s.offset < s.tokens.len()
                    && !paused.contains(&s.sequence_id.0)
            });
        let floor = match lead.filter(|b| b.is_live()) {
            Some(lead) => Some(lead),
            None => high_section.then_some(DecodePriority::High),
        };
        let high_only = floor.is_some();
        let below_floor = |b: DecodePriority| floor.is_some_and(|f| rank(b) < rank(f));
        if high_only {
            candidates.retain(|p| !below_floor(self.decode_priority_or_high(p.work.sequence_id)));
        }
        // ── A background group prefills its long prompts as a gang ───────────
        //
        // With nothing live leading, the long prompts are prefilled together so
        // they reach their first tokens in the same forward and decode side by
        // side, rather than one at a time and one row wide
        // (`docs/prefill_gangs.md`). The shorts ride ahead of the gang and the
        // longs outside it behind; stable, so each lane keeps its band and
        // nearest-first order.
        let gang = self.prefill_gang_for(!high_only, &paused);
        if !high_only && !gang.is_empty() {
            candidates.sort_by_key(|p| {
                if gang.contains(&p.work.sequence_id.0) {
                    1
                } else if p.work.tokens.len().saturating_sub(p.offset) > cap {
                    2
                } else {
                    0
                }
            });
        }
        for p in candidates {
            // An opening stopped at a boundary waits for its reprojection to
            // hand it back on its new view.
            if p.error.is_some()
                || p.finished.is_some()
                || p.awaiting
                || p.offset >= p.work.tokens.len()
            {
                continue;
            }
            // Every prompt that can advance is a candidate, each offering at
            // most a pass. Who rides and how far is decided below, where the
            // members share the budget — stopping here at the first prompt that
            // does not fit whole would decide it first, and hold every prompt
            // behind that one back from a forward it could have shared. An
            // opening on a filled slot offers no further than its next
            // boundary (§7.4); the shares below only cut an advance shorter.
            members.push(WaveMember::Prefill {
                seq_id: p.work.sequence_id.0,
                advance: p.advance_cap().min(cap),
            });
        }
        self.prefill_gang = gang;
        // ── One adapter per wave ─────────────────────────────────────────────
        //
        // Same rule the decode cohort follows, applied where the prefill group
        // is formed: the projections run once over every row, so a group carries
        // one adapter or none. The first ready prefill sets it and the rest wait
        // for a group of their own — they are still active, so nothing is lost,
        // and successive groups drain each adapter's queue in turn.
        //
        // Sections are filtered by the same key below rather than after the
        // fact: a section chunk is an ordinary row of this forward, and an
        // unadapted ingest riding an adapted group would be prefilled through
        // the wrong projections and its KV written that way permanently.
        let group_adapter = members
            .first()
            .map(|m| self.session.sequence_adapter(m.seq_id()))
            .unwrap_or(None)
            .map(|s| s.to_owned());
        members.retain(|m| self.session.sequence_adapter(m.seq_id()) == group_adapter.as_deref());

        // ── Bounded by what one forward can carry, filled in order ───────────
        //
        // The group's rows are the sum of its members' advances, and the forward
        // prices its transient tier from that sum. Unbounded, a burst of queued
        // turns became one forward: npcd's world ingest on the routed
        // Qwen3.6-35B-A3B asked for a 3.3 GB tier against a 3.0 GB gap between the
        // KV frontier and the weight floor, and every turn in the wave failed with
        // it. So the pass budget is filled in the order above — the nearest
        // prompt takes all it has, the next what is left, a gang in proportion
        // to its remainders — and the rest rides the next group. See
        // `gang::gang_shares`.
        let remaining: Vec<usize> = members
            .iter()
            .map(|m| match m {
                WaveMember::Prefill { advance, .. } | WaveMember::Section { advance, .. } => {
                    *advance
                }
            })
            .collect();
        // A queue that passes the cap by no more than the model's slack rides
        // whole, rather than leaving a straggler group a sweep of its own.
        let budget = self.prefill_group_budget(remaining.iter().sum());
        let seqs: Vec<usize> = members.iter().map(WaveMember::seq_id).collect();
        // A live section has a turn blocked on its seal, and the prompts would
        // otherwise spend the whole budget: it rode only what they left, so
        // beside a steady queue of prompts it never rode at all.
        let live_section_rows: usize = self
            .active_section_ingests
            .iter()
            .filter(|s| {
                include_sections
                    && joins_a_full_forward(s.band, s.tokens.len())
                    && s.error.is_none()
                    && !paused.contains(&s.sequence_id.0)
                    && !below_floor(s.band)
                    && self.session.sequence_adapter(s.sequence_id.0) == group_adapter.as_deref()
            })
            .map(|s| s.tokens.len().saturating_sub(s.offset))
            .sum();
        let shares = self.cohort_shares(
            &seqs,
            &remaining,
            gang::prompt_budget(budget, live_section_rows),
        );
        let members_before = std::mem::take(&mut members);
        for (m, share) in members_before.into_iter().zip(shares) {
            if share == 0 {
                continue;
            }
            members.push(match m {
                WaveMember::Prefill { seq_id, .. } => WaveMember::Prefill {
                    seq_id,
                    advance: share,
                },
                WaveMember::Section { seq_id, .. } => WaveMember::Section {
                    seq_id,
                    advance: share,
                },
            });
        }
        let mut used: usize = members
            .iter()
            .map(|m| match m {
                WaveMember::Prefill { advance, .. } | WaveMember::Section { advance, .. } => {
                    *advance
                }
            })
            .sum();

        if include_sections && !members.is_empty() {
            for i in 0..self.active_section_ingests.len() {
                let s = &self.active_section_ingests[i];
                if s.error.is_some()
                    || s.offset >= s.tokens.len()
                    || paused.contains(&s.sequence_id.0)
                    || below_floor(s.band)
                {
                    continue;
                }
                if self.session.sequence_adapter(s.sequence_id.0) != group_adapter.as_deref() {
                    continue;
                }
                // Sections share the forward, so they draw on the budget the
                // prefills above have already spent and fill only what is left.
                // The cohort already guarantees this group makes progress, so no
                // section is forced in past it; the rest ride a later group or the
                // standalone section pass. A live one takes what fits and rides
                // the rest in the next.
                let wanted = (s.tokens.len() - s.offset).min(budget);
                let advance = if joins_a_full_forward(s.band, s.tokens.len()) {
                    wanted.min(budget.saturating_sub(used))
                } else {
                    wanted
                };
                if advance == 0 || used + advance > budget {
                    break;
                }
                members.push(WaveMember::Section {
                    seq_id: s.sequence_id.0,
                    advance,
                });
                used += advance;
            }
        }
        if members.is_empty() {
            self.wave_prefill_members = members;
            return;
        }
        self.cut_to_short_pieces(&mut members);
        self.fit_cohort_to_ground(&mut members);
        self.cut_at_recurrent_spans(&mut members);
        self.cut_at_page_breaks(&mut members);
        let used: usize = members
            .iter()
            .map(|m| match m {
                WaveMember::Prefill { advance, .. } | WaveMember::Section { advance, .. } => {
                    *advance
                }
            })
            .sum();
        let band = self.creep_priority(&members);
        // The decodes that set the ratio, by band — and how many count as
        // `High` only because their layer did not resolve.
        let mut decodes_by_band = [0usize; 4];
        let mut unresolved = 0usize;
        for sid in self.foreground_decodes() {
            decodes_by_band[rank(self.decode_priority_or_high(sid))] += 1;
            unresolved += usize::from(self.decode_layer_priority(sid).is_none());
        }
        tracing::debug!(
            target: "candle_conversation::scheduler::creep",
            band = ?band,
            ratio = self.airtime_ratio_over(band),
            budget,
            members = members.len(),
            tokens = used,
            urgent_decodes = decodes_by_band[3],
            high_decodes = decodes_by_band[2],
            normal_decodes = decodes_by_band[1],
            low_decodes = decodes_by_band[0],
            unresolved_decodes = unresolved,
            "creep group formed"
        );
        self.sweep_forecast.sweep_formed(std::time::Instant::now());
        let held = self.held_arena_bytes();
        self.sweep_forecast.held_formed(held);
        self.wave_prefill_members = members;
    }

    /// Cut a group of short prompts to one short span each (`short_piece`), so
    /// the step carrying it sweeps every layer at once (`sweeps_whole`) and no
    /// member claims its prefill scratch: a prompt of up to four spans rides
    /// four steps, a piece each, as a stencil's played run does.
    ///
    /// **Only a group whose every member is short.** A group with a longer
    /// member is sliced whatever its short members ride, and cut to a span they
    /// would take several of its sweeps to finish what one carries now. Their
    /// windows then claim the scratch, which the fit prices from the rows each
    /// writes.
    pub(super) fn cut_to_short_pieces(&self, members: &mut [WaveMember]) {
        let span = self.model.short_span_rows();
        let left = |m: &WaveMember| match m {
            WaveMember::Prefill { seq_id, .. } => self
                .active_prefills
                .iter()
                .find(|p| p.work.sequence_id.0 == *seq_id)
                .map(|p| p.work.tokens.len().saturating_sub(p.offset)),
            WaveMember::Section { seq_id, .. } => self
                .active_section_ingests
                .iter()
                .find(|s| s.sequence_id.0 == *seq_id)
                .map(|s| s.tokens.len().saturating_sub(s.offset)),
        };
        let pieces: Option<Vec<usize>> = members
            .iter()
            .map(|m| left(m).and_then(|l| short_piece(span, l)))
            .collect();
        let Some(pieces) = pieces else {
            return;
        };
        for (m, piece) in members.iter_mut().zip(pieces) {
            let (WaveMember::Prefill { advance, .. } | WaveMember::Section { advance, .. }) = m;
            *advance = (*advance).min(piece);
        }
    }

    /// End each prefill's chunk just past the first page break it would cross
    /// (`page_cut`), so the page closes there when the group's sweep commits it
    /// (`complete_wave_group`). A chunk that runs an armed recurrent span is left
    /// whole: the model honours the arm only on exactly that span. Members stay
    /// in the one group, so the cut costs a member one more sweep per break and
    /// no launch of its own.
    pub(super) fn cut_at_page_breaks(&self, members: &mut [WaveMember]) {
        let breaks = self.prefill_breaks();
        if breaks.is_empty() {
            return;
        }
        for m in members.iter_mut() {
            let WaveMember::Prefill { seq_id, advance } = m else {
                continue;
            };
            let Some(p) = self
                .active_prefills
                .iter()
                .find(|p| p.work.sequence_id.0 == *seq_id)
            else {
                continue;
            };
            if span_at(&p.work.recurrent_spans, p.offset).is_some() {
                continue;
            }
            *advance = break_advance(&p.work.tokens[..], p.offset, *advance, breaks);
        }
    }

    /// Cut each prefill's chunk so that every recurrent span of its turn runs as
    /// a chunk of its own, and arm the model for a chunk that is one — the model
    /// honours an arm only on exactly that span
    /// (`docs/deltanet_timeline.md` §14.4). The last step of forming a
    /// group: the members are fixed for its life, so the cut holds across every
    /// window of its sweep.
    pub(super) fn cut_at_recurrent_spans(&mut self, members: &mut [WaveMember]) {
        for m in members.iter_mut() {
            let WaveMember::Prefill { seq_id, advance } = m else {
                continue;
            };
            let Some(i) = self
                .active_prefills
                .iter()
                .position(|p| p.work.sequence_id.0 == *seq_id)
            else {
                continue;
            };
            let p = &self.active_prefills[i];
            *advance = snap_advance(&p.work.recurrent_spans, p.offset, *advance);
            let Some(span) = span_at(&p.work.recurrent_spans, p.offset).cloned() else {
                continue;
            };
            let at = self.session.sequence_offset(*seq_id).unwrap_or(0);
            tracing::info!(
                target: "candle_conversation::scheduler::recurrent_span",
                seq = *seq_id,
                at,
                len = span.tokens.len(),
                offset = p.offset,
                capture = matches!(span.kind, ArmKind::Capture),
                "arming a recurrent span"
            );
            if let Err(e) = self
                .model
                .arm_recurrent_span(*seq_id, at, span.tokens.len(), span.kind)
            {
                self.active_prefills[i].error = Some(ConversationError::Model(e));
            }
        }
    }

    /// **Fit a cohort to the tier its windows will place**, not to the width
    /// cap's estimate of it.
    ///
    /// The cap prices a one-sequence tier with none of the terms that follow
    /// depth — the attention launch's KV-stage carve and the sparse
    /// selection's carve over every candidate block — and a cohort's every
    /// full-attention window pays both. Above the weight floor the slack hid
    /// it; at the floor, on the RTX 3090 (2026-10-18), an 8,192-row cohort the
    /// cap admitted was refused at its first full-attention layer by 16–240 MB,
    /// every five seconds. So the cohort is priced through `wave_tier_bytes` —
    /// the function the tier is placed from — over its members' real rows and
    /// depths, with its KV, against the ground less what the decodes will take
    /// over the sweep (`sweep_forecast`), and its shares shrink until it fits.
    /// Never below one chunk: a cohort that cannot fit one is the partition's
    /// refusal to answer, which narrows the next.
    pub(super) fn fit_cohort_to_ground(&mut self, members: &mut Vec<WaveMember>) {
        /// Shrinks one fit may make; each scales by the shortfall, so a few do.
        const FIT_PASSES: usize = 6;
        let dtype = self.session.activation_dtype();
        let Some(ground) = self.model.prefill_ground_bytes() else {
            return;
        };
        let kv_row = self.model.kv_bytes_per_row(dtype).unwrap_or(0);
        let ground = ground.saturating_sub(self.sweep_forecast.withheld_bytes(kv_row));
        let decode_stage = self.decode_stage();
        let held_next = match self.session.device().location() {
            candle::DeviceLocation::Cuda { gpu_id } => {
                candle_nn::kv_cache::held_ground_next_claim(gpu_id)
            }
            _ => None,
        };
        let advance = |m: &WaveMember| match m {
            WaveMember::Prefill { advance, .. } | WaveMember::Section { advance, .. } => *advance,
        };
        for _ in 0..FIT_PASSES {
            let q_lens: Vec<usize> = members.iter().map(advance).collect();
            let offsets: Vec<usize> = members
                .iter()
                .map(|m| self.session.sequence_offset(m.seq_id()).unwrap_or(0))
                .collect();
            let rows: usize = q_lens.iter().sum();
            // Over each member's live index, as its windows carve it.
            let seqs: Vec<usize> = members.iter().map(|m| m.seq_id()).collect();
            let Some(tier) = self.model.wave_tier_bytes(
                rows,
                members.len(),
                &q_lens,
                &offsets,
                &seqs,
                &decode_stage,
                dtype,
            ) else {
                return;
            };
            let need = (tier as usize).saturating_add(rows.saturating_mul(kv_row));
            // The KV the cohort writes and the residual it holds may fill holes
            // below the frontier; its tier may not. But only **as far as the
            // holes go**: past them its claims raise the frontier into the
            // tier's own ground, a window at a time. So the frontier's ground
            // must hold the tier and whatever of those claims the holes cannot.
            // Measured on the 3090 scratch daemon (2026-10-18): a cohort fitted
            // with its 3.83 GB tier alone against 3,948 MiB, the holes running
            // down 288 → 96 MiB as it swept, the ground then 3,790 MiB at
            // window 7 — two regions short and every member lost.
            // What the cohort claims as it sweeps, priced as admission prices it:
            // each member's K/V in the formats it is written in, block-rounded;
            // the residual its rows hold; and a recurrent store for every member
            // that does not hold one yet, which its first window builds
            // (`ensure_seq_state`). Without the stores the fit charged 592 MiB
            // for a cohort that claimed 976 MiB over its sweep (WhiteWorld,
            // 6b26e4976): its final window, the widest, found the ground 192 MiB
            // short.
            let per_block = self.per_block_kv_bytes();
            let stores = members
                .iter()
                .filter(|m| !self.model.has_recurrent_state(m.seq_id()))
                .count() as u64;
            // And the prefill scratch a member that kept its store still claims
            // at its first window — one more slot per recurrent layer — when
            // its rows run past the model's short span. Within it the rows are
            // written into the recurrent timeline and claim none: a 7-row
            // cohort was charged 144 MiB it never took (WhiteWorld,
            // 2026-11-23).
            let span = self.model.short_span_rows();
            let scratch: u64 = members
                .iter()
                .zip(&q_lens)
                .filter(|(_, &q)| span.is_none_or(|s| q > s))
                .map(|(m, _)| self.model.prefill_scratch_bytes(m.seq_id()) as u64)
                .sum();
            let claims = (q_lens
                .iter()
                .map(|&q| super::super::admission::prefill_cost_bytes(q, per_block))
                .sum::<u64>()
                .saturating_add(stores.saturating_mul(self.recurrent_cost()))
                .saturating_add(scratch) as usize)
                .saturating_add(self.model.creep_residual_bytes(&self.session, rows))
                // And the selection index pages its rows open, member by member.
                .saturating_add(
                    q_lens
                        .iter()
                        .zip(&offsets)
                        .map(|(&q, &at)| self.model.index_growth_bytes(at, q))
                        .sum::<usize>(),
                );
            let frontier = self.frontier_tier_bytes();
            let holes = self.session.kv_region_stats().map_or(0, |s| {
                s.live_watermark.saturating_sub(s.live) * candle_nn::kv_cache::REGION_BYTES
            });
            // And the K/V the running decodes add while the cohort sweeps: decodes
            // step without admission, their claims land wherever the free list
            // gives, and past the holes that is the frontier's ground — the
            // tier's. The elastic check above withheld it; this one did not, and
            // on WhiteWorld (673f67a2a) a cohort sweeping beside 24 decodes for
            // 4.7 minutes found its window nine regions short.
            let growth = self.decode_growth_bytes();
            // And what held ground claims at its next provisioning, which runs
            // between the cohort's windows: its growth lands at the lowest free
            // regions as the cohort's claims do, holes first and then the
            // frontier, so it is one more claim on the same ground; the landing
            // for an arena it vacates is claimed only below that arena, so it
            // spends holes the claims would otherwise fill. Unpriced, an ingest
            // cohort fitted with 195 MiB to spare met held ground grown to 256
            // MiB at the frontier by its last window, and was refused five
            // regions short (WhiteWorld, c8221fe39, 2026-10-19).
            let (held_growth, held_landing) =
                held_next.as_ref().map_or((0, 0), |c| (c.growth, c.landing));
            // Its next pass is the least it will take; over a whole sweep it
            // provisions at every window, so what it has measured to grow by
            // over one sweep is what the cohort must leave it
            // (`SweepForecast::held_growth_bytes`).
            let held_growth = held_growth.max(self.sweep_forecast.held_growth_bytes() as usize);
            // The decodes' static runs claim nothing between its windows: each
            // rides a step a piece at a time, within the model's short span
            // (`stencil_wave`).
            let claims = claims.saturating_add(held_growth);
            let holes = holes.saturating_sub(held_landing);
            let at_frontier = frontier.and_then(|g| {
                fitted_rows(
                    rows,
                    (tier as usize)
                        .saturating_add(claims.saturating_sub(holes))
                        .saturating_add(growth),
                    g,
                )
            });
            // And the width its rows still pay for beside the running decodes
            // (`traded_width`): a cohort is re-traded at every fit, so one that
            // went wide with no decode running narrows once decodes arrive.
            let decodes = self.decode_width();
            let traded = self
                .wave_rate
                .as_ref()
                .map(|r| r.routed_for(decodes))
                .and_then(|routed| self.traded_width(&q_lens, &offsets, &seqs, decodes, routed))
                .map(|(w, _)| w)
                .filter(|&w| w < rows);
            let Some(target) = [fitted_rows(rows, need, ground), at_frontier, traded]
                .into_iter()
                .flatten()
                .min()
            else {
                // Fitted: what it was fitted to, once per cohort, so a window
                // later refused can be read against the price it formed on —
                // with the span as it stood, and the K/V priced both ways the
                // engine prices it (the geometry the fit uses, and the formats
                // admission charges), so the region delta to a refusal says
                // which one the cohort actually claimed at.
                let kv_blocks: u64 = q_lens
                    .iter()
                    .map(|&q| super::super::admission::prefill_cost_bytes(q, per_block))
                    .sum();
                let (live, frontier_regions) = self
                    .session
                    .kv_region_stats()
                    .map_or((0, 0), |s| (s.live, s.live_watermark));
                // Held ground as the refusal's census reads it, so the K/V a
                // sweep claimed is what is left of the live delta once every
                // tenant's and held ground's are taken off.
                let (held_regions, held_top) = match self.session.device().location() {
                    candle::DeviceLocation::Cuda { gpu_id } => {
                        candle_nn::kv_cache::held_ground_stats(gpu_id).map_or((0, 0), |h| {
                            (
                                h.arena_bytes / candle_nn::kv_cache::REGION_BYTES,
                                h.arena_top,
                            )
                        })
                    }
                    _ => (0, 0),
                };
                // Who holds the span at formation, as the refusal line names it
                // then — the difference between the two is who claimed the
                // cohort's ground while it swept.
                let tenants: String = candle_nn::kv_cache::SlotTenant::ALL
                    .iter()
                    .map(|&t| {
                        format!(
                            "{}={}",
                            t.label(),
                            candle_nn::kv_cache::arena_regions(self.session.device(), t)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                tracing::info!(
                    target: "candle_conversation::scheduler::creep",
                    rows,
                    members = members.len(),
                    decodes = self.decode_width(),
                    tier_mib = tier >> 20,
                    kv_geometry_mib = rows.saturating_mul(kv_row) >> 20,
                    stores,
                    scratch_mib = scratch >> 20,
                    growth_mib = growth >> 20,
                    held_growth_mib = held_growth >> 20,
                    held_landing_mib = held_landing >> 20,
                    held_regions,
                    held_top,
                    kv_arenas = self.session.kv_gpu_class_stats().map_or(0, |s| s.total_arenas()),
                    kv_live_mib = self.session.kv_gpu_class_stats().map_or(0, |s| s.classes.iter().map(|c| c.live_bytes).sum::<usize>()) >> 20,
                    kv_blocks_mib = kv_blocks >> 20,
                    live_regions = live,
                    frontier_regions,
                    tenants = %tenants,
                    claims_mib = claims >> 20,
                    holes_mib = holes >> 20,
                    frontier_mib = frontier.unwrap_or(0) >> 20,
                    ground_mib = ground >> 20,
                    "creep cohort fitted to the ground its windows will place on"
                );
                // What the cohort itself will claim as it sweeps, for admission to
                // hold back from the turns it lets in meanwhile
                // (`held_claims_outstanding`): its own claims, and of them the K/V
                // its windows write layer by layer.
                let own = claims.saturating_sub(held_growth) as u64;
                self.held_fit_claims = Some((own, kv_blocks));
                return;
            };
            tracing::debug!(
                target: "candle_conversation::scheduler::creep",
                rows,
                target,
                need_mib = need >> 20,
                ground_mib = ground >> 20,
                "creep cohort narrowed to the tier its windows will place"
            );
            let shares = self.cohort_shares(&seqs, &q_lens, target);
            let before = std::mem::take(members);
            for (m, share) in before.into_iter().zip(shares) {
                if share == 0 {
                    continue;
                }
                members.push(match m {
                    WaveMember::Prefill { seq_id, .. } => WaveMember::Prefill {
                        seq_id,
                        advance: share,
                    },
                    WaveMember::Section { seq_id, .. } => WaveMember::Section {
                        seq_id,
                        advance: share,
                    },
                });
            }
            if members.is_empty() {
                return;
            }
        }
    }
}
