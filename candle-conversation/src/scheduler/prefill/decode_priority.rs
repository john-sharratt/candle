//! Sets how much airtime a prompt's prefill takes beside the decodes around it,
//! so a live conversation keeps its token rate while another one prefills.
//!
//! A decode and a prefill share one device by a ratio of decode airtime to
//! prefill airtime. Work yields to a decode of a higher band at that decode's
//! ratio, and a `High` prompt yields to a `High` decode at a shared ratio, so
//! two live conversations each keep a share instead of one stalling the other.
//! Work owes decode airtime only for decodes its band does not outrank, because
//! background work that rode a person's step did not run that decode up. The
//! ratio is re-read every wave, so a prefill returns to full speed the wave the
//! last such decode finishes.

use super::*;

/// The decode-to-prefill airtime ratio `R` that work of priority `work` yields
/// to, given the priorities of the decodes running beside it: the highest ratio
/// among the decodes that **outrank** it — and, for `High` work, among the
/// `High` decodes too — or 1 when there are none.
///
/// **A `High` decode protects itself from `High` work — at
/// [`HIGH_BESIDE_HIGH_RATIO`], not its own ratio.** Two live conversations are
/// equals: one decoding, the other waiting on its prompt — a tool result, the
/// next question — to reach its first token. A slice of creep can no longer
/// stall the decode (`creep_slice` bounds it to a decode step's worth), so the
/// decode keeps two-thirds of the device and the waiting conversation's prompt
/// a third. It was paced in tokens instead — nine prompt tokens per decode
/// token landed — which beside one decoding dialogue moved a prompt at ~180
/// tokens a second: a 26K-token file read waited 20–27 s for its first token
/// (`.5`, 2026-10-17), and would have waited minutes. The ratio is re-read every
/// wave, so the creep returns to full speed (`R = 1`) the wave the last such
/// decode finishes.
///
/// Below `High`, latency matters less than throughput, so a `Normal` or `Low`
/// creep is never held back by its own band — only by a decode that outranks it.
///
/// The band is all that counts ([`rank`]), whichever tenant the work belongs to.
pub(super) fn throttle_ratio(
    work: DecodePriority,
    decodes: impl IntoIterator<Item = DecodePriority>,
) -> u32 {
    decodes
        .into_iter()
        .filter_map(|d| {
            if rank(d) > rank(work) {
                Some(d.ratio())
            } else if d == work && work.is_live() {
                Some(HIGH_BESIDE_HIGH_RATIO)
            } else {
                None
            }
        })
        .max()
        .unwrap_or(1)
        .max(1)
}

/// Whether a step carrying `work` owes the decodes beside it airtime: not when
/// it outranks every one of them.
///
/// The debt is decode's protection from the work that ran it up, and a decode
/// that band outranks is not protected from it. Those rows rode the step anyway.
/// Owed regardless, a dialogue's own tool results repaid background decodes step
/// for step: on BlackWorld (2026-10-31) `owed` steps took 34 s of a minute beside
/// `Low` decodes, and a 2,328-token `Urgent` prompt took 7.8 s, ~300 tokens a
/// second on a card that publishes 1,500–1,800.
pub(super) fn owes_decode(
    work: DecodePriority,
    decodes: impl IntoIterator<Item = DecodePriority>,
) -> bool {
    decodes.into_iter().any(|d| rank(d) >= rank(work))
}

/// How long a `Normal` prompt waits for admission before the gateway is told
/// it stands in a placed turn's way (`LearnedRates::high_backlog_tokens`).
pub(super) const STALLED_PROMPT: Duration = Duration::from_secs(30);

/// A compaction sweep this long is logged by holder (`compaction sweep split`).
pub(super) const SLOW_SWEEP: Duration = Duration::from_millis(50);

/// `R` for a `High` creep beside a `High` decode — see [`throttle_ratio`].
pub(super) const HIGH_BESIDE_HIGH_RATIO: u32 = 2;

/// Consecutive failures of one mid-sweep creep window that are retried in place
/// before its members fail. A refusal is usually ground the next step finds —
/// an expert concession, a tier released — and a window that keeps failing is
/// not going to be placed by waiting.
pub(super) const MID_SWEEP_RETRIES: u32 = 4;

impl Scheduler {
    /// Every `High` prompt outstanding — prefilling or still queued — as its
    /// remaining rows and how long it has waited since it was submitted, for
    /// the split's per-prompt deadline (`turn_airtime`).
    pub(in super::super) fn high_prompt_waits(&self) -> Vec<(usize, std::time::Duration)> {
        let now = Instant::now();
        let high = |id: SequenceId| self.decode_priority_or_high(id).is_live();
        self.active_prefills
            .iter()
            .filter(|p| p.error.is_none() && p.finished.is_none() && high(p.work.sequence_id))
            .map(|p| {
                (
                    p.work.tokens.len().saturating_sub(p.offset),
                    now.saturating_duration_since(p.work.submitted_at),
                )
            })
            .chain(
                self.prefill_queue
                    .iter()
                    .filter(|w| high(w.sequence_id))
                    .map(|w| {
                        (
                            w.tokens.len(),
                            now.saturating_duration_since(w.submitted_at),
                        )
                    }),
            )
            .collect()
    }

    /// `R` for co-batched work of priority `work`: the decode-to-prefill airtime
    /// ratio of the active foreground decodes that protect themselves from it
    /// ([`throttle_ratio`], `docs/continuous_fair_waves.md` §3). The creep advances `ceil(N / R)`
    /// layers a wave, and the decode quantum owes decode that share of a
    /// co-batched step.
    ///
    /// Each decode counts at its target layer's `decode_priority`, `High`
    /// where that does not resolve.
    pub(in super::super) fn airtime_ratio_over(&self, work: DecodePriority) -> u32 {
        throttle_ratio(
            work,
            self.foreground_decodes()
                .map(|sid| self.decode_priority_or_high(sid)),
        )
    }

    /// Whether a prompt or section waiting outside the creep in flight outranks
    /// it.
    ///
    /// Every creep step is bounded in time — by the decode cadence, or by
    /// [`DECODE_LESS_SLICE`] with nothing decoding — so a group takes many
    /// steps to reach its sweep boundary whether it is throttled or not, and
    /// work of a higher band waiting for that boundary waits the whole sweep.
    /// Measured on `.5` (2026-10-24): a dialogue's journal entry, 38 rows,
    /// waited out a 6,213-row ingest sweep and then rode the next one — 41 s to
    /// seal beside four `Low` decodes, where the creep was never throttled.
    pub(in super::super) fn creep_outranked_by_a_waiting_prompt(&self) -> bool {
        let band = self.creep_priority(&self.wave_prefill_members);
        self.waiting_above(band, |seq| {
            self.wave_prefill_members.iter().any(|m| m.seq_id() == seq)
                || self.paused_creep.as_ref().is_some_and(|c| c.holds(seq))
        })
    }

    /// Whether a queued prompt — carved or still a submission — belongs to a
    /// band above `band`.
    pub(in super::super) fn a_queued_prompt_outranks(&self, band: DecodePriority) -> bool {
        self.prefill_queue
            .iter()
            .map(|w| w.sequence_id)
            .chain(self.pending_submits.iter().map(|p| p.parent()))
            .any(|id| rank(self.decode_priority_or_high(id)) > rank(band))
    }

    /// Whether a queued prompt — carved or still a submission — belongs to a
    /// band above every prefill and section in flight: one the work running
    /// would be set aside for once it is admitted.
    pub(in super::super) fn a_queued_prompt_outranks_the_work_in_flight(&self) -> bool {
        let queued = self
            .prefill_queue
            .iter()
            .map(|w| rank(self.decode_priority_or_high(w.sequence_id)))
            .chain(
                self.pending_submits
                    .iter()
                    .map(|p| rank(self.decode_priority_or_high(p.parent()))),
            )
            .max();
        let Some(queued) = queued else {
            return false;
        };
        let running = self
            .active_prefills
            .iter()
            .filter(|p| p.error.is_none() && p.finished.is_none())
            .map(|p| rank(self.decode_priority_or_high(p.work.sequence_id)))
            .chain(
                self.active_section_ingests
                    .iter()
                    .filter(|s| s.error.is_none())
                    .map(|s| rank(s.band)),
            )
            .max();
        running.is_none_or(|r| queued > r)
    }

    /// Whether a prefill or section that can advance, and that `held` does not
    /// claim, belongs to a band above `band`.
    pub(in super::super) fn waiting_above(
        &self,
        band: DecodePriority,
        held: impl Fn(usize) -> bool,
    ) -> bool {
        let above = |p: DecodePriority| rank(p) > rank(band);
        let prefills = self
            .active_prefills
            .iter()
            .filter(|p| p.error.is_none() && p.finished.is_none() && p.offset < p.work.tokens.len())
            .filter(|p| !held(p.work.sequence_id.0))
            .any(|p| above(self.decode_priority_or_high(p.work.sequence_id)));
        let sections = self
            .active_section_ingests
            .iter()
            .filter(|s| s.error.is_none() && s.offset < s.tokens.len())
            .filter(|s| !held(s.sequence_id.0))
            .any(|s| above(s.band));
        prefills || sections
    }

    /// The band a section ingest in flight runs at — see
    /// [`SchedulerRequest::IngestSection`]'s `band`. `Low` once it has left.
    pub(super) fn section_band(&self, seq_id: usize) -> DecodePriority {
        self.active_section_ingests
            .iter()
            .find(|s| s.sequence_id.0 == seq_id)
            .map_or(DecodePriority::Low, |s| s.band)
    }

    /// The band a creep group runs at: its highest-ranked member's. A dialogue
    /// prefill runs at its slot's own band, a section chunk at its own.
    pub(super) fn creep_priority(&self, members: &[WaveMember]) -> DecodePriority {
        members
            .iter()
            .map(|m| match m {
                WaveMember::Prefill { seq_id, .. } => {
                    self.decode_priority_or_high(SequenceId(*seq_id))
                }
                WaveMember::Section { seq_id, .. } => self.section_band(*seq_id),
            })
            .max_by_key(|&p| rank(p))
            .unwrap_or(DecodePriority::Low)
    }

    /// The slot that owns `sid`'s projection state: a decode runs on a VIEW
    /// sequence, but the projection target (which carries the layer's
    /// decode_priority) and the conversation are pinned on the view's PARENT
    /// slot. A slot that decodes directly (no view) owns itself.
    pub(super) fn owning_slot(&self, sid: SequenceId) -> SequenceId {
        self.turn_views
            .get(&sid)
            .map(|v| v.parent_id)
            .unwrap_or(sid)
    }

    /// Resolve the `decode_priority` of a decode slot's target layer, or `None`
    /// when the slot's target/timeline isn't resolvable (the caller then defaults
    /// to the protective `High`).
    ///
    /// A summary probe decodes on a scratch slot bound to the conversation it
    /// compresses, so its target is that conversation's layer — the dialogue's,
    /// `High`. It is background to that dialogue, so it runs at `Normal`. An
    /// ephemeral fork runs at the band its asker named.
    pub(in super::super) fn decode_layer_priority(
        &self,
        sid: SequenceId,
    ) -> Option<DecodePriority> {
        if self
            .active_decodes
            .get(&sid)
            .is_some_and(|s| matches!(s.seal_action, SealAction::CompressionPass { .. }))
        {
            return Some(DecodePriority::Normal);
        }
        let slot = self.owning_slot(sid);
        if let Some(band) = self.slot_bands.get(&slot) {
            return Some(*band);
        }
        let target = self.slot_targets.get(&slot)?;
        let builder = self.timeline_projections.get(&target.timeline)?;
        builder
            .schema()
            .layers
            .iter()
            .find(|l| l.id == target.layer)
            .map(|l| l.decode_priority)
    }
}
