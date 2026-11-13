//! The turn a reprojection holds while it rebuilds the slot
//! (`docs/hit_driven_projection.md` §7.4).
//!
//! A reprojection takes its turn's state out of the maps the loop reads, swaps
//! the turn onto a rebuilt view, and hands the state back under the new view's
//! id. The turn is either decoding its reply, or an opening stopped at a
//! boundary partway through its prefill — a long user message or tool result
//! cut every `reserve_tokens` so the slot moves as it arrives. Both carry what
//! the reprojection reads (the policy, the belief, how far the turn has come)
//! and both go back where they came from: a decode to the decode loop, an
//! opening to the prefill group, to prefill its next stride on the new view.

use std::time::Instant;

use super::*;

/// The turn a reprojection holds between taking it and handing it back.
pub(super) enum HeldTurn {
    Decode(DecodeState),
    Opening(Box<ActivePrefill>),
}

/// What a reprojection reads of its turn before it takes it.
pub(super) struct TurnReading {
    pub(super) policy: ReprojectionPolicy,
    pub(super) belief: PriorBelief,
    /// The tokens the turn has appended past its opening's start that the
    /// slot's room is counted less: the reply so far, or the opening's
    /// prefilled part.
    pub(super) appended: usize,
    /// No projection has run against the turn's own content yet.
    pub(super) is_turn_boundary: bool,
}

impl HeldTurn {
    /// The channel the turn's caller reads.
    pub(super) fn event_tx(&self) -> &Sender<TurnEvent> {
        match self {
            HeldTurn::Decode(d) => &d.event_tx,
            HeldTurn::Opening(p) => &p.work.event_tx,
        }
    }

    /// The belief the next reprojection seeds from.
    pub(super) fn set_belief(&mut self, belief: PriorBelief) {
        match self {
            HeldTurn::Decode(d) => d.belief = belief,
            HeldTurn::Opening(p) => p.work.belief = belief,
        }
    }

    /// Mark the projection point — where in the reply this projection
    /// governs from, and the seconds since the turn began producing — and
    /// record it as the decode's last projection. An opening has decoded
    /// nothing, so its point is the reply's start.
    pub(super) fn projection_point(&mut self, now: Instant) -> (u32, f64) {
        match self {
            HeldTurn::Decode(d) => {
                let at = d.generated_tokens.len() as u32;
                d.last_projection_end = at;
                (at, now.duration_since(d.decode_start).as_secs_f64())
            }
            HeldTurn::Opening(p) => {
                let since = p.prefill_start.unwrap_or(p.work.submitted_at);
                (0, now.duration_since(since).as_secs_f64())
            }
        }
    }

    /// The turn's tokens the slot holds, in order: the opening as far as it
    /// has been forwarded, then what was decoded.
    pub(super) fn slot_tokens(&self) -> (&[u32], &[u32]) {
        match self {
            HeldTurn::Decode(d) => (&d.prefill_tokens, &d.generated_tokens),
            HeldTurn::Opening(p) => (&p.work.tokens[..p.offset], &[]),
        }
    }

    /// Whether the turn is a case of a turn group.
    pub(super) fn is_group_case(&self) -> bool {
        let action = match self {
            HeldTurn::Decode(d) => &d.seal_action,
            HeldTurn::Opening(p) => &p.work.seal_action,
        };
        matches!(action, SealAction::GroupCase(_))
    }

    /// End the turn on a fault: its caller gets `error`. A decode is then
    /// discarded unsealed, as every faulted turn is; an opening never
    /// produced a reply to seal.
    pub(super) fn fail(&mut self, error: ConversationError) {
        match self {
            HeldTurn::Decode(d) => d.fail(error),
            HeldTurn::Opening(p) => {
                let _ = p.work.event_tx.send(TurnEvent::Error(error));
            }
        }
    }
}

impl Scheduler {
    /// What a reprojection of `view` reads of its turn: a decode that has not
    /// finished, or an opening waiting at a boundary. `None` when neither is
    /// there, or the turn has no reprojection policy.
    pub(super) fn turn_reading(&self, view: SequenceId) -> Option<TurnReading> {
        if let Some(d) = self.active_decodes.get(&view).filter(|d| !d.finished) {
            return Some(TurnReading {
                policy: d.reprojection.clone()?,
                belief: d.belief.clone(),
                appended: d.generated_tokens.len(),
                is_turn_boundary: d.last_projection_end == 0,
            });
        }
        let p = self.awaiting_opening(view)?;
        Some(TurnReading {
            policy: p.work.reprojection.clone()?,
            belief: p.work.belief.clone(),
            appended: p.offset,
            is_turn_boundary: true,
        })
    }

    /// The opening of `view` stopped at a boundary, if there is one.
    pub(super) fn awaiting_opening(&self, view: SequenceId) -> Option<&ActivePrefill> {
        self.active_prefills
            .iter()
            .find(|p| p.work.sequence_id == view && p.awaiting && p.error.is_none())
    }

    /// Take `view`'s turn out of the loop's maps for the reprojection.
    pub(super) fn take_turn(&mut self, view: SequenceId) -> Option<HeldTurn> {
        if let Some(d) = self.active_decodes.remove(&view) {
            return Some(HeldTurn::Decode(d));
        }
        let i = self
            .active_prefills
            .iter()
            .position(|p| p.work.sequence_id == view && p.awaiting)?;
        Some(HeldTurn::Opening(Box::new(self.active_prefills.remove(i))))
    }

    /// Hand `turn` back under `view`, the view the reprojection carved: a
    /// decode to the decode loop, an opening to the prefill group to prefill
    /// its next stride against the rebuilt slot.
    pub(super) fn rebind_turn(&mut self, old: SequenceId, view: SequenceId, turn: HeldTurn) {
        match turn {
            HeldTurn::Decode(d) => {
                self.active_decodes.insert(view, d);
            }
            HeldTurn::Opening(mut p) => {
                p.work.sequence_id = view;
                p.awaiting = false;
                for seq in self.prefill_gang.iter_mut() {
                    if *seq == old.0 {
                        *seq = view.0;
                    }
                }
                self.active_prefills.push(*p);
            }
        }
    }

    /// Fail `view`'s turn where it stands, before a reprojection took it.
    pub(super) fn fail_turn_in_place(&mut self, view: SequenceId, error: ConversationError) {
        if let Some(d) = self.active_decodes.get_mut(&view) {
            d.fail(error);
            return;
        }
        if let Some(p) = self
            .active_prefills
            .iter_mut()
            .find(|p| p.work.sequence_id == view && p.awaiting)
        {
            p.error = Some(error);
            p.awaiting = false;
        }
    }

    /// Let `view`'s turn go on as it stood after a reprojection that did not
    /// take it — refused ground, or a selection the slot already holds: a
    /// decode decodes on (nothing of it moved), an opening prefills its next
    /// stride on the slot it has.
    pub(super) fn resume_unreprojected(&mut self, view: SequenceId) {
        if let Some(p) = self
            .active_prefills
            .iter_mut()
            .find(|p| p.work.sequence_id == view && p.awaiting)
        {
            p.awaiting = false;
        }
    }
}
