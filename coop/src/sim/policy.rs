//! The placement policies the simulator compares
//! (`docs/cooperative_admission.md` §3). A policy answers one question at
//! each turn's start: which member holds the conversation's lease for this
//! turn. It chooses among the members the engine offers — up, and holding the
//! conversation's state through its latest turn — and sees only what a
//! gateway sees: each member's advertised figures, never its insides.

use super::model::MemberModel;

/// A member as a policy sees it.
#[derive(Debug, Clone)]
pub struct MemberView<'a> {
    pub model: &'a MemberModel,
    /// Conversations whose turn is running here.
    pub held: usize,
    /// Rows decoding here now.
    pub decoding: usize,
    /// Prompt tokens queued here for prefill.
    pub prefill_backlog: usize,
}

/// The turn a decision is for.
#[derive(Debug, Clone, Copy)]
pub struct Asking {
    /// Where its last turn ran; `None` for a conversation not yet placed.
    pub home: Option<usize>,
    pub prefill: usize,
    pub decode: usize,
}

pub trait Policy {
    fn name(&self) -> &'static str;

    /// The member the turn runs on, one of those `eligible` marks.
    fn place(&mut self, members: &[MemberView], eligible: &[bool], asking: &Asking) -> usize;
}

/// The eligible member that minimises `key`; ties go to the lower index.
fn best_by(eligible: &[bool], mut key: impl FnMut(usize) -> f64) -> usize {
    (0..eligible.len())
        .filter(|&m| eligible[m])
        .min_by(|&a, &b| key(a).total_cmp(&key(b)).then(a.cmp(&b)))
        .expect("a turn has an eligible member")
}

/// Today's gateway: a new conversation goes `Best` — the most `power` per
/// conversation held — and stays where it is while its member can take it.
pub struct Today;

impl Policy for Today {
    fn name(&self) -> &'static str {
        "today (Best, sticky)"
    }

    fn place(&mut self, members: &[MemberView], eligible: &[bool], asking: &Asking) -> usize {
        if let Some(h) = asking.home.filter(|h| eligible[*h]) {
            return h;
        }
        best_by(eligible, |m| {
            -(f64::from(members[m].model.power) / (1.0 + members[m].held as f64))
        })
    }
}

/// Decodes concentrated: every turn on the decode member (the first) while it
/// declares room — its rows under `max_rows` and its decode hit rate above
/// `floor_hit` — otherwise on the other member with the most share.
pub struct Concentrate {
    pub floor_hit: f64,
}

impl Policy for Concentrate {
    fn name(&self) -> &'static str {
        "concentrate decodes"
    }

    fn place(&mut self, members: &[MemberView], eligible: &[bool], _asking: &Asking) -> usize {
        let m = &members[0];
        let room =
            m.decoding < m.model.max_rows && m.model.decode_hit(m.held + 1) >= self.floor_hit;
        if eligible[0] && room {
            return 0;
        }
        best_by(eligible, |m| {
            -(f64::from(members[m].model.power) / (1.0 + members[m].held as f64))
        })
    }
}

/// Each member advertises what this turn would cost it now — its prefill
/// backlog and this prompt at its prefill rate, then the reply at the step
/// time one more row gives it — and the turn goes where it finishes soonest.
pub struct EarliestFinish;

impl EarliestFinish {
    /// Seconds `m` advertises for a turn of `asking`.
    pub fn declared(m: &MemberView, asking: &Asking) -> f64 {
        let model = m.model;
        let held = m.held + 1;
        let prefill_tokens = m.prefill_backlog + asking.prefill;
        let forwards = prefill_tokens.div_ceil(model.prefill_width).max(1);
        let filled = prefill_tokens as f64 / (forwards * model.prefill_width) as f64;
        let prefill = forwards as f64 * model.prefill_ms(model.prefill_width, 0.0, held) * filled;
        let decode =
            asking.decode as f64 * model.decode_ms(m.decoding + 1, held, m.prefill_backlog > 0);
        (prefill + decode) / 1000.0
    }
}

impl Policy for EarliestFinish {
    fn name(&self) -> &'static str {
        "earliest advertised finish"
    }

    fn place(&mut self, members: &[MemberView], eligible: &[bool], asking: &Asking) -> usize {
        best_by(eligible, |m| Self::declared(&members[m], asking))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn views<'a>(models: &'a [MemberModel], held: &[usize]) -> Vec<MemberView<'a>> {
        models
            .iter()
            .zip(held)
            .map(|(m, h)| MemberView {
                model: m,
                held: *h,
                decoding: *h,
                prefill_backlog: 0,
            })
            .collect()
    }

    fn new_turn() -> Asking {
        Asking {
            home: None,
            prefill: 2_000,
            decode: 300,
        }
    }

    const BOTH: [bool; 2] = [true, true];

    /// Today's share puts four on BlackWorld for every one on WhiteWorld, and
    /// keeps a conversation where it is while it may stay.
    #[test]
    fn today_shares_by_power_and_sticks() {
        let models = [MemberModel::black_world(), MemberModel::white_world()];
        let mut p = Today;
        assert_eq!(p.place(&views(&models, &[0, 0]), &BOTH, &new_turn()), 0);
        assert_eq!(p.place(&views(&models, &[4, 0]), &BOTH, &new_turn()), 1);
        assert_eq!(p.place(&views(&models, &[3, 0]), &BOTH, &new_turn()), 0);
        let on_ww = Asking {
            home: Some(1),
            ..new_turn()
        };
        assert_eq!(p.place(&views(&models, &[0, 9]), &BOTH, &on_ww), 1);
        assert_eq!(p.place(&views(&models, &[0, 9]), &[true, false], &on_ww), 0);
    }

    /// Concentration fills the decode member until it declares no room, and
    /// never places on a member that is not eligible.
    #[test]
    fn concentration_fills_the_decode_member_first() {
        let models = [MemberModel::black_world(), MemberModel::white_world()];
        let mut p = Concentrate { floor_hit: 0.9 };
        assert_eq!(p.place(&views(&models, &[30, 0]), &BOTH, &new_turn()), 0);
        assert_eq!(p.place(&views(&models, &[64, 0]), &BOTH, &new_turn()), 1);
        let not_replicated = [false, true];
        assert_eq!(
            p.place(&views(&models, &[0, 0]), &not_replicated, &new_turn()),
            1
        );
    }

    /// A turn goes where it is advertised to finish soonest: the idle strong
    /// member first, the other once the strong one's backlog outweighs it.
    #[test]
    fn a_turn_goes_where_it_finishes_soonest() {
        let models = [MemberModel::black_world(), MemberModel::white_world()];
        let mut p = EarliestFinish;
        assert_eq!(p.place(&views(&models, &[0, 0]), &BOTH, &new_turn()), 0);
        let mut swamped = views(&models, &[20, 0]);
        swamped[0].prefill_backlog = 200_000;
        assert_eq!(p.place(&swamped, &BOTH, &new_turn()), 1);
        assert_eq!(p.place(&swamped, &[true, false], &new_turn()), 0);
    }
}
