//! The placement policies the simulator compares
//! (`docs/cooperative_admission.md` §3). A policy answers two questions:
//! which member holds a conversation's lease when a turn starts, and whether a
//! segment's prefill runs on another member, proxied.
//!
//! What a policy sees is what a gateway sees: each member's declared role and
//! what it holds — never another member's insides.

use super::model::MemberModel;

/// A member as a policy sees it.
#[derive(Debug, Clone)]
pub struct MemberView<'a> {
    pub model: &'a MemberModel,
    /// Conversations whose turn is running here (their lease is here, or
    /// their prefill is proxied here).
    pub held: usize,
    /// Rows decoding here now.
    pub decoding: usize,
    /// Prompt tokens queued here for prefill.
    pub prefill_backlog: usize,
}

/// The conversation a decision is for.
#[derive(Debug, Clone, Copy)]
pub struct Asking {
    /// Where its lease is; `None` for a conversation not yet placed.
    pub home: Option<usize>,
    /// It moved at its last turn, so it stays this one.
    pub stay: bool,
    pub prefill: usize,
    pub decode: usize,
}

pub trait Policy {
    fn name(&self) -> &'static str;

    /// The member whose lease the turn runs under. A policy returning another
    /// member than `asking.home` while `asking.stay` is held to `home`.
    fn place(&mut self, members: &[MemberView], asking: &Asking) -> usize;

    /// The member a segment's prefill runs on, when not its home.
    fn proxy(&mut self, _members: &[MemberView], _asking: &Asking, _home: usize) -> Option<usize> {
        None
    }
}

/// Today's gateway: a new conversation goes `Best` — the most `power` per
/// conversation held, `power / (1 + held)` — and stays where it is.
pub struct Today;

impl Policy for Today {
    fn name(&self) -> &'static str {
        "today (Best, sticky)"
    }

    fn place(&mut self, members: &[MemberView], asking: &Asking) -> usize {
        if let Some(h) = asking.home {
            return h;
        }
        best_share(members)
    }
}

fn best_share(members: &[MemberView]) -> usize {
    (0..members.len())
        .max_by(|&a, &b| {
            let sa = f64::from(members[a].model.power) / (1.0 + members[a].held as f64);
            let sb = f64::from(members[b].model.power) / (1.0 + members[b].held as f64);
            sa.total_cmp(&sb).then(b.cmp(&a))
        })
        .expect("a cluster has a member")
}

/// Decodes concentrated: every conversation on the decode member (the first)
/// while it declares room — its rows under `max_rows` and its zone above
/// `floor_hit` — otherwise on the member with the most share, as today.
pub struct Concentrate {
    pub floor_hit: f64,
}

impl Concentrate {
    fn has_room(&self, m: &MemberView) -> bool {
        m.decoding < m.model.max_rows && m.model.decode_hit(m.held + 1) >= self.floor_hit
    }
}

impl Policy for Concentrate {
    fn name(&self) -> &'static str {
        "concentrate decodes"
    }

    fn place(&mut self, members: &[MemberView], asking: &Asking) -> usize {
        if let Some(h) = asking.home {
            if h == 0 || asking.stay || !self.has_room(&members[0]) {
                return h;
            }
            return 0;
        }
        if self.has_room(&members[0]) {
            0
        } else {
            best_share(members)
        }
    }
}

/// [`Concentrate`], and a segment whose prefill is at least `min_prefill`
/// tokens proxied to the prefill member (the second) while its backlog is
/// under `max_backlog`.
pub struct ConcentrateProxy {
    pub concentrate: Concentrate,
    pub min_prefill: usize,
    pub max_backlog: usize,
}

impl Policy for ConcentrateProxy {
    fn name(&self) -> &'static str {
        "concentrate + proxied prefill"
    }

    fn place(&mut self, members: &[MemberView], asking: &Asking) -> usize {
        self.concentrate.place(members, asking)
    }

    fn proxy(&mut self, members: &[MemberView], asking: &Asking, home: usize) -> Option<usize> {
        let to = 1;
        (home != to
            && members.len() > to
            && asking.prefill >= self.min_prefill
            && members[to].prefill_backlog < self.max_backlog)
            .then_some(to)
    }
}

/// Each member declares what this turn would cost it now — its prefill
/// backlog and this prompt at its prefill rate, then the reply at the step
/// time one more row gives it — and a turn goes where it finishes soonest. A
/// running conversation moves at a turn boundary only when the other member
/// is better by `margin`, and then stays a turn.
pub struct EarliestFinish {
    pub margin: f64,
}

impl EarliestFinish {
    /// Seconds `m` declares for a turn of `asking`.
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
        "earliest declared finish"
    }

    fn place(&mut self, members: &[MemberView], asking: &Asking) -> usize {
        let est: Vec<f64> = members.iter().map(|m| Self::declared(m, asking)).collect();
        let best = (0..members.len())
            .min_by(|&a, &b| est[a].total_cmp(&est[b]).then(a.cmp(&b)))
            .expect("a cluster has a member");
        match asking.home {
            Some(h) if asking.stay || est[best] * (1.0 + self.margin) >= est[h] => h,
            _ => best,
        }
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
            stay: false,
            prefill: 2_000,
            decode: 300,
        }
    }

    /// Today's share puts four on BlackWorld for every one on WhiteWorld, and
    /// never moves a conversation.
    #[test]
    fn today_shares_by_power_and_sticks() {
        let models = [MemberModel::black_world(), MemberModel::white_world()];
        let mut p = Today;
        assert_eq!(p.place(&views(&models, &[0, 0]), &new_turn()), 0);
        assert_eq!(p.place(&views(&models, &[4, 0]), &new_turn()), 1);
        assert_eq!(p.place(&views(&models, &[3, 0]), &new_turn()), 0);
        let moved = Asking {
            home: Some(1),
            ..new_turn()
        };
        assert_eq!(p.place(&views(&models, &[0, 9]), &moved), 1);
    }

    /// Concentration fills the decode member until it declares no room, and
    /// brings a conversation home at a turn boundary unless it must stay.
    #[test]
    fn concentration_fills_the_decode_member_first() {
        let models = [MemberModel::black_world(), MemberModel::white_world()];
        let mut p = Concentrate { floor_hit: 0.9 };
        assert_eq!(p.place(&views(&models, &[30, 0]), &new_turn()), 0);
        assert_eq!(p.place(&views(&models, &[64, 0]), &new_turn()), 1);
        let away = Asking {
            home: Some(1),
            ..new_turn()
        };
        assert_eq!(p.place(&views(&models, &[10, 1]), &away), 0);
        let staying = Asking { stay: true, ..away };
        assert_eq!(p.place(&views(&models, &[10, 1]), &staying), 1);
    }

    /// A turn goes where it is declared to finish soonest: the idle strong
    /// member first, the other once the strong one's backlog outweighs it; a
    /// running conversation moves only past the margin, and not while it
    /// stays.
    #[test]
    fn a_turn_goes_where_it_finishes_soonest() {
        let models = [MemberModel::black_world(), MemberModel::white_world()];
        let mut p = EarliestFinish { margin: 0.25 };
        let idle = views(&models, &[0, 0]);
        assert_eq!(p.place(&idle, &new_turn()), 0);
        let mut swamped = views(&models, &[20, 0]);
        swamped[0].prefill_backlog = 200_000;
        assert_eq!(p.place(&swamped, &new_turn()), 1);
        let home_bw = Asking {
            home: Some(0),
            ..new_turn()
        };
        assert_eq!(p.place(&swamped, &home_bw), 1);
        let staying = Asking {
            stay: true,
            ..home_bw
        };
        assert_eq!(p.place(&swamped, &staying), 0);
        let mut mild = views(&models, &[1, 0]);
        mild[0].prefill_backlog = 1_000;
        let home_ww = Asking {
            home: Some(1),
            ..new_turn()
        };
        assert_eq!(p.place(&mild, &home_ww), 0);
    }

    /// Only a long prefill is proxied, only from the decode member, and only
    /// while the prefill member keeps up.
    #[test]
    fn only_a_long_prefill_is_proxied() {
        let models = [MemberModel::black_world(), MemberModel::white_world()];
        let mut p = ConcentrateProxy {
            concentrate: Concentrate { floor_hit: 0.9 },
            min_prefill: 4_000,
            max_backlog: 16_000,
        };
        let v = views(&models, &[10, 0]);
        let long = Asking {
            home: Some(0),
            prefill: 6_000,
            ..new_turn()
        };
        assert_eq!(p.proxy(&v, &long, 0), Some(1));
        let short = Asking {
            prefill: 200,
            ..long
        };
        assert_eq!(p.proxy(&v, &short, 0), None);
        assert_eq!(p.proxy(&v, &long, 1), None);
        let mut busy = views(&models, &[10, 0]);
        busy[1].prefill_backlog = 20_000;
        assert_eq!(p.proxy(&busy, &long, 0), None);
    }
}
