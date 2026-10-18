//! The simulated cluster's event loop.
//!
//! Each member runs one forward at a time, alternating a decode step over
//! every row decoding under its lease with a prefill forward packing the
//! prompt tokens queued on it, up to its width. A conversation's lease is held
//! for one turn — its prefills and its decodes — and its next turn is placed
//! among the members that are up and hold its state through the turn just
//! ended. A turn's end replicates its state to every other member over the
//! link, FIFO, and a member is eligible for the next turn once its copy has
//! arrived: a member still waiting for it is skipped, never waited on.
//!
//! Conversation `i`'s state is `convs[i]`, and its spec `workload[i]`.

use super::link::Link;
use super::model::MemberModel;
use super::policy::{Asking, MemberView, Policy};
use super::report::{Report, TurnRecord};
use super::workload::{Conversation, Segment, Turn};

/// A member taken out mid-run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Outage {
    pub member: usize,
    pub at: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Phase {
    NotYet,
    /// The person's pause before the next turn.
    Waiting {
        until: f64,
    },
    /// A turn due, not yet placed.
    TurnDue,
    Prefill {
        left: usize,
    },
    Decode {
        left: usize,
    },
    Done,
}

#[derive(Debug, Clone)]
struct Conv {
    /// The member whose lease the turn runs under, or ran its last under.
    home: Option<usize>,
    turn: usize,
    seg: usize,
    phase: Phase,
    kv: f64,
    /// When each member came to hold its state through its latest turn;
    /// `None` for a member that does not yet.
    replicated: Vec<Option<f64>>,
    due: f64,
    first_token: Option<f64>,
    last_token: Option<f64>,
    longest_gap: f64,
    tokens: usize,
    restarted: bool,
}

impl Conv {
    /// Clear the running turn's measurements, to start it again.
    fn restart(&mut self) {
        self.seg = 0;
        self.first_token = None;
        self.last_token = None;
        self.longest_gap = 0.0;
        self.tokens = 0;
        self.restarted = true;
    }

    fn running(&self) -> bool {
        matches!(self.phase, Phase::Prefill { .. } | Phase::Decode { .. })
    }
}

#[derive(Debug, Clone)]
enum Job {
    Decode(Vec<usize>),
    Prefill(Vec<(usize, usize)>),
}

#[derive(Debug, Clone)]
struct Member {
    busy_until: f64,
    job: Option<Job>,
    last_was_decode: bool,
    up: bool,
}

/// Everything a run carries between steps.
struct Run<'a> {
    models: &'a [MemberModel],
    workload: &'a [Conversation],
    members: Vec<Member>,
    convs: Vec<Conv>,
    link: Link,
    records: Vec<TurnRecord>,
}

/// Run `workload` on `models` under `policy`, over `link`, optionally taking
/// a member out. `models[0]` is the member a turn's ideal time is measured on.
pub fn simulate(
    models: &[MemberModel],
    workload: &[Conversation],
    policy: &mut dyn Policy,
    link: Link,
    outage: Option<Outage>,
) -> Report {
    let mut run = Run {
        models,
        workload,
        members: models
            .iter()
            .map(|_| Member {
                busy_until: 0.0,
                job: None,
                last_was_decode: false,
                up: true,
            })
            .collect(),
        convs: workload
            .iter()
            .map(|c| Conv {
                home: None,
                turn: 0,
                seg: 0,
                phase: Phase::NotYet,
                kv: c.context as f64,
                // Every member holds a conversation's history before its first
                // turn: the substrate replicated it.
                replicated: vec![Some(0.0); models.len()],
                due: c.arrive_s,
                first_token: None,
                last_token: None,
                longest_gap: 0.0,
                tokens: 0,
                restarted: false,
            })
            .collect(),
        link,
        records: Vec::new(),
    };
    let first = workload
        .iter()
        .map(|c| c.arrive_s)
        .fold(f64::INFINITY, f64::min);
    let start = if first.is_finite() { first } else { 0.0 };
    let mut now = start;
    let mut out = outage;
    loop {
        if let Some(o) = out.filter(|o| now >= o.at) {
            run.take_out(o.member);
            out = None;
        }
        run.finish_forwards(now);
        run.wake(now);
        run.start_turns(now, policy);
        run.start_forwards(now);
        let mut next = run.next_event();
        if let Some(o) = out {
            next = next.min(o.at);
        }
        if !next.is_finite() {
            break;
        }
        now = next.max(now);
    }
    Report::of(
        policy.name(),
        &run.records,
        models.len(),
        start,
        run.link.carried,
    )
}

impl Run<'_> {
    fn segment(&self, i: usize) -> Segment {
        let c = &self.convs[i];
        self.workload[i].turns[c.turn].segments[c.seg]
    }

    fn held(&self, m: usize) -> usize {
        self.convs
            .iter()
            .filter(|c| c.home == Some(m) && c.running())
            .count()
    }

    fn views(&self) -> Vec<MemberView<'_>> {
        (0..self.models.len())
            .map(|m| MemberView {
                model: &self.models[m],
                held: self.held(m),
                decoding: self
                    .convs
                    .iter()
                    .filter(|c| c.home == Some(m) && matches!(c.phase, Phase::Decode { .. }))
                    .count(),
                prefill_backlog: self
                    .convs
                    .iter()
                    .filter(|c| c.home == Some(m))
                    .map(|c| match c.phase {
                        Phase::Prefill { left } => left,
                        _ => 0,
                    })
                    .sum(),
            })
            .collect()
    }

    /// A member dropping: its forward is lost, and every turn running under
    /// its lease starts again wherever it is placed next.
    fn take_out(&mut self, member: usize) {
        self.members[member].up = false;
        self.members[member].job = None;
        for c in self.convs.iter_mut() {
            c.replicated[member] = None;
            if c.home == Some(member) {
                c.home = None;
                if c.running() {
                    c.phase = Phase::TurnDue;
                    c.restart();
                }
            }
        }
    }

    fn finish_forwards(&mut self, now: f64) {
        for m in 0..self.members.len() {
            if self.members[m].busy_until > now {
                continue;
            }
            let Some(job) = self.members[m].job.take() else {
                continue;
            };
            let t = self.members[m].busy_until;
            match job {
                Job::Decode(rows) => {
                    for i in rows {
                        self.decoded(i, t);
                    }
                }
                Job::Prefill(parts) => {
                    for (i, tokens) in parts {
                        self.prefilled(i, tokens);
                    }
                }
            }
        }
    }

    fn decoded(&mut self, i: usize, t: f64) {
        let c = &mut self.convs[i];
        let Phase::Decode { left } = c.phase else {
            return;
        };
        if let Some(last) = c.last_token {
            c.longest_gap = c.longest_gap.max(t - last);
        }
        c.first_token.get_or_insert(t);
        c.last_token = Some(t);
        c.tokens += 1;
        c.kv += 1.0;
        if left > 1 {
            c.phase = Phase::Decode { left: left - 1 };
            return;
        }
        let turn: &Turn = &self.workload[i].turns[c.turn];
        if c.seg + 1 < turn.segments.len() {
            c.seg += 1;
            let next = turn.segments[c.seg];
            c.phase = Phase::Prefill { left: next.prefill };
            return;
        }
        self.end_turn(i, t);
    }

    fn prefilled(&mut self, i: usize, tokens: usize) {
        let seg = self.segment(i);
        let c = &mut self.convs[i];
        let Phase::Prefill { left } = c.phase else {
            return;
        };
        c.kv += tokens as f64;
        c.phase = if left > tokens {
            Phase::Prefill {
                left: left - tokens,
            }
        } else {
            Phase::Decode { left: seg.decode }
        };
    }

    /// The turn is done: recorded, its state replicated to every other member,
    /// and the person's pause before the next.
    fn end_turn(&mut self, i: usize, t: f64) {
        let ideal = self.ideal(i);
        let turn_tokens: usize = self.workload[i].turns[self.convs[i].turn]
            .segments
            .iter()
            .map(|s| s.prefill + s.decode)
            .sum();
        let home = self.convs[i].home.expect("a running turn has a home");
        let c = &self.convs[i];
        self.records.push(TurnRecord {
            conversation: i,
            turn: c.turn,
            member: home,
            due: c.due,
            first_token: c.first_token.unwrap_or(t),
            end: t,
            longest_gap: c.longest_gap,
            tokens: c.tokens,
            ideal,
            restarted: c.restarted,
        });
        let mut replicated = vec![None; self.members.len()];
        replicated[home] = Some(t);
        for (m, member) in self.members.iter().enumerate() {
            if m != home && member.up {
                replicated[m] = Some(self.link.send(t, Link::state_bytes(turn_tokens)));
            }
        }
        let c = &mut self.convs[i];
        c.replicated = replicated;
        c.first_token = None;
        c.last_token = None;
        c.longest_gap = 0.0;
        c.tokens = 0;
        c.restarted = false;
        c.seg = 0;
        let turns = &self.workload[i].turns;
        if c.turn + 1 < turns.len() {
            c.turn += 1;
            c.phase = Phase::Waiting {
                until: t + turns[c.turn].think_s,
            };
        } else {
            c.phase = Phase::Done;
        }
    }

    /// The turn's time alone on `models[0]`, idle. Called at its end, so its
    /// own tokens come off the context first.
    fn ideal(&self, i: usize) -> f64 {
        let m = &self.models[0];
        let c = &self.convs[i];
        let segments = &self.workload[i].turns[c.turn].segments;
        let own: usize = segments.iter().map(|s| s.prefill + s.decode).sum();
        let mut kv = c.kv - own as f64;
        let mut s = 0.0;
        for seg in segments {
            let mut left = seg.prefill;
            while left > 0 {
                let take = left.min(m.prefill_width);
                s += m.prefill_ms(take, kv, 1) / 1000.0;
                left -= take;
            }
            s += seg.decode as f64 * m.decode_ms(1, 1, false) / 1000.0;
            kv += (seg.prefill + seg.decode) as f64;
        }
        s
    }

    fn wake(&mut self, now: f64) {
        for c in self.convs.iter_mut() {
            match c.phase {
                Phase::NotYet if c.due <= now => c.phase = Phase::TurnDue,
                Phase::Waiting { until } if until <= now => {
                    c.due = until;
                    c.phase = Phase::TurnDue;
                }
                _ => {}
            }
        }
    }

    /// Place each due turn among the members that are up and hold its state
    /// through its latest turn, and queue its first prefill there.
    fn start_turns(&mut self, now: f64, policy: &mut dyn Policy) {
        for i in 0..self.convs.len() {
            if self.convs[i].phase != Phase::TurnDue {
                continue;
            }
            let eligible: Vec<bool> = (0..self.members.len())
                .map(|m| {
                    self.members[m].up && self.convs[i].replicated[m].is_some_and(|at| at <= now)
                })
                .collect();
            if !eligible.iter().any(|e| *e) {
                // Nobody holds it yet: the only copy is still on the link.
                continue;
            }
            let seg = self.segment(i);
            let asking = Asking {
                home: self.convs[i].home,
                prefill: seg.prefill,
                decode: seg.decode,
            };
            let home = policy.place(&self.views(), &eligible, &asking);
            let c = &mut self.convs[i];
            c.home = Some(home);
            c.phase = Phase::Prefill { left: seg.prefill };
        }
    }

    fn start_forwards(&mut self, now: f64) {
        for m in 0..self.members.len() {
            let member = &self.members[m];
            if !member.up || member.job.is_some() || member.busy_until > now {
                continue;
            }
            let held = self.held(m);
            let mine = |i: &usize| self.convs[*i].home == Some(m);
            let mut rows: Vec<usize> = (0..self.convs.len())
                .filter(mine)
                .filter(|&i| matches!(self.convs[i].phase, Phase::Decode { .. }))
                .collect();
            // The rows furthest behind go first when there are more than a
            // step carries.
            rows.sort_by_key(|&i| (self.convs[i].tokens, i));
            rows.truncate(self.models[m].max_rows);
            let mut queue: Vec<usize> = (0..self.convs.len())
                .filter(mine)
                .filter(|&i| matches!(self.convs[i].phase, Phase::Prefill { .. }))
                .collect();
            queue.sort_by(|&a, &b| {
                self.convs[a]
                    .due
                    .total_cmp(&self.convs[b].due)
                    .then(a.cmp(&b))
            });
            let decode = !rows.is_empty() && (!self.members[m].last_was_decode || queue.is_empty());
            if decode {
                let ms = self.models[m].decode_ms(rows.len(), held, !queue.is_empty());
                let member = &mut self.members[m];
                member.busy_until = now + ms / 1000.0;
                member.job = Some(Job::Decode(rows));
                member.last_was_decode = true;
            } else if !queue.is_empty() {
                let width = self.models[m].prefill_width;
                let mut parts = Vec::new();
                let (mut used, mut kv) = (0, 0.0);
                for i in queue {
                    let Phase::Prefill { left } = self.convs[i].phase else {
                        continue;
                    };
                    let take = left.min(width - used);
                    if take == 0 {
                        break;
                    }
                    parts.push((i, take));
                    used += take;
                    kv += self.convs[i].kv;
                }
                let ms = self.models[m].prefill_ms(used, kv, held);
                let member = &mut self.members[m];
                member.busy_until = now + ms / 1000.0;
                member.job = Some(Job::Prefill(parts));
                member.last_was_decode = false;
            }
        }
    }

    /// The next moment anything happens: a forward ending, a pause ending, a
    /// conversation arriving, or a replica landing that makes a due turn
    /// placeable.
    fn next_event(&self) -> f64 {
        let mut next = f64::INFINITY;
        for m in &self.members {
            if m.up && m.job.is_some() {
                next = next.min(m.busy_until);
            }
        }
        for c in &self.convs {
            match c.phase {
                Phase::NotYet => next = next.min(c.due),
                Phase::Waiting { until } => next = next.min(until),
                Phase::TurnDue => {
                    for at in c.replicated.iter().flatten() {
                        next = next.min(*at);
                    }
                }
                _ => {}
            }
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::policy::{Concentrate, EarliestFinish, Today};
    use crate::sim::workload::{generate, Kind};

    fn one_turn(prefill: usize, decode: usize) -> Vec<Conversation> {
        vec![Conversation {
            id: 0,
            arrive_s: 0.0,
            context: 10_000,
            turns: vec![Turn {
                think_s: 0.0,
                segments: vec![Segment { prefill, decode }],
            }],
        }]
    }

    /// One conversation alone runs in exactly its ideal time: one prefill
    /// forward, then each decode step alone.
    #[test]
    fn a_lone_turn_takes_its_ideal() {
        let models = [MemberModel::black_world()];
        let r = simulate(
            &models,
            &one_turn(1_000, 10),
            &mut Today,
            Link::gigabit(),
            None,
        );
        assert_eq!(r.turns, 1);
        assert!((r.slowdown_p95 - 1.0).abs() < 1e-9, "{}", r.slowdown_p95);
        let b = &models[0];
        let prefill = b.prefill_ms(1_000, 10_000.0, 1) / 1000.0;
        assert!((r.first_token_p50 - (prefill + b.decode_ms(1, 1, false) / 1000.0)).abs() < 1e-9);
        assert_eq!(r.link_gb, 0.0);
    }

    /// Every turn of every conversation finishes, under every policy, and a
    /// turn's state crosses the link once per other member.
    #[test]
    fn every_turn_finishes_under_every_policy() {
        let models = [MemberModel::black_world(), MemberModel::white_world()];
        let work = generate(Kind::Read, 12, 20.0, 5);
        let turns: usize = work.iter().map(|c| c.turns.len()).sum();
        let mut policies: Vec<Box<dyn Policy>> = vec![
            Box::new(Today),
            Box::new(Concentrate { floor_hit: 0.9 }),
            Box::new(EarliestFinish),
        ];
        for p in policies.iter_mut() {
            let r = simulate(&models, &work, p.as_mut(), Link::gigabit(), None);
            assert_eq!(r.turns, turns, "{}", r.policy);
            assert!(r.link_gb >= turns as f64 * 0.113, "{}", r.policy);
            assert!(r.fairness > 0.0 && r.fairness <= 1.0);
        }
    }

    /// A member dropping mid-run: its turns start again elsewhere and every
    /// turn still finishes.
    #[test]
    fn a_dropped_member_s_turns_finish_elsewhere() {
        let models = [MemberModel::black_world(), MemberModel::white_world()];
        let work = generate(Kind::Chat, 16, 5.0, 9);
        let turns: usize = work.iter().map(|c| c.turns.len()).sum();
        let r = simulate(
            &models,
            &work,
            &mut Today,
            Link::gigabit(),
            Some(Outage { member: 1, at: 6.0 }),
        );
        assert_eq!(r.turns, turns);
        assert!(r.restarted > 0);
        assert_eq!(r.per_member[1], 0, "nothing finishes on a dropped member");
    }

    /// A turn is never placed on a member still waiting for the
    /// conversation's state: over a link too slow to deliver any, no
    /// conversation moves, though the policy would move some given the chance.
    #[test]
    fn a_member_without_the_state_is_skipped() {
        let models = [MemberModel::black_world(), MemberModel::white_world()];
        let work = generate(Kind::Read, 24, 5.0, 3);
        let turns: usize = work.iter().map(|c| c.turns.len()).sum();
        let stuck = simulate(&models, &work, &mut EarliestFinish, Link::new(1.0), None);
        assert_eq!(stuck.turns, turns);
        assert_eq!(stuck.moves, 0);
        let free = simulate(&models, &work, &mut EarliestFinish, Link::new(1e12), None);
        assert!(
            free.moves > 0,
            "the policy moves some turns given the state"
        );
    }
}
