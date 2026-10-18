//! The simulated cluster's event loop.
//!
//! Each member runs one forward at a time, alternating a decode step over
//! every row decoding under its lease with a prefill forward packing the
//! prompt tokens queued on it, up to its width. A conversation moves through
//! its turns' segments: queued for prefill (at home, or proxied), across the
//! link when its state must move, decoding, then the person's pause. Each
//! turn's end replicates its state to every other member over the link, and a
//! conversation moved to another member waits for that copy to have arrived.
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
enum Next {
    Prefill { at: usize, left: usize },
    Decode { left: usize },
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
    /// The next segment of a running turn, not yet started.
    SegmentDue,
    /// State on the link.
    Crossing {
        until: f64,
        next: Next,
    },
    Prefill {
        at: usize,
        left: usize,
    },
    Decode {
        left: usize,
    },
    Done,
}

#[derive(Debug, Clone)]
struct Conv {
    home: Option<usize>,
    stay: bool,
    turn: usize,
    seg: usize,
    phase: Phase,
    kv: f64,
    /// When its latest turn's state reached the other members.
    replicated_at: f64,
    due: f64,
    first_token: Option<f64>,
    last_token: Option<f64>,
    longest_gap: f64,
    tokens: usize,
    restarted: bool,
    /// Prompt tokens of this turn prefilled by a proxy: their KV came back
    /// with its snapshot and is not replicated again at the turn's end.
    proxied: usize,
}

impl Conv {
    /// Clear the running turn's measurements, to start it again.
    fn restart(&mut self) {
        self.seg = 0;
        self.first_token = None;
        self.last_token = None;
        self.longest_gap = 0.0;
        self.tokens = 0;
        self.proxied = 0;
        self.restarted = true;
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
                stay: false,
                turn: 0,
                seg: 0,
                phase: Phase::NotYet,
                kv: c.context as f64,
                replicated_at: 0.0,
                due: c.arrive_s,
                first_token: None,
                last_token: None,
                longest_gap: 0.0,
                tokens: 0,
                restarted: false,
                proxied: 0,
            })
            .collect(),
        link,
        records: Vec::new(),
    };
    let start = workload
        .iter()
        .map(|c| c.arrive_s)
        .fold(f64::INFINITY, f64::min);
    let mut now = if start.is_finite() { start } else { 0.0 };
    let mut out = outage;
    loop {
        if let Some(o) = out.filter(|o| now >= o.at) {
            run.take_out(o.member);
            out = None;
        }
        run.finish_forwards(now);
        run.wake(now);
        run.start_turns_and_segments(now, policy);
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
        if start.is_finite() { start } else { 0.0 },
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
            .filter(|c| match c.phase {
                Phase::Prefill { at, .. } => at == m || c.home == Some(m),
                Phase::Crossing { .. } | Phase::Decode { .. } | Phase::SegmentDue => {
                    c.home == Some(m)
                }
                _ => false,
            })
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
                    .map(|c| match c.phase {
                        Phase::Prefill { at, left } if at == m => left,
                        _ => 0,
                    })
                    .sum(),
            })
            .collect()
    }

    /// A member dropping: its forward is lost, every turn running under its
    /// lease starts again wherever it is placed next, and a prefill proxied to
    /// it goes back to its home.
    fn take_out(&mut self, member: usize) {
        self.members[member].up = false;
        self.members[member].job = None;
        for c in self.convs.iter_mut() {
            let running = matches!(
                c.phase,
                Phase::Crossing { .. }
                    | Phase::Prefill { .. }
                    | Phase::Decode { .. }
                    | Phase::SegmentDue
            );
            if c.home == Some(member) {
                c.home = None;
                c.stay = false;
                if running {
                    c.phase = Phase::TurnDue;
                    c.restart();
                }
            } else if let (Phase::Prefill { at, left }, Some(h)) = (c.phase, c.home) {
                if at == member {
                    c.phase = Phase::Prefill { at: h, left };
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
                        self.prefilled(i, tokens, t);
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
            c.phase = Phase::SegmentDue;
            return;
        }
        self.end_turn(i, t);
    }

    fn prefilled(&mut self, i: usize, tokens: usize, t: f64) {
        let seg = self.segment(i);
        let c = &mut self.convs[i];
        let Phase::Prefill { at, left } = c.phase else {
            return;
        };
        c.kv += tokens as f64;
        if left > tokens {
            c.phase = Phase::Prefill {
                at,
                left: left - tokens,
            };
            return;
        }
        c.phase = if Some(at) == c.home {
            Phase::Decode { left: seg.decode }
        } else {
            // Proxied: its KV and snapshot come back before it decodes.
            c.proxied += seg.prefill;
            Phase::Crossing {
                until: self.link.send(t, Link::state_bytes(seg.prefill)),
                next: Next::Decode { left: seg.decode },
            }
        };
    }

    /// The turn is done: recorded, its state replicated to every other
    /// member, and the person's pause before the next.
    fn end_turn(&mut self, i: usize, t: f64) {
        let ideal = self.ideal(i);
        let others = self
            .members
            .iter()
            .filter(|m| m.up)
            .count()
            .saturating_sub(1);
        let c = &mut self.convs[i];
        let home = c.home.expect("a running turn has a home");
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
        let turn_tokens: usize = self.workload[i].turns[c.turn]
            .segments
            .iter()
            .map(|s| s.prefill + s.decode)
            .sum::<usize>()
            - c.proxied;
        let mut arrived = t;
        for _ in 0..others {
            arrived = self.link.send(t, Link::state_bytes(turn_tokens));
        }
        c.replicated_at = arrived;
        c.first_token = None;
        c.last_token = None;
        c.longest_gap = 0.0;
        c.tokens = 0;
        c.restarted = false;
        c.proxied = 0;
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
                Phase::Crossing { until, next } if until <= now => {
                    c.phase = match next {
                        Next::Prefill { at, left } => Phase::Prefill { at, left },
                        Next::Decode { left } => Phase::Decode { left },
                    };
                }
                _ => {}
            }
        }
    }

    fn start_turns_and_segments(&mut self, now: f64, policy: &mut dyn Policy) {
        for i in 0..self.convs.len() {
            match self.convs[i].phase {
                Phase::TurnDue => {
                    self.place(i, policy);
                    self.begin_segment(i, now, policy);
                }
                Phase::SegmentDue => self.begin_segment(i, now, policy),
                _ => {}
            }
        }
    }

    /// Where a due turn runs: the policy's choice, held where it is while it
    /// must stay, and never on a member that is down.
    fn place(&mut self, i: usize, policy: &mut dyn Policy) {
        let seg = self.segment(i);
        let asking = Asking {
            home: self.convs[i].home,
            stay: self.convs[i].stay,
            prefill: seg.prefill,
            decode: seg.decode,
        };
        let chosen = policy.place(&self.views(), &asking);
        let c = &self.convs[i];
        let mut home = match c.home {
            Some(h) if c.stay => h,
            _ => chosen,
        };
        if !self.members[home].up {
            home = (0..self.members.len())
                .find(|m| self.members[*m].up)
                .expect("a member is up");
        }
        let c = &mut self.convs[i];
        let moved = c.home.is_some_and(|h| h != home);
        c.stay = moved;
        c.home = Some(home);
    }

    /// Queue the segment's prefill: at home, or proxied. A turn starts from
    /// the state its last turn replicated; a segment in the middle of a turn
    /// carries the home's state across first.
    fn begin_segment(&mut self, i: usize, now: f64, policy: &mut dyn Policy) {
        let seg = self.segment(i);
        let home = self.convs[i].home.expect("a running turn has a home");
        let asking = Asking {
            home: Some(home),
            stay: self.convs[i].stay,
            prefill: seg.prefill,
            decode: seg.decode,
        };
        let proxy = policy
            .proxy(&self.views(), &asking, home)
            .filter(|to| *to != home && self.members.get(*to).is_some_and(|m| m.up));
        let at = proxy.unwrap_or(home);
        let c = &self.convs[i];
        let moved_home = c.stay;
        let ready = if at == home && !moved_home {
            now
        } else if c.seg == 0 {
            c.replicated_at.max(now)
        } else if at == home {
            now
        } else {
            self.link.send(now, Link::state_bytes(0))
        };
        let next = Next::Prefill {
            at,
            left: seg.prefill,
        };
        self.convs[i].phase = if ready > now {
            Phase::Crossing { until: ready, next }
        } else {
            Phase::Prefill {
                at,
                left: seg.prefill,
            }
        };
    }

    fn start_forwards(&mut self, now: f64) {
        for m in 0..self.members.len() {
            let member = &self.members[m];
            if !member.up || member.job.is_some() || member.busy_until > now {
                continue;
            }
            let held = self.held(m);
            let mut rows: Vec<usize> = (0..self.convs.len())
                .filter(|&i| {
                    self.convs[i].home == Some(m)
                        && matches!(self.convs[i].phase, Phase::Decode { .. })
                })
                .collect();
            // The rows furthest behind go first when there are more than a
            // step carries.
            rows.sort_by_key(|&i| (self.convs[i].tokens, i));
            rows.truncate(self.models[m].max_rows);
            let mut queue: Vec<usize> = (0..self.convs.len())
                .filter(|&i| matches!(self.convs[i].phase, Phase::Prefill { at, .. } if at == m))
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
                    let Phase::Prefill { left, .. } = self.convs[i].phase else {
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
                Phase::Waiting { until } | Phase::Crossing { until, .. } => next = next.min(until),
                _ => {}
            }
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::policy::{Concentrate, ConcentrateProxy, Today};
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
            Box::new(ConcentrateProxy {
                concentrate: Concentrate { floor_hit: 0.9 },
                min_prefill: 4_000,
                max_backlog: 16_000,
            }),
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
}
