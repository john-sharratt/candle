//! A dialogue's slot as one fill (`docs/hit_driven_projection.md` §7.2, §8).
//!
//! The pieces a dialogue's slot draws on — its inherited lineage, its working
//! set and its own exchanges — become the units of [`fill`]:
//!
//! - **the lineage**, each exchange whole, scored from its base (§8.4);
//! - **the working set**: a locked piece's every round as locks; any other
//!   piece's request and answer as its floor, and its tool rounds — a reading's
//!   pages — by the piece's score (§8.2);
//! - **the dialogue** (§8.3): its newest exchange as the current one — its
//!   request, newest tool round and reply kept, its older tool rounds shed
//!   first — its first exchange's request and answer, and its `recent`
//!   exchanges before the current one, whole, as floors; every older exchange
//!   as its pair, scored, and its tool rounds, each scored and entering only
//!   with the pair.
//!
//! What the fill places comes back as each exchange's carving
//! ([`rounds::carve`]): whole turns where both records are kept, one record
//! where only one is, in the order the slot seats them.

use std::collections::{HashMap, HashSet};

use crate::projection::fill::{fill, Caps, FillInput, FloorExchange, Kind, Unit};
use crate::projection::rounds::{self, records_of, round_count, Placement};
use crate::projection::{TurnIndex, TurnKey};
use crate::turn_layout::TurnRecord;

/// One working-set piece: its kind for the caps, whether it is locked, its
/// score, and its exchanges, each its turns in order.
#[derive(Debug, Clone)]
pub struct WorkingPiece {
    pub kind: Kind,
    pub locked: bool,
    pub score: f32,
    pub exchanges: Vec<Vec<TurnKey>>,
}

/// Everything a dialogue's fill decides over.
#[derive(Debug, Clone)]
pub struct DialogueSlot {
    pub room: usize,
    pub caps: Caps,
    /// Exchanges before the current one that hold their seats whole.
    pub recent: usize,
    /// The inherited lineage's exchanges, each with its score, in order.
    pub lineage: Vec<(Vec<TurnKey>, f32)>,
    pub working: Vec<WorkingPiece>,
    /// The dialogue's own exchanges, oldest first.
    pub own: Vec<Vec<TurnKey>>,
}

/// A turn placed whole (`None`) or one of its records alone.
pub type Placed = (TurnKey, Option<TurnRecord>);

/// What the fill placed, part by part, each in the order the slot seats it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DialoguePlan {
    pub lineage: Vec<Placed>,
    /// One list per working piece, in [`DialogueSlot::working`]'s order.
    pub working: Vec<Vec<Placed>>,
    pub own: Vec<Placed>,
    /// How far locks and floors carried the slot past its room.
    pub over_room: usize,
}

/// Which exchange a unit's rounds belong to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Part {
    Lineage(usize),
    Working(usize, usize),
    Own(usize),
}

/// Units under construction: each unit's exchange and rounds.
#[derive(Default)]
struct Units {
    of: Vec<(Part, Vec<usize>)>,
}

impl Units {
    fn add(
        &mut self,
        part: Part,
        rounds: Vec<usize>,
        piece: usize,
        kind: Kind,
        tokens: usize,
        score: f32,
    ) -> Unit {
        let id = self.of.len();
        self.of.push((part, rounds));
        Unit {
            id,
            piece,
            kind,
            tokens,
            score,
        }
    }
}

/// Plan `slot` (module docs). `score` is a turn's score; `tokens` a record's.
pub fn plan(
    slot: &DialogueSlot,
    score: impl Fn(TurnKey) -> f32,
    tokens: impl Fn(TurnKey, TurnRecord) -> usize,
) -> DialoguePlan {
    let indices = |ex: &[TurnKey]| -> Vec<TurnIndex> { ex.iter().map(|k| k.index).collect() };
    let round_tokens = |ex: &[TurnKey], r: usize| -> usize {
        let idx = indices(ex);
        records_of(&idx, r)
            .into_iter()
            .map(|(t, rec)| {
                let key = ex.iter().find(|k| k.index == t).copied().unwrap_or(ex[0]);
                tokens(key, rec)
            })
            .sum()
    };
    let round_score = |ex: &[TurnKey], r: usize| -> f32 {
        let idx = indices(ex);
        records_of(&idx, r)
            .into_iter()
            .filter_map(|(t, _)| ex.iter().find(|k| k.index == t))
            .map(|k| score(*k))
            .fold(0.0, f32::max)
    };
    let mut units = Units::default();
    let mut input = FillInput {
        room: slot.room,
        ..Default::default()
    };
    // Pieces are numbered across every part, so each exchange is its own.
    let mut next_piece = 0usize;
    let mut piece = || {
        next_piece += 1;
        next_piece - 1
    };

    // The lineage: each exchange whole, at its score.
    for (i, (ex, s)) in slot.lineage.iter().enumerate() {
        let all: Vec<usize> = (0..round_count(ex.len())).collect();
        let t = all.iter().map(|&r| round_tokens(ex, r)).sum();
        let u = units.add(Part::Lineage(i), all, piece(), Kind::Dialogue, t, *s);
        input.scored.push(u);
    }

    // The working set.
    for (p, wp) in slot.working.iter().enumerate() {
        let pid = piece();
        let mut floor: Vec<Unit> = Vec::new();
        for (e, ex) in wp.exchanges.iter().enumerate() {
            let n = round_count(ex.len());
            if n == 0 {
                continue;
            }
            let part = Part::Working(p, e);
            if wp.locked {
                for r in 0..n {
                    let u = units.add(part, vec![r], pid, wp.kind, round_tokens(ex, r), wp.score);
                    input.locks.push(u);
                }
                continue;
            }
            let pair_tokens = round_tokens(ex, 0) + round_tokens(ex, n - 1);
            let pair = units.add(part, vec![0, n - 1], pid, wp.kind, pair_tokens, wp.score);
            floor.push(pair);
            if n == 2 {
                // A plain question and answer: the pair is all there is.
                input.scored.push(pair);
            }
            for r in 1..n - 1 {
                let u = units.add(part, vec![r], pid, wp.kind, round_tokens(ex, r), wp.score);
                input.scored.push(u);
            }
        }
        if !floor.is_empty() {
            input.piece_floors.insert(pid, floor);
        }
    }

    // The dialogue's own exchanges.
    let n_ex = slot.own.len();
    let current = n_ex.checked_sub(1);
    let recent_from = current.map_or(0, |c| c.saturating_sub(slot.recent));
    for (e, ex) in slot.own.iter().enumerate() {
        let n = round_count(ex.len());
        if n == 0 {
            continue;
        }
        let pid = piece();
        let part = Part::Own(e);
        let one = |units: &mut Units, r: usize| {
            units.add(
                part,
                vec![r],
                pid,
                Kind::Dialogue,
                round_tokens(ex, r),
                round_score(ex, r),
            )
        };
        let tools: Vec<usize> = (1..n - 1).collect();
        if Some(e) == current {
            // Its request, its newest tool round and its reply are kept; the
            // older tool rounds go first.
            let mut kept = vec![one(&mut units, 0)];
            if let Some(&newest) = tools.last() {
                kept.push(one(&mut units, newest));
            }
            kept.push(one(&mut units, n - 1));
            let older = tools[..tools.len().saturating_sub(1)]
                .iter()
                .map(|&r| one(&mut units, r))
                .collect();
            input.current = FloorExchange {
                kept,
                rounds: older,
            };
            continue;
        }
        let pair_tokens = round_tokens(ex, 0) + round_tokens(ex, n - 1);
        let pair_score = round_score(ex, 0).max(round_score(ex, n - 1));
        let pair = units.add(
            part,
            vec![0, n - 1],
            pid,
            Kind::Dialogue,
            pair_tokens,
            pair_score,
        );
        if e >= recent_from {
            // A recent exchange holds every round it has.
            let rounds = tools.iter().map(|&r| one(&mut units, r)).collect();
            input.recent.push(FloorExchange {
                kept: vec![pair],
                rounds,
            });
            continue;
        }
        if e == 0 {
            // The dialogue's first request, with its answer.
            input.first_request = Some(pair);
        } else {
            input.scored.push(pair);
        }
        input.piece_floors.insert(pid, vec![pair]);
        for r in tools {
            let u = one(&mut units, r);
            input.scored.push(u);
        }
    }
    // Floors run newest first.
    input.recent.reverse();

    let filled = fill(&input, &slot.caps);
    let mut kept: HashMap<Part, HashSet<usize>> = HashMap::new();
    for id in &filled.placed {
        let (part, rounds) = &units.of[*id];
        kept.entry(*part)
            .or_default()
            .extend(rounds.iter().copied());
    }
    let carve_of = |ex: &[TurnKey], part: Part| -> Vec<Placed> {
        let Some(k) = kept.get(&part) else {
            return Vec::new();
        };
        let n = round_count(ex.len());
        // A carving keeps a request and an answer; an exchange whose pair
        // did not enter is not placed at all.
        if !k.contains(&0) || !k.contains(&(n - 1)) {
            return Vec::new();
        }
        rounds::carve(&indices(ex), |r| k.contains(&r))
            .into_iter()
            .filter_map(|p| match p {
                Placement::Turn(t) => ex.iter().find(|k| k.index == t).map(|k| (*k, None)),
                Placement::Record(t, rec) => {
                    ex.iter().find(|k| k.index == t).map(|k| (*k, Some(rec)))
                }
            })
            .collect()
    };
    DialoguePlan {
        lineage: slot
            .lineage
            .iter()
            .enumerate()
            .flat_map(|(i, (ex, _))| carve_of(ex, Part::Lineage(i)))
            .collect(),
        working: slot
            .working
            .iter()
            .enumerate()
            .map(|(p, wp)| {
                wp.exchanges
                    .iter()
                    .enumerate()
                    .flat_map(|(e, ex)| carve_of(ex, Part::Working(p, e)))
                    .collect()
            })
            .collect(),
        own: slot
            .own
            .iter()
            .enumerate()
            .flat_map(|(e, ex)| carve_of(ex, Part::Own(e)))
            .collect(),
        over_room: filled.over_room,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::TimelineId;

    fn k(tl: u64, i: u32) -> TurnKey {
        TurnKey::new(TimelineId::from_raw(tl).unwrap(), TurnIndex(i))
    }

    fn caps() -> Caps {
        Caps {
            budget: 10_000,
            folder: 10_000,
            guidance: 10_000,
            max_file: 10_000,
            window: 10_000,
        }
    }

    /// Every record is 10 tokens: a round of two records is 20.
    fn ten(_: TurnKey, _: TurnRecord) -> usize {
        10
    }

    /// With room for everything, every exchange goes in whole, turn by turn.
    #[test]
    fn with_room_every_turn_goes_in_whole() {
        let slot = DialogueSlot {
            room: 10_000,
            caps: caps(),
            recent: 16,
            lineage: vec![(vec![k(9, 0)], 1_000.0)],
            working: vec![WorkingPiece {
                kind: Kind::Reading,
                locked: false,
                score: 500.0,
                exchanges: vec![vec![k(5, 0), k(5, 1), k(5, 2)]],
            }],
            own: vec![vec![k(1, 0)], vec![k(1, 1), k(1, 2)]],
        };
        let p = plan(&slot, |_| 1.0, ten);
        assert_eq!(p.lineage, vec![(k(9, 0), None)]);
        assert_eq!(
            p.working,
            vec![vec![(k(5, 0), None), (k(5, 1), None), (k(5, 2), None)]]
        );
        assert_eq!(
            p.own,
            vec![(k(1, 0), None), (k(1, 1), None), (k(1, 2), None)]
        );
        assert_eq!(p.over_room, 0);
    }

    /// A reading too large for its room keeps its request and answer and the
    /// pages that score highest; the pages it drops split their turns into
    /// the records the kept rounds hold.
    #[test]
    fn a_reading_is_carved_to_its_best_pages() {
        // A reading of four turns: request, three page rounds, answer — five
        // rounds; with 20-token rounds and 10-token request and answer, room
        // for the pair and one page.
        let reading = vec![k(5, 0), k(5, 1), k(5, 2), k(5, 3)];
        let slot = DialogueSlot {
            room: 40,
            caps: caps(),
            recent: 0,
            lineage: vec![],
            working: vec![WorkingPiece {
                kind: Kind::Reading,
                locked: false,
                score: 500.0,
                exchanges: vec![reading.clone()],
            }],
            own: vec![],
        };
        let p = plan(&slot, |_| 0.0, ten);
        // Ties break oldest first, so page round 1 is the one kept: the
        // request (5:0 user) and round 1 (5:0 assistant, 5:1 user) — 5:0
        // whole — then 5:1's call is dropped with round 2, 5:2 entirely, and
        // the answer, 5:3's assistant record.
        assert_eq!(
            p.working,
            vec![vec![
                (k(5, 0), None),
                (k(5, 1), Some(TurnRecord::User)),
                (k(5, 3), Some(TurnRecord::Assistant)),
            ]]
        );
    }

    /// An older exchange the room has no place for leaves whole; the current
    /// exchange and the first request stay.
    #[test]
    fn an_older_exchange_leaves_whole() {
        let slot = DialogueSlot {
            room: 40,
            caps: caps(),
            recent: 0,
            lineage: vec![],
            working: vec![],
            own: vec![vec![k(1, 0)], vec![k(1, 1)], vec![k(1, 2)]],
        };
        let p = plan(&slot, |_| 0.0, ten);
        assert_eq!(p.own, vec![(k(1, 0), None), (k(1, 2), None)]);
    }

    /// A locked piece goes in whole even past the room.
    #[test]
    fn a_locked_piece_goes_in_whole() {
        let slot = DialogueSlot {
            room: 10,
            caps: caps(),
            recent: 0,
            lineage: vec![],
            working: vec![WorkingPiece {
                kind: Kind::Reading,
                locked: true,
                score: 0.0,
                exchanges: vec![vec![k(5, 0), k(5, 1)]],
            }],
            own: vec![],
        };
        let p = plan(&slot, |_| 0.0, ten);
        assert_eq!(p.working, vec![vec![(k(5, 0), None), (k(5, 1), None)]]);
        assert_eq!(p.over_room, 30);
    }
}
