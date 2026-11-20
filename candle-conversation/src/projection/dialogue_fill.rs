//! A dialogue's slot as one fill (`docs/hit_driven_projection.md` §7.2, §8).
//!
//! The pieces a dialogue's slot draws on — its inherited lineage, its working
//! set and its own exchanges — become the units of [`fill`], every one of them
//! whole turns ([`rounds`]):
//!
//! - **the lineage**, each exchange whole, scored from its base (§8.4);
//! - **the working set**: each piece's first and last turn — its request and
//!   its answer — as its floor, scored by the better of the two, and its tool
//!   turns — a reading's pages — each at its own turn's score (§7.1, §8.2): every
//!   unit competes on the one band. A reading's momentum in the pool orders
//!   only the minimum ([`FillInput::minimum`]); each unit carries how much the
//!   model's selection read its turns, which orders what leaves;
//! - **the dialogue** (§8.3): its newest exchange as the current one — its
//!   first turn, its newest tool turn and its last kept, its older tool turns
//!   shed first — its first exchange's floor, and its `recent` exchanges before
//!   the current one, whole, as floors; every older exchange as its floor,
//!   scored, and its tool turns, each scored and entering only with the floor.
//!
//! - **every other group** of the projection: each selected exchange whole, at
//!   its selection score, under its layer's window.
//!
//! What the fill places comes back as each exchange's carving
//! ([`rounds::carve`]): its kept turns, whole, in the order the slot seats them.

use std::collections::{HashMap, HashSet};

use crate::projection::fill::{fill, Allowance, Caps, FillInput, FloorExchange, Kind, Unit};
use crate::projection::rounds;
use crate::projection::TurnKey;

/// One working-set piece: its kind for the caps, its exchanges, each its
/// turns in order, and its momentum in the pool — what ranks the readings
/// for the minimum.
#[derive(Debug, Clone)]
pub struct WorkingPiece {
    pub kind: Kind,
    pub exchanges: Vec<Vec<TurnKey>>,
    pub momentum: f32,
}

/// One selected exchange of another group: its kind for the caps
/// ([`Kind::Dialogue`] on the dialogue's layer, [`Kind::Layer`] elsewhere), its
/// turns in order and its selection score. It enters whole or not at all.
#[derive(Debug, Clone)]
pub struct ContextExchange {
    pub kind: Kind,
    pub exchange: Vec<TurnKey>,
    pub score: f32,
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
    /// Every other group's selected exchanges.
    pub context: Vec<ContextExchange>,
    /// The turns the slot holds now, from the boundary before.
    pub held: HashSet<TurnKey>,
    /// What may enter per working-set kind at this boundary; `None` on the
    /// first boundary after a new user message ([`fill`]).
    pub allowance: Option<Allowance>,
}

/// What the fill placed, part by part, each in the order the slot seats it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DialoguePlan {
    pub lineage: Vec<TurnKey>,
    /// One list per working piece, in [`DialogueSlot::working`]'s order.
    pub working: Vec<Vec<TurnKey>>,
    pub own: Vec<TurnKey>,
    /// One list per context exchange, in [`DialogueSlot::context`]'s order.
    pub context: Vec<Vec<TurnKey>>,
    /// How far the floors carried the slot past its room.
    pub over_room: usize,
}

/// Which exchange a unit's rounds belong to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Part {
    Lineage(usize),
    Working(usize, usize),
    Own(usize),
    Context(usize),
}

/// Units under construction: each unit's exchange and rounds.
#[derive(Default)]
struct Units {
    of: Vec<(Part, Vec<usize>)>,
}

impl Units {
    /// `u` under the next id, its rounds those of `part`.
    fn add(&mut self, part: Part, rounds: Vec<usize>, u: Unit) -> Unit {
        let id = self.of.len();
        self.of.push((part, rounds));
        Unit { id, ..u }
    }
}

/// An exchange's floor: its first turn and its last, or the one turn of a
/// plain question and answer.
fn floor_rounds(n: usize) -> Vec<usize> {
    match n {
        0 => Vec::new(),
        1 => vec![0],
        n => vec![0, n - 1],
    }
}

/// Plan `slot` (module docs). `score` is a turn's score; `tokens` a turn's;
/// `reads` how much the model's own selection read it ([`Unit::reads`]).
pub fn plan(
    slot: &DialogueSlot,
    score: impl Fn(TurnKey) -> f32,
    tokens: impl Fn(TurnKey) -> usize,
    reads: impl Fn(TurnKey) -> f32,
) -> DialoguePlan {
    let sum =
        |ex: &[TurnKey], rounds: &[usize]| -> usize { rounds.iter().map(|&r| tokens(ex[r])).sum() };
    let best_of = |ex: &[TurnKey], rounds: &[usize]| -> f32 {
        rounds.iter().map(|&r| score(ex[r])).fold(0.0, f32::max)
    };
    // A unit of `rounds` of `ex`: their tokens, the best of their scores and
    // of their reads.
    let unit = |ex: &[TurnKey], rounds: &[usize], piece: usize, kind: Kind| -> Unit {
        Unit {
            id: 0,
            piece,
            kind,
            tokens: sum(ex, rounds),
            score: best_of(ex, rounds),
            reads: rounds.iter().map(|&r| reads(ex[r])).fold(0.0, f32::max),
        }
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
        let all: Vec<usize> = (0..ex.len()).collect();
        let u = Unit {
            score: *s,
            ..unit(ex, &all, piece(), Kind::Dialogue)
        };
        let u = units.add(Part::Lineage(i), all, u);
        input.scored.push(u);
    }

    // Every other group's exchanges: each whole, at its selection score.
    for (i, cx) in slot.context.iter().enumerate() {
        let all: Vec<usize> = (0..cx.exchange.len()).collect();
        let u = Unit {
            score: cx.score,
            ..unit(&cx.exchange, &all, piece(), cx.kind)
        };
        let u = units.add(Part::Context(i), all, u);
        input.scored.push(u);
    }

    // The working set: every unit at its own turns' scores, on the band.
    let mut readings: Vec<(usize, f32)> = Vec::new();
    for (p, wp) in slot.working.iter().enumerate() {
        let pid = piece();
        let mut floor: Vec<Unit> = Vec::new();
        for (e, ex) in wp.exchanges.iter().enumerate() {
            let n = ex.len();
            if n == 0 {
                continue;
            }
            let part = Part::Working(p, e);
            let fr = floor_rounds(n);
            let pair = units.add(part, fr.clone(), unit(ex, &fr, pid, wp.kind));
            floor.push(pair);
            if n <= 2 {
                // A floor of every turn there is: nothing else to rank.
                input.scored.push(pair);
            }
            for r in 1..n.saturating_sub(1) {
                let u = units.add(part, vec![r], unit(ex, &[r], pid, wp.kind));
                input.scored.push(u);
            }
        }
        if !floor.is_empty() {
            input.piece_floors.insert(pid, floor);
        }
        if wp.kind == Kind::Reading {
            readings.push((pid, wp.momentum));
        }
    }

    // The dialogue's own exchanges.
    let n_ex = slot.own.len();
    let current = n_ex.checked_sub(1);
    let recent_from = current.map_or(0, |c| c.saturating_sub(slot.recent));
    for (e, ex) in slot.own.iter().enumerate() {
        let n = ex.len();
        if n == 0 {
            continue;
        }
        let pid = piece();
        let part = Part::Own(e);
        let one = |units: &mut Units, r: usize| {
            units.add(part, vec![r], unit(ex, &[r], pid, Kind::Dialogue))
        };
        let tools: Vec<usize> = (1..n.saturating_sub(1)).collect();
        if Some(e) == current {
            // Its first turn, its newest tool turn and its last are kept; the
            // older tool turns go first.
            let mut kept: Vec<Unit> = floor_rounds(n)
                .into_iter()
                .map(|r| one(&mut units, r))
                .collect();
            if let Some(&newest) = tools.last() {
                kept.push(one(&mut units, newest));
            }
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
        let fr = floor_rounds(n);
        let pair = units.add(part, fr.clone(), unit(ex, &fr, pid, Kind::Dialogue));
        if e >= recent_from {
            // A recent exchange holds every turn it has.
            let rounds = tools.iter().map(|&r| one(&mut units, r)).collect();
            input.recent.push(FloorExchange {
                kept: vec![pair],
                rounds,
            });
            continue;
        }
        if e == 0 {
            // The dialogue's first exchange's floor.
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

    // A unit is held when the slot holds every turn of it.
    let exchange_of = |part: Part| -> &[TurnKey] {
        match part {
            Part::Lineage(i) => &slot.lineage[i].0,
            Part::Working(p, e) => &slot.working[p].exchanges[e],
            Part::Own(e) => &slot.own[e],
            Part::Context(i) => &slot.context[i].exchange,
        }
    };
    input.held = units
        .of
        .iter()
        .enumerate()
        .filter(|(_, (part, rounds))| {
            let ex = exchange_of(*part);
            !rounds.is_empty() && rounds.iter().all(|&r| slot.held.contains(&ex[r]))
        })
        .map(|(id, _)| id)
        .collect();
    input.allowance = slot.allowance.clone();
    // The readings take the minimum strongest first by the pool's momentum,
    // ties in the working set's order.
    readings.sort_by(|a, b| b.1.total_cmp(&a.1));
    input.minimum = readings.into_iter().map(|(pid, _)| pid).collect();

    let filled = fill(&input, &slot.caps);
    let mut kept: HashMap<Part, HashSet<usize>> = HashMap::new();
    for id in &filled.placed {
        let (part, rounds) = &units.of[*id];
        kept.entry(*part)
            .or_default()
            .extend(rounds.iter().copied());
    }
    let carve_of = |ex: &[TurnKey], part: Part| -> Vec<TurnKey> {
        let Some(k) = kept.get(&part) else {
            return Vec::new();
        };
        // A carving keeps its floor; an exchange whose floor did not enter is
        // not placed at all.
        if !floor_rounds(ex.len()).iter().all(|r| k.contains(r)) {
            return Vec::new();
        }
        let indices: Vec<_> = ex.iter().map(|t| t.index).collect();
        rounds::carve(&indices, |r| k.contains(&r))
            .into_iter()
            .filter_map(|t| ex.iter().find(|key| key.index == t).copied())
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
        context: slot
            .context
            .iter()
            .enumerate()
            .map(|(i, cx)| carve_of(&cx.exchange, Part::Context(i)))
            .collect(),
        over_room: filled.over_room,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{TimelineId, TurnIndex};

    fn k(tl: u64, i: u32) -> TurnKey {
        TurnKey::new(TimelineId::from_raw(tl).unwrap(), TurnIndex(i))
    }

    fn caps() -> Caps {
        Caps {
            readings: 10_000,
            folders: 10_000,
            guides: 10_000,
            journals: 10_000,
            dialogue: 10_000,
            layers: vec![30],
            reading_piece: 10_000,
            minimum: 0,
        }
    }

    /// Every turn is 20 tokens.
    fn twenty(_: TurnKey) -> usize {
        20
    }

    /// A reading of four turns on timeline `tl`.
    fn reading_of(tl: u64) -> WorkingPiece {
        WorkingPiece {
            kind: Kind::Reading,
            exchanges: vec![vec![k(tl, 0), k(tl, 1), k(tl, 2), k(tl, 3)]],
            momentum: 0.0,
        }
    }

    fn reading() -> WorkingPiece {
        reading_of(5)
    }

    fn slot(room: usize, caps: Caps, working: Vec<WorkingPiece>) -> DialogueSlot {
        DialogueSlot {
            room,
            caps,
            recent: 0,
            lineage: vec![],
            working,
            own: vec![],
            context: vec![],
            held: HashSet::new(),
            allowance: None,
        }
    }

    /// With room for everything, every exchange goes in whole, turn by turn.
    #[test]
    fn with_room_every_turn_goes_in_whole() {
        let slot = DialogueSlot {
            room: 10_000,
            caps: caps(),
            recent: 16,
            lineage: vec![(vec![k(9, 0)], 1_000.0)],
            working: vec![reading()],
            own: vec![vec![k(1, 0)], vec![k(1, 1), k(1, 2)]],
            context: vec![],
            held: HashSet::new(),
            allowance: None,
        };
        let p = plan(&slot, |_| 1.0, twenty, |_| 0.0);
        assert_eq!(p.lineage, vec![k(9, 0)]);
        assert_eq!(p.working, vec![vec![k(5, 0), k(5, 1), k(5, 2), k(5, 3)]]);
        assert_eq!(p.own, vec![k(1, 0), k(1, 1), k(1, 2)]);
        assert_eq!(p.over_room, 0);
    }

    /// A reading too large for its room keeps its first and last turns and the
    /// tool turn that ties go to, the oldest, each whole.
    #[test]
    fn a_reading_is_carved_to_whole_turns() {
        let p = plan(&slot(60, caps(), vec![reading()]), |_| 0.0, twenty, |_| 0.0);
        assert_eq!(p.working, vec![vec![k(5, 0), k(5, 1), k(5, 3)]]);
    }

    /// **A reading past its cap keeps the page its own score names, not its
    /// first**: room for everything, but a readings cap of 60 holds the floor
    /// and one tool turn. Turn 5:2 is the one the scan scored, so it is kept
    /// over 5:1, which ties would have kept.
    #[test]
    fn a_capped_reading_keeps_its_best_scored_turn() {
        let capped = Caps {
            readings: 60,
            ..caps()
        };
        let s = slot(10_000, capped, vec![reading()]);
        let p = plan(
            &s,
            |t| if t == k(5, 2) { 900.0 } else { 0.0 },
            twenty,
            |_| 0.0,
        );
        assert_eq!(p.working, vec![vec![k(5, 0), k(5, 2), k(5, 3)]]);
    }

    /// **Rounds compete on the band, whichever piece they come from.** With
    /// room under the readings cap for one round and its floor, the round
    /// scored 800 enters and brings its reading; the other reading's round,
    /// scored 300, does not — nothing of a piece's own strength reorders them.
    #[test]
    fn a_working_round_ranks_by_its_own_score() {
        let capped = Caps {
            readings: 60,
            ..caps()
        };
        let s = slot(10_000, capped, vec![reading_of(5), reading_of(6)]);
        let scores = |t: TurnKey| match (t.timeline.raw(), t.index.0) {
            (5, 1) => 300.0,
            (6, 1) => 800.0,
            _ => 0.0,
        };
        let p = plan(&s, scores, twenty, |_| 0.0);
        assert_eq!(p.working, vec![vec![], vec![k(6, 0), k(6, 1), k(6, 3)]]);
    }

    /// An older exchange the room has no place for leaves whole; the current
    /// exchange and the first exchange's floor stay.
    #[test]
    fn an_older_exchange_leaves_whole() {
        let slot = DialogueSlot {
            own: vec![vec![k(1, 0)], vec![k(1, 1)], vec![k(1, 2)]],
            ..slot(40, caps(), vec![])
        };
        let p = plan(&slot, |_| 0.0, twenty, |_| 0.0);
        assert_eq!(p.own, vec![k(1, 0), k(1, 2)]);
    }

    /// The current exchange past the room sheds its older tool turns, oldest
    /// first, and keeps its first turn, its newest tool turn and its last.
    #[test]
    fn the_current_exchange_sheds_its_oldest_tool_turns() {
        let slot = DialogueSlot {
            own: vec![vec![k(1, 0), k(1, 1), k(1, 2), k(1, 3), k(1, 4)]],
            ..slot(80, caps(), vec![])
        };
        let p = plan(&slot, |_| 0.0, twenty, |_| 0.0);
        assert_eq!(p.own, vec![k(1, 0), k(1, 2), k(1, 3), k(1, 4)]);
    }

    /// **A reading the slot holds keeps its pages at a boundary with an
    /// allowance**: the slot holds 5:1 (and the floor); a stronger page 5:2
    /// may enter only while the readings' allowance has room — none here — so
    /// the held page stays and the new one waits.
    #[test]
    fn a_held_page_stays_while_the_allowance_is_spent() {
        let capped = Caps {
            readings: 60,
            ..caps()
        };
        let scores = |t: TurnKey| if t == k(5, 2) { 900.0 } else { 10.0 };
        let mut s = slot(10_000, capped, vec![reading()]);
        s.held = HashSet::from([k(5, 0), k(5, 1), k(5, 3)]);
        s.allowance = Some(Allowance::from([(Kind::Reading, 0)]));
        let p = plan(&s, scores, twenty, |_| 0.0);
        assert_eq!(p.working, vec![vec![k(5, 0), k(5, 1), k(5, 3)]]);
        // The first boundary after a question chooses whole, by score.
        s.allowance = None;
        let p = plan(&s, scores, twenty, |_| 0.0);
        assert_eq!(p.working, vec![vec![k(5, 0), k(5, 2), k(5, 3)]]);
    }

    /// Another layer's exchanges enter whole by score under that layer's
    /// window: with room for 30, the higher-scored 20-token exchange goes in
    /// and the next, which would make 40, stays out.
    #[test]
    fn another_layers_exchanges_enter_whole_under_its_window() {
        let cx = |tl, s| ContextExchange {
            kind: Kind::Layer(0),
            exchange: vec![k(tl, 0)],
            score: s,
        };
        let slot = DialogueSlot {
            context: vec![cx(7, 0.5), cx(8, 0.9)],
            ..slot(10_000, caps(), vec![])
        };
        let p = plan(&slot, |_| 0.0, twenty, |_| 0.0);
        assert_eq!(p.context, vec![vec![], vec![k(8, 0)]]);
    }
}
