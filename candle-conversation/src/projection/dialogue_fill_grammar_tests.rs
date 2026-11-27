//! **Every carving the slot fill makes keeps whole turns and its floor**
//! (`docs/hit_driven_projection.md` §8.1, §13.2): over random pieces — a
//! lineage, working-set pieces, the dialogue's own exchanges and other groups'
//! exchanges — at random turn sizes, scores, caps and room, each exchange the
//! plan places holds its first and its last turn, and between them only some
//! of its own turns, in order. Every turn is placed whole, so the roles
//! alternate `U A U A …` by construction. The caps bind: no working-set kind
//! passes its share of the room.

use std::collections::{HashMap, HashSet};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::dialogue_fill::{plan, ContextExchange, DialoguePlan, DialogueSlot, WorkingPiece};
use super::fill::{minimum_cap, reading_piece_cap, Caps, Kind};
use super::{TimelineId, TurnIndex, TurnKey};

/// One exchange of `n` turns on a timeline of its own, so the plan's turns can
/// be told apart by exchange.
fn exchange(next_timeline: &mut u64, n: u32) -> Vec<TurnKey> {
    *next_timeline += 1;
    let tl = TimelineId::from_raw(*next_timeline).unwrap();
    (0..n).map(|i| TurnKey::new(tl, TurnIndex(i))).collect()
}

/// A random slot, with each turn's tokens and score.
struct Case {
    slot: DialogueSlot,
    tokens: HashMap<TurnKey, usize>,
    scores: HashMap<TurnKey, f32>,
}

fn random_case(rng: &mut StdRng) -> Case {
    let mut tl = 0u64;
    let mut ex = |rng: &mut StdRng, max: u32| exchange(&mut tl, rng.random_range(1..=max));
    let lineage = (0..rng.random_range(0..=3))
        .map(|_| (ex(rng, 4), rng.random_range(0.0..1000.0)))
        .collect();
    let kinds = [Kind::Reading, Kind::Folder, Kind::Guide, Kind::Journal];
    let working = (0..rng.random_range(0..=4))
        .map(|_| WorkingPiece {
            kind: kinds[rng.random_range(0..kinds.len())],
            exchanges: (0..rng.random_range(1..=3)).map(|_| ex(rng, 6)).collect(),
            momentum: rng.random_range(0.0..1000.0),
        })
        .collect();
    let own = (0..rng.random_range(0..=5)).map(|_| ex(rng, 5)).collect();
    let context = (0..rng.random_range(0..=3))
        .map(|_| ContextExchange {
            kind: if rng.random_bool(0.5) {
                Kind::Dialogue
            } else {
                Kind::Layer(0)
            },
            exchange: ex(rng, 3),
            score: rng.random_range(0.0..1.0),
        })
        .collect();
    let readings = rng.random_range(0..=400);
    let caps = Caps {
        readings,
        folders: rng.random_range(0..=400),
        guides: rng.random_range(0..=400),
        journals: rng.random_range(0..=400),
        dialogue: rng.random_range(0..=400),
        layers: vec![rng.random_range(0..=400)],
        reading_piece: reading_piece_cap(readings),
        minimum: minimum_cap(readings),
    };
    // Half the cases are a boundary with an allowance, the slot holding some
    // of the turns already.
    let allowance = rng.random_bool(0.5).then(|| caps.allowance());
    let mut slot = DialogueSlot {
        room: rng.random_range(0..=600),
        caps,
        window: rng.random_range(0..=400),
        lineage,
        working,
        own,
        context,
        held: HashSet::new(),
        allowance,
    };
    let held: HashSet<TurnKey> = slot
        .working
        .iter()
        .flat_map(|w| w.exchanges.iter().flatten())
        .copied()
        .filter(|_| rng.random_bool(0.4))
        .collect();
    slot.held = held;
    let mut tokens = HashMap::new();
    let mut scores = HashMap::new();
    let every_exchange: Vec<&Vec<TurnKey>> = slot
        .lineage
        .iter()
        .map(|(e, _)| e)
        .chain(slot.working.iter().flat_map(|w| w.exchanges.iter()))
        .chain(slot.own.iter())
        .chain(slot.context.iter().map(|c| &c.exchange))
        .collect();
    for e in every_exchange {
        for &k in e {
            tokens.insert(k, rng.random_range(1..=80));
            scores.insert(k, rng.random_range(0.0..1000.0));
        }
    }
    Case {
        slot,
        tokens,
        scores,
    }
}

/// Every turn a plan places, in the order it seats them.
fn all_placed(plan: &DialoguePlan) -> Vec<TurnKey> {
    plan.lineage
        .iter()
        .chain(plan.working.iter().flatten())
        .chain(plan.own.iter())
        .chain(plan.context.iter().flatten())
        .copied()
        .collect()
}

/// The turns of `ex` the plan placed, in the order it seats them.
fn placed_of(plan: &DialoguePlan, ex: &[TurnKey]) -> Vec<TurnKey> {
    all_placed(plan)
        .into_iter()
        .filter(|k| k.timeline == ex[0].timeline)
        .collect()
}

/// `placed` is a carving of `ex`: its first and last turn, and between them
/// only its own turns, each once, in order.
fn assert_carving(ex: &[TurnKey], placed: &[TurnKey], what: &str) {
    assert_eq!(
        placed.first(),
        ex.first(),
        "{what}: opens on its first turn"
    );
    assert_eq!(placed.last(), ex.last(), "{what}: closes on its last turn");
    assert!(
        placed.iter().all(|k| ex.contains(k)),
        "{what}: only its own turns: {placed:?}"
    );
    assert!(
        placed.windows(2).all(|w| w[0].index < w[1].index),
        "{what}: in order, each once: {placed:?}"
    );
}

#[test]
fn every_carving_of_a_random_slot_keeps_whole_turns_and_its_caps() {
    let mut rng = StdRng::seed_from_u64(0x5107_F111);
    for case_no in 0..2000 {
        let case = random_case(&mut rng);
        let slot = &case.slot;
        let plan = plan(slot, |k| case.scores[&k], |k| case.tokens[&k], |_| 0.0);
        let tokens_of = |keys: &[TurnKey]| -> usize { keys.iter().map(|k| case.tokens[k]).sum() };
        let check = |ex: &[TurnKey], what: &str| -> Vec<TurnKey> {
            let placed = placed_of(&plan, ex);
            if !placed.is_empty() {
                assert_carving(ex, &placed, &format!("case {case_no} {what}"));
            }
            placed
        };

        // The lineage and other groups' exchanges enter whole or not at all.
        for (i, (ex, _)) in slot.lineage.iter().enumerate() {
            let placed = check(ex, &format!("lineage {i}"));
            assert!(
                placed.is_empty() || placed.len() == ex.len(),
                "case {case_no} lineage {i}"
            );
        }
        for (i, cx) in slot.context.iter().enumerate() {
            let placed = check(&cx.exchange, &format!("context {i}"));
            assert!(
                placed.is_empty() || placed.len() == cx.exchange.len(),
                "case {case_no} context {i}"
            );
        }
        // Each working-set kind stays under its cap. (Only the dialogue's own
        // floors may pass theirs; the working set has none of its own.)
        let mut by_kind: HashMap<Kind, usize> = HashMap::new();
        for (p, wp) in slot.working.iter().enumerate() {
            for (e, ex) in wp.exchanges.iter().enumerate() {
                let placed = check(ex, &format!("working {p}/{e}"));
                *by_kind.entry(wp.kind).or_default() += tokens_of(&placed);
            }
        }
        for (kind, cap) in [
            (Kind::Reading, slot.caps.readings),
            (Kind::Folder, slot.caps.folders),
            (Kind::Guide, slot.caps.guides),
            (Kind::Journal, slot.caps.journals),
        ] {
            let used = by_kind.get(&kind).copied().unwrap_or(0);
            assert!(
                used <= cap,
                "case {case_no}: {kind:?} placed {used} past its cap {cap}"
            );
        }
        // The dialogue's exchanges by birth: the current one and every past
        // one from the window's first, each whole; none before it.
        let n_own = slot.own.len();
        let lineage_tokens: usize = slot.lineage.iter().map(|(ex, _)| tokens_of(ex)).sum();
        let mut left = slot.window.saturating_sub(lineage_tokens);
        let mut from = n_own.saturating_sub(1);
        for e in (0..n_own.saturating_sub(1)).rev() {
            let t = tokens_of(&slot.own[e]);
            if t > left {
                break;
            }
            left -= t;
            from = e;
        }
        for (e, ex) in slot.own.iter().enumerate() {
            let placed = check(ex, &format!("own {e}"));
            let want = if e >= from { ex.len() } else { 0 };
            assert_eq!(
                placed.len(),
                want,
                "case {case_no}: own {e} is whole inside the window and gone outside it"
            );
        }

        // What is placed fits the room, but for what floors carried past it.
        let total = tokens_of(&all_placed(&plan));
        assert!(
            total <= slot.room + plan.over_room,
            "case {case_no}: {total} placed in a room of {} carried {} past it",
            slot.room,
            plan.over_room
        );
    }
}
