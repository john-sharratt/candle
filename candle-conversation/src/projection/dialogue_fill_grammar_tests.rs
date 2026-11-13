//! **Every carving the slot fill makes reads `U (A U)* A`**
//! (`docs/hit_driven_projection.md` §8.1, §13.2): over random pieces — a
//! lineage, working-set pieces locked and not, the dialogue's own exchanges and
//! other groups' exchanges — at random record sizes, scores, caps and room,
//! each exchange the plan places opens on its request and closes on its
//! answer, and between them holds only whole tool rounds: a call followed by
//! the result of that same call.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::dialogue_fill::{
    plan, ContextExchange, DialoguePlan, DialogueSlot, Placed, WorkingPiece,
};
use super::fill::{Caps, Kind};
use super::{TimelineId, TurnIndex, TurnKey};
use crate::turn_layout::TurnRecord;

/// One exchange of `n` turns on a timeline of its own, so the plan's records
/// can be told apart by exchange.
fn exchange(next_timeline: &mut u64, n: u32) -> Vec<TurnKey> {
    *next_timeline += 1;
    let tl = TimelineId::from_raw(*next_timeline).unwrap();
    (0..n).map(|i| TurnKey::new(tl, TurnIndex(i))).collect()
}

/// A random slot, with each record's tokens and each turn's score.
struct Case {
    slot: DialogueSlot,
    tokens: HashMap<(TurnKey, TurnRecord), usize>,
    scores: HashMap<TurnKey, f32>,
}

fn random_case(rng: &mut StdRng) -> Case {
    let mut tl = 0u64;
    let mut ex = |rng: &mut StdRng, max: u32| exchange(&mut tl, rng.random_range(1..=max));
    let lineage = (0..rng.random_range(0..=3))
        .map(|_| (ex(rng, 4), rng.random_range(0.0..1000.0)))
        .collect();
    let kinds = [Kind::Reading, Kind::Folder, Kind::Guidance];
    let working = (0..rng.random_range(0..=4))
        .map(|_| WorkingPiece {
            kind: kinds[rng.random_range(0..kinds.len())],
            locked: rng.random_bool(0.2),
            score: rng.random_range(0.0..1000.0),
            exchanges: (0..rng.random_range(1..=3)).map(|_| ex(rng, 6)).collect(),
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
    let slot = DialogueSlot {
        room: rng.random_range(0..=600),
        caps: Caps {
            budget: rng.random_range(0..=400),
            folder: rng.random_range(0..=400),
            guidance: rng.random_range(0..=400),
            max_file: rng.random_range(0..=400),
            window: rng.random_range(0..=400),
            layers: vec![rng.random_range(0..=400)],
        },
        recent: rng.random_range(0..=3),
        lineage,
        working,
        own,
        context,
    };
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
            tokens.insert((k, TurnRecord::User), rng.random_range(1..=40));
            tokens.insert((k, TurnRecord::Assistant), rng.random_range(1..=40));
            scores.insert(k, rng.random_range(0.0..1000.0));
        }
    }
    Case {
        slot,
        tokens,
        scores,
    }
}

/// Every record a plan places, in order, a whole turn as its two records.
fn records(placed: &[Placed]) -> Vec<(TurnKey, TurnRecord)> {
    placed
        .iter()
        .flat_map(|&(k, r)| match r {
            None => vec![(k, TurnRecord::User), (k, TurnRecord::Assistant)],
            Some(r) => vec![(k, r)],
        })
        .collect()
}

/// The records of `ex` the plan placed, in the order it seats them.
fn placed_of(plan: &DialoguePlan, ex: &[TurnKey]) -> Vec<(TurnKey, TurnRecord)> {
    let all: Vec<&Placed> = plan
        .lineage
        .iter()
        .chain(plan.working.iter().flatten())
        .chain(plan.own.iter())
        .chain(plan.context.iter().flatten())
        .collect();
    let mine: Vec<Placed> = all
        .into_iter()
        .filter(|(k, _)| k.timeline == ex[0].timeline)
        .copied()
        .collect();
    records(&mine)
}

/// `placed` reads `U (A U)* A` over `ex`: the request, then whole tool rounds
/// — the call of turn `i` and the result in turn `i + 1` — then the answer.
fn assert_grammar(ex: &[TurnKey], placed: &[(TurnKey, TurnRecord)], what: &str) {
    let n = ex.len();
    assert!(
        placed.len() >= 2,
        "{what}: a placed exchange holds a request and an answer: {placed:?}"
    );
    assert_eq!(
        placed[0],
        (ex[0], TurnRecord::User),
        "{what}: opens on its request"
    );
    assert_eq!(
        placed[placed.len() - 1],
        (ex[n - 1], TurnRecord::Assistant),
        "{what}: closes on its answer"
    );
    let middle = &placed[1..placed.len() - 1];
    assert_eq!(
        middle.len() % 2,
        0,
        "{what}: tool rounds come whole: {placed:?}"
    );
    let mut last_call = None;
    for round in middle.chunks(2) {
        let (call, call_rec) = round[0];
        let (result, result_rec) = round[1];
        assert_eq!(call_rec, TurnRecord::Assistant, "{what}: {placed:?}");
        assert_eq!(result_rec, TurnRecord::User, "{what}: {placed:?}");
        assert_eq!(
            result.index.0,
            call.index.0 + 1,
            "{what}: a call is followed by its own result: {placed:?}"
        );
        assert!(
            last_call.is_none_or(|c: u32| call.index.0 > c),
            "{what}: rounds in order: {placed:?}"
        );
        last_call = Some(call.index.0);
    }
}

#[test]
fn every_carving_of_a_random_slot_keeps_the_grammar() {
    let mut rng = StdRng::seed_from_u64(0x5107_F111);
    for case_no in 0..2000 {
        let case = random_case(&mut rng);
        let slot = &case.slot;
        let plan = plan(slot, |k| case.scores[&k], |k, r| case.tokens[&(k, r)]);
        let check = |ex: &[TurnKey], what: &str| -> Vec<(TurnKey, TurnRecord)> {
            let placed = placed_of(&plan, ex);
            if !placed.is_empty() {
                assert_grammar(ex, &placed, &format!("case {case_no} {what}"));
            }
            placed
        };
        let whole = |ex: &[TurnKey]| 2 * ex.len();

        // The lineage and other groups' exchanges enter whole or not at all.
        for (i, (ex, _)) in slot.lineage.iter().enumerate() {
            let placed = check(ex, &format!("lineage {i}"));
            assert!(
                placed.is_empty() || placed.len() == whole(ex),
                "case {case_no} lineage {i}"
            );
        }
        for (i, cx) in slot.context.iter().enumerate() {
            let placed = check(&cx.exchange, &format!("context {i}"));
            assert!(
                placed.is_empty() || placed.len() == whole(&cx.exchange),
                "case {case_no} context {i}"
            );
        }
        // A locked piece goes in whole; any other is carved by the grammar.
        for (p, wp) in slot.working.iter().enumerate() {
            for (e, ex) in wp.exchanges.iter().enumerate() {
                let placed = check(ex, &format!("working {p}/{e}"));
                if wp.locked {
                    assert_eq!(placed.len(), whole(ex), "case {case_no} locked {p}/{e}");
                }
            }
        }
        // The current exchange keeps its request, newest tool round and reply;
        // the dialogue's first request keeps its answer unless it is one of
        // the recent exchanges, which give way.
        let n_own = slot.own.len();
        let recent_from = n_own.saturating_sub(1).saturating_sub(slot.recent);
        for (e, ex) in slot.own.iter().enumerate() {
            let placed = check(ex, &format!("own {e}"));
            if e + 1 == n_own {
                assert!(
                    !placed.is_empty(),
                    "case {case_no}: the current exchange stays"
                );
                if ex.len() >= 2 {
                    let newest = ex.len() - 1;
                    assert!(
                        placed.contains(&(ex[newest - 1], TurnRecord::Assistant))
                            && placed.contains(&(ex[newest], TurnRecord::User)),
                        "case {case_no}: the current exchange keeps its newest tool round"
                    );
                }
            } else if e == 0 && e < recent_from {
                assert!(
                    !placed.is_empty(),
                    "case {case_no}: the first request stays"
                );
            }
        }

        // What is placed fits the room, but for what locks and floors carried
        // past it.
        let total: usize = records(
            &plan
                .lineage
                .iter()
                .chain(plan.working.iter().flatten())
                .chain(plan.own.iter())
                .chain(plan.context.iter().flatten())
                .copied()
                .collect::<Vec<_>>(),
        )
        .iter()
        .map(|kr| case.tokens[kr])
        .sum();
        assert!(
            total <= slot.room + plan.over_room,
            "case {case_no}: {total} placed in a room of {} carried {} past it",
            slot.room,
            plan.over_room
        );
    }
}
