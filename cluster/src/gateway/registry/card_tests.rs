//! Rule 3 over a registry (`docs/static_placement.md` §3.2, §4): every app's
//! units count against the card they land on, a drawing card takes nothing
//! but a draw, a card held for a draw takes only that draw, and a key's holder
//! is never refused.

use std::time::{Duration, Instant};

use super::*;
use crate::dispatch::KeySource;
use crate::frame::{BuildId, Epoch, Unclaim};
use crate::gateway::planner::Planner;
use crate::machines::MachineTable;

/// A ready member of `app` on `host`'s card `card`.
fn on_card(app: &str, node: u64, host: &str, card: u64) -> MemberView {
    MemberView {
        node,
        host: host.into(),
        card,
        app: app.into(),
        build: BuildId {
            count: 1,
            hash: "h".into(),
            local: String::new(),
        },
        weight: 100,
        session: node,
        repos: Vec::new(),
        capacity: Some(Capacity {
            state: MemberState::Ready,
            guest_reach: 1024,
            ..Capacity::default()
        }),
        routes: None,
        high_water: 0,
        applied: Vec::new(),
        lagging: false,
        syncing: false,
        silent: false,
        in_flight: 0,
        drawing: 0,
        handed: 0,
        closing: false,
    }
}

fn best(r: &mut Registry, app: &str) -> Pick {
    r.pick(app, &Dispatch::Best, None, None, None)
}

fn registry(table: &str) -> Registry {
    let mut r = Registry::new();
    r.set_machines(MachineTable::from_yaml(table).unwrap());
    r
}

/// **One process is one card.** zend's and npcd's members share card 7: what
/// npcd is handed counts against zend's room there, and a card of its own
/// beside them is counted alone. Every unit sums into its card.
#[test]
fn every_app_counts_against_the_card_it_lands_on() {
    let mut r = registry(
        "box:\n  conversations: {rank: 1, min: 21}\nlone:\n  conversations: {rank: 2, min: 4}\n",
    );
    r.join(on_card("zend", 1, "box", 7));
    r.join(on_card("npcd", 2, "box", 7));
    r.join(on_card("npcd", 3, "lone", 0));
    for _ in 0..20 {
        r.begin("npcd", 2, false);
    }
    r.begin("npcd", 3, false);
    let cards = r.cards();
    assert_eq!(
        cards[&CardKey::Card(7)],
        CardLoad {
            in_flight: 20,
            drawing: 0,
            latency: 0,
        }
    );
    assert_eq!(
        cards[&CardKey::Alone {
            app: "npcd".into(),
            node: 3
        }]
            .in_flight,
        1
    );
    // One unit left on card 7: zend's next fits, the one after does not.
    assert_eq!(best(&mut r, "zend"), Pick::Member(1));
    let session = r.begin("zend", 1, false).unwrap();
    assert_eq!(r.cards()[&CardKey::Card(7)].latency, 1);
    assert_eq!(
        best(&mut r, "zend"),
        Pick::Wait,
        "npcd's units filled the machine"
    );
    r.end("zend", 1, session, false);
    assert_eq!(best(&mut r, "zend"), Pick::Member(1), "an answer made room");
}

/// **A drawing card takes nothing but a draw, and one draw at a time**: the
/// draw holds its admission while it runs.
#[test]
fn a_drawing_card_takes_only_draws() {
    let mut r = registry("box:\n  conversations: {rank: 1, min: 4}\n  pictures: 1\n");
    r.join(on_card("npcd", 1, "box", 9));
    r.join(on_card("zend", 2, "box", 9));
    assert_eq!(
        r.pick("npcd", &Dispatch::Best, Some(512 * 512), None, None),
        Pick::Member(1),
        "a draw goes to the card that reaches it"
    );
    r.begin("npcd", 1, true);
    assert_eq!(best(&mut r, "zend"), Pick::Wait);
    assert_eq!(
        r.pick("npcd", &Dispatch::Best, Some(512 * 512), None, None),
        Pick::Wait,
        "a card draws one picture at a time"
    );
}

/// **A machine that draws nothing is given no draw**, and a draw no machine
/// will ever take does not wait for one: it is refused, unplaced.
#[test]
fn a_machine_that_draws_nothing_is_given_no_draw() {
    let mut r = registry("box:\n  conversations: {rank: 1, min: 4}\n");
    r.join(on_card("npcd", 1, "box", 9));
    assert_eq!(
        r.pick("npcd", &Dispatch::Best, Some(512 * 512), None, None),
        Pick::Unplaced
    );
    assert_eq!(best(&mut r, "npcd"), Pick::Member(1));
}

/// **An idle picture card that lags draws before a busy one that does not**: a
/// turn in flight weighs first, the lag after it.
#[test]
fn an_idle_lagging_card_draws_before_a_busy_current_one() {
    let mut r = registry(
        "bw:\n  conversations: {rank: 1, min: 4}\n  pictures: 1\n\
         ww:\n  conversations: {rank: 2, min: 4}\n  pictures: 2\n",
    );
    r.join(on_card("npcd", 1, "bw", 7));
    r.join(on_card("npcd", 2, "ww", 8));
    r.begin("npcd", 1, false);
    r.set_lagging("npcd", 2, true);
    assert_eq!(
        r.pick("npcd", &Dispatch::Best, Some(512 * 512), None, None),
        Pick::Member(2),
        "bw has a turn in flight"
    );
    r.set_lagging("npcd", 2, false);
    let session = r.member("npcd", 1).unwrap().session;
    r.end("npcd", 1, session, false);
    r.set_lagging("npcd", 1, true);
    assert_eq!(
        r.pick("npcd", &Dispatch::Best, Some(512 * 512), None, None),
        Pick::Member(2),
        "both idle: the one not lagging"
    );
}

/// **An errand waits for no card**: a request routed by its subject that
/// names none and draws nothing — a tool listing, a read of the caller's own
/// — goes at once while a draw holds every card, and is counted as no unit
/// of work; one that names its subject is placed as work.
#[test]
fn an_errand_waits_for_no_card() {
    let mut r = registry("box:\n  conversations: {rank: 1, min: 4}\n  pictures: 1\n");
    r.join(on_card("npcd", 1, "box", 9));
    r.begin("npcd", 1, true);
    let subject = Dispatch::Owner(KeySource::Header("x-subject".into()));
    assert_eq!(best(&mut r, "npcd"), Pick::Wait, "work waits");
    assert_eq!(
        r.pick("npcd", &subject, None, None, None),
        Pick::Member(1),
        "an errand goes at once"
    );
    assert!(is_errand(&subject, None, None));
    assert!(
        !is_errand(&subject, Some("w1"), None),
        "it names its subject"
    );
    assert!(!is_errand(&subject, None, Some(512 * 512)), "it draws");
    assert!(
        !is_errand(&Dispatch::Best, None, None),
        "not routed by a subject"
    );
    assert_eq!(
        r.pick("npcd", &subject, None, Some("w1"), None),
        Pick::Wait,
        "a subject no member holds is placed as work"
    );
}

/// **A card held for a waiting draw takes no new unit but a draw**, and the
/// unit goes to the next machine instead; with no other, it waits.
#[test]
fn a_held_card_takes_only_its_draw() {
    let mut r = registry(
        "bw:\n  conversations: {rank: 1, min: 4}\n  pictures: 1\nww:\n  conversations: {rank: 2, min: 4}\n",
    );
    r.join(on_card("zend", 1, "bw", 7));
    r.join(on_card("npcd", 2, "bw", 7));
    r.join(on_card("zend", 3, "ww", 8));
    r.set_reserved([CardKey::Card(7)].into_iter().collect());
    assert_eq!(best(&mut r, "zend"), Pick::Member(3), "the next machine");
    assert_eq!(
        r.pick("npcd", &Dispatch::Best, Some(512 * 512), None, None),
        Pick::Member(2),
        "the draw it was held for"
    );
    r.leave("zend", 3, 3);
    assert_eq!(best(&mut r, "zend"), Pick::Wait, "no other machine");
}

/// **A worker's delegated task makes its machine busy, and the cast leaves and
/// comes back** (`docs/static_placement.md` §5.1). The task's turns run inside
/// BlackWorld's zend member, so no request of theirs is in flight at the
/// gateway — but the member holds the task's conversation claim, and that
/// makes the card busy: the plan gives BlackWorld nothing while WhiteWorld can
/// host. The claim let go, the card is free 120 s later and its share returns.
#[test]
fn a_delegated_task_makes_its_machine_busy() {
    let mut r = registry(
        "bw:\n  conversations: {rank: 1, min: 4}\n  characters: 3\n\
         ww:\n  conversations: {rank: 2, min: 1}\n  characters: 1\n",
    );
    r.join(on_card("zend", 1, "bw", 7));
    r.join(on_card("npcd", 2, "bw", 7));
    r.join(on_card("npcd", 3, "ww", 8));
    let t0 = Instant::now();
    let mut p = Planner::default();
    assert_eq!(p.plan(t0, &r.card_views()).weights, vec![(2, 3), (3, 1)]);

    let task = Epoch { ms: 1, node: 1 };
    r.claims("zend").claim(b"task-conversation", task);
    assert_eq!(
        r.cards()[&CardKey::Card(7)].in_flight,
        0,
        "nothing at the gateway"
    );
    let working = p.plan(t0 + Duration::from_secs(5), &r.card_views());
    assert_eq!(
        working.weights,
        vec![(2, 0), (3, 1)],
        "the cast leaves BlackWorld"
    );

    let done = t0 + Duration::from_secs(10);
    p.plan(done, &r.card_views());
    r.claims("zend").unclaim(b"task-conversation", task);
    assert_eq!(
        p.plan(done + Duration::from_secs(119), &r.card_views())
            .weights,
        vec![(2, 0), (3, 1)],
        "within two minutes of the task's end"
    );
    assert_eq!(
        p.plan(done + Duration::from_secs(121), &r.card_views())
            .weights,
        vec![(2, 3), (3, 1)],
        "and back"
    );
}

/// **A member that crashed holding a task's claim leaves its card free once it
/// is back**: its re-admission — before the old session's close is seen —
/// forgets its claims, as the hub does before building its `ADMIT`, and each
/// is released through its own records heard; a session that ends takes its
/// claims with it. Kept, they made the card busy for good and the cast never
/// came back.
#[test]
fn a_rejoined_member_s_card_is_free_again() {
    let mut r = registry(
        "bw:\n  conversations: {rank: 1, min: 4}\n  characters: 3\n\
         ww:\n  conversations: {rank: 2, min: 1}\n  characters: 1\n",
    );
    r.join(on_card("zend", 1, "bw", 7));
    r.join(on_card("npcd", 2, "bw", 7));
    r.join(on_card("npcd", 3, "ww", 8));
    let t0 = Instant::now();
    let mut p = Planner::default();
    r.claims("zend")
        .claim(b"task-conversation", Epoch { ms: 1, node: 1 });
    assert_eq!(
        p.plan(t0, &r.card_views()).weights,
        vec![(2, 0), (3, 1)],
        "busy with the task"
    );

    // It crashes and comes back as a new session.
    r.applied("zend", 1, vec![(1, 40)]);
    let released = r.forget_claims("zend", 1);
    assert_eq!(
        released,
        vec![Unclaim {
            key: b"task-conversation".to_vec(),
            epoch: Epoch { ms: 1, node: 1 },
            through: 40,
        }],
        "told to the others, through its records heard"
    );
    r.join(MemberView {
        session: 11,
        ..on_card("zend", 1, "bw", 7)
    });
    assert!(!r.leave("zend", 1, 1), "the old session's close, late");
    let back = t0 + Duration::from_secs(5);
    assert_eq!(p.plan(back, &r.card_views()).weights, vec![(2, 0), (3, 1)]);
    assert_eq!(
        p.plan(back + Duration::from_secs(121), &r.card_views())
            .weights,
        vec![(2, 3), (3, 1)],
        "free 120 s after it came back"
    );

    // A session that ends takes its claims with it.
    r.claims("zend")
        .claim(b"task-conversation", Epoch { ms: 2, node: 1 });
    assert_eq!(r.card_views()[0].latency, 1);
    assert!(r.leave("zend", 1, 11));
    assert_eq!(r.forget_claims("zend", 1).len(), 1);
    assert!(r.claims("zend").is_empty());
}

/// **A canary of an app that runs no model, alone and on a drawing card, makes
/// the unit wait** — the draw ends, and the card takes it — rather than
/// refusing it as though no machine ever would.
#[test]
fn a_lone_drawing_canary_of_a_model_less_app_waits() {
    let mut r = registry("box:\n  conversations: {rank: 1, min: 4}\n  pictures: 1\n");
    r.join(on_card("npcd", 2, "box", 7));
    r.join(on_card("agent", 3, "box", 7));
    r.begin("npcd", 2, true);
    assert_eq!(
        r.pick_canary("agent", &Dispatch::Best, None, None, None, 3, &mut || true),
        Pick::Wait
    );
    let session = r.member("npcd", 2).unwrap().session;
    r.end("npcd", 2, session, true);
    assert_eq!(
        r.pick_canary("agent", &Dispatch::Best, None, None, None, 3, &mut || true),
        Pick::Member(3)
    );
}

/// **A key's holder is never held back for room** (rule 1 before rule 3): a
/// full card still takes the conversation it holds.
#[test]
fn a_holder_takes_its_key_however_full() {
    let mut r = registry("box:\n  conversations: {rank: 1, min: 1}\n");
    r.join(on_card("zend", 1, "box", 7));
    for _ in 0..40 {
        r.begin("zend", 1, false);
    }
    assert_eq!(best(&mut r, "zend"), Pick::Wait);
    r.claims("zend").claim(b"conv", Epoch { ms: 1, node: 1 });
    let sticky = Dispatch::sticky_path("id");
    assert_eq!(
        r.pick("zend", &sticky, None, Some("conv"), None),
        Pick::Member(1)
    );
}
