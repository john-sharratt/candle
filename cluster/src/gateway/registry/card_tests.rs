//! Rule 3 over a registry (`docs/static_placement.md` §3.2, §4): every app's
//! units count against the card they land on, a drawing card takes nothing
//! but a draw, a card held for a draw takes only that draw, and a key's holder
//! is never refused.

use super::*;
use crate::dispatch::KeySource;
use crate::frame::{BuildId, Epoch};
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
/// will ever take does not wait for one: nobody takes it, and the route's
/// configured upstream answers.
#[test]
fn a_machine_that_draws_nothing_is_given_no_draw() {
    let mut r = registry("box:\n  conversations: {rank: 1, min: 4}\n");
    r.join(on_card("npcd", 1, "box", 9));
    assert_eq!(
        r.pick("npcd", &Dispatch::Best, Some(512 * 512), None, None),
        Pick::Configured
    );
    assert_eq!(best(&mut r, "npcd"), Pick::Member(1));
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
