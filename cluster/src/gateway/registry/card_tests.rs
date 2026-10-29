//! Rule 3 over a registry (`docs/cooperative_admission.md` §4.2): every app's
//! units count against the card they land on, a drawing card takes nothing
//! but a draw, and the fence holds rows back from latency work.

use super::*;
use crate::frame::{BuildId, Epoch};

/// A ready member of `app` on `card`, with `slots` free and the line `next_us`
/// widening to `wide_us`.
fn on_card(app: &str, node: u64, card: u64, next_us: u32, wide_us: u32) -> MemberView {
    MemberView {
        node,
        host: format!("m{node}"),
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
            power: 100,
            idle: false,
            slots_free: 64,
            decode_step_next_us: next_us,
            decode_step_wide_us: wide_us,
            ..Capacity::default()
        }),
        routes: None,
        high_water: 0,
        applied: Vec::new(),
        lagging: false,
        syncing: false,
        in_flight: 0,
        drawing: 0,
        handed: 0,
        placed: 0,
        closing: false,
    }
}

fn best(r: &mut Registry, app: &str) -> Pick {
    r.pick(app, &Dispatch::Best, None, None, None)
}

/// **One process is one card.** zend's and npcd's members share card 7: what
/// npcd is handed counts against zend's room there, and a card of its own
/// beside them is counted alone. Every unit sums into its card.
#[test]
fn every_app_counts_against_the_card_it_lands_on() {
    let mut r = Registry::new();
    // Card 7 steps 100 ms widening to 400 ms at sixteen more: 21 rows within
    // 500 ms (the 21st steps 480 ms, the 22nd would step 520).
    r.join(on_card("zend", 1, 7, 100_000, 400_000));
    r.join(on_card("npcd", 2, 7, 100_000, 400_000));
    r.join(on_card("npcd", 3, 0, 100_000, 400_000));
    for _ in 0..20 {
        r.begin("npcd", 2, false);
    }
    r.begin("npcd", 3, false);
    let cards = r.cards();
    assert_eq!(
        cards[&CardKey::Card(7)],
        CardLoad {
            in_flight: 20,
            placed: 20,
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
    // One row left on card 7: zend's next unit fits, the one after does not.
    assert_eq!(best(&mut r, "zend"), Pick::Member(1));
    r.begin("zend", 1, false);
    assert_eq!(r.cards()[&CardKey::Card(7)].latency, 1);
    assert_eq!(
        best(&mut r, "zend"),
        Pick::Wait,
        "npcd's units filled the card"
    );
    // A `CAPACITY` reports what was handed: its line now carries them, and
    // nothing handed since is waiting for room.
    let c = r.member("npcd", 2).unwrap().capacity.clone().unwrap();
    r.capacity("npcd", 2, c.clone());
    r.capacity("zend", 1, c);
    assert_eq!(best(&mut r, "zend"), Pick::Member(1));
}

/// **A drawing card takes nothing but a draw**, whatever its width: the draw
/// holds its admission while it runs.
#[test]
fn a_drawing_card_takes_only_draws() {
    let mut r = Registry::new();
    r.join(on_card("npcd", 1, 9, 40_000, 40_000));
    r.join(on_card("zend", 2, 9, 40_000, 40_000));
    r.begin("npcd", 1, true);
    assert_eq!(best(&mut r, "zend"), Pick::Wait);
    let mut reaching = r.member("npcd", 1).unwrap().capacity.clone().unwrap();
    reaching.guest_reach = 1024;
    r.capacity("npcd", 1, reaching);
    assert_eq!(
        r.pick("npcd", &Dispatch::Best, Some(512 * 512), None, None),
        Pick::Member(1),
        "a draw still goes to the card that reaches it"
    );
}

/// **The fence holds rows back from latency work.** A card 21 rows wide
/// fenced at half holds back 11: with 9 units handed it takes one more and
/// then waits; unfenced it would take every row.
#[test]
fn the_fence_holds_rows_back() {
    let mut r = Registry::new();
    r.join(on_card("zend", 1, 7, 100_000, 400_000));
    for _ in 0..9 {
        r.begin("zend", 1, false);
    }
    r.set_fences(Fences::from([(CardKey::Card(7), 500)]));
    assert_eq!(best(&mut r, "zend"), Pick::Member(1));
    r.begin("zend", 1, false);
    assert_eq!(
        best(&mut r, "zend"),
        Pick::Wait,
        "the floor's rows are held"
    );
    r.set_fences(Fences::new());
    assert_eq!(best(&mut r, "zend"), Pick::Member(1));
}

/// **A key's holder is never held back for room** (rule 1 before rule 3): a
/// full card still takes the conversation it holds.
#[test]
fn a_holder_takes_its_key_however_full() {
    let mut r = Registry::new();
    r.join(on_card("zend", 1, 7, 100_000, 400_000));
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
