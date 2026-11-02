//! The queue through the hub (`docs/static_placement.md` §3.4, §4): a unit
//! that fits on no card waits, is placed the moment an answer ends there; a
//! conversation waits as long as its client, any other unit the hold; and a
//! draw waiting for a card holds it from new units.

use std::sync::Arc;
use std::time::Duration;

use super::hub::Hub;
use super::queue::{Placing, HOLD};
use super::registry::{MemberView, Pick};
use crate::dispatch::Dispatch;
use crate::frame::{BuildId, Capacity, MemberState};
use crate::handshake::GroupKey;
use crate::machines::MachineTable;

/// A ready member of `app` on `host`'s card `card`.
fn member(app: &str, node: u64, host: &str, card: u64) -> MemberView {
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
        capacity: None,
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

/// A hub over `table`, with each of `members` ready, reaching `reach`.
fn hub(table: &str, members: Vec<MemberView>, reach: u32) -> Arc<Hub> {
    let hub = Hub::new(GroupKey::from_bytes([1; 32]), "gw");
    hub.set_machines(MachineTable::from_yaml(table).unwrap());
    {
        let mut s = hub.state();
        for m in members {
            let (app, node) = (m.app.clone(), m.node);
            s.registry.join(m);
            s.registry.capacity(
                &app,
                node,
                Capacity {
                    state: MemberState::Ready,
                    guest_reach: reach,
                    ..Capacity::default()
                },
            );
        }
    }
    hub
}

/// One machine that takes two units, its zend member on card 7.
fn narrow() -> Arc<Hub> {
    hub(
        "m1:\n  conversations: {rank: 1, min: 2}\n",
        vec![member("zend", 1, "m1", 7)],
        0,
    )
}

fn unit(app: &str) -> Placing {
    Placing {
        req: 1,
        app: app.into(),
        dispatch: Dispatch::Best,
        draw: None,
        key: None,
        pinned: None,
        sender: None,
        route: None,
    }
}

fn draw(area: u64) -> Placing {
    Placing {
        draw: Some(area),
        ..unit("npcd")
    }
}

/// **A unit that fits nowhere waits, and is placed the moment an answer ends
/// on its card.** Two units fill the machine; the third waits until the first
/// is answered, then takes its place.
#[tokio::test(start_paused = true)]
async fn a_waiting_unit_is_placed_when_an_answer_ends() {
    let hub = narrow();
    let first = hub.place(unit("zend")).await.expect("room");
    let second = hub.place(unit("zend")).await.expect("room");
    assert_eq!(
        (first.pick.clone(), second.pick.clone()),
        (Pick::Member(1), Pick::Member(1))
    );
    assert_eq!(
        hub.pick("zend", &Dispatch::Best, None, None, None).pick,
        Pick::Wait
    );

    let waiting = tokio::spawn({
        let hub = Arc::clone(&hub);
        async move { hub.place(unit("zend")).await }
    });
    tokio::time::sleep(Duration::from_secs(5)).await;
    assert!(!waiting.is_finished(), "the machine is full");
    drop(first);
    let third = waiting.await.unwrap().expect("released, not refused");
    assert_eq!(third.pick, Pick::Member(1));
    assert!(
        third.load.is_some(),
        "counted against the card like any unit"
    );
}

/// **A conversation waits as long as its client does**: past the hold it is
/// still queued, and is placed once there is room.
#[tokio::test(start_paused = true)]
async fn a_conversation_waits_past_the_hold() {
    let hub = narrow();
    let first = hub.place(unit("zend")).await.expect("room");
    let _second = hub.place(unit("zend")).await.expect("room");
    let waiting = tokio::spawn({
        let hub = Arc::clone(&hub);
        async move { hub.place(unit("zend")).await }
    });
    tokio::time::sleep(HOLD * 3).await;
    assert!(!waiting.is_finished(), "still queued past the hold");
    assert_eq!(hub.waiting(), 1);
    drop(first);
    assert_eq!(
        waiting.await.unwrap().expect("placed").pick,
        Pick::Member(1)
    );
}

/// **Any other unit that waits the whole hold is refused**, and leaves nothing
/// queued.
#[tokio::test(start_paused = true)]
async fn a_unit_that_is_no_conversation_is_refused_after_the_hold() {
    let hub = hub(
        "m1:\n  conversations: {rank: 1, min: 1}\n",
        vec![member("npcd", 1, "m1", 7)],
        0,
    );
    let _first = hub.place(unit("npcd")).await.expect("room");
    let started = tokio::time::Instant::now();
    assert!(hub.place(unit("npcd")).await.is_none());
    assert_eq!(started.elapsed(), HOLD);
    assert_eq!(hub.waiting(), 0);
}

/// **A draw only one card reaches waits for that card's turns to end, and
/// holds it meanwhile.** The card's machine takes four units and has one in
/// flight; the draw queues, and the next conversation — which would have fit —
/// waits too, until the turn ends: then the draw goes first, and the
/// conversation after it.
#[tokio::test(start_paused = true)]
async fn a_waiting_draw_holds_its_card() {
    let hub = hub(
        "m1:\n  conversations: {rank: 1, min: 4}\n  pictures: 1\n",
        vec![member("zend", 1, "m1", 7), member("npcd", 2, "m1", 7)],
        1024,
    );
    let turn = hub.place(unit("zend")).await.expect("room");
    let drawing = tokio::spawn({
        let hub = Arc::clone(&hub);
        async move { hub.place(draw(512 * 512)).await }
    });
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(!drawing.is_finished(), "the card has a turn in flight");
    let chat = tokio::spawn({
        let hub = Arc::clone(&hub);
        async move { hub.place(unit("zend")).await }
    });
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(!chat.is_finished(), "the card is held for the draw");
    drop(turn);
    let drew = drawing.await.unwrap().expect("drawn");
    assert_eq!(drew.pick, Pick::Member(2));
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(!chat.is_finished(), "the card is drawing");
    drop(drew);
    assert_eq!(chat.await.unwrap().expect("placed").pick, Pick::Member(1));
}
