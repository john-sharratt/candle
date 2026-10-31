//! The queue through the hub (`docs/cooperative_admission.md` §4.3): a unit
//! that fits on no card waits, is placed the moment an answer ends there, and
//! is refused once it has waited the hold.

use std::sync::Arc;
use std::time::Duration;

use super::hub::Hub;
use super::queue::{Placing, HOLD};
use super::registry::{MemberView, Pick};
use crate::dispatch::Dispatch;
use crate::frame::{BuildId, Capacity, MemberState};
use crate::handshake::GroupKey;

/// A hub with one zend member on card 7, whose line — 400 ms widening to
/// 1,600 ms at sixteen more — takes two rows within the 500 ms target.
fn narrow() -> Arc<Hub> {
    let hub = Hub::new(GroupKey::from_bytes([1; 32]), "gw");
    {
        let mut s = hub.state();
        s.registry.join(MemberView {
            node: 1,
            host: "m1".into(),
            card: 7,
            app: "zend".into(),
            build: BuildId {
                count: 1,
                hash: "h".into(),
                local: String::new(),
            },
            weight: 100,
            session: 1,
            repos: Vec::new(),
            capacity: None,
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
        });
        s.registry.capacity(
            "zend",
            1,
            Capacity {
                state: MemberState::Ready,
                power: 100,
                slots_free: 64,
                decode_step_next_us: 400_000,
                decode_step_wide_us: 1_600_000,
                ..Capacity::default()
            },
        );
    }
    hub
}

fn chat() -> Placing {
    Placing {
        req: 1,
        app: "zend".into(),
        dispatch: Dispatch::Best,
        draw: None,
        key: None,
        pinned: None,
        sender: None,
        route: None,
    }
}

/// **A unit that fits nowhere waits, and is placed the moment an answer ends
/// on its card.** Two units fill the card; the third waits until the first is
/// answered, then takes its row.
#[tokio::test(start_paused = true)]
async fn a_waiting_unit_is_placed_when_an_answer_ends() {
    let hub = narrow();
    let first = hub.place(chat()).await.expect("room");
    let second = hub.place(chat()).await.expect("room");
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
        async move { hub.place(chat()).await }
    });
    tokio::time::sleep(Duration::from_secs(5)).await;
    assert!(!waiting.is_finished(), "the card is full");
    drop(first);
    let third = waiting.await.unwrap().expect("released, not refused");
    assert_eq!(third.pick, Pick::Member(1));
    assert!(
        third.load.is_some(),
        "counted against the card like any unit"
    );
}

/// **A unit that waits the whole hold is refused**, and leaves nothing queued.
#[tokio::test(start_paused = true)]
async fn a_unit_held_past_the_hold_is_refused() {
    let hub = narrow();
    let _first = hub.place(chat()).await.expect("room");
    let _second = hub.place(chat()).await.expect("room");
    let started = tokio::time::Instant::now();
    assert!(hub.place(chat()).await.is_none());
    assert_eq!(started.elapsed(), HOLD);
    assert_eq!(hub.waiting(), 0);
}
