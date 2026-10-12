//! Modes, dispatch, claims and UI pinning over a registry of members.

use super::*;
use crate::dispatch::{KeySource, Merge};
use crate::frame::Epoch;

fn capacity(state: MemberState, power: u32, idle: bool, slots_free: u32) -> Capacity {
    Capacity {
        state,
        power,
        idle,
        slots_free,
        decodes: 0,
        prefill_backlog: 0,
        vram_free_mib: 0,
        cores: 8,
        ..Capacity::default()
    }
}

fn member(node: u64, session: u64) -> MemberView {
    MemberView {
        node,
        host: format!("m{node}"),
        app: "zend".into(),
        build: BuildId {
            count: 1,
            hash: "h".into(),
            local: String::new(),
        },
        weight: 1,
        session,
        repos: Vec::new(),
        capacity: None,
        routes: None,
        high_water: 0,
        lagging: false,
    }
}

/// A lagging member is routed nothing and never answers a catch-up, though it
/// stays ready and the app stays clustered (§4.4).
#[test]
fn a_lagging_member_is_routed_nothing() {
    let mut r = three();
    assert!(r.set_lagging("zend", 2, true));
    assert!(!r.set_lagging("zend", 2, true), "no change");
    for _ in 0..6 {
        assert_ne!(r.pick("zend", &Dispatch::Any, None, None), Pick::Member(2));
        assert_ne!(r.answerer("zend", 1), Some(2));
    }
    assert!(r.clustered("zend"));
    r.set_lagging("zend", 2, false);
    let picks: Vec<Pick> = (0..3)
        .map(|_| r.pick("zend", &Dispatch::Any, None, None))
        .collect();
    assert!(picks.contains(&Pick::Member(2)));
}

/// While a candidate certifies, a `Best`, `Any` or new sticky request is its
/// when the share says so, and otherwise goes to the baseline with the
/// candidate left out; a held key and an aggregate are picked as ever; alone,
/// the candidate takes everything (§3.8).
#[test]
fn a_candidate_takes_its_share_and_no_more() {
    let mut r = three();
    // Node 3 is the candidate: idle, power 200; node 1 is the idle baseline.
    let mut yes = || true;
    let mut no = || false;
    assert_eq!(
        r.pick_canary("zend", &Dispatch::Best, None, None, 3, &mut yes),
        Pick::Member(3)
    );
    assert_eq!(
        r.pick_canary("zend", &Dispatch::Best, None, None, 3, &mut no),
        Pick::Member(1)
    );
    for _ in 0..4 {
        assert_ne!(
            r.pick_canary("zend", &Dispatch::Any, None, None, 3, &mut no),
            Pick::Member(3)
        );
    }
    let sticky = Dispatch::sticky_path("id");
    assert_eq!(
        r.pick_canary("zend", &sticky, Some("new"), None, 3, &mut yes),
        Pick::Member(3)
    );
    r.claims("zend").claim(b"held", Epoch { ms: 1, node: 2 });
    assert_eq!(
        r.pick_canary("zend", &sticky, Some("held"), None, 3, &mut yes),
        Pick::Member(2),
        "a held key stays with its holder"
    );
    assert_eq!(
        r.pick_canary(
            "zend",
            &Dispatch::Aggregate(Merge::Envelope),
            None,
            None,
            3,
            &mut yes
        ),
        Pick::All(vec![1, 2, 3])
    );
    let mut alone = Registry::new();
    alone.join(member(3, 3));
    alone.capacity("zend", 3, capacity(MemberState::Ready, 1, true, 1));
    assert_eq!(
        alone.pick_canary("zend", &Dispatch::Best, None, None, 3, &mut no),
        Pick::Member(3)
    );
}

/// A catch-up's answerer is picked round robin among the other ready members,
/// never the requester; alone, there is none.
#[test]
fn the_answerer_is_another_member_round_robin() {
    let mut r = three();
    let picks: Vec<Option<u64>> = (0..4).map(|_| r.answerer("zend", 2)).collect();
    assert_eq!(picks, vec![Some(1), Some(3), Some(1), Some(3)]);
    let mut alone = Registry::new();
    alone.join(member(1, 1));
    alone.capacity("zend", 1, capacity(MemberState::Ready, 1, true, 1));
    assert_eq!(alone.answerer("zend", 1), None);
}

/// Three ready members: power 100 (idle), 300 (busy), 200 (idle).
fn three() -> Registry {
    let mut r = Registry::new();
    for (node, power, idle, slots) in [(1, 100, true, 2), (2, 300, false, 5), (3, 200, true, 1)] {
        r.join(member(node, node));
        r.capacity(
            "zend",
            node,
            capacity(MemberState::Ready, power, idle, slots),
        );
    }
    r
}

fn sticky() -> Dispatch {
    Dispatch::Sticky(vec![KeySource::Body("conv_id".into())])
}

/// A route with a cluster goes to its configured URL until a member is ready,
/// switches when one is, and reverts when the last leaves.
#[test]
fn the_mode_follows_ready_members() {
    let mut r = Registry::new();
    assert_eq!(r.pick("zend", &Dispatch::Any, None, None), Pick::Configured);
    r.join(member(1, 1));
    assert!(!r.clustered("zend"), "admitted but loading");
    assert_eq!(r.pick("zend", &Dispatch::Any, None, None), Pick::Configured);
    r.capacity("zend", 1, capacity(MemberState::Ready, 1, true, 1));
    assert!(r.clustered("zend"));
    assert_eq!(r.pick("zend", &Dispatch::Any, None, None), Pick::Member(1));
    assert!(r.leave("zend", 1, 1));
    assert_eq!(r.pick("zend", &Dispatch::Any, None, None), Pick::Configured);
    assert!(!r.clustered("npcd"), "another app is untouched");
}

/// A member admitted and not ready stands by for its app's loading page, the
/// lowest node first; once one is ready the app is clustered and the others
/// still stand by; with every member ready, or none, nobody does.
#[test]
fn a_loading_member_stands_by() {
    let mut r = Registry::new();
    assert_eq!(r.standing_by("npcd"), None);
    r.join(member(7, 1));
    r.join(member(3, 1));
    assert_eq!(r.standing_by("zend"), Some(3));
    assert_eq!(
        r.standing_by("npcd"),
        None,
        "another app's member is not this one's"
    );
    r.capacity("zend", 3, capacity(MemberState::Ready, 1, true, 1));
    assert_eq!(r.standing_by("zend"), Some(7));
    r.capacity("zend", 7, capacity(MemberState::Ready, 1, true, 1));
    assert_eq!(r.standing_by("zend"), None);
}

#[test]
fn best_takes_the_highest_idle_power_else_the_most_free_slots() {
    let mut r = three();
    assert_eq!(r.pick("zend", &Dispatch::Best, None, None), Pick::Member(3));
    r.capacity("zend", 1, capacity(MemberState::Ready, 100, false, 2));
    r.capacity("zend", 3, capacity(MemberState::Ready, 200, false, 1));
    assert_eq!(r.pick("zend", &Dispatch::Best, None, None), Pick::Member(2));
    // A member in fault mode or upgrading is not ready, and takes nothing.
    r.capacity("zend", 2, capacity(MemberState::Fault, 300, true, 9));
    assert_eq!(r.pick("zend", &Dispatch::Best, None, None), Pick::Member(1));
}

#[test]
fn any_goes_round_robin() {
    let mut r = three();
    let picks: Vec<Pick> = (0..4)
        .map(|_| r.pick("zend", &Dispatch::Any, None, None))
        .collect();
    assert_eq!(
        picks,
        [
            Pick::Member(1),
            Pick::Member(2),
            Pick::Member(3),
            Pick::Member(1)
        ]
    );
}

/// A claimed key goes to its holder; an unclaimed one goes Best; one held by a
/// member not here goes Best too, and that member names the holder.
#[test]
fn a_sticky_key_follows_its_claim() {
    let mut r = three();
    r.claims("zend").claim(b"c-1", Epoch { ms: 1, node: 1 });
    assert_eq!(
        r.pick("zend", &sticky(), Some("c-1"), None),
        Pick::Member(1)
    );
    assert_eq!(
        r.pick("zend", &sticky(), Some("c-2"), None),
        Pick::Member(3)
    );
    r.claims("zend").claim(b"c-3", Epoch { ms: 1, node: 9 });
    assert_eq!(
        r.pick("zend", &sticky(), Some("c-3"), None),
        Pick::Member(3)
    );
    assert_eq!(r.pick("zend", &sticky(), None, None), Pick::Member(3));
}

/// A page's files come from one member: a new page is pinned round robin, and
/// every request carrying the pin reaches that member while it is ready; a
/// pinned member that leaves is replaced.
#[test]
fn ui_files_come_from_the_pinned_member() {
    let mut r = three();
    let Pick::Pinned(first) = r.pick("zend", &Dispatch::Ui, None, None) else {
        panic!()
    };
    for _ in 0..3 {
        assert_eq!(
            r.pick("zend", &Dispatch::Ui, None, Some(first)),
            Pick::Member(first)
        );
    }
    r.leave("zend", first, first);
    assert!(matches!(
        r.pick("zend", &Dispatch::Ui, None, Some(first)),
        Pick::Pinned(n) if n != first
    ));
}

#[test]
fn an_aggregate_reaches_every_ready_member() {
    let mut r = three();
    r.capacity("zend", 2, capacity(MemberState::Loading, 1, true, 1));
    assert_eq!(
        r.pick("zend", &Dispatch::Aggregate(Merge::Envelope), None, None),
        Pick::All(vec![1, 3])
    );
}

/// A node's older session leaving does not remove its newer one.
#[test]
fn a_replaced_session_leaving_removes_nothing() {
    let mut r = Registry::new();
    r.join(member(5, 1));
    r.join(member(5, 2));
    assert!(!r.leave("zend", 5, 1));
    assert!(r.member("zend", 5).is_some());
    let (epoch, listed) = r.listing("zend");
    assert_eq!((epoch, listed.len()), (2, 1));
}
