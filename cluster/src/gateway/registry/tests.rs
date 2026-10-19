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
        applied: Vec::new(),
        lagging: false,
        syncing: false,
        in_flight: 0,
        handed: 0,
        closing: false,
    }
}

/// A member whose session has ended is routed nothing, even as a canary or
/// when it would rank best — but only that session: a newer session of the
/// same node is left alone (§5.3).
#[test]
fn a_closing_member_is_routed_nothing() {
    let mut r = three();
    r.closing("zend", 2, 99);
    let picks: Vec<Pick> = (0..6)
        .map(|_| r.pick("zend", &Dispatch::Any, None, None))
        .collect();
    assert!(picks.contains(&Pick::Member(2)), "another session's end");
    r.closing("zend", 2, 2);
    for _ in 0..6 {
        assert_ne!(r.pick("zend", &Dispatch::Any, None, None), Pick::Member(2));
        assert_ne!(r.pick("zend", &Dispatch::Best, None, None), Pick::Member(2));
        assert_ne!(
            r.pick_canary("zend", &Dispatch::Best, None, None, 2, &mut || true),
            Pick::Member(2),
            "a closing canary is no canary"
        );
    }
    assert!(r.clustered("zend"));
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
    // Node 3 is the candidate: idle, power 200; node 2, power 300, is the
    // baseline `Best` picks with the candidate left out.
    let mut yes = || true;
    let mut no = || false;
    assert_eq!(
        r.pick_canary("zend", &Dispatch::Best, None, None, 3, &mut yes),
        Pick::Member(3)
    );
    assert_eq!(
        r.pick_canary("zend", &Dispatch::Best, None, None, 3, &mut no),
        Pick::Member(2)
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

/// Three ready members: power 100 (idle, 2 slots), 300 (busy, 5 slots), 200
/// (idle, 1 slot).
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
    // The first to join stands by, whatever its node: 7 was admitted before 3.
    r.join(member(7, 1));
    r.join(member(3, 2));
    assert_eq!(r.standing_by("zend"), Some(7));
    assert_eq!(
        r.standing_by("npcd"),
        None,
        "another app's member is not this one's"
    );
    r.capacity("zend", 7, capacity(MemberState::Ready, 1, true, 1));
    assert_eq!(
        r.standing_by("zend"),
        Some(3),
        "the first is ready: the next joined stands by"
    );
    r.capacity("zend", 3, capacity(MemberState::Ready, 1, true, 1));
    assert_eq!(r.standing_by("zend"), None);
}

fn best(r: &mut Registry) -> Pick {
    r.pick("zend", &Dispatch::Best, None, None)
}

/// **`Best` divides each member's power by its requests in flight** (§5.3):
/// the most powerful takes new work until its share per request falls below
/// another's, busy or not — a member's own background work does not count. A
/// member out of room is passed over while another has room; an equal share
/// goes to an idle member, then the lowest node.
#[test]
fn best_takes_the_largest_share_of_power_per_request() {
    let mut r = three();
    // Shares 100, 300, 200: the busy 300 still has five free slots.
    assert_eq!(best(&mut r), Pick::Member(2));
    let first = r.begin("zend", 2).expect("admitted");
    // 100, 300/2 = 150, 200.
    assert_eq!(best(&mut r), Pick::Member(3));
    r.begin("zend", 3);
    // 100, 150, 200/2 = 100.
    assert_eq!(best(&mut r), Pick::Member(2));
    r.begin("zend", 2);
    r.begin("zend", 2);
    // 100, 300/4 = 75, 100: two idle members tie, and the lower node takes it.
    assert_eq!(best(&mut r), Pick::Member(1));
    // Node 1 busy with no free slot has no room; node 3, idle, still has.
    r.capacity("zend", 1, capacity(MemberState::Ready, 100, false, 0));
    assert_eq!(best(&mut r), Pick::Member(3));
    // With nobody left with room, the largest share of all takes it: node 2
    // has three requests on five slots, so it has room until it holds five.
    r.capacity("zend", 3, capacity(MemberState::Ready, 200, false, 1));
    assert_eq!(best(&mut r), Pick::Member(2), "the only one with room");
    r.begin("zend", 2);
    r.begin("zend", 2);
    assert_eq!(
        r.member("zend", 2).map(|m| m.in_flight),
        Some(5),
        "five begun on node 2"
    );
    // No room anywhere: 100/1, 300/6 = 50, 200/2 = 100 — tied, lower node.
    assert_eq!(best(&mut r), Pick::Member(1));

    // An answer ends its request; one from a session since replaced leaves
    // the newer session's count alone.
    r.end("zend", 2, first);
    assert_eq!(r.member("zend", 2).map(|m| m.in_flight), Some(4));
    r.join(member(2, 9));
    r.capacity("zend", 2, capacity(MemberState::Ready, 300, false, 5));
    r.begin("zend", 2);
    r.end("zend", 2, first);
    assert_eq!(r.member("zend", 2).map(|m| m.in_flight), Some(1));

    // A member in fault mode or upgrading is not ready, and takes nothing.
    r.capacity("zend", 2, capacity(MemberState::Fault, 300, true, 9));
    assert_ne!(best(&mut r), Pick::Member(2));
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
        Pick::Member(2)
    );
    r.claims("zend").claim(b"c-3", Epoch { ms: 1, node: 9 });
    assert_eq!(
        r.pick("zend", &sticky(), Some("c-3"), None),
        Pick::Member(2)
    );
    assert_eq!(r.pick("zend", &sticky(), None, None), Pick::Member(2));
}

/// **A key let go goes to the best member holding what its holder wrote**
/// (`docs/cooperative_admission.md` §2.2): the most powerful member is skipped
/// while it lacks the last turn, never waited on. Once every member holds it the
/// handover is forgotten and the key goes `Best`. When nobody holds it — its
/// holder left first — it goes among them all.
#[test]
fn a_released_key_goes_to_a_member_holding_its_last_turn() {
    let mut r = three();
    r.released("zend", b"c", 1, 500);
    r.applied("zend", 2, vec![(1, 100)]);
    r.applied("zend", 3, vec![(1, 500)]);
    // Without the handover, node 2 (power 300) takes it; it lacks node 1's
    // records past 100, so among 1 (its own writer) and 3 the larger share.
    assert_eq!(r.pick("zend", &sticky(), Some("c"), None), Pick::Member(3));
    assert_eq!(
        r.pick("zend", &sticky(), Some("other"), None),
        Pick::Member(2),
        "a key never let go is not held back"
    );

    r.applied("zend", 2, vec![(1, 500)]);
    assert_eq!(r.pick("zend", &sticky(), Some("c"), None), Pick::Member(2));
    assert!(
        r.apps["zend"].handovers.is_empty(),
        "every member holds it: forgotten"
    );

    // Released by a member gone, before anyone applied its last records.
    r.released("zend", b"d", 9, 10);
    assert_eq!(
        r.pick("zend", &sticky(), Some("d"), None),
        Pick::Member(2),
        "nobody holds it: among them all"
    );
}

/// Set node `n`'s learned rates, its backlog and its decodes on `r`.
fn learned(r: &mut Registry, n: u64, power: u32, backlog: u32, prefill_tok_s: u32, next_us: u32) {
    r.capacity(
        "zend",
        n,
        Capacity {
            prefill_backlog: backlog,
            prefill_tok_s,
            decode_step_next_us: next_us,
            ..capacity(MemberState::Ready, power, true, 4)
        },
    );
}

/// **With every member's rates learned, a turn goes where it finishes
/// soonest** (`docs/cooperative_admission.md` §2.1): the fast prefiller with
/// a backlog against the slower one with none, decided by the arithmetic,
/// not by power. A turn handed to a member counts as backlog until its next
/// `CAPACITY`. One member that has learned nothing yet puts the whole pick
/// back on power.
#[test]
fn a_turn_goes_where_it_finishes_soonest() {
    let mut r = three();
    // Node 1: 4,000 tok/s with 20,000 queued → (20,000 + 2,048) / 4,000 =
    // 5.51 s, then 512 × 20 ms = 10.24 s: 15.75 s.
    learned(&mut r, 1, 100, 20_000, 4_000, 20_000);
    // Node 2: 2,000 tok/s, nothing queued → 1.02 s, then 512 × 40 ms =
    // 20.48 s: 21.50 s — the most power, the slowest reply.
    learned(&mut r, 2, 300, 0, 2_000, 40_000);
    // Node 3: 1,000 tok/s, nothing queued → 2.05 s, then 512 × 25 ms = 12.8 s:
    // 14.85 s.
    learned(&mut r, 3, 200, 0, 1_000, 25_000);
    assert_eq!(best(&mut r), Pick::Member(3));
    // A turn handed to node 3 is 2,048 more tokens there: 16.90 s, and node 1
    // is now the soonest.
    r.begin("zend", 3);
    assert_eq!(best(&mut r), Pick::Member(1));

    // Node 2 has not learned a decode yet: every member goes back to power
    // per request — 100, 300, 200/2.
    learned(&mut r, 2, 300, 0, 2_000, 0);
    assert_eq!(best(&mut r), Pick::Member(2));
}

/// **A burst spills to the slower member once the faster one's queue costs
/// more than the slower one's reply.** Run 21's heavy-decode burst, with the
/// figures each member advertised at 21:33:01Z: BlackWorld 1,196 tok/s and a
/// 28.9 ms step, WhiteWorld 980 tok/s and a 94.9 ms step. A turn on WhiteWorld
/// finishes in 2.09 + 48.57 = 50.66 s. On BlackWorld with `k` turns already
/// handed, it finishes in 1.712 × (k + 1) + 14.78 s, which passes that at
/// k = 20 (50.73 s). So the first 20 go to BlackWorld and the 21st to
/// WhiteWorld.
#[test]
fn a_burst_spills_to_the_slower_member_at_the_crossover() {
    let mut r = Registry::new();
    for n in [1, 2] {
        r.join(member(n, n));
    }
    learned(&mut r, 1, 541, 0, 1_196, 28_862);
    learned(&mut r, 2, 27, 0, 980, 94_871);
    let picks: Vec<Pick> = (0..24)
        .map(|_| {
            let p = best(&mut r);
            if let Pick::Member(n) = p {
                r.begin("zend", n);
            }
            p
        })
        .collect();
    let first_on_2 = picks.iter().position(|p| *p == Pick::Member(2));
    assert_eq!(first_on_2, Some(20), "the 21st turn is the first on node 2");
    assert!(picks[..20].iter().all(|p| *p == Pick::Member(1)));
}

/// The canary takes a key let go only once it holds the last holder's writes.
#[test]
fn a_canary_lacking_the_last_turn_is_not_handed_the_key() {
    let mut r = three();
    r.released("zend", b"c", 1, 500);
    r.applied("zend", 2, vec![(1, 500)]);
    r.applied("zend", 3, vec![(1, 400)]);
    let mut take = || true;
    assert_eq!(
        r.pick_canary("zend", &sticky(), Some("c"), None, 3, &mut take),
        Pick::Member(2),
        "the canary lacks it: picked as ever"
    );
    r.applied("zend", 3, vec![(1, 500)]);
    assert_eq!(
        r.pick_canary("zend", &sticky(), Some("c"), None, 3, &mut take),
        Pick::Member(3)
    );
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

/// **A job picked for an agent counts as handed to it until its next
/// `CAPACITY`**, which reports it running and clears the count.
#[test]
fn a_picked_job_is_handed_until_the_next_capacity() {
    let mut r = Registry::new();
    for node in [1, 2] {
        let mut m = member(node, node);
        m.app = "agent".into();
        m.repos = vec![RepoAd {
            name: "candle".into(),
            origin: "o".into(),
            push: false,
        }];
        r.join(m);
        r.capacity("agent", node, capacity(MemberState::Ready, 0, true, 1));
    }
    let need = Need {
        repo: "candle".into(),
        toolchain: None,
        push: false,
    };
    assert_eq!(r.pick_agent("agent", None, &need, false), Choice::Run(1));
    assert_eq!(r.member("agent", 1).map(|m| m.handed), Some(1));
    assert_eq!(
        r.pick_agent("agent", None, &need, false),
        Choice::Run(2),
        "node 1's one slot was handed out"
    );
    assert_eq!(r.pick_agent("agent", None, &need, false), Choice::Wait);
    r.capacity("agent", 1, capacity(MemberState::Ready, 0, true, 1));
    assert_eq!(r.member("agent", 1).map(|m| m.handed), Some(0));
    assert_eq!(r.pick_agent("agent", None, &need, false), Choice::Run(1));
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

/// **A member away still holds what it wrote.** A catch-up the gateway cannot
/// route is owed while the requester lacks writes seen from an origin whose
/// member has left — `DONE` then made a newcomer join empty — and is not owed
/// once the requester holds them, or when no other origin was ever heard.
#[test]
fn writes_seen_from_a_member_that_left_are_still_owed() {
    let mut r = Registry::new();
    assert!(
        !r.owed("zend", 2, &AppliedSet::default()),
        "a group's first member is owed nothing"
    );
    r.join(member(1, 7));
    r.applied("zend", 1, vec![(1, 500)]);
    assert!(r.leave("zend", 1, 7));
    assert!(r.owed("zend", 2, &AppliedSet::default()));
    let mut held = AppliedSet::default();
    held.insert_range(1, 1, 500);
    assert!(!r.owed("zend", 2, &held));
    assert!(
        !r.owed("zend", 1, &AppliedSet::default()),
        "its own writes are not owed to it"
    );
}

/// **A member joining a serving group serves nothing until it is in sync** —
/// the gateway's judgement from its `ACK`s, not the member's word. Its own
/// conversations still reach it, aggregates still ask it, and with nobody
/// else to serve, it serves.
#[test]
fn a_syncing_member_is_routed_only_its_own() {
    let mut r = three();
    r.applied("zend", 1, vec![(1, 500)]);
    r.set_syncing("zend", 3, true);
    for _ in 0..6 {
        assert_ne!(r.pick("zend", &Dispatch::Any, None, None), Pick::Member(3));
        assert_ne!(r.answerer("zend", 2), Some(3));
    }
    let sticky = Dispatch::sticky_path("id");
    r.claims("zend").claim(b"mine", Epoch { ms: 1, node: 3 });
    assert_eq!(
        r.pick("zend", &sticky, Some("mine"), None),
        Pick::Member(3),
        "a conversation it holds is its, syncing or not"
    );
    let Pick::All(asked) = r.pick("zend", &Dispatch::Aggregate(Merge::Union), None, None) else {
        panic!("an aggregate asks every loaded member");
    };
    assert!(
        asked.contains(&3),
        "a syncing member is asked: it holds what it wrote cut off"
    );

    let serving: BTreeMap<u64, u64> = [(1, 500), (2, 80)].into_iter().collect();
    assert!(
        !Registry::in_sync(3, &[(1, 400), (2, 80)], &serving),
        "100 behind node 1"
    );
    assert!(
        Registry::in_sync(3, &[(1, 500 - SYNC_SLACK), (2, 80)], &serving),
        "within the slack"
    );
    assert!(
        !Registry::in_sync(3, &[(1, 500)], &serving),
        "nothing applied of node 2, which the serving members hold"
    );
    assert!(
        Registry::in_sync(3, &[(1, 500), (2, 80)], &serving),
        "a write seen of a member away that nobody serving holds is not waited on"
    );
    assert!(r.set_syncing("zend", 3, false));

    r.set_syncing("zend", 3, true);
    r.set_lagging("zend", 1, true);
    r.set_lagging("zend", 2, true);
    assert_eq!(
        r.pick("zend", &Dispatch::Any, None, None),
        Pick::Member(3),
        "nobody in sync is serving: the syncing member serves"
    );
}

/// A member joining beside a serving one starts syncing; the first member of a
/// group does not.
#[test]
fn only_a_member_joining_beside_a_serving_one_syncs() {
    let r = three();
    assert!(r.has_routable("zend", 4));
    assert!(!Registry::new().has_routable("zend", 1));
}
