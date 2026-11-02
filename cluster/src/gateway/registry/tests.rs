//! Modes, dispatch, claims, UI pinning and the static placement
//! (`docs/static_placement.md`) over a registry of members.

use super::*;
use crate::dispatch::{KeySource, Merge, Route};
use crate::frame::Epoch;
use crate::machines::MachineTable;

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
        card: node,
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
        silent: false,
        in_flight: 0,
        drawing: 0,
        handed: 0,
        closing: false,
    }
}

/// A machine table giving each `(host, rank, min, max)` its conversations, and
/// each a picture preference equal to its rank.
fn table(machines: &[(&str, u32, u32, u32)]) -> MachineTable {
    let yaml: String = machines
        .iter()
        .map(|(host, rank, min, max)| {
            format!(
                "{host}:\n  conversations: {{rank: {rank}, min: {min}, max: {max}}}\n  pictures: {rank}\n"
            )
        })
        .collect();
    MachineTable::from_yaml(&yaml).unwrap()
}

/// **A member whose process is still loading may host no character**, while
/// it stays routable for everything else.
#[test]
fn a_member_still_loading_its_process_hosts_no_character() {
    let mut m = member(1, 1);
    m.capacity = Some(capacity(MemberState::Ready, 1, true, 1));
    assert!(m.may_host());
    m.capacity.as_mut().unwrap().process_loading = true;
    assert!(m.routable(), "still served");
    assert!(!m.may_host(), "but lent no characters");
    m.capacity = Some(capacity(MemberState::Loading, 1, true, 1));
    assert!(!m.may_host(), "and none before it is ready at all");
}

/// **A hole its own origin lacks does not hold a member syncing**: one whose
/// gap-free `ACK` stops at that hole is in sync once it holds up to what is
/// served and misses no more of the origin than the origin does of itself —
/// and still not while it misses more.
#[test]
fn a_hole_the_origin_lacks_does_not_hold_a_member_syncing() {
    let serving: BTreeMap<u64, u64> = [(1, 74_494_768)].into_iter().collect();
    let acks = [(1, 13_240_132)];
    let held_with = |missing| -> BTreeMap<u64, Held> {
        [(
            1,
            Held {
                top: 74_494_768,
                missing,
            },
        )]
        .into_iter()
        .collect()
    };
    let origin_lacks: BTreeMap<u64, u64> = [(1, 1)].into_iter().collect();
    let none = BTreeMap::new();
    assert!(
        !Registry::in_sync(3, &acks, &serving, &held_with(1), &none),
        "the origin is not known to lack it"
    );
    assert!(Registry::in_sync(
        3,
        &acks,
        &serving,
        &held_with(1),
        &origin_lacks
    ));
    assert!(
        !Registry::in_sync(3, &acks, &serving, &held_with(2), &origin_lacks),
        "one more missing than the origin is a gap to fill"
    );
    let short: BTreeMap<u64, Held> = [(
        1,
        Held {
            top: 70_000_000,
            missing: 1,
        },
    )]
    .into_iter()
    .collect();
    assert!(
        !Registry::in_sync(3, &acks, &serving, &short, &origin_lacks),
        "nor while it holds short of what is served"
    );
}

/// A member whose session has ended is routed nothing, even as a canary or
/// when it would be placed first — but only that session: a newer session of
/// the same node is left alone (§5.3).
#[test]
fn a_closing_member_is_routed_nothing() {
    let mut r = three();
    r.closing("zend", 2, 99);
    let picks: Vec<Pick> = (0..6)
        .map(|_| r.pick("zend", &Dispatch::Any, None, None, None))
        .collect();
    assert!(picks.contains(&Pick::Member(2)), "another session's end");
    r.closing("zend", 2, 2);
    for _ in 0..6 {
        assert_ne!(
            r.pick("zend", &Dispatch::Any, None, None, None),
            Pick::Member(2)
        );
        assert_ne!(
            r.pick("zend", &Dispatch::Best, None, None, None),
            Pick::Member(2)
        );
        assert_ne!(
            r.pick_canary("zend", &Dispatch::Best, None, None, None, 2, &mut || true),
            Pick::Member(2),
            "a closing canary is no canary"
        );
    }
    assert!(r.clustered("zend"));
}

/// **A route goes only to the members that declare it**: tables differ, and a
/// request is matched in one that has its route, then dealt in turn among the
/// members declaring it. Where every member declares it, nothing narrows.
#[test]
fn a_route_goes_only_to_the_members_declaring_it() {
    let mut r = three();
    let plain = RouteTable::new(vec![Route::new(&["POST"], "/v1/mcp", Dispatch::Any)]).unwrap();
    let with_memory = RouteTable::new(vec![
        Route::new(&["POST"], "/v1/mcp", Dispatch::Any),
        Route::new(&["POST"], "/v1/mcp/memory", Dispatch::Any),
    ])
    .unwrap();
    r.routes("zend", 1, plain.clone());
    r.routes("zend", 2, with_memory.clone());
    r.routes("zend", 3, with_memory.clone());
    assert_eq!(r.declaring("zend", "POST", "/v1/mcp"), None, "every member");
    let memory = r.declaring("zend", "POST", "/v1/mcp/memory").unwrap();
    assert_eq!(memory, [2, 3]);
    let dealt: Vec<Option<u64>> = (0..4)
        .map(|_| r.round_robin_among("zend", &memory))
        .collect();
    assert!(
        dealt.iter().all(|n| matches!(n, Some(2) | Some(3))),
        "{dealt:?}"
    );
    assert!(dealt.contains(&Some(2)) && dealt.contains(&Some(3)));
    assert!(r
        .table_for("zend", "POST", "/v1/mcp/memory")
        .is_some_and(|t| t.lookup("POST", "/v1/mcp/memory").is_some()));
    r.routes("zend", 2, plain.clone());
    r.routes("zend", 3, plain);
    assert_eq!(
        r.declaring("zend", "POST", "/v1/mcp/memory"),
        Some(Vec::new()),
        "no member declares it: nobody takes it"
    );
    assert_eq!(r.round_robin_among("zend", &[]), None);
}

/// A lagging member is routed nothing and never answers a catch-up, though it
/// stays ready and the app stays clustered (§4.4).
#[test]
fn a_lagging_member_is_routed_nothing() {
    let mut r = three();
    assert!(r.set_lagging("zend", 2, true));
    assert!(!r.set_lagging("zend", 2, true), "no change");
    for _ in 0..6 {
        assert_ne!(
            r.pick("zend", &Dispatch::Any, None, None, None),
            Pick::Member(2)
        );
        assert_ne!(r.answerer("zend", 1), Some(2));
    }
    assert!(r.clustered("zend"));
    r.set_lagging("zend", 2, false);
    let picks: Vec<Pick> = (0..3)
        .map(|_| r.pick("zend", &Dispatch::Any, None, None, None))
        .collect();
    assert!(picks.contains(&Pick::Member(2)));
}

/// **A silent member is passed over while another can take the request** —
/// round robin, `Best`, a declared route, a catch-up's answer, a canary's
/// share — and is routed again once it reports. When every member is
/// silent they are all still served: the gateway's judgement never leaves
/// the app answering nothing (§4.4).
#[test]
fn a_silent_member_is_passed_over_while_another_can_answer() {
    let mut r = three();
    assert!(r.set_silent("zend", 2, true));
    assert!(!r.set_silent("zend", 2, true), "no change");
    for _ in 0..6 {
        assert_ne!(
            r.pick("zend", &Dispatch::Any, None, None, None),
            Pick::Member(2)
        );
        assert_ne!(
            r.pick("zend", &Dispatch::Best, None, None, None),
            Pick::Member(2)
        );
        assert_ne!(r.answerer("zend", 1), Some(2));
        assert_ne!(
            r.pick_canary("zend", &Dispatch::Best, None, None, None, 2, &mut || true),
            Pick::Member(2),
            "a silent canary takes no share"
        );
    }
    assert!(r.clustered("zend"));
    r.set_silent("zend", 2, false);
    let picks: Vec<Pick> = (0..3)
        .map(|_| r.pick("zend", &Dispatch::Any, None, None, None))
        .collect();
    assert!(picks.contains(&Pick::Member(2)), "reporting again");

    for node in [1, 2, 3] {
        r.set_silent("zend", node, true);
    }
    let picks: Vec<Pick> = (0..3)
        .map(|_| r.pick("zend", &Dispatch::Any, None, None, None))
        .collect();
    assert!(
        picks.iter().all(|p| matches!(p, Pick::Member(_))),
        "every member silent: still served, {picks:?}"
    );
    assert!(matches!(
        r.pick("zend", &Dispatch::Best, None, None, None),
        Pick::Member(_)
    ));
}

/// While a candidate certifies, a `Best`, `Any` or new sticky request is its
/// when the share says so, and otherwise goes to the baseline with the
/// candidate left out; a held key and an aggregate are picked as ever; alone,
/// the candidate takes everything (§3.8).
#[test]
fn a_candidate_takes_its_share_and_no_more() {
    let mut r = three();
    // Node 3 is the candidate; node 2, ranked first, is the baseline the
    // placement picks with the candidate left out.
    let mut yes = || true;
    let mut no = || false;
    assert_eq!(
        r.pick_canary("zend", &Dispatch::Best, None, None, None, 3, &mut yes),
        Pick::Member(3)
    );
    assert_eq!(
        r.pick_canary("zend", &Dispatch::Best, None, None, None, 3, &mut no),
        Pick::Member(2)
    );
    for _ in 0..4 {
        assert_ne!(
            r.pick_canary("zend", &Dispatch::Any, None, None, None, 3, &mut no),
            Pick::Member(3)
        );
    }
    let sticky = Dispatch::sticky_path("id");
    assert_eq!(
        r.pick_canary("zend", &sticky, None, Some("new"), None, 3, &mut yes),
        Pick::Member(3)
    );
    r.claims("zend").claim(b"held", Epoch { ms: 1, node: 2 });
    assert_eq!(
        r.pick_canary("zend", &sticky, None, Some("held"), None, 3, &mut yes),
        Pick::Member(2),
        "a held key stays with its holder"
    );
    assert_eq!(
        r.pick_canary(
            "zend",
            &Dispatch::Aggregate(Merge::Envelope),
            None,
            None,
            None,
            3,
            &mut yes
        ),
        Pick::All(vec![1, 2, 3])
    );
    let mut alone = Registry::new();
    alone.set_machines(table(&[("m3", 1, 1, 4)]));
    alone.join(member(3, 3));
    alone.capacity("zend", 3, capacity(MemberState::Ready, 1, true, 1));
    assert_eq!(
        alone.pick_canary("zend", &Dispatch::Best, None, None, None, 3, &mut no),
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

/// Three ready members, on machines `m1`, `m2` and `m3`, which the table ranks
/// 3, 1 and 2: `m2` takes the first unit, then `m3`, then `m1`.
fn three() -> Registry {
    let mut r = Registry::new();
    r.set_machines(table(&[("m1", 3, 1, 4), ("m2", 1, 1, 4), ("m3", 2, 1, 4)]));
    for node in [1, 2, 3] {
        r.join(member(node, node));
        r.capacity("zend", node, capacity(MemberState::Ready, 1, true, 1));
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
    assert_eq!(
        r.pick("zend", &Dispatch::Any, None, None, None),
        Pick::Configured
    );
    r.join(member(1, 1));
    assert!(!r.clustered("zend"), "admitted but loading");
    assert_eq!(
        r.pick("zend", &Dispatch::Any, None, None, None),
        Pick::Configured
    );
    r.capacity("zend", 1, capacity(MemberState::Ready, 1, true, 1));
    assert!(r.clustered("zend"));
    assert_eq!(
        r.pick("zend", &Dispatch::Any, None, None, None),
        Pick::Member(1)
    );
    assert!(r.leave("zend", 1, 1));
    assert_eq!(
        r.pick("zend", &Dispatch::Any, None, None, None),
        Pick::Configured
    );
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
    r.pick("zend", &Dispatch::Best, None, None, None)
}

/// A unit placed and handed to its member: its node, or the pick when it was
/// not placed.
fn placed(r: &mut Registry) -> Result<u64, Pick> {
    match best(r) {
        Pick::Member(n) => {
            r.begin("zend", n, false).expect("admitted");
            Ok(n)
        }
        other => Err(other),
    }
}

/// **Each machine is filled to its `min` in rank order, then the units go
/// round to each one's `max`, then they wait** (`docs/static_placement.md`
/// §3.2). BlackWorld ranks first and is filled to 2 of its 4, WhiteWorld to its
/// 1; then the round robin gives BlackWorld, WhiteWorld — full at 2 — and
/// BlackWorld again, full at 4; the next waits. An answer ending makes room,
/// and the cursor goes on from where it was.
#[test]
fn units_fill_to_each_min_in_rank_order_then_go_round_to_each_max() {
    let mut r = Registry::new();
    r.set_machines(table(&[("BlackWorld", 1, 2, 4), ("WhiteWorld", 2, 1, 2)]));
    for (node, host) in [(1, "BlackWorld"), (2, "WhiteWorld")] {
        r.join(MemberView {
            host: host.into(),
            ..member(node, node)
        });
        r.capacity("zend", node, capacity(MemberState::Ready, 1, true, 1));
    }
    let picks: Vec<Result<u64, Pick>> = (0..7).map(|_| placed(&mut r)).collect();
    assert_eq!(
        picks,
        vec![Ok(1), Ok(1), Ok(2), Ok(1), Ok(2), Ok(1), Err(Pick::Wait)]
    );
    let session = r.member("zend", 2).unwrap().session;
    r.end("zend", 2, session, false);
    assert_eq!(placed(&mut r), Ok(2), "room on WhiteWorld again");
    assert_eq!(placed(&mut r), Err(Pick::Wait));
}

/// **A machine the table does not name is given no work**, whatever else it
/// is: a unit waits for a named machine's room rather than going to it — and
/// when no machine the unit could go to takes conversations at all, it does not
/// wait for room that will never come: nobody takes it, and the route's
/// configured upstream answers.
#[test]
fn a_machine_the_table_does_not_name_takes_nothing() {
    let mut r = Registry::new();
    r.set_machines(table(&[("m1", 1, 1, 1)]));
    for node in [1, 9] {
        r.join(member(node, node));
        r.capacity("zend", node, capacity(MemberState::Ready, 1, true, 1));
    }
    assert_eq!(placed(&mut r), Ok(1));
    assert_eq!(placed(&mut r), Err(Pick::Wait), "m9 is named nowhere");

    let mut unnamed = Registry::new();
    unnamed.set_machines(table(&[("m1", 1, 1, 1)]));
    unnamed.join(member(9, 9));
    unnamed.capacity("zend", 9, capacity(MemberState::Ready, 1, true, 1));
    assert_eq!(placed(&mut unnamed), Err(Pick::Configured));
    let mut empty = Registry::new();
    empty.join(member(1, 1));
    empty.capacity("zend", 1, capacity(MemberState::Ready, 1, true, 1));
    assert_eq!(placed(&mut empty), Err(Pick::Configured), "no table at all");
}

/// **A lagging member with room takes a unit every other member is too full
/// for**: the lag is weighed among the members with room, never before.
#[test]
fn a_lagging_member_with_room_takes_what_the_others_cannot() {
    let mut r = Registry::new();
    r.set_machines(table(&[("m1", 1, 1, 1), ("m2", 2, 1, 2)]));
    for n in [1, 2] {
        r.join(member(n, n));
        r.capacity("zend", n, capacity(MemberState::Ready, 1, true, 1));
    }
    r.set_lagging("zend", 2, true);
    assert_eq!(placed(&mut r), Ok(1), "the current one, while it has room");
    assert_eq!(placed(&mut r), Ok(2), "m1 full: the lagging one");
}

/// **Each app has its own round robin**: npcd's picks do not move zend's
/// cursor.
#[test]
fn each_app_has_its_own_round_robin() {
    let mut r = Registry::new();
    r.set_machines(table(&[("m1", 1, 0, 4), ("m2", 1, 0, 4)]));
    for n in [1, 2] {
        r.join(member(n, n));
        r.capacity("zend", n, capacity(MemberState::Ready, 1, true, 1));
        let mut npcd = member(n, n);
        npcd.app = "npcd".into();
        r.join(npcd);
        r.capacity("npcd", n, capacity(MemberState::Ready, 1, true, 1));
    }
    let zend = |r: &mut Registry| r.pick("zend", &Dispatch::Best, None, None, None);
    let npcd = |r: &mut Registry| r.pick("npcd", &Dispatch::Best, None, None, None);
    assert_eq!(zend(&mut r), Pick::Member(1));
    assert_eq!(
        npcd(&mut r),
        Pick::Member(1),
        "npcd's own cursor starts afresh"
    );
    assert_eq!(npcd(&mut r), Pick::Member(2));
    assert_eq!(
        zend(&mut r),
        Pick::Member(2),
        "zend's cursor, where zend left it"
    );
}

/// **A lagging member is passed over while another can take the unit, a
/// drawing one too** — and with nobody else, the lagging one serves.
#[test]
fn a_lagging_or_drawing_member_is_passed_over_while_another_can_take_it() {
    let mut r = three();
    r.set_lagging("zend", 2, true);
    assert_eq!(placed(&mut r), Ok(3), "m2 ranks first but lags");
    r.begin("zend", 1, true);
    assert_eq!(placed(&mut r), Ok(3), "m1's card draws");
    r.set_lagging("zend", 3, true);
    r.set_lagging("zend", 1, true);
    assert_eq!(placed(&mut r), Ok(2), "every one lags: by rank as ever");
}

/// **An app that runs no model goes to the member with the fewest requests in
/// flight**: the table places the model's work, and an agent's host takes no
/// conversations.
#[test]
fn an_app_that_runs_no_model_goes_to_the_least_busy() {
    let mut r = Registry::new();
    for node in [1, 2] {
        let mut m = member(node, node);
        m.app = "agent".into();
        r.join(m);
        r.capacity("agent", node, capacity(MemberState::Ready, 0, true, 1));
    }
    let pick = |r: &mut Registry| r.pick("agent", &Dispatch::Best, None, None, None);
    assert_eq!(pick(&mut r), Pick::Member(1));
    r.begin("agent", 1, false);
    assert_eq!(pick(&mut r), Pick::Member(2));
    r.begin("agent", 2, false);
    assert_eq!(pick(&mut r), Pick::Member(1), "even: the lower node");
}

#[test]
fn any_goes_round_robin() {
    let mut r = three();
    let picks: Vec<Pick> = (0..4)
        .map(|_| r.pick("zend", &Dispatch::Any, None, None, None))
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

/// A claimed key goes to its holder; an unclaimed one is placed; one held by a
/// member not here is placed too, and that member names the holder.
#[test]
fn a_sticky_key_follows_its_claim() {
    let mut r = three();
    r.claims("zend").claim(b"c-1", Epoch { ms: 1, node: 1 });
    assert_eq!(
        r.pick("zend", &sticky(), None, Some("c-1"), None),
        Pick::Member(1)
    );
    assert_eq!(
        r.pick("zend", &sticky(), None, Some("c-2"), None),
        Pick::Member(2)
    );
    r.claims("zend").claim(b"c-3", Epoch { ms: 1, node: 9 });
    assert_eq!(
        r.pick("zend", &sticky(), None, Some("c-3"), None),
        Pick::Member(2)
    );
    assert_eq!(r.pick("zend", &sticky(), None, None, None), Pick::Member(2));
}

/// **A key let go goes to the first member, by the table, holding what its
/// holder wrote** (`docs/cooperative_admission.md` §2.2): the first-ranked
/// member is skipped while it lacks the last turn, never waited on. Once every
/// member holds it the handover is forgotten and the key is placed as any
/// unit. When nobody holds it — its holder left first — it goes among them
/// all.
#[test]
fn a_released_key_goes_to_a_member_holding_its_last_turn() {
    let mut r = three();
    r.released("zend", b"c", 1, 500);
    r.applied("zend", 2, vec![(1, 100)]);
    r.applied("zend", 3, vec![(1, 500)]);
    // Without the handover, node 2 (ranked first) takes it; it lacks node 1's
    // records past 100, so among 1 (its own writer) and 3, the better ranked.
    assert_eq!(
        r.pick("zend", &sticky(), None, Some("c"), None),
        Pick::Member(3)
    );
    assert_eq!(
        r.pick("zend", &sticky(), None, Some("other"), None),
        Pick::Member(2),
        "a key never let go is not held back"
    );

    r.applied("zend", 2, vec![(1, 500)]);
    assert_eq!(
        r.pick("zend", &sticky(), None, Some("c"), None),
        Pick::Member(2)
    );
    assert!(
        r.apps["zend"].handovers.is_empty(),
        "every member holds it: forgotten"
    );

    // Released by a member gone, before anyone applied its last records.
    r.released("zend", b"d", 9, 10);
    assert_eq!(
        r.pick("zend", &sticky(), None, Some("d"), None),
        Pick::Member(2),
        "nobody holds it: among them all"
    );
}

/// **A release is forgotten once every member holds it, asked for or not**: a
/// conversation never resumed, a picture never edited, leaves nothing behind
/// once its records have reached everyone. One still replicating stands.
#[test]
fn a_release_every_member_holds_is_forgotten_without_being_asked_for() {
    let mut r = three();
    r.released("zend", b"never-asked", 1, 500);
    r.released("zend", b"still-going", 1, 900);
    r.applied("zend", 2, vec![(1, 500)]);
    assert_eq!(r.apps["zend"].handovers.len(), 2, "node 3 holds neither");
    r.applied("zend", 3, vec![(1, 600)]);
    assert!(
        !r.apps["zend"]
            .handovers
            .contains_key(b"never-asked".as_slice()),
        "every member holds it: forgotten"
    );
    assert!(r.apps["zend"]
        .handovers
        .contains_key(b"still-going".as_slice()));
}

/// **A member reporting a key complete may take it, whatever else of its last
/// holder's it lacks** (`COMPLETE`, `docs/cooperative_admission.md` §2.2).
/// Node 2 has applied node 1's records only to 100, far short of the release
/// at 500 — but it holds the conversation `c` names complete, and says so.
/// A report for a key with no release standing is ignored, and a new release
/// forgets the reports made against the one before.
#[test]
fn a_member_holding_a_released_key_complete_may_take_it() {
    let mut r = three();
    r.released("zend", b"c", 1, 500);
    r.applied("zend", 2, vec![(1, 100)]);
    r.applied("zend", 3, vec![(1, 100)]);
    // Node 1 is the only member holding it.
    assert_eq!(
        r.pick("zend", &sticky(), None, Some("c"), None),
        Pick::Member(1)
    );
    r.completed("zend", 2, &[b"c".to_vec(), b"never-released".to_vec()]);
    assert_eq!(
        r.pick("zend", &sticky(), None, Some("c"), None),
        Pick::Member(2),
        "node 2, ranked first, now holds it complete"
    );
    assert!(
        !r.apps["zend"]
            .handovers
            .contains_key(b"never-released".as_slice()),
        "a report for a key nobody let go makes no handover"
    );

    // Released again, through a later record: node 2's report was for the
    // release before.
    r.released("zend", b"c", 1, 900);
    assert_eq!(
        r.pick("zend", &sticky(), None, Some("c"), None),
        Pick::Member(1)
    );
}

/// **A lagging member takes no new conversation while another can, but still
/// holds what it wrote** (`docs/static_placement.md` §3.2). Node 2, ranked
/// first, lags: a new conversation, a key nobody holds and a round-robin
/// request go to node 1, and it answers no catch-up. A key node 1 let go goes
/// back to node 1 while node 2 lacks its last records, and still to node 1 once
/// node 2 holds them — node 2 lags — and to node 2 once it no longer does.
#[test]
fn a_lagging_member_takes_no_new_conversation() {
    let mut r = Registry::new();
    r.set_machines(table(&[("m1", 2, 4, 4), ("m2", 1, 4, 4)]));
    for n in [1, 2] {
        r.join(member(n, n));
        r.capacity("zend", n, capacity(MemberState::Ready, 1, true, 1));
    }
    r.set_lagging("zend", 2, true);
    assert_eq!(best(&mut r), Pick::Member(1), "a new conversation");
    assert_eq!(
        r.pick("zend", &sticky(), None, Some("new"), None),
        Pick::Member(1),
        "a key nobody holds or let go"
    );
    assert_eq!(r.answerer("zend", 3), Some(1), "never the answerer");
    for _ in 0..4 {
        assert_eq!(
            r.pick("zend", &Dispatch::Any, None, None, None),
            Pick::Member(1)
        );
    }

    r.released("zend", b"old", 1, 900);
    r.applied("zend", 2, vec![(1, 500)]);
    assert_eq!(
        r.pick("zend", &sticky(), None, Some("old"), None),
        Pick::Member(1)
    );
    r.applied("zend", 2, vec![(1, 900)]);
    assert_eq!(
        r.pick("zend", &sticky(), None, Some("old"), None),
        Pick::Member(1),
        "node 2 holds it, and lags"
    );
    r.set_lagging("zend", 2, false);
    r.released("zend", b"old", 1, 900);
    assert_eq!(
        r.pick("zend", &sticky(), None, Some("old"), None),
        Pick::Member(2)
    );
}

/// **A lagging member that alone holds a released conversation still takes
/// it**: the lag is weighed after the holders are found, never before.
#[test]
fn a_lagging_holder_alone_still_takes_its_conversation() {
    let mut r = three();
    r.released("zend", b"c", 2, 700);
    r.set_lagging("zend", 2, true);
    assert_eq!(
        r.pick("zend", &sticky(), None, Some("c"), None),
        Pick::Member(2)
    );
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
        r.pick_canary("zend", &sticky(), None, Some("c"), None, 3, &mut take),
        Pick::Member(2),
        "the canary lacks it: picked as ever"
    );
    r.applied("zend", 3, vec![(1, 500)]);
    assert_eq!(
        r.pick_canary("zend", &sticky(), None, Some("c"), None, 3, &mut take),
        Pick::Member(3)
    );
}

/// A page's files come from one member: a new page is pinned round robin, and
/// every request carrying the pin reaches that member while it is ready; a
/// pinned member that leaves is replaced.
#[test]
fn ui_files_come_from_the_pinned_member() {
    let mut r = three();
    let Pick::Pinned(first) = r.pick("zend", &Dispatch::Ui, None, None, None) else {
        panic!()
    };
    for _ in 0..3 {
        assert_eq!(
            r.pick("zend", &Dispatch::Ui, None, None, Some(first)),
            Pick::Member(first)
        );
    }
    r.leave("zend", first, first);
    assert!(matches!(
        r.pick("zend", &Dispatch::Ui, None, None,Some(first)),
        Pick::Pinned(n) if n != first
    ));
}

#[test]
fn an_aggregate_reaches_every_ready_member() {
    let mut r = three();
    r.capacity("zend", 2, capacity(MemberState::Loading, 1, true, 1));
    assert_eq!(
        r.pick(
            "zend",
            &Dispatch::Aggregate(Merge::Envelope),
            None,
            None,
            None
        ),
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
        assert_ne!(
            r.pick("zend", &Dispatch::Any, None, None, None),
            Pick::Member(3)
        );
        assert_ne!(r.answerer("zend", 2), Some(3));
    }
    let sticky = Dispatch::sticky_path("id");
    r.claims("zend").claim(b"mine", Epoch { ms: 1, node: 3 });
    assert_eq!(
        r.pick("zend", &sticky, None, Some("mine"), None),
        Pick::Member(3),
        "a conversation it holds is its, syncing or not"
    );
    let Pick::All(asked) = r.pick("zend", &Dispatch::Aggregate(Merge::Union), None, None, None)
    else {
        panic!("an aggregate asks every loaded member");
    };
    assert!(
        asked.contains(&3),
        "a syncing member is asked: it holds what it wrote cut off"
    );

    let serving: BTreeMap<u64, u64> = [(1, 500), (2, 80)].into_iter().collect();
    let (held, origin_missing) = (BTreeMap::new(), BTreeMap::new());
    let in_sync =
        |acks: &[(u64, u64)]| Registry::in_sync(3, acks, &serving, &held, &origin_missing);
    assert!(!in_sync(&[(1, 400), (2, 80)]), "100 behind node 1");
    assert!(
        in_sync(&[(1, 500 - SYNC_SLACK), (2, 80)]),
        "within the slack"
    );
    assert!(
        !in_sync(&[(1, 500)]),
        "nothing applied of node 2, which the serving members hold"
    );
    assert!(
        in_sync(&[(1, 500), (2, 80)]),
        "a write seen of a member away that nobody serving holds is not waited on"
    );
    assert!(r.set_syncing("zend", 3, false));

    r.set_syncing("zend", 3, true);
    r.set_lagging("zend", 1, true);
    r.set_lagging("zend", 2, true);
    assert_eq!(
        r.pick("zend", &Dispatch::Any, None, None, None),
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

/// Three members whose cards reach 512², 1024² and 2048², on machines that rank
/// them 1, 2 and 3 for conversations and prefer them 3, 2 and 1 for pictures:
/// the largest card draws first.
fn reaching() -> Registry {
    let mut r = Registry::new();
    r.set_machines(
        MachineTable::from_yaml(
            "m1:\n  conversations: {rank: 1, min: 4}\n  pictures: 3\n\
             m2:\n  conversations: {rank: 2, min: 4}\n  pictures: 2\n\
             m3:\n  conversations: {rank: 3, min: 4}\n  pictures: 1\n",
        )
        .unwrap(),
    );
    for (node, reach) in [(1, 512), (2, 1024), (3, 2048)] {
        r.join(member(node, node));
        r.capacity(
            "zend",
            node,
            Capacity {
                guest_reach: reach,
                ..capacity(MemberState::Ready, 1, true, 4)
            },
        );
    }
    r
}

/// **A draw goes only to a member whose card reaches it**, the preferred card
/// of those; a request that draws nothing is placed by rank. Reach is an area:
/// a 768 × 1344 draw is one a 1024² card serves.
#[test]
fn a_draw_goes_only_to_a_member_whose_card_reaches_it() {
    let mut r = Registry::new();
    r.set_machines(table(&[("m1", 1, 4, 4), ("m2", 2, 4, 4)]));
    for (node, reach) in [(1, 512), (2, 1024)] {
        r.join(member(node, node));
        r.capacity(
            "zend",
            node,
            Capacity {
                guest_reach: reach,
                ..capacity(MemberState::Ready, 1, true, 4)
            },
        );
    }
    let mut pick = |draw| r.pick("zend", &Dispatch::Best, draw, None, None);
    assert_eq!(pick(None), Pick::Member(1));
    assert_eq!(
        pick(Some(512 * 512)),
        Pick::Member(1),
        "both reach: preferred"
    );
    assert_eq!(
        pick(Some(768 * 1344)),
        Pick::Member(2),
        "only the larger card"
    );
    assert_eq!(pick(Some(1024 * 1040)), Pick::Member(2), "the furthest");
}

/// A draw of `area` placed and handed to its member: its node.
fn draw(r: &mut Registry, area: u64) -> u64 {
    match r.pick("zend", &Dispatch::Best, Some(area), None, None) {
        Pick::Member(node) => {
            r.begin("zend", node, true).expect("admitted");
            node
        }
        other => panic!("a draw was placed {other:?}"),
    }
}

/// **A card draws one picture at a time, the preferred card first**: a burst
/// takes each card that reaches it once, in order of preference, and the rest
/// wait; a card that finishes its draw is the next one's. A request that draws
/// nothing is placed by rank on a card that is not drawing.
#[test]
fn a_card_draws_one_at_a_time_the_preferred_first() {
    let mut r = reaching();
    let burst: Vec<u64> = (0..3).map(|_| draw(&mut r, 512 * 512)).collect();
    assert_eq!(burst, [3, 2, 1]);
    assert_eq!(
        r.pick("zend", &Dispatch::Best, Some(512 * 512), None, None),
        Pick::Wait,
        "every card draws"
    );
    r.end("zend", 1, 1, true);
    assert_eq!(draw(&mut r, 512 * 512), 1, "the card that finished");

    let mut r = reaching();
    let large: Vec<u64> = (0..2).map(|_| draw(&mut r, 1024 * 1024)).collect();
    assert_eq!(large, [3, 2], "only the cards that reach it");
    assert_eq!(
        r.pick("zend", &Dispatch::Best, None, None, None),
        Pick::Member(1),
        "a request that draws nothing, on the card not drawing"
    );
}

/// **A draw prefers a card with no unit in flight**, whatever the
/// preferences: a draw on a card mid-turn stalls that turn. With every card
/// that reaches it busy, it waits; and a card held for a waiting draw takes no
/// new unit but a draw.
#[test]
fn a_draw_prefers_a_card_with_nothing_in_flight() {
    let mut r = reaching();
    r.begin("zend", 3, false);
    assert_eq!(draw(&mut r, 512 * 512), 2, "the preferred card has a turn");
    let mut r = reaching();
    r.begin("zend", 3, false);
    assert_eq!(
        r.pick("zend", &Dispatch::Best, Some(2048 * 2048), None, None),
        Pick::Wait,
        "only the busy card reaches it"
    );
    assert_eq!(
        r.draw_target("zend", 2048 * 2048, &r.cards()),
        Some(CardKey::Card(3)),
        "the card it waits for"
    );
    r.set_reserved([CardKey::Card(1)].into_iter().collect());
    assert_eq!(
        r.pick("zend", &Dispatch::Best, None, None, None),
        Pick::Member(2),
        "the held card takes no new unit"
    );
    r.begin("zend", 2, false);
    assert_eq!(
        draw(&mut r, 512 * 512),
        1,
        "but takes a draw: the only card with nothing in flight"
    );
}

/// **A member's own `Best` request goes to a peer while one can take it**,
/// and back to the sender only when it is alone; anyone else's is placed as
/// ever.
#[test]
fn a_members_own_request_goes_to_a_peer() {
    let mut r = reaching();
    let mut ask = |sender, canary| {
        r.pick_from(
            "zend",
            &Dispatch::Best,
            None,
            None,
            None,
            PeerAsk { sender, canary },
        )
    };
    assert_eq!(
        ask(Some(1), None),
        Pick::Member(2),
        "the first-ranked member but the sender"
    );
    assert_eq!(
        ask(Some(1), Some(2)),
        Pick::Member(3),
        "and a canary avoided as well"
    );
    assert_eq!(
        ask(None, None),
        Pick::Member(1),
        "a client's request is placed by rank as ever"
    );

    // Only the sender and a canary: the sender, not the unproven build.
    let mut two = Registry::new();
    two.set_machines(table(&[("m1", 2, 4, 4), ("m2", 1, 4, 4)]));
    for node in [1, 2] {
        two.join(member(node, node));
        two.capacity("zend", node, capacity(MemberState::Ready, 1, true, 4));
    }
    let ask = PeerAsk {
        sender: Some(1),
        canary: Some(2),
    };
    assert_eq!(
        two.pick_from("zend", &Dispatch::Best, None, None, None, ask),
        Pick::Member(1)
    );

    let mut alone = Registry::new();
    alone.set_machines(table(&[("m1", 1, 4, 4)]));
    alone.join(member(1, 1));
    alone.capacity("zend", 1, capacity(MemberState::Ready, 100, true, 4));
    let ask = PeerAsk {
        sender: Some(1),
        canary: None,
    };
    assert_eq!(
        alone.pick_from("zend", &Dispatch::Best, None, None, None, ask),
        Pick::Member(1),
        "alone, the sender takes its own request"
    );
}

/// **A draw answered by a session since replaced does not count against the
/// new one**, and a new session starts with none drawing.
#[test]
fn a_replaced_sessions_draws_do_not_count_against_the_new_one() {
    let mut r = reaching();
    assert_eq!(draw(&mut r, 512 * 512), 3);
    r.join(member(3, 30));
    r.capacity(
        "zend",
        3,
        Capacity {
            guest_reach: 2048,
            ..capacity(MemberState::Ready, 100, true, 4)
        },
    );
    assert_eq!(r.member("zend", 3).unwrap().drawing, 0);
    assert_eq!(draw(&mut r, 512 * 512), 3);
    r.end("zend", 3, 3, true);
    assert_eq!(
        r.member("zend", 3).unwrap().drawing,
        1,
        "the old session's end"
    );
}

/// **A draw no card reaches goes to the one that reaches furthest**, whose own
/// refusal names the largest draw the app serves — not to the configured
/// upstream, and not to the first-ranked member, which would name a smaller
/// one.
#[test]
fn a_draw_no_card_reaches_goes_to_the_furthest() {
    let mut r = reaching();
    assert_eq!(
        r.pick("zend", &Dispatch::Best, Some(4096 * 4096), None, None),
        Pick::Member(3)
    );
}

/// **A key nobody holds is placed by reach too; a held key stays with its
/// holder**, whose own refusal answers a draw its card does not reach.
#[test]
fn a_held_key_keeps_its_holder_and_a_free_one_is_placed_by_reach() {
    let mut r = reaching();
    let owner = Dispatch::Owner(KeySource::Body("image".into()));
    assert_eq!(
        r.pick("zend", &owner, Some(2048 * 2048), Some("free"), None),
        Pick::Member(3)
    );
    r.claims("zend").claim(b"held", Epoch { ms: 1, node: 1 });
    assert_eq!(
        r.pick("zend", &owner, Some(2048 * 2048), Some("held"), None),
        Pick::Member(1)
    );
}

/// **A canary is never handed a draw its card does not reach**, even on its
/// share: the draw is placed as ever.
#[test]
fn a_canary_takes_no_draw_its_card_does_not_reach() {
    let mut r = reaching();
    let mut yes = || true;
    assert_eq!(
        r.pick_canary(
            "zend",
            &Dispatch::Best,
            Some(2048 * 2048),
            None,
            None,
            1,
            &mut yes
        ),
        Pick::Member(3)
    );
    assert_eq!(
        r.pick_canary(
            "zend",
            &Dispatch::Best,
            Some(512 * 512),
            None,
            None,
            1,
            &mut yes
        ),
        Pick::Member(1),
        "one it reaches is its share"
    );
}
