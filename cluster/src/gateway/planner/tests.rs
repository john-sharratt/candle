//! The plan over the scenarios of `docs/static_placement.md` §9: a quiet
//! cluster, a chat, chats everywhere, a card alone, and members not fit to
//! host.

use super::*;

const S: Duration = Duration::from_secs(1);

/// BlackWorld (card 1, share 3) and WhiteWorld (card 2, share 1), each with one
/// npcd member: nodes 10 and 20.
fn cluster(bw_latency: u32, ww_latency: u32) -> Vec<CardView> {
    vec![
        CardView {
            key: CardKey::Card(1),
            latency: bw_latency,
            hosts: vec![Host {
                node: 10,
                share: 3,
                eligible: true,
            }],
        },
        CardView {
            key: CardKey::Card(2),
            latency: ww_latency,
            hosts: vec![Host {
                node: 20,
                share: 1,
                eligible: true,
            }],
        },
    ]
}

/// **A quiet cluster lends by share everywhere.** The first plan is epoch 1;
/// the same plan again moves nothing and keeps it.
#[test]
fn a_quiet_cluster_lends_by_share() {
    let t0 = Instant::now();
    let mut p = Planner::default();
    assert!(p.due(t0));
    let first = p.plan(t0, &cluster(0, 0));
    assert_eq!(first.weights, vec![(10, 3), (20, 1)]);
    assert_eq!((first.epoch, first.changed), (1, true));
    assert!(!p.due(t0 + 4 * S));
    let again = p.plan(t0 + 5 * S, &cluster(0, 0));
    assert_eq!((again.epoch, again.changed), (1, false));
    assert_eq!(p.current(), Some((1, vec![(10, 3), (20, 1)])));
}

/// **A chat on BlackWorld moves its cast off at once, and back two minutes
/// after it ends.** The first plan that sees the chat gives BlackWorld nothing;
/// WhiteWorld, free, keeps its share. The chat's last unit is at 30 s: at
/// 149 s BlackWorld is still busy, at 150 s it is free and has its share back.
#[test]
fn a_chat_vacates_its_card_at_once_and_returns_after() {
    let t0 = Instant::now();
    let mut p = Planner::default();
    p.plan(t0, &cluster(0, 0));
    assert!(!p.busy(&CardKey::Card(1), t0));
    let chat = p.plan(t0 + S, &cluster(1, 0));
    assert!(chat.cards[0].busy);
    assert!(p.busy(&CardKey::Card(1), t0 + S));
    assert_eq!(chat.weights, vec![(10, 0), (20, 1)]);
    assert!(chat.changed);
    p.plan(t0 + 30 * S, &cluster(1, 0));
    let lingering = p.plan(t0 + 149 * S, &cluster(0, 0));
    assert_eq!(lingering.weights, vec![(10, 0), (20, 1)]);
    let free = p.plan(t0 + 150 * S, &cluster(0, 0));
    assert!(!free.cards[0].busy);
    assert_eq!(free.weights, vec![(10, 3), (20, 1)]);
}

/// **With chats on every card, every card lends by its share**: the cast runs
/// beneath the conversations rather than nowhere.
#[test]
fn every_card_busy_lends_by_share() {
    let t0 = Instant::now();
    let mut p = Planner::default();
    let ww = p.plan(t0, &cluster(0, 1));
    assert_eq!(ww.weights, vec![(10, 3), (20, 0)]);
    let both = p.plan(t0 + S, &cluster(1, 1));
    assert!(both.cards.iter().all(|c| c.busy));
    assert_eq!(both.weights, vec![(10, 3), (20, 1)]);
}

/// **A card alone lends however busy**: there is no free card to take its
/// cast.
#[test]
fn a_card_alone_lends_however_busy() {
    let t0 = Instant::now();
    let mut p = Planner::default();
    let mut alone = cluster(1, 0);
    alone.truncate(1);
    let plan = p.plan(t0, &alone);
    assert!(plan.cards[0].busy);
    assert_eq!(plan.weights, vec![(10, 3)]);
}

/// **A member not fit to host lends nothing, once that has held a minute** —
/// and a free card unfit to host takes no cast off a busy one. A member with
/// no share lends nothing whatever it is.
#[test]
fn an_unfit_member_lends_nothing_once_settled() {
    let t0 = Instant::now();
    let mut p = Planner::default();
    let mut cards = cluster(1, 0);
    assert_eq!(p.plan(t0, &cards).weights, vec![(10, 0), (20, 1)]);
    cards[1].hosts[0].eligible = false;
    assert_eq!(
        p.plan(t0 + 30 * S, &cards).weights,
        vec![(10, 0), (20, 1)],
        "just seen unfit is not settled"
    );
    assert_eq!(
        p.plan(t0 + 89 * S, &cards).weights,
        vec![(10, 0), (20, 1)],
        "59 s after it was first seen unfit"
    );
    assert_eq!(
        p.plan(t0 + 90 * S, &cards).weights,
        vec![(10, 3), (20, 0)],
        "settled: no free card hosts, so the busy one lends"
    );
    cards[1].hosts[0] = Host {
        node: 20,
        share: 0,
        eligible: true,
    };
    let none = p.plan(t0 + 200 * S, &cards);
    assert_eq!(none.weights, vec![(10, 3), (20, 0)]);
}

/// **A member first seen unfit is lent nothing, and lent to the moment it is
/// fit**: a member joining while it still syncs hosts once it has caught up,
/// with no minute to wait.
#[test]
fn a_member_joining_hosts_once_it_has_caught_up() {
    let t0 = Instant::now();
    let mut p = Planner::default();
    let mut cards = cluster(0, 0);
    cards[1].hosts[0].eligible = false;
    assert_eq!(p.plan(t0, &cards).weights, vec![(10, 3), (20, 0)]);
    cards[1].hosts[0].eligible = true;
    assert_eq!(p.plan(t0 + S, &cards).weights, vec![(10, 3), (20, 1)]);
}
