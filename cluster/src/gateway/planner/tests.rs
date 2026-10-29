//! The plan over the scenarios of `docs/cooperative_admission.md` §4: a quiet
//! cluster, a chat, ingest, every card busy, and members not fit to host.

use super::*;

const S: Duration = Duration::from_secs(1);

/// BlackWorld (card 1, power 300) and WhiteWorld (card 2, power 140), each with
/// one npcd member: nodes 10 and 20.
fn cluster(bw_latency: u32, ww_latency: u32, bw_bulk: u32) -> Vec<CardView> {
    vec![
        CardView {
            key: CardKey::Card(1),
            latency: bw_latency,
            bulk: bw_bulk,
            hosts: vec![Host {
                node: 10,
                weight: 300,
                eligible: true,
            }],
        },
        CardView {
            key: CardKey::Card(2),
            latency: ww_latency,
            bulk: 0,
            hosts: vec![Host {
                node: 20,
                weight: 140,
                eligible: true,
            }],
        },
    ]
}

/// **A quiet cluster lends by power everywhere**, and the floor is fenced on
/// WhiteWorld, the weaker card with no latency on it. The first plan is epoch
/// 1; the same plan again moves nothing and keeps it.
#[test]
fn a_quiet_cluster_lends_by_power() {
    let t0 = Instant::now();
    let mut p = Planner::default();
    assert!(p.due(t0));
    let first = p.plan(t0, &cluster(0, 0, 0));
    assert_eq!(first.weights, vec![(10, 300), (20, 140)]);
    assert_eq!((first.epoch, first.changed), (1, true));
    assert_eq!(first.fences, Fences::from([(CardKey::Card(2), 472)]));
    assert!(!p.due(t0 + 4 * S));
    let again = p.plan(t0 + 5 * S, &cluster(0, 0, 0));
    assert_eq!((again.epoch, again.changed), (1, false));
    assert_eq!(p.current(), Some((1, vec![(10, 300), (20, 140)])));
}

/// **A chat on BlackWorld moves its characters to WhiteWorld a quarter at a
/// time**, every 30 s: 300, 225, 150, 75, then none. WhiteWorld, quiet, lends
/// by its whole power throughout. Ten minutes after the chat's last unit
/// BlackWorld is free again and climbs back the same way.
#[test]
fn a_chat_moves_characters_off_its_card_slowly() {
    let t0 = Instant::now();
    let mut p = Planner::default();
    p.plan(t0, &cluster(0, 0, 0));
    let mut seen = Vec::new();
    for step in 1..=5 {
        let at = t0 + 30 * step * S;
        let plan = p.plan(at, &cluster(1, 0, 0));
        seen.push(plan.weights.clone());
        assert!(plan.cards[0].busy);
        assert_eq!(plan.cards[0].target, 0, "busy and unfenced");
    }
    assert_eq!(
        seen,
        vec![
            vec![(10, 225), (20, 140)],
            vec![(10, 150), (20, 140)],
            vec![(10, 75), (20, 140)],
            vec![(10, 0), (20, 140)],
            vec![(10, 0), (20, 140)],
        ]
    );
    // Between steps nothing moves: 10 s after a step is too soon.
    let early = p.plan(t0 + 160 * S, &cluster(1, 0, 0));
    assert!(!early.changed);
    // The chat ends at 150 s; at 700 s BlackWorld is still within the linger,
    // and at 760 s it is free and starts back.
    let lingering = p.plan(t0 + 700 * S, &cluster(0, 0, 0));
    assert_eq!(lingering.weights, vec![(10, 0), (20, 140)]);
    let free = p.plan(t0 + 760 * S, &cluster(0, 0, 0));
    assert!(!free.cards[0].busy);
    assert_eq!(free.weights, vec![(10, 75), (20, 140)]);
}

/// **A chat on the fenced card keeps its floor share there.** With a chat on
/// WhiteWorld and none on BlackWorld, the fence moves to BlackWorld (220
/// thousandths) and WhiteWorld lends nothing. With chats on both, every card
/// is busy and both lend by their whole power — characters run beneath the
/// conversations everywhere.
#[test]
fn every_card_busy_lends_everywhere() {
    let t0 = Instant::now();
    let mut p = Planner::default();
    let ww = p.plan(t0, &cluster(0, 1, 0));
    assert_eq!(ww.fences, Fences::from([(CardKey::Card(1), 220)]));
    assert_eq!((ww.cards[1].busy, ww.cards[1].target), (true, 0));
    let both = p.plan(t0 + 30 * S, &cluster(1, 1, 0));
    assert!(both.cards.iter().all(|c| c.busy && c.target == QUARTERS));
}

/// **A card alone holds the whole floor, and busy it still lends**: it is the
/// fence, holding 15% of its own power, and as the only hosting card it is
/// every hosting card — so busy with a chat it lends by its whole power, and
/// its characters run beneath the conversation rather than nowhere.
#[test]
fn a_card_alone_lends_however_busy() {
    let t0 = Instant::now();
    let mut p = Planner::default();
    // One card alone: it is the fence, holding the whole floor — 15% of its
    // own power, one quarter — and busy with a chat.
    let alone = vec![CardView {
        key: CardKey::Card(1),
        latency: 0,
        bulk: 0,
        hosts: vec![Host {
            node: 10,
            weight: 300,
            eligible: true,
        }],
    }];
    p.plan(t0, &alone);
    let mut busy = alone.clone();
    busy[0].latency = 1;
    let plan = p.plan(t0 + 30 * S, &busy);
    assert_eq!(plan.fences, Fences::from([(CardKey::Card(1), 150)]));
    assert!(plan.cards[0].busy);
    assert_eq!(plan.cards[0].target, QUARTERS, "every hosting card is busy");
}

/// **Ingest makes a card busy once it has held a minute.** BlackWorld's ingest
/// at 0 s and 30 s moves nothing; at 60 s the card is bulk-busy and its
/// weight starts down.
#[test]
fn ingest_makes_a_card_busy_after_a_minute() {
    let t0 = Instant::now();
    let mut p = Planner::default();
    p.plan(t0, &cluster(0, 0, 4));
    assert!(!p.plan(t0 + 30 * S, &cluster(0, 0, 4)).cards[0].busy);
    let held = p.plan(t0 + 60 * S, &cluster(0, 0, 4));
    assert!(held.cards[0].busy);
    assert_eq!(held.weights, vec![(10, 225), (20, 140)]);
}

/// **A member not fit to host lends nothing** — syncing, lagging, not ready,
/// or taking no lease — and its card's power leaves the floor's sum.
#[test]
fn an_unfit_member_lends_nothing() {
    let t0 = Instant::now();
    let mut p = Planner::default();
    let mut cards = cluster(0, 0, 0);
    cards[1].hosts[0].eligible = false;
    let plan = p.plan(t0, &cards);
    assert_eq!(plan.weights, vec![(10, 300), (20, 0)]);
    // 15% of 300 is 45, which BlackWorld covers alone.
    assert_eq!(plan.fences, Fences::from([(CardKey::Card(1), 150)]));
    cards[1].hosts[0] = Host {
        node: 20,
        weight: 0,
        eligible: true,
    };
    assert_eq!(p.plan(t0 + 5 * S, &cards).weights, vec![(10, 300), (20, 0)]);
}
