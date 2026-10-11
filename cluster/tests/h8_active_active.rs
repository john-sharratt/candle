//! H8 — active-active and cut off (§14.2). Two in-process gateways, three
//! members joined to both, 20 k events and a stream of sticky requests through
//! both gateways. One gateway is killed mid-run and restarted; one member is
//! cut off from both, keeps writing to a conversation another member also
//! writes, then reconnects.
//!
//! No event is lost and duplicates are skipped: every member ends with the
//! same applied set and records, through retransmit and catch-up alone, with
//! no full resync after a process's first admission. A request in flight on
//! the killed gateway fails; every later one succeeds through the other.
//! Claims survive, and the restarted gateway learns the members and their
//! claims again. The cut-off member catches up before it sends what it wrote
//! cut off, under its fixed generation, and on every member the shared
//! conversation holds the group's turns from the cut-off period, then its own.

mod support;

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use cluster::dispatch::Dispatch;
use cluster::gateway::{Hub, Pick};
use cluster::member::{Check, Link};
use cluster::replicate::{MemoryStore, Phase, Store};

use support::{converged, gateway, member_on, slot, until, Node, APP};

const WITHIN: Duration = Duration::from_secs(60);
const EVENTS: usize = 20_000;

fn sticky() -> Dispatch {
    Dispatch::sticky_path("id")
}

/// A sticky request for conversation `key` through gateway `gw`, cluster `c`:
/// the gateway routes it, and the member it reaches claims the key, writes
/// the turn, flushes its events onto the request's socket, and answers.
async fn request(
    gw: &Arc<Hub>,
    c: usize,
    nodes: &[&Node],
    key: &str,
    turn: u32,
) -> Result<u64, String> {
    let Pick::Member(n) = gw.pick(APP, &sticky(), Some(key), None) else {
        return Err("no member to route to".into());
    };
    let node = nodes
        .iter()
        .find(|m| m.node == n)
        .ok_or("an unknown member")?;
    match node.app.claim(key.as_bytes()) {
        Check::Held(h) => return Err(format!("{key} is held by {h}")),
        Check::Took(epoch) => node.store.set_epoch(epoch.ms),
        Check::Mine => {}
    }
    node.store
        .write(format!("turn/{key}/{turn}").as_bytes(), &turn.to_le_bytes());
    node.app.replicator().unwrap().flush(c).await;
    let mut conn = gw.open(APP, n, APP).ok_or("no socket to the member")?;
    let body = format!("{key}/{turn}");
    conn.write_all(body.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    conn.shutdown().await.map_err(|e| e.to_string())?;
    let mut answer = Vec::new();
    conn.read_to_end(&mut answer)
        .await
        .map_err(|e| e.to_string())?;
    if answer != body.as_bytes() {
        return Err("a wrong answer".into());
    }
    Ok(n)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn h8_active_active_and_cut_off() {
    let started = Instant::now();
    let g0 = gateway("g0");
    let g1 = gateway("g1");
    let slots = [slot(&g0), slot(&g1)];
    let a = member_on(&slots, 1, Arc::new(MemoryStore::new(1)));
    let b = member_on(&slots, 2, Arc::new(MemoryStore::new(2)));
    let c = member_on(&slots, 3, Arc::new(MemoryStore::new(3)));
    let nodes = [&a, &b, &c];
    until("ready on both gateways", WITHIN, || {
        nodes.iter().all(|n| n.ready())
            && (1..=3).all(|n| g0.member(APP, n).is_some() && g1.member(APP, n).is_some())
    })
    .await;

    // Events and sticky requests through both gateways.
    let per = EVENTS / 3;
    for k in 0..per {
        for n in nodes {
            n.store.write(format!("e/{}/{k}", n.node).as_bytes(), b"e");
        }
        if k % 500 == 0 {
            let turn = k as u32;
            let (gw, c) = if (k / 500) % 2 == 0 {
                (&g0, 0)
            } else {
                (&g1, 1)
            };
            request(gw, c, &nodes, &format!("conv{}", k % 3), turn)
                .await
                .unwrap();
        }
    }
    let claims_before = g0.status(0).claims.len();
    assert!(claims_before >= 1);

    // A request in flight on g1 when it is killed fails; later ones succeed
    // through g0.
    let Pick::Member(n) = g1.pick(APP, &sticky(), Some("conv0"), None) else {
        panic!("g1 routes nothing");
    };
    let mut in_flight = g1.open(APP, n, APP).unwrap();
    in_flight.write_all(b"half a request").await.unwrap();
    *slots[1].lock().unwrap() = None;
    g1.close_all("killed");
    let mut answer = Vec::new();
    assert!(
        in_flight.read_to_end(&mut answer).await.is_err(),
        "the request in flight on the killed gateway fails"
    );
    for turn in 0..10 {
        request(&g0, 0, &nodes, "conv0", 100_000 + turn)
            .await
            .unwrap();
    }

    // g1 restarts: a new gateway learns the members and their claims again.
    let g1b = gateway("g1");
    *slots[1].lock().unwrap() = Some(g1b.clone());
    until("g1 re-learned the members", WITHIN, || {
        (1..=3).all(|n| g1b.member(APP, n).is_some_and(|m| m.ready()))
    })
    .await;
    until("g1 re-learned the claims", WITHIN, || {
        g1b.status(0).claims.len() >= claims_before
    })
    .await;
    request(&g1b, 1, &nodes, "conv1", 200_000).await.unwrap();

    // C is cut off from both gateways and keeps writing to a conversation A
    // also writes.
    a.store.write(b"turn/shared/0", b"opening");
    until("everyone has the opening", WITHIN, || {
        nodes
            .iter()
            .all(|n| n.store.value(b"turn/shared/0").is_some())
    })
    .await;
    c.reach(0, false);
    c.reach(1, false);
    c.app.close_all("the network went");
    until("c cut off", WITHIN, || {
        c.app.replicator().unwrap().phase() == Phase::CutOff
    })
    .await;
    assert_eq!(c.app.link(), Link::CutOff, "c serves its own listener");
    for k in 0..5u32 {
        c.store
            .write(format!("turn/shared/c{k}").as_bytes(), b"cut off");
        a.store
            .write(format!("turn/shared/a{k}").as_bytes(), b"group");
    }
    c.reach(0, true);
    c.reach(1, true);
    until("c rejoined", WITHIN, || {
        c.app.replicator().unwrap().phase() == Phase::Joined && c.app.link() == Link::Joined
    })
    .await;

    let stores: Vec<&MemoryStore> = nodes.iter().map(|n| &*n.store).collect();
    until("every member holds the same", WITHIN, || {
        converged(&stores) && stores.iter().all(|s| s.applied() == stores[0].applied())
    })
    .await;
    for s in &stores {
        let shared: Vec<Vec<u8>> = s
            .ordered(b"turn/shared/")
            .into_iter()
            .map(|(_, v, _)| v)
            .collect();
        assert_eq!(shared.len(), 11);
        assert_eq!(shared[0], b"opening");
        assert!(shared[1..6].iter().all(|v| v == b"group"), "{shared:?}");
        assert!(shared[6..].iter().all(|v| v == b"cut off"), "{shared:?}");
    }
    // No full resync after each process's first admission: one offer each at
    // most, across both gateways.
    let offers = [&g0, &g1b]
        .iter()
        .flat_map(|g| g.log.read(Some(APP), None, 100_000))
        .filter(|e| e.text.starts_with("offered what it holds"))
        .count();
    assert!(offers <= 3, "{offers} resyncs");
    println!("H8: converged in {:?}", started.elapsed());
}
