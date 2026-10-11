//! H3 — relay fan-out (§14.2): one in-process gateway, four members, 50 k
//! events. Every member gets every other member's events exactly once, none
//! is echoed to its writer, and each origin's events arrive in the order it
//! wrote them; the rate stays above its floor.

mod support;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cluster::consensus::votes::MemoryVotes;
use cluster::replicate::Store;

use support::{converged, gateway, member_over, until, Recording};

/// Events written across the four members.
const EVENTS: usize = 50_000;

/// Delivered events per second: about a third of the first measurement, on the
/// RTX PRO 5000 Blackwell box (195 k/s), so a slower dev machine still clears
/// it.
const FLOOR: f64 = 60_000.0;

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn h3_relay_fan_out() {
    let gw = gateway("gw");
    let stores: Vec<Arc<Recording>> = (1..=4).map(Recording::new).collect();
    let nodes: Vec<_> = stores
        .iter()
        .enumerate()
        .map(|(i, s)| {
            member_over(
                &[&gw],
                i as u64 + 1,
                s.clone(),
                s.inner.clone(),
                MemoryVotes::new(),
            )
        })
        .collect();
    let within = Duration::from_secs(30);
    until("ready", within, || nodes.iter().all(|n| n.ready())).await;
    let inner: Vec<_> = stores.iter().map(|s| &*s.inner).collect();
    until("settled", within, || converged(&inner)).await;
    let before: Vec<u64> = stores.iter().map(|s| s.own_through()).collect();
    for s in &stores {
        s.arrivals.lock().unwrap().clear();
        s.at.lock().unwrap().clear();
    }

    let per = EVENTS / nodes.len();
    let started = Instant::now();
    for k in 0..per {
        for n in &nodes {
            n.store
                .write(format!("e/{k}").as_bytes(), &(k as u32).to_le_bytes());
        }
    }
    let tops: Vec<u64> = stores.iter().map(|s| s.own_through()).collect();
    until("every event delivered", within, || {
        stores.iter().all(|s| {
            let applied = s.applied();
            (0..4).all(|j| applied.through(j as u64 + 1) >= tops[j])
        })
    })
    .await;
    let elapsed = started.elapsed();
    let rate = (EVENTS * 3) as f64 / elapsed.as_secs_f64();
    println!("H3: {EVENTS} events to 3 peers each in {elapsed:?}: {rate:.0} deliveries/s");

    for (i, s) in stores.iter().enumerate() {
        let me = i as u64 + 1;
        let arrivals = s.arrivals.lock().unwrap();
        let mut by_origin: HashMap<u64, Vec<u64>> = HashMap::new();
        for (origin, oseq) in arrivals.iter() {
            assert_ne!(*origin, me, "member {me} was sent its own event {oseq}");
            by_origin.entry(*origin).or_default().push(*oseq);
        }
        for (origin, oseqs) in by_origin {
            let j = origin as usize - 1;
            let want: Vec<u64> = (before[j] + 1..=tops[j]).collect();
            assert_eq!(
                oseqs, want,
                "member {me}: origin {origin}'s events once each, in order"
            );
        }
    }
    assert!(
        rate >= FLOOR,
        "{rate:.0} deliveries/s under the floor {FLOOR}"
    );
}
