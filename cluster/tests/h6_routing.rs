//! H6, routing (§14.2): 100 k dispatch decisions with capacity churn — members
//! joining and leaving, readiness, power, idleness and free slots changing,
//! claims taken and released — each checked against a reference model written
//! from §5.2–§5.4 alone. Budget 1 s. (The aggregate half, an 8-member
//! aggregate with one member past its deadline, runs against the gateway's
//! HTTP in `web/tests/h6_aggregate.rs`.)

use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use cluster::dispatch::{Dispatch, Merge};
use cluster::frame::{BuildId, Capacity, Epoch, MemberState};
use cluster::gateway::{MemberView, Pick, Registry};

const DECISIONS: usize = 100_000;
const BUDGET: Duration = Duration::from_secs(1);
const APP: &str = "zend";

#[derive(Clone, Copy, Debug)]
struct Cap {
    ready: bool,
    power: u32,
    idle: bool,
    slots: u32,
}

/// The reference: what §5.2–§5.4 say each class picks.
#[derive(Default)]
struct Model {
    members: BTreeMap<u64, Option<Cap>>,
    claims: BTreeMap<String, Epoch>,
    rr: usize,
}

impl Model {
    fn ready(&self) -> Vec<(u64, Cap)> {
        self.members
            .iter()
            .filter_map(|(n, c)| c.filter(|c| c.ready).map(|c| (*n, c)))
            .collect()
    }

    /// Nothing is in flight here — the registry is asked directly, never
    /// through a hub — so each share of power is the whole power, and every
    /// card has room: a card with nothing placed on it is never full
    /// (`docs/cooperative_admission.md` §4.2), whatever its slots say. The
    /// highest power; of equals, an idle one; then the lowest node.
    fn best(&self) -> Option<u64> {
        let ready = self.ready();
        let mut pool: Vec<&(u64, Cap)> = ready.iter().collect();
        pool.sort_by_key(|(n, c)| (Reverse(c.power), Reverse(c.idle), *n));
        pool.first().map(|(n, _)| *n)
    }

    fn rr(&mut self) -> Option<u64> {
        let ready = self.ready();
        if ready.is_empty() {
            return None;
        }
        let pick = ready[self.rr % ready.len()].0;
        self.rr += 1;
        Some(pick)
    }

    fn pick(&mut self, dispatch: &Dispatch, key: Option<&str>, pinned: Option<u64>) -> Pick {
        if self.ready().is_empty() {
            return Pick::Configured;
        }
        let is_ready = |m: &Model, n: u64| m.ready().iter().any(|(x, _)| *x == n);
        let node = match dispatch {
            Dispatch::Aggregate(_) => {
                return Pick::All(self.ready().iter().map(|(n, _)| *n).collect())
            }
            Dispatch::Best => self.best(),
            Dispatch::Any => self.rr(),
            // A key held by a member here goes to it, ready or still loading:
            // the conversation is the holder's.
            Dispatch::Sticky(_) | Dispatch::Owner(_) => {
                match key.and_then(|k| self.claims.get(k)).map(|e| e.node) {
                    Some(h) if self.members.contains_key(&h) => Some(h),
                    _ => self.best(),
                }
            }
            Dispatch::Ui => {
                if let Some(p) = pinned.filter(|p| is_ready(self, *p)) {
                    return Pick::Member(p);
                }
                return self.rr().map_or(Pick::Configured, Pick::Pinned);
            }
        };
        node.map_or(Pick::Configured, Pick::Member)
    }
}

fn view(node: u64, session: u64) -> MemberView {
    MemberView {
        node,
        host: format!("m{node}"),
        card: node,
        app: APP.into(),
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
        drawing: 0,
        handed: 0,
        placed: 0,
        closing: false,
    }
}

fn capacity(c: Cap) -> Capacity {
    Capacity {
        state: if c.ready {
            MemberState::Ready
        } else {
            MemberState::Loading
        },
        power: c.power,
        idle: c.idle,
        slots_free: c.slots,
        decodes: 0,
        prefill_backlog: 0,
        vram_free_mib: 0,
        ..Capacity::default()
    }
}

#[test]
fn h6_routing_decisions_match_the_reference() {
    let started = Instant::now();
    let mut rng = StdRng::seed_from_u64(6);
    let mut registry = Registry::new();
    let mut model = Model::default();
    let mut session = 0u64;
    let mut ms = 1u64;
    let dispatches = [
        Dispatch::Best,
        Dispatch::Any,
        Dispatch::Ui,
        Dispatch::sticky_path("id"),
        Dispatch::owner_path("wid"),
        Dispatch::Aggregate(Merge::Envelope),
    ];
    let mut decided = 0;
    let mut by_kind = [0usize; 6];
    while decided < DECISIONS {
        // Churn, about one change per four decisions.
        if rng.random_ratio(1, 4) {
            let node = rng.random_range(1..=12u64);
            match rng.random_range(0..10) {
                0 => {
                    session += 1;
                    registry.join(view(node, session));
                    model.members.insert(node, None);
                }
                1 => {
                    let held = registry.member(APP, node).map(|m| m.session);
                    if let Some(s) = held {
                        assert!(registry.leave(APP, node, s));
                        model.members.remove(&node);
                    }
                }
                2..=6 => {
                    if model.members.contains_key(&node) {
                        let c = Cap {
                            ready: rng.random_ratio(4, 5),
                            power: rng.random_range(1..5) * 100,
                            idle: rng.random_bool(0.6),
                            slots: rng.random_range(0..4),
                        };
                        registry.capacity(APP, node, capacity(c));
                        model.members.insert(node, Some(c));
                    }
                }
                7 | 8 => {
                    let key = format!("k{}", rng.random_range(0..20));
                    ms += 1;
                    let epoch = Epoch { ms, node };
                    registry.claims(APP).claim(key.as_bytes(), epoch);
                    model.claims.insert(key, epoch);
                }
                _ => {
                    let key = format!("k{}", rng.random_range(0..20));
                    if let Some(epoch) = model.claims.get(&key).copied() {
                        assert!(registry.claims(APP).unclaim(key.as_bytes(), epoch));
                        model.claims.remove(&key);
                    }
                }
            }
            continue;
        }
        let k = rng.random_range(0..dispatches.len());
        let dispatch = &dispatches[k];
        let key = format!("k{}", rng.random_range(0..25));
        let key =
            matches!(dispatch, Dispatch::Sticky(_) | Dispatch::Owner(_)).then_some(key.as_str());
        let pinned = matches!(dispatch, Dispatch::Ui)
            .then(|| rng.random_range(0..14u64))
            .filter(|p| *p > 0);
        let got = registry.pick(APP, dispatch, None, key, pinned);
        let want = model.pick(dispatch, key, pinned);
        assert_eq!(
            got, want,
            "decision {decided}: {dispatch:?} key {key:?} pinned {pinned:?}"
        );
        by_kind[k] += 1;
        decided += 1;
    }
    let elapsed = started.elapsed();
    println!("H6 (routing): {DECISIONS} decisions {by_kind:?} in {elapsed:?}");
    assert!(by_kind.iter().all(|n| *n > DECISIONS / 10), "{by_kind:?}");
    assert!(elapsed <= BUDGET, "took {elapsed:?}");
}
