//! H6, routing (§14.2): 100 k dispatch decisions with capacity churn — members
//! joining and leaving, readiness, power, idleness and free slots changing,
//! claims taken and released — each checked against a reference model written
//! from §5.2–§5.4 alone. Budget 1 s. (The aggregate half, an 8-member
//! aggregate with one member past its deadline, runs against the gateway's
//! HTTP in `web/tests/h6_aggregate.rs`.)

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use cluster::dispatch::{Dispatch, Merge};
use cluster::frame::{BuildId, Capacity, Epoch, MemberState};
use cluster::gateway::{MemberView, Pick, Registry};
use cluster::machines::MachineTable;

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
    /// Each claim, with the generation of its node it was made in.
    claims: BTreeMap<String, (Epoch, u64)>,
    /// How many times each node has been admitted or has left.
    generation: BTreeMap<u64, u64>,
    /// Each key whose claim ended with its holder's session, and that holder:
    /// the key's next request goes to a member holding what it wrote
    /// (`docs/cooperative_admission.md` §2.2) — here only the holder itself,
    /// since no member reports applying another's records.
    released: BTreeMap<String, u64>,
    rr: usize,
}

impl Model {
    /// A node admitted, or its session over: a new generation, and every claim
    /// it held is released.
    fn turn(&mut self, node: u64) {
        let ended: Vec<String> = self
            .claims
            .keys()
            .filter(|k| self.claim_of(k).is_some_and(|e| e.node == node))
            .cloned()
            .collect();
        for key in ended {
            self.released.insert(key, node);
        }
        *self.generation.entry(node).or_default() += 1;
    }

    /// Where a request for released `key` goes: to its releaser while it is
    /// ready, else as any unit. The release is forgotten once every ready
    /// member holds it — the releaser alone is ready, or nobody is.
    fn after_release(&mut self, key: &str, by: u64) -> Option<u64> {
        let ready: Vec<u64> = self.ready().iter().map(|(n, _)| *n).collect();
        let holding = ready.contains(&by);
        if ready.len() == usize::from(holding) {
            self.released.remove(key);
        }
        if holding {
            Some(by)
        } else {
            self.best()
        }
    }

    /// `key`'s claim, while it stands: a claim lasts as long as the session
    /// of its node it was made in (`docs/static_placement.md` §5.1).
    fn claim_of(&self, key: &str) -> Option<Epoch> {
        let (epoch, made_in) = self.claims.get(key)?;
        (self.generation.get(&epoch.node).copied().unwrap_or(0) == *made_in).then_some(*epoch)
    }

    fn ready(&self) -> Vec<(u64, Cap)> {
        self.members
            .iter()
            .filter_map(|(n, c)| c.filter(|c| c.ready).map(|c| (*n, c)))
            .collect()
    }

    /// Nothing is in flight here — the registry is asked directly, never
    /// through a hub — so every machine is below its `min` and the
    /// deterministic fill (`docs/static_placement.md` §3.2) gives the unit to
    /// the lowest-ranked ready member, whatever its power, idleness and slots
    /// say; of equal ranks, the lowest node.
    fn best(&self) -> Option<u64> {
        self.ready()
            .iter()
            .map(|(n, _)| *n)
            .min_by_key(|n| (rank(*n), *n))
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
                match key.and_then(|k| self.claim_of(k)).map(|e| e.node) {
                    Some(h) if self.members.contains_key(&h) => Some(h),
                    _ => match key.and_then(|k| Some((k, *self.released.get(k)?))) {
                        Some((k, by)) => self.after_release(k, by),
                        None => self.best(),
                    },
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

/// Node `n`'s machine's rank: every rank distinct, and in no order of the
/// nodes.
fn rank(node: u64) -> u64 {
    node * 7 % 13
}

/// The table naming every machine the nodes run on, each ranked by [`rank`].
fn table() -> MachineTable {
    let yaml: String = (1..=12u64)
        .map(|n| format!("m{n}:\n  conversations: {{rank: {}, min: 1}}\n", rank(n)))
        .collect();
    MachineTable::from_yaml(&yaml).unwrap()
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
        silent: false,
        in_flight: 0,
        drawing: 0,
        handed: 0,
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
    registry.set_machines(table());
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
                    // As the hub admits a node: its claims forgotten before
                    // it joins.
                    registry.forget_claims(APP, node);
                    registry.join(view(node, session));
                    model.members.insert(node, None);
                    model.turn(node);
                }
                1 => {
                    let held = registry.member(APP, node).map(|m| m.session);
                    if let Some(s) = held {
                        assert!(registry.leave(APP, node, s));
                        registry.forget_claims(APP, node);
                        model.members.remove(&node);
                        model.turn(node);
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
                    let made_in = model.generation.get(&node).copied().unwrap_or(0);
                    model.claims.insert(key, (epoch, made_in));
                }
                _ => {
                    let key = format!("k{}", rng.random_range(0..20));
                    if let Some(epoch) = model.claim_of(&key) {
                        assert!(registry.claims(APP).unclaim(key.as_bytes(), epoch));
                    }
                    model.claims.remove(&key);
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
