//! H7 (§14.2): consensus under a hostile network, in virtual time.
//!
//! Five members, every frame carried through two gateways — so each arrives
//! twice, independently delayed, dropped or duplicated — 10 k world leases and
//! 1 k decisions; 50 membership changes; one member restarted mid-run and one
//! partitioned into a minority. Asserts: rendezvous moves only keys whose
//! winner changed and shares follow the weights; no world ever has two
//! holders at any instant; no decision ever changes; the partitioned minority
//! holds and decides nothing; once proposers stop competing, every world has a
//! holder and every key a decision. Budget 20 µs of engine time per Paxos
//! round — what the engine costs, whatever number of rounds the backoff makes.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::rc::Rc;
use std::time::{Duration, Instant};

use cluster::consensus::rendezvous::score;
use cluster::consensus::votes::MemoryVotes;
use cluster::consensus::{Config, Millis, Out, Paxos};
use cluster::frame::Frame;

/// SplitMix64: the simulation draws several numbers per frame, and a
/// generator this small is inlined and optimised with the harness, where a
/// cryptographic one would run at the dependency's optimisation level.
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `lo..hi`.
    fn below(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo)
    }

    fn chance(&mut self, p: f64) -> bool {
        let unit = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        unit < p
    }
}

const MEMBERS: usize = 5;
const QUORUM: usize = 3;
const WORLDS: usize = 10_000;
const DECISIONS: usize = 1_000;
const TTL: Duration = Duration::from_secs(20);
/// A holder renews when this little of its term is left.
const RENEW_UNDER: Duration = Duration::from_secs(6);
const WEIGHTS: [f64; MEMBERS] = [1.0, 2.0, 3.0, 4.0, 5.0];
const CHAOS_END: u64 = 30_000;
/// Three terms after the competition stops: the leases still running out
/// after the chaos, and a proposer that lost a duel retrying and perhaps
/// losing once more to a round still in flight (§5.9).
const END: u64 = CHAOS_END + 3 * 20_000;
const STEP: u64 = 100;
const DROP: f64 = 0.02;
const DUPLICATE: f64 = 0.01;
/// Engine time per Paxos round. A budget on the whole run would judge the
/// backoff — how many rounds a duel takes — rather than the engine's cost.
const BUDGET_PER_ROUND: Duration = Duration::from_micros(20);

fn world(w: usize) -> Vec<u8> {
    format!("world/{w}").into_bytes()
}

fn gen_key(d: usize) -> Vec<u8> {
    format!("gen/{d}").into_bytes()
}

struct Sim {
    now: u64,
    members: Vec<Paxos<MemoryVotes>>,
    stores: Vec<MemoryVotes>,
    online: [bool; MEMBERS],
    partitioned: Option<usize>,
    hostile: bool,
    rng: Mix,
    /// Deliveries by time: `(at, seq, recipient, slot)`.
    queue: BinaryHeap<Reverse<(u64, u64, usize, usize)>>,
    /// Frames in flight, shared by their copies; freed slots are reused.
    frames: Vec<Option<Rc<Frame>>>,
    free: Vec<usize>,
    seq: u64,
    /// Per world, who holds it until when, as each holder believes.
    holders: Vec<Vec<(usize, u64)>>,
    decided: HashMap<Vec<u8>, Vec<u8>>,
    rounds: u64,
    violations: Vec<String>,
}

impl Sim {
    fn new() -> Sim {
        let stores: Vec<MemoryVotes> = (0..MEMBERS).map(|_| MemoryVotes::new()).collect();
        let members = (0..MEMBERS).map(|i| Self::open(i, &stores[i], 0)).collect();
        Sim {
            now: 0,
            members,
            stores,
            online: [true; MEMBERS],
            partitioned: None,
            hostile: true,
            rng: Mix(7),
            queue: BinaryHeap::new(),
            frames: Vec::new(),
            free: Vec::new(),
            seq: 0,
            holders: vec![Vec::new(); WORLDS],
            decided: HashMap::new(),
            rounds: 0,
            violations: Vec::new(),
        }
    }

    fn open(i: usize, store: &MemoryVotes, now: u64) -> Paxos<MemoryVotes> {
        let mut p = Paxos::open(
            Config::new(i as u64 + 1),
            store.clone(),
            Millis(now),
            1000 + i as u64 + now,
        )
        .unwrap();
        p.set_quorum(QUORUM);
        p
    }

    fn reachable(&self, i: usize) -> bool {
        self.online[i] && self.partitioned != Some(i)
    }

    /// Carry `frame` from `from` to `to` through both gateways.
    fn carry(&mut self, from: usize, to: usize, frame: &Rc<Frame>) {
        if !self.reachable(from) || !self.reachable(to) {
            return;
        }
        for _gateway in 0..2 {
            let copies = if self.hostile && self.rng.chance(DUPLICATE) {
                2
            } else {
                1
            };
            for _ in 0..copies {
                if self.hostile && self.rng.chance(DROP) {
                    continue;
                }
                let delay = self.rng.below(1, 31);
                let slot = match self.free.pop() {
                    Some(s) => {
                        self.frames[s] = Some(frame.clone());
                        s
                    }
                    None => {
                        self.frames.push(Some(frame.clone()));
                        self.frames.len() - 1
                    }
                };
                self.seq += 1;
                self.queue
                    .push(Reverse((self.now + delay, self.seq, to, slot)));
            }
        }
    }

    fn route(&mut self, from: usize, outs: Vec<Out>) {
        for out in outs {
            match out {
                Out::Broadcast(frame) => {
                    if matches!(frame, Frame::Prepare(_)) {
                        self.rounds += 1;
                    }
                    let frame = Rc::new(frame);
                    for to in 0..MEMBERS {
                        if to != from {
                            self.carry(from, to, &frame);
                        }
                    }
                }
                Out::Reply { to, frame } => self.carry(from, to as usize - 1, &Rc::new(frame)),
                Out::Decided { key, value } => self.on_decided(from, key, value),
                Out::LeaseHeld { key, until } => self.on_held(from, &key, until.0),
                Out::LeaseLost { key } => self.drop_holder(from, &key),
                Out::LeaseBusy { .. } => {}
            }
        }
    }

    fn world_index(key: &[u8]) -> Option<usize> {
        std::str::from_utf8(key)
            .ok()?
            .strip_prefix("world/")?
            .parse()
            .ok()
    }

    fn on_held(&mut self, m: usize, key: &[u8], until: u64) {
        if self.partitioned == Some(m) {
            self.violations
                .push(format!("partitioned member {m} took {key:?}"));
        }
        let w = Self::world_index(key).expect("only worlds are leased");
        let now = self.now;
        let holders = &mut self.holders[w];
        holders.retain(|(h, u)| *u > now && *h != m);
        if let Some((other, u)) = holders.first() {
            self.violations.push(format!(
                "world {w}: member {m} took it at {now} while {other} holds it to {u}"
            ));
        }
        holders.push((m, until));
    }

    fn drop_holder(&mut self, m: usize, key: &[u8]) {
        if let Some(w) = Self::world_index(key) {
            self.holders[w].retain(|(h, _)| *h != m);
        }
    }

    fn on_decided(&mut self, m: usize, key: Vec<u8>, value: Vec<u8>) {
        if self.partitioned == Some(m) {
            self.violations
                .push(format!("partitioned member {m} decided {key:?}"));
        }
        match self.decided.get(&key) {
            Some(first) if *first != value => self
                .violations
                .push(format!("{key:?} changed from {first:?} to {value:?}")),
            Some(_) => {}
            None => {
                self.decided.insert(key.clone(), value.clone());
            }
        }
        // The decision is written as a replicated record: every member learns.
        for i in 0..MEMBERS {
            self.members[i].learn(&key, &value).unwrap();
        }
    }

    fn deliver_until(&mut self, t: u64) {
        while let Some(Reverse((at, _, to, slot))) = self.queue.peek().copied() {
            if at > t {
                break;
            }
            self.queue.pop();
            let frame = self.frames[slot].take().unwrap();
            self.free.push(slot);
            if !self.reachable(to) {
                continue;
            }
            self.now = at;
            let outs = self.members[to].handle(&frame, Millis(at)).unwrap();
            self.route(to, outs);
        }
        self.now = t;
    }

    fn tick(&mut self) {
        for i in 0..MEMBERS {
            let outs = self.members[i].tick(Millis(self.now)).unwrap();
            if self.reachable(i) {
                self.route(i, outs);
            } else {
                // Lapsed leases still lapse for a member nobody can reach.
                for out in outs {
                    if let Out::LeaseLost { key } = out {
                        self.drop_holder(i, &key);
                    }
                }
            }
        }
    }

    /// Rendezvous moved worlds: members stop proposing for worlds they no
    /// longer own.
    fn reassign(&mut self, owners: &[usize]) {
        for (w, owner) in owners.iter().enumerate() {
            let key = world(w);
            for m in 0..MEMBERS {
                if m != *owner {
                    self.members[m].abandon(&key);
                }
            }
        }
    }

    fn owners(&self, scores: &[[f64; MEMBERS]]) -> Vec<usize> {
        scores
            .iter()
            .map(|s| {
                (0..MEMBERS)
                    .filter(|m| self.reachable(*m))
                    .max_by(|a, b| s[*a].total_cmp(&s[*b]))
                    .unwrap()
            })
            .collect()
    }

    /// Each owner takes or renews the worlds rendezvous gives it.
    fn renew(&mut self, owners: &[usize], slice: usize, slices: usize) {
        let now = Millis(self.now);
        for w in (slice..WORLDS).step_by(slices) {
            let m = owners[w];
            let key = world(w);
            let held_long = self.holders[w]
                .iter()
                .any(|(h, u)| *h == m && *u > self.now + RENEW_UNDER.as_millis() as u64);
            if !held_long && !self.members[m].proposing(&key) {
                let outs = self.members[m].acquire(&key, TTL, now).unwrap();
                self.route(m, outs);
            }
        }
    }
}

#[test]
fn h7_consensus() {
    let started = Instant::now();
    let mut sim = Sim::new();

    // Rendezvous scores, computed once.
    let scores: Vec<[f64; MEMBERS]> = (0..WORLDS)
        .map(|w| std::array::from_fn(|m| score(&world(w), m as u64 + 1, WEIGHTS[m])))
        .collect();
    let mut owners = sim.owners(&scores);
    let shares = |owners: &[usize]| {
        let mut c = [0usize; MEMBERS];
        for o in owners {
            c[*o] += 1;
        }
        c
    };
    let all = shares(&owners);
    let total: f64 = WEIGHTS.iter().sum();
    for m in 0..MEMBERS {
        let got = all[m] as f64 / WORLDS as f64;
        assert!(
            (got - WEIGHTS[m] / total).abs() < 0.03,
            "member {m}: share {got}"
        );
    }

    // The decisions: two or three members propose each key at about the same
    // moment, spread over the chaos phase.
    let mut decide_at: Vec<(u64, usize, usize)> = Vec::new();
    for d in 0..DECISIONS {
        let at = sim.rng.below(0, CHAOS_END - 2_000);
        let n = sim.rng.below(2, 4);
        for _ in 0..n {
            let m = sim.rng.below(0, MEMBERS as u64) as usize;
            decide_at.push((at + sim.rng.below(0, 50), d, m));
        }
    }
    decide_at.sort();
    let mut next_decision = 0;
    let mut changes = 0;
    let restart_at = 15_000;
    let partition = (10_000, 20_000, 4usize);

    let slices = 20;
    let mut slice = 0;
    while sim.now < END {
        let t = sim.now + STEP;
        sim.deliver_until(t);

        // The partition and the restart.
        if sim.now == partition.0 || sim.now == partition.1 {
            sim.partitioned = (sim.now == partition.0).then_some(partition.2);
            owners = sim.owners(&scores);
            sim.reassign(&owners);
        }
        if sim.now == restart_at {
            let r = 2;
            sim.members[r] = Sim::open(r, &sim.stores[r], sim.now);
            for (key, value) in sim.decided.clone() {
                sim.members[r].learn(&key, &value).unwrap();
            }
            for w in 0..WORLDS {
                sim.holders[w].retain(|(h, _)| *h != r);
            }
        }

        // Membership changes in the chaos phase: a member leaves or comes
        // back, keeping a quorum online besides the partitioned member.
        if sim.now < CHAOS_END && sim.now.is_multiple_of(500) && changes < 50 {
            let candidates: Vec<usize> = (0..MEMBERS - 1).collect();
            let online = candidates.iter().filter(|m| sim.online[**m]).count();
            let leave = online > QUORUM && sim.rng.chance(0.5) || online == MEMBERS - 1;
            let pool: Vec<usize> = candidates
                .into_iter()
                .filter(|m| sim.online[*m] == leave)
                .collect();
            let m = pool[sim.rng.below(0, pool.len() as u64) as usize];
            {
                let before = owners.clone();
                let leaving = sim.online[m];
                sim.online[m] = !sim.online[m];
                owners = sim.owners(&scores);
                for w in 0..WORLDS {
                    if before[w] != owners[w] {
                        let moved_right = if leaving {
                            before[w] == m
                        } else {
                            owners[w] == m
                        };
                        assert!(
                            moved_right,
                            "world {w} moved {} → {} on member {m}",
                            before[w], owners[w]
                        );
                    }
                }
                sim.reassign(&owners);
                changes += 1;
            }
        }
        // Competition in the chaos phase: members grab worlds they do not own.
        if sim.now < CHAOS_END {
            for _ in 0..20 {
                let m = sim.rng.below(0, MEMBERS as u64) as usize;
                let w = sim.rng.below(0, WORLDS as u64) as usize;
                if sim.reachable(m) && !sim.members[m].proposing(&world(w)) {
                    let outs = sim.members[m]
                        .acquire(&world(w), TTL, Millis(sim.now))
                        .unwrap();
                    sim.route(m, outs);
                }
            }
            while next_decision < decide_at.len() && decide_at[next_decision].0 <= sim.now {
                let (_, d, m) = decide_at[next_decision];
                next_decision += 1;
                if sim.reachable(m) {
                    let value = format!("from {m}").into_bytes();
                    let outs = sim.members[m]
                        .decide(&gen_key(d), value, Millis(sim.now))
                        .unwrap();
                    sim.route(m, outs);
                }
            }
        } else if sim.now == CHAOS_END {
            // Proposers stop competing; the network heals.
            sim.hostile = false;
            sim.online = [true; MEMBERS];
            owners = sim.owners(&scores);
            sim.reassign(&owners);
            for d in 0..DECISIONS {
                if !sim.decided.contains_key(&gen_key(d)) {
                    let outs = sim.members[0]
                        .decide(&gen_key(d), b"late".to_vec(), Millis(sim.now))
                        .unwrap();
                    sim.route(0, outs);
                }
            }
        }

        sim.renew(&owners, slice, slices);
        slice = (slice + 1) % slices;
        sim.tick();
    }
    let elapsed = started.elapsed();
    println!(
        "H7: {} Paxos rounds, {} decisions, {changes} membership changes in {elapsed:?}",
        sim.rounds,
        sim.decided.len()
    );

    assert!(
        sim.violations.is_empty(),
        "{:#?}",
        &sim.violations[..sim.violations.len().min(10)]
    );
    let unheld: Vec<usize> = (0..WORLDS)
        .filter(|w| !sim.holders[*w].iter().any(|(_, u)| *u > sim.now))
        .collect();
    let held = WORLDS - unheld.len();
    let detail: Vec<String> = unheld
        .iter()
        .take(5)
        .map(|w| {
            let o = owners[*w];
            let states: Vec<String> = (0..MEMBERS)
                .map(|m| {
                    format!(
                        "{m}: {:?}",
                        sim.members[m].status(&world(*w), Millis(sim.now))
                    )
                })
                .collect();
            format!(
                "world {w}: owner {o}; tracker {:?}; {states:#?}",
                sim.holders[*w]
            )
        })
        .collect();
    assert_eq!(
        held, WORLDS,
        "every world has a holder once competition stops: {detail:#?}"
    );
    assert_eq!(sim.decided.len(), DECISIONS, "every key is decided");
    assert_eq!(changes, 50);
    assert!(sim.rounds >= 100_000, "{} rounds", sim.rounds);
    let per_round = elapsed / sim.rounds as u32;
    assert!(
        per_round <= BUDGET_PER_ROUND,
        "{per_round:?} a round, {} rounds in {elapsed:?}",
        sim.rounds
    );
}
