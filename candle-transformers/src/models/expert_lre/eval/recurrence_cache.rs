//! The same-layer recurrence statistics that set eviction value, and the cache
//! simulator that tests eviction policies against them.
//!
//! A resident slot's next use is the same layer on the next token, so
//! `RecurrenceModel` learns how often an expert routed at a layer is routed there
//! again, smoothed toward the global prior. `recurrence` measures that
//! stickiness and the recency window that an eviction policy must beat. `Sim`
//! replays the trace through a fixed budget. Its PMI prefetch fills only free
//! room and never evicts to make space, and `Evict::Recurrence` moves an expert's
//! last-used clock by its keep value, so a scale of zero is exactly LRU.

use super::*;

/// Measure same-layer token-to-token recurrence structure — the statistic that
/// governs eviction value (a `(layer, expert)` slot's next use is the same
/// layer on the next token).  Reports how predictable "needed next token" is
/// from the current/recent active sets, versus the base rate and versus LRU.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn recurrence() {
    let Some(records) = load_fixture() else {
        println!("\n[recurrence] no fixture; capture first (see eval::report).");
        return;
    };
    let trace = Trace::from_records(&records);
    const DECODE_MAX: usize = 16;
    let seqs = per_layer_token_sequences(&trace, DECODE_MAX);

    // Adjacent-token statistics, aggregated over all layers.
    // sticky: e ∈ S_next given e ∈ S_now.  arrival: e ∈ S_next given e ∉ S_now.
    let (mut sticky_hit, mut sticky_tot) = (0u64, 0u64);
    let (mut arrive_hit, mut arrive_tot) = (0u64, 0u64);
    let (mut next_sz, mut next_n) = (0u64, 0u64);
    // Of the experts needed next token, what share were already active now.
    let (mut need_from_active, mut need_tot) = (0u64, 0u64);

    let e = trace.num_experts;
    for layer_seq in &seqs {
        for w in layer_seq.windows(2) {
            let now: HashSet<usize> = w[0].iter().copied().collect();
            let next: HashSet<usize> = w[1].iter().copied().collect();
            next_sz += next.len() as u64;
            next_n += 1;
            for x in 0..e {
                let in_next = next.contains(&x);
                if now.contains(&x) {
                    sticky_tot += 1;
                    if in_next {
                        sticky_hit += 1;
                    }
                } else {
                    arrive_tot += 1;
                    if in_next {
                        arrive_hit += 1;
                    }
                }
            }
            for &x in &next {
                need_tot += 1;
                if now.contains(&x) {
                    need_from_active += 1;
                }
            }
        }
    }

    let base_rate = ratio(next_sz as usize, (next_n as usize) * e);
    let stickiness = ratio(sticky_hit as usize, sticky_tot as usize);
    let arrival = ratio(arrive_hit as usize, arrive_tot as usize);
    let recall_from_active = ratio(need_from_active as usize, need_tot as usize);

    println!("\n=== Same-layer token recurrence (decode passes) ===");
    println!(
        "layers={}  experts={}  adjacent-token steps measured={}",
        trace.num_layers, e, next_n
    );
    println!(
        "base rate  P(e routed next token at L)          = {:.1}%",
        base_rate * 100.0
    );
    println!(
        "stickiness P(e next | e active now)             = {:.1}%",
        stickiness * 100.0
    );
    println!(
        "arrival    P(e next | e NOT active now)         = {:.2}%",
        arrival * 100.0
    );
    println!(
        "lift (stickiness / base rate)                  = {:.1}×",
        if base_rate > 0.0 {
            stickiness / base_rate
        } else {
            0.0
        }
    );
    println!(
        "of experts needed next token, share already active = {:.1}%",
        recall_from_active * 100.0
    );

    // Recency window vs LRU: predict "needed next" = union of active sets over
    // the last `w` tokens.  Reports recall (cold experts covered) and the cost
    // (how many experts that keeps "hot").  This is the bar a recurrence-scored
    // eviction policy must beat.
    println!("\nrecency window → recall of next-token experts / avg kept-hot set:");
    for w in [1usize, 2, 4, 8] {
        let (mut rec_hit, mut rec_tot) = (0u64, 0u64);
        let (mut kept_sz, mut kept_n) = (0u64, 0u64);
        for layer_seq in &seqs {
            if layer_seq.len() <= w {
                continue;
            }
            for i in w..layer_seq.len() {
                let mut union: HashSet<usize> = HashSet::new();
                for prev in &layer_seq[i - w..i] {
                    union.extend(prev.iter().copied());
                }
                let next: HashSet<usize> = layer_seq[i].iter().copied().collect();
                kept_sz += union.len() as u64;
                kept_n += 1;
                for &x in &next {
                    rec_tot += 1;
                    if union.contains(&x) {
                        rec_hit += 1;
                    }
                }
            }
        }
        println!(
            "  w={w}: recall={:>5.1}%   kept-hot≈{:>4.1} experts/layer",
            ratio(rec_hit as usize, rec_tot as usize) * 100.0,
            kept_sz as f64 / kept_n.max(1) as f64,
        );
    }
    println!();
}

/// Online per-(layer, expert) recurrence model: learns how often an expert,
/// once routed at a layer, is routed there again on the next token.  This is
/// the eviction-value signal (a resident `(layer, expert)` slot's next use is
/// the same layer on the next token).
#[derive(Clone)]
pub(super) struct RecurrenceModel {
    /// `active[L][e]` = times expert `e` was routed at layer `L`.
    pub(super) active: Vec<Vec<u32>>,
    /// `recur[L][e]` = times `e` at `L` was followed by `e` at `L` next token.
    pub(super) recur: Vec<Vec<u32>>,
    /// Previous token's active set per layer.
    pub(super) prev: Vec<Option<Vec<usize>>>,
}

impl RecurrenceModel {
    pub(super) fn new(num_layers: usize, num_experts: usize) -> Self {
        Self {
            active: vec![vec![0; num_experts]; num_layers],
            recur: vec![vec![0; num_experts]; num_layers],
            prev: vec![None; num_layers],
        }
    }

    /// Observe layer `L`'s active set for the current token, updating the
    /// recurrence statistics against the previous token at the same layer.
    pub(super) fn observe(&mut self, layer: usize, experts: &[usize]) {
        let cur: HashSet<usize> = experts.iter().copied().collect();
        if let Some(prev) = &self.prev[layer] {
            for &e in prev {
                self.active[layer][e] += 1;
                if cur.contains(&e) {
                    self.recur[layer][e] += 1;
                }
            }
        }
        self.prev[layer] = Some(experts.to_vec());
    }

    /// Clear cross-token state (call at prompt boundaries during training).
    pub(super) fn reset(&mut self) {
        self.prev.iter_mut().for_each(|p| *p = None);
    }

    /// Smoothed recurrence probability for a resident `(layer, expert)` — the
    /// estimated chance it is routed at this layer again next token.  Higher =
    /// more valuable to keep.  Unseen experts fall back to the base prior.
    pub(super) fn keep_value(&self, layer: usize, expert: usize, prior: f64) -> f64 {
        let a = self.active[layer][expert] as f64;
        let r = self.recur[layer][expert] as f64;
        // Laplace toward the global prior with a weight of 2 pseudo-obs.
        (r + 2.0 * prior) / (a + 2.0)
    }
}

/// Eviction victim-selection policy for the cache simulator.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Evict {
    /// Least-recently-used (the production behind-layer/last_used stand-in).
    Lru,
    /// Recency primary, recurrence as a refinement worth ~`scale` clock steps:
    /// evict min(last_used + keep_value × scale).  A sticky expert effectively
    /// looks one cyclic cohort "newer" and is kept slightly longer.  `scale = 0`
    /// is exactly LRU.
    Blend { scale: f64 },
}

/// Cache simulator: replays the trace through a fixed VRAM budget, counting
/// expert misses under a given eviction policy and optional PMI prefetch.
/// Lower miss rate = fewer cold loads = higher throughput.
pub(super) struct Sim {
    pub(super) budget: usize,
    /// key = layer * num_experts + expert → (last_used, keep_value).
    pub(super) resident: HashMap<usize, (u64, f64)>,
    pub(super) clock: u64,
    pub(super) misses: u64,
    pub(super) accesses: u64,
}

impl Sim {
    pub(super) fn new(budget: usize) -> Self {
        Self {
            budget,
            resident: HashMap::new(),
            clock: 0,
            misses: 0,
            accesses: 0,
        }
    }

    pub(super) fn touch(&mut self, key: usize, keep: f64) {
        self.clock += 1;
        let c = self.clock;
        self.resident
            .entry(key)
            .and_modify(|v| *v = (c, keep))
            .or_insert((c, keep));
    }

    /// Insert `key`, evicting the worst victim under `policy` if full.
    pub(super) fn insert(&mut self, key: usize, keep: f64, policy: Evict) {
        if !self.resident.contains_key(&key) && self.resident.len() >= self.budget {
            // Pick a victim.
            let victim = self
                .resident
                .iter()
                .min_by(|(_, a), (_, b)| {
                    let (la, ka) = **a;
                    let (lb, kb) = **b;
                    match policy {
                        Evict::Lru => la.cmp(&lb),
                        Evict::Blend { scale } => {
                            let sa = la as f64 + ka * scale;
                            let sb = lb as f64 + kb * scale;
                            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                        }
                    }
                })
                .map(|(k, _)| *k);
            if let Some(v) = victim {
                self.resident.remove(&v);
            }
        }
        self.touch(key, keep);
    }
}

/// Global same-layer stickiness P(e routed next token | e active now), used as
/// the recurrence prior: a freshly-loaded expert was just active, so its prior
/// chance of recurring is the stickiness — not the much lower base rate.
pub(super) fn global_stickiness(trace: &Trace, decode_max: usize) -> f64 {
    let seqs = per_layer_token_sequences(trace, decode_max);
    let (mut hit, mut tot) = (0u64, 0u64);
    for layer_seq in &seqs {
        for w in layer_seq.windows(2) {
            let next: HashSet<usize> = w[1].iter().copied().collect();
            for &x in &w[0] {
                tot += 1;
                if next.contains(&x) {
                    hit += 1;
                }
            }
        }
    }
    ratio(hit as usize, tot as usize)
}

/// Run one full cache simulation over the trace; returns the miss rate.
pub(super) fn simulate(
    trace: &Trace,
    budget: usize,
    policy: Evict,
    prefetch_pmi_k: usize,
    prior: f64,
) -> f64 {
    let e = trace.num_experts;
    let mut sim = Sim::new(budget);
    let mut recur = RecurrenceModel::new(trace.num_layers, e);
    // Prefetch predictor (PMI) learned online alongside.
    let mut pmi = MatrixPredictor::new(trace.num_layers, e, Scoring::Pmi { alpha: 0.5 });

    for pass in &trace.passes {
        pmi.reset_pass();
        for (l, experts) in &pass.layers {
            // PMI prefetch into the cache (free, before the layer "runs").
            if prefetch_pmi_k > 0 {
                for pe in pmi.predict(*l, experts, prefetch_pmi_k) {
                    let key = l * e + pe;
                    let keep = recur.keep_value(*l, pe, prior);
                    // Prefetch only fills if there is room (never evict to prefetch).
                    if sim.resident.contains_key(&key) || sim.resident.len() < sim.budget {
                        sim.touch(key, keep);
                    }
                }
            }
            // Access the experts this layer actually needs.
            for &x in experts {
                sim.accesses += 1;
                let key = l * e + x;
                let keep = recur.keep_value(*l, x, prior);
                if sim.resident.contains_key(&key) {
                    sim.touch(key, keep);
                } else {
                    sim.misses += 1;
                    sim.insert(key, keep, policy);
                }
            }
            // Learn recurrence from this layer's set, and feed the PMI matrix.
            recur.observe(*l, experts);
            pmi.observe(*l, experts);
        }
    }
    ratio(sim.misses as usize, sim.accesses as usize)
}

#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn cache_sim() {
    let Some(records) = load_fixture() else {
        println!("\n[cache_sim] no fixture; capture first (see eval::report).");
        return;
    };
    let trace = Trace::from_records(&records);

    // Total distinct (layer, expert) the trace ever touches sets the scale.
    let distinct = {
        let mut s = HashSet::new();
        for p in &trace.passes {
            for (l, es) in &p.layers {
                for &x in es {
                    s.insert(l * trace.num_experts + x);
                }
            }
        }
        s.len()
    };

    const DECODE_MAX: usize = 16;
    let prior = global_stickiness(&trace, DECODE_MAX);

    println!("\n=== Cache simulation (miss rate, lower is better) ===");
    println!(
        "distinct (layer,expert) touched = {distinct}   recurrence prior = {:.1}%",
        prior * 100.0
    );
    // One cyclic cohort ≈ accesses per token (layers × avg active set).
    let cohort = (trace.num_layers as f64 * trace.avg_set_size()).max(1.0);
    println!(
        "\n{:>7} {:>9} {:>12} {:>13} {:>13} {:>14}",
        "budget", "LRU", "blend×1", "blend×4", "blend×4+PMI", "(Δ vs LRU)"
    );
    println!("{}", "-".repeat(72));
    for frac in [0.3f64, 0.5, 0.7, 0.85] {
        let budget = ((distinct as f64) * frac).ceil() as usize;
        let lru = simulate(&trace, budget, Evict::Lru, 0, prior);
        let b1 = simulate(&trace, budget, Evict::Blend { scale: cohort }, 0, prior);
        let b4 = simulate(
            &trace,
            budget,
            Evict::Blend {
                scale: cohort * 4.0,
            },
            0,
            prior,
        );
        let b4_pmi = simulate(
            &trace,
            budget,
            Evict::Blend {
                scale: cohort * 4.0,
            },
            4,
            prior,
        );
        println!(
            "{:>6.0}% {:>8.1}% {:>11.1}% {:>12.1}% {:>12.1}% {:>+13.1}",
            frac * 100.0,
            lru * 100.0,
            b1 * 100.0,
            b4 * 100.0,
            b4_pmi * 100.0,
            (b4_pmi - lru) * 100.0,
        );
    }
    println!("\nbudget = fraction of distinct (layer,expert) touched; Δ = best − LRU (negative = fewer misses).\n");
}
