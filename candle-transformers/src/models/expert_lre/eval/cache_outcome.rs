//! The end-to-end expert cache simulation at a fixed VRAM budget, classifying each
//! decode access as a hit, a soft miss or a hard miss.
//!
//! The champion predictor drives free-slot prefetch one layer ahead, and an LFRU
//! policy drives eviction, with end-of-pass batch eviction making room for
//! prefetch. A hard miss is a demand load on the critical path and stalls the
//! pipeline, while a soft miss is overlapped. `stall_cost` weights hard misses
//! above soft ones, and `bandwidth` counts demand and prefetch transfers per
//! hundred accesses, so a sweep can find the cheapest stall that fits a bandwidth
//! budget.

use super::*;

/// Outcome of the end-to-end cache simulation (counts over decode accesses).
#[derive(Clone, Copy, Default)]
pub(super) struct Outcome {
    pub(super) hit: u64,      // resident, never evicted
    pub(super) soft: u64,     // resident only via prefetch (overlapped load)
    pub(super) hard: u64,     // demand-load on the critical path (stall)
    pub(super) prefetch: u64, // total experts speculatively loaded (used + wasted)
}

impl Outcome {
    pub(super) fn accesses(&self) -> u64 {
        self.hit + self.soft + self.hard
    }
    /// Stall-weighted cost per 100 accesses: hard×`w_hard` + soft×`w_soft`.
    pub(super) fn stall_cost(&self, w_hard: f64, w_soft: f64) -> f64 {
        100.0 * (self.hard as f64 * w_hard + self.soft as f64 * w_soft)
            / self.accesses().max(1) as f64
    }
    /// PCIe transfers per 100 accesses (bandwidth): demand loads + all prefetch.
    pub(super) fn bandwidth(&self) -> f64 {
        100.0 * (self.hard + self.prefetch) as f64 / self.accesses().max(1) as f64
    }
}

/// End-to-end cache-outcome simulation: the champion predictor drives free-slot
/// prefetch (one layer ahead), LFRU×16 drives eviction, with end-of-pass batch
/// eviction creating prefetch headroom.  Classifies every decode-layer expert
/// access into hit / soft-miss / hard-miss and totals the prefetch volume.
pub(super) fn cache_outcome(
    test: &Trace,
    budget: usize,
    champ: &mut Velocity,
    cohort: f64,
    evict_frac: f64,
    k_prefetch: usize,
) -> Outcome {
    let e = test.num_experts;
    let nl = test.num_layers;
    let lfru = |last: u64, freq: u64| last as f64 + (freq as f64).ln_1p() * cohort * 16.0;
    // key -> (last_used, freq, pending_prefetch)
    let mut resident: HashMap<usize, (u64, u64, bool)> = HashMap::new();
    let mut clock = 0u64;
    let mut o = Outcome::default();

    for pass in &test.passes {
        if !pass.layers.iter().all(|(_, x)| x.len() <= 16) {
            continue;
        }
        champ.reset_pass();
        for (l, ex) in &pass.layers {
            for &x in ex {
                let key = l * e + x;
                clock += 1;
                if let Some(s) = resident.get_mut(&key) {
                    if s.2 {
                        o.soft += 1;
                        s.2 = false;
                    } else {
                        o.hit += 1;
                    }
                    s.0 = clock;
                    s.1 += 1;
                } else {
                    o.hard += 1;
                    // Demand-load on the critical path: evict an LFRU victim
                    // (never a pinned-layer expert) if the cache is full.
                    if resident.len() >= budget {
                        if let Some(v) = resident
                            .iter()
                            .filter(|(&k, _)| k / e >= PINNED)
                            .min_by(|a, b| {
                                lfru(a.1 .0, a.1 .1)
                                    .partial_cmp(&lfru(b.1 .0, b.1 .1))
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(k, _)| *k)
                        {
                            resident.remove(&v);
                        }
                    }
                    resident.insert(key, (clock, 1, false));
                }
            }
            champ.observe(*l, ex);
            // Free-slot-only prefetch of the next layer's predicted experts.
            if l + 1 < nl && l + 1 >= PINNED {
                for pe in champ.predict(*l, ex, k_prefetch) {
                    let pk = (l + 1) * e + pe;
                    if resident.len() >= budget {
                        break;
                    }
                    if resident.get(&pk).is_none() {
                        resident.insert(pk, (clock, 0, true));
                        o.prefetch += 1;
                    }
                }
            }
        }
        // End-of-pass: batch-evict `evict_frac` of non-pinned slots (headroom).
        let np_count = resident.iter().filter(|(&k, _)| k / e >= PINNED).count();
        let nevict = ((np_count as f64) * evict_frac).ceil() as usize;
        if nevict > 0 {
            let mut ranked: Vec<(usize, f64)> = resident
                .iter()
                .filter(|(&k, _)| k / e >= PINNED)
                .map(|(&k, &(lu, fr, _))| (k, lfru(lu, fr)))
                .collect();
            ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(k, _) in ranked.iter().take(nevict) {
                resident.remove(&k);
            }
        }
    }
    o
}

/// §10 — modelled cache outcome at 60% VRAM budget, 5% eviction, our models.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn cache_model_60() {
    let Some(records) = load_fixture() else {
        println!("\n[cache_model_60] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let cohort = (l as f64 * trace.avg_set_size()).max(1.0);

    // (budget_frac, K_prefetch, evict_frac).  Train once per fold, clone per scenario.
    let scen: Vec<(f64, usize, f64)> = vec![
        // 60% budget: K and eviction-headroom sweep.
        (0.60, 4, 0.05),
        (0.60, 8, 0.05),
        (0.60, 8, 0.10),
        (0.60, 8, 0.20),
        // budget sweep at the chosen K=8 / 5% evict.
        (0.40, 8, 0.05),
        (0.50, 8, 0.05),
        (0.70, 8, 0.05),
        (0.80, 8, 0.05),
    ];
    let mut agg = vec![(0u64, 0u64, 0u64); scen.len()];

    for &c in &trace.configs() {
        let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
        prior.arrivals_only = true;
        train_predictor(&mut prior, &trace.for_configs_except(c), 2);
        let base = champion(prior, 0.01, 0.5, 2.0, 0.8, 0.97);
        let test = trace.for_config(c);
        let distinct = {
            let mut set = HashSet::new();
            for p in &test.passes {
                if p.layers.iter().all(|(_, x)| x.len() <= 16) {
                    for (ly, ex) in &p.layers {
                        for &x in ex {
                            set.insert(ly * e + x);
                        }
                    }
                }
            }
            set.len()
        };
        for (si, &(bf, kp, ef)) in scen.iter().enumerate() {
            let budget = ((distinct as f64) * bf).ceil() as usize;
            let mut champ = base.clone();
            let o = cache_outcome(&test, budget, &mut champ, cohort, ef, kp);
            agg[si].0 += o.hit;
            agg[si].1 += o.soft;
            agg[si].2 += o.hard;
        }
    }

    let row = |bf: f64, kp: usize, ef: f64, a: (u64, u64, u64)| {
        let (h, s, hd) = a;
        let tot = (h + s + hd).max(1) as f64;
        println!(
            "{:>6.0}% {:>4} {:>6.0}% {:>9.1}% {:>10.1}% {:>10.1}% {:>11.1}%",
            bf * 100.0,
            kp,
            ef * 100.0,
            h as f64 / tot * 100.0,
            s as f64 / tot * 100.0,
            hd as f64 / tot * 100.0,
            (h + s) as f64 / tot * 100.0,
        );
    };

    println!("\n=== §10 Modelled cache outcome — champion prefetch + LFRU×16 evict ===");
    println!("21-fold held-out, all decode accesses\n");
    println!(
        "{:>7} {:>4} {:>6} {:>10} {:>11} {:>11} {:>12}",
        "budget", "K", "evict", "hit", "soft-miss", "hard-miss", "no-stall"
    );
    println!("{}", "-".repeat(64));
    println!("  -- 60% budget: K and eviction-headroom sweep --");
    for si in 0..4 {
        let (bf, kp, ef) = scen[si];
        row(bf, kp, ef, agg[si]);
    }
    println!("  -- budget sweep (K=8, 5% evict) --");
    row(0.40, 8, 0.05, agg[4]);
    row(0.50, 8, 0.05, agg[5]);
    row(0.60, 8, 0.05, agg[1]);
    row(0.70, 8, 0.05, agg[6]);
    row(0.80, 8, 0.05, agg[7]);
    println!(
        "\nhit = resident (never evicted); soft = prefetched (overlapped); hard = demand-load stall.\n"
    );
}

/// §11 — cost-driven optimisation.  Objective: minimise stall cost
/// (hard×10 + soft×1) — hard misses stall the pipeline on full PCIe latency,
/// soft misses are overlapped.  Bandwidth (demand + prefetch transfers) is the
/// second dimension.  Sweeps forced churn × K (preloads per layer) at 60% VRAM.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn cost_optimize() {
    let Some(records) = load_fixture() else {
        println!("\n[cost_optimize] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let cohort = (l as f64 * trace.avg_set_size()).max(1.0);
    const BUDGET_FRAC: f64 = 0.60;
    const W_HARD: f64 = 10.0;
    const W_SOFT: f64 = 1.0;
    let churns = [0.0f64, 0.005, 0.01, 0.02, 0.03];
    let ks = [4usize, 6, 8, 12, 16];
    let mut agg = vec![vec![Outcome::default(); ks.len()]; churns.len()];

    for &c in &trace.configs() {
        let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
        prior.arrivals_only = true;
        train_predictor(&mut prior, &trace.for_configs_except(c), 2);
        let base = champion(prior, 0.01, 0.5, 2.0, 0.8, 0.97);
        let test = trace.for_config(c);
        let distinct = {
            let mut set = HashSet::new();
            for p in &test.passes {
                if p.layers.iter().all(|(_, x)| x.len() <= 16) {
                    for (ly, ex) in &p.layers {
                        for &x in ex {
                            set.insert(ly * e + x);
                        }
                    }
                }
            }
            set.len()
        };
        let budget = ((distinct as f64) * BUDGET_FRAC).ceil() as usize;
        for (ci, &ch) in churns.iter().enumerate() {
            for (ki, &k) in ks.iter().enumerate() {
                let mut champ = base.clone();
                let o = cache_outcome(&test, budget, &mut champ, cohort, ch, k);
                let a = &mut agg[ci][ki];
                a.hit += o.hit;
                a.soft += o.soft;
                a.hard += o.hard;
                a.prefetch += o.prefetch;
            }
        }
    }

    let grid = |title: &str, f: &dyn Fn(&Outcome) -> f64| {
        println!("\n{title}");
        print!("{:<10}", "churn\\K");
        for k in ks {
            print!("{:>8}", k);
        }
        println!();
        for (ci, &ch) in churns.iter().enumerate() {
            print!("{:<10}", format!("{:.0}%", ch * 100.0));
            for (ki, _) in ks.iter().enumerate() {
                print!("{:>8.2}", f(&agg[ci][ki]));
            }
            println!();
        }
    };

    println!("\n=== §11 Cost optimisation (60% VRAM; hard×10 + soft×1) ===");
    grid("-- stall cost / 100 accesses (LOWER = better) --", &|o| {
        o.stall_cost(W_HARD, W_SOFT)
    });
    grid("-- hard-miss % (the stalls) --", &|o| {
        100.0 * o.hard as f64 / o.accesses().max(1) as f64
    });
    grid("-- soft-miss % (overlapped) --", &|o| {
        100.0 * o.soft as f64 / o.accesses().max(1) as f64
    });
    grid("-- bandwidth: PCIe transfers / 100 accesses --", &|o| {
        o.bandwidth()
    });
    grid("-- prefetch waste % (loaded, never used) --", &|o| {
        100.0 * (o.prefetch.saturating_sub(o.soft)) as f64 / o.prefetch.max(1) as f64
    });

    // Bandwidth-constrained optimum: min stall cost whose PCIe transfer rate
    // fits a budget B (transfers / 100 accesses).  Maps the sweet spot to the
    // hardware's available PCIe bandwidth.
    println!("\n-- bandwidth-constrained sweet spot (min stall cost s.t. transfers/100acc ≤ B) --");
    println!(
        "{:>8} {:>8} {:>5} {:>10} {:>9} {:>9}",
        "B", "churn", "K", "stallcost", "hard%", "soft%"
    );
    for &cap in &[11.0f64, 12.0, 14.0, 17.0, 21.0, 99.0] {
        let mut best = (f64::MAX, 0.0f64, 0usize, 0.0f64, 0.0f64);
        for (ci, &ch) in churns.iter().enumerate() {
            for (ki, &k) in ks.iter().enumerate() {
                let o = &agg[ci][ki];
                if o.bandwidth() <= cap {
                    let cost = o.stall_cost(W_HARD, W_SOFT);
                    if cost < best.0 {
                        let acc = o.accesses().max(1) as f64;
                        best = (
                            cost,
                            ch,
                            k,
                            100.0 * o.hard as f64 / acc,
                            100.0 * o.soft as f64 / acc,
                        );
                    }
                }
            }
        }
        let tag = if cap >= 99.0 {
            "∞".to_string()
        } else {
            format!("{cap:.0}")
        };
        println!(
            "{:>8} {:>7.1}% {:>5} {:>10.2} {:>8.2}% {:>8.2}%",
            tag,
            best.1 * 100.0,
            best.2,
            best.0,
            best.3,
            best.4
        );
    }
    println!();
}
