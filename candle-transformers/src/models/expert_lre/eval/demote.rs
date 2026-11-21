//! Eviction-policy evaluation: whether a frequency bonus over plain LRU lowers
//! expert misses under group eviction.
//!
//! The `Demote` enum names the policies compared, from least-recently-used through
//! least-frequently-used and frequency-weighted recency to two learned recurrence
//! keep-values. Lower scores are evicted first. `demote_sim` runs one held-out
//! prompt with end-of-pass batch eviction of the bottom fraction and free-slot PMI
//! prefetch, and reports misses, accesses, evictions and re-queries within a
//! window, which measure regret. `tune_demote` sets the frequency bonus scale, and
//! zero is plain LRU.

use super::*;

/// §9.5 — demote LFRU frequency-bonus scale tuning vs LRU.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn tune_demote() {
    let Some(records) = load_fixture() else {
        println!("\n[tune_demote] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let prior = global_stickiness(&trace, 16);
    let base_cohort = (l as f64 * trace.avg_set_size()).max(1.0);
    // LFRU bonus scale = cohort * mult; mult=0 is plain LRU.
    let mults = [0.0f64, 1.0, 4.0, 8.0, 16.0, 32.0, 64.0];
    let budgets = [0.5f64, 0.7];
    let mut acc = vec![vec![(0u64, 0u64); budgets.len()]; mults.len()];
    for &c in &trace.configs() {
        let train = trace.for_configs_except(c);
        let mut pmi0 = MatrixPredictor::new(l, e, Scoring::Pmi { alpha: 0.5 });
        train_predictor(&mut pmi0, &train, 2);
        let mut recur0 = RecurrenceModel::new(l, e);
        train_recur(&mut recur0, &train);
        let test = trace.for_config(c);
        let distinct = {
            let mut s = HashSet::new();
            for p in &test.passes {
                if p.layers.iter().all(|(_, x)| x.len() <= 16) {
                    for (ly, ex) in &p.layers {
                        for &x in ex {
                            s.insert(ly * e + x);
                        }
                    }
                }
            }
            s.len()
        };
        for (bi, &frac) in budgets.iter().enumerate() {
            let budget = ((distinct as f64) * frac).ceil() as usize;
            for (mi, &mult) in mults.iter().enumerate() {
                let pol = if mult == 0.0 {
                    Demote::Lru
                } else {
                    Demote::Lfru
                };
                let mut r = recur0.clone();
                let mut p = pmi0.clone();
                let (misses, accesses, _, _) = demote_sim(
                    &test,
                    budget,
                    pol,
                    &mut r,
                    prior,
                    0.05,
                    8,
                    &mut p,
                    base_cohort * mult,
                );
                acc[mi][bi].0 += misses;
                acc[mi][bi].1 += accesses;
            }
        }
    }
    println!("\n=== §9.5 Demote LFRU freq-bonus scale (miss-rate, lower better) ===");
    println!(
        "{:<14} {:>12} {:>12}",
        "scale(×cohort)", "budget=50%", "budget=70%"
    );
    println!("{}", "-".repeat(40));
    for (mi, &mult) in mults.iter().enumerate() {
        let tag = if mult == 0.0 {
            "LRU".to_string()
        } else {
            format!("LFRU×{mult}")
        };
        println!(
            "{:<14} {:>11.1}% {:>11.1}%",
            tag,
            ratio(acc[mi][0].0 as usize, acc[mi][0].1 as usize) * 100.0,
            ratio(acc[mi][1].0 as usize, acc[mi][1].1 as usize) * 100.0
        );
    }
    println!();
}

/// Eviction policy for the demote evaluation.  Lower score = evict first.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Demote {
    /// Least-recently-used (production stand-in; temporal locality).
    Lru,
    /// Least-frequently-used (long-tail popularity).
    Lfu,
    /// Frequency-weighted recency: frequent experts get a recency bonus.
    Lfru,
    /// Learned recurrence keep-value, recency as refinement (recency primary).
    Blend,
    /// Pure learned recurrence keep-value (no recency).
    Recurrence,
}

/// Train a recurrence model on a multi-prompt trace, resetting cross-token
/// state at each prompt boundary.
pub(super) fn train_recur(r: &mut RecurrenceModel, trace: &Trace) {
    let mut cfg: Option<u16> = None;
    for pass in &trace.passes {
        if !pass.layers.iter().all(|(_, x)| x.len() <= 16) {
            continue;
        }
        if cfg != Some(pass.config) {
            r.reset();
            cfg = Some(pass.config);
        }
        for (l, ex) in &pass.layers {
            r.observe(*l, ex);
        }
    }
}

/// Production-style demote simulation on one held-out prompt: end-of-pass batch
/// eviction of the bottom `evict_frac` (group decision) + free-slot PMI
/// prefetch.  Returns (misses, accesses, evicted, re-queried-within-W).
#[allow(clippy::too_many_arguments)]
pub(super) fn demote_sim(
    test: &Trace,
    budget: usize,
    policy: Demote,
    recur: &mut RecurrenceModel,
    prior: f64,
    evict_frac: f64,
    regret_w: usize,
    pmi: &mut MatrixPredictor,
    cohort: f64,
) -> (u64, u64, u64, u64) {
    let e = test.num_experts;
    let decode: Vec<&Pass> = test
        .passes
        .iter()
        .filter(|p| p.layers.iter().all(|(_, x)| x.len() <= 16))
        .collect();
    // key = layer*E + expert → token indices it is accessed at (for regret).
    let mut access_tokens: HashMap<usize, Vec<usize>> = HashMap::new();
    for (t, pass) in decode.iter().enumerate() {
        for (l, ex) in &pass.layers {
            for &x in ex {
                access_tokens.entry(l * e + x).or_default().push(t);
            }
        }
    }

    // resident: key → (last_used, freq)
    let mut resident: HashMap<usize, (u64, u64)> = HashMap::new();
    let mut clock = 0u64;
    let (mut misses, mut accesses, mut evicted, mut requeried) = (0u64, 0u64, 0u64, 0u64);

    // Eviction score (lower = evict first).
    let escore = |key: usize, last_used: u64, freq: u64, recur: &RecurrenceModel| -> f64 {
        let (l, ex) = (key / e, key % e);
        match policy {
            Demote::Lru => last_used as f64,
            Demote::Lfu => freq as f64,
            Demote::Lfru => last_used as f64 + (freq as f64).ln_1p() * cohort,
            Demote::Recurrence => recur.keep_value(l, ex, prior),
            Demote::Blend => last_used as f64 + recur.keep_value(l, ex, prior) * (cohort * 4.0),
        }
    };

    for (t, pass) in decode.iter().enumerate() {
        pmi.reset_pass();
        for (l, ex) in &pass.layers {
            // Free-slot-only PMI prefetch.
            for pe in pmi.score_topk(*l, ex, 4, Scoring::Pmi { alpha: 0.5 }) {
                let key = l * e + pe;
                if resident.contains_key(&key) || resident.len() < budget {
                    clock += 1;
                    resident.entry(key).or_insert((clock, 0)).0 = clock;
                }
            }
            for &x in ex {
                accesses += 1;
                let key = l * e + x;
                clock += 1;
                if let Some(v) = resident.get_mut(&key) {
                    v.0 = clock;
                    v.1 += 1;
                } else {
                    misses += 1;
                    if resident.len() >= budget {
                        // Evict the single worst by policy.
                        if let Some(victim) = resident
                            .iter()
                            .min_by(|a, b| {
                                escore(*a.0, a.1 .0, a.1 .1, recur)
                                    .partial_cmp(&escore(*b.0, b.1 .0, b.1 .1, recur))
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(k, _)| *k)
                        {
                            resident.remove(&victim);
                        }
                    }
                    resident.insert(key, (clock, 1));
                }
            }
            recur.observe(*l, ex);
            pmi.observe(*l, ex);
        }

        // End-of-pass batch eviction of the bottom `evict_frac` group.
        let nevict = ((resident.len() as f64) * evict_frac).ceil() as usize;
        if nevict > 0 && resident.len() > nevict {
            let mut ranked: Vec<(usize, f64)> = resident
                .iter()
                .map(|(&k, &(lu, fr))| (k, escore(k, lu, fr, recur)))
                .collect();
            ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(k, _) in ranked.iter().take(nevict) {
                resident.remove(&k);
                evicted += 1;
                if let Some(ts) = access_tokens.get(&k) {
                    if ts.iter().any(|&tt| tt > t && tt <= t + regret_w) {
                        requeried += 1;
                    }
                }
            }
        }
    }
    (misses, accesses, evicted, requeried)
}

/// §5.5 — demote group-eviction evaluation via 21-fold LOOCV.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn loocv_demote() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_demote] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let prior = global_stickiness(&trace, 16);
    let cohort = (l as f64 * trace.avg_set_size()).max(1.0);
    const EVICT_FRAC: f64 = 0.05;
    const REGRET_W: usize = 8;

    let policies = [
        ("LRU", Demote::Lru),
        ("LFU", Demote::Lfu),
        ("LFRU", Demote::Lfru),
        ("recurrence", Demote::Recurrence),
        ("blend", Demote::Blend),
    ];
    let budgets = [0.5f64, 0.7];
    // [policy][budget] → (misses, accesses, evicted, requeried)
    let mut acc = vec![vec![(0u64, 0u64, 0u64, 0u64); budgets.len()]; policies.len()];

    for &c in &trace.configs() {
        let train = trace.for_configs_except(c);
        let mut recur0 = RecurrenceModel::new(l, e);
        train_recur(&mut recur0, &train);
        let mut pmi0 = MatrixPredictor::new(l, e, Scoring::Pmi { alpha: 0.5 });
        train_predictor(&mut pmi0, &train, 2);

        let test = trace.for_config(c);
        let distinct = {
            let mut s = HashSet::new();
            for p in &test.passes {
                if p.layers.iter().all(|(_, x)| x.len() <= 16) {
                    for (ly, ex) in &p.layers {
                        for &x in ex {
                            s.insert(ly * e + x);
                        }
                    }
                }
            }
            s.len()
        };

        for (bi, &frac) in budgets.iter().enumerate() {
            let budget = ((distinct as f64) * frac).ceil() as usize;
            for (pi, &(_, pol)) in policies.iter().enumerate() {
                let mut r = recur0.clone();
                let mut p = pmi0.clone();
                let (mi, ac, ev, rq) = demote_sim(
                    &test, budget, pol, &mut r, prior, EVICT_FRAC, REGRET_W, &mut p, cohort,
                );
                let s = &mut acc[pi][bi];
                s.0 += mi;
                s.1 += ac;
                s.2 += ev;
                s.3 += rq;
            }
        }
    }

    println!(
        "\n=== Demote: 21-fold LOOCV (batch-evict {:.0}%/token, regret window W={}) ===",
        EVICT_FRAC * 100.0,
        REGRET_W
    );
    println!(
        "prior(stickiness)={:.1}%   lower miss-rate and group-regret are better",
        prior * 100.0
    );
    for (bi, &frac) in budgets.iter().enumerate() {
        println!("\n── VRAM budget = {:.0}% of working set ──", frac * 100.0);
        println!(
            "{:<12} {:>12} {:>14}",
            "policy", "miss-rate", "group-regret"
        );
        println!("{}", "-".repeat(40));
        for (pi, &(label, _)) in policies.iter().enumerate() {
            let (mi, ac, ev, rq) = acc[pi][bi];
            println!(
                "{:<12} {:>11.1}% {:>13.1}%",
                label,
                ratio(mi as usize, ac as usize) * 100.0,
                ratio(rq as usize, ev as usize) * 100.0
            );
        }
    }
    println!();
}
