//! Ablations of the session-prior predictor, each varying one ingredient of its
//! prior, session and fork.
//!
//! Every function uses the same 21-fold leave-one-out scheme. The variants are a
//! literal fork that clones the base and adapts it with per-token decay, a session
//! matrix with its own decay, a velocity boost, the prior-weight sweep,
//! mass-weighted counts, arrival-only training, recency fusion, and a joint alpha
//! and lambda tune. `loocv_metric` builds a fresh predictor per fold and trains it
//! on the other configs, and `score_into` folds one held-out trace into per-k
//! metrics, with `learn` continuing online updates during the test.

use super::*;

/// §8b — literal "fork": the session model is a *clone* of the base (strong
/// warm-start) that then fast-adapts via per-token decay and may re-score with a
/// different α.  Sweeps λ (adaptation speed) × α.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn loocv_fork() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_fork] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let lambdas = [1.0f64, 0.99, 0.97, 0.95];
    let alphas = [0.4f64, 0.5];
    let mut grid = vec![vec![Metrics::default(); alphas.len()]; lambdas.len()];

    for &c in &trace.configs() {
        let mut base = MatrixPredictor::new(l, e, Scoring::Raw);
        train_predictor(&mut base, &trace.for_configs_except(c), 2);
        let test = trace.for_config(c);
        for (lami, &lam) in lambdas.iter().enumerate() {
            for (ai, &alpha) in alphas.iter().enumerate() {
                // Fork = clone of the base, fast-adapting within the session.
                let mut fork = base.clone();
                fork.decay = lam;
                fork.scoring = Scoring::Pmi { alpha };
                let mut m = vec![Metrics::default(); 1];
                score_into(&mut fork, &test, &[4usize], 4, true, &mut m);
                grid[lami][ai].pred_total += m[0].pred_total;
                grid[lami][ai].pred_hits += m[0].pred_hits;
                grid[lami][ai].miss_total += m[0].miss_total;
                grid[lami][ai].miss_covered += m[0].miss_covered;
            }
        }
    }

    println!("\n=== §8b Literal fork: clone(base) + fast decay (k=4 decode-cov, +online) ===");
    println!("reference: SessionPrior champion = 42.5%");
    print!("{:<8}", "λ\\α");
    for a in alphas {
        print!("{:>8.2}", a);
    }
    println!();
    let mut best = (0.0f64, 0.0f64, 0.0f64);
    for (lami, &lam) in lambdas.iter().enumerate() {
        print!("{:<8.3}", lam);
        for (ai, _) in alphas.iter().enumerate() {
            let cov = grid[lami][ai].coverage() * 100.0;
            print!("{:>7.1}%", cov);
            if cov > best.0 {
                best = (cov, lam, alphas[ai]);
            }
        }
        println!();
    }
    println!("best fork: {:.1}%  λ={} α={}\n", best.0, best.1, best.2);
}

/// §8c — SessionPrior (count-level blend) with a fast-learning session fork:
/// give the session matrix its own decay so it tracks the most recent tokens.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn loocv_sessiondecay() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_sessiondecay] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let decays = [1.0f64, 0.997, 0.99, 0.98, 0.96];
    let mut out = vec![Metrics::default(); decays.len()];
    for &c in &trace.configs() {
        let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
        train_predictor(&mut prior, &trace.for_configs_except(c), 2);
        let test = trace.for_config(c);
        for (di, &sd) in decays.iter().enumerate() {
            let mut sp = SessionPrior::new(prior.clone(), 0.02, 0.5);
            sp.session.decay = sd; // fast-learning fork
            let mut m = vec![Metrics::default(); 1];
            score_into(&mut sp, &test, &[4usize], 4, true, &mut m);
            out[di].pred_total += m[0].pred_total;
            out[di].pred_hits += m[0].pred_hits;
            out[di].miss_total += m[0].miss_total;
            out[di].miss_covered += m[0].miss_covered;
        }
    }
    println!("\n=== §8c SessionPrior + fast-learning fork (session decay, k=4, +online) ===");
    println!("{:<14} {:>10}", "session decay", "k=4 cov");
    println!("{}", "-".repeat(26));
    for (di, &sd) in decays.iter().enumerate() {
        println!(
            "{:<14} {:>9.1}%",
            format!("{sd}"),
            out[di].coverage() * 100.0
        );
    }
    println!();
}

/// §7.6 — velocity (rising-trend) boost on top of SessionPrior.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn loocv_velocity() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_velocity] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [1usize, 2, 4];
    let gammas = [0.0f64, 0.5, 1.0, 2.0];
    let mut out = vec![vec![Metrics::default(); ks.len()]; gammas.len()];
    for &c in &trace.configs() {
        let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
        train_predictor(&mut prior, &trace.for_configs_except(c), 2);
        let test = trace.for_config(c);
        for (gi, &gamma) in gammas.iter().enumerate() {
            let sp = SessionPrior::new(prior.clone(), 0.02, 0.5);
            let mut v = Velocity::new(sp, gamma);
            score_into(&mut v, &test, &ks, 4, true, &mut out[gi]);
        }
    }
    println!("\n=== §7.6 Velocity boost (SessionPrior β=0.02 α=0.5, +online) ===");
    println!(
        "{:<14} {:>10} {:>10} {:>10}",
        "γ (velocity)", "k=1 cov", "k=2 cov", "k=4 cov"
    );
    println!("{}", "-".repeat(48));
    for (gi, &gamma) in gammas.iter().enumerate() {
        let tag = if gamma == 0.0 {
            "0 (sp only)".to_string()
        } else {
            format!("{gamma}")
        };
        println!(
            "{:<14} {:>9.1}% {:>9.1}% {:>9.1}%",
            tag,
            out[gi][0].coverage() * 100.0,
            out[gi][1].coverage() * 100.0,
            out[gi][2].coverage() * 100.0
        );
    }
    println!();
}

/// §7.2 — Bayesian prior+session interpolation: sweep β (prior weight) at α=0.5.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn loocv_sessionprior() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_sessionprior] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [4usize];
    let betas = [0.0f64, 0.02, 0.04, 0.06, 0.08, 0.12];
    let alphas = [0.4f64, 0.5, 0.6];

    // grid[beta][alpha] @ k=4.  Train one prior per fold, reuse across (β,α).
    let mut grid = vec![vec![Metrics::default(); alphas.len()]; betas.len()];
    for &c in &trace.configs() {
        let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
        train_predictor(&mut prior, &trace.for_configs_except(c), 2);
        let test = trace.for_config(c);
        for (bi, &beta) in betas.iter().enumerate() {
            for (ai, &alpha) in alphas.iter().enumerate() {
                let mut p = SessionPrior::new(prior.clone(), beta, alpha);
                let mut m = vec![Metrics::default(); ks.len()];
                score_into(&mut p, &test, &ks, 4, true, &mut m);
                grid[bi][ai].pred_total += m[0].pred_total;
                grid[bi][ai].pred_hits += m[0].pred_hits;
                grid[bi][ai].miss_total += m[0].miss_total;
                grid[bi][ai].miss_covered += m[0].miss_covered;
            }
        }
    }

    println!("\n=== §7.2 Bayesian prior+session interpolation (k=4 decode-cov, +online) ===");
    print!("{:<10}", "β\\α");
    for a in alphas {
        print!("{:>8.2}", a);
    }
    println!();
    let mut best = (0.0f64, 0.0f64, 0.0f64);
    for (bi, &beta) in betas.iter().enumerate() {
        print!("{:<10}", format!("{beta}"));
        for (ai, _) in alphas.iter().enumerate() {
            let cov = grid[bi][ai].coverage() * 100.0;
            print!("{:>7.1}%", cov);
            if cov > best.0 {
                best = (cov, beta, alphas[ai]);
            }
        }
        println!();
    }
    println!("best: cov={:.1}%  β={}  α={}\n", best.0, best.1, best.2);
}

/// Train a matrix over all passes, optionally mass-weighting observations.
pub(super) fn train_mass(p: &mut MatrixPredictor, trace: &Trace, epochs: usize, mass: bool) {
    for _ in 0..epochs {
        for pass in &trace.passes {
            p.reset_pass();
            for (i, (li, ex)) in pass.layers.iter().enumerate() {
                if mass {
                    let m: Vec<f64> = pass.masses[i].iter().map(|&x| x as f64).collect();
                    p.observe_mass(*li, ex, &m);
                } else {
                    p.observe(*li, ex);
                }
            }
        }
    }
}

/// §7.3 — mass-weighted observations (routing-weight P7).  Best base config
/// (SessionPrior β=0.02, α=0.5); compares uniform vs mass-weighted counts.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn loocv_mass() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_mass] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [1usize, 2, 4];
    let variants = [("uniform", false), ("mass-weighted", true)];
    let mut out = vec![vec![Metrics::default(); ks.len()]; variants.len()];

    for &c in &trace.configs() {
        let train = trace.for_configs_except(c);
        let test = trace.for_config(c);
        for (vi, &(_, mass)) in variants.iter().enumerate() {
            let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
            train_mass(&mut prior, &train, 2, mass);
            let mut sp = SessionPrior::new(prior, 0.02, 0.5);
            for pass in &test.passes {
                if !pass.layers.iter().all(|(_, x)| x.len() <= 16) {
                    continue;
                }
                sp.session.reset_pass();
                let mut preds: Vec<Vec<usize>> = Vec::with_capacity(pass.layers.len());
                for (i, (li, ex)) in pass.layers.iter().enumerate() {
                    preds.push(sp.predict(*li, ex, 4));
                    if mass {
                        let m: Vec<f64> = pass.masses[i].iter().map(|&x| x as f64).collect();
                        sp.session.observe_mass(*li, ex, &m);
                    } else {
                        sp.session.observe(*li, ex);
                    }
                }
                for i in 0..pass.layers.len().saturating_sub(1) {
                    let (li, ref cur) = pass.layers[i];
                    let (ln, ref next) = pass.layers[i + 1];
                    if ln != li + 1 || ln < PINNED {
                        continue;
                    }
                    for (ki, &k) in ks.iter().enumerate() {
                        let pr = &preds[i];
                        out[vi][ki].add(&pr[..k.min(pr.len())], cur, next);
                    }
                }
            }
        }
    }

    println!("\n=== §7.3 Mass-weighted observations (SessionPrior β=0.02 α=0.5, +online) ===");
    println!(
        "{:<16} {:>10} {:>10} {:>10}",
        "variant", "k=1 cov", "k=2 cov", "k=4 cov"
    );
    println!("{}", "-".repeat(50));
    for (vi, &(label, _)) in variants.iter().enumerate() {
        println!(
            "{:<16} {:>9.1}% {:>9.1}% {:>9.1}%",
            label,
            out[vi][0].coverage() * 100.0,
            out[vi][1].coverage() * 100.0,
            out[vi][2].coverage() * 100.0
        );
    }
    println!();
}

/// §7.4 — arrival-specialised training (only learn cold transitions).
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn loocv_arrivals() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_arrivals] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [1usize, 2, 4];
    let variants = [("full", false), ("arrivals-only", true)];
    let mut out = vec![vec![Metrics::default(); ks.len()]; variants.len()];

    for &c in &trace.configs() {
        let train = trace.for_configs_except(c);
        let test = trace.for_config(c);
        for (vi, &(_, arr)) in variants.iter().enumerate() {
            let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
            prior.arrivals_only = arr;
            train_predictor(&mut prior, &train, 2);
            let mut sp = SessionPrior::new(prior, 0.02, 0.5);
            sp.session.arrivals_only = arr;
            score_into(&mut sp, &test, &ks, 4, true, &mut out[vi]);
        }
    }

    println!("\n=== §7.4 Arrival-specialised training (SessionPrior β=0.02 α=0.5, +online) ===");
    println!(
        "{:<16} {:>10} {:>10} {:>10}",
        "training", "k=1 cov", "k=2 cov", "k=4 cov"
    );
    println!("{}", "-".repeat(50));
    for (vi, &(label, _)) in variants.iter().enumerate() {
        println!(
            "{:<16} {:>9.1}% {:>9.1}% {:>9.1}%",
            label,
            out[vi][0].coverage() * 100.0,
            out[vi][1].coverage() * 100.0,
            out[vi][2].coverage() * 100.0
        );
    }
    println!();
}

/// §7.5 — recency-fusion: add a same-layer recency channel to SessionPrior.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn loocv_combo() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_combo] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [1usize, 2, 4];
    let gammas = [0.0f64, 0.25, 0.5, 1.0, 2.0];
    let mut out = vec![vec![Metrics::default(); ks.len()]; gammas.len()];

    for &c in &trace.configs() {
        let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
        train_predictor(&mut prior, &trace.for_configs_except(c), 2);
        let test = trace.for_config(c);
        for (gi, &gamma) in gammas.iter().enumerate() {
            let sp = SessionPrior::new(prior.clone(), 0.02, 0.5);
            let mut r = Recency::new(sp, 0.8, gamma);
            score_into(&mut r, &test, &ks, 4, true, &mut out[gi]);
        }
    }

    println!("\n=== §7.5 Recency-fusion (SessionPrior + same-layer recency, +online) ===");
    println!(
        "{:<14} {:>10} {:>10} {:>10}",
        "γ (recency)", "k=1 cov", "k=2 cov", "k=4 cov"
    );
    println!("{}", "-".repeat(48));
    for (gi, &gamma) in gammas.iter().enumerate() {
        let tag = if gamma == 0.0 {
            "0 (sp only)".to_string()
        } else {
            format!("{gamma}")
        };
        println!(
            "{:<14} {:>9.1}% {:>9.1}% {:>9.1}%",
            tag,
            out[gi][0].coverage() * 100.0,
            out[gi][1].coverage() * 100.0,
            out[gi][2].coverage() * 100.0
        );
    }
    println!();
}

/// §7.1 — joint α × λ fine-tune with *within-session* momentum (decay applied
/// only during the held-out generation, not training).  Trains one full matrix
/// per fold, clones per λ.  Prints the k=4 held-out coverage grid (λ × α).
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn loocv_tune() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_tune] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let lambdas = [1.0f64, 0.997, 0.995, 0.99, 0.985, 0.98, 0.97];
    let alphas = [0.3f64, 0.4, 0.5, 0.6, 0.7];
    let maxk = 4usize;
    // grid[lambda][alpha] = Metrics @ k=4
    let mut grid = vec![vec![Metrics::default(); alphas.len()]; lambdas.len()];

    for &c in &trace.configs() {
        let mut base = MatrixPredictor::new(l, e, Scoring::Raw);
        train_predictor(&mut base, &trace.for_configs_except(c), 2); // full prior, no decay
        let test = trace.for_config(c);
        for (lami, &lam) in lambdas.iter().enumerate() {
            let mut p = base.clone();
            p.decay = lam; // within-session momentum only
            for pass in &test.passes {
                if !pass.layers.iter().all(|(_, x)| x.len() <= 16) {
                    continue;
                }
                p.reset_pass();
                let mut preds: Vec<Vec<Vec<usize>>> = Vec::with_capacity(pass.layers.len());
                for (li, ex) in &pass.layers {
                    preds.push(
                        alphas
                            .iter()
                            .map(|&a| p.score_topk(*li, ex, maxk, Scoring::Pmi { alpha: a }))
                            .collect(),
                    );
                    p.observe(*li, ex);
                }
                for i in 0..pass.layers.len().saturating_sub(1) {
                    let (li, ref cur) = pass.layers[i];
                    let (ln, ref next) = pass.layers[i + 1];
                    if ln != li + 1 || ln < PINNED {
                        continue;
                    }
                    for ai in 0..alphas.len() {
                        grid[lami][ai].add(&preds[i][ai], cur, next);
                    }
                }
            }
        }
    }

    println!("\n=== §7.1 Promote α × λ fine-tune (k=4 decode-cov, within-session momentum) ===");
    print!("{:<8}", "λ\\α");
    for a in alphas {
        print!("{:>8.2}", a);
    }
    println!();
    let mut best = (0.0f64, 1.0f64, 0.5f64);
    for (lami, &lam) in lambdas.iter().enumerate() {
        print!("{:<8.3}", lam);
        for (ai, _) in alphas.iter().enumerate() {
            let cov = grid[lami][ai].coverage() * 100.0;
            print!("{:>7.1}%", cov);
            if cov > best.0 {
                best = (cov, lam, alphas[ai]);
            }
        }
        println!();
    }
    println!("best: cov={:.1}%  λ={}  α={}\n", best.0, best.1, best.2);
}

/// §5.3 — promote input-signal ablation (channels), pmi(0.5), +online.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn loocv_inputs() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_inputs] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [1usize, 2, 4];
    use Channel::*;
    let configs: Vec<(&str, Vec<Channel>)> = vec![
        ("L-1 (base)", vec![Prev]),
        ("L-1,L-2", vec![Prev, Prev2]),
        ("L-1,recur", vec![Prev, Recur]),
        ("L-1,pinned", vec![Prev, Pinned]),
        ("all", vec![Prev, Prev2, Recur, Pinned]),
    ];
    println!("\n=== Promote input ablation (pmi 0.5, +online, train_epochs=2) ===");
    println!(
        "{:<14} {:>10} {:>10} {:>10}",
        "inputs", "k=1 cov", "k=2 cov", "k=4 cov"
    );
    println!("{}", "-".repeat(48));
    for (label, chans) in &configs {
        let chans = chans.clone();
        let m = loocv_metric(
            &trace,
            || MultiSource::new(l, e, chans.clone()),
            2,
            &ks,
            true,
        );
        println!(
            "{:<14} {:>9.1}% {:>9.1}% {:>9.1}%",
            label,
            m[0].coverage() * 100.0,
            m[1].coverage() * 100.0,
            m[2].coverage() * 100.0
        );
    }
    println!();
}

/// Score a trained predictor on one held-out trace, folding into `m` (one
/// Metrics per k).  `learn` continues causal online learning during the test.
pub(super) fn score_into<P: Predictor>(
    p: &mut P,
    test: &Trace,
    ks: &[usize],
    maxk: usize,
    learn: bool,
    m: &mut [Metrics],
) {
    for pass in &test.passes {
        if !pass.layers.iter().all(|(_, x)| x.len() <= 16) {
            continue;
        }
        p.reset_pass();
        let mut preds: Vec<Vec<usize>> = Vec::with_capacity(pass.layers.len());
        for (li, ex) in &pass.layers {
            preds.push(p.predict(*li, ex, maxk));
            if learn {
                p.observe(*li, ex);
            }
        }
        for i in 0..pass.layers.len().saturating_sub(1) {
            let (li, ref cur) = pass.layers[i];
            let (ln, ref next) = pass.layers[i + 1];
            if ln != li + 1 || ln < PINNED {
                continue;
            }
            for (ki, &k) in ks.iter().enumerate() {
                let pr = &preds[i];
                m[ki].add(&pr[..k.min(pr.len())], cur, next);
            }
        }
    }
}

/// Generic 21-fold LOOCV: build a fresh predictor per fold via `make`, train on
/// the other 20 configs, score the held-out one.  Returns one Metrics per k.
pub(super) fn loocv_metric<P: Predictor>(
    trace: &Trace,
    mut make: impl FnMut() -> P,
    epochs: usize,
    ks: &[usize],
    learn: bool,
) -> Vec<Metrics> {
    let mut m = vec![Metrics::default(); ks.len()];
    let maxk = *ks.iter().max().unwrap();
    for &c in &trace.configs() {
        let mut p = make();
        train_predictor(&mut p, &trace.for_configs_except(c), epochs);
        score_into(&mut p, &trace.for_config(c), ks, maxk, learn, &mut m);
    }
    m
}

/// §5.2 — promote matrix size (layer resolution) ablation, pmi(0.5), +online.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn loocv_size() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_size] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [1usize, 2, 4];
    let sc = Scoring::Pmi { alpha: 0.5 };
    println!("\n=== Promote size ablation (pmi 0.5, +online, train_epochs=2) ===");
    println!(
        "{:<12} {:>10} {:>10} {:>10}",
        "resolution", "k=1 cov", "k=2 cov", "k=4 cov"
    );
    println!("{}", "-".repeat(46));
    for (label, group) in [
        ("per-pair", 1usize),
        ("group-4", 4),
        ("group-8", 8),
        ("shared", l),
    ] {
        let m = loocv_metric(
            &trace,
            || MatrixPredictor::new_cfg(l, e, sc, group, 1.0),
            2,
            &ks,
            true,
        );
        println!(
            "{:<12} {:>9.1}% {:>9.1}% {:>9.1}%",
            label,
            m[0].coverage() * 100.0,
            m[1].coverage() * 100.0,
            m[2].coverage() * 100.0
        );
    }
    println!();
}

/// §5.4 — promote scoring-formula sweep.  Transform rules share one trained
/// matrix per fold (counts identical); momentum re-trains with a decay.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn loocv_formula() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_formula] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [1usize, 2, 4];
    let maxk = 4usize;

    // Transform rules evaluated on one stationary matrix per fold.
    let transforms: Vec<(&str, Scoring)> = vec![
        ("raw", Scoring::Raw),
        ("conditional", Scoring::Conditional),
        ("pmi(0.25)", Scoring::Pmi { alpha: 0.25 }),
        ("pmi(0.5)", Scoring::Pmi { alpha: 0.5 }),
        ("pmi(0.75)", Scoring::Pmi { alpha: 0.75 }),
        ("pmi(1.0)", Scoring::Pmi { alpha: 1.0 }),
        ("dirichlet(2)", Scoring::Dirichlet { alpha: 2.0 }),
        ("wilson(1.0)", Scoring::Wilson { z: 1.0 }),
    ];
    let mut mt = vec![vec![Metrics::default(); ks.len()]; transforms.len()];
    for &c in &trace.configs() {
        let mut p = MatrixPredictor::new(l, e, Scoring::Raw);
        train_predictor(&mut p, &trace.for_configs_except(c), 2);
        let test = trace.for_config(c);
        for pass in &test.passes {
            if !pass.layers.iter().all(|(_, x)| x.len() <= 16) {
                continue;
            }
            p.reset_pass();
            let mut preds: Vec<Vec<Vec<usize>>> = Vec::with_capacity(pass.layers.len());
            for (li, ex) in &pass.layers {
                preds.push(
                    transforms
                        .iter()
                        .map(|&(_, s)| p.score_topk(*li, ex, maxk, s))
                        .collect(),
                );
                p.observe(*li, ex); // +online
            }
            for i in 0..pass.layers.len().saturating_sub(1) {
                let (li, ref cur) = pass.layers[i];
                let (ln, ref next) = pass.layers[i + 1];
                if ln != li + 1 || ln < PINNED {
                    continue;
                }
                for ti in 0..transforms.len() {
                    for (ki, &k) in ks.iter().enumerate() {
                        let pr = &preds[i][ti];
                        mt[ti][ki].add(&pr[..k.min(pr.len())], cur, next);
                    }
                }
            }
        }
    }

    // Momentum: re-trained matrices with per-pass decay, scored as pmi(0.5).
    let sc = Scoring::Pmi { alpha: 0.5 };
    let momentum: Vec<(&str, f64)> = vec![
        ("momentum(0.99)", 0.99),
        ("momentum(0.97)", 0.97),
        ("momentum(0.95)", 0.95),
        ("momentum(0.90)", 0.90),
        ("momentum(0.80)", 0.80),
    ];
    let mom_metrics: Vec<Vec<Metrics>> = momentum
        .iter()
        .map(|&(_, d)| {
            loocv_metric(
                &trace,
                || MatrixPredictor::new_cfg(l, e, sc, 1, d),
                2,
                &ks,
                true,
            )
        })
        .collect();

    println!("\n=== Promote formula sweep (+online, train_epochs=2) ===");
    println!(
        "{:<16} {:>10} {:>10} {:>10}",
        "formula", "k=1 cov", "k=2 cov", "k=4 cov"
    );
    println!("{}", "-".repeat(50));
    for ti in 0..transforms.len() {
        println!(
            "{:<16} {:>9.1}% {:>9.1}% {:>9.1}%",
            transforms[ti].0,
            mt[ti][0].coverage() * 100.0,
            mt[ti][1].coverage() * 100.0,
            mt[ti][2].coverage() * 100.0
        );
    }
    for (mi, &(label, _)) in momentum.iter().enumerate() {
        println!(
            "{:<16} {:>9.1}% {:>9.1}% {:>9.1}%",
            label,
            mom_metrics[mi][0].coverage() * 100.0,
            mom_metrics[mi][1].coverage() * 100.0,
            mom_metrics[mi][2].coverage() * 100.0
        );
    }
    println!();
}
