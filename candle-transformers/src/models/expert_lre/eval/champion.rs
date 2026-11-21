//! The champion transition predictor and the leave-one-out evaluations that
//! choose its settings.
//!
//! Each evaluation holds out one prompt at a time, so no prompt trains the
//! predictor that scores it. `loocv_prefetch` trains once per fold and scores
//! every scoring rule at each fan-out k in a single traversal, first frozen and
//! then with online learning, and it excludes pinned target layers because those
//! are always resident. The remaining functions sweep the velocity, training
//! amount, fan-out and two-stage variants of the same model.

use super::*;

/// Per-config (per-prompt) breakdown: proves the harness distinguishes test
/// cases and shows how predictability varies by prompt.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn per_config() {
    let Some(records) = load_fixture() else {
        println!("\n[per_config] no fixture; capture first (see eval::report).");
        return;
    };
    let trace = Trace::from_records(&records);
    const DECODE_MAX: usize = 16;

    println!("\n=== Per-config (per-prompt) breakdown ===");
    println!(
        "{:>4} {:>9} {:>7} {:>9} {:>12} {:>12}",
        "cfg", "records", "passes", "avg|set|", "pmi-cov@4", "stickiness"
    );
    println!("{}", "-".repeat(58));
    for c in trace.configs() {
        let sub = trace.for_config(c);
        let records: usize = sub.passes.iter().map(|p| p.layers.len()).sum();
        let mut pmi =
            MatrixPredictor::new(sub.num_layers, sub.num_experts, Scoring::Pmi { alpha: 0.5 });
        let (_, d) = replay(&sub, &mut pmi, 4, DECODE_MAX, 0);
        let stick = global_stickiness(&sub, DECODE_MAX);
        println!(
            "{:>4} {:>9} {:>7} {:>9.1} {:>11.1}% {:>11.1}%",
            c,
            records,
            sub.passes.len(),
            sub.avg_set_size(),
            d.coverage() * 100.0,
            stick * 100.0,
        );
    }
    println!();
}

/// Unbiased promote (prefetch) evaluation via 21-fold leave-one-out CV.
///
/// For each held-out prompt the matrix is trained once on the other 20, then
/// every scoring × k is evaluated in a single held-out traversal (the counts
/// are identical across scorings, so training is not repeated).  Run with
/// `--release` for fast iteration.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn loocv_prefetch() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_prefetch] no fixture; capture first (see eval::report).");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    const EPOCHS: usize = 2;
    const DECODE_MAX: usize = 16;

    let scorings: [(&str, Scoring); 3] = [
        ("raw", Scoring::Raw),
        ("conditional", Scoring::Conditional),
        ("pmi(0.5)", Scoring::Pmi { alpha: 0.5 }),
    ];
    let ks = [1usize, 2, 4];

    // metrics[mode][scoring][k]; mode 0 = frozen, 1 = +online.
    let mut m = vec![vec![vec![Metrics::default(); ks.len()]; scorings.len()]; 2];
    let mut mpop = vec![vec![Metrics::default(); ks.len()]; 2];

    for &c in &trace.configs() {
        let train = trace.for_configs_except(c);
        let test = trace.for_config(c);
        let mut mtx = MatrixPredictor::new(l, e, Scoring::Raw);
        train_predictor(&mut mtx, &train, EPOCHS);
        let mut pop = Popularity::new(l, e);
        train_predictor(&mut pop, &train, EPOCHS);

        // Frozen first (no mutation), then +online (mutates) on the same state.
        for mode in 0..2 {
            let learn = mode == 1;
            for pass in &test.passes {
                if !pass.layers.iter().all(|(_, ex)| ex.len() <= DECODE_MAX) {
                    continue;
                }
                mtx.reset_pass();
                pop.reset_pass();
                let nl = pass.layers.len();
                let maxk = *ks.iter().max().unwrap();
                // Rank once per scoring (top-maxk); each k is a prefix.
                let mut lp: Vec<Vec<Vec<usize>>> = Vec::with_capacity(nl); // [layer][scoring]
                let mut pp: Vec<Vec<usize>> = Vec::with_capacity(nl); // [layer]
                for (li, ex) in &pass.layers {
                    lp.push(
                        scorings
                            .iter()
                            .map(|&(_, sc)| mtx.score_topk(*li, ex, maxk, sc))
                            .collect(),
                    );
                    pp.push(pop.predict(*li, ex, maxk));
                    if learn {
                        mtx.observe(*li, ex);
                        pop.observe(*li, ex);
                    }
                }
                for i in 0..nl.saturating_sub(1) {
                    let (li, ref cur) = pass.layers[i];
                    let (ln, ref next) = pass.layers[i + 1];
                    // Adjacent transitions only; exclude pinned target layers
                    // (always resident — predicting them is moot).
                    if ln != li + 1 || ln < PINNED {
                        continue;
                    }
                    for (ki, &k) in ks.iter().enumerate() {
                        for si in 0..scorings.len() {
                            let p = &lp[i][si];
                            m[mode][si][ki].add(&p[..k.min(p.len())], cur, next);
                        }
                        let pk = &pp[i];
                        mpop[mode][ki].add(&pk[..k.min(pk.len())], cur, next);
                    }
                }
            }
        }
    }

    println!(
        "\n=== Promote: 21-fold leave-one-out CV (train_epochs={EPOCHS}) ===\n\
         held-out decode coverage — every prompt tested on a model that never saw it"
    );
    for (mode, name) in [
        (0usize, "frozen (pure generalization)"),
        (1, "+online (realistic)"),
    ] {
        println!("\n── {name} ──");
        println!(
            "{:<14} {:>3} {:>8} {:>10} {:>12}",
            "predictor", "k", "top1", "precision", "decode-cov"
        );
        println!("{}", "-".repeat(52));
        for ki in 0..ks.len() {
            let pm = &mpop[mode][ki];
            println!(
                "{:<14} {:>3} {:>7.1}% {:>9.1}% {:>11.1}%",
                "popularity",
                ks[ki],
                pm.top1() * 100.0,
                pm.precision() * 100.0,
                pm.coverage() * 100.0
            );
            for si in 0..scorings.len() {
                let mm = &m[mode][si][ki];
                println!(
                    "{:<14} {:>3} {:>7.1}% {:>9.1}% {:>11.1}%",
                    scorings[si].0,
                    ks[ki],
                    mm.top1() * 100.0,
                    mm.precision() * 100.0,
                    mm.coverage() * 100.0
                );
            }
            println!();
        }
    }
}

/// §7.7 — final combined model: SessionPrior (β=0.02, α=0.5) + arrival-
/// specialised training + velocity boost.  Reports the best vs the §5 winner.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn loocv_final() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_final] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [1usize, 2, 4];
    // (label, beta, arrivals, velocity-gamma)
    let configs: Vec<(&str, f64, bool, f64)> = vec![
        ("stationary pmi(0.5)", 1.0, false, 0.0),
        ("SessionPrior", 0.02, false, 0.0),
        ("  + arrivals", 0.02, true, 0.0),
        ("  + velocity", 0.02, false, 1.5),
        ("  + both", 0.02, true, 1.5),
    ];
    let mut out = vec![vec![Metrics::default(); ks.len()]; configs.len()];
    for &c in &trace.configs() {
        let test = trace.for_config(c);
        let train = trace.for_configs_except(c);
        for (ci, &(_, beta, arr, gamma)) in configs.iter().enumerate() {
            let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
            prior.arrivals_only = arr;
            train_predictor(&mut prior, &train, 2);
            let mut sp = SessionPrior::new(prior, beta, 0.5);
            sp.session.arrivals_only = arr;
            let mut v = Velocity::new(sp, gamma);
            score_into(&mut v, &test, &ks, 4, true, &mut out[ci]);
        }
    }
    println!("\n=== §7.7 Final combined promote model (+online, held-out) ===");
    println!("{:<22} {:>9} {:>9} {:>9}", "model", "k=1", "k=2", "k=4");
    println!("{}", "-".repeat(52));
    for (ci, &(label, ..)) in configs.iter().enumerate() {
        println!(
            "{:<22} {:>8.1}% {:>8.1}% {:>8.1}%",
            label,
            out[ci][0].coverage() * 100.0,
            out[ci][1].coverage() * 100.0,
            out[ci][2].coverage() * 100.0
        );
    }
    println!();
}

/// §8 — two-stage base + session-fork architecture: sweep fork weight and fork
/// formula (base fixed at pmi(0.5)).  Compares to the §6 SessionPrior champion.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn loocv_twostage() {
    let Some(records) = load_fixture() else {
        println!("\n[loocv_twostage] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [4usize];
    // fork formula candidates
    let forks: Vec<(&str, Scoring)> = vec![
        ("conditional", Scoring::Conditional),
        ("pmi(0.25)", Scoring::Pmi { alpha: 0.25 }),
        ("pmi(0.5)", Scoring::Pmi { alpha: 0.5 }),
    ];
    let weights = [0.5f64, 1.0, 2.0, 4.0, 8.0];
    let decays = [1.0f64, 0.9];
    // grid[decay][fork][weight] @k=4
    let mut grid = vec![vec![vec![Metrics::default(); weights.len()]; forks.len()]; decays.len()];

    for &c in &trace.configs() {
        let mut base = MatrixPredictor::new(l, e, Scoring::Raw);
        train_predictor(&mut base, &trace.for_configs_except(c), 2);
        let test = trace.for_config(c);
        for (di, &decay) in decays.iter().enumerate() {
            for (fi, &(_, fsc)) in forks.iter().enumerate() {
                for (wi, &w) in weights.iter().enumerate() {
                    let mut ts =
                        TwoStage::new(base.clone(), Scoring::Pmi { alpha: 0.5 }, fsc, w, decay);
                    let mut m = vec![Metrics::default(); ks.len()];
                    score_into(&mut ts, &test, &ks, 4, true, &mut m);
                    grid[di][fi][wi].pred_total += m[0].pred_total;
                    grid[di][fi][wi].pred_hits += m[0].pred_hits;
                    grid[di][fi][wi].miss_total += m[0].miss_total;
                    grid[di][fi][wi].miss_covered += m[0].miss_covered;
                }
            }
        }
    }

    println!("\n=== §8 Two-stage base(pmi0.5) + fast fork (k=4 decode-cov, +online) ===");
    println!("reference: SessionPrior count-blend = 41.8% ; champion = 42.5%");
    let mut best = (0.0f64, String::new());
    for (di, &decay) in decays.iter().enumerate() {
        println!("\n── fork decay = {decay} ──");
        print!("{:<14}", "fork\\w_sess");
        for w in weights {
            print!("{:>8.1}", w);
        }
        println!();
        for (fi, &(flabel, _)) in forks.iter().enumerate() {
            print!("{:<14}", flabel);
            for (wi, _) in weights.iter().enumerate() {
                let cov = grid[di][fi][wi].coverage() * 100.0;
                print!("{:>7.1}%", cov);
                if cov > best.0 {
                    best = (
                        cov,
                        format!("decay={decay} fork={flabel} w={}", weights[wi]),
                    );
                }
            }
            println!();
        }
    }
    println!("\nbest two-stage: {:.1}%  ({})\n", best.0, best.1);
}

/// Build the champion predictor at given hyperparameters.
pub(super) fn champion(
    prior: MatrixPredictor,
    beta: f64,
    alpha: f64,
    gamma: f64,
    fast: f64,
    slow: f64,
) -> Velocity {
    let mut sp = SessionPrior::new(prior, beta, alpha);
    sp.session.arrivals_only = true;
    Velocity::new_decays(sp, gamma, fast, slow)
}

/// §9.2 — velocity tuning at β=0.01, α=0.5 (arrivals): γ × (fast,slow) decays.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn tune_vel() {
    let Some(records) = load_fixture() else {
        println!("\n[tune_vel] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let gammas = [0.0f64, 1.0, 2.0, 3.0, 4.0];
    let decays = [
        ("0.7/0.95", 0.7, 0.95),
        ("0.6/0.92", 0.6, 0.92),
        ("0.8/0.97", 0.8, 0.97),
    ];
    let mut grid = vec![vec![Metrics::default(); gammas.len()]; decays.len()];
    for &c in &trace.configs() {
        let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
        prior.arrivals_only = true;
        train_predictor(&mut prior, &trace.for_configs_except(c), 2);
        let test = trace.for_config(c);
        for (di, &(_, fd, sd)) in decays.iter().enumerate() {
            for (gi, &g) in gammas.iter().enumerate() {
                let mut v = champion(prior.clone(), 0.01, 0.5, g, fd, sd);
                let mut m = vec![Metrics::default(); 1];
                score_into(&mut v, &test, &[4usize], 4, true, &mut m);
                grid[di][gi].pred_total += m[0].pred_total;
                grid[di][gi].pred_hits += m[0].pred_hits;
                grid[di][gi].miss_total += m[0].miss_total;
                grid[di][gi].miss_covered += m[0].miss_covered;
            }
        }
    }
    println!("\n=== §9.2 Velocity tuning (β=0.01 α=0.5 arrivals, k=4 cov) ===");
    print!("{:<10}", "decay\\γ");
    for g in gammas {
        print!("{:>8.1}", g);
    }
    println!();
    let mut best = (0.0f64, String::new());
    for (di, &(dl, ..)) in decays.iter().enumerate() {
        print!("{:<10}", dl);
        for (gi, &g) in gammas.iter().enumerate() {
            let cov = grid[di][gi].coverage() * 100.0;
            print!("{:>7.2}%", cov);
            if cov > best.0 {
                best = (cov, format!("decay={dl} γ={g}"));
            }
        }
        println!();
    }
    println!("best: {:.2}%  ({})\n", best.0, best.1);
}

/// §9.3 — base-prior training amount (train_epochs) at the champion config.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn tune_epochs() {
    let Some(records) = load_fixture() else {
        println!("\n[tune_epochs] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let epochs = [1usize, 2, 3, 4, 6];
    let mut out = vec![Metrics::default(); epochs.len()];
    for &c in &trace.configs() {
        let test = trace.for_config(c);
        let train = trace.for_configs_except(c);
        for (ei, &ep) in epochs.iter().enumerate() {
            let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
            prior.arrivals_only = true;
            train_predictor(&mut prior, &train, ep);
            let mut v = champion(prior, 0.01, 0.5, 2.0, 0.7, 0.95);
            let mut m = vec![Metrics::default(); 1];
            score_into(&mut v, &test, &[4usize], 4, true, &mut m);
            out[ei].pred_total += m[0].pred_total;
            out[ei].pred_hits += m[0].pred_hits;
            out[ei].miss_total += m[0].miss_total;
            out[ei].miss_covered += m[0].miss_covered;
        }
    }
    println!("\n=== §9.3 Base-prior train_epochs (champion, k=4 cov) ===");
    println!("{:<10} {:>10}", "epochs", "k=4 cov");
    println!("{}", "-".repeat(22));
    for (ei, &ep) in epochs.iter().enumerate() {
        println!("{:<10} {:>9.2}%", ep, out[ei].coverage() * 100.0);
    }
    println!();
}

/// §9.4 — locked champion: coverage vs fan-out k (deployment knob).
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn champion_kcurve() {
    let Some(records) = load_fixture() else {
        println!("\n[champion_kcurve] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let ks = [1usize, 2, 3, 4, 6, 8];
    let mut out = vec![Metrics::default(); ks.len()];
    for &c in &trace.configs() {
        let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
        prior.arrivals_only = true;
        train_predictor(&mut prior, &trace.for_configs_except(c), 2);
        let mut v = champion(prior, 0.01, 0.5, 2.0, 0.7, 0.95);
        score_into(
            &mut v,
            &trace.for_config(c),
            &ks,
            *ks.iter().max().unwrap(),
            true,
            &mut out,
        );
    }
    println!("\n=== §9.4 Locked champion — coverage & precision vs k ===");
    println!("{:<6} {:>10} {:>12}", "k", "cov", "precision");
    println!("{}", "-".repeat(30));
    for (ki, &k) in ks.iter().enumerate() {
        println!(
            "{:<6} {:>9.1}% {:>11.1}%",
            k,
            out[ki].coverage() * 100.0,
            out[ki].precision() * 100.0
        );
    }
    println!();
}

/// §9.1 — champion fine-tune: fine β×α grid on the full stack (arrival-
/// specialised prior+session).  Trains once per fold, scores all (β,α).
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn tune_champion() {
    let Some(records) = load_fixture() else {
        println!("\n[tune_champion] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let (l, e) = (trace.num_layers, trace.num_experts);
    let betas = [0.005f64, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04];
    let alphas = [0.40f64, 0.45, 0.50, 0.55, 0.60];
    let mut grid = vec![vec![Metrics::default(); alphas.len()]; betas.len()];

    for &c in &trace.configs() {
        let mut prior = MatrixPredictor::new(l, e, Scoring::Raw);
        prior.arrivals_only = true;
        train_predictor(&mut prior, &trace.for_configs_except(c), 2);
        let mut sp = SessionPrior::new(prior, 0.02, 0.5);
        sp.session.arrivals_only = true;
        let test = trace.for_config(c);
        for pass in &test.passes {
            if !pass.layers.iter().all(|(_, x)| x.len() <= 16) {
                continue;
            }
            sp.session.reset_pass();
            let mut preds: Vec<Vec<Vec<usize>>> = Vec::with_capacity(pass.layers.len());
            for (li, ex) in &pass.layers {
                let mut row = Vec::with_capacity(betas.len() * alphas.len());
                for &b in &betas {
                    for &a in &alphas {
                        let p = sp
                            .scores_with(*li, ex, b, a)
                            .map(|s| top_k_excluding(&s, ex, 4))
                            .unwrap_or_default();
                        row.push(p);
                    }
                }
                preds.push(row);
                sp.session.observe(*li, ex);
            }
            for i in 0..pass.layers.len().saturating_sub(1) {
                let (li, ref cur) = pass.layers[i];
                let (ln, ref next) = pass.layers[i + 1];
                if ln != li + 1 || ln < PINNED {
                    continue;
                }
                for bi in 0..betas.len() {
                    for ai in 0..alphas.len() {
                        grid[bi][ai].add(&preds[i][bi * alphas.len() + ai], cur, next);
                    }
                }
            }
        }
    }

    println!("\n=== §9.1 Champion β×α fine-tune (k=4 decode-cov, arrivals, +online) ===");
    print!("{:<8}", "β\\α");
    for a in alphas {
        print!("{:>8.2}", a);
    }
    println!();
    let mut best = (0.0f64, 0.0f64, 0.0f64);
    for (bi, &b) in betas.iter().enumerate() {
        print!("{:<8}", format!("{b}"));
        for (ai, _) in alphas.iter().enumerate() {
            let cov = grid[bi][ai].coverage() * 100.0;
            print!("{:>7.1}%", cov);
            if cov > best.0 {
                best = (cov, b, alphas[ai]);
            }
        }
        println!();
    }
    println!("best: cov={:.2}%  β={}  α={}\n", best.0, best.1, best.2);
}
