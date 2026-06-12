//! Diagnostic: compare discrimination ratios across all ScoreFormula variants
//! and depth-weight combinations to find the highest-signal configuration.
//!
//! Run with `-- --nocapture` to see output.

use std::collections::HashMap;

use crate::corpus::{load_fixtures, load_layer_fixtures, CaseType, LayerCaseType, TOOLS};
use crate::harness::Harness;
use candle_conversation::projection::{
    ContentResolver, DepthWeights, PerDepthScores, ScoreFormula, SectionId,
};
use candle_conversation::provenance::{BdpScanner, SigEntry};

fn ratio(intra: f32, inter_mean: f32) -> f32 {
    if inter_mean > 0.0 {
        intra / inter_mean
    } else {
        f32::INFINITY
    }
}

#[test]
fn formula_and_depth_weight_comparison() {
    let (manifest, pf) = load_fixtures();
    let h = Harness::build();
    let resolvers = h.scan_all_pos1(&pf, &manifest);

    let equal_weights = DepthWeights {
        syntactic: 1.0,
        semantic: 1.0,
        pragmatic: 1.0,
    };

    // All formulas with equal depth weights.
    let formulas: &[(&str, ScoreFormula)] = &[
        ("max", ScoreFormula::Max),
        ("mean", ScoreFormula::Mean),
        ("sum", ScoreFormula::Sum),
        ("top_k_mean_8", ScoreFormula::TopKMean { k: 8 }),
        ("count", ScoreFormula::Count),
    ];

    println!("\n── Formula comparison (equal depth weights) ──────────────────────────────");
    println!(
        "{:<16} {:>10} {:>12} {:>9} {:>9} {:>9}",
        "formula", "min_ratio", "mean_ratio", "max_ratio", "min_intra", "min_inter"
    );

    for (name, formula) in formulas {
        let mut ratios = Vec::new();
        let mut intras = Vec::new();
        let mut inter_means = Vec::new();

        for (probe_tool, resolver) in &resolvers {
            let intra =
                resolver.section_score(h.tool_section_ids[*probe_tool], *formula, &equal_weights);
            let inter: Vec<f32> = TOOLS
                .iter()
                .filter(|&&t| t != *probe_tool)
                .map(|&t| resolver.section_score(h.tool_section_ids[t], *formula, &equal_weights))
                .collect();
            let im = inter.iter().sum::<f32>() / inter.len() as f32;
            ratios.push(ratio(intra, im));
            intras.push(intra);
            inter_means.push(im);
        }

        let min_ratio = ratios.iter().cloned().fold(f32::INFINITY, f32::min);
        let mean_ratio = ratios.iter().sum::<f32>() / ratios.len() as f32;
        let max_ratio = ratios.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_intra = intras.iter().cloned().fold(f32::INFINITY, f32::min);
        let min_inter = inter_means.iter().cloned().fold(f32::INFINITY, f32::min);

        println!(
            "{:<16} {:>10.4} {:>12.4} {:>9.4} {:>9.1} {:>9.1}",
            name, min_ratio, mean_ratio, max_ratio, min_intra, min_inter
        );
    }

    // Depth-weight sensitivity for the best candidate formulas.
    let weight_configs: &[(&str, DepthWeights)] = &[
        (
            "syn=1 sem=1 prag=1",
            DepthWeights {
                syntactic: 1.0,
                semantic: 1.0,
                pragmatic: 1.0,
            },
        ),
        (
            "syn=0 sem=1 prag=2",
            DepthWeights {
                syntactic: 0.0,
                semantic: 1.0,
                pragmatic: 2.0,
            },
        ),
        (
            "syn=0 sem=0 prag=1",
            DepthWeights {
                syntactic: 0.0,
                semantic: 0.0,
                pragmatic: 1.0,
            },
        ),
        (
            "syn=1 sem=0 prag=0",
            DepthWeights {
                syntactic: 1.0,
                semantic: 0.0,
                pragmatic: 0.0,
            },
        ),
        (
            "syn=0 sem=1 prag=0",
            DepthWeights {
                syntactic: 0.0,
                semantic: 1.0,
                pragmatic: 0.0,
            },
        ),
    ];

    for (fname, formula) in &[("count", ScoreFormula::Count), ("mean", ScoreFormula::Mean)] {
        println!("\n── Depth-weight sensitivity [{fname}] ─────────────────────────────────");
        println!(
            "{:<24} {:>10} {:>12} {:>9}",
            "weights", "min_ratio", "mean_ratio", "max_ratio"
        );

        for (wname, weights) in weight_configs {
            let mut ratios = Vec::new();
            for (probe_tool, resolver) in &resolvers {
                let intra =
                    resolver.section_score(h.tool_section_ids[*probe_tool], *formula, weights);
                let inter: Vec<f32> = TOOLS
                    .iter()
                    .filter(|&&t| t != *probe_tool)
                    .map(|&t| resolver.section_score(h.tool_section_ids[t], *formula, weights))
                    .collect();
                let im = inter.iter().sum::<f32>() / inter.len() as f32;
                ratios.push(ratio(intra, im));
            }
            let min_r = ratios.iter().cloned().fold(f32::INFINITY, f32::min);
            let mean_r = ratios.iter().sum::<f32>() / ratios.len() as f32;
            let max_r = ratios.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!(
                "{:<24} {:>10.4} {:>12.4} {:>9.4}",
                wname, min_r, mean_r, max_r
            );
        }
    }

    // Per-tool count breakdown: show intra count, best-inter count, ratio.
    println!("\n── Per-tool count breakdown (equal weights) ───────────────────────────────");
    println!(
        "{:<14} {:>12} {:>14} {:>9} {:>14}",
        "tool", "intra_count", "best_inter_count", "ratio", "best_inter_tool"
    );
    for (probe_tool, resolver) in &resolvers {
        let intra = resolver.section_score(
            h.tool_section_ids[*probe_tool],
            ScoreFormula::Count,
            &equal_weights,
        );
        let (best_inter_tool, best_inter_score) = TOOLS
            .iter()
            .filter(|&&t| t != *probe_tool)
            .map(|&t| {
                (
                    t,
                    resolver.section_score(
                        h.tool_section_ids[t],
                        ScoreFormula::Count,
                        &equal_weights,
                    ),
                )
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        println!(
            "{:<14} {:>12.0} {:>14.0} {:>9.4} {:>14}",
            probe_tool,
            intra,
            best_inter_score,
            ratio(intra, best_inter_score),
            best_inter_tool
        );
    }

    // Show hit_rate (count * mean / sum) which normalises for corpus size.
    println!("\n── Hit-rate (count×mean/sum) comparison ───────────────────────────────────");
    println!(
        "{:<14} {:>12} {:>14} {:>9}",
        "tool", "intra_rate%", "inter_mean_rate%", "ratio"
    );
    for (probe_tool, resolver) in &resolvers {
        let sid_intra = h.tool_section_ids[*probe_tool];
        let scores_intra = &resolver.section_scores[&sid_intra];
        // hit_rate per depth = count / (sum / mean)
        let hit_rate = |s: &candle_conversation::projection::TurnScores| -> f32 {
            if s.sum > 0.0 {
                s.count * s.mean / s.sum
            } else {
                0.0
            }
        };
        let intra_rate = (hit_rate(&scores_intra.syn)
            + hit_rate(&scores_intra.sem)
            + hit_rate(&scores_intra.prag))
            / 3.0;

        let inter_rates: Vec<f32> = TOOLS
            .iter()
            .filter(|&&t| t != *probe_tool)
            .map(|&t| {
                let sid = h.tool_section_ids[t];
                let s = &resolver.section_scores[&sid];
                (hit_rate(&s.syn) + hit_rate(&s.sem) + hit_rate(&s.prag)) / 3.0
            })
            .collect();
        let inter_mean_rate = inter_rates.iter().sum::<f32>() / inter_rates.len() as f32;
        println!(
            "{:<14} {:>12.4} {:>14.4} {:>9.4}",
            probe_tool,
            intra_rate * 100.0,
            inter_mean_rate * 100.0,
            ratio(intra_rate, inter_mean_rate)
        );
    }
    println!();
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Length-normalised count: fraction of probe tokens that hit something above
/// threshold in this corpus section.  Immune to corpus entry length bias.
fn hit_rate_score(scores: &PerDepthScores, weights: &DepthWeights) -> f32 {
    let hr = |s: &candle_conversation::projection::TurnScores| -> f32 {
        if s.sum > 0.0 {
            s.count * s.mean / s.sum
        } else {
            0.0
        }
    };
    let w = weights.syntactic + weights.semantic + weights.pragmatic;
    if w == 0.0 {
        return 0.0;
    }
    (hr(&scores.syn) * weights.syntactic
        + hr(&scores.sem) * weights.semantic
        + hr(&scores.prag) * weights.pragmatic)
        / w
}

fn section_hit_rate(
    h: &Harness,
    resolver: &crate::resolver::HarnessResolver,
    tool: &str,
    weights: &DepthWeights,
) -> f32 {
    let sid = h.tool_section_ids[tool];
    resolver
        .section_scores
        .get(&sid)
        .map(|s| hit_rate_score(s, weights))
        .unwrap_or(0.0)
}

// ── grid sweep ────────────────────────────────────────────────────────────────

/// Full grid sweep over depth weights × {max, top_k_mean_8, hit_rate}.
/// Reports the top-20 configs by min_ratio (worst-case discrimination).
#[test]
fn depth_weight_grid_sweep() {
    let (manifest, pf) = load_fixtures();
    let h = Harness::build();
    let resolvers = h.scan_all_pos1(&pf, &manifest);

    let levels = [0.0_f32, 0.5, 1.0, 2.0];

    // Generate all (s, e, p) triples excluding (0,0,0).
    let mut grid: Vec<(f32, f32, f32)> = Vec::new();
    for &s in &levels {
        for &e in &levels {
            for &p in &levels {
                if s + e + p > 0.0 {
                    grid.push((s, e, p));
                }
            }
        }
    }

    // Metric closures: return (intra, inter_mean) per (probe_tool, resolver).
    type Metric =
        Box<dyn Fn(&Harness, &crate::resolver::HarnessResolver, &str, &DepthWeights) -> f32>;
    let metrics: &[(&str, Metric)] = &[
        (
            "max",
            Box::new(|h, r, t, w| r.section_score(h.tool_section_ids[t], ScoreFormula::Max, w)),
        ),
        (
            "top_k_mean_8",
            Box::new(|h, r, t, w| {
                r.section_score(h.tool_section_ids[t], ScoreFormula::TopKMean { k: 8 }, w)
            }),
        ),
        ("hit_rate", Box::new(section_hit_rate)),
    ];

    for (mname, score_fn) in metrics {
        let mut rows: Vec<(f32, f32, f32, f32, f32)> = Vec::new(); // (min_r, mean_r, s, e, p)

        for &(s, e, p) in &grid {
            let w = DepthWeights {
                syntactic: s,
                semantic: e,
                pragmatic: p,
            };
            let mut ratios: Vec<f32> = Vec::new();
            for (probe_tool, resolver) in &resolvers {
                let intra = score_fn(&h, resolver, probe_tool, &w);
                let inter: Vec<f32> = TOOLS
                    .iter()
                    .filter(|&&t| t != *probe_tool)
                    .map(|&t| score_fn(&h, resolver, t, &w))
                    .collect();
                let im = inter.iter().sum::<f32>() / inter.len() as f32;
                ratios.push(ratio(intra, im));
            }
            let min_r = ratios.iter().cloned().fold(f32::INFINITY, f32::min);
            let mean_r = ratios.iter().sum::<f32>() / ratios.len() as f32;
            rows.push((min_r, mean_r, s, e, p));
        }

        // Sort by min_ratio desc, then mean_ratio desc.
        rows.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap()
                .then(b.1.partial_cmp(&a.1).unwrap())
        });

        println!("\n── Grid sweep [{mname}] — top 20 by min_ratio ──────────────────────────");
        println!(
            "{:<6} {:<6} {:<6} {:>10} {:>12}",
            "syn", "sem", "prag", "min_ratio", "mean_ratio"
        );
        for &(min_r, mean_r, s, e, p) in rows.iter().take(20) {
            println!(
                "{:<6.1} {:<6.1} {:<6.1} {:>10.4} {:>12.4}",
                s, e, p, min_r, mean_r
            );
        }

        // Also show the single-band results explicitly.
        println!("\n  Single-band results:");
        for &(label, s, e, p) in &[
            ("syn", 1.0_f32, 0.0_f32, 0.0_f32),
            ("sem", 0.0, 1.0, 0.0),
            ("prag", 0.0, 0.0, 1.0),
        ] {
            if let Some(&(min_r, mean_r, _, _, _)) = rows
                .iter()
                .find(|&&(_, _, rs, re, rp)| rs == s && re == e && rp == p)
            {
                println!("  {label:<6}: min={min_r:.4}  mean={mean_r:.4}");
            }
        }
    }
    println!();
}

// ── calibration sweep ─────────────────────────────────────────────────────────

/// Calibration sweep: finds the best (score_formula, depth_weights) per
/// content type, using the existing tool scenario corpus as calibration data.
///
/// Adds `ScoreFormula::Span { alpha: 2.0 }` to the comparison alongside
/// the previously studied max / top_k_mean_8 / hit_rate metrics.
///
/// Outputs a recommended `depth_weights:` YAML fragment for each content type
/// that can be pasted directly into projection.yaml.
///
/// Run with `-- --nocapture` to see the table and recommendations.
#[test]
fn content_type_calibration_sweep() {
    let (manifest, pf) = load_fixtures();
    let h = Harness::build();
    let resolvers = h.scan_all_pos1(&pf, &manifest);

    // Fine-grained grid: 7 levels per dimension gives good resolution
    // without excessive runtime.
    let levels = [0.0_f32, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0];
    let mut grid: Vec<(f32, f32, f32)> = Vec::new();
    for &s in &levels {
        for &e in &levels {
            for &p in &levels {
                if s + e + p > 0.0 {
                    grid.push((s, e, p));
                }
            }
        }
    }

    type MetricFn =
        Box<dyn Fn(&Harness, &crate::resolver::HarnessResolver, &str, &DepthWeights) -> f32>;
    let metrics: Vec<(&str, MetricFn)> = vec![
        (
            "max",
            Box::new(|h, r, t, w| r.section_score(h.tool_section_ids[t], ScoreFormula::Max, w)),
        ),
        (
            "top_k_mean_8",
            Box::new(|h, r, t, w| {
                r.section_score(h.tool_section_ids[t], ScoreFormula::TopKMean { k: 8 }, w)
            }),
        ),
        (
            "span_a2",
            Box::new(|h, r, t, w| {
                r.section_score(h.tool_section_ids[t], ScoreFormula::Span { alpha: 2.0 }, w)
            }),
        ),
        ("hit_rate", Box::new(section_hit_rate)),
    ];

    // (metric_name, best_min_ratio, best_mean_ratio, best_syn, best_sem, best_prag)
    let mut best_per_metric: Vec<(&str, f32, f32, f32, f32, f32)> = Vec::new();

    println!("\n════════════════════════════════════════════════════════════════════════════");
    println!("  Content-type calibration sweep — corpus: tools (8 tools × positive scenarios)");
    println!("  Metric: min_ratio (worst-case intra/inter discrimination across all tools)");
    println!("════════════════════════════════════════════════════════════════════════════");

    for (mname, score_fn) in &metrics {
        let mut rows: Vec<(f32, f32, f32, f32, f32)> = Vec::new();

        for &(s, e, p) in &grid {
            let w = DepthWeights {
                syntactic: s,
                semantic: e,
                pragmatic: p,
            };
            let mut ratios: Vec<f32> = Vec::new();
            for (probe_tool, resolver) in &resolvers {
                let intra = score_fn(&h, resolver, probe_tool, &w);
                let inter: Vec<f32> = TOOLS
                    .iter()
                    .filter(|&&t| t != *probe_tool)
                    .map(|&t| score_fn(&h, resolver, t, &w))
                    .collect();
                let im = inter.iter().sum::<f32>() / inter.len() as f32;
                ratios.push(ratio(intra, im));
            }
            let min_r = ratios.iter().cloned().fold(f32::INFINITY, f32::min);
            let mean_r = ratios.iter().sum::<f32>() / ratios.len() as f32;
            rows.push((min_r, mean_r, s, e, p));
        }

        rows.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap()
                .then(b.1.partial_cmp(&a.1).unwrap())
        });

        println!("\n── [{mname}] top-5 ──────────────────────────────────────────────────────");
        println!(
            "{:<6} {:<6} {:<6} {:>10} {:>12}",
            "syn", "sem", "prag", "min_ratio", "mean_ratio"
        );
        for &(min_r, mean_r, s, e, p) in rows.iter().take(5) {
            println!(
                "{:<6.2} {:<6.2} {:<6.2} {:>10.4} {:>12.4}",
                s, e, p, min_r, mean_r
            );
        }

        if let Some(&(min_r, mean_r, s, e, p)) = rows.first() {
            best_per_metric.push((mname, min_r, mean_r, s, e, p));
        }
    }

    // ── Cross-metric summary ──────────────────────────────────────────────────
    println!("\n── Cross-metric summary ─────────────────────────────────────────────────────");
    println!(
        "{:<14} {:>10} {:>12}  best weights (syn/sem/prag)",
        "metric", "min_ratio", "mean_ratio"
    );
    for &(name, min_r, mean_r, s, e, p) in &best_per_metric {
        println!(
            "{:<14} {:>10.4} {:>12.4}  {:.2}/{:.2}/{:.2}",
            name, min_r, mean_r, s, e, p
        );
    }

    // Pick the formula with the highest min_ratio as the recommended choice.
    let best = best_per_metric
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    if let Some(&(name, min_r, mean_r, s, e, p)) = best {
        // Normalise weights to sum=1.0 for the YAML fragment.
        let total = s + e + p;
        let (yn, ye, yp) = (s / total, e / total, p / total);

        println!("\n── Recommended projection.yaml fragment for `tools` collection ────────────");
        println!("  Formula: {name}  (min_ratio={min_r:.4}  mean_ratio={mean_r:.4})");
        println!("  Raw weights: syn={s:.2} sem={e:.2} prag={p:.2}");
        println!("  Normalised:  syn={yn:.3} sem={ye:.3} prag={yp:.3}");
        println!();

        let (formula_yaml, alpha_line) = if name == "span_a2" {
            ("span", Some("    score_formula_alpha: 2.0"))
        } else if name == "top_k_mean_8" {
            ("top_k_mean", Some("    score_formula_k: 8"))
        } else {
            (name, None)
        };

        println!("  - kind: collection");
        println!("    name: tools");
        println!("    score_formula: {formula_yaml}");
        if let Some(line) = alpha_line {
            println!("{line}");
        }
        println!("    score_threshold: 0.0");
        println!("    selection: {{ kind: top_k, k: 3 }}");
        println!("    depth_weights:");
        println!("      syntactic: {yn:.3}");
        println!("      semantic: {ye:.3}");
        println!("      pragmatic: {yp:.3}");
    }
    println!();
}

// ── threshold + weight calibration ───────────────────────────────────────────

/// Calibrate `score_threshold` and `depth_weights` for the tools collection
/// using real span scores from the corpus.
///
/// Span α=2.0 scores are on a different scale from the old max-agreement
/// formula.  This test measures the actual intra vs inter score distributions,
/// finds the clean separation gap, and derives:
///
///   1. The optimal depth_weights blend (gap-maximising, not just ratio)
///   2. The recommended score_threshold (midpoint of gap, or margin above max
///      inter-score when a clean gap exists)
///
/// Outputs per scenario type (positive/boundary/negative/no_tool) so you can
/// see which cases drive the threshold and which are lost in noise.
///
/// Run with `-- --nocapture`.
#[test]
fn score_threshold_and_weight_calibration() {
    let (manifest, pf) = load_fixtures();
    let h = Harness::build();
    let formula = ScoreFormula::Span { alpha: 2.0 };

    // ── helpers ──────────────────────────────────────────────────────────────

    fn pct(sorted: &[f32], p: f32) -> f32 {
        if sorted.is_empty() {
            return 0.0;
        }
        let idx = ((sorted.len() - 1) as f32 * p).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    fn describe(label: &str, mut v: Vec<f32>) {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if v.is_empty() {
            return;
        }
        println!("    {label:<12} n={:>3}  min={:>7.2}  p25={:>7.2}  p50={:>7.2}  p75={:>7.2}  max={:>7.2}",
            v.len(),
            v[0], pct(&v, 0.25), pct(&v, 0.5), pct(&v, 0.75), v[v.len()-1]);
    }

    // ── scan all probe scenarios once, cache resolvers ────────────────────────
    // Only scan scenarios with a tool label — no_tool probes have no
    // "correct" tool to measure intra-score against.
    let resolvers: Vec<_> = manifest
        .scenarios
        .iter()
        .filter(|s| s.tool.is_some())
        .map(|scen| (scen, h.scan(&pf, &manifest, &scen.id)))
        .collect();

    // ── weight configs to compare ─────────────────────────────────────────────
    let weight_configs: &[(&str, DepthWeights)] = &[
        (
            "equal",
            DepthWeights {
                syntactic: 1.0,
                semantic: 1.0,
                pragmatic: 1.0,
            },
        ),
        (
            "sem+prag",
            DepthWeights {
                syntactic: 0.0,
                semantic: 1.0,
                pragmatic: 1.0,
            },
        ),
        (
            "sem_heavy",
            DepthWeights {
                syntactic: 0.0,
                semantic: 2.0,
                pragmatic: 1.0,
            },
        ),
        (
            "prag_heavy",
            DepthWeights {
                syntactic: 0.0,
                semantic: 1.0,
                pragmatic: 2.0,
            },
        ),
        (
            "prag_only",
            DepthWeights {
                syntactic: 0.0,
                semantic: 0.0,
                pragmatic: 1.0,
            },
        ),
        (
            "sem_only",
            DepthWeights {
                syntactic: 0.0,
                semantic: 1.0,
                pragmatic: 0.0,
            },
        ),
    ];

    // ── per-weight analysis ───────────────────────────────────────────────────
    #[allow(dead_code)]
    struct WeightResult {
        name: &'static str,
        weights: DepthWeights,
        gap: f32,
        threshold: f32,
        min_intra: f32,
        max_inter: f32,
        tp_rate: f32, // fraction of positive intra scores above threshold
        fp_rate: f32, // fraction of inter scores above threshold
    }
    let mut results: Vec<WeightResult> = Vec::new();

    for &(wname, weights) in weight_configs {
        // Collect scores split by scenario type and intra/inter.
        let mut pos_intra: Vec<f32> = Vec::new();
        let mut bnd_intra: Vec<f32> = Vec::new();
        let mut neg_intra: Vec<f32> = Vec::new();
        let mut all_inter: Vec<f32> = Vec::new();

        for (scen, resolver) in &resolvers {
            let probe_tool = scen.tool.as_deref().unwrap();
            let intra = resolver.section_score(h.tool_section_ids[probe_tool], formula, &weights);

            match scen.case_type {
                CaseType::Positive => pos_intra.push(intra),
                CaseType::Boundary => bnd_intra.push(intra),
                CaseType::Negative => neg_intra.push(intra),
                CaseType::NoTool => {}
            }

            for &other in TOOLS.iter().filter(|&&t| t != probe_tool) {
                let inter = resolver.section_score(h.tool_section_ids[other], formula, &weights);
                all_inter.push(inter);
            }
        }

        // Aggregate intra (positive + boundary only — negatives intentionally
        // score low since the probe is NOT asking for that tool).
        let mut intra_gate: Vec<f32> = pos_intra.iter().chain(&bnd_intra).copied().collect();
        intra_gate.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut inter_sorted = all_inter.clone();
        inter_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_intra = intra_gate.first().copied().unwrap_or(0.0);
        let max_inter = inter_sorted.last().copied().unwrap_or(0.0);
        let gap = min_intra - max_inter;

        // Threshold: if clean gap → midpoint; if overlap → just above max_inter.
        let threshold = if gap > 0.0 {
            max_inter + gap * 0.5
        } else {
            // No clean gap: use the 90th percentile of inter as threshold
            // (accepts ~10% false positives, minimises misses).
            pct(&inter_sorted, 0.90) * 1.05
        };

        // TP/FP rates at this threshold.
        let tp_rate = intra_gate.iter().filter(|&&s| s >= threshold).count() as f32
            / intra_gate.len().max(1) as f32;
        let fp_rate = inter_sorted.iter().filter(|&&s| s >= threshold).count() as f32
            / inter_sorted.len().max(1) as f32;

        println!("\n── [{wname}] ──────────────────────────────────────────────────────────");
        println!("  Intra distribution (positive + boundary probes):");
        describe("positive", pos_intra);
        describe("boundary", bnd_intra);
        describe("negative", neg_intra);
        println!("  Inter distribution (all other-tool scores):");
        describe("inter", all_inter);
        if gap > 0.0 {
            println!(
                "  ✓ Clean gap: {gap:.2}  (min_intra={min_intra:.2} − max_inter={max_inter:.2})"
            );
        } else {
            println!(
                "  ✗ Overlap:  {:.2}  (min_intra={min_intra:.2} < max_inter={max_inter:.2})",
                -gap
            );
        }
        println!(
            "  Recommended threshold: {threshold:.2}  →  TP={:.0}%  FP={:.0}%",
            tp_rate * 100.0,
            fp_rate * 100.0
        );

        results.push(WeightResult {
            name: wname,
            weights,
            gap,
            threshold,
            min_intra,
            max_inter,
            tp_rate,
            fp_rate,
        });
    }

    // ── fine grid sweep for gap maximisation ──────────────────────────────────
    println!("\n── Fine grid sweep — maximise gap (min_intra − max_inter) ─────────────");
    println!(
        "{:<10} {:<6} {:<6} {:>8} {:>9} {:>9} {:>8} {:>6}",
        "weights", "syn", "prag", "gap", "min_intra", "max_inter", "thresh", "TP%"
    );

    let levels = [0.0_f32, 0.25, 0.5, 1.0, 1.5, 2.0];
    let mut grid_rows: Vec<(f32, f32, f32, f32, f32, f32, f32, f32)> = Vec::new(); // (gap, sem, prag, min_i, max_i, thresh, tp, fp)

    for &e in &levels {
        for &p in &levels {
            if e + p == 0.0 {
                continue;
            }
            let w = DepthWeights {
                syntactic: 0.0,
                semantic: e,
                pragmatic: p,
            };

            let mut intra_gate: Vec<f32> = Vec::new();
            let mut inter_sorted: Vec<f32> = Vec::new();

            for (scen, resolver) in &resolvers {
                let probe_tool = scen.tool.as_deref().unwrap();
                let intra = resolver.section_score(h.tool_section_ids[probe_tool], formula, &w);
                if matches!(scen.case_type, CaseType::Positive | CaseType::Boundary) {
                    intra_gate.push(intra);
                }
                for &other in TOOLS.iter().filter(|&&t| t != probe_tool) {
                    inter_sorted.push(resolver.section_score(
                        h.tool_section_ids[other],
                        formula,
                        &w,
                    ));
                }
            }

            intra_gate.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            inter_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let min_i = intra_gate.first().copied().unwrap_or(0.0);
            let max_i = inter_sorted.last().copied().unwrap_or(0.0);
            let gap = min_i - max_i;
            let thresh = if gap > 0.0 {
                max_i + gap * 0.5
            } else {
                pct(&inter_sorted, 0.90) * 1.05
            };
            let tp = intra_gate.iter().filter(|&&s| s >= thresh).count() as f32
                / intra_gate.len().max(1) as f32;
            let fp = inter_sorted.iter().filter(|&&s| s >= thresh).count() as f32
                / inter_sorted.len().max(1) as f32;
            grid_rows.push((gap, e, p, min_i, max_i, thresh, tp, fp));
        }
    }
    grid_rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    for &(gap, e, p, min_i, max_i, thresh, tp, _fp) in grid_rows.iter().take(15) {
        let total = e + p;
        let tag = format!("sem={:.2} prag={:.2}", e / total, p / total);
        println!(
            "{:<28} {:>8.2} {:>9.2} {:>9.2} {:>8.2} {:>5.0}%",
            tag,
            gap,
            min_i,
            max_i,
            thresh,
            tp * 100.0
        );
    }

    // ── summary: best config ──────────────────────────────────────────────────
    if let Some(best) = grid_rows.first() {
        let (gap, e, p, _, max_i, thresh, tp, fp) = *best;
        let total = e + p;
        let (sn, sp) = (e / total, p / total);
        println!("\n════════════════════════════════════════════════════════════════════════");
        println!("  RECOMMENDED — tools collection depth_weights and score_threshold");
        println!("  Weights:   semantic={sn:.3}  pragmatic={sp:.3}  (syntactic=0.0)");
        println!(
            "  Gap:       {gap:.2}  (min_intra={:.2} − max_inter={max_i:.2})",
            best.3
        );
        println!(
            "  Threshold: {thresh:.2}  → TP={:.0}%  FP={:.0}%",
            tp * 100.0,
            fp * 100.0
        );
        println!();
        println!("  depth_weights:");
        println!("    syntactic: 0.0");
        println!("    semantic:  {sn:.3}");
        println!("    pragmatic: {sp:.3}");
        println!("  score_threshold: {thresh:.2}");
        println!("════════════════════════════════════════════════════════════════════════");
    }
    println!();

    // Assert: the best config achieves TP >= 90%.  A clean gap (min_intra > max_inter) is
    // desirable but unachievable here — some probes genuinely touch multiple tool domains,
    // so a small fraction of wrong-tool inter scores land above the intra floor.  TopK
    // selection tolerates this: the correct tool still ranks first by a wide margin.
    assert!(
        results.iter().any(|r| r.tp_rate >= 0.9),
        "Expected at least one weight config to achieve TP >= 90%"
    );
}

// ── Multi-layer calibration sweep ─────────────────────────────────────────────

/// Calibrate score_threshold and depth_weights for all 8 cognitive layers.
///
/// Uses real KV/Q data from `<type>_provenance_real_data/` when available
/// (generated by `gen_real_layer_provenance_data`), falling back to the
/// synthetic corpus otherwise.  With real data the MH_XOR_QQ_l0xl4 sign-bit
/// signatures carry genuine depth differentiation, so depth_weights are
/// meaningfully calibrated.  With synthetic data all three bands are
/// identical and equal weights always win — only score_threshold is
/// calibrated.
///
/// Run with `-- --nocapture` to see the calibration table.
#[test]
fn multi_layer_calibration_sweep() {
    // (real_dir, synthetic_dir, display_name, item_list)
    const LAYERS: &[(&str, &str, &str, &[&str])] = &[
        (
            "code_reading_provenance_real_data",
            "code_reading_provenance_data",
            "code_reading",
            &[
                "decode_step",
                "kv_arena",
                "attention_fwd",
                "quant_block",
                "bdp_scan",
                "moe_route",
                "prefill_run",
                "rope_enc",
            ],
        ),
        (
            "static_analysis_provenance_real_data",
            "static_analysis_provenance_data",
            "static_analysis",
            &[
                "cache_rs",
                "arena_rs",
                "compress_rs",
                "scan_rs",
                "scheduler_rs",
                "projection_rs",
                "engine_rs",
                "config_rs",
            ],
        ),
        (
            "dependency_analysis_provenance_real_data",
            "dependency_analysis_provenance_data",
            "dependency_analysis",
            &[
                "cache_deps",
                "arena_deps",
                "compress_deps",
                "scan_deps",
                "scheduler_deps",
                "projection_deps",
                "engine_deps",
                "config_deps",
            ],
        ),
        (
            "architectural_analysis_provenance_real_data",
            "architectural_analysis_provenance_data",
            "architectural_analysis",
            &[
                "paged_kv",
                "quant_policy",
                "bdp_retrieval",
                "moe_predict",
                "wave_batch",
                "three_tier",
                "o1_theorem",
                "proj_schema",
            ],
        ),
        (
            "critical_analysis_provenance_real_data",
            "critical_analysis_provenance_data",
            "critical_analysis",
            &[
                "kv_frag",
                "quant_drift",
                "bdp_collision",
                "sched_block",
                "mem_pressure",
                "dtype_mismatch",
                "attn_overflow",
                "moe_imbalance",
            ],
        ),
        (
            "bug_analysis_provenance_real_data",
            "bug_analysis_provenance_data",
            "bug_analysis",
            &[
                "chunk_oob",
                "q4_sign",
                "kv_misalign",
                "sink_scale",
                "mask_skip",
                "arena_leak",
                "dtype_cast",
                "flash_oob",
            ],
        ),
        (
            "daily_history_provenance_real_data",
            "daily_history_provenance_data",
            "daily_history",
            &[
                "day_kv",
                "day_quant",
                "day_bdp",
                "day_moe",
                "day_proj",
                "day_calib",
                "day_bugfix",
                "day_batch",
            ],
        ),
        (
            "dream_log_provenance_real_data",
            "dream_log_provenance_data",
            "dream_log",
            &[
                "dream_distrib",
                "dream_neural",
                "dream_stream",
                "dream_sinks",
                "dream_prefetch",
                "dream_cluster",
                "dream_dynwin",
                "dream_fedkv",
            ],
        ),
    ];

    let formula = ScoreFormula::Span { alpha: 2.0 };

    fn pct(sorted: &[f32], p: f32) -> f32 {
        if sorted.is_empty() {
            return 0.0;
        }
        let idx = ((sorted.len() - 1) as f32 * p).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    // Full band-combination sweep for real data.
    let weight_configs: &[(&str, DepthWeights)] = &[
        (
            "equal",
            DepthWeights {
                syntactic: 1.0,
                semantic: 1.0,
                pragmatic: 1.0,
            },
        ),
        (
            "syn_only",
            DepthWeights {
                syntactic: 1.0,
                semantic: 0.0,
                pragmatic: 0.0,
            },
        ),
        (
            "sem_only",
            DepthWeights {
                syntactic: 0.0,
                semantic: 1.0,
                pragmatic: 0.0,
            },
        ),
        (
            "prag_only",
            DepthWeights {
                syntactic: 0.0,
                semantic: 0.0,
                pragmatic: 1.0,
            },
        ),
        (
            "syn+sem",
            DepthWeights {
                syntactic: 1.0,
                semantic: 1.0,
                pragmatic: 0.0,
            },
        ),
        (
            "syn+prag",
            DepthWeights {
                syntactic: 1.0,
                semantic: 0.0,
                pragmatic: 1.0,
            },
        ),
        (
            "sem+prag",
            DepthWeights {
                syntactic: 0.0,
                semantic: 1.0,
                pragmatic: 1.0,
            },
        ),
        (
            "prag_heavy",
            DepthWeights {
                syntactic: 0.0,
                semantic: 1.0,
                pragmatic: 2.0,
            },
        ),
    ];

    fn mean_f(v: &[f32]) -> f32 {
        if v.is_empty() {
            return 0.0;
        }
        v.iter().sum::<f32>() / v.len() as f32
    }

    /// Count non-zero weights — used as tie-break (prefer more active bands).
    fn band_count(w: &DepthWeights) -> u32 {
        (w.syntactic > 0.0) as u32 + (w.semantic > 0.0) as u32 + (w.pragmatic > 0.0) as u32
    }

    struct ConfigResult {
        name: &'static str,
        weights: DepthWeights,
        ratio: f32, // mean_intra / mean_inter
        mean_intra: f32,
        mean_inter: f32,
        p10_intra: f32,
        p90_inter: f32,
        rob_gap: f32,   // p10_intra - p90_inter
        threshold: f32, // p95_inter — conservative cut
        tp: f32,
        fp: f32,
    }
    struct LayerResult {
        name: String,
        is_real: bool,
        configs: Vec<ConfigResult>,
        chosen: usize,
    }
    let mut layer_results: Vec<LayerResult> = Vec::new();

    for &(real_dir, synth_dir, display, items) in LAYERS {
        // Prefer real data; fall back to synthetic.
        let real_path = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/"))
            .join(real_dir)
            .join("MANIFEST.json");
        let (dir, is_real) = if real_path.exists() {
            (real_dir, true)
        } else {
            (synth_dir, false)
        };
        let (manifest, pf) = load_layer_fixtures(dir);

        // Assign a SectionId per item (1-based, locally scoped to this layer).
        let section_ids: Vec<SectionId> = (1u32..=items.len() as u32).map(SectionId::new).collect();
        let item_to_sid: HashMap<&str, SectionId> = items
            .iter()
            .copied()
            .zip(section_ids.iter().copied())
            .collect();

        let probes: Vec<&crate::corpus::LayerScenario> = manifest
            .scenarios
            .iter()
            .filter(|s| {
                s.item.is_some()
                    && matches!(
                        s.case_type,
                        LayerCaseType::Positive | LayerCaseType::Boundary
                    )
            })
            .collect();

        // Cache per-probe raw PerDepthScores (independent of weights).
        struct ProbeScores {
            probe_sid: SectionId,
            scores_by_sid: HashMap<SectionId, candle_conversation::projection::PerDepthScores>,
        }
        let mut probe_score_cache: Vec<ProbeScores> = Vec::with_capacity(probes.len());

        for probe in &probes {
            let probe_item = probe.item.as_deref().unwrap();
            let probe_sid = item_to_sid[probe_item];

            let (probe_syn, probe_sem, probe_prag) = pf
                .read_entry(SigEntry {
                    byte_offset: probe.byte_offset,
                    token_count: probe.token_count,
                })
                .expect("read probe sigs failed");

            let corpus: Vec<(SectionId, Vec<SigEntry>)> = items
                .iter()
                .copied()
                .zip(section_ids.iter().copied())
                .map(|(item, sid)| {
                    let entries: Vec<SigEntry> = manifest
                        .scenarios
                        .iter()
                        .filter(|s| {
                            s.item.as_deref() == Some(item)
                                && s.case_type == LayerCaseType::Positive
                                && s.id != probe.id
                        })
                        .map(|s| SigEntry {
                            byte_offset: s.byte_offset,
                            token_count: s.token_count,
                        })
                        .collect();
                    (sid, entries)
                })
                .collect();

            let mut scanner = BdpScanner::new();
            scanner
                .scan_sections(&pf, &probe_syn, &probe_sem, &probe_prag, &corpus)
                .expect("scan_sections failed");

            let scores_by_sid: HashMap<SectionId, _> = scanner
                .section_scores()
                .iter()
                .map(|(&sid, &s)| (sid, s))
                .collect();
            probe_score_cache.push(ProbeScores {
                probe_sid,
                scores_by_sid,
            });
        }

        // Sweep all weight configs; pick best by mean_intra/mean_inter ratio,
        // tie-break by band count (prefer more active bands when SNR is similar).
        let configs_to_sweep: &[(&str, DepthWeights)] = if is_real {
            weight_configs
        } else {
            // Synthetic data: all bands identical, only equal weights make sense.
            &weight_configs[..1]
        };

        let mut configs: Vec<ConfigResult> = Vec::with_capacity(configs_to_sweep.len());

        for &(wname, weights) in configs_to_sweep {
            let mut intra: Vec<f32> = Vec::new();
            let mut inter: Vec<f32> = Vec::new();

            for ps in &probe_score_cache {
                for (&sid, &s) in &ps.scores_by_sid {
                    let score = weights.combine(
                        s.syn.pick(formula),
                        s.sem.pick(formula),
                        s.prag.pick(formula),
                    );
                    if sid == ps.probe_sid {
                        intra.push(score);
                    } else {
                        inter.push(score);
                    }
                }
            }

            intra.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            inter.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let mean_intra = mean_f(&intra);
            let mean_inter = mean_f(&inter);
            let ratio = if mean_inter > 0.0 {
                mean_intra / mean_inter
            } else {
                1.0
            };
            let p10_intra = pct(&intra, 0.10);
            let p90_inter = pct(&inter, 0.90);
            let rob_gap = p10_intra - p90_inter;
            // Conservative threshold: 95th-percentile of inter-class scores.
            // Only ~5% of background items pass, keeping FP manageable.
            let threshold = pct(&inter, 0.95);
            let tp = intra.iter().filter(|&&s| s >= threshold).count() as f32
                / intra.len().max(1) as f32;
            let fp = inter.iter().filter(|&&s| s >= threshold).count() as f32
                / inter.len().max(1) as f32;

            configs.push(ConfigResult {
                name: wname,
                weights,
                ratio,
                mean_intra,
                mean_inter,
                p10_intra,
                p90_inter,
                rob_gap,
                threshold,
                tp,
                fp,
            });
        }

        // Choose: highest ratio; tie-break by number of active bands (more = better).
        let chosen = configs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.ratio
                    .partial_cmp(&b.ratio)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| band_count(&a.weights).cmp(&band_count(&b.weights)))
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Per-layer detail table.
        println!(
            "\n── {display} {} ──",
            if is_real { "(real)" } else { "(synth)" }
        );
        println!(
            "  {:<12} {:>7} {:>11} {:>11} {:>10} {:>10} {:>10} {:>10} {:>5} {:>5}",
            "weights",
            "ratio",
            "mean_intra",
            "mean_inter",
            "p10_intra",
            "p90_inter",
            "rob_gap",
            "threshold",
            "TP%",
            "FP%"
        );
        println!("  {}", "-".repeat(103));
        for (i, c) in configs.iter().enumerate() {
            let marker = if i == chosen { "* " } else { "  " };
            println!("  {marker}{:<10} {:>7.3} {:>11.1} {:>11.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>4.0}% {:>4.0}%",
                c.name, c.ratio, c.mean_intra, c.mean_inter,
                c.p10_intra, c.p90_inter, c.rob_gap, c.threshold,
                c.tp * 100.0, c.fp * 100.0);
        }
        let ch = &configs[chosen];
        let total = ch.weights.syntactic + ch.weights.semantic + ch.weights.pragmatic;
        let (yn, ye, yp) = if total > 0.0 {
            (
                ch.weights.syntactic / total,
                ch.weights.semantic / total,
                ch.weights.pragmatic / total,
            )
        } else {
            (0.333_f32, 0.333_f32, 0.334_f32)
        };
        println!(
            "  → chosen: {}  threshold={:.2}  syn:{:.3} sem:{:.3} prag:{:.3}",
            ch.name, ch.threshold, yn, ye, yp
        );

        layer_results.push(LayerResult {
            name: display.to_string(),
            is_real,
            configs,
            chosen,
        });
    }

    // Summary table.
    println!("\n{}", "=".repeat(110));
    println!("Calibration summary  (chosen config per layer)");
    println!(
        "{:<28} {:<12} {:>7} {:>10} {:>5} {:>5}  depth_weights (syn/sem/prag)",
        "layer", "weights", "ratio", "threshold", "TP%", "FP%"
    );
    println!("{}", "-".repeat(110));
    for r in &layer_results {
        let c = &r.configs[r.chosen];
        let src = if r.is_real { "real" } else { "synth" };
        let total = c.weights.syntactic + c.weights.semantic + c.weights.pragmatic;
        let (yn, ye, yp) = if total > 0.0 {
            (
                c.weights.syntactic / total,
                c.weights.semantic / total,
                c.weights.pragmatic / total,
            )
        } else {
            (0.333_f32, 0.333_f32, 0.334_f32)
        };
        println!(
            "  {:<26}  {:<12} {:>7.3} {:>10.2} {:>4.0}% {:>4.0}%  {:.3}/{:.3}/{:.3}  ({})",
            r.name,
            c.name,
            c.ratio,
            c.threshold,
            c.tp * 100.0,
            c.fp * 100.0,
            yn,
            ye,
            yp,
            src
        );
    }
    println!("{}", "=".repeat(110));
}

/// Derive conservative score_threshold for each layer using its calibrated
/// depth_weights (from the 2026-05-16 cross-corpus sweep stored in projection.yaml).
///
/// For each layer, computes the 95th percentile of inter-class Span α=2.0 scores
/// over all positive+boundary probes.  That threshold means ~5% of background items
/// pass (FP≈5%) while true retrievals are preserved.
///
/// Run with `-- --nocapture` then copy the printed values into each layer's
/// group `score_threshold:` in projection.yaml.
#[test]
fn calibrated_threshold_derivation() {
    let layers: &[(&str, &str, &str, &[&str], DepthWeights)] = &[
        (
            "code_reading_provenance_real_data",
            "code_reading_provenance_data",
            "code_reading",
            &[
                "decode_step",
                "kv_arena",
                "attention_fwd",
                "quant_block",
                "bdp_scan",
                "moe_route",
                "prefill_run",
                "rope_enc",
            ],
            DepthWeights {
                syntactic: 0.0,
                semantic: 3.0,
                pragmatic: 4.0,
            },
        ),
        (
            "static_analysis_provenance_real_data",
            "static_analysis_provenance_data",
            "static_analysis",
            &[
                "cache_rs",
                "arena_rs",
                "compress_rs",
                "scan_rs",
                "scheduler_rs",
                "projection_rs",
                "engine_rs",
                "config_rs",
            ],
            DepthWeights {
                syntactic: 1.0,
                semantic: 0.0,
                pragmatic: 0.0,
            },
        ),
        (
            "dependency_analysis_provenance_real_data",
            "dependency_analysis_provenance_data",
            "dependency_analysis",
            &[
                "cache_deps",
                "arena_deps",
                "compress_deps",
                "scan_deps",
                "scheduler_deps",
                "projection_deps",
                "engine_deps",
                "config_deps",
            ],
            DepthWeights {
                syntactic: 1.0,
                semantic: 0.0,
                pragmatic: 3.0,
            },
        ),
        (
            "architectural_analysis_provenance_real_data",
            "architectural_analysis_provenance_data",
            "architectural_analysis",
            &[
                "paged_kv",
                "quant_policy",
                "bdp_retrieval",
                "moe_predict",
                "wave_batch",
                "three_tier",
                "o1_theorem",
                "proj_schema",
            ],
            DepthWeights {
                syntactic: 3.0,
                semantic: 2.0,
                pragmatic: 0.0,
            },
        ),
        (
            "critical_analysis_provenance_real_data",
            "critical_analysis_provenance_data",
            "critical_analysis",
            &[
                "kv_frag",
                "quant_drift",
                "bdp_collision",
                "sched_block",
                "mem_pressure",
                "dtype_mismatch",
                "attn_overflow",
                "moe_imbalance",
            ],
            DepthWeights {
                syntactic: 0.0,
                semantic: 1.0,
                pragmatic: 0.0,
            },
        ),
        (
            "bug_analysis_provenance_real_data",
            "bug_analysis_provenance_data",
            "bug_analysis",
            &[
                "chunk_oob",
                "q4_sign",
                "kv_misalign",
                "sink_scale",
                "mask_skip",
                "arena_leak",
                "dtype_cast",
                "flash_oob",
            ],
            DepthWeights {
                syntactic: 1.0,
                semantic: 0.0,
                pragmatic: 0.0,
            },
        ),
        (
            "daily_history_provenance_real_data",
            "daily_history_provenance_data",
            "daily_history",
            &[
                "day_kv",
                "day_quant",
                "day_bdp",
                "day_moe",
                "day_proj",
                "day_calib",
                "day_bugfix",
                "day_batch",
            ],
            DepthWeights {
                syntactic: 1.0,
                semantic: 3.0,
                pragmatic: 0.0,
            },
        ),
        (
            "dream_log_provenance_real_data",
            "dream_log_provenance_data",
            "dream_log",
            &[
                "dream_distrib",
                "dream_neural",
                "dream_stream",
                "dream_sinks",
                "dream_prefetch",
                "dream_cluster",
                "dream_dynwin",
                "dream_fedkv",
            ],
            DepthWeights {
                syntactic: 3.0,
                semantic: 1.0,
                pragmatic: 2.0,
            },
        ),
    ];

    let formula = ScoreFormula::Span { alpha: 2.0 };

    fn pct(sorted: &[f32], p: f32) -> f32 {
        if sorted.is_empty() {
            return 0.0;
        }
        let idx = ((sorted.len() - 1) as f32 * p).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
    fn mean_f(v: &[f32]) -> f32 {
        if v.is_empty() {
            return 0.0;
        }
        v.iter().sum::<f32>() / v.len() as f32
    }

    struct LayerThreshold {
        name: String,
        threshold: f32,
        mean_intra: f32,
        mean_inter: f32,
        ratio: f32,
        tp: f32,
        fp: f32,
        is_real: bool,
    }
    let mut summary: Vec<LayerThreshold> = Vec::new();

    for &(real_dir, synth_dir, display, items, weights) in layers {
        let real_path = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/"))
            .join(real_dir)
            .join("MANIFEST.json");
        let (dir, is_real) = if real_path.exists() {
            (real_dir, true)
        } else {
            (synth_dir, false)
        };
        let (manifest, pf) = load_layer_fixtures(dir);

        let section_ids: Vec<SectionId> = (1u32..=items.len() as u32).map(SectionId::new).collect();
        let item_to_sid: HashMap<&str, SectionId> = items
            .iter()
            .copied()
            .zip(section_ids.iter().copied())
            .collect();

        let probes: Vec<&crate::corpus::LayerScenario> = manifest
            .scenarios
            .iter()
            .filter(|s| {
                s.item.is_some()
                    && matches!(
                        s.case_type,
                        LayerCaseType::Positive | LayerCaseType::Boundary
                    )
            })
            .collect();

        let mut intra: Vec<f32> = Vec::new();
        let mut inter: Vec<f32> = Vec::new();

        for probe in &probes {
            let probe_item = probe.item.as_deref().unwrap();
            let probe_sid = item_to_sid[probe_item];

            let (probe_syn, probe_sem, probe_prag) = pf
                .read_entry(SigEntry {
                    byte_offset: probe.byte_offset,
                    token_count: probe.token_count,
                })
                .expect("read probe sigs failed");

            let corpus: Vec<(SectionId, Vec<SigEntry>)> = items
                .iter()
                .copied()
                .zip(section_ids.iter().copied())
                .map(|(item, sid)| {
                    let entries: Vec<SigEntry> = manifest
                        .scenarios
                        .iter()
                        .filter(|s| {
                            s.item.as_deref() == Some(item)
                                && s.case_type == LayerCaseType::Positive
                                && s.id != probe.id
                        })
                        .map(|s| SigEntry {
                            byte_offset: s.byte_offset,
                            token_count: s.token_count,
                        })
                        .collect();
                    (sid, entries)
                })
                .collect();

            let mut scanner = BdpScanner::new();
            scanner
                .scan_sections(&pf, &probe_syn, &probe_sem, &probe_prag, &corpus)
                .expect("scan_sections failed");

            for (&sid, &s) in scanner.section_scores().iter() {
                let score = weights.combine(
                    s.syn.pick(formula),
                    s.sem.pick(formula),
                    s.prag.pick(formula),
                );
                if sid == probe_sid {
                    intra.push(score);
                } else {
                    inter.push(score);
                }
            }
        }

        intra.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        inter.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let threshold = pct(&inter, 0.95);
        let mean_intra = mean_f(&intra);
        let mean_inter = mean_f(&inter);
        let ratio = if mean_inter > 0.0 {
            mean_intra / mean_inter
        } else {
            1.0
        };
        let tp =
            intra.iter().filter(|&&s| s >= threshold).count() as f32 / intra.len().max(1) as f32;
        let fp =
            inter.iter().filter(|&&s| s >= threshold).count() as f32 / inter.len().max(1) as f32;

        // Percentile table so we can see where the distributions actually sit.
        let pcts = [
            0.0_f32, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.0,
        ];
        println!(
            "\n── {} {} ──",
            display,
            if is_real { "(real)" } else { "(synth)" }
        );
        println!("  {:>5}  {:>10}  {:>10}", "pct", "intra", "inter");
        for &p in &pcts {
            let vi = pct(&intra, p);
            let ve = pct(&inter, p);
            // Mark rows where intra crosses inter (potential threshold zone).
            let marker = if vi > ve { " ←" } else { "" };
            println!(
                "  {:>4.0}%  {:>10.1}  {:>10.1}{}",
                p * 100.0,
                vi,
                ve,
                marker
            );
        }
        println!(
            "  mean   {:>10.1}  {:>10.1}  ratio={:.3}",
            mean_intra, mean_inter, ratio
        );

        summary.push(LayerThreshold {
            name: display.to_string(),
            threshold,
            mean_intra,
            mean_inter,
            ratio,
            tp,
            fp,
            is_real,
        });
    }

    // ── Tools collection calibration ──────────────────────────────────────────
    // Mirror the cognitive-layer calibration for the tools collection.
    // Uses prag_only depth weights (from score_threshold_and_weight_calibration).
    // Prefers tool_provenance_real_data; falls back to tool_provenance_data.
    {
        let real_tool_path =
            std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/"))
                .join("tool_provenance_real_data")
                .join("MANIFEST.json");
        let (tool_dir, tool_is_real) = if real_tool_path.exists() {
            ("tool_provenance_real_data", true)
        } else {
            ("tool_provenance_data", false)
        };

        let tool_dir_path =
            std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/")).join(tool_dir);
        let tool_json = std::fs::read_to_string(tool_dir_path.join("MANIFEST.json"))
            .unwrap_or_else(|e| panic!("{}/MANIFEST.json not found: {}", tool_dir, e));
        let tool_manifest: crate::corpus::Manifest = serde_json::from_str(&tool_json)
            .unwrap_or_else(|e| panic!("{}/MANIFEST.json parse failed: {}", tool_dir, e));
        let tool_pf = candle_conversation::provenance::ProvenanceFile::open(
            tool_dir_path.join("signatures.prov"),
        )
        .unwrap_or_else(|e| panic!("{}/signatures.prov open failed: {}", tool_dir, e));

        let tools_weights = DepthWeights {
            syntactic: 0.0,
            semantic: 0.0,
            pragmatic: 1.0,
        };

        // Assign SectionId 1..=8 for the 8 tools.
        let tool_names = crate::corpus::TOOLS;
        let tool_to_sid: HashMap<&str, SectionId> = tool_names
            .iter()
            .enumerate()
            .map(|(i, &t)| (t, SectionId::new((i + 1) as u32)))
            .collect();

        let mut tool_intra: Vec<f32> = Vec::new();
        let mut tool_inter: Vec<f32> = Vec::new();

        for probe in tool_manifest.scenarios.iter().filter(|s| {
            s.tool.is_some()
                && matches!(
                    s.case_type,
                    crate::corpus::CaseType::Positive | crate::corpus::CaseType::Boundary
                )
        }) {
            let probe_tool = probe.tool.as_deref().unwrap();
            let probe_sid = tool_to_sid[probe_tool];

            let (probe_syn, probe_sem, probe_prag) = tool_pf
                .read_entry(SigEntry {
                    byte_offset: probe.byte_offset,
                    token_count: probe.token_count,
                })
                .expect("read probe sigs");

            let corpus: Vec<(SectionId, Vec<SigEntry>)> = tool_names
                .iter()
                .map(|&t| {
                    let sid = tool_to_sid[t];
                    let entries: Vec<SigEntry> = tool_manifest
                        .scenarios
                        .iter()
                        .filter(|s| {
                            s.tool.as_deref() == Some(t)
                                && s.case_type == crate::corpus::CaseType::Positive
                                && s.id != probe.id
                        })
                        .map(|s| SigEntry {
                            byte_offset: s.byte_offset,
                            token_count: s.token_count,
                        })
                        .collect();
                    (sid, entries)
                })
                .collect();

            let mut scanner = BdpScanner::new();
            scanner
                .scan_sections(&tool_pf, &probe_syn, &probe_sem, &probe_prag, &corpus)
                .expect("scan_sections failed");

            for (&sid, &s) in scanner.section_scores().iter() {
                let score = tools_weights.combine(
                    s.syn.pick(formula),
                    s.sem.pick(formula),
                    s.prag.pick(formula),
                );
                if sid == probe_sid {
                    tool_intra.push(score);
                } else {
                    tool_inter.push(score);
                }
            }
        }

        tool_intra.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        tool_inter.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let tool_threshold = pct(&tool_inter, 0.95);
        let tool_mean_intra = mean_f(&tool_intra);
        let tool_mean_inter = mean_f(&tool_inter);
        let tool_ratio = if tool_mean_inter > 0.0 {
            tool_mean_intra / tool_mean_inter
        } else {
            1.0
        };
        let tool_tp = tool_intra.iter().filter(|&&s| s >= tool_threshold).count() as f32
            / tool_intra.len().max(1) as f32;
        let tool_fp = tool_inter.iter().filter(|&&s| s >= tool_threshold).count() as f32
            / tool_inter.len().max(1) as f32;

        let pcts = [
            0.0_f32, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.0,
        ];
        println!(
            "\n── tools {} ──",
            if tool_is_real { "(real)" } else { "(synth)" }
        );
        println!("  {:>5}  {:>10}  {:>10}", "pct", "intra", "inter");
        for &p in &pcts {
            let vi = pct(&tool_intra, p);
            let ve = pct(&tool_inter, p);
            let marker = if vi > ve { " ←" } else { "" };
            println!(
                "  {:>4.0}%  {:>10.1}  {:>10.1}{}",
                p * 100.0,
                vi,
                ve,
                marker
            );
        }
        println!(
            "  mean   {:>10.1}  {:>10.1}  ratio={:.3}",
            tool_mean_intra, tool_mean_inter, tool_ratio
        );

        summary.push(LayerThreshold {
            name: "tools".to_string(),
            threshold: tool_threshold,
            mean_intra: tool_mean_intra,
            mean_inter: tool_mean_inter,
            ratio: tool_ratio,
            tp: tool_tp,
            fp: tool_fp,
            is_real: tool_is_real,
        });
    }

    // ── Full detail table ─────────────────────────────────────────────────────
    let sep = "=".repeat(96);
    let dash = "-".repeat(96);
    println!("\n{sep}");
    println!("Calibrated threshold derivation  |  p95_inter  |  Span α=2.0");
    println!(
        "{:<28} {:>10} {:>10} {:>10} {:>7} {:>7} {:>7}",
        "layer", "threshold", "mean_intra", "mean_inter", "ratio", "TP%", "FP%"
    );
    println!("{dash}");
    for lt in &summary {
        let src = if lt.is_real { "real" } else { "synth" };
        println!(
            "  {:<26}  {:>10.2}  {:>10.1}  {:>10.1}  {:>7.3}  {:>5.0}%  {:>5.0}%  ({})",
            lt.name,
            lt.threshold,
            lt.mean_intra,
            lt.mean_inter,
            lt.ratio,
            lt.tp * 100.0,
            lt.fp * 100.0,
            src
        );
    }
    println!("{sep}");

    // ── Ready-to-paste YAML block ─────────────────────────────────────────────
    println!("\n  Copy into each layer's group score_threshold in projection.yaml:\n");
    for lt in &summary {
        println!(
            "  {:<30}  score_threshold: {:>10.2}   # TP={:.0}% FP={:.0}% (calibrated 2026-05-16)",
            lt.name,
            lt.threshold,
            lt.tp * 100.0,
            lt.fp * 100.0
        );
    }
    println!();
}

/// Cross-corpus provenance sweep: all 64 items (8 content types × 8 items) are pooled
/// into one candidate list.  For each probe scenario the full 64-item combined corpus
/// is scanned; a hit means the probe's own item is the top-ranked result.
///
/// Sweeps ≥30 distinct weight combinations (grid {0..=4}^3 over syntactic / semantic /
/// pragmatic bands, Span α=2.0) and reports MRR + top-1 accuracy per content type.
///
/// Expected runtime: ~25–35 min.  Run with `-- --nocapture` for live progress.
///
/// Ignored by default — far too slow for routine runs.  Invoke explicitly:
/// `cargo test -p candle-conversation --test projection_harness
///  cross_corpus_provenance_sweep -- --ignored --nocapture`.
#[test]
#[ignore = "long-running calibration sweep (~25-35 min); run explicitly with --ignored"]
fn cross_corpus_provenance_sweep() {
    const LAYERS: &[(&str, &str, &str, &[&str])] = &[
        (
            "code_reading_provenance_real_data",
            "code_reading_provenance_data",
            "code_reading",
            &[
                "decode_step",
                "kv_arena",
                "attention_fwd",
                "quant_block",
                "bdp_scan",
                "moe_route",
                "prefill_run",
                "rope_enc",
            ],
        ),
        (
            "static_analysis_provenance_real_data",
            "static_analysis_provenance_data",
            "static_analysis",
            &[
                "cache_rs",
                "arena_rs",
                "compress_rs",
                "scan_rs",
                "scheduler_rs",
                "projection_rs",
                "engine_rs",
                "config_rs",
            ],
        ),
        (
            "dependency_analysis_provenance_real_data",
            "dependency_analysis_provenance_data",
            "dependency_analysis",
            &[
                "cache_deps",
                "arena_deps",
                "compress_deps",
                "scan_deps",
                "scheduler_deps",
                "projection_deps",
                "engine_deps",
                "config_deps",
            ],
        ),
        (
            "architectural_analysis_provenance_real_data",
            "architectural_analysis_provenance_data",
            "architectural_analysis",
            &[
                "paged_kv",
                "quant_policy",
                "bdp_retrieval",
                "moe_predict",
                "wave_batch",
                "three_tier",
                "o1_theorem",
                "proj_schema",
            ],
        ),
        (
            "critical_analysis_provenance_real_data",
            "critical_analysis_provenance_data",
            "critical_analysis",
            &[
                "kv_frag",
                "quant_drift",
                "bdp_collision",
                "sched_block",
                "mem_pressure",
                "dtype_mismatch",
                "attn_overflow",
                "moe_imbalance",
            ],
        ),
        (
            "bug_analysis_provenance_real_data",
            "bug_analysis_provenance_data",
            "bug_analysis",
            &[
                "chunk_oob",
                "q4_sign",
                "kv_misalign",
                "sink_scale",
                "mask_skip",
                "arena_leak",
                "dtype_cast",
                "flash_oob",
            ],
        ),
        (
            "daily_history_provenance_real_data",
            "daily_history_provenance_data",
            "daily_history",
            &[
                "day_kv",
                "day_quant",
                "day_bdp",
                "day_moe",
                "day_proj",
                "day_calib",
                "day_bugfix",
                "day_batch",
            ],
        ),
        (
            "dream_log_provenance_real_data",
            "dream_log_provenance_data",
            "dream_log",
            &[
                "dream_distrib",
                "dream_neural",
                "dream_stream",
                "dream_sinks",
                "dream_prefetch",
                "dream_cluster",
                "dream_dynwin",
                "dream_fedkv",
            ],
        ),
    ];

    let formula = ScoreFormula::Span { alpha: 2.0 };
    let n_layers = LAYERS.len();

    // ── Load manifests + provenance files (prefer real data) ─────────────────
    let mut layer_data = Vec::with_capacity(n_layers);
    for &(real_dir, synth_dir, _, _) in LAYERS {
        let real_path = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/"))
            .join(real_dir)
            .join("MANIFEST.json");
        let dir = if real_path.exists() {
            real_dir
        } else {
            synth_dir
        };
        layer_data.push(load_layer_fixtures(dir));
    }

    // ── Build per-layer corpus ───────────────────────────────────────────────
    // Global SectionId: layer_idx * 8 + item_idx + 1  (1-based, 1..=64)
    let corpus_per_layer: Vec<Vec<(SectionId, Vec<SigEntry>)>> = LAYERS
        .iter()
        .enumerate()
        .map(|(li, &(_, _, _, items))| {
            let manifest = &layer_data[li].0;
            items
                .iter()
                .enumerate()
                .map(|(ii, &item)| {
                    let sid = SectionId::new((li * 8 + ii + 1) as u32);
                    let entries: Vec<SigEntry> = manifest
                        .scenarios
                        .iter()
                        .filter(|s| {
                            s.item.as_deref() == Some(item)
                                && s.case_type == LayerCaseType::Positive
                        })
                        .map(|s| SigEntry {
                            byte_offset: s.byte_offset,
                            token_count: s.token_count,
                        })
                        .collect();
                    (sid, entries)
                })
                .collect()
        })
        .collect();

    // ── Collect all probes (positive + boundary from every layer) ────────────
    struct ProbeDef {
        layer_idx: usize,
        probe_sid: SectionId,
        sig_entry: SigEntry,
    }
    let mut probes: Vec<ProbeDef> = Vec::new();
    for (li, &(_, _, _, items)) in LAYERS.iter().enumerate() {
        for (ii, &item) in items.iter().enumerate() {
            let probe_sid = SectionId::new((li * 8 + ii + 1) as u32);
            for s in layer_data[li].0.scenarios.iter().filter(|s| {
                s.item.as_deref() == Some(item)
                    && matches!(
                        s.case_type,
                        LayerCaseType::Positive | LayerCaseType::Boundary
                    )
            }) {
                probes.push(ProbeDef {
                    layer_idx: li,
                    probe_sid,
                    sig_entry: SigEntry {
                        byte_offset: s.byte_offset,
                        token_count: s.token_count,
                    },
                });
            }
        }
    }
    let n_probes = probes.len();
    eprintln!("\n[cross_corpus] {n_probes} probes × 64 items — scanning (this takes ~25 min)...");

    // ── Scan all probes against full 64-item corpus, cache raw per-depth scores ──
    // Each probe: 8 separate scan_sections calls (one per corpus layer's .prov file).
    struct CachedScores {
        layer_idx: usize,
        probe_sid: SectionId,
        scores: HashMap<SectionId, PerDepthScores>,
    }
    let mut cached: Vec<CachedScores> = Vec::with_capacity(n_probes);
    let mut scanner = BdpScanner::new();
    for (pi, probe) in probes.iter().enumerate() {
        if pi % 100 == 0 {
            eprintln!("  probe {pi}/{n_probes}");
        }

        let (probe_syn, probe_sem, probe_prag) = layer_data[probe.layer_idx]
            .1
            .read_entry(probe.sig_entry)
            .expect("read probe sigs");

        let mut scores: HashMap<SectionId, PerDepthScores> = HashMap::with_capacity(64);
        for (li, corpus) in corpus_per_layer.iter().enumerate() {
            scanner
                .scan_sections(
                    &layer_data[li].1,
                    &probe_syn,
                    &probe_sem,
                    &probe_prag,
                    corpus,
                )
                .expect("scan_sections failed");
            for (&sid, &s) in scanner.section_scores().iter() {
                scores.insert(sid, s);
            }
        }
        cached.push(CachedScores {
            layer_idx: probe.layer_idx,
            probe_sid: probe.probe_sid,
            scores,
        });
    }
    eprintln!(
        "[cross_corpus] scanning complete; sweeping {} weight configs...",
        {
            // Count distinct configs (computed below) — just print after sweep.
            0
        }
    );

    // ── Generate weight configs: grid {0..=4}^3, deduplicated by normalised value ──
    let weight_configs: Vec<DepthWeights> = {
        let mut seen = std::collections::HashSet::<(u32, u32, u32)>::new();
        let mut cfgs = Vec::new();
        for a in 0u32..=4 {
            for b in 0u32..=4 {
                for c in 0u32..=4 {
                    let s = a + b + c;
                    if s == 0 {
                        continue;
                    }
                    // Key = normalised fractions × 1000 (rounded), deduplicates e.g. (1,0,0) and (2,0,0).
                    let key = (a * 1000 / s, b * 1000 / s, c * 1000 / s);
                    if seen.insert(key) {
                        cfgs.push(DepthWeights {
                            syntactic: a as f32,
                            semantic: b as f32,
                            pragmatic: c as f32,
                        });
                    }
                }
            }
        }
        cfgs
    };
    let n_configs = weight_configs.len();
    eprintln!("[cross_corpus] {n_configs} distinct weight configs");

    // ── Sweep ────────────────────────────────────────────────────────────────
    struct SweepResult {
        weights: DepthWeights,
        per_layer_top1: Vec<f32>,
        per_layer_mrr: Vec<f32>,
        total_top1: f32,
        total_mrr: f32,
    }
    let mut results: Vec<SweepResult> = Vec::with_capacity(n_configs);

    for &weights in &weight_configs {
        let mut hits1 = vec![0u32; n_layers];
        let mut totals = vec![0u32; n_layers];
        let mut rr_sums = vec![0.0f32; n_layers];

        for cs in &cached {
            let mut scored: Vec<(SectionId, f32)> = cs
                .scores
                .iter()
                .map(|(&sid, &s)| {
                    let score = weights.combine(
                        s.syn.pick(formula),
                        s.sem.pick(formula),
                        s.prag.pick(formula),
                    );
                    (sid, score)
                })
                .collect();
            scored.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            let rank = scored
                .iter()
                .position(|(sid, _)| *sid == cs.probe_sid)
                .map(|i| i + 1)
                .unwrap_or(scored.len() + 1);

            totals[cs.layer_idx] += 1;
            if rank == 1 {
                hits1[cs.layer_idx] += 1;
            }
            rr_sums[cs.layer_idx] += 1.0 / rank as f32;
        }

        let per_layer_top1: Vec<f32> = (0..n_layers)
            .map(|li| hits1[li] as f32 / totals[li].max(1) as f32)
            .collect();
        let per_layer_mrr: Vec<f32> = (0..n_layers)
            .map(|li| rr_sums[li] / totals[li].max(1) as f32)
            .collect();
        let total_n: u32 = totals.iter().sum();
        let total_rr: f32 = rr_sums.iter().sum();
        let total_h: u32 = hits1.iter().sum();

        results.push(SweepResult {
            weights,
            per_layer_top1,
            per_layer_mrr,
            total_top1: total_h as f32 / total_n.max(1) as f32,
            total_mrr: total_rr / total_n.max(1) as f32,
        });
    }

    // Sort by total MRR desc.
    results.sort_unstable_by(|a, b| {
        b.total_mrr
            .partial_cmp(&a.total_mrr)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    fn w_label(w: &DepthWeights) -> String {
        let t = w.syntactic + w.semantic + w.pragmatic;
        if t <= 0.0 {
            return "0/0/0".into();
        }
        format!(
            "{:.2}/{:.2}/{:.2}",
            w.syntactic / t,
            w.semantic / t,
            w.pragmatic / t
        )
    }

    // Abbreviate layer names to 9 chars for table columns.
    let col: Vec<&str> = LAYERS
        .iter()
        .map(|&(_, _, d, _)| &d[..d.len().min(9)])
        .collect();

    let sep = "=".repeat(125);
    let dash = "-".repeat(125);

    // ── MRR table (all configs, best first) ──────────────────────────────────
    println!("\n{sep}");
    println!("Cross-corpus sweep  |  64 items  |  Span α=2.0  |  {n_probes} probes  |  {n_configs} weight configs");
    println!(
        "MRR = Mean Reciprocal Rank (1/rank).  Rank 1 = probe's item is top result out of 64."
    );
    println!("{sep}");

    print!("{:<16}", "syn/sem/prag");
    for c in &col {
        print!("  {:>9}", c);
    }
    println!("  {:>9}", "TOTAL");
    println!("{dash}");

    for r in &results {
        print!("{:<16}", w_label(&r.weights));
        for &m in &r.per_layer_mrr {
            print!("  {:>9.4}", m);
        }
        println!("  {:>9.4}", r.total_mrr);
    }

    // ── Top-1 table ──────────────────────────────────────────────────────────
    println!("\n  Top-1 accuracy (% of probes where correct item is ranked #1 out of 64):");
    print!("{:<16}", "syn/sem/prag");
    for c in &col {
        print!("  {:>8}", c);
    }
    println!("  {:>8}", "TOTAL");
    println!("{}", "-".repeat(110));

    for r in &results {
        print!("{:<16}", w_label(&r.weights));
        for &t in &r.per_layer_top1 {
            print!("  {:>7.1}%", t * 100.0);
        }
        println!("  {:>7.1}%", r.total_top1 * 100.0);
    }

    // ── Best config per layer ─────────────────────────────────────────────────
    println!("\n  Best config per content type (by MRR across {n_configs} weight combos):");
    println!(
        "{:<28}  {:<16}  {:>8}  {:>8}",
        "layer", "syn/sem/prag", "MRR", "Top-1%"
    );
    println!("{}", "-".repeat(68));
    for (li, layer) in LAYERS.iter().enumerate().take(n_layers) {
        let best = results
            .iter()
            .max_by(|a, b| {
                a.per_layer_mrr[li]
                    .partial_cmp(&b.per_layer_mrr[li])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        println!(
            "  {:<26}  {:<16}  {:>8.4}  {:>7.1}%",
            layer.2,
            w_label(&best.weights),
            best.per_layer_mrr[li],
            best.per_layer_top1[li] * 100.0
        );
    }
    println!("{sep}");
}

/// Per-tool score dump for specific probe scenarios.
///
/// For each named probe, prints all 8 tool scores in descending order so
/// you can see exactly which tools the projection engine would select at a
/// given threshold and k.
///
/// Run with `-- --nocapture`.
#[test]
fn tool_score_dump() {
    let (manifest, pf) = crate::corpus::load_fixtures();
    let h = crate::harness::Harness::build();

    // Probes to inspect — the new "use a tool" calculator cases plus two
    // control cases (a classic arithmetic query and a no-tool probe).
    let probe_ids: &[&str] = &[
        "calculator_pos_6", // "Use a tool and determine 14634535 + 623452345."
        "calculator_pos_7", // "Use a tool to compute 9872534 * 3."
        "calculator_pos_8", // "Use a tool to calculate 55^3."
        "calculator_pos_9", // "Use a tool to figure out 2^20 - 1."
        "calculator_pos_0", // "What's 847 divided by 23?"  — classic
        "calculator_pos_1", // "Calculate the square root of 1764."
        "datetime_pos_0",   // "What's the current time in New York?"
        "weather_pos_0",    // "What's the weather like in Seattle today?"
    ];

    let formula = candle_conversation::projection::ScoreFormula::Span { alpha: 2.0 };
    let weights = candle_conversation::projection::DepthWeights {
        syntactic: 0.0,
        semantic: 0.0,
        pragmatic: 1.0,
    };

    let threshold = 140.70_f32;
    let k = 3usize;

    println!("\n{}", "=".repeat(90));
    println!("Per-tool BDP scores  |  Span α=2.0  |  prag_only  |  threshold={threshold}  k={k}");
    println!("{}", "=".repeat(90));

    for &probe_id in probe_ids {
        // Look up the probe to display its user prompt.
        let _probe_text = manifest
            .scenarios
            .iter()
            .find(|s| s.id == probe_id)
            .map(|s| s.id.as_str())
            .unwrap_or(probe_id);

        // Skip probes not in the real manifest (might not exist yet).
        if !manifest.scenarios.iter().any(|s| s.id == probe_id) {
            println!("\n  [skip] {} — not in real manifest", probe_id);
            continue;
        }

        let resolver = h.scan(&pf, &manifest, probe_id);

        // Collect (tool, score) pairs and sort descending.
        let mut scores: Vec<(&str, f32)> = TOOLS
            .iter()
            .map(|&t| {
                let sid = h.tool_section_ids[t];
                let s = resolver
                    .section_scores
                    .get(&sid)
                    .copied()
                    .unwrap_or_default();
                let score = weights.combine(
                    s.syn.pick(formula),
                    s.sem.pick(formula),
                    s.prag.pick(formula),
                );
                (t, score)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        println!("\n  probe: {}", probe_id);
        println!("  {:<14} {:>10}  selected?", "tool", "score");
        println!("  {}", "-".repeat(45));
        let mut selected = 0;
        for (i, (tool, score)) in scores.iter().enumerate() {
            let passes_thresh = *score >= threshold;
            let in_topk = i < k && passes_thresh;
            if passes_thresh {
                selected += 1;
            }
            let marker = if in_topk && selected <= k {
                format!("✓ rank {}", i + 1)
            } else if passes_thresh {
                format!("  rank {} (over threshold but k limit)", i + 1)
            } else {
                format!("  rank {} (below threshold)", i + 1)
            };
            println!("  {:<14} {:>10.1}  {}", tool, score, marker);
        }
        let selected_tools: Vec<&str> = scores
            .iter()
            .take(k)
            .filter(|(_, s)| *s >= threshold)
            .map(|(t, _)| *t)
            .collect();
        println!("  → projected tools: {:?}", selected_tools);
    }
    println!("\n{}", "=".repeat(90));
}

/// Validate that every scenario in both manifests has readable, non-zero data
/// in its backing prov file and that no entry runs past EOF.
///
/// Run with `-- --nocapture` to see the per-scenario report.
#[test]
fn validate_corpus() {
    use candle_conversation::provenance::SigEntry;

    let dir = std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/tool_provenance_real_data",
    ));

    // ── signatures.prov ──────────────────────────────────────────────────────

    let sig_path = dir.join("signatures.prov");
    let sig_file_len = std::fs::metadata(&sig_path)
        .expect("signatures.prov not found")
        .len();

    let (manifest, pf) = crate::corpus::load_fixtures();
    const BYTES_PER_TOKEN: u64 = 3 * 16; // NUM_DEPTHS * TokenSignature::BYTE_LEN

    let mut sig_errors = 0usize;
    let mut sig_ok = 0usize;

    println!(
        "\n── signatures.prov  ({} bytes, {} scenarios) ──",
        sig_file_len,
        manifest.scenarios.len()
    );
    for s in &manifest.scenarios {
        let entry_bytes = s.token_count as u64 * BYTES_PER_TOKEN;
        let end = s.byte_offset + entry_bytes;

        if end > sig_file_len {
            println!(
                "  FAIL  {}  byte_offset={} token_count={} → end={} > file_len={}",
                s.id, s.byte_offset, s.token_count, end, sig_file_len
            );
            sig_errors += 1;
            continue;
        }

        let entry = SigEntry {
            byte_offset: s.byte_offset,
            token_count: s.token_count,
        };
        let (syn, sem, prag) = pf.read_entry(entry).expect("read_entry failed");

        if syn.is_empty() {
            println!("  FAIL  {}  read_entry returned empty vectors", s.id);
            sig_errors += 1;
            continue;
        }

        // Check that at least one signature is non-zero across all three depths.
        let all_zero = syn
            .iter()
            .chain(sem.iter())
            .chain(prag.iter())
            .all(|sig| sig.as_bytes().iter().all(|&b| b == 0));
        if all_zero {
            println!("  FAIL  {}  all signatures are zero", s.id);
            sig_errors += 1;
            continue;
        }

        sig_ok += 1;
    }
    println!("  {} ok, {} failed", sig_ok, sig_errors);

    // ── raw_kvq.prov ─────────────────────────────────────────────────────────

    if let Some((raw_manifest, raw_pf)) = crate::corpus::try_load_raw_fixtures() {
        let raw_path = dir.join("raw_kvq.prov");
        let raw_file_len = std::fs::metadata(&raw_path)
            .expect("raw_kvq.prov not found")
            .len();
        let bpt = raw_pf.header().bytes_per_token() as u64;

        let mut raw_errors = 0usize;
        let mut raw_ok = 0usize;

        println!(
            "\n── raw_kvq.prov  ({} bytes, {} scenarios, bpt={}) ──",
            raw_file_len,
            raw_manifest.scenarios.len(),
            bpt
        );
        for s in &raw_manifest.scenarios {
            let entry_bytes = s.raw_token_count as u64 * bpt;
            let end = s.raw_byte_offset + entry_bytes;

            if end > raw_file_len {
                println!(
                    "  FAIL  {}  raw_byte_offset={} raw_token_count={} → end={} > file_len={}",
                    s.id, s.raw_byte_offset, s.raw_token_count, end, raw_file_len
                );
                raw_errors += 1;
                continue;
            }

            use candle_conversation::provenance::RawSigEntry;
            let entry = RawSigEntry {
                byte_offset: s.raw_byte_offset,
                token_count: s.raw_token_count,
            };
            match raw_pf.read_entry_bytes(entry) {
                Ok(bytes) if bytes.is_empty() => {
                    println!("  FAIL  {}  read_entry_bytes returned empty", s.id);
                    raw_errors += 1;
                }
                Ok(bytes) => {
                    let all_zero = bytes.iter().all(|&b| b == 0);
                    if all_zero {
                        println!("  FAIL  {}  all bytes are zero", s.id);
                        raw_errors += 1;
                    } else {
                        raw_ok += 1;
                    }
                }
                Err(e) => {
                    println!("  FAIL  {}  read_entry_bytes error: {}", s.id, e);
                    raw_errors += 1;
                }
            }
        }
        println!("  {} ok, {} failed", raw_ok, raw_errors);

        assert_eq!(
            raw_errors, 0,
            "{raw_errors} raw corpus entries are corrupt or out-of-bounds"
        );
    }

    assert_eq!(
        sig_errors, 0,
        "{sig_errors} signature corpus entries are corrupt or out-of-bounds"
    );
}
