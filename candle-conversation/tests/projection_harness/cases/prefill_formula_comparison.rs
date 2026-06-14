//! Diagnostic: find the best ScoreFormula + depth-weight config for the
//! **prefill-phase** Q vectors — the initial guess available before any
//! decode probe runs.
//!
//! This mirrors `formula_comparison::content_type_calibration_sweep` but runs
//! against the prefill corpus (`prefill_signatures.prov`) instead of the
//! decode corpus.  The decode sweep is left untouched — `Span { alpha: 2.0 }`
//! is the proven decode champion and is not in question here.
//!
//! Prefill Q vectors have a structurally different signal shape: the model is
//! *reading* the user prompt, so tool-topic signal is concentrated in content
//! words and scattered among low-signal function words rather than forming one
//! coherent run.  A run-rewarding formula (`Span` with high alpha) may
//! therefore mismatch prefill — so the sweep keeps an open mind and includes
//! flat-counting formulas (`Count`, `Sum`, `Span { alpha: 1.0 }`) alongside it.
//!
//! The primary metric is **top-1 rank accuracy** (did the correct tool score
//! highest), because the production question for the prefill round is "which
//! sections do we inject as the initial guess" — a ranking question.  The
//! intra/inter ratio is reported as a secondary signal.
//!
//! Run with `-- --nocapture` to see the tables.

use crate::corpus::{try_load_prefill_fixtures, Manifest, TOOLS};
use crate::harness::Harness;
use candle_conversation::projection::{ContentResolver, DepthWeights, ScoreFormula};
use candle_conversation::provenance::ProvenanceFile;

fn ratio(intra: f32, inter_mean: f32) -> f32 {
    if inter_mean > 0.0 {
        intra / inter_mean
    } else {
        f32::INFINITY
    }
}

/// Per-config evaluation: discrimination ratios + rank accuracy across the
/// 8 probe tools.
struct Eval {
    min_ratio: f32,
    mean_ratio: f32,
    /// Fraction of probe tools whose own section scored strictly highest.
    top1_acc: f32,
    /// Fraction of probe tools whose own section landed in the top 3.
    top3_acc: f32,
}

fn evaluate(
    h: &Harness,
    resolvers: &[(&'static str, crate::resolver::HarnessResolver)],
    formula: ScoreFormula,
    weights: &DepthWeights,
) -> Eval {
    let mut ratios = Vec::new();
    let mut top1_hits = 0usize;
    let mut top3_hits = 0usize;

    for (probe_tool, resolver) in resolvers {
        // Score every tool section for this probe.
        let mut scored: Vec<(&str, f32)> = TOOLS
            .iter()
            .map(|&t| {
                (
                    t,
                    resolver.section_score(h.tool_section_ids[t], formula, weights),
                )
            })
            .collect();

        let intra = scored
            .iter()
            .find(|(t, _)| t == probe_tool)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        let inter: Vec<f32> = scored
            .iter()
            .filter(|(t, _)| t != probe_tool)
            .map(|(_, s)| *s)
            .collect();
        let inter_mean = inter.iter().sum::<f32>() / inter.len() as f32;
        ratios.push(ratio(intra, inter_mean));

        // Rank accuracy: sort descending, find the probe tool's rank.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let rank = scored
            .iter()
            .position(|(t, _)| t == probe_tool)
            .unwrap_or(usize::MAX);
        if rank == 0 {
            top1_hits += 1;
        }
        if rank < 3 {
            top3_hits += 1;
        }
    }

    let n = resolvers.len() as f32;
    Eval {
        min_ratio: ratios.iter().cloned().fold(f32::INFINITY, f32::min),
        mean_ratio: ratios.iter().sum::<f32>() / ratios.len() as f32,
        top1_acc: top1_hits as f32 / n,
        top3_acc: top3_hits as f32 / n,
    }
}

#[test]
fn prefill_content_type_calibration_sweep() {
    let Some((manifest, pf)): Option<(Manifest, ProvenanceFile)> = try_load_prefill_fixtures()
    else {
        println!(
            "\n[prefill_formula_comparison] prefill_signatures.prov not present \
             (or no prefill offsets in MANIFEST.json) — skipping.\n\
             Generate it with: cargo run -p candle-conversation --release \
             --features cuda,hub --example gen_real_provenance_data -- --force\n"
        );
        return;
    };

    let h = Harness::build();
    let resolvers = h.scan_all_pos1(&pf, &manifest);

    // Fine-grained depth-weight grid — same resolution as the decode sweep.
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

    // Open-minded formula set: run-rewarding Span at several alphas alongside
    // flat-counting formulas that suit scattered prefill signal.
    let formulas: &[(&str, ScoreFormula)] = &[
        ("max", ScoreFormula::Max),
        ("mean", ScoreFormula::Mean),
        ("sum", ScoreFormula::Sum),
        ("count", ScoreFormula::Count),
        ("top_k_mean_8", ScoreFormula::TopKMean { k: 8 }),
        ("span_a1.0", ScoreFormula::Span { alpha: 1.0 }),
        ("span_a1.5", ScoreFormula::Span { alpha: 1.5 }),
        ("span_a2.0", ScoreFormula::Span { alpha: 2.0 }),
    ];

    println!("\n════════════════════════════════════════════════════════════════════════════");
    println!("  PREFILL-phase calibration sweep — corpus: 8 tools × positive scenarios");
    println!("  Primary metric: top-1 rank accuracy (did the correct tool score highest)");
    println!(
        "  Probe scenarios: {} (prefill Q vectors only)",
        resolvers.len()
    );
    println!("════════════════════════════════════════════════════════════════════════════");

    // (formula_name, formula, best Eval, best weights)
    let mut best_per_formula: Vec<(&str, ScoreFormula, Eval, (f32, f32, f32))> = Vec::new();

    for (fname, formula) in formulas {
        // Sweep the grid; keep all rows for the top-5 table.
        let mut rows: Vec<(Eval, (f32, f32, f32))> = grid
            .iter()
            .map(|&(s, e, p)| {
                let w = DepthWeights {
                    syntactic: s,
                    semantic: e,
                    pragmatic: p,
                };
                (evaluate(&h, &resolvers, *formula, &w), (s, e, p))
            })
            .collect();

        // Rank by top-1 accuracy, then top-3, then min_ratio — production
        // cares about getting the right tool ranked, not raw ratio.
        rows.sort_by(|a, b| {
            b.0.top1_acc
                .partial_cmp(&a.0.top1_acc)
                .unwrap()
                .then(b.0.top3_acc.partial_cmp(&a.0.top3_acc).unwrap())
                .then(b.0.min_ratio.partial_cmp(&a.0.min_ratio).unwrap())
        });

        println!("\n── [{fname}] top-5 weight configs ───────────────────────────────────────");
        println!(
            "{:<6} {:<6} {:<6} {:>8} {:>8} {:>10} {:>11}",
            "syn", "sem", "prag", "top1", "top3", "min_ratio", "mean_ratio"
        );
        for (ev, (s, e, p)) in rows.iter().take(5) {
            println!(
                "{:<6.2} {:<6.2} {:<6.2} {:>7.0}% {:>7.0}% {:>10.3} {:>11.3}",
                s,
                e,
                p,
                ev.top1_acc * 100.0,
                ev.top3_acc * 100.0,
                ev.min_ratio,
                ev.mean_ratio
            );
        }

        let (best_ev, best_w) = rows.into_iter().next().unwrap();
        best_per_formula.push((fname, *formula, best_ev, best_w));
    }

    // ── Cross-formula summary ─────────────────────────────────────────────────
    println!("\n── Cross-formula summary (best config per formula) ──────────────────────────");
    println!(
        "{:<14} {:>8} {:>8} {:>10} {:>11}  weights (syn/sem/prag)",
        "formula", "top1", "top3", "min_ratio", "mean_ratio"
    );
    for (name, _, ev, (s, e, p)) in &best_per_formula {
        println!(
            "{:<14} {:>7.0}% {:>7.0}% {:>10.3} {:>11.3}  {:.2}/{:.2}/{:.2}",
            name,
            ev.top1_acc * 100.0,
            ev.top3_acc * 100.0,
            ev.min_ratio,
            ev.mean_ratio,
            s,
            e,
            p
        );
    }

    // Recommend by top-1 accuracy, then min_ratio.
    let best = best_per_formula.iter().max_by(|a, b| {
        a.2.top1_acc
            .partial_cmp(&b.2.top1_acc)
            .unwrap()
            .then(a.2.min_ratio.partial_cmp(&b.2.min_ratio).unwrap())
    });

    if let Some((name, _, ev, (s, e, p))) = best {
        let total = s + e + p;
        let (yn, ye, yp) = (s / total, e / total, p / total);

        println!("\n── Recommended prefill-round config ─────────────────────────────────────────");
        println!("  Formula: {name}");
        println!("  top-1 accuracy : {:.0}%", ev.top1_acc * 100.0);
        println!("  top-3 accuracy : {:.0}%", ev.top3_acc * 100.0);
        println!(
            "  min_ratio      : {:.3}   mean_ratio: {:.3}",
            ev.min_ratio, ev.mean_ratio
        );
        println!("  depth_weights  : syn={yn:.3} sem={ye:.3} prag={yp:.3}");
        println!();
        println!("  NOTE: this is the *initial guess* config used before the decode");
        println!("  probe refines section selection.  The decode round keeps its own");
        println!("  proven Span(alpha=2.0) formula — see formula_comparison.rs.");
    }
    println!();
}
