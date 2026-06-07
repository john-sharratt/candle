//! Diagnostic for the production tool-selection failure.
//!
//! The zend run on "use a tool, determine 123891283 + 123124" ranked
//! `calculator` ~5th of 93 tools — beaten by noise peaks.  That query was
//! added to the calibration set as `calculator_pos_prod_1`; this test probes
//! with it against the 8-tool harness corpus and sweeps every formula to
//! isolate two questions:
//!
//!   1. Is the digit-heavy production query intrinsically hard — does it
//!      rank `calculator` worse than a clean calibration query even on the
//!      8-tool corpus?
//!   2. Which formula / phase gives `calculator` the cleanest win + margin?
//!
//! It also breaks down the raw hit log: which probe tokens are *promiscuous*
//! (hit many tools — noise) vs *discriminative* (hit only calculator).
//!
//! Harness limit: the corpus is 8 tools of prefill/decode-Q of user prompts,
//! not the 93-tool description corpus zend uses.  This isolates formula and
//! query effects; it cannot reproduce the full 93-tool noise floor.
//!
//! Run: `cargo test -p candle-conversation --test projection_harness
//!       production_query_diagnostic -- --nocapture`

use std::collections::{HashMap, HashSet};

use crate::corpus::{load_fixtures, try_load_prefill_fixtures, Manifest, TOOLS};
use crate::harness::Harness;
use candle_conversation::projection::{ContentResolver, DepthWeights, ScoreFormula};
use candle_conversation::provenance::ProvenanceFile;

const PROBE: &str = "calculator_pos_prod_1";
const CORRECT: &str = "calculator";

/// Rank `calculator` among the 8 tools under `formula` + `weights`.
/// Returns (rank 1-based, calc_score, best_competitor_name, best_competitor_score).
fn rank_calculator(
    h: &Harness,
    resolver: &crate::resolver::HarnessResolver,
    formula: ScoreFormula,
    weights: &DepthWeights,
) -> (usize, f32, &'static str, f32) {
    let mut scored: Vec<(&'static str, f32)> = TOOLS
        .iter()
        .map(|&t| (t, resolver.section_score(h.tool_section_ids[t], formula, weights)))
        .collect();
    let calc = scored.iter().find(|(t, _)| *t == CORRECT).map(|(_, s)| *s).unwrap_or(0.0);
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let rank = scored.iter().position(|(t, _)| *t == CORRECT).unwrap_or(usize::MAX) + 1;
    let (bc, bs) = scored
        .iter()
        .find(|(t, _)| *t != CORRECT)
        .copied()
        .unwrap_or(("?", 0.0));
    (rank, calc, bc, bs)
}

fn sweep_phase(
    label: &str,
    h: &Harness,
    pf: &ProvenanceFile,
    manifest: &Manifest,
) {
    if !manifest.scenarios.iter().any(|s| s.id == PROBE) {
        println!("  [{label}] probe scenario {PROBE} not in manifest — skipping");
        return;
    }
    let (resolver, hit_log) = h.scan_with_hits(pf, manifest, PROBE);

    let weights: &[(&str, DepthWeights)] = &[
        ("semantic", DepthWeights { syntactic: 0.0, semantic: 1.0, pragmatic: 0.0 }),
        ("pragmatic", DepthWeights { syntactic: 0.0, semantic: 0.0, pragmatic: 1.0 }),
        ("equal", DepthWeights { syntactic: 1.0, semantic: 1.0, pragmatic: 1.0 }),
    ];
    let formulas: &[(&str, ScoreFormula)] = &[
        ("max", ScoreFormula::Max),
        ("mean", ScoreFormula::Mean),
        ("sum", ScoreFormula::Sum),
        ("count", ScoreFormula::Count),
        ("top_k_mean_3", ScoreFormula::TopKMean { k: 3 }),
        ("top_k_mean_8", ScoreFormula::TopKMean { k: 8 }),
        ("span_a1.0", ScoreFormula::Span { alpha: 1.0 }),
        ("span_a2.0", ScoreFormula::Span { alpha: 2.0 }),
    ];

    println!("\n══ [{label}] formula sweep — calculator rank among 8 tools ══");
    for (wname, w) in weights {
        println!("  depth weights = {wname}");
        println!("    {:<14} {:>6} {:>10} {:>10}  {:<14} {:>10}  {:>8}",
            "formula", "rank", "calc", "2nd", "(2nd tool)", "score", "margin");
        for (fname, formula) in formulas {
            let (rank, calc, bc, bs) = rank_calculator(h, &resolver, *formula, w);
            let margin = calc - bs;
            let flag = if rank == 1 { " <-- win" } else { "" };
            println!("    {:<14} {:>6} {:>10.3} {:>10} {:<14} {:>10.3}  {:>8.3}{}",
                fname, rank, calc, "", format!("({bc})"), bs, margin, flag);
        }
    }

    // ── Hit-log promiscuity (semantic depth only) ──────────────────────────
    // For each probe token, how many of the 8 tool sections it hit.  A token
    // that hits many tools is noise cross-talk; one that hits few is signal.
    let mut tok_tools: HashMap<u16, HashSet<&str>> = HashMap::new();
    let mut tok_best: HashMap<u16, (u32, &str)> = HashMap::new(); // probe_tok -> (best agreement, tool)
    for &tool in TOOLS {
        let sid = h.tool_section_ids[tool];
        if let Some(hits) = hit_log.get(&sid) {
            for hh in hits.iter().filter(|x| x.depth == 1) {
                tok_tools.entry(hh.probe_tok).or_default().insert(tool);
                let e = tok_best.entry(hh.probe_tok).or_insert((0, tool));
                if hh.agreement > e.0 {
                    *e = (hh.agreement, tool);
                }
            }
        }
    }
    let mut spread = [0usize; 9]; // spread[n] = #probe tokens hitting exactly n tools
    for set in tok_tools.values() {
        spread[set.len().min(8)] += 1;
    }
    println!("\n  ── hit-log promiscuity (semantic depth) ──");
    println!("    probe tokens hitting N tools:");
    for (n, &count) in spread.iter().enumerate().take(9).skip(1) {
        if count > 0 {
            println!("      {n} tool(s): {count} probe token(s)");
        }
    }
    let discriminative: Vec<u16> = tok_tools
        .iter()
        .filter(|(_, set)| set.len() == 1 && set.contains(CORRECT))
        .map(|(t, _)| *t)
        .collect();
    println!("    probe tokens hitting ONLY calculator: {} {:?}",
        discriminative.len(), {
            let mut d = discriminative.clone();
            d.sort_unstable();
            d
        });
    // Best agreement among calculator-exclusive tokens vs promiscuous tokens.
    let excl_best: Vec<u32> = discriminative.iter()
        .filter_map(|t| tok_best.get(t).map(|(a, _)| *a)).collect();
    if !excl_best.is_empty() {
        let mx = excl_best.iter().copied().max().unwrap();
        let mn = excl_best.iter().copied().min().unwrap();
        println!("    calculator-exclusive token agreement range: {mn}..{mx}");
    }
}

#[test]
fn production_query_diagnostic() {
    let h = Harness::build();

    println!("\n════════════════════════════════════════════════════════════════════");
    println!("  Production-query SNR diagnostic");
    println!("  probe = {PROBE}  (\"use a tool, determine 123891283 + 123124\")");
    println!("  corpus = 8 harness tools (user-prompt Q vectors)");
    println!("════════════════════════════════════════════════════════════════════");

    // Decode phase.
    let (decode_manifest, decode_pf) = load_fixtures();
    sweep_phase("DECODE", &h, &decode_pf, &decode_manifest);

    // Prefill phase.
    match try_load_prefill_fixtures() {
        Some((pre_manifest, pre_pf)) => sweep_phase("PREFILL", &h, &pre_pf, &pre_manifest),
        None => println!("\n  [PREFILL] prefill_signatures.prov not present — skipping"),
    }
    println!();
}
