//! Formula experiments — graded / recentered / per-probe-token scoring.
//!
//! The production failure is a signal-to-noise problem.  The XOR agreement is
//! already graded (0..=128, random baseline 64), but the shipped formulas
//! either (a) pool every (probe,corpus) pair — so one promiscuous probe token
//! inflates the score — or (b) sum raw agreement, where a 64-per-pair noise
//! pedestal swamps the signal.
//!
//! This test scans the production query (`calculator_pos_prod_1`) with a low
//! hit threshold so the *full graded* hit structure is captured, then computes
//! a battery of candidate formulas directly from the hit log and reports how
//! each ranks `calculator` among the 8 harness tools — for both phases.
//!
//! Candidates share two ideas the shipped formulas miss:
//!   - **recentering**: score `excess = agreement - 64`, not raw agreement.
//!   - **per-probe-token reduction**: collapse to one best value *per probe
//!     token* before aggregating, so probe-token *diversity* drives the score.
//!
//! Run: `cargo test -p candle-conversation --test projection_harness
//!       formula_experiments -- --nocapture`

use std::collections::HashMap;

use crate::corpus::{load_fixtures, try_load_prefill_fixtures, CaseType, Manifest, TOOLS};
use crate::harness::Harness;
use candle_conversation::projection::SectionId;
use candle_conversation::provenance::{BdpScanner, ProvenanceFile, SigEntry, TokenHit};

const PROBE: &str = "calculator_pos_prod_1";
const CORRECT: &str = "calculator";
const BASELINE: f32 = 64.0; // random XOR-popcount agreement

/// Scan `probe_id` against the 8-tool corpus with a low hit threshold so the
/// hit log captures the full graded structure (every pair >= `hit_threshold`).
/// Returns (probe_len, per-section hit log).
fn scan_rich(
    h: &Harness,
    pf: &ProvenanceFile,
    manifest: &Manifest,
    probe_id: &str,
    hit_threshold: u32,
) -> (usize, HashMap<SectionId, Vec<TokenHit>>) {
    let probe = manifest
        .scenarios
        .iter()
        .find(|s| s.id == probe_id)
        .expect("probe scenario not found");
    let (probe_syn, probe_sem, probe_prag) = pf
        .read_entry(SigEntry { byte_offset: probe.byte_offset, token_count: probe.token_count })
        .expect("read probe sigs");

    let corpus: Vec<(SectionId, Vec<SigEntry>)> = TOOLS
        .iter()
        .map(|&tool| {
            let sid = h.tool_section_ids[tool];
            let entries: Vec<SigEntry> = manifest
                .scenarios
                .iter()
                .filter(|s| {
                    s.tool.as_deref() == Some(tool)
                        && s.case_type == CaseType::Positive
                        && s.id != probe_id
                })
                .map(|s| SigEntry { byte_offset: s.byte_offset, token_count: s.token_count })
                .collect();
            (sid, entries)
        })
        .collect();

    let mut scanner = BdpScanner::new()
        .with_hit_threshold(hit_threshold)
        .with_record_hits(true);
    scanner
        .scan_sections(pf, &probe_syn, &probe_sem, &probe_prag, &corpus)
        .expect("scan_sections");

    let log = scanner
        .section_hit_log()
        .iter()
        .map(|(&sid, hits)| (sid, hits.clone()))
        .collect();
    (probe_sem.len(), log)
}

/// Best agreement per probe-token index, at one depth.
fn pertok_best(hits: &[TokenHit], depth: u8) -> HashMap<u16, u32> {
    let mut best: HashMap<u16, u32> = HashMap::new();
    for hh in hits.iter().filter(|x| x.depth == depth) {
        let e = best.entry(hh.probe_tok).or_insert(0);
        if hh.agreement > *e {
            *e = hh.agreement;
        }
    }
    best
}

/// All candidate formula scores for one section at one depth.
struct Candidates {
    raw_max: f32,
    raw_sum: f32,
    raw_mean: f32,
    max_excess: f32,
    pertok_sum_excess: f32,
    pertok_count_88: f32,
    pertok_count_95: f32,
    pertok_count_100: f32,
    pertok_top5_excess: f32,
    graded_span: f32,
    /// Shipped decode formula: Span α=2.0 at hit threshold 90.
    span90: f32,
}

fn candidates(hits: &[TokenHit], depth: u8) -> Candidates {
    let depth_hits: Vec<&TokenHit> = hits.iter().filter(|x| x.depth == depth).collect();
    let raw_max = depth_hits.iter().map(|h| h.agreement).max().unwrap_or(0) as f32;
    let raw_sum: f32 = depth_hits.iter().map(|h| h.agreement as f32).sum();
    let raw_mean = if depth_hits.is_empty() {
        0.0
    } else {
        raw_sum / depth_hits.len() as f32
    };

    let best = pertok_best(hits, depth);

    // Per-probe-token excess values.
    let mut excess: Vec<(u16, f32)> = best
        .iter()
        .map(|(&t, &a)| (t, (a as f32 - BASELINE).max(0.0)))
        .collect();
    let pertok_sum_excess: f32 = excess.iter().map(|(_, e)| *e).sum();
    let pertok_count_88 = best.values().filter(|&&a| a >= 88).count() as f32;
    let pertok_count_95 = best.values().filter(|&&a| a >= 95).count() as f32;
    let pertok_count_100 = best.values().filter(|&&a| a >= 100).count() as f32;

    excess.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let pertok_top5_excess: f32 = excess.iter().take(5).map(|(_, e)| *e).sum();

    // Graded span: consecutive runs of probe positions whose best agreement
    // clears 88; each run scored by (sum of excess in run) * run length —
    // rewards both magnitude and sustained runs.
    let mut graded_span = 0.0f32;
    let mut sorted_idx: Vec<u16> = best
        .iter()
        .filter(|(_, &a)| a >= 88)
        .map(|(&t, _)| t)
        .collect();
    sorted_idx.sort_unstable();
    let mut run_sum = 0.0f32;
    let mut run_len = 0u32;
    let mut prev: i32 = -2;
    for &t in &sorted_idx {
        let e = (best[&t] as f32 - BASELINE).max(0.0);
        if t as i32 == prev + 1 {
            run_sum += e;
            run_len += 1;
        } else {
            graded_span += run_sum * run_len as f32;
            run_sum = e;
            run_len = 1;
        }
        prev = t as i32;
    }
    graded_span += run_sum * run_len as f32;

    // span90: shipped decode formula — Span α=2.0 at hit threshold 90.
    // A probe token is a "hit" iff its best agreement clears 90; score the
    // runs of consecutive hit positions by Σ runlen².
    let mut hit_idx: Vec<u16> = best
        .iter()
        .filter(|(_, &a)| a >= 90)
        .map(|(&t, _)| t)
        .collect();
    hit_idx.sort_unstable();
    let mut span90 = 0.0f32;
    let mut run = 0u32;
    let mut prev: i32 = -2;
    for &t in &hit_idx {
        if t as i32 == prev + 1 {
            run += 1;
        } else {
            span90 += (run as f32).powi(2);
            run = 1;
        }
        prev = t as i32;
    }
    span90 += (run as f32).powi(2);

    Candidates {
        raw_max,
        raw_sum,
        raw_mean,
        max_excess: (raw_max - BASELINE).max(0.0),
        pertok_sum_excess,
        pertok_count_88,
        pertok_count_95,
        pertok_count_100,
        pertok_top5_excess,
        graded_span,
        span90,
    }
}

/// Rank calculator under a candidate-extracting closure.
fn rank<F: Fn(&Candidates) -> f32>(
    h: &Harness,
    log: &HashMap<SectionId, Vec<TokenHit>>,
    depth: u8,
    pick: F,
) -> (usize, f32, &'static str, f32) {
    let mut scored: Vec<(&'static str, f32)> = TOOLS
        .iter()
        .map(|&tool| {
            let sid = h.tool_section_ids[tool];
            let empty = Vec::new();
            let hits = log.get(&sid).unwrap_or(&empty);
            (tool, pick(&candidates(hits, depth)))
        })
        .collect();
    let calc = scored.iter().find(|(t, _)| *t == CORRECT).map(|(_, s)| *s).unwrap_or(0.0);
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let r = scored.iter().position(|(t, _)| *t == CORRECT).unwrap_or(usize::MAX) + 1;
    let (bc, bs) = scored
        .iter()
        .find(|(t, _)| *t != CORRECT)
        .copied()
        .unwrap_or(("?", 0.0));
    (r, calc, bc, bs)
}

fn run_phase(label: &str, h: &Harness, pf: &ProvenanceFile, manifest: &Manifest) {
    if !manifest.scenarios.iter().any(|s| s.id == PROBE) {
        println!("  [{label}] probe {PROBE} not in manifest — skipping");
        return;
    }
    // Low threshold (70) so the hit log keeps the graded structure that a
    // threshold-90 scan would discard — especially for prefill.
    let (probe_len, log) = scan_rich(h, pf, manifest, PROBE, 70);
    println!("\n══ [{label}] probe_len={probe_len}  (hit log captured at threshold 70) ══");

    let cands: &[(&str, fn(&Candidates) -> f32)] = &[
        ("raw_max",            |c| c.raw_max),
        ("raw_sum",            |c| c.raw_sum),
        ("raw_mean",           |c| c.raw_mean),
        ("max_excess",         |c| c.max_excess),
        ("pertok_sum_excess",  |c| c.pertok_sum_excess),
        ("pertok_count_88",    |c| c.pertok_count_88),
        ("pertok_count_95",    |c| c.pertok_count_95),
        ("pertok_count_100",   |c| c.pertok_count_100),
        ("pertok_top5_excess", |c| c.pertok_top5_excess),
        ("graded_span",        |c| c.graded_span),
    ];

    for depth in [1u8, 2u8] {
        let dname = if depth == 1 { "semantic" } else { "pragmatic" };
        println!("  depth = {dname}");
        println!("    {:<20} {:>5} {:>12} {:>14} {:>12} {:>10}",
            "candidate", "rank", "calc", "2nd tool", "2nd score", "margin");
        for (name, f) in cands {
            let (r, calc, bc, bs) = rank(h, &log, depth, f);
            let flag = if r == 1 { "  WIN" } else { "" };
            println!("    {:<20} {:>5} {:>12.2} {:>14} {:>12.2} {:>10.2}{}",
                name, r, calc, bc, bs, calc - bs, flag);
        }
    }
}

/// Full-corpus sweep: rank every positive probe under each formula and
/// aggregate top-1 / top-3 accuracy + intra/inter ratio.
fn corpus_sweep(label: &str, h: &Harness, pf: &ProvenanceFile, manifest: &Manifest) {
    let probes: Vec<&crate::corpus::Scenario> = manifest
        .scenarios
        .iter()
        .filter(|s| s.tool.is_some() && s.case_type == CaseType::Positive)
        .collect();
    if probes.is_empty() {
        println!("  [{label}] no positive probes — skipping");
        return;
    }

    let formulas: &[(&str, fn(&Candidates) -> f32)] = &[
        ("raw_max",           |c| c.raw_max),
        ("span_a2 (shipped)", |c| c.span90),
        ("pertok_sum_excess", |c| c.pertok_sum_excess),
    ];

    println!("\n══ [{label}] full-corpus sweep — {} positive probes ══", probes.len());

    for depth in [1u8, 2u8] {
        let dname = if depth == 1 { "semantic" } else { "pragmatic" };
        // (formula_idx) -> (top1, top3, sum_ratio, min_ratio, n)
        let mut top1 = vec![0usize; formulas.len()];
        let mut top3 = vec![0usize; formulas.len()];
        let mut sum_ratio = vec![0.0f64; formulas.len()];
        let mut min_ratio = vec![f64::INFINITY; formulas.len()];

        for probe in &probes {
            let correct = probe.tool.as_deref().unwrap();
            let (_plen, log) = scan_rich(h, pf, manifest, &probe.id, 64);

            for (fi, (_, pick)) in formulas.iter().enumerate() {
                let mut scored: Vec<(&str, f32)> = TOOLS
                    .iter()
                    .map(|&tool| {
                        let empty = Vec::new();
                        let hits = log.get(&h.tool_section_ids[tool]).unwrap_or(&empty);
                        (tool, pick(&candidates(hits, depth)))
                    })
                    .collect();
                let intra = scored.iter().find(|(t, _)| *t == correct).map(|(_, s)| *s).unwrap_or(0.0);
                let inter: f64 = scored.iter().filter(|(t, _)| *t != correct)
                    .map(|(_, s)| *s as f64).sum::<f64>() / (TOOLS.len() - 1) as f64;
                let ratio = if inter > 0.0 { intra as f64 / inter } else if intra > 0.0 { f64::INFINITY } else { 1.0 };
                sum_ratio[fi] += if ratio.is_finite() { ratio } else { 0.0 };
                if ratio < min_ratio[fi] { min_ratio[fi] = ratio; }

                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let rank = scored.iter().position(|(t, _)| *t == correct).unwrap_or(99) + 1;
                if rank == 1 { top1[fi] += 1; }
                if rank <= 3 { top3[fi] += 1; }
            }
        }

        let n = probes.len() as f64;
        println!("  depth = {dname}");
        println!("    {:<22} {:>8} {:>8} {:>12} {:>11}",
            "formula", "top-1", "top-3", "mean_ratio", "min_ratio");
        for (fi, (fname, _)) in formulas.iter().enumerate() {
            println!("    {:<22} {:>7.0}% {:>7.0}% {:>12.3} {:>11.3}",
                fname,
                top1[fi] as f64 / n * 100.0,
                top3[fi] as f64 / n * 100.0,
                sum_ratio[fi] / n,
                min_ratio[fi]);
        }
    }
}

#[test]
fn formula_corpus_sweep() {
    let h = Harness::build();
    println!("\n════════════════════════════════════════════════════════════════════");
    println!("  Full-corpus formula sweep — pertok_sum_excess vs shipped Span α=2.0");
    println!("════════════════════════════════════════════════════════════════════");

    let (dm, dpf) = load_fixtures();
    corpus_sweep("DECODE", &h, &dpf, &dm);

    match try_load_prefill_fixtures() {
        Some((pm, ppf)) => corpus_sweep("PREFILL", &h, &ppf, &pm),
        None => println!("\n  [PREFILL] prefill_signatures.prov not present — skipping"),
    }
    println!();
}

#[test]
fn formula_experiments() {
    let h = Harness::build();
    println!("\n════════════════════════════════════════════════════════════════════");
    println!("  Formula experiments — probe = {PROBE}");
    println!("  candidates: recentered (excess over 64) + per-probe-token reduction");
    println!("════════════════════════════════════════════════════════════════════");

    let (dm, dpf) = load_fixtures();
    run_phase("DECODE", &h, &dpf, &dm);

    match try_load_prefill_fixtures() {
        Some((pm, ppf)) => run_phase("PREFILL", &h, &ppf, &pm),
        None => println!("\n  [PREFILL] prefill_signatures.prov not present — skipping"),
    }
    println!();
}
