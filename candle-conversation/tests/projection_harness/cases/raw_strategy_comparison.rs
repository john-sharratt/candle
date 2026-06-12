//! Comprehensive raw-strategy discrimination analysis.
//!
//! Runs a single pass over all probe scenarios and reports every relevant
//! dimension of signal quality in one `--nocapture` output:
//!
//!   § 1  All strategy × (layer, head) — min/mean/max intra/inter ratio
//!   § 2  Window-mean probe sweep — windows 1/2/4 at sampled (layer, head)
//!   § 3  Span scoring — Count vs Span(α=1.5) vs Span(α=2.0) for top strategies
//!   § 4  Per-tool breakdown — best strategy from § 1
//!   § 5  Sensitivity tables — QK and FloatSimHash at all (layer × head)
//!   § 6  BandMean head sensitivity
//!   § 7  Span per-tool breakdown — best strategy from § 3
//!   § 8  Dual-layer combinations: l0 × l4 — Options A / B / C
//!
//! Run:
//!   cargo test -p candle-conversation --test projection_harness \
//!     -- raw_signature_strategy_comparison --nocapture --release

use std::collections::{HashMap, HashSet};
use std::io::Write as _;

use candle_conversation::projection::{ContentResolver, DepthWeights, ScoreFormula, SectionId};
use candle_conversation::provenance::TokenHit;

use crate::corpus::{try_load_raw_fixtures as load_raw_fixtures_opt, CaseType, TOOLS};
use crate::harness::{span_score_mean, Harness, RawCorpusCache, SignatureStrategy};

/// `println!` + immediate stdout flush so output appears progressively when
/// piped through `tee` (which would otherwise block-buffer the writes).
macro_rules! fprintln {
    ($($arg:tt)*) => {{
        println!($($arg)*);
        let _ = std::io::stdout().flush();
    }};
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn ratio(intra: f32, inter_mean: f32) -> f32 {
    if inter_mean > 0.0 {
        intra / inter_mean
    } else if intra > 0.0 {
        f32::INFINITY // genuine perfect discrimination: target hits, nothing else does
    } else {
        1.0 // both zero: strategy produced no BDP signal at all
    }
}

/// Compute intra/inter discrimination ratio for one (probe_tool, resolver) pair.
fn discrimination(
    h: &Harness,
    resolver: &impl ContentResolver,
    probe_tool: &str,
    formula: ScoreFormula,
    weights: &DepthWeights,
) -> (f32, f32) {
    let intra = resolver.section_score(h.tool_section_ids[probe_tool], formula, weights);
    let inter_mean = TOOLS
        .iter()
        .filter(|&&t| t != probe_tool)
        .map(|&t| resolver.section_score(h.tool_section_ids[t], formula, weights))
        .sum::<f32>()
        / (TOOLS.len() - 1) as f32;
    (intra, inter_mean)
}

/// Sweep all probe_ids with one strategy → (min, mean, max) ratio.
///
/// Precomputes corpus signatures once and reuses them across all probes,
/// avoiding the N-fold redundant raw KVQ reads that the naive approach incurs.
fn sweep_ratios(
    h: &Harness,
    raw_pf: &candle_conversation::provenance::RawProvenanceFile,
    raw_manifest: &crate::corpus::RawManifest,
    probe_ids: &[(&'static str, String)],
    strategy: &SignatureStrategy,
    formula: ScoreFormula,
    weights: &DepthWeights,
) -> (f32, f32, f32) {
    let cache = h.build_raw_corpus_cache(raw_pf, raw_manifest, strategy);
    sweep_ratios_with_cache(
        h,
        raw_pf,
        raw_manifest,
        probe_ids,
        strategy,
        formula,
        weights,
        &cache,
    )
}

/// Inner sweep that reuses a precomputed `RawCorpusCache`.
fn sweep_ratios_with_cache(
    h: &Harness,
    raw_pf: &candle_conversation::provenance::RawProvenanceFile,
    raw_manifest: &crate::corpus::RawManifest,
    probe_ids: &[(&'static str, String)],
    strategy: &SignatureStrategy,
    formula: ScoreFormula,
    weights: &DepthWeights,
    cache: &RawCorpusCache,
) -> (f32, f32, f32) {
    let ratios: Vec<f32> = probe_ids
        .iter()
        .map(|(tool, id)| {
            let resolver = h.scan_raw_cached(raw_pf, raw_manifest, id, strategy, cache);
            let (intra, inter) = discrimination(h, &resolver, tool, formula, weights);
            ratio(intra, inter)
        })
        .collect();
    let min_r = ratios.iter().cloned().fold(f32::INFINITY, f32::min);
    let mean_r = ratios.iter().sum::<f32>() / ratios.len() as f32;
    let max_r = ratios.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (min_r, mean_r, max_r)
}

/// Span-score discrimination ratio: mean-of-depths span score, equal weights.
fn span_discrimination(
    h: &Harness,
    hit_log: &HashMap<SectionId, Vec<TokenHit>>,
    probe_tool: &str,
    alpha: f32,
) -> (f32, f32) {
    let intra = hit_log
        .get(&h.tool_section_ids[probe_tool])
        .map(|hits| span_score_mean(hits, alpha))
        .unwrap_or(0.0);
    let inter_mean = TOOLS
        .iter()
        .filter(|&&t| t != probe_tool)
        .map(|&t| {
            hit_log
                .get(&h.tool_section_ids[t])
                .map(|hits| span_score_mean(hits, alpha))
                .unwrap_or(0.0)
        })
        .sum::<f32>()
        / (TOOLS.len() - 1) as f32;
    (intra, inter_mean)
}

// ── Dual-layer helpers ────────────────────────────────────────────────────────

/// Span score over an explicit set of probe token indices (for gated variants).
fn span_score_from_toks(mut toks: Vec<u16>, alpha: f32) -> f32 {
    toks.sort_unstable();
    toks.dedup();
    let mut score = 0.0f32;
    let mut run_len = 0usize;
    let mut prev = u16::MAX;
    for &tok in &toks {
        if prev == u16::MAX || tok != prev + 1 {
            score += (run_len as f32).powf(alpha);
            run_len = 1;
        } else {
            run_len += 1;
        }
        prev = tok;
    }
    score += (run_len as f32).powf(alpha);
    score
}

/// Option B — additive span+count with per-probe normalisation.
///
/// For each probe:
///   1. Compute span(α) scores per tool using `strat_span` hit log.
///   2. Compute count scores per tool using `strat_count` resolver.
///   3. Normalise each component by its mean across all tools for that probe
///      so both contribute equally regardless of absolute scale difference.
///   4. combined(tool) = norm_span(tool) + norm_count(tool)
///   5. ratio = combined(intra) / mean(combined(inter))
fn sweep_ratios_additive_b(
    h: &Harness,
    raw_pf: &candle_conversation::provenance::RawProvenanceFile,
    raw_manifest: &crate::corpus::RawManifest,
    probe_ids: &[(&'static str, String)],
    strat_span: &SignatureStrategy,
    strat_count: &SignatureStrategy,
    alpha: f32,
) -> (f32, f32, f32) {
    let eq = DepthWeights {
        syntactic: 1.0,
        semantic: 1.0,
        pragmatic: 1.0,
    };
    let count_formula = ScoreFormula::Count;

    let cache_span = h.build_raw_corpus_cache(raw_pf, raw_manifest, strat_span);
    let cache_count = h.build_raw_corpus_cache(raw_pf, raw_manifest, strat_count);

    let ratios: Vec<f32> = probe_ids
        .iter()
        .map(|(probe_tool, probe_id)| {
            let (_, hit_log) = h.scan_raw_with_hits_cached(
                raw_pf,
                raw_manifest,
                probe_id,
                strat_span,
                &cache_span,
            );
            let resolver =
                h.scan_raw_cached(raw_pf, raw_manifest, probe_id, strat_count, &cache_count);

            let span_scores: Vec<f32> = TOOLS
                .iter()
                .map(|&t| {
                    hit_log
                        .get(&h.tool_section_ids[t])
                        .map(|hits| span_score_mean(hits, alpha))
                        .unwrap_or(0.0)
                })
                .collect();
            let count_scores: Vec<f32> = TOOLS
                .iter()
                .map(|&t| resolver.section_score(h.tool_section_ids[t], count_formula, &eq))
                .collect();

            let span_mean = span_scores.iter().sum::<f32>() / TOOLS.len() as f32;
            let count_mean = count_scores.iter().sum::<f32>() / TOOLS.len() as f32;

            let combined: Vec<f32> = span_scores
                .iter()
                .zip(count_scores.iter())
                .map(|(&s, &c)| {
                    let ns = if span_mean > 0.0 { s / span_mean } else { 1.0 };
                    let nc = if count_mean > 0.0 {
                        c / count_mean
                    } else {
                        1.0
                    };
                    ns + nc
                })
                .collect();

            let intra_idx = TOOLS.iter().position(|&t| t == *probe_tool).unwrap();
            let intra = combined[intra_idx];
            let inter_mean = combined
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != intra_idx)
                .map(|(_, &v)| v)
                .sum::<f32>()
                / (TOOLS.len() - 1) as f32;
            ratio(intra, inter_mean)
        })
        .collect();

    let min_r = ratios.iter().cloned().fold(f32::INFINITY, f32::min);
    let mean_r = ratios.iter().sum::<f32>() / ratios.len() as f32;
    let max_r = ratios.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (min_r, mean_r, max_r)
}

/// Mean span score for one section from two hit logs, gated to only probe tokens
/// that appear in BOTH `hit_log_gate` AND `hit_log_span` (intersection).
fn gated_span_score_section(
    hit_log_span: &HashMap<SectionId, Vec<TokenHit>>,
    hit_log_gate: &HashMap<SectionId, Vec<TokenHit>>,
    section: SectionId,
    alpha: f32,
) -> f32 {
    let mut total = 0.0f32;
    for depth in 0u8..3 {
        let span_toks: HashSet<u16> = hit_log_span
            .get(&section)
            .map(|hits| {
                hits.iter()
                    .filter(|h| h.depth == depth)
                    .map(|h| h.probe_tok)
                    .collect()
            })
            .unwrap_or_default();
        let gate_toks: HashSet<u16> = hit_log_gate
            .get(&section)
            .map(|hits| {
                hits.iter()
                    .filter(|h| h.depth == depth)
                    .map(|h| h.probe_tok)
                    .collect()
            })
            .unwrap_or_default();
        let gated: Vec<u16> = span_toks.intersection(&gate_toks).cloned().collect();
        total += span_score_from_toks(gated, alpha);
    }
    total / 3.0
}

/// Option C — gated span: a probe token only contributes to the span score for a
/// section if BOTH `strat_span` (l0) AND `strat_gate` (l4) produced a hit for it.
/// This reduces spurious span starts while retaining genuine sustained-attention runs.
fn sweep_ratios_gated_c(
    h: &Harness,
    raw_pf: &candle_conversation::provenance::RawProvenanceFile,
    raw_manifest: &crate::corpus::RawManifest,
    probe_ids: &[(&'static str, String)],
    strat_span: &SignatureStrategy,
    strat_gate: &SignatureStrategy,
    alpha: f32,
) -> (f32, f32, f32) {
    let cache_span = h.build_raw_corpus_cache(raw_pf, raw_manifest, strat_span);
    let cache_gate = h.build_raw_corpus_cache(raw_pf, raw_manifest, strat_gate);

    let ratios: Vec<f32> = probe_ids
        .iter()
        .map(|(probe_tool, probe_id)| {
            let (_, hit_log_span) = h.scan_raw_with_hits_cached(
                raw_pf,
                raw_manifest,
                probe_id,
                strat_span,
                &cache_span,
            );
            let (_, hit_log_gate) = h.scan_raw_with_hits_cached(
                raw_pf,
                raw_manifest,
                probe_id,
                strat_gate,
                &cache_gate,
            );

            let intra_sid = h.tool_section_ids[*probe_tool];
            let intra = gated_span_score_section(&hit_log_span, &hit_log_gate, intra_sid, alpha);
            let inter_mean = TOOLS
                .iter()
                .filter(|&&t| t != *probe_tool)
                .map(|&t| {
                    gated_span_score_section(
                        &hit_log_span,
                        &hit_log_gate,
                        h.tool_section_ids[t],
                        alpha,
                    )
                })
                .sum::<f32>()
                / (TOOLS.len() - 1) as f32;
            ratio(intra, inter_mean)
        })
        .collect();

    let min_r = ratios.iter().cloned().fold(f32::INFINITY, f32::min);
    let mean_r = ratios.iter().sum::<f32>() / ratios.len() as f32;
    let max_r = ratios.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (min_r, mean_r, max_r)
}

// ── Test ──────────────────────────────────────────────────────────────────────

#[test]
fn raw_signature_strategy_comparison() {
    let Some((raw_manifest, raw_pf)) = load_raw_fixtures_opt() else {
        eprintln!("SKIP: raw_kvq.prov not present — run gen_real_provenance_data without --skip-raw to generate it");
        return;
    };
    let h = Harness::build();

    let n_layers = raw_manifest.n_layers_per_band as usize;
    let n_heads = raw_manifest.n_kv_heads as usize;
    let eq = DepthWeights {
        syntactic: 1.0,
        semantic: 1.0,
        pragmatic: 1.0,
    };
    let count = ScoreFormula::Count;

    // Sample layers and heads; keep it dense enough to see trends.
    let test_layers: Vec<usize> = match n_layers {
        0 => vec![],
        1 => vec![0],
        2 => vec![0, 1],
        _ => (0..n_layers)
            .step_by((n_layers - 1).max(1))
            .chain(std::iter::once(n_layers - 1))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .chain(std::iter::once(n_layers / 2))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect(),
    };
    let test_heads: Vec<usize> = (0..n_heads).collect(); // all heads (only 4 for Qwen3-30B-A3B)

    // All positive probe scenarios (6 per tool = 48 total).
    let probe_ids: Vec<(&'static str, String)> = TOOLS
        .iter()
        .flat_map(|&tool| {
            raw_manifest
                .scenarios
                .iter()
                .filter(|s| s.tool.as_deref() == Some(tool) && s.case_type == CaseType::Positive)
                .map(|s| (tool, s.id.clone()))
                .collect::<Vec<_>>()
        })
        .collect();

    if probe_ids.is_empty() {
        fprintln!("No pos_1 probes in raw manifest — run gen_real_provenance_data first.");
        return;
    }

    // ── § 1: Full strategy sweep ──────────────────────────────────────────────
    let mut all_strategies: Vec<SignatureStrategy> = Vec::new();
    for &l in &test_layers {
        for &head in &test_heads {
            all_strategies.push(SignatureStrategy::QQ { layer: l, head });
            all_strategies.push(SignatureStrategy::QK { layer: l, head });
            all_strategies.push(SignatureStrategy::KK { layer: l, head });
            all_strategies.push(SignatureStrategy::FloatSimHash { layer: l, head });
        }
        all_strategies.push(SignatureStrategy::MultiHeadXorQQ { layer: l });
        all_strategies.push(SignatureStrategy::MultiHeadXorQK { layer: l });
        all_strategies.push(SignatureStrategy::MultiHeadMeanQQ { layer: l });
        all_strategies.push(SignatureStrategy::MultiHeadMeanQK { layer: l });
    }
    for &head in &test_heads {
        all_strategies.push(SignatureStrategy::BandMeanQQ { head });
        all_strategies.push(SignatureStrategy::BandMeanQK { head });
    }

    fprintln!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    fprintln!("║  § 1  Strategy sweep (Count formula, equal depth weights)               ║");
    fprintln!("╚══════════════════════════════════════════════════════════════════════════╝");
    fprintln!(
        "{:<28} {:>10} {:>12} {:>9}",
        "strategy",
        "min_ratio",
        "mean_ratio",
        "max_ratio"
    );
    fprintln!("{}", "─".repeat(62));

    let mut scored: Vec<(String, f32, f32, f32)> = Vec::new();
    for strat in &all_strategies {
        let (min_r, mean_r, max_r) =
            sweep_ratios(&h, &raw_pf, &raw_manifest, &probe_ids, strat, count, &eq);
        fprintln!(
            "{:<28} {:>10.4} {:>12.4} {:>9.4}",
            strat.name(),
            min_r,
            mean_r,
            max_r
        );
        scored.push((strat.name(), min_r, mean_r, max_r));
    }

    // Sort by (min_ratio DESC, mean_ratio DESC): reliability first, then average signal.
    // min_ratio > 1.0 means the strategy discriminates correctly on every probe.
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap()
            .then(b.2.partial_cmp(&a.2).unwrap())
    });
    let top5: Vec<String> = scored.iter().take(5).map(|(n, ..)| n.clone()).collect();
    fprintln!("\nTop-5 by (min_ratio, mean_ratio): {}", top5.join("  |  "));

    // ── § 2: Window-mean sweep ────────────────────────────────────────────────
    fprintln!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    fprintln!("║  § 2  WindowMeanQ sweep (windows 1/2/4, all heads, sampled layers)      ║");
    fprintln!("╚══════════════════════════════════════════════════════════════════════════╝");
    fprintln!(
        "{:<32} {:>10} {:>12} {:>9}",
        "strategy",
        "min_ratio",
        "mean_ratio",
        "max_ratio"
    );
    fprintln!("{}", "─".repeat(66));

    let win_layers: Vec<usize> = if test_layers.is_empty() {
        vec![]
    } else {
        vec![test_layers[test_layers.len() / 2]]
    }; // mid band layer
    for &l in &win_layers {
        for &head in &test_heads {
            for &w in &[1usize, 2, 4] {
                let strat = SignatureStrategy::WindowMeanQ {
                    window: w,
                    layer: l,
                    head,
                };
                let (min_r, mean_r, max_r) =
                    sweep_ratios(&h, &raw_pf, &raw_manifest, &probe_ids, &strat, count, &eq);
                fprintln!(
                    "{:<32} {:>10.4} {:>12.4} {:>9.4}",
                    strat.name(),
                    min_r,
                    mean_r,
                    max_r
                );
                scored.push((strat.name(), min_r, mean_r, max_r));
            }
        }
    }
    // Re-rank top-5 including window variants (same reliability-first sort).
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap()
            .then(b.2.partial_cmp(&a.2).unwrap())
    });
    let top5_all: Vec<String> = scored.iter().take(5).map(|(n, ..)| n.clone()).collect();
    fprintln!(
        "\nTop-5 (including window variants): {}",
        top5_all.join("  |  ")
    );

    // ── § 3: Span scoring comparison ──────────────────────────────────────────
    // Re-run the top-5 strategies (from § 1+2) with record_hits=true and
    // compare Count vs Span(α=1.5) vs Span(α=2.0) discrimination.
    let top_strats_for_span: Vec<&SignatureStrategy> = all_strategies
        .iter()
        .filter(|s| top5_all.contains(&s.name()))
        .take(5)
        .collect();

    fprintln!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    fprintln!("║  § 3  Span scoring — Count vs Span(α=1.5) vs Span(α=2.0)               ║");
    fprintln!("╚══════════════════════════════════════════════════════════════════════════╝");
    fprintln!(
        "{:<28} {:>10} {:>12} {:>12}",
        "strategy",
        "count_mean",
        "span1.5_mean",
        "span2.0_mean"
    );
    fprintln!("{}", "─".repeat(65));

    let mut best_span_strategy_name = String::new();
    let mut best_span_mean = f32::NEG_INFINITY;

    for strat in &top_strats_for_span {
        let span_cache = h.build_raw_corpus_cache(&raw_pf, &raw_manifest, strat);
        let mut count_ratios = Vec::new();
        let mut span15_ratios = Vec::new();
        let mut span20_ratios = Vec::new();

        for (probe_tool, probe_id) in &probe_ids {
            let (resolver, hit_log) =
                h.scan_raw_with_hits_cached(&raw_pf, &raw_manifest, probe_id, strat, &span_cache);

            let (intra_c, inter_c) = discrimination(&h, &resolver, probe_tool, count, &eq);
            count_ratios.push(ratio(intra_c, inter_c));

            let (intra_s15, inter_s15) = span_discrimination(&h, &hit_log, probe_tool, 1.5);
            span15_ratios.push(ratio(intra_s15, inter_s15));

            let (intra_s20, inter_s20) = span_discrimination(&h, &hit_log, probe_tool, 2.0);
            span20_ratios.push(ratio(intra_s20, inter_s20));
        }

        let mean_count = count_ratios.iter().sum::<f32>() / count_ratios.len() as f32;
        let mean_s15 = span15_ratios.iter().sum::<f32>() / span15_ratios.len() as f32;
        let mean_s20 = span20_ratios.iter().sum::<f32>() / span20_ratios.len() as f32;

        fprintln!(
            "{:<28} {:>10.4} {:>12.4} {:>12.4}",
            strat.name(),
            mean_count,
            mean_s15,
            mean_s20
        );

        let best_span = mean_s15.max(mean_s20);
        if best_span > best_span_mean {
            best_span_mean = best_span;
            best_span_strategy_name = strat.name();
        }
    }

    // ── § 4: Per-tool breakdown for best § 1 strategy ────────────────────────
    let best_s1_name = &scored[0].0;
    let best_s1_strat = all_strategies
        .iter()
        .find(|s| &s.name() == best_s1_name)
        .unwrap();

    fprintln!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    fprintln!(
        "║  § 4  Per-tool breakdown: {:<48}║",
        format!("{best_s1_name} (best § 1)")
    );
    fprintln!("╚══════════════════════════════════════════════════════════════════════════╝");
    fprintln!(
        "{:<14} {:>12} {:>14} {:>9} {:>9}",
        "tool",
        "mean_intra",
        "mean_inter",
        "mean_r",
        "min_r"
    );
    fprintln!("{}", "─".repeat(61));

    let s4_cache = h.build_raw_corpus_cache(&raw_pf, &raw_manifest, best_s1_strat);
    // Aggregate per tool: collect (intra, best_inter) across all positive probes.
    let mut s4_intras: HashMap<&str, Vec<f32>> = TOOLS.iter().map(|&t| (t, vec![])).collect();
    let mut s4_inters: HashMap<&str, Vec<f32>> = TOOLS.iter().map(|&t| (t, vec![])).collect();
    for (probe_tool, probe_id) in &probe_ids {
        let resolver =
            h.scan_raw_cached(&raw_pf, &raw_manifest, probe_id, best_s1_strat, &s4_cache);
        let intra = resolver.section_score(h.tool_section_ids[*probe_tool], count, &eq);
        let best_inter = TOOLS
            .iter()
            .filter(|&&t| t != *probe_tool)
            .map(|&t| resolver.section_score(h.tool_section_ids[t], count, &eq))
            .fold(f32::NEG_INFINITY, f32::max);
        s4_intras.get_mut(probe_tool).unwrap().push(intra);
        s4_inters.get_mut(probe_tool).unwrap().push(best_inter);
    }
    for &tool in TOOLS {
        let intras = &s4_intras[tool];
        let inters = &s4_inters[tool];
        let mean_intra = intras.iter().sum::<f32>() / intras.len() as f32;
        let mean_inter = inters.iter().sum::<f32>() / inters.len() as f32;
        let min_r = intras
            .iter()
            .zip(inters.iter())
            .map(|(&i, &e)| ratio(i, e))
            .fold(f32::INFINITY, f32::min);
        fprintln!(
            "{:<14} {:>12.1} {:>14.1} {:>9.4} {:>9.4}",
            tool,
            mean_intra,
            mean_inter,
            ratio(mean_intra, mean_inter),
            min_r
        );
    }

    // ── § 5: QK and FloatSimHash layer × head sensitivity ────────────────────
    let sensitivity_pairs: &[(&str, Box<dyn Fn(usize, usize) -> SignatureStrategy>)] = &[
        (
            "QK",
            Box::new(|l, hd| SignatureStrategy::QK { layer: l, head: hd }),
        ),
        (
            "FloatSimHash",
            Box::new(|l, hd| SignatureStrategy::FloatSimHash { layer: l, head: hd }),
        ),
    ];
    for (label, make) in sensitivity_pairs {
        fprintln!("\n╔══════════════════════════════════════════════════════════════════════════╗");
        fprintln!("║  § 5  {label} layer × head sensitivity (mean_ratio)                     ║");
        fprintln!("╚══════════════════════════════════════════════════════════════════════════╝");
        print!("{:<6}", "layer");
        for &head in &test_heads {
            print!("  h{head:<9}");
        }
        // flush after the header row
        fprintln!();
        fprintln!("{}", "─".repeat(6 + test_heads.len() * 12));

        for &l in &test_layers {
            print!("{:<6}", l);
            for &head in &test_heads {
                let strat = make(l, head);
                let (_, mean_r, _) =
                    sweep_ratios(&h, &raw_pf, &raw_manifest, &probe_ids, &strat, count, &eq);
                print!("  {:<10.4}", mean_r);
            }
            // flush after each data row
            fprintln!();
        }
    }

    // ── § 6: BandMean head sensitivity ───────────────────────────────────────
    fprintln!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    fprintln!("║  § 6  BandMean head sensitivity                                         ║");
    fprintln!("╚══════════════════════════════════════════════════════════════════════════╝");
    fprintln!(
        "{:<28} {:>10} {:>12}",
        "strategy",
        "min_ratio",
        "mean_ratio"
    );
    fprintln!("{}", "─".repeat(52));

    for &head in &test_heads {
        for strat in [
            SignatureStrategy::BandMeanQQ { head },
            SignatureStrategy::BandMeanQK { head },
        ] {
            let (min_r, mean_r, _) =
                sweep_ratios(&h, &raw_pf, &raw_manifest, &probe_ids, &strat, count, &eq);
            fprintln!("{:<28} {:>10.4} {:>12.4}", strat.name(), min_r, mean_r);
        }
    }

    // ── § 7: Span per-tool breakdown for best span strategy ──────────────────
    if let Some(span_strat) = all_strategies
        .iter()
        .find(|s| s.name() == best_span_strategy_name)
    {
        fprintln!("\n╔══════════════════════════════════════════════════════════════════════════╗");
        fprintln!(
            "║  § 7  Span per-tool breakdown: {:<43}║",
            format!("{best_span_strategy_name} (best § 3)")
        );
        fprintln!("╚══════════════════════════════════════════════════════════════════════════╝");
        fprintln!(
            "{:<14} {:>10} {:>12} {:>12} {:>9} {:>9}",
            "tool",
            "cnt_mean",
            "sp1.5_mean",
            "sp2.0_mean",
            "cnt_min",
            "sp2_min"
        );
        fprintln!("{}", "─".repeat(68));

        let s7_cache = h.build_raw_corpus_cache(&raw_pf, &raw_manifest, span_strat);
        let mut s7: HashMap<&str, (Vec<f32>, Vec<f32>, Vec<f32>)> = TOOLS
            .iter()
            .map(|&t| (t, (vec![], vec![], vec![])))
            .collect();
        for (probe_tool, probe_id) in &probe_ids {
            let (resolver, hit_log) = h.scan_raw_with_hits_cached(
                &raw_pf,
                &raw_manifest,
                probe_id,
                span_strat,
                &s7_cache,
            );
            let (ic, ec) = discrimination(&h, &resolver, probe_tool, count, &eq);
            let (is15, es15) = span_discrimination(&h, &hit_log, probe_tool, 1.5);
            let (is20, es20) = span_discrimination(&h, &hit_log, probe_tool, 2.0);
            let e = s7.get_mut(probe_tool).unwrap();
            e.0.push(ratio(ic, ec));
            e.1.push(ratio(is15, es15));
            e.2.push(ratio(is20, es20));
        }
        for &tool in TOOLS {
            let (cnt, sp15, sp20) = &s7[tool];
            let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
            let min = |v: &[f32]| v.iter().cloned().fold(f32::INFINITY, f32::min);
            fprintln!(
                "{:<14} {:>10.4} {:>12.4} {:>12.4} {:>9.4} {:>9.4}",
                tool,
                mean(cnt),
                mean(sp15),
                mean(sp20),
                min(cnt),
                min(sp20)
            );
        }
    }

    // ── § 8: Dual-layer combinations ─────────────────────────────────────────
    // Test three ways of combining MH_XOR_QQ_l0 and MH_XOR_QQ_l4, comparing
    // each against the single-layer baselines.
    let l0_strat = SignatureStrategy::MultiHeadXorQQ { layer: 0 };
    let l4_strat = SignatureStrategy::MultiHeadXorQQ { layer: 4 };
    let dual_strat = SignatureStrategy::MultiHeadXorQQDual {
        layer_a: 0,
        layer_b: 4,
    };

    fprintln!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    fprintln!("║  § 8  Dual-layer combinations: l0 × l4 (48 probes)                     ║");
    fprintln!("╚══════════════════════════════════════════════════════════════════════════╝");
    fprintln!(
        "{:<40} {:>10} {:>12} {:>9}",
        "strategy",
        "min_ratio",
        "mean_ratio",
        "max_ratio"
    );
    fprintln!("{}", "─".repeat(74));

    // Baselines (count)
    let (l0_cnt_min, l0_cnt_mean, l0_cnt_max) = sweep_ratios(
        &h,
        &raw_pf,
        &raw_manifest,
        &probe_ids,
        &l0_strat,
        count,
        &eq,
    );
    fprintln!(
        "{:<40} {:>10.4} {:>12.4} {:>9.4}",
        "MH_XOR_QQ_l0 count (baseline)",
        l0_cnt_min,
        l0_cnt_mean,
        l0_cnt_max
    );

    let (l4_cnt_min, l4_cnt_mean, l4_cnt_max) = sweep_ratios(
        &h,
        &raw_pf,
        &raw_manifest,
        &probe_ids,
        &l4_strat,
        count,
        &eq,
    );
    fprintln!(
        "{:<40} {:>10.4} {:>12.4} {:>9.4}",
        "MH_XOR_QQ_l4 count (baseline)",
        l4_cnt_min,
        l4_cnt_mean,
        l4_cnt_max
    );

    // Baseline span2.0 for l0 (the champion from § 7)
    {
        let s8_l0_cache = h.build_raw_corpus_cache(&raw_pf, &raw_manifest, &l0_strat);
        let s8_sp20_ratios: Vec<f32> = probe_ids
            .iter()
            .map(|(probe_tool, probe_id)| {
                let (_, hit_log) = h.scan_raw_with_hits_cached(
                    &raw_pf,
                    &raw_manifest,
                    probe_id,
                    &l0_strat,
                    &s8_l0_cache,
                );
                let (is20, es20) = span_discrimination(&h, &hit_log, probe_tool, 2.0);
                ratio(is20, es20)
            })
            .collect();
        let sp_min = s8_sp20_ratios.iter().cloned().fold(f32::INFINITY, f32::min);
        let sp_mean = s8_sp20_ratios.iter().sum::<f32>() / s8_sp20_ratios.len() as f32;
        let sp_max = s8_sp20_ratios
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        fprintln!(
            "{:<40} {:>10.4} {:>12.4} {:>9.4}",
            "MH_XOR_QQ_l0 span2.0 (baseline)",
            sp_min,
            sp_mean,
            sp_max
        );
    }

    fprintln!("{}", "─".repeat(74));

    // Option A — 8-head XOR, count
    let (a_cnt_min, a_cnt_mean, a_cnt_max) = sweep_ratios(
        &h,
        &raw_pf,
        &raw_manifest,
        &probe_ids,
        &dual_strat,
        count,
        &eq,
    );
    fprintln!(
        "{:<40} {:>10.4} {:>12.4} {:>9.4}",
        "A: MH_XOR_QQ_l0xl4 count",
        a_cnt_min,
        a_cnt_mean,
        a_cnt_max
    );

    // Option A — 8-head XOR, span α=2.0
    let (a_sp_min, a_sp_mean, a_sp_max) = {
        let s8_a_cache = h.build_raw_corpus_cache(&raw_pf, &raw_manifest, &dual_strat);
        let sp_ratios: Vec<f32> = probe_ids
            .iter()
            .map(|(probe_tool, probe_id)| {
                let (_, hit_log) = h.scan_raw_with_hits_cached(
                    &raw_pf,
                    &raw_manifest,
                    probe_id,
                    &dual_strat,
                    &s8_a_cache,
                );
                let (is20, es20) = span_discrimination(&h, &hit_log, probe_tool, 2.0);
                ratio(is20, es20)
            })
            .collect();
        let sp_min = sp_ratios.iter().cloned().fold(f32::INFINITY, f32::min);
        let sp_mean = sp_ratios.iter().sum::<f32>() / sp_ratios.len() as f32;
        let sp_max = sp_ratios.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        (sp_min, sp_mean, sp_max)
    };
    fprintln!(
        "{:<40} {:>10.4} {:>12.4} {:>9.4}",
        "A: MH_XOR_QQ_l0xl4 span2.0",
        a_sp_min,
        a_sp_mean,
        a_sp_max
    );

    // Option B — additive: norm_span2.0(l0) + norm_count(l4)
    let (b_min, b_mean, b_max) = sweep_ratios_additive_b(
        &h,
        &raw_pf,
        &raw_manifest,
        &probe_ids,
        &l0_strat,
        &l4_strat,
        2.0,
    );
    fprintln!(
        "{:<40} {:>10.4} {:>12.4} {:>9.4}",
        "B: norm span2.0(l0) + count(l4)",
        b_min,
        b_mean,
        b_max
    );

    // Option C — gated span: l0 span filtered by l4 gate, α=2.0
    let (c_min, c_mean, c_max) = sweep_ratios_gated_c(
        &h,
        &raw_pf,
        &raw_manifest,
        &probe_ids,
        &l0_strat,
        &l4_strat,
        2.0,
    );
    fprintln!(
        "{:<40} {:>10.4} {:>12.4} {:>9.4}",
        "C: gated span2.0(l0, gate=l4)",
        c_min,
        c_mean,
        c_max
    );

    fprintln!("{}", "─".repeat(74));

    // Identify winner among A-count, A-span, B, C by (min_ratio DESC, mean_ratio DESC)
    let s8_candidates = [
        ("A-count", a_cnt_min, a_cnt_mean),
        ("A-span2.0", a_sp_min, a_sp_mean),
        ("B", b_min, b_mean),
        ("C", c_min, c_mean),
    ];
    let s8_winner = s8_candidates
        .iter()
        .max_by(|x, y| {
            x.1.partial_cmp(&y.1)
                .unwrap()
                .then(x.2.partial_cmp(&y.2).unwrap())
        })
        .unwrap();
    fprintln!(
        "§ 8 winner: {}  (min={:.4}  mean={:.4})",
        s8_winner.0,
        s8_winner.1,
        s8_winner.2
    );

    // Per-tool breakdown for Option A span2.0 (most structurally interesting)
    fprintln!("\n  Per-tool breakdown — A: MH_XOR_QQ_l0xl4 span2.0");
    fprintln!(
        "  {:<14} {:>10} {:>12} {:>9} {:>9}",
        "tool",
        "cnt_mean",
        "sp2.0_mean",
        "cnt_min",
        "sp2_min"
    );
    fprintln!("  {}", "─".repeat(57));
    {
        let s8_a_cache = h.build_raw_corpus_cache(&raw_pf, &raw_manifest, &dual_strat);
        let mut a_cnt_by_tool: HashMap<&str, Vec<f32>> =
            TOOLS.iter().map(|&t| (t, vec![])).collect();
        let mut a_sp20_by_tool: HashMap<&str, Vec<f32>> =
            TOOLS.iter().map(|&t| (t, vec![])).collect();
        for (probe_tool, probe_id) in &probe_ids {
            let (resolver, hit_log) = h.scan_raw_with_hits_cached(
                &raw_pf,
                &raw_manifest,
                probe_id,
                &dual_strat,
                &s8_a_cache,
            );
            let (ic, ec) = discrimination(&h, &resolver, probe_tool, count, &eq);
            let (is20, es20) = span_discrimination(&h, &hit_log, probe_tool, 2.0);
            a_cnt_by_tool
                .get_mut(probe_tool)
                .unwrap()
                .push(ratio(ic, ec));
            a_sp20_by_tool
                .get_mut(probe_tool)
                .unwrap()
                .push(ratio(is20, es20));
        }
        for &tool in TOOLS {
            let cnt = &a_cnt_by_tool[tool];
            let sp20 = &a_sp20_by_tool[tool];
            let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
            let min = |v: &[f32]| v.iter().cloned().fold(f32::INFINITY, f32::min);
            fprintln!(
                "  {:<14} {:>10.4} {:>12.4} {:>9.4} {:>9.4}",
                tool,
                mean(cnt),
                mean(sp20),
                min(cnt),
                min(sp20)
            );
        }
    }

    fprintln!("\n── Run complete ────────────────────────────────────────────────────────────");
    fprintln!("Best strategy (§ 1, Count):  {best_s1_name}");
    fprintln!(
        "Best strategy (§ 3, Span):   {best_span_strategy_name}  (mean_ratio={best_span_mean:.4})"
    );
    fprintln!(
        "Best dual-layer (§ 8):       {}  (min={:.4}  mean={:.4})",
        s8_winner.0,
        s8_winner.1,
        s8_winner.2
    );
    fprintln!();
}
