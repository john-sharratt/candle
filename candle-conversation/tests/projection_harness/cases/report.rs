//! Non-assertion diagnostic report: 8×8 BDP score matrix, intra/inter ratios,
//! and projection top-3 selection per tool.
//!
//! Run with `-- --nocapture` to see output.

use candle_conversation::provenance::TokenHit;

use crate::corpus::{load_fixtures, TOOLS};
use crate::harness::Harness;

#[test]
fn score_matrix_and_ratio_report() {
    let (manifest, pf) = load_fixtures();
    let h = Harness::build();
    let target = h.target();
    let resolvers = h.scan_all_pos1(&pf, &manifest);

    let print_matrix = |use_max: bool| {
        let label = if use_max { "max-agreement" } else { "mean-agreement" };
        println!(
            "\n── BDP section score matrix [{label}]  (probe _pos_1, corpus _pos_0,2–5) ──"
        );
        let header: String = TOOLS
            .iter()
            .map(|t| format!("{:>11}", &t[..t.len().min(11)]))
            .collect::<Vec<_>>()
            .join(" ");
        println!("{:>14}  {header}  (intra*)", "probe ↓");

        for (probe_tool, resolver) in &resolvers {
            let row: String = TOOLS
                .iter()
                .map(|&ct| {
                    let score = h.section_score_formula(resolver, ct, use_max);
                    let mark = if ct == *probe_tool { "*" } else { " " };
                    format!("{score:>11.2}{mark}")
                })
                .collect::<Vec<_>>()
                .join("");
            println!("{:>14}  {row}", probe_tool);
        }

        println!("\n── Intra / inter-tool ratio [{label}] ─────────────────────────────────");
        println!("{:<14} {:>10} {:>12} {:>9}", "tool", "intra", "inter_avg", "ratio");
        for (probe_tool, resolver) in &resolvers {
            let intra = h.section_score_formula(resolver, probe_tool, use_max);
            let inter: Vec<f32> = TOOLS
                .iter()
                .filter(|&&t| t != *probe_tool)
                .map(|&t| h.section_score_formula(resolver, t, use_max))
                .collect();
            let inter_avg = inter.iter().sum::<f32>() / inter.len() as f32;
            let ratio = if inter_avg > 0.0 { intra / inter_avg } else { f32::INFINITY };
            println!(
                "{:<14} {:>10.2} {:>12.2} {:>9.2}",
                probe_tool, intra, inter_avg, ratio
            );
        }
    };

    print_matrix(true);
    print_matrix(false);

    println!("\n── Projection top-3 selections ────────────────────────────────────────");
    println!("{:<14} {:>6}  {}", "probe", "ok?", "emitted tools");
    for (probe_tool, resolver) in &resolvers {
        let projection = h.builder.project(target, resolver);
        let emitted = h.emitted_tools(&projection);
        let hit = emitted.contains(probe_tool);
        println!(
            "{:<14} {:>6}  {:?}",
            probe_tool, if hit { "✓" } else { "✗" }, emitted
        );
    }
    println!();
}

/// Hit-token report: for a given probe, shows where (probe_tok, corpus_tok)
/// pairs above the hit threshold cluster — separately for the intra-tool
/// match and the highest-scoring inter-tool match.
///
/// Run with `-- --nocapture` to see output.
#[test]
fn hit_token_report() {
    let (manifest, pf) = load_fixtures();
    let h = Harness::build();

    // Use weather as the probe tool (first in TOOLS, semantically clean).
    let probe_tool = "weather";
    let probe_id = "weather_pos_1";

    let (resolver, hit_log) = h.scan_with_hits(&pf, &manifest, probe_id);

    // Find the highest-scoring inter-tool section.
    let inter_tool = TOOLS
        .iter()
        .filter(|&&t| t != probe_tool)
        .max_by(|&&a, &&b| {
            h.section_score(&resolver, a)
                .partial_cmp(&h.section_score(&resolver, b))
                .unwrap()
        })
        .copied()
        .unwrap();

    println!(
        "\n── Hit token report: {probe_id} probe ─────────────────────────────────"
    );
    println!(
        "  intra tool : {probe_tool}  score={:.2}",
        h.section_score(&resolver, probe_tool)
    );
    println!(
        "  best inter : {inter_tool}  score={:.2}",
        h.section_score(&resolver, inter_tool)
    );

    let depth_names = ["syn", "sem", "prag"];

    for &focus_tool in &[probe_tool, inter_tool] {
        let sid = h.tool_section_ids[focus_tool];
        let hits = match hit_log.get(&sid) {
            Some(h) => h,
            None => {
                println!("\n  [{focus_tool}]  no hits recorded");
                continue;
            }
        };

        println!("\n  ── [{focus_tool}]  {} total hits ─────────────────", hits.len());

        for depth in 0u8..3 {
            let depth_hits: Vec<&TokenHit> = hits.iter().filter(|h| h.depth == depth).collect();
            if depth_hits.is_empty() {
                continue;
            }

            // Probe-token frequency histogram.
            let max_probe_tok = depth_hits.iter().map(|h| h.probe_tok).max().unwrap_or(0);
            let mut probe_freq = vec![0u32; max_probe_tok as usize + 1];
            for hit in &depth_hits {
                probe_freq[hit.probe_tok as usize] += 1;
            }

            // Corpus-token frequency histogram.
            let max_corpus_tok = depth_hits.iter().map(|h| h.corpus_tok).max().unwrap_or(0);
            let mut corpus_freq = vec![0u32; max_corpus_tok as usize + 1];
            for hit in &depth_hits {
                corpus_freq[hit.corpus_tok as usize] += 1;
            }

            let max_freq = probe_freq.iter().chain(corpus_freq.iter()).copied().max().unwrap_or(1);
            let bar_scale = 30.0 / max_freq as f32;

            println!(
                "\n    depth={} ({})  {} hits  avg_agreement={:.1}",
                depth,
                depth_names[depth as usize],
                depth_hits.len(),
                depth_hits.iter().map(|h| h.agreement as f32).sum::<f32>()
                    / depth_hits.len() as f32,
            );

            println!("    probe_tok hit freq:");
            for (tok, &freq) in probe_freq.iter().enumerate() {
                if freq == 0 { continue; }
                let bar = "#".repeat((freq as f32 * bar_scale).round() as usize);
                println!("      tok {:>3}: {:<30} {}", tok, bar, freq);
            }

            println!("    corpus_tok hit freq:");
            for (tok, &freq) in corpus_freq.iter().enumerate() {
                if freq == 0 { continue; }
                let bar = "#".repeat((freq as f32 * bar_scale).round() as usize);
                println!("      tok {:>3}: {:<30} {}", tok, bar, freq);
            }
        }
    }
    println!();
}
