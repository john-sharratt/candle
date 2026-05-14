//! Dump provenance signature samples from a real data directory.
//!
//! Loads MANIFEST.json + signatures.prov from a real-data directory and
//! prints per-scenario signature bytes, bit-density stats, and cross-scenario
//! similarity scan scores so you can eyeball whether the real Q vectors
//! actually separate tool topics.
//!
//! Usage:
//!   cargo run -p candle-conversation --example dump_provenance_samples -- \
//!     --data-dir tests/tool_provenance_real_data
//!
//! Expected output for good data:
//!   • Same-tool scenario pairs score >> different-tool pairs in the scan.
//!   • Bit density near 50% (random-projection sign bits are roughly Bernoulli).
//!   • No all-zero or all-one signatures (would indicate a quantisation bug).

use candle_conversation::{ProbeSignatures, ProvenanceFile, SigEntry};
use clap::Parser;
use serde::Deserialize;
use std::path::PathBuf;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(about = "Dump and validate real provenance signature samples")]
struct Args {
    /// Directory containing MANIFEST.json and signatures.prov.
    #[arg(long, default_value_os_t = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/tool_provenance_real_data"
    )))]
    data_dir: PathBuf,

    /// Number of scenarios to display in the header dump section.
    #[arg(long, default_value = "8")]
    show_n: usize,

    /// Hit threshold for scan_entries (0–128; random baseline = 64).
    #[arg(long, default_value = "80")]
    threshold: u32,

    /// Show the top-K matches per probe in the similarity scan.
    #[arg(long, default_value = "5")]
    top_k: usize,
}

// ── Manifest types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CaseType {
    Positive,
    Boundary,
    Negative,
    NoTool,
}

#[derive(Debug, Clone, Deserialize)]
struct Scenario {
    id: String,
    tool: Option<String>,
    case_type: CaseType,
    user_prompt: String,
    generated_text: String,
    byte_offset: u64,
    token_count: u16,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    model: String,
    provenance_layer_indices: [usize; 3],
    scenarios: Vec<Scenario>,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn hex_u128(v: u128) -> String {
    format!("{v:032x}")
}

fn bit_density(sigs: &[candle_conversation::TokenSignature]) -> f64 {
    if sigs.is_empty() {
        return 0.0;
    }
    let ones: u32 = sigs.iter().map(|s| s.as_u128().count_ones()).sum();
    ones as f64 / (sigs.len() as f64 * 128.0)
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let manifest_path = args.data_dir.join("MANIFEST.json");
    let sig_path = args.data_dir.join("signatures.prov");

    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).map_err(|e| {
            anyhow::anyhow!("failed to read {}: {}", manifest_path.display(), e)
        })?)?;

    println!("=== Provenance Sample Dump ===");
    println!("Model  : {}", manifest.model);
    println!("Layers : {:?}", manifest.provenance_layer_indices);
    println!("Scenarios: {}", manifest.scenarios.len());
    println!("File   : {}", sig_path.display());
    println!();

    let pf = ProvenanceFile::open(&sig_path)?;

    // ── Per-scenario header dump ──────────────────────────────────────────────

    println!("── Signature samples (first {} scenarios) ──────────────", args.show_n);
    for s in manifest.scenarios.iter().take(args.show_n) {
        let entry = SigEntry { byte_offset: s.byte_offset, token_count: s.token_count };
        let (syn, sem, prag) = pf.read_entry(entry)?;

        if syn.is_empty() {
            println!("[{}] EMPTY — skipping", s.id);
            continue;
        }

        println!(
            "\n[{}] tool={:?} case={:?} tokens={}",
            s.id,
            s.tool,
            s.case_type,
            s.token_count
        );
        let user_preview: String = s.user_prompt.chars().take(70).collect();
        let gen_preview: String = s.generated_text.chars().take(70).collect();
        println!("  user   : {}", user_preview);
        println!("  gen    : {}", gen_preview);
        println!("  Syn[0] : {}", hex_u128(syn[0].as_u128()));
        println!("  Sem[0] : {}", hex_u128(sem[0].as_u128()));
        println!("  Prg[0] : {}", hex_u128(prag[0].as_u128()));
        if syn.len() > 1 {
            println!("  Syn[-1]: {}", hex_u128(syn.last().unwrap().as_u128()));
            println!("  Sem[-1]: {}", hex_u128(sem.last().unwrap().as_u128()));
            println!("  Prg[-1]: {}", hex_u128(prag.last().unwrap().as_u128()));
        }
        println!(
            "  density syn={:.1}%  sem={:.1}%  prg={:.1}%",
            bit_density(&syn) * 100.0,
            bit_density(&sem) * 100.0,
            bit_density(&prag) * 100.0,
        );
    }

    // ── Similarity scan per tool ──────────────────────────────────────────────

    println!("\n── Similarity scan (threshold={} top_k={}) ─────────────", args.threshold, args.top_k);

    // Build full entries index.
    let entries: Vec<(u64, SigEntry)> = manifest
        .scenarios
        .iter()
        .enumerate()
        .map(|(i, s)| (i as u64, SigEntry { byte_offset: s.byte_offset, token_count: s.token_count }))
        .collect();

    // Collect unique tool names (preserving first-occurrence order).
    let mut seen_tools: Vec<&str> = Vec::new();
    for s in &manifest.scenarios {
        if let Some(t) = s.tool.as_deref() {
            if !seen_tools.contains(&t) {
                seen_tools.push(t);
            }
        }
    }

    for tool in &seen_tools {
        // Probe: first positive scenario for this tool, using its last token.
        let first_pos = manifest.scenarios.iter().enumerate().find(|(_, s)| {
            s.tool.as_deref() == Some(tool) && s.case_type == CaseType::Positive
        });
        let Some((probe_idx, probe_s)) = first_pos else {
            continue;
        };

        let probe_entry = entries[probe_idx].1;
        let (syn, sem, prag) = pf.read_entry(probe_entry)?;
        if syn.is_empty() {
            continue;
        }

        // Use the last token's signature as the query probe.
        let probe = ProbeSignatures {
            syntactic: *syn.last().unwrap(),
            semantic: *sem.last().unwrap(),
            pragmatic: *prag.last().unwrap(),
        };

        let ranks = pf.scan_entries(&entries, &probe, args.threshold, args.top_k)?;

        let probe_user_preview: String = probe_s.user_prompt.chars().take(60).collect();
        println!("\nProbe: {} | {} tokens", probe_s.id, probe_s.token_count);
        println!("       user: {}", probe_user_preview);
        if ranks.is_empty() {
            println!("  (no results above threshold)");
            continue;
        }
        for r in &ranks {
            let m = &manifest.scenarios[r.turn_id as usize];
            let mark = if m.tool.as_deref() == Some(tool) { "✓" } else { "✗" };
            println!(
                "  {} [{:6.2}] hits={:?}  {}",
                mark, r.score, r.hit_counts, m.id
            );
        }
    }

    // ── Intra-tool vs inter-tool score summary ────────────────────────────────

    println!("\n── Intra vs inter-tool score summary ────────────────────");
    println!("{:<14} {:>12} {:>12} {:>10}", "tool", "intra avg", "inter avg", "ratio");

    for tool in &seen_tools {
        // All positive entries for this tool.
        let pos_entries: Vec<(usize, &Scenario)> = manifest
            .scenarios
            .iter()
            .enumerate()
            .filter(|(_, s)| s.tool.as_deref() == Some(tool) && s.case_type == CaseType::Positive)
            .collect();

        if pos_entries.is_empty() {
            continue;
        }

        let probe_idx = pos_entries[0].0;
        let probe_entry = entries[probe_idx].1;
        let (syn, sem, prag) = pf.read_entry(probe_entry)?;
        if syn.is_empty() {
            continue;
        }
        let probe = ProbeSignatures {
            syntactic: *syn.last().unwrap(),
            semantic: *sem.last().unwrap(),
            pragmatic: *prag.last().unwrap(),
        };

        let all_ranks = pf.scan_entries(&entries, &probe, 64, manifest.scenarios.len())?;

        let mut intra_scores: Vec<f64> = Vec::new();
        let mut inter_scores: Vec<f64> = Vec::new();
        for r in &all_ranks {
            let m = &manifest.scenarios[r.turn_id as usize];
            if m.tool.as_deref() == Some(tool) {
                intra_scores.push(r.score);
            } else {
                inter_scores.push(r.score);
            }
        }

        let intra_avg = if intra_scores.is_empty() {
            0.0
        } else {
            intra_scores.iter().sum::<f64>() / intra_scores.len() as f64
        };
        let inter_avg = if inter_scores.is_empty() {
            0.0
        } else {
            inter_scores.iter().sum::<f64>() / inter_scores.len() as f64
        };
        let ratio = if inter_avg > 0.0 { intra_avg / inter_avg } else { f64::INFINITY };

        println!("{:<14} {:>12.3} {:>12.3} {:>10.2}", tool, intra_avg, inter_avg, ratio);
    }

    Ok(())
}
