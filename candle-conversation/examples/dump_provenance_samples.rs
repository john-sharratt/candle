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
    /// Directory containing MANIFEST.json, signatures.prov, and
    /// optionally prefill_signatures.prov.
    #[arg(long, default_value_os_t = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/tool_provenance_real_data"
    )))]
    data_dir: PathBuf,

    /// Number of scenarios to display in the header dump section.
    #[arg(long, default_value = "8")]
    show_n: usize,

    /// Hit threshold for the decode-phase scan (0–128; random baseline = 64).
    #[arg(long, default_value = "80")]
    decode_threshold: u32,

    /// Hit threshold for the prefill-phase scan.  Prefill Q vectors tend to
    /// be noisier, so a lower threshold often recovers more signal.
    #[arg(long, default_value = "72")]
    prefill_threshold: u32,

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
    /// Decode-phase Q vectors.
    byte_offset: u64,
    token_count: u16,
    /// Prefill-phase Q vectors (present only if generated with the dual path).
    #[serde(default)]
    prefill_byte_offset: Option<u64>,
    #[serde(default)]
    prefill_token_count: Option<u16>,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    model: String,
    provenance_layer_indices: [usize; 6],
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

fn intra_inter(
    pf: &ProvenanceFile,
    scenarios: &[Scenario],
    entries: &[(u64, SigEntry)],
    tool: &str,
    threshold: u32,
) -> anyhow::Result<(f64, f64)> {
    let pos_entries: Vec<(usize, &Scenario)> = scenarios
        .iter()
        .enumerate()
        .filter(|(_, s)| s.tool.as_deref() == Some(tool) && s.case_type == CaseType::Positive)
        .collect();
    if pos_entries.is_empty() {
        return Ok((0.0, 0.0));
    }
    let probe_idx = pos_entries[0].0;
    let probe_entry = entries[probe_idx].1;
    let (syn, sem, prag) = pf.read_entry(probe_entry)?;
    if syn.is_empty() {
        return Ok((0.0, 0.0));
    }
    let probe = ProbeSignatures {
        syntactic: *syn.last().unwrap(),
        semantic: *sem.last().unwrap(),
        pragmatic: *prag.last().unwrap(),
    };
    let all_ranks = pf.scan_entries(entries, &probe, threshold, scenarios.len())?;
    let mut intra: Vec<f64> = Vec::new();
    let mut inter: Vec<f64> = Vec::new();
    for r in &all_ranks {
        let m = &scenarios[r.turn_id as usize];
        if m.tool.as_deref() == Some(tool) {
            intra.push(r.score);
        } else {
            inter.push(r.score);
        }
    }
    let intra_avg = if intra.is_empty() { 0.0 } else { intra.iter().sum::<f64>() / intra.len() as f64 };
    let inter_avg = if inter.is_empty() { 0.0 } else { inter.iter().sum::<f64>() / inter.len() as f64 };
    Ok((intra_avg, inter_avg))
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let manifest_path = args.data_dir.join("MANIFEST.json");
    let sig_path = args.data_dir.join("signatures.prov");
    let prefill_sig_path = args.data_dir.join("prefill_signatures.prov");

    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).map_err(|e| {
            anyhow::anyhow!("failed to read {}: {}", manifest_path.display(), e)
        })?)?;

    let has_prefill = prefill_sig_path.exists()
        && manifest.scenarios.iter().any(|s| s.prefill_byte_offset.is_some());

    println!("=== Provenance Sample Dump ===");
    println!("Model  : {}", manifest.model);
    println!("Layers : {:?}", manifest.provenance_layer_indices);
    println!("Scenarios: {}", manifest.scenarios.len());
    println!("Decode file : {}", sig_path.display());
    if has_prefill {
        println!("Prefill file: {}", prefill_sig_path.display());
        println!("Mode: two-round (prefill → decode)");
        println!("  prefill threshold = {}  decode threshold = {}",
            args.prefill_threshold, args.decode_threshold);
    } else {
        println!("Mode: decode-only (no prefill_signatures.prov or missing prefill fields)");
        println!("  decode threshold = {}", args.decode_threshold);
    }
    println!();

    let pf = ProvenanceFile::open(&sig_path)?;
    let prefill_pf = if has_prefill {
        Some(ProvenanceFile::open(&prefill_sig_path)?)
    } else {
        None
    };

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
            "\n[{}] tool={:?} case={:?} decode_tokens={}",
            s.id, s.tool, s.case_type, s.token_count
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
        if let (Some(ppf), Some(off), Some(tc)) =
            (&prefill_pf, s.prefill_byte_offset, s.prefill_token_count)
        {
            let pe = SigEntry { byte_offset: off, token_count: tc };
            if let Ok((ps, _, _)) = ppf.read_entry(pe) {
                println!(
                    "  prefill_density syn={:.1}%  ({} tokens)",
                    bit_density(&ps) * 100.0,
                    tc
                );
            }
        }
    }

    // Build decode and (optional) prefill entry indices.
    let decode_entries: Vec<(u64, SigEntry)> = manifest
        .scenarios
        .iter()
        .enumerate()
        .map(|(i, s)| (i as u64, SigEntry { byte_offset: s.byte_offset, token_count: s.token_count }))
        .collect();

    let prefill_entries: Vec<(u64, SigEntry)> = manifest
        .scenarios
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            Some((i as u64, SigEntry {
                byte_offset: s.prefill_byte_offset?,
                token_count: s.prefill_token_count?,
            }))
        })
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

    // ── Similarity scan per tool ──────────────────────────────────────────────

    println!("\n── Similarity scan (top_k={}) ──────────────────────────", args.top_k);

    for tool in &seen_tools {
        let first_pos = manifest.scenarios.iter().enumerate().find(|(_, s)| {
            s.tool.as_deref() == Some(tool) && s.case_type == CaseType::Positive
        });
        let Some((probe_idx, probe_s)) = first_pos else { continue };

        println!("\nTool: {tool}");
        let probe_user_preview: String = probe_s.user_prompt.chars().take(60).collect();
        println!("  probe: {} | \"{}\"", probe_s.id, probe_user_preview);

        // Round 1: prefill probe (if available).
        if let (Some(ppf), true) = (&prefill_pf, !prefill_entries.is_empty()) {
            if let Some(&(_, pre_entry)) = prefill_entries.iter().find(|(i, _)| *i == probe_idx as u64) {
                let (syn, sem, prag) = ppf.read_entry(pre_entry)?;
                if !syn.is_empty() {
                    let probe = ProbeSignatures {
                        syntactic: *syn.last().unwrap(),
                        semantic: *sem.last().unwrap(),
                        pragmatic: *prag.last().unwrap(),
                    };
                    let ranks = ppf.scan_entries(&prefill_entries, &probe, args.prefill_threshold, args.top_k)?;
                    println!("  [prefill  thr={}]", args.prefill_threshold);
                    if ranks.is_empty() {
                        println!("    (no results above threshold)");
                    } else {
                        for r in &ranks {
                            let m = &manifest.scenarios[r.turn_id as usize];
                            let mark = if m.tool.as_deref() == Some(tool) { "✓" } else { "✗" };
                            println!("    {} [{:6.2}]  {}", mark, r.score, m.id);
                        }
                    }
                }
            }
        }

        // Round 2: decode probe.
        let probe_entry = decode_entries[probe_idx].1;
        let (syn, sem, prag) = pf.read_entry(probe_entry)?;
        if !syn.is_empty() {
            let probe = ProbeSignatures {
                syntactic: *syn.last().unwrap(),
                semantic: *sem.last().unwrap(),
                pragmatic: *prag.last().unwrap(),
            };
            let ranks = pf.scan_entries(&decode_entries, &probe, args.decode_threshold, args.top_k)?;
            println!("  [decode   thr={}]", args.decode_threshold);
            if ranks.is_empty() {
                println!("    (no results above threshold)");
            } else {
                for r in &ranks {
                    let m = &manifest.scenarios[r.turn_id as usize];
                    let mark = if m.tool.as_deref() == Some(tool) { "✓" } else { "✗" };
                    println!("    {} [{:6.2}]  {}", mark, r.score, m.id);
                }
            }
        }
    }

    // ── Intra vs inter-tool score summary ─────────────────────────────────────

    if has_prefill {
        println!("\n── Intra vs inter-tool score summary (prefill | decode) ─");
        println!("{:<14} {:>10} {:>10} {:>8}  {:>10} {:>10} {:>8}",
            "tool",
            "pre-intra", "pre-inter", "pre-ratio",
            "dec-intra", "dec-inter", "dec-ratio");

        for tool in &seen_tools {
            let ppf = prefill_pf.as_ref().unwrap();
            let (pre_intra, pre_inter) = intra_inter(
                ppf, &manifest.scenarios, &prefill_entries, tool, args.prefill_threshold)?;
            let (dec_intra, dec_inter) = intra_inter(
                &pf, &manifest.scenarios, &decode_entries, tool, args.decode_threshold)?;
            let pre_ratio = if pre_inter > 0.0 { pre_intra / pre_inter } else { f64::INFINITY };
            let dec_ratio = if dec_inter > 0.0 { dec_intra / dec_inter } else { f64::INFINITY };
            println!("{:<14} {:>10.3} {:>10.3} {:>8.2}  {:>10.3} {:>10.3} {:>8.2}",
                tool,
                pre_intra, pre_inter, pre_ratio,
                dec_intra, dec_inter, dec_ratio);
        }
    } else {
        println!("\n── Intra vs inter-tool score summary ────────────────────");
        println!("{:<14} {:>12} {:>12} {:>10}", "tool", "intra avg", "inter avg", "ratio");

        for tool in &seen_tools {
            let (intra_avg, inter_avg) = intra_inter(
                &pf, &manifest.scenarios, &decode_entries, tool, args.decode_threshold)?;
            let ratio = if inter_avg > 0.0 { intra_avg / inter_avg } else { f64::INFINITY };
            println!("{:<14} {:>12.3} {:>12.3} {:>10.2}", tool, intra_avg, inter_avg, ratio);
        }
    }

    Ok(())
}

