//! Full-corpus verification of generated provenance data.
//!
//! Checks every scenario in tool_provenance_real_data for:
//!   1. Completeness  — no scenarios missing, no zero token_count
//!   2. No padding    — no all-zero TokenSignatures (uninitialized arena slots)
//!   3. Quality       — bit density in [25%, 75%] for every token's signature
//!   4. Consistency   — byte_offset + token_count stays within file bounds
//!
//! Run:
//!   cargo run -p candle-conversation --example verify_provenance_data
//!
//! Exits non-zero if any check fails.

use candle_conversation::provenance::store::ENTRY_BYTES_PER_TOKEN;
use candle_conversation::{ProvenanceFile, SigEntry, TokenSignature};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)]
enum CaseType {
    Positive,
    Boundary,
    Negative,
    NoTool,
}

#[derive(Debug, Deserialize)]
struct Scenario {
    id: String,
    tool: Option<String>,
    case_type: CaseType,
    generated_text: String,
    byte_offset: u64,
    token_count: u16,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    scenarios: Vec<Scenario>,
}

fn bit_density(sigs: &[TokenSignature]) -> f64 {
    if sigs.is_empty() {
        return 0.0;
    }
    let ones: u32 = sigs.iter().map(|s| s.as_u128().count_ones()).sum();
    ones as f64 / (sigs.len() as f64 * 128.0)
}

fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/tool_provenance_real_data"
    ));

    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("MANIFEST.json"))?)?;
    let sig_path = dir.join("signatures.prov");
    let file_bytes = std::fs::metadata(&sig_path)?.len();
    let pf = ProvenanceFile::open(&sig_path)?;

    let total = manifest.scenarios.len();
    println!("Verifying {total} scenarios  ({file_bytes} bytes in signatures.prov)");
    println!();

    let mut errors: Vec<String> = Vec::new();
    let mut zero_tok_total = 0usize;
    let mut density_lo_total = 0usize;
    let mut density_hi_total = 0usize;

    // Per-depth density accumulators for the global summary.
    let mut sum_den = [0f64; 3];
    let mut sum_n = [0usize; 3];

    for s in &manifest.scenarios {
        // ── 1. Non-zero token count ────────────────────────────────────────
        if s.token_count == 0 {
            errors.push(format!("[{}] token_count = 0", s.id));
            continue;
        }

        let entry = SigEntry {
            byte_offset: s.byte_offset,
            token_count: s.token_count,
        };

        // ── 2. Byte range within file ──────────────────────────────────────
        use ENTRY_BYTES_PER_TOKEN;
        let end_byte = s.byte_offset + s.token_count as u64 * ENTRY_BYTES_PER_TOKEN as u64;
        if end_byte > file_bytes {
            errors.push(format!(
                "[{}] byte range [{}, {}) exceeds file size {}",
                s.id, s.byte_offset, end_byte, file_bytes
            ));
            continue;
        }

        // ── 3. Read back ───────────────────────────────────────────────────
        let (syn, sem, prag) = match pf.read_entry(entry) {
            Ok(t) => t,
            Err(e) => {
                errors.push(format!("[{}] read_entry failed: {e}", s.id));
                continue;
            }
        };

        if syn.len() != s.token_count as usize
            || sem.len() != s.token_count as usize
            || prag.len() != s.token_count as usize
        {
            errors.push(format!(
                "[{}] token_count={} but read back syn={} sem={} prag={}",
                s.id,
                s.token_count,
                syn.len(),
                sem.len(),
                prag.len()
            ));
            continue;
        }

        // ── 4. No all-zero signatures (zero-padding from uninitialized arena) ──
        let depths = [("syn", &syn), ("sem", &sem), ("prag", &prag)];
        for (name, sigs) in &depths {
            let zeros: usize = sigs.iter().filter(|t| t.as_u128() == 0).count();
            if zeros > 0 {
                zero_tok_total += zeros;
                errors.push(format!(
                    "[{}] {name}: {zeros}/{} tokens are all-zero (padding artifact)",
                    s.id,
                    sigs.len()
                ));
            }
        }

        // ── 5. Bit density [25%, 75%] ──────────────────────────────────────
        for (i, (name, sigs)) in depths.iter().enumerate() {
            let d = bit_density(sigs);
            sum_den[i] += d;
            sum_n[i] += 1;
            if d < 0.25 {
                density_lo_total += 1;
                errors.push(format!(
                    "[{}] {name}: bit density {:.1}% < 25% (Q vectors look degenerate)",
                    s.id,
                    d * 100.0
                ));
            } else if d > 0.75 {
                density_hi_total += 1;
                errors.push(format!(
                    "[{}] {name}: bit density {:.1}% > 75% (Q vectors look degenerate)",
                    s.id,
                    d * 100.0
                ));
            }
        }
    }

    // ── Summary ───────────────────────────────────────────────────────────────

    println!("── Density summary (all scenarios) ──────────────────────────");
    let labels = ["syn", "sem", "prag"];
    for i in 0..3 {
        let avg = if sum_n[i] > 0 {
            sum_den[i] / sum_n[i] as f64
        } else {
            0.0
        };
        println!("  {}: avg bit density {:.2}%", labels[i], avg * 100.0);
    }

    // Total token + file stats.
    let total_tokens: u64 = manifest
        .scenarios
        .iter()
        .map(|s| s.token_count as u64)
        .sum();
    let expected_bytes = total_tokens * ENTRY_BYTES_PER_TOKEN as u64;
    println!();
    println!("── Token / file stats ───────────────────────────────────────");
    println!("  scenarios     : {total}");
    println!("  total tokens  : {total_tokens}");
    println!("  expected bytes: {expected_bytes}");
    println!("  actual bytes  : {file_bytes}");
    if file_bytes == expected_bytes {
        println!("  file size     : OK (exact match)");
    } else {
        println!(
            "  file size     : MISMATCH (delta = {})",
            file_bytes as i64 - expected_bytes as i64
        );
        errors.push(format!(
            "file size mismatch: expected {expected_bytes}, got {file_bytes}"
        ));
    }

    // Token distribution.
    let mut counts: Vec<u16> = manifest.scenarios.iter().map(|s| s.token_count).collect();
    counts.sort_unstable();
    let min_tc = counts.first().copied().unwrap_or(0);
    let max_tc = counts.last().copied().unwrap_or(0);
    let med_tc = counts[counts.len() / 2];
    println!("  token_count   : min={min_tc}  median={med_tc}  max={max_tc}");

    // Scenarios with very short generated text (potential quality concern).
    let short: Vec<&str> = manifest
        .scenarios
        .iter()
        .filter(|s| s.token_count < 4)
        .map(|s| s.id.as_str())
        .collect();
    if !short.is_empty() {
        println!(
            "  WARNING: {} scenarios have < 4 tokens: {:?}",
            short.len(),
            short
        );
    }

    println!();

    // ── Result ────────────────────────────────────────────────────────────────
    if errors.is_empty() {
        println!("PASS — all {total} scenarios complete and clean");
        println!("  zero signatures     : {zero_tok_total}");
        println!("  density-lo failures : {density_lo_total}");
        println!("  density-hi failures : {density_hi_total}");
        Ok(())
    } else {
        println!("FAIL — {} error(s):", errors.len());
        for e in &errors {
            println!("  • {e}");
        }
        std::process::exit(1);
    }
}
