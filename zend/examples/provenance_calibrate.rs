//! §85 — the real `normalization` module, driven over the whole substrate.
//!
//! Replays every dialogue turn (empty-tag conversation turns) as a probe against
//! the repo_map turn-group gallery, in substrate order. For each probe it calls
//! the production `NormalizationCache::normalize` (read) then `observe` (write,
//! once per turn — the seal cadence), so this is an end-to-end check that the
//! module reproduces the calibrated accuracy (`docs/provenance_score_normalization.md`
//! §7). Reports the normalized-score distribution and, for FOCUS probes (the
//! candle-core questions), whether the right cluster wins after normalization.
//!
//! ```text
//! cargo run -p zend --example provenance_calibrate --release -- [workspace]
//!   REPO_TL=<id>            pin the repo_map timeline (else newest by turn count)
//!   FOCUS=candle-core       substring of the "correct" cluster for focus probes
//!   FOCUS_PROBES=0x..,0x..   dialogue turns whose correct answer is FOCUS
//!   ALPHA_UP / ALPHA_DN / HIT_PRIOR / FLOOR_MIN / FLOOR_PCTL  NormConfig overrides
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use candle_conversation::normalization::{ChildKey, NormConfig, NormalizationCache, ScopeKey};
use candle_conversation::persistence::streams::StreamDecl;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::provenance::{decode_wide_sigs, score_provenance_late_fusion, WideQSig};
use candle_conversation::substrate::Substrate;

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn pctl(vals: &[f32], p: f32) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut v = vals.to_vec();
    v.sort_by(f32::total_cmp);
    v[((v.len() as f32 * p) as usize).min(v.len() - 1)]
}

fn main() -> anyhow::Result<()> {
    let workspace = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let focus = std::env::var("FOCUS").unwrap_or_else(|_| "candle-core".to_string());
    let focus_probes: Vec<u64> = std::env::var("FOCUS_PROBES")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .filter_map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
        .collect();
    let defaults = NormConfig::default();
    let cfg = NormConfig {
        alpha_up: env_f32("ALPHA_UP", defaults.alpha_up),
        alpha_dn: env_f32("ALPHA_DN", defaults.alpha_dn),
        hit_prior: env_f32("HIT_PRIOR", defaults.hit_prior),
        floor_min: env_f32("FLOOR_MIN", defaults.floor_min),
        floor_pctl: env_f32("FLOOR_PCTL", defaults.floor_pctl),
        scale: defaults.scale,
    };

    let mut substrate = Substrate::new();
    let _p = SubstratePersistence::open_in_with_substrate(&workspace, &mut substrate)
        .map_err(|e| anyhow::anyhow!("open substrate: {e}"))?;

    // ── Gallery: repo_map clusters on one timeline ──────────────────────────────
    let mut per_tl: HashMap<u64, Vec<(String, Vec<WideQSig>)>> = HashMap::new();
    for (_sid, e) in substrate.all_streams() {
        let Some(StreamDecl::Turn(d)) = &e.decl else {
            continue;
        };
        if !d.tags.iter().any(|t| t == "repo_map") {
            continue;
        }
        let Some(sig) = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)) else {
            continue;
        };
        if sig.is_empty() {
            continue;
        }
        let path = d
            .tags
            .iter()
            .find(|t| *t != "repo_map")
            .cloned()
            .unwrap_or_else(|| ".".into());
        per_tl.entry(d.timeline_id).or_default().push((path, sig));
    }
    let repo_tl = std::env::var("REPO_TL")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    let clusters = if let Some(tl) = repo_tl {
        per_tl
            .remove(&tl)
            .ok_or_else(|| anyhow::anyhow!("no repo_map timeline {tl}"))?
    } else {
        per_tl
            .into_iter()
            .max_by_key(|(tl, v)| (v.len(), *tl))
            .map(|(_, v)| v)
            .ok_or_else(|| anyhow::anyhow!("no repo_map clusters"))?
    };
    let names: Vec<String> = clusters.iter().map(|(p, _)| p.clone()).collect();
    let n = names.len();
    let mut gallery: Vec<WideQSig> = Vec::new();
    let mut gallery_case: Vec<u32> = Vec::new();
    for (case, (_p, sig)) in clusters.iter().enumerate() {
        for tok in sig {
            gallery.push(tok.clone());
            gallery_case.push(case as u32);
        }
    }
    let gref: Vec<&WideQSig> = gallery.iter().collect();

    // ── Probes: dialogue turns (empty tags) with a signature, in substrate order ─
    let mut probes: Vec<(u64, Vec<WideQSig>)> = Vec::new();
    for (sid, e) in substrate.all_streams() {
        let Some(StreamDecl::Turn(d)) = &e.decl else {
            continue;
        };
        if !d.tags.is_empty() {
            continue;
        }
        let Some(sig) = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)) else {
            continue;
        };
        if sig.is_empty() {
            continue;
        }
        probes.push((sid.0, sig));
    }

    println!(
        "═══ §85 normalization module over {} dialogue probes ═══",
        probes.len()
    );
    println!("scope: repo_map turn-group, {n} clusters | cfg {cfg:?}\n");

    // The production module — the thing under test.
    let scope = ScopeKey::turn_group(1, 1);
    let mut cache = NormalizationCache::new(cfg);
    let child_keys: Vec<ChildKey> = names.iter().map(ChildKey::named).collect();

    let mut raw_mean = vec![0.0f32; n]; // diagnostics only (loudest clusters)
    let mut top1: Vec<f32> = Vec::new();
    let mut top2: Vec<f32> = Vec::new();
    let mut all: Vec<f32> = Vec::new();
    let is_focus = |nm: &str| nm.contains(&focus);
    let mut focus_rows: Vec<String> = Vec::new();

    for (turn_i, (pid, sig)) in probes.iter().enumerate() {
        let raw = score_provenance_late_fusion(sig, &gref, &gallery_case, n);
        let raw_pairs: Vec<(ChildKey, f32)> =
            (0..n).map(|ci| (child_keys[ci].clone(), raw[ci])).collect();

        // READ: normalize against hit levels as they stand before this turn.
        let norm = cache.normalize(&scope, &raw_pairs);
        let norm_by_i: Vec<f32> = norm.iter().map(|(_, v)| *v).collect(); // preserves order

        let mut ranked: Vec<(usize, f32)> = norm_by_i.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        top1.push(ranked[0].1);
        if ranked.len() > 1 {
            top2.push(ranked[1].1);
        }
        all.extend(&norm_by_i);

        if focus_probes.contains(pid) {
            let mut raw_ranked: Vec<(usize, f32)> = raw.iter().copied().enumerate().collect();
            raw_ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            let raw_rank = raw_ranked.iter().position(|(ci, _)| is_focus(&names[*ci]));
            let norm_rank = ranked.iter().position(|(ci, _)| is_focus(&names[*ci]));
            let norm_sc = ranked
                .iter()
                .find(|(ci, _)| is_focus(&names[*ci]))
                .map(|(_, s)| *s)
                .unwrap_or(0.0);
            focus_rows.push(format!(
                "  #{turn_i:<3} {pid:#018x}  raw #{:<3} → norm #{:<3} ({norm_sc:>6.0})   norm top1: {} ({:.0})",
                raw_rank.map(|r| r as i32 + 1).unwrap_or(-1),
                norm_rank.map(|r| r as i32 + 1).unwrap_or(-1),
                short(&names[ranked[0].0]),
                ranked[0].1,
            ));
        }

        // WRITE: fold this turn into the hit levels (seal cadence, once per turn).
        cache.observe(&scope, &raw_pairs);
        for ci in 0..n {
            raw_mean[ci] += (raw[ci] - raw_mean[ci]) / (turn_i + 1) as f32;
        }
    }

    println!("## normalized-score distribution (0-1000 scale)\n");
    let ps = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];
    println!(
        "  {:<12} {}",
        "percentile:",
        ps.iter()
            .map(|p| format!("{:>7.0}%", p * 100.0))
            .collect::<String>()
    );
    for (label, v) in [("top-1", &top1), ("top-2", &top2), ("all (noise)", &all)] {
        let row: String = ps.iter().map(|p| format!("{:>8.0}", pctl(v, *p))).collect();
        println!("  {label:<12}{row}");
    }
    println!();

    let mut by_mean: Vec<usize> = (0..n).collect();
    by_mean.sort_by(|a, b| raw_mean[*b].total_cmp(&raw_mean[*a]));
    println!("## loudest clusters by raw mean (the discounted ones)\n");
    for &ci in by_mean.iter().take(5) {
        println!("  raw_mean {:>7.1}   {}", raw_mean[ci], names[ci]);
    }
    println!();

    if !focus_rows.is_empty() {
        println!("## FOCUS probes ({focus}) — raw rank → normalized rank\n");
        for r in &focus_rows {
            println!("{r}");
        }
    }
    Ok(())
}

fn short(path: &str) -> String {
    let t = path.trim_end_matches('/');
    let segs: Vec<&str> = t.rsplit('/').take(2).collect();
    if segs.is_empty() {
        ".".into()
    } else {
        segs.iter().rev().cloned().collect::<Vec<_>>().join("/")
    }
}
