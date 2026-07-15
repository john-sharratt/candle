//! §85 — per-scope running-mean normalization, modelled over the whole substrate.
//!
//! Replays every dialogue turn (empty-tag conversation turns) as a probe against
//! a scope's gallery, in substrate order, maintaining a per-member RUNNING MEAN of
//! raw scores (the carry-forward promiscuity baseline). Each probe is normalized
//! `1000 × raw / max(running_mean, prior)` using the means as they stood BEFORE
//! that probe (causal), then the probe is folded into the means. Reports:
//!   * the distribution of normalized scores (top-1 / top-3 / selected) — the data
//!     to calibrate min_score / evict on the 0-1000 scale;
//!   * for FOCUS probes (e.g. the candle-core questions), the raw vs normalized
//!     rank + score of the focus cluster, to see whether the right target wins.
//!
//! ```text
//! cargo run -p zend --example provenance_calibrate --release -- [workspace]
//!   REPO_TL=<id>        pin the repo_map timeline (else newest by turn count)
//!   FOCUS=candle-core   substring of the "correct" cluster for focus probes
//!   FOCUS_PROBES=0x..,0x..   dialogue turns whose correct answer is FOCUS
//!   WARMUP=3            probes a member needs before its own mean is trusted
//!   FLOOR_PCTL=0.92     percentile (over member means) used as the denom floor
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use candle_conversation::persistence::streams::StreamDecl;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::provenance::{decode_wide_sigs, score_provenance_late_fusion, WideQSig};
use candle_conversation::substrate::Substrate;

/// A member's carry-forward baseline: running mean of raw scores + count.
#[derive(Clone, Default)]
struct Running {
    mean: f32,
    count: u32,
}
impl Running {
    fn update(&mut self, x: f32) {
        self.count += 1;
        self.mean += (x - self.mean) / self.count as f32;
    }
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
    let warmup: u32 = std::env::var("WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let floor_pctl: f32 = std::env::var("FLOOR_PCTL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.92);
    // Cold-start prior: the denominator a cluster gets before it has warmed up, and
    // a hard minimum for the floor — kills the "divide by ~1" early explosions.
    let prior: f32 = std::env::var("PRIOR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50.0);
    // HIT-LEVEL normalizer: divide by the score a cluster reaches when it IS the
    // answer, so a full hit ≈ 1000 and a decode lock-on rides above it. Tracked as
    // an asymmetric EWMA — rises fast toward a strong match, decays slowly — so it
    // settles at the cluster's characteristic hit magnitude, not its mean.
    let a_up: f32 = std::env::var("ALPHA_UP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.30);
    let a_dn: f32 = std::env::var("ALPHA_DN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.02);
    let hit_prior: f32 = std::env::var("HIT_PRIOR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(400.0);

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
            continue; // gallery turn, not dialogue
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
        "═══ §85 running-mean normalization over {} dialogue probes ═══",
        probes.len()
    );
    println!("scope: repo_map group, {n} clusters | warmup {warmup} | floor pctl {floor_pctl}\n");

    let mut run: Vec<Running> = vec![Running::default(); n];
    let mut hit: Vec<f32> = vec![hit_prior; n]; // per-cluster hit level (EWMA)
                                                // Distributions to calibrate thresholds.
    let mut top1_norm: Vec<f32> = Vec::new();
    let mut top2_norm: Vec<f32> = Vec::new();
    let mut all_norm: Vec<f32> = Vec::new();
    // Focus tracking.
    let is_focus = |nm: &str| nm.contains(&focus);
    let mut focus_rows: Vec<String> = Vec::new();

    for (pid, sig) in &probes {
        let raw = score_provenance_late_fusion(sig, &gref, &gallery_case, n);
        // Floor = percentile over the warmed members' current means; prior for
        // cold members = the same floor.
        // Denominator = the cluster's hit level, floored, so a full hit ≈ 1000.
        let floor = pctl(
            &hit.iter().copied().filter(|h| *h > 0.0).collect::<Vec<_>>(),
            0.10,
        )
        .max(prior);
        let norm: Vec<(usize, f32)> = (0..n)
            .map(|ci| (ci, 1000.0 * raw[ci] / hit[ci].max(floor)))
            .collect();
        let mut ns = norm.clone();
        ns.sort_by(|a, b| b.1.total_cmp(&a.1));

        top1_norm.push(ns[0].1);
        if ns.len() > 1 {
            top2_norm.push(ns[1].1);
        }
        for (_, v) in &norm {
            all_norm.push(*v);
        }

        if focus_probes.contains(pid) {
            let mut rs: Vec<(usize, f32)> = (0..n).map(|ci| (ci, raw[ci])).collect();
            rs.sort_by(|a, b| b.1.total_cmp(&a.1));
            let raw_rank = rs
                .iter()
                .position(|(ci, _)| is_focus(&names[*ci]))
                .map(|r| r + 1);
            let norm_rank = ns
                .iter()
                .position(|(ci, _)| is_focus(&names[*ci]))
                .map(|r| r + 1);
            let norm_sc = ns
                .iter()
                .find(|(ci, _)| is_focus(&names[*ci]))
                .map(|(_, s)| *s)
                .unwrap_or(0.0);
            let top = &names[ns[0].0];
            focus_rows.push(format!(
                "  {pid:#018x}  raw #{:<3} → norm #{:<3} ({norm_sc:>6.0})   norm top1: {} ({:.0})",
                raw_rank.map(|r| r as i32).unwrap_or(-1),
                norm_rank.map(|r| r as i32).unwrap_or(-1),
                short(top),
                ns[0].1,
            ));
        }

        // Fold this probe into the running stats (causal: after normalizing).
        for ci in 0..n {
            run[ci].update(raw[ci]);
            // Asymmetric EWMA hit level: rise fast toward a strong match, decay slow.
            let a = if raw[ci] > hit[ci] { a_up } else { a_dn };
            hit[ci] += a * (raw[ci] - hit[ci]);
        }
    }

    // ── Calibration distributions ───────────────────────────────────────────────
    println!("## normalized-score distribution (calibrate min_score / evict here)\n");
    let ps = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];
    println!(
        "  {:<14} {}",
        "percentile:",
        ps.iter()
            .map(|p| format!("{:>7.0}%", p * 100.0))
            .collect::<String>()
    );
    let show = |label: &str, v: &[f32]| {
        let row: String = ps.iter().map(|p| format!("{:>8.0}", pctl(v, *p))).collect();
        println!("  {label:<14}{row}");
    };
    show("top-1 norm", &top1_norm);
    show("top-2 norm", &top2_norm);
    show("all norm", &all_norm);
    println!();

    // Per-cluster final means — the loudest (promiscuous) clusters.
    let mut by_mean: Vec<usize> = (0..n).collect();
    by_mean.sort_by(|a, b| run[*b].mean.total_cmp(&run[*a].mean));
    println!("## loudest clusters by final running mean (the discounted ones)\n");
    for &ci in by_mean.iter().take(6) {
        println!(
            "  mean {:>8.1}  (n={})   {}",
            run[ci].mean, run[ci].count, names[ci]
        );
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
