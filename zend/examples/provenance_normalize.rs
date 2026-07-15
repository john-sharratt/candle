//! §84 — repo_map score NORMALIZATION probe.
//!
//! Hypothesis: cluster retrieval fails not because the signal is absent but
//! because clusters live on different ABSOLUTE score scales. The late-fusion
//! tally sums `z × margin` votes per cluster, so a "loud" cluster (root, docs —
//! many tokens, agrees with many query tokens) accumulates a high absolute score
//! for EVERY probe, drowning a specific dir that is genuinely the best *relative*
//! match. If true, then normalising each cluster against its own cross-probe
//! baseline should make the right dir peak on its own query.
//!
//! This harness scores a set of probes (different query types) against the
//! repo_map gallery, builds the cluster×probe matrix, then for each probe ranks
//! clusters two ways:
//!   RAW        — the production score (absolute tally).
//!   NORMALISED — per-cluster z-score across the probe set: (raw - mean) / std,
//!                i.e. "how unusually strong is this probe for this cluster,
//!                relative to how this cluster scores in general".
//!
//! ```text
//! cargo run -p zend --example provenance_normalize --release -- [workspace]
//!   REPO_TL=<id>   pin the repo_map timeline (else newest by turn count)
//!   PROBES=label:0xsid,label:0xsid,...   the probe set (required)
//!   FOCUS=candle-core   substring marking the "correct" cluster to track
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use candle_conversation::persistence::streams::StreamDecl;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::provenance::{decode_wide_sigs, score_provenance_late_fusion, WideQSig};
use candle_conversation::substrate::Substrate;

fn main() -> anyhow::Result<()> {
    let workspace = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let focus = std::env::var("FOCUS").unwrap_or_else(|_| "candle-core".to_string());
    let probes_spec =
        std::env::var("PROBES").map_err(|_| anyhow::anyhow!("PROBES=label:0xsid,... required"))?;

    let mut probes: Vec<(String, u64)> = Vec::new();
    for entry in probes_spec.split(',') {
        let (label, sid) = entry
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("bad PROBES entry {entry}, want label:0xsid"))?;
        let sid = u64::from_str_radix(sid.trim().trim_start_matches("0x"), 16)
            .map_err(|e| anyhow::anyhow!("bad sid {sid}: {e}"))?;
        probes.push((label.trim().to_string(), sid));
    }

    let mut substrate = Substrate::new();
    let _p = SubstratePersistence::open_in_with_substrate(&workspace, &mut substrate)
        .map_err(|e| anyhow::anyhow!("open substrate: {e}"))?;

    // ── Gallery: repo_map clusters on one timeline, one case per cluster ─────────
    let mut per_timeline: HashMap<u64, Vec<(String, Vec<WideQSig>)>> = HashMap::new();
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
            .unwrap_or_else(|| ".".to_string());
        per_timeline
            .entry(d.timeline_id)
            .or_default()
            .push((path, sig));
    }
    let repo_tl = std::env::var("REPO_TL")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    let mut clusters = if let Some(tl) = repo_tl {
        per_timeline
            .remove(&tl)
            .ok_or_else(|| anyhow::anyhow!("no repo_map timeline {tl}"))?
    } else {
        per_timeline
            .into_iter()
            .max_by_key(|(tl, v)| (v.len(), *tl))
            .map(|(_, v)| v)
            .ok_or_else(|| anyhow::anyhow!("no repo_map clusters"))?
    };
    clusters.sort_by(|a, b| a.0.cmp(&b.0));

    let names: Vec<String> = clusters.iter().map(|(p, _)| p.clone()).collect();
    let mut gallery: Vec<WideQSig> = Vec::new();
    let mut gallery_case: Vec<u32> = Vec::new();
    for (case, (_p, sig)) in clusters.iter().enumerate() {
        for tok in sig {
            gallery.push(tok.clone());
            gallery_case.push(case as u32);
        }
    }
    let gref: Vec<&WideQSig> = gallery.iter().collect();

    // ── Score matrix: rows = clusters, cols = probes ────────────────────────────
    let n = names.len();
    let mut matrix: Vec<Vec<f32>> = vec![vec![0.0; probes.len()]; n];
    for (pi, (label, sid)) in probes.iter().enumerate() {
        let probe: Vec<WideQSig> = substrate
            .all_streams()
            .into_iter()
            .find(|(s, _)| s.0 == *sid)
            .and_then(|(_, e)| e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)))
            .ok_or_else(|| anyhow::anyhow!("probe {label} ({sid:#x}) has no sig"))?;
        let scores = score_provenance_late_fusion(&probe, &gref, &gallery_case, n);
        for (ci, s) in scores.iter().enumerate() {
            matrix[ci][pi] = *s;
        }
    }

    // Per-cluster PROMISCUITY via LEAVE-ONE-OUT (the seal-time stand-in for the
    // query-corpus mean): probe = cluster ci scored against the gallery with ci's
    // own tokens REMOVED, so its votes flow to whatever it is otherwise closest to.
    // Accumulate into each cluster's column — how often it is the runner-up match
    // for other clusters = how generically it is matched. A generic cluster (root)
    // is lit up by many → high; a distinct one (candle-core) → low. (§84 verdict:
    // this proxy still ≈ raw — real promiscuity needs the query corpus, not the
    // gallery — so the calibrator §85 uses the running query mean instead.)
    let mut promisc = vec![0.0f32; n];
    for (ci, (_p, sig)) in clusters.iter().enumerate() {
        // Gallery minus ci, cases renumbered to a dense 0..n-1 skipping ci.
        let mut loo_gallery: Vec<&WideQSig> = Vec::new();
        let mut loo_case: Vec<u32> = Vec::new();
        let mut back: Vec<usize> = Vec::new(); // dense case -> original cluster idx
        for (orig, (_, s)) in clusters.iter().enumerate() {
            if orig == ci {
                continue;
            }
            let dense = back.len() as u32;
            back.push(orig);
            for tok in s {
                loo_gallery.push(tok);
                loo_case.push(dense);
            }
        }
        let sc = score_provenance_late_fusion(sig, &loo_gallery, &loo_case, back.len());
        for (dense, s) in sc.iter().enumerate() {
            promisc[back[dense]] += *s;
        }
    }
    for p in promisc.iter_mut() {
        *p /= (n - 1) as f32;
    }
    // Reference-set mean of the cluster's LIVE scores across the probe set — the
    // "ground-truth" promiscuity (needs a probe corpus; promisc[] approximates it
    // self-referentially). Compared here to see how close the seal-time proxy is.
    let np = probes.len() as f32;
    let probe_mean: Vec<f32> = (0..n)
        .map(|ci| matrix[ci].iter().sum::<f32>() / np)
        .collect();

    // Floor denominators so a quiet cluster can't blow a partial match up. The
    // floor PERCENTILE is the key knob — too low and tiny-mean clusters amplify.
    let pctl = std::env::var("FLOOR_PCTL")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .unwrap_or(0.75);
    let floor = |v: &[f32]| -> f32 {
        let mut s: Vec<f32> = v.iter().copied().filter(|x| *x > 0.0).collect();
        s.sort_by(f32::total_cmp);
        if s.is_empty() {
            1.0
        } else {
            s[((s.len() as f32 * pctl) as usize).min(s.len() - 1)].max(1.0)
        }
    };
    let pf = floor(&promisc);
    let mf = floor(&probe_mean);

    println!("═══ §84 repo_map normalisation probe ═══\n");
    println!(
        "gallery: {n} clusters, {} probes  (promisc-floor {pf:.1}, mean-floor {mf:.1})\n",
        probes.len()
    );

    let mut by_p: Vec<usize> = (0..n).collect();
    by_p.sort_by(|a, b| promisc[*b].total_cmp(&promisc[*a]));
    println!("## per-cluster promiscuity (cross-mean) — top 8 loudest\n");
    for &ci in by_p.iter().take(8) {
        println!(
            "  promisc {:>8.1}  (probe_mean {:>8.1})   {}",
            promisc[ci], probe_mean[ci], names[ci]
        );
    }
    println!();

    // For each probe: RAW vs PROMISC-normalised vs PROBE-MEAN-normalised.
    for (pi, (label, _)) in probes.iter().enumerate() {
        let raw: Vec<(usize, f32)> = (0..n).map(|ci| (ci, matrix[ci][pi])).collect();
        let norm: Vec<(usize, f32)> = (0..n)
            .map(|ci| (ci, 1000.0 * matrix[ci][pi] / promisc[ci].max(pf)))
            .collect();
        let norm2: Vec<(usize, f32)> = (0..n)
            .map(|ci| (ci, 1000.0 * matrix[ci][pi] / probe_mean[ci].max(mf)))
            .collect();
        let rank_of = |v: &[(usize, f32)], pred: &dyn Fn(&str) -> bool| -> Option<(usize, f32)> {
            let mut s = v.to_vec();
            s.sort_by(|a, b| b.1.total_cmp(&a.1));
            s.iter()
                .enumerate()
                .find(|(_, (ci, _))| pred(&names[*ci]))
                .map(|(r, (_, sc))| (r + 1, *sc))
        };
        let is_focus = |nm: &str| nm.contains(&focus);
        let mut raw_s = raw.clone();
        raw_s.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut norm_s = norm.clone();
        norm_s.sort_by(|a, b| b.1.total_cmp(&a.1));

        println!("── probe: {label} ──");
        print!("  RAW  top5: ");
        for (ci, sc) in raw_s.iter().take(5) {
            print!("{}({:.0}) ", short(&names[*ci]), sc);
        }
        if let Some((r, sc)) = rank_of(&raw, &is_focus) {
            println!("  | {focus}: #{r} ({sc:.0})");
        } else {
            println!();
        }
        print!("  PROMISC-norm top5: ");
        for (ci, sc) in norm_s.iter().take(5) {
            print!("{}({:.0}) ", short(&names[*ci]), sc);
        }
        if let Some((r, sc)) = rank_of(&norm, &is_focus) {
            println!("  | {focus}: #{r} ({sc:.0})");
        } else {
            println!();
        }
        let mut norm2_s = norm2.clone();
        norm2_s.sort_by(|a, b| b.1.total_cmp(&a.1));
        print!("  MEAN-norm    top5: ");
        for (ci, sc) in norm2_s.iter().take(5) {
            print!("{}({:.0}) ", short(&names[*ci]), sc);
        }
        if let Some((r, sc)) = rank_of(&norm2, &is_focus) {
            println!("  | {focus}: #{r} ({sc:.0})");
        } else {
            println!();
        }
        println!();
    }

    Ok(())
}

/// Last two path segments, for compact printing.
fn short(path: &str) -> String {
    let t = path.trim_end_matches('/');
    let segs: Vec<&str> = t.rsplit('/').take(2).collect();
    if segs.is_empty() {
        ".".to_string()
    } else {
        segs.iter().rev().cloned().collect::<Vec<_>>().join("/")
    }
}
