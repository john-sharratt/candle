//! §83 — repo_map retrieval layer-group sweep.
//!
//! Takes ONE query turn's stored wide-Q signature as the probe and the repo_map
//! cluster turns as the gallery (one case per cluster, keyed by its path tag),
//! then scores the probe against the gallery under different **layer-group
//! selections** — the whole folded signature, the token/lower group alone, the
//! upper groups alone, and pairs — to find which layers actually carry the
//! path/structure retrieval signal.
//!
//! Motivation: the folded signature is `[46, 1, 1]` layer-groups — group 0 is
//! L0–45 (the noise-absorbing lower/"token" layers), groups 1 and 2 are L46 and
//! L47 (tuned for tool *identity*). Tool retrieval prioritises the upper groups.
//! Repo_map matching is closer to literal token identity (candle-core, backprop,
//! device), so the lower group may carry more of the signal here. This harness
//! measures it directly against the failing candle-core case.
//!
//! ```text
//! cargo run -p zend --example provenance_layers --release -- [workspace]
//!   QUERY=0x…    the probe turn's stream id (default: the candle-core question)
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use candle_conversation::persistence::streams::StreamDecl;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::provenance::{
    decode_wide_sigs, score_provenance_late_fusion, score_provenance_late_fusion_weighted, WideQSig,
};
use candle_conversation::substrate::Substrate;

/// KV heads per layer-group — the granularity `score_provenance_late_fusion`
/// slices the signature into (`n_groups = n_heads / HEADS_PER_GROUP`).
const HEADS_PER_GROUP: usize = 4;

/// Project a signature down to only the layer-`groups` listed, concatenated in
/// order. Produces a `WideQSig` the scorer reads as having exactly those groups,
/// so scoring it is scoring on that layer subset alone.
fn project_groups(sig: &WideQSig, groups: &[usize]) -> WideQSig {
    let n_heads = sig.n_heads as usize;
    if n_heads == 0 || sig.words.is_empty() {
        return sig.clone();
    }
    let wph = sig.words.len() / n_heads;
    let gw = HEADS_PER_GROUP * wph; // words per layer-group
    let mut words = Vec::with_capacity(groups.len() * gw);
    for &g in groups {
        let s = g * gw;
        let e = s + gw;
        if e <= sig.words.len() {
            words.extend_from_slice(&sig.words[s..e]);
        }
    }
    WideQSig {
        n_heads: (groups.len() * HEADS_PER_GROUP) as u16,
        words,
    }
}

/// Score the probe against the (already-flattened) gallery on `groups` only.
fn rank_on_groups(
    probe: &[WideQSig],
    gallery: &[WideQSig],
    gallery_case: &[u32],
    n_cases: usize,
    groups: &[usize],
) -> Vec<f32> {
    let p: Vec<WideQSig> = probe.iter().map(|s| project_groups(s, groups)).collect();
    let g: Vec<WideQSig> = gallery.iter().map(|s| project_groups(s, groups)).collect();
    let gref: Vec<&WideQSig> = g.iter().collect();
    score_provenance_late_fusion(&p, &gref, gallery_case, n_cases)
}

fn main() -> anyhow::Result<()> {
    let workspace = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let query_hex = std::env::var("QUERY").unwrap_or_else(|_| "0xb3cd93c08783d343".to_string());
    let query_sid = u64::from_str_radix(query_hex.trim_start_matches("0x"), 16)
        .map_err(|e| anyhow::anyhow!("bad QUERY stream id {query_hex}: {e}"))?;

    let mut substrate = Substrate::new();
    let _p = SubstratePersistence::open_in_with_substrate(&workspace, &mut substrate)
        .map_err(|e| anyhow::anyhow!("open substrate: {e}"))?;

    // ── Probe: the query turn's whole stored signature ──────────────────────────
    let probe: Vec<WideQSig> = substrate
        .all_streams()
        .into_iter()
        .find(|(sid, _)| sid.0 == query_sid)
        .and_then(|(_, e)| e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)))
        .ok_or_else(|| anyhow::anyhow!("no wide-Q sig on QUERY stream {query_hex}"))?;
    if probe.is_empty() {
        anyhow::bail!("QUERY stream {query_hex} has an empty signature");
    }

    // ── Gallery: repo_map clusters, one case per cluster, keyed by path ──────────
    // Multiple repo_map conversations can coexist (re-scans); use only the one
    // with the most cluster turns so a cluster appears exactly once.
    let mut per_timeline: HashMap<u64, Vec<(u64, String, Vec<WideQSig>)>> = HashMap::new();
    for (sid, e) in substrate.all_streams() {
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
            .push((sid.0, path, sig));
    }
    // REPO_TL forces a specific repo_map timeline; else pick deterministically —
    // most turns, tie-broken by newest (highest) timeline id — so repeated
    // rebuilds' equal-length timelines don't flip the gallery between runs.
    let repo_tl = std::env::var("REPO_TL")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    let (used_tl, mut clusters) = if let Some(tl) = repo_tl {
        let v = per_timeline
            .remove(&tl)
            .ok_or_else(|| anyhow::anyhow!("no repo_map timeline {tl}"))?;
        (tl, v)
    } else {
        per_timeline
            .into_iter()
            .max_by_key(|(tl, v)| (v.len(), *tl))
            .ok_or_else(|| anyhow::anyhow!("no repo_map clusters in substrate"))?
    };
    clusters.sort_by(|a, b| a.1.cmp(&b.1));

    // Flatten to (per-token gallery, case, case→name).
    let mut gallery: Vec<WideQSig> = Vec::new();
    let mut gallery_case: Vec<u32> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    for (_sid, path, sig) in &clusters {
        let case = names.len() as u32;
        names.push(path.clone());
        for tok in sig {
            gallery.push(tok.clone());
            gallery_case.push(case);
        }
    }

    let shape = &probe[0];
    let wph = if shape.n_heads == 0 {
        0
    } else {
        shape.words.len() / shape.n_heads as usize
    };
    let n_groups = shape.n_heads as usize / HEADS_PER_GROUP;
    println!("═══ §83 repo_map layer-group sweep ═══\n");
    println!("probe : {query_hex}  ({} tokens)", probe.len());
    println!(
        "gallery: {} clusters ({} tokens), signature {}h × {}wph → {} layer-groups\n",
        names.len(),
        gallery.len(),
        shape.n_heads,
        wph,
        n_groups
    );

    let candle = |n: &str| n.contains("candle-core");
    let report = |label: &str, scores: &[f32]| {
        let mut ranked: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        println!("── {label} ──");
        for (rank, (ci, sc)) in ranked.iter().take(6).enumerate() {
            let mark = if candle(&names[*ci]) {
                " ◀ candle-core"
            } else {
                ""
            };
            println!("  #{:<2} {:>9.2}  {}{}", rank + 1, sc, names[*ci], mark);
        }
        if let Some((r, (ci, sc))) = ranked
            .iter()
            .enumerate()
            .find(|(_, (ci, _))| candle(&names[*ci]))
        {
            println!(
                "  → best candle-core: #{} ({}) @ {:.2}\n",
                r + 1,
                names[*ci],
                sc
            );
        } else {
            println!("  → no candle-core cluster scored\n");
        }
    };

    // Pass 1 — isolate each group (0/1 masks) to see where the signal is.
    println!("## group isolation (which layers carry it)\n");
    report(
        "all groups (current)",
        &rank_on_groups(
            &probe,
            &gallery,
            &gallery_case,
            names.len(),
            &(0..n_groups).collect::<Vec<_>>(),
        ),
    );
    for g in 0..n_groups {
        let lbl = if g == 0 {
            "group 0 alone (L0-45 token/lower)".to_string()
        } else {
            format!("group {g} alone (upper)")
        };
        report(
            &lbl,
            &rank_on_groups(&probe, &gallery, &gallery_case, names.len(), &[g]),
        );
    }

    // Pass 2 — soft group WEIGHTS on the full signature (down-weight lower, keep
    // all layers). This is what we'd actually ship: a weight vector, not a skip.
    println!("## soft weights (keep every layer, re-weight the vote)\n");
    let weight_sets: &[(&str, &[f32])] = &[
        ("uniform [1,1,1]", &[1.0, 1.0, 1.0]),
        // Additive monotonic ramp bottom→top (user's "top adds most" idea).
        ("ramp [0.2, 0.6, 1]", &[0.2, 0.6, 1.0]),
        ("ramp [0.1, 0.5, 1]", &[0.1, 0.5, 1.0]),
        ("ramp [0, 0.3, 1]", &[0.0, 0.3, 1.0]),
        // L46-dominant (where the data says the signal actually is).
        ("[0.3, 1, 0.3]", &[0.3, 1.0, 0.3]),
        ("[0.2, 1, 0.2]", &[0.2, 1.0, 0.2]),
        ("[0.1, 1, 0.1]", &[0.1, 1.0, 0.1]),
        ("[0.1, 1, 0.3]", &[0.1, 1.0, 0.3]),
        ("[0, 1, 0] (group1 only)", &[0.0, 1.0, 0.0]),
    ];
    for (lbl, w) in weight_sets {
        if w.len() < n_groups {
            continue;
        }
        let scores = score_provenance_late_fusion_weighted(
            &probe,
            &gallery.iter().collect::<Vec<_>>(),
            &gallery_case,
            names.len(),
            w,
        );
        report(lbl, &scores);
    }

    Ok(())
}
