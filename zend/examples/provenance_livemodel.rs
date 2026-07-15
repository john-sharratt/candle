//! §86 — model the LIVE probe mismatch and the two candidate fixes.
//!
//! Live reprojection normalizes against a **trailing-window** probe (query head +
//! last `max_probe_tokens`), but the hit levels are learned at seal from the
//! **whole-turn** sig. That mismatch is what let an obscure cluster (whisper)
//! amplify past candle-core. This harness reproduces it and evaluates:
//!   fix (a) OBSERVE_TRAIL=1 — learn from a trailing-window probe too (match scales)
//!   fix (b) FLOOR_MIN / FLOOR_PCTL — raise the denominator floor to cap blow-ups
//! over MANY candle-core turns, reporting candle-core's rank and whether a spurious
//! cluster tops each — so we see which helps AND whether it regresses.
//!
//! ```text
//! cargo run -p zend --example provenance_livemodel --release -- [workspace]
//!   REPO_TL=<id>  FOCUS=candle-core  FOCUS_PROBES=0x..,0x..
//!   OBSERVE_TRAIL=0|1   ALPHA_UP ALPHA_DN HIT_PRIOR FLOOR_MIN FLOOR_PCTL
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use candle_conversation::normalization::{ChildKey, NormConfig, NormalizationCache, ScopeKey};
use candle_conversation::persistence::streams::StreamDecl;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::provenance::{decode_wide_sigs, score_provenance_late_fusion, WideQSig};
use candle_conversation::substrate::Substrate;

const QUERY_HEAD: usize = 64; // QUERY_HEAD_CHUNKS(2) * 32
const MAX_PROBE: usize = 256; // reproject_max_probe_tokens

fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

/// The live reprojection probe: query head + trailing window of a turn's sig.
fn trailing(sig: &[WideQSig]) -> Vec<WideQSig> {
    let len = sig.len();
    let head = QUERY_HEAD.min(len);
    let tail_start = len.saturating_sub(MAX_PROBE).max(head);
    let mut out: Vec<WideQSig> = sig[..head].to_vec();
    out.extend_from_slice(&sig[tail_start..]);
    out
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
        .filter_map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
        .collect();
    let observe_trail = std::env::var("OBSERVE_TRAIL").ok().as_deref() == Some("1");
    let d = NormConfig::default();
    let cfg = NormConfig {
        alpha_up: env_f32("ALPHA_UP", d.alpha_up),
        alpha_dn: env_f32("ALPHA_DN", d.alpha_dn),
        hit_prior: env_f32("HIT_PRIOR", d.hit_prior),
        floor_min: env_f32("FLOOR_MIN", d.floor_min),
        floor_pctl: env_f32("FLOOR_PCTL", d.floor_pctl),
        scale: d.scale,
    };

    let mut substrate = Substrate::new();
    let _p = SubstratePersistence::open_in_with_substrate(&workspace, &mut substrate)
        .map_err(|e| anyhow::anyhow!("open: {e}"))?;

    // Gallery: repo_map clusters on one timeline.
    let mut per_tl: HashMap<u64, Vec<(String, Vec<WideQSig>)>> = HashMap::new();
    for (_s, e) in substrate.all_streams() {
        let Some(StreamDecl::Turn(dd)) = &e.decl else {
            continue;
        };
        if !dd.tags.iter().any(|t| t == "repo_map") {
            continue;
        }
        let Some(sig) = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)) else {
            continue;
        };
        if sig.is_empty() {
            continue;
        }
        let path = dd
            .tags
            .iter()
            .find(|t| *t != "repo_map")
            .cloned()
            .unwrap_or_else(|| ".".into());
        per_tl.entry(dd.timeline_id).or_default().push((path, sig));
    }
    let repo_tl = std::env::var("REPO_TL")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    let clusters = if let Some(tl) = repo_tl {
        per_tl
            .remove(&tl)
            .ok_or_else(|| anyhow::anyhow!("no timeline {tl}"))?
    } else {
        per_tl
            .into_iter()
            .max_by_key(|(tl, v)| (v.len(), *tl))
            .map(|(_, v)| v)
            .ok_or_else(|| anyhow::anyhow!("no clusters"))?
    };
    let names: Vec<String> = clusters.iter().map(|(p, _)| p.clone()).collect();
    let n = names.len();
    let mut gallery: Vec<WideQSig> = Vec::new();
    let mut gcase: Vec<u32> = Vec::new();
    for (case, (_p, sig)) in clusters.iter().enumerate() {
        for t in sig {
            gallery.push(t.clone());
            gcase.push(case as u32);
        }
    }
    let gref: Vec<&WideQSig> = gallery.iter().collect();
    let child_keys: Vec<ChildKey> = names.iter().map(ChildKey::named).collect();
    let score = |probe: &[WideQSig]| score_provenance_late_fusion(probe, &gref, &gcase, n);

    // Dialogue turns (empty tags) in deterministic order.
    let mut dialogue: Vec<(u64, u64, u32, Vec<WideQSig>)> = Vec::new();
    for (sid, e) in substrate.all_streams() {
        let Some(StreamDecl::Turn(dd)) = &e.decl else {
            continue;
        };
        if !dd.tags.is_empty() {
            continue;
        }
        let Some(sig) = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)) else {
            continue;
        };
        if !sig.is_empty() {
            dialogue.push((sid.0, dd.timeline_id, dd.turn_index, sig));
        }
    }
    dialogue.sort_by_key(|(_, tl, idx, _)| (*tl, *idx));

    // WARM the hit levels from non-focus dialogue turns (a focus turn isn't sealed
    // when its own reprojections fire live, so it shouldn't warm its own test).
    let scope = ScopeKey::turn_group(1, 1);
    let mut cache = NormalizationCache::new(cfg);
    for (sid, _, _, sig) in &dialogue {
        if focus_probes.contains(sid) {
            continue;
        }
        let probe = if observe_trail {
            trailing(sig)
        } else {
            sig.clone()
        };
        let raw = score(&probe);
        let pairs: Vec<(ChildKey, f32)> =
            (0..n).map(|ci| (child_keys[ci].clone(), raw[ci])).collect();
        cache.observe(&scope, &pairs);
    }

    println!(
        "═══ §86 live-probe model | OBSERVE_TRAIL={} | floor_min {} floor_pctl {} ═══\n",
        observe_trail, cfg.floor_min, cfg.floor_pctl
    );

    let is_focus = |nm: &str| nm.contains(&focus);
    let mut top3_hits = 0;
    let mut spurious_top1 = 0;
    for (sid, _, _, sig) in &dialogue {
        if !focus_probes.contains(sid) {
            continue;
        }
        // NORMALIZE with the live trailing-window probe.
        let raw = score(&trailing(sig));
        let pairs: Vec<(ChildKey, f32)> =
            (0..n).map(|ci| (child_keys[ci].clone(), raw[ci])).collect();
        let normed = cache.normalize(&scope, &pairs);
        let mut ranked: Vec<(usize, f32)> = normed.iter().map(|(_, v)| *v).enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let cc_rank = ranked
            .iter()
            .position(|(ci, _)| is_focus(&names[*ci]))
            .map(|r| r + 1);
        let cc_sc = ranked
            .iter()
            .find(|(ci, _)| is_focus(&names[*ci]))
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        let top1 = &names[ranked[0].0];
        if matches!(cc_rank, Some(r) if r <= 3) {
            top3_hits += 1;
        }
        if !is_focus(top1) {
            spurious_top1 += 1;
        }
        println!(
            "  {sid:#018x}  candle-core #{:<3} ({cc_sc:>6.0})   top1: {} ({:.0})",
            cc_rank.map(|r| r as i32).unwrap_or(-1),
            short(top1),
            ranked[0].1,
        );
    }
    println!(
        "\n  SUMMARY: candle-core top-3 on {}/{} turns | spurious top1 on {}/{}",
        top3_hits,
        focus_probes.len(),
        spurious_top1,
        focus_probes.len(),
    );
    Ok(())
}

fn short(p: &str) -> String {
    let t = p.trim_end_matches('/');
    let s: Vec<&str> = t.rsplit('/').take(2).collect();
    if s.is_empty() {
        ".".into()
    } else {
        s.iter().rev().cloned().collect::<Vec<_>>().join("/")
    }
}
