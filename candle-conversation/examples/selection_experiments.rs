//! `selection_experiments` — offline experiment battery over the
//! selection-replay fixture (`tests/selection_replay_data/export/`), probing how
//! the shipped provenance scan behaves on the captured probes and galleries and
//! testing the load-bearing assumptions of
//! `docs/provenance_adaptive_projection.md` before the design is refined.
//!
//! Every experiment is pure CPU scoring over the checked-in sig blobs — no
//! model, no substrate load — so runs are deterministic and cheap to iterate.
//!
//! ```bash
//! cargo run -p candle-conversation --release --example selection_experiments
//! ```
//!
//! Experiments:
//! - **E1** anatomy of the ModelBuilder miss: per-exchange builder.rs scores vs
//!   the observed junk (Concept C/D viability — is there a seed hit to amplify?)
//! - **E2** promiscuity normalization: per-slot level = mean score over all 30
//!   captured dialogue probes, re-rank tour + ModelBuilder by `score/level`
//!   (Concept A's premise on real data)
//! - **E3** attention-mass formulas: raw sum vs gated / concentration /
//!   normalized variants on the recall-vs-code contrast (Concept B's input)
//! - **E4** fold-group decomposition: which layer-groups carry code / cluster
//!   identity (the margin-id analog, §83 group weights)
//! - **E5** probe-window sweep: ranking sensitivity to probe length
//! - **E6** probe evolution across the recorded reprojection cadence: question
//!   probe vs reasoning probes (Concept E's premise, early lock-on value)
//! - **E7** self-match sanity + structure-gallery anatomy: gallery-side health
//!   and where the workspace-root cluster ranks for the tour probe

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use candle_conversation::projection::{decode_events, ProjectionEvent};
use candle_conversation::provenance::{decode_wide_sigs, score_slots_weighted, WideQSig};
use serde_json::Value;

/// Production probe window (`reproject_max_probe_tokens`).
const MAX_PROBE_TOKENS: usize = 256;
/// The tour conversation's dialogue timeline.
const TOUR_TL: u64 = 10454652321042114835;
/// Dialogue turn indices of the probes under study.
const T_TOUR: u64 = 0;
const T_MODELBUILDER: u64 = 5;
const T_RECALL: u64 = 6;

struct TurnRef {
    timeline: u64,
    index: u64,
    tags: Vec<String>,
}

impl TurnRef {
    fn path(&self) -> &str {
        self.tags.get(1).map(String::as_str).unwrap_or("?")
    }
}

struct Fixture {
    sigs: HashMap<(u64, u64), Vec<WideQSig>>,
    events: BTreeMap<(u64, u64), Vec<ProjectionEvent>>,
    dialogue: Vec<TurnRef>,
    candidates: Vec<TurnRef>,
    targets: BTreeMap<String, Vec<TurnRef>>,
}

fn export_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/selection_replay_data/export")
}

fn parse_refs(v: &Value) -> Vec<TurnRef> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter(|t| t["has_sig"].as_bool() == Some(true))
                .filter_map(|t| {
                    Some(TurnRef {
                        timeline: t["timeline"].as_str()?.parse().ok()?,
                        index: t["index"].as_u64()?,
                        tags: t["tags"]
                            .as_array()
                            .map(|a| {
                                a.iter()
                                    .filter_map(|v| v.as_str().map(str::to_string))
                                    .collect()
                            })
                            .unwrap_or_default(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn load_fixture() -> Fixture {
    let dir = export_dir();
    let raw = std::fs::read(dir.join("manifest.json")).expect("manifest.json");
    let manifest: Value = serde_json::from_slice(&raw).expect("manifest parses");

    let mut sigs = HashMap::new();
    for entry in std::fs::read_dir(dir.join("sigs")).expect("sigs dir") {
        let path = entry.expect("dir entry").path();
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some((tl, idx)) = stem.split_once('_') else {
            continue;
        };
        let (Ok(tl), Ok(idx)) = (tl.parse::<u64>(), idx.parse::<u64>()) else {
            continue;
        };
        let raw = std::fs::read(&path).expect("sig blob reads");
        let decoded = decode_wide_sigs(&raw)
            .unwrap_or_else(|| panic!("sig blob {} fails to decode", path.display()));
        sigs.insert((tl, idx), decoded);
    }

    let mut events = BTreeMap::new();
    for d in manifest["dialogue_turns"].as_array().expect("dialogue") {
        let (Some(tl), Some(idx)) = (
            d["timeline"].as_str().and_then(|s| s.parse::<u64>().ok()),
            d["index"].as_u64(),
        ) else {
            continue;
        };
        let Some(name) = d["events"].as_str() else {
            continue;
        };
        let raw = std::fs::read(dir.join("events").join(name)).expect("events file");
        events.insert((tl, idx), decode_events(&raw));
    }

    Fixture {
        sigs,
        events,
        dialogue: parse_refs(&manifest["dialogue_turns"]),
        candidates: parse_refs(&manifest["selected_candidates"]),
        targets: manifest["targets"]
            .as_object()
            .expect("targets")
            .iter()
            .map(|(name, v)| (name.clone(), parse_refs(v)))
            .collect(),
    }
}

impl Fixture {
    fn sig(&self, tl: u64, idx: u64) -> &[WideQSig] {
        &self.sigs[&(tl, idx)]
    }

    /// Probe window at generated-token position `t` (mirrors the TDD suite).
    fn probe_at(&self, tl: u64, idx: u64, t: u32) -> Vec<WideQSig> {
        let all = self.sig(tl, idx);
        let generated_total = self.events[&(tl, idx)]
            .iter()
            .map(|e| e.start_token)
            .max()
            .unwrap_or(0) as usize;
        let user_len = all.len().saturating_sub(generated_total);
        let end = (user_len + t as usize).min(all.len()).max(1);
        all[end.saturating_sub(MAX_PROBE_TOKENS)..end].to_vec()
    }

    /// Trailing probe of the whole turn (its final reprojection's view).
    fn probe(&self, tl: u64, idx: u64) -> Vec<WideQSig> {
        let all = self.sig(tl, idx);
        all[all.len().saturating_sub(MAX_PROBE_TOKENS)..].to_vec()
    }

    fn code_candidates(&self, exclude_path: Option<&str>) -> Vec<&TurnRef> {
        self.candidates
            .iter()
            .filter(|c| c.tags.first().map(String::as_str) == Some("code"))
            .filter(|c| !exclude_path.is_some_and(|ex| c.path().contains(ex)))
            .collect()
    }
}

/// One named slot in a ranking pool: label + its gallery windows.
struct Slot<'a> {
    label: String,
    windows: Vec<&'a [WideQSig]>,
}

fn rank_weighted(probe: &[WideQSig], slots: &[Slot], weights: &[f32]) -> Vec<f32> {
    let mut windows: Vec<&[WideQSig]> = Vec::new();
    let mut slot_of: Vec<usize> = Vec::new();
    for (i, s) in slots.iter().enumerate() {
        for w in &s.windows {
            windows.push(w);
            slot_of.push(i);
        }
    }
    score_slots_weighted(probe, &windows, &slot_of, slots.len(), weights)
}

fn sorted_desc(slots: &[Slot], scores: &[f32]) -> Vec<(String, f32)> {
    let mut out: Vec<(String, f32)> = slots
        .iter()
        .map(|s| s.label.clone())
        .zip(scores.iter().copied())
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

fn print_top(title: &str, ranked: &[(String, f32)], n: usize) {
    println!("  {title}");
    for (i, (label, score)) in ranked.iter().take(n).enumerate() {
        println!("    {:>2}. {score:>9.1}  {label}", i + 1);
    }
}

fn rank_of(ranked: &[(String, f32)], pred: impl Fn(&str) -> bool) -> Option<(usize, f32)> {
    ranked
        .iter()
        .enumerate()
        .find(|(_, (l, _))| pred(l))
        .map(|(i, (_, s))| (i + 1, *s))
}

/// The junk pool the recorded projections actually selected (one slot per turn),
/// optionally excluding a path substring.
fn junk_slots<'a>(f: &'a Fixture, exclude: Option<&str>) -> Vec<Slot<'a>> {
    f.code_candidates(exclude)
        .iter()
        .map(|c| Slot {
            label: format!("{}#{}", c.path(), c.index),
            windows: vec![f.sig(c.timeline, c.index)],
        })
        .collect()
}

/// Per-exchange builder.rs slots (one slot per exported turn of the file's
/// conversation), labeled by turn index.
fn builder_slots<'a>(f: &'a Fixture) -> Vec<Slot<'a>> {
    f.targets["models/builder.rs"]
        .iter()
        .map(|t| Slot {
            label: format!("builder.rs#{}", t.index),
            windows: vec![f.sig(t.timeline, t.index)],
        })
        .collect()
}

fn structure_slots<'a>(f: &'a Fixture) -> Vec<Slot<'a>> {
    f.targets["repo_map"]
        .iter()
        .map(|t| Slot {
            label: format!("cluster:{}#{}", t.path(), t.index),
            windows: vec![f.sig(t.timeline, t.index)],
        })
        .collect()
}

// ── E1 — anatomy of the ModelBuilder miss ────────────────────────────────────

fn e1(f: &Fixture) {
    println!("\n═══ E1: ModelBuilder probe — builder.rs per-exchange anatomy ═══");
    let mut slots = builder_slots(f);
    let n_builder = slots.len();
    slots.extend(junk_slots(f, Some("models/builder.rs")));
    let probe = f.probe(TOUR_TL, T_MODELBUILDER);
    let scores = rank_weighted(&probe, &slots, &[]);
    let ranked = sorted_desc(&slots, &scores);

    print_top(
        "top 12 (builder.rs exchanges as individual slots vs junk):",
        &ranked,
        12,
    );
    let mut b: Vec<f32> = scores[..n_builder].to_vec();
    b.sort_by(|x, y| y.total_cmp(x));
    let best = rank_of(&ranked, |l| l.starts_with("builder.rs")).unwrap();
    println!(
        "  builder.rs: {} exchanges — best rank {} (score {:.1}); \
         per-exchange max/median/min = {:.1}/{:.1}/{:.1}",
        n_builder,
        best.0,
        best.1,
        b[0],
        b[n_builder / 2],
        b[n_builder - 1]
    );
    let in_top20 = ranked
        .iter()
        .take(20)
        .filter(|(l, _)| l.starts_with("builder.rs"))
        .count();
    println!("  builder.rs exchanges in the top 20: {in_top20}");
}

// ── E2 — promiscuity levels + normalized re-ranking ──────────────────────────

fn e2(f: &Fixture) -> Vec<f32> {
    println!("\n═══ E2: promiscuity normalization (Concept A premise) ═══");
    let mut slots = junk_slots(f, None);
    let n_junk = slots.len();
    slots.extend(builder_slots(f));
    let n_builder = slots.len() - n_junk;
    slots.extend(structure_slots(f));
    println!(
        "  pool: {n_junk} junk + {n_builder} builder.rs + {} structure slots",
        slots.len() - n_junk - n_builder
    );

    // Level = mean score over every captured dialogue probe (all 6
    // conversations, 30 turns) — the promiscuity of each slot against real but
    // mostly-unrelated probes. Approximates the EWMA hit-level a promiscuous
    // child learns (and self-mutes by) under the production normalizer.
    let mut level = vec![0.0f32; slots.len()];
    let mut n_probes = 0usize;
    for d in &f.dialogue {
        let probe = f.probe(d.timeline, d.index);
        if probe.is_empty() {
            continue;
        }
        let s = rank_weighted(&probe, &slots, &[]);
        for (l, v) in level.iter_mut().zip(&s) {
            *l += v;
        }
        n_probes += 1;
    }
    for l in level.iter_mut() {
        *l /= n_probes.max(1) as f32;
    }
    println!("  levels learned from {n_probes} dialogue probes");

    let mut by_level = sorted_desc(&slots, &level);
    print_top("most promiscuous slots (highest mean level):", &by_level, 8);
    by_level.reverse();

    for (name, tl, idx, target) in [
        ("tour", TOUR_TL, T_TOUR, "cluster:"),
        ("ModelBuilder", TOUR_TL, T_MODELBUILDER, "builder.rs"),
    ] {
        let probe = f.probe(tl, idx);
        let raw = rank_weighted(&probe, &slots, &[]);
        let ranked_raw = sorted_desc(&slots, &raw);
        let norm: Vec<f32> = raw
            .iter()
            .zip(&level)
            .map(|(s, l)| s / l.max(1.0))
            .collect();
        let ranked_norm = sorted_desc(&slots, &norm);
        println!("  ── {name} probe ──");
        print_top("raw top 5:", &ranked_raw, 5);
        print_top("score/level top 5:", &ranked_norm, 5);
        let raw_rank = rank_of(&ranked_raw, |l| l.starts_with(target));
        let norm_rank = rank_of(&ranked_norm, |l| l.starts_with(target));
        println!(
            "    target '{target}*': raw rank {:?} → normalized rank {:?}",
            raw_rank.map(|r| r.0),
            norm_rank.map(|r| r.0)
        );
    }
    level
}

// ── E3 — attention-mass formula alternatives ─────────────────────────────────

fn e3(f: &Fixture, level: &[f32]) {
    println!("\n═══ E3: mass formulas on the recall-vs-code contrast (Concept B) ═══");
    // Same slot order as E2 so `level` aligns; mass measured over the junk pool
    // (code candidates) exactly like the red test.
    let mut slots = junk_slots(f, None);
    let n_junk = slots.len();
    slots.extend(builder_slots(f));
    slots.extend(structure_slots(f));

    let probes = [
        ("code (ModelBuilder, t5)", f.probe(TOUR_TL, T_MODELBUILDER)),
        ("recall (history, t6)", f.probe(TOUR_TL, T_RECALL)),
        ("tour (t0)", f.probe(TOUR_TL, T_TOUR)),
    ];
    println!(
        "  {:<26} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "probe", "sum", "top5", "top1shr", "entropy", "n-sum", "n-gated"
    );
    for (name, probe) in &probes {
        let raw = rank_weighted(probe, &slots, &[]);
        let code = &raw[..n_junk];
        let mut sorted: Vec<f32> = code.to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let sum: f32 = code.iter().sum();
        let top5: f32 = sorted.iter().take(5).sum();
        let top1_share = if sum > 0.0 { sorted[0] / sum } else { 0.0 };
        let entropy: f32 = if sum > 0.0 {
            -code
                .iter()
                .filter(|&&s| s > 0.0)
                .map(|s| {
                    let p = s / sum;
                    p * p.ln()
                })
                .sum::<f32>()
        } else {
            0.0
        };
        let norm: Vec<f32> = code
            .iter()
            .zip(&level[..n_junk])
            .map(|(s, l)| s / l.max(1.0))
            .collect();
        let n_sum: f32 = norm.iter().sum();
        // The design's mass formula on the normalized band: gated sum over
        // candidates clearing a floor (here 2× the slot's own mean level).
        let n_gated: f32 = norm.iter().filter(|&&s| s >= 2.0).sum();
        println!(
            "  {name:<26} {sum:>9.0} {top5:>9.0} {top1_share:>9.3} {entropy:>9.2} {n_sum:>9.1} {n_gated:>9.1}"
        );
    }
    println!(
        "  (Concept B needs code-probe mass > recall-probe mass; compare each \
         column's t5 vs t6 row)"
    );
}

// ── E4 — fold-group decomposition ────────────────────────────────────────────

fn e4(f: &Fixture) {
    println!("\n═══ E4: fold-group decomposition (margin-id analog, §83) ═══");
    let weight_sets: [(&str, &[f32]); 5] = [
        ("uniform", &[]),
        ("L0-45 only", &[1.0, 0.0, 0.0]),
        ("L46 only", &[0.0, 1.0, 0.0]),
        ("L47 only", &[0.0, 0.0, 1.0]),
        ("[0,1,1] id-groups", &[0.0, 1.0, 1.0]),
    ];

    let mut mb_slots = builder_slots(f);
    mb_slots.extend(junk_slots(f, Some("models/builder.rs")));
    let mb_probe = f.probe(TOUR_TL, T_MODELBUILDER);

    let mut tour_slots = structure_slots(f);
    tour_slots.extend(junk_slots(f, None));
    let tour_probe = f.probe(TOUR_TL, T_TOUR);

    println!(
        "  {:<20} {:>14} {:>14}   {}",
        "weights", "builder best", "structure best", "top slot (ModelBuilder | tour)"
    );
    for (name, w) in weight_sets {
        let mb = sorted_desc(&mb_slots, &rank_weighted(&mb_probe, &mb_slots, w));
        let tour = sorted_desc(&tour_slots, &rank_weighted(&tour_probe, &tour_slots, w));
        let b = rank_of(&mb, |l| l.starts_with("builder.rs"));
        let s = rank_of(&tour, |l| l.starts_with("cluster:"));
        println!(
            "  {name:<20} {:>14} {:>14}   {} | {}",
            b.map(|r| format!("#{} @{:.0}", r.0, r.1))
                .unwrap_or_default(),
            s.map(|r| format!("#{} @{:.0}", r.0, r.1))
                .unwrap_or_default(),
            mb[0].0,
            tour[0].0
        );
    }
}

// ── E5 — probe-window sweep ──────────────────────────────────────────────────

fn e5(f: &Fixture) {
    println!("\n═══ E5: probe-window sweep (ModelBuilder probe) ═══");
    let mut slots = builder_slots(f);
    slots.extend(junk_slots(f, Some("models/builder.rs")));
    let all = f.sig(TOUR_TL, T_MODELBUILDER);
    println!("  turn has {} sig tokens", all.len());
    for window in [32usize, 64, 128, 256, 512] {
        let probe = &all[all.len().saturating_sub(window)..];
        let ranked = sorted_desc(&slots, &rank_weighted(probe, &slots, &[]));
        let b = rank_of(&ranked, |l| l.starts_with("builder.rs")).unwrap();
        println!(
            "  window {window:>4}: builder.rs best rank {:>2} @{:>7.1}; top = {} @{:.1}",
            b.0, b.1, ranked[0].0, ranked[0].1
        );
    }
}

// ── E6 — probe evolution across the recorded reprojection cadence ────────────

fn e6(f: &Fixture) {
    println!("\n═══ E6: probe evolution over decode (question → reasoning) ═══");
    for (name, idx, target) in [
        ("ModelBuilder t5", T_MODELBUILDER, "builder.rs"),
        ("tour t0", T_TOUR, "cluster:"),
    ] {
        let mut slots: Vec<Slot> = if target == "builder.rs" {
            builder_slots(f)
        } else {
            structure_slots(f)
        };
        slots.extend(junk_slots(
            f,
            (target == "builder.rs").then_some("models/builder.rs"),
        ));
        println!("  ── {name} (target {target}*) ──");
        let events = &f.events[&(TOUR_TL, idx)];
        for ev in events {
            let probe = f.probe_at(TOUR_TL, idx, ev.start_token);
            let ranked = sorted_desc(&slots, &rank_weighted(&probe, &slots, &[]));
            let t = rank_of(&ranked, |l| l.starts_with(target)).unwrap();
            println!(
                "    t={:>5}: target best rank {:>3} @{:>7.1}; top = {} @{:.1}",
                ev.start_token, t.0, t.1, ranked[0].0, ranked[0].1
            );
        }
    }
}

// ── E7 — self-match sanity + structure anatomy ───────────────────────────────

fn e7(f: &Fixture) {
    println!("\n═══ E7: gallery-side sanity ═══");

    // Self-match: probe a builder.rs exchange's own trailing window against the
    // full code pool — the gallery side of the domain gap.
    let mut slots = builder_slots(f);
    slots.extend(junk_slots(f, Some("models/builder.rs")));
    let builder_turns = &f.targets["models/builder.rs"];
    let mut hits = 0usize;
    let mut tried = 0usize;
    for t in builder_turns.iter().take(8) {
        let all = f.sig(t.timeline, t.index);
        if all.len() < 8 {
            continue;
        }
        let probe = &all[all.len().saturating_sub(MAX_PROBE_TOKENS)..];
        let ranked = sorted_desc(&slots, &rank_weighted(probe, &slots, &[]));
        tried += 1;
        if ranked[0].0.starts_with("builder.rs") {
            hits += 1;
        }
    }
    println!(
        "  self-match: {hits}/{tried} builder.rs exchange probes rank a \
         builder.rs slot top-1 against the junk pool"
    );

    // Structure anatomy: which clusters top the tour probe, and where the
    // workspace-root cluster sits.
    let slots = structure_slots(f);
    let ranked = sorted_desc(
        &slots,
        &rank_weighted(&f.probe(TOUR_TL, T_TOUR), &slots, &[]),
    );
    print_top(
        "tour probe vs structure-only gallery (top 8 clusters):",
        &ranked,
        8,
    );
    let root = rank_of(&ranked, |l| l.starts_with("cluster:.#"));
    println!("  workspace-root cluster rank: {:?}", root.map(|r| r.0));
}

// ── R2a — normalization level-floor sweep ────────────────────────────────────

/// E2 found normalization flips the tour probe to structure but promotes
/// quiet-slot noise (a two-line `ops.rs` fragment wins the ModelBuilder probe —
/// exactly what the daemon's recorded selection did). Sweep the floor under the
/// learned level to find the band where promiscuous slots stay muted without
/// amplifying fragments whose level is near zero.
fn r2a(f: &Fixture, level: &[f32]) {
    println!("\n═══ R2a: normalization level-floor sweep ═══");
    let mut slots = junk_slots(f, None);
    slots.extend(builder_slots(f));
    slots.extend(structure_slots(f));

    let tour = rank_weighted(&f.probe(TOUR_TL, T_TOUR), &slots, &[]);
    let mb = rank_weighted(&f.probe(TOUR_TL, T_MODELBUILDER), &slots, &[]);
    println!(
        "  {:>7} {:>13} {:>15}   {}",
        "floor", "tour target", "builder target", "ModelBuilder top slot"
    );
    for floor in [0.5f32, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0] {
        let norm = |raw: &[f32]| -> Vec<f32> {
            raw.iter()
                .zip(level)
                .map(|(s, l)| s / l.max(floor))
                .collect()
        };
        let rt = sorted_desc(&slots, &norm(&tour));
        let rm = sorted_desc(&slots, &norm(&mb));
        let t = rank_of(&rt, |l| l.starts_with("cluster:"));
        let b = rank_of(&rm, |l| l.starts_with("builder.rs"));
        println!(
            "  {floor:>7.1} {:>13} {:>15}   {}",
            t.map(|r| format!("#{}", r.0)).unwrap_or_default(),
            b.map(|r| format!("#{}", r.0)).unwrap_or_default(),
            rm[0].0
        );
    }
}

// ── R2b — cross-group consensus gating ───────────────────────────────────────

/// E4 found `session.rs#35`'s 2004 comes almost entirely from one fold group
/// (L46). Test whether requiring multi-group agreement — combining the three
/// single-group score vectors by min / geometric mean / agreement count —
/// suppresses single-group spikes without killing genuine matches.
fn r2b(f: &Fixture) {
    println!("\n═══ R2b: cross-group consensus (kill single-group spikes) ═══");
    let mut slots = builder_slots(f);
    slots.extend(junk_slots(f, Some("models/builder.rs")));
    let probe = f.probe(TOUR_TL, T_MODELBUILDER);

    let g0 = rank_weighted(&probe, &slots, &[1.0, 0.0, 0.0]);
    let g1 = rank_weighted(&probe, &slots, &[0.0, 1.0, 0.0]);
    let g2 = rank_weighted(&probe, &slots, &[0.0, 0.0, 1.0]);

    for who in ["zend/src/session.rs#35", "builder.rs"] {
        let i = slots.iter().position(|s| s.label.starts_with(who)).unwrap();
        // For builder.rs report its best exchange per group, not exchange 0.
        let (a, b, c) = if who == "builder.rs" {
            let best = |v: &[f32]| {
                slots
                    .iter()
                    .zip(v)
                    .filter(|(s, _)| s.label.starts_with("builder.rs"))
                    .map(|(_, x)| *x)
                    .fold(0.0f32, f32::max)
            };
            (best(&g0), best(&g1), best(&g2))
        } else {
            (g0[i], g1[i], g2[i])
        };
        println!("  {who:<28} per-group scores: L0-45 {a:>7.1}  L46 {b:>7.1}  L47 {c:>7.1}");
    }

    let combos: [(&str, Box<dyn Fn(f32, f32, f32) -> f32>); 4] = [
        ("sum (today)", Box::new(|a, b, c| a + b + c)),
        ("min", Box::new(|a, b, c| a.min(b).min(c))),
        ("geo-mean", Box::new(|a, b, c| (a * b * c).max(0.0).cbrt())),
        (
            "2-of-3 gate × sum",
            Box::new(|a, b, c| {
                let n = [a, b, c].iter().filter(|&&x| x > 5.0).count();
                if n >= 2 {
                    a + b + c
                } else {
                    0.0
                }
            }),
        ),
    ];
    for (name, combine) in combos {
        let fused: Vec<f32> = (0..slots.len())
            .map(|i| combine(g0[i], g1[i], g2[i]))
            .collect();
        let ranked = sorted_desc(&slots, &fused);
        let b = rank_of(&ranked, |l| l.starts_with("builder.rs")).unwrap();
        let s = rank_of(&ranked, |l| l.starts_with("zend/src/session.rs")).unwrap();
        println!(
            "  {name:<18} builder best #{:<3} session.rs #{:<3} top = {} @{:.1}",
            b.0, s.0, ranked[0].0, ranked[0].1
        );
    }
}

// ── R2c — question-anchored probe fusion ─────────────────────────────────────

/// E6 found the question-time probe is weak but *right* while the decode-tail
/// probe is strong but *wrong* (it echoes whatever junk is already projected —
/// a contamination feedback loop). Test fusing the two scans: normalize each
/// scan's scores to its own max, then `w_q × question + (1−w_q) × tail`.
fn r2c(f: &Fixture) {
    println!("\n═══ R2c: question-anchored probe fusion ═══");
    for (name, idx, target) in [
        ("tour t0", T_TOUR, "cluster:"),
        ("ModelBuilder t5", T_MODELBUILDER, "builder.rs"),
    ] {
        let mut slots: Vec<Slot> = if target == "builder.rs" {
            builder_slots(f)
        } else {
            structure_slots(f)
        };
        slots.extend(junk_slots(
            f,
            (target == "builder.rs").then_some("models/builder.rs"),
        ));
        let question = rank_weighted(&f.probe_at(TOUR_TL, idx, 0), &slots, &[]);
        let tail = rank_weighted(&f.probe(TOUR_TL, idx), &slots, &[]);
        let unit = |v: &[f32]| -> Vec<f32> {
            let m = v.iter().fold(0.0f32, |a, &b| a.max(b)).max(1e-6);
            v.iter().map(|x| x / m).collect()
        };
        let (qn, tn) = (unit(&question), unit(&tail));
        println!("  ── {name} (target {target}*) ──");
        for wq in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let fused: Vec<f32> = qn
                .iter()
                .zip(&tn)
                .map(|(q, t)| wq * q + (1.0 - wq) * t)
                .collect();
            let ranked = sorted_desc(&slots, &fused);
            let t = rank_of(&ranked, |l| l.starts_with(target)).unwrap();
            println!(
                "    w_q={wq:.2}: target best rank {:>3}; top = {}",
                t.0, ranked[0].0
            );
        }
    }
}

// ── R2d — the winning combination: question probe × cross-group consensus ────

/// R2b (min-fusion across fold groups kills single-group junk spikes) and R2c
/// (the question window carries the true intent) each independently lift the
/// targets. Combine them: score the question probe AND the decode tail per
/// fold-group, min-fuse across groups, then take the per-slot max of the two
/// scans (either window may hold the needle; contamination is scan-specific).
fn r2d(f: &Fixture) {
    println!("\n═══ R2d: question probe × cross-group consensus, max-fused ═══");
    for (name, idx, target) in [
        ("tour t0", T_TOUR, "cluster:"),
        ("ModelBuilder t5", T_MODELBUILDER, "builder.rs"),
    ] {
        let mut slots: Vec<Slot> = if target == "builder.rs" {
            builder_slots(f)
        } else {
            structure_slots(f)
        };
        slots.extend(junk_slots(
            f,
            (target == "builder.rs").then_some("models/builder.rs"),
        ));
        let consensus = |probe: &[WideQSig]| -> Vec<f32> {
            let g0 = rank_weighted(probe, &slots, &[1.0, 0.0, 0.0]);
            let g1 = rank_weighted(probe, &slots, &[0.0, 1.0, 0.0]);
            let g2 = rank_weighted(probe, &slots, &[0.0, 0.0, 1.0]);
            (0..slots.len())
                .map(|i| g0[i].min(g1[i]).min(g2[i]))
                .collect()
        };
        let unit = |v: &[f32]| -> Vec<f32> {
            let m = v.iter().fold(0.0f32, |a, &b| a.max(b)).max(1e-6);
            v.iter().map(|x| x / m).collect()
        };
        let q = unit(&consensus(&f.probe_at(TOUR_TL, idx, 0)));
        let t = unit(&consensus(&f.probe(TOUR_TL, idx)));
        let fused: Vec<f32> = q.iter().zip(&t).map(|(a, b)| a.max(*b)).collect();
        for (scan, scores) in [("question", &q), ("tail", &t), ("max-fused", &fused)] {
            let ranked = sorted_desc(&slots, scores);
            let tr = rank_of(&ranked, |l| l.starts_with(target)).unwrap();
            println!(
                "  {name:<16} {scan:<10} target best rank {:>3}; top = {}",
                tr.0, ranked[0].0
            );
        }
    }
}

// ═══ Round 3 — the revised design's unmeasured claims (design doc v2) ════════
//
// R3a: tools-axis fusion gate (§13 open question 1) — leave-one-out tool
//      ranking over the fixture's 95-tool gallery under additive vs consensus.
// R3b: size-aware level prior (A.4) — does floor ∝ T_ref/tokens resolve the
//      floor conflict a flat floor cannot (R2a)?
// R3c: full pipeline composition (F+G+A) — window-fusion variants against the
//      tour + ModelBuilder ideals; is per-child level division alone enough to
//      make Q- and D-windows comparable, or does F.1 need per-window scaling?
// R3d: mass constants (B.1) — k/ρ sweep on the winning pipeline's scores
//      against the three ideal mass orderings.
// R3e: generalization (§13) — old vs new pipeline over every captured dialogue
//      turn, not just the two turns the mechanisms were derived on.

/// Per-group scans for Concept G fusion operators. `production` is the shipped
/// single-pass additive scorer (needle gate over the summed per-token
/// magnitude); `g[0..3]` are the three per-group-gated scans (L0–45 content,
/// L46, L47 identity).
struct GroupScans {
    production: Vec<f32>,
    g: [Vec<f32>; 3],
}

fn group_scans(probe: &[WideQSig], slots: &[Slot]) -> GroupScans {
    GroupScans {
        production: rank_weighted(probe, slots, &[]),
        g: [
            rank_weighted(probe, slots, &[1.0, 0.0, 0.0]),
            rank_weighted(probe, slots, &[0.0, 1.0, 0.0]),
            rank_weighted(probe, slots, &[0.0, 0.0, 1.0]),
        ],
    }
}

/// The fusion-operator zoo. Every operator maps `(g0, g1, g2)` → score; the
/// production additive baseline is carried separately (its needle gate spans
/// groups). `content_gated` is the "identity confirms, content decides"
/// candidate: identity-group votes count only when the content group agrees at
/// all — kills pure id-spikes while preserving additive magnitude (and recall)
/// for balanced matches.
const FUSION_MODES: [(&str, fn(f32, f32, f32) -> f32); 5] = [
    ("additive (grouped)", |a, b, c| a + b + c),
    ("consensus_min", |a, b, c| a.min(b).min(c)),
    ("consensus_geo", |a, b, c| (a * b * c).max(0.0).cbrt()),
    (
        "content_gated",
        |a, b, c| if a > 0.0 { a + b + c } else { 0.0 },
    ),
    ("min(content, id-sum)", |a, b, c| a.min(b + c)),
];

fn fuse(scans: &GroupScans, mode: fn(f32, f32, f32) -> f32) -> Vec<f32> {
    (0..scans.g[0].len())
        .map(|i| mode(scans.g[0][i], scans.g[1][i], scans.g[2][i]))
        .collect()
}

/// Head window: the first `n` sig tokens of a turn — the question region.
///
/// The recorded events' `start_token` values exceed the sealed sig length for
/// long turns (they count view tokens, not turn sig tokens), so the TDD-suite
/// `user_len` reconstruction degenerates to a 1-token window on those turns
/// (measured in round 3's first pass: every "question" probe was `all[0..1]`).
/// The turn's sig sequence starts at its user prefix, so a head window is the
/// faithful offline approximation of Concept F's Q-window; production captures
/// the real prefix bounds at turn open and has no such ambiguity.
const HEAD_PROBE_TOKENS: usize = 64;

fn head_probe<'a>(f: &'a Fixture, tl: u64, idx: u64) -> &'a [WideQSig] {
    let all = f.sig(tl, idx);
    &all[..HEAD_PROBE_TOKENS.min(all.len())]
}

fn r3a(f: &Fixture) {
    println!("\n═══ R3a: tools-axis fusion gate — LOO over the fixture tool gallery ═══");
    // Group the tool gallery by tool tag; leave-one-out per turn.
    let mut by_tool: BTreeMap<String, Vec<&TurnRef>> = BTreeMap::new();
    for t in &f.targets["tool"] {
        let Some(tool) = t.tags.iter().find(|g| *g != "tool") else {
            continue;
        };
        by_tool.entry(tool.clone()).or_default().push(t);
    }
    let tools: Vec<&String> = by_tool.keys().collect();
    let mut top1 = [0usize; 6];
    let mut top5 = [0usize; 6];
    let mut n_probes = 0usize;
    for (tool, turns) in &by_tool {
        if turns.len() < 2 {
            continue;
        }
        for probe_turn in turns {
            let slots: Vec<Slot> = by_tool
                .iter()
                .map(|(name, ts)| Slot {
                    label: name.clone(),
                    windows: ts
                        .iter()
                        .filter(|t| {
                            (t.timeline, t.index) != (probe_turn.timeline, probe_turn.index)
                        })
                        .map(|t| f.sig(t.timeline, t.index))
                        .collect(),
                })
                .collect();
            let all = f.sig(probe_turn.timeline, probe_turn.index);
            let probe = &all[all.len().saturating_sub(MAX_PROBE_TOKENS)..];
            let scans = group_scans(probe, &slots);
            let idx = tools.iter().position(|t| *t == tool).unwrap();
            let mut mode_scores: Vec<Vec<f32>> = vec![scans.production.clone()];
            mode_scores.extend(FUSION_MODES.iter().map(|(_, m)| fuse(&scans, *m)));
            for (m, scores) in mode_scores.iter().enumerate() {
                let rank = scores.iter().filter(|&&s| s > scores[idx]).count() + 1;
                if rank == 1 {
                    top1[m] += 1;
                }
                if rank <= 5 {
                    top5[m] += 1;
                }
            }
            n_probes += 1;
        }
    }
    println!("  {n_probes} LOO probes over {} tools:", by_tool.len());
    let mut names = vec!["additive (production)"];
    names.extend(FUSION_MODES.iter().map(|(n, _)| *n));
    for (m, name) in names.iter().enumerate() {
        println!(
            "  {name:<26} Top-1 {:>5.1}%   Top-5 {:>5.1}%",
            100.0 * top1[m] as f32 / n_probes as f32,
            100.0 * top5[m] as f32 / n_probes as f32
        );
    }
}

/// Shared round-3 pool (same order everywhere): junk + builder + structure.
fn r3_pool<'a>(f: &'a Fixture) -> (Vec<Slot<'a>>, usize, usize) {
    let mut slots = junk_slots(f, None);
    let n_junk = slots.len();
    slots.extend(builder_slots(f));
    let n_builder = slots.len() - n_junk;
    slots.extend(structure_slots(f));
    (slots, n_junk, n_builder)
}

/// The round-3 pipeline fusion operator: content-gated (see `FUSION_MODES`).
fn gated(scans: &GroupScans) -> Vec<f32> {
    fuse(scans, FUSION_MODES[3].1)
}

/// Promiscuity levels of the pipeline's score distribution: mean content-gated
/// score over every captured dialogue probe.
fn gated_levels(f: &Fixture, slots: &[Slot]) -> Vec<f32> {
    let mut level = vec![0.0f32; slots.len()];
    let mut n = 0usize;
    for d in &f.dialogue {
        let probe = f.probe(d.timeline, d.index);
        if probe.is_empty() {
            continue;
        }
        let s = gated(&group_scans(&probe, slots));
        for (l, v) in level.iter_mut().zip(&s) {
            *l += v;
        }
        n += 1;
    }
    for l in level.iter_mut() {
        *l /= n.max(1) as f32;
    }
    level
}

fn r3b(f: &Fixture) -> (Vec<f32>, f32, f32) {
    println!("\n═══ R3b: size-aware level prior (A.4) on content-gated scores ═══");
    let (slots, _, _) = r3_pool(f);
    let level = gated_levels(f, &slots);
    let tokens: Vec<usize> = slots.iter().map(|s| s.windows[0].len()).collect();

    // Q-window (head) scan — the probe the level prior must serve (Concept F
    // makes it a permanent scoring component).
    let tour = gated(&group_scans(head_probe(f, TOUR_TL, T_TOUR), &slots));
    let mb = gated(&group_scans(head_probe(f, TOUR_TL, T_MODELBUILDER), &slots));
    println!(
        "  {:>10} {:>9} {:>12} {:>14} {:>14}   {}",
        "floor_base", "cap", "tour target", "builder target", "ops.rs#47", "ModelBuilder top"
    );
    let (mut best, mut best_key) = (usize::MAX, (0.0f32, 0.0f32));
    for base in [0.5f32, 1.0, 2.0, 5.0] {
        for cap in [1.0f32, 4.0, 8.0, 16.0] {
            let norm = |raw: &[f32]| -> Vec<f32> {
                raw.iter()
                    .zip(&level)
                    .zip(&tokens)
                    .map(|((s, l), t)| {
                        let floor =
                            base * (MAX_PROBE_TOKENS as f32 / (*t).max(1) as f32).clamp(1.0, cap);
                        s / l.max(floor)
                    })
                    .collect()
            };
            let rt = sorted_desc(&slots, &norm(&tour));
            let rm = sorted_desc(&slots, &norm(&mb));
            let t = rank_of(&rt, |l| l.starts_with("cluster:"))
                .map(|r| r.0)
                .unwrap_or(999);
            let b = rank_of(&rm, |l| l.starts_with("builder.rs"))
                .map(|r| r.0)
                .unwrap_or(999);
            let o = rank_of(&rm, |l| l.contains("ops.rs"))
                .map(|r| r.0)
                .unwrap_or(999);
            println!(
                "  {base:>10.1} {cap:>9.0} {t:>12} {b:>14} {o:>14}   {}",
                rm[0].0
            );
            if t + b < best {
                best = t + b;
                best_key = (base, cap);
            }
        }
    }
    println!(
        "  best (tour+builder rank sum): floor_base {} cap {}",
        best_key.0, best_key.1
    );
    (level, best_key.0, best_key.1)
}

/// The full revised pipeline: per-window consensus-min → level-prior
/// normalization → window fusion. `scale_per_window` controls whether each
/// window's vector is unit-max scaled before the per-slot max (the F.1
/// comparability question).
struct PipelineParams<'a> {
    level: &'a [f32],
    tokens: &'a [usize],
    floor_base: f32,
    floor_cap: f32,
}

fn pipeline_scores(
    f: &Fixture,
    tl: u64,
    idx: u64,
    slots: &[Slot],
    p: &PipelineParams,
    scale_per_window: bool,
    apply_level: bool,
) -> Vec<f32> {
    let windows = [head_probe(f, tl, idx).to_vec(), f.probe(tl, idx)];
    let mut fused = vec![0.0f32; slots.len()];
    for w in &windows {
        let mut s = gated(&group_scans(w, slots));
        if apply_level {
            for ((v, l), t) in s.iter_mut().zip(p.level).zip(p.tokens) {
                let floor = p.floor_base
                    * (MAX_PROBE_TOKENS as f32 / (*t).max(1) as f32).clamp(1.0, p.floor_cap);
                *v /= l.max(floor);
            }
        }
        if scale_per_window {
            let m = s.iter().fold(0.0f32, |a, &b| a.max(b)).max(1e-6);
            for v in s.iter_mut() {
                *v /= m;
            }
        }
        for (o, v) in fused.iter_mut().zip(&s) {
            *o = o.max(*v);
        }
    }
    fused
}

fn r3c(f: &Fixture, level: &[f32], floor_base: f32, floor_cap: f32) -> (Vec<f32>, Vec<f32>) {
    println!("\n═══ R3c: full pipeline (F+G+A) — window-fusion variants ═══");
    let (slots, _, _) = r3_pool(f);
    let tokens: Vec<usize> = slots.iter().map(|s| s.windows[0].len()).collect();
    let p = PipelineParams {
        level,
        tokens: &tokens,
        floor_base,
        floor_cap,
    };
    let variants: [(&str, bool, bool); 3] = [
        ("gated+unit-max (no A)", true, false),
        ("gated+level (A only)", false, true),
        ("gated+level+unit-max", true, true),
    ];
    let mut winner: Option<(Vec<f32>, Vec<f32>)> = None;
    for (name, scale, lvl) in variants {
        let tour = pipeline_scores(f, TOUR_TL, T_TOUR, &slots, &p, scale, lvl);
        let mb = pipeline_scores(f, TOUR_TL, T_MODELBUILDER, &slots, &p, scale, lvl);
        let rt = sorted_desc(&slots, &tour);
        let rm = sorted_desc(&slots, &mb);
        let t = rank_of(&rt, |l| l.starts_with("cluster:")).unwrap().0;
        let b = rank_of(&rm, |l| l.starts_with("builder.rs")).unwrap().0;
        println!(
            "  {name:<26} tour target #{t:<3} (top {})  builder #{b:<3} (top {})",
            rt[0].0, rm[0].0
        );
        if lvl && !scale {
            winner = Some((tour, mb));
        }
    }
    let (tour, mb) = winner.expect("full variant ran");

    // Decode-trajectory stability of the full pipeline (Concept E premise: the
    // fused signal should hold rank across the whole decode, unlike E6).
    let (slots2, _, _) = r3_pool(f);
    println!("  trajectory (full variant), ModelBuilder turn:");
    for ev in &f.events[&(TOUR_TL, T_MODELBUILDER)] {
        let windows = [
            head_probe(f, TOUR_TL, T_MODELBUILDER).to_vec(),
            f.probe_at(TOUR_TL, T_MODELBUILDER, ev.start_token),
        ];
        let mut fused = vec![0.0f32; slots2.len()];
        for w in &windows {
            let mut s = gated(&group_scans(w, &slots2));
            for ((v, l), t) in s.iter_mut().zip(level).zip(&tokens) {
                let floor = floor_base
                    * (MAX_PROBE_TOKENS as f32 / (*t).max(1) as f32).clamp(1.0, floor_cap);
                *v /= l.max(floor);
            }
            for (o, v) in fused.iter_mut().zip(&s) {
                *o = o.max(*v);
            }
        }
        let ranked = sorted_desc(&slots2, &fused);
        let b = rank_of(&ranked, |l| l.starts_with("builder.rs")).unwrap();
        println!(
            "    t={:>4}: builder best rank {:>2}; top = {}",
            ev.start_token, b.0, ranked[0].0
        );
    }
    (tour, mb)
}

fn r3d(f: &Fixture, tour: &[f32], mb: &[f32], level: &[f32], floor_base: f32, floor_cap: f32) {
    println!("\n═══ R3d: mass constants (B.1) on the full pipeline ═══");
    let (slots, n_junk, n_builder) = r3_pool(f);
    let tokens: Vec<usize> = slots.iter().map(|s| s.windows[0].len()).collect();
    let p = PipelineParams {
        level,
        tokens: &tokens,
        floor_base,
        floor_cap,
    };
    let recall = pipeline_scores(f, TOUR_TL, T_RECALL, &slots, &p, true, true);

    let n_code = n_junk + n_builder;
    let mass = |scores: &[f32], range: std::ops::Range<usize>, k: usize, rho: f32| -> f32 {
        let mut s: Vec<f32> = scores[range].iter().copied().filter(|v| *v > 0.0).collect();
        s.sort_by(|a, b| b.total_cmp(a));
        let sum: f32 = s.iter().sum();
        if sum <= 0.0 {
            return 0.0;
        }
        let topk: f32 = s.iter().take(k).sum();
        sum * (topk / sum).powf(rho)
    };
    println!(
        "  {:>3} {:>4}  {:>22}  {:>22}  {:>24}",
        "k", "rho", "tour: struct vs code", "MB: code vs struct", "recall-code vs MB-code"
    );
    for k in [1usize, 3, 5] {
        for rho in [0.0f32, 0.5, 1.0, 2.0] {
            let tour_s = mass(tour, n_code..slots.len(), k, rho);
            let tour_c = mass(tour, 0..n_code, k, rho);
            let mb_c = mass(mb, 0..n_code, k, rho);
            let mb_s = mass(mb, n_code..slots.len(), k, rho);
            let rec_c = mass(&recall, 0..n_code, k, rho);
            let ok1 = tour_s > tour_c;
            let ok2 = mb_c > mb_s;
            let ok3 = rec_c < 0.5 * mb_c;
            println!(
                "  {k:>3} {rho:>4.1}  {:>9.2} vs {:<9.2}{}  {:>9.2} vs {:<9.2}{}  {:>9.2} vs {:<9.2}{}",
                tour_s,
                tour_c,
                if ok1 { "✓" } else { "✗" },
                mb_c,
                mb_s,
                if ok2 { "✓" } else { "✗" },
                rec_c,
                mb_c,
                if ok3 { "✓" } else { "✗" },
            );
        }
    }
    println!("  (✓ = the ideal ordering from ideal_projections.json holds)");
}

fn r3e(f: &Fixture, level: &[f32], floor_base: f32, floor_cap: f32) {
    println!("\n═══ R3e: generalization — old vs new pipeline, every dialogue turn ═══");
    let (slots, _, _) = r3_pool(f);
    let tokens: Vec<usize> = slots.iter().map(|s| s.windows[0].len()).collect();
    let p = PipelineParams {
        level,
        tokens: &tokens,
        floor_base,
        floor_cap,
    };
    println!(
        "  {:<26} {:>10} {:>10}   {}",
        "turn", "old struct", "new struct", "new top slot"
    );
    for d in &f.dialogue {
        if !f.events.contains_key(&(d.timeline, d.index)) {
            continue;
        }
        let old = rank_weighted(&f.probe(d.timeline, d.index), &slots, &[]);
        let new = pipeline_scores(f, d.timeline, d.index, &slots, &p, false, true);
        let ro = sorted_desc(&slots, &old);
        let rn = sorted_desc(&slots, &new);
        let so = rank_of(&ro, |l| l.starts_with("cluster:"))
            .map(|r| r.0)
            .unwrap_or(999);
        let sn = rank_of(&rn, |l| l.starts_with("cluster:"))
            .map(|r| r.0)
            .unwrap_or(999);
        let tl_short = d.timeline % 10_000;
        println!(
            "  tl…{tl_short:<5}#{:<3} ({:>4} tok)  {so:>8} {sn:>10}   {}",
            d.index,
            f.sig(d.timeline, d.index).len(),
            rn[0].0
        );
    }
    println!("  (struct rank matters for tour-shaped turns; tool/recall turns are context)");
}

// ═══ Round 4 — closing the remaining design open questions ═══════════════════
//
// R4a (momentum, Concept E): simulate the per-event selection sequence of every
//      tour-conversation turn under the full pipeline, with a velocity term at
//      μ ∈ {0, 0.5, 1.0} — does any real probe sequence show rising interest
//      that plain per-event ranking loses (the pattern momentum exists to fix)?
// R4b (root cluster, §13): within-structure competition under the full
//      pipeline — where does the workspace-root cluster rank for tour probes,
//      and what does a k=2 structure budget plus the `default {tag "."}` floor
//      actually project?

fn r4a(f: &Fixture, level: &[f32], floor_base: f32, floor_cap: f32) {
    println!("\n═══ R4a: momentum evidence — per-event stability under the pipeline ═══");
    let (slots, _, _) = r3_pool(f);
    let tokens: Vec<usize> = slots.iter().map(|s| s.windows[0].len()).collect();
    let normalize = |s: &mut [f32]| {
        for ((v, l), t) in s.iter_mut().zip(level).zip(&tokens) {
            let floor =
                floor_base * (MAX_PROBE_TOKENS as f32 / (*t).max(1) as f32).clamp(1.0, floor_cap);
            *v /= l.max(floor);
        }
    };
    println!(
        "  {:<8} {:>7} {:>13} {:>11} {:>11}   {}",
        "turn", "events", "target-top1", "top1-churn", "mu-gain", "target"
    );
    for (idx, target) in [
        (T_TOUR, "cluster:"),
        (T_MODELBUILDER, "builder.rs"),
        (T_RECALL, "-"),
    ] {
        let events = &f.events[&(TOUR_TL, idx)];
        // Head-window scan once per turn; tail scan per event.
        let mut head = gated(&group_scans(head_probe(f, TOUR_TL, idx), &slots));
        normalize(&mut head);
        let per_event: Vec<Vec<f32>> = events
            .iter()
            .map(|ev| {
                let mut tail = gated(&group_scans(
                    &f.probe_at(TOUR_TL, idx, ev.start_token),
                    &slots,
                ));
                normalize(&mut tail);
                head.iter().zip(&tail).map(|(a, b)| a.max(*b)).collect()
            })
            .collect();

        for mu in [0.0f32, 0.5, 1.0] {
            let mut v = vec![0.0f32; slots.len()];
            let mut prev = vec![0.0f32; slots.len()];
            let mut target_top1 = 0usize;
            let mut top1_slots: Vec<usize> = Vec::new();
            for scores in &per_event {
                let seeded: Vec<f32> = scores.iter().zip(&v).map(|(s, vel)| s + mu * vel).collect();
                let top = (0..slots.len())
                    .max_by(|a, b| seeded[*a].total_cmp(&seeded[*b]))
                    .unwrap();
                if !top1_slots.contains(&top) {
                    top1_slots.push(top);
                }
                if target != "-" && slots[top].label.starts_with(target) {
                    target_top1 += 1;
                }
                for i in 0..v.len() {
                    v[i] = 0.5 * v[i] + (scores[i] - prev[i]).max(0.0);
                }
                prev.clone_from(scores);
            }
            println!(
                "  t{idx} mu={mu:<4} {:>6} {:>10}/{:<2} {:>11} {:>11}   {target}",
                per_event.len(),
                target_top1,
                per_event.len(),
                top1_slots.len(),
                if mu == 0.0 { "baseline" } else { "vs mu=0" },
            );
        }
    }
    println!(
        "  (momentum earns its gain only if mu>0 raises target-top1 or lowers churn \
         vs the mu=0 baseline)"
    );
}

fn r4b(f: &Fixture, level: &[f32], floor_base: f32, floor_cap: f32) {
    println!("\n═══ R4b: root-cluster rank within structure (full pipeline) ═══");
    let (slots, n_junk, n_builder) = r3_pool(f);
    let tokens: Vec<usize> = slots.iter().map(|s| s.windows[0].len()).collect();
    let p = PipelineParams {
        level,
        tokens: &tokens,
        floor_base,
        floor_cap,
    };
    let n_code = n_junk + n_builder;
    // Root cluster = the repo_map turn whose cluster tag is "." (or the
    // shortest path tag — the workspace root).
    let root_label = f.targets["repo_map"]
        .iter()
        .min_by_key(|t| t.path().len())
        .map(|t| format!("cluster:{}#{}", t.path(), t.index))
        .unwrap();
    println!("  root cluster slot: {root_label}");
    for d in &f.dialogue {
        if !f.events.contains_key(&(d.timeline, d.index)) {
            continue;
        }
        let scores = pipeline_scores(f, d.timeline, d.index, &slots, &p, false, true);
        let mut structure: Vec<(usize, f32)> =
            (n_code..slots.len()).map(|i| (i, scores[i])).collect();
        structure.sort_by(|a, b| b.1.total_cmp(&a.1));
        let root_rank = structure
            .iter()
            .position(|(i, _)| slots[*i].label == root_label)
            .map(|r| r + 1)
            .unwrap_or(0);
        let top2: Vec<&str> = structure
            .iter()
            .take(2)
            .map(|(i, _)| slots[*i].label.as_str())
            .collect();
        println!(
            "  tl…{:<5}#{:<3} root within-structure rank {:>3}   k=2 picks: {}",
            d.timeline % 10_000,
            d.index,
            root_rank,
            top2.join(" | ")
        );
    }
    println!(
        "  (the `default {{tag \".\"}}` floor guarantees root PRESENCE regardless; \
         this measures whether it ever wins organically)"
    );
}

// R4c (short-probe residual, §13): the ~24-token turns that rank a quiet code
// file top are all TOOL-shaped questions ("what time is it?") — no code slot is
// right for them, so the code-axis top is arbitrary. What matters is whether
// their ABSOLUTE pipeline scores/mass sit far below a genuine code question's,
// so Concept B's gate mutes the layer. Measure exactly that.
fn r4c(f: &Fixture, level: &[f32], floor_base: f32, floor_cap: f32) {
    println!("\n═══ R4c: short tool-question turns — absolute mass vs a code question ═══");
    let (slots, n_junk, n_builder) = r3_pool(f);
    let tokens: Vec<usize> = slots.iter().map(|s| s.windows[0].len()).collect();
    let p = PipelineParams {
        level,
        tokens: &tokens,
        floor_base,
        floor_cap,
    };
    let n_code = n_junk + n_builder;
    // Timeline 0 = unresolved here; the loop below resolves it by suffix.
    let cases: [(&str, u64, u64); 5] = [
        ("what time is it? (tour#1)", TOUR_TL, 1),
        ("what is the time? (crates_b#1)", 0, 1),
        ("ModelBuilder (tour#5)", TOUR_TL, 5),
        ("tour (tour#0)", TOUR_TL, 0),
        ("recall (tour#6)", TOUR_TL, 6),
    ];
    println!(
        "  {:<34} {:>8} {:>10} {:>12}",
        "turn", "tokens", "code top1", "code mass"
    );
    for (name, tl, idx) in cases {
        // Resolve the crates_b timeline by suffix when the literal isn't known.
        let (tl, idx) = if f.events.contains_key(&(tl, idx)) {
            (tl, idx)
        } else {
            match f
                .dialogue
                .iter()
                .find(|d| d.timeline % 10_000 == 3568 && d.index == idx)
            {
                Some(d) => (d.timeline, d.index),
                None => continue,
            }
        };
        let scores = pipeline_scores(f, tl, idx, &slots, &p, false, true);
        let code = &scores[..n_code];
        let top1 = code.iter().fold(0.0f32, |a, &b| a.max(b));
        let mass: f32 = code.iter().filter(|&&s| s > 0.0).sum();
        println!(
            "  {name:<34} {:>8} {top1:>10.2} {mass:>12.2}",
            f.sig(tl, idx).len()
        );
    }
    println!(
        "  (the residual closes if the tool-question rows sit far below the \
         ModelBuilder row — Concept B's gate then mutes the code layer for them)"
    );
}

// ═══ Round 5 — locking the PRODUCTION chain configuration ═══════════════════
//
// The battery's wins used mean-based levels; production uses the EWMA
// hit-level cache with priors/floors and warms content levels by SELF-MATCH
// (`warm_ingest_normalization`). R5 evaluates the exact production scoring
// chain (`score_slots_fused` + `NormalizationCache`) across the candidate
// configurations, on the three headline targets, to pick the shipping config:
//   fusion:  full-gated (additive × gate)  vs  grouped-sum (per-group tallies)
//   levels:  self-match warm (production ingest warm)  vs  dialogue-traffic
//   floors:  A.4 size floors REPLACING prior/scope-floor  vs  stacked default

use candle_conversation::normalization::{ChildKey, NormConfig, NormalizationCache, ScopeKey};
use candle_conversation::provenance::{score_slots_fused, FusionMode};

fn r5(f: &Fixture) {
    println!("\n═══ R5: production-chain config sweep (fusion × levels × floors) ═══");
    let (slots, n_junk, n_builder) = r3_pool(f);
    let tokens: Vec<usize> = slots.iter().map(|s| s.windows[0].len()).collect();
    let n = slots.len();
    let n_code = n_junk + n_builder;

    let scan = |probe: &[WideQSig], grouped_sum: bool| -> Vec<f32> {
        if grouped_sum {
            gated(&group_scans(probe, &slots))
        } else {
            let mut windows: Vec<&[WideQSig]> = Vec::new();
            let mut slot_of: Vec<usize> = Vec::new();
            for (i, s) in slots.iter().enumerate() {
                for w in &s.windows {
                    windows.push(w);
                    slot_of.push(i);
                }
            }
            score_slots_fused(
                probe,
                &windows,
                &slot_of,
                n,
                &[],
                FusionMode::ContentGated { gate_group: 0 },
            )
        }
    };

    for grouped_sum in [false, true] {
        // Levels via the production cache. Self-match warm: each slot's own
        // trailing window is the probe, its raw score observed — the
        // `warm_ingest_normalization` analog. Dialogue warm: the captured
        // turns' seal observes.
        for self_warm in [true, false] {
            let scope = ScopeKey::turn_group(0, 0);
            let mut cache = NormalizationCache::new(NormConfig::default());
            if self_warm {
                for (i, s) in slots.iter().enumerate() {
                    let w = s.windows[0];
                    let probe = &w[w.len().saturating_sub(MAX_PROBE_TOKENS)..];
                    let raw = scan(probe, grouped_sum);
                    cache.observe(&scope, &[(ChildKey::named(slots[i].label.clone()), raw[i])]);
                }
            } else {
                let mut dialogue: Vec<(u64, u64)> = f.events.keys().copied().collect();
                dialogue.sort_unstable();
                for (tl, idx) in dialogue {
                    let raw = scan(&f.probe(tl, idx), grouped_sum);
                    let pairs: Vec<(ChildKey, f32)> = slots
                        .iter()
                        .zip(&raw)
                        .map(|(s, &v)| (ChildKey::named(s.label.clone()), v))
                        .collect();
                    cache.observe(&scope, &pairs);
                }
            }
            for replace_floors in [true, false] {
                let floors: Vec<f32> = tokens
                    .iter()
                    .map(|&t| 2.0 * (256.0 / t.max(1) as f32).clamp(1.0, 16.0))
                    .collect();
                let normalize = |raw: Vec<f32>| -> Vec<f32> {
                    let pairs: Vec<(ChildKey, f32)> = slots
                        .iter()
                        .zip(&raw)
                        .map(|(s, &v)| (ChildKey::named(s.label.clone()), v))
                        .collect();
                    let out = if replace_floors {
                        cache.normalize_with_floors(&scope, &pairs, &floors)
                    } else {
                        cache.normalize(&scope, &pairs)
                    };
                    out.into_iter().map(|(_, v)| v).collect()
                };
                let pipeline = |tl: u64, idx: u64| -> Vec<f32> {
                    let all = f.sig(tl, idx);
                    let q = normalize(scan(&all[..64.min(all.len())], grouped_sum));
                    let t = normalize(scan(&f.probe(tl, idx), grouped_sum));
                    q.iter().zip(&t).map(|(a, b)| a.max(*b)).collect()
                };
                let tour = pipeline(TOUR_TL, T_TOUR);
                let mb = pipeline(TOUR_TL, T_MODELBUILDER);
                let recall = pipeline(TOUR_TL, T_RECALL);
                let rank_of_prefix = |scores: &[f32], prefix: &str| -> usize {
                    let mut idxs: Vec<usize> = (0..n).collect();
                    idxs.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));
                    idxs.iter()
                        .position(|&i| slots[i].label.starts_with(prefix))
                        .map(|p| p + 1)
                        .unwrap_or(0)
                };
                let mass = |s: &[f32]| -> f32 {
                    let mut g: Vec<f32> = s[..n_code]
                        .iter()
                        .copied()
                        .filter(|v| *v > 0.0)
                        .map(|v| v.min(1000.0))
                        .collect();
                    g.sort_by(|a, b| b.total_cmp(a));
                    let sum: f32 = g.iter().sum();
                    if sum <= 0.0 {
                        return 0.0;
                    }
                    sum * (g.iter().take(3).sum::<f32>() / sum)
                };
                let (rm, rc) = (mass(&recall), mass(&mb));
                println!(
                    "  fusion={} levels={} floors={}  tour cluster #{:<3} builder #{:<3} \
                     mass recall/code {:.1}/{:.1} {}",
                    if grouped_sum { "grouped" } else { "fullgate" },
                    if self_warm { "self " } else { "dialog" },
                    if replace_floors {
                        "size-replace"
                    } else {
                        "stacked-dflt"
                    },
                    rank_of_prefix(&tour, "cluster:"),
                    rank_of_prefix(&mb, "builder.rs"),
                    rm,
                    rc,
                    if rm < 0.5 * rc { "✓" } else { "✗" },
                );
            }
        }
    }
}

fn main() {
    let f = load_fixture();
    let total_tokens: usize = f.sigs.values().map(Vec::len).sum();
    println!(
        "fixture: {} sig blobs ({} tokens), {} dialogue turns, {} candidates, targets: {:?}",
        f.sigs.len(),
        total_tokens,
        f.dialogue.len(),
        f.candidates.len(),
        f.targets
            .iter()
            .map(|(k, v)| format!("{k}={}", v.len()))
            .collect::<Vec<_>>()
    );
    let round = std::env::args().nth(1);
    let run = |r: &str| round.as_deref().is_none_or(|v| v == r);
    if run("1") || run("2") {
        e1(&f);
        let level = e2(&f);
        e3(&f, &level);
        e4(&f);
        e5(&f);
        e6(&f);
        e7(&f);
        if run("2") {
            r2a(&f, &level);
            r2b(&f);
            r2c(&f);
            r2d(&f);
        }
    }
    if run("3") {
        r3a(&f);
        let (level, floor_base, floor_cap) = r3b(&f);
        let (tour, mb) = r3c(&f, &level, floor_base, floor_cap);
        r3d(&f, &tour, &mb, &level, floor_base, floor_cap);
        r3e(&f, &level, floor_base, floor_cap);
    }
    if run("4") {
        let (slots, _, _) = r3_pool(&f);
        let level = gated_levels(&f, &slots);
        r4a(&f, &level, 0.5, 1.0);
        r4b(&f, &level, 0.5, 1.0);
        r4c(&f, &level, 0.5, 1.0);
    }
    if run("5") {
        r5(&f);
    }
}
