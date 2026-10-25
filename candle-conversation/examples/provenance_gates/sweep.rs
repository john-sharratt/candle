//! The scorer's free settings swept over gate 3's probes: which turns the
//! gallery holds, per-fold-group vote weights, the needle gate's keep fraction,
//! and the fold from votes to file scores. One GPU scan a probe and gallery
//! ([`GpuGallery::votes`]); every other setting is weighed, gated and folded on
//! the host.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use candle_conversation::provenance::boundaries::BoundaryRule;
use candle_conversation::provenance::hubness::{background, corrected, peak, HubCorrection};
use candle_conversation::provenance::window_score::{gate, Vote};
use candle_conversation::provenance::{WideQSig, FOLD_GROUPS};
use candle_conversation::substrate::Substrate;

use crate::corpus::{gather, ByLayer, File};
use crate::gpu_gallery::{GpuGallery, Questions};
use crate::ranking::order;
use crate::retrieval::{probes, Probe};
use crate::stats::mrr;

const WEIGHTS: [[f32; 3]; 8] = [
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
    [0.25, 1.0, 1.0],
    [0.5, 1.0, 2.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
];
const KEEP: [f32; 4] = [0.1, 0.25, 0.5, 1.0];

/// Which turns of each file the gallery holds, and whether its calibration
/// questions stand beside them.
const GALLERIES: [&str; 3] = ["all", "no-request", "all+questions"];

/// Whether gallery `gi` holds the calibration questions.
fn questions_of(gi: usize) -> Questions {
    if gi == 2 {
        Questions::With
    } else {
        Questions::Without
    }
}

/// How a setting's gated votes become file scores: summed per file, or summed
/// and set against each file's background.
const FOLDS: [HubCorrection; 5] = [
    HubCorrection::None,
    HubCorrection::Less,
    HubCorrection::Over { floor: 0.001 },
    HubCorrection::OverPeak { floor: 1.0 },
    HubCorrection::OverPeak { floor: 100.0 },
];

/// The probe length a peak and an over-peak score are scaled to: the live
/// probe cap, as production's normalization scales them.
const T_REF: usize = 256;

/// `(gallery, weights, keep)` → each file's background mean share and peak.
type Backgrounds = HashMap<(usize, usize, usize), (Vec<f32>, Vec<f32>)>;

/// One probe's per-token, per-fold-group votes.
pub type TokenVotes = Vec<[Option<Vote>; FOLD_GROUPS]>;

/// Gated votes summed per file.
fn sum_scores(kept: &[Vote], n_files: usize) -> Vec<f32> {
    let mut s = vec![0.0f32; n_files];
    for v in kept {
        s[v.file] += v.weight;
    }
    s
}

/// `(gallery, weights, keep, fold)` → source → ranks.
type Ranks<'a> = BTreeMap<(usize, usize, usize, usize), BTreeMap<&'a str, Vec<usize>>>;

/// Sweep over `per_source` probes a source, each file's background learned
/// from `background_probes` windows of files' own pages — each scored with its
/// own file left out, so a file's background is what it wins of probes it is
/// not the answer to.
#[allow(clippy::too_many_arguments)]
pub fn run(
    sub: &Substrate,
    files: &[File],
    by_layer: &mut ByLayer,
    rule: &BoundaryRule,
    per_source: usize,
    background_probes: usize,
    only: Option<&str>,
) -> anyhow::Result<()> {
    let t = Instant::now();
    let gpu = GpuGallery::new(files)?;
    let probes: Vec<Probe> = probes(sub, files, by_layer, rule, per_source)
        .into_iter()
        .filter(|p| only.is_none_or(|s| p.0 == s))
        .collect();
    println!("probes: {}", probes.len());
    // Each file's first turn: its request and the tool-call JSON, alike in
    // every file.
    let first_turn: Vec<Option<usize>> = files
        .iter()
        .map(|f| (0..f.turns.len()).find(|&i| f.turns[i].covered()))
        .collect();
    let backgrounds = backgrounds(&gpu, files, rule, &first_turn, background_probes)?;
    println!(
        "{background_probes} background probes in {:.1}s",
        t.elapsed().as_secs_f64()
    );
    let mut ranks = Ranks::new();
    for probe in probes.iter().filter(|p| p.0 != "self (control)") {
        for gi in 0..GALLERIES.len() {
            let (_, _, sigs, left_out) = probe;
            let votes = gpu.votes(
                &[sigs.as_slice()],
                &|f, t| Some((f, t)) == *left_out || (gi == 1 && first_turn[f] == Some(t)),
                questions_of(gi),
                &|_, _| false,
            )?;
            score_settings(probe, &votes[0], files.len(), gi, &backgrounds, &mut ranks);
        }
    }
    println!(
        "swept {} settings over {} probes in {:.1}s\n",
        GALLERIES.len() * WEIGHTS.len() * KEEP.len() * FOLDS.len(),
        probes.len(),
        t.elapsed().as_secs_f64()
    );
    report(&ranks);
    Ok(())
}

/// Up to `n` background probes, spread evenly over `files`: the middle window
/// of each one's last covered turn, with the file it came from.
pub fn background_sources(
    files: &[File],
    rule: &BoundaryRule,
    n: usize,
) -> Vec<(usize, Vec<WideQSig>)> {
    let stride = files.len().div_ceil(n.max(1)).max(1);
    files
        .iter()
        .enumerate()
        .step_by(stride)
        .filter_map(|(fi, f)| {
            let turn = f.turns.iter().rev().find(|t| t.covered())?;
            let windows = turn.windows(rule);
            let w = windows.get(windows.len() / 2)?;
            Some((fi, gather(&turn.sigs, w)))
        })
        .collect()
}

/// Gated votes summed per file.
pub fn summed(votes: &TokenVotes, weights: &[f32], keep: f32, n_files: usize) -> Vec<f32> {
    sum_scores(&gate(votes, weights, keep), n_files)
}

/// Every file's background share and peak under every gallery, weights and
/// keep.
fn backgrounds(
    gpu: &GpuGallery,
    files: &[File],
    rule: &BoundaryRule,
    first_turn: &[Option<usize>],
    n_probes: usize,
) -> anyhow::Result<Backgrounds> {
    let sources = background_sources(files, rule, n_probes);
    let mut out = Backgrounds::new();
    for gi in 0..GALLERIES.len() {
        let mut votes: Vec<TokenVotes> = Vec::with_capacity(sources.len());
        for (fi, sigs) in &sources {
            let v = gpu.votes(
                &[sigs.as_slice()],
                &|f, t| f == *fi || (gi == 1 && first_turn[f] == Some(t)),
                questions_of(gi),
                &|f, _| f == *fi,
            )?;
            votes.extend(v);
        }
        for (wi, w) in WEIGHTS.iter().enumerate() {
            for (ki, &keep) in KEEP.iter().enumerate() {
                let per_probe: Vec<Vec<f32>> = votes
                    .iter()
                    .map(|v| sum_scores(&gate(v, w, keep), files.len()))
                    .collect();
                let with_len: Vec<(Vec<f32>, usize)> = per_probe
                    .iter()
                    .zip(&sources)
                    .map(|(s, (_, sigs))| (s.clone(), sigs.len()))
                    .collect();
                out.insert(
                    (gi, wi, ki),
                    (
                        background(&per_probe, files.len()),
                        peak(&with_len, files.len(), T_REF),
                    ),
                );
            }
        }
    }
    Ok(out)
}

/// Every setting's rank of `probe`'s target from its `votes` against gallery
/// `gi`.
fn score_settings<'a>(
    probe: &'a Probe,
    votes: &TokenVotes,
    n_files: usize,
    gi: usize,
    backgrounds: &Backgrounds,
    ranks: &mut Ranks<'a>,
) {
    let (source, target, sigs, _) = probe;
    for (wi, w) in WEIGHTS.iter().enumerate() {
        for (ki, &keep) in KEEP.iter().enumerate() {
            let summed = sum_scores(&gate(votes, w, keep), n_files);
            let (mean, peaked) = &backgrounds[&(gi, wi, ki)];
            for (fi, fold) in FOLDS.iter().enumerate() {
                let bg = match fold {
                    HubCorrection::OverPeak { .. } => peaked,
                    _ => mean,
                };
                let scores = match fold {
                    HubCorrection::None => summed.clone(),
                    h => corrected(&summed, bg, *h, sigs.len(), T_REF),
                };
                let rank = order(&scores)
                    .iter()
                    .position(|f| f == target)
                    .unwrap_or(n_files)
                    + 1;
                ranks
                    .entry((gi, wi, ki, fi))
                    .or_default()
                    .entry(source)
                    .or_default()
                    .push(rank);
            }
        }
    }
}

/// The best settings for each source by its MRR, and today's. Each source is
/// ranked alone: a gallery without the request turns lifts request probes and
/// takes summary probes' answers away, so a mean over sources ranks galleries
/// by an artefact.
fn report(ranks: &Ranks<'_>) {
    let sources: Vec<&str> = ranks
        .values()
        .next()
        .map(|m| m.keys().copied().collect())
        .unwrap_or_default();
    let rows: Vec<(Vec<f32>, bool, String)> = ranks
        .iter()
        .map(|(&(gi, wi, ki, fi), by_source)| {
            let per: Vec<f32> = sources.iter().map(|s| mrr(&by_source[s])).collect();
            let cells: Vec<String> = per.iter().map(|m| format!("{m:.3}")).collect();
            let today = gi == 0 && wi == 0 && KEEP[ki] == 0.25 && fi == 0;
            let row = format!(
                "{:<10} {:<17} keep {:<4} {:<26} {}",
                GALLERIES[gi],
                format!("{:?}", WEIGHTS[wi]),
                KEEP[ki],
                FOLDS[fi].name(),
                cells.join("  ")
            );
            (per, today, row)
        })
        .collect();
    let header = format!(
        "gallery    weights           keep      fold                       \
         MRR per source: {}",
        sources.join(" / ")
    );
    for (si, source) in sources.iter().enumerate() {
        let mut order: Vec<&(Vec<f32>, bool, String)> = rows.iter().collect();
        order.sort_by(|a, b| b.0[si].total_cmp(&a.0[si]));
        println!("best for {source}:\n{header}");
        for (_, _, row) in order.iter().take(15) {
            println!("{row}");
        }
        println!();
    }
    if let Some((_, _, row)) = rows.iter().find(|r| r.1) {
        println!("today: {row}");
    }
}
