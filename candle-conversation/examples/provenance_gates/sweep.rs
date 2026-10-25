//! The scorer's free settings swept over gate 3's probes: which turns the
//! gallery holds, per-fold-group vote weights, the needle gate's keep fraction,
//! and the file aggregate. One scan a probe and gallery ([`group_votes`]);
//! every other setting is weighed, gated and aggregated on the host, so a
//! sweep costs about two gate-3 runs.

use std::collections::BTreeMap;
use std::time::Instant;

use candle_conversation::provenance::boundaries::BoundaryRule;
use candle_conversation::provenance::window_score::{
    file_scores, gate, group_votes, window_tally, FileAggregate,
};
use candle_conversation::substrate::Substrate;

use crate::corpus::{ByLayer, File, Gallery};
use crate::ranking::{order, token_files};
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

/// Which turns of each file the gallery holds.
const GALLERIES: [&str; 2] = ["all", "no-request"];

/// How a setting's gated votes become file scores: summed per file (today), or
/// tallied per window under an aggregate.
#[derive(Clone, Copy)]
enum Fold {
    Sum,
    Windows(FileAggregate),
}

impl Fold {
    fn name(self) -> String {
        match self {
            Fold::Sum => "sum".into(),
            Fold::Windows(a) => a.name(),
        }
    }
}

const FOLDS: [Fold; 3] = [
    Fold::Sum,
    Fold::Windows(FileAggregate::CountNormalised { strength: 1.0 }),
    Fold::Windows(FileAggregate::IdentityOrCount { strength: 1.0 }),
];

/// `(gallery, weights, keep, fold)` → source → ranks.
type Ranks<'a> = BTreeMap<(usize, usize, usize, usize), BTreeMap<&'a str, Vec<usize>>>;

pub fn run(
    sub: &Substrate,
    files: &[File],
    gallery: &Gallery,
    by_layer: &mut ByLayer,
    rule: &BoundaryRule,
    per_source: usize,
    only: Option<&str>,
) {
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
    let t = Instant::now();
    let mut ranks = Ranks::new();
    for probe in probes.iter().filter(|p| p.0 != "self (control)") {
        for gi in 0..GALLERIES.len() {
            let masked = |f: usize, t: usize| gi == 1 && first_turn[f] == Some(t);
            score_settings(probe, gallery, files.len(), gi, &masked, &mut ranks);
        }
    }
    println!(
        "swept {} settings over {} probes in {:.1}s\n",
        GALLERIES.len() * WEIGHTS.len() * KEEP.len() * FOLDS.len(),
        probes.len(),
        t.elapsed().as_secs_f64()
    );
    report(&ranks);
}

/// Every setting's rank of `probe`'s target over the gallery without the turns
/// `masked` names.
fn score_settings<'a>(
    probe: &'a Probe,
    gallery: &Gallery,
    n_files: usize,
    gi: usize,
    masked: &dyn Fn(usize, usize) -> bool,
    ranks: &mut Ranks<'a>,
) {
    let (source, target, sigs, left_out) = probe;
    let mut token_file = token_files(gallery, *left_out);
    for (f, t, r) in &gallery.turns {
        if masked(*f, *t) {
            token_file[r.clone()].fill(usize::MAX);
        }
    }
    let live: Vec<bool> = gallery
        .window_turn
        .iter()
        .map(|&(f, t)| Some((f, t)) != *left_out && !masked(f, t))
        .collect();
    let votes = group_votes(sigs, &gallery.flat, &token_file, n_files);
    for (wi, w) in WEIGHTS.iter().enumerate() {
        for (ki, &keep) in KEEP.iter().enumerate() {
            let kept = gate(&votes, w, keep);
            let tally = window_tally(&kept, &gallery.windows, gallery.flat.len());
            for (fi, fold) in FOLDS.iter().enumerate() {
                let scores = match fold {
                    Fold::Sum => {
                        let mut s = vec![0.0f32; n_files];
                        for v in &kept {
                            s[v.file] += v.weight;
                        }
                        s
                    }
                    Fold::Windows(a) => file_scores(
                        &tally,
                        &gallery.window_file,
                        &gallery.identity,
                        &live,
                        n_files,
                        *a,
                    ),
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

/// The best settings by mean MRR over the sources, and today's.
fn report(ranks: &Ranks<'_>) {
    let sources: Vec<&str> = ranks
        .values()
        .next()
        .map(|m| m.keys().copied().collect())
        .unwrap_or_default();
    let mut rows: Vec<(f32, bool, String)> = ranks
        .iter()
        .map(|(&(gi, wi, ki, fi), by_source)| {
            let per: Vec<f32> = sources.iter().map(|s| mrr(&by_source[s])).collect();
            let mean = per.iter().sum::<f32>() / per.len().max(1) as f32;
            let cells: Vec<String> = per.iter().map(|m| format!("{m:.3}")).collect();
            let today = gi == 0 && wi == 0 && KEEP[ki] == 0.25 && fi == 0;
            let row = format!(
                "{mean:.3}  {:<10} {:<17} keep {:<4} {:<26} {}",
                GALLERIES[gi],
                format!("{:?}", WEIGHTS[wi]),
                KEEP[ki],
                FOLDS[fi].name(),
                cells.join("  ")
            );
            (mean, today, row)
        })
        .collect();
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!(
        "mean   gallery    weights           keep      fold                       \
         MRR per source: {}",
        sources.join(" / ")
    );
    for (_, _, row) in rows.iter().take(25) {
        println!("{row}");
    }
    if let Some((_, _, row)) = rows.iter().find(|r| r.1) {
        println!("\ntoday: {row}");
    }
}
