//! The scorer's free settings swept over gate 3's probes: per-fold-group vote
//! weights, the needle gate's keep fraction, and the file aggregate. One scan
//! a probe ([`group_votes`]); every setting is then weighed, gated and
//! aggregated on the host, so a sweep costs one gate-3 run.

use std::collections::BTreeMap;
use std::time::Instant;

use candle_conversation::provenance::boundaries::BoundaryRule;
use candle_conversation::provenance::window_score::{
    file_scores, gate, group_votes, window_tally, FileAggregate,
};
use candle_conversation::substrate::Substrate;

use crate::corpus::{ByLayer, File, Gallery};
use crate::ranking::{order, token_files};
use crate::retrieval::probes;
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

pub fn run(
    sub: &Substrate,
    files: &[File],
    gallery: &Gallery,
    by_layer: &mut ByLayer,
    rule: &BoundaryRule,
    per_source: usize,
) {
    let probes = probes(sub, files, by_layer, rule, per_source);
    println!("probes: {}", probes.len());
    let t = Instant::now();
    // (weights, keep, fold) → source → ranks.
    let mut ranks: BTreeMap<(usize, usize, usize), BTreeMap<&str, Vec<usize>>> = BTreeMap::new();
    for (source, target, probe, left_out) in &probes {
        if *source == "self (control)" {
            continue;
        }
        let token_file = token_files(gallery, *left_out);
        let votes = group_votes(probe, &gallery.flat, &token_file, files.len());
        let live: Vec<bool> = gallery
            .window_turn
            .iter()
            .map(|wt| Some(*wt) != *left_out)
            .collect();
        for (wi, w) in WEIGHTS.iter().enumerate() {
            for (ki, &keep) in KEEP.iter().enumerate() {
                let kept = gate(&votes, w, keep);
                let tally = window_tally(&kept, &gallery.windows, gallery.flat.len());
                for (fi, fold) in FOLDS.iter().enumerate() {
                    let scores = match fold {
                        Fold::Sum => {
                            let mut s = vec![0.0f32; files.len()];
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
                            files.len(),
                            *a,
                        ),
                    };
                    let rank = order(&scores)
                        .iter()
                        .position(|&f| f == *target)
                        .unwrap_or(files.len())
                        + 1;
                    ranks
                        .entry((wi, ki, fi))
                        .or_default()
                        .entry(source)
                        .or_default()
                        .push(rank);
                }
            }
        }
    }
    println!(
        "swept {} settings over {} probes in {:.1}s\n",
        WEIGHTS.len() * KEEP.len() * FOLDS.len(),
        probes.len(),
        t.elapsed().as_secs_f64()
    );
    let sources: Vec<&str> = ranks
        .values()
        .next()
        .map(|m| m.keys().copied().collect())
        .unwrap_or_default();
    let mut rows: Vec<(f32, String)> = ranks
        .iter()
        .map(|(&(wi, ki, fi), by_source)| {
            let per: Vec<f32> = sources.iter().map(|s| mrr(&by_source[s])).collect();
            let mean = per.iter().sum::<f32>() / per.len().max(1) as f32;
            let cells: Vec<String> = per.iter().map(|m| format!("{m:.3}")).collect();
            (
                mean,
                format!(
                    "{mean:.3}  {:<17} keep {:<4} {:<26} {}",
                    format!("{:?}", WEIGHTS[wi]),
                    KEEP[ki],
                    FOLDS[fi].name(),
                    cells.join("  ")
                ),
            )
        })
        .collect();
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!(
        "mean   weights           keep      fold                       MRR per source: {}",
        sources.join(" / ")
    );
    for (_, row) in rows.iter().take(25) {
        println!("{row}");
    }
    if let Some((_, base)) = rows.iter().find(|(_, r)| {
        r.contains("[1.0, 1.0, 1.0]") && r.contains("keep 0.25") && r.contains(" sum ")
    }) {
        println!("\ntoday: {base}");
    }
}
