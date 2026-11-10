//! What normalization does to the files that win everything, and which
//! formula gives every file an equal chance while keeping the real signal.
//!
//! Every calibration question probes the gallery of the files' own turns, and
//! each file's scores over the questions that are not its own are its
//! background ([`crate::background`]). A formula is judged on two things:
//!
//! - **Signal**: each question finds its own file (R@1, R@10, MRR), the
//!   hubs' own questions included.
//! - **Equal chance**: on the questions that are not theirs, the files' wins
//!   are spread evenly — the Gini of the wins, and the share the biggest
//!   winner takes.
//!
//! The formulas: raw; production's, taught with the same questions as
//! traffic, each question's own file its answer; and background-relative
//! ones — a z-score, a zero-inflated log-normal tail, the empirical tail, and
//! the empirical tail blended with the score's size.

use std::cmp::Reverse;

use candle_conversation::normalization::{ChildKey, NormalizationCache, ScopeKey};

use crate::background::{learn, question_probes, scaled_scores, T_REF};
use crate::corpus::File;
use crate::ranking::{rank_of, wins};
use crate::remote_gallery::{Daemon, Questions, RemoteGallery};
use crate::stats::{gini, mrr, recall};

/// One way of turning a probe's scores into the ones files are ranked by.
#[derive(Clone, Copy, PartialEq)]
enum Norm {
    Raw,
    Production,
    ZScore,
    LogNormalTail,
    /// The empirical tail plus `weight × ln(1 + score)`.
    Empirical(f32),
}

const NORMS: [(Norm, &str); 8] = [
    (Norm::Raw, "raw"),
    (Norm::Production, "production (background tail)"),
    (Norm::ZScore, "z-score"),
    (Norm::LogNormalTail, "log-normal tail surprise"),
    (Norm::Empirical(0.0), "empirical tail surprise"),
    (Norm::Empirical(0.25), "empirical + 0.25 ln(1+x)"),
    (Norm::Empirical(0.5), "empirical + 0.5 ln(1+x)"),
    (Norm::Empirical(1.0), "empirical + 1.0 ln(1+x)"),
];

pub fn run(files: &[File], targets: &[String], daemon: &Daemon) -> anyhow::Result<()> {
    let gpu = RemoteGallery::new(files, daemon);
    let probes = question_probes(files);
    println!(
        "normalization probe: {} questions over {} files' own turns",
        probes.len(),
        files.len()
    );
    let n = files.len();
    let scaled = scaled_scores(&gpu, &probes, Questions::Without, n)?;
    let backgrounds = learn(&probes, &scaled, n);

    // Production's cache, taught with every question as one probe, its own
    // file the answer left out of that file's background — as `learn` leaves a
    // file's own questions out. Fed reference-scaled scores at the reference
    // length, the same values.
    let t_ref = T_REF as usize;
    let scope = ScopeKey::pool(0);
    let child: Vec<ChildKey> = files
        .iter()
        .map(|f| ChildKey::timeline(f.timeline))
        .collect();
    let pairs = |scores: &[f32]| -> Vec<(ChildKey, f32)> {
        child.iter().cloned().zip(scores.iter().copied()).collect()
    };
    let cache = NormalizationCache::default();
    for (p, ((owner, _), scores)) in probes.iter().zip(&scaled).enumerate() {
        cache.observe(
            &scope,
            p as u64,
            &pairs(scores),
            Some(&child[*owner]),
            t_ref,
        );
    }
    let normed = |norm: Norm, p: usize| -> Vec<f32> {
        let x = &scaled[p];
        match norm {
            Norm::Raw => x.clone(),
            Norm::Production => cache
                .normalize(&scope, &pairs(x), t_ref)
                .into_iter()
                .map(|(_, s)| s)
                .collect(),
            Norm::ZScore => (0..n).map(|f| backgrounds[f].z(x[f])).collect(),
            Norm::LogNormalTail => (0..n)
                .map(|f| backgrounds[f].log_normal_surprise(x[f]))
                .collect(),
            Norm::Empirical(w) => (0..n).map(|f| backgrounds[f].blended(x[f], w)).collect(),
        }
    };

    // The hubs: the files that win the most other files' questions, raw.
    let owners: Vec<usize> = probes.iter().map(|(o, _)| *o).collect();
    let raw_wins = wins(&owners, &scaled, n);
    let mut hubs: Vec<usize> = (0..n).collect();
    hubs.sort_by_key(|&f| Reverse(raw_wins[f]));
    let mut watched: Vec<usize> = hubs.iter().take(5).copied().collect();
    for t in targets {
        watched.extend((0..n).filter(|&f| {
            !files[f].questions.is_empty()
                && files[f]
                    .path
                    .as_deref()
                    .is_some_and(|p| p.contains(t.as_str()))
        }));
    }
    watched.dedup();

    println!(
        "\n{:<32} {:>6} {:>6} {:>6} {:>6} {:>9} {:>8}",
        "formula", "R@1", "R@10", "MRR", "hubs", "win Gini", "top win"
    );
    let mut per_file: Vec<Vec<f32>> = Vec::new();
    for &(norm, name) in &NORMS {
        let scored: Vec<Vec<f32>> = (0..probes.len()).map(|p| normed(norm, p)).collect();
        let ranks: Vec<usize> = probes
            .iter()
            .zip(&scored)
            .map(|((owner, _), s)| rank_of(s, *owner))
            .collect();
        let own_mrr: Vec<f32> = watched
            .iter()
            .map(|&f| {
                let r: Vec<usize> = probes
                    .iter()
                    .zip(&ranks)
                    .filter(|((owner, _), _)| *owner == f)
                    .map(|(_, &r)| r)
                    .collect();
                mrr(&r)
            })
            .collect();
        let hub_mrr = own_mrr.iter().take(5).sum::<f32>() / 5.0;
        let w = wins(&owners, &scored, n);
        let total: usize = w.iter().sum();
        println!(
            "{name:<32} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>9.3} {:>7.1}%",
            recall(&ranks, 1),
            recall(&ranks, 10),
            mrr(&ranks),
            hub_mrr,
            gini(&w),
            100.0 * *w.iter().max().unwrap_or(&0) as f32 / total.max(1) as f32
        );
        per_file.push(own_mrr);
    }

    println!("\nown-question MRR per watched file, by formula in the order above");
    for (i, &f) in watched.iter().enumerate() {
        print!("{:<56}", name_of(&files[f]));
        for own in &per_file {
            print!(" {:>6.3}", own[i]);
        }
        println!();
    }
    Ok(())
}

fn name_of(f: &File) -> String {
    let Some(path) = f.path.as_deref() else {
        return format!("<no path> {:016x}", f.timeline);
    };
    let start = path.len().saturating_sub(54);
    path[start..].to_string()
}
