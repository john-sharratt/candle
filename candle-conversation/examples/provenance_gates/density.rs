//! Question density: does each file's calibration compound with more questions?
//!
//! Every calibrated file's question 0 asks the gallery for its file; the
//! gallery holds every file's own turns and questions `1..=k`, for a rising
//! `k`. Question 0 of every file stays out at every `k`, so no probe meets
//! itself. Each added question is one more window voting for its file: if the
//! signals compound, recall climbs with `k`; if it flattens, more questions buy
//! nothing; if it falls, questions are hubs to question probes.
//!
//! A calibration turn is its question inside the chat framing every
//! calibration turn shares — the user opener, the turn's close, the empty
//! answer. Two cuts separate the question from its frame: the probe as the
//! whole turn or as its question alone (a live question probe is the turn's
//! user half), and the gallery's questions whole or cut to their words.

use std::cmp::Reverse;

use candle_conversation::persistence::content_hash::turn_stream_id;
use candle_conversation::provenance::WideQSig;

use crate::corpus::File;
use crate::ranking::{rank_of, wins};
use crate::remote_gallery::{Daemon, Questions, RemoteGallery};
use crate::stats::{gini, mrr, recall};
use crate::sweep::summed;

/// Probes per launch.
const BATCH: usize = 256;

/// The needle gate's keep fraction, today's.
const KEEP: f32 = 0.25;

/// Questions per file the gallery holds, beyond question 0.
const DENSITIES: [usize; 5] = [0, 1, 4, 8, 15];

/// How a probe question is cut.
#[derive(Clone, Copy, PartialEq)]
enum Probe {
    /// The whole calibration turn, framing and all.
    Turn,
    /// The question's own words.
    Body,
}

/// The rows: the probe's cut, and whether the gallery's questions are cut to
/// their words.
const ROWS: [(Probe, bool, &str); 3] = [
    (Probe::Turn, false, "turn vs turn"),
    (Probe::Body, false, "body vs turn"),
    (Probe::Body, true, "body vs body"),
];

pub fn run(files: &[File], daemon: &Daemon) -> anyhow::Result<()> {
    let gpu = RemoteGallery::new(files, daemon);
    let n = files.len();
    let asked: Vec<usize> = (0..n)
        .filter(|&fi| !files[fi].questions.is_empty())
        .collect();
    println!(
        "density: {} held-out questions; questions per calibrated file: max {}",
        asked.len(),
        files.iter().map(|f| f.questions.len()).max().unwrap_or(0)
    );
    println!(
        "\n{:<14} {:<10} {:>6} {:>6} {:>6} {:>9} {:>8}  most-won files (not the asker's)",
        "cut", "questions", "R@1", "R@10", "MRR", "win Gini", "top win"
    );
    for &(probe, cut, name) in &ROWS {
        let held: Vec<(usize, &[WideQSig])> = asked
            .iter()
            .map(|&fi| {
                let q = &files[fi].questions[0];
                let sigs = match probe {
                    Probe::Turn => q.sigs.as_slice(),
                    Probe::Body => q.body_sigs(),
                };
                (fi, sigs)
            })
            .collect();
        for &k in &DENSITIES {
            row(&gpu, files, &held, k, cut, name)?;
        }
    }
    Ok(())
}

/// One row: `held` asked of a gallery holding every file's questions `1..=k`,
/// cut to their words when `cut`.
fn row(
    gpu: &RemoteGallery,
    files: &[File],
    held: &[(usize, &[WideQSig])],
    k: usize,
    cut: bool,
    name: &str,
) -> anyhow::Result<()> {
    let n = files.len();
    let questions = if k == 0 {
        Questions::Without
    } else {
        Questions::With
    };
    // Each kept question's framing, cut from the gallery.
    let excluded: Vec<(u64, usize, usize)> = if cut && k > 0 {
        files
            .iter()
            .flat_map(|f| f.questions.iter().enumerate().skip(1).take(k))
            .flat_map(|(_, q)| {
                let sid = turn_stream_id(q.timeline, q.turn).0;
                [(sid, 0, q.body.start), (sid, q.body.end, q.sigs.len())]
            })
            .filter(|&(_, s, e)| s < e)
            .collect()
    } else {
        Vec::new()
    };
    let mut scores: Vec<Vec<f32>> = Vec::with_capacity(held.len());
    for batch in held.chunks(BATCH) {
        let sigs: Vec<&[WideQSig]> = batch.iter().map(|(_, s)| *s).collect();
        let votes = gpu.votes_without(
            &sigs,
            &|_, _| false,
            questions,
            &|_, qi| qi == 0 || qi > k,
            &excluded,
        )?;
        scores.extend(votes.iter().map(|v| summed(v, &[1.0, 1.0, 1.0], KEEP, n)));
    }
    let owners: Vec<usize> = held.iter().map(|(o, _)| *o).collect();
    let ranks: Vec<usize> = owners
        .iter()
        .zip(&scores)
        .map(|(o, s)| rank_of(s, *o))
        .collect();
    let w = wins(&owners, &scores, n);
    let total: usize = w.iter().sum();
    let mut top: Vec<usize> = (0..n).collect();
    top.sort_by_key(|&f| Reverse(w[f]));
    let named: Vec<String> = top
        .iter()
        .take(3)
        .map(|&f| {
            let path = files[f].path.as_deref().unwrap_or("<no path>");
            let short = path.rsplit('/').next().unwrap_or(path);
            format!("{short} {}", w[f])
        })
        .collect();
    println!(
        "{name:<14} {:<10} {:>6.3} {:>6.3} {:>6.3} {:>9.3} {:>7.1}%  {}",
        format!("1..={k}"),
        recall(&ranks, 1),
        recall(&ranks, 10),
        mrr(&ranks),
        gini(&w),
        100.0 * *w.iter().max().unwrap_or(&0) as f32 / total.max(1) as f32,
        named.join(", ")
    );
    Ok(())
}
