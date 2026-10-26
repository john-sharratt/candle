//! The calibration competition's cut as production learns it, judged.
//!
//! Production learns the cut from each file's first `LEARN` calibration
//! questions, asked of every other calibration question with the asking ones
//! held out (`Conversation::learn_chunk_exclusion`), over every calibrated
//! file — old calibrations of questions alone beside new ones of questions
//! and their thinking. Here question 0 of each file is never learned from nor
//! scanned in the gallery: it is the probe, asked of the content plus the
//! calibration competition (questions `1..`), with the competition cut at
//! several thresholds.

use candle_conversation::provenance::WideQSig;

use crate::anomaly;
use crate::attribution::{attributed, background_mass, Chunks, BATCH, KEEP};
use crate::corpus::File;
use crate::judge::{header, Judge};
use crate::remote_gallery::{Daemon, Questions, RemoteGallery};
use crate::sweep::summed;

/// Questions per file the cut is learned from, after question 0: production's
/// `EXCLUSION_QUESTIONS_PER_FILE`.
const LEARN: usize = 2;

/// The calibration competition's weight, production's.
const WEIGHT: f32 = 0.5;

/// Thresholds tried; `None` for no cut.
const KS: [Option<f32>; 5] = [None, Some(6.0), Some(5.0), Some(4.0), Some(3.0)];

/// The share of each file's chunks a cut leaves it.
const CUT_KEEP: f32 = 0.5;

/// `a + w·b` per file, per probe.
fn combined(a: &[Vec<f32>], b: &[Vec<f32>], w: f32) -> Vec<Vec<f32>> {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| p + w * q).collect())
        .collect()
}

pub fn run(files: &[File], daemon: &Daemon) -> anyhow::Result<()> {
    let gpu = RemoteGallery::new(files, daemon);
    let n = files.len();
    let asked: Vec<usize> = (0..n)
        .filter(|&fi| files[fi].questions.len() > LEARN + 1)
        .collect();
    let thought = asked
        .iter()
        .filter(|&&fi| !files[fi].questions[0].answer.is_empty())
        .count();
    println!(
        "calibration cut: {} files asked ({thought} calibrated with thinking)",
        asked.len()
    );

    // Learned as production learns it: questions 1..=LEARN ask; the gallery
    // is every other calibration question, question 0 held out too.
    let learning = |_: usize, qi: usize| qi <= LEARN;
    let chunks = Chunks::of(files, false, &|_, qi| !learning(0, qi));
    let background: Vec<(usize, &[WideQSig])> = asked
        .iter()
        .flat_map(|&fi| {
            files[fi].questions[1..=LEARN]
                .iter()
                .map(move |q| (fi, q.sigs.as_slice()))
        })
        .collect();
    let bg = attributed(
        &gpu,
        &chunks,
        &background,
        false,
        Questions::With,
        &learning,
    )?;
    let mass = background_mass(&chunks, &background, &bg);
    println!(
        "competition chunks {}, background {} questions",
        mass.len(),
        background.len()
    );

    let held: Vec<(usize, &[WideQSig])> = asked
        .iter()
        .map(|&fi| (fi, files[fi].questions[0].sigs.as_slice()))
        .collect();
    let content = Judge {
        gpu: &gpu,
        held_out: &held,
        hubs: &[],
        n,
        questions: Questions::Without,
        masked_question: &|_, _| false,
    };
    let c = content.scores(&[])?;
    header("question 0 probes");
    content.print("content alone", &c);
    let scoring = |_: usize, qi: usize| qi == 0;
    for &k in &KS {
        let cut: Vec<(u64, usize, usize)> = match k {
            Some(k) => anomaly::excluded(&mass, &chunks.file_of, k, CUT_KEEP)
                .into_iter()
                .map(|c| chunks.range_of[c])
                .collect(),
            None => Vec::new(),
        };
        let q = competition(&gpu, &held, n, &scoring, &cut)?;
        let label = k.map_or("no cut".to_string(), |k| {
            format!("k {k} ({} cut)", cut.len())
        });
        content.print(
            &format!("  + {WEIGHT} x competition, {label}"),
            &combined(&c, &q, WEIGHT),
        );
    }
    Ok(())
}

/// `held` asked of the calibration competition alone — every file turn masked
/// — with `masked` questions left out and `excluded` cut.
fn competition(
    gpu: &RemoteGallery,
    held: &[(usize, &[WideQSig])],
    n: usize,
    masked: &dyn Fn(usize, usize) -> bool,
    excluded: &[(u64, usize, usize)],
) -> anyhow::Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(held.len());
    for batch in held.chunks(BATCH) {
        let sigs: Vec<&[WideQSig]> = batch.iter().map(|(_, s)| *s).collect();
        let votes = gpu.votes_without(&sigs, &|_, _| true, Questions::With, masked, excluded)?;
        out.extend(votes.iter().map(|v| summed(v, &[1.0, 1.0, 1.0], KEEP, n)));
    }
    Ok(out)
}
