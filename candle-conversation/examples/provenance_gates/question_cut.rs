//! Anomalous-signal exclusion over a gallery that holds calibration questions.
//!
//! A question exemplar and a question probe come from the same distribution:
//! the generic words every question shares agree far more with other
//! questions than with any file's content, so against a background that is
//! mostly content they win outsized votes for whichever file's question
//! matched — the density gate's drop when the first question joins the
//! gallery. The exclusion that cuts boilerplate content cuts these too: their
//! background mass is the vote they win on other files' questions.
//!
//! Every calibrated file's questions `1..=GALLERY` stand in the gallery; its
//! questions past those are the background the mass is learned on; its
//! question 0, in neither, is asked. Content chunks and question chunks are
//! cut alike, or each alone, to see where the gain is.

use std::cmp::Reverse;

use candle_conversation::provenance::WideQSig;

use crate::anomaly;
use crate::attribution::{attributed, background_mass, Chunks};
use crate::corpus::File;
use crate::judge::{header, Judge};
use crate::ranking::wins;
use crate::remote_gallery::{Daemon, Questions, RemoteGallery};

/// The questions each file keeps in the gallery: `1..=GALLERY`.
const GALLERY: usize = 8;

/// Robust deviations above the median log mass a chunk is cut at; each file
/// keeps half its chunks.
const KS: [f32; 4] = [3.0, 4.0, 5.0, 6.0];

/// The share of each file's chunks a cut leaves it.
const KEEP_SHARE: f32 = 0.5;

pub fn run(files: &[File], daemon: &Daemon) -> anyhow::Result<()> {
    let gpu = RemoteGallery::new(files, daemon);
    let n = files.len();
    let in_gallery = |qi: usize| (1..=GALLERY).contains(&qi);
    let masked = |_: usize, qi: usize| !in_gallery(qi);
    let chunks = Chunks::of(files, true, &|_, qi| in_gallery(qi));
    let mut background: Vec<(usize, &[WideQSig])> = Vec::new();
    let mut held_out: Vec<(usize, &[WideQSig])> = Vec::new();
    for (fi, f) in files.iter().enumerate() {
        for (qi, q) in f.questions.iter().enumerate() {
            if qi == 0 {
                held_out.push((fi, q.sigs.as_slice()));
            } else if qi > GALLERY {
                background.push((fi, q.sigs.as_slice()));
            }
        }
    }
    let question_chunks = chunks.turn_of.iter().filter(|t| t.is_none()).count();
    println!(
        "question cut: {} chunks ({question_chunks} of questions 1..={GALLERY}); \
         background {} questions, judged on {}",
        chunks.file_of.len(),
        background.len(),
        held_out.len()
    );

    let bg = attributed(&gpu, &chunks, &background, true, Questions::With, &masked)?;
    let mass = background_mass(&chunks, &background, &bg);
    let won_by_questions: f32 = (0..mass.len())
        .filter(|&c| chunks.turn_of[c].is_none())
        .map(|c| mass[c])
        .sum();
    println!(
        "background mass won at question chunks: {:.1}% of {:.0}",
        100.0 * won_by_questions / mass.iter().sum::<f32>().max(1e-6),
        mass.iter().sum::<f32>()
    );

    // The hubs: the files that win the most held-out questions not theirs,
    // with the questions in and nothing cut.
    let with_questions = Judge {
        gpu: &gpu,
        held_out: &held_out,
        hubs: &[],
        n,
        questions: Questions::With,
        masked_question: &masked,
    };
    let raw = with_questions.scores(&[])?;
    let owners: Vec<usize> = held_out.iter().map(|(o, _)| *o).collect();
    let w = wins(&owners, &raw, n);
    let mut hubs: Vec<usize> = (0..n).collect();
    hubs.sort_by_key(|&f| Reverse(w[f]));
    hubs.truncate(10);
    let judge = Judge {
        hubs: &hubs,
        ..with_questions
    };
    let without = Judge {
        questions: Questions::Without,
        ..judge
    };

    header("gallery and cut");
    without.row("no questions, no cut", &[])?;
    judge.print(&format!("questions 1..={GALLERY}, no cut"), &raw);
    let ranges = |pick: &dyn Fn(usize) -> bool, cut: &[usize]| -> Vec<(u64, usize, usize)> {
        cut.iter()
            .filter(|&&c| pick(c))
            .map(|&c| chunks.range_of[c])
            .collect()
    };
    for &k in &KS {
        let cut = anomaly::excluded(&mass, &chunks.file_of, k, KEEP_SHARE);
        let is_question = |c: usize| chunks.turn_of[c].is_none();
        let q_cut = cut.iter().filter(|&&c| is_question(c)).count();
        judge.row(
            &format!("k {k}: all ({}, {q_cut} q)", cut.len()),
            &ranges(&|_| true, &cut),
        )?;
        judge.row(
            &format!("k {k}: questions only ({q_cut})"),
            &ranges(&is_question, &cut),
        )?;
        judge.row(
            &format!("k {k}: content only ({})", cut.len() - q_cut),
            &ranges(&|c| !is_question(c), &cut),
        )?;
    }
    Ok(())
}
