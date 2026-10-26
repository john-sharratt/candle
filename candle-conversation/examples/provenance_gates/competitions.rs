//! Calibration questions in a competition of their own.
//!
//! Scored in one pool with the files' content, a question probe's votes go
//! almost all to other questions — 96.5% of the background's vote mass is won
//! at question chunks, 2% of the gallery — because question and probe share a
//! distribution and `z` is taken over a background that is mostly content. In
//! a competition of their own, a question's match is measured against other
//! questions: each competition's votes have their own `z` and margin, and a
//! file scores by the two combined.
//!
//! Each calibrated file's question 0 is asked of its content alone, of every
//! file's questions `1..=k` alone, and of both in one pool (today's); the two
//! competitions are combined at several weights, raw and each normalized to
//! its probe's best file.

use candle_conversation::provenance::WideQSig;

use crate::anomaly;
use crate::attribution::{attributed, background_mass, Chunks, BATCH, KEEP};
use crate::corpus::File;
use crate::judge::{header, Judge};
use crate::remote_gallery::{Daemon, Questions, RemoteGallery};
use crate::sweep::summed;

/// Questions per file the question competition holds, beyond question 0.
const DENSITIES: [usize; 3] = [4, 8, 15];

/// Weights of the question competition against the content's.
const WEIGHTS: [f32; 5] = [0.25, 0.5, 1.0, 2.0, 4.0];

/// `a + w·b` per file, per probe.
fn combined(a: &[Vec<f32>], b: &[Vec<f32>], w: f32) -> Vec<Vec<f32>> {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| p + w * q).collect())
        .collect()
}

/// Each probe's scores over its best file's.
fn normalized(s: &[Vec<f32>]) -> Vec<Vec<f32>> {
    s.iter()
        .map(|row| {
            let top = row.iter().copied().fold(0.0f32, f32::max);
            if top > 0.0 {
                row.iter().map(|x| x / top).collect()
            } else {
                row.clone()
            }
        })
        .collect()
}

pub fn run(files: &[File], daemon: &Daemon) -> anyhow::Result<()> {
    let gpu = RemoteGallery::new(files, daemon);
    let n = files.len();
    let held: Vec<(usize, &[WideQSig])> = files
        .iter()
        .enumerate()
        .filter(|(_, f)| !f.questions.is_empty())
        .map(|(fi, f)| (fi, f.questions[0].sigs.as_slice()))
        .collect();
    println!("competitions: {} held-out questions", held.len());
    let content = Judge {
        gpu: &gpu,
        held_out: &held,
        hubs: &[],
        n,
        questions: Questions::Without,
        masked_question: &|_, _| false,
    };
    let c = content.scores(&[])?;
    let cn = normalized(&c);
    header("gallery");
    content.print("content alone", &c);
    for &k in &DENSITIES {
        let masked = move |_: usize, qi: usize| qi == 0 || qi > k;
        let pool = Judge {
            questions: Questions::With,
            masked_question: &masked,
            ..content
        };
        pool.print(&format!("one pool, questions 1..={k}"), &pool.scores(&[])?);
        let q = questions_alone(&gpu, &held, n, &masked)?;
        content.print(&format!("questions 1..={k} alone"), &q);
        let qn = normalized(&q);
        for &w in &WEIGHTS {
            content.print(
                &format!("  content + {w} x questions"),
                &combined(&c, &q, w),
            );
        }
        for &w in &WEIGHTS {
            content.print(&format!("  normalized, + {w} x"), &combined(&cn, &qn, w));
        }
    }

    // Both: the content competition with its anomalous chunks cut
    // (`chunk_factor`), and the questions' competition beside it.
    let chunks = Chunks::of(files, true, &|_, _| false);
    let background: Vec<(usize, &[WideQSig])> = files
        .iter()
        .enumerate()
        .flat_map(|(fi, f)| {
            f.questions
                .iter()
                .skip(1)
                .map(move |q| (fi, q.sigs.as_slice()))
        })
        .collect();
    let bg = attributed(
        &gpu,
        &chunks,
        &background,
        true,
        Questions::Without,
        &|_, _| false,
    )?;
    let mass = background_mass(&chunks, &background, &bg);
    let cut = anomaly::excluded(&mass, &chunks.file_of, CUT_K, CUT_KEEP);
    let ranges: Vec<(u64, usize, usize)> = cut.iter().map(|&c| chunks.range_of[c]).collect();
    let c_cut = content.scores(&ranges)?;
    let k = *DENSITIES.last().unwrap_or(&0);
    let masked = move |_: usize, qi: usize| qi == 0 || qi > k;
    let q = questions_alone(&gpu, &held, n, &masked)?;
    content.print(&format!("content, {} chunks cut", cut.len()), &c_cut);
    for &w in &WEIGHTS {
        content.print(
            &format!("  cut + {w} x questions 1..={k}"),
            &combined(&c_cut, &q, w),
        );
    }
    Ok(())
}

/// The anomalous-chunk cut stacked on the competitions: robust deviations
/// above the median log mass, and the share of each file's chunks it keeps.
const CUT_K: f32 = 5.0;
const CUT_KEEP: f32 = 0.5;

/// `held` asked of every file's calibration questions alone — every turn of
/// every file masked — those `masked` names left out too.
fn questions_alone(
    gpu: &RemoteGallery,
    held: &[(usize, &[WideQSig])],
    n: usize,
    masked: &dyn Fn(usize, usize) -> bool,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(held.len());
    for batch in held.chunks(BATCH) {
        let sigs: Vec<&[WideQSig]> = batch.iter().map(|(_, s)| *s).collect();
        let votes = gpu.votes(&sigs, &|_, _| true, Questions::With, masked)?;
        out.extend(votes.iter().map(|v| summed(v, &[1.0, 1.0, 1.0], KEEP, n)));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn competitions_combine_per_file_raw_or_over_the_best() {
        let a = vec![vec![2.0, 4.0]];
        let b = vec![vec![1.0, 0.0]];
        assert_eq!(combined(&a, &b, 2.0), vec![vec![4.0, 4.0]]);
        assert_eq!(normalized(&a), vec![vec![0.5, 1.0]]);
        assert_eq!(normalized(&[vec![0.0, 0.0]]), vec![vec![0.0, 0.0]]);
    }
}
