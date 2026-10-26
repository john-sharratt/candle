//! A calibration factor per stored signature chunk, from its background.
//!
//! Every vote is won at one gallery token. A chunk of 32 tokens that wins
//! votes on questions about other files is generic — boilerplate, shared
//! phrasing — and a hub is a file with many such chunks. Each chunk's factor
//! comes from the vote mass it wins on questions that are not about its file;
//! a probe's vote is then weighed by the factor of the chunk it was won at,
//! so generic chunks count for less and distinctive ones keep their weight.
//!
//! Each vote is credited to the chunk it was won at ([`crate::attribution`]).
//!
//! A factor of 0 is also tried where the kernel would apply one: the most
//! generic chunks cut from the daemon's gallery, so their votes go to the
//! next-best agreement instead of shrinking.
//!
//! The background is questions 1.. of every calibrated file; the factors are
//! judged on each file's question 0, which the background never saw.

use std::cmp::Reverse;

use candle_conversation::provenance::WideQSig;
use tokenizers::Tokenizer;

use crate::attribution::{attributed, background_mass, Chunks, Won};
use crate::corpus::File;
use crate::judge::Judge;
use crate::ranking::{rank_of, wins};
use crate::remote_gallery::{Daemon, Questions, RemoteGallery};
use crate::stats::{gini, mrr, recall};
use crate::{anomaly, boilerplate};

/// A shape for the factor, from a chunk's background vote mass `w` over the
/// mean nonzero mass `w0`.
#[derive(Clone, Copy)]
enum Shape {
    None,
    /// `(1 + w / w0)^-alpha`.
    Power(f32),
    /// `1 / (1 + ln(1 + w / w0))`.
    LogIdf,
}

const SHAPES: [(Shape, &str); 6] = [
    (Shape::None, "raw (factor 1)"),
    (Shape::Power(0.5), "(1 + w/w0)^-0.5"),
    (Shape::Power(1.0), "(1 + w/w0)^-1"),
    (Shape::Power(1.5), "(1 + w/w0)^-1.5"),
    (Shape::Power(2.0), "(1 + w/w0)^-2"),
    (Shape::LogIdf, "1 / (1 + ln(1 + w/w0))"),
];

impl Shape {
    fn factor(self, w: f32, w0: f32) -> f32 {
        let r = w / w0.max(1e-6);
        match self {
            Shape::None => 1.0,
            Shape::Power(a) => (1.0 + r).powf(-a),
            Shape::LogIdf => 1.0 / (1.0 + r.ln_1p()),
        }
    }
}

/// How many of the most generic chunks are printed, with their text.
const SHOWN: usize = 25;

pub fn run(files: &[File], tokenizer: &Tokenizer, daemon: &Daemon) -> anyhow::Result<()> {
    let gpu = RemoteGallery::new(files, daemon);
    let chunks = Chunks::of(files, true, &|_, _| false);
    let n = files.len();
    let (mut background, mut held_out): (Vec<(usize, &[WideQSig])>, Vec<(usize, &[WideQSig])>) =
        (Vec::new(), Vec::new());
    for (fi, f) in files.iter().enumerate() {
        for (qi, q) in f.questions.iter().enumerate() {
            let p = (fi, q.sigs.as_slice());
            if qi == 0 {
                held_out.push(p);
            } else {
                background.push(p);
            }
        }
    }
    println!(
        "chunk factor: {} chunks over {n} files; background {} questions, judged on {}",
        chunks.file_of.len(),
        background.len(),
        held_out.len()
    );

    // Each chunk's vote mass on questions not about its file.
    let bg = attributed(
        &gpu,
        &chunks,
        &background,
        true,
        Questions::Without,
        &|_, _| false,
    )?;
    let mass = background_mass(&chunks, &background, &bg);
    let nonzero: Vec<f32> = mass.iter().copied().filter(|&m| m > 0.0).collect();
    let w0 = nonzero.iter().sum::<f32>() / nonzero.len().max(1) as f32;
    let mut sorted = nonzero.clone();
    sorted.sort_by(f32::total_cmp);
    let at = |q: f32| {
        sorted
            .get(((q * sorted.len() as f32) as usize).min(sorted.len().saturating_sub(1)))
            .copied()
            .unwrap_or(0.0)
    };
    println!(
        "background mass: {} of {} chunks won anything; mean {w0:.1}, p50 {:.1}, p90 {:.1}, p99 {:.1}, max {:.1}",
        nonzero.len(),
        mass.len(),
        at(0.5),
        at(0.9),
        at(0.99),
        at(1.0)
    );

    // What the most generic chunks say: the ones a cut takes first.
    let mut by_mass: Vec<usize> = (0..mass.len()).collect();
    by_mass.sort_by(|&a, &b| mass[b].total_cmp(&mass[a]));
    println!("\nthe most generic chunks");
    for &c in by_mass.iter().take(SHOWN) {
        let Some((fi, ti)) = chunks.turn_of[c] else {
            continue;
        };
        let (_, start, end) = chunks.range_of[c];
        let text = tokenizer
            .decode(&files[fi].turns[ti].tokens[start..end], false)
            .unwrap_or_default()
            .replace('\n', "⏎");
        println!(
            "{:>10.0}  {} turn {ti} [{start}..{end}]  {text}",
            mass[c],
            files[fi].path.as_deref().unwrap_or("<no path>")
        );
    }

    let judged = attributed(
        &gpu,
        &chunks,
        &held_out,
        true,
        Questions::Without,
        &|_, _| false,
    )?;
    let owners: Vec<usize> = held_out.iter().map(|(o, _)| *o).collect();
    // The hubs: the files that win the most held-out questions not theirs, raw.
    let raw_scores = scores(&judged, &mass, w0, Shape::None, n);
    let raw_wins = wins(&owners, &raw_scores, n);
    let mut hubs: Vec<usize> = (0..n).collect();
    hubs.sort_by_key(|&f| Reverse(raw_wins[f]));
    let hubs: Vec<usize> = hubs.into_iter().take(10).collect();

    println!(
        "\n{:<26} {:>6} {:>6} {:>6} {:>9} {:>8} {:>10}",
        "factor", "R@1", "R@10", "MRR", "win Gini", "top win", "hub wins"
    );
    for &(shape, name) in &SHAPES {
        let s = scores(&judged, &mass, w0, shape, n);
        let ranks: Vec<usize> = owners
            .iter()
            .zip(&s)
            .map(|(owner, sc)| rank_of(sc, *owner))
            .collect();
        let w = wins(&owners, &s, n);
        let total: usize = w.iter().sum();
        println!(
            "{name:<26} {:>6.3} {:>6.3} {:>6.3} {:>9.3} {:>7.1}% {:>10}",
            recall(&ranks, 1),
            recall(&ranks, 10),
            mrr(&ranks),
            gini(&w),
            100.0 * *w.iter().max().unwrap_or(&0) as f32 / total.max(1) as f32,
            hubs.iter().map(|&f| w[f]).sum::<usize>()
        );
    }

    // A factor of 0, applied where the kernel would apply one: the scan never
    // sees a cut span, so its votes go to the next-best agreement instead of
    // shrinking.
    let judge = Judge {
        gpu: &gpu,
        held_out: &held_out,
        hubs: &hubs,
        n,
        questions: Questions::Without,
        masked_question: &|_, _| false,
    };
    let chunk_ranges = |cut: &[usize]| -> Vec<(u64, usize, usize)> {
        cut.iter().map(|&c| chunks.range_of[c]).collect()
    };
    for &cut in &CUTS {
        let n_cut = (cut * mass.len() as f32) as usize;
        judge.row(
            &format!("cut top {:.2}% ({n_cut})", cut * 100.0),
            &chunk_ranges(&by_mass[..n_cut]),
        )?;
    }

    // IDF-like: a chunk cut by how many other files' questions it wins votes
    // on, whatever its mass.
    let df = file_frequency(&background, &bg, &chunks.file_of);
    for &k in &DF_CUTS {
        let cut: Vec<usize> = (0..df.len()).filter(|&c| df[c] >= k).collect();
        judge.row(&format!("df >= {k} ({})", cut.len()), &chunk_ranges(&cut))?;
    }

    // Anomalous-signal exclusion: background outliers in log space, each file
    // keeping its share.
    let (med, spread) = anomaly::log_spread(&mass);
    println!(
        "background ln(mass): median {med:.2}, robust spread {spread:.2} (mass {:.0} at k=1)",
        (med + spread).exp()
    );
    for &(k, keep) in &ANOMALY {
        let cut = anomaly::excluded(&mass, &chunks.file_of, k, keep);
        judge.row(
            &format!("k {k} keep {keep} ({})", cut.len()),
            &chunk_ranges(&cut),
        )?;
    }

    // Structural: runs many files share verbatim, with no scan behind them.
    for &share in &BOILERPLATE_SHARES {
        let min_files = ((share * files.len() as f32) as usize).max(2);
        let spans = boilerplate::ranges(files, min_files);
        let tokens: usize = spans.iter().map(|(_, s, e)| e - s).sum();
        judge.row(
            &format!("runs in >= {:.0}% ({}k tok)", share * 100.0, tokens / 1000),
            &spans,
        )?;
    }
    Ok(())
}

/// The shares of chunks, most generic first, cut from the gallery.
const CUTS: [f32; 4] = [0.0, 0.0002, 0.001, 0.005];

/// Distinct other files' questions a chunk must win votes on to be cut.
const DF_CUTS: [usize; 5] = [300, 150, 75, 40, 20];

/// `(k, keep)`: robust deviations above the median log mass, and the share of
/// each file's chunks it keeps.
const ANOMALY: [(f32, f32); 7] = [
    (3.0, 0.5),
    (4.0, 0.5),
    (5.0, 0.5),
    (6.0, 0.5),
    (7.0, 0.5),
    (4.0, 0.8),
    (4.0, 0.95),
];

/// Shares of files a run must occur in to be boilerplate.
const BOILERPLATE_SHARES: [f32; 5] = [0.5, 0.25, 0.1, 0.05, 0.02];

/// How many distinct files' background questions — not its own file's — each
/// chunk won a kept vote on.
fn file_frequency(
    background: &[(usize, &[WideQSig])],
    won: &[Vec<Won>],
    file_of: &[usize],
) -> Vec<usize> {
    let mut pairs: Vec<(usize, usize)> = background
        .iter()
        .zip(won)
        .flat_map(|((owner, _), won)| {
            won.iter()
                .filter(|v| file_of[v.chunk] != *owner)
                .map(move |v| (v.chunk, *owner))
        })
        .collect();
    pairs.sort_unstable();
    pairs.dedup();
    let mut df = vec![0; file_of.len()];
    for (chunk, _) in pairs {
        df[chunk] += 1;
    }
    df
}

/// Every probe's file scores: its kept votes, each weighed by its chunk's
/// factor.
fn scores(judged: &[Vec<Won>], mass: &[f32], w0: f32, shape: Shape, n: usize) -> Vec<Vec<f32>> {
    judged
        .iter()
        .map(|won| {
            let mut s = vec![0f32; n];
            for v in won {
                s[v.file] += v.weight * shape.factor(mass[v.chunk], w0);
            }
            s
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_factor_shape_is_one_at_no_mass_and_falls_with_it() {
        for (shape, _) in SHAPES {
            assert_eq!(shape.factor(0.0, 10.0), 1.0);
        }
        assert_eq!(Shape::Power(1.0).factor(10.0, 10.0), 0.5);
        assert_eq!(Shape::Power(2.0).factor(10.0, 10.0), 0.25);
        assert!((Shape::LogIdf.factor(10.0, 10.0) - 1.0 / (1.0 + 2.0f32.ln())).abs() < 1e-6);
    }
}
