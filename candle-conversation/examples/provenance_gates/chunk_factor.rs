//! A calibration factor per stored signature chunk, from its background.
//!
//! Every vote is won at one gallery token. A chunk of 32 tokens that wins
//! votes on questions about other files is generic — boilerplate, shared
//! phrasing — and a hub is a file with many such chunks. Each chunk's factor
//! comes from the vote mass it wins on questions that are not about its file;
//! a probe's vote is then weighed by the factor of the chunk it was won at,
//! so generic chunks count for less and distinctive ones keep their weight.
//!
//! Attribution: the daemon's scan names the winning file and the vote of each
//! probe token and fold group; the winning token is found on the host by the
//! kernel's own measure — sign agreement — over that file's tokens alone.
//!
//! A factor of 0 is also tried where the kernel would apply one: the most
//! generic chunks cut from the daemon's gallery, so their votes go to the
//! next-best agreement instead of shrinking.
//!
//! The background is questions 1.. of every calibrated file; the factors are
//! judged on each file's question 0, which the background never saw.

use std::cmp::Reverse;
use std::iter::repeat_n;

use candle_conversation::persistence::content_hash::turn_stream_id;
use candle_conversation::provenance::window_score::{gate, Vote};
use candle_conversation::provenance::{WideQSig, FOLD_GROUPS};
use rayon::prelude::*;
use tokenizers::Tokenizer;

use crate::corpus::{File, CHUNK};
use crate::ranking::{rank_of, wins};
use crate::remote_gallery::{Daemon, Questions, RemoteGallery};
use crate::stats::{gini, mrr, recall};
use crate::sweep::summed;

/// Probes per launch.
const BATCH: usize = 256;

/// Words per fold group of a token.
const GROUP_WORDS: usize = 8;

/// The needle gate's keep fraction, today's.
const KEEP: f32 = 0.25;

/// Every covered turn's signatures, per file, and each chunk's global id.
struct Chunks<'a> {
    /// `(file, signatures, first chunk id)` per covered turn, by file.
    turns: Vec<Vec<(&'a [WideQSig], usize)>>,
    /// The file each chunk belongs to.
    file_of: Vec<usize>,
    /// Each chunk's turn, by stream id, and its `[start, end)` tokens there.
    range_of: Vec<(u64, usize, usize)>,
    /// Each chunk's file and turn there.
    turn_of: Vec<(usize, usize)>,
}

impl<'a> Chunks<'a> {
    fn of(files: &'a [File]) -> Chunks<'a> {
        let mut file_of = Vec::new();
        let mut range_of = Vec::new();
        let mut turn_of = Vec::new();
        let turns = files
            .iter()
            .enumerate()
            .map(|(fi, f)| {
                f.turns
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| t.covered())
                    .map(|(ti, t)| {
                        let first = file_of.len();
                        let n = t.sigs.len();
                        let sid = turn_stream_id(f.timeline, ti as u32).0;
                        file_of.extend(repeat_n(fi, n.div_ceil(CHUNK)));
                        turn_of.extend(repeat_n((fi, ti), n.div_ceil(CHUNK)));
                        range_of.extend(
                            (0..n.div_ceil(CHUNK))
                                .map(|c| (sid, c * CHUNK, ((c + 1) * CHUNK).min(n))),
                        );
                        (t.sigs.as_slice(), first)
                    })
                    .collect()
            })
            .collect();
        Chunks {
            turns,
            file_of,
            range_of,
            turn_of,
        }
    }

    /// The chunk of `file` holding the token that agrees best with `probe`
    /// in fold group `group`.
    fn winner(&self, file: usize, probe: &WideQSig, group: usize) -> Option<usize> {
        let words = group * GROUP_WORDS..(group + 1) * GROUP_WORDS;
        let q = probe.words.get(words.clone())?;
        let mut best: Option<(u32, usize)> = None;
        for (sigs, first) in &self.turns[file] {
            for (i, s) in sigs.iter().enumerate() {
                let Some(g) = s.words.get(words.clone()) else {
                    continue;
                };
                let agree: u32 = q.iter().zip(g).map(|(a, b)| (!(a ^ b)).count_ones()).sum();
                if best.is_none_or(|(b, _)| agree > b) {
                    best = Some((agree, first + i / CHUNK));
                }
            }
        }
        best.map(|(_, c)| c)
    }
}

/// One kept vote: the file it went to, its weight, and the chunk it was won
/// at.
struct Won {
    file: usize,
    weight: f32,
    chunk: usize,
}

/// Every probe's kept votes, attributed to chunks.
fn attributed(
    gpu: &RemoteGallery,
    chunks: &Chunks,
    probes: &[(usize, &[WideQSig])],
    n_files: usize,
) -> anyhow::Result<Vec<Vec<Won>>> {
    let mut out: Vec<Vec<Won>> = Vec::with_capacity(probes.len());
    for batch in probes.chunks(BATCH) {
        let sigs: Vec<&[WideQSig]> = batch.iter().map(|(_, s)| *s).collect();
        let votes = gpu.votes(&sigs, &|_, _| false, Questions::Without, &|_, _| false)?;
        let kept: Vec<Vec<Vote>> = votes
            .iter()
            .map(|per_token| {
                // The token slot carries `probe token × groups + group` through
                // the gate, so each kept vote can be attributed.
                let tagged: Vec<[Option<Vote>; FOLD_GROUPS]> = per_token
                    .iter()
                    .enumerate()
                    .map(|(t, groups)| {
                        let mut g = *groups;
                        for (gi, v) in g.iter_mut().enumerate() {
                            if let Some(v) = v {
                                v.token = t * FOLD_GROUPS + gi;
                            }
                        }
                        g
                    })
                    .collect();
                gate(&tagged, &[], KEEP)
            })
            .collect();
        let batch_won: Vec<Vec<Won>> = batch
            .par_iter()
            .zip(kept.par_iter())
            .map(|((_, sigs), kept)| {
                kept.iter()
                    .filter(|v| v.file < n_files)
                    .filter_map(|v| {
                        let (t, g) = (v.token / FOLD_GROUPS, v.token % FOLD_GROUPS);
                        let chunk = chunks.winner(v.file, &sigs[t], g)?;
                        Some(Won {
                            file: v.file,
                            weight: v.weight,
                            chunk,
                        })
                    })
                    .collect()
            })
            .collect();
        out.extend(batch_won);
    }
    Ok(out)
}

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
    let chunks = Chunks::of(files);
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
    let bg = attributed(&gpu, &chunks, &background, n)?;
    let mut mass = vec![0f32; chunks.file_of.len()];
    for ((owner, _), won) in background.iter().zip(&bg) {
        for v in won {
            if chunks.file_of[v.chunk] != *owner {
                mass[v.chunk] += v.weight;
            }
        }
    }
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
        let (fi, ti) = chunks.turn_of[c];
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

    let judged = attributed(&gpu, &chunks, &held_out, n)?;
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

    // A factor of 0 for the most generic chunks, applied where the kernel
    // would apply one: the scan never sees them, so their votes go to the
    // next-best agreement instead of shrinking.
    for &cut in &CUTS {
        let n_cut = (cut * mass.len() as f32) as usize;
        let excluded: Vec<(u64, usize, usize)> = by_mass[..n_cut]
            .iter()
            .map(|&c| chunks.range_of[c])
            .collect();
        let mut s: Vec<Vec<f32>> = Vec::with_capacity(held_out.len());
        for batch in held_out.chunks(BATCH) {
            let sigs: Vec<&[WideQSig]> = batch.iter().map(|(_, s)| *s).collect();
            let votes = gpu.votes_without(
                &sigs,
                &|_, _| false,
                Questions::Without,
                &|_, _| false,
                &excluded,
            )?;
            s.extend(votes.iter().map(|v| summed(v, &[1.0, 1.0, 1.0], KEEP, n)));
        }
        let ranks: Vec<usize> = owners
            .iter()
            .zip(&s)
            .map(|(owner, sc)| rank_of(sc, *owner))
            .collect();
        let w = wins(&owners, &s, n);
        let total: usize = w.iter().sum();
        println!(
            "{:<26} {:>6.3} {:>6.3} {:>6.3} {:>9.3} {:>7.1}% {:>10}",
            format!("cut top {:.2}% ({n_cut})", cut * 100.0),
            recall(&ranks, 1),
            recall(&ranks, 10),
            mrr(&ranks),
            gini(&w),
            100.0 * *w.iter().max().unwrap_or(&0) as f32 / total.max(1) as f32,
            hubs.iter().map(|&f| w[f]).sum::<usize>()
        );
    }
    Ok(())
}

/// The shares of chunks, most generic first, cut from the gallery.
const CUTS: [f32; 8] = [0.0, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01];

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
