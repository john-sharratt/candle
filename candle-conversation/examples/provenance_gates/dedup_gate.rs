//! Dedup: how much of the content gallery survives near-duplicate reduction
//! within each file, and what its loss costs retrieval.
//!
//! A seeded sample of calibrated `code_reading` files is the competition —
//! every other file masked, so a reduced file never competes with an unreduced
//! one. Their content tokens' signatures over the fold groups content weighs
//! (1 and 2) are named by a learned codebook ([`crate::codebook`]) at several
//! sizes, and each setting keeps, per file, the first few tokens of each code
//! — beside controls with no codebook that keep one token in k, by position
//! and by a seeded draw. Each sample file's calibration
//! questions then probe for it on the daemon's gallery with the dropped tokens
//! cut (`excluded`), scored on content alone and as production scores a file:
//! content plus [`CALIBRATION`] × the calibration competition, which the
//! reduction leaves whole.

use std::collections::HashSet;
use std::slice::from_ref;
use std::time::Instant;

use candle_conversation::persistence::content_hash::turn_stream_id;
use candle_conversation::provenance::WideQSig;

use crate::attribution::{BATCH, KEEP};
use crate::codebook::{
    assign, distance, dropped_runs, every, file_counts, kept, sampled, train, Slice, Tree,
};
use crate::corpus::File;
use crate::ranking::rank_of;
use crate::remote_gallery::{Daemon, Questions, RemoteGallery};
use crate::stats::{mrr, recall};
use crate::sweep::summed;

/// Words per fold group of a token.
const GROUP_WORDS: usize = 8;
/// The fold groups content is weighed in, and production's weights.
const CONTENT_GROUPS: [usize; 2] = [1, 2];
const CONTENT_WEIGHTS: [f32; 3] = [0.0, 1.0, 1.0];
/// Production's calibration competition: its fold-group weights and its weight
/// against the content's (`[0,1,1] + 0.5 × [0,0,1]`).
const CALIBRATION_WEIGHTS: [f32; 3] = [0.0, 0.0, 1.0];
const CALIBRATION: f32 = 0.5;
/// Codebook sizes tried.
const CODEBOOKS: [usize; 1] = [4_096];
/// Tokens kept per code per file.
const PER_CODE: [usize; 2] = [1, 2];
/// The controls with no codebook: one token in each of these kept, by
/// position and by a seeded draw.
const STRIDES: [usize; 3] = [3, 4, 6];
/// The two-level codebook tried: coarse codes, fine codes in each.
const TREE: (usize, usize) = (64, 64);
/// Points a codebook trains on, sampled from the gallery.
const TRAIN_POINTS: usize = 200_000;
/// Rounds of k-means.
const ITERATIONS: usize = 4;
/// Bytes a kept token holds on the card in content's two fold groups.
const TOKEN_BYTES: usize = CONTENT_GROUPS.len() * GROUP_WORDS * 8;

/// Every fold group of a token, which a codebook shared by every reader of a
/// turn — whatever groups each weighs — names it over.
const ALL_GROUPS: [usize; 3] = [0, 1, 2];

/// A token's signature over the contiguous fold groups `groups`.
fn slice_of(sig: &WideQSig, groups: &[usize]) -> Option<Slice> {
    let end = (groups[groups.len() - 1] + 1) * GROUP_WORDS;
    (sig.words.len() >= end).then(|| sig.words[groups[0] * GROUP_WORDS..end].to_vec())
}

/// A token's signature over the content groups.
fn slice(sig: &WideQSig) -> Option<Slice> {
    slice_of(sig, &CONTENT_GROUPS)
}

/// `sample` files, picked by a seeded order of their timelines from the
/// calibrated files with signed content.
fn sample(files: &[File], n: usize) -> Vec<usize> {
    let mut ok: Vec<usize> = (0..files.len())
        .filter(|&f| {
            !files[f].questions.is_empty() && files[f].turns.iter().any(|t| !t.sigs.is_empty())
        })
        .collect();
    ok.sort_by_key(|&f| files[f].timeline.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    ok.truncate(n);
    ok.sort();
    ok
}

/// One file's content turns in gallery order: `(stream, first point, len)`.
type Turns = Vec<(u64, usize, usize)>;

/// The cut that keeps `keep[f]` of each file's tokens (its turns laid end to
/// end in `turns_of[f]`), as each turn's dropped runs, and the tokens kept.
fn cut(turns_of: &[Turns], keep: &[Vec<bool>]) -> (Vec<(u64, usize, usize)>, usize) {
    let mut excluded = Vec::new();
    let mut tokens = 0;
    for (turns, keep) in turns_of.iter().zip(keep) {
        tokens += keep.iter().filter(|&&k| k).count();
        let mut offset = 0;
        for &(stream, _, len) in turns {
            excluded.extend(
                dropped_runs(&keep[offset..offset + len])
                    .into_iter()
                    .map(|(s, e)| (stream, s, e)),
            );
            offset += len;
        }
    }
    (excluded, tokens)
}

/// Each file's keep mask: the first `per_code` of each code in `of`.
fn firsts(turns_of: &[Turns], of: &[u32], per_code: usize) -> Vec<Vec<bool>> {
    turns_of
        .iter()
        .map(|turns| {
            let codes: Vec<u32> = turns
                .iter()
                .flat_map(|&(_, at, len)| of[at..at + len].iter().copied())
                .collect();
            kept(&codes, per_code)
        })
        .collect()
}

/// `R@1`, `R@10` and MRR of `scores` for each probe's owner.
fn judged(owners: &[usize], scores: &[Vec<f32>]) -> (f32, f32, f32) {
    let ranks: Vec<usize> = owners
        .iter()
        .zip(scores)
        .map(|(o, s)| rank_of(s, *o))
        .collect();
    (recall(&ranks, 1), recall(&ranks, 10), mrr(&ranks))
}

pub fn run(
    files: &[File],
    sample_files: usize,
    per_file: usize,
    daemon: &Daemon,
) -> anyhow::Result<()> {
    let picked = sample(files, sample_files);
    let in_sample: HashSet<usize> = picked.iter().copied().collect();
    let mut points: Vec<Slice> = Vec::new();
    let mut turns_of: Vec<Turns> = Vec::with_capacity(picked.len());
    for &f in &picked {
        let mut turns = Turns::new();
        for (ti, t) in files[f].turns.iter().enumerate() {
            let slices: Vec<Slice> = t.sigs.iter().filter_map(slice).collect();
            if slices.len() != t.sigs.len() {
                continue;
            }
            turns.push((
                turn_stream_id(files[f].timeline, ti as u32).0,
                points.len(),
                slices.len(),
            ));
            points.extend(slices);
        }
        turns_of.push(turns);
    }
    let probes: Vec<(usize, &[WideQSig])> = picked
        .iter()
        .flat_map(|&f| {
            files[f]
                .questions
                .iter()
                .take(per_file)
                .map(move |q| (f, q.body_sigs()))
        })
        .filter(|(_, p)| !p.is_empty())
        .collect();
    let owners: Vec<usize> = probes.iter().map(|(o, _)| *o).collect();
    println!(
        "dedup: {} files, {} content tokens ({:.1} MiB in groups 1-2), {} question probes",
        picked.len(),
        points.len(),
        (points.len() * TOKEN_BYTES) as f64 / (1 << 20) as f64,
        probes.len()
    );

    let gpu = RemoteGallery::new(files, daemon);
    let outside = |f: usize, _: usize| !in_sample.contains(&f);
    let content_on = |gpu: &RemoteGallery,
                      excluded: &[(u64, usize, usize)]|
     -> anyhow::Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(probes.len());
        for batch in probes.chunks(BATCH) {
            let sigs: Vec<&[WideQSig]> = batch.iter().map(|(_, s)| *s).collect();
            let votes =
                gpu.votes_without(&sigs, &outside, Questions::Without, &|_, _| false, excluded)?;
            out.extend(
                votes
                    .iter()
                    .map(|v| summed(v, &CONTENT_WEIGHTS, KEEP, files.len())),
            );
        }
        Ok(out)
    };
    let content = |excluded: &[(u64, usize, usize)]| content_on(&gpu, excluded);
    // The calibration competition, the sample's questions but the ones asked;
    // the reduction leaves it whole, so it is scored once.
    let asked = |f: usize, q: usize| !in_sample.contains(&f) || q < per_file;
    let calibration_on = |gpu: &RemoteGallery| -> anyhow::Result<Vec<Vec<f32>>> {
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(probes.len());
        for batch in probes.chunks(BATCH) {
            let sigs: Vec<&[WideQSig]> = batch.iter().map(|(_, s)| *s).collect();
            let votes = gpu.votes(&sigs, &|_, _| true, Questions::With, &asked)?;
            out.extend(
                votes
                    .iter()
                    .map(|v| summed(v, &CALIBRATION_WEIGHTS, KEEP, files.len())),
            );
        }
        Ok(out)
    };
    let calibration = calibration_on(&gpu)?;
    let fused = |c: &[Vec<f32>], calibration: &[Vec<f32>]| -> Vec<Vec<f32>> {
        c.iter()
            .zip(calibration)
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + CALIBRATION * y).collect())
            .collect()
    };
    let production = |c: &[Vec<f32>]| fused(c, &calibration);

    println!(
        "\n{:<24} {:>8} {:>8} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6}",
        "setting", "kept", "MiB", "R@1", "R@10", "MRR", "R@1", "R@10", "MRR"
    );
    println!(
        "{:<24} {:>8} {:>8} | {:^20} | {:^20}",
        "", "", "", "content alone", "production rule"
    );
    let row = |name: &str, tokens: usize, scores: &[Vec<f32>]| {
        let (c1, c10, cm) = judged(&owners, scores);
        let (p1, p10, pm) = judged(&owners, &production(scores));
        println!(
            "{name:<24} {:>7.1}% {:>8.1} | {c1:>6.3} {c10:>6.3} {cm:>6.3} | {p1:>6.3} {p10:>6.3} {pm:>6.3}",
            100.0 * tokens as f64 / points.len().max(1) as f64,
            (tokens * TOKEN_BYTES) as f64 / (1 << 20) as f64,
        );
    };
    row("full gallery", points.len(), &content(&[])?);

    let lens: Vec<usize> = turns_of
        .iter()
        .map(|turns| turns.iter().map(|&(_, _, len)| len).sum())
        .collect();
    println!("\ncontrols, no codebook");
    for &stride in &STRIDES {
        let by_position: Vec<Vec<bool>> = lens.iter().map(|&n| every(n, stride)).collect();
        let (excluded, tokens) = cut(&turns_of, &by_position);
        row(&format!("  every {stride}th"), tokens, &content(&excluded)?);
        let drawn: Vec<Vec<bool>> = lens
            .iter()
            .enumerate()
            .map(|(f, &n)| sampled(n, stride, 0x0dd5_eed0 + f as u64))
            .collect();
        let (excluded, tokens) = cut(&turns_of, &drawn);
        row(
            &format!("  random 1 in {stride}"),
            tokens,
            &content(&excluded)?,
        );
    }

    let step = (points.len() / TRAIN_POINTS).max(1);
    let training: Vec<Slice> = points.iter().step_by(step).cloned().collect();
    for &c in &CODEBOOKS {
        let t = Instant::now();
        let codes = train(&training, c, ITERATIONS, 0x5eed_c0de);
        let of = assign(&points, &codes);
        let bits: Vec<u32> = points
            .iter()
            .zip(&of)
            .map(|(p, &k)| distance(p, &codes[k as usize]))
            .collect();
        let by_file: Vec<Vec<u32>> = turns_of
            .iter()
            .map(|turns| {
                turns
                    .iter()
                    .flat_map(|&(_, at, len)| of[at..at + len].iter().copied())
                    .collect()
            })
            .collect();
        println!(
            "\ncodebook {c}: {} codes used, mean {:.0} of {} bits from a token to its code ({:.0}s)",
            file_counts(&by_file).len(),
            bits.iter().map(|&b| b as f64).sum::<f64>() / bits.len().max(1) as f64,
            CONTENT_GROUPS.len() * GROUP_WORDS * 64,
            t.elapsed().as_secs_f64(),
        );
        for &per_code in &PER_CODE {
            let (excluded, tokens) = cut(&turns_of, &firsts(&turns_of, &of, per_code));
            row(
                &format!("  {per_code} per code, first"),
                tokens,
                &content(&excluded)?,
            );
        }
    }

    let t = Instant::now();
    let tree = Tree::train(&training, TREE.0, TREE.1, ITERATIONS, 0x5eed_c0de);
    let of = tree.assign(&points);
    println!(
        "\ncodebook {}x{} (two-level): {} codes used ({:.0}s)",
        TREE.0,
        TREE.1,
        file_counts(from_ref(&of)).len(),
        t.elapsed().as_secs_f64(),
    );
    for &per_code in &PER_CODE {
        let (excluded, tokens) = cut(&turns_of, &firsts(&turns_of, &of, per_code));
        row(
            &format!("  {per_code} per code, first"),
            tokens,
            &content(&excluded)?,
        );
        // Deduplicated within each turn rather than across a file's turns —
        // what a gallery that uploads a turn at a time can do on its own.
        let by_turn: Vec<Vec<bool>> = turns_of
            .iter()
            .map(|turns| {
                turns
                    .iter()
                    .flat_map(|&(_, at, len)| kept(&of[at..at + len], per_code))
                    .collect()
            })
            .collect();
        let (excluded, tokens) = cut(&turns_of, &by_turn);
        row(
            &format!("  {per_code} per code, per turn"),
            tokens,
            &content(&excluded)?,
        );
    }

    // The same tree over every fold group: one mask per turn for every reader,
    // whichever groups it weighs.
    let whole: Vec<Slice> = picked
        .iter()
        .flat_map(|&f| files[f].turns.iter())
        .filter(|t| t.sigs.iter().all(|s| slice(s).is_some()))
        .flat_map(|t| t.sigs.iter().filter_map(|s| slice_of(s, &ALL_GROUPS)))
        .collect();
    assert_eq!(
        whole.len(),
        points.len(),
        "one whole slice per content point"
    );
    let t = Instant::now();
    let training: Vec<Slice> = whole.iter().step_by(step).cloned().collect();
    let tree = Tree::train(&training, TREE.0, TREE.1, ITERATIONS, 0x5eed_c0de);
    let of = tree.assign(&whole);
    println!(
        "\ncodebook {}x{} over every group: {} codes used ({:.0}s)",
        TREE.0,
        TREE.1,
        file_counts(from_ref(&of)).len(),
        t.elapsed().as_secs_f64(),
    );
    for &per_code in &PER_CODE {
        let by_turn: Vec<Vec<bool>> = turns_of
            .iter()
            .map(|turns| {
                turns
                    .iter()
                    .flat_map(|&(_, at, len)| kept(&of[at..at + len], per_code))
                    .collect()
            })
            .collect();
        let (excluded, tokens) = cut(&turns_of, &by_turn);
        row(
            &format!("  {per_code} per code, per turn"),
            tokens,
            &content(&excluded)?,
        );
    }

    // The daemon's own overlay (`docs/gallery_dedup.md`): its codebook, its
    // masks, on its card — content deduped, then the calibration competition's
    // questions deduped too.
    println!("\nthe daemon's overlay (kept: as the codebook above, not counted here)");
    for &per_code in &PER_CODE {
        let deduped = gpu.deduped(per_code as u8);
        let c = content_on(&deduped, &[])?;
        row(&format!("  {per_code} per code, content"), 0, &c);
        let (p1, p10, pm) = judged(&owners, &fused(&c, &calibration_on(&deduped)?));
        println!(
            "  {per_code} per code, content + questions{:>32} | {p1:>6.3} {p10:>6.3} {pm:>6.3}",
            ""
        );
    }
    Ok(())
}
