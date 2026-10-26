//! Does the working set's momentum turn better single scans into a better
//! selection?
//!
//! A dialogue that stays on a file asks about it turn after turn, and each
//! turn reprojects at its opening (the question) and again once it has
//! thought (question and thinking). Each scan's fresh score — the file's
//! content plus a weight × its calibration competition, each summed over its
//! fold groups by a [`Scoring`] — feeds `m ← (1 − β)·m + fresh`
//! (`working_set::WorkingSet::observe`), and the file is ranked by `m` after
//! every scan.
//!
//! Two sequences, over every file calibrated with thinking:
//! - **staying**: file A's held-out questions `0..HELD`, each as two scans;
//! - **switching**: the same, then file B's — how many scans until B leads.
//!
//! `β = 1` keeps nothing between scans: it is the single scan's ranking.

use candle_conversation::provenance::WideQSig;

use crate::attribution::{BATCH, KEEP};
use crate::corpus::File;
use crate::ranking::rank_of;
use crate::remote_gallery::{Daemon, Questions, RemoteGallery};
use crate::stats::{mrr, recall};
use crate::sweep::{summed, TokenVotes};
use crate::thinking_gate::thought;

/// Questions per file asked in sequence, held out of the gallery.
const HELD: usize = 3;

/// Questions per file the calibration competition holds, after the held ones.
const GALLERY: usize = 8;

/// Leaks tried; production's is 0.2.
const BETAS: [f32; 5] = [1.0, 0.5, 0.2, 0.1, 0.0];

/// How a scan's votes become a file's fresh score.
struct Scoring {
    name: &'static str,
    content: [f32; 3],
    calibration: [f32; 3],
    weight: f32,
}

/// Production's scoring, and the `layers` gate's best on live-turn probes.
const SCORINGS: [Scoring; 2] = [
    Scoring {
        name: "production [1,1,1] + 0.5 x [1,1,1]",
        content: [1.0, 1.0, 1.0],
        calibration: [1.0, 1.0, 1.0],
        weight: 0.5,
    },
    Scoring {
        name: "layers [0,1,1] + 0.25 x [0,0,1]",
        content: [0.0, 1.0, 1.0],
        calibration: [0.0, 0.0, 1.0],
        weight: 0.25,
    },
];

/// One scan of a turn: its question alone, or question and thinking.
#[derive(Clone, Copy)]
enum Scan {
    Opening,
    Thought,
}

/// The scans of a dialogue asking `HELD` questions about one file.
const SCANS: [(usize, Scan); 2 * HELD] = [
    (0, Scan::Opening),
    (0, Scan::Thought),
    (1, Scan::Opening),
    (1, Scan::Thought),
    (2, Scan::Opening),
    (2, Scan::Thought),
];

/// `m ← (1 − β)·m + fresh` over a sequence of scans: the momentum after each.
fn momentum(fresh: &[&[f32]], beta: f32) -> Vec<Vec<f32>> {
    let mut m = vec![0.0f32; fresh.first().map_or(0, |f| f.len())];
    fresh
        .iter()
        .map(|f| {
            for (mi, &fi) in m.iter_mut().zip(f.iter()) {
                *mi = (1.0 - beta) * *mi + fi;
            }
            m.clone()
        })
        .collect()
}

/// `content + w × calibration` per probe, each summed over its fold groups.
fn fresh_of(content: &[TokenVotes], cal: &[TokenVotes], s: &Scoring, n: usize) -> Vec<Vec<f32>> {
    content
        .iter()
        .zip(cal)
        .map(|(c, q)| {
            let c = summed(c, &s.content, KEEP, n);
            let q = summed(q, &s.calibration, KEEP, n);
            c.iter().zip(&q).map(|(a, b)| a + s.weight * b).collect()
        })
        .collect()
}

pub fn run(files: &[File], daemon: &Daemon) -> anyhow::Result<()> {
    let gpu = RemoteGallery::new(files, daemon);
    let n = files.len();
    let ready: Vec<bool> = files
        .iter()
        .map(|f| {
            f.questions.len() >= HELD + GALLERY && f.questions[..HELD + GALLERY].iter().all(thought)
        })
        .collect();
    let asked: Vec<usize> = (0..n).filter(|&fi| ready[fi]).collect();
    println!(
        "momentum: {} files calibrated with thinking, {HELD} questions each asked in sequence",
        asked.len()
    );
    anyhow::ensure!(
        asked.len() >= 2,
        "fewer than two files calibrated with thinking"
    );
    let masked = |fi: usize, qi: usize| !ready[fi] || !(HELD..HELD + GALLERY).contains(&qi);

    // Every scan's votes, content and competition apart: `votes[s]` holds
    // asked file `a`'s probe at scan `s` at index `a`.
    let mut votes: Vec<(Vec<TokenVotes>, Vec<TokenVotes>)> = Vec::with_capacity(SCANS.len());
    for &(qi, scan) in &SCANS {
        let held: Vec<&[WideQSig]> = asked
            .iter()
            .map(|&fi| {
                let q = &files[fi].questions[qi];
                match scan {
                    Scan::Opening => q.body_sigs(),
                    Scan::Thought => q.sigs.as_slice(),
                }
            })
            .collect();
        let (mut content, mut cal) = (Vec::new(), Vec::new());
        for batch in held.chunks(BATCH) {
            content.extend(gpu.votes_without(
                batch,
                &|_, _| false,
                Questions::Without,
                &|_, _| false,
                &[],
            )?);
            cal.extend(gpu.votes_without(batch, &|_, _| true, Questions::With, &masked, &[])?);
        }
        votes.push((content, cal));
    }

    for scoring in &SCORINGS {
        let fresh: Vec<Vec<Vec<f32>>> = votes
            .iter()
            .map(|(c, q)| fresh_of(c, q, scoring, n))
            .collect();
        println!("\n== {}", scoring.name);
        staying(&fresh, &asked);
        switching(&fresh, &asked);
    }
    Ok(())
}

/// Staying on one file: MRR after each scan, R@1 / R@10 after the last.
fn staying(fresh: &[Vec<Vec<f32>>], asked: &[usize]) {
    println!("staying on one file: MRR after each scan, R@1 / R@10 after the last");
    print!("{:<8}", "beta");
    for (qi, scan) in SCANS {
        let s = match scan {
            Scan::Opening => "open",
            Scan::Thought => "thought",
        };
        print!(" {:>9}", format!("q{qi} {s}"));
    }
    println!(" {:>6} {:>6}", "R@1", "R@10");
    for &beta in &BETAS {
        let mut ranks: Vec<Vec<usize>> = vec![Vec::with_capacity(asked.len()); SCANS.len()];
        for (a, &fi) in asked.iter().enumerate() {
            let seq: Vec<&[f32]> = fresh.iter().map(|f| f[a].as_slice()).collect();
            for (s, m) in momentum(&seq, beta).iter().enumerate() {
                ranks[s].push(rank_of(m, fi));
            }
        }
        print!("{beta:<8}");
        for r in &ranks {
            print!(" {:>9.3}", mrr(r));
        }
        let last = &ranks[SCANS.len() - 1];
        println!(" {:>6.3} {:>6.3}", recall(last, 1), recall(last, 10));
    }
}

/// Switching: file A's scans, then file B's (the next file asked). B's MRR
/// after each of its scans, and the share of dialogues where B leads A.
fn switching(fresh: &[Vec<Vec<f32>>], asked: &[usize]) {
    println!("switching to another file: B's MRR after each of B's scans, and the share where B outranks A");
    print!("{:<8}", "beta");
    for i in 0..SCANS.len() {
        print!(" {:>9}", format!("B scan {}", i + 1));
    }
    println!(" {:>12}", "B over A, end");
    for &beta in &BETAS {
        let mut ranks: Vec<Vec<usize>> = vec![Vec::with_capacity(asked.len()); SCANS.len()];
        let mut over = 0usize;
        for (a, &fa) in asked.iter().enumerate() {
            let b = (a + 1) % asked.len();
            let fb = asked[b];
            let seq: Vec<&[f32]> = fresh
                .iter()
                .map(|f| f[a].as_slice())
                .chain(fresh.iter().map(|f| f[b].as_slice()))
                .collect();
            let ms = momentum(&seq, beta);
            for (s, m) in ms[SCANS.len()..].iter().enumerate() {
                ranks[s].push(rank_of(m, fb));
            }
            let end = &ms[ms.len() - 1];
            if end[fb] > end[fa] {
                over += 1;
            }
        }
        print!("{beta:<8}");
        for r in &ranks {
            print!(" {:>9.3}", mrr(r));
        }
        println!(" {:>12.3}", over as f32 / asked.len() as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn momentum_leaks_beta_of_the_past_each_scan() {
        let a = [10.0f32, 0.0];
        let b = [0.0f32, 4.0];
        let seq: Vec<&[f32]> = vec![&a, &b, &b];
        assert_eq!(
            momentum(&seq, 0.5),
            vec![vec![10.0, 0.0], vec![5.0, 4.0], vec![2.5, 6.0]]
        );
        // β = 1 keeps nothing: each scan stands alone.
        assert_eq!(momentum(&seq, 1.0)[1], vec![0.0, 4.0]);
        // β = 0 sums every scan.
        assert_eq!(momentum(&seq, 0.0)[2], vec![10.0, 8.0]);
    }
}
