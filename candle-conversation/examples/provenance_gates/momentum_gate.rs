//! Does the working set's momentum turn better single scans into a better
//! selection?
//!
//! A dialogue that stays on a file asks about it turn after turn, and each
//! turn reprojects at its opening (the question) and again once it has
//! thought (question and thinking). Each scan's fresh score — the file's
//! content plus `WEIGHT ×` its calibration competition, as production scores
//! it — feeds `m ← (1 − β)·m + fresh` (`working_set::WorkingSet::observe`),
//! and the file is ranked by `m` after every scan.
//!
//! Two sequences, over every file calibrated with thinking:
//! - **staying**: file A's held-out questions `0..HELD`, each as two scans;
//! - **switching**: the same, then file B's — how many scans until B leads.
//!
//! `β = 1` keeps nothing between scans: it is the single scan's ranking.

use candle_conversation::provenance::WideQSig;

use crate::corpus::File;
use crate::judge::Judge;
use crate::ranking::rank_of;
use crate::remote_gallery::{Daemon, Questions, RemoteGallery};
use crate::stats::{mrr, recall};
use crate::thinking_gate::{combined, competition, thought};

/// Questions per file asked in sequence, held out of the gallery.
const HELD: usize = 3;

/// Questions per file the calibration competition holds, after the held ones.
const GALLERY: usize = 8;

/// The calibration competition's weight, production's.
const WEIGHT: f32 = 0.5;

/// Leaks tried; production's is 0.2.
const BETAS: [f32; 5] = [1.0, 0.5, 0.2, 0.1, 0.0];

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

    // Every file's fresh scores at every scan: `fresh[s][a]` is asked file
    // `a`'s probe at scan `s`, scored over all `n` files.
    let mut fresh: Vec<Vec<Vec<f32>>> = Vec::with_capacity(SCANS.len());
    for &(qi, scan) in &SCANS {
        let held: Vec<(usize, &[WideQSig])> = asked
            .iter()
            .map(|&fi| {
                let q = &files[fi].questions[qi];
                let sigs = match scan {
                    Scan::Opening => q.body_sigs(),
                    Scan::Thought => &q.sigs,
                };
                (fi, sigs)
            })
            .collect();
        let judge = Judge {
            gpu: &gpu,
            held_out: &held,
            hubs: &[],
            n,
            questions: Questions::Without,
            masked_question: &|_, _| false,
        };
        let content = judge.scores(&[])?;
        let calibration = competition(&gpu, &held, n, &masked, &[])?;
        fresh.push(combined(&content, &calibration, WEIGHT));
    }

    println!("\nstaying on one file: MRR after each scan, R@1 / R@10 after the last");
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

    // Switching: file A's scans, then file B's (the next file asked). Rank of
    // B after each of its scans, and the share of dialogues where B leads A.
    println!("\nswitching to another file: B's MRR after each of B's scans, and the share where B outranks A");
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
    Ok(())
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
