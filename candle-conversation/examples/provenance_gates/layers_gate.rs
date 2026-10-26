//! Which fold groups carry a file, and which carry its calibration?
//!
//! Every vote is cast per fold group (`sweep::FOLD_GROUPS`), and a group's
//! `layer_weights` scale its votes before they are summed. `code_reading`
//! weighs every group alike, a weighting tuned before its calibration
//! competition existed, and the competition sums its groups alike too. The
//! questions and thinking a file is calibrated with need not live in the
//! groups its content does, so content and competition are weighed apart
//! here: one scan each, every weighting re-summed from the same votes.
//!
//! Probes as `thinking_gate`'s: each file calibrated with thinking asks its
//! held-out question 0 — its words, its thinking, and both as one turn.

use candle_conversation::provenance::WideQSig;

use crate::attribution::{BATCH, KEEP};
use crate::corpus::File;
use crate::ranking::rank_of;
use crate::remote_gallery::{Daemon, Questions, RemoteGallery};
use crate::stats::{mrr, recall};
use crate::sweep::{summed, TokenVotes};
use crate::thinking_gate::thought;

/// Questions per file the competition holds, beyond question 0.
const GALLERY: usize = 8;

/// Fold-group weightings tried, for content and for the competition alike.
const WEIGHTINGS: [[f32; 3]; 8] = [
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [0.1, 1.0, 0.1],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, 0.5],
];

/// Calibration-competition weights tried.
const CAL_WEIGHTS: [f32; 3] = [0.25, 0.5, 0.75];

/// Combined settings shown per probe, best MRR first.
const SHOWN: usize = 12;

/// The probe's part of a question.
#[derive(Clone, Copy)]
enum Probe {
    Question,
    Thinking,
    Turn,
}

/// One table row's measures over `ranks`.
fn row(name: &str, ranks: &[usize]) {
    println!(
        "{name:<44} {:>6.3} {:>6.3} {:>6.3}",
        recall(ranks, 1),
        recall(ranks, 10),
        mrr(ranks)
    );
}

/// Each probe's per-file score under `weights`.
fn scored(votes: &[TokenVotes], weights: &[f32; 3], n: usize) -> Vec<Vec<f32>> {
    votes.iter().map(|v| summed(v, weights, KEEP, n)).collect()
}

/// Each probe's rank of its own file under `content + w × competition`.
fn ranks(owners: &[usize], content: &[Vec<f32>], cal: Option<(&[Vec<f32>], f32)>) -> Vec<usize> {
    owners
        .iter()
        .enumerate()
        .map(|(p, &owner)| {
            let s: Vec<f32> = match cal {
                Some((c, w)) => content[p]
                    .iter()
                    .zip(&c[p])
                    .map(|(a, b)| a + w * b)
                    .collect(),
                None => content[p].clone(),
            };
            rank_of(&s, owner)
        })
        .collect()
}

pub fn run(files: &[File], daemon: &Daemon) -> anyhow::Result<()> {
    let gpu = RemoteGallery::new(files, daemon);
    let n = files.len();
    let ready: Vec<bool> = files
        .iter()
        .map(|f| f.questions.len() > GALLERY && f.questions[..=GALLERY].iter().all(thought))
        .collect();
    let asked: Vec<usize> = (0..n).filter(|&fi| ready[fi]).collect();
    println!("layers: {} files calibrated with thinking", asked.len());
    anyhow::ensure!(!asked.is_empty(), "no file is calibrated with thinking yet");
    let masked = |fi: usize, qi: usize| !ready[fi] || qi == 0 || qi > GALLERY;

    for (name, probe) in [
        ("question", Probe::Question),
        ("thinking", Probe::Thinking),
        ("turn", Probe::Turn),
    ] {
        let held: Vec<&[WideQSig]> = asked
            .iter()
            .map(|&fi| {
                let q = &files[fi].questions[0];
                match probe {
                    Probe::Question => q.body_sigs(),
                    Probe::Thinking => &q.sigs[q.answer.clone()],
                    Probe::Turn => q.sigs.as_slice(),
                }
            })
            .collect();
        let (mut content, mut cal): (Vec<TokenVotes>, Vec<TokenVotes>) = (Vec::new(), Vec::new());
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

        println!(
            "\n{:<44} {:>6} {:>6} {:>6}",
            format!("{name} probes ({})", held.len()),
            "R@1",
            "R@10",
            "MRR"
        );
        let contents: Vec<Vec<Vec<f32>>> =
            WEIGHTINGS.iter().map(|w| scored(&content, w, n)).collect();
        let cals: Vec<Vec<Vec<f32>>> = WEIGHTINGS.iter().map(|w| scored(&cal, w, n)).collect();
        for (w, c) in WEIGHTINGS.iter().zip(&contents) {
            row(&format!("content {w:?}"), &ranks(&asked, c, None));
        }
        for (w, c) in WEIGHTINGS.iter().zip(&cals) {
            row(&format!("competition {w:?}"), &ranks(&asked, c, None));
        }
        let mut combos: Vec<(f32, String, Vec<usize>)> = Vec::new();
        for (wc, c) in WEIGHTINGS.iter().zip(&contents) {
            for (wq, q) in WEIGHTINGS.iter().zip(&cals) {
                for &w in &CAL_WEIGHTS {
                    let r = ranks(&asked, c, Some((q, w)));
                    combos.push((mrr(&r), format!("{wc:?} + {w} x {wq:?}"), r));
                }
            }
        }
        let production = combos
            .iter()
            .find(|(_, label, _)| label == "[1.0, 1.0, 1.0] + 0.5 x [1.0, 1.0, 1.0]")
            .map(|(_, l, r)| (format!("production: {l}"), r.clone()));
        combos.sort_by(|a, b| b.0.total_cmp(&a.0));
        if let Some((label, r)) = production {
            row(&label, &r);
        }
        for (_, label, r) in combos.iter().take(SHOWN) {
            row(label, r);
        }
    }
    Ok(())
}
