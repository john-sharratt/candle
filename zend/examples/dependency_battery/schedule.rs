//! The order the battery's answers are asked in (`docs/unbounded_agents.md`
//! §9.12, *Running*): every arm runs every question three times, and the
//! queue interleaves them so whatever drifts across the run — the gallery,
//! ingest, the card's load — falls on every arm alike.
//!
//! Run by run, the questions come in an order shuffled for that run, and each
//! question's seven arms in an order shuffled for that question; sixteen at a
//! time are drawn from the front of the queue, so every batch spans two or
//! three questions and mixes the arms.

use crate::arms::{Arm, ARMS};
use crate::stats::SplitMix64;

/// One answer to ask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Job {
    pub run: u32,
    /// The question's index in the battery.
    pub question: usize,
    pub arm: Arm,
}

/// `xs` shuffled in place by Fisher–Yates from `rng`.
pub fn shuffle<T>(xs: &mut [T], rng: &mut SplitMix64) {
    for i in (1..xs.len()).rev() {
        xs.swap(i, rng.below(i + 1));
    }
}

/// The whole battery's queue: `runs` runs of `questions` questions, every
/// arm of each, from `seed`'s stream.
pub fn queue(questions: usize, runs: u32, seed: u64) -> Vec<Job> {
    let mut rng = SplitMix64::new(seed);
    let mut out = Vec::with_capacity(questions * runs as usize * ARMS.len());
    for run in 1..=runs {
        let mut order: Vec<usize> = (0..questions).collect();
        shuffle(&mut order, &mut rng);
        for question in order {
            let mut arms = ARMS;
            shuffle(&mut arms, &mut rng);
            out.extend(arms.into_iter().map(|arm| Job { run, question, arm }));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    /// **Every (run, question, arm) is asked exactly once**, and the same
    /// seed gives the same queue.
    #[test]
    fn the_queue_asks_everything_once() {
        let q = queue(128, 3, 9);
        assert_eq!(q.len(), 128 * 3 * 7);
        let distinct: HashSet<(u32, usize, Arm)> =
            q.iter().map(|j| (j.run, j.question, j.arm)).collect();
        assert_eq!(distinct.len(), q.len());
        assert_eq!(q, queue(128, 3, 9));
        assert_ne!(q, queue(128, 3, 10));
    }

    /// **Every batch of sixteen mixes the arms**: none is a single arm's,
    /// and each holds at least five of the seven.
    #[test]
    fn every_batch_mixes_the_arms() {
        let q = queue(128, 3, 9);
        for batch in q.chunks(16) {
            let arms: HashSet<Arm> = batch.iter().map(|j| j.arm).collect();
            assert!(arms.len() >= 5, "{batch:?}");
        }
    }

    /// **The runs come in order**, so the third run of a question is never
    /// asked before its first.
    #[test]
    fn runs_come_in_order() {
        let q = queue(10, 3, 1);
        assert!(q.windows(2).all(|w| w[0].run <= w[1].run));
    }
}
