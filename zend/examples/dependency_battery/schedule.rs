//! The order the battery's answers are asked in (`docs/unbounded_agents.md`
//! §9.12, *Running*): every arm runs every question up to its highest run,
//! and the queue interleaves them so whatever drifts across the run — the
//! gallery, ingest, the card's load — falls on every arm alike.
//!
//! Only what the results file does not already record is queued, in passes:
//! a question's first owed run of each arm is in the first pass, its second
//! in the second, and so on. On a fresh file a pass is a run. When some arms
//! have runs recorded and others do not — runs 4–6 added to two arms beside
//! runs 1–3 of two new ones — their owed runs share passes, so the new
//! answers interleave with each other rather than one arm's following
//! another's. Pass by pass, the questions come in an order shuffled for that
//! pass, and each question's arms in an order shuffled for that question;
//! sixteen at a time are drawn from the front of the queue, so every batch
//! spans two or three questions and mixes the arms.

use std::collections::HashSet;

use crate::arms::Arm;
use crate::stats::SplitMix64;

/// One answer to ask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Job {
    pub run: u32,
    /// The question's index in the battery.
    pub question: usize,
    pub arm: Arm,
}

/// The `(arm, run, question id)` of every answer already recorded.
pub type Recorded = HashSet<(String, u32, String)>;

/// `xs` shuffled in place by Fisher–Yates from `rng`.
pub fn shuffle<T>(xs: &mut [T], rng: &mut SplitMix64) {
    for i in (1..xs.len()).rev() {
        xs.swap(i, rng.below(i + 1));
    }
}

/// The queue of every answer `done` does not record: each question of `ids`
/// under each arm of `arms` up to that arm's highest run, from `seed`'s
/// stream.
pub fn queue(ids: &[String], arms: &[(Arm, u32)], done: &Recorded, seed: u64) -> Vec<Job> {
    // owed[q][a]: the runs arm `a` still owes question `q`, in order.
    let owed: Vec<Vec<Vec<u32>>> = ids
        .iter()
        .map(|id| {
            arms.iter()
                .map(|&(arm, upto)| {
                    (1..=upto)
                        .filter(|&run| !done.contains(&(arm.id().to_string(), run, id.clone())))
                        .collect()
                })
                .collect()
        })
        .collect();
    let passes = owed.iter().flatten().map(Vec::len).max().unwrap_or(0);
    let mut rng = SplitMix64::new(seed);
    let mut out = Vec::new();
    for pass in 0..passes {
        let mut order: Vec<usize> = (0..ids.len()).collect();
        shuffle(&mut order, &mut rng);
        for question in order {
            let mut here: Vec<Job> = arms
                .iter()
                .zip(&owed[question])
                .filter_map(|(&(arm, _), runs)| {
                    runs.get(pass).map(|&run| Job { run, question, arm })
                })
                .collect();
            shuffle(&mut here, &mut rng);
            out.extend(here);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arms::ARMS;

    fn ids(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("q{i:03}")).collect()
    }

    fn every_arm(runs: u32) -> Vec<(Arm, u32)> {
        ARMS.iter().map(|&a| (a, runs)).collect()
    }

    /// **Every (run, question, arm) is asked exactly once**, and the same
    /// seed gives the same queue.
    #[test]
    fn the_queue_asks_everything_once() {
        let none = Recorded::new();
        let q = queue(&ids(128), &every_arm(3), &none, 9);
        assert_eq!(q.len(), 128 * 3 * ARMS.len());
        let distinct: HashSet<(u32, usize, Arm)> =
            q.iter().map(|j| (j.run, j.question, j.arm)).collect();
        assert_eq!(distinct.len(), q.len());
        assert_eq!(q, queue(&ids(128), &every_arm(3), &none, 9));
        assert_ne!(q, queue(&ids(128), &every_arm(3), &none, 10));
    }

    /// **Every batch of sixteen mixes the arms**: none is a single arm's,
    /// and each holds at least five of the nine.
    #[test]
    fn every_batch_mixes_the_arms() {
        let q = queue(&ids(128), &every_arm(3), &Recorded::new(), 9);
        for batch in q.chunks(16) {
            let arms: HashSet<Arm> = batch.iter().map(|j| j.arm).collect();
            assert!(arms.len() >= 5, "{batch:?}");
        }
    }

    /// **The runs come in order**, so the third run of a question is never
    /// asked before its first.
    #[test]
    fn runs_come_in_order() {
        let q = queue(&ids(10), &every_arm(3), &Recorded::new(), 1);
        assert!(q.windows(2).all(|w| w[0].run <= w[1].run));
    }

    /// **Only what is not recorded is asked**: with runs 1–3 of C1 and B1
    /// recorded, asking them to run 6 queues runs 4–6 alone, and the new arms'
    /// runs 1–3 interleave with them pass by pass.
    #[test]
    fn recorded_runs_are_skipped_and_the_rest_interleave() {
        let qs = ids(128);
        let mut done = Recorded::new();
        for arm in ARMS.iter().filter(|a| !matches!(a, Arm::R1 | Arm::T1)) {
            for run in 1..=3 {
                for id in &qs {
                    done.insert((arm.id().to_string(), run, id.clone()));
                }
            }
        }
        let only_c1_b1 = queue(&qs, &[(Arm::C1, 6), (Arm::B1, 6)], &done, 9);
        assert_eq!(only_c1_b1.len(), 2 * 128 * 3);
        let runs: HashSet<u32> = only_c1_b1.iter().map(|j| j.run).collect();
        assert_eq!(runs, HashSet::from([4, 5, 6]));

        let plan = [(Arm::R1, 3), (Arm::T1, 3), (Arm::C1, 6), (Arm::B1, 6)];
        let q = queue(&qs, &plan, &done, 9);
        assert_eq!(q.len(), 4 * 128 * 3);
        let asked: HashSet<(Arm, u32)> = q.iter().map(|j| (j.arm, j.run)).collect();
        let want: HashSet<(Arm, u32)> = [
            (Arm::R1, 1),
            (Arm::R1, 2),
            (Arm::R1, 3),
            (Arm::T1, 1),
            (Arm::T1, 2),
            (Arm::T1, 3),
            (Arm::C1, 4),
            (Arm::C1, 5),
            (Arm::C1, 6),
            (Arm::B1, 4),
            (Arm::B1, 5),
            (Arm::B1, 6),
        ]
        .into();
        assert_eq!(asked, want);
        for batch in q.chunks(16) {
            let arms: HashSet<Arm> = batch.iter().map(|j| j.arm).collect();
            assert_eq!(arms.len(), 4, "{batch:?}");
        }
        assert!(
            q[..4 * 128].iter().all(|j| matches!(j.run, 1 | 4)),
            "the first pass is each arm's first owed run"
        );
    }

    /// **A run cut short is taken up where it stopped**: the one answer a
    /// question still owes is the only one queued for it.
    #[test]
    fn a_partly_recorded_question_owes_only_what_is_missing() {
        let qs = ids(2);
        let mut done = Recorded::new();
        for run in 1..=3 {
            done.insert(("c1".to_string(), run, "q000".to_string()));
        }
        done.insert(("c1".to_string(), 1, "q001".to_string()));
        done.insert(("c1".to_string(), 3, "q001".to_string()));
        assert_eq!(
            queue(&qs, &[(Arm::C1, 3)], &done, 4),
            vec![Job {
                run: 2,
                question: 1,
                arm: Arm::C1
            }]
        );
    }
}
