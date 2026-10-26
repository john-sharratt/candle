//! Held-out questions scored through the daemon under a gallery and its cuts,
//! one table row per setting.

use candle_conversation::provenance::WideQSig;

use crate::attribution::{BATCH, KEEP};
use crate::ranking::{rank_of, wins};
use crate::remote_gallery::{Questions, RemoteGallery};
use crate::stats::{gini, mrr, recall};
use crate::sweep::summed;

/// The questions asked, and the gallery they are asked of.
pub struct Judge<'a> {
    pub gpu: &'a RemoteGallery<'a>,
    /// `(owner, probe)` per held-out question.
    pub held_out: &'a [(usize, &'a [WideQSig])],
    /// The files whose wins on other files' questions are counted.
    pub hubs: &'a [usize],
    pub n: usize,
    /// Whether the gallery holds calibration questions, and which it leaves
    /// out, as `(file, question)`.
    pub questions: Questions,
    pub masked_question: &'a dyn Fn(usize, usize) -> bool,
}

impl Judge<'_> {
    /// Every held-out question's file scores with `excluded` cut.
    pub fn scores(&self, excluded: &[(u64, usize, usize)]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut s: Vec<Vec<f32>> = Vec::with_capacity(self.held_out.len());
        for batch in self.held_out.chunks(BATCH) {
            let sigs: Vec<&[WideQSig]> = batch.iter().map(|(_, s)| *s).collect();
            let votes = self.gpu.votes_without(
                &sigs,
                &|_, _| false,
                self.questions,
                self.masked_question,
                excluded,
            )?;
            s.extend(
                votes
                    .iter()
                    .map(|v| summed(v, &[1.0, 1.0, 1.0], KEEP, self.n)),
            );
        }
        Ok(s)
    }

    /// One table row: the held-out questions scored with `excluded` cut.
    pub fn row(&self, name: &str, excluded: &[(u64, usize, usize)]) -> anyhow::Result<()> {
        let s = self.scores(excluded)?;
        self.print(name, &s);
        Ok(())
    }

    /// One table row for scores already taken.
    pub fn print(&self, name: &str, s: &[Vec<f32>]) {
        let owners: Vec<usize> = self.held_out.iter().map(|(o, _)| *o).collect();
        let ranks: Vec<usize> = owners
            .iter()
            .zip(s)
            .map(|(owner, sc)| rank_of(sc, *owner))
            .collect();
        let w = wins(&owners, s, self.n);
        let total: usize = w.iter().sum();
        println!(
            "{name:<30} {:>6.3} {:>6.3} {:>6.3} {:>9.3} {:>7.1}% {:>10}",
            recall(&ranks, 1),
            recall(&ranks, 10),
            mrr(&ranks),
            gini(&w),
            100.0 * *w.iter().max().unwrap_or(&0) as f32 / total.max(1) as f32,
            self.hubs.iter().map(|&f| w[f]).sum::<usize>()
        );
    }
}

/// The table's header, for [`Judge::row`]'s columns.
pub fn header(first: &str) {
    println!(
        "\n{first:<30} {:>6} {:>6} {:>6} {:>9} {:>8} {:>10}",
        "R@1", "R@10", "MRR", "win Gini", "top win", "hub wins"
    );
}
