//! The questions gate: each calibrated file's questions, one at a time, probe
//! for the file — the probe is a question, as a live one is, and its own turn
//! is left out of the gallery. Ranked by every rule ([`crate::rules`]), so the
//! same probes say what the questions buy over the files' content alone.

use anyhow::Context;
use candle_conversation::provenance::boundaries::BoundaryRule;

use crate::corpus::File;
use crate::ranking::order;
use crate::remote_gallery::{Daemon, RemoteGallery};
use crate::rules::{Scorer, RULES};
use crate::stats::mrr;

pub fn run(
    files: &[File],
    rule: &BoundaryRule,
    per_file: usize,
    daemon: &Daemon,
) -> anyhow::Result<()> {
    let gpu = RemoteGallery::new(files, daemon);
    let scorer = Scorer::new(&gpu, files, rule).context("learning the background peaks")?;
    let calibrated = files.iter().filter(|f| !f.questions.is_empty()).count();
    println!(
        "questions gate: {calibrated} of {} files calibrated, up to {per_file} probes each",
        files.len()
    );
    let mut ranks: Vec<Vec<usize>> = vec![Vec::new(); RULES.len()];
    for (fi, f) in files.iter().enumerate() {
        for (qi, q) in f.questions.iter().enumerate().take(per_file) {
            let scores = scorer
                .score(&q.sigs, &|_, _| false, &|f2, q2| f2 == fi && q2 == qi)
                .with_context(|| {
                    format!(
                        "file {fi} question {qi} ({} signatures, calibration {:016x} turn {})",
                        q.sigs.len(),
                        q.timeline,
                        q.turn
                    )
                })?;
            for (ri, s) in scores.iter().enumerate() {
                let rank = order(s)
                    .iter()
                    .position(|&x| x == fi)
                    .unwrap_or(files.len())
                    + 1;
                ranks[ri].push(rank);
            }
        }
    }
    println!(
        "\n{:<26} {:>7} {:>6} {:>6} {:>6} {:>6}",
        "rule", "probes", "R@1", "R@5", "R@10", "MRR"
    );
    for (r, rs) in RULES.iter().zip(&ranks) {
        let at = |k: usize| rs.iter().filter(|&&x| x <= k).count() as f32 / rs.len().max(1) as f32;
        println!(
            "{:<26} {:>7} {:>6.3} {:>6.3} {:>6.3} {:>6.3}",
            r.name,
            rs.len(),
            at(1),
            at(5),
            at(10),
            mrr(rs)
        );
    }
    Ok(())
}
