//! Every rule's file scores for one probe: late fusion's votes summed per file
//! (today's scorer), and tallied per window under each file aggregate.

use std::iter::repeat_n;

use candle_conversation::provenance::window_score::{
    file_scores, late_fusion_votes, window_tally, FileAggregate,
};
use candle_conversation::provenance::WideQSig;

use crate::corpus::Gallery;

/// The rules, in the order [`score`] returns them.
pub fn rule_names() -> Vec<String> {
    ["late-fusion (today)".to_string()]
        .into_iter()
        .chain(
            FileAggregate::ALL
                .iter()
                .map(|a| format!("window votes / {}", a.name())),
        )
        .collect()
}

/// Every rule's score per file for `probe`, with the turn `left_out` — as
/// `(file, turn)` — taking no part.
pub fn score(
    probe: &[WideQSig],
    gallery: &Gallery,
    n_files: usize,
    left_out: Option<(usize, usize)>,
) -> Vec<Vec<f32>> {
    let token_file: Vec<usize> = gallery
        .turns
        .iter()
        .flat_map(|(f, t, r)| {
            let file = if Some((*f, *t)) == left_out {
                usize::MAX
            } else {
                *f
            };
            repeat_n(file, r.len())
        })
        .collect();
    let votes = late_fusion_votes(probe, &gallery.tokens, &token_file, n_files, &[]);
    let mut today = vec![0.0f32; n_files];
    for v in &votes {
        today[v.file] += v.weight;
    }
    let tally = window_tally(&votes, &gallery.windows, gallery.tokens.len());
    let live: Vec<bool> = gallery
        .window_turn
        .iter()
        .map(|wt| Some(*wt) != left_out)
        .collect();
    let mut out = vec![today];
    for a in FileAggregate::ALL {
        out.push(file_scores(
            &tally,
            &gallery.window_file,
            &gallery.identity,
            &live,
            n_files,
            a,
        ));
    }
    out
}

/// Files in descending order of `scores`.
pub fn order(scores: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    order
}
