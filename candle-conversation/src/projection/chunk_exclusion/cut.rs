//! The cut a learner's state yields (`docs/incremental_chunk_exclusion.md`
//! §2, §4.4): each entry's vote from its statistics, the needle gate per
//! question, each kept vote for a file not the question's own credited to the
//! 32-token chunk holding the gallery token that won it, and today's outlier
//! rule over every chunk's mass.
//!
//! A winning token is found only where an entry has none: an entry keeps its
//! winner until its leader changes, so a fold that moved few leaders resolves
//! few tokens. Those it does resolve are found a file at a time, each file's
//! signatures decoded once, files in parallel.

use std::array::from_fn;
use std::collections::{BTreeMap, HashMap};

use rayon::prelude::*;

use super::fold::LearnerState;
use super::gallery::{Competition, GalleryStream};
use crate::persistence::streams::StreamId;
use crate::provenance::chunk_exclusion::anomaly::{outliers, KEEP};
use crate::provenance::chunk_exclusion::stats::FileId;
use crate::provenance::chunk_exclusion::CHUNK;
use crate::provenance::scan::NEEDLE_KEEP_FRAC;
use crate::provenance::window_score::{gate, Vote};
use crate::provenance::{WideQSig, FOLD_GROUPS};

/// One competition's cut, and what it was cut from.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompetitionCut {
    /// `(stream, start, end)` token ranges cut.
    pub cut: Vec<(StreamId, u32, u32)>,
    /// Chunks weighed.
    pub chunks: usize,
    /// Kept wrong-file votes credited.
    pub votes: usize,
    /// The fewest votes any cut chunk won — the margin the sample leaves.
    pub weakest_cut_votes: Option<u32>,
}

/// A kept vote to credit: the sample question, its token and fold group, and
/// its weight.
struct Kept {
    question: StreamId,
    token: usize,
    group: usize,
    weight: f32,
}

/// Competition `comp`'s cut over `streams` — its gallery as it stands — from
/// `state`, resolving the winning token of every kept entry that has none.
pub(crate) fn cut_of(
    state: &mut LearnerState,
    comp: Competition,
    streams: &[&GalleryStream],
) -> CompetitionCut {
    let ci = comp.index();
    let n = state.n[ci];
    // Every chunk the gallery's windows hold, file by file.
    let mut file_index: HashMap<FileId, usize> = HashMap::new();
    let mut by_file: BTreeMap<FileId, Vec<&GalleryStream>> = BTreeMap::new();
    for &s in streams {
        by_file.entry(s.file).or_default().push(s);
    }
    let mut chunks: Vec<(StreamId, usize, usize)> = Vec::new();
    let mut file_of: Vec<usize> = Vec::new();
    let mut chunk_at: HashMap<(StreamId, usize), usize> = HashMap::new();
    for (fi, (&file, ss)) in by_file.iter().enumerate() {
        file_index.insert(file, fi);
        for s in ss {
            let len = s.sig.len();
            for &(start, end) in &s.windows {
                for c in start / CHUNK..end.div_ceil(CHUNK) {
                    chunk_at.entry((s.sid, c)).or_insert_with(|| {
                        chunks.push((s.sid, c, len));
                        file_of.push(fi);
                        chunks.len() - 1
                    });
                }
            }
        }
    }

    // The votes each sample question keeps through its gate.
    let mut kept: Vec<Kept> = Vec::new();
    for (&sid, q) in &state.sample {
        let entries = &q.entries[ci];
        let tagged: Vec<[Option<Vote>; FOLD_GROUPS]> = (0..q.tokens.len())
            .map(|t| {
                from_fn(|g| {
                    let e = &entries[t * FOLD_GROUPS + g];
                    let file = *file_index.get(&e.stats.c1)?;
                    (!e.dirty).then_some(Vote {
                        token: t * FOLD_GROUPS + g,
                        file,
                        weight: e.stats.vote(n),
                    })
                })
            })
            .collect();
        for v in gate(&tagged, &[], NEEDLE_KEEP_FRAC) {
            let e = &entries[v.token];
            if e.stats.c1 == q.owner || v.weight <= 0.0 {
                continue;
            }
            kept.push(Kept {
                question: sid,
                token: v.token / FOLD_GROUPS,
                group: v.token % FOLD_GROUPS,
                weight: v.weight,
            });
        }
    }

    // Winning tokens not yet found, a file at a time.
    let mut wanted: BTreeMap<FileId, Vec<(StreamId, usize, usize)>> = BTreeMap::new();
    for k in &kept {
        let e = &state.sample[&k.question].entries[ci][k.token * FOLD_GROUPS + k.group];
        if e.win.is_none() {
            wanted
                .entry(e.stats.c1)
                .or_default()
                .push((k.question, k.token, k.group));
        }
    }
    let found: Vec<((StreamId, usize, usize), Option<(StreamId, u32)>)> = wanted
        .into_par_iter()
        .flat_map_iter(|(file, asks)| {
            let ss = by_file.get(&file).map_or(&[][..], Vec::as_slice);
            let decoded: Vec<(StreamId, &[(usize, usize)], Vec<WideQSig>)> = ss
                .iter()
                .map(|s| (s.sid, s.windows.as_slice(), s.sig.decode()))
                .collect();
            asks.into_iter()
                .map(|(qsid, t, g)| {
                    let token = &state.sample[&qsid].tokens[t];
                    ((qsid, t, g), best_token(&decoded, token, g))
                })
                .collect::<Vec<_>>()
        })
        .collect();
    for ((qsid, t, g), win) in found {
        if let Some(q) = state.sample.get_mut(&qsid) {
            q.entries[ci][t * FOLD_GROUPS + g].win = win;
        }
    }

    // Mass, and the outliers of it.
    let mut mass = vec![0f32; chunks.len()];
    let mut wins = vec![0u32; chunks.len()];
    let mut votes = 0usize;
    for k in &kept {
        let e = &state.sample[&k.question].entries[ci][k.token * FOLD_GROUPS + k.group];
        if let Some((wsid, pos)) = e.win {
            if let Some(&c) = chunk_at.get(&(wsid, pos as usize / CHUNK)) {
                mass[c] += k.weight;
                wins[c] += 1;
                votes += 1;
            }
        }
    }
    let cut_chunks = outliers(&mass, &file_of, comp.k(), KEEP);
    let weakest_cut_votes = cut_chunks.iter().map(|&c| wins[c]).min();
    let cut = cut_chunks
        .into_iter()
        .filter_map(|c| {
            let (sid, i, len) = chunks[c];
            let start = i * CHUNK;
            (start < len).then(|| (sid, start as u32, ((i + 1) * CHUNK).min(len) as u32))
        })
        .collect();
    CompetitionCut {
        cut,
        chunks: chunks.len(),
        votes,
        weakest_cut_votes,
    }
}

/// The token of `streams` — each `(stream, windows, decoded tokens)` — agreeing
/// best with `token` in fold group `group`, as `(stream, position)`: the first
/// strictly best in stream, window and position order.
fn best_token(
    streams: &[(StreamId, &[(usize, usize)], Vec<WideQSig>)],
    token: &WideQSig,
    group: usize,
) -> Option<(StreamId, u32)> {
    let gw = token.words.len() / FOLD_GROUPS;
    let words = group * gw..(group + 1) * gw;
    let q = token.words.get(words.clone())?;
    let mut best: Option<(u32, StreamId, usize)> = None;
    for (sid, windows, toks) in streams {
        for &(s, e) in *windows {
            let Some(span) = toks.get(s..e.min(toks.len())) else {
                continue;
            };
            for (i, tok) in span.iter().enumerate() {
                let Some(t) = tok.words.get(words.clone()) else {
                    continue;
                };
                let agree: u32 = q.iter().zip(t).map(|(a, b)| (!(a ^ b)).count_ones()).sum();
                if best.is_none_or(|(b, _, _)| agree > b) {
                    best = Some((agree, *sid, s + i));
                }
            }
        }
    }
    best.map(|(_, sid, pos)| (sid, pos as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(fill: u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: vec![fill; 24],
        }
    }

    /// **The best token is the first strictly best**, in stream, window and
    /// position order, in the asked fold group only.
    #[test]
    fn the_best_token_is_the_first_strictly_best() {
        let hub = sig(0xF0F0);
        let mut a: Vec<WideQSig> = (0..40).map(|_| sig(0)).collect();
        a[35] = hub.clone();
        let mut b: Vec<WideQSig> = (0..8).map(|_| sig(0)).collect();
        b[2] = hub.clone();
        let streams = vec![
            (StreamId(1), &[(0usize, 40usize)][..], a),
            (StreamId(2), &[(0, 8)][..], b),
        ];
        assert_eq!(best_token(&streams, &hub, 1), Some((StreamId(1), 35)));
        // Outside the windows scanned, a token is not found.
        let narrow = vec![(StreamId(1), &[(0usize, 30usize)][..], streams[0].2.clone())];
        assert_eq!(best_token(&narrow, &hub, 0), Some((StreamId(1), 0)));
    }
}
