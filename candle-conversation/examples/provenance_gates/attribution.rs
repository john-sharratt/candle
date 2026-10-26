//! Where each vote was won: the gallery chunk holding the token that won it.
//!
//! The daemon's scan names the winning file and the vote of each probe token
//! and fold group; the winning token is found here by the kernel's own measure
//! — sign agreement — over that file's tokens alone, and credited to the
//! 32-token chunk holding it. A file's tokens are its covered turns and, when
//! the gallery holds them, its calibration questions.

use std::iter::repeat_n;

use candle_conversation::persistence::content_hash::turn_stream_id;
use candle_conversation::provenance::window_score::{gate, Vote};
use candle_conversation::provenance::{WideQSig, FOLD_GROUPS};
use rayon::prelude::*;

use crate::corpus::{File, CHUNK};
use crate::remote_gallery::{Questions, RemoteGallery};

/// Probes per launch.
pub const BATCH: usize = 256;

/// Words per fold group of a token.
const GROUP_WORDS: usize = 8;

/// The needle gate's keep fraction, today's.
pub const KEEP: f32 = 0.25;

/// Every gallery turn's signatures, per file, and each chunk's global id.
pub struct Chunks<'a> {
    /// `(signatures, first chunk id)` per gallery turn, by file.
    turns: Vec<Vec<(&'a [WideQSig], usize)>>,
    /// The file each chunk belongs to.
    pub file_of: Vec<usize>,
    /// Each chunk's turn, by stream id, and its `[start, end)` tokens there.
    pub range_of: Vec<(u64, usize, usize)>,
    /// Each chunk's file and turn there; `None` for a calibration question.
    pub turn_of: Vec<Option<(usize, usize)>>,
}

impl<'a> Chunks<'a> {
    /// Every file's covered turns, and each question `question(file, index)`
    /// keeps.
    pub fn of(files: &'a [File], question: &dyn Fn(usize, usize) -> bool) -> Chunks<'a> {
        let mut chunks = Chunks {
            turns: Vec::with_capacity(files.len()),
            file_of: Vec::new(),
            range_of: Vec::new(),
            turn_of: Vec::new(),
        };
        for (fi, f) in files.iter().enumerate() {
            let mut turns = Vec::new();
            for (ti, t) in f.turns.iter().enumerate().filter(|(_, t)| t.covered()) {
                let sid = turn_stream_id(f.timeline, ti as u32).0;
                turns.push(chunks.add(fi, sid, &t.sigs, Some((fi, ti))));
            }
            for (qi, q) in f.questions.iter().enumerate() {
                if question(fi, qi) {
                    let sid = turn_stream_id(q.timeline, q.turn).0;
                    turns.push(chunks.add(fi, sid, &q.sigs, None));
                }
            }
            chunks.turns.push(turns);
        }
        chunks
    }

    fn add(
        &mut self,
        file: usize,
        sid: u64,
        sigs: &'a [WideQSig],
        turn: Option<(usize, usize)>,
    ) -> (&'a [WideQSig], usize) {
        let first = self.file_of.len();
        let n = sigs.len().div_ceil(CHUNK);
        self.file_of.extend(repeat_n(file, n));
        self.turn_of.extend(repeat_n(turn, n));
        self.range_of
            .extend((0..n).map(|c| (sid, c * CHUNK, ((c + 1) * CHUNK).min(sigs.len()))));
        (sigs, first)
    }

    /// The chunk of `file` holding the token that agrees best with `probe`
    /// in fold group `group`.
    fn winner(&self, file: usize, probe: &WideQSig, group: usize) -> Option<usize> {
        let words = group * GROUP_WORDS..(group + 1) * GROUP_WORDS;
        let q = probe.words.get(words.clone())?;
        let mut best: Option<(u32, usize)> = None;
        for (sigs, first) in &self.turns[file] {
            for (i, s) in sigs.iter().enumerate() {
                let Some(g) = s.words.get(words.clone()) else {
                    continue;
                };
                let agree: u32 = q.iter().zip(g).map(|(a, b)| (!(a ^ b)).count_ones()).sum();
                if best.is_none_or(|(b, _)| agree > b) {
                    best = Some((agree, first + i / CHUNK));
                }
            }
        }
        best.map(|(_, c)| c)
    }
}

/// One kept vote: the file it went to, its weight, and the chunk it was won
/// at.
pub struct Won {
    pub file: usize,
    pub weight: f32,
    pub chunk: usize,
}

/// Every probe's kept votes, attributed to chunks, against the gallery
/// `questions` and `masked_question` describe — which must hold exactly the
/// questions `chunks` does.
pub fn attributed(
    gpu: &RemoteGallery,
    chunks: &Chunks,
    probes: &[(usize, &[WideQSig])],
    questions: Questions,
    masked_question: &dyn Fn(usize, usize) -> bool,
) -> anyhow::Result<Vec<Vec<Won>>> {
    let n_files = chunks.turns.len();
    let mut out: Vec<Vec<Won>> = Vec::with_capacity(probes.len());
    for batch in probes.chunks(BATCH) {
        let sigs: Vec<&[WideQSig]> = batch.iter().map(|(_, s)| *s).collect();
        let votes = gpu.votes(&sigs, &|_, _| false, questions, masked_question)?;
        let kept: Vec<Vec<Vote>> = votes
            .iter()
            .map(|per_token| {
                // The token slot carries `probe token × groups + group` through
                // the gate, so each kept vote can be attributed.
                let tagged: Vec<[Option<Vote>; FOLD_GROUPS]> = per_token
                    .iter()
                    .enumerate()
                    .map(|(t, groups)| {
                        let mut g = *groups;
                        for (gi, v) in g.iter_mut().enumerate() {
                            if let Some(v) = v {
                                v.token = t * FOLD_GROUPS + gi;
                            }
                        }
                        g
                    })
                    .collect();
                gate(&tagged, &[], KEEP)
            })
            .collect();
        let batch_won: Vec<Vec<Won>> = batch
            .par_iter()
            .zip(kept.par_iter())
            .map(|((_, sigs), kept)| {
                kept.iter()
                    .filter(|v| v.file < n_files)
                    .filter_map(|v| {
                        let (t, g) = (v.token / FOLD_GROUPS, v.token % FOLD_GROUPS);
                        let chunk = chunks.winner(v.file, &sigs[t], g)?;
                        Some(Won {
                            file: v.file,
                            weight: v.weight,
                            chunk,
                        })
                    })
                    .collect()
            })
            .collect();
        out.extend(batch_won);
    }
    Ok(out)
}

/// Each chunk's vote mass on background probes not about its file.
pub fn background_mass(
    chunks: &Chunks,
    background: &[(usize, &[WideQSig])],
    won: &[Vec<Won>],
) -> Vec<f32> {
    let mut mass = vec![0f32; chunks.file_of.len()];
    for ((owner, _), won) in background.iter().zip(won) {
        for v in won {
            if chunks.file_of[v.chunk] != *owner {
                mass[v.chunk] += v.weight;
            }
        }
    }
    mass
}
