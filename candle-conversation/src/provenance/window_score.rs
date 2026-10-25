//! Windowed provenance scoring (`docs/provenance_scoped_signatures.md` §2.3).
//!
//! Late fusion gives every probe token, per fold group, to one file — the file
//! holding the best-agreeing gallery token — weighted by `z × margin`, the
//! margin over the best token of any other file; a token every file matches
//! alike votes for nothing. Summed per file, a long file collects votes over
//! all its tokens, so size alone ranks it up. Here each vote is kept with the
//! gallery token it was won at and credited to the windows holding that token
//! — windows cut to the probe's own shape (`probe_windows`) — so a window
//! scores by how many of the probe's tokens it wins **together**, and a file by
//! one of [`FileAggregate`]'s rules over its windows. Measured against today's
//! per-file sum in `docs/results/provenance_gates_2026-10-25.md`.

use std::ops::Range;

use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use super::popcount::{agreement8_vpopcnt, has_vpopcnt, GROUP_WORDS};
use super::scan::NEEDLE_KEEP_FRAC;
use super::{heads_per_group, WideQSig, FOLD_GROUPS};

/// A gallery's signatures laid end to end in one array, `wpt` words a token —
/// what a scan streams. A `WideQSig` holds its words in a heap allocation of
/// its own, so a scan over a gallery of them chases a pointer per token per
/// probe token; flat, the gallery is one contiguous run the cache prefetches.
pub struct FlatSigs {
    words: Vec<u64>,
    wpt: usize,
    n_heads: u16,
}

impl FlatSigs {
    /// `tokens` laid flat, or `None` when they are empty or not all one width.
    pub fn new(tokens: &[WideQSig]) -> Option<FlatSigs> {
        let first = tokens.first()?;
        let wpt = first.words.len();
        if wpt == 0 || tokens.iter().any(|t| t.words.len() != wpt) {
            return None;
        }
        let mut words = Vec::with_capacity(tokens.len() * wpt);
        for t in tokens {
            words.extend_from_slice(&t.words);
        }
        Some(FlatSigs {
            words,
            wpt,
            n_heads: first.n_heads,
        })
    }

    pub fn len(&self) -> usize {
        self.words.len() / self.wpt
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }
}

/// Probe tokens scored together in one pass over a gallery slice: a gallery
/// past the last-level cache makes the scan memory-bound, so each pass over
/// the slice's words serves this many probe tokens.
const PROBE_TILE: usize = 32;

/// Gallery tokens a slice aims for. Slices cut at file boundaries and run in
/// parallel, so every core is busy whatever the probe's length.
const SLICE_TOKENS: usize = 32 * 1024;

/// One slice's answer for one probe token and fold group: its best file's
/// agreement, file and token, its runner-up file's agreement, and the
/// agreement sums the z is taken over.
#[derive(Clone, Copy)]
struct SliceBest {
    top1: u32,
    file: usize,
    token: u32,
    top2: u32,
    sum: u64,
    sumsq: u64,
}

impl SliceBest {
    const NONE: SliceBest = SliceBest {
        top1: 0,
        file: usize::MAX,
        token: u32::MAX,
        top2: 0,
        sum: 0,
        sumsq: 0,
    };

    /// Two slices' answers as one: the best two files of both, the sums
    /// added. Files never span slices, so the two best are distinct files.
    fn merge(self, o: SliceBest) -> SliceBest {
        let (hi, lo) = match (self.file, o.file) {
            (usize::MAX, _) => (o, self),
            (_, usize::MAX) => (self, o),
            _ if o.top1 > self.top1 => (o, self),
            _ => (self, o),
        };
        let runner = if lo.file == usize::MAX { 0 } else { lo.top1 };
        SliceBest {
            top2: hi.top2.max(runner),
            sum: self.sum + o.sum,
            sumsq: self.sumsq + o.sumsq,
            ..hi
        }
    }
}

/// One probe tile's pass over one gallery slice: every gallery token against
/// every probe token of the tile, per fold group — the agreement sums into
/// `sums` (`[probe token in tile][group]`) and each file's best agreement and
/// token into `best` (`[probe token in tile][file in slice][group]`).
struct TilePass<'a> {
    rows: &'a [u64],
    wpt: usize,
    gw: usize,
    /// The slice's first gallery token.
    start: usize,
    token_file: &'a [usize],
    n_files: usize,
    /// The slice's lowest file id, and how many file ids it spans.
    lo: usize,
    width: usize,
    tile: &'a [&'a WideQSig],
    sums: &'a mut [SliceBest],
    best: &'a mut [(u32, u32)],
}

impl TilePass<'_> {
    /// The pass with the widest popcount this CPU runs: one 512-bit XOR and
    /// vector popcount a group where the CPU has `VPOPCNTDQ` and the group is
    /// the locked eight words, else straight-line `POPCNT`s. Both count the
    /// same bits.
    fn run(&mut self) {
        #[cfg(target_arch = "x86_64")]
        if self.gw == GROUP_WORDS && has_vpopcnt() {
            // SAFETY: the CPU reported `avx512f` and `avx512vpopcntdq`.
            unsafe { self.run_vpopcnt() };
            return;
        }
        self.scan(agree);
    }

    /// # Safety
    ///
    /// The CPU must support `avx512f` and `avx512vpopcntdq`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    unsafe fn run_vpopcnt(&mut self) {
        // SAFETY: this function's own features, and `gw` is the eight words the
        // vector loads read.
        self.scan(|a, b| unsafe { agreement8_vpopcnt(a, b) });
    }

    #[inline(always)]
    fn scan(&mut self, agree: impl Fn(&[u64], &[u64]) -> u32) {
        let (gw, width) = (self.gw, self.width);
        for (i, row) in self.rows.chunks_exact(self.wpt).enumerate() {
            let t = self.start + i;
            let f = self.token_file[t];
            let local = (f < self.n_files).then(|| f - self.lo);
            for (qi, q) in self.tile.iter().enumerate() {
                for g in 0..FOLD_GROUPS {
                    let base = g * gw;
                    let ag = agree(&q.words[base..base + gw], &row[base..base + gw]);
                    let o = &mut self.sums[qi * FOLD_GROUPS + g];
                    o.sum += ag as u64;
                    o.sumsq += (ag as u64) * (ag as u64);
                    if let Some(lf) = local {
                        let b = &mut self.best[(qi * width + lf) * FOLD_GROUPS + g];
                        if b.1 == u32::MAX || ag > b.0 {
                            *b = (ag, t as u32);
                        }
                    }
                }
            }
        }
    }
}

/// Cut `0..token_file.len()` into ranges of about [`SLICE_TOKENS`], each
/// ending where the file changes, so no file spans two slices.
fn slices(token_file: &[usize]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < token_file.len() {
        let mut end = (start + SLICE_TOKENS).min(token_file.len());
        while end < token_file.len() && token_file[end] == token_file[end - 1] {
            end += 1;
        }
        out.push(start..end);
        start = end;
    }
    out
}

/// `popcount(!(a ^ b))` over equal-length word runs — the inner loop, kept
/// inline so the pass over the gallery compiles to straight-line `POPCNT`s.
#[inline(always)]
fn agree(a: &[u64], b: &[u64]) -> u32 {
    let mut n = 0u32;
    for (x, y) in a.iter().zip(b) {
        n += (!(x ^ y)).count_ones();
    }
    n
}

/// One probe token's late-fusion vote: the gallery token that won it, the file
/// that holds it, and its weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vote {
    pub token: usize,
    pub file: usize,
    pub weight: f32,
}

/// Late fusion's votes, kept with the token each was won at: per probe token
/// and fold group, the file holding the best-agreeing gallery token wins a
/// `z × margin` vote, the margin over the best token of any **other** file;
/// then the needle gate keeps the top quarter of probe tokens by vote
/// magnitude. Summed per file these are today's scores. `token_file[t]` is
/// gallery token `t`'s file; a token whose file is `n_files` or past it takes
/// no part. `group_weights` weighs the fold groups (empty: uniform).
///
pub fn late_fusion_votes(
    probe: &[WideQSig],
    gallery: &FlatSigs,
    token_file: &[usize],
    n_files: usize,
    group_weights: &[f32],
) -> Vec<Vote> {
    gate(
        &group_votes(probe, gallery, token_file, n_files),
        group_weights,
        NEEDLE_KEEP_FRAC,
    )
}

/// Each usable probe token's vote per fold group, unweighted and ungated:
/// `z × margin`, won by the file holding the best-agreeing gallery token, the
/// margin over the best token of any other file. [`gate`] weighs and gates
/// them, so one scan serves any weights and keep fraction.
///
/// The gallery is cut into slices at file boundaries ([`SLICE_TOKENS`]) that
/// run in parallel, each scoring the probe [`PROBE_TILE`] tokens to a pass
/// over its words and keeping, per probe token and group, only its best two
/// files and its agreement sums; the slices then merge exactly.
pub fn group_votes(
    probe: &[WideQSig],
    gallery: &FlatSigs,
    token_file: &[usize],
    n_files: usize,
) -> Vec<[Option<Vote>; FOLD_GROUPS]> {
    let n_tokens = gallery.len();
    let wph = gallery.wpt / gallery.n_heads.max(1) as usize;
    let hpg = heads_per_group(gallery.n_heads as usize);
    if wph == 0 || hpg == 0 || n_tokens == 0 || n_tokens != token_file.len() {
        return Vec::new();
    }
    let gw = hpg * wph;
    let need = FOLD_GROUPS * gw;
    let wpt = gallery.wpt;
    let n = n_tokens as f32;
    let usable: Vec<&WideQSig> = probe.iter().filter(|q| q.words.len() >= need).collect();
    let nq = usable.len();
    // [probe token][group], merged over every slice.
    let merged: Vec<SliceBest> = slices(token_file)
        .into_par_iter()
        .map(|range| {
            let mut out = vec![SliceBest::NONE; nq * FOLD_GROUPS];
            // The slice's files are a contiguous run of ids; a token of no
            // file takes part in the sums only.
            let files = token_file[range.clone()]
                .iter()
                .copied()
                .filter(|&f| f < n_files);
            let (lo, hi) = files.fold((usize::MAX, 0), |(lo, hi), f| (lo.min(f), hi.max(f)));
            let width = if lo == usize::MAX { 0 } else { hi - lo + 1 };
            for (t0, tile) in usable.chunks(PROBE_TILE).enumerate() {
                let q0 = t0 * PROBE_TILE;
                // [probe token in tile][file in slice][group] → (agreement, token).
                let mut best = vec![(0u32, u32::MAX); tile.len() * width * FOLD_GROUPS];
                let mut pass = TilePass {
                    rows: &gallery.words[range.start * wpt..range.end * wpt],
                    wpt,
                    gw,
                    start: range.start,
                    token_file,
                    n_files,
                    lo,
                    width,
                    tile,
                    sums: &mut out[q0 * FOLD_GROUPS..(q0 + tile.len()) * FOLD_GROUPS],
                    best: &mut best,
                };
                pass.run();
                for qi in 0..tile.len() {
                    for g in 0..FOLD_GROUPS {
                        let o = &mut out[(q0 + qi) * FOLD_GROUPS + g];
                        for lf in 0..width {
                            let (m, t) = best[(qi * width + lf) * FOLD_GROUPS + g];
                            if t == u32::MAX {
                                continue;
                            }
                            if o.file == usize::MAX || m > o.top1 {
                                if o.file != usize::MAX {
                                    o.top2 = o.top2.max(o.top1);
                                }
                                o.top1 = m;
                                o.file = lo + lf;
                                o.token = t;
                            } else if m > o.top2 {
                                o.top2 = m;
                            }
                        }
                    }
                }
            }
            out
        })
        .reduce(
            || vec![SliceBest::NONE; nq * FOLD_GROUPS],
            |a, b| a.into_iter().zip(b).map(|(x, y)| x.merge(y)).collect(),
        );
    (0..nq)
        .map(|qi| {
            let mut votes = [None; FOLD_GROUPS];
            for (g, vote) in votes.iter_mut().enumerate() {
                let b = merged[qi * FOLD_GROUPS + g];
                if b.file == usize::MAX {
                    continue;
                }
                let mean = b.sum as f32 / n;
                let var = (b.sumsq as f32 / n - mean * mean).max(1e-6);
                let z = ((b.top1 as f32 - mean) / var.sqrt()).max(0.0);
                let margin = b.top1.saturating_sub(b.top2) as f32;
                *vote = Some(Vote {
                    token: b.token as usize,
                    file: b.file,
                    weight: z * margin,
                });
            }
            votes
        })
        .collect()
}

/// [`group_votes`] weighed by `group_weights` (empty: uniform) and gated: the
/// probe tokens whose summed vote is in the top `keep` fraction keep their
/// votes, the rest are dropped — late fusion's needle gate at
/// [`NEEDLE_KEEP_FRAC`].
pub fn gate(
    per_token: &[[Option<Vote>; FOLD_GROUPS]],
    group_weights: &[f32],
    keep: f32,
) -> Vec<Vote> {
    let weighed: Vec<Vec<Vote>> = per_token
        .iter()
        .map(|votes| {
            votes
                .iter()
                .enumerate()
                .filter_map(|(g, v)| {
                    let w = group_weights.get(g).copied().unwrap_or(1.0);
                    v.map(|v| Vote {
                        weight: v.weight * w,
                        ..v
                    })
                })
                .collect()
        })
        .collect();
    let mags: Vec<f32> = weighed
        .iter()
        .map(|v| v.iter().map(|x| x.weight).sum())
        .collect();
    if mags.is_empty() {
        return Vec::new();
    }
    let n_keep = ((keep * mags.len() as f32).ceil() as usize).clamp(1, mags.len());
    let mut sorted = mags.clone();
    sorted.sort_unstable_by(|a, b| b.total_cmp(a));
    let thresh = sorted[n_keep - 1];
    weighed
        .into_iter()
        .zip(mags)
        .filter(|(_, m)| *m >= thresh)
        .flat_map(|(v, _)| v)
        .collect()
}

/// Each window's tally of `votes`: a vote counts for every window holding the
/// token it was won at. `windows[w]` are window `w`'s ranges of a gallery of
/// `n_tokens` tokens.
pub fn window_tally(votes: &[Vote], windows: &[Vec<Range<usize>>], n_tokens: usize) -> Vec<f32> {
    let mut at = vec![0.0f32; n_tokens];
    for v in votes {
        if v.token < n_tokens {
            at[v.token] += v.weight;
        }
    }
    // Prefix sums turn each range's tally into two reads.
    let mut prefix = Vec::with_capacity(n_tokens + 1);
    let mut run = 0.0f64;
    prefix.push(run);
    for x in &at {
        run += *x as f64;
        prefix.push(run);
    }
    windows
        .iter()
        .map(|ranges| {
            ranges
                .iter()
                .map(|r| {
                    let e = r.end.min(n_tokens);
                    let s = r.start.min(e);
                    (prefix[e] - prefix[s]) as f32
                })
                .sum()
        })
        .collect()
}

/// How a file's windows make its one score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FileAggregate {
    /// Its best window.
    BestWindow,
    /// Its best window, less `strength` times what the best of that many
    /// windows reaches by chance: `σ · √(2 ln W)` for `W` windows, `σ` the
    /// spread of every window's score for this probe.
    CountNormalised { strength: f32 },
    /// Its best identity window — its first turn's request and its last turn's
    /// closing summary.
    Identity,
    /// The better of its identity score and its count-normalised best window
    /// at `strength`, each standardised over the files that have one.
    IdentityOrCount { strength: f32 },
}

impl FileAggregate {
    pub const ALL: [FileAggregate; 6] = [
        FileAggregate::BestWindow,
        FileAggregate::CountNormalised { strength: 1.0 },
        FileAggregate::CountNormalised { strength: 0.5 },
        FileAggregate::Identity,
        FileAggregate::IdentityOrCount { strength: 1.0 },
        FileAggregate::IdentityOrCount { strength: 0.5 },
    ];

    pub fn name(self) -> String {
        match self {
            FileAggregate::BestWindow => "best-window".into(),
            FileAggregate::CountNormalised { strength } => format!("count-normalised x{strength}"),
            FileAggregate::Identity => "identity".into(),
            FileAggregate::IdentityOrCount { strength } => {
                format!("identity-or-count x{strength}")
            }
        }
    }
}

/// `scores` standardised over its finite entries; a non-finite entry stays.
fn standardised(scores: &[f32]) -> Vec<f32> {
    let finite: Vec<f64> = scores
        .iter()
        .filter(|s| s.is_finite())
        .map(|&s| s as f64)
        .collect();
    if finite.is_empty() {
        return scores.to_vec();
    }
    let n = finite.len() as f64;
    let mean = finite.iter().sum::<f64>() / n;
    let sd = (finite.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n)
        .sqrt()
        .max(1e-9);
    scores
        .iter()
        .map(|&s| match s.is_finite() {
            true => ((s as f64 - mean) / sd) as f32,
            false => s,
        })
        .collect()
}

/// Each file's score under `mode`: `scores[w]` is window `w`'s, `file[w]` its
/// file, `identity[w]` whether it is one of its file's identity windows, and
/// `live[w]` whether it takes part. A file with no window taking part scores
/// `f32::NEG_INFINITY`.
pub fn file_scores(
    scores: &[f32],
    file: &[usize],
    identity: &[bool],
    live: &[bool],
    n_files: usize,
    mode: FileAggregate,
) -> Vec<f32> {
    if let FileAggregate::IdentityOrCount { strength } = mode {
        let by = |m| standardised(&file_scores(scores, file, identity, live, n_files, m));
        let ident = by(FileAggregate::Identity);
        let count = by(FileAggregate::CountNormalised { strength });
        return ident.iter().zip(&count).map(|(a, b)| a.max(*b)).collect();
    }
    let mut best = vec![f32::NEG_INFINITY; n_files];
    let mut count = vec![0usize; n_files];
    let (mut sum, mut sumsq, mut n) = (0f64, 0f64, 0f64);
    for w in 0..scores.len() {
        if !live[w] {
            continue;
        }
        sum += scores[w] as f64;
        sumsq += (scores[w] as f64).powi(2);
        n += 1.0;
        if mode == FileAggregate::Identity && !identity[w] {
            continue;
        }
        let f = file[w];
        count[f] += 1;
        best[f] = best[f].max(scores[w]);
    }
    let FileAggregate::CountNormalised { strength } = mode else {
        return best;
    };
    if n == 0.0 {
        return best;
    }
    let sd = (sumsq / n - (sum / n).powi(2)).max(0.0).sqrt() as f32;
    best.iter()
        .zip(&count)
        .map(|(&b, &c)| match c {
            0 => f32::NEG_INFINITY,
            c => b - strength * sd * (2.0 * (c as f32).ln()).max(0.0).sqrt(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::score_slots_weighted;

    /// A deterministic pseudo-random folded signature: 3 groups × 4 heads ×
    /// 2 words.
    fn sig(seed: u64) -> WideQSig {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let words = (0..24)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            })
            .collect();
        WideQSig { n_heads: 12, words }
    }

    /// **Summed per file, the votes are today's late fusion.** Same gallery,
    /// one case a file: the per-file sums equal `score_slots_weighted`'s.
    #[test]
    fn votes_summed_per_file_are_late_fusion() {
        let tokens: Vec<WideQSig> = (0..300).map(sig).collect();
        let token_file: Vec<usize> = (0..300).map(|t| t / 60).collect();
        let probe: Vec<WideQSig> = tokens[130..170]
            .iter()
            .cloned()
            .chain((900..920).map(sig))
            .collect();
        let flat = FlatSigs::new(&tokens).unwrap();
        let votes = late_fusion_votes(&probe, &flat, &token_file, 5, &[]);
        let mut per_file = vec![0.0f32; 5];
        for v in &votes {
            assert_eq!(token_file[v.token], v.file);
            per_file[v.file] += v.weight;
        }
        let windows: Vec<&[WideQSig]> = (0..5).map(|f| &tokens[f * 60..f * 60 + 60]).collect();
        let slots: Vec<usize> = (0..5).collect();
        let want = score_slots_weighted(&probe, &windows, &slots, 5, &[]);
        for (a, b) in per_file.iter().zip(&want) {
            assert!(
                (a - b).abs() <= 1e-3 * (1.0 + b.abs()),
                "{per_file:?} vs {want:?}"
            );
        }
    }

    /// A token whose file is past the last takes no part: the votes go to the
    /// files that remain.
    #[test]
    fn a_token_of_no_file_wins_nothing() {
        let tokens: Vec<WideQSig> = (0..120).map(sig).collect();
        let token_file: Vec<usize> = (0..120).map(|t| if t < 60 { 2 } else { 1 }).collect();
        let probe: Vec<WideQSig> = tokens[10..30].to_vec();
        let flat = FlatSigs::new(&tokens).unwrap();
        let votes = late_fusion_votes(&probe, &flat, &token_file, 2, &[]);
        assert!(!votes.is_empty());
        assert!(votes.iter().all(|v| v.file == 1 && v.token >= 60));
    }

    /// **The slices change nothing.** A gallery many slices long votes exactly
    /// as late fusion over it whole, ties included.
    #[test]
    fn a_sliced_gallery_votes_as_one_scan() {
        let n = SLICE_TOKENS * 3 + 517;
        let tokens: Vec<WideQSig> = (0..n as u64).map(|s| sig(s % 4093)).collect();
        // Files of uneven length, one spanning a slice cut, and a token of no
        // file in the middle.
        let mut token_file: Vec<usize> = (0..n).map(|t| (t * 3 / SLICE_TOKENS).min(20)).collect();
        token_file[SLICE_TOKENS + 5] = usize::MAX;
        let probe: Vec<WideQSig> = (0..40u64).map(|s| sig(s * 97 % 4093)).collect();
        let flat = FlatSigs::new(&tokens).unwrap();
        let votes = late_fusion_votes(&probe, &flat, &token_file, 21, &[]);
        let mut per_file = vec![0.0f32; 21];
        for v in &votes {
            per_file[v.file] += v.weight;
        }
        let refs: Vec<&WideQSig> = tokens.iter().collect();
        let case: Vec<u32> = token_file
            .iter()
            .map(|&f| if f < 21 { f as u32 } else { u32::MAX })
            .collect();
        let want = crate::provenance::score_provenance_late_fusion(&probe, &refs, &case, 21);
        for (a, b) in per_file.iter().zip(&want) {
            assert!(
                (a - b).abs() <= 1e-3 * (1.0 + b.abs()),
                "{per_file:?} vs {want:?}"
            );
        }
        assert!(slices(&token_file).len() >= 3);
    }

    /// Laid flat only when every token is one width.
    #[test]
    fn a_gallery_lies_flat_only_at_one_width() {
        let tokens: Vec<WideQSig> = (0..3).map(sig).collect();
        let flat = FlatSigs::new(&tokens).unwrap();
        assert_eq!(flat.len(), 3);
        assert_eq!(&flat.words[24..48], &tokens[1].words[..]);
        let mut ragged = tokens.clone();
        ragged[2].words.pop();
        assert!(FlatSigs::new(&ragged).is_none());
        assert!(FlatSigs::new(&[]).is_none());
    }

    /// A vote counts for every window holding its token, and for no other.
    #[test]
    fn a_window_tallies_the_votes_won_inside_it() {
        let votes = [
            Vote {
                token: 3,
                file: 0,
                weight: 2.0,
            },
            Vote {
                token: 10,
                file: 0,
                weight: 5.0,
            },
            Vote {
                token: 3,
                file: 0,
                weight: 1.0,
            },
        ];
        let windows = vec![vec![0..5], vec![2..12], vec![0..2, 9..11], vec![12..20]];
        assert_eq!(window_tally(&votes, &windows, 20), vec![3.0, 8.0, 5.0, 0.0]);
    }

    /// Best window takes the maximum; count normalisation charges a file for
    /// how many windows it drew; identity reads only identity windows; a
    /// window not taking part counts for nothing.
    #[test]
    fn each_aggregate_reads_the_windows_its_rule_names() {
        // File 0: one window of 5. File 1: four windows, best 5.5.
        let scores = [5.0, 1.0, 2.0, 5.5, 3.0];
        let file = [0, 1, 1, 1, 1];
        let identity = [true, true, false, false, false];
        let live = [true; 5];
        let agg = |mode, live: &[bool]| file_scores(&scores, &file, &identity, live, 2, mode);
        assert_eq!(agg(FileAggregate::BestWindow, &live), vec![5.0, 5.5]);
        assert_eq!(agg(FileAggregate::Identity, &live), vec![5.0, 1.0]);
        let norm = agg(FileAggregate::CountNormalised { strength: 1.0 }, &live);
        assert_eq!(norm[0], 5.0, "one window draws no extra chance");
        assert!(norm[1] < 5.0, "four windows pay for their draws: {norm:?}");
        let half = agg(FileAggregate::CountNormalised { strength: 0.5 }, &live);
        assert!(half[1] > norm[1] && half[1] < 5.5, "{half:?}");
        // Identity standardised: [1, -1]; count standardised: file 0 above
        // file 1. The better of the two, per file.
        let both = agg(FileAggregate::IdentityOrCount { strength: 1.0 }, &live);
        assert_eq!(both[0], 1.0);
        assert!(both[1] < 1.0, "{both:?}");
        let masked = [true, true, true, false, true];
        assert_eq!(agg(FileAggregate::BestWindow, &masked), vec![5.0, 3.0]);
        let none = agg(FileAggregate::BestWindow, &[false; 5]);
        assert!(none.iter().all(|s| *s == f32::NEG_INFINITY));
    }
}
