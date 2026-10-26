//! Learning a belief group's anomalous-chunk exclusion from its files'
//! calibration questions (`crate::provenance::chunk_exclusion`).
//!
//! The questions are scanned raw against the group's content alone — every
//! file one case, no calibration in the gallery, nothing cut, so the
//! exclusion never learns from a gallery it already shaped. Each kept vote
//! (the needle gate's, today's) for a file that is not the question's own is
//! credited to the chunk holding the gallery token that won it: the best
//! agreement in the vote's fold group among that file's own tokens, by the
//! kernel's measure. The mass a chunk gathers is what it wins on other files'
//! questions; its outliers are cut.
//!
//! The attribution reads only the winning file's tokens, decoded once per
//! file, files in parallel, outside every lock — host work over a sample of
//! the questions, run once an ingest pass, never on the forward path.

use std::array::from_fn;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::iter::repeat_n;

use rayon::prelude::*;

use super::belief_files::{assemble_file_scans, FileScan};
use super::group_votes::GroupVotes;
use super::ids::{GroupId, TimelineId};
use crate::persistence::content_hash::turn_stream_id;
use crate::persistence::streams::StreamId;
use crate::projection::Conversation;
use crate::provenance::chunk_exclusion::anomaly::{outliers, K, KEEP, K_CALIBRATION};
use crate::provenance::chunk_exclusion::{ChunkExclusion, CHUNK};
use crate::provenance::gallery_arena::TokenVotes;
use crate::provenance::scan::NEEDLE_KEEP_FRAC;
use crate::provenance::window_score::{gate, Vote};
use crate::provenance::SigWindow;
use crate::provenance::{WideQSig, FOLD_GROUPS};

/// Every chunk of the group's files and the background mass it won.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct ChunkMass {
    /// `(turn, chunk index, turn length)` per chunk, file by file.
    pub chunks: Vec<(StreamId, usize, usize)>,
    /// Each chunk's file, an index into the scanned files.
    pub file_of: Vec<usize>,
    pub mass: Vec<f32>,
}

impl ChunkMass {
    /// The chunks the mass calls for cutting: its outliers at `k`, each file
    /// keeping its share.
    pub fn outliers(&self, k: f32) -> impl Iterator<Item = (StreamId, usize, usize)> + '_ {
        outliers(&self.mass, &self.file_of, k, KEEP)
            .into_iter()
            .map(|c| self.chunks[c])
    }
}

/// Which of a file's windows a scan saw: its own content, or its
/// calibrations' — the two competitions a file-scored group is scored in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Competition {
    Content,
    Calibration,
}

impl Competition {
    /// `f`'s windows in this competition, `(arc, start, end)`.
    fn windows(self, f: &FileScan) -> Vec<(usize, usize, usize)> {
        match self {
            Competition::Content => f.windows.iter().map(|&(k, s, e, _)| (k, s, e)).collect(),
            Competition::Calibration => f.cal_windows.clone(),
        }
    }

    /// The robust deviations a chunk of this competition is cut at.
    fn k(self) -> f32 {
        match self {
            Competition::Content => K,
            Competition::Calibration => K_CALIBRATION,
        }
    }
}

/// What one learning pass did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ExclusionReport {
    /// Files of the group scanned, and those with calibration questions.
    pub files: usize,
    pub calibrated: usize,
    /// Questions asked, and per competition the chunks weighed and cut.
    pub probes: usize,
    pub chunks: usize,
    pub cut: usize,
    pub calibration_chunks: usize,
    pub calibration_cut: usize,
}

impl Conversation {
    /// Learn `group`'s anomalous-chunk exclusion from up to `per_file` of each
    /// file's calibration questions and set it for every live scan.
    ///
    /// Each competition is learned against itself: the questions are scanned
    /// raw against the group's content alone, and again against its
    /// calibrations alone — every file's own turns and the asking questions
    /// masked — with nothing cut, so the exclusion never learns from a
    /// gallery it already shaped, nor from a question finding itself. `scan`
    /// is the scheduler's [`Self::scan_group_votes`] — `(calibrations,
    /// masked turns, probes)` — and answers in its own file order, which the
    /// cases are read back through. A group with no calibrated file learns an
    /// empty exclusion.
    pub fn learn_chunk_exclusion(
        &self,
        group: GroupId,
        per_file: usize,
        scan: impl Fn(bool, HashSet<StreamId>, Vec<Vec<WideQSig>>) -> candle::Result<GroupVotes>,
    ) -> candle::Result<ExclusionReport> {
        let (files, probes) = {
            let sub = self.inner.read().unwrap();
            let timelines: Vec<TimelineId> = sub.active_timelines_for_group(group).collect();
            let (files, _) = assemble_file_scans(&sub, timelines);
            let width = files
                .iter()
                .find_map(|f| f.arcs_kept.first())
                .map_or(0, SigWindow::width);
            let mut probes: Vec<(usize, StreamId, Vec<WideQSig>)> = Vec::new();
            for (fi, f) in files.iter().enumerate() {
                let questions = sub.calibrations_of(f.timeline).flat_map(|cal| {
                    (0..sub.turn_count(cal)).map(move |i| turn_stream_id(cal.raw(), i))
                });
                for sid in questions.take(per_file) {
                    let Some(window) = sub.wide_sig_window(sid) else {
                        continue;
                    };
                    let sigs: Vec<WideQSig> = window
                        .decode()
                        .into_iter()
                        .filter(|t| t.words.len() == width)
                        .collect();
                    if !sigs.is_empty() {
                        probes.push((fi, sid, sigs));
                    }
                }
            }
            (files, probes)
        };
        let mut report = ExclusionReport {
            files: files.len(),
            calibrated: files.iter().filter(|f| !f.cal_windows.is_empty()).count(),
            probes: probes.len(),
            ..Default::default()
        };
        let exclusion = if probes.is_empty() {
            ChunkExclusion::default()
        } else {
            let sigs: Vec<Vec<WideQSig>> = probes.iter().map(|(_, _, s)| s.clone()).collect();
            let owned: Vec<(usize, &[WideQSig])> = probes
                .iter()
                .map(|(fi, _, s)| (*fi, s.as_slice()))
                .collect();
            // The calibration scan's gallery: every calibration question but
            // the asking ones — each would otherwise find itself, and its vote
            // go to its own file — and none of the files' own turns.
            let not_calibration: HashSet<StreamId> = files
                .iter()
                .flat_map(|f| {
                    f.arc_turn
                        .iter()
                        .map(|t| turn_stream_id(f.timeline.raw(), t.0))
                })
                .chain(probes.iter().map(|(_, sid, _)| *sid))
                .collect();
            let content = scan(false, HashSet::new(), sigs.clone())?;
            let content = chunk_mass(
                &files,
                &owned,
                &in_file_order(&files, &content),
                Competition::Content,
            );
            let calibration = scan(true, not_calibration, sigs)?;
            let calibration = chunk_mass(
                &files,
                &owned,
                &in_file_order(&files, &calibration),
                Competition::Calibration,
            );
            report.chunks = content.mass.len();
            report.calibration_chunks = calibration.mass.len();
            let cut: Vec<(StreamId, usize, usize)> =
                content.outliers(Competition::Content.k()).collect();
            let cal_cut: Vec<(StreamId, usize, usize)> =
                calibration.outliers(Competition::Calibration.k()).collect();
            report.cut = cut.len();
            report.calibration_cut = cal_cut.len();
            ChunkExclusion::of_chunks(cut.into_iter().chain(cal_cut))
        };
        self.inner
            .write()
            .unwrap()
            .set_chunk_exclusion(group, exclusion);
        Ok(report)
    }
}

/// `scanned`'s votes with each case named by its file's index in `files`; a
/// case for a file not among them is no vote.
fn in_file_order(files: &[FileScan], scanned: &GroupVotes) -> TokenVotes {
    let index: HashMap<TimelineId, usize> = files
        .iter()
        .enumerate()
        .map(|(i, f)| (f.timeline, i))
        .collect();
    let local: Vec<i32> = scanned
        .files
        .iter()
        .map(|tl| index.get(tl).map_or(-1, |&i| i as i32))
        .collect();
    let mut votes = scanned.votes.clone();
    for c in &mut votes.case {
        if *c >= 0 {
            *c = local.get(*c as usize).copied().unwrap_or(-1);
        }
    }
    votes
}

/// One kept vote to attribute: the probe token that cast it, its fold group,
/// and its weight.
struct Cast<'a> {
    token: &'a WideQSig,
    group: usize,
    weight: f32,
}

/// The mass every chunk of `files`' windows in `competition` won on `probes`
/// — each `(owner file, full-width tokens)` — from their raw votes `votes`,
/// scanned over that competition with `files`' order as the cases.
pub(crate) fn chunk_mass(
    files: &[FileScan],
    probes: &[(usize, &[WideQSig])],
    votes: &TokenVotes,
    competition: Competition,
) -> ChunkMass {
    // Kept votes for a file not the question's own, by the file they went to.
    let mut by_file: Vec<Vec<Cast>> = (0..files.len()).map(|_| Vec::new()).collect();
    for (p, &(owner, sigs)) in probes.iter().enumerate() {
        let t0 = votes.first_token(p);
        let n = votes.per_probe.get(p).copied().unwrap_or(0).min(sigs.len());
        // The token slot carries `probe token × groups + group` through the
        // gate, so each kept vote can be attributed.
        let tagged: Vec<[Option<Vote>; FOLD_GROUPS]> = (0..n)
            .map(|t| {
                from_fn(|g| {
                    votes.at(t0 + t, g, 0).map(|(file, weight)| Vote {
                        token: t * FOLD_GROUPS + g,
                        file,
                        weight,
                    })
                })
            })
            .collect();
        for v in gate(&tagged, &[], NEEDLE_KEEP_FRAC) {
            if v.file == owner || v.file >= files.len() {
                continue;
            }
            by_file[v.file].push(Cast {
                token: &sigs[v.token / FOLD_GROUPS],
                group: v.token % FOLD_GROUPS,
                weight: v.weight,
            });
        }
    }
    let per_file: Vec<(Vec<(StreamId, usize, usize)>, Vec<f32>)> = files
        .par_iter()
        .zip(by_file.par_iter())
        .map(|(f, casts)| file_mass(f, &competition.windows(f), casts))
        .collect();
    let mut out = ChunkMass::default();
    for (fi, (chunks, mass)) in per_file.into_iter().enumerate() {
        out.file_of.extend(repeat_n(fi, chunks.len()));
        out.chunks.extend(chunks);
        out.mass.extend(mass);
    }
    out
}

/// Every chunk of `windows` — `f`'s, `(arc, start, end)` — and the mass
/// `casts` credit each.
fn file_mass(
    f: &FileScan,
    windows: &[(usize, usize, usize)],
    casts: &[Cast],
) -> (Vec<(StreamId, usize, usize)>, Vec<f32>) {
    let decoded: Vec<Vec<WideQSig>> = f.arcs_kept.iter().map(|a| a.decode()).collect();
    let mut index: BTreeMap<(StreamId, usize), usize> = BTreeMap::new();
    let mut chunks = Vec::new();
    for &(k, s, e) in windows {
        let len = decoded[k].len();
        for c in s / CHUNK..e.div_ceil(CHUNK) {
            index.entry((f.arc_sids[k], c)).or_insert_with(|| {
                chunks.push((f.arc_sids[k], c, len));
                chunks.len() - 1
            });
        }
    }
    let mut mass = vec![0f32; chunks.len()];
    for cast in casts {
        if let Some((k, pos)) = best_token(windows, &decoded, cast) {
            if let Some(&c) = index.get(&(f.arc_sids[k], pos / CHUNK)) {
                mass[c] += cast.weight;
            }
        }
    }
    (chunks, mass)
}

/// The arc and position of the token of `windows` agreeing best with `cast`'s
/// probe token in its fold group.
fn best_token(
    windows: &[(usize, usize, usize)],
    decoded: &[Vec<WideQSig>],
    cast: &Cast,
) -> Option<(usize, usize)> {
    let gw = cast.token.words.len() / FOLD_GROUPS;
    let words = cast.group * gw..(cast.group + 1) * gw;
    let q = cast.token.words.get(words.clone())?;
    let mut best: Option<(u32, usize, usize)> = None;
    for &(k, s, e) in windows {
        for (i, tok) in decoded[k][s..e].iter().enumerate() {
            let Some(g) = tok.words.get(words.clone()) else {
                continue;
            };
            let agree: u32 = q.iter().zip(g).map(|(a, b)| (!(a ^ b)).count_ones()).sum();
            if best.is_none_or(|(b, _, _)| agree > b) {
                best = Some((agree, k, s + i));
            }
        }
    }
    best.map(|(_, k, pos)| (k, pos))
}

// The test file is one turn of one exchange, which the lint takes for a
// mistyped range.
#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::projection::ids::TurnIndex;
    use crate::provenance::encode_wide_sigs;

    /// A signature whose 24 words are all `fill`.
    fn sig(fill: u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: vec![fill; 24],
        }
    }

    /// A file of one turn of `tokens`.
    fn file(raw: u64, tokens: &[WideQSig]) -> FileScan {
        let window = SigWindow::new(Arc::from(encode_wide_sigs(tokens))).expect("a stored window");
        FileScan {
            timeline: TimelineId::for_test(raw),
            arc_turn: vec![TurnIndex(0)],
            ex_ranges: vec![0..1],
            n_slots: 1,
            arcs_kept: vec![window],
            arc_sids: vec![turn_stream_id(raw, 0)],
            windows: vec![(0, 0, tokens.len(), 0)],
            cal_windows: Vec::new(),
            ex_tokens: vec![tokens.len()],
        }
    }

    /// **A vote is credited to the chunk that won it, and only on another
    /// file's question.** File 0 holds a 40-token turn whose token 35 — chunk
    /// 1 — is the probes' token exactly; file 1 is all zeros. Two probes of
    /// one token each vote for file 0 in every group: one asked by file 1
    /// (credited, 3 groups × 2.0), one by file 0 itself (not).
    #[test]
    fn a_vote_is_credited_to_the_chunk_that_won_it() {
        let hub = sig(0xF0F0_F0F0_F0F0_F0F0);
        let mut a: Vec<WideQSig> = (0..40).map(|_| sig(0)).collect();
        a[35] = hub.clone();
        let zeros: Vec<WideQSig> = (0..8).map(|_| sig(0)).collect();
        let files = vec![file(1, &a), file(2, &zeros)];
        let probe = [hub];
        let votes = TokenVotes {
            case: vec![0; 2 * FOLD_GROUPS],
            vote: vec![2.0; 2 * FOLD_GROUPS],
            per_probe: vec![1, 1],
            n_groups: FOLD_GROUPS,
            n_segments: 1,
        };
        let m = chunk_mass(
            &files,
            &[(1, &probe[..]), (0, &probe[..])],
            &votes,
            Competition::Content,
        );
        let (sa, sb) = (turn_stream_id(1, 0), turn_stream_id(2, 0));
        assert_eq!(m.chunks, vec![(sa, 0, 40), (sa, 1, 40), (sb, 0, 8)]);
        assert_eq!(m.file_of, vec![0, 0, 1]);
        assert_eq!(m.mass, vec![0.0, 6.0, 0.0]);
    }
}
