//! The learner's state and one fold of what changed into it
//! (`docs/incremental_chunk_exclusion.md` §4.3).
//!
//! The state is the sample's entries, each competition's gallery count, and
//! the gallery streams folded so far — each with its own signature held, so a
//! stream the substrate drops can still be scanned to take it out. A fold
//! compares the state with the galleries as they stand: streams removed are
//! taken out, streams appended folded in, questions that entered the sample
//! scanned against the whole gallery, and once enough entries wait on a
//! removed leader, those are rescanned in one pass.

use std::collections::{BTreeMap, HashMap, HashSet};

use candle::Result;

use super::gallery::{Competition, GalleryStream, GalleryView};
use super::scan_jobs::{scan, ScanShape};
use crate::persistence::streams::StreamId;
use crate::provenance::chunk_exclusion::records::{CutRecord, Entry};
use crate::provenance::chunk_exclusion::sample::{bottom_k, SAMPLE};
use crate::provenance::chunk_exclusion::stats::{EntryStats, FileId};
use crate::provenance::gallery_arena::BeliefArena;
use crate::provenance::{WideQSig, FOLD_GROUPS};

/// The share of the sample's entries that may wait on a rescan before one
/// runs: a rescan pass reads the whole gallery, so it is batched.
pub(crate) const RESCAN_SHARE: f64 = 0.02;

/// A sample question as the learner holds it.
#[derive(Clone, Debug)]
pub(crate) struct SampleQuestion {
    pub owner: FileId,
    /// Its full-width tokens, as scanned.
    pub tokens: Vec<WideQSig>,
    /// Per competition, token major, fold group minor.
    pub entries: [Vec<Entry>; 2],
    /// The cursor its record was last written at; `None` when it has none
    /// yet — a question that entered the sample this fold.
    pub written: Option<u64>,
    /// The cursor its entries have folded to, when behind the state's: a
    /// question loaded from a record written before the group's cursor, whose
    /// streams above it the next fold catches it up on.
    pub behind: Option<u64>,
}

/// See the module documentation.
#[derive(Clone, Debug, Default)]
pub(crate) struct LearnerState {
    pub sample: BTreeMap<StreamId, SampleQuestion>,
    /// Each competition's gallery count.
    pub n: [u64; 2],
    /// Each competition's folded streams.
    pub folded: [HashMap<StreamId, GalleryStream>; 2],
    /// The highest signature stamp folded.
    pub cursor: u64,
    /// Question records whose question is no longer in the substrate, found
    /// at load: tombstoned by the next write.
    pub orphans: Vec<StreamId>,
    /// The group record as last written or read — the next is written only
    /// when it differs.
    pub written_cut: Option<CutRecord>,
}

/// What one fold did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FoldCounts {
    pub appended: usize,
    pub removed: usize,
    pub entered: usize,
    pub left: usize,
    pub rescanned: usize,
    /// Questions loaded behind the cursor, caught up on what is above theirs.
    pub caught_up: usize,
}

/// A question's full-width tokens, at the galleries' `width`.
pub(crate) fn full_width(tokens: Vec<WideQSig>, width: usize) -> Vec<WideQSig> {
    tokens
        .into_iter()
        .filter(|t| t.words.len() == width)
        .collect()
}

/// The sample `view` holds: the [`SAMPLE`] questions at the galleries' width
/// with the smallest hash. A question at another width has no token to scan,
/// so it never takes a place.
pub(crate) fn sample_of(view: &GalleryView, width: usize) -> HashSet<StreamId> {
    let at_width = view
        .questions
        .iter()
        .filter(|q| q.sig.width() == width)
        .map(|q| q.sid);
    bottom_k(at_width, SAMPLE).into_iter().collect()
}

impl LearnerState {
    /// Fold `view` into the state, scanning on `arena`, each competition in
    /// its `shapes`. Returns what it did and the questions that left the
    /// sample, whose records go.
    pub(crate) fn fold(
        &mut self,
        view: &GalleryView,
        arena: &dyn BeliefArena,
        shapes: [ScanShape; 2],
    ) -> Result<(FoldCounts, Vec<StreamId>)> {
        let mut counts = FoldCounts::default();
        let Some(width) = view.width() else {
            return Ok((counts, Vec::new()));
        };
        let chosen = sample_of(view, width);
        let left: Vec<StreamId> = self
            .sample
            .keys()
            .copied()
            .filter(|s| !chosen.contains(s))
            .collect();
        for sid in &left {
            self.sample.remove(sid);
        }
        counts.left = left.len();
        counts.caught_up = self.catch_up(arena, shapes)?;

        // The galleries as they stand: the calibration one without the sample.
        let current = |comp: Competition| -> HashMap<StreamId, &GalleryStream> {
            view.streams(comp)
                .iter()
                .filter(|s| comp == Competition::Content || !chosen.contains(&s.sid))
                .map(|s| (s.sid, s))
                .collect()
        };
        for comp in Competition::BOTH {
            let ci = comp.index();
            let now = current(comp);
            let removed: Vec<GalleryStream> = self.folded[ci]
                .values()
                .filter(|f| now.get(&f.sid).is_none_or(|s| s.hlc != f.hlc))
                .cloned()
                .collect();
            let appended: Vec<&GalleryStream> = now
                .values()
                .copied()
                .filter(|s| self.folded[ci].get(&s.sid).is_none_or(|f| f.hlc != s.hlc))
                .collect();
            counts.removed += removed.len();
            counts.appended += appended.len();
            if !removed.is_empty() {
                let refs: Vec<&GalleryStream> = removed.iter().collect();
                self.take_out(ci, &refs, arena, shapes[ci])?;
                for r in &removed {
                    self.folded[ci].remove(&r.sid);
                }
            }
            if !appended.is_empty() {
                self.fold_in(ci, &appended, arena, shapes[ci])?;
                for s in appended {
                    self.folded[ci].insert(s.sid, s.clone());
                }
            }
        }

        // Questions entering the sample: scanned against each whole gallery,
        // whose count they fix.
        let questions: HashMap<StreamId, _> = view.questions.iter().map(|q| (q.sid, q)).collect();
        let mut entered: Vec<StreamId> = chosen
            .iter()
            .copied()
            .filter(|s| !self.sample.contains_key(s))
            .collect();
        entered.sort_by_key(|s| s.0);
        if !entered.is_empty() {
            let mut fresh: Vec<(StreamId, SampleQuestion)> = entered
                .iter()
                .filter_map(|sid| {
                    let q = questions.get(sid)?;
                    let tokens = full_width(q.sig.decode(), width);
                    (!tokens.is_empty()).then(|| {
                        let blank = vec![Entry::default(); tokens.len() * FOLD_GROUPS];
                        (
                            *sid,
                            SampleQuestion {
                                owner: q.owner,
                                tokens,
                                entries: [blank.clone(), blank],
                                written: None,
                                behind: None,
                            },
                        )
                    })
                })
                .collect();
            counts.entered = fresh.len();
            let probes: Vec<&[WideQSig]> = fresh.iter().map(|(_, q)| q.tokens.as_slice()).collect();
            let mut scanned = Vec::new();
            if probes.is_empty() {
                return self.finish(view, &chosen, arena, shapes, counts, left);
            }
            for comp in Competition::BOTH {
                let ci = comp.index();
                let streams: Vec<&GalleryStream> = self.folded[ci].values().collect();
                let got = scan(arena, &streams, &probes, shapes[ci])?;
                self.n[ci] = got.n;
                scanned.push(got.stats);
            }
            for (ci, per_probe) in scanned.into_iter().enumerate() {
                for ((_, q), per_token) in fresh.iter_mut().zip(per_probe) {
                    for (t, groups) in per_token.into_iter().enumerate() {
                        for (g, stats) in groups.into_iter().enumerate() {
                            q.entries[ci][t * FOLD_GROUPS + g].stats = stats;
                        }
                    }
                }
            }
            self.sample.extend(fresh);
        }
        self.finish(view, &chosen, arena, shapes, counts, left)
    }

    /// The end of a fold: the rescan when due, and the cursor raised to the
    /// highest stamp folded.
    fn finish(
        &mut self,
        view: &GalleryView,
        chosen: &HashSet<StreamId>,
        arena: &dyn BeliefArena,
        shapes: [ScanShape; 2],
        mut counts: FoldCounts,
        left: Vec<StreamId>,
    ) -> Result<(FoldCounts, Vec<StreamId>)> {
        self.mark_vanished_leaders();
        counts.rescanned = self.rescan_if_due(arena, shapes)?;
        let highest = self
            .folded
            .iter()
            .flat_map(|m| m.values().map(|s| s.hlc))
            .chain(
                view.questions
                    .iter()
                    .filter(|q| chosen.contains(&q.sid))
                    .map(|q| q.hlc),
            )
            .max()
            .unwrap_or(0);
        self.cursor = self.cursor.max(highest);
        Ok((counts, left))
    }

    /// The sample's tokens, in sample order, as probes.
    fn probes(&self) -> (Vec<StreamId>, Vec<&[WideQSig]>) {
        self.sample
            .iter()
            .map(|(sid, q)| (*sid, q.tokens.as_slice()))
            .unzip()
    }

    /// Take `streams` out of competition `ci`: subtract their sums, and leave
    /// every entry one of their files led waiting on a rescan.
    fn take_out(
        &mut self,
        ci: usize,
        streams: &[&GalleryStream],
        arena: &dyn BeliefArena,
        shape: ScanShape,
    ) -> Result<()> {
        let files: HashSet<FileId> = streams.iter().map(|s| s.file).collect();
        let (sids, probes) = self.probes();
        if probes.is_empty() {
            return Ok(());
        }
        let got = scan(arena, streams, &probes, shape)?;
        self.n[ci] = self.n[ci].saturating_sub(got.n);
        for (sid, per_token) in sids.iter().zip(got.stats) {
            let q = self.sample.get_mut(sid).expect("a sample question");
            for (t, groups) in per_token.into_iter().enumerate() {
                for (g, stats) in groups.into_iter().enumerate() {
                    let e = &mut q.entries[ci][t * FOLD_GROUPS + g];
                    e.stats.subtract_sums(&stats);
                    if files.iter().any(|&f| e.stats.leads(f)) {
                        e.dirty = true;
                    }
                }
            }
        }
        Ok(())
    }

    /// Fold `streams` into competition `ci`.
    fn fold_in(
        &mut self,
        ci: usize,
        streams: &[&GalleryStream],
        arena: &dyn BeliefArena,
        shape: ScanShape,
    ) -> Result<()> {
        let (sids, probes) = self.probes();
        if probes.is_empty() {
            return Ok(());
        }
        let got = scan(arena, streams, &probes, shape)?;
        self.n[ci] += got.n;
        self.merge_into(ci, &sids, got.stats);
        Ok(())
    }

    /// Merge scanned statistics into the entries of questions `sids`, in
    /// competition `ci`; an entry whose leader moved loses its winning token.
    fn merge_into(
        &mut self,
        ci: usize,
        sids: &[StreamId],
        stats: Vec<Vec<[EntryStats; FOLD_GROUPS]>>,
    ) {
        for (sid, per_token) in sids.iter().zip(stats) {
            let q = self.sample.get_mut(sid).expect("a sample question");
            for (t, groups) in per_token.into_iter().enumerate() {
                for (g, s) in groups.into_iter().enumerate() {
                    let e = &mut q.entries[ci][t * FOLD_GROUPS + g];
                    let before = (e.stats.a1, e.stats.c1);
                    e.stats.merge(&s);
                    if (e.stats.a1, e.stats.c1) != before {
                        e.win = None;
                    }
                }
            }
        }
    }

    /// Catch every question loaded behind the state's cursor up on the
    /// streams folded above its own: questions are grouped by the cursor their
    /// record was written at, and each group scans only what is above it. The
    /// gallery counts already hold those streams. How many questions caught up.
    fn catch_up(&mut self, arena: &dyn BeliefArena, shapes: [ScanShape; 2]) -> Result<usize> {
        let mut by_cursor: BTreeMap<u64, Vec<StreamId>> = BTreeMap::new();
        for (sid, q) in &self.sample {
            if let Some(c) = q.behind {
                by_cursor.entry(c).or_default().push(*sid);
            }
        }
        let mut caught = 0usize;
        for (cursor, sids) in by_cursor {
            for comp in Competition::BOTH {
                let ci = comp.index();
                let above: Vec<GalleryStream> = self.folded[ci]
                    .values()
                    .filter(|s| s.hlc > cursor)
                    .cloned()
                    .collect();
                if above.is_empty() {
                    continue;
                }
                let refs: Vec<&GalleryStream> = above.iter().collect();
                let probes: Vec<&[WideQSig]> = sids
                    .iter()
                    .map(|s| self.sample[s].tokens.as_slice())
                    .collect();
                let got = scan(arena, &refs, &probes, shapes[ci])?;
                self.merge_into(ci, &sids, got.stats);
            }
            for sid in &sids {
                if let Some(q) = self.sample.get_mut(sid) {
                    q.behind = None;
                }
            }
            caught += sids.len();
        }
        Ok(caught)
    }

    /// Leave every entry whose leader or runner-up names a file no longer in
    /// its gallery waiting on a rescan: a stream removed while no member held
    /// its window — a removal a takeover finds — was never taken out, and the
    /// entries it led would otherwise vote for a file that is gone.
    fn mark_vanished_leaders(&mut self) {
        for comp in Competition::BOTH {
            let ci = comp.index();
            let present: HashSet<FileId> = self.folded[ci].values().map(|s| s.file).collect();
            for q in self.sample.values_mut() {
                for e in &mut q.entries[ci] {
                    let gone = |f: FileId| f != 0 && !present.contains(&f);
                    if gone(e.stats.c1) || gone(e.stats.c2) {
                        e.dirty = true;
                    }
                }
            }
        }
    }

    /// Entries waiting on a rescan, and entries in all.
    pub(crate) fn dirty(&self) -> (usize, usize) {
        let all = self
            .sample
            .values()
            .flat_map(|q| q.entries.iter().flatten());
        all.fold((0, 0), |(d, n), e| (d + usize::from(e.dirty), n + 1))
    }

    /// Rescan every entry waiting on one against the whole gallery once they
    /// pass [`RESCAN_SHARE`] of the entries; how many were rescanned.
    fn rescan_if_due(&mut self, arena: &dyn BeliefArena, shapes: [ScanShape; 2]) -> Result<usize> {
        let (dirty, all) = self.dirty();
        if dirty == 0 || (dirty as f64) < RESCAN_SHARE * all as f64 {
            return Ok(0);
        }
        for comp in Competition::BOTH {
            let ci = comp.index();
            // Each question's tokens holding a dirty entry in this competition.
            let mut at: Vec<(StreamId, Vec<usize>)> = Vec::new();
            for (sid, q) in &self.sample {
                let tokens: Vec<usize> = (0..q.tokens.len())
                    .filter(|&t| {
                        q.entries[ci][t * FOLD_GROUPS..(t + 1) * FOLD_GROUPS]
                            .iter()
                            .any(|e| e.dirty)
                    })
                    .collect();
                if !tokens.is_empty() {
                    at.push((*sid, tokens));
                }
            }
            if at.is_empty() {
                continue;
            }
            let owned: Vec<Vec<WideQSig>> = at
                .iter()
                .map(|(sid, ts)| {
                    ts.iter()
                        .map(|&t| self.sample[sid].tokens[t].clone())
                        .collect()
                })
                .collect();
            let probes: Vec<&[WideQSig]> = owned.iter().map(Vec::as_slice).collect();
            let streams: Vec<&GalleryStream> = self.folded[ci].values().collect();
            let got = scan(arena, &streams, &probes, shapes[ci])?;
            for ((sid, ts), per_token) in at.iter().zip(got.stats) {
                let q = self.sample.get_mut(sid).expect("a sample question");
                for (&t, groups) in ts.iter().zip(per_token) {
                    for (g, stats) in groups.into_iter().enumerate() {
                        let e = &mut q.entries[ci][t * FOLD_GROUPS + g];
                        if e.dirty {
                            let sums = (e.stats.sum, e.stats.sumsq);
                            e.stats = EntryStats {
                                sum: sums.0,
                                sumsq: sums.1,
                                ..EntryStats::default()
                            };
                            e.stats.merge_leaders(&stats);
                            e.dirty = false;
                            e.win = None;
                        }
                    }
                }
            }
        }
        Ok(dirty)
    }
}
