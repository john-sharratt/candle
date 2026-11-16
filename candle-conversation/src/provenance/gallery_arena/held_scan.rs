//! One belief-scan launch as owned data — what a thread that is not the
//! arena's scan thread sends there to run ([`super::BeliefArena`]).
//!
//! A [`PagedSegment`] borrows its turns' signatures from the scan that built
//! it. Held, a stored window is its shared blob (a reference count, not a
//! copy) and decoded signatures are copied once; the arena's thread reads the
//! held launch back as the very segments it was built from.

use std::sync::Arc;

use super::super::{SigWindow, WideQSig};
use super::scan_split::{self, ArenaScanSplit};
use super::{
    GalleryArena, GroupSet, Holder, Keep, PagedSegment, PagedWindow, TokenVotes, TurnSigs,
};
use crate::persistence::streams::StreamId;

/// A turn's signatures, held.
#[derive(Clone, Debug)]
enum HeldSigs {
    Decoded(Arc<[WideQSig]>),
    Encoded(SigWindow),
}

impl HeldSigs {
    fn of(sigs: TurnSigs<'_>) -> Self {
        match sigs {
            TurnSigs::Decoded(s) => Self::Decoded(Arc::from(s)),
            TurnSigs::Encoded(w) => Self::Encoded(w.clone()),
        }
    }

    fn view(&self) -> TurnSigs<'_> {
        match self {
            Self::Decoded(s) => TurnSigs::Decoded(s),
            Self::Encoded(w) => TurnSigs::Encoded(w),
        }
    }
}

/// A [`PagedWindow`], held.
#[derive(Clone, Debug)]
struct HeldWindow {
    sid: StreamId,
    fingerprint: u64,
    turn: HeldSigs,
    start: usize,
    end: usize,
    case: usize,
}

/// A [`PagedSegment`], held.
#[derive(Clone, Debug)]
pub struct HeldSegment {
    windows: Vec<HeldWindow>,
    n_cases: usize,
    holder: Holder,
    groups: GroupSet,
    keep: Keep,
}

impl HeldSegment {
    pub fn of(segment: &PagedSegment<'_>) -> Self {
        Self {
            windows: segment
                .windows
                .iter()
                .map(|w| HeldWindow {
                    sid: w.sid,
                    fingerprint: w.fingerprint,
                    turn: HeldSigs::of(w.turn),
                    start: w.start,
                    end: w.end,
                    case: w.case,
                })
                .collect(),
            n_cases: segment.n_cases,
            holder: segment.holder,
            groups: segment.groups,
            keep: segment.keep,
        }
    }

    /// The segment this was held from.
    pub fn view(&self) -> PagedSegment<'_> {
        PagedSegment {
            windows: self
                .windows
                .iter()
                .map(|w| PagedWindow {
                    sid: w.sid,
                    fingerprint: w.fingerprint,
                    turn: w.turn.view(),
                    start: w.start,
                    end: w.end,
                    case: w.case,
                })
                .collect(),
            n_cases: self.n_cases,
            holder: self.holder,
            groups: self.groups,
            keep: self.keep,
        }
    }
}

/// Which launch, and what it weighs.
#[derive(Clone, Debug)]
enum Launch {
    Weighted(Vec<f32>),
    SegmentWeighted(Vec<Vec<f32>>),
    Votes,
}

/// One launch of [`super::BeliefArena`], held. See the module documentation.
#[derive(Clone, Debug)]
pub struct HeldScan {
    segments: Vec<HeldSegment>,
    probes: Vec<Vec<WideQSig>>,
    launch: Launch,
}

/// What a [`HeldScan`] answers, its error as text: a launch's error crosses a
/// thread, and only what it says is read on the other side.
#[derive(Debug, PartialEq)]
pub enum ScanAnswer {
    Scores(Result<Vec<Vec<f32>>, String>),
    Votes(Result<TokenVotes, String>),
}

/// A [`HeldScan`]'s answer, and where the arena's time went running it — the
/// scan that sent it counts that as its own ([`scan_split::merge`]).
#[derive(Debug, PartialEq)]
pub struct ScanReply {
    pub answer: ScanAnswer,
    pub split: ArenaScanSplit,
}

impl HeldScan {
    fn held(segments: &[PagedSegment], probes: &[&[WideQSig]], launch: Launch) -> Self {
        Self {
            segments: segments.iter().map(HeldSegment::of).collect(),
            probes: probes.iter().map(|p| p.to_vec()).collect(),
            launch,
        }
    }

    /// [`GalleryArena::scan_weighted`], held.
    pub fn weighted(segments: &[PagedSegment], probes: &[&[WideQSig]], weights: &[f32]) -> Self {
        Self::held(segments, probes, Launch::Weighted(weights.to_vec()))
    }

    /// [`GalleryArena::scan_segment_weighted`], held.
    pub fn segment_weighted(
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        weights: &[&[f32]],
    ) -> Self {
        Self::held(
            segments,
            probes,
            Launch::SegmentWeighted(weights.iter().map(|w| w.to_vec()).collect()),
        )
    }

    /// [`GalleryArena::scan_votes`], held.
    pub fn votes(segments: &[PagedSegment], probes: &[&[WideQSig]]) -> Self {
        Self::held(segments, probes, Launch::Votes)
    }

    /// The segments, as the scan that sent them built them.
    pub fn segments(&self) -> Vec<PagedSegment<'_>> {
        self.segments.iter().map(HeldSegment::view).collect()
    }

    /// The probes, as the scan that sent them passed them.
    pub fn probes(&self) -> Vec<&[WideQSig]> {
        self.probes.iter().map(Vec::as_slice).collect()
    }

    /// Run the launch on `arena` — on its scan thread — counting its time
    /// apart from that thread's own scans.
    pub fn run(&self, arena: &GalleryArena) -> ScanReply {
        let segments = self.segments();
        let probes = self.probes();
        let text = |e: candle::Error| e.to_string();
        let (answer, split) = scan_split::apart(|| match &self.launch {
            Launch::Weighted(w) => {
                ScanAnswer::Scores(arena.scan_weighted(&segments, &probes, w).map_err(text))
            }
            Launch::SegmentWeighted(w) => {
                let w: Vec<&[f32]> = w.iter().map(Vec::as_slice).collect();
                ScanAnswer::Scores(
                    arena
                        .scan_segment_weighted(&segments, &probes, &w)
                        .map_err(text),
                )
            }
            Launch::Votes => ScanAnswer::Votes(arena.scan_votes(&segments, &probes).map_err(text)),
        });
        ScanReply { answer, split }
    }

    /// The answer when there is no arena to run on: the scheduler that said it
    /// had one has since lost it. Says so in the shape the launch answers.
    pub fn no_arena(&self) -> ScanReply {
        let gone = "the scheduler has no gallery arena".to_string();
        ScanReply {
            answer: match self.launch {
                Launch::Votes => ScanAnswer::Votes(Err(gone)),
                _ => ScanAnswer::Scores(Err(gone)),
            },
            split: ArenaScanSplit::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::encode_wide_sigs;

    fn sig(fill: u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: vec![fill; 24],
        }
    }

    /// **A held segment reads back as the segment it was held from**: every
    /// window's stream, fingerprint, range, case and signatures, the case
    /// count, holder, groups and keep — for a stored window and decoded
    /// signatures alike.
    #[test]
    fn a_held_segment_reads_back_as_itself() {
        let decoded = vec![sig(1), sig(2), sig(3)];
        let stored = SigWindow::new(Arc::from(encode_wide_sigs(&[sig(7), sig(8)]))).unwrap();
        let segment = PagedSegment {
            windows: vec![
                PagedWindow {
                    sid: StreamId(11),
                    fingerprint: 0xF00D,
                    turn: TurnSigs::from(&decoded),
                    start: 1,
                    end: 3,
                    case: 0,
                },
                PagedWindow {
                    sid: StreamId(12),
                    fingerprint: stored.fingerprint(),
                    turn: TurnSigs::from(&stored),
                    start: 0,
                    end: 2,
                    case: 4,
                },
            ],
            n_cases: 5,
            holder: Holder::Shared,
            groups: GroupSet::all(3),
            keep: Keep::All,
        };
        let held = HeldSegment::of(&segment);
        drop(decoded);
        let back = held.view();
        assert_eq!(back.n_cases, 5);
        assert_eq!(back.holder, Holder::Shared);
        assert_eq!(back.groups, GroupSet::all(3));
        assert_eq!(back.keep, Keep::All);
        let fields: Vec<(u64, u64, usize, usize, usize)> = back
            .windows
            .iter()
            .map(|w| (w.sid.0, w.fingerprint, w.start, w.end, w.case))
            .collect();
        assert_eq!(
            fields,
            vec![(11, 0xF00D, 1, 3, 0), (12, stored.fingerprint(), 0, 2, 4)]
        );
        assert_eq!(
            back.windows[0].turn.with_decoded(<[WideQSig]>::to_vec),
            vec![sig(1), sig(2), sig(3)]
        );
        assert_eq!(
            back.windows[1].turn.with_decoded(<[WideQSig]>::to_vec),
            vec![sig(7), sig(8)]
        );
    }

    /// **A held scan keeps its probes and weights**, in order.
    #[test]
    fn a_held_scan_keeps_its_probes_and_weights() {
        let tail = vec![sig(1), sig(2)];
        let question = vec![sig(3)];
        let held = HeldScan::segment_weighted(&[], &[&tail, &question], &[&[1.0, 0.5], &[2.0]]);
        assert_eq!(held.probes(), vec![tail.as_slice(), question.as_slice()]);
        assert!(held.segments().is_empty());
        match &held.launch {
            Launch::SegmentWeighted(w) => assert_eq!(w, &vec![vec![1.0, 0.5], vec![2.0]]),
            other => panic!("a segment-weighted launch, got {other:?}"),
        }
        match &HeldScan::weighted(&[], &[&tail], &[0.25]).launch {
            Launch::Weighted(w) => assert_eq!(w, &vec![0.25]),
            other => panic!("a weighted launch, got {other:?}"),
        }
    }
}
