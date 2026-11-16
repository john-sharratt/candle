//! The launches a belief scan makes on the gallery, as a trait: the arena runs
//! them where it is, and a thread that is not the arena's scan thread runs them
//! through a handle that sends each one there (`docs/gallery_residency.md` §7).
//!
//! A belief scan is mostly host work — assembling each collection's gallery and
//! each turn group's files from the substrate, then normalizing what comes back
//! — around a few launches. Only the launches need the arena's one scan thread,
//! so a scan taken off that thread hands it those and nothing else.

use candle::Result;

use super::super::WideQSig;
use super::{GalleryArena, PagedSegment, TokenStats, TokenVotes};

/// The gallery as a belief scan reads it. See the module documentation.
pub trait BeliefArena {
    /// Fold groups the gallery's pages are laid out in.
    fn n_groups(&self) -> usize;

    /// [`GalleryArena::scan_weighted`].
    fn scan_weighted(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        group_weights: &[f32],
    ) -> Result<Vec<Vec<f32>>>;

    /// [`GalleryArena::scan_segment_weighted`].
    fn scan_segment_weighted(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        segment_weights: &[&[f32]],
    ) -> Result<Vec<Vec<f32>>>;

    /// [`GalleryArena::scan_votes`].
    fn scan_votes(&self, segments: &[PagedSegment], probes: &[&[WideQSig]]) -> Result<TokenVotes>;

    /// [`GalleryArena::scan_stats`].
    fn scan_stats(&self, segments: &[PagedSegment], probes: &[&[WideQSig]]) -> Result<TokenStats>;
}

impl BeliefArena for GalleryArena {
    fn n_groups(&self) -> usize {
        GalleryArena::n_groups(self)
    }

    fn scan_weighted(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        group_weights: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        GalleryArena::scan_weighted(self, segments, probes, group_weights)
    }

    fn scan_segment_weighted(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        segment_weights: &[&[f32]],
    ) -> Result<Vec<Vec<f32>>> {
        GalleryArena::scan_segment_weighted(self, segments, probes, segment_weights)
    }

    fn scan_votes(&self, segments: &[PagedSegment], probes: &[&[WideQSig]]) -> Result<TokenVotes> {
        GalleryArena::scan_votes(self, segments, probes)
    }

    fn scan_stats(&self, segments: &[PagedSegment], probes: &[&[WideQSig]]) -> Result<TokenStats> {
        GalleryArena::scan_stats(self, segments, probes)
    }
}
