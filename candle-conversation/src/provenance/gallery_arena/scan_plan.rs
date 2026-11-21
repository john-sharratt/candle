//! A segment set's index key, worked out once.
//!
//! The arena caches a scan's index under a fingerprint of its segments
//! ([`fingerprint_segments`]), which walks every window. A belief turn group's
//! segments are a pure function of its assembled files, which a reprojection
//! reuses scan after scan (`projection::group_assembly`): on BlackWorld
//! (2026-11-21) building the ~23,000 windows of a scan's segments and
//! fingerprinting them took ~15 ms of every reprojection, for an index that was
//! reused every time. A [`ScanPlan`] is that key with what a reused index's
//! launch needs besides it — the groups each segment weighs and its window
//! count — so a caller holding the plan builds the segments only when the
//! arena must build an index (`GalleryArena::scan_keyed`).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::scan::PagedSegment;
use super::GroupSet;

/// See the module documentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanPlan {
    /// The index key: [`fingerprint_segments`] of the segments.
    pub key: u64,
    /// The fold groups each segment weighs, in segment order.
    pub weighed: Vec<GroupSet>,
    /// Windows over every segment.
    pub windows: usize,
}

impl ScanPlan {
    /// The plan of `segments`.
    pub fn of(segments: &[PagedSegment]) -> Self {
        Self {
            key: fingerprint_segments(segments),
            weighed: segments.iter().map(|s| s.groups).collect(),
            windows: segments.iter().map(|s| s.windows.len()).sum(),
        }
    }
}

/// Fingerprint the segment structure the built index depends on — file order,
/// each file's case count, holder, keep and `top_k`, and every window's `(sid,
/// fingerprint, turn length, start, end, case)`. Cheap (per-window, not
/// per-token). The holder is in it so an index built for one reader never
/// stands in for another's, whose build would widen what its turns are held
/// for; the keep, so an index over a turn's kept tokens never stands in for a
/// reader asking for more of them.
///
/// The fold groups are not: readers of one segment set weighing different
/// groups share its index, which a reuse checks covers what each segment
/// weighs (`GalleryArena::reuse_index`). Keyed on them, a turn two groups read
/// under different weights — a sealed turn in its own conversation's dialogue
/// and in a shared layer — was two indices and two launches every scan.
///
/// **Trust boundary:** two guarantees back reuse. (1) Each turn's *run id*
/// covers physical page moves — an evict/re-seal/compaction that relocates a
/// turn's pages gives it a new one, so the cache can't serve stale device
/// addresses. (2) This 64-bit
/// SipHash covers logical changes to the scan; a caller MUST vary `w.fingerprint`
/// whenever the turn's content changes
/// ([`SigWindow::fingerprint`](super::super::SigWindow::fingerprint) folds in the
/// stored buffer's address + a content sample), and `turn.len()` is hashed directly
/// so a length change alone (which moves its entries) also invalidates. A 64-bit
/// hash collision under an unchanged generation could serve a wrong index — a
/// ~2⁻⁶⁴ event we accept, the standard content-address trade.
pub(super) fn fingerprint_segments(segments: &[PagedSegment]) -> u64 {
    let mut h = DefaultHasher::new();
    segments.len().hash(&mut h);
    for seg in segments {
        seg.n_cases.hash(&mut h);
        seg.holder.hash(&mut h);
        seg.keep.hash(&mut h);
        // The kernel's rows rank to it; the power is the tally's alone.
        seg.law.top_k.hash(&mut h);
        seg.windows.len().hash(&mut h);
        for w in &seg.windows {
            w.sid.0.hash(&mut h);
            w.fingerprint.hash(&mut h);
            w.turn.len().hash(&mut h);
            w.start.hash(&mut h);
            w.end.hash(&mut h);
            w.case.hash(&mut h);
        }
    }
    h.finish()
}
