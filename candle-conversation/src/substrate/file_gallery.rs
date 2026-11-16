//! A conversation's belief gallery, kept on its timeline entry and built once per
//! change to what it is built from.
//!
//! A belief scan scores every file of a group, and assembling a file's gallery —
//! its turns, their signature windows and seams, its exchanges — walks every one
//! of its turns: ~110 ms of a ~220 ms scan over 7,449 files and 22,384 turns,
//! on every reprojection (BlackWorld, 2026-11-17), for galleries that had not
//! changed since the last one.
//!
//! **Held in the version, filled by readers.** Each [`TimelineEntry`] carries a
//! [`GalleryMemo`]. A reader that finds it empty builds the file's
//! [`FileGallery`] from the version it reads and installs it; every later reader
//! of that entry — in that version and in every later one that shares it —
//! reads it back. A substrate edit does no gallery work at all, so nothing here
//! slows the compare-and-swap every writer goes through
//! (`docs/substrate_lock_free.md`).
//!
//! **Cleared wherever what it reads changes.** A file's gallery reads its
//! timeline entry (turns, summary metadata, couplings) and the streams of its
//! turns (signature windows, the seams in their projection events). Every edit
//! of an entry goes through [`entry_mut`], which clears it; a turn's
//! declaration, signature or events, written on its stream, clear the owning
//! timeline's the same way ([`Substrate::gallery_changed`]); the few edits that
//! can change a stream without naming its timeline move every memo's epoch
//! ([`Substrate::galleries_changed`]). An edit always clears its version's own
//! copy of the entry, so it never empties a memo an older version still reads.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use arc_swap::ArcSwapOption;

use super::pcoll::{arc_mut, PMap};
use super::stream_index::StreamKey;
use super::{Substrate, TimelineEntry};
use crate::persistence::streams::StreamId;
use crate::projection::{TimelineId, TurnIndex};
use crate::provenance::SigWindow;

/// One conversation's belief gallery as its turns build it: every non-summary
/// turn in order, grouped into exchanges, and the windows that score them.
#[derive(Debug, Clone, Default)]
pub struct FileGallery {
    /// Every non-summary turn, in order — the subsequence couplings project
    /// onto.
    pub arc_turn: Vec<TurnIndex>,
    /// Each exchange's span of `arc_turn`.
    pub ex_ranges: Vec<Range<usize>>,
    pub n_slots: usize,
    /// The stored signature windows referenced by `windows`, not decoded.
    pub arcs_kept: Vec<SigWindow>,
    /// Stream id of each kept arc's turn — the arena residency key.
    pub arc_sids: Vec<StreamId>,
    /// `(arc index into arcs_kept, window start, window end, exchange slot)`.
    pub windows: Vec<(usize, usize, usize, usize)>,
}

/// A built gallery and the memo epoch it was built under.
struct Built {
    gallery: Arc<FileGallery>,
    epoch: u64,
}

/// A timeline entry's gallery, or nothing until a reader builds it. Lock-free
/// both ways: a load and a store. A copy of an entry starts empty.
#[derive(Default)]
pub struct GalleryMemo(ArcSwapOption<Built>);

impl Clone for GalleryMemo {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl fmt::Debug for GalleryMemo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if self.0.load().is_some() {
            "GalleryMemo(built)"
        } else {
            "GalleryMemo(empty)"
        })
    }
}

impl GalleryMemo {
    /// Empty it. Called only on an edit's own copy of an entry ([`entry_mut`]).
    fn clear(&self) {
        self.0.store(None);
    }

    /// The gallery held, when it was built under `epoch`.
    fn get(&self, epoch: u64) -> Option<Arc<FileGallery>> {
        self.0
            .load()
            .as_ref()
            .filter(|b| b.epoch == epoch)
            .map(|b| Arc::clone(&b.gallery))
    }

    fn put(&self, gallery: Arc<FileGallery>, epoch: u64) {
        self.0.store(Some(Arc::new(Built { gallery, epoch })));
    }
}

/// `timeline`'s entry, this version's own copy, its gallery cleared: what every
/// edit of an entry goes through.
pub(super) fn entry_mut<'a>(
    timelines: &'a mut PMap<TimelineId, Arc<TimelineEntry>>,
    timeline: &TimelineId,
) -> Option<&'a mut TimelineEntry> {
    arc_mut(timelines, timeline).inspect(|entry| entry.gallery.clear())
}

impl Substrate {
    /// `timeline`'s gallery: the one held on its entry, or the one `build`
    /// makes of this version, kept for every later reader. `None` for a
    /// timeline this version does not hold.
    pub(crate) fn file_gallery(
        &self,
        timeline: TimelineId,
        build: impl FnOnce(&Substrate) -> FileGallery,
    ) -> Option<Arc<FileGallery>> {
        let entry = self.timelines.get(&timeline)?;
        if let Some(gallery) = entry.gallery.get(self.gallery_memo_epoch) {
            return Some(gallery);
        }
        let gallery = Arc::new(build(self));
        entry
            .gallery
            .put(Arc::clone(&gallery), self.gallery_memo_epoch);
        Some(gallery)
    }

    /// Something `timeline`'s gallery is built from changed.
    pub(super) fn gallery_changed(&mut self, timeline: TimelineId) {
        entry_mut(&mut self.timelines, &timeline);
    }

    /// A value on `key`'s stream changed: the gallery of the timeline whose
    /// turn it is reads it. A stream not declared as a turn yet names none,
    /// and its declaration clears its timeline's when it comes.
    pub(super) fn stream_gallery_changed(&mut self, key: StreamKey) {
        if let Some(timeline) = self
            .streams
            .turn(key)
            .and_then(|t| TimelineId::from_raw(t.decl.timeline_id))
        {
            self.gallery_changed(timeline);
        }
    }

    /// Something some gallery is built from changed, and which is not known:
    /// every gallery is built again when next read.
    pub(super) fn galleries_changed(&mut self) {
        self.gallery_memo_epoch += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{GroupId, LayerId};

    fn gallery(n: usize) -> FileGallery {
        FileGallery {
            arc_turn: (0..n as u32).map(TurnIndex).collect(),
            n_slots: n,
            ..FileGallery::default()
        }
    }

    fn substrate_with(timelines: &[TimelineId]) -> Substrate {
        let mut s = Substrate::new();
        for &tl in timelines {
            s.timelines.insert(
                tl,
                Arc::new(TimelineEntry::new(
                    LayerId::for_test(1),
                    GroupId::for_test(1),
                )),
            );
        }
        s
    }

    /// **A gallery is built once and read back**, by the version that built it
    /// and by a later one sharing the entry.
    #[test]
    fn a_gallery_is_built_once_and_read_back() {
        let tl = TimelineId::for_test(1);
        let s = substrate_with(&[tl]);
        let first = s.file_gallery(tl, |_| gallery(2)).unwrap();
        let later = s.clone();
        let again = later.file_gallery(tl, |_| panic!("built again")).unwrap();
        assert!(Arc::ptr_eq(&first, &again));
    }

    /// **An edit of the entry clears its gallery, and only its own**: another
    /// file's stays, and the version before the edit keeps reading its own.
    #[test]
    fn an_edit_clears_its_own_entrys_gallery_only() {
        let (a, b) = (TimelineId::for_test(1), TimelineId::for_test(2));
        let before = substrate_with(&[a, b]);
        before.file_gallery(a, |_| gallery(1));
        before.file_gallery(b, |_| gallery(1));
        let mut after = before.clone();
        after.gallery_changed(a);
        assert_eq!(after.file_gallery(a, |_| gallery(3)).unwrap().n_slots, 3);
        after.file_gallery(b, |_| panic!("b did not change"));
        assert_eq!(
            before
                .file_gallery(a, |_| panic!("the old version's own"))
                .unwrap()
                .n_slots,
            1
        );
    }

    /// **An unattributed change rebuilds every gallery**, once.
    #[test]
    fn an_unattributed_change_rebuilds_every_gallery() {
        let (a, b) = (TimelineId::for_test(1), TimelineId::for_test(2));
        let mut s = substrate_with(&[a, b]);
        s.file_gallery(a, |_| gallery(1));
        s.file_gallery(b, |_| gallery(1));
        s.galleries_changed();
        assert_eq!(s.file_gallery(a, |_| gallery(4)).unwrap().n_slots, 4);
        assert_eq!(s.file_gallery(b, |_| gallery(5)).unwrap().n_slots, 5);
        s.file_gallery(a, |_| panic!("built once"));
    }

    /// A timeline the version does not hold has no gallery.
    #[test]
    fn an_unknown_timeline_has_no_gallery() {
        let s = substrate_with(&[]);
        assert!(s
            .file_gallery(TimelineId::for_test(9), |_| gallery(1))
            .is_none());
    }
}
