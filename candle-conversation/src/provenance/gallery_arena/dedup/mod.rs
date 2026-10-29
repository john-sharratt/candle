//! The gallery's dedup overlay: which of a turn's tokens the card holds
//! (`docs/gallery_dedup.md`).
//!
//! The substrate keeps every token's signature. The overlay decides, per turn,
//! which of them the arena uploads: a learned [`Codebook`] names each token by
//! its nearest code, and a reader asking for [`Keep::FirstPerCode`] has the
//! card hold only the first few of each code within the turn — near duplicates
//! of a token already held vote as it does. The turn's pages pack its kept
//! tokens in order ([`KeepMask::rank`]), and a scan index reads only them.

mod codebook;
mod keep_mask;

pub use codebook::Codebook;
pub use keep_mask::KeepMask;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;

use crate::persistence::streams::StreamId;

use super::TurnSigs;

/// Which of a turn's tokens a reader has the gallery hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Keep {
    /// Every token.
    #[default]
    All,
    /// The first `n` of each code within the turn.
    FirstPerCode(u8),
}

impl Keep {
    /// Whether a turn held under `self` holds every token `other` would: all
    /// of them covers anything, and more per code covers fewer.
    pub fn covers(self, other: Keep) -> bool {
        match (self, other) {
            (Keep::All, _) => true,
            (Keep::FirstPerCode(_), Keep::All) => false,
            (Keep::FirstPerCode(a), Keep::FirstPerCode(b)) => a >= b,
        }
    }

    /// The narrowest keep covering both — what a turn two readers read is
    /// held under.
    pub fn union(self, other: Keep) -> Keep {
        if self.covers(other) {
            self
        } else {
            other
        }
    }
}

/// Coarse codes of the codebook.
const COARSE: usize = 64;
/// Fine codes within each coarse code.
const FINE: usize = 64;
/// Rounds of k-means a level.
const ITERATIONS: usize = 4;
/// Tokens the codebook trains on, at most.
const TRAIN_POINTS: usize = 200_000;
/// The codebook's training seed.
const SEED: u64 = 0x5eed_c0de;

/// A turn's mask under one keep as computed: valid while the turn's
/// fingerprint and the codebook it was computed under are.
struct CachedMask {
    fingerprint: u64,
    book: u64,
    mask: Arc<KeepMask>,
}

/// The arena's overlay state: its codebook, trained once, and every turn's
/// mask under each keep a reader asks for, computed once per content.
pub(super) struct Dedup {
    width: usize,
    book: Mutex<Option<Arc<Codebook>>>,
    masks: Mutex<HashMap<(StreamId, Keep), CachedMask>>,
}

impl Dedup {
    /// Overlay state for `width`-word signatures.
    pub(super) fn new(width: usize) -> Self {
        Self {
            width,
            book: Mutex::new(None),
            masks: Mutex::new(HashMap::new()),
        }
    }

    /// The codebook, trained from `turns` if there is none yet: every
    /// `k`-th turn whole, `k` spreading the sample over all of them, up to
    /// [`TRAIN_POINTS`] tokens. The first corpus a dedup reader scans trains
    /// it; it then names every turn for the life of the arena.
    pub(super) fn book_from(&self, turns: &[TurnSigs<'_>]) -> Arc<Codebook> {
        let mut book = self.book.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(b) = book.as_ref() {
            return Arc::clone(b);
        }
        let t0 = Instant::now();
        let total: usize = turns.iter().map(TurnSigs::len).sum();
        let every = (total / TRAIN_POINTS).max(1);
        let mut points: Vec<u64> = Vec::new();
        for turn in turns.iter().step_by(every) {
            if points.len() / self.width >= TRAIN_POINTS {
                break;
            }
            points.extend(turn.words(self.width));
        }
        points.truncate(TRAIN_POINTS * self.width);
        let trained = Arc::new(Codebook::train(
            &points, self.width, COARSE, FINE, ITERATIONS, SEED,
        ));
        tracing::info!(
            target: "candle_conversation::provenance::gallery_arena",
            turns = turns.len(),
            points = points.len() / self.width,
            took_ms = t0.elapsed().as_millis() as u64,
            "trained the gallery dedup codebook"
        );
        *book = Some(Arc::clone(&trained));
        trained
    }

    /// The mask `keep` gives a turn under `book`: computed now and kept for the
    /// turn's next upload, or the one already computed for this content.
    pub(super) fn mask(
        &self,
        sid: StreamId,
        fingerprint: u64,
        sigs: TurnSigs<'_>,
        keep: Keep,
        book: Option<&Codebook>,
    ) -> Arc<KeepMask> {
        let (Keep::FirstPerCode(per_code), Some(book)) = (keep, book) else {
            return Arc::new(KeepMask::all(sigs.len()));
        };
        if let Some(m) = self.cached(sid, fingerprint, keep, book.id()) {
            return m;
        }
        let mask = Arc::new(KeepMask::first_per_code(
            &book.codes(&sigs.words(self.width)),
            per_code as usize,
        ));
        self.masks.lock().unwrap_or_else(|e| e.into_inner()).insert(
            (sid, keep),
            CachedMask {
                fingerprint,
                book: book.id(),
                mask: Arc::clone(&mask),
            },
        );
        mask
    }

    /// The mask computed for this content under `keep` and the codebook `book`
    /// names, if one was.
    pub(super) fn cached(
        &self,
        sid: StreamId,
        fingerprint: u64,
        keep: Keep,
        book: u64,
    ) -> Option<Arc<KeepMask>> {
        let masks = self.masks.lock().unwrap_or_else(|e| e.into_inner());
        masks
            .get(&(sid, keep))
            .filter(|m| m.fingerprint == fingerprint && m.book == book)
            .map(|m| Arc::clone(&m.mask))
    }

    /// The codebook, if one is trained.
    pub(super) fn book(&self) -> Option<Arc<Codebook>> {
        self.book.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Compute every mask `turns` lack under their keeps, in parallel — so the
    /// uploads after it, one turn at a time under the residency lock, find
    /// theirs computed.
    pub(super) fn prepare(&self, turns: &[(StreamId, TurnSigs<'_>, u64, Keep)], book: &Codebook) {
        let missing: Vec<&(StreamId, TurnSigs<'_>, u64, Keep)> = turns
            .iter()
            .filter(|(sid, _, fp, keep)| {
                *keep != Keep::All && self.cached(*sid, *fp, *keep, book.id()).is_none()
            })
            .collect();
        let t0 = Instant::now();
        missing.par_iter().for_each(|(sid, sigs, fp, keep)| {
            self.mask(*sid, *fp, *sigs, *keep, Some(book));
        });
        if !missing.is_empty() {
            tracing::debug!(
                target: "candle_conversation::provenance::gallery_arena",
                turns = missing.len(),
                tokens = missing.iter().map(|t| t.1.len()).sum::<usize>(),
                took_ms = t0.elapsed().as_millis() as u64,
                "computed gallery dedup masks"
            );
        }
    }

    /// Forget the masks of turns no scan can read any more.
    pub(super) fn forget(&self, sids: &[StreamId]) {
        let gone: HashSet<StreamId> = sids.iter().copied().collect();
        self.masks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|(sid, _), _| !gone.contains(sid));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::WideQSig;

    fn sig(fill: u64) -> WideQSig {
        WideQSig {
            n_heads: 1,
            words: vec![fill; 2],
        }
    }

    /// **All covers everything; more per code covers fewer**, and the union
    /// is the wider.
    #[test]
    fn keeps_cover_and_unite() {
        let (one, two) = (Keep::FirstPerCode(1), Keep::FirstPerCode(2));
        assert!(Keep::All.covers(two) && two.covers(one) && !one.covers(two));
        assert!(!two.covers(Keep::All));
        assert_eq!(one.union(two), two);
        assert_eq!(two.union(Keep::All), Keep::All);
        assert_eq!(Keep::default(), Keep::All);
    }

    /// **A turn's mask keeps the first token of each code, and is computed
    /// once per content**: a second ask under the same fingerprint is the
    /// cached mask, a new fingerprint recomputes, and `All` keeps every token.
    #[test]
    fn masks_keep_the_first_of_each_code_once_per_content() {
        let dedup = Dedup::new(2);
        let sigs = vec![sig(0), sig(0), sig(u64::MAX), sig(0), sig(u64::MAX)];
        let turn = TurnSigs::from(&sigs);
        let book = dedup.book_from(&[turn]);
        assert!(Arc::ptr_eq(&book, &dedup.book_from(&[])), "trained once");
        let keep = Keep::FirstPerCode(1);
        let m = dedup.mask(StreamId(1), 7, turn, keep, Some(&book));
        assert_eq!(
            (0..5).map(|t| m.keeps(t)).collect::<Vec<_>>(),
            vec![true, false, true, false, false]
        );
        let again = |fp| dedup.mask(StreamId(1), fp, turn, keep, Some(&book));
        assert!(Arc::ptr_eq(&m, &again(7)));
        assert!(!Arc::ptr_eq(&m, &again(8)));
        let every = dedup.mask(StreamId(1), 8, turn, Keep::All, Some(&book));
        assert_eq!(every.kept(), 5);
        let unnamed = dedup.mask(StreamId(1), 8, turn, keep, None);
        assert_eq!(unnamed.kept(), 5, "no codebook");
        dedup.forget(&[StreamId(1)]);
        assert!(dedup.cached(StreamId(1), 8, keep, book.id()).is_none());
        dedup.prepare(&[(StreamId(1), turn, 8, keep)], &book);
        let prepared = dedup.cached(StreamId(1), 8, keep, book.id()).unwrap();
        assert_eq!(prepared.kept(), 2);
    }
}
