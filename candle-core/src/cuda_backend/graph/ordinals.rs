//! What each segment ordinal held the last time a wave of the same kind
//! reached it — the prediction a segment's mode is decided by before it is
//! recorded (`hub::start_segment`).
//!
//! **Kept per kind of wave, not per ordinal alone.** A decode step opens
//! several waves of its own — the draft walk, the verify, the forward — one
//! after another, and each puts different launches at the same ordinal. Kept
//! by ordinal alone, the prediction for a verify segment was what the draft
//! walk put there a moment earlier: a key the segment never has. Every trial
//! counts its hits on the prediction, so the shapes the library held were
//! credited only on the rare wave that followed one of its own kind — on `.5`
//! (2026-10-17) four trials judged 1,472 shapes and banned every one as
//! rarely hit while those same shapes recurred at a handful of widths. A wave's
//! kind is where it was opened: the `#[track_caller]` location of
//! `CudaDevice::begin_wave_capture`.

use super::exec::{NodeDims, NodeKey};
use std::collections::HashMap;
use std::panic::Location;

/// A recorded segment's `(topology, shape)`: the topology is what the library
/// judges (`regime::Key`), the shape what the diagnostics compare.
pub(super) type Recorded = (u64, u64);

/// Where a wave was opened: the file, line and column of its
/// `begin_wave_capture`.
pub(super) type WaveKind = (&'static str, u32, u32);

/// The kind of the wave `at` opens.
pub(super) fn wave_kind(at: &'static Location<'static>) -> WaveKind {
    (at.file(), at.line(), at.column())
}

/// One wave kind's history, by segment ordinal.
#[derive(Default)]
pub(super) struct Ordinals {
    /// The topology each ordinal held in the last wave that recorded it —
    /// what a new topology is compared against when it is logged.
    pub(super) topology: Vec<Vec<NodeKey>>,
    /// The node dimensions each ordinal held in the last wave that recorded
    /// it — what a new shape of the same topology is compared against.
    pub(super) dims: Vec<Vec<NodeDims>>,
    /// What each ordinal held in the last wave that recorded it — the
    /// prediction a segment's mode is decided by.
    pub(super) key: Vec<Option<Recorded>>,
    /// Segments each ordinal has run eagerly in a row on its prediction alone
    /// (`hub::RECHECK_EVERY`).
    pub(super) eager_runs: Vec<u32>,
}

impl Ordinals {
    /// What `ordinal` is predicted to hold.
    pub(super) fn predicted(&self, ordinal: usize) -> Option<Recorded> {
        self.key.get(ordinal).copied().flatten()
    }

    /// Grow every table to hold `ordinal`.
    pub(super) fn reach(&mut self, ordinal: usize) {
        let n = ordinal + 1;
        if self.topology.len() < n {
            self.topology.resize_with(n, Vec::new);
        }
        if self.dims.len() < n {
            self.dims.resize_with(n, Vec::new);
        }
        if self.key.len() < n {
            self.key.resize(n, None);
        }
        if self.eager_runs.len() < n {
            self.eager_runs.resize(n, 0);
        }
    }
}

/// Every wave kind's [`Ordinals`].
#[derive(Default)]
pub(super) struct OrdinalHistory {
    by_kind: HashMap<WaveKind, Ordinals>,
}

impl OrdinalHistory {
    /// `kind`'s history, grown to hold `ordinal`.
    pub(super) fn at(&mut self, kind: WaveKind, ordinal: usize) -> &mut Ordinals {
        let o = self.by_kind.entry(kind).or_default();
        o.reach(ordinal);
        o
    }

    /// Forget every prediction.
    #[cfg(test)]
    pub(super) fn clear(&mut self) {
        self.by_kind.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VERIFY: WaveKind = ("spec.rs", 338, 9);
    const DRAFT: WaveKind = ("draft_walk.rs", 171, 9);

    /// **Each kind of wave predicts from its own history**: a draft wave
    /// recording ordinal 0 between two verify waves leaves the verify's
    /// prediction for ordinal 0 standing.
    #[test]
    fn kinds_of_wave_keep_their_own_predictions() {
        let mut h = OrdinalHistory::default();
        h.at(VERIFY, 0).key[0] = Some((1, 10));
        h.at(DRAFT, 0).key[0] = Some((2, 20));
        assert_eq!(h.at(VERIFY, 0).predicted(0), Some((1, 10)));
        assert_eq!(h.at(DRAFT, 0).predicted(0), Some((2, 20)));
    }

    /// An ordinal no wave of the kind has reached predicts nothing, and
    /// reaching it grows every table.
    #[test]
    fn an_unreached_ordinal_predicts_nothing() {
        let mut h = OrdinalHistory::default();
        let o = h.at(VERIFY, 3);
        assert_eq!(o.predicted(3), None);
        assert_eq!(o.predicted(7), None);
        assert_eq!(
            (
                o.topology.len(),
                o.dims.len(),
                o.key.len(),
                o.eager_runs.len()
            ),
            (4, 4, 4, 4)
        );
    }

    /// A clear forgets every kind's predictions.
    #[test]
    fn a_clear_forgets_every_prediction() {
        let mut h = OrdinalHistory::default();
        h.at(VERIFY, 0).key[0] = Some((1, 10));
        h.clear();
        assert_eq!(h.at(VERIFY, 0).predicted(0), None);
    }
}
