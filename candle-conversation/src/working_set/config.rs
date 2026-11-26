//! A dialogue layer's `working_set:` block (`docs/zend_working_set.md` §4.2).

use super::fill_shares::FillShares;
use super::type_shares::TypeShares;

/// How a dialogue's pool is ranked and its slot shared out.
///
/// Declared on the dialogue layer. A projection whose target layer declares
/// none selects nothing into its `working_set` groups — every ingest
/// conversation is such a target. The pool's size is the card's, not the
/// schema's (`super::PoolSize`).
#[derive(Debug, Clone, PartialEq)]
pub struct WorkingSetConfig {
    /// Per-piece momentum leak: `m ← (1 − β)·m + fresh`.
    pub beta: f32,
    /// Strength below which a piece leaves the map and is no pool candidate.
    pub min_momentum: f32,
    /// What the slot keeps back for what can arrive between two projections:
    /// one cadence of decoded tokens, or one prefill chunk of an opening. The
    /// slot itself is the engine's trained window
    /// (`docs/hit_driven_projection.md` §4).
    pub reserve_tokens: usize,
    /// The most of the pool's tokens each type of document may hold
    /// (`docs/zend_working_set.md` §4.10); a piece of no declared type is
    /// unbounded.
    pub type_shares: TypeShares,
    /// Each kind's share of the slot's room: the slot fill's caps
    /// (`docs/hit_driven_projection.md` §7.2 step 4).
    pub fill_shares: FillShares,
}

/// The window a working set fills, and why it left the trained one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotWindow {
    pub window: usize,
    /// Set when the reserve passes the engine's trained window and the slot
    /// runs on the rung above it: what passed it.
    pub expanded: Option<String>,
}

impl WorkingSetConfig {
    /// The window this working set fills on an engine trained to `trained`
    /// positions: the trained window itself while the reserve leaves room in
    /// it; otherwise the rung above it, twice the trained window under YaRN ×2
    /// (`docs/hit_driven_projection.md` §4), with the reason in
    /// [`SlotWindow::expanded`].
    pub fn slot_window(&self, trained: usize) -> SlotWindow {
        let expanded = (self.reserve_tokens >= trained).then(|| {
            format!(
                "working_set: reserve_tokens {} leaves nothing of the engine's trained window \
                 {trained}",
                self.reserve_tokens
            )
        });
        SlotWindow {
            window: if expanded.is_some() {
                trained * 2
            } else {
                trained
            },
            expanded,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> WorkingSetConfig {
        WorkingSetConfig {
            beta: 0.2,
            min_momentum: 100.0,
            reserve_tokens: 16_384,
            type_shares: TypeShares::default(),
            fill_shares: FillShares::new(0.225, 0.02, 0.05, 0.105, 0.60).unwrap(),
        }
    }

    #[test]
    fn a_working_set_that_fits_fills_the_trained_window() {
        let w = cfg().slot_window(262_144);
        assert_eq!((w.window, w.expanded), (262_144, None));
    }

    #[test]
    fn a_reserve_past_the_trained_window_fills_twice_it() {
        let reserve = cfg().slot_window(16_384);
        assert_eq!(reserve.window, 32_768, "the reserve leaves nothing");
        assert!(reserve.expanded.is_some());
    }
}
