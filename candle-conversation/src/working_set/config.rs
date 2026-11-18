//! A dialogue layer's `working_set:` block (`docs/zend_working_set.md` §4.2).

use std::num::NonZeroUsize;

use super::state::Limits;
use super::type_shares::TypeShares;

/// The fresh score of a full-strength hit — the top of the normalized band a
/// reprojection scores a file on.
const FULL_HIT: f32 = 1000.0;

/// How much already-ingested content a dialogue carries ahead of its own turns,
/// and how that content comes and goes.
///
/// Declared on the dialogue layer. A projection whose target layer declares
/// none selects nothing into its `working_set` groups — every ingest
/// conversation is such a target.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkingSetConfig {
    /// Tokens the working set holds: seeds and provenance, folders and files
    /// together.
    pub budget_tokens: usize,
    /// The most of `budget_tokens` the folder members may hold between them.
    /// The files take what the folders leave.
    pub folder_tokens: usize,
    /// The most the guidance members may hold between them, beside
    /// `budget_tokens` rather than inside it (`docs/guidance_layer.md` §1).
    pub guidance_tokens: usize,
    /// Per-file momentum leak: `m ← (1 − β)·m + fresh`.
    pub beta: f32,
    /// Momentum below which a file leaves the map and is no provenance candidate.
    pub min_momentum: f32,
    /// The most of one reading a slot places ([`Limits::max_file_tokens`]).
    pub max_file_tokens: usize,
    /// Workspace paths a dialogue starts out attending to: `repo/dir/` for a
    /// folder (trailing `/`), `repo/path/file` for a file. Each enters
    /// provenance at [`Self::seed_momentum`] when the conversation opens, while
    /// the budget has room, and fades from there.
    pub seeds: Vec<String>,
    /// The most provenance members one reprojection admits; the rest keep
    /// their momentum and enter on later ones (`WorkingSet::observe`). Seeds
    /// are not counted. Never zero: the smallest cap still admits one whole
    /// conversation.
    pub max_admits: NonZeroUsize,
    /// What the slot keeps back for what can arrive between two projections:
    /// one cadence of decoded tokens, or one prefill chunk of an opening. The
    /// slot itself is the engine's trained window
    /// (`docs/hit_driven_projection.md` §4).
    pub reserve_tokens: usize,
    /// The most of `budget_tokens` each type of document may hold
    /// (`docs/zend_working_set.md` §4.10); a unit of no declared type is
    /// unbounded.
    pub type_shares: TypeShares,
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

impl WorkingSetConfig {
    /// What a seed starts at: one full-strength hit. With no further hits it
    /// falls under `min_momentum` after `ln(1000 / min_momentum) / ln(1 / (1 − β))`
    /// reprojections — eleven at β 0.2 and a floor of 100.
    pub fn seed_momentum(&self) -> f32 {
        FULL_HIT
    }

    /// What every admission — seed or provenance — is checked against.
    pub fn limits(&self) -> Limits {
        Limits {
            budget_tokens: self.budget_tokens,
            folder_tokens: self.folder_tokens,
            guidance_tokens: self.guidance_tokens,
            max_file_tokens: self.max_file_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> WorkingSetConfig {
        WorkingSetConfig {
            budget_tokens: 350_000,
            folder_tokens: 30_000,
            guidance_tokens: 8_000,
            beta: 0.2,
            min_momentum: 100.0,
            max_file_tokens: 100_000,
            seeds: Vec::new(),
            max_admits: NonZeroUsize::new(2).unwrap(),
            reserve_tokens: 16_384,
            type_shares: TypeShares::default(),
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

    #[test]
    fn a_seed_starts_at_one_full_hit() {
        assert_eq!(cfg().seed_momentum(), 1000.0);
    }

    #[test]
    fn the_limits_carry_every_share() {
        let l = cfg().limits();
        assert_eq!(
            (
                l.budget_tokens,
                l.folder_tokens,
                l.guidance_tokens,
                l.max_file_tokens
            ),
            (350_000, 30_000, 8_000, 100_000)
        );
    }
}
