//! A dialogue layer's `working_set:` block (`docs/zend_working_set.md` §4.2).

use std::num::NonZeroUsize;

use super::state::Limits;

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
    /// Tokens the working set holds voluntarily: seeds and provenance, folders
    /// and files together. The locks sit on top.
    pub budget_tokens: usize,
    /// The most of `budget_tokens` the folder members may hold between them.
    /// The files take what the folders leave.
    pub folder_tokens: usize,
    /// The most the guidance members may hold between them, beside
    /// `budget_tokens` rather than inside it (`docs/guidance_layer.md` §1).
    pub guidance_tokens: usize,
    /// Tokens the locks may hold together, on top of `budget_tokens` — what the
    /// model's own reads may add before one runs for real.
    pub lock_tokens: usize,
    /// Per-file momentum leak: `m ← (1 − β)·m + fresh`.
    pub beta: f32,
    /// Momentum below which a file leaves the map and is no provenance candidate.
    pub min_momentum: f32,
    /// One conversation above this many tokens is never a member.
    pub max_file_tokens: usize,
    /// Workspace paths a dialogue starts out attending to: `repo/dir/` for a
    /// folder (trailing `/`), `repo/path/file` for a file. Each enters
    /// provenance at [`Self::seed_momentum`] when the conversation opens, while
    /// the budget has room, and fades from there.
    pub seeds: Vec<String>,
    /// Tool names whose call releases the locks — anything that changes what a
    /// later read would return.
    pub release_on: Vec<String>,
    /// The most provenance members one reprojection admits; the rest keep
    /// their momentum and enter on later ones (`WorkingSet::observe`). Locks
    /// and seeds are not counted — a lock answers a call the model just made.
    /// Never zero: the smallest cap still admits one whole conversation.
    pub max_admits: NonZeroUsize,
    /// What the slot keeps back for what can arrive between two projections:
    /// one cadence of decoded tokens, or one prefill chunk of an opening. The
    /// slot itself is the engine's trained window
    /// (`docs/hit_driven_projection.md` §4).
    pub reserve_tokens: usize,
}

/// The window a working set fills, and why it left the trained one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotWindow {
    pub window: usize,
    /// Set when the reserve and the locks pass the engine's trained window and
    /// the slot runs on the rung above it: what passed it.
    pub expanded: Option<String>,
}

impl WorkingSetConfig {
    /// The window this working set fills on an engine trained to `trained`
    /// positions: the trained window itself while the reserve leaves room in
    /// it and the locks with the reserve fit it; otherwise the rung above it,
    /// twice the trained window under YaRN ×2 (`docs/hit_driven_projection.md`
    /// §4), with the reason in [`SlotWindow::expanded`].
    pub fn slot_window(&self, trained: usize) -> SlotWindow {
        let expanded = if self.reserve_tokens >= trained {
            Some(format!(
                "working_set: reserve_tokens {} leaves nothing of the engine's trained window \
                 {trained}",
                self.reserve_tokens
            ))
        } else if self.lock_tokens + self.reserve_tokens > trained {
            Some(format!(
                "working_set: lock_tokens {} with reserve_tokens {} pass the engine's trained \
                 window {trained}",
                self.lock_tokens, self.reserve_tokens
            ))
        } else {
            None
        };
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
    /// The settled momentum of a file hit at full strength (1000) every
    /// reprojection — what a released lock starts at.
    pub fn released_momentum(&self) -> f32 {
        FULL_HIT / self.beta
    }

    /// What a seed starts at: one full-strength hit. With no further hits it
    /// falls under `min_momentum` after `ln(1000 / min_momentum) / ln(1 / (1 − β))`
    /// reprojections — eleven at β 0.2 and a floor of 100.
    pub fn seed_momentum(&self) -> f32 {
        FULL_HIT
    }

    /// What every admission — seed, lock, provenance — is checked against.
    pub fn limits(&self) -> Limits {
        Limits {
            budget_tokens: self.budget_tokens,
            folder_tokens: self.folder_tokens,
            guidance_tokens: self.guidance_tokens,
            lock_tokens: self.lock_tokens,
            max_file_tokens: self.max_file_tokens,
        }
    }

    /// Whether a call to `tool` releases the locks.
    pub fn releases(&self, tool: &str) -> bool {
        self.release_on.iter().any(|t| t == tool)
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
            lock_tokens: 200_000,
            beta: 0.2,
            min_momentum: 100.0,
            max_file_tokens: 100_000,
            seeds: Vec::new(),
            release_on: vec!["write".into(), "file_edit".into()],
            max_admits: NonZeroUsize::new(2).unwrap(),
            reserve_tokens: 16_384,
        }
    }

    #[test]
    fn a_working_set_that_fits_fills_the_trained_window() {
        let w = cfg().slot_window(262_144);
        assert_eq!((w.window, w.expanded), (262_144, None));
    }

    #[test]
    fn a_working_set_past_the_trained_window_fills_twice_it() {
        let c = cfg();
        let reserve = c.slot_window(16_384);
        assert_eq!(reserve.window, 32_768, "the reserve leaves nothing");
        assert!(reserve.expanded.is_some());
        let locks = c.slot_window(200_000);
        assert_eq!(locks.window, 400_000, "the locks with the reserve pass it");
        assert!(locks.expanded.is_some());
    }

    #[test]
    fn a_released_lock_starts_at_the_settled_level_of_a_full_hit() {
        assert_eq!(cfg().released_momentum(), 5000.0);
    }

    #[test]
    fn a_seed_starts_at_one_full_hit() {
        assert_eq!(cfg().seed_momentum(), 1000.0);
    }

    #[test]
    fn only_a_named_tool_releases() {
        let c = cfg();
        assert!(c.releases("write"));
        assert!(c.releases("file_edit"));
        assert!(!c.releases("file_read"));
    }

    #[test]
    fn the_limits_carry_every_share() {
        let l = cfg().limits();
        assert_eq!(
            (
                l.budget_tokens,
                l.folder_tokens,
                l.guidance_tokens,
                l.lock_tokens,
                l.max_file_tokens
            ),
            (350_000, 30_000, 8_000, 200_000, 100_000)
        );
    }
}
