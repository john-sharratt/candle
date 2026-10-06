//! A dialogue layer's `working_set:` block (`docs/zend_working_set.md` §4.2).

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
    /// Tokens the whole working set may hold: seeds, locks and provenance,
    /// folders and files together.
    pub budget_tokens: usize,
    /// The most of `budget_tokens` the folder members may hold between them.
    /// The files take what the folders leave.
    pub folder_tokens: usize,
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
    pub max_admits: usize,
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
            beta: 0.2,
            min_momentum: 100.0,
            max_file_tokens: 100_000,
            seeds: Vec::new(),
            release_on: vec!["write".into(), "file_edit".into()],
            max_admits: 2,
        }
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
    fn the_limits_carry_both_shares() {
        let l = cfg().limits();
        assert_eq!(
            (l.budget_tokens, l.folder_tokens, l.max_file_tokens),
            (350_000, 30_000, 100_000)
        );
    }
}
