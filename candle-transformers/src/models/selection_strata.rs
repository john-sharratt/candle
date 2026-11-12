//! How a selecting attention divides a query's candidates before ranking them,
//! stated in positions (`docs/qsa_stratified_selection.md`).
//!
//! This is the caller's side of the setting, and it names no architecture: an
//! engine states one [`StrataTokens`] and hands it to whatever model it loaded
//! through `ManagedBatchedModel::set_selection_strata`. A model whose attention
//! selects turns it into its own blocks at its own compression ratio
//! (Qwen3.8-Flash-Next: `qwen4exp::qsa_select::Strata`); a model whose attention
//! reads the whole causal prefix has no candidates to divide, and the setting
//! is nothing to it.

/// How the recent span enters a stratified selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recent {
    /// Ranked in every window beside that window's own blocks — more chances to
    /// be chosen, and still chosen only on score.
    Candidate,
    /// Attended whole, and left out of every window's ranking.
    Forced,
}

/// A stratified selection in positions: the candidates cut into windows
/// walking forward from the start, each spending the whole budget on its own
/// blocks plus the system prompt's, with the recent span nearest the query
/// ranked in, or forced into, every one — and, outside every window, the
/// sequence's pinned ranges attended whole
/// (`docs/hit_driven_projection.md` §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrataTokens {
    /// Positions per window; `0` is one window spanning every candidate.
    pub window: usize,
    /// Positions nearest the query that form the recent span.
    pub recent: usize,
    /// How the recent span enters the selection.
    pub mode: Recent,
    /// The most positions one sequence may pin. Pinned blocks are attended in
    /// addition to the budget, so this bounds every row's width; a row whose
    /// pins cover more gives up the oldest of its last pinned range, and with
    /// a bound of `0` nothing is pinned at all.
    pub pinned: usize,
}

impl StrataTokens {
    /// One window over every candidate, no recent span and nothing pinned: the
    /// checkpoint's own selection, exactly.
    pub const WHOLE: Self = Self {
        window: 0,
        recent: 0,
        mode: Recent::Candidate,
        pinned: 0,
    };

    /// What an engine runs unless told otherwise: windows of 262,144
    /// positions — the trained window, the length a selecting checkpoint's
    /// indexer was trained to rank — and the 8K positions nearest the query
    /// ranked as a candidate in every window. A context inside the trained
    /// window is one window, so this selects exactly what [`Self::WHOLE`] does
    /// there; a deeper one is ranked a trained window at a time. Nothing is
    /// pinned unless the engine says how much may be.
    pub const DEFAULT: Self = Self {
        window: 262_144,
        recent: 8192,
        mode: Recent::Candidate,
        pinned: 0,
    };

    /// These strata, letting a sequence pin up to `positions` positions.
    pub fn with_pinned(self, positions: usize) -> Self {
        Self {
            pinned: positions,
            ..self
        }
    }
}

impl Default for StrataTokens {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::{Recent, StrataTokens};

    #[test]
    fn the_default_is_trained_window_windows_with_an_8k_recent_candidate() {
        assert_eq!(
            StrataTokens::default(),
            StrataTokens {
                window: 262_144,
                recent: 8192,
                mode: Recent::Candidate,
                pinned: 0,
            }
        );
        assert_ne!(StrataTokens::default(), StrataTokens::WHOLE);
        assert_eq!(StrataTokens::WHOLE.with_pinned(1024).pinned, 1024);
    }
}
