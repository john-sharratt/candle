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
/// ranked in, or forced into, every one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrataTokens {
    /// Positions per window; `0` is one window spanning every candidate.
    pub window: usize,
    /// Positions nearest the query that form the recent span.
    pub recent: usize,
    /// How the recent span enters the selection.
    pub mode: Recent,
}

impl StrataTokens {
    /// One window over every candidate and no recent span: the checkpoint's
    /// own selection, exactly.
    pub const WHOLE: Self = Self {
        window: 0,
        recent: 0,
        mode: Recent::Candidate,
    };

    /// What an engine runs unless told otherwise: 128K-position windows — the
    /// span a selecting checkpoint is trained to choose among — and the 8K
    /// positions nearest the query ranked as a candidate in every window.
    ///
    /// Measured on Qwen3.8-Flash-Next at a 294–394K production prompt
    /// (`docs/results/qsa_stratified_selection_2026-10-06.md`): every recall
    /// probe answered, against the forced span's confused post-tool reasoning
    /// at +93% decode; this costs about +15% decode over [`Self::WHOLE`]. Below
    /// one window's depth it selects exactly what [`Self::WHOLE`] does — the
    /// recent span and the prompt are then inside the only window.
    pub const DEFAULT: Self = Self {
        window: 131_072,
        recent: 8192,
        mode: Recent::Candidate,
    };
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
    fn the_default_is_128k_windows_with_an_8k_recent_candidate() {
        assert_eq!(
            StrataTokens::default(),
            StrataTokens {
                window: 131_072,
                recent: 8192,
                mode: Recent::Candidate,
            }
        );
        assert_ne!(StrataTokens::default(), StrataTokens::WHOLE);
    }
}
