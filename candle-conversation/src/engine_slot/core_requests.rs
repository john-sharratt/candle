//! What tenants ask of the engine they share, and how two asks become one.
//!
//! Some builder settings belong to the model or the scheduler rather than to
//! a substrate, so tenants sharing one engine can ask for them differently.
//! Each tenant's are collected before the engine is built and merged by the
//! rule `docs/shared_model_host.md` §3.3 gives each one; the engine is built
//! with the merge.

use crate::guest::GuestRegistry;
use crate::models::StrataTokens;

/// The engine-wide settings one tenant asks for.
pub struct CoreRequests {
    /// Tokens one prefill forward carries; `None` sizes it to the card.
    pub prefill_pass_tokens: Option<usize>,
    /// The lowest progressive-YaRN factor any sequence runs at.
    pub min_rope_factor: Option<f32>,
    /// A QSA selection budget in place of the checkpoint's own.
    pub qsa_selection_budget: Option<usize>,
    /// How a selecting attention divides the context.
    pub qsa_strata: StrataTokens,
    /// The longest reply any conversation may decode.
    pub max_response_tokens: usize,
    /// How many conversations may run at once.
    pub max_concurrent: usize,
    /// The model-level thinking switch.
    pub thinking: Option<bool>,
    /// Co-resident models that borrow the card between waves.
    pub guests: GuestRegistry,
}

impl CoreRequests {
    /// Merge every tenant's requests, highest-ranked first — a daemon the
    /// process leaves out still asks at its rank, so `asks[0]` is the
    /// highest-ranked seat, running or not:
    ///
    /// | setting | rule |
    /// |---|---|
    /// | `prefill_pass_tokens`, `min_rope_factor`, `max_response_tokens` | the largest asked |
    /// | `qsa_selection_budget`, `qsa_strata` | the highest-ranked tenant's |
    /// | `max_concurrent` | the sum — each tenant runs its own conversations |
    /// | `thinking` | on when any tenant turns it on; per-turn modes decide |
    /// | `guests` | the union |
    ///
    /// `None` when `asks` is empty.
    pub fn merge(asks: Vec<CoreRequests>) -> Option<CoreRequests> {
        let mut asks = asks.into_iter();
        let mut merged = asks.next()?;
        for ask in asks {
            merged.prefill_pass_tokens =
                max_some(merged.prefill_pass_tokens, ask.prefill_pass_tokens);
            merged.min_rope_factor = match (merged.min_rope_factor, ask.min_rope_factor) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            };
            merged.qsa_selection_budget = merged.qsa_selection_budget.or(ask.qsa_selection_budget);
            merged.max_response_tokens = merged.max_response_tokens.max(ask.max_response_tokens);
            merged.max_concurrent += ask.max_concurrent;
            merged.thinking = match (merged.thinking, ask.thinking) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (a, b) => a.or(b),
            };
            merged.guests.merge(ask.guests);
        }
        Some(merged)
    }
}

fn max_some(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zend() -> CoreRequests {
        CoreRequests {
            prefill_pass_tokens: None,
            min_rope_factor: Some(2.0),
            qsa_selection_budget: Some(4096),
            qsa_strata: StrataTokens::WHOLE,
            max_response_tokens: 65_536,
            max_concurrent: 4,
            thinking: Some(true),
            guests: GuestRegistry::new(),
        }
    }

    fn npcd() -> CoreRequests {
        CoreRequests {
            prefill_pass_tokens: Some(8192),
            min_rope_factor: None,
            qsa_selection_budget: None,
            qsa_strata: StrataTokens::DEFAULT,
            max_response_tokens: 16_384,
            max_concurrent: 4,
            thinking: None,
            guests: GuestRegistry::new(),
        }
    }

    /// The §3.3 table, row by row, for the two tenants it was written for.
    #[test]
    fn two_tenants_merge_by_each_settings_rule() {
        let m = CoreRequests::merge(vec![zend(), npcd()]).unwrap();
        assert_eq!(m.prefill_pass_tokens, Some(8192), "max");
        assert_eq!(
            m.min_rope_factor,
            Some(2.0),
            "max — the RoPE schedule is the model's"
        );
        assert_eq!(m.qsa_selection_budget, Some(4096), "the first tenant's");
        assert_eq!(m.qsa_strata, StrataTokens::WHOLE, "the first tenant's");
        assert_eq!(m.max_response_tokens, 65_536, "max");
        assert_eq!(m.max_concurrent, 8, "each tenant's conversations, summed");
        assert_eq!(m.thinking, Some(true));
    }

    #[test]
    fn one_tenants_requests_are_its_own() {
        let m = CoreRequests::merge(vec![npcd()]).unwrap();
        assert_eq!(m.prefill_pass_tokens, Some(8192));
        assert_eq!(m.max_concurrent, 4);
        assert_eq!(m.thinking, None);
        assert!(CoreRequests::merge(Vec::new()).is_none());
    }
}
