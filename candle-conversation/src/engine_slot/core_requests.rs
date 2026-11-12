//! What tenants ask of the engine they share, and how two asks become one.
//!
//! Some builder settings belong to the model or the scheduler rather than to
//! a substrate, so tenants sharing one engine can ask for them differently.
//! Each tenant's are collected before the engine is built and merged by the
//! rule `docs/shared_model_host.md` §3.3 gives each one; the engine is built
//! with the merge.

use crate::guest::GuestRegistry;
use crate::models::{HitPercentiles, StrataTokens};

/// The engine-wide settings one tenant asks for.
pub struct CoreRequests {
    /// Tokens one prefill forward carries; `None` sizes it to the card.
    pub prefill_pass_tokens: Option<usize>,
    /// A QSA selection budget in place of the checkpoint's own.
    pub qsa_selection_budget: Option<usize>,
    /// How a selecting attention divides the context.
    pub qsa_strata: StrataTokens,
    /// Which selecting layers count hits; `None` asks for none in particular.
    pub hit_layers: Option<HitPercentiles>,
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
    /// | `prefill_pass_tokens`, `max_response_tokens` | the largest asked |
    /// | `qsa_selection_budget`, `hit_layers` | the highest-ranked tenant that asks |
    /// | `qsa_strata` | the highest-ranked tenant's |
    /// | `qsa_strata.pinned` | the largest asked — every tenant's pins fit the rows |
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
            merged.qsa_selection_budget = merged.qsa_selection_budget.or(ask.qsa_selection_budget);
            merged.hit_layers = merged.hit_layers.take().or(ask.hit_layers);
            // The strata are the first tenant's, but every tenant's pins must
            // fit the rows they size, so the pinned bound is the largest asked.
            merged.qsa_strata.pinned = merged.qsa_strata.pinned.max(ask.qsa_strata.pinned);
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
            qsa_selection_budget: Some(4096),
            qsa_strata: StrataTokens::WHOLE,
            hit_layers: Some(HitPercentiles::new(vec![50]).unwrap()),
            max_response_tokens: 65_536,
            max_concurrent: 4,
            thinking: Some(true),
            guests: GuestRegistry::new(),
        }
    }

    fn npcd() -> CoreRequests {
        CoreRequests {
            prefill_pass_tokens: Some(8192),
            qsa_selection_budget: None,
            qsa_strata: StrataTokens::DEFAULT,
            hit_layers: None,
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
        assert_eq!(m.qsa_selection_budget, Some(4096), "the first tenant's");
        assert_eq!(m.qsa_strata, StrataTokens::WHOLE, "the first tenant's");
        assert_eq!(
            m.hit_layers.as_ref().map(HitPercentiles::percentiles),
            Some(&[50u8][..]),
            "the first tenant's"
        );
        // A tenant that asks for none takes the next one's.
        let m = CoreRequests::merge(vec![npcd(), zend()]).unwrap();
        assert_eq!(
            m.hit_layers.as_ref().map(HitPercentiles::percentiles),
            Some(&[50u8][..]),
            "npcd asks none, so zend's"
        );
        // A lower-ranked tenant's pins still fit: the bound is the largest.
        let mut pinning = npcd();
        pinning.qsa_strata = pinning.qsa_strata.with_pinned(1024);
        let m = CoreRequests::merge(vec![zend(), pinning]).unwrap();
        assert_eq!(m.qsa_strata.window, StrataTokens::WHOLE.window);
        assert_eq!(m.qsa_strata.pinned, 1024);
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
