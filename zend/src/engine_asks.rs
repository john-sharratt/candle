//! What zend asks of the engine it shares, in one place.
//!
//! zend's load hands these to its model builder, and a host that runs its
//! other daemons without zend seats them in zend's place
//! ([`crate::cli::ZendArgs::engine_requests`]), so the engine is the same one
//! whichever daemons run. The RoPE floor is part of every substrate's model
//! key: an engine built without zend's would refuse the substrates its
//! tenants wrote beside zend.

use candle_conversation::models::{ModelBuilder, StrataTokens};
use candle_conversation::projection::Builder;

use crate::session::MAX_TURN_TOKENS;

/// `builder` asking for zend's engine-wide settings: the schema's `rope:
/// min_yarn_factor:` (engine-wide, so read once here rather than per layer),
/// the selection the flags name, zend's longest turn, and thinking on at the
/// model level — per-turn suppression is the section tree's `no_think`
/// selector, not this switch.
pub(crate) fn ask(
    builder: ModelBuilder,
    schema: &Builder,
    qsa_selection_budget: Option<usize>,
    qsa_strata: StrataTokens,
) -> ModelBuilder {
    builder
        .qsa_selection_budget(qsa_selection_budget)
        .qsa_strata(qsa_strata)
        .min_rope_factor(schema.schema().min_yarn_factor)
        .max_response_tokens(MAX_TURN_TOKENS)
        .thinking(true)
}
