//! What zend asks of the engine it shares, in one place.
//!
//! zend's load hands these to its model builder, and a host that runs its
//! other daemons without zend seats them in zend's place
//! ([`crate::cli::ZendArgs::engine_requests`]), so the engine is the same one
//! whichever daemons run.

use candle_conversation::models::{ModelBuilder, StrataTokens};
use candle_conversation::projection::Builder;

use crate::session::MAX_TURN_TOKENS;

/// `builder` asking for zend's engine-wide settings: the schema's `hits:`
/// layers and its pinned bound (engine-wide, so read once here rather than per
/// layer), the selection the flags name, zend's longest turn, and thinking on
/// at the model level — per-turn suppression is the section tree's `no_think`
/// selector, not this switch.
pub(crate) fn ask(
    builder: ModelBuilder,
    schema: &Builder,
    qsa_selection_budget: Option<usize>,
    qsa_strata: StrataTokens,
) -> ModelBuilder {
    let s = schema.schema();
    builder
        .qsa_selection_budget(qsa_selection_budget)
        .qsa_strata(qsa_strata.with_pinned(s.system_prompt.pins.tokens))
        .hit_layers(s.hits.layers.clone())
        .max_response_tokens(MAX_TURN_TOKENS)
        .thinking(true)
}
