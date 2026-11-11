//! A turn that opens with a tool call rather than choosing whether to.
//!
//! A turn under the `<tool_call>` trigger enters the call grammar only when the
//! model writes the marker, so "no call at all" is always reachable: the model
//! may end the turn on its first token. A turn whose first act must be a call
//! — a file's reading, which reads before it says anything — is given the
//! marker as its assistant prefill and begins inside the very loop the trigger
//! would have entered ([`TurnOptions::turn_grammar`](crate::TurnOptions)), so
//! its tokens are the ones a model that chose to call writes, and the calls
//! after the first are offered exactly as the trigger's loop offers them.

use std::sync::Arc;

use super::tree::StencilTree;

/// What a turn opening with a call is given: the marker the trigger would
/// have consumed, prefilled as text, and the call loop it begins inside.
#[derive(Debug, Clone)]
pub struct OpeningCall {
    /// The dialect's call marker — `<tool_call>`.
    pub marker: String,
    /// The call loop over the turn's tools, as the trigger compiles it.
    pub tree: Arc<StencilTree>,
}
