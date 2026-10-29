//! The size a placement prices a turn at.

use crate::frame::Capacity;
use crate::gateway::registry::MemberView;

/// The prefill a turn brings, in tokens, when no candidate has said what the
/// turns it begins bring ([`TurnSize`]).
pub const NOMINAL_PROMPT_TOKENS: u64 = 2048;

/// The reply a turn decodes, in tokens, when no candidate has said.
pub const NOMINAL_REPLY_TOKENS: u64 = 512;

/// The size a placement prices a turn at: its prompt and its reply, in tokens.
///
/// **One size for every candidate.** A request carries only its new message;
/// the prompt a member prefills is what its projection makes of the
/// conversation, the same on whichever member serves it. So the size is the
/// turn's, not the member's: the mean of what the candidates report the turns
/// they begin bring (`Capacity::prompt_tokens`, `reply_tokens`), applied to
/// them all. Priced at each member's own, one that had served only short turns
/// would look cheap for every turn. A fixed nominal priced heavy-decode's
/// 30-token prompts as 2,048, and its 256-token replies as 512, against a
/// card whose wide prefill rate was steep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnSize {
    pub prompt: u64,
    pub reply: u64,
}

impl TurnSize {
    /// The mean of what `pool` reports, each figure over the candidates that
    /// report one; the nominal where none does.
    pub fn of(pool: &[&MemberView]) -> TurnSize {
        let mean = |f: fn(&Capacity) -> u32, nominal: u64| {
            let said: Vec<u64> = pool
                .iter()
                .filter_map(|m| m.capacity.as_ref())
                .map(|c| u64::from(f(c)))
                .filter(|&v| v > 0)
                .collect();
            if said.is_empty() {
                nominal
            } else {
                said.iter().sum::<u64>() / said.len() as u64
            }
        };
        TurnSize {
            prompt: mean(|c| c.prompt_tokens, NOMINAL_PROMPT_TOKENS),
            reply: mean(|c| c.reply_tokens, NOMINAL_REPLY_TOKENS),
        }
    }
}
