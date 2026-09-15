//! One bubble of a conversation recovered from the substrate — what
//! [`Sequence::recovered_history`](crate::Sequence::recovered_history) returns.

use crate::turn::Role;
use crate::turn_layout::ThinkingLength;

/// The markup that opens each tool result in a tool-response turn — one
/// `<tool_response>…</tool_response>` block per call, in call order.
pub(crate) const TOOL_RESPONSE_OPEN: &str = "<tool_response>";

#[derive(Debug, Clone)]
pub struct RecoveredMessage {
    pub role: Role,
    pub text: String,
    /// The turn's recorded thinking-suppressed flag, set on the USER bubble so
    /// the GUI can re-render the `/no_think` soft-switch on prior turns exactly
    /// as the assembler does for the model (see `turn_no_think`); `false` on the
    /// assistant bubble.
    pub no_think: bool,
    /// On the ASSISTANT bubble, how long the turn's reasoning was
    /// ([`TurnLayout::thinking_length`](crate::turn_layout::TurnLayout::thinking_length)).
    /// `None` on the user bubble and for a turn that did not reason.
    pub thinking: Option<ThinkingLength>,
    /// On a USER bubble that carries tool results: each result's length in
    /// tokens as it sits in the context, one per `<tool_response>` block in
    /// order — counted off the turn's own sealed ids (see
    /// [`tool_response_lengths`]). Empty everywhere else.
    pub tool_tokens: Vec<u32>,
}

/// Split the ids of a tool-response turn's user body into its blocks and
/// measure each: a block runs from one `open` marker to the next (or to the end
/// of the body), so it counts the wrapper, the result and whatever separates it
/// from the next — exactly the ids that result put into the context. Ids before
/// the first marker belong to no block. Empty when `open` is empty or never
/// occurs.
pub(crate) fn tool_response_lengths(ids: &[u32], open: &[u32]) -> Vec<u32> {
    if open.is_empty() || ids.len() < open.len() {
        return Vec::new();
    }
    let starts: Vec<usize> = (0..=ids.len() - open.len())
        .filter(|&i| ids[i..i + open.len()] == *open)
        .collect();
    starts
        .iter()
        .enumerate()
        .map(|(k, &start)| {
            let end = starts.get(k + 1).copied().unwrap_or(ids.len());
            (end - start) as u32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPEN: u32 = 151665;
    const CLOSE: u32 = 151666;
    const NL: u32 = 198;

    #[test]
    fn each_block_runs_from_its_marker_to_the_next() {
        // <tool_response> 5 6 7 </tool_response> \n <tool_response> 8 </tool_response> \n
        let ids = [OPEN, 5, 6, 7, CLOSE, NL, OPEN, 8, CLOSE, NL];
        assert_eq!(tool_response_lengths(&ids, &[OPEN]), vec![6, 4]);
    }

    #[test]
    fn a_multi_token_marker_is_matched_whole() {
        let open = [60, 61];
        let ids = [60, 61, 9, 9, 60, 9, 60, 61, 9];
        assert_eq!(tool_response_lengths(&ids, &open), vec![6, 3]);
    }

    #[test]
    fn a_body_with_no_marker_has_no_blocks() {
        assert_eq!(
            tool_response_lengths(&[1, 2, 3], &[OPEN]),
            Vec::<u32>::new()
        );
        assert_eq!(tool_response_lengths(&[], &[OPEN]), Vec::<u32>::new());
        assert_eq!(tool_response_lengths(&[OPEN, 1], &[]), Vec::<u32>::new());
    }

    #[test]
    fn ids_before_the_first_marker_belong_to_no_block() {
        let ids = [4, 4, OPEN, 5, CLOSE];
        assert_eq!(tool_response_lengths(&ids, &[OPEN]), vec![3]);
    }
}
