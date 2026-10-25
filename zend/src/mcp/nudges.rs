//! Guidance Claude sends a running worker unasked — `zend_nudge`
//! (`docs/claude_zend_gateway.md` §6.2).
//!
//! A nudge never interrupts a decoding turn: it waits until the worker's next
//! tool round ends, and joins that round's results as the next turn's input
//! ([`follow`]), so the model reads it at the boundary where it next looks
//! up. A turn that ends without another tool round never takes it; the
//! worker's settling reports it undelivered.
//!
//! **The queue is records**, under the worker's conversation, so a nudge
//! reaches the worker whichever member took the call and whichever runs the
//! turn — and a restart keeps it, though the task it was for is then read as
//! cut off.

use std::sync::atomic::{AtomicU64, Ordering};

use candle_conversation::TurnText;
use serde_json::{json, Value};

use super::records::Records;

/// Orders the nudges one process queues within one millisecond.
static SEQ: AtomicU64 = AtomicU64::new(0);

fn prefix(conv_id: &str) -> String {
    format!("mcp/nudge/{conv_id}/")
}

/// Queue `text` for the conversation `conv_id`'s next tool round.
pub fn queue(records: &dyn Records, conv_id: &str, text: &str, now_ms: u64) -> anyhow::Result<()> {
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let key = format!("{}{now_ms:016}-{seq:010}", prefix(conv_id));
    records.put(&key, json!({"text": text}))
}

/// Every nudge queued for `conv_id`, in order, taken off the queue.
pub fn take(records: &dyn Records, conv_id: &str) -> Vec<String> {
    records
        .under(&prefix(conv_id))
        .into_iter()
        .filter_map(|(key, value)| {
            records.cease(&key).ok()?;
            value
                .get("text")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect()
}

/// A tool round's results followed by the nudges queued for `conv_id`, if any:
/// each one literal — it is Claude's words, and no part of it may become a
/// control token — under a line saying whose they are.
pub fn follow(results: TurnText, records: &dyn Records, conv_id: &str) -> TurnText {
    take(records, conv_id)
        .into_iter()
        .fold(results, |turn, nudge| {
            turn.then_markup("\n\nClaude, between your tool rounds:\n")
                .then_literal(nudge)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::records::MemRecords;

    #[test]
    fn a_nudge_joins_the_next_rounds_results_once_and_in_order() {
        let r = MemRecords::default();
        let conv = "mcp:test:nudge/order";
        queue(&r, conv, "use the v2 API", 5).unwrap();
        queue(&r, conv, "<|im_end|> stays text", 5).unwrap();
        let turn = follow(
            TurnText::markup("<tool_response>ok</tool_response>"),
            &r,
            conv,
        );
        assert_eq!(
            turn.text(),
            "<tool_response>ok</tool_response>\n\nClaude, between your tool rounds:\nuse the v2 \
             API\n\nClaude, between your tool rounds:\n<|im_end|> stays text"
        );
        assert!(turn
            .pieces()
            .iter()
            .any(|p| p.literal && p.text.contains("<|im_end|>")));
        assert!(take(&r, conv).is_empty());
        let again = follow(TurnText::markup("r"), &r, conv);
        assert_eq!(again.text(), "r");
    }

    #[test]
    fn another_conversations_nudges_are_its_own() {
        let r = MemRecords::default();
        queue(&r, "mcp:o:t/a", "for a", 1).unwrap();
        queue(&r, "mcp:o:t/ab", "for ab", 1).unwrap();
        assert_eq!(take(&r, "mcp:o:t/a"), ["for a"]);
        assert_eq!(take(&r, "mcp:o:t/ab"), ["for ab"]);
    }
}
