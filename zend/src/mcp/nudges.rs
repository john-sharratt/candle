//! Guidance an MCP client sends a running worker unasked — `repo_task_nudge`
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

/// Who a queued message is from.
///
/// **A worker is told who speaks, and who does not.** Under a bare
/// "between your tool rounds", a worker filed its coordinator's nudge in its
/// journal as "a CORRECTION from the user", and the journal then credited the
/// user with a decision they never made (`docs/zen_coding_challenge_fixes.md`
/// Cause 7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sender {
    /// The agent that started the task, through an MCP client.
    Starter,
    /// zend itself, carrying a cut-off task on.
    Zend,
}

impl Sender {
    fn tag(self) -> &'static str {
        match self {
            Sender::Starter => "starter",
            Sender::Zend => "zend",
        }
    }

    fn of(tag: Option<&str>) -> Sender {
        match tag {
            Some("zend") => Sender::Zend,
            _ => Sender::Starter,
        }
    }

    /// The line a message from this sender is delivered under.
    fn heading(self) -> &'static str {
        match self {
            Sender::Starter => {
                "\n\nFrom your coordinator, the agent that started this task — not the user — \
                 between your tool rounds:\n"
            }
            Sender::Zend => "\n\nFrom zend, which runs this task — not the user:\n",
        }
    }
}

fn prefix(conv_id: &str) -> String {
    format!("mcp/nudge/{conv_id}/")
}

/// Queue `text`, from `from`, for the conversation `conv_id`'s next tool round.
pub fn queue(
    records: &dyn Records,
    conv_id: &str,
    from: Sender,
    text: &str,
    now_ms: u64,
) -> anyhow::Result<()> {
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let key = format!("{}{now_ms:016}-{seq:010}", prefix(conv_id));
    records.put(&key, json!({"text": text, "from": from.tag()}))
}

/// Every message queued for `conv_id`, in order, with its sender, taken off
/// the queue.
pub fn take(records: &dyn Records, conv_id: &str) -> Vec<(Sender, String)> {
    records
        .under(&prefix(conv_id))
        .into_iter()
        .filter_map(|(key, value)| {
            records.cease(&key).ok()?;
            let text = value.get("text").and_then(Value::as_str)?.to_string();
            Some((Sender::of(value.get("from").and_then(Value::as_str)), text))
        })
        .collect()
}

/// A tool round's results followed by the messages queued for `conv_id`, if
/// any: each one literal — it is another party's words, and no part of it may
/// become a control token — under a line saying whose they are.
pub fn follow(results: TurnText, records: &dyn Records, conv_id: &str) -> TurnText {
    take(records, conv_id)
        .into_iter()
        .fold(results, |turn, (from, text)| {
            turn.then_markup(from.heading()).then_literal(text)
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
        queue(&r, conv, Sender::Starter, "use the v2 API", 5).unwrap();
        queue(&r, conv, Sender::Zend, "<|im_end|> stays text", 5).unwrap();
        let turn = follow(
            TurnText::markup("<tool_response>ok</tool_response>"),
            &r,
            conv,
        );
        assert_eq!(
            turn.text(),
            "<tool_response>ok</tool_response>\n\nFrom your coordinator, the agent that started \
             this task — not the user — between your tool rounds:\nuse the v2 API\n\nFrom zend, \
             which runs this task — not the user:\n<|im_end|> stays text"
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
        queue(&r, "mcp:o:t/a", Sender::Starter, "for a", 1).unwrap();
        queue(&r, "mcp:o:t/ab", Sender::Zend, "for ab", 1).unwrap();
        assert_eq!(
            take(&r, "mcp:o:t/a"),
            [(Sender::Starter, "for a".to_string())]
        );
        assert_eq!(
            take(&r, "mcp:o:t/ab"),
            [(Sender::Zend, "for ab".to_string())]
        );
    }
}
