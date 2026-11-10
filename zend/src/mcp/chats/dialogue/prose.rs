//! A conversation's turn as a reader of it through a chat tool wants it
//! (`docs/zen_chat.md` §6): the reply without its reasoning, each call it made
//! as one line, and a round's tool results as a count — the same text the web
//! UI draws as collapsed blocks, said in a line each.

use serde_json::Value;

/// The most of one message a read gives.
pub const MAX_MESSAGE: usize = 6000;

/// `text` from an assistant's turn: every `<think>` block dropped, an
/// unclosed one to the end, and every `<tool_call>` said as `[called <tool>]`.
pub fn assistant(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    loop {
        let think = rest.find("<think>");
        let call = rest.find("<tool_call>");
        let next = match (think, call) {
            (Some(t), Some(c)) => Some(t.min(c)),
            (t, c) => t.or(c),
        };
        let Some(at) = next else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..at]);
        let block = &rest[at..];
        if block.starts_with("<think>") {
            match block.find("</think>") {
                Some(end) => rest = &block[end + "</think>".len()..],
                None => break,
            }
        } else {
            let body_at = "<tool_call>".len();
            let (body, after) = match block.find("</tool_call>") {
                Some(end) => (&block[body_at..end], &block[end + "</tool_call>".len()..]),
                None => (&block[body_at..], ""),
            };
            let tool = serde_json::from_str::<Value>(body.trim())
                .ok()
                .and_then(|v| v.get("name").and_then(Value::as_str).map(str::to_string))
                .unwrap_or_else(|| "a tool".into());
            out.push_str(&format!("\n[called {tool}]\n"));
            rest = after;
        }
    }
    tidy(&out)
}

/// `text` from a user's turn: as written, or — a round's tool results — how
/// many there were.
pub fn user(text: &str) -> String {
    let results = text.matches("<tool_response>").count();
    if results > 0 && text.trim_start().starts_with("<tool_response>") {
        return format!(
            "[{results} tool result{}]",
            if results == 1 { "" } else { "s" }
        );
    }
    tidy(text)
}

/// `text` trimmed, its runs of blank lines folded to one, at most
/// [`MAX_MESSAGE`] characters, marked where it was cut.
fn tidy(text: &str) -> String {
    let mut out = String::new();
    let mut blank = 0;
    for line in text.trim().lines() {
        if line.trim().is_empty() {
            blank += 1;
            if blank > 1 {
                continue;
            }
        } else {
            blank = 0;
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
    let out = out.trim_end();
    if out.chars().count() > MAX_MESSAGE {
        let cut: String = out.chars().take(MAX_MESSAGE).collect();
        format!("{cut}…")
    } else {
        out.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reply_loses_its_reasoning_and_says_its_calls() {
        let text = "<think>\nwhich file?\n</think>\n\nReading it.\n<tool_call>\n{\"name\": \"file_read\", \"arguments\": {\"path\": \"a.rs\"}}\n</tool_call>\n\n\n\nDone.";
        assert_eq!(
            assistant(text),
            "Reading it.\n\n[called file_read]\n\nDone."
        );
    }

    /// **An unclosed block runs to the end**: reasoning still being written is
    /// not shown, and a call still being written is said as one.
    #[test]
    fn an_unclosed_block_runs_to_the_end() {
        assert_eq!(assistant("Before.<think>still going"), "Before.");
        assert_eq!(
            assistant("<tool_call>{\"name\": \"file_ed"),
            "[called a tool]"
        );
    }

    #[test]
    fn a_rounds_results_are_counted() {
        assert_eq!(
            user("<tool_response>[1/2 journal_note]\nnoted</tool_response>\n<tool_response>{}</tool_response>"),
            "[2 tool results]"
        );
        assert_eq!(user("  Fix the paper.  "), "Fix the paper.");
    }

    #[test]
    fn a_long_message_is_cut_and_marked() {
        let long = "x".repeat(MAX_MESSAGE + 10);
        let cut = assistant(&long);
        assert_eq!(cut.chars().count(), MAX_MESSAGE + 1);
        assert!(cut.ends_with('…'));
    }
}
