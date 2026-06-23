//! Server-side ChatML turn splitting.
//!
//! Each substrate "turn" stores one iteration's full ChatML stream
//! (`<|im_start|>user\n…<|im_end|><|im_start|>assistant\n…`). The tokenizer's
//! decoder strips the special tokens, leaving literal `user` / `assistant` /
//! `system` role markers on their own lines. `GET /v1/conversations/{id}`
//! returns history already split into role-attributed bubbles so the client
//! adapter stays thin (docs/zend_ui_redesign.md decision 9).
//!
//! Within a `user` segment the `/no_think` dialect prefix is stripped (it is the
//! no-thinking switch — decision 10 — and is not human content). A
//! `<tool_response>…</tool_response>` segment is kept: the GUI pairs each with the
//! preceding tool call and renders it as that call's output.

use crate::types::Role;

/// Split one stored ChatML turn into role-attributed sub-messages.
///
/// `fallback` is the message's own role, used for any leading text before the
/// first role-marker line (or for the whole content when there are no markers).
pub fn split_turn(fallback: Role, content: &str) -> Vec<(Role, String)> {
    let mut out: Vec<(Role, String)> = Vec::new();
    let mut role = fallback;
    let mut buf: Vec<&str> = Vec::new();

    for line in content.split('\n') {
        let boundary = match line.trim() {
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            "system" => Some(Role::System),
            _ => None,
        };
        match boundary {
            Some(next) => {
                push_segment(role, &buf, &mut out);
                buf.clear();
                role = next;
            }
            None => buf.push(line),
        }
    }
    push_segment(role, &buf, &mut out);
    out
}

fn push_segment(role: Role, lines: &[&str], out: &mut Vec<(Role, String)>) {
    let text = lines.join("\n");
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return;
    }
    let cleaned = if role == Role::User {
        strip_no_think(trimmed)
    } else {
        trimmed.to_string()
    };
    if !cleaned.is_empty() {
        out.push((role, cleaned));
    }
}

/// Strip a leading `/no_think` dialect prefix (with any following whitespace).
fn strip_no_think(s: &str) -> String {
    match s.strip_prefix("/no_think") {
        Some(rest) => rest.trim_start().to_string(),
        None => s.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_markers_uses_fallback_role() {
        assert_eq!(
            split_turn(Role::User, "plain text"),
            vec![(Role::User, "plain text".to_string())]
        );
    }

    #[test]
    fn splits_user_and_assistant_on_marker_lines() {
        let content = "user\nhi there\nassistant\nhello back";
        assert_eq!(
            split_turn(Role::Assistant, content),
            vec![
                (Role::User, "hi there".to_string()),
                (Role::Assistant, "hello back".to_string()),
            ]
        );
    }

    #[test]
    fn strips_no_think_prefix_from_user() {
        assert_eq!(
            split_turn(Role::User, "/no_think trace the redo log"),
            vec![(Role::User, "trace the redo log".to_string())]
        );
        // only stripped for user segments
        assert_eq!(
            split_turn(Role::Assistant, "/no_think kept"),
            vec![(Role::Assistant, "/no_think kept".to_string())]
        );
    }

    #[test]
    fn keeps_tool_response_segment() {
        // Tool results are real turns — the GUI pairs them with the preceding
        // tool call and renders them as that call's output, so they must survive
        // the split (they used to be dropped as "scaffolding").
        assert_eq!(
            split_turn(Role::User, "<tool_response>{\"ok\":true}</tool_response>"),
            vec![(
                Role::User,
                "<tool_response>{\"ok\":true}</tool_response>".to_string()
            )]
        );
    }

    #[test]
    fn empty_segments_are_skipped() {
        let content = "user\n\nassistant\nanswer";
        assert_eq!(
            split_turn(Role::User, content),
            vec![(Role::Assistant, "answer".to_string())]
        );
    }

    #[test]
    fn system_marker_recognized() {
        let content = "system\nyou are zen-code\nuser\nhi";
        assert_eq!(
            split_turn(Role::System, content),
            vec![
                (Role::System, "you are zen-code".to_string()),
                (Role::User, "hi".to_string()),
            ]
        );
    }
}
