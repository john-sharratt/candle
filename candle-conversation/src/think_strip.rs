//! Strips `<think>…</think>` reasoning blocks from model-generated text.
//!
//! Some models (Qwen2.5, DeepSeek-R1, etc.) wrap internal chain-of-thought in
//! `<think>…</think>` tags.  When this text is stored as a conversation summary
//! or returned to the caller, the reasoning trace is noise.  This module
//! removes it while preserving every other character.
//!
//! # Rules
//!
//! - Tags are matched **case-insensitively** (`<THINK>`, `<Think>`, etc.).
//! - Whitespace between a closing `</think>` and the next non-whitespace
//!   character is collapsed to a single space, preventing double-spaces and
//!   leading/trailing whitespace in the returned string.
//! - An **unterminated** block (`<think>` with no matching `</think>`) is
//!   treated as if the tag runs to the end of the string — the entire tail is
//!   stripped.
//! - **Nested** tags are not supported; the first `</think>` closes the most
//!   recently opened `<think>`.
//! - The function is `O(n)` in the length of the input.

/// Remove all `<think>…</think>` blocks from `text`.
///
/// Returns the cleaned string with leading/trailing whitespace trimmed.
/// If the entire input is inside think blocks, returns an empty string.
///
/// **Stray closing tags**: a lone `</think>` with no matching opening tag is
/// treated as noise and removed (just the tag, not the content around it).
/// This handles models that accidentally append `</think>` to their response.
pub fn strip_think_blocks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    loop {
        let Some(open_start) = find_tag(rest, "<think>") else {
            // No more open tags — keep what remains, stripping any stray
            // </think> close tags that have no matching open.
            out.push_str(&remove_stray_close_tags(rest));
            break;
        };

        // Keep text before the opening tag.
        out.push_str(&remove_stray_close_tags(&rest[..open_start]));

        let after_open = open_start + "<think>".len();

        match find_tag(&rest[after_open..], "</think>") {
            Some(close_start) => {
                let after_close = after_open + close_start + "</think>".len();
                rest = &rest[after_close..];
                // Insert a space at the join point so surrounding words don't
                // run together; collapse_whitespace() squashes any runs.
                out.push(' ');
            }
            None => {
                // Unterminated block: strip to end.
                break;
            }
        }
    }

    let collapsed = collapse_whitespace(&out);
    collapsed.trim().to_string()
}

/// Remove all stray `</think>` close tags (those with no matching open tag)
/// from `s`, inserting a space at each removal site.
fn remove_stray_close_tags(s: &str) -> String {
    if !s.to_ascii_lowercase().contains("</think>") {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut remaining = s;
    loop {
        match find_tag(remaining, "</think>") {
            None => {
                out.push_str(remaining);
                break;
            }
            Some(pos) => {
                out.push_str(&remaining[..pos]);
                out.push(' ');
                remaining = &remaining[pos + "</think>".len()..];
            }
        }
    }
    out
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Case-insensitive substring search.  Returns the byte offset of the first
/// occurrence of `needle` (lowered) inside `haystack` (lowered).
fn find_tag(haystack: &str, needle: &str) -> Option<usize> {
    let lower = haystack.to_ascii_lowercase();
    let needle_lower = needle.to_ascii_lowercase();
    lower.find(&needle_lower)
}

/// Collapse runs of ASCII whitespace (spaces, tabs, newlines) that are
/// separated only by whitespace into a single space.
/// Runs of whitespace *within* kept text that were already single spaces are
/// left untouched; only runs of two or more are squashed.
fn collapse_whitespace(s: &str) -> String {
    // Fast path: no double-whitespace at all.
    if !s.contains("  ") && !s.contains('\n') && !s.contains('\t') && !s.contains("\r") {
        return s.to_string();
    }

    let mut out = String::with_capacity(s.len());
    let mut last_was_ws = false;
    for ch in s.chars() {
        if ch.is_ascii_whitespace() {
            if !last_was_ws {
                out.push(' ');
            }
            last_was_ws = true;
        } else {
            out.push(ch);
            last_was_ws = false;
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::strip_think_blocks;

    // ── Basic ──────────────────────────────────────────────────────────────

    #[test]
    fn empty_string() {
        assert_eq!(strip_think_blocks(""), "");
    }

    #[test]
    fn no_think_tags() {
        let s = "Hello, world! This is a normal response.";
        assert_eq!(strip_think_blocks(s), s);
    }

    #[test]
    fn think_block_only() {
        assert_eq!(strip_think_blocks("<think>internal reasoning</think>"), "");
    }

    #[test]
    fn think_at_start() {
        let s = "<think>reasoning goes here</think>This is the answer.";
        assert_eq!(strip_think_blocks(s), "This is the answer.");
    }

    #[test]
    fn think_at_end() {
        let s = "The answer is 42.<think>double-checking…</think>";
        assert_eq!(strip_think_blocks(s), "The answer is 42.");
    }

    #[test]
    fn think_in_middle() {
        let s = "Before.<think>hidden</think>After.";
        assert_eq!(strip_think_blocks(s), "Before. After.");
    }

    // ── Multiple blocks ────────────────────────────────────────────────────

    #[test]
    fn two_think_blocks() {
        let s = "<think>A</think>Middle<think>B</think>End.";
        assert_eq!(strip_think_blocks(s), "Middle End.");
    }

    #[test]
    fn three_consecutive_think_blocks() {
        let s = "<think>1</think><think>2</think><think>3</think>Result.";
        assert_eq!(strip_think_blocks(s), "Result.");
    }

    #[test]
    fn think_blocks_with_text_between() {
        let s = "A<think>x</think>B<think>y</think>C";
        assert_eq!(strip_think_blocks(s), "A B C");
    }

    // ── Case insensitivity ─────────────────────────────────────────────────

    #[test]
    fn uppercase_tags() {
        let s = "<THINK>hidden</THINK>visible";
        assert_eq!(strip_think_blocks(s), "visible");
    }

    #[test]
    fn mixed_case_open_tag() {
        let s = "<Think>hidden</Think>visible";
        assert_eq!(strip_think_blocks(s), "visible");
    }

    #[test]
    fn mixed_case_close_tag_only() {
        // open is lowercase, close is upper — should still match
        let s = "<think>hidden</THINK>visible";
        assert_eq!(strip_think_blocks(s), "visible");
    }

    // ── Unterminated blocks ────────────────────────────────────────────────

    #[test]
    fn unterminated_at_start() {
        let s = "<think>reasoning that never ends";
        assert_eq!(strip_think_blocks(s), "");
    }

    #[test]
    fn unterminated_after_good_text() {
        let s = "Summary text.<think>more reasoning that never ends";
        assert_eq!(strip_think_blocks(s), "Summary text.");
    }

    #[test]
    fn unterminated_after_completed_block() {
        let s = "<think>A</think>good<think>unterminated";
        assert_eq!(strip_think_blocks(s), "good");
    }

    // ── Whitespace handling ────────────────────────────────────────────────

    #[test]
    fn leading_trailing_whitespace_trimmed() {
        let s = "   <think>x</think>   answer   ";
        assert_eq!(strip_think_blocks(s), "answer");
    }

    #[test]
    fn newlines_around_block() {
        let s = "Before.\n<think>\nmulti\nline\nreasoning\n</think>\nAfter.";
        assert_eq!(strip_think_blocks(s), "Before. After.");
    }

    #[test]
    fn only_whitespace_remains_after_strip() {
        let s = "   <think>everything</think>   ";
        assert_eq!(strip_think_blocks(s), "");
    }

    #[test]
    fn whitespace_between_two_blocks_collapsed() {
        let s = "<think>A</think>   <think>B</think>result";
        assert_eq!(strip_think_blocks(s), "result");
    }

    #[test]
    fn tabs_and_spaces_collapsed() {
        let s = "A<think>x</think>\t\t\tB";
        assert_eq!(strip_think_blocks(s), "A B");
    }

    // ── Content variety ────────────────────────────────────────────────────

    #[test]
    fn empty_think_block() {
        let s = "Before.<think></think>After.";
        assert_eq!(strip_think_blocks(s), "Before. After.");
    }

    #[test]
    fn think_block_with_only_whitespace() {
        let s = "Before.<think>   \n  </think>After.";
        assert_eq!(strip_think_blocks(s), "Before. After.");
    }

    #[test]
    fn unicode_outside_block_preserved() {
        let s = "<think>ignore</think>日本語テスト 🎉";
        assert_eq!(strip_think_blocks(s), "日本語テスト 🎉");
    }

    #[test]
    fn unicode_inside_block_removed() {
        let s = "<think>思考: これは隠す</think>This is visible.";
        assert_eq!(strip_think_blocks(s), "This is visible.");
    }

    #[test]
    fn html_like_content_inside_block() {
        let s = "<think><p>Some <b>HTML</b></p></think>Clean output.";
        assert_eq!(strip_think_blocks(s), "Clean output.");
    }

    #[test]
    fn angle_brackets_outside_block_preserved() {
        let s = "Use 3 < 4 and 5 > 2 as examples.";
        assert_eq!(strip_think_blocks(s), s);
    }

    #[test]
    fn look_alike_tags_not_affected() {
        // <thinker> is not a think block
        let s = "<thinker>someone</thinker> said something.";
        assert_eq!(strip_think_blocks(s), s);
    }

    #[test]
    fn stray_close_tag_removes_just_the_tag() {
        // A lone </think> has no matching open — just the tag is removed,
        // content before it is kept (it's part of the actual response).
        let s = "Before</think>After.";
        assert_eq!(strip_think_blocks(s), "Before After.");
    }

    #[test]
    fn stray_close_tag_at_end_removed() {
        // Qwen2-0.5B pattern: model appends a lone </think> at the end.
        let s = "<p>Hello, how can I help?</p>\n</think>";
        assert_eq!(strip_think_blocks(s), "<p>Hello, how can I help?</p>");
    }

    #[test]
    fn stray_close_tag_only() {
        let s = "</think>";
        assert_eq!(strip_think_blocks(s), "");
    }

    #[test]
    fn multiple_stray_close_tags() {
        let s = "A</think>B</think>C";
        assert_eq!(strip_think_blocks(s), "A B C");
    }

    // ── Real-world model output patterns ─────────────────────────────────

    #[test]
    fn qwen_style_output() {
        let s = "\n<think>\nLet me reason step by step.\n1. The user asked X.\n2. The answer is Y.\n</think>\n\nThe answer is Y.";
        assert_eq!(strip_think_blocks(s), "The answer is Y.");
    }

    #[test]
    fn deepseek_style_output() {
        let s = "<think>\nOkay, I need to summarize this conversation.\nThe key points are:\n- Point A\n- Point B\n</think>\n\nHere is a summary: Point A and Point B were discussed.";
        assert_eq!(
            strip_think_blocks(s),
            "Here is a summary: Point A and Point B were discussed."
        );
    }

    #[test]
    fn response_with_no_content_after_think() {
        // Model only produced a think block and nothing else — considered empty.
        let s = "<think>\nI wonder what to say...\n</think>\n";
        assert_eq!(strip_think_blocks(s), "");
    }

    #[test]
    fn large_think_block_small_output() {
        let reasoning = "step ".repeat(500);
        let s = format!("<think>{}</think>Final answer.", reasoning);
        assert_eq!(strip_think_blocks(&s), "Final answer.");
    }
}
