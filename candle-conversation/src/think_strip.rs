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
    let collapsed = collapse_whitespace(&strip_to_raw(text));
    collapsed.trim().to_string()
}

/// Strip `<think>…</think>` blocks while preserving the surviving text's line
/// structure and indentation.
///
/// Unlike [`strip_think_blocks`], this does **not** collapse internal
/// whitespace — newlines, blank lines, and leading indentation are kept so a
/// stored assistant reply renders back with its original markdown paragraphs,
/// lists, and code blocks intact. Only leading/trailing whitespace is trimmed.
/// This is the right choice for the persisted reply the GUI re-displays on
/// reload; the whitespace-collapsing [`strip_think_blocks`] is for the compact
/// one-line summaries the summariser stores.
pub fn strip_think_blocks_keep_layout(text: &str) -> String {
    strip_to_raw(text).trim().to_string()
}

/// Remove only the **empty** `<think></think>` blocks — those whose inner
/// content is whitespace-only — while preserving every non-empty reasoning block
/// and all surrounding text verbatim.
///
/// A `/no_think` decode collapses its reasoning to a bare `<think></think>`; that
/// empty block is pure noise in a stored/displayed turn, but a turn carrying
/// *real* reasoning should keep it. Unlike [`strip_think_blocks`], this leaves a
/// non-empty block untouched. The whitespace hugging a removed empty block is
/// collapsed (so a leading `<think></think>` doesn't leave a blank gap); a single
/// space is inserted only between two surviving non-empty runs so words never
/// merge. Tags are matched case-sensitively — the models emit lowercase `<think>`,
/// matching the turn-layout splitter's own convention.
pub fn strip_empty_think_blocks(text: &str) -> String {
    if !text.contains("<think>") {
        return strip_trailing_orphan_close(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        let Some(open) = rest.find("<think>") else {
            out.push_str(rest);
            break;
        };
        let after_open = open + "<think>".len();
        let Some(rel_close) = rest[after_open..].find("</think>") else {
            // Unterminated — not a block we can classify; leave the tail verbatim.
            out.push_str(rest);
            break;
        };
        let after_close = after_open + rel_close + "</think>".len();
        let inner = &rest[after_open..after_open + rel_close];
        if inner.trim().is_empty() {
            let before = rest[..open].trim_end();
            out.push_str(before);
            let tail = rest[after_close..].trim_start();
            if !out.is_empty() && !tail.is_empty() {
                out.push(' ');
            }
            rest = tail;
        } else {
            out.push_str(&rest[..after_close]);
            rest = &rest[after_close..];
        }
    }
    out
}

/// Drop a closing think tag left dangling at the END of a turn that never opened
/// one.
///
/// **Why a suppressed turn produces these.** With thinking off the assistant grid
/// opens on an already-closed block, so the model is outside one — and
/// `think_close_ban_active` bans the real `</think>` id whenever the sampler is
/// outside a block. Wanting to close anyway, the model reaches for a spelling
/// that is not the banned id: measured on the repo_map ingest, 2 of 22 summaries
/// ended in a bare `</thinking>`. Hence the tolerant tag match — `</think>` plus
/// any alphabetic tail.
///
/// **Deliberately narrow on both axes.** Only the trailing tag, and only when the
/// text opens no block at all (the caller checks). Sealed text is paired with a
/// K/V span by offset, so anything removed here shifts that pairing — mid-text
/// edits would corrupt the mapping `tool_exchange_segments` carves sub-segments
/// from, to chase a case never observed. A turn carrying real reasoning has an
/// opener and never reaches this.
fn strip_trailing_orphan_close(text: &str) -> String {
    let trimmed = text.trim_end();
    let lower = trimmed.to_ascii_lowercase();
    let Some(start) = lower.rfind("</think") else {
        return text.to_string();
    };
    match close_tag_len(&lower[start..]) {
        // The tag must BE the tail — a `</think>` with prose after it is being
        // used as text, not as a stray closer.
        Some(len) if start + len == trimmed.len() => trimmed[..start].trim_end().to_string(),
        _ => text.to_string(),
    }
}

/// Byte length of a closing think tag at the start of `s`, if there is one.
///
/// Accepts `</think>` and the alphabetic variants a model invents when the
/// canonical token is unavailable (`</thinking>`). Lowercase input only — the
/// callers pass an already-lowered copy whose byte layout matches the original,
/// since `to_ascii_lowercase` never changes a byte's width.
fn close_tag_len(s: &str) -> Option<usize> {
    let rest = s.strip_prefix("</think")?;
    let extra = rest.chars().take_while(|c| c.is_ascii_alphabetic()).count();
    rest[extra..]
        .starts_with('>')
        .then_some("</think".len() + extra + 1)
}

/// Remove every `<think>…</think>` block (and stray unmatched `</think>` tags),
/// inserting a single space at each removal seam. Returns the raw result with
/// no whitespace normalization — the caller decides whether to collapse.
fn strip_to_raw(text: &str) -> String {
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
                // run together.
                out.push(' ');
            }
            None => {
                // Unterminated block: strip to end.
                break;
            }
        }
    }

    out
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
    use super::{strip_empty_think_blocks, strip_think_blocks, strip_think_blocks_keep_layout};

    // ── Empty-only variant ─────────────────────────────────────────────────

    #[test]
    fn empty_only_no_think_unchanged() {
        let s = "Just a summary sentence.";
        assert_eq!(strip_empty_think_blocks(s), s);
    }

    #[test]
    fn empty_only_drops_leading_bare_block() {
        assert_eq!(
            strip_empty_think_blocks("<think></think>The summary."),
            "The summary."
        );
    }

    #[test]
    fn empty_only_drops_leading_block_and_collapses_whitespace() {
        assert_eq!(
            strip_empty_think_blocks("<think>\n\n</think>\n\nThe summary."),
            "The summary."
        );
    }

    #[test]
    fn empty_only_keeps_real_reasoning_verbatim() {
        let s = "<think>real reasoning here</think>The answer.";
        assert_eq!(strip_empty_think_blocks(s), s);
    }

    #[test]
    fn empty_only_removes_empty_keeps_nonempty() {
        assert_eq!(
            strip_empty_think_blocks("<think></think>A<think>keep me</think>B"),
            "A<think>keep me</think>B"
        );
    }

    /// **The suppressed turn's leftover.** With thinking off the model is
    /// already outside a block and the real `</think>` id is banned, so it
    /// closes with a spelling of its own. Measured: 2 of 22 ingest summaries
    /// ended in a bare `</thinking>`. The summary text must survive intact.
    #[test]
    fn an_orphan_close_tag_is_dropped_without_touching_the_text() {
        assert_eq!(
            strip_empty_think_blocks("The folder defines Qwen presets.</thinking>"),
            "The folder defines Qwen presets."
        );
        assert_eq!(strip_empty_think_blocks("A summary.</think>"), "A summary.");
        // Trailing whitespace after the tag is still the tail.
        assert_eq!(
            strip_empty_think_blocks("A summary.</thinking>\n\n"),
            "A summary."
        );
    }

    /// **Only the TRAILING tag, and only in a turn that opened no block.**
    ///
    /// Sealed text is paired with its K/V span by offset, so a mid-text deletion
    /// would shift that pairing for every consumer that maps text onto tokens.
    /// A `</think>` with prose after it is being used as text — quoted source,
    /// an explanation of the markup — and is left alone.
    #[test]
    fn a_mid_text_close_tag_is_left_alone() {
        assert_eq!(
            strip_empty_think_blocks("Two</thinking> sentences."),
            "Two</thinking> sentences."
        );
        // Prose *about* the tag survives intact — this repo's own sources say
        // exactly this sort of thing, and the ingest reads them back.
        let quoted = "The gate holds until </think> arrives, then opens.";
        assert_eq!(strip_empty_think_blocks(quoted), quoted);
    }

    /// A turn that opened a block is never touched by the orphan strip — it has
    /// an opener, so the trailing-tag path is not even reached. Its reasoning is
    /// preserved verbatim, which is what a client rendering a reasoning pane
    /// needs.
    #[test]
    fn a_turn_that_opened_a_block_is_left_intact() {
        assert_eq!(
            strip_empty_think_blocks("<think>real reasoning</think>Answer."),
            "<think>real reasoning</think>Answer."
        );
        // Even a doubled close stays: the turn opened a block, so this is not the
        // suppressed-turn artifact the strip exists for, and guessing here would
        // shift the text/K-V offset pairing on a case never observed.
        assert_eq!(
            strip_empty_think_blocks("<think>r</think>Answer.</think>"),
            "<think>r</think>Answer.</think>"
        );
    }

    /// Text that merely mentions the tag shape without closing it is left alone
    /// — no `>` means no tag, so prose about `</think` is not mangled.
    #[test]
    fn an_unterminated_tag_shape_is_not_a_tag() {
        assert_eq!(
            strip_empty_think_blocks("talking about </think without a close"),
            "talking about </think without a close"
        );
    }

    #[test]
    fn empty_only_separates_runs_around_a_mid_text_empty_block() {
        assert_eq!(strip_empty_think_blocks("A<think></think>B"), "A B");
    }

    #[test]
    fn empty_only_all_empty_becomes_empty() {
        assert_eq!(strip_empty_think_blocks("<think>   \n  </think>"), "");
    }

    // ── Layout-preserving variant ──────────────────────────────────────────

    #[test]
    fn keep_layout_preserves_paragraphs_and_lists() {
        let s = "First paragraph.\n\nSecond paragraph.\n\n- a\n- b";
        assert_eq!(strip_think_blocks_keep_layout(s), s);
    }

    #[test]
    fn keep_layout_preserves_code_block_indentation() {
        let s = "Here:\n\n```rust\nfn main() {\n    println!(\"hi\");\n}\n```";
        assert_eq!(strip_think_blocks_keep_layout(s), s);
    }

    #[test]
    fn keep_layout_strips_leading_think_block_and_trims() {
        let s = "<think>\nreasoning\nmore\n</think>\n\nThe answer:\n\n- one\n- two";
        assert_eq!(
            strip_think_blocks_keep_layout(s),
            "The answer:\n\n- one\n- two"
        );
    }

    #[test]
    fn keep_layout_drops_inline_think_but_keeps_following_lines() {
        let s = "<think>x</think>Line one\nLine two";
        assert_eq!(strip_think_blocks_keep_layout(s), "Line one\nLine two");
    }

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
