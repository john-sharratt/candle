//! Prefill rendering for the `code_reading` layer's per-file
//! tool-call conversation.
//!
//! Each file becomes ONE conversation. Every carved part of the file
//! contributes a prefilled tool-call exchange that teaches the model
//! the canonical "use a tool, read the response" pattern, and the
//! conversation ends with a single DECODED whole-file summary:
//!
//! ````text
//! Part turn (user, prefilled):
//!   Source excerpt — `src/auth/handler.rs` lines 47-93:
//!
//! Part turn (assistant, prefilled — tool call):
//!   <tool_call>{"name":"read_file","arguments":{"path":"src/auth/handler.rs",
//!               "start_line":47,"end_line":93}}</tool_call>
//!
//! Part turn (user, prefilled — tool response):
//!   <tool_response>
//!   src/auth/handler.rs (lines 47-93):
//!
//!   ```rust
//!       47  impl AuthHandler {
//!       48      pub fn validate_token(&self, token: &str) -> Result<Claims> {
//!       ...
//!       93  }
//!   ```
//!   </tool_response>
//!
//! ... (one such exchange per part, in file order) ...
//!
//! Final turn (assistant, DECODED):
//!   Summarize the entire file `src/auth/handler.rs` in no more than
//!   200 words. → <whole-file summary, generated live by the model>
//! ````
//!
//! The per-part prompt is rendered by [`render_part_user_prompt`] and
//! the parts are inserted via [`crate::turn_sink::InsertTurnSink::
//! insert_prefill_turn`].  The final summary prompt comes from
//! [`render_file_summary_prompt`]; the turn is submitted via
//! [`crate::turn_sink::InsertTurnSink::decode_summary_turn`] and the
//! model's output becomes part of the code_reading trunk.
//!
//! The `<tool_call>` / `</tool_call>` tags in the part turns mirror the
//! Hermes format the dialogue layer's tool-call extractor scans for
//! at decode time — they're fine in prefilled context because the
//! extractor only runs on the dialogue model's OWN decode output,
//! never on retrieved context from another layer.

use super::types::Scope;
use crate::repo_scan::Language;

/// Maximum decoded tokens for the final whole-file summary. The prompt
/// caps the model at 200 words; ~1.4 tokens/word leaves headroom.
pub const FILE_SUMMARY_MAX_TOKENS: usize = 300;

/// Per-part user prompt for the one-conversation-per-file layout: a
/// labelled reference header naming the file and line range (no per-part
/// summary — the whole file is summarised in the final turn instead).
pub fn render_part_user_prompt(path: &str, scope: &Scope) -> String {
    // Reference header, not a first-person request: this prefill turn is
    // context-stuffed source the model attends to as background, so framing it
    // as "Read X" would make it read as a user question in conversation-history
    // recall. A labelled excerpt header reads as injected reference instead.
    format!(
        "Source excerpt — `{path}` lines {start}-{end}:",
        path = path,
        start = scope.start_line,
        end = scope.end_line,
    )
}

/// Final-turn user prompt: summarise the whole file (which the model has
/// now read part-by-part across the prior prefilled turns) in ≤200 words.
pub fn render_file_summary_prompt(path: &str) -> String {
    format!(
        "Summarize the entire file `{path}` in no more than 200 words, \
         covering its purpose, key types/functions, and how they fit together.",
        path = path,
    )
}

/// Assistant-side `<tool_call>` echo — the first half of a part
/// turn's assistant message (the `<tool_response>` is appended after
/// it).  Prefilled, so the model doesn't decode this; it learns the
/// pattern by seeing it in context.
pub fn render_tool_call(path: &str, scope: &Scope) -> String {
    format!(
        "<tool_call>{{\"name\":\"read_file\",\"arguments\":{{\"path\":\"{path}\",\
         \"start_line\":{start},\"end_line\":{end}}}}}</tool_call>",
        path = path,
        start = scope.start_line,
        end = scope.end_line,
    )
}

/// Assistant-side `<tool_response>` carrying the actual file content
/// in a language-tagged markdown fence with `cat -n` style line
/// numbers — appended after [`render_tool_call`] to form the part
/// turn's assistant message.  `body` is the verbatim source slice for
/// `scope.start_line..=scope.end_line`.
pub fn render_tool_response(path: &str, scope: &Scope, language: Language, body: &str) -> String {
    let max_line = scope.end_line;
    let width = digit_width(max_line);
    let mut numbered = String::with_capacity(body.len() + (max_line as usize) * (width + 4));
    let mut lines = body.split('\n').peekable();
    let mut idx: u32 = 0;
    while let Some(line) = lines.next() {
        // A trailing newline on `body` produces a final empty
        // element in the split — skip it so we don't emit a phantom
        // numbered line past `end_line`.
        if line.is_empty() && lines.peek().is_none() && body.ends_with('\n') {
            break;
        }
        let line_no = scope.start_line + idx;
        numbered.push_str(&format!("{line_no:width$}  {line}\n", width = width));
        idx += 1;
    }

    let tag = language.fence_tag();
    let fence_open = if tag.is_empty() {
        String::from("```\n")
    } else {
        format!("```{tag}\n")
    };
    format!(
        "<tool_response>\n{path} (lines {start}-{end}):\n\n{fence_open}{numbered}```\n</tool_response>",
        path = path,
        start = scope.start_line,
        end = scope.end_line,
        fence_open = fence_open,
        numbered = numbered,
    )
}

fn digit_width(mut n: u32) -> usize {
    if n == 0 {
        return 1;
    }
    let mut w = 0;
    while n > 0 {
        n /= 10;
        w += 1;
    }
    w
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_read::types::ChunkKind;

    fn scope(start: u32, end: u32) -> Scope {
        Scope {
            path: vec!["dummy".into()],
            kind: ChunkKind::Function,
            start_line: start,
            end_line: end,
        }
    }

    // ── per-file layout: render_part_user_prompt / render_file_summary_prompt ──

    #[test]
    fn part_user_prompt_reads_range_without_summary_instruction() {
        let p = render_part_user_prompt("src/lib.rs", &scope(10, 20));
        assert_eq!(p, "Source excerpt — `src/lib.rs` lines 10-20:");
        // The per-part turn is prefill-only; the summary happens once at
        // the end of the file conversation, not per part.
        assert!(!p.to_lowercase().contains("summarize"));
    }

    #[test]
    fn part_user_prompt_quotes_path_with_backticks() {
        let p = render_part_user_prompt("packages/my-pkg/src/lib.rs", &scope(1, 5));
        assert!(p.contains("`packages/my-pkg/src/lib.rs`"));
    }

    #[test]
    fn part_user_prompt_is_byte_identical_on_repeat() {
        let s = scope(1, 10);
        assert_eq!(
            render_part_user_prompt("src/x.rs", &s),
            render_part_user_prompt("src/x.rs", &s)
        );
    }

    #[test]
    fn file_summary_prompt_names_file_and_caps_at_200_words() {
        let p = render_file_summary_prompt("src/auth/handler.rs");
        assert!(p.contains("`src/auth/handler.rs`"));
        assert!(p.contains("200 words"));
    }

    // ── render_tool_call ─────────────────────────────────────────────────────

    #[test]
    fn tool_call_is_hermes_style_json() {
        let tc = render_tool_call("src/lib.rs", &scope(142, 187));
        assert!(tc.starts_with("<tool_call>"));
        assert!(tc.ends_with("</tool_call>"));
        assert!(tc.contains("\"name\":\"read_file\""));
        assert!(tc.contains("\"path\":\"src/lib.rs\""));
        assert!(tc.contains("\"start_line\":142"));
        assert!(tc.contains("\"end_line\":187"));
    }

    #[test]
    fn tool_call_is_single_line_for_clean_parsing() {
        let tc = render_tool_call("src/lib.rs", &scope(1, 5));
        assert!(!tc.contains('\n'), "tool_call should not contain newlines");
    }

    // ── render_tool_response ─────────────────────────────────────────────────

    #[test]
    fn tool_response_wraps_body_in_tool_response_tags() {
        let r = render_tool_response("src/x.rs", &scope(1, 1), Language::Rust, "fn alpha() {}\n");
        assert!(r.starts_with("<tool_response>\n"));
        assert!(r.ends_with("</tool_response>"));
    }

    #[test]
    fn tool_response_includes_path_and_range_header() {
        let r = render_tool_response("src/x.rs", &scope(47, 93), Language::Rust, "x\n");
        assert!(r.contains("src/x.rs (lines 47-93):"));
    }

    #[test]
    fn tool_response_prefixes_each_line_with_line_number() {
        let body = "fn alpha() {\n    return 1;\n}\n";
        let r = render_tool_response("src/x.rs", &scope(10, 12), Language::Rust, body);
        assert!(r.contains("10  fn alpha() {"));
        assert!(r.contains("11      return 1;"));
        assert!(r.contains("12  }"));
    }

    #[test]
    fn tool_response_uses_language_fence_tag() {
        for (lang, tag) in [
            (Language::Rust, "rust"),
            (Language::Python, "python"),
            (Language::TypeScript, "typescript"),
            (Language::Go, "go"),
            (Language::C, "c"),
            (Language::Cpp, "cpp"),
            (Language::Java, "java"),
            (Language::Ruby, "ruby"),
            (Language::Php, "php"),
            (Language::Bash, "bash"),
            (Language::Html, "html"),
            (Language::Css, "css"),
        ] {
            let r = render_tool_response("f.x", &scope(1, 1), lang, "// hi\n");
            assert!(
                r.contains(&format!("```{tag}\n")),
                "expected ```{tag} fence in {r}",
            );
        }
    }

    #[test]
    fn tool_response_pads_line_numbers_to_widest() {
        let body = "x\ny\nz\n";
        let r = render_tool_response("src/x.rs", &scope(9998, 10000), Language::Rust, body);
        assert!(r.contains(" 9998  x"));
        assert!(r.contains(" 9999  y"));
        assert!(r.contains("10000  z"));
    }

    #[test]
    fn tool_response_handles_no_trailing_newline() {
        let body = "fn a() {}";
        let r = render_tool_response("src/x.rs", &scope(1, 1), Language::Rust, body);
        let numbered_lines = r
            .lines()
            .filter(|l| l.trim_start().starts_with("1  "))
            .count();
        assert_eq!(numbered_lines, 1);
        assert!(!r.contains("2  "));
    }

    #[test]
    fn tool_response_preserves_tabs_in_indentation() {
        let body = "fn a() {\n\tlet x = 1;\n}\n";
        let r = render_tool_response("src/x.rs", &scope(1, 3), Language::Rust, body);
        assert!(r.contains("2  \tlet x = 1;"));
    }

    #[test]
    fn tool_response_preserves_utf8_content() {
        let body = "fn greet() { println!(\"héllo — 世界\"); }\n";
        let r = render_tool_response("src/x.rs", &scope(1, 1), Language::Rust, body);
        assert!(r.contains("héllo — 世界"));
    }

    #[test]
    fn tool_response_plain_text_uses_untagged_fence() {
        let r = render_tool_response("notes.txt", &scope(1, 1), Language::PlainText, "hello\n");
        assert!(r.contains("```\n"));
        assert!(!r.contains("```text"));
    }

    // ── digit_width ──────────────────────────────────────────────────────────

    #[test]
    fn digit_width_matches_decimal_digits() {
        assert_eq!(digit_width(0), 1);
        assert_eq!(digit_width(1), 1);
        assert_eq!(digit_width(9), 1);
        assert_eq!(digit_width(10), 2);
        assert_eq!(digit_width(99), 2);
        assert_eq!(digit_width(100), 3);
        assert_eq!(digit_width(9_999_999), 7);
    }
}
