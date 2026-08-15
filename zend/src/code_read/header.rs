//! Prefill rendering for the `code_reading` layer's per-file
//! tool-call conversation.
//!
//! Each file becomes ONE conversation. Every carved part of the file
//! contributes a prefilled tool exchange that ends on a real, DECODED assistant
//! turn — a two-sentence summary the model generates over the excerpt it just
//! "read":
//!
//! ````text
//! Segment 1 (user — the request):
//!   Summarize `src/auth/handler.rs` (lines 47-93) in no more than two sentences.
//!
//! Segment 2 (assistant — tool call):
//!   <tool_call>{"name":"file_read","arguments":{"path":"src/auth/handler.rs",
//!               "start_line":47,"end_line":93}}</tool_call>
//!
//! Segment 3 (user — tool response):
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
//! Segment 4 (assistant — DECODED two-sentence summary):
//!   AuthHandler::validate_token verifies a bearer token and returns its Claims,
//!   erroring on an expired or malformed token. …
//!
//! ... (one such exchange per part, in file order) ...
//! ````
//!
//! Segments 1 and 4 are the turn's user opener and assistant closer; segments
//! 2 and 3 carry the call and its result, joined into the assistant string with
//! the dialect role boundaries the substrate frames around. So a scope
//! reconstructs as a complete `user → assistant → user → assistant`
//! alternation, never ending on a user turn. Segment 1 (a genuine summarise
//! *request*) is rendered by [`render_part_user_prompt`], the call/response by
//! [`render_tool_call`] / [`render_tool_response`]; the prefilled string stops
//! at the `<tool_response>`'s user-end, and the scheduler then DECODES segment 4
//! — the two-sentence summary — as the closing assistant turn. That decoded
//! summary doubles as the scope's provenance anchor (a semantic key retrieves
//! better than raw source), so there is no separate async whole-file summary for
//! these turns — the summariser is disabled on the code_reading timeline.
//!
//! The `<tool_call>` / `</tool_call>` tags in the part turns mirror the
//! Hermes format the dialogue layer's tool-call extractor scans for
//! at decode time — they're fine in prefilled context because the
//! extractor only runs on the dialogue model's OWN decode output,
//! never on retrieved context from another layer.

use super::types::Scope;
use crate::repo_scan::Language;

/// Per-part user prompt for the one-conversation-per-file layout: a genuine
/// summarise request naming the file (full path) and line range. The scheduler
/// answers it by decoding a two-sentence summary as the closing assistant turn,
/// so this turn reconstructs as a real request→answer exchange (not a
/// context-stuffed reference blob) and the decoded summary anchors the scope for
/// provenance retrieval.
pub fn render_part_user_prompt(path: &str, scope: &Scope) -> String {
    format!(
        "Summarize `{path}` (lines {start}-{end}) in no more than two sentences.",
        path = path,
        start = scope.start_line,
        end = scope.end_line,
    )
}

/// Assistant-side `<tool_call>` echo — the assistant segment of a part
/// turn. The caller splices a role boundary
/// ([`InsertTurnSink::tool_exchange_boundaries`](crate::turn_sink::InsertTurnSink::tool_exchange_boundaries))
/// after this, then appends [`render_tool_response`] as a distinct user
/// segment. Prefilled, so the model doesn't decode this; it learns the
/// pattern by seeing it in context.
///
/// The call names the canonical `file_read` tool (not its `read_file` alias) so
/// it matches the tool definition the summary projection force-pins into the
/// catalog (`FORCE_TOOL_SELECTOR` → `file_read`) — the prefilled call and the
/// one presented tool agree on name, keeping the tool context coherent.
pub fn render_tool_call(path: &str, scope: &Scope) -> String {
    format!(
        "<tool_call>{{\"name\":\"file_read\",\"arguments\":{{\"path\":\"{path}\",\
         \"start_line\":{start},\"end_line\":{end}}}}}</tool_call>",
        path = path,
        start = scope.start_line,
        end = scope.end_line,
    )
}

/// User-side `<tool_response>` carrying the actual file content in a
/// language-tagged markdown fence with `cat -n` style line numbers. It forms
/// the part turn's second **user** segment — the caller emits it after
/// [`render_tool_call`] and a role boundary, mirroring how a real tool result
/// returns in a user turn.  `body` is the verbatim source slice for
/// `scope.start_line..=scope.end_line`.
pub fn render_tool_response(path: &str, scope: &Scope, language: Language, body: &str) -> String {
    // One renderer, shared with the live `file_read` tool
    // (`zend_tools::tools::file::render`), so an ingested response and a runtime
    // one are the same bytes. `total_lines` is the scope's own end: an ingest
    // excerpt is a complete scope, so the header takes the plain `(lines a-b)`
    // form rather than the truncated `(lines a-b of N)` one.
    let excerpt = zend_tools::tools::file::render::numbered_excerpt(
        path,
        scope.start_line,
        scope.end_line,
        scope.end_line,
        language.fence_tag(),
        body,
    );
    format!("<tool_response>{excerpt}</tool_response>")
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

    // ── per-file layout: render_part_user_prompt ─────────────────────────────

    #[test]
    fn part_user_prompt_is_a_two_sentence_summary_request() {
        let p = render_part_user_prompt("src/lib.rs", &scope(10, 20));
        assert_eq!(
            p,
            "Summarize `src/lib.rs` (lines 10-20) in no more than two sentences."
        );
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

    // ── render_tool_call ─────────────────────────────────────────────────────

    #[test]
    fn tool_call_is_hermes_style_json() {
        let tc = render_tool_call("src/lib.rs", &scope(142, 187));
        assert!(tc.starts_with("<tool_call>"));
        assert!(tc.ends_with("</tool_call>"));
        assert!(tc.contains("\"name\":\"file_read\""));
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

    /// The live `file_read` tool derives its fence tag from the path while the
    /// ingest derives it from a parsed [`Language`]; the two tables must agree or
    /// a runtime read renders under a different tag than the corpus that taught
    /// the model to read it. This is the only crate that can see both.
    #[test]
    fn fence_tags_agree_between_the_ingest_and_the_live_tool() {
        for ext in [
            "rs", "py", "pyi", "ts", "tsx", "js", "jsx", "mjs", "cjs", "go", "c", "h", "cc", "cpp",
            "cxx", "hpp", "hxx", "hh", "java", "rb", "rake", "ru", "gemspec", "php", "phtml", "sh",
            "bash", "zsh", "html", "htm", "css", "scss", "sass", "less", "md", "markdown", "mdx",
            "yaml", "yml", "toml", "json", "json5", "jsonc", "txt", "rst", "adoc", "asciidoc",
        ] {
            let from_ingest = Language::from_extension(ext)
                .expect("allowlisted extension")
                .fence_tag();
            let from_tool =
                zend_tools::tools::file::render::fence_tag_for_path(&format!("a/b.{ext}"));
            assert_eq!(from_ingest, from_tool, "extension {ext:?} disagrees");
        }
    }
}
