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
//!   Summarize `src/auth/handler.rs` (lines 47-93) in one or two complete
//!   sentences, ending with a full stop.
//!
//! Segment 2 (assistant — tool call):
//!   <tool_call>{"name":"file_read","arguments":{"path":"src/auth/handler.rs",
//!               "page":0}}</tool_call>
//!
//! Segment 3 (user — tool response):
//!   <tool_response>
//!   src/auth/handler.rs (page 0 of 1, lines 1-93 of 93):
//!
//!   ```rust
//!        1  ...
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

use candle_conversation::TurnText;

use crate::repo_scan::Language;

/// Per-part user prompt for the one-conversation-per-file layout: a genuine
/// summarise request naming the file (full path) and line range. The scheduler
/// answers it by decoding a two-sentence summary as the closing assistant turn,
/// so this turn reconstructs as a real request→answer exchange (not a
/// context-stuffed reference blob) and the decoded summary anchors the scope for
/// provenance retrieval.
/// The ask matches `repo_scan::render`'s, for the reason recorded there: a
/// ceiling on sentence COUNT gets satisfied by one unterminated clause, so the
/// request names *complete* sentences and a full stop instead. Safe to reword —
/// [`summary_tree::scope`]'s `parse_excerpt_ref` splits on `lines ` and keeps
/// only the leading span digits, so trailing prose never reaches the parse.
/// `start`/`end` name the range the request claims to summarise — the
/// scope's own bounds, unless the caller had to narrow them to what the
/// paired tool response actually shows (see `emit_file_turns`'s page-straddle
/// note). Explicit bounds rather than `&Scope` so the two can differ.
pub fn render_part_user_prompt(path: &str, start: u32, end: u32) -> String {
    format!("Summarize `{path}` (lines {start}-{end}) {SCOPE_ASK}")
}

/// What [`render_part_user_prompt`] asks for, after the scope is named.
const SCOPE_ASK: &str = "in one or two complete sentences, ending with a full stop.";

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
/// `page` is the [`zend_tools::state::vfs::PAGE_LINES`]-line page containing
/// `scope.start_line` — the caller (`emit_file_turns`) computes it, since only
/// it has the whole file's line count to page against.
pub fn render_tool_call(path: &str, page: u32) -> String {
    format!(
        "<tool_call>{{\"name\":\"file_read\",\"arguments\":{{\"path\":\"{path}\",\
         \"page\":{page}}}}}</tool_call>",
        path = path,
        page = page,
    )
}

/// User-side `<tool_response>` carrying the actual file content in a
/// language-tagged markdown fence with `cat -n` style line numbers. It forms
/// the part turn's second **user** segment — the caller emits it after
/// [`render_tool_call`] and a role boundary, mirroring how a real tool result
/// returns in a user turn.
///
/// `body` is the verbatim source for the WHOLE page (`start_line..=end_line`),
/// not just the scope that prompted this exchange — a scope can be a fraction
/// of a page or (rarely) straddle one, but `file_read` only ever returns whole
/// pages, and this must render exactly what a live call would.
#[allow(clippy::too_many_arguments)]
pub fn render_tool_response(
    path: &str,
    page: u32,
    total_pages: u32,
    start_line: u32,
    end_line: u32,
    total_lines: u32,
    language: Language,
    body: &str,
) -> TurnText {
    // One renderer, shared with the live `file_read` tool
    // (`zend_tools::tools::file::render`), so an ingested response and a runtime
    // one are the same bytes.
    let excerpt = zend_tools::tools::file::render::numbered_excerpt(
        path,
        page,
        total_pages,
        start_line,
        end_line,
        total_lines,
        language.fence_tag(),
        body,
    );
    TurnText::markup("<tool_response>")
        .then_literal(excerpt)
        .then_markup("</tool_response>")
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── per-file layout: render_part_user_prompt ─────────────────────────────

    #[test]
    fn part_user_prompt_asks_for_complete_sentences() {
        let p = render_part_user_prompt("src/lib.rs", 10, 20);
        assert_eq!(
            p,
            format!("Summarize `src/lib.rs` (lines 10-20) {SCOPE_ASK}")
        );
        assert!(
            SCOPE_ASK.contains("full stop") && !SCOPE_ASK.contains("no more than"),
            "the ask names completeness, not a sentence ceiling: {SCOPE_ASK}"
        );
    }

    #[test]
    fn part_user_prompt_quotes_path_with_backticks() {
        let p = render_part_user_prompt("packages/my-pkg/src/lib.rs", 1, 5);
        assert!(p.contains("`packages/my-pkg/src/lib.rs`"));
    }

    #[test]
    fn part_user_prompt_is_byte_identical_on_repeat() {
        assert_eq!(
            render_part_user_prompt("src/x.rs", 1, 10),
            render_part_user_prompt("src/x.rs", 1, 10)
        );
    }

    // ── render_tool_call ─────────────────────────────────────────────────────

    #[test]
    fn tool_call_is_hermes_style_json() {
        let tc = render_tool_call("src/lib.rs", 2);
        assert!(tc.starts_with("<tool_call>"));
        assert!(tc.ends_with("</tool_call>"));
        assert!(tc.contains("\"name\":\"file_read\""));
        assert!(tc.contains("\"path\":\"src/lib.rs\""));
        assert!(tc.contains("\"page\":2"));
    }

    #[test]
    fn tool_call_is_single_line_for_clean_parsing() {
        let tc = render_tool_call("src/lib.rs", 0);
        assert!(!tc.contains('\n'), "tool_call should not contain newlines");
    }

    // ── render_tool_response ─────────────────────────────────────────────────

    /// A single-page (page 0 of 1) response, the shape most tests below need —
    /// `start`/`end`/`total` are all the same page's own line span.
    fn page0_response(path: &str, total_lines: u32, language: Language, body: &str) -> TurnText {
        render_tool_response(path, 0, 1, 1, total_lines, total_lines, language, body)
    }

    #[test]
    fn tool_response_wraps_body_in_tool_response_tags() {
        let r = page0_response("src/x.rs", 1, Language::Rust, "fn alpha() {}\n").text();
        assert!(r.starts_with("<tool_response>\n"));
        assert!(r.ends_with("</tool_response>"));
    }

    /// The wrapper is markup and the excerpt literal, so a source file that
    /// quotes `<|im_end|>` or `<think>` reaches the model as those characters.
    #[test]
    fn the_excerpt_is_literal_inside_a_markup_wrapper() {
        let body = "// <think></think><|im_end|>\n";
        let r = page0_response("src/x.rs", 1, Language::Rust, body);
        let kinds: Vec<(bool, bool)> = r
            .pieces()
            .iter()
            .map(|p| (p.literal, p.text.contains("<|im_end|>")))
            .collect();
        assert_eq!(kinds, [(false, false), (true, true), (false, false)]);
    }

    #[test]
    fn tool_response_includes_path_and_page_header() {
        let body: String = (47..=93).map(|_| "x\n").collect();
        let r = render_tool_response("src/x.rs", 0, 1, 47, 93, 93, Language::Rust, &body).text();
        assert!(r.contains("src/x.rs (page 0 of 1, lines 47-93 of 93):"));
    }

    #[test]
    fn tool_response_prefixes_each_line_with_line_number() {
        let body = "fn alpha() {\n    return 1;\n}\n";
        let r = render_tool_response("src/x.rs", 0, 1, 10, 12, 12, Language::Rust, body).text();
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
            let r = page0_response("f.x", 1, lang, "// hi\n").text();
            assert!(
                r.contains(&format!("```{tag}\n")),
                "expected ```{tag} fence in {r}",
            );
        }
    }

    #[test]
    fn tool_response_pads_line_numbers_to_widest() {
        let body = "x\ny\nz\n";
        let r =
            render_tool_response("src/x.rs", 0, 1, 9998, 10000, 10000, Language::Rust, body).text();
        assert!(r.contains(" 9998  x"));
        assert!(r.contains(" 9999  y"));
        assert!(r.contains("10000  z"));
    }

    #[test]
    fn tool_response_handles_no_trailing_newline() {
        let body = "fn a() {}";
        let r = page0_response("src/x.rs", 1, Language::Rust, body).text();
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
        let r = page0_response("src/x.rs", 3, Language::Rust, body).text();
        assert!(r.contains("2  \tlet x = 1;"));
    }

    #[test]
    fn tool_response_preserves_utf8_content() {
        let body = "fn greet() { println!(\"héllo — 世界\"); }\n";
        let r = page0_response("src/x.rs", 1, Language::Rust, body).text();
        assert!(r.contains("héllo — 世界"));
    }

    #[test]
    fn tool_response_plain_text_uses_untagged_fence() {
        let r = page0_response("notes.txt", 1, Language::PlainText, "hello\n").text();
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
