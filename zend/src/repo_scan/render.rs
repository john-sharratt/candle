//! Prefill rendering for the `repo_map` layer's per-directory conversation.
//!
//! A directory is explored as TWO `code_read`-shaped tool round-trips on one
//! conversation, each `request → <tool_call> → <tool_response> → DECODED answer`:
//!
//! ````text
//! Round-trip 1 — what is in here?
//!   user       List the files in the `zend/src/code_read/` folder.
//!   assistant  <tool_call>{"name":"file_list","arguments":{"prefix":"zend/src/code_read/"}}</tool_call>
//!   user       <tool_response>{"files":[…],"paging":{…},"total_bytes":0}</tool_response>
//!   assistant  ← DECODED: what the folder contains
//!
//! Round-trip 2 — what is it for?
//!   user       Summarize the `zend/src/code_read/` folder in one or two complete
//!              sentences, ending with a full stop. …
//!   assistant  <tool_call>{"name":"file_read","arguments":{"path":"zend/src/code_read/mod.rs",
//!                          "start_line":1,"end_line":24}}</tool_call>
//!   user       <tool_response>
//!              zend/src/code_read/mod.rs (lines 1-24 of 1135):
//!
//!              ```rust
//!               1  //! `code_reading` layer ingestion.
//!              24  //! …
//!              ```
//!              </tool_response>
//!   assistant  ← DECODED: the two-sentence folder summary this layer retrieves
//! ````
//!
//! **List before read.** The listing round-trip comes first so the read is a
//! discovery the conversation earned: a call naming `mod.rs` before anything
//! revealed it exists teaches the model to guess paths. A directory with no
//! anchor file is a single round-trip — the listing is the only evidence, and its
//! answer is the summary.
//!
//! **Every decode answers a request one turn back** — see
//! [`render_round_trips`] for why that invariant, and not the prose of any one
//! prompt, is what keeps the closing decode on task.
//!
//! Both tool responses are produced by invoking the REAL tools
//! ([`zend_tools::run`]) rather than hand-written here, so a prefilled response
//! cannot drift from what the model will see at runtime — including details no
//! hand-written copy would keep in step, like `serde_json` emitting object keys
//! in sorted order.

use candle_conversation::stencil::ToolCallEnvelope;
use serde_json::json;
use zend_tools::tools::file::render::numbered_excerpt;
use zend_tools::ToolContext;

use super::anchor::Anchor;
use super::dir_unit::DirUnit;

/// Tools whose definitions must be present for this chain's prefilled calls to
/// be coherent — pinned into the catalog via `FORCE_TOOL_SELECTOR`.
pub const CHAIN_TOOLS: &[&str] = &["file_list", "file_read"];

/// User-side request opening the folder's conversation. Mirrors the
/// `code_reading` scope prompt (`Summarize \`path\` (lines a-b) …`) so both
/// ingests teach one request→answer shape; the unit here is a folder, not a span.
///
/// A directory holding a manifest carries its hint in parentheses (`(crate:
/// candle-nn)`) — the one thing about a folder that the listing states only
/// obliquely, as a filename.
///
/// **"One or two complete sentences", not "no more than two sentences".** The
/// old phrasing described a ceiling and got measured against: of 61 summaries
/// from one pass, 55 were a single sentence and 31 ended with no full stop at
/// all, on a complete clause — the request said how many sentences were allowed
/// and nothing about finishing one. Naming *complete* sentences asks for the
/// property that was actually missing, and costs nothing when the model was
/// going to write one sentence anyway.
///
/// **And it must not refuse.** The listing is sometimes the only evidence (a
/// directory with no anchor file is a single round-trip — see the module docs),
/// and asked to summarize from filenames alone the model has answered "there
/// isn't enough information available here yet to summarize them accurately",
/// which seals as that folder's `repo_map` entry. Saying that names and paths
/// are a legitimate basis removes the excuse; `summarize_examples` already
/// teaches the shape.
pub fn render_request(unit: &DirUnit) -> String {
    let folder = folder_phrase(unit);
    let tail = SUMMARY_ASK;
    match unit.module_hint() {
        Some(hint) => format!("Summarize {folder} ({hint}) {tail}", hint = hint.render()),
        None => format!("Summarize {folder} {tail}"),
    }
}

/// What [`render_request`] asks for, after the folder is named. A constant so
/// the tests below assert the real string rather than a copy of it — a prompt
/// this load-bearing should not be able to drift from its own assertions.
const SUMMARY_ASK: &str = "in one or two complete sentences, ending with a full stop. \
     Summarize from whatever the conversation has already shown you — file names and \
     paths alone are enough; never reply that there is not enough information.";

/// How a request names the folder. A real directory is named by its path in
/// backticks; the workspace root is named in words. `.` is the tag and cache key,
/// not something to show a reader — asked to summarize "the `.` folder" the model
/// writes about "the `.()` directory".
fn folder_phrase(unit: &DirUnit) -> String {
    if unit.list_prefix().is_empty() {
        "the root folder of this project".to_string()
    } else {
        format!("the `{}` folder", unit.label())
    }
}

/// Assistant-side `<tool_call>` listing the folder, in the checkpoint's own
/// call syntax.
///
/// **Rendered from the dialect's envelope, not written out here.** These calls
/// are PREFILLED — the model reads them as its own prior output, so their shape
/// is what it learns to produce. Written as Qwen3 JSON they taught
/// `{"name":…,"arguments":…}` to a Qwen3.5/3.8 checkpoint whose grammar emits a
/// `<function=…>` element, so the ingest and the chat stencil disagreed about
/// the syntax of the same tool on the same weights — thousands of turns of it
/// in one pass. [`ToolCallEnvelope::for_dialect`] exists to be the single
/// answer to that question; see its note on a literal being "a second opinion
/// about the checkpoint actually loaded".
pub fn render_list_call(env: &ToolCallEnvelope, unit: &DirUnit) -> String {
    env.render("file_list", &[("prefix", unit.list_prefix())])
}

/// Assistant-side `<tool_call>` reading the anchor excerpt, in the checkpoint's
/// own call syntax. See [`render_list_call`].
pub fn render_read_call(env: &ToolCallEnvelope, anchor: &Anchor) -> String {
    env.render(
        "file_read",
        &[
            ("path", anchor.path.as_str()),
            ("start_line", &anchor.start_line.to_string()),
            ("end_line", &anchor.end_line.to_string()),
        ],
    )
}

/// User-side `<tool_response>` for the listing — produced by running the real
/// `file_list` against `ctx`, so the bytes are the tool's own.
pub fn render_list_response(ctx: &ToolContext, unit: &DirUnit) -> String {
    let args = json!({ "prefix": unit.list_prefix() });
    let value = zend_tools::run("file_list", "repo_map_prefill", &args, ctx);
    let body = serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string());
    format!("<tool_response>{body}</tool_response>")
}

/// User-side `<tool_response>` for the anchor excerpt — the same numbered, fenced
/// shape the live `file_read` returns and the `code_reading` ingest prefills.
pub fn render_read_response(anchor: &Anchor) -> String {
    let excerpt = numbered_excerpt(
        &anchor.path,
        anchor.start_line,
        anchor.end_line,
        anchor.total_lines,
        anchor.language.fence_tag(),
        &anchor.body,
    );
    format!("<tool_response>{excerpt}</tool_response>")
}

/// The `error` field of a `<tool_response>` body, when it carries one. Only a
/// JSON body can be an error — the excerpt response is raw numbered text.
fn error_detail(turn: &str) -> Option<String> {
    let body = turn
        .strip_prefix("<tool_response>")?
        .strip_suffix("</tool_response>")?;
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let error = value.get("error")?.as_str()?;
    let detail = value.get("detail").and_then(|d| d.as_str()).unwrap_or("");
    Some(format!("{error}: {detail}"))
}

/// The whole chain for one directory: the prefilled `(user, assistant)` pairs
/// followed by the final user turn whose assistant half the model DECODES.
///
/// ONE request, two tool calls, one decode. The request opens the chain and the
/// decode closes it, with the listing and the excerpt gathered in between —
/// which is what a real exploration looks like, and what `code_read`'s scope
/// read is, one call longer.
///
/// This only works because an ingest conversation projects its OWN turns (see
/// `ContentResolver::target_is_ingest_self`). While those turns were belief-
/// gated out, the decode could see nothing but the excerpt in its own user turn,
/// and every workaround for that — restating the ask inside the decode's turn,
/// splitting into two round-trips so a request sat adjacent — was compensating
/// for the missing history rather than fixing it. Splitting in particular cost a
/// throwaway "what is in here" decode whose chatty answer then set the style for
/// the summary that followed.
///
/// Two pairs when the folder has an anchor (list, then read), one when it does
/// not — the listing is then the only evidence and the summary follows it.
pub fn render_chain(
    ctx: &ToolContext,
    unit: &DirUnit,
    env: &ToolCallEnvelope,
) -> (Vec<(String, String)>, String) {
    let listing = render_list_response(ctx, unit);
    match &unit.anchor {
        Some(anchor) => (
            vec![
                (render_request(unit), render_list_call(env, unit)),
                (listing, render_read_call(env, anchor)),
            ],
            render_read_response(anchor),
        ),
        None => (
            vec![(render_request(unit), render_list_call(env, unit))],
            listing,
        ),
    }
}

/// The tool-error detail carried by a rendered chain, if any of its tool
/// responses is an error rather than a result.
///
/// `zend_tools::run` reports a failure as a JSON body with an `error` key rather
/// than by returning `Err`, so a directory the tools cannot read would otherwise
/// prefill an error as if it were evidence — teaching the model a tool
/// interaction that failed, and grounding the folder's summary in nothing.
pub fn chain_error(prefilled: &[(String, String)], decode_user: &str) -> Option<String> {
    prefilled
        .iter()
        .map(|(user, _)| user.as_str())
        .chain(std::iter::once(decode_user))
        .find_map(error_detail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo_scan::dir_unit::build_units;
    use crate::repo_scan::types::{FileEntry, Language, ModuleHint, RepoMap};
    use candle_conversation::models::Dialect;
    use std::path::Path;

    /// The envelope the chain tests render through. ChatML, so the assertions
    /// that predate the dialect wiring keep asserting the shape they always did;
    /// `tool_calls_follow_the_dialects_call_style` covers the other style.
    fn env() -> ToolCallEnvelope {
        ToolCallEnvelope::for_dialect(&Dialect::chat_ml())
    }

    fn workspace(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (rel, body) in files {
            let p = dir.path().join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, body).unwrap();
        }
        dir
    }

    fn map_of(root: &Path, paths: &[(&str, Language)]) -> RepoMap {
        let _ = root;
        RepoMap {
            files: paths
                .iter()
                .map(|(p, l)| FileEntry {
                    path: p.to_string(),
                    line_count: 1,
                    language: *l,
                    size_bytes: 1,
                    module_hint: None,
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn request_names_the_folder_and_asks_for_complete_sentences() {
        let d = workspace(&[("a/x.rs", "fn x() {}\n")]);
        let m = map_of(d.path(), &[("a/x.rs", Language::Rust)]);
        let units = build_units(&m, d.path());
        assert_eq!(
            render_request(&units[0]),
            format!("Summarize the `a/` folder {SUMMARY_ASK}"),
        );
    }

    /// The two properties the ask exists to obtain, named so a reworded prompt
    /// that drops either one fails here rather than in a substrate audit weeks
    /// later: **complete** sentences with a terminal stop (31 of 61 summaries in
    /// one pass ended with no full stop), and no refusal when the file listing is
    /// the only evidence (which sealed "there isn't enough information available
    /// here yet" as a folder's summary).
    #[test]
    fn the_ask_demands_a_full_stop_and_forbids_a_refusal() {
        assert!(
            SUMMARY_ASK.contains("complete sentences") && SUMMARY_ASK.contains("full stop"),
            "the ask must require a terminated sentence: {SUMMARY_ASK}"
        );
        assert!(
            SUMMARY_ASK.contains("never reply that there is not enough information"),
            "the ask must forbid a refusal: {SUMMARY_ASK}"
        );
        assert!(
            !SUMMARY_ASK.contains("no more than"),
            "a ceiling on sentence COUNT is what produced 55 one-sentence answers \
             out of 61; the ask names completeness instead"
        );
    }

    /// The workspace root is named in words. Asked to summarize "the `.` folder"
    /// the model writes about "the `.()` directory" — `.` is the tag and cache
    /// key, never something to put in front of a reader.
    #[test]
    fn the_workspace_root_is_named_in_words_not_as_a_dot() {
        let d = workspace(&[(
            "top.rs",
            "fn x() {}
",
        )]);
        let m = map_of(d.path(), &[("top.rs", Language::Rust)]);
        let units = build_units(&m, d.path());
        assert_eq!(units[0].dir, ".", "the tag/cache key stays `.`");

        let request = render_request(&units[0]);
        assert_eq!(
            request,
            format!("Summarize the root folder of this project {SUMMARY_ASK}"),
        );
        assert!(!request.contains('`'), "no backticked path for the root");
    }

    /// A crate root announces itself rather than leaving the model to infer it
    /// from a `Cargo.toml` in the listing.
    #[test]
    fn request_carries_a_manifest_hint_when_the_folder_has_one() {
        let d = workspace(&[("a/x.rs", "fn x() {}\n")]);
        let mut m = map_of(d.path(), &[("a/x.rs", Language::Rust)]);
        m.files[0].module_hint = Some(ModuleHint::CargoPackage {
            name: "demo".to_string(),
        });
        let units = build_units(&m, d.path());
        assert_eq!(
            render_request(&units[0]),
            format!("Summarize the `a/` folder (crate: demo) {SUMMARY_ASK}"),
        );
    }

    /// A JSON-dialect checkpoint gets the Hermes object — **in the layout the
    /// grammar compiles**, which is not quite the layout this was hardcoded to.
    ///
    /// The old literal was one unspaced line (`{"name":"file_list",…}`); the
    /// envelope the stencil builds from opens `<tool_call>\n{"name": "` and
    /// closes `}}\n</tool_call>`. So the prefill diverged from the model's own
    /// output even on Qwen3 — mildly, in whitespace, rather than in syntax, but
    /// in the same direction and for the same reason. Asserted in full here so
    /// the prefill and the grammar stay one shape.
    #[test]
    fn tool_calls_are_hermes_json_on_a_json_dialect() {
        let env = ToolCallEnvelope::for_dialect(&Dialect::chat_ml());
        let d = workspace(&[("a/mod.rs", "//! One.\n//! Two.\nfn x() {}\n")]);
        let m = map_of(d.path(), &[("a/mod.rs", Language::Rust)]);
        let units = build_units(&m, d.path());
        let call = render_list_call(&env, &units[0]);
        assert_eq!(
            call,
            "<tool_call>\n{\"name\": \"file_list\", \"arguments\": {\"prefix\": \"a/\"}}\n</tool_call>",
        );

        let read = render_read_call(&env, units[0].anchor.as_ref().unwrap());
        assert_eq!(
            read,
            "<tool_call>\n{\"name\": \"file_read\", \"arguments\": {\"path\": \"a/mod.rs\", \
             \"start_line\": 1, \"end_line\": 2}}\n</tool_call>",
        );
        assert!(
            read.contains("\"start_line\": 1"),
            "a line number is a JSON number, as the grammar emits it: {read}"
        );
    }

    /// **The prefills speak the checkpoint's syntax, not a literal's.**
    ///
    /// These calls are prefilled as the model's own prior output, so their shape
    /// is what it learns to emit. Hardcoded as Qwen3 JSON they taught
    /// `{"name":…}` to a Qwen3.5/3.8 checkpoint whose grammar emits a
    /// `<function=…>` element — the ingest and the chat stencil disagreeing
    /// about the same tool on the same weights, for a whole workspace pass.
    /// Nothing asserted the connection, which is why it survived the merge that
    /// introduced `FunctionBlock`.
    #[test]
    fn tool_calls_follow_the_dialects_call_style() {
        let d = workspace(&[("a/mod.rs", "//! One.\n//! Two.\nfn x() {}\n")]);
        let m = map_of(d.path(), &[("a/mod.rs", Language::Rust)]);
        let units = build_units(&m, d.path());
        let anchor = units[0].anchor.as_ref().unwrap();

        // What the ingest prefills must equal what the DIALECT's envelope
        // renders — asserted against the envelope rather than against a literal
        // shape, because a literal here is the second opinion that caused the
        // original divergence. `Dialect::qwen35` declares `CallStyle::JsonBlock`
        // today and declared `FunctionBlock` yesterday; this test is about the
        // two staying joined, not about which one is current.
        let env = ToolCallEnvelope::for_dialect(&Dialect::qwen35());
        assert_eq!(
            render_list_call(&env, &units[0]),
            env.render("file_list", &[("prefix", units[0].list_prefix())]),
            "the listing call must be the dialect envelope's own rendering",
        );
        assert_eq!(
            render_read_call(&env, anchor),
            env.render(
                "file_read",
                &[
                    ("path", anchor.path.as_str()),
                    ("start_line", &anchor.start_line.to_string()),
                    ("end_line", &anchor.end_line.to_string()),
                ],
            ),
            "the read call must be the dialect envelope's own rendering",
        );

        // And it is the CHECKPOINT's envelope, not a hardcoded family: ask the
        // dialect for a different style and the rendering follows it.
        let lines = ToolCallEnvelope::for_dialect(&Dialect::llama3());
        assert_eq!(
            render_list_call(&lines, &units[0]),
            lines.render("file_list", &[("prefix", units[0].list_prefix())]),
        );
    }

    /// A path is the one part of a call an author does not control. Spliced in
    /// raw, a quote or backslash would emit a `<tool_call>` the extractor cannot
    /// parse — so the arguments must be JSON-escaped and stay round-trippable.
    ///
    /// This is the surviving form of a concern that used to need two tests. While
    /// `Dialect::qwen35` declared `CallStyle::FunctionBlock`, values rode raw and
    /// the hazard was a value containing `</parameter>` and closing its own span;
    /// that dialect now declares `JsonBlock`, so there are no raw values and no
    /// `</parameter>` anywhere, and escaping is the whole of the question.
    #[test]
    fn a_path_with_json_metacharacters_stays_parseable() {
        let anchor = Anchor {
            path: "a/we\"ird\\dir/mod.rs".to_string(),
            start_line: 1,
            end_line: 2,
            total_lines: 3,
            body: "//! One.\n//! Two.\n".to_string(),
            language: Language::Rust,
        };
        let env = ToolCallEnvelope::for_dialect(&Dialect::chat_ml());
        let call = render_read_call(&env, &anchor);
        let body = call
            .strip_prefix("<tool_call>")
            .and_then(|s| s.trim_end().strip_suffix("</tool_call>"))
            .map(str::trim_end)
            .expect("tag wrapper");
        let parsed: serde_json::Value = serde_json::from_str(body).expect("valid JSON");
        assert_eq!(parsed["name"], "file_read");
        assert_eq!(parsed["arguments"]["path"], "a/we\"ird\\dir/mod.rs");
    }

    /// The excerpt response is the shared numbered/fenced format, framed in tags.
    #[test]
    fn read_response_is_the_shared_excerpt_format() {
        let d = workspace(&[("a/mod.rs", "//! One.\n//! Two.\nfn x() {}\n")]);
        let m = map_of(d.path(), &[("a/mod.rs", Language::Rust)]);
        let units = build_units(&m, d.path());
        let out = render_read_response(units[0].anchor.as_ref().unwrap());
        assert_eq!(
            out,
            "<tool_response>\na/mod.rs (lines 1-2 of 3):\n\n```rust\n1  //! One.\n2  //! Two.\n```\n</tool_response>",
        );
    }

    /// The listing response is the live tool's own bytes — the test asserts the
    /// framing and that the tool actually ran against the workspace.
    #[test]
    fn list_response_comes_from_the_real_tool() {
        let d = workspace(&[
            ("a/mod.rs", "//! One.\n//! Two.\n"),
            ("a/x.rs", "fn x() {}\n"),
        ]);
        let m = map_of(
            d.path(),
            &[("a/mod.rs", Language::Rust), ("a/x.rs", Language::Rust)],
        );
        let units = build_units(&m, d.path());
        let ctx = ToolContext::with_workspace(d.path());
        let out = render_list_response(&ctx, &units[0]);
        assert!(out.starts_with("<tool_response>{"), "{out}");
        assert!(out.ends_with("</tool_response>"));
        assert!(out.contains("\"path\":\"a/mod.rs\""), "{out}");
        assert!(out.contains("\"paging\""), "{out}");
    }

    /// The chain lists BEFORE it reads: a `file_read` naming `mod.rs` before
    /// anything revealed the file exists teaches the model to guess paths.
    #[test]
    fn a_folder_with_an_anchor_reads_after_it_lists() {
        let d = workspace(&[(
            "a/mod.rs",
            "//! One.
//! Two.
fn x() {}
",
        )]);
        let m = map_of(d.path(), &[("a/mod.rs", Language::Rust)]);
        let units = build_units(&m, d.path());
        let ctx = ToolContext::with_workspace(d.path());
        let (prefilled, decode_user) = render_chain(&ctx, &units[0], &env());
        assert_eq!(prefilled.len(), 2, "request+list, listing+read");
        assert!(prefilled[0].0.starts_with("Summarize the `a/` folder"));
        assert!(prefilled[0].1.contains("\"name\": \"file_list\""));
        assert!(
            prefilled[1].0.starts_with("<tool_response>{"),
            "the listing"
        );
        assert!(prefilled[1].1.contains("\"name\": \"file_read\""));
        assert!(
            decode_user.contains("```rust"),
            "the decode follows the excerpt"
        );
    }

    /// No anchor: one prefilled pair, and the listing is the last evidence.
    #[test]
    fn a_folder_without_an_anchor_stops_at_the_listing() {
        let d = workspace(&[(
            "a/x.rs",
            "fn x() {}
",
        )]);
        let m = map_of(d.path(), &[("a/x.rs", Language::Rust)]);
        let units = build_units(&m, d.path());
        let ctx = ToolContext::with_workspace(d.path());
        let (prefilled, decode_user) = render_chain(&ctx, &units[0], &env());
        assert_eq!(prefilled.len(), 1);
        assert!(prefilled[0].1.contains("\"name\": \"file_list\""));
        assert!(decode_user.starts_with("<tool_response>{"));
    }

    /// A healthy chain reports no tool error.
    #[test]
    fn a_successful_chain_carries_no_tool_error() {
        let d = workspace(&[(
            "a/mod.rs",
            "//! One.
//! Two.
fn x() {}
",
        )]);
        let m = map_of(d.path(), &[("a/mod.rs", Language::Rust)]);
        let units = build_units(&m, d.path());
        let ctx = ToolContext::with_workspace(d.path());
        let (prefilled, decode_user) = render_chain(&ctx, &units[0], &env());
        assert_eq!(chain_error(&prefilled, &decode_user), None);
    }

    /// A tool failure surfaces as an `error` body, not an `Err` — so it must be
    /// detected from the rendered text or it would be prefilled as evidence.
    #[test]
    fn an_error_tool_response_is_detected() {
        let turn = "<tool_response>{\"error\":\"unknown_tool\",\"detail\":\"no tool registered\"}</tool_response>";
        assert_eq!(
            chain_error(&[(turn.to_string(), String::new())], ""),
            Some("unknown_tool: no tool registered".to_string()),
        );
        assert_eq!(
            chain_error(&[], turn),
            Some("unknown_tool: no tool registered".to_string()),
            "the decode-side response is checked too",
        );
    }

    /// The excerpt response is raw numbered text, not JSON — it must never be
    /// mistaken for a malformed error body.
    #[test]
    fn an_excerpt_response_is_not_an_error() {
        let d = workspace(&[("a/mod.rs", "//! One.\n//! Two.\nfn x() {}\n")]);
        let m = map_of(d.path(), &[("a/mod.rs", Language::Rust)]);
        let units = build_units(&m, d.path());
        let excerpt = render_read_response(units[0].anchor.as_ref().unwrap());
        assert_eq!(chain_error(&[], &excerpt), None);
    }

    /// Rendering is deterministic — the resume cache depends on it.
    #[test]
    fn rendering_is_byte_identical_on_repeat() {
        let d = workspace(&[("a/mod.rs", "//! One.\n//! Two.\nfn x() {}\n")]);
        let m = map_of(d.path(), &[("a/mod.rs", Language::Rust)]);
        let units = build_units(&m, d.path());
        let ctx = ToolContext::with_workspace(d.path());
        assert_eq!(render_chain(&ctx, &units[0], &env()), render_chain(&ctx, &units[0], &env()));
    }
}
