//! Prefill rendering for the `repo_map` layer's per-directory conversation.
//!
//! A directory is ONE `code_read`-shaped tool round-trip on one conversation —
//! `request → <tool_call> → <tool_response> → DECODED answer`:
//!
//! ````text
//!   user       Summarize the `zend/src/code_read/` folder in the `candle`
//!              repository in ONE complete sentence, ending with a full stop. …
//!   assistant  <tool_call>{"name":"file_list","arguments":{"repo":"candle",
//!                          "path":"zend/src/code_read"}}</tool_call>
//!   user       <tool_response>{"repo":"candle","entries":[…],"paging":{…},…}</tool_response>
//!   assistant  ← DECODED: the one-sentence folder summary this layer retrieves
//! ````
//!
//! **The listing is the whole of the evidence, and that is deliberate.** A
//! second round-trip used to read an "anchor" file — a `README.md` or module
//! root — on the reasoning that a folder is best described by the file that
//! describes it. Two things were wrong with it. The excerpt is a large file
//! pasted into the turn, so the root folder's unit carried 9,315 characters to
//! produce one sentence, paid again on every projection that selects it. And a
//! prefilled `file_read` is a *suggestion*: the model answers the call it is
//! shown, so given one it issued another, and the root's conversation spent all
//! three turns reading `README.md` — the second time for its last five lines —
//! and never wrote a summary at all.
//!
//! The tool response is produced by invoking the REAL tool
//! ([`zend_tools::run`]) rather than hand-written here, so a prefilled response
//! cannot drift from what the model will see at runtime — including details no
//! hand-written copy would keep in step, like the order `serde_json` emits a
//! response's keys in (the struct's field order — the workspace builds it with
//! `preserve_order`).
//!
//! A unit's directory is workspace-relative (`candle/zend/src/`); the call
//! addresses it as its repository and the path inside it
//! ([`crate::repo_path::split`]), the arguments the live tool takes — every
//! folder is inside one repository, as every `file_list` is.

use candle_conversation::stencil::ToolCallEnvelope;
use candle_conversation::TurnText;
use serde_json::{json, Map, Value};
use zend_tools::ToolContext;

use super::dir_unit::DirUnit;
use crate::repo_path::split;

/// Tools whose definitions must be present for this chain's prefilled calls to
/// be coherent — pinned into the catalog via `FORCE_TOOL_SELECTOR`.
pub const CHAIN_TOOLS: &[&str] = &["file_list", "file_read"];

/// User-side request opening the folder's conversation. Mirrors the
/// `code_reading` request so both ingests teach one request→answer shape; the
/// unit here is a folder, not a file.
///
/// A directory holding a manifest carries its hint in parentheses (`(crate:
/// candle-nn)`) — the one thing about a folder that the listing states only
/// obliquely, as a filename.
///
/// **"ONE complete sentence, ending with a full stop".** An older phrasing
/// described a ceiling ("no more than two sentences") and got measured against:
/// of 61 summaries from one pass, 55 were a single sentence and 31 ended with no
/// full stop at all, on a complete clause — the request said how many sentences
/// were allowed and nothing about finishing one. Naming a *complete* sentence
/// asks for the property that was actually missing.
///
/// **And it must not refuse.** The listing is the only evidence, and asked to
/// summarize from filenames alone the model has answered "there isn't enough
/// information available here yet to summarize them accurately", which seals as
/// that folder's `repo_map` entry. Saying that names and paths are a legitimate
/// basis removes the excuse.
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
const SUMMARY_ASK: &str = "in ONE complete sentence, ending with a full stop. \
     The file listing is the evidence — names and paths alone are enough; never reply \
     that there is not enough information, and do not read any file.";

/// How a request names the folder: a repository's root by the repository's
/// name, any other folder by its path in backticks with the repository it is
/// in.
fn folder_phrase(unit: &DirUnit) -> String {
    let (repo, inner) = split(&unit.dir);
    folder_anchor(repo, inner)
}

/// How a folder unit names its folder — the repository, and the folder inside
/// it (`""` for its root) — which is the folder's anchor. The listing itself
/// names only its repository, and the call that made it is rendered in the
/// checkpoint's own syntax, so the request's phrase is the one string every
/// folder unit carries regardless of dialect. A reply that points the model at
/// a listing it already holds names it by exactly this.
pub fn folder_anchor(repo: &str, inner: &str) -> String {
    if inner.is_empty() {
        format!("the `{repo}` repository")
    } else {
        format!("the `{inner}/` folder in the `{repo}` repository")
    }
}

/// The `file_list` arguments for `unit`: the repository alone for its root,
/// the repository and path otherwise — in the order the tool's schema
/// declares them.
fn list_args(unit: &DirUnit) -> Vec<(&'static str, &str)> {
    match split(&unit.dir) {
        (repo, "") => vec![("repo", repo)],
        (repo, inner) => vec![("repo", repo), ("path", inner)],
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
    env.render("file_list", &list_args(unit))
}

/// User-side `<tool_response>` for the listing — produced by running the real
/// `file_list` against `ctx`, so the bytes are the tool's own. The listing is
/// literal: a file name is the workspace's text, not markup.
pub fn render_list_response(ctx: &ToolContext, unit: &DirUnit) -> TurnText {
    let args: Map<String, Value> = list_args(unit)
        .into_iter()
        .map(|(k, v)| (k.to_string(), json!(v)))
        .collect();
    let value = zend_tools::run("file_list", "repo_map_prefill", &Value::Object(args), ctx);
    let body = serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string());
    TurnText::markup("<tool_response>")
        .then_literal(body)
        .then_markup("</tool_response>")
}

/// The `error` field of a `<tool_response>` body, when it carries one.
fn error_detail(turn: &str) -> Option<String> {
    let body = turn
        .strip_prefix("<tool_response>")?
        .strip_suffix("</tool_response>")?;
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let error = value.get("error")?.as_str()?;
    let detail = value.get("detail").and_then(|d| d.as_str()).unwrap_or("");
    Some(format!("{error}: {detail}"))
}

/// The whole chain for one directory: the prefilled `(user, assistant)` pair
/// followed by the final user turn whose assistant half the model DECODES.
///
/// **One pair, always: the request and the folder's listing.** A folder is
/// summarised from what is in it, and `file_list` is what says that. The
/// request opens the chain and the decode closes it, with the listing in
/// between — see the module docs for why a second, reading round-trip was
/// removed.
///
/// This only works because an ingest conversation projects its OWN turns (see
/// `ContentResolver::target_is_ingest_self`). While those turns were belief-
/// gated out, the decode could see nothing but the listing in its own user
/// turn, and every workaround for that — restating the ask inside the decode's
/// turn, splitting into two round-trips so a request sat adjacent — was
/// compensating for the missing history rather than fixing it.
pub fn render_chain(
    ctx: &ToolContext,
    unit: &DirUnit,
    env: &ToolCallEnvelope,
) -> (Vec<(TurnText, String)>, TurnText) {
    (
        vec![(render_request(unit).into(), render_list_call(env, unit))],
        render_list_response(ctx, unit),
    )
}

/// The tool-error detail carried by a rendered chain, if any of its tool
/// responses is an error rather than a result.
///
/// `zend_tools::run` reports a failure as a JSON body with an `error` key rather
/// than by returning `Err`, so a directory the tools cannot read would otherwise
/// prefill an error as if it were evidence — teaching the model a tool
/// interaction that failed, and grounding the folder's summary in nothing.
pub fn chain_error(prefilled: &[(TurnText, String)], decode_user: &TurnText) -> Option<String> {
    prefilled
        .iter()
        .map(|(user, _)| user)
        .chain(std::iter::once(decode_user))
        .find_map(|turn| error_detail(&turn.text()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::branch_ingest::units::test_units_reading;
    use crate::repo_scan::types::Language;
    use candle_conversation::models::Dialect;
    use std::path::Path;
    use zend_vfs::{RepoSpec, Workspace};

    /// A tool context over `d` as a workspace whose repositories are its
    /// top-level folders — every test here keys its files under repository `a`.
    fn ctx_for(d: &tempfile::TempDir) -> ToolContext {
        let repos = std::fs::read_dir(d.path())
            .unwrap()
            .flatten()
            .filter(|e| e.path().is_dir())
            .map(|e| RepoSpec::named(&e.file_name().to_string_lossy()))
            .collect();
        ToolContext::with_workspace(Workspace::new(d.path(), repos).unwrap())
    }

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

    /// The folder units of one repository's files, each manifest read from
    /// `root` as the walk reads it from its commit.
    fn build_units(root: &Path, paths: &[(&str, Language)]) -> Vec<DirUnit> {
        test_units_reading(paths, |path| std::fs::read(root.join(path)).ok())
            .iter()
            .map(DirUnit::of)
            .collect()
    }

    #[test]
    fn request_names_the_folder_and_asks_for_complete_sentences() {
        let d = workspace(&[("a/src/x.rs", "fn x() {}\n")]);
        let units = build_units(d.path(), &[("a/src/x.rs", Language::Rust)]);
        assert_eq!(
            render_request(&units[0]),
            format!("Summarize the `src/` folder in the `a` repository {SUMMARY_ASK}"),
        );
    }

    /// A repository's own root is named as the repository.
    #[test]
    fn a_repository_root_is_named_as_the_repository() {
        let d = workspace(&[("a/x.rs", "fn x() {}\n")]);
        let units = build_units(d.path(), &[("a/x.rs", Language::Rust)]);
        assert_eq!(
            render_request(&units[0]),
            format!("Summarize the `a` repository {SUMMARY_ASK}"),
        );
    }

    /// Every folder unit's request carries the folder's anchor verbatim — the
    /// string a served `file_list` points the model at.
    #[test]
    fn the_request_carries_the_folders_anchor() {
        let d = workspace(&[("a/src/x.rs", "fn x() {}\n"), ("a/y.rs", "fn y() {}\n")]);
        let units = build_units(
            d.path(),
            &[("a/src/x.rs", Language::Rust), ("a/y.rs", Language::Rust)],
        );
        let anchors: Vec<String> = units
            .iter()
            .map(|u| {
                let (repo, inner) = split(&u.dir);
                folder_anchor(repo, inner)
            })
            .collect();
        assert!(anchors.contains(&"the `a` repository".to_string()));
        assert!(anchors.contains(&"the `src/` folder in the `a` repository".to_string()));
        for (unit, anchor) in units.iter().zip(&anchors) {
            assert!(render_request(unit).contains(anchor.as_str()), "{anchor}");
        }
    }

    /// The three properties the ask exists to obtain, named so a reworded prompt
    /// that drops one fails here rather than in a substrate audit weeks later: a
    /// **complete** sentence with a terminal stop (31 of 61 summaries in one pass
    /// ended with no full stop), no refusal when the listing is the only evidence
    /// (which sealed "there isn't enough information available here yet" as a
    /// folder's summary), and no file reading — the chain offers `file_list`
    /// alone, and a model that asks for a `file_read` spends the unit's whole
    /// budget on it and never writes the summary.
    #[test]
    fn the_ask_demands_one_full_stopped_sentence_and_no_reading() {
        assert!(
            SUMMARY_ASK.contains("ONE complete sentence") && SUMMARY_ASK.contains("full stop"),
            "the ask must require a terminated sentence: {SUMMARY_ASK}"
        );
        assert!(
            SUMMARY_ASK.contains("do not read any file"),
            "the ask must forbid reading: {SUMMARY_ASK}"
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

    /// **A repository's root is listed with the repository alone** — `file_list`
    /// lists inside one repository, never the workspace, so no call names `*`
    /// — and its response holds that repository's own entries, repo-relative.
    #[test]
    fn a_repository_root_lists_with_the_repository_alone() {
        let d = workspace(&[("a/x.rs", "fn x() {}\n"), ("b/y.rs", "fn y() {}\n")]);
        let units = build_units(d.path(), &[("a/x.rs", Language::Rust)]);
        let root = &units[0];
        assert_eq!(root.dir, "a/");
        assert_eq!(
            render_list_call(&env(), root),
            "<tool_call>\n{\"name\": \"file_list\", \"arguments\": {\"repo\": \"a\"}}\n</tool_call>",
        );
        let listing = render_list_response(&ctx_for(&d), root).text();
        assert!(
            listing.starts_with(
                "<tool_response>{\"repo\":\"a\",\"entries\":[{\"path\":\"x.rs\",\"bytes\":10}]"
            ),
            "{listing}"
        );
        assert!(!listing.contains("y.rs"), "b's files are b's: {listing}");
    }

    /// A crate root announces itself rather than leaving the model to infer it
    /// from a `Cargo.toml` in the listing.
    #[test]
    fn request_carries_a_manifest_hint_when_the_folder_has_one() {
        let d = workspace(&[
            ("a/Cargo.toml", "[package]\nname = \"demo\"\n"),
            ("a/x.rs", "fn x() {}\n"),
        ]);
        let units = build_units(
            d.path(),
            &[("a/Cargo.toml", Language::Toml), ("a/x.rs", Language::Rust)],
        );
        assert_eq!(
            render_request(&units[0]),
            format!("Summarize the `a` repository (crate: demo) {SUMMARY_ASK}"),
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
        let d = workspace(&[("a/src/mod.rs", "//! One.\n//! Two.\nfn x() {}\n")]);
        let units = build_units(d.path(), &[("a/src/mod.rs", Language::Rust)]);
        let call = render_list_call(&env, &units[0]);
        assert_eq!(
            call,
            "<tool_call>\n{\"name\": \"file_list\", \"arguments\": {\"repo\": \"a\", \
             \"path\": \"src\"}}\n</tool_call>",
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
        let units = build_units(d.path(), &[("a/mod.rs", Language::Rust)]);

        // What the ingest prefills must equal what the DIALECT's envelope
        // renders — asserted against the envelope rather than against a literal
        // shape, because a literal here is the second opinion that caused the
        // original divergence. `Dialect::qwen35` declares `CallStyle::JsonBlock`
        // today and declared `FunctionBlock` yesterday; this test is about the
        // two staying joined, not about which one is current.
        let env = ToolCallEnvelope::for_dialect(&Dialect::qwen35());
        assert_eq!(
            render_list_call(&env, &units[0]),
            env.render("file_list", &[("repo", "a")]),
            "the listing call must be the dialect envelope's own rendering",
        );
        // And it is the CHECKPOINT's envelope, not a hardcoded family: ask the
        // dialect for a different style and the rendering follows it.
        let lines = ToolCallEnvelope::for_dialect(&Dialect::llama3());
        assert_eq!(
            render_list_call(&lines, &units[0]),
            lines.render("file_list", &[("repo", "a")]),
        );
    }

    /// A path is the one part of a call an author does not control. Spliced in
    /// raw, a quote or backslash would emit a `<tool_call>` the extractor cannot
    /// parse — so the arguments must be JSON-escaped and stay round-trippable.
    #[test]
    fn a_path_with_json_metacharacters_stays_parseable() {
        // The unit is built by hand rather than from a real directory: Windows
        // refuses a filename containing `"`, and the escaping under test is the
        // renderer's, not the filesystem's.
        let unit = DirUnit {
            dir: "a/we\"ird\\dir/".to_string(),
            listed: Vec::new(),
            module_hint: None,
            content_key: "abc".to_string(),
        };
        let env = ToolCallEnvelope::for_dialect(&Dialect::chat_ml());
        let call = render_list_call(&env, &unit);
        let body = call
            .strip_prefix("<tool_call>")
            .and_then(|s| s.trim_end().strip_suffix("</tool_call>"))
            .map(str::trim_end)
            .expect("tag wrapper");
        let parsed: serde_json::Value = serde_json::from_str(body).expect("valid JSON");
        assert_eq!(parsed["name"], "file_list");
        assert_eq!(parsed["arguments"]["repo"], "a");
        assert_eq!(parsed["arguments"]["path"], "we\"ird\\dir");
    }

    /// The listing response is the live tool's own bytes — the test asserts the
    /// framing and that the tool actually ran against the workspace.
    #[test]
    fn list_response_comes_from_the_real_tool() {
        let d = workspace(&[
            ("a/mod.rs", "//! One.\n//! Two.\n"),
            ("a/x.rs", "fn x() {}\n"),
        ]);
        let units = build_units(
            d.path(),
            &[("a/mod.rs", Language::Rust), ("a/x.rs", Language::Rust)],
        );
        let ctx = ctx_for(&d);
        let out = render_list_response(&ctx, &units[0]).text();
        assert!(out.starts_with("<tool_response>{"), "{out}");
        assert!(out.ends_with("</tool_response>"));
        assert!(out.contains("\"path\":\"mod.rs\""), "{out}");
        assert!(out.contains("\"repo\":\"a\""), "{out}");
        assert!(out.contains("\"paging\""), "{out}");
    }

    /// **A folder holding a README still only lists.** The README is named in
    /// the listing and never opened: no read call, and no excerpt of it anywhere
    /// in the turns.
    #[test]
    fn a_folder_with_a_readme_still_only_lists() {
        let d = workspace(&[
            ("a/mod.rs", "//! One.\n//! Two.\nfn x() {}\n"),
            ("a/README.md", "# a\n\nHolds the widgets.\n"),
        ]);
        let units = build_units(
            d.path(),
            &[
                ("a/mod.rs", Language::Rust),
                ("a/README.md", Language::Markdown),
            ],
        );
        let ctx = ctx_for(&d);
        let (prefilled, decode_user) = render_chain(&ctx, &units[0], &env());
        assert_eq!(prefilled.len(), 1, "request+list, and nothing after it");
        assert!(prefilled[0]
            .0
            .text()
            .starts_with("Summarize the `a` repository"));
        assert!(prefilled[0].1.contains("\"name\": \"file_list\""));
        assert!(
            decode_user.text().starts_with("<tool_response>{"),
            "the decode follows the listing",
        );
        let whole = format!("{}{}", prefilled[0].1, decode_user.text());
        assert!(!whole.contains("file_read"), "no read call: {whole}");
        assert!(!whole.contains("Holds the widgets"), "no excerpt: {whole}");
    }

    /// A healthy chain reports no tool error.
    #[test]
    fn a_successful_chain_carries_no_tool_error() {
        let d = workspace(&[("a/mod.rs", "//! One.\n//! Two.\nfn x() {}\n")]);
        let units = build_units(d.path(), &[("a/mod.rs", Language::Rust)]);
        let ctx = ctx_for(&d);
        let (prefilled, decode_user) = render_chain(&ctx, &units[0], &env());
        assert_eq!(chain_error(&prefilled, &decode_user), None);
    }

    /// A tool failure surfaces as an `error` body, not an `Err` — so it must be
    /// detected from the rendered text or it would be prefilled as evidence.
    #[test]
    fn an_error_tool_response_is_detected() {
        let turn = "<tool_response>{\"error\":\"unknown_tool\",\"detail\":\"no tool registered\"}</tool_response>";
        assert_eq!(
            chain_error(
                &[(TurnText::from(turn), String::new())],
                &TurnText::default()
            ),
            Some("unknown_tool: no tool registered".to_string()),
        );
        assert_eq!(
            chain_error(&[], &TurnText::from(turn)),
            Some("unknown_tool: no tool registered".to_string()),
            "the decode-side response is checked too",
        );
    }

    /// Rendering is deterministic — a unit's content key stands for the turns
    /// it shows, so the same unit must render the same bytes every time.
    #[test]
    fn rendering_is_byte_identical_on_repeat() {
        let d = workspace(&[("a/mod.rs", "//! One.\n//! Two.\nfn x() {}\n")]);
        let units = build_units(d.path(), &[("a/mod.rs", Language::Rust)]);
        let ctx = ctx_for(&d);
        assert_eq!(
            render_chain(&ctx, &units[0], &env()),
            render_chain(&ctx, &units[0], &env())
        );
    }
}
