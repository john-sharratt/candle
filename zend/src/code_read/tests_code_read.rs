//! Tests of the prompts, file keys and reading rules that a file's conversation
//! relies on.
//!
//! They pin that a file's first turn opens inside its call, that a read which
//! found no file fails the reading, and that an empty answer is no reading. The
//! prompt tests check that the file's length and call count are stated and that
//! its ranges are the ones the reads hold, and the reading test checks that one
//! round runs per page before the answer.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::tools::ToolCall;
    use candle_conversation::stencil::{
        compile, compile_tool_call_tree, parse_tools, TestVocab, ToolCallEnvelope,
    };
    use serde_json::json;
    use std::slice::from_ref;

    /// **A file read runs with thinking off in both halves, and its one tool.**
    /// The prompt's no-think switch and the effort dial agree with the
    /// `ThinkMode::Off` steering, so the prompt never invites a block the
    /// steering closes.
    #[test]
    fn a_file_read_runs_with_thinking_off_and_only_file_read() {
        let s = file_selection();
        assert_eq!(s.optional(NO_THINK_SELECTOR), Some(OptionalState::Present));
        assert_eq!(s.get("thinking_effort"), Some("off"));
        assert_eq!(
            s.optional(TOOLS_ENABLED_SELECTOR),
            Some(OptionalState::Present)
        );
        assert_eq!(s.get(FORCE_TOOL_SELECTOR), Some(FILE_READ_TOOL));
    }

    /// **A file's first turn begins inside its call**: the marker prefilled
    /// and the call loop armed, everything else the turn was given kept.
    #[test]
    fn the_opening_turn_begins_inside_the_file_read_call() {
        let tools = parse_tools(
            r#"[{"name":"file_read","params":[{"name":"path","type":"string","required":true}]}]"#,
        )
        .unwrap();
        let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
        let opening = OpeningCall {
            marker: "<tool_call>".into(),
            tree: Arc::new(compile(&spec, &TestVocab::new()).unwrap()),
        };
        let given = TurnOptions {
            max_tokens: Some(7),
            tags: vec!["code".into()],
            ..Default::default()
        };
        let opened = opening_options(given.clone(), &opening, None);
        assert_eq!(opened.assistant_prefill.as_deref(), Some("<tool_call>"));
        let grammar = opened.turn_grammar.as_ref().unwrap();
        assert!(Arc::ptr_eq(&grammar.tree, &opening.tree));
        assert!(grammar.held.is_none());
        assert_eq!(opened.max_tokens, Some(7));
        assert_eq!(opened.tags, ["code"]);

        // Held, the grammar carries the reading's own holds.
        let held = Arc::new(HeldValues::build(
            &reading_holds("candle/a.rs", &[(1, 3)]),
            &TestVocab::new(),
            &mut HeldRules::default(),
        ));
        let opened = opening_options(given, &opening, Some(&held));
        let grammar = opened.turn_grammar.as_ref().unwrap();
        assert!(Arc::ptr_eq(grammar.held.as_ref().unwrap(), &held));
    }

    /// **A read that found no file fails the reading**, naming what it asked
    /// for; reads that found their file, and other tools' errors, do not.
    #[test]
    fn a_file_read_not_found_fails_the_reading() {
        let result = |name: &str, path: &str, response: Value| ToolResult {
            call: ToolCall {
                name: name.into(),
                arguments: json!({"repo": "candle", "path": path, "lines": "1-214"}),
            },
            response,
        };
        let found = result(
            FILE_READ_TOOL,
            "candle-conversation/src/turn_layout.rs",
            json!({"parts": []}),
        );
        let missing = result(
            FILE_READ_TOOL,
            "src/turn_layout.rs",
            json!({"error": "not_found", "detail": "file not found: src/turn_layout.rs"}),
        );
        let other = result(
            "file_list",
            "src",
            json!({"error": "not_found", "detail": "x"}),
        );
        assert_eq!(read_not_found(from_ref(&found)), None);
        assert_eq!(
            read_not_found(&[other]),
            None,
            "only a file_read fails a reading"
        );
        assert_eq!(
            read_not_found(&[found, missing]).as_deref(),
            Some("candle:src/turn_layout.rs")
        );
    }

    /// **An answer of nothing is no reading**: a turn that ended at once, or
    /// one that only reasoned, keeps nothing a summary could be.
    #[test]
    fn an_empty_answer_is_no_reading() {
        assert!(produced_no_answer(""));
        assert!(produced_no_answer("<think>\nI will read it"));
        assert!(!produced_no_answer("ladder.rs builds the load ladder."));
    }

    /// A file job's key is the branch walk's key for the same file — the one
    /// the pass plans with and the fast path looks up.
    #[test]
    fn a_file_jobs_key_is_its_path_and_blob() {
        let job = FileJob {
            path: "candle/src/lib.rs".into(),
            blob: Oid::parse("ce013625030ba8dba906f756967f9e9ca394464a").unwrap(),
            language: Language::Rust,
            at: None,
            branches: Vec::new(),
        };
        assert_eq!(
            job.key(),
            "v6:candle/src/lib.rs@ce013625030ba8dba906f756967f9e9ca394464a"
        );
    }

    #[test]
    fn is_upload_path_matches_top_level_uploads_only() {
        // Top-level uploads/ (any case, since the win32 FS is case-insensitive).
        assert!(is_upload_path("uploads"));
        assert!(is_upload_path("uploads/notes.py"));
        assert!(is_upload_path("Uploads/notes.py"));
        assert!(is_upload_path("UPLOADS/a.rs"));
        // NOT a nested source dir, nor a lookalike.
        assert!(!is_upload_path("src/uploads/real.rs"));
        assert!(!is_upload_path("uploadsx/a.py"));
        assert!(!is_upload_path("docs/uploads.md"));
        assert!(!is_upload_path("src/main.rs"));
    }

    #[test]
    fn opening_prompt_names_the_path_and_the_one_tool() {
        let p = opening_prompt("candle/src/lib.rs", 120, &[(1, 120)]);
        assert!(
            p.starts_with("Read the entire contents of `src/lib.rs` in the `candle` repository")
        );
        assert!(p.contains(FILE_READ_TOOL));
    }

    /// **The prompt must state the file's length and the exact number of calls.**
    ///
    /// Told only the per-call cap, a model has to guess how many calls cover the
    /// file — and that guess is a decision it can spend its entire decode budget
    /// failing to reach. A README read did exactly that: it weighed whether the
    /// file was under the cap and whether a read past EOF would error until the
    /// budget cut it off mid-word, with no `file_read` call emitted, no coupling,
    /// no second turn and no summary. The arithmetic belongs in the prompt.
    ///
    /// The ranges are the file's own, so the boundaries are pinned as text.
    #[test]
    fn opening_prompt_states_the_length_and_the_call_count() {
        assert_eq!(
            PAGE_LINES, 200,
            "the expected text below is written for 200-line reads"
        );
        let one = opening_prompt("candle/src/small.rs", 200, &[(1, 200)]);
        assert!(
            one.contains(
                "The file is 200 lines long and reads in 1 range(s) of whole parts, at most \
                 200 lines each — so it takes exactly 1 file_read call(s), each passing one \
                 range as `lines`, ONE call per reply, in this order: lines=1-200."
            ),
            "{one:?}",
        );
        let two = opening_prompt("candle/src/mid.rs", 201, &[(1, 196), (197, 201)]);
        assert!(
            two.contains(
                "reads in 2 range(s) of whole parts, at most 200 lines each — so it takes \
                 exactly 2 file_read call(s), each passing one range as `lines`, ONE call per \
                 reply, in this order: lines=1-196, lines=197-201."
            ),
            "{two:?}",
        );
        // An empty file still asks for one call rather than zero.
        let empty = opening_prompt("candle/src/empty.rs", 0, &[]);
        assert!(
            empty.contains("exactly 1 file_read call(s)") && empty.contains("order: lines=1."),
            "never zero calls: {empty:?}",
        );
    }

    /// **The ranges the prompt names are the ones the reads hold**: one call per
    /// range, whole parts each, and together every line.
    #[test]
    fn the_ranges_are_the_files_reads() {
        let text: String = (0..150).map(|i| format!("p{i} a\np{i} b\n\n")).collect();
        assert_eq!(windows("candle/a.md", &text), vec![(1, 300), (301, 450)]);
        let p = opening_prompt("candle/a.md", 450, &windows("candle/a.md", &text));
        assert!(p.contains("lines=1-300, lines=301-450."), "{p:?}");
    }

    /// The measured failure this prompt exists to close: left as "read as much
    /// as you need", a model sometimes skipped `file_read` entirely and
    /// guessed a summary from the filename alone. The prompt must require
    /// reading the file, not merely permit it.
    #[test]
    fn opening_prompt_requires_reading_the_whole_file() {
        let p = opening_prompt("candle/CHANGELOG.md", 40, &[(1, 40)]);
        assert!(
            p.contains("entire") || p.contains("whole"),
            "must ask for the whole file, not an unspecified amount: {p:?}",
        );
        assert!(
            p.contains("only this file"),
            "must scope the read to this file alone: {p:?}",
        );
    }

    /// **One read per reply, whatever the file's length.** Each tool round of a
    /// reading is one page, which the projection places or drops at that page's
    /// cost (`docs/hit_driven_projection.md` §8.2); the grammar holds a reply to
    /// [`READ_CALLS_PER_REPLY`] calls and the prompt asks for no more.
    #[test]
    fn opening_prompt_asks_for_one_read_per_reply() {
        assert_eq!(READ_CALLS_PER_REPLY, 1);
        let ranges: Vec<(usize, usize)> = (0..20).map(|k| (k * 200 + 1, k * 200 + 200)).collect();
        let p = opening_prompt("candle/src/big.rs", 4000, &ranges);
        assert!(
            p.contains("exactly 20 file_read call(s), each passing one range as `lines`, ONE call per reply"),
            "{p:?}"
        );
        assert!(!p.contains("SAME reply"), "{p:?}");
    }

    /// **A reading runs one round a page** and then answers, so a file of any
    /// length reads to its end and no further.
    #[test]
    fn a_reading_runs_one_round_a_page() {
        assert_eq!(read_rounds(0), 1, "an empty file is still one read");
        assert_eq!(read_rounds(1), 1);
        assert_eq!(read_rounds(37), 37);
    }
}
