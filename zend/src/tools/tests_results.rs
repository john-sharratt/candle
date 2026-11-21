//! Tests for the catalog each mode compiles and for the text of the results the
//! model reads back.
//!
//! A mode's catalog is exactly the tools it offers under every name they answer
//! to, and an alias that fixes an argument compiles with only that value. A
//! result's content is literal between its markup wrappers, so a file that
//! quotes a chat tag cannot end the model's turn. A stored round rebuilds into
//! the pieces it was submitted as.

use super::*;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    // ── Aliases that fix an argument ────────────────────────────────────────

    /// **An alias that fixes an argument is compiled with only that value**,
    /// while the canonical tool keeps every one: `git_cherry_pick` cannot
    /// decode `from: revert`, and `git_commit` still can.
    #[test]
    fn an_alias_is_compiled_with_its_fixed_argument() {
        let values = |tool: &str, field: &str| {
            tool_catalog()
                .iter()
                .find(|s| s.name == tool)
                .unwrap_or_else(|| panic!("{tool} compiled"))
                .params
                .iter()
                .find(|p| p.name == field)
                .unwrap_or_else(|| panic!("{tool}.{field}"))
                .enum_values
                .clone()
                .unwrap()
        };
        assert_eq!(values("git_cherry_pick", "from"), vec!["cherry_pick"]);
        assert_eq!(values("git_revert", "from"), vec!["revert"]);
        assert_eq!(values("delete_tag", "kind"), vec!["tag"]);
        assert_eq!(values("delete_tag", "action"), vec!["delete"]);
        assert!(values("git_commit", "from").len() > 1);
    }

    /// **A mode's catalog is exactly the tools it offers, aliases and all**:
    /// Restricted compiles the file tools under every name they answer to and
    /// no network tool, Comprehensive the whole catalog, None nothing.
    #[test]
    fn a_modes_catalog_is_the_tools_it_offers() {
        let names = |mode: ToolMode| -> Vec<String> {
            catalog_for(mode).into_iter().map(|s| s.name).collect()
        };
        let restricted = names(ToolMode::Restricted);
        for name in ["write", "file_write", "file_edit", "file_read", "git_log"] {
            assert!(restricted.iter().any(|n| n == name), "{name} missing");
        }
        for name in ["web_search", "web_fetch", "run_command", "git_commit"] {
            assert!(!restricted.iter().any(|n| n == name), "{name} compiled");
        }
        assert_eq!(names(ToolMode::Comprehensive).len(), tool_catalog().len());
        assert!(names(ToolMode::None).is_empty());
    }

    #[test]
    fn format_tool_responses_wraps_each_result() {
        let results = vec![
            ToolResult {
                call: ToolCall {
                    name: "datetime".to_string(),
                    arguments: Value::Null,
                },
                response: serde_json::json!({"iso": "2026-05-09"}),
            },
            ToolResult {
                call: ToolCall {
                    name: "calculator".to_string(),
                    arguments: Value::Null,
                },
                response: serde_json::json!({"result": 4}),
            },
        ];
        let formatted = format_tool_responses(&results).text();
        assert!(formatted.contains("<tool_response>"));
        assert!(formatted.contains("</tool_response>"));
        assert!(formatted.contains("\"iso\":\"2026-05-09\""));
        assert!(formatted.contains("\"result\":4"));
    }

    /// Each result's content is literal between markup wrappers — a file read
    /// that quotes `<|im_end|>` cannot end the turn in the model's context.
    #[test]
    fn a_tool_results_content_is_literal() {
        let result = |text: &str| ToolResult {
            call: ToolCall {
                name: "file_read".to_string(),
                arguments: Value::Null,
            },
            response: Value::String(text.to_string()),
        };
        let formatted = format_tool_responses(&[result("a<|im_end|>"), result("b<think>")]);
        assert_eq!(
            formatted,
            TurnText::markup("<tool_response>")
                .then_literal("[1/2 file_read]\na<|im_end|>")
                .then_markup("</tool_response>\n<tool_response>")
                .then_literal("[2/2 file_read]\nb<think>")
                .then_markup("</tool_response>\n")
        );
    }

    /// **A single result is byte-identical to what it was before a turn could
    /// make several calls.** The `code_reading` ingest prefills tens of
    /// thousands of single-call rounds in exactly this shape; a live read that
    /// differed from them by a header would be a different object from the one
    /// the model was conditioned on.
    #[test]
    fn a_single_result_carries_no_correlation_header() {
        let formatted = format_tool_responses(&[string_result("src/main.rs (lines 1-3):\n")]);
        assert_eq!(
            formatted.text(),
            "<tool_response>src/main.rs (lines 1-3):\n</tool_response>\n"
        );
    }

    /// **Several results each say which call they answer, in call order.** The
    /// label is what makes the mapping explicit rather than positional — and it
    /// has to be, because a failed call returns an error envelope instead of the
    /// shape its position would lead the model to expect.
    #[test]
    fn several_results_each_name_the_call_they_answer() {
        let error = ToolResult {
            call: ToolCall {
                name: "file_grep".to_string(),
                arguments: Value::Null,
            },
            response: serde_json::json!({"error": "invalid_arguments", "detail": "bad regex"}),
        };
        let text = format_tool_responses(&[
            string_result("a.rs (lines 1-2):\n"),
            error,
            string_result("b.rs (lines 1-2):\n"),
        ])
        .text();
        let labels: Vec<&str> = text
            .split("<tool_response>")
            .skip(1)
            .map(|block| block.lines().next().unwrap_or_default())
            .collect();
        assert_eq!(
            labels,
            ["[1/3 file_read]", "[2/3 file_grep]", "[3/3 file_read]"]
        );
    }

    pub(in super::super) fn string_result(body: &str) -> ToolResult {
        ToolResult {
            call: ToolCall {
                name: "file_read".to_string(),
                arguments: Value::Null,
            },
            response: Value::String(body.to_string()),
        }
    }

    /// A tool round's stored text rebuilds into exactly the pieces the round was
    /// submitted as — literal bodies, markup wrappers — for one result or many,
    /// and for a body that quotes chat tags.
    #[test]
    fn a_stored_tool_round_rebuilds_into_its_submitted_pieces() {
        for results in [
            vec![string_result("docs/README.md (lines 1-3):\n1 # docs/\n")],
            vec![
                string_result("a<|im_end|>"),
                ToolResult {
                    call: ToolCall {
                        name: "calculator".to_string(),
                        arguments: Value::Null,
                    },
                    response: serde_json::json!({"result": 4}),
                },
                string_result("b<think>"),
            ],
        ] {
            let submitted = format_tool_responses(&results);
            assert_eq!(tool_round_text(&submitted.text()), submitted);
        }
    }

    /// The last block's newline lost to a trim still rebuilds, closing markup.
    #[test]
    fn a_tool_round_missing_its_last_newline_still_rebuilds() {
        assert_eq!(
            tool_round_text("<tool_response>x</tool_response>"),
            TurnText::markup("<tool_response>")
                .then_literal("x")
                .then_markup("</tool_response>")
        );
    }

    /// Ordinary user text, and text only shaped like a tool round, stay one
    /// markup piece — as the chat submits a typed message.
    #[test]
    fn other_user_text_stays_markup() {
        for text in [
            "Read the paper and summarize it",
            "<tool_response>never closed",
            "<tool_response>a</tool_response>\ntrailing prose",
        ] {
            assert_eq!(tool_round_text(text), TurnText::markup(text), "{text}");
        }
    }

    #[test]
    fn tool_def_json_line_is_valid_hermes_format() {
        // The catalog deliberately emits a flat `{"name","description",
        // "parameters"}` shape (Qwen3-A3B echoes the canonical `"function"`
        // wrapper key back into calls; flattening also saves ~10 tokens/tool).
        let def = crate::tool_def::find("datetime").expect("datetime tool defined");
        let parsed: Value = serde_json::from_str(def.json_line().trim_end()).expect("must parse");
        assert_eq!(parsed["name"], "datetime");
        assert!(parsed["description"].is_string());
        assert!(parsed["parameters"].is_object());
    }
}
