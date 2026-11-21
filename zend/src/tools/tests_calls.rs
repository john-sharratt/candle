//! Tests for call extraction across each shape a model produces a call in.
//!
//! Each test drives `extract_tool_calls` with a shape a real model emitted and
//! asserts the calls that come back, their argument types and their order. A
//! call inside reasoning must stay undispatched, and a truncated function block
//! must yield no call rather than one whose missing arguments were invented.

use super::*;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    // ── The function-block syntax (Qwen3.5 / Qwen3.8) ───────────────────────
    //
    // These drive `extract_tool_calls` with the shape the stencil FORCES on that
    // lineage. Before the translation every one of them returned zero calls, and
    // zero calls reads as "the model produced a final answer" — so the turn
    // ended, no tool ran, and nothing logged a failure while the GUI drew a card
    // from the markers. The unit tests for the translation itself live in
    // `candle_conversation::stencil::function_block`; these assert the daemon's
    // end of it, against the real bundled catalog.

    /// The canonical forced shape, in a whole assistant turn — **and the typing
    /// that makes it dispatchable.** `tcp_session_open` declares `host` a string
    /// and `port` an integer, so `port` must come back as a number the tool can
    /// deserialize rather than the string `"443"`.
    #[test]
    fn extract_tool_calls_reads_a_function_block_call() {
        let text = "<tool_call>\n<function=tcp_session_open>\n\
                    <parameter=host>\nexample.com</parameter>\n\
                    <parameter=port>\n443</parameter>\n</function>\n</tool_call>";
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert_eq!(calls[0].name, "tcp_session_open");
        assert_eq!(calls[0].arguments["host"], serde_json::json!("example.com"));
        assert_eq!(calls[0].arguments["port"], serde_json::json!(443));
        assert!(
            calls[0].arguments["port"].is_number(),
            "a declared integer must not arrive as a string: {:?}",
            calls[0].arguments,
        );
    }

    /// A call with no arguments is a call. `datetime` takes none, and it is the
    /// tool the whole failure was first noticed on.
    #[test]
    fn extract_tool_calls_reads_a_function_block_with_no_arguments() {
        let text = "<tool_call>\n<function=datetime>\n</function>\n</tool_call>";
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert_eq!(calls[0].name, "datetime");
        assert_eq!(calls[0].arguments, serde_json::json!({}));
    }

    /// **The invariant the translation must not hole.** A function block written
    /// inside the reasoning block is deliberation, exactly as a JSON call there
    /// is — which is why the translation runs *after* the think-strip and not
    /// before it.
    #[test]
    fn extract_tool_calls_ignores_a_function_block_inside_think() {
        let text = "<think>\nI could call <function=datetime>\n</function> here.\n</think>\n\
                    On reflection, no.";
        assert!(
            extract_tool_calls(text).is_empty(),
            "a call inside the reasoning block was dispatched",
        );
    }

    /// The counterpart: the same block *after* the reasoning closes is the
    /// invocation, and must dispatch.
    #[test]
    fn extract_tool_calls_dispatches_a_function_block_after_think() {
        let text = "<think>\nThe clock, then.\n</think>\n\
                    <tool_call>\n<function=datetime>\n</function>\n</tool_call>";
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert_eq!(calls[0].name, "datetime");
    }

    /// Two calls in one turn, with prose in front of them. The prose matters:
    /// both of the translator's bail-out paths once emitted the text before a
    /// block twice, and that is invisible when a block starts at byte zero.
    #[test]
    fn extract_tool_calls_reads_two_function_blocks_after_prose() {
        let text = "Checking both.\n\
                    <tool_call>\n<function=datetime>\n</function>\n</tool_call>\n\
                    <tool_call>\n<function=tcp_session_open>\n\
                    <parameter=host>\nexample.com</parameter>\n\
                    <parameter=port>\n80</parameter>\n</function>\n</tool_call>";
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 2, "{calls:?}");
        assert_eq!(calls[0].name, "datetime");
        assert_eq!(calls[1].name, "tcp_session_open");
        assert_eq!(calls[1].arguments["port"], serde_json::json!(80));
    }

    /// A decode cut off mid-call dispatches nothing, rather than a call whose
    /// missing arguments were invented for it.
    #[test]
    fn extract_tool_calls_ignores_a_truncated_function_block() {
        let text = "<tool_call>\n<function=tcp_session_open>\n<parameter=host>\nexample.co";
        assert!(
            extract_tool_calls(text).is_empty(),
            "a truncated call was dispatched",
        );
    }

    #[test]
    fn extract_tool_calls_finds_well_formed_block() {
        let text = r#"Here is the result.
<tool_call>
{"name": "datetime", "arguments": {"timezone": "UTC"}}
</tool_call>
"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "datetime");
        assert_eq!(calls[0].arguments["timezone"], "UTC");
    }

    #[test]
    fn extract_tool_calls_finds_multiple_blocks() {
        let text = r#"
<tool_call>{"name": "datetime", "arguments": {}}</tool_call>
some text
<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>
"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "datetime");
        assert_eq!(calls[1].name, "calculator");
    }

    #[test]
    fn extract_tool_calls_skips_malformed() {
        let text = r#"
<tool_call>not json</tool_call>
<tool_call>{"name": "datetime", "arguments": {}}</tool_call>
<tool_call>{}</tool_call>
"#;
        let calls = extract_tool_calls(text);
        // Only the second block has a valid name.
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "datetime");
    }

    #[test]
    fn extract_tool_calls_handles_no_calls() {
        let text = "Just a normal response with no tools.";
        let calls = extract_tool_calls(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn extract_tool_calls_tolerates_missing_opener() {
        // Qwen3-A3B in `/no_think` mode often emits the JSON without
        // the leading `<tool_call>` tag — the close is reliable but
        // the open is not.  Make sure we still recover the call.
        let text = "<think>\n\n</think>\n\n{\"name\": \"calculator\", \
                    \"arguments\": {\"expression\": \"1+1\"}}\n</tool_call>";
        let calls = extract_tool_calls(text);
        assert_eq!(
            calls.len(),
            1,
            "expected the opener-elided tool call to be recovered, got {calls:?}",
        );
        assert_eq!(calls[0].name, "calculator");
        assert_eq!(calls[0].arguments["expression"], "1+1");
    }

    #[test]
    fn extract_tool_calls_mixed_well_formed_and_opener_elided() {
        // First call has both tags, second only has the closer — both
        // should be recovered, in order.
        let text = r#"
<tool_call>{"name": "datetime", "arguments": {}}</tool_call>
some text
{"name": "calculator", "arguments": {"expression": "2+2"}}
</tool_call>
"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 2, "got: {calls:?}");
        assert_eq!(calls[0].name, "datetime");
        assert_eq!(calls[1].name, "calculator");
    }

    #[test]
    fn extract_tool_calls_lenient_does_not_double_count_well_formed() {
        // A perfectly-formed `<tool_call>...</tool_call>` should
        // produce exactly one ToolCall — the lenient pass must not
        // re-match the same close-tag.
        let text = r#"<tool_call>{"name": "datetime", "arguments": {}}</tool_call>"#;
        let calls = extract_tool_calls(text);
        assert_eq!(
            calls.len(),
            1,
            "well-formed block double-counted: {calls:?}"
        );
    }

    #[test]
    fn extract_tool_calls_lenient_ignores_orphan_close_without_json() {
        // A stray `</tool_call>` with no JSON object before it must
        // not synthesize a tool call from prior text.
        let text = "just some narrative</tool_call>";
        let calls = extract_tool_calls(text);
        assert!(calls.is_empty(), "got phantom call: {calls:?}");
    }

    #[test]
    fn extract_tool_calls_recovers_bare_json_no_tags() {
        // Qwen3 sometimes emits the call as raw JSON with no tags at all.
        let text = r#"{"name": "web_search", "arguments": {"query": "q", "max_results": 5}}"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1, "bare JSON call not recovered: {calls:?}");
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].arguments["max_results"], 5);
    }

    #[test]
    fn extract_tool_calls_bare_json_with_preamble_and_nested_args() {
        // Preamble prose before a bare, multi-line call with nested arguments.
        let text = "I'll run that for you.\n{\n  \"name\": \"calculator\",\n  \"arguments\": {\"expression\": \"(1+2)*3\"}\n}\n";
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1, "got: {calls:?}");
        assert_eq!(calls[0].name, "calculator");
        assert_eq!(calls[0].arguments["expression"], "(1+2)*3");
    }

    #[test]
    fn extract_tool_calls_bare_json_resolves_alias() {
        // A bare call under a known alias is recovered (the registry gate
        // resolves it); the canonical name is settled later by `run_tool`.
        let text = r#"{"name": "file_create", "arguments": {"path": "a.txt", "content": "hi"}}"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1, "aliased bare call not recovered: {calls:?}");
        assert_eq!(calls[0].name, "file_create");
        assert!(registry::find(&calls[0].name).is_some());
    }

    #[test]
    fn extract_tool_calls_bare_json_accepts_the_tool_key() {
        // Verbatim shape from Qwen3.6-35B-A3B when told to emit a call: the
        // right tool under the key `"tool"` instead of `"name"`, with no tags.
        // Without the alias the entire call is silently dropped and the model
        // answers from memory instead.
        let text = "<think>\nI need the current time.\n</think>\n\n\
                    {\n  \"tool\": \"datetime\",\n  \"arguments\": {}\n}";
        let calls = extract_tool_calls(text);
        assert_eq!(
            calls.len(),
            1,
            "`tool`-keyed bare call not recovered: {calls:?}"
        );
        assert_eq!(calls[0].name, "datetime");
    }

    #[test]
    fn extract_tool_calls_bare_json_ignores_non_tool_and_fabricated_response() {
        // Prose JSON that doesn't name a real tool, and a fabricated tool
        // *response*, must not be mistaken for calls.
        let unknown = r#"{"name": "definitely_not_a_tool", "arguments": {}}"#;
        assert!(extract_tool_calls(unknown).is_empty());
        let fabricated = r#"{"error": "No active HTTP sessions found."}"#;
        assert!(extract_tool_calls(fabricated).is_empty());
        // A tool *definition* echo (top-level `description` + a `parameters`
        // schema) is not a call — even though `parameters` is now accepted as the
        // args key, the `description` gate rejects it.
        let definition = r#"{"name": "ping_icmp", "description": "ping a host", "parameters": {}}"#;
        assert!(extract_tool_calls(definition).is_empty());
    }

    #[test]
    fn extract_tool_calls_recovers_bare_parameters_keyed_call() {
        // The exact Qwen3-A3B degradation from the substrate: no `<tool_call>`
        // wrapper AND the schema key `parameters` in place of `arguments`. Recover
        // it as a real call, carrying the values.
        let text = r#"{"name":"datetime","parameters":{"timezone":"Australia/Sydney"}}"#;
        let calls = extract_tool_calls(text);
        assert_eq!(
            calls.len(),
            1,
            "bare parameters-keyed call not recovered: {calls:?}",
        );
        assert_eq!(calls[0].name, "datetime");
        assert_eq!(calls[0].arguments["timezone"], "Australia/Sydney");
    }

    #[test]
    fn extract_tool_calls_parameters_key_still_rejects_definition() {
        // A full definition echo carries `parameters` too, but its top-level
        // `description` must keep it from being mistaken for a call now that
        // `parameters` is an accepted args key.
        let def = r#"{"name":"datetime","description":"current time in a timezone","parameters":{"type":"object","properties":{"timezone":{"type":"string"}}}}"#;
        assert!(
            extract_tool_calls(def).is_empty(),
            "definition echo taken for a call",
        );
    }

    #[test]
    fn extract_tool_calls_tagged_call_keeps_parameters_values() {
        // Even inside a `<tool_call>` wrapper Qwen3 sometimes uses `parameters`;
        // the args must survive rather than dropping to null.
        let text =
            r#"<tool_call>{"name": "datetime", "parameters": {"timezone": "UTC"}}</tool_call>"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1, "got: {calls:?}");
        assert_eq!(calls[0].arguments["timezone"], "UTC");
    }

    #[test]
    fn extract_tool_calls_bare_pass_does_not_double_count_tagged() {
        // A fully-tagged call must yield exactly one ToolCall — the bare pass
        // must skip the object already claimed by the strict match.
        let text = r#"<tool_call>{"name": "datetime", "arguments": {}}</tool_call>"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1, "tagged call double-counted: {calls:?}");
        assert_eq!(calls[0].name, "datetime");
    }

    #[test]
    fn extract_tool_calls_ignores_call_inside_think_block() {
        // A tool call the model writes while reasoning is deliberation, not an
        // invocation — it must not be dispatched.
        let text = r#"<think>
Maybe I should call <tool_call>{"name": "datetime", "arguments": {}}</tool_call> to check.
</think>
The time doesn't matter here."#;
        let calls = extract_tool_calls(text);
        assert!(calls.is_empty(), "in-think call was dispatched: {calls:?}");
    }

    #[test]
    fn extract_tool_calls_dispatches_call_after_think_block() {
        // The model reasons, closes the block, then emits the real call.
        let text = r#"<think>
I'll check the time.
</think>
<tool_call>{"name": "datetime", "arguments": {"timezone": "UTC"}}</tool_call>"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1, "post-think call not dispatched: {calls:?}");
        assert_eq!(calls[0].name, "datetime");
    }

    #[test]
    fn extract_tool_calls_ignores_in_think_keeps_post_think() {
        // A call mid-thought is ignored; the one after `</think>` is kept.
        let text = r#"<think>
I could <tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call> but no.
</think>
<tool_call>{"name": "datetime", "arguments": {}}</tool_call>"#;
        let calls = extract_tool_calls(text);
        assert_eq!(
            calls.len(),
            1,
            "expected only the post-think call: {calls:?}"
        );
        assert_eq!(calls[0].name, "datetime");
    }

    #[test]
    fn extract_tool_calls_ignores_bare_json_call_inside_think() {
        // The bare-JSON recovery pass must also respect the think boundary.
        let text =
            "<think>\n{\"name\": \"datetime\", \"arguments\": {}}\n</think>\nNo tool needed.";
        let calls = extract_tool_calls(text);
        assert!(
            calls.is_empty(),
            "bare in-think call was dispatched: {calls:?}"
        );
    }
}
