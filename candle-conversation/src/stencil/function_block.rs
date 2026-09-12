//! The inverse of [`CallStyle::FunctionBlock`] — read a function-block call
//! back as JSON.
//!
//! [`ToolCallEnvelope::qwen35`] writes a call as a nested element per argument
//! with raw, unescaped values:
//!
//! ```text
//! <tool_call>
//! <function=file_read>
//! <parameter=path>
//! src/main.rs</parameter>
//! <parameter=start_line>
//! 1</parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! This rewrites the `<function=…>…</function>` span into the canonical call
//! object — `{"name":…,"arguments":{…}}` — and leaves every other byte of the
//! text alone, `<tool_call>` wrapper included.
//!
//! # Why translate instead of teaching the consumers a second syntax
//!
//! Two consumers read a completed answer, and both of them already understand
//! JSON: the daemon's extractor (three passes over `<tool_call>{json}</tool_call>`
//! and its degraded variants) and the web GUI, whose tool card captures the body
//! between the markers and parses it. Neither knows this shape. Teaching both
//! would be two new parsers with the same job, free to disagree — and the
//! disagreement surfaces as one of them dispatching a call the other renders as
//! an error.
//!
//! Translating instead means the shape exists in exactly two places: the
//! envelope that writes it, and this file that reads it. They sit in the same
//! module for that reason.
//!
//! **This is the gap that silently broke tool calling on the Qwen3.5/3.8
//! lineage.** The preset declares `DialectType::Qwen35`, whose `call_style` is
//! `FunctionBlock`, so the stencil *forced* every call into a shape nothing
//! downstream could read: the model chose the right tool, the grammar guaranteed
//! a well-formed call, the GUI drew a card because the `<tool_call>` markers were
//! there — and the extractor, looking for a `{` straight after the opening
//! marker, returned zero calls. Zero calls means "the model produced a final
//! answer", so the turn ended, no tool ran, and nothing anywhere logged a
//! failure. A whole class of request was inert with every individual part
//! working as written.
//!
//! # The values are raw, and the catalog is what types them
//!
//! A `<parameter>` body is unescaped text that may hold quotes, braces and
//! newlines — that is the point of the syntax, and `serde_json` does the escaping
//! on the way out, so a value survives intact rather than being cut at its first
//! quote.
//!
//! What it cannot do is guess whether `1` is the number one or the string `"1"`.
//! So the catalog decides, via the same [`ToolSpec`] list the grammar was
//! compiled from: a parameter declared `integer` parses as one, and a parameter
//! declared `string` stays a string **even when its text is all digits**. A
//! looks-like-a-number heuristic gets that second case wrong, and gets it wrong
//! on exactly the arguments a coding assistant passes around — a line number is
//! a number, but a file named `2024` is not.
//!
//! [`CallStyle::FunctionBlock`]: candle_transformers::models::dialect::CallStyle::FunctionBlock
//! [`ToolCallEnvelope::qwen35`]: super::ToolCallEnvelope::qwen35

use serde_json::{Map, Number, Value};

use super::tool_call::{ParamType, ToolSpec};

/// Opens the call and precedes its name. Mirrors `ToolCallEnvelope::open`'s tail.
const FN_OPEN: &str = "<function=";
/// Closes the call. Mirrors `ToolCallEnvelope::close`'s head.
const FN_CLOSE: &str = "</function>";
/// Opens one argument and precedes its name. Mirrors `param_open`.
const PARAM_OPEN: &str = "<parameter=";
/// Ends one argument's value. Mirrors `param_close` — and, as there, it is the
/// tag *alone*: the grammar's value span consumes it, so no layout is implied
/// around it and none may be required here.
const PARAM_CLOSE: &str = "</parameter>";

/// Rewrite every function-block call in `text` as a canonical JSON call object,
/// typing each argument through `tools`.
///
/// Returns `None` when `text` holds no function block at all, so the ordinary
/// path costs one substring search and allocates nothing. Surrounding text —
/// narration, reasoning blocks, the `<tool_call>` markers — is copied through
/// byte for byte.
///
/// A tool absent from `tools` still translates, with its arguments left as
/// strings: whether an unknown name is a call at all is the caller's question to
/// answer (the daemon's extractor gates the bare-object pass on its registry),
/// and dropping the block here would take the evidence away from whoever has to
/// answer it.
pub fn function_blocks_to_json(text: &str, tools: &[ToolSpec]) -> Option<String> {
    if !text.contains(FN_OPEN) {
        return None;
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(FN_OPEN) {
        // Anything before the block is the model's prose and is kept: an answer
        // that explained itself before calling still said it.
        out.push_str(&rest[..at]);
        let after = &rest[at + FN_OPEN.len()..];
        // **An unterminated block is left as text rather than guessed at.** A
        // truncated decode is prose, not a call whose arguments we invent — and
        // a half-call dispatched is worse than a half-call shown.
        // `rest[..at]` is already out, so only the block and what follows it are
        // owed — pushing `rest` whole here emitted the preceding prose twice.
        let (Some(name_end), Some(body_end)) = (after.find('>'), after.find(FN_CLOSE)) else {
            out.push_str(&rest[at..]);
            return Some(out);
        };
        if name_end > body_end {
            out.push_str(&rest[at..]);
            return Some(out);
        }
        let name = after[..name_end].trim();
        let body = &after[name_end + 1..body_end];
        let spec = tools.iter().find(|t| t.name == name);

        let mut args = Map::new();
        let mut scan = body;
        while let Some(p) = scan.find(PARAM_OPEN) {
            let tail = &scan[p + PARAM_OPEN.len()..];
            let Some(key_end) = tail.find('>') else {
                break;
            };
            // **`</function>` closes an open parameter.** The body is already
            // bounded by it, so a value with no `</parameter>` of its own runs to
            // the end of the body instead of being discarded.
            //
            // Not tolerance for sloppiness — it is what the grammar produces when
            // a value span ends on an intercepted EOS. The closing tag is
            // consumed by the span's terminator rather than emitted by the tree,
            // so a span that ends any other way never writes one.
            let val_end = tail.find(PARAM_CLOSE).unwrap_or(tail.len());
            if key_end > val_end {
                break;
            }
            let key = tail[..key_end].trim();
            // The element's own newlines are framing, not content: the grammar
            // writes `<parameter=k>\n` before the value. Stripping exactly one
            // leading and one trailing newline keeps a value that deliberately
            // ends in a blank line, which `trim` would eat.
            let raw = &tail[key_end + 1..val_end];
            let raw = raw.strip_prefix('\n').unwrap_or(raw);
            let raw = raw.strip_suffix('\n').unwrap_or(raw);
            if !key.is_empty() {
                args.insert(key.to_string(), typed_value(raw, param_type(spec, key)));
            }
            scan = &tail[(val_end + PARAM_CLOSE.len()).min(tail.len())..];
        }

        let mut obj = Map::new();
        obj.insert("name".to_string(), Value::String(name.to_string()));
        obj.insert("arguments".to_string(), Value::Object(args));
        out.push_str(&Value::Object(obj).to_string());
        rest = &after[body_end + FN_CLOSE.len()..];
    }
    out.push_str(rest);
    Some(out)
}

/// The declared type of `key` on `spec`, defaulting to a string.
///
/// An unknown tool or an argument the catalog does not declare falls back to
/// `String`, which is the shape that loses nothing: the raw text is preserved
/// exactly, and a consumer that knows better can still read it.
fn param_type(spec: Option<&ToolSpec>, key: &str) -> ParamType {
    spec.and_then(|s| s.params.iter().find(|p| p.name == key))
        .map(|p| p.ty)
        .unwrap_or(ParamType::String)
}

/// One raw parameter body as the JSON value its declared type calls for.
///
/// A value that does not parse as its declared type is kept as a string rather
/// than dropped or defaulted. The grammar constrains the decode, so this should
/// not happen — and when it does, the string carries what the model actually
/// wrote to whoever is diagnosing it, where a `0` or a missing key would not.
fn typed_value(raw: &str, ty: ParamType) -> Value {
    let as_string = || Value::String(raw.to_string());
    match ty {
        ParamType::String => as_string(),
        ParamType::Integer => raw
            .trim()
            .parse::<i64>()
            .map(Value::from)
            .unwrap_or_else(|_| as_string()),
        ParamType::Number => raw
            .trim()
            .parse::<f64>()
            .ok()
            .and_then(Number::from_f64)
            .map(Value::Number)
            .unwrap_or_else(as_string),
        ParamType::Boolean => match raw.trim() {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => as_string(),
        },
        // Structured arguments arrive as JSON text already — the grammar emits
        // them through `Terminator::JsonValue`, which validates the structure.
        ParamType::Array | ParamType::Object => {
            serde_json::from_str(raw.trim()).unwrap_or_else(|_| as_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stencil::{parse_tools, ToolCallEnvelope};

    /// The catalog the tests type against — one tool per parameter shape, so a
    /// coercion can be asserted without inventing a schema per test.
    fn catalog() -> Vec<ToolSpec> {
        parse_tools(
            r#"[
              {"name":"file_read","params":[
                {"name":"path","type":"string","required":true},
                {"name":"start_line","type":"integer","required":false}]},
              {"name":"datetime","params":[]},
              {"name":"say","params":[
                {"name":"to","type":"string","required":true},
                {"name":"words","type":"string","required":true}]},
              {"name":"scale","params":[
                {"name":"factor","type":"number","required":true}]},
              {"name":"toggle","params":[
                {"name":"on","type":"boolean","required":true}]},
              {"name":"bulk","params":[
                {"name":"items","type":"array","required":true},
                {"name":"opts","type":"object","required":false}]},
              {"name":"named","params":[
                {"name":"label","type":"string","required":true}]}
            ]"#,
        )
        .unwrap()
    }

    /// Parse the single call object out of a translated string.
    fn call_of(s: &str) -> Value {
        let start = s.find('{').expect("a translated call contains an object");
        let end = s.rfind('}').expect("a translated call contains an object");
        serde_json::from_str(&s[start..=end]).expect("the translation emits valid JSON")
    }

    // ── the ordinary path ───────────────────────────────────────────────────

    /// **No block, no work.** The overwhelmingly common answer has no call in
    /// it, and `None` is what lets the caller skip the whole path rather than
    /// copy the string to learn nothing changed.
    #[test]
    fn text_with_no_function_block_translates_to_nothing() {
        assert!(function_blocks_to_json("just prose", &catalog()).is_none());
        assert!(function_blocks_to_json("", &catalog()).is_none());
        // A JSON call already in the canonical shape is somebody else's job.
        assert!(
            function_blocks_to_json(r#"<tool_call>{"name":"datetime"}</tool_call>"#, &catalog())
                .is_none()
        );
    }

    /// The shape the grammar actually emits, end to end — the literal string
    /// `stencil::driver`'s own round-trip test drives through the compiled tree.
    #[test]
    fn the_canonical_emitted_call_becomes_a_canonical_json_call() {
        let emitted = "<tool_call>\n<function=say>\n<parameter=to>\nMira</parameter>\
                       \n<parameter=words>\nyou take the order now.</parameter>\
                       \n</function>\n</tool_call>";
        let out = function_blocks_to_json(emitted, &catalog()).expect("a block is present");
        // The wrapper survives untouched, which is what the GUI keys on and what
        // the extractor's strict pass requires.
        assert!(out.starts_with("<tool_call>\n"), "{out}");
        assert!(out.ends_with("\n</tool_call>"), "{out}");
        assert_eq!(
            call_of(&out),
            serde_json::json!({
                "name": "say",
                "arguments": {"to": "Mira", "words": "you take the order now."}
            }),
        );
    }

    /// A call with no arguments is a call, not a malformed one — `datetime`
    /// takes none, and it is the tool the failure was first noticed on.
    #[test]
    fn a_call_with_no_parameters_gets_an_empty_arguments_object() {
        let out = function_blocks_to_json(
            "<tool_call>\n<function=datetime>\n</function>\n</tool_call>",
            &catalog(),
        )
        .expect("a block is present");
        assert_eq!(
            call_of(&out),
            serde_json::json!({"name": "datetime", "arguments": {}}),
        );
    }

    /// The translated body is what the daemon's strict pass needs: a JSON object
    /// separated from the markers by whitespace only.
    ///
    /// Asserted as that property rather than by re-running the daemon's own
    /// regex. A copy of the pattern here would be a second statement of the same
    /// rule, free to drift from the one that actually gates dispatch — and the
    /// property is what the rule is *for*.
    #[test]
    fn the_translated_body_is_a_bare_json_object_between_the_markers() {
        let out = function_blocks_to_json(
            "<tool_call>\n<function=datetime>\n</function>\n</tool_call>",
            &catalog(),
        )
        .unwrap();
        let start = out.find("<tool_call>").expect("the opening marker survives") + "<tool_call>".len();
        let end = out.find("</tool_call>").expect("the closing marker survives");
        let body = out[start..end].trim();
        assert!(body.starts_with('{'), "body must open on the object: {out}");
        assert!(body.ends_with('}'), "body must close on the object: {out}");
        let json: Value = serde_json::from_str(body).expect("the body parses as JSON");
        assert_eq!(json["name"], "datetime");
        assert_eq!(json["arguments"], serde_json::json!({}));
    }

    // ── the catalog types the values ────────────────────────────────────────

    #[test]
    fn an_integer_parameter_comes_back_as_a_number() {
        let out = function_blocks_to_json(
            "<function=file_read>\n<parameter=path>\nsrc/main.rs</parameter>\
             \n<parameter=start_line>\n42</parameter>\n</function>",
            &catalog(),
        )
        .unwrap();
        let call = call_of(&out);
        assert_eq!(call["arguments"]["start_line"], serde_json::json!(42));
        assert!(call["arguments"]["start_line"].is_number());
    }

    /// **The case a digit-sniffing heuristic gets wrong.** `path` is declared a
    /// string, so a file called `2024` stays the string `"2024"` — the catalog
    /// is what knows, and this is why the translator takes it.
    #[test]
    fn a_string_parameter_whose_text_is_all_digits_stays_a_string() {
        let out = function_blocks_to_json(
            "<function=file_read>\n<parameter=path>\n2024</parameter>\n</function>",
            &catalog(),
        )
        .unwrap();
        let call = call_of(&out);
        assert_eq!(call["arguments"]["path"], serde_json::json!("2024"));
        assert!(call["arguments"]["path"].is_string());
    }

    #[test]
    fn a_number_parameter_parses_as_floating_point() {
        let out = function_blocks_to_json(
            "<function=scale>\n<parameter=factor>\n1.5</parameter>\n</function>",
            &catalog(),
        )
        .unwrap();
        assert_eq!(call_of(&out)["arguments"]["factor"], serde_json::json!(1.5));
    }

    #[test]
    fn a_boolean_parameter_parses_both_ways() {
        for (text, want) in [("true", true), ("false", false)] {
            let out = function_blocks_to_json(
                &format!("<function=toggle>\n<parameter=on>\n{text}</parameter>\n</function>"),
                &catalog(),
            )
            .unwrap();
            assert_eq!(
                call_of(&out)["arguments"]["on"],
                serde_json::json!(want),
                "{text}"
            );
        }
    }

    #[test]
    fn array_and_object_parameters_parse_as_structure() {
        let out = function_blocks_to_json(
            "<function=bulk>\n<parameter=items>\n[1, 2, 3]</parameter>\
             \n<parameter=opts>\n{\"deep\": true}</parameter>\n</function>",
            &catalog(),
        )
        .unwrap();
        let call = call_of(&out);
        assert_eq!(call["arguments"]["items"], serde_json::json!([1, 2, 3]));
        assert_eq!(call["arguments"]["opts"], serde_json::json!({"deep": true}));
    }

    /// A value that contradicts its declared type is kept verbatim rather than
    /// dropped or defaulted: the grammar should have prevented it, and when
    /// something has gone wrong the raw text is the evidence.
    #[test]
    fn a_value_that_does_not_parse_as_its_type_survives_as_a_string() {
        for (tool, param, text) in [
            ("file_read", "start_line", "not-a-number"),
            ("scale", "factor", "huge"),
            ("toggle", "on", "yes"),
            ("bulk", "items", "1, 2, 3"),
        ] {
            let out = function_blocks_to_json(
                &format!("<function={tool}>\n<parameter={param}>\n{text}</parameter>\n</function>"),
                &catalog(),
            )
            .unwrap();
            assert_eq!(
                call_of(&out)["arguments"][param],
                serde_json::json!(text),
                "{tool}.{param}",
            );
        }
    }

    #[test]
    fn an_unknown_tool_translates_with_string_arguments_and_keeps_its_name() {
        let out = function_blocks_to_json(
            "<function=not_a_tool>\n<parameter=x>\n7</parameter>\n</function>",
            &catalog(),
        )
        .unwrap();
        let call = call_of(&out);
        assert_eq!(call["name"], "not_a_tool");
        assert_eq!(call["arguments"]["x"], serde_json::json!("7"));
    }

    #[test]
    fn an_undeclared_parameter_is_kept_as_a_string() {
        let out = function_blocks_to_json(
            "<function=datetime>\n<parameter=surprise>\n12</parameter>\n</function>",
            &catalog(),
        )
        .unwrap();
        assert_eq!(
            call_of(&out)["arguments"]["surprise"],
            serde_json::json!("12"),
        );
    }

    // ── raw values stay raw ─────────────────────────────────────────────────

    /// The whole reason the syntax exists: a value needs no escaping on the
    /// wire, and `serde_json` escapes it on the way out rather than the value
    /// being cut at its first quote.
    #[test]
    fn a_value_containing_quotes_is_escaped_not_truncated() {
        let out = function_blocks_to_json(
            "<function=say>\n<parameter=to>\nMira</parameter>\
             \n<parameter=words>\nshe said \"no\" twice</parameter>\n</function>",
            &catalog(),
        )
        .unwrap();
        assert!(out.contains(r#"\"no\""#), "{out}");
        assert_eq!(
            call_of(&out)["arguments"]["words"],
            serde_json::json!("she said \"no\" twice"),
        );
    }

    #[test]
    fn a_multi_line_value_keeps_its_interior_newlines() {
        let out = function_blocks_to_json(
            "<function=say>\n<parameter=words>\nfirst\nsecond\nthird</parameter>\n</function>",
            &catalog(),
        )
        .unwrap();
        assert_eq!(
            call_of(&out)["arguments"]["words"],
            serde_json::json!("first\nsecond\nthird"),
        );
    }

    /// Exactly one framing newline comes off each end, so a value that
    /// deliberately ends in a blank line keeps it — `trim` would eat it.
    #[test]
    fn only_the_framing_newlines_are_stripped() {
        let out = function_blocks_to_json(
            "<function=say>\n<parameter=words>\nends in a blank line\n\n</parameter>\n</function>",
            &catalog(),
        )
        .unwrap();
        assert_eq!(
            call_of(&out)["arguments"]["words"],
            serde_json::json!("ends in a blank line\n"),
        );
    }

    /// The grammar writes no newline before `</parameter>` — that was a
    /// deliberate fix, because a delimiter has to be something the model either
    /// wrote or did not, and layout is not that. Both spellings read back the
    /// same.
    #[test]
    fn a_value_closing_with_or_without_a_newline_reads_the_same() {
        let tight = function_blocks_to_json(
            "<function=say>\n<parameter=words>\nno newline</parameter>\n</function>",
            &catalog(),
        )
        .unwrap();
        let loose = function_blocks_to_json(
            "<function=say>\n<parameter=words>\nno newline\n</parameter>\n</function>",
            &catalog(),
        )
        .unwrap();
        assert_eq!(
            call_of(&tight)["arguments"]["words"],
            serde_json::json!("no newline"),
        );
        assert_eq!(call_of(&tight), call_of(&loose));
    }

    #[test]
    fn an_empty_value_is_an_empty_string() {
        let out = function_blocks_to_json(
            "<function=say>\n<parameter=words>\n</parameter>\n</function>",
            &catalog(),
        )
        .unwrap();
        assert_eq!(call_of(&out)["arguments"]["words"], serde_json::json!(""));
    }

    /// Braces in a value must not be mistaken for JSON structure — the value is
    /// text until `serde_json` quotes it.
    #[test]
    fn a_value_containing_braces_and_angle_brackets_is_still_text() {
        let out = function_blocks_to_json(
            "<function=say>\n<parameter=words>\nfn main() { if a < b { ok } }</parameter>\
             \n</function>",
            &catalog(),
        )
        .unwrap();
        assert_eq!(
            call_of(&out)["arguments"]["words"],
            serde_json::json!("fn main() { if a < b { ok } }"),
        );
    }

    // ── the shapes a truncated decode produces ──────────────────────────────

    /// **`</function>` closes an open parameter.** The grammar's value span
    /// consumes its own closing tag, so a span ended by an intercepted EOS never
    /// writes one — and dropping the value there cost the whole call.
    #[test]
    fn a_parameter_with_no_close_runs_to_the_end_of_the_block() {
        let out = function_blocks_to_json(
            "<function=say>\n<parameter=words>\ncut short here\n</function>",
            &catalog(),
        )
        .unwrap();
        assert_eq!(
            call_of(&out)["arguments"]["words"],
            serde_json::json!("cut short here"),
        );
    }

    /// An unterminated *block* is prose. Nothing is invented from a decode that
    /// stopped mid-call, and the text is handed on exactly as it arrived so
    /// whoever is reading it can see what happened.
    #[test]
    fn an_unterminated_block_is_left_as_text() {
        let truncated = "<tool_call>\n<function=say>\n<parameter=words>\nstill typing";
        assert_eq!(
            function_blocks_to_json(truncated, &catalog()).as_deref(),
            Some(truncated),
        );
    }

    /// A `<function=` with no `>` at all cannot even name a tool.
    #[test]
    fn a_block_with_no_name_terminator_is_left_as_text() {
        let broken = "<function=say";
        assert_eq!(
            function_blocks_to_json(broken, &catalog()).as_deref(),
            Some(broken),
        );
    }

    /// A name that runs past the closing tag — `<function=say</function>more>` —
    /// is garbled rather than merely truncated, and takes the same route out:
    /// left as text, nothing invented.
    ///
    /// **Pinned with prose in front of it on purpose.** Both bail-outs used to
    /// push the text before the block and then push the whole remainder again,
    /// duplicating that prose — and the fault is invisible when the block starts
    /// at byte zero, which is why the sibling test above passed throughout.
    #[test]
    fn a_name_running_past_the_closing_tag_is_left_as_text_without_duplicating_prose() {
        let garbled = "Let me check.\n<function=say</function>more>";
        assert_eq!(
            function_blocks_to_json(garbled, &catalog()).as_deref(),
            Some(garbled),
        );
    }

    // ── more than one call, and prose around them ──────────────────────────

    #[test]
    fn two_calls_in_one_turn_both_translate() {
        let out = function_blocks_to_json(
            "<tool_call>\n<function=datetime>\n</function>\n</tool_call>\
             \n<tool_call>\n<function=file_read>\n<parameter=path>\nsrc/lib.rs</parameter>\
             \n</function>\n</tool_call>",
            &catalog(),
        )
        .unwrap();
        assert_eq!(out.matches("<tool_call>").count(), 2, "{out}");
        assert!(!out.contains("<function="), "{out}");
        assert!(out.contains(r#""name":"datetime""#), "{out}");
        assert!(out.contains(r#""name":"file_read""#), "{out}");
    }

    #[test]
    fn prose_before_and_after_a_call_is_preserved() {
        let out = function_blocks_to_json(
            "Let me check.\n<function=datetime>\n</function>\nThat should do it.",
            &catalog(),
        )
        .unwrap();
        assert!(out.starts_with("Let me check.\n"), "{out}");
        assert!(out.ends_with("\nThat should do it."), "{out}");
    }

    /// A reasoning block is text like any other here. Whether a call inside one
    /// counts is the extractor's decision — it strips think blocks before
    /// looking — and this must not pre-empt it by mangling the block.
    #[test]
    fn a_think_block_is_copied_through_untouched() {
        let out = function_blocks_to_json(
            "<think>I should ask the clock.</think>\n<function=datetime>\n</function>",
            &catalog(),
        )
        .unwrap();
        assert!(
            out.starts_with("<think>I should ask the clock.</think>\n"),
            "{out}",
        );
    }

    #[test]
    fn whitespace_around_a_name_or_key_is_trimmed() {
        let out = function_blocks_to_json(
            "<function= named >\n<parameter= label >\nvalue</parameter>\n</function>",
            &catalog(),
        )
        .unwrap();
        let call = call_of(&out);
        assert_eq!(call["name"], "named");
        assert_eq!(call["arguments"]["label"], serde_json::json!("value"));
    }

    // ── the property that keeps the two halves honest ───────────────────────

    /// **Renderer and parser are inverses, checked against the renderer
    /// itself.** `ToolCallEnvelope::render` is what worked examples and the
    /// ingest chains emit, and the grammar is compiled from the same fields — so
    /// a change to either side that breaks the round trip fails here rather than
    /// in production, silently, as a tool that never runs.
    #[test]
    fn a_rendered_call_parses_back_to_what_was_rendered() {
        let env = ToolCallEnvelope::qwen35();
        let cases: &[(&str, &[(&str, &str)])] = &[
            ("datetime", &[]),
            ("file_read", &[("path", "src/main.rs")]),
            ("file_read", &[("path", "src/main.rs"), ("start_line", "42")]),
            ("say", &[("to", "Mira"), ("words", "she said \"no\"")]),
            ("say", &[("words", "first\nsecond")]),
            ("toggle", &[("on", "true")]),
            ("scale", &[("factor", "1.5")]),
        ];
        for (name, args) in cases {
            let rendered = env.render(name, args);
            let out = function_blocks_to_json(&rendered, &catalog())
                .unwrap_or_else(|| panic!("no block found in {rendered:?}"));
            let call = call_of(&out);
            assert_eq!(call["name"], *name, "{rendered:?}");
            for (k, v) in *args {
                // Compared through the same typing the catalog declares, so this
                // asserts the value survived rather than restating the coercion.
                let want = typed_value(v, param_type(catalog().iter().find(|t| t.name == *name), k));
                assert_eq!(&call["arguments"][*k], &want, "{rendered:?} key {k}");
            }
        }
    }
}
