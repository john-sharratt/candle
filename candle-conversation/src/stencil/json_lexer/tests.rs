//! The lexer on its own: which bytes continue a value, what each repair
//! writes, and what completes a value cut anywhere. Byte-exact.
//!
//! Token boundaries do not exist at this level — that the same repair comes
//! out however a model's text is split into tokens is asserted end to end, in
//! `edge_tests::intent`.

use super::*;

/// Feed `text` one byte at a time, committing what each byte writes, until the
/// value ends. An invalid byte ends it with its completion appended, as the
/// terminator does. Returns what was committed and how it ended.
fn run(mut lexer: JsonLexer, text: &[u8]) -> (JsonLexer, Vec<u8>, Effect) {
    let mut out = Vec::new();
    for &b in text {
        let step = lexer.step(b);
        let effect = step.effect;
        match (step.write, effect) {
            (Some(w), _) => out.extend(w),
            (None, Effect::Continue | Effect::Complete) => out.push(b),
            (None, _) => {}
        }
        match effect {
            Effect::Continue => {}
            Effect::Complete | Effect::Delimiter => return (lexer, out, effect),
            Effect::Invalid | Effect::TrailingComma => {
                out.extend(lexer.completion());
                return (lexer, out, effect);
            }
        }
    }
    (lexer, out, Effect::Continue)
}

fn show(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

/// `text` committed and then completed, as a cut value would be.
fn completed(start: JsonLexer, text: &[u8]) -> Vec<u8> {
    let (lexer, mut out, effect) = run(start, text);
    if effect == Effect::Continue {
        out.extend(lexer.completion());
    }
    out
}

fn assert_json(bytes: &[u8]) -> serde_json::Value {
    serde_json::from_str(show(bytes).trim())
        .unwrap_or_else(|e| panic!("{:?} is not JSON: {e}", show(bytes)))
}

// ── Valid JSON is committed exactly as written ──────────────────────────────

#[test]
fn valid_values_are_committed_unchanged() {
    for text in [
        &b"[1, -2.5e+3, 0, 0.0, 1E-9, true, false, null, \"a\\\"\\u00e9\\\\\", {\"k\": [{}]}]"[..],
        b"{\"a\": {\"b\": [[], {}]}, \"c\": \"}]\\n\"}",
        b"\"plain\"",
        b"[\"it's\", \"x\\/y\"]",
        b"[ 1 , 2 ]",
    ] {
        let (_, out, effect) = run(JsonLexer::value(), text);
        assert_eq!(effect, Effect::Complete, "{}", show(text));
        assert_eq!(out, text, "rewrote valid JSON");
        assert_json(&out);
    }
}

/// A scalar at the top level is complete only when something that is not
/// part of it arrives, and only a delimiter of the enclosing object is handed
/// on.
#[test]
fn a_top_level_scalar_ends_at_a_delimiter() {
    for (text, effect) in [
        (&b"42,"[..], Effect::Delimiter),
        (b"true }", Effect::Delimiter),
        (b"-0.5e3]", Effect::Delimiter),
        (b"42x", Effect::Invalid),
        (b"01", Effect::Invalid),
        (b"5 6", Effect::Invalid),
    ] {
        assert_eq!(run(JsonLexer::value(), text).2, effect, "{}", show(text));
    }
}

// ── Repairs that keep the meaning ───────────────────────────────────────────

#[test]
fn literals_in_any_case_and_other_languages_spelling() {
    let cases: &[(&[u8], &[u8])] = &[
        (b"True", b"true"),
        (b"TRUE", b"true"),
        (b"tRuE", b"true"),
        (b"False", b"false"),
        (b"FALSE", b"false"),
        (b"Null", b"null"),
        (b"NULL", b"null"),
        (b"None", b"null"),
        (b"NONE", b"null"),
        (b"nil", b"null"),
        (b"Nil", b"null"),
        (b"undefined", b"null"),
        (b"NaN", b"null"),
        (b"nan", b"null"),
        (b"Infinity", b"null"),
        (b"inf", b"null"),
        (b"Inf", b"null"),
    ];
    for (text, want) in cases {
        // Top level: the word is followed by the enclosing delimiter.
        let mut delimited = text.to_vec();
        delimited.push(b',');
        let (_, out, effect) = run(JsonLexer::value(), &delimited);
        assert_eq!(effect, Effect::Delimiter, "{}", show(text));
        assert_eq!(out, *want, "{}", show(text));

        // Inside an array, and cut short after the word.
        let mut inner = b"[".to_vec();
        inner.extend_from_slice(text);
        inner.extend_from_slice(b", ");
        inner.extend_from_slice(text);
        inner.push(b']');
        let mut want_inner = b"[".to_vec();
        want_inner.extend_from_slice(want);
        want_inner.extend_from_slice(b", ");
        want_inner.extend_from_slice(want);
        want_inner.push(b']');
        assert_eq!(
            run(JsonLexer::value(), &inner).1,
            want_inner,
            "{}",
            show(text)
        );
        assert_eq!(completed(JsonLexer::value(), text), *want, "{}", show(text));
    }
}

/// A word cut short is finished as the literal it had to be.
#[test]
fn a_partial_word_completes_to_its_literal() {
    for (text, want) in [
        (&b"T"[..], &b"true"[..]),
        (b"Fa", b"false"),
        (b"No", b"null"),
        (b"N", b"null"),
        (b"undef", b"null"),
        (b"Infin", b"null"),
        (b"in", b"null"),
    ] {
        assert_eq!(completed(JsonLexer::value(), text), want, "{}", show(text));
    }
}

/// A word no literal begins with is not guessed at: there is no value, so
/// `null` stands in for it.
#[test]
fn a_word_no_literal_begins_with_is_not_guessed() {
    for (text, want) in [
        (&b"yes"[..], &b" null"[..]),
        (b"x", b" null"),
        (b"[1, yes]", b"[1, null]"),
    ] {
        let (_, out, effect) = run(JsonLexer::value(), text);
        assert_eq!(effect, Effect::Invalid, "{}", show(text));
        assert_eq!(out, want, "{}", show(text));
    }
}

/// A word that begins as a literal and then breaks off is finished as the one
/// it began — it had already committed to it.
#[test]
fn a_word_that_breaks_off_finishes_as_the_literal_it_began() {
    for (text, want) in [
        (&b"[Truth]"[..], &b"[true]"[..]),
        (b"[nope]", b"[null]"),
        (b"[falsy]", b"[false]"),
    ] {
        let (_, out, effect) = run(JsonLexer::value(), text);
        assert_eq!(effect, Effect::Invalid, "{}", show(text));
        assert_eq!(out, want, "{}", show(text));
    }
}

#[test]
fn single_quoted_strings_and_keys_become_double_quoted() {
    let cases: &[(&[u8], &[u8])] = &[
        (b"'abc'", b"\"abc\""),
        (b"'say \"hi\"'", b"\"say \\\"hi\\\"\""),
        (b"'don\\'t'", b"\"don\\u0027t\""),
        (b"{'a': 'b'}", b"{\"a\": \"b\"}"),
        (b"['x', \"y\", 'z']", b"[\"x\", \"y\", \"z\"]"),
        (b"{'k': {'n': ['v']}}", b"{\"k\": {\"n\": [\"v\"]}}"),
    ];
    for (text, want) in cases {
        let out = completed(JsonLexer::value(), text);
        assert_eq!(out, *want, "{}", show(text));
        assert_json(&out);
    }
    assert_eq!(
        assert_json(&completed(JsonLexer::value(), b"'say \"hi\"'")),
        "say \"hi\""
    );
    assert_eq!(
        assert_json(&completed(JsonLexer::value(), b"'don\\'t'")),
        "don't"
    );
}

#[test]
fn unquoted_keys_are_quoted() {
    let cases: &[(&[u8], serde_json::Value)] = &[
        (b"{a: 1}", serde_json::json!({"a": 1})),
        (
            b"{path: 'a.rs', start_line: 3}",
            serde_json::json!({"path": "a.rs", "start_line": 3}),
        ),
        (
            b"{_x1: [], $y: {}}",
            serde_json::json!({"_x1": [], "$y": {}}),
        ),
        (
            b"{1: 'one', 2: 'two'}",
            serde_json::json!({"1": "one", "2": "two"}),
        ),
        (
            b"{outer: {inner: True}}",
            serde_json::json!({"outer": {"inner": true}}),
        ),
        (b"{a : 1}", serde_json::json!({"a": 1})),
    ];
    for (text, want) in cases {
        assert_eq!(
            assert_json(&completed(JsonLexer::value(), text)),
            *want,
            "{}",
            show(text)
        );
    }
}

#[test]
fn missing_colons_and_commas_are_supplied() {
    let cases: &[(&[u8], serde_json::Value)] = &[
        (b"{\"a\" 1}", serde_json::json!({"a": 1})),
        (b"{\"a\" \"b\"}", serde_json::json!({"a": "b"})),
        (b"{a 1}", serde_json::json!({"a": 1})),
        (b"[1 2 3]", serde_json::json!([1, 2, 3])),
        (b"[\"a\" \"b\"]", serde_json::json!(["a", "b"])),
        (
            b"[{\"a\": 1} {\"b\": 2}]",
            serde_json::json!([{"a": 1}, {"b": 2}]),
        ),
        (b"[[1] [2]]", serde_json::json!([[1], [2]])),
        (b"{\"a\": 1 \"b\": 2}", serde_json::json!({"a": 1, "b": 2})),
        (b"{\"a\": 1 b: 2}", serde_json::json!({"a": 1, "b": 2})),
        (b"[true false None]", serde_json::json!([true, false, null])),
    ];
    for (text, want) in cases {
        assert_eq!(
            assert_json(&completed(JsonLexer::value(), text)),
            *want,
            "{}",
            show(text)
        );
    }
}

#[test]
fn number_shorthand_is_spelled_out() {
    let cases: &[(&[u8], &[u8])] = &[
        (b".5,", b"0.5"),
        (b"-.25,", b"-0.25"),
        (b"5.,", b"5.0"),
        (b"[5., 6]", b"[5.0, 6]"),
        (b"[1.e3]", b"[1.0e3]"),
    ];
    for (text, want) in cases {
        let (_, out, _) = run(JsonLexer::value(), text);
        assert_eq!(out, *want, "{}", show(text));
    }
}

#[test]
fn escapes_other_languages_have_become_unicode_escapes() {
    let cases: &[(&[u8], &str)] = &[
        (b"\\x41B\"", "AB"),
        (b"\\'\"", "'"),
        (b"\\a\\v\"", "\u{7}\u{b}"),
        (b"a\\qb\"", "a\\qb"),
        (b"\\u12G\"", "\u{1200}G"),
    ];
    for (text, want) in cases {
        let (_, out, effect) = run(JsonLexer::string(), text);
        assert_eq!(effect, Effect::Complete, "{}", show(text));
        let mut quoted = b"\"".to_vec();
        quoted.extend(&out);
        assert_eq!(assert_json(&quoted), *want, "{}", show(text));
    }
}

#[test]
fn raw_control_characters_in_a_string_are_escaped() {
    let (_, out, _) = run(JsonLexer::string(), b"\t\r\n\x08\x0c\x01\"");
    assert_eq!(out, b"\\t\\r\\n\\b\\f\\u0001\"");
}

/// A closer straight after a separator is reported, not decided: whether the
/// separator can be taken back depends on the token it came in.
#[test]
fn a_trailing_comma_is_reported_to_the_caller() {
    for text in [&b"[1, 2,]"[..], b"[1,\n]", b"{\"a\": 1,}", b"{\"a\": 1, }"] {
        assert_eq!(
            run(JsonLexer::value(), text).2,
            Effect::TrailingComma,
            "{}",
            show(text)
        );
    }
    // Taken back, the closer closes what came before.
    let (mut lexer, _, _) = run(JsonLexer::value(), b"[1, 2,");
    assert_eq!(lexer.step(b']').effect, Effect::TrailingComma);
    lexer.retract_separator(Some(b'2'));
    assert_eq!(lexer.step(b']').effect, Effect::Complete);
}

// ── Values that cannot be read end, completed as written ────────────────────

#[test]
fn an_invalid_byte_ends_the_value_as_written() {
    let cases: &[(&[u8], &[u8])] = &[
        (b"[1, [2}", b"[1, [2]]"),
        (b"{\"a\": [1}", b"{\"a\": [1]}"),
        (b"{\"a\": 1]", b"{\"a\": 1}"),
        (b"[1, // c", b"[1, null]"),
        (b"#", b" null"),
        (b",", b" null"),
        (b"}", b" null"),
        (b"{-}", b"{}"),
        // Directly after `[` the array may simply close.
        (b"[yes]", b"[]"),
    ];
    for (text, want) in cases {
        let (_, out, effect) = run(JsonLexer::value(), text);
        assert_eq!(effect, Effect::Invalid, "{}", show(text));
        assert_eq!(out, *want, "{}", show(text));
        assert_json(&out);
    }
}

#[test]
fn completions_are_minimal_and_valid() {
    let cases: &[(&[u8], &[u8])] = &[
        (b"", b" null"),
        (b" ", b" null"),
        (b"[", b"[]"),
        (b"[1,", b"[1, null]"),
        (b"[1, ", b"[1, null]"),
        (b"[[{\"a\": [", b"[[{\"a\": []}]]"),
        (b"{", b"{}"),
        (b"{\"k", b"{\"k\": null}"),
        (b"{\"k\"", b"{\"k\": null}"),
        (b"{\"k\":", b"{\"k\": null}"),
        (b"{\"k\": 1,", b"{\"k\": 1, \"\": null}"),
        (b"{ke", b"{\"ke\": null}"),
        (b"{'k", b"{\"k\": null}"),
        (b"'ab", b"\"ab\""),
        (b"\"x\\", b"\"x\\\\\""),
        (b"\"\\u1", b"\"\\u1000\""),
        (b"tr", b"true"),
        (b"fal", b"false"),
        (b"n", b"null"),
        (b"-", b"-0"),
        (b"1.", b"1.0"),
        (b"1e", b"1e0"),
        (b"1E-", b"1E-0"),
        (b"12", b"12"),
    ];
    for (text, want) in cases {
        let out = completed(JsonLexer::value(), text);
        assert_eq!(out, *want, "{} completed to {}", show(text), show(&out));
        assert_json(&out);
    }
}

/// An invalid byte leaves the state as it was, so the completion is of the
/// value before it.
#[test]
fn an_invalid_byte_does_not_change_the_state() {
    let (before, _, _) = run(JsonLexer::value(), b"[1, {\"a\": tr");
    let mut after = before.clone();
    assert_eq!(after.step(b'x').effect, Effect::Invalid);
    assert_eq!(after, before);
}
