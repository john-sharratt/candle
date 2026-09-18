//! **A repair keeps what the model meant.**
//!
//! Every case here is a value a model writes the way another language spells
//! it — `True`, `None`, `'quoted'`, `{key: 1}`, `[1 2]`, `.5` — paired with the
//! JSON it means. The assertion is on that meaning: the call must parse *to the
//! intended value*, not merely parse. Replacing `True` with `null` parses too,
//! and says the opposite.
//!
//! And every case runs under every way a tokenizer could have cut the text:
//! one token per byte, the whole text as one token, and each split into two.
//! A repair is made one token at a time against what is already committed, so
//! a repair that came out right only for some cuts would be right by accident.
//! The one repair that genuinely depends on the cut — a trailing comma, which
//! can only be taken back before it is committed — has its own test saying
//! exactly where the line falls.

use std::collections::VecDeque;

use serde_json::{json, Value};

use super::{parse, tree, Rng};
use crate::stencil::mask::AllowedSet;
use crate::stencil::session::Observe;
use crate::stencil::sim::{simulate, Oracle};
use crate::stencil::vocab::{TestVocab, TokenId, Vocab};

/// How a tokenizer could cut `text`: a token per byte, the whole text, and
/// every split into two at a character boundary.
fn cuts(text: &str) -> Vec<Vec<String>> {
    let mut cuts = vec![
        text.chars().map(String::from).collect(),
        vec![text.to_string()],
    ];
    for (i, _) in text.char_indices().skip(1) {
        cuts.push(vec![text[..i].to_string(), text[i..].to_string()]);
    }
    cuts
}

/// How the model's turn ends after the value's tokens.
#[derive(Clone, Copy, PartialEq)]
enum Then {
    /// It closes the call.
    Close,
    /// It stops: EOS, where the value may still be open.
    Eos,
}

/// What `pack`'s free `values` came out as when the model wrote `parts` — each
/// part one token — and then did `then`. Also returns the committed text of the
/// call and whether any token was repaired.
fn write_value(parts: &[String], then: Then) -> (Value, String, bool) {
    // A multi-byte part is one token of its own; a single byte already is.
    let mut v = TestVocab::new();
    let mut named: Vec<&String> = Vec::new();
    let mut tokens: VecDeque<TokenId> = parts
        .iter()
        .map(|part| match part.len() {
            1 => part.as_bytes()[0] as TokenId,
            _ => {
                let index = named.iter().position(|s| *s == part).unwrap_or_else(|| {
                    named.push(part);
                    named.len() - 1
                });
                300 + index as TokenId
            }
        })
        .collect();
    for (i, part) in named.iter().enumerate() {
        v = v.with_special(part, 300 + i as TokenId);
    }
    if then == Then::Eos {
        tokens.push_back(super::EOS);
    }

    // The grammar is compiled against the plain vocabulary: the cut is the
    // model's, and must not change how the grammar's own text tokenizes (a
    // generated cut like `"pa` would otherwise merge into the tool's name).
    // The walk decodes and re-tokenizes with the model's vocabulary.
    let tree = tree(&TestVocab::new());
    let mut name = b"pack\"".to_vec();
    let vocab = v.clone();
    let oracle = Oracle::Policy(Box::new(
        move |allowed: Option<&AllowedSet>| match allowed {
            // The name: the allowed token that writes what is left of it.
            Some(set) if !name.is_empty() => {
                let (t, len) = set
                    .tokens()
                    .iter()
                    .map(|&t| (t, vocab.token_bytes(t)))
                    .filter(|(_, b)| !b.is_empty() && name.starts_with(b))
                    .map(|(t, b)| (t, b.len()))
                    .max_by_key(|&(_, len)| len)
                    .expect("the name is writable");
                name.drain(..len);
                t
            }
            // After the value: close the call rather than add a label.
            Some(set) => *set
                .tokens()
                .iter()
                .find(|&&t| vocab.token_bytes(t).first() != Some(&b','))
                .expect("the close is always offered"),
            None => tokens.pop_front().unwrap_or(b'}' as TokenId),
        },
    ));
    let run = simulate(tree, &v, oracle, 10_000).unwrap_or_else(|e| panic!("{parts:?}: {e}"));
    let text = run.text(&v);
    assert!(
        !run.observes.contains(&Observe::Bailed),
        "{parts:?} bailed: {text:?}"
    );
    let repaired = run
        .observes
        .iter()
        .any(|o| matches!(o, Observe::Repaired { .. }));
    let call = parse(&text);
    (call["arguments"]["values"].clone(), text, repaired)
}

/// Every cut of `written` must mean `intended`.
fn assert_means(written: &str, intended: Value) {
    for parts in cuts(written) {
        let (got, text, _) = write_value(&parts, Then::Close);
        assert_eq!(
            got, intended,
            "{written:?} cut as {parts:?} committed {text:?}"
        );
    }
}

// ── Literals ────────────────────────────────────────────────────────────────

#[test]
fn python_literals_keep_their_meaning() {
    assert_means(" True", json!(true));
    assert_means(" False", json!(false));
    assert_means(" None", json!(null));
    assert_means(" [True, False, None]", json!([true, false, null]));
    assert_means(
        " {\"on\": True, \"off\": False, \"unset\": None}",
        json!({"on": true, "off": false, "unset": null}),
    );
}

#[test]
fn literals_in_any_case_keep_their_meaning() {
    for (written, meant) in [
        (" TRUE", json!(true)),
        (" tRuE", json!(true)),
        (" FALSE", json!(false)),
        (" fAlSe", json!(false)),
        (" NULL", json!(null)),
        (" Null", json!(null)),
        (" [TRUE, False, nULL]", json!([true, false, null])),
    ] {
        assert_means(written, meant);
    }
}

/// Other languages' words for no value, and the non-finite numbers JSON cannot
/// hold, which `JSON.stringify` itself writes as `null`.
#[test]
fn other_languages_empty_values_become_null() {
    for written in [
        " nil",
        " Nil",
        " undefined",
        " NaN",
        " nan",
        " Infinity",
        " inf",
        " Inf",
    ] {
        assert_means(written, json!(null));
    }
    assert_means(
        " {\"a\": nil, \"b\": undefined, \"c\": NaN, \"d\": [inf, 1]}",
        json!({"a": null, "b": null, "c": null, "d": [null, 1]}),
    );
}

/// A literal is only a literal as a whole word: one cut short by the end of the
/// call is finished, and valid JSON literals pass untouched.
#[test]
fn a_literal_cut_short_is_finished_as_the_word_it_began() {
    for (written, meant) in [
        (" Tru", json!(true)),
        (" T", json!(true)),
        (" Fals", json!(false)),
        (" No", json!(null)),
        (" [Tr", json!([true])),
        (" {\"k\": Fa", json!({"k": false})),
    ] {
        // The model stops mid-word; the value is completed.
        for parts in cuts(written) {
            let (got, text, _) = write_value(&parts, Then::Eos);
            assert_eq!(got, meant, "{written:?} cut as {parts:?}: {text:?}");
        }
    }
}

// ── Strings ─────────────────────────────────────────────────────────────────

#[test]
fn single_quoted_strings_keep_their_content() {
    for (written, meant) in [
        (" 'abc'", json!("abc")),
        (" ''", json!("")),
        (" 'say \"hi\"'", json!("say \"hi\"")),
        (" 'don\\'t'", json!("don't")),
        (" 'a\\\\b'", json!("a\\b")),
        (" 'tab\\there'", json!("tab\there")),
        (" ['x', \"y\", 'z']", json!(["x", "y", "z"])),
        (" 'it is \"fine\", really'", json!("it is \"fine\", really")),
    ] {
        assert_means(written, meant);
    }
}

#[test]
fn escapes_json_lacks_keep_their_character() {
    for (written, meant) in [
        (" \"don\\'t\"", json!("don't")),
        (" \"\\x41\\x42\"", json!("AB")),
        (" '\\x7e'", json!("~")),
        (" \"bell\\a\"", json!("bell\u{7}")),
        (" \"vt\\v\"", json!("vt\u{b}")),
        // No language gives these a meaning: the backslash is kept as written.
        (" \"C:\\qdir\"", json!("C:\\qdir")),
        (" \"\\p{L}\"", json!("\\p{L}")),
    ] {
        assert_means(written, meant);
    }
}

#[test]
fn raw_control_characters_keep_their_character() {
    for (written, meant) in [
        (" \"line one\nline two\"", json!("line one\nline two")),
        (" \"a\tb\"", json!("a\tb")),
        (" \"cr\r\"", json!("cr\r")),
        (" 'x\ny'", json!("x\ny")),
        (" [\"\u{1}\"]", json!(["\u{1}"])),
    ] {
        assert_means(written, meant);
    }
}

// ── Objects ─────────────────────────────────────────────────────────────────

#[test]
fn unquoted_and_single_quoted_keys_keep_their_names() {
    for (written, meant) in [
        (" {a: 1}", json!({"a": 1})),
        (
            " {path: 'a.rs', start_line: 3}",
            json!({"path": "a.rs", "start_line": 3}),
        ),
        (
            " {_private: true, $ref: 'x'}",
            json!({"_private": true, "$ref": "x"}),
        ),
        (" {1: 'one', 2: 'two'}", json!({"1": "one", "2": "two"})),
        (" {'a': 1, \"b\": 2, c: 3}", json!({"a": 1, "b": 2, "c": 3})),
        (
            " {outer: {inner: [True]}}",
            json!({"outer": {"inner": [true]}}),
        ),
        (" {a : 1}", json!({"a": 1})),
    ] {
        assert_means(written, meant);
    }
}

#[test]
fn a_missing_colon_is_supplied() {
    for (written, meant) in [
        (" {\"a\" 1}", json!({"a": 1})),
        (" {\"a\" \"b\"}", json!({"a": "b"})),
        (" {'a' True}", json!({"a": true})),
        (" {a [1, 2]}", json!({"a": [1, 2]})),
    ] {
        assert_means(written, meant);
    }
}

#[test]
fn a_missing_comma_is_supplied() {
    for (written, meant) in [
        (" [1 2 3]", json!([1, 2, 3])),
        (" [\"a\" \"b\"]", json!(["a", "b"])),
        (" [True False]", json!([true, false])),
        (" [[1] [2]]", json!([[1], [2]])),
        (" [{\"a\": 1} {\"b\": 2}]", json!([{"a": 1}, {"b": 2}])),
        (" {\"a\": 1 \"b\": 2}", json!({"a": 1, "b": 2})),
        (" {a: 1 b: 2}", json!({"a": 1, "b": 2})),
        (" {'a': 'x'\n'b': 'y'}", json!({"a": "x", "b": "y"})),
    ] {
        assert_means(written, meant);
    }
}

// ── Numbers ─────────────────────────────────────────────────────────────────

#[test]
fn number_shorthand_keeps_its_value() {
    for (written, meant) in [
        (" [.5]", json!([0.5])),
        (" [-.25]", json!([-0.25])),
        (" [5.]", json!([5.0])),
        (" [5., 6]", json!([5.0, 6])),
        (" [1.e3]", json!([1000.0])),
        (" {\"ratio\": .75}", json!({"ratio": 0.75})),
    ] {
        assert_means(written, meant);
    }
}

// ── Mixed ───────────────────────────────────────────────────────────────────

/// A Python dict literal, as a model writes one when it forgets which language
/// the call is in: every repair at once, and the meaning intact.
#[test]
fn a_python_dict_literal_means_the_same_json() {
    assert_means(
        " {'files': [{'path': 'src/main.rs', 'start': None, 'recursive': True}], \
         'depth': inf, 'note': 'it\\'s \"quoted\"', 'ratio': .5}",
        json!({
            "files": [{"path": "src/main.rs", "start": null, "recursive": true}],
            "depth": null,
            "note": "it's \"quoted\"",
            "ratio": 0.5
        }),
    );
}

/// A JavaScript object literal: unquoted keys, `undefined`, single quotes.
#[test]
fn a_javascript_object_literal_means_the_same_json() {
    assert_means(
        " {name: 'x', value: undefined, list: [1 2], ok: TRUE}",
        json!({"name": "x", "value": null, "list": [1, 2], "ok": true}),
    );
}

// ── What is not touched ─────────────────────────────────────────────────────

/// Valid JSON is committed byte for byte under every cut, and nothing is
/// reported as repaired — the repairs cost nothing on the path that needs none.
#[test]
fn valid_json_is_committed_exactly_as_written() {
    for written in [
        " [1, -2.5e+3, 0, true, false, null]",
        " {\"a\": {\"b\": [[], {}]}, \"c\": \"}]'\\\"\"}",
        " \"it's plain\"",
        " [ 1 , 2 ]",
        " {\"t\": \"True\", \"n\": \"None\"}",
    ] {
        for parts in cuts(written) {
            let (_, text, repaired) = write_value(&parts, Then::Close);
            assert!(!repaired, "{written:?} cut as {parts:?} was rewritten");
            assert!(
                text.contains(&format!("\"values\":{written}")),
                "{written:?} cut as {parts:?} committed {text:?}"
            );
        }
    }
}

/// Words that are not JSON and have no JSON meaning are not given one: the
/// value they began stands, completed.
#[test]
fn text_with_no_json_meaning_is_not_guessed() {
    for (written, meant) in [
        (" [1, yes]", json!([1, null])),
        (" [1, // two\n2]", json!([1, null])),
        (" {\"a\": #}", json!({"a": null})),
    ] {
        assert_means(written, meant);
    }
}

// ── Generated: any value, any spelling, any cut ─────────────────────────────

/// A random JSON value, a few levels deep, whose strings carry the characters
/// every repair has to get right — both quotes, backslashes, raw newlines and
/// the structural bytes — and whose keys include ones that can be written bare.
fn value(rng: &mut Rng, depth: usize) -> Value {
    const CONTENT: &[&str] = &[
        "a", "Z", " ", "'", "\"", "\\", "\n", "\t", ",", "]", "}", ":", "é", "x1", "True",
    ];
    const KEYS: &[&str] = &[
        "a",
        "path",
        "_x",
        "$ref",
        "k1",
        "1",
        "two words",
        "it's",
        "q\"",
        "",
    ];
    let kinds = if depth == 0 { 5 } else { 7 };
    match rng.below(kinds) {
        0 => Value::Null,
        1 => Value::Bool(rng.chance(50)),
        2 => json!(rng.below(2001) as i64 - 1000),
        // Quarters, so every value is exact in binary and prints as written.
        3 => json!((rng.below(41) as f64 - 20.0) / 4.0),
        4 => Value::String(
            (0..rng.below(6))
                .map(|_| CONTENT[rng.below(CONTENT.len())])
                .collect(),
        ),
        5 => Value::Array((0..rng.below(4)).map(|_| value(rng, depth - 1)).collect()),
        _ => Value::Object(
            (0..rng.below(4))
                .map(|_| {
                    (
                        KEYS[rng.below(KEYS.len())].to_string(),
                        value(rng, depth - 1),
                    )
                })
                .collect(),
        ),
    }
}

/// `v` written the way a model forgetting which language it is in might write
/// it: literals in another language or case, single-quoted strings and keys,
/// bare keys, `.5`, and colons and commas left out — but never a trailing comma,
/// which only some cuts can take back.
fn misspell(rng: &mut Rng, v: &Value) -> String {
    match v {
        Value::Null => [
            "null",
            "None",
            "NULL",
            "nil",
            "undefined",
            "NaN",
            "inf",
            "Infinity",
        ][rng.below(8)]
        .to_string(),
        Value::Bool(b) => {
            let spellings: [&str; 3] = match b {
                true => ["true", "True", "TRUE"],
                false => ["false", "False", "FALSE"],
            };
            spellings[rng.below(3)].to_string()
        }
        Value::Number(n) => {
            let text = n.to_string();
            // `0.5` as `.5`, `-0.25` as `-.25`.
            match (
                rng.chance(50),
                text.strip_prefix("0."),
                text.strip_prefix("-0."),
            ) {
                (true, Some(frac), _) => format!(".{frac}"),
                (true, _, Some(frac)) => format!("-.{frac}"),
                _ => text,
            }
        }
        Value::String(s) => quoted(rng, s),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(|i| misspell(rng, i)).collect();
            format!("[{}]", join(rng, &parts))
        }
        Value::Object(fields) => {
            let parts: Vec<String> = fields
                .iter()
                .map(|(k, val)| {
                    let bare = !k.is_empty()
                        && k.bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'$');
                    let key = match (bare && rng.chance(40), rng.chance(50)) {
                        (true, _) => k.clone(),
                        (false, true) => quoted(rng, k),
                        (false, false) => serde_json::to_string(k).unwrap(),
                    };
                    let colon = match rng.below(4) {
                        0 => " ",
                        1 => ":",
                        _ => ": ",
                    };
                    format!("{key}{colon}{}", misspell(rng, val))
                })
                .collect();
            format!("{{{}}}", join(rng, &parts))
        }
    }
}

/// Members separated by `, `, `,`, or — sometimes — only a space.
fn join(rng: &mut Rng, parts: &[String]) -> String {
    let mut out = String::new();
    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            out.push_str([", ", ",", " "][rng.below(3)]);
        }
        out.push_str(part);
    }
    out
}

/// A string in single quotes (escaping `'`, leaving `"` raw) or double quotes
/// (escaping `"`, and `'` either way), with raw newlines and tabs left raw.
fn quoted(rng: &mut Rng, s: &str) -> String {
    let single = rng.chance(50);
    let (quote, other) = match single {
        true => ('\'', '"'),
        false => ('"', '\''),
    };
    let mut out = String::from(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if c == other && !single && rng.chance(30) => out.push_str("\\'"),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// A random cut of `text` into tokens of one to six characters.
fn random_cut(rng: &mut Rng, text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut parts = Vec::new();
    let mut at = 0;
    while at < chars.len() {
        let len = (1 + rng.below(6)).min(chars.len() - at);
        parts.push(chars[at..at + len].iter().collect());
        at += len;
    }
    parts
}

/// **The property the whole repair set exists for**: whatever value a model
/// meant, however it spelled it and however the tokenizer cut it, the call
/// means that value.
#[test]
fn any_value_misspelled_any_way_and_cut_anywhere_means_the_same_json() {
    for seed in 0..600u64 {
        let mut rng = Rng(seed);
        let meant = value(&mut rng, 3);
        let written = format!(" {}", misspell(&mut rng, &meant));
        let parts = random_cut(&mut rng, &written);
        let (got, text, _) = write_value(&parts, Then::Close);
        assert_eq!(
            got, meant,
            "seed {seed}: {written:?} cut as {parts:?} committed {text:?}"
        );
    }
}

// ── The one cut that matters ────────────────────────────────────────────────

/// **A trailing comma can be taken back only before it is committed.** When
/// the `,` and the closer arrive in one token the comma is dropped and the
/// array means what was written. When the `,` came in an earlier token it is
/// already in the K/V: the array has to be completed after it, and the least
/// it can hold there is `null`.
#[test]
fn a_trailing_comma_is_dropped_when_it_shares_a_token_with_the_closer() {
    let written = |parts: &[&str]| {
        let parts: Vec<String> = parts.iter().map(|s| s.to_string()).collect();
        write_value(&parts, Then::Close).0
    };
    for parts in [
        &[" [1, 2,]"][..],
        &[" [1, 2", ",]"],
        &[" [1, 2", ", ]"],
        &[" [1,", " 2,\n]"],
    ] {
        assert_eq!(written(parts), json!([1, 2]), "{parts:?}");
    }
    for parts in [&[" [1, 2,", "]"][..], &[" [1, 2,", " ]"]] {
        assert_eq!(written(parts), json!([1, 2, null]), "{parts:?}");
    }
    assert_eq!(written(&[" {\"a\": 1,}"]), json!({"a": 1}));
    assert_eq!(written(&[" {\"a\": 1,", "}"]), json!({"a": 1, "": null}));
}
