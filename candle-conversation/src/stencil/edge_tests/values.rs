//! The free spans: a value the model writes whole, attacked by being empty,
//! cut short, malformed, or not JSON at all. The stencil cannot mask a free
//! decode, so it has to repair what arrives — close what is open, escape what
//! must be escaped, and supply a value where there is none.

use std::sync::Arc;

use super::{call, parse, run, Step, Step::Text as T, Step::Token as K, EOS};
use crate::stencil::builder::StencilTreeBuilder;
use crate::stencil::compile::compile;
use crate::stencil::sim::{simulate, Oracle};
use crate::stencil::terminator::Terminator;
use crate::stencil::tree::FreeTextLimits;
use crate::stencil::vocab::TestVocab;

fn pack(values: &str) -> String {
    call("pack", &format!("\"values\":{values}"))
}

fn element(fields: &str) -> String {
    call(
        "read_files",
        &format!("\"files\": [{{\"path\": \"a\"{fields}}}]"),
    )
}

fn commands(inner: &str) -> String {
    call("run_commands", &format!("\"commands\": [{inner}]"))
}

/// Run `pack` with the free value written as `value`, then the model closing
/// the call.
fn run_pack(v: &TestVocab, value: &[Step]) -> String {
    let mut steps = vec![T("pack\"")];
    steps.extend(value.iter().map(|s| match s {
        Step::Text(t) => T(t),
        Step::Token(id) => K(*id),
    }));
    steps.push(T("}}\n</tool_call>"));
    run(v, &steps).0
}

// ── Empty values ────────────────────────────────────────────────────────────

/// The model closes the field before writing its value. `"start_line":,` is
/// not JSON; the stencil writes `null` in the gap and drops the delimiter, and
/// the model carries on from the gate.
#[test]
fn an_empty_value_becomes_null() {
    let v = TestVocab::new();
    let with = |tail: &[&'static str]| {
        let mut steps = vec![
            T("read_files\""),
            T("{"),
            T(" \"a\""),
            T(", \"start_line\":"),
        ];
        steps.extend(tail.iter().map(|t| T(t)));
        run(&v, &steps).0
    };

    assert_eq!(
        with(&[",", ", \"end_line\":", " 2", "}", "]"]),
        element(", \"start_line\": null, \"end_line\": 2")
    );
    // A closer instead — the right one, a stray `]`, or either after a space.
    for tail in [
        &["}", "}", "]"][..],
        &["]", "}", "]"][..],
        &[" ", "}", "}", "]"][..],
        &[" ", "]", "}", "]"][..],
    ] {
        assert_eq!(with(tail), element(", \"start_line\": null"), "{tail:?}");
    }
}

// ── Malformed structure inside a free value ─────────────────────────────────

/// A brace closing an array: the free value is closed with the brackets it
/// actually opened, and the stray brace is dropped.
#[test]
fn a_mismatched_closer_closes_what_was_opened() {
    let v = TestVocab::new();
    assert_eq!(run_pack(&v, &[T(" [1, [2"), T("}")]), pack(" [1, [2]]"));
    assert_eq!(
        run_pack(&v, &[T(" {\"a\": [1"), T("}")]),
        pack(" {\"a\": [1]}")
    );
}

/// A complete value followed by more in the same token: the value keeps its
/// own bytes and the rest belongs to the grammar.
#[test]
fn trailing_bytes_after_a_complete_value_are_left_to_the_grammar() {
    let v = TestVocab::new().with_special(" [1]]}", 300);
    assert_eq!(run_pack(&v, &[K(300)]), pack(" [1]"));
}

/// A second value after a complete scalar, and a number that cannot continue
/// (`01`): the first value stands, and the extra is dropped.
#[test]
fn bytes_that_cannot_extend_a_complete_scalar_end_it() {
    let v = TestVocab::new();
    let head = || {
        vec![
            T("read_files\""),
            T("{"),
            T(" \"a\""),
            T(", \"start_line\":"),
        ]
    };

    let mut steps = head();
    steps.extend([T(" 5 "), T("6"), T("}"), T("]")]);
    assert_eq!(run(&v, &steps).0, element(", \"start_line\": 5 "));

    let mut steps = head();
    steps.extend([T(" 0"), T("1"), T("}"), T("]")]);
    assert_eq!(run(&v, &steps).0, element(", \"start_line\": 0"));
}

/// Not JSON as written. A Python literal, a single-quoted string and an
/// unquoted key each have one JSON meaning, and keep it — spelled correctly,
/// not replaced (`edge_tests::intent` covers them in depth). `NaN` becomes
/// `null`, as `JSON.stringify` writes it. A comment has no meaning at all: the
/// value before it stands, completed.
#[test]
fn text_that_is_not_json_is_repaired_or_closed() {
    let v = TestVocab::new()
        .with_special(" True", 301)
        .with_special(" 'a'", 302)
        .with_special("a: 1}", 303)
        .with_special("// no", 304)
        .with_special(" NaN", 305);
    assert_eq!(run_pack(&v, &[K(301)]), pack(" true"));
    assert_eq!(run_pack(&v, &[K(302)]), pack(" \"a\""));
    assert_eq!(run_pack(&v, &[T(" {"), K(303)]), pack(" {\"a\": 1}"));
    assert_eq!(run_pack(&v, &[T(" [1, "), K(304)]), pack(" [1, null]"));
    assert_eq!(run_pack(&v, &[K(305)]), pack(" null"));
}

// ── Cut short: EOS mid-value ────────────────────────────────────────────────

/// An EOS anywhere inside a free value completes it minimally: open strings
/// close, open containers close in the order they opened, a partial literal is
/// finished, a partial number gets its missing digit, and a missing value or
/// key is supplied.
#[test]
fn eos_mid_value_completes_it() {
    let v = TestVocab::new();
    let cases: &[(&str, &str)] = &[
        (" [1, {\"a\": \"x", " [1, {\"a\": \"x\"}]"),
        (" [1,", " [1, null]"),
        (" [", " []"),
        (" [[[[", " [[[[]]]]"),
        (" {", " {}"),
        (" {\"a", " {\"a\": null}"),
        (" {\"a\"", " {\"a\": null}"),
        (" {\"a\":", " {\"a\": null}"),
        (" {\"a\": 1,", " {\"a\": 1, \"\": null}"),
        (" \"x\\", " \"x\\\\\""),
        (" \"\\u1", " \"\\u1000\""),
        (" tru", " true"),
        (" f", " false"),
        (" nul", " null"),
        (" -", " -0"),
        (" 1.", " 1.0"),
        (" 1e", " 1e0"),
        (" 1E+", " 1E+0"),
        (" 1e-", " 1e-0"),
        ("", " null"),
        (" ", " null"),
    ];
    for (written, repaired) in cases {
        let text = run_pack(&v, &[T(written), K(EOS)]);
        assert_eq!(text, pack(repaired), "after {written:?}");
    }
}

/// The hard limit is a cut too, and completes the same way.
#[test]
fn a_forced_close_completes_the_value() {
    let v = TestVocab::new();
    let limits = FreeTextLimits {
        ramp_start: None,
        ramp_len: 0,
        boost: 0.0,
        forced_after: 4,
    };
    let spec = StencilTreeBuilder::new("t")
        .root("v")
        .free_text("v", Terminator::JsonValue, false, limits, "done")
        .end("done")
        .build()
        .unwrap();
    let tree = Arc::new(compile(&spec, &v).unwrap());
    let run = simulate(
        tree,
        &v,
        Oracle::Policy(Box::new({
            let mut it = " [1,".bytes().chain(std::iter::repeat(b' '));
            move |_| it.next().unwrap() as u32
        })),
        100,
    )
    .unwrap();
    assert_eq!(run.forced_closes, 1);
    assert_eq!(run.text(&v), " [1, null]");
}

// ── Strings that are not valid JSON strings ─────────────────────────────────

/// A raw control character is not allowed inside a JSON string. Models write
/// raw newlines in shell commands all the time; the stencil escapes them in
/// place and the string carries on.
#[test]
fn raw_control_characters_in_a_string_are_escaped() {
    let v = TestVocab::new().with_special("a\nb", 306);
    let cases: &[(&str, &str, &str)] = &[
        ("\n", "echo a\\nb", "echo a\nb"),
        ("\t", "echo a\\tb", "echo a\tb"),
        ("\r", "echo a\\rb", "echo a\rb"),
        ("\u{1}", "echo a\\u0001b", "echo a\u{1}b"),
    ];
    for (ctl, written, value) in cases {
        let (text, _) = run(
            &v,
            &[
                T("run_commands\""),
                T("\""),
                T("echo a"),
                T(ctl),
                T("b\""),
                T("]"),
            ],
        );
        assert_eq!(text, commands(&format!("\"{written}\"")));
        assert_eq!(parse(&text)["arguments"]["commands"][0], *value);
    }

    // Merged into a longer token: the token is rewritten whole.
    let (text, _) = run(&v, &[T("run_commands\""), T("\""), K(306), T("\""), T("]")]);
    assert_eq!(text, commands("\"a\\nb\""));
}

/// An escape JSON does not have becomes a literal backslash; a `\u` with too
/// few hex digits is padded.
#[test]
fn invalid_escapes_are_made_valid() {
    let v = TestVocab::new();
    let (text, _) = run(
        &v,
        &[T("run_commands\""), T("\""), T("a\\q"), T("\""), T("]")],
    );
    assert_eq!(parse(&text)["arguments"]["commands"][0], "a\\q");

    let (text, _) = run(
        &v,
        &[T("run_commands\""), T("\""), T("\\u12G"), T("\""), T("]")],
    );
    assert_eq!(parse(&text)["arguments"]["commands"][0], "\u{1200}G");
}
