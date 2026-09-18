//! The grammar-written arrays and objects, and the delimiters a model gets
//! wrong around them: a separator where the element closes, a delimiter merged
//! into the value before it, a call cut short inside an element.

use super::{call, run, Step::Text as T, Step::Token as K, EOS};
use crate::stencil::vocab::TestVocab;

const READ: &str = "read_files";

fn files(inner: &str) -> String {
    call(READ, &format!("\"files\": [{inner}]"))
}

fn commands(inner: &str) -> String {
    call("run_commands", &format!("\"commands\": [{inner}]"))
}

/// A `,` after the element's last field has nothing to separate: dropped, and
/// the grammar closes the element.
#[test]
fn a_comma_after_the_last_field_is_dropped() {
    let v = TestVocab::new();
    let (text, _) = run(
        &v,
        &[
            T("read_files\""),
            T("{"),
            T("a\""),
            T(", \"start_line\":"),
            T(" 1"),
            T(","),
            T(" \"end_line\":"),
            T(" 2"),
            T(","),
            T("]"),
        ],
    );
    assert_eq!(
        text,
        files("{\"path\": \"a\", \"start_line\": 1, \"end_line\": 2}")
    );
}

/// The live failure's shape as one token: every closer the model thought it
/// owed, merged. It opens on `}`, which the grammar would take, but the token
/// is not the grammar's — dropped whole, and the structure is written.
#[test]
fn a_merged_run_of_closers_after_a_value_is_dropped() {
    let v = TestVocab::new().with_special("]}}}", 300);
    for closers in [K(300), T("]"), T("}")] {
        let (text, _) = run(
            &v,
            &[
                T("read_files\""),
                T("{"),
                T("a\""),
                T(", \"end_line\":"),
                T(" 2"),
                closers,
                T("]"),
            ],
        );
        assert_eq!(text, files("{\"path\": \"a\", \"end_line\": 2}"));
    }
}

/// A number merged with the delimiter after it keeps its digits and loses the
/// delimiter, right or wrong — the grammar writes what follows.
#[test]
fn digits_merged_with_a_closer_keep_the_digits() {
    let v = TestVocab::new()
        .with_special("20}", 301)
        .with_special("20]", 302)
        .with_special("20,", 303);
    for merged in [301, 302, 303] {
        let (text, _) = run(
            &v,
            &[
                T("read_files\""),
                T("{"),
                T("a\""),
                T(", \"end_line\":"),
                T(" "),
                K(merged),
                T("]"),
            ],
        );
        assert_eq!(text, files("{\"path\": \"a\", \"end_line\": 20}"));
    }
}

/// A string element's closing quote merged with the array's close, or with the
/// next element's separator and quote: the string keeps its quote, and the
/// model chooses again at the array's branch.
#[test]
fn a_string_close_merged_with_what_follows_keeps_only_the_quote() {
    let v = TestVocab::new()
        .with_special("ls\"]", 304)
        .with_special("ls\", \"", 305);
    let (text, _) = run(&v, &[T("run_commands\""), T("\""), K(304), T("]")]);
    assert_eq!(text, commands("\"ls\""));

    let (text, _) = run(
        &v,
        &[
            T("run_commands\""),
            T("\""),
            K(305),
            T(", \""),
            T("pwd\""),
            T("]"),
        ],
    );
    assert_eq!(text, commands("\"ls\", \"pwd\""));
}

/// A dropped delimiter does not decide the array: after the grammar closes the
/// element, the model may still add another.
#[test]
fn after_a_dropped_delimiter_the_array_can_continue() {
    let v = TestVocab::new();
    let (text, _) = run(
        &v,
        &[
            T("read_files\""),
            T("{"),
            T("a\""),
            T(", \"end_line\":"),
            T(" 2"),
            T(","),
            T(", {"),
            T("b\""),
            T("}"),
            T("]"),
        ],
    );
    assert_eq!(
        text,
        files("{\"path\": \"a\", \"end_line\": 2}, {\"path\": \"b\"}")
    );
}

/// Brackets and braces inside a string element are its content, not structure.
#[test]
fn closers_inside_a_string_element_are_content() {
    let v = TestVocab::new();
    let (text, _) = run(
        &v,
        &[T("run_commands\""), T("\""), T("a}]\\\"["), T("\""), T("]")],
    );
    assert_eq!(text, commands("\"a}]\\\"[\""));
}

/// An EOS inside an element's string: the string is closed, and the model
/// finishes the element and the array from the branches that follow.
#[test]
fn eos_inside_an_element_string_closes_the_string() {
    let v = TestVocab::new();
    let (text, _) = run(
        &v,
        &[
            T("read_files\""),
            T("{"),
            T("src/ma"),
            K(EOS),
            T("}"),
            T("]"),
        ],
    );
    assert_eq!(text, files("{\"path\": \"src/ma\"}"));
}

/// An EOS where an element's number should be: a value is written for it, and
/// the grammar closes the element.
#[test]
fn eos_in_an_element_number_writes_a_value() {
    let v = TestVocab::new();
    let (text, _) = run(
        &v,
        &[
            T("read_files\""),
            T("{"),
            T("a\""),
            T(", \"end_line\":"),
            K(EOS),
            T("]"),
        ],
    );
    assert_eq!(text, files("{\"path\": \"a\", \"end_line\": null}"));

    let (text, _) = run(
        &v,
        &[
            T("read_files\""),
            T("{"),
            T("a\""),
            T(", \"end_line\":"),
            T(" 34"),
            K(EOS),
            T("]"),
        ],
    );
    assert_eq!(text, files("{\"path\": \"a\", \"end_line\": 34}"));
}

/// The grammar holds the shape, not the scalar type: an array where a line
/// number belongs is still one JSON value, and the element still closes.
#[test]
fn a_structure_where_a_number_belongs_is_still_one_value() {
    let v = TestVocab::new();
    let (text, _) = run(
        &v,
        &[
            T("read_files\""),
            T("{"),
            T("a\""),
            T(", \"start_line\":"),
            T(" [1, {\"x\": \"]\"}]"),
            T("}"),
            T("]"),
        ],
    );
    assert_eq!(
        text,
        files("{\"path\": \"a\", \"start_line\": [1, {\"x\": \"]\"}]}")
    );
}
