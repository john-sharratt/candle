//! The engine against exact expected files.

use super::{apply, Matched, ReplaceError, Replaced};

fn edited(content: &str, replacements: usize, matched: Matched) -> Replaced {
    Replaced {
        content: content.to_string(),
        replacements,
        already_applied: false,
        matched,
    }
}

fn unchanged(content: &str, matched: Matched) -> Replaced {
    Replaced {
        content: content.to_string(),
        replacements: 0,
        already_applied: true,
        matched,
    }
}

// ── Exactly ─────────────────────────────────────────────────────────────────

#[test]
fn one_exact_occurrence_is_replaced() {
    assert_eq!(
        apply("[net]\nport = 8080\n", "port = 8080", "port = 9090", false),
        Ok(edited("[net]\nport = 9090\n", 1, Matched::Exact))
    );
}

/// A substring edit works inside a line: the edit need not be whole lines.
#[test]
fn a_substring_is_replaced_in_place() {
    assert_eq!(
        apply("let host = \"localhost\";\n", "localhost", "0.0.0.0", false),
        Ok(edited("let host = \"0.0.0.0\";\n", 1, Matched::Exact))
    );
}

#[test]
fn several_occurrences_are_ambiguous_unless_all_are_asked_for() {
    let file = "a = 1\nb = 1\n";
    assert_eq!(
        apply(file, "= 1", "= 2", false),
        Err(ReplaceError::Ambiguous(
            "`old_text` occurs 2 times in the file — include more of the lines around the one to \
             change so it matches once, or set `replace_all` to change all 2"
                .to_string()
        ))
    );
    assert_eq!(
        apply(file, "= 1", "= 2", true),
        Ok(edited("a = 2\nb = 2\n", 2, Matched::Exact))
    );
}

// ── Already applied ─────────────────────────────────────────────────────────

#[test]
fn an_edit_whose_result_is_there_is_already_applied() {
    let file = "port = 9090\n";
    assert_eq!(
        apply(file, "port = 8080", "port = 9090", false),
        Ok(unchanged(file, Matched::Exact))
    );
}

/// **`3` → `30` sent twice is `30`, not `300`.** The `retries = 3` inside
/// `retries = 30` is the edit's own result.
#[test]
fn a_resent_edit_whose_new_text_contains_the_old_is_not_applied_twice() {
    let first = apply("retries = 3\n", "retries = 3", "retries = 30", false).unwrap();
    assert_eq!(first, edited("retries = 30\n", 1, Matched::Exact));
    assert_eq!(
        apply(&first.content, "retries = 3", "retries = 30", false),
        Ok(unchanged("retries = 30\n", Matched::Exact))
    );
}

/// An insertion quotes the line it follows; re-sent, it inserts nothing.
#[test]
fn a_resent_insertion_inserts_nothing() {
    let first = apply("target/\n*.log\n", "*.log\n", "*.log\n.env\n", false).unwrap();
    assert_eq!(first, edited("target/\n*.log\n.env\n", 1, Matched::Exact));
    assert_eq!(
        apply(&first.content, "*.log\n", "*.log\n.env\n", false),
        Ok(unchanged("target/\n*.log\n.env\n", Matched::Exact))
    );
}

/// A deletion leaves nothing to recognise: its old text gone is a failure,
/// never a success that wrote nothing.
#[test]
fn a_missing_deletion_is_unmatched() {
    assert!(matches!(
        apply("a\nc\n", "b\n", "", false),
        Err(ReplaceError::Unmatched(_))
    ));
    assert_eq!(
        apply("a\nb\nc\n", "b\n", "", false),
        Ok(edited("a\nc\n", 1, Matched::Exact))
    );
}

// ── Line endings ────────────────────────────────────────────────────────────

#[test]
fn an_lf_edit_lands_in_a_crlf_file_as_crlf() {
    assert_eq!(
        apply("a\r\nb\r\nc\r\n", "a\nb\n", "a\nB\nb2\n", false),
        Ok(edited("a\r\nB\r\nb2\r\nc\r\n", 1, Matched::Exact))
    );
}

// ── Indentation aside ───────────────────────────────────────────────────────

/// **The live failure, fixed.** The model quoted `list()` at four spaces
/// where the file has two, and wrote the new method at four too. The lines
/// match with their indentation ignored, and the new method lands at the
/// file's indentation.
#[test]
fn lines_quoted_at_the_wrong_indentation_are_found_and_reindented() {
    let file = "export class Inventory {\n  // Items sorted by SKU.\n  list() {\n    return [...this.items.values()];\n  }\n}\n";
    let old = "    // Items sorted by SKU.\n    list() {\n      return [...this.items.values()];\n    }\n";
    let new = "    // Items sorted by SKU.\n    list() {\n      return [...this.items.values()];\n    }\n\n    // Items below the threshold.\n    lowStock(threshold) {\n      return this.list().filter(item => item.quantity < threshold);\n    }\n";
    let expected = "export class Inventory {\n  // Items sorted by SKU.\n  list() {\n    return [...this.items.values()];\n  }\n\n  // Items below the threshold.\n  lowStock(threshold) {\n    return this.list().filter(item => item.quantity < threshold);\n  }\n}\n";
    let done = apply(file, old, new, false).unwrap();
    assert_eq!(done, edited(expected, 1, Matched::Indentation));
    // Re-sent, it is already there.
    assert_eq!(
        apply(expected, old, new, false),
        Ok(unchanged(expected, Matched::Indentation))
    );
}

/// Quoted at less indentation than the file's, the replacement is indented
/// up to the file's; trailing whitespace is ignored too.
#[test]
fn lines_quoted_with_too_little_indentation_are_indented_up() {
    assert_eq!(
        apply(
            "fn f() {\n    let x = 1;   \n    x\n}\n",
            "let x = 1;\nx\n",
            "let x = 2;\nx + 1\n",
            false
        ),
        Ok(edited(
            "fn f() {\n    let x = 2;\n    x + 1\n}\n",
            1,
            Matched::Indentation
        ))
    );
}

/// The line-by-line match keeps each file's line ending and its missing
/// final newline.
#[test]
fn the_line_match_keeps_the_files_endings() {
    assert_eq!(
        apply("a\r\n  b\r\nc", "    b\nc", "    B\nC", false),
        Ok(edited("a\r\n  B\r\nC", 1, Matched::Indentation))
    );
}

/// Lines found twice with their indentation ignored are ambiguous — on the
/// line path, where nothing matched exactly — and `replace_all` replaces
/// both, each re-indented to its own place.
#[test]
fn lines_found_twice_are_ambiguous_unless_all_are_asked_for() {
    let file = "  a\n  b\n    a\n    b\n";
    assert!(matches!(
        apply(file, "a\nb\n", "A\nB\n", false),
        Err(ReplaceError::Ambiguous(_)),
    ));
    assert_eq!(
        apply(file, "a\nb\n", "A\nB\n", true),
        Ok(edited("  A\n  B\n    A\n    B\n", 2, Matched::Indentation))
    );
}

/// **An indented line quoted shallower than the file matches only as a line**
/// — never partway into the file's indentation, where the lines after it
/// would land at the depth the model wrote.
#[test]
fn an_indented_line_quoted_shallower_is_re_indented_not_spliced() {
    assert_eq!(
        apply("    foo();\n", "  foo();\n", "  foo();\n  bar();\n", false),
        Ok(edited("    foo();\n    bar();\n", 1, Matched::Indentation))
    );
}

/// **Each line is re-indented by what its own depth maps to, not by the first
/// line's shift.** Measured live settling a merge: the model quoted the
/// conflict block with its markers at the margin — where the file has them —
/// and the code line at four spaces where the file has two. The first line's
/// shift is none, so the settled line was written back at four.
#[test]
fn lines_quoted_at_uneven_depths_are_each_re_indented_to_the_file() {
    let file = "  const a = 1;\n<<<<<<< yours\n  lines.push(`Grand total: ${t}`);\n=======\n  lines.push(`Sum: ${t}`);\n>>>>>>> origin/main\n  return a;\n";
    let old = "<<<<<<< yours\n    lines.push(`Grand total: ${t}`);\n=======\n    lines.push(`Sum: ${t}`);\n>>>>>>> origin/main";
    let new = "    lines.push(`Grand total: ${t}`);";
    assert_eq!(
        apply(file, old, new, false),
        Ok(edited(
            "  const a = 1;\n  lines.push(`Grand total: ${t}`);\n  return a;\n",
            1,
            Matched::Indentation
        ))
    );
}

/// A line deeper than any `old` quoted keeps its depth below the deepest
/// indentation that begins it, mapped as that one is.
#[test]
fn a_line_deeper_than_old_keeps_its_depth_under_the_nearest_mapping() {
    assert_eq!(
        apply(
            "if x {\n        y();\n}\n",
            "if x {\n    y();\n}\n",
            "if x {\n    y();\n        z();\n}\n",
            false
        ),
        Ok(edited(
            "if x {\n        y();\n            z();\n}\n",
            1,
            Matched::Indentation
        ))
    );
}

/// **A `new` made only of `old`'s own lines is never "already applied"**: a
/// deletion quoting its context, with a typo in the line to go, and a
/// re-indent quoted at the wrong depth both leave `new` standing in the file
/// whether or not they were made.
#[test]
fn an_edit_made_of_olds_own_lines_is_never_already_applied() {
    let file = "    x = f();\n    return y;\n";
    let typo = apply(file, "x = f( );\nreturn y;", "return y;", false);
    assert!(matches!(typo, Err(ReplaceError::Unmatched(_))), "{typo:?}");
    let reindent = apply("    x();\n", "  x();", "x();", false);
    assert!(
        matches!(reindent, Err(ReplaceError::Unmatched(_))),
        "{reindent:?}"
    );
}

/// **Overlapping line matches are ambiguous**: `}` `}` stands twice in three
/// closing braces.
#[test]
fn overlapping_line_matches_are_ambiguous() {
    let file = "    }\n    }\n    }\n";
    assert!(matches!(
        apply(file, "}\n}", "};\n}", false),
        Err(ReplaceError::Ambiguous(_))
    ));
}

/// **A quoted depth standing at two depths in the file is refused**: flat
/// Python cannot say where the new body belongs.
#[test]
fn a_quoted_depth_at_two_file_depths_is_refused() {
    let out = apply("if x:\n    foo()\n", "if x:\nfoo()", "if x:\nbar()", false);
    assert_eq!(
        out,
        Err(ReplaceError::Unmatched(
            "`old_text` matches the file only with its indentation ignored, and lines it quotes \
             at one indentation stand at different indentations in the file, so where \
             `new_text`'s lines belong cannot be worked out. Copy the lines to change exactly \
             as the file indents them"
                .to_string()
        ))
    );
}

/// **Deeper lines take the file's kind of indentation**: quoted in spaces
/// against a tab-indented file, a new nested block is written in tabs.
#[test]
fn deeper_lines_take_the_files_tabs() {
    assert_eq!(
        apply(
            "fn f() {\n\tlet x = 1;\n}\n",
            "fn f() {\n    let x = 1;\n}",
            "fn f() {\n    let x = 1;\n    if x {\n        y();\n    }\n}",
            false
        ),
        Ok(edited(
            "fn f() {\n\tlet x = 1;\n\tif x {\n\t\ty();\n\t}\n}\n",
            1,
            Matched::Indentation
        ))
    );
}

/// **Deleting a line quoted at the wrong indentation deletes it** — the kept
/// line's text standing in the file is not mistaken for the edit being done.
#[test]
fn a_deletion_quoted_at_the_wrong_indentation_deletes() {
    assert_eq!(
        apply("    a\n    b\n", "  a\n  b\n", "  b\n", false),
        Ok(edited("    b\n", 1, Matched::Indentation))
    );
}

/// **A mis-quoted `old` is not found, even when `new` happens to be in the
/// file**: as a substring, or as a line of nothing but punctuation.
#[test]
fn a_mis_quoted_old_is_not_found_whatever_new_is() {
    let file = "fn f() {\n    let total = 3;\n}\n";
    for new in ["3", "}", "total"] {
        assert!(
            matches!(
                apply(file, "let totl = 3;", new, false),
                Err(ReplaceError::Unmatched(_))
            ),
            "{new:?}"
        );
    }
}

/// Overlapping occurrences are two places, not one.
#[test]
fn overlapping_occurrences_are_ambiguous() {
    assert!(matches!(
        apply("aaa", "aa", "b", false),
        Err(ReplaceError::Ambiguous(_))
    ));
}

/// A mixed-ending file stays mixed: only the line replaced changes.
#[test]
fn a_mixed_ending_file_keeps_every_other_lines_ending() {
    assert_eq!(
        apply("a\r\nb\nc\r\nd\n", "b\n", "B\n", false),
        Ok(edited("a\r\nB\r\nc\r\nd\n", 1, Matched::Indentation))
    );
}

// ── Refusals ────────────────────────────────────────────────────────────────

#[test]
fn text_not_in_the_file_says_where_its_first_line_is() {
    assert_eq!(
        apply("a\nb\nc\n", "b\nX\n", "q\n", false),
        Err(ReplaceError::Unmatched(
            "`old_text` is not in the file — not exactly, and not with its indentation ignored. \
             Copy the lines to change exactly as the file has them, without the line numbers \
             `file_read` shows beside them. Its first line is line 2 of the file, but the lines \
             after it differ from the file's"
                .to_string()
        ))
    );
    assert_eq!(
        apply("a\nb\n", "zzz", "q", false),
        Err(ReplaceError::Unmatched(
            "`old_text` is not in the file — not exactly, and not with its indentation ignored. \
             Copy the lines to change exactly as the file has them, without the line numbers \
             `file_read` shows beside them. Its first line, `zzz`, is not in the file at all — \
             read the file again"
                .to_string()
        ))
    );
}

#[test]
fn an_empty_or_unchanging_edit_is_refused() {
    assert!(matches!(
        apply("a\n", "  \n", "b", false),
        Err(ReplaceError::Invalid(_))
    ));
    assert!(matches!(
        apply("a\n", "a", "a", false),
        Err(ReplaceError::Invalid(_))
    ));
}
