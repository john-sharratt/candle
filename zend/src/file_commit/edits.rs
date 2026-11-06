//! Exact edits to a text file (`docs/mcp_file_tools.md` §3.1): each `old`
//! must occur in the file exactly once, as it stands after the edits before
//! it, and is replaced by `new`. An edit that matches nothing or more than
//! once fails the whole call, naming which.

/// One `old` → `new` replacement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edit {
    pub old: String,
    pub new: String,
}

/// Why an edit did not apply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Missed {
    /// The edit's `old` is empty, which matches everywhere.
    Empty { index: usize },
    /// `old` is not in the file.
    NotFound { index: usize },
    /// `old` is in the file more than once, so which is meant is unknown.
    Ambiguous { index: usize, count: usize },
}

impl Missed {
    /// The refusal, for `path`.
    pub fn describe(&self, path: &str) -> String {
        match self {
            Missed::Empty { index } => {
                format!("{path}: edit {index} has an empty `old`, which names no place in the file")
            }
            Missed::NotFound { index } => format!(
                "{path}: edit {index}'s `old` is not in the file as it stands after the edits \
                 before it — read the file again and copy the text exactly"
            ),
            Missed::Ambiguous { index, count } => format!(
                "{path}: edit {index}'s `old` occurs {count} times — include enough of the \
                 surrounding text to name one place"
            ),
        }
    }
}

/// `text` with `edits` applied in order.
pub fn apply(text: &str, edits: &[Edit]) -> Result<String, Missed> {
    let mut out = text.to_string();
    for (index, edit) in edits.iter().enumerate() {
        let index = index + 1;
        if edit.old.is_empty() {
            return Err(Missed::Empty { index });
        }
        let count = out.matches(edit.old.as_str()).count();
        match count {
            0 => return Err(Missed::NotFound { index }),
            1 => out = out.replacen(edit.old.as_str(), &edit.new, 1),
            count => return Err(Missed::Ambiguous { index, count }),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edit(old: &str, new: &str) -> Edit {
        Edit {
            old: old.into(),
            new: new.into(),
        }
    }

    #[test]
    fn edits_apply_in_order_each_against_the_result_of_the_last() {
        let text = "colour: red;\nsize: 12px;\n";
        assert_eq!(
            apply(
                text,
                &[
                    edit("red", "var(--accent)"),
                    edit("var(--accent);\n", "var(--accent);\nweight: 600;\n")
                ]
            ),
            Ok("colour: var(--accent);\nweight: 600;\nsize: 12px;\n".to_string())
        );
    }

    #[test]
    fn an_edit_must_name_exactly_one_place() {
        let text = "a b a";
        assert_eq!(
            apply(text, &[edit("c", "d")]),
            Err(Missed::NotFound { index: 1 })
        );
        assert_eq!(
            apply(text, &[edit("b", "c"), edit("a", "z")]),
            Err(Missed::Ambiguous { index: 2, count: 2 })
        );
        assert_eq!(
            apply(text, &[edit("", "x")]),
            Err(Missed::Empty { index: 1 })
        );
        assert_eq!(apply(text, &[edit("a b", "x")]), Ok("x a".to_string()));
    }

    /// **Vietnamese text edits byte for byte** — multi-byte characters are
    /// matched and replaced whole.
    #[test]
    fn multibyte_text_is_matched_and_replaced_whole() {
        let text = "<h1>Công thức</h1>\n";
        assert_eq!(
            apply(text, &[edit("Công thức", "Món ăn — ngon")])
                .unwrap()
                .as_bytes(),
            "<h1>Món ăn — ngon</h1>\n".as_bytes()
        );
    }

    #[test]
    fn a_refusal_names_the_file_and_the_edit() {
        assert_eq!(
            Missed::Ambiguous { index: 2, count: 3 }.describe("apps/a/pages/home.js"),
            "apps/a/pages/home.js: edit 2's `old` occurs 3 times — include enough of the \
             surrounding text to name one place"
        );
    }
}
