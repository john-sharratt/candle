//! Parsing a file into a YAML tree, with the line of whatever is wrong.

use serde_yaml::{Error as YamlError, Value};

use crate::error::Problem;

/// A whole file. A syntax error names its line.
pub(crate) fn parse_doc(file: &str, text: &str) -> Result<Value, Problem> {
    if text.trim().is_empty() {
        return Err(Problem::new(file, None, "the file is empty"));
    }
    serde_yaml::from_str(text).map_err(|e| syntax(file, None, &e))
}

/// One line of a `lines.yaml` or archive: `- {…}`, a flow mapping on one
/// text line. `n` is the line's number in the file.
pub(crate) fn parse_flow_line(file: &str, n: usize, line: &str) -> Result<Value, Problem> {
    let body = line
        .strip_prefix("- ")
        .map(str::trim)
        .filter(|b| b.starts_with('{') && b.ends_with('}'))
        .ok_or_else(|| {
            Problem::new(
                file,
                Some(n),
                "each line is one entry, written `- {id: …, text: \"…\"}`",
            )
        })?;
    let v: Value = serde_yaml::from_str(body).map_err(|e| syntax(file, Some(n), &e))?;
    if !v.is_mapping() {
        return Err(Problem::new(file, Some(n), "the entry is not a mapping"));
    }
    Ok(v)
}

fn syntax(file: &str, line: Option<usize>, e: &YamlError) -> Problem {
    let at = line.or_else(|| e.location().map(|l| l.line()));
    // serde_yaml's message ends with its own "at line N column M", which
    // the problem already says.
    let msg = e.to_string();
    let msg = match msg.find(" at line ") {
        Some(i) => msg[..i].to_string(),
        None => msg,
    };
    Problem::new(file, at, format!("not valid YAML: {msg}"))
}

/// The 1-based line where top-level-or-nested key `key:` is first written.
pub(crate) fn key_line(text: &str, key: &str) -> Option<usize> {
    let want = format!("{key}:");
    text.lines()
        .position(|l| {
            let l = l.trim_start().trim_start_matches("- ");
            l.starts_with(&want) || l.starts_with(&format!("{{{want}"))
        })
        .map(|i| i + 1)
}

/// The 1-based line of the `index`th `- ` item after `key:`.
pub(crate) fn item_line(text: &str, key: &str, index: usize) -> Option<usize> {
    let start = key_line(text, key)?;
    text.lines()
        .enumerate()
        .skip(start)
        .filter(|(_, l)| l.trim_start().starts_with("- "))
        .nth(index)
        .map(|(i, _)| i + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_syntax_error_names_its_line() {
        let p = parse_doc("journal.yaml", "name: a\ncue: [unclosed\nfolded: false\n").unwrap_err();
        assert_eq!(p.file, "journal.yaml");
        assert!(p.line.is_some(), "{p}");
        assert!(p.what.starts_with("not valid YAML: "), "{p}");
        assert!(!p.what.contains(" at line "), "{p}");
    }

    #[test]
    fn an_empty_file_is_an_error() {
        assert_eq!(
            parse_doc("family.yaml", "  \n\n").unwrap_err().to_string(),
            "family.yaml: the file is empty"
        );
    }

    #[test]
    fn duplicate_keys_are_an_error() {
        assert!(parse_doc("f", "name: a\nname: b\n").is_err());
        assert!(parse_flow_line("f", 1, "- {id: a, id: b}").is_err());
    }

    #[test]
    fn a_flow_line_is_one_mapping_on_one_line() {
        assert!(parse_flow_line("l", 1, "- {id: a, text: \"x\"}").is_ok());
        for bad in [
            "{id: a}",
            "-{id: a}",
            "- id: a",
            "- [a, b]",
            "- {id: a",
            "- {id: a} # trailing",
            "  - {id: a}",
        ] {
            let p = parse_flow_line("l", 7, bad).unwrap_err();
            assert_eq!(p.line, Some(7), "{bad:?}: {p}");
        }
    }

    #[test]
    fn keys_and_items_are_found_by_line() {
        let text = "family: sharratt\nmembers:\n  - {person: john}\n  - {person: yen}\n";
        assert_eq!(key_line(text, "members"), Some(2));
        assert_eq!(key_line(text, "family"), Some(1));
        assert_eq!(key_line(text, "nobody"), None);
        assert_eq!(item_line(text, "members", 0), Some(3));
        assert_eq!(item_line(text, "members", 1), Some(4));
        assert_eq!(item_line(text, "members", 2), None);
    }
}
