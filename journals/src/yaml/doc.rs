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
            parse_doc("group.yaml", "  \n\n").unwrap_err().to_string(),
            "group.yaml: the file is empty"
        );
    }

    #[test]
    fn duplicate_keys_are_an_error() {
        assert!(parse_doc("f", "name: a\nname: b\n").is_err());
        assert!(parse_doc("f", "{name: a, name: b}\n").is_err());
    }

    #[test]
    fn keys_and_items_are_found_by_line() {
        let text = "group: sharratt\nmembers:\n  - {person: john}\n  - {person: yen}\n";
        assert_eq!(key_line(text, "members"), Some(2));
        assert_eq!(key_line(text, "group"), Some(1));
        assert_eq!(key_line(text, "nobody"), None);
        assert_eq!(item_line(text, "members", 0), Some(3));
        assert_eq!(item_line(text, "members", 1), Some(4));
        assert_eq!(item_line(text, "members", 2), None);
    }
}
