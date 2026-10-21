//! The files a dependency analysis says its file depends on — read from the
//! answer's closing section, which its prompt asks for in a fixed shape:
//!
//! ```text
//! DEPENDS ON:
//! - candle/zend/src/session.rs
//! - candle/zend/src/tools.rs
//! ```
//!
//! Only files the branches hold are kept: a name that matches none — a crate,
//! a type, a guess — is no edge a change can be followed along. A path written
//! inside its repository (`zend/src/session.rs`) is taken for the one file
//! that ends that way, and dropped when none or several do.

use std::collections::BTreeSet;

/// The heading the section opens with.
pub const HEADING: &str = "DEPENDS ON";

/// The files `answer` names under its last [`HEADING`], among `known`
/// (workspace-relative paths), never `own` itself.
pub fn edges<'k>(answer: &str, own: &str, known: &[&'k str]) -> Vec<&'k str> {
    let lines: Vec<&str> = answer.lines().collect();
    let Some(start) = lines.iter().rposition(|l| is_heading(l)) else {
        return Vec::new();
    };
    let mut out = BTreeSet::new();
    for line in &lines[start + 1..] {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let Some(item) = t
            .strip_prefix("- ")
            .or_else(|| t.strip_prefix("* "))
            .or_else(|| t.strip_prefix('-'))
        else {
            break; // the list ended
        };
        if let Some(path) = resolve(clean(item), known) {
            if path != own {
                out.insert(path);
            }
        }
    }
    out.into_iter().collect()
}

fn is_heading(line: &str) -> bool {
    let t = line
        .trim()
        .trim_matches(|c| c == '*' || c == '#' || c == ' ');
    let t = t
        .strip_suffix(':')
        .unwrap_or(t)
        .trim_end_matches('*')
        .trim();
    t.eq_ignore_ascii_case(HEADING)
}

/// The path an item names: its first word, without quoting or a trailing
/// remark.
fn clean(item: &str) -> &str {
    let item = item.trim().trim_start_matches(['`', '"', '\'', '*']);
    let end = item
        .find(|c: char| {
            c.is_whitespace() || matches!(c, '`' | '"' | '\'' | '*' | ',' | ';' | ')' | '(')
        })
        .unwrap_or(item.len());
    item[..end]
        .trim_end_matches(['.', ':'])
        .trim_start_matches("./")
}

fn resolve<'k>(name: &str, known: &[&'k str]) -> Option<&'k str> {
    if name.is_empty() {
        return None;
    }
    if let Some(k) = known.iter().find(|k| **k == name) {
        return Some(k);
    }
    let suffix = format!("/{name}");
    let mut hits = known.iter().filter(|k| k.ends_with(&suffix));
    match (hits.next(), hits.next()) {
        (Some(k), None) => Some(k),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KNOWN: &[&str] = &[
        "candle/zend/src/session.rs",
        "candle/zend/src/tools.rs",
        "candle/zend/src/lib.rs",
        "candle/host/src/lib.rs",
    ];

    #[test]
    fn the_last_section_names_the_known_files() {
        let answer = "It reads the session.\n\nDEPENDS ON:\n- candle/zend/src/session.rs\n\
                      - `zend/src/tools.rs` (for the registry)\n- serde\n- candle/zend/src/lib.rs\n";
        assert_eq!(
            edges(answer, "candle/zend/src/lib.rs", KNOWN),
            ["candle/zend/src/session.rs", "candle/zend/src/tools.rs"]
        );
    }

    /// An ambiguous repository-relative name is no edge; nor is anything past
    /// the list's end.
    #[test]
    fn an_ambiguous_name_and_trailing_prose_are_dropped() {
        let answer = "**Depends on:**\n* src/lib.rs\n* candle/zend/src/tools.rs\n\nThat is all.\n- candle/zend/src/session.rs";
        assert_eq!(edges(answer, "x", KNOWN), ["candle/zend/src/tools.rs"]);
    }

    #[test]
    fn with_no_section_there_are_no_edges() {
        assert!(edges("candle/zend/src/tools.rs is used.", "x", KNOWN).is_empty());
    }
}
