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
    let repo = own.split('/').next().unwrap_or("");
    let mut out = BTreeSet::new();
    for line in &lines[start + 1..] {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let Some(item) = bullet(t) else {
            break; // the list ended
        };
        if let Some(path) = resolve(clean(item), repo, known) {
            if path != own {
                out.insert(path);
            }
        }
    }
    out.into_iter().collect()
}

/// A list item's text: after `- `, `* `, `-`, or a number (`1.`, `1)`).
fn bullet(line: &str) -> Option<&str> {
    if let Some(item) = line
        .strip_prefix("- ")
        .or_else(|| line.strip_prefix("* "))
        .or_else(|| line.strip_prefix('-'))
    {
        return Some(item);
    }
    let digits = line.find(|c: char| !c.is_ascii_digit())?;
    if digits == 0 {
        return None;
    }
    line[digits..]
        .strip_prefix(". ")
        .or_else(|| line[digits..].strip_prefix(") "))
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

/// The path an item names: a Markdown link's target, else its first word,
/// without quoting, a line number (`:12`, `:12-30`) or a trailing remark.
fn clean(item: &str) -> &str {
    let item = item.trim();
    let item = match (item.find("]("), item.starts_with('[')) {
        (Some(i), true) => &item[i + 2..],
        _ => item,
    };
    let item = item.trim_start_matches(['`', '"', '\'', '*']);
    let end = item
        .find(|c: char| {
            c.is_whitespace() || matches!(c, '`' | '"' | '\'' | '*' | ',' | ';' | ')' | '(' | '#')
        })
        .unwrap_or(item.len());
    let path = item[..end].trim_end_matches(['.', ':']);
    // `path:12` and `path:12-30` name lines of the file.
    let path = match path.rsplit_once(':') {
        Some((p, lines))
            if !lines.is_empty() && lines.chars().all(|c| c.is_ascii_digit() || c == '-') =>
        {
            p
        }
        _ => path,
    };
    path.trim_start_matches("./")
}

/// The known file `name` is: itself, else the one file in the unit's own
/// repository `repo` it names, else the one file anywhere that ends with it.
fn resolve<'k>(name: &str, repo: &str, known: &[&'k str]) -> Option<&'k str> {
    if name.is_empty() {
        return None;
    }
    let in_repo = format!("{repo}/{name}");
    if let Some(k) = known.iter().find(|k| **k == in_repo) {
        return Some(k);
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

    /// Numbered items, Markdown links and line numbers name files too, and a
    /// name inside the unit's own repository is taken from it first.
    #[test]
    fn numbered_links_and_line_numbers_name_their_files() {
        let answer = "DEPENDS ON:\n1. candle/zend/src/session.rs:12-30\n2) [tools](candle/zend/src/tools.rs)\n\
                      3. `zend/src/lib.rs:4`\n";
        assert_eq!(
            edges(answer, "candle/host/Cargo.toml", KNOWN),
            [
                "candle/zend/src/lib.rs",
                "candle/zend/src/session.rs",
                "candle/zend/src/tools.rs"
            ]
        );
    }

    #[test]
    fn with_no_section_there_are_no_edges() {
        assert!(edges("candle/zend/src/tools.rs is used.", "x", KNOWN).is_empty());
    }
}
