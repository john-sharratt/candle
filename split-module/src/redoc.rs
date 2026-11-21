//! Writing new module docs over a folder's files: each file's leading `//!`
//! block, and nothing else, is replaced by the doc written for it.
//!
//! The docs come in one text, a file each under `=== <name>.rs`, as `//!`
//! lines. A file the text names that is not in the folder, or a doc with no
//! `//!` line, is refused before any file is touched; each file is then written
//! whole under a temporary name and renamed into place, as `write` does.

use std::fs;
use std::path::Path;

/// Each file's name and its new doc, as `//!` lines.
pub fn parse_docs(text: &str) -> Result<Vec<(String, Vec<String>)>, String> {
    let mut docs: Vec<(String, Vec<String>)> = Vec::new();
    // Where the lines under the last `===` go; `None` after a bare `===`,
    // which ends the docs rather than naming a file.
    let mut current: Option<usize> = None;
    for (n, raw) in text.lines().enumerate() {
        let line = raw.trim_end();
        if let Some(name) = line.strip_prefix("===") {
            let name = name.trim();
            current = if name.is_empty() {
                None
            } else if let Some(at) = docs.iter().position(|(d, _)| d == name) {
                // A file written twice keeps the later doc.
                docs[at].1.clear();
                Some(at)
            } else {
                docs.push((name.to_string(), Vec::new()));
                Some(docs.len() - 1)
            };
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let at =
            current.ok_or_else(|| format!("line {}: a doc line before any `=== <file>`", n + 1))?;
        let lines = &mut docs[at].1;
        if !line.trim_start().starts_with("//!") {
            return Err(format!("line {}: not a `//!` line: {line}", n + 1));
        }
        lines.push(line.trim_start().to_string());
    }
    if let Some((name, _)) = docs.iter().find(|(_, l)| l.is_empty()) {
        return Err(format!("{name} has no doc"));
    }
    Ok(docs)
}

/// `source` with its leading `//!` block replaced by `doc`, written in the
/// file's own line endings. The block is measured with its line endings, so a
/// `\r\n` file loses exactly its doc lines and nothing after them.
pub fn replace_doc(source: &str, doc: &[String]) -> String {
    let body_at: usize = source
        .split_inclusive('\n')
        .take_while(|l| l.trim_start().starts_with("//!"))
        .map(str::len)
        .sum();
    let rest = source[body_at..].trim_start_matches(['\n', '\r']);
    let nl = if source.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    };
    let mut out = doc.join(nl);
    out.push_str(nl);
    out.push_str(nl);
    out.push_str(rest);
    out
}

/// Write `docs` over the files of `dir`. Returns the files written.
pub fn redoc(dir: &Path, docs: &[(String, Vec<String>)]) -> Result<Vec<String>, String> {
    let mut next: Vec<(String, String)> = Vec::new();
    for (name, doc) in docs {
        let path = dir.join(name);
        let source = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        next.push((name.clone(), replace_doc(&source, doc)));
    }
    for (name, text) in &next {
        let tmp = dir.join(format!("{name}.redoc-tmp"));
        fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    }
    for (name, _) in &next {
        let tmp = dir.join(format!("{name}.redoc-tmp"));
        fs::rename(&tmp, dir.join(name)).map_err(|e| format!("{name}: {e}"))?;
    }
    Ok(next.into_iter().map(|(n, _)| n).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A bare `===` ends the docs, and a file written twice keeps its
    /// later doc.**
    #[test]
    fn a_bare_head_ends_and_a_repeat_replaces() {
        let d = parse_docs("=== a.rs\n//! One.\n=== a.rs\n//! Two.\n===\n").unwrap();
        assert_eq!(d, vec![("a.rs".to_string(), vec!["//! Two.".to_string()])]);
    }

    #[test]
    fn docs_parse_by_file() {
        let d = parse_docs("=== a.rs\n//! A.\n//!\n//! More.\n\n=== b.rs\n//! B.\n").unwrap();
        assert_eq!(
            d,
            vec![
                (
                    "a.rs".to_string(),
                    vec!["//! A.".into(), "//!".into(), "//! More.".into()]
                ),
                ("b.rs".to_string(), vec!["//! B.".into()]),
            ]
        );
        assert_eq!(
            parse_docs("//! x\n").unwrap_err(),
            "line 1: a doc line before any `=== <file>`"
        );
        assert_eq!(
            parse_docs("=== a.rs\nfn x() {}\n").unwrap_err(),
            "line 2: not a `//!` line: fn x() {}"
        );
        assert_eq!(parse_docs("=== a.rs\n").unwrap_err(), "a.rs has no doc");
    }

    /// **Only the leading `//!` block changes**; a file with none gains one.
    #[test]
    fn only_the_module_doc_is_replaced() {
        let new = vec!["//! New.".to_string(), "//!".into(), "//! Why.".into()];
        assert_eq!(
            replace_doc(
                "//! Old.\n//! Old 2.\n\nuse super::*;\n\n//! not a module doc\nfn a() {}\n",
                &new
            ),
            "//! New.\n//!\n//! Why.\n\nuse super::*;\n\n//! not a module doc\nfn a() {}\n"
        );
        assert_eq!(
            replace_doc("use x;\n", &new),
            "//! New.\n//!\n//! Why.\n\nuse x;\n"
        );
    }

    /// **A `\r\n` file loses exactly its doc lines**, and keeps its endings:
    /// measured without them, a five-line doc stopped five bytes short and left
    /// the tail of its last line behind as code.
    #[test]
    fn a_crlf_files_doc_is_replaced_whole() {
        let new = vec!["//! New.".to_string(), "//!".into(), "//! Why.".into()];
        assert_eq!(
            replace_doc(
                "//! Old one,\r\n//! and its end.\r\n\r\nuse super::*;\r\n",
                &new
            ),
            "//! New.\r\n//!\r\n//! Why.\r\n\r\nuse super::*;\r\n"
        );
    }
}
