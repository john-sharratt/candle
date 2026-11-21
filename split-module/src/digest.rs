//! What a doc writer reads of a module folder in place of its code: each
//! file's items, with every doc comment on them and every comment inside them.
//!
//! A file's own comments are where its reasons are written — why a table is
//! cleared after a failed launch, why a row's dials come from a per-row array —
//! and a module doc written from them says why the file exists, not only what
//! it holds. The code itself is left out: the comments are a fraction of its
//! size, and they are what a doc can truthfully claim.

use std::fs;
use std::path::Path;

use crate::layout::layout;

/// The digest of every `.rs` file in `dir` but `mod.rs`, by file name.
pub fn digest_dir(dir: &Path) -> Result<String, String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".rs") && n != "mod.rs")
        .collect();
    names.sort();
    let mut out = String::new();
    for name in names {
        let path = dir.join(&name);
        let source = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        out.push_str(&format!("=== {name}\n"));
        out.push_str(&digest(&source)?);
        out.push('\n');
    }
    Ok(out)
}

/// One file's digest: each item's name, then its comments, a line each.
pub fn digest(source: &str) -> Result<String, String> {
    let l = layout(source)?;
    let mut out = String::new();
    for item in &l.items {
        match &item.body {
            Some(body) => {
                out.push_str(&format!("{}\n", item.name));
                push_comments(&mut out, &source[body.head.clone()], "  ");
                for m in body.members.iter().filter(|m| !m.is_use) {
                    out.push_str(&format!("  {}\n", m.name));
                    push_comments(&mut out, &source[m.span.clone()], "    ");
                }
            }
            None => {
                out.push_str(&format!("{}\n", item.name));
                push_comments(&mut out, &source[item.span.clone()], "  ");
            }
        }
    }
    Ok(out)
}

/// Every comment line of `text`, its comment marker taken off, indented by
/// `indent`. A blank comment line, a paragraph break, is kept as an empty one
/// only between two that say something.
fn push_comments(out: &mut String, text: &str, indent: &str) {
    let mut pending_break = false;
    let mut any = false;
    for line in text.lines().map(str::trim) {
        let Some(rest) = line.strip_prefix("//") else {
            continue;
        };
        let words = rest.trim_start_matches(['/', '!']).trim();
        if words.is_empty() {
            pending_break = any;
            continue;
        }
        if pending_break {
            out.push_str(indent);
            out.push('\n');
            pending_break = false;
        }
        out.push_str(indent);
        out.push_str(words);
        out.push('\n');
        any = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A digest is each item's name and every comment on it and in it** —
    /// docs and inline comments alike, no code.
    #[test]
    fn a_digest_keeps_the_comments_and_drops_the_code() {
        let src = "//! The file's own doc.\n\nuse super::*;\n\n/// A pool.\n///\n/// Grown in place.\npub struct Pool {\n    \
                   /// How many.\n    n: usize,\n}\n\nimpl Pool {\n    /// Grow it.\n    fn grow(&mut self) {\n        \
                   // A grow never moves a block.\n        self.n += 1;\n    }\n}\n";
        assert_eq!(
            digest(src).unwrap(),
            "struct Pool\n  A pool.\n  \n  Grown in place.\n  How many.\nimpl Pool\n  fn grow\n    Grow it.\n    \
             A grow never moves a block.\n"
        );
    }
}
