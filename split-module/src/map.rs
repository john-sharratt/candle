//! The map a planner reads in place of the file: one line per movable item
//! and per member of a container, each with the names of the file's other
//! items it uses.
//!
//! An item is `N`; a member of a container is `N.M`. A plan names a whole
//! item by `N`, or its members one by one by `N.M` (`plan`).

use std::collections::{BTreeSet, HashMap};
use std::fmt::Write;
use std::ops::Range;

use crate::layout::Layout;

/// The most names one line lists as used.
const MAX_REFS: usize = 16;

/// The map of `layout` over `source`.
pub fn render(source: &str, layout: &Layout) -> String {
    let line_of = line_index(source);
    // Every bare name a reference can resolve to, and who declares it.
    let mut declared: HashMap<&str, Vec<String>> = HashMap::new();
    for (i, item) in layout.items.iter().enumerate() {
        if let Some(n) = bare_name(&item.name) {
            declared.entry(n).or_default().push(format!("{}", i + 1));
        }
        if let Some(body) = &item.body {
            for (j, m) in body.members.iter().enumerate() {
                if let Some(n) = bare_name(&m.name) {
                    declared
                        .entry(n)
                        .or_default()
                        .push(format!("{}.{}", i + 1, j + 1));
                }
            }
        }
    }
    let mut out = String::new();
    let kept_bytes: usize = layout.kept.iter().map(|k| k.span.len()).sum();
    let _ = writeln!(
        out,
        "# {} items; {} bytes stay in mod.rs (docs, uses, macros)",
        layout.items.len(),
        kept_bytes
    );
    // The file's own module doc, which each new file's doc is written from.
    let module_doc: Vec<&str> = layout
        .kept
        .iter()
        .flat_map(|k| source[k.span.clone()].lines())
        .map(str::trim)
        .filter_map(|l| l.strip_prefix("//!"))
        .map(str::trim)
        .collect();
    if !module_doc.is_empty() {
        let _ = writeln!(out, "# module doc:");
        for l in module_doc {
            let _ = writeln!(out, "#   {l}");
        }
    }
    let _ = writeln!(out, "# id\tlines\tbytes\tname\tuses\tdoc");
    for (i, item) in layout.items.iter().enumerate() {
        let id = format!("{}", i + 1);
        line(
            &mut out, &id, &item.name, &item.span, source, &line_of, &declared,
        );
        if let Some(body) = &item.body {
            for (j, m) in body.members.iter().enumerate() {
                let mid = format!("  {}.{}", i + 1, j + 1);
                // A `use` or inner attribute goes with every share, and is
                // never named.
                let name = if m.is_use {
                    format!("~ {}", m.name)
                } else {
                    m.name.clone()
                };
                line(&mut out, &mid, &name, &m.span, source, &line_of, &declared);
            }
        }
    }
    out
}

fn line(
    out: &mut String,
    id: &str,
    name: &str,
    span: &Range<usize>,
    source: &str,
    line_of: &[usize],
    declared: &HashMap<&str, Vec<String>>,
) {
    let own = bare_name(name);
    let uses: BTreeSet<&str> = identifiers(&source[span.clone()])
        .filter(|w| Some(*w) != own && declared.contains_key(w))
        .collect();
    let mut uses: Vec<&str> = uses.into_iter().collect();
    let more = uses.len().saturating_sub(MAX_REFS);
    uses.truncate(MAX_REFS);
    let mut tail = uses.join(", ");
    if more > 0 {
        let _ = write!(tail, " +{more}");
    }
    let _ = writeln!(
        out,
        "{id}\t{}-{}\t{}\t{name}\t{tail}\t{}",
        line_number(line_of, span.start),
        line_number(line_of, span.end.saturating_sub(1)),
        span.len(),
        summary(&source[span.clone()])
    );
}

/// The most characters of an item's doc the map shows.
const MAX_SUMMARY: usize = 110;

/// The first line of an item's doc comment, or of the comment over it when it
/// has no doc — what a planner reads of what the item is for.
fn summary(text: &str) -> String {
    let comments: Vec<&str> = text
        .lines()
        .map(str::trim)
        .take_while(|l| l.starts_with("//") || l.starts_with("#[") || l.is_empty())
        .filter(|l| l.starts_with("//"))
        .collect();
    let first = comments
        .iter()
        .find_map(|l| l.strip_prefix("///"))
        .or_else(|| {
            comments
                .iter()
                .map(|l| {
                    l.trim_start_matches('/')
                        .trim()
                        .trim_matches(|c| c == '─' || c == '-' || c == ' ')
                })
                .find(|l| !l.is_empty())
        })
        .unwrap_or("")
        .trim();
    first.chars().take(MAX_SUMMARY).collect()
}

/// The name a reference to an item is written as: `fn new` → `new`,
/// `struct Pool` → `Pool`; `None` for an `impl`, which is referred to by its
/// type.
pub fn bare_name(name: &str) -> Option<&str> {
    let (word, rest) = name.split_once(' ')?;
    matches!(
        word,
        "fn" | "struct" | "enum" | "union" | "trait" | "type" | "const" | "static"
    )
    .then_some(rest)
}

/// The identifier-shaped words of `text`.
pub fn identifiers(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|w| {
            w.chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        })
}

/// The byte offset each line starts at.
fn line_index(source: &str) -> Vec<usize> {
    std::iter::once(0)
        .chain(source.match_indices('\n').map(|(i, _)| i + 1))
        .collect()
}

/// The 1-based line `at` is on.
fn line_number(line_of: &[usize], at: usize) -> usize {
    line_of.partition_point(|&s| s <= at)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::layout;

    /// **One line per item and per member**, with its lines, its size, and
    /// the file's other items it uses — never itself.
    #[test]
    fn the_map_names_what_each_item_uses() {
        let src = "//! The pool.\n\nuse std::fmt;\n\n/// Blocks, grown on demand.\npub struct Pool {\n    n: usize,\n}\n\n\
                   impl Pool {\n    fn grow(&mut self) {\n        self.n = width();\n    }\n}\n\n\
                   // ── sizing ──\nfn width() -> usize {\n    4\n}\n";
        let l = layout(src).unwrap();
        assert_eq!(
            render(src, &l),
            "# 3 items; 27 bytes stay in mod.rs (docs, uses, macros)\n\
             # module doc:\n#   The pool.\n\
             # id\tlines\tbytes\tname\tuses\tdoc\n\
             1\t5-8\t62\tstruct Pool\t\tBlocks, grown on demand.\n\
             2\t10-14\t70\timpl Pool\tPool, grow, width\t\n\
             \x20\x202.1\t11-13\t56\tfn grow\twidth\t\n\
             3\t16-19\t53\tfn width\t\tsizing\n"
        );
    }

    #[test]
    fn bare_names_are_what_a_reference_writes() {
        assert_eq!(bare_name("fn new"), Some("new"));
        assert_eq!(bare_name("struct Pool"), Some("Pool"));
        assert_eq!(bare_name("impl Pool"), None);
        assert_eq!(bare_name("mod tests"), None);
    }
}
