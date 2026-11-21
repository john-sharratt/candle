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
                    declared.entry(n).or_default().push(format!("{}.{}", i + 1, j + 1));
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
    let _ = writeln!(out, "# id\tlines\tbytes\tname\tuses");
    for (i, item) in layout.items.iter().enumerate() {
        let id = format!("{}", i + 1);
        line(&mut out, &id, &item.name, &item.span, source, &line_of, &declared);
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
        "{id}\t{}-{}\t{}\t{name}\t{tail}",
        line_number(line_of, span.start),
        line_number(line_of, span.end.saturating_sub(1)),
        span.len()
    );
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
        .filter(|w| w.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_'))
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
        let src = "use std::fmt;\n\npub struct Pool {\n    n: usize,\n}\n\n\
                   impl Pool {\n    fn grow(&mut self) {\n        self.n = width();\n    }\n}\n\n\
                   fn width() -> usize {\n    4\n}\n";
        let l = layout(src).unwrap();
        assert_eq!(
            render(src, &l),
            "# 3 items; 13 bytes stay in mod.rs (docs, uses, macros)\n\
             # id\tlines\tbytes\tname\tuses\n\
             1\t3-5\t33\tstruct Pool\t\n\
             2\t7-11\t70\timpl Pool\tPool, grow, width\n\
             \x20\x202.1\t8-10\t56\tfn grow\twidth\n\
             3\t13-15\t29\tfn width\t\n"
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
