//! A kernel source laid out for a split: its header comment, its units, and
//! the containers big enough to open; and the map a planner reads.
//!
//! A unit a plan can place is named by its number, `N`; a container opened by
//! the layout — an `extern "C" { }`, a namespace, an `#if` with no `#else`,
//! whose body is larger than [`OPEN_BYTES`] — is placed member by member,
//! `N.M`, and its opening and closing lines stay in the file. So do the
//! `#include` and `#pragma once` lines, marked `~` in the map.

use std::ops::Range;

use super::scan::{units, Kind, Unit};

/// A container whose body is larger than this is opened, so its members can
/// go to different files.
pub const OPEN_BYTES: usize = 16 * 1024;

/// One top-level unit, and its members when the layout opens it.
#[derive(Debug, Clone)]
pub struct Item {
    pub unit: Unit,
    pub members: Option<Vec<Unit>>,
}

/// A kernel source, laid out.
#[derive(Debug, Clone)]
pub struct Layout {
    /// The file's own header comment, which stays with it.
    pub header: Range<usize>,
    pub items: Vec<Item>,
}

/// Lay `src` out.
pub fn layout(src: &str) -> Result<Layout, String> {
    let header = 0..header_end(src);
    let mut items = Vec::new();
    for unit in units(src, header.end..src.len())? {
        let members = match &unit.body {
            Some(body) if body.len() > OPEN_BYTES => Some(units(src, body.clone())?),
            _ => None,
        };
        items.push(Item { unit, members });
    }
    Ok(Layout { header, items })
}

/// The end of the comment lines a file opens with, when a blank line parts
/// them from what follows; 0 when it opens with anything else.
fn header_end(src: &str) -> usize {
    let mut at = 0;
    let mut in_block = false;
    let mut seen = false;
    for line in src.split_inclusive('\n') {
        let t = line.trim();
        if in_block {
            in_block = !t.contains("*/");
            at += line.len();
            continue;
        }
        if t.starts_with("//") {
            seen = true;
            at += line.len();
        } else if t.starts_with("/*") {
            seen = true;
            in_block = !t.contains("*/");
            at += line.len();
        } else if t.is_empty() && seen {
            return at;
        } else {
            return 0;
        }
    }
    0
}

/// The 1-based line of byte `at`.
fn line_of(src: &str, at: usize) -> usize {
    src[..at].matches('\n').count() + 1
}

/// The map of `src` a planner reads in place of the file.
pub fn render(src: &str, l: &Layout) -> String {
    let mut out = String::new();
    let doc: Vec<&str> = src[l.header.clone()]
        .lines()
        .map(|t| {
            t.trim()
                .trim_start_matches("//")
                .trim_start_matches("/*")
                .trim()
        })
        .filter(|t| !t.is_empty())
        .collect();
    out.push_str(&format!(
        "# {} units; a line marked ~ stays in the file, and so do an opened block's own lines\n# file doc:\n",
        l.items.len()
    ));
    for d in doc {
        out.push_str(&format!("#   {d}\n"));
    }
    out.push_str("# id\tlines\tbytes\tname\tdoc\n");
    let row = |out: &mut String, id: String, u: &Unit, opened: bool| {
        let mark = if u.kind == Kind::Kept { "~ " } else { "" };
        out.push_str(&format!(
            "{mark}{id}\t{}-{}\t{}\t{}{}\t{}\n",
            line_of(src, u.span.start),
            line_of(src, u.span.end.saturating_sub(1)),
            u.span.len(),
            u.name,
            if opened {
                " (opened: place its members)"
            } else {
                ""
            },
            u.doc
        ));
    };
    for (i, item) in l.items.iter().enumerate() {
        let n = i + 1;
        row(&mut out, n.to_string(), &item.unit, item.members.is_some());
        if let Some(members) = &item.members {
            for (j, m) in members.iter().enumerate() {
                let mut indented = String::new();
                row(&mut indented, format!("{n}.{}", j + 1), m, false);
                out.push_str("  ");
                out.push_str(&indented);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The header stays, big blocks open, and the map marks what stays.**
    #[test]
    fn a_kernel_is_laid_out_and_mapped() {
        let big: String = (0..800)
            .map(|i| format!("__device__ int v{i};\n"))
            .collect();
        let src = format!(
            "// The kernels.\n//\n// All of them.\n\n#pragma once\n#include \"a.cuh\"\n\n// One.\nint a;\n\nextern \"C\" {{\n{big}}}\n"
        );
        let l = layout(&src).unwrap();
        assert_eq!(
            &src[l.header.clone()],
            "// The kernels.\n//\n// All of them.\n"
        );
        assert_eq!(l.items.len(), 4);
        assert_eq!(l.items[3].members.as_ref().unwrap().len(), 800);
        let map = render(&src, &l);
        assert!(map.contains("#   The kernels.\n#   All of them.\n"));
        assert!(map.contains("~ 1\t5-5\t13\t#pragma once\t\n"));
        assert!(map.contains("3\t8-9\t15\ta\tOne.\n"), "{map}");
        assert!(map.contains("extern \"C\" (opened: place its members)"));
        assert!(map.contains("  4.800\t"));
    }
}
