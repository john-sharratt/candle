//! The line-structured carves: paragraphs of prose, Markdown's paragraphs,
//! and YAML's lines.

use super::types::{ChunkKind, Scope};
use super::{blank, trimmed};

/// The paragraphs of `lines[start..=end]` (1-indexed): each run of non-blank
/// lines between blank lines.
pub fn runs(lines: &[&str], start: u32, end: u32) -> Vec<Scope> {
    let mut out = Vec::new();
    let mut l = start;
    while l <= end {
        if blank(lines[l as usize - 1]) {
            l += 1;
            continue;
        }
        let s = l;
        while l < end && !blank(lines[l as usize]) {
            l += 1;
        }
        out.push(paragraph(s, l));
        l += 1;
    }
    out
}

/// Every non-blank line its own unit.
pub fn lines(lines: &[&str]) -> Vec<Scope> {
    (1..=lines.len() as u32)
        .filter(|&l| !blank(lines[l as usize - 1]))
        .map(|l| Scope {
            path: Vec::new(),
            kind: ChunkKind::Line,
            start_line: l,
            end_line: l,
        })
        .collect()
}

/// Markdown's units: paragraphs, with
///
/// - **YAML front matter** — a first line of `---` through the next `---` —
///   one unit;
/// - **a fenced code block** (```` ``` ```` or `~~~`) one unit, blank lines and
///   all, so a blank line inside a fence never splits it;
/// - **a heading** its own unit even when text follows it without a blank line.
pub fn markdown(lines: &[&str]) -> Vec<Scope> {
    let n = lines.len() as u32;
    let line = |l: u32| lines[l as usize - 1];
    let mut out = Vec::new();
    let mut l = 1;

    if n > 0 && line(1).trim() == "---" {
        if let Some(close) = (2..=n).find(|&k| line(k).trim() == "---") {
            out.push(paragraph(1, close));
            l = close + 1;
        }
    }

    while l <= n {
        if blank(line(l)) {
            l += 1;
            continue;
        }
        if heading(line(l)) {
            out.push(paragraph(l, l));
            l += 1;
            continue;
        }
        let s = l;
        let mut e = l;
        while e <= n {
            let text = line(e);
            if let Some(marker) = fence(text) {
                // The fence and everything to its close is part of this run.
                let close = (e + 1..=n)
                    .find(|&k| line(k).trim_start().starts_with(marker))
                    .unwrap_or(n);
                e = close + 1;
                continue;
            }
            if blank(text) || (e > s && heading(text)) {
                break;
            }
            e += 1;
        }
        if let Some((a, b)) = trimmed(lines, s, e - 1) {
            out.push(paragraph(a, b));
        }
        l = e;
    }
    out
}

fn paragraph(start: u32, end: u32) -> Scope {
    Scope {
        path: Vec::new(),
        kind: ChunkKind::Paragraph,
        start_line: start,
        end_line: end,
    }
}

/// An ATX heading: one to six `#` and then a space or the end of the line.
fn heading(line: &str) -> bool {
    let t = line.trim_start();
    let hashes = t.chars().take_while(|&c| c == '#').count();
    (1..=6).contains(&hashes) && t[hashes..].chars().next().is_none_or(|c| c == ' ')
}

/// The fence marker a line opens a code block with, if it does.
fn fence(line: &str) -> Option<&'static str> {
    let t = line.trim_start();
    if t.starts_with("```") {
        Some("```")
    } else if t.starts_with("~~~") {
        Some("~~~")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::super::split_lines;
    use super::*;

    fn spans(units: &[Scope]) -> Vec<(u32, u32)> {
        units.iter().map(|u| (u.start_line, u.end_line)).collect()
    }

    #[test]
    fn prose_is_carved_into_paragraphs() {
        let src = "One.\nStill one.\n\nTwo.\n\n\nThree.";
        let lines = split_lines(src);
        assert_eq!(
            spans(&runs(&lines, 1, lines.len() as u32)),
            vec![(1, 2), (4, 4), (7, 7)]
        );
    }

    #[test]
    fn a_heading_is_its_own_unit() {
        let src = "# The Body\nYou woke.\nThe tank hummed.\n\n## Later\n\nNothing.\n";
        assert_eq!(
            spans(&markdown(&split_lines(src))),
            vec![(1, 1), (2, 3), (5, 5), (7, 7)]
        );
    }

    #[test]
    fn a_fenced_block_with_blank_lines_is_one_unit() {
        let src = "Before.\n\n```rust\nfn a() {}\n\nfn b() {}\n```\n\nAfter.\n";
        assert_eq!(
            spans(&markdown(&split_lines(src))),
            vec![(1, 1), (3, 7), (9, 9)]
        );
    }

    #[test]
    fn front_matter_is_one_unit() {
        let src = "---\ntitle: x\n\ntags: [a]\n---\n\nBody.\n";
        assert_eq!(spans(&markdown(&split_lines(src))), vec![(1, 5), (7, 7)]);
    }

    #[test]
    fn a_scene_break_between_blank_lines_is_its_own_unit() {
        let src = "End of one.\n\n* * *\n\nStart of two.\n";
        assert_eq!(
            spans(&markdown(&split_lines(src))),
            vec![(1, 1), (3, 3), (5, 5)]
        );
    }

    #[test]
    fn hashes_without_a_space_are_not_a_heading() {
        assert!(heading("# x"));
        assert!(heading("###"));
        assert!(!heading("#hashtag"));
        assert!(!heading("####### seven"));
    }

    #[test]
    fn yaml_is_carved_into_lines() {
        let src = "name: Bram\n\nmood:\n  calm: 2\n";
        assert_eq!(
            spans(&lines(&split_lines(src))),
            vec![(1, 1), (3, 3), (4, 4)]
        );
    }
}
