//! `lines.yaml` and `archive/<period>.yaml`: a journal's lines, one entry to
//! a text line (§3.2). Blank lines and `#` comments a hand edit leaves are
//! read past; this crate never writes them.

use std::collections::BTreeSet;

use crate::error::Problem;
use crate::line::Line;
use crate::yaml::{parse_flow_line, Fields};

/// Every line of `text`, in file order. An unreadable entry or a repeated id
/// is an error naming its line.
pub fn read_lines(file: &str, text: &str) -> Result<Vec<Line>, Problem> {
    let mut out = Vec::new();
    let mut ids = BTreeSet::new();
    for (i, raw) in text.lines().enumerate() {
        let n = i + 1;
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // `[]` alone is an empty list: what a hand-made file may say.
        if trimmed == "[]" && out.is_empty() {
            continue;
        }
        let v = parse_flow_line(file, n, raw.trim_end())?;
        let mut f = Fields::of(&v, file, Some(n), None)?;
        let line = Line::read(&mut f)?;
        f.done()?;
        if !ids.insert(line.id.clone()) {
            return Err(Problem::new(
                file,
                Some(n),
                format!(
                    "the id {} is here twice — two edits of one line merged; keep one",
                    line.id
                ),
            ));
        }
        out.push(line);
    }
    Ok(out)
}

/// The file holding `lines`: one `- {…}` per entry and a final newline.
pub fn render_lines(lines: &[Line]) -> String {
    let mut out = String::new();
    for line in lines {
        out.push_str("- ");
        out.push_str(&line.render());
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::note;

    #[test]
    fn lines_round_trip_byte_for_byte() {
        let lines = vec![
            note("2026-10-30T09:10:02Z", "first"),
            note("2026-10-30T09:12:40Z", "second, with: punctuation"),
        ];
        let text = render_lines(&lines);
        assert_eq!(text.lines().count(), 2);
        assert!(text.ends_with("}\n"));
        assert_eq!(read_lines("lines.yaml", &text).unwrap(), lines);
        assert_eq!(
            render_lines(&read_lines("lines.yaml", &text).unwrap()),
            text
        );
    }

    #[test]
    fn blank_lines_comments_and_an_empty_list_are_read_past() {
        let line = note("2026-10-30T09:10:02Z", "kept");
        let text = format!("# by hand\n\n- {}\n   \n", line.render());
        assert_eq!(read_lines("l", &text).unwrap(), vec![line]);
        assert_eq!(read_lines("l", "[]\n").unwrap(), vec![]);
        assert_eq!(read_lines("l", "").unwrap(), vec![]);
        assert_eq!(render_lines(&[]), "");
    }

    #[test]
    fn crlf_endings_from_a_windows_edit_are_read() {
        let line = note("2026-10-30T09:10:02Z", "kept");
        let text = format!("- {}\r\n", line.render());
        assert_eq!(read_lines("l", &text).unwrap(), vec![line]);
    }

    #[test]
    fn a_repeated_id_is_the_union_merge_of_two_edits_and_an_error() {
        let a = note("2026-10-30T09:10:02Z", "one version");
        let mut b = a.clone();
        b.text = "the other version".into();
        let text = render_lines(&[a, b]);
        let err = read_lines("projects/candle/lines.yaml", &text).unwrap_err();
        assert_eq!(err.line, Some(2));
        assert!(err.what.contains("twice"), "{err}");
    }

    #[test]
    fn a_multi_line_entry_is_an_error_at_its_first_line() {
        let text = "- id: 20261030-0912-7e3a\n  text: x\n";
        let err = read_lines("l", text).unwrap_err();
        assert_eq!(err.line, Some(1));
    }

    #[test]
    fn the_first_bad_line_is_named_whatever_follows() {
        let good = note("2026-10-30T09:10:02Z", "kept");
        let text = format!("- {}\n- {{id: nope}}\n- {}\n", good.render(), good.render());
        let err = read_lines("l", &text).unwrap_err();
        assert_eq!(err.line, Some(2));
        assert!(err.what.contains("not a line id"), "{err}");
    }
}
