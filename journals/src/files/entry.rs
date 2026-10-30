//! A journal's entries: each line its own file, `<id>.yaml`, in its journal's
//! folder beside `journal.yaml` and the sub-journals' folders — or, once its
//! period has folded, in `archive/<period>/` (§3.2). The file's name is the
//! line's id, so a folder lists its entries in the order they were written,
//! and two writers adding entries at once add two files that never conflict.

use crate::error::Problem;
use crate::line::{Line, LineId};
use crate::yaml::{parse_doc, Fields};

/// The file name the line `id` is written to.
pub fn entry_file(id: &LineId) -> String {
    format!("{id}.yaml")
}

/// The id a file named `name` holds the line of, when it is an entry's.
pub fn entry_id(name: &str) -> Option<LineId> {
    name.strip_suffix(".yaml")
        .and_then(|stem| LineId::parse(stem).ok())
}

/// The line `id` read from its file at `path`.
pub fn read_entry(path: &str, id: LineId, text: &str) -> Result<Line, Problem> {
    let v = parse_doc(path, text)?;
    let mut f = Fields::of(&v, path, None, Some(text))?;
    let line = Line::read(&mut f, id)?;
    f.done()?;
    Ok(line)
}

/// Lines in the order they happened: by time, and by id within a second.
pub fn in_order(lines: &mut [Line]) {
    lines.sort_by(|a, b| (a.at, &a.id).cmp(&(b.at, &b.id)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::note;

    #[test]
    fn an_entry_is_its_own_file_named_for_its_id() {
        let line = note("2026-10-30T09:10:02Z", "first");
        let name = entry_file(&line.id);
        assert!(
            name.starts_with("2026-10-30-0910-") && name.ends_with(".yaml"),
            "{name}"
        );
        assert_eq!(entry_id(&name), Some(line.id.clone()));
        let path = format!("groups/sharratt/topics/{name}");
        assert_eq!(
            read_entry(&path, line.id.clone(), &line.render()).unwrap(),
            line
        );
        for not in [
            "journal.yaml",
            "README.md",
            "2026-10-30-0910.yaml",
            "notes.yaml",
        ] {
            assert_eq!(entry_id(not), None, "{not}");
        }
    }

    #[test]
    fn a_broken_entry_names_its_file_and_line() {
        let id = LineId::parse("2026-10-30-0910-0001").unwrap();
        let err = read_entry(
            "f.yaml",
            id,
            "at: 2026-10-30T09:10:02Z\nkind: remark\nby: zend\ntext: x\n",
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "f.yaml:2: `kind`: `remark` is not a kind: note, episode or fold"
        );
    }

    #[test]
    fn lines_are_ordered_by_time_then_id() {
        let a = note("2026-10-30T09:10:59Z", "later in the minute");
        let b = note("2026-10-30T09:10:01Z", "earlier in the minute");
        let c = note("2026-10-29T23:00:00Z", "the day before");
        let mut lines = vec![a.clone(), b.clone(), c.clone()];
        in_order(&mut lines);
        assert_eq!(lines, vec![c, b, a]);
    }
}
