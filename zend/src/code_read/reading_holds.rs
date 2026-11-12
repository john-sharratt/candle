//! What a file's reading holds its calls to: the one file it reads
//! (`docs/stencil_tree.md` §9.2.2).
//!
//! The reading's prompt names the repository, the path inside it and every
//! range to read, and the model — not the grammar — used to write them. A path
//! whose first directory is, or starts with, a repository's name
//! (`candle-conversation/…`, `journals/…` in `candle`) came back without it, as
//! though it were the repository prefix every read fence shows, and the read
//! found no file: `candle-conversation/src/turn_layout.rs` was read as
//! `src/turn_layout.rs` nine times over. Held, nothing else can be written.

use std::collections::HashMap;

use candle_conversation::stencil::{FieldHold, ValueHolds};
use zend_tools::tools::file::render::range_value;

use super::FILE_READ_TOOL;
use crate::repo_path::split;

/// The holds of the reading of `key` (workspace-relative) in `ranges`:
/// `file_read`'s `repo` its repository, `path` its path inside it, and `lines`
/// one of its ranges — `1` for an empty file, read as one range.
pub(crate) fn reading_holds(key: &str, ranges: &[(usize, usize)]) -> ValueHolds {
    let (repo, path) = split(key);
    let lines: Vec<String> = match ranges.is_empty() {
        true => vec!["1".to_string()],
        false => ranges.iter().map(|&r| range_value(r)).collect(),
    };
    let one = |field: &str, values: Vec<String>| FieldHold {
        tools: vec![FILE_READ_TOOL.to_string()],
        field: field.to_string(),
        on: Vec::new(),
        table: HashMap::from([(Vec::new(), values)]),
    };
    ValueHolds {
        fields: vec![
            one("repo", vec![repo.to_string()]),
            one("path", vec![path.to_string()]),
            one("lines", lines),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values<'a>(holds: &'a ValueHolds, field: &str) -> &'a [String] {
        let hold = holds.fields.iter().find(|f| f.field == field).unwrap();
        assert_eq!(hold.tools, [FILE_READ_TOOL]);
        assert!(hold.on.is_empty());
        &hold.table[&Vec::new()]
    }

    /// **A reading holds `file_read` to its own file**: the repository and the
    /// path inside it as the key splits them, every range it reads, and
    /// nothing else.
    #[test]
    fn a_reading_holds_its_calls_to_its_file() {
        let holds = reading_holds(
            "candle/candle-conversation/src/turn_layout.rs",
            &[(1, 198), (199, 391)],
        );
        assert_eq!(values(&holds, "repo"), ["candle"]);
        assert_eq!(
            values(&holds, "path"),
            ["candle-conversation/src/turn_layout.rs"]
        );
        assert_eq!(values(&holds, "lines"), ["1-198", "199-391"]);
        assert_eq!(holds.fields.len(), 3);
    }

    /// An empty file is read as one range, `1`, as its prompt asks.
    #[test]
    fn an_empty_file_is_held_to_one_range() {
        let holds = reading_holds("journals/README.md", &[]);
        assert_eq!(values(&holds, "repo"), ["journals"]);
        assert_eq!(values(&holds, "path"), ["README.md"]);
        assert_eq!(values(&holds, "lines"), ["1"]);
    }
}
