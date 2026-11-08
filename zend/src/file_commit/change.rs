//! The changes one `file_edit_and_commit` call makes (`docs/mcp_file_tools.md`
//! §3.1): text files only, each named by its repository-relative path.

use serde::Deserialize;

use super::edits::Edit;

/// One `old` → `new`, as the call spells it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditArg {
    pub old: String,
    pub new: String,
}

/// One file's change.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Change {
    /// Exact replacements in an existing text file, in order.
    Edit { path: String, edits: Vec<EditArg> },
    /// The file's whole new text; it is created when it does not exist.
    Write { path: String, content: String },
    /// Remove the file.
    Delete { path: String },
    /// Move the file, unchanged, to a path that does not exist yet.
    Move { from: String, to: String },
    /// The file's whole new bytes, any bytes; created when it does not exist.
    /// The transfers' alone (`file_download_and_commit`,
    /// `file_upload_and_commit`, `docs/mcp_file_tools.md` §3.8): no call of
    /// `file_edit_and_commit` can name it, which stays text-only.
    #[serde(skip_deserializing)]
    Bytes { path: String, bytes: Vec<u8> },
}

impl Change {
    /// Every path the change touches.
    pub fn paths(&self) -> Vec<&str> {
        match self {
            Change::Edit { path, .. }
            | Change::Write { path, .. }
            | Change::Delete { path }
            | Change::Bytes { path, .. } => vec![path],
            Change::Move { from, to } => vec![from, to],
        }
    }

    pub fn edits(args: &[EditArg]) -> Vec<Edit> {
        args.iter()
            .map(|e| Edit {
                old: e.old.clone(),
                new: e.new.clone(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn each_change_reads_from_its_op() {
        let changes: Vec<Change> = serde_json::from_value(json!([
            {"op": "edit", "path": "a.js", "edits": [{"old": "x", "new": "y"}]},
            {"op": "write", "path": "b.css", "content": ":root {}\n"},
            {"op": "delete", "path": "c.html"},
            {"op": "move", "from": "d.js", "to": "e.js"},
        ]))
        .unwrap();
        assert_eq!(
            changes,
            vec![
                Change::Edit {
                    path: "a.js".into(),
                    edits: vec![EditArg {
                        old: "x".into(),
                        new: "y".into()
                    }]
                },
                Change::Write {
                    path: "b.css".into(),
                    content: ":root {}\n".into()
                },
                Change::Delete {
                    path: "c.html".into()
                },
                Change::Move {
                    from: "d.js".into(),
                    to: "e.js".into()
                },
            ]
        );
        assert_eq!(changes[3].paths(), vec!["d.js", "e.js"]);
        assert!(serde_json::from_value::<Change>(json!({"op": "chmod", "path": "a"})).is_err());
        assert!(
            serde_json::from_value::<Change>(json!({"op": "bytes", "path": "a", "bytes": [0]}))
                .is_err(),
            "no call names bytes"
        );
        assert!(
            serde_json::from_value::<Change>(json!({"op": "delete", "path": "a", "x": 1})).is_err()
        );
    }
}
