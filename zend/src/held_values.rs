//! What a round holds its file calls' values to (`docs/stencil_tree.md`
//! §9.2): for each file the conversation has read or edited, the anchors it
//! carves into now and each part's line count — so `part` names a part the
//! file has, and `from` and `to` a line the part has.
//!
//! Read from the conversation's own store each round, its edits included, so
//! an anchor an edit retired is gone and one it made is there.

use std::collections::HashMap;
use std::iter;

use candle_conversation::stencil::{FieldHold, ValueHolds};
use doc_units::Doc;
use serde_json::Value;
use zend_tools::registry;
use zend_vfs::RepoFiles;

use crate::tools::ToolResult;

/// The most files a round holds: the conversation's most recent.
pub const MAX_FILES: usize = 32;

/// The tools a held file's values are held for, and the anchor fields each
/// names a part by.
const ANCHOR_FIELDS: [(&str, &[&str]); 2] = [
    ("file_read", &["part"]),
    ("file_edit", &["part", "through", "dest"]),
];

/// The tools whose calls name a file the conversation then works in.
const NOTED: [&str; 2] = ["file_read", "file_edit"];

/// The files a conversation has read or edited, most recent last.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FilesSeen {
    files: Vec<(String, String)>,
}

impl FilesSeen {
    /// Note the files `results`' calls read or edited.
    pub fn note(&mut self, results: &[ToolResult]) {
        for r in results {
            let Some(tool) = registry::find(&r.call.name) else {
                continue;
            };
            if !NOTED.contains(&tool.name) {
                continue;
            }
            let text = |k: &str| r.call.arguments.get(k).and_then(Value::as_str);
            if let (Some(repo), Some(path)) = (text("repo"), text("path")) {
                self.seen(repo, path);
            }
        }
    }

    fn seen(&mut self, repo: &str, path: &str) {
        let file = (repo.to_string(), path.to_string());
        self.files.retain(|f| *f != file);
        self.files.push(file);
        if self.files.len() > MAX_FILES {
            self.files.remove(0);
        }
    }

    /// What a round holds the calls to these files to, each read from `files`
    /// as it stands. A file that cannot be read is held to nothing.
    pub fn holds(&self, files: &RepoFiles) -> ValueHolds {
        let docs: Vec<(&str, &str, Doc)> = self
            .files
            .iter()
            .filter_map(|(repo, path)| {
                let text = files.repo(repo).ok()?.read(path).ok()??;
                Some((repo.as_str(), path.as_str(), Doc::new(path, &text)))
            })
            .collect();
        holds_of(&docs)
    }
}

/// `tool` and every alias it answers to.
fn names_of(tool: &str) -> Vec<String> {
    iter::once(tool)
        .chain(registry::aliases(tool).iter().copied())
        .map(str::to_string)
        .collect()
}

/// The holds of `docs`, each `(repo, path, the file carved)`.
fn holds_of(docs: &[(&str, &str, Doc)]) -> ValueHolds {
    let mut anchors: HashMap<Vec<String>, Vec<String>> = HashMap::new();
    let mut lines: HashMap<Vec<String>, Vec<String>> = HashMap::new();
    for (repo, path, doc) in docs {
        let file = vec![repo.to_string(), path.to_string()];
        anchors.insert(file.clone(), doc.anchors());
        for unit in doc.units() {
            let mut at = file.clone();
            at.push(unit.anchor.clone());
            lines.insert(
                at,
                (1..=unit.end - unit.start)
                    .map(|n| format!("+{n}"))
                    .collect(),
            );
        }
    }
    let mut fields = Vec::new();
    for (tool, anchor_fields) in ANCHOR_FIELDS {
        for field in anchor_fields {
            fields.push(FieldHold {
                tools: names_of(tool),
                field: field.to_string(),
                on: vec!["repo".into(), "path".into()],
                table: anchors.clone(),
            });
        }
    }
    for field in ["from", "to"] {
        fields.push(FieldHold {
            tools: names_of("file_edit"),
            field: field.into(),
            on: vec!["repo".into(), "path".into(), "part".into()],
            table: lines.clone(),
        });
    }
    ValueHolds { fields }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::tools::ToolCall;

    fn result(name: &str, args: Value) -> ToolResult {
        ToolResult {
            call: ToolCall {
                name: name.into(),
                arguments: args,
            },
            response: json!({}),
        }
    }

    /// **A file read or edited is noted, most recent last and once**; a call
    /// to any other tool, or one naming no file, notes nothing.
    #[test]
    fn the_files_read_and_edited_are_noted() {
        let mut seen = FilesSeen::default();
        seen.note(&[
            result("file_read", json!({"repo": "r", "path": "a.rs", "list": 1})),
            result(
                "file_edit",
                json!({"repo": "r", "path": "b.rs", "part": "KehT"}),
            ),
            result("file_grep", json!({"repo": "r", "pattern": "x"})),
            result("file_read", json!({"repo": "r"})),
            result(
                "file_read",
                json!({"repo": "r", "path": "a.rs", "part": "KehT"}),
            ),
        ]);
        assert_eq!(
            seen.files,
            [
                ("r".to_string(), "b.rs".to_string()),
                ("r".to_string(), "a.rs".to_string())
            ]
        );
    }

    /// **A file's anchors hold its parts, and each part's lines its `from`
    /// and `to`** — exactly the file's anchors, and `+1` to the part's length.
    #[test]
    fn a_files_parts_and_lines_are_held() {
        let doc = Doc::new("a.rs", "fn a() {\n    1\n}\n\nfn b() {}\n");
        let units: Vec<(String, usize)> = doc
            .units()
            .iter()
            .map(|u| (u.anchor.clone(), u.end - u.start))
            .collect();
        let holds = holds_of(&[("r", "a.rs", doc)]);
        let file = vec!["r".to_string(), "a.rs".to_string()];
        let part = holds
            .fields
            .iter()
            .find(|f| f.field == "part" && f.tools.contains(&"file_edit".to_string()))
            .unwrap();
        assert_eq!(
            part.table[&file],
            units.iter().map(|(a, _)| a.clone()).collect::<Vec<_>>()
        );
        let to = holds.fields.iter().find(|f| f.field == "to").unwrap();
        for (anchor, len) in &units {
            let mut at = file.clone();
            at.push(anchor.clone());
            let want: Vec<String> = (1..=*len).map(|n| format!("+{n}")).collect();
            assert_eq!(to.table[&at], want);
        }
        assert_eq!(to.on, ["repo", "path", "part"]);
    }
}
