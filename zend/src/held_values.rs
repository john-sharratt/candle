//! What a round holds its file calls' values to (`docs/stencil_tree.md`
//! §9.2): for each file the conversation has read or edited, the anchors it
//! carves into now and each part's line count — so `part` names a part the
//! file has, and `from` and `to` a line the part has.
//!
//! Read from the conversation's own store each round, its edits included, so
//! an anchor an edit retired is gone and one it made is there. A file is
//! carved again only when its content has changed, and a value set's rule is
//! built again only when the set has ([`HeldRules`]).

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::iter;
use std::mem;
use std::sync::{Arc, Mutex, PoisonError};

use candle_conversation::stencil::{FieldHold, HeldRules, TriggerRegistry, ValueHolds};
use candle_conversation::ConversationEngine;
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

/// A file as its holds read it: its anchors, and each part's line count.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Carved {
    anchors: Vec<String>,
    parts: Vec<(String, usize)>,
}

impl Carved {
    fn of(doc: &Doc) -> Carved {
        Carved {
            anchors: doc.anchors(),
            parts: doc
                .units()
                .iter()
                .map(|u| (u.anchor.clone(), u.end - u.start))
                .collect(),
        }
    }
}

/// What one round's holds leave for the next: each held file carved, by the
/// digest of the content it was carved from, and the rules built.
#[derive(Debug, Default)]
struct Kept {
    carved: HashMap<(String, String), (u64, Arc<Carved>)>,
    rules: HeldRules,
}

/// The files a conversation has read or edited, most recent last.
#[derive(Debug, Clone, Default)]
pub struct FilesSeen {
    files: Vec<(String, String)>,
    /// Shared by every clone: a round's holds are built on a clone, off the
    /// conversation's state.
    kept: Arc<Mutex<Kept>>,
}

fn digest(text: &str) -> u64 {
    let mut h = DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
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

    /// `base` with the calls to these files held to what each holds in
    /// `files` as it stands (`ConversationEngine::holding_values`).
    pub fn triggers(
        &self,
        engine: &ConversationEngine,
        base: &Arc<TriggerRegistry>,
        files: &RepoFiles,
    ) -> Arc<TriggerRegistry> {
        let mut kept = self.kept.lock().unwrap_or_else(PoisonError::into_inner);
        let holds = self.holds(files, &mut kept.carved);
        engine.holding_values(base, &holds, &mut kept.rules)
    }

    /// What a round holds the calls to these files to, each read from `files`
    /// as it stands and carved again only when it has changed since `carved`
    /// had it; `carved` is left with these files alone. A file that cannot be
    /// read is held to nothing.
    fn holds(
        &self,
        files: &RepoFiles,
        carved: &mut HashMap<(String, String), (u64, Arc<Carved>)>,
    ) -> ValueHolds {
        let mut last = mem::take(carved);
        for (repo, path) in &self.files {
            let read = match files.repo(repo) {
                Ok(store) => store.read(path).map_err(|e| e.to_string()),
                Err(e) => Err(e.to_string()),
            };
            let text = match read {
                Ok(Some(text)) => text,
                Ok(None) => {
                    tracing::debug!(repo = %repo, path = %path, "a held file is not in the store: its calls are not held");
                    continue;
                }
                Err(e) => {
                    tracing::warn!(repo = %repo, path = %path, "a held file could not be read, so its calls are not held: {e}");
                    continue;
                }
            };
            let file = (repo.clone(), path.clone());
            let at = digest(&text);
            let entry = match last.remove(&file) {
                Some((was, kept)) if was == at => (at, kept),
                _ => (at, Arc::new(Carved::of(&Doc::new(path, &text)))),
            };
            carved.insert(file, entry);
        }
        let docs: Vec<(&str, &str, &Carved)> = self
            .files
            .iter()
            .filter_map(|file| {
                let (_, c) = carved.get(file)?;
                Some((file.0.as_str(), file.1.as_str(), c.as_ref()))
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
fn holds_of(docs: &[(&str, &str, &Carved)]) -> ValueHolds {
    let mut anchors: HashMap<Vec<String>, Vec<String>> = HashMap::new();
    let mut lines: HashMap<Vec<String>, Vec<String>> = HashMap::new();
    for (repo, path, carved) in docs {
        let file = vec![repo.to_string(), path.to_string()];
        anchors.insert(file.clone(), carved.anchors.clone());
        for (anchor, len) in &carved.parts {
            let mut at = file.clone();
            at.push(anchor.clone());
            lines.insert(at, (1..=*len).map(|n| format!("+{n}")).collect());
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
    use std::path::Path;
    use std::process::Command;

    use serde_json::json;
    use zend_vfs::{RepoSpec, Workspace};

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
        let holds = holds_of(&[("r", "a.rs", &Carved::of(&doc))]);
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

    fn git(dir: &Path, args: &[&str]) {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args([
                "-c",
                "core.hooksPath=",
                "-c",
                "user.name=T",
                "-c",
                "user.email=t@x",
            ])
            .args(args)
            .output()
            .expect("git runs");
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// **A file is carved again only when it has changed**: a second round
    /// over the same content reuses its carving, an edit carves it afresh,
    /// and a file the store does not have is held to nothing.
    #[test]
    fn a_file_is_carved_again_only_when_it_changes() {
        let root = tempfile::tempdir().unwrap();
        let code = root.path().join("code");
        std::fs::create_dir_all(&code).unwrap();
        git(&code, &["init", "-q", "-b", "main"]);
        std::fs::write(code.join("a.rs"), "fn a() {\n    1\n}\n").unwrap();
        git(&code, &["add", "a.rs"]);
        git(&code, &["commit", "-q", "-m", "first"]);
        let ws = Workspace::new(root.path(), vec![RepoSpec::named("code")]).unwrap();
        let files = RepoFiles::overlay(ws).unwrap();
        let mut seen = FilesSeen::default();
        seen.seen("code", "a.rs");
        seen.seen("code", "gone.rs");
        let file = ("code".to_string(), "a.rs".to_string());
        let mut carved = HashMap::new();

        let first = seen.holds(&files, &mut carved);
        let was = Arc::clone(&carved[&file].1);
        assert_eq!(carved.len(), 1, "`gone.rs` is held to nothing");
        assert_eq!(seen.holds(&files, &mut carved), first);
        assert!(Arc::ptr_eq(&was, &carved[&file].1));

        files
            .repo("code")
            .unwrap()
            .write("a.rs", "fn a() {\n    1\n}\n\nfn b() {}\n".into())
            .unwrap();
        assert_ne!(seen.holds(&files, &mut carved), first);
        assert!(!Arc::ptr_eq(&was, &carved[&file].1));
        assert_eq!(carved[&file].1.parts.len(), was.parts.len() + 1);
    }
}
