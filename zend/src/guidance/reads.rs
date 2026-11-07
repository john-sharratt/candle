//! Which file a call read, and the unit that read it at the conversation's
//! base.

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;
use zend_vfs::{RepoFiles, Workspace};

use crate::branch_ingest::keys::{file_key, CONTENT_KEY};
use crate::code_read::chain_finished;
use crate::tools::ToolCall;
use crate::working_set::corpus::read_target;

/// The tool that reads a guide by name.
const GUIDE_READ: &str = "guide_read";

/// Where the guides `guide_read` serves live: the repository and the folder
/// the `guides` crate embeds.
const GUIDES_REPO: &str = "candle";
const GUIDES_DIR: &str = "mcp/src/guides";

/// One file a call read: its repository and its repository-relative path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Read {
    pub repo: String,
    pub rel: String,
}

impl Read {
    /// Its workspace-relative path, the form a group's `draws` names.
    pub fn path(&self) -> String {
        format!("{}/{}", self.repo, self.rel)
    }
}

/// The file `call` read: a `guide_read`'s guide, or a `file_read`'s file.
/// `None` for any other call, and for one naming no file.
pub fn read_path(call: &ToolCall, workspace: &Workspace) -> Option<Read> {
    if call.name == GUIDE_READ {
        let name = call.arguments.get("name")?.as_str()?.trim();
        if name.is_empty() || name.contains(['/', '\\', '.']) {
            return None;
        }
        return Some(Read {
            repo: GUIDES_REPO.to_string(),
            rel: format!("{GUIDES_DIR}/{name}.md"),
        });
    }
    let (repo, rel) = read_target(call, workspace)?;
    Some(Read { repo, rel })
}

/// The finished `code_reading` unit that read `read` at the version the
/// conversation's base holds, with its content key — `None` when the
/// conversation changed the file, or no finished unit read that version.
pub fn unit_of(
    engine: &ConversationEngine,
    files: &RepoFiles,
    read: &Read,
) -> Option<(TimelineId, String)> {
    let (blob, _) = files.repo(&read.repo).ok()?.content_id(&read.rel).ok()??;
    let key = file_key(&read.path(), &blob);
    let unit = engine
        .find_conversations_by_metadata(CONTENT_KEY, &key)
        .into_iter()
        .find(|&tl| chain_finished(engine, tl))?;
    Some((unit, key))
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use zend_vfs::RepoSpec;

    use super::*;

    fn workspace() -> Workspace {
        Workspace::new(
            "D:/prog",
            vec![RepoSpec::named("candle"), RepoSpec::named("journals")],
        )
        .unwrap()
    }

    fn call(name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            name: name.to_string(),
            arguments,
        }
    }

    fn read(repo: &str, rel: &str) -> Option<Read> {
        Some(Read {
            repo: repo.into(),
            rel: rel.into(),
        })
    }

    /// **A guide is read from the folder the guides crate embeds**, by name.
    #[test]
    fn a_guide_read_names_the_guides_file() {
        let ws = workspace();
        let got = read_path(&call(GUIDE_READ, json!({"name": " deploy "})), &ws);
        assert_eq!(got, read("candle", "mcp/src/guides/deploy.md"));
        assert_eq!(
            got.unwrap().path(),
            "candle/mcp/src/guides/deploy.md",
            "the form a group's draws name"
        );
    }

    /// A name that is no plain guide name names no file.
    #[test]
    fn a_guide_name_that_is_a_path_names_nothing() {
        let ws = workspace();
        for name in ["", "../secrets", "a/b", "deploy.md"] {
            assert_eq!(
                read_path(&call(GUIDE_READ, json!({"name": name})), &ws),
                None,
                "{name:?}"
            );
        }
        assert_eq!(read_path(&call(GUIDE_READ, json!({})), &ws), None);
    }

    /// **A file read names its file**; any other call names none.
    #[test]
    fn a_file_read_names_its_file_and_nothing_else_does() {
        let ws = workspace();
        let got = read_path(
            &call(
                "file_read",
                json!({"repo": "journals", "path": "./zen/topics/candle.md"}),
            ),
            &ws,
        );
        assert_eq!(got, read("journals", "zen/topics/candle.md"));
        assert_eq!(
            read_path(&call("guide_list", json!({})), &ws),
            None,
            "a listing reads no guide"
        );
        assert_eq!(
            read_path(
                &call("write", json!({"repo": "candle", "path": "a.md"})),
                &ws
            ),
            None
        );
    }
}
