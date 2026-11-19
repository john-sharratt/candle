//! The ingested readings of a file: which finished `code_reading` conversation
//! read the version a dialogue's base holds, and which readings a turn's own
//! reads put out of the working set (`docs/zend_working_set.md` §4.5).
//!
//! A `code_reading` conversation is a whole file, read once, sealed in the
//! substrate and keyed by its path and blob id (`docs/zend_branch_ingest.md`
//! §6.1). The seeds place such readings ahead of a dialogue. A `file_read` the
//! dialogue makes runs for real, and its page then stands verbatim in the
//! dialogue, so the readings of the version it read are kept out of the
//! working set for the rest of the turn's loop ([`readings_of`]).

use std::collections::BTreeSet;
use std::path::Path;

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;
use zend_vfs::{Oid, RepoFiles, Workspace};

use super::coverage::Coverage;
use crate::branch_ingest::keys::{file_key, CONTENT_KEY, LINES_KEY};
use crate::code_read::chain_finished;
use crate::tools::ToolCall;

/// The one tool whose reads are readings of a file.
const FILE_READ: &str = "file_read";

/// The repository-relative form of a `path` argument.
///
/// The key is path-qualified, so a call that names the same file differently —
/// a leading `./`, a backslash separator, an absolute path inside the
/// repository — hashes to something else and misses every time, silently and
/// forever. Normalising here is what keeps the two sides addressing the same
/// content.
pub fn normalise(path: &str, repo_dir: &Path) -> String {
    let cleaned = path.replace('\\', "/");
    let cleaned = cleaned.trim_start_matches("./");
    let dir = repo_dir.to_string_lossy().replace('\\', "/");
    let dir = dir.trim_end_matches('/');
    cleaned
        .strip_prefix(&format!("{dir}/"))
        .unwrap_or(cleaned)
        .trim_start_matches('/')
        .to_string()
}

/// The repository and normalised path a `file_read` names. `None` for any
/// other call, a read without both, or a repository the workspace does not
/// list.
pub(crate) fn read_target(call: &ToolCall, workspace: &Workspace) -> Option<(String, String)> {
    if call.name != FILE_READ {
        return None;
    }
    let repo = call.arguments.get("repo")?.as_str()?;
    let path = call.arguments.get("path")?.as_str()?;
    // A repository read from the substrate has no folder: an absolute path
    // into it names where it would sit under the workspace.
    let dir = match workspace.repo(repo)?.folder() {
        Some(dir) => dir.to_path_buf(),
        None => workspace.root().join(repo),
    };
    Some((repo.to_string(), normalise(path, &dir)))
}

/// Every ingested reading of the versions of the files `calls` read, in call
/// order: the conversations a turn that ran them keeps out of its working set.
/// One pass of the substrate for the whole round.
///
/// Nothing for a call but a `file_read`, and nothing for a file the
/// conversation has changed — its read returned its own copy, which no
/// reading holds.
pub fn readings_of_all(
    engine: &ConversationEngine,
    files: &RepoFiles,
    workspace: &Workspace,
    calls: &[ToolCall],
) -> Vec<TimelineId> {
    let keys: Vec<String> = calls
        .iter()
        .filter_map(|call| {
            let (repo, rel) = read_target(call, workspace)?;
            file_key_of(files, &repo, &rel)
        })
        .collect();
    if keys.is_empty() {
        return Vec::new();
    }
    let wanted: BTreeSet<String> = keys.iter().cloned().collect();
    let found = engine.find_conversations_by_metadata_values(CONTENT_KEY, &wanted);
    keys.iter()
        .flat_map(|k| found.get(k).into_iter().flatten().copied())
        .collect()
}

/// The content key of `rel` in `repo` at the blob the conversation's base
/// holds — `None` when the conversation changed the file (its own copy is
/// what its read must return), or there is no such file.
pub fn file_key_of(files: &RepoFiles, repo: &str, rel: &str) -> Option<String> {
    let (blob, _) = committed(files, repo, rel)?;
    Some(file_key(&format!("{repo}/{rel}"), &blob))
}

/// The finished, complete `code_reading` conversation among `readings` — the
/// conversations that read one version of a file — or `None` when none read
/// every line.
pub fn file_conversation_among(
    engine: &ConversationEngine,
    coverage: &Coverage,
    readings: &[TimelineId],
) -> Option<TimelineId> {
    readings
        .iter()
        .copied()
        // A read whose chain never finished holds no summary: placed, it
        // would put an assistant that deliberates and answers nothing into
        // this conversation.
        .filter(|&tl| chain_finished(engine, tl))
        // A reading with turns empty here — not yet fetched from the member
        // that wrote them, or retired — places nothing for them; another
        // reading of the same version may be whole.
        .filter(|&tl| engine.holds_whole(tl))
        .find(|&tl| {
            lines_of(engine, tl).is_some_and(|lines| coverage.is_complete(engine, tl, lines))
        })
}

/// The blob id and size of the file `rel` in `repo` as the conversation's
/// base holds it — `None` when the conversation has changed it (its own copy
/// is what its read must return), or when there is no such file.
fn committed(files: &RepoFiles, repo: &str, rel: &str) -> Option<(Oid, u64)> {
    files.repo(repo).ok()?.content_id(rel).ok()?
}

/// How many lines the file `timeline` read has, as its ingest recorded it.
/// `None` for a conversation that recorded none.
fn lines_of(engine: &ConversationEngine, timeline: TimelineId) -> Option<usize> {
    lines_in(
        engine
            .conversation_metadata_value(timeline, LINES_KEY)
            .as_deref(),
    )
}

fn lines_in(recorded: Option<&str>) -> Option<usize> {
    recorded?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use zend_vfs::{ObjectFormat, RepoSpec};

    /// A repository's folder, the base [`normalise`] strips.
    fn ws() -> &'static Path {
        Path::new("D:/prog/candle")
    }

    fn workspace() -> Workspace {
        Workspace::new("D:/prog", vec![RepoSpec::named("candle")]).unwrap()
    }

    fn call(name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            name: name.to_string(),
            arguments: args,
        }
    }

    /// **A read names its repository and the normalised path** — the
    /// workspace-relative key the ingest hashed the file under.
    #[test]
    fn a_read_names_its_repository_and_path() {
        let read = call(
            FILE_READ,
            json!({"repo": "candle", "path": "./zend/src/main.rs"}),
        );
        assert_eq!(
            read_target(&read, &workspace()),
            Some(("candle".to_string(), "zend/src/main.rs".to_string()))
        );
    }

    /// **Only a `file_read` with a listed repository and a path is a read.**
    /// Another tool naming a path, an unlisted repository, or a read missing
    /// either argument names no reading.
    #[test]
    fn only_a_whole_file_read_names_a_reading() {
        for repo in ["other", "..", ""] {
            let read = call(FILE_READ, json!({"repo": repo, "path": "x.rs"}));
            assert_eq!(read_target(&read, &workspace()), None, "{repo:?}");
        }
        let write = call("write", json!({"repo": "candle", "path": "a.rs"}));
        assert_eq!(read_target(&write, &workspace()), None);
        let list = call("file_list", json!({"repo": "candle", "path": "zend/src"}));
        assert_eq!(read_target(&list, &workspace()), None);
        let bare = call(FILE_READ, json!({"path": "zend/src/main.rs"}));
        assert_eq!(read_target(&bare, &workspace()), None);
    }

    #[test]
    fn a_plain_relative_path_is_already_normal() {
        assert_eq!(normalise("zend/src/main.rs", ws()), "zend/src/main.rs");
    }

    #[test]
    fn backslashes_become_the_walkers_separator() {
        assert_eq!(normalise("zend\\src\\main.rs", ws()), "zend/src/main.rs");
    }

    #[test]
    fn a_dot_slash_prefix_is_dropped() {
        assert_eq!(normalise("./zend/src/main.rs", ws()), "zend/src/main.rs");
    }

    /// The model often answers with the absolute path a listing showed it; that
    /// has to hash the same as the walker's relative form or it misses forever.
    #[test]
    fn an_absolute_path_inside_the_repository_becomes_relative() {
        assert_eq!(
            normalise("D:/prog/candle/zend/src/main.rs", ws()),
            "zend/src/main.rs"
        );
        assert_eq!(
            normalise("D:\\prog\\candle\\zend\\src\\main.rs", ws()),
            "zend/src/main.rs"
        );
    }

    #[test]
    fn a_leading_slash_is_dropped() {
        assert_eq!(normalise("/zend/src/main.rs", ws()), "zend/src/main.rs");
    }

    /// A path outside the repository keeps its shape — it will simply find no
    /// conversation, which is the correct outcome rather than a false hit.
    #[test]
    fn a_path_outside_the_repository_is_left_alone() {
        assert_eq!(normalise("C:/elsewhere/x.rs", ws()), "C:/elsewhere/x.rs");
    }

    /// **A file is keyed as the conversation's store holds it, and never once
    /// the conversation has changed it** — its own copy is what it must read.
    #[test]
    fn only_a_file_the_conversation_left_alone_is_offered() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("candle")).unwrap();
        std::fs::write(dir.path().join("candle/a.rs"), b"a\n").unwrap();
        std::fs::write(dir.path().join("candle/b.rs"), b"b\n").unwrap();
        let ws = Workspace::new(dir.path(), vec![RepoSpec::named("candle")]).unwrap();
        let files = RepoFiles::overlay(ws).unwrap();
        files
            .repo("candle")
            .unwrap()
            .write("b.rs", "mine\n".into())
            .unwrap();
        assert_eq!(
            committed(&files, "candle", "a.rs"),
            Some((ObjectFormat::Sha1.blob_id(b"a\n"), 2))
        );
        assert_eq!(committed(&files, "candle", "b.rs"), None);
        assert_eq!(committed(&files, "candle", "nope.rs"), None);
        assert_eq!(committed(&files, "other", "a.rs"), None);
    }

    /// **The line count is the one the ingest recorded**, and a conversation
    /// that recorded none — or something that is no count — answers nothing.
    #[test]
    fn the_line_count_is_the_one_the_ingest_recorded() {
        assert_eq!(lines_in(None), None);
        assert_eq!(lines_in(Some("350")), Some(350));
        assert_eq!(lines_in(Some("many")), None);
    }
}
