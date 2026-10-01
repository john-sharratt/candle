//! Serving a `file_read` or a `file_list` from content the corpus has already
//! ingested.
//!
//! A `code_reading` conversation is a whole file, read once, sealed in the
//! substrate and keyed by its path and blob id (`docs/zend_branch_ingest.md`
//! §6.1). When a dialogue asks for a file its base holds at a blob a
//! conversation already read, the cheapest correct answer is not to read the
//! file again: it is to carry that conversation into this one's projection
//! and say so. The K/V is already there, so the call costs an elevation
//! instead of a prefill and a decode.
//!
//! A `repo_map` conversation is the same for a folder: it holds the folder's
//! `file_list` response and its summary, keyed by what the listing shows
//! (§6.2). A `file_list` of the first page of a folder whose unit the
//! conversation's base holds carries that unit instead of listing again.
//!
//! **The hit test is the whole file; the injection is the whole file.** A call
//! for page 1 of a file hits on the file's hash, and what lands in context is
//! the entire read — so the requested page is necessarily present, and no page
//! bookkeeping is needed to know it. A folder unit holds only its listing's
//! first page, so only a call for that page is served.
//!
//! What it does NOT do is claim more than it delivers. A hit is only returned
//! once the conversation has been admitted to the projection
//! (`Substrate::fast_path_admit`), which refuses a read too large for the
//! layer's budget; a refusal falls through to a real read. Telling the model a
//! file is already in context when it is not would be worse than any number of
//! redundant reads — it answers from nothing rather than looking again.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;
use serde_json::{json, Value};
use zend_vfs::{Oid, RepoFiles, Workspace};

use crate::branch_ingest::keys::{file_key, CONTENT_KEY, LINES_KEY};
use crate::code_read::chain_finished;
use crate::tool_round::{plan, Step};
use crate::tools::ToolResult;

/// Whole-file reads are content-addressed by `code_reading`.
const FILE_READ: &str = "file_read";

/// Folder listings are content-addressed by `repo_map`.
const FILE_LIST: &str = "file_list";

/// Above this estimated size a file is read normally rather than carried.
///
/// One enormous file would fill the whole fast-path budget and evict every
/// other read to do it, so the conversation ends up carrying one file instead
/// of the thirty it would otherwise have. A read that large is also the case
/// where paging through ranges is what the model actually wants.
const MAX_FAST_PATH_FILE_TOKENS: usize = 100_000;

/// Bytes per token, for sizing a file without tokenizing it.
///
/// Deliberately an estimate: this decides whether to take a shortcut, and
/// being wrong costs a normal read — the outcome the cap exists to produce.
/// Tokenizing every candidate to answer it exactly would spend more than the
/// shortcut saves.
const BYTES_PER_TOKEN: usize = 4;

/// Whether a file of `bytes` is small enough to carry rather than re-read.
fn fits_fast_path(bytes: u64) -> bool {
    bytes / BYTES_PER_TOKEN as u64 <= MAX_FAST_PATH_FILE_TOKENS as u64
}

/// The answer a served call gets.
///
/// **No `error`, and no `detail`.** Both mark a failed call — the GUI renders
/// either as a red card (`is_error`, `zend/web/index.html`) and the model reads
/// a failure as grounds to try again, which here means doing the very read the
/// fast path just avoided.
fn served_response(repo: &str, path: &str, lines: usize) -> serde_json::Value {
    json!({
        "status": "already_read",
        "repo": repo,
        "path": path,
        "lines": lines,
        "note": format!(
            "`{path}` in {repo} is unchanged since it was read, and its full \
             contents ({lines} lines) are already in this conversation's context — \
             including any lines this call asked for. Read it from there \
             rather than calling file_read for it again."
        ),
    })
}

/// The answer a served `file_list` gets — as few tokens as says it, since the
/// listing it stands in for is small and sits right beside it in context.
/// Carries no failure marker, for the reason [`served_response`] gives.
fn listed_response() -> Value {
    json!({"status": "already_listed", "note": "Listing already in context."})
}

/// One call the fast path answered, for the caller to log and report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Served {
    /// The tool whose call was answered.
    pub tool: &'static str,
    /// Workspace-relative: a file (`candle/src/lib.rs`) or a folder with its
    /// trailing `/` (`candle/src/`).
    pub path: String,
    pub timeline: TimelineId,
}

/// Finds the committed `repo_map` unit for a folder — repository and path
/// inside it, `""` for its root — as the conversation's base lists it
/// (`RetrievalScope::folder_unit`).
pub type FolderOf<'a> = &'a dyn Fn(&str, &str) -> Option<TimelineId>;

/// What one screening consults: the conversation, its view of every
/// repository, and where the ingested units are.
pub struct Screen<'a> {
    pub engine: &'a Mutex<ConversationEngine>,
    pub target: TimelineId,
    pub workspace: &'a Workspace,
    pub files: &'a RepoFiles,
    pub folder_of: FolderOf<'a>,
    /// Tokens the conversation's fast-path set may hold.
    pub budget_tokens: usize,
}

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

/// The `repo` and `path` arguments of a `file_read` call, when it has both.
fn read_target(step: &Step) -> Option<(&str, &str)> {
    match step {
        Step::Run(call) if call.name == FILE_READ => Some((
            call.arguments.get("repo")?.as_str()?,
            call.arguments.get("path")?.as_str()?,
        )),
        _ => None,
    }
}

/// The workspace-relative path `code_reading` keys a `file_read`'s file
/// under — the repository, then the normalised path inside it — with the
/// repository and inner path beside it. `None` for a repository the workspace
/// does not list: the call then runs for real and the tool refuses it, rather
/// than this reading whatever an unchecked name joins to.
fn read_key(step: &Step, workspace: &Workspace) -> Option<(String, String, String)> {
    let (repo, path) = read_target(step)?;
    let dir = &workspace.repo(repo)?.dir;
    let rel = normalise(path, dir);
    Some((format!("{repo}/{rel}"), repo.to_string(), rel))
}

/// The repository and the folder inside it (`""` for its root) a `file_list`
/// call asks for — only a call for the listing's first page, the one page a
/// folder unit holds. `None` for a repository the workspace does not list.
fn list_key(step: &Step, workspace: &Workspace) -> Option<(String, String)> {
    let Step::Run(call) = step else {
        return None;
    };
    if call.name != FILE_LIST {
        return None;
    }
    let args = &call.arguments;
    let first_page = match args.get("page") {
        None | Some(Value::Null) => true,
        Some(page) => page.as_u64() == Some(0),
    };
    if !first_page {
        return None;
    }
    let repo = args.get("repo")?.as_str()?;
    let dir = &workspace.repo(repo)?.dir;
    let path = match args.get("path") {
        None | Some(Value::Null) => "",
        Some(path) => path.as_str()?,
    };
    let inner = normalise(path, dir);
    let inner = inner.trim_end_matches('/');
    let inner = if inner == "." { "" } else { inner };
    Some((repo.to_string(), inner.to_string()))
}

/// A call the corpus can answer: the conversation standing in for it, admitted
/// to the target's set, and the answer to give.
struct Hit {
    served: Served,
    response: Value,
}

impl Screen<'_> {
    /// The hit for `step`, admitted to the target's fast-path set — `None`
    /// when anything is unsure, so the call runs for real.
    ///
    /// Takes the engine's `Mutex` rather than a locked engine: the lookups are
    /// git work, and holding the engine across them would stall every other
    /// conversation and the ingest worker for the length of a round. The lock
    /// is taken per candidate, around the lookup and admit only.
    fn hit(&self, step: &Step) -> Option<Hit> {
        if let Some((key, repo, rel)) = read_key(step, self.workspace) {
            return self.read_hit(key, &repo, &rel);
        }
        let (repo, inner) = list_key(step, self.workspace)?;
        self.list_hit(&repo, &inner)
    }

    /// A `file_read` of a file the conversation left alone, whose version a
    /// finished `code_reading` chain read, within the size cap.
    fn read_hit(&self, key: String, repo: &str, rel: &str) -> Option<Hit> {
        let (blob, size) = committed(self.files, repo, rel)?;
        if !fits_fast_path(size) {
            tracing::debug!(
                target: "zend::fast_path",
                path = %key,
                bytes = size,
                "file is past the fast-path size cap — reading it for real",
            );
            return None;
        }
        let looked_up = {
            let e = self.engine.lock().unwrap();
            e.find_conversations_by_metadata(CONTENT_KEY, &file_key(&key, &blob))
                .into_iter()
                // A read whose chain never finished holds no summary: handed
                // over as "already read", it would put an assistant that
                // deliberates and answers nothing into this conversation.
                .filter(|&tl| chain_finished(&e, tl))
                .find_map(|tl| Some((tl, lines_of(&e, tl)?)))
                // Admit BEFORE answering, under the same lock: a read the
                // budget refuses is not in the projection, so claiming it
                // would be a lie.
                .filter(|(tl, _)| e.fast_path_admit(self.target, *tl, self.budget_tokens))
        };
        let Some((timeline, lines)) = looked_up else {
            tracing::debug!(
                target: "zend::fast_path",
                path = %key,
                "no admitted conversation carries this file — reading it for real",
            );
            return None;
        };
        Some(Hit {
            response: served_response(repo, rel, lines),
            served: Served {
                tool: FILE_READ,
                path: key,
                timeline,
            },
        })
    }

    /// A `file_list` of a folder whose finished `repo_map` unit the
    /// conversation's base lists exactly as the unit shows it.
    fn list_hit(&self, repo: &str, inner: &str) -> Option<Hit> {
        let dir = if inner.is_empty() {
            format!("{repo}/")
        } else {
            format!("{repo}/{inner}/")
        };
        // Looked up before the engine is locked: the lookup takes it itself.
        let unit = (self.folder_of)(repo, inner);
        let admitted = unit.filter(|&tl| {
            let e = self.engine.lock().unwrap();
            chain_finished(&e, tl) && e.fast_path_admit(self.target, tl, self.budget_tokens)
        });
        let Some(timeline) = admitted else {
            tracing::debug!(
                target: "zend::fast_path",
                path = %dir,
                "no admitted conversation carries this folder — listing it for real",
            );
            return None;
        };
        Some(Hit {
            response: listed_response(),
            served: Served {
                tool: FILE_LIST,
                path: dir,
                timeline,
            },
        })
    }
}

/// Replace every `file_read` and `file_list` in `steps` whose result the
/// corpus already holds with an answer carrying that conversation, and admit
/// each to the target's projection.
///
/// A call is left alone — and so runs for real — whenever anything is unsure:
/// the conversation has changed what it names (its own copy is not the
/// committed one the corpus ingested), its base holds no such file or folder,
/// no finished conversation carries its key, or the conversation does not fit
/// the budget. Nothing is read: the keys come from the base's tree, the line
/// count from the conversation that read the file.
pub fn screen(screen: &Screen<'_>, steps: Vec<Step>) -> (Vec<Step>, Vec<Served>) {
    if screen.budget_tokens == 0 {
        return (steps, Vec::new());
    }
    let mut served = Vec::new();
    let out = steps
        .into_iter()
        .map(|step| {
            let Some(hit) = screen.hit(&step) else {
                return step;
            };
            let Step::Run(call) = step else {
                unreachable!("a hit is only found for a Run step")
            };
            served.push(hit.served);
            Step::Served(ToolResult {
                response: hit.response,
                call,
            })
        })
        .collect();
    (out, served)
}

/// The blob id and size of the file `rel` in `repo` as the conversation's
/// base holds it — `None` when the conversation has changed it (its own copy
/// is what its read must return, and no corpus read that), or when there is
/// no such file.
fn committed(files: &RepoFiles, repo: &str, rel: &str) -> Option<(Oid, u64)> {
    files.repo(repo).ok()?.content_id(rel).ok()?
}

/// How many lines the file `timeline` read holds, as its ingest recorded it.
/// `None` for a conversation that recorded none: an answer that cannot say
/// how much the model already has is not given.
fn lines_of(engine: &ConversationEngine, timeline: TimelineId) -> Option<usize> {
    lines_in(&engine.conversation_metadata(timeline)?)
}

fn lines_in(meta: &BTreeMap<String, String>) -> Option<usize> {
    meta.get(LINES_KEY)?.parse().ok()
}

/// Rebuild the target's fast-path set by replaying its own `file_read` and
/// `file_list` calls.
///
/// The set is in-memory, so a restart loses it while the conversation it
/// describes is still durable — and a conversation resumed without it would be
/// told nothing is in context, re-read every file, and quietly undo the saving.
///
/// The conversation's turns are the record: each assistant turn carries the
/// `<tool_call>` blocks it wrote, which `tool_round::plan` already parses. The
/// keys are taken again from `files` — the conversation's own view of each
/// branch — rather than stored, so a file its base holds at another blob now
/// keys to a miss and is read again — which is the correct answer, and one no
/// persisted table could have given.
///
/// Oldest turn first, so the most recent read ends up at the front of the set
/// exactly as it would have during the live conversation.
pub fn rebuild(screen: &Screen<'_>) -> usize {
    if screen.budget_tokens == 0 {
        return 0;
    }
    let texts = {
        let e = screen.engine.lock().unwrap();
        e.fast_path_clear(screen.target);
        e.assistant_turn_texts(screen.target)
    };
    let admitted = texts
        .iter()
        .flat_map(|text| plan(text))
        .filter(|step| screen.hit(step).is_some())
        .count();
    if admitted > 0 {
        tracing::info!(
            target: "zend::fast_path",
            timeline = screen.target.raw(),
            admitted,
            "rebuilt the fast-path set from the conversation's own reads and listings",
        );
    }
    admitted
}

#[cfg(test)]
mod tests {
    use super::*;

    use zend_vfs::{ObjectFormat, RepoSpec};

    /// A repository's folder, the base [`normalise`] strips.
    fn ws() -> &'static Path {
        Path::new("D:/prog/candle")
    }

    fn workspace() -> Workspace {
        Workspace::new("D:/prog", vec![RepoSpec::named("candle")]).unwrap()
    }

    /// **A read's key is its repository plus the normalised path** — the
    /// workspace-relative key the ingest hashed the file under.
    #[test]
    fn a_read_is_keyed_by_its_repository_and_path() {
        let step = call(
            FILE_READ,
            json!({"repo": "candle", "path": "./zend/src/main.rs"}),
        );
        assert_eq!(
            read_key(&step, &workspace()),
            Some((
                "candle/zend/src/main.rs".to_string(),
                "candle".to_string(),
                "zend/src/main.rs".to_string()
            ))
        );
    }

    /// **An unlisted repository is never joined onto the workspace.** The call
    /// runs for real and the tool refuses it; the fast path reads nothing.
    #[test]
    fn an_unlisted_repository_is_not_a_candidate() {
        for repo in ["other", "..", ""] {
            let step = call(FILE_READ, json!({"repo": repo, "path": "x.rs"}));
            assert_eq!(read_key(&step, &workspace()), None, "{repo:?}");
        }
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

    fn call(name: &str, args: serde_json::Value) -> Step {
        Step::Run(crate::tools::ToolCall {
            name: name.to_string(),
            arguments: args,
        })
    }

    /// **A served call is a success, and must not read as a failure.**
    ///
    /// The GUI marks a tool card red when the response carries `error` OR
    /// `detail` (`is_error`, `zend/web/index.html`), and the model reads a
    /// failed call as grounds to try again — which here would mean doing the
    /// very read the fast path just avoided. The first version used `detail`
    /// for its prose and showed up as an error in the GUI.
    #[test]
    fn the_served_response_carries_no_failure_marker() {
        let response = served_response("candle", "Cargo.toml", 18);
        assert!(response.get("error").is_none(), "{response}");
        assert!(response.get("detail").is_none(), "{response}");
        assert_eq!(response["status"], "already_read");
        assert_eq!(response["repo"], "candle");
        assert_eq!(response["path"], "Cargo.toml");
        assert!(
            response["note"].as_str().unwrap().contains("18 lines"),
            "the note tells the model how much it already has: {response}",
        );
    }

    /// **A served listing is short and a success.** The whole answer is these
    /// bytes, so its size is pinned exactly — a note that grew back to a
    /// paragraph would cost more than the small listing it stands in for.
    #[test]
    fn the_listed_response_is_short_and_carries_no_failure_marker() {
        assert_eq!(
            serde_json::to_string(&listed_response()).unwrap(),
            r#"{"status":"already_listed","note":"Listing already in context."}"#
        );
    }

    /// A listing is keyed by its repository and the folder inside it, however
    /// the call spells the folder; the root is the empty folder.
    #[test]
    fn a_listing_is_keyed_by_its_repository_and_folder() {
        let key = |args: serde_json::Value| list_key(&call(FILE_LIST, args), &workspace());
        let src = Some(("candle".to_string(), "zend/src".to_string()));
        assert_eq!(key(json!({"repo": "candle", "path": "zend/src"})), src);
        assert_eq!(key(json!({"repo": "candle", "path": "./zend/src/"})), src);
        assert_eq!(key(json!({"repo": "candle", "path": "zend\\src"})), src);
        let root = Some(("candle".to_string(), String::new()));
        assert_eq!(key(json!({"repo": "candle"})), root);
        assert_eq!(key(json!({"repo": "candle", "path": ""})), root);
        assert_eq!(key(json!({"repo": "candle", "path": "."})), root);
        assert_eq!(
            key(json!({"repo": "candle", "path": null, "page": 0})),
            root
        );
    }

    /// **Only the first page is served** — it is the one page a folder unit
    /// holds — and an unlisted repository never is.
    #[test]
    fn a_later_page_or_an_unlisted_repository_is_not_a_candidate() {
        let key = |args: serde_json::Value| list_key(&call(FILE_LIST, args), &workspace());
        assert_eq!(key(json!({"repo": "candle", "page": 1})), None);
        assert_eq!(key(json!({"repo": "candle", "page": "0"})), None);
        assert_eq!(key(json!({"repo": "other"})), None);
        assert_eq!(key(json!({"path": "zend/src"})), None);
        assert_eq!(
            list_key(&call(FILE_READ, json!({"repo": "candle"})), &workspace()),
            None
        );
    }

    /// The cap is on the file, checked at its boundary: a file estimated at
    /// exactly the ceiling is still carried, one token past it is not.
    #[test]
    fn the_size_cap_admits_up_to_the_ceiling_and_no_further() {
        let ceiling = (MAX_FAST_PATH_FILE_TOKENS * BYTES_PER_TOKEN) as u64;
        assert!(fits_fast_path(ceiling));
        assert!(!fits_fast_path(ceiling + BYTES_PER_TOKEN as u64));
        assert!(fits_fast_path(0), "an empty file is not oversized");
    }

    #[test]
    fn a_file_read_offers_its_repository_and_path() {
        let step = call(
            FILE_READ,
            json!({"repo": "candle", "path": "zend/src/main.rs"}),
        );
        assert_eq!(read_target(&step), Some(("candle", "zend/src/main.rs")));
    }

    /// Only `file_read` is content-addressed. Another tool naming a `path` —
    /// `write`, say — must never be served from an earlier read of that file.
    #[test]
    fn another_tool_with_a_path_is_not_a_candidate() {
        let step = call(
            "write",
            json!({"repo": "candle", "path": "zend/src/main.rs"}),
        );
        assert_eq!(read_target(&step), None);
    }

    #[test]
    fn a_file_read_without_a_path_or_repo_is_not_a_candidate() {
        let step = call(FILE_READ, json!({"start": 1, "end": 200}));
        assert_eq!(read_target(&step), None);
        let step = call(FILE_READ, json!({"path": "zend/src/main.rs"}));
        assert_eq!(read_target(&step), None);
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
        let files = RepoFiles::overlay(ws);
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
        let mut meta = BTreeMap::from([(CONTENT_KEY.to_string(), "k".to_string())]);
        assert_eq!(lines_in(&meta), None);
        meta.insert(LINES_KEY.to_string(), "42".to_string());
        assert_eq!(lines_in(&meta), Some(42));
        meta.insert(LINES_KEY.to_string(), "many".to_string());
        assert_eq!(lines_in(&meta), None);
    }

    /// An already-answered step is never re-examined — it has no file to read.
    #[test]
    fn an_already_answered_step_is_not_a_candidate() {
        let step = Step::Served(ToolResult {
            call: crate::tools::ToolCall {
                name: FILE_READ.to_string(),
                arguments: json!({"repo": "candle", "path": "a.rs"}),
            },
            response: json!({}),
        });
        assert_eq!(read_target(&step), None);
    }
}
