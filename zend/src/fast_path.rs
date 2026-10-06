//! Serving a `file_read` from content the corpus has already ingested, by
//! locking that conversation into the dialogue's working set
//! (`docs/zend_working_set.md` §4.5).
//!
//! A `code_reading` conversation is a whole file, read once, sealed in the
//! substrate and keyed by its path and blob id (`docs/zend_branch_ingest.md`
//! §6.1). When a dialogue asks for a file its base holds at a blob a
//! conversation already read, the cheapest correct answer is not to read the
//! file again: it is to pin that conversation into this one's projection and
//! say so. The K/V is already there, so the call costs an elevation instead of
//! a prefill and a decode.
//!
//! **Only `file_read` is served.** A `file_list` always runs and its listing is
//! prefilled as the tool's response: a listing is short, so serving it saves
//! little, and the unit that would stand in for it is a placed `file_list`
//! round trip the model is told it did not make — handed that as its answer,
//! it lost track of whether it had listed the folder at all.
//!
//! **The hit test is the whole file; the lock is the whole file.** A call for
//! page 1 of a file hits on the file's content key, and what lands in context
//! is the entire read — so the requested page is necessarily present. That is
//! only true of a chain that read every page, which [`coverage`] checks.
//!
//! What it does NOT do is claim more than it delivers. A hit is only returned
//! once the conversation has been locked into the working set, which refuses
//! rather than evicts when the set is full; a refusal falls through to a real
//! read. Telling the model a file is already in context when it is not would be
//! worse than any number of redundant reads — it answers from nothing rather
//! than looking again.

pub mod coverage;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

use candle_conversation::projection::TimelineId;
use candle_conversation::working_set::WorkingSetConfig;
use candle_conversation::ConversationEngine;
use serde_json::{json, Value};
use zend_tools::tools::file::render::file_anchor;
use zend_vfs::{Oid, RepoFiles, Workspace};

use crate::branch_ingest::keys::{file_key, CONTENT_KEY, LINES_KEY};
use crate::code_read::chain_finished;
use crate::tool_round::Step;
use crate::tools::ToolResult;
use crate::working_set::releases;
use coverage::Coverage;

/// Whole-file reads are content-addressed by `code_reading`.
const FILE_READ: &str = "file_read";

/// The status a served call answers with. It is also the name of the system
/// prompt's rule for placed content (`in_context` in `projection.yaml`), so the
/// reply calls that rule to attention by the same token instead of restating
/// it: the reply stands in for a read, and every word it repeats is paid on
/// every served call.
const IN_CONTEXT: &str = "in_context";

/// The answer a served call gets: the status, and the anchor the content
/// begins with — the exact string the model will find ahead of the
/// conversation, so it looks for that rather than for a read of its own.
///
/// **No `error`, and no `detail`.** Both mark a failed call — the GUI renders
/// either as a red card (`is_error`, `zend/web/index.html`) and the model reads
/// a failure as grounds to try again, which here means doing the very read the
/// fast path just avoided.
///
/// **Not "already read".** A served file is usually one a background ingest
/// read, not this conversation. Told it had already read it, the model
/// searched its own reads, found none, and called the reply false. `in_context`
/// says what is true: the content is here, at this anchor.
fn served_response(anchor: String) -> Value {
    json!({"status": IN_CONTEXT, "anchor": anchor})
}

/// One call the fast path answered, for the caller to log and to mark on the
/// turn that carries the round's results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Served {
    /// The tool whose call was answered.
    pub tool: &'static str,
    /// Workspace-relative: `candle/src/lib.rs`.
    pub path: String,
    pub timeline: TimelineId,
}

/// What one screening consults: the conversation, its view of every
/// repository, and the working set a read locks into.
pub struct Screen<'a> {
    pub engine: &'a Mutex<ConversationEngine>,
    pub target: TimelineId,
    pub workspace: &'a Workspace,
    pub files: &'a RepoFiles,
    pub config: &'a WorkingSetConfig,
    pub coverage: &'a Coverage,
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

/// The finished, complete `code_reading` conversation that read `rel` in
/// `repo` at the blob the conversation's base holds, with its line count —
/// `None` when the conversation changed the file (its own copy is what its
/// read must return), no conversation read that version, or none that did
/// read every page.
pub fn file_conversation(
    engine: &ConversationEngine,
    files: &RepoFiles,
    coverage: &Coverage,
    repo: &str,
    rel: &str,
) -> Option<(TimelineId, usize)> {
    let (blob, _) = committed(files, repo, rel)?;
    engine
        .find_conversations_by_metadata(CONTENT_KEY, &file_key(&format!("{repo}/{rel}"), &blob))
        .into_iter()
        // A read whose chain never finished holds no summary: handed over as
        // in context, it would put an assistant that deliberates and
        // answers nothing into this conversation.
        .filter(|&tl| chain_finished(engine, tl))
        .find_map(|tl| {
            let lines = lines_of(engine, tl)?;
            coverage
                .is_complete(engine, tl, lines)
                .then_some((tl, lines))
        })
}

/// A call the corpus can answer: the conversation standing in for it, locked
/// into the target's working set, and the answer to give.
struct Hit {
    served: Served,
    response: Value,
}

impl Screen<'_> {
    /// The hit for `step`, locked into the target's working set — `None` when
    /// anything is unsure, so the call runs for real.
    ///
    /// Takes the engine's `Mutex` rather than a locked engine: the lookups are
    /// git work, and holding the engine across them would stall every other
    /// conversation and the ingest worker for the length of a round. The lock
    /// is taken per candidate, around the lookup and lock only.
    fn hit(&self, step: &Step) -> Option<Hit> {
        let (key, repo, rel) = read_key(step, self.workspace)?;
        self.read_hit(key, &repo, &rel)
    }

    /// A `file_read` of a file the conversation left alone, whose version a
    /// finished `code_reading` chain read in full.
    fn read_hit(&self, key: String, repo: &str, rel: &str) -> Option<Hit> {
        let locked = {
            let e = self.engine.lock().unwrap();
            // Locked BEFORE answering, under the same engine lock: a read the
            // working set refuses is not in the projection, so claiming it
            // would be a lie.
            file_conversation(&e, self.files, self.coverage, repo, rel)
                .map(|(tl, _)| tl)
                .filter(|&tl| self.lock(&e, tl, &key))
        };
        let Some(timeline) = locked else {
            tracing::debug!(
                target: "zend::fast_path",
                path = %key,
                "no complete conversation carries this file into the working set — reading it for real",
            );
            return None;
        };
        Some(Hit {
            response: served_response(file_anchor(repo, rel)),
            served: Served {
                tool: FILE_READ,
                path: key,
                timeline,
            },
        })
    }

    /// Lock `timeline` into the target's working set; a refusal is logged
    /// with its reason and reads as a miss.
    fn lock(&self, engine: &ConversationEngine, timeline: TimelineId, path: &str) -> bool {
        match engine.working_set_lock(self.target, timeline, self.config.limits()) {
            Ok(()) => true,
            Err(refusal) => {
                tracing::debug!(
                    target: "zend::fast_path",
                    path,
                    ?refusal,
                    "working set refused the lock — the call runs for real",
                );
                false
            }
        }
    }
}

/// A round after screening: the steps to dispatch, and the calls the working
/// set answered, in the order they were served.
#[derive(Debug)]
pub struct Screened {
    pub steps: Vec<Step>,
    pub served: Vec<Served>,
}

/// Replace every `file_read` in `steps` whose file the corpus already holds
/// with an answer carrying that conversation, locked into the target's working
/// set. Every other call runs.
///
/// **Only the calls before the round's first `release_on` call are screened.**
/// A round runs in order, and a write may change what a later read would
/// return — `[file_edit X, file_read X]` must read X as edited, not be served
/// the version the corpus holds. Everything from the first write on runs for
/// real.
///
/// A call is also left alone — and so runs for real — whenever anything is
/// unsure: the conversation has changed the file (its own copy is not the
/// committed one the corpus ingested), its base holds no such file, no
/// finished conversation read it whole, or the working set refuses
/// it. Nothing is read: the keys come from the base's tree, the line count
/// from the conversation that read the file.
pub fn screen(screen: &Screen<'_>, steps: Vec<Step>) -> Screened {
    let mut served = Vec::new();
    let cut = first_release(screen.config, &steps);
    let steps = steps
        .into_iter()
        .enumerate()
        .map(|(i, step)| {
            if i >= cut {
                return step;
            }
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
    Screened { steps, served }
}

/// The position of the round's first `release_on` call — every step before it
/// may be served, it and every step after run for real. The round's length
/// when it makes none.
fn first_release(config: &WorkingSetConfig, steps: &[Step]) -> usize {
    steps
        .iter()
        .position(|step| releases(config, step.name()))
        .unwrap_or(steps.len())
}

/// The blob id and size of the file `rel` in `repo` as the conversation's
/// base holds it — `None` when the conversation has changed it (its own copy
/// is what its read must return), or when there is no such file.
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

    /// **A served call is a short success naming its anchor**, pinned to the
    /// byte: the reply stands in for a read, so a note that grew back would be
    /// paid on every served call. The GUI marks a tool card red when the
    /// response carries `error` OR `detail` (`is_error`,
    /// `zend/web/index.html`), and the model reads a failed call as grounds to
    /// try again — doing the very read the fast path just avoided.
    #[test]
    fn a_served_call_is_the_status_and_the_anchor() {
        assert_eq!(
            serde_json::to_string(&served_response(file_anchor("candle", "Cargo.toml"))).unwrap(),
            r#"{"status":"in_context","anchor":"file=candle/Cargo.toml"}"#
        );
    }

    /// **The status is the system prompt's rule name**, so the reply recalls
    /// the rule by the same token. A rename on either side breaks the link
    /// silently; this makes it loud.
    #[test]
    fn the_status_names_the_system_prompts_rule() {
        let yaml = include_str!("prompts/projection.yaml");
        assert!(
            yaml.lines()
                .any(|l| l.trim() == format!("id: {IN_CONTEXT}")),
            "projection.yaml must declare a section named {IN_CONTEXT}"
        );
        assert!(yaml.contains(&format!("{IN_CONTEXT} — ")));
    }

    /// **A listing is never served.** It always runs and its listing is
    /// prefilled, so a folder in the corpus is no candidate, whatever the call
    /// asks for.
    #[test]
    fn a_listing_is_never_a_candidate() {
        for args in [
            json!({"repo": "candle"}),
            json!({"repo": "candle", "path": "zend/src"}),
            json!({"repo": "candle", "path": "zend/src/main.rs"}),
        ] {
            assert_eq!(
                read_key(&call("file_list", args.clone()), &workspace()),
                None,
                "{args}"
            );
        }
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

    fn config() -> WorkingSetConfig {
        WorkingSetConfig {
            budget_tokens: 1_000,
            folder_tokens: 100,
            beta: 0.2,
            min_momentum: 100.0,
            max_file_tokens: 500,
            seeds: Vec::new(),
            release_on: vec!["write".into(), "file_edit".into()],
            max_admits: 2,
        }
    }

    /// **Edit then read in one round reads the edit.** Only the calls before
    /// the round's first write may be served; from the write on, the round
    /// runs for real, so `[file_edit X, file_read X]` never returns X's
    /// pre-edit blob. The alias `file_write` is the same write.
    #[test]
    fn calls_from_the_first_write_on_run_for_real() {
        let read = || {
            call(
                FILE_READ,
                json!({"repo": "candle", "path": "a.rs", "page": 0}),
            )
        };
        let edit = call("file_edit", json!({"repo": "candle", "path": "a.rs"}));
        let alias = call("file_write", json!({"repo": "candle", "path": "a.rs"}));
        assert_eq!(first_release(&config(), &[read(), read()]), 2);
        assert_eq!(first_release(&config(), &[read(), edit.clone(), read()]), 1);
        assert_eq!(first_release(&config(), &[edit, read()]), 0);
        assert_eq!(first_release(&config(), &[read(), alias, read()]), 1);
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
