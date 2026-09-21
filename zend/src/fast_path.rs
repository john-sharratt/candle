//! Serving a `file_read` from content the corpus has already read.
//!
//! A `code_reading` conversation is a whole file, read once, sealed in the
//! substrate and content-addressed by `content_sha256`. When a dialogue asks
//! for a file whose bytes hash to a conversation that already exists, the
//! cheapest correct answer is not to read the file again: it is to carry that
//! conversation into this one's projection and say so. The K/V is already
//! there, so the call costs an elevation instead of a prefill and a decode.
//!
//! **The hit test is the whole file; the injection is the whole file.** A call
//! for page 1 of a file hits on the file's hash, and what lands in context is
//! the entire read — so the requested page is necessarily present, and no page
//! bookkeeping is needed to know it.
//!
//! What it does NOT do is claim more than it delivers. A hit is only returned
//! once the conversation has been admitted to the projection
//! (`Substrate::fast_path_admit`), which refuses a read too large for the
//! layer's budget; a refusal falls through to a real read. Telling the model a
//! file is already in context when it is not would be worse than any number of
//! redundant reads — it answers from nothing rather than looking again.

use std::path::Path;
use std::sync::Mutex;

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;
use serde_json::json;

use crate::code_read::file_content_hash;
use crate::tool_round::Step;
use crate::tools::ToolResult;

/// The tool this serves. Only whole-file reads are content-addressed, so this
/// is the only call whose result another conversation can stand in for.
const FILE_READ: &str = "file_read";

/// The substrate metadata key a `code_reading` conversation records its
/// content hash under — the content-addressed key both sides must agree on.
const HASH_KEY: &str = "content_sha256";

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
fn fits_fast_path(bytes: usize) -> bool {
    bytes / BYTES_PER_TOKEN <= MAX_FAST_PATH_FILE_TOKENS
}

/// The answer a served call gets.
///
/// **No `error`, and no `detail`.** Both mark a failed call — the GUI renders
/// either as a red card (`is_error`, `zend/web/index.html`) and the model reads
/// a failure as grounds to try again, which here means doing the very read the
/// fast path just avoided.
fn served_response(path: &str, lines: usize) -> serde_json::Value {
    json!({
        "status": "already_read",
        "path": path,
        "lines": lines,
        "note": format!(
            "`{path}` is unchanged since it was read, and its full contents \
             ({lines} lines) are already in this conversation's context — \
             including any lines this call asked for. Read it from there \
             rather than calling file_read for it again."
        ),
    })
}

/// One call the fast path answered, for the caller to log and report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Served {
    pub path: String,
    pub timeline: TimelineId,
}

/// The workspace-relative form `code_reading` hashes under.
///
/// The hash is path-qualified, so a call that names the same file differently —
/// a leading `./`, a backslash separator, an absolute path inside the
/// workspace — hashes to something else and misses every time, silently and
/// forever. Normalising here is what keeps the two sides addressing the same
/// content.
pub fn normalise(path: &str, workspace: &Path) -> String {
    let cleaned = path.replace('\\', "/");
    let cleaned = cleaned.trim_start_matches("./");
    let ws = workspace.to_string_lossy().replace('\\', "/");
    let ws = ws.trim_end_matches('/');
    cleaned
        .strip_prefix(&format!("{ws}/"))
        .unwrap_or(cleaned)
        .trim_start_matches('/')
        .to_string()
}

/// The `path` argument of a `file_read` call, when it has one.
fn read_path(step: &Step) -> Option<&str> {
    match step {
        Step::Run(call) if call.name == FILE_READ => call.arguments.get("path")?.as_str(),
        _ => None,
    }
}

/// Replace every `file_read` in `steps` whose content the corpus has already
/// read with an answer carrying that conversation, and admit each to `target`'s
/// projection.
///
/// A call is left alone — and so runs for real — whenever anything is unsure:
/// the file cannot be read from disk, no conversation carries its hash, or the
/// read does not fit `budget_tokens`.
/// Takes the engine's `Mutex` rather than a locked engine: the file reads and
/// hashing below are disk work, and holding the engine across them would stall
/// every other conversation and the ingest worker for the length of a round.
/// The lock is taken per candidate, around the lookup and admit only.
pub fn screen(
    engine: &Mutex<ConversationEngine>,
    target: TimelineId,
    workspace: &Path,
    budget_tokens: usize,
    steps: Vec<Step>,
) -> (Vec<Step>, Vec<Served>) {
    if budget_tokens == 0 {
        return (steps, Vec::new());
    }
    let mut served = Vec::new();
    let out = steps
        .into_iter()
        .map(|step| {
            let Some(raw_path) = read_path(&step) else {
                return step;
            };
            let rel = normalise(raw_path, workspace);
            let Ok(bytes) = std::fs::read(workspace.join(&rel)) else {
                return step;
            };
            if !fits_fast_path(bytes.len()) {
                tracing::debug!(
                    target: "zend::fast_path",
                    path = %rel,
                    bytes = bytes.len(),
                    "file is past the fast-path size cap — reading it for real",
                );
                return step;
            }
            let hash = file_content_hash(&rel, &bytes);
            let looked_up = {
                let e = engine.lock().unwrap();
                e.find_conversations_by_metadata(HASH_KEY, &hash)
                    .into_iter()
                    .next()
                    // Admit BEFORE answering, under the same lock: a read the
                    // budget refuses is not in the projection, so claiming it
                    // would be a lie.
                    .filter(|tl| e.fast_path_admit(target, *tl, budget_tokens))
            };
            let Some(timeline) = looked_up else {
                tracing::debug!(
                    target: "zend::fast_path",
                    path = %rel,
                    "no admitted conversation carries this file — reading it for real",
                );
                return step;
            };
            let lines = bytes.iter().filter(|b| **b == b'\n').count() + 1;
            let Step::Run(call) = step else {
                unreachable!("read_path matched a Run step")
            };
            served.push(Served {
                path: rel.clone(),
                timeline,
            });
            Step::Served(ToolResult {
                response: served_response(&rel, lines),
                call,
            })
        })
        .collect();
    (out, served)
}

/// Rebuild `target`'s fast-path set by replaying its own `file_read` calls.
///
/// The set is in-memory, so a restart loses it while the conversation it
/// describes is still durable — and a conversation resumed without it would be
/// told nothing is in context, re-read every file, and quietly undo the saving.
///
/// The conversation's turns are the record: each assistant turn carries the
/// `<tool_call>` blocks it wrote, which `tool_round::plan` already parses. The
/// hashes are recomputed from disk rather than stored, so a file edited while
/// the daemon was down re-hashes to a miss and is read again — which is the
/// correct answer, and one no persisted table could have given.
///
/// Oldest turn first, so the most recent read ends up at the front of the set
/// exactly as it would have during the live conversation.
pub fn rebuild(
    engine: &Mutex<ConversationEngine>,
    target: TimelineId,
    workspace: &Path,
    budget_tokens: usize,
) -> usize {
    if budget_tokens == 0 {
        return 0;
    }
    let texts = {
        let e = engine.lock().unwrap();
        e.fast_path_clear(target);
        e.assistant_turn_texts(target)
    };
    let mut admitted = 0usize;
    for text in texts {
        for step in crate::tool_round::plan(&text) {
            let Some(raw_path) = read_path(&step) else {
                continue;
            };
            let rel = normalise(raw_path, workspace);
            let Ok(bytes) = std::fs::read(workspace.join(&rel)) else {
                continue;
            };
            if !fits_fast_path(bytes.len()) {
                continue;
            }
            let hash = file_content_hash(&rel, &bytes);
            let e = engine.lock().unwrap();
            if let Some(tl) = e
                .find_conversations_by_metadata(HASH_KEY, &hash)
                .into_iter()
                .next()
            {
                if e.fast_path_admit(target, tl, budget_tokens) {
                    admitted += 1;
                }
            }
        }
    }
    if admitted > 0 {
        tracing::info!(
            target: "zend::fast_path",
            timeline = target.raw(),
            admitted,
            "rebuilt the fast-path set from the conversation's own reads",
        );
    }
    admitted
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws() -> &'static Path {
        Path::new("D:/prog/candle")
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
    fn an_absolute_path_inside_the_workspace_becomes_relative() {
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

    /// A path outside the workspace keeps its shape — it will simply find no
    /// conversation, which is the correct outcome rather than a false hit.
    #[test]
    fn a_path_outside_the_workspace_is_left_alone() {
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
        let response = served_response("Cargo.toml", 18);
        assert!(response.get("error").is_none(), "{response}");
        assert!(response.get("detail").is_none(), "{response}");
        assert_eq!(response["status"], "already_read");
        assert_eq!(response["path"], "Cargo.toml");
        assert!(
            response["note"].as_str().unwrap().contains("18 lines"),
            "the note tells the model how much it already has: {response}",
        );
    }

    /// The cap is on the file, checked at its boundary: a file estimated at
    /// exactly the ceiling is still carried, one token past it is not.
    #[test]
    fn the_size_cap_admits_up_to_the_ceiling_and_no_further() {
        assert!(fits_fast_path(MAX_FAST_PATH_FILE_TOKENS * BYTES_PER_TOKEN));
        assert!(!fits_fast_path(
            MAX_FAST_PATH_FILE_TOKENS * BYTES_PER_TOKEN + BYTES_PER_TOKEN
        ));
        assert!(fits_fast_path(0), "an empty file is not oversized");
    }

    #[test]
    fn a_file_read_offers_its_path() {
        let step = call(FILE_READ, json!({"path": "zend/src/main.rs"}));
        assert_eq!(read_path(&step), Some("zend/src/main.rs"));
    }

    /// Only `file_read` is content-addressed. Another tool naming a `path` —
    /// `write`, say — must never be served from an earlier read of that file.
    #[test]
    fn another_tool_with_a_path_is_not_a_candidate() {
        let step = call("write", json!({"path": "zend/src/main.rs"}));
        assert_eq!(read_path(&step), None);
    }

    #[test]
    fn a_file_read_without_a_path_is_not_a_candidate() {
        let step = call(FILE_READ, json!({"start": 1, "end": 200}));
        assert_eq!(read_path(&step), None);
    }

    /// An already-answered step is never re-examined — it has no file to read.
    #[test]
    fn an_already_answered_step_is_not_a_candidate() {
        let step = Step::Served(ToolResult {
            call: crate::tools::ToolCall {
                name: FILE_READ.to_string(),
                arguments: json!({"path": "a.rs"}),
            },
            response: json!({}),
        });
        assert_eq!(read_path(&step), None);
    }
}
