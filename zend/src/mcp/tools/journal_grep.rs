//! `journal_grep {journal, pattern, entry?, max_results?}`: the lines of a
//! journal's turns that hold `pattern` — one worker's, or a whole thread's
//! when Claude does not yet know which worker learned something
//! (`docs/claude_zend_gateway.md` §5, §5.3).

use serde_json::{json, Value};

use super::{string, Answer, Ctx, LOADING};
use crate::mcp::names::Path;
use crate::mcp::threads::{self, project_journal, Worker};

pub const NAME: &str = "journal_grep";

/// The most hits one search answers, as the journal's own search allows.
const MAX_HITS: u64 = 50;
const DEFAULT_HITS: u64 = 20;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Search journals for a word or phrase (case-insensitive substring), to find the \
            evidence behind a claim without paging through it. `journal` is a worker's path \
            (`T:<thread>/<name>`) to search the turns behind its entries — or one `entry` of them — or a \
            thread's subtree (`T:<thread>/*`) to search every worker on it, `\"self\"` / `T:<thread>` \
            for the thread's own journal, or `P:<repository>` for a project's standing rules. Each hit \
            names its journal, entry, turn and line; read around one with `journal_read` and the entry.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "journal": {"type": "string", "description": "`T:<thread>/<name>`, `T:<thread>/*`, `T:<thread>`, `P:<repository>` or `\"self\"`."},
                "pattern": {"type": "string", "description": "The text to find, ignoring case."},
                "entry": {"type": "integer", "minimum": 0, "description": "One of a worker's entries, to search it alone."},
                "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_HITS, "description": "Hits to answer (20 by default)."}
            },
            "required": ["journal", "pattern"],
            "additionalProperties": false
        }
    })
}

/// What `journal` names to search.
enum Scope {
    /// A thread's or project's own journal, under the name its notes are
    /// kept by, and its path.
    Notes(String, String),
    /// One worker's turns.
    Worker(String, String),
    /// Every worker on the thread.
    Subtree(String),
}

fn scope(ctx: &Ctx, journal: &str) -> Result<Scope, String> {
    let thread = |t: String| Scope::Notes(format!("T:{t}"), t);
    if journal == "self" {
        return ctx.attached().map(thread).ok_or_else(|| {
            "`self` names the attached thread, and none is attached: call `zend_attach` first"
                .into()
        });
    }
    if let Some(prefix) = journal.strip_suffix("/*") {
        return match Path::parse(prefix)? {
            Path::Thread(t) => Ok(Scope::Subtree(t)),
            _ => Err(format!("`{journal}`: a subtree is `T:<thread>/*`")),
        };
    }
    Ok(match Path::parse(journal)? {
        Path::Project(repo) => Scope::Notes(project_journal(&repo), project_journal(&repo)),
        Path::Thread(t) => thread(t),
        Path::Worker { thread, worker } => Scope::Worker(thread, worker),
    })
}

/// The hits in `worker`'s turns, each line under the worker's address.
fn grep_worker(
    ctx: &Ctx,
    w: &Worker,
    pattern: &str,
    entry: Option<u64>,
    max: usize,
) -> Result<(Vec<String>, usize), String> {
    match ctx.session.journal_grep(&w.conv_id, pattern, entry, max) {
        None => Err(LOADING.into()),
        Some(Err(miss)) => Err(format!("{}: {}", w.path(), miss.code())),
        Some(Ok(hits)) => Ok((
            hits.hits
                .iter()
                .map(|h| {
                    format!(
                        "{}#{} turn {} line {}: {}",
                        w.path(),
                        h.entry,
                        h.part,
                        h.line,
                        h.text.trim()
                    )
                })
                .collect(),
            hits.total,
        )),
    }
}

/// What a search answers: its hits, and how many there were in all.
fn answer(lines: Vec<String>, total: usize, pattern: &str) -> Answer {
    if lines.is_empty() {
        return Answer::ok(format!("No line holds `{pattern}`."));
    }
    let shown = lines.len();
    let mut out = lines.join("\n");
    if total > shown {
        out.push_str(&format!(
            "\n({shown} of {total} hits shown; narrow the pattern or name an entry)"
        ));
    }
    Answer::ok(out)
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let (journal, pattern) = match (string(args, "journal"), string(args, "pattern")) {
        (Ok(j), Ok(p)) => (j, p),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    let scope = match scope(ctx, journal) {
        Ok(s) => s,
        Err(e) => return Answer::error(e),
    };
    let max = args
        .get("max_results")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_HITS)
        .clamp(1, MAX_HITS) as usize;
    let entry = args.get("entry").and_then(Value::as_u64);
    let Some(records) = ctx.session.shared_substrate() else {
        return Answer::error("zend is still loading its model; try again shortly");
    };
    let owner = ctx.caller.key();
    match scope {
        Scope::Notes(path, key) => {
            if path.starts_with("T:") && threads::thread(&records, &owner, &key).is_none() {
                return Answer::error(format!("there is no thread `{path}`"));
            }
            let needle = pattern.to_lowercase();
            let lines: Vec<String> = threads::notes(&records, &owner, &key)
                .iter()
                .enumerate()
                .filter(|(_, n)| {
                    format!("{} {}", n.label, n.text)
                        .to_lowercase()
                        .contains(&needle)
                })
                .map(|(i, n)| format!("{path}#{} {} {}", i + 1, n.label, n.text))
                .collect();
            let total = lines.len();
            answer(lines.into_iter().take(max).collect(), total, pattern)
        }
        Scope::Worker(thread, name) => {
            let Some(w) = threads::worker(&records, &owner, &thread, &name) else {
                return Answer::error(format!("there is no worker `T:{thread}/{name}`"));
            };
            match grep_worker(ctx, &w, pattern, entry, max) {
                Ok((lines, total)) => answer(lines, total, pattern),
                Err(e) => Answer::error(e),
            }
        }
        Scope::Subtree(thread) => {
            if entry.is_some() {
                return Answer::error(
                    "`entry` names one worker's entry; search a worker's path with it",
                );
            }
            if !ctx.session.is_loaded() {
                return Answer::error(LOADING);
            }
            let mut lines = Vec::new();
            let mut total = 0;
            for w in threads::workers(&records, &owner, &thread) {
                // A worker with nothing kept yet has nothing to search.
                if let Ok((found, n)) = grep_worker(ctx, &w, pattern, None, max) {
                    total += n;
                    lines.extend(found);
                }
            }
            lines.truncate(max);
            answer(lines, total, pattern)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_search_says_when_it_shows_only_some_hits() {
        let a = answer(vec!["T:t/w#1 turn 2 line 3: x".into()], 4, "x");
        assert_eq!(
            a.text,
            "T:t/w#1 turn 2 line 3: x\n(1 of 4 hits shown; narrow the pattern or name an entry)"
        );
        assert_eq!(answer(Vec::new(), 0, "zz").text, "No line holds `zz`.");
    }
}
