//! `journal_read {journal, entry?, offset?, limit?}`: a journal's entries, or
//! the turns behind one of them (`docs/claude_zend_gateway.md` §5.3).
//!
//! `journal` is `"self"` — the attached thread — or a path: a thread's journal
//! is the entries the server wrote as Claude delegated (§4); a worker's is the
//! journal its own conversation kept.

use serde_json::{json, Value};

use super::{string, Answer, Ctx, LOADING};
use crate::mcp::names::Path;
use crate::mcp::threads::{self, project_journal};

pub const NAME: &str = "journal_read";

/// The most lines one page holds, as the journal's own reader allows.
const MAX_LINES: u64 = 400;
const DEFAULT_LINES: u64 = 200;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Read a journal. `journal` is `\"self\"` for the thread this session is attached to, \
            a thread's path (`T:<thread>`) for every task started on it and how each ended, a worker's \
            path (`T:<thread>/<name>`) for the entries its conversation kept, or a project's \
            (`P:<repository>`) for its standing rules. With `entry`, pages the turns behind one of a \
            worker's entries — the evidence behind a claim — `limit` lines from `offset`. Read the entry \
            first, and the turns only where a claim needs checking. Re-read the thread's journal after \
            your context is compacted.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "journal": {"type": "string", "description": "`\"self\"`, `T:<thread>`, `T:<thread>/<name>` or `P:<repository>`."},
                "entry": {"type": "integer", "minimum": 0, "description": "A worker's entry id, to page the turns behind it."},
                "offset": {"type": "integer", "minimum": 1, "description": "The first line of the page (1 by default)."},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LINES, "description": "Lines in the page (200 by default)."}
            },
            "required": ["journal"],
            "additionalProperties": false
        }
    })
}

/// The path `journal` names for this call.
fn resolve(ctx: &Ctx, journal: &str) -> Result<Path, String> {
    if journal == "self" {
        return ctx.mcp.attached.clone().map(Path::Thread).ok_or_else(|| {
            "`self` names the attached thread, and none is attached: call `zend_attach` first"
                .into()
        });
    }
    Path::parse(journal)
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let path = match string(args, "journal").and_then(|j| resolve(ctx, j)) {
        Ok(p) => p,
        Err(e) => return Answer::error(e),
    };
    let Some(records) = ctx.session.shared_substrate() else {
        return Answer::error("zend is still loading its model; try again shortly");
    };
    let owner = ctx.caller.key();
    match path {
        Path::Project(repo) => {
            if args.get("entry").is_some() {
                return Answer::error(
                    "a project's entries are its whole record; `entry` pages a worker's turns",
                );
            }
            let journal = project_journal(&repo);
            Answer::ok(threads::render_journal(
                &journal,
                &threads::notes(&records, &owner, &journal),
            ))
        }
        Path::Thread(thread) => {
            if threads::thread(&records, &owner, &thread).is_none() {
                return Answer::error(format!("there is no thread `T:{thread}`"));
            }
            if args.get("entry").is_some() {
                return Answer::error(
                    "a thread's entries are its whole record; `entry` pages a worker's turns",
                );
            }
            Answer::ok(threads::render_journal(
                &format!("T:{thread}"),
                &threads::notes(&records, &owner, &thread),
            ))
        }
        Path::Worker { thread, worker } => {
            let Some(w) = threads::worker(&records, &owner, &thread, &worker) else {
                return Answer::error(format!("there is no worker `T:{thread}/{worker}`"));
            };
            match args.get("entry").and_then(Value::as_u64) {
                // A worker's journal is the model's to read: before it is
                // loaded there is nothing to say, which is not "no entries".
                None => match ctx.session.journal(&w.conv_id) {
                    Some(view) => Answer::ok(render_worker(&w.path(), Some(&view))),
                    None => Answer::error(LOADING),
                },
                Some(entry) => {
                    let offset = args
                        .get("offset")
                        .and_then(Value::as_u64)
                        .unwrap_or(1)
                        .max(1);
                    let limit = args
                        .get("limit")
                        .and_then(Value::as_u64)
                        .unwrap_or(DEFAULT_LINES)
                        .clamp(1, MAX_LINES);
                    match ctx.session.journal_turns(
                        &w.conv_id,
                        entry,
                        offset as usize,
                        limit as usize,
                    ) {
                        None => Answer::error("zend is still loading its model; try again shortly"),
                        Some(Ok(page)) => Answer::ok(
                            serde_json::to_string_pretty(&page).unwrap_or_else(|e| e.to_string()),
                        ),
                        Some(Err(miss)) => {
                            Answer::error(format!("{}#{entry}: {}", w.path(), miss.code()))
                        }
                    }
                }
            }
        }
    }
}

/// A worker's journal as the tool prints it: each entry under its address,
/// then its open items.
pub fn render_worker(path: &str, view: Option<&Value>) -> String {
    let mut out = format!("Journal of {path}, written by that worker");
    let entries = view
        .and_then(|v| v.get("entries"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if entries.is_empty() {
        out.push_str(" — no entries yet.");
    }
    for e in &entries {
        let id = e.get("id").and_then(Value::as_u64).unwrap_or(0);
        let text = e.get("text").and_then(Value::as_str).unwrap_or("");
        out.push_str(&format!("\n\n{path}#{id}\n{text}"));
    }
    let open: Vec<String> = view
        .and_then(|v| v.get("open"))
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|i| i.as_str().map_or_else(|| i.to_string(), str::to_string))
                .collect()
        })
        .unwrap_or_default();
    if !open.is_empty() {
        out.push_str("\n\nStill open:\n");
        out.push_str(&open.join("\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_workers_journal_prints_each_entry_under_its_address() {
        let view = json!({
            "entries": [{"id": 2, "text": "Entry 2 — read it"}],
            "open": ["check the tail"],
        });
        assert_eq!(
            render_worker("T:r/v", Some(&view)),
            "Journal of T:r/v, written by that worker\n\nT:r/v#2\nEntry 2 — read it\n\n\
             Still open:\ncheck the tail"
        );
        assert_eq!(
            render_worker("T:r/v", None),
            "Journal of T:r/v, written by that worker — no entries yet."
        );
    }
}
