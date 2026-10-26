//! `journal_read {journal, entry?, offset?, limit?}`: a journal's entries, or
//! the turns behind one of them (`docs/claude_zend_gateway.md` §5.3).
//!
//! `journal` is `"self"` — the attached thread — or a path: a thread's journal
//! is the entries the server wrote as Claude delegated (§4); a worker's is the
//! journal its own conversation kept.

use serde_json::{json, Value};
use zend_tools::tools::journal::address;

use super::{string, Answer, Ctx, LOADING};
use crate::mcp::names::Path;
use crate::mcp::threads::{self, project_journal, Note};

pub const NAME: &str = "journal_read";

/// The most lines one page holds, as the journal's own reader allows.
const MAX_LINES: u64 = 400;
const DEFAULT_LINES: u64 = 200;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Read a journal: a thread's, a worker's or a project's. `journal` names it — \
            `\"self\"` or `T:<thread>` for a thread (every task started on it, how each ended, and the notes on \
            it); `T:<thread>/<name>` for a worker (the entries it wrote about its work); `P:<repository>` for a \
            project (its standing rules). With `entry`, or an address such as `T:<thread>/<name>#12`, it reads \
            one entry instead: a thread's or project's note, or — for a worker — the turns behind that entry, \
            `limit` lines from `offset`, which are the evidence for what it claims. Read the entry first and its \
            turns only where a claim needs checking. Re-read the thread's journal after your context is \
            compacted.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "journal": {"type": "string", "description": "`\"self\"`, `T:<thread>`, `T:<thread>/<name>` or `P:<repository>` — or an entry's address, `T:<thread>/<name>#12`."},
                "entry": {"type": "integer", "minimum": 0, "description": "An entry's number: a worker's, to page the turns behind it; a thread's or project's, to read that note."},
                "offset": {"type": "integer", "minimum": 1, "description": "The first line of a worker entry's page (1 by default)."},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LINES, "description": "Lines in a worker entry's page (200 by default)."}
            },
            "required": ["journal"],
            "additionalProperties": false
        }
    })
}

/// The path `journal` names for this call.
fn resolve(ctx: &Ctx, journal: &str) -> Result<Path, String> {
    if journal == "self" {
        return ctx.attached().map(Path::Thread).ok_or_else(|| {
            "`self` names the attached thread, and none is attached: call `thread_attach` first"
                .into()
        });
    }
    Path::parse(journal)
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let journal = match string(args, "journal") {
        Ok(j) => j,
        Err(e) => return Answer::error(e),
    };
    // An address names its entry: `T:<thread>/<name>#12`, or `#12.3` for one
    // of its claims.
    let (journal, addressed) = address(journal);
    let path = match resolve(ctx, journal) {
        Ok(p) => p,
        Err(e) => return Answer::error(e),
    };
    let entry = match one_entry(addressed, args.get("entry").and_then(Value::as_u64)) {
        Ok(e) => e,
        Err(e) => return Answer::error(e),
    };
    let mut args = args.clone();
    if let Some(fields) = args.as_object_mut() {
        match entry {
            Some(e) => fields.insert("entry".into(), json!(e)),
            None => fields.remove("entry"),
        };
    }
    let args = &args;
    let Some(records) = ctx.session.shared_substrate() else {
        return Answer::error("the model is still loading; try again shortly");
    };
    let owner = ctx.caller.key();
    match path {
        Path::Project(repo) => {
            let journal = project_journal(&repo);
            let notes = threads::notes(&records, &owner, &journal);
            notes_answer(&journal, &notes, args.get("entry").and_then(Value::as_u64))
        }
        Path::Thread(thread) => {
            if threads::thread(&records, &owner, &thread).is_none() {
                return Answer::error(format!("there is no thread `T:{thread}`"));
            }
            let notes = threads::notes(&records, &owner, &thread);
            notes_answer(
                &format!("T:{thread}"),
                &notes,
                args.get("entry").and_then(Value::as_u64),
            )
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
                        None => Answer::error("the model is still loading; try again shortly"),
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

/// The one entry a call names — by its address, by `entry`, or by both alike —
/// or why the two disagree.
fn one_entry(addressed: Option<u64>, entry: Option<u64>) -> Result<Option<u64>, String> {
    match (addressed, entry) {
        (Some(a), Some(e)) if a != e => Err(format!(
            "the address names entry {a} and `entry` names {e}: name one"
        )),
        (a, e) => Ok(a.or(e)),
    }
}

/// A thread's or project's journal at `path`: the whole of it, or with
/// `entry` that one note — its entries are its record, with no turns behind
/// them.
fn notes_answer(path: &str, notes: &[Note], entry: Option<u64>) -> Answer {
    let Some(n) = entry else {
        return Answer::ok(threads::render_journal(path, notes));
    };
    let lines = threads::note_lines(notes);
    match n.checked_sub(1).and_then(|i| lines.get(i as usize)) {
        Some(line) => Answer::ok(format!("{path}{line}")),
        None => Answer::error(format!(
            "{path} has no entry {n}: it holds {} entries",
            notes.len()
        )),
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

    /// **A thread's entry address reads that note**; one past its end says
    /// how many there are.
    #[test]
    fn a_notes_address_reads_that_note() {
        let notes = vec![
            Note {
                label: "Claude started:".into(),
                text: "verify the expiry".into(),
                at_ms: 0,
                worker: None,
            },
            Note {
                label: "Decided:".into(),
                text: "parts cite their files".into(),
                at_ms: 1,
                worker: None,
            },
        ];
        assert_eq!(
            notes_answer("T:review", &notes, Some(2)).text,
            "T:review#2 Decided: parts cite their files"
        );
        let past = notes_answer("T:review", &notes, Some(3));
        assert!(past.is_error);
        assert_eq!(past.text, "T:review has no entry 3: it holds 2 entries");
        assert!(notes_answer("T:review", &notes, None)
            .text
            .starts_with("Journal of T:review, written by Claude"));
    }

    #[test]
    fn an_address_and_an_entry_must_agree() {
        assert_eq!(one_entry(Some(3), None), Ok(Some(3)));
        assert_eq!(one_entry(None, Some(4)), Ok(Some(4)));
        assert_eq!(one_entry(Some(3), Some(3)), Ok(Some(3)));
        assert_eq!(one_entry(None, None), Ok(None));
        assert_eq!(
            one_entry(Some(2), Some(5)),
            Err("the address names entry 2 and `entry` names 5: name one".to_string())
        );
    }

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
