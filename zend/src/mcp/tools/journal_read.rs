//! `journal_read {journal, entry?, offset?, limit?}`: a journal's entries, or
//! the turns behind one of them (`docs/claude_zend_gateway.md` §5.3).
//!
//! `journal` is `"self"` — the attached thread — or a path: a thread's journal
//! is the entries the server wrote as Claude delegated (§4); a worker's is the
//! journal its own conversation kept. A memory path — `@group/…`, a section,
//! `here` — opens one of the memory's journals (`docs/memory_backbone.md`
//! §6.5), or with `#<line id>` reads one line of it.

use journals::path::JournalPath;
use journals::time::Stamp;
use journals::tree::WINDOW;
use journals::view::open::open;
use journals::view::page::before;
use journals::view::who::render_line;
use journals::view::NOT_INSTRUCTIONS;
use serde_json::{json, Value};
use zend_tools::tools::journal::address;

use super::fit::clip_lines;
use super::mem::PATHS;
use super::{mem, string, Answer, Ctx, LOADING};
use crate::mcp::memory::{is_memory_path, Memory};
use crate::mcp::names::Path;
use crate::mcp::threads::{self, project_journal, Numbered};

pub const NAME: &str = "journal_read";

/// The most lines one page holds, as the journal's own reader allows.
const MAX_LINES: u64 = 400;
const DEFAULT_LINES: u64 = 200;
/// How many characters of a worker's final answer one read answers.
const ANSWER_PAGE: usize = 16_000;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": format!("Read a journal. The memory's: `@<group>` is the master — what the group is, \
            who is in it, the few things that last; below it `people/<person>` (each member, Zen among them), \
            `topics/<topic>` (projects and subjects, broken down into sub-topics), `conversations/<began>-<channel>` \
            and `here`, this conversation. Each opens with its summary, its references, its sub-journals by cue \
            and its newest lines; walk down by cue to the journal a question is about, and read one line \
            with `<path>#<line id>`. {PATHS} Zen's: `\"self\"` or `T:<thread>` for a thread (every task started \
            on it, how each ended, and the notes on it); `T:<thread>/<name>` for a worker (the entries it \
            wrote about its work); `P:<repository>` for a project (its standing rules). With `entry`, or an \
            address such as `T:<thread>/<name>#12`, it reads one entry instead: a thread's or project's note, \
            or — for a worker — the turns behind that entry, `limit` lines from `offset`, which are the \
            evidence for what it claims. Read the entry first and its turns only where a claim needs \
            checking. Re-read the thread's journal after your context is compacted."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "journal": {"type": "string", "description": "`@<group>`, a memory path such as `topics/candle` or `here`, or `<path>#<line id>`; or `\"self\"`, `T:<thread>`, `T:<thread>/<name>`, `P:<repository>`, or an entry's address, `T:<thread>/<name>#12`."},
                "entry": {"type": "integer", "minimum": 0, "description": "An entry's number: a worker's, to page the turns behind it; a thread's or project's, to read that note."},
                "offset": {"type": "integer", "minimum": 1, "description": "With `entry`, the first line of a worker entry's page; without it, the first entry of a thread's, project's or worker's journal (1 by default)."},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LINES, "description": "Lines in a worker entry's page (200 by default), or in a memory journal's page with `after` (50 by default)."},
                "after": {"type": "string", "description": "Memory only: read the lines before this one, `<path>#<line id>` — the oldest line the read before showed, as its answer names it — going on into the journal's archives."},
                "answer_from": {"type": "integer", "minimum": 0, "description": "For a worker's path: read its final answer instead, a page of characters from this one (0 to start)."}
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
    if is_memory_path(journal) {
        return read_memory(ctx, journal, args);
    }
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
            notes_answer(
                &journal,
                &notes,
                args.get("entry").and_then(Value::as_u64),
                first_entry(args),
            )
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
                first_entry(args),
            )
        }
        Path::Worker { thread, worker } => {
            let Some(w) = threads::worker(&records, &owner, &thread, &worker) else {
                return Answer::error(format!("there is no worker `T:{thread}/{worker}`"));
            };
            if let Some(from) = args.get("answer_from").and_then(Value::as_u64) {
                return Answer::ok(answer_page(&w.path(), &w.final_text, from as usize));
            }
            match args.get("entry").and_then(Value::as_u64) {
                // A worker's journal is the model's to read: before it is
                // loaded there is nothing to say, which is not "no entries".
                None => match ctx.session.journal(&w.conv_id) {
                    Some(view) => {
                        Answer::ok(render_worker(&w.path(), Some(&view), first_entry(args)))
                    }
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
                        // A tool result is one stored line, so a page of
                        // lines can hold a whole file in one of them.
                        Some(Ok(page)) => Answer::ok(clip_lines(
                            &serde_json::to_string_pretty(&page).unwrap_or_else(|e| e.to_string()),
                        )),
                        Some(Err(miss)) => {
                            Answer::error(format!("{}#{entry}: {}", w.path(), miss.code()))
                        }
                    }
                }
            }
        }
    }
}

/// One of the memory's journals, opened; one line of it, by its address; or,
/// with `after`, a page of the lines before that one.
fn read_memory(ctx: &Ctx, journal: &str, args: &Value) -> Answer {
    let m = match mem::read(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let actor = m.actor();
    if journal.contains('#') {
        let address = match m.address(journal) {
            Ok(a) => a,
            Err(e) => return Answer::error(e),
        };
        let j = match Memory::readable(&m.tree, &actor, &address.path) {
            Ok(j) => j,
            Err(e) => return Answer::error(e),
        };
        return match j.line(&address.id) {
            Some(line) => Answer::ok(format!(
                "{NOT_INSTRUCTIONS}\n{} {}",
                address.path,
                render_line(&m.tree, line)
            )),
            None => Answer::error(format!("{address} is no line")),
        };
    }
    if let Some(here) = m.here_unopened(journal) {
        return Answer::ok(unopened(here));
    }
    let path = match m.path(journal) {
        Ok(p) => p,
        Err(e) => return Answer::error(e),
    };
    let j = match Memory::readable(&m.tree, &actor, &path) {
        Ok(j) => j,
        Err(e) => return Answer::error(e),
    };
    // A page of what came before a line: `after` is the line the page before
    // began with, as its answer names it.
    match args.get("after").and_then(Value::as_str) {
        None => Answer::ok(open(&m.tree, &path, j, WINDOW, Stamp::now())),
        Some(after) => {
            let address = match m.find_address(after) {
                Ok(a) if a.path == path => a,
                Ok(a) => {
                    return Answer::error(format!(
                        "`after` names a line of {}, not of {path}",
                        a.path
                    ))
                }
                Err(e) => return Answer::error(String::from(e)),
            };
            let lines = args
                .get("limit")
                .and_then(Value::as_u64)
                .map_or(WINDOW, |n| n as usize);
            match before(&m.tree, &path, j, &address.id, lines, Stamp::now()) {
                Some(page) => Answer::ok(page),
                None => Answer::error(format!("{address} is no line")),
            }
        }
    }
}

/// This conversation's journal before anything has written to it: empty,
/// and named as its first write will open it.
pub fn unopened(here: &JournalPath) -> String {
    format!(
        "{NOT_INSTRUCTIONS}\n{here} (here). Nothing is written in it yet: it opens at this \
         conversation's first write."
    )
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

/// The first entry a journal's read starts at: `offset`, or the first.
fn first_entry(args: &Value) -> usize {
    args.get("offset")
        .and_then(Value::as_u64)
        .map_or(1, |o| o.max(1) as usize)
}

/// One page of a worker's final answer from character `from`, and where to
/// read on.
fn answer_page(path: &str, text: &str, from: usize) -> String {
    let total = text.chars().count();
    let to = (from + ANSWER_PAGE).min(total);
    let page: String = text.chars().skip(from).take(ANSWER_PAGE).collect();
    let mut out = format!(
        "{path}'s answer, characters {}–{to} of {total}",
        from.min(total)
    );
    if to < total {
        out.push_str(&format!(
            " (call again with `answer_from: {to}` to read on)"
        ));
    }
    out.push('\n');
    out.push_str(&page);
    out
}

/// A thread's or project's journal at `path`: its entries from `first` on, or
/// with `entry` that one note — its entries are its record, with no turns
/// behind them.
fn notes_answer(path: &str, notes: &[Numbered], entry: Option<u64>, first: usize) -> Answer {
    let Some(n) = entry else {
        if first <= 1 {
            return Answer::ok(threads::render_journal(path, notes));
        }
        let shown: Vec<&Numbered> = notes.iter().filter(|e| e.number >= first as u64).collect();
        let last = shown.iter().map(|e| e.number).max().unwrap_or(first as u64);
        let lines: Vec<String> = shown.into_iter().map(threads::note_line).collect();
        return Answer::ok(format!(
            "{}, entries {first}–{last} of {}\n{}",
            threads::journal_head(path),
            notes.len(),
            lines.join("\n")
        ));
    };
    match notes.iter().find(|e| e.number == n) {
        Some(e) => Answer::ok(format!("{path}{}", threads::note_line(e))),
        None => Answer::error(format!(
            "{path} has no entry {n}: it holds {} entries",
            notes.len()
        )),
    }
}

/// A worker's journal as the tool prints it: each entry from `first` on under
/// its address, then its open items.
pub fn render_worker(path: &str, view: Option<&Value>, first: usize) -> String {
    let mut out = format!("Journal of {path}, written by that worker");
    let entries = view
        .and_then(|v| v.get("entries"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if entries.is_empty() {
        out.push_str(" — no entries yet.");
    }
    for e in entries.iter().skip(first.saturating_sub(1)) {
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
    use crate::mcp::threads::Note;

    /// **A thread's entry address reads that note**; one past its end says
    /// how many there are.
    #[test]
    fn a_notes_address_reads_that_note() {
        let entry = |number, label: &str, text: &str| Numbered {
            number,
            note: Note {
                label: label.into(),
                text: text.into(),
                at_ms: number,
                worker: None,
                client: None,
            },
        };
        let notes = vec![
            entry(1, "Claude started:", "verify the expiry"),
            entry(2, "Decided:", "parts cite their files"),
        ];
        assert_eq!(
            notes_answer("T:review", &notes, Some(2), 1).text,
            "T:review#2 Decided: parts cite their files"
        );
        let past = notes_answer("T:review", &notes, Some(3), 1);
        assert!(past.is_error);
        assert_eq!(past.text, "T:review has no entry 3: it holds 2 entries");
        assert!(notes_answer("T:review", &notes, None, 1)
            .text
            .starts_with("Journal of T:review, written by Claude"));
        assert_eq!(
            notes_answer("T:review", &notes, None, 2).text,
            "Journal of T:review, written by Claude, entries 2–2 of 2\n\
             #2 Decided: parts cite their files"
        );
    }

    /// **A long answer reads on a page at a time**, for a caller who cannot
    /// reach the worker's conversation itself.
    #[test]
    fn a_workers_answer_reads_on_a_page_at_a_time() {
        let text = "é".repeat(ANSWER_PAGE + 3);
        let first = answer_page("T:r/v", &text, 0);
        assert!(first.starts_with(
            "T:r/v's answer, characters 0–16000 of 16003 (call again with `answer_from: 16000` \
             to read on)\n"
        ));
        assert_eq!(
            answer_page("T:r/v", &text, 16_000),
            "T:r/v's answer, characters 16000–16003 of 16003\nééé"
        );
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
            render_worker("T:r/v", Some(&view), 1),
            "Journal of T:r/v, written by that worker\n\nT:r/v#2\nEntry 2 — read it\n\n\
             Still open:\ncheck the tail"
        );
        assert_eq!(
            render_worker("T:r/v", None, 1),
            "Journal of T:r/v, written by that worker — no entries yet."
        );
        assert_eq!(
            render_worker("T:r/v", Some(&view), 2),
            "Journal of T:r/v, written by that worker\n\nStill open:\ncheck the tail",
            "an offset past the entries leaves its open items"
        );
    }
}
