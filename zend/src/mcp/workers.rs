//! Running a task on a worker, answering a worker that asked, and waiting on
//! workers (`docs/claude_zend_gateway.md` §3.3, §4, §5, §6.2).
//!
//! A task's turn is an ordinary dialogue turn on the worker's own zend
//! conversation, submitted as `POST /v1/chat/completions` submits one, so it is
//! projected with provenance, the working set and the journal like any
//! dialogue. It runs in a task of its own, never inside a tool call: the call
//! that started it answers within `wait_s`, and `job_status` reads how it stands.
//!
//! When a turn ends, the task asks for a journal draft at once instead of
//! waiting for the cadence, waits for it, and takes the entries written since
//! the turn began. Then the worker settles:
//!
//! - **asking** — its final response ended with the question block of
//!   [`ASKING`] — it waits, live, for `repo_task_reply`;
//! - **stuck** — the repeat guard or the round cap closed its tools — it waits
//!   the same way;
//! - **done** — its answer's citations are audited (`super::audit`); an
//!   answer citing what it never opened is sent back once to open it. Then it
//!   is kept whole, to be continued with `repo_task_reply`, until its thread is
//!   archived or it sits a day idle (`super::archiving`);
//! - **failed** — it is archived, its evidence kept.
//!
//! Each settling is entered on the thread. An answer is the worker's next turn,
//! and the worker settles again after it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::http::{HeaderName, HeaderValue, Method};
use candle_conversation::persistence::SharedSubstrate;
use futures::StreamExt;
use host::cluster::up::send_up;
use host::substrate_views::TimelineDetail;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::Notify;
use web::auth::Role as AccessRole;

use super::activity::{Activity, Doing};
use super::archiving;
use super::audit::{Audit, Turn};
use super::caller::{Caller, Person};
use super::chats::naming::is_chat_thread;
use super::chats::talk;
use super::names;
use super::nudges::{self, Sender};
use super::ops::NODE_HEADER;
use super::driven::{Driven, DRIVEN_SELECTOR};
use super::outcomes::{self, Outcome};
use super::records::Records;
use super::resuming;
use super::roles::{WorkerKind, ROLE_SELECTOR};
use super::steps;
use super::stops;
use super::threads::{self, Effort, Entry, Note, Priority, State, Worker, BEAT_EVERY_MS};
use crate::access;
use crate::api::chat::{apply_tools_dial, dial_selection};
use crate::hand_records::{self, Carrying};
use crate::ingest_lease::{self, Guard, Share};
use crate::session::{NoDraft, Resume, StreamItem, ZendSession, CLOSED_BY_CLAUDE};
use crate::tools::Dispatch;
use crate::worker_hop;

/// How long a task's draft may take before its earlier entries are taken.
const DRAFT_DEADLINE: Duration = Duration::from_secs(15 * 60);
/// How often a draft's progress is read.
const DRAFT_POLL: Duration = Duration::from_secs(2);
/// The least time between two writes of a running worker's record.
const STEP_GAP: Duration = Duration::from_secs(5);
/// How often a wait says again what a worker still running has done, with
/// its time brought up to date, when it has taken no new step — inside the
/// stream's own repeat (`http::PROGRESS_EVERY`), so the line the person sees
/// is a worker's, with its time current.
const PROGRESS_AGAIN: Duration = Duration::from_secs(10);
/// How much of a prompt the thread's journal quotes.
const PROMPT_QUOTE: usize = 160;

/// The line a worker's question block opens with.
pub const QUESTION_MARK: &str = "QUESTION:";

/// Held while a worker's state is checked and its turn begun — naming a new
/// worker, answering one — so two calls at once (Claude issuing them in
/// parallel) cannot both take one name, or both answer one question, and run
/// two turns on one conversation.
static BEGIN: Mutex<()> = Mutex::new(());

/// Hold [`BEGIN`] — what archiving takes too, so no turn begins on a worker
/// between its being read and its record saying archived
/// (`super::archiving`). Held only across reads, claims and record writes,
/// never a distil or a turn; every record written under it is whole on its
/// own, so a lock poisoned by a panic is taken as it is rather than wedging
/// every later task.
pub fn holding() -> MutexGuard<'static, ()> {
    BEGIN.lock().unwrap_or_else(|e| e.into_inner())
}

/// What a task sent back by its answer's check is told first: who sent it
/// back. Unheaded, a worker took the check's words for the user's, and its
/// journal credited the user with asking for them
/// (`docs/zen_coding_challenge_fixes.md` Cause 7).
pub const SENT_BACK: &str = "From zend, which checked your answer — not the user:\n\n";

/// What every task's prompt ends with: how to ask Claude instead of guessing.
/// Asking ends the turn — the worker's last words are the question — so
/// nothing waits mid-turn, a restart finds a finished turn, and the answer is
/// simply the worker's next turn.
pub const ASKING: &str =
    "\n\n---\nIf you cannot go on without a decision that is not yours to make \
— the task is ambiguous, two approaches trade off and the choice is a design decision, the same \
failure keeps coming back — do not guess and do not stop. Ask: end your final response with this \
block and nothing after it, and the answer comes back as your next message.\n\
QUESTION: what you need decided or explained, in a sentence or two\n\
OPTIONS: each choice you see, and what it leads to\n\
TRIED: what you already did, and what happened\n\
POINTERS: the files and lines, commands or turns the question rests on\n\
Never ask what you can settle yourself by reading, running or searching.";

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub(super) fn loading() -> String {
    "the model is still loading; try again when `cluster_status` says it is ready".into()
}

/// The newest entry id `view` — a journal as `GET …/journal` answers it —
/// holds.
pub fn newest_entry(view: Option<&Value>) -> Option<u64> {
    view?
        .get("entries")?
        .as_array()?
        .iter()
        .filter_map(|e| e.get("id").and_then(Value::as_u64))
        .max()
}

/// The entries `view` holds after `after`, in order.
pub fn entries_after(view: Option<&Value>, after: Option<u64>) -> Vec<Entry> {
    let Some(entries) = view
        .and_then(|v| v.get("entries"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|e| {
            let id = e.get("id")?.as_u64()?;
            let text = e.get("text")?.as_str()?.to_string();
            after.is_none_or(|a| id > a).then_some(Entry { id, text })
        })
        .collect()
}

/// Whether `view` says a draft is running.
pub fn drafting(view: Option<&Value>) -> bool {
    view.and_then(|v| v.get("drafting"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// The reply of a final round, whole: what follows the `</think>` that closes
/// its reasoning — the first. A reply may quote `</think>` itself, as a review
/// of the code that closes a turn does, and the reply runs on past the quote.
pub fn final_reply(text: &str) -> String {
    text.split_once("</think>")
        .map_or(text, |(_, after)| after)
        .trim()
        .to_string()
}

/// The question block a reply ends with, from its [`QUESTION_MARK`] line on;
/// `None` when it asks nothing.
pub fn question_of(reply: &str) -> Option<String> {
    let at = reply
        .rmatch_indices(QUESTION_MARK)
        .map(|(i, _)| i)
        .find(|&i| i == 0 || reply[..i].ends_with('\n'))?;
    let block = reply[at..].trim();
    (block.len() > QUESTION_MARK.len()).then(|| block.to_string())
}

/// The first line of `text`, at most `max` characters.
pub fn first_line(text: &str, max: usize) -> String {
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    let mut out: String = line.chars().take(max).collect();
    if line.chars().count() > max {
        out.push('…');
    }
    out
}

/// An entry's summary: its `Entry N — …` line, which follows the date the
/// journal puts above an entry that starts a new hour; the first line when it
/// has none.
pub fn summary_line(entry: &str, max: usize) -> String {
    match entry.lines().find(|l| l.trim_start().starts_with("Entry ")) {
        Some(line) => first_line(line, max),
        None => first_line(entry, max),
    }
}

/// Start `prompt` on a new worker named after `wanted` under `thread`,
/// decoding at `priority`, thinking at `effort`, doing `kind`'s work and
/// driven as `driven` says; the worker's record, `running`. A worker a
/// conversation of Zen's starts — `from` names it — begins on that
/// conversation's branches and uncommitted changes (`conv_inherit`).
#[allow(clippy::too_many_arguments)]
pub fn start(
    session: &Arc<ZendSession>,
    caller: &Caller,
    thread: &str,
    wanted: &str,
    prompt: &str,
    (priority, effort, kind, driven): (Priority, Effort, WorkerKind, Driven),
    by_task: bool,
    from: Option<&str>,
) -> Result<Worker, String> {
    let records = session.shared_substrate().ok_or_else(loading)?;
    let owner = caller.key();
    names::check(thread)?;
    if prompt.trim().is_empty() {
        return Err("the task's `prompt` is empty: state the task in full".into());
    }
    let now = now_ms();
    // Attached under the lock archiving takes, so a sweep never archives the
    // thread between the attach and the worker it begins.
    let _held = holding();
    threads::attach(&records, &owner, thread, now).map_err(|e| e.to_string())?;
    let taken = taken(&records, &owner, thread, now);
    let name = names::worker_name(thread, wanted, &taken)?;
    let mut worker = fresh(
        session,
        &owner,
        thread,
        name,
        prompt,
        (
            Some(caller.person.clone()),
            caller.role,
            priority,
            effort,
            kind,
        ),
        now,
    );
    worker.by_task = by_task;
    worker.driven = driven;
    if let Some(Err(e)) = from.and_then(|starter| session.begin_on(&worker.conv_id, starter)) {
        return Err(format!(
            "the task could not be given its starter's work: {e}"
        ));
    }
    // Its answer ends in a `Steps:` section of them, written by the grammar.
    let given = steps::of(prompt);
    if given.len() >= 2 {
        if let Some(Err(e)) = session.answer_steps(&worker.conv_id, &given) {
            return Err(format!("the task's steps could not be recorded: {e}"));
        }
    }
    let started = if by_task {
        "A task started:"
    } else {
        "Claude started:"
    };
    note(
        &records,
        &owner,
        &worker,
        started,
        first_line(prompt, PROMPT_QUOTE),
    )?;
    begin(
        session,
        &records,
        &owner,
        worker,
        format!("{prompt}{ASKING}"),
    )
}

/// Every worker under `thread`, by name, and whether it is live.
///
/// A name is taken by its record's key, read or not: a record that no longer
/// decodes still names a conversation that exists, so its name is never given
/// out again — it counts as ended, and a new task asking for it gets the next
/// numbered sibling.
fn taken(records: &dyn Records, owner: &str, thread: &str, now: u64) -> Vec<(String, bool)> {
    let prefix = threads::worker_prefix(owner, thread);
    records
        .under(&prefix)
        .into_iter()
        .filter_map(|(key, value)| {
            let name = key.strip_prefix(&prefix)?.to_string();
            let live =
                serde_json::from_value::<Worker>(value).is_ok_and(|w| w.observed(now).state.live());
            Some((name, live))
        })
        .collect()
}

/// A new worker's record, `running`, its turn beginning `now`, working for
/// `person` as `role` at `priority`, thinking at `effort`, doing `kind`'s
/// work.
pub(super) fn fresh(
    session: &ZendSession,
    owner: &str,
    thread: &str,
    name: String,
    prompt: &str,
    (person, role, priority, effort, kind): (
        Option<Person>,
        AccessRole,
        Priority,
        Effort,
        WorkerKind,
    ),
    now: u64,
) -> Worker {
    let conv_id = threads::conv_id(owner, thread, &name);
    Worker {
        thread: thread.to_string(),
        entries_after: newest_entry(session.journal(&conv_id).as_ref()),
        name,
        conv_id,
        prompt: prompt.to_string(),
        state: State::Running,
        role,
        person,
        priority,
        effort,
        kind,
        driven: Driven::default(),
        started_ms: now,
        turn_ms: now,
        beat_ms: now,
        ended_ms: None,
        entries: Vec::new(),
        final_text: String::new(),
        error: None,
        question: None,
        activity: None,
        no_entry: None,
        stopped: false,
        resumable: false,
        resumes: 0,
        audit: None,
        checked: false,
        // `begin` names the member it begins on.
        runner: None,
        by_task: false,
    }
}

/// Answer the worker `name` of `thread`, which asked, is stuck, or finished
/// and is kept to be continued: the answer is its next turn, in the same
/// conversation. The worker's record, `running` again.
pub fn answer(
    session: &Arc<ZendSession>,
    caller: &Caller,
    thread: &str,
    name: &str,
    text: &str,
) -> Result<Worker, String> {
    let records = session.shared_substrate().ok_or_else(loading)?;
    let owner = caller.key();
    if text.trim().is_empty() {
        return Err("the `answer` is empty".into());
    }
    let _held = holding();
    let Some(found) = threads::worker(&records, &owner, thread, name) else {
        return Err(format!("there is no task `task:{thread}/{name}`"));
    };
    let mut worker = found.observed(now_ms());
    if archiving::being_archived(&worker.conv_id) {
        return Err(format!(
            "`{}` is being archived, so it cannot be continued: start a new task, citing its \
             entries",
            worker.path()
        ));
    }
    let label = match worker.state {
        State::Question | State::Stuck => "Claude answered:",
        State::Done if worker.resumable => "Claude continued:",
        State::Running => {
            return Err(format!(
                "`{}` is still working: nudge it with `repo_task_nudge`, or wait",
                worker.path()
            ))
        }
        State::Done => {
            return Err(format!(
                "`{}` finished and was archived, so it cannot be continued: start a new task, \
                 citing its entries",
                worker.path()
            ))
        }
        _ => {
            return Err(format!(
                "`{}` ended and cannot be continued: start a new task",
                worker.path()
            ))
        }
    };
    let now = now_ms();
    // Continuing a worker is work on its thread: it reopens a thread archived
    // while the worker was left whole.
    threads::attach(&records, &owner, thread, now).map_err(|e| e.to_string())?;
    ready_for_turn(session, &mut worker, caller, now);
    note(
        &records,
        &owner,
        &worker,
        label,
        first_line(text, PROMPT_QUOTE),
    )?;
    begin(session, &records, &owner, worker, format!("{text}{ASKING}"))
}

/// `worker`, which settled, made ready for its next turn, run for `caller` —
/// as their role now — from `now`: running again, its last ending's marks
/// cleared, and its entries counted from its journal's newest.
pub(super) fn ready_for_turn(
    session: &ZendSession,
    worker: &mut Worker,
    caller: &Caller,
    now: u64,
) {
    worker.state = State::Running;
    worker.role = caller.role;
    worker.person = Some(caller.person.clone());
    worker.beat_ms = now;
    worker.turn_ms = now;
    worker.ended_ms = None;
    worker.resumable = false;
    worker.audit = None;
    worker.question = None;
    worker.entries_after = newest_entry(session.journal(&worker.conv_id).as_ref());
    worker.entries = Vec::new();
}

/// What a task cut off by a restart is told first.
const RESUMED: &str = "Your work was cut off, so your last turn may not have finished. Your \
    conversation and your journal are intact; changes you had not committed may be gone. Check \
    where the work stands (`git_status`, reading what you wrote) before repeating anything, then \
    finish the task below. It is your task, whatever else you remember.";

/// What Zen's side of a chat, cut off, is told first: a chat worker's prompt is
/// the message it was answering, not a task.
const CHAT_RESUMED: &str = "Your answer was cut off by a restart, so it may not have finished. \
    Answer the message below again, whole, as your next message in the chat.";

/// What a turn's message ends with: for a task, how to ask Claude; for a chat,
/// nothing — Zen answers Claude in the chat itself.
fn closing_of(kind: WorkerKind) -> &'static str {
    if kind.is_chat() {
        ""
    } else {
        ASKING
    }
}

/// How many of a cut-off task's journal entries it is reminded of.
const RESUMED_ENTRIES: usize = 8;
/// How much of each it is reminded of.
const RESUMED_ENTRY_QUOTE: usize = 300;

/// What `worker`, cut off, is told: [`RESUMED`], its task in full, and what its
/// journal says it did. A turn's context is selected by relevance, so nothing
/// else keeps the task in front of it: told only to finish "the task you were
/// given", a worker 88 turns in took up something else it could see, reported
/// that done, and the task it had was never finished (2026-10-31).
fn resumed_message(worker: &Worker) -> String {
    if worker.kind.is_chat() {
        return format!(
            "{CHAT_RESUMED}\n\nThe message you were answering:\n{}",
            worker.prompt
        );
    }
    let mut message = format!("{RESUMED}\n\nYour task:\n{}", worker.prompt);
    let skipped = worker.entries.len().saturating_sub(RESUMED_ENTRIES);
    let kept = &worker.entries[skipped..];
    if !kept.is_empty() {
        message.push_str("\n\nWhat your journal says you did so far:");
        for entry in kept {
            message.push_str("\n- ");
            message.push_str(&first_line(&entry.text, RESUMED_ENTRY_QUOTE));
        }
    }
    message
}

/// Make `worker`, cut off, run again from `now`, the try entered on its thread.
/// What it should be told is read first, off the entries this clears.
fn revive(
    session: &ZendSession,
    records: &dyn Records,
    owner: &str,
    worker: &mut Worker,
    now: u64,
) -> Result<String, String> {
    let told = resumed_message(worker);
    worker.state = State::Running;
    worker.error = None;
    worker.ended_ms = None;
    worker.resumes += 1;
    worker.beat_ms = now;
    worker.turn_ms = now;
    worker.entries_after = newest_entry(session.journal(&worker.conv_id).as_ref());
    worker.entries = Vec::new();
    note(
        records,
        owner,
        worker,
        "Resumed:",
        format!("cut off, try {}", worker.resumes),
    )?;
    Ok(told)
}

/// Start the lost task `name` of `thread` again, in its own conversation
/// (`super::resuming`), under `lease`. `false` when it no longer needs it: read
/// again under the lock a reply takes, it is running, or was answered, or
/// archived.
///
/// A round it stopped in the middle of is finished first, with what it is told
/// following that round's results: a new message would leave those calls
/// unanswered in the conversation.
pub fn resume(
    session: &Arc<ZendSession>,
    owner: &str,
    thread: &str,
    name: &str,
    now: u64,
    lease: Guard,
) -> Result<bool, String> {
    let records = session.shared_substrate().ok_or_else(loading)?;
    let _held = holding();
    // Read under the lock a disposal marks a chat closing under, so a chat
    // being disposed of is never started again.
    if !resuming::chat_open(&records, owner, thread) {
        return Ok(false);
    }
    let Some(found) = threads::worker(&records, owner, thread, name) else {
        return Ok(false);
    };
    let mut worker = found.observed(now);
    if !resuming::due(&worker, now)
        || archiving::being_archived(&worker.conv_id)
        || session.live_turn(&worker.conv_id).is_some()
    {
        return Ok(false);
    }
    let told = revive(session, &records, owner, &mut worker, now)?;
    let first = match session.pending_round(&worker.conv_id) {
        Some(call_turn) => {
            nudges::queue(&records, &worker.conv_id, Sender::Zend, &told, now)
                .map_err(|e| e.to_string())?;
            First::Interrupted { call_turn }
        }
        None => First::Message(format!("{told}{}", closing_of(worker.kind))),
    };
    begin_held(session, &records, owner, worker, first, Some(lease))?;
    Ok(true)
}

/// Finish, as `name` of `thread`'s own, the round at `call_turn` this member's
/// last run stopped in the middle of: the worker is carried on from there, with
/// its beat, its settling and its lease, so no other member takes it for lost
/// and starts it over the same round. `false` when the worker is not one to
/// carry on — it ended, or it was cut off too often — and the
/// round is simply finished. `true` too when another member holds its turns.
pub async fn carry_on(
    session: &Arc<ZendSession>,
    owner: &str,
    thread: &str,
    name: &str,
    call_turn: u32,
) -> Result<bool, String> {
    let records = session.shared_substrate().ok_or_else(loading)?;
    let cut_off = |w: &Worker| {
        let seen = w.clone().observed(now_ms());
        seen.resumes < resuming::MAX_RESUMES
            && (seen.state == State::Running || archiving::lost(&seen))
    };
    let Some(found) = threads::worker(&records, owner, thread, name).filter(cut_off) else {
        return Ok(false);
    };
    let conv_id = found.conv_id;
    let share = tokio::task::spawn_blocking(move || ingest_lease::share_worker(&conv_id))
        .await
        .map_err(|e| e.to_string())?;
    let Share::Mine(lease) = share else {
        return Ok(true);
    };
    let _held = holding();
    let Some(mut worker) = threads::worker(&records, owner, thread, name).filter(cut_off) else {
        return Ok(false);
    };
    let now = now_ms();
    let told = revive(session, &records, owner, &mut worker, now)?;
    nudges::queue(&records, &worker.conv_id, Sender::Zend, &told, now)
        .map_err(|e| e.to_string())?;
    begin_held(
        session,
        &records,
        owner,
        worker,
        First::Interrupted { call_turn },
        Some(lease),
    )?;
    Ok(true)
}

/// Queue `text` for the running worker `name` of `thread`, taken at its next
/// tool round (`super::nudges`).
pub fn nudge(
    session: &ZendSession,
    caller: &Caller,
    thread: &str,
    name: &str,
    text: &str,
) -> Result<Worker, String> {
    let records = session.shared_substrate().ok_or_else(loading)?;
    let owner = caller.key();
    if text.trim().is_empty() {
        return Err("the `message` is empty".into());
    }
    let Some(found) = threads::worker(&records, &owner, thread, name) else {
        return Err(format!("there is no task `task:{thread}/{name}`"));
    };
    let worker = found.observed(now_ms());
    if !worker.state.working() {
        return Err(format!(
            "`{}` is not working, so it would never read a nudge; {}",
            worker.path(),
            if worker.state.live() {
                "it waits on an answer — use `repo_task_reply`"
            } else {
                "it has finished — start a new task"
            }
        ));
    }
    nudges::queue(&records, &worker.conv_id, Sender::Starter, text, now_ms())
        .map_err(|e| e.to_string())?;
    note(
        &records,
        &owner,
        &worker,
        "Claude nudged:",
        first_line(text, PROMPT_QUOTE),
    )?;
    Ok(worker)
}

/// Stop the running worker `name` of `thread` at its next tool round, for
/// `reason` (`super::stops`).
pub fn stop(
    session: &ZendSession,
    caller: &Caller,
    thread: &str,
    name: &str,
    reason: &str,
) -> Result<Worker, String> {
    let records = session.shared_substrate().ok_or_else(loading)?;
    let owner = caller.key();
    // Ordered against a turn beginning, which clears a stop meant for the
    // turn before it.
    let _held = holding();
    let Some(found) = threads::worker(&records, &owner, thread, name) else {
        return Err(format!("there is no task `task:{thread}/{name}`"));
    };
    let worker = found.observed(now_ms());
    if !worker.state.working() {
        return Err(format!(
            "`{}` is not running, so there is nothing to stop; {}",
            worker.path(),
            if worker.state.live() {
                "it waits on an answer — redirect it with `repo_task_reply`"
            } else {
                "it has finished"
            }
        ));
    }
    stops::request(&records, &worker.conv_id, reason).map_err(|e| e.to_string())?;
    let line = if reason.trim().is_empty() {
        "at its next tool round".to_string()
    } else {
        format!(
            "at its next tool round: {}",
            first_line(reason, PROMPT_QUOTE)
        )
    };
    note(&records, &owner, &worker, "Claude stopped:", line)?;
    Ok(worker)
}

/// Enter a line about `worker` on its thread's journal — none for Zen's side
/// of a chat, whose thread keeps no copy of what was said
/// (`docs/zen_chat.md` §2).
fn note(
    records: &dyn Records,
    owner: &str,
    worker: &Worker,
    label: &str,
    text: String,
) -> Result<(), String> {
    if is_chat_thread(&worker.thread) {
        return Ok(());
    }
    threads::add_note(
        records,
        owner,
        &worker.thread,
        &Note {
            at_ms: now_ms(),
            label: label.into(),
            text: format!("{} — {text}", worker.path()),
            worker: Some(worker.path()),
            client: None,
        },
    )
    .map_err(|e| e.to_string())
}

/// Keep `worker` as running and start its turn on `message`.
pub(super) fn begin(
    session: &Arc<ZendSession>,
    records: &dyn Records,
    owner: &str,
    worker: Worker,
    message: String,
) -> Result<Worker, String> {
    begin_held(
        session,
        records,
        owner,
        worker,
        First::Message(message),
        None,
    )
}

/// Keep `worker` as running and start its turn from `first`, under `lease` when
/// the caller already took it.
fn begin_held(
    session: &Arc<ZendSession>,
    records: &dyn Records,
    owner: &str,
    worker: Worker,
    first: First,
    lease: Option<Guard>,
) -> Result<Worker, String> {
    // A stop sent as the last turn ended — after its settling looked — is
    // not this turn's.
    stops::take(records, &worker.conv_id);
    // Its last turns may have run on another member, after what this one
    // keeps of the conversation in memory: the turn here is built afresh.
    let mut worker = worker;
    let here = worker_hop::own_node();
    if worker.runner != here {
        session.forget(&worker.conv_id);
    }
    worker.runner = here;
    threads::put_worker(records, owner, &worker).map_err(|e| e.to_string())?;
    // Recorded before the turn makes the conversation's slot, which reads it.
    if let Some(Err(e)) = session.set_decode_band(&worker.conv_id, worker.priority.band()) {
        tracing::warn!(conv_id = %worker.conv_id, "mcp: the task's priority could not be kept: {e}");
    }
    session.mcp().bump();
    tokio::spawn(run(
        Arc::clone(session),
        owner.to_string(),
        worker.clone(),
        first,
        lease,
    ));
    Ok(worker)
}

/// What a worker's next turn starts from.
enum First {
    /// A message: the task, an answer, a correction.
    Message(String),
    /// A round whose call turn `call_turn`, round `iteration` of its turn, is
    /// sealed and whose tools have not run — handed here by a slower member,
    /// or kept when a handing failed.
    Round { call_turn: u32, iteration: usize },
    /// A round whose call turn `call_turn` is sealed and which a restart cut
    /// off: some of its tools may have run.
    Interrupted { call_turn: u32 },
}

impl First {
    /// What the turn is sent, and the round it finishes.
    fn into_start(self) -> (String, Option<Resume>) {
        match self {
            First::Message(message) => (message, None),
            First::Round {
                call_turn,
                iteration,
            } => (
                String::new(),
                Some(Resume {
                    call_turn,
                    dispatch: Dispatch::Live,
                    iteration,
                }),
            ),
            First::Interrupted { call_turn } => (
                String::new(),
                Some(Resume {
                    call_turn,
                    dispatch: Dispatch::Resumed,
                    iteration: 0,
                }),
            ),
        }
    }
}

/// How long a member starting a worker's turns waits for their lease: long
/// enough for one a restart left behind to run out ([`ingest_lease`]'s
/// `INGEST_TTL`).
const LEASE_WAIT: Duration = Duration::from_secs(150);
/// How often it asks again.
const LEASE_LOOK: Duration = Duration::from_secs(2);

/// Take the lease on `conv_id`'s turns, waiting up to [`LEASE_WAIT`] for
/// whoever holds it.
async fn hold_turns(conv_id: &str) -> Option<Guard> {
    let until = Instant::now() + LEASE_WAIT;
    loop {
        let conv_id = conv_id.to_string();
        let share = tokio::task::spawn_blocking(move || ingest_lease::share_worker(&conv_id))
            .await
            .ok()?;
        if let Share::Mine(lease) = share {
            return Some(lease);
        }
        if Instant::now() >= until {
            return None;
        }
        tokio::time::sleep(LEASE_LOOK).await;
    }
}

/// End `worker`, whose turns this member could not get the lease on.
fn give_up(session: &Arc<ZendSession>, records: &dyn Records, owner: &str, worker: &mut Worker) {
    let why = "another member holds this task's turns, so it was not started here";
    worker.state = State::Error;
    worker.error = Some(why.into());
    worker.ended_ms = Some(now_ms());
    // A chat hears of it as of any answer that failed, and one closed while
    // it waited is disposed of now.
    let chat = worker.kind.is_chat();
    if chat {
        talk::answered(records, owner, worker);
    }
    if let Err(e) = threads::put_worker(records, owner, worker) {
        tracing::warn!(conv_id = %worker.conv_id, "mcp: the worker's outcome could not be kept: {e}");
    }
    if chat {
        talk::settled(session, owner, worker);
    }
    if let Err(e) = note(
        records,
        owner,
        worker,
        "Task ended:",
        format!("error: {why}"),
    ) {
        tracing::warn!(conv_id = %worker.conv_id, "mcp: the outcome could not be entered on the thread: {e}");
    }
}

/// How long a handed turn's commit may take to land before the member keeps
/// the round itself.
const HAND_COMMIT_WAIT: Duration = Duration::from_secs(20);
/// How long the member a turn is handed to may take to answer each of the
/// handing's two asks — it waits to hold the turn's records first
/// ([`CONTINUE_HOLD_WAIT`]), and the second ask carries the records it lacked.
/// Both, with the commit's wait, [`TAKEN_WAIT`] and a beat's interval, are
/// under the beat's own limit (`threads::BEAT_LOST_MS`): the beat is stopped
/// while the handing runs.
const HAND_ANSWER_WAIT: Duration = Duration::from_secs(25);
/// How long the member taking a handed worker waits to hold its records.
pub const CONTINUE_HOLD_WAIT: Duration = Duration::from_secs(10);
/// The error a member taking a worker answers when it lacks the handing
/// member's records, with the ranges it lacks as `missing`.
pub const MISSING: &str = "missing";
/// The route a worker is handed on (`crate::api::worker_continue`).
pub const CONTINUE_PATH: &str = "/v1/mcp/worker/continue";

/// What a member hands a worker on with (`crate::api::worker_continue`).
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Handing {
    /// The member handing it on, and the member it is handed to: signed, so
    /// the proof names both and stands alone.
    pub from: u64,
    pub to: u64,
    pub owner: String,
    pub thread: String,
    pub name: String,
    pub call_turn: u32,
    /// The round of its turn the call turn was.
    pub iteration: usize,
    /// The handing member's newest commit of the worker's conversation.
    pub through: u64,
    /// Every record the handing member wrote up to this `oseq` is in its log:
    /// the commit, and the round's resume entry and the worker's record after
    /// it. A member lacking any of them says which (`crate::hand_records`).
    pub written: u64,
    /// When it was signed, Unix ms (`worker_hop::fresh`).
    pub signed_ms: u64,
}

/// The ranges a refusal says its member lacks of the handing member's
/// records, when it is one that says so ([`MISSING`]).
pub fn lacked(status: u16, body: &[u8]) -> Option<Vec<(u64, u64)>> {
    if status != 409 {
        return None;
    }
    let v: Value = serde_json::from_slice(body).ok()?;
    if v.get("error").and_then(Value::as_str) != Some(MISSING) {
        return None;
    }
    serde_json::from_value(v.get("missing")?.clone()).ok()
}

/// Hand `worker`, whose turn stopped before running round `iteration`'s call
/// turn `call_turn`, to member `to` (`crate::worker_hop`): once this member's
/// commit of it has landed and its writes are in the log, `to` is asked to
/// take it, signed with the members' handing key, and answers once it holds
/// the commit and runs the worker. When `to` answers that it lacks records of
/// this member's, it is asked again carrying them, if they are few enough
/// (`crate::hand_records`). `Err` keeps the worker here.
async fn hand_on(
    session: &ZendSession,
    owner: &str,
    worker: &Worker,
    to: u64,
    call_turn: u32,
    iteration: usize,
) -> Result<(), String> {
    let member = worker_hop::member().ok_or("this member is in no cluster")?;
    let through = session
        .committed_through(&worker.conv_id, HAND_COMMIT_WAIT)
        .await
        .ok_or("its turn's commit did not land in time")?;
    let written = session
        .written_through()
        .await
        .ok_or("its writes could not be waited for")?;
    let from = worker_hop::own_node().ok_or("this member is in no cluster")?;
    let mut carrying = Carrying::default();
    loop {
        let handing = serde_json::to_vec(&Handing {
            from,
            to,
            owner: owner.to_string(),
            thread: worker.thread.clone(),
            name: worker.name.clone(),
            call_turn,
            iteration,
            through,
            written,
            signed_ms: now_ms(),
        })
        .map_err(|e| e.to_string())?;
        let body = hand_records::encode(&handing, &carrying.records);
        let proof = worker_hop::sign(&body).ok_or("this member holds no handing key")?;
        let headers = [
            (
                HeaderName::from_static(NODE_HEADER),
                HeaderValue::from_str(&format!("{to:016x}")).map_err(|e| e.to_string())?,
            ),
            (
                HeaderName::from_static(worker_hop::PROOF_HEADER),
                HeaderValue::from_str(&proof).map_err(|e| e.to_string())?,
            ),
        ];
        let reply = send_up(
            &member,
            Method::POST,
            CONTINUE_PATH,
            &headers,
            body,
            Some(HAND_ANSWER_WAIT),
        )
        .await
        .map_err(|e| format!("the gateway could not be reached: {e:#}"))?;
        if reply.status == 202 {
            return Ok(());
        }
        let refused = format!(
            "{to:016x} answered {}: {}",
            reply.status,
            String::from_utf8_lossy(&reply.body)
        );
        // Asked once carrying what it lacked, it is not asked again.
        let missing = match lacked(reply.status, &reply.body) {
            Some(missing) if carrying.records.is_empty() => missing,
            _ => return Err(refused),
        };
        carrying = session
            .carrying(&worker.conv_id, missing)
            .await
            .map_err(|why| format!("{refused} — and none are carried: {why}"))?;
        if carrying.records.is_empty() {
            return Err(format!("{refused} — and this member holds none of them"));
        }
        tracing::info!(
            conv_id = %worker.conv_id,
            to = %format!("{to:016x}"),
            "mcp: the member taking the worker lacks records of this member's, so the handing \
             carries them: {}",
            carrying.told()
        );
    }
}

/// Whether `worker`'s record says member `to` runs it now: a handing whose
/// answer was lost took effect all the same.
fn taken_by(records: &dyn Records, owner: &str, worker: &Worker, to: u64) -> bool {
    threads::worker(records, owner, &worker.thread, &worker.name)
        .is_some_and(|w| w.state == State::Running && w.runner == Some(to))
}

/// How long a member whose handing failed waits for the record to say the
/// peer took it after all — its write may still be on its way here.
const TAKEN_WAIT: Duration = Duration::from_secs(10);

/// [`taken_by`], looked at until it holds or [`TAKEN_WAIT`] passes.
async fn taken_within(records: &dyn Records, owner: &str, worker: &Worker, to: u64) -> bool {
    let until = Instant::now() + TAKEN_WAIT;
    loop {
        if taken_by(records, owner, worker, to) {
            return true;
        }
        if Instant::now() >= until {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// Say on `worker`'s record that it still runs, now.
fn beat_now(records: &dyn Records, owner: &str, worker: &Worker) {
    if let Some(mut w) = threads::worker(records, owner, &worker.thread, &worker.name) {
        w.beat_ms = now_ms();
        if let Err(e) = threads::put_worker(records, owner, &w) {
            tracing::warn!(conv_id = %w.conv_id, "mcp: the task's beat could not be kept: {e}");
        }
    }
}

/// Take `handing`'s worker, which member `by` handed here at a round
/// boundary: run its round's tools and carry the worker on — its record, its
/// beat, its settling — as if it had started here. Taken only while it runs on
/// `by`, its round still to run, so a handing sent twice, or by anyone but its
/// runner, runs nothing.
pub fn continue_handed(
    session: &Arc<ZendSession>,
    handing: &Handing,
    by: u64,
) -> Result<Worker, String> {
    if handing.from != by || Some(handing.to) != worker_hop::own_node() {
        return Err(format!(
            "the handing is from {:016x} to {:016x}, not from {by:016x} to this member",
            handing.from, handing.to
        ));
    }
    let records = session.shared_substrate().ok_or_else(loading)?;
    let _held = holding();
    let Some(mut worker) =
        threads::worker(&records, &handing.owner, &handing.thread, &handing.name)
    else {
        return Err(format!(
            "there is no worker `T:{}/{}`",
            handing.thread, handing.name
        ));
    };
    if worker.state != State::Running || worker.runner != Some(by) {
        return Err(format!(
            "`{}` is not running on {by:016x}, so it is not {by:016x}'s to hand on",
            worker.path()
        ));
    }
    if !session.round_pending(&worker.conv_id, handing.call_turn) {
        return Err(format!(
            "`{}` has no round at call turn {} still to run",
            worker.path(),
            handing.call_turn
        ));
    }
    // Its turns were written on `by` after anything this member keeps of the
    // conversation in memory.
    session.forget(&worker.conv_id);
    worker.runner = worker_hop::own_node();
    worker.beat_ms = now_ms();
    threads::put_worker(&records, &handing.owner, &worker).map_err(|e| e.to_string())?;
    if let Some(Err(e)) = session.set_decode_band(&worker.conv_id, worker.priority.band()) {
        tracing::warn!(conv_id = %worker.conv_id, "mcp: the task's priority could not be kept: {e}");
    }
    session.mcp().bump();
    tokio::spawn(run(
        Arc::clone(session),
        handing.owner.clone(),
        worker.clone(),
        First::Round {
            call_turn: handing.call_turn,
            iteration: handing.iteration,
        },
        None,
    ));
    Ok(worker)
}

/// How a turn ended, before the worker settles.
enum Ended {
    /// It ended on its own: its whole text.
    Replied(String),
    /// Its tools were closed — by the repeat guard, the round cap, or Claude
    /// stopping it (`by`, a [`StreamItem::Closing`] reason): its whole text.
    Closed {
        text: String,
        by: &'static str,
    },
    Failed(String),
}

/// What a turn came to: an ending the worker settles on, or a move.
enum Turned {
    Ended(Ended),
    /// It stopped at round `iteration` to move to member `to`, its call turn
    /// `call_turn` sealed and its tools not run ([`StreamItem::Handed`]).
    Handed {
        to: u64,
        call_turn: u32,
        iteration: usize,
    },
}

impl Ended {
    /// Whether Claude stopped the turn (`job_cancel`).
    fn stopped(&self) -> bool {
        matches!(self, Ended::Closed { by, .. } if *by == CLOSED_BY_CLAUDE)
    }
}

/// The turn's life after [`begin`], and the worker's settling, under the lease
/// on its turns from start to end: `lease`, or one taken here.
async fn run(
    session: Arc<ZendSession>,
    owner: String,
    mut worker: Worker,
    first: First,
    lease: Option<Guard>,
) {
    let Some(records) = session.shared_substrate() else {
        return;
    };
    // The record says the task still runs for as long as it does — through
    // the turn and the draft after it — so no reader takes it for cut off,
    // and carries what the turn has done so far.
    let live = Live::default();
    let (thread, name) = (worker.thread.clone(), worker.name.clone());
    let start_beat = || {
        tokio::spawn(beat(
            Arc::clone(&session),
            records.clone(),
            owner.clone(),
            thread.clone(),
            name.clone(),
            live.clone(),
        ))
    };
    let mut beating = start_beat();
    let mut lease = match lease {
        Some(lease) => Some(lease),
        None => hold_turns(&worker.conv_id).await,
    };
    if lease.is_none() {
        beating.abort();
        let _ = beating.await;
        give_up(&session, &records, &owner, &mut worker);
        session.mcp().bump();
        return;
    }
    let mut next = first;
    // Whether a turn here may move to a markedly faster member: not again,
    // once a handing failed, so a peer that cannot take it is not asked at
    // every round.
    let mut may_hand = true;
    // The answer, audit and settling of a task sent back to open what it
    // cited: what it ends with when the check round does not finish.
    let mut first: Option<(String, Audit, (&'static str, String))> = None;
    let (label, outcome) = loop {
        // A check round does not move: the answer it falls back on is kept
        // here, in `first`.
        let movable = may_hand
            && first.is_none()
            && worker.runner.is_some()
            && worker.runner == worker_hop::own_node();
        let ended = match turn(&session, &worker, next, movable, &live).await {
            Turned::Ended(ended) => ended,
            Turned::Handed {
                to,
                call_turn,
                iteration,
            } => {
                // Stopped first, so the record the member taking it writes is
                // the last word on who runs it.
                beating.abort();
                let _ = beating.await;
                // Handed back with the turn's writes, for the member taking it.
                drop(lease.take());
                let handed = hand_on(&session, &owner, &worker, to, call_turn, iteration).await;
                let moved = match handed {
                    Ok(()) => true,
                    Err(why) if taken_within(&records, &owner, &worker, to).await => {
                        tracing::warn!(
                            conv_id = %worker.conv_id,
                            to = %format!("{to:016x}"),
                            "mcp: the handing's answer was lost, but the worker moved: {why}"
                        );
                        true
                    }
                    Err(why) => {
                        tracing::warn!(
                            conv_id = %worker.conv_id,
                            to = %format!("{to:016x}"),
                            "mcp: the worker could not move, so it goes on here: {why}"
                        );
                        false
                    }
                };
                // The member it moved to beats, settles and answers for it
                // now; what Claude sent it, a nudge or a stop, is read there.
                if moved {
                    session.forget(&worker.conv_id);
                    tracing::info!(
                        conv_id = %worker.conv_id,
                        to = %format!("{to:016x}"),
                        call_turn,
                        "mcp: the worker moved to a faster member"
                    );
                    return;
                }
                may_hand = false;
                // Said at once, so the stretch without a beat stays short.
                beat_now(&records, &owner, &worker);
                beating = start_beat();
                lease = hold_turns(&worker.conv_id).await;
                if lease.is_none() {
                    beating.abort();
                    let _ = beating.await;
                    give_up(&session, &records, &owner, &mut worker);
                    session.mcp().bump();
                    return;
                }
                next = First::Round {
                    call_turn,
                    iteration,
                };
                continue;
            }
        };
        live.set_doing(Doing::Journaling);
        // A nudge that came after the turn's last tool round was never read.
        for (_, undelivered) in nudges::take(&records, &worker.conv_id) {
            let line = format!(
                "not delivered — the turn ended without another tool round: {}",
                first_line(&undelivered, PROMPT_QUOTE)
            );
            if let Err(e) = note(&records, &owner, &worker, "Nudge dropped:", line) {
                tracing::warn!(conv_id = %worker.conv_id, "mcp: an undelivered nudge could not be entered: {e}");
            }
        }
        worker.no_entry = None;
        // A chat's answer is ready when its turn ends: no draft stands between
        // Claude and the reply (`docs/zen_chat.md` §2).
        let drafted = match ended {
            Ended::Failed(_) => None,
            _ if worker.kind.is_chat() => None,
            _ => Some(finish_journal(&session, &worker.conv_id).await),
        };
        // A stop that came after the turn's last tool round — while it
        // answered or drafted — found it finishing.
        if stops::take(&records, &worker.conv_id).is_some() {
            let line = "not reached — the turn ended without another tool round".to_string();
            if let Err(e) = note(&records, &owner, &worker, "Stop dropped:", line) {
                tracing::warn!(conv_id = %worker.conv_id, "mcp: an unreached stop could not be entered: {e}");
            }
        }
        // A failed turn still carries what its cadence drafted while it ran.
        let view = session.journal(&worker.conv_id);
        worker.entries = entries_after(view.as_ref(), worker.entries_after);
        if let Some(drafted) = drafted.filter(|_| worker.entries.is_empty()) {
            worker.no_entry = Some(why_no_entry(view.as_ref(), drafted));
        }
        if worker.kind.is_chat() {
            break settle_chat(&mut worker, ended);
        }
        let settled = settle(&mut worker, ended);
        if worker.state != State::Done {
            // A check round that failed, was stopped or asked leaves the task
            // done, with the answer it had: the round said nothing better.
            let Some((answer, audit, (label, outcome))) = first.take() else {
                break settled;
            };
            let round = keep_first_answer(&mut worker, answer, &audit);
            break (label, format!("{outcome}; its answer's check {round}"));
        }
        let audit = match audit_of(&session, &worker).await {
            Ok(audit) => audit,
            // Said, so an answer is never read as citing nothing because its
            // audit could not run.
            Err(e) => {
                worker.audit = Some(format!(
                    "Citation audit: could not be run ({e}); its citations are unchecked."
                ));
                break settled;
            }
        };
        // A task's first answer citing what it never opened, or saying nothing
        // of steps its brief gave it (`super::steps`), goes back once to put
        // that right; the second answer is kept with its audit, whatever it
        // says. A continuation is Claude's to direct, with the audit in hand,
        // so it is not sent back.
        let given = steps::of(&worker.prompt);
        let unsaid = steps::unaccounted(&given, &worker.final_text);
        if (audit.unopened() || !unsaid.is_empty()) && first.is_none() && !worker.checked {
            let mut told = Vec::new();
            let mut why = Vec::new();
            if audit.unopened() {
                let to_correct = audit.miscopied.len() + audit.seen.len() + audit.never_seen.len();
                told.push(audit.correction());
                why.push(format!(
                    "{to_correct} of its answer's citations are not by an address it opened"
                ));
            }
            if !unsaid.is_empty() {
                told.push(steps::correction(&given, &unsaid));
                why.push(format!(
                    "its answer says nothing of {} of the {} steps it was given",
                    unsaid.len(),
                    given.len()
                ));
            }
            let why = why.join(", and ");
            let line = format!("{why}; it is putting that right");
            if let Err(e) = note(&records, &owner, &worker, "Task sent back:", line) {
                tracing::warn!(conv_id = %worker.conv_id, "mcp: a send-back could not be entered: {e}");
            }
            next = First::Message(format!("{SENT_BACK}{}{ASKING}", told.join("\n\n")));
            first = Some((worker.final_text.clone(), audit, settled));
            worker.state = State::Running;
            worker.checked = true;
            // Said on its record while the check runs, so a wait shows why a
            // worker that answered is running again. Only these fields: the
            // beat owns the rest of the record while the turn runs.
            let checking = format!("Checking its answer: sent back because {why}.");
            if let Some(mut stored) =
                threads::worker(&records, &owner, &worker.thread, &worker.name)
            {
                stored.audit = Some(checking);
                stored.checked = true;
                if let Err(e) = threads::put_worker(&records, &owner, &stored) {
                    tracing::warn!(conv_id = %worker.conv_id, "mcp: a send-back could not be kept: {e}");
                }
            }
            continue;
        }
        worker.audit = audit.cites().then(|| audit.render());
        break settled;
    };
    // The beat stops, and is done writing, before the settled record is.
    beating.abort();
    let _ = beating.await;
    // Another member took the worker while this one finished a round it
    // thought its own: the record and the settling are that member's.
    let runner_now =
        threads::worker(&records, &owner, &worker.thread, &worker.name).and_then(|w| w.runner);
    if worker_hop::runs_elsewhere(true, runner_now, worker_hop::own_node()) {
        tracing::warn!(
            conv_id = %worker.conv_id,
            "mcp: another member runs this worker now, so its outcome here is not kept"
        );
        return;
    }
    worker.activity = Some(live.snapshot());
    if !worker.state.live() {
        worker.ended_ms = Some(now_ms());
        // A finished task is kept whole, to be continued with `repo_task_reply`,
        // until its thread is archived or it sits a day idle
        // (`super::archiving`); a failed task is archived now. A chat's
        // conversation is never distilled: disposal tombstones it
        // (`docs/zen_chat.md` §4).
        worker.resumable = worker.state == State::Done;
        if !worker.resumable && !worker.kind.is_chat() {
            match session.archive_worker(&worker.conv_id) {
                Some(Ok(())) | None => {}
                Some(Err(e)) => tracing::warn!(
                    conv_id = %worker.conv_id,
                    "mcp: the worker could not be archived: {e}"
                ),
            }
        }
    }
    let chat = worker.kind.is_chat();
    // Said on the chat before the record says the answer ended, so a reader
    // woken by the record finds the message there.
    if chat {
        talk::answered(&records, &owner, &worker);
    }
    if let Err(e) = threads::put_worker(&records, &owner, &worker) {
        tracing::warn!(conv_id = %worker.conv_id, "mcp: the worker's outcome could not be kept: {e}");
    }
    // Nobody waited on an alone task: its end is kept for its owner's next
    // conversation (`super::outcomes`).
    if worker.driven == Driven::Alone {
        let kept = Outcome {
            at_ms: now_ms(),
            thread: worker.thread.clone(),
            name: worker.name.clone(),
            label: label.to_string(),
            text: outcome.clone(),
        };
        if let Err(e) = outcomes::keep(&records, &owner, &kept) {
            tracing::warn!(conv_id = %worker.conv_id, "mcp: an alone task's outcome could not be kept for its owner: {e}");
        }
    }
    if let Err(e) = note(&records, &owner, &worker, label, outcome) {
        tracing::warn!(conv_id = %worker.conv_id, "mcp: the outcome could not be entered on the thread: {e}");
    }
    // A chat closed while Zen answered is disposed of once the answer has
    // ended.
    if chat {
        talk::settled(&session, &owner, &worker);
    }
    session.mcp().bump();
}

/// What a running turn has done, shared between the turn that counts it and
/// the beat that writes it.
#[derive(Clone, Default)]
struct Live {
    activity: Arc<Mutex<Activity>>,
    /// Woken when a step is worth telling at once.
    moved: Arc<Notify>,
}

impl Live {
    fn observe(&self, item: &StreamItem) {
        let step = self
            .activity
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .observe_at(item, now_ms());
        if step {
            self.moved.notify_one();
        }
    }

    fn set_doing(&self, doing: Doing) {
        self.activity.lock().unwrap_or_else(|e| e.into_inner()).now = doing;
        self.moved.notify_one();
    }

    fn snapshot(&self) -> Activity {
        self.activity
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

/// Say on the worker's record that its task still runs, what its turn has
/// done, and the journal entries drafted so far — every [`BEAT_EVERY_MS`], and
/// at once when a round ends or a batch of tool calls begins — then wake every
/// waiter; until aborted. The only writer of the record while the turn runs.
async fn beat(
    session: Arc<ZendSession>,
    records: SharedSubstrate,
    owner: String,
    thread: String,
    name: String,
    live: Live,
) {
    let mut every = tokio::time::interval(Duration::from_millis(BEAT_EVERY_MS));
    every.tick().await;
    let mut kept = Instant::now();
    loop {
        tokio::select! {
            _ = every.tick() => {}
            _ = live.moved.notified() => {}
        }
        // Steps that come quicker than the gap are told together: the record
        // replicates, and every write of it is kept.
        let since = kept.elapsed();
        if since < STEP_GAP {
            tokio::time::sleep(STEP_GAP - since).await;
        }
        kept = Instant::now();
        let Some(mut w) = threads::worker(&records, &owner, &thread, &name) else {
            return;
        };
        // Another member took it over: the record is that member's to keep.
        if worker_hop::runs_elsewhere(true, w.runner, worker_hop::own_node()) {
            return;
        }
        w.beat_ms = now_ms();
        w.activity = Some(live.snapshot());
        w.entries = entries_after(session.journal(&w.conv_id).as_ref(), w.entries_after);
        if let Err(e) = threads::put_worker(&records, &owner, &w) {
            tracing::warn!(conv_id = %w.conv_id, "mcp: the task's beat could not be kept: {e}");
        }
        session.mcp().bump();
    }
}

/// The citation audit of `worker`'s final response, over every turn of its
/// conversation (`super::audit`): decoding them reads the substrate, so it
/// runs off the runtime's threads.
async fn audit_of(session: &Arc<ZendSession>, worker: &Worker) -> Result<Audit, String> {
    let own = worker.conv_id.clone();
    let session = Arc::clone(session);
    let answer = worker.final_text.clone();
    let audited = tokio::task::spawn_blocking(move || {
        let detail: Option<TimelineDetail> = session.conversation_detail(&own);
        let turns: Vec<Turn<'_>> = detail
            .iter()
            .flat_map(|d| &d.turns)
            .filter(|t| t.kind == "normal")
            .map(|t| t.user.as_str())
            .collect();
        Audit::of(&answer, &turns)
    })
    .await;
    audited.map_err(|e| {
        tracing::warn!(conv_id = %worker.conv_id, "mcp: the citation audit failed: {e}");
        e.to_string()
    })
}

/// Settle `worker` back on the `answer` it gave before it was sent to open
/// what it cited, when the check round ended any way but done: done, with
/// that answer, its `audit`, and what became of the round — which it answers,
/// for the thread's line.
fn keep_first_answer(worker: &mut Worker, answer: String, audit: &Audit) -> String {
    let round = match worker.state {
        State::Error => format!(
            "failed: {}",
            first_line(worker.error.as_deref().unwrap_or("the turn failed"), 200)
        ),
        State::Question => format!(
            "ended asking: {}",
            first_line(worker.question.as_deref().unwrap_or(""), 200)
        ),
        _ if worker.stopped => "was stopped by Claude".to_string(),
        _ => "had its tools closed before it finished".to_string(),
    };
    worker.audit = Some(format!(
        "{}\nIt was sent back to open these, and the check round {round}; the answer above is \
         the one it gave before. Continue it with `repo_task_reply` to finish the check.",
        audit.render()
    ));
    worker.final_text = answer;
    worker.state = State::Done;
    worker.error = None;
    worker.question = None;
    worker.stopped = false;
    round
}

/// Settle `worker` on how its turn `ended`: its new state, and the thread's
/// line about it.
fn settle(worker: &mut Worker, ended: Ended) -> (&'static str, String) {
    let cite = |w: &Worker| match w.entries.last() {
        Some(e) => format!(" ({}#{})", w.path(), e.id),
        None => String::new(),
    };
    worker.stopped = ended.stopped();
    match ended {
        Ended::Failed(error) => {
            let line = format!("error: {}", first_line(&error, 200));
            worker.error = Some(error);
            worker.state = State::Error;
            ("Task ended:", line)
        }
        Ended::Closed { text, .. } if worker.stopped => {
            worker.final_text = final_reply(&text);
            worker.state = State::Stuck;
            let line = format!(
                "stopped by Claude; it ended: {}{}",
                first_line(&worker.final_text, 200),
                cite(worker)
            );
            ("Task stopped:", line)
        }
        Ended::Closed { text, .. } => {
            worker.final_text = final_reply(&text);
            worker.state = State::Stuck;
            let line = format!(
                "stuck: its tools were closed; it ended: {}{}",
                first_line(&worker.final_text, 200),
                cite(worker)
            );
            ("Task stuck:", line)
        }
        Ended::Replied(text) => {
            worker.final_text = final_reply(&text);
            match question_of(&worker.final_text) {
                Some(q) => {
                    let line = format!("{}{}", first_line(&q, 200), cite(worker));
                    worker.question = Some(q);
                    worker.state = State::Question;
                    ("Task asks:", line)
                }
                None => {
                    worker.state = State::Done;
                    let line = match worker.entries.last() {
                        Some(e) => format!("{}{}", summary_line(&e.text, 200), cite(worker)),
                        None => format!(
                            "no entry was kept; it ended: {}",
                            first_line(&worker.final_text, 200)
                        ),
                    };
                    ("Task ended:", line)
                }
            }
        }
    }
}

/// Settle Zen's side of a chat on how its turn `ended`: `done` with its reply —
/// a closed loop's text still its reply, and a question block no question,
/// since Zen asks Claude in the chat itself — or `error`. The line is for no
/// journal: a chat's thread keeps none ([`note`]).
fn settle_chat(worker: &mut Worker, ended: Ended) -> (&'static str, String) {
    worker.stopped = ended.stopped();
    match ended {
        Ended::Failed(error) => {
            let line = format!("error: {}", first_line(&error, 200));
            worker.error = Some(error);
            worker.state = State::Error;
            ("Chat answer:", line)
        }
        Ended::Closed { text, .. } | Ended::Replied(text) => {
            worker.final_text = final_reply(&text);
            worker.state = State::Done;
            ("Chat answer:", "done".into())
        }
    }
}

/// Run one turn of the worker on `message` to its end, as its role, counting
/// what it does into `live`.
async fn turn(
    session: &Arc<ZendSession>,
    worker: &Worker,
    first: First,
    may_hand: bool,
    live: &Live,
) -> Turned {
    let mode = access::effective_mode(worker.role, None);
    let mut selection = dial_selection(Some(worker.effort.level()), None, Some(true));
    selection.select(ROLE_SELECTOR, worker.kind.option());
    selection.select(DRIVEN_SELECTOR, worker.driven.option());
    apply_tools_dial(&mut selection, mode);
    let (message, round) = first.into_start();
    let mut stream = session
        .submit_worker(
            message,
            worker.conv_id.clone(),
            mode,
            selection,
            round,
            may_hand,
        )
        .await;
    // The round being streamed, and the last one that ended: a turn runs a
    // round per tool call, and its reply is the last round's alone.
    let mut text = String::new();
    let mut ended = String::new();
    let mut failed: Option<String> = None;
    let mut closed: Option<&'static str> = None;
    let mut handed: Option<(u64, u32, usize)> = None;
    while let Some(item) = stream.next().await {
        if let Ok(item) = &item {
            live.observe(item);
        }
        match item {
            Ok(StreamItem::Token(t)) => text.push_str(&t),
            Ok(StreamItem::TurnEnd { .. }) => ended = std::mem::take(&mut text),
            Ok(StreamItem::Error(e)) => {
                failed.get_or_insert(e);
            }
            Ok(StreamItem::Closing(by)) => closed = Some(by),
            Ok(StreamItem::Handed {
                to,
                call_turn,
                iteration,
            }) => handed = Some((to, call_turn, iteration)),
            Ok(_) => {}
            Err(e) => {
                failed.get_or_insert(format!("{e:#}"));
            }
        }
    }
    let last = if text.is_empty() { ended } else { text };
    turned(last, failed, closed, handed)
}

/// What a turn that streamed `text` came to: a failure first, since nothing
/// it said can be trusted; then a move, since its round is not over; then a
/// close; else a reply.
fn turned(
    text: String,
    failed: Option<String>,
    closed: Option<&'static str>,
    handed: Option<(u64, u32, usize)>,
) -> Turned {
    match (failed, closed, handed) {
        (Some(e), _, _) => Turned::Ended(Ended::Failed(e)),
        (None, _, Some((to, call_turn, iteration))) => Turned::Handed {
            to,
            call_turn,
            iteration,
        },
        (None, Some(by), None) => Turned::Ended(Ended::Closed { text, by }),
        (None, None, None) => Turned::Ended(Ended::Replied(text)),
    }
}

/// How a turn's closing draft went.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Drafted {
    /// A draft over the turn's stretch ran and ended: its record is the
    /// journal's last.
    Ended,
    /// None ran, for this reason.
    NotRun(NoDraft),
    /// The conversation keeps no journal.
    NoJournal,
    /// The turn or its draft did not end before [`DRAFT_DEADLINE`].
    TimedOut,
}

/// Draft the journal over the turn's stretch now, and wait for the draft to
/// end.
async fn finish_journal(session: &ZendSession, conv_id: &str) -> Drafted {
    let deadline = Instant::now() + DRAFT_DEADLINE;
    // The reply's stream ends before its turn is committed; the draft reads
    // the committed history, so it waits for the turn to leave the live set.
    while session.live_turn(conv_id).is_some() {
        if Instant::now() >= deadline {
            return Drafted::TimedOut;
        }
        tokio::time::sleep(DRAFT_POLL).await;
    }
    loop {
        match session.journal_draft(conv_id) {
            Some(Ok(_)) => break,
            // A draft already running — the cadence's — may not cover the
            // turn's last stretch: wait it out and ask again.
            Some(Err(NoDraft::Running)) => {
                if Instant::now() >= deadline {
                    return Drafted::TimedOut;
                }
                tokio::time::sleep(DRAFT_POLL).await;
            }
            Some(Err(why)) => return Drafted::NotRun(why),
            None => return Drafted::NoJournal,
        }
    }
    while drafting(session.journal(conv_id).as_ref()) {
        if Instant::now() >= deadline {
            return Drafted::TimedOut;
        }
        tokio::time::sleep(DRAFT_POLL).await;
    }
    Drafted::Ended
}

/// Why a turn whose journal `view` shows no new entry kept none — worded for
/// the caller, who reads it beside the answer: how its closing draft went,
/// and when one ran, its record's reason.
pub fn why_no_entry(view: Option<&Value>, drafted: Drafted) -> String {
    match drafted {
        Drafted::Ended => {}
        Drafted::TimedOut => {
            return format!(
                "its journal draft did not finish within {} minutes",
                DRAFT_DEADLINE.as_secs() / 60
            )
        }
        Drafted::NotRun(NoDraft::NothingNew) => {
            return "nothing in its turn was new to its journal".into()
        }
        Drafted::NotRun(NoDraft::NotOpen) => {
            return "its conversation was not open where it ran, so no draft could run".into()
        }
        Drafted::NotRun(NoDraft::NotDue | NoDraft::Running) => {
            return "no journal draft ran".into()
        }
        Drafted::NoJournal => return "its conversation keeps no journal".into(),
    }
    let Some(last) = view
        .and_then(|v| v.get("drafts"))
        .and_then(Value::as_array)
        .and_then(|d| d.last())
    else {
        return "no journal draft ran".into();
    };
    let detail = last
        .get("detail")
        .and_then(Value::as_str)
        .unwrap_or("no reason was recorded");
    match last.get("result").and_then(Value::as_str) {
        Some("nothing") => format!("the journal judged it added nothing: {detail}"),
        Some("abandoned") => format!("its journal draft failed: {detail}"),
        Some("not_started") => format!("no journal draft ran: {detail}"),
        Some("wrote") => "an earlier entry already covers its turn".into(),
        _ => "no journal draft ran".into(),
    }
}

/// What a read of the waited workers says: per worker still running, what its
/// turn has done, keyed by its path; per journal entry it has written, the
/// entry's summary line, keyed by its address. Each with the line, and the
/// step it tells — the line less its elapsed time, which alone is no step.
pub fn progress_lines(
    read: &[(String, Option<Worker>)],
    now: u64,
) -> Vec<(String, String, String)> {
    let mut lines = Vec::new();
    for w in read.iter().filter_map(|(_, w)| w.as_ref()) {
        if !w.state.working() {
            continue;
        }
        let path = w.path();
        // The step is the sentence with no time passed: what the turn did.
        let step = w
            .activity
            .as_ref()
            .map_or_else(|| Activity::default().sentence(0), |a| a.sentence(0));
        lines.push((path.clone(), step, format!("{path} — {}", w.doing(now))));
        for e in &w.entries {
            let at = format!("{path}#{}", e.id);
            let line = format!("{at} — {}", summary_line(&e.text, 200));
            lines.push((at, String::new(), line));
        }
    }
    lines
}

/// Wait up to `wait` for any of `wanted` — `(thread, worker)` pairs of
/// `owner`'s — to stop working; every one of them as it then stands, `None`
/// for a name that is not a worker. Each step the waited workers take while it
/// waits is said on `progress`, once.
pub async fn wait(
    session: &ZendSession,
    owner: &str,
    wanted: &[(String, String)],
    wait: Duration,
    progress: Option<&UnboundedSender<String>>,
) -> Result<Vec<(String, Option<Worker>)>, String> {
    let records = session.shared_substrate().ok_or_else(loading)?;
    let deadline = Instant::now() + wait;
    let mut told: HashMap<String, (String, Instant)> = HashMap::new();
    loop {
        // Subscribe before reading, so a change between the read and the wait
        // still wakes it.
        let mut changed = session.mcp().watch();
        let now = now_ms();
        let read: Vec<(String, Option<Worker>)> = wanted
            .iter()
            .map(|(thread, name)| {
                let path = format!("task:{thread}/{name}");
                (
                    path,
                    threads::worker(&records, owner, thread, name).map(|w| w.observed(now)),
                )
            })
            .collect();
        let any_settled = read
            .iter()
            .any(|(_, w)| w.as_ref().is_none_or(|w| !w.state.working()));
        let left = deadline.saturating_duration_since(Instant::now());
        if any_settled || left.is_zero() {
            return Ok(read);
        }
        if let Some(progress) = progress {
            for (key, step, line) in progress_lines(&read, now) {
                // A worker's line is said again once it is stale; an entry's
                // never changes, so it is said once.
                let fresh = told.get(&key).is_some_and(|(said, at)| {
                    *said == step && (step.is_empty() || at.elapsed() < PROGRESS_AGAIN)
                });
                if !fresh {
                    told.insert(key, (step, Instant::now()));
                    let _ = progress.send(line);
                }
            }
        }
        // Wake on a change, and at least every [`PROGRESS_AGAIN`]: a task
        // another member runs changes the record without bumping this
        // process, and a stale line is said again.
        let nap = left.min(PROGRESS_AGAIN);
        let _ = tokio::time::timeout(nap, changed.changed()).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::records::MemRecords;
    use crate::session::CLOSED_BY_REPEATS;
    use serde_json::json;

    fn view() -> Value {
        json!({
            "drafting": false,
            "entries": [
                {"id": 3, "text": "Entry 3 — read the plan"},
                {"id": 5, "text": "Entry 5 — verified expiry\n- Saw: ok"},
            ]
        })
    }

    fn worker() -> Worker {
        Worker {
            thread: "review".into(),
            name: "verify".into(),
            conv_id: "c".into(),
            prompt: "p".into(),
            state: State::Running,
            role: AccessRole::Admin,
            person: None,
            priority: Priority::Normal,
            effort: Effort::Deep,
            kind: WorkerKind::Coder,
            driven: Driven::Attended,
            started_ms: 0,
            turn_ms: 0,
            beat_ms: 0,
            ended_ms: None,
            entries_after: None,
            entries: vec![Entry {
                id: 2,
                text: "2026-10-25, 06:00 UTC\nEntry 2 — Checked the expiry.".into(),
            }],
            final_text: String::new(),
            error: None,
            question: None,
            activity: None,
            no_entry: None,
            stopped: false,
            resumable: false,
            resumes: 0,
            audit: None,
            checked: false,
            runner: None,
            by_task: false,
        }
    }

    /// **A cut-off worker is told its task whole**, and what its journal says
    /// it did, the newest entries when there are many.
    #[test]
    fn a_resumed_worker_is_told_its_task_and_its_progress() {
        let mut w = worker();
        w.prompt = "Replace the Battle Cities icon.\nRegenerate all six sizes.".into();
        w.entries = (1..=10)
            .map(|id| Entry {
                id,
                text: format!("Entry {id} — step {id}\n- detail"),
            })
            .collect();
        let told = resumed_message(&w);
        assert!(told.starts_with(RESUMED));
        assert!(
            told.contains("Your task:\nReplace the Battle Cities icon.\nRegenerate all six sizes."),
            "{told}"
        );
        assert!(!told.contains("Entry 2 —"), "{told}");
        assert!(told.ends_with("\n- Entry 10 — step 10"), "{told}");
        w.entries.clear();
        assert!(!resumed_message(&w).contains("journal says"));
    }

    /// **A round cut off is run as resumed, one handed on as live**: only the
    /// first may have run some of its calls.
    #[test]
    fn a_turn_starts_as_what_it_finishes() {
        assert_eq!(
            First::Message("m".into()).into_start(),
            ("m".to_string(), None)
        );
        assert_eq!(
            First::Interrupted { call_turn: 7 }.into_start(),
            (
                String::new(),
                Some(Resume {
                    call_turn: 7,
                    dispatch: Dispatch::Resumed,
                    iteration: 0
                })
            )
        );
        assert_eq!(
            First::Round {
                call_turn: 7,
                iteration: 3
            }
            .into_start(),
            (
                String::new(),
                Some(Resume {
                    call_turn: 7,
                    dispatch: Dispatch::Live,
                    iteration: 3
                })
            )
        );
    }

    /// **A check round that does not finish leaves the task done** with the
    /// answer it gave before, its audit saying what became of the round.
    #[test]
    fn a_check_round_that_fails_keeps_the_first_answer() {
        let audit = Audit {
            seen: vec!["https://b.com".into()],
            ..Audit::default()
        };
        let mut w = worker();
        w.state = State::Error;
        w.error = Some("the engine went away".into());
        w.final_text = String::new();
        let round = keep_first_answer(&mut w, "First answer, citing https://b.com".into(), &audit);
        assert_eq!(round, "failed: the engine went away");
        assert_eq!(w.state, State::Done);
        assert_eq!(w.final_text, "First answer, citing https://b.com");
        assert_eq!(w.error, None);
        assert_eq!(
            w.audit.as_deref(),
            Some(
                "Citation audit: 1 cited — 0 opened, 1 seen only in tool results, 0 never seen \
                 in the run.\nSeen only in tool results — a search hit, a link, an error — never \
                 opened: https://b.com\nIt was sent back to open these, and the check round failed: the \
                 engine went away; the answer above is the one it gave before. Continue it with \
                 `repo_task_reply` to finish the check."
            )
        );
    }

    /// **A record that does not read still takes its name**: it counts as
    /// ended, so its name is not handed out again for a new conversation.
    #[test]
    fn a_name_is_taken_by_its_key_whether_or_not_its_record_reads() {
        let records = MemRecords::default();
        let mut live = worker();
        live.beat_ms = 1_000;
        threads::put_worker(&records, "o", &live).unwrap();
        records
            .put(
                &threads::worker_key("o", "review", "old"),
                json!({"name": "old", "written": "by an earlier build"}),
            )
            .unwrap();
        let mut t = taken(&records, "o", "review", 1_000);
        t.sort();
        assert_eq!(
            t,
            [("old".to_string(), false), ("verify".to_string(), true)]
        );
        assert_eq!(names::worker_name("review", "old", &t).unwrap(), "old.2");
    }

    #[test]
    fn a_tasks_entries_are_the_ones_after_the_newest_before_it() {
        let v = view();
        assert_eq!(newest_entry(Some(&v)), Some(5));
        assert_eq!(newest_entry(None), None);
        let after = entries_after(Some(&v), Some(3));
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].id, 5);
        assert_eq!(entries_after(Some(&v), None).len(), 2);
        assert!(!drafting(Some(&v)));
        assert!(drafting(Some(&json!({"drafting": true}))));
    }

    #[test]
    fn the_final_reply_is_all_that_follows_the_thinking() {
        assert_eq!(
            final_reply("<think>hmm</think>\n\nVerdict: ok"),
            "Verdict: ok"
        );
        assert_eq!(final_reply("plain"), "plain");
        let long = "x".repeat(5_000);
        assert_eq!(final_reply(&long), long);
        // A reply that quotes the close of a thought keeps everything after
        // its own thinking, the quote and what follows it included.
        assert_eq!(
            final_reply(
                "<think>check the close</think>\nThe turn ends at `</think>` or `<|im_end|>`; \
                 both are fine."
            ),
            "The turn ends at `</think>` or `<|im_end|>`; both are fine."
        );
    }

    /// **A turn with no entry says why**, from its last draft's record.
    #[test]
    fn a_turn_with_no_entry_says_why_from_its_draft() {
        let with = |draft: Value| json!({"entries": [], "drafts": [draft]});
        assert_eq!(
            why_no_entry(
                Some(&with(
                    json!({"result": "abandoned", "detail": "the entry was refused: x"})
                )),
                Drafted::Ended
            ),
            "its journal draft failed: the entry was refused: x"
        );
        assert_eq!(
            why_no_entry(
                Some(&with(
                    json!({"result": "nothing", "detail": "more of the same"})
                )),
                Drafted::Ended
            ),
            "the journal judged it added nothing: more of the same"
        );
        assert_eq!(
            why_no_entry(Some(&json!({"drafts": []})), Drafted::Ended),
            "no journal draft ran"
        );
        assert_eq!(
            why_no_entry(None, Drafted::TimedOut),
            "its journal draft did not finish within 15 minutes"
        );
        // A draft that did not run this turn is not blamed on an earlier
        // turn's record.
        let earlier = with(json!({"result": "abandoned", "detail": "the entry was refused: x"}));
        assert_eq!(
            why_no_entry(Some(&earlier), Drafted::NotRun(NoDraft::NothingNew)),
            "nothing in its turn was new to its journal"
        );
    }

    /// **Progress is a line per running worker and per entry it wrote**, each
    /// keyed so a repeat is not said twice; the step a worker's line tells
    /// leaves out the elapsed time, which alone is no step.
    #[test]
    fn progress_is_a_line_per_running_worker_and_per_entry() {
        let mut running = worker();
        running.turn_ms = 1_000;
        running.activity = Some(Activity {
            rounds: 1,
            ..Activity::default()
        });
        let mut done = worker();
        done.name = "done".into();
        done.state = State::Done;
        let read = vec![
            ("T:review/verify".to_string(), Some(running)),
            ("T:review/done".to_string(), Some(done)),
            ("T:review/gone".to_string(), None),
        ];
        let lines = progress_lines(&read, 46_000);
        assert_eq!(
            lines,
            vec![
                (
                    "T:review/verify".to_string(),
                    "running 0s, 1 round: 0 thinking tokens, no tool calls yet; now starting"
                        .to_string(),
                    "T:review/verify — running 45s, 1 round: 0 thinking tokens, no tool calls \
                     yet; now starting"
                        .to_string(),
                ),
                (
                    "T:review/verify#2".to_string(),
                    String::new(),
                    "T:review/verify#2 — Entry 2 — Checked the expiry.".to_string(),
                ),
            ]
        );
        assert_eq!(progress_lines(&read, 90_000)[0].1, lines[0].1);
    }

    #[test]
    fn a_question_block_is_read_from_its_line_and_never_from_mid_text() {
        let reply = "I read the spec.\nQUESTION: Should expiry be inclusive?\nOPTIONS: yes; no";
        assert_eq!(
            question_of(reply).unwrap(),
            "QUESTION: Should expiry be inclusive?\nOPTIONS: yes; no"
        );
        assert_eq!(question_of("QUESTION: why?").unwrap(), "QUESTION: why?");
        assert_eq!(question_of("Its QUESTION: field is parsed."), None);
        assert_eq!(question_of("QUESTION:"), None);
        assert_eq!(question_of("Verdict: done."), None);
    }

    #[test]
    fn a_reply_settles_done_asking_stuck_or_failed() {
        let mut w = worker();
        let (label, line) = settle(&mut w, Ended::Replied("Verdict: ok".into()));
        assert_eq!((w.state, label), (State::Done, "Task ended:"));
        assert_eq!(line, "Entry 2 — Checked the expiry. (T:review/verify#2)");

        let mut w = worker();
        let (label, line) = settle(
            &mut w,
            Ended::Replied("Read it.\nQUESTION: Inclusive?\nOPTIONS: a; b".into()),
        );
        assert_eq!((w.state, label), (State::Question, "Task asks:"));
        assert_eq!(line, "QUESTION: Inclusive? (T:review/verify#2)");
        assert_eq!(
            w.question.as_deref(),
            Some("QUESTION: Inclusive?\nOPTIONS: a; b")
        );

        let mut w = worker();
        let (label, _) = settle(
            &mut w,
            Ended::Closed {
                text: "I could not finish.".into(),
                by: CLOSED_BY_REPEATS,
            },
        );
        assert_eq!((w.state, label), (State::Stuck, "Task stuck:"));
        assert!(!w.stopped);

        // **A stopped worker is live, and says Claude stopped it**: stuck,
        // so `repo_task_reply` redirects it.
        let mut w = worker();
        let (label, line) = settle(
            &mut w,
            Ended::Closed {
                text: "Found v2 only.".into(),
                by: CLOSED_BY_CLAUDE,
            },
        );
        assert_eq!((w.state, label), (State::Stuck, "Task stopped:"));
        assert!(w.stopped && w.state.live());
        assert_eq!(
            line,
            "stopped by Claude; it ended: Found v2 only. (T:review/verify#2)"
        );

        let mut w = worker();
        let (label, line) = settle(&mut w, Ended::Failed("boom".into()));
        assert_eq!((w.state, label), (State::Error, "Task ended:"));
        assert_eq!(line, "error: boom");
    }

    /// **A turn that failed failed, a turn that moved moved, whatever else
    /// it said**; a closed turn is its close; else its reply.
    #[test]
    fn a_turn_comes_to_its_failure_its_move_its_close_or_its_reply() {
        let t = || "said".to_string();
        assert!(matches!(
            turned(t(), Some("boom".into()), Some("round_cap"), Some((1, 2, 3))),
            Turned::Ended(Ended::Failed(e)) if e == "boom"
        ));
        assert!(matches!(
            turned(t(), None, Some("round_cap"), Some((1, 2, 3))),
            Turned::Handed {
                to: 1,
                call_turn: 2,
                iteration: 3
            }
        ));
        assert!(matches!(
            turned(t(), None, Some("round_cap"), None),
            Turned::Ended(Ended::Closed {
                by: "round_cap",
                ..
            })
        ));
        assert!(matches!(
            turned(t(), None, None, None),
            Turned::Ended(Ended::Replied(text)) if text == "said"
        ));
    }

    #[test]
    fn every_task_is_told_how_to_ask() {
        assert!(ASKING.contains(QUESTION_MARK));
        assert!(ASKING.contains("OPTIONS:") && ASKING.contains("POINTERS:"));
    }

    /// **Zen's side of a chat settles done with its reply, never as a
    /// question** — Zen asks Claude in the chat itself — and its message ends
    /// with no question block; resumed, it is told the message it was
    /// answering, not a task.
    #[test]
    fn a_chat_turn_settles_done_with_its_reply() {
        let chat = || Worker {
            thread: "~0123456789abcdef".into(),
            name: "zen".into(),
            kind: WorkerKind::Chat,
            prompt: "how big is the arena?".into(),
            ..worker()
        };
        let mut w = chat();
        settle_chat(
            &mut w,
            Ended::Replied("<think>x</think>Which arena?\nQUESTION: which?".into()),
        );
        assert_eq!(w.state, State::Done);
        assert_eq!(w.final_text, "Which arena?\nQUESTION: which?");
        assert_eq!(w.question, None);

        let mut w = chat();
        settle_chat(
            &mut w,
            Ended::Closed {
                text: "About 30 GiB.".into(),
                by: CLOSED_BY_CLAUDE,
            },
        );
        assert_eq!((w.state, w.stopped), (State::Done, true));
        assert_eq!(w.final_text, "About 30 GiB.");

        let mut w = chat();
        settle_chat(&mut w, Ended::Failed("boom".into()));
        assert_eq!((w.state, w.error.as_deref()), (State::Error, Some("boom")));

        assert_eq!(closing_of(WorkerKind::Chat), "");
        assert_eq!(closing_of(WorkerKind::Coder), ASKING);
        assert_eq!(
            resumed_message(&chat()),
            format!("{CHAT_RESUMED}\n\nThe message you were answering:\nhow big is the arena?")
        );
    }

    /// **A chat's thread keeps no journal line**; a task's thread does.
    #[test]
    fn a_chat_workers_notes_are_not_entered() {
        let r = MemRecords::default();
        let chat = Worker {
            thread: "~0123456789abcdef".into(),
            name: "zen".into(),
            kind: WorkerKind::Chat,
            ..worker()
        };
        note(&r, "o", &chat, "Claude started:", "hello".into()).unwrap();
        assert_eq!(threads::note_count(&r, "o", "~0123456789abcdef"), 0);
        let task = worker();
        note(&r, "o", &task, "Claude started:", "verify".into()).unwrap();
        assert_eq!(threads::note_count(&r, "o", &task.thread), 1);
    }

    #[test]
    fn an_entrys_summary_skips_the_date_above_it() {
        assert_eq!(
            summary_line(
                "2026-10-25, 06:00 UTC\nEntry 1 — Read the file.\n- Saw: x",
                200
            ),
            "Entry 1 — Read the file."
        );
        assert_eq!(summary_line("plain text", 200), "plain text");
    }

    #[test]
    fn a_first_line_is_cut_and_marked() {
        assert_eq!(
            first_line("\n  Verify the expiry  \nmore", 40),
            "Verify the expiry"
        );
        assert_eq!(first_line("abcdef", 3), "abc…");
        assert_eq!(first_line("", 3), "");
    }
}
