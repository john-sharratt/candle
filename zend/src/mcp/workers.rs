//! Running a task on a worker, answering a worker that asked, and waiting on
//! workers (`docs/claude_zend_gateway.md` §3.3, §4, §5, §6.2).
//!
//! A task's turn is an ordinary dialogue turn on the worker's own zend
//! conversation, submitted as `POST /v1/chat/completions` submits one, so it is
//! projected with provenance, the working set and the journal like any
//! dialogue. It runs in a task of its own, never inside a tool call: the call
//! that started it answers within `wait_s`, and `task_wait` reads how it stands.
//!
//! When a turn ends, the task asks for a journal draft at once instead of
//! waiting for the cadence, waits for it, and takes the entries written since
//! the turn began. Then the worker settles:
//!
//! - **asking** — its final response ended with the question block of
//!   [`ASKING`] — it waits, live, for `task_reply`;
//! - **stuck** — the repeat guard or the round cap closed its tools — it waits
//!   the same way;
//! - **done** or **failed** — it is archived, its evidence kept.
//!
//! Each settling is entered on the thread. An answer is the worker's next turn,
//! and the worker settles again after it.
//!
//! A **part** — what a worker starts under itself with `sub_run`
//! (`super::parts`) — runs the same way, as the role its worker runs as, but
//! asks nobody: it ends `done` or `error`, is archived either way, and its
//! settling is its worker's business, not an entry on Claude's thread.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use candle_conversation::persistence::SharedSubstrate;
use futures::StreamExt;
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::Notify;
use web::auth::Role as AccessRole;

use super::activity::{Activity, Doing};
use super::caller::Caller;
use super::names;
use super::nudges;
use super::records::Records;
use super::stops;
use super::threads::{self, Entry, Note, Priority, State, Worker, BEAT_EVERY_MS};
use crate::access;
use crate::api::chat::{apply_tools_dial, dial_selection};
use crate::session::{NoDraft, StreamItem, ZendSession, CLOSED_BY_CLAUDE};
use crate::types::{ChatMessage, Role};

/// The effort a task thinks at: `deep`. Tokens are not the constraint.
const TASK_EFFORT: u8 = 3;
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

/// What every part's prompt ends with: whose part it is, and that nobody
/// answers it.
pub fn part_close(parent: &str) -> String {
    format!(
        "\n\n---\nYou are a part of the task of {parent}, which started you with `sub_run` and \
         waits for you: your final response, and the journal entry drafted from it, are what it \
         gets back. Nobody can answer a question from you. If something blocks you, say exactly \
         what in your final response — what you tried, what happened, and the files, lines or \
         commands it rests on — and end there."
    )
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn loading() -> String {
    "zend is still loading its model; try again when `cluster_status` says it is ready".into()
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

/// The reply of a final response, whole: what follows its last `</think>`.
pub fn final_reply(text: &str) -> String {
    text.rsplit_once("</think>")
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
/// decoding at `priority`; the worker's record, `running`.
pub fn start(
    session: &Arc<ZendSession>,
    caller: &Caller,
    thread: &str,
    wanted: &str,
    prompt: &str,
    priority: Priority,
) -> Result<Worker, String> {
    let records = session.shared_substrate().ok_or_else(loading)?;
    let owner = caller.key();
    names::check(thread)?;
    if prompt.trim().is_empty() {
        return Err("the task's `prompt` is empty: state the task in full".into());
    }
    let now = now_ms();
    threads::attach(&records, &owner, thread, now).map_err(|e| e.to_string())?;
    let _held = BEGIN.lock().unwrap();
    let taken = taken(&records, &owner, thread, now);
    let name = names::child_name(thread, "", wanted, &taken)?;
    let worker = fresh(
        session,
        &owner,
        thread,
        name,
        prompt,
        (caller.role, priority),
        now,
    );
    note(
        &records,
        &owner,
        &worker,
        "Claude asked:",
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

/// Every conversation under `thread`, by lineage name, and whether it is live.
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

/// A new worker's record, `running`, its turn beginning `now`, running as
/// `role` at `priority`.
fn fresh(
    session: &ZendSession,
    owner: &str,
    thread: &str,
    name: String,
    prompt: &str,
    (role, priority): (AccessRole, Priority),
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
        priority,
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
    }
}

/// Start a part named after `wanted` under `parent`, running `prompt` as the
/// role `parent` runs as; the part's record, `running`.
pub fn start_part(
    session: &Arc<ZendSession>,
    owner: &str,
    parent: &Worker,
    wanted: &str,
    prompt: &str,
) -> Result<Worker, String> {
    let records = session.shared_substrate().ok_or_else(loading)?;
    if prompt.trim().is_empty() {
        return Err("the part's `prompt` is empty: state its task in full".into());
    }
    let now = now_ms();
    let _held = BEGIN.lock().unwrap();
    let taken = taken(&records, owner, &parent.thread, now);
    let name = names::child_name(&parent.thread, &parent.name, wanted, &taken)?;
    let part = fresh(
        session,
        owner,
        &parent.thread,
        name,
        prompt,
        (parent.role, parent.priority),
        now,
    );
    let message = format!("{prompt}{}", part_close(&parent.path()));
    begin(session, &records, owner, part, message)
}

/// Answer the worker `name` of `thread`, which asked or is stuck: the answer
/// is its next turn. The worker's record, `running` again.
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
    let _held = BEGIN.lock().unwrap();
    let Some(found) = threads::worker(&records, &owner, thread, name) else {
        return Err(format!("there is no worker `T:{thread}/{name}`"));
    };
    let mut worker = found.observed(now_ms());
    if !matches!(worker.state, State::Question | State::Stuck) {
        return Err(format!(
            "`{}` is {}, not waiting on an answer",
            worker.path(),
            match worker.state {
                State::Running => "still working",
                _ => "finished",
            }
        ));
    }
    let now = now_ms();
    worker.state = State::Running;
    worker.role = caller.role;
    worker.beat_ms = now;
    worker.turn_ms = now;
    worker.question = None;
    worker.entries_after = newest_entry(session.journal(&worker.conv_id).as_ref());
    worker.entries = Vec::new();
    note(
        &records,
        &owner,
        &worker,
        "Claude answered:",
        first_line(text, PROMPT_QUOTE),
    )?;
    begin(session, &records, &owner, worker, format!("{text}{ASKING}"))
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
        return Err(format!("there is no worker `T:{thread}/{name}`"));
    };
    let worker = found.observed(now_ms());
    if !worker.state.working() {
        return Err(format!(
            "`{}` is not working, so it would never read a nudge; {}",
            worker.path(),
            if worker.state.live() {
                "it waits on an answer — use `task_reply`"
            } else {
                "it has finished — start a new task"
            }
        ));
    }
    nudges::queue(&records, &worker.conv_id, text, now_ms()).map_err(|e| e.to_string())?;
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
    let _held = BEGIN.lock().unwrap();
    let Some(found) = threads::worker(&records, &owner, thread, name) else {
        return Err(format!("there is no worker `T:{thread}/{name}`"));
    };
    let worker = found.observed(now_ms());
    if !worker.state.working() {
        return Err(format!(
            "`{}` is not running, so there is nothing to stop; {}",
            worker.path(),
            if worker.state.live() {
                "it waits on an answer — redirect it with `task_reply`"
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

/// Enter a line about `worker` on its thread's journal.
fn note(
    records: &dyn Records,
    owner: &str,
    worker: &Worker,
    label: &str,
    text: String,
) -> Result<(), String> {
    threads::add_note(
        records,
        owner,
        &worker.thread,
        &Note {
            at_ms: now_ms(),
            label: label.into(),
            text: format!("{} — {text}", worker.path()),
            worker: Some(worker.path()),
        },
    )
    .map_err(|e| e.to_string())
}

/// Keep `worker` as running and start its turn on `message`.
fn begin(
    session: &Arc<ZendSession>,
    records: &dyn Records,
    owner: &str,
    worker: Worker,
    message: String,
) -> Result<Worker, String> {
    // A stop sent as the last turn ended — after its settling looked — is
    // not this turn's.
    stops::take(records, &worker.conv_id);
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
        message,
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

impl Ended {
    /// Whether Claude stopped the turn (`task_stop`).
    fn stopped(&self) -> bool {
        matches!(self, Ended::Closed { by, .. } if *by == CLOSED_BY_CLAUDE)
    }
}

/// The turn's life after [`begin`], and the worker's settling.
async fn run(session: Arc<ZendSession>, owner: String, mut worker: Worker, message: String) {
    let Some(records) = session.shared_substrate() else {
        return;
    };
    // The record says the task still runs for as long as it does — through
    // the turn and the draft after it — so no reader takes it for cut off,
    // and carries what the turn has done so far.
    let live = Live::default();
    let beating = tokio::spawn(beat(
        Arc::clone(&session),
        records.clone(),
        owner.clone(),
        worker.thread.clone(),
        worker.name.clone(),
        live.clone(),
    ));
    let ended = turn(&session, &worker, message, &live).await;
    live.set_doing(Doing::Journaling);
    // A nudge that came after the turn's last tool round was never read.
    for undelivered in nudges::take(&records, &worker.conv_id) {
        let line = format!(
            "not delivered — the turn ended without another tool round: {}",
            first_line(&undelivered, PROMPT_QUOTE)
        );
        if let Err(e) = note(&records, &owner, &worker, "Nudge dropped:", line) {
            tracing::warn!(conv_id = %worker.conv_id, "mcp: an undelivered nudge could not be entered: {e}");
        }
    }
    worker.no_entry = None;
    let drafted = match ended {
        Ended::Failed(_) => None,
        _ => Some(finish_journal(&session, &worker.conv_id).await),
    };
    // A stop that came after the turn's last tool round — while it answered
    // or drafted — found it finishing.
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
    let (label, outcome) = if worker.is_part() {
        settle_part(&mut worker, ended)
    } else {
        settle(&mut worker, ended)
    };
    // The beat stops, and is done writing, before the settled record is.
    beating.abort();
    let _ = beating.await;
    worker.activity = Some(live.snapshot());
    if !worker.state.live() {
        worker.ended_ms = Some(now_ms());
        match session.archive_worker(&worker.conv_id) {
            Some(Ok(())) | None => {}
            Some(Err(e)) => tracing::warn!(
                conv_id = %worker.conv_id,
                "mcp: the worker could not be archived: {e}"
            ),
        }
    }
    if let Err(e) = threads::put_worker(&records, &owner, &worker) {
        tracing::warn!(conv_id = %worker.conv_id, "mcp: the worker's outcome could not be kept: {e}");
    }
    // The thread is Claude's record of what Claude did; a part's outcome goes
    // to the worker that started it, as its `sub_run` result.
    if !worker.is_part() {
        if let Err(e) = note(&records, &owner, &worker, label, outcome) {
            tracing::warn!(conv_id = %worker.conv_id, "mcp: the outcome could not be entered on the thread: {e}");
        }
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
            .observe(item);
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
        w.beat_ms = now_ms();
        w.activity = Some(live.snapshot());
        w.entries = entries_after(session.journal(&w.conv_id).as_ref(), w.entries_after);
        if let Err(e) = threads::put_worker(&records, &owner, &w) {
            tracing::warn!(conv_id = %w.conv_id, "mcp: the task's beat could not be kept: {e}");
        }
        session.mcp().bump();
    }
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

/// Settle the part `part` on how its turn `ended`: `done` with its final
/// response — a part asks nobody, so a question block or a closed loop is
/// still its answer, said as such — or `error`.
fn settle_part(part: &mut Worker, ended: Ended) -> (&'static str, String) {
    match ended {
        Ended::Failed(error) => {
            let line = format!("error: {}", first_line(&error, 200));
            part.error = Some(error);
            part.state = State::Error;
            ("Part ended:", line)
        }
        Ended::Closed { text, by } => {
            let why = if by == CLOSED_BY_CLAUDE {
                "Claude stopped it"
            } else {
                "the same call repeated, or the round cap reached"
            };
            part.final_text = format!(
                "Its tools were closed before it finished — {why} — and it ended:\n{}",
                final_reply(&text)
            );
            part.state = State::Done;
            ("Part ended:", "its tools were closed".into())
        }
        Ended::Replied(text) => {
            part.final_text = final_reply(&text);
            part.state = State::Done;
            ("Part ended:", "done".into())
        }
    }
}

/// Run one turn of the worker on `message` to its end, as its role, counting
/// what it does into `live`.
async fn turn(session: &Arc<ZendSession>, worker: &Worker, message: String, live: &Live) -> Ended {
    let mode = access::effective_mode(worker.role, None);
    let mut selection = dial_selection(Some(TASK_EFFORT), None, Some(true));
    apply_tools_dial(&mut selection, mode);
    let mut stream = session
        .submit(
            vec![ChatMessage::new(Role::User, message)],
            None,
            worker.conv_id.clone(),
            None,
            None,
            false,
            mode,
            None,
            selection,
        )
        .await;
    let mut text = String::new();
    let mut failed: Option<String> = None;
    let mut closed: Option<&'static str> = None;
    while let Some(item) = stream.next().await {
        if let Ok(item) = &item {
            live.observe(item);
        }
        match item {
            Ok(StreamItem::Token(t)) => text.push_str(&t),
            Ok(StreamItem::Error(e)) => {
                failed.get_or_insert(e);
            }
            Ok(StreamItem::Closing(by)) => closed = Some(by),
            Ok(_) => {}
            Err(e) => {
                failed.get_or_insert(format!("{e:#}"));
            }
        }
    }
    match (failed, closed) {
        (Some(e), _) => Ended::Failed(e),
        (None, Some(by)) => Ended::Closed { text, by },
        (None, None) => Ended::Replied(text),
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
                let path = format!("T:{thread}/{name}");
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
    use crate::session::{CLOSED_BY_CAP, CLOSED_BY_REPEATS};
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
            priority: Priority::Normal,
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
        }
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
        assert_eq!(names::child_name("review", "", "old", &t).unwrap(), "old.2");
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
        // so `task_reply` redirects it.
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

    #[test]
    fn every_task_is_told_how_to_ask() {
        assert!(ASKING.contains(QUESTION_MARK));
        assert!(ASKING.contains("OPTIONS:") && ASKING.contains("POINTERS:"));
    }

    /// **A part never waits on anyone**: a question block or a closed loop is
    /// its answer, settled `done` — said as such — and only a failure is
    /// `error`. Either way it is no longer live.
    #[test]
    fn a_part_settles_done_or_failed_and_never_waits() {
        let part = || Worker {
            name: "verify/read".into(),
            ..worker()
        };
        let mut p = part();
        settle_part(&mut p, Ended::Replied("Read it.\nQUESTION: which?".into()));
        assert_eq!(p.state, State::Done);
        assert_eq!(p.final_text, "Read it.\nQUESTION: which?");
        assert_eq!(p.question, None);

        let mut p = part();
        settle_part(
            &mut p,
            Ended::Closed {
                text: "<think>x</think>I could not finish.".into(),
                by: CLOSED_BY_CAP,
            },
        );
        assert_eq!(p.state, State::Done);
        assert_eq!(
            p.final_text,
            "Its tools were closed before it finished — the same call repeated, or the \
             round cap reached — and it ended:\nI could not finish."
        );

        let mut p = part();
        settle_part(&mut p, Ended::Failed("boom".into()));
        assert_eq!((p.state, p.error.as_deref()), (State::Error, Some("boom")));
        assert!(!p.state.live());
    }

    #[test]
    fn a_part_is_told_whose_it_is_and_that_nobody_answers_it() {
        let close = part_close("T:review/verify");
        assert!(close.contains("the task of T:review/verify"), "{close}");
        assert!(
            close.contains("Nobody can answer a question from you"),
            "{close}"
        );
        assert!(!close.contains(QUESTION_MARK));
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
