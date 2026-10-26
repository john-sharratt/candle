//! Threads, workers and the thread's journal, as records
//! (`docs/claude_zend_gateway.md` §4).
//!
//! - A **thread** is one feature's place: `mcp/thread/<owner>/<name>`.
//! - A **worker** is one task on it: `mcp/worker/<owner>/<thread>/<name>`,
//!   naming the zend conversation the task runs in and how it stands.
//! - The **thread's journal is written, not drafted**: each task Claude starts
//!   and each one that ends is one entry, kept as its own record under
//!   `mcp/note/<owner>/<thread>/` so two tasks ending at once never race to
//!   rewrite one record. An entry's number is its place in key order.
//!
//! Every key starts with the owner's key, so one person's records are never
//! another's, whatever the names.

use std::cmp::Reverse;
use std::sync::atomic::{AtomicU64, Ordering};

use candle_conversation::projection::DecodePriority;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use web::auth::Role;

use super::activity::Activity;
use super::names::{parent_of, Path};
use super::records::Records;

/// Orders the notes one process writes within one millisecond.
static NOTE_SEQ: AtomicU64 = AtomicU64::new(0);

/// How a worker stands (`docs/claude_zend_gateway.md` §3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    Running,
    /// Its turn ended asking Claude a question; it waits, live, for
    /// `repo_task_reply`.
    Question,
    /// The repeat guard or the round cap closed its tools; it waits, live, for
    /// `repo_task_reply` to say how to go on.
    Stuck,
    Done,
    Error,
}

impl State {
    /// Whether the worker is still live: its name is taken and it is not
    /// archived — working, or waiting on Claude.
    pub fn live(self) -> bool {
        matches!(self, Self::Running | Self::Question | Self::Stuck)
    }

    /// Whether zend is still working on it: what `repo_task_wait` waits out.
    pub fn working(self) -> bool {
        matches!(self, Self::Running)
    }
}

/// How urgently a task decodes (`docs/claude_zend_gateway.md` §3.4): `normal`,
/// below a live user, unless someone waits on it interactively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    #[default]
    Normal,
    High,
}

impl Priority {
    /// The priority a call names, `normal` when it names none.
    pub fn of(name: Option<&str>) -> Result<Self, String> {
        match name {
            None | Some("normal") => Ok(Self::Normal),
            Some("high") => Ok(Self::High),
            Some(other) => Err(format!("`priority` is `normal` or `high`, not `{other}`")),
        }
    }

    /// The band its conversation decodes at.
    pub fn band(self) -> DecodePriority {
        match self {
            Self::Normal => DecodePriority::Normal,
            Self::High => DecodePriority::High,
        }
    }
}

/// How hard a task's turns think: `deep` unless the call asks otherwise.
/// Separate from [`Priority`], which says only how soon its turns decode.
/// A task that weighs many sources against each other — research that must
/// reconcile what it fetched — is one to ask `exhaustive` for: thinking is
/// where it notices that two of its findings disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
    Balanced,
    #[default]
    Deep,
    Exhaustive,
}

impl Effort {
    /// The effort a call names, `deep` when it names none.
    pub fn of(name: Option<&str>) -> Result<Self, String> {
        match name {
            Some("balanced") => Ok(Self::Balanced),
            None | Some("deep") => Ok(Self::Deep),
            Some("exhaustive") => Ok(Self::Exhaustive),
            Some(other) => Err(format!(
                "`effort` is `balanced`, `deep` or `exhaustive`, not `{other}`"
            )),
        }
    }

    /// The effort dial's level (`crate::api::chat::EFFORT_OPTIONS`).
    pub fn level(self) -> u8 {
        match self {
            Self::Balanced => 2,
            Self::Deep => 3,
            Self::Exhaustive => 4,
        }
    }
}

/// One of a task's journal entries, as the worker's journal kept it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub id: u64,
    pub text: String,
}

/// A worker's record — a task Claude started, or a part a worker started
/// under itself with `sub_run`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Worker {
    pub thread: String,
    /// Its lineage name below the thread: `verify` for a task, `verify/read`
    /// for a part of it ([`super::names`]).
    pub name: String,
    /// The zend conversation the task runs in.
    pub conv_id: String,
    pub prompt: String,
    pub state: State,
    /// The role its current turn runs as: its owner's role when the turn
    /// began, read afresh for every task and answer, so a part runs with
    /// exactly the tools its worker has.
    pub role: Role,
    /// How urgently its turns decode; a part decodes as its worker does.
    pub priority: Priority,
    /// How hard its turns think; a part thinks as its worker does.
    #[serde(default)]
    pub effort: Effort,
    pub started_ms: u64,
    /// When its current turn began: a part this worker started in an earlier
    /// turn is never mistaken for the answer to a call in this one.
    pub turn_ms: u64,
    /// When the member running the task last said it still was
    /// ([`BEAT_EVERY_MS`]).
    #[serde(default)]
    pub beat_ms: u64,
    #[serde(default)]
    pub ended_ms: Option<u64>,
    /// The newest entry id the worker's journal held before the task, so the
    /// task's own entries are the ones after it.
    #[serde(default)]
    pub entries_after: Option<u64>,
    /// The task's entries: those drafted so far while it runs, all of them
    /// once it ended.
    #[serde(default)]
    pub entries: Vec<Entry>,
    /// The task's final response, whole, once it ended.
    #[serde(default)]
    pub final_text: String,
    #[serde(default)]
    pub error: Option<String>,
    /// What it asks, while it is a [`State::Question`]: the question block its
    /// final response ended with.
    #[serde(default)]
    pub question: Option<String>,
    /// What its current turn has done so far, counted from the turn's stream —
    /// none until its first turn begins.
    pub activity: Option<Activity>,
    /// Why its turn ended with no journal entry, as the draft recorded it;
    /// none when it kept one.
    pub no_entry: Option<String>,
    /// Whether Claude stopped its last turn (`repo_task_stop`): it is stuck, live,
    /// for `repo_task_reply` to redirect.
    #[serde(default)]
    pub stopped: bool,
    /// Whether a finished task's conversation is still whole, so `repo_task_reply`
    /// can continue it: true from when it ends `done` until it is archived —
    /// with its thread, or after a day idle (`super::archiving`).
    #[serde(default)]
    pub resumable: bool,
    /// The citation audit of its final response (`super::audit`), once it
    /// ended `done` citing anything.
    #[serde(default)]
    pub audit: Option<String>,
}

/// How often the member running a task says so on its record.
pub const BEAT_EVERY_MS: u64 = 30_000;
/// How long a running record may go unsaid before it is read as cut off: four
/// beats missed, so a slow write is never mistaken for a lost task.
pub const BEAT_LOST_MS: u64 = 4 * BEAT_EVERY_MS;

impl Worker {
    /// The worker as it stands at `now_ms`: a task whose member stopped
    /// saying it runs — it restarted, or lost the task — is read as ended in
    /// an error, since nothing will finish it. Any member reads it the same
    /// way, which is why the beat is on the replicated record and not in one
    /// process's memory.
    pub fn observed(mut self, now_ms: u64) -> Self {
        if self.state == State::Running && now_ms.saturating_sub(self.beat_ms) > BEAT_LOST_MS {
            self.state = State::Error;
            self.error = Some(
                "the task stopped being run: the member running it restarted or lost it. \
                 Its conversation keeps what it did; run the task again"
                    .into(),
            );
        }
        self
    }

    pub fn path(&self) -> String {
        Path::Worker {
            thread: self.thread.clone(),
            worker: self.name.clone(),
        }
        .text()
    }

    /// What its current turn has done, in one sentence, at `now_ms`.
    pub fn doing(&self, now_ms: u64) -> String {
        let elapsed = now_ms.saturating_sub(self.turn_ms);
        match &self.activity {
            Some(a) => a.sentence(elapsed),
            None => Activity::default().sentence(elapsed),
        }
    }

    /// Whether it is a part of another worker's task, not a task Claude
    /// started.
    pub fn is_part(&self) -> bool {
        !parent_of(&self.name).is_empty()
    }
}

/// A thread's record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Thread {
    pub name: String,
    pub created_ms: u64,
    /// The feature is done (`thread_archive`); attaching to it again reopens it.
    #[serde(default)]
    pub archived: bool,
    /// When Claude last attached to it, to the nearest [`ATTACH_MARK_MS`]:
    /// activity, to the sweep that archives a thread left a day idle.
    #[serde(default)]
    pub attached_ms: u64,
}

/// How stale a thread's [`Thread::attached_ms`] may grow before an attach
/// writes it again: the record replicates, so not every attach is kept.
pub const ATTACH_MARK_MS: u64 = 60 * 60 * 1000;

/// One entry of a thread's journal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Note {
    pub at_ms: u64,
    /// `Claude started:` or `Task ended:` — who the line is from, said in the
    /// line, since the thread holds no model turns to cite.
    pub label: String,
    pub text: String,
    /// The worker the entry is about, as a path.
    #[serde(default)]
    pub worker: Option<String>,
}

pub fn thread_key(owner: &str, thread: &str) -> String {
    format!("mcp/thread/{owner}/{thread}")
}

pub fn worker_prefix(owner: &str, thread: &str) -> String {
    format!("mcp/worker/{owner}/{thread}/")
}

pub fn worker_key(owner: &str, thread: &str, name: &str) -> String {
    format!("{}{name}", worker_prefix(owner, thread))
}

fn note_prefix(owner: &str, thread: &str) -> String {
    format!("mcp/note/{owner}/{thread}/")
}

/// The journal of the project `repo`, `P:<repo>`: the standing rules every
/// task in every thread inherits, kept like a thread's notes under the name
/// — which no thread can take, since a thread's name holds no `:`.
pub fn project_journal(repo: &str) -> String {
    format!("P:{repo}")
}

/// The zend conversation a worker's task runs in. Owner, thread and name are
/// all in it, so two people's same-named workers are two conversations.
pub fn conv_id(owner: &str, thread: &str, name: &str) -> String {
    format!("mcp:{owner}:{thread}/{name}")
}

/// The owner, thread and lineage name a worker's conversation id names —
/// [`conv_id`] read back; `None` for any other conversation.
pub fn conv_names(conv_id: &str) -> Option<(&str, &str, &str)> {
    let (owner, rest) = conv_id.strip_prefix("mcp:")?.split_once(':')?;
    let (thread, name) = rest.split_once('/')?;
    (!owner.is_empty() && !thread.is_empty() && !name.is_empty()).then_some((owner, thread, name))
}

fn decode<T: for<'de> Deserialize<'de>>(value: Value) -> Option<T> {
    serde_json::from_value(value).ok()
}

pub fn thread(records: &dyn Records, owner: &str, name: &str) -> Option<Thread> {
    records.get(&thread_key(owner, name)).and_then(decode)
}

/// The thread `name`, created when it does not exist yet, and reopened when
/// it was archived; its attach marked, when the mark is stale.
pub fn attach(
    records: &dyn Records,
    owner: &str,
    name: &str,
    now_ms: u64,
) -> anyhow::Result<Thread> {
    let t = match thread(records, owner, name) {
        Some(t) if !t.archived && now_ms.saturating_sub(t.attached_ms) < ATTACH_MARK_MS => {
            return Ok(t)
        }
        Some(t) => Thread {
            archived: false,
            attached_ms: now_ms,
            ..t
        },
        None => Thread {
            name: name.to_string(),
            created_ms: now_ms,
            archived: false,
            attached_ms: now_ms,
        },
    };
    put_thread(records, owner, &t)?;
    Ok(t)
}

pub fn put_thread(records: &dyn Records, owner: &str, t: &Thread) -> anyhow::Result<()> {
    records.put(&thread_key(owner, &t.name), serde_json::to_value(t)?)
}

/// Every thread the owner holds, newest first.
pub fn threads(records: &dyn Records, owner: &str) -> Vec<Thread> {
    let mut all: Vec<Thread> = records
        .under(&format!("mcp/thread/{owner}/"))
        .into_iter()
        .filter_map(|(_, v)| decode(v))
        .collect();
    all.sort_by_key(|t| Reverse(t.created_ms));
    all
}

/// Every thread of every owner, each with the owner key it is kept under.
pub fn every_thread(records: &dyn Records) -> Vec<(String, Thread)> {
    records
        .under("mcp/thread/")
        .into_iter()
        .filter_map(|(key, v)| {
            let (owner, _) = key.strip_prefix("mcp/thread/")?.split_once('/')?;
            Some((owner.to_string(), decode(v)?))
        })
        .collect()
}

pub fn worker(records: &dyn Records, owner: &str, thread: &str, name: &str) -> Option<Worker> {
    records
        .get(&worker_key(owner, thread, name))
        .and_then(decode)
}

pub fn workers(records: &dyn Records, owner: &str, thread: &str) -> Vec<Worker> {
    records
        .under(&worker_prefix(owner, thread))
        .into_iter()
        .filter_map(|(_, v)| decode(v))
        .collect()
}

pub fn put_worker(records: &dyn Records, owner: &str, w: &Worker) -> anyhow::Result<()> {
    records.put(
        &worker_key(owner, &w.thread, &w.name),
        serde_json::to_value(w)?,
    )
}

/// Add `note` to the thread's journal.
pub fn add_note(
    records: &dyn Records,
    owner: &str,
    thread: &str,
    note: &Note,
) -> anyhow::Result<()> {
    // Time first, zero-padded, so key order is the order they were written;
    // the sequence after it keeps two notes of one millisecond in order.
    let seq = NOTE_SEQ.fetch_add(1, Ordering::Relaxed);
    let key = format!("{}{:016}-{seq:010}", note_prefix(owner, thread), note.at_ms);
    records.put(&key, serde_json::to_value(note)?)
}

/// The thread's journal, oldest first.
pub fn notes(records: &dyn Records, owner: &str, thread: &str) -> Vec<Note> {
    records
        .under(&note_prefix(owner, thread))
        .into_iter()
        .filter_map(|(_, v)| decode(v))
        .collect()
}

/// A thread's or project's journal, one numbered line per entry, oldest first.
pub fn note_lines(notes: &[Note]) -> Vec<String> {
    notes
        .iter()
        .enumerate()
        .map(|(i, n)| format!("#{} {} {}", i + 1, n.label, n.text))
        .collect()
}

/// A thread's or project's journal as `journal_read` prints it: one numbered
/// line per entry, under its path.
pub fn render_journal(path: &str, notes: &[Note]) -> String {
    let mut out = format!("Journal of {path}, written by Claude");
    if notes.is_empty() {
        out.push_str(" — no entries yet.");
        return out;
    }
    for line in note_lines(notes) {
        out.push('\n');
        out.push_str(&line);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::records::MemRecords;

    #[test]
    fn attaching_to_an_archived_thread_reopens_it_and_keeps_its_birth() {
        let r = MemRecords::default();
        let t = attach(&r, "o", "review", 10).unwrap();
        put_thread(
            &r,
            "o",
            &Thread {
                archived: true,
                ..t
            },
        )
        .unwrap();
        let back = attach(&r, "o", "review", 99).unwrap();
        assert!(!back.archived);
        assert_eq!((back.created_ms, back.attached_ms), (10, 99));
        assert!(!thread(&r, "o", "review").unwrap().archived);
    }

    /// **An attach is activity**: marked when the mark is an hour stale, so
    /// a thread Claude came back to is not swept as idle under it.
    #[test]
    fn an_attach_marks_the_thread_at_most_hourly() {
        let r = MemRecords::default();
        attach(&r, "o", "review", 10).unwrap();
        attach(&r, "o", "review", 10 + ATTACH_MARK_MS - 1).unwrap();
        assert_eq!(thread(&r, "o", "review").unwrap().attached_ms, 10);
        attach(&r, "o", "review", 10 + ATTACH_MARK_MS).unwrap();
        assert_eq!(
            thread(&r, "o", "review").unwrap().attached_ms,
            10 + ATTACH_MARK_MS
        );
    }

    #[test]
    fn attaching_twice_is_one_thread_and_owners_never_meet() {
        let r = MemRecords::default();
        let a = attach(&r, "o1", "review", 10).unwrap();
        assert_eq!(attach(&r, "o1", "review", 20).unwrap(), a);
        attach(&r, "o2", "review", 30).unwrap();
        assert_eq!(threads(&r, "o1").len(), 1);
        assert_eq!(thread(&r, "o2", "review").unwrap().created_ms, 30);
    }

    #[test]
    fn notes_read_back_in_the_order_they_were_written() {
        let r = MemRecords::default();
        let note = |at_ms, label: &str, text: &str| Note {
            at_ms,
            label: label.into(),
            text: text.into(),
            worker: Some("T:review/verify".into()),
        };
        add_note(&r, "o", "review", &note(5, "Claude started:", "verify")).unwrap();
        add_note(&r, "o", "review", &note(5, "Task ended:", "verify — ok")).unwrap();
        add_note(&r, "o", "review", &note(9, "Claude started:", "check")).unwrap();
        let got = notes(&r, "o", "review");
        assert_eq!(got.len(), 3);
        assert_eq!(
            render_journal("T:review", &got),
            "Journal of T:review, written by Claude\n\
             #1 Claude started: verify\n#2 Task ended: verify — ok\n#3 Claude started: check"
        );
        assert_eq!(
            render_journal("P:candle", &[]),
            "Journal of P:candle, written by Claude — no entries yet."
        );
    }

    #[test]
    fn a_running_task_whose_beat_stopped_reads_as_cut_off() {
        let w = Worker {
            thread: "t".into(),
            name: "n".into(),
            conv_id: "c".into(),
            prompt: "p".into(),
            state: State::Running,
            role: Role::User,
            priority: Priority::Normal,
            effort: Effort::Deep,
            started_ms: 0,
            turn_ms: 0,
            beat_ms: 1_000,
            ended_ms: None,
            entries_after: None,
            entries: Vec::new(),
            final_text: String::new(),
            error: None,
            question: None,
            activity: None,
            no_entry: None,
            stopped: false,
            resumable: false,
            audit: None,
        };
        assert_eq!(
            w.clone().observed(1_000 + BEAT_LOST_MS).state,
            State::Running
        );
        let lost = w.clone().observed(1_001 + BEAT_LOST_MS);
        assert_eq!(lost.state, State::Error);
        assert!(lost.error.unwrap().contains("run the task again"));
        let done = Worker {
            state: State::Done,
            ..w
        };
        assert_eq!(done.observed(u64::MAX).state, State::Done);
    }

    #[test]
    fn a_worker_waiting_on_claude_is_live_but_not_working() {
        for s in [State::Question, State::Stuck] {
            assert!(s.live() && !s.working(), "{s:?}");
        }
        assert!(State::Running.live() && State::Running.working());
        for s in [State::Done, State::Error] {
            assert!(!s.live() && !s.working(), "{s:?}");
        }
    }

    #[test]
    fn a_task_decodes_normal_unless_asked_high() {
        assert_eq!(Priority::of(None).unwrap().band(), DecodePriority::Normal);
        assert_eq!(Priority::of(Some("normal")).unwrap(), Priority::Normal);
        assert_eq!(
            Priority::of(Some("high")).unwrap().band(),
            DecodePriority::High
        );
        assert_eq!(
            Priority::of(Some("low")).unwrap_err(),
            "`priority` is `normal` or `high`, not `low`"
        );
    }

    /// **A task thinks deep unless asked**, at the effort dial's own levels,
    /// and its effort is no part of how soon it decodes.
    #[test]
    fn a_task_thinks_deep_unless_asked() {
        assert_eq!(Effort::of(None).unwrap(), Effort::Deep);
        assert_eq!(
            ["balanced", "deep", "exhaustive"].map(|e| Effort::of(Some(e)).unwrap().level()),
            [2, 3, 4]
        );
        assert_eq!(
            Effort::of(Some("quick")).unwrap_err(),
            "`effort` is `balanced`, `deep` or `exhaustive`, not `quick`"
        );
        let kept: Worker = serde_json::from_value(serde_json::json!({
            "thread": "t", "name": "n", "conv_id": "c", "prompt": "p",
            "state": "running", "role": "user", "priority": "normal",
            "started_ms": 0, "turn_ms": 0
        }))
        .unwrap();
        assert_eq!(kept.effort, Effort::Deep);
    }

    #[test]
    fn a_workers_conversation_names_its_owner_thread_and_name() {
        assert_eq!(
            conv_id("abc", "review", "verify.2"),
            "mcp:abc:review/verify.2"
        );
        assert_eq!(
            conv_id("abc", "review", "verify/read"),
            "mcp:abc:review/verify/read"
        );
        assert_eq!(
            conv_names("mcp:abc:review/verify/read"),
            Some(("abc", "review", "verify/read"))
        );
        for other in [
            "c-123",
            "mcp:abc:review",
            "mcp::t/n",
            "mcp:abc:/n",
            "mcp:abc:t/",
        ] {
            assert_eq!(conv_names(other), None, "{other}");
        }
    }

    /// A record reads back as written, and a part is told from a task by its
    /// lineage name.
    #[test]
    fn a_part_is_known_by_its_name() {
        let kept = serde_json::json!({
            "thread": "t", "name": "verify/read", "conv_id": "c", "prompt": "p",
            "state": "running", "role": "admin", "priority": "high",
            "started_ms": 0, "turn_ms": 5, "beat_ms": 5
        });
        let w: Worker = serde_json::from_value(kept).unwrap();
        assert_eq!(
            (w.role, w.priority, w.turn_ms),
            (Role::Admin, Priority::High, 5)
        );
        assert!(w.is_part());
        assert_eq!(w.path(), "T:t/verify/read");
        let task = Worker {
            name: "verify".into(),
            ..w
        };
        assert!(!task.is_part());
    }
}
