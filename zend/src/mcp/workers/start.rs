//! Starts a worker on a new conversation, named after the first free name under
//! its thread.
//!
//! A name is taken by its record's key, whether or not the record reads, so a name
//! is never handed out again for a new conversation. A worker started from one of
//! Zen's conversations begins on that conversation's branches and uncommitted
//! changes. It is attached under the archiving lock, so a sweep cannot archive the
//! thread between the attach and the worker. Its answer ends in a `Steps` section
//! written by the grammar. A task's prompt closes by telling it how to ask Claude,
//! and a chat's closes with nothing, since Zen answers Claude in the chat.

use super::*;

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
pub(super) fn taken(
    records: &dyn Records,
    owner: &str,
    thread: &str,
    now: u64,
) -> Vec<(String, bool)> {
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
pub(in super::super) fn fresh(
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
        handed_ms: None,
        by_task: false,
    }
}

/// What a turn's message ends with: for a task, how to ask Claude; for a chat,
/// nothing — Zen answers Claude in the chat itself.
pub(super) fn closing_of(kind: WorkerKind) -> &'static str {
    if kind.is_chat() {
        ""
    } else {
        ASKING
    }
}
