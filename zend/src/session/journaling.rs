//! Connects a conversation to its journal: the sections its prompt carries, the
//! drafts and folds that write its entries, and the journal reads.
//!
//! A round reads the journal as it last sealed, and a new entry joins the
//! prompt the first round after its section seals, so no round waits on a
//! section it did not submit. Only the first reconcile of a conversation waits
//! briefly for its journal to seal. Drafts run as tasks of their own, one at a
//! time, and a finished request's entry is never skipped: it starts when the
//! running draft ends. A fold keeps its entries' text until its record is
//! written, so a failed fold leaves its evidence readable.

use super::*;

/// The journal `timeline`'s metadata records, if it records one.
pub(super) fn journal_record(state: &InferenceState, timeline: TimelineId) -> Option<Kept> {
    kept_in(
        state
            .engine
            .conversation_metadata_value(timeline, JOURNAL_KEY)
            .as_deref(),
    )
}

/// Bring the conversation's journal sections in line with its journal, as far
/// as `changes` lets them change, and pin the ones that have sealed in
/// `selection`.
///
/// Nothing waits for a section to seal: a round reads the journal as it last
/// sealed, and a new entry joins the prompt the first round after its section
/// seals (`journal::carry`). Called at each round's head, and by a draft as it
/// ends, so a new entry is already sealing before the next round begins.
/// While `in_request` — a request's tool rounds — it is carried
/// [`Changes::Growing`] from its steps and close ([`growing_end`]): entries
/// join the chain's end, the steps and the open items change as the request
/// goes, and a fold, a forgotten or edited entry or the oldest leaving waits
/// for the next request (`docs/placed_section_layout.md` §3.7).
pub(super) async fn carry_journal(
    state: &InferenceState,
    conv_id: &str,
    timeline: TimelineId,
    cs: &mut ConvState,
    selection: &mut SelectionState,
    in_request: bool,
) {
    let t_open = Instant::now();
    state.prewarm.note(conv_id, t_open);
    let journal = state
        .journals
        .of(conv_id, || journal_record(state, timeline));
    let opened = open_journal(state, timeline, &journal);
    // A journal this process has not seen a round of yet — after a restart, a
    // takeover or a resume — reads the request's steps from its turns, once.
    if !journal.lock().unwrap().steps().known() {
        let request = state
            .engine
            .request_replies(timeline)
            .map(|(from, replies)| (from.0, replies.iter().map(|r| own_notes(r)).collect()));
        journal.lock().unwrap().steps_mut().rebuild(request);
    }
    let open_ms = t_open.elapsed().as_millis() as u64;
    let t_sections = Instant::now();
    let (prompt, sections, sharing) = journal_sections(state, conv_id, &journal);
    let sections_ms = t_sections.elapsed().as_millis() as u64;
    let wants: Vec<Want<'_>> = sections
        .iter()
        .zip(&sharing)
        .map(|(s, &sharing)| section_want(s, sharing))
        .collect();
    let changes = match in_request {
        true => Changes::Growing {
            end: growing_end(&sections),
        },
        false => Changes::Any,
    };
    let lineage_sweep = lineage::sweep_names();
    let memory_sweep = memory_sections::sweep_names();
    let mut sweep = prompt.sweep();
    if owner_of(conv_id).is_some() {
        sweep.extend(lineage_sweep.iter().map(String::as_str));
        sweep.extend(memory_sweep.iter().map(String::as_str));
    }
    let ConvState {
        conv,
        journal: carried,
        ..
    } = cs;
    // A conversation whose state was just built — its first turn, or one
    // taken up again after its state was let go — holds no section yet, and
    // this reconcile submits the whole journal.
    let fresh = carried.holds_nothing();
    let t_reconcile = Instant::now();
    let mut sealed = carried
        .reconcile(
            conv,
            JOURNAL,
            &wants,
            Linking::Chain,
            changes,
            &sweep,
            None,
            tokio::time::sleep,
        )
        .await;
    // A round whose journal was all just submitted would otherwise answer
    // with none: a one-round turn taken up after its state was let go ("who
    // are you?") had no journal at all, and the opening is what a first
    // round answers its opening question from. So it gives the journal a
    // moment to seal. Everything is submitted by now: this reconcile only
    // waits. Later rounds take new entries as they seal, without waiting.
    let reconcile_ms = t_reconcile.elapsed().as_millis() as u64;
    let t_wait = Instant::now();
    if (opened || fresh) && sealed.len() < wants.len() {
        sealed = carried
            .reconcile(
                conv,
                JOURNAL,
                &wants,
                Linking::Chain,
                changes,
                &sweep,
                Some(OPENING_SEAL_WAIT),
                tokio::time::sleep,
            )
            .await;
    }
    journal.lock().unwrap().journal_mut().swept(&prompt);
    carry::select(selection, &sealed);
    // Where a round's journal time went: opening the journal and its steps,
    // building its sections (ancestors' journals, the memory, its own),
    // submitting them, and a first round's wait for them to seal.
    tracing::info!(
        conv_id = %conv_id,
        open_ms,
        sections_ms,
        sections = wants.len(),
        reconcile_ms,
        seal_wait_ms = t_wait.elapsed().as_millis() as u64,
        sealed = sealed.len(),
        "journal carry"
    );
}

/// Where the journal's growing end begins in `sections`: its current
/// request's heading and steps, or its close, whichever comes first — what a
/// request changes as it goes. Everything ahead of it — the ancestors'
/// journals, the memory, the entries — holds while a request runs.
pub(super) fn growing_end(sections: &[Section]) -> usize {
    sections
        .iter()
        .position(|s| s.name == CURRENT_SECTION || s.name == CLOSE_SECTION)
        .unwrap_or(sections.len())
}

/// The journal sections the conversation `conv_id` carries, in the order its
/// collection holds them, beside its journal's prompt: what `carry_journal`
/// reconciles, and what the projection panel shows as their text.
pub(super) fn journal_sections(
    state: &InferenceState,
    conv_id: &str,
    journal: &Mutex<SessionJournal>,
) -> (JournalPrompt, Vec<Section>, Vec<Sharing>) {
    let (prompt, own_shareable) = {
        let held = journal.lock().unwrap();
        let prompt = held
            .journal()
            .prompt_with_steps(&held.steps().steps(), held.style());
        let shareable: Vec<bool> = prompt
            .sections
            .iter()
            .map(|s| journal_section_is_shareable(&s.name, held.journal()))
            .collect();
        (prompt, shareable)
    };
    // A conversation owned through `/mcp` carries its ancestors' journals
    // ahead of its own, and the reader's line after it (`mcp::lineage`).
    let mut inherited = inherited_journals(state, conv_id);
    let reader = inherited.pop();
    // Then the memory its thread is about, before its own journal
    // (`mcp::memory_sections`).
    let memory = carried_memory(state, conv_id);
    let as_section = |i: lineage::Inherited| Section {
        name: i.name,
        text: i.text,
    };
    // The ancestors' journals and the memory are the thread's; the reader's
    // line is the conversation's (`docs/shared_journal_sections.md` §2).
    let mut shareable: Vec<bool> = vec![true; inherited.len() + memory.len()];
    shareable.extend(own_shareable);
    shareable.extend(reader.iter().map(|_| false));
    let mut sections: Vec<Section> = inherited.into_iter().map(as_section).collect();
    sections.extend(memory.into_iter().map(as_section));
    sections.extend(prompt.sections.iter().cloned());
    sections.extend(reader.map(as_section));
    (prompt, sections, chain_sharing(&shareable, false))
}

/// Whether the journal section `name` of `journal` says nothing of the
/// conversation's own: the entries' heading, an entry of the opening, and the
/// close while nothing is open. A drafted entry, the request's steps and an
/// open item are the conversation's.
pub(super) fn journal_section_is_shareable(name: &str, journal: &Journal) -> bool {
    if name == HISTORY_SECTION {
        return true;
    }
    if name == CLOSE_SECTION {
        return journal.open().is_empty();
    }
    journal
        .entries()
        .find(|e| entry_section(e.id) == name)
        .is_some_and(opening::is_opening)
}

/// Open the journal of the conversation at `timeline` with what it read
/// before its session began (`journal::opening`): the two entries, kept and
/// recorded, for a conversation that descends from the priming chain and
/// whose journal has never kept an entry. A forgotten opening is not written
/// again. Whether it wrote them.
pub(super) fn open_journal(
    state: &InferenceState,
    timeline: TimelineId,
    journal: &Mutex<SessionJournal>,
) -> bool {
    if journal.lock().unwrap().journal().written() > 0 {
        return false;
    }
    let seeded = state
        .engine
        .conversation_metadata_value(timeline, Substrate::FORKED_FROM_KEY)
        .is_some_and(|parent| parent == Substrate::SEED_PARENT);
    let entries = state.branch_ingest.seed_read().entries(now_ms());
    if !seeded || entries.is_empty() {
        return false;
    }
    let prompt = {
        let mut held = journal.lock().unwrap();
        // Asked again under the lock: a draft kept since the check above is
        // the conversation's own journal begun, and the opening comes before
        // its first entry or not at all.
        if held.journal().written() > 0 {
            return false;
        }
        for entry in entries {
            held.journal_mut().keep_closing(entry);
        }
        held.journal().prompt(held.style())
    };
    if let Err(e) = state
        .engine
        .set_conversation_metadata_many(timeline, &metadata_of(&prompt))
    {
        tracing::warn!(
            timeline = timeline.raw(),
            "the journal's opening could not be recorded: {e}"
        );
    }
    true
}

/// A section of the journal as the reconcile wants it.
pub(super) fn section_want(s: &Section, sharing: Sharing) -> Want<'_> {
    Want {
        name: &s.name,
        text: &s.text,
        sharing,
    }
}

/// The journals `conv_id` inherits, as the sections that carry them — none for
/// a conversation not owned through `/mcp`: its thread's and its projects'
/// (`docs/claude_zend_gateway.md` §5.1).
pub(super) fn inherited_journals(state: &InferenceState, conv_id: &str) -> Vec<lineage::Inherited> {
    let Some((_, thread, name)) = conv_names(conv_id) else {
        return Vec::new();
    };
    let nearest_first: Vec<lineage::Ancestor> = journals_above(state, conv_id)
        .into_iter()
        .map(|a| a.ancestor)
        .collect();
    lineage::sections(&format!("T:{thread}/{name}"), &nearest_first)
}

/// The memory sections `conv_id` carries: the topics its thread is about that
/// its reader still reads, from the memory's tree as last published — none
/// for a conversation not owned through `/mcp`, a thread about nothing, or a
/// memory not open.
pub(super) fn carried_memory(state: &InferenceState, conv_id: &str) -> Vec<lineage::Inherited> {
    let Some((owner, thread, _)) = conv_names(conv_id) else {
        return Vec::new();
    };
    let records = state.engine.conversation().shared();
    let Some(bound) = thread_export::bound(&records, owner, thread) else {
        return Vec::new();
    };
    let Ok(tree) = state.memory.tree() else {
        return Vec::new();
    };
    let about: Vec<JournalPath> = bound
        .about
        .iter()
        .filter_map(|p| path_from_wire(p).ok())
        .collect();
    let reader = state.retrieval_scope.reader_of(conv_id);
    memory_sections::sections(&memory_sections::topics(
        &tree,
        &about,
        &reader,
        Stamp::now(),
    ))
}

/// One journal above a conversation: the ancestor it is and the bare name it
/// answers to.
pub(super) struct JournalAbove {
    pub(super) ancestor: lineage::Ancestor,
    pub(super) name: String,
    /// The entries a citation may name: its notes' numbers.
    pub(super) entries: Vec<u64>,
}

/// The journals above `conv_id`, nearest first — none for a conversation not
/// owned through `/mcp`: its thread's notes, then each project's standing
/// rules.
pub(super) fn journals_above(state: &InferenceState, conv_id: &str) -> Vec<JournalAbove> {
    let Some((owner, thread, _)) = conv_names(conv_id) else {
        return Vec::new();
    };
    let records = state.engine.conversation().shared();
    let mut nearest_first = Vec::new();
    let notes = |journal: &str| -> (Vec<String>, Vec<u64>) {
        let entries = threads::notes(&records, owner, journal);
        (
            threads::note_lines(&entries),
            entries.iter().map(|e| e.number).collect(),
        )
    };
    let (lines, entries) = notes(thread);
    nearest_first.push(JournalAbove {
        entries,
        ancestor: lineage::Ancestor {
            path: format!("T:{thread}"),
            whose: lineage::Whose::Thread,
            lines,
        },
        name: thread.to_string(),
    });
    for repo in state
        .workspace
        .repos()
        .iter()
        .filter(|r| r.name != UPLOADS_REPO)
    {
        let project = project_journal(&repo.name);
        let (lines, entries) = notes(&project);
        nearest_first.push(JournalAbove {
            entries,
            ancestor: lineage::Ancestor {
                lines,
                path: project,
                whose: lineage::Whose::Project,
            },
            name: repo.name.clone(),
        });
    }
    nearest_first
}

/// The journals above `conv_id` as its claims may cite them.
pub(super) fn citable_above(state: &InferenceState, conv_id: &str) -> Vec<Citable> {
    journals_above(state, conv_id)
        .into_iter()
        .map(|a| Citable {
            path: a.ancestor.path,
            entries: a.entries,
        })
        .collect()
}

/// `conv_id`'s journal — one line per entry and one per open item — and its
/// entries' ids.
pub(super) fn journal_of(state: &InferenceState, conv_id: &str) -> (Vec<String>, Vec<u64>) {
    let journal = state.journals.of(conv_id, || {
        journal_record(state, conversation_timeline(conv_id))
    });
    let held = journal.lock().unwrap();
    let mut lines: Vec<String> = held
        .journal()
        .entries()
        .map(|e| format!("#{} {}", e.id, e.summary))
        .collect();
    let ids = held.journal().entries().map(|e| e.id).collect();
    lines.extend(
        held.journal()
            .open()
            .iter()
            .map(|item| format!("Still open #{}: {}", item.id, item.text)),
    );
    (lines, ids)
}

/// When a journal draft starts.
#[derive(Clone, Copy)]
pub(super) enum Cadence<'a> {
    /// Turn `newest` of `request` has sealed, and its round's projection
    /// selected `selected` of the conversation's turns: start one if the
    /// request finished, or if the cadence or the selection pressure asks for
    /// one within it (`journal::state`).
    Due {
        newest: u32,
        selected: Option<&'a [u32]>,
        request: Request,
    },
    /// Start one now over whatever is not covered: on `POST …/journal/draft`,
    /// which may skip, or for the finished request a running draft held back,
    /// which may not.
    Now { skip: Skip },
}

/// Why no journal draft started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoDraft {
    /// The conversation is not open on this member, so there is no live slot
    /// to fork the draft's questions from.
    NotOpen,
    /// The cadence does not ask for one yet.
    NotDue,
    /// A draft is running: it may not cover the latest turns, so ask again
    /// once it ends.
    Running,
    /// Nothing has happened since the journal last looked.
    NothingNew,
}

/// Start a journal draft of `conv_id` if `cadence` calls for one, and return
/// the stretch it covers. The draft runs as a task of its own.
pub(super) fn begin_journal_draft(
    state: &Arc<InferenceState>,
    conv_id: &str,
    timeline: TimelineId,
    cadence: Cadence<'_>,
) -> Result<Span, NoDraft> {
    let open = state.journals.open(conv_id).ok_or(NoDraft::NotOpen)?;
    let journal = state
        .journals
        .of(conv_id, || journal_record(state, timeline));
    let (covered, skip) = {
        let mut held = journal.lock().unwrap();
        let skip = match cadence {
            Cadence::Due {
                newest,
                selected,
                request,
            } => {
                let pressed = selected
                    .is_some_and(|s| pressure::left_out(s, held.first_uncovered_turn(), newest));
                let Some(due) = held.due(newest, pressed, request) else {
                    return Err(NoDraft::NotDue);
                };
                // A finished request's entry is never lost to the single
                // flight: it starts when the running draft ends.
                if held.journal().pending() && request.finished {
                    held.owe();
                }
                due.skip
            }
            Cadence::Now { skip } => skip,
        };
        if held.journal().pending() {
            return Err(NoDraft::Running);
        }
        (held.journal().covered_to(), skip)
    };
    // Only the turns not yet covered: the covered part's own turn is read
    // again, since its reply half may not be covered yet.
    let from = origin::turn_of(covered).unwrap_or(0);
    let history = state
        .base_conv
        .recovered_history_from(timeline, false, from);
    let newest = match cadence {
        Cadence::Due { newest, .. } => newest,
        Cadence::Now { .. } => history
            .iter()
            .map(|m| m.turn)
            .max()
            .ok_or(NoDraft::NothingNew)?,
    };
    let written = state.engine.timeline_turn_wall_ms(timeline);
    let parts = origin::parts_after(&history, &written, covered);
    let span = {
        // Under the same lock as the begin: a draft that started since the
        // check above is running, not nothing new.
        let mut held = journal.lock().unwrap();
        if held.journal().pending() {
            return Err(NoDraft::Running);
        }
        held.begin(parts, newest).ok_or(NoDraft::NothingNew)?
    };
    spawn_journal_draft(
        Arc::clone(state),
        conv_id.to_string(),
        timeline,
        open,
        journal,
        span.clone(),
        skip,
    );
    Ok(span)
}

/// The desk a draft of `timeline`'s journal puts its questions through: forks
/// of the conversation's slot, each answer held to one call of the offered ones
/// in the dialect's own shape, the reasoning block written closed, decoded
/// greedily.
pub(super) fn journal_desk(
    state: &Arc<InferenceState>,
    conv_id: &str,
    timeline: TimelineId,
    asker: UnsealedAsker,
    max_tokens: usize,
) -> ZendDesk {
    // A worker's conversation — the one `style_for` and the journals above
    // it read — is the one whose entries are cited by address.
    let folds_keep_turns = conv_names(conv_id).is_some();
    let dialect = state.refresh_config.dialect.clone();
    let env = ToolCallEnvelope::for_assistant_calls(&dialect);
    let close = ToolCallEnvelope::turn_close(&dialect);
    let prefill = format!(
        "{}{}",
        dialect.no_think_block,
        ToolCallEnvelope::for_dialect(&dialect).marker
    );
    let compiling = Arc::clone(state);
    let compile: Compile = Arc::new(move |specs: &[Value]| {
        let specs = specs
            .iter()
            .map(|spec| serde_json::from_value(spec.clone()))
            .collect::<Result<Vec<ToolSpec>, _>>()?;
        let tree = compile_action_loop(&specs, &env, 1, &close, None)?;
        let compiled = compiling.engine.compile_stencil(&tree)?;
        Ok(Arc::new(compiled))
    });
    let mut sampling = state.refresh_config.sampling.clone();
    sampling.temperature = 0.0;
    let recording = Arc::clone(state);
    let record: Record = Box::new(
        move |prompt: &JournalPrompt, filed: Option<&Filed>, ceased: &[u64]| {
            let conversation = recording.engine.conversation();
            let shared = conversation.shared();
            let text = JournalText::of(shared, timeline);
            if let Some(filed) = filed {
                text.file(filed)?;
            }
            conversation
                .set_conversation_metadata_many(timeline, &metadata_of(prompt))
                .map_err(|e| anyhow::anyhow!("the journal could not be recorded: {e}"))?;
            // Only once the record no longer holds them: a fold whose record
            // fails is abandoned with its entries still in the journal, and
            // their text must still be there to read. A conversation owned
            // through `/mcp` files a fold's entries' turns under the fold, so
            // what its readers cite keeps its evidence
            // (`docs/claude_zend_gateway.md` §5.2).
            match filed {
                Some(fold) if folds_keep_turns && !ceased.is_empty() => {
                    text.refile(ceased, fold.entry)?;
                }
                _ => {
                    for entry in ceased {
                        text.cease_entry(*entry)?;
                    }
                }
            }
            Ok(())
        },
    );
    ZendDesk::new(
        Asking {
            asker,
            compile,
            prefill,
            sampling,
            max_tokens,
        },
        record,
    )
}

/// Cease the text behind `timeline`'s journal entries — the conversation was
/// tombstoned or archived, and nothing reads it again.
pub(super) fn cease_journal_text(
    engine: &ConversationEngine,
    timeline: TimelineId,
) -> Result<(), ConversationError> {
    JournalText::of(engine.conversation().shared(), timeline)
        .cease_all()
        .map(|_| ())
        .map_err(|e| ConversationError::Other(e.to_string()))
}

/// `timeline`'s journal text: the turns behind its entries.
pub(super) fn journal_text(state: &InferenceState, timeline: TimelineId) -> JournalText {
    let shared = state.engine.conversation().shared();
    JournalText::of(shared, timeline)
}

/// The journals `conv_id`'s journal tools read: its own, and every journal
/// above it (`docs/claude_zend_gateway.md` §5.3) — read when a tool first
/// opens one, so a round that calls none reads nothing for it.
pub(super) fn journal_shelf(
    state: &Arc<InferenceState>,
    conv_id: &str,
    timeline: TimelineId,
) -> Arc<dyn JournalShelf> {
    let state = Arc::clone(state);
    let conv_id = conv_id.to_string();
    Arc::new(Lazy::new(move || shelf_of(&state, &conv_id, timeline)))
}

/// `conv_id`'s journals as they stand now.
pub(super) fn shelf_of(state: &InferenceState, conv_id: &str, timeline: TimelineId) -> Shelf {
    let own = Drafted::new(journal_of(state, conv_id).0, journal_text(state, timeline));
    let above = journals_above(state, conv_id)
        .into_iter()
        .map(|a| Above {
            path: a.ancestor.path,
            name: a.name,
            book: Arc::new(Notes::new(a.ancestor.lines)) as Arc<dyn JournalBook>,
        })
        .collect();
    Shelf::new(Arc::new(own), above)
}

/// Run one journal draft over `span` as a task of its own, and log how it
/// ended. Its record is the journal's too, for the API.
pub(super) fn spawn_journal_draft(
    state: Arc<InferenceState>,
    conv_id: String,
    timeline: TimelineId,
    open: Open,
    journal: Arc<Mutex<SessionJournal>>,
    span: Span,
    skip: Skip,
) {
    tokio::spawn(async move {
        let mut desk = journal_desk(&state, &conv_id, timeline, open.asker, WRITE_TOKENS);
        let world = JournalsAbove(citable_above(&state, &conv_id));
        // Nobody reads a deleted conversation's journal: its draft is dropped
        // where it stands, and its forks of the slot are freed with it.
        let drafting = draft::draft(&mut desk, &journal, &span, &world, skip);
        let engine = &state.engine;
        let Some(record) = state
            .deletions
            .unless_deleted(drafting, || engine.is_timeline_tombstoned(timeline))
            .await
        else {
            // The delete dropped the journal too (`journals.forget`): nothing
            // reads the draft it was running.
            tracing::info!(
                target: "zend::journal",
                conv_id = %conv_id,
                from_part = span.from_turn,
                to_part = span.to_turn,
                "journal draft stopped: the conversation was deleted",
            );
            return;
        };
        tracing::info!(
            target: "zend::journal",
            conv_id = %conv_id,
            from_part = record.from_turn,
            to_part = record.to_turn,
            result = ?record.result,
            entry = ?record.entry,
            rounds = record.rounds,
            took_ms = record.took_ms,
            write_ms = record.write_ms,
            detail = ?record.detail,
            checks = ?record.checks,
            "journal draft ended",
        );
        reconcile_journal_now(&state, &conv_id, timeline).await;
        // A request that finished while this draft ran is owed its entry: it
        // starts now, over what is still uncovered, and may not skip.
        let owed = journal.lock().unwrap().take_owed();
        if owed {
            match begin_journal_draft(
                &state,
                &conv_id,
                timeline,
                Cadence::Now {
                    skip: Skip::Refused,
                },
            ) {
                Ok(span) => tracing::info!(
                    target: "zend::journal",
                    conv_id = %conv_id,
                    from_part = span.from_turn,
                    to_part = span.to_turn,
                    parts = span.turns.len(),
                    "journal draft started for a request that finished while one ran",
                ),
                Err(why) => tracing::debug!(
                    target: "zend::journal",
                    conv_id = %conv_id,
                    ?why,
                    "no draft for the owed request",
                ),
            }
            return;
        }
        // A draft held the journal; a period that ended meanwhile folds now.
        begin_journal_folds(&state, &conv_id, timeline);
    });
}

/// Hand `conv_id`'s journal sections to its conversation now rather than at
/// the next round's head, so they are sealing — a chain one link after another
/// — while the user is still reading the reply, and the next round reads them.
/// A conversation busy in a round is left to that round's own reconcile, which
/// does the same.
pub(super) async fn reconcile_journal_now(
    state: &Arc<InferenceState>,
    conv_id: &str,
    timeline: TimelineId,
) {
    let held = state.conversations.get(conv_id);
    if let Some(conversation) = held {
        if let Ok(mut cs) = conversation.try_lock() {
            let mut unused = SelectionState::default();
            carry_journal(state, conv_id, timeline, &mut cs, &mut unused, false).await;
        }
    }
}

/// Milliseconds since the Unix epoch, now.
pub(super) fn wall_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Start folding `conv_id`'s journal if a period of it has ended, the
/// conversation is open — the fold asks it, through a fork of its slot — no
/// draft or fold is running, and no fold was given up on lately
/// ([`SessionJournal::may_fold`]).
pub(super) fn begin_journal_folds(
    state: &Arc<InferenceState>,
    conv_id: &str,
    timeline: TimelineId,
) {
    let Some(open) = state.journals.open(conv_id) else {
        return;
    };
    let journal = state
        .journals
        .of(conv_id, || journal_record(state, timeline));
    {
        let held = journal.lock().unwrap();
        let entries: Vec<Entry> = held.journal().entries().cloned().collect();
        if held.journal().pending()
            || !held.may_fold(Instant::now())
            || fold::next(&entries, wall_ms(), held.style()).is_none()
        {
            return;
        }
    }
    spawn_journal_folds(
        Arc::clone(state),
        conv_id.to_string(),
        timeline,
        open,
        journal,
    );
}

/// Fold every period of the journal that has ended, one after another, as a
/// task of its own; then hand the journal to the conversation.
pub(super) fn spawn_journal_folds(
    state: Arc<InferenceState>,
    conv_id: String,
    timeline: TimelineId,
    open: Open,
    journal: Arc<Mutex<SessionJournal>>,
) {
    tokio::spawn(async move {
        let mut desk = journal_desk(&state, &conv_id, timeline, open.asker, FOLD_TOKENS);
        let mut folded = 0usize;
        let style = journal.lock().unwrap().style();
        loop {
            match folding::run(&mut desk, &journal, wall_ms(), style).await {
                Folded::Kept { fold, entry } => {
                    folded += 1;
                    tracing::info!(
                        target: "zend::journal",
                        conv_id = %conv_id,
                        into = fold.into.name(),
                        took = ?fold.ids,
                        entry = entry.id,
                        summary = %entry.summary,
                        "journal fold kept",
                    );
                }
                Folded::Nothing => break,
                Folded::Abandoned(why) => {
                    tracing::warn!(
                        target: "zend::journal",
                        conv_id = %conv_id,
                        why = ?why,
                        "journal fold abandoned",
                    );
                    journal.lock().unwrap().fold_failed(Instant::now());
                    break;
                }
            }
        }
        if folded > 0 {
            reconcile_journal_now(&state, &conv_id, timeline).await;
        }
    });
}

impl ZendSession {
    /// `GET /v1/conversations/{id}/journal`: the journal's entries as the
    /// prompt reads them, its open items, what it covers, and its recent drafts.
    /// A conversation that has written nothing has an empty one. `None` while
    /// the model is not loaded.
    pub fn journal(&self, conv_id: &str) -> Option<Value> {
        let state = self.inference.load_full()?;
        let timeline = conversation_timeline(conv_id);
        let view = match state.journals.get(conv_id) {
            Some(journal) => {
                let held = journal.lock().unwrap();
                journal_view(held.journal(), held.style())
            }
            None => {
                let held = journal_record(&state, timeline)
                    .map_or_else(SessionJournal::new, SessionJournal::restore);
                journal_view(held.journal(), style_for(conv_id))
            }
        };
        Some(view)
    }

    /// `POST /v1/conversations/{id}/journal/draft`: draft an entry now, over
    /// whatever the journal does not cover, without waiting for the cadence.
    /// Answers the stretch it covers; how it ends shows in the journal's
    /// drafts. `None` while the model is not loaded.
    pub fn journal_draft(&self, conv_id: &str) -> Option<Result<Value, NoDraft>> {
        let state = self.inference.load_full()?;
        let timeline = conversation_timeline(conv_id);
        Some(
            begin_journal_draft(
                &state,
                conv_id,
                timeline,
                Cadence::Now {
                    skip: Skip::Allowed,
                },
            )
            .map(|span| {
                json!({
                    "from_turn": origin::turn_of(span.from_turn),
                    "to_turn": origin::turn_of(span.to_turn),
                    "parts": span.turns.len(),
                })
            }),
        )
    }

    /// `GET /v1/conversations/{id}/journal/{eid}/turns`: the turns behind entry
    /// `entry` — a step's stretch, a folded entry's digest — exactly as
    /// `journal_read` pages them, `lines` lines from line `from`. `None` while
    /// the model is not loaded.
    pub fn journal_turns(
        &self,
        conv_id: &str,
        entry: u64,
        from: usize,
        lines: usize,
    ) -> Option<Result<Page, Miss>> {
        let state = self.inference.load_full()?;
        let timeline = conversation_timeline(conv_id);
        Some(journal_text(&state, timeline).read(entry, from, lines))
    }

    /// The lines of `conv_id`'s journal text that hold `pattern`, ignoring
    /// case — in `entry`'s turns, or every entry's — the first `max`, exactly
    /// as `journal_grep` finds them. `None` while the model is not loaded.
    pub fn journal_grep(
        &self,
        conv_id: &str,
        pattern: &str,
        entry: Option<u64>,
        max: usize,
    ) -> Option<Result<Hits, Miss>> {
        let state = self.inference.load_full()?;
        let timeline = conversation_timeline(conv_id);
        Some(journal_text(&state, timeline).grep(pattern, entry, max))
    }

    /// `DELETE /v1/conversations/{id}/journal/{eid}`: forget one entry. It
    /// leaves the journal and its record at once, and its section leaves the
    /// conversation's prompt at the next round. Answers the journal as it now
    /// reads; `Some(Ok(None))` when no entry by that number is held (only the
    /// newest few are). `None` while the model is not loaded.
    pub fn journal_forget(
        &self,
        conv_id: &str,
        entry: u64,
    ) -> Option<anyhow::Result<Option<Value>>> {
        let state = self.inference.load_full()?;
        let timeline = conversation_timeline(conv_id);
        let journal = state
            .journals
            .of(conv_id, || journal_record(&state, timeline));
        let prompt = {
            let mut held = journal.lock().unwrap();
            if held.journal_mut().forget(&[entry]).is_empty() {
                return Some(Ok(None));
            }
            held.journal().prompt(held.style())
        };
        let recorded = {
            let engine = &state.engine;
            engine
                .set_conversation_metadata_many(timeline, &metadata_of(&prompt))
                .map_err(|e| anyhow::anyhow!("the journal could not be recorded: {e}"))
                .and_then(|()| {
                    JournalText::of(engine.conversation().shared(), timeline).forget(entry)
                })
        };
        Some(recorded.map(|_| {
            let held = journal.lock().unwrap();
            Some(journal_view(held.journal(), held.style()))
        }))
    }
}
