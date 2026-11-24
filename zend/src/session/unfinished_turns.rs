//! Finishes the requests a previous run left unfinished, on this member and
//! under the conversation's claim.
//!
//! A tool round stopped part way is run from its sealed call turn, and a call
//! that cannot safely run twice is answered rather than run again. A
//! conversation is resumed only when this member holds it whole, since a round
//! over a history with missing turns would answer as if they were never said.
//! A hole that only this member could have filled is asked over rather than
//! waited for. Resumption runs one conversation at a time, so a boot-time burst
//! does not compete with the first real request.

use super::*;

/// The tool calls a sealed call turn is still waiting on — the round this
/// daemon did not run, because the process stopped between the call turn's seal
/// and its results.
///
/// Synchronous on purpose, and called as one statement: a `&Sequence` is not
/// `Send`, so reading the turn must not span the await that runs the tools.
/// Empty when there is nothing to finish, which the caller reads as "retract
/// the entry and leave the recorded answer standing".
pub(super) fn resumed_calls(
    conv: &Sequence,
    conv_id: &str,
    timeline: TimelineId,
    call_turn: u32,
) -> Vec<ToolCall> {
    let Some((_, assistant)) = conv.resolve_turn_text(timeline, call_turn) else {
        tracing::warn!(
            conv_id = %conv_id,
            call_turn,
            "resume: the marked turn is not in the substrate",
        );
        return Vec::new();
    };
    let calls = extract_tool_calls(&assistant);
    if calls.is_empty() {
        tracing::warn!(
            conv_id = %conv_id,
            call_turn,
            "resume: the marked turn holds no tool call — nothing to finish",
        );
    }
    calls
}

/// Rebuild the turn state a conversation was last held at, so a resumed turn
/// runs at the settings its author chose rather than at whatever the defaults
/// happen to be.
///
/// Built through the same [`dial_selection`] / [`apply_tools_dial`] the HTTP
/// path uses, rather than by restating the mapping here.
pub(super) fn resume_dials(dials: Option<ConversationDials>) -> (SelectionState, ToolMode) {
    let (effort, verbosity, think, tools) = match dials {
        Some(d) => (d.effort, d.verbosity, d.think, d.tools),
        None => resume::DEFAULT_DIALS,
    };
    let mut selection = dial_selection(Some(effort), Some(verbosity), Some(think));
    let tools_mode = ToolMode::from_level(tools);
    apply_tools_dial(&mut selection, tools_mode);
    (selection, tools_mode)
}

/// Whether `conv_id` is a worker's that another member runs now: its record
/// still beats, and names that member as its runner. A worker this member ran
/// before it restarted is this member's to finish.
pub(super) fn worker_runs_elsewhere(session: &ZendSession, conv_id: &str) -> bool {
    let Some((owner, thread, name)) = conv_names(conv_id) else {
        return false;
    };
    let Some(records) = session.shared_substrate() else {
        return false;
    };
    threads::worker(&records, owner, thread, name).is_some_and(|w| {
        let w = w.observed(workers::now_ms());
        worker_hop::runs_elsewhere(
            w.state == threads::State::Running,
            w.runner,
            worker_hop::own_node(),
        )
    })
}

/// How long a resume waits for catch-up to fill a conversation's gaps.
pub(super) const WHOLE_WAIT: Duration = Duration::from_secs(120);

/// How often it looks.
pub(super) const WHOLE_LOOK: Duration = Duration::from_millis(500);

/// How whole this member holds `timeline`, waiting up to [`WHOLE_WAIT`] while
/// a gap is still arriving for catch-up to fetch it. A hole no catch-up fills
/// is answered at once (`resume::Wholeness::Lost`): waiting cannot mend it.
pub(super) async fn wait_whole(state: &InferenceState, timeline: TimelineId) -> Wholeness {
    let until = tokio::time::Instant::now() + WHOLE_WAIT;
    loop {
        let held = resume::wholeness_of(&state.engine, timeline);
        if held != Wholeness::Arriving || tokio::time::Instant::now() >= until {
            return held;
        }
        tokio::time::sleep(WHOLE_LOOK).await;
    }
}

/// The conversations a resume may touch, of those `wanted` names.
/// Passthrough conversations are the client's own turns and are not
/// resumable here; an archived one has been distilled and must not take
/// another turn. A deleted one is already absent — `known_conversations`
/// excludes tombstoned timelines. An ingest layer, and a title job, carry no
/// `conv_id` at all and are never listed here.
pub(super) fn resumable_conversations(
    state: &InferenceState,
    wanted: impl Fn(TimelineId) -> bool,
) -> Vec<(TimelineId, String)> {
    state
        .engine
        .known_conversations()
        .into_iter()
        .filter(|(tl, conv_id, _, archived, _)| {
            !*archived && !conv_id.starts_with(passthrough::CONV_ID_PREFIX) && wanted(*tl)
        })
        .map(|(tl, conv_id, _, _, _)| (tl, conv_id))
        .collect()
}

/// Finish the requests the previous run left unfinished: a tool round it
/// stopped in the middle of, or a first round whose message never sealed
/// (`crate::resume`).
///
/// Spawned once, just after `ready`. Only a conversation carrying an entry is
/// touched, so one that ended normally is passed over. In practice the set
/// here is whatever the daemon had in flight when it stopped, usually nothing.
///
/// Resumed one at a time, deliberately: each is a full decode, and starting
/// every one of them at once would hand the scheduler a boot-time burst that
/// competes with the first real request.
pub(super) fn resume_unfinished_turns(session: Arc<ZendSession>, state: Arc<InferenceState>) {
    tokio::spawn(async move {
        let mut resumed = 0usize;
        let now = now_ms();
        for (timeline, conv_id) in resumable_conversations(&state, |_| true) {
            // **A request cut off with no entry left is recovered too.** The
            // entry is what a stop leaves, but a request can lose it — the
            // build that ended a request over a lost turn retracted it — and
            // without it nothing ever looked at the conversation again
            // (9a7088fa, 2026-11-21). Its history says the same: a call or a
            // tool round with no answer after it. Only a recent one whose last
            // turn did not fail is taken up unasked
            // (`resume::recovers_unasked`).
            if resume::unfinished(&state.engine, timeline).is_none() {
                let turns = state.engine.timeline_turn_wall_ms(timeline);
                let last_ms = turns.iter().map(|&(_, ms)| ms).max();
                let last_turn = turns.iter().map(|&(t, _)| t.0).max();
                let errors = session.conversation_turn_errors(&conv_id);
                let last_failed = last_turn.is_some_and(|t| errors.contains_key(&t));
                if !resume::recovers_unasked(last_ms, now, last_failed)
                    || !rearm_cut_off(&state, &conv_id, timeline)
                {
                    continue;
                }
            }
            if resume_conversation(&session, &state, timeline, &conv_id).await {
                resumed += 1;
            }
        }
        if resumed > 0 {
            tracing::info!(resumed, "resume: finished the turns the last run left open");
        }
    });
}

/// Resume, every [`RESUME_SWEEP`], each conversation a leaving member handed
/// on (`docs/turn_handover.md` §5), each in its own task so one long reply
/// holds up no other. A member that is itself leaving stops sweeping.
pub(super) fn resume_handed_turns(session: Arc<ZendSession>, state: Arc<InferenceState>) {
    tokio::spawn(async move {
        let mut every = tokio::time::interval(RESUME_SWEEP);
        loop {
            every.tick().await;
            if state.leaving.is_leaving() {
                return;
            }
            let handed: HashSet<TimelineId> = resume::handed(&state.engine).into_iter().collect();
            if handed.is_empty() {
                continue;
            }
            for (timeline, conv_id) in resumable_conversations(&state, |tl| handed.contains(&tl)) {
                if session.live_turn(&conv_id).is_some() {
                    continue;
                }
                let (session, state) = (Arc::clone(&session), Arc::clone(&state));
                tokio::spawn(async move {
                    resume_conversation(&session, &state, timeline, &conv_id).await;
                });
            }
        }
    });
}

/// Re-arm `conv_id`'s request when its history shows it cut off with no entry
/// left (`resume::cut_off_request`): its opening message is recorded to be
/// asked again. `false` for a request that ended.
pub(super) fn rearm_cut_off(state: &InferenceState, conv_id: &str, timeline: TimelineId) -> bool {
    let history = dialogue_history(state, conv_id);
    let made_calls = |text: &str| !extract_tool_calls(text).is_empty();
    let Some(message) = resume::cut_off_request(&history, made_calls) else {
        return false;
    };
    tracing::warn!(
        conv_id = %conv_id,
        "resume: a request cut off with no entry left; it is asked again from its opening \
         message",
    );
    resume::reask(&state.engine, timeline, &message);
    true
}

/// What `POST /v1/conversations/:id/resume` found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeAsked {
    /// Its request is resuming here.
    Resuming,
    /// A turn of it is running now.
    Live,
    /// Its request ended: nothing to resume.
    Finished,
    /// The model is not loaded yet.
    NotLoaded,
}

impl ZendSession {
    /// Resume `conv_id`'s request if it was cut off: by its entries, as the
    /// boot scan does, or — when they were retracted — by its history, a last
    /// message that is a call no turn answers or a tool round no answer
    /// follows, asked again from its opening message
    /// (`resume::cut_off_request`). Started in the background; a request that
    /// ended is never asked twice.
    pub fn resume_request(self: &Arc<Self>, conv_id: &str) -> ResumeAsked {
        let Some(state) = self.inference.load_full() else {
            return ResumeAsked::NotLoaded;
        };
        if self.live_turn(conv_id).is_some() {
            return ResumeAsked::Live;
        }
        let timeline = timeline_for(conv_id);
        if resume::unfinished(&state.engine, timeline).is_none()
            && !rearm_cut_off(&state, conv_id, timeline)
        {
            return ResumeAsked::Finished;
        }
        let (session, conv_id) = (Arc::clone(self), conv_id.to_string());
        tokio::spawn(async move {
            resume_conversation(&session, &state, timeline, &conv_id).await;
        });
        ResumeAsked::Resuming
    }
}

/// Resume what `conv_id`'s request left unfinished — a tool round, or its
/// first round's message — on this member, under the conversation's claim.
/// `true` when a turn ran. A conversation with nothing unfinished has its
/// stale entries retracted: a call turn answered, a handed mark.
pub(super) async fn resume_conversation(
    session: &Arc<ZendSession>,
    state: &Arc<InferenceState>,
    timeline: TimelineId,
    conv_id: &str,
) -> bool {
    let Some(_busy) = state.resuming.begin(timeline) else {
        return false;
    };
    let engine = &state.engine;
    let Some(mut unfinished) = resume::unfinished(engine, timeline) else {
        // The round's response turn is in the substrate, so the round
        // completed and the entry outlived it. Retract it.
        if let Some(call_turn) = resume::marked_call_turn(engine, timeline) {
            tracing::info!(
                conv_id = %conv_id,
                call_turn,
                "resume: the round completed before the daemon stopped — entry retired",
            );
            resume::clear_call_turn(engine, timeline);
        }
        if resume::is_handed(engine, timeline) {
            resume::clear_handed(engine, timeline);
        }
        return false;
    };
    if let Some((owner, thread, name)) = conv_names(conv_id) {
        let Unfinished::Round(call_turn) = unfinished else {
            // A worker's first round is its prompt, on its record: the
            // worker sweep starts it again (`mcp::resuming`).
            return false;
        };
        // A worker still beating is run by a live member, which finishes the
        // round itself: a round handed on keeps its entry until the member it
        // moved to clears it (`worker_hop`), and running it here too would run
        // its tools twice.
        if worker_runs_elsewhere(session, conv_id) {
            tracing::info!(
                conv_id = %conv_id,
                call_turn,
                "resume: its worker runs on another member, which finishes the round",
            );
            return false;
        }
        // A worker this member ran is carried on as itself, with its beat and
        // its settling, not finished with nothing listening: that left its
        // record saying nobody ran it, and another member's sweep started it
        // again over the same round.
        match workers::carry_on(session, owner, thread, name, call_turn).await {
            Ok(true) => return true,
            Ok(false) => {}
            Err(e) => tracing::warn!(
                conv_id = %conv_id,
                "resume: the worker could not be carried on: {e}"
            ),
        }
    }

    // The conversation is this member's to write only while it holds its
    // claim: another live member holding it is running this very request —
    // the entry read here is that member's, replicated — and finishes it
    // itself (`turn_claims::take`).
    let claim = match turn_claims::take(conv_id).await {
        Ok(claim) => claim,
        Err(NotTaken::Held(holder)) => {
            tracing::info!(
                conv_id = %conv_id,
                ?unfinished,
                holder = format_args!("{holder:016x}"),
                "resume: another member holds the conversation and runs its request",
            );
            return false;
        }
        // Not held complete here: the member that let it go finished the
        // request or left it for one that holds it.
        Err(NotTaken::Incomplete) => {
            tracing::warn!(
                conv_id = %conv_id,
                ?unfinished,
                "resume: this member does not hold the conversation complete; its request is \
                 left to one that does",
            );
            return false;
        }
    };
    // **Only a conversation held whole is resumed.** Another member's turns
    // arrive as gaps until their content is fetched, and the projection skips
    // an empty turn — so a round resumed now would run over a history missing
    // turns, answering as if they had never been said. On 2026-11-09 a member
    // resumed 425bad8d moments after its restart, its copy still holding the
    // other member's newest turns as gaps, and its rounds reasoned from a
    // conversation that stopped short of them. Catch-up fills the gaps; this
    // waits for it.
    match wait_whole(state, timeline).await {
        Wholeness::Whole => {}
        Wholeness::Arriving => {
            tracing::warn!(
                conv_id = %conv_id,
                ?unfinished,
                "resume: this member does not hold the conversation whole; its request is left \
                 to one that does",
            );
            turn_claims::release(claim, conv_id).await;
            return false;
        }
        // **A hole only this member could have filled is asked over, not
        // waited for.** Its own turn whose K/V died with the process before it
        // was written, or one retired in place: no member holds it, so leaving
        // the request to "one that does" left it unfinished forever — the chat
        // looking mid-answer to its reader, and every sweep passing it over
        // (BlackWorld, 2026-11-21). A round cannot carry on over a history
        // missing turns, but the request can be asked again: its opening
        // message runs as a first round, over the turns that are whole, its
        // tools again under `Dispatch::Resumed`. A request with no message to
        // ask ends at its last sealed round.
        Wholeness::Lost => match &unfinished {
            Unfinished::Message(_) => tracing::warn!(
                conv_id = %conv_id,
                "resume: a turn of the conversation is lost here; its request is asked again \
                 over the turns that are whole",
            ),
            Unfinished::Round(_) => {
                match resume::opening_message(&dialogue_history(state, conv_id)) {
                    Some(message) => {
                        tracing::warn!(
                            conv_id = %conv_id,
                            ?unfinished,
                            "resume: a turn of the conversation is lost here and no member \
                             holds it; the request is asked again from its opening message",
                        );
                        resume::reask(engine, timeline, &message);
                        unfinished = Unfinished::Message(message);
                    }
                    None => {
                        tracing::warn!(
                            conv_id = %conv_id,
                            ?unfinished,
                            "resume: a turn of the conversation is lost here and its request \
                             has no message to ask again; it is ended at its last sealed round",
                        );
                        resume::abandon(engine, timeline);
                        turn_claims::release(claim, conv_id).await;
                        return false;
                    }
                }
            }
        },
    }
    // Read again under the claim: a member that held it until a moment ago
    // may have finished the request in between.
    if resume::unfinished(engine, timeline).as_ref() != Some(&unfinished) {
        tracing::info!(
            conv_id = %conv_id,
            ?unfinished,
            "resume: the request moved on while this member waited for its claim",
        );
        turn_claims::release(claim, conv_id).await;
        return false;
    }
    // Taken: a turn cut off again marks it handed again on its way out.
    if resume::is_handed(engine, timeline) {
        resume::clear_handed(engine, timeline);
    }

    let (selection, tools_mode) = resume_dials(session.conversation_dials(conv_id));
    let (user_message, asked, start) = match unfinished {
        // The turn carries no question of its own: its message is the tool
        // output, built inside the loop from the call turn's own request. Its
        // calls may have run before it stopped: a call that cannot run twice
        // is answered instead of run again. The round it stopped at is not
        // recorded: the cap counts afresh from the round it finishes.
        Unfinished::Round(call_turn) => (
            String::new(),
            None,
            TurnStart {
                resume: Some(Resume {
                    call_turn,
                    dispatch: Dispatch::Resumed,
                    iteration: 0,
                }),
                may_hand: false,
                images: Vec::new(),
            },
        ),
        // The first round never sealed: its message is asked again, its images
        // read back by the ids its record names them by.
        Unfinished::Message(text) => {
            let (ids, message) = split_markers(&text);
            let start = TurnStart {
                images: ids.into_iter().map(ImageSource::Kept).collect(),
                ..TurnStart::default()
            };
            (message.to_string(), Some(message.to_string()), start)
        }
    };
    tracing::info!(conv_id = %conv_id, resumes_round = asked.is_none(), "resume: resuming the request");

    // Run as a chat turn runs, with no request behind it: the claim is held
    // until the turn's last record is written, so the gateway sends the page
    // here, and the reply is recorded live, so a page open across the stop —
    // or reloaded — follows the rest of it instead of finding each round only
    // once it has sealed.
    let history_before = dialogue_history(state, conv_id).len();
    let live = session.live_turns.begin(conv_id, asked, history_before);
    let mut turn = run_inference_stream(
        Arc::clone(state),
        conv_id.to_string(),
        user_message,
        None,
        None,
        None,
        None,
        false,
        tools_mode,
        None,
        selection,
        start,
    );
    relay_live(&mut turn, None, &live, conv_id).await;
    session.live_turns.end(conv_id, &live);
    drop(turn);
    turn_claims::release(claim, conv_id).await;
    true
}
