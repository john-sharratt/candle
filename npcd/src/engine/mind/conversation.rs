//! Opens, rejoins and retires a character's one conversation per day, so a
//! restart continues the day rather than abandoning it.
//!
//! A character's conversation is named for its id and the day, and tagged with
//! a fingerprint of the frame it was opened under. A restart rejoins the live
//! conversation only when both match, because a history written under a
//! different prompt or act catalogue is one the character could no longer
//! produce. Retirement tombstones a conversation rather than deleting it, so
//! its turns become reclaimable and stop being selected by default. A
//! conversation whose resume fails is retired with the rest, instead of failing
//! again on every restart. A conversation missing either tag is never rejoined,
//! which costs continuity and not correctness.

use super::*;

/// How many completed exchanges the GPU sequence carries per turn.
///
/// Matches [`crate::engine::window::DEFAULT_TURNS`] in intent — a bounded
/// verbatim tail, with continuity coming from the gather — but counts exchanges
/// rather than turns, so it is half the number.
pub const CONTEXT_WINDOW_TURNS: usize = 32;

/// The GPU tail and the perception window are two bounds with one intent,
/// counted in different units — exchanges here, turns there. Held at compile
/// time rather than by a test, so a change to either constant has to reckon with
/// the other even in a build nobody runs the tests for.
const _: () = assert!(CONTEXT_WINDOW_TURNS * 2 <= crate::engine::window::DEFAULT_TURNS);

/// The metadata key carrying a conversation's `npc-<id>-day-<n>` name.
///
/// The same string as its `conv_id`, deliberately duplicated into the free-form
/// bag: `conv_id` is a display name and answering "which timeline is this?" from
/// it means walking every conversation the workspace knows, where the metadata
/// side is an indexed lookup that already excludes tombstones. Resume runs on
/// every character's first tick, so which of those it is matters.
pub(super) const META_CONVERSATION: &str = "npc_conversation";

/// The metadata key carrying the fingerprint of the frame a conversation was
/// opened under — see [`frame_fingerprint`].
pub(super) const META_FRAME: &str = "frame_sha256";

impl Minds {
    /// Retire every conversation this character has, so the next one it opens
    /// starts fresh instead of rejoining where it stopped.
    ///
    /// **Before the character has thought this process**, which is the only
    /// place it is called: the conversation it holds in memory is opened on its
    /// first turn, so at startup there is none to strand. The same retirement
    /// [`Self::open_conversation`] runs on everything it did not rejoin — with
    /// nothing kept — so there is one way a conversation is retired, not two.
    ///
    /// Tombstoned, not deleted: the turns become reclaimable and compaction
    /// takes them. The character's memory, beliefs and relationships are
    /// untouched; only the conversation it was carrying on is gone.
    ///
    /// Returns how many conversations it retired.
    pub fn forget_conversations(&self, npc_id: u64) -> usize {
        retire_superseded(&self.engine, npc_id, None)
    }

    /// Rejoin this character's conversation, or open it a new one.
    ///
    /// # Rejoin before mint
    ///
    /// A character's conversation is named for `(npc_id, day)` and tagged with a
    /// fingerprint of the frame it was opened under. So the question a restart
    /// asks is answerable from the log alone: *is there a live conversation with
    /// this name, written under this frame?* If there is, the character carries
    /// on inside it — its turns, its recurrent state and its carried selection
    /// belief all come back — and if there is not, one is minted and tagged so
    /// the next restart can ask the same question.
    ///
    /// Before this, the answer was always no, because nothing was ever asked: a
    /// conversation open always minted a fresh timeline, so the one a previous
    /// process was using was abandoned the moment this one started, and a
    /// character that had lived all morning woke with no memory of it beyond
    /// what the gather happened to retrieve. Measured: 128 timelines for three
    /// characters on one day — roughly forty abandoned generations, each a full
    /// conversation's K/V left live and unreachable, which was the bulk of a
    /// 132 GB log.
    ///
    /// # Why the frame is checked and not just the name
    ///
    /// A conversation is only worth continuing under the frame that wrote it.
    /// Change the prompt, the act catalog or the call syntax and the history on
    /// disk is a history *this* character would never have produced — the model
    /// reads its own past making calls in a shape it can no longer make, which
    /// is worse than reading nothing. So a name match with a frame mismatch is
    /// superseded rather than rejoined, and says so. This is the same discipline
    /// zend's ingest cache runs on its `content_sha256` tag.
    ///
    /// # Order
    ///
    /// The rejoin is attempted **before** anything is retired, so what gets
    /// superseded is decided by the conversation we are actually holding rather
    /// than the one we hoped to hold. A resume that fails is therefore retired
    /// along with the rest, instead of being left live to fail again on every
    /// restart forever.
    pub(super) fn open_conversation(
        &self,
        npc_id: u64,
        day: u64,
        chapter: u64,
        persona: &Persona<'_>,
        mode: Mode,
    ) -> anyhow::Result<Live> {
        let id = conversation_id(npc_id, day, chapter);
        let mut cfg = self.base_config.clone();
        cfg.context_window_turns = CONTEXT_WINDOW_TURNS;
        // **The mind's own schema, when there is one.**
        //
        // This was a rendered copy for a long time, and the comment here said
        // why: under the schema the checkpoint opened `<think>` and never closed
        // it, so a turn ran to `max_response_tokens` and was discarded whole as
        // reasoning — a decode that reasons forever is a successful decode, so
        // every tick produced no acts and logged no error. Three suppressions
        // were tried and none held: the `no_think` glue, `thinking_effort: off`,
        // and the `ThinkMode::Off` close budget in the sampling config.
        //
        // **None of those three is the mechanism that works.** All are requests —
        // a marker this family does not honour, a selector, and a budget that
        // caps a runaway rather than preventing one. What holds is a steering
        // stencil bound to the decoded `<think>` token, which makes the block
        // unrepresentable rather than discouraged; `compile_act_loop` carries it
        // and [`Self::grammar_for`] has been putting it on every turn since.
        // The rendered prompt was working around a fault that had already been
        // fixed underneath it.
        //
        // Falling back to the rendered copy when there is no mind is not a dual
        // path kept alive for safety — a daemon with no `projection.yaml` has no
        // schema to open against, and the copy is the only prompt there is.
        let under = self.projected();
        let (system, projected) = match under.as_ref() {
            Some(p) => (p.prompt.clone(), Some(p)),
            // The same envelope `compile_act_loop` compiles, so what the prompt
            // shows a character is what the grammar will hold it to.
            None => (
                prompt::build(
                    persona,
                    mode,
                    &for_mode(mode),
                    &ToolCallEnvelope::for_dialect(&self.base_config.dialect),
                ),
                None,
            ),
        };
        let frame = match projected {
            Some(p) => p.frame.clone(),
            None => frame_fingerprint(&system),
        };

        let rejoined = resumable(&self.engine, &id, &frame).and_then(|timeline| {
            let engine = &self.engine;
            // Resumed against the same schema it was opened under. A conversation
            // minted on the projection and rejoined on the synthetic one would
            // carry a different prompt under the same fingerprint, which is the
            // one thing the fingerprint exists to make impossible.
            let resumed = match projected {
                Some(p) => engine.resume_conversation_with_projection(
                    timeline,
                    &system,
                    self.scoped_for(&p.builder, npc_id),
                    cfg.clone(),
                ),
                None => engine.resume_conversation(timeline, &system, cfg.clone()),
            };
            match resumed {
                Ok(sequence) => {
                    let depth = engine.timeline_turn_count(timeline);
                    let watermark = retention::resume_watermark(engine, timeline, depth);
                    tracing::info!(
                        depth,
                        watermark,
                        "conversation {id} rejoined — the character carries on from where it \
                         stopped rather than starting over beside its own history"
                    );
                    Some((sequence, watermark))
                }
                Err(e) => {
                    tracing::warn!(
                        "conversation {id} could not be rejoined: {e:?} — it is superseded below \
                         and a fresh one is opened in its place"
                    );
                    None
                }
            }
        });

        // **Retire everything else this character left behind.**
        //
        // Its conversations are all named `npc-<id>-day-<n>`, so its whole
        // history in the log is exactly the timelines carrying that prefix —
        // yesterday's, and every generation a previous process abandoned. None
        // of them will be read again now that today's is in hand.
        //
        // The same move zend makes when it re-ingests a file and supersedes the
        // old conversation (`code_read`): tombstone it, and compaction drops its
        // records wholesale.
        retire_superseded(
            &self.engine,
            npc_id,
            rejoined.as_ref().map(|(s, _)| s.timeline_id()),
        );

        if let Some((sequence, watermark)) = rejoined {
            return Ok(Live {
                sequence,
                mission: Carried::new(false),
                journal: Carried::new(false),
                channels: Carried::new(false),
                journal_record: String::new(),
                day,
                chapter,
                id,
                // Not zero: a conversation this deep has already retired its
                // tail, and re-walking it from the bottom would cost the first
                // few hundred turns of the restart. See
                // [`retention::resume_watermark`].
                retired_through: watermark,
                pending: Vec::new(),
            });
        }

        let sequence = {
            let engine = &self.engine;
            match projected {
                // Its own dreams and nobody else's, a few lines deep — a turn
                // in the room is reminded of one when something resonates.
                Some(p) => engine.new_conversation_with_projection(
                    &system,
                    self.scoped_for(&p.builder, npc_id),
                    p.layer,
                    p.group,
                    cfg,
                )?,
                None => engine.new_conversation(&system, cfg)?,
            }
        };
        let timeline = sequence.timeline_id();
        {
            let engine = &self.engine;
            // **The derived id, given to the substrate.** Minting it and keeping
            // it in this struct made it a log label and nothing else: the
            // timeline went into the redo log anonymous, so nothing downstream
            // could attribute a day's turns to the character that lived them.
            if let Err(e) = engine.set_conversation_conv_id(timeline, &id) {
                tracing::warn!(
                    "conversation {id} could not be named in the log: {e:?} — its turns will not \
                     be attributable to this character"
                );
            }
            // **What makes the next restart able to find this again.**
            //
            // The fingerprint is the shape and the name is what is searched on,
            // and they are written in that order deliberately: the name is the
            // commit marker, so a conversation `resumable` can find is one whose
            // fingerprint is already down. Written the other way, a crash
            // between the two writes would leave a conversation findable by name
            // with nothing to check it against — which is the one state that
            // would have to be guessed about rather than decided.
            //
            // A conversation missing either tag is simply never rejoined, which
            // is the behaviour there was before this. A failed tag therefore
            // costs continuity, not correctness.
            for (key, value) in [
                (META_FRAME, frame.as_str()),
                (META_CONVERSATION, id.as_str()),
            ] {
                if let Err(e) = engine.set_conversation_metadata(timeline, key, value) {
                    tracing::warn!(
                        "conversation {id} could not be tagged {key}: {e:?} — a restart will open \
                         a new conversation rather than rejoining this one"
                    );
                }
            }
        }
        Ok(Live {
            sequence,
            mission: Carried::new(true),
            journal: Carried::new(true),
            channels: Carried::new(true),
            journal_record: String::new(),
            day,
            chapter,
            id,
            retired_through: 0,
            pending: Vec::new(),
        })
    }

    /// Let a character's conversation go without retiring it, because the
    /// character now thinks on another member.
    ///
    /// Its slot and resident K/V are freed here; the conversation stays live in
    /// the log, and whichever member runs the character next rejoins it from
    /// there. Kept instead, the conversation stood on this member's slot as it
    /// was when the character left, while the member it went to wrote its next
    /// turns — and when the character came back, its first turn continued that
    /// stale slot against a record turns ahead of it, and failed every turn
    /// after (`slot 1 parked at turn 1025 but its record holds turn 1038`).
    ///
    /// A turn in flight keeps its own handle and finishes; only the next turn
    /// is affected, and it is the next holder's.
    pub fn let_go(&self, npc_id: u64) -> bool {
        self.release(npc_id).is_some()
    }

    /// Retire a character's conversation, because the character is gone.
    ///
    /// **On delete only, and deliberately not on shutdown.** A tombstone is the
    /// record that a conversation is finished, and a daemon stopping is not that
    /// — the cast is meant to still be there when it starts again. Retiring here
    /// would make [`Minds::open_conversation`]'s rejoin unreachable, since the
    /// lookup it runs on excludes tombstoned timelines by design.
    ///
    /// Returns how many conversations with a party it retired.
    pub async fn retire_npc(&self, npc_id: u64) -> usize {
        // Taken out of the map first, then locked: a character mid-decode is
        // awaited rather than tombstoned underneath.
        let gone = self.release(npc_id);
        if let Some(l) = gone {
            retire(&self.engine, &*l.conversation.lock().await);
        }
        // Its conversations too, of every party and generation (§4.1).
        self.retire_talks(npc_id)
    }
}

/// A fingerprint of the frame a character thinks under.
///
/// The fingerprint of a conversation's whole frame, which is everything that
/// decides what a turn in it looks like: who the character is, which acts
/// exist, what a call to one is spelled as. Change any of them and a history
/// written before the change is a history the character can no longer produce —
/// see [`Minds::open_conversation`] for why that is a reason not to rejoin it.
///
/// `frame` is the rendered prompt for a daemon with no mind, and
/// [`crate::engine::schema::frame`] under the projection.
///
/// Content-addressed rather than versioned, so nothing has to remember to bump a
/// number when it edits a prompt. That is the property zend's ingest cache
/// relies on for the same job.
pub fn frame_fingerprint(frame: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(frame.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// The live conversation named `id` and written under `frame`, if there is one.
///
/// The lookup is indexed and already excludes tombstoned timelines, so "a
/// conversation that was retired" and "no conversation" are the same answer
/// here — which is what makes retiring one the way to refuse a resume.
///
/// A name match under a *different* frame is passed over and logged, because
/// that is the case an operator needs to see: it looks identical from outside to
/// a character that simply had no history, and the reason it had none is
/// something they just changed.
///
/// More than one candidate should not happen — the name is derived from
/// `(npc_id, day)` and every open supersedes the rest — so the deepest is taken
/// rather than the first, which makes the choice deterministic instead of
/// dependent on lookup order. The others are retired by
/// [`retire_superseded`] on the way past.
pub(super) fn resumable(
    engine: &Arc<ConversationEngine>,
    id: &str,
    frame: &str,
) -> Option<TimelineId> {
    let mut best: Option<(u64, TimelineId)> = None;
    for timeline in engine.find_conversations_by_metadata(META_CONVERSATION, id) {
        match engine.conversation_metadata_value(timeline, META_FRAME) {
            Some(under) if under == frame => {}
            Some(_) => {
                tracing::info!(
                    "conversation {id} was written under a different frame — superseded rather \
                     than rejoined, because a history this character could no longer produce \
                     reads worse than no history at all"
                );
                continue;
            }
            // Named but never fingerprinted, which the tag order makes
            // unreachable: the name is written second precisely so that finding
            // one means the fingerprint is already down. Reported rather than
            // folded in with a mismatch, because the two mean different things —
            // a mismatch is a prompt that changed, this is a write that tore.
            None => {
                tracing::warn!(
                    "conversation {id} carries a name but no frame fingerprint — superseded, \
                     since there is nothing to decide against"
                );
                continue;
            }
        }
        let depth = engine.timeline_turn_count(timeline);
        if best.is_none_or(|(deepest, _)| depth > deepest) {
            best = Some((depth, timeline));
        }
    }
    best.map(|(_, timeline)| timeline)
}

/// Retire every conversation this character has left behind, except `keep`.
///
/// **A character's conversations are named `npc-<id>-day-<n>`** by
/// [`conversation_id`], so its whole history in the log is exactly the
/// timelines carrying that prefix. Opening today's supersedes all of them —
/// yesterday's, and every generation a previous process abandoned.
///
/// `keep` is the one this process just rejoined, and is the only conversation
/// that survives. It is passed as the timeline rather than looked up by name
/// because the two can disagree: a resume that *failed* has a matching name and
/// must still be retired, or it stays live and fails again on every restart.
///
/// The prefix, rather than the metadata tag [`resumable`] searches on: every
/// conversation ever opened has a `conv_id`, where only those opened since the
/// tag existed carry the tag. Sweeping on the name is what lets this retire a
/// backlog it did not write.
///
/// The lookup is prefix-scoped in the substrate rather than a filter over
/// [`ConversationEngine::known_conversations`], which materialises every
/// conversation the workspace has ever held — tombstoned included, since that
/// call feeds a sidebar. Retirement bounds the live set, not the registry, so
/// that list only grows; this one is the size of one character's history.
///
/// Already-tombstoned timelines are excluded by the lookup, which is what makes
/// this affordable to run on every conversation open rather than once at boot:
/// a character woken for the first time today retires the whole backlog, and
/// every wake after that finds nothing to do.
///
/// Failure is logged, never propagated — a tombstone that did not land leaves
/// bulk on disk, and refusing to let a character think over its bookkeeping
/// would be far worse.
///
/// Returns how many it retired.
pub(super) fn retire_superseded(
    engine: &Arc<ConversationEngine>,
    npc_id: u64,
    keep: Option<TimelineId>,
) -> usize {
    // Both the action conversations and their narrator threads, by their two
    // conv-id namespaces. The narrator thread is never rejoined — its prose is
    // derived and cheaply rebuilt — so every stale narrator generation goes,
    // and only the just-rejoined action timeline (`keep`) is spared.
    let stale: Vec<(TimelineId, String)> = [
        format!("npc-{npc_id}-day-"),
        format!("narrator-{npc_id}-day-"),
    ]
    .iter()
    .flat_map(|prefix| engine.conversations_with_conv_id_prefix(prefix))
    .filter(|(tl, _)| Some(*tl) != keep)
    .collect();
    if stale.is_empty() {
        return 0;
    }
    let mut retired = 0;
    for (tl, conv) in &stale {
        match engine.tombstone_timeline(*tl) {
            Ok(()) => retired += 1,
            Err(e) => tracing::warn!(
                "superseded conversation {conv} could not be retired: {e:?} — its turns stay \
                 live in the log and nothing will ever read them"
            ),
        }
    }
    if retired > 0 {
        tracing::info!(
            retired,
            "retired conversations this character had left behind; their turns are now \
             reclaimable"
        );
    }
    retired
}

/// Tombstone a conversation's timeline.
///
/// Failure is logged, never propagated. A tombstone that did not land leaves
/// yesterday's turns selectable — untidy, and strictly better than refusing to
/// open today's conversation over it.
pub(super) fn retire(engine: &Arc<ConversationEngine>, l: &Live) {
    match engine.tombstone_timeline(l.sequence.timeline_id()) {
        Ok(()) => tracing::info!("conversation {} retired (tombstoned)", l.id),
        Err(e) => tracing::warn!(
            "conversation {} could not be tombstoned: {e:?} — yesterday stays selectable",
            l.id
        ),
    }
}
