//! Runs one thinking step for a character: its reads in, its acts out.
//!
//! A tick brings the character's mission and journal sections in line, chooses
//! what its turn may reach from where it stands, and decodes one turn under its
//! own lock, with the answers to its last acts at the head. Errors are returned
//! rather than swallowed, because a failed decode and a character that chose to
//! do nothing both yield no acts, and the caller must be able to tell them
//! apart. A day or mission boundary starts a new conversation. Turns that fall
//! below the horizon are tombstoned after the conversation's lock is released,
//! so a failed write catches up on a later turn instead of stranding turns.
//! The outcomes of the acts are held for the next turn, so they are never
//! delivered into a conversation that did not ask the questions.

use super::*;

/// How long a turn waits for a mission section it just submitted to seal before
/// it reads the stand-in member instead.
pub(super) const SEAL_WAIT: Duration = Duration::from_secs(20);

/// What one character's tick produced.
#[derive(Debug, Default)]
pub struct Thought {
    pub parsed: Parsed,
    /// Set when this tick opened a new day's conversation.
    pub rolled_over: Option<(u64, u64)>,
    /// How many turns this tick retired from the log.
    ///
    /// Reported rather than silent so an operator watching the pulse can see
    /// retention working — a log that quietly stops growing looks identical to
    /// one that quietly stopped being written. Ordinarily one; more than one
    /// means a gap was being closed.
    pub retired: u32,
    /// The curated third-person prose the character actually read this tick — the
    /// narrator's rendering of the events, or empty on a quiet tick with nothing
    /// to narrate. Reported so the pulse view shows what went to the model rather
    /// than the raw events behind it. See [`crate::engine::narrator`].
    pub perception: String,
}

impl Minds {
    /// Run one thinking step: perception in, acts out.
    ///
    /// Errors are returned rather than swallowed. A decode that failed and a
    /// character that chose to do nothing produce the same empty act list, and
    /// the caller has to be able to tell them apart — Pulse renders them
    /// differently, and it should.
    // Each is a different axis of one turn: who, as what, in what mode, on what
    // day, given what arrived, and from what the room offers. Bundling them into
    // a struct would name the bundle rather than the axes and hide that every
    // caller must supply all of them — which is the property that matters.
    #[allow(clippy::too_many_arguments)]
    pub async fn think(
        &self,
        npc_id: u64,
        persona: &Persona<'_>,
        mode: Mode,
        day: u64,
        chapter: u64,
        events: &[Event],
        within: &tools::Within,
    ) -> anyhow::Result<Thought> {
        let mut thought = Thought::default();

        // This character's conversation for the day and chapter: the one it
        // holds, or one opened in its place — see `Minds::resident_for`, which a
        // trial opens through too.
        let (resident, rolled_over) = self
            .resident_for(npc_id, day, chapter, persona, mode)
            .await?;
        thought.rolled_over = rolled_over;

        // **The answers to last turn's acts, then everything that has happened
        // since — one message.**
        //
        // The results ride at the head because the protocol puts them there:
        // what a character did is answered before the world is allowed to speak
        // again. A fat batch is one better-informed thinking step rather than
        // several thrashing ones — the mind design is explicit that this is
        // what a busy character should get.
        //
        // The answers and the perception are built inside the decode block
        // below, under the character's own lock: the events are first curated
        // into prose by this character's narrator thread (see
        // [`crate::engine::narrator`]), so the turn is one decode's worth of
        // narrator work followed by the act decode, both under the one guard.
        // **The act stencil is armed here, not hoped for.** `act::parse` reads
        // the decode, and what it reads is a grammar's output rather than a
        // guess at one: the name is a real tool, the required parameters are
        // present, the JSON closes. Everything it still rejects is a genuine
        // choice the character made, which is what makes a rejection worth
        // handing back to it.
        // **Who this turn is, and how hard it thinks.**
        //
        // The identity collections are `Named`, so an unpinned one emits
        // nothing — a character with no identity at all, and silently. Every
        // collection is pinned, down to its generic member.
        //
        // The deliberation dial is set **either way**. It is a property of the
        // turn rather than of the schema — Qwen3 honours `/no_think` only from
        // the user opener, which the conversation layer bakes from this
        // selection — so a daemon running the rendered prompt needs it just as
        // much. It used to be built only on the projected branch, so the path
        // the daemon actually ran never injected the switch at all.
        // The selection is made under the character's lock, after its mission
        // section has been reconciled, because which member the `mission`
        // collection pins depends on whether that section has sealed.
        let turn_grammar =
            self.grammar_for(identity::Deliberation::default(), Catalog::ROOM, within);
        let mut options = TurnOptions {
            assistant_prefill: self
                .opening(identity::Deliberation::default(), turn_grammar.is_some()),
            turn_grammar: turn_grammar.map(TurnGrammar::of),
            sampling: Some(self.sampling_for(identity::Deliberation::default())),
            // **Inside its own dreams' scope.** A turn teaches the hit levels of
            // the scopes its tags name, and the dream group is scoped to this
            // character's tag — so without it a character's own turns, the only
            // traffic its dreams are ever read against, would teach them
            // nothing. See `Minds::warm_dreams`.
            tags: vec![dreams::tag(npc_id)],
            ..Default::default()
        };
        let (response, timeline, watermark) = {
            // The async guard is DESIGNED to live across this await (module
            // *Locking* note): it is what makes one turn at a time per
            // character true while the rest of the cast decodes concurrently.
            let mut live = resident.conversation.lock().await;
            options.selection = self.prepare(&mut live, npc_id, persona, within).await;
            // The answers to last turn's acts, taken here so they die with the
            // conversation on a roll-over rather than being delivered into one
            // that never asked the questions.
            let answers = std::mem::take(&mut live.pending);
            // **Curate this tick's events into prose.** The raw events —
            // first-person intent, condensed "does it:" action lines — are
            // decoded through this character's narrator thread into the
            // third-person passage it actually reads. Two decodes under the one
            // guard: the narrator, then the act. A quiet tick with nothing to
            // narrate skips the narrator decode entirely. See
            // [`crate::engine::narrator`].
            let narration = match narrator::build_turn(
                persona.name,
                &narrator::sketch(persona.identity, persona.manner),
                &within.company,
                events,
            ) {
                Some(turn) => narration::render(&turn),
                None => String::new(),
            };
            let narration = narration.trim().to_string();
            // What the pulse reports as perceived: the curated prose, not the
            // raw events behind it — this is what actually went to the model.
            thought.perception = narration.clone();
            let perception = compose_narrated(&answers, &narration);
            // **What this turn may reach for, from where it stands** — set
            // before the send, so the prefill's projection and every one the
            // decode makes after it read the same shelves.
            let scopes = self.scopes.load_full();
            let bearing = scopes.as_ref().map(|s| {
                s.apply(
                    &self.engine,
                    live.sequence.timeline_id(),
                    persona.personality,
                    &within.parts,
                    within.year,
                )
            });
            let response = live
                .sequence
                .send_turn_with_options_async(&perception, options)
                .await?;
            // **Whether a dream reached this turn**, said in the log whenever
            // one did. A dream is only worth having if it comes back, and it
            // comes back by winning the gather rather than by being pasted in —
            // so the only way to see it is to ask the projection what it held.
            if let Some(ev) = live.sequence.projection_event(&response.stats) {
                self.system_prompts.pin().insert(
                    npc_id,
                    rendered::render(&ev, &live.sequence.section_contents()),
                );
                let recalled: Vec<(u64, f32)> = ev
                    .selection
                    .turns
                    .iter()
                    .filter(|t| t.layer == dreams::LAYER && t.selected)
                    .filter_map(|t| Some((t.timeline?, t.score)))
                    .collect();
                // **What the shared layers brought in.** The world, the eras
                // and the stories reach a character only by winning the gather,
                // and for as long as they sat where no gather could read them
                // nothing said so: every turn's count here was zero.
                let own = self
                    .projected()
                    .and_then(|p| {
                        let layers = &p.builder.schema().layers;
                        Some(layers.iter().find(|l| l.id == p.layer)?.name.clone())
                    })
                    .unwrap_or_default();
                let shared = recalled_by_layer(&ev.selection.turns, &own);
                tracing::info!(
                    "npc {npc_id}: recalled from beneath{} — {}",
                    format!(
                        "{}{}",
                        bearing
                            .as_deref()
                            .map(|b| format!(" at {b}"))
                            .unwrap_or_default(),
                        within.year.map(|y| format!(" in {y}")).unwrap_or_default()
                    ),
                    match shared.is_empty() {
                        true => "nothing".to_string(),
                        false => shared
                            .iter()
                            .map(|(layer, r)| {
                                let top = r
                                    .top
                                    .and_then(|t| scopes.as_ref()?.address(t))
                                    .unwrap_or("?");
                                format!("{layer} {} (best {:.0} {top})", r.count, r.best)
                            })
                            .collect::<Vec<_>>()
                            .join(", "),
                    }
                );
                if !recalled.is_empty() {
                    let from: BTreeSet<u64> = recalled.iter().map(|(tl, _)| *tl).collect();
                    // The scores too: whether a line came back because it
                    // resonated or only because the group had room is the one
                    // thing a gate on this layer would be set from.
                    let scores: Vec<String> =
                        recalled.iter().map(|(_, s)| format!("{s:.0}")).collect();
                    tracing::info!(
                        "npc {npc_id}: this turn was reminded of {} line(s) from {} of its \
                         dream(s), scored {} — timelines {:?}",
                        recalled.len(),
                        from.len(),
                        scores.join("/"),
                        from
                    );
                }
            }
            (response, live.sequence.timeline_id(), live.retired_through)
        };
        // **Retire whatever now sits below the horizon.**
        //
        // On the insert rather than on a timer: a turn is the only thing that
        // pushes another one out, so the moment one lands is exactly when the
        // question has a new answer. The sweep looks *back* rather than
        // assuming one turn fell out since last time, which is what makes a
        // failed write or a restarted daemon catch up instead of stranding
        // those turns below the horizon for good.
        //
        // The conversation's own lock is released first: retiring writes the
        // substrate, and holding the conversation across it would put this
        // character's next turn behind its own bookkeeping.
        if let Some(keep) = self.keep_turns {
            let settled = {
                let engine = &self.engine;
                // **The substrate's count, not the sequence's.**
                //
                // `Sequence::turn_count` counts a user and an assistant message
                // separately and starts from zero whenever a process opens a
                // fresh sequence; turn *indices* advance once per exchange. A
                // horizon from that counter therefore runs at twice the rate of
                // the turns it indexes — measured live, `turn_count=65` gave
                // `horizon=0` and by 65 exchanges the horizon had passed the
                // last real turn, retiring the whole conversation. `keep_turns:
                // 64` came to mean keep nothing.
                let depth = engine.timeline_turn_count(timeline);
                Some(retention::retire_expired(
                    engine, timeline, depth, keep, watermark,
                ))
            };
            if let Some((through, retired)) = settled {
                resident.conversation.lock().await.retired_through = through;
                thought.retired = retired;
            }
        }
        thought.parsed = act::parse(&response.text, &act_specs(Catalog::ROOM, within));
        Ok(thought)
    }

    /// What a turn of this character's reads: who it is, what it has been asked
    /// to do and written down, and what it can do where it stands.
    ///
    /// Its mission and journal sections are brought in line with what the
    /// character holds first. A mission waits up to [`SEAL_WAIT`] for wording
    /// just submitted to seal, because until it has the turn reads the mission's
    /// stand-in member. A journal does not wait: the turn pins the journal
    /// sections that have sealed, and an entry still sealing joins a later turn,
    /// while the turns it was written from are still in the window. The journal
    /// record is written to the conversation's metadata as well, so a journal
    /// carried into a new day is the one a restart restores.
    ///
    /// A mission carries its own [`identity::Deliberation`], but none declares a
    /// level, so every character is on the default: act, do not deliberate.
    /// Every act is installed in the schema's `tools` collection and a turn names
    /// the ones it can take — the same set `grammar_for` compiles the mask from —
    /// so the list a character reads is never wider or narrower than what it can
    /// decode. A question put to it reads the same selection, which is what makes
    /// its answer about the character that acts.
    pub(super) async fn prepare(
        &self,
        live: &mut Live,
        npc_id: u64,
        persona: &Persona<'_>,
        within: &tools::Within,
    ) -> SelectionState {
        self.prepare_waiting(live, npc_id, persona, within, None)
            .await
    }

    /// [`Self::prepare`], waiting up to `journal_wait` for journal entries still
    /// sealing, so what is read does not change as they seal. A trial waits: two
    /// trials of one body must read the same journal.
    pub(super) async fn prepare_waiting(
        &self,
        live: &mut Live,
        npc_id: u64,
        persona: &Persona<'_>,
        within: &tools::Within,
        journal_wait: Option<Duration>,
    ) -> SelectionState {
        let mission = persona.mission.map(|m| Want {
            name: identity::MISSION_SECTION,
            text: &m.text,
            sharing: Sharing::Owned,
        });
        let mission_sealed = live
            .mission
            .reconcile(
                &mut live.sequence,
                identity::MISSION,
                &mission.into_iter().collect::<Vec<_>>(),
                Linking::Apart,
                Changes::Any,
                &[identity::MISSION_SECTION],
                Some(SEAL_WAIT),
                tokio::time::sleep,
            )
            .await;
        let journal_sealed = live
            .journal
            .reconcile(
                &mut live.sequence,
                identity::JOURNAL,
                &journal_wants(persona.journal),
                Linking::Chain,
                Changes::Any,
                &journal_sweep(persona.journal),
                journal_wait,
                tokio::time::sleep,
            )
            .await;
        if let Some(journal) = persona.journal {
            if let Err(e) = self.record_journal(npc_id, live, journal.record()) {
                tracing::warn!("npc {npc_id}: {e:#}");
            }
        }
        let declared = self
            .projection
            .load()
            .as_ref()
            .is_some_and(|p| p.identities.channels);
        let channels: Vec<Want<'_>> = persona
            .channels
            .iter()
            .filter(|_| declared)
            .map(|c| Want {
                name: &c.name,
                text: &c.text,
                sharing: Sharing::Owned,
            })
            .collect();
        let channels_sealed = live
            .channels
            .reconcile(
                &mut live.sequence,
                identity::CHANNELS,
                &channels,
                Linking::Apart,
                Changes::Any,
                &[],
                None,
                tokio::time::sleep,
            )
            .await;
        let mut selection = self.selection(
            npc_id,
            persona,
            within,
            !mission_sealed.is_empty(),
            &journal_sealed,
        );
        identity::carry_channels(&mut selection, &channels_sealed);
        selection
    }

    /// The selection a turn reads, given which of the character's mission and
    /// journal sections have sealed.
    pub(super) fn selection(
        &self,
        npc_id: u64,
        persona: &Persona<'_>,
        within: &tools::Within,
        mission_sealed: bool,
        journal_sealed: &[String],
    ) -> SelectionState {
        let mut selection = self
            .projection
            .load()
            .as_ref()
            .map(|p| {
                let mut sel = p.identities.selection_for(
                    npc_id,
                    persona.personality,
                    persona.world_id,
                    persona.building,
                    identity::Deliberation::default(),
                );
                p.identities
                    .act_as(&mut sel, persona.personality, persona.mission.is_some());
                sel
            })
            .unwrap_or_else(|| identity::deliberation(identity::Deliberation::default()));
        if persona.mission.is_some() {
            identity::carry_mission(&mut selection, mission_sealed);
        }
        if persona.journal.is_some() {
            identity::carry_journal(&mut selection, journal_sealed);
        }
        tools::show_within(&mut selection, Catalog::ROOM.mode, within);
        selection
    }

    /// Hand back what became of the acts this character just called for.
    ///
    /// One entry per call it made, in the order it made them, whatever the
    /// verdict — an act that was refused, or a call that was malformed, needs
    /// an answer at least as much as one that worked. They ride at the head of
    /// this character's next turn as `<tool_response>` blocks; see [`compose`].
    ///
    /// Called after the acts have been enacted, because that is when their
    /// outcomes exist. A character with no live conversation is a no-op rather
    /// than an error: it has no turn for them to ride on, so there is nothing
    /// to hold them against.
    pub async fn deliver_outcomes(&self, npc_id: u64, outcomes: Vec<String>) {
        if outcomes.is_empty() {
            return;
        }
        if let Some(l) = self.resident_of(npc_id) {
            l.conversation.lock().await.pending.extend(outcomes);
        }
    }
}
