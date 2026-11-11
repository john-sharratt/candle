//! A character's conversations, as the mind holds them
//! (`docs/npc_conversations.md` §6): one timeline per party, on the schema's
//! `talk` layer, opened for a turn and parked after it.
//!
//! **A turn is two decodes.** The decision runs on an ephemeral fork under the
//! thread's catalog and writes nothing; the words are then the real turn,
//! sealed on the conversation with no grammar, no reasoning and no call, under
//! the response templates of where they go. Nothing is prefilled in either.
//!
//! **Resumed only under the frame it was written under** (§4.1): the frame is
//! the projection's under the `talking` stance, and a change to it opens the
//! next generation and retires the older ones.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use candle_conversation::projection::{DecodePriority, SelectionState};
use candle_conversation::{AskScope, Carried, Linking, Sequence, TurnOptions, Want};

use super::{
    act_specs, frame_fingerprint, journal_sweep, journal_wants, Minds, CONTEXT_WINDOW_TURNS,
    META_CONVERSATION, META_FRAME, SEAL_WAIT,
};
use crate::engine::act::{self, Act};
use crate::engine::identity::{self, Deliberation};
use crate::engine::prompt::{Persona, Stance, STANCE_SELECTOR};
use crate::engine::reflect::plain_prose;
use crate::engine::talk::catchup::{self, CatchUp};
use crate::engine::talk::decide;
use crate::engine::talk::key::{self, Party};
use crate::engine::talk::templates::Destination;
use crate::engine::tools::{self, Catalog, Within};

/// The tokens a conversation's words may not decode.
const BANNED: [&str; 2] = ["<think>", "<tool_call>"];

/// The most a line to the party decodes: one Discord message.
pub const LINE_TOKENS: usize = 320;

/// The most a journal entry decodes.
pub const JOURNAL_TOKENS: usize = 480;

/// How long a park waits for the turn's state to be durable before it lets
/// the slot go regardless.
const DURABLE_WAIT: Duration = Duration::from_secs(30);

/// One conversation, open for a turn.
pub struct Talk {
    sequence: Sequence,
    mission: Carried,
    journal: Carried,
    catchup: Carried,
    /// The generation this is.
    pub generation: u32,
}

/// One turn, as the runner hands it to the mind.
pub struct TalkTurn<'a> {
    /// The party, as the character names them.
    pub party: &'a str,
    /// The time and the gap: "It is Tuesday 14:05. You last spoke with Mira 3
    /// days ago."
    pub head: &'a str,
    /// What arrived: the burst's lines, the purposes, the compass.
    pub body: &'a str,
    pub catchup: &'a CatchUp,
    /// The gap as the summary question names it.
    pub gap: &'a str,
    pub band: DecodePriority,
}

/// What a turn decided, and the words it said.
#[derive(Clone, Debug)]
pub struct Spoken {
    pub acts: Vec<Act>,
    /// Calls that tried and failed.
    pub rejected: Vec<String>,
    pub to: Destination,
    pub words: String,
}

impl Minds {
    /// The conversation between `npc_id` and `party`: its newest generation,
    /// rejoined when it was written under this frame, else a new generation
    /// above it. Every other generation is retired.
    pub fn open_talk(
        &self,
        npc_id: u64,
        party: &Party,
        band: DecodePriority,
    ) -> anyhow::Result<Talk> {
        let p = self.projected().ok_or_else(|| {
            anyhow::anyhow!("no projection: conversations need the mind's schema")
        })?;
        let (layer, group) = p
            .talk
            .ok_or_else(|| anyhow::anyhow!("the schema declares no `talk` layer"))?;
        let frame = frame_fingerprint(&format!("{}\n{}", p.frame, Stance::Talking.id()));
        let mut cfg = self.base_config.clone();
        cfg.context_window_turns = CONTEXT_WINDOW_TURNS;
        let engine = &self.engine;
        let prefix = key::generations_of(npc_id, party);
        let mut held: Vec<(u32, _)> = engine
            .conversations_with_conv_id_prefix(&prefix)
            .into_iter()
            .filter_map(|(tl, id)| Some((key::generation_of(npc_id, party, &id)?, tl)))
            .collect();
        held.sort_by_key(|(g, _)| *g);
        let newest = held.last().copied();
        let rejoin = newest.filter(|(_, tl)| {
            engine
                .conversation_metadata_value(*tl, META_FRAME)
                .as_deref()
                == Some(frame.as_str())
        });
        let (sequence, generation) = match rejoin {
            Some((generation, timeline)) => {
                let s = engine.resume_conversation_with_projection(
                    timeline,
                    &p.prompt,
                    self.scoped_for(&p.builder, npc_id),
                    cfg,
                )?;
                (s, generation)
            }
            None => {
                let generation = newest.map_or(0, |(g, _)| g + 1);
                let s = engine.new_conversation_with_projection(
                    &p.prompt,
                    self.scoped_for(&p.builder, npc_id),
                    layer,
                    group,
                    cfg,
                )?;
                let timeline = s.timeline_id();
                let id = key::timeline(npc_id, party, generation);
                engine.set_conversation_conv_id(timeline, &id)?;
                for (k, v) in [
                    (META_FRAME, frame.as_str()),
                    (DecodePriority::METADATA_KEY, band.name()),
                    (META_CONVERSATION, id.as_str()),
                ] {
                    engine.set_conversation_metadata(timeline, k, v)?;
                }
                (s, generation)
            }
        };
        let fresh = rejoin.is_none();
        for (_, tl) in held.iter().filter(|(_, tl)| *tl != sequence.timeline_id()) {
            if let Err(e) = engine.tombstone_timeline(*tl) {
                tracing::warn!("npc {npc_id}: a superseded conversation was not retired: {e:?}");
            }
        }
        Ok(Talk {
            sequence,
            mission: Carried::new(fresh),
            journal: Carried::new(fresh),
            // Swept on every open: a crash between a reply and the release
            // leaves nothing behind (§6.1).
            catchup: Carried::new(false),
            generation,
        })
    }

    /// The conversation's state, as its timeline keeps it.
    pub fn talk_meta(&self, talk: &Talk) -> BTreeMap<String, String> {
        talk.sequence.metadata().unwrap_or_default()
    }

    /// Keep `meta` in the conversation's timeline metadata.
    pub fn set_talk_meta(
        &self,
        talk: &Talk,
        meta: &BTreeMap<String, String>,
    ) -> anyhow::Result<()> {
        talk.sequence.set_metadata_many(meta)?;
        Ok(())
    }

    /// One turn: carry the catch-up, decide on a fork, say the words.
    pub async fn talk_turn(
        &self,
        talk: &mut Talk,
        npc_id: u64,
        persona: &Persona<'_>,
        within: &Within,
        turn: TalkTurn<'_>,
    ) -> anyhow::Result<Spoken> {
        let started = Instant::now();
        let selection = self
            .talk_selection(talk, npc_id, persona, within, turn.catchup)
            .await;
        talk.sequence.publish_sections();
        let asker = talk.sequence.unsealed_asker();

        // What has happened since, as the character makes it out — or the
        // lines themselves when there are few.
        let caught = match turn.catchup.lines {
            0 => String::new(),
            _ if turn.catchup.summarised() => {
                let mut sel = selection.clone();
                identity::deliberate(&mut sel, Deliberation::None);
                let options = TurnOptions {
                    sampling: Some(self.sampling_for(Deliberation::None)),
                    max_tokens: Some(catchup::SUMMARY_TOKENS),
                    turn_banned_tokens: self.banned(),
                    selection: sel,
                    ..Default::default()
                };
                let q = catchup::summary_question(turn.party, turn.gap);
                let r = asker
                    .ask(&q, turn.band, AskScope::Conversation, options)
                    .await?;
                plain_prose(&r.text)
            }
            _ => turn.catchup.text.clone(),
        };
        let user = [turn.head, caught.as_str(), turn.body]
            .into_iter()
            .filter(|s| !s.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");

        // The decision, on a fork that writes nothing.
        let grammar = self.grammar_for(Deliberation::None, Catalog::THREAD, within);
        let options = TurnOptions {
            assistant_prefill: self.opening(Deliberation::None, grammar.is_some()),
            turn_grammar: grammar,
            sampling: Some(self.sampling_for(Deliberation::None)),
            selection: selection.clone(),
            ..Default::default()
        };
        let decided = asker
            .ask(&user, turn.band, AskScope::Whole, options)
            .await?;
        let parsed = act::parse(&decided.text, &act_specs(Catalog::THREAD, within));
        let to = decide::destination(&parsed.acts);

        // The words: the real turn, sealed.
        let mut sel = selection;
        identity::deliberate(&mut sel, Deliberation::None);
        if let Some(p) = self.projection.load().as_ref() {
            p.templates.offer(&mut sel, to);
        }
        let options = TurnOptions {
            sampling: Some(self.sampling_for(Deliberation::None)),
            max_tokens: Some(match to {
                Destination::Journal => JOURNAL_TOKENS,
                _ => LINE_TOKENS,
            }),
            turn_banned_tokens: self.banned(),
            selection: sel,
            ..Default::default()
        };
        let said = format!("{user}\n\n{}", decide::restated(turn.party, &parsed.acts));
        let response = talk
            .sequence
            .send_turn_with_options_async(&said, options)
            .await?;
        let words = plain_prose(&response.text);

        // Released once the words are sealed.
        talk.catchup
            .reconcile(
                &mut talk.sequence,
                catchup::COLLECTION,
                &[],
                Linking::Apart,
                &[catchup::SECTION],
                None,
                tokio::time::sleep,
            )
            .await;
        tracing::info!(
            npc_id,
            to = ?to,
            acts = parsed.acts.len(),
            ms = started.elapsed().as_millis() as u64,
            "a conversation's turn"
        );
        if words.trim().is_empty() {
            anyhow::bail!("the conversation's words came back empty");
        }
        Ok(Spoken {
            acts: parsed.acts,
            rejected: parsed.rejected.iter().map(|r| r.line()).collect(),
            to,
            words,
        })
    }

    /// The selection a conversation's turn reads: who the character is, its
    /// mission and journal, the catch-up, the thread's acts, the `talking`
    /// stance.
    async fn talk_selection(
        &self,
        talk: &mut Talk,
        npc_id: u64,
        persona: &Persona<'_>,
        within: &Within,
        caught: &CatchUp,
    ) -> SelectionState {
        let mission = persona.mission.map(|m| Want {
            name: identity::MISSION_SECTION,
            text: &m.text,
        });
        let mission_sealed = talk
            .mission
            .reconcile(
                &mut talk.sequence,
                identity::MISSION,
                &mission.into_iter().collect::<Vec<_>>(),
                Linking::Apart,
                &[identity::MISSION_SECTION],
                Some(SEAL_WAIT),
                tokio::time::sleep,
            )
            .await;
        let journal_sealed = talk
            .journal
            .reconcile(
                &mut talk.sequence,
                identity::JOURNAL,
                &journal_wants(persona.journal),
                Linking::Chain,
                &journal_sweep(persona.journal),
                None,
                tokio::time::sleep,
            )
            .await;
        let wants: Vec<Want<'_>> = (caught.lines > 0)
            .then_some(Want {
                name: catchup::SECTION,
                text: &caught.text,
            })
            .into_iter()
            .collect();
        let catchup_sealed = talk
            .catchup
            .reconcile(
                &mut talk.sequence,
                catchup::COLLECTION,
                &wants,
                Linking::Apart,
                &[catchup::SECTION],
                Some(SEAL_WAIT),
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
        tools::show_within(&mut selection, Catalog::THREAD.mode, within);
        selection.select(STANCE_SELECTOR, Stance::Talking.id());
        selection.select_all(
            catchup::COLLECTION,
            catchup_sealed.iter().map(String::as_str),
        );
        selection
    }

    /// The turn's banned tokens: reasoning and calls.
    fn banned(&self) -> Vec<i32> {
        let tok = self.engine.tokenizer();
        BANNED
            .iter()
            .filter_map(|t| tok.token_to_id(t))
            .map(|id| id as i32)
            .collect()
    }

    /// Let the conversation go: once the turn's state is durable, so a resume
    /// on any slot restores this turn's state rather than an older one's
    /// (§6.4).
    pub async fn park_talk(&self, talk: Talk) {
        let until = Instant::now() + DURABLE_WAIT;
        while !talk.sequence.sealed_state_durable() {
            if Instant::now() >= until {
                tracing::warn!(
                    "a conversation's state was not durable within {DURABLE_WAIT:?} — parked \
                     regardless; its next resume may restore an older turn's state"
                );
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        drop(talk);
    }

    /// Retire every conversation `npc_id` holds, of every generation and
    /// party: the character is gone.
    pub fn retire_talks(&self, npc_id: u64) -> usize {
        let mut retired = 0;
        for (tl, id) in self
            .engine
            .conversations_with_conv_id_prefix(&key::all_of(npc_id))
        {
            match self.engine.tombstone_timeline(tl) {
                Ok(()) => retired += 1,
                Err(e) => tracing::warn!("conversation {id} could not be retired: {e:?}"),
            }
        }
        retired
    }
}
