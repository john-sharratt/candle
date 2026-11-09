//! One turn of a conversation, end to end (`docs/npc_conversations.md` §6,
//! §7.1, §9): resume, catch up, decide, say, apply, write it down, park.
//!
//! **It runs on the character's lease holder**, counted as one of the
//! character's turns so a hand-back waits for it. Everything it does to the
//! world goes through the ordinary act path as [`ActSource::Talk`], so the
//! authority writes it — and stamps none of the character's own cooldowns.

use std::sync::Arc;
use std::time::Duration;

use candle_conversation::projection::DecodePriority;
use journals::actor::Actor;
use journals::experience::ExperienceKind;
use journals::path::JournalPath;
use journals::time::Stamp;
use serde_json::Map;

use crate::api::now_ms;
use crate::authority::ops::{ActSource, OpReply, WorldOp};
use crate::engine::act::Act;
use crate::engine::event::{EventKind, Salience, TalkLine};
use crate::engine::experience::line;
use crate::engine::mind::TalkTurn;
use crate::engine::runtime::Runtime;
use crate::engine::talk::catchup::{self, CatchUp};
use crate::engine::talk::clock::{self, Ended, Input};
use crate::engine::talk::critical;
use crate::engine::talk::guard::{self, Words};
use crate::engine::talk::key::Party;
use crate::engine::talk::runner::{Inputs, Key};
use crate::engine::talk::state::TalkState;
use crate::engine::talk::templates::Destination;
use crate::engine::talk::within::OnThread;
use crate::engine::talk::authority;
use crate::sim::phone::other_of;

/// How often the runner looks for a conversation whose turn is due.
pub const TICK: Duration = Duration::from_millis(250);

/// What the thread says, read once for a turn.
struct OnTheThread {
    thread: String,
    party_name: String,
    me: String,
    /// The character's own last lines on it, newest last.
    own_last: Vec<String>,
    last_said_ms: Option<u64>,
}

impl Runtime {
    /// A `Talk` or `Purpose` event for `npc_id`: its conversation's, not its
    /// mind's.
    pub fn talk_arrived(&self, npc_id: u64, kind: EventKind) {
        let now = now_ms();
        match kind {
            EventKind::Talk { thread, lines } => {
                // Off the thread's id, not this member's copy of it: the
                // lines arrive ahead of the thread's record (§9).
                let Some((_, body)) = self.body_of(npc_id) else {
                    return;
                };
                if let Some(member) = other_of(&thread, &body) {
                    self.talks.lines(npc_id, &member, &thread, lines, now);
                }
            }
            EventKind::Purpose { party, intent } => {
                self.talks.purpose(npc_id, &party, &intent, now);
            }
            _ => {}
        }
    }

    /// Run every conversation held here, for the runtime's life: each
    /// [`TICK`], the due ones take their turn.
    pub fn start_talking(self: &Arc<Self>) {
        let Some(spawner) = self.spawner() else {
            return;
        };
        let back = Arc::downgrade(self);
        spawner.spawn(async move {
            loop {
                tokio::time::sleep(TICK).await;
                let Some(rt) = back.upgrade() else {
                    return;
                };
                for key in rt.talks.due(now_ms()) {
                    let turn = tokio::spawn(Arc::clone(&rt).talk_turn(key.clone()));
                    // Recorded once spawned; a turn that ended first leaves
                    // only a finished handle, which cancels nothing.
                    rt.talks.running(key, turn.abort_handle());
                }
            }
        });
    }

    /// Rescan `npc_id`'s pair threads for lines its conversations have not
    /// answered: a `Told` pulse is not persisted, so a character started or
    /// woken here picks up what waited for it (§9).
    pub fn rescan_talks(&self, npc_id: u64) {
        let Some((hosted, body)) = self.body_of(npc_id) else {
            return;
        };
        let waiting: Vec<(String, String, Vec<TalkLine>)> = hosted.sim(|s| {
            s.threads
                .unanswered_for(&body)
                .into_iter()
                .map(|(thread, member, lines)| {
                    let lines = lines
                        .into_iter()
                        .map(|m| TalkLine {
                            name: s.threads.name_of(&m.from).unwrap_or(&m.from).to_string(),
                            seq: m.seq,
                            from: m.from,
                            text: m.text,
                            account: m.account,
                            at_ms: m.at_ms,
                        })
                        .collect();
                    (thread, member, lines)
                })
                .collect()
        });
        for (thread, member, lines) in waiting {
            self.talks.lines(npc_id, &member, &thread, lines, now_ms());
        }
    }

    async fn talk_turn(self: Arc<Self>, key: Key) {
        let (npc, member) = key.clone();
        let inputs = self.talks.take(&key);
        if inputs.lines.is_empty() && inputs.purposes.is_empty() {
            self.talks.finish(&key, None);
            return;
        }
        if !self.standing.begin_turn(npc) {
            // Being handed back: whoever holds it next rescans the thread.
            self.talks.finish(&key, None);
            self.talks.ran(&key);
            return;
        }
        // Ends the turn's count however the turn ends — cancelled included,
        // or a hand-back would wait on it for ever.
        let _counted = Counted {
            rt: Arc::clone(&self),
            key: key.clone(),
        };
        let ran = self.run_talk_turn(npc, &member, &inputs).await;
        match ran {
            Ok(ended) => self.talks.finish(&key, Some(ended)),
            Err(e) => {
                tracing::warn!(npc, party = %member, "a conversation's turn failed: {e:#}");
                self.talks.finish(&key, None);
            }
        }
    }

    async fn run_talk_turn(&self, npc: u64, member: &str, inputs: &Inputs) -> anyhow::Result<Ended> {
        let minds = self
            .minds
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| anyhow::anyhow!("the engine is not loaded"))?;
        let persona = self
            .persona_of_async(npc)
            .await
            .ok_or_else(|| anyhow::anyhow!("npc {npc} has no persona"))?;
        let (hosted, body) = self
            .body_of(npc)
            .ok_or_else(|| anyhow::anyhow!("npc {npc} has no body"))?;
        let world = hosted.id().to_string();
        let party = Party::of_member(member, |b| self.bodies.mind_of(&world, b));
        let seen = hosted
            .sim(|s| {
                let t = s.threads.direct_between(&body, member)?;
                let own_last: Vec<String> = t
                    .since_joined(&body)
                    .iter()
                    .filter(|m| m.from == body)
                    .map(|m| m.text.clone())
                    .collect();
                Some(OnTheThread {
                    thread: t.id.clone(),
                    party_name: t.as_named_to(&body),
                    me: s.roster_name(&body).unwrap_or(&body).to_string(),
                    last_said_ms: t.last_said_by(&body).map(|m| m.at_ms),
                    own_last,
                })
            })
            .ok_or_else(|| anyhow::anyhow!("npc {npc} has no thread with {member}"))?;
        let band = match party.is_person() {
            true => DecodePriority::High,
            false => DecodePriority::Low,
        };
        let mut talk = minds.open_talk(npc, &party, band)?;
        let now = Stamp::now();
        let first = inputs.lines.first().map_or(0, |l| l.seq.saturating_sub(1));
        let mut state = TalkState::from_meta(&minds.talk_meta(&talk))
            .unwrap_or_else(|| TalkState::begin(party.clone(), seen.thread.clone(), first, now));
        state.generation = talk.generation;
        let lines: Vec<_> = inputs
            .lines
            .iter()
            .filter(|l| l.seq > state.answered_seq)
            .collect();
        if lines.is_empty() && inputs.purposes.is_empty() {
            minds.park_talk(talk).await;
            return Ok(inputs.last.unwrap_or(Ended::Quiet));
        }

        // What was heard is the character's own experience.
        let about = match &party {
            Party::Person(handle) => self.journals.person(&world, handle),
            Party::Npc(_) => self.journals.bot(&world, member),
        };
        for l in &lines {
            self.journals.note(
                &world,
                &body,
                ExperienceKind::Heard,
                &line::heard(&l.name, &line::on_thread(true, ""), &l.text),
                about.clone(),
            );
        }
        let input = match lines.is_empty() {
            true => Input::Purpose,
            false => Input::Line,
        };
        state.activation = clock::activation(state.activation, inputs.last, input);

        // What happened since the last turn.
        let caught = self.catch_up(&world, &body, about.as_ref(), &state, now);

        // What the turn can do.
        let world_ms = self.world_ms(npc);
        let owner = self
            .npcs()
            .and_then(|n| n.view().payload(npc).map(|p| p.owner_id.clone()));
        let (authorizers, owed) = hosted.sim(|s| {
            (
                s.missions.authorizers().to_vec(),
                s.owed_to(&seen.me, &seen.party_name),
            )
        });
        let standing = authority::standing(&party, owner.as_deref(), &authorizers);
        let within = self.within(npc).on_thread(&OnThread {
            party: &party,
            name: &seen.party_name,
            owed,
            may_change_mission: authority::may_change(
                standing,
                self.talk_changes.last(npc),
                world_ms,
            ),
            capped: guard::capped(&party, state.activation),
            cooling: guard::cooling(&party, seen.last_said_ms, now_ms()),
        });

        let gap = catchup::gap(now.secs() - state.last_turn.secs());
        let head = format!("You last spoke with {} {gap} ago.", seen.party_name);
        let mut arrived: Vec<String> = lines
            .iter()
            .map(|l| format!("{}: {}", l.name, l.text.trim()))
            .collect();
        arrived.extend(
            inputs
                .purposes
                .iter()
                .map(|p| format!("You wanted to tell {}: {p}", seen.party_name)),
        );

        // A critical step: signed by the yes to what was asked, or asked for.
        let next_critical = hosted.sim(|s| {
            s.missions
                .active(&body)
                .and_then(|m| m.critical_next().map(str::to_string))
        });
        let said_by_them: Vec<&str> = lines.iter().map(|l| l.text.as_str()).collect();
        let signed = critical::signs(
            state.pending_critical.as_deref(),
            next_critical.as_deref(),
            standing,
            &said_by_them,
        );
        if let (Some(step), Party::Person(handle)) = (signed, &party) {
            let op = WorldOp::Authorize {
                npc,
                step: step.clone(),
                by: handle.clone(),
                name: seen.party_name.clone(),
            };
            if let Ok(OpReply::Done { ok: true }) = self.world_write(&world, op).await {
                state.pending_critical = None;
                arrived.push(format!("{} said yes: '{step}' is signed off.", seen.party_name));
            }
        }
        if state.pending_critical.is_none() {
            state.pending_critical = critical::asks(standing, next_critical.as_deref(), input);
        }
        if let Some(step) = &state.pending_critical {
            arrived.push(format!(
                "Your mission waits on {}'s yes to: {step}. Ask them for it plainly.",
                seen.party_name
            ));
        }
        let spoken = minds
            .talk_turn(
                &mut talk,
                npc,
                &persona.as_persona(),
                &within,
                TalkTurn {
                    party: &seen.party_name,
                    head: &head,
                    body: &arrived.join("\n"),
                    catchup: &caught,
                    gap: &gap,
                    band,
                },
            )
            .await?;

        // What it decided reaches the world as a conversation's acts.
        let source = ActSource::Talk {
            party: member.to_string(),
        };
        // What the main loop is told this conversation did (§5.4).
        let mut elsewhere: Vec<String> = Vec::new();
        for act in spoken
            .acts
            .iter()
            .filter(|a| !matches!(a.tool, "reply" | "sign_off" | "reflect"))
        {
            let done = self.record_act_said(npc, act, None, &source);
            if !done.landed {
                continue;
            }
            let arg = |k: &str| act.args.get(k).and_then(|v| v.as_str()).unwrap_or_default();
            match act.tool {
                "promise" => {
                    let promised = line::promised(arg("to"), arg("what"), arg("by"));
                    self.journals.note(
                        &world,
                        &body,
                        ExperienceKind::Promised,
                        &promised,
                        about.clone(),
                    );
                    elsewhere.push(promised);
                }
                t if t.starts_with("mission_") => {
                    self.talk_changes.mark(npc, world_ms);
                    elsewhere.push(done.answer.clone());
                }
                _ => {}
            }
        }
        if let Some(reflect) = spoken.acts.iter().find(|a| a.tool == "reflect") {
            self.note_feeling(npc, reflect);
        }

        // And the words go where it decided.
        let own: Vec<&str> = seen.own_last.iter().map(String::as_str).collect();
        let words = guard::words_for(&seen.party_name, &own, &spoken.words);
        let decided = |tool: &str| spoken.acts.iter().rev().find(|a| a.tool == tool).cloned();
        let ended = match (spoken.to, words) {
            (Destination::Reply, Words::Send(w)) => {
                let act = decided("reply").unwrap_or_else(|| to_party("reply", &seen.party_name));
                self.say_to_party(npc, &act, &w, &source, &world, &body, &seen, about.clone())?;
                elsewhere.push(format!("You said: {w}"));
                Ended::Spoke
            }
            (Destination::Parting, Words::Send(w)) => {
                let act =
                    decided("sign_off").unwrap_or_else(|| to_party("sign_off", &seen.party_name));
                self.say_to_party(npc, &act, &w, &source, &world, &body, &seen, about.clone())?;
                elsewhere.push(format!("You said goodbye: {w}"));
                state.open = false;
                Ended::Quiet
            }
            (_, Words::Send(w) | Words::Journal(w)) => {
                self.journals
                    .note(&world, &body, ExperienceKind::Wrote, &w, about.clone());
                elsewhere.push(format!("You wrote down: {w}"));
                // Read through what it took in, so the party's typing
                // indicator stops (§8).
                if let Some(seq) = lines.iter().map(|l| l.seq).max() {
                    hosted.with_sim(|s| s.threads.read_through(&body, &seen.thread, seq));
                }
                Ended::Quiet
            }
        };

        if let Some(seq) = lines.iter().map(|l| l.seq).max() {
            state.answered_seq = state.answered_seq.max(seq);
        }
        state.last_turn = Stamp::now();
        state.seen = caught.seen;
        minds.set_talk_meta(&talk, &state.to_meta())?;
        // Told quietly: it waits for the main loop's next turn rather than
        // starting one.
        self.scheduler.deliver(
            npc,
            world_ms,
            Salience::IDLE,
            EventKind::Elsewhere {
                party: seen.party_name.clone(),
                lines: elsewhere,
            },
        );
        minds.park_talk(talk).await;
        Ok(ended)
    }

    /// The catch-up for a turn at `now`: the character's journal, the world's,
    /// and the party's, since the bookmark.
    fn catch_up(
        &self,
        world: &str,
        body: &str,
        about: Option<&JournalPath>,
        state: &TalkState,
        now: Stamp,
    ) -> CatchUp {
        let (Ok(tree), Some(bot), Some(group)) = (
            self.journals.tree(),
            self.journals.bot(world, body),
            self.journals.group_of(world),
        ) else {
            return CatchUp::default();
        };
        let mut paths = vec![bot.clone()];
        paths.extend(self.journals.world(world));
        paths.extend(about.cloned());
        let paths: Vec<_> = paths.into_iter().filter(|p| tree.get(p).is_some()).collect();
        let pending = self.journals.pending(&bot);
        catchup::gather(
            &tree,
            &Actor::Npcd { group },
            &paths,
            state.last_turn,
            &state.seen,
            &pending,
            now,
        )
        .unwrap_or_else(|e| {
            tracing::warn!("a conversation's catch-up could not be read: {e}");
            CatchUp::default()
        })
    }

    /// Put the conversation's words on the pair's thread as `act`, and write
    /// them down as said.
    #[allow(clippy::too_many_arguments)]
    fn say_to_party(
        &self,
        npc: u64,
        act: &Act,
        words: &str,
        source: &ActSource,
        world: &str,
        body: &str,
        seen: &OnTheThread,
        about: Option<JournalPath>,
    ) -> anyhow::Result<()> {
        let done = self.record_act_said(npc, act, Some(words), source);
        if !done.landed {
            anyhow::bail!("the conversation's words did not land: {}", done.answer);
        }
        self.journals.note(
            world,
            body,
            ExperienceKind::Said,
            &line::said(&seen.party_name, &line::on_thread(true, ""), words),
            about,
        );
        Ok(())
    }
}

/// A conversation's turn, counted among the character's: ended when dropped.
struct Counted {
    rt: Arc<Runtime>,
    key: Key,
}

impl Drop for Counted {
    fn drop(&mut self) {
        self.rt.standing.end_turn(self.key.0);
        self.rt.talks.ran(&self.key);
    }
}

/// `tool` addressed to the party, with no intent of its own: the words carry
/// it.
fn to_party(tool: &'static str, party: &str) -> Act {
    let mut args = Map::new();
    args.insert("to".into(), party.into());
    args.insert("intent".into(), "".into());
    Act { tool, args }
}
