//! One turn of a conversation, end to end (`docs/npc_conversations.md` §6,
//! §7.1, §9): resume, catch up, decide, say, apply, write it down, park.
//!
//! **It runs on the character's lease holder**, counted as one of the
//! character's turns so a hand-back waits for it. Everything it does to the
//! world goes through the ordinary act path as [`ActSource::Talk`], so the
//! authority writes it — and stamps none of the character's own cooldowns.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use candle_conversation::projection::DecodePriority;
use journal_core::journal::Holds;
use journal_core::workflow::Outcome;
use journals::actor::Actor;
use journals::experience::ExperienceKind;
use journals::path::JournalPath;
use journals::time::Stamp;
use serde_json::Map;

use crate::api::now_ms;
use crate::authority::ops::{ActSource, OpReply, Voiced, WorldOp};
use crate::engine::act::Act;
use crate::engine::event::{EventKind, Salience, TalkLine};
use crate::engine::experience::line;
use crate::engine::mind::{Minds, Talk, TalkTurn};
use crate::engine::runtime::Runtime;
use crate::engine::talk::authority;
use crate::engine::talk::catchup::{self, CatchUp};
use crate::engine::talk::clock::{self, Ended, Input};
use crate::engine::talk::critical;
use crate::engine::talk::guard::{self, Words};
use crate::engine::talk::hand_off;
use crate::engine::talk::journal::{self, TalkJournal, TALK};
use crate::engine::talk::key::Party;
use crate::engine::talk::runner::{Inputs, Key};
use crate::engine::talk::state::TalkState;
use crate::engine::talk::templates::Destination;
use crate::engine::talk::within::OnThread;
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
                    // A held character — suspended, or a drone at rest —
                    // answers nobody: its lines wait with the conversation and
                    // are answered once it is let go.
                    if rt.scheduler.is_held(key.0) {
                        continue;
                    }
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
        let mut wrote = false;
        let ran = self.run_talk_turn(npc, &member, &inputs, &mut wrote).await;
        match ran {
            Ok(ended) => self.talks.finish(&key, Some(ended)),
            // Nothing written yet: the same lines can be taken again with
            // nothing said or noted twice.
            Err(e) if !wrote => {
                let wait = self.talks.put_back(&key, inputs, now_ms());
                tracing::warn!(
                    npc,
                    party = %member,
                    retry_in_ms = wait,
                    "a conversation's turn could not start: {e:#} — its lines wait for the next"
                );
            }
            Err(e) => {
                tracing::warn!(npc, party = %member, "a conversation's turn failed: {e:#}");
                self.talks.finish(&key, None);
            }
        }
    }

    /// One turn on `inputs`. `wrote` is set before the turn first writes
    /// anything — the character's journal, the world, the thread — so a turn
    /// that fails with it unset changed nothing and may run again.
    async fn run_talk_turn(
        &self,
        npc: u64,
        member: &str,
        inputs: &Inputs,
        wrote: &mut bool,
    ) -> anyhow::Result<Ended> {
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
        let band = talk_band(&party);
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

        let about = match &party {
            Party::Person(handle) => self.journals.person(&world, handle),
            Party::Npc(_) => self.journals.bot(&world, member),
        };
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
        let (authorizers, owed, carried) = hosted.sim(|s| {
            // What a character that takes orders carries is what its ordered
            // mission acts name.
            let carried = match s.missions.is_unordered(&body) {
                true => None,
                false => s.missions.active(&body).map(|m| m.mission_text()),
            };
            (
                s.missions.authorizers().to_vec(),
                s.owed_to(&seen.me, &seen.party_name),
                carried,
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
            carried,
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
            // Signing the step off writes the world.
            *wrote = true;
            let op = WorldOp::Authorize {
                npc,
                step: step.clone(),
                by: handle.clone(),
                name: seen.party_name.clone(),
            };
            if let Ok(OpReply::Done { ok: true }) = self.world_write(&world, op).await {
                state.pending_critical = None;
                arrived.push(format!(
                    "{} said yes: '{step}' is signed off.",
                    seen.party_name
                ));
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
        // A turn the character began itself, with nothing arrived and no yes
        // to ask for: it tells what it set out to, and decides nothing. What
        // the party said before was answered when it was said, and the head
        // says so — read in the history with nothing saying it was closed, an
        // old question was answered again ahead of every new thing told.
        let telling = (lines.is_empty() && state.pending_critical.is_none())
            .then(|| inputs.purposes.join("; "))
            .filter(|t| !t.trim().is_empty());
        let head = match telling {
            Some(_) => format!(
                "{head} {} has said nothing since, and nothing they said before waits on an \
                 answer: you are the one speaking now, to tell them one thing.",
                seen.party_name
            ),
            None => format!("{head} {ANSWERING}"),
        };
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
                    telling: telling.as_deref(),
                },
            )
            .await?;

        // What was heard is the character's own experience, written once the
        // turn has decided — the turn's first write but a sign-off's. Written
        // before the decision, a turn the model could not run (no room on the
        // card, a forward that failed) had already written and could not be
        // tried again, and its lines were lost; the turn carries them in its
        // body either way.
        *wrote = true;
        for l in &lines {
            self.journals.note(
                &world,
                &body,
                ExperienceKind::Heard,
                &line::heard(&l.name, &line::on_thread(true, ""), &l.text),
                about.clone(),
            );
        }

        // What it decided reaches the world as a conversation's acts, through
        // the world's authority wherever that runs (§7.1).
        let source = ActSource::Talk {
            party: member.to_string(),
        };
        // What the main loop is told this conversation did (§5.4), and whether
        // it settled anything that changes what the character does.
        let mut elsewhere: Vec<String> = Vec::new();
        let mut settled = false;
        for act in spoken
            .acts
            .iter()
            .filter(|a| !matches!(a.tool, "reply" | "sign_off" | "reflect"))
        {
            let done = self
                .perform_said(npc, &hand_off::as_performed(act), None, None, &source)
                .await
                .recorded;
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
                    settled = true;
                }
                // Marked here too: the authority's mark is on its own member,
                // and this one offers the next turn's acts.
                t if t.starts_with("mission_") => {
                    self.talk_changes.mark(npc, world_ms);
                    elsewhere.push(done.answer.clone());
                    settled = true;
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
        // A line said while a critical step waits on the party asks them for
        // it (§7.2).
        let voiced = |w: String| Voiced {
            text: w,
            requests: state.pending_critical.clone(),
        };
        let ended = match (spoken.to, words) {
            (Destination::Reply, Words::Send(w)) => {
                let act = decided("reply").unwrap_or_else(|| to_party("reply", &seen.party_name));
                let said = voiced(w);
                self.say_to_party(
                    npc,
                    &act,
                    &said,
                    &source,
                    &world,
                    &body,
                    &seen,
                    about.clone(),
                )
                .await?;
                elsewhere.push(format!("You said: {}", said.text));
                Ended::Spoke
            }
            (Destination::Parting, Words::Send(w)) => {
                let act =
                    decided("sign_off").unwrap_or_else(|| to_party("sign_off", &seen.party_name));
                let said = voiced(w);
                self.say_to_party(
                    npc,
                    &act,
                    &said,
                    &source,
                    &world,
                    &body,
                    &seen,
                    about.clone(),
                )
                .await?;
                elsewhere.push(format!("You said goodbye: {}", said.text));
                state.open = false;
                Ended::Quiet
            }
            (_, Words::Send(w) | Words::Journal(w)) => {
                self.journals
                    .note(&world, &body, ExperienceKind::Wrote, &w, about.clone());
                elsewhere.push(format!("You wrote down: {w}"));
                // Read through what it took in, so the party's typing
                // indicator stops (§8) — on the authority, which writes the
                // thread and runs the indicator.
                if let Some(seq) = lines.iter().map(|l| l.seq).max() {
                    let op = WorldOp::ReadThrough {
                        npc,
                        thread: seen.thread.clone(),
                        seq,
                    };
                    if let Err(e) = self.world_write(&world, op).await {
                        tracing::warn!(npc, "a conversation's read did not reach its world: {e:#}");
                    }
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
        // **The conversation writes its own journal, and a kept entry is news
        // to the mind** (§5.5): what was asked and answered, what the
        // character means to do about it and what it left open, drafted once
        // the words have gone — so the party never waits on it — and told so
        // that it wakes the mind. Zen told a person "I am going to connect to
        // the terminal now", and nothing ever reached the mind that could.
        // **What the mind had said for it wakes the mind once it is said**:
        // the telling was the mind's own act, and its landing is that act's
        // outcome. Before the journal, and whether or not an entry is kept: a
        // telling hears nothing, so its stretch often adds none, and Zen —
        // having told a person "I will now survey the devices" — was woken by
        // nothing and surveyed nothing.
        // **A kept entry is what the mind is told, and nothing beside it**: it
        // says what was asked, what was answered and what is meant to be done
        // about it, so the line said and the acts taken would be the same
        // thing twice.
        let told = telling.is_some() && matches!(ended, Ended::Spoke);
        if told {
            self.scheduler.deliver(
                npc,
                world_ms,
                Salience::NORMAL,
                EventKind::Concluded {
                    party: seen.party_name.clone(),
                    applied: said_for_the_mind(&spoken.words),
                },
            );
            // The mind has told this person something: what they asked of it
            // stands no longer, once it has worked at it.
            if party.is_person() {
                self.scheduler.told(npc, &seen.party_name);
            }
        }
        if matches!(ended, Ended::Spoke) {
            if let Some(entry) = self
                .journal_turn(
                    &minds,
                    &talk,
                    npc,
                    &lines,
                    &seen.me,
                    &spoken.words,
                    world_ms,
                    band,
                )
                .await
            {
                // **What a person asked, and the character means to do, is
                // held until it is done and told** (`Scheduler::ask`): the
                // entry reaches the mind once, and the next thing to wake it
                // says nothing of it.
                if party.is_person() && !told && entry.contains(TALK.meaning_to) {
                    self.scheduler.ask(npc, &seen.party_name, &entry, world_ms);
                }
                elsewhere = vec![entry];
                // A telling has already woken the mind; its entry is read on
                // the turn that wake starts.
                settled = settled || !told;
            }
        }
        if settled {
            // **A promise or a mission change is told so it wakes the mind**
            // (§5.4, D10): it is how a conversation changes what the character
            // does, and the mind is what does it. Told quietly, it waited for
            // whatever woke the mind next — for a character standing in a quiet
            // room, nothing — so what was settled on the phone was never done.
            self.scheduler.deliver(
                npc,
                world_ms,
                Salience::NORMAL,
                EventKind::Concluded {
                    party: seen.party_name.clone(),
                    applied: elsewhere.join(" "),
                },
            );
        } else {
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
        }
        minds.park_talk(talk).await;
        Ok(ended)
    }

    /// Draft the conversation's journal over the turn just taken — what
    /// `lines` said and what `reply` answered — and keep its record with the
    /// conversation. The entry as it reads in full when one was kept; `None`
    /// when nobody spoke, the stretch added nothing, or the draft did not
    /// finish, which costs an entry and nothing else.
    #[allow(clippy::too_many_arguments)]
    async fn journal_turn(
        &self,
        minds: &Minds,
        talk: &Talk,
        npc: u64,
        lines: &[&TalkLine],
        me: &str,
        reply: &str,
        world_ms: u64,
        band: DecodePriority,
    ) -> Option<String> {
        let turns = journal::stretch(lines, me, reply, world_ms);
        let newest = turns.last()?.id;
        let mut held = TalkJournal::restore(&minds.talk_meta(talk));
        let span = held.journal_mut().begin(turns, newest)?;
        let state = Mutex::new(held);
        let (outcome, _, record) = minds.talk_journal(talk, &state, &span, band).await;
        if let Some(record) = record {
            let kept = BTreeMap::from([(journal::META_JOURNAL.to_string(), record)]);
            if let Err(e) = minds.set_talk_meta(talk, &kept) {
                tracing::warn!(npc, "a conversation's journal was not recorded: {e:#}");
            }
        }
        match outcome {
            Outcome::Wrote { entry, .. } => {
                tracing::info!(
                    npc,
                    entry = entry.id,
                    "a conversation's journal entry was kept — {}",
                    entry.summary
                );
                Some(entry.detail(&journal::TALK))
            }
            Outcome::Nothing { why } => {
                tracing::info!(npc, "nothing worth a conversation's journal entry — {why}");
                None
            }
            Outcome::Abandoned(why) => {
                tracing::warn!(
                    npc,
                    "a conversation's journal entry was not written — {why:?}"
                );
                None
            }
        }
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
        let paths: Vec<_> = paths
            .into_iter()
            .filter(|p| tree.get(p).is_some())
            .collect();
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
    async fn say_to_party(
        &self,
        npc: u64,
        act: &Act,
        said: &Voiced,
        source: &ActSource,
        world: &str,
        body: &str,
        seen: &OnTheThread,
        about: Option<JournalPath>,
    ) -> anyhow::Result<()> {
        let done = self
            .perform_said(npc, act, None, Some(said), source)
            .await
            .recorded;
        if !done.landed {
            anyhow::bail!("the conversation's words did not land: {}", done.answer);
        }
        self.journals.note(
            world,
            body,
            ExperienceKind::Said,
            &line::said(&seen.party_name, &line::on_thread(true, ""), &said.text),
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

/// What a turn that answers somebody is told about what it can do.
///
/// **A conversation only talks; the doing is the same character's, elsewhere.**
/// Its decision is offered the thread's acts and nothing of the world, and a
/// model that reads no `terminal_list` among them says so: asked to list its
/// terminals, Zen answered that it had no such tool "in this session" and
/// could not see the house — while its mind held every one of them.
const ANSWERING: &str = "On the phone you only talk. Whatever they ask you to do, find out or \
                         check is done by you once you have answered, away from the phone, where \
                         everything you can work is to hand — so never say you cannot or have no \
                         way to: say what you will do, and it becomes yours to do. Never say it \
                         is done, posted or sent: nothing is, until you have done it away from \
                         the phone.";

/// What the mind reads once the words it asked for have been said: the words,
/// that saying them again adds nothing, and that what they promised is still
/// to be done — woken by its own telling, a mind with nothing new told the
/// person the same thing again a minute later.
fn said_for_the_mind(words: &str) -> String {
    format!(
        "You have now said it: \"{}\" It is said; saying it again tells them nothing. Whatever \
         you said you would do is still yours to do — do it, and speak to them again only when \
         there is something new to tell.",
        words.trim()
    )
}

/// The band a conversation with `party` decodes at.
///
/// **A person on the other end is waiting on the turn right now** — the
/// engine's `Urgent`, the band of zend's live dialogue. At `High` a direct
/// message to Zen shared the card with the Makers' composing, which runs at
/// `High` by design, and each answer took three to four minutes to come back.
/// Another character is not waiting on anyone, and talks in the background.
fn talk_band(party: &Party) -> DecodePriority {
    match party.is_person() {
        true => DecodePriority::Urgent,
        false => DecodePriority::Low,
    }
}

#[cfg(test)]
mod tests {
    use super::{said_for_the_mind, talk_band, DecodePriority, Party};

    /// The mind is told what was said, and that the doing is still its own.
    #[test]
    fn a_telling_comes_back_as_said_and_still_to_be_done() {
        assert_eq!(
            said_for_the_mind(" Connected to the house network. Scanning now.\n"),
            "You have now said it: \"Connected to the house network. Scanning now.\" It is said; \
             saying it again tells them nothing. Whatever you said you would do is still yours \
             to do — do it, and speak to them again only when there is something new to tell."
        );
    }

    /// A person is answered at `Urgent`; another character in the background.
    #[test]
    fn a_person_is_answered_at_urgent() {
        assert_eq!(
            talk_band(&Party::Person("u_154797f3".into())),
            DecodePriority::Urgent
        );
        assert_eq!(talk_band(&Party::Npc(7)), DecodePriority::Low);
    }
}
