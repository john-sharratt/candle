//! Putting a character's act into the world and recording what came of it.
//!
//! An act is recorded twice, for two readers. The feed shows what was asked and
//! what came of it, so a person scanning it can set the two side by side. The
//! character is told only the world's verdict, because its own arguments handed
//! back as an outcome would read as one. A refusal keeps the world's words,
//! since there the prose is the information. Costs and follow-up turns follow the
//! verdict: a refused act stalls nothing, and an act the world answered brings the
//! character straight back to read the answer.

use super::*;

impl Runtime {
    /// The acts that are a word addressed to somebody, by the argument naming
    /// them.
    pub(super) const SPOKEN: &'static [&'static str] =
        &["ask", "tell", "message", "reply", "gesture", "give"];

    /// Sign off the steps of this character's mission that the word it just
    /// addressed to somebody was about, and say what is next. `None` when the
    /// act was not a word to somebody, or no step was about them.
    pub(super) fn mark_spoken(&self, npc_id: u64, act: &Act) -> Option<String> {
        if !Self::SPOKEN.contains(&act.tool) {
            return None;
        }
        let who = act
            .args
            .get("to")
            .and_then(|v| v.as_str())?
            .trim()
            .to_string();
        let (hosted, body) = self.body_of(npc_id)?;
        // Only a word to somebody in the world: a message to the channel is to
        // nobody in particular, and signs no step about anybody off.
        if !hosted.read(|w| w.actors().any(|a| a.name.eq_ignore_ascii_case(&who))) {
            return None;
        }
        hosted.with_sim(|s| {
            if !s.missions.spoke_with(&body, &who) {
                return None;
            }
            s.missions.observe(&body, act.summary());
            tracing::info!(npc = %body, %who, "mission step signed off by a word to somebody");
            let next = s.missions.active(&body).and_then(Mission::next_step);
            Some(progress_line(
                &format!("You have spoken to {who}"),
                next,
                &[],
            ))
        })
    }

    /// Let the room see that something standing in it refused this character.
    /// See [`witnessed::show`].
    pub(crate) fn show_refusal(&self, npc_id: u64, act: &Act, answer: &str) {
        if let Some((hosted, body)) = self.body_of(npc_id) {
            witnessed::show(&hosted, &body, act, answer);
        }
    }

    /// Arm the scheduling half of an act the world accepted — the **stall**.
    ///
    /// Read from [`crate::engine::cooldown::Cost::stall`] rather than named
    /// here, so an act's two costs are chosen together in one table instead of
    /// one being a rate and the other an `if` in this file. Most acts stall for
    /// nothing: only crossing a room and putting a hand on somebody take time
    /// the world can see.
    ///
    /// **The mood is recorded either way.** It rode inside the stall while
    /// `reflect` was the only act that stalled, which made a scheduling
    /// decision quietly load-bearing for a durable record: giving reflection
    /// its honest zero-length stall would have stopped every character's
    /// feeling being written down, and nothing would have said so. They are two
    /// unrelated things about the same act and are now spelled that way.
    pub(super) fn arm_pause(&self, npc_id: u64, act: &Act, now_ms: u64) {
        if let Some(stall) = crate::engine::cooldown::stall_after(act.tool) {
            self.scheduler
                .pause_for(npc_id, now_ms, stall.as_millis() as u64);
        }
        self.note_feeling(npc_id, act);
    }

    /// Give a character another turn when its act came back with something.
    ///
    /// **The world answered, so the character has to get to use the answer.**
    /// What a board says exists in the act's outcome and nowhere else — nothing
    /// in the world perceives a document being read, so no sweep will ever
    /// deliver it — and a character is otherwise unscheduled the moment its
    /// queue empties. The result was a Maker that read the muster board, had the
    /// contents written into its window, went to sleep, and read the board again
    /// the next time anything woke it.
    ///
    /// The same holds for a refusal and for work on a document — the world said
    /// something back — and not for speech, which is the treadmill: a
    /// character that speaks and is immediately asked again speaks again, into
    /// the same silence, until its own window holds nothing but its own voice.
    /// [`body::continues`] holds the rule.
    ///
    /// **Whatever the act, one that `moved` the mission comes straight back.**
    /// A word that signed a step off, a time machine set, a mission taken up
    /// or handed in: each is answered with what comes next, and that is the
    /// world answering, not the character talking.
    ///
    /// **So does what a mind with no body did through a terminal.** What the
    /// cluster reads, what a television is doing, whether a light took the
    /// call: each is in the act's answer and nowhere else, and whoever asked
    /// is waiting to be told. Left to the next thing that happened to wake
    /// it, Zen read the cluster and told John what it said minutes later,
    /// when a Drone posted on the channel.
    pub(super) fn arm_followup(&self, npc_id: u64, act: &Act, landed: bool, moved: bool) {
        let bridged = is_of_the_bridge(act.tool) && self.is_bodiless(npc_id);
        let through_a_terminal = answers_through_a_terminal(act.tool) || bridged;
        // A reading taken is what makes telling whoever asked an answer — not
        // the connect that opened the way to it ([`terminal_reads`]).
        if (terminal_reads(act.tool) || bridged) && landed {
            self.scheduler.worked(npc_id);
        }
        if moved || through_a_terminal || body::continues(act, landed) {
            self.scheduler.think_again(npc_id);
        }
    }

    /// The world and body a character acts through, if it has one.
    pub fn body_of(&self, npc_id: u64) -> Option<(Arc<Hosted>, String)> {
        let at = self.bodies.bound(npc_id)?;
        Some((self.hosted.get(&at.world)?, at.body))
    }

    /// Put one act into the world, if it is a body's act and there is a body.
    ///
    /// [`Outcome::NotOfTheBody`] when neither is true — a character with no
    /// body, or an act that happens inside a head. Those are the caller's to
    /// record as intent.
    ///
    /// The **world's** verdict comes back rather than the character's intent,
    /// and the caller must keep the two apart: an act that was refused and one
    /// that succeeded must not read the same afterwards, or a character spends
    /// the rest of the day reasoning from a move it never made.
    pub fn act_on_world(&self, npc_id: u64, act: &Act) -> Outcome {
        self.act_on_world_said(npc_id, act, None, &ActSource::Mind)
    }

    /// [`Self::act_on_world`], with the line the lease holder voiced for a
    /// phone act and where the act came from. A conversation's act stamps
    /// none of the character's cooldowns: its guards are its own
    /// (`docs/npc_conversations.md` §8).
    pub fn act_on_world_said(
        &self,
        npc_id: u64,
        act: &Act,
        said: Option<&Voiced>,
        source: &ActSource,
    ) -> Outcome {
        if !body::is_of_the_body(act.tool) {
            return Outcome::NotOfTheBody;
        }
        let Some((hosted, body)) = self.body_of(npc_id) else {
            return Outcome::NotOfTheBody;
        };
        // Nothing is pushed here. An act changes the world; **the world's next
        // moment is when anyone perceives that**, including the character that
        // acted — one rule, one batched prefill per moment, no path where
        // perception arrives outside the sweep.
        //
        // The actor is not left in the dark meanwhile: a refusal comes back in
        // the same turn, which is the whole reason it is rendered rather than
        // logged.
        // A conversation's mission act is checked again where it lands, and
        // an ordered body's is its own (`engine::talk::orders`).
        if let (true, ActSource::Talk { party }) = (agenda::is_mine(act.tool), source) {
            return self.talk_mission_act(npc_id, &hosted, &body, act, party, said);
        }
        let done = body::perform_said(&hosted, &body, act, said);
        // **Only what actually happened costs a wait.** A refused act is not a
        // thing the body did, and charging for one would leave a character that
        // mis-named a room standing still for fifteen seconds over a typo.
        if done.happened() && *source == ActSource::Mind {
            // The act, not its name: `act` is two rates wearing one word, and
            // this is the only caller that knows which of them happened.
            self.cooldowns.took_act(npc_id, act);
        }
        done
    }

    /// Put one act into the world and render the single line that goes into the
    /// character's window and the Pulse feed.
    ///
    /// **The world's verdict decides which of two forms this takes.**
    ///
    /// An act that landed is recorded as the act: [`Act::summary`], the same
    /// `tool — intent` shape every act outside a world already has. The world's
    /// own reply to a successful act reads as narration — "You shout, for
    /// anyone within earshot." — and taking that instead made speech the one act that did not
    /// look like an act. That is wrong twice over: in the feed, where it broke
    /// a column of single-word tools, and in the character's own window, where
    /// the model reads its history and is being shown what an act looks like.
    ///
    /// A refusal keeps the world's line, because there the prose *is* the
    /// information — which rooms are reachable, who is actually here. So the
    /// two no longer merely differ in wording; they differ in form, which is a
    /// stronger version of the distinction this path exists to preserve.
    /// **An act that answers keeps the world's line too**, on the same
    /// reasoning as a refusal — see [`body::ANSWERS`]. Reading a document is
    /// not something the room perceives, so the contents are in the outcome and
    /// nowhere else; summarising it would record that the character read
    /// something and drop what it read.
    ///
    /// # Two readers, two lines
    ///
    /// Everything above is about the **feed** — what a person watching Pulse
    /// reads. The character is a different reader with a different need, and
    /// [`Recorded::answer`] is its line: always the world's own verdict, for
    /// every act, because that is the only thing that tells it whether what it
    /// tried actually happened.
    ///
    /// One string served both while nothing carried results back to the model.
    /// Now that `<tool_response>` does, the summary is the wrong thing to send:
    /// it hands a character *its own arguments* back as though they were an
    /// outcome. Measured live, `gesture — for Wailen to see that the room is
    /// ours for now` — which is what the character asked for, not what came of
    /// it. `body::ANSWERS` was an early, narrower version of exactly this
    /// concern, written when the feed was the only channel there was.
    pub fn record_act(&self, npc_id: u64, act: &Act) -> Recorded {
        self.record_act_said(npc_id, act, None, &ActSource::Mind)
    }

    /// [`Self::record_act`], with the line the lease holder voiced for a
    /// phone act and where the act came from.
    pub fn record_act_said(
        &self,
        npc_id: u64,
        act: &Act,
        said: Option<&Voiced>,
        source: &ActSource,
    ) -> Recorded {
        let outcome = self.act_on_world_said(npc_id, act, said, source);
        let landed = outcome.happened();
        // What the character is told, whatever the verdict. `NotOfTheBody` has
        // no line of its own — nothing in a world happened — so it says so
        // rather than echoing the call back.
        let mut answer = match outcome.line() {
            Some(line) => line.to_string(),
            None => format!("Nothing in the world answers `{}`.", act.tool),
        };
        // A word to somebody that a mission step asked for signs the step off,
        // and the character is told so beneath what came of it.
        if landed {
            if let Some(line) = self.mark_spoken(npc_id, act) {
                answer.push('\n');
                answer.push_str(&line);
            }
            // A notable act of its own is its experience (§5.1): what changed
            // the world, never walking, waiting or speech.
            if experience::notable(act.tool) {
                if let Some((hosted, body)) = self.body_of(npc_id) {
                    let did = answer.lines().next().unwrap_or_default();
                    self.journals
                        .note(hosted.id(), &body, ExperienceKind::Did, did, None);
                }
            }
        }
        // **Both halves, always, in one shape.**
        //
        // The feed used to show one or the other and the choice depended on the
        // verdict: an act that landed showed the *call*, an act refused showed
        // the *world's reply*. So "Wailen Wylde is already on it." sat beside
        // "invite — Yaelis Vayne; …" with nothing saying they were the same
        // kind of event, that one had failed, or which act the sentence was
        // even about.
        //
        // Now every line reads `tool — what was asked → what came of it`, with
        // `✗` in place of the arrow when it did not land. The console splits on
        // exactly those marks to set each part in its own face
        // (`pulse.js::splitAct`), which is why they are single characters with
        // spaces around them and not words.
        let departing = outcome.departing();
        let feed = match outcome {
            Outcome::Refused(why) => format!("{} {REFUSED} {why}", act.summary()),
            // A landed act — spoken here or called after somebody leaving —
            // reads `tool — what was asked → what came of it`.
            Outcome::Did(line) | Outcome::Departed(line) => {
                format!("{} {LANDED} {line}", act.summary())
            }
            // Nothing in a world happened, so there is nothing to arrow to.
            Outcome::NotOfTheBody => act.summary(),
        };
        Recorded {
            feed,
            answer,
            landed,
            departing,
        }
    }

    /// Remember a register a character has just named for itself.
    ///
    /// **Only from an act the world accepted**, and only `pause` names one —
    /// so this sits beside [`Runtime::arm_pause`] rather than inside the act,
    /// for the reason the deadline does: what a room sees is the world's, and
    /// what outlives the process is the registry's.
    ///
    /// A word the mind has no mood for is still recorded. The grammar steers to
    /// the library where there is one, and where there is not the character is
    /// free-decoding — refusing what it said would mean a character that felt
    /// something the daemon had no vocabulary for felt nothing at all.
    pub(crate) fn note_feeling(&self, npc_id: u64, act: &Act) {
        if act.tool != "reflect" {
            return;
        }
        let Some(feeling) = act.args.get("feeling").and_then(|v| v.as_str()) else {
            return;
        };
        let feeling = feeling.trim();
        if feeling.is_empty() {
            return;
        }
        if let Some(sink) = self.mood_sink.read().unwrap().as_ref() {
            sink(npc_id, feeling);
        }
    }
}
