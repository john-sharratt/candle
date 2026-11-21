//! One world moment: how the world advances, and what it delivers to each
//! character when it does.
//!
//! A moment is the single point where journeys advance, visitors catch up with
//! the characters they follow, abandoned sessions give their bodies back, and the
//! command table hands out its work and notices. Doing these in the metronome's
//! closure would leave them out of every other way the world advances, so a world
//! ticked by a test would show visitors standing where their hosts had left. A
//! character is woken by the world's delivery, and its standing task reaches it on
//! the world's clock, only after a stretch of quiet, never inside a conversation.

use super::*;

impl Runtime {
    /// What a character is set on, as the turn that keeps it on course.
    ///
    /// One standing instruction, restated whenever nothing else is happening.
    /// A character with a mission gets nothing, because its mission is in its
    /// prompt; a character with none gets [`NO_MISSION`] — which is not a
    /// placeholder for one. A person with
    /// nothing assigned does not stand still, and neither should a character:
    /// looking around and talking to people is what there is to do.
    pub fn nudge_for(&self, npc_id: u64) -> Option<String> {
        // Only for a character with a body. One with no world has nowhere to
        // explore and nobody to talk to, and telling it otherwise would be
        // instructing it to do something it cannot.
        let (hosted, body) = self.body_of(npc_id)?;
        // A carried mission is already in the character's system prompt (see
        // [`Persona::mission`]), so there is no standing task to restate: saying
        // its wording again on a clock is what reads as a loop.
        //
        // **With no mission while the table gives work, there is no task to
        // restate**: the table hands a free character its next mission where it
        // stands ([`Self::assign_work`]), and one with nothing to be handed is
        // at rest ([`Self::follow_rest`]) — not told to go anywhere, and not
        // woken to be told that nothing is asked of it.
        //
        // **An unordered character is given no task** — the world gives it no
        // orders — but it is still woken: with nothing else arriving, this is
        // what rouses a character gone quiet. It is told its own idle thought
        // when its personality has one, and otherwise only that its time is its
        // own. See [`YOUR_OWN_TIME`].
        let (on_mission, unordered) = hosted.sim(|s| {
            (
                s.missions.is_on_mission(&body),
                s.missions.is_unordered(&body),
            )
        });
        if on_mission {
            return None;
        }
        let idle = self.idles.read().unwrap().get(&npc_id).cloned();
        if unordered {
            return Some(idle.unwrap_or_else(|| YOUR_OWN_TIME.to_string()));
        }
        if self.table_gives_to(&hosted, &body) {
            return None;
        }
        // Who is here — which decides *which* standing task this is, and, when
        // there is company, is itself the most useful thing in it. Reading the
        // room already told us the names; the version of this that answered
        // only `alone: bool` threw them away and left the character to guess at
        // an addressee. See [`IN_COMPANY`].
        let company: Vec<String> = hosted.read(|w| {
            let Some(here) = w.actor(&body).map(|a| a.at.clone()) else {
                return Vec::new();
            };
            w.actors_at(&here)
                .into_iter()
                .filter(|other| other.id != body)
                .map(|other| other.name.clone())
                .collect()
        });
        // A character's own idle thought is what it turns to alone; with
        // company, who is here is still the most useful thing to be told.
        Some(match company.is_empty() {
            true => idle.unwrap_or_else(|| NO_MISSION.to_string()),
            false => IN_COMPANY.replace(
                "{who}",
                &format!(
                    "{} {}",
                    npc_map::text::list(&company),
                    if company.len() == 1 { "is" } else { "are" },
                ),
            ),
        })
    }

    /// Whether the table gives `body` its work in `hosted`: the table is open
    /// and the body takes orders, in a world that draws nothing — where a
    /// free body is at rest until work is posted — or one whose table the body
    /// can reach to report at. Such a body is handed its missions
    /// ([`Self::assign_work`]) and has no standing task without one.
    pub(super) fn table_gives_to(&self, hosted: &Hosted, body: &str) -> bool {
        if body.starts_with("visitor:") {
            return false;
        }
        let (open, ordered, draws) = hosted.sim(|s| {
            (
                s.table_open,
                !s.missions.is_unordered(body),
                s.missions.draws(),
            )
        });
        open && ordered && (!draws || Self::reaches_table(hosted, body))
    }

    /// Whether `body` is a drone at rest in `hosted`
    /// (`docs/zen_everywhere.md` §3): the table gives it its work, it carries
    /// nothing, and nothing waits on the table that it may take.
    pub(super) fn at_rest(&self, hosted: &Hosted, body: &str) -> bool {
        // The mission state first: it is a map lookup, and asked of every body
        // every moment.
        hosted.sim(|s| s.missions.at_rest(body)) && self.table_gives_to(hosted, body)
    }

    /// Hold at rest exactly the characters running here whose bodies in
    /// `hosted` are drones at rest, and wake the others.
    ///
    /// **A drone with no mission does nothing and holds nothing on the card.**
    /// Put at rest it is held as a suspended character is
    /// ([`Scheduler::rest`]) and its conversation is let go
    /// ([`Minds::let_go`]): no engine slot, no resident K/V, no recurrent
    /// store, no index. Only a mission wakes it — the moment it carries one
    /// this lets it go, and the turn it then takes rejoins its conversation
    /// from the log and reads what reached it meanwhile, bounded per source.
    pub(crate) fn follow_rest(&self, hosted: &Hosted) {
        let minds = self.minds.read().unwrap().clone();
        for (npc, body) in self.bodies.in_world(hosted.id()) {
            if !self.standing.runs(npc) {
                continue;
            }
            let rest = self.at_rest(hosted, &body);
            if !self.scheduler.rest(npc, rest) {
                continue;
            }
            match rest {
                true => {
                    let freed = minds.as_ref().is_some_and(|m| m.let_go(npc));
                    tracing::info!(
                        npc,
                        freed,
                        "drone at rest: it carries no mission, and takes no turn until it is \
                         given one"
                    );
                    self.tell_whole(hosted, &body, "has come to rest, with nothing to carry.");
                }
                false => tracing::info!(npc, "drone woken: it carries a mission"),
            }
        }
    }

    /// Give every free hand in `hosted` the table's next mission for it, the
    /// one longest at rest first, and tell each what it was given.
    ///
    /// **Missions are assigned, not collected** (`docs/zen_everywhere.md` §3).
    /// The world's authority runs this every moment, so work is handed over
    /// the moment it is posted, and a character that reports is handed the
    /// next thing waiting before it can come to rest. A hand is a body that
    /// takes orders whose character is not suspended; work for a suspended
    /// one waits for it.
    pub(crate) fn assign_work(&self, hosted: &Hosted) {
        let world = hosted.id();
        let suspended = self.suspended_bodies(world);
        if hosted.sim(|s| s.missions.suspended() != &suspended) {
            hosted.with_sim(|s| s.missions.set_suspended(suspended));
        }
        // Longest at rest first; a character not at rest — it has only just
        // put a mission down — comes after every one that is.
        let mut free: Vec<(u64, u64, String)> = self
            .bodies
            .in_world(world)
            .into_iter()
            .filter(|(_, body)| {
                // The mission state first, as in [`Self::at_rest`]: whether a
                // table is in reach is a search of the map.
                hosted.sim(|s| {
                    s.missions.is_hand(body)
                        && !s.missions.is_on_mission(body)
                        && (s.missions.draws() || s.missions.has_work_for(body))
                }) && self.table_gives_to(hosted, body)
            })
            .map(|(npc, body)| {
                (
                    self.scheduler.resting_since(npc).unwrap_or(u64::MAX),
                    npc,
                    body,
                )
            })
            .collect();
        free.sort_unstable();
        for (_, npc, body) in free {
            let Some(brief) = work::take_up(hosted, &body) else {
                continue;
            };
            // Woken before it is told, so the brief is read in the turn the
            // mission wakes it into rather than counted among what waited.
            if self.standing.runs(npc) && self.scheduler.rest(npc, false) {
                tracing::info!(npc, "drone woken: it carries a mission");
            }
            self.tell(
                npc,
                self.world_ms(npc),
                Salience::URGENT,
                EventKind::Nudge {
                    text: format!("{MISSION_GIVEN}\n{brief}"),
                },
            );
            let asked = hosted
                .sim(|s| s.missions.active(&body).map(|m| m.mission_text()))
                .unwrap_or_default();
            self.tell_whole(
                hosted,
                &body,
                &format!("has taken up work: {}", first_sentence(&asked)),
            );
        }
    }

    /// Tell every mind of `hosted` that takes no orders — the minds that are
    /// the world — what `body` has just done, by its name: "Wick has come to
    /// rest". A thing that happened, so it accumulates; it asks nothing, so it
    /// is a [`EventKind::Notice`], read with whatever next wakes the mind.
    pub(crate) fn tell_whole(&self, hosted: &Hosted, body: &str, did: &str) {
        let Some(name) = hosted.read(|w| w.name_of(body).map(str::to_string)) else {
            return;
        };
        for (npc, mind) in self.bodies.in_world(hosted.id()) {
            if mind == body || !hosted.sim(|s| s.missions.is_unordered(&mind)) {
                continue;
            }
            self.tell(
                npc,
                self.world_ms(npc),
                Salience::NORMAL,
                EventKind::Notice {
                    text: format!("{name} {did}"),
                },
            );
        }
    }

    /// Fix who this character can address, as of now — taken just before the
    /// grammar is built, so the company the grammar offers and the company the
    /// world will answer an addressee against are one set. A person it offered
    /// who walks out during the decode is then called after, not refused. See
    /// [`npc_map::world::World::begin_decision`].
    pub fn begin_decision(&self, npc_id: u64) {
        if let Some((hosted, body)) = self.body_of(npc_id) {
            hosted.begin_decision(&body);
        }
    }

    /// The decision is over: acts that follow are answered from the room as it
    /// is. See [`npc_map::world::World::end_decision`].
    pub fn end_decision(&self, npc_id: u64) {
        if let Some((hosted, body)) = self.body_of(npc_id) {
            hosted.end_decision(&body);
        }
    }

    /// Walk a person into the world, beside the character they came to see.
    ///
    /// **A physical interaction is physical.** Until this, "being present" meant
    /// speech delivered straight to one character's inbox while the person had
    /// no body at all: they were not in `Within::company`, so `tell`, `ask`,
    /// `give`, `touch` and `gesture` — every act whose target binds to who is
    /// standing there — were not even in the character's grammar. It heard you
    /// and could not answer you, and went on waiting for somebody to arrive
    /// while you were talking to it.
    ///
    /// So the person gets a body. It stands where the character stands, other
    /// characters in the room see it arrive, and it is taken out again when the
    /// conversation ends — see [`Runtime::leave_world`].
    ///
    /// Returns where they ended up. `None` if the character has no body to
    /// stand beside.
    pub fn enter_world_beside(
        &self,
        npc_id: u64,
        visitor: &str,
        name: &str,
    ) -> Option<npc_map::world::Where> {
        let (hosted, body) = self.body_of(npc_id)?;
        let at = hosted.read(|w| w.actor(&body).map(|a| a.at.clone()))?;
        let arrived = hosted.with(|w| {
            // Already here is not an error — a console that reopens the same
            // conversation is the ordinary case, and re-entering would log a
            // second arrival for somebody who never left.
            match w.actor(visitor).is_some() {
                true => w.place(visitor, at.clone()).ok().map(|_| false),
                false => w.enter(visitor, name, at.clone()).ok().map(|_| true),
            }
        })?;
        if let (true, Some(handle)) = (arrived, visitor.strip_prefix(VISITOR)) {
            let room = hosted
                .read(|w| w.node(&at).map(|n| n.name.clone()))
                .unwrap_or_else(|| at.to_string());
            self.changed(
                &hosted,
                Change::Party {
                    member: handle.to_string(),
                    line: format!("{name} arrived in {room}."),
                },
            );
        }
        Some(at)
    }

    /// Take a person back out of the world.
    ///
    /// `false` if they were not in it. Everything a body was holding is given
    /// up on the way — see [`npc_map::world::World::leave`].
    pub fn leave_world(&self, visitor: &str) -> bool {
        self.hosted
            .ids()
            .iter()
            .filter_map(|id| self.hosted.get(id))
            .any(|hosted| {
                let name = hosted.read(|w| w.name_of(visitor).map(str::to_string));
                let left = hosted.with(|w| w.leave(visitor).is_ok());
                if let (true, Some(name), Some(handle)) =
                    (left, name, visitor.strip_prefix(VISITOR))
                {
                    self.changed(
                        &hosted,
                        Change::Party {
                            member: handle.to_string(),
                            line: format!("{name} left."),
                        },
                    );
                }
                left
            })
    }

    /// Tell the journals a change concerns that `change` happened in
    /// `hosted`, right where it was applied (`docs/npc_conversations.md`
    /// §5.1.1).
    pub(crate) fn changed(&self, hosted: &Hosted, change: Change) {
        let world = hosted.id();
        for (hears, line) in hosted.sim(|s| concern(&change, s)) {
            match hears {
                Hears::Bot(body) => {
                    self.journals
                        .note(world, &body, ExperienceKind::Changed, &line, None)
                }
                Hears::World => self.journals.note_world(world, &line),
            }
        }
    }

    /// Keep every visiting body with the character it came to see.
    ///
    /// **A conversation does not end because somebody walked out of the room.**
    /// A character asked to go to the chronicle goes, and a visitor left behind
    /// is talking to an empty room while the character it came for is two
    /// levels away — so the visitor goes too. Placed rather than walked: a
    /// person at a console is *with* somebody, and making them cover the
    /// distance would mean arriving after the character had moved again.
    ///
    /// Runs in the sweep, after journeys advance, so a visitor follows to where
    /// the character ended up rather than where it set out from.
    pub(super) fn keep_visitors_with_their_hosts(&self, hosted: &Hosted) {
        for (visitor, theirs) in self.visitors_astray(hosted) {
            hosted.with(|w| {
                let _ = w.place(&visitor, theirs);
            });
        }
    }

    /// [`Self::keep_visitors_with_their_hosts`] for a world this daemon
    /// follows: the placings are the authority's to make, so they go to it,
    /// each once until the world shows it made.
    pub(crate) fn follow_visitors(self: &Arc<Self>, hosted: &Hosted) {
        let Some(handle) = self.spawner() else {
            return;
        };
        for (visitor, theirs) in self.visitors_astray(hosted) {
            let at = theirs.to_string();
            let key = format!("place {visitor} {at}");
            if !self.standing.begin_sending(&key) {
                continue;
            }
            let rt = Arc::clone(self);
            let world = hosted.id().to_string();
            handle.spawn(async move {
                let op = WorldOp::Place { visitor, at };
                if let Err(e) = rt.world_write(&world, op).await {
                    tracing::warn!("world `{world}`: a visitor was not placed: {e:#}");
                }
                rt.standing.end_sending(&key);
            });
        }
    }

    /// Every visitor in `hosted` not standing where the character it came to
    /// see is, with where that is.
    pub(super) fn visitors_astray(&self, hosted: &Hosted) -> Vec<(String, Where)> {
        let now = crate::api::now_ms();
        self.interactions
            .visiting(hosted.id(), now)
            .into_iter()
            .filter_map(|(visitor, npc_id)| {
                let (_, body) = self.body_of(npc_id)?;
                hosted.read(|w| {
                    let theirs = w.actor(&body)?.at.clone();
                    let mine = w.actor(&visitor).map(|a| a.at.clone());
                    (mine.as_ref() != Some(&theirs)).then(|| (visitor.clone(), theirs))
                })
            })
            .collect()
    }

    /// One moment of world time, and everything a moment entails.
    ///
    /// **Everything, in one place.** [`environment::advance`] moves journeys on
    /// and hands out what was perceived; a moment is also when a visitor
    /// catches up with whoever it is following and when an abandoned session
    /// gives its body back. Putting the last two in the metronome's closure
    /// instead left them out of every other route by which the world advances —
    /// so a world ticked any other way had visitors standing in rooms the
    /// character had left, and no test could see it because the tests advance
    /// the world directly.
    ///
    /// The order matters: journeys first, so a visitor follows to where the
    /// character ended up rather than where it set out from.
    pub fn moment(self: &Arc<Self>, hosted: &Hosted) -> environment::Moment {
        let here = |npc_id: u64| self.standing.runs(npc_id);
        let elsewhere = |npc_id: u64, at_ms: u64, salience: Salience, kind: EventKind| {
            self.tell(npc_id, at_ms, salience, kind)
        };
        let reach = Reach {
            here: &here,
            elsewhere: &elsewhere,
        };
        let moment = environment::advance_for(
            hosted,
            &self.bodies,
            &self.scheduler,
            &|id| self.world_ms(id),
            &reach,
        );
        self.standing_tasks(hosted);
        // Whoever an operation called off has stood down is told, whatever
        // called it off — see [`crate::sim::missions::Missions::take_stood_down`].
        for body in hosted.with_sim(|s| s.missions.take_stood_down()) {
            if let Some(npc_id) = self.bodies.mind_of(hosted.id(), &body) {
                self.tell(
                    npc_id,
                    self.world_ms(npc_id),
                    Salience::URGENT,
                    EventKind::Nudge {
                        text: MISSION_WITHDRAWN.to_string(),
                    },
                );
            }
        }
        // What the order table owes: a post wakes whoever waits for it, a report
        // tells its poster. The world's authority tells, once.
        if self.standing.writes(hosted.id()) {
            // Work is handed out before the notices, so a report and the next
            // mission reach their readers in one moment.
            self.assign_work(hosted);
            for notice in hosted.with_sim(|s| s.missions.take_notices()) {
                self.give_notice(hosted, notice);
            }
            // A new day is the world's news, told once by the member that
            // writes it, on the clock of any character in it.
            let clock = self.bodies.in_world(hosted.id()).first().map(|(id, _)| *id);
            if let Some(line) =
                clock.and_then(|id| self.days.rolled(hosted.id(), self.world_ms(id)))
            {
                self.changed(hosted, Change::World { line });
            }
        }
        // After the table has given its work out: a drone handed a mission
        // this moment is awake, and one left with none comes to rest.
        self.follow_rest(hosted);
        self.keep_visitors_with_their_hosts(hosted);
        self.show_out_the_expired();
        moment
    }

    /// Deliver one of the order table's notices
    /// (`docs/zen_moon_mind.md` §8.2–§8.3).
    ///
    /// - **A post** tells nobody here: the table hands it to its carrier
    ///   ([`Self::assign_work`]), which is told then, with the mission.
    /// - **A report** tells its poster who reported what, that the report waits
    ///   at the table, and — with nothing left on the table — that it is empty.
    pub(super) fn give_notice(&self, hosted: &Hosted, notice: Notice) {
        let world = hosted.id();
        let tell = |npc_id: u64, text: String| {
            self.tell(
                npc_id,
                self.world_ms(npc_id),
                Salience::URGENT,
                EventKind::Nudge { text },
            )
        };
        match notice {
            Notice::Posted { .. } => {}
            Notice::Reported(filed) => {
                // A standing mission's report is for every body here that
                // gives orders; a posted one's for its poster.
                let posters: Vec<u64> = match filed.poster == EVERY_POSTER {
                    true => self
                        .bodies
                        .in_world(world)
                        .into_iter()
                        .filter(|(_, body)| hosted.sim(|s| s.missions.is_unordered(body)))
                        .map(|(npc_id, _)| npc_id)
                        .collect(),
                    false => self
                        .bodies
                        .mind_of(world, &filed.poster)
                        .into_iter()
                        .collect(),
                };
                if posters.is_empty() {
                    return;
                }
                let carrier = hosted
                    .read(|w| w.actor(&filed.carrier).map(|a| a.name.clone()))
                    .unwrap_or_else(|| filed.carrier.clone());
                let how = match filed.outcome {
                    Verdict::Pass => "reported it done",
                    Verdict::Fail => "reported that it could not finish it",
                };
                let empty = match hosted.sim(|s| s.missions.table_empty()) {
                    true => " The table is empty: nothing you posted waits to be taken.",
                    false => "",
                };
                let line = format!(
                    "{carrier} has {how}: {}. The report is at the order table — \
                     `order_results`.{empty}",
                    filed.brief.trim_end_matches('.')
                );
                for npc_id in posters {
                    tell(npc_id, line.clone());
                }
            }
        }
    }

    /// Restate the standing task of every character in `hosted` that runs
    /// here. It is a reading of the world and a delivery to the character's
    /// own inbox, so the member running the character asks it, from its own
    /// copy of the world.
    pub(crate) fn standing_tasks(&self, hosted: &Hosted) {
        for (npc_id, _) in self.bodies.in_world(hosted.id()) {
            if self.standing.runs(npc_id) {
                self.standing_task(npc_id);
            }
        }
    }

    /// Hand a character something, wherever it runs: into its inbox when it
    /// runs here, and as a `told` pulse from a world this daemon is the
    /// authority of when it runs on another member (§9.3). A world run cut off
    /// reaches only the characters that run here.
    pub(crate) fn tell(&self, npc_id: u64, at_ms: u64, salience: Salience, kind: EventKind) {
        if self.standing.runs(npc_id) {
            self.scheduler.deliver(npc_id, at_ms, salience, kind);
            return;
        }
        let Some(at) = self.bodies.bound(npc_id) else {
            return;
        };
        if self.standing.role(&at.world).is_authority() {
            self.standing
                .tell(&at.world, Told::new(npc_id, at_ms, salience, kind));
        }
    }

    /// What this character is set on, restated when it has gone quiet.
    ///
    /// **Asked on the world's clock, not the character's.** It used to be asked
    /// at the head of the character's own turn — which only comes when
    /// something wakes it, so a character with nothing waking it was never
    /// asked, and the standing task written for exactly that character never
    /// reached it. Every other way of being stranded ended here with nothing to
    /// rescue it. Asked from the moment, every bound character is, and the
    /// delivery itself is the wake.
    ///
    /// **Only after a stretch of quiet.** Gated on an empty inbox it never
    /// fired, because the situation is re-sent every moment; ungated it landed
    /// inside conversations, at the most recent position in the window, telling
    /// a character mid-conversation that nothing had been asked of it — two
    /// characters alternated for a hundred turns that way. So two clocks, both
    /// [`IDLE_AFTER_MS`]: quiet since anything happened, and since the task
    /// was last restated (without the second the gate latches open, because the
    /// task is not news).
    ///
    /// **A character at rest is told nothing**: it has no standing task, and
    /// nothing said here would be read before a mission wakes it.
    ///
    /// **A character with a mission is told its next step.** Its brief is in
    /// its prompt, so the brief is not restated; but a Maker gone quiet at its
    /// desk had nothing at all to bring it back, and the next step — what to do
    /// now — is what it needs.
    pub(super) fn standing_task(&self, npc_id: u64) {
        if self.scheduler.is_held(npc_id) {
            return;
        }
        let world_ms = self.world_ms(npc_id);
        // Asked twice a second for everybody, so the clock first.
        if !self.scheduler.nudge_due(npc_id, world_ms, IDLE_AFTER_MS) {
            return;
        }
        let Some(text) = self
            .nudge_for(npc_id)
            .or_else(|| self.mission_compass(npc_id))
        else {
            return;
        };
        self.scheduler
            .deliver(npc_id, world_ms, Salience::IDLE, EventKind::Nudge { text });
    }

    /// Take out the bodies of anybody whose conversation has gone quiet.
    ///
    /// **A session expiring has to actually remove somebody.** `is_live` makes
    /// an abandoned one read as gone, which is enough for a console and does
    /// nothing whatever to an actor standing in a room — so without this a
    /// visitor who closed the tab stays in the vault for ever, and the
    /// character goes on being told it has company.
    pub(crate) fn show_out_the_expired(self: &Arc<Self>) {
        for gone in self.interactions.take_expired(crate::api::now_ms()) {
            if gone.mode != tools::Mode::Physical {
                continue;
            }
            match gone.world.as_deref() {
                Some(world) if !self.standing.writes(world) => {
                    self.send_op(world, WorldOp::Leave { visitor: gone.body })
                }
                _ => {
                    self.leave_world(&gone.body);
                }
            }
        }
    }
}
