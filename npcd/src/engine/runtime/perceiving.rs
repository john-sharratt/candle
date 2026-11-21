//! What a character can see, reach and do from where its body stands right now.
//!
//! The grammar a character decodes under is built from the same read the
//! situation text is built from, so the names it is shown and the names it may
//! address cannot disagree. A body that takes no orders has no room to stand in,
//! and a mind with no body reaches the outside only through a terminal. Acts a
//! character has just taken, and acts its loop guard has struck, are left out of
//! the grammar, so a decode cannot repeat itself on the same few acts.

use super::*;

impl Runtime {
    /// What the world offers this character *right now*, as the grammar's own
    /// vocabulary.
    ///
    /// **The same read the situation and the standing task already do**, handed
    /// to the stencil so the three cannot disagree. They did: the prompt named
    /// who was here, the standing task named them again, and the mask let the
    /// character write any string at all — so it wrote a first name, or a
    /// person in another room, and was refused. A grammar built from this is
    /// masked to exactly what the character was told.
    ///
    /// Empty for a character with no body, which is the honest answer: nobody
    /// is standing next to somebody who is nowhere.
    pub fn within(&self, npc_id: u64) -> tools::Within {
        let Some((hosted, body)) = self.body_of(npc_id) else {
            return tools::Within::nowhere();
        };
        // Names, because that is what the grammar binds and what the character
        // was shown. It used to carry the body ids alongside, to look up whose
        // mind was behind each name and ask what it was waiting for; nothing
        // waits any more, so the second half went with the wait.
        //
        // The people among them are the `visitor:{handle}` bodies, and whether
        // a shout from here reaches one is perception's own rule
        // (`docs/npc_conversations.md` §12).
        //
        // **A mind that takes no orders has no body** (`docs/zen_everywhere.md`
        // §4): it is in nobody's company, walks, rides and attends nowhere, and
        // works no station where its seat is. What it is offered is the order
        // table, its agenda and its messaging.
        let bodiless = hosted.sim(|s| s.missions.is_unordered(&body));
        let (company, people, heard_by_person) = hosted.read(|w| {
            let Some(mine) = w.actor(&body).filter(|_| !bodiless) else {
                return (Vec::new(), Vec::new(), false);
            };
            let here = mine.at.clone();
            let others: Vec<_> = w
                .actors_at(&here)
                .into_iter()
                .filter(|a| a.id != body)
                .collect();
            let company = others.iter().map(|a| a.name.clone()).collect();
            let people = others
                .iter()
                .filter(|a| people::is_person(&a.id))
                .map(|a| a.name.clone())
                .collect();
            (company, people, people::hear_a_shout(w, &here))
        });
        // The lift, from where this body stands: whether it is on a landing,
        // whether the car is open there, and the floors it could ride to. What
        // `lift_call` / `lift_use` are gated on, and what the ride binds to — one
        // read, from [`World::lift_within`], so the situation and the grammar
        // cannot disagree about it.
        //
        // **A body that moves by attention neither walks nor rides**: it has the
        // world's rooms to `focus` on instead, and no destinations, no lift.
        let attends = hosted.sim(|s| s.attends(&body));
        let (at_lift, lift_here, floors) = match attends || bodiless {
            true => (false, false, Vec::new()),
            false => hosted.read(|w| w.lift_within(&body)),
        };
        // The verb-paths reachable from here — the live enum `invoke`'s `url` is
        // bound to, so the grammar forces a real address the device serves and a
        // character can never `invoke` a hallucinated one or a bare resource
        // (effector design §6; `Choices::InvokeUrl`).
        let invokable = self.invokable_urls(&hosted, &body);
        let base = tools::Within {
            company,
            people,
            heard_by_person,
            at_lift,
            lift_here,
            floors,
            invokable,
            // Never where it stands — see [`body::reachable`]. Walking to your
            // own room was refused, and refusal is not a lesson.
            places: match attends || bodiless {
                true => Vec::new(),
                false => body::reachable(&hosted, &body),
            },
            rooms: match bodiless {
                true => Vec::new(),
                false => body::rooms(&hosted, &body)
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect(),
            },
            parts: hosted.read(|w| {
                let Some(node) = w
                    .actor(&body)
                    .filter(|_| !bodiless)
                    .and_then(|a| w.node(&a.at))
                else {
                    return Vec::new();
                };
                w.map()
                    .parts_at(node)
                    .map(|(part, _)| part.id.clone())
                    .collect()
            }),
            // What this body has done too recently to do again, and what the
            // loop guard has struck to break a repetition. Both asked here, at
            // the one place a situation is composed, so no route can build a
            // grammar that has forgotten about either. The real-time body pace
            // and the turn-based loop break are two different rate limits; their
            // union is what the grammar is built without.
            //
            // **And `reflect`, while no dream can follow it.** A reflection is
            // what a dream grows from; with the character's next dream not due
            // (`dreams::DREAM_EVERY_MS`), or one being written, it would be
            // minutes of decode that lead nowhere, and the turn goes on with
            // the work instead.
            //
            // **A mind with no body neither reflects nor dreams.** It keeps
            // the watch: with nothing to do it rests, and time spent turning
            // the quiet over — "terrified that the silence means they have
            // stopped breathing" — was time a person's request waited behind.
            cooling: {
                let mut cooling = self.cooldowns.cooling(npc_id);
                cooling.extend(self.loop_guards.cooling(npc_id));
                if bodiless || !self.dream_available(npc_id) {
                    cooling.push(tools::REFLECT.name.to_string());
                }
                // **A mind with no body does not look about the world**: every
                // room is shown to it as it changes, and what is outside it is
                // reached through a terminal. Offered `scan`, Zen — connected
                // to the house network and asked what was on it — scanned its
                // own rooms four times and surveyed nothing.
                if bodiless {
                    cooling.extend(tools::WAYFINDING.iter().map(|a| a.to_string()));
                }
                cooling.sort_unstable();
                cooling.dedup();
                cooling
            },
            feelings: self.feelings.read().unwrap().clone(),
            me: hosted.read(|w| w.actor(&body).map(|a| a.name.clone()).unwrap_or_default()),
            // The station this body has claimed, named by what it binds or else
            // by the part itself. The orders it holds are the sim's, added below.
            held: hosted.read(|w| {
                w.actor(&body)
                    .and_then(|a| a.hold.as_ref())
                    .map(|h| h.subject.clone().unwrap_or_else(|| h.part.clone()))
                    .into_iter()
                    .collect()
            }),
            ..Default::default()
        }
        // What the house's bridges can work here, and what the carried mission
        // lets this body build (`docs/zen_moon_mind.md` §13–§15).
        .in_the_house(&hosted, &body)
        // And the acts that drive the mission's task, wherever it stands
        // (`docs/mission_templates.md` §7).
        .with_task_grants(&hosted, &body);
        // What the world's own state adds: what this body carries, what stands
        // in the room with it, what is outside. An empty answer to any of them
        // takes the acts that need it out of the grammar, which is how a world
        // without hostiles ends up without `engage`.
        let place = hosted.place_of(&body);
        let within = hosted.sim(|sim| base.clone().from_sim(sim, &body, &place));
        if !bodiless {
            return within;
        }
        // A mind with no body reaches real things through its terminals: the
        // three terminal acts, and — connected — the one terminal's verbs
        // (`docs/zen_everywhere.md` §5).
        let machines = self.machines();
        let on_cluster = self.standing.member().is_some();
        hosted.sim(|sim| within.at_terminals(sim, &machines, on_cluster, &body))
    }

    /// The verb-paths an `invoke` may act on from where `body` stands, for the
    /// grammar's `invoke` enum ([`tools::Choices::InvokeUrl`]).
    ///
    /// One per verb a served instance affords: `<instance-url>/<verb>`
    /// ([`address_of`], [`station::verbs_at`]), with the act each one runs,
    /// computed straight off the map. A part no route answers for (a seat, a
    /// blast door) is not listed, so `invoke` can only ever name a verb the world
    /// will actually serve — not a bare resource, not a missing verb.
    pub(super) fn invokable_urls(&self, hosted: &Arc<Hosted>, body: &str) -> Vec<Invokable> {
        let mut invoke = Vec::new();
        // Only what this body can do here — see [`mission_acts::offered`].
        let carrying = hosted.sim(|s| mission_acts::carrying(s, body));
        hosted.read(|w| {
            let Some(at) = w.actor(body).map(|a| a.at.clone()) else {
                return;
            };
            // **A character that takes no orders works the order table from
            // its seat, and nothing else there** (`docs/zen_everywhere.md` §4):
            // every table in its world is in reach, not only one in its room,
            // and no other station where its seat is.
            let here: Vec<_> = w
                .map()
                .instances_at(&at)
                .into_iter()
                .filter(|i| !carrying.unordered || COMMAND_DESK.contains(&i.part_id()))
                .collect();
            //
            // **And one carrying a mission can always give it up**: from
            // anywhere, a table's `report_stuck` is in reach, with why. A Drone
            // stuck where its step left it — Wick at the merge table, its brief
            // naming a publish its steps never reached — had no way to fail the
            // mission, so it stood there carrying it for hours and the one who
            // posted it never heard. Reporting it done stays at the table.
            let away_from_tables = carrying.unordered || (carrying.mission && !carrying.own);
            let mut tables = Vec::new();
            if away_from_tables {
                for area in w.map().areas() {
                    for node in &area.nodes {
                        let there = Where::new(area.id.as_str(), node.id.as_str());
                        if there == at {
                            continue;
                        }
                        tables.extend(
                            w.map()
                                .instances_at(&there)
                                .into_iter()
                                .filter(|i| COMMAND_DESK.contains(&i.part_id())),
                        );
                    }
                }
            }
            let reach = here
                .into_iter()
                .map(|i| (i, true))
                .chain(tables.into_iter().map(|i| (i, false)));
            for (inst, standing_here) in reach {
                let Some(url) = address_of(&inst) else {
                    continue;
                };
                for (verb, tool) in station::verbs_at(inst.part_id()) {
                    let in_reach = standing_here
                        || carrying.unordered
                        || tool.name == mission_acts::REPORT_STUCK.name;
                    if in_reach && mission_acts::offered(tool.name, carrying) {
                        invoke.push(Invokable::new(format!("{url}/{verb}"), tool.name));
                    }
                }
            }
        });
        invoke
    }

    pub(super) fn persona_of(&self, npc_id: u64) -> Option<OwnedPersona> {
        let who = {
            let g = self.persona.read().unwrap();
            g.as_ref().and_then(|f| f(npc_id))
        }?;
        Some(self.place_persona(npc_id, who))
    }

    /// [`Self::persona_of`] for an async caller. The persona resolver reads the
    /// app state's `tokio::sync::RwLock`s, so a character or reflection task
    /// runs it on the blocking pool rather than panicking on `blocking_read`. Placing the body afterwards reads only the
    /// runtime's own (non-async) world state, so it stays on the task.
    pub(crate) async fn persona_of_async(&self, npc_id: u64) -> Option<OwnedPersona> {
        let src = self.persona.read().unwrap().clone()?;
        let who = tokio::task::spawn_blocking(move || src(npc_id))
            .await
            .ok()
            .flatten()?;
        Some(self.place_persona(npc_id, who))
    }

    /// Put the resolved persona in the building its body stands in — which the
    /// authored record knows nothing about, because it comes from the map and
    /// the body. Only that one building: a world holds more than one, and a
    /// character told about all of them as "where you work" places itself in
    /// whichever it read most about. Reads the runtime's own world state, not
    /// the app-state locks the resolver did, so both `persona_of` paths share
    /// it.
    pub(super) fn place_persona(&self, npc_id: u64, mut who: OwnedPersona) -> OwnedPersona {
        if let Some((hosted, body)) = self.body_of(npc_id) {
            if let Some((key, text)) = self.building_of(&hosted, &body) {
                // A mind with no body is the place, not somebody in it.
                who.building = match hosted.sim(|s| s.missions.is_unordered(&body)) {
                    true => identity::whole_key(&key),
                    false => key,
                };
                who.place = text;
            }
            // The mission worked, with the rest of an agenda listed beneath it.
            who.mission = hosted.sim(|s| s.missions.agenda_prompt(&body));
            who.channels =
                hosted.sim(|s| channels::sections(&s.threads, &body, s.has_phone(&body)));
        }
        who.journal = self
            .scheduler
            .journal_of(npc_id, |j| j.prompt())
            .filter(|p| !p.is_empty());
        who
    }

    /// Every building of a world, as anyone living in it would describe it —
    /// [`describe::places`], keyed by part.
    ///
    /// Cached per world. Generating it is cheap but not free, and every
    /// character in a building would otherwise re-render the identical text on
    /// every conversation it opens.
    pub(super) fn places_of(&self, hosted: &Hosted) -> Vec<(String, String)> {
        if let Some(known) = self.places.lock().unwrap().get(hosted.id()) {
            return known.clone();
        }
        let parts = hosted.read(|w| describe::places(w.map()));
        self.places
            .lock()
            .unwrap()
            .insert(hosted.id().to_string(), parts.clone());
        parts
    }

    /// The building a body is standing in: its [`identity::building_key`] and
    /// what anyone who lives there knows of it. `None` for a body that is not
    /// in the world, or standing somewhere no part of the map describes.
    pub(super) fn building_of(&self, hosted: &Hosted, body: &str) -> Option<(String, String)> {
        let part = hosted.read(|w| {
            let area = w.actor(body)?.at.area.clone();
            describe::enclosing(w.map(), &area).map(str::to_string)
        })?;
        let text = self
            .places_of(hosted)
            .into_iter()
            .find(|(id, _)| *id == part)?
            .1;
        Some((identity::building_key(hosted.id(), &part), text))
    }

    /// Where a character is right now, as the world puts it: the room its body
    /// stands in and who is there. `None` for a character with no body.
    pub(super) fn situation_of(&self, npc_id: u64) -> Option<String> {
        let (hosted, body) = self.body_of(npc_id)?;
        let seen = hosted.read(|w| perceive::percept(w, &body));
        (!seen.trim().is_empty()).then_some(seen)
    }
}
