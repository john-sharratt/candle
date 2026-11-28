//! Missions: what a character is handed, what it is told about it, and when it
//! reports.
//!
//! A mission is given to a character where it stands rather than fetched, so a
//! free character is at rest until work is posted and a character already
//! carrying one is not handed another. The compass a character is told names its
//! next step and the way to the room or table that step needs. A character at the
//! command table with its work done is told to report, on every tick it stands
//! there, and only then. Missions and working copies outlive a restart through
//! the world's state.

use super::*;

impl Runtime {
    /// What a character **standing at the command table** with its mission's
    /// work done is told: report it, here, now — `None` otherwise.
    ///
    /// The tick loop appends this to *every* tick's perception, because a
    /// character is at the table for only a turn or two before it drifts off
    /// again, and the report is filed nowhere but the table: one that came
    /// back with everything done and read only its brief walked back out to
    /// begin again. The window is short by nature — it stops the moment the
    /// report is filed.
    pub fn at_table_report(&self, npc_id: u64) -> Option<String> {
        let (hosted, body) = self.body_of(npc_id)?;
        // An unordered character standing at the table is only standing there.
        if !self.at_command_table(npc_id) || hosted.sim(|s| s.missions.is_unordered(&body)) {
            return None;
        }
        // **"Report it now" only when the report will be taken.** A Maker told
        // every tick that all but the report was done, and refused at every
        // report by the quality gate, went round until it said it wanted to
        // scream. With its work done but the document not yet able to stand,
        // it is told what is in the way instead.
        let work_done = hosted.sim(|s| {
            s.missions
                .active(&body)
                .is_some_and(|m| !m.todo.is_empty() && m.next_step().is_none_or(|t| t.reports))
        });
        if work_done {
            if let Some(block) = work::report_block(&hosted, &body) {
                return Some(block);
            }
        }
        hosted.sim(|s| match s.missions.active(&body) {
            None => None,
            // A mission carried on its ask alone has no steps to have done, so
            // it is not told they are.
            Some(m)
                if !m.todo.is_empty()
                    && m.next_step().is_none_or(|t| t.reports)
                    && m.may_reject() =>
            {
                Some(VERDICT_AT_THE_TABLE.to_string())
            }
            Some(m) => (!m.todo.is_empty() && m.next_step().is_none_or(|t| t.reports)).then(|| {
                match m.observed.is_empty() {
                    true => REPORT_AT_THE_TABLE.to_string(),
                    false => format!(
                        "{REPORT_AT_THE_TABLE} What you saw on it: {}.",
                        m.observed.join("; ")
                    ),
                }
            }),
        })
    }

    /// Sign off what this character's mission asked of the room it is standing
    /// in — the journey to it, and the reading of any machine standing in it —
    /// and say so, with the readings and what comes next. `None` when nothing was
    /// signed off: no mission, nothing asked of this room, or already ticked.
    ///
    /// **Standing in the room is the reading.** The situation a character reads
    /// on arrival already lists every machine within reach with its state —
    /// "breaker panel (closed), coolant valve (weeping)" — so a mission to go and
    /// read one is met by being there. Waiting for a `scan` the character had no
    /// reason to make, with the answer already in front of it, left it walking
    /// between the room and the table until it gave the mission up as stuck.
    pub fn mark_arrival(&self, npc_id: u64) -> Option<String> {
        let (hosted, body) = self.body_of(npc_id)?;
        hosted.with_both(|w, s| {
            let at = w.actor(&body).map(|a| a.at.clone())?;
            let room = w.node(&at).map(|n| n.name.clone())?;
            let place = format!("{}/{}", at.area, at.node);
            let here: Vec<(String, String)> = s
                .devices
                .iter()
                .filter(|d| d.at == place)
                .map(|d| (d.name.clone(), d.mode.clone()))
                .collect();
            let wanted: Vec<String> = s
                .missions
                .active(&body)?
                .todo
                .iter()
                .filter(|t| !t.done)
                .map(|t| t.text.clone())
                .collect();
            let level = w
                .map()
                .get(&at.area)
                .map(|a| a.name.clone())
                .unwrap_or_default();
            let arrived = s.missions.arrived_in(&body, &room, &level);
            let names: Vec<String> = here.iter().map(|(n, _)| n.clone()).collect();
            let read = s.missions.read_off(&body, &names);
            // Whoever is in the room with it: a step to find them is met.
            let company: Vec<String> = w
                .actors_at(&at)
                .into_iter()
                .filter(|other| other.id != body)
                .map(|other| other.name.clone())
                .collect();
            let found: Vec<String> = company
                .iter()
                .filter(|who| s.missions.met(&body, who))
                .cloned()
                .collect();
            if !arrived && !read && found.is_empty() {
                return None;
            }
            // The readings the mission asked for, in the words of the situation,
            // and kept as what the engine saw.
            let asked: Vec<&(String, String)> = here
                .iter()
                .filter(|(name, _)| {
                    wanted
                        .iter()
                        .any(|step| step.to_lowercase().contains(&name.to_lowercase()))
                })
                .collect();
            let readings: Vec<String> = asked
                .iter()
                .map(|(name, mode)| format!("{name} here reads {mode}"))
                .collect();
            for (name, mode) in &asked {
                s.missions
                    .observe(&body, format!("in {room}, {name}: {mode}"));
            }
            for who in &found {
                s.missions.observe(&body, format!("was with {who} in {room}"));
            }
            // Who was there when it got there — the answer to "go and see who is
            // in…", and evidence an operator can check the report against.
            if arrived {
                let there = match company.is_empty() {
                    true => "nobody else".to_string(),
                    false => npc_map::text::list(&company),
                };
                s.missions.observe(&body, format!("in {room}: {there}"));
            }
            let lead = match (readings.is_empty(), found.is_empty()) {
                (true, true) => format!("You are in {room}"),
                (false, _) => format!("You are in {room}, and {}", npc_map::text::list(&readings)),
                (true, false) => format!(
                    "{} {} here with you",
                    npc_map::text::list(&found),
                    npc_map::text::is_are(found.len())
                ),
            };
            let next = s.missions.active(&body).and_then(Mission::next_step);
            tracing::info!(npc = %body, %room, arrived, read, "mission steps signed off on arrival");
            Some(progress_line(&lead, next, &names))
        })
    }

    /// What the order table holds of this character's posting
    /// ([`orders_line`]) — `None` for one that takes orders rather than gives
    /// them, which posts nothing.
    pub fn orders_out(&self, npc_id: u64) -> Option<String> {
        let (hosted, body) = self.body_of(npc_id)?;
        let out = hosted.sim(|s| {
            s.missions
                .is_unordered(&body)
                .then(|| s.missions.out_for(&body))
        })?;
        let named: Vec<(String, Option<String>)> = out
            .into_iter()
            .map(|(brief, with)| {
                let with = with.map(|b| {
                    hosted
                        .read(|w| w.actor(&b).map(|a| a.name.clone()))
                        .unwrap_or(b)
                });
                (brief, with)
            })
            .collect();
        let post = self
            .invokable_urls(&hosted, &body)
            .into_iter()
            .find(|i| i.act == mission_acts::ORDER_POST.name)
            .map(|i| i.url);
        Some(orders_line(&named, post.as_deref()))
    }

    /// The mission compass for this character as things stand — its mission's
    /// next step and where whoever it names is now, or the way to the table to
    /// report when that is all that is left. `None` with no mission, or one
    /// carried on its ask alone.
    pub fn mission_compass(&self, npc_id: u64) -> Option<String> {
        let (hosted, body) = self.body_of(npc_id)?;
        let at_table = self.at_command_table(npc_id);
        // The desk's own addresses, for a step about a document — read before
        // the world is borrowed below, which this also reads.
        let urls = self.invokable_urls(&hosted, &body);
        let url_of = |act: &str| urls.iter().find(|i| i.act == act).map(|i| i.url.clone());
        let desk = DeskVerbs {
            read: url_of("file_read"),
            compose: url_of("compose"),
            edit: url_of("file_edit"),
            commit: url_of("bench_commit"),
        };
        let (merge_status, merge_publish) = (url_of("merge_status"), url_of("merge_publish"));
        hosted.read_both(|w, s| {
            // Riding the lift, there is nothing to do but arrive: told the way
            // from the landing it is still standing on, a rider tried to call
            // the car it was already in.
            if w.riding(&body).is_some() {
                return None;
            }
            let mission = s.missions.active(&body)?;
            let next = mission.next_step()?.clone();
            let here = w.actor(&body)?.at.clone();
            // **A body that moves by attention is told to `focus`.** It has no
            // `move_to` and no lift, so a way given in those verbs was a way it
            // could not take: Zen, told to ride the lift to the dispatch room,
            // decided the lift was locked and spent a night asking the Drones
            // to override it.
            let attends = s.attends(&body);
            // A mind that takes no orders goes nowhere: a place its mission
            // names is named, with no way to it.
            let bodiless = s.missions.is_unordered(&body);
            // The way to a place, in the grammar's verbs: one `move_to`, by the
            // name the grammar offers it under, wherever it is — the world
            // rides the lift for a journey that changes floor.
            let way_to = |to: &Where| -> Option<String> {
                let room = w.node(to)?.name.clone();
                if bodiless {
                    return Some(room);
                }
                if attends {
                    return Some(format!("{room}: `focus` on it"));
                }
                let name = body::destination_name(w, &here, to)?;
                Some(match to.area == here.area {
                    true => format!("{room}: `move_to` it"),
                    false => {
                        let on = w.map().get(&to.area)?.name.clone();
                        format!("{room}, on {on}: `move_to` {name}")
                    }
                })
            };
            let reach: Vec<Where> = std::iter::once(here.clone())
                .chain(route::reachable_from(w.map(), &here))
                .collect();
            let place = |at: &Where| format!("{}/{}", at.area, at.node);
            // A publish step is done when its work lands, which the report
            // alone cannot do: the way is to the merge table, not the order
            // table that would refuse it.
            if next.reports && work::unpublished(s, &body).is_some() {
                let doc = mission.work.as_ref()?.writes.clone();
                let way = merge_publish
                    .is_none()
                    .then(|| {
                        reach
                            .iter()
                            .find(|at| s.part_offers(&place(at), "merge_publish"))
                            .and_then(&way_to)
                    })
                    .flatten();
                return Some(publish_compass(
                    &doc,
                    merge_status.as_deref(),
                    merge_publish.as_deref(),
                    way.as_deref(),
                ));
            }
            if next.reports {
                // The way back to the table, lift and all: told only the room, a
                // character that had done its reading three floors down rode
                // nowhere and paced the level it was on for twenty minutes.
                let tables: Vec<&Where> = reach
                    .iter()
                    .filter(|at| s.part_offers(&place(at), mission_acts::REPORT_DONE.name))
                    .collect();
                let table = tables
                    .iter()
                    .find(|at| at.area == here.area)
                    .or_else(|| tables.first())
                    .and_then(|at| way_to(at));
                return Some(compass_report_line(
                    table.as_deref(),
                    at_table,
                    &mission.observed,
                    mission.may_reject(),
                ));
            }
            if let Some(year) = time_step(&next.text) {
                let has_machine = |at: &Where| {
                    w.node(at).is_some_and(|n| {
                        w.map()
                            .parts_at(n)
                            .any(|(part, _)| part.id == "time-machine")
                    })
                };
                let here_has = has_machine(&here);
                let way = (!here_has)
                    .then(|| reach.iter().find(|at| has_machine(at)).and_then(&way_to))
                    .flatten();
                return Some(time_compass(year, here_has, way.as_deref()));
            }
            if let Some(doc) = DocStep::of(&next.text) {
                // The nearest desk that does this step, when this is not one. A
                // piece is written at a desk that composes: standing at a
                // terminal that only reads, a Maker was told "at a desk, `scan`
                // shows its address", scanned the terminal it stood at, and
                // never found the desk it needed.
                // A document whose text the brief gives is written as given,
                // with `file_write`, not composed.
                let given = mission.writes_given();
                let write = url_of("file_write");
                let (act, here_does) = match &doc {
                    DocStep::Read(_) => ("file_read", desk.read.is_some()),
                    DocStep::Write(_) if given => ("file_write", write.is_some()),
                    DocStep::Write(_) => ("compose", desk.compose.is_some()),
                };
                // Never the room it stands in: this one does not serve the act
                // from where the body is, whatever its parts list.
                let way = (!here_does)
                    .then(|| {
                        reach
                            .iter()
                            .filter(|at| **at != here)
                            .find(|at| s.part_offers(&place(at), act))
                            .and_then(&way_to)
                    })
                    .flatten();
                if let (DocStep::Write(p), true) = (&doc, given) {
                    return Some(given_compass(
                        p,
                        write.as_deref(),
                        desk.commit.as_deref(),
                        way.as_deref(),
                        mission.gate_refusal.as_deref(),
                    ));
                }
                if let (DocStep::Write(p), Some(refusal)) = (&doc, &mission.gate_refusal) {
                    return Some(refused_compass(p, refusal, &desk, way.as_deref()));
                }
                let written = written_up_to_floor(s, &body, mission, &doc);
                return Some(doc_compass(&doc, &desk, way.as_deref(), written));
            }
            // A critical step is not done until an authorizer approves: the
            // compass tells the carrier to seek the go-ahead, not to do it.
            if next.critical {
                return Some(critical_compass(&next.text, s.missions.authorizers()));
            }
            // The way to the room or machine a step names, when it names one
            // somewhere other than here.
            let way_of = |step: &str| {
                Aim::of(step).and_then(|aim| {
                    let target = reach
                        .iter()
                        .find(|at| {
                            let level = w.map().get(&at.area).map(|a| a.name.as_str());
                            w.node(at)
                                .is_some_and(|n| aim.is_room(&n.name, level.unwrap_or_default()))
                        })
                        .or_else(|| {
                            let d = s.devices.iter().find(|d| aim.is_machine(&d.name))?;
                            reach.iter().find(|at| place(at) == d.at)
                        })?;
                    (target != &here).then(|| way_to(target)).flatten()
                })
            };
            let way = way_of(&next.text);
            // **A judgement about somebody is not a walk to them.** Told where
            // Jitter was while noting what Jitter needed, a Drone that already
            // had the answer followed Jitter between the lift and the room.
            let judging = is_judgement(&next.text);
            let people: Vec<(String, Option<String>)> = w
                .actors()
                .filter(|a| a.id != body && !judging)
                .map(|a| {
                    let room = (a.at != here).then(|| way_to(&a.at)).flatten();
                    (a.name.clone(), room)
                })
                .collect();
            let way = way.map(|w| format!("It is in {w}."));
            let mut line = compass_line(&next, &people, way.as_deref());
            if judging {
                line.push(' ');
                match mission.step_to_do() {
                    None => line.push_str(JUDGEMENT_LINE),
                    Some(later) => {
                        line.push_str(&judgement_then(&later.text, way_of(&later.text).as_deref()))
                    }
                }
            }
            // **Work is posted from the world's catalogue.** Told only "post work
            // on the order table", Zen never saw the jobs it could post by name
            // and wrote its own instead — a crisis a turn, each one a template
            // the Drones were set to write.
            let posts = next.text.to_lowercase().starts_with("post work");
            if posts && s.missions.is_unordered(&body) {
                line.push(' ');
                line.push_str(&post_work_line(
                    s.missions.routines(),
                    url_of("order_post").as_deref(),
                ));
            }
            Some(line)
        })
    }

    /// The chapter this character's turn is taken in — how many missions it has
    /// closed — closing the last one in its journal first when a mission has
    /// been handed in since its last turn. `0` for a character with no body.
    ///
    /// **The work, carried across.** The turn after a hand-in opens a new
    /// conversation (`Minds::think`); the journal is what crosses with it, so the
    /// mission just closed is written into it — by the engine, from the
    /// mission's record — before the turn's persona reads it.
    pub fn turn_chapter(&self, npc_id: u64, world_ms: u64) -> u64 {
        let Some((hosted, body)) = self.body_of(npc_id) else {
            return 0;
        };
        let (chapter, closed) = hosted.sim(|s| {
            (
                s.missions.chapter(&body),
                s.missions.last_closed(&body).cloned(),
            )
        });
        let before = self.chapters.lock().unwrap().insert(npc_id, chapter);
        let turned = before.is_some_and(|b| b < chapter);
        if let (true, Some(mission)) = (turned, closed) {
            if self
                .scheduler
                .journal_close_chapter(npc_id, &mission, world_ms)
                .is_some()
            {
                tracing::info!(npc_id, chapter, "journal: a mission's chapter closed");
            }
        }
        chapter
    }

    /// The mission compass, when it says something this character has not
    /// already been told. Recorded as told either way.
    pub fn compass_news(&self, npc_id: u64) -> Option<String> {
        let now = self.mission_compass(npc_id);
        let mut told = self.compass_told.lock().unwrap();
        match now {
            Some(line) if told.get(&npc_id) != Some(&line) => {
                told.insert(npc_id, line.clone());
                Some(line)
            }
            Some(_) => None,
            None => {
                told.remove(&npc_id);
                None
            }
        }
    }

    /// Mark every character as its personality now says — `takes_orders`,
    /// `moves`, `idle:` and `reaches:` read afresh — after the personalities are read
    /// again. A character is marked as it is embodied, and one already standing
    /// in its world is not embodied again, so without this an edited
    /// personality reached its characters only at the next restart. The world's
    /// state is its authority's to mark; the idle thought is every member's.
    pub async fn mark_orders(&self) {
        for world in self.hosted.ids() {
            self.mark_orders_in(&world).await;
        }
    }

    /// [`Self::mark_orders`], for the characters of world `world` alone.
    ///
    /// **Also run when this member takes the world's authority.** A character
    /// is marked as it is embodied only by the world's authority, and a member
    /// starts as a replica: the cast it embodies at startup goes unmarked there,
    /// and the lease it takes a moment later is what makes it the authority.
    /// Without this a character whose personality takes no orders kept the
    /// mission it carried — Zen worked a Drone's routine for hours after
    /// `takes_orders: false` was written.
    pub async fn mark_orders_in(&self, world: &str) {
        let Some(hosted) = self.hosted.get(world) else {
            return;
        };
        let writes = self.standing.writes(world);
        for (npc_id, body) in self.bodies.in_world(world) {
            // A character whose persona cannot be read now keeps its marks.
            let Some(p) = self.persona_of_async(npc_id).await else {
                continue;
            };
            self.set_idle(npc_id, p.idle.as_deref());
            self.set_reaches(npc_id, &p.reaches);
            if !writes {
                continue;
            }
            let stands = hosted.sim(|s| (s.missions.is_unordered(&body), s.attends(&body)));
            if stands != (p.unordered, p.attends) {
                hosted.with_sim(|s| {
                    if s.missions.is_unordered(&body) != p.unordered {
                        s.set_unordered(&body, p.unordered);
                    }
                    s.set_attends(&body, p.attends);
                });
            }
        }
    }

    /// Keep `npc_id`'s idle thought, or forget it.
    pub(super) fn set_idle(&self, npc_id: u64, idle: Option<&str>) {
        let mut idles = self.idles.write().unwrap();
        match idle.map(str::trim).filter(|t| !t.is_empty()) {
            Some(text) => idles.insert(npc_id, text.to_string()),
            None => idles.remove(&npc_id),
        };
    }

    /// Keep what `npc_id` reaches beyond its world; none, forget it.
    pub(crate) fn set_reaches(&self, npc_id: u64, reaches: &[Group]) {
        let mut all = self.reaches.write().unwrap();
        match reaches.is_empty() {
            true => all.remove(&npc_id),
            false => all.insert(npc_id, reaches.to_vec()),
        };
    }

    /// What `npc_id` reaches beyond its world (`docs/zen_moon_mind.md` §16).
    pub fn reaches_of(&self, npc_id: u64) -> Vec<Group> {
        self.reaches
            .read()
            .unwrap()
            .get(&npc_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Whether the world gives this character orders: every character but one
    /// whose personality says `takes_orders: false` — see
    /// [`crate::sim::missions::Missions::set_unordered`]. A character with no
    /// body has no table to be called to, and is not one of those.
    pub fn takes_orders(&self, npc_id: u64) -> bool {
        self.body_of(npc_id)
            .is_none_or(|(hosted, body)| !hosted.sim(|s| s.missions.is_unordered(&body)))
    }

    /// Whether this character stands where a mission is reported — i.e. the
    /// order table's `report_done` is among the acts the room offers it.
    pub(super) fn at_command_table(&self, npc_id: u64) -> bool {
        let Some((hosted, body)) = self.body_of(npc_id) else {
            return false;
        };
        let place = hosted.place_of(&body);
        hosted.sim(|s| {
            s.station_tools(&place)
                .iter()
                .any(|t| t == mission_acts::REPORT_DONE.name)
        })
    }

    /// Whether `body` can get to a command table from where it stands: the
    /// room it is in, or any room the map's doors, portals and lifts lead on
    /// to, offers `report_done`.
    ///
    /// `false` for a world with no order table, or a building cut off from the
    /// one that has it, so no mission is given to a body with nowhere to
    /// report it.
    pub(super) fn reaches_table(hosted: &Hosted, body: &str) -> bool {
        hosted
            .with_both(|w, s| {
                let here = w.actor(body).map(|a| a.at.clone())?;
                let offers = |at: &Where| {
                    s.part_offers(
                        &format!("{}/{}", at.area, at.node),
                        mission_acts::REPORT_DONE.name,
                    )
                };
                Some(offers(&here) || route::reachable_from(w.map(), &here).iter().any(offers))
            })
            .unwrap_or(false)
    }

    /// Open or shut the command table on every world this daemon runs.
    ///
    /// The state [`Self::assign_work`] and [`Self::follow_rest`] read: an open
    /// table gives work, a shut one gives none and puts nobody at rest.
    /// Opening it does not itself announce anything —
    /// [`Self::say_to_all_channels`] and the tannoy do.
    ///
    /// The flag is world state: each world's authority writes it with the rest
    /// of the world's records on its next beat, so a reboot, and every member
    /// following the world, finds it as it was left. A world this daemon only
    /// follows is its authority's to open or shut.
    pub fn set_table_open(&self, open: bool) {
        for id in self.hosted.ids() {
            if !self.standing.writes(&id) {
                continue;
            }
            if let Some(hosted) = self.hosted.get(&id) {
                hosted.with_sim(|s| s.table_open = open);
            }
        }
    }

    /// Cancel a character's open mission, if it has one, and tell it so
    /// ([`MISSION_WITHDRAWN`]) at `world_ms`. Returns whether one was cancelled —
    /// it is called off, not reported, so it is not kept. Without the notice the
    /// character keeps working from its history and its own thread what the
    /// mission was, long after the mission is gone.
    pub fn cancel_mission(&self, npc_id: u64, world_ms: u64) -> bool {
        let Some((hosted, body)) = self.body_of(npc_id) else {
            return false;
        };
        let title = hosted.sim(|s| s.missions.active(&body).map(|m| m.mission_text()));
        let cancelled = hosted.with_sim(|s| s.missions.cancel(&body));
        if cancelled {
            let line = match title {
                Some(t) => format!("Your mission was withdrawn: {}", first_sentence(&t)),
                None => "Your mission was withdrawn.".to_string(),
            };
            self.changed(&hosted, Change::Mission { body, line });
            self.tell(
                npc_id,
                world_ms,
                Salience::URGENT,
                EventKind::Nudge {
                    text: MISSION_WITHDRAWN.to_string(),
                },
            );
        }
        cancelled
    }

    /// Sign off `npc_id`'s critical step `step`, which `by` — an account
    /// handle, `name` by name — said yes to on a conversation that asked them
    /// (`docs/npc_conversations.md` §7.2). Refused unless `step` is still the
    /// mission's next and `by` is one of the world's authorizers. The
    /// character is told, and its journal keeps it.
    pub fn authorize_step(&self, npc_id: u64, step: &str, by: &str, name: &str) -> bool {
        let Some((hosted, body)) = self.body_of(npc_id) else {
            return false;
        };
        let signed = hosted.with_sim(|s| {
            let authorizer = authorize::is_authorizer(by, s.missions.authorizers());
            let Some(m) = s.missions.active_mut(&body) else {
                return false;
            };
            authorizer && m.critical_next() == Some(step) && m.authorize()
        });
        if signed {
            let applied = format!("{name} signed off '{}'.", step.trim_end_matches('.'));
            self.journals
                .note(hosted.id(), &body, ExperienceKind::Mission, &applied, None);
            self.tell(
                npc_id,
                self.world_ms(npc_id),
                Salience::URGENT,
                EventKind::Concluded {
                    party: name.to_string(),
                    applied,
                },
            );
        }
        signed
    }

    /// Cancel the open mission of every character in every world this daemon
    /// runs, each noticed at the world time `world_ms` gives for it. Returns
    /// how many were cancelled.
    pub fn cancel_all_missions(&self, world_ms: impl Fn(u64) -> u64) -> usize {
        self.hosted
            .ids()
            .into_iter()
            .filter(|id| self.standing.writes(id))
            .flat_map(|id| self.bodies.in_world(&id))
            .filter(|(npc_id, _)| self.cancel_mission(*npc_id, world_ms(*npc_id)))
            .count()
    }

    /// Where this character's mission stands — which one it carries, and how
    /// many of its steps are done — or `None` with none. Two reads either side
    /// of an act that differ mean the act moved the mission: a step signed off,
    /// a mission taken up or handed in.
    pub(crate) fn mission_mark(&self, npc_id: u64) -> Option<(String, usize)> {
        let (hosted, body) = self.body_of(npc_id)?;
        hosted.sim(|s| {
            s.missions
                .active(&body)
                .map(|m| (m.prompt.clone(), m.todo.iter().filter(|t| t.done).count()))
        })
    }
}
