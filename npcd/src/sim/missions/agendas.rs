//! Keeps each body's agenda: the missions it has queued, the one it is working now,
//! and the order the rest are worked in.
//!
//! A mission that outranks the one being worked takes its place at once, and the
//! one it replaces is set aside with its steps as they stand. Steps the world sees
//! done are signed off as they happen, so only a step the engine cannot see is
//! closed by the agenda's own judgment. A critical step is signed only by an
//! authorizer, never by a body saying that its mission is done. An agenda closes
//! each mission once all of its steps are signed off, with no report needed.

use super::*;

impl Missions {
    /// A place on an agenda for a mission coming onto it now.
    pub(super) fn agendum(&mut self, title: &str, priority: Priority) -> Agendum {
        self.agenda_count += 1;
        Agendum {
            title: title.trim().to_string(),
            priority,
            since: self.agenda_count,
        }
    }

    /// Every mission on `body`'s agenda: the one worked first, then the rest
    /// by rank.
    pub fn agenda(&self, body: &str) -> Vec<&Mission> {
        self.active
            .get(body)
            .into_iter()
            .chain(self.held.get(body).into_iter().flatten())
            .collect()
    }

    /// The titles of `body`'s agenda — what its agenda acts name a mission
    /// by.
    pub fn agenda_titles(&self, body: &str) -> Vec<String> {
        self.agenda(body)
            .into_iter()
            .filter_map(|m| m.agendum.as_ref().map(|a| a.title.clone()))
            .collect()
    }

    /// Put `mission` on `body`'s agenda as `title` at `priority`: worked at
    /// once when it outranks the one being worked, which is set aside with
    /// its steps as they stand.
    ///
    /// Refused when the agenda is full ([`KEEP`]), naming the lowest to finish
    /// or drop first, or when a mission there is already called `title`.
    pub fn agenda_add(
        &mut self,
        body: &str,
        mission: Mission,
        title: &str,
        priority: Priority,
    ) -> Result<(), String> {
        let titles = self.agenda_titles(body);
        if titles.iter().any(|t| t.eq_ignore_ascii_case(title.trim())) {
            return Err(format!(
                "Your agenda already has a mission called \"{}\": change it with \
                 `mission_edit`, or name this one differently.",
                title.trim()
            ));
        }
        if self.agenda(body).len() >= KEEP {
            let lowest = self
                .agenda(body)
                .into_iter()
                .filter_map(|m| m.agendum.as_ref())
                .min_by_key(|a| (a.priority, Reverse(a.since)))
                .map(|a| a.title.clone())
                .unwrap_or_default();
            return Err(format!(
                "Your agenda is full — {KEEP} missions. Finish or drop one first: the lowest is \
                 \"{lowest}\"."
            ));
        }
        let agendum = self.agendum(title, priority);
        self.held
            .entry(body.to_string())
            .or_default()
            .push(mission.on_agenda(agendum));
        self.agenda_rank(body);
        Ok(())
    }

    /// Change the mission on `body`'s agenda called `title`: its brief, its
    /// steps — those it had signed off that it keeps stay signed off — or its
    /// priority, worked at once if it now outranks the one being worked.
    pub fn agenda_edit(
        &mut self,
        body: &str,
        title: &str,
        brief: Option<String>,
        steps: Option<Vec<String>>,
        priority: Option<Priority>,
    ) -> Result<(), String> {
        let m = self.agenda_mut(body, title)?;
        if let Some(brief) = brief {
            m.prompt = brief;
        }
        if let Some(steps) = steps {
            m.todo = kept_steps(&m.todo, steps);
        }
        if let (Some(p), Some(a)) = (priority, m.agendum.as_mut()) {
            a.priority = p;
        }
        self.agenda_rank(body);
        Ok(())
    }

    /// Close the mission on `body`'s agenda called `title` as done, on
    /// `found` — the agenda's own close, for a mission whose steps are Zen's to
    /// judge.
    pub fn agenda_done(&mut self, body: &str, title: &str, found: &str) -> Result<Mission, String> {
        // **A critical step is signed by an authorizer, never by saying the
        // mission is done** (`docs/npc_conversations.md` §7.1).
        let unsigned = self
            .agenda(body)
            .into_iter()
            .find(|m| {
                m.agendum
                    .as_ref()
                    .is_some_and(|a| a.title.eq_ignore_ascii_case(title))
            })
            .and_then(|m| m.todo.iter().find(|t| t.critical && !t.done))
            .map(|t| t.text.clone());
        if let Some(step) = unsigned {
            return Err(format!(
                "\"{title}\" waits on an authorizer's yes to \"{step}\": it is not done until \
                 they sign it."
            ));
        }
        // **A step the world sees done is not done by saying so.** A journey or a
        // reading is signed off as it happens; only a step the engine cannot see
        // is the agenda's to judge. Asked to have a Drone read a panel, Zen put
        // "go to the watch room" on its own agenda from a conversation, closed it
        // a turn later with "the sensor panel is active and reading stable", and
        // told the Creator it had read it — of a panel nobody had looked at.
        let unseen = self
            .agenda(body)
            .into_iter()
            .find(|m| {
                m.agendum
                    .as_ref()
                    .is_some_and(|a| a.title.eq_ignore_ascii_case(title))
            })
            .and_then(|m| {
                m.todo
                    .iter()
                    .find(|t| !t.done && !t.reports && Aim::of(&t.text).is_some())
            })
            .map(|t| t.text.clone());
        if let Some(step) = unseen {
            return Err(format!(
                "\"{title}\" is not done: \"{step}\" is a step the world sees done, and it has \
                 not seen it. Do it — `focus` on the room, and its machines are read as your \
                 attention arrives — or, if it is a Drone's to do, post it at the order table \
                 (`order_post`), and it is done when they report it."
            ));
        }
        self.agenda_work(body, title)?;
        self.report(body, Outcome::Pass, found, None)
            .ok_or_else(|| format!("You have no mission called \"{title}\"."))
    }

    /// Drop the mission on `body`'s agenda called `title`, for `why`.
    pub fn agenda_cancel(&mut self, body: &str, title: &str, why: &str) -> Result<Mission, String> {
        self.agenda_work(body, title)?;
        let m = self
            .active
            .get(body)
            .cloned()
            .ok_or_else(|| format!("You have no mission called \"{title}\"."))?;
        self.cancel_for(body, why);
        let mut m = m;
        m.complete(Outcome::Fail, why, None);
        Ok(m)
    }

    /// The mission on `body`'s agenda called `title`, wherever it stands.
    pub(super) fn agenda_mut(&mut self, body: &str, title: &str) -> Result<&mut Mission, String> {
        let named = |m: &Mission| {
            m.agendum
                .as_ref()
                .is_some_and(|a| a.title.eq_ignore_ascii_case(title.trim()))
        };
        if self.active.get(body).is_some_and(named) {
            return Ok(self.active.get_mut(body).expect("just found"));
        }
        self.held
            .get_mut(body)
            .and_then(|h| h.iter_mut().find(|m| named(m)))
            .ok_or_else(|| format!("You have no mission called \"{}\".", title.trim()))
    }

    /// Make the mission on `body`'s agenda called `title` the one worked, the
    /// one it replaces set aside.
    pub(super) fn agenda_work(&mut self, body: &str, title: &str) -> Result<(), String> {
        self.agenda_mut(body, title)?;
        let held = self.held.entry(body.to_string()).or_default();
        if let Some(at) = held.iter().position(|m| {
            m.agendum
                .as_ref()
                .is_some_and(|a| a.title.eq_ignore_ascii_case(title.trim()))
        }) {
            let m = held.remove(at);
            if let Some(was) = self.active.insert(body.to_string(), m) {
                self.held.entry(body.to_string()).or_default().push(was);
            }
        }
        Ok(())
    }

    /// Put `body`'s agenda in order: the highest-ranked mission worked, the
    /// rest held behind it by rank — priority, then oldest first.
    pub(super) fn agenda_rank(&mut self, body: &str) {
        let mut all: Vec<Mission> = self.active.remove(body).into_iter().collect();
        all.extend(self.held.remove(body).unwrap_or_default());
        if all.is_empty() {
            return;
        }
        all.sort_by_key(|m| m.agendum.as_ref().map(|a| (Reverse(a.priority), a.since)));
        let mut all = all.into_iter();
        if let Some(top) = all.next() {
            self.active.insert(body.to_string(), top);
        }
        let rest: Vec<Mission> = all.collect();
        if !rest.is_empty() {
            self.held.insert(body.to_string(), rest);
        }
    }

    /// The agenda's prompt section for `body`: the mission worked, as any
    /// carried mission reads, and the rest listed beneath it — `None` with
    /// nothing carried.
    pub fn agenda_prompt(&self, body: &str) -> Option<MissionPrompt> {
        let worked = self.active.get(body)?.prompt();
        let held = self.held.get(body).filter(|h| !h.is_empty());
        let Some(held) = held else {
            return Some(worked);
        };
        let lines: Vec<String> = held
            .iter()
            .filter_map(|m| {
                let a = m.agendum.as_ref()?;
                let next = m.next_step().map_or("every step done".to_string(), |t| {
                    format!("next: {}", t.text.trim())
                });
                Some(format!("- {} ({}) — {next}", a.title, a.priority.as_str()))
            })
            .collect();
        let text = format!(
            "{}\nAlso on your agenda, set aside until this is done:\n{}",
            worked.text,
            lines.join("\n")
        );
        let mut h = DefaultHasher::new();
        text.hash(&mut h);
        Some(MissionPrompt {
            text,
            fingerprint: h.finish(),
        })
    }

    /// Close every mission worked on an agenda whose steps are all signed off,
    /// as done — an agenda is closed by what the engine saw, with no report
    /// (`docs/zen_moon_mind.md` §17.5). Returns each body closed, the mission's
    /// title, and how.
    pub fn settle_agendas(&mut self) -> Vec<(String, String, Outcome)> {
        let closing: Vec<(String, String, Outcome)> = self
            .active
            .iter()
            .filter(|(_, m)| m.all_todos_done())
            .filter_map(|(body, m)| {
                let title = m.agendum.as_ref()?.title.clone();
                Some((body.clone(), title, Outcome::Pass))
            })
            .collect();
        for (body, _, outcome) in &closing {
            self.report(body, *outcome, "every step was seen done", None);
        }
        closing
    }
}
