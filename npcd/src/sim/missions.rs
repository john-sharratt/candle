//! Which character is carrying which mission — the world's half of the mission
//! system.
//!
//! The pure [`Mission`](crate::engine::mission::Mission) model owns what a
//! mission *is*: the ask, its steps, the answer it builds, the report that
//! closes it. This owns *whose* it is, so it lives on [`Sim`](crate::sim::Sim),
//! where an act at the command desk can reach it through `with_sim`, where the
//! standing-task nudge can read it, and where it is serialized and persisted
//! with the rest of the world.
//!
//! A body carries at most one open mission at a time — the one it reads each
//! quiet turn and works through. Missions lodged for a body it has not collected
//! wait in a queue, oldest drawn first. A mission it has reported on is kept in
//! `done` so an operator can still read the outcome and answer after the
//! character has moved on to the next.
//!
//! The command table's generator (`engine::mission_gen`) keeps a **pool** here
//! of the missions it has written, which anybody collecting draws after its own
//! lodged ones and before the routine bank, and a **ledger** of every target it
//! has found — in hand, done, stuck, or found to need nothing — so the same work
//! is not set twice and settled work waits until what it is about changes.
//!
//! Every generated mission carries one step of an **operation** the table holds
//! ([`super::operations`]), run on a workflow from the mind's `missions.yaml`.
//! A target is settled when its operation is, not when one of its steps is
//! reported.

use std::cmp::Reverse;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use super::agenda::{Agendum, Priority, KEEP};
use super::operations::{Operation, Operations};
use super::orders::{Filed, Notice, Orders, EVERY_POSTER};
use super::round::RoundState;
use crate::engine::mission::bank::{self, Duty, Facts};
use crate::engine::mission::{Mission, MissionPrompt, Origin, Outcome, StepOutcome, Todo};
use crate::engine::mission_gen::round::{Item as RoundItem, STALE_MS};
use crate::engine::mission_gen::world::{self, Draw, Routine};
use crate::engine::workflow::{Next, Taker, Where, Workflow, STUCK};

/// The outcome a Maker's `report_done` reports on a step that offers a choice.
pub const PASS: &str = "pass";

/// The outcome a Maker's `report_rejected` reports.
pub const REJECT: &str = "reject";

/// What a posted mission's poster is told of one called off before it was
/// reported.
pub const CALLED_OFF: &str = "It was called off before it was done.";

/// What a posted mission's poster is told of one whose carrier, or the one it
/// was set for, is gone from the world.
pub const GONE: &str = "Whoever it was for is gone, and it was never done.";

/// What the world offers a routine, owned: [`Sim::mission_material`]'s answer,
/// which a caller borrows as [`Facts`] while it draws.
///
/// [`Sim::mission_material`]: crate::sim::Sim::mission_material
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MissionMaterial {
    pub makers: Vec<String>,
    pub records: Vec<String>,
    pub duties: Vec<Duty>,
    pub table: Option<String>,
}

impl MissionMaterial {
    /// The same material, borrowed for [`Missions::collect`].
    pub fn facts(&self) -> Facts<'_> {
        Facts {
            makers: &self.makers,
            records: &self.records,
            duties: &self.duties,
            table: self.table.as_deref(),
        }
    }
}

/// What has become of a piece of work the generator found.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Settled {
    /// A mission for it waits at the table.
    Pooled,
    /// Somebody has taken it up.
    Carried,
    /// Reported done.
    Done,
    /// Reported as not done.
    Stuck,
    /// The generator looked and found nothing to do.
    Nothing,
}

/// One target in the ledger.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ledgered {
    pub state: Settled,
    /// What the target held when it was last settled — see
    /// `engine::mission_gen::target::Target::fingerprint`.
    pub fingerprint: u64,
    /// Which generator found it.
    pub generator: String,
    /// The mission's brief, or why there was none.
    pub note: String,
    /// How many times a mission for it has been reported stuck while it held
    /// this fingerprint.
    #[serde(default)]
    pub stuck: u32,
    /// The document its mission writes, when it writes one.
    #[serde(default)]
    pub writes: Option<String>,
}

/// How many stuck reports a target takes before it is left until it changes.
const STUCK_LIMIT: u32 = 2;

impl Ledgered {
    /// Whether a target with `fingerprint` is not to be worked now.
    ///
    /// In hand is always blocked. Done, or found to need nothing, is blocked
    /// until what it is about changes — a life gains an event, a document is
    /// edited. Stuck is worked again, until it has been stuck
    /// [`STUCK_LIMIT`] times on the same text.
    pub fn blocks(&self, fingerprint: u64) -> bool {
        match self.state {
            Settled::Pooled | Settled::Carried => true,
            Settled::Done | Settled::Nothing => self.fingerprint == fingerprint,
            Settled::Stuck => self.fingerprint == fingerprint && self.stuck >= STUCK_LIMIT,
        }
    }
}

/// Every character's mission, and the ones lodged for characters yet to collect.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Missions {
    /// The open mission each body is carrying, by body id. At most one.
    active: BTreeMap<String, Mission>,
    /// Missions lodged for a body that it has not collected yet, oldest first.
    lodged: BTreeMap<String, Vec<Mission>>,
    /// The last mission each body finished, kept so an operator can read the
    /// outcome and answer after the character has reported and moved on.
    done: BTreeMap<String, Mission>,
    /// The seed for the next random routine drawn from the bank, bumped each
    /// draw so a character collecting twice does not get the same routine and a
    /// test can pin the sequence.
    seed: u64,
    /// Missions the command table's generator has written, waiting for whoever
    /// collects next, oldest first. Drawn after anything lodged for the body
    /// and before the routine bank.
    #[serde(default)]
    pool: Vec<Mission>,
    /// What has become of every target the generator found, by target key.
    #[serde(default)]
    targets: BTreeMap<String, Ledgered>,
    /// How many generations have been drawn, so the next rotates among the
    /// generators and among equally good targets.
    #[serde(default)]
    drawn: u64,
    /// Every operation the table has held.
    #[serde(default)]
    operations: Operations,
    /// How many missions each body has closed — reported, rejected or called
    /// off. Each is a chapter of its working life, and a character starts a new
    /// conversation with each (`Minds::think`), its journal carrying the work
    /// across.
    #[serde(default)]
    closed: BTreeMap<String, u64>,
    /// The mission each body closed last, however it closed — what its journal
    /// is told it finished as the chapter turns.
    #[serde(default)]
    last_closed: BTreeMap<String, Mission>,
    /// Bodies stood down because the operation they were carrying a stage of
    /// was called off, not yet told — see [`Self::take_stood_down`].
    #[serde(default)]
    stood_down: Vec<String>,
    /// Bodies the world gives no orders — a personality with `takes_orders:
    /// false` — see [`Self::set_unordered`].
    #[serde(default)]
    unordered: BTreeSet<String>,
    /// The missions posted on the table, from the poster's side — see
    /// [`super::orders`].
    #[serde(default)]
    orders: Orders,
    /// When each standing mission was last done in each room, by `id@room` —
    /// `None` for one reported done and not yet stamped by the table's loop,
    /// which keeps the clock ([`Self::stamp_standing`]).
    #[serde(default)]
    standing: BTreeMap<String, Option<u64>>,
    /// When each standing mission last failed in each room, by `id@room`,
    /// stamped the same way — until it is done there.
    #[serde(default)]
    standing_failed: BTreeMap<String, Option<u64>>,
    /// The world's own routines, from `<mind>/missions/<world>.yaml` —
    /// drawn in place of the built-in [`bank`] when there are any. The mind's,
    /// read afresh rather than saved, like the workflows.
    #[serde(skip)]
    routines: Vec<Routine>,
    /// Whether a character with nothing set for it draws a routine — the
    /// mind's, with the routines.
    #[serde(skip)]
    draw: Draw,
    /// What the world's round has done — see [`super::round`].
    #[serde(default)]
    round: RoundState,
    /// The world's round, from `<mind>/missions/<world>/round.yaml` — the
    /// mind's, read afresh like the routines.
    #[serde(skip)]
    round_items: Vec<RoundItem>,
    /// The missions on each agenda set aside behind the one worked, by body
    /// (`docs/zen_moon_mind.md` §17.1): the worked one is the body's
    /// [`Self::active`].
    #[serde(default)]
    held: BTreeMap<String, Vec<Mission>>,
    /// How many missions have come onto any agenda — what orders missions of
    /// one priority, oldest first.
    #[serde(default)]
    agenda_count: u64,
}

impl Missions {
    /// The bodies stood down since this was last asked, each to be told its
    /// mission was withdrawn.
    ///
    /// **Called off is something a Maker has to be told.** An operation called
    /// off — by an operator, or because its draft left the record — took its
    /// carrier's mission away with no word and no chapter closed: the Maker
    /// went on working from its window on a mission that no longer existed,
    /// and one at a quiet desk was not even woken to find out.
    pub fn take_stood_down(&mut self) -> Vec<String> {
        std::mem::take(&mut self.stood_down)
    }

    /// Mark the body unordered, or not: **the world gives an unordered
    /// character no orders.** The table never calls it, never hands it work and
    /// never offers it `collect_mission`, nothing is lodged for it, and its
    /// idle wake sets it nothing — what it does with its time is its own. Made
    /// unordered, it is stood down from whatever it carries, and told so, and
    /// what was lodged for it is dropped. `docs/mission_generator.md`, "A
    /// character that takes no orders".
    pub fn set_unordered(&mut self, body: &str, unordered: bool) {
        if !unordered {
            self.unordered.remove(body);
            return;
        }
        if self.cancel(body) {
            self.stood_down.push(body.to_string());
        }
        self.lodged.remove(body);
        self.unordered.insert(body.to_string());
    }

    /// Whether the world gives this body no orders — see
    /// [`Self::set_unordered`].
    pub fn is_unordered(&self, body: &str) -> bool {
        self.unordered.contains(body)
    }

    /// Set the world's round (`docs/zen_moon_mind.md` §17) — empty for a
    /// world with none.
    pub fn set_round(&mut self, items: Vec<RoundItem>) {
        self.round_items = items;
    }

    /// The world's round.
    pub fn round_items(&self) -> &[RoundItem] {
        &self.round_items
    }

    /// Whether the world has a round, which is what a body that takes no
    /// orders does when nothing calls for it.
    pub fn has_round(&self) -> bool {
        !self.round_items.is_empty()
    }

    /// What the round has done.
    pub fn round(&self) -> &RoundState {
        &self.round
    }

    /// Give `body` a round item's `mission` at `now_ms`, at `later` on its
    /// agenda, named `title`.
    pub fn take_round(&mut self, body: &str, mission: Mission, title: &str, now_ms: u64) {
        if let Origin::Round { id } = &mission.origin {
            self.round.set(body, id, now_ms);
        }
        let agendum = self.agendum(title, Priority::Later);
        self.active
            .insert(body.to_string(), mission.on_agenda(agendum));
    }

    /// A place on an agenda for a mission coming onto it now.
    fn agendum(&mut self, title: &str, priority: Priority) -> Agendum {
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
            let was = std::mem::take(&mut m.todo);
            m.todo = steps
                .into_iter()
                .map(|text| {
                    was.iter()
                        .find(|t| t.text.trim().eq_ignore_ascii_case(text.trim()))
                        .cloned()
                        .unwrap_or_else(|| Todo::new(text))
                })
                .collect();
        }
        if let (Some(p), Some(a)) = (priority, m.agendum.as_mut()) {
            a.priority = p;
        }
        self.agenda_rank(body);
        Ok(())
    }

    /// Close the mission on `body`'s agenda called `title` as done, on
    /// `found` — the agenda's own close, for a mission no step shows done.
    pub fn agenda_done(&mut self, body: &str, title: &str, found: &str) -> Result<Mission, String> {
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
    fn agenda_mut(&mut self, body: &str, title: &str) -> Result<&mut Mission, String> {
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
    fn agenda_work(&mut self, body: &str, title: &str) -> Result<(), String> {
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
    fn agenda_rank(&mut self, body: &str) {
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

    /// Stamp every round item done or failed since the last stamp at
    /// `now_ms`.
    pub fn stamp_round(&mut self, now_ms: u64) {
        self.round.stamp(now_ms);
    }

    /// Close every mission worked on an agenda whose steps are all signed off,
    /// as done, and call off as failed every round mission carried
    /// [`STALE_MS`] without — an agenda is closed by what the engine saw, with
    /// no report (`docs/zen_moon_mind.md` §17.5). Returns each body closed, the
    /// mission's title, and how.
    pub fn settle_agendas(&mut self, now_ms: u64) -> Vec<(String, String, Outcome)> {
        let closing: Vec<(String, String, Outcome)> = self
            .active
            .iter()
            .filter_map(|(body, m)| {
                let title = m.agendum.as_ref()?.title.clone();
                let stale = matches!(m.origin, Origin::Round { .. })
                    && self
                        .round
                        .set_at(body)
                        .is_some_and(|at| now_ms >= at.saturating_add(STALE_MS));
                match (m.all_todos_done(), stale) {
                    (true, _) => Some((body.clone(), title, Outcome::Pass)),
                    (false, true) => Some((body.clone(), title, Outcome::Fail)),
                    (false, false) => None,
                }
            })
            .collect();
        for (body, _, outcome) in &closing {
            let notes = match outcome {
                Outcome::Pass => "every step was seen done",
                Outcome::Fail => "carried an hour without every step done",
            };
            self.report(body, *outcome, notes, None);
        }
        closing
    }

    /// Sign off the body's open steps that post work — its `order_post`
    /// landed. Returns whether anything was signed off.
    pub fn posted_work(&mut self, body: &str) -> bool {
        self.active.get_mut(body).is_some_and(Mission::posted_work)
    }

    /// Sign off the body's open steps that read what came back — its
    /// `order_results` landed. Returns whether anything was signed off.
    pub fn read_results(&mut self, body: &str) -> bool {
        self.active.get_mut(body).is_some_and(Mission::read_results)
    }

    /// Sign off the body's open steps that read its messages — a `read` of
    /// its phone landed. Returns whether anything was signed off.
    pub fn read_messages(&mut self, body: &str) -> bool {
        self.active
            .get_mut(body)
            .is_some_and(Mission::read_messages)
    }

    /// The body's open mission, if it is on one.
    pub fn active(&self, body: &str) -> Option<&Mission> {
        self.active.get(body)
    }

    /// The last mission the body finished, if any.
    pub fn done(&self, body: &str) -> Option<&Mission> {
        self.done.get(body)
    }

    /// The chapter of its working life the body is in: how many missions it has
    /// closed, not counting the table's routines.
    pub fn chapter(&self, body: &str) -> u64 {
        self.closed.get(body).copied().unwrap_or(0)
    }

    /// The work the body closed last — reported, rejected or called off — which
    /// is what its chapter closes with; the table's routines are not counted.
    pub fn last_closed(&self, body: &str) -> Option<&Mission> {
        self.last_closed.get(body)
    }

    /// `mission` of the body's is closed: the next chapter begins — unless it
    /// was one of the table's routines.
    ///
    /// **A routine is an interlude, not a stretch of work.** A chapter closes so
    /// the work it held is carried across in the journal and the next starts in
    /// a conversation of its own, which reseals the journal from nothing. The
    /// routines a Maker draws while the table has no work for it — read a
    /// station, walk the halls — take a couple of minutes each: on WW they
    /// turned 52 chapters in an hour, each a whole journal resealed, and the
    /// reseals took the engine the table needed to write the work that would
    /// have ended the routines.
    ///
    /// A routine is not what the chapter closes with either: one handed in
    /// between a mission and the next turn would otherwise write the chapter's
    /// journal entry in the mission's place.
    fn close(&mut self, body: &str, mission: &Mission) {
        // A standing job posted at the table is a routine still, whoever set
        // it, and so is an item of the round.
        if matches!(
            mission.origin,
            Origin::Random { .. }
                | Origin::Ordered { job: Some(_), .. }
                | Origin::Round { .. }
                | Origin::Own
        ) {
            return;
        }
        *self.closed.entry(body.to_string()).or_insert(0) += 1;
        self.last_closed.insert(body.to_string(), mission.clone());
    }

    /// The mission to answer an operator's question about this body with — the
    /// open one if it is on one, otherwise the last it finished.
    pub fn latest(&self, body: &str) -> Option<&Mission> {
        self.active.get(body).or_else(|| self.done.get(body))
    }

    /// Whether the body is carrying an open mission right now.
    pub fn is_on_mission(&self, body: &str) -> bool {
        self.active.contains_key(body)
    }

    /// Whether a mission has been lodged for the body and not yet collected.
    pub fn has_lodged(&self, body: &str) -> bool {
        self.lodged.get(body).is_some_and(|queue| !queue.is_empty())
    }

    /// Lodge a mission for a body to collect at the desk. Oldest is drawn first.
    pub fn lodge(&mut self, body: &str, mission: Mission) {
        self.lodged
            .entry(body.to_string())
            .or_default()
            .push(mission);
    }

    /// Give the body a mission to carry now, replacing any open one it held. A
    /// generated mission replaced is called off, and its target released to be
    /// found again — left marked as carried, nothing would ever set it again.
    pub fn assign(&mut self, body: &str, mission: Mission) {
        if let Some(old) = self.active.insert(body.to_string(), mission) {
            self.called_off(body, &old, CALLED_OFF);
            if let Some(k) = target_of(&old) {
                if target_of(&self.active[body]) != Some(k) {
                    self.targets.remove(k);
                }
            }
            if let Some((id, _)) = old.operation() {
                if self.active[body].operation().map(|(i, _)| i) != Some(id) {
                    self.operations.cancel(id, "its mission was replaced");
                }
            }
        }
    }

    /// Hold `key` while a mission is being written for it, so a second
    /// generation running at the same time does not choose it too. `false`
    /// when it is already blocked. Released by [`Self::release`] if no mission
    /// comes of it; settled by [`Self::offer`] or [`Self::decline`] if one does.
    pub fn reserve(&mut self, key: &str, generator: &str, fingerprint: u64) -> bool {
        if self.blocks(key, fingerprint) {
            return false;
        }
        let stuck = self
            .targets
            .get(key)
            .filter(|e| e.fingerprint == fingerprint)
            .map_or(0, |e| e.stuck);
        self.targets.insert(
            key.to_string(),
            Ledgered {
                state: Settled::Pooled,
                fingerprint,
                generator: generator.to_string(),
                note: "a mission is being written for it".to_string(),
                stuck,
                writes: None,
            },
        );
        true
    }

    /// Let go of a target [`Self::reserve`] held, when no mission came of it.
    ///
    /// **A target that was stuck goes back to stuck, count and all.** Forgotten
    /// instead, a target whose generation kept failing after the reservation
    /// would never reach [`STUCK_LIMIT`] however often it was reported stuck.
    pub fn release(&mut self, key: &str) {
        if self.pool.iter().any(|m| target_of(m) == Some(key)) {
            return;
        }
        let Some(entry) = self.targets.get_mut(key) else {
            return;
        };
        if entry.state != Settled::Pooled || entry.writes.is_some() {
            return;
        }
        if entry.stuck > 0 {
            entry.state = Settled::Stuck;
            entry.note = "released: no mission came of it".to_string();
        } else {
            self.targets.remove(key);
        }
    }

    /// Collect a mission at the desk: the next one lodged for this body, then
    /// the oldest waiting at the table, or — when neither is — a fresh routine:
    /// one of the world's own when it has any ([`Self::set_routines`]),
    /// otherwise a non-destructive one drawn from the [`bank`] against who and
    /// what is around. It becomes the body's open mission, and the collected one
    /// is returned so the desk can hand back its brief.
    ///
    /// `None` in a world that draws nothing ([`Draw::None`]) with nothing set
    /// for this body: it waits at the table until work is posted.
    pub fn collect(&mut self, body: &str, facts: &Facts) -> Option<&Mission> {
        let mission = match self.take_lodged(body).or_else(|| self.take_pooled(body)) {
            Some(mission) => mission,
            None if self.draw == Draw::None => return None,
            None => {
                let drawn = world::draw(&self.routines, facts.table, self.seed)
                    .unwrap_or_else(|| bank::random(facts, self.seed));
                self.seed = self.seed.wrapping_add(1);
                drawn
            }
        };
        self.active.insert(body.to_string(), mission);
        self.active.get(body)
    }

    /// Whether this world hands a character with nothing set for it a routine
    /// — see [`Draw`].
    pub fn draws(&self) -> bool {
        self.draw == Draw::Routines
    }

    /// Whether anything waits at the table for `body`: lodged for it, or open
    /// to whoever takes it.
    pub fn has_work_for(&self, body: &str) -> bool {
        self.has_lodged(body) || self.has_pooled_for(body)
    }

    /// Post `mission` on the table: for the body `to`, or, with none, for
    /// whoever takes it next. Whoever it is for is told
    /// ([`Self::take_notices`]).
    pub fn post(&mut self, mission: Mission, to: Option<&str>) {
        self.orders.posted(to, &mission.prompt);
        match to {
            Some(body) => self.lodge(body, mission),
            None => self.pool.push(mission),
        }
    }

    /// The notices owed since this was last asked — see [`Notice`].
    pub fn take_notices(&mut self) -> Vec<Notice> {
        self.orders.take_notices()
    }

    /// Put a standing mission up on the table, for whoever takes work when
    /// nothing else waits. Whoever waits at the table is told.
    pub fn put_up_standing(&mut self, mission: Mission) {
        self.orders.posted(None, &mission.prompt);
        self.pool.push(mission);
    }

    /// Whether standing mission `id` is up or carried in `room` — the one per
    /// room there may be at a time.
    pub fn standing_out(&self, id: &str, room: &str) -> bool {
        let is = |m: &Mission| matches!(&m.origin, Origin::Standing { id: i, room: r } if i == id && r == room);
        self.pool.iter().any(is)
            || self.active.values().any(is)
            || self.lodged.values().flatten().any(is)
    }

    /// When standing mission `id` was last done in `room`: `None` never,
    /// `Some(None)` done and not yet stamped.
    pub fn standing_done(&self, id: &str, room: &str) -> Option<Option<u64>> {
        self.standing.get(&format!("{id}@{room}")).copied()
    }

    /// When standing mission `id` last failed in `room`, as
    /// [`Self::standing_done`] — `None` when it has not failed since it was
    /// last done.
    pub fn standing_failed(&self, id: &str, room: &str) -> Option<Option<u64>> {
        self.standing_failed.get(&format!("{id}@{room}")).copied()
    }

    /// Stamp every standing mission done or failed since the last stamp as
    /// done or failed at `now_ms`.
    pub fn stamp_standing(&mut self, now_ms: u64) {
        for at in self
            .standing
            .values_mut()
            .chain(self.standing_failed.values_mut())
        {
            at.get_or_insert(now_ms);
        }
    }

    /// Whether a report on what `poster` posted waits unread.
    pub fn unread_for(&self, poster: &str) -> bool {
        self.orders.unread_for(poster)
    }

    /// The reports on what `poster` posted that it has not read, newest first
    /// — read now. See [`Orders::read_by`].
    pub fn read_by(&mut self, poster: &str) -> Vec<Filed> {
        self.orders.read_by(poster)
    }

    /// The world's standing job named `id`, if it has one.
    pub fn routine(&self, id: &str) -> Option<&Routine> {
        self.routines.iter().find(|r| r.id == id)
    }

    /// What `poster` posted that is not yet reported: each brief, and the body
    /// carrying it or that it is set for — `None` while it waits for whoever
    /// takes it.
    pub fn out_for(&self, poster: &str) -> Vec<(String, Option<String>)> {
        let posted_by =
            |m: &Mission| matches!(&m.origin, Origin::Ordered { poster: p, .. } if p == poster);
        let carried = self
            .active
            .iter()
            .filter(|(_, m)| posted_by(m))
            .map(|(body, m)| (m.prompt.clone(), Some(body.clone())));
        let lodged = self
            .lodged
            .iter()
            .flat_map(|(body, q)| q.iter().map(move |m| (body, m)))
            .filter(|(_, m)| posted_by(m))
            .map(|(body, m)| (m.prompt.clone(), Some(body.clone())));
        let waiting = self
            .pool
            .iter()
            .filter(|m| posted_by(m))
            .map(|m| (m.prompt.clone(), None));
        waiting.chain(lodged).chain(carried).collect()
    }

    /// Whether nothing waits on the table: nothing open to take, and nothing
    /// lodged for anybody. The world's standing upkeep is not work waiting —
    /// it is what is done when there is none.
    pub fn table_empty(&self) -> bool {
        self.pool.iter().all(is_standing) && self.lodged.values().all(Vec::is_empty)
    }

    /// The world's own routines — in a world that draws nothing, the standing
    /// jobs a poster may set when the table is empty.
    pub fn routines(&self) -> &[Routine] {
        &self.routines
    }

    /// Whether a mission waits at the table that `body` may take — see
    /// [`Self::take_pooled`].
    pub fn has_pooled_for(&self, body: &str) -> bool {
        !self.is_unordered(body) && self.pool.iter().any(|m| self.may_take(body, m))
    }

    /// Whether `body` may take mission `m` from the table: anybody may, but
    /// a step of an operation only somebody its workflow lets take it.
    fn may_take(&self, body: &str, m: &Mission) -> bool {
        match m.operation() {
            Some((id, _)) => self.operations.may_take(id, body),
            None => true,
        }
    }

    /// The oldest generated mission waiting at the table that `body` may take,
    /// now carried by it. A review is never taken by the Maker who drafted it.
    ///
    /// **A standing mission is given out last** (`docs/zen_moon_mind.md`
    /// §12): taken only when nothing else waiting may be.
    fn take_pooled(&mut self, body: &str) -> Option<Mission> {
        let at = self
            .pool
            .iter()
            .position(|m| self.may_take(body, m) && !is_standing(m))
            .or_else(|| {
                self.pool
                    .iter()
                    .position(|m| self.may_take(body, m) && is_standing(m))
            })?;
        let mission = self.pool.remove(at);
        if let Some(entry) = target_of(&mission).and_then(|k| self.targets.get_mut(k)) {
            entry.state = Settled::Carried;
        }
        if let Some((id, _)) = mission.operation() {
            self.operations.taken(id, body, None);
        }
        Some(mission)
    }

    /// Set the workflows operations run on — see [`Operations::set_workflows`].
    pub fn set_workflows(&mut self, workflows: Vec<Workflow>) {
        self.operations.set_workflows(workflows);
    }

    /// Set the world's own routines, and whether they are drawn — empty, and
    /// [`Draw::Routines`], for a world that runs on `missions.yaml` and draws
    /// from the built-in [`bank`].
    pub fn set_routines(&mut self, routines: Vec<Routine>, draw: Draw) {
        self.routines = routines;
        self.draw = draw;
    }

    /// Take `from`'s workflows and routines in place of these: they are the
    /// mind's, not the world's, and a world taken in from elsewhere keeps the
    /// ones this machine loaded.
    pub fn adopt_workflows(&mut self, from: &mut Missions) {
        self.operations.adopt_workflows(&mut from.operations);
        self.routines = std::mem::take(&mut from.routines);
        self.draw = from.draw;
    }

    /// Call off every operation not yet over, for `why` — what a world that
    /// has come to run on its own missions does with the record's work opened
    /// before it did. Returns the ids called off.
    pub fn call_off_operations(&mut self, why: &str) -> Vec<u64> {
        let open: Vec<u64> = self
            .operations
            .all()
            .filter(|o| !o.settled())
            .map(|o| o.id)
            .collect();
        open.into_iter()
            .filter(|id| self.cancel_operation(*id, why))
            .collect()
    }

    /// Record what operation `id`'s document said when its step was taken up.
    pub fn found(&mut self, id: u64, text: String) {
        if let Some(o) = self.operations.get_mut(id) {
            o.found = Some(text);
        }
    }

    /// Open an operation on `workflow` for a generator's accepted proposal,
    /// for `fingerprint` of its target, waiting on the workflow's first step —
    /// set on the table by the engine's loop like every other step.
    ///
    /// `objective` says in a line what it is for, `brief` what its document is
    /// to tell, `fields` the proposal's own, which every step's prompt is filled
    /// from, and `reads` what its work reads. Returns its id.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &mut self,
        workflow: &str,
        generator: &str,
        target: &str,
        fingerprint: u64,
        objective: &str,
        document: &str,
        brief: &str,
        fields: BTreeMap<String, String>,
        reads: Vec<String>,
    ) -> Result<u64, String> {
        let id = self
            .operations
            .open(workflow, None, generator, target, objective, document)?;
        self.operations.briefed(id, brief, fields, reads);
        let stuck = self
            .targets
            .get(target)
            .filter(|e| e.fingerprint == fingerprint)
            .map_or(0, |e| e.stuck);
        self.targets.insert(
            target.to_string(),
            Ledgered {
                state: Settled::Pooled,
                fingerprint,
                generator: generator.to_string(),
                note: objective.to_string(),
                stuck,
                writes: Some(document.to_string()),
            },
        );
        Ok(id)
    }

    /// Put `mission` on the table as the step operation `id` waits on.
    pub fn offer_step(&mut self, id: u64, mut mission: Mission) {
        let step = self
            .operations
            .get(id)
            .and_then(|o| o.step())
            .unwrap_or_default()
            .to_string();
        if let Origin::Generated {
            operation,
            step: carried,
            ..
        } = &mut mission.origin
        {
            *operation = id;
            *carried = step;
        }
        self.operations.offered(id);
        self.pool.push(mission);
    }

    /// The table took the step operation `id` waits on — one of its call's
    /// verdicts, `outcome`, with what it found — and the operation moves on;
    /// its target is settled once it is over.
    pub fn table_took(&mut self, id: u64, outcome: &str, found: &str) -> Result<Where, String> {
        let at = self
            .operations
            .advance(id, &Taker::Table, Some(outcome), found)?;
        self.settle_if_over(id);
        Ok(at)
    }

    /// Settle operation `id`'s target, and stand its carriers down, once the
    /// operation is over: done is the target done; failed, stuck; called off,
    /// free to be found again.
    fn settle_if_over(&mut self, id: u64) {
        let Some(o) = self.operations.get(id) else {
            return;
        };
        let (target, at) = (o.target.clone(), o.run.at.clone());
        match at {
            Where::NextStep(_) => {}
            Where::Done => self.settle_target(Some(target), Outcome::Pass, "it stands"),
            Where::Failed(why) => self.settle_target(Some(target), Outcome::Fail, &why),
            Where::Cancelled(_) => {
                self.targets.remove(&target);
            }
        }
    }

    /// Record that a generator looked at a target and found nothing to do.
    pub fn decline(&mut self, key: &str, generator: &str, fingerprint: u64, why: &str) {
        self.targets.insert(
            key.to_string(),
            Ledgered {
                state: Settled::Nothing,
                fingerprint,
                generator: generator.to_string(),
                note: why.to_string(),
                stuck: 0,
                writes: None,
            },
        );
    }

    /// Whether the target `key`, holding `fingerprint`, is not to be worked now.
    pub fn blocks(&self, key: &str, fingerprint: u64) -> bool {
        self.targets.get(key).is_some_and(|e| e.blocks(fingerprint))
    }

    /// Every operation the table has held.
    pub fn operations(&self) -> &Operations {
        &self.operations
    }

    /// Put settled operation `id` back on `step` of its workflow, in a new
    /// round — succeeded lore sent to be checked again, say.
    pub fn reopen(&mut self, id: u64, step: &str) -> Result<(), String> {
        self.operations.reopen(id, step)?;
        if let Some(o) = self.operations.get(id) {
            if let Some(entry) = self.targets.get_mut(&o.target) {
                entry.state = Settled::Pooled;
            }
        }
        Ok(())
    }

    /// Send operation `id` to `step` of its workflow: moved there when it is
    /// running ([`Self::move_to`]), reopened there when it is over
    /// ([`Self::reopen`]).
    pub fn send_to_step(&mut self, id: u64, step: &str) -> Result<(), String> {
        let settled = self
            .operations
            .get(id)
            .map(Operation::settled)
            .ok_or("no such operation")?;
        match settled {
            true => self.reopen(id, step),
            false => self.move_to(id, step),
        }
    }

    /// Move unsettled operation `id` to `step`, whatever it waited on: a step
    /// waiting at the table leaves it, and the operation goes on from `step` in
    /// a new round. `false` when it is settled or carried.
    pub fn move_to(&mut self, id: u64, step: &str) -> Result<(), String> {
        if self.carrying(id).is_some() {
            return Err("its step is being carried".into());
        }
        let target = self
            .operations
            .get(id)
            .map(|o| o.target.clone())
            .ok_or("no such operation")?;
        if !self.operations.cancel(id, "moved by hand") {
            return Err("it is over".into());
        }
        self.pool
            .retain(|m| m.operation().map(|(i, _)| i) != Some(id));
        self.operations.reopen(id, step)?;
        if let Some(entry) = self.targets.get_mut(&target) {
            entry.state = Settled::Pooled;
        }
        Ok(())
    }

    /// Record that failed operation `id`'s document has been settled.
    pub fn mark_retired(&mut self, id: u64) {
        self.operations.mark_retired(id);
    }

    /// Open an operation on `workflow` that starts at `step` with `document`,
    /// which already stands — an operator putting a document through the
    /// table's reading and a Maker's review. Returns its id.
    pub fn review_document(
        &mut self,
        workflow: &str,
        step: &str,
        document: &str,
    ) -> Result<u64, String> {
        self.operations.open(
            workflow,
            Some(step),
            "operator",
            &format!("doc:{document}"),
            &format!("Review {document}"),
            document,
        )
    }

    /// Call operation `id` off: its waiting mission leaves the table and a
    /// Maker carrying one of its stages is stood down. `false` when it was
    /// already over or there is no such operation.
    pub fn cancel_operation(&mut self, id: u64, why: &str) -> bool {
        let Some(target) = self.operations.get(id).map(|o| o.target.clone()) else {
            return false;
        };
        if !self.operations.cancel(id, why) {
            return false;
        }
        self.pool
            .retain(|m| m.operation().map(|(i, _)| i) != Some(id));
        let carriers: Vec<String> = self
            .active
            .iter()
            .filter(|(_, m)| m.operation().map(|(i, _)| i) == Some(id))
            .map(|(body, _)| body.clone())
            .collect();
        for body in carriers {
            if let Some(m) = self.active.remove(&body) {
                self.close(&body, &m);
                self.stood_down.push(body);
            }
        }
        self.targets.remove(&target);
        true
    }

    /// Edit operation `id`: its name, what it is for, and — while its next
    /// mission still waits at the table — that mission's brief.
    pub fn edit_operation(
        &mut self,
        id: u64,
        name: Option<&str>,
        objective: Option<&str>,
        brief: Option<&str>,
    ) -> Result<&Operation, String> {
        if self.operations.get(id).is_none() {
            return Err("no such operation".into());
        }
        if let Some(n) = name {
            self.operations.rename(id, n)?;
        }
        if let Some(o) = objective {
            self.operations.set_objective(id, o)?;
        }
        if let Some(b) = brief.map(str::trim) {
            if b.is_empty() {
                return Err("a brief cannot be empty".into());
            }
            let waiting = self
                .pool
                .iter_mut()
                .find(|m| m.operation().map(|(i, _)| i) == Some(id))
                .ok_or("its mission is not waiting at the table, so its brief is set")?;
            waiting.prompt = b.to_string();
        }
        Ok(self.operations.get(id).expect("checked above"))
    }

    /// The brief of operation `id`'s mission waiting at the table, if one is.
    pub fn waiting_brief(&self, id: u64) -> Option<&str> {
        self.pool
            .iter()
            .find(|m| m.operation().map(|(i, _)| i) == Some(id))
            .map(|m| m.prompt.as_str())
    }

    /// Who is carrying a stage of operation `id` right now.
    pub fn carrying(&self, id: u64) -> Option<&str> {
        self.active
            .iter()
            .find(|(_, m)| m.operation().map(|(i, _)| i) == Some(id))
            .map(|(b, _)| b.as_str())
    }

    /// The generated missions waiting at the table.
    pub fn pooled(&self) -> &[Mission] {
        &self.pool
    }

    /// Whether the table holds enough: at least `keep` missions, and something
    /// each of `makers` may take — up to `keep` more than there are Makers,
    /// past which nothing more is written however the pool falls.
    ///
    /// **Stock is per Maker, not per table.** With four missions waiting the
    /// table counted as stocked while every one of them was a stage its one
    /// free Maker had already carried: it drew routine after routine — the
    /// pressure door, read six times in twenty minutes — and no new work was
    /// ever written for it.
    ///
    /// The world's standing upkeep is not stock: a table holding only that is
    /// a table with no work on it.
    pub fn stocked_for(&self, makers: &[String], keep: usize) -> bool {
        let work = || self.pool.iter().filter(|m| !is_standing(m));
        let n = work().count();
        let has_work_for =
            |body: &str| !self.is_unordered(body) && work().any(|m| self.may_take(body, m));
        n >= keep + makers.len() || (n >= keep && makers.iter().all(|m| has_work_for(m)))
    }

    /// Strike, from every mission carried, waiting at the table or lodged, each
    /// read of a document `holds` says is not there for it — asked with the
    /// operation the mission is a step of, whose store it reads through; see
    /// [`Mission::strike_gone_reads`]. Returns each body or the pool (`None`)
    /// with the paths struck from it.
    pub fn strike_gone_reads(
        &mut self,
        holds: &dyn Fn(Option<u64>, &str) -> bool,
    ) -> Vec<(Option<String>, Vec<String>)> {
        let carried = self
            .active
            .iter_mut()
            .chain(
                self.lodged
                    .iter_mut()
                    .flat_map(|(b, q)| q.iter_mut().map(move |m| (b, m))),
            )
            .map(|(b, m)| (Some(b.clone()), m.strike_gone_reads(holds)));
        let pooled = self
            .pool
            .iter_mut()
            .map(|m| (None, m.strike_gone_reads(holds)));
        carried
            .chain(pooled)
            .filter(|(_, gone)| !gone.is_empty())
            .collect()
    }

    /// Call off every operation whose next mission waits at the table for a
    /// Maker none of `makers` may be: each of them has already carried a stage
    /// of it. Returns the operations called off.
    ///
    /// **Work nobody may take is not work waiting.** Nobody checks their own
    /// work or checks it twice ([`Operations::may_take`]), so with a small
    /// cast a review reported stuck twice can run out of Makers who have not
    /// touched it. It then sat at the table for ever: never taken, its target
    /// blocked from being found again, and counted as stock — enough of them
    /// and the generator, keeping the table stocked, never opened another.
    /// Called off, its target is free to be found afresh.
    pub fn call_off_untakeable(&mut self, makers: &[String]) -> Vec<u64> {
        // A world nobody has stood in yet — at boot, before anybody is bound —
        // is not one with nobody left.
        if makers.is_empty() {
            return Vec::new();
        }
        let stranded: Vec<u64> = self
            .pool
            .iter()
            .filter_map(|m| m.operation().map(|(id, _)| id))
            .filter(|id| !makers.iter().any(|b| self.operations.may_take(*id, b)))
            .collect();
        stranded
            .into_iter()
            .filter(|id| {
                self.cancel_operation(*id, "no Maker is left who has not already worked on it")
            })
            .collect()
    }

    /// Every target the generator has found, and what became of it.
    pub fn targets(&self) -> &BTreeMap<String, Ledgered> {
        &self.targets
    }

    /// Take the next generation's turn number.
    pub fn next_draw(&mut self) -> u64 {
        let turn = self.drawn;
        self.drawn += 1;
        turn
    }

    /// Forget every target that is not in somebody's hands — done, stuck and
    /// found-to-need-nothing alike — so all of it can be found again. Returns
    /// how many were forgotten.
    pub fn forget_settled(&mut self) -> usize {
        let before = self.targets.len();
        self.targets
            .retain(|_, e| matches!(e.state, Settled::Pooled | Settled::Carried));
        before - self.targets.len()
    }

    /// Throw away every generated mission still waiting, releasing its target
    /// to be found again. Returns how many there were.
    pub fn discard_pool(&mut self) -> usize {
        let gone = std::mem::take(&mut self.pool);
        for m in &gone {
            if let Some(k) = target_of(m) {
                self.targets.remove(k);
            }
            if let Some((id, _)) = m.operation() {
                self.operations
                    .cancel(id, "the table's waiting missions were discarded");
            }
        }
        gone.len()
    }

    /// Draw the next lodged mission for a body, oldest first, if one waits. The
    /// caller assigns it; taking and carrying are two steps so a draw that finds
    /// nothing can fall back to a random one without a mission going missing.
    pub fn take_lodged(&mut self, body: &str) -> Option<Mission> {
        let queue = self.lodged.get_mut(body)?;
        if queue.is_empty() {
            return None;
        }
        let mission = queue.remove(0);
        if queue.is_empty() {
            self.lodged.remove(body);
        }
        Some(mission)
    }

    /// Sign a step off on the body's open mission. `false` when it has no open
    /// mission or nothing matched — see [`Mission::check_off`].
    pub fn check_off(&mut self, body: &str, step: &str, outcome: StepOutcome) -> bool {
        self.active
            .get_mut(body)
            .is_some_and(|m| m.check_off(step, outcome))
    }

    /// Sign off the journeys on the body's open mission that end in `room` on
    /// `level`, where it now stands — see [`Mission::arrived_in`].
    pub fn arrived_in(&mut self, body: &str, room: &str, level: &str) -> bool {
        self.active
            .get_mut(body)
            .is_some_and(|m| m.arrived_in(room, level))
    }

    /// Sign off the readings of `path` on the body's open mission — see
    /// [`Mission::read_doc`].
    pub fn read_doc(&mut self, body: &str, path: &str) -> bool {
        self.active.get_mut(body).is_some_and(|m| m.read_doc(path))
    }

    /// Sign off the writes of `paths` on the body's open mission — see
    /// [`Mission::committed`].
    pub fn committed(&mut self, body: &str, paths: &[String]) -> bool {
        self.active
            .get_mut(body)
            .is_some_and(|m| m.committed(paths))
    }

    /// Put the writing of the body's mission document back in front of it —
    /// see [`Mission::reopen_write`].
    /// What the table said when it last refused the body's mission document,
    /// or `None` once a report of it is taken — see `Mission::gate_refusal`.
    pub fn set_gate_refusal(&mut self, body: &str, refusal: Option<String>) {
        if let Some(m) = self.active.get_mut(body) {
            m.gate_refusal = refusal;
        }
    }

    pub fn reopen_write(&mut self, body: &str) {
        if let Some(m) = self.active.get_mut(body) {
            if let Some(path) = m.work.as_ref().map(|w| w.writes.clone()) {
                m.reopen_write(&path);
            }
        }
    }

    /// Set the year the body's open mission is worked in — see
    /// [`Mission::travelled`]. `None` when it carries no mission; otherwise
    /// whether a step was signed off.
    pub fn travelled(&mut self, body: &str, year: u32) -> Option<bool> {
        self.active.get_mut(body).map(|m| m.travelled(year))
    }

    /// The year the body works in, set at a time machine for the mission it
    /// carries. `None` is the present.
    pub fn year_of(&self, body: &str) -> Option<u32> {
        self.active.get(body).and_then(|m| m.year)
    }

    /// Sign off the readings on the body's open mission of the machines it just
    /// scanned — see [`Mission::read_off`].
    pub fn read_off(&mut self, body: &str, seen: &[String]) -> bool {
        self.active.get_mut(body).is_some_and(|m| m.read_off(seen))
    }

    /// Sign off the steps on the body's open mission that are about being with
    /// `who`, who is in the room with it — see [`Mission::met`].
    pub fn met(&mut self, body: &str, who: &str) -> bool {
        self.active.get_mut(body).is_some_and(|m| m.met(who))
    }

    /// Sign off the steps on the body's open mission that are about being with
    /// or speaking to `who`, to whom it just said something — see
    /// [`Mission::spoke_with`].
    pub fn spoke_with(&mut self, body: &str, who: &str) -> bool {
        self.active.get_mut(body).is_some_and(|m| m.spoke_with(who))
    }

    /// Note on the body's open mission something the engine saw — see
    /// [`Mission::observe`].
    pub fn observe(&mut self, body: &str, line: impl Into<String>) {
        if let Some(m) = self.active.get_mut(body) {
            m.observe(line);
        }
    }

    /// Count a `report_stuck` turned away on the body's open mission — see
    /// [`Mission::refuse_stuck`]. `0` with no open mission.
    pub fn refuse_stuck(&mut self, body: &str) -> u32 {
        self.active.get_mut(body).map_or(0, Mission::refuse_stuck)
    }

    /// Count a `report_stuck` turned away while its way is open, when it is a
    /// real one — see [`Mission::refuse_stuck_on_the_way`]. `0` with no open
    /// mission.
    pub fn refuse_stuck_on_the_way(&mut self, body: &str) -> u32 {
        self.active
            .get_mut(body)
            .map_or(0, Mission::refuse_stuck_on_the_way)
    }

    /// Count a `report_stuck` turned away over its document's faults, when the
    /// document was rewritten since the last — see
    /// [`Mission::refuse_stuck_over_faults`]. `0` with no open mission.
    pub fn refuse_stuck_over_faults(&mut self, body: &str, draft: u64) -> u32 {
        self.active
            .get_mut(body)
            .map_or(0, |m| m.refuse_stuck_over_faults(draft))
    }

    /// The world refused `body` an act; noted on its open mission, if any.
    pub fn refused_an_act(&mut self, body: &str) {
        if let Some(m) = self.active.get_mut(body) {
            m.refused_an_act();
        }
    }

    /// How many missions are lodged for the body and not yet collected.
    pub fn lodged_count(&self, body: &str) -> usize {
        self.lodged.get(body).map_or(0, Vec::len)
    }

    /// Add a step to the body's open mission. `false` when it has no open
    /// mission, or the step was blank or a duplicate — see [`Mission::add_todo`].
    pub fn add_todo(&mut self, body: &str, step: &str) -> bool {
        self.active.get_mut(body).is_some_and(|m| m.add_todo(step))
    }

    /// Cancel the body's open mission without a report — it is called off, not
    /// finished, so it is not kept in `done`. Returns whether there was one.
    pub fn cancel(&mut self, body: &str) -> bool {
        self.cancel_for(body, CALLED_OFF)
    }

    /// [`Self::cancel`], its poster told `why` when it was posted.
    fn cancel_for(&mut self, body: &str, why: &str) -> bool {
        match self.active.remove(body) {
            Some(m) => {
                // Called off is not settled: its target can be found again.
                if let Some(k) = target_of(&m) {
                    self.targets.remove(k);
                }
                if let Some((id, _)) = m.operation() {
                    self.operations.cancel(id, "its mission was called off");
                }
                if matches!(m.origin, Origin::Round { .. }) {
                    self.round.dropped(body);
                }
                self.called_off(body, &m, why);
                self.close(body, &m);
                self.agenda_rank(body);
                true
            }
            None => false,
        }
    }

    /// A body is gone from the world: what it carried is called off, what was
    /// lodged for it is dropped, and the poster of any of it is told — a
    /// mission set for somebody who is no longer there is otherwise out for
    /// good, and the table never reads empty again.
    pub fn forget(&mut self, body: &str) {
        self.cancel_for(body, GONE);
        for m in self.lodged.remove(body).unwrap_or_default() {
            self.called_off(body, &m, GONE);
        }
        self.unordered.remove(body);
    }

    /// File `m`, a posted mission ended other than by its carrier's report, to
    /// its poster as not done, for `why` — so its poster is told, and it is no
    /// longer counted as out.
    fn called_off(&mut self, body: &str, m: &Mission, why: &str) {
        if let Origin::Ordered { poster, .. } = &m.origin {
            self.orders.filed(Filed {
                poster: poster.clone(),
                carrier: body.to_string(),
                brief: m.prompt.clone(),
                outcome: Outcome::Fail,
                notes: why.to_string(),
            });
        }
    }

    /// Operation `id`'s store landed on the record as `commit`.
    pub fn published(&mut self, id: u64, commit: &str) {
        self.operations.published(id, commit);
    }

    /// Offer operation `id`'s publish waiting on git at once. `false` when
    /// none is waiting.
    pub fn retry_now(&mut self, id: u64) -> bool {
        self.operations.retry_now(id)
    }

    /// The body's publish could not reach git, for `why`, at `now_ms`: its
    /// mission is put down without a report — neither an outcome nor a stuck
    /// one — and its operation's step is offered again once the wait is over
    /// (`docs/replicated_substrate.md` §10.3). Returns the operation.
    pub fn publish_waits(&mut self, body: &str, why: &str, now_ms: u64) -> Option<u64> {
        let (id, _) = self.active.get(body)?.operation()?;
        let m = self.active.remove(body)?;
        self.close(body, &m);
        self.operations.released(id);
        self.operations.publish_waits(id, why, now_ms);
        Some(id)
    }

    /// File the body's completion report, closing the open mission and keeping
    /// it in `done` so its outcome and answer can still be read. Returns the
    /// closed mission, or `None` if the body had no open mission to report.
    pub fn report(
        &mut self,
        body: &str,
        outcome: Outcome,
        notes: &str,
        answer: Option<String>,
    ) -> Option<Mission> {
        // **The target is settled when its operation is.** A step reported is
        // the operation moved on: where to is its workflow's.
        //
        // **A report the workflow will not take still closes the mission.**
        // Kept open, its Maker would carry a step nobody can hand in — the
        // operation called off under it, or its step moved by hand — and stand
        // at the table with it for good. The step, if it still waits, goes
        // back on offer.
        if let Some((id, step)) = self.active.get(body)?.operation() {
            let step = step.to_string();
            let said = match outcome {
                Outcome::Pass => self.forward(id),
                Outcome::Fail => Some(STUCK),
            };
            if let Err(e) =
                self.operations
                    .advance(id, &Taker::Actor(body.to_string()), said, notes)
            {
                tracing::warn!(operation = id, %step, "a report the workflow refused: {e}");
                let held = self.operations.get(id).and_then(|o| o.holder.as_deref());
                if held == Some(body) {
                    self.operations.released(id);
                }
            }
        }
        let mut mission = self.active.remove(body)?;
        mission.complete(outcome, notes, answer);
        match mission.operation() {
            Some((id, _)) => self.settle_if_over(id),
            None => self.settle(&mission, outcome, notes),
        }
        // A posted mission's report is its poster's; a standing mission's is
        // filed for whoever gives orders here, and, done, it is done there.
        match &mission.origin {
            Origin::Ordered { poster, .. } => self.orders.filed(Filed {
                poster: poster.clone(),
                carrier: body.to_string(),
                brief: mission.prompt.clone(),
                outcome,
                notes: notes.to_string(),
            }),
            Origin::Standing { id, room } => {
                let key = format!("{id}@{room}");
                match outcome {
                    Outcome::Pass => {
                        self.standing_failed.remove(&key);
                        self.standing.insert(key, None);
                    }
                    Outcome::Fail => {
                        self.standing_failed.insert(key, None);
                    }
                }
                self.orders.filed(Filed {
                    poster: EVERY_POSTER.to_string(),
                    carrier: body.to_string(),
                    brief: mission.prompt.clone(),
                    outcome,
                    notes: notes.to_string(),
                });
            }
            Origin::Round { id } => self.round.closed(body, id, outcome),
            _ => {}
        }
        self.done.insert(body.to_string(), mission.clone());
        self.close(body, &mission);
        // The next on the agenda is worked, where it was left.
        self.agenda_rank(body);
        Some(mission)
    }

    /// The outcome a report of done is, on the step operation `id` waits on: a
    /// plain completion where the step has one result, [`PASS`] where it
    /// offers a choice.
    fn forward(&self, id: u64) -> Option<&'static str> {
        let o = self.operations.get(id)?;
        let step = self.operations.workflow_of(o)?.step(o.step()?)?;
        match step.next {
            Next::Single(_) => None,
            Next::Outcomes(_) => Some(PASS),
        }
    }

    /// Send back the work the body's step judges: its mission closes as failed
    /// and the operation goes where the step's [`REJECT`] leads — in the
    /// workflows npcd runs, to be fixed. Returns the closed mission, or `None`
    /// when the body's step may not reject.
    pub fn reject(&mut self, body: &str, why: &str) -> Option<Mission> {
        let m = self.active.get(body)?;
        if !m.may_reject() {
            return None;
        }
        let (id, _) = m.operation()?;
        self.operations
            .advance(id, &Taker::Actor(body.to_string()), Some(REJECT), why)
            .ok()?;
        let mut mission = self.active.remove(body)?;
        mission.complete(Outcome::Fail, why, None);
        self.settle_if_over(id);
        self.done.insert(body.to_string(), mission.clone());
        self.close(body, &mission);
        Some(mission)
    }

    /// Settle the ledger entry of the target a mission that is not an
    /// operation's is about.
    fn settle(&mut self, mission: &Mission, outcome: Outcome, notes: &str) {
        let key = target_of(mission).map(str::to_string);
        self.settle_target(key, outcome, notes);
    }

    fn settle_target(&mut self, key: Option<String>, outcome: Outcome, notes: &str) {
        if let Some(entry) = key.and_then(|k| self.targets.get_mut(&k)) {
            match outcome {
                Outcome::Pass => entry.state = Settled::Done,
                Outcome::Fail => {
                    entry.state = Settled::Stuck;
                    entry.stuck += 1;
                }
            }
            entry.note = notes.to_string();
        }
    }
}

/// Whether `m` is one of the world's standing missions
/// (`docs/zen_moon_mind.md` §12).
fn is_standing(m: &Mission) -> bool {
    matches!(m.origin, Origin::Standing { .. })
}

/// The target a generated mission is about.
fn target_of(m: &Mission) -> Option<&str> {
    match &m.origin {
        Origin::Generated { target, .. } => Some(target),
        _ => None,
    }
}

#[cfg(test)]
#[path = "missions_operation_tests.rs"]
mod operation_tests;

#[cfg(test)]
mod tests {
    use super::Missions;
    use crate::engine::mission::bank::Facts;
    use crate::engine::mission::{Mission, Origin, Outcome, StepOutcome, Todo};
    use crate::engine::mission_gen::world::{Draw, Routine};

    fn a_mission(prompt: &str) -> Mission {
        Mission::new(
            prompt,
            vec![Todo::new("step one"), Todo::new("step two")],
            Origin::Lodged {
                by: "u_op".to_string(),
            },
        )
    }

    /// **Each mission closed — reported or called off — turns a chapter**, and
    /// the mission closed last is kept to close it with.
    #[test]
    fn each_mission_closed_turns_a_chapter() {
        let mut m = Missions::default();
        assert_eq!(m.chapter("bram"), 0);
        m.assign("bram", a_mission("first"));
        m.report("bram", Outcome::Pass, "done", None);
        assert_eq!(m.chapter("bram"), 1);
        assert_eq!(
            m.last_closed("bram").map(|x| x.prompt.as_str()),
            Some("first")
        );
        m.assign("bram", a_mission("second"));
        assert!(m.cancel("bram"));
        assert_eq!(m.chapter("bram"), 2);
        assert_eq!(
            m.last_closed("bram").map(|x| x.prompt.as_str()),
            Some("second"),
            "a mission called off closes a chapter too"
        );
        assert!(!m.cancel("bram"), "nothing open, nothing closed");
        assert_eq!(m.chapter("bram"), 2);
        assert_eq!(m.chapter("cindy"), 0);
    }

    /// **A world with routines of its own hands them out in place of the
    /// bank's**, closed by the report at its table; with none, the bank again.
    #[test]
    fn a_world_s_own_routines_replace_the_bank() {
        let routines = || {
            vec![Routine {
                id: "ward-shift".to_string(),
                brief: "Take a shift on the ward.".to_string(),
                steps: vec!["go to the ward room".to_string()],
                ..Default::default()
            }]
        };
        let facts = Facts {
            table: Some("the watch room"),
            ..Facts::default()
        };
        let mut m = Missions::default();
        m.set_routines(routines(), Draw::Routines);
        let got = m.collect("kettle", &facts).unwrap().clone();
        assert_eq!(got.prompt, "Take a shift on the ward.");
        assert_eq!(
            got.origin,
            Origin::Random {
                routine: "ward-shift".to_string()
            }
        );
        let steps: Vec<&str> = got.todo.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            steps,
            [
                "go to the ward room",
                "go back to the table in the watch room and report it"
            ]
        );

        let mut adopted = Missions::default();
        adopted.adopt_workflows(&mut m);
        let again = adopted.collect("kettle", &facts).unwrap().clone();
        assert_eq!(again.prompt, got.prompt, "adopted with the workflows");

        adopted.set_routines(Vec::new(), Draw::Routines);
        let bank = adopted.collect("biscuit", &facts).unwrap().clone();
        assert!(
            matches!(&bank.origin, Origin::Random { routine } if routine != "ward-shift"),
            "{:?}",
            bank.origin
        );
    }

    /// **A routine handed in turns no chapter**, and a chapter closes with the
    /// work that turned it, not with a routine handed in after it.
    #[test]
    fn a_routine_closed_turns_no_chapter() {
        let routine = || {
            Mission::new(
                "read the breaker panel",
                vec![Todo::new("go to the plant room")],
                Origin::Random {
                    routine: "read-a-station".to_string(),
                },
            )
        };
        let mut m = Missions::default();
        m.assign("bram", routine());
        m.report("bram", Outcome::Pass, "it reads tight", None);
        assert_eq!(m.chapter("bram"), 0);
        assert!(m.last_closed("bram").is_none());

        m.assign("bram", a_mission("the work"));
        m.report("bram", Outcome::Pass, "done", None);
        m.assign("bram", routine());
        m.report("bram", Outcome::Pass, "it reads tight", None);
        assert_eq!(m.chapter("bram"), 1);
        assert_eq!(
            m.last_closed("bram").map(|x| x.prompt.as_str()),
            Some("the work")
        );
    }

    /// **A character made unordered is stood down from what it carries, and
    /// told**; what was lodged for it goes, and marking it again changes
    /// nothing.
    #[test]
    fn an_unordered_character_carries_nothing_and_is_told_so() {
        let mut m = Missions::default();
        m.assign("zen", a_mission("visit the hatchery"));
        m.lodge("zen", a_mission("then the ward"));
        m.assign("wick", a_mission("read a cradle"));
        m.set_unordered("zen", true);
        assert!(m.is_unordered("zen"));
        assert!(!m.is_on_mission("zen"));
        assert!(!m.has_lodged("zen"));
        assert_eq!(m.take_stood_down(), ["zen"]);
        assert!(!m.is_unordered("wick"));
        assert!(m.is_on_mission("wick"), "nobody else is touched");

        m.set_unordered("zen", true);
        assert!(m.take_stood_down().is_empty(), "nothing left to stand down");
        m.set_unordered("zen", false);
        assert!(!m.is_unordered("zen"));
    }

    #[test]
    fn a_lodged_mission_is_drawn_oldest_first_and_then_the_queue_empties() {
        let mut m = Missions::default();
        assert!(!m.has_lodged("bram"));
        m.lodge("bram", a_mission("first"));
        m.lodge("bram", a_mission("second"));
        assert!(m.has_lodged("bram"));

        assert_eq!(m.take_lodged("bram").unwrap().prompt, "first");
        assert_eq!(m.take_lodged("bram").unwrap().prompt, "second");
        assert!(m.take_lodged("bram").is_none(), "queue empties");
        assert!(!m.has_lodged("bram"));
    }

    #[test]
    fn assigning_a_mission_makes_it_the_active_one() {
        let mut m = Missions::default();
        assert!(!m.is_on_mission("wren"));
        m.assign("wren", a_mission("survey the ridge"));
        assert!(m.is_on_mission("wren"));
        assert_eq!(m.active("wren").unwrap().prompt, "survey the ridge");
        // A second assignment replaces the first — a body carries one at a time.
        m.assign("wren", a_mission("read the canon"));
        assert_eq!(m.active("wren").unwrap().prompt, "read the canon");
    }

    #[test]
    fn steps_are_checked_off_and_added_only_on_an_open_mission() {
        let mut m = Missions::default();
        // No mission: nothing to act on.
        assert!(!m.check_off("pax", "step one", StepOutcome::Achieved));
        assert!(!m.add_todo("pax", "a new step"));

        m.assign("pax", a_mission("do the thing"));
        assert!(
            m.check_off("pax", "STEP one", StepOutcome::Achieved),
            "trim + case-insensitive"
        );
        assert!(
            !m.check_off("pax", "step one", StepOutcome::Thwarted),
            "already done"
        );
        assert!(m.add_todo("pax", "a discovered step"));
        assert!(
            !m.add_todo("pax", "  a discovered step  "),
            "open duplicate"
        );
        assert_eq!(m.active("pax").unwrap().todo.len(), 3);
    }

    #[test]
    fn reporting_closes_the_open_mission_and_keeps_it_for_the_operator() {
        let mut m = Missions::default();
        assert!(
            m.report("soren", Outcome::Pass, "n/a", None).is_none(),
            "nothing to report without an open mission"
        );

        m.assign("soren", a_mission("check the record"));
        let closed = m
            .report(
                "soren",
                Outcome::Pass,
                "it holds against the storyline",
                Some("the eastern date is wrong".to_string()),
            )
            .expect("an open mission is reported");
        assert!(!closed.is_open());
        assert_eq!(closed.report.as_ref().unwrap().outcome, Outcome::Pass);

        // The body is now free of its mission, but the operator can still read it.
        assert!(!m.is_on_mission("soren"));
        assert!(m.active("soren").is_none());
        let done = m.done("soren").expect("kept for the operator");
        assert_eq!(done.answer.as_deref(), Some("the eastern date is wrong"));
        assert_eq!(m.latest("soren").unwrap().prompt, "check the record");
    }

    #[test]
    fn collecting_takes_a_lodged_mission_first_then_draws_from_the_bank() {
        let mut m = Missions::default();
        let makers = vec!["Wren".to_string(), "Pax".to_string()];
        let records = vec!["the-charge".to_string()];
        let facts = Facts {
            makers: &makers,
            records: &records,
            ..Facts::default()
        };

        // A lodged mission is taken first, verbatim, and becomes active.
        m.lodge("bram", a_mission("the lodged one"));
        assert_eq!(m.collect("bram", &facts).unwrap().prompt, "the lodged one");
        assert!(m.is_on_mission("bram"));
        assert!(!m.has_lodged("bram"), "the lodged queue is drawn down");

        // With nothing lodged, a routine is drawn from the bank — a valid, open,
        // non-empty, non-destructive mission it can act on.
        let drawn = m.collect("bram", &facts).unwrap().clone();
        assert!(drawn.is_open());
        assert!(!drawn.todo.is_empty(), "a routine gives steps to act on");
        assert!(matches!(drawn.origin, Origin::Random { .. }));
    }

    /// A read struck wherever a mission stands — carried, lodged or pooled —
    /// and each place it was struck named.
    #[test]
    fn a_gone_read_is_struck_from_every_mission_wherever_it_waits() {
        let gone = "layers/stories/gone.md";
        let with_read = |p: &str| {
            Mission::new(
                p,
                vec![Todo::new(format!("read {gone}")), Todo::report("report")],
                Origin::Random {
                    routine: "r".into(),
                },
            )
        };
        let mut m = Missions::default();
        m.assign("vespera", with_read("carried"));
        m.lodge("sila", with_read("lodged"));
        m.pool.push(with_read("pooled"));
        m.assign("paxon", a_mission("reads nothing"));
        let struck = m.strike_gone_reads(&|_, p| p != gone);
        assert_eq!(
            struck,
            [
                (Some("vespera".to_string()), vec![gone.to_string()]),
                (Some("sila".to_string()), vec![gone.to_string()]),
                (None, vec![gone.to_string()]),
            ]
        );
        for mission in [&m.active["vespera"], &m.lodged["sila"][0], &m.pool[0]] {
            assert_eq!(mission.todo.len(), 1, "{}", mission.prompt);
            assert!(mission.todo[0].reports);
        }
    }

    #[test]
    fn cancelling_removes_the_open_mission_without_recording_it() {
        let mut m = Missions::default();
        m.assign("bram", a_mission("do the thing"));
        assert!(m.cancel("bram"), "an open mission is cancelled");
        assert!(!m.is_on_mission("bram"));
        assert!(
            m.done("bram").is_none(),
            "a cancelled mission is not a finished one"
        );
        assert!(!m.cancel("bram"), "nothing to cancel twice");
    }

    #[test]
    fn missions_survive_a_round_trip_through_json() {
        let mut m = Missions::default();
        m.assign("bram", a_mission("do the thing"));
        m.add_todo("bram", "look twice");
        m.lodge("bram", a_mission("then this"));
        m.assign("yen", a_mission("first task"));
        m.report("yen", Outcome::Pass, "done", Some("it holds".to_string()));
        let blob = serde_json::to_value(&m).unwrap();
        let back: Missions = serde_json::from_value(blob).unwrap();
        assert_eq!(back, m);
        assert_eq!(
            back.active("bram").unwrap().todo.last().unwrap().text,
            "look twice"
        );
        assert_eq!(
            back.done("yen").unwrap().answer.as_deref(),
            Some("it holds")
        );
    }

    #[test]
    fn latest_is_the_open_mission_over_a_finished_one() {
        let mut m = Missions::default();
        m.assign("yen", a_mission("first task"));
        m.report("yen", Outcome::Pass, "done", None);
        m.assign("yen", a_mission("second task"));
        // With both a finished and an open mission, the open one is what an
        // operator's question is about.
        assert_eq!(m.latest("yen").unwrap().prompt, "second task");
    }
}
