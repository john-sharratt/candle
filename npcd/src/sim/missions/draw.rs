//! Hands a body its next mission from what was lodged for it, from the table, or
//! from a routine drawn for it.
//!
//! A mission lodged for the body is taken first, oldest first. After it comes the
//! oldest mission waiting at the table that the body may take. A standing mission
//! is given out last. Only then is a routine drawn, from the world's own list or
//! from the built-in bank. A body that gives orders takes nothing from the table,
//! and a review is never taken by the maker who drafted it. A world that draws
//! nothing hands out nothing, so its bodies wait until work is posted.

use super::*;

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

impl Missions {
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

    /// Collect a mission at the desk: the next one lodged for this body, then
    /// the oldest waiting at the table, or — when neither is — a fresh routine:
    /// one of the world's own when it has any ([`Self::set_routines`]),
    /// otherwise a non-destructive one drawn from the [`bank`] against who and
    /// what is around. It becomes the body's open mission, and the collected one
    /// is returned so the desk can hand back its brief.
    ///
    /// `None` in a world that draws nothing ([`Draw::None`]) with nothing set
    /// for this body: it waits at the table until work is posted.
    ///
    /// **One who gives orders takes nothing from the table, ever.** Its
    /// missions come from the people who message it; the table is where it
    /// puts work for those who take orders. Refused here, where the table hands
    /// work out, not only by the callers that ask.
    pub fn collect(&mut self, body: &str, facts: &Facts) -> Option<&Mission> {
        if self.is_unordered(body) {
            return None;
        }
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
    pub(super) fn may_take(&self, body: &str, m: &Mission) -> bool {
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
    pub(super) fn take_pooled(&mut self, body: &str) -> Option<Mission> {
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

    /// Set the world's own routines, and whether they are drawn — empty, and
    /// [`Draw::Routines`], for a world that runs on `missions.yaml` and draws
    /// from the built-in [`bank`].
    pub fn set_routines(&mut self, routines: Vec<Routine>, draw: Draw) {
        self.routines = routines;
        self.draw = draw;
    }

    /// The generated missions waiting at the table.
    pub fn pooled(&self) -> &[Mission] {
        &self.pool
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

    /// Take the next generation's turn number.
    pub fn next_draw(&mut self) -> u64 {
        let turn = self.drawn;
        self.drawn += 1;
        turn
    }

    /// Throw away every mission still waiting on the table — the pool, and what
    /// a character posted for another by name — releasing each target to be
    /// found again. Returns how many there were. A mission an operator lodged
    /// for a character is the operator's, and stays.
    pub fn discard_pool(&mut self) -> usize {
        let mut gone = std::mem::take(&mut self.pool);
        for queue in self.lodged.values_mut() {
            let (posted, kept): (Vec<Mission>, Vec<Mission>) = std::mem::take(queue)
                .into_iter()
                .partition(|m| matches!(m.origin, Origin::Ordered { .. }));
            *queue = kept;
            gone.extend(posted);
        }
        self.lodged.retain(|_, queue| !queue.is_empty());
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

    /// How many missions are lodged for the body and not yet collected.
    pub fn lodged_count(&self, body: &str) -> usize {
        self.lodged.get(body).map_or(0, Vec::len)
    }
}
