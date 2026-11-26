//! The scheduler's entry points for events: delivering them, handing back a
//! turn that never happened, and recording what a character did.
//!
//! A turn the decode never took is given back, so the events it was owed are
//! read by the turn that runs. A person's request stands until the character
//! has worked at it and told them the outcome. Delivery never refuses an event,
//! since a filter that drops evidence before it lands makes a false belief
//! permanent.

use super::*;

impl Scheduler {
    /// Give a turn that never happened back to the character: its events go
    /// back to the head of the queue, and it is owed a turn `after_ms` from
    /// `now_ms`.
    ///
    /// **A turn the decode never took is not a turn taken.** `begin_tick` had
    /// already drained the queue and taken what was owed, so a decode that
    /// failed — or an engine not yet loaded — ended the tick with what the
    /// character had been told consumed and nothing done about it, and nothing
    /// scheduled again. The events were in its window and never in front of the
    /// model. Handed back, they are read by the turn that does run; they are not
    /// written into the window a second time.
    pub fn retry(&self, npc_id: u64, now_ms: u64, after_ms: u64, events: Vec<Event>) -> bool {
        let waker = {
            let mut g = self.inboxes.lock().unwrap();
            let Some(i) = g.get_mut(&npc_id) else {
                return false;
            };
            i.replayed += events.len();
            for e in events.into_iter().rev() {
                i.queue.push_front(e);
            }
            i.due_at = now_ms.saturating_add(after_ms);
            i.scheduled = true;
            i.owed = true;
            Arc::clone(&i.waker)
        };
        waker.notify_one();
        true
    }

    /// Bring a character straight back, because its last act answered with
    /// something. See [`Inbox::think_again`].
    pub fn think_again(&self, npc_id: u64) -> bool {
        let waker = {
            let mut g = self.inboxes.lock().unwrap();
            let Some(i) = g.get_mut(&npc_id) else {
                return false;
            };
            i.think_again();
            Arc::clone(&i.waker)
        };
        waker.notify_one();
        true
    }

    /// Complete an act's row with the result that came back after it was
    /// recorded — in the character's window and in the Pulse ring both, so the
    /// feed and the transcript say the same thing. See [`Window::amend_npc`].
    ///
    /// Returns whether the row was found in the window.
    pub fn amend_act(&self, npc_id: u64, from: &str, to: String) -> bool {
        let found = self
            .inboxes
            .lock()
            .unwrap()
            .get_mut(&npc_id)
            .is_some_and(|i| i.window.amend_npc(from, to.clone()));
        let mut recent = self.recent.lock().unwrap();
        if let Some(act) = recent
            .iter_mut()
            .rev()
            .filter(|r| r.npc_id == npc_id)
            .flat_map(|r| r.acts.iter_mut())
            .find(|a| a.as_str() == from)
        {
            *act = to;
        }
        found
    }

    /// Whether this character has acted since news last reached it; true of
    /// one with no inbox, which nothing has reached. See
    /// [`Inbox::acted_since_news`].
    pub fn acted_since_news(&self, npc_id: u64) -> bool {
        self.inboxes
            .lock()
            .unwrap()
            .get(&npc_id)
            .is_none_or(Inbox::acted_since_news)
    }

    /// Hold `request`, which `party` made of this character on the phone,
    /// until the character has worked at it and told them what came of it.
    ///
    /// **A request has to outlast the turn it arrived on.** It reaches the
    /// mind once, as what a conversation settled; the next thing to wake the
    /// mind — a Drone taking up work, a message on a channel — begins a turn
    /// that says nothing of it, and a mind that reads only what is in front
    /// of it connected to a terminal and then rested. Held here, it is
    /// restated at the head of every turn ([`Self::asked`]) and stopping is
    /// refused while it stands.
    pub fn ask(&self, npc_id: u64, party: &str, request: &str, world_ms: u64) {
        if let Some(i) = self.inboxes.lock().unwrap().get_mut(&npc_id) {
            i.asked = Some(Asked {
                party: party.to_string(),
                request: request.trim().to_string(),
                at_ms: world_ms,
                worked: false,
            });
        }
    }

    /// What is still asked of this character at `world_ms`: who asked, and
    /// what. See [`Inbox::asked`].
    pub fn asked(&self, npc_id: u64, world_ms: u64) -> Option<(String, String)> {
        self.inboxes
            .lock()
            .unwrap()
            .get_mut(&npc_id)
            .and_then(|i| i.asked(world_ms))
    }

    /// Note that an act of this character's came back with something: from
    /// here, telling whoever asked is an answer.
    pub fn worked(&self, npc_id: u64) {
        if let Some(asked) = self
            .inboxes
            .lock()
            .unwrap()
            .get_mut(&npc_id)
            .and_then(|i| i.asked.as_mut())
        {
            asked.worked = true;
        }
    }

    /// Note that this character has told `party` something. What `party`
    /// asked stands no longer once the character has worked at it; told
    /// before any act of its own came back, it was promised an answer and is
    /// still owed one.
    pub fn told(&self, npc_id: u64, party: &str) {
        if let Some(i) = self.inboxes.lock().unwrap().get_mut(&npc_id) {
            if i.asked
                .as_ref()
                .is_some_and(|a| a.worked && a.party == party)
            {
                i.asked = None;
            }
        }
    }

    /// Whether this character is due the standing task. See [`Inbox::nudge_due`].
    pub fn nudge_due(&self, npc_id: u64, world_ms: u64, after_ms: u64) -> bool {
        self.inboxes
            .lock()
            .unwrap()
            .get(&npc_id)
            .is_some_and(|i| i.nudge_due(world_ms, after_ms))
    }

    /// Note that this character's `rest` landed at `world_ms`. See
    /// [`Inbox::rested`].
    pub fn rested(&self, npc_id: u64, world_ms: u64) {
        if let Some(i) = self.inboxes.lock().unwrap().get_mut(&npc_id) {
            i.rested(world_ms);
        }
    }

    /// The next event sequence number, for a caller that builds an [`Event`] to
    /// place directly into a tick already in hand rather than [`Self::deliver`]
    /// it — the call to report at the table rides every tick this way, and `deliver`'s
    /// `due_at = 0` would make that a tight re-tick loop.
    pub fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, AtomicOrdering::Relaxed)
    }

    /// Deliver an event to a character. `false` if there is no such character.
    pub fn deliver(&self, npc_id: u64, world_ms: u64, salience: Salience, kind: EventKind) -> bool {
        if let (EventKind::Talk { .. } | EventKind::Purpose { .. }, Some(talk)) =
            (&kind, self.talk.get())
        {
            talk(npc_id, kind);
            return true;
        }
        let seq = self.seq.fetch_add(1, AtomicOrdering::Relaxed);
        let event = Event::new(seq, world_ms, salience, kind);
        let wakes = event.wakes();
        let waker = {
            let mut inboxes = self.inboxes.lock().unwrap();
            let Some(inbox) = inboxes.get_mut(&npc_id) else {
                return false;
            };
            inbox.push(event);
            if !wakes {
                return true;
            }
            // Due immediately: preempted whatever it planned, or roused out of
            // a wait. The waker coalesces — a burst of twenty arrivals stores
            // one permit, and the one wake drains the whole burst. An arrival
            // *during* a decode stores its permit too, so the character is
            // re-woken the moment its task returns to its wait.
            inbox.due_at = 0;
            Arc::clone(&inbox.waker)
        };
        waker.notify_one();
        true
    }

    /// Deliver to every character at once — a world event nobody is exempt from.
    ///
    /// `world_ms` is a function rather than a value: characters live in
    /// different worlds at different paces, so there is no single instant to
    /// stamp them all with. Thunder over two worlds happens at two times.
    pub fn broadcast<F>(&self, world_ms: F, salience: Salience, kind: EventKind) -> usize
    where
        F: Fn(u64) -> u64,
    {
        // Ids collected first, and the lock released, because `deliver` takes it
        // again per character.
        let ids: Vec<u64> = self.inboxes.lock().unwrap().keys().copied().collect();
        ids.iter()
            .filter(|id| self.deliver(**id, world_ms(**id), salience, kind.clone()))
            .count()
    }

    /// Read a character's window under the lock, without cloning it out.
    ///
    /// A closure rather than a getter: the window lives inside the inbox map, and
    /// handing out a clone of every turn to answer one request would copy the
    /// whole tail per poll of the Pulse view.
    pub fn window_of<T, F>(&self, npc_id: u64, f: F) -> Option<T>
    where
        F: FnOnce(&Window) -> T,
    {
        let inboxes = self.inboxes.lock().unwrap();
        inboxes.get(&npc_id).map(|i| f(&i.window))
    }

    /// The scheduler's own monotonic clock, in milliseconds — the `now_ms`
    /// every character task ticks against, so a tick's `at_ms` and its age in
    /// the Pulse ring are measured on one clock.
    pub fn now_ms(&self) -> u64 {
        self.born.elapsed().as_millis() as u64
    }
}
