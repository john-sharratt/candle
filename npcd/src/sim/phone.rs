//! The thing a character carries that lets it reach somebody who is not here.
//!
//! # A phone, not a room
//!
//! The distinction is the whole design, and getting it wrong produces something
//! that looks close and behaves nothing like it:
//!
//! | | A room | A phone |
//! |---|---|---|
//! | How many at once | exactly one | as many threads as you have |
//! | Who hears | whoever is standing there | whoever is on that thread |
//! | When | now, while you are there | whenever, and it waits |
//! | Reach | the walls | anybody you have a thread with |
//! | Losing it | you walk out | it is taken off you |
//!
//! So messaging is **not** `say` and `tell` pointed at a different audience. A
//! character texts while standing in a room, in the middle of something else,
//! to somebody a level away — and the room hears none of it. Room acts and
//! phone acts are different acts because they reach different people at
//! different times, which is a distinction the world can represent.
//!
//! # It is carried, so it can be absent
//!
//! The phone is an ordinary item in the pack (`Kind::Gear`). A character without
//! one has no threads, so every argument that binds to a thread is empty, so the
//! phone acts are not in the grammar at all. Nothing declares that — it is the
//! same empty-set rule that takes `move_to` out of a room with nowhere to go.
//!
//! That also means a phone can be given away, taken, lost or left behind, and
//! being out of contact is a state the world can put somebody in rather than a
//! flag somebody has to remember to set.
//!
//! # A direct thread is found, never duplicated
//!
//! Reaching out twice to the same person continues the conversation you already
//! have with them. Two threads over one pair is how a message lands in the one
//! nobody is reading.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The item that makes any of this reachable. Carried like anything else.
pub const PHONE: &str = "handset";

/// What every world's standing channel is called.
///
/// # Why there is one at all
///
/// A character alone could reach nobody it could not already see. The percept
/// says who is in this room and who is visible from it; `move_to` offers rooms
/// with no indication of who is in any of them. So the whole building was, to a
/// solitary character, a list of identical doors — and "go and find somebody"
/// was not a choice being declined, it was a choice with nothing to act on.
///
/// The channel is the answer, and the reason it is a *comms channel* rather
/// than a find-people feature is that finding people is downstream of it. A
/// character that says where it is and what it is working on has handed
/// everybody else a room name with a person in it, which is the fact that was
/// missing. Anything else a cast needs to say to people who are not in front of
/// it goes the same way.
///
/// # Why it is joined rather than opened
///
/// Every character is on it from the moment it has a handset, without doing
/// anything. An act for joining would be one more thing to choose on a turn, it
/// would be chosen at the wrong times, and a character that had not yet chosen
/// it would be in exactly the isolated state this exists to end.
///
/// # The name is the plainest one available
///
/// It was *the open channel*, which is a description wearing the clothes of a
/// name — there is only one, nobody has to distinguish it from a closed one,
/// and the extra word is read on every line of every feed and every prompt for
/// no information. A character says *the channel* because that is what a person
/// would say.
///
/// It avoids the word *band*, which was the other candidate: the vault has
/// rooms called *band one* and *band two*, and `move_to` binds destinations to a
/// closed set of room names. A channel sharing a word with a room is a
/// character reading "the band" in one line and a destination list containing
/// "band one" in the next.
pub const CHANNEL: &str = "the channel";

/// What sort of thread it is.
///
/// Derived from the membership rather than declared, because it *is* the
/// membership: a direct thread that gains a third member has become a group,
/// and nothing should have to be told so separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Two people.
    Direct,
    /// Three or more.
    Group,
}

/// One conversation on a phone, and who is on it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Thread {
    pub id: String,
    /// What a group is called. A direct thread carries the other party's name,
    /// which is how a character refers to it and how the argument binds.
    pub name: String,
    /// Everybody on it, by the name the world writes down — the same address a
    /// character uses in a room, so one person is never two.
    pub members: Vec<String>,
    /// What has been said, oldest first.
    pub messages: Vec<Message>,
    /// Whether it is still live. A thread everybody has left is kept rather
    /// than deleted: what was said in it still happened.
    pub open: bool,
    /// Whether it was opened as a named group.
    ///
    /// **Not derived from the head count**, which is what it used to be. A
    /// named group of three that loses one is still that group — it keeps its
    /// name, its history and whatever it was for. Deriving the kind from the
    /// membership silently turned it into a direct thread, so the two remaining
    /// members stopped seeing it under the name they had been using and started
    /// seeing each other's instead.
    ///
    /// A pair that *gains* somebody does become a group, because it has grown
    /// into one. The asymmetry is real: growing changes what a conversation is,
    /// shrinking does not.
    pub named: bool,
    /// Whether this thread is part of the world rather than something somebody
    /// started. See [`CHANNEL`].
    ///
    /// **Exempt from [`KEEP_THREADS`], and it has to be.** Eviction drops the
    /// thread that went quiet longest ago, and a standing channel is precisely
    /// the thread that goes quiet — a cast that spends an hour talking in rooms
    /// and opens a few groups meanwhile would have its channel collected out
    /// from under it. Nothing would report that: every character would simply
    /// stop being offered it, and the capability they are meant to have without
    /// asking would be gone until the daemon restarted.
    #[serde(default)]
    pub permanent: bool,
    /// When anything was last said on it, on [`Threads`]'s own counter.
    ///
    /// Not a clock: nothing here needs to know *when*, only which of two
    /// conversations went quiet first — see [`KEEP_THREADS`]. `serde(default)`
    /// because saved worlds predate the field, and a thread that loads at zero
    /// is simply the first candidate to be forgotten, which is the right answer
    /// for one nothing has said anything on since the daemon restarted.
    #[serde(default)]
    pub touched: u64,
    /// Which message each member arrived at.
    ///
    /// **A thread's history is not handed to somebody who joins it.** Bringing
    /// a third person onto a conversation gives them what is said from then on;
    /// what was said before is the inviter's to repeat or not, which is what the
    /// act says and what anybody would expect. Without this the newcomer's
    /// phone lit up with an argument they had not been part of.
    ///
    /// Kept as an arrival index rather than by marking the backlog "seen" by
    /// them, because that would be a record claiming somebody read something
    /// they never got.
    joined_at: BTreeMap<String, usize>,
    /// How much of the thread each member's *mind* has already been handed.
    ///
    /// **Separate from `seen_by`, which is a different question.** `seen_by`
    /// says who has looked at a message; this says who has been *told* it
    /// arrived. A character is told once, on the moment after it was sent, and
    /// then it is up to the character whether it looks — exactly as the map's
    /// [`npc_map::Attention`] cursor works for what a body perceives, and for
    /// the same reason: without a cursor the sweep either re-delivers the whole
    /// thread every moment or delivers nothing at all.
    delivered: BTreeMap<String, usize>,
}

/// Most messages a direct thread keeps.
///
/// **A thread is a rolling record, not an archive.** Two characters left
/// talking for a week would otherwise grow a `Vec` that is read in full by
/// `unread_for` on every situation build, serialised whole with the world, and
/// handed to `read` in one turn — and a character cannot act on four hundred
/// messages any more than a person can.
///
/// Sized to the useful span rather than to a byte budget: what a character
/// needs is the shape of the conversation it is in, and anything older than
/// this is something it would have had to write down.
pub const KEEP_DIRECT: usize = 64;

/// Most messages a group keeps.
///
/// Larger than a direct thread because a group of six produces six voices into
/// one record, so the same number of *turns each* needs more room — not because
/// a group's history is worth more.
pub const KEEP_GROUP: usize = 128;

/// One thing said on a thread.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub from: String,
    /// What was meant. The narrator renders it, as with every other speech act
    /// — a phone does not make a character write its own dialogue.
    pub intent: String,
    /// Who has seen it. **Not a flag on the message**, because "delivered" is a
    /// fact about each reader rather than about the message: a group of four
    /// has four answers to it.
    pub seen_by: Vec<String>,
}

impl Thread {
    pub fn kind(&self) -> Kind {
        match self.named || self.members.len() > 2 {
            true => Kind::Group,
            false => Kind::Direct,
        }
    }

    pub fn has(&self, who: &str) -> bool {
        let want = who.trim().to_lowercase();
        self.members.iter().any(|m| m.to_lowercase() == want)
    }

    /// Everybody but one.
    pub fn others(&self, than: &str) -> Vec<String> {
        let me = than.trim().to_lowercase();
        self.members
            .iter()
            .filter(|m| m.to_lowercase() != me)
            .cloned()
            .collect()
    }

    /// What this thread is called *to one member* — a group by its name, a
    /// direct thread by the other person's.
    pub fn as_named_to(&self, who: &str) -> String {
        match self.kind() {
            Kind::Group => self.name.clone(),
            Kind::Direct => self
                .others(who)
                .first()
                .cloned()
                .unwrap_or_else(|| self.name.clone()),
        }
    }

    /// Say something on this thread, dropping the oldest once it has grown past
    /// what a thread of its kind keeps.
    ///
    /// **Every index into `messages` moves when the front is dropped**, and
    /// there are two of them — where each member joined, and how much each
    /// member's mind has been handed. Trimming without shifting both would hand
    /// somebody a backlog it had already been told about, or show a newcomer
    /// the argument it joined after: the indices stay valid numbers and start
    /// pointing at the wrong messages, which is the kind of wrong nothing
    /// reports.
    pub fn say(&mut self, message: Message) {
        self.messages.push(message);
        let keep = match self.kind() {
            Kind::Direct => KEEP_DIRECT,
            Kind::Group => KEEP_GROUP,
        };
        let Some(drop) = self.messages.len().checked_sub(keep).filter(|d| *d > 0) else {
            return;
        };
        self.messages.drain(..drop);
        for at in self
            .joined_at
            .values_mut()
            .chain(self.delivered.values_mut())
        {
            *at = at.saturating_sub(drop);
        }
    }

    /// What this member's mind has not been told about yet.
    ///
    /// Never the backlog from before they joined — the rule [`Thread::unread_for`]
    /// keeps, for the same reason.
    pub fn undelivered(&self, who: &str) -> &[Message] {
        let from = self
            .delivered
            .get(who)
            .copied()
            .unwrap_or(0)
            .max(self.arrived_at(who));
        self.messages.get(from..).unwrap_or_default()
    }

    /// Mark everything said so far as handed to this member's mind.
    pub fn mark_delivered(&mut self, who: &str) {
        self.delivered.insert(who.to_string(), self.messages.len());
    }

    /// Where in the thread somebody came in. Zero for anybody who was here from
    /// the start.
    pub fn arrived_at(&self, who: &str) -> usize {
        self.joined_at.get(who).copied().unwrap_or(0)
    }

    /// What `who` has not read yet — of what was said since they arrived.
    pub fn unread_for(&self, who: &str) -> usize {
        self.messages
            .iter()
            .skip(self.arrived_at(who))
            .filter(|m| m.from != who && !m.seen_by.iter().any(|s| s == who))
            .count()
    }
}

/// How many conversations a world keeps.
///
/// # Why a bound on the threads and not only on the messages in them
///
/// [`KEEP_DIRECT`] and [`KEEP_GROUP`] cap what one conversation holds, which
/// bounds nothing on its own: `open_group` mints a new thread every time it is
/// called and a character can call it on any turn it likes. Unbounded threads
/// times a capped thread is unbounded messages, so the cap that was there was
/// answering a question nobody had asked.
///
/// Direct threads are self-limiting — `reach` returns the existing one for a
/// pair, so they are bounded by the cast and the people talking to it. Groups
/// are the pressure, and this is the bound on them.
///
/// # What is dropped
///
/// The conversation that went quiet longest ago. A world holding sixty-four
/// live threads and one nobody has said anything on in days has an obvious
/// candidate, and [`Thread::touched`] is what names it.
pub const KEEP_THREADS: usize = 64;

/// Every thread in a world.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Threads {
    threads: BTreeMap<String, Thread>,
    next: u64,
    /// Ticks once per thing said, so threads can be ordered by how recently
    /// anybody spoke on them. See [`Thread::touched`].
    #[serde(default)]
    clock: u64,
}

impl Threads {
    pub fn new() -> Threads {
        Threads::default()
    }

    pub fn is_empty(&self) -> bool {
        self.threads.is_empty()
    }

    pub fn len(&self) -> usize {
        self.threads.len()
    }

    pub fn get(&self, id: &str) -> Option<&Thread> {
        self.threads.get(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Thread> {
        self.threads.values()
    }

    /// The open threads `who` is on.
    pub fn of(&self, who: &str) -> Vec<&Thread> {
        self.threads
            .values()
            .filter(|t| t.open && t.has(who))
            .collect()
    }

    /// What `who` calls each of its threads — what a thread-bound argument
    /// offers.
    pub fn names_for(&self, who: &str) -> Vec<String> {
        self.of(who)
            .into_iter()
            .map(|t| t.as_named_to(who))
            .collect()
    }

    /// Find one of `who`'s threads by the name `who` calls it.
    pub fn by_name_for(&self, who: &str, name: &str) -> Option<&Thread> {
        let want = name.trim().to_lowercase();
        self.of(who)
            .into_iter()
            .find(|t| t.as_named_to(who).to_lowercase() == want || t.id == want)
    }

    fn id_by_name_for(&self, who: &str, name: &str) -> Option<String> {
        self.by_name_for(who, name).map(|t| t.id.clone())
    }

    /// The direct thread between two people, if there is one.
    pub fn direct_between(&self, a: &str, b: &str) -> Option<&Thread> {
        self.threads
            .values()
            .find(|t| t.open && t.kind() == Kind::Direct && t.has(a) && t.has(b))
    }

    /// The counter's next value, stamped on whatever just happened.
    fn now(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Drop the conversations that went quiet longest ago, down to
    /// [`KEEP_THREADS`].
    ///
    /// Called after opening one, which is the only way the count goes up. The
    /// thread just created is the most recently touched, so it is never the one
    /// evicted — a caller is always handed back a thread that exists.
    ///
    /// **A permanent thread is neither a candidate nor counted.** It is part of
    /// the world rather than something somebody started, and it is also the
    /// likeliest thing in here to be quiet — see [`Thread::permanent`].
    ///
    /// Counted but not evictable, it would silently spend one of the world's
    /// conversation slots for ever: a cast would hold sixty-three conversations
    /// instead of sixty-four and nothing would say why. So the bound is over
    /// the threads that actually accumulate, and a world's total is
    /// [`KEEP_THREADS`] plus its channels.
    fn forget_quiet_threads(&mut self) {
        let mut age: Vec<(u64, String)> = self
            .threads
            .values()
            .filter(|t| !t.permanent)
            .map(|t| (t.touched, t.id.clone()))
            .collect();
        let Some(over) = age.len().checked_sub(KEEP_THREADS).filter(|o| *o > 0) else {
            return;
        };
        age.sort();
        for (_, id) in age.into_iter().take(over) {
            self.threads.remove(&id);
        }
    }

    /// Open a thread with somebody, or return the one that already stands.
    pub fn reach(&mut self, from: &str, to: &str) -> String {
        if let Some(t) = self.direct_between(from, to) {
            return t.id.clone();
        }
        self.next += 1;
        let touched = self.now();
        let id = format!("th{}", self.next);
        self.threads.insert(
            id.clone(),
            Thread {
                id: id.clone(),
                name: to.to_string(),
                members: vec![from.to_string(), to.to_string()],
                messages: Vec::new(),
                open: true,
                named: false,
                permanent: false,
                joined_at: BTreeMap::new(),
                delivered: BTreeMap::new(),
                touched,
            },
        );
        self.forget_quiet_threads();
        id
    }

    /// The world's standing channel, with `who` on it.
    ///
    /// **Join-or-create, and idempotent**, because it is called on every
    /// arrival and an arrival is not a special occasion: a character that walks
    /// back into a world it was already in must not find itself on the channel
    /// twice, and must not mint a second channel beside the first.
    ///
    /// A latecomer comes in at the end of what has been said, exactly as
    /// [`Threads::invite`] does. Handing somebody four hours of a conversation
    /// they were not part of is not a welcome, it is a backlog — and on a
    /// channel every character is on, it would be the largest one in the world.
    ///
    /// Returns the thread id.
    pub fn join_channel(&mut self, name: &str, who: &str) -> String {
        if let Some(t) = self
            .threads
            .values_mut()
            .find(|t| t.permanent && t.name == name)
        {
            if !t.has(who) {
                t.joined_at.insert(who.to_string(), t.messages.len());
                t.members.push(who.to_string());
            }
            return t.id.clone();
        }
        self.next += 1;
        let touched = self.now();
        let id = format!("th{}", self.next);
        self.threads.insert(
            id.clone(),
            Thread {
                id: id.clone(),
                name: name.to_string(),
                members: vec![who.to_string()],
                messages: Vec::new(),
                open: true,
                // A channel is a named group from the first member, not once it
                // has three. Derived from the head count it would spend its
                // first two joins looking like a direct thread, and each of
                // those two would see it under the *other* member's name.
                named: true,
                permanent: true,
                joined_at: BTreeMap::new(),
                delivered: BTreeMap::new(),
                touched,
            },
        );
        // Deliberately no `forget_quiet_threads` here. A permanent thread is
        // not a candidate for eviction, so creating one cannot be what pushes
        // the world over its bound — and calling it would evict an ordinary
        // conversation to make room for something that never needed room.
        id
    }

    /// Start a named group outright, rather than growing one from a pair.
    pub fn open_group(&mut self, name: &str, members: &[String]) -> String {
        self.next += 1;
        let touched = self.now();
        let id = format!("th{}", self.next);
        self.threads.insert(
            id.clone(),
            Thread {
                id: id.clone(),
                name: name.to_string(),
                members: members.to_vec(),
                messages: Vec::new(),
                open: true,
                named: true,
                permanent: false,
                joined_at: BTreeMap::new(),
                delivered: BTreeMap::new(),
                touched,
            },
        );
        // **The one that can run away.** A character may open a group on any
        // turn, and nothing about a group makes the next one a duplicate of the
        // last, so this is the act that would grow the world without bound.
        self.forget_quiet_threads();
        id
    }

    /// Say something on one of `from`'s threads, named the way `from` names it.
    pub fn send(&mut self, from: &str, named: &str, intent: &str) -> Result<Vec<String>, String> {
        let Some(id) = self.id_by_name_for(from, named) else {
            return Err(format!("You have no conversation with {named}."));
        };
        // Stamped before the borrow, so a live conversation is the last thing
        // `forget_quiet_threads` would take and a silent one the first.
        let touched = self.now();
        let t = self.threads.get_mut(&id).expect("just found");
        t.touched = touched;
        t.say(Message {
            from: from.to_string(),
            intent: intent.to_string(),
            // The sender has seen what they sent.
            seen_by: vec![from.to_string()],
        });
        Ok(t.others(from))
    }

    /// Bring somebody onto one of `by`'s threads.
    pub fn invite(&mut self, by: &str, named: &str, who: &str) -> Result<Kind, String> {
        let Some(id) = self.id_by_name_for(by, named) else {
            return Err(format!("You have no conversation with {named}."));
        };
        let t = self.threads.get_mut(&id).expect("just found");
        if t.has(who) {
            return Err(format!("{who} is already on it."));
        }
        // They come in at the end of what has been said, not the start of it.
        t.joined_at.insert(who.to_string(), t.messages.len());
        t.members.push(who.to_string());
        Ok(t.kind())
    }

    /// Leave one. The thread closes when there is nobody left to answer.
    pub fn leave(&mut self, who: &str, named: &str) -> Result<(), String> {
        let Some(id) = self.id_by_name_for(who, named) else {
            return Err(format!("You have no conversation with {named}."));
        };
        let t = self.threads.get_mut(&id).expect("just found");
        let want = who.trim().to_lowercase();
        t.members.retain(|m| m.to_lowercase() != want);
        if t.members.len() < 2 {
            t.open = false;
        }
        Ok(())
    }

    /// Mark everything on a thread read by somebody, and hand back what they
    /// had not seen.
    pub fn read(&mut self, who: &str, named: &str) -> Result<Vec<Message>, String> {
        let Some(id) = self.id_by_name_for(who, named) else {
            return Err(format!("You have no conversation with {named}."));
        };
        let t = self.threads.get_mut(&id).expect("just found");
        let from_index = t.joined_at.get(who).copied().unwrap_or(0);
        let mut fresh = Vec::new();
        for m in t.messages.iter_mut().skip(from_index) {
            if m.from != who && !m.seen_by.iter().any(|s| s == who) {
                m.seen_by.push(who.to_string());
                fresh.push(m.clone());
            }
        }
        Ok(fresh)
    }

    /// How many messages are waiting for somebody, across everything.
    ///
    /// **What makes a phone a phone.** A room reaches a character because it is
    /// standing there; a phone reaches it because something arrived while it
    /// was doing something else, and this is the number that says so.
    /// Everything nobody has told `who` about yet, thread by thread.
    ///
    /// **The messaging half of the perception sweep.** What a body perceives
    /// comes off the map through `npc_map::Attention`; a phone reaches somebody
    /// who is nowhere near, so it has no delta and needs its own cursor. This
    /// is that: read once per moment, handed to the mind, marked.
    ///
    /// Returned as the name `who` calls the thread, so what the character is
    /// told matches the argument it would use to answer.
    pub fn undelivered_for(&self, who: &str) -> Vec<(String, Kind, Vec<Message>)> {
        self.of(who)
            .into_iter()
            .filter_map(|t| {
                let waiting: Vec<Message> = t
                    .undelivered(who)
                    .iter()
                    .filter(|m| m.from != who)
                    .cloned()
                    .collect();
                (!waiting.is_empty()).then(|| (t.as_named_to(who), t.kind(), waiting))
            })
            .collect()
    }

    /// Mark every thread as told, for one person.
    pub fn mark_delivered_for(&mut self, who: &str) {
        for t in self.threads.values_mut() {
            if t.has(who) {
                t.mark_delivered(who);
            }
        }
    }

    pub fn waiting_for(&self, who: &str) -> usize {
        self.of(who).into_iter().map(|t| t.unread_for(who)).sum()
    }

    /// Which threads have something waiting, by the name their reader uses.
    pub fn waiting_names_for(&self, who: &str) -> Vec<String> {
        self.of(who)
            .into_iter()
            .filter(|t| t.unread_for(who) > 0)
            .map(|t| t.as_named_to(who))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── what a thread keeps ─────────────────────────────────────────────────

    /// A pair that has been talking for a long time.
    fn long_direct(n: usize) -> Threads {
        let mut p = Threads::new();
        p.reach("Wren", "Soren");
        for i in 0..n {
            p.send("Soren", "Wren", &format!("line {i}")).unwrap();
            p.send("Wren", "Soren", &format!("line {i}")).unwrap();
        }
        p
    }

    /// **A thread is a rolling record, not an archive.** Two characters left
    /// talking would otherwise grow a list read in full on every situation
    /// build and handed whole to a character that cannot act on it.
    #[test]
    fn a_direct_thread_keeps_only_the_most_recent() {
        let mut p = Threads::new();
        p.reach("Wren", "Soren");
        for i in 0..(KEEP_DIRECT + 40) {
            p.send("Soren", "Wren", &format!("line {i}")).unwrap();
        }
        let t = p.direct_between("Wren", "Soren").expect("a thread");
        assert_eq!(t.messages.len(), KEEP_DIRECT);
        // The newest survive and the oldest are the ones gone.
        assert!(t
            .messages
            .last()
            .unwrap()
            .intent
            .ends_with(&format!("{}", KEEP_DIRECT + 39)));
        assert!(t.messages.iter().all(|m| m.intent != "line 0"));
    }

    #[test]
    fn a_group_keeps_more_than_a_pair_does() {
        let mut p = Threads::new();
        p.open_group(
            "the boundary",
            &["Wren".into(), "Soren".into(), "Perrin".into()],
        );
        for i in 0..(KEEP_GROUP + 10) {
            p.send("Soren", "the boundary", &format!("line {i}"))
                .unwrap();
        }
        assert_eq!(
            p.by_name_for("Wren", "the boundary")
                .unwrap()
                .messages
                .len(),
            KEEP_GROUP
        );
        // A group of six puts six voices into one record, so the same number of
        // turns each needs more room than a pair does.
        const _: () = assert!(KEEP_GROUP > KEEP_DIRECT);
    }

    /// **Trimming moves every index into the message list.** A newcomer's
    /// arrival point and each member's delivery cursor both point into it, and
    /// shifting one but not the other leaves valid numbers aimed at the wrong
    /// messages — which nothing reports.
    #[test]
    fn trimming_keeps_the_arrival_point_pointing_at_the_right_place() {
        let mut p = Threads::new();
        p.reach("Wren", "Soren");
        for i in 0..10 {
            p.send("Soren", "Wren", &format!("early {i}")).unwrap();
        }
        p.invite("Wren", "Soren", "Perrin").unwrap();
        // Perrin sees nothing said before the invitation…
        let thread = p
            .names_for("Perrin")
            .first()
            .expect("Perrin is on something")
            .clone();
        assert_eq!(
            p.by_name_for("Perrin", &thread)
                .unwrap()
                .unread_for("Perrin"),
            0
        );

        // …and still sees nothing of it after the backlog has been trimmed away.
        let soren_calls_it = p.names_for("Soren").first().cloned().expect("on it");
        for i in 0..(KEEP_GROUP + 20) {
            p.send("Soren", &soren_calls_it, &format!("later {i}"))
                .unwrap();
        }
        let t = p.by_name_for("Perrin", &thread).unwrap();
        assert!(t.messages.len() <= KEEP_GROUP);
        assert!(
            t.unread_for("Perrin") <= t.messages.len(),
            "an arrival index outlived the messages it pointed at"
        );
        assert!(t.messages.iter().all(|m| !m.intent.starts_with("early")));
    }

    #[test]
    fn a_long_thread_does_not_grow_without_bound() {
        let p = long_direct(200);
        for t in p.iter() {
            assert!(t.messages.len() <= KEEP_GROUP.max(KEEP_DIRECT));
        }
    }

    // ── delivery ────────────────────────────────────────────────────────────

    /// **A message is handed to a mind exactly once.** Without a cursor the
    /// sweep either re-delivers the whole thread every moment or delivers
    /// nothing — and it delivered nothing, so a sender was told "they will see
    /// it when they next look" while the recipient was never told there was
    /// anything to look at.
    #[test]
    fn what_is_waiting_is_handed_over_once_and_then_not_again() {
        let mut p = Threads::new();
        p.reach("Wren", "Soren");
        p.send("Soren", "Wren", "that the redoubt burned twice")
            .unwrap();

        let waiting = p.undelivered_for("Wren");
        assert_eq!(waiting.len(), 1);
        let (thread, kind, messages) = &waiting[0];
        assert_eq!(
            thread, "Soren",
            "a direct thread is named for the other party"
        );
        assert_eq!(*kind, Kind::Direct);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].intent, "that the redoubt burned twice");

        p.mark_delivered_for("Wren");
        assert!(p.undelivered_for("Wren").is_empty(), "handed over twice");

        // And the next one is handed over on its own.
        p.send("Soren", "Wren", "and again in the spring").unwrap();
        let again = p.undelivered_for("Wren");
        assert_eq!(again.len(), 1);
        assert_eq!(again[0].2.len(), 1, "the whole thread was re-delivered");
    }

    /// You are not told what you said yourself.
    #[test]
    fn your_own_messages_are_not_delivered_back_to_you() {
        let mut p = Threads::new();
        p.reach("Wren", "Soren");
        p.send("Wren", "Soren", "mine").unwrap();
        assert!(p.undelivered_for("Wren").is_empty());
        assert_eq!(p.undelivered_for("Soren").len(), 1);
    }

    /// A newcomer is told what is said from then on, never the backlog.
    #[test]
    fn a_newcomer_is_not_handed_the_argument_it_joined_after() {
        let mut p = Threads::new();
        p.reach("Wren", "Soren");
        p.send("Soren", "Wren", "before").unwrap();
        p.invite("Wren", "Soren", "Perrin").unwrap();
        assert!(
            p.undelivered_for("Perrin").is_empty(),
            "the backlog was delivered to somebody who was not there for it"
        );
        let thread = p.names_for("Perrin").first().cloned().expect("on it");
        p.send("Soren", &thread, "after").unwrap();
        let waiting = p.undelivered_for("Perrin");
        assert_eq!(waiting.len(), 1);
        assert_eq!(waiting[0].2[0].intent, "after");
    }

    /// A group is named by its name, for everybody on it.
    #[test]
    fn a_group_is_delivered_under_the_name_everybody_calls_it() {
        let mut p = Threads::new();
        p.open_group(
            "the boundary",
            &["Wren".into(), "Soren".into(), "Perrin".into()],
        );
        p.send(
            "Soren",
            "the boundary",
            "that nobody should date anything yet",
        )
        .unwrap();
        for who in ["Wren", "Perrin"] {
            let waiting = p.undelivered_for(who);
            assert_eq!(waiting.len(), 1, "{who} was told nothing");
            assert_eq!(waiting[0].0, "the boundary");
            assert_eq!(waiting[0].1, Kind::Group);
        }
    }

    // ── how much a world keeps ──────────────────────────────────────────────

    /// A member list, which `open_group` takes owned.
    fn names(ns: &[&str]) -> Vec<String> {
        ns.iter().map(|s| s.to_string()).collect()
    }

    /// **Capping the messages in a thread bounds nothing on its own.** A
    /// character may open a group on any turn, and every call mints a new one —
    /// so unbounded threads times a capped thread is unbounded messages, and
    /// the world grew for as long as anybody kept talking.
    #[test]
    fn a_world_stops_collecting_conversations() {
        let mut p = Threads::new();
        for i in 0..(KEEP_THREADS * 3) {
            p.open_group(&format!("crew {i}"), &names(&["Wren", "Soren", "Orion"]));
        }
        assert_eq!(p.len(), KEEP_THREADS);
    }

    /// And what goes is what went quiet longest ago — not what was opened
    /// longest ago. A thread from this morning that people are still talking on
    /// outlives one opened since and abandoned.
    #[test]
    fn the_conversation_nobody_has_spoken_on_is_the_one_dropped() {
        let mut p = Threads::new();
        // Opened first, and never spoken on since.
        p.open_group("dead air", &names(&["Wren", "Soren", "Orion"]));
        // Opened after it, and still being used — so it is the *newer* thread
        // that is also the busier one. Ordering it the other way would pass
        // against an "oldest opened" rule too, and prove nothing.
        p.reach("Wren", "Soren");
        p.send("Wren", "Soren", "still going").unwrap();

        // One more conversation than the world keeps, so exactly one must go.
        for i in 0..(KEEP_THREADS - 1) {
            p.open_group(&format!("crew {i}"), &names(&["Wren", "Orion"]));
        }

        assert_eq!(p.len(), KEEP_THREADS);
        assert!(
            p.direct_between("Wren", "Soren").is_some(),
            "the live conversation went before the abandoned one"
        );
        assert!(
            p.iter().all(|t| t.name != "dead air"),
            "the abandoned conversation survived"
        );
    }

    // ── the standing channel ────────────────────────────────────────────────

    /// Joining is idempotent: an arrival is not a special occasion, and a
    /// character re-entering a world it was already in calls this again.
    #[test]
    fn joining_the_channel_twice_neither_duplicates_it_nor_the_member() {
        let mut p = Threads::new();
        let a = p.join_channel(CHANNEL, "Maker-01");
        let b = p.join_channel(CHANNEL, "Maker-01");
        assert_eq!(a, b, "a second channel was minted");
        assert_eq!(p.len(), 1);
        assert_eq!(p.get(&a).unwrap().members, vec!["Maker-01"]);

        p.join_channel(CHANNEL, "Maker-02");
        assert_eq!(p.get(&a).unwrap().members.len(), 2);
    }

    /// It is a named group from its first member — not once it has three.
    ///
    /// Derived from the head count, a channel with two people on it would be a
    /// direct thread, and each of the two would see it under the *other's*
    /// name rather than under the channel's.
    #[test]
    fn the_channel_is_a_group_from_the_first_member() {
        let mut p = Threads::new();
        p.join_channel(CHANNEL, "Maker-01");
        p.join_channel(CHANNEL, "Maker-02");
        let t = p.by_name_for("Maker-01", CHANNEL).expect("on it");
        assert_eq!(t.kind(), Kind::Group);
        assert_eq!(p.names_for("Maker-01"), vec![CHANNEL]);
        assert_eq!(p.names_for("Maker-02"), vec![CHANNEL]);
    }

    /// **The failure this was built to prevent.**
    ///
    /// Eviction takes the thread that went quiet longest ago, and a standing
    /// channel is exactly the thread that goes quiet. Collected, it would take
    /// with it the one capability every character is supposed to have without
    /// asking — and nothing anywhere would report it.
    #[test]
    fn the_channel_survives_a_world_that_fills_up_with_conversations() {
        let mut p = Threads::new();
        p.join_channel(CHANNEL, "Maker-01");
        p.join_channel(CHANNEL, "Maker-02");
        // Quiet from here on, while the world does nothing but open groups.
        for i in 0..(KEEP_THREADS * 3) {
            p.open_group(&format!("crew {i}"), &names(&["Wren", "Soren", "Orion"]));
        }
        assert!(
            p.by_name_for("Maker-01", CHANNEL).is_some(),
            "the channel was collected out from under the cast"
        );
        // And the bound still holds over everything that is not permanent.
        assert_eq!(p.iter().filter(|t| !t.permanent).count(), KEEP_THREADS);
    }

    /// A latecomer joins at the end of what has been said. On a channel every
    /// character is on, the backlog would be the largest in the world.
    #[test]
    fn somebody_joining_the_channel_is_not_handed_what_it_missed() {
        let mut p = Threads::new();
        p.join_channel(CHANNEL, "Maker-01");
        p.join_channel(CHANNEL, "Maker-02");
        p.send("Maker-01", CHANNEL, "that I am in the chronicle")
            .unwrap();

        p.join_channel(CHANNEL, "Maker-03");
        assert_eq!(
            p.waiting_for("Maker-03"),
            0,
            "a newcomer was handed the backlog"
        );
        assert_eq!(p.waiting_for("Maker-02"), 1, "a member lost theirs");

        // What is said after it arrives does reach it.
        p.send("Maker-01", CHANNEL, "and staying a while").unwrap();
        assert_eq!(p.waiting_for("Maker-03"), 1);
    }

    /// Everybody on it hears it, and nobody else does. The ordinary group rule
    /// — asserted here because this is the thread the whole cast shares, so a
    /// leak would reach every character at once.
    #[test]
    fn what_is_said_on_the_channel_reaches_every_member_and_no_one_else() {
        let mut p = Threads::new();
        for who in ["Maker-01", "Maker-02", "Maker-03"] {
            p.join_channel(CHANNEL, who);
        }
        p.reach("Maker-01", "Outsider");

        let reached = p
            .send("Maker-01", CHANNEL, "that the lift is stuck on five")
            .unwrap();
        assert_eq!(reached.len(), 2);
        assert_eq!(p.waiting_for("Outsider"), 0, "it carried off the channel");
        for who in ["Maker-02", "Maker-03"] {
            assert_eq!(p.waiting_for(who), 1, "{who} was not told");
        }
    }

    /// A thread is handed back by id, so evicting the one just opened would
    /// give a caller an id for something that does not exist.
    #[test]
    fn opening_a_conversation_never_evicts_the_one_it_returns() {
        let mut p = Threads::new();
        let mut last = String::new();
        for i in 0..(KEEP_THREADS * 2) {
            last = p.open_group(&format!("crew {i}"), &names(&["Wren", "Orion"]));
            assert!(
                p.get(&last).is_some(),
                "handed back a thread it had dropped"
            );
        }
        assert!(p.get(&last).is_some());
    }

    /// Talking on a thread keeps it. The counter has to move on `send`, or
    /// every thread stays at the value it was opened with and eviction silently
    /// becomes "oldest opened" — which drops the conversation you are having.
    #[test]
    fn saying_something_keeps_a_conversation_alive() {
        let mut p = Threads::new();
        p.reach("Wren", "Soren");
        let opened = p.direct_between("Wren", "Soren").unwrap().touched;
        p.send("Wren", "Soren", "anything").unwrap();
        assert!(
            p.direct_between("Wren", "Soren").unwrap().touched > opened,
            "a thread went quiet the moment it was opened"
        );
    }

    #[test]
    fn reaching_out_twice_to_one_person_continues_the_same_thread() {
        let mut p = Threads::new();
        let a = p.reach("Wren", "Soren");
        assert_eq!(p.reach("Wren", "Soren"), a, "a second thread over one pair");
        assert_eq!(
            p.reach("Soren", "Wren"),
            a,
            "the other side was a new thread"
        );
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn a_direct_thread_becomes_a_group_by_gaining_somebody() {
        let mut p = Threads::new();
        p.reach("Wren", "Soren");
        assert_eq!(
            p.invite("Wren", "Soren", "Perrin Vastwood").unwrap(),
            Kind::Group
        );
    }

    /// **What was said before somebody joined is not handed to them.**
    ///
    /// Bringing a third person onto a conversation gives them what is said from
    /// then on; the backlog is the inviter's to repeat or not. Otherwise a
    /// newcomer's phone lights up with an argument they were not part of.
    #[test]
    fn joining_a_thread_does_not_hand_over_what_was_said_before() {
        let mut p = Threads::new();
        p.reach("Wren", "Soren");
        p.send("Wren", "Soren", "one").unwrap();
        p.send("Wren", "Soren", "two").unwrap();
        p.invite("Wren", "Soren", "Orion Vance").unwrap();

        assert_eq!(
            p.waiting_for("Orion Vance"),
            0,
            "the backlog was handed over"
        );
        assert_eq!(p.waiting_for("Soren"), 2, "the original member lost theirs");

        p.send("Wren", "Soren", "three").unwrap();
        assert_eq!(
            p.waiting_for("Orion Vance"),
            1,
            "what came after did not arrive"
        );
    }

    /// A named group that shrinks to two is still that group. Growing changes
    /// what a conversation is; shrinking does not.
    #[test]
    fn a_named_group_keeps_its_name_when_it_loses_a_member() {
        let mut p = Threads::new();
        p.open_group(
            "the sweep",
            &["Wren".into(), "Soren".into(), "Orion Vance".into()],
        );
        p.leave("Wren", "the sweep").unwrap();
        assert_eq!(p.names_for("Soren"), vec!["the sweep"]);
    }

    #[test]
    fn a_thread_is_named_after_the_other_person_and_a_group_after_itself() {
        let mut p = Threads::new();
        p.reach("Wren", "Soren");
        assert_eq!(p.names_for("Wren"), vec!["Soren"]);
        assert_eq!(
            p.names_for("Soren"),
            vec!["Wren"],
            "both sides see the other"
        );

        p.open_group(
            "the eastern sweep",
            &["Wren".into(), "Soren".into(), "Orion Vance".into()],
        );
        let mut names = p.names_for("Wren");
        names.sort();
        assert_eq!(names, vec!["Soren", "the eastern sweep"]);
    }

    #[test]
    fn what_is_sent_reaches_everybody_else_on_the_thread() {
        let mut p = Threads::new();
        p.reach("Wren", "Soren");
        p.invite("Wren", "Soren", "Orion Vance").unwrap();
        let reached = p.send("Wren", "Soren", "the ridge is clear").unwrap();
        assert_eq!(reached.len(), 2, "a group message reached only one");
        assert!(reached.contains(&"Orion Vance".to_string()));
    }

    #[test]
    fn sending_to_somebody_you_have_no_thread_with_is_refused() {
        let mut p = Threads::new();
        p.reach("Wren", "Soren");
        let err = p.send("Wren", "Perrin Vastwood", "hello").unwrap_err();
        assert!(err.contains("no conversation"), "{err}");
    }

    /// **The property that makes it a phone rather than a room.** A message
    /// waits for somebody who was doing something else, and keeps waiting until
    /// they look.
    #[test]
    fn a_message_waits_until_it_is_read() {
        let mut p = Threads::new();
        p.reach("Wren", "Soren");
        p.send("Wren", "Soren", "the ridge is clear").unwrap();

        assert_eq!(p.waiting_for("Soren"), 1);
        assert_eq!(p.waiting_for("Wren"), 0, "your own message waited for you");
        assert_eq!(p.waiting_names_for("Soren"), vec!["Wren"]);

        let fresh = p.read("Soren", "Wren").unwrap();
        assert_eq!(fresh.len(), 1);
        assert_eq!(p.waiting_for("Soren"), 0);
        assert!(p.read("Soren", "Wren").unwrap().is_empty(), "read twice");
    }

    #[test]
    fn each_reader_has_their_own_answer_to_whether_they_have_seen_it() {
        let mut p = Threads::new();
        p.open_group(
            "the sweep",
            &["Wren".into(), "Soren".into(), "Orion Vance".into()],
        );
        p.send("Wren", "the sweep", "moving now").unwrap();

        assert_eq!(p.waiting_for("Soren"), 1);
        assert_eq!(p.waiting_for("Orion Vance"), 1);
        p.read("Soren", "the sweep").unwrap();
        assert_eq!(p.waiting_for("Soren"), 0);
        assert_eq!(
            p.waiting_for("Orion Vance"),
            1,
            "one person reading it marked it read for everybody"
        );
    }

    #[test]
    fn leaving_a_pair_closes_it_and_leaving_a_group_does_not() {
        let mut p = Threads::new();
        p.reach("Wren", "Soren");
        p.leave("Wren", "Soren").unwrap();
        assert!(
            p.of("Soren").is_empty(),
            "a conversation of one stayed open"
        );

        let g = p.open_group(
            "the sweep",
            &["Wren".into(), "Soren".into(), "Orion Vance".into()],
        );
        p.leave("Wren", "the sweep").unwrap();
        assert!(p.get(&g).unwrap().open, "a group closed when one left");
        assert_eq!(p.get(&g).unwrap().members.len(), 2);
    }

    #[test]
    fn a_closed_thread_is_kept_because_what_was_said_still_happened() {
        let mut p = Threads::new();
        p.reach("Wren", "Soren");
        p.send("Wren", "Soren", "before it ended").unwrap();
        p.leave("Wren", "Soren").unwrap();
        assert_eq!(p.len(), 1, "the thread was deleted");
        assert!(
            p.names_for("Soren").is_empty(),
            "a closed one was still offered"
        );
    }

    #[test]
    fn what_a_thread_offers_is_stable_across_two_builds() {
        let build = || {
            let mut p = Threads::new();
            p.reach("Wren", "Soren");
            p.open_group(
                "the sweep",
                &["Wren".into(), "Orion Vance".into(), "Soren".into()],
            );
            p
        };
        assert_eq!(build(), build());
        assert_eq!(build().names_for("Wren"), build().names_for("Wren"));
    }
}
