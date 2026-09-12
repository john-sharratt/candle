//! The durable small things: promises, orders, verdicts, and who is asleep.
//!
//! # Each of these makes a whole cluster of tasks possible
//!
//! None of them is large, and every one closes something that was unexecutable
//! without it:
//!
//! | Kept here | What it lets a character do |
//! |---|---|
//! | promises | commit to a thing by a time, and be held to it |
//! | orders | give somebody work — the queue that had four consumers and no producer |
//! | verdicts | judge a made thing durably, so `verdict` can be spent |
//! | sleepers | stop, which is what makes a day mean anything |
//!
//! # A promise that leaves nothing behind is a sentence
//!
//! `tell` can already carry "I will have it by nightfall". What it cannot do is
//! make the commitment survive the conversation, and an obligation that does not
//! outlive the meeting cannot span two of them. `remind` binds its argument to
//! *this person's outstanding promises*, so reminding somebody of a thing they
//! never promised is not a mistake available to a character — which is only true
//! because the promises are here to be enumerated.
//!
//! # A verdict is the label on somebody else's work
//!
//! `verdict` is a currency three clusters consume, and filing has the signature
//! `accord, verdict → filed`. A judgement that is only something somebody said
//! cannot be spent by anything, so it attaches to the thing judged and persists.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A question one body put to another and has not had answered.
///
/// The lightest obligation in here and the one that carries a conversation:
/// unlike a promise it has no deadline and nothing to keep, it simply stands
/// until the two of them speak. See [`Ledger::asked`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Asked {
    /// Who asked, and is waiting.
    pub by: String,
    /// Who was asked, and owes the answer.
    pub of: String,
    /// What was asked, in the asker's words.
    pub what: String,
}

/// A commitment one body made to another, with a time on it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Promise {
    pub id: u64,
    /// Who made it.
    pub by: String,
    /// Who it was made to.
    pub to: String,
    /// What was promised, in the maker's own words — read by a model later,
    /// which is what makes free text right here and wrong in a stance.
    pub what: String,
    /// When it comes due, as the world names its times.
    pub by_when: String,
    pub kept: bool,
}

/// Work given to somebody, or lying on a board waiting for a taker.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Order {
    pub id: u64,
    /// What is being asked for.
    pub what: String,
    /// Who set it.
    pub by: String,
    /// On whose authority, when that is not the setter — Keeper issues orders
    /// carrying the Tower Lord's authority and not its own, and *Delegation &
    /// authority* turns on being able to tell the difference.
    pub on_behalf_of: Option<String>,
    /// Who holds it. `None` while it lies on the board.
    pub held_by: Option<String>,
    pub done: bool,
}

/// Somebody's judgement of a made thing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verdict {
    pub id: u64,
    /// What was judged.
    pub on: String,
    pub by: String,
    /// The judgement itself.
    pub judgement: String,
    /// What would change it. Present on a refusal and absent on a pass — the
    /// difference between the two is that a refusal names its remedy.
    pub what_would_change_it: Option<String>,
}

/// Everything durable and small, for one world.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ledger {
    next: u64,
    /// Questions put and not yet answered — see [`Ledger::asked`].
    questions: Vec<Asked>,
    promises: Vec<Promise>,
    orders: Vec<Order>,
    verdicts: Vec<Verdict>,
    /// body → the world time it means to wake at.
    sleepers: BTreeMap<String, String>,
    /// body → the standing orders as that body last read them.
    ///
    /// **The same cursor a posting keeps, for the same reason.** The unheld
    /// orders are readable from anywhere and read the same every time, so `read`
    /// on them was repeatable with an unchanged answer — and `read` is in
    /// `body::ANSWERS`, which brings a character straight back to use what it
    /// learnt. That is the treadmill that had a live cast spending forty-seven
    /// of fifty acts on one line, and fixing it only for
    /// [`crate::sim::posting`] would have moved it here rather than ended it.
    ///
    /// The whole text rather than a count: an order taken and another set in its
    /// place leaves the count identical and the board different.
    #[serde(default)]
    orders_read: BTreeMap<String, String>,
}

impl Ledger {
    pub fn new() -> Ledger {
        Ledger::default()
    }

    fn id(&mut self) -> u64 {
        self.next += 1;
        self.next
    }

    // ---- questions somebody is still waiting on ----

    /// Somebody put a question to somebody else.
    ///
    /// **A question is an obligation, and it was the only one nothing wrote
    /// down.** A promise outlives the meeting because it is here; a question
    /// did not, so it existed for exactly one turn — as one line of perception
    /// among a sagging jacket and a humming conduit — and was gone.
    ///
    /// Measured live: three `ask`s delivered and perceived correctly, and not
    /// one `say` or `tell` in reply. One character was asked directly twice in
    /// forty ticks and reflected both times. They will *start* a conversation
    /// and never return one, because by its next turn there is nothing left
    /// saying anybody is waiting.
    ///
    /// **One outstanding question per pair — the latest.**
    ///
    /// A new question from the same person replaces whatever they were waiting
    /// on, because that is what waiting on somebody means: the thing they most
    /// recently asked and have not had answered.
    ///
    /// This appended instead, deduplicating only on the exact wording, so a
    /// character asking a hundred different things of somebody who never
    /// replied accumulated a hundred debts — and [`Self::awaiting_from`] feeds
    /// the percept, so the situation a character reads would have grown without
    /// bound along with it. Bounded now by the size of the cast, not by how
    /// talkative anybody is.
    pub fn asked(&mut self, by: &str, of: &str, what: &str) {
        self.questions.retain(|q| !(q.by == by && q.of == of));
        self.questions.push(Asked {
            by: by.to_string(),
            of: of.to_string(),
            what: what.to_string(),
        });
    }

    /// `me` said something to `to`, which discharges whatever they were waiting
    /// on.
    ///
    /// **Any speech clears it, not a matched answer.** Deciding whether a
    /// sentence answered a question needs a judge this engine does not have and
    /// should not grow; what it can see is that the two of them are talking,
    /// which is the thing the obligation exists to restart. A debt that could
    /// only be cleared by the right words would outlive every conversation it
    /// was meant to start.
    pub fn answered(&mut self, me: &str, to: &str) {
        self.questions.retain(|q| !(q.of == me && q.by == to));
    }

    /// Somebody left, or is otherwise no longer anybody's to answer.
    pub fn forget_questions(&mut self, who: &str) {
        self.questions.retain(|q| q.by != who && q.of != who);
    }

    /// Who is waiting on an answer from `me`, and what they asked.
    ///
    /// What the percept reports every turn until it is discharged — which is
    /// the whole point, against an event that scrolled past in one.
    pub fn awaiting_from(&self, me: &str) -> Vec<(String, String)> {
        self.questions
            .iter()
            .filter(|q| q.of == me)
            .map(|q| (q.by.clone(), q.what.clone()))
            .collect()
    }

    // ---- promises ----

    pub fn promise(&mut self, by: &str, to: &str, what: &str, by_when: &str) -> Promise {
        let p = Promise {
            id: self.id(),
            by: by.to_string(),
            to: to.to_string(),
            what: what.to_string(),
            by_when: by_when.to_string(),
            kept: false,
        };
        self.promises.push(p.clone());
        p
    }

    pub fn promises(&self) -> &[Promise] {
        &self.promises
    }

    /// What `remind`'s `which` offers: the outstanding things **this** person
    /// promised **you**. Not what you promised them, which is the other
    /// direction and would be an accusation with the parties reversed.
    pub fn owed_to(&self, me: &str, by: &str) -> Vec<String> {
        self.promises
            .iter()
            .filter(|p| !p.kept && p.to == me && p.by == by)
            .map(|p| p.what.clone())
            .collect()
    }

    /// Everything a body still owes anybody, which is what *Closing the day*
    /// reads.
    pub fn owed_by(&self, by: &str) -> Vec<&Promise> {
        self.promises
            .iter()
            .filter(|p| !p.kept && p.by == by)
            .collect()
    }

    /// Mark one kept, by what was promised. `false` when no such promise stands.
    pub fn keep(&mut self, by: &str, what: &str) -> bool {
        let want = what.trim().to_lowercase();
        match self
            .promises
            .iter_mut()
            .find(|p| !p.kept && p.by == by && p.what.to_lowercase() == want)
        {
            Some(p) => {
                p.kept = true;
                true
            }
            None => false,
        }
    }

    // ---- orders ----

    pub fn set_order(&mut self, what: &str, by: &str, on_behalf_of: Option<&str>) -> Order {
        let o = Order {
            id: self.id(),
            what: what.to_string(),
            by: by.to_string(),
            on_behalf_of: on_behalf_of.map(str::to_string),
            held_by: None,
            done: false,
        };
        self.orders.push(o.clone());
        o
    }

    pub fn orders(&self) -> &[Order] {
        &self.orders
    }

    /// What lies on the board unheld — what `claim` offers at an order table.
    pub fn unheld(&self) -> Vec<String> {
        self.orders
            .iter()
            .filter(|o| !o.done && o.held_by.is_none())
            .map(|o| o.what.clone())
            .collect()
    }

    /// The standing orders as they read now, when they say something this body
    /// has not already been told.
    ///
    /// `None` when there is nothing unheld, and `None` when there is but this
    /// body has read exactly that — which is what takes them out of
    /// [`crate::sim::Sim::readable_at`] and so out of the grammar.
    pub fn unheld_unseen_by(&self, who: &str) -> Option<String> {
        let unheld = self.unheld();
        if unheld.is_empty() {
            return None;
        }
        let now = unheld.join("; ");
        match self.orders_read.get(who) {
            Some(seen) if seen == &now => None,
            _ => Some(now),
        }
    }

    /// Remember that this body has read the orders as they now stand.
    pub fn mark_orders_read(&mut self, who: &str, text: &str) {
        self.orders_read.insert(who.to_string(), text.to_string());
    }

    /// What one body is holding.
    pub fn held_by(&self, who: &str) -> Vec<&Order> {
        self.orders
            .iter()
            .filter(|o| !o.done && o.held_by.as_deref() == Some(who))
            .collect()
    }

    /// Hand an order to somebody, whether or not it was on the board.
    pub fn hand_to(&mut self, what: &str, who: &str) -> Result<(), String> {
        let want = what.trim().to_lowercase();
        let Some(o) = self
            .orders
            .iter_mut()
            .find(|o| !o.done && o.what.to_lowercase() == want)
        else {
            return Err(format!("There is no standing order to {what}."));
        };
        if let Some(holder) = &o.held_by {
            if holder != who {
                return Err(format!("{holder} is holding that one."));
            }
        }
        o.held_by = Some(who.to_string());
        Ok(())
    }

    /// Take an unheld order. `Err` when somebody already has it — the same
    /// refusal shape `World::take` gives for a claimed character.
    pub fn take_order(&mut self, what: &str, who: &str) -> Result<(), String> {
        let want = what.trim().to_lowercase();
        let Some(o) = self
            .orders
            .iter_mut()
            .find(|o| !o.done && o.what.to_lowercase() == want)
        else {
            return Err(format!("There is no standing order to {what}."));
        };
        match &o.held_by {
            Some(holder) if holder != who => Err(format!("{holder} is holding that one.")),
            _ => {
                o.held_by = Some(who.to_string());
                Ok(())
            }
        }
    }

    /// Put one back on the board.
    pub fn give_back(&mut self, who: &str) -> Option<String> {
        let o = self
            .orders
            .iter_mut()
            .find(|o| !o.done && o.held_by.as_deref() == Some(who))?;
        o.held_by = None;
        Some(o.what.clone())
    }

    /// Report one finished.
    pub fn finish(&mut self, who: &str, what: &str) -> bool {
        let want = what.trim().to_lowercase();
        match self
            .orders
            .iter_mut()
            .find(|o| !o.done && o.held_by.as_deref() == Some(who) && o.what.to_lowercase() == want)
        {
            Some(o) => {
                o.done = true;
                o.held_by = None;
                true
            }
            None => false,
        }
    }

    // ---- verdicts ----

    pub fn record_verdict(
        &mut self,
        on: &str,
        by: &str,
        judgement: &str,
        what_would_change_it: Option<&str>,
    ) -> Verdict {
        let v = Verdict {
            id: self.id(),
            on: on.to_string(),
            by: by.to_string(),
            judgement: judgement.to_string(),
            what_would_change_it: what_would_change_it.map(str::to_string),
        };
        self.verdicts.push(v.clone());
        v
    }

    pub fn verdicts(&self) -> &[Verdict] {
        &self.verdicts
    }

    /// Every judgement standing against one thing.
    pub fn verdicts_on(&self, on: &str) -> Vec<&Verdict> {
        let want = on.trim().to_lowercase();
        self.verdicts
            .iter()
            .filter(|v| v.on.to_lowercase() == want)
            .collect()
    }

    // ---- sleep ----

    /// Lie down until a named time. Replaces an earlier intention, because a
    /// body has one.
    pub fn sleep(&mut self, body: &str, until: &str) {
        self.sleepers.insert(body.to_string(), until.to_string());
    }

    pub fn asleep(&self, body: &str) -> Option<&str> {
        self.sleepers.get(body).map(String::as_str)
    }

    /// Wake somebody. `true` when they were actually asleep — being woken is a
    /// thing that happens *to* a character, so the caller is the world.
    pub fn wake(&mut self, body: &str) -> bool {
        self.sleepers.remove(body).is_some()
    }

    pub fn sleepers(&self) -> impl Iterator<Item = (&String, &String)> {
        self.sleepers.iter()
    }
}

/// The times the world's clock can name, which is what `sleep`'s `until` binds
/// to. A closed set, because "until I feel like it" is not a time anything can
/// wake you at.
pub const WAKE_TIMES: &[&str] = &[
    "dawn",
    "midday",
    "dusk",
    "nightfall",
    "the next bell",
    "an hour from now",
];

#[cfg(test)]
mod tests {
    use super::*;

    /// **A question stands until the two of them speak.**
    ///
    /// This is the whole point of writing it down: as an event it existed for
    /// one turn and was gone, and a live cast delivered three questions and
    /// returned zero answers.
    #[test]
    fn a_question_is_owed_by_the_person_it_was_put_to() {
        let mut l = Ledger::new();
        l.asked("m1", "m2", "what orders have changed");
        assert_eq!(
            l.awaiting_from("m2"),
            vec![("m1".to_string(), "what orders have changed".to_string())]
        );
        // Not the other way round — the asker owes nothing.
        assert!(l.awaiting_from("m1").is_empty());
    }

    /// Speaking to them clears it, whatever was said. Judging whether a
    /// sentence *answered* would need a judge this engine does not have, and a
    /// debt only the right words could clear would outlive every conversation
    /// it was meant to start.
    #[test]
    fn speaking_to_somebody_discharges_what_they_were_waiting_on() {
        let mut l = Ledger::new();
        l.asked("m1", "m2", "what orders have changed");
        l.answered("m2", "m1");
        assert!(l.awaiting_from("m2").is_empty());
    }

    /// And it clears only that pair. Answering one person does not discharge
    /// what somebody else is waiting on.
    #[test]
    fn answering_one_person_leaves_everybody_else_waiting() {
        let mut l = Ledger::new();
        l.asked("m1", "m2", "what orders have changed");
        l.asked("m3", "m2", "where the third era is");
        l.answered("m2", "m1");
        assert_eq!(
            l.awaiting_from("m2"),
            vec![("m3".to_string(), "where the third era is".to_string())]
        );
    }

    /// Asking the same thing twice is one debt, not two — otherwise a
    /// character that repeats itself builds a list nobody could ever clear.
    #[test]
    fn asking_the_same_thing_twice_is_still_one_question() {
        let mut l = Ledger::new();
        l.asked("m1", "m2", "what orders have changed");
        l.asked("m1", "m2", "what orders have changed");
        assert_eq!(l.awaiting_from("m2").len(), 1);
    }

    /// **And asking something *different* replaces it rather than stacking.**
    ///
    /// What somebody is waiting on is the last thing they asked. Appending
    /// instead meant a character talking to somebody who never replies built an
    /// unbounded list — and the percept renders every entry, so the situation
    /// it reads would grow without bound too.
    #[test]
    fn a_new_question_replaces_what_they_were_waiting_on() {
        let mut l = Ledger::new();
        l.asked("m1", "m2", "where the chips are");
        l.asked("m1", "m2", "who came through the door");
        assert_eq!(
            l.awaiting_from("m2"),
            vec![("m1".to_string(), "who came through the door".to_string())]
        );
    }

    /// Bounded by the cast, not by how talkative it is: several askers each
    /// keep their own one outstanding question.
    #[test]
    fn each_asker_keeps_exactly_one_outstanding_question() {
        let mut l = Ledger::new();
        for i in 0..50 {
            l.asked("m1", "m2", &format!("question {i}"));
            l.asked("m3", "m2", &format!("other {i}"));
        }
        assert_eq!(l.awaiting_from("m2").len(), 2);
    }

    /// **Every way of speaking to somebody discharges it, not just the one in
    /// the room.**
    ///
    /// The obligation is cleared by `tell` and by `message`, and the second was
    /// missed at first: a character that walked out and texted the answer
    /// stayed marked as owing one, with a standing nudge it could only clear by
    /// finding the person again in the flesh. Both call the same function, so
    /// what this pins is that the function does not care which act reached it.
    #[test]
    fn any_way_of_reaching_them_clears_what_they_were_waiting_on() {
        let mut l = Ledger::new();
        l.asked("m1", "m2", "who came through the door");
        assert_eq!(l.awaiting_from("m2").len(), 1);
        // Whether that came from speech in the room or a handset two levels up,
        // the ledger sees one thing: m2 reached m1.
        l.answered("m2", "m1");
        assert!(l.awaiting_from("m2").is_empty());
    }

    /// Somebody retired takes their questions with them, in both directions —
    /// a debt owed to nobody is one a character can never discharge.
    #[test]
    fn a_body_that_is_gone_leaves_no_obligations_behind() {
        let mut l = Ledger::new();
        l.asked("m1", "m2", "a");
        l.asked("m2", "m3", "b");
        l.forget_questions("m2");
        assert!(l.awaiting_from("m2").is_empty());
        assert!(l.awaiting_from("m3").is_empty());
    }

    #[test]
    fn a_promise_is_owed_to_the_person_it_was_made_to() {
        let mut l = Ledger::new();
        l.promise("m1", "m2", "the eastern span read through", "dusk");
        assert_eq!(
            l.owed_to("m2", "m1"),
            vec!["the eastern span read through".to_string()]
        );
        // Not the other way round: reminding somebody of your own promise is a
        // different act with the parties reversed.
        assert!(l.owed_to("m1", "m2").is_empty());
    }

    #[test]
    fn a_kept_promise_stops_being_offered_for_reminding() {
        let mut l = Ledger::new();
        l.promise("m1", "m2", "the span", "dusk");
        assert!(l.keep("m1", "the span"));
        assert!(l.owed_to("m2", "m1").is_empty());
        assert!(!l.keep("m1", "the span"), "kept twice");
    }

    #[test]
    fn what_a_body_owes_is_readable_without_naming_who_it_owes_it_to() {
        let mut l = Ledger::new();
        l.promise("m1", "m2", "a", "dawn");
        l.promise("m1", "m3", "b", "dusk");
        l.promise("m2", "m1", "c", "dusk");
        assert_eq!(l.owed_by("m1").len(), 2);
    }

    #[test]
    fn an_order_can_be_set_taken_and_reported_done() {
        let mut l = Ledger::new();
        l.set_order("survey the eastern ridge", "keeper", Some("the tower lord"));
        assert_eq!(l.unheld(), vec!["survey the eastern ridge".to_string()]);

        l.take_order("survey the eastern ridge", "c1").unwrap();
        assert!(l.unheld().is_empty(), "a held order stayed on the board");
        assert_eq!(l.held_by("c1").len(), 1);

        assert!(l.finish("c1", "survey the eastern ridge"));
        assert!(l.held_by("c1").is_empty());
        assert!(
            l.unheld().is_empty(),
            "a finished order went back on the board"
        );
    }

    #[test]
    fn an_order_records_whose_authority_it_carries() {
        let mut l = Ledger::new();
        let o = l.set_order("hold the gate", "keeper", Some("the tower lord"));
        assert_eq!(o.by, "keeper");
        assert_eq!(o.on_behalf_of.as_deref(), Some("the tower lord"));
    }

    #[test]
    fn an_order_somebody_else_holds_is_refused_and_names_the_holder() {
        let mut l = Ledger::new();
        l.set_order("mind the drill", "keeper", None);
        l.take_order("mind the drill", "c1").unwrap();
        let err = l.take_order("mind the drill", "c2").unwrap_err();
        assert!(err.contains("c1"), "the holder was not named: {err}");
    }

    #[test]
    fn giving_an_order_back_puts_it_where_the_next_taker_will_find_it() {
        let mut l = Ledger::new();
        l.set_order("mind the drill", "keeper", None);
        l.take_order("mind the drill", "c1").unwrap();
        assert_eq!(l.give_back("c1").as_deref(), Some("mind the drill"));
        assert_eq!(l.unheld().len(), 1);
        assert!(l.give_back("c1").is_none(), "gave back what was not held");
    }

    #[test]
    fn an_order_nobody_set_cannot_be_taken() {
        let mut l = Ledger::new();
        assert!(l.take_order("invent a reason", "c1").is_err());
        assert!(l.hand_to("invent a reason", "c1").is_err());
    }

    #[test]
    fn a_verdict_attaches_to_the_thing_and_a_refusal_names_its_remedy() {
        let mut l = Ledger::new();
        l.record_verdict(
            "the third era",
            "m2",
            "it cannot be filed",
            Some("dates either side"),
        );
        l.record_verdict("the third era", "m3", "it reads well enough", None);

        let on = l.verdicts_on("The Third Era");
        assert_eq!(on.len(), 2, "case decided whether a judgement was found");
        assert!(on.iter().any(|v| v.what_would_change_it.is_some()));
        assert!(on.iter().any(|v| v.what_would_change_it.is_none()));
    }

    #[test]
    fn sleeping_replaces_an_earlier_intention_and_waking_is_done_to_you() {
        let mut l = Ledger::new();
        l.sleep("m1", "dawn");
        l.sleep("m1", "midday");
        assert_eq!(l.asleep("m1"), Some("midday"));
        assert!(l.wake("m1"));
        assert!(!l.wake("m1"), "woken twice");
        assert!(l.asleep("m1").is_none());
    }

    #[test]
    fn ids_are_handed_out_once_each_across_every_kind() {
        // One counter, so nothing shares an id with anything else in the ledger.
        let mut l = Ledger::new();
        let p = l.promise("a", "b", "x", "dawn").id;
        let o = l.set_order("y", "a", None).id;
        let v = l.record_verdict("z", "a", "no", None).id;
        assert_eq!([p, o, v], [1, 2, 3]);
    }
}
