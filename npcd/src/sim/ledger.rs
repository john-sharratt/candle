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
    promises: Vec<Promise>,
    orders: Vec<Order>,
    verdicts: Vec<Verdict>,
    /// body → the world time it means to wake at.
    sleepers: BTreeMap<String, String>,
}

impl Ledger {
    pub fn new() -> Ledger {
        Ledger::default()
    }

    fn id(&mut self) -> u64 {
        self.next += 1;
        self.next
    }

    // ---- promises ----

    pub fn promise(
        &mut self,
        by: &str,
        to: &str,
        what: &str,
        by_when: &str,
    ) -> Promise {
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
        match self.orders.iter_mut().find(|o| {
            !o.done && o.held_by.as_deref() == Some(who) && o.what.to_lowercase() == want
        }) {
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
        assert!(l.unheld().is_empty(), "a finished order went back on the board");
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
        l.record_verdict("the third era", "m2", "it cannot be filed", Some("dates either side"));
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
