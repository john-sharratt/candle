//! Who is deciding what the tower does about what is pressing on it.
//!
//! # A decision nobody owns gets asked about
//!
//! "What should the tower do next?" put to a cast of equals has no answer
//! anywhere in the world, so it goes between them — each asks the others to
//! decide, none does, and the question is the loop. What breaks it is not a
//! rule against asking but an owner: one character holding the decision, and
//! the others able to see that they do.
//!
//! The ledger already has the shape of that. An order is put up, exactly one
//! character takes it ([`Ledger::take_order`] is exclusive), and a second one
//! reaching for it is told who has it. So a pressing cause of the tower's
//! ([`Tower::pressing`]) is put on the order board as an order set by the tower
//! itself, a character takes it from the order table, and the tower's acts are
//! then that character's to make.
//!
//! # The board follows the tower
//!
//! [`Sim::watch_tower`] reconciles the two each beat. A cause that has come up
//! is put on the board once and announced once; one that has passed — the
//! contact folded away from, the reserve restored — is taken down, held or not,
//! together with whatever was decided about it, so that it is a new decision if
//! it returns. One that stands and has been settled is not put up again.
//!
//! # What it asks of a character
//!
//! Nothing before somebody holds a decision: the tower can be commanded by
//! anyone standing at its console, and a character that does so on its own
//! account is not in anybody's way. Once a decision is held, the tower is its
//! holder's to command, and anybody else who reaches for it is told whose it is
//! rather than being let contradict them.

use crate::sim::ledger::Order;
use crate::sim::Sim;

/// Who the world's own orders are set by.
pub const SETTER: &str = "the tower";

/// One of the tower's open decisions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    /// The sentence on the board.
    pub what: String,
    /// The body deciding it, if anybody is.
    pub held_by: Option<String>,
}

impl From<&Order> for Decision {
    fn from(o: &Order) -> Decision {
        Decision {
            what: o.what.clone(),
            held_by: o.held_by.clone(),
        }
    }
}

impl Sim {
    /// Bring the order board into line with what is pressing on the tower.
    /// Returns what the crew should be told about the causes that have just come
    /// up, once each.
    pub fn watch_tower(&mut self) -> Vec<String> {
        let Some(tower) = self.tower.as_ref() else {
            return Vec::new();
        };
        let pressing = tower.pressing();
        let standing: Vec<&str> = pressing.iter().map(|p| p.decision.as_str()).collect();
        self.ledger.withdraw_unless(SETTER, &standing);

        let mut told = Vec::new();
        for cause in &pressing {
            if !self.ledger.set_by(SETTER, &cause.decision) {
                self.ledger.set_order(&cause.decision, SETTER, None);
                told.push(format!(
                    "{} Nobody is deciding it yet. To make it yours, claim \"{}\".",
                    cause.alarm, cause.decision
                ));
            }
        }
        told
    }

    /// The area the tower is commanded from, which is where a klaxon is heard.
    /// `None` in a world that has no tower or no console to command it from.
    pub fn tower_area(&self) -> Option<&str> {
        self.tower.as_ref()?;
        self.part_tools
            .iter()
            .find(|(_, tools)| tools.iter().any(|t| t == "command_tower"))
            .and_then(|(place, _)| place.split_once('/'))
            .map(|(area, _)| area)
    }

    /// The decisions on the board now, held or not.
    pub fn tower_decisions(&self) -> Vec<Decision> {
        self.ledger
            .open_set_by(SETTER)
            .into_iter()
            .map(Decision::from)
            .collect()
    }

    /// What stands between this body and the tower: a decision somebody else is
    /// holding, as `(the decision, whoever holds it)`. `None` when the body is
    /// free to command it — nobody holds anything, or the body is a holder.
    pub fn tower_held_against(&self, body: &str) -> Option<(String, String)> {
        let decisions = self.tower_decisions();
        if decisions.iter().any(|d| d.held_by.as_deref() == Some(body)) {
            return None;
        }
        decisions
            .into_iter()
            .find_map(|d| d.held_by.map(|holder| (d.what, holder)))
    }

    /// A body has acted on the tower, so what it was holding is decided. Returns
    /// how many decisions that closed.
    pub fn settle_tower_decisions(&mut self, body: &str) -> usize {
        self.ledger.finish_held_set_by(body, SETTER)
    }
}

#[cfg(test)]
mod tests {
    use npc_map::MapSet;

    use super::*;
    use crate::sim::field::Resource;
    use crate::sim::seed;
    use crate::sim::tower::{Coord, Tower};
    use crate::sim::upkeep::CONTACT_EVERY;

    fn sim() -> Sim {
        let mut s = Sim::new();
        let mut t = Tower::new("the Redoubt", Coord::new(0, 0));
        t.put(Resource::Energy, 4_000);
        s.tower = Some(t);
        s
    }

    fn with_contact() -> Sim {
        let mut s = sim();
        s.tower.as_mut().unwrap().advance(CONTACT_EVERY);
        s
    }

    fn whats(s: &Sim) -> Vec<String> {
        s.tower_decisions().into_iter().map(|d| d.what).collect()
    }

    /// The klaxon is heard where the tower is commanded from, and a world with
    /// no tower has nowhere for one to sound.
    #[test]
    fn the_tower_is_commanded_from_its_own_area() {
        let map = MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"))
            .expect("the shipped maps must load");
        assert_eq!(
            seed::battle_cities(Some(&map)).tower_area(),
            Some("tower-redoubt")
        );
        assert_eq!(seed::vault(Some(&map)).tower_area(), None);
        assert_eq!(Sim::new().tower_area(), None);
    }

    /// **A cause that comes up is put on the board and announced, once.**
    #[test]
    fn a_pressing_cause_is_put_on_the_board_once() {
        let mut s = with_contact();
        let told = s.watch_tower();
        assert_eq!(told.len(), 1, "{told:?}");
        assert!(told[0].contains("closing on the tower"), "{told:?}");
        assert_eq!(whats(&s).len(), 1);
        assert!(whats(&s)[0].contains("contact 1"), "{:?}", whats(&s));
        assert!(
            s.unheld_orders().iter().any(|o| o.contains("contact 1")),
            "the decision was not on the board for somebody to take"
        );

        assert!(s.watch_tower().is_empty(), "the same cause was told twice");
        assert_eq!(whats(&s).len(), 1, "the same cause was put up twice");
    }

    /// **The klaxon says how to take the decision on**, with the words the
    /// board and `claim` know it by, so that hearing it is enough to act on.
    #[test]
    fn the_klaxon_names_the_decision_and_how_to_claim_it() {
        let mut s = with_contact();
        let told = s.watch_tower();
        let what = whats(&s)[0].clone();
        assert!(
            told[0].contains(&format!("claim \"{what}\"")),
            "the klaxon did not say how to take the decision on: {told:?}"
        );
        assert!(
            s.claimable_at("tower-redoubt/bridge").contains(&what),
            "what the klaxon names is not something `claim` offers"
        );
    }

    /// A tower with nothing pressing, and a world with no tower, put up nothing.
    #[test]
    fn nothing_pressing_puts_up_nothing() {
        let mut quiet = sim();
        assert!(quiet.watch_tower().is_empty());
        assert!(whats(&quiet).is_empty());

        let mut none = Sim::new();
        assert!(none.watch_tower().is_empty());
        assert!(none.tower_decisions().is_empty());
    }

    /// **A cause that has passed takes its decision with it, held or not.**
    #[test]
    fn a_cause_that_passes_comes_off_the_board_even_held() {
        let mut s = with_contact();
        s.watch_tower();
        s.ledger.take_order(&whats(&s)[0], "c1").unwrap();

        s.tower.as_mut().unwrap().fold_to(Coord::new(9, 9));
        s.watch_tower();

        assert!(whats(&s).is_empty());
        assert!(
            s.orders_held_by("c1").is_empty(),
            "a dead decision stayed held"
        );
    }

    /// **A decision that has been made is not put up again while its cause
    /// stands**, which is what would otherwise re-prompt the whole cast every
    /// beat.
    #[test]
    fn a_settled_decision_is_not_put_up_again_while_its_cause_stands() {
        let mut s = with_contact();
        s.watch_tower();
        let what = whats(&s)[0].clone();
        s.ledger.take_order(&what, "c1").unwrap();
        assert_eq!(s.settle_tower_decisions("c1"), 1);
        assert!(whats(&s).is_empty());

        assert!(
            s.watch_tower().is_empty(),
            "a settled cause was announced again"
        );
        assert!(whats(&s).is_empty(), "a settled cause was put up again");
    }

    /// And a cause that passes and comes back is a new one.
    #[test]
    fn a_cause_that_returns_is_a_new_decision() {
        let mut s = sim();
        s.tower.as_mut().unwrap().draw(Resource::Energy, 2_600);
        assert_eq!(s.watch_tower().len(), 1);
        let what = whats(&s)[0].clone();
        s.ledger.take_order(&what, "c1").unwrap();
        s.settle_tower_decisions("c1");

        s.tower.as_mut().unwrap().put(Resource::Energy, 3_000);
        s.watch_tower();
        assert!(whats(&s).is_empty());

        s.tower.as_mut().unwrap().draw(Resource::Energy, 3_000);
        assert_eq!(
            s.watch_tower().len(),
            1,
            "the returning cause was not raised"
        );
        assert_eq!(whats(&s), vec![what]);
    }

    /// **Somebody holding a decision is the tower's one voice**: anybody else is
    /// told whose it is and by what decision.
    #[test]
    fn once_a_decision_is_held_the_tower_is_the_holders() {
        let mut s = with_contact();
        s.watch_tower();
        let what = whats(&s)[0].clone();
        s.ledger.take_order(&what, "c1").unwrap();

        assert_eq!(s.tower_held_against("c1"), None, "the holder was blocked");
        assert_eq!(
            s.tower_held_against("c2"),
            Some((what, "c1".to_string())),
            "a bystander was let command a held tower"
        );
    }

    /// **Nobody is in the way until somebody has taken it.**
    #[test]
    fn an_unheld_decision_blocks_nobody() {
        let mut s = with_contact();
        s.watch_tower();
        assert_eq!(s.tower_held_against("c1"), None);
        assert_eq!(s.tower_held_against("c2"), None);
    }

    /// A body holding one decision is not locked out by another's holder: the
    /// tower's acts are not partitioned by cause.
    #[test]
    fn a_holder_of_any_decision_may_command() {
        let mut s = with_contact();
        s.tower.as_mut().unwrap().draw(Resource::Energy, 2_600);
        assert_eq!(s.watch_tower().len(), 2);
        let both = whats(&s);
        s.ledger.take_order(&both[0], "c1").unwrap();
        s.ledger.take_order(&both[1], "c2").unwrap();

        assert_eq!(s.tower_held_against("c1"), None);
        assert_eq!(s.tower_held_against("c2"), None);
        assert!(s.tower_held_against("c3").is_some());
    }

    /// Settling closes the decisions a body holds and leaves everybody else's.
    #[test]
    fn settling_closes_only_what_the_body_holds() {
        let mut s = with_contact();
        s.tower.as_mut().unwrap().draw(Resource::Energy, 2_600);
        s.watch_tower();
        let both = whats(&s);
        s.ledger.take_order(&both[0], "c1").unwrap();
        s.ledger.take_order(&both[1], "c2").unwrap();

        assert_eq!(s.settle_tower_decisions("c1"), 1);

        let left = s.tower_decisions();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].held_by.as_deref(), Some("c2"));
        assert_eq!(s.settle_tower_decisions("c3"), 0);
    }
}
