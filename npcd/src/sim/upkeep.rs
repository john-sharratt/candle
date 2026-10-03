//! How the tower's state moves with time, and what that asks of somebody.
//!
//! # A choice with no cause is a loop
//!
//! The tower used to be a ledger that only changed when a character changed it.
//! Its stockpile never drained, nothing ever approached it, and so *relocate,
//! raise shields, drill down, lift the siege* were five options with no reason
//! to pick any of them — and a cast handed five unmotivated options does what
//! people do with those: it asks the others which, and is asked back.
//!
//! This gives the tower a pulse. Keeping it standing costs energy every minute,
//! more with the shields up or a siege open and less dug in, so the stockpile
//! really falls and "how long can we hold?" has an answer. And now and then
//! something comes at it, so that shields, the ground and a fold are answers to
//! a question the world asked rather than levers on a panel.
//!
//! # What the world asks, [`Tower::pressing`] says
//!
//! Each standing cause is a decision to be made, named by the sentence that
//! would go on the muster board. The decisions are the world's to put up and to
//! take down — [`crate::sim::decisions`] does that — and are a function of the
//! tower's state, so a cause that has passed (the contact folded away from, the
//! reserve restored) takes its decision with it.
//!
//! # Time is a duration the caller states
//!
//! [`Tower::advance`] takes the time that has passed rather than reading a
//! clock, for the reason [`crate::engine::rooms::Rooms::stir_at`] does: a test
//! that had to wait out a contact's approach in real minutes would not be
//! written. Sub-unit remainders are carried, because the world calls this twice
//! a second and a rate of twenty a minute would otherwise round to nothing.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::sim::field::Resource;
use crate::sim::tower::{Coord, Posture, Tower, FOLD_COST};

/// What standing costs, per minute, before anything is switched on.
pub const BASE_DRAW: u64 = 20;
/// What the shields add, per minute.
pub const SHIELD_DRAW: u64 = 30;
/// What an open siege adds, per minute.
pub const SIEGE_DRAW: u64 = 25;
/// What each queued batch adds, per minute.
pub const BATCH_DRAW: u64 = 6;

/// How long the tower is left alone between one contact and the next sighting.
pub const CONTACT_EVERY: Duration = Duration::from_secs(20 * 60);
/// How long a contact takes to close from first sighting to the tower.
pub const CONTACT_CLOSES_IN: Duration = Duration::from_secs(15 * 60);
/// What the shields spend taking a contact.
pub const ABSORB_COST: u64 = 120;
/// What an unshielded contact takes in energy.
pub const HIT_ENERGY: u64 = 100;
/// The reserves below which the tower is asked to live on less, lowest first.
pub const LOW_ENERGY: [u64; 2] = [500, 1_500];
/// What the tower keeps back for the drill. Standing, a held contact and the
/// shields spend only what lies above it, so a tower that has run down can
/// still dig in and recover rather than sit dead on an empty stockpile.
pub const RESERVE: u64 = 200;
/// What a buried tower draws from the ground, per minute.
pub const TAP_YIELD: u64 = 30;

const MS_PER_MINUTE: u64 = 60_000;

/// What comes at a tower, by sighting number.
const KINDS: [&str; 5] = [
    "a drone swarm",
    "a crawler column",
    "a raider convoy",
    "a siege walker",
    "a scavenger pack",
];

/// Something closing on the tower.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contact {
    /// Which sighting it was. Never reused, so the decision it raises is its own.
    pub id: u32,
    /// What it is, as a character would say it.
    pub kind: String,
    /// How long until it reaches the tower, in milliseconds.
    pub arrives_in_ms: u64,
}

impl Contact {
    fn sighted(id: u32) -> Contact {
        Contact {
            id,
            kind: KINDS[(id as usize - 1) % KINDS.len()].to_string(),
            arrives_in_ms: CONTACT_CLOSES_IN.as_millis() as u64,
        }
    }

    /// What the decision it raises calls it.
    pub fn label(&self) -> String {
        format!("contact {} ({})", self.id, self.kind)
    }

    /// Whole minutes until it arrives, rounded up so it never reads zero while
    /// it is still coming.
    pub fn minutes_out(&self) -> u64 {
        self.arrives_in_ms.div_ceil(MS_PER_MINUTE)
    }
}

/// What the tower's clock has accumulated and not yet turned into anything.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Clock {
    /// Energy-milliseconds owed and not yet drawn: the part of a unit of energy
    /// that has been spent but is not yet a whole one.
    carry: u64,
    /// Energy-milliseconds drawn from the ground and not yet a whole unit.
    tap_carry: u64,
    /// How long the tower has gone without a contact.
    quiet_ms: u64,
    /// How many contacts there have been.
    sightings: u32,
}

/// One thing the world is asking of whoever has the tower.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pressing {
    /// The decision, as the sentence that goes on the muster board. Also what
    /// identifies the cause, so it is the same sentence for as long as it stands.
    pub decision: String,
    /// What the klaxon says when it is first put up.
    pub alarm: String,
}

impl Tower {
    /// What keeping the tower up costs now, in energy per minute.
    pub fn draw_per_minute(&self) -> u64 {
        let mut rate = BASE_DRAW;
        if self.shields {
            rate += SHIELD_DRAW;
        }
        if self.besieging.is_some() {
            rate += SIEGE_DRAW;
        }
        rate += BATCH_DRAW * self.queued().len() as u64;
        if self.posture == Posture::DugIn {
            rate = rate.div_ceil(2);
        }
        rate
    }

    /// What the ground gives the tower, per minute. Only a buried tower is
    /// tapped into it.
    pub fn tap_per_minute(&self) -> u64 {
        match self.posture {
            Posture::DugIn => TAP_YIELD,
            _ => 0,
        }
    }

    /// What the tower can spend before it is down to its reserve.
    pub fn spare_energy(&self) -> u64 {
        self.stock_of(Resource::Energy).saturating_sub(RESERVE)
    }

    /// How long the spare energy lasts at the present draw, less what the
    /// ground gives back. `None` when the ground gives back at least what is
    /// drawn, because then it does not run out.
    pub fn minutes_of_energy(&self) -> Option<u64> {
        let net = self
            .draw_per_minute()
            .checked_sub(self.tap_per_minute())
            .filter(|net| *net > 0)?;
        Some(self.spare_energy() / net)
    }

    /// Let `dt` pass. Returns what the crew would be told: shields failing,
    /// and how a contact that arrived was met.
    pub fn advance(&mut self, dt: Duration) -> Vec<String> {
        let ms = u64::try_from(dt.as_millis()).unwrap_or(u64::MAX);
        let mut said = Vec::new();

        self.clock.tap_carry = self
            .clock
            .tap_carry
            .saturating_add(self.tap_per_minute().saturating_mul(ms));
        self.put(Resource::Energy, self.clock.tap_carry / MS_PER_MINUTE);
        self.clock.tap_carry %= MS_PER_MINUTE;

        self.clock.carry = self
            .clock
            .carry
            .saturating_add(self.draw_per_minute().saturating_mul(ms));
        let owed = self.clock.carry / MS_PER_MINUTE;
        self.clock.carry %= MS_PER_MINUTE;
        let paid = owed.min(self.spare_energy());
        self.draw(Resource::Energy, paid);
        if self.spare_energy() == 0 && self.shields {
            self.shields = false;
            said.push("The shields fail for want of power.".to_string());
        }

        said.extend(self.advance_contact(ms));
        said
    }

    fn advance_contact(&mut self, ms: u64) -> Option<String> {
        let Some(mut contact) = self.contact.take() else {
            self.clock.quiet_ms = self.clock.quiet_ms.saturating_add(ms);
            if self.clock.quiet_ms >= CONTACT_EVERY.as_millis() as u64 {
                self.clock.quiet_ms = 0;
                self.clock.sightings += 1;
                self.contact = Some(Contact::sighted(self.clock.sightings));
            }
            return None;
        };
        if ms < contact.arrives_in_ms {
            contact.arrives_in_ms -= ms;
            self.contact = Some(contact);
            return None;
        }
        self.clock.quiet_ms = 0;
        Some(self.meet(&contact))
    }

    /// A contact reaches the tower, and what it finds decides what it costs.
    fn meet(&mut self, contact: &Contact) -> String {
        let label = contact.label();
        if self.posture == Posture::DugIn {
            return format!("{label} passes over the buried tower and goes on its way.");
        }
        if self.shields {
            let cost = ABSORB_COST.min(self.spare_energy());
            self.draw(Resource::Energy, cost);
            return format!(
                "The shields take {label}. Holding them against it cost {cost} energy."
            );
        }
        let energy = HIT_ENERGY.min(self.spare_energy());
        let metal = self.stock_of(Resource::Metal) / 10;
        self.draw(Resource::Energy, energy);
        self.draw(Resource::Metal, metal);
        format!(
            "{label} reaches the tower with the shields down. It cost {energy} energy and {metal} metal."
        )
    }

    /// Fold to somewhere else. A tower that has folded away is out of reach of
    /// whatever was closing on it.
    pub fn fold_to(&mut self, at: Coord) {
        self.draw(Resource::Energy, FOLD_COST);
        self.at = at;
        self.contact = None;
        self.clock.quiet_ms = 0;
    }

    /// What the world is asking of whoever has the tower right now.
    pub fn pressing(&self) -> Vec<Pressing> {
        let mut out = Vec::new();
        if let Some(c) = &self.contact {
            out.push(Pressing {
                decision: format!("Decide how the tower answers {}", c.label()),
                alarm: format!(
                    "Contact {} ({}) is closing on the tower, {} minutes out.",
                    c.id,
                    c.kind,
                    c.minutes_out()
                ),
            });
        }
        let energy = self.stock_of(Resource::Energy);
        if let Some(floor) = LOW_ENERGY.iter().find(|floor| energy < **floor) {
            let outlook = match self.minutes_of_energy() {
                Some(minutes) => format!("At the present draw it lasts {minutes} minutes."),
                None => "The ground is giving back more than the tower draws.".to_string(),
            };
            out.push(Pressing {
                decision: format!("Decide how the tower is to live on under {floor} energy"),
                alarm: format!("The energy reserve is under {floor}. {outlook}"),
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tower() -> Tower {
        let mut t = Tower::new("the Redoubt", Coord::new(0, 0));
        t.put(Resource::Energy, 4_000);
        t.put(Resource::Metal, 900);
        t
    }

    fn minutes(n: u64) -> Duration {
        Duration::from_secs(60 * n)
    }

    /// **Standing costs energy, and it comes off the stockpile.** The reason a
    /// character who reads the stockpile twice reads two different numbers.
    #[test]
    fn standing_costs_energy_every_minute() {
        let mut t = tower();
        t.advance(minutes(10));
        assert_eq!(t.stock_of(Resource::Energy), 4_000 - 10 * BASE_DRAW);
    }

    /// **The world calls this twice a second.** A rate of twenty a minute is a
    /// sixth of a unit a beat, which would round to nothing every time and leave
    /// the stockpile standing still for ever.
    #[test]
    fn a_rate_smaller_than_a_unit_a_beat_still_drains() {
        let mut t = tower();
        for _ in 0..(10 * 60 * 2) {
            t.advance(Duration::from_millis(500));
        }
        assert_eq!(t.stock_of(Resource::Energy), 4_000 - 10 * BASE_DRAW);
    }

    /// Shields and a siege are paid for as long as they are up, a queued batch
    /// is a running fabricator, and being dug in halves the lot.
    #[test]
    fn what_is_switched_on_raises_the_draw_and_the_ground_lowers_it() {
        let mut t = tower();
        assert_eq!(t.draw_per_minute(), BASE_DRAW);
        t.shields = true;
        assert_eq!(t.draw_per_minute(), BASE_DRAW + SHIELD_DRAW);
        t.besieging = Some("Ash Keep".into());
        assert_eq!(t.draw_per_minute(), BASE_DRAW + SHIELD_DRAW + SIEGE_DRAW);
        t.posture = Posture::DugIn;
        assert_eq!(
            t.draw_per_minute(),
            (BASE_DRAW + SHIELD_DRAW + SIEGE_DRAW).div_ceil(2)
        );
    }

    /// A reserve that is asked about says how long what can be spent of it
    /// lasts at the draw it has.
    #[test]
    fn the_reserve_says_how_long_it_lasts() {
        let mut t = tower();
        assert_eq!(t.minutes_of_energy(), Some((4_000 - RESERVE) / BASE_DRAW));
        t.shields = true;
        assert_eq!(
            t.minutes_of_energy(),
            Some((4_000 - RESERVE) / (BASE_DRAW + SHIELD_DRAW))
        );
    }

    /// **Running down stops at the reserve, not at nothing**, and the shields go
    /// with the power that was spendable.
    #[test]
    fn the_shields_fail_when_the_spare_power_does() {
        let mut t = tower();
        t.shields = true;
        t.draw(Resource::Energy, 3_700);
        let said = t.advance(minutes(5));
        assert_eq!(t.stock_of(Resource::Energy), RESERVE);
        assert!(!t.shields, "the shields stayed up on no power");
        assert!(said.iter().any(|s| s.contains("shields fail")), "{said:?}");
    }

    /// **A tower that has run down is still there to dig in with**: however long
    /// it stands, the drill's reserve is not spent on standing.
    #[test]
    fn standing_never_spends_the_reserve() {
        let mut t = tower();
        t.draw(Resource::Energy, 3_700);
        t.advance(minutes(600));
        assert_eq!(t.stock_of(Resource::Energy), RESERVE);
        assert_eq!(t.spare_energy(), 0);
    }

    /// **A contact does not take the reserve either.**
    #[test]
    fn a_contact_does_not_take_the_reserve() {
        let mut t = with_contact_arriving(tower());
        t.draw(Resource::Energy, 3_390);
        t.advance(CONTACT_CLOSES_IN);
        assert_eq!(t.stock_of(Resource::Energy), RESERVE);
    }

    /// **A buried tower is paid by the ground**, and being paid more than it
    /// draws is how a tower that has run low comes back.
    #[test]
    fn a_buried_tower_recovers_energy() {
        let mut t = tower();
        t.draw(Resource::Energy, 3_500);
        t.posture = Posture::DugIn;
        t.advance(minutes(10));
        let net = TAP_YIELD - BASE_DRAW.div_ceil(2);
        assert_eq!(t.stock_of(Resource::Energy), 500 + 10 * net);
    }

    /// A standing tower is tapped into nothing.
    #[test]
    fn a_standing_tower_gets_nothing_from_the_ground() {
        let mut t = tower();
        assert_eq!(t.tap_per_minute(), 0);
        t.posture = Posture::DugIn;
        assert_eq!(t.tap_per_minute(), TAP_YIELD);
    }

    /// **The ground's yield arrives a beat at a time** for the reason the draw
    /// does: thirty a minute is a quarter of a unit every half second.
    #[test]
    fn a_yield_smaller_than_a_unit_a_beat_still_arrives() {
        let mut t = tower();
        t.draw(Resource::Energy, 3_500);
        t.posture = Posture::DugIn;
        for _ in 0..(10 * 60 * 2) {
            t.advance(Duration::from_millis(500));
        }
        assert_eq!(
            t.stock_of(Resource::Energy),
            500 + 10 * (TAP_YIELD - BASE_DRAW.div_ceil(2))
        );
    }

    /// **A tower the ground out-gives does not run out**, and says so rather
    /// than naming a number of minutes it will never reach.
    #[test]
    fn a_tower_the_ground_out_gives_does_not_run_out() {
        let mut t = tower();
        t.posture = Posture::DugIn;
        assert_eq!(t.minutes_of_energy(), None);
        t.draw(Resource::Energy, 3_000);
        let p = t.pressing();
        assert_eq!(p.len(), 1);
        assert!(p[0].alarm.contains("giving back more"), "{:?}", p[0]);
    }

    /// **Nothing comes at the tower until it has been left alone a while**, and
    /// then something does, with time to answer it.
    #[test]
    fn a_contact_is_sighted_after_a_quiet_spell_and_closes_in_time() {
        let mut t = tower();
        t.advance(CONTACT_EVERY - Duration::from_secs(1));
        assert!(t.contact.is_none(), "something came before its time");

        t.advance(Duration::from_secs(1));
        let c = t.contact.clone().expect("a contact was sighted");
        assert_eq!(c.id, 1);
        assert_eq!(c.minutes_out(), CONTACT_CLOSES_IN.as_secs() / 60);

        t.advance(minutes(5));
        assert_eq!(
            t.contact.as_ref().unwrap().minutes_out(),
            CONTACT_CLOSES_IN.as_secs() / 60 - 5
        );
    }

    fn with_contact_arriving(mut t: Tower) -> Tower {
        t.advance(CONTACT_EVERY);
        assert!(t.contact.is_some());
        t
    }

    /// **Shielded, it is met and it costs energy to hold.**
    #[test]
    fn a_shielded_tower_takes_a_contact_for_a_price() {
        let mut t = with_contact_arriving(tower());
        t.shields = true;
        let before = t.stock_of(Resource::Energy);
        let said = t.advance(CONTACT_CLOSES_IN);
        assert!(
            t.contact.is_none(),
            "the contact stayed once it had arrived"
        );
        let draw = t.draw_per_minute() * CONTACT_CLOSES_IN.as_secs() / 60;
        assert_eq!(t.stock_of(Resource::Energy), before - draw - ABSORB_COST);
        assert!(said.iter().any(|s| s.contains("shields take")), "{said:?}");
        assert_eq!(t.stock_of(Resource::Metal), 900, "the shields let metal go");
    }

    /// **Dug in, it passes overhead and costs nothing.**
    #[test]
    fn a_buried_tower_is_passed_over() {
        let mut t = with_contact_arriving(tower());
        t.posture = Posture::DugIn;
        t.depth = 30;
        let said = t.advance(CONTACT_CLOSES_IN);
        assert!(t.contact.is_none());
        assert_eq!(t.stock_of(Resource::Metal), 900);
        assert!(said.iter().any(|s| s.contains("passes over")), "{said:?}");
    }

    /// **Standing in the open with the shields down, it is hit**, and what it
    /// loses is metal as well as energy.
    #[test]
    fn an_unshielded_tower_in_the_open_is_hit() {
        let mut t = with_contact_arriving(tower());
        let before = t.stock_of(Resource::Energy);
        let said = t.advance(CONTACT_CLOSES_IN);
        let draw = t.draw_per_minute() * CONTACT_CLOSES_IN.as_secs() / 60;
        assert_eq!(t.stock_of(Resource::Energy), before - draw - HIT_ENERGY);
        assert_eq!(t.stock_of(Resource::Metal), 810);
        assert!(said.iter().any(|s| s.contains("shields down")), "{said:?}");
    }

    /// The next contact is a different one, a quiet spell later.
    #[test]
    fn contacts_are_numbered_and_come_one_at_a_time() {
        let mut t = with_contact_arriving(tower());
        assert_eq!(t.contact.as_ref().unwrap().id, 1);
        t.advance(CONTACT_CLOSES_IN);
        assert!(t.contact.is_none());
        t.advance(CONTACT_EVERY);
        assert_eq!(t.contact.as_ref().unwrap().id, 2);
        assert_ne!(
            t.contact.as_ref().unwrap().kind,
            Contact::sighted(1).kind,
            "two sightings were the same thing"
        );
    }

    /// **A tower that has folded away is out of reach.**
    #[test]
    fn folding_leaves_a_contact_behind() {
        let mut t = with_contact_arriving(tower());
        let before = t.stock_of(Resource::Energy);
        t.fold_to(Coord::new(50, 50));
        assert!(t.contact.is_none());
        assert_eq!(t.at, Coord::new(50, 50));
        assert_eq!(t.stock_of(Resource::Energy), before - FOLD_COST);
    }

    /// **A contact is a decision, with the sentence that goes on the board and
    /// the line the klaxon says.**
    #[test]
    fn a_contact_presses_for_a_decision() {
        let t = with_contact_arriving(tower());
        let p = t.pressing();
        assert_eq!(p.len(), 1);
        assert!(p[0].decision.contains("contact 1"), "{:?}", p[0]);
        assert!(p[0].alarm.contains("15 minutes out"), "{:?}", p[0]);
    }

    /// A decision is a function of the state: the sentence does not move while
    /// the cause stands, and it is gone when the cause is.
    #[test]
    fn a_decision_stands_for_as_long_as_its_cause_does() {
        let mut t = with_contact_arriving(tower());
        let first = t.pressing()[0].decision.clone();
        t.advance(minutes(3));
        assert_eq!(t.pressing()[0].decision, first);
        t.fold_to(Coord::new(9, 9));
        assert!(t.pressing().is_empty());
    }

    /// **A thin reserve presses, and the lower it is the lower the line.**
    #[test]
    fn a_thin_reserve_presses_and_a_thinner_one_presses_harder() {
        let mut t = tower();
        assert!(t.pressing().is_empty(), "a full reserve pressed");
        t.draw(Resource::Energy, 2_600);
        let low = t.pressing();
        assert_eq!(low.len(), 1);
        assert!(low[0].decision.contains("1500"), "{:?}", low[0]);
        assert!(low[0].alarm.contains("minutes"), "{:?}", low[0]);

        t.draw(Resource::Energy, 1_000);
        let lower = t.pressing();
        assert!(lower[0].decision.contains("500"), "{:?}", lower[0]);
        assert_ne!(lower[0].decision, low[0].decision);
    }
}
