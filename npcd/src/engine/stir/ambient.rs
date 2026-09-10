//! The building being a building, and nothing more than that.
//!
//! # Why the vault needed this
//!
//! Every other fixture here is a *mechanism*: it has a state, it goes wrong, it
//! publishes a condition that something else reads. That is what makes the
//! incidents work, and it is also what made the first version of this module
//! implausible — a room where every single thing that happens is a fault, a
//! transition or a consequence reads as a base in the last hour before it falls
//! over. Real buildings are overwhelmingly *quiet*, and the few things that
//! matter matter because of everything around them that does not.
//!
//! So this fixture is deliberately the opposite of the rest. It holds no state
//! worth the name, it never publishes a [`Cond`], it never reaches
//! [`Salience::NORMAL`], and nothing anywhere reacts to it. Its entire job is to
//! be the ninety per cent, so that a breaker going is one event in twenty rather
//! than one in three.
//!
//! # It still reads the room
//!
//! A flat list would be padding. These are grouped by the condition they belong
//! to, so a cold building notices cold things and a working one notices the
//! machines — the observation is idle, but it is idle *about the building it is
//! actually in*. The conditional pools are consulted first and the general pool
//! carries the rest.
//!
//! # And it deals from a bag
//!
//! Same reasoning as [`crate::engine::stir::broadcast`]: drawing at random from
//! a hundred lines still repeats within a dozen draws, which is exactly the
//! failure this module exists to fix. Dealing from a shuffled bag means every
//! line in the general pool is used before any is used twice.

use crate::engine::event::Salience;
use crate::engine::stir::{Cond, Fixture, Rng, Stirring, Watch};

/// The building at rest. Nothing here is a fault, a warning, or a cause.
const IDLE: &[&str] = &[
    // Machines and electronics, at rest.
    "A cabinet fan somewhere under the bench changes speed and settles.",
    "A terminal on the side bench scrolls two lines of log and stops.",
    "The screen on the wall panel wakes at nothing and dims itself again.",
    "A hard drive in the rack chatters briefly and goes quiet.",
    "An indicator on a junction box blinks twice, amber, and returns to green.",
    "A power brick under the desk gives off a faint whine that comes and goes.",
    "The keyboard on the bench terminal has one key lit that should not be.",
    "A cable reel on the wall unwinds a quarter turn under its own weight.",
    "The printer in the corner draws in a sheet, thinks about it, and gives it back.",
    "A monitor in the next bay flickers through its self-test and settles.",
    "Somewhere behind a panel a capacitor discharges with a small tick.",
    "A handheld unit left on charge beeps once to say it is full.",
    "The bench light has a flicker in it at the very edge of noticing.",
    "A cooling fin somewhere clicks as it gives up the last of its heat.",
    "An idle console runs its screensaver, which is a slowly rotating logo.",
    "A speaker in the ceiling pops faintly as something upstream switches.",
    "Two indicator lights on the far wall come into step and drift apart again.",
    "A tape drive in the corner spools, stops, and spools the other way.",
    // Doors, hatches, floors, walls.
    "A door two rooms away opens and closes without anybody coming through.",
    "The floor plate by the doorway rocks a fraction when weight comes off it.",
    "A hatch cover settles against its frame with a soft metallic knock.",
    "The door seal sighs as the pressure on the two sides of it evens out.",
    "A wall panel that was never fastened properly taps once against its stud.",
    "The threshold strip at the door has worked loose at one end.",
    "A handrail somewhere takes a knock and rings for a moment.",
    "The floor grating gives its usual note under a shift of weight.",
    "A door catch releases on its own and the door stands an inch open.",
    "Dust comes off the top of the door frame in a thin line.",
    "The rubber on the doorway seal has taken a set and no longer sits flat.",
    "A stair tread on the other side of the wall gives under somebody's foot.",
    // Pipes, ducts, and the fabric.
    "A pipe behind the wall carries a knock along its length and away.",
    "Something in the ductwork overhead shifts and is quiet again.",
    "The lagging on an overhead run creaks as it cools.",
    "A bracket somewhere in the ceiling void takes up a fraction of slack.",
    "Water moves in a pipe overhead, briefly, going somewhere else.",
    "A duct joint ticks twice as the air behind it changes pressure.",
    "The extract grille above the door draws a scrap of dust up against itself.",
    "A run of conduit along the wall gives a single hollow note.",
    "Somewhere above the ceiling a valve closes and the flow stops.",
    "The trunking along the skirting has come away from the wall at one clip.",
    // Objects, furniture, paper.
    "A stack of printed pages slides an inch across the bench and stops.",
    "Somebody's mug on the far bench has gone cold and left a ring.",
    "A pen rolls to the edge of the table and does not go over.",
    "The chair at the terminal turns a few degrees on its own.",
    "A drawer that was not closed properly slides open an inch.",
    "A clipboard slips off its hook and swings on the lanyard.",
    "The corner of a wall notice has come unstuck and curls forward.",
    "A tool left on the bench edge shifts and settles into the groove.",
    "A cardboard box on the floor gives up a fold and sags.",
    "Loose change or washers rattle in a tray as something passes.",
    "A folded chart on the side table opens itself out one crease.",
    "The label on a bin has faded to the point of saying nothing.",
    "A roll of tape unsticks a turn of itself with a small tearing sound.",
    "Somebody's jacket on the back of the chair slides down another inch.",
    "A crate lid rests on its box without being latched to it.",
    "The bench surface has a scratch across it that catches the light.",
    // Other people, other levels, elsewhere in the building.
    "Somebody laughs on another level and the sound arrives without the words.",
    "A conversation goes past in the corridor outside and does not stop.",
    "Footsteps cross the ceiling overhead, unhurried, from one side to the other.",
    "A trolley goes along a corridor somewhere with one bad wheel.",
    "Something heavy is set down on the level above and the floor registers it.",
    "A door slams two levels down and the sound comes up the stairwell.",
    "The lift moves in its shaft without stopping at this level.",
    "Somebody whistles four notes in the corridor and stops.",
    "A voice on another level calls a name and gets no answer anybody can hear.",
    "Machinery starts up somewhere far off and settles into a steady note.",
    "The stairwell carries a scrape of something being dragged up it.",
    "Two sets of footsteps go past the door in step and then out of it.",
    "A tool is dropped on a hard floor somewhere below and bounces twice.",
    "Somebody swears quietly on the other side of the wall.",
    "The lift doors open on an empty landing and close again.",
    // Small mechanical events.
    "A spring-loaded catch somewhere lets go with a flat snap.",
    "Something small and metal drops onto the floor and rolls under a bench.",
    "A castor on the trolley turns itself to face the other way.",
    "A latch rattles in its keeper as the pressure shifts.",
    "The bench vice handle swings down against the jaw with a clank.",
    "A magnetic strip on the wall gives up one of its tools.",
    "A hinge somewhere in the room complains and then does not.",
    "The waste chute flap opens a crack and falls shut.",
    "A screw that has been backing out for weeks finally clears its thread.",
    // Air, smell, temperature.
    "The air in the room carries a faint smell of hot dust from somewhere.",
    "A draught crosses the floor at ankle height and is gone.",
    "The air tastes faintly of solder for a few seconds and then does not.",
    "Something in the room smells briefly of machine oil.",
    "The temperature in the room has gone up a degree without any announcement.",
    "A warm current comes off the back of the rack and rises past the light.",
    "The air near the door is noticeably fresher than the air by the wall.",
    "A smell of hot plastic comes and goes with no source anybody could point at.",
    // Light and the look of things.
    "The light off the wall panel puts a moving shape on the ceiling.",
    "A reflection crosses the dark half of a screen and is gone.",
    "Dust turns slowly in the light coming off the ceiling fitting.",
    "The shadow under the bench moves as something overhead changes brightness.",
    "A bright spot on the floor shifts an inch as a fitting settles.",
    "The polished edge of the bench throws a line of light along the wall.",
    "Everything in the room takes on a slightly greener cast for a moment.",
    "A screen behind the bench goes to black and shows the room reflected in it.",
];

/// Cold enough to be worth a remark.
const COLD: &[&str] = &[
    "Breath shows faintly in the air near the door.",
    "The metal of the bench is cold enough to be unpleasant to lean on.",
    "A skin of condensation has come up on the outside of the water bottle.",
    "The chill off the wall panel is noticeable from a foot away.",
    "Somebody's abandoned mug has gone stone cold and stopped steaming.",
    "The seal around the door has stiffened up in the cold.",
    "Fingers of cold air come down from the ceiling vent and pool on the floor.",
    "The floor plate is cold enough through a boot sole to feel.",
    "Frost has formed in the corner of the window and gone no further.",
];

/// Dark enough to change what the room looks like.
const DARK: &[&str] = &[
    "The indicator lights on the rack are suddenly the brightest thing in the room.",
    "Shapes at the far end of the room have stopped being distinct from each other.",
    "A screen's glow is doing all the work of lighting the bench.",
    "The doorway is a lighter rectangle in a wall that has otherwise gone flat.",
    "Everything in the room has lost its colour and kept its outline.",
    "A reflection on the far wall is the only way of telling where the wall is.",
    "The gloom at the back of the room has swallowed the shelving entirely.",
];

/// Damp enough to be in the air.
const DAMP: &[&str] = &[
    "The air has the flat, mineral smell of standing water in it.",
    "Paper left on the bench has gone soft at the corners.",
    "A film of moisture has come up on the cold surfaces in the room.",
    "The floor by the wall is tacky underfoot in a way it should not be.",
    "Everything metal in the room has taken on a dull, beaded look.",
    "The smell in this corner is the smell of a cellar.",
];

/// The machines are working.
const WORKING: &[&str] = &[
    "The hum from the racks has become the kind of noise nobody hears any more.",
    "Warm air off the back of the machines is moving the dust in the light.",
    "A note in the machine noise wanders slightly sharp and comes back.",
    "The rack's activity lights have settled into a pattern that repeats.",
    "The floor carries the vibration from the machines up through the bench legs.",
    "Somewhere in the racks a fan is running faster than its neighbours.",
    "The noise in the room has a second, deeper layer under the first one.",
];

/// Quiet enough that small things carry.
const QUIET: &[&str] = &[
    "The room is quiet enough that the hum of the light is audible.",
    "A clock or a counter somewhere is ticking, and has been for some time.",
    "The quiet in the room has got deep enough to have a texture.",
    "Every small sound in the room is arriving with its echo attached.",
    "The loudest thing in the room is the air moving past the door frame.",
];

/// How many lines back the fixture refuses to repeat itself.
///
/// The bag covers [`IDLE`], but the conditional pools are small and are drawn
/// from directly — nine cold lines in a cold building put frost on the window
/// twice inside five minutes, which is precisely the tell this whole module
/// exists to remove. This is the guard that covers every pool at once.
const RECENT: usize = 24;

pub struct Ambient {
    rng: Rng,
    /// Indices of [`IDLE`] not yet used this time round.
    bag: Vec<usize>,
    /// The last [`RECENT`] lines, newest last.
    recent: Vec<&'static str>,
}

impl Ambient {
    pub fn new(seed: u64) -> Ambient {
        Ambient {
            rng: Rng::new(seed),
            bag: Vec::new(),
            recent: Vec::new(),
        }
    }

    fn said_lately(&self, line: &str) -> bool {
        self.recent.contains(&line)
    }

    fn remember(&mut self, line: &'static str) {
        if self.recent.len() >= RECENT {
            self.recent.remove(0);
        }
        self.recent.push(line);
    }

    /// How many benign lines it can draw on. The count is the point of the
    /// fixture, so it is worth being able to ask.
    pub fn depth() -> usize {
        IDLE.len() + COLD.len() + DARK.len() + DAMP.len() + WORKING.len() + QUIET.len()
    }

    fn deal(&mut self) -> &'static str {
        if self.bag.is_empty() {
            self.bag = (0..IDLE.len()).collect();
            for i in (1..self.bag.len()).rev() {
                self.bag.swap(i, self.rng.below(i + 1));
            }
        }
        self.bag.pop().map(|at| IDLE[at]).unwrap_or(IDLE[0])
    }
}

impl Fixture for Ambient {
    fn id(&self) -> &'static str {
        "ambient"
    }

    fn consider(&mut self, w: &Watch) -> Option<Stirring> {
        // **The one fixture with no clock of its own.** Every other part of the
        // building is waiting for something; a building at rest is not waiting,
        // it simply is at rest, and the only question is whether anybody looked.
        // So this always has an answer, and `stir::AT_REST` — which is the only
        // thing that can see the whole building's rate — decides how often that
        // answer is the one taken.

        // What is true of the building gets a look in first, so an idle remark
        // is still an idle remark about *this* room.
        let mut pools: Vec<&[&str]> = Vec::new();
        if w.is(Cond::Cold) {
            pools.push(COLD);
        }
        if w.is(Cond::Dark) {
            pools.push(DARK);
        }
        if w.is(Cond::Damp) {
            pools.push(DAMP);
        }
        if w.is(Cond::Working) {
            pools.push(WORKING);
        }
        if w.is(Cond::Quiet) {
            pools.push(QUIET);
        }

        // Roughly one remark in three is about the condition when there is one
        // to be about. More than that and a cold room talks about nothing but
        // the cold.
        //
        // A few attempts, then take what comes: a building that has been cold
        // for hours will run out of unsaid cold lines, and saying a repeat is
        // better than looping here or saying nothing.
        let mut line = IDLE[0];
        for _ in 0..4 {
            line = match pools.is_empty() || !self.rng.one_in(3) {
                true => self.deal(),
                false => {
                    let pool = self.rng.pick(&pools).copied().unwrap_or(IDLE);
                    self.rng.pick(pool).copied().unwrap_or(IDLE[0])
                }
            };
            if !self.said_lately(line) {
                break;
            }
        }
        self.remember(line);

        // Never above idle, never tagged. Nothing in the building reacts to any
        // of this, and that is deliberate — it is the floor everything else is
        // heard against.
        Some(Stirring::new("ambient", line, Salience::IDLE))
    }
}
