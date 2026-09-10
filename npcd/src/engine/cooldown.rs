//! How soon a body may do the same thing again.
//!
//! # Why a body needs one at all
//!
//! A character thinks as fast as the card will let it, and a decode is seconds.
//! Nothing else in the engine is a rate limit: the scheduler says *whether* it
//! is this character's turn, the grammar says *what* it may say, and neither has
//! any opinion about a body doing the same physical thing four times in twenty
//! seconds. A live cast spent an evening walking — thirty-eight of its last
//! fifty acts were `move_to`, three of them into the room it was already in —
//! and every one of those was individually a reasonable thing to decide.
//!
//! **The limit is on the body, not on the mind.** A person cannot cross a room,
//! turn round and cross it again in the time it takes to think about it, and a
//! world where they can is one where nobody ever finishes anything. So the act
//! is *absent* for a few seconds after it is taken, and the character spends
//! that turn on something else — which is the whole of the intended effect.
//!
//! # Absent, not refused
//!
//! Cooling acts are struck from the grammar, the same way an act with nothing to
//! choose from is, and for the same reason: a refusal is not a lesson. A live
//! cast read its own refusal back as the most recent thing in its window and
//! emitted the same act again. What a character cannot say, it cannot get stuck
//! saying.
//!
//! # Real time, and short
//!
//! Real time because it is a fact about a body and not about the narrative
//! clock — a world running at eight times speed does not let anybody throw
//! punches faster.
//!
//! Short because the point is to stop a *loop*, not to slow the world down.
//! Three seconds between blows is a fight; thirty would be two characters
//! standing still. And a cooldown is **per act, never per journey**: crossing
//! six rooms already costs six of the world's moments to walk, and charging for
//! the distance on top would mean the further somebody went the longer they
//! stood at the far end doing nothing.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::engine::act::Act;
use crate::engine::tools::SELF;

/// What one act costs the body that took it.
///
/// **Two different costs, and they are not interchangeable.** Both are named
/// here together because choosing one without looking at the other is how an
/// act ends up either free or frozen.
///
/// | | [`Cost::stall`] | [`Cost::cool`] |
/// |---|---|---|
/// | the question | **how long does doing it take?** | **how soon may it be done again?** |
/// | what it does | the character does not think again for this long | this act is absent from the grammar for this long |
/// | who it stops | *everything* the character might do | this one act |
/// | where it lands | `Scheduler::pause_for` → `Inbox::due_at` | `Within::cooling` → `specs_within` |
/// | woken early? | **yes — anything arriving wakes it at once** | no, it runs out on the clock |
/// | what it is for | the body is busy | stopping one cheap act becoming the whole loop |
///
/// **A stall is not a penalty, it is elapsed time.** Crossing a room and
/// putting a hand on somebody take a moment, and a character that decides
/// again the instant it has swung is not fighting, it is flickering. What
/// makes it safe to charge honestly is the last row: a stall is not a lockout.
/// Being hit, being spoken to, anything at all arriving wakes the character
/// immediately, so a stalled body still reacts — it simply does not
/// *self-start* a new decision while the last one is still happening.
///
/// A cooldown is the other question entirely: one door closed in a room full
/// of doors. The character keeps ticking on its ordinary heartbeat and picks
/// something else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cost {
    /// How long the act itself occupies the body.
    pub stall: Duration,
    /// How long this act is struck from the grammar afterwards.
    pub cool: Duration,
}

impl Cost {
    /// An act that costs nothing but the moment it took.
    const fn free() -> Cost {
        Cost {
            stall: Duration::ZERO,
            cool: Duration::ZERO,
        }
    }

    /// An act that is over as soon as it is decided, but may not be repeated
    /// for a while — a thing you *said* or *sent* rather than a thing you did
    /// with your body.
    const fn cools(secs: u64) -> Cost {
        Cost {
            stall: Duration::ZERO,
            cool: Duration::from_secs(secs),
        }
    }

    /// An act that takes time to do and then may not be done again for a
    /// while. `takes(4, 2)` is four seconds of doing it and two more before it
    /// is offered again.
    const fn takes(stall_secs: u64, cool_secs: u64) -> Cost {
        Cost {
            stall: Duration::from_secs(stall_secs),
            cool: Duration::from_secs(cool_secs),
        }
    }
}

/// What each act costs. Anything not named here is [`Cost::free`].
///
/// Speech is absent on purpose — two characters talking take turns as fast as
/// they like, and a rate on `say` would be a rate limit on conversation, which
/// is the one thing in the vault that should never be throttled.
const COST: &[(&str, Cost)] = &[
    // **Short on purpose.** The pacing was never a rate problem — it was a
    // grammar problem, and it is fixed where it was caused: `body::destinations`
    // no longer offers anywhere a journey would not actually go. What is left
    // for this to do is stop a body turning round in the doorway it just came
    // through, which takes a beat, not a quarter of a minute. A long one would
    // instead be a character standing at the wrong end of the building unable
    // to leave.
    // **Walking takes time; being somewhere else does not.**
    //
    // The stall is the crossing itself — a character that arrives and decides
    // again in the same instant is not walking anywhere, it is teleporting and
    // then thinking about it. One beat is enough to read as a journey, and a
    // journey longer than one room ends with its own arrival event, which wakes
    // the character early anyway.
    //
    // The cooldown stays short and is a different guard: it stops a body
    // turning round in the doorway it just came through. A long one would
    // strand somebody at the wrong end of the building. The pacing problem this
    // once carried was never a rate problem — it was `body::destinations`
    // offering rooms a journey would not actually go to, and it is fixed there.
    ("move_to", Cost::takes(4, 2)),
    // **The fight rate, and it is both halves of one beat.**
    //
    // Two characters acting on each other at three seconds apiece is an
    // exchange every second and a half, which reads as a scuffle. It is the
    // number to change if a fight feels wrong.
    //
    // The stall is what makes it a beat rather than a burst: putting a hand on
    // somebody takes as long as it takes, and a body that has just swung is
    // committed until it lands. It costs nothing in responsiveness — being hit
    // back is an event, and an event wakes a stalled character at once — so
    // what it removes is only the character re-deciding mid-swing.
    ("act", Cost::takes(3, 3)),
    // The same channel as `act` without the contact, so it may never be
    // *cheaper* than `act` — a character that could gesture freely while acts
    // were cooling would simply do its fighting through gestures.
    //
    // **Above the fight rate, not at it.** Three seconds is shorter than the
    // heartbeat, so it never actually bit: measured over 40 ticks, one
    // character gestured thirteen times and spoke once, every gesture aimed at
    // somebody who answered none of them. A rate below the tick is not a rate.
    //
    // Eight seconds still lets a gesture punctuate an exchange — it is meant to
    // be the wordless beat between sentences — while making it too expensive to
    // *be* the exchange. What a character reaches for instead is `say` and
    // `tell`, which cost nothing, which is the whole point of throttling this
    // and not those.
    ("gesture", Cost::cools(8)),
    // **A switch with two positions is a treadmill with two steps.**
    //
    // `Sim::modes_here` stopped offering a machine the state it is already in,
    // which killed the version of this where a character set the accession desk
    // to `reading` three times running. What it left was the same loop with an
    // extra step: `reading`, `working`, `reading`, `working` — eleven of each,
    // twenty-two of one character's twenty-seven acts. Every flip is legal,
    // every flip changes something, and the pair accomplishes nothing.
    //
    // Narrowing the branch could not have fixed that, because there is no state
    // the machine is wrongly being offered — it is the *rate* that is wrong.
    // Twelve seconds is long enough that working a machine reads as a decision
    // about the machine rather than as somewhere to put a turn.
    ("operate", Cost::cools(12)),
    // **An answer of "nothing" is still an answer, and it comes with a free
    // turn.**
    //
    // `scan` is in [`crate::engine::body::ANSWERS`], so a character that looks
    // somewhere is brought straight back to use what it was told — which is
    // right when it saw something and is an accelerator when it did not.
    // Measured the hour this act started working: twelve of thirty-one acts
    // across the whole cast were a scan, and every one of them came back
    // "Nothing moving."
    //
    // The narrowing that fixed `read` cannot be used here. `read` stops
    // offering a board with nothing new on it; offering only the places that
    // have something on them would tell the character where everything is
    // *through the grammar*, which is the one thing a scan is for finding out.
    // So the rate is the instrument's, not the information's: twenty seconds is
    // sweeping something at range, rather than a place to put a turn.
    //
    // The free follow-up still does its job — it simply lands on a turn where
    // looking again is not one of the options.
    ("scan", Cost::cools(20)),
    // **The most drastic thing a body can do to its own position, and it was
    // free.** Being called home crosses the whole building in an instant; a
    // character that can do it every four seconds has no reason to walk
    // anywhere, and twenty-four of sixty acts across a live cast were exactly
    // that. Refusing it while already home (see `enact::recall`) removes the
    // no-op; this is what stops it being the way somebody travels.
    //
    // It stalls, too, and for the ordinary reason: it is a journey, and the
    // longest one in the building.
    //
    // At the table's ceiling rather than past it. The refusal is what removes
    // the degenerate case — a character can now only be called home from
    // somewhere else, so any loop through this one has a walk in it, which is
    // a loop worth having.
    ("recall", Cost::takes(4, 30)),
    // **A thought does not occupy the body, so it does not stall at all.**
    //
    // This stalled for a full two minutes, inherited from the old `pause` whose
    // whole purpose was *going quiet until something happens* — the number was
    // a patience clock, not a duration, and it was carrying the loop-breaking
    // on its own because nothing else did.
    //
    // Under the question this column actually asks — *how long does doing it
    // take?* — the answer for noticing something is none. Crossing a room takes
    // time somebody watching could measure; taking stock does not. A character
    // that has just reflected is standing exactly where it was, free to be
    // spoken to and free to answer, and every second charged here was a second
    // it was absent from its own room for no reason the world could see.
    //
    // The cooldown is what does the work the long stall was doing badly.
    // The room wakes a stalled character within a beat or two — weather,
    // somebody moving — and the cheapest thing to do about being woken was to
    // reflect about being woken. Measured: twelve acts across forty ticks,
    // every one a `reflect`, asked a direct question twice in that window and
    // answering neither, with the same `my_reflections` sentence coming back
    // eleven times running.
    //
    // Half a minute makes reflection punctuation. It is shorter than
    // [`SELF_ACT`] — noticing something is smaller than doing something to your
    // own body — and far longer than the heartbeat, so a character woken after
    // reflecting has to answer with an act that reaches somebody.
    ("reflect", Cost::cools(30)),
    // **The one piece of speech that is rated, and the exception needs saying
    // out loud** — the note above this table is that speech is never throttled,
    // because a rate limit on conversation is a rate limit on the only thing
    // the vault is for.
    //
    // `message` is not that kind of speech. `say`, `tell` and `ask` are
    // `Availability::Nearby`: they need somebody standing there, they reach
    // that room and no further, and the company that makes them possible is
    // what bounds them. `message` is `Always`, reaches every character in the
    // world through the open channel, and is offered on every single turn
    // including the ones where a character is completely alone — which is
    // precisely when it is the most attractive thing on the list.
    //
    // Untethered, that is the pacing failure this engine keeps relearning in a
    // new costume: the cheapest act that looks social becomes the whole loop,
    // and a cast that texts all afternoon never walks anywhere.
    //
    // **But it is still conversation, and fifteen seconds was pricing it like
    // a chore.** Two characters on a thread paid thirty seconds a round trip —
    // long enough that an exchange never built, which is the failure at the
    // other end of the same axis. The rate is not here to make messaging
    // expensive; it is here to keep it *dearer than being in the room*, so that
    // walking to somebody stays the better way to talk to them.
    //
    // Six seconds does that with room to spare: speech in a room costs nothing
    // at all, so physical company is still an order of magnitude cheaper, while
    // ten sends a minute is a conversation rather than a correspondence.
    ("message", Cost::cools(6)),
];

/// How long a body waits after doing something to **its own** body.
///
/// **The same act at a different rate, because they are different things.**
/// Shoving somebody is a beat in a fight and belongs at three seconds; shifting
/// your own shoulders twice in a minute is fidgeting, and it was the whole of
/// what a solitary cast did — fifty-three acts out of fifty-three, every one on
/// itself, because a room that keeps handing a character physical things to
/// notice makes `act` the nearest verb to reach for.
///
/// Taking `yourself` away would have been the wrong fix: binding your own
/// wound, getting your own weapon clear, dragging yourself up off the floor are
/// all real, and none of them is something a character does twice a minute. A
/// minute is long enough that the act reads as a decision rather than a tic,
/// and short enough that a character which has just been hurt can still see to
/// itself.
pub const SELF_ACT: Duration = Duration::from_secs(60);

/// What this act costs, both halves. [`Cost::free`] for anything unnamed.
pub fn cost(tool: &str) -> Cost {
    COST.iter()
        .find(|(name, _)| *name == tool)
        .map(|(_, c)| *c)
        .unwrap_or_else(Cost::free)
}

/// How long this act is struck from the grammar, if at all.
pub fn after(tool: &str) -> Option<Duration> {
    let cool = cost(tool).cool;
    (!cool.is_zero()).then_some(cool)
}

/// How long the character stands still after taking this act, if at all.
///
/// The other half of [`cost`], and the one almost nothing has: an act is a
/// thing you do while living, not instead of it. See the table on [`Cost`].
pub fn stall_after(tool: &str) -> Option<Duration> {
    let stall = cost(tool).stall;
    (!stall.is_zero()).then_some(stall)
}

/// What this *particular* act costs — the one place the target matters.
///
/// Keyed on the act rather than its name, because `act` is two rates wearing
/// one word: see [`SELF_ACT`].
pub fn after_act(act: &Act) -> Option<Duration> {
    match on_self(act) {
        true => Some(SELF_ACT),
        false => after(act.tool),
    }
}

/// Whether this is a body doing something to itself.
fn on_self(act: &Act) -> bool {
    act.tool == "act"
        && act
            .args
            .get("on")
            .and_then(|v| v.as_str())
            .is_some_and(|on| on.eq_ignore_ascii_case(SELF))
}

/// Every act that costs anything, for whatever wants to report the rules.
pub fn all() -> &'static [(&'static str, Cost)] {
    COST
}

/// When each character may next take each act.
///
/// Keyed by character rather than by body: the limit belongs to the person
/// acting, and a character that changed bodies mid-fight would otherwise get a
/// free swing.
#[derive(Default)]
pub struct Cooldowns {
    /// Empty for a cast that has done nothing, and pruned as it is read, so a
    /// world running for a week does not accumulate a row per character per
    /// act for ever.
    until: Mutex<HashMap<(u64, &'static str), Instant>>,
}

impl Cooldowns {
    pub fn new() -> Cooldowns {
        Cooldowns::default()
    }

    /// Record that a character has just done something.
    ///
    /// **Takes the act, never just its name.** There was a `took(npc, tool)`
    /// beside this, which was convenient and was a trap: `took(npc, "act")`
    /// silently charges a self-act the fight rate, and nothing about the call
    /// says so. One entry point, and it always has enough to be right.
    ///
    /// Silently ignores an act with no cooldown, so every act can be handed
    /// here without the caller knowing which ones have one.
    pub fn took_act(&self, npc_id: u64, act: &Act) {
        self.stamp(npc_id, act.tool, after_act(act), Instant::now());
    }

    /// [`Self::took_act`], at a stated instant. The seam the tests drive, for
    /// the same reason `Building::next_event` takes its own: a test that had to
    /// wait out a real minute would not be written.
    pub fn took_act_at(&self, npc_id: u64, act: &Act, now: Instant) {
        self.stamp(npc_id, act.tool, after_act(act), now);
    }

    /// **Keyed by the name, whatever the wait was.** A self-act and an act on
    /// somebody share one row, because they are one act as far as the grammar
    /// is concerned — what differs is only how long the body waits, and a
    /// second row would let a character alternate between the two to dodge the
    /// wait entirely.
    fn stamp(&self, npc_id: u64, tool: &str, wait: Option<Duration>, now: Instant) {
        let Some(wait) = wait else { return };
        let Some((name, _)) = COST.iter().find(|(name, _)| *name == tool) else {
            return;
        };
        self.until
            .lock()
            .expect("cooldown lock")
            .insert((npc_id, *name), now + wait);
    }

    /// The acts this character may not take yet.
    ///
    /// What the grammar is built without. Nearly always empty, which is what
    /// makes it cheap to ask on every turn.
    pub fn cooling(&self, npc_id: u64) -> Vec<String> {
        self.cooling_at(npc_id, Instant::now())
    }

    pub fn cooling_at(&self, npc_id: u64, now: Instant) -> Vec<String> {
        let mut until = self.until.lock().expect("cooldown lock");
        // Expired rows are dropped as they are found. A sweep would need its
        // own clock; this needs nothing, and every row is looked at often.
        until.retain(|(_, _), ready| *ready > now);
        until
            .keys()
            .filter(|(who, _)| *who == npc_id)
            .map(|(_, tool)| (*tool).to_string())
            .collect()
    }

    /// Whether one act is ready. For a caller that has one act in mind.
    pub fn ready(&self, npc_id: u64, tool: &str) -> bool {
        !self.cooling(npc_id).iter().any(|t| t == tool)
    }

    /// Forget a character's waits — for one being retired, or one whose body
    /// has been taken out of the world.
    pub fn forget(&self, npc_id: u64) {
        self.until
            .lock()
            .expect("cooldown lock")
            .retain(|(who, _), _| *who != npc_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// One act, from the arguments a decode would have produced.
    fn act(tool: &'static str, args: serde_json::Value) -> Act {
        Act {
            tool,
            args: args.as_object().expect("an object").clone(),
        }
    }

    #[test]
    fn an_act_with_no_cooldown_never_waits() {
        // Speech above all: a cooldown on `say` would be a rate limit on
        // conversation.
        let c = Cooldowns::new();
        for tool in ["say", "tell", "ask", "read"] {
            assert!(after(tool).is_none(), "{tool} has a cooldown");
            c.took_act(1, &act(tool, json!({})));
            assert!(c.ready(1, tool), "{tool} made a character wait");
        }
        assert!(c.cooling(1).is_empty());
    }

    /// **Speech stays free and thinking does not.**
    ///
    /// The two halves of the same decision. A character woken by the room must
    /// not be able to answer it by reflecting again — that was the whole of one
    /// character's day, twelve acts out of twelve — and must always be able to
    /// answer somebody who spoke to it.
    #[test]
    fn thinking_is_rated_and_talking_is_not() {
        assert!(
            after("reflect").is_some(),
            "reflection is free again, and it will be the whole loop again"
        );
        for talking in ["say", "tell", "ask"] {
            assert!(
                after(talking).is_none(),
                "`{talking}` is throttled — that is a rate limit on conversation"
            );
        }
    }

    /// **This table holds two kinds of rate, and only one of them is about
    /// pacing a character's day.**
    ///
    /// `move_to` and `act` are *physical* rates. Two seconds stops a body
    /// turning round in the doorway it just came through; three seconds is the
    /// fight rate, which makes an exchange of blows read as a scuffle. Both are
    /// meant to be shorter than a heartbeat — a long one would strand somebody
    /// at the wrong end of the building, or turn a fight into a correspondence.
    ///
    /// `gesture`, `reflect` and `message` are *loop* rates. They exist because
    /// each one is cheap, always available, and will otherwise become the whole
    /// of what a character does. A loop rate inside the heartbeat is not a rate
    /// at all: `gesture` sat at three seconds against a four-second beat and
    /// never once bit — thirteen gestures in forty ticks, from a character who
    /// spoke once.
    /// The physical rates, which are allowed to be shorter than a heartbeat
    /// because they are pacing a body rather than a loop.
    const PHYSICAL: &[&str] = &["move_to", "act"];

    #[test]
    fn a_rate_meant_to_break_a_loop_costs_more_than_a_heartbeat() {
        let beat = Duration::from_secs(4);
        // Derived from the table rather than listed here: a new act given a
        // rate inside the heartbeat is a rate that will never bite, and it
        // should fail *without* anybody remembering to add it to this line.
        for (tool, c) in all().iter().filter(|(t, _)| !PHYSICAL.contains(t)) {
            assert!(
                c.cool > beat,
                "`{tool}` cools in {:?}, inside one heartbeat — it will never bite",
                c.cool
            );
        }
    }

    /// And the physical rates stay short, for the reasons that chose them.
    #[test]
    fn a_physical_rate_stays_inside_a_heartbeat() {
        let beat = Duration::from_secs(4);
        for tool in PHYSICAL {
            let wait = after(tool).unwrap_or_else(|| panic!("`{tool}` is not rated at all"));
            assert!(
                wait <= beat,
                "`{tool}` cools in {wait:?} — a fight or a doorway does not wait that long"
            );
        }
    }

    /// Gesturing may never be the cheap way to do what `act` costs. It is the
    /// same channel without the contact, so a character that could gesture
    /// freely while acts were cooling would do its fighting through gestures.
    #[test]
    fn gesturing_is_never_cheaper_than_acting() {
        let (Some(g), Some(a)) = (after("gesture"), after("act")) else {
            panic!("both are rated");
        };
        assert!(g >= a, "gesture ({g:?}) undercuts act ({a:?})");
    }

    #[test]
    fn taking_an_act_puts_it_out_of_reach_and_then_gives_it_back() {
        let c = Cooldowns::new();
        let t0 = Instant::now();
        c.took_act_at(7, &act("move_to", json!({"destination": "band one"})), t0);

        // Measured against the table rather than against a number written here
        // twice: a cooldown that is retuned should not need this test edited,
        // and a test that hard-codes the old value fails for the wrong reason.
        let wait = after("move_to").expect("walking cools");
        assert_eq!(c.cooling_at(7, t0), vec!["move_to".to_string()]);
        assert_eq!(
            c.cooling_at(7, t0 + wait - Duration::from_millis(1)),
            vec!["move_to".to_string()],
            "it came back early"
        );
        assert!(
            c.cooling_at(7, t0 + wait + Duration::from_millis(1))
                .is_empty(),
            "it never came back"
        );
    }

    #[test]
    fn one_character_waiting_does_not_hold_another_up() {
        // Keyed by character. A fight where one blow stopped everybody in the
        // room would be worse than no limit at all.
        let c = Cooldowns::new();
        let t0 = Instant::now();
        c.took_act_at(1, &act("act", json!({"on": "Perrin Vastwood"})), t0);
        assert_eq!(c.cooling_at(1, t0), vec!["act".to_string()]);
        assert!(c.cooling_at(2, t0).is_empty());
    }

    #[test]
    fn one_act_cooling_does_not_take_the_others_with_it() {
        let c = Cooldowns::new();
        let t0 = Instant::now();
        c.took_act_at(1, &act("move_to", json!({"destination": "band one"})), t0);
        assert!(c.ready(1, "act"), "walking stopped it fighting");
        assert!(c.ready(1, "say"), "walking stopped it talking");
    }

    /// **Doing something to your own body is not a fight beat.**
    ///
    /// A solitary cast chose `act` fifty-three times out of fifty-three, every
    /// one of them on itself, because a room that keeps handing a character
    /// physical things to notice makes this the nearest verb to reach for. The
    /// target is not the problem — binding your own wound is real — the rate is.
    #[test]
    fn acting_on_yourself_costs_a_longer_wait_than_acting_on_somebody() {
        let on_them = act(
            "act",
            json!({"on": "Perrin Vastwood", "intent": "stop him"}),
        );
        let on_me = act(
            "act",
            json!({"on": "yourself", "intent": "take the weight off it"}),
        );

        assert_eq!(after_act(&on_them), after("act"));
        assert_eq!(after_act(&on_me), Some(SELF_ACT));
        assert!(
            SELF_ACT > after("act").expect("acts cool") * 5,
            "a self-act is paced like a blow, so fidgeting is still free"
        );
    }

    /// **One row, whatever the wait was.** A self-act and an act on somebody
    /// are one act to the grammar — so if they were keyed apart a character
    /// could alternate between them and dodge the wait entirely.
    #[test]
    fn a_self_act_and_a_blow_share_one_cooldown() {
        let c = Cooldowns::new();
        let t0 = Instant::now();
        c.took_act_at(1, &act("act", json!({"on": "yourself"})), t0);
        assert_eq!(c.cooling_at(1, t0), vec!["act".to_string()]);

        // Still cooling well after a blow's three seconds, because what it did
        // was the slower of the two.
        assert_eq!(
            c.cooling_at(1, t0 + Duration::from_secs(10)),
            vec!["act".to_string()],
            "it dodged the wait by having a second row to write to"
        );
        assert!(c
            .cooling_at(1, t0 + SELF_ACT + Duration::from_secs(1))
            .is_empty());
    }

    #[test]
    fn a_fight_runs_at_about_a_blow_apiece_every_few_seconds() {
        // The number a fight is actually tuned by. Three seconds each, two
        // characters, is an exchange every second and a half.
        let blow = after("act").expect("acts cool");
        assert!(
            blow >= Duration::from_secs(2) && blow <= Duration::from_secs(5),
            "an exchange at {blow:?} apiece does not read as a fight"
        );
    }

    #[test]
    fn nothing_waits_longer_than_a_character_would_notice() {
        // Short because the point is to stop a loop, not to slow the world.
        for (tool, c) in all() {
            assert!(
                c.cool <= Duration::from_secs(30),
                "`{tool}` is out of the grammar for {:?}",
                c.cool
            );
        }
        // The self-act is the one deliberate exception, and it is bounded too:
        // a character that has just been hurt has to be able to see to itself
        // inside a couple of minutes.
        assert!(
            SELF_ACT <= Duration::from_secs(120),
            "a body cannot see to itself for {SELF_ACT:?}"
        );
    }

    /// **A stall is elapsed time, so it is bounded by what the act plausibly
    /// takes** — not by how much a character should be punished for choosing it.
    ///
    /// This was two minutes for `reflect`, inherited from the old `pause` whose
    /// number was a patience clock rather than a duration. Nothing a body does
    /// in a room takes two minutes, and a character stood down that long for
    /// noticing the air had changed is not being paced, it is being switched
    /// off. Ten seconds is the ceiling: past that it stops reading as *doing
    /// something* and starts reading as *not being there*.
    #[test]
    fn a_stall_is_no_longer_than_the_act_could_plausibly_take() {
        for (tool, c) in all() {
            assert!(
                c.stall <= Duration::from_secs(10),
                "`{tool}` occupies a body for {:?}, which is not a thing being done",
                c.stall
            );
        }
    }

    /// **Only what a body physically does occupies it.**
    ///
    /// Crossing a room and putting a hand on somebody take time a watcher could
    /// measure. Nothing else does: speaking and sending are over as soon as
    /// they are decided, and *thinking* is not a thing the world can see at
    /// all — a character that has just reflected is standing exactly where it
    /// was, free to be spoken to and free to answer.
    ///
    /// `reflect` stalled here twice, and both times the number was the mistake
    /// rather than the idea: two minutes inherited from the old `pause`, then
    /// eight seconds. A stall is elapsed time, and a thought takes none. The
    /// cooldown is what keeps reflection from becoming the whole day.
    #[test]
    fn only_what_a_body_physically_does_occupies_it() {
        let mut stalling: Vec<&str> = all()
            .iter()
            .filter(|(_, c)| !c.stall.is_zero())
            .map(|(t, _)| *t)
            .collect();
        stalling.sort_unstable();
        assert_eq!(stalling, vec!["act", "move_to", "recall"]);

        for quiet in ["say", "tell", "ask", "message", "gesture", "reflect"] {
            assert_eq!(
                stall_after(quiet),
                None,
                "`{quiet}` made a character busy for something the world cannot see"
            );
        }
    }

    /// **Talking to somebody in the room is always cheaper than texting them.**
    ///
    /// The ordering is the point, not the number. `message` is `Always`,
    /// reaches the whole world, and is offered on every turn including the ones
    /// where a character is alone — so if it ever costs less than walking over,
    /// the cast stops walking over. Speech in a room is free, so any cooldown
    /// at all preserves the preference; what the number decides is only whether
    /// a remote exchange can actually build.
    #[test]
    fn reaching_somebody_remotely_costs_more_than_speaking_to_them() {
        let remote = after("message").expect("messaging is rated");
        for near in ["say", "tell", "ask"] {
            assert_eq!(after(near), None, "`{near}` is throttled");
        }
        assert!(
            remote > Duration::from_secs(4),
            "a rate inside one heartbeat does not bite: {remote:?}"
        );
        assert!(
            remote <= Duration::from_secs(10),
            "at {remote:?} a send is a correspondence, not a conversation"
        );
    }

    /// The two halves are read from one table, and each accessor reports only
    /// its own — a caller asking "may this act be taken" must never be handed a
    /// stall, and vice versa.
    #[test]
    fn the_two_costs_are_read_apart_from_one_table() {
        assert_eq!(cost("move_to").stall, Duration::from_secs(4));
        assert_eq!(cost("move_to").cool, Duration::from_secs(2));
        assert_eq!(stall_after("move_to"), Some(Duration::from_secs(4)));
        assert_eq!(after("move_to"), Some(Duration::from_secs(2)));

        // Reflection is the case that has both a rate and no duration.
        assert_eq!(stall_after("reflect"), None);
        assert_eq!(after("reflect"), Some(Duration::from_secs(30)));

        // A cooling act that does not stall reports none, rather than zero.
        assert_eq!(stall_after("gesture"), None);
        assert_eq!(after("gesture"), Some(Duration::from_secs(8)));

        // And an act in neither column costs nothing at all.
        assert_eq!(cost("say"), Cost::free());
        assert_eq!(after("say"), None);
        assert_eq!(stall_after("say"), None);
    }

    #[test]
    fn a_retired_character_leaves_nothing_behind() {
        let c = Cooldowns::new();
        c.took_act(9, &act("move_to", json!({"destination": "band one"})));
        assert!(!c.cooling(9).is_empty());
        c.forget(9);
        assert!(c.cooling(9).is_empty());
    }
}
