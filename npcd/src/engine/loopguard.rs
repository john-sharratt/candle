//! Breaking a character out of a loop without letting it stop.
//!
//! # The failure this exists to stop
//!
//! npcd characters are *always active by design*: the tool stencil forces one act
//! per turn and there is deliberately no "sleep/idle" act — a character must
//! always be doing something. The failure that follows is a **loop**: a character
//! with nothing new to do, forced to act anyway in a situation that does not
//! resolve, settles into repeating one act. It was watched live — Ulysses asking
//! Pax the same question, verbatim, turn after turn after turn.
//!
//! # It is not degraded state — it is act selection
//!
//! A controlled experiment ruled out every stored-state explanation. From the
//! real looping conversation, the loop reproduces regardless of:
//!   - attention KV fidelity (reused/quantized vs fresh/lossless),
//!   - the persisted DeltaNet recurrent buffer (the looping buffer vs a freshly
//!     recomputed one — installed and decoded, it is coherent and carries no
//!     collapse),
//!   - context breadth (a narrow window vs the full ~20k-token history).
//!
//! Under the real tool grammar and the real sampler the loop reproduces from any
//! of those. So it is a **behavioural attractor of constrained act-selection in a
//! non-resolving situation**, not corruption in anything stored — which is why
//! the fix lives here, at act selection, and not in the persistence layer.
//!
//! # The design: a loop is the same thing done again
//!
//! **What is caught.** An act taken again *meaning the same as* one of its last
//! [`LOOKBACK`] takings — by token overlap at [`CLOSE`] for most acts. Any one of
//! them is enough: the rule that wanted *all* of the last three to match was
//! blinded by a single different call between repeats, and a Maker called a
//! chair the same refused way five times before it fired. Two kinds of act are
//! judged their own way:
//!
//! - **Speech is judged as a family.** `tell`, `ask`, `whisper` and `shout` are
//!   compared against each other, at the looser [`SPEECH_CLOSE`]: a settled piece
//!   of news is retold in fresh words and through a different act each time.
//! - **Drafting is revision.** Two drafts of one piece share most of their words,
//!   so a write or an edit is a repeat only when it is the same text again
//!   ([`DRAFTING`]).
//!
//! A device call is judged by the verb at its address, not as `invoke`: striking
//! `invoke` struck every device in reach — the table's `report_done` along with
//! the chair a Maker had been calling wrongly. What it carried is compared
//! whole, and a call is a repeat only when its body is the same again
//! ([`Took::call`]): a body names things, and two that name different ones are
//! different calls however many words they share.
//!
//! **What happens.** The looping act — for speech, every way of speaking — is
//! struck from the grammar, and nothing else is: struck, not refused, because
//! what a character cannot say it cannot get stuck saying (the same rule as
//! [`crate::engine::cooldown`]). The strike lasts `2^n` turns for the `n`th catch
//! in a row of that act, capped at [`COOL_CAP`], so a loop that resumes as soon
//! as the strike lifts is struck for longer each time; taking the act again
//! with something new ends its run. A [`NUDGE`] explains the strike.
//!
//! **What is not done**, each of which was, and each of which was the guard
//! causing what it was meant to stop:
//!
//! - *Cooling an act for being used.* Every act but a few was struck for
//!   `2^(uses in a window)` turns whether or not anything repeated, so a Maker
//!   reading, editing and committing at a desk had its own ordinary work taken
//!   off it.
//! - *Forcing a reflect.* A catch struck everything but `reflect` and `move_to`
//!   for a turn. A Maker at the command table with a report to file was put
//!   through that every other turn, reflected the same words each time — the
//!   forced reflect overrode the strike on a looping `reflect` — and was never
//!   offered `report_done`; another, caught writing one draft twice, could only
//!   think or walk out of its writing room, and walked.
//! - *Striking a walk.* `move_to` is how a character gets anywhere, including
//!   out of whatever it is stuck on, and is never struck.
//!
//! Acts the game wants repeated — `act` in a fight — skip the guard while
//! something hostile is here, paced by `cooldown`'s fight rate instead.

use std::collections::{HashMap, HashSet, VecDeque};
use std::slice::from_ref;
use std::sync::Mutex;

use serde_json::Value;

use crate::engine::act::Act;

/// Acts the game wants repeated in a fight — they skip the guard while
/// something here is hostile, paced by `cooldown`'s real-time fight rate.
const GUARD_EXEMPT: &[&str] = &["act"];

/// The acts that are never struck. `move_to` and the lift's `lift_use` and
/// `lift_call` are how a character gets anywhere, including out of whatever it
/// is going round on — and calling the car again, naming the same level, is
/// waiting for it, not saying the same thing twice. `compose` carries no words
/// to compare — every sitting writes the piece afresh — and the engine refuses
/// a sitting over a piece already written itself (`Runtime::compose`).
const NEVER_STRUCK: &[&str] = &["move_to", "lift_use", "lift_call", "compose"];

/// The acts that put words into a document. A repeat is the same text again —
/// see [`same_text`].
const DRAFTING: &[&str] = &["file_write", "file_edit"];

/// Every way a character speaks, judged and struck as one — on a thread as in
/// the room. With `message` outside the family, four Makers told each other
/// "the lift is broken" for fifty minutes, turn about between the channel and
/// a shout, and none of them called the car.
const SPEECH: &[&str] = &["tell", "ask", "whisper", "shout", "message"];

/// Token overlap at which two takings of an act mean the same.
const CLOSE: f32 = 0.6;

/// Token overlap at which two things said are the same news. Looser than
/// [`CLOSE`], because settled news is retold in new words and lands near 0.5.
const SPEECH_CLOSE: f32 = 0.45;

/// How many of an act's last takings a new one is compared with.
const LOOKBACK: usize = 3;

/// How many of its last utterances something said is compared with.
const SPEECH_LOOKBACK: usize = 4;

/// The longest any strike lasts, in turns.
const COOL_CAP: usize = 16;

/// The catch count at which a strike stops doubling (`2^4` turns).
const ESCALATION_CAP: usize = 4;

/// What a character reads at the head of the turn after a catch, so the strike
/// is *explained* and not just imposed. Delivered as a `<tool_response>`
/// through the ordinary outcome channel — see
/// [`crate::engine::mind::Minds::deliver_outcomes`].
///
/// It points at the work, not away from the room: "turn to someone or something
/// else" sent a Maker off mid-conversation, out of earshot of the colleague who
/// was about to answer it. The runtime follows it with the character's next
/// mission step when it has one.
pub const NUDGE: &str =
    "You seem to be repeating yourself; what you have already said has been heard. Try something \
     different — take the next step of what you are working on, or answer what you have been \
     asked.";

/// One act a character took this turn, as the guard weighs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Took {
    /// The act — for a device call, the verb at its address.
    pub act: String,
    /// What it meant — see [`salient`].
    pub intent: String,
    /// Whether it was a device call, whose body names things — a path, a line,
    /// a field — and so is the same only when it is identical.
    pub call: bool,
}

impl Took {
    pub fn new(act: impl Into<String>, intent: impl Into<String>) -> Took {
        Took {
            act: act.into(),
            intent: intent.into(),
            call: false,
        }
    }

    /// A device call to `act`, carrying `body`.
    ///
    /// **A call is the same only when it is the same.** Compared by overlap,
    /// with its address in the text, every read at a desk shared most of its
    /// words with the last — the address's words and the layer's — and a
    /// reviewer reading its draft, then the era, then the next page was struck
    /// for repeating itself.
    pub fn call(act: impl Into<String>, body: impl Into<String>) -> Took {
        Took {
            act: act.into(),
            intent: body.into(),
            call: true,
        }
    }
}

/// The text of an act that the closeness test compares — what the character
/// *meant*, pulled from whichever argument carries the substance.
///
/// The catalog's content arguments by act: `ask` uses `about`, the speech and
/// gesture acts use `intent`, `reflect` its `inner_thoughts`, `move_to` its
/// `destination`, `follow` its `target`. Anything else falls back to every
/// string argument joined, so a new act is compared on *something* rather than
/// silently never looping.
pub fn salient(act: &Act) -> String {
    // A device call is what it carried: the guard already keeps it under the
    // verb at its address. The fields arrive as an object, which the string
    // fallback below would drop — leaving every call to one address, whatever
    // it wrote, looking the same.
    if act.tool == "invoke" {
        return match act.args.get("body") {
            Some(Value::String(s)) => s.clone(),
            Some(body) => body.to_string(),
            None => String::new(),
        };
    }
    for key in ["about", "intent", "inner_thoughts", "destination", "target"] {
        if let Some(s) = act.args.get(key).and_then(Value::as_str) {
            if !s.trim().is_empty() {
                return s.to_string();
            }
        }
    }
    let mut joined: Vec<&str> = act
        .args
        .values()
        .filter_map(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .collect();
    joined.sort_unstable();
    joined.join(" ")
}

/// The fewest words for which overlap, rather than identity, decides whether
/// two takings mean the same.
const SHORT: usize = 6;

/// The words a taking is compared on: every word of three letters or more, and
/// every number — `era-0` and `era-1` are two documents, not one.
fn words(s: &str) -> HashSet<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 2 || (!t.is_empty() && t.chars().all(|c| c.is_ascii_digit())))
        .map(str::to_string)
        .collect()
}

/// Whether two takings mean the same: the same words, for a short one; word
/// overlap at `close` or above, for a longer one.
///
/// **A short taking is the same only when it is the same.** Two reads of
/// different documents, or two calls naming different things, share most of
/// their few words — overlap called them one, and ordinary work was struck as a
/// loop. A long one is a paraphrase when it shares most of its words, and
/// overlap is what catches that.
fn means_the_same(a: &str, b: &str, close: f32) -> bool {
    let (wa, wb) = (words(a), words(b));
    if wa.is_empty() || wb.is_empty() {
        return false;
    }
    if wa.len().min(wb.len()) < SHORT {
        return wa == wb;
    }
    let shared = wa.intersection(&wb).count() as f32;
    let either = wa.union(&wb).count() as f32;
    shared / either >= close
}

/// Two drafts that are the same text, whitespace aside.
fn same_text(a: &str, b: &str) -> bool {
    a.split_whitespace().eq(b.split_whitespace())
}

/// One character's loop-guard state.
#[derive(Default)]
struct Guard {
    /// The turn counter, advanced once per decode (not once per act).
    turn: usize,
    /// Act (or speech, as one) → what its last takings meant, newest last —
    /// [`LOOKBACK`] of them, or [`SPEECH_LOOKBACK`] for speech. Kept per act,
    /// so a long strike walked out elsewhere does not push the looping act out
    /// of memory and let the loop start again as if new.
    taken: HashMap<String, VecDeque<String>>,
    /// Act name → the turn it is offered again.
    cool_until: HashMap<String, usize>,
    /// Act (or speech, as one) → catches in a row. A run ends only when that
    /// act is taken again without repeating, so a walk or a different act
    /// between two repeats does not restart it.
    runs: HashMap<String, usize>,
}

/// What an act's takings are kept and judged under: speech as one, every
/// other act by its name.
fn kept_as(act: &str) -> &str {
    match SPEECH.contains(&act) {
        true => SPEECH[0],
        false => act,
    }
}

impl Guard {
    /// Whether `intent`, taken as `act`, means the same as one of its recent
    /// takings — see the module's design.
    fn repeats(&self, act: &str, intent: &str, call: bool) -> bool {
        let Some(earlier) = self.taken.get(kept_as(act)) else {
            return false;
        };
        if call {
            return earlier.iter().any(|prev| same_text(prev, intent));
        }
        if SPEECH.contains(&act) {
            return earlier
                .iter()
                .any(|said| means_the_same(said, intent, SPEECH_CLOSE));
        }
        match DRAFTING.contains(&act) {
            true => earlier.iter().any(|prev| same_text(prev, intent)),
            false => earlier
                .iter()
                .any(|prev| means_the_same(prev, intent, CLOSE)),
        }
    }

    /// Keep what `act` meant this time, dropping the oldest past its lookback.
    fn keep(&mut self, act: &str, intent: &str) {
        let depth = match SPEECH.contains(&act) {
            true => SPEECH_LOOKBACK,
            false => LOOKBACK,
        };
        let kept = self.taken.entry(kept_as(act).to_string()).or_default();
        kept.push_back(intent.to_string());
        while kept.len() > depth {
            kept.pop_front();
        }
    }

    /// Acts to strike from this turn's grammar.
    fn cooling(&self) -> Vec<String> {
        self.cool_until
            .iter()
            .filter(|(_, until)| **until > self.turn)
            .map(|(act, _)| act.clone())
            .collect()
    }

    /// Fold this decode's acts in. Returns whether a loop was caught, so the
    /// caller can deliver the [`NUDGE`].
    fn record(&mut self, acts: &[Took], fighting: bool) -> bool {
        let turn = self.turn;
        let mut caught = false;
        for took in acts {
            let (act, intent) = (took.act.as_str(), took.intent.as_str());
            if NEVER_STRUCK.contains(&act) || (fighting && GUARD_EXEMPT.contains(&act)) {
                continue;
            }
            let spoken = SPEECH.contains(&act);
            let run_of = kept_as(act);
            if !self.repeats(act, intent, took.call) {
                self.runs.remove(run_of);
                continue;
            }
            let run = self.runs.entry(run_of.to_string()).or_insert(0);
            *run += 1;
            let strike = (1usize << (*run).min(ESCALATION_CAP)).min(COOL_CAP);
            let struck: &[&str] = match spoken {
                true => SPEECH,
                false => from_ref(&act),
            };
            for name in struck {
                let until = self.cool_until.entry(name.to_string()).or_insert(0);
                *until = (*until).max(turn + 1 + strike);
            }
            caught = true;
        }
        for took in acts {
            self.keep(&took.act, &took.intent);
        }
        self.turn += 1;
        self.cool_until.retain(|_, until| *until > self.turn);
        caught
    }
}

/// Every character's loop guard. Keyed by character rather than body, the same as
/// [`crate::engine::cooldown::Cooldowns`]: the tendency to loop belongs to the
/// mind, not whatever body it is wearing.
#[derive(Default)]
pub struct LoopGuards {
    guards: Mutex<HashMap<u64, Guard>>,
}

impl LoopGuards {
    pub fn new() -> LoopGuards {
        LoopGuards::default()
    }

    /// The acts this character may not take this turn — struck from its grammar.
    /// Nearly always empty, which is what makes it cheap to ask every turn.
    pub fn cooling(&self, npc_id: u64) -> Vec<String> {
        self.guards
            .lock()
            .expect("loop-guard lock")
            .get(&npc_id)
            .map(Guard::cooling)
            .unwrap_or_default()
    }

    /// Record what this character just chose to do. `acts` is every well-formed
    /// act from one decode, each with its [`salient`] intent, in call order — a
    /// device call under the verb at its address. Returns whether a loop was
    /// caught; the caller then delivers the [`NUDGE`].
    pub fn record(&self, npc_id: u64, acts: &[Took], fighting: bool) -> bool {
        if acts.is_empty() {
            return false;
        }
        self.guards
            .lock()
            .expect("loop-guard lock")
            .entry(npc_id)
            .or_default()
            .record(acts, fighting)
    }

    /// Forget a character's history — for one being retired, or one whose body
    /// has left the world.
    pub fn forget(&self, npc_id: u64) {
        self.guards.lock().expect("loop-guard lock").remove(&npc_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn act(tool: &'static str, args: serde_json::Value) -> Act {
        Act {
            tool,
            args: args.as_object().expect("an object").clone(),
        }
    }

    fn took(g: &LoopGuards, npc: u64, tool: &str, intent: &str) -> bool {
        g.record(npc, &[Took::new(tool, intent)], false)
    }

    fn took_fighting(g: &LoopGuards, npc: u64, tool: &str, intent: &str) -> bool {
        g.record(npc, &[Took::new(tool, intent)], true)
    }

    fn struck(g: &LoopGuards, npc: u64, tool: &str) -> bool {
        g.cooling(npc).contains(&tool.to_string())
    }

    /// The salient text is the content argument, whichever one carries it.
    #[test]
    fn the_salient_text_is_the_content_argument() {
        assert_eq!(
            salient(&act("ask", json!({"to": "Pax", "about": "the lights"}))),
            "the lights"
        );
        assert_eq!(
            salient(&act("tell", json!({"to": "Pax", "intent": "come here"}))),
            "come here"
        );
        assert_eq!(
            salient(&act("move_to", json!({"destination": "the deck"}))),
            "the deck"
        );
    }

    /// **Using an act is not looping on it.** A Maker reading, editing and
    /// committing — different things, one after another, again and again — has
    /// nothing struck. The old guard cooled almost every act for being used.
    #[test]
    fn ordinary_work_is_never_struck() {
        let g = LoopGuards::new();
        let commits = [
            "the first city opens, as the shooter saw it",
            "the archive day in 2491, written in full",
            "Conan at the unlocked door in the contested cities",
            "the drone's recovery on line seven",
            "the ledger the governor kept and the keeper burned",
            "Ash coming back after nineteen dark years",
        ];
        for (i, commit) in commits.iter().enumerate() {
            assert!(!took(
                &g,
                1,
                "file_read",
                &format!("layers/eras/era-{i}.md")
            ));
            assert!(!took(&g, 1, "bench_commit", commit));
            assert!(!took(
                &g,
                1,
                "ask",
                &format!("question number {i} about {}", i * 13)
            ));
        }
        assert!(g.cooling(1).is_empty(), "{:?}", g.cooling(1));
    }

    /// **Reading at a desk is not a loop.** Device reads of different
    /// documents and of successive pages of one are different calls; the same
    /// page read again is the same call. A sitting to compose is never judged.
    #[test]
    fn reading_at_a_desk_is_judged_by_what_it_names() {
        let g = LoopGuards::new();
        let read = |body: serde_json::Value| {
            let a = act(
                "invoke",
                json!({"url": "http://local/bench/desk~1/file_read", "body": body}),
            );
            g.record(1, &[Took::call("file_read", salient(&a))], false)
        };
        assert!(!read(
            json!({"path": "layers/life/keeper/2786 The Charge.md"})
        ));
        assert!(!read(json!({"path": "layers/eras/the-retreat.md"})));
        assert!(!read(
            json!({"path": "layers/eras/the-retreat.md", "start_line": "201"})
        ));
        assert!(!read(json!({"path": "layers/eras/the-fall.md"})));
        assert!(g.cooling(1).is_empty(), "{:?}", g.cooling(1));
        assert!(
            read(json!({"path": "layers/eras/the-fall.md"})),
            "the same page again"
        );

        let compose = act(
            "invoke",
            json!({"url": "http://local/bench/desk~1/compose"}),
        );
        for _ in 0..4 {
            assert!(!g.record(2, &[Took::call("compose", salient(&compose))], false));
        }
        assert!(g.cooling(2).is_empty());
    }

    /// **The same thing done again is caught the second time, and only it is
    /// struck.** Everything else the character could do stays on offer — the
    /// guard used to force a reflect, taking everything else away.
    #[test]
    fn a_repeat_is_caught_and_only_it_is_struck() {
        let g = LoopGuards::new();
        let present = "http://local/creator/creators-chair~0/present the mission is complete";
        assert!(!took(&g, 1, "creator_present", present));
        assert!(took(&g, 1, "creator_present", present), "a repeat");
        assert!(struck(&g, 1, "creator_present"));
        for still in ["report_done", "reflect", "tell", "gesture", "file_read"] {
            assert!(!struck(&g, 1, still), "{still} was struck with it");
        }
    }

    /// **A repeat is a repeat with something else between.** The rule that
    /// wanted every one of the last three takings to match was blinded by a
    /// single different call, and a refused chair was called five times over.
    #[test]
    fn a_repeat_with_something_else_between_is_still_caught() {
        let g = LoopGuards::new();
        let same = "http://local/creator/creators-chair~0/present I documented the shift";
        took(&g, 1, "creator_present", same);
        took(&g, 1, "reflect", "the ground is no longer static");
        took(
            &g,
            1,
            "creator_present",
            "http://local/creator/creators-chair~0/present I have returned to the level",
        );
        assert!(took(&g, 1, "creator_present", same));
    }

    /// **The strike grows while the loop goes on.** A character that takes the
    /// act again the moment it is offered is struck for longer each time, so
    /// in forty turns it comes back a handful of times, not every few.
    #[test]
    fn a_loop_that_resumes_is_struck_for_longer_each_time() {
        let g = LoopGuards::new();
        let mut taken = 0;
        for turn in 0..40 {
            match struck(&g, 1, "act") {
                true => {
                    took(&g, 1, "move_to", &format!("room{turn}"));
                }
                false => {
                    took(&g, 1, "act", "binding your own wound tighter on the bench");
                    taken += 1;
                }
            }
        }
        assert!(
            taken <= 6,
            "the act came back {taken} times in 40 turns; the strike must keep growing"
        );
    }

    /// **The same reflection is struck, and nothing forces another.** A Maker
    /// produced one reflection word for word every other turn, because a forced
    /// reflect overrode the strike on it.
    #[test]
    fn a_reflection_said_again_is_struck() {
        let g = LoopGuards::new();
        let thought = "I keep trying to invoke a function, but that's not how this works. I need \
                       to simply state the facts.";
        took(&g, 1, "reflect", thought);
        took(&g, 1, "tell", "that the mission is complete");
        assert!(took(&g, 1, "reflect", thought));
        assert!(struck(&g, 1, "reflect"));
        assert!(!struck(&g, 1, "invoke"), "nothing else is taken away");
    }

    /// Taking the act again with something new ends its run: a later repeat is
    /// struck from the short strike again, not the long one.
    #[test]
    fn something_new_ends_the_run() {
        let g = LoopGuards::new();
        let same = "binding your own wound tighter on the bench";
        // The act is taken only when it is offered, as the grammar allows, and
        // its strikes are walked out in between.
        let mut walked = 0;
        let mut wait_it_out = |g: &LoopGuards| {
            while struck(g, 1, "act") {
                took(g, 1, "move_to", &format!("room{walked}"));
                walked += 1;
            }
        };
        for _ in 0..5 {
            wait_it_out(&g);
            took(&g, 1, "act", same);
        }
        wait_it_out(&g);
        took(
            &g,
            1,
            "act",
            "checking the valve on the coolant loop for a leak",
        );
        took(
            &g,
            1,
            "act",
            "reading the gauge on the far wall until it settles",
        );
        took(
            &g,
            1,
            "act",
            "wiping the dust from the console glass with a sleeve",
        );
        took(
            &g,
            1,
            "act",
            "counting the bolts along the lower rail one by one",
        );
        took(&g, 1, "act", same);
        assert!(took(&g, 1, "act", same), "the repeat is caught again");
        let mut turns = 0;
        while struck(&g, 1, "act") {
            took(&g, 1, "move_to", &format!("hall{turns}"));
            turns += 1;
        }
        assert!(
            turns <= 3,
            "a fresh loop starts from the short strike: {turns}"
        );
    }

    const STRIP_DONE: &str = "that Pax has re-secured the threshold strip and that it's done, so \
                              we don't need to keep asking about it";
    const STRIP_RESOLVED: &str = "that Pax has re-secured the threshold strip and that we can \
                                  consider this matter resolved, so there's no need to keep \
                                  asking or repeating";

    /// **Speech drifts in wording while the content stands still.** A settled
    /// piece of news retold in new words is the same news.
    #[test]
    fn a_reworded_restatement_is_caught() {
        let g = LoopGuards::new();
        assert!(!took(&g, 1, "tell", STRIP_DONE));
        assert!(took(&g, 1, "tell", STRIP_RESOLVED));
    }

    /// Switching to another way of speaking does not make it new.
    #[test]
    fn a_restatement_is_caught_across_speech_acts() {
        let g = LoopGuards::new();
        took(&g, 1, "tell", STRIP_DONE);
        assert!(took(
            &g,
            1,
            "ask",
            "whether the threshold strip that Pax has re-secured is done"
        ));
    }

    /// Every way of speaking is struck together, so the same news cannot be
    /// found again in another one — and nothing but speech is struck.
    #[test]
    fn a_restatement_strikes_every_way_of_speaking_and_nothing_else() {
        let g = LoopGuards::new();
        took(&g, 1, "tell", STRIP_DONE);
        took(&g, 1, "tell", STRIP_RESOLVED);
        for speech in SPEECH {
            assert!(struck(&g, 1, speech), "{speech} is still on offer");
        }
        assert!(!struck(&g, 1, "reflect"));
        assert!(!struck(&g, 1, "report_done"));
    }

    /// Saying something new is not a restatement, however recently the
    /// character spoke.
    #[test]
    fn saying_something_new_is_not_a_restatement() {
        let g = LoopGuards::new();
        assert!(!took(&g, 1, "tell", STRIP_DONE));
        assert!(!took(
            &g,
            1,
            "tell",
            "that the lift is stuck between floors and somebody should look at the cable"
        ));
        assert!(!took(&g, 1, "ask", "who has the key to the lower stores"));
    }

    const DRAFT_ONE: &str =
        "layers/life/zen/2491 The Silence.md The maintenance was scheduled and \
                             completed. The log records nothing anomalous for this day.";
    const DRAFT_TWO: &str = "layers/life/zen/2491 The Silence.md The maintenance was scheduled. \
                             It was completed. The log records nothing anomalous for this day, \
                             and the printer in the corner gives the sheet back.";

    /// **Revising a document is not a loop.**
    #[test]
    fn a_revision_of_a_document_is_not_a_repeat() {
        let g = LoopGuards::new();
        let three = format!("{DRAFT_TWO} The sheet is blank on both sides.");
        let four = format!("{three} Nobody signs for it.");
        for draft in [DRAFT_ONE, DRAFT_TWO, &three, &four] {
            assert!(!took(&g, 1, "file_write", draft), "a revision was caught");
            assert!(g.cooling(1).is_empty(), "{:?}", g.cooling(1));
        }
    }

    /// Going back to a draft already written, word for word, is going round.
    #[test]
    fn returning_to_an_earlier_draft_is_a_repeat() {
        let g = LoopGuards::new();
        took(&g, 1, "file_write", DRAFT_ONE);
        took(&g, 1, "file_write", DRAFT_TWO);
        assert!(took(&g, 1, "file_write", DRAFT_ONE));
    }

    /// The same text written over itself again is the loop it looks like, and
    /// the rest of the bench stays on offer.
    #[test]
    fn the_same_draft_written_again_strikes_only_the_write() {
        let g = LoopGuards::new();
        took(&g, 1, "file_write", DRAFT_ONE);
        assert!(took(&g, 1, "file_write", &DRAFT_ONE.replace(". ", ".  ")));
        assert!(struck(&g, 1, "file_write"));
        for still in ["file_edit", "bench_commit", "file_read"] {
            assert!(!struck(&g, 1, still), "{still} was struck with it");
        }
    }

    /// A device call is weighed by what it carried, not by the address alone.
    #[test]
    fn a_draft_through_a_device_is_weighed_by_what_it_carried() {
        let call = |content: &str| {
            act(
                "invoke",
                json!({"url": "http://local/bench/desk~1/file_write",
                       "body": {"path": "a.md", "content": content}}),
            )
        };
        let (one, two) = (
            salient(&call("first draft")),
            salient(&call("second draft")),
        );
        assert_ne!(one, two);
        let g = LoopGuards::new();
        assert!(!took(&g, 1, "file_write", &one));
        assert!(!took(&g, 1, "file_write", &two));
        assert!(took(&g, 1, "file_write", &two), "the same draft again");
    }

    /// **A walk is never struck** — it is how a character gets out of anything.
    #[test]
    fn walking_the_same_way_is_never_struck() {
        let g = LoopGuards::new();
        for _ in 0..6 {
            assert!(!took(&g, 1, "move_to", "the board room"));
        }
        assert!(!struck(&g, 1, "move_to"));
        // Calling the car again, naming the same level, is waiting for it.
        for _ in 0..6 {
            assert!(!took(&g, 2, "lift_use", "the cartography level"));
            assert!(!took(&g, 2, "lift_call", ""));
        }
        assert!(g.cooling(2).is_empty(), "{:?}", g.cooling(2));
    }

    /// **A message is speech.** The same news sent on the channel and then
    /// shouted is one thing said twice, and every way of speaking is struck.
    #[test]
    fn a_message_and_a_shout_saying_the_same_are_one_repeat() {
        let g = LoopGuards::new();
        let news = "The lift is broken. I need to get to the cartography level, but I cannot \
                    find the stairs or any other way down.";
        assert!(!took(&g, 1, "message", news));
        assert!(took(&g, 1, "shout", news), "said again, aloud");
        for speech in ["message", "shout", "tell"] {
            assert!(struck(&g, 1, speech), "{speech}");
        }
    }

    /// A combat act opts out of the guard while something is being fought.
    #[test]
    fn a_combat_act_is_never_struck() {
        let g = LoopGuards::new();
        for _ in 0..5 {
            assert!(!took_fighting(&g, 1, "act", "strike him again"));
            assert!(!struck(&g, 1, "act"));
        }
    }

    /// The same `act` with nothing hostile here is a tic, and is caught.
    #[test]
    fn a_social_act_off_the_field_is_guarded_like_any_other() {
        let g = LoopGuards::new();
        took(&g, 1, "act", "shake Pax to make them see the pattern");
        assert!(took(&g, 1, "act", "shake Pax to make them see the pattern"));
        assert!(struck(&g, 1, "act"));
    }

    /// A retired character leaves nothing behind.
    #[test]
    fn forget_clears_a_character() {
        let g = LoopGuards::new();
        took(&g, 9, "ask", "again and again and again");
        took(&g, 9, "ask", "again and again and again");
        assert!(!g.cooling(9).is_empty());
        g.forget(9);
        assert!(g.cooling(9).is_empty());
    }

    /// One character looping does not strike another's grammar.
    #[test]
    fn one_character_does_not_cool_another() {
        let g = LoopGuards::new();
        took(&g, 1, "ask", "stuck on this question here");
        took(&g, 1, "ask", "stuck on this question here");
        assert!(struck(&g, 1, "ask"));
        assert!(g.cooling(2).is_empty());
    }

    /// The nudge sends the character to its work, never away from the room.
    #[test]
    fn the_nudge_points_at_the_work() {
        assert!(!NUDGE.contains("someone or something else"), "{NUDGE}");
        assert!(NUDGE.contains("next step"), "{NUDGE}");
    }
}
