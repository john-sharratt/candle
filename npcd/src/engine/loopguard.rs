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
//! Under the real tool grammar and the real sampler the loop reproduces from any
//! of those. So it is a **behavioural attractor of constrained act-selection in a
//! non-resolving situation**, not corruption in anything stored — which is why
//! the fix lives here, at act selection, and not in the persistence layer.
//!
//! # The design
//!
//! Keep "always active", but forbid *sameness* and drive toward *novelty*. This
//! is the turn-based counterpart to [`crate::engine::cooldown`], which paces the
//! *body* in real time (a fight rate, a doorway); this paces *choice* in turns.
//!
//! 1. **Exponential cooldown on most acts.** An act repeated within a sliding
//!    window is struck from the grammar for `2^(uses in window)` turns — the more
//!    it repeats, the longer it is gone. Struck, not refused: what a character
//!    cannot say, it cannot get stuck saying (same rule as `cooldown`). A
//!    **protected set — `reflect`, `move_to`, `tell` — is exempt** from the
//!    exponential rule: these keep a character *interacting* or *moving*, so they
//!    stay reachable. `reflect`/`tell` still carry a light **anti-repeat** (struck
//!    the one following turn, so no immediate repeat); **`move_to` is never
//!    struck** — it is an ongoing activity, not a repeatable tic.
//!
//! 2. **Closeness-triggered circuit-breaker.** A similarity test over the last
//!    few same-act intents catches a near-verbatim loop the cooldown alone can
//!    ride — an A-B-A-B cycle, or paraphrase repeats of the same act name. On
//!    detection it fires a graduated breaker, three parts covering each other:
//!      a. a **nudge** into the next turn's perception ("you seem stuck, try
//!         something different"). Weak alone — the model demonstrably reads such
//!         self-nudges and repeats anyway — so it only *explains* the redirect;
//!      b. an **adaptive cooldown** on the offending act for `repetitions + 1`
//!         turns. This is the *guarantee*: whatever the model concludes, it
//!         cannot re-emit the act;
//!      c. a **forced single reflect** next turn — everything but `reflect` (and
//!         `move_to`, so the grammar is never empty) is struck. `reflect` is a
//!         genuinely different activity (the entropy channel), and when the
//!         character returns the offending act is still cooled, so it must land
//!         somewhere new. Single-shot, so it cannot become a reflect loop itself.
//!    Per-act opt-out: acts the game *wants* repeated (`act` — a fight needs
//!    repetition) skip the whole guard, paced only by `cooldown`'s fight rate.
//!
//! Impulsivity-driven thinking and synthetic directed interrupts were the other
//! two mechanisms trialled in the bench. They are not ported: the bench found
//! they add complexity (empty-intent artifacts from quick-think turns) without
//! beating the two above, and *directed interrupts happen for free* in the live
//! cast — one character acting on another is already a real, novel event that
//! arrives and breaks the target's cycle, with nothing to synthesize.
//!
//! # Bench results that chose this design (2026-09-14, Qwen3.6-35B, real sampler)
//!
//! From the real looping state (16-turn window + persisted looping recurrent
//! buffer), 12 constrained turns per policy:
//!
//! | policy | longest same-act streak | verdict |
//! |---|---|---|
//! | baseline (no guard) | 11 (`ask` ×11, verbatim) | LOOP — reproduces the live failure |
//! | exponential cooldown + protected set | 1 | broken — rotates ask/tell/gesture, content advances |
//! | closeness breaker (+ exp cooldown) | 1 | broken — coherent forward progress |
//! | combined (+ impulsivity + interrupts) | 1 | broken, but quick-think turns emit empty intents |
//!
//! The exponential cooldown alone breaks the verbatim loop and is the primary
//! lever; the closeness breaker is the backstop that catches paraphrase loops the
//! per-act-name cooldown misses, and its forced reflect gives the most coherent
//! trajectory.

use std::collections::HashMap;
use std::sync::Mutex;

use serde_json::Value;

use crate::engine::act::Act;
use crate::engine::tools::CATALOG;

/// Acts exempt from the exponential rule — the ones that keep a character
/// interacting or moving. See design note 1. `tell` is the ordinary speaking
/// voice; `ask` is deliberately *not* here, because the loop watched live was an
/// unanswerable question asked over and over, and letting it escalate is exactly
/// what breaks it.
const PROTECTED: &[&str] = &["reflect", "move_to", "tell"];

/// Of the protected set, these still get a one-turn anti-repeat so they cannot be
/// emitted twice running. `move_to` is not here — it is an ongoing activity.
const ANTI_REPEAT: &[&str] = &["reflect", "tell"];

/// Acts the game wants repeated — they opt out of the whole guard. A fight is
/// repetition by nature, and `cooldown`'s real-time fight rate already paces it.
const GUARD_EXEMPT: &[&str] = &["act"];

/// The act that is never struck, so the grammar under a forced reflect (or any
/// pile-up of cooldowns) always has at least this and `reflect` to offer.
const NEVER_STRUCK: &str = "move_to";

/// The redirect a forced reflect must always be able to reach.
const ALWAYS_REFLECT: &str = "reflect";

/// The sliding window, in history entries, the exponential use-count looks over.
const WINDOW: usize = 6;
/// The ceiling on any single cooldown, in turns.
const COOL_CAP: usize = 16;
/// Jaccard token overlap between two intents that counts as "the same".
const CLOSE_THRESHOLD: f32 = 0.6;
/// How many past same-act intents the breaker compares against.
const BREAKER_LOOKBACK: usize = 3;
/// History is only ever read `WINDOW` deep; keep a little more than that so a
/// world running for a week does not accumulate a row per act for ever.
const HISTORY_CAP: usize = WINDOW * 2;

/// What a character reads at the head of the turn a forced reflect lands on, so
/// the redirect is *explained* and not just imposed. Delivered as a
/// `<tool_response>` through the ordinary outcome channel — see
/// [`crate::engine::mind::Minds::deliver_outcomes`].
pub const NUDGE: &str =
    "You seem to be repeating yourself; nothing new is happening here. Try something different — \
     do something new, or turn to someone or something else.";

/// The text of an act that the closeness test compares — what the character
/// *meant*, pulled from whichever argument carries the substance.
///
/// The catalog's content arguments by act: `ask` uses `about`, the speech and
/// gesture acts use `intent`, `reflect` its `inner_thoughts`, `move_to` its
/// `destination`, `follow` its `target`. Anything else falls back to every
/// string argument joined, so a new act is compared on *something* rather than
/// silently never looping.
pub fn salient(act: &Act) -> String {
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

/// Token-set Jaccard similarity — cheap, and it catches the near-verbatim and
/// light-paraphrase repeats the loop produces. A tighter embedding similarity is
/// the eventual upgrade; this is enough to detect the loop.
fn similarity(a: &str, b: &str) -> f32 {
    let toks = |s: &str| -> std::collections::HashSet<String> {
        s.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() > 2)
            .map(|t| t.to_string())
            .collect()
    };
    let (sa, sb) = (toks(a), toks(b));
    if sa.is_empty() || sb.is_empty() {
        return 0.0;
    }
    let inter = sa.intersection(&sb).count() as f32;
    let union = sa.union(&sb).count() as f32;
    inter / union
}

/// One character's loop-guard state.
#[derive(Default)]
struct Guard {
    /// The turn counter, advanced once per decode (not once per act).
    turn: usize,
    /// `(act, intent)` in order, bounded to the last [`HISTORY_CAP`].
    history: Vec<(String, String)>,
    /// Act name → the turn it becomes available again.
    cool_until: HashMap<String, usize>,
    /// Set by the breaker, read by [`Guard::cooling`] on the following turn, and
    /// reset by the next [`Guard::record`].
    force_reflect: bool,
    /// Consecutive breaker detections for the currently-looping act, so the
    /// adaptive cooldown grows as `repetitions + 1`.
    loop_reps: usize,
}

impl Guard {
    /// Acts to strike from this turn's grammar.
    fn cooling(&self) -> Vec<String> {
        if self.force_reflect {
            // Everything the catalog offers except the redirect and the one act
            // that is never struck, so the character can only reflect (or walk).
            return CATALOG
                .iter()
                .map(|t| t.name)
                .filter(|n| *n != ALWAYS_REFLECT && *n != NEVER_STRUCK)
                .map(str::to_string)
                .collect();
        }
        self.cool_until
            .iter()
            .filter(|(act, until)| **until > self.turn && act.as_str() != NEVER_STRUCK)
            .map(|(act, _)| act.clone())
            .collect()
    }

    /// Fold this decode's chosen acts in and set up the next turn. Returns
    /// whether the breaker fired, so the caller can deliver the [`NUDGE`].
    fn record(&mut self, acts: &[(String, String)]) -> bool {
        let turn = self.turn;
        self.force_reflect = false;
        let mut fired = false;

        for (act, intent) in acts {
            let act = act.as_str();
            if GUARD_EXEMPT.contains(&act) {
                continue;
            }

            // ── exponential cooldown + protected set (design note 1) ──────────
            if act != NEVER_STRUCK {
                if PROTECTED.contains(&act) {
                    if ANTI_REPEAT.contains(&act) {
                        self.cool_until.insert(act.to_string(), turn + 2);
                    }
                } else {
                    let uses = self
                        .history
                        .iter()
                        .rev()
                        .take(WINDOW)
                        .filter(|(a, _)| a == act)
                        .count()
                        + 1;
                    let cool = (1usize << uses.min(5)).min(COOL_CAP); // 2^uses, capped
                    self.cool_until.insert(act.to_string(), turn + cool);
                }
            }

            // ── closeness circuit-breaker (design note 2) ─────────────────────
            let recent_same: Vec<&String> = self
                .history
                .iter()
                .rev()
                .filter(|(a, _)| a == act)
                .map(|(_, i)| i)
                .take(BREAKER_LOOKBACK)
                .collect();
            let looping = !recent_same.is_empty()
                && recent_same
                    .iter()
                    .all(|prev| similarity(prev, intent) >= CLOSE_THRESHOLD);
            if looping {
                self.loop_reps += 1;
                self.cool_until
                    .insert(act.to_string(), turn + self.loop_reps + 1);
                self.force_reflect = true;
                fired = true;
            } else {
                self.loop_reps = 0;
            }
        }

        for (act, intent) in acts {
            self.history.push((act.clone(), intent.clone()));
        }
        if self.history.len() > HISTORY_CAP {
            let drop = self.history.len() - HISTORY_CAP;
            self.history.drain(0..drop);
        }
        self.turn += 1;
        self.cool_until.retain(|_, until| *until > self.turn);
        fired
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
    /// act from one decode, each paired with its [`salient`] intent, in call
    /// order. Returns whether the breaker fired — the caller then delivers the
    /// [`NUDGE`] so the forced reflect next turn is explained.
    pub fn record(&self, npc_id: u64, acts: &[(String, String)]) -> bool {
        if acts.is_empty() {
            return false;
        }
        self.guards
            .lock()
            .expect("loop-guard lock")
            .entry(npc_id)
            .or_default()
            .record(acts)
    }

    /// Forget a character's history — for one being retired, or one whose body
    /// has left the world.
    pub fn forget(&self, npc_id: u64) {
        self.guards
            .lock()
            .expect("loop-guard lock")
            .remove(&npc_id);
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

    fn took(g: &LoopGuards, npc: u64, tool: &'static str, intent: &str) -> bool {
        g.record(npc, &[(tool.to_string(), intent.to_string())])
    }

    /// The salient text is the act's content argument, whichever one carries it.
    #[test]
    fn the_salient_text_is_the_content_argument() {
        assert_eq!(salient(&act("ask", json!({"to": "Pax", "about": "the lights"}))), "the lights");
        assert_eq!(salient(&act("tell", json!({"to": "Pax", "intent": "come here"}))), "come here");
        assert_eq!(
            salient(&act("move_to", json!({"destination": "the deck"}))),
            "the deck"
        );
    }

    /// **The baseline failure, reproduced then broken.** A non-protected act
    /// repeated verbatim escalates out of the grammar; without the guard it would
    /// simply be offered again.
    #[test]
    fn a_repeated_unprotected_act_is_struck_from_the_grammar() {
        let g = LoopGuards::new();
        // First `ask` is not yet a repeat, so nothing is cooling before it.
        assert!(g.cooling(1).is_empty());
        took(&g, 1, "ask", "did you see the lights flicker");
        // Now `ask` is on an exponential cooldown and cannot be re-emitted.
        assert!(
            g.cooling(1).contains(&"ask".to_string()),
            "a repeated act must be struck"
        );
    }

    /// The protected set stays reachable. `tell` gets only a one-turn anti-repeat;
    /// `move_to` is never struck at all.
    #[test]
    fn the_protected_set_is_not_exponentially_cooled() {
        let g = LoopGuards::new();
        for i in 0..4 {
            took(&g, 1, "move_to", &format!("room {i}"));
            assert!(
                !g.cooling(1).contains(&"move_to".to_string()),
                "move_to must never be struck"
            );
        }
        // tell is anti-repeat: struck the turn right after, back the turn after.
        let g2 = LoopGuards::new();
        took(&g2, 2, "tell", "hello");
        assert!(g2.cooling(2).contains(&"tell".to_string()), "no immediate repeat");
        took(&g2, 2, "reflect", "thinking"); // a turn passes
        assert!(
            !g2.cooling(2).contains(&"tell".to_string()),
            "tell comes back after one turn"
        );
    }

    /// The exponential rule: the more an act repeats in the window, the longer it
    /// is gone. Two uses cool longer than one.
    #[test]
    fn the_cooldown_grows_with_repetition() {
        let cool_for = |uses: usize| (1usize << uses.min(5)).min(COOL_CAP);
        assert!(cool_for(2) > cool_for(1));
        assert!(cool_for(3) > cool_for(2));
        assert_eq!(cool_for(9), COOL_CAP, "capped");
    }

    /// **The circuit-breaker.** Near-verbatim repeats of the same act — an
    /// A-B-A-B paraphrase loop the per-name cooldown could ride — trip the
    /// breaker, which forces a reflect and asks for the nudge.
    #[test]
    fn near_verbatim_repeats_trip_the_breaker() {
        let g = LoopGuards::new();
        // Two very similar `ask`s with a differently-named act between them, so
        // the exponential cooldown on `ask` has expired but the intents match.
        assert!(!took(&g, 1, "ask", "what did you see in the corridor just now"));
        // Breaker compares against prior same-act intents; the second near-match
        // fires it.
        let fired = took(&g, 1, "ask", "what did you see in the corridor just now");
        assert!(fired, "a near-verbatim repeat must trip the breaker");
        // A forced reflect: everything but reflect and move_to is struck.
        let cooling = g.cooling(1);
        assert!(!cooling.contains(&"reflect".to_string()), "reflect stays reachable");
        assert!(!cooling.contains(&"move_to".to_string()), "move_to stays reachable");
        assert!(cooling.contains(&"ask".to_string()), "the looping act is struck");
        assert!(cooling.contains(&"gesture".to_string()), "other acts are struck too");
    }

    /// The forced reflect is single-shot: it clears the turn after, so it cannot
    /// itself become a reflect loop.
    #[test]
    fn the_forced_reflect_is_single_shot() {
        let g = LoopGuards::new();
        took(&g, 1, "ask", "the same thing over and over here");
        assert!(took(&g, 1, "ask", "the same thing over and over here"));
        assert!(g.cooling(1).contains(&"gesture".to_string()), "forced reflect this turn");
        // The character reflects; next turn is free again (bar ordinary cooldowns).
        took(&g, 1, "reflect", "I am going in circles and should try something else");
        assert!(
            !g.cooling(1).contains(&"gesture".to_string()),
            "the forced reflect did not clear"
        );
    }

    /// A combat act opts out of the whole guard — a fight is repetition by
    /// nature, and it is paced by the real-time fight rate instead.
    #[test]
    fn a_combat_act_is_never_struck() {
        let g = LoopGuards::new();
        for _ in 0..5 {
            let fired = took(&g, 1, "act", "strike him again");
            assert!(!fired, "combat must not trip the breaker");
            assert!(
                !g.cooling(1).contains(&"act".to_string()),
                "combat must never be struck"
            );
        }
    }

    /// Distinct acts, or distinct intents, are not a loop and cool nothing beyond
    /// the ordinary exponential step.
    #[test]
    fn varied_action_does_not_trip_the_breaker() {
        let g = LoopGuards::new();
        assert!(!took(&g, 1, "ask", "where are the others"));
        assert!(!took(&g, 1, "gesture", "point at the hatch"));
        assert!(!took(&g, 1, "tell", "we should move now"));
        assert!(!took(&g, 1, "ask", "did anything come through the vents"));
    }

    /// A retired character leaves nothing behind.
    #[test]
    fn forget_clears_a_character() {
        let g = LoopGuards::new();
        took(&g, 9, "ask", "again and again");
        assert!(!g.cooling(9).is_empty());
        g.forget(9);
        assert!(g.cooling(9).is_empty());
    }

    /// One character looping does not strike another's grammar.
    #[test]
    fn one_character_does_not_cool_another() {
        let g = LoopGuards::new();
        took(&g, 1, "ask", "stuck on this");
        assert!(g.cooling(1).contains(&"ask".to_string()));
        assert!(g.cooling(2).is_empty());
    }
}
