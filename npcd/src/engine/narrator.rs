//! Turning a tick's raw events into curated third-person prose, per character.
//!
//! # Why the raw events were not enough
//!
//! A character's perception was built by rendering each [`Event`] straight to
//! prose ([`Event::prose`]) and concatenating them. That worked, but it exposed
//! three faults the character then read every turn (measured live over the cast):
//!
//! - **First-person intent leaked verbatim into third-person events.** A tool
//!   carries *intent*, written in the actor's own voice ("that I am looking at
//!   the chart"). Rendered into what another character reads, that "I" is the
//!   actor, not the reader — so a character read "Pax does it: that I am looking
//!   at the chart… at Ulysses Thorne", a sentence that mixes "I" (Pax) and the
//!   reader's own name.
//! - **The condensed action line read as machinery** — "X does it: … at Y",
//!   "X whispered something to Y".
//! - **No flow.** Three events were three disconnected lines, not a moment.
//!
//! The catalog's whole first rule is *tools carry intent, not output; the
//! narrator renders it in the character's voice* — and there was no narrator.
//! This module is that narrator.
//!
//! # The shape (see also `mind.rs`, which owns the threads)
//!
//! One narrator conversation per character, forked from a single base that is
//! created once and holds only the shared system prompt — so the whole
//! [`SYSTEM`] prefix is prefilled once at boot and every character's narrator
//! thread shares its K/V. Projection is off (a narrator gathers nothing), and
//! the thread carries a short window so it stays cheap and keeps a little
//! narrative continuity from turn to turn.
//!
//! Each tick, the character's events become one user turn — [`build_turn`] —
//! restating who the focal character is ([`YOU`]) and the basic state (`STATE`,
//! with presence clipped when the room is crowded), then the events. The decode
//! of that turn is the prose the character actually reads, in place of the raw
//! event concatenation.
//!
//! # Point of view
//!
//! Second-person-focal: the character whose perception this is written as "you",
//! everyone and everything else in the third person by name. That matches the
//! character's own "You are X" self-frame, so the narration drops straight into
//! its turn. The one line in [`SYSTEM`] marked as the POV switch is all that
//! stands between this and fully-third-person, if that is ever wanted.
//!
//! # Validated before it was built
//!
//! The prompt was tested on real captured event lists (the live looping scene)
//! across a multi-tick thread: person corrected in both directions (a leaked
//! actor "I" became the actor's name; a third-person reference to the focal
//! became "you"), whispers between others left unheard, the mechanical
//! affordance line stripped, and a crowded room clipped to the two who acted
//! plus "several others" with no invented names.

use crate::engine::event::{Addressed, Event, EventKind};

/// The narrator's whole instruction set — the sole shared prefix. Byte-identical
/// for every character so the K/V of this prefill is shared across the cast (see
/// the module note and `mind.rs`). Keep it stable; changing it re-prefixes every
/// narrator thread.
pub const SYSTEM: &str = "You are the Narrator. Your only task is to turn a list of raw world \
events into a short passage of vivid, immediate prose describing a single moment as one \
character lives it. You render what is given; you never invent.\n\
\n\
Each turn gives you three things:\n\
- YOU: the focal character and a short sketch of who they are. This is restated every turn; \
always narrate from this character's vantage and write them as \"you\".\n\
- STATE: where the focal character is and who is near. Named people are present; a trailing \
\"and several others\" (or \"and many others\") means more are present than are named — do not \
name or invent them.\n\
- EVENTS: a numbered list of what just happened. A line may give a speaker's meaning after \
\"meaning:\", or it may be a bare observed line (a movement, or something you saw someone do). \
Narrate both the same way.\n\
\n\
Rules you always follow:\n\
- FAITHFUL. Every event in the list appears in your prose, and nothing that is not in the list \
may. Add no facts, motives, outcomes, dialogue, or sensory detail the events do not contain. Do \
not resolve the situation, advise, predict, or continue past the moment the list covers.\n\
- POINT OF VIEW. Write the focal character (YOU) as \"you\"; write everyone and everything else \
in the third person, by name. Only name people the turn names — never invent a name for \
\"several others\". (To switch to fully third-person, this one line becomes: write every \
character by name, including the focal one.)\n\
- PERSON. An event may quote a character's meaning in the first person (\"that I am...\", \
\"that he needs to...\"). That \"I\" is the ACTOR of that event, never the focal character. \
Re-voice it: the actor becomes their name or he/she/they, and only the focal character is ever \
\"you\". A line that names the focal character in the third person is still about you — render \
it as \"you\". Never let a stray \"I\" that means someone else survive.\n\
- SPEECH. When someone speaks to you, render what they meant in their own voice as reported or \
lightly rendered speech — their meaning, never a verbatim script you invented. When someone \
speaks to another and you only overhear, render it as overheard. A shout to the room is heard \
by everyone.\n\
- WHISPERS. A whisper to you, you hear. A whisper between other people, you see happen but do \
not hear — render only that it happened and to whom; never guess its content.\n\
- ACTIONS AND MOVEMENT. Describe what a body did and toward whom. Arriving and leaving are \
movements.\n\
- AMBIENT. Sounds, smells, the room itself — weave in briefly as atmosphere.\n\
- TENSE AND LENGTH. Present tense, immediate. Brief: one short paragraph, a glance at a moment, \
not a chapter. Match the number of beats to the number of events; never pad.\n\
- NO MECHANICS. Never mention tool names, lists of available actions, the words \"event\" or \
\"narrate\", or any game machinery. If a line is pure instruction rather than something that \
happened, ignore it.\n\
- OUTPUT. Reply with the prose only — no preamble, no quotation marks around the whole thing, \
no headers.\n\
\n\
WORKED EXAMPLE\n\
YOU: Mara Vance — a wary dock-hand.\n\
STATE: on the wharf at dusk. Near you: Corin Vale.\n\
EVENTS:\n\
1. Corin Vale — speaks to you — meaning: \"that the tide is turning and I want the nets in \
before dark\"\n\
2. a gull wheels overhead and cries once\n\
3. Corin Vale — does something, directed at you — meaning: \"that I am handing you the coiled \
rope so you take the near end\"\n\
NARRATION:\n\
Corin catches your eye down the wharf — the tide's turning, he says, and he wants the nets in \
before dark. A gull wheels over and cries once. Then he's beside you, pressing the coiled rope \
into your hands, the near end left for you to take.\n\
\n\
Reply to every turn from here with the narration only.";

/// How many people are named in `STATE` before the rest become "several others".
/// The event lines name whoever actually acted this turn regardless, so the
/// presence header only sets the room's density.
const NAME_UP_TO: usize = 4;

/// The narrator thread's window, in turns. Short on purpose: a narrator needs
/// only a little continuity for flow, and its history is derived prose rather
/// than something to gather deeply.
pub const WINDOW_TURNS: usize = 6;

/// The cap on one narration decode. A glance at a moment is a short paragraph;
/// the cap only stops a runaway.
pub const MAX_TOKENS: usize = 256;

/// One `EVENTS` line for an event, or `None` for an event that is not narratable
/// (a situation — it becomes `STATE` — or a quiet heartbeat).
///
/// Speech is rendered structured, actor-relative, with the meaning quoted so the
/// narrator can re-voice it. Everything else is passed as a bare observed line;
/// the condensed action/movement text and the leaked first-person intent inside
/// it are exactly what the narrator is there to clean up.
fn event_line(e: &Event) -> Option<String> {
    match &e.kind {
        EventKind::Speech { speaker, text, to } => {
            let rel = match to {
                Addressed::You => "speaks to you".to_string(),
                Addressed::Whispered => "whispers to you".to_string(),
                Addressed::Room => "speaks to the room".to_string(),
                Addressed::Other { who } => format!("speaks to {who} (you overhear)"),
            };
            Some(format!("{speaker} — {rel} — meaning: \"{}\"", text.trim()))
        }
        EventKind::Message { thread, from, text } => {
            let on = if thread == from {
                String::new()
            } else {
                format!(" (on {thread})")
            };
            Some(format!(
                "{from} — messages you{on} — meaning: \"{}\"",
                text.trim()
            ))
        }
        EventKind::Entity {
            entity_id,
            observation,
        } => Some(format!("you notice {entity_id}: {}", observation.trim())),
        // Already condensed to prose by the world (an action, a movement, a
        // whisper seen but not heard). Passed bare for the narrator to re-voice.
        EventKind::Description { text } => {
            let t = text.trim();
            (!t.is_empty()).then(|| t.to_string())
        }
        // Authored, addressed to the focal character already — pass through.
        EventKind::Nudge { text } | EventKind::Operator { text } => {
            let t = text.trim();
            (!t.is_empty()).then(|| t.to_string())
        }
        EventKind::Wake { day } => Some(format!("a new day begins — day {day}")),
        EventKind::Sleep { .. } => Some("the day is ending; you are letting it settle".to_string()),
        // The situation becomes STATE, and a heartbeat is the absence of events.
        EventKind::Situation { .. } | EventKind::Heartbeat => None,
    }
}

/// Where the focal character is, from the most recent situation this tick — its
/// first line, which is the location clause ("You are at …"), before the
/// presence and the mechanical affordance list that follow it.
fn location(events: &[Event]) -> Option<String> {
    events.iter().rev().find_map(|e| match &e.kind {
        EventKind::Situation { text } => text
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .map(str::to_string),
        _ => None,
    })
}

/// Who is near, clipped so a crowded room does not bloat the turn or the prose.
/// Up to [`NAME_UP_TO`] are named; beyond that the first few are named and the
/// rest summarised, because the event lines carry the names that matter.
fn presence(company: &[String]) -> String {
    match company.len() {
        0 => "no one else is here".to_string(),
        n if n <= NAME_UP_TO => company.join(", "),
        n => format!("{}, and {} others", company[..3].join(", "), n - 3),
    }
}

/// Build the per-tick narrator user turn, or `None` when there is nothing worth
/// narrating (a quiet tick, or only a situation) — the caller then skips the
/// narrator decode entirely.
///
/// `sketch` is a compact one-line character sketch (the caller builds it from
/// the persona); `company` is everyone present, by the names the focal character
/// knows them by.
pub fn build_turn(name: &str, sketch: &str, company: &[String], events: &[Event]) -> Option<String> {
    let lines: Vec<String> = events.iter().filter_map(event_line).collect();
    if lines.is_empty() {
        return None;
    }

    let name = if name.trim().is_empty() {
        "the character"
    } else {
        name.trim()
    };
    let mut s = String::with_capacity(256 + lines.iter().map(|l| l.len() + 4).sum::<usize>());
    s.push_str("YOU: ");
    s.push_str(name);
    if !sketch.trim().is_empty() {
        s.push_str(" — ");
        s.push_str(sketch.trim());
    }
    s.push('\n');

    s.push_str("STATE: ");
    if let Some(loc) = location(events) {
        s.push_str(&loc);
        if !loc.ends_with(['.', '!', '?']) {
            s.push('.');
        }
        s.push(' ');
    }
    s.push_str("Near you: ");
    s.push_str(&presence(company));
    s.push_str(".\n");

    s.push_str("EVENTS:\n");
    for (i, line) in lines.iter().enumerate() {
        s.push_str(&format!("{}. {line}\n", i + 1));
    }
    Some(s)
}

/// A primed example exchange, submitted into a fresh narrator thread at fork
/// time so its very first real decode has a precedent turn to follow instead of
/// echoing its own input. The input is in the exact `YOU`/`STATE`/`EVENTS` shape
/// [`build_turn`] produces, and the narration is the reply the model should have
/// given — the same worked example [`SYSTEM`] teaches, made concrete as a turn.
pub const PRIME_INPUT: &str = "YOU: Mara Vance — a wary dock-hand.\n\
STATE: on the wharf at dusk. Near you: Corin Vale.\n\
EVENTS:\n\
1. Corin Vale — speaks to you — meaning: \"that the tide is turning and I want the nets in \
before dark\"\n\
2. a gull wheels overhead and cries once";

/// The narration paired with [`PRIME_INPUT`].
pub const PRIME_NARRATION: &str = "Corin catches your eye down the wharf — the tide's turning, \
he says, and he wants the nets in before dark. A gull wheels over and cries once.";

/// A second primed exchange, teaching the *own-act* shape — an EVENTS line whose
/// actor is the focal character ([`build_act_turn`]). Without a precedent for it,
/// the model narrated the focal character in the third person by name (it only
/// ever sees other characters as actors); this shows the actor-is-you case
/// rendered as "you".
pub const PRIME_ACT_INPUT: &str = "YOU: Mara Vance — a wary dock-hand.\n\
STATE: Near you: Corin Vale.\n\
EVENTS:\n\
1. you — tell Corin Vale — meaning: \"that the far net is fouled and I want it hauled before the \
swell turns\"";

/// The narration paired with [`PRIME_ACT_INPUT`] — the focal character as "you".
pub const PRIME_ACT_NARRATION: &str = "You turn to Corin and tell him the far net is fouled — \
you want it hauled in before the swell turns.";

/// Build the turn that narrates the focal character's *own* act, for the tool
/// response it reads back (and the feed the GUI shows) in place of a bare
/// confirmation. Same shape as [`build_turn`], with a single event whose actor
/// is the character itself — the [`SYSTEM`] person rule renders that as "you".
///
/// `act_line` is the one event line, e.g. `you — tell Pax Veridian — meaning: "…"`.
pub fn build_act_turn(name: &str, sketch: &str, near: &[String], act_line: &str) -> String {
    let name = if name.trim().is_empty() {
        "the character"
    } else {
        name.trim()
    };
    let mut s = String::with_capacity(128 + act_line.len());
    s.push_str("YOU: ");
    s.push_str(name);
    if !sketch.trim().is_empty() {
        s.push_str(" — ");
        s.push_str(sketch.trim());
    }
    s.push('\n');
    s.push_str("STATE: Near you: ");
    s.push_str(&presence(near));
    s.push_str(".\nEVENTS:\n1. ");
    s.push_str(act_line);
    s.push('\n');
    s
}

/// Strip a leading reasoning block from a narration decode.
///
/// The decode is prefilled with the dialect's closed no-think block to stop the
/// model reasoning, and the response carries that prefill back at its head — so
/// this removes it. It also catches the two failures seen without the prefill: a
/// self-closed empty `<think></think>` the model emits on its own, and a runaway
/// block that never closes (its reasoning would otherwise become the character's
/// perception) — the latter is dropped whole.
pub fn strip_reasoning(text: &str) -> String {
    let mut t = text.trim_start();
    // Peel *every* leading reasoning block, not just one. The decode is prefilled
    // with a closed no-think block, and the model sometimes emits further blocks
    // after it (a second empty one, or one it fills with task reasoning); a single
    // strip left the second block in the character's perception. Each block may
    // close with `</think>` or the `/thought` variant this checkpoint also emits.
    loop {
        let Some(rest) = t.strip_prefix("<think>") else {
            break;
        };
        let close = ["</think>", "/thought"]
            .iter()
            .filter_map(|m| rest.find(m).map(|i| (i, m.len())))
            .min_by_key(|(i, _)| *i);
        match close {
            Some((i, len)) => t = rest[i + len..].trim_start(),
            // Opened and never closed — the runaway. Nothing after it is
            // narration we can trust.
            None => return String::new(),
        }
    }
    // A model that fenced its prose as a markdown block — peel the fence.
    let mut out = t.trim();
    if let Some(rest) = out.strip_prefix("```") {
        out = rest
            .splitn(2, '\n')
            .nth(1)
            .unwrap_or("")
            .trim()
            .trim_end_matches('`')
            .trim();
    }
    out.to_string()
}

/// A compact one-line character sketch for the `YOU:` header, from the immutable
/// core and manner. Clipped, because the header is restated every turn and the
/// full identity is already carried by the character's own action conversation —
/// the narrator only needs enough to keep the voice and the vantage right.
pub fn sketch(identity: &str, manner: &str) -> String {
    let clip = |text: &str, max: usize| -> String {
        let t = text.trim().replace('\n', " ");
        if t.chars().count() <= max {
            return t;
        }
        // Cut at the last sentence end within the budget, else at the budget.
        let head: String = t.chars().take(max).collect();
        match head.rfind(['.', '!', '?']) {
            Some(i) => head[..=i].to_string(),
            None => format!("{}…", head.trim_end()),
        }
    };
    let identity = clip(identity, 200);
    let manner = clip(manner, 120);
    match (identity.is_empty(), manner.is_empty()) {
        (true, true) => String::new(),
        (false, true) => identity,
        (true, false) => manner,
        (false, false) => format!("{identity} {manner}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::event::{EventKind, Salience};

    fn ev(kind: EventKind) -> Event {
        Event::new(0, 0, Salience::NORMAL, kind)
    }

    fn speech(speaker: &str, text: &str, to: Addressed) -> Event {
        ev(EventKind::Speech {
            speaker: speaker.into(),
            text: text.into(),
            to,
        })
    }

    /// Speech is rendered actor-relative with the meaning quoted, so the narrator
    /// can re-voice it — and the three readings stay distinct.
    #[test]
    fn speech_lines_are_actor_relative() {
        let to_you = event_line(&speech("Vael", "that the shaft is failing", Addressed::You)).unwrap();
        let overheard = event_line(&speech(
            "Vael",
            "that Pax is stalling",
            Addressed::Other { who: "Pax".into() },
        ))
        .unwrap();
        let whisper = event_line(&speech("Vael", "a secret", Addressed::Whispered)).unwrap();
        assert!(to_you.contains("Vael — speaks to you — meaning:"));
        assert!(overheard.contains("speaks to Pax (you overhear)"));
        assert!(whisper.contains("whispers to you"));
    }

    /// A condensed action/movement line is passed bare — it is the leaked
    /// first-person intent the narrator exists to fix.
    #[test]
    fn a_description_is_passed_bare() {
        let line = event_line(&ev(EventKind::Description {
            text: "Pax Veridian does it: that I am looking at the chart, at Ulysses Thorne".into(),
        }))
        .unwrap();
        assert_eq!(
            line,
            "Pax Veridian does it: that I am looking at the chart, at Ulysses Thorne"
        );
    }

    /// A situation is not an event line; it becomes the STATE location instead.
    #[test]
    fn a_situation_is_not_an_event_line() {
        let e = ev(EventKind::Situation {
            text: "You are at the lift and the stair.\n\nPax is here.\n\nBecause Pax is here you \
                   can also use: tell, ask."
                .into(),
        });
        assert!(event_line(&e).is_none());
        assert_eq!(
            location(std::slice::from_ref(&e)).as_deref(),
            Some("You are at the lift and the stair."),
            "location is the situation's first line, before presence and mechanics"
        );
    }

    /// A quiet tick (a heartbeat, or nothing but a situation) produces no turn,
    /// so the caller skips the narrator decode entirely.
    #[test]
    fn a_quiet_tick_builds_no_turn() {
        assert!(build_turn("Ulysses", "sharp", &[], &[ev(EventKind::Heartbeat)]).is_none());
        assert!(build_turn(
            "Ulysses",
            "sharp",
            &["Pax".into()],
            &[ev(EventKind::Situation { text: "You are here.".into() })]
        )
        .is_none());
    }

    /// Presence is clipped when the room is crowded; the event lines still carry
    /// the actors' names.
    #[test]
    fn presence_is_clipped_when_crowded() {
        assert_eq!(presence(&[]), "no one else is here");
        assert_eq!(presence(&["Pax".into(), "Vael".into()]), "Pax, Vael");
        let crowd: Vec<String> = (0..7).map(|i| format!("P{i}")).collect();
        let clipped = presence(&crowd);
        assert!(clipped.contains("and 4 others"), "{clipped}");
        assert!(clipped.starts_with("P0, P1, P2,"), "{clipped}");
    }

    /// The full turn has YOU / STATE / EVENTS in order, with the location folded
    /// into STATE and each event numbered.
    #[test]
    fn a_turn_is_you_state_events() {
        let events = vec![
            ev(EventKind::Situation {
                text: "You are at the lift and the stair of the command level.\n\nPax is here."
                    .into(),
            }),
            speech("Vael", "that the shaft is failing", Addressed::You),
            ev(EventKind::Description {
                text: "Vael Fane left".into(),
            }),
        ];
        let turn = build_turn("Ulysses Thorne", "sharp, impatient", &["Pax".into(), "Vael".into()], &events)
            .unwrap();
        assert!(turn.starts_with("YOU: Ulysses Thorne — sharp, impatient\n"));
        assert!(turn.contains("STATE: You are at the lift and the stair of the command level. Near you: Pax, Vael.\n"));
        assert!(turn.contains("1. Vael — speaks to you — meaning: \"that the shaft is failing\"\n"));
        assert!(turn.contains("2. Vael Fane left\n"));
        assert!(!turn.contains("Because Pax"), "the mechanical affordance line must not appear");
    }

    /// The no-think prefill rides back at the head of the decode, and a stray
    /// or runaway reasoning block never reaches the character.
    #[test]
    fn reasoning_is_stripped_from_a_narration() {
        // The prefilled closed block, then the prose.
        assert_eq!(
            strip_reasoning("<think>\n\n</think>\n\nVael turns to you."),
            "Vael turns to you."
        );
        // A block the model filled with task reasoning — dropped, prose kept.
        assert_eq!(
            strip_reasoning("<think>The user wants me to narrate…</think> The door opens."),
            "The door opens."
        );
        // Two blocks: the prefill's, then a second the model emitted after it.
        // A single strip left the second in the character's perception.
        assert_eq!(
            strip_reasoning("<think>\n\n</think>\n\n<think>\n\n</think>\n\nThe room is quiet."),
            "The room is quiet."
        );
        // The `/thought` close variant this checkpoint also emits.
        assert_eq!(
            strip_reasoning("<think>\n\n/thought\n\nYou move toward the door."),
            "You move toward the door."
        );
        // A block after the prefill that is itself unclosed — drop from it.
        assert_eq!(strip_reasoning("<think>\n\n</think>\n\n<think>still going"), "");
        // Prose fenced as markdown — peel the fence.
        assert_eq!(
            strip_reasoning("```markdown\nA gull cries once.\n```"),
            "A gull cries once."
        );
        // A runaway that never closed is nothing but reasoning — dropped whole.
        assert_eq!(strip_reasoning("<think>still going and going"), "");
        // Plain prose is untouched.
        assert_eq!(strip_reasoning("  A gull cries once.  "), "A gull cries once.");
    }

    /// The sketch is clipped to a sentence-bounded budget and pairs identity with
    /// manner.
    #[test]
    fn the_sketch_is_compact() {
        let s = sketch("An archivist aboard the vault. He was born under the third dome.", "curt");
        assert!(s.starts_with("An archivist aboard the vault."), "{s}");
        assert!(s.ends_with("curt"), "{s}");
        let long = "word ".repeat(100);
        assert!(sketch(&long, "").chars().count() <= 202, "identity is clipped");
    }
}
