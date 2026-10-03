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
//! # The shape (see also `narration`, which renders the turn to prose)
//!
//! Each tick, the character's events become one structured turn — [`build_turn`]
//! — restating who the focal character is (`YOU`) and the basic state (`STATE`,
//! with presence clipped when the room is crowded), then the numbered events.
//! [`crate::engine::narration::render`] then turns that turn into the prose the
//! character actually reads — deterministically, in place of the raw event
//! concatenation. This module composes the turn; `narration` renders it.
//!
//! # Point of view
//!
//! Second-person-focal: the character whose perception this is written as "you",
//! everyone and everything else in the third person by name. That matches the
//! character's own "You are X" self-frame, so the narration drops straight into
//! its turn. Every event line the world builds already carries the focal
//! character as the literal token "you", which is what the renderer keys off.

use crate::clock::{WorldTime, DAY_MS};
use crate::engine::event::{mind_control_line, Addressed, Event, EventKind};

/// How many people are named in `STATE` before the rest become "several others".
/// The event lines name whoever actually acted this turn regardless, so the
/// presence header only sets the room's density.
const NAME_UP_TO: usize = 4;

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
        // A word put to the whole world — framed as reaching everyone so the
        // narrator re-voices it as an announcement, not a thing in the room.
        EventKind::Announcement { text } => {
            let t = text.trim();
            (!t.is_empty()).then(|| format!("word reaches everyone across the world: {t}"))
        }
        // Authored, addressed to the focal character already — pass through.
        EventKind::Nudge { text } | EventKind::Operator { text } => {
            let t = text.trim();
            (!t.is_empty()).then(|| t.to_string())
        }
        // Its own thought, not a happening: handed over as the character's
        // inner line so the narrator keeps it in the first person.
        EventKind::MindControl { text } => {
            let t = text.trim();
            (!t.is_empty()).then(|| mind_control_line(t))
        }
        EventKind::Wake { day } => Some(format!(
            "a new day begins — {}",
            WorldTime::of(day * DAY_MS).date()
        )),
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
pub fn build_turn(
    name: &str,
    sketch: &str,
    company: &[String],
    events: &[Event],
) -> Option<String> {
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

/// Build the turn that narrates the focal character's *own* act, for the tool
/// response it reads back (and the feed the GUI shows) in place of a bare
/// confirmation. Same shape as [`build_turn`], with a single event whose actor
/// is the character itself — the literal token "you", which
/// [`crate::engine::narration::render`] renders as the focal character.
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
        let to_you =
            event_line(&speech("Vael", "that the shaft is failing", Addressed::You)).unwrap();
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
            text: "You are at the lift.\n\nPax is here.\n\nBecause Pax is here you \
                   can also use: tell, ask."
                .into(),
        });
        assert!(event_line(&e).is_none());
        assert_eq!(
            location(std::slice::from_ref(&e)).as_deref(),
            Some("You are at the lift."),
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
            &[ev(EventKind::Situation {
                text: "You are here.".into()
            })]
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
                text: "You are at the lift of the command level.\n\nPax is here.".into(),
            }),
            speech("Vael", "that the shaft is failing", Addressed::You),
            ev(EventKind::Description {
                text: "Vael Fane left".into(),
            }),
        ];
        let turn = build_turn(
            "Ulysses Thorne",
            "sharp, impatient",
            &["Pax".into(), "Vael".into()],
            &events,
        )
        .unwrap();
        assert!(turn.starts_with("YOU: Ulysses Thorne — sharp, impatient\n"));
        assert!(turn
            .contains("STATE: You are at the lift of the command level. Near you: Pax, Vael.\n"));
        assert!(turn.contains("1. Vael — speaks to you — meaning: \"that the shaft is failing\"\n"));
        assert!(turn.contains("2. Vael Fane left\n"));
        assert!(
            !turn.contains("Because Pax"),
            "the mechanical affordance line must not appear"
        );
    }

    /// The sketch is clipped to a sentence-bounded budget and pairs identity with
    /// manner.
    #[test]
    fn the_sketch_is_compact() {
        let s = sketch(
            "An archivist aboard the vault. He was born under the third dome.",
            "curt",
        );
        assert!(s.starts_with("An archivist aboard the vault."), "{s}");
        assert!(s.ends_with("curt"), "{s}");
        let long = "word ".repeat(100);
        assert!(
            sketch(&long, "").chars().count() <= 202,
            "identity is clipped"
        );
    }
}
