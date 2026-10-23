//! A window turn as a journal stretch holds it: what a claim citing it would rest
//! on, read off the turn's [`Origin`] and never off its wording.

use journal_core::entry::Kind;
use journal_core::span::SpanTurn;

use crate::engine::body;
use crate::engine::window::{Origin, Speaker, Turn};

/// How one turn supports a claim that cites it, or `None` when no claim may rest
/// on it.
pub fn evidence(origin: &Origin) -> Option<Kind> {
    if !origin.citable() {
        return None;
    }
    Some(match origin {
        Origin::Event { tag, .. } => match *tag {
            "speech" | "message" | "announcement" | "operator" => Kind::Heard,
            _ => Kind::Observed,
        },
        // A tool whose line is something the world told the character, rather
        // than the character's own act being recorded.
        Origin::Act { verb } if body::answers(verb) => Kind::Observed,
        Origin::Act { .. } => Kind::Did,
    })
}

/// `turn` as the draft is shown it: the character's own lines marked as its own.
pub fn span_turn(turn: &Turn) -> SpanTurn {
    let who = match turn.speaker {
        Speaker::Npc => "You: ",
        Speaker::World => "",
    };
    SpanTurn {
        id: turn.id,
        at_ms: turn.at_ms,
        evidence: evidence(&turn.origin),
        text: format!("{who}{}", turn.text.trim()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::event::{Event, EventKind, Salience};
    use crate::engine::window::Window;

    fn turns(w: &Window) -> Vec<SpanTurn> {
        w.turns().map(span_turn).collect()
    }

    #[test]
    fn what_was_said_is_heard_and_what_was_done_is_did() {
        let mut w = Window::with_default_cap();
        w.push_event(&Event::new(
            0,
            1,
            Salience::NORMAL,
            EventKind::Speech {
                speaker: "Pax".into(),
                text: "Fabricator 5 is lost.".into(),
                to: Default::default(),
            },
        ));
        w.push_npc("tell — to Pax: on my way", 2);
        let t = turns(&w);
        assert_eq!(t[0].evidence, Some(Kind::Heard));
        assert_eq!(t[1].evidence, Some(Kind::Did));
        assert_eq!(t[1].text, "You: tell — to Pax: on my way");
        assert!(!t[0].text.starts_with("You: "));
    }

    #[test]
    fn scenery_and_reflection_may_not_be_cited() {
        let mut w = Window::with_default_cap();
        w.push_event(&Event::new(
            0,
            1,
            Salience::IDLE,
            EventKind::Description {
                text: "The conduit hums.".into(),
            },
        ));
        w.push_npc("reflect — thinking it over", 2);
        assert!(turns(&w).iter().all(|t| t.evidence.is_none()));
    }

    #[test]
    fn a_description_that_is_not_ambient_was_observed() {
        let mut w = Window::with_default_cap();
        w.push_event(&Event::new(
            0,
            1,
            Salience::NORMAL,
            EventKind::Description {
                text: "The east door slams shut.".into(),
            },
        ));
        assert_eq!(turns(&w)[0].evidence, Some(Kind::Observed));
    }
}
