//! Turning a character's authored record into the person the model reads.
//!
//! # Why the numbers become words
//!
//! An authored relationship carries `trust: 0.2, affect: -0.6, familiarity: 0.9`.
//! Handing those to a language model as numbers is asking it to invent a scale —
//! it has no idea whether 0.2 is low, and neither does anyone else without the
//! surrounding distribution. The same three numbers rendered as *"you have known
//! him a long time, you do not trust him, and you dislike him"* say exactly what
//! the author meant, in the register the character thinks in.
//!
//! That conversion is this module. It is the one place a dial's meaning is
//! decided, which is what stops two parts of the prompt describing the same
//! number differently.
//!
//! # Confidence is not rendered as a number either
//!
//! A belief's `confidence` decides how it is *stated*, not a figure printed
//! beside it. A character does not think "I believe with 0.9 confidence that
//! Hess burned the granary" — it thinks "Hess burned the granary", or "I am
//! fairly sure Hess burned the granary", and the difference in phrasing is the
//! confidence.

use candle_conversation::persistence::record::{
    AuthoredBelief, AuthoredRelationship, AuthoredStrategy, NpcPayload,
};

use crate::engine::runtime::OwnedPersona;
use crate::engine::tools::Mode;

/// How a belief reads at a given confidence.
///
/// The bands are wide on purpose. A model cannot act on the difference between
/// 0.71 and 0.74, and pretending it can produces prompts that differ in wording
/// for no reason a reader could name.
fn believe(b: &AuthoredBelief) -> String {
    let s = b.statement.trim();
    match b.confidence {
        c if c >= 0.9 => s.to_string(),
        c if c >= 0.7 => format!("{s} — you are sure of this"),
        c if c >= 0.4 => format!("{s} — you think so, though you have wondered"),
        _ => format!("{s} — you half believe it, no more"),
    }
}

/// How a relationship reads.
///
/// Familiarity first, because it sets whether the other two are even meaningful:
/// distrusting a stranger is a posture, distrusting someone you have known for
/// years is a history.
fn know(r: &AuthoredRelationship) -> String {
    let who = if r.display.trim().is_empty() {
        r.entity_id.trim()
    } else {
        r.display.trim()
    };
    // Familiarity, trust, affect — in that order, and the order is the point.
    let clauses = [
        match r.familiarity {
            f if f >= 0.8 => "you have known them a long time",
            f if f >= 0.4 => "you know them",
            f if f >= 0.1 => "you have met",
            _ => "they are a stranger to you",
        },
        match r.trust {
            t if t >= 0.6 => "you trust them",
            t if t >= 0.2 => "you would take their word on most things",
            t if t > -0.2 => "you have no particular reason to trust or doubt them",
            t if t > -0.6 => "you do not quite trust them",
            _ => "you do not trust them at all",
        },
        match r.affect {
            a if a >= 0.6 => "you are fond of them",
            a if a >= 0.2 => "you like them well enough",
            a if a > -0.2 => "you feel little either way",
            a if a > -0.6 => "they grate on you",
            _ => "you cannot stand them",
        },
    ];

    let mut s = format!("{who}: {}", clauses.join(", "));
    let notes = r.notes.trim();
    if !notes.is_empty() {
        // The author's own words go last and unaltered. Whatever they wrote is
        // more specific than any of the bands above, and it should read as the
        // thing that qualifies them.
        s.push_str(". ");
        s.push_str(notes);
    }
    s
}

/// The character's current intent: the first active strategy, if any.
///
/// First rather than a list, because an intent is what the character is *set
/// on*, and a character set on four things at once is set on nothing. The rest
/// of the tree stays in the substrate and the gather can reach it.
fn intent(agency: &[AuthoredStrategy]) -> Option<String> {
    agency
        .iter()
        .find(|s| s.state == "active" && !s.statement.trim().is_empty())
        .map(|s| s.statement.trim().to_string())
}

/// Build the prompt-facing persona from an authored record.
///
/// `world` is the world's own description and `anchor` is the personality's,
/// both resolved by the caller — this module does not reach into the registry,
/// so it stays testable without one.
///
/// Pass an empty `anchor` where the anchor is delivered some other way. The one
/// caller that does is the render of the `WHO` collection member, because there
/// the anchor is its own collection and printing it into the character block as
/// well would put it in the prompt twice.
pub fn of(n: &NpcPayload, world: &str, anchor: &str) -> OwnedPersona {
    OwnedPersona {
        name: n.name.clone(),
        identity: n.persona_description.trim().to_string(),
        anchor: anchor.trim().to_string(),
        // The authored record has no separate manner field yet; the personality
        // template supplies it through `persona_description`. Left empty rather
        // than duplicating the description into both slots, which would print
        // the same paragraph twice under two headings.
        // Slugs, not prose: what a turn pins its personality's anchor and its
        // world's setting to in the projection.
        personality: n.personality_id.clone(),
        world_id: n.world_id.clone(),
        manner: String::new(),
        beliefs: n.beliefs.iter().map(believe).collect(),
        relationships: n.relationships.iter().map(know).collect(),
        intent: intent(&n.agency),
        situation: String::new(),
        world: world.trim().to_string(),
        // Physical until an interaction says otherwise. The restrictive default:
        // a character on the ambient tick is standing somewhere, not messaging.
        // Filled in by the runtime, which is the only thing that knows whether
        // this character has a body and what building it stands in. An
        // authored record carries no map.
        place: String::new(),
        building: String::new(),
        mode: Mode::Physical,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn belief(statement: &str, confidence: f32) -> AuthoredBelief {
        AuthoredBelief {
            belief_id: "b1".into(),
            statement: statement.into(),
            confidence,
            threshold: 0.5,
        }
    }

    /// A minimal record. `NpcPayload` has no `Default` — deliberately, since
    /// every field on it is a decision somebody made — so a test that needs one
    /// spells it out.
    fn payload(name: &str, description: &str) -> NpcPayload {
        NpcPayload {
            npc_id: 1,
            owner_id: "acct".into(),
            revision: 1,
            created_ms: 0,
            updated_ms: 0,
            state: "idle".into(),
            name: name.into(),
            world_id: "w".into(),
            personality_id: "p".into(),
            hidden: false,
            heartbeat_ms: 120_000,
            salience_gate: 0.5,
            tags: Vec::new(),
            persona_description: description.into(),
            persona_origin: "authored".into(),
            portrait_image_id: None,
            portrait_origin: None,
            at: None,
            mood: None,
            beliefs: Vec::new(),
            relationships: Vec::new(),
            agency: Vec::new(),
            modulation: Default::default(),
        }
    }

    fn rel(trust: f32, affect: f32, familiarity: f32) -> AuthoredRelationship {
        AuthoredRelationship {
            entity_id: "hess".into(),
            display: "Hess".into(),
            trust,
            affect,
            familiarity,
            notes: String::new(),
        }
    }

    /// A near-certain belief is stated flatly. Hedging a belief the author marked
    /// certain is the single most common way a character reads as wishy-washy.
    #[test]
    fn a_certain_belief_is_stated_without_hedging() {
        assert_eq!(
            believe(&belief("Hess burned the east granary.", 0.95)),
            "Hess burned the east granary."
        );
    }

    #[test]
    fn a_weak_belief_is_hedged_in_words_not_numbers() {
        let s = believe(&belief("Hess burned the granary.", 0.2));
        assert!(s.contains("half believe"));
        // The number itself must never reach the prompt — a model has no scale
        // for it and will invent one.
        for n in ["0.2", "0.20", "confidence"] {
            assert!(!s.contains(n), "leaked the raw dial: {s}");
        }
    }

    /// Every band must produce different words, or the dial does nothing.
    #[test]
    fn each_confidence_band_reads_differently() {
        let mut seen: Vec<String> = Vec::new();
        for c in [0.95, 0.75, 0.5, 0.1] {
            let s = believe(&belief("X.", c));
            assert!(
                !seen.contains(&s),
                "two confidence bands read identically: {s}"
            );
            seen.push(s);
        }
    }

    /// Familiarity leads, because it decides whether the other dials describe a
    /// posture or a history.
    #[test]
    fn a_relationship_leads_with_how_well_they_are_known() {
        let s = know(&rel(-0.8, -0.7, 0.9));
        assert!(s.starts_with("Hess: you have known them a long time"));
        assert!(s.contains("do not trust them at all"));
        assert!(s.contains("cannot stand them"));
    }

    #[test]
    fn a_stranger_reads_as_a_stranger() {
        let s = know(&rel(0.0, 0.0, 0.0));
        assert!(s.contains("stranger to you"));
        assert!(s.contains("no particular reason"));
    }

    /// The author's own note is more specific than any band and goes last,
    /// unaltered.
    #[test]
    fn an_authored_note_survives_verbatim_and_goes_last() {
        let mut r = rel(0.5, 0.5, 0.5);
        r.notes = "She pulled you out of the river in the spring.".into();
        let s = know(&r);
        assert!(s.ends_with("She pulled you out of the river in the spring."));
    }

    /// No raw dial may reach the prompt, on any path.
    #[test]
    fn no_relationship_dial_reaches_the_prompt_as_a_number() {
        for (t, a, f) in [(0.7, -0.9, 0.95), (-0.3, 0.1, 0.2), (0.0, 0.0, 0.0)] {
            let s = know(&rel(t, a, f));
            for leak in ["0.", "trust:", "affect:", "familiarity:"] {
                assert!(!s.contains(leak), "leaked {leak:?}: {s}");
            }
        }
    }

    /// A character set on four things at once is set on nothing.
    #[test]
    fn only_the_first_active_strategy_becomes_the_intent() {
        let agency = vec![
            AuthoredStrategy {
                strategy_id: "s0".into(),
                statement: "Keep the granary running.".into(),
                parent_id: None,
                state: "finished".into(),
            },
            AuthoredStrategy {
                strategy_id: "s1".into(),
                statement: "Get the ledger out of the district.".into(),
                parent_id: None,
                state: "active".into(),
            },
            AuthoredStrategy {
                strategy_id: "s2".into(),
                statement: "Find out who talked.".into(),
                parent_id: None,
                state: "active".into(),
            },
        ];
        assert_eq!(
            intent(&agency).as_deref(),
            Some("Get the ledger out of the district.")
        );
    }

    #[test]
    fn a_character_with_nothing_active_has_no_intent() {
        assert_eq!(intent(&[]), None);
    }

    /// The ambient tick has a character standing somewhere, not messaging — and
    /// the restrictive mode is the safe default.
    #[test]
    fn a_ticking_character_is_physically_present_by_default() {
        let p = of(&payload("Vasska", ""), "A besieged city.", "");
        assert_eq!(p.mode, Mode::Physical);
        assert_eq!(p.name, "Vasska");
        assert_eq!(p.world, "A besieged city.");
    }

    /// The description must not be printed twice under two headings.
    #[test]
    fn the_description_does_not_appear_as_both_identity_and_manner() {
        let p = of(
            &payload("V", "A quartermaster who has outlived two garrisons."),
            "",
            "",
        );
        assert_eq!(
            p.identity,
            "A quartermaster who has outlived two garrisons."
        );
        assert!(p.manner.is_empty());
    }
}
