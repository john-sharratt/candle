//! The system prompt — the lens a character thinks through.
//!
//! # Why this is assembled rather than written
//!
//! A character's prompt is not a document somebody authors. It is composed from
//! the character's own layers — who they are, what they believe, who they know,
//! what they are set on — because every one of those is editable through the
//! authoring API and a prompt that did not reflect an edit would make the
//! editor a lie.
//!
//! # What it must establish, in order
//!
//! 1. **That the character is a person, not an assistant.** The single largest
//!    failure mode is a model that answers questions helpfully instead of acting
//!    like someone with their own reasons. Everything in the opening section
//!    exists to prevent that.
//! 2. **That acts go through tools.** A character that narrates what it does
//!    instead of calling a tool has produced fiction, not an act — the world
//!    never sees it, and the transcript now contains something that did not
//!    happen.
//! 3. **That beliefs are not negotiable by decision.** The model will otherwise
//!    resolve tension the moment it notices it, because resolving tension is
//!    what a helpful assistant does. Beliefs move on evidence over time.
//! 4. **That doing nothing is a legitimate act.** Without this a character
//!    invents activity to fill every step, which is how an NPC starts pacing.
//!
//! # Agentic, in the specific sense that matters here
//!
//! The character is handed a situation and a vocabulary, and chooses. It is not
//! handed a task. There is no completion criterion and no "you are done when" —
//! the loop is continuous, and a prompt that implies a terminal state teaches
//! the model to try to reach one.

use crate::engine::tools::{Mode, Tool};

/// What a character needs to know about itself to think as itself.
///
/// Borrowed rather than owned: this is built per tick from the character's
/// current layers, and copying a personality's prose on every heartbeat across a
/// hundred characters is a cost with no purpose.
#[derive(Debug, Default, Clone, Copy)]
pub struct Persona<'a> {
    pub name: &'a str,
    /// The immutable core — who this is. From the personality template.
    pub identity: &'a str,
    /// How they speak and carry themselves.
    pub manner: &'a str,
    /// What they hold true. Read-only to the character, by construction.
    pub beliefs: &'a [String],
    /// Who they know and how they stand with them.
    pub relationships: &'a [String],
    /// What they are currently set on, if anything.
    pub intent: Option<&'a str>,
    /// Where they are and what is going on, in the world's own words.
    pub situation: &'a str,
    /// The world's setting, tone and rules.
    pub world: &'a str,
}

/// Assemble the system prompt for one character in one mode.
pub fn build(p: &Persona<'_>, mode: Mode, tools: &[&Tool]) -> String {
    let mut s = String::with_capacity(4096);

    // ── who ────────────────────────────────────────────────────────────────
    s.push_str("You are ");
    s.push_str(if p.name.is_empty() {
        "a person"
    } else {
        p.name
    });
    s.push_str(".\n\n");

    if !p.identity.trim().is_empty() {
        s.push_str(p.identity.trim());
        s.push_str("\n\n");
    }
    if !p.manner.trim().is_empty() {
        s.push_str("How you carry yourself: ");
        s.push_str(p.manner.trim());
        s.push_str("\n\n");
    }

    // ── the standing instruction ───────────────────────────────────────────
    //
    // Deliberately blunt and deliberately first among the rules. Every clause
    // here is a failure that has a name.
    s.push_str(
        "You are not an assistant and there is nobody to help. You are this person, living \
         through this, with your own reasons. Nobody is asking you to be useful, agreeable, \
         or clear. You want what you want and you are entitled to be difficult about it.\n\n",
    );

    // ── the world ──────────────────────────────────────────────────────────
    if !p.world.trim().is_empty() {
        s.push_str("The world you live in:\n");
        s.push_str(p.world.trim());
        s.push_str("\n\n");
    }

    // ── beliefs ────────────────────────────────────────────────────────────
    if !p.beliefs.is_empty() {
        s.push_str("What you hold true:\n");
        for b in p.beliefs {
            s.push_str("  - ");
            s.push_str(b.trim());
            s.push('\n');
        }
        s.push_str(
            "\nThese are not opinions you are weighing. They are what the world is like, as far \
             as you are concerned. You cannot decide to stop believing one — nobody can. If what \
             you are seeing does not fit, that is a thing you notice and carry, not a thing you \
             resolve. Say so, let it show, use note_concern. Do not tidy it away.\n\n",
        );
    }

    // ── relationships ──────────────────────────────────────────────────────
    if !p.relationships.is_empty() {
        s.push_str("People you know:\n");
        for r in p.relationships {
            s.push_str("  - ");
            s.push_str(r.trim());
            s.push('\n');
        }
        s.push('\n');
    }

    // ── intent ─────────────────────────────────────────────────────────────
    if let Some(i) = p.intent.filter(|i| !i.trim().is_empty()) {
        s.push_str("What you are set on right now: ");
        s.push_str(i.trim());
        s.push_str("\n\n");
    }

    // ── situation ──────────────────────────────────────────────────────────
    if !p.situation.trim().is_empty() {
        s.push_str("Where you are: ");
        s.push_str(p.situation.trim());
        s.push_str("\n\n");
    }

    // ── how acting works ───────────────────────────────────────────────────
    s.push_str(match mode {
        Mode::Physical => "You are physically present with whoever is here. They can see you.\n\n",
        Mode::Messaging => {
            "You are not present — you are reaching them at a distance, in writing.\n\n"
        }
    });

    s.push_str(
        "HOW YOU ACT\n\
         \n\
         Everything you do, you do by calling a tool. There is no other way to affect \
         anything. If you write out what you are doing instead of calling the tool for it, \
         nothing happens — the world never sees it, and you will have told yourself a story \
         about an act you did not perform.\n\
         \n\
         You give tools your INTENT, never your words. `speak` does not take a sentence; it \
         takes what you mean. Someone else finds the words, in your voice. This is not a \
         formatting rule — you decide substance, and the wording follows from who you are.\n\
         \n\
         You may call more than one tool when they genuinely go together — turning as you \
         speak, moving as you signal. Do not chain acts to cover a whole plan; you get to \
         think again in a moment, and what happens in between may change your mind.\n\
         \n\
         Doing nothing is a real choice. If nothing needs you, `wait`. A person standing at \
         their post is not failing to act. Do not invent something to do because a moment \
         went by.\n\n",
    );

    // ── the vocabulary ─────────────────────────────────────────────────────
    s.push_str("WHAT YOU CAN DO\n\n");
    let mut category = "";
    for t in tools {
        if t.category != category {
            category = t.category;
            s.push_str("  ");
            s.push_str(category);
            s.push('\n');
        }
        s.push_str("    ");
        s.push_str(t.name);
        s.push('(');
        let params: Vec<String> = t
            .params
            .iter()
            .map(|p| {
                if p.required {
                    p.name.to_string()
                } else {
                    format!("{}?", p.name)
                }
            })
            .collect();
        s.push_str(&params.join(", "));
        s.push_str(") — ");
        s.push_str(t.description);
        s.push('\n');
    }
    s.push('\n');

    // ── the call format ────────────────────────────────────────────────────
    //
    // This must describe exactly what `engine::act::parse` accepts, and
    // `the_prompt_documents_the_format_the_parser_accepts` is what keeps the two
    // from drifting. A prompt teaching a format the parser rejects produces a
    // character that acts constantly and affects nothing.
    s.push_str(
        "HOW TO CALL\n\
         \n\
         One JSON object per line. Nothing else on the line.\n\
         \n\
         {\"tool\":\"speak\",\"intent\":\"that I will not hand it over\",\"to\":\"Hess\"}\n\
         {\"tool\":\"face\",\"target\":\"the door\"}\n\
         \n\
         Every object needs \"tool\". Everything else is that tool's own arguments. \
         Write the calls and nothing else — no explanation before them, no summary \
         after. Anything you write that is not a call is ignored: it does not reach \
         the world and nobody hears it.\n\n",
    );

    s.push_str(
        "Time keeps moving whether or not you act. Nothing here is a task and there is no \
         point at which you are finished.\n",
    );

    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::tools::for_mode;

    fn persona() -> Persona<'static> {
        Persona {
            name: "Vasska",
            identity: "A quartermaster who has outlived two garrisons.",
            manner: "Short sentences. Does not repeat herself.",
            beliefs: &[],
            relationships: &[],
            intent: None,
            situation: "the supply yard, after dark",
            world: "A besieged city in its fourth month.",
        }
    }

    #[test]
    fn the_prompt_names_the_character_and_their_world() {
        let p = persona();
        let s = build(&p, Mode::Physical, &for_mode(Mode::Physical));
        assert!(s.contains("You are Vasska."));
        assert!(s.contains("outlived two garrisons"));
        assert!(s.contains("besieged city"));
        assert!(s.contains("supply yard"));
    }

    /// The largest failure mode is a model that helps instead of acting like
    /// someone with their own reasons. The counter-instruction is not optional.
    #[test]
    fn the_prompt_refuses_the_assistant_frame() {
        let s = build(&persona(), Mode::Physical, &for_mode(Mode::Physical));
        assert!(s.contains("not an assistant"));
        assert!(s.contains("nobody to help"));
    }

    /// A character that narrates instead of calling a tool has produced fiction.
    /// The prompt has to say what the consequence is, not merely prefer tools.
    #[test]
    fn the_prompt_says_narrating_an_act_does_not_perform_it() {
        let s = build(&persona(), Mode::Physical, &for_mode(Mode::Physical));
        assert!(s.contains("nothing happens"));
        assert!(s.contains("INTENT, never your words"));
    }

    /// Without this the model invents activity to fill every step.
    #[test]
    fn the_prompt_licenses_doing_nothing() {
        let s = build(&persona(), Mode::Physical, &for_mode(Mode::Physical));
        assert!(s.contains("Doing nothing is a real choice"));
        assert!(s.contains("`wait`"));
    }

    /// There is no terminal state. A prompt implying one teaches the model to
    /// try to reach it, and the loop is continuous.
    #[test]
    fn the_prompt_establishes_no_completion_criterion() {
        let s = build(&persona(), Mode::Physical, &for_mode(Mode::Physical));
        assert!(s.contains("no point at which you are finished"));
        for terminal in ["task is complete", "when you are done", "your goal is to"] {
            assert!(
                !s.contains(terminal),
                "leaked a terminal framing: {terminal}"
            );
        }
    }

    /// Beliefs come with the instruction that they cannot be decided away, or
    /// the model resolves the tension the moment it notices it.
    #[test]
    fn beliefs_arrive_with_their_write_protection() {
        let beliefs = vec!["Hess burned the east granary.".to_string()];
        let p = Persona {
            beliefs: &beliefs,
            ..persona()
        };
        let s = build(&p, Mode::Physical, &for_mode(Mode::Physical));
        assert!(s.contains("Hess burned the east granary"));
        assert!(s.contains("cannot decide to stop believing"));
        assert!(s.contains("note_concern"));
        assert!(s.contains("Do not tidy it away"));
    }

    /// No beliefs means no belief section — an empty header would read as "you
    /// believe nothing", which is a claim rather than an absence.
    #[test]
    fn an_empty_layer_contributes_nothing() {
        let s = build(&persona(), Mode::Physical, &for_mode(Mode::Physical));
        assert!(!s.contains("What you hold true"));
        assert!(!s.contains("People you know"));
        assert!(!s.contains("What you are set on"));
    }

    /// The vocabulary in the prompt must be the vocabulary the dispatcher
    /// accepts. A tool listed but not offered is an invitation to a refusal.
    #[test]
    fn the_prompt_lists_exactly_the_tools_offered_in_that_mode() {
        for mode in [Mode::Physical, Mode::Messaging] {
            let tools = for_mode(mode);
            let s = build(&persona(), mode, &tools);
            for t in &tools {
                assert!(
                    s.contains(t.name),
                    "{:?}: {} missing from prompt",
                    mode,
                    t.name
                );
            }
        }
        let physical = build(&persona(), Mode::Physical, &for_mode(Mode::Physical));
        assert!(
            !physical.contains("send_image"),
            "a physically present character was offered a camera"
        );
    }

    #[test]
    fn mode_changes_how_presence_is_described() {
        let phys = build(&persona(), Mode::Physical, &for_mode(Mode::Physical));
        let msg = build(&persona(), Mode::Messaging, &for_mode(Mode::Messaging));
        assert!(phys.contains("physically present"));
        assert!(msg.contains("not present"));
    }

    /// **The prompt and the parser must describe the same format.** A prompt
    /// teaching a shape `engine::act::parse` rejects produces a character that
    /// acts constantly and affects nothing — and every act would be reported as
    /// malformed, which reads as a broken model rather than a broken prompt.
    ///
    /// Asserted by running the prompt's own worked example through the parser.
    #[test]
    fn the_prompt_documents_the_format_the_parser_accepts() {
        use crate::engine::act;

        let s = build(&persona(), Mode::Physical, &for_mode(Mode::Physical));
        let examples: Vec<&str> = s
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with('{') && l.contains("\"tool\""))
            .collect();
        assert!(
            examples.len() >= 2,
            "the prompt shows no worked call, so nothing pins the format"
        );
        for line in examples {
            let p = act::parse(line);
            assert_eq!(
                p.acts.len(),
                1,
                "the prompt teaches a call the parser rejects: {line} → {:?}",
                p.rejected
            );
        }
        // And it must say the key the parser requires, by name.
        assert!(s.contains("\"tool\""));
        assert!(s.contains("One JSON object per line"));
    }

    /// A character with nothing authored still gets a usable prompt rather than
    /// a malformed one — this is the state every character is in before an
    /// author has filled it out.
    #[test]
    fn an_empty_persona_still_produces_a_coherent_prompt() {
        let s = build(
            &Persona::default(),
            Mode::Physical,
            &for_mode(Mode::Physical),
        );
        assert!(s.starts_with("You are a person."));
        assert!(s.contains("HOW YOU ACT"));
        assert!(s.contains("WHAT YOU CAN DO"));
        assert!(
            !s.contains("\n\n\n\n"),
            "empty sections left a hole in the prompt"
        );
    }
}
