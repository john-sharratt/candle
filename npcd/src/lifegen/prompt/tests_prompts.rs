//! Tests for what every lifegen prefix carries, and for what it must not carry.
//!
//! The prefixes are checked for the character sheet, the cast, the voice and the
//! seed, and for the absence of anything that names a particular year, month or
//! day, since that would be per-fork content in the shared text. An era note must
//! read as an instruction rather than as a sentence the diary could have written.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::lifegen::plan::Content;

    use crate::lifegen::seed::{check, Cadence, EraKind, Seed, WorldEvent};

    use serde_json::json;

    pub(in super::super) fn seed() -> Seed {
        Seed {
            who: "cindy-tan".into(),
            display: "Cindy Tan".into(),
            born: "1998-09-14".into(),
            through: "2001-03-02".into(),
            place: "Nanyang".into(),
            role: "a records clerk".into(),
            cadence: Cadence::Punctuated,
            facts: vec!["She never learned to swim.".into()],
            world: vec![WorldEvent {
                date: "2001-04-02".into(),
                what: "The east granary burned.".into(),
            }],
            cast: Vec::new(),
            eras: Vec::new(),
        }
    }

    pub(in super::super) fn plan() -> Plan {
        let mut p = Plan::new(&check(&seed()).unwrap());
        p.story.content = Content {
            title: "An Arc".into(),
            text: "You were born in the rain.".into(),
            edited: false,
            stale: false,
        };
        p.story.cast.push(CastMember {
            entity_id: "prof-lim".into(),
            display: "Professor Lim".into(),
            what: "taught you to question".into(),
            npc_id: Some(3),
            from_seed: false,
        });
        p.story.outline.push(YearBeat {
            year: 1999,
            title: "The Quiet Year".into(),
            premise: "Nothing arrives.".into(),
        });
        p
    }

    pub(in super::super) fn cons(tool: &str, args: serde_json::Value) -> Consequence {
        Consequence {
            tool: tool.into(),
            args: match args {
                serde_json::Value::Object(m) => m,
                _ => unreachable!(),
            },
        }
    }

    /// **The character's own idiom reaches every phase.**
    ///
    /// These documents are injected into that character's prefix, so a life
    /// written in a neutral literary register is a few thousand tokens of
    /// somebody else's voice teaching the model how to sound.
    #[test]
    fn the_characters_voice_reaches_every_phase() {
        let person = json!({
            "anchor": "You count the rounds and you say the number out loud.",
            "personality": { "voice": "Flat, arithmetical, never hurried." },
        });
        let v = voice_of(&person);
        assert!(v.contains("say the number out loud"), "no anchor: {v}");
        assert!(v.contains("Flat, arithmetical"), "no voice trait: {v}");

        let c = check(&seed()).unwrap();
        let p = plan();
        for prefix in [
            story_prefix(&c, &v),
            years_prefix(&p, &v),
            months_prefix(&p, &v),
            days_prefix(&p, &v),
        ] {
            assert!(prefix.contains("say the number out loud"), "voice missing");
            assert!(prefix.contains("WHOSE VOICE THIS IS IN"), "no heading");
            assert!(prefix.contains("diary"), "not asked for as a diary");
        }
    }

    /// A personality with neither an anchor nor a voice still generates — in the
    /// plain diary register, rather than against an empty heading that reads as
    /// a section somebody forgot to fill in.
    #[test]
    fn a_character_with_no_authored_voice_gets_no_empty_heading() {
        assert_eq!(voice_of(&json!({})), "");
        let p = plan();
        assert!(!years_prefix(&p, "").contains("WHOSE VOICE THIS IS IN"));
        // The diary framing is in VOICE itself, so it survives a missing anchor.
        assert!(years_prefix(&p, "").contains("diary"));
    }

    /// **The voice's example must not be usable as content.**
    ///
    /// It was a sentence about a yard and a burnt granary, and two different
    /// models reproduced it verbatim as the opening line of lives that had
    /// neither — including a combat drone's. A demonstration of style sits in
    /// the same prompt as the material, and a model cannot tell them apart by
    /// position, so the demonstration has to be inert.
    #[test]
    fn the_voice_demonstrates_grammar_without_giving_away_a_scene() {
        // `VOICE` itself, not a rendered prefix: an author's own seed may
        // legitimately name a granary, and that is content the life is supposed
        // to be about. What must not carry an image is the instruction.
        let v = VOICE.to_lowercase();
        for stolen in ["granary", "yard"] {
            assert!(
                !v.contains(stolen),
                "the voice still hands the model a `{stolen}` to write about",
            );
        }
        // The grammatical point still has to be made, and made by contrast.
        assert!(VOICE.contains("second person"), "no voice instruction");
        assert!(VOICE.contains("You waited"), "no example of the right form");
        assert!(VOICE.contains("She waited"), "no example of the wrong form");
    }

    /// **Every rung knows who it is writing about.**
    ///
    /// The shared prefix used to carry only what the story phase produced — the
    /// arc, the cast, the outline — and none of what it was produced from. So a
    /// month knew the shape of the life and not the person living it, and a day,
    /// the most concrete rung of the four, had the least to be concrete with.
    #[test]
    fn every_phase_below_the_story_carries_the_character_sheet() {
        let p = plan();
        for prefix in [
            years_prefix(&p, ""),
            months_prefix(&p, ""),
            days_prefix(&p, ""),
        ] {
            assert!(prefix.contains("THE CHARACTER"), "no character sheet");
            assert!(prefix.contains("Cindy Tan"), "no name");
            assert!(prefix.contains("Nanyang"), "no place");
            assert!(prefix.contains("a records clerk"), "no trade");
            // The author's fixed facts are the closest thing the seed has to an
            // anchor, and they are what a day has to stay true to.
            assert!(prefix.contains("never learned to swim"), "no facts");
            // And the world, so a day cannot contradict what other lives in it
            // remember.
            assert!(prefix.contains("east granary"), "no world events");
        }
    }

    /// **The load-bearing property of the whole fan-out.** Anything that varies
    /// per fork must be absent from the shared prefix, or every fork attends a
    /// prompt built for one of them.
    #[test]
    fn no_phase_prefix_names_a_specific_year_month_or_day() {
        let p = plan();
        for prefix in [
            years_prefix(&p, ""),
            months_prefix(&p, ""),
            days_prefix(&p, ""),
        ] {
            // The outline legitimately lists years; what must not appear is an
            // instruction naming one particular unit to expand.
            for needle in ["January", "the 14th", "Write the year 1999"] {
                assert!(
                    !prefix.contains(needle),
                    "a per-fork detail reached the shared prefix: {needle}"
                );
            }
        }
    }

    /// The prefix carries the cast, and says the lower phases may not add to it
    /// — the rule that stops one professor becoming four slugs.
    #[test]
    fn every_expansion_prefix_carries_the_cast_and_forbids_inventing_people() {
        let p = plan();
        for prefix in [
            years_prefix(&p, ""),
            months_prefix(&p, ""),
            days_prefix(&p, ""),
        ] {
            assert!(prefix.contains("prof-lim"));
            assert!(prefix.contains("Do not introduce anyone who is not on this list"));
        }
    }

    /// Every instruction below the story says *expand*, because a child that
    /// invents breaks the parallelism rather than merely reading differently.
    #[test]
    fn the_month_prefix_says_why_inventing_would_break_things() {
        let p = plan();
        let m = months_prefix(&p, "");
        assert!(m.contains("Do not invent events the year does not contain"));
        assert!(m.contains("cannot see what you write"));
        assert!(m.contains("Every month is written"));
    }

    /// The register is not optional: a third-person document is a biography of
    /// somebody the character has never met.
    #[test]
    fn every_prefix_asks_for_the_characters_own_second_person() {
        let c = check(&seed()).unwrap();
        let p = plan();
        for prefix in [
            story_prefix(&c, ""),
            years_prefix(&p, ""),
            months_prefix(&p, ""),
            days_prefix(&p, ""),
        ] {
            assert!(prefix.contains("second person"), "{prefix}");
            // The third-person prohibition, made by contrast rather than by
            // naming the grammar — see `VOICE` for why the example is a pronoun
            // and not a scene.
            assert!(prefix.contains("never \"She waited\""));
            assert!(prefix.contains("never the character's name as the"));
        }
    }

    #[test]
    fn the_story_prefix_carries_the_seed_and_the_cadence() {
        let c = check(&seed()).unwrap();
        let s = story_prefix(&c, "");
        assert!(s.contains("Cindy Tan") && s.contains("Nanyang") && s.contains("1998-09-14"));
        assert!(s.contains("She never learned to swim."));
        assert!(s.contains("The east granary burned."));
        assert!(s.contains(Cadence::Punctuated.instruction()));
        // The world events are marked as shared, so a life cannot move one.
        assert!(s.contains("Other people in this world remember them the same way"));
    }

    /// **The shared prefix does not grow with the length of the life.**
    ///
    /// It carried every line of the outline, which was affordable at eight beats
    /// and ruinous at one per year lived: a 367-year life put 4,900 tokens of
    /// year list in front of every fork, and a character's years phase refused
    /// all forty-one of its nodes because the prompt no longer left room for an
    /// answer. A fork's own beat is in its turn; its neighbours are a window.
    #[test]
    fn the_shared_prefix_does_not_carry_the_whole_outline() {
        let mut s = seed();
        s.born = "2719-01-01".into();
        s.through = "3087-01-01".into();
        let mut p = Plan::new(&check(&s).unwrap());

        let short = years_prefix(&p, "").len();
        p.story.outline = (2719..3087)
            .map(|year| YearBeat {
                year,
                title: format!("Year {year}"),
                premise: "You held the ground that cost them most to take.".into(),
            })
            .collect();
        let long = years_prefix(&p, "").len();

        assert!(
            long.abs_diff(short) < 200,
            "the prefix moved {} chars for 368 years of outline",
            long.abs_diff(short)
        );
    }

    /// An era note may not be a sentence the diary could have written.
    ///
    /// A model handed `"They were being developed inside a simulation for this
    /// — conscious, but in a world with rules somebody else set…"` moved it to
    /// the second person and used it as narration, three times in one story.
    /// The tell is grammatical, so the test is too: an era note opens on a
    /// fragment or an imperative, never on a pronoun with a past-tense verb
    /// behind it, and it carries an instruction to the writer.
    #[test]
    fn an_era_note_reads_as_an_instruction_and_not_as_prose() {
        for kind in [EraKind::Lived, EraKind::Developed, EraKind::Dormant] {
            let note = kind.instruction();

            let opener = note.split_whitespace().next().unwrap().to_lowercase();
            assert!(
                !["they", "he", "she", "you", "it"].contains(&opener.as_str()),
                "{kind:?} note opens on a pronoun, so it reads as narration: {note}"
            );

            // Every note tells the writer what to do with the span.
            let lower = note.to_lowercase();
            assert!(
                lower.contains("write") || lower.contains("do not narrate"),
                "{kind:?} note gives the writer no instruction: {note}"
            );
        }
    }
}
