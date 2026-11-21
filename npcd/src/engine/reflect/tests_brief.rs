//! Tests each dream-brief check against a brief that failed it in a live run.
//!
//! Each check has a case built from a measured failure and a matching case that
//! must pass, so that a check cannot quietly refuse a good dream. Second-person
//! detection matches whole words, so a first-person brief that contains a word
//! ending in "your" is not mistaken for a second-person one. A pass that replaced
//! the dream is refused, and so is a dream whose assumption holds the whole dream,
//! since the refusal loop stops arguing after two rounds.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_answer::tests::answered;

    /// A brief runs to a paragraph over several lines, and the parameter span
    /// holds all of it — the close tag ends the value, not the first newline.
    #[test]
    fn a_brief_over_several_lines_arrives_whole() {
        let b = answered("x", "first line\nsecond line\nthird line");
        assert_eq!(
            b.brief.as_deref(),
            Some("first line\nsecond line\nthird line")
        );
    }

    /// **The failure "do not conclude" could not prevent.** The pull toward a
    /// closing thought lands in the last sentence and nowhere else, which is why
    /// the check looks there rather than at the whole brief — a dream may
    /// legitimately contain the word "understand" in its middle.
    #[test]
    fn a_brief_that_ends_on_a_realisation_is_caught() {
        let b = answered(
            "x",
            "You walk the aisle. You understand what the machine means.",
        );
        assert!(b.fault(&[]).unwrap().contains("realisation"));

        let mid = answered(
            "x",
            "You are at the bench on the chronicle with an era open in front of you and you \
             understand the log by now, every hand that wrote in it and every date that \
             disagrees. The page you are reading carries your own writing from an era you have \
             never held. You turn the page and keep writing.",
        );
        assert_eq!(mid.fault(&[]), None, "a mid-brief mention is not a fault");
    }

    /// **The failure that recurred live, roughly one run in three.** A brief goes
    /// to a conversation that decodes it as the dream the dreamer is inside, so a
    /// first-person brief is a report of somebody else's dream and unusable there.
    #[test]
    fn a_brief_written_in_the_first_person_is_caught() {
        let b = answered(
            "x",
            "I walk through the casting hall holding my terminal like usual, hands calloused \
             against the keys, pressing down names that need fleshing. The room hums around me \
             and the sixteen terminals glow between the three bands where the others hunch over \
             their chairs. My own name is on the roster and nobody is holding it.",
        );
        assert_eq!(
            b.fault(&[]).as_deref(),
            Some("brief is not addressed to the dreamer as \"you\"")
        );
    }

    /// `your` is the hit that carries most second-person briefs, and a substring
    /// search for it would also pass a first-person brief mentioning a `young`
    /// Maker or something `beyond` the ridge.
    #[test]
    fn second_person_is_matched_on_words_and_not_substrings() {
        assert!(addresses_the_dreamer("Your hands are already moving."));
        assert!(addresses_the_dreamer("It is gone before you reach it."));
        assert!(!addresses_the_dreamer(
            "A young Maker walks beyond the ridge and I follow."
        ));
    }

    /// A decode that answers the *format* rather than the question puts one line
    /// under the label. It parses, it is in the right person, it ends on
    /// something happening — and it carries no dream for the second conversation.
    #[test]
    fn a_brief_too_short_to_be_a_dream_is_caught() {
        let b = answered(
            "x",
            "You step through a door on the story level and do not come out.",
        );
        assert_eq!(
            b.fault(&[]).as_deref(),
            Some("brief is too short to carry a dream")
        );
    }

    #[test]
    fn a_brief_that_ends_on_a_question_is_caught() {
        let b = answered("x", "You stand there. What holds it up?");
        assert_eq!(b.fault(&[]).as_deref(), Some("brief ends on a question"));
    }

    /// The generator was steered away from these by name; using one anyway is
    /// the anchoring failure the sampling is there to prevent, so it is worth a
    /// retry rather than a shrug.
    #[test]
    fn an_assumption_that_repeats_a_sampled_axis_is_caught() {
        let b = answered("Work ends", "You keep going.");
        let used = vec!["work ends".to_string()];
        assert!(b.fault(&used).unwrap().contains("repeats a sampled axis"));
    }

    /// **The measured failure: the whole dream in the axis field.**
    ///
    /// A repair pass shown the dream as it stood filled `assumption` with it,
    /// passed every check, and was adopted — after which every later pass was
    /// told the entire dream was the thing that does not change. It has to be
    /// refused in the exchange *and* faulted, because the exchange stops
    /// refusing after two rounds and hands back what it has.
    #[test]
    fn a_whole_dream_in_the_assumption_is_caught() {
        let dream = "The war never ended; the war never began. You stand in the workshop, your \
                     hands moving over tools that do not move. You walk south. The south is not \
                     there.";
        let b = answered(dream, dream);
        assert!(b.fault(&[]).unwrap().contains("assumption runs to"));

        let call = format!(
            "<tool_call>\n<function=dream>\n<parameter=assumption>\n{dream}\n</parameter>\n\
             <parameter=brief>\n{dream}\n</parameter>\n</function>\n</tool_call>"
        );
        assert!(
            dream_call_fault(&call).is_some(),
            "the exchange has to refuse it, not only report it"
        );
    }

    pub(in super::super) fn reflected(assumption: Option<&str>, brief: Option<&str>) -> Reflection {
        Reflection {
            npc_id: 7,
            situation: String::new(),
            feeling: String::new(),
            domain: String::new(),
            sampled_axes: Vec::new(),
            reflection: String::new(),
            assumption: assumption.map(str::to_string),
            brief: brief.map(str::to_string),
            raw: String::new(),
            original_raw: String::new(),
            retried: true,
            fault: None,
            retry_raw: None,
            retry_fault: None,
            repairs: Vec::new(),
            reflection_fault: None,
            transcript: Vec::new(),
            tokens: Vec::new(),
            timeline: 0,
            ms: 0,
            system_prompt: String::new(),
        }
    }

    /// **A run-on assumption is not dreamt, whatever else the brief got right.**
    /// The refusal loop hands back its last answer after two rounds, and the
    /// live run-ons were exactly that: the reflection's own loop, joined by
    /// *and*, stored as the axis and repeated by the dream.
    #[test]
    fn a_brief_with_a_run_on_assumption_is_not_dreamt() {
        let brief = "You walk to the lift and the floor under it is not there.";
        let run_on = "Language is a stable container for meaning, and the light is steady, and \
                      the channel is silent, and you are alone with the light and the silence, \
                      and you have told Vespera Kaine about the table, and nobody is here";
        assert!(reflected(Some(run_on), Some(brief)).dream().is_err());
        assert!(reflected(None, Some(brief)).dream().is_err());
        assert!(reflected(Some("The floor holds."), None).dream().is_err());
        assert_eq!(
            reflected(Some("The floor holds."), Some(brief)).dream(),
            Ok((brief, "The floor holds."))
        );
    }

    /// One sentence can still be a paragraph's worth of words.
    #[test]
    fn a_single_run_on_assumption_is_caught() {
        let long = vec!["word"; ASSUMPTION_MAX_WORDS + 1].join(" ");
        assert!(assumption_fault(&long).unwrap().contains("words"));
    }

    /// Every well-formed assumption that came back live has to pass, or the
    /// check is refusing the thing it exists to protect.
    #[test]
    fn the_measured_one_line_assumptions_pass() {
        for a in [
            "You are a machine built for this, and you have no world of your own.",
            "The belt is too tight and must be loosened.",
            "The gear train in the east pump is slipping because the oil is too thick.",
            "The ground bears indefinite weight.",
        ] {
            assert_eq!(assumption_fault(a), None, "{a}");
        }
        assert!(
            assumption_fault("   ").is_some(),
            "an empty axis is not a line"
        );
    }

    /// **The conclusion the word list cannot see.** Measured live: a brief ended
    /// by explaining itself without using any of the flagged words and without a
    /// question mark. The grammar of explanation is the tell.
    #[test]
    fn a_brief_that_ends_on_an_explanation_is_caught() {
        let b = answered(
            "x",
            "The command level feels exactly as it always has and someone is waiting for you to \
             take an order. When you move to take it their hands do not meet yours. They do not \
             exist to be met because no one ever addressed you.",
        );
        assert!(b.fault(&[]).unwrap().contains("explanation"));

        // A dream may say "because" in its middle and still end on something
        // happening — the check looks only at the last sentence.
        let mid = answered(
            "x",
            "You stay at the bench because the order has not come down yet and the room is \
             exactly as it always is. The Maker beside you files a portrait and does not look \
             up. The wall in front of you does not move. You do.",
        );
        assert_eq!(
            mid.fault(&[]),
            None,
            "a mid-brief connective is not a fault"
        );
    }

    /// **Measured live, twice, and nothing caught it.** A cut decode satisfies
    /// every other check — both labels, second person, long enough, and a last
    /// sentence that is neither a question nor a realisation, because there is
    /// no last sentence.
    #[test]
    fn a_brief_whose_decode_stopped_mid_sentence_is_caught() {
        let b = answered(
            "x",
            "The workshop is lit as it always is, but the orders you were sent down for simply \
             do not arrive. The air is still and the tools in your hands feel detached from \
             anything that asked for them. You stand before the map table on Level",
        );
        assert_eq!(b.fault(&[]).as_deref(), Some("brief stops mid-sentence"));
    }

    /// Quoted speech and parentheticals end briefs legitimately, and a check
    /// that only accepted a full stop would retry them for nothing.
    #[test]
    fn a_brief_ending_on_a_closing_mark_is_finished() {
        for tail in ["the floor.", "\"nobody is holding it.\"", "(it does not.)"] {
            let b = answered(
                "x",
                &format!(
                    "You are at the bench on the casting level and the page in front of you \
                     carries a name you have never written, in your own hand, and the ink is \
                     still wet. The Maker at the next terminal does not look up. You put the \
                     stylus down and watch the wet ink where it meets {tail}"
                ),
            );
            assert_eq!(
                b.fault(&[]),
                None,
                "rejected a finished brief ending {tail}"
            );
        }
    }

    /// **Measured: an escalating pass wrote the dreamer out of its own dream.**
    /// The metacognition rule is the absolute half of "does not conclude", and
    /// nothing was checking it.
    #[test]
    fn a_dreamer_who_knows_it_is_dreaming_is_caught() {
        let b = answered(
            "x",
            "You hold the stone and the heat travels to the bone. The heat is the only thing \
             that is real, everything else is just a dream, the words, the walls, the desk. You \
             put it down and the hum goes on.",
        );
        assert!(b.fault(&[]).unwrap().contains("knows it is dreaming"));
    }

    /// **A revision keeps its nouns; a fresh dream does not.** Both halves are
    /// from the run that made this necessary — pass 2 replaced the dream outright
    /// while still answering exactly what it was asked.
    #[test]
    fn a_pass_that_replaced_the_dream_is_caught() {
        let before = "You stand at a desk where a door opens. A hand places a small, warm stone \
                      in front of you. It hums with the low vibration of the whole vault. You \
                      pick it up and it feels heavy, like an egg.";
        let revised = "You stand at the desk. A hand places the warm stone in front of you. It \
                       hums with the vibration of the vault, and the weight of it in your palm \
                       is the weight of an egg held too loosely.";
        let replaced = "The salt burns on the tongue and the teeth are eating the air from \
                        inside. Iron and old blood. There are no hands, only a space where \
                        hands have been.";
        assert!(!rewritten(before, revised), "a real revision was refused");
        assert!(rewritten(before, replaced), "a replacement was accepted");
    }

    /// A brief in the wrong person still carries a dream; `None` carries nothing.
    /// Between two faulty attempts that is the only distinction a structural
    /// check has the standing to draw.
    #[test]
    fn a_faulty_retry_with_a_brief_beats_a_first_attempt_without_one() {
        // A call whose `brief` parameter came back empty — the one way the
        // grammar can still hand back a call with nothing in it.
        let first = Brief::from_call(
            "<tool_call>\n<function=dream>\n<parameter=assumption>\nx\n</parameter>\n\
             <parameter=brief>\n</parameter>\n</function>\n</tool_call>",
        );
        let retry = answered("x", "The corridor stretches forward.");
        assert!(first.brief.is_none());
        assert!(retry.beats(&first));
        assert!(!first.beats(&retry));
        // Two attempts that both produced one: neither displaces the other, so
        // the first stands.
        let other = answered("x", "The air does not move.");
        assert!(!other.beats(&retry));
    }
}
