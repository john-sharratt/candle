//! Tests for reading a decode back: its sections, its headings in each spelling,
//! its fences and its reasoning.
//!
//! Each test gives the reader an answer shaped the way a model returns it, and
//! asserts what the document keeps. The prose must never include a reasoning
//! block, a fence or a heading. A heading in any spelling must still parse, and a
//! word inside a sentence must not be taken for one.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// A fenced answer is a document, not a diary with backticks in it.
    ///
    /// Asked for prose plus two `##` sections, a model wrapped the whole answer
    /// in a code fence and then opened a second, empty one after it. The
    /// headings still parsed and the outline still landed — the only casualty
    /// was the prose, which is the part written into the character's memory and
    /// read back to them.
    #[test]
    fn a_fenced_answer_keeps_its_prose_and_loses_its_fences() {
        let out = parse_story(
            "```\nYou read ground before you read people.\n```\n\n```\n\n\
             ## Years\n- 2751 | Certified | You were counted into the Twelve Hundred.\n",
        );

        assert_eq!(out.prose, "You read ground before you read people.");
        assert!(!out.prose.contains("```"), "markup in a character's memory");
        assert_eq!(
            out.outline.len(),
            1,
            "the outline still parses: {:?}",
            out.outline
        );
        assert_eq!(out.outline[0].year, 2751);
    }

    /// An outline turn asks for one thing, so its rows need no heading over
    /// them. Requiring one would discard an answer that is entirely correct for
    /// want of a label nobody needs.
    #[test]
    fn an_outline_turn_parses_bare_rows() {
        let out = parse_outline(
            "- 2751 | Certified | You were counted into the Twelve Hundred.\n\
             - 2752 | The First Position | You held ground that cost them to take.\n",
        );

        assert_eq!(out.len(), 2, "{out:?}");
        assert_eq!(out[0].year, 2751);
        assert_eq!(out[0].title, "Certified");
        assert_eq!(out[1].year, 2752);
    }

    /// A model that supplies a heading anyway is not punished for it.
    #[test]
    fn an_outline_turn_parses_rows_under_a_heading() {
        let out = parse_outline("## Years\n- 2751 | Certified | You were counted in.\n");
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0].year, 2751);
    }

    /// A bare label is a heading too. One character's decode carried a full cast
    /// and a six-year outline written as plain `Cast` and `Years` lines, and
    /// parsed to nothing at all: the whole tail was read as prose and pasted
    /// onto the end of that character's own life story.
    #[test]
    fn an_unmarked_heading_is_still_a_heading() {
        let out = parse_story(
            "You laid fire and watched it take.\n\n\
             Cast\n- hart | Hart | brought you to the Volunteer Call\n\n\
             Years\n- 2722 | Workshop Years | You begin as a new beam.\n",
        );

        assert_eq!(out.prose, "You laid fire and watched it take.");
        assert_eq!(out.cast.len(), 1, "cast: {:?}", out.cast);
        assert_eq!(out.cast[0].entity_id, "hart");
        assert_eq!(out.outline.len(), 1, "outline: {:?}", out.outline);
        assert_eq!(out.outline[0].year, 2722);
    }

    /// Only an exact section label counts. A sentence that happens to contain
    /// the word is prose, and losing a paragraph to a false heading would be a
    /// worse failure than the one this catches.
    #[test]
    fn a_word_inside_a_sentence_is_not_a_heading() {
        assert_eq!(heading_name("The cast of your life was small."), None);
        assert_eq!(heading_name("Years passed."), None);
        assert_eq!(heading_name("  Cast  "), Some("Cast".to_string()));
        assert_eq!(heading_name("`## Years`"), Some("Years".to_string()));
    }

    /// A model shown its output shape as inline code answers in inline code.
    /// The quoting is decoration, so the structured tail still has to parse —
    /// otherwise the whole outline is read as prose and pasted onto the life.
    #[test]
    fn a_backtick_quoted_heading_is_still_a_heading() {
        let out = parse_story(
            "You read ground before you read people.\n\n\
             `## Cast`\n`- vasko | Vasko | certified you`\n\n\
             `## Years`\n`- 2751 | Certified | You were counted in.`\n",
        );

        assert_eq!(out.prose, "You read ground before you read people.");
        assert_eq!(out.cast.len(), 1, "cast: {:?}", out.cast);
        assert_eq!(out.cast[0].entity_id, "vasko");
        assert_eq!(out.outline.len(), 1, "outline: {:?}", out.outline);
        assert_eq!(out.outline[0].year, 2751);
    }

    /// Only a line that is *entirely* a fence goes. A sentence that happens to
    /// contain backticks is prose and stays whole.
    #[test]
    fn unfencing_leaves_prose_alone() {
        assert_eq!(
            unfence("You waited.\nIt did not stop."),
            "You waited.\nIt did not stop."
        );
        assert_eq!(unfence("```rust\nYou waited.\n```"), "You waited.");
    }

    /// **A thinking model's reasoning is not the document.** Stored verbatim it would be
    /// ingested and eventually read back to the character as something they remember thinking —
    /// and the real one contained "Total = 2792 - 2461 + 1 = 332. Correct."
    #[test]
    fn a_reasoning_block_never_reaches_the_document() {
        let raw = "<think>\nTotal = 2792 - 2461 + 1 = 332. Correct.\n</think>\n\n\
                   You were built to remember.\n\n## Cast\n- kaelor | Kaelor | the commander\n";
        let out = parse_story(raw);
        assert_eq!(out.prose, "You were built to remember.");
        assert!(!out.prose.contains("think"));
        assert_eq!(out.cast.len(), 1);

        // The other two parsers strip it too — a year or a month decode reasons just as much.
        let (t, p) = parse_titled("<think>weighing it up</think>\n# A Year\n\nIt rained.");
        assert_eq!((t.as_str(), p.as_str()), ("A Year", "It rained."));
        assert_eq!(
            parse_month("<think>x</think>\n# May\n\nCold.", 1998, 5).title,
            "May"
        );

        // Absent, and unterminated — the second is a decode that was *all* reasoning, and an
        // empty answer the caller reports beats a page of the model's notes.
        assert_eq!(parse_titled("# Plain\n\nProse.").0, "Plain");
        assert_eq!(parse_story("<think>never closed").prose, "");
    }

    #[test]
    fn a_story_decode_reads_back_into_prose_cast_and_outline() {
        let raw = "\
You were born in the rain.

It did not stop for a week.

## Cast
- prof-lim | Professor Lim | taught you to question assumptions
- hess | Hess | the man who burned the granary

## Years
- 1998 | Born | You arrive.
- 1999 | The Quiet Year | Nothing arrives.
";
        let out = parse_story(raw);
        assert_eq!(
            out.prose,
            "You were born in the rain.\n\nIt did not stop for a week."
        );
        assert_eq!(out.cast.len(), 2);
        assert_eq!(out.cast[0].entity_id, "prof-lim");
        assert_eq!(out.cast[0].display, "Professor Lim");
        assert_eq!(out.cast[1].what, "the man who burned the granary");
        assert_eq!(
            out.outline.iter().map(|b| b.year).collect::<Vec<_>>(),
            vec![1998, 1999]
        );
        assert_eq!(out.outline[1].title, "The Quiet Year");
    }

    /// **The id is the join key between a relationship and an NPC**, so it is
    /// normalised rather than trusted — a stray capital is a second person who
    /// never appears again.
    #[test]
    fn cast_ids_are_normalised_and_duplicates_dropped() {
        let out = parse_story(
            "p\n## Cast\n- Prof Lim | Professor Lim | x\n- prof-lim | Again | y\n- | Nameless | z\n",
        );
        assert_eq!(out.cast.len(), 1);
        assert_eq!(out.cast[0].entity_id, "prof-lim");
        assert_eq!(out.cast[0].display, "Professor Lim");
    }

    #[test]
    fn slugs_are_lowercase_hyphenated_and_free_of_punctuation() {
        assert_eq!(slug("Professor Lim"), "professor-lim");
        assert_eq!(slug("  O'Brien, Jr. "), "o-brien-jr");
        assert_eq!(slug("已知"), "");
        assert_eq!(slug("hess"), "hess");
    }

    /// A heading a model indented or ran together is still an answer.
    #[test]
    fn section_headings_are_read_leniently() {
        let out = parse_story("p\n  ##cast\n- a | A | x\n  ## YEARS \n- 1998 | T | P\n");
        assert_eq!(out.cast.len(), 1);
        assert_eq!(out.outline.len(), 1);
    }

    /// **The prose ends at the first heading, whichever heading that is.**
    ///
    /// The prose was collected until the wanted section started producing lines,
    /// which is only the same rule when the wanted heading comes first. A decode
    /// that puts `## Years` before `## Cast` had its whole year table read as
    /// prose — and the arc is what every year of the life is then expanded from,
    /// so twelve forks each opened with a list of year titles pasted onto the end
    /// of the story they were meant to be continuing.
    #[test]
    fn a_section_written_out_of_order_does_not_become_prose() {
        let out = parse_story(
            "She grew up in the delta.\n\n\
             ## Years\n- 1998 | Arrival | She arrives.\n\n\
             ## Cast\n- lim | Professor Lim | her tutor\n",
        );
        assert_eq!(out.prose, "She grew up in the delta.");
        assert_eq!(out.cast.len(), 1);
        assert_eq!(out.outline.len(), 1);
    }

    /// The same rule for a month: the day list is a section, not part of the
    /// month's prose, whichever order the model writes them in.
    #[test]
    fn a_months_prose_stops_at_the_first_heading() {
        let out = parse_month(
            "# A Quiet January\n\nNothing much happened.\n\n\
             ## Notes\nignore me\n\n\
             ## Days\n- 3 | the letter\n",
            1999,
            1,
        );
        assert!(
            !out.prose.contains("ignore me"),
            "another section's body became the month's prose: {:?}",
            out.prose
        );
        assert_eq!(out.days.len(), 1);
    }

    #[test]
    fn a_titled_document_splits_into_its_title_and_prose() {
        let (t, p) = parse_titled("# First Week at Nanyang\n\nYou arrived late.\n");
        assert_eq!(t, "First Week at Nanyang");
        assert_eq!(p, "You arrived late.");
    }

    /// A decode that forgot the title is not a failure — the prose is the
    /// valuable half, and a document titled from its opening beats one that did
    /// not get written.
    #[test]
    fn a_document_with_no_title_line_is_titled_from_its_opening() {
        let (t, p) = parse_titled("You came back to the yard and it was gone.");
        assert_eq!(t, "You came back to the yard");
        assert_eq!(p, "You came back to the yard and it was gone.");
    }

    #[test]
    fn a_month_decode_reads_back_with_its_notable_days() {
        let raw = "\
# First Term

You arrived late and stayed late.

## Days
- 14 | First Week at Nanyang | You met Lim.
- 30 | The Argument | It ended badly.
";
        let out = parse_month(raw, 1998, 9);
        assert_eq!(out.title, "First Term");
        assert_eq!(out.prose, "You arrived late and stayed late.");
        assert_eq!(
            out.days,
            vec![
                (14, "First Week at Nanyang".to_string()),
                (30, "The Argument".to_string())
            ]
        );
    }

    /// **A day outside the month is dropped, not clamped.** Clamping would put
    /// an episode on a day nobody chose, and the document would look
    /// deliberate.
    #[test]
    fn a_day_the_month_does_not_have_is_dropped() {
        let out = parse_month(
            "# F\n\nP.\n\n## Days\n- 31 | Nope | x\n- 0 | Nope | x\n- 12 | Yes | x\n",
            1999,
            2,
        );
        assert_eq!(out.days, vec![(12, "Yes".to_string())]);
        // February 2000 is a leap year, so the 29th is real.
        let leap = parse_month("# F\n\nP.\n\n## Days\n- 29 | Yes | x\n", 2000, 2);
        assert_eq!(leap.days.len(), 1);
    }

    #[test]
    fn a_month_with_no_notable_days_still_has_its_prose() {
        let out = parse_month("# A Quiet Month\n\nNothing came.\n", 1998, 9);
        assert_eq!(out.title, "A Quiet Month");
        assert_eq!(out.prose, "Nothing came.");
        assert!(out.days.is_empty());
    }

    #[test]
    fn duplicate_and_unordered_days_are_deduplicated_and_sorted() {
        let out = parse_month(
            "# F\n\nP.\n\n## Days\n- 30 | Late | x\n- 14 | Early | x\n- 14 | Again | x\n",
            1998,
            9,
        );
        assert_eq!(out.days, vec![(14, "Early".into()), (30, "Late".into())]);
    }
}
