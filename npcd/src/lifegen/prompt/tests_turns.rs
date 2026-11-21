//! Tests for the per-fork instructions: that each names its span, carries its
//! context, and asks for its consequences as direction.
//!
//! The tests check that a year is told it is a span and a month its weeks, that a
//! day is written against its year and its month, and that the story turn names
//! the subject and the register. They also check that dormant spans get no outline
//! line, and that no raw dial reaches the writer.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::lifegen::plan::{Content, Month, NodeId, Year};

    use crate::lifegen::seed::check;

    use serde_json::json;

    use super::super::tests_prompts::tests::{cons, plan, seed};

    /// **Each rung has to say how much time it covers.**
    ///
    /// A year and a day are the same instruction otherwise, separated only by a
    /// number the model has no reason to read as a duration — and a year of
    /// Yen's life came back as one morning in a courtyard. The rung that stated
    /// its span ("One day, closely") was the rung that got its span right.
    #[test]
    fn a_year_is_told_it_is_a_span_and_a_month_is_told_its_weeks() {
        let p = plan();
        let y = year_turn(&p, 1999);
        assert!(
            y.contains("whole year"),
            "the year does not claim the year:\n{y}"
        );
        assert!(
            y.to_lowercase().contains("not a scene"),
            "the year is not warned off writing a scene:\n{y}"
        );
        assert!(
            y.contains("second person"),
            "the year does not restate person:\n{y}"
        );

        let m = month_turn(&p, 1999, 9);
        assert!(
            m.contains("whole month"),
            "the month does not claim the month:\n{m}"
        );
        // Case-insensitive: the rungs emphasise the register differently — the
        // month rung shouts it, because it is the last line of that turn and the
        // last instruction read is the one that wins. What matters is that the
        // turn restates it at all, not how loudly.
        assert!(
            m.to_lowercase().contains("second person"),
            "the month does not restate person:\n{m}"
        );
    }

    /// **A day is written against its year as well as its month.**
    ///
    /// The month is the immediate context; the year is the shape the day is a
    /// moment inside. A day handed only its month writes a scene with no idea
    /// what the year was doing around it.
    #[test]
    fn a_day_carries_both_the_year_and_the_month_above_it() {
        let mut p = plan();
        // `Plan::new` already laid the years out from the seed's span, so this
        // fills the one that is there rather than pushing a second 1999 that
        // `Plan::year` would never reach.
        let y: &mut Year = p
            .years
            .iter_mut()
            .find(|y| y.year == 1999)
            .expect("the seed spans 1999");
        y.content = Content {
            title: "The Quiet Year".into(),
            text: "THE-YEAR-PROSE".into(),
            edited: false,
            stale: false,
        };
        // Same again for the month: the seed's grain may already have laid
        // September out, and a pushed duplicate is one `Plan::month` never
        // returns.
        let content = Content {
            title: "September".into(),
            text: "THE-MONTH-PROSE".into(),
            edited: false,
            stale: false,
        };
        match y.months.iter_mut().find(|m| m.month == 9) {
            Some(m) => m.content = content,
            None => y.months.push(Month {
                month: 9,
                content,
                days: Vec::new(),
            }),
        }
        let turn = day_turn(&p, 1999, 9, 14);
        assert!(
            turn.contains("THE-YEAR-PROSE"),
            "the year is missing:\n{turn}"
        );
        assert!(
            turn.contains("THE-MONTH-PROSE"),
            "the month is missing:\n{turn}"
        );
        // Widest first, so the model reads the year and narrows into the month.
        let (y, m) = (
            turn.find("THE-YEAR-PROSE").unwrap(),
            turn.find("THE-MONTH-PROSE").unwrap(),
        );
        assert!(y < m, "the year should precede the month it contains");
    }

    #[test]
    fn the_story_turn_names_the_whole_span_of_years() {
        let c = check(&seed()).unwrap();
        let t = story_turn(&c);
        assert!(t.contains("from 1998 to 2001 inclusive, none skipped"));
        assert!(t.contains("## Cast"));
        // The outline is `outline_turn`'s, a run of years at a time. Asked for
        // here it came back four to eighteen lines long for lives of three
        // hundred and sixty-seven years, whatever the token budget.
        assert!(
            !t.contains("## Years"),
            "the story turn is asking for the outline again: {t}"
        );
    }

    /// **A gap is named as a gap, and no line is asked for inside it.** Asking for "every
    /// year from first to last" would have Keeper outline 295 years of hibernation — an
    /// account of being unconscious, which is the one thing that cannot be written.
    #[test]
    fn the_story_turn_asks_for_no_outline_inside_a_dormant_span() {
        use crate::lifegen::seed::{Era, EraKind, Grain};
        let mut s = seed();
        s.born = "2461-01-01".into();
        s.through = "3087-01-01".into();
        s.eras = vec![
            Era {
                from: "2461-01-01".into(),
                to: "2792-12-31".into(),
                kind: EraKind::Lived,
                grain: Grain::Years,
                what: "The long watch.".into(),
            },
            Era {
                from: "2793-01-01".into(),
                to: "3087-01-01".into(),
                kind: EraKind::Dormant,
                grain: Grain::Years,
                what: "Hibernation.".into(),
            },
        ];
        let c = check(&s).unwrap();
        let t = story_turn(&c);
        assert!(t.contains("332 year(s) in total"), "{t}");
        assert!(t.contains("skipping 2793 to 3087"), "{t}");
        assert!(!t.contains("none skipped"));

        // The outline itself can no longer stray into the gap by construction
        // rather than by instruction: `build_outline` asks only for the years
        // the plan actually has, and a dormant span has none.
        let want: Vec<i32> = c.years();
        assert!(!want.is_empty());
        assert!(
            !want.iter().any(|y| (2793..=3087).contains(y)),
            "a dormant span produced years to outline"
        );

        // And the prefix tells the writer the same thing, so the arc does not flow across it.
        let p = story_prefix(&c, "");
        assert!(p.contains("HOW THIS EXISTENCE DIVIDES"));
        assert!(p.contains("do not narrate this span"));
        assert!(p.contains("should read as one"));
    }

    /// What a fork actually needs of its neighbours: the years either side, and
    /// which one is its own.
    #[test]
    fn a_year_turn_carries_a_window_of_the_outline_and_not_all_of_it() {
        let mut s = seed();
        s.born = "2719-01-01".into();
        s.through = "3087-01-01".into();
        let mut p = Plan::new(&check(&s).unwrap());
        p.story.outline = (2719..3087)
            .map(|year| YearBeat {
                year,
                title: format!("Year {year}"),
                premise: "Held.".into(),
            })
            .collect();

        let t = year_turn(&p, 2800);
        assert!(t.contains("Year 2800"), "its own beat is missing: {t}");
        assert!(t.contains("<- this one"), "its own beat is unmarked: {t}");
        assert!(t.contains("Year 2797"), "the years before are missing");
        assert!(t.contains("Year 2803"), "the years after are missing");
        assert!(!t.contains("Year 2750"), "the whole outline came along");
        assert!(!t.contains("Year 2850"), "the whole outline came along");
    }

    /// The turn carries the years it wants, the arc they follow, and the tail of
    /// what has already been written — which is what lets a year follow from the
    /// ones before it rather than being invented beside them.
    #[test]
    fn an_outline_turn_names_its_run_and_what_came_before() {
        let recent = vec![YearBeat {
            year: 2750,
            title: "The Floor".into(),
            premise: "You were tested.".into(),
        }];
        let t = outline_turn("Rook", "You read ground.", &recent, &[2751, 2752]);

        assert!(t.contains("You read ground."), "the arc is missing: {t}");
        assert!(t.contains("2750 | The Floor"), "the tail is missing: {t}");
        assert!(t.contains("Carry that forward"));
        assert!(t.contains("2751, 2752"), "the run is not named: {t}");
        assert!(t.contains("Write the next 2 year(s)"), "{t}");
    }

    /// The first run has no tail, and has to say so rather than opening on an
    /// empty heading that reads as a section the model failed to receive.
    #[test]
    fn the_first_outline_turn_says_the_outline_is_unstarted() {
        let t = outline_turn("Rook", "You read ground.", &[], &[2751]);
        assert!(t.contains("has not been started"), "{t}");
        assert!(!t.contains("Carry that forward"));
    }

    /// The turn shows the two sections as themselves rather than describing them
    /// as steps. Described, a model narrates the description back — "Then, on a
    /// line of its own:" arrived at the top of a life story, with every heading
    /// quoted as code beneath it.
    #[test]
    fn the_story_turn_shows_the_sections_rather_than_narrating_them() {
        let mut s = seed();
        s.display = "Rook".into();
        let t = story_turn(&check(&s).unwrap());

        assert!(
            t.contains("\n## Cast\n"),
            "the heading is not shown as one: {t}"
        );
        assert!(
            !t.contains('`'),
            "a backtick in the turn is a backtick in the answer: {t}"
        );
        assert!(
            !t.contains("on a line of its own"),
            "the turn still describes its own steps, which get narrated back: {t}"
        );
    }

    /// The month rung shows its section the same way the story rung does, and
    /// for the same reason: a described heading comes back quoted, and a quoted
    /// heading is not a heading, so no day is ever written from the list.
    #[test]
    fn the_month_turn_shows_its_section_rather_than_narrating_it() {
        let mut p = Plan::new(&check(&seed()).unwrap());
        if let Some(y) = p.years.first_mut() {
            y.content.text = "You held the line.".into();
        }
        let year = p.years.first().map(|y| y.year).unwrap_or_default();
        let t = month_turn(&p, year, 3);

        assert!(
            t.contains("\n## Days\n"),
            "the heading is not shown as one: {t}"
        );
        assert!(
            !t.contains("on a line of its own"),
            "the turn describes its own steps, which get narrated back: {t}"
        );
        assert!(
            t.contains("SECOND person"),
            "the register is not restated: {t}"
        );
    }

    /// The turn is written *about* the character throughout — it has to be — so
    /// the register has to be restated in it, and last, or the answer comes back
    /// as a biography. It did: "She was ground, position, and the reading of a
    /// room."
    #[test]
    fn the_story_turn_ends_by_naming_the_second_person() {
        let t = story_turn(&check(&seed()).unwrap());

        assert!(t.contains("SECOND person"), "{t}");
        let tail = t.trim_end();
        let last = tail.lines().last().unwrap_or_default();
        assert!(
            last.contains("diary") || last.contains("SECOND person"),
            "the register is not the last thing read, so it loses: {last}"
        );
    }

    /// A fork's turn carries the parent text, which is exactly what must NOT be
    /// in the prefix.
    #[test]
    fn a_month_turn_carries_its_own_years_text_and_names_the_month() {
        let mut p = plan();
        p.content_mut(NodeId::Year { year: 1999 })
            .unwrap()
            .generated("The Quiet Year".into(), "You learned to wait.".into());
        let t = month_turn(&p, 1999, 1);
        assert!(t.contains("You learned to wait."));
        assert!(t.contains("Write January of 1999"));
        assert!(t.contains("## Days"));
    }

    #[test]
    fn a_year_turn_carries_the_beat_the_story_assigned_it() {
        let t = year_turn(&plan(), 1999);
        assert!(t.contains("Write the year 1999"));
        assert!(t.contains("The Quiet Year | Nothing arrives."));
    }

    /// **The inversion.** The consequences are an input to the day, rendered as
    /// direction to a writer rather than as dials nobody has a scale for.
    #[test]
    fn a_day_turn_asks_the_prose_to_earn_its_consequences() {
        let mut p = plan();
        p.content_mut(NodeId::Month {
            year: 1999,
            month: 1,
        })
        .unwrap()
        .generated("January".into(), "The month was cold.".into());
        let d = p.ensure_day(1999, 1, 14).unwrap();
        d.content.title = "The Argument".into();
        d.consequences = vec![
            cons(
                "form_belief",
                json!({"statement": "Hess burned the east granary", "confidence": 0.95}),
            ),
            cons(
                "revise_relationship",
                json!({"entity_id": "hess", "trust": -0.8}),
            ),
        ];
        let t = day_turn(&p, 1999, 1, 14);

        assert!(t.contains("The month was cold."), "carries its month");
        assert!(t.contains("Write the 14th of January 1999"));
        assert!(t.contains("the day the month calls \"The Argument\""));
        assert!(t.contains("THIS DAY MUST PRODUCE"));
        assert!(t.contains("\"Hess burned the east granary\" — and they are certain of it."));
        assert!(t.contains("this day costs them their trust in hess"));
        assert!(t.contains("Do not announce them; earn them."));
        // No raw dials reach the writer — a model handed -0.8 has to invent a
        // scale, and two prompts would invent different ones.
        assert!(!t.contains("-0.8") && !t.contains("0.95"));
    }

    /// A day with nothing required still gets written; the section is simply
    /// absent rather than empty.
    #[test]
    fn a_day_with_no_required_consequences_omits_the_section() {
        let mut p = plan();
        p.ensure_day(1999, 1, 3).unwrap();
        let t = day_turn(&p, 1999, 1, 3);
        assert!(!t.contains("THIS DAY MUST PRODUCE"));
        assert!(t.contains("Write the 3rd of January 1999"));
    }

    #[test]
    fn ordinals_read_the_way_a_person_says_them() {
        let cases = [
            (1, "1st"),
            (2, "2nd"),
            (3, "3rd"),
            (4, "4th"),
            (11, "11th"),
            (12, "12th"),
            (13, "13th"),
            (21, "21st"),
            (22, "22nd"),
            (31, "31st"),
        ];
        for (d, want) in cases {
            assert_eq!(ordinal(d), want);
        }
    }

    /// **The turn names its subject.** Without it a thinking model spends its reasoning on the
    /// shape of the answer and then invents the person — it produced a woman called Elara
    /// Vance on Mars when asked for Keeper.
    #[test]
    fn the_story_turn_names_who_it_is_about() {
        let c = check(&seed()).unwrap();
        let t = story_turn(&c);
        assert!(t.contains("Write the life of Cindy Tan, a records clerk"));
        assert!(t.contains("the character described above"));
        assert!(t.contains("Do not invent a different person"));
        assert!(t.contains("the arc of Cindy Tan's life"));
    }

    /// A catalog tool with no bespoke rendering still reaches the writer named
    /// and with its arguments, rather than silently not being asked for.
    #[test]
    fn every_catalog_tool_renders_as_an_instruction() {
        for t in crate::engine::authoring::CATALOG {
            let parsed =
                crate::engine::authoring::parse(&format!("<tool_call>{}</tool_call>", t.example));
            let c = Consequence {
                tool: parsed.calls[0].tool.to_string(),
                args: parsed.calls[0].args.clone(),
            };
            let d = describe(&c);
            assert!(!d.trim().is_empty(), "{} renders as nothing", t.name);
            assert!(!d.contains("0."), "{} leaked a raw dial: {d}", t.name);
        }
    }
}
