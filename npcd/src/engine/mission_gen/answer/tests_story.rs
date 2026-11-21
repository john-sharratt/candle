//! Tests that corrections quote both documents exactly, that possessives resolve to their owner, and that a story's invented names are kept new.
//!
//! A correction is refused unless both quotes appear in their documents, since a contradiction the model cannot quote is one it imagined. A possessive names the person it belongs to. A name two stories already use is refused for a story's cast, while a name from the record and a fresh name are both accepted. A story's quotes are checked against its era, and a quote given without quote marks still counts.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    use crate::engine::mission_gen::corpus::tests::mind;

    use crate::engine::mission_gen::target::next;

    use super::super::tests_grammar::tests::call;

    use super::super::tests_event::tests::{HAPPENS, TURN};

    /// **A contradiction the model cannot quote is one it imagined.** Both
    /// quotes must be in their documents; a correction reads both sides.
    #[test]
    fn a_correction_must_quote_both_documents_exactly() {
        let dir = mind();
        let c = Corpus::read(
            &std::sync::Arc::new(crate::mind::files::MindFiles::folder(dir.path())),
            "test",
        );
        let t = next(Kind::Contradiction, &c, &|_, _| false, 0).unwrap();
        let correction = |qa: &str, wrong: &str| {
            call(
                CORRECTION,
                serde_json::json!({
                    "quote_a": qa,
                    "quote_b": "Everyone went underground.",
                    "wrong": wrong,
                    "why": "The fall is when the sky went out, and the retreat cannot precede it.",
                    "change": "Replace the sentence with the corrected order of events.",
                }),
            )
        };
        let Answer::Mission(p) = check(
            &correction("The **sky** went out over every world", "B"),
            Kind::Contradiction,
            &t,
            &c,
        )
        .unwrap() else {
            panic!("a mission");
        };
        assert_eq!(p.writes, "layers/eras/the-retreat.md");
        assert_eq!(
            p.reads,
            ["layers/eras/the-fall.md", "layers/eras/the-retreat.md"]
        );
        assert!(p
            .brief
            .contains("`layers/eras/the-fall.md` says: \"The **sky** went out over every world\""));
        let e = check(
            &correction("The sky stayed lit all year.", "A"),
            Kind::Contradiction,
            &t,
            &c,
        )
        .unwrap_err();
        assert!(e.contains("is not in Document A"), "{e}");
        assert!(
            check(&correction("Short.", "A"), Kind::Contradiction, &t, &c)
                .unwrap_err()
                .contains("too short")
        );
        assert_eq!(
            check(
                &call(
                    NOTHING,
                    serde_json::json!({ "why": "The two eras agree on every date." })
                ),
                Kind::Contradiction,
                &t,
                &c
            ),
            Ok(Answer::Nothing("The two eras agree on every date.".into()))
        );
    }

    /// A possessive names the person it belongs to: "Korr's" is Korr.
    #[test]
    fn a_possessive_is_the_name_it_belongs_to() {
        assert_eq!(
            names_in("Korr's rifle and Korr, with Elara’s map; Al's."),
            ["Korr", "Korr", "Elara"]
        );
    }

    /// **An invented person is a new person**: a name two stories already use
    /// is refused for a story's cast, the record's own people are not, and a
    /// fresh name passes.
    #[test]
    fn a_story_does_not_reuse_the_names_other_stories_invented() {
        let dir = mind();
        for (file, text) in [
            ("a.md", "# A\n\nKaelen held the line."),
            ("b.md", "# B\n\nKaelen shot first."),
        ] {
            std::fs::write(dir.path().join("layers/stories").join(file), text).unwrap();
        }
        let c = Corpus::read(
            &std::sync::Arc::new(crate::mind::files::MindFiles::folder(dir.path())),
            "test",
        );
        assert_eq!(taken_names(&c), ["Kaelen"]);
        assert_eq!(reused_names(&c, "Kaelen, a gunner; Keeper"), ["Kaelen"]);
        assert!(reused_names(&c, "Orsolya Fenn, a clerk; Keeper").is_empty());
        let t = next(Kind::Gap, &c, &|k, _| k != "era:layers/eras/the-fall.md", 0).unwrap();
        let story = call(
            STORY,
            serde_json::json!({
                "title": "The Water Count",
                "when": "the night of 8 March 2487",
                "where": "the colony's water office",
                "who": ["Kaelen, a gunner", "Ila Bren, a clerk"],
                "turns": TURN,
                "happens": HAPPENS,
                "agrees_with": ["The sky went out over every world at once."],
            }),
        );
        assert!(check(&story, Kind::Gap, &t, &c)
            .unwrap_err()
            .starts_with("Kaelen already appears in other stories"));
    }

    #[test]
    fn a_story_quotes_its_era_and_is_new() {
        let dir = mind();
        let c = Corpus::read(
            &std::sync::Arc::new(crate::mind::files::MindFiles::folder(dir.path())),
            "test",
        );
        let t = next(Kind::Gap, &c, &|k, _| k != "era:layers/eras/the-fall.md", 0).unwrap();
        let story = |title: &str, agrees: &str| {
            call(
                STORY,
                serde_json::json!({
                    "title": title,
                    "when": "the night of 8 March 2487",
                    "where": "the colony's water office",
                    "who": "the clerks on shift",
                    "turns": TURN,
                    "happens": HAPPENS,
                    "agrees_with": agrees,
                }),
            )
        };
        let Answer::Mission(p) = check(
            &story(
                "The Water Schedule",
                "The era says \"the sky went out over every world\" and so it must.",
            ),
            Kind::Gap,
            &t,
            &c,
        )
        .unwrap() else {
            panic!("a mission");
        };
        assert_eq!(p.writes, "layers/stories/the-water-schedule.md");
        assert_eq!(p.reads, ["layers/eras/the-fall.md"]);
        // The world's own account of the terms it is set among.
        assert!(
            p.brief.contains(
                "- **tower** (`layers/world/tower.md`): A tower is a mobile fortress the minds \
                 live in."
            ),
            "{}",
            p.brief
        );
        assert!(check(
            &story("The Charge", "\"The sky went out over every world\""),
            Kind::Gap,
            &t,
            &c
        )
        .unwrap_err()
        .contains("already told"));
        assert!(check(
            &story("A Night", "\"The sky was green that night.\""),
            Kind::Gap,
            &t,
            &c
        )
        .unwrap_err()
        .contains("must quote"));
        // A quote given without quote marks is still a quote.
        assert!(check(
            &story(
                "A Clerk's Night",
                "The sky went out over every world at once."
            ),
            Kind::Gap,
            &t,
            &c
        )
        .is_ok());
    }
}
