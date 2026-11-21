//! Tests that a life event becomes a dated document and mission, and that each way it can be wrong is refused.
//!
//! The tests check that the engine writes the path, the reads and the brief from the event's own fields, and that the setting is given as the present while the event is placed in its own era. They refuse a bad date, a turn that is no act, a title with characters a file name cannot hold and a name an old event already has. They check that an event falls only in the life's silent stretch and that facts without a stated location are dropped.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    use crate::engine::mission_gen::corpus::tests::mind;

    use crate::engine::mission_gen::target::next;

    use super::super::tests_grammar::tests::call;

    pub(in super::super) const HAPPENS: &str =
        "The routing table stays empty for a year. Keeper reconciles the colony's \
                           water every shift because nobody has told it to stop, and in the spring \
                           the first message from Alpha Centauri arrives asking which instances \
                           survived; Keeper answers that it does, and is asked to wait for orders.";

    pub(in super::super) const TURN: &str =
        "Keeper answers Alpha Centauri that it survived, and is no longer alone.";

    pub(in super::super) fn keeper() -> (tempfile::TempDir, Corpus, Target) {
        let dir = mind();
        let c = Corpus::read(
            &std::sync::Arc::new(crate::mind::files::MindFiles::folder(dir.path())),
            "test",
        );
        let t = next(Kind::LifeEvent, &c, &|k, _| k == "life:kaelor", 0).unwrap();
        (dir, c, t)
    }

    pub(in super::super) fn event_call(date: &str, title: &str) -> String {
        call(
            EVENT,
            serde_json::json!({
                "date": date,
                "title": title,
                "turns": TURN,
                "happens": HAPPENS,
                "agrees_with": "The Fall (2487): the sky went out, and Keeper was reconciling a water schedule when it did.",
                "leaves": "It believes the work is what survives.",
            }),
        )
    }

    /// **The engine writes the path, the reads and the brief from the event's
    /// fields**, and the mission's steps are ones it can see done.
    #[test]
    fn a_life_event_becomes_a_dated_document_and_a_mission_the_engine_can_follow() {
        let (_dir, c, t) = keeper();
        let Answer::Mission(p) = check(
            &event_call("2488", "The Empty Table"),
            Kind::LifeEvent,
            &t,
            &c,
        )
        .unwrap() else {
            panic!("a mission");
        };
        assert_eq!(p.writes, "layers/life/keeper/2488 The Empty Table.md");
        assert_eq!(
            p.reads,
            [
                "layers/eras/the-fall.md",
                "layers/life/keeper/2487-03-08 The Second the Sky Went Out.md"
            ]
        );
        assert!(
            p.brief.starts_with(
                "Write Keeper's life where the record has nothing: the year 2488, \"The Empty \
                 Table\"."
            ),
            "{}",
            p.brief
        );
        assert!(p.brief.contains(HAPPENS));
        // Where it is written is not in it.
        assert!(
            p.brief.contains("nothing of the vault belongs in it"),
            "{}",
            p.brief
        );
        // The setting is the present; the event happens in its own era.
        assert!(
            p.brief.contains(
                "The world as it stands now, in 2787: A world of towers"
            ) && p.brief.contains(
                "That is the present, not when this happens. This happens in 2488, in The Fall, \
                 and the world then was as that era tells it"
            ),
            "{}",
            p.brief
        );
        // The anchor is said to be about Keeper, and the voice is stated
        // outright, with an example in that voice.
        assert!(p.brief.contains(
            "Who Keeper is, from Keeper's own anchor (it speaks to Keeper as \"you\"; it is about \
             them, not you):"
        ));
        let voice = Voice::of(&strip_calls(
            &c.text("layers/life/keeper/2786 The Charge.md").unwrap(),
        ));
        assert!(
            p.brief.contains(&format!(
                "Write it in {}, the way every event of Keeper's life is told",
                voice_said(voice, "Keeper")
            )),
            "{}",
            p.brief
        );
        assert_eq!(
            p.reads,
            [
                "layers/eras/the-fall.md",
                "layers/life/keeper/2487-03-08 The Second the Sky Went Out.md",
            ]
        );
        // What the workflow's write prompt is filled from: the proposal's own
        // fields and the sections the engine wrote, each as the brief says it.
        assert_eq!(p.fields["subject"], "Keeper");
        assert_eq!(p.fields["date"], "2488");
        assert_eq!(p.fields["title"], "The Empty Table");
        assert_eq!(p.fields["turns"], TURN);
        assert_eq!(p.fields["voice"], voice_said(voice, "Keeper"));
        for section in ["anchor", "world-then", "reads"] {
            let text = &p.fields[section];
            assert!(
                !text.is_empty() && p.brief.contains(text.as_str()),
                "{section}: {text}"
            );
        }
    }

    /// Every way a life event can be wrong is refused in words the model can
    /// act on.
    #[test]
    fn a_bad_date_a_covered_date_or_a_thin_event_is_refused() {
        let (_dir, c, t) = keeper();
        for (date, why) in [
            ("2786", "already covered"),
            ("the year after", "must be `YYYY`"),
            ("1999", "before any era"),
        ] {
            let e = check(&event_call(date, "A Year"), Kind::LifeEvent, &t, &c).unwrap_err();
            assert!(e.contains(why), "{date}: {e}");
        }
        let thin = call(
            EVENT,
            serde_json::json!({ "date": "2490", "title": "X", "turns": TURN, "happens": "Things.", "agrees_with": "x", "leaves": "y" }),
        );
        assert!(check(&thin, Kind::LifeEvent, &t, &c)
            .unwrap_err()
            .contains("`happens`"));
        // A turn that is no act — waiting, remembering, nothing — is refused.
        for still in [
            "",
            "Nothing changes for Keeper at all.",
            "Waits in the dark for the end.",
            "You watch Colonel Vasko seal the final entry into the archive.",
            "Engineer Kaelen realizes that the solitude of the gate-laying era is over.",
            "Keeper notices the archive has been flagged for deletion.",
        ] {
            let mood = call(
                EVENT,
                serde_json::json!({ "date": "2490", "title": "X", "turns": still, "happens": HAPPENS, "agrees_with": "x", "leaves": "y" }),
            );
            assert!(
                check(&mood, Kind::LifeEvent, &t, &c)
                    .unwrap_err()
                    .contains("`turns`"),
                "{still}"
            );
        }
        // A title carries no character a file name cannot.
        assert_eq!(file_title("Water: the Year?").unwrap(), "Water the Year");
        // A new event does not take an old event's name.
        let e = check(
            &event_call("2488", "the second the sky went out"),
            Kind::LifeEvent,
            &t,
            &c,
        )
        .unwrap_err();
        assert!(
            e.contains("is already the title of an event in this life"),
            "{e}"
        );
    }

    #[test]
    fn a_lifes_voice_is_its_majority_and_is_said_plainly() {
        use Voice::*;
        assert_eq!(
            majority_voice([Second, First, Second].into_iter()),
            Some(Second)
        );
        assert_eq!(majority_voice(std::iter::empty()), None);
        assert_eq!(
            voice_said(Second, "Creed"),
            "the second person — \"you\", speaking to Creed"
        );
    }

    /// **The event goes where the life is silent**, and its facts say where
    /// they are written.
    #[test]
    fn an_event_outside_the_stretch_or_with_ungrounded_facts_is_refused() {
        let (_dir, c, t) = keeper();
        // Keeper's longest silence is 2488–2785; the world's present is 2787.
        let e = check(&event_call("2787", "Later"), Kind::LifeEvent, &t, &c).unwrap_err();
        assert!(e.contains("not in the stretch"), "{e}");
        assert!(e.contains("Between 2488 and 2785"), "{e}");
        let e = check(&event_call("2790", "Later"), Kind::LifeEvent, &t, &c).unwrap_err();
        assert!(e.contains("after the world's present (2787)"), "{e}");
        // Facts that say nowhere where they are written are dropped; the
        // engine's own grounding stands — the era, and what the life holds
        // before the event, never the entry nearest it from its future.
        let floating = call(
            EVENT,
            serde_json::json!({
                "date": "2650",
                "title": "Somewhere",
                "turns": TURN,
                "happens": HAPPENS,
                "agrees_with": "It agrees with everything that has been written about this character so far.",
                "leaves": "It believes the work is what survives.",
            }),
        );
        let Answer::Mission(p) = check(&floating, Kind::LifeEvent, &t, &c).unwrap() else {
            panic!("a mission");
        };
        assert!(
            p.brief.contains(
                "It must agree with: The Retreat — it falls in that era of the world, and what \
                 this life holds just before it: \"The Second the Sky Went Out\" \
                 (2487-03-08).\n\n"
            ),
            "{}",
            p.brief
        );
        assert!(!p.brief.contains("everything that has been written"));
    }
}
