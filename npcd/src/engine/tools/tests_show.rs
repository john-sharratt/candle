//! Tests that the prompt shows exactly the acts the grammar offers, and that a
//! routed act is described where its address is offered.
//!
//! The prompt must name every act and both reflection answers in one collection,
//! and a schema without that collection is refused rather than skipped. What a turn
//! shows moves with the room and replaces what it showed before, so a reflection
//! never still shows the dream from a turn earlier. A routed act is absent from the
//! grammar even at its station, and it is described only where `invoke` offers its
//! address.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// A schema with an empty `tools` collection, authored the wrong way round
    /// on purpose — `install` has to make it named. The frame section is there
    /// because a system prompt of nothing but an empty collection is refused by
    /// the builder before `install` ever sees it.
    pub(in super::super) const PROMPT: &str =
        "system_prompt:\n  items:\n    - kind: section\n      id: frame\n      \
                          content: hello\n    - kind: collection\n      name: tools\n      \
                          selection: { kind: always_visible }\n      sections: []\nlayers: []\n";

    /// **Every act, and both of the reflection's answers, go into one
    /// collection**, and it ends up selected by name whatever the schema said.
    /// A name shared between the two lists would fail the install, which is
    /// why the reflection's calls are checked here too.
    #[test]
    fn every_tool_installs_as_a_member_of_the_tools_collection() {
        use candle_conversation::projection::SystemPromptItem;
        let mut b = Builder::from_yaml(PROMPT).unwrap();
        let all: Vec<&Tool> = CATALOG
            .iter()
            .chain(crate::engine::reflect::ASKED)
            .collect();
        let mut counted = Vec::new();
        let n = install(&mut b, all.iter().copied(), |i, _| counted.push(i)).unwrap();
        assert_eq!(n, all.len());
        assert_eq!(counted, (1..=all.len()).collect::<Vec<_>>());
        for t in &all {
            assert!(
                b.id_for_system_section(&member(t.name)).is_some(),
                "`{}` was not installed",
                t.name
            );
        }
        let rule = b.schema().system_prompt.items.iter().find_map(|i| match i {
            SystemPromptItem::Collection(c) if c.name == COLLECTION => Some(c.selection.clone()),
            _ => None,
        });
        assert!(
            matches!(
                rule,
                Some(SelectionRule::Offered { ref selector, k }) if selector == COLLECTION && k == SHOWN_ACTS
            ),
            "{rule:?}"
        );
        let mandatory: Vec<_> = b
            .schema()
            .system_prompt
            .items
            .iter()
            .find_map(|i| match i {
                SystemPromptItem::Collection(c) if c.name == COLLECTION => Some(c.mandatory.len()),
                _ => None,
            })
            .into_iter()
            .collect();
        let always = all.iter().filter(|t| shown_wherever_offered(t)).count();
        assert_eq!(mandatory, [always], "the acts offered in one place only");
    }

    /// The lift's acts, a granted act and a routed one are described wherever
    /// they are offered; an act offered everywhere competes for its place.
    #[test]
    fn an_act_offered_in_one_place_is_always_shown() {
        for name in ["lift_use", "lift_call", "terminal_connect", "report_done"] {
            let t = by_name(name).unwrap_or_else(|| panic!("no `{name}`"));
            assert!(shown_wherever_offered(t), "`{name}` could be left out");
        }
        for name in ["move_to", "scan", "message"] {
            let t = by_name(name).unwrap_or_else(|| panic!("no `{name}`"));
            assert!(!shown_wherever_offered(t), "`{name}` is always shown");
        }
    }

    /// A turn that forces one answer pins exactly that definition, whatever the
    /// room offers.
    #[test]
    fn showing_pins_the_definitions_it_names() {
        let mut sel = SelectionState::new();
        show(&mut sel, ["reflection", "dream"]);
        assert_eq!(
            sel.get(FORCE_TOOL_SELECTOR),
            Some(format!("{},{}", member("reflection"), member("dream")).as_str())
        );
        let mut room = SelectionState::new();
        show_within(&mut room, Mode::Physical, &Within::nowhere());
        assert_eq!(
            room.get(FORCE_TOOL_SELECTOR),
            None,
            "the room offers, it does not pin"
        );
    }

    /// Without the collection there is nowhere to put them, and a character
    /// held to acts it has never read about is the failure this exists to end
    /// — so it is refused rather than skipped.
    #[test]
    fn a_schema_without_the_collection_is_refused() {
        let yaml = "system_prompt:\n  items:\n    - kind: section\n      id: frame\n      \
                    content: hello\nlayers: []\n";
        let mut b = Builder::from_yaml(yaml).unwrap();
        assert!(install(&mut b, CATALOG.iter(), |_, _| {}).is_err());
    }

    /// **What a turn shows is exactly what its grammar offers**, and it moves
    /// with the room: alone there is nobody to `tell` anything to, so `tell` is
    /// not shown; with company it is. `reflect` needs nothing, so it is always
    /// there.
    #[test]
    fn a_turn_shows_exactly_the_acts_its_grammar_offers() {
        let shown = |within: &Within| {
            let mut sel = SelectionState::new();
            show_within(&mut sel, Mode::Physical, within);
            sel.members(COLLECTION).to_vec()
        };
        for within in [Within::nowhere(), Within::among(&["Maker-02"])] {
            let offered: Vec<String> = specs_within(Mode::Physical, &within)
                .iter()
                .map(|s| member(&s.name))
                .collect();
            assert_eq!(shown(&within), offered);
        }
        let alone = shown(&Within::nowhere());
        let company = shown(&Within::among(&["Maker-02"]));
        assert!(!alone.contains(&member("tell")), "{alone:?}");
        assert!(company.contains(&member("tell")), "{company:?}");
        assert!(alone.contains(&member("reflect")) && company.contains(&member("reflect")));
    }

    /// Showing is a replacement, not an addition: a reflection turn that shows
    /// `dream` does not still show the `reflection` it asked for a turn ago.
    #[test]
    fn showing_replaces_what_was_shown() {
        let mut sel = SelectionState::new();
        show(&mut sel, ["reflection"]);
        show(&mut sel, ["dream"]);
        assert_eq!(sel.members(COLLECTION), &[member("dream")]);
    }

    /// An entry names every parameter with what it takes — the half of a tool
    /// the grammar cannot teach — and says which are optional.
    #[test]
    fn an_entry_names_every_parameter_and_what_it_takes() {
        for t in CATALOG.iter() {
            let e = entry(t);
            assert!(e.starts_with(&one_line(t)), "{e}");
            for p in t.params {
                assert!(
                    e.contains(&format!("{}: {}", p.name, p.description))
                        || e.contains(&format!("{} (optional): {}", p.name, p.description)),
                    "`{}` does not describe `{}`",
                    t.name,
                    p.name
                );
            }
            assert!(
                !e.ends_with('\n'),
                "the glue separates entries, not the entry"
            );
        }
        let tell = entry(by_name("tell").unwrap());
        assert!(tell.contains("\n  manner (optional): "), "{tell}");
        assert!(tell.contains("\n  intent: "), "{tell}");
    }

    /// **A migrated act leaves the grammar even where its station stands.**
    /// `report_done` is `routed` to the effector device, so a body standing
    /// at the command table — `within.station` naming it — is still not offered
    /// it as a compiled call. It reaches missions through `invoke` on the
    /// command table now, which is the whole of the Step 6 cut.
    #[test]
    fn a_routed_act_is_absent_from_the_grammar_at_its_station() {
        let mut at_table = Within::nowhere();
        // What `sim.station_tools` would report at the command table: the acts its
        // parts carry, by name — including the migrated mission acts.
        at_table.station = vec!["report_done".to_string(), "report_stuck".to_string()];
        // `invoke` is offered only when there is an address to name, so give it
        // the one the device would list for the command table.
        at_table.invokable = vec![Invokable::new(
            "http://local/command/order-table~0/report_done",
            "report_done",
        )];
        let offered: Vec<String> = specs_within(Mode::Physical, &at_table)
            .iter()
            .map(|s| s.name.clone())
            .collect();
        for migrated in ["report_done", "report_stuck"] {
            assert!(
                !offered.iter().any(|n| n == migrated),
                "`{migrated}` is routed to the device and must not be in the grammar: {offered:?}"
            );
            assert!(routed(migrated), "`{migrated}` should be marked routed");
        }
        // And `invoke` — the fixed device surface — is there.
        assert!(offered.iter().any(|n| n == "invoke"));
    }

    /// **An act only `invoke` reaches is still read about.** At the command table
    /// the grammar offers the addresses of `report_done` and `report_stuck`, so
    /// the turn's prompt must carry their entries — and must not, away from the
    /// table, where no address is offered.
    #[test]
    fn a_turn_describes_the_routed_acts_whose_addresses_it_offers() {
        let shown = |within: &Within| {
            let mut sel = SelectionState::new();
            show_within(&mut sel, Mode::Physical, within);
            sel.members(COLLECTION).to_vec()
        };
        let mut at_table = Within::nowhere();
        at_table.invokable = vec![
            Invokable::new(
                "http://local/command/order-table~0/report_done",
                "report_done",
            ),
            Invokable::new(
                "http://local/command/order-table~0/report_stuck",
                "report_stuck",
            ),
        ];
        let here = shown(&at_table);
        assert!(here.contains(&member("report_done")), "{here:?}");
        assert!(here.contains(&member("report_stuck")), "{here:?}");
        assert!(!here.contains(&member("order_post")), "{here:?}");
        let away = shown(&Within::nowhere());
        assert!(!away.contains(&member("report_done")), "{away:?}");
    }

    /// A routed act's entry says how it is reached, because its name is not a
    /// call the grammar will take.
    #[test]
    fn a_routed_acts_entry_says_it_is_reached_with_invoke() {
        let e = entry(by_name("report_done").unwrap());
        assert!(
            e.starts_with(&one_line(by_name("report_done").unwrap())),
            "{e}"
        );
        assert!(e.contains("reached with `invoke`"), "{e}");
        assert!(!entry(by_name("tell").unwrap()).contains("reached with `invoke`"));
    }
}
