//! Tests of the pulse routes' request parsing, defaults, refusals and the catalogue
//! the console is served.
//!
//! They cover the bodies the routes accept and the defaults a bare direct line
//! takes, the refusals of a blank line or announcement, the feed's clamped limit,
//! and the fields the tools page reads. They also assert that a conditional act
//! names what it needs and that the command catalogue offered to the console
//! parses, since an operator reads a rejected command as the character ignoring
//! them.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// No `npc_id` at all is no filter — the feed's default, interlaced view.
    #[test]
    fn an_absent_npc_id_filter_is_no_filter() {
        assert_eq!(parse_npc_id_filter(None), Ok(None));
    }

    /// A real base-36 id decodes to the filter it names.
    #[test]
    fn a_valid_npc_id_decodes_to_its_filter() {
        let id = 6_817_662_845_163_923_144_u64;
        let wire = npcs::npc_id_wire(id);
        assert_eq!(parse_npc_id_filter(Some(&wire)), Ok(Some(id)));
    }

    /// **A malformed `npc_id` is reported, not silently treated as no
    /// filter.** Before this fix, a decode failure fell through to "show
    /// everything" — the opposite of what a caller asking for one character's
    /// ticks would expect from a typo.
    #[test]
    fn a_malformed_npc_id_is_reported_rather_than_ignored() {
        assert_eq!(parse_npc_id_filter(Some("abc123!")), Err("abc123!"));
        assert_eq!(parse_npc_id_filter(Some("")), Err(""));
    }

    /// A read that does not ask for everything is the owner-scoped one.
    #[test]
    fn a_read_request_defaults_to_owner_scope() {
        let q: AllQuery = serde_json::from_str("{}").expect("empty query parses");
        assert!(!q.all);
        let q: AllQuery = serde_json::from_str(r#"{"all":true}"#).expect("all parses");
        assert!(q.all);
    }

    /// A body with a speaker: `serde` fills the `Option` fields we do not send.
    fn direct_body(text: &str, speaker: Option<&str>, salience: Option<f32>) -> DirectBody {
        DirectBody {
            text: text.to_owned(),
            speaker: speaker.map(str::to_owned),
            salience,
            to: None,
        }
    }

    /// The bare case: a line and nothing else is the operator, aimed at the
    /// character it reaches, loud enough to be heard now — the route exists to
    /// reach the character, so its default rouses a waiting one rather than
    /// sitting unheard until the next scheduled thought. Everything a caller can
    /// leave out has a defined default, because most direct lines will leave all
    /// of it out.
    #[test]
    fn a_bare_direct_line_is_the_operator_aimed_at_you_and_loud_enough_to_rouse() {
        let (salience, kind) =
            direct_event(direct_body("the east door is open", None, None), "Mira")
                .expect("a line with words in it is delivered");
        assert_eq!(salience, Salience::URGENT);
        assert!(
            salience.rouses(),
            "the default must reach a waiting character"
        );
        assert_eq!(
            kind,
            EventKind::Speech {
                speaker: "Mira".to_owned(),
                text: "the east door is open".to_owned(),
                to: Addressed::You,
            }
        );
    }

    /// The influence the route exists for: a named voice reaches the character
    /// in the operator's place, so a mission-giver or a terminal can speak
    /// without the operator standing in for them.
    #[test]
    fn a_named_speaker_takes_the_operators_place() {
        let (_, kind) = direct_event(
            direct_body("come to the desk", Some("The Archivist"), None),
            "Mira",
        )
        .expect("delivered");
        let EventKind::Speech { speaker, .. } = kind else {
            panic!("a direct line is speech");
        };
        assert_eq!(speaker, "The Archivist");
    }

    /// A speaker field that is present but blank is not a voice — it falls back
    /// to the operator rather than putting an empty name in the character's ear.
    #[test]
    fn a_blank_speaker_falls_back_to_the_operator() {
        let (_, kind) =
            direct_event(direct_body("hello", Some("   "), None), "Mira").expect("delivered");
        let EventKind::Speech { speaker, .. } = kind else {
            panic!("speech");
        };
        assert_eq!(speaker, "Mira");
    }

    /// A line with nothing in it is the one body the route refuses — an empty
    /// utterance is not influence, and delivering it would put a blank speech
    /// event on the character's inbox.
    #[test]
    fn a_blank_line_is_refused() {
        assert!(direct_event(direct_body("   ", None, None), "Mira").is_none());
        assert!(direct_event(direct_body("", Some("The Archivist"), Some(0.9)), "Mira").is_none());
    }

    /// The text is trimmed before it is spoken, like a message — leading and
    /// trailing whitespace is transport, not part of what was said.
    #[test]
    fn the_line_is_trimmed() {
        let (_, kind) =
            direct_event(direct_body("  go now  ", None, None), "Mira").expect("delivered");
        let EventKind::Speech { text, .. } = kind else {
            panic!("speech");
        };
        assert_eq!(text, "go now");
    }

    /// A wire salience is clamped through the same path a constructed one is, so
    /// an out-of-range figure lands at the edge and still orders correctly — an
    /// urgent line at `2.0` preempts, it is not rejected for being over one.
    #[test]
    fn a_wire_salience_is_clamped_and_still_preempts() {
        let (salience, _) =
            direct_event(direct_body("now", None, Some(2.0)), "Mira").expect("delivered");
        assert_eq!(salience.get(), 1.0);
        assert!(salience.preempts());

        let (idle, _) =
            direct_event(direct_body("mm", None, Some(-1.0)), "Mira").expect("delivered");
        assert_eq!(idle.get(), 0.0);
        assert!(!idle.preempts());
    }

    /// The address survives from the body — a line can be thrown to the room,
    /// overheard by the character, rather than aimed at it, when the caller says
    /// so; the default is only what fills the field's absence.
    #[test]
    fn an_explicit_address_is_kept() {
        let body = DirectBody {
            text: "the hall is closing".to_owned(),
            speaker: None,
            salience: None,
            to: Some(Addressed::Room),
        };
        let (_, kind) = direct_event(body, "Mira").expect("delivered");
        let EventKind::Speech { to, .. } = kind else {
            panic!("speech");
        };
        assert_eq!(to, Addressed::Room);
    }

    /// An announcement becomes a world-scale `Announcement` event, trimmed, and
    /// renders as a word put to everyone rather than a thing in the room — the
    /// framing is the whole reason it is its own kind and not a `Description`.
    #[test]
    fn an_announcement_is_a_world_event_read_as_reaching_everyone() {
        let kind = announce_event(AnnounceBody {
            text: "  the eastern gate is sealed  ".to_owned(),
        })
        .expect("a line with words in it is announced");
        assert_eq!(
            kind,
            EventKind::Announcement {
                text: "the eastern gate is sealed".to_owned(),
            }
        );
        let prose = Event::new(0, 0, Salience::URGENT, kind).prose();
        assert_eq!(
            prose,
            "Word goes out across the world: the eastern gate is sealed"
        );
    }

    /// A blank announcement is the one body the route refuses — an empty word to
    /// the world is not an announcement, and broadcasting it would put a blank
    /// event on every character's inbox at once.
    #[test]
    fn a_blank_announcement_is_refused() {
        assert!(announce_event(AnnounceBody {
            text: "   ".to_owned()
        })
        .is_none());
        assert!(announce_event(AnnounceBody {
            text: String::new()
        })
        .is_none());
    }

    /// The feed's bound has to be enforced on the request, not trusted from it —
    /// `?limit=1000000` would otherwise render the scheduler's whole ring into
    /// one response.
    #[test]
    fn the_feed_limit_is_clamped_at_both_ends() {
        for (asked, want) in [
            (None, DEFAULT_FEED),
            (Some(0), 1),
            (Some(9_999_999), MAX_FEED),
            (Some(10), 10),
        ] {
            let got = asked.unwrap_or(DEFAULT_FEED).clamp(1, MAX_FEED);
            assert_eq!(got, want, "limit={asked:?}");
        }
    }

    /// **Every field the tools page reads is a field this route sends.**
    ///
    /// The console renders `name`, `description`, `modes`, `source`,
    /// `calibrated` and `parameters`. The route used to serialise `Tool`
    /// directly, which has none of the last four — so three of the table's five
    /// columns were blank and the schema modal showed `{}`, on a page whose
    /// whole job is to say what a character can do. Nothing failed; the page
    /// simply rendered `undefined` as empty.
    ///
    /// Asserted against the names the page actually uses, so adding a column
    /// there and forgetting this breaks a test rather than a view.
    /// The station names the shipped maps actually give, which is what an
    /// operator will read.
    fn shipped_part_names() -> PartNames {
        let set =
            npc_map::MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"))
                .expect("the shipped maps must load");
        let mut names = PartNames::new();
        for area in set.areas() {
            for node in &area.nodes {
                for (part, _) in set.parts_at(node) {
                    names
                        .entry(part.id.clone())
                        .or_insert_with(|| part.name.clone());
                }
            }
        }
        names
    }

    #[test]
    fn the_tools_route_sends_every_field_the_console_renders() {
        let v = describe_catalog(&shipped_part_names());

        assert_eq!(
            v["uncalibrated"].as_u64(),
            Some(0),
            "an act shipped with no examples"
        );
        assert!(v["acts_per_turn"].as_u64().unwrap() >= 1);

        let ts = v["tools"].as_array().expect("an array");
        assert_eq!(ts.len(), crate::engine::tools::CATALOG.len());
        for t in ts {
            for field in [
                "name",
                "category",
                "description",
                "modes",
                "source",
                "calibrated",
            ] {
                assert!(!t[field].is_null(), "`{}` has no `{field}`", t["name"]);
            }
            assert_eq!(t["parameters"]["type"], "object", "{}", t["name"]);
            assert!(t["parameters"]["properties"].is_object());
            assert!(t["parameters"]["required"].is_array());
            assert!(
                !t["modes"].as_array().unwrap().is_empty(),
                "`{}` is offered in no mode at all",
                t["name"]
            );
        }
    }

    /// The page prints a `needs` chip to explain an act's absence. An act that
    /// is conditional must therefore say what its condition is, or an operator
    /// hunting a tool a character never calls has nothing to read.
    #[test]
    fn a_conditional_act_says_what_it_needs() {
        let v = describe_catalog(&shipped_part_names());
        let ts = v["tools"].as_array().unwrap();
        let find = |n: &str| {
            ts.iter()
                .find(|t| t["name"] == n)
                .expect("in the catalog")
                .clone()
        };

        // Room speech is for people: aimed at somebody, it needs a person here
        // to aim it at; a shout, a person within its reach, or the field.
        assert_eq!(find("tell")["needs"], "a person here");
        // And a whisper needs somebody besides its listener to keep it from.
        assert_eq!(find("whisper")["needs"], "a person here, and somebody else");
        assert_eq!(
            find("shout")["needs"],
            "a person within earshot, or the field"
        );
        assert_eq!(find("gesture")["needs"], "somebody else here");
        assert!(
            find("reflect")["needs"].is_null(),
            "stopping must be available whatever else is not, or a character \
             with nothing it can do has no way to spend a turn"
        );
        assert_eq!(find("move_to")["needs"], "a body");
        assert_eq!(find("act")["needs"], "being present");

        // **A station act names the station**, in the words the map uses for
        // it. "Standing at the station that carries it" is true of every one of
        // them and answers nothing.
        assert_eq!(
            find("chronicle_add_entry")["needs"],
            "standing at a world history terminal"
        );
        assert_eq!(
            find("record_appraise")["needs"],
            "standing at the appraisal bench"
        );

        // An act reaching several says so as alternatives — you need one of
        // them, not all six.
        // **An act reaching many stations is counted, not listed.** Six noun
        // phrases in a table cell wrap the row and are unreadable at a glance,
        // and a reader made to parse all six to learn "a workstation" has been
        // given less rather than more.
        let bench = find("bench_branch");
        assert_eq!(bench["needs"], "standing at any of 6 stations");

        // The list is not lost — it is in the field the detail view renders,
        // where there is room for it, with the article each name calls for.
        let at: Vec<String> = bench["at_named"]
            .as_array()
            .expect("the full list")
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(at.len(), 6, "{at:?}");
        assert!(
            at.contains(&"a world history terminal".to_string()),
            "{at:?}"
        );
        assert!(
            at.contains(&"an easel".to_string()),
            "the article was assumed: {at:?}"
        );

        // **An act narrowed by channel reports the narrowing.** `send_image`
        // goes down a thread and not into a room — you are standing in front of
        // them, so you hold the thing up — and the catalogue the console reads
        // has to say so, or an operator is shown an act as if it worked
        // everywhere.
        let img = find("send_image");
        let modes: Vec<String> = img["modes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_str().unwrap().to_string())
            .collect();
        let modes: Vec<&str> = modes.iter().map(String::as_str).collect();
        assert_eq!(modes, vec!["instant_message"], "{modes:?}");
    }

    /// With no world open there is no map to read a name off, and the field
    /// says the honest generic thing rather than inventing one.
    #[test]
    fn a_station_with_no_map_loaded_is_not_given_an_invented_name() {
        let v = describe_catalog(&PartNames::new());
        let ts = v["tools"].as_array().unwrap();
        let needs = ts
            .iter()
            .find(|t| t["name"] == "chronicle_add_entry")
            .expect("in the catalog")["needs"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(needs.contains("chronicle-terminal"), "{needs}");
        assert!(
            needs.contains("nothing places one"),
            "an unplaced station should say so: {needs}"
        );
    }

    /// The console's autocomplete is the daemon's own list. A console holding
    /// its own copy would offer a command that gets rejected, and an operator
    /// reads a rejection as the character ignoring them.
    #[test]
    fn the_command_catalog_is_served_not_duplicated() {
        assert!(!slash::CATALOG.is_empty());
        // Everything offered must parse — the same guarantee `slash`'s own
        // tests make, asserted here because this is the surface that publishes
        // it.
        for c in slash::CATALOG {
            assert!(
                slash::parse(c.example).is_ok(),
                "/{} is published with an example that does not parse",
                c.name
            );
        }
    }
}
