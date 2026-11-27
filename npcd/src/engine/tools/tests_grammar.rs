//! Tests that the act catalogue compiles into a working grammar for every room a
//! character can stand in.
//!
//! A grammar that fails to compile does not fail loudly: the turn falls back to
//! free decoding and nothing reports it. The tests therefore compile each catalogue,
//! the empty room and a full one, in both call syntaxes, all the way to the token
//! tree. They check that required parameters are forced, that every free string is
//! bounded, that a full room and a full conversation stay inside the path budget,
//! and that every act has a description and calibration examples.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::engine::talk;

    use crate::engine::talk::key::Party;

    use crate::engine::talk::within::OnThread;

    use serde_json::{json, Value};

    use crate::engine::mission_gen::world::{Draw, Routine};
    use crate::sim::missions::Missions;

    /// **What the job chosen brings is carried with it**: a post of
    /// `coding-task` names its `what` and `repository` as fields of the call;
    /// a job that takes nothing, or none at all, brings nothing; and no other
    /// act is read for them.
    #[test]
    fn a_chosen_job_names_the_fields_it_brings() {
        let mut missions = Missions::default();
        missions.set_routines(
            vec![
                Routine {
                    id: "coding-task".into(),
                    brief: "{what} in {repository}".into(),
                    steps: vec!["see the task through".into()],
                    params: vec!["what".into(), "repository".into()],
                    ..Default::default()
                },
                Routine {
                    id: "ward-shift".into(),
                    brief: "Take a shift.".into(),
                    steps: vec!["go".into()],
                    ..Default::default()
                },
            ],
            Draw::None,
        );
        let fields = |v: Value| v.as_object().unwrap().clone();
        let coding = fields(json!({"job": "coding-task", "what": "w", "repository": "candle"}));
        assert_eq!(
            declared_fields("order_post", &coding, &missions),
            ["what", "repository"]
        );
        let shift = fields(json!({"job": "ward-shift"}));
        assert!(declared_fields("order_post", &shift, &missions).is_empty());
        let brief = fields(json!({"brief": "b", "steps": "s"}));
        assert!(declared_fields("order_post", &brief, &missions).is_empty());
        assert!(declared_fields("report_done", &coding, &missions).is_empty());
    }

    /// **A turn of plain words is offered no act**, whatever its selection
    /// listed before.
    #[test]
    fn plain_words_are_offered_no_act() {
        let mut selection = SelectionState::new();
        selection.select_all(COLLECTION, ["reply", "sign_off"]);
        show_none(&mut selection);
        assert!(selection.members(COLLECTION).is_empty());
    }

    /// **The catalog compiles into a grammar, and a required parameter becomes
    /// unskippable in it.**
    ///
    /// The property the whole stencil exists for. `tell` takes a required `to`,
    /// and under the compiled tree that field is a node the decode is forced
    /// through rather than a request it can decline — which is the difference
    /// between a character that addresses somebody and one that spends an
    /// evening being refused for a field it never learned it needed.
    #[test]
    fn a_required_parameter_is_forced_by_the_compiled_grammar() {
        use candle_conversation::stencil::{compile_tool_call_tree, ToolCallEnvelope};

        // In company, because `tell` needs somebody to speak to — alone it is not
        // in the grammar at all, which is the point of
        // [`a_character_alone_cannot_reach_the_acts_that_need_company`].
        let specs = specs_within(Mode::Physical, &Within::among(&["Maker-02"]));
        // Not the whole catalog: a room with company but nothing in it offers
        // no `gather`, no `operate`, no `produce`, because those bind required
        // arguments to sets this world has nothing in. That is the empty-set
        // rule working, so the check is that the acts *company* makes possible
        // are all here rather than that the count matches.
        // `give` is absent from this list on purpose: it binds `what` to the
        // pack, and a character carrying nothing has nothing to hand over.
        // Company is necessary for it and not sufficient.
        // `whisper` is absent too: with one other person here there is nobody
        // to keep it from — see `Availability::PersonAmongOthers`.
        for expected in ["tell", "shout", "ask", "gesture", "act", "promise"] {
            assert!(
                specs.iter().any(|s| s.name == expected),
                "`{expected}` was dropped in a room with somebody in it"
            );
        }

        let tell = specs
            .iter()
            .find(|s| s.name == "tell")
            .expect("`tell` is in the catalog");
        let to = tell
            .params
            .iter()
            .find(|p| p.name == "to")
            .expect("`tell` names an addressee");
        assert!(to.required, "the addressee stopped being required");

        // And the catalog is actually compilable — a name collision or a
        // parameter the trie rejects would otherwise surface as an engine that
        // free-decodes, which is silent.
        compile_tool_call_tree(&specs, &ToolCallEnvelope::qwen3())
            .expect("the act catalog must compile into a stencil");
    }

    /// Every free string an act takes closes at [`ACT_VALUE_TOKENS`], and
    /// nothing else is bounded: a runaway `reflect` ran to the turn's cap and the
    /// call around it was lost.
    #[test]
    fn every_string_an_act_takes_is_bounded() {
        let specs = specs_within(Mode::Physical, &Within::among(&["Perrin Vastwood"]));
        let reflect = specs.iter().find(|s| s.name == "reflect").unwrap();
        let situation = reflect
            .params
            .iter()
            .find(|p| p.name == "situation")
            .unwrap();
        assert_eq!(situation.max_tokens, Some(1024));
        for p in specs.iter().flat_map(|s| s.params.iter()) {
            let bound = match p.ty {
                ParamType::String => Some(ACT_VALUE_TOKENS),
                _ => None,
            };
            assert_eq!(p.max_tokens, bound, "`{}`", p.name);
        }
    }

    /// **A device act is offered only where its device stands, and its `device`
    /// is bound to what is there.** Every bridge act binds `device` to the live
    /// set of devices of that kind in the room, so the empty-set rule takes it
    /// out of the grammar where there is none — which is what stops a character
    /// inventing a device name for a device that is not present. Before this the
    /// cast acts (and `sonos_source`, the extended receiver acts, and the two
    /// pair acts) were free text and offered everywhere, so a cast standing in a
    /// room with no devices spent its turns casting to a TV that was not there.
    #[test]
    fn a_device_act_is_offered_only_where_its_device_is() {
        // A room with nothing in it offers none of them — the loop the cast was
        // stuck in cannot form, because there is no act to emit.
        let bare = specs_within(Mode::Physical, &Within::nowhere());
        for gone in [
            "cast_status",
            "cast_cast",
            "cast_volume",
            "sonos_play",
            "sonos_source",
            "receiver_on",
            "receiver_input",
            "pair_device",
            "pair_code",
        ] {
            assert!(
                !bare.iter().any(|s| s.name == gone),
                "`{gone}` was offered in a room with no device to work"
            );
        }

        // With the devices standing here, each act is back and its `device` is
        // the names present rather than a string the model fills in.
        let furnished = Within {
            cast: vec!["the TV".into()],
            sonos: vec!["the Sonos".into()],
            receiver: vec!["the stereo".into()],
            pairable: vec!["the LG TV".into()],
            pairing: vec!["the Google TV".into()],
            ..Within::nowhere()
        };
        let specs = specs_within(Mode::Physical, &furnished);
        let bound = |name: &str, want: &str| {
            let t = specs
                .iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("`{name}` was dropped with its device present"));
            let dev = t
                .params
                .iter()
                .find(|p| p.name == "device")
                .unwrap_or_else(|| panic!("`{name}` names no device"));
            assert_eq!(
                dev.enum_values.as_deref(),
                Some(&[want.to_string()][..]),
                "`{name}`'s device is free text, not bound to the device present"
            );
        };
        bound("cast_status", "the TV");
        bound("sonos_play", "the Sonos");
        bound("receiver_on", "the stereo");
        bound("pair_device", "the LG TV");
        bound("pair_code", "the Google TV");
    }

    /// Every parameter in the catalog has a stencil type. `param_type` refuses
    /// an unknown one rather than defaulting it, because the default is "any
    /// JSON value" — a quietly weaker constraint on the one path whose job is
    /// to be exact.
    #[test]
    fn every_parameter_type_maps_to_the_grammar() {
        for t in CATALOG.iter() {
            for p in t.params {
                // Panics on an unmapped type, which is the assertion.
                let _ = param_type(p.ty);
            }
        }
    }

    /// **Every act says what it is for, and every parameter says what goes in
    /// it.**
    ///
    /// The description and the parameter blurbs are the whole of what a model
    /// has to go on when it chooses between a hundred acts — the name is four
    /// tokens and the grammar admits them all equally. A one-line description
    /// is not a style problem: it is an act that will be chosen at random
    /// against its neighbours, and nothing anywhere reports that.
    ///
    /// The bounds are deliberately low. This is a floor against an act being
    /// added with a placeholder, not a word count anybody should be writing to.
    #[test]
    fn every_act_and_every_parameter_is_described() {
        /// Long enough to say what an act is for and when to reach for it.
        const ENOUGH: usize = 60;
        /// A parameter blurb has one job and can do it shorter.
        const ENOUGH_FOR_A_PARAM: usize = 20;

        // Collected rather than asserted one at a time: fixing these is one
        // pass over the catalogue, and a test that names the first of nine
        // turns that into nine runs.
        let mut thin: Vec<String> = Vec::new();
        for t in CATALOG.iter() {
            if t.description.len() < ENOUGH {
                thin.push(format!("{} ({} chars)", t.name, t.description.len()));
            }
            assert!(
                !t.category.is_empty(),
                "`{}` is in no category, so nothing groups it",
                t.name
            );
            for p in t.params {
                if p.description.len() < ENOUGH_FOR_A_PARAM {
                    thin.push(format!(
                        "{}.{} ({} chars)",
                        t.name,
                        p.name,
                        p.description.len()
                    ));
                }
            }
        }
        assert!(
            thin.is_empty(),
            "{} description(s) too thin to choose on: {thin:#?}",
            thin.len()
        );
    }

    /// An act's example has to actually call *that* act, or calibration teaches
    /// the model a shape belonging to a different one.
    #[test]
    fn every_example_supplies_the_acts_required_arguments() {
        for t in CATALOG.iter() {
            for e in t.examples {
                let call: serde_json::Value =
                    serde_json::from_str(e.call).expect("checked to parse elsewhere");
                let obj = call.as_object().unwrap_or_else(|| {
                    panic!("`{}`: an example call that is not an object", t.name)
                });
                for p in t.params.iter().filter(|p| p.required) {
                    assert!(
                        obj.contains_key(p.name),
                        "`{}`'s example omits the required `{}` — it prefills a call the world \
                         would refuse",
                        t.name,
                        p.name
                    );
                }
            }
        }
    }

    /// Every tool must be calibrated. An example-less tool is usable and selects
    /// measurably worse, which is exactly the kind of regression that never gets
    /// noticed because nothing errors.
    #[test]
    fn every_tool_carries_calibration_examples() {
        for t in CATALOG.iter() {
            assert!(
                !t.examples.is_empty(),
                "{} has no examples — it will select worse than its neighbours and nothing \
                 will report it",
                t.name
            );
            for e in t.examples {
                assert!(
                    !e.situation.is_empty(),
                    "{}: an example with no situation",
                    t.name
                );
                assert!(
                    !e.because.is_empty(),
                    "{}: an example that shows the shape but not the choice",
                    t.name
                );
                // The call has to parse, or calibration prefills a malformed
                // trajectory and teaches the model to emit one.
                serde_json::from_str::<serde_json::Value>(e.call).unwrap_or_else(|err| {
                    panic!("{}: example call is not JSON: {err}\n{}", t.name, e.call)
                });
            }
        }
    }

    /// **The effector device is the fixed surface, always offered.**
    ///
    /// `invoke` is a body act (in `BODY_ACTS`), `Availability::Always` so it is
    /// reachable wherever the body stands, carries calibration examples, and takes
    /// string parameters the grammar can bound. See [`INVOKE`].
    #[test]
    fn the_effector_device_offers_invoke_always() {
        let invoke = by_name("invoke").unwrap_or_else(|| panic!("`invoke` is not in the catalog"));
        assert!(
            BODY_ACTS.iter().any(|b| b.name == "invoke"),
            "`invoke` is not a body act — it must be in BODY_ACTS"
        );
        assert_eq!(
            invoke.availability,
            Availability::Always,
            "`invoke` must be offered wherever the body stands"
        );
        assert!(
            !invoke.examples.is_empty(),
            "`invoke` carries no calibration example"
        );
        for p in invoke.params {
            assert_eq!(
                p.ty, "string",
                "`invoke`.{} is `{}` — the device body is a string until the stencil shapes it",
                p.name, p.ty
            );
        }

        // `invoke` acts at one address, with an optional body.
        let url = invoke.params.iter().find(|p| p.name == "url").expect("url");
        assert!(url.required, "an address to act at is required");
        let body = invoke
            .params
            .iter()
            .find(|p| p.name == "body")
            .expect("body");
        assert!(
            !body.required,
            "the body is optional — some addresses take none"
        );
    }

    /// Every argument an example passes must be a parameter the tool declares.
    /// A typo here teaches the model to emit a field the dispatcher will reject.
    #[test]
    fn examples_only_pass_declared_parameters() {
        for t in CATALOG.iter() {
            for e in t.examples {
                let v: serde_json::Value = serde_json::from_str(e.call).unwrap();
                let obj = v.as_object().expect("an example call is an object");
                for key in obj.keys() {
                    assert!(
                        t.params.iter().any(|p| p.name == key),
                        "{}: example passes undeclared parameter {key:?}",
                        t.name
                    );
                }
                for p in t.params.iter().filter(|p| p.required) {
                    assert!(
                        obj.contains_key(p.name),
                        "{}: example omits required parameter {:?}",
                        t.name,
                        p.name
                    );
                }
            }
        }
    }

    /// **The property this whole module exists to preserve.**
    ///
    /// A speech tool takes intent. The moment one takes a finished sentence, the
    /// model is writing dialogue directly and every guarantee about acts being
    /// grounded in gathered state is worth nothing. "Just let it write the line"
    /// is always the shortest path to a demo that reads well, so this is a build
    /// failure rather than a review note.
    #[test]
    fn speech_tools_take_intent_not_prose() {
        for t in CATALOG.iter().filter(|t| t.plane == Plane::Speech) {
            let carries_intent = t
                .params
                .iter()
                .any(|p| matches!(p.name, "intent" | "about"));
            // **Or the world says the substance outright.** `remind` names one
            // of the promises that actually stand, from a set the world
            // enumerates — the model cannot write a sentence there because it
            // cannot write anything the world did not already say. That is the
            // same guarantee this test protects, arrived at from the other
            // side, and a stricter one than free intent rather than an
            // exemption from it.
            let world_says_it = t
                .params
                .iter()
                .any(|p| live_choice(t.name, p.name).is_some() && p.required);
            assert!(
                carries_intent || world_says_it,
                "{} is a speech tool with no intent-carrying parameter",
                t.name
            );
            for p in t.params {
                assert!(
                    !matches!(p.name, "text" | "sentence" | "line" | "words" | "message"),
                    "{} takes {:?} — that is output, not intent",
                    t.name,
                    p.name
                );
            }
        }
    }

    /// Every wayfinding act is a real act: a name that matched none would
    /// withhold nothing from a mind with no body, and say nothing about it.
    #[test]
    fn every_wayfinding_act_is_in_the_catalog() {
        for name in WAYFINDING {
            assert!(by_name(name).is_some(), "`{name}` is no act");
        }
    }

    /// The action plane cannot write beliefs. Not "should not" — there is no tool
    /// that does, and this is the assertion that keeps it that way when somebody
    /// reasonably proposes `revise_belief`.
    #[test]
    fn no_tool_writes_beliefs() {
        for t in CATALOG.iter() {
            assert!(
                !t.name.contains("belief"),
                "{} names beliefs — the action plane cannot write them",
                t.name
            );
            for p in t.params {
                assert!(
                    !p.name.contains("belief"),
                    "{} takes a belief parameter",
                    t.name
                );
            }
        }
        // The pressure valve must exist, or the model has nothing to do with a
        // belief under strain and will reach for something that does not exist.
        //
        // It used to be `note_concern`, which wrote nothing anywhere — a valve
        // that vents into no pipe is not a valve, it is a way to spend a turn
        // looking thoughtful. `gesture` is the real one: a belief that cannot be
        // acted on can still show, and showing reaches the room.
        assert!(by_name("gesture").is_some());
    }

    /// **The grammar has to build for every room a character can be in**, and
    /// the empty one is the room it is in most.
    ///
    /// A catalog that fails to compile does not fail loudly: `grammar_for`
    /// returns `None`, the turn free-decodes, and what comes back is a model
    /// writing "Thinking Process: 1. Analyze the Request" in prose — a
    /// successful decode by every measure the daemon has.
    ///
    /// **Compiled all the way to a tree, not just to a `TreeSpec`.** The first
    /// version of this test stopped at the spec and passed while the daemon was
    /// refusing to arm: an empty `enum` is a well-formed *spec* and an
    /// unrepresentable *node*, and only the tokenizing compile says so —
    /// "branch at node 9 has no arms". `TestVocab` costs nothing and is the
    /// difference between a test that checks the shape and one that checks the
    /// thing that actually has to work.
    #[test]
    fn the_grammar_compiles_for_an_empty_room_and_for_a_full_one() {
        use candle_conversation::stencil::{
            compile, compile_action_loop, TestVocab, ToolCallEnvelope,
        };

        // **Both call shapes, because both are shipped.** The catalog has to
        // compile into whichever syntax the loaded checkpoint's template says —
        // and the two differ structurally, not just in their strings, so a
        // catalog that is a valid grammar in one is not thereby a valid grammar
        // in the other. An empty arm, a duplicate arm, a name that tokenizes to
        // nothing: each has to be absent from both.
        for env in [ToolCallEnvelope::qwen3(), ToolCallEnvelope::qwen35()] {
            for within in [
                Within::nowhere(),
                Within::among(&["Perrin Vastwood"]),
                Within::among(&["Perrin Vastwood", "Orion Vance"]),
                // Nowhere to walk to: `move_to` has a *required* destination, so
                // the act goes rather than the parameter — the other half of the
                // empty-set rule, and the half that would otherwise leave a
                // zero-arm branch and stop the whole grammar compiling.
                Within {
                    places: Vec::new(),
                    ..Within::among(&["Perrin Vastwood"])
                },
                // **Exactly one thread, and exactly one contact.** A live set of
                // one is the ordinary state of a character that has spoken to one
                // person — and it is the state the daemon was observed failing in,
                // every four seconds, with `branch arm "none\"" tokenizes empty`
                // and the whole turn falling through to a free decode.
                Within {
                    threads: vec!["none".into()],
                    contacts: vec!["Wren Wylde".into()],
                    ..Within::among(&["Perrin Vastwood"])
                },
                // **Two threads of the same name.** A live set is a set only by
                // intention: nothing stops a character opening two groups called
                // the same thing, and two identical arms tokenize to one common
                // prefix with nothing left over, which is
                // `BuildError::EmptyArm` — the whole grammar, gone, for every turn
                // that character takes afterwards.
                Within {
                    threads: vec!["none".into(), "none".into()],
                    ..Within::among(&["Perrin Vastwood"])
                },
                // **Mid-fight, with everything physical cooling at once.** The
                // state a character reaches by throwing a punch and then walking:
                // three acts gone from the catalog in the same turn. Whatever is
                // left has to still be a grammar, or the fight ends in a free
                // decode.
                Within {
                    cooling: vec!["move_to".into(), "act".into(), "gesture".into()],
                    ..Within::among(&["Perrin Vastwood"])
                },
            ] {
                let specs = specs_within(Mode::Physical, &within);
                assert!(!specs.is_empty(), "no acts at all for {within:?}");
                // No parameter may carry an empty closed set — that is the node the
                // compile refuses, and naming it here says which act is at fault
                // rather than which node number.
                for t in &specs {
                    for p in &t.params {
                        assert!(
                            p.enum_values.as_ref().is_none_or(|v| !v.is_empty()),
                            "`{}` offers `{}` with nothing to choose, in {within:?}",
                            t.name,
                            p.name
                        );
                    }
                }
                let spec = compile_action_loop(&specs, &env, ACTS_PER_TURN, "<|im_end|>", None)
                    .unwrap_or_else(|e| {
                        panic!(
                            "the grammar will not build for {within:?} in {:?}: {e}",
                            env.style
                        )
                    });
                compile(&spec, &TestVocab::new()).unwrap_or_else(|e| {
                    panic!(
                        "the grammar will not compile for {within:?} in {:?}: {e}",
                        env.style
                    )
                });
            }
        }
    }

    /// The act that keeps its aim optional: alone, `use` stays but its `on`
    /// goes — a stimpak used on nobody is still a stimpak used.
    ///
    /// This was `gesture` until gesture came to need company. The rule it
    /// demonstrates is unchanged and is the other half of the empty-set rule: a
    /// *required* parameter with nothing to choose takes its act out of the
    /// grammar, an *optional* one takes only itself.
    #[test]
    fn an_optional_addressee_drops_rather_than_emptying_its_branch() {
        let alone = Within {
            usable: vec!["a stimpak".into()],
            ..Within::nowhere()
        };
        let specs = specs_within(Mode::Physical, &alone);
        let g = specs.iter().find(|t| t.name == "use").expect("kept");
        assert!(
            g.params.iter().all(|p| p.name != "on"),
            "an aim at nobody: {:?}",
            g.params.iter().map(|p| &p.name).collect::<Vec<_>>()
        );
        assert!(
            g.params.iter().any(|p| p.name == "what"),
            "still uses something"
        );

        let with = Within {
            usable: vec!["a stimpak".into()],
            ..Within::among(&["Perrin Vastwood"])
        };
        let together = specs_within(Mode::Physical, &with);
        let g = together.iter().find(|t| t.name == "use").unwrap();
        let to = g
            .params
            .iter()
            .find(|p| p.name == "on")
            .expect("aimable again");
        assert_eq!(
            to.enum_values.as_deref(),
            Some(&["Perrin Vastwood".to_string()][..])
        );
    }

    /// **No parameter may declare a type the grammar cannot bound.**
    ///
    /// The stencil compiles `string` to a terminated span, `boolean` to a
    /// two-arm branch, an `enum` to a closed branch over its values, and an
    /// `integer` to digits unrolled to
    /// [`candle_conversation::stencil::MAX_INTEGER_DIGITS`] with nothing but a
    /// digit admitted. For `number`, `array` and `object` it compiles *any
    /// structurally-valid JSON value* — and JSON nests without limit, so
    /// enumerating those states never terminates.
    ///
    /// This is not a style rule. A catalog `number` or `array` made
    /// [`tests::the_grammar_compiles_for_an_empty_room_and_for_a_full_one`]
    /// allocate about 390 MB every second without bound, which exhausts a
    /// 64 GB machine's RAM and page file in roughly three minutes and takes
    /// down whatever else is running with it. It cost two crashed editors
    /// before it was understood. The path budget below is what keeps the
    /// integers honest.
    #[test]
    fn every_parameter_is_a_type_the_grammar_can_bound() {
        for tool in CATALOG.iter() {
            for p in tool.params {
                assert!(
                    matches!(p.ty, "string" | "boolean" | "integer"),
                    "`{}` declares `{}` as `{}` — the grammar cannot bound that type, and \
                     compiling it does not terminate. Carry it as a string and parse it in the \
                     act.",
                    tool.name,
                    p.name,
                    p.ty,
                );
            }
        }
    }

    /// **The grammar a busy room compiles must stay inside its path budget.**
    ///
    /// The guard that stands between a catalog change and a machine that stops
    /// responding. The decodes a turn admits are exponential in
    /// [`ACTS_PER_TURN`]. Adding one act with three enumerated arguments took a
    /// turn from about four thousand paths to 10¹¹, and the compile then
    /// allocated without bound until the machine died.
    ///
    /// The worst case is not the empty room: it is a character with company to
    /// address, somewhere to walk, things to carry and things to fight, because
    /// every one of those fills a live set that was empty before.
    #[test]
    fn a_full_room_stays_inside_the_path_budget() {
        let mut within = Within::among(&["Perrin Vastwood", "Orion Vance"]);
        within.carried = vec!["mono sword".into(), "stimpak".into(), "bolt rounds".into()];
        within.equippable = vec!["mono sword".into()];
        within.usable = vec!["stimpak".into()];
        within.operable = vec!["the blast door".into(), "wall turret 1".into()];
        within.device_modes = crate::sim::device::modes::TURRET
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        within.hostiles = vec!["a drone".into(), "a mech".into()];
        within.extractable = vec!["the ore seam".into()];
        within.makeable = vec!["bolt rounds".into(), "stimpaks".into()];
        within.queues = (1..=8).map(|q| q.to_string()).collect();
        within.tower_actions = vec!["relocate".into(), "siege".into(), "drill down".into()];
        within.claimable = vec!["fabricator 1".into()];
        within.readable = vec!["the muster board".into()];
        within.owed = vec!["the eastern span".into()];
        // **The widest closed set in the catalog by a factor of ten.** The
        // shipped mind holds a hundred and sixteen moods, and `pause` offers
        // every one — so this is the arm most likely to be the one that puts a
        // future change over the bound.
        within.feelings = (0..116).map(|i| format!("mood-{i}")).collect();

        let specs = specs_within(Mode::Physical, &within);
        let paths = estimated_paths(&specs, ACTS_PER_TURN);
        assert!(
            paths <= MAX_TURN_PATHS,
            "a turn in a busy room admits {paths} decodes, over the {MAX_TURN_PATHS} budget — \
             the compile is exponential in this and will allocate until the machine dies. \
             Fewer enumerated arguments per act, or fewer acts per turn."
        );
    }

    /// **A conversation's decision stays inside the same budget at
    /// [`TALK_ACTS`] calls**, with every act of the thread catalog offered:
    /// an authority over the agenda, somebody who owes, a full phone and the
    /// whole mood library.
    #[test]
    fn a_full_thread_stays_inside_the_path_budget() {
        let mut zen = Within::nowhere();
        zen.feelings = (0..116).map(|i| format!("mood-{i}")).collect();
        zen.private = (0..12).map(|i| format!("friend-{i}")).collect();
        zen.channels = vec!["the channel".into(), "#research".into()];
        zen.threads = zen.private.iter().chain(&zen.channels).cloned().collect();
        zen.contacts = (0..12).map(|i| format!("contact-{i}")).collect();
        zen.agenda = (0..8).map(|i| format!("item-{i}")).collect();
        zen.priorities = Priority::ALL
            .iter()
            .map(|p| p.as_str().to_string())
            .collect();
        let party = Party::Npc(12);
        let within = zen.on_thread(&OnThread {
            party: &party,
            name: "friend-0",
            owed: vec!["the garden report".into(), "the key".into()],
            may_change_mission: true,
            may_post: false,
            carried: None,
            capped: false,
            cooling: Vec::new(),
        });
        let specs = specs_within(Catalog::THREAD.mode, &within);
        assert_eq!(specs.len(), talk::within::ACTS.len(), "{specs:?}");
        assert!(
            estimated_paths(&specs, 3) > MAX_TURN_PATHS,
            "three acts fit after all: `TALK_ACTS` can be three"
        );
        let paths = estimated_paths(&specs, TALK_ACTS);
        assert!(
            paths <= MAX_TURN_PATHS,
            "a conversation's decision admits {paths} decodes, over the {MAX_TURN_PATHS} budget"
        );
    }

    /// A typo in [`LIVE`] names no real parameter and would silently leave the
    /// value free — the failure the side table trades for its brevity.
    #[test]
    fn every_live_parameter_names_a_real_one() {
        for (tool, param, _) in LIVE {
            let t = by_name(tool).unwrap_or_else(|| panic!("`{tool}` is not in the catalog"));
            assert!(
                t.params.iter().any(|p| p.name == *param),
                "`{tool}` has no `{param}` parameter"
            );
        }
    }

    /// A typo in [`REQUIRES`] names nothing and would leave the call free to
    /// close without what its choice depends on.
    #[test]
    fn every_requirement_names_a_real_choice_and_real_fields() {
        for (tool, param, rules) in REQUIRES {
            let t = by_name(tool).unwrap_or_else(|| panic!("`{tool}` is not in the catalog"));
            let chooser = t
                .params
                .iter()
                .find(|p| p.name == *param)
                .unwrap_or_else(|| panic!("`{tool}` has no `{param}` parameter"));
            assert!(
                chooser.required,
                "`{tool}.{param}` decides, so it is required"
            );
            assert!(
                live_choice(tool, param).is_some() || fixed_values(tool, param).is_some(),
                "`{tool}.{param}` has no closed set to decide over"
            );
            for (value, fields) in *rules {
                assert!(!fields.is_empty(), "`{value}` requires nothing");
                for field in *fields {
                    let f = t
                        .params
                        .iter()
                        .find(|p| p.name == *field)
                        .unwrap_or_else(|| panic!("`{tool}` has no `{field}` parameter"));
                    assert!(!f.required, "`{tool}.{field}` is already required outright");
                }
            }
        }
    }

    /// Every [`BRINGS`] rule names a required choice with a closed set, and
    /// fields that are the act's own optional parameters — so a brought field
    /// is one the act declares and documents, written only after its value.
    #[test]
    fn every_brought_field_is_a_real_optional_field() {
        for (tool, param, rules) in BRINGS {
            let t = by_name(tool).unwrap_or_else(|| panic!("`{tool}` is not in the catalog"));
            let chooser = t.params.iter().find(|p| p.name == *param).unwrap();
            assert!(
                chooser.required,
                "`{tool}.{param}` decides, so it is required"
            );
            assert!(fixed_values(tool, param).is_some() || live_choice(tool, param).is_some());
            for (_, fields) in *rules {
                for field in *fields {
                    let f = t
                        .params
                        .iter()
                        .find(|p| p.name == *field)
                        .unwrap_or_else(|| panic!("`{tool}` has no `{field}` parameter"));
                    assert!(!f.required, "`{tool}.{field}` is already required outright");
                }
            }
        }
    }

    /// **An edit writes what its operation takes, after the operation**: a
    /// replace its text, a move where it goes, a delete nothing — and none of
    /// those fields stands beside `do` as a sibling.
    #[test]
    fn each_edit_operation_brings_the_fields_it_takes() {
        let params = stencil_params(by_name("file_edit").unwrap(), &Within::among(&[]));
        let names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["path", "at", "do", "old", "to"]);
        let act = params.iter().find(|p| p.name == "do").unwrap();
        assert_eq!(
            act.enum_values.as_deref().unwrap(),
            crate::engine::bench::EDIT_ACTIONS
        );
        let brings = |value: &str| -> Vec<String> {
            act.shapes
                .iter()
                .find(|(v, _)| v == value)
                .map(|(_, f)| f.iter().map(|p| p.name.clone()).collect())
                .unwrap_or_default()
        };
        assert_eq!(brings("replace"), ["text"]);
        assert_eq!(brings("insert_after"), ["text"]);
        assert_eq!(brings("move"), ["dest", "where"]);
        assert!(brings("delete").is_empty());
        let place = &act.shapes.iter().find(|(v, _)| v == "move").unwrap().1[1];
        assert_eq!(
            place.enum_values.as_deref().unwrap(),
            crate::engine::bench::MOVE_PLACES
        );
    }

    /// **A message says how it goes before where**: `via` offers `private`
    /// only while there is somebody to message alone and `channel` only while
    /// there is a channel or group, and each brings its own `to` — the people
    /// for one, the channels for the other — never a name of the other kind
    /// and never as a sibling. With neither, `message` is not offered.
    #[test]
    fn a_message_says_how_it_goes_and_then_to_which_of_that_kind() {
        let mut within = Within::among(&[]);
        within.private = vec!["Soren".into()];
        within.channels = vec!["the channel".into(), "#research".into()];
        let params = stencil_params(by_name("message").unwrap(), &within);
        let names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["via", "intent"]);
        let via = &params[0];
        assert_eq!(
            via.enum_values.as_deref().unwrap(),
            [PRIVATE.to_string(), CHANNEL_WAY.to_string()]
        );
        let to = |value: &str| -> Vec<String> {
            let (_, fields) = via.shapes.iter().find(|(v, _)| v == value).unwrap();
            assert_eq!(fields.len(), 1);
            assert_eq!((fields[0].name.as_str(), fields[0].required), ("to", true));
            fields[0].enum_values.clone().unwrap()
        };
        assert_eq!(to(PRIVATE), ["Soren"]);
        assert_eq!(to(CHANNEL_WAY), ["the channel", "#research"]);

        within.private.clear();
        let params = stencil_params(by_name("message").unwrap(), &within);
        assert_eq!(
            params[0].enum_values.as_deref().unwrap(),
            [CHANNEL_WAY.to_string()]
        );
        assert_eq!(
            params[0].shapes.len(),
            1,
            "a way not offered brought a `to`"
        );

        within.channels.clear();
        assert!(!specs_within(Mode::Physical, &within)
            .iter()
            .any(|s| s.name == "message"));
    }

    /// A brought field as a test reads it: its name, whether it is required,
    /// its bound, and the values it may take.
    type Field<'a> = (&'a str, bool, Option<u32>, Option<Vec<String>>);

    /// **A standing job brings its own arguments**: chosen as `order_post`'s
    /// `job`, `coding-task` brings `what`, required and bounded as an act's
    /// string is, and `repository`, required and one of the repositories the
    /// world names; a job that takes none brings nothing; and there is no free
    /// `args` field to write them into.
    #[test]
    fn a_standing_job_brings_its_own_arguments() {
        let mut within = Within::among(&[]);
        within.jobs = vec!["coding-task".into(), "ward-shift".into()];
        within.takers = vec!["Wick".into(), "Umbra".into()];
        within.job_params = vec![(
            "coding-task".into(),
            vec![
                JobParam {
                    name: "what".into(),
                    choices: Vec::new(),
                },
                JobParam {
                    name: "repository".into(),
                    choices: vec!["candle".into(), "battle-mind".into()],
                },
            ],
        )];
        let params = stencil_params(by_name("order_post").unwrap(), &within);
        let names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["job", "brief", "steps", "for"]);
        assert_eq!(
            params[3].enum_values,
            Some(vec!["Wick".to_string(), "Umbra".to_string()]),
            "`for` is one of those who take the orders"
        );
        let mut alone = within.clone();
        alone.takers.clear();
        let names: Vec<String> = stencil_params(by_name("order_post").unwrap(), &alone)
            .into_iter()
            .map(|p| p.name)
            .collect();
        assert_eq!(names, ["job", "brief", "steps"], "nobody to name, no `for`");
        let job = &params[0];
        assert_eq!(
            job.enum_values.as_deref().unwrap(),
            ["coding-task".to_string(), "ward-shift".to_string()]
        );
        let brought: Vec<(&str, Vec<Field>)> = job
            .shapes
            .iter()
            .map(|(v, fields)| {
                (
                    v.as_str(),
                    fields
                        .iter()
                        .map(|f| {
                            (
                                f.name.as_str(),
                                f.required,
                                f.max_tokens,
                                f.enum_values.clone(),
                            )
                        })
                        .collect(),
                )
            })
            .collect();
        assert_eq!(
            brought,
            [(
                "coding-task",
                vec![
                    ("what", true, Some(ACT_VALUE_TOKENS), None),
                    (
                        "repository",
                        true,
                        None,
                        Some(vec!["candle".to_string(), "battle-mind".to_string()])
                    )
                ]
            )]
        );
    }

    /// **A fold cannot be sent without somewhere to fold to.** The refusal
    /// "A fold needs somewhere to fold to" was a grammar that offered the close
    /// right after `relocate`: each action takes different fields, none of them
    /// is needed by every action, so left to `required` the lot were optional.
    #[test]
    fn each_tower_action_requires_the_fields_it_takes() {
        let mut within = Within::among(&[]);
        within.tower_actions = vec!["relocate".into(), "siege".into(), "drill down".into()];
        let params = stencil_params(by_name("command_tower").unwrap(), &within);
        let action = params.iter().find(|p| p.name == "action").unwrap();
        let needs = |value: &str| -> Vec<&str> {
            action
                .requires
                .iter()
                .filter(|(v, _)| v == value)
                .flat_map(|(_, fields)| fields.iter().map(String::as_str))
                .collect()
        };
        assert_eq!(needs("relocate"), ["x", "y"]);
        assert_eq!(needs("siege"), ["target"]);
        assert_eq!(needs("drill down"), ["depth"]);
        for free in ["lift siege", "surface", "raise shields", "drop shields"] {
            assert!(needs(free).is_empty(), "{free} takes nothing");
        }
        assert!(
            params
                .iter()
                .filter(|p| p.name != "action")
                .all(|p| p.requires.is_empty()),
            "only the choosing field carries requirements"
        );
    }

    /// **A coordinate and a depth are numbers in the grammar, so they cannot be
    /// empty.** Written as strings they were free text, and a live tower wrote
    /// `"depth": ""` — the model's own closing quote lands where its opening one
    /// does. An integer is written digit by digit under the mask; a depth is
    /// never negative, and a coordinate may be.
    #[test]
    fn a_tower_coordinate_and_depth_are_integers() {
        let within = Within::among(&[]);
        let params = stencil_params(by_name("command_tower").unwrap(), &within);
        let field = |name: &str| params.iter().find(|p| p.name == name).unwrap();
        for name in ["x", "y", "depth"] {
            assert_eq!(field(name).ty, ParamType::Integer, "{name}");
        }
        assert_eq!(field("x").minimum, None);
        assert_eq!(field("y").minimum, None);
        assert_eq!(field("depth").minimum, Some(0.0));
        assert_eq!(field("target").ty, ParamType::String);
    }

    /// A rename would otherwise leave a minimum bounding nothing, or bounding a
    /// string, which the grammar ignores.
    #[test]
    fn every_minimum_names_a_real_integer_parameter() {
        for (tool, param, _) in MINIMUM {
            let t = by_name(tool).unwrap_or_else(|| panic!("no tool `{tool}`"));
            let p = t
                .params
                .iter()
                .find(|p| p.name == *param)
                .unwrap_or_else(|| panic!("`{tool}` has no `{param}` parameter"));
            assert_eq!(p.ty, "integer", "`{tool}.{param}` is not an integer");
        }
    }

    /// An act with no dependent fields compiles exactly as before.
    #[test]
    fn an_act_with_no_dependent_fields_requires_nothing() {
        let within = Within::among(&["Perrin Vastwood"]);
        for t in CATALOG
            .iter()
            .filter(|t| !REQUIRES.iter().any(|(tool, _, _)| *tool == t.name))
        {
            assert!(
                stencil_params(t, &within)
                    .iter()
                    .all(|p| p.requires.is_empty()),
                "{} carries a requirement",
                t.name
            );
        }
    }
}
