//! Tests which acts each mode and kind of company offers, so that acts needing
//! somebody are absent when nobody is there.
//!
//! Alone, acts that need a listener are absent from the grammar rather than
//! discouraged, because a model offered a field fills it in with an invented
//! addressee. A whisper needs a third party to keep it from, and room speech needs
//! a person to hear it. A body with no body of its own is not offered walking,
//! stopping is always available as the floor, and two characters pausing at each
//! other each come back on their own clock.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// **Alone, the acts that need somebody are not in the grammar at all.**
    ///
    /// Not discouraged — absent. Offering `ask` to a character with nobody to
    /// ask is offering a field, and a model handed a field fills it in: it
    /// invents an addressee, the world refuses, and being refused teaches it
    /// nothing, so it does it again. The prompt already withheld these; the
    /// mask did not, and the mask is the half that decides.
    #[test]
    fn a_character_alone_cannot_reach_the_acts_that_need_company() {
        // Alone but *somewhere*: `nowhere()` means no world at all, and a
        // character with no world cannot walk either, which would confuse two
        // separate absences into one assertion.
        let solitary = Within {
            company: Vec::new(),
            ..Within::among(&[])
        };
        let alone: Vec<String> = specs_within(Mode::Physical, &solitary)
            .into_iter()
            .map(|t| t.name)
            .collect();
        // A shout is the one speech act offered alone, while a person is
        // within its reach: it is how a character with nobody here reaches the
        // rooms around it.
        assert!(!alone.contains(&"shout".to_string()), "{alone:?}");
        let overheard = specs_within(
            Mode::Physical,
            &Within {
                heard_by_person: true,
                ..solitary.clone()
            },
        );
        assert!(overheard.iter().any(|t| t.name == "shout"), "{overheard:?}");
        for needs_company in ["ask", "tell", "whisper", "gesture"] {
            assert!(
                !alone.contains(&needs_company.to_string()),
                "`{needs_company}` reaches nobody and was offered anyway: {alone:?}"
            );
        }
        // **`tell` and `gesture` are in that list**, which they were not. Speech
        // into an empty room reaches nobody and changes nothing, and offering
        // it produced a cast that narrated the scenery at itself: three
        // characters alone in three rooms, saying every noise the building made
        // back out loud, twelve of the last fourteen acts in the feed.
        //
        // What it can still do alone is untouched.
        assert!(alone.contains(&"move_to".to_string()), "{alone:?}");
        assert!(alone.contains(&"reflect".to_string()), "{alone:?}");

        // And with nowhere to go, walking leaves the grammar too — while
        // **stopping never does**. It is the floor: a character with nobody to
        // speak to and nowhere to walk must still have a way to spend a turn,
        // and `pause` is the one act that is always available whatever the room
        // is like.
        let stuck: Vec<String> = specs_within(Mode::Physical, &Within::nowhere())
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert!(!stuck.contains(&"move_to".to_string()), "{stuck:?}");
        assert!(stuck.contains(&"reflect".to_string()), "{stuck:?}");

        let with = Within::among(&["Perrin Vastwood"]);
        let together: Vec<String> = specs_within(Mode::Physical, &with)
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert!(together.contains(&"ask".to_string()), "{together:?}");
        assert!(together.contains(&"tell".to_string()), "{together:?}");
    }

    /// **A whisper needs somebody to keep it from.** With one other person in
    /// the room it is `tell` said quietly, and offered there it was what the
    /// cast said everything with — so it arrives with the second person and
    /// leaves with them, while `tell` stays.
    #[test]
    fn a_whisper_is_offered_only_with_somebody_else_to_keep_it_from() {
        let names = |w: &Within| -> Vec<String> {
            specs_within(Mode::Physical, w)
                .into_iter()
                .map(|t| t.name)
                .collect()
        };
        let one = names(&Within::among(&["Perrin Vastwood"]));
        let two = names(&Within::among(&["Perrin Vastwood", "Wren Weaver"]));
        assert!(!one.contains(&"whisper".to_string()), "{one:?}");
        assert!(one.contains(&"tell".to_string()), "{one:?}");
        assert!(two.contains(&"whisper".to_string()), "{two:?}");
        assert!(two.contains(&"tell".to_string()), "{two:?}");

        // The situation's "you can also use" line follows the same rule — it
        // named `whisper` beside one person, and the cast took it at its word.
        let told = |company: usize| -> Vec<&str> {
            nearby(company, company, false)
                .into_iter()
                .map(|t| t.name)
                .collect()
        };
        assert!(told(0).is_empty(), "{:?}", told(0));
        assert!(!told(1).contains(&"whisper"), "{:?}", told(1));
        assert!(told(1).contains(&"tell"), "{:?}", told(1));
        assert!(told(2).contains(&"whisper"), "{:?}", told(2));
    }

    /// **Room speech is for people** (`docs/npc_conversations.md` §12): the
    /// exact speech acts offered for each kind of company.
    #[test]
    fn room_speech_is_offered_only_where_a_person_would_hear_it() {
        // Sorted: which acts, not the catalog's order, is the question.
        let speech = |w: &Within| -> Vec<String> {
            let mut acts: Vec<String> = specs_within(Mode::Physical, w)
                .into_iter()
                .map(|t| t.name)
                .filter(|n| {
                    [
                        "tell", "ask", "whisper", "shout", "promise", "remind", "gesture",
                        "message", "reply",
                    ]
                    .contains(&n.as_str())
                })
                .collect();
            acts.sort();
            acts
        };
        let phone = |mut w: Within| {
            w.threads = vec!["Wick".into()];
            w.private = vec!["Wick".into()];
            w.awaiting = vec!["Wick".into()];
            w
        };
        // Characters only: they gesture, and message each other.
        let characters = phone(Within::among_characters(&["Wick", "Soren"]));
        assert_eq!(speech(&characters), ["gesture", "message", "reply"]);
        // Alone, the phone is all there is.
        assert_eq!(
            speech(&phone(Within::among_characters(&[]))),
            ["message", "reply"]
        );
        // A visitor among them, owing something: room speech, and a whisper
        // with someone else to keep it from. A shout from the room a person
        // stands in reaches them.
        let mut with_wren = phone(Within::among_characters(&["Wick", "Wren"]));
        with_wren.people = vec!["Wren".into()];
        with_wren.heard_by_person = true;
        with_wren.owed = vec!["the key".into()];
        assert_eq!(
            speech(&with_wren),
            [
                "ask", "gesture", "message", "promise", "remind", "reply", "shout", "tell",
                "whisper"
            ]
        );
        // A visitor within a shout's reach, and nobody here.
        let heard = Within {
            heard_by_person: true,
            ..phone(Within::among_characters(&[]))
        };
        assert_eq!(speech(&heard), ["message", "reply", "shout"]);
        // Field play.
        let field = Within {
            in_field: true,
            ..phone(Within::among_characters(&["Wick"]))
        };
        assert_eq!(speech(&field), ["gesture", "message", "reply", "shout"]);
    }

    /// **An addressee can only be somebody who is here.**
    ///
    /// A free-text `to` let a character write any string, and it wrote a first
    /// name the world had not registered — every tick, for as long as it stood
    /// there, because a refusal is not a lesson. Bound to the names present,
    /// the wrong name is not a mistake available to it.
    #[test]
    fn an_addressee_is_drawn_from_who_is_actually_here() {
        let with = Within::among(&["Perrin Vastwood", "Orion Vance"]);
        let specs = specs_within(Mode::Physical, &with);

        let ask = specs.iter().find(|t| t.name == "ask").expect("in company");
        let to = ask.params.iter().find(|p| p.name == "to").unwrap();
        assert_eq!(
            to.enum_values.as_deref(),
            Some(&["Perrin Vastwood".to_string(), "Orion Vance".to_string()][..]),
        );

        // Only the parameter that names somebody. What the character *means*
        // stays its own — a closed set there would be the machinery writing
        // its lines.
        let about = ask.params.iter().find(|p| p.name == "about").unwrap();
        assert_eq!(about.enum_values, None);
        let shout = specs.iter().find(|t| t.name == "shout").unwrap();
        assert!(shout.params.iter().all(|p| p.enum_values.is_none()));
    }

    /// **Stopping has nothing to *choose*, anywhere.**
    ///
    /// The whole reason it replaced the typed wait. That act needed a condition
    /// the world could answer, which needed a live set of kinds filtered by who
    /// was in the room, which needed a rule excluding anybody already waiting on
    /// you so a pair could not wait at each other — and every one of those was a
    /// closed branch that could go empty and take the whole grammar with it.
    ///
    /// What a pause takes instead is free text about the character's own head,
    /// which the world never has to answer and which therefore cannot be empty.
    #[test]
    fn stopping_has_nothing_to_choose_and_so_cannot_go_wrong() {
        for within in [
            Within::nowhere(),
            Within::among(&["Perrin Vastwood"]),
            Within::among(&["Perrin Vastwood", "Orion Vance"]),
        ] {
            let specs = specs_within(Mode::Physical, &within);
            let p = specs
                .iter()
                .find(|t| t.name == "reflect")
                .expect("a character can always stop");
            for param in &p.params {
                assert!(
                    param.enum_values.is_none(),
                    "`{}` is a closed set, so a pause can now be taken out of the grammar by an \
                     empty room",
                    param.name
                );
            }
        }
        assert!(
            by_name("wait_for").is_none(),
            "`wait_for` is back — one concept, one act"
        );
    }

    /// **Stopping asks what is on your mind, and it is not optional.**
    ///
    /// It is the one act whose outward half is nothing at all, so without this
    /// the record of a character's quietest hours is a column of identical rows
    /// and there is no telling a character that is thinking from one that has
    /// run out of things to do.
    #[test]
    fn stopping_asks_what_the_character_is_thinking() {
        let p = by_name("reflect").expect("a character can always stop and think");
        let thoughts = p
            .params
            .iter()
            .find(|p| p.name == "inner_thoughts")
            .expect("a pause says nothing about the inside");
        assert!(thoughts.required, "a silent pause records nothing at all");

        // **Every one of them, always.** A field a model may skip is a field a
        // model does skip, and the quiet turns are exactly the ones with
        // nothing else in them to read.
        for name in ["situation", "inner_thoughts", "feeling"] {
            let param = p
                .params
                .iter()
                .find(|p| p.name == name)
                .unwrap_or_else(|| panic!("a pause does not ask `{name}`"));
            assert!(param.required, "`{name}` is one a character may skip");
        }
        // And nothing asking it to write its own conclusion: that is what the
        // reflection conversation answers with.
        assert_eq!(p.params.len(), 3, "{:?}", p.params);
    }

    /// **How it feels is chosen from the moods the mind actually holds.**
    ///
    /// A free-text register is a word nothing downstream can act on. Steered to
    /// the library, a character naming `cornered` has named a register the
    /// projection can select, which is what makes it worth recording at all.
    #[test]
    fn how_a_character_feels_is_one_of_the_registers_the_mind_holds() {
        let held = Within {
            feelings: vec!["alert".into(), "cornered".into(), "battle_weary".into()],
            ..Within::among(&["Perrin Vastwood"])
        };
        let specs = specs_within(Mode::Physical, &held);
        let p = specs.iter().find(|t| t.name == "reflect").expect("offered");
        let feeling = p
            .params
            .iter()
            .find(|p| p.name == "feeling")
            .expect("nowhere to say how it feels");
        assert_eq!(
            feeling.enum_values.as_deref(),
            Some(
                &[
                    "alert".to_string(),
                    "cornered".to_string(),
                    "battle_weary".to_string()
                ][..]
            )
        );
    }

    /// **A daemon with no moods still stops, and is still asked.**
    ///
    /// A vocabulary is not a possibility — see [`Choices::steers`]. An empty
    /// list of registers means nobody has authored the words yet, not that the
    /// character has no feeling, so the parameter falls back to free text
    /// rather than emptying its branch and taking `pause` out of the grammar.
    /// A character that cannot stop is how the pacing came back last time.
    #[test]
    fn a_daemon_with_no_moods_can_still_stop_and_is_still_asked() {
        let specs = specs_within(Mode::Physical, &Within::nowhere());
        let p = specs
            .iter()
            .find(|t| t.name == "reflect")
            .expect("a mindless daemon took away stopping");

        let feeling = p
            .params
            .iter()
            .find(|p| p.name == "feeling")
            .expect("the question went with the vocabulary");
        assert!(
            feeling.required,
            "it became skippable rather than unsteered"
        );
        assert!(
            feeling.enum_values.is_none(),
            "a branch with no arms, which stops the whole grammar compiling"
        );
    }

    /// **The vocabulary rule applies to exactly one set.**
    ///
    /// Every other closed set here enumerates what is *possible*, and an empty
    /// one has to take its act out of the grammar — `ask` with nobody to name,
    /// `move_to` with nowhere to go. Loosening that for anything else would
    /// turn a refusal a character cannot learn from into a branch it can
    /// free-decode into.
    #[test]
    fn only_a_vocabulary_survives_being_empty() {
        for c in [
            Choices::Company,
            Choices::CompanyOrSelf,
            Choices::Reachable,
            Choices::Rooms,
            Choices::Carried,
            Choices::Operable,
            Choices::Threads,
            Choices::Postures,
        ] {
            assert!(c.steers(), "{c:?} stopped meaning what is possible");
        }
        assert!(!Choices::Feelings.steers());
    }

    /// **Two characters cannot deadlock on it.**
    ///
    /// The old wait needed a dedicated rule for this: waiting on somebody woke
    /// them, so if they could wait back the pair ping-ponged, and a live cast
    /// went completely silent — every act in the feed a `wait_for`. A pause
    /// wakes nobody and is on nobody, so two characters pausing at each other
    /// each come back on their own clock.
    #[test]
    fn two_characters_pausing_at_each_other_both_come_back() {
        let together = Within::among(&["Perrin Vastwood"]);
        let specs = specs_within(Mode::Physical, &together);
        assert!(specs.iter().any(|t| t.name == "reflect"));
        // And they can still speak to each other, which is the way out.
        let ask = specs.iter().find(|t| t.name == "ask").unwrap();
        let to = ask.params.iter().find(|p| p.name == "to").unwrap();
        assert!(to
            .enum_values
            .as_deref()
            .is_some_and(|v| v.contains(&"Perrin Vastwood".to_string())));
    }

    pub(in super::super) fn named(mode: Mode) -> Vec<&'static str> {
        for_mode(mode).iter().map(|t| t.name).collect()
    }

    /// `send_image` is absent in physical mode rather than present-and-refused.
    #[test]
    fn a_physically_present_character_is_not_offered_a_camera() {
        assert!(!named(Mode::Physical).contains(&"send_image"));
        assert!(named(Mode::InstantMessage).contains(&"send_image"));
    }

    pub(in super::super) fn offered_to(within: &Within) -> Vec<String> {
        specs_within(Mode::Physical, within)
            .into_iter()
            .map(|t| t.name)
            .collect()
    }

    /// **Giving back is offered only to a body that is holding something.**
    ///
    /// `release` takes no argument, so the empty-set rule has nothing to empty,
    /// and it was `Always` — offered to a body with nothing in its hands and
    /// refused afterwards, which a live cast did twice in a minute.
    #[test]
    fn release_is_offered_only_while_holding_something() {
        let empty_handed = Within::among(&["Perrin Vastwood"]);
        assert!(
            !offered_to(&empty_handed).contains(&"release".to_string()),
            "release was offered to a body holding nothing"
        );

        let holding = Within {
            held: vec!["fabricator 1".into()],
            ..Within::among(&["Perrin Vastwood"])
        };
        assert!(offered_to(&holding).contains(&"release".to_string()));
    }

    /// Holding something is not company, a place or a station: it is offered
    /// alone in a corridor as much as in a crowd.
    #[test]
    fn release_does_not_depend_on_the_room() {
        let alone = Within {
            held: vec!["close the longest silence in the record".into()],
            ..Within::nowhere()
        };
        assert!(offered_to(&alone).contains(&"release".to_string()));
        assert!(!offered_to(&Within::nowhere()).contains(&"release".to_string()));
    }

    /// The prompt is written once, so an act that depends on what a body holds
    /// this moment cannot be in it — it arrives with the situation.
    #[test]
    fn release_is_not_in_the_prompt_written_once() {
        assert!(!named(Mode::Physical).contains(&"release"));
        assert!(!named(Mode::InstantMessage).contains(&"release"));
    }

    /// **Carrying a picture is asked separately from being apart**, and it has
    /// to stay that way even while the two answers agree.
    ///
    /// They agree on both current modes, so nothing distinguishes the two
    /// predicates by behaviour any more and `carries_pictures` could be deleted
    /// in favour of `remote` without a single test going red. It must not be:
    /// the questions are different — where the parties are, and what the
    /// channel can carry — and the last time they were collapsed a character on
    /// a voice call was handed a way to text a photo down it. This is the test
    /// that says so out loud.
    #[test]
    fn carrying_a_picture_is_a_question_about_the_channel() {
        // Remote and carries a picture: writing.
        assert!(Mode::InstantMessage.remote() && Mode::InstantMessage.carries_pictures());
        // Neither — and *not* because the channel is too thin. There is no
        // channel; you are standing in front of them holding the thing.
        assert!(!Mode::Physical.remote() && !Mode::Physical.carries_pictures());
        assert!(!named(Mode::Physical).contains(&"send_image"));
    }

    /// Leaving and opening a conversation belong to every remote channel.
    #[test]
    fn a_remote_channel_can_be_opened_and_left() {
        assert!(named(Mode::InstantMessage).contains(&"sign_off"));
        assert!(named(Mode::InstantMessage).contains(&"reach_out"));
    }

    /// **The phone is carried, not entered — so it is not gated by mode.**
    ///
    /// These used to be `MessagingOnly`, on the reading that messaging *is* a
    /// mode. It is not: a character texts somebody a level away while standing
    /// in a room, in the middle of something else, and the room hears none of
    /// it. What gates them is the handset, through an empty thread list, which
    /// is why a character with no phone is offered none of them and one with a
    /// phone is offered them wherever it happens to be.
    #[test]
    fn the_phone_acts_are_not_gated_by_mode() {
        for mode in [Mode::Physical, Mode::InstantMessage] {
            let acts = named(mode);
            for phone in ["message", "reach_out", "invite", "open_group", "sign_off"] {
                assert!(
                    acts.contains(&phone),
                    "`{phone}` is missing in {mode:?} — the handset does not care about the mode"
                );
            }
        }
    }

    /// Laying a hand on somebody is the mirror of sending a picture: face to
    /// face only.
    #[test]
    fn nobody_puts_a_hand_on_anybody_down_a_line() {
        assert!(named(Mode::Physical).contains(&"act"));
        assert!(!named(Mode::InstantMessage).contains(&"act"));
    }

    /// **One word for one idea, on both sides of the room.** A character does
    /// something to a person with `act`; the person at the console does it back
    /// with `/act` (`engine::slash`). It was `touch` here, which read as gentle
    /// contact in a game about a war and did not match the console — so the two
    /// halves of one exchange had different names for the same thing.
    #[test]
    fn a_character_can_act_on_its_own_body() {
        // **The addressee is grammar-constrained.** A value that is not in this
        // set is one the decoder physically cannot emit, so a character binding
        // its own wound could not say so — not because the world refused it,
        // but because there was no token path to the sentence.
        let with_company = Within {
            company: vec!["Perrin Vastwood".into()],
            ..Within::default()
        };
        let who = live_set(Choices::CompanyOrSelf, &with_company);
        assert!(who.contains(&SELF.to_string()), "{who:?}");
        assert!(who.contains(&"Perrin Vastwood".to_string()), "{who:?}");
    }

    /// And **alone**, where `company` is empty and the empty-set rule would
    /// otherwise take the whole tool out of the grammar. Your own body is always
    /// here, so `act` never leaves for want of somebody to act on.
    ///
    /// **That reachability is deliberate and it is not free.** A solitary cast
    /// chose it every single turn, because the room kept handing it physical
    /// things to notice and this was the nearest verb. The answer is the
    /// cooldown a self-act serves, not taking the target away — see
    /// [`crate::engine::cooldown::SELF_ACT`].
    #[test]
    fn acting_survives_being_alone() {
        let alone = Within::default();
        assert!(live_set(Choices::Company, &alone).is_empty());
        assert_eq!(live_set(Choices::CompanyOrSelf, &alone), vec![SELF]);
        let offered: Vec<String> = specs_within(Mode::Physical, &alone)
            .iter()
            .map(|s| s.name.clone())
            .collect();
        assert!(
            offered.contains(&"act".to_string()),
            "a character alone lost the ability to act on itself: {offered:?}"
        );
    }

    /// **An act a body has just taken is absent, not refused.**
    ///
    /// The same discipline as every other absence here, and for the same
    /// reason: a live cast read its own refusals back as the most recent thing
    /// in its window and emitted the same act again. Thirty-eight of its last
    /// fifty acts were `move_to`.
    #[test]
    fn an_act_that_is_cooling_is_not_in_the_grammar_at_all() {
        let ready = Within::among(&["Perrin Vastwood"]);
        let name = |ts: &[ToolSpec]| ts.iter().map(|t| t.name.clone()).collect::<Vec<_>>();

        let before = name(&specs_within(Mode::Physical, &ready));
        assert!(before.contains(&"move_to".to_string()));
        assert!(before.contains(&"act".to_string()));

        let cooling = Within {
            cooling: vec!["move_to".into()],
            ..ready
        };
        let after = name(&specs_within(Mode::Physical, &cooling));
        assert!(
            !after.contains(&"move_to".to_string()),
            "a body that just walked was offered another walk: {after:?}"
        );
        // And only that one. A cooldown that took the rest of the catalog with
        // it would leave a character with nothing to do but stand there.
        assert!(after.contains(&"act".to_string()));
        assert!(after.contains(&"tell".to_string()));
        assert_eq!(after.len() + 1, before.len(), "{after:?}");
    }

    #[test]
    fn every_act_that_cools_is_an_act_that_exists() {
        // The table names acts by string. A rename would otherwise leave a
        // cooldown on nothing, silently, and the act it was meant to slow would
        // run free.
        for (tool, _) in crate::engine::cooldown::all() {
            assert!(
                by_name(tool).is_some(),
                "`{tool}` has a cooldown and is not in the catalog"
            );
        }
    }

    #[test]
    fn a_character_acts_on_somebody_with_the_same_word_the_console_uses() {
        assert!(by_name("act").is_some());
        assert!(
            by_name("touch").is_none(),
            "`touch` is back — one concept, one word, whichever side it comes from"
        );
        assert!(
            crate::engine::slash::lookup("act").is_some(),
            "the console lost /act, so the two sides no longer agree"
        );
    }

    /// **A mind with no body is not offered a way to walk.**
    ///
    /// Keeper has no embodiment path and never will, so every act needing hands,
    /// feet or a place to stand is absent rather than refused — a model handed a
    /// field fills it in.
    #[test]
    fn a_mind_with_no_body_keeps_speech_and_loses_everything_physical() {
        let bodied: Vec<&str> = for_body(Mode::Physical, true)
            .iter()
            .map(|t| t.name)
            .collect();
        let bodiless: Vec<&str> = for_body(Mode::Physical, false)
            .iter()
            .map(|t| t.name)
            .collect();

        for gone in [
            "move_to", "follow", "act", "gather", "engage", "equip", "use",
        ] {
            assert!(bodied.contains(&gone), "{gone} should exist for a body");
            assert!(
                !bodiless.contains(&gone),
                "{gone} was offered to a mind with no body"
            );
        }
        // What it keeps is what it can actually do without a body: attend,
        // reach somewhere it is not, and read.
        //
        // **`tell` is not in this list and is not missing.** Speech needs
        // somebody to hear it, so like `ask` and `whisper` it arrives with the
        // situation rather than with the prompt — this is `for_body`, which is
        // the prompt's half and deliberately withholds everything conditional
        // on the room. `command_tower` and `produce` are absent for the same
        // reason, one step further out: they are reached by standing at a
        // console or a bay.
        for kept in ["scan", "read", "reflect"] {
            assert!(bodiless.contains(&kept), "{kept} was taken from Keeper");
        }
        // **`recall` is conditional on the *room* rather than on company, and
        // it belongs here for the same reason.** A body standing at its own
        // muster point has no journey home to make, so the act arrives with the
        // situation once it is somewhere else. It was `Embodied` — true, and
        // not enough: offered at home it reported crossing the building and
        // moved nobody, twenty-four times in sixty acts.
        for conditional in ["tell", "ask", "whisper", "gesture", "recall"] {
            assert!(
                !bodiless.contains(&conditional) && !bodied.contains(&conditional),
                "{conditional} depends on the situation, so the prompt cannot know it"
            );
        }
    }

    /// The default must be the mode that offers *fewer* tools. A mode that
    /// failed to resolve would otherwise hand a camera to a character standing
    /// in front of you.
    #[test]
    fn the_default_mode_is_the_restrictive_one() {
        assert_eq!(Mode::default(), Mode::Physical);
        assert!(!named(Mode::default()).contains(&"send_image"));
    }

    /// Every mode round-trips through the name the interaction contract uses.
    #[test]
    fn every_mode_answers_to_its_own_wire_name() {
        for m in [Mode::Physical, Mode::InstantMessage] {
            assert_eq!(Mode::parse(m.as_wire()), Some(m));
        }
        assert_eq!(Mode::parse("carrier pigeon"), None);
        // The two that were removed are gone from the wire too, rather than
        // quietly parsing to something else — a console still asking for a
        // voice call gets a `bad_mode` naming what it can have, not a silent
        // downgrade into a room.
        for withdrawn in ["voice_call", "voice", "video_call", "video"] {
            assert_eq!(Mode::parse(withdrawn), None, "{withdrawn} still parses");
        }
    }

    /// **Every act declared in a module reaches the catalogue, exactly once.**
    ///
    /// The catalogue is composed from four lists, and this is what stops one of
    /// them being written and not chained. Three messaging acts sat in
    /// `acts::WORLD_ACTS` for a while while `CATALOG` was built from a
    /// hand-written list beside it — declared, documented, dispatched, and
    /// invisible to every character in the world. Nothing failed; they were
    /// simply never offered.
    #[test]
    fn every_declared_act_reaches_the_catalog_exactly_once() {
        let mut declared: Vec<&str> = BODY_ACTS
            .iter()
            .chain(acts::WORLD_ACTS)
            .chain(station::STATION_ACTS)
            .chain(bench::BENCH_ACTS)
            .chain(mission_acts::MISSION_ACTS)
            .chain(AGENDA_ACTS)
            .chain(BRIDGE_ACTS)
            .chain(TERMINAL_ACTS)
            .chain(HOUSE_ACTS)
            .chain(BUILD_ACTS)
            .chain(TASK_ACTS)
            .chain(BEYOND_ACTS.iter())
            .chain(HAND_OFF_ACTS)
            .map(|t| t.name)
            .collect();
        let n = declared.len();
        declared.sort_unstable();
        declared.dedup();
        assert_eq!(declared.len(), n, "an act is declared in two lists");
        assert_eq!(CATALOG.len(), n, "a declared act never reached the catalog");
    }

    #[test]
    fn names_are_unique_and_lowercase() {
        let mut names: Vec<&str> = CATALOG.iter().map(|t| t.name).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "two tools share a name");
        for t in CATALOG.iter() {
            assert_eq!(t.name, t.name.to_lowercase(), "{} is not lowercase", t.name);
            assert!(!t.description.is_empty());
        }
    }

    /// The catalog is the character's whole vocabulary. If it is thin, the model
    /// forces everything through `tell` and the world never changes.
    ///
    /// `Social` and `Internal` are gone deliberately, and not by thinning:
    /// `greet`/`offer`/`refuse`/`threaten` were `tell` with a stance, which its
    /// `manner` already carries, and every `Internal` act wrote nothing
    /// anywhere. What the character can *do* is unchanged; what it can do
    /// pointlessly is not. See [`CATALOG`].
    #[test]
    fn the_catalog_covers_every_category_the_design_names() {
        for c in [
            "Speech",
            "Movement",
            "Gesture",
            "Attention",
            "Messaging",
            "Meta",
        ] {
            assert!(
                CATALOG.iter().any(|t| t.category == c),
                "no tool in category {c}"
            );
        }
    }

    /// **Every act in the catalog must be able to change something.**
    ///
    /// The rule the reform was for, held here so it cannot rot back: an act
    /// that reaches no world and writes no state is a way to spend a turn and
    /// look busy, and it is invisible from every angle the daemon has — the
    /// call is well-formed, the tick succeeds, the feed shows a character
    /// acting. Sixteen of twenty-one were like that, and a live cast spent a
    /// hundred turns choosing them.
    #[test]
    fn every_act_offered_is_one_that_does_something() {
        for t in for_mode(Mode::Physical) {
            assert!(
                crate::engine::body::is_of_the_body(t.name)
                    || crate::engine::body::is_device(t.name)
                    || crate::engine::body::is_of_the_house(t.name)
                    || t.name == "wait",
                "`{}` is offered but reaches no world — implement it in `body::perform`, route it \
                 through the effector device, or take it out of the catalog",
                t.name
            );
        }
    }
}
