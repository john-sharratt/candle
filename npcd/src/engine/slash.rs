//! `/` commands — an operator speaking to a character's loop rather than to the
//! character.
//!
//! # What this is for
//!
//! Watching an NPC think is hard because the interesting part is asynchronous:
//! events arrive from the world, the loop ticks when salience says so, and the
//! acts come out somewhere else entirely. The Pulse view shows that traffic; this
//! is how you *inject* into it, from inside a conversation, without standing up a
//! world simulation first.
//!
//! Typing `/act puts a knife against your throat` into the console is both an
//! instrument and a way to play. It puts an event on the character's inbox
//! exactly as a world would, at a salience high enough to preempt, and the Pulse
//! view shows the tick it causes and what comes out. That loop — poke, watch,
//! poke again — is the whole point.
//!
//! # Why parsing lives here rather than in the console
//!
//! Because the daemon is the only thing that can be authoritative about what
//! commands exist. A console with its own list would drift, and would drift
//! silently: an unknown command would be sent as ordinary speech and the operator
//! would think the character had ignored them. `GET /v1/commands` serves
//! [`CATALOG`], so the console's autocomplete is the daemon's own answer.
//!
//! # An unrecognised slash is an error, not speech
//!
//! `/hrut` is a typo, not a line of dialogue. Sending it to the character as
//! speech would be the worst outcome — it would appear to work. The parse fails
//! loudly instead and names the near miss.

use serde::Serialize;

use crate::engine::event::{Addressed, EventKind, Salience};

/// A command an operator can type.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Command {
    pub name: &'static str,
    /// Which heading it sits under in the palette. The console groups by this,
    /// and a catalogue this size is unreadable as one flat list.
    pub group: &'static str,
    /// One line, for the palette. [`Command::description`] is the long form and
    /// is shown once a command has actually been chosen.
    pub summary: &'static str,
    /// What the rest of the line means for this command.
    pub argument: &'static str,
    pub description: &'static str,
    /// The kind of event it puts on the inbox, for the palette's hint.
    pub emits: &'static str,
    /// How urgently the resulting event is taken. Named on the command rather
    /// than typed each time, because "does this preempt" is a property of the
    /// kind of thing that happened, not of the operator's mood.
    pub salience: f32,
    pub example: &'static str,
}

/// Every `/` command. Served at `GET /v1/commands`.
///
/// # A command is a difference the engine can tell, not a verb
///
/// Battle Cities is a war, and for a while the only physical act an operator
/// had was `/hurt` — which describes damage arriving from nowhere: **"You are
/// hurt: a bolt through the shoulder"**, with nobody holding the bow. A
/// character told only that it has been hit cannot fear anybody, answer
/// anybody, or hold it against anybody afterwards, and the person who did it is
/// standing in the room.
///
/// The first fix for that was thirteen verbs — strike, stab, shoot, kill, grab,
/// shove, restrain, disarm and the rest. That was the wrong shape, and the
/// reason is worth keeping written down: **every one of them produced the same
/// event.** Same kind, same agent, same side of the preempt bar. They differed
/// only in the sentence they rendered, and the operator was already typing that
/// sentence in the argument. A catalogue of synonyms is a menu you have to read
/// to discover it had one entry.
///
/// So there is [`one`](CATALOG) command for doing something to a character, and
/// the words are the operator's. What earns a separate command is a difference
/// the engine can act on — who is named, what kind of event it is, which side of
/// the preempt bar it falls — which is exactly what separates `do` from `hurt`,
/// `say` from `overhear`, and `see` from `urgent`.
pub const CATALOG: &[Command] = &[
    // ── speech ──────────────────────────────────────────────────────────────
    Command {
        name: "say",
        group: "speech",
        summary: "Speak to the character",
        argument: "what is said to the character",
        description: "Someone speaks directly to the character. The default if you type a bare \
                      line with no slash at all.",
        emits: "speech",
        salience: 0.6,
        example: "/say Hess is at the gate and he is asking for you by name",
    },
    Command {
        name: "overhear",
        group: "speech",
        summary: "Speech it catches but is not part of",
        argument: "what is said nearby",
        description: "Speech the character catches but is not part of. Whether silence is rude \
                      depends on this, so it is its own command.",
        emits: "speech",
        salience: 0.4,
        example: "/overhear two guards, quietly: the quartermaster has been selling the grain",
    },
    // ── contact ─────────────────────────────────────────────────────────────
    Command {
        name: "act",
        group: "contact",
        summary: "Do something — you are named as the one who did it",
        argument: "what you do, and to whom",
        description: "Anything you do rather than say: shake its hand, hit it, hold it, take its \
                      weapon, put a knife against its throat — or do something to yourself it \
                      can see you do. Your words are the act, so say who it lands on: `punches \
                      you in the face`, `reloads his own rifle`. It reaches the character with \
                      your name on it, which is what lets it fear you, answer you, or hold it \
                      against you afterwards. Preempts: nothing anybody was doing survives being \
                      taken hold of.",
        emits: "description",
        salience: 0.95,
        example: "/act shakes your hand, and holds it a moment too long",
    },
    // ── the room ────────────────────────────────────────────────────────────
    Command {
        name: "see",
        group: "the room",
        summary: "Something happens in front of it",
        argument: "what happens, described",
        description: "Something happens in front of the character. The general-purpose event.",
        emits: "description",
        salience: 0.5,
        example: "/see the east granary is burning and nobody is fighting it",
    },
    Command {
        name: "urgent",
        group: "the room",
        summary: "Something that cannot wait",
        argument: "what happens",
        description: "Something that cannot wait. Like /see, but preempts.",
        emits: "description",
        salience: 0.9,
        example: "/urgent the roof beam above you cracks and begins to give",
    },
    Command {
        name: "notice",
        group: "the room",
        summary: "A named thing is observed doing something",
        argument: "<entity> : <what it is doing>",
        description: "A specific entity is observed. Splits on the first colon.",
        emits: "entity",
        salience: 0.5,
        example: "/notice a scout in Hess's colours : moving along the ridge line, unhurried",
    },
    Command {
        name: "here",
        group: "the room",
        summary: "Where it is, and what is true there",
        argument: "where the character is and what is true there",
        description: "The situation, in prose. Replaces the previous one rather than adding to \
                      it — it is a point in time, not a thing that happened.",
        emits: "situation",
        salience: 0.1,
        example: "/here You are in the green room. Maker-04 is here.",
    },
    // ── the loop ────────────────────────────────────────────────────────────
    Command {
        name: "wake",
        group: "the loop",
        summary: "Force a tick with nothing new",
        argument: "(nothing)",
        description: "Force a tick now with no new information. Shows you what the character \
                      does with what it already has.",
        emits: "heartbeat",
        salience: 0.85,
        example: "/wake",
    },
    Command {
        name: "sleep",
        group: "the loop",
        summary: "End its day now",
        argument: "(nothing)",
        description: "End the character's day now: fold the conversation into memory, tombstone \
                      it, and open tomorrow's. Normally the clock does this.",
        emits: "sleep",
        salience: 0.7,
        example: "/sleep",
    },
];

/// What an operator's line turned into.
#[derive(Clone, Debug, PartialEq)]
pub struct Parsed {
    pub kind: EventKind,
    pub salience: Salience,
    /// The command that produced it, for the Pulse view's label.
    pub command: &'static str,
}

impl Parsed {
    /// Say who did this.
    ///
    /// **The parser cannot know.** It turns a typed line into an event and has
    /// no idea whose console it came from, so speech is written with the
    /// placeholder speaker `you` — which renders as *"you says to you: …"*, a
    /// sentence both ungrammatical and wrong about who spoke — and a physical
    /// act is written with `{who}` standing where the name goes.
    ///
    /// Both are filled in here, by the one caller that knows the account.
    ///
    /// **A blow has to name the person who landed it.** A character told only
    /// that it has been hit cannot fear anybody, answer anybody, or hold it
    /// against anybody afterwards — and whoever did it is standing in the room
    /// with a name the world already knows.
    ///
    /// Safe on any parse: an event with neither a speaker nor a `{who}` is
    /// returned untouched.
    pub fn attributed_to(mut self, who: &str) -> Parsed {
        let who = who.trim();
        if who.is_empty() {
            return self;
        }
        match &mut self.kind {
            EventKind::Speech { speaker, .. } if speaker == "you" => {
                *speaker = who.to_string();
            }
            EventKind::Description { text } => *text = text.replace("{who}", who),
            _ => {}
        }
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ParseError {
    /// Not a command this daemon has. Carries the closest match, if there is a
    /// plausible one — a typo is the overwhelmingly common case.
    Unknown {
        typed: String,
        did_you_mean: Option<&'static str>,
    },
    /// The command needs an argument and got none.
    NeedsArgument {
        command: &'static str,
        argument: &'static str,
    },
    /// The command's argument has internal structure that was not supplied.
    Malformed {
        command: &'static str,
        expected: &'static str,
    },
}

impl ParseError {
    pub fn message(&self) -> String {
        match self {
            ParseError::Unknown {
                typed,
                did_you_mean,
            } => match did_you_mean {
                Some(m) => format!("no command `/{typed}` — did you mean `/{m}`?"),
                None => format!("no command `/{typed}`"),
            },
            ParseError::NeedsArgument { command, argument } => {
                format!("`/{command}` needs {argument}")
            }
            ParseError::Malformed { command, expected } => {
                format!("`/{command}` expects {expected}")
            }
        }
    }
}

pub fn lookup(name: &str) -> Option<&'static Command> {
    CATALOG.iter().find(|c| c.name == name)
}

/// The closest command name to a typo, if one is close enough to suggest.
///
/// Prefix match first — an operator who typed half a name meant that name — then
/// edit distance with a tight bound. A loose bound is worse than none: suggesting
/// `/say` for `/map` sends somebody down the wrong path with confidence.
fn nearest(typed: &str) -> Option<&'static str> {
    if typed.is_empty() {
        return None;
    }
    if let Some(c) = CATALOG.iter().find(|c| c.name.starts_with(typed)) {
        return Some(c.name);
    }
    CATALOG
        .iter()
        .map(|c| (c.name, distance(typed, c.name)))
        .filter(|(_, d)| *d <= 2)
        .min_by_key(|(_, d)| *d)
        .map(|(n, _)| n)
}

/// Levenshtein, two rows. Small inputs — command names are one word.
fn distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let sub = prev[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// One physical act, as the character will read it.
///
/// `{arg}` is the operator's own words and `{who}` is left standing for
/// [`Parsed::attributed_to`]. Trailing punctuation in the argument is not
/// doubled: an operator who ends the line with a full stop meant it, and
/// `hard.` followed by the template's own `.` reads as a typo in the world.
fn did(template: &str, arg: &str) -> EventKind {
    let arg = arg.trim_end_matches(['.', '!', ' ']);
    EventKind::Description {
        text: template.replace("{arg}", arg),
    }
}

/// Turn an operator's line into an event.
///
/// A line with no leading `/` is `say` — the common case is talking to the
/// character, and making that the one thing you have to type a command for would
/// be backwards.
pub fn parse(line: &str) -> Result<Parsed, ParseError> {
    let line = line.trim();
    let Some(rest) = line.strip_prefix('/') else {
        return Ok(Parsed {
            kind: EventKind::Speech {
                speaker: "you".into(),
                text: line.to_string(),
                to: Addressed::You,
            },
            salience: Salience::new(0.6),
            command: "say",
        });
    };

    let (name, arg) = match rest.split_once(char::is_whitespace) {
        Some((n, a)) => (n, a.trim()),
        None => (rest, ""),
    };
    let name = name.to_lowercase();

    let Some(cmd) = lookup(&name) else {
        return Err(ParseError::Unknown {
            typed: name.clone(),
            did_you_mean: nearest(&name),
        });
    };
    let salience = Salience::new(cmd.salience);

    // Commands that take nothing are checked first, so `/wake anything` is not
    // silently accepted with the argument thrown away.
    let kind = match cmd.name {
        "wake" => EventKind::Heartbeat,
        // The day is filled in by the caller, which is the only thing that knows
        // the clock. Zero here is a placeholder the scheduler overwrites.
        "sleep" => EventKind::Sleep { day: 0 },
        _ if arg.is_empty() => {
            return Err(ParseError::NeedsArgument {
                command: cmd.name,
                argument: cmd.argument,
            })
        }
        "say" => EventKind::Speech {
            speaker: "you".into(),
            text: arg.to_string(),
            to: Addressed::You,
        },
        // Overheard from the operator's console has no named addressee to
        // resolve — whoever it was for, it was not this character.
        "overhear" => EventKind::Speech {
            speaker: "someone nearby".into(),
            text: arg.to_string(),
            to: Addressed::Other {
                who: "somebody else".into(),
            },
        },
        "see" | "urgent" => EventKind::Description {
            text: arg.to_string(),
        },
        /* **Somebody did this to you, and the character has to know who.**
         *
         * The shape deliberately mirrors how speech reads to a character —
         * *"X says to you: …"* — so an act and an utterance from the same person
         * in the same room arrive as the same kind of sentence with the same
         * name at the front. `{who}` is filled in by [`Parsed::attributed_to`],
         * written as a placeholder rather than passed in because the parser is
         * handed a line and nothing else.
         *
         * The colon is what makes the operator's own words safe to drop in:
         * they can write `shakes your hand` or `puts a boot through your knee`
         * without having to conjugate it into somebody else's sentence.
         *
         * **It does not say "to you", and that is the point.** An act does not
         * always land on the character — the person in the room can reload
         * their own weapon, bind their own arm, put a hand on the wall — and a
         * template that asserted a target would make every one of those a lie
         * the character then reasons from. Who it lands on is in the operator's
         * own words, which is the same answer the character's `act` gives with
         * its `on` argument. */
        "act" => did("{who} does this: {arg}", arg),
        "notice" => {
            let Some((entity, obs)) = arg.split_once(':') else {
                return Err(ParseError::Malformed {
                    command: "notice",
                    expected: "<entity> : <what it is doing>",
                });
            };
            let (entity, obs) = (entity.trim(), obs.trim());
            if entity.is_empty() || obs.is_empty() {
                return Err(ParseError::Malformed {
                    command: "notice",
                    expected: "<entity> : <what it is doing>",
                });
            }
            EventKind::Entity {
                entity_id: entity.to_string(),
                observation: obs.to_string(),
            }
        }
        // A console cannot type a newline into a single-line input, and a
        // situation is two sentences, so the escape is what makes the second
        // one reachable at all.
        "here" => EventKind::Situation {
            text: arg.replace("\\n", "\n"),
        },
        other => {
            // Unreachable while the catalog and this match agree — and
            // `every_command_in_the_catalog_parses` is what keeps them agreeing.
            return Err(ParseError::Unknown {
                typed: other.to_string(),
                did_you_mean: None,
            });
        }
    };

    Ok(Parsed {
        kind,
        salience,
        command: cmd.name,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The common case is talking to the character, so it is the one that needs
    /// no syntax.
    /// **The speaker is named, or the character reads "you says to you".**
    ///
    /// The parser cannot know whose console a line came from, so it writes a
    /// placeholder and the caller substitutes. Left unsubstituted it renders
    /// through `EventKind::prose` as an ungrammatical sentence that is also
    /// wrong about who spoke — and it went to the model exactly like that.
    #[test]
    fn a_speaker_can_be_named_after_parsing() {
        let p = parse("where were you last night?")
            .unwrap()
            .attributed_to("Johnathan Sharratt");
        let EventKind::Speech { speaker, to, .. } = &p.kind else {
            panic!("a bare line is speech");
        };
        assert_eq!(speaker, "Johnathan Sharratt");
        assert_eq!(*to, Addressed::You, "naming the speaker moved the aim");

        let rendered = crate::engine::event::Event::new(0, 0, p.salience, p.kind.clone()).prose();
        assert_eq!(
            rendered,
            "Johnathan Sharratt says to you: where were you last night?"
        );
        assert!(!rendered.starts_with("you says"), "{rendered}");
    }

    /// Naming a speaker on something nobody said leaves it alone.
    #[test]
    fn naming_a_speaker_touches_nothing_that_is_not_speech() {
        let before = parse("/sleep").unwrap();
        let after = parse("/sleep").unwrap().attributed_to("Johnathan Sharratt");
        assert_eq!(before.kind, after.kind);
    }

    // ── who did it ──────────────────────────────────────────────────────────

    /// **A blow has to name the person who landed it.** A character told only
    /// that it has been hit cannot fear anybody, answer anybody, or hold it
    /// against anybody afterwards — and whoever did it is standing in the room.
    #[test]
    fn a_physical_act_names_who_did_it() {
        let p = parse("/act shakes your hand")
            .unwrap()
            .attributed_to("Wren");
        let EventKind::Description { text } = &p.kind else {
            panic!("an act reads as something that happened");
        };
        assert_eq!(text, "Wren does this: shakes your hand");
    }

    /// **An act does not always land on the character.** The person in the room
    /// can reload their own weapon or bind their own arm, and a template that
    /// asserted a target would make every one of those a lie the character then
    /// reasons from. Who it lands on is in the operator's own words.
    #[test]
    fn an_act_does_not_assert_who_it_landed_on() {
        let p = parse("/act reloads his own rifle, not looking at you")
            .unwrap()
            .attributed_to("Wren");
        let EventKind::Description { text } = &p.kind else {
            panic!("an act reads as something that happened");
        };
        assert!(!text.contains("to you"), "{text}");
    }

    /// **One command, not a menu of synonyms.** Thirteen verbs — strike, stab,
    /// shoot, kill, grab, shove and the rest — all produced the same event: same
    /// kind, same agent, same side of the preempt bar. They differed only in the
    /// sentence they rendered, which the operator was already typing. A command
    /// earns its place by being a difference the engine can act on.
    #[test]
    fn there_is_one_way_to_act_on_a_character() {
        for gone in [
            "strike", "stab", "shoot", "kill", "aim", "threaten", "grab", "shove", "restrain",
            "disarm", "touch", "give", "take", "hurt",
        ] {
            assert!(
                lookup(gone).is_none(),
                "/{gone} is back — say it with /act, or show what the engine does differently"
            );
        }
        assert!(lookup("act").is_some());
    }

    /// The operator's words go in whole, so any phrasing works without being
    /// conjugated into somebody else's sentence.
    #[test]
    fn an_act_takes_the_operators_own_words() {
        for what in [
            "shakes your hand",
            "puts a boot through your knee",
            "drags you off the console by the collar",
        ] {
            let p = parse(&format!("/act {what}"))
                .unwrap()
                .attributed_to("Wren");
            let EventKind::Description { text } = &p.kind else {
                panic!("an act reads as something that happened");
            };
            assert!(text.ends_with(what), "{text}");
        }
    }

    /// An act reads the way speech from the same person in the same room does —
    /// name, then colon, then what it was. Two channels, one sentence shape.
    #[test]
    fn an_act_reads_like_speech_from_the_same_person() {
        let said = parse("/say get back").unwrap().attributed_to("Wren");
        let did = parse("/act shoves you back").unwrap().attributed_to("Wren");
        let rendered =
            |p: &Parsed| crate::engine::event::Event::new(0, 0, p.salience, p.kind.clone()).prose();
        assert!(rendered(&said).starts_with("Wren "), "{}", rendered(&said));
        assert!(rendered(&did).starts_with("Wren "), "{}", rendered(&did));
    }

    /// **No placeholder ever reaches a character.** `{who}` is machinery, and a
    /// character handed `{who} strikes you` reads it as literally as anything
    /// else it is told.
    #[test]
    fn the_placeholder_never_survives_into_the_world() {
        for cmd in CATALOG {
            let p = parse(cmd.example).expect(cmd.example).attributed_to("Wren");
            let rendered = format!("{:?}", p.kind);
            assert!(
                !rendered.contains("{who}"),
                "/{} leaked its placeholder: {rendered}",
                cmd.name
            );
        }
    }

    /// Harm with nobody behind it — a round out of the dark, masonry off a roof
    /// — is the room happening to the character, not somebody acting on it.
    #[test]
    fn harm_from_nobody_names_nobody() {
        // No attacker to name — masonry off a roof, a round from the dark. That
        // is `/urgent`, and it is the same event `/hurt` used to make with a
        // prefix the operator can type themselves.
        let p = parse("/urgent a bolt comes out of the dark and takes you through the shoulder")
            .unwrap()
            .attributed_to("Wren");
        let EventKind::Description { text } = &p.kind else {
            panic!("urgent is a description");
        };
        assert!(!text.contains("Wren"), "{text}");
        assert!(p.salience.preempts());
    }

    /// Being taken hold of is not something anybody finishes their turn first.
    #[test]
    fn an_act_preempts() {
        let cmd = lookup("act").expect("in the catalogue");
        assert!(
            Salience::new(cmd.salience).preempts(),
            "/act let the character finish what it was doing"
        );
    }

    /// The console groups the palette by this and shows the summary in it. A
    /// catalogue this size is unreadable as one flat list, and a command with
    /// neither reads as a blank row.
    #[test]
    fn every_command_is_grouped_and_summarised() {
        for c in CATALOG {
            assert!(!c.group.is_empty(), "/{} has no group", c.name);
            assert!(!c.summary.is_empty(), "/{} has no summary", c.name);
            assert!(
                !c.emits.is_empty(),
                "/{} says nothing about what it emits",
                c.name
            );
        }
    }

    /// An overheard line already names its speaker, and is not the caller.
    #[test]
    fn an_overheard_line_keeps_its_own_speaker() {
        let p = parse("/overhear the gate is open")
            .unwrap()
            .attributed_to("Johnathan");
        let EventKind::Speech { speaker, .. } = &p.kind else {
            panic!("overhearing is speech");
        };
        assert_eq!(speaker, "someone nearby");
    }

    #[test]
    fn a_bare_line_is_speech_to_the_character() {
        let p = parse("where were you last night?").unwrap();
        assert_eq!(p.command, "say");
        assert_eq!(
            p.kind,
            EventKind::Speech {
                speaker: "you".into(),
                text: "where were you last night?".into(),
                to: Addressed::You
            }
        );
    }

    /// **A typo must not become dialogue.** Sending `/hrut` to the character as
    /// speech is the worst possible outcome, because it looks like it worked.
    #[test]
    fn an_unknown_command_is_an_error_and_never_speech() {
        let e = parse("/ubrgent badly").unwrap_err();
        match e {
            ParseError::Unknown {
                ref typed,
                did_you_mean,
            } => {
                assert_eq!(typed, "ubrgent");
                assert_eq!(did_you_mean, Some("urgent"));
            }
            other => panic!("expected Unknown, got {other:?}"),
        }
        assert!(e.message().contains("did you mean `/urgent`"));
    }

    /// A loose suggestion bound is worse than none — it sends somebody down the
    /// wrong path with confidence.
    #[test]
    fn a_wild_command_suggests_nothing() {
        let e = parse("/xyzzyqwerty").unwrap_err();
        assert!(matches!(
            e,
            ParseError::Unknown {
                did_you_mean: None,
                ..
            }
        ));
    }

    #[test]
    fn a_prefix_suggests_the_command_it_starts() {
        assert_eq!(nearest("ove"), Some("overhear"));
        assert_eq!(nearest("sl"), Some("sleep"));
    }

    #[test]
    fn what_cannot_wait_preempts_and_ordinary_sight_does_not() {
        assert!(parse("/act shoves you back").unwrap().salience.preempts());
        assert!(parse("/urgent the beam gives").unwrap().salience.preempts());
        assert!(!parse("/see it is raining").unwrap().salience.preempts());
        assert!(!parse("/overhear a rumour").unwrap().salience.preempts());
    }

    #[test]
    fn directed_and_overheard_speech_differ() {
        let say = parse("/say answer me").unwrap();
        let hear = parse("/overhear he is lying").unwrap();
        assert!(matches!(
            say.kind,
            EventKind::Speech {
                to: Addressed::You,
                ..
            }
        ));
        assert!(matches!(
            hear.kind,
            EventKind::Speech {
                to: Addressed::Other { .. },
                ..
            }
        ));
    }

    #[test]
    fn notice_splits_on_the_first_colon() {
        let p = parse("/notice a scout : moving along the ridge : slowly").unwrap();
        assert_eq!(
            p.kind,
            EventKind::Entity {
                entity_id: "a scout".into(),
                observation: "moving along the ridge : slowly".into()
            }
        );
    }

    #[test]
    fn notice_without_a_colon_is_malformed() {
        assert!(matches!(
            parse("/notice a scout moving").unwrap_err(),
            ParseError::Malformed {
                command: "notice",
                ..
            }
        ));
        assert!(matches!(
            parse("/notice  : moving").unwrap_err(),
            ParseError::Malformed { .. }
        ));
    }

    /// A single-line console input cannot contain a newline, so the escape is
    /// what makes a two-sentence situation reachable at all.
    #[test]
    fn a_situation_unescapes_newlines_and_replaces_the_one_before_it() {
        let p = parse("/here You are in the green room.\\nMaker-04 is here.").unwrap();
        let EventKind::Situation { text } = &p.kind else {
            panic!("not a situation");
        };
        assert_eq!(text, "You are in the green room.\nMaker-04 is here.");
        assert_eq!(p.kind.replaces().as_deref(), Some("situation"));
        // A situation says the world moved, not that anything wants answering.
        assert!(!p.salience.preempts());
    }

    /// `/wake` takes nothing, and must not silently accept and discard an
    /// argument — that would hide a mistyped command that happens to start right.
    #[test]
    fn argumentless_commands_do_not_need_one() {
        assert_eq!(parse("/wake").unwrap().kind, EventKind::Heartbeat);
        assert!(matches!(
            parse("/sleep").unwrap().kind,
            EventKind::Sleep { .. }
        ));
        assert!(parse("/wake").unwrap().salience.preempts());
    }

    #[test]
    fn a_command_missing_its_argument_says_what_it_needs() {
        let e = parse("/say").unwrap_err();
        assert!(matches!(
            e,
            ParseError::NeedsArgument { command: "say", .. }
        ));
        assert!(e.message().contains("needs"));
    }

    #[test]
    fn commands_are_case_insensitive() {
        assert_eq!(parse("/ACT shakes your hand").unwrap().command, "act");
        assert_eq!(parse("/Say hello").unwrap().command, "say");
    }

    /// **The catalog and the parser must not drift.** The catalog is what the
    /// console's autocomplete offers; a command listed there and unhandled here
    /// is an operator typing something that looks supported and gets an error.
    #[test]
    fn every_command_in_the_catalog_parses() {
        for c in CATALOG {
            let line = format!("/{} {}", c.name, sample_arg(c.name));
            let p = parse(&line).unwrap_or_else(|e| {
                panic!(
                    "catalog command /{} does not parse: {}",
                    c.name,
                    e.message()
                )
            });
            assert_eq!(p.command, c.name);
            assert_eq!(
                p.salience.get(),
                Salience::new(c.salience).get(),
                "/{} parses with a salience the catalog does not declare",
                c.name
            );
        }
    }

    /// Each catalog entry's own `example` must parse. An example that does not
    /// work is worse than no example — it is documentation that lies.
    #[test]
    fn every_documented_example_parses() {
        for c in CATALOG {
            let p = parse(c.example).unwrap_or_else(|e| {
                panic!("/{}'s documented example fails: {}", c.name, e.message())
            });
            assert_eq!(
                p.command, c.name,
                "/{}'s example invokes another command",
                c.name
            );
        }
    }

    fn sample_arg(name: &str) -> &'static str {
        match name {
            "wake" | "sleep" => "",
            "notice" => "a scout : moving",
            "here" => "You are in the green room.",
            _ => "something happened",
        }
    }

    #[test]
    fn catalog_names_are_unique() {
        let mut n: Vec<&str> = CATALOG.iter().map(|c| c.name).collect();
        let len = n.len();
        n.sort_unstable();
        n.dedup();
        assert_eq!(n.len(), len, "two commands share a name");
    }
}
