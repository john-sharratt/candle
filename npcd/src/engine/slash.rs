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
//! Typing `/hurt badly, left arm` into the console is a debugging instrument. It
//! puts an event on the character's inbox exactly as a world would, at a salience
//! high enough to preempt, and the Pulse view shows the tick it causes and what
//! comes out. That loop — poke, watch, poke again — is the whole point.
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

use crate::engine::event::{EventKind, Salience};

/// A command an operator can type.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Command {
    pub name: &'static str,
    /// What the rest of the line means for this command.
    pub argument: &'static str,
    pub description: &'static str,
    /// How urgently the resulting event is taken. Named on the command rather
    /// than typed each time, because "does this preempt" is a property of the
    /// kind of thing that happened, not of the operator's mood.
    pub salience: f32,
    pub example: &'static str,
}

/// Every `/` command. Served at `GET /v1/commands`.
pub const CATALOG: &[Command] = &[
    Command {
        name: "say",
        argument: "what is said to the character",
        description: "Someone speaks directly to the character. The default if you type a bare \
                      line with no slash at all.",
        salience: 0.6,
        example: "/say Hess is at the gate and he is asking for you by name",
    },
    Command {
        name: "overhear",
        argument: "what is said nearby",
        description: "Speech the character catches but is not part of. Whether silence is rude \
                      depends on this, so it is its own command.",
        salience: 0.4,
        example: "/overhear two guards, quietly: the quartermaster has been selling the grain",
    },
    Command {
        name: "see",
        argument: "what happens, described",
        description: "Something happens in front of the character. The general-purpose event.",
        salience: 0.5,
        example: "/see the east granary is burning and nobody is fighting it",
    },
    Command {
        name: "notice",
        argument: "<entity> : <what it is doing>",
        description: "A specific entity is observed. Splits on the first colon.",
        salience: 0.5,
        example: "/notice a scout in Hess's colours : moving along the ridge line, unhurried",
    },
    Command {
        name: "map",
        argument: "<zoom> : <ascii>",
        description: "A spatial picture at a zoom band. Replaces the previous map at that band \
                      rather than adding to it.",
        salience: 0.4,
        example: "/map tactical : ..#..\\n.@...",
    },
    Command {
        name: "hurt",
        argument: "what happened and how badly",
        description: "The character takes damage. High salience — preempts whatever it was \
                      doing and forces a tick now.",
        salience: 0.95,
        example: "/hurt a crossbow bolt through the left shoulder, badly",
    },
    Command {
        name: "urgent",
        argument: "what happens",
        description: "Something that cannot wait. Like /see, but preempts.",
        salience: 0.9,
        example: "/urgent the roof beam above you cracks and begins to give",
    },
    Command {
        name: "wake",
        argument: "(nothing)",
        description: "Force a tick now with no new information. Shows you what the character \
                      does with what it already has.",
        salience: 0.85,
        example: "/wake",
    },
    Command {
        name: "sleep",
        argument: "(nothing)",
        description: "End the character's day now: fold the conversation into memory, tombstone \
                      it, and open tomorrow's. Normally the clock does this.",
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
                directed: true,
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
            directed: true,
        },
        "overhear" => EventKind::Speech {
            speaker: "someone nearby".into(),
            text: arg.to_string(),
            directed: false,
        },
        "see" | "urgent" => EventKind::Description {
            text: arg.to_string(),
        },
        "hurt" => EventKind::Description {
            text: format!("You are hurt: {arg}"),
        },
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
        "map" => {
            let Some((zoom, ascii)) = arg.split_once(':') else {
                return Err(ParseError::Malformed {
                    command: "map",
                    expected: "<zoom> : <ascii>",
                });
            };
            let (zoom, ascii) = (zoom.trim(), ascii.trim());
            if zoom.is_empty() || ascii.is_empty() {
                return Err(ParseError::Malformed {
                    command: "map",
                    expected: "<zoom> : <ascii>",
                });
            }
            EventKind::Map {
                zoom: zoom.to_string(),
                // A console cannot type a newline into a single-line input, so
                // the escape is what makes multi-line maps reachable at all.
                ascii: ascii.replace("\\n", "\n"),
                legend: None,
            }
        }
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
    #[test]
    fn a_bare_line_is_speech_to_the_character() {
        let p = parse("where were you last night?").unwrap();
        assert_eq!(p.command, "say");
        assert_eq!(
            p.kind,
            EventKind::Speech {
                speaker: "you".into(),
                text: "where were you last night?".into(),
                directed: true
            }
        );
    }

    /// **A typo must not become dialogue.** Sending `/hrut` to the character as
    /// speech is the worst possible outcome, because it looks like it worked.
    #[test]
    fn an_unknown_command_is_an_error_and_never_speech() {
        let e = parse("/hrut badly").unwrap_err();
        match e {
            ParseError::Unknown {
                ref typed,
                did_you_mean,
            } => {
                assert_eq!(typed, "hrut");
                assert_eq!(did_you_mean, Some("hurt"));
            }
            other => panic!("expected Unknown, got {other:?}"),
        }
        assert!(e.message().contains("did you mean `/hurt`"));
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
    fn hurt_preempts_and_ordinary_sight_does_not() {
        assert!(parse("/hurt a bolt through the shoulder")
            .unwrap()
            .salience
            .preempts());
        assert!(parse("/urgent the beam gives").unwrap().salience.preempts());
        assert!(!parse("/see it is raining").unwrap().salience.preempts());
        assert!(!parse("/overhear a rumour").unwrap().salience.preempts());
    }

    #[test]
    fn directed_and_overheard_speech_differ() {
        let say = parse("/say answer me").unwrap();
        let hear = parse("/overhear he is lying").unwrap();
        assert!(matches!(say.kind, EventKind::Speech { directed: true, .. }));
        assert!(matches!(
            hear.kind,
            EventKind::Speech {
                directed: false,
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
    /// what makes a multi-line map reachable at all.
    #[test]
    fn a_map_unescapes_newlines_and_replaces_its_band() {
        let p = parse("/map tactical : ..#..\\n.@...").unwrap();
        let EventKind::Map { zoom, ascii, .. } = &p.kind else {
            panic!("not a map");
        };
        assert_eq!(zoom, "tactical");
        assert_eq!(ascii, "..#..\n.@...");
        assert_eq!(p.kind.replaces().as_deref(), Some("map:tactical"));
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
        assert_eq!(parse("/HURT the arm").unwrap().command, "hurt");
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
            "map" => "tactical : ..#..",
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
