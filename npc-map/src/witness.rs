//! What a body saw happen.
//!
//! [`crate::perceive`] answers *what is true here* — a snapshot, a pure
//! function of the world at an instant, the same answer however many times it
//! is asked. This answers *what happened while you were not looking*, which is
//! a different kind of question and needs different machinery: an ordered log,
//! a cursor per reader, and a rule about what each reader could make out.
//!
//! Keeping them apart matters for three reasons, and the first is the one that
//! forced it.
//!
//! **A snapshot must be pure.** While the events lived inside the percept, the
//! percept's answer depended on when somebody last called `mark_seen` — so the
//! function documented as *what is true here* was quietly a function of
//! reading history too. Two callers asking the same question of the same world
//! got different answers.
//!
//! **The stream has other readers.** The dispatch board wants who did what and
//! when, not prose. So does a replay, an audit, anything persisting the world.
//! None of them want a percept, and none of them should have to parse one.
//!
//! **They are consumed differently.** A percept is taken when a body is about
//! to act. Events accumulate whether or not anybody looks, and a body that
//! never looks still has to be able to be woken by one.
//!
//! # Censorship lives here
//!
//! What a reader can make out of an event is a perception rule, not a fact
//! about the event, so [`Happening`] is recorded in full and narrowed on the
//! way out. In the same room, everything. From a room that can only *see* into
//! it: a body moving, a console lighting up — and not what the console holds,
//! nor a word of what was said. That asymmetry is the whole reason the vault
//! has rooms people walk to.

use std::collections::BTreeSet;

use crate::world::{Happening, Tick, Where, World};

/// How near a happening was to a body — the whole of what decides whether it
/// was witnessed at all, and in how much detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reach {
    /// In the same room. Everything carries.
    Here,
    /// Somewhere this body can see into. Bodies and lit consoles carry;
    /// subjects and speech do not.
    InSight,
    /// Nothing carries. Another room on this level, another level, another
    /// building — all the same answer.
    OutOfReach,
}

/// The places a body standing somewhere can witness anything at all.
///
/// **This is the only place scope is decided**, and it is a type rather than a
/// filter buried in a loop so that it can be asserted about directly: for any
/// node, the set of places it can witness is exactly itself plus what it can
/// see. Nothing else in the vault is reachable, however loud.
///
/// A further sense would widen this one function and nothing else. Today the
/// vault has sight only, so [`Scope::seen`] is the node's sightlines — doors,
/// plus whatever a map file adds by hand.
#[derive(Clone, Debug)]
pub struct Scope {
    pub here: Where,
    seen: BTreeSet<String>,
}

impl Scope {
    /// What a body at `at` can witness.
    pub fn at(world: &World, at: &Where) -> Scope {
        Scope {
            here: at.clone(),
            seen: world
                .node(at)
                .map(|n| n.visible.iter().cloned().collect())
                .unwrap_or_default(),
        }
    }

    /// How near a place is — the question every event is asked.
    pub fn reach(&self, place: &Where) -> Reach {
        if place == &self.here {
            Reach::Here
        } else if place.area == self.here.area && self.seen.contains(&place.node) {
            Reach::InSight
        } else {
            Reach::OutOfReach
        }
    }

    /// Every place this body could witness something in, its own included.
    ///
    /// Useful for asserting the scope of a room without staging an event in
    /// every other room to find out.
    pub fn places(&self) -> Vec<Where> {
        let mut out = vec![self.here.clone()];
        out.extend(
            self.seen
                .iter()
                .map(|n| Where::new(self.here.area.clone(), n.clone())),
        );
        out
    }
}

/// One thing a particular body made out, already narrowed to what it could
/// tell from where it was standing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Witnessed {
    pub at: Tick,
    /// Who did it, by actor id.
    pub actor: String,
    /// Who did it, as a reader would name them.
    pub name: String,
    pub place: Where,
    /// Whether it happened in the reader's own room, which is what decides
    /// both how much was legible and how it reads.
    pub here: bool,
    /// Whose view this is. Carried rather than assumed, so a `Witnessed` that
    /// has been stored, sent or handed on still knows who made it out — and so
    /// [`Witnessed::mine`] and [`Witnessed::addressed`] cannot fall out of step
    /// with each other the way two booleans could.
    pub reader: String,
    /// What was made out — narrowed, not the whole truth.
    pub what: Happening,
}

impl Witnessed {
    /// Whether the reader is the one it happened to. Only ever true of an
    /// outcome — how a journey the reader began turned out — and it changes
    /// how the clause reads, from *Maker-04 came in* to *you got there*.
    pub fn mine(&self) -> bool {
        self.actor == self.reader
    }

    /// Whether the reader is who it was aimed at.
    ///
    /// Everyone in the room hears an utterance either way; this is the
    /// difference between being told something and watching somebody else be
    /// told it, which are different facts about one event and read differently.
    pub fn addressed(&self) -> bool {
        matches!(&self.what, Happening::Said { to: Some(t), .. } if t == &self.reader)
    }
}

/// Everything a body could make out since it last looked, oldest first.
///
/// Its own doings are left out: an NPC does not need telling what it just did,
/// and a stream that reports it reads like a machine narrating a machine.
pub fn since(world: &World, id: &str) -> Vec<Witnessed> {
    let Some(reader) = world.actor(id) else {
        return Vec::new();
    };
    // **Scope is where the reader was when the thing happened**, not where it
    // is standing now. Reading is not a second act of perception: walking to
    // the green room cannot let you overhear what was said there before you
    // arrived, and cannot take away what was said beside you in the room you
    // just left. Judging old events by a new position does both.
    //
    // The position history is already in the log — every move announces itself
    // with an arrival — so one forward pass tracks it as it goes. The fallback
    // is the reader's current place, which is only reachable if its own arrival
    // has been trimmed out from under it.
    let mut scope = Scope::at(world, &reader.at);
    let mut out = Vec::new();

    for e in world.log() {
        if e.actor == reader.id && e.what == Happening::Arrived {
            scope = Scope::at(world, &e.place);
        }
        if e.at <= reader.looked {
            continue;
        }
        // Your own doings are left out, except how a journey turned out: you
        // know what you did, and you do not yet know whether you got there.
        // Nothing of your own is ever narrowed — the reach rules are about what
        // carries across a room, and nothing has to carry to reach you.
        let mine = e.actor == reader.id;
        let reach = if mine {
            Reach::Here
        } else {
            scope.reach(&e.place)
        };
        let what = if mine {
            match e.what.is_outcome() {
                true => e.what.clone(),
                false => continue,
            }
        } else {
            match legible(reach, &e.what) {
                Some(what) => what,
                None => continue,
            }
        };
        out.push(Witnessed {
            at: e.at,
            actor: e.actor.clone(),
            // Through `name_of`, which answers for somebody who has already
            // left — the event outlives the body, and a reader told about a
            // departure is being told about exactly that case.
            name: world
                .name_of(&e.actor)
                .map(str::to_string)
                .unwrap_or_else(|| e.actor.clone()),
            place: e.place.clone(),
            here: reach == Reach::Here,
            reader: reader.id.clone(),
            what,
        });
    }
    out
}

/// How much of a happening carries at a given nearness, if any of it does.
fn legible(reach: Reach, what: &Happening) -> Option<Happening> {
    // Where somebody meant to go, and their own sense of having got there or
    // given up, happen inside a head. Standing next to them does not help.
    if what.is_private() {
        return None;
    }
    match reach {
        Reach::OutOfReach => None,
        Reach::Here => Some(what.clone()),
        Reach::InSight => match what {
            // Nobody lip-reads across a room.
            Happening::Said { .. } => None,
            // A console lighting up is visible; what is on it is not.
            Happening::TookStation { .. } => Some(Happening::TookStation { subject: None }),
            Happening::LeftStation { .. } => Some(Happening::LeftStation { subject: None }),
            other => Some(other.clone()),
        },
    }
}

/// The stream as a body would recall it — one short paragraph, or nothing.
///
/// A run of things done by one body becomes one sentence: *said something, let
/// go of a character and left*, rather than three clauses each naming the same
/// person again. And a place named once in a sentence is not named again, so
/// *left a station dark in band one and left* rather than *…and left band
/// one*. A log reads as a log; this has to read as a memory of the last
/// minute.
pub fn narrate(world: &World, seen: &[Witnessed]) -> Option<String> {
    if seen.is_empty() {
        return None;
    }
    let seen = &condense(seen);
    let mut sentences: Vec<String> = Vec::new();
    let mut run: Vec<String> = Vec::new();
    let mut whose: Option<String> = None;
    let mut named: BTreeSet<String> = BTreeSet::new();

    for w in seen {
        // You are "you", and you are never in a run with anybody else, so
        // *you got to the command room* stands as its own sentence beside
        // whatever the room was doing while you arrived in it.
        let subject = if w.mine() {
            "You".to_string()
        } else {
            w.name.clone()
        };
        if !matches!(&whose, Some(name) if name == &subject) {
            if let Some(name) = whose.take() {
                sentences.push(format!("{name} {}.", crate::text::list(&run)));
            }
            run.clear();
            named.clear();
            whose = Some(subject);
        }
        if let Some(verb) = verb_phrase(world, w, &mut named) {
            run.push(verb);
        }
    }
    if let Some(name) = whose {
        sentences.push(format!("{name} {}.", crate::text::list(&run)));
    }
    (!sentences.is_empty()).then(|| sentences.join(" "))
}

/// Collapse each body's journey to where it set out from and where it got to.
///
/// **A body crossing your field of view is one thing that happened, not one
/// thing per room it passed through.** Watching from a corridor, the raw stream
/// is a footstep at a time — *left, came in, left, went onto the east run* —
/// which is exactly the log-shaped prose the module exists to avoid, and it
/// gets worse the busier the level is, which is the wrong way round.
///
/// So per body, per stretch of unbroken movement, everything between the first
/// leaving and the last arriving goes. Nothing else is touched, and the order
/// of what is left is the order it happened in: a body that walked in, sat
/// down and walked out again is three things, because sitting down interrupts
/// the journey and is worth remarking on in its own right.
fn condense(seen: &[Witnessed]) -> Vec<Witnessed> {
    let motion = |w: &Witnessed| matches!(w.what, Happening::Arrived | Happening::Left);
    let mut out: Vec<Witnessed> = Vec::new();

    for w in seen {
        if !motion(w) {
            out.push(w.clone());
            continue;
        }
        // The stretch this belongs to is whatever this body has been doing
        // since it last did something that was not walking — looking past
        // everybody else, because a second body crossing the room does not
        // interrupt the first one's journey and must not split it in two.
        let run: Vec<usize> = out
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, p)| p.actor == w.actor)
            .take_while(|(_, p)| motion(p))
            .map(|(i, _)| i)
            .collect();
        match (run.len(), &w.what) {
            // Still going: replace where it had got to with where it is now.
            (2.., Happening::Arrived) => {
                let last = run[0];
                out[last] = w.clone();
            }
            // A departure in the middle of a crossing adds nothing: it was
            // already leaving.
            (2.., Happening::Left) => {}
            // One event so far. Keep both, so a crossing reads as leaving one
            // place and reaching another.
            _ => out.push(w.clone()),
        }
    }
    out
}

/// One thing somebody did, in the past.
///
/// `named` carries the places this sentence has already mentioned, so a second
/// clause about the same room says "left" rather than naming it again.
fn verb_phrase(world: &World, w: &Witnessed, named: &mut BTreeSet<String>) -> Option<String> {
    let node = world.node(&w.place)?;
    let known = !named.insert(node.name.clone());
    // Motion takes a different preposition from standing still: you are *in*
    // band one but you go *into* it.
    let into = if known {
        String::new()
    } else {
        format!(" {} {}", node.stand().toward(), node.name)
    };
    let at = if known {
        String::new()
    } else {
        format!(" {} {}", node.stand().at(), node.name)
    };
    let from = if known {
        String::new()
    } else {
        format!(" {}", node.name)
    };

    Some(match &w.what {
        Happening::Arrived if w.here => "came in".into(),
        Happening::Arrived => format!("went{into}"),
        Happening::Left if w.here => "left".into(),
        Happening::Left => format!("left{from}"),
        Happening::TookStation { subject: Some(s) } => format!("took {s}"),
        Happening::TookStation { subject: None } if w.here => "sat down to work".into(),
        Happening::TookStation { subject: None } => format!("lit a station{at}"),
        Happening::LeftStation { subject: Some(s) } => format!("let go of {s}"),
        Happening::LeftStation { subject: None } if w.here => "got up".into(),
        Happening::LeftStation { subject: None } => format!("left a station dark{at}"),
        // Delivery is by place and direction is by address, so one utterance
        // reads three ways in the same room: told to you, watched being told to
        // somebody else, or said to everybody.
        // Reported, never quoted — an utterance carries what somebody meant to
        // convey rather than the words they used, so quotation marks here would
        // put a sentence in their mouth that nobody said.
        Happening::Said { to: None, words } => format!("said {words}"),
        Happening::Said { words, .. } if w.addressed() => format!("told you {words}"),
        Happening::Said {
            to: Some(other),
            words,
        } => format!("told {} {words}", who(world, other)),
        // A gesture reads the same three ways as an utterance, and for the same
        // reason: it happens in the room, and who it was aimed at is something
        // everybody present can see. Rendered as a plain deed — no "you see
        // that" framing — because it is one.
        Happening::Did { to: None, what } => what.clone(),
        Happening::Did { what, .. } if w.addressed() => format!("{what}, at you"),
        Happening::Did {
            to: Some(other),
            what,
        } => format!("{what}, at {}", who(world, other)),
        // Only ever your own, and read back to you as the answer to what you
        // set in motion — so it names the destination, which is the thing you
        // were waiting on, rather than the room you happen to be standing in.
        Happening::SetOut { toward } => format!("set off for {}", place_of(world, toward)),
        Happening::GotThere { toward } => format!("got to {}", place_of(world, toward)),
        Happening::LostTheWay { toward, why } => {
            format!("never got to {}, {why}", place_of(world, toward))
        }
    })
}

/// A place as a reader would say it, falling back to its id if the map has
/// somehow lost it.
/// A body as a reader would name it, falling back to its id.
fn who(world: &World, id: &str) -> String {
    world
        .name_of(id)
        .map(str::to_string)
        .unwrap_or_else(|| id.to_string())
}

fn place_of(world: &World, at: &Where) -> String {
    world
        .node(at)
        .map(|n| n.name.clone())
        .unwrap_or_else(|| at.node.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::MapSet;

    fn vault() -> World {
        World::new(
            MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/maps"))
                .expect("the vault must load"),
        )
    }

    fn casting(node: &str) -> Where {
        Where::new("vault-casting", node)
    }

    #[test]
    fn nothing_happened_is_nothing_said() {
        let mut w = vault();
        w.enter("m1", "Maker-01", casting("band-one")).unwrap();
        w.mark_seen("m1");
        assert!(since(&w, "m1").is_empty());
        assert!(narrate(&w, &[]).is_none());
    }

    #[test]
    fn a_reader_never_witnesses_itself() {
        let mut w = vault();
        w.enter("m1", "Maker-01", casting("band-one")).unwrap();
        w.mark_seen("m1");
        w.take("m1", Some("cindy")).unwrap();
        assert!(since(&w, "m1").is_empty());
    }

    #[test]
    fn the_stream_is_typed_before_it_is_prose() {
        // The dispatch board and a replay want this, not a paragraph.
        let mut w = vault();
        w.enter("m1", "Maker-01", casting("band-one")).unwrap();
        w.enter("m2", "Maker-02", casting("band-one")).unwrap();
        w.mark_seen("m2");
        w.take("m1", Some("cindy")).unwrap();

        let seen = since(&w, "m2");
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].name, "Maker-01");
        assert!(seen[0].here);
        assert_eq!(
            seen[0].what,
            Happening::TookStation {
                subject: Some("cindy".into())
            }
        );
    }

    #[test]
    fn what_a_body_could_make_out_is_fixed_when_it_happened_not_when_it_reads() {
        // A Maker is spoken to in band one, walks to the green room, and only
        // then looks. It must still have heard what was said beside it — the
        // green room's walls were not between them at the time.
        let mut w = vault();
        w.enter("m1", "Maker-01", casting("band-one")).unwrap();
        w.enter("m2", "Maker-02", casting("band-one")).unwrap();
        w.mark_seen("m1");

        w.say("m2", "the redoubt burned twice").unwrap();
        w.set_off("m1", casting("green-room")).unwrap();
        w.settle();

        let heard: Vec<Happening> = since(&w, "m1").into_iter().map(|s| s.what).collect();
        assert!(
            heard.iter().any(|h| matches!(h, Happening::Said { .. })),
            "walking away unheard it: {heard:?}"
        );
    }

    #[test]
    fn arriving_somewhere_does_not_backdate_what_was_said_before_you_got_there() {
        // The other half of the same rule, and the one that would break the
        // building: if walking into a room let you hear what it had already
        // said, nobody would ever need to be there at the time.
        let mut w = vault();
        w.enter("m1", "Maker-01", casting("band-one")).unwrap();
        w.enter("m2", "Maker-02", casting("green-room")).unwrap();
        w.mark_seen("m1");

        w.say("m2", "said before anybody came in").unwrap();
        w.set_off("m1", casting("green-room")).unwrap();
        w.settle();

        for s in since(&w, "m1") {
            assert!(
                !matches!(s.what, Happening::Said { .. }),
                "overheard the past: {:?}",
                s.what
            );
        }
    }

    #[test]
    fn the_subject_is_stripped_before_it_leaves_the_room() {
        let mut w = vault();
        w.enter("m1", "Maker-01", casting("band-one")).unwrap();
        w.enter("m2", "Maker-02", casting("ring-north")).unwrap();
        w.mark_seen("m2");
        w.take("m1", Some("cindy")).unwrap();

        let seen = since(&w, "m2");
        assert_eq!(seen.len(), 1);
        assert!(!seen[0].here);
        // Narrowed on the way out, not recorded narrow: the world still knows.
        assert_eq!(seen[0].what, Happening::TookStation { subject: None });
        assert!(matches!(
            w.log().last().map(|e| &e.what),
            Some(Happening::TookStation { subject: Some(_) })
        ));
    }
}
