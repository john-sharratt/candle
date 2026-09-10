//! What is true here, at this instant.
//!
//! **A percept is a point in time, not a change in time.** It says what is
//! standing in front of a body right now, and nothing about how it got that
//! way — what changed since anybody last looked is [`crate::witness`], and it
//! is a different kind of thing: a stream with a cursor per reader, where this
//! is a snapshot with none.
//!
//! Keeping the two apart is what makes this function honest. While the events
//! lived in here, the answer depended on when somebody last called
//! `mark_seen`, so a function documented as *what is true here* was quietly a
//! function of reading history as well. Now two callers asking the same
//! question of the same world get the same answer, always.
//!
//! # A percept says only what the memory cannot
//!
//! [`crate::describe`] is memory — the building, which never changes and is
//! shared by everyone who knows it. An earlier version of this file opened
//! every percept with *the way on: the north run*, which is true, is already
//! written in the memory, does not stop being true, and was therefore being
//! paid for sixteen times a turn for ever. Exits are architecture. So is how
//! many stations a room has. So is which level a room is on.
//!
//! What is left is the variable half: who is here, what is free, who can be
//! seen from here. That is two short paragraphs, and it is short because the
//! memory did the heavy work once.
//!
//! # It reads like noticing, not like a readout
//!
//! Labels invite a model to answer in labels, and a percept in a different
//! register from the memory is a seam to cross every turn. So: sentences, in
//! paragraphs that match the shape of paying attention — where you are and
//! what is in your hands, then what is around you.
//!
//! Nothing empty is mentioned. A corridor with nobody in it is not news, and
//! saying so every turn is how a generator fills a page without adding a fact.
//!
//! # Sight only, and deliberately poor
//!
//! You can see a lit console and the person at it. You cannot see what they
//! are working on. Every social mechanism in the vault depends on that: if
//! what a Maker held could be read from the doorway, nobody would ever need to
//! walk to the green room and ask, and the room would be furniture.
//!
//! # Affordances, never recommendations
//!
//! *Four terminals stand free* is a fact. *You could take one* is a standing
//! order smuggled into a percept — and since every NPC gets the same wording,
//! sixteen of them would act on it identically and it would look like
//! emergence.

use std::fmt::Write as _;

use crate::part::PartKind;
use crate::text::{cap, is_are, list, plural, spell};
use crate::world::{Actor, Where, World};

/// What an NPC can tell about where it is standing, at this instant.
///
/// A pure function of the world: no cursor, no bookkeeping, the same answer
/// every time it is asked of the same state. Two paragraphs, the second
/// dropped when there is nothing around — a body alone in a quiet corridor
/// gets one sentence, which is the right answer.
pub fn percept(world: &World, id: &str) -> String {
    let Some(actor) = world.actor(id) else {
        return String::new();
    };
    if world.node(&actor.at).is_none() {
        return String::new();
    }

    let mut paragraphs: Vec<String> = vec![standing(world, actor)];
    if let Some(p) = around(world, actor) {
        paragraphs.push(p);
    }
    paragraphs.join("\n\n") + "\n"
}

/// What a place is, in the words whoever built it wrote down.
///
/// **A character that walks into a room should meet the room.** Without this a
/// body arriving somewhere new is told only its name and who is in it, which is
/// the same thing it was told about the last four rooms — so every room in the
/// vault reads alike and there is nothing to have an opinion about. The lift
/// being a car and a stairwell sharing one shaft is the difference between a
/// place and a label.
///
/// **Said on arrival and not in the percept**, which is where it started. A
/// percept is what is true *now*, and it is re-sent whenever any of it changes
/// — so a character that had been standing in one room for an hour was handed
/// the room's description again every time somebody walked past. Walking in is
/// the moment the description is news; after that it is furniture.
///
/// Authored, never generated: [`crate::schema::Node::character`] is filled once,
/// offline, and stored, so every reader gets the same room. `None` for a node
/// nobody has described yet, which is the honest answer and reads as a plainer
/// room rather than as a gap.
pub fn what_it_is_like(world: &World, at: &Where) -> Option<String> {
    let said: Vec<&str> = world
        .node(at)?
        .character
        .as_deref()?
        .split_whitespace()
        .collect();
    match said.is_empty() {
        true => None,
        false => Some(said.join(" ")),
    }
}

/// Where you are and what is in your hands.
///
/// The one thing a percept says about what anybody holds, because it is the
/// one hold the reader can see into.
fn standing(world: &World, actor: &Actor) -> String {
    let node = world.node(&actor.at).expect("checked by the caller");
    let place = format!("{} {}", node.stand().at(), place_name(world, &actor.at));

    match (&actor.hold, &actor.walk) {
        (Some(hold), _) => match &hold.subject {
            Some(subject) => format!("You are working {place}, holding {subject}."),
            None => format!("You are working {place}."),
        },
        // Mid-journey, where you are is the smaller half of where you are. The
        // count is in stops rather than doorways, because a stop is a tick and
        // a tick is a turn — it is what the journey will actually cost. A fact,
        // not advice about whether to press on.
        (None, Some(walk)) => {
            let far = match walk.to_go(&actor.at) {
                1 => "one stop away".to_string(),
                n => format!("{} stops away", spell(n)),
            };
            format!(
                "You are {place}, on your way to {}, {far}.",
                place_name(world, &walk.toward)
            )
        }
        (None, None) => format!("You are {place}."),
    }
}

/// Who else is here, what is free, and who can be made out beyond.
fn around(world: &World, actor: &Actor) -> Option<String> {
    let node = world.node(&actor.at).expect("checked by the caller");
    let mut said: Vec<String> = Vec::new();

    let others: Vec<&Actor> = world
        .actors_at(&actor.at)
        .into_iter()
        .filter(|a| a.id != actor.id)
        .collect();
    let working: Vec<String> = others
        .iter()
        .filter(|a| a.hold.is_some())
        .map(|a| a.name.clone())
        .collect();
    let idle: Vec<String> = others
        .iter()
        .filter(|a| a.hold.is_none())
        .map(|a| a.name.clone())
        .collect();

    // Who is here is one sentence and what is free is another. Joining them
    // with "and" reads as one fact about two unrelated things.
    if !working.is_empty() {
        said.push(format!(
            "{} {} working",
            list(&working),
            is_are(working.len())
        ));
    }
    if !idle.is_empty() {
        let tail = if working.is_empty() { "here" } else { "not" };
        said.push(format!("{} {} {tail}", list(&idle), is_are(idle.len())));
    }

    let mut out = String::new();
    if !said.is_empty() {
        let _ = write!(out, "{}.", cap(&said.join(", ")));
    }

    // How many places are left to work is the variable half; how many there
    // are altogether is the memory's, and was being repeated every turn.
    let capacity = world.stations_here(&actor.at);
    if capacity > 0 {
        let free = capacity.saturating_sub(world.stations_taken(&actor.at));
        let station = world
            .map()
            .parts_of(node, PartKind::Station)
            .next()
            .map(|(p, _)| p.count_name(free))
            .unwrap_or_else(|| plural("station", free as usize));
        let clause = match free {
            0 => "Nothing here is free.".to_string(),
            n => format!("{} {station} stand free.", cap(&spell(n as usize))),
        };
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&clause);
    }

    // Only what has somebody in it. An empty corridor is not news, and saying
    // it is empty every turn is how a page gets filled without a fact being
    // added to it.
    let seen: Vec<String> = node
        .visible
        .iter()
        .filter_map(|target| {
            let there = Where::new(actor.at.area.clone(), target.clone());
            let other = world.node(&there)?;
            let who: Vec<String> = world
                .actors_at(&there)
                .into_iter()
                .map(|a| a.name.clone())
                .collect();
            if who.is_empty() {
                return None;
            }
            // "You can see Maker-05 on the north run" — no verb, because the
            // seeing is the verb.
            Some(format!(
                "{} {} {}",
                list(&who),
                other.stand().at(),
                other.name
            ))
        })
        .collect();
    if !seen.is_empty() {
        if !out.is_empty() {
            out.push(' ');
        }
        let _ = write!(out, "You can see {}.", list(&seen));
    }

    (!out.is_empty()).then_some(out)
}

/// The parts a body could reach from where it stands, by id.
///
/// Read off the room, so walking away takes them with it. What *acts* those
/// parts make available is the engine's to say — an act names the stations it
/// attaches to — which is why this returns the things and not a vocabulary.
pub fn within_reach<'a>(world: &'a World, id: &str) -> Vec<&'a str> {
    let Some(actor) = world.actor(id) else {
        return Vec::new();
    };
    match world.node(&actor.at) {
        Some(node) => world.map().part_ids_at(node),
        None => Vec::new(),
    }
}

/// A place's name, with its level added only when the name alone is ambiguous.
///
/// *Band one* exists once in the vault and needs no help. *The north run*
/// exists on all six levels, so it gets one. This is what a person does — you
/// say "the north run on five" and just "the green room" — and it means the
/// commonest case costs nothing.
fn place_name(world: &World, at: &Where) -> String {
    let Some(node) = world.node(at) else {
        return at.node.clone();
    };
    let repeated = world
        .map()
        .areas()
        .filter(|a| a.id != at.area)
        .any(|a| a.nodes.iter().any(|n| n.name == node.name));
    match (repeated, world.map().get(&at.area)) {
        (true, Some(area)) => format!("{} of {}", node.name, area.name),
        _ => node.name.clone(),
    }
}
