//! Acts that land on a world.
//!
//! [`crate::engine::act`] reads what a character meant to do out of what it
//! said. This is where the ones a *body* can do reach the world it stands in,
//! and where the world's answer comes back.
//!
//! # Not every act is a body act
//!
//! `note_concern` and `set_intent` happen inside a head; `speak` and `move_to`
//! happen in a room. Only the second kind belongs here, and a tool this module
//! does not recognise is reported as such rather than silently succeeding —
//! because an act that quietly did nothing is indistinguishable from one the
//! character chose not to take, and those need completely different fixes.
//!
//! # A refusal is an answer, not an error
//!
//! Every way an act can fail is an ordinary fact about the world that a
//! character has to be able to act on. A full room is not an error; it is a
//! full room. `npc_map` answers with a typed refusal that **names somebody to
//! go and ask** — *cindy is already held by Maker-07* — which is something a
//! mind can do next, where *your premise is stale* is not.
//!
//! So the world's refusals are rendered in the same second person as everything
//! else the character reads, and handed back the way a result is. Nothing here
//! logs a warning and moves on.
//!
//! # Where a character may go, and who it may address
//!
//! Both are read off the world rather than trusted from the call. A
//! destination is a place the map names; an addressee is somebody standing in
//! the same room. Neither is a validation layer bolted over a free-text field —
//! it is the same rule as the tools within reach, which is that **what a body
//! can do is a function of where it is.**

use serde_json::{Map, Value};

use npc_map::world::{Refused, Where};

use crate::engine::act::Act;
use crate::engine::waiting::Kind;
use crate::world::Hosted;

/// What came of an act.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// It happened. Carries how it reads back to the character.
    Did(String),
    /// The world would not have it, and why — in the character's own second
    /// person, because it is going to read this.
    Refused(String),
    /// Not something a body does. The caller handles it, or reports that
    /// nothing does.
    NotOfTheBody,
}

impl Outcome {
    pub fn happened(&self) -> bool {
        matches!(self, Outcome::Did(_))
    }

    /// The line the character reads, if there is one.
    pub fn line(&self) -> Option<&str> {
        match self {
            Outcome::Did(s) | Outcome::Refused(s) => Some(s),
            Outcome::NotOfTheBody => None,
        }
    }
}

/// Whether this tool is one a body performs.
///
/// Every name here is answered by [`perform`]. It used to list two — `wait` and
/// `observe` — that fell through to [`Outcome::NotOfTheBody`], which read as
/// "this happens in a head" when it actually meant "nothing happens at all".
/// The acts whose whole product is the line they come back with.
///
/// **An act that answers a question keeps the world's words**, for the same
/// reason a refusal does: there the prose *is* the information, and there is no
/// later moment that will deliver it. Nothing in the world perceives a document
/// being read — the contents exist in the outcome and nowhere else, so
/// recording `file_read — layers/eras/third.md` and dropping the rest hands a
/// character the fact that it read something and not what it read. It then
/// reads it again, having learned nothing, for as long as it runs.
///
/// A side table rather than a field on [`Tool`], matching how `LIVE` names the
/// world-bound parameters — and held to the catalog by
/// [`tests::every_answering_act_is_a_real_one`] so a rename cannot leave a
/// silent entry behind.
pub const ANSWERS: &[&str] = &[
    // The documents.
    "file_read",
    "file_list",
    "library_read",
    "portrait_prompt_read",
    // The bench's own history, which is not the character's to write.
    "bench_diff",
    "bench_status",
    "bench_log",
    "bench_blame",
    // Reading a thing that stands in a room, which has the same shape and the
    // same defect: what a board says is only ever in the outcome.
    "read",
    "scan",
];

/// Whether this act's product is the line it answers with.
pub fn answers(tool: &str) -> bool {
    ANSWERS.contains(&tool)
}

pub fn is_of_the_body(tool: &str) -> bool {
    matches!(
        tool,
        "say"
            | "tell"
            | "ask"
            | "speak"
            | "gesture"
            | "move_to"
            | "follow"
            | "flee"
            | "observe"
            | "wait_for"
    ) || crate::engine::enact::is_mine(tool)
        || crate::engine::work::is_mine(tool)
}

/// Perform one act against the world a body stands in.
pub fn perform(hosted: &Hosted, body: &str, act: &Act) -> Outcome {
    match act.tool {
        // `speak` is not offered any more, but a model that has seen it will
        // reach for it — and reading it as what it plainly means costs one arm
        // and beats refusing a character for using the word for speaking.
        "say" | "speak" => say(hosted, body, &act.args),
        "tell" => tell(hosted, body, &act.args),
        "ask" => ask(hosted, body, &act.args),
        "gesture" => gesture(hosted, body, &act.args),
        "move_to" => move_to(hosted, body, &act.args),
        // Breaking away and following are journeys with a reason attached. The
        // reason is the character's; the journey is the same one.
        "flee" | "follow" => move_to(hosted, body, &act.args),
        "observe" => observe(hosted, body, &act.args),
        "wait_for" => wait_for(hosted, body, &act.args),
        // The acts that reach what the world *holds* rather than its shape —
        // carrying, working, digging, fighting, the tower. Same dispatch, one
        // file down, because they need the sim as well as the map.
        // What a body does through a station — the record's own acts and the
        // working loop over them. Checked first because the station surface is
        // the larger of the two and its names are unambiguous.
        t if crate::engine::work::is_mine(t) => crate::engine::work::perform(hosted, body, act),
        _ => crate::engine::enact::perform(hosted, body, act),
    }
}

/// Stop and wait for one named thing, and let the person it is about see it.
///
/// **The arming is the caller's**, in `engine::runtime` — this half is what the
/// room sees. Waiting on somebody is not invisible: you look at them, and the
/// silence is aimed rather than empty. That visibility is the whole mechanism
/// against a deadlock: two characters waiting on each other used to sit until
/// something else moved, and now the first wait wakes the second, who has
/// something to answer.
///
/// The pair is `Outcome::Did` plus the parsed wait, so the caller does not have
/// to re-read the arguments to arm what was just announced.
pub fn wait_for(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(kind) = text(args, "for").as_deref().and_then(Kind::parse) else {
        return Outcome::Refused(
            "You meant to wait, but not for what. Wait for someone_speaks, someone_arrives or \
             someone_leaves."
                .into(),
        );
    };
    // Nobody named: the ambient wait. Nothing to show, because nobody is being
    // asked for anything.
    let Some(to) = text(args, "who") else {
        return Outcome::Did(format!("You settle down to wait {}.", kind.as_done()));
    };
    let Some(id) = here_by_name(hosted, body, &to) else {
        return Outcome::Refused(format!("{to} is not here. {}", who_is_here(hosted, body)));
    };
    match hosted.with(|w| w.show(body, Some(&id), kind.as_seen())) {
        Ok(()) => Outcome::Did(format!("You wait, and {to} can see you waiting.")),
        Err(why) => Outcome::Refused(refusal(hosted, &why)),
    }
}

/// A question put to somebody here.
///
/// Speech, and it lands as speech — what differs is that the listener perceives
/// a question, which has an obvious next act where a statement has none. That
/// difference is the whole reason the act exists; see `tools::ASK`.
fn ask(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(about) = text(args, "about") else {
        return Outcome::Refused("You meant to ask something, but not what.".into());
    };
    let Some(to) = text(args, "to") else {
        // Nobody named is a question to the room, which is a fair thing to ask
        // and exactly what `say` carries.
        return match hosted.with(|w| w.say(body, format!("asking {about}"))) {
            Ok(()) => Outcome::Did(format!("You ask the room {about}")),
            Err(why) => Outcome::Refused(refusal(hosted, &why)),
        };
    };
    let Some(id) = here_by_name(hosted, body, &to) else {
        return Outcome::Refused(format!("{to} is not here. {}", who_is_here(hosted, body)));
    };
    match hosted.with(|w| w.tell(body, &id, format!("asking {about}"))) {
        Ok(()) => Outcome::Did(format!("You ask {to} {about}")),
        Err(why) => Outcome::Refused(refusal(hosted, &why)),
    }
}

/// Something done in the room without speaking.
///
/// The non-verbal channel. Without it every act anybody else can perceive is
/// speech, so a character with nothing to say has nothing to *do* that
/// registers — see `npc_map::world::Happening::Did`.
fn gesture(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(intent) = text(args, "intent") else {
        return Outcome::Refused("You meant to show something, but not what.".into());
    };
    let aimed = match text(args, "to") {
        None => None,
        Some(to) => match here_by_name(hosted, body, &to) {
            Some(id) => Some((to, id)),
            None => {
                return Outcome::Refused(format!(
                    "{to} is not here. {}",
                    who_is_here(hosted, body)
                ))
            }
        },
    };
    let at = aimed.as_ref().map(|(_, id)| id.clone());
    match hosted.with(|w| w.show(body, at.as_deref(), intent.clone())) {
        Ok(()) => Outcome::Did(match aimed {
            Some((to, _)) => format!("You show {to}: {intent}"),
            None => format!("You show it: {intent}"),
        }),
        Err(why) => Outcome::Refused(refusal(hosted, &why)),
    }
}

/// Spend the turn finding out rather than doing.
///
/// **It has to come back with something.** An `observe` that returned nothing
/// was a turn spent to learn nothing, which is worse than idling because it
/// looks like diligence. What it returns is what the body can actually make out
/// from where it stands — the same reading the environment gives it, asked for
/// deliberately instead of waiting to be handed one.
fn observe(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(target) = text(args, "target") else {
        return Outcome::Refused("You meant to look at something, but not what.".into());
    };
    let found = hosted.read(|w| {
        let here = w.actor(body).map(|a| a.at.clone())?;
        let place = w.node(&here).map(|n| n.name.clone())?;
        let others: Vec<String> = w
            .actors_at(&here)
            .into_iter()
            .filter(|a| a.id != body)
            .map(|a| a.name.clone())
            .collect();
        Some((place, others))
    });
    let Some((place, others)) = found else {
        return Outcome::Refused("You are nowhere you can look around.".into());
    };
    let company = match others.len() {
        0 => "You are alone here.".to_string(),
        _ => format!("{} is here with you.", npc_map::text::list(&others)),
    };
    Outcome::Did(format!("You look at {target}. You are in {place}. {company}"))
}

fn say(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(intent) = text(args, "intent") else {
        return Outcome::Refused("You meant to say something, but not what.".into());
    };
    // A `to` on a `say` means it was meant for one person and reached for the
    // wrong tool. Honour the meaning rather than the spelling.
    if text(args, "to").is_some() {
        return tell(hosted, body, args);
    }
    match hosted.with(|w| w.say(body, intent.clone())) {
        Ok(()) => Outcome::Did(format!("You say, to the room: {intent}")),
        Err(why) => Outcome::Refused(refusal(hosted, &why)),
    }
}

fn tell(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(intent) = text(args, "intent") else {
        return Outcome::Refused("You meant to say something, but not what.".into());
    };
    let Some(to) = text(args, "to") else {
        // No addressee is a `say` that named the wrong tool, and saying it to
        // the room is what it plainly meant.
        return match hosted.with(|w| w.say(body, intent.clone())) {
            Ok(()) => Outcome::Did(format!("You say, to the room: {intent}")),
            Err(why) => Outcome::Refused(refusal(hosted, &why)),
        };
    };

    // Addressed by the name the character knows them by, resolved against who
    // is actually here. A name it cannot see is not a name it can speak to.
    let Some(id) = here_by_name(hosted, body, &to) else {
        return Outcome::Refused(format!("{to} is not here. {}", who_is_here(hosted, body)));
    };
    match hosted.with(|w| w.tell(body, &id, intent.clone())) {
        Ok(()) => Outcome::Did(format!("You say, to {to}: {intent}")),
        Err(why) => Outcome::Refused(refusal(hosted, &why)),
    }
}

/// Who a character could have addressed, for a refusal that is worth reading.
///
/// A refusal that only says no leaves a character to guess again; one that says
/// who *is* here lets it get the next attempt right.
fn who_is_here(hosted: &Hosted, body: &str) -> String {
    let others: Vec<String> = hosted.read(|w| {
        let Some(here) = w.actor(body).map(|a| a.at.clone()) else {
            return Vec::new();
        };
        w.actors_at(&here)
            .into_iter()
            .filter(|a| a.id != body)
            .map(|a| a.name.clone())
            .collect()
    });
    match others.len() {
        0 => "You are alone.".into(),
        // Agreement matters here because this line is read by a model that is
        // about to write dialogue: "Wyneth Vayne and Perrin Vastwood is here"
        // is the refusal teaching bad grammar back to the thing it refused.
        n => format!(
            "{} {} here.",
            npc_map::text::list(&others),
            npc_map::text::is_are(n)
        ),
    }
}

fn move_to(hosted: &Hosted, body: &str, args: &Map<String, Value>) -> Outcome {
    let Some(want) = text(args, "destination") else {
        return Outcome::Refused("You meant to go somewhere, but not where.".into());
    };
    let Some(dest) = place_by_name(hosted, body, &want) else {
        // **A thing standing in this room is not a place to walk to**, and
        // saying "there is nowhere called that" about something the character
        // can see is a refusal it cannot learn from. It named the desk it is
        // standing beside; being handed a list of other rooms leaves it to
        // guess again, and it guesses the same way, because the room really
        // does have a desk in it.
        if let Some(thing) = thing_here_called(hosted, body, &want) {
            return Outcome::Refused(format!(
                "{thing} is here with you, but it is a thing rather than somewhere to go, so \
                 there is nowhere to walk to. You are already beside it."
            ));
        }
        // Naming what *is* reachable, because a refusal that only says no
        // leaves a character to guess again — and it guessed its way here. The
        // rooms off this level are what it can actually reach in one move.
        return Outcome::Refused(format!(
            "There is nowhere called \"{want}\". {}",
            rooms_on_this_level(hosted, body)
        ));
    };
    let name = hosted.read(|w| w.node(&dest).map(|n| n.name.clone()));
    let name = name.unwrap_or_else(|| want.clone());

    match hosted.with(|w| w.set_off(body, dest.clone())) {
        // **Walking to where you already stand is not a journey.**
        //
        // It was a `Did`, and that is how a character got stuck: standing in
        // the command room, it emitted `move_to — the command room` on every
        // tick for a hundred turns, was told each time that the act had
        // succeeded, and had no reason to try anything else. A `Did` whose text
        // says nothing happened is the worst of both — the caller records a
        // completed act and the character reads a confirmation.
        //
        // It is a refusal, and one worth reading: it names where the character
        // actually is, which is the fact it had wrong.
        // **And it names no rooms.**
        //
        // The first version of this refusal helpfully listed everywhere the
        // character could go instead — and a live cast read that list, which was
        // now the most recent thing in its window, and walked. Into the room it
        // was already in. Which produced the list again. Three characters spent
        // an evening in a loop whose fuel was the refusal meant to end it.
        //
        // A refusal for a movement act must not answer with movement. This one
        // says the journey is over and turns the character back to what is in
        // front of it, which is where its work was all along.
        Ok(0) => Outcome::Refused(format!(
            "You are already at {name}. There is nowhere to walk to, so whatever you want here, \
             do it here."
        )),
        Ok(1) => Outcome::Did(format!("You set off for {name}. It is one stop.")),
        Ok(n) => Outcome::Did(format!("You set off for {name}. It is {n} stops.")),
        Err(why) => Outcome::Refused(refusal(hosted, &why)),
    }
}

/// The rooms on a body's own level, named the way the memory names them.
///
/// Corridors are left out: nobody goes to a corridor, and listing six of them
/// buries the four rooms that are somewhere to be.
/// A part standing in this room that the character has named, as it would name
/// it back.
///
/// Matched loosely — singular or plural, with or without an article — because
/// the room describes them in the plural ("six story desks stand free") and a
/// character asking for one says "a story desk". A refusal that turns on that
/// difference would be refusing its own vocabulary.
fn thing_here_called(hosted: &Hosted, body: &str, want: &str) -> Option<String> {
    let asked = bare(want);
    hosted.read(|w| {
        let here = w.actor(body)?.at.clone();
        let node = w.node(&here)?;
        w.map()
            .parts_at(node)
            .map(|(part, count)| part.count_name(count))
            .find(|name| {
                let named = bare(name);
                named == asked
                    || named.strip_suffix('s') == Some(asked.as_str())
                    || asked.strip_suffix('s') == Some(named.as_str())
            })
    })
}

/// Everywhere this body may walk to: the rooms off its own level and the other
/// levels, **never where it is standing**.
///
/// The grammar's list for `move_to`. Excluding the current room is the point:
/// walking to where you already are was refused, and a refusal is not a lesson
/// — a live cast emitted it every tick for an evening, read its own refusal
/// back as the most recent thing in the window, and emitted it again. Absent
/// from the branch, it is not a mistake the character can make.
pub fn reachable(hosted: &Hosted, body: &str) -> Vec<String> {
    hosted.read(|w| {
        let Some(here) = w.actor(body).map(|a| a.at.clone()) else {
            return Vec::new();
        };
        let rooms = w.map().get(&here.area).into_iter().flat_map(|area| {
            area.nodes
                .iter()
                .filter(|n| n.kind != npc_map::NodeKind::Passage && n.id != here.node)
                .map(|n| n.name.clone())
                .collect::<Vec<_>>()
        });
        // The other levels too, so somewhere off this floor is still nameable —
        // otherwise the answer to "I want the chronicle" is a list without the
        // chronicle in it.
        let levels = w
            .map()
            .areas()
            .filter(|a| !a.nodes.is_empty() && a.id != here.area)
            .map(|a| a.name.clone())
            .collect::<Vec<_>>();
        rooms.chain(levels).collect()
    })
}

fn rooms_on_this_level(hosted: &Hosted, body: &str) -> String {
    let names: Vec<String> = hosted.read(|w| {
        let Some(here) = w.actor(body).map(|a| a.at.clone()) else {
            return Vec::new();
        };
        let Some(area) = w.map().get(&here.area) else {
            return Vec::new();
        };
        area.nodes
            .iter()
            .filter(|n| n.kind != npc_map::NodeKind::Passage && n.id != here.node)
            .map(|n| n.name.clone())
            .collect()
    });
    // The levels too, because a character that asked for somewhere off this
    // floor is told what floors there are rather than only what is on this one
    // — otherwise the answer to "I want the chronicle" is a list that does not
    // contain the chronicle.
    let levels: Vec<String> = hosted.read(|w| {
        let Some(here) = w.actor(body).map(|a| a.at.clone()) else {
            return Vec::new();
        };
        w.map()
            .areas()
            .filter(|a| !a.nodes.is_empty() && a.id != here.area)
            .map(|a| a.name.clone())
            .collect()
    });

    let mut said = match names.is_empty() {
        true => String::from("There is nowhere else on this level."),
        false => format!("On this level: {}.", npc_map::text::list(&names)),
    };
    if !levels.is_empty() {
        said.push_str(&format!(
            " Elsewhere in the building: {}.",
            npc_map::text::list(&levels)
        ));
    }
    said
}

/// A body in the same room, by the name a character would use.
///
/// Names rather than ids, because a name is what the character has: the percept
/// says *Maker-04 is here*, so *Maker-04* is what it can address. Matching is
/// case-insensitive on the whole name — a model that writes `maker-04` meant
/// the person standing in front of it, and refusing that would be pedantry
/// dressed as rigour.
fn here_by_name(hosted: &Hosted, body: &str, name: &str) -> Option<String> {
    let want = name.trim();
    hosted.read(|w| {
        let here = w.actor(body)?.at.clone();
        let others: Vec<_> = w
            .actors_at(&here)
            .into_iter()
            .filter(|a| a.id != body)
            .collect();
        // The whole name, as the situation writes it. Always wins, so two
        // people whose first names collide are still each reachable.
        if let Some(a) = others
            .iter()
            .find(|a| a.name.eq_ignore_ascii_case(want))
        {
            return Some(a.id.clone());
        }
        // **Otherwise the name people are actually called by.**
        //
        // A character asked for "Perrin" and was refused because the world had
        // written down "Perrin Vastwood" — then asked again, every tick, because
        // being refused is not being taught. Requiring the full name is
        // requiring a character to address its companion the way a register
        // does; the prompt saying "name them exactly as written" did not stop
        // it, and would not, because shortening a name is what talking is.
        //
        // Same discipline as [`place_by_name`] and `thing_here_called` directly
        // below: a refusal that turns on a difference the character cannot hear
        // is refusing its own vocabulary.
        //
        // Only when it is unambiguous. Two people here who answer to it means
        // the character has to say which, and the refusal names them both.
        let mut hit = others.iter().filter(|a| {
            a.name
                .split_whitespace()
                .next()
                .is_some_and(|first| first.eq_ignore_ascii_case(want))
        });
        let one = hit.next()?;
        match hit.next() {
            None => Some(one.id.clone()),
            Some(_) => None,
        }
    })
}

/// A place in the world, by the name the character remembers it as.
///
/// Searched on this level first and then across the building, because *the
/// green room* means the one on your own floor unless there is none. Beyond
/// that the memory names a level too, and a character that asks for somewhere
/// it cannot reach is refused before it stands up.
fn place_by_name(hosted: &Hosted, body: &str, want: &str) -> Option<Where> {
    let want = want.trim();
    hosted.read(|w| {
        let here = w.actor(body)?.at.clone();
        let matches = |place: &Where| -> bool {
            w.node(place).is_some_and(|n| {
                n.name.eq_ignore_ascii_case(want) || n.id.eq_ignore_ascii_case(want)
            })
        };

        // This level.
        if let Some(area) = w.map().get(&here.area) {
            if let Some(node) = area
                .nodes
                .iter()
                .find(|n| n.name.eq_ignore_ascii_case(want) || n.id.eq_ignore_ascii_case(want))
            {
                return Some(Where::new(here.area.clone(), node.id.clone()));
            }
        }
        // Anywhere else in the world, in a stable order so the same word means
        // the same room twice.
        if let Some(room) = w
            .map()
            .areas()
            .flat_map(|a| {
                a.nodes
                    .iter()
                    .map(move |n| Where::new(a.id.clone(), n.id.clone()))
            })
            .find(|place| matches(place))
        {
            return Some(room);
        }

        // **A level is somewhere to go.** The memory names levels as well as
        // rooms — "Level 2, the chronicle" — so a character reading it asks for
        // the chronicle, meaning the floor rather than any room on it. Refusing
        // that is refusing the map's own vocabulary: it is exactly what a
        // person means, and the way in is the way in.
        let asked = bare(want);
        w.map()
            .areas()
            .find(|a| a.id.eq_ignore_ascii_case(want) || bare(&a.name).eq_ignore_ascii_case(&asked))
            .and_then(|a| w.map().arrival_in(&a.id))
    })
}

/// A level's name with the parts a character drops taken off both sides.
///
/// The memory prints "Level 2, the chronicle", so a character asks for "the
/// chronicle", "chronicle", or "the chronicle level" depending on how it read
/// the line. All three mean the same floor, and none of them is a mistake worth
/// refusing somebody over.
fn bare(name: &str) -> String {
    let lower = name.trim().to_ascii_lowercase();
    // Articles as well as "the": a character asks for "a story desk" while the
    // room, counting six of them, calls them "story desks". Matching on the
    // article would be refusing the room's own vocabulary back at it.
    let stripped = ["the ", "an ", "a "]
        .iter()
        .find_map(|a| lower.strip_prefix(a))
        .unwrap_or(&lower);
    stripped.trim_end_matches(" level").trim().to_string()
}

/// The world's refusal, in the second person the character reads everything in.
///
/// Every one of these names the fact that stopped the act, and where there is
/// somebody to go and ask, it names them. That is the difference between a
/// refusal a mind can act on and one it can only be stuck behind.
fn refusal(hosted: &Hosted, why: &Refused) -> String {
    match why {
        Refused::NoSuchActor(_) => "You are not anywhere.".into(),
        Refused::NoSuchPlace(_) => "There is no such place.".into(),
        Refused::NoWay { to, .. } => {
            let name = hosted.read(|w| w.node(to).map(|n| n.name.clone()));
            match name {
                Some(name) => format!("There is no way from here to {name}."),
                None => "There is no way from here to there.".into(),
            }
        }
        Refused::NothingToWorkAt => "There is nothing here to work at.".into(),
        Refused::EveryStationTaken { of } => {
            format!("All {of} places here are taken.")
        }
        Refused::AlreadyHeld { subject, by } => {
            format!("{by} has {subject}. You would have to ask.")
        }
        Refused::AlreadyAtAStation => "You are already working at something.".into(),
        Refused::SubjectNeeded { binds } => {
            format!("Working here means taking {binds}, and you named none.")
        }
        Refused::SubjectRefused => "There is nothing here to take hold of.".into(),
        Refused::NotAtAStation => "You are not working at anything.".into(),
        Refused::NoTeleport => "There is nowhere to be called back to.".into(),
        Refused::NotHere { who } => format!("{who} is not here."),
        Refused::SpeakingToYourself => "You are the only one you were talking to.".into(),
    }
}

fn text(args: &Map<String, Value>, key: &str) -> Option<String> {
    let s = args.get(key)?.as_str()?.trim();
    (!s.is_empty()).then(|| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A rename would otherwise leave a dead entry that silently stops an act
    /// answering — the same failure the `LIVE` guard exists for.
    #[test]
    fn every_answering_act_is_a_real_one() {
        for tool in ANSWERS {
            assert!(
                crate::engine::tools::by_name(tool).is_some(),
                "`{tool}` answers but is not in the catalog"
            );
            assert!(
                is_of_the_body(tool),
                "`{tool}` answers but never reaches a world"
            );
        }
    }

    fn vault() -> Hosted {
        Hosted::load(
            "creators-vault",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
        )
        .expect("the shipped vault must load")
    }

    fn at(node: &str) -> Where {
        Where::new("vault-casting", node)
    }

    /// **A thing in the room is not somewhere to walk to, and the refusal says
    /// so rather than listing other rooms.**
    ///
    /// A room describes what stands in it — *six character terminals stand
    /// free* — and a character reaches for one the only way it has: by naming
    /// it as a destination. Told "there is nowhere called that" and handed a
    /// list of rooms, it has learned nothing about the terminal it is standing
    /// beside, so it tries again the same way. Observed doing exactly that,
    /// repeatedly, against the story desks on the writing level.
    #[test]
    fn naming_something_in_the_room_is_refused_as_a_thing_not_a_missing_place() {
        let h = vault();
        h.with(|w| w.enter("m1", "Maker-01", at("band-one")).unwrap());

        // Singular, with an article, against a room that counts six of them.
        for want in ["a character terminal", "character terminals"] {
            let out = perform(&h, "m1", &act("move_to", json!({ "destination": want })));
            let Outcome::Refused(said) = out else {
                panic!("{want} was not refused: {out:?}");
            };
            assert!(
                said.contains("is here with you"),
                "{want} was refused as a missing room: {said}"
            );
            assert!(
                !said.contains("There is nowhere called"),
                "{want} sent it looking for a room: {said}"
            );
        }

        // A genuinely unknown name still gets the list of real rooms, which is
        // what a lost character needs.
        let out = perform(
            &h,
            "m1",
            &act("move_to", json!({ "destination": "the moon" })),
        );
        let Outcome::Refused(said) = out else {
            panic!("an invented place was not refused");
        };
        assert!(said.contains("There is nowhere called"), "{said}");
        assert!(said.contains("On this level:"), "{said}");
    }

    /// Three Makers in the green room, each having seen the others arrive.
    ///
    /// Grounded rather than freshly entered, so a test asserting that nothing
    /// was said is not reading three arrivals and calling them speech.
    fn room() -> Hosted {
        let h = vault();
        h.with(|w| {
            for i in 1..=3 {
                w.enter(format!("m{i}"), format!("Maker-{i:02}"), at("green-room"))
                    .unwrap();
            }
        });
        for i in 1..=3 {
            h.delta(&format!("m{i}"));
        }
        h
    }

    fn act(tool: &'static str, args: Value) -> Act {
        Act {
            tool,
            args: args.as_object().expect("an object").clone(),
        }
    }

    // -- what is and is not a body act -------------------------------------

    #[test]
    fn a_tool_that_happens_inside_a_head_is_not_performed_here() {
        let h = room();
        for inward in ["note_concern", "set_intent", "broadcast_strategy"] {
            let out = perform(&h, "m1", &act_named(inward));
            assert_eq!(out, Outcome::NotOfTheBody, "{inward}");
            assert!(!out.happened());
            assert!(out.line().is_none());
        }
    }

    fn act_named(tool: &str) -> Act {
        // The catalog's names are `&'static str`; a test needs one for a tool
        // this module deliberately does not know.
        Act {
            tool: Box::leak(tool.to_string().into_boxed_str()),
            args: Map::new(),
        }
    }

    #[test]
    fn what_this_module_claims_and_what_it_performs_agree() {
        // A tool named as a body act that falls through to `NotOfTheBody` would
        // be silently dropped by a caller trusting the claim.
        let h = room();
        for tool in ["speak", "move_to", "follow", "flee"] {
            assert!(is_of_the_body(tool), "{tool} not claimed");
            assert_ne!(
                perform(&h, "m1", &act_named(tool)),
                Outcome::NotOfTheBody,
                "{tool} claimed and not performed"
            );
        }
    }

    // -- speaking ----------------------------------------------------------

    #[test]
    fn speaking_to_the_room_lands_in_the_room() {
        let h = room();
        let out = perform(
            &h,
            "m1",
            &act("speak", json!({"intent": "that the redoubt burned twice"})),
        );
        assert!(out.happened(), "{out:?}");
        assert!(out.line().unwrap().contains("to the room"), "{out:?}");
        assert!(h.peek("m2").events.iter().any(|e| e.here));
    }

    #[test]
    fn speaking_to_somebody_here_is_aimed_at_them() {
        let h = room();
        let out = perform(
            &h,
            "m1",
            &act(
                "speak",
                json!({"intent": "that it is done", "to": "Maker-02"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        assert!(out.line().unwrap().contains("to Maker-02"));
        assert!(h.peek("m2").events.iter().any(|e| e.addressed()));
        assert!(!h.peek("m3").events.iter().any(|e| e.addressed()));
    }

    #[test]
    fn speaking_to_somebody_who_is_not_here_is_refused_and_says_so() {
        let h = room();
        h.with(|w| w.enter("far", "Maker-09", at("band-one")).unwrap());

        let out = perform(
            &h,
            "m1",
            &act("speak", json!({"intent": "anything", "to": "Maker-09"})),
        );
        assert!(!out.happened());
        assert!(
            out.line().unwrap().contains("Maker-09 is not here"),
            "{out:?}"
        );
        // And nothing was said at all — not even to the room.
        assert!(h.peek("m2").events.is_empty());
    }

    #[test]
    fn a_name_written_the_way_a_model_writes_it_still_reaches_the_person() {
        // Refusing `maker-02` for the person standing in front of you is
        // pedantry dressed as rigour.
        let h = room();
        let out = perform(
            &h,
            "m1",
            &act("speak", json!({"intent": "yes", "to": "  maker-02 "})),
        );
        assert!(out.happened(), "{out:?}");
    }

    #[test]
    fn speaking_without_saying_what_is_refused_rather_than_sent_empty() {
        let h = room();
        for args in [json!({}), json!({"intent": "   "})] {
            let out = perform(&h, "m1", &act("speak", args));
            assert!(!out.happened());
            assert!(h.peek("m2").events.is_empty(), "an empty utterance landed");
        }
    }

    #[test]
    fn a_body_cannot_talk_to_itself_through_the_to_field() {
        let h = room();
        let out = perform(
            &h,
            "m1",
            &act("speak", json!({"intent": "well then", "to": "Maker-01"})),
        );
        assert!(!out.happened(), "{out:?}");
    }

    // -- moving ------------------------------------------------------------

    #[test]
    fn a_place_on_this_level_is_a_journey_of_one_stop() {
        let h = room();
        let out = perform(
            &h,
            "m1",
            &act("move_to", json!({"destination": "band one"})),
        );
        assert_eq!(
            out,
            Outcome::Did("You set off for band one. It is one stop.".into())
        );
        assert!(h.read(|w| w.actor("m1").unwrap().walk.is_some()));
    }

    #[test]
    fn a_place_on_another_level_says_what_it_will_cost() {
        let h = room();
        let out = perform(
            &h,
            "m1",
            &act("move_to", json!({"destination": "the command room"})),
        );
        let line = out.line().unwrap();
        assert!(line.contains("3 stops"), "{line}");
    }

    #[test]
    fn a_place_that_is_not_in_the_world_is_refused_before_standing_up() {
        let h = room();
        let out = perform(
            &h,
            "m1",
            &act("move_to", json!({"destination": "the observatory"})),
        );
        assert!(!out.happened());
        let line = out.line().unwrap();
        assert!(
            line.contains("nowhere called \"the observatory\""),
            "{line}"
        );
        // And it names somewhere real. A refusal that only says no leaves a
        // character to guess again — and it guessed its way here.
        assert!(line.contains("band one"), "{line}");
        assert!(h.read(|w| w.actor("m1").unwrap().walk.is_none()));
    }

    /// And so it is **refused**, not done.
    ///
    /// It used to be a `Did` whose text said nothing had happened, which is how
    /// a character got pinned: standing in the command room, it emitted
    /// `move_to — the command room` every tick for a hundred turns and was told
    /// each time that the act succeeded.
    #[test]
    fn going_where_you_already_are_is_no_journey() {
        let h = room();
        let out = perform(
            &h,
            "m1",
            &act("move_to", json!({"destination": "the green room"})),
        );
        assert!(!out.happened(), "a walk to nowhere read as a journey: {out:?}");
        let why = out.line().unwrap();
        assert!(why.contains("already"), "{why}");
        assert!(h.read(|w| w.actor("m1").unwrap().walk.is_none()));

        // **And it names nowhere else.**
        //
        // This refusal used to list every room on the level, to be helpful. A
        // live cast read the list — the most recent thing in its window — and
        // walked, into the room it was already in, which produced the list
        // again. The loop's fuel was the refusal meant to end it. A refusal for
        // a movement act must not answer with somewhere to move.
        let rooms = h.read(|w| {
            w.map()
                .children("creators-vault")
                .iter()
                .flat_map(|a| a.nodes.iter().map(|n| n.name.clone()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        });
        for room in rooms.iter().filter(|r| *r != "the green room") {
            assert!(
                !why.contains(room.as_str()),
                "the refusal offers somewhere to go (`{room}`): {why}"
            );
        }
    }

    /// **A character calls its companion what people call each other.**
    ///
    /// The world writes down "Perrin Vastwood"; a character says "Perrin". A
    /// live cast deadlocked on exactly that — the one member trying to talk was
    /// refused every tick for using a first name, and being refused is not being
    /// taught, so it asked again the same way. The prompt telling it to name
    /// them exactly as written did not help and could not: shortening a name is
    /// what talking is.
    #[test]
    fn somebody_is_addressable_by_the_name_they_go_by() {
        let h = vault();
        h.with(|w| {
            w.enter("m1", "Wyneth Vayne", at("green-room")).unwrap();
            w.enter("m2", "Perrin Vastwood", at("green-room")).unwrap();
        });

        // The whole name, as the situation writes it.
        assert_eq!(
            here_by_name(&h, "m1", "Perrin Vastwood").as_deref(),
            Some("m2")
        );
        // And the name they are called by, in any case, with stray spacing.
        assert_eq!(here_by_name(&h, "m1", "Perrin").as_deref(), Some("m2"));
        assert_eq!(here_by_name(&h, "m1", " perrin ").as_deref(), Some("m2"));
        // Never yourself, and never somebody who is not a person here.
        assert_eq!(here_by_name(&h, "m1", "Wyneth"), None);
        assert_eq!(here_by_name(&h, "m1", "Hess"), None);
    }

    /// Two people who answer to it means the character has to say which — and
    /// the full name still reaches each of them.
    #[test]
    fn a_first_name_two_people_share_is_refused_rather_than_guessed() {
        let h = vault();
        h.with(|w| {
            w.enter("m1", "Wyneth Vayne", at("green-room")).unwrap();
            w.enter("m2", "Perrin Vastwood", at("green-room")).unwrap();
            w.enter("m3", "Perrin Aldis", at("green-room")).unwrap();
        });
        assert_eq!(here_by_name(&h, "m1", "Perrin"), None, "guessed between two");
        assert_eq!(
            here_by_name(&h, "m1", "Perrin Aldis").as_deref(),
            Some("m3")
        );
    }

    /// The refusal is read by something about to write dialogue, so it has to
    /// be a sentence: "Wyneth Vayne and Perrin Vastwood **is** here" was the
    /// refusal teaching its own bad grammar back.
    #[test]
    fn the_refusal_that_names_who_is_here_agrees_with_itself() {
        let h = room();
        let one = {
            let h1 = vault();
            h1.with(|w| {
                w.enter("m1", "Maker-01", at("green-room")).unwrap();
                w.enter("m2", "Maker-02", at("green-room")).unwrap();
            });
            who_is_here(&h1, "m1")
        };
        assert!(one.contains("is here"), "{one}");
        let many = who_is_here(&h, "m1");
        assert!(many.contains("are here"), "{many}");
    }

    /// **The person waited on is told**, and that is the mechanism.
    ///
    /// Two characters waiting on each other used to sit until something else
    /// moved. Now the first wait reaches the second as something they perceive,
    /// so they have a reason to speak, and speaking is what ends the wait.
    #[test]
    fn waiting_on_somebody_is_something_they_can_see() {
        let h = vault();
        h.with(|w| {
            w.enter("m1", "Wyneth Vayne", at("green-room")).unwrap();
            w.enter("m2", "Perrin Vastwood", at("green-room")).unwrap();
        });
        for b in ["m1", "m2"] {
            h.delta(b);
        }

        let out = perform(
            &h,
            "m1",
            &act("wait_for", json!({"for": "someone_speaks", "who": "Perrin Vastwood"})),
        );
        assert!(out.happened(), "{out:?}");

        // It reaches the room, aimed at them — the same channel a gesture uses.
        let seen = h.delta("m2");
        let text = format!("{seen:?}");
        assert!(
            text.contains("waiting for you to say something"),
            "the one waited on was not told: {text}"
        );
    }

    /// A wait on nobody asks nothing of anybody, so nothing is shown — and it
    /// is still a wait.
    #[test]
    fn an_unnamed_wait_puts_nobody_under_an_obligation() {
        let h = room();
        let out = perform(&h, "m1", &act("wait_for", json!({"for": "someone_arrives"})));
        assert!(out.happened(), "{out:?}");
        let seen = h.delta("m2");
        assert!(
            !format!("{seen:?}").contains("waiting"),
            "an unnamed wait leaned on somebody: {seen:?}"
        );
    }

    #[test]
    fn a_wait_for_nothing_in_particular_is_refused() {
        let h = room();
        // No `for`: the old `wait`, and the thing the typed act exists to
        // prevent — a wait nothing in the world can ever answer.
        let out = perform(&h, "m1", &act("wait_for", json!({"who": "Maker-02"})));
        let Outcome::Refused(why) = out else {
            panic!("a wait for nothing was allowed: {out:?}");
        };
        assert!(why.contains("someone_speaks"), "{why}");

        // A kind the world cannot settle is refused the same way.
        let out = perform(
            &h,
            "m1",
            &act("wait_for", json!({"for": "the silence to speak"})),
        );
        assert!(!out.happened(), "{out:?}");
    }

    #[test]
    fn waiting_on_somebody_who_is_not_here_is_refused_and_names_who_is() {
        let h = room();
        let out = perform(
            &h,
            "m1",
            &act("wait_for", json!({"for": "someone_speaks", "who": "Hess"})),
        );
        let Outcome::Refused(why) = out else {
            panic!("waited on a ghost: {out:?}");
        };
        assert!(why.contains("Hess is not here"), "{why}");
        assert!(why.contains("Maker-02"), "it did not name who is: {why}");
    }

    #[test]
    fn a_room_name_means_the_one_on_your_own_level_first() {
        // Every level has a north run. Asking for it from the casting floor
        // must not send a body to the chronicle level's.
        let h = vault();
        h.with(|w| w.enter("m1", "Maker-01", at("band-one")).unwrap());
        perform(
            &h,
            "m1",
            &act("move_to", json!({"destination": "the north run"})),
        );
        let toward = h.read(|w| w.actor("m1").unwrap().walk.as_ref().unwrap().toward.clone());
        assert_eq!(toward, at("ring-north"));
    }

    /// **A level is somewhere to go.** The memory names levels as well as
    /// rooms, so a character reading it asks for "the chronicle" — meaning the
    /// floor. Refusing that refuses the map's own vocabulary, and it is what a
    /// Maker actually did: thirty attempts an hour at a name printed in its own
    /// memory, every one of them turned down.
    #[test]
    fn a_level_can_be_walked_to_by_the_name_the_memory_prints() {
        for named in [
            "the chronicle",
            "the chronicle level",
            "chronicle",
            "vault-chronicle",
        ] {
            let h = room();
            let out = perform(&h, "m1", &act("move_to", json!({ "destination": named })));
            assert!(out.happened(), "`{named}` was refused: {out:?}");
            let toward = h.read(|w| w.actor("m1").unwrap().walk.as_ref().unwrap().toward.clone());
            assert_eq!(toward.area, "vault-chronicle", "`{named}` went elsewhere");
        }
    }

    #[test]
    fn a_room_is_preferred_to_a_level_that_shares_its_name() {
        // Rooms are searched first and on your own floor first, so a word that
        // could be either means the nearer thing.
        let h = room();
        perform(
            &h,
            "m1",
            &act("move_to", json!({"destination": "band one"})),
        );
        let toward = h.read(|w| w.actor("m1").unwrap().walk.as_ref().unwrap().toward.clone());
        assert_eq!(toward, at("band-one"));
    }

    #[test]
    fn a_refusal_names_the_levels_as_well_as_the_rooms() {
        // Otherwise the answer to "I want the chronicle" is a list that does
        // not contain the chronicle.
        let h = room();
        let out = perform(
            &h,
            "m1",
            &act("move_to", json!({"destination": "the observatory"})),
        );
        let line = out.line().unwrap();
        assert!(line.contains("Elsewhere in the building"), "{line}");
        assert!(line.contains("chronicle"), "{line}");
    }

    #[test]
    fn a_place_can_be_named_by_its_id_as_well_as_its_name() {
        // The memory says "band one"; a map file says `band-one`. Both are
        // things the character has seen written down.
        let h = room();
        assert!(perform(
            &h,
            "m1",
            &act("move_to", json!({"destination": "band-one"}))
        )
        .happened());
    }

    #[test]
    fn breaking_away_and_following_are_journeys_like_any_other() {
        for tool in ["flee", "follow"] {
            // A fresh world each time: setting off twice from the same body
            // would be testing diversion rather than the act.
            let h = room();
            let out = perform(&h, "m1", &act(tool, json!({"destination": "band one"})));
            assert!(out.happened(), "{tool}: {out:?}");
        }
    }

    // -- the world's answers -----------------------------------------------

    #[test]
    fn a_claim_somebody_else_holds_names_who_to_ask() {
        // The whole point of a typed refusal: it hands the mind something to do
        // next. "Your premise is stale" does not.
        let h = vault();
        h.with(|w| {
            w.enter("m1", "Maker-01", at("band-one")).unwrap();
            w.enter("m2", "Maker-02", at("band-one")).unwrap();
            w.take("m1", Some("cindy")).unwrap();
        });
        let why = h.with(|w| w.take("m2", Some("cindy"))).unwrap_err();
        let line = refusal(&h, &why);
        assert!(line.contains("Maker-01"), "{line}");
        assert!(line.contains("cindy"), "{line}");
        assert!(line.contains("ask"), "{line}");
    }

    #[test]
    fn every_refusal_the_world_can_give_reads_as_a_sentence() {
        // A refusal is something the character reads, so an unrendered variant
        // would put a debug format in front of the model.
        let h = room();
        let all = [
            Refused::NoSuchActor("m9".into()),
            Refused::NoSuchPlace(at("nowhere")),
            Refused::NoWay {
                from: at("band-one"),
                to: at("green-room"),
            },
            Refused::NothingToWorkAt,
            Refused::EveryStationTaken { of: 6 },
            Refused::AlreadyHeld {
                subject: "cindy".into(),
                by: "Maker-07".into(),
            },
            Refused::AlreadyAtAStation,
            Refused::SubjectNeeded {
                binds: "one character".into(),
            },
            Refused::SubjectRefused,
            Refused::NotAtAStation,
            Refused::NoTeleport,
            Refused::NotHere {
                who: "Maker-09".into(),
            },
            Refused::SpeakingToYourself,
        ];
        for why in all {
            let line = refusal(&h, &why);
            assert!(!line.is_empty(), "{why:?} rendered to nothing");
            assert!(line.ends_with('.'), "{why:?}: {line}");
            for leak in ["{", "}", "Refused", "vault-casting", "_"] {
                assert!(!line.contains(leak), "{why:?} leaked {leak:?}: {line}");
            }
        }
    }
}
