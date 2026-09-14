//! Turning a map into what an NPC remembers of it.
//!
//! This is the *learned* half of what an NPC carries: the building it works
//! in, which it knows because it has worked there, not because it is looking
//! at it. It never changes, so it sits at the front of the context and is
//! shared by every NPC that knows the same places.
//!
//! # What it has to be good enough for
//!
//! An NPC reading this must be able to answer three questions without asking
//! anything: *where am I*, *where is the thing I want to do done*, and *how do
//! I get there*. A building memory carries an index of what is taken up where,
//! which answers the second. A level memory lists rooms grouped by what they
//! are for, which answers the first and third together.
//!
//! # Corridors are not described
//!
//! An earlier version named every passage and then indexed the rooms by which
//! one they opened off. It was correct and it was exhausting: a reader had to
//! hold six corridor names in mind to work out where anything was, and the
//! corridors were the one thing nobody ever wanted to go to.
//!
//! So the route is one sentence of *shape* — a ring, a spine, the lift landing
//! on it — and the rooms are listed under what they are for. Which door leads
//! where is a question for the moment somebody is actually walking, and at
//! that moment perception answers it.
//!
//! # Facts are generated; character is stored
//!
//! Everything here is a deterministic function of the map — same input, same
//! bytes out, no inference on the read path. The parts that would otherwise
//! read as machine-written are authored fields (`character`, `habit`) carried
//! through verbatim, and every one is optional: a place with none of them
//! still describes correctly, just drily. That split is what lets this scale.
//!
//! # Description, never instruction
//!
//! A memory says what a place *is*, never what to do about it. "The quiet room
//! is where reading happens undisturbed" is a fact about a room; "go to the
//! quiet room when you need to concentrate" is a standing order smuggled into
//! the world. The second kind is worse here than anywhere else, because this
//! text is cached and shared — one line of it would shape every NPC that knows
//! the building, identically, for ever, and would never appear in a log.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::load::MapSet;
use crate::part::PartKind;
use crate::schema::{Area, AreaKind, Node, NodeKind};
use crate::text::{cap, list, plural, spell, tidy};

/// Which places an NPC has learned.
///
/// The Makers know the whole vault, so they pass [`Known::All`]. A character
/// out in a world knows where it has been, and passes the set. Same renderer
/// either way — the filter is here from the start because retrofitting it once
/// a world has a million places in it would mean rewriting every caller.
#[derive(Debug, Clone)]
pub enum Known {
    All,
    Only(BTreeSet<String>),
}

impl Known {
    pub fn knows(&self, area_id: &str) -> bool {
        match self {
            Known::All => true,
            Known::Only(set) => set.contains(area_id),
        }
    }
}

/// The width generated prose is wrapped to.
const WRAP: usize = 76;

// =========================================================================
// A building
// =========================================================================

/// What an NPC remembers of a building: what it is, what is on each level,
/// and what kind of work is taken up where.
///
/// Deliberately shallow on layout. This is what you know about the levels you
/// are not standing on — enough to decide which one to go to, not enough to
/// work there. The detail arrives from [`level`].
/// Everything a body who lives here knows about where it lives — the whole
/// building and every level of it, in one piece.
///
/// **This is what belongs in a system prompt.** It is learned once and never
/// changes, so it is the prefix every turn is read inside rather than something
/// paid for per tick. About two thousand words for a building of six levels,
/// which is the whole of what somebody who works there could tell you and is
/// cheaper than one turn of getting it wrong.
///
/// Its absence is not a smaller prompt. A body that has not been told the rooms
/// exist cannot name one, so asked to go somewhere it invents a destination —
/// and every invented one is refused, because no such place is on the map.
///
/// Levels come in the order the building lists them, because that is the order
/// a person would be shown them and the order they are numbered in.
pub fn place(set: &MapSet) -> String {
    places(set)
        .into_iter()
        .map(|(_, text)| text)
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// [`place`], one part of the world at a time, keyed by that part's area id and
/// in the order [`place`] joins them: each building with every level of it, and
/// then each area with places of its own that no building claims.
///
/// **One part is what a body standing in it knows.** A world holds more than
/// one building, and the whole of it rendered as "the building you work in"
/// told a character in the Redoubt about six levels of a vault it has never
/// been in — so asked where it was, it named a room in the vault. Keyed by
/// part, a caller hands a body the part it is standing in; see [`enclosing`]
/// for which part that is.
///
/// **A building's own rooms are part of it.** A building may keep its rooms on
/// itself rather than on levels, as the Redoubt does, and those were left out
/// entirely: the building was introduced and its rooms never named, while every
/// level of the vault beside it was described in full. They follow its levels,
/// without a second heading — [`building`] has already said what it is.
pub fn places(set: &MapSet) -> Vec<(String, String)> {
    // Buildings first, then anything with places of its own that no building
    // claimed — a world of open ground has no building and still has to be
    // describable.
    let buildings: Vec<&Area> = set
        .areas()
        .filter(|a| a.kind == AreaKind::Building)
        .collect();

    let mut out = Vec::new();
    for b in &buildings {
        let mut text = building(set, &b.id, &Known::All).trim_end().to_string();
        for level in set.children(&b.id) {
            text.push_str("\n\n");
            text.push_str(self::level(set, &level.id).trim_end());
        }
        if !b.nodes.is_empty() {
            let mut own = String::new();
            level_body(set, b, &mut own, false);
            if !own.trim().is_empty() {
                text.push_str("\n\n");
                text.push_str(own.trim());
            }
        }
        out.push((b.id.clone(), text));
    }

    let claimed: BTreeSet<&str> = buildings
        .iter()
        .flat_map(|b| set.children(&b.id).into_iter().map(|c| c.id.as_str()))
        .chain(buildings.iter().map(|b| b.id.as_str()))
        .collect();
    for area in set.areas() {
        if area.nodes.is_empty() || claimed.contains(area.id.as_str()) {
            continue;
        }
        out.push((
            area.id.clone(),
            self::level(set, &area.id).trim_end().to_string(),
        ));
    }
    out
}

/// Which part of the world [`places`] describes an area under: the building it
/// sits in, or the area itself when no building holds it. A vault level is in
/// the vault, the Redoubt is itself, the waste is itself.
///
/// Walks `within` up to the nearest building. Bounded, because `within` is
/// authored and a file naming its own ancestor as its parent must not hang the
/// caller. `None` for an area the set does not hold.
pub fn enclosing<'a>(set: &'a MapSet, area_id: &str) -> Option<&'a str> {
    let start = set.get(area_id)?;
    let mut here = start;
    for _ in 0..64 {
        if here.kind == AreaKind::Building {
            return Some(here.id.as_str());
        }
        match here.within.as_deref().and_then(|w| set.get(w)) {
            Some(parent) => here = parent,
            None => break,
        }
    }
    Some(start.id.as_str())
}

pub fn building(set: &MapSet, id: &str, known: &Known) -> String {
    let Some(area) = set.get(id) else {
        return String::new();
    };
    let mut out = String::new();

    let children: Vec<&Area> = set
        .children(id)
        .into_iter()
        .filter(|c| known.knows(&c.id))
        .collect();

    let mut head = building_head(area, &children);
    if let Some(landing) = landing(area, &children) {
        let _ = write!(head, " {landing}");
    }
    para(&mut out, &head);
    para(&mut out, &area.summary);
    if let Some(character) = &area.character {
        para(&mut out, character);
    }

    if !children.is_empty() {
        out.push('\n');
        for child in &children {
            let label = match child.ordinal {
                Some(n) => format!("{} {}, {}", cap(child.kind.noun()), n, child.name),
                None => cap(&child.name),
            };
            bullet(
                &mut out,
                &format!("{label} — {}", first_sentence(&child.summary)),
            );
        }
    }

    if let Some(index) = holdings(set, &children) {
        para(&mut out, "What a station takes up, and where:");
        out.push('\n');
        for line in index {
            bullet(&mut out, &line);
        }
    }

    if !area.lacks.is_empty() {
        para(&mut out, &lacks_sentence(&area.lacks));
    }

    out
}

/// "X is a building of six levels, joined by a lift and a stair."
///
/// The count comes from what the reader actually knows, not from what exists,
/// so an NPC that has learned two levels is told about two.
fn building_head(area: &Area, children: &[&Area]) -> String {
    let mut s = format!("{} is a {}", area.name, area.kind.noun());
    if !children.is_empty() {
        let _ = write!(
            s,
            " of {} {}",
            spell(children.len()),
            plural(children[0].kind.noun(), children.len())
        );
    }
    let joins: BTreeSet<&str> = area.portals.iter().map(|p| p.kind.as_str()).collect();
    if !joins.is_empty() {
        let joins: Vec<String> = joins.into_iter().map(String::from).collect();
        let _ = write!(s, ", joined by {}", list(&joins));
    }
    s.push('.');
    s
}

/// "Every level is entered at the same place: the lift and the stair."
///
/// Only said when it is true — every portal in the building landing on a node
/// with one id. It is the most useful navigational fact about a building of
/// repeating floors, and it is the difference between knowing the levels and
/// knowing how to move between them.
fn landing(area: &Area, children: &[&Area]) -> Option<String> {
    if area.portals.is_empty() || children.len() < 2 {
        return None;
    }
    let mut node_ids: BTreeSet<&str> = BTreeSet::new();
    for portal in &area.portals {
        for end in &portal.between {
            let (_, node_id) = MapSet::split_ref(end)?;
            node_ids.insert(node_id);
        }
    }
    let [only] = node_ids.into_iter().collect::<Vec<_>>()[..] else {
        return None;
    };
    // Every child has to agree, or "the same place" is a lie.
    if !children.iter().all(|c| c.node(only).is_some()) {
        return None;
    }
    let name = children.iter().find_map(|c| c.node(only))?.name.clone();
    Some(format!(
        "Every {} is entered at the same place: {name}.",
        children[0].kind.noun()
    ))
}

/// One line per thing a station can take up, and every place it can be taken.
///
/// This is the index that lets an NPC route a job to a level without being
/// told which level to go to. It is generated entirely from `binds`, so a new
/// kind of work appears here the moment a map file mentions it.
fn holdings<'a>(set: &'a MapSet, children: &[&'a Area]) -> Option<Vec<String>> {
    let mut by_bind: BTreeMap<&str, Vec<(String, u32)>> = BTreeMap::new();
    for area in children {
        let mut here: BTreeMap<&str, u32> = BTreeMap::new();
        for node in &area.nodes {
            for (part, n) in set.parts_of(node, PartKind::Station) {
                if let Some(binds) = &part.binds {
                    *here.entry(binds.as_str()).or_default() += n;
                }
            }
        }
        for (binds, count) in here {
            by_bind
                .entry(binds)
                .or_default()
                .push((area.name.clone(), count));
        }
    }
    if by_bind.is_empty() {
        return None;
    }
    Some(
        by_bind
            .into_iter()
            .map(|(binds, places)| {
                let where_: Vec<String> = places
                    .iter()
                    .map(|(name, n)| {
                        format!(
                            "{} {} on {name}",
                            spell(*n as usize),
                            plural("station", *n as usize)
                        )
                    })
                    .collect();
                format!("{}, at {}.", cap(binds), list(&where_))
            })
            .collect(),
    )
}

// =========================================================================
// A level
// =========================================================================

/// What an NPC remembers of one level: everything, because it works there.
///
/// Four beats, and no more: what the level is for, what it is like, the shape
/// of the route through it, and its rooms under three headings. Anything past
/// that is either plumbing or something perception will say better.
pub fn level(set: &MapSet, id: &str) -> String {
    let Some(area) = set.get(id) else {
        return String::new();
    };
    let mut out = String::new();

    para(&mut out, &level_head(set, area));
    para(&mut out, &area.summary);
    if let Some(character) = &area.character {
        para(&mut out, character);
    }
    level_body(set, area, &mut out, true);
    out
}

/// Everything [`level`] says after what a place is: how it hangs together,
/// its rooms under their headings, and what it looks out on.
///
/// Split out so a building that keeps rooms on itself can have them described
/// under [`building`]'s own heading rather than a second one. `with_lacks` is
/// off there because [`building`] has already said what the place has not got.
fn level_body(set: &MapSet, area: &Area, out: &mut String, with_lacks: bool) {
    if let Some(s) = shape(area) {
        para(out, &s);
    }

    for kind in [
        NodeKind::Ground,
        NodeKind::Work,
        NodeKind::Store,
        NodeKind::Social,
    ] {
        let entries = entries(set, area, kind);
        if entries.is_empty() {
            continue;
        }
        para(out, &format!("{}:", kind.heading().unwrap_or("Here")));
        out.push('\n');
        for entry in entries {
            bullet(out, &entry);
        }
    }

    let mut tail: Vec<String> = Vec::new();
    if let Some(s) = sight(area) {
        tail.push(s);
    }
    if let Some(s) = habits(area) {
        tail.push(s);
    }
    if with_lacks && !area.lacks.is_empty() {
        tail.push(lacks_sentence(&area.lacks));
    }
    if !tail.is_empty() {
        para(out, &tail.join(" "));
    }
}

fn level_head(set: &MapSet, area: &Area) -> String {
    let within = area
        .within
        .as_deref()
        .and_then(|w| set.get(w))
        .map(|p| p.name.clone());
    match (area.ordinal, within) {
        (Some(n), Some(parent)) => {
            format!(
                "{} {} of {parent}: {}.",
                cap(area.kind.noun()),
                n,
                area.name
            )
        }
        (Some(n), None) => format!("{} {}: {}.", cap(area.kind.noun()), n, area.name),
        (None, Some(parent)) => format!("{}, in {parent}.", cap(&area.name)),
        (None, None) => format!("{}.", cap(&area.name)),
    }
}

/// One sentence: how the level hangs together, and where you come in.
///
/// Deliberately without a single corridor name. What a reader needs is that
/// the place has one route and everything is off it — from which "nothing is
/// far from anything" follows, and that is the whole of what shape is for.
fn shape(area: &Area) -> Option<String> {
    let core = area.of_kind(NodeKind::Core).next();
    let rooms = area.rooms().count();
    // "Rooms" where walls make them and "places" where nothing does. Outdoors
    // the word matters: nobody calls a crossroads a room.
    let noun = plural(
        if area.outdoors() {
            NodeKind::Ground.collective()
        } else {
            NodeKind::Work.collective()
        },
        rooms,
    );

    let mut s = match area.spine.as_ref() {
        Some(spine) if spine.loops => format!(
            "{} {noun} open off {}, which runs right round the level",
            spell(rooms),
            spine.name
        ),
        Some(spine) => format!(
            "{} {noun} open off {}, which runs the length of the level",
            spell(rooms),
            spine.name
        ),
        None if rooms > 0 => format!("{} {noun} here", spell(rooms)),
        None => return None,
    };
    s = cap(&s);
    match core {
        Some(core) => {
            let _ = write!(
                s,
                ", and {} {} onto it.",
                core.name,
                core.verb("opens", "open")
            );
            if let Some(c) = &core.character {
                let _ = write!(s, " {}", tidy(c));
            }
        }
        None => s.push('.'),
    }
    Some(s)
}

/// The rooms of one kind, as entries under a heading.
///
/// Work rooms that do the same thing are collapsed into one entry — three
/// doors onto the same job are one fact, not three, and that is how somebody
/// who works there thinks of them.
fn entries(set: &MapSet, area: &Area, kind: NodeKind) -> Vec<String> {
    match kind {
        // Rooms holding the same kinds of thing are one fact, not several:
        // three doors onto the same job is how anybody who works there thinks
        // of them, and saying it once is shorter and truer both.
        NodeKind::Work => grouped_entries(set, area, kind),
        _ => area
            .of_kind(kind)
            .map(|n| entry(&n.name, &room_body(set, &[n])))
            .collect(),
    }
}

fn grouped_entries(set: &MapSet, area: &Area, kind: NodeKind) -> Vec<String> {
    let mut groups: BTreeMap<Vec<String>, Vec<&Node>> = BTreeMap::new();
    let mut order: Vec<Vec<String>> = Vec::new();
    for node in area.of_kind(kind) {
        let mut key: Vec<String> = node.parts.iter().map(|p| p.part().to_string()).collect();
        key.sort();
        if !groups.contains_key(&key) {
            order.push(key.clone());
        }
        groups.entry(key).or_default().push(node);
    }
    order
        .into_iter()
        .filter_map(|key| {
            let rooms = groups.get(&key)?;
            let names: Vec<String> = rooms.iter().map(|n| n.name.clone()).collect();
            Some(entry(&list(&names), &room_body(set, rooms)))
        })
        .collect()
}

/// Everything worth saying about one room, or about several that hold the
/// same things.
///
/// Always in the same order — what stands in it, what taking one commits you
/// to, what the things are like, what the place is like — so a reader learns
/// the shape once and can skim the rest.
fn room_body(set: &MapSet, rooms: &[&Node]) -> String {
    let alone = rooms.len() == 1;
    let mut s = String::new();

    let held = contents(set, rooms);
    if !held.is_empty() {
        let tail = if alone { "." } else { " between them." };
        let _ = write!(s, "{}{tail} ", list(&held));
    }

    // What taking one of these commits you to. Generated from the part, so
    // the same terminal makes the same promise in every room it stands in.
    let mut said: BTreeSet<&str> = BTreeSet::new();
    for (part, count) in stations(set, rooms) {
        let Some(binds) = &part.binds else { continue };
        if !said.insert(binds.as_str()) {
            continue;
        }
        let subject = if count == 1 { "It" } else { "Each" };
        let _ = write!(s, "{subject} takes {binds} and holds it until you leave. ");
    }

    // The parts' own clauses, then the room's. A part speaks for itself in
    // every room that has one; the room speaks only for itself.
    let mut clauses: BTreeSet<&str> = BTreeSet::new();
    for room in rooms {
        for (part, _) in set.parts_at(room) {
            if let Some(short) = &part.short {
                if clauses.insert(short.as_str()) {
                    let _ = write!(s, "{} ", tidy(short));
                }
            }
        }
    }
    if alone {
        if let Some(g) = underfoot(rooms[0]) {
            let _ = write!(s, "{g} ");
        }
        if let Some(c) = &rooms[0].character {
            let _ = write!(s, "{} ", tidy(c));
        }
    }
    s
}

/// What stands in these rooms, counted across them and phrased by kind.
fn contents(set: &MapSet, rooms: &[&Node]) -> Vec<String> {
    let mut order: Vec<&str> = Vec::new();
    let mut totals: BTreeMap<&str, (&crate::part::Part, u32)> = BTreeMap::new();
    for room in rooms {
        for (part, n) in set.parts_at(room) {
            let slot = totals.entry(part.id.as_str()).or_insert((part, 0));
            if slot.1 == 0 {
                order.push(part.id.as_str());
            }
            slot.1 += n;
        }
    }
    order
        .into_iter()
        .filter_map(|id| {
            let (part, n) = totals.get(id)?;
            let n = *n;
            // A room named after the thing standing in it does not say the
            // thing twice: *the relations table — the relations table* is what
            // a generator produces and a person never would.
            if rooms.iter().any(|r| r.name == part.name) {
                return None;
            }
            Some(match part.kind {
                // A fixture is one of a kind and is named, not counted.
                PartKind::Fixture => part.name.clone(),
                _ if n == 1 => format!("{} {}", article(&part.name), part.name),
                _ => format!("{} {}", spell(n as usize), part.count_name(n)),
            })
        })
        .collect()
}

/// "a" or "an", by the sound the next word starts with.
///
/// Letters, not phonetics: the exceptions ("an hour", "a university") need a
/// dictionary, and a place name that hits one is worth spelling out in the
/// part's own `plural` or name rather than teaching this to guess.
fn article(word: &str) -> &'static str {
    match word.chars().next() {
        Some(c) if "aeiouAEIOU".contains(c) => "an",
        _ => "a",
    }
}

fn stations<'a>(set: &'a MapSet, rooms: &[&'a Node]) -> Vec<(&'a crate::part::Part, u32)> {
    let mut totals: BTreeMap<&str, (&crate::part::Part, u32)> = BTreeMap::new();
    for room in rooms {
        for (part, n) in set.parts_of(room, PartKind::Station) {
            let slot = totals.entry(part.id.as_str()).or_insert((part, 0));
            slot.1 += n;
        }
    }
    totals.into_values().collect()
}

/// A name, and whatever is worth saying about it.
///
/// The dash only appears when something follows it. A room with no stations,
/// no character and nothing to do is a room whose name is the whole entry, and
/// a trailing "—" there is the sort of thing a generator leaves behind when
/// every field it expected happened to be empty.
fn entry(name: &str, rest: &str) -> String {
    let rest = rest.trim();
    if rest.is_empty() {
        format!("{name}.")
    } else {
        format!("{name} — {rest}")
    }
}

/// What is underfoot, when it is worth saying.
///
/// The whole of what a terrain grid contributes: not a pixel, a summary of
/// the surfaces across a place. Indoors the field is empty and the clause
/// never appears, because a floor that is a building part like any other is
/// not news.
fn underfoot(node: &Node) -> Option<String> {
    if node.ground.is_empty() {
        return None;
    }
    Some(format!("Underfoot, {}.", list(&node.ground)))
}

/// What can be seen from where — as a rule, plus whatever breaks it.
///
/// Listing every doorway both ways round is how a generator gives itself away,
/// and it buries the one line that matters. Sight through a door is the rule;
/// only a sightline that is *not* a door is worth a reader's attention.
fn sight(area: &Area) -> Option<String> {
    let mut odd: Vec<String> = Vec::new();
    let mut said: BTreeSet<(&str, &str)> = BTreeSet::new();
    let mut any_door = false;

    for node in &area.nodes {
        if !node.exits.is_empty() {
            any_door = true;
        }
        for target in &node.visible {
            if node.exits.contains(target) {
                continue;
            }
            let Some(other) = area.node(target) else {
                continue;
            };
            let pair = if node.id < other.id {
                (node.id.as_str(), other.id.as_str())
            } else {
                (other.id.as_str(), node.id.as_str())
            };
            if said.insert(pair) {
                odd.push(format!("between {} and {}", node.name, other.name));
            }
        }
    }

    // The doorway rule is an indoor fact. Outdoors nothing bounds anything, so
    // there is nothing for a rule to be about and every sightline stands on
    // its own.
    let rule = any_door && !area.outdoors();
    if !rule && odd.is_empty() {
        return None;
    }
    let mut s = if rule {
        "You can see into a room from the corridor it opens off, and back out again".to_string()
    } else {
        "You can see".to_string()
    };
    if !odd.is_empty() {
        let joiner = if rule { ", and " } else { " " };
        let _ = write!(s, "{joiner}{}", list(&odd));
    }
    s.push('.');
    Some(s)
}

/// What each place is usually like. Learned, and deliberately not live: it
/// says where to *look* for somebody without saying where anybody is.
fn habits(area: &Area) -> Option<String> {
    let parts: Vec<String> = area
        .nodes
        .iter()
        .filter_map(|n| {
            n.habit
                .as_ref()
                .map(|h| format!("{} {} {h}", n.name, n.verb("is", "are")))
        })
        .collect();
    if parts.is_empty() {
        return None;
    }
    Some(format!("{}.", cap(&list(&parts))))
}

fn lacks_sentence(lacks: &[String]) -> String {
    format!("There is no {}.", list(lacks))
}

// ---- text helpers -------------------------------------------------------

fn para(out: &mut String, text: &str) {
    if !out.is_empty() {
        out.push('\n');
    }
    for line in wrap(text, WRAP) {
        out.push_str(&line);
        out.push('\n');
    }
}

/// An indented item, with the continuation lines hanging under the first so a
/// wrapped entry does not read as two.
fn bullet(out: &mut String, text: &str) {
    for (i, line) in wrap(text, WRAP - 4).into_iter().enumerate() {
        let indent = if i == 0 { "  " } else { "    " };
        let _ = writeln!(out, "{indent}{line}");
    }
}

/// Greedy wrap. Deterministic, which matters more here than being clever:
/// these strings are asserted against expected bytes in the tests.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > width {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

/// The opening sentence of an authored paragraph, for a one-line summary.
fn first_sentence(s: &str) -> String {
    let flat = tidy(s);
    match flat.find(". ") {
        Some(i) => flat[..=i].to_string(),
        None => flat,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_list_of_one_is_just_the_item() {
        assert_eq!(list(&["a".into()]), "a");
    }

    #[test]
    fn a_list_of_two_takes_no_comma() {
        assert_eq!(list(&["a".into(), "b".into()]), "a and b");
    }

    #[test]
    fn a_longer_list_takes_commas_and_a_final_and() {
        assert_eq!(list(&["a".into(), "b".into(), "c".into()]), "a, b and c");
    }

    #[test]
    fn small_numbers_are_spelled_and_large_ones_are_not() {
        assert_eq!(spell(6), "six");
        assert_eq!(spell(16), "sixteen");
        assert_eq!(spell(64), "64");
    }

    #[test]
    fn wrapping_never_splits_a_word_and_never_exceeds_the_width() {
        let text = "the quick brown fox jumps over the lazy dog and keeps on going";
        for line in wrap(text, 20) {
            assert!(line.chars().count() <= 20, "{line:?}");
        }
    }

    #[test]
    fn the_first_sentence_stops_at_the_full_stop() {
        assert_eq!(first_sentence("One. Two. Three."), "One.");
        assert_eq!(first_sentence("Only one"), "Only one");
    }

    fn shipped() -> MapSet {
        MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/maps")).expect("the maps must load")
    }

    fn part<'a>(parts: &'a [(String, String)], id: &str) -> &'a str {
        parts
            .iter()
            .find(|(k, _)| k == id)
            .map(|(_, t)| t.as_str())
            .unwrap_or_else(|| panic!("no part {id}"))
    }

    #[test]
    fn a_building_that_keeps_its_rooms_on_itself_names_them() {
        let set = shipped();
        let parts = places(&set);
        let redoubt = part(&parts, "tower-redoubt").to_lowercase();
        for room in ["muster hall", "gatehouse", "barracks", "foundry", "bridge"] {
            assert!(redoubt.contains(room), "{room} missing from:\n{redoubt}");
        }
    }

    #[test]
    fn one_building_is_not_told_about_the_rooms_of_another() {
        let set = shipped();
        let parts = places(&set);
        let vault = part(&parts, "creators-vault").to_lowercase();
        assert!(!vault.contains("muster hall"), "{vault}");
        let redoubt = part(&parts, "tower-redoubt").to_lowercase();
        assert!(!redoubt.contains("casting"), "{redoubt}");
    }

    #[test]
    fn the_whole_place_is_every_part_of_it_in_order() {
        let set = shipped();
        let joined = places(&set)
            .into_iter()
            .map(|(_, t)| t)
            .collect::<Vec<_>>()
            .join("\n\n");
        assert_eq!(place(&set), joined);
    }

    #[test]
    fn an_area_is_described_under_the_building_that_holds_it() {
        let set = shipped();
        assert_eq!(enclosing(&set, "vault-command"), Some("creators-vault"));
        assert_eq!(enclosing(&set, "creators-vault"), Some("creators-vault"));
        assert_eq!(enclosing(&set, "tower-redoubt"), Some("tower-redoubt"));
        // Open ground has no building over it, so it is its own part.
        assert_eq!(enclosing(&set, "the-waste"), Some("the-waste"));
        assert_eq!(enclosing(&set, "nowhere"), None);
        // Every area resolves to a part that places() actually describes.
        let keys: BTreeSet<String> = places(&set).into_iter().map(|(k, _)| k).collect();
        for area in set.areas() {
            if area.nodes.is_empty() {
                continue;
            }
            let at = enclosing(&set, &area.id).unwrap();
            assert!(keys.contains(at), "{} → {at} is not described", area.id);
        }
    }

    #[test]
    fn a_wrapped_bullet_hangs_its_continuation_lines() {
        let mut out = String::new();
        bullet(&mut out, &"word ".repeat(30));
        let lines: Vec<&str> = out.lines().collect();
        assert!(lines.len() > 1);
        assert!(lines[0].starts_with("  w"));
        assert!(lines[1].starts_with("    w"));
    }
}
