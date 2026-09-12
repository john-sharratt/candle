//! What a map file says.
//!
//! One file is one [`Area`] — a world, a region, a building, a level. An area
//! *contains* other areas by id, *holds* [`Node`]s, or does both: the outdoor
//! map has its own places and the buildings standing on it are children of it.
//! Areas join across files by id alone, so a world is built by writing more
//! files rather than by editing a bigger one.
//!
//! # Deriving this from a game world
//!
//! A tile map hands most of it over. Flood-fill the floor parts, bounded by
//! walls: each connected region is a [`Node`], and each door part joining two
//! regions is an exit. What flood-fill cannot supply is the half that makes a
//! memory worth reading — a name, a kind, what a place is for and what it is
//! like — and that comes from metadata on the parts, in exactly the fields
//! below. This schema is the metadata schema; there is no translation layer.
//!
//! **The reference runs one way.** A building instance names the area it is;
//! a marker part names the node it stands in. Nothing here names a
//! coordinate, so a wall moved by a level designer cannot leave this stale —
//! and stale is what it would be, silently, since there is no picture left to
//! check it against.
//!
//! # An author writes attachments, not doors
//!
//! A node says what it opens [`off`](Node::off) — usually one corridor — and
//! [`crate::load`] weaves the doors both ways. A [`Spine`] that loops has its
//! own links woven the same way. So a level of fifteen rooms is fifteen `off:`
//! lines rather than sixty exits that have to agree with each other, and the
//! commonest authoring mistake in the whole schema — a door wired one way —
//! becomes impossible to write.
//!
//! # There is no geometry, and no compass
//!
//! A node has attachments and sightlines and no position. Nothing here could
//! draw a floor plan, and that is the point: the moment a place needs
//! coordinates, every generated place needs them too, and a million of them
//! would each have to stay consistent with a picture nobody looks at.
//!
//! Bearings are gone for the same reason. They described a level nobody can
//! see, cost a line of YAML on every door, and answered no question an NPC
//! asks — *which way round the ring* comes from the [`Spine`], and *how do I
//! get to the green room* is a route, not a heading.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::part::Placement;

/// What sort of place an area is.
///
/// Closed, so a new kind is a compile error everywhere it is described rather
/// than a string that renders as itself in the middle of a sentence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AreaKind {
    World,
    Region,
    Building,
    Level,
}

impl AreaKind {
    /// What a reader calls it, in a sentence.
    pub fn noun(self) -> &'static str {
        match self {
            AreaKind::World => "world",
            AreaKind::Region => "region",
            AreaKind::Building => "building",
            AreaKind::Level => "level",
        }
    }
}

/// What sort of place a node is.
///
/// This is what the description groups by, and the grouping is the whole of
/// why a level reads as a place rather than an inventory. An NPC arriving is
/// choosing between *work*, *something to consult* and *company* — so those
/// are the three headings, and passages are left out of the description
/// entirely because nobody goes to a corridor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeKind {
    /// The lift and the stair. Where an area is entered and left.
    Core,
    /// A corridor. Plumbing: walked through, never gone to.
    Passage,
    /// A room with workstations in it.
    Work,
    /// A room for company — seats, a table, no stations.
    Social,
    /// A room holding something to be read rather than written.
    Store,
    /// Open ground outdoors — a crossroads, a yard, a scrapyard. Somewhere to
    /// be, with no walls making it one.
    ///
    /// Indoors a place is a region of floor bounded by walls, which a tile map
    /// hands over for free. Outdoors nothing bounds anything, so open ground
    /// is authored or derived from roads and landmarks rather than from
    /// geometry — a different job, and the reason it is a different kind.
    Ground,
}

impl NodeKind {
    pub fn slug(self) -> &'static str {
        match self {
            NodeKind::Core => "core",
            NodeKind::Passage => "passage",
            NodeKind::Work => "work",
            NodeKind::Social => "social",
            NodeKind::Store => "store",
            NodeKind::Ground => "ground",
        }
    }

    /// The heading a place of this kind appears under.
    pub fn heading(self) -> Option<&'static str> {
        match self {
            NodeKind::Ground => Some("Open ground"),
            NodeKind::Work => Some("The work here"),
            NodeKind::Store => Some("To consult"),
            NodeKind::Social => Some("For company"),
            NodeKind::Core | NodeKind::Passage => None,
        }
    }

    /// What a reader calls one of these in the plural — "rooms" indoors,
    /// "places" where there are no walls to make a room.
    pub fn collective(self) -> &'static str {
        match self {
            NodeKind::Ground => "place",
            _ => "room",
        }
    }
}

/// The route through an area, named so the description can say its shape.
///
/// A list of corridors is an inventory; a spine is a shape, and one sentence
/// of shape replaces every corridor name in the description. When it loops,
/// [`crate::load`] weaves the links between consecutive passages, so the
/// author writes the order and nothing else.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spine {
    /// What the route is called: "the ring", "the main hall".
    pub name: String,
    /// Whether it closes on itself.
    #[serde(default, rename = "loop")]
    pub loops: bool,
    /// The passages it runs through, in order.
    pub through: Vec<String>,
}

/// One place an NPC can be.
///
/// A node is somewhere a body stands; everything *in* it is a [`Part`], placed
/// by reference. Sixteen workstations are one room holding sixteen terminals,
/// not sixteen rooms — otherwise a chair is a location, and at a million
/// locations that is the whole budget spent on seating.
///
/// Nothing a room contains is described in the room's own file. A terminal is
/// defined once, and every room that has one gets the same name, the same
/// claim and the same tools — including the ones written later.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub kind: NodeKind,
    /// What it is called, lower case, as it appears mid-sentence.
    pub name: String,
    /// Whether the name takes a plural verb. "The stacks *hold*", "the green
    /// room *holds*".
    ///
    /// A flag rather than a guess, because English does not signal it in the
    /// word: *the stacks* is plural, *the address* is not, and no rule over
    /// the letters gets both right at a million place names.
    #[serde(default)]
    pub plural: bool,
    /// How a body occupies this place: *in* the green room, *on* the north
    /// run, *at* the relations table.
    ///
    /// Defaults from the node's kind, which is right nearly everywhere. The
    /// exception is a place named after the thing standing in it — you are
    /// *at* a table, never *in* one — and English gives no rule for telling
    /// those apart, so it is data, for the same reason [`Node::plural`] is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stand: Option<Stand>,
    /// What this node opens onto, by id. Doors are woven both ways at load,
    /// so only one end has to say it.
    #[serde(default)]
    pub off: Vec<String>,
    /// One line of what the place is like. Authored data, never generated at
    /// read time — at scale this is filled once, offline, and stored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub character: Option<String>,
    /// What stands in here, by part id: terminals, boards, chairs.
    ///
    /// This is where the tool surface comes from. A tool reachable at this
    /// node is a tool one of these parts carries, so what an NPC can do is
    /// computed from where its body is rather than declared about who it is.
    #[serde(default)]
    pub parts: Vec<Placement>,
    /// What is underfoot: "dirt", "metal plate", "grass".
    ///
    /// The whole of what a terrain grid contributes to a memory. Not per
    /// pixel — a summary of the surfaces across the place, which is what
    /// somebody standing there would say — and empty indoors, where the floor
    /// is a building part like any other and worth no remark.
    #[serde(default)]
    pub ground: Vec<String>,
    /// What this place is usually like — busy, empty, quiet. Learned, and
    /// deliberately not live: it says where to *look* for somebody without
    /// saying where anybody is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub habit: Option<String>,
    /// Sightlines that are **not** doorways.
    ///
    /// Seeing through your own door is the rule and is woven at load, so this
    /// field carries only what breaks it — a gallery rail over a room it
    /// cannot reach. Storing it the way it is described keeps the two from
    /// drifting apart.
    #[serde(default)]
    pub sees: Vec<String>,

    /// Every node reachable in one step. Woven at load from `off`, from the
    /// spine, and from whatever else names this node; never authored.
    #[serde(skip)]
    pub exits: Vec<String>,
    /// Every node visible from here. Woven at load from the doors plus
    /// `sees`; never authored.
    #[serde(skip)]
    pub visible: Vec<String>,
}

impl Node {
    /// Whether this node's name takes a plural verb.
    pub fn verb<'a>(&self, singular: &'a str, plural: &'a str) -> &'a str {
        if self.plural {
            plural
        } else {
            singular
        }
    }

    /// How a body occupies this place — the node's own answer, or its kind's.
    pub fn stand(&self) -> Stand {
        self.stand.unwrap_or(match self.kind {
            NodeKind::Core => Stand::At,
            NodeKind::Passage | NodeKind::Ground => Stand::On,
            _ => Stand::In,
        })
    }
}

/// How a body occupies a place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Stand {
    In,
    On,
    At,
}

impl Stand {
    /// Where a body *is*: in a room, on a run, at a table.
    pub fn at(self) -> &'static str {
        match self {
            Stand::In => "in",
            Stand::On => "on",
            Stand::At => "at",
        }
    }

    /// Where a body *goes*: into a room, onto a run, to a table.
    pub fn toward(self) -> &'static str {
        match self {
            Stand::In => "into",
            Stand::On => "onto",
            Stand::At => "to",
        }
    }
}

/// One place in a world: which area, and which node in it.
///
/// A coordinate on the map rather than a fact about the world, which is why it
/// lives here — a route is computed from the map alone, and nothing about who
/// is standing where is needed to work one out.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Where {
    pub area: String,
    pub node: String,
}

impl Where {
    pub fn new(area: impl Into<String>, node: impl Into<String>) -> Where {
        Where {
            area: area.into(),
            node: node.into(),
        }
    }

    /// Parse an `area-id/node-id` reference, the form maps write portals in.
    pub fn parse(reference: &str) -> Option<Where> {
        let (area, node) = reference.split_once('/')?;
        (!area.is_empty() && !node.is_empty()).then(|| Where::new(area, node))
    }
}

impl fmt::Display for Where {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.area, self.node)
    }
}

/// A way between two areas — a lift, a stair, a gate.
///
/// Declared by whichever area contains both ends, because neither end can know
/// about the other without the file that joins them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portal {
    /// The two node references it joins, each `area-id/node-id`.
    pub between: [String; 2],
    /// What it is: "a lift", "a stair", "a gate".
    pub kind: String,
}

/// One map file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Area {
    pub id: String,
    pub kind: AreaKind,
    /// What it is called, as a title: "The Casting Floor".
    pub name: String,
    /// The area this one sits inside, by id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub within: Option<String>,
    /// Where it sits in its parent's order — a floor number, a district index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ordinal: Option<u32>,
    /// What this place is for, in one or two sentences. What somebody standing
    /// outside it knows about it.
    pub summary: String,
    /// What it is like to be in. Authored, stored, never inferred.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub character: Option<String>,
    /// What this place has not got, so nobody searches for it.
    #[serde(default)]
    pub lacks: Vec<String>,
    /// What the address system plays here, from time to time.
    ///
    /// The building's standing recordings, in the words of whoever left them.
    /// `npcd`'s [`stir`](../../npcd/src/engine/stir) fixtures are facts about
    /// buildings in general and carry their own prose; a tannoy is the one that
    /// is not, because what a place says to itself is the most particular thing
    /// about it. So the engine holds none and quotes these verbatim.
    ///
    /// Each is a whole sentence ending in a full stop, because it is quoted
    /// inside a framing sentence and a fragment would read as one. They are
    /// dealt from a shuffled bag rather than drawn at random, so the length of
    /// this list is exactly how long the building goes before repeating itself.
    #[serde(default)]
    pub announcements: Vec<String>,
    /// Child areas, by id.
    #[serde(default)]
    pub contains: Vec<String>,
    /// Ways between children, or between this area and elsewhere.
    #[serde(default)]
    pub portals: Vec<Portal>,
    /// Where a body arrives when it enters this area, as `area-id/node-id`.
    ///
    /// A world is bigger than any one character's part of it. A Maker belongs
    /// in the vault and a soldier belongs in a city, and both are the same
    /// world — so "where do I start" is not one answer per world, it is one
    /// answer per **part** of it, and each part names its own.
    ///
    /// Absent for a part with an obvious way in: [`crate::MapSet::arrival_in`]
    /// falls back to the core, which is what a core is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arrival: Option<String>,
    /// The one place in this area a body can teleport to, as `area-id/node-id`.
    ///
    /// Teleporting is a convenience, and a convenience that went anywhere would
    /// dissolve the building: if a Maker could arrive at any room by wishing
    /// it, no corridor would ever be walked and no room would ever be passed
    /// through, so nobody would meet anybody. One destination keeps the cost of
    /// distance everywhere except the trip everyone makes most — back to where
    /// orders are given — and every journey *out* is still walked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teleport_to: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spine: Option<Spine>,
    #[serde(default)]
    pub nodes: Vec<Node>,
}

impl Area {
    pub fn node(&self, id: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Nodes of one kind, in file order.
    ///
    /// File order rather than sorted: an author lists rooms in the order a
    /// reader should meet them, and that ordering is information the schema
    /// would otherwise throw away.
    pub fn of_kind(&self, kind: NodeKind) -> impl Iterator<Item = &Node> {
        self.nodes.iter().filter(move |n| n.kind == kind)
    }

    /// Every node that is somewhere to go rather than something to walk
    /// through — the whole of what a description lists.
    pub fn rooms(&self) -> impl Iterator<Item = &Node> {
        self.nodes
            .iter()
            .filter(|n| !matches!(n.kind, NodeKind::Passage | NodeKind::Core))
    }

    /// Whether this area is outside.
    ///
    /// Open ground is the tell, and it changes what can be said about sight:
    /// indoors, seeing into a room from the corridor it opens off is a rule
    /// that covers nearly every sightline there is, and only what breaks it is
    /// worth a reader's attention. Outdoors nothing bounds anything, so there
    /// is no rule and every sightline is its own fact.
    pub fn outdoors(&self) -> bool {
        self.of_kind(NodeKind::Ground).next().is_some()
    }
}
