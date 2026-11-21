//! Resolving a character's authored state into the persona its prompt is built
//! from.
//!
//! A persona is owned rather than borrowed, because it crosses from the async
//! authored state into the tick thread, and borrowing would hold the authoring
//! lock across a decode. Its anchor travels both as prose and as a slug: an acting
//! turn gathers its anchor from the projection, but a reflection builds its prompt
//! without one, so the anchor has to be in the frame for both. A character's kind
//! decides where it enters its world and whether the world may order it.

use super::*;

/// What kind of character a body is entered for, from its personality: where in
/// the world it belongs, and whether the world may order it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Kind<'a> {
    /// The personality's `home` — where a character with no remembered room
    /// arrives. `None` arrives at the world's own door.
    pub home: Option<&'a str>,
    /// The personality says `takes_orders: false`: the world gives it no
    /// orders — see [`crate::sim::missions::Missions::set_unordered`].
    pub unordered: bool,
    /// The personality says `moves: attention`: it `focus`es on a room rather
    /// than walking to it — see [`Sim::set_attends`].
    pub attends: bool,
    /// The personality's `idle:` — what its idle wake says (see [`Runtime`]'s
    /// `idles`).
    pub idle: Option<&'a str>,
}

/// Everything the prompt needs about one character, owned.
///
/// Owned rather than borrowed because it crosses from the async world (where the
/// authored state lives behind an `RwLock`) into the tick thread. The alternative
/// is holding that lock across a decode, which would block every authoring write
/// for the length of a generation.
#[derive(Debug, Default, Clone)]
pub struct OwnedPersona {
    pub name: String,
    /// The personality and world this character belongs to, by slug.
    ///
    /// Never rendered — they are what a turn pins its personality's anchor, its
    /// world's setting and its building to in the projection. See
    /// [`crate::engine::identity`].
    pub personality: String,
    pub world_id: String,
    /// Where in its world a character of its personality belongs — the
    /// personality's `home` — and so where its body first enters. Never
    /// rendered.
    pub home: Option<String>,
    /// Whether the world gives a character of its personality no orders — the
    /// personality's `takes_orders: false`. Never rendered; see [`Kind`].
    pub unordered: bool,
    /// Whether it moves by attention — the personality's `moves: attention`.
    /// Never rendered; see [`Kind`].
    pub attends: bool,
    /// Its idle thought — the personality's `idle:`. Never rendered as part of
    /// the persona; see [`Kind`].
    pub idle: Option<String>,
    pub identity: String,
    /// The personality's anchor — the floor every character of it reads.
    ///
    /// **Carried as prose as well as pinned as a slug**, because the two
    /// conversation kinds receive it by different routes and only one of them
    /// gathers. An acting turn opens against the projection, which selects the
    /// `ANCHOR` collection member named by `personality`; a reflection builds its
    /// whole prompt from [`crate::engine::prompt::build_for`] and selects
    /// nothing, so for it the anchor has to be in the frame or it is nowhere.
    ///
    /// It was nowhere. A Maker's anchor opens *"You write a world, and you are
    /// not the only one… you have no world of your own"*, and no reflection had
    /// ever read a word of it — so a character told only `You are Tace.` and
    /// `The world you live in:` placed itself inside the story it writes, gave
    /// itself tools no Maker carries, and invented a colleague.
    pub anchor: String,
    pub manner: String,
    pub beliefs: Vec<String>,
    pub relationships: Vec<String>,
    pub intent: Option<String>,
    pub situation: String,
    pub world: String,
    /// The building this character lives in, as it remembers it. Filled in by
    /// the runtime rather than by the persona source: it comes from the map,
    /// which the authored record knows nothing about.
    pub place: String,
    /// Which part of the world `place` describes — see [`Persona::building`].
    pub building: String,
    /// The body's open mission — see [`Persona::mission`]. Filled in by the
    /// runtime, which owns the missions.
    pub mission: Option<MissionPrompt>,
    /// What the character has written down — see [`Persona::journal`]. Filled in
    /// by the runtime, which owns the journals.
    pub journal: Option<JournalPrompt>,
    /// The channels the character is on — see [`Persona::channels`]. Filled
    /// in by the runtime from the world's threads.
    pub channels: Vec<ChannelSection>,
    pub mode: Mode,
}

impl OwnedPersona {
    /// What kind of character this is, for its body — see [`Kind`].
    pub fn kind(&self) -> Kind<'_> {
        Kind {
            home: self.home.as_deref(),
            unordered: self.unordered,
            attends: self.attends,
            idle: self.idle.as_deref(),
        }
    }

    pub fn as_persona(&self) -> Persona<'_> {
        Persona {
            name: &self.name,
            personality: &self.personality,
            world_id: &self.world_id,
            identity: &self.identity,
            anchor: &self.anchor,
            manner: &self.manner,
            beliefs: &self.beliefs,
            relationships: &self.relationships,
            intent: self.intent.as_deref(),
            situation: &self.situation,
            world: &self.world,
            place: &self.place,
            building: &self.building,
            mission: self.mission.as_ref(),
            journal: self.journal.as_ref(),
            channels: &self.channels,
        }
    }
}
