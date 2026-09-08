//! Places an NPC can be, and the memory it carries of them.
//!
//! A world is a directory of YAML files. Each file is one [`Area`] — a world,
//! a region, a building, a level — and areas join by id, so a world is grown
//! by adding files rather than by editing a bigger one. The leaves hold
//! [`Node`]s, and a node is one place a body can stand.
//!
//! | File | Concern |
//! |---|---|
//! | [`schema`] | what a map file says |
//! | [`load`] | reading a directory into one joined world |
//! | [`validate`] | what has to be true before anything walks it |
//! | [`describe`] | turning a map into what an NPC remembers of it |
//!
//! # Why this is data and not prose
//!
//! The same building has to be described at three magnifications — the world
//! an NPC carries, the level it is standing on, and the place it is standing —
//! and those three must never disagree. Written by hand they drift, quietly,
//! because nothing checks one paragraph against another. Generated from one
//! file they cannot.
//!
//! The other reason is arithmetic. A game world has millions of places in it.
//! None of them can be hand-written, and none of them can afford inference at
//! the moment somebody walks in. So the structure is authored (or generated),
//! the character of each place is stored beside it, and the description is a
//! pure function of the two — same bytes every time, for nothing.
//!
//! # There is no geometry
//!
//! Nothing here could draw a floor plan, and that is deliberate. Coordinates
//! exist to serve a picture, and a picture that every generated place must
//! stay consistent with is a constraint carried by millions of places to serve
//! something nobody looks at. Navigation needs to know which places adjoin
//! which, and that is a graph.

pub mod delta;
pub mod describe;
pub mod load;
pub mod part;
pub mod perceive;
pub mod route;
pub mod salience;
pub mod schema;
pub mod stream;
pub mod text;
pub mod validate;
pub mod witness;
pub mod world;

pub use delta::{Attention, Delta};
pub use describe::Known;
pub use load::MapSet;
pub use part::{Part, PartKind, Placement};
pub use salience::Weight;
pub use schema::{Area, AreaKind, Node, NodeKind, Portal, Spine, Stand, Where};
pub use stream::Stream;
pub use witness::{Reach, Scope, Witnessed};
pub use world::{Actor, Event, Happening, Lost, Refused, Walk, World};
