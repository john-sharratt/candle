//! A world's one authority (`docs/replicated_substrate.md` §9.2–§9.6).
//!
//! Every npcd member hosts every world, but one at a time is its authority:
//! the member holding its lease, which runs it and writes its state. The others
//! follow it — its state records as they replicate, its pulses as they arrive —
//! and run the characters lent to them, sending what those characters write to
//! the world to the authority. A member cut off runs a world locally for its own
//! characters and persists nothing of it until it is back. A daemon with no
//! cluster is the authority of every world it hosts and runs every character.
//!
//! | Module | |
//! |---|---|
//! | [`role`] | what this member is to each world |
//! | [`worlds`] | what this member knows of its worlds: roles, what was written, who runs here |
//! | [`beat`] | a world's beat, as the role has it, and its restore from records |
//! | [`world_lease`] | asking for, renewing and handing over a world's lease |
//! | [`driver`] | the leases driven, and the moves between roles |
//! | [`npc_lease`] | which member each character runs on |
//! | [`cast`] | the characters run here kept to their leases |
//! | [`ops`] | a write to a world, as it travels |
//! | [`perform`] | a write made where the world runs |
//! | [`forward`] | an act as the call line it travels as |
//! | [`pulse`] | what the world is doing, as pulses |
//! | [`told`] | what the world tells characters that run elsewhere |
//! | [`follow`] | joining, and the pulses taken in |
//! | [`report`] | the worlds in `COORD` |
//! | [`routes`] | the authority's routes |

pub mod beat;
pub mod cast;
pub mod driver;
pub mod follow;
pub mod forward;
pub mod npc_lease;
pub mod ops;
pub mod perform;
pub mod pulse;
pub mod report;
pub mod role;
pub mod routes;
pub mod told;
pub mod world_lease;
pub mod worlds;

pub use role::{Role, Roles};
pub use worlds::Standing;
