//! The house's device bridges (`docs/zen_moon_mind.md` §15).
//!
//! A **bridge** is how the world reaches a kind of real thing on the home
//! network. This crate is the library of them, compiled once and run on an
//! in-network agent: an NPC's effector command is sent to the gateway, which
//! gives it to an agent on a machine in the house, and the agent calls [`work`]
//! (or [`wake`]) here to do the real network commands locally — the same path
//! the git commands take (`docs/replicated_substrate.md` §11).
//!
//! It has no dependency on the engine, the cluster or a GPU: it is transports
//! and device knowledge and nothing else, so it compiles in seconds and the
//! same code links into the daemon that submits the work and the agent that
//! runs it.
//!
//! - [`scope`] — the house network's edge: the member's own private subnets,
//!   and the test every address is held to.
//! - [`finding`] — a device a survey found, and the profile that reaches it.
//! - [`proto`] — the shared protocol libraries (SOAP, a TCP line protocol, UDP).
//! - [`profile`] — one module per kind of device the world can work, each thin
//!   on top of [`proto`].
//! - [`work`] — the one abstraction a caller reaches any device through.

pub mod finding;
pub mod profile;
pub mod proto;
pub mod scope;
pub mod work;

pub use finding::{profile_of, Finding};
pub use scope::{in_house, nets, LocalNet};
pub use work::{profile_act, wake, work};
