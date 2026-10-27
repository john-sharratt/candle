//! Prompt authorization, and drawing before it is given
//! (`docs/guest_draw_orchestration.md`).
//!
//! | Module | What |
//! |---|---|
//! | [`token`] | An authorization: a prompt cleared for a purpose, signed |
//! | [`clearance`] | [`Clearance`] and the [`Held`] picture that opens only with one |
//! | [`cache`] | The authorizations this member has, for their lifetime |
//! | [`check`] | [`ContentCheck`], the check behind an authorization |
//! | [`peer`] | [`PeerAuthorizer`], asking another member while this one draws |
//! | [`draw`] | [`GuestDraw`], submitting a draw and withdrawing it |
//! | [`orchestrate`] | The one call: draw now, authorize on a peer, release on both |
//! | [`route`] | `POST /v1/prompt/authorize` |

pub mod cache;
pub mod check;
pub mod clearance;
pub mod draw;
pub mod orchestrate;
pub mod peer;
pub mod route;
pub mod token;

pub use cache::Cache;
pub use check::{local, Compliance, ContentCheck};
pub use clearance::{AdminWaiver, Clearance, Held};
pub use draw::{Engine, GuestDraw};
pub use orchestrate::{draw, Parts, Released, Stopped};
pub use peer::{ClusterPeers, PeerAuthorizer, PeerVerdict};
pub use token::{Authorization, Invalid, Signer, IMAGE};
