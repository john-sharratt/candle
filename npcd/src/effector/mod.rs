//! The effector device — one fixed HTTP surface a character acts through.
//!
//! A body keeps its hands for what it does with itself (speech, movement); for
//! everything the *world* supplies — terminals, lifts, ledgers, the phone — it
//! reaches for a handheld effector, and the effector addresses the world at
//! `http://local/...`. The full design is `docs/the_effector.md`; this module
//! is its foundation slice, CPU-testable with no model behind it.
//!
//! The concerns, one per file:
//!
//! - [`token`] — the device's identity: one opaque, persisted secret per
//!   character, resolvable to a body in O(1).
//! - [`auth`] — the device-only middleware: a bearer token, and never the
//!   gateway's `x-tokera-*`. It is the reason this surface cannot reuse
//!   [`crate::guard::Api`], which hardcodes the operator role check.
//! - [`namespace`] — part id → world-API namespace, the one place the mapping
//!   from a catalogue part to the path segment its routes mount under lives.
//! - [`router`] — the `local` axum router and its near-you index, `GET
//!   http://local/`, composed from where the caller's body stands.
//! - [`enact_route`] — the reusable route envelope: one mapping from an act's
//!   [`Outcome`](crate::engine::body::Outcome) to an HTTP response, shared by
//!   every migrated namespace's routes.
//! - [`lift`] — the lift's routes (`GET`/`OPTIONS`/`POST`), the first namespace
//!   lifted onto the `local` router over the real act dispatch.
//! - [`station`] — the generic station routes: one sub-router that lifts every
//!   part namespace's verbs onto `local` at once, driven by the tool catalogue
//!   ([`crate::engine::tools`]), mounted per namespace by [`router`].
//! - [`here`] — the world-state acts as routes (`/here`): the catalogue's
//!   non-part-bound acts on things and rooms (`claim`, `operate`, `give`, …),
//!   reachable wherever the body stands, with their live `Choices` sets as the
//!   schema (§7.2, Appendix G).
//! - [`phone`] — the personal messaging surface (`/phone`): the migrated phone
//!   acts over the real dispatch, reachable wherever the body stands (§7.2).
//! - [`history`] — the personal read-only window onto the body's witnessed past
//!   (`/history`), a non-destructive peek at the world log (§7.2).
//! - [`selfsurface`] — the personal `/self` reads: a character's own plan,
//!   orders, beliefs and memory, rendered from the shared cast (§7.2, C.27).
//! - [`plan`] — the projected store (`/plan`, `/orders`): `plan_*`/`orders_*`
//!   write the substrate agency layer, which projects into the next turn's
//!   prompt (§9.2, C.11/C.21, Appendix E "The bridge").
//! - [`reshape`] — runtime topology mutation (`/reshape/:world`): the
//!   operator/embedder surface onto [`World::reshape`](npc_map::World::reshape),
//!   direct-scope only, that adds or drowns rooms in a running world and writes
//!   the change back to the authored map (§8.3, Appendix F).
//!
//! The engine drives the router in-process through
//! [`crate::engine::runtime::Runtime::effector_invoke`] (the fast path, §8.1),
//! and the same router is mounted externally under `/v1/local` (§8.2).

pub mod auth;
pub mod enact_route;
pub mod here;
pub mod history;
pub mod lift;
pub mod namespace;
pub mod phone;
pub mod plan;
pub mod reshape;
pub mod router;
pub mod selfsurface;
pub mod station;
pub mod token;
