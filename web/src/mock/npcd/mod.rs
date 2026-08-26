//! The mocked `npcd` surface: every `/v1/*` route and both `/ws/*` streams,
//! answering with correctly-shaped data and no engine, no model, no GPU.
//!
//! The wire contract is `docs/npc_api_gui_design.md` Part B, and it is the
//! contract that matters here — not these bodies. When the engine lands, `npcd`
//! grows its own `api.rs` against the same routes and stops calling
//! [`router`]; this stays as the console's development fixture, which is the
//! only thing it was ever for.
//!
//! It is reachable two ways, and they are the same code either way: `npcd`
//! serves it as its API today, and `web --authoritative` serves it for the
//! `npcd` site with no daemon running at all.

mod api;
mod data;
mod schema;
mod turns;
mod ws;

pub use api::router;
