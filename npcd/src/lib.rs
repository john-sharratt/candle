//! `npcd` — the NPC engine, as a library.
//!
//! The `npcd` binary (and `motherd`, which hosts npcd beside zend) is one
//! consumer of this crate and deliberately not a privileged one. The design names four entry points over a single core — the
//! HTTP API, the console, a test harness, and direct crate embedding with no
//! socket at all — and holds that no capability may exist at only one of them.
//! A binary-only crate makes two of those four impossible, so the core is a
//! library and the binary is a thin shim that binds a port over it.
//!
//! That is not only a design point. A test that has to live inside the binary
//! is a test that cannot use what a real consumer uses, and the harness rule is
//! that needing a back door is evidence of a missing capability rather than a
//! reason to add a seam.
//!
//! | Module | What it owns |
//! |---|---|
//! | [`tenant`] | npcd as a tenant of the `host` process — alone in `npcd`, beside zend in `motherd` |
//! | [`reports`] | npcd's share of the host's shared endpoints |
//! | [`engine`] | the cast's loop: perception, the act vocabulary, the tick |
//! | [`api`] | the authored corpus, the mind, the cast, the authoring plane |
//! | [`mind`] | the authored corpus |
//! | [`drafts`] | an admin's writes to the mind: drafts in their own store, published from the console |
//! | [`opstore`] | the stores over the mind repository, an operation's and an author's |
//! | [`registry`] | authored documents, and what may become a file name |
//! | [`projection`] | the schema and the content libraries it reads |

pub mod accounts;
pub mod api;
pub mod authority;
pub mod clock;
pub mod collections;
pub mod compliance;
pub mod console;
pub mod describe;
pub mod drafts;
pub mod effector;
pub mod engine;
pub mod guest_routes;
pub mod guests;
pub mod identity;
pub mod images;
pub mod lifegen;
pub mod mcp;
pub mod mind;
pub mod mind_store;
pub mod model;
pub mod namegen;
pub mod ndjson;
pub mod npcs;
pub mod opstore;
pub mod personality_portrait;
pub mod portrait;
pub mod projection;
pub mod prose;
pub mod refimage;
pub mod registry;
pub mod reports;
pub mod sim;
pub mod substrate;
pub mod tenant;
pub mod visibility;
pub mod world;
pub mod world_state;
