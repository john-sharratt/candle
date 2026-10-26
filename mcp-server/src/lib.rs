//! An MCP server, once (`docs/mcp_app.md`): JSON-RPC, both protocol eras,
//! sessions, the sign-in challenge and progress over an event stream. zend and
//! npcd each serve their own tools on it at `/v1/mcp`, and the `mcp` app
//! serves the tools of both, aggregated, at `/mcp`.
//!
//! An app implements [`Server`] — who it is, who is calling, where sessions
//! are kept, its tools — and mounts [`http::post`], [`http::get`] and
//! [`http::delete`].
//!
//! | Module | What |
//! |---|---|
//! | [`http`] | The routes, the challenge, the event stream a call answers on |
//! | [`protocol`] | The MCP methods in the 2025-11-25 and 2026-07-28 eras |
//! | [`jsonrpc`] | One JSON-RPC message per POST |
//! | [`server`] | What an app supplies: [`Server`], [`Identity`], [`Call`] |
//! | [`session`] | Sessions and the [`Store`] they are kept in |
//! | [`local_store`] | Sessions kept by one process, in memory or in a file |
//! | [`sealed_store`] | Sessions carried whole in their ids, for a server that holds nothing on them |
//! | [`client`] | Which client a session is for, which bounds a call's wait |
//! | [`origin`] | The public origin a request reached the server at |

pub mod client;
pub mod http;
pub mod jsonrpc;
pub mod local_store;
pub mod origin;
pub mod protocol;
pub mod sealed_store;
pub mod server;
pub mod session;

pub use client::Client;
pub use local_store::LocalStore;
pub use sealed_store::SealedStore;
pub use server::{Call, Identity, Server};
pub use session::{stateless_key, Session, Store};
