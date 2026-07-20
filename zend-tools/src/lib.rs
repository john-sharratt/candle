//! Tool registry and execution engine for the Zen Code daemon (`zend`).
//!
//! This crate implements all 93 server-side tools described in `docs/tool-system.md`.
//! Tools are stateless Rust functions — all shared state lives in [`ToolContext`].
//! The orchestrator calls [`run`] with a tool name and JSON arguments and always
//! gets back a JSON value that the LLM can act on, whether the call succeeded or
//! failed.
//!
//! # Module structure
//!
//! - [`tool`] — the [`Tool`] trait every tool implements, plus [`ToolError`],
//!   [`ConfirmationDetails`], and the subagent runner interface
//! - [`registry`] — the static [`RegisteredTool`] table and name-based lookup
//! - [`runner`] — [`run`] and [`confirmation`], the two dispatch entry points used
//!   by the orchestrator
//! - [`context`] — [`ToolContext`], the `Arc`-shared bundle of state stores passed
//!   into every tool invocation
//! - [`state`] — the individual stores: [`state::VfsStore`], [`state::CredentialStore`],
//!   [`state::NotesStore`], [`state::SessionRegistry`], [`state::HashStateStore`]
//! - [`tools`] — all 93 tool implementations, one module per tool group
//!
//! # Authoring a tool
//!
//! ```ignore
//! use schemars::JsonSchema;
//! use serde::{Deserialize, Serialize};
//! use thiserror::Error;
//! use validator::Validate;
//! use zend_tools::{ConfirmationDetails, RegisteredTool, Tool, ToolContext, ToolError};
//!
//! #[derive(Deserialize, JsonSchema, Validate)]
//! pub struct Request {
//!     #[validate(length(min = 1))]
//!     pub query: String,
//! }
//!
//! #[derive(Serialize)]
//! pub struct Response { pub results: Vec<String> }
//!
//! #[derive(Debug, Error)]
//! pub enum Error {
//!     #[error("provider unavailable: {0}")]
//!     ProviderUnavailable(String),
//! }
//!
//! impl ToolError for Error {
//!     fn code(&self) -> &'static str {
//!         match self {
//!             Error::ProviderUnavailable(_) => "provider_unavailable",
//!         }
//!     }
//! }
//!
//! pub struct MyTool;
//! impl Tool for MyTool {
//!     const NAME: &'static str = "my_tool";
//!     const DESCRIPTION: &'static str = "...";
//!     type Request = Request;
//!     type Response = Response;
//!     type Error = Error;
//!     fn run(_ctx: &ToolContext, req: Request) -> Result<Response, Error> {
//!         Ok(Response { results: vec![req.query] })
//!     }
//! }
//!
//! pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<MyTool>();
//! ```

pub mod context;
mod numfmt;
pub mod registry;
pub mod runner;
pub mod state;
pub mod tool;
pub mod tools;

pub use context::ToolContext;
pub use registry::RegisteredTool;
pub use runner::{confirmation, run};
pub use tool::{
    ConfirmationDetails, SubagentRequest, SubagentResponse, SubagentRunner, Tool, ToolError,
};
