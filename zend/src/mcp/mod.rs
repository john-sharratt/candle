//! The remote MCP endpoint, `/mcp` (`docs/claude_zend_gateway.md`).
//!
//! Claude — in Claude Code or an ordinary claude.ai conversation — reaches zend
//! through it: Claude orchestrates and reviews, zend does the volume. Each task
//! runs on a native zend conversation (a worker) under a thread, the
//! feature's place, and its result is the worker's journal entry. Signing the
//! caller in is the gateway's (`web::auth::mcp`); the identity it states is the
//! caller here.
//!
//! | Module | What |
//! |---|---|
//! | [`http`] | The route: `/mcp`, and the challenge that starts a sign-in |
//! | [`protocol`] | The MCP methods, in the 2025-11-25 and 2026-07-28 eras |
//! | [`jsonrpc`] | One JSON-RPC message per POST |
//! | [`tools`] | The tools, one file each |
//! | [`workers`] | Running a task, answering a worker, and waiting on workers |
//! | [`activity`] | What a running worker is doing, counted from its stream and said in one sentence |
//! | [`reported`] | The journal entries already handed to Claude, so a later result gives only new ones |
//! | [`parts`] | `sub_run`: a worker's parts, started under it and waited on |
//! | [`lineage`] | The journals a worker or part inherits from above it |
//! | [`ops`] | What the operations tools share: Admins only, the decision gate, the way up |
//! | [`nudges`] | Guidance for a running worker, taken at its next tool round |
//! | [`threads`] | Threads, workers and the thread's journal, as records |
//! | [`names`] | Names and paths |
//! | [`caller`] | Who is calling, and the key their records are kept under |
//! | [`ownership`] | Who may open a worker's conversation on zend's other routes |
//! | [`records`] | The small records kept in the substrate |
//! | [`state`] | What is held in memory: sessions, running tasks |
//! | [`origin`] | The public origin a request reached `/mcp` at |

pub mod activity;
pub mod caller;
pub mod http;
pub mod jsonrpc;
pub mod lineage;
pub mod names;
pub mod nudges;
pub mod ops;
pub mod origin;
pub mod ownership;
pub mod parts;
pub mod protocol;
pub mod records;
pub mod reported;
pub mod state;
pub mod threads;
pub mod tools;
pub mod workers;
