//! zend's MCP server, `/v1/mcp` (`docs/claude_zend_gateway.md`,
//! `docs/mcp_app.md` §3).
//!
//! Claude — in Claude Code or an ordinary claude.ai conversation — reaches zend
//! through it, by way of the aggregator at `code.tokera.com/mcp`: Claude
//! orchestrates and reviews, zend does the volume. Each task runs on a native
//! zend conversation (a worker) under a thread, the feature's place, and its
//! result is the worker's journal entry. Signing the caller in is the
//! gateway's (`web::auth::mcp`); the identity it states is the caller here.
//! The protocol is `mcp_server`'s.
//!
//! | Module | What |
//! |---|---|
//! | [`server`] | zend as an MCP server: its card, its caller, its sessions, its tools |
//! | [`tools`] | The tools, one file each |
//! | [`workers`] | Running a task, answering a worker, and waiting on workers |
//! | [`audit`] | A finished task's citation audit: each cited URL opened, seen, or never seen |
//! | [`archiving`] | Archiving threads and workers: by `thread_archive`, and after a day idle |
//! | [`activity`] | What a running worker is doing, counted from its stream and said in one sentence |
//! | [`reported`] | The journal entries already handed to Claude, so a later result gives only new ones |
//! | [`parts`] | `sub_run`: a worker's parts, started under it and waited on |
//! | [`lineage`] | The journals a worker or part inherits from above it |
//! | [`ops`] | What the operations tools share: Admins only, the decision gate, the way up |
//! | [`nudges`] | Guidance for a running worker, taken at its next tool round |
//! | [`stops`] | Claude stopping a running worker, at its next tool round |
//! | [`threads`] | Threads, workers and the thread's journal, as records |
//! | [`memory`] | The memory backbone: the user's and their groups' journals, and the context made of them |
//! | [`names`] | Names and paths |
//! | [`caller`] | Who is calling, and the key their records are kept under |
//! | [`ownership`] | Who may open a worker's conversation on zend's other routes |
//! | [`records`] | The small records kept in the substrate |
//! | [`state`] | Sessions as records, the attached thread, the change signal |
//! | [`tending`] | The sessions' noted use written in, and the stale swept, off the request path |

pub mod activity;
pub mod archiving;
pub mod audit;
pub mod caller;
pub mod lineage;
pub mod memory;
pub mod names;
pub mod nudges;
pub mod ops;
pub mod ownership;
pub mod parts;
pub mod records;
pub mod reported;
pub mod resuming;
pub mod server;
pub mod state;
pub mod stops;
pub mod tending;
pub mod threads;

#[cfg(test)]
mod latency_tests;
pub mod tools;
pub mod workers;
