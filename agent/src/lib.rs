//! `tokera-agent` (`docs/replicated_substrate.md` §11): a CPU-only member of
//! the `agent` app. It holds a checkout of every repository it advertises and
//! the credentials to push to their origins, and serves the job API over the
//! streams its gateways open — a command run against a conversation's or an
//! operation's store, a push of commits a member built, or a fetch of what
//! origin holds for a member that cannot reach it.
//!
//! | Module | |
//! |---|---|
//! | [`config`] | the agent's file: its checkouts, its job slots, its jobs folder |
//! | [`checkouts`] | each checkout: its repository, sandbox and origin |
//! | [`wire`] | the job API's bodies |
//! | [`jobs`] | running jobs, pushes and fetches, and what each came to |
//! | [`routes`] | the job API's routes |
//! | [`member`] | joining the clusters as an `agent` member |
//! | [`client`] | a member's side: sending its jobs to an agent |
//! | [`pusher`] | a member's pushes, made by an agent |
//! | [`fetcher`] | a member's fetches, made by an agent |
//! | [`install`] | installing a new build of the agent (§3.9) |

pub mod checkouts;
pub mod client;
pub mod config;
pub mod fetcher;
pub mod install;
pub mod jobs;
pub mod member;
pub mod pusher;
pub mod routes;
pub mod wire;
