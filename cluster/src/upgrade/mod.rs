//! A member's side of A/B upgrades (`docs/replicated_substrate.md` §3.8–§3.11):
//! what a build offers, fetching it, whether it can open this member's
//! substrate, the in-place swap, the builds kept for rollback, and `keep`'s
//! restore of a build that does not become ready.
//!
//! | Module | |
//! |---|---|
//! | [`artifacts`] | the sibling binaries reporting the same build — each with its UI compiled in |
//! | [`parts`] | downloads as resumable part files, verified by SHA-256 |
//! | [`compat`] | whether a build can open a substrate: log format, record types, model |
//! | [`decisions`] | verdicts, forced rollouts and pauses as Paxos decisions |
//! | [`kept`] | `builds/<count>-<hash>/` and pruning |
//! | [`swap`] | the in-place swap and its restore |
//! | [`handoff`] | the old process's last act: swap, write the plan, start the new build |
//! | [`watchdog`] | `recover`: put the previous executable back for `keep` to start |
//! | [`os`] | process liveness, and the successor's wait for the predecessor |
//! | [`sum`] | SHA-256 and its hex spelling |
//! | [`runtime`] | a process's upgrades: announcing, fetching, certifying, installing |
//!
//! Here rather than in the host, because an agent upgrades the same way and
//! links no host.

pub mod artifacts;
pub mod compat;
pub mod decisions;
pub mod handoff;
pub mod kept;
pub mod os;
pub mod parts;
pub mod runtime;
pub mod sum;
pub mod swap;
pub mod watchdog;
