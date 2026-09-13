//! Immutable summary forest — the algorithm-layer surface.
//!
//! # ⚠ This subsystem is DISCONNECTED
//!
//! Nothing in the engine drives it. `Engine::new` builds
//! [`SummariserThread::disabled`] unconditionally, and every timeline is created
//! with `summarize = false`, so no turn is ever enqueued and no summary node is
//! ever built. The code below compiles, is unit-tested, and is kept deliberately
//! — but a reader tracing "where do summaries come from?" in a live conversation
//! will find the answer is *nowhere*.
//!
//! **Why it was switched off.** Compression was producing bad memory rather than
//! a saving. Measured on a 16-turn conversation, 5 of 9 summary nodes were
//! unfaithful: two echoed the user's question back, one echoed the compressor's
//! own instruction verbatim, and the merge node standing for the WHOLE
//! conversation read *"I am an AI assistant."* These nodes are written in the
//! first person, as if they were the reply, and a later projection reads them as
//! history — so a wrong one is a **false memory** the model cannot distinguish
//! from something it actually said, not merely a missing summary. Retrieval
//! quality is being pursued through provenance selection, which ranks real turns
//! instead of manufacturing new text.
//!
//! **What re-enabling takes.** The gate is one field: `Timeline::summarize`, read
//! by `push_pending_summary` and both `append_*` seal paths. Deciding which
//! timelines set it, and spawning the thread in `Engine::new`, is the whole
//! wiring — the substrate side is intact. The tests here opt their own timeline
//! in explicitly, so they exercise the machinery as it would run.
//!
//! **Consequence while it is off:** a conversation keeps every turn whole. There
//! is no compression tier between "raw turn" and "projected context", so context
//! growth is bounded only by projection and selection.
//!
//! Per `docs/archived/immutable_summary_forest.md`, each timeline's summary structure
//! is an append-only **Merkle Mountain Range**: a node's parent is fixed by
//! arrival order and position, never by rebalancing, so once a node exists its
//! children — and therefore its content and Q-fingerprint — never change.
//! This module contains the **pure data-structure** counterpart of that
//! design — no substrate, no scheduler, no model.  Every algorithm here
//! is testable in isolation with hand-crafted node sets.
//!
//! # Three node kinds
//!
//! ```text
//!   SummaryOfSummaries  (internal)        exactly MERGE_FANOUT summary
//!         │                               children, all of the same level
//!   SummaryOfTurns      (leaf)            exactly one exchange
//!         │
//!   Normal              (content sub-leaf) no children
//! ```
//!
//! Leaves are appended on the right in chronological order; whenever the last
//! `MERGE_FANOUT` peaks share a level they carry up into one `SummaryOfSummaries`
//! a level higher (the base-`MERGE_FANOUT` carry), so the peak count is the
//! base-`MERGE_FANOUT` digit sum of the leaf count.  There is no balancing and no
//! `dirty` bit: the canonical shape is a pure function of the leaf sequence, so a
//! shape that doesn't match is simply rebuilt on load.
//!
//! For score-density selection (`select_dense`), **every node is scoreable** —
//! the provenance scan stamps a score on Normal turns, `SummaryOfTurns` leaves,
//! and `SummaryOfSummaries` internals alike — and the recency anchor
//! ([`recency_score`]) keeps the newest Normal turns in the window verbatim.

mod diagnostics;
pub mod exchange;
pub mod fixture;
pub mod probe;
mod recency;
pub mod scope;
mod select;
mod structural;
pub mod summariser;
mod tree;

pub use diagnostics::{SelectionDiagnostics, SelectionOrigin};
pub use fixture::{
    ExpectedInvariants, FixtureError, FixtureManifest, PlantSpec, ProbeSpec, SubstrateFixture,
};
pub use probe::{ProbeError, ProbeRequest, ProbeResponse, ProbeRunner};
pub use recency::{recency_score, RecencyConfig};
pub use select::{select_dense, Selection};
pub use structural::{leaf_skeleton, structural_rollup, StructuralRollup};
pub use summariser::{ChannelProbeRunner, MockProbeRunner, SummariserThread, SummariserTrigger};
pub use tree::{carry_run, Node, NodeId, SummaryTree, TurnKind, MERGE_FANOUT};
