//! Immutable summary forest — the algorithm-layer surface.
//!
//! Per `docs/immutable_summary_forest.md`, each timeline's summary structure
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
