//! Infinite-conversation summary tree — the algorithm-layer surface.
//!
//! Per `docs/infinite_conversations.md`, the conversation timeline is
//! organised as a self-balancing binary tree of summary nodes, with
//! Normal turns hanging off the binary leaves as content sub-leaves.
//! This module contains the **pure data-structure** counterpart of that
//! design — no substrate, no scheduler, no model.  Every algorithm here
//! is testable in isolation with hand-crafted node sets.
//!
//! # Three node kinds
//!
//! ```text
//!   SummaryOfSummaries  (internal)        exactly 2 summary children
//!         │
//!   SummaryOfTurns      (binary leaf)     N Normal-turn children
//!         │
//!   Normal              (content sub-leaf) no children
//! ```
//!
//! The **AVL balance invariant** applies only to the binary structure
//! formed by `SummaryOfSummaries` → `SummaryOfTurns`.  Normal turns are
//! content; their count per `SummaryOfTurns` parent is variable and
//! they do not participate in the balance.
//!
//! For score-density selection (`select_dense`), however, **every node
//! is scoreable** — the provenance scan stamps a score on Normal turns,
//! `SummaryOfTurns` leaves, and `SummaryOfSummaries` internals alike.

mod diagnostics;
pub mod fixture;
pub mod probe;
mod recency;
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
pub use tree::{carry_triple, Node, NodeId, SummaryTree, TurnKind, MERGE_FANOUT};
