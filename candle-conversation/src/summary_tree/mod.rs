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
//! Selection over the tree (`select_budget_fit`) treats every node as a
//! candidate: the frontier starts at the `SummaryOfSummaries` peaks and
//! refines down through `SummaryOfTurns` leaves to the raw Normal turns.

mod diagnostics;
pub mod fixture;
pub mod probe;
mod select;
mod structural;
pub mod summariser;
mod tree;

pub use diagnostics::{SelectionDiagnostics, SelectionOrigin};
pub use fixture::{
    ExpectedInvariants, FixtureError, FixtureManifest, PlantSpec, ProbeSpec, SubstrateFixture,
};
pub use probe::{ProbeError, ProbeRequest, ProbeResponse, ProbeRunner};
pub use select::{select_budget_fit, Selection};
pub use structural::{leaf_skeleton, structural_rollup, StructuralRollup};
pub use summariser::{ChannelProbeRunner, MockProbeRunner, SummariserThread, SummariserTrigger};
pub use tree::{carry_run, Node, NodeId, SummaryTree, TurnKind, MERGE_FANOUT};
