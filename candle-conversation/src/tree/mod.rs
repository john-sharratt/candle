//! Attention-Organized Sequence Tree.
//!
//! This module implements the `ConversationTree` as described in the
//! *Attention-Organized Sequence Trees* paper. The goal is a single
//! canonical history structure that replaces the flat `Vec<Turn>` and
//! carries the full type system for all future work without requiring
//! struct redefinitions.
//!
//! # Cognitive architecture
//!
//! The character's mind has four layers, from deepest to most visible:
//!
//! ```text
//! Beliefs  — static declarative propositions; injected into every system
//!            prompt regardless of turn type; never change
//!            ↓
//! Limbic   — emotional and associative substrate
//!   Sleep    end-of-day batch prospective simulation (20–50 parallel dreams)
//!   Thought  associative flicker after Reality turns, gated by idle window
//!            ↓
//! Frontal  — executive planning through self-dialogue
//!   Reason   produces the Plan; injected into Reality system prompts
//!            ↓
//! Reality  — live user↔assistant conversation; acts from the Plan
//! ```
//!
//! All four turn types are first-class `ConversationNode`s in the tree.
//! `beliefs` and `plan` are **not** nodes — they are static/mutable text
//! fields on `ConversationTree` injected into system prompts by the engine.
//!
//! # Current scope
//!
//! All operations are **synchronous**; the tree is **intentionally
//! unbalanced** (right-growing chain, no rotation). This gives a stable,
//! testable base before the full paper architecture is layered on:
//!
//! - ✅ `ConversationTree` as canonical history (replaces `Vec<Turn>`)
//! - ✅ System prompt pinned as a tree-level field (not a node), prefilled in BF16
//! - ✅ `beliefs` field — static propositions injected into all system prompts
//! - ✅ `plan` field — Reason-produced intention injected into Reality prompts
//! - ✅ Temporal marker injection `[T-{day}.{seq}]` with per-tree opt-in
//! - ✅ Summarization trigger after N turns or on day boundary (stub — logs only)
//! - ✅ `ConversationTreeFork` / `TreePatch` for background work (infrastructure ready)
//! - ✅ N-ary field layout (`children`, `decode_context`) baked in from day 1
//! - ✅ HOT/WARM/COLD tier enum defined; all nodes currently stay HOT
//! - ❌ Tree rebalancing and segment node creation (TODO)
//! - ❌ Tier promotion/demotion and VRAM eviction (TODO)
//! - ❌ Actual summarization inference (TODO: replace `run_summarize()` stub)
//! - ❌ Sleep pipeline — batch prospective simulation (TODO)
//! - ❌ Daydream pipeline — resonance probe + latency gate (TODO)
//! - ❌ Reason pipeline — self-dialogue + Plan update (TODO)
//!
//! # Tree shape
//!
//! ```text
//! ConversationTree {
//!   system_prompt: "You are..."   ← BF16 KV, tree-level field
//!   beliefs: "Trust must be..."  ← static, injected into ALL prompts
//!   plan: "I need to..."         ← mutable, injected into Reality prompts
//! }
//!          │
//!       nodes[0]: Turn [T-0.1]   ← Reality turn
//!                      │
//!                   nodes[1]: Turn [T-0.2]
//!                                  │
//!                               nodes[2]: Turn [T-1.3]   ← day boundary
//!                                              │
//!                                           nodes[3]: ...
//! ```
//!
//! The tree will eventually interleave `ConversationNode::Segment` nodes
//! between turns when summarization produces a compressed section.
//!
//! # N-ary tree infrastructure
//!
//! Every [`ConversationNode`] carries `children: Vec<ConversationNode>` and
//! `decode_context: Vec<ConversationNode>`. Trees are currently right-growing
//! chains (at most one child per node), but the field layout is correct for
//! multi-child N-ary rebalancing from day one — branching logic slots in
//! without changing any struct definitions.
//!
//! # COW forking
//!
//! Every node is cheap-clonable via the `Arc` newtype wrappers
//! ([`ConversationTurn`], [`ConversationSegment`]). Cloning
//! a [`ConversationTree`] bumps Arc ref-counts without copying node data.
//! [`ConversationTree::fork`] produces a [`ConversationTreeFork`] that is
//! `Send`; background workers send back a [`TreePatch`] which the main
//! engine applies on its next idle cycle via
//! [`ConversationTree::apply_patch`].
//!
//! # Key types
//!
//! | Type | Location | Purpose |
//! |---|---|---|
//! | [`TurnType`] | `types` | Cognitive layer: Reality / Sleep (limbic) / Thought (limbic) / Reason (frontal) |
//! | [`StorageTier`] | `types` | KV location: Hot / Warm / Cold |
//! | [`TurnId`] | `types` | `(day, seq)` coordinates — `Copy`, derives `[T-day.seq]` marker |
//! | [`SegmentId`] | `types` | `(start_turn, end_turn)` range — `Copy` |
//! | [`NodeId`] | `types` | Enum of `Turn(TurnId)` or `Segment(SegmentId)` — `Copy` |
//! | [`ConversationTree`] | `conversation_tree` | Root owner: system prompt + beliefs + plan + node vec + config + clock |
//! | [`ConversationNode`] | `node` | Enum of `Turn(ConversationTurn)` or `Segment(ConversationSegment)` |
//! | [`ConversationTreeConfig`] | `config` | Clone-able policy: markers, summarization, sleep batch size, daydream threshold, KV format per tier |
//! | [`TokenizedText`] | `token_text` | Text + lazily-computed token ids; used in all node and prompt types |
//! | [`TreePatch`] | `patch` | Delta from a background fork, applied via `apply_patch()` |
//! | [`TEMPORAL_MARKER_POSTFIX`] | `prompts` | Fixed system prompt postfix appended when markers are enabled |

mod config;
mod conversation_tree;
mod node;
mod patch;
mod summarize;
mod task;
mod types;
pub mod token_text;

#[cfg(test)]
mod tests;

// ── Public re-exports ──────────────────────────────────────────────────────

pub use config::ConversationTreeConfig;

pub use conversation_tree::{ConversationTree, ConversationTreeFork};
pub use crate::prompts::TEMPORAL_MARKER_POSTFIX;

pub use node::{
    ConversationNode, ConversationSegment, ConversationSegmentInner, ConversationSystemPrompt,
    ConversationTurn, ConversationTurnInner,
};

pub use patch::{TreeMetadataDelta, TreePatch};

pub use token_text::TokenizedText;

pub use types::{NodeId, SegmentId, StorageTier, TurnId, TurnType};

// FixedTimeSource and TimeSource are re-exported here so callers can write
// `use candle_conversation::tree::{FixedTimeSource, TimeSource}`.
pub use crate::time_source::{FixedTimeSource, TimeSource};

// Cognitive task types — used by Sequence to hold and poll in-flight
// background inference work drained from the tree after each turn.
pub(crate) use task::{CognitiveTask, TaskPoll};
