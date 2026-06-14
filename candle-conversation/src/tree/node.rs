//! Node types: [`ConversationTurn`], [`ConversationSegment`],
//! [`ConversationNode`], [`ConversationSystemPrompt`].
//!
//! # Node structure
//!
//! Every node type stores an `Arc<…Inner>` so that [`ConversationNode`]
//! clones are O(1) ref-count bumps. `decode_context` and `children` hold
//! those cheap clones, giving each node full access to its ancestors and
//! descendants without allocation.
//!
//! # System prompt
//!
//! [`ConversationSystemPrompt`] is *not* a node. It lives as a top-level
//! field on [`ConversationTree`](super::conversation_tree::ConversationTree)
//! and is always the implicit first element of every prefill — but it has no
//! [`NodeId`], no children, and is never rotated or summarized.
//! Its KV is tracked separately in `KvCacheLayer::system_prompt_entry` and
//! stored in BF16 (see `ConversationSystemPrompt` docs for the rationale).

use std::sync::Arc;

use super::token_text::TokenizedText;
use super::types::{NodeId, SegmentId, TurnId, TurnType};

// ────────────────────────────────────────────────────────────────────────────
// ConversationTurnInner / ConversationTurn
// ────────────────────────────────────────────────────────────────────────────

/// Inner data for a turn node — an actual user↔assistant exchange.
///
/// Stored behind an `Arc` so the newtype [`ConversationTurn`] is cheap to
/// clone (O(1) ref-count bump).
#[derive(Debug)]
pub struct ConversationTurnInner {
    /// Globally stable identity of this turn (day + seq coordinates).
    pub turn_id: TurnId,

    /// The cognitive mode that produced this turn. Defaults to
    /// [`TurnType::Reality`] for all turns from the live request/response loop.
    pub turn_type: TurnType,

    /// The user's message in this exchange, with pre-tokenized ids.
    pub user: TokenizedText,

    /// The assistant's response (without markers or thinking blocks), with
    /// pre-tokenized ids.
    pub assistant: TokenizedText,

    /// KV recipe: the exact ordered list of ancestor nodes whose KV was
    /// present in the prefill context when this turn was originally decoded.
    ///
    /// - `decode_context[0]` = leftmost node in the KV prefix.
    /// - Last element = immediately-preceding node.
    /// - The system prompt is the implicit first prefix and is **not**
    ///   included (it is always present as a tree-level field).
    /// - Empty for the first turn in the tree.
    ///
    /// This is the exact recipe for Cold→Warm KV regeneration: replay a
    /// prefill pass with these nodes (in order) plus the system prompt, and
    /// the KV cache will be identical to the original decode.
    /// TODO: used to warm Cold nodes on demand once tier management is implemented.
    pub decode_context: Vec<ConversationNode>,

    /// Child links. Currently at most one child (right-growing chain).
    /// TODO: N-ary rebalancing will allow multiple children per node;
    /// `ConversationTreeConfig::max_children_per_node` caps the fan-out after
    /// segment insertion.
    pub children: Vec<ConversationNode>,
}

/// A turn node — a cheap-clonable newtype wrapping `Arc<ConversationTurnInner>`.
///
/// Represents an actual user↔assistant exchange (turn-pair) in the tree.
#[derive(Debug, Clone)]
pub struct ConversationTurn(pub Arc<ConversationTurnInner>);

impl ConversationTurn {
    /// Create a new turn. `decode_context` and `children` start empty.
    pub fn new(
        turn_id: TurnId,
        turn_type: TurnType,
        user: impl Into<TokenizedText>,
        assistant: impl Into<TokenizedText>,
    ) -> Self {
        ConversationTurn(Arc::new(ConversationTurnInner {
            turn_id,
            turn_type,
            user: user.into(),
            assistant: assistant.into(),
            decode_context: Vec::new(),
            children: Vec::new(),
        }))
    }

    /// The node identity of this turn.
    pub fn node_id(&self) -> NodeId {
        NodeId::Turn(self.0.turn_id)
    }

    /// Borrow the inner data.
    pub fn inner(&self) -> &ConversationTurnInner {
        &self.0
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ConversationSegmentInner / ConversationSegment
// ────────────────────────────────────────────────────────────────────────────

/// Inner data for a segment node — a system-created summary of prior turns.
#[derive(Debug)]
pub struct ConversationSegmentInner {
    /// Globally stable identity: the inclusive range of turns summarized.
    /// Set at creation, never mutated.
    pub segment_id: SegmentId,

    /// The compressed summary text (the distillation of the summarized turns),
    /// with pre-tokenized ids.
    pub summary_text: TokenizedText,

    /// Ordered list of ancestor nodes whose KV was present when this segment
    /// was materialized. Empty initially; computed on integration.
    pub decode_context: Vec<ConversationNode>,

    /// Child links. Currently at most one child. TODO: N-ary.
    pub children: Vec<ConversationNode>,
}

/// A segment node — a cheap-clonable newtype wrapping
/// `Arc<ConversationSegmentInner>`.
///
/// Represents a system-created summary of a contiguous range of prior turns.
#[derive(Debug, Clone)]
pub struct ConversationSegment(pub Arc<ConversationSegmentInner>);

impl ConversationSegment {
    /// Create a new segment. `decode_context` and `children` start empty.
    pub fn new(segment_id: SegmentId, summary_text: impl Into<TokenizedText>) -> Self {
        ConversationSegment(Arc::new(ConversationSegmentInner {
            segment_id,
            summary_text: summary_text.into(),
            decode_context: Vec::new(),
            children: Vec::new(),
        }))
    }

    /// Attach children to this segment (the turns it summarises).
    ///
    /// Consumes `self` and returns a new `ConversationSegment` with the given
    /// children set. Panics if the `Arc` has more than one strong reference,
    /// which would indicate an unexpected shared alias at this early stage of
    /// the segment's life.
    pub fn with_children(self, children: Vec<ConversationNode>) -> Self {
        let inner =
            Arc::try_unwrap(self.0).expect("with_children called on a shared ConversationSegment");
        ConversationSegment(Arc::new(ConversationSegmentInner {
            segment_id: inner.segment_id,
            summary_text: inner.summary_text,
            decode_context: inner.decode_context,
            children,
        }))
    }

    /// The node identity of this segment.
    pub fn node_id(&self) -> NodeId {
        NodeId::Segment(self.0.segment_id)
    }

    /// Borrow the inner data.
    pub fn inner(&self) -> &ConversationSegmentInner {
        &self.0
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ConversationNode
// ────────────────────────────────────────────────────────────────────────────

/// A node in the conversation tree: either an exchange or a summary.
///
/// The enum wraps strongly-typed cheap-clonable newtypes so cloning a
/// `ConversationNode` is O(1) (Arc ref-count bump).
#[derive(Debug, Clone)]
pub enum ConversationNode {
    /// An actual user↔assistant exchange (turn-pair).
    Turn(ConversationTurn),
    /// A compressed summary section covering a range of prior turns.
    Segment(ConversationSegment),
}

impl ConversationNode {
    /// The stable identity of this node.
    pub fn node_id(&self) -> NodeId {
        match self {
            ConversationNode::Turn(t) => t.node_id(),
            ConversationNode::Segment(s) => s.node_id(),
        }
    }

    /// Returns `Some(&ConversationTurn)` if this is a turn node.
    pub fn as_turn(&self) -> Option<&ConversationTurn> {
        match self {
            ConversationNode::Turn(t) => Some(t),
            ConversationNode::Segment(_) => None,
        }
    }

    /// Returns `Some(&ConversationSegment)` if this is a segment node.
    pub fn as_segment(&self) -> Option<&ConversationSegment> {
        match self {
            ConversationNode::Turn(_) => None,
            ConversationNode::Segment(s) => Some(s),
        }
    }

    /// Ordering seq — used for chronological ordering across mixed node types.
    pub fn ordering_seq(&self) -> u32 {
        self.node_id().ordering_seq()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ConversationSystemPrompt
// ────────────────────────────────────────────────────────────────────────────

/// The system prompt held at the tree level.
///
/// Has no [`NodeId`]; it is a tree-level concern, not a node. Always prefilled
/// first in every decode pass, never rotated, never summarized.
///
/// # Why a tree-level field, not a node?
///
/// The system prompt is present unconditionally in 100% of all prefills. It
/// has no preceding context, no turn identity, and never changes the
/// conversation structure. Promoting it to a node would contaminate every
/// `decode_context` list, complicate segment boundaries, and waste a `NodeId`
/// on a structural constant.
///
/// # BF16 KV rationale
///
/// The system prompt KV is the most-attended tensor across all turns. Q8_0
/// introduces approximately 1% reconstruction error per element; in the
/// most-referenced vectors (those attending back to the system prompt on every
/// single token) those errors compound across thousands of turns. BF16
/// eliminates quantization error there entirely.
///
/// The overhead is negligible: ~800 system-prompt tokens × 48 Llama-3-70B
/// layers × 64 heads × 64 head-dim × 2 bytes ≈ **1.5 MB** — a fixed cost
/// independent of conversation length.
///
/// The temporal marker postfix (if enabled) is already baked into `content`
/// at `ConversationTree::new()` time so the system-prompt KV is frozen at
/// that point.
#[derive(Debug, Clone)]
pub struct ConversationSystemPrompt {
    /// Prompt text (with any temporal marker postfix) and pre-tokenized ids.
    /// Token ids may be empty until `ConversationTree::set_system_prompt_tokens`
    /// has been called.
    pub content: TokenizedText,
}
