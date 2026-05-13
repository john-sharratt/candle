//! [`ConversationTreeConfig`] — clone-able policy stored on the tree.

use candle::DType;
use candle_nn::kv_cache::KvFormat;

use super::types::StorageTier;

/// Clone-able policy configuration stored by value on [`ConversationTree`](super::conversation_tree::ConversationTree).
///
/// Forks receive a full copy of the config that was active at fork time.
#[derive(Debug, Clone)]
pub struct ConversationTreeConfig {
    /// When true, `[T-{day}.{seq}]` markers are injected into each turn's
    /// assistant text and a postfix is appended to the system prompt.
    /// Default: `false`.
    pub temporal_markers_enabled: bool,

    /// Turn-pairs between summarization steps.
    ///
    /// `0` disables count-based summarization entirely. When non-zero, a
    /// summarization trigger fires once `turns_since_last_summarize()`
    /// reaches this value. Default: `8`.
    pub summarize_every: u32,

    /// Top-level segments between higher-level (recursive) summarization steps.
    ///
    /// `0` disables segment-level summarization entirely (default). When
    /// non-zero, a segment-level trigger fires once the number of top-level
    /// `Segment` nodes reaches this value — those segments are themselves
    /// summarized into a parent segment, which becomes their structural parent.
    /// The same threshold applies at every depth, producing unlimited
    /// hierarchical compression.
    pub segment_summarize_every: u32,

    /// Fire summarization when the UTC calendar day changes between consecutive
    /// turns. The check compares the `day` of the new turn with
    /// `last_turn_day` on the tree. Default: `true`.
    pub summarize_on_day_boundary: bool,

    /// Minimum children per parent node after N-ary rebalancing.
    /// TODO: structural slot present; not yet enforced (every node currently
    /// has at most one child). Default: `4`.
    pub min_children_per_node: u32,

    /// Maximum children per parent node.
    /// TODO: enforced during segment insertion once N-ary rebalancing is
    /// implemented. Field is stored but not yet checked. Default: `8`.
    pub max_children_per_node: u32,

    /// [`KvFormat`] for turn nodes at [`StorageTier::Hot`] (VRAM-resident,
    /// immediate attention access). Also used for the system prompt KV.
    /// Default: `KvFormat::Float(DType::BF16)`.
    ///
    /// BF16 is chosen over Q8_0 because the system prompt KV is attended on
    /// every token of every turn; Q8_0 reconstruction error compounds over
    /// thousands of turns on those most-referenced vectors.
    pub hot_turn_format: KvFormat,

    /// [`KvFormat`] for turn nodes at [`StorageTier::Warm`] (VRAM staging
    /// during Cold→Warm regeneration). Default: `KvFormat::Float(DType::BF16)`.
    pub warm_turn_format: KvFormat,

    /// Number of parallel dream simulations generated per Sleep cycle.
    ///
    /// Dreams are short (100–200 tokens each) and run simultaneously using the
    /// engine's batched decode capability. After generation each dream is
    /// scored against the character's core identity nodes; only those that
    /// resonate are inserted into the tree as `Sleep` turns. Default: `32`.
    pub sleep_batch_size: u32,

    /// Attention-score threshold above which a cold node triggers a Daydream.
    ///
    /// After each Reality turn completes, the last user message is probed
    /// against cold node representative K vectors. If any node scores above
    /// this threshold the character's mind has snagged on something, and a
    /// daydream generation is started in the background. Default: `0.7`.
    pub daydream_resonance_threshold: f32,

    /// System prompt for summarization inference.
    ///
    /// Injected into the temporary scheduler slot used to compress a window of
    /// turns into a [`ConversationSegment`](super::node::ConversationSegment).
    /// Override per-character as needed. Default: `prompts/summarize.txt`.
    pub summarization_system_prompt: String,

    /// Maximum tokens the model may generate for a summary. Default: `256`.
    pub summarization_max_tokens: u32,

    /// System prompt for Daydream inference.
    ///
    /// Used when attention on a cold node crosses `daydream_resonance_threshold`
    /// and a short associative thought is generated in the background.
    /// Override per-character as needed. Default: `prompts/daydream.txt`.
    pub daydream_system_prompt: String,

    /// System prompt for Sleep inference.
    ///
    /// Used during the end-of-day prospective sleep batch. Each dream in the
    /// batch of `sleep_batch_size` is generated from a memory seed using this
    /// prompt. Override per-character as needed. Default: `prompts/sleep.txt`.
    pub sleep_system_prompt: String,

    /// System prompt for Reason inference.
    ///
    /// Used when the executive self-dialogue turn runs to produce an updated
    /// plan that is then injected into Reality system prompts.
    /// Override per-character as needed. Default: `prompts/reason.txt`.
    pub reason_system_prompt: String,

    /// Maximum number of turns stored in this conversation tree before
    /// it begins culling the oldest versions. This is used by conversations
    /// that only need a short history to function properly
    pub max_turns: Option<usize>,
}

impl Default for ConversationTreeConfig {
    fn default() -> Self {
        Self {
            temporal_markers_enabled: false,
            summarize_every: 8,
            segment_summarize_every: 0,
            summarize_on_day_boundary: true,
            min_children_per_node: 4,
            max_children_per_node: 8,
            hot_turn_format: KvFormat::Float(DType::BF16),
            warm_turn_format: KvFormat::Float(DType::BF16),
            sleep_batch_size: 32,
            daydream_resonance_threshold: 0.7,
            summarization_system_prompt: crate::prompts::SUMMARIZE_PROMPT.to_string(),
            summarization_max_tokens: 256,
            daydream_system_prompt: crate::prompts::DAYDREAM_PROMPT.to_string(),
            sleep_system_prompt: crate::prompts::SLEEP_PROMPT.to_string(),
            reason_system_prompt: crate::prompts::REASON_PROMPT.to_string(),
            max_turns: None,
        }
    }
}

impl ConversationTreeConfig {
    /// Resolve the [`KvFormat`] for a storage tier.
    ///
    /// Returns `None` for [`StorageTier::Cold`] — Cold nodes are disk-only;
    /// their KV is regenerated on demand by replaying the node's
    /// `decode_context` recipe through a prefill pass. Currently all nodes
    /// stay `Hot`, so this always returns `Some`.
    pub fn storage_tier_format(&self, tier: StorageTier) -> Option<KvFormat> {
        match tier {
            StorageTier::Hot => Some(self.hot_turn_format),
            StorageTier::Warm => Some(self.warm_turn_format),
            StorageTier::Cold => None,
        }
    }
}
