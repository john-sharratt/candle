//! Holds the settings for one conversation's GPU sequence: its window, its
//! re-projection cadence and its turn policy.
//!
//! The sequence is rebuilt each turn from the system prompt and a bounded number
//! of completed exchanges, so the memory it holds does not grow with the
//! conversation. The full history stays in the conversation tree. Continuous
//! re-projection reruns the provenance scan on a token cadence and at chosen
//! text boundaries, so attention can move as the model's intent does, and it is
//! off for append-only ingests where re-projecting every turn would be quadratic.
//! The hot-turn ceiling is derived from the arena's capacity and the size of a
//! turn. Per-conversation compression and K/V format overrides let a reference
//! layer be compressed harder than live dialogue, and a flag keeps a
//! conversation's turns in native precision for capturing exemplars.

use super::*;

/// Pick a reasonable `max_hot_turns` ceiling from arena dimensions and
/// expected turn length.
///
/// Derives how many completed turns worth of KV data fit in the paged
/// arena before Hot → Warm eviction should fire, using the formula from
/// the sealed-sequence-tiers design:
///
/// ```text
/// hot_turns = (arena_chunks × chunk_size) / tokens_per_turn
/// ```
///
/// Model geometry (n_layers, n_kv_heads, head_dim) cancels in the ratio —
/// the limit is determined entirely by arena capacity versus turn size.
///
/// # Arguments
///
/// * `arena_chunks`      — Number of chunk slots in the KV arena
///   ([`BatchedConfig::arena_chunks`]).
/// * `chunk_size`        — Tokens per chunk ([`BatchedConfig::chunk_size`]).
/// * `tokens_per_turn`   — Expected maximum tokens per completed turn
///   (use `SequenceConfig::max_response_tokens` as a conservative upper bound).
///
/// Returns the number of hot turns, clamped to `[1, 256]`.
/// Returns `0` when `tokens_per_turn` is zero (disables eviction).
pub fn pick_max_hot_turns(arena_chunks: usize, chunk_size: usize, tokens_per_turn: usize) -> usize {
    if tokens_per_turn == 0 {
        return 0;
    }
    ((arena_chunks * chunk_size) / tokens_per_turn).clamp(1, 256)
}

// Sequence Configuration

/// [`SequenceConfig::reproject_every_n_tokens`]'s default.
pub const REPROJECT_EVERY_N_TOKENS: usize = 64;

/// [`SequenceConfig::reproject_max_probe_tokens`]'s default.
pub const REPROJECT_MAX_PROBE_TOKENS: usize = 256;

/// [`SequenceConfig::reproject_trigger_texts`]'s default.
pub const REPROJECT_TRIGGER_TEXTS: [&str; 1] = ["\n"];

/// Per-conversation configuration.
#[derive(Debug, Clone)]
pub struct SequenceConfig {
    /// Dialect of the chat
    pub dialect: Dialect,

    /// Maximum tokens to generate per turn.
    pub max_response_tokens: usize,

    /// Default sampling configuration.
    pub sampling: SamplingConfig,

    /// Sequence tree policy: summarization cadence, KV tier formats,
    /// cognitive task prompts, etc.
    pub tree: ConversationTreeConfig,

    /// Number of most-recent turns to include in the context window sent to
    /// the GPU for each new turn.
    ///
    /// When non-zero the conversation rebuilds its KV sequence from scratch on
    /// every new turn by prefilling the system prompt plus the last
    /// `context_window_turns` completed exchanges (from the conversation tree)
    /// together with the new user message.  This bounds memory growth to a
    /// fixed context window regardless of conversation length.
    ///
    /// The conversation tree continues to accumulate the full history (including
    /// summarization). Only the GPU sequence is windowed.
    ///
    /// `0` = disabled — the sequence grows continuously (legacy behaviour).
    /// Default: `8`.
    pub context_window_turns: usize,

    /// Reserved for future Hot → Warm eviction policy.  Currently
    /// unused — substrate-tier eviction is not wired through to the
    /// workspace substrate yet.  Kept as a config field so callers
    /// can express their intent without churning the API.
    ///
    /// `0` = disabled (default).
    pub max_hot_turns: usize,

    /// Continuous re-projection cadence in decoded tokens.
    ///
    /// When `> 0`, the scheduler fires a provenance scan + re-projection + view
    /// swap each time the decoded-token count for an active turn becomes a
    /// multiple of this value.  The view's borrowed-block ranges are
    /// recomputed from scratch using the live Q-vector signatures of the
    /// turn-so-far as the provenance probe — letting attention "wander" through
    /// the substrate as the model's intent evolves mid-decode.
    ///
    /// Should be `>= CHUNK_SIZE` (32) so at least one freshly-sealed
    /// chunk's worth of Q-sigs is available between fires.
    ///
    /// **Default: `64`** — re-project every two sealed chunks.  Set to
    /// `0` to opt out (e.g. for benchmarking the static-projection path).
    /// Single-shot paths (RULER eval, summarisation) always disable
    /// continuous re-projection regardless of this setting.
    pub reproject_every_n_tokens: usize,

    /// Maximum tokens to include in the provenance probe at each
    /// reprojection event, looking backward from the current decode
    /// position.  Caps the "thought window" — beyond this many tokens
    /// the prior reprojection already accounted for the older intent.
    ///
    /// Should be a small multiple of `CHUNK_SIZE` (32) for tidy block
    /// arithmetic; **default: `256`** — a full tool-call turn's discriminative
    /// span (the §80.2 sweep shows the true tool is always within Top-3 at this
    /// width; a narrower 64-token tail needs a much larger budget for 100% recall).
    pub reproject_max_probe_tokens: usize,

    /// Texts that, when sampled as a single token during decode, fire a
    /// reprojection in addition to the cadence trigger — but only once
    /// **more than 16 non-trigger tokens** have been decoded since the last
    /// projection.  That content gate stops short lines and runs of trigger
    /// tokens (e.g. consecutive newlines) from each re-orienting attention.
    /// Encoded via the conversation's tokenizer at policy build time; every
    /// token a text encodes to is a trigger (`provenance::boundaries`).
    ///
    /// **Default: `["\n"]`** — re-orient attention at line breaks,
    /// which usually mark paragraph or section boundaries where the
    /// model's intent shifts.  Set to `vec![]` to disable
    /// punctuation-driven reprojection (cadence-only).
    pub reproject_trigger_texts: Vec<String>,

    /// Whether a boundary inside the turn's think block reprojects. `false`
    /// holds the projection the block opened on until it closes, and
    /// reprojects once on the closing `</think>`, so the answer is decoded
    /// against a selection the whole thought informed. **Default: `true`.**
    pub reproject_in_thinking: bool,

    /// When `true`, turns on this conversation skip the per-turn
    /// projection rebuild once the slot is seeded: the prefill appends
    /// onto the cumulative slot instead of resetting + re-projecting, and
    /// continuous mid-decode reprojection is forced off. The turn still
    /// seals into the substrate. Used by append-only utility ingests
    /// (`code_reading`, `repo_map`) where re-projecting the whole trunk
    /// every turn is unnecessary and O(n²). **Default: `false`.**
    pub disable_reprojection: bool,

    /// Per-conversation KV-compression level override for this
    /// conversation's *turns*. `None` ⇒ use the engine-wide turn policy
    /// (the level set on the model builder). `Some(level)` quantizes this
    /// conversation's sealed turns at `level` instead — used to compress
    /// cold reference layers (e.g. `code_reading` at C8) harder than live
    /// dialogue (C4). Section policies are unaffected. **Default: `None`.**
    pub kv_compression_level: Option<u8>,

    /// When `true`, this conversation's turns drop the engine-wide K-format
    /// override (Q4_KS) so K is adaptively quantized per-block like V. Only
    /// meaningful alongside `kv_compression_level`. Set for `code_reading`
    /// so its K and V are both fully quantized. **Default: `false`.**
    pub kv_disable_k_override: bool,

    /// Pin this conversation's turns to a single uniform K quant format,
    /// bypassing adaptive per-block selection *and* the engine-wide Q4_KS K
    /// override. `None` keeps the adaptive/level path. Reserved for layers that
    /// want a fixed near-lossless seal (e.g. summary turns); currently unset by
    /// all callers. **Default: `None`.**
    pub kv_force_k_format: Option<QuantFormat>,

    /// V counterpart to [`Self::kv_force_k_format`]. **Default: `None`.**
    pub kv_force_v_format: Option<QuantFormat>,

    /// When `true`, skip the hot→warm quantize pass for this conversation's
    /// turns entirely — persist K/V in the native R16/F16 form (lossless). Used
    /// to capture full-resolution tool-call exemplars for the provenance work.
    /// Overrides the level/force fields above. **Default: `false`.**
    pub kv_lossless: bool,
}
