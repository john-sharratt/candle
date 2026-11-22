//! Holds the engine-wide and scheduler settings the conversation engine is built
//! with.
//!
//! The substrate is opened and bound to the model before the engine is built, so
//! the engine cannot run against a substrate another model wrote. The
//! ingest-cancel latch is the daemon's, so a shutdown reaches an ingest that
//! began before the engine existed. The dialect supplies the structural strings
//! the turn boundaries need and defaults to ChatML. Per-layer corrupt-turn
//! policies are set before the startup reload, so the reload already sees them,
//! and with none every layer drops its conversation on an unrecoverable turn.
//! The scheduler's per-forward prefill target is the main knob for expert-load
//! amortisation, and it is set where enough work is queued to fill each forward.

use super::*;

// Engine Configuration

/// Engine-wide configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Configuration for the batched inference session (chunk size, arena size, KV format).
    pub batched_config: BatchedConfig,

    /// Maximum number of concurrent conversations.
    pub max_concurrent_conversations: usize,

    /// EOS token ID. Generation stops when this token is produced.
    pub eos_tokens: TokenBuffer,

    /// Vocabulary size for the model. Required for penalty tracking buffers.
    pub vocab_size: usize,

    /// Maximum recent token history length for repeat/DRY penalties.
    /// Larger values allow longer-range penalty detection but use more memory.
    pub max_recent_len: usize,

    /// Scheduler-specific configuration.
    pub scheduler: SchedulerConfig,

    /// When `true`, special tokens (e.g. `<think>`, `</think>`, `<|im_end|>`)
    /// are included in streamed text output instead of being stripped.
    pub show_special_tokens: bool,

    /// Optional path to write penalty state to during decoding.
    /// Useful for debugging why tokens are being penalized.
    pub penalty_log_path: Option<std::path::PathBuf>,

    /// Decode health monitoring configuration.
    ///
    /// Checks only run when built with `--features decode-health` AND
    /// `health.enabled == true`. Safe to set unconditionally.
    pub health: DecodeHealthConfig,

    /// The substrate the engine persists to, already bound to the model it
    /// runs ([`BoundSubstrate::bind`]): the engine never opens one itself, so it
    /// cannot run on a substrate another model wrote. `ModelBuilder` opens and
    /// binds it once the model has loaded, so the expert cache pins its warm
    /// tier against a machine the substrate's index does not yet fill.
    pub substrate: BoundSubstrate,

    /// The substrate's ingest-cancel latch — the daemon's, handed over through
    /// `ModelBuilder::cancel_token` so a shutdown reaches its ingest from
    /// before the engine exists. A fresh token by default.
    pub cancel: CancelToken,

    /// The rank of the tenant the substrate belongs to — where its sequences
    /// stand against another tenant's attached to the same engine.
    /// [`TenantRank::FIRST`] by default.
    pub tenant_rank: TenantRank,

    /// The model's chat dialect — the structural-token strings the
    /// engine needs to know about for inter-turn boundary handling.
    /// Pre-tokenised at engine construction into the assembler's
    /// `BoundaryMarkers` so every projection can wrap each
    /// `Sealed::Turn` in a live-prefilled boundary run without
    /// re-tokenising.  Defaults to ChatML (the most common dialect
    /// in the modern model lineup the engine supports); model
    /// builders override via `ret.dialect = ...`.
    pub dialect: Dialect,

    /// Per-layer [`CorruptTurnPolicy`] (keyed by `LayerId`), applied when a turn
    /// is unrecoverable during the startup substrate reload. Set on the substrate
    /// in `ConversationEngine::new` *before* the reload thread is spawned, so the
    /// reload sees the right policy per layer (empty ⇒ every layer defaults to
    /// `DropConversation`). Populated by `ModelBuilder::corrupt_turn_policies`
    /// from the projection schema. Empty by default.
    pub layer_corrupt_turn: HashMap<LayerId, CorruptTurnPolicy>,
}

impl EngineConfig {
    pub fn new(eos_tokens: TokenBuffer, substrate: BoundSubstrate) -> Self {
        let r16 = KvFormat::Quantized(QuantFormat::R16);
        Self {
            batched_config: BatchedConfig {
                k_format: r16,
                v_format: r16,
                ..BatchedConfig::default()
            },
            max_concurrent_conversations: 4,
            eos_tokens,
            vocab_size: 128_000, // Common for modern LLMs
            // 2048 tokens ≈ 10 turns of 200-token responses; large enough for DRY and
            // cross-turn penalties to cover a full roleplay conversation window without
            // letting earlier turns fall out of the penalty buffer undetected.
            max_recent_len: 2048,
            scheduler: SchedulerConfig::default(),
            show_special_tokens: false,
            penalty_log_path: None,
            health: DecodeHealthConfig::default(),
            substrate,
            cancel: CancelToken::new(),
            tenant_rank: TenantRank::FIRST,
            dialect: Dialect::chat_ml(),
            layer_corrupt_turn: HashMap::new(),
        }
    }
}

/// Scheduler-specific configuration.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Small-prefill threshold, in tokens: the boundary between a small prefill
    /// that rides along with decode and a large one that batches as a backlog
    /// wave. Carried in the config; no scheduler path reads it.
    pub small_prefill_threshold: usize,

    /// **Per-forward prefill token target** (`max_prefill_pass_tokens`): the
    /// tokens one prefill forward carries. The primary knob for the expert-load
    /// amortization curve (design §1.2/§6) — a larger target streams the same
    /// all-expert weight load over more tokens. Bounded above by the model-side
    /// per-forward ceiling ([`crate::models::batched_inference`]'s
    /// `MAX_PREFILL_TOKENS`); set at the point where enough parallel work is
    /// reliably queued to fill every forward, so no forward runs starved.
    pub large_prefill_max_tokens: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            small_prefill_threshold: 128,
            // 2048 tokens/forward: with the parallel scope-ingest keeping ~24
            // scopes queued at once (`CODE_READ_PARALLELISM`), every forward
            // fills to this target — saturating ~all experts by the
            // coupon-collector bound — instead of stalling on a 4096-token
            // target that the reduced-scope scope KV rarely reaches. ≤ the
            // model's `MAX_PREFILL_TOKENS` ceiling.
            //
            // Raising it to that 4096 ceiling is a measured dead end on a 16 GiB
            // card, and the reason is not the one the ceiling suggests. A bigger
            // forward IS faster in isolation — 4036 tokens ran at 1223 tok/s and
            // 7322 at 1464, against ~900 at 3000 — but its KV is allocated
            // BEFORE admission measures what is left, so ~576 MiB more in flight
            // dropped availability from 1470 to 966 MiB and narrowed admission
            // from 4.6 sequences per pass to 1.9. The wide forwards still
            // happened; they became too rare to pay for the ones they starved.
            large_prefill_max_tokens: 2048,
        }
    }
}
