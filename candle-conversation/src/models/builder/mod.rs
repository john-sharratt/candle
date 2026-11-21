//! The `ModelBuilder`: the fluent configuration that becomes a conversation engine.
//!
//! A `ModelBuilder` collects every setting a model is built with: the model and
//! tokenizer files or the spec's pack, sampling, thinking, compression, the
//! substrate and how it is opened, LoRA adapters, the selection budget and its
//! strata, and the hit-layer percentiles. Setters return the builder for chaining.
//! The builder becomes an engine by resolving the pack, binding the substrate,
//! loading the weights and building the engine, or by attaching to an engine another
//! tenant already built, in which case its settings are merged with the others'.
//! Settings a model cannot honour are refused before a load.
//!
//! - `build_engine`: the link measurement, the engine configuration and the
//!   substrate binding.
//! - `builder_config`: the conversation configuration and the GGUF header read.
//! - `load_arch`: each architecture's loader and the settings it refuses.
//! - `pack`: resolving the model pack and reading its identity.
//! - `tests_header`, `tests_identity`, `tests_settings`: tests.

use super::gguf_rope::gguf_rope;
#[cfg(not(feature = "hub"))]
use super::pack_source::NoFetch;
use super::pack_source::{is_pack, resolve_local, resolve_spec, ResolvedModel};
use super::{ModelArch, ModelSpec, RopePreset};
use crate::cancel::CancelToken;
use crate::config::{
    pick_max_hot_turns, DecodeHealthConfig, EngineConfig, SamplingConfig, SequenceConfig,
    REPROJECT_EVERY_N_TOKENS, REPROJECT_MAX_PROBE_TOKENS, REPROJECT_TRIGGER_TEXTS,
};
use crate::engine::{ConversationEngine, EngineAttacher};
use crate::engine_slot::CoreRequests;
use crate::error::ConversationError;
use crate::guest::checkpoint::gguf_header;
use crate::models::identity::{rope_key, ModelIdentity, WeightFiles};
use crate::models::DialectType;
use crate::persistence::{Admit, BoundSubstrate, OpenProgressHook, OpenStage, SharedSubstrate};
use crate::prefill_width::prefill_pass_tokens;
use crate::projection::{CorruptTurnPolicy, LayerId};
use crate::tenant_rank::TenantRank;
use crate::tree::ConversationTreeConfig;
use crate::ManagedBatchedModel;
use candle::cuda_backend::{link_probe, LoadWindow};
use candle::quantized::Int8Mode;
use candle::{DType, Device};
use candle_nn::kv_cache::{class_for_format, elems_per_chunk, KvFormat, SizeClass, N_PALETTE};
use candle_nn::CHUNK_SIZE;
use candle_transformers::models::batched_model::{BatchedInference, BatchedModelCore};
use candle_transformers::models::hit_layers::HitPercentiles;
use candle_transformers::models::load_phase::{LoadPhase, LoadProgress};
use candle_transformers::models::model_pack::keys::TOKENIZER_JSON;
#[cfg(feature = "hub")]
use candle_transformers::models::model_pack::HubFetch;
use candle_transformers::models::model_pack::{ModelPack, SourceFetch};
use candle_transformers::models::quantized_qwen38_moe::engine_model_digest;
use candle_transformers::models::selection_strata::StrataTokens;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

// ModelBuilder

/// Fluent builder for configuring and constructing a
/// [`ConversationEngine`](crate::ConversationEngine).
///
/// Created via [`Model::builder`]. All setters return `self` for chaining.
///
/// ```ignore
/// let b = Model::Qwen3_8B_Q4.builder()
///     .temperature(0.8)
///     .top_p(0.95)
///     .max_response_tokens(16384)
///     .seed(123);
///
/// let engine = b.engine(&device)?;
/// let mut conv = engine.new_conversation(
///     &b.system_prompt(),
///     b.conversation_config(),
/// )?;
/// ```
#[derive(Debug, Clone)]
pub struct ModelBuilder {
    pub(super) spec: ModelSpec,
    /// Co-resident models this deployment wants available between waves.
    ///
    /// Empty by default and empty for every caller that does not ask, which
    /// costs one atomic load per scheduler pass and nothing else. See
    /// [`Self::with_guest`] and [`crate::guest`].
    guests: crate::guest::GuestRegistry,
    /// Override: a local model pack, or a local checkpoint to pack (skips the
    /// spec's sources).
    model_path: Option<PathBuf>,
    /// Override: system prompt text
    system_prompt: Option<String>,
    /// Override: the `tokenizer.json` a local checkpoint is packed with. A
    /// pack carries its own.
    tokenizer_path: Option<PathBuf>,
    /// Sampling configuration (initialised from [`ModelSpec::default_sampling`]).
    sampling: SamplingConfig,
    /// `true` if the user explicitly called any sampling setter on this builder.
    /// When `false`, [`engine()`](Self::engine) auto-detects defaults from the
    /// GGUF `general.architecture` metadata field.
    sampling_user_set: bool,
    /// Max tokens to generate per assistant turn.
    max_response_tokens: usize,
    /// KV cache sequence length.
    max_seq_len: usize,
    /// Engine-level: max concurrent conversations.
    max_concurrent: usize,
    /// Thinking mode override.
    ///
    /// - `None` (default) — no override; the model may or may not think.
    /// - `Some(true)` (`--thinking`) — explicitly enable thinking.
    /// - `Some(false)` (`--no-thinking`) — explicitly suppress thinking:
    ///   `/no_think` is prepended to the system prompt, an empty `<think></think>`
    ///   block is prefilled, and non-thinking sampling parameters are used.
    thinking: Option<bool>,
    /// KV adaptive compression level (0–9).  Defaults to C5.
    kv_compression_level: u8,
    /// Show special tokens in streamed output (debug/diagnostics).
    show_special_tokens: bool,
    /// Optional path to write penalty state to during decoding.
    penalty_log_path: Option<PathBuf>,
    /// Decode health monitoring configuration.
    health_config: DecodeHealthConfig,
    /// Maximum Hot-tier turns before triggering Hot → Warm eviction.
    /// `0` = auto-compute from arena geometry in [`engine()`](Self::engine).
    max_hot_turns: usize,
    /// Workspace root whose `substrate/` directory backs the persistence
    /// redo log. `None` falls back to the process working directory.
    ///
    /// Ignored when [`Self::substrate`] handed over an already-open one.
    workspace_path: Option<PathBuf>,
    /// A substrate the host process opened and still writes to itself.
    ///
    /// `None` — the ordinary case — means the engine opens the directory
    /// [`Self::workspace_path`] names. A host that appends its own record
    /// classes to the same log must pass one instead, because a second writable
    /// handle to one `substrate/` silently drops records; see
    /// [`SharedSubstrate`].
    substrate: Option<SharedSubstrate>,
    /// Open the workspace's substrate read-only
    /// ([`SharedSubstrate::open_in_read_only`]) — for a tool reading a workspace
    /// another process may be appending to. `false` by default.
    read_only_substrate: bool,
    /// How long the substrate keeps a timeline tombstone before a compaction
    /// may shrink it to its deleted-set entry
    /// (`SubstratePersistence::set_tombstone_retention`). `None` leaves the
    /// substrate's own — its default, or what the host that handed it over
    /// set.
    tombstone_retention: Option<Duration>,
    /// Told how the substrate open goes when this builder opens it — the
    /// stage, and the records rewritten when a store written before
    /// identities is stamped (minutes on a large one). `None` when nobody is
    /// watching. See [`ModelBuilder::on_substrate_open`].
    open_progress: Option<OpenProgressHook>,
    /// The substrate's ingest-cancel latch, forwarded to
    /// [`EngineConfig::cancel`]. See [`ModelBuilder::cancel_token`].
    cancel: CancelToken,
    /// The rank of the tenant the engine's first substrate belongs to,
    /// forwarded to [`EngineConfig::tenant_rank`].
    tenant_rank: TenantRank,
    /// Per-layer corrupt-turn policy (from the projection schema), forwarded to
    /// [`EngineConfig::layer_corrupt_turn`] so the startup reload drops the whole
    /// conversation (ingest layers) or just the turn (dialogue) per layer. Empty
    /// by default ⇒ every layer defaults to `DropConversation`.
    layer_corrupt_turn: HashMap<LayerId, CorruptTurnPolicy>,
    /// Override for `SchedulerConfig::large_prefill_max_tokens`.
    ///
    /// `None` sizes it to the card at engine start, by its VRAM and its measured
    /// host→device link (`crate::prefill_width`). A workload that reliably
    /// has more than a turn or two waiting — a bulk ingest — may ask for more;
    /// see [`ModelBuilder::prefill_pass_tokens`]. An ask is clamped to the card's
    /// width either way.
    prefill_pass_tokens: Option<usize>,
    /// LoRA adapters to load alongside the base weights — `(name, directory)`.
    ///
    /// Empty by default. See [`ModelBuilder::lora`].
    loras: Vec<(String, PathBuf)>,
    /// QSA selection budget, in positions, for an architecture whose attention
    /// selects. `None` runs the checkpoint's own. See
    /// [`ModelBuilder::qsa_selection_budget`].
    qsa_selection_budget: Option<usize>,
    /// How a selecting attention divides a query's candidates.
    /// [`StrataTokens::DEFAULT`] unless set. See [`ModelBuilder::qsa_strata`].
    qsa_strata: StrataTokens,
    /// Which selecting layers count hits. [`HitPercentiles::standard`] unless
    /// set — `None` until a tenant asks, so a merge can take another tenant's.
    /// See [`ModelBuilder::hit_layers`].
    hit_layers: Option<HitPercentiles>,
}

impl ModelBuilder {
    /// Create a builder from a [`ModelSpec`], using its defaults.
    ///
    /// This is called automatically by [`Model::builder`] and
    /// [`Model::custom`], but can also be used directly.
    pub fn from_spec(spec: ModelSpec) -> Self {
        Self {
            guests: crate::guest::GuestRegistry::new(),
            sampling: spec.default_sampling.clone(),
            sampling_user_set: false,
            max_seq_len: spec.max_seq_len,
            max_response_tokens: 16384,
            max_concurrent: 4,
            system_prompt: None,
            model_path: None,
            tokenizer_path: None,
            thinking: None,
            kv_compression_level: 5,
            show_special_tokens: false,
            penalty_log_path: None,
            health_config: DecodeHealthConfig::default(),
            max_hot_turns: 0,
            workspace_path: None,
            substrate: None,
            read_only_substrate: false,
            tombstone_retention: None,
            open_progress: None,
            cancel: CancelToken::new(),
            tenant_rank: TenantRank::FIRST,
            layer_corrupt_turn: HashMap::new(),
            prefill_pass_tokens: None,
            loras: Vec::new(),
            qsa_selection_budget: None,
            qsa_strata: StrataTokens::DEFAULT,
            hit_layers: None,
            spec,
        }
    }

    /// Load a PEFT LoRA adapter from `dir` under `name`, available to any
    /// conversation that asks for it.
    ///
    /// `dir` holds `adapter_config.json` and `adapter_model.safetensors` as PEFT
    /// writes them. The adapter is loaded **up front**, with the model: it is a
    /// few hundred megabytes shared by every conversation that opts in, and
    /// loading it per conversation would pay that repeatedly for one copy.
    ///
    /// **Nothing is merged into the base weights.** The base projection is still
    /// the quantized `Wx`; the adapter adds a rank-`r` term to its result. So
    /// one resident checkpoint serves adapted and unadapted conversations at the
    /// same time, which is what makes the opt-in per conversation possible at
    /// all — see [`Conversation::set_lora`](crate::Conversation::set_lora).
    ///
    /// Called more than once to load several adapters; each is addressed by its
    /// own name. Waves are grouped by adapter, so a conversation's choice never
    /// affects another's output — only which wave it rides in.
    pub fn lora(mut self, name: impl Into<String>, dir: impl Into<PathBuf>) -> Self {
        self.loras.push((name.into(), dir.into()));
        self
    }

    /// Make co-resident models available between the engine's waves.
    ///
    /// The registry holds *constructors*: nothing is loaded until a job for a
    /// guest is queued, and each drain builds a fresh instance so a guest that
    /// failed half-way through a load is not inherited by the next one. A
    /// deployment that configures none pays one atomic load per scheduler pass
    /// and nothing else.
    ///
    /// ```ignore
    /// let mut guests = GuestRegistry::new();
    /// guests.register(Guest::Matte, move || {
    ///     Box::new(MatteGuest::new(MatteSpec::new(onnx.clone(), MatteFamily::default())))
    /// });
    /// let engine = ModelBuilder::from_spec(spec).guests(guests).engine(&device)?;
    /// ```
    ///
    /// See [`crate::guest`] for what a drain does and why normal inference is
    /// blocked while one runs.
    pub fn guests(mut self, guests: crate::guest::GuestRegistry) -> Self {
        self.guests = guests;
        self
    }

    /// How many tokens one prefill forward carries.
    ///
    /// **A wave's fixed cost is paid per slab regardless of width** — the
    /// sweep of every layer's launches and routed-expert fetches, measured at ~2.37 s on the
    /// 35B — while the compute-saturation cap is soft: tokens past it cost the
    /// same per token as the ones before. So the budget decides how often that
    /// fixed cost is paid, and a workload whose items are nearly as large as the
    /// budget pays it *per item*.
    ///
    /// That is exactly what it cost `npcd`'s layer ingest. Its documents average
    /// ~1,900 tokens against the 2,048 default, so every document was its own
    /// slab and every slab paid the whole fixed sweep: measured 1.9 s a document
    /// with the GPU at 100%, for work that should run at the batched-forward
    /// gate's rate.
    ///
    /// The default is deliberately conservative because interactive serving
    /// rarely has more than a turn queued, and a forward that runs starved
    /// wastes the same fixed cost on fewer tokens. Raise it only where the
    /// workload reliably keeps a wide forward fed. Bounded above by the card's
    /// width (`crate::prefill_width`), past which each window's tier is bought
    /// from the expert zone, and by the model's own per-forward ceiling, which
    /// the scheduler clamps to.
    pub fn prefill_pass_tokens(mut self, tokens: usize) -> Self {
        self.prefill_pass_tokens = Some(tokens);
        self
    }

    /// Set the per-layer corrupt-turn policy map (from the projection schema).
    /// Forwarded to the engine so the startup reload applies the right policy per
    /// layer — `drop_conversation` for ingest layers, `drop_turn` for dialogue.
    pub fn corrupt_turn_policies(
        mut self,
        policies: HashMap<LayerId, CorruptTurnPolicy>,
    ) -> ModelBuilder {
        self.layer_corrupt_turn = policies;
        self
    }

    /// Set the workspace root whose `substrate/` directory backs the
    /// persistence redo log.
    ///
    /// Has no effect once [`Self::substrate`] has handed over an open one —
    /// that substrate already knows where it lives.
    pub fn workspace_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.workspace_path = Some(path.into());
        self
    }

    /// Hand the engine a substrate this process already opened, instead of a
    /// path to open its own at.
    ///
    /// Required of any host that writes its own records into the same redo log:
    /// one `substrate/` admits exactly one writable handle per process, and a
    /// second one loses records rather than failing. See [`SharedSubstrate`].
    pub fn substrate(mut self, shared: SharedSubstrate) -> Self {
        self.substrate = Some(shared);
        self
    }

    /// Open the workspace's substrate **read-only**.
    ///
    /// For a tool that inspects a workspace a running daemon owns: the engine
    /// resumes, prefills and decodes entirely in RAM and writes nothing under
    /// `substrate/` from start through shutdown. The store must already exist.
    ///
    /// Has no effect once [`Self::substrate`] has handed over an open one —
    /// that handle's own mode rules.
    pub fn read_only_substrate(mut self, read_only: bool) -> Self {
        self.read_only_substrate = read_only;
        self
    }

    /// How long the substrate keeps a timeline tombstone before a compaction
    /// may shrink it to its deleted-set entry — the daemon's
    /// `--tombstone-retention` (`docs/replicated_substrate.md` §6.8). Applies
    /// to a handed-over substrate as to one this builder opens.
    pub fn tombstone_retention(mut self, retention: Duration) -> Self {
        self.tombstone_retention = Some(retention);
        self
    }

    /// Report the substrate open this builder makes — its stage, and the
    /// records rewritten when the store is stamped with identities — to
    /// `hook`, for a loading screen. Has no effect once [`Self::substrate`] has
    /// handed over an open one.
    pub fn on_substrate_open(mut self, hook: OpenProgressHook) -> Self {
        self.open_progress = Some(hook);
        self
    }

    /// The latch that stops this substrate's ingest — the daemon's own, made
    /// before the engine so a shutdown during the load still reaches the ingest
    /// the load starts. Every conversation on the engine's substrate carries it.
    pub fn cancel_token(mut self, cancel: CancelToken) -> Self {
        self.cancel = cancel;
        self
    }

    /// Rank the engine's first substrate as tenant `rank` — where its
    /// sequences stand against a substrate attached later
    /// ([`crate::ConversationEngine::attach`]). [`TenantRank::FIRST`] unless set.
    pub fn tenant_rank(mut self, rank: TenantRank) -> Self {
        self.tenant_rank = rank;
        self
    }

    /// Run the QSA selection with `positions` in place of the checkpoint's own
    /// budget; `None` keeps the checkpoint's.
    ///
    /// A budget of at least the checkpoint's context makes the selection the
    /// identity — the same engine and the same code path, reading every cell —
    /// which is the dense control a retrieval failure is judged against. Only
    /// an architecture whose attention selects takes one: [`Self::load_model`]
    /// refuses it for any other rather than accept a setting that changes
    /// nothing, and refuses a budget its selection kernel cannot run.
    pub fn qsa_selection_budget(mut self, positions: Option<usize>) -> Self {
        self.qsa_selection_budget = positions;
        self
    }

    /// How a selecting attention divides a query's candidates
    /// (`docs/qsa_stratified_selection.md`): cut into windows that each spend
    /// the whole budget, with the system prompt and the recent span ranked in,
    /// or forced into, every one, all stated in positions.
    ///
    /// [`StrataTokens::DEFAULT`] unless set; [`StrataTokens::WHOLE`] is the
    /// checkpoint's single ranking over every candidate. Handed to every model
    /// through `ManagedBatchedModel::set_selection_strata`, after its budget:
    /// a model whose attention attends every position has nothing to divide,
    /// and a selecting one refuses a strata whose windows its kernel could not
    /// hold at the deepest position it reaches.
    pub fn qsa_strata(mut self, strata: StrataTokens) -> Self {
        self.qsa_strata = strata;
        self
    }

    /// Which selecting layers keep index rows for unloaded pieces and count
    /// hits (`docs/hit_driven_projection.md` §6.1), as percentiles over them.
    /// Handed to every model through `ManagedBatchedModel::set_hit_layers`; a
    /// model whose attention reads every position has nothing to count.
    pub fn hit_layers(mut self, layers: HitPercentiles) -> Self {
        self.hit_layers = Some(layers);
        self
    }

    // ── File overrides ─────────────────────────────────────────────────

    /// Load a local file instead of the spec's: a model pack as it is, or a
    /// checkpoint, which is packed into the model cache with the
    /// [`Self::tokenizer_path`] it then needs — never beside the file, and the
    /// file is never deleted.
    pub fn model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// The `tokenizer.json` a local checkpoint ([`Self::model_path`]) is packed
    /// with.
    pub fn tokenizer_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.tokenizer_path = Some(path.into());
        self
    }

    /// Set both model and tokenizer paths from a directory.
    ///
    /// Resolves to `<dir>/<model_filename>` (from the spec) and
    /// `<dir>/tokenizer.json`.
    pub fn model_dir(mut self, dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        self.model_path = Some(dir.join(&self.spec.model_filename));
        self.tokenizer_path = Some(dir.join("tokenizer.json"));
        self
    }

    /// Auto-detect model from a directory containing a `.gguf` file and
    /// `tokenizer.json`.
    ///
    /// Scans the directory for a single `.gguf` file, reads its GGUF
    /// metadata to determine architecture, dialect, sampling, context
    /// length, and thinking support.  Returns a fully configured builder
    /// that requires no `--model` preset.
    ///
    /// If the directory contains multiple `.gguf` files, the first one
    /// found (alphabetically) is used.
    ///
    /// ```ignore
    /// let builder = ModelBuilder::from_gguf_dir("/models/my-model")?;
    /// let engine = builder.engine(&device)?;
    /// ```
    pub fn from_gguf_dir(dir: impl AsRef<Path>) -> crate::Result<Self> {
        let dir = dir.as_ref();

        // Find the .gguf file.
        let mut gguf_files: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| {
                ConversationError::Model(candle::Error::Msg(format!(
                    "read dir '{}': {}",
                    dir.display(),
                    e
                )))
            })?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .map(|ext| ext == "gguf")
                    .unwrap_or(false)
            })
            .collect();

        gguf_files.sort_by_key(|e| e.file_name());

        let gguf_entry = gguf_files.first().ok_or_else(|| {
            ConversationError::Model(candle::Error::Msg(format!(
                "no .gguf file found in '{}'",
                dir.display()
            )))
        })?;
        let gguf_path = gguf_entry.path();
        let gguf_filename = gguf_entry.file_name().to_string_lossy().into_owned();

        tracing::info!("Auto-detected GGUF: {}", gguf_path.display());

        // Read metadata from the GGUF header.
        let info = Self::detect_sampling_from_gguf(&gguf_path)?;
        let model_bytes = std::fs::metadata(&gguf_path)
            .map_err(|e| ConversationError::Model(candle::Error::Msg(format!("{e}"))))?
            .len();

        let arch = info.arch.unwrap_or(ModelArch::Llama);
        let dialect_type = info.dialect.unwrap_or(DialectType::ChatML);

        let spec = ModelSpec {
            arch,
            // A model detected from a local GGUF brings no adapters of its own.
            // The caller attaches any it wants with `ModelBuilder::lora`, which
            // is the same path a preset's adapters take.
            loras: Vec::new(),
            chat_format: dialect_type,
            dialect: dialect_type.dialect(),
            model_repo: String::new(),
            model_filename: gguf_filename,
            // A custom model already IS a local file — the caller handed one
            // over. There is no prepare step to report and no repo to skip.
            prepared_from_source: false,
            model_bytes,
            // Built from a local file, which has no repository to pin a revision in.
            model_rev: String::new(),
            // A custom model replaced nothing, so there is no base to read gates from. Such a
            // checkpoint is refused outright if its recurrent gates are quantized — which is
            // the right answer when the caller chose the file themselves.
            gate_donor: None,
            // Nor anything to assemble it from: a custom model is the one file handed over.
            tensor_overrides: Vec::new(),
            tokenizer_repo: String::new(),
            // A custom model is built from local files; there is no repo to
            // pin a revision of.
            tokenizer_rev: String::new(),
            default_system_prompt: "You are a helpful, accurate, and concise assistant.".into(),
            max_seq_len: info.context_length.unwrap_or(8192),
            rope: arch.file_rope(),
            default_sampling: info.sampling.clone(),
            supports_thinking: info.has_thinking,
            non_thinking_sampling: info.non_thinking.clone(),
        };

        let mut builder = Self::from_spec(spec);
        builder.model_path = Some(gguf_path);
        builder.tokenizer_path = Some(dir.join("tokenizer.json"));
        // The sampling came from GGUF detection, not user override.
        builder.sampling = info.sampling;
        builder.sampling_user_set = false;
        Ok(builder)
    }

    // ── Sampling ───────────────────────────────────────────────────────

    /// Set the full sampling configuration, replacing the model default.
    pub fn sampling(mut self, s: SamplingConfig) -> Self {
        self.sampling = s;
        self.sampling_user_set = true;
        self
    }

    /// Apply a named sampling preset, replacing the model default.
    ///
    /// See [`SamplingConfig::preset_names()`] for available presets.
    /// Panics if `name` is not a recognized preset.
    pub fn sampler_preset(mut self, name: &str) -> Self {
        self.sampling = SamplingConfig::preset(name)
            .unwrap_or_else(|| panic!("unknown sampler preset: '{}'", name));
        self.sampling_user_set = true;
        self
    }

    /// Adjust the temperature.
    ///
    /// Values <= 0 enable argmax (greedy) decoding.
    pub fn temperature(mut self, t: f32) -> Self {
        self.sampling.temperature = t;
        self.sampling_user_set = true;
        self
    }

    /// Set the KV adaptive compression level (0 = near-lossless, 9 = maximum).
    ///
    /// Defaults to C5 (moderate compression, minimal quality loss).
    pub fn compression_level(mut self, level: u8) -> Self {
        self.kv_compression_level = level;
        self
    }

    /// Configure decode health monitoring.
    ///
    /// Health checks are only active when the crate is built with
    /// `--features decode-health`. The config is always accepted; if the
    /// feature is absent the config is stored but has no effect.
    ///
    /// # Example
    /// ```ignore
    /// builder.health(DecodeHealthConfig::for_chat())
    /// ```
    pub fn health(mut self, config: DecodeHealthConfig) -> Self {
        self.health_config = config;
        self
    }

    /// Adjust top-p (nucleus sampling threshold).
    ///
    /// Values >= 1.0 disable top-p filtering.
    pub fn top_p(mut self, p: f32) -> Self {
        self.sampling.top_p = p;
        self.sampling_user_set = true;
        self
    }

    /// Adjust top-k filtering threshold.
    ///
    /// Values <= 0 disable top-k filtering.
    pub fn top_k(mut self, k: i32) -> Self {
        self.sampling.top_k = k;
        self.sampling_user_set = true;
        self
    }

    /// Set the repeat penalty for recently generated tokens.
    ///
    /// Values <= 1.0 disable repeat penalty.
    pub fn repeat_penalty(mut self, p: f32) -> Self {
        self.sampling.repeat_penalty = p;
        self.sampling_user_set = true;
        self
    }

    /// Set the frequency penalty.
    ///
    /// Values <= 0.0 disable frequency penalty.
    pub fn frequency_penalty(mut self, p: f32) -> Self {
        self.sampling.frequency_penalty = p;
        self.sampling_user_set = true;
        self
    }

    /// Set the presence penalty.
    ///
    /// Values <= 0.0 disable presence penalty.
    pub fn presence_penalty(mut self, p: f32) -> Self {
        self.sampling.presence_penalty = p;
        self.sampling_user_set = true;
        self
    }

    /// Set the RNG seed for sampling.
    pub fn seed(mut self, s: u64) -> Self {
        self.sampling.seed = s;
        self.sampling_user_set = true;
        self
    }

    // ── Generation limits ──────────────────────────────────────────────

    /// Maximum tokens to generate per assistant turn.
    pub fn max_response_tokens(mut self, n: usize) -> Self {
        self.max_response_tokens = n;
        self
    }

    /// Enable thinking/reasoning mode.
    ///
    /// When `true`, thinking models are allowed to produce `<think>` blocks.
    /// Has no effect on models where
    /// [`supports_thinking`](ModelSpec::supports_thinking) is `false`.
    pub fn thinking(mut self, enable: bool) -> Self {
        self.thinking = Some(enable);
        self
    }

    /// Whether thinking suppression is explicitly active.
    ///
    /// Returns `true` only when the user has explicitly passed
    /// `thinking(false)` (i.e. `--no-thinking`).  In this mode,
    /// `/no_think` is prepended to the system prompt, an empty `<think></think>`
    /// block is prefilled, and non-thinking sampling parameters are used.
    pub fn should_suppress_thinking(&self) -> bool {
        self.thinking == Some(false)
    }

    /// Enable or disable automatic boundary-injection KV prototype generation.
    ///
    /// Show special tokens in streamed output.
    ///
    /// When `true`, tokens like `<think>`, `</think>`, and `<|im_end|>`
    /// are included verbatim in the text stream instead of being stripped.
    /// Useful for debugging prompt/response formatting.
    pub fn show_special_tokens(mut self, enable: bool) -> Self {
        self.show_special_tokens = enable;
        self
    }

    /// Path to write penalty state log during decoding (for debugging).
    pub fn penalty_log(mut self, path: impl Into<PathBuf>) -> Self {
        self.penalty_log_path = Some(path.into());
        self
    }

    /// Maximum sequence length (KV cache allocation).
    pub fn max_seq_len(mut self, n: usize) -> Self {
        self.max_seq_len = n;
        self
    }

    /// Maximum concurrent conversations on the engine.
    pub fn max_concurrent(mut self, n: usize) -> Self {
        self.max_concurrent = n;
        self
    }

    /// Override the auto-computed Hot → Warm eviction limit.
    ///
    /// By default, [`engine()`](Self::engine) derives `max_hot_turns` from
    /// the KV arena size and `max_response_tokens` via [`pick_max_hot_turns`].
    /// Pass a non-zero value here to fix the limit manually.
    /// `0` re-enables auto-computation.
    pub fn max_hot_turns(mut self, n: usize) -> Self {
        self.max_hot_turns = n;
        self
    }
}

// Display

impl std::fmt::Display for ModelBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = self
            .spec
            .model_filename
            .strip_suffix(".gguf")
            .unwrap_or(&self.spec.model_filename);
        write!(f, "{name}")
    }
}

mod build_engine;
mod builder_config;
mod load_arch;
mod pack;
#[cfg(test)]
mod tests_header;
#[cfg(test)]
mod tests_identity;
#[cfg(test)]
mod tests_settings;

use build_engine::*;
use pack::*;
