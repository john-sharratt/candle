//! Pre-configured model presets with a builder pattern for streamlined setup.
//!
//! The [`Model`] enum provides strongly-typed presets (HF coordinates, chat
//! format, EOS token, default sampling). Call [`.builder()`](Model::builder)
//! to customise any setting before constructing a
//! [`ConversationEngine`](crate::ConversationEngine).
//!
//! # Quick Start
//!
//! ```ignore
//! use candle_conversation::models::Model;
//!
//! let device = candle::Device::cuda_if_available(0)?;
//!
//! // One-liner (downloads from HuggingFace, requires `hub` feature):
//! let engine = Model::Qwen3_8B_Q4.engine(&device)?;
//!
//! // With customisation:
//! let b = Model::Qwen3_14B_Q4.builder()
//!     .temperature(0.8)
//!     .max_response_tokens(4096);
//! let engine = b.engine(&device)?;
//! let mut conv = engine.new_conversation(&b.system_prompt(), b.conversation_config())?;
//! let resp = conv.send("Hello!")?;
//! ```
//!
//! # Local Files
//!
//! ```ignore
//! // Directory containing the GGUF and tokenizer.json:
//! let engine = Model::Qwen3_8B_Q4.builder()
//!     .model_dir("/models/qwen3-8b")
//!     .engine(&device)?;
//!
//! // Or explicit paths:
//! let engine = Model::Qwen3_8B_Q4.builder()
//!     .model_path("/models/Qwen3-8B-Q4_K_M.gguf")
//!     .tokenizer_path("/models/tokenizer.json")
//!     .engine(&device)?;
//! ```
//!
//! # MoE Support
//!
//! **Qwen3-30B-A3B (MoE)** is supported via `quantized_qwen3_moe` with an LRU
//! expert cache and `cudaHostAllocMapped` for non-expert weight overflow.
//! Requires a CUDA GPU with ≥16 GB VRAM.

mod builder;
mod dialect;
mod hermes3;
pub mod overrides;
mod qwen2;
mod qwen3;
mod qwen35_dense;
mod qwen36_moe;
mod qwen38_flash_next;
mod qwen3_moe;

pub use builder::ModelBuilder;
pub use dialect::*;

use crate::config::{SamplingConfig, SequenceConfig};
use crate::error::ConversationError;
use candle::DType;
use std::path::Path;

// ────────────────────────────────────────────────────────────────────────────
// Enums
// ────────────────────────────────────────────────────────────────────────────

/// Model architecture — selects the quantised weight loader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArch {
    /// `quantized_qwen3::ModelWeights`
    Qwen3,
    /// `quantized_qwen3_moe::ModelWeights`
    Qwen3Moe,
    /// `quantized_qwen2::ModelWeights`
    Qwen2,
    /// `quantized_llama::ModelWeights`
    Llama,
    /// `latent_moe::BatchedEngine` over the resident `latent_moe::Engine`
    /// (offline KO-repacked GGUF; paged-latent kernel attention).
    DeepSeekV4,
    /// `qwen35::HybridBatched` — the gated-DeltaNet ⁄ attention hybrid lineage
    /// (Qwen3.5 and its Qwen3.6 point release; GGUF arch string `qwen35moe`).
    ///
    /// The only arch here that carries per-sequence state **outside** the paged
    /// K/V, which is why it declares
    /// `ManagedBatchedModel::carries_recurrent_state`.
    Qwen35Hybrid,
    /// `qwen4exp::Qwen4ExpBatched` — Qwen3.8-Flash-Next: a 3:1 gated-DeltaNet ⁄
    /// sparse-attention hybrid over a 4-stream gated residual, 512 experts on
    /// every layer, an n-gram hash embedding injected at layer 1, and QSA block
    /// selection on the 12 full-attention layers.
    ///
    /// Carries **four** per-sequence states, the most of any arch here: the
    /// DeltaNet recurrence, the PLE convolution tail, the QSA index cache, and
    /// the paged K/V. The first three all live outside the K/V, so this
    /// declares `carries_recurrent_state` for the same reason
    /// [`Self::Qwen35Hybrid`] does.
    ///
    /// Loaded from a **locally prepared** merged GGUF whose experts are
    /// `Q4_KO` (a bit-exact import of the vendor's W4A16 release, not a
    /// requant — `qwen4exp/convert.rs`), the same posture as
    /// [`Self::DeepSeekV4`]'s offline KO artifact.
    Qwen4Exp,
    /// `qwen35::HybridBatched` over a **dense** checkpoint of the same lineage —
    /// the same DeltaNet/attention stack with the mixture taken out.
    ///
    /// Separate from [`ModelArch::Qwen35Hybrid`] because the loaders are not
    /// interchangeable, and each refuses the other's checkpoint by design: a
    /// routed file loaded densely would silently drop its experts, and a dense
    /// file loaded through the routed path would stand up an expert cache over
    /// weights that have none. Both refusals are correct, and having one arch
    /// for both is what makes them reachable — `npcd` hit exactly that, five
    /// minutes into a load, with a `is a dense checkpoint` error from the routed
    /// loader.
    ///
    /// The GGUF arch string does not distinguish them; the presence of MoE
    /// metadata does, and each loader checks it.
    Qwen35Dense,
}

impl ModelArch {
    /// The float width this architecture's reference implementation computes
    /// in, when it is known to differ from what the KV storage format implies.
    ///
    /// `BatchedInferenceSession::activation_dtype` otherwise derives the
    /// activation width from the KV cache's format, which reports F16 for a
    /// quantized backing because the live arena really is F16 (K in `R16`, V in
    /// plain F16). That is right for the arena and wrong for the residual
    /// stream, and the two are unrelated: how KV is *stored* says nothing about
    /// how wide the activations flowing between layers should be.
    ///
    /// It matters because F16 and BF16 are both 16 bits but not both able to
    /// hold the same values — BF16 carries F32's exponent range while F16 stops
    /// at 65504. The hybrid lineage's MoE routinely produces block outputs in
    /// the 1e5–1e6 range, which BF16 holds and F16 turns into `inf`; the `inf`
    /// then reaches the next layer's arithmetic as a NaN and, because this is
    /// the one arch carrying recurrent state, is written into that state and
    /// persists across waves.
    ///
    /// `Qwen3.5-0.8B`, `Qwen3.6-35B-A3B` and `Qwen3.8-27B` all declare
    /// `"dtype": "bfloat16"` in their published `config.json`, and all three
    /// share the `qwen35` GGUF arch string, so they share this arm.
    ///
    /// `None` for every other arch: their thresholds and gates were derived
    /// under F16 activations, and changing the width under them would
    /// invalidate that calibration without re-deriving it.
    pub fn native_activation_dtype(self) -> Option<DType> {
        match self {
            // Every member of the lineage — routed, dense, and Flash-Next. The
            // activation width is the stack's: the mixture does not change it,
            // and Qwen3.8-Flash-Next declares `"dtype": "bfloat16"` like the
            // rest, with a gate ladder that runs BF16.
            Self::Qwen35Hybrid | Self::Qwen35Dense | Self::Qwen4Exp => Some(DType::BF16),
            Self::Qwen3 | Self::Qwen3Moe | Self::Qwen2 | Self::Llama | Self::DeepSeekV4 => None,
        }
    }
}

/// Pre-configured model presets.
///
/// Each variant carries HF coordinates, architecture, chat format, and
/// recommended defaults. Call [`.builder()`](Model::builder) to customise,
/// or use convenience methods directly.
///
/// Use [`Model::custom`] (or [`ModelBuilder::from_spec`]) to bring your own
/// GGUF model that isn't in the preset list.
///
/// | Variant | Params | Quant | Arch | Format | VRAM |
/// |---|---|---|---|---|---|
/// | `Qwen3_8B_Q4` | 8 B | Q4_K_M | Qwen3 | ChatML | ~5 GB |
/// | `Qwen3_8B_Q6` | 8 B | Q6_K | Qwen3 | ChatML | ~7 GB |
/// | `Qwen3_14B_Q4` | 14 B | Q4_K_M | Qwen3 | ChatML | ~8 GB |
/// | `Qwen3_14B_Q5` | 14 B | Q5_K_M | Qwen3 | ChatML | ~10 GB |
/// | `Qwen3_14B_Q6` | 14 B | Q6_K | Qwen3 | ChatML | ~12 GB |
/// | `Qwen2_0_5B` | 0.5 B | Q4_0 | Qwen2 | ChatML | ~0.4 GB |
/// | `Hermes3_3B_Q6` | 3 B | Q6_K | Llama | ChatML | ~3 GB |
/// | `Hermes3_70B_Q4` | 70 B | Q4_K_M | Llama | ChatML | ~40 GB |
/// | `Qwen3_30B_A3B_Q4` | 30 B (3B active) | Q4_K_M | Qwen3Moe | ChatML | ~17 GB (LRU) |
/// | `Qwen3_30B_A3B_Q6` | 30 B (3B active) | Q6_K | Qwen3Moe | ChatML | ~25 GB (LRU) |
/// | `Qwen36_35B_A3B_Q4` | 35 B (3B active) | UD-Q4_K_M | Qwen35Hybrid | ChatML | ~22 GB (tiered) |
/// | `Qwen36_35B_A3B_AntiLoop_StyleTune` | 35 B (3B active) | Q4_K_M | Qwen35Hybrid | Qwen35 | ~22 GB (tiered) |
/// | `Custom(_)` | — | — | any | any | — |
#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub enum Model {
    // ── Qwen3 ──────────────────────────────────────────────────────────
    /// Qwen3-8B Q4_K_M — strong general-purpose 8B model (~5 GB).
    Qwen3_8B_Q4,
    /// Qwen3-8B Q6_K — higher-quality 8B quantisation (~7 GB).
    Qwen3_8B_Q6,
    /// Qwen3-14B Q4_K_M — strong 14B model (~8 GB).
    Qwen3_14B_Q4,
    /// Qwen3-14B Q5_K_M — balanced quality/size (~10 GB).
    Qwen3_14B_Q5,
    /// Qwen3-14B Q6_K — near-FP16 quality (~12 GB).
    Qwen3_14B_Q6,

    // ── Qwen3 MoE ──────────────────────────────────────────────────────
    /// Qwen3-30B-A3B Q4_K_M — MoE, 128 experts, 8 active (~17 GB with LRU).
    /// The sub-24 GB-VRAM fit; zend selects Q4 vs Q6 by measured VRAM.
    Qwen3_30B_A3B_Q4,
    /// Qwen3-30B-A3B Q6_K — MoE, 128 experts, 8 active (~25 GB with LRU).
    Qwen3_30B_A3B_Q6,

    // ── Qwen3.5 / 3.6 hybrid MoE ───────────────────────────────────────
    /// Qwen3.6-35B-A3B UD-Q4_K_M — the **hybrid**: 40 layers at 3:1, so 30
    /// gated-DeltaNet layers carrying a recurrent state and 10 attention layers
    /// carrying paged K/V. 256 experts, 8 active (~22 GB file; runs sub-24 GB
    /// through the three-tier expert cache).
    ///
    /// Three quarters of this stack has no K/V to splice, so a conversation's
    /// history cannot be reconstructed from sealed chunks alone — see
    /// `docs/deltanet_state_persistence.md`.
    Qwen36_35B_A3B_Q4,

    /// **The hybrid** — AntiLoop's Qwen3.6-35B-A3B, a fine-tune trained against
    /// repetition loops, under StyleTune's output head, both at Q4_K_M (~22 GB).
    /// `npcd`'s model. The same architecture, tokenizer and KV row as
    /// [`Model::Qwen36_35B_A3B_Q4`], with the stock checkpoint as its
    /// recurrent-gate donor.
    Qwen36_35B_A3B_AntiLoop_StyleTune,

    /// Qwen3.8-Flash-Next Q4_KO — 250 B total, ~13 B active, 48 layers at 3:1
    /// (36 gated-DeltaNet, 12 attention). Native 262,144-token context.
    ///
    /// Carries two classes of state no other preset does — a PLE window and a
    /// QSA index — neither of which is a delta-rule matrix, so both ride the
    /// turn record's model-opaque blob (`docs/qwen38_index_persistence.md`).
    ///
    /// **Its engine GGUF is prepared locally, not downloaded** — see
    /// [`qwen38_flash_next`] and [`ModelSpec::prepared_from_source`].
    Qwen38_FlashNext_Q4KO,

    /// Qwen3.5-9B Q6_K — the lineage's **dense** member (~7.5 GB), same hybrid
    /// attention/DeltaNet stack with the mixture taken out.
    ///
    /// A hundred characters thinking about different things is the case where a
    /// routed model's expert amortisation is weakest, and the dense one has no
    /// expert cache to thrash — so a character costs the same whether it is the
    /// only one awake or one of a hundred.
    ///
    /// A deployment wanting a fine-tune of it in its place says so in
    /// `models.override.yaml` rather than adding a variant here — see
    /// [`overrides`].
    Qwen35_9B_Q6,

    /// Qwen3.5-0.8B Q8_0 — the lineage's smallest dense member (~0.8 GB), the
    /// same hybrid stack, dialect and tool-call style as its larger siblings.
    ///
    /// What zend's end-to-end tool suite runs: it loads in seconds where the
    /// production model takes most of a minute, and it speaks the production
    /// model's dialect, so the orchestration the suite drives is the real one.
    Qwen35_0_8B_Q8,

    // ── Qwen2 ──────────────────────────────────────────────────────────
    /// Qwen2-0.5B-Instruct Q4_0 — tiny, great for CI and testing (~0.4 GB).
    Qwen2_0_5B,

    // ── Hermes-3 / Llama ───────────────────────────────────────────────
    /// Hermes-3-Llama-3.2-3B Q6_K — fast, ChatML-tuned (~3 GB).
    Hermes3_3B_Q6,
    /// Hermes-3-Llama-3.1-70B Q4_K_M — large, needs ≥48 GB VRAM (~40 GB).
    Hermes3_70B_Q4,

    // ── Custom ─────────────────────────────────────────────────────────
    /// User-provided model specification.
    ///
    /// Construct via [`Model::custom`]:
    ///
    /// ```ignore
    /// let spec = ModelSpec {
    ///     arch: ModelArch::Llama,
    ///     chat_format: DialectType::ChatML,
    ///     model_repo: "my-org/my-model-GGUF".into(),
    ///     model_filename: "my-model-Q4_K_M.gguf".into(),
    ///     model_bytes: 4_500_000_000,
    ///     tokenizer_repo: "my-org/my-model".into(),
    ///     eos_token: "<|im_end|>".into(),
    ///     default_system_prompt: "You are a helpful assistant.".into(),
    ///     max_seq_len: 8192,
    ///     default_sampling: SamplingConfig::top_p(0.9, 0.7),
    /// };
    /// let engine = Model::custom(spec)
    ///     .model_dir("/models/my-model")
    ///     .engine(&device)?;
    /// ```
    Custom(ModelSpec),
}

// ────────────────────────────────────────────────────────────────────────────
// ModelSpec
// ────────────────────────────────────────────────────────────────────────────

/// One PEFT LoRA adapter's coordinates.
///
/// The adapter directory is a whole HF repo — PEFT writes `adapter_config.json`
/// and `adapter_model.safetensors` side by side, and the loader reads both — so
/// this names a repo rather than a file, which is the one place a LoRA's
/// coordinates differ from a GGUF's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraSpec {
    /// What a conversation calls it in
    /// [`Conversation::set_lora`](crate::Conversation::set_lora). Local to this
    /// deployment and deliberately short — the repo name is not a usable handle.
    pub name: String,
    /// HuggingFace repository holding the PEFT files.
    pub repo: String,
    /// Pinned revision, for the reason [`ModelSpec::tokenizer_rev`] gives at
    /// length: an unpinned repo names a moving target, and an adapter that
    /// silently changes under a deployment changes what its characters say with
    /// nothing in this codebase changing. `"main"` where a pin is not available.
    pub revision: String,
}

/// One tensor a preset reads from another published checkpoint instead of its own.
///
/// What lets one model be assembled from two conversions of the same base — see
/// [`ModelSpec::tensor_overrides`]. Coordinates as [`ModelSpec`] names its own checkpoint,
/// pinned the same way and for the same reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorOverrideSpec {
    /// The tensor's GGUF name, e.g. `output.weight`.
    pub tensor: String,
    /// HuggingFace repository of the checkpoint it is read from.
    pub repo: String,
    /// Pinned revision of that repository.
    pub revision: String,
    /// GGUF filename within it.
    pub filename: String,
}

/// Immutable metadata for a model variant.
///
/// Returned by [`Model::spec`]. For built-in presets the string fields are
/// populated from static literals; for [`Model::Custom`] they can be any
/// owned [`String`].
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Weight-loader architecture.
    pub arch: ModelArch,
    /// PEFT LoRA adapters loaded alongside this model's weights.
    ///
    /// Loaded once with the checkpoint and shared by every conversation that
    /// opts into one — nothing is merged, so the base stays quantized and one
    /// resident model serves adapted and unadapted conversations at the same
    /// time. A conversation chooses by name with
    /// [`Conversation::set_lora`](crate::Conversation::set_lora); the default is
    /// the base model.
    ///
    /// Empty for most presets. Non-empty means "this model always has these
    /// available", not "every conversation uses them".
    pub loras: Vec<LoraSpec>,
    /// Chat template format.
    pub chat_format: DialectType,
    /// Dialect used to construct chat messages
    pub dialect: Dialect,
    /// HuggingFace repository containing the GGUF file.
    ///
    /// When [`Self::prepared_from_source`] is set this names the repo the
    /// artifact is *built from*, not one that publishes it.
    pub model_repo: String,
    /// GGUF filename within the repository.
    pub model_filename: String,
    /// Set when [`Self::model_filename`] names an artifact this codebase
    /// **prepares** from [`Self::model_repo`]'s published files, rather than one
    /// published under that name.
    ///
    /// Flash-Next is the case that needs it: its engine GGUF is a local build —
    /// the pinned Q8_0 split, plus the W4A16 release's expert tensors imported
    /// to Q4_KO, plus the MTP head folded in as the block past the trunk, merged
    /// into one file. Nothing on the hub is that file.
    ///
    /// Resolution therefore skips the network for these: a 404 on a name that
    /// was never published is a confusing way to say "you have not run the
    /// prepare step", and retrying it on every start is worse. The resolver
    /// looks in the local cache and, failing that, says what to run.
    pub prepared_from_source: bool,
    /// Pinned revision of [`Self::model_repo`], as [`Self::tokenizer_rev`] pins the tokenizer's.
    ///
    /// **The weights are the one coordinate it was still possible to leave unpinned.** A repo
    /// and a filename name a moving target: resolution falls back to `refs/main`, so an
    /// upstream re-upload changes the weights a deployment serves with nothing in this
    /// codebase changing — and this lineage has already been bitten by exactly that once, on
    /// the gate side, where "an upstream re-upload silently invalidated a threshold tuning".
    /// The gates pinned a commit in response; the serving path kept resolving `main`, so the
    /// gate and the daemon could load different bytes from the same repo name.
    ///
    /// Empty means unpinned, which is right for a custom model built from a local file and is
    /// a gap anywhere else. A preset pins one wherever a verified commit exists for it.
    pub model_rev: String,
    /// Exact on-disk size of the GGUF file in bytes. Presets pin the
    /// published file's length; custom models read it from the local file.
    /// Downloaders use it for progress totals when the server omits
    /// Content-Length.
    pub model_bytes: u64,
    /// The checkpoint this one replaced, as `(repo, rev, filename)`, when it replaced one.
    ///
    /// **Not a second model — a source of last resort for tensors the primary got wrong.**
    /// The DeltaNet recurrent gates (`ssm_alpha`/`ssm_beta`) must be F32, and a conversion
    /// whose quant rules do not know this architecture stores them like any other 2-D weight;
    /// the model then loads cleanly and generates incoherent text from its first token. The
    /// base checkpoint still has them, and they are the tensors a fine-tune has least reason
    /// to have moved.
    ///
    /// Set by [`overrides`] when a local override replaces a preset's checkpoint, so it is
    /// always the preset the override displaced. `None` for a preset used as published — a
    /// stock conversion has nothing to repair — and for a custom model, which has no base.
    /// Nothing is fetched unless the primary is found to need it.
    pub gate_donor: Option<(String, String, String)>,
    /// Tensors read from another checkpoint instead of this one.
    ///
    /// **Part of the model, not a repair.** Where [`Self::gate_donor`] is held in reserve and
    /// fetched only when the primary turns out to need it, each of these is fetched on every
    /// load, because the model this spec names is made of them. The loader requires each to
    /// match the primary's shape, repacks it from its own quant type, and refuses a load that
    /// never reads one. Only the qwen35 lineage's loader reads them; the builder refuses a spec
    /// that names any for another arch.
    ///
    /// Dropped by an override that replaces [`Self::model_repo`]: they were chosen against the
    /// checkpoint that override displaced.
    pub tensor_overrides: Vec<TensorOverrideSpec>,
    /// HuggingFace repository containing `tokenizer.json`.
    pub tokenizer_repo: String,
    /// Pinned revision of [`Self::tokenizer_repo`], as the gates pin theirs.
    ///
    /// A repo alone names a moving target: resolution falls back to `refs/main`
    /// or "whichever snapshot happens to be cached", so the vocabulary a run
    /// encodes with can change without anything in this codebase changing. The
    /// gates have always pinned a revision for exactly that reason — an
    /// upstream re-upload once invalidated a threshold tuning silently — and
    /// the serving path pinning nothing is the same exposure with a substrate
    /// behind it. Empty means unpinned, which is only appropriate for a custom
    /// model built from a local file.
    pub tokenizer_rev: String,
    /// Default system prompt text (before chat-format wrapping).
    pub default_system_prompt: String,
    /// Maximum sequence length for KV cache allocation.
    pub max_seq_len: usize,
    /// Recommended default sampling strategy for this model family.
    pub default_sampling: SamplingConfig,
    /// Whether this model supports thinking/reasoning mode (`<think>` blocks).
    pub supports_thinking: bool,

    /// Alternate sampling parameters used when thinking is suppressed.
    ///
    /// Qwen3 recommends different settings for non-thinking mode:
    /// temperature=0.7, top_p=0.8, top_k=20.
    pub non_thinking_sampling: Option<SamplingConfig>,
}

// ────────────────────────────────────────────────────────────────────────────
// Model impl
// ────────────────────────────────────────────────────────────────────────────
impl Model {
    /// Create a [`Model::Custom`] from a user-provided [`ModelSpec`].
    ///
    /// Returns a [`ModelBuilder`] directly — no need to call `.builder()`.
    ///
    /// ```ignore
    /// let engine = Model::custom(spec)
    ///     .model_dir("/models/my-model")
    ///     .engine(&device)?;
    /// ```
    pub fn custom(spec: ModelSpec) -> ModelBuilder {
        ModelBuilder::from_spec(spec)
    }

    /// Full specification for this model variant, **with any local override
    /// applied**.
    ///
    /// Every consumer of a preset goes through here — the builder, the loader,
    /// npcd's console — so this is the one place an override has to be applied
    /// for it to be applied everywhere. [`Self::preset_spec`] is the same answer
    /// without the override, which only the override machinery and its tests
    /// want.
    pub fn spec(self) -> ModelSpec {
        let key = self.override_key();
        let spec = self.preset_spec();
        match key {
            Some(k) => overrides::apply(&k, spec),
            // `Custom` is already whatever the caller said it was. Overriding it
            // would mean a file on disk silently rewriting a spec the caller
            // constructed by hand, which is the opposite of what `Custom` is
            // for.
            None => spec,
        }
    }

    /// The variant's name as `models.override.yaml` addresses it, or `None` for
    /// [`Model::Custom`], which is not overridable.
    ///
    /// Taken from `Debug` rather than a hand-written match. A match would be a
    /// second list of every variant, and when the two drifted the symptom would
    /// be an override key that quietly stopped matching — a config file that
    /// looks right, parses, and does nothing. Every overridable variant is a
    /// unit variant, so its `Debug` output *is* the identifier;
    /// `every_variants_override_key_is_its_identifier` holds that true.
    pub fn override_key(&self) -> Option<String> {
        match self {
            Model::Custom(_) => None,
            other => Some(format!("{other:?}")),
        }
    }

    /// Every preset, in declaration order — what a name given on a command line
    /// is resolved against ([`Self::from_override_key`]).
    pub const PRESETS: &'static [Model] = &[
        Model::Qwen3_8B_Q4,
        Model::Qwen3_8B_Q6,
        Model::Qwen3_14B_Q4,
        Model::Qwen3_14B_Q5,
        Model::Qwen3_14B_Q6,
        Model::Qwen3_30B_A3B_Q4,
        Model::Qwen3_30B_A3B_Q6,
        Model::Qwen36_35B_A3B_Q4,
        Model::Qwen38_FlashNext_Q4KO,
        Model::Qwen35_9B_Q6,
        Model::Qwen35_0_8B_Q8,
        Model::Qwen2_0_5B,
        Model::Hermes3_3B_Q6,
        Model::Hermes3_70B_Q4,
    ];

    /// The preset whose [`Self::override_key`] is `key` — the variant's own
    /// identifier, the same name `models.override.yaml` addresses it by.
    pub fn from_override_key(key: &str) -> Option<Model> {
        Self::PRESETS
            .iter()
            .find(|m| m.override_key().as_deref() == Some(key))
            .cloned()
    }

    /// Full specification for this model variant, as the repository declares it
    /// and before any local override.
    pub fn preset_spec(self) -> ModelSpec {
        match self {
            // Qwen3
            Model::Qwen3_8B_Q4 => qwen3::qwen3_8b_q4(),
            Model::Qwen3_8B_Q6 => qwen3::qwen3_8b_q6(),
            Model::Qwen3_14B_Q4 => qwen3::qwen3_14b_q4(),
            Model::Qwen3_14B_Q5 => qwen3::qwen3_14b_q5(),
            Model::Qwen3_14B_Q6 => qwen3::qwen3_14b_q6(),
            // Qwen3 MoE
            Model::Qwen36_35B_A3B_Q4 => qwen36_moe::qwen36_35b_a3b_q4(),
            Model::Qwen36_35B_A3B_AntiLoop_StyleTune => {
                qwen36_moe::qwen36_35b_a3b_antiloop_styletune()
            }
            Model::Qwen38_FlashNext_Q4KO => qwen38_flash_next::qwen38_flash_next_q4ko(),
            Model::Qwen35_9B_Q6 => qwen35_dense::qwen35_9b_q6(),
            Model::Qwen35_0_8B_Q8 => qwen35_dense::qwen35_0_8b_q8(),
            Model::Qwen3_30B_A3B_Q4 => qwen3_moe::qwen3_30b_a3b_q4(),
            Model::Qwen3_30B_A3B_Q6 => qwen3_moe::qwen3_30b_a3b_q6(),
            // Qwen2
            Model::Qwen2_0_5B => qwen2::qwen2_0_5b(),
            // Hermes-3 / Llama
            Model::Hermes3_3B_Q6 => hermes3::hermes3_3b_q6(),
            Model::Hermes3_70B_Q4 => hermes3::hermes3_70b_q4(),
            // Custom
            Model::Custom(spec) => spec,
        }
    }

    /// Create a [`ModelBuilder`] pre-loaded with this model's defaults.
    ///
    /// The builder lets you override sampling, file paths, sequence length,
    /// and other settings before calling [`.engine()`](ModelBuilder::engine).
    pub fn builder(self) -> ModelBuilder {
        ModelBuilder::from_spec(self.spec())
    }

    // ── Convenience shortcuts (delegate to a fresh builder) ────────────

    /// Shortcut: `self.builder().engine(device)`.
    ///
    /// Downloads from HuggingFace (requires `hub` feature) or returns an
    /// error if local paths are not configured.
    pub fn engine(self, device: &candle::Device) -> crate::Result<crate::ConversationEngine> {
        self.builder().engine(device)
    }

    /// Shortcut: `self.builder().conversation_config()`.
    pub fn conversation_config(self) -> SequenceConfig {
        self.builder().conversation_config()
    }

    /// Shortcut: `self.builder().system_prompt()`.
    pub fn default_system_prompt(self) -> String {
        self.builder().format_system_prompt()
    }

    /// Load a tokenizer from a local `tokenizer.json` file.
    pub fn load_tokenizer(path: &Path) -> crate::Result<tokenizers::Tokenizer> {
        tokenizers::Tokenizer::from_file(path)
            .map_err(|e| ConversationError::Tokenizer(e.to_string()))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Display
// ────────────────────────────────────────────────────────────────────────────

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Model::Custom(spec) => {
                let name = spec
                    .model_filename
                    .strip_suffix(".gguf")
                    .unwrap_or(&spec.model_filename);
                write!(f, "Custom({name})")
            }
            other => {
                let spec = other.clone().spec();
                let name = spec
                    .model_filename
                    .strip_suffix(".gguf")
                    .unwrap_or(&spec.model_filename);
                write!(f, "{name}")
            }
        }
    }
}

impl std::fmt::Display for ModelArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelArch::Qwen3 => write!(f, "Qwen3"),
            ModelArch::Qwen3Moe => write!(f, "Qwen3Moe"),
            ModelArch::Qwen2 => write!(f, "Qwen2"),
            ModelArch::Llama => write!(f, "Llama"),
            ModelArch::DeepSeekV4 => write!(f, "DeepSeekV4"),
            ModelArch::Qwen4Exp => write!(f, "Qwen4Exp"),
            ModelArch::Qwen35Hybrid => write!(f, "Qwen35Hybrid"),
            ModelArch::Qwen35Dense => write!(f, "Qwen35Dense"),
        }
    }
}
