//! Compile-time prompt constants, embedded from adjacent Markdown files.
//!
//! All prompts are embedded at compile time via [`include_str!`]. Changing
//! any `.md` file invalidates downstream KV caches and may alter model
//! behaviour — treat prompt files like code.

/// System prompt postfix injected when temporal markers are enabled.
///
/// Appended to the system prompt text (with a preceding newline) when
/// [`ConversationTreeConfig::temporal_markers_enabled`](crate::tree::ConversationTreeConfig) is true.
/// Teaches the model to interpret `[T-{days}.{seq}]` temporal markers.
///
/// # Why a compile-time constant?
///
/// 1. **Baked into the BF16 system prompt KV.** The system prompt — including
///    this postfix — is prefilled once and its KV is pinned in BF16 for the
///    lifetime of the conversation. Changing the postfix text at runtime would
///    invalidate that cached KV and require a full reprefill. Treating it as a
///    compile-time constant makes the invariant explicit: same binary = same
///    KV fingerprint.
///
/// 2. **Model "vocabulary".** The marker format is an agreement between the
///    code and the model's behaviour. If the description changed, the model
///    would misinterpret markers from existing conversations. Version-coupling
///    via the constant makes a format change a breaking change at compile time.
///
/// 3. **No configuration complexity.** Allowing the postfix to be overridden
///    would require plumbing it through every fork/patch roundtrip and every
///    serialized conversation, with no practical benefit — users cannot
///    fine-tune the model at conversation time anyway.
pub const TEMPORAL_MARKER_POSTFIX: &str = include_str!("temporal_marker.md");

/// System prompt for summarization inference.
///
/// Injected as the system prompt for the temporary scheduler slot used to
/// compress a window of turns into a `ConversationSegment`.
pub const SUMMARIZE_PROMPT: &str = include_str!("summarize.md");

/// System prompt for Daydream inference.
///
/// Used when attention on a cold node crosses the resonance threshold and a
/// short associative thought is generated in the background.
pub const DAYDREAM_PROMPT: &str = include_str!("daydream.md");

/// System prompt for Sleep inference.
///
/// Used during the end-of-day prospective sleep batch.
pub const SLEEP_PROMPT: &str = include_str!("sleep.md");

/// System prompt for Reason inference.
///
/// Used when the executive self-dialogue turn runs to produce an updated plan
/// that is then injected into Reality system prompts.
pub const REASON_PROMPT: &str = include_str!("reason.md");

// ── tree_gen pipeline prompts ─────────────────────────────────────────────────

/// Guide prompt: synthesise a period of the life timeline into background material.
///
/// Used once per named period by `tree_gen`. The user message contains the
/// period name and its timeline entries; the assistant produces a Life Story
/// section and a Cast section listing all named characters.
pub const GUIDE_SUMMARIZE_PERIOD_PROMPT: &str = include_str!("guide_summarize_period.md");

/// Guide prompt: plan a single day as an ordered waypoint list.
///
/// Used once per day by `tree_gen`. The user message contains DATE,
/// DESCRIPTION, YESTERDAY, and LAST_MONTH fields. The assistant produces
/// a numbered list of 5–15 concrete waypoints for the day.
pub const GUIDE_TODAY_PROMPT: &str = include_str!("guide_today.md");

/// Director prompt: narrate the next scene for the character.
///
/// Used as a single-shot inference per waypoint. The user message contains
/// LAST RESPONSE (character's previous output), WAYPOINT (current event),
/// and NEXT (upcoming event). Produces 2–6 sentences of literary prose in
/// second-person present tense.
pub const DIRECTOR_PROMPT: &str = include_str!("director.md");

// ── Narrator / Text-to-Input converter prompts ───────────────────────────────

/// Narrator waypoint system prompt for the text-to-inputs converter.
///
/// Used when `author` is `None` in `text_to_inputs`. Instructs the model to
/// convert third-person narrative prose into a JSON array of `Input` objects.
/// The model receives raw event text and must return only `[{...}, ...]` —
/// no fences, no explanation.
pub const NARRATOR_WAYPOINT_SYSTEM_PROMPT: &str = include_str!("narrator_waypoint.md");

/// Narrator author system prompt template for the text-to-inputs converter.
///
/// Used when `author` is `Some(name)` in `text_to_inputs`. Contains the
/// literal string `{author}` which is replaced at call time with the
/// character name. Instructs the model to resolve first-person references
/// to the named author character.
pub const NARRATOR_AUTHOR_SYSTEM_PROMPT_TEMPLATE: &str = include_str!("narrator_author.md");
