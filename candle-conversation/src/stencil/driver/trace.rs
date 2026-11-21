//! The tally of the path a walk took through its tree, reported when the walk
//! finishes so that a malformed call can be diagnosed.
//!
//! Each step of the walk is counted by its kind: static runs prefilled whole, masked
//! choices, free decodes inside values, exit tokens healed, close signals dropped,
//! delimiters dropped at the end of a lookahead value, and out-of-grammar tokens that
//! forced the bail failsafe. Dropped close signals cover both a thinking block's close
//! and an EOS sampled inside a tool-call value, so that one counter totals both.

/// A running tally of the path a [`StencilDriver`] walk took — emitted as the
/// steering's finish trace so a malformed call is diagnosable at a glance
/// (e.g. `bailed=true`, or `free_tokens=0` where a value was expected).
#[derive(Debug, Clone, Copy, Default)]
pub struct PathStats {
    /// Static runs prefilled atomically (structural scaffold: envelope, keys,
    /// punctuation).
    pub prefills: u32,
    /// Tokens across those prefilled runs.
    pub prefill_tokens: u32,
    /// Masked branch decodes (constrained choices: tool name, enum, bool,
    /// optional-field gates).
    pub branch_tokens: u32,
    /// Free decodes inside value spans (the model writing argument content).
    pub free_tokens: u32,
    /// Exit-token heals applied (the model merged a value's closing char with
    /// the next delimiter and was steered back).
    pub heals: u32,
    /// Close signals intercepted and dropped inside a span, of either kind.
    ///
    /// Two things land here and they are the same event to the driver: a
    /// thinking block's `</think>`, dropped so the successor can re-steer the
    /// model back into reasoning with a continuation phrase; and an **EOS
    /// sampled inside a tool-call value**, dropped so it never reaches the
    /// sequence and never seals the turn while the tree still has structure to
    /// emit.
    ///
    /// It was `think_continuations` and counted only the first, which stopped
    /// being true when EOS interception was extended to every span — the
    /// counter kept its name and quietly began totalling both.
    pub intercepted_closes: u32,
    /// Delimiters dropped at the end of a lookahead value because the grammar
    /// does not continue with them (a `]` where the element's `}` comes next);
    /// the grammar wrote the structure in their place.
    pub dropped_delimiters: u32,
    /// Free-span tokens committed as a repair instead of as written: a
    /// character escaped inside a string, or a malformed value ended and
    /// completed.
    pub repairs: u32,
    /// An out-of-grammar token escaped the mask and forced the bail failsafe.
    pub bailed: bool,
}
