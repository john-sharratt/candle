//! Per-scope provenance score normalization.
//!
//! Raw late-fusion scores are not comparable across candidates — a generic
//! "stopword" cluster (the repo root, whose listing names every crate) scores
//! high against almost any probe, drowning a specific target that is the genuine
//! best *relative* match. This module rescales each candidate's raw score against
//! its own learned **hit level** — the score it reaches when it *is* the answer —
//! so a full hit lands at ~1000 and a decode lock-on rides above it, on a common
//! 0–1000 band that makes selection thresholds uniform across scopes.
//!
//! See `docs/provenance_score_normalization.md`. The module is deliberately
//! std-only: it knows nothing about the schema, substrate, or provenance types.
//! The scan maps its members (a repo_map path tag, a tool name) to string
//! [`ChildKey`]s and its budget node to a [`ScopeKey`]; everything here is
//! `String`/`f32` arithmetic, so it unit-tests in isolation.
//!
//! Two operations, both driven from the scan:
//! - [`NormalizationCache::normalize`] — read path, every reprojection: rescale a
//!   scope's raw scores to 0–1000 for selection. Pure; never mutates.
//! - [`NormalizationCache::observe`] — write path, **once per turn at seal**: fold
//!   the turn's raw scores into each child's hit-level EWMA. The only writer.

mod cache;
mod hit_level;
mod scope;

#[cfg(test)]
mod tests;

pub use cache::NormalizationCache;

/// Identity of a **score-competition** budget scope (a turn group, a section
/// collection, or a sub-window). NOT a layer — the layer is token distribution,
/// not a score competition. Structured (not a formatted string) so it allocates
/// nothing on the scan hot path and so the cache can evict a group's stale scopes
/// (see [`NormalizationCache::observe`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScopeKey {
    /// A turn group's gallery on a specific timeline. A re-scan mints a new
    /// timeline, hence a new scope; the old one is evicted on the next observe.
    TurnGroup { group: u64, timeline: u64 },
    /// A section collection (the tool catalog); its gallery is stable, no timeline.
    Collection { group: u64, name: String },
    /// One PHASE of a section collection's gallery — the same competition scored
    /// over only the user / thinking / response region of each exemplar.
    ///
    /// A hit level is a denominator, so it is only meaningful over the
    /// distribution it was learned on. A live query is a *question*, and
    /// `docs/tool_selection_provenance_results.md` §20.3 measured the phases as
    /// carrying very different signal for tool routing — `user` 65.6 against
    /// `think+resp` 21.5. Folding all three into one level therefore normalizes
    /// the question against a denominator that is mostly not questions.
    CollectionPhase {
        group: u64,
        name: String,
        phase: Phase,
    },
    /// A sub-window within one turn (future — file-within-listing selection).
    SubWindow { turn: u64 },
}

impl ScopeKey {
    pub fn turn_group(group: u64, timeline: u64) -> Self {
        ScopeKey::TurnGroup { group, timeline }
    }
    pub fn collection(group: u64, name: impl Into<String>) -> Self {
        ScopeKey::Collection {
            group,
            name: name.into(),
        }
    }
    pub fn collection_phase(group: u64, name: impl Into<String>, phase: Phase) -> Self {
        ScopeKey::CollectionPhase {
            group,
            name: name.into(),
            phase,
        }
    }
    pub fn sub_window(turn: u64) -> Self {
        ScopeKey::SubWindow { turn }
    }
}

/// Which region of a turn a signature window covers.
///
/// A turn is `user question → <think>…</think> → assistant answer`, and those
/// regions do not carry the same retrieval signal. Measured for tool routing
/// (results doc §20.3): routing on `user` scores 65.6 and on `think+resp` 21.5 —
/// "the task description is what maps to the tool definition, while by the time
/// the model is emitting the `<tool_call>` JSON it has committed". Scoring and
/// normalizing per phase is that finding applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Phase {
    /// The user's message — the question being asked.
    User,
    /// The `<think>…</think>` reasoning block.
    Thinking,
    /// The assistant's answer body, including any emitted tool call.
    Response,
}

/// Stable identity of a candidate within a scope. A turn group keys by turn index
/// (allocation-free, and stable within a gallery version); a collection keys by
/// its member name (tool / section).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ChildKey {
    Turn(u64),
    Named(String),
}

impl ChildKey {
    pub fn turn(index: u64) -> Self {
        ChildKey::Turn(index)
    }
    pub fn named(name: impl Into<String>) -> Self {
        ChildKey::Named(name.into())
    }
}

/// Tuning constants for the hit-level normalizer. Defaults are the values §85
/// calibrated (`docs/provenance_score_normalization.md` §7).
#[derive(Clone, Copy, Debug)]
pub struct NormConfig {
    /// EWMA rate when a probe beats the current hit level (rise fast).
    pub alpha_up: f32,
    /// EWMA rate otherwise (decay slow) — keeps a rarely-queried child's level
    /// from collapsing between hits.
    pub alpha_dn: f32,
    /// Cold-start hit level for a child never yet observed.
    pub hit_prior: f32,
    /// Hard minimum on the denominator floor — stops a child whose level decayed
    /// low from amplifying a partial match.
    pub floor_min: f32,
    /// Percentile of a scope's current hit levels used as the denominator floor.
    pub floor_pctl: f32,
    /// Output scale: a hit at the hit level maps to this (1000 = full hit).
    pub scale: f32,
}

impl Default for NormConfig {
    fn default() -> Self {
        NormConfig {
            alpha_up: 0.30,
            alpha_dn: 0.02,
            hit_prior: 400.0,
            floor_min: 50.0,
            floor_pctl: 0.10,
            scale: 1000.0,
        }
    }
}
