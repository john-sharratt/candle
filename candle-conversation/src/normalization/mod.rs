//! Per-scope provenance score normalization.
//!
//! Raw late-fusion scores are not comparable across candidates — a generic
//! "stopword" cluster (the repo root, whose listing names every crate) scores
//! high against almost any probe, drowning a specific target that is the genuine
//! best *relative* match. This module reads each candidate's raw score against
//! its own learned **background** — the scores it reaches on probes that are not
//! about it — as tail surprise, so a loud candidate wins no more often by chance
//! than a quiet one, and puts that on a common band where a score beyond all but
//! [`NormConfig::full_hit_tail`] of the background lands at 1000.
//!
//! See `docs/provenance_score_normalization.md`. The module is deliberately
//! std-only: it knows nothing about the schema, substrate, or provenance types.
//! The scan maps its members (a repo_map path tag, a tool name) to string
//! [`ChildKey`]s and its budget node to a [`ScopeKey`]; everything here is
//! `String`/`f32` arithmetic, so it unit-tests in isolation.
//!
//! Two operations, both driven from the scan:
//! - [`NormalizationCache::normalize`] — read path, every reprojection: put a
//!   scope's raw scores on the band for selection. Pure; never mutates.
//! - [`NormalizationCache::observe`] — write path, once per turn at seal and in
//!   the warm-ups: fold a probe's raw scores into the children's backgrounds.
//!   The only writer.

mod background;
mod cache;
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
    /// A turn group's pooled documents — every one-exchange timeline of the
    /// group, competing as one gallery (`projection::belief_files::competitions`).
    /// One competition, so one scope: its children are the documents, keyed by
    /// timeline ([`ChildKey::Timeline`]), and its floor is taken across them.
    Pool { group: u64 },
    /// A section collection (the tool catalog); its gallery is stable, no timeline.
    Collection { group: u64, name: String },
    /// One PHASE of a section collection's gallery — the same competition scored
    /// over only the user / thinking / response region of each exemplar.
    ///
    /// A background is only meaningful over the distribution it was learned on.
    /// A live query is a *question*, and
    /// `docs/tool_selection_provenance_results.md` §20.3 measured the phases as
    /// carrying very different signal for tool routing — `user` 65.6 against
    /// `think+resp` 21.5. Kept for the offline phase analysis
    /// (`substrate_inspect --gallery-phase`); the engine scans whole turns.
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
    pub fn pool(group: u64) -> Self {
        ScopeKey::Pool { group }
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
    /// A pooled document, by its timeline's raw id — see [`ScopeKey::Pool`].
    Timeline(u64),
}

impl ChildKey {
    pub fn turn(index: u64) -> Self {
        ChildKey::Turn(index)
    }
    pub fn timeline(raw: u64) -> Self {
        ChildKey::Timeline(raw)
    }
    pub fn named(name: impl Into<String>) -> Self {
        ChildKey::Named(name.into())
    }
}

/// The background normalizer's constants (`docs/provenance_score_normalization.md`
/// §3).
#[derive(Clone, Copy, Debug)]
pub struct NormConfig {
    /// The band value of a full hit: a score beyond all but
    /// [`Self::full_hit_tail`] of the child's background.
    pub scale: f32,
    /// The share of a child's background a full hit is beyond: 0.01, so 1000 is
    /// a 1-in-100 score against the probes that are not about it, the median
    /// background score lands near 150, a 1-in-10 near 500, and a score past
    /// everything the background holds rides above 1000.
    pub full_hit_tail: f32,
    /// Samples a child's own background needs before a score is read against
    /// it; fewer, and the scope's pooled background stands in.
    pub min_background: u32,
    /// Reference probe length, in signature tokens, that every raw score is
    /// scaled to before it meets the band.
    ///
    /// **A raw provenance score is a SUM over probe tokens**, so it grows with
    /// the probe. A background is learned at seal from a capped tail probe,
    /// while the read path runs at every reprojection with whatever window is
    /// current — six tokens at submit, up to the cap mid-decode. Comparing a
    /// six-token sum with samples learned at 256 is not a smaller measurement of
    /// the same quantity, it is a different unit, and it was measured: the same
    /// `datetime` match scored 4,658 against a six-token query and ~82,850
    /// against an eighty-token one.
    ///
    /// Scaling both the read and the write path to this reference makes the band
    /// a RATE rather than a total, so a hit reads the same whatever the probe
    /// length. It also makes Concept F's `max(tail, question)` fusion mean
    /// something: those two probes differ in length by an order of magnitude, so
    /// on raw sums the longer one won structurally rather than on evidence.
    ///
    /// **Linear, and deliberately uncapped.** Score-per-token × reference length
    /// is the assumption-free reading of a sum; a cap would be a tuning constant
    /// invented without measurement. The cost is that a very short probe's
    /// variance is amplified with its signal, so if short-window noise starts
    /// winning selections, the answer is a measured ceiling here — not a guess.
    ///
    /// 256 = `reproject_max_probe_tokens`, the live probe cap and therefore the
    /// length the backgrounds are in practice learned at.
    pub probe_t_ref: usize,
}

impl Default for NormConfig {
    fn default() -> Self {
        NormConfig {
            scale: 1000.0,
            full_hit_tail: 0.01,
            min_background: 8,
            probe_t_ref: 256,
        }
    }
}
