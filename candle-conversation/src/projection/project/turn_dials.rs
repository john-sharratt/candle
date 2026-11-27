//! The three per-turn dials that decide how much a turn retrieves, carried on
//! its [`SelectionState`] beside the thinking and tools dials, so a caller
//! chooses them per request and every stage that reads the selection — the
//! slot fill, the reprojection policy, the opening's seeding — reads the same
//! answer.
//!
//! - **`projection`** — what the slot takes beyond the conversation's own
//!   turns: [`ProjectionLevel::Full`] every kind, [`ProjectionLevel::Limited`]
//!   the journals alone, [`ProjectionLevel::Minimal`] nothing (the tool
//!   catalog, which the system prompt carries under the tools dial, is all
//!   that is left; with tools off, nothing is).
//! - **`seed`** — whether a free knowledge base places its mandatory files
//!   (`files::mandatory`).
//! - **`reprojection`** — whether the decode re-selects inside its think
//!   block ([`ReprojectionMode::Full`]), holds the block and re-selects only
//!   at its close and through the answer ([`ReprojectionMode::Held`]), or
//!   selects once at the opening and never again
//!   ([`ReprojectionMode::Opening`]).
//!
//! An unset dial is the production default: full projection, seeds on, and
//! the conversation's configured reprojection.

use super::section_tree::{OptionalState, SelectionState};
use crate::projection::fill::Kind;

/// Selector for [`ProjectionLevel`].
pub const PROJECTION_SELECTOR: &str = "projection";
/// Selector for the seed dial: [`OptionalState::Absent`] places no mandatory
/// file.
pub const SEED_SELECTOR: &str = "seed";
/// Selector for [`ReprojectionMode`].
pub const REPROJECTION_SELECTOR: &str = "reprojection";

/// How much of the retrieved memory a turn's slot takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProjectionLevel {
    /// Every kind provenance selects from.
    #[default]
    Full,
    /// The journals alone.
    Limited,
    /// None of the retrieved memory.
    Minimal,
}

impl ProjectionLevel {
    /// The option id this level is selected by.
    pub const fn as_id(self) -> &'static str {
        match self {
            ProjectionLevel::Full => "full",
            ProjectionLevel::Limited => "limited",
            ProjectionLevel::Minimal => "minimal",
        }
    }

    /// The level `id` names, if it names one.
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "full" => Some(ProjectionLevel::Full),
            "limited" => Some(ProjectionLevel::Limited),
            "minimal" => Some(ProjectionLevel::Minimal),
            _ => None,
        }
    }

    /// Whether a unit of `kind` may be placed at this level. The
    /// conversation's own rounds are [`Kind::Dialogue`] and always placed;
    /// what a lower level leaves out is the retrieved memory.
    pub fn places(self, kind: Kind) -> bool {
        match self {
            ProjectionLevel::Full => true,
            ProjectionLevel::Limited => matches!(kind, Kind::Journal | Kind::Dialogue),
            ProjectionLevel::Minimal => kind == Kind::Dialogue,
        }
    }

    /// Whether the conversation's inherited lineage — the priming chain of
    /// repository roots and guides it was forked from — is placed.
    pub fn places_lineage(self) -> bool {
        self == ProjectionLevel::Full
    }
}

/// When a decode re-selects its working set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReprojectionMode {
    /// At every boundary, inside the think block as in the answer.
    Full,
    /// Not inside the think block: once at its close, and through the answer.
    Held,
    /// Only at the turn's opening: the question's first reading and opening
    /// boundary choose the working set, and no decode boundary re-selects it.
    Opening,
}

impl ReprojectionMode {
    /// The option id this mode is selected by.
    pub const fn as_id(self) -> &'static str {
        match self {
            ReprojectionMode::Full => "full",
            ReprojectionMode::Held => "held",
            ReprojectionMode::Opening => "opening",
        }
    }

    /// The mode `id` names, if it names one.
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "full" => Some(ReprojectionMode::Full),
            "held" => Some(ReprojectionMode::Held),
            "opening" => Some(ReprojectionMode::Opening),
            _ => None,
        }
    }
}

impl SelectionState {
    /// The turn's projection level: [`ProjectionLevel::Full`] unless the dial
    /// names another.
    pub fn projection_level(&self) -> ProjectionLevel {
        self.get(PROJECTION_SELECTOR)
            .and_then(ProjectionLevel::from_id)
            .unwrap_or_default()
    }

    /// Set the projection dial.
    pub fn set_projection_level(&mut self, level: ProjectionLevel) -> &mut Self {
        self.select(PROJECTION_SELECTOR, level.as_id())
    }

    /// Whether the knowledge base places its mandatory files: yes unless the
    /// dial is [`OptionalState::Absent`].
    pub fn seeds(&self) -> bool {
        self.optional(SEED_SELECTOR) != Some(OptionalState::Absent)
    }

    /// Set the seed dial.
    pub fn set_seeds(&mut self, on: bool) -> &mut Self {
        self.set_optional(
            SEED_SELECTOR,
            if on {
                OptionalState::Present
            } else {
                OptionalState::Absent
            },
        )
    }

    /// The reprojection the turn asks for, `None` for the conversation's
    /// configured one.
    pub fn reprojection_mode(&self) -> Option<ReprojectionMode> {
        self.get(REPROJECTION_SELECTOR)
            .and_then(ReprojectionMode::from_id)
    }

    /// Set the reprojection dial.
    pub fn set_reprojection_mode(&mut self, mode: ReprojectionMode) -> &mut Self {
        self.select(REPROJECTION_SELECTOR, mode.as_id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An empty selection is the production default.
    #[test]
    fn unset_dials_read_as_production() {
        let s = SelectionState::new();
        assert_eq!(s.projection_level(), ProjectionLevel::Full);
        assert!(s.seeds());
        assert_eq!(s.reprojection_mode(), None);
    }

    #[test]
    fn each_dial_reads_back_what_was_set() {
        let mut s = SelectionState::new();
        s.set_projection_level(ProjectionLevel::Limited)
            .set_seeds(false)
            .set_reprojection_mode(ReprojectionMode::Opening);
        assert_eq!(s.projection_level(), ProjectionLevel::Limited);
        assert!(!s.seeds());
        assert_eq!(s.reprojection_mode(), Some(ReprojectionMode::Opening));
        assert_eq!(s.get(PROJECTION_SELECTOR), Some("limited"));
        assert_eq!(s.get(SEED_SELECTOR), Some("absent"));
        assert_eq!(s.get(REPROJECTION_SELECTOR), Some("opening"));
    }

    /// An id no level names falls back to the default rather than to a
    /// lower level.
    #[test]
    fn an_unknown_id_is_the_default() {
        let mut s = SelectionState::new();
        s.select(PROJECTION_SELECTOR, "some");
        s.select(REPROJECTION_SELECTOR, "never");
        assert_eq!(s.projection_level(), ProjectionLevel::Full);
        assert_eq!(s.reprojection_mode(), None);
    }

    #[test]
    fn levels_place_the_kinds_they_name() {
        let every = [
            Kind::Reading,
            Kind::Folder,
            Kind::Guide,
            Kind::Journal,
            Kind::Dialogue,
            Kind::Layer(0),
        ];
        assert!(every.iter().all(|&k| ProjectionLevel::Full.places(k)));
        let limited: Vec<Kind> = every
            .into_iter()
            .filter(|&k| ProjectionLevel::Limited.places(k))
            .collect();
        assert_eq!(limited, vec![Kind::Journal, Kind::Dialogue]);
        let minimal: Vec<Kind> = every
            .into_iter()
            .filter(|&k| ProjectionLevel::Minimal.places(k))
            .collect();
        assert_eq!(minimal, vec![Kind::Dialogue]);
        assert!(ProjectionLevel::Full.places_lineage());
        assert!(!ProjectionLevel::Limited.places_lineage());
        assert!(!ProjectionLevel::Minimal.places_lineage());
    }

    #[test]
    fn ids_round_trip() {
        for l in [
            ProjectionLevel::Full,
            ProjectionLevel::Limited,
            ProjectionLevel::Minimal,
        ] {
            assert_eq!(ProjectionLevel::from_id(l.as_id()), Some(l));
        }
        for m in [
            ReprojectionMode::Full,
            ReprojectionMode::Held,
            ReprojectionMode::Opening,
        ] {
            assert_eq!(ReprojectionMode::from_id(m.as_id()), Some(m));
        }
    }
}
