//! The record of what provenance selected for a projection, with the belief
//! scores that seed the next projection.
//!
//! Each projection stamps the belief score of every section and turn it selected.
//! A turn is keyed by its full `(timeline, index)` identity, since a group holds
//! many conversations and an index alone would collide across them. Whether an
//! item qualified on its own belief, rather than entering scope through the
//! minimum-budget force-fill, is kept with it, so the early-decode carry protects
//! only a pick that showed real evidence. Serde defaults keep records written
//! before these fields loadable, so an older substrate opens without migration.

use super::*;

/// The belief confidence per selected section/turn at a projection, produced by
/// the online selection loop and stamped onto the [`ProjectionEvent`] so the next
/// reprojection can seed its belief from the prior event. Unscored ids read `0.0`.
#[derive(Debug, Clone, Default)]
pub struct SelectionScores {
    pub(super) sections: HashMap<SectionId, f32>,
    pub(super) turns: HashMap<TurnKey, f32>,
    /// Sections/turns that reached the selection threshold on their own belief
    /// rather than entering scope via the min-budget force-fill. Absent ⇒ not
    /// qualified.
    pub(super) section_qualified: HashSet<SectionId>,
    pub(super) turn_qualified: HashSet<TurnKey>,
}

impl SelectionScores {
    /// Belief score for a section, or `0.0` if it carried none.
    pub fn section(&self, id: SectionId) -> f32 {
        self.sections.get(&id).copied().unwrap_or(0.0)
    }

    /// Belief score for a turn, or `0.0` if it carried none. Keyed by the
    /// turn's full `(timeline, index)` identity — a group holds many
    /// conversations, so a bare index would collide across them.
    pub fn turn(&self, turn: TurnKey) -> f32 {
        self.turns.get(&turn).copied().unwrap_or(0.0)
    }

    /// Whether a section qualified on its own belief this turn.
    pub fn section_qualified(&self, id: SectionId) -> bool {
        self.section_qualified.contains(&id)
    }

    /// Whether a turn qualified on its own belief this turn.
    pub fn turn_qualified(&self, turn: TurnKey) -> bool {
        self.turn_qualified.contains(&turn)
    }

    /// Record a section's belief score and whether it qualified.
    pub fn set_section(&mut self, id: SectionId, score: f32, qualified: bool) {
        self.sections.insert(id, score);
        if qualified {
            self.section_qualified.insert(id);
        }
    }

    /// Record a turn's belief score and whether it qualified.
    pub fn set_turn(&mut self, turn: TurnKey, score: f32, qualified: bool) {
        self.turns.insert(turn, score);
        if qualified {
            self.turn_qualified.insert(turn);
        }
    }
}

/// Which category a materialized segment falls into. Drives the GUI bar-map
/// color and ordering (system leftmost, then section groups, then turns).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BucketKind {
    /// System-prompt instructions + structural framing.
    System,
    /// A named system-prompt collection (`code_read`, `repo_map`, `tools`).
    Section,
    /// Conversation turns selected by provenance.
    Turns,
}

impl BucketKind {
    /// Emission rank: system leftmost, section groups next, turns last.
    pub(super) fn rank(self) -> u8 {
        match self {
            BucketKind::System => 0,
            BucketKind::Section => 1,
            BucketKind::Turns => 2,
        }
    }
}

/// One system-prompt section (bare or inside a collection) with the tokens it
/// holds and whether provenance selected it for this projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectedSection {
    pub name: String,
    pub tokens: u32,
    pub selected: bool,
    /// The section's belief score at this projection — stamped so the next
    /// reprojection can seed its [`PriorBelief`](super::project::PriorBelief) from
    /// the prior event (RelLeak decay/reinforcement across a turn). `0.0` when the
    /// section carried no belief score.
    #[serde(default)]
    pub score: f32,
    /// Whether this section reached the selection threshold on its own belief at
    /// some point this turn, as opposed to entering scope through the min-budget
    /// force-fill. Carried into the next reprojection's
    /// [`PriorBelief`](super::project::PriorBelief) so the early-decode carry
    /// floor only ever protects a pick that showed real evidence.
    #[serde(default)]
    pub qualified: bool,
}

/// One item of the system prompt, in materialized (declaration) order. Carries
/// the structural template glue (role/block markers) alongside content so the
/// panel shows the *complete* prompt — including the bits that close the tool
/// block and the system block — not just the retrieved content sections.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SystemItem {
    /// A structural template segment (e.g. `tool_block_close`, `system_end`) —
    /// emitted verbatim as `Generated` glue. `content` is its authored marker
    /// text; shown as a glue row, never collapsible.
    Glue {
        name: String,
        content: String,
        tokens: u32,
    },
    /// A bare system-prompt content section (collapsible to its authored text).
    Section { name: String, tokens: u32 },
    /// A named collection with *every* member flagged selected/skipped — so the
    /// GUI can show what provenance picked AND skipped.
    Collection {
        name: String,
        sections: Vec<SelectedSection>,
        /// The collection's `member_glue`, prefilled BETWEEN consecutive
        /// selected members (never before the first) — so a consumer rebuilding
        /// the materialized text interleaves it exactly as the projection did.
        /// Empty when the collection declares none.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        member_glue: String,
        /// Tokens in one `member_glue` run; `0` when empty.
        #[serde(default)]
        member_glue_tokens: u32,
    },
}

/// A conversation turn provenance pulled into this projection's window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectedTurn {
    /// The schema layer this turn's group belongs to (`dialogue`, `code_reading`,
    /// …) — so the panel can surface every projected memory tier, not just the
    /// dialogue conversation.
    pub layer: String,
    pub group: String,
    pub index: u32,
    pub role: String,
    pub tokens: u32,
    /// Forest kind: `normal` for a raw conversation turn, or
    /// `summary_of_turns` / `summary_of_summaries` when a summary node stood in
    /// for the turns beneath it. This is the "did my context get a summary
    /// instead of the real turn?" signal — otherwise invisible once a node is
    /// materialized as a turn segment.
    #[serde(default = "default_turn_kind")]
    pub kind: TurnKind,
    /// Why this turn won its slot, when known: score-density origin (hard anchor /
    /// provenance / coverage / …) from the summary-forest path, or `recent` /
    /// `historical` from the rule-based path. `None` for the live user message.
    #[serde(default)]
    pub reason: Option<SelectionOrigin>,
    /// Whether this turn reached the selection threshold on its own belief,
    /// rather than entering scope through the min-budget force-fill. Gates the
    /// early-decode carry floor on the turn axis (mirrors
    /// [`SelectedSection::qualified`]).
    #[serde(default)]
    pub qualified: bool,
    /// The resolved source timeline (raw id) this turn belongs to — stamped from
    /// [`ResolvedTurn::timeline`], which the projection fixed unambiguously at
    /// selection time. A turn's identity is `(timeline, index)`; consumers resolve
    /// its body by this key directly, NEVER by re-deriving `group → timeline`
    /// (the shared substrate holds many conversations under one group, so that
    /// re-derivation is non-deterministic). `None` only for the live user message.
    #[serde(default)]
    pub timeline: Option<u64>,
    /// Whether provenance selected this turn for the projection. Always `true`
    /// for a turn that made it into the materialized segments; the field mirrors
    /// [`SelectedSection::selected`] so the belief carry generalizes across the
    /// section and turn axes. Serde-default keeps redo-log records back-compat.
    #[serde(default)]
    pub selected: bool,
    /// The turn's belief score at this projection — stamped so the next
    /// reprojection can seed its [`PriorBelief`](super::project::PriorBelief) turn
    /// group from the prior event (RelLeak decay/reinforcement across a turn).
    /// `0.0` when the turn carried no belief score (recency groups, live message).
    #[serde(default)]
    pub score: f32,
}

/// Serde default for [`SelectedTurn::kind`] on records written before the field
/// existed — a missing tag means a plain conversation turn.
pub(super) fn default_turn_kind() -> TurnKind {
    TurnKind::Normal
}

/// One piece of the materialized conversation, in injection order, exactly as
/// the assembler's `assemble_pieces` produced it — the single source of truth
/// for the inter-turn boundary glue. The panel renders the dialogue region from
/// this so the glue it shows is literally what the engine injected, not a
/// client-side reconstruction. The system prompt is covered separately by
/// [`ProjectionSelection::system`]; this is the conversation region only (and
/// excludes the live, still-decoding user turn).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MaterializedPiece {
    /// A glue island the assembler gap-fills verbatim — the inter-turn boundary
    /// markers (`user_start`, the `/no_think` soft-switch, `assistant_end`),
    /// merged into one island and decoded to text.
    Glue { text: String },
    /// A sealed turn (memory tier or dialogue), carrying its forest kind + why so
    /// the card can show summary-vs-raw in place.
    Turn { turn: SelectedTurn },
}
