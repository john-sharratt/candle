//! The selector ids and option states a section tree uses to choose a branch.
//!
//! The string ids that select a branch are constants here, so the schema, the
//! dial mapping and the reads cannot drift apart. `OptionalState` is the typed
//! form of an optional node's present and absent ids. `SelectionState` is the
//! runtime override that maps each selector to the option or the member names it
//! chose, and any selector it does not set falls back to the schema default.

use super::*;

/// The section-tree selector id the engine reads to drive per-turn thinking
/// suppression.  The dialogue schema authors an `optional` node with this id;
/// [`OptionalState::Present`] suppresses reasoning for the turn, [`Absent`]
/// enables it.  The contract lives here (not as scattered `"no_think"` literals)
/// so the schema author, the dial mapping, and the suppression read can't drift.
///
/// [`Absent`]: OptionalState::Absent
pub const NO_THINK_SELECTOR: &str = "no_think";

/// Selector for the `tools_enabled` optional_group that gates the WHOLE tool block
/// (overview, `<tools>` markers, catalog, summary). `Absent` omits it entirely —
/// what a tools-off layer (e.g. `code_reading`) sets so the projection never
/// selects (or elevates) any tool section. Kept here so the string can't drift
/// between the schema, chat.rs, and the ingest path.
pub const TOOLS_ENABLED_SELECTOR: &str = "tools_enabled";

/// Selector that force-pins a single member of a belief-driven (`TopK`) tool
/// catalog. When set to a tool's section name the `TopK` arm emits **exactly**
/// that member and skips belief selection entirely — score-independent, the same
/// by-name pick `SelectionRule::Named` does, but available on the production
/// `tools` collection without switching its rule. The code_read summary path
/// sets it to `file_read` so the prefilled `read_file` tool_call/response is
/// backed by a coherent, present tool definition (tools ON, one tool, forced).
/// Unset ⇒ ordinary belief selection. Kept here so the string can't drift.
pub const FORCE_TOOL_SELECTOR: &str = "force_tool";

/// Selector marking a projection for a **tool round** — a turn whose user
/// message is the results of the calls the turn before it made.
/// [`OptionalState::Present`] leaves out every non-target layer that declares
/// `in_tool_rounds: false` (see [`LayerSchema::in_tool_rounds`]); unset or
/// [`OptionalState::Absent`] is an ordinary turn. Kept here so the string can't
/// drift between the host that sets it and the assembly that reads it.
pub const TOOL_ROUND_SELECTOR: &str = "tool_round";

/// Separator between member names in a [`FORCE_TOOL_SELECTOR`] value. A pin may
/// name several tools when one ingest turn prefills calls to more than one.
pub const FORCE_TOOL_SEPARATOR: char = ',';

/// The two states of an `optional` section-tree node: whether its content is
/// projected (`Present`) or omitted (`Absent`).
///
/// Unlike a selector's *option* ids (which are authored in YAML and so are
/// inherently dynamic strings), these two ids are a fixed convention the
/// `optional` lowering synthesizes — a closed set the engine owns.  This enum is
/// the typed form; [`Self::as_id`] / [`Self::from_id`] are the only place the
/// `"present"` / `"absent"` strings are defined, so a typo can't silently slip
/// past the compiler and fall through to a default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionalState {
    Present,
    Absent,
}

impl OptionalState {
    /// The synthesized option id this state serializes to in a [`SelectionState`].
    pub const fn as_id(self) -> &'static str {
        match self {
            OptionalState::Present => "present",
            OptionalState::Absent => "absent",
        }
    }

    /// Parse an `optional` node's option id back to its typed state; `None` for
    /// any id that isn't one of the two synthesized optional ids.
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "present" => Some(OptionalState::Present),
            "absent" => Some(OptionalState::Absent),
            _ => None,
        }
    }
}

/// Runtime override of the selectors one projection resolves.
///
/// Maps a selector id to what it chose: a section-tree dimension's option id,
/// or the names of the members a [`SelectionRule::Named`] collection shows. Any
/// selector absent from the map falls back to its authored default, so an empty
/// state reproduces the schema defaults exactly.
///
/// **A selector holds a set.** A dimension chooses one option and most callers
/// name one member — [`Self::select`] — but a collection whose members come and
/// go with the state of the world has to show several at once, chosen by the
/// caller rather than scored: a character's acts are the ones its body can do
/// *here*, which is a fact the caller has and provenance does not.
/// [`Self::select_all`] sets such a set; [`Self::get`] reads its first name,
/// which is the whole of it for every single-valued selector.
///
/// Selector and option ids are `String`s by necessity — they are authored in
/// the projection YAML, so the override map is data-driven and can't be a closed
/// Rust type.  For the engine's own closed conventions (the `optional`
/// present/absent state), use the typed [`Self::optional`] / [`Self::set_optional`]
/// accessors rather than bare string literals.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SelectionState {
    pub(super) chosen: HashMap<String, Vec<String>>,
    /// Sealed sections to project right after a catalog summary, by summary.
    pub(super) summary_notes: HashMap<SectionId, Vec<SectionId>>,
    /// The members a played-back projection selected, by name, in place of
    /// any selection (`super::script`).
    pub(super) scripted_members: Option<std::sync::Arc<std::collections::HashSet<String>>>,
    /// Members no collection projects, by name: tools a round's grammar holds
    /// out, so the prompt never shows a signature the turn cannot call.
    pub(super) withheld_members: BTreeSet<String>,
}

impl SelectionState {
    /// Project none of `names` from any collection, mandatory or not — the
    /// tools a round's grammar holds out, so what the prompt shows is what the
    /// turn can call. Replaces any earlier set.
    pub fn withhold_members<'a>(&mut self, names: impl IntoIterator<Item = &'a str>) -> &mut Self {
        self.withheld_members = names.into_iter().map(str::to_string).collect();
        self
    }

    /// Whether `name` is held out of every collection.
    pub(super) fn is_withheld(&self, name: &str) -> bool {
        self.withheld_members.contains(name)
    }

    /// Select exactly `members` from every collection, as a recording did.
    pub(in super::super) fn script_members(
        &mut self,
        members: std::sync::Arc<std::collections::HashSet<String>>,
    ) -> &mut Self {
        self.scripted_members = Some(members);
        self
    }

    /// Project `notes`, sealed sections, right after the catalog summary
    /// `summary`, where a turn reads what the listing means for it now.
    /// Replaces any earlier notes on it.
    pub fn note_summary(&mut self, summary: SectionId, notes: Vec<SectionId>) -> &mut Self {
        self.summary_notes.insert(summary, notes);
        self
    }

    /// The notes to project after `summary`.
    pub(super) fn summary_notes(&self, summary: SectionId) -> &[SectionId] {
        self.summary_notes
            .get(&summary)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// An empty selection — every selector falls back to its authored default.
    pub fn new() -> Self {
        Self::default()
    }

    /// Select a selector's option by id.  Returns `&mut self` for chaining.
    pub fn select(&mut self, selector: impl Into<String>, option: impl Into<String>) -> &mut Self {
        self.chosen.insert(selector.into(), vec![option.into()]);
        self
    }

    /// Choose several members at once — a [`SelectionRule::Named`] collection
    /// emits them in its own declaration order, whatever order they are named
    /// in. An empty set chooses nothing. Replaces whatever the selector held.
    pub fn select_all<S: Into<String>>(
        &mut self,
        selector: impl Into<String>,
        names: impl IntoIterator<Item = S>,
    ) -> &mut Self {
        self.chosen
            .insert(selector.into(), names.into_iter().map(Into::into).collect());
        self
    }

    /// Every name a selector holds — empty when it is unset.
    pub fn members(&self, selector: &str) -> &[String] {
        self.chosen.get(selector).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Set an `optional` selector to a typed [`OptionalState`].  Returns
    /// `&mut self` for chaining.  Prefer this over `select(id, "present")` so the
    /// present/absent ids stay in one place.
    pub fn set_optional(&mut self, selector: impl Into<String>, state: OptionalState) -> &mut Self {
        self.select(selector, state.as_id())
    }

    /// The chosen option id for a selector — the first of its set — if one was
    /// set.
    pub fn get(&self, selector: &str) -> Option<&str> {
        self.members(selector).first().map(String::as_str)
    }

    /// The typed state of an `optional` selector.  `None` when the selector is
    /// unset (the caller falls back to its schema/config default) or resolves to
    /// an id that isn't one of the optional present/absent ids.
    pub fn optional(&self, selector: &str) -> Option<OptionalState> {
        self.get(selector).and_then(OptionalState::from_id)
    }
}
