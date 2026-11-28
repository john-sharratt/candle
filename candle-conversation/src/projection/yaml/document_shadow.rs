//! The YAML shadow types for the system prompt, the summaries and the section
//! tree nodes, matching the keys an author writes.
//!
//! The shadow types match the YAML key names exactly and are converted to schema
//! types afterwards, so parsing and validation are kept apart from the schema the
//! engine uses. The system prompt takes two forms: a legacy flat list of sections,
//! and an ordered list of sections, collections and section trees. Tree nodes are
//! sections, optional toggles, selectors, nested trees and groups. Items flagged
//! `pin` are named by `pinned_names`. Hit counting and the free-tool-call setting
//! are opt-in.

use super::*;

// ── Shadow deserialization types ─────────────────────────────────────────────

#[derive(Deserialize)]
pub(super) struct YamlSchema {
    pub(super) layers: Vec<YamlLayer>,
    /// The single system prompt shared by every projection target. Required —
    /// must contain at least one section.
    pub(super) system_prompt: YamlSystemPrompt,
    /// Schema-wide default selection policy, inherited by any layer/collection/
    /// group that declares none. Absent → [`SelectionPolicy::default_policy`].
    #[serde(default)]
    pub(super) default_policy: Option<YamlPolicy>,
    /// Turn every repetition penalty off while a tool call is being emitted.
    ///
    /// **Opt-in, because it is right for one kind of caller and wrong for the
    /// other.** See [`super::schema::Schema::free_tool_calls_from_penalties`].
    #[serde(default)]
    pub(super) free_tool_calls_from_penalties: bool,
    /// How hits are counted and move a piece's score — see [`YamlHits`].
    #[serde(default)]
    pub(super) hits: Option<YamlHits>,
}

/// The top-level `hits:` block (`docs/hit_driven_projection.md` §6.1, §7.1).
/// Every field left out takes [`HitsConfig::default`]'s.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct YamlHits {
    /// Percentiles over the selecting layers.
    #[serde(default)]
    pub(super) layers: Option<Vec<u8>>,
    #[serde(default)]
    pub(super) gain: Option<f32>,
    #[serde(default)]
    pub(super) beta: Option<f32>,
}

impl YamlHits {
    pub(super) fn resolve(self) -> Result<HitsConfig, ConstructionError> {
        let d = HitsConfig::default();
        HitsConfig::new(
            self.layers
                .unwrap_or_else(|| d.layers.percentiles().to_vec()),
            self.gain.unwrap_or(d.gain),
            self.beta.unwrap_or(d.beta),
        )
        .map_err(ConstructionError::InvalidHits)
    }
}

#[derive(Deserialize, Default)]
pub(super) struct YamlSystemPrompt {
    /// Legacy shorthand: a flat list of always-emit sections.
    /// Translated into `SystemPromptItem::Section` entries on build.
    /// May be combined with `items` — sections appear first, followed
    /// by anything in `items`.
    #[serde(default)]
    pub(super) sections: Vec<YamlSection>,
    /// Modern interleaved form: ordered list of either single
    /// sections or collections (named buckets with their own selection
    /// rules).  Use this when authoring needs static framing
    /// interleaved with selectable subsets.
    #[serde(default)]
    pub(super) items: Vec<YamlSystemPromptItem>,
    /// The most positions the items flagged `pin: true` may cover together
    /// (`docs/hit_driven_projection.md` §5); required when any item is.
    #[serde(default)]
    pub(super) pin_tokens: usize,
}

/// The names of the items flagged `pin: true`, in declaration order: top-level
/// sections and collections, and section-tree nodes at any depth.
pub(super) fn pinned_names(items: &[YamlSystemPromptItem]) -> Vec<String> {
    fn nodes(out: &mut Vec<String>, ns: &[YamlTreeNode]) {
        for n in ns {
            match n {
                YamlTreeNode::Section { id, pin: true, .. }
                | YamlTreeNode::Optional { id, pin: true, .. }
                | YamlTreeNode::Selector { id, pin: true, .. } => out.push(id.clone()),
                YamlTreeNode::Collection {
                    name, pin: true, ..
                } => out.push(name.clone()),
                YamlTreeNode::SectionTree { nodes: inner } => nodes(out, inner),
                YamlTreeNode::OptionalGroup {
                    nodes: inner,
                    absent,
                    ..
                } => {
                    nodes(out, inner);
                    nodes(out, absent);
                }
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    for item in items {
        match item {
            YamlSystemPromptItem::Section { id, pin: true, .. } => out.push(id.clone()),
            YamlSystemPromptItem::Collection {
                name, pin: true, ..
            } => out.push(name.clone()),
            YamlSystemPromptItem::SectionTree { nodes: ns } => nodes(&mut out, ns),
            _ => {}
        }
    }
    out
}

/// A collection's `range:` — positions on a model whose native context is
/// 128K or more, and on a smaller one, and which end the members stand at.
///
/// ```yaml
/// range: { tokens: 8192, small_context: 2048, fill: bottom }
/// range: { tokens: 1024, most: 1 }
/// ```
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct YamlReservedRange {
    pub(super) tokens: usize,
    /// Positions on a model whose native context is below 128K. The full size
    /// when not given.
    #[serde(default)]
    pub(super) small_context: Option<usize>,
    /// `top` (default): the members end at the range's top, the hole ahead of
    /// them. `bottom`: they start at its bottom, the hole after them.
    #[serde(default)]
    pub(super) fill: Option<String>,
    /// The most members the range holds. As many as fit when not given.
    #[serde(default)]
    pub(super) most: Option<usize>,
}

#[derive(Deserialize)]
pub(super) struct YamlSection {
    pub(super) id: String,
    #[serde(default)]
    pub(super) content: String,
    /// Static fallback priority used as a tie-breaker inside a
    /// collection's selection.  Default 50.0.  Must be > 0.
    #[serde(default)]
    pub(super) priority: Option<f32>,
}

/// One compression framing + instruction pair. Both fields mandatory.
#[derive(Deserialize)]
pub(super) struct YamlCompressionPrompt {
    /// System-prompt framing for the compression pass.
    pub(super) system_prompt: String,
    /// Compression instruction, prefilled after the content.
    pub(super) user_prompt: String,
}

/// One tree-level of a layer's `summary:` — the question and answer halves plus
/// a shared decode cap. All fields mandatory (no serde default).
#[derive(Deserialize)]
pub(super) struct YamlTurnSummary {
    /// Hard decode-token ceiling for the compressed assistant half.
    pub(super) max_tokens: usize,
    /// Compresses the assistant-response half of a turn (`Role::Assistant`).
    /// Unused when `content: structural`.
    pub(super) assistant: YamlCompressionPrompt,
    /// How the user-message half (the scope) is derived from the children's:
    /// `union` (default), `line_spans`, or `directory`. Never decoded — see
    /// [`Scope`].
    #[serde(default)]
    pub(super) scope: String,
    /// How the assistant half is produced: `decode` (default) or `structural`.
    #[serde(default)]
    pub(super) content: String,
}

/// A layer's `summary:` block — required on every layer. `turns` (SummaryOfTurns)
/// is mandatory; `summaries` (SummaryOfSummaries) is optional and falls back to
/// `turns` when omitted.
#[derive(Deserialize)]
pub(super) struct YamlLayerSummary {
    pub(super) turns: YamlTurnSummary,
    #[serde(default)]
    pub(super) summaries: Option<YamlTurnSummary>,
}

/// One entry in a YAML `items:` list — tagged via the `kind` discriminator.
///
/// ```yaml
/// items:
///   - kind: section
///     id: frame
///     content: "..."
///   - kind: template
///     id: system_open
///     dialect: system_start
///   - kind: collection
///     name: tools
///     selection: { kind: top_k, k: 3 }
///     sections:
///       - id: tool_a
///         content: "..."
/// ```
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(super) enum YamlSystemPromptItem {
    Section {
        id: String,
        /// Every selecting query attends this section whole
        /// (`docs/hit_driven_projection.md` §5).
        #[serde(default)]
        pin: bool,
        #[serde(default)]
        content: String,
        #[serde(default)]
        priority: Option<f32>,
        /// Conditional emission: this section only emits at projection
        /// time if the named sibling Collection materialised ≥ 1 of its
        /// members. Ingested unconditionally either way.
        #[serde(default)]
        depends_on: Option<String>,
        /// Inverse gate: emit only when the named sibling Collection
        /// materialised **zero** members (e.g. a no-tools variant).
        #[serde(default)]
        depends_on_absent: Option<String>,
        /// Emit whenever the named sibling Collection is CONFIGURED — has ≥ 1
        /// member — regardless of what it selected this projection. The gate for
        /// prose that introduces a facility, where `depends_on` would delete the
        /// introduction on any turn provenance happened to select nothing.
        #[serde(default)]
        depends_on_configured: Option<String>,
        /// Inverse of `depends_on_configured`: emit only when the named sibling
        /// Collection has no members at all, i.e. the facility is switched off.
        #[serde(default)]
        depends_on_unconfigured: Option<String>,
        /// The least the section's slot takes (`docs/placed_section_layout.md`
        /// §3.1c): room for its text to change, or for a gated section to emit
        /// or not, without moving anything after it.
        #[serde(default)]
        min_tokens: usize,
    },
    /// References a [`DialectTemplate`] catalog entry by snake-case
    /// name.  Resolves to the dialect's string at build time and lands
    /// in a `SectionSchema` with `is_template = true`, which the
    /// projection assembler routes through live prefill rather than
    /// the substrate-backed sealed path.
    Template {
        id: String,
        /// Catalog reference, e.g. `system_start`, `tool_block_open`,
        /// `no_think_prefix`.
        dialect: String,
        /// Conditional emission gate — same semantics as
        /// [`Self::Section::depends_on`].
        #[serde(default)]
        depends_on: Option<String>,
        /// As [`Self::Section::min_tokens`].
        #[serde(default)]
        min_tokens: usize,
    },
    Collection {
        name: String,
        /// Every selecting query attends the collection's members whole, the
        /// newest first as far as `pin_tokens` allows.
        #[serde(default)]
        pin: bool,
        #[serde(default)]
        selection: YamlSelection,
        #[serde(default)]
        score_threshold: f32,
        /// Selection policy override; inherits the enclosing layer's when absent.
        /// Its budget/thresholds subsume `selection`/`score_threshold`.
        #[serde(default)]
        policy: Option<YamlPolicy>,
        #[serde(default)]
        sections: Vec<YamlSection>,
        /// Glue string emitted (live-tokenised) BETWEEN consecutive selected
        /// members at projection — a real structural token, not baked into any
        /// member's seal, so it survives provenance dropping a member.  Empty ⇒
        /// members concatenate directly.
        #[serde(default)]
        member_glue: Option<String>,
        /// Fallback section (by name) when selection is empty.
        #[serde(default)]
        default: Option<YamlSelectionDefault>,
        /// Concept B: mass-driven member-budget extension.
        #[serde(default)]
        budget_adaptive: Option<YamlMemberBudgetAdaptive>,
        /// The positions the collection always takes — its members at the top
        /// of them, a hole ahead (`docs/placed_section_layout.md` §3.1).
        #[serde(default)]
        range: Option<YamlReservedRange>,
        /// The positions its catalog summary and the turn's notes on it always
        /// take, the summary at their front and a hole after it
        /// (`docs/placed_section_layout.md` §3.1c). `fill` does not apply.
        #[serde(default)]
        summary_range: Option<YamlReservedRange>,
        /// The compression level its members are sealed at.
        #[serde(default)]
        kv_level: Option<u8>,
    },
    /// An ordered, individually-toggleable tree of sealed sections.  Each node
    /// is a `section` (mandatory), an `optional` (binary toggle), or a
    /// `selector` (pick one of N options).  See [`super::SectionTree`].
    ///
    /// ```yaml
    /// - kind: section_tree
    ///   nodes:
    ///     - kind: optional
    ///       id: no_think
    ///       dialect: no_think_prefix
    ///       default: present
    ///     - kind: section
    ///       id: frame
    ///       content: "You are ..."
    ///     - kind: selector
    ///       id: thinking_effort
    ///       default: balanced
    ///       options:
    ///         - id: off       content: "Answer directly."
    ///         - id: balanced  content: "Reason at a natural pace."
    /// ```
    SectionTree { nodes: Vec<YamlTreeNode> },
}

/// One node in a `kind: section_tree` item — tagged by `kind`.
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(super) enum YamlTreeNode {
    /// Mandatory single-content node — always emits its content.
    Section {
        id: String,
        /// Attended whole by every selecting query.
        #[serde(default)]
        pin: bool,
        #[serde(default)]
        content: String,
        /// Dialect template source (e.g. `no_think_prefix`); resolved to a
        /// string and sealed as content.
        #[serde(default)]
        dialect: Option<String>,
        /// When set to a collection name, this section is a PLACEHOLDER: it
        /// still seals + anchors the K/V below it, but at projection its own
        /// content is replaced by that collection's top-k selection.  The
        /// collection then suppresses its own emission (see
        /// `TreeCollection::deferred_projection`).
        #[serde(default)]
        inject_collection: Option<String>,
        /// The least the node's slot takes (`docs/placed_section_layout.md`
        /// §3.1c): room for its content to change without moving anything
        /// after it.
        #[serde(default)]
        min_tokens: usize,
    },
    /// Binary toggle — `present` (the content) vs `absent` (empty).  Its `id`
    /// is the selector id the runtime flips.
    Optional {
        id: String,
        /// Attended whole by every selecting query, when present.
        #[serde(default)]
        pin: bool,
        #[serde(default)]
        content: String,
        #[serde(default)]
        dialect: Option<String>,
        /// Authored default presence: `present` (default) or `absent`.
        #[serde(default)]
        default: Option<String>,
        /// As [`Self::Section::min_tokens`].
        #[serde(default)]
        min_tokens: usize,
    },
    /// N-way selector — emits the one option whose id the runtime selects (by
    /// the node `id`).  Exactly one option emits.
    Selector {
        id: String,
        /// Whichever option emits is attended whole by every selecting query.
        #[serde(default)]
        pin: bool,
        /// The option id used when the runtime supplies no selection.
        default: String,
        options: Vec<YamlTreeOption>,
        /// As [`Self::Section::min_tokens`].
        #[serde(default)]
        min_tokens: usize,
    },
    /// A nested `section_tree` — its nodes expand inline into the enclosing
    /// tree, so its selectors become dims declared **after** every node above
    /// this point.  Each inner node therefore seals against the full outer
    /// prefix and fans out across every outer selector assignment (it inherits
    /// the outer dims).  Used to keep a high-cardinality but cheap selector
    /// group (e.g. effort × length directives) *below* the expensive static
    /// content, so that content multiplies only by the outer selectors, never
    /// by the inner ones.
    ///
    /// ```yaml
    /// - kind: section_tree
    ///   nodes:
    ///     - kind: selector
    ///       id: thinking_effort
    ///       default: balanced
    ///       options: [ ... ]
    /// ```
    SectionTree { nodes: Vec<YamlTreeNode> },
    /// A collection embedded as a tree node — same config as a top-level
    /// `kind: collection`, but its members seal ×(outer selectors above it) and
    /// it is PREFIX-TRANSPARENT (nodes below it seal as if its variable members
    /// were not there).  Lets the expensive tool catalog sit under the
    /// `no_think` selector (×2) while the directive selectors below the
    /// catalog never multiply it.
    Collection {
        name: String,
        /// Every selecting query attends the members whole, the newest first
        /// as far as `pin_tokens` allows.
        #[serde(default)]
        pin: bool,
        #[serde(default)]
        selection: YamlSelection,
        #[serde(default)]
        score_threshold: f32,
        #[serde(default)]
        sections: Vec<YamlSection>,
        /// Glue string emitted (live-tokenised) BETWEEN consecutive selected
        /// members at projection — see the top-level `Collection::member_glue`.
        #[serde(default)]
        member_glue: Option<String>,
    },
    /// A togglable SUB-TREE — a binary `present`/`absent` dimension whose
    /// `present` value expands to a whole group of child nodes (sections,
    /// collections, even nested groups), and whose `absent` value emits nothing.
    /// Nodes BELOW the group seal ×(present/absent): the present branch carries
    /// the whole group in its prefix, the absent branch carries none of it.  Used
    /// to gate the entire tool block (overview, `<tools>`, the catalog + its
    /// inject placeholder + glue, `</tools>`) on whether tools are enabled, so a
    /// no-tools turn omits the block — markers included — rather than emitting an
    /// empty shell.  The runtime drives it by the group `id`, exactly like an
    /// `optional` toggle.
    OptionalGroup {
        id: String,
        /// Authored default presence: `present` (default) or `absent`.
        #[serde(default)]
        default: Option<String>,
        /// Present-branch sub-tree (emitted when the toggle is `present`).
        nodes: Vec<YamlTreeNode>,
        /// Optional ABSENT-branch sub-tree (emitted when the toggle is `absent`).
        /// Lets the two branches carry *different* content without a new
        /// dimension — e.g. a tools-aware vs tools-free grounding statement under
        /// the same `tools_enabled` toggle.  Empty ⇒ absent emits nothing.
        #[serde(default)]
        absent: Vec<YamlTreeNode>,
    },
}

/// One option of a `kind: selector` tree node.
#[derive(Deserialize)]
pub(super) struct YamlTreeOption {
    pub(super) id: String,
    #[serde(default)]
    pub(super) content: String,
    #[serde(default)]
    pub(super) dialect: Option<String>,
}
