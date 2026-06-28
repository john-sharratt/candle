//! YAML deserialisation for [`super::Schema`].
//!
//! # Pipeline
//!
//! ```text
//!   YAML text
//!     │  serde_yaml::from_str
//!     ▼
//!   YamlSchema (shadow types — match YAML key names exactly)
//!     │  build()
//!     │   ├─ assign SectionId(1..n)   in declaration order
//!     │   ├─ assign LayerId(1..n)     in declaration order
//!     │   ├─ assign GroupId(1..n)     globally (across all layers)
//!     │   ├─ build NameMaps (string → id) for caller name lookup
//!     │   └─ run parse-time validation (kinds, percentages, priorities…)
//!     ▼
//!   (Schema, NameMaps)   ←  consumed by Builder::from_yaml
//! ```
//!
//! # YAML grammar (informal)
//!
//! ```yaml
//! layers:
//!   - name: <string>
//!     description: <string>
//!     window: <usize>              # total turn-budget when this layer is the
//!                                  # projection target — distributed via flex
//!                                  # across all visible layers below it.
//!     depth_weights:               # optional; default 1.0/1.0/1.0 equal weights
//!       syntactic: <float>
//!       semantic: <float>
//!       pragmatic: <float>
//!     score_threshold: <float>     # default 0.0
//!     budget: { priority, min_percent, max_percent }   # this layer's flex
//!                                  # weight when some *other* layer is the
//!                                  # target (lower-than-target visibility).
//!
//!     system_prompt:               # REQUIRED — framing for THIS layer
//!       sections:                  # as the projection target.  Must contain
//!         - id: <string>           # at least one section.
//!           content: <string>      # sections emit in declaration order.
//!
//!     groups:
//!       - id: <string>
//!         selection:
//!           kind: always_visible | top_k | single | conversation
//!           k: <usize>                   # required for top_k
//!           recent: <usize>              # for conversation
//!           historical_top_k: <usize>    # for conversation
//!         score_threshold: <float>
//!         budget: { ... }              # in-layer group flex
//! ```
//!
//! Section names are scoped per-layer: `dialogue` and `bug_analysis` may
//! both declare a `frame` section. Group names are still globally unique.
//!
//! Defaults are crate-defined; YAML omitting a field gets the default
//! documented in [`super::schema`]. Parse errors surface as
//! [`ConstructionError::YamlParse`], structural errors as the more specific
//! variants.

use std::collections::HashMap;

use candle_transformers::models::dialect::{Dialect, DialectTemplate};
use serde::Deserialize;

use super::error::ConstructionError;
use super::ids::{CollectionId, GroupId, LayerId, SectionId};
use super::project::OptionalState;
use super::schema::{
    Budget, CompressionPrompt, DepthWeights, GroupSchema, GroupSummary, GroupSummaryStage,
    LayerSchema, LayerSummary, Schema, SectionCollection, SectionSchema, SectionTree,
    SelectionRule, SummaryMode, SystemPromptItem, SystemPromptSchema, TreeCollection, TreeDim,
    TreeNode, TreeOption, TreeVariant, TurnSummary,
};

/// Sequential SectionId allocator. Ids are globally unique across the whole
/// schema even though names are per-layer scoped.
struct SectionIdAlloc(u32);
impl SectionIdAlloc {
    fn next(&mut self) -> SectionId {
        self.0 += 1;
        SectionId::new(self.0)
    }
}

/// Sequential CollectionId allocator.  Ids are globally unique across the
/// whole schema; collection names are layer-scoped (analogous to
/// sections, not groups).
struct CollectionIdAlloc(u32);
impl CollectionIdAlloc {
    fn next(&mut self) -> CollectionId {
        self.0 += 1;
        CollectionId::new(self.0)
    }
}

// ── Name lookup maps produced alongside the Schema ───────────────────────────

/// Reverse maps from YAML string names to crate-assigned ids.
///
/// Built by [`from_yaml`] alongside the [`Schema`], stashed inside the
/// [`super::Builder`], and queried by `id_for_layer` / `id_for_group` /
/// `id_for_section_in`. Empty when the builder is constructed via
/// `from_schema` (no YAML names available).
///
/// Section names are **per-layer scoped** — the same section name may be
/// declared in multiple layers. The lookup map is keyed by `(LayerId,
/// name)`.
#[derive(Debug, Default, Clone)]
pub struct NameMaps {
    pub layer_names: HashMap<String, LayerId>,
    pub group_names: HashMap<String, GroupId>,
    pub section_names: HashMap<(LayerId, String), SectionId>,
    pub collection_names: HashMap<(LayerId, String), CollectionId>,
}

// ── Public entry-point ────────────────────────────────────────────────────────

/// Parse a YAML schema string. Runs parse-time validation; returns
/// `(Schema, NameMaps)` on success.
///
/// When `dialect` is `Some`, `kind: template` items resolve against it.  When
/// `None`, any `kind: template` item in the YAML produces
/// [`ConstructionError::DialectRequired`].
pub fn from_yaml(
    yaml: &str,
    dialect: Option<&Dialect>,
) -> Result<(Schema, NameMaps), ConstructionError> {
    let raw: YamlSchema =
        serde_yaml::from_str(yaml).map_err(|e| ConstructionError::YamlParse(e.to_string()))?;
    build(raw, dialect)
}

// ── Shadow deserialization types ─────────────────────────────────────────────

#[derive(Deserialize)]
struct YamlSchema {
    layers: Vec<YamlLayer>,
}

#[derive(Deserialize, Default)]
struct YamlSystemPrompt {
    /// Legacy shorthand: a flat list of always-emit sections.
    /// Translated into `SystemPromptItem::Section` entries on build.
    /// May be combined with `items` — sections appear first, followed
    /// by anything in `items`.
    #[serde(default)]
    sections: Vec<YamlSection>,
    /// Modern interleaved form: ordered list of either single
    /// sections or collections (named buckets with their own selection
    /// rules).  Use this when authoring needs static framing
    /// interleaved with selectable subsets.
    #[serde(default)]
    items: Vec<YamlSystemPromptItem>,
}

#[derive(Deserialize)]
struct YamlSection {
    id: String,
    #[serde(default)]
    content: String,
    /// Static fallback priority used as a tie-breaker inside a
    /// collection's selection.  Default 50.0.  Must be > 0.
    #[serde(default)]
    priority: Option<f32>,
}

/// One compression framing + instruction pair. Both fields mandatory.
#[derive(Deserialize)]
struct YamlCompressionPrompt {
    /// System-prompt framing for the compression pass.
    system_prompt: String,
    /// Compression instruction, prefilled after the content.
    user_prompt: String,
}

/// One tree-level of a layer's `summary:` — the question and answer halves plus
/// a shared decode cap. All fields mandatory (no serde default).
#[derive(Deserialize)]
struct YamlTurnSummary {
    /// Hard decode-token ceiling for each compressed half.
    max_tokens: usize,
    /// Compresses the user-message half of a turn (`Role::User`).
    user: YamlCompressionPrompt,
    /// Compresses the assistant-response half of a turn (`Role::Assistant`).
    assistant: YamlCompressionPrompt,
    /// `single_pass` (default) or `structural` — the validated structural
    /// pipeline. Only meaningful on a `summaries` level.
    #[serde(default)]
    mode: String,
}

/// A layer's `summary:` block — required on every layer. `turns` (SummaryOfTurns)
/// is mandatory; `summaries` (SummaryOfSummaries) is optional and falls back to
/// `turns` when omitted.
#[derive(Deserialize)]
struct YamlLayerSummary {
    turns: YamlTurnSummary,
    #[serde(default)]
    summaries: Option<YamlTurnSummary>,
}

/// A section group's `summary:` block — a two-stage categorize→assign workflow
/// (a group is a catalog, summarised by grouping its sections rather than by a
/// single compression pass). All fields mandatory.
#[derive(Deserialize)]
struct YamlGroupSummary {
    /// Sections per stage-2 assignment chunk.
    chunk: usize,
    /// Stage 1 — the model proposes the categories.
    categorize: YamlGroupSummaryStage,
    /// Stage 2 — assign each section to one of the categories.
    assign: YamlGroupSummaryStage,
}

/// One stage of a group summary: a system/user prompt pair + decode cap.
#[derive(Deserialize)]
struct YamlGroupSummaryStage {
    /// System-prompt framing for the pass.
    system_prompt: String,
    /// Instruction, prefilled after the content.
    user_prompt: String,
    /// Hard decode-token ceiling for this stage.
    max_tokens: usize,
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
enum YamlSystemPromptItem {
    Section {
        id: String,
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
    },
    Collection {
        name: String,
        #[serde(default)]
        selection: YamlSelection,
        #[serde(default)]
        score_threshold: f32,
        #[serde(default)]
        depth_weights: Option<YamlDepthWeights>,
        /// Required — how this section group is compressed (config only for now).
        summary: YamlGroupSummary,
        #[serde(default)]
        sections: Vec<YamlSection>,
        /// Glue string emitted (live-tokenised) BETWEEN consecutive selected
        /// members at projection — a real structural token, not baked into any
        /// member's seal, so it survives provenance dropping a member.  Empty ⇒
        /// members concatenate directly.
        #[serde(default)]
        member_glue: Option<String>,
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
enum YamlTreeNode {
    /// Mandatory single-content node — always emits its content.
    Section {
        id: String,
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
    },
    /// Binary toggle — `present` (the content) vs `absent` (empty).  Its `id`
    /// is the selector id the runtime flips.
    Optional {
        id: String,
        #[serde(default)]
        content: String,
        #[serde(default)]
        dialect: Option<String>,
        /// Authored default presence: `present` (default) or `absent`.
        #[serde(default)]
        default: Option<String>,
    },
    /// N-way selector — emits the one option whose id the runtime selects (by
    /// the node `id`).  Exactly one option emits.
    Selector {
        id: String,
        /// The option id used when the runtime supplies no selection.
        default: String,
        options: Vec<YamlTreeOption>,
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
        #[serde(default)]
        selection: YamlSelection,
        #[serde(default)]
        score_threshold: f32,
        #[serde(default)]
        depth_weights: Option<YamlDepthWeights>,
        summary: YamlGroupSummary,
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
struct YamlTreeOption {
    id: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    dialect: Option<String>,
}

#[derive(Deserialize)]
struct YamlLayer {
    name: String,
    #[serde(default)]
    description: String,
    /// Total turn-budget when this layer is the projection target.
    window: usize,
    #[serde(default)]
    score_threshold: f32,
    #[serde(default)]
    budget: YamlBudget,
    /// Required system prompt — every layer must declare its framing.
    /// Validated to contain at least one section at construction time.
    system_prompt: YamlSystemPrompt,
    /// Required — how this layer's turns are compressed across the summary tree
    /// (`turns` + optional `summaries`, each with question/answer halves).
    summary: YamlLayerSummary,
    groups: Vec<YamlGroup>,
    /// Optional weights for the three BDP signature depths.  Default is
    /// `(1.0, 1.0, 1.0)` — equal mean.
    #[serde(default)]
    depth_weights: Option<YamlDepthWeights>,
}

#[derive(Deserialize, Default)]
struct YamlDepthWeights {
    #[serde(default)]
    syntactic: Option<f32>,
    #[serde(default)]
    semantic: Option<f32>,
    #[serde(default)]
    pragmatic: Option<f32>,
}

#[derive(Deserialize)]
struct YamlGroup {
    id: String,
    #[serde(default)]
    selection: YamlSelection,
    #[serde(default)]
    score_threshold: f32,
    #[serde(default)]
    budget: YamlBudget,
}

#[derive(Deserialize, Default)]
struct YamlBudget {
    #[serde(default)]
    priority: Option<f32>,
    #[serde(default)]
    min_percent: Option<f32>,
    #[serde(default)]
    max_percent: Option<f32>,
}

#[derive(Deserialize, Default)]
struct YamlSelection {
    #[serde(default)]
    kind: String,
    #[serde(default)]
    k: Option<usize>,
    #[serde(default)]
    recent: Option<usize>,
    #[serde(default)]
    historical_top_k: Option<usize>,
}

// ── Conversion ────────────────────────────────────────────────────────────────

fn build(
    raw: YamlSchema,
    dialect: Option<&Dialect>,
) -> Result<(Schema, NameMaps), ConstructionError> {
    let mut maps = NameMaps::default();
    let mut section_alloc = SectionIdAlloc(0);
    let mut collection_alloc = CollectionIdAlloc(0);

    let mut layers = Vec::with_capacity(raw.layers.len());
    let mut global_group_counter: u32 = 0;

    for (li, yl) in raw.layers.iter().enumerate() {
        let lid = LayerId::new(li as u32 + 1);
        maps.layer_names.insert(yl.name.clone(), lid);

        let layer_budget = parse_budget(&yl.name, &yl.budget)?;

        // ── this layer's system_prompt items ─────────────────────────────────
        // Build the ordered Vec<SystemPromptItem> from the YAML.  Two
        // input forms are accepted:
        //   1. Legacy `sections:` — flat list of always-emit sections.
        //      Translated into a sequence of `SystemPromptItem::Section`
        //      entries at the head of `items`.
        //   2. Modern `items:` — interleaved list of `Section` and
        //      `Collection` entries.  Authored when ordering matters
        //      (e.g. tools_intro, then a `tools` collection, then
        //      tools_outro).
        // Section names must be unique across the whole layer, regardless
        // of whether they're top-level or inside a collection.  Collection
        // names must also be unique per layer.
        let mut items: Vec<SystemPromptItem> = Vec::new();
        let mut layer_section_names: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut layer_collection_names: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        // Legacy `sections:` shortcut — emit before any items.
        for s in &yl.system_prompt.sections {
            if !layer_section_names.insert(s.id.clone()) {
                return Err(ConstructionError::DuplicateSectionName(s.id.clone()));
            }
            let sid = section_alloc.next();
            maps.section_names.insert((lid, s.id.clone()), sid);
            let priority = s.priority.unwrap_or(50.0);
            validate_priority(&format!("{}/{}", yl.name, s.id), priority)?;
            items.push(SystemPromptItem::Section(SectionSchema {
                id: sid,
                name: s.id.clone(),
                content: s.content.clone(),
                priority,
                depends_on: None,
                depends_on_absent: None,
                is_template: false,
                template_tokens: None,
            }));
        }

        // Two-pass over `items:` so a Section's `depends_on: <collection>`
        // can reference a Collection declared later in the same layer.
        //
        // First pass — pre-allocate every Collection's id and stash its
        // name in `layer_collections` so the second pass can resolve
        // forward references. Duplicates are caught here.
        let mut layer_collections: std::collections::HashMap<String, super::ids::CollectionId> =
            std::collections::HashMap::new();
        for entry in &yl.system_prompt.items {
            match entry {
                YamlSystemPromptItem::Collection { name, .. } => {
                    register_collection_id(
                        name,
                        lid,
                        &mut layer_collection_names,
                        &mut layer_collections,
                        &mut maps,
                        &mut collection_alloc,
                    )?;
                }
                // Collections embedded as section-tree nodes also need their id
                // pre-allocated so `depends_on` references resolve.
                YamlSystemPromptItem::SectionTree { nodes } => {
                    register_tree_collection_ids(
                        nodes,
                        lid,
                        &mut layer_collection_names,
                        &mut layer_collections,
                        &mut maps,
                        &mut collection_alloc,
                    )?;
                }
                _ => {}
            }
        }

        // Second pass — actually build the schema items, resolving
        // every `depends_on` string against the pre-allocated
        // collection-name map.
        for entry in &yl.system_prompt.items {
            match entry {
                YamlSystemPromptItem::Section {
                    id,
                    content,
                    priority,
                    depends_on,
                    depends_on_absent,
                } => {
                    if !layer_section_names.insert(id.clone()) {
                        return Err(ConstructionError::DuplicateSectionName(id.clone()));
                    }
                    let sid = section_alloc.next();
                    maps.section_names.insert((lid, id.clone()), sid);
                    let pri = priority.unwrap_or(50.0);
                    validate_priority(&format!("{}/{}", yl.name, id), pri)?;
                    let resolve_dep = |name: &Option<String>| -> Result<_, ConstructionError> {
                        match name {
                            Some(name) => {
                                Ok(Some(*layer_collections.get(name).ok_or_else(|| {
                                    ConstructionError::UnknownCollection(format!(
                                        "{}: section '{}' depends_on unknown collection '{}'",
                                        yl.name, id, name,
                                    ))
                                })?))
                            }
                            None => Ok(None),
                        }
                    };
                    let depends_on_cid = resolve_dep(depends_on)?;
                    let depends_on_absent_cid = resolve_dep(depends_on_absent)?;
                    items.push(SystemPromptItem::Section(SectionSchema {
                        id: sid,
                        name: id.clone(),
                        content: content.clone(),
                        priority: pri,
                        depends_on: depends_on_cid,
                        depends_on_absent: depends_on_absent_cid,
                        is_template: false,
                        template_tokens: None,
                    }));
                }
                YamlSystemPromptItem::Template {
                    id,
                    dialect: dialect_name,
                    depends_on,
                } => {
                    // Dialect required; cannot resolve template content
                    // without one. Surface a schema-locatable error.
                    let dlct = dialect.ok_or_else(|| ConstructionError::DialectRequired {
                        item: format!("{}/{}", yl.name, id),
                    })?;
                    let template =
                        DialectTemplate::from_yaml_name(dialect_name).ok_or_else(|| {
                            ConstructionError::UnknownDialectTemplate {
                                item: format!("{}/{}", yl.name, id),
                                name: dialect_name.clone(),
                            }
                        })?;
                    let content = dlct.template(template);
                    // Empty-string templates (e.g. NoThinkPrefix for a
                    // dialect that doesn't suppress thinking) are dropped
                    // at build time — projection never sees a no-op
                    // segment.
                    if content.is_empty() {
                        continue;
                    }
                    if !layer_section_names.insert(id.clone()) {
                        return Err(ConstructionError::DuplicateSectionName(id.clone()));
                    }
                    let sid = section_alloc.next();
                    maps.section_names.insert((lid, id.clone()), sid);
                    let depends_on_cid = match depends_on {
                        Some(name) => Some(*layer_collections.get(name).ok_or_else(|| {
                            ConstructionError::UnknownCollection(format!(
                                "{}: template '{}' depends_on unknown collection '{}'",
                                yl.name, id, name,
                            ))
                        })?),
                        None => None,
                    };
                    items.push(SystemPromptItem::Section(SectionSchema {
                        id: sid,
                        name: id.clone(),
                        content: content.to_string(),
                        priority: 50.0,
                        depends_on: depends_on_cid,
                        depends_on_absent: None,
                        is_template: true,
                        template_tokens: None,
                    }));
                }
                YamlSystemPromptItem::Collection {
                    name,
                    selection,
                    score_threshold,
                    depth_weights: coll_depth_weights_yaml,
                    summary,
                    sections,
                    member_glue,
                } => {
                    let cid = *layer_collections
                        .get(name)
                        .expect("first-pass pre-allocated every collection id");
                    let label = format!("{}/{}", yl.name, name);
                    let coll_selection = parse_selection(&label, selection)?;
                    if *score_threshold < 0.0 {
                        return Err(ConstructionError::NegativeScoreThreshold {
                            name: label.clone(),
                            value: *score_threshold,
                        });
                    }

                    let mut sec_schemas = Vec::with_capacity(sections.len());
                    for s in sections {
                        if !layer_section_names.insert(s.id.clone()) {
                            return Err(ConstructionError::DuplicateSectionName(s.id.clone()));
                        }
                        let sid = section_alloc.next();
                        maps.section_names.insert((lid, s.id.clone()), sid);
                        let pri = s.priority.unwrap_or(50.0);
                        validate_priority(&format!("{}/{}", yl.name, s.id), pri)?;
                        sec_schemas.push(SectionSchema {
                            id: sid,
                            name: s.id.clone(),
                            content: s.content.clone(),
                            priority: pri,
                            depends_on: None,
                            depends_on_absent: None,
                            is_template: false,
                            template_tokens: None,
                        });
                    }

                    let coll_dw = parse_depth_weights(&label, coll_depth_weights_yaml.as_ref())?;
                    let coll_dw_opt = if coll_depth_weights_yaml.is_some() {
                        Some(coll_dw)
                    } else {
                        None
                    };
                    let coll_summary = build_group_summary(
                        &label,
                        summary,
                        format!("__summary__{name}"),
                        &mut section_alloc,
                    )?;
                    items.push(SystemPromptItem::Collection(SectionCollection {
                        id: cid,
                        name: name.clone(),
                        sections: sec_schemas,
                        selection: coll_selection,
                        score_threshold: *score_threshold,
                        depth_weights: coll_dw_opt,
                        summary: coll_summary,
                        summary_section: None,
                        member_glue: member_glue.clone().unwrap_or_default(),
                        member_glue_tokens: None,
                    }));
                }
                YamlSystemPromptItem::SectionTree { nodes } => {
                    items.push(build_section_tree(
                        &yl.name,
                        lid,
                        nodes,
                        dialect,
                        &mut section_alloc,
                        &mut maps,
                        &mut layer_section_names,
                        &layer_collections,
                    )?);
                }
            }
        }

        // ── this layer's groups ──────────────────────────────────────────────
        let mut groups = Vec::with_capacity(yl.groups.len());
        for yg in &yl.groups {
            if maps.group_names.contains_key(&yg.id) {
                return Err(ConstructionError::DuplicateGroupName(yg.id.clone()));
            }
            global_group_counter += 1;
            let gid = GroupId::new(global_group_counter);
            maps.group_names.insert(yg.id.clone(), gid);

            if yg.score_threshold < 0.0 {
                return Err(ConstructionError::NegativeScoreThreshold {
                    name: yg.id.clone(),
                    value: yg.score_threshold,
                });
            }

            let selection = parse_selection(&yg.id, &yg.selection)?;
            let group_budget = parse_budget(&yg.id, &yg.budget)?;

            groups.push(GroupSchema {
                id: gid,
                name: yg.id.clone(),
                selection,
                score_threshold: yg.score_threshold,
                budget: group_budget,
            });
        }

        if yl.score_threshold < 0.0 {
            return Err(ConstructionError::NegativeScoreThreshold {
                name: yl.name.clone(),
                value: yl.score_threshold,
            });
        }

        let depth_weights = parse_depth_weights(&yl.name, yl.depth_weights.as_ref())?;

        let layer_summary = build_layer_summary(&yl.name, &yl.summary, &mut section_alloc)?;

        layers.push(LayerSchema {
            id: lid,
            name: yl.name.clone(),
            description: yl.description.clone(),
            score_threshold: yl.score_threshold,
            window: yl.window,
            budget: layer_budget,
            system_prompt: SystemPromptSchema { items },
            summary: layer_summary,
            groups,
            depth_weights,
        });
    }

    let schema = Schema { layers };
    Ok((schema, maps))
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a [`CompressionPrompt`] from a YAML framing/instruction pair. Allocates
/// a `SectionId` for its system-prompt framing — that section is never emitted in
/// a normal projection (it is reached only by id during compression), so it is
/// deliberately NOT registered in `NameMaps` or `system_prompt.items`. Validates
/// both prompts are non-empty.
fn build_compression_prompt(
    owner_label: &str,
    yc: &YamlCompressionPrompt,
    section_name: String,
    section_alloc: &mut SectionIdAlloc,
) -> Result<CompressionPrompt, ConstructionError> {
    if yc.system_prompt.trim().is_empty() || yc.user_prompt.trim().is_empty() {
        return Err(ConstructionError::InvalidSummary {
            owner: owner_label.to_string(),
        });
    }
    let sid = section_alloc.next();
    Ok(CompressionPrompt {
        system_prompt: SectionSchema {
            id: sid,
            name: section_name,
            content: yc.system_prompt.clone(),
            priority: 50.0,
            depends_on: None,
            depends_on_absent: None,
            is_template: false,
            template_tokens: None,
        },
        user_prompt: yc.user_prompt.clone(),
    })
}

/// Build one tree-level [`TurnSummary`] (question + answer halves + decode cap).
/// Validates `max_tokens >= 1`; each half allocates its own framing section.
fn build_turn_summary(
    owner_label: &str,
    yt: &YamlTurnSummary,
    q_name: String,
    a_name: String,
    section_alloc: &mut SectionIdAlloc,
) -> Result<TurnSummary, ConstructionError> {
    if yt.max_tokens == 0 {
        return Err(ConstructionError::InvalidSummary {
            owner: owner_label.to_string(),
        });
    }
    let mode = match yt.mode.as_str() {
        "" | "single_pass" => SummaryMode::SinglePass,
        "structural" => SummaryMode::Structural,
        other => {
            return Err(ConstructionError::InvalidSummaryMode {
                owner: owner_label.to_string(),
                mode: other.to_string(),
            })
        }
    };
    Ok(TurnSummary {
        user: build_compression_prompt(owner_label, &yt.user, q_name, section_alloc)?,
        assistant: build_compression_prompt(owner_label, &yt.assistant, a_name, section_alloc)?,
        max_tokens: yt.max_tokens,
        mode,
    })
}

/// Build a layer's [`LayerSummary`] — the mandatory `turns` level plus the
/// optional `summaries` level.
fn build_layer_summary(
    owner_label: &str,
    yl: &YamlLayerSummary,
    section_alloc: &mut SectionIdAlloc,
) -> Result<LayerSummary, ConstructionError> {
    let turns = build_turn_summary(
        owner_label,
        &yl.turns,
        "__summary_turns_user__".to_string(),
        "__summary_turns_assistant__".to_string(),
        section_alloc,
    )?;
    let summaries = match &yl.summaries {
        Some(ys) => Some(build_turn_summary(
            &format!("{owner_label} (summaries)"),
            ys,
            "__summary_sums_user__".to_string(),
            "__summary_sums_assistant__".to_string(),
            section_alloc,
        )?),
        None => None,
    };
    Ok(LayerSummary { turns, summaries })
}

/// Build a section group's [`GroupSummary`] — a single compression prompt and
/// decode cap. Validates `max_tokens >= 1` and non-empty prompts.
fn build_group_summary(
    owner_label: &str,
    yg: &YamlGroupSummary,
    section_base: String,
    section_alloc: &mut SectionIdAlloc,
) -> Result<GroupSummary, ConstructionError> {
    if yg.chunk == 0 || yg.categorize.max_tokens == 0 || yg.assign.max_tokens == 0 {
        return Err(ConstructionError::InvalidSummary {
            owner: owner_label.to_string(),
        });
    }
    let build_stage = |stage: &YamlGroupSummaryStage,
                       suffix: &str,
                       alloc: &mut SectionIdAlloc|
     -> Result<GroupSummaryStage, ConstructionError> {
        let prompt = build_compression_prompt(
            owner_label,
            &YamlCompressionPrompt {
                system_prompt: stage.system_prompt.clone(),
                user_prompt: stage.user_prompt.clone(),
            },
            format!("{section_base}_{suffix}"),
            alloc,
        )?;
        Ok(GroupSummaryStage {
            prompt,
            max_tokens: stage.max_tokens,
        })
    };
    let categorize = build_stage(&yg.categorize, "categorize", section_alloc)?;
    let assign = build_stage(&yg.assign, "assign", section_alloc)?;
    Ok(GroupSummary {
        categorize,
        assign,
        chunk: yg.chunk,
    })
}

/// Mixed-radix pack of `selection[0..radices.len()]` — the variant key for a
/// node whose ancestor dims have the given option counts.
fn pack_assignment(selection: &[u8], radices: &[u8]) -> u32 {
    let mut key = 0u32;
    let mut mult = 1u32;
    for (i, &r) in radices.iter().enumerate() {
        key += selection[i] as u32 * mult;
        mult *= r as u32;
    }
    key
}

/// Append `value` to every suspended branch's assignment.  Called when a dim is
/// declared inside an open `optional_group`: the suspended (other-side) branches
/// never multiplied by that dim, so they must still carry a slot for it to keep
/// every branch's assignment the same width as `radices`.  The slot is the dim's
/// DEFAULT, and the dim is recorded as gated so [`super::schema::SectionTree::pack`]
/// masks the runtime value back to that same default when the group is on the
/// other side — landing the projection on exactly this sealed variant.
fn pad_suspended(suspended: &mut [Vec<(Vec<u8>, Vec<SectionId>)>], value: u8) {
    for branch_set in suspended.iter_mut() {
        for (asm, _) in branch_set.iter_mut() {
            asm.push(value);
        }
    }
}


/// Pre-allocate one collection's [`CollectionId`] and register its name so
/// `depends_on` forward references resolve.  Errors on a duplicate name.
fn register_collection_id(
    name: &str,
    lid: LayerId,
    layer_collection_names: &mut std::collections::HashSet<String>,
    layer_collections: &mut std::collections::HashMap<String, CollectionId>,
    maps: &mut NameMaps,
    collection_alloc: &mut CollectionIdAlloc,
) -> Result<(), ConstructionError> {
    if !layer_collection_names.insert(name.to_string()) {
        return Err(ConstructionError::DuplicateCollectionName(name.to_string()));
    }
    let cid = collection_alloc.next();
    layer_collections.insert(name.to_string(), cid);
    maps.collection_names.insert((lid, name.to_string()), cid);
    Ok(())
}

/// Recurse a section-tree node list, pre-allocating ids for every embedded
/// `Collection` node (including those inside nested `section_tree` nodes).
fn register_tree_collection_ids(
    nodes: &[YamlTreeNode],
    lid: LayerId,
    layer_collection_names: &mut std::collections::HashSet<String>,
    layer_collections: &mut std::collections::HashMap<String, CollectionId>,
    maps: &mut NameMaps,
    collection_alloc: &mut CollectionIdAlloc,
) -> Result<(), ConstructionError> {
    for n in nodes {
        match n {
            YamlTreeNode::Collection { name, .. } => register_collection_id(
                name,
                lid,
                layer_collection_names,
                layer_collections,
                maps,
                collection_alloc,
            )?,
            YamlTreeNode::SectionTree { nodes } => register_tree_collection_ids(
                nodes,
                lid,
                layer_collection_names,
                layer_collections,
                maps,
                collection_alloc,
            )?,
            YamlTreeNode::OptionalGroup { nodes, absent, .. } => {
                for branch in [nodes, absent] {
                    register_tree_collection_ids(
                        branch,
                        lid,
                        layer_collection_names,
                        layer_collections,
                        maps,
                        collection_alloc,
                    )?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// One resolved tree node in the flat spec stream the cross-product walk
/// consumes.  Nested `section_tree` nodes are already expanded; a `Collection`
/// node borrows its YAML so Pass 2 can allocate per-branch member ids.
enum TreeSpec<'a> {
    /// A selector / optional / mandatory section node.
    Node {
        name: String,
        options: Vec<(String, String)>,
        default_option: u8,
        /// Placeholder node: its projection is replaced by this collection's
        /// top-k (resolved to a `CollectionId` in Pass 2).
        inject_collection: Option<&'a str>,
        /// `true` for a `dialect:` mandatory section (a structural marker like
        /// `<tools>`): it becomes live-prefilled glue — no sealed variants,
        /// prefix-transparent, emitted as `Generated` (see [`TreeNode::glue`]).
        is_glue: bool,
    },
    /// A prefix-transparent embedded collection.
    Collection {
        name: &'a str,
        selection: &'a YamlSelection,
        score_threshold: f32,
        depth_weights: &'a Option<YamlDepthWeights>,
        summary: &'a YamlGroupSummary,
        sections: &'a [YamlSection],
        member_glue: Option<&'a str>,
    },
    /// Opens a togglable sub-tree ([`YamlTreeNode::OptionalGroup`]) — a binary
    /// dim whose present branch (option 0) carries the specs up to the matching
    /// [`Self::GroupElse`] (or [`Self::GroupEnd`]), and whose absent branch
    /// (option 1) carries the specs from `GroupElse` to `GroupEnd` (none if there
    /// is no `GroupElse`).
    GroupStart {
        name: String,
        /// `0` = present (default), `1` = absent.
        default_option: u8,
    },
    /// Switches sealing from the present branch to the absent branch of the
    /// innermost open group — the specs after it seal on the `absent` sub-tree.
    GroupElse,
    /// Closes the innermost open [`Self::GroupStart`].
    GroupEnd,
}

/// Resolve a tree node/option's content from an explicit `content:` or a
/// dialect template; either way the section is SEALED (never a live template).
fn resolve_tree_content(
    item: &str,
    content: &str,
    dlct_name: &Option<String>,
    dialect: Option<&Dialect>,
) -> Result<String, ConstructionError> {
    match dlct_name {
        Some(dname) => {
            let dlct = dialect.ok_or_else(|| ConstructionError::DialectRequired {
                item: item.to_string(),
            })?;
            let template = DialectTemplate::from_yaml_name(dname).ok_or_else(|| {
                ConstructionError::UnknownDialectTemplate {
                    item: item.to_string(),
                    name: dname.clone(),
                }
            })?;
            Ok(dlct.template(template).to_string())
        }
        None => Ok(content.to_string()),
    }
}

/// Flatten a tree-node list into a flat [`TreeSpec`] stream, expanding nested
/// `section_tree` nodes inline.  A nested tree's nodes are appended in place —
/// so its selectors become dims declared *after* everything above the nested
/// node, and each inner node seals against the full outer prefix (inheriting
/// every outer selector).  Recursion is bounded by the YAML's nesting depth.
fn flatten_tree_specs<'a>(
    layer_name: &str,
    nodes: &'a [YamlTreeNode],
    dialect: Option<&Dialect>,
    layer_section_names: &mut std::collections::HashSet<String>,
    specs: &mut Vec<TreeSpec<'a>>,
) -> Result<(), ConstructionError> {
    for n in nodes {
        let (id, options, default_option) = match n {
            YamlTreeNode::Section {
                id,
                content,
                dialect: d,
                inject_collection: _,
            } => {
                let item = format!("{layer_name}/{id}");
                (
                    id.clone(),
                    vec![(
                        "content".to_string(),
                        resolve_tree_content(&item, content, d, dialect)?,
                    )],
                    0u8,
                )
            }
            YamlTreeNode::Optional {
                id,
                content,
                dialect: d,
                default,
            } => {
                let item = format!("{layer_name}/{id}");
                // `present` is option 0, `absent` option 1 — an unspecified
                // `default:` falls back to present.
                let default_option = match default.as_deref() {
                    None => 0u8,
                    Some(s) => match OptionalState::from_id(s) {
                        Some(OptionalState::Present) => 0u8,
                        Some(OptionalState::Absent) => 1u8,
                        None => {
                            return Err(ConstructionError::InvalidToggleDefault {
                                item,
                                value: s.to_string(),
                            })
                        }
                    },
                };
                (
                    id.clone(),
                    vec![
                        (
                            OptionalState::Present.as_id().to_string(),
                            resolve_tree_content(&item, content, d, dialect)?,
                        ),
                        (OptionalState::Absent.as_id().to_string(), String::new()),
                    ],
                    default_option,
                )
            }
            YamlTreeNode::Selector {
                id,
                default,
                options,
            } => {
                let item = format!("{layer_name}/{id}");
                let mut opts = Vec::with_capacity(options.len());
                for o in options {
                    opts.push((
                        o.id.clone(),
                        resolve_tree_content(
                            &format!("{item}/{}", o.id),
                            &o.content,
                            &o.dialect,
                            dialect,
                        )?,
                    ));
                }
                let default_option =
                    opts.iter()
                        .position(|(oid, _)| oid == default)
                        .ok_or_else(|| ConstructionError::UnknownTreeOption {
                            item,
                            option: default.clone(),
                        })? as u8;
                (id.clone(), opts, default_option)
            }
            YamlTreeNode::SectionTree { nodes } => {
                flatten_tree_specs(layer_name, nodes, dialect, layer_section_names, specs)?;
                continue;
            }
            YamlTreeNode::Collection {
                name,
                selection,
                score_threshold,
                depth_weights,
                summary,
                sections,
                member_glue,
            } => {
                specs.push(TreeSpec::Collection {
                    name,
                    selection,
                    score_threshold: *score_threshold,
                    depth_weights,
                    summary,
                    sections,
                    member_glue: member_glue.as_deref(),
                });
                continue;
            }
            YamlTreeNode::OptionalGroup {
                id,
                default,
                nodes,
                absent,
            } => {
                let item = format!("{layer_name}/{id}");
                // present = option 0, absent = option 1 (mirrors `optional`).
                let default_option = match default.as_deref() {
                    None => 0u8,
                    Some(s) => match OptionalState::from_id(s) {
                        Some(OptionalState::Present) => 0u8,
                        Some(OptionalState::Absent) => 1u8,
                        None => {
                            return Err(ConstructionError::InvalidToggleDefault {
                                item,
                                value: s.to_string(),
                            })
                        }
                    },
                };
                if !layer_section_names.insert(id.clone()) {
                    return Err(ConstructionError::DuplicateSectionName(id.clone()));
                }
                specs.push(TreeSpec::GroupStart {
                    name: id.clone(),
                    default_option,
                });
                flatten_tree_specs(layer_name, nodes, dialect, layer_section_names, specs)?;
                if !absent.is_empty() {
                    specs.push(TreeSpec::GroupElse);
                    flatten_tree_specs(layer_name, absent, dialect, layer_section_names, specs)?;
                }
                specs.push(TreeSpec::GroupEnd);
                continue;
            }
        };
        if !layer_section_names.insert(id.clone()) {
            return Err(ConstructionError::DuplicateSectionName(id));
        }
        let inject_collection = match n {
            YamlTreeNode::Section {
                inject_collection, ..
            } => inject_collection.as_deref(),
            _ => None,
        };
        // A `dialect:` mandatory section is a structural marker → live glue.
        // (An `optional`/`selector` with a dialect is a real dimension, not glue.)
        let is_glue = matches!(
            n,
            YamlTreeNode::Section {
                dialect: Some(_),
                ..
            }
        );
        specs.push(TreeSpec::Node {
            name: id,
            options,
            default_option,
            inject_collection,
            is_glue,
        });
    }
    Ok(())
}

/// Build one `kind: section_tree` item — the full cross-product fan-out.
///
/// Each node carries a list of options; nodes with >1 option are SELECTOR
/// dimensions.  A breadth-first walk maintains the live branch set —
/// `(option-assignment-of-dims-so-far, in-tree prefix of emitted variant ids)`.
/// At each node it seals one [`TreeVariant`] per (non-empty option × current
/// branch); then a mandatory node appends its single option's variant to every
/// branch, while a selector multiplies the branch set by its option count (each
/// non-empty option's variant joining that sub-branch's prefix).  The node's
/// declared name maps to its default-selection variant id.
#[allow(clippy::too_many_arguments)]
fn build_section_tree<'a>(
    layer_name: &str,
    lid: LayerId,
    nodes: &'a [YamlTreeNode],
    dialect: Option<&Dialect>,
    section_alloc: &mut SectionIdAlloc,
    maps: &mut NameMaps,
    layer_section_names: &mut std::collections::HashSet<String>,
    layer_collections: &std::collections::HashMap<String, CollectionId>,
) -> Result<SystemPromptItem, ConstructionError> {
    // ── Pass 1: resolve each node into a flat spec stream, expanding nested
    // `section_tree` nodes inline (their selectors inherit the outer dims). ──
    let mut specs: Vec<TreeSpec<'a>> = Vec::with_capacity(nodes.len());
    flatten_tree_specs(layer_name, nodes, dialect, layer_section_names, &mut specs)?;

    // ── Pass 2: cross-product fan-out ────────────────────────────────────────
    // A branch = (option assignment over the dims seen so far, in-tree prefix).
    let mut branches: Vec<(Vec<u8>, Vec<SectionId>)> = vec![(Vec::new(), Vec::new())];
    let mut radices: Vec<u8> = Vec::new(); // option_count of each dim, in order
    let mut dims: Vec<TreeDim> = Vec::new();
    let mut default_selection: Vec<u8> = Vec::new();
    let mut tree_nodes: Vec<TreeNode> = Vec::with_capacity(specs.len());
    // Open `optional_group` scopes: each holds the ABSENT sub-branches set aside
    // while the group's children seal/extend only the PRESENT sub-branches. Popped
    // and merged back at the matching `GroupEnd`.
    let mut suspended: Vec<Vec<(Vec<u8>, Vec<SectionId>)>> = Vec::new();
    // Parallel to `suspended`: the (dim-index, active-side) of each open group, so
    // a dim declared inside it is recorded as gated by that group's current side
    // (`0` present / `1` absent — flipped at `GroupElse`).
    let mut group_ctx: Vec<(usize, u8)> = Vec::new();

    for spec in &specs {
        let ancestor_dims = radices.len();
        match spec {
            TreeSpec::Node {
                name,
                options: spec_options,
                default_option,
                inject_collection,
                is_glue,
            } => {
                // A live-prefilled structural marker (`<tools>` etc.): allocate NO
                // sealed variant, do NOT extend the prefix (nodes below seal as if
                // it weren't here), and record the branches it lives in so it emits
                // its `Generated` run only there.  Its text stays in `options[0]`
                // for tokenisation + the GUI glue label.
                if *is_glue {
                    let active_keys: Vec<u32> = branches
                        .iter()
                        .map(|(asm, _)| pack_assignment(asm, &radices))
                        .collect();
                    tree_nodes.push(TreeNode {
                        name: name.clone(),
                        options: vec![TreeOption {
                            id: spec_options[0].0.clone(),
                            content: spec_options[0].1.clone(),
                            variants: Vec::new(),
                        }],
                        collection: None,
                        inject_collection: None,
                        glue: Some(super::schema::TreeGlue {
                            tokens: None,
                            active_keys,
                        }),
                        dim: None,
                        ancestor_dims,
                    });
                    continue;
                }
                let is_dim = spec_options.len() > 1;
                // Resolve the placeholder's target collection name → id (if any).
                let inject_cid = match inject_collection {
                    Some(cname) => Some(*layer_collections.get(*cname).ok_or_else(|| {
                        ConstructionError::UnknownCollection(format!(
                            "{layer_name}: node '{name}' inject_collection references unknown \
                             collection '{cname}'",
                        ))
                    })?),
                    None => None,
                };

                // Seal one variant per (non-empty option × current branch).
                let mut options: Vec<TreeOption> = Vec::with_capacity(spec_options.len());
                for (oid, content) in spec_options {
                    let mut variants = Vec::new();
                    if !content.is_empty() {
                        for (asm, prefix) in &branches {
                            variants.push(TreeVariant {
                                ancestors: pack_assignment(asm, &radices),
                                id: section_alloc.next(),
                                in_tree_prefix: prefix.clone(),
                            });
                        }
                    }
                    options.push(TreeOption {
                        id: oid.clone(),
                        content: content.clone(),
                        variants,
                    });
                }

                // The declared name resolves to the default selection's variant.
                let default_opt = &options[*default_option as usize];
                if let Some(v) =
                    default_opt.variant_for(pack_assignment(&default_selection, &radices))
                {
                    maps.section_names.insert((lid, name.clone()), v.id);
                }

                // Advance the live branch set + dims.
                if is_dim {
                    let dim_index = dims.len();
                    dims.push(TreeDim {
                        selector_id: name.clone(),
                        node_index: tree_nodes.len(),
                        option_count: spec_options.len() as u8,
                        default_option: *default_option,
                        // Gated by the innermost open group's current side (if any).
                        gate: group_ctx.last().copied(),
                    });
                    let mut next = Vec::with_capacity(branches.len() * spec_options.len());
                    for (asm, prefix) in &branches {
                        let key = pack_assignment(asm, &radices);
                        for (opt_i, option) in options.iter().enumerate() {
                            let mut sub_asm = asm.clone();
                            sub_asm.push(opt_i as u8);
                            let mut sub_prefix = prefix.clone();
                            if let Some(v) = option.variant_for(key) {
                                sub_prefix.push(v.id); // non-empty option joins the prefix
                            }
                            next.push((sub_asm, sub_prefix));
                        }
                    }
                    branches = next;
                    radices.push(spec_options.len() as u8);
                    default_selection.push(*default_option);
                    // Inside a group, the suspended (other-side) branches never saw
                    // this selector — pad them with its default to keep widths even;
                    // the gate above makes `pack` mask to that same default.
                    pad_suspended(&mut suspended, *default_option);
                    tree_nodes.push(TreeNode {
                        name: name.clone(),
                        options,
                        collection: None,
                        inject_collection: inject_cid,
                        glue: None,
                        dim: Some(dim_index),
                        ancestor_dims,
                    });
                } else {
                    // Mandatory: append its single option's variant to every branch.
                    for (asm, prefix) in branches.iter_mut() {
                        let key = pack_assignment(asm, &radices);
                        if let Some(v) = options[0].variant_for(key) {
                            prefix.push(v.id);
                        }
                    }
                    tree_nodes.push(TreeNode {
                        name: name.clone(),
                        options,
                        collection: None,
                        inject_collection: inject_cid,
                        glue: None,
                        dim: None,
                        ancestor_dims,
                    });
                }
            }
            TreeSpec::Collection {
                name,
                selection,
                score_threshold,
                depth_weights,
                summary,
                sections,
                member_glue,
            } => {
                // A prefix-transparent embedded collection: seal each member
                // ONCE PER ancestor branch (so members fan out ×outer-selectors),
                // but DO NOT touch `branches`/`radices` — nodes below seal as if
                // the variable members were not here (they anchor on the next
                // mandatory node instead).
                let label = format!("{layer_name}/{name}");
                let cid = *layer_collections.get(*name).ok_or_else(|| {
                    ConstructionError::UnknownCollection(format!(
                        "{layer_name}: embedded collection '{name}' has no pre-allocated id",
                    ))
                })?;
                let coll_selection = parse_selection(&label, selection)?;
                if *score_threshold < 0.0 {
                    return Err(ConstructionError::NegativeScoreThreshold {
                        name: label.clone(),
                        value: *score_threshold,
                    });
                }
                let coll_dw = parse_depth_weights(&label, depth_weights.as_ref())?;
                let coll_dw_opt = depth_weights.is_some().then_some(coll_dw);
                let coll_summary = build_group_summary(
                    &label,
                    summary,
                    format!("__summary__{name}"),
                    section_alloc,
                )?;

                let default_key = pack_assignment(&default_selection, &radices);
                let mut member_variants: Vec<Vec<TreeVariant>> = Vec::with_capacity(sections.len());
                let mut sec_schemas: Vec<SectionSchema> = Vec::with_capacity(sections.len());
                for s in sections.iter() {
                    if !layer_section_names.insert(s.id.clone()) {
                        return Err(ConstructionError::DuplicateSectionName(s.id.clone()));
                    }
                    let pri = s.priority.unwrap_or(50.0);
                    validate_priority(&format!("{layer_name}/{}", s.id), pri)?;
                    let mut variants = Vec::with_capacity(branches.len());
                    for (asm, prefix) in &branches {
                        variants.push(TreeVariant {
                            ancestors: pack_assignment(asm, &radices),
                            id: section_alloc.next(),
                            in_tree_prefix: prefix.clone(),
                        });
                    }
                    // The default-branch variant id is canonical: it drives
                    // `depends_on` gating, BDP scoring, and `id_for_section`.
                    let canonical = variants
                        .iter()
                        .find(|v| v.ancestors == default_key)
                        .map(|v| v.id)
                        .unwrap_or(variants[0].id);
                    maps.section_names.insert((lid, s.id.clone()), canonical);
                    sec_schemas.push(SectionSchema {
                        id: canonical,
                        name: s.id.clone(),
                        content: s.content.clone(),
                        priority: pri,
                        depends_on: None,
                        depends_on_absent: None,
                        is_template: false,
                        template_tokens: None,
                    });
                    member_variants.push(variants);
                }
                let collection = SectionCollection {
                    id: cid,
                    name: name.to_string(),
                    sections: sec_schemas,
                    selection: coll_selection,
                    score_threshold: *score_threshold,
                    depth_weights: coll_dw_opt,
                    summary: coll_summary,
                    summary_section: None,
                    member_glue: member_glue.map(str::to_string).unwrap_or_default(),
                    member_glue_tokens: None,
                };
                // Capture the branch templates so runtime member additions
                // (the tool catalog) can seal ×branch without re-deriving them.
                let branch_templates: Vec<(u32, Vec<SectionId>)> = branches
                    .iter()
                    .map(|(asm, prefix)| (pack_assignment(asm, &radices), prefix.clone()))
                    .collect();
                tree_nodes.push(TreeNode {
                    name: name.to_string(),
                    options: Vec::new(),
                    collection: Some(TreeCollection {
                        collection,
                        variants: member_variants,
                        branches: branch_templates,
                        default_branch: default_key,
                        // Set below if a placeholder node injects this collection.
                        deferred_projection: false,
                    }),
                    inject_collection: None,
                    glue: None,
                    dim: None,
                    ancestor_dims,
                });
            }
            TreeSpec::GroupStart {
                name,
                default_option,
            } => {
                // A binary dim whose options carry NO content of their own (the
                // group's child nodes are what emit).  Split every live branch into
                // a present (option 0) sub-branch — which the children below will
                // seal/extend — and an absent (option 1) sub-branch, set aside until
                // `GroupEnd`.  A synthetic toggle node anchors the dim so the runtime
                // can drive presence by `name`; it emits nothing itself.
                let dim_index = dims.len();
                dims.push(TreeDim {
                    selector_id: name.clone(),
                    node_index: tree_nodes.len(),
                    option_count: 2,
                    default_option: *default_option,
                    // A nested group is itself gated by its ENCLOSING group's side.
                    gate: group_ctx.last().copied(),
                });
                tree_nodes.push(TreeNode {
                    name: name.clone(),
                    options: vec![
                        TreeOption {
                            id: OptionalState::Present.as_id().to_string(),
                            content: String::new(),
                            variants: Vec::new(),
                        },
                        TreeOption {
                            id: OptionalState::Absent.as_id().to_string(),
                            content: String::new(),
                            variants: Vec::new(),
                        },
                    ],
                    collection: None,
                    inject_collection: None,
                    glue: None,
                    dim: Some(dim_index),
                    ancestor_dims,
                });
                let mut present = Vec::with_capacity(branches.len());
                let mut absent = Vec::with_capacity(branches.len());
                for (asm, prefix) in &branches {
                    let mut p_asm = asm.clone();
                    p_asm.push(0); // present
                    present.push((p_asm, prefix.clone()));
                    let mut a_asm = asm.clone();
                    a_asm.push(1); // absent
                    absent.push((a_asm, prefix.clone()));
                }
                // An ENCLOSING group's suspended branches never saw this nested
                // group's dim — pad them with its default (this group's own `absent`
                // already carries the dim, so pad before pushing it).
                pad_suspended(&mut suspended, *default_option);
                suspended.push(absent);
                branches = present;
                radices.push(2);
                default_selection.push(*default_option);
                group_ctx.push((dim_index, 0)); // sealing the PRESENT side first
            }
            TreeSpec::GroupElse => {
                // Switch sealing to the ABSENT sub-branches: stash the extended
                // present branches and restore the (untouched) absent ones, so the
                // following specs seal on the absent branch.
                let absent = suspended
                    .pop()
                    .expect("GroupElse without a matching GroupStart");
                suspended.push(std::mem::take(&mut branches)); // extended present
                branches = absent;
                group_ctx.last_mut().expect("GroupElse without GroupStart").1 = 1; // absent side
            }
            TreeSpec::GroupEnd => {
                // Fold the OTHER branch (the absent sub-branches when there was no
                // `GroupElse`, or the extended present sub-branches when there was)
                // back in.  Nodes below this point seal across BOTH (×present/absent).
                let other = suspended
                    .pop()
                    .expect("GroupEnd without a matching GroupStart");
                branches.extend(other);
                group_ctx.pop();
            }
        }
    }

    // A placeholder node (`inject_collection`) takes over its target collection's
    // projection, so mark that collection deferred — it must NOT also emit at its
    // own tree position.
    let injected: std::collections::HashSet<CollectionId> = tree_nodes
        .iter()
        .filter_map(|n| n.inject_collection)
        .collect();
    for node in tree_nodes.iter_mut() {
        if let Some(tc) = node.collection.as_mut() {
            if injected.contains(&tc.collection.id) {
                tc.deferred_projection = true;
            }
        }
    }

    let mut tree = SectionTree {
        nodes: tree_nodes,
        dims,
        default_selection,
        default_present_ids: Vec::new(),
    };
    // The default selection's emitted variant ids — what sections after the tree
    // (and priming) attend to.  Collection nodes are prefix-transparent and emit
    // a runtime-selected subset, so they contribute nothing here.
    tree.default_present_ids = tree
        .nodes
        .iter()
        .filter(|n| n.collection.is_none())
        .filter_map(|n| {
            let opt = &n.options[n.chosen(&tree.default_selection)];
            opt.variant_for(tree.pack(&tree.default_selection, n.ancestor_dims))
                .map(|v| v.id)
        })
        .collect();

    Ok(SystemPromptItem::SectionTree(tree))
}

fn validate_priority(name: &str, v: f32) -> Result<(), ConstructionError> {
    if v <= 0.0 {
        return Err(ConstructionError::InvalidPriority {
            name: name.to_string(),
            value: v,
        });
    }
    Ok(())
}

fn validate_percent(name: &str, v: f32) -> Result<(), ConstructionError> {
    if !(0.0..=100.0).contains(&v) {
        return Err(ConstructionError::InvalidPercentage {
            name: name.to_string(),
            value: v,
        });
    }
    Ok(())
}

fn parse_budget(name: &str, yb: &YamlBudget) -> Result<Budget, ConstructionError> {
    let priority = yb.priority.unwrap_or(50.0);
    validate_priority(name, priority)?;
    if let Some(v) = yb.min_percent {
        validate_percent(name, v)?;
    }
    if let Some(v) = yb.max_percent {
        validate_percent(name, v)?;
    }
    Ok(Budget {
        priority,
        min_percent: yb.min_percent,
        max_percent: yb.max_percent,
    })
}

fn parse_depth_weights(
    layer_name: &str,
    yw: Option<&YamlDepthWeights>,
) -> Result<DepthWeights, ConstructionError> {
    let default = DepthWeights::default();
    let Some(yw) = yw else {
        return Ok(default);
    };

    let syntactic = yw.syntactic.unwrap_or(default.syntactic);
    let semantic = yw.semantic.unwrap_or(default.semantic);
    let pragmatic = yw.pragmatic.unwrap_or(default.pragmatic);

    for (depth, value) in [
        ("syntactic", syntactic),
        ("semantic", semantic),
        ("pragmatic", pragmatic),
    ] {
        if value < 0.0 {
            return Err(ConstructionError::NegativeDepthWeight {
                layer: layer_name.to_string(),
                depth,
                value,
            });
        }
    }

    if syntactic == 0.0 && semantic == 0.0 && pragmatic == 0.0 {
        return Err(ConstructionError::AllDepthWeightsZero {
            layer: layer_name.to_string(),
        });
    }

    Ok(DepthWeights {
        syntactic,
        semantic,
        pragmatic,
    })
}

fn parse_selection(name: &str, ys: &YamlSelection) -> Result<SelectionRule, ConstructionError> {
    match ys.kind.as_str() {
        "" | "always_visible" => Ok(SelectionRule::AlwaysVisible),
        "top_k" => {
            let k = ys.k.ok_or_else(|| ConstructionError::InvalidTopK {
                name: name.to_string(),
            })?;
            if k == 0 {
                return Err(ConstructionError::InvalidTopK {
                    name: name.to_string(),
                });
            }
            Ok(SelectionRule::TopK { k })
        }
        "single" => Ok(SelectionRule::Single),
        "conversation" => {
            let recent = ys.recent.unwrap_or(0);
            let historical_top_k = ys.historical_top_k.unwrap_or(0);
            if recent == 0 && historical_top_k == 0 {
                return Err(ConstructionError::InvalidConversationK {
                    name: name.to_string(),
                });
            }
            Ok(SelectionRule::Sequence {
                recent,
                historical_top_k,
            })
        }
        other => Err(ConstructionError::UnknownSelectionKind(other.to_string())),
    }
}
