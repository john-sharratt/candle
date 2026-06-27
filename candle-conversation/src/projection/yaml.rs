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
    SelectionRule, SummaryMode, SystemPromptItem, SystemPromptSchema, TreeDim, TreeNode,
    TreeOption, TreeVariant, TurnSummary,
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
            if let YamlSystemPromptItem::Collection { name, .. } = entry {
                if !layer_collection_names.insert(name.clone()) {
                    return Err(ConstructionError::DuplicateCollectionName(name.clone()));
                }
                let cid = collection_alloc.next();
                layer_collections.insert(name.clone(), cid);
                maps.collection_names.insert((lid, name.clone()), cid);
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
fn build_section_tree(
    layer_name: &str,
    lid: LayerId,
    nodes: &[YamlTreeNode],
    dialect: Option<&Dialect>,
    section_alloc: &mut SectionIdAlloc,
    maps: &mut NameMaps,
    layer_section_names: &mut std::collections::HashSet<String>,
) -> Result<SystemPromptItem, ConstructionError> {
    // Resolve a node/option's content from an explicit `content:` or a dialect
    // template; either way the section is SEALED (never a live template).
    let resolve_content = |item: &str,
                           content: &str,
                           dlct_name: &Option<String>|
     -> Result<String, ConstructionError> {
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
    };

    // ── Pass 1: resolve each node's (id, options, default option index) ──────
    struct Spec {
        name: String,
        options: Vec<(String, String)>, // (option id, content)
        default_option: u8,
    }
    let mut specs: Vec<Spec> = Vec::with_capacity(nodes.len());
    for n in nodes {
        let (id, options, default_option) = match n {
            YamlTreeNode::Section {
                id,
                content,
                dialect: d,
            } => {
                let item = format!("{layer_name}/{id}");
                (
                    id.clone(),
                    vec![("content".to_string(), resolve_content(&item, content, d)?)],
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
                            resolve_content(&item, content, d)?,
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
                        resolve_content(&format!("{item}/{}", o.id), &o.content, &o.dialect)?,
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
        };
        if !layer_section_names.insert(id.clone()) {
            return Err(ConstructionError::DuplicateSectionName(id));
        }
        specs.push(Spec {
            name: id,
            options,
            default_option,
        });
    }

    // ── Pass 2: cross-product fan-out ────────────────────────────────────────
    // A branch = (option assignment over the dims seen so far, in-tree prefix).
    let mut branches: Vec<(Vec<u8>, Vec<SectionId>)> = vec![(Vec::new(), Vec::new())];
    let mut radices: Vec<u8> = Vec::new(); // option_count of each dim, in order
    let mut dims: Vec<TreeDim> = Vec::new();
    let mut default_selection: Vec<u8> = Vec::new();
    let mut tree_nodes: Vec<TreeNode> = Vec::with_capacity(specs.len());

    for spec in &specs {
        let ancestor_dims = radices.len();
        let is_dim = spec.options.len() > 1;

        // Seal one variant per (non-empty option × current branch).
        let mut options: Vec<TreeOption> = Vec::with_capacity(spec.options.len());
        for (oid, content) in &spec.options {
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
        let default_opt = &options[spec.default_option as usize];
        if let Some(v) = default_opt.variant_for(pack_assignment(&default_selection, &radices)) {
            maps.section_names.insert((lid, spec.name.clone()), v.id);
        }

        // Advance the live branch set + dims.
        if is_dim {
            let dim_index = dims.len();
            dims.push(TreeDim {
                selector_id: spec.name.clone(),
                node_index: tree_nodes.len(),
                option_count: spec.options.len() as u8,
                default_option: spec.default_option,
            });
            let mut next = Vec::with_capacity(branches.len() * spec.options.len());
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
            radices.push(spec.options.len() as u8);
            default_selection.push(spec.default_option);
            tree_nodes.push(TreeNode {
                name: spec.name.clone(),
                options,
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
                name: spec.name.clone(),
                options,
                dim: None,
                ancestor_dims,
            });
        }
    }

    let mut tree = SectionTree {
        nodes: tree_nodes,
        dims,
        default_selection,
        default_present_ids: Vec::new(),
    };
    // The default selection's emitted variant ids — what sections after the tree
    // (and priming) attend to.
    tree.default_present_ids = tree
        .nodes
        .iter()
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
