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
use super::schema::{
    Budget, CompressionPrompt, DepthWeights, GroupSchema, GroupSummary, GroupSummaryStage,
    LayerSchema, LayerSummary, Schema, SectionCollection, SectionSchema, SelectionRule,
    SummaryMode, SystemPromptItem, SystemPromptSchema, TurnSummary,
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
                } => {
                    if !layer_section_names.insert(id.clone()) {
                        return Err(ConstructionError::DuplicateSectionName(id.clone()));
                    }
                    let sid = section_alloc.next();
                    maps.section_names.insert((lid, id.clone()), sid);
                    let pri = priority.unwrap_or(50.0);
                    validate_priority(&format!("{}/{}", yl.name, id), pri)?;
                    let depends_on_cid = match depends_on {
                        Some(name) => Some(*layer_collections.get(name).ok_or_else(|| {
                            ConstructionError::UnknownCollection(format!(
                                "{}: section '{}' depends_on unknown collection '{}'",
                                yl.name, id, name,
                            ))
                        })?),
                        None => None,
                    };
                    items.push(SystemPromptItem::Section(SectionSchema {
                        id: sid,
                        name: id.clone(),
                        content: content.clone(),
                        priority: pri,
                        depends_on: depends_on_cid,
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
