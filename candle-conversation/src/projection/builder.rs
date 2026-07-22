//! Public [`Builder`] type — the projection engine's user-facing API.
//!
//! # Lifecycle
//!
//! ```text
//!   ┌────────────────────────────────────────┐
//!   │ Builder::from_yaml_with_vars(s, vars)  │   parse + validate + substitute
//!   └─────────────────┬──────────────────────┘
//!                     │
//!                     ▼
//!   ┌────────────────────────────────────────┐
//!   │ id_for_layer/group/section(name) -> Id │   name → opaque id
//!   └─────────────────┬──────────────────────┘
//!                     │
//!                     ▼
//!   ┌────────────────────────────────────────┐   ← turn state lives here,
//!   │ Substrate (or any ContentResolver│     not in Builder
//!   │   .append(group, tokens) -> TurnIndex  │
//!   │   .set_score(group, index, score)      │
//!   └─────────────────┬──────────────────────┘
//!                     │
//!                     ▼
//!   ┌────────────────────────────────────────┐   ← pure given valid schema
//!   │ builder.project(target, &resolver)     │     + working resolver
//!   └────────────────────────────────────────┘
//! ```
//!
//! # Construction validation
//!
//! After YAML parse, [`Builder::from_yaml`] runs the construction-time
//! checks documented in [`super::error::ConstructionError`]:
//!
//! - Window split fits inside the total budget
//! - No sibling group of any one parent declares mins summing past 100%
//! - `max_percent ≥ min_percent` for every node where both are declared
//! - All `score_threshold` values are non-negative
//! - Selection rule numeric fields are in range
//!
//! Anything that survives this is guaranteed to project without errors.

use super::error::ConstructionError;
use super::ids::{CollectionId, GroupId, LayerId, Reserved, SectionId};
use super::policy::SelectionPolicy;
use super::project::{
    run, run_with_sink, PriorBelief, Projection, ProjectionMode, ProjectionTarget, SelectionState,
};
use super::schema::{
    Budget, CompressionPrompt, Content, DecodePriority, GatherScope, GroupSchema, GroupSummary,
    LayerDials, LayerSchema, LayerSummary, Schema, SectionCollection, SectionSchema, SelectionRule,
    SystemPromptItem, SystemPromptSchema, TreeCollection, TreeVariant, TurnSummary,
};
use super::yaml::{from_yaml, NameMaps};
use crate::substrate::ContentResolver;
use crate::summary_tree::scope::Scope;
use crate::summary_tree::SelectionDiagnostics;

/// Run all construction-time validation checks against a parsed schema.
///
/// Called by both [`Builder::from_yaml`] and [`Builder::from_schema`] so
/// programmatically-constructed schemas get the same guarantees as YAML
/// ones.
fn validate(schema: &Schema) -> Result<(), ConstructionError> {
    // Layer sibling mins. Each layer's budget.min_percent is its share of
    // the *target's* window when this layer is visible-but-not-target.
    // Mins across all layers must still sum to ≤ 100% of any one target's
    // window.
    let layer_min_sum: f32 = schema
        .layers
        .iter()
        .filter_map(|l| l.budget.min_percent)
        .sum();
    if layer_min_sum > 100.0 + f32::EPSILON {
        return Err(ConstructionError::MinPercentExceedsTotal {
            parent: "layers".to_string(),
            sum: layer_min_sum,
        });
    }

    // The single shared system prompt must render at least one section, so every
    // layer is usable as a projection target.
    if schema.system_prompt.items.is_empty()
        || schema.system_prompt.all_section_ids().next().is_none()
    {
        return Err(ConstructionError::EmptySystemPrompt);
    }

    for layer in &schema.layers {
        validate_budget_bounds(&layer.name, &layer.budget)?;

        // Group sibling mins within this layer.
        let group_min_sum: f32 = layer
            .groups
            .iter()
            .filter_map(|g| g.budget.min_percent)
            .sum();
        if group_min_sum > 100.0 + f32::EPSILON {
            return Err(ConstructionError::MinPercentExceedsTotal {
                parent: layer.name.clone(),
                sum: group_min_sum,
            });
        }

        for group in &layer.groups {
            validate_budget_bounds(&group.name, &group.budget)?;
            validate_selection(&group.name, &group.selection)?;

            if group.score_threshold < 0.0 {
                return Err(ConstructionError::NegativeScoreThreshold {
                    name: group.name.clone(),
                    value: group.score_threshold,
                });
            }
        }

        if layer.score_threshold < 0.0 {
            return Err(ConstructionError::NegativeScoreThreshold {
                name: layer.name.clone(),
                value: layer.score_threshold,
            });
        }
    }

    Ok(())
}

fn validate_budget_bounds(name: &str, budget: &Budget) -> Result<(), ConstructionError> {
    if let (Some(min), Some(max)) = (budget.min_percent, budget.max_percent) {
        if max < min {
            return Err(ConstructionError::MaxLessThanMin {
                name: name.to_string(),
            });
        }
    }
    Ok(())
}

/// Replace every `{ident}` in `template` with its value from `vars`.
///
/// Returns `UnresolvedVariable` on the first `{ident}` that has no entry.
/// Identifier syntax: `[A-Za-z_][A-Za-z0-9_]*`. Other `{...}` patterns
/// (e.g. JSON-like fragments in content) are left untouched.
fn substitute_template(template: &str, vars: &[(&str, &str)]) -> Result<String, ConstructionError> {
    let lookup: std::collections::HashMap<&str, &str> = vars.iter().copied().collect();
    let mut out = String::with_capacity(template.len());
    let bytes = template.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            // Try to read an identifier ending at '}'.
            let id_start = i + 1;
            let mut j = id_start;
            while j < bytes.len() {
                let c = bytes[j];
                let ok = if j == id_start {
                    c.is_ascii_alphabetic() || c == b'_'
                } else {
                    c.is_ascii_alphanumeric() || c == b'_'
                };
                if !ok {
                    break;
                }
                j += 1;
            }
            // A valid placeholder is `{<ident>}` with at least 1 char and a closing brace.
            if j > id_start && j < bytes.len() && bytes[j] == b'}' {
                let name = &template[id_start..j];
                if let Some(value) = lookup.get(name) {
                    out.push_str(value);
                    i = j + 1;
                    continue;
                } else {
                    return Err(ConstructionError::UnresolvedVariable {
                        name: name.to_string(),
                    });
                }
            }
            // Not a recognised placeholder shape — emit `{` literally.
        }
        // SAFETY: index is on a UTF-8 byte boundary because we only advanced
        // past ASCII bytes above; otherwise we fall through with the original `i`.
        let ch_start = i;
        let mut ch_end = ch_start + 1;
        while ch_end < bytes.len() && (bytes[ch_end] & 0xC0) == 0x80 {
            ch_end += 1;
        }
        out.push_str(&template[ch_start..ch_end]);
        i = ch_end;
    }
    Ok(out)
}

fn validate_selection(name: &str, rule: &SelectionRule) -> Result<(), ConstructionError> {
    use SelectionRule;
    match rule {
        SelectionRule::TopK { k } if *k == 0 => Err(ConstructionError::InvalidTopK {
            name: name.to_string(),
        }),
        SelectionRule::Sequence {
            recent,
            historical_top_k,
        } if *recent == 0 && *historical_top_k == 0 => {
            Err(ConstructionError::InvalidConversationK {
                name: name.to_string(),
            })
        }
        SelectionRule::Named { selector } if selector.trim().is_empty() => {
            Err(ConstructionError::EmptyNamedSelector {
                name: name.to_string(),
            })
        }
        _ => Ok(()),
    }
}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Compiled schema for the multi-layer projection substrate.
///
/// Holds the immutable [`Schema`] and reverse-lookup maps from YAML string
/// names to opaque ids. Produces a [`Projection`] on each
/// [`Builder::project`] call.
///
/// **Turn state lives in the resolver, not here.** Pass a [`super::resolver::Substrate`]
/// (or any [`ContentResolver`] impl) to [`Builder::project`]; the schema
/// never changes once constructed.
#[derive(Debug, Clone)]
pub struct Builder {
    pub(super) schema: Schema,
    /// Reverse maps from YAML string names to ids. Empty when constructed
    /// via [`Builder::from_schema`].
    pub(super) name_maps: NameMaps,
    /// The resolved YAML this builder was parsed from, retained so the
    /// substrate can persist it as the `Template` record (the projection
    /// schema is reconstructable via [`Builder::from_yaml`]). `None` for
    /// programmatic [`Builder::from_schema`] construction.
    source_yaml: Option<String>,
}

impl Builder {
    /// Parse `yaml`, assign ids, run all validations, return a ready
    /// builder.
    ///
    /// The YAML must be fully literal — any `{name}` placeholder triggers
    /// [`ConstructionError::UnresolvedVariable`]. Use
    /// [`Builder::from_yaml_with_vars`] when the template carries
    /// substitution variables.
    ///
    /// # Errors
    ///
    /// See [`ConstructionError`] for the full list of parse-time and
    /// construction-time failures.
    pub fn from_yaml(yaml: &str) -> Result<Self, ConstructionError> {
        Self::from_yaml_with_vars(yaml, &[])
    }

    /// Parse `yaml_template` with `{name}` placeholders substituted from
    /// `vars`, then build the schema as for [`Builder::from_yaml`].
    ///
    /// # Substitution contract
    ///
    /// **All substitution happens here, exactly once, at builder
    /// construction time.** The resulting [`Schema`] is immutable and
    /// carries the substituted content forever — there is no projection-
    /// time substitution path. This guarantees the prefilled system-prompt
    /// KV cache stays stable across every projection issued from the same
    /// builder.
    ///
    /// # Placeholder syntax
    ///
    /// Placeholders are `{name}` where `name` matches `[A-Za-z_][A-Za-z0-9_]*`.
    /// Any literal `{` followed by an identifier and `}` will be
    /// interpreted as a placeholder; if you need a literal `{ident}` in
    /// content, supply it via `vars` with itself as the value.
    ///
    /// # Errors
    ///
    /// - [`ConstructionError::UnresolvedVariable`] if the template
    ///   contains a `{name}` for which no entry exists in `vars`.
    /// - All other [`ConstructionError`] variants from the post-
    ///   substitution YAML parse and schema validation.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let yaml = include_str!("projection.yaml"); // contains `{workspace}`
    /// let builder = Builder::from_yaml_with_vars(
    ///     yaml,
    ///     &[("workspace", "candle")],
    /// )?;
    /// ```
    pub fn from_yaml_with_vars(
        yaml_template: &str,
        vars: &[(&str, &str)],
    ) -> Result<Self, ConstructionError> {
        Self::from_yaml_with_vars_and_dialect(yaml_template, vars, None)
    }

    /// Full-fat parser: substitutes `{name}` placeholders, parses YAML,
    /// validates, and resolves any `kind: template` items against `dialect`.
    ///
    /// When `dialect` is `None`, any `kind: template` item in the YAML
    /// produces [`ConstructionError::DialectRequired`]. Use this entry
    /// point whenever the schema's `items:` list mixes content sections with
    /// dialect-template references.
    pub fn from_yaml_with_vars_and_dialect(
        yaml_template: &str,
        vars: &[(&str, &str)],
        dialect: Option<&candle_transformers::models::dialect::Dialect>,
    ) -> Result<Self, ConstructionError> {
        let resolved = substitute_template(yaml_template, vars)?;
        let (schema, maps) = from_yaml(&resolved, dialect)?;
        validate(&schema)?;
        Ok(Self {
            schema,
            name_maps: maps,
            source_yaml: Some(resolved),
        })
    }

    /// Pre-tokenise every `is_template` section's content with the
    /// caller-supplied tokenizer.
    ///
    /// Walks the schema, finds every [`SectionSchema`] whose
    /// `is_template` flag is set, calls the tokenize closure to convert
    /// its content string to token IDs, and stores the result in
    /// `template_tokens`.  Must be called once before the first
    /// `project()` if the schema contains any template items —
    /// otherwise the projection engine has no tokens to emit in the
    /// `Generated` segments it would produce for those items.
    ///
    /// The closure form keeps this crate tokenizer-agnostic;
    /// `candle-conversation` doesn't pull in a tokenizer dependency.
    /// Callers typically pass a closure that wraps their
    /// `tokenizers::Tokenizer::encode`.
    pub fn tokenize_templates<E, F>(&mut self, mut tokenize: F) -> Result<(), E>
    where
        F: FnMut(&str) -> Result<Vec<u32>, E>,
    {
        for item in self.schema.system_prompt.items.iter_mut() {
            match item {
                SystemPromptItem::Section(s) => {
                    if s.is_template && s.template_tokens.is_none() {
                        s.template_tokens = Some(std::sync::Arc::new(tokenize(&s.content)?));
                    }
                }
                SystemPromptItem::Collection(c) => {
                    for s in c.sections.iter_mut() {
                        if s.is_template && s.template_tokens.is_none() {
                            s.template_tokens = Some(std::sync::Arc::new(tokenize(&s.content)?));
                        }
                    }
                    if !c.member_glue.is_empty() && c.member_glue_tokens.is_none() {
                        c.member_glue_tokens = Some(std::sync::Arc::new(tokenize(&c.member_glue)?));
                    }
                }
                // Tree-node sections are sealed content, but a `glue` node
                // (e.g. `<tools>`) is a live-prefilled structural marker, and
                // an embedded collection's `member_glue` is one too — pre-tokenise
                // both here.
                SystemPromptItem::SectionTree(tree) => {
                    for node in tree.nodes.iter_mut() {
                        let need_glue = node.glue.as_ref().is_some_and(|g| g.tokens.is_none());
                        if need_glue {
                            let content = node.options[0].content.clone();
                            let toks = std::sync::Arc::new(tokenize(&content)?);
                            node.glue.as_mut().unwrap().tokens = Some(toks);
                        }
                        if let Some(tc) = node.collection.as_mut() {
                            let c = &mut tc.collection;
                            if !c.member_glue.is_empty() && c.member_glue_tokens.is_none() {
                                c.member_glue_tokens =
                                    Some(std::sync::Arc::new(tokenize(&c.member_glue)?));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Use a pre-built [`Schema`] (e.g. constructed programmatically).
    ///
    /// Construction validation runs identically to [`Builder::from_yaml`].
    /// Name-based lookups (`id_for_layer` etc.) will return `None` since
    /// no YAML string→id maps are available.
    pub fn from_schema(schema: Schema) -> Result<Self, ConstructionError> {
        validate(&schema)?;
        Ok(Self {
            schema,
            name_maps: NameMaps::default(),
            source_yaml: None,
        })
    }

    // ── Schema accessors ──────────────────────────────────────────────────────

    /// The resolved YAML this builder was parsed from, if any — the
    /// projection `Template` the substrate persists. `None` for
    /// programmatically-built schemas.
    pub fn source_yaml(&self) -> Option<&str> {
        self.source_yaml.as_deref()
    }

    /// Direct access to the underlying [`Schema`].
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Look up a layer by id. Returns `None` only if the id was issued by
    /// a different builder.
    pub fn layer(&self, id: LayerId) -> Option<&LayerSchema> {
        self.schema.layers.iter().find(|l| l.id == id)
    }

    /// Look up a group by its globally-unique id.
    pub fn group(&self, id: GroupId) -> Option<&GroupSchema> {
        self.schema
            .layers
            .iter()
            .flat_map(|l| l.groups.iter())
            .find(|g| g.id == id)
    }

    /// Look up a system-prompt section by id (top-level sections + sections
    /// nested in collections of the shared prompt).
    pub fn section(&self, id: SectionId) -> Option<&SectionSchema> {
        self.schema
            .system_prompt
            .all_sections()
            .find(|s| s.id == id)
    }

    // ── Name-based lookups (YAML string → id) ─────────────────────────────────

    /// Resolve a layer's YAML `name:` string to its id.
    /// Returns `None` for unknown names or when constructed via
    /// [`Builder::from_schema`] (which has no name maps).
    pub fn id_for_layer(&self, name: &str) -> Option<LayerId> {
        self.name_maps.layer_names.get(name).copied()
    }

    /// Resolve a group's YAML `id:` string to its id.
    pub fn id_for_group(&self, name: &str) -> Option<GroupId> {
        self.name_maps.group_names.get(name).copied()
    }

    /// Resolve a shared system-prompt section's YAML `id:` string to its id.
    pub fn id_for_system_section(&self, name: &str) -> Option<SectionId> {
        self.name_maps
            .section_names
            .get(&(LayerId::system_prompt(), name.to_string()))
            .copied()
    }

    /// Resolve a shared system-prompt collection's name to its id.
    pub fn id_for_system_collection(&self, name: &str) -> Option<CollectionId> {
        self.name_maps
            .collection_names
            .get(&(LayerId::system_prompt(), name.to_string()))
            .copied()
    }

    /// Override a collection's selection rule after construction.
    ///
    /// Used to force a collection (e.g. `tools`) to [`SelectionRule::AlwaysVisible`]
    /// for a high-resolution capture conversation, so projection and reprojection
    /// stop filtering its members. The schema is otherwise identical (same section
    /// ids), so the override can be applied to a clone shared with already-sealed
    /// sequences.
    pub fn set_collection_selection(
        &mut self,
        name: &str,
        selection: SelectionRule,
    ) -> Result<(), ConstructionError> {
        validate_selection(name, &selection)?;
        let items = &mut self.schema.system_prompt.items;
        match locate_collection(items, |c| c.name == name) {
            Some(CollLoc::TopLevel(ii)) => {
                if let SystemPromptItem::Collection(coll) = &mut items[ii] {
                    coll.selection = selection;
                }
                Ok(())
            }
            Some(CollLoc::Tree { item, node }) => {
                if let SystemPromptItem::SectionTree(t) = &mut items[item] {
                    if let Some(tc) = t.nodes[node].collection.as_mut() {
                        tc.collection.selection = selection;
                    }
                }
                Ok(())
            }
            None => Err(ConstructionError::UnknownCollection(name.to_string())),
        }
    }

    /// Restrict a collection to a single named member and force it
    /// [`SelectionRule::AlwaysVisible`] — projection + reprojection emit exactly
    /// that one section and drop the rest. Used to test whether the model
    /// invokes a tool when only its own definition is in context (vs the full
    /// catalog). The dropped sections keep their sealed KV in the substrate;
    /// they are simply not projected for this conversation.
    pub fn set_collection_single_section(
        &mut self,
        collection: &str,
        section: &str,
    ) -> Result<(), ConstructionError> {
        let keep: std::collections::HashSet<String> =
            std::iter::once(section.to_string()).collect();
        let items = &mut self.schema.system_prompt.items;
        match locate_collection(items, |c| c.name == collection) {
            Some(CollLoc::TopLevel(ii)) => {
                let SystemPromptItem::Collection(coll) = &mut items[ii] else {
                    unreachable!()
                };
                coll.sections.retain(|s| s.name == section);
                if coll.sections.is_empty() {
                    return Err(ConstructionError::UnknownSection(section.to_string()));
                }
                coll.selection = SelectionRule::AlwaysVisible;
                Ok(())
            }
            Some(CollLoc::Tree { item, node }) => {
                let SystemPromptItem::SectionTree(t) = &mut items[item] else {
                    unreachable!()
                };
                let tc = t.nodes[node].collection.as_mut().expect("collection node");
                retain_tree_members(tc, &keep);
                if tc.collection.sections.is_empty() {
                    return Err(ConstructionError::UnknownSection(section.to_string()));
                }
                tc.collection.selection = SelectionRule::AlwaysVisible;
                Ok(())
            }
            None => Err(ConstructionError::UnknownCollection(collection.to_string())),
        }
    }

    /// Retain only the collection members whose name is in `keep`, dropping the
    /// rest, and keep the collection's existing selection rule. Unlike
    /// [`Self::set_collection_single_section`] this allows the collection to end
    /// up **empty** (when `keep` is empty) — that is how "None" tools mode drops
    /// the whole catalog: the `<tools>`/`</tools>` wrapper sections `depends_on`
    /// the collection, so they stop emitting once it materialises nothing.
    /// "Restricted" mode passes the safe-tool names, leaving the top-k rule to
    /// surface the most relevant safe tools. The dropped members keep their
    /// sealed K/V in the substrate; they are simply not projected.
    pub fn retain_collection_sections(
        &mut self,
        collection: &str,
        keep: &std::collections::HashSet<String>,
    ) -> Result<(), ConstructionError> {
        let items = &mut self.schema.system_prompt.items;
        match locate_collection(items, |c| c.name == collection) {
            Some(CollLoc::TopLevel(ii)) => {
                if let SystemPromptItem::Collection(coll) = &mut items[ii] {
                    coll.sections.retain(|s| keep.contains(&s.name));
                }
                Ok(())
            }
            Some(CollLoc::Tree { item, node }) => {
                if let SystemPromptItem::SectionTree(t) = &mut items[item] {
                    if let Some(tc) = t.nodes[node].collection.as_mut() {
                        retain_tree_members(tc, keep);
                    }
                }
                Ok(())
            }
            None => Err(ConstructionError::UnknownCollection(collection.to_string())),
        }
    }

    // ── Runtime mutators ─────────────────────────────────────────────────────
    //
    // These mutate the schema after YAML parse, before the first
    // `project()`.  They're how dynamic catalogs (tool definitions,
    // retrieval candidates) get injected without baking 1000s of lines
    // of generated content into YAML.

    /// Append a top-level always-emit section to a layer's
    /// system_prompt, returning its newly-allocated [`SectionId`].
    ///
    /// This is the simple form: no selection filter, no collection.
    /// Use [`Self::add_collection`] + [`Self::add_section_to_collection`]
    /// when the new section belongs to a selectable subset.
    ///
    /// # Constraints
    ///
    /// - Must be called before the first `project()` — the schema is
    ///   considered immutable from that point so prefilled KV stays
    ///   correlated to section ids.
    /// - The section's `name` must be unique within its layer
    ///   (top-level + nested combined).
    /// - `priority` must be `> 0`.
    pub fn add_section(
        &mut self,
        name: impl Into<String>,
        content: impl Into<String>,
        priority: f32,
    ) -> Result<SectionId, ConstructionError> {
        let name: String = name.into();
        let content: String = content.into();
        if priority <= 0.0 {
            return Err(ConstructionError::InvalidPriority {
                name: name.clone(),
                value: priority,
            });
        }
        self.assert_section_name_free(&name)?;
        let new_id = SectionId::new(self.next_section_id_raw());
        self.schema
            .system_prompt
            .items
            .push(SystemPromptItem::Section(SectionSchema {
                id: new_id,
                name: name.clone(),
                content,
                priority,
                depends_on: None,
                depends_on_absent: None,
                is_template: false,
                template_tokens: None,
            }));
        self.name_maps
            .section_names
            .insert((LayerId::system_prompt(), name), new_id);
        Ok(new_id)
    }

    /// Append a [`SectionCollection`] to a layer's system_prompt.
    ///
    /// Returns the collection's id; sections can then be added to it
    /// via [`Self::add_section_to_collection`].
    ///
    /// `selection` controls how the collection's contents are filtered
    /// at projection time — `TopK { k }` keeps the K highest-scored,
    /// `Single` keeps the one highest, `AlwaysVisible` keeps all.
    pub fn add_collection(
        &mut self,
        name: impl Into<String>,
        selection: SelectionRule,
        score_threshold: f32,
    ) -> Result<CollectionId, ConstructionError> {
        let name: String = name.into();
        if self
            .name_maps
            .collection_names
            .contains_key(&(LayerId::system_prompt(), name.clone()))
        {
            return Err(ConstructionError::DuplicateCollectionName(name));
        }
        if score_threshold < 0.0 {
            return Err(ConstructionError::NegativeScoreThreshold {
                name: name.clone(),
                value: score_threshold,
            });
        }
        let new_id = CollectionId::new(self.next_collection_id_raw());
        // Derive the belief policy from a `top_k` rule (budget max = k,
        // min/evict = threshold) so a runtime-added collection selects like the
        // old top_k; other rules keep the default policy.
        let policy = match &selection {
            SelectionRule::TopK { k } => {
                let mut config = SelectionPolicy::default_policy().config;
                config.min_score = score_threshold;
                config.evict_score = score_threshold;
                config.budget_min = 0;
                config.budget_max = *k;
                SelectionPolicy {
                    config,
                    tags: Vec::new(),
                    layer_weights: Vec::new(),
                }
            }
            _ => SelectionPolicy::default_policy(),
        };
        self.schema
            .system_prompt
            .items
            .push(SystemPromptItem::Collection(SectionCollection {
                id: new_id,
                name: name.clone(),
                sections: Vec::new(),
                selection,
                score_threshold,
                policy,
                // Runtime-added collections carry a placeholder summary; no
                // compression path reads a group summary.
                summary: GroupSummary::default(),
                summary_section: None,
                member_glue: String::new(),
                member_glue_tokens: None,
                default: None,
            }));
        self.name_maps
            .collection_names
            .insert((LayerId::system_prompt(), name), new_id);
        Ok(new_id)
    }

    /// Append a section to an existing collection in a layer,
    /// returning its newly-allocated [`SectionId`].
    ///
    /// Sections in a collection are filtered by the collection's
    /// selection rule at projection time.  Names must still be unique
    /// across the whole layer (top-level + nested combined), since the
    /// substrate keys per-section state by `(LayerId, name)` and
    /// downstream prefill captures sigs by section id.
    pub fn add_section_to_collection(
        &mut self,
        collection: CollectionId,
        name: impl Into<String>,
        content: impl Into<String>,
        priority: f32,
    ) -> Result<SectionId, ConstructionError> {
        let name: String = name.into();
        let content: String = content.into();
        if priority <= 0.0 {
            return Err(ConstructionError::InvalidPriority {
                name: name.clone(),
                value: priority,
            });
        }
        self.assert_section_name_free(&name)?;
        // Locate the collection (top-level OR tree-embedded) before any mutation.
        let items = &self.schema.system_prompt.items;
        let loc = locate_collection(items, |c| c.id == collection).ok_or_else(|| {
            ConstructionError::UnknownCollection(format!("CollectionId({collection:?})"))
        })?;
        // Allocate id(s) while only holding an immutable borrow.  A tree-embedded
        // collection seals each member ×branch, so it takes one CONTIGUOUS id per
        // branch (above the current max — guaranteed disjoint).
        let base = self.next_section_id_raw();
        let items = &mut self.schema.system_prompt.items;
        let new_id = match loc {
            CollLoc::TopLevel(ii) => {
                let SystemPromptItem::Collection(coll) = &mut items[ii] else {
                    unreachable!("located a top-level collection")
                };
                let id = SectionId::new(base);
                coll.sections.push(SectionSchema {
                    id,
                    name: name.clone(),
                    content,
                    priority,
                    depends_on: None,
                    depends_on_absent: None,
                    is_template: false,
                    template_tokens: None,
                });
                id
            }
            CollLoc::Tree { item, node } => {
                let SystemPromptItem::SectionTree(t) = &mut items[item] else {
                    unreachable!("located a section tree")
                };
                let tc = t.nodes[node]
                    .collection
                    .as_mut()
                    .expect("located a collection node");
                // One sealed variant per branch, ids `base..base+n` contiguous.
                let variants: Vec<TreeVariant> = tc
                    .branches
                    .iter()
                    .enumerate()
                    .map(|(bi, (ancestors, prefix))| TreeVariant {
                        ancestors: *ancestors,
                        id: SectionId::new(base + bi as u32),
                        in_tree_prefix: prefix.clone(),
                    })
                    .collect();
                // The canonical (default-branch) id backs scoring / gating / name
                // resolution — the same role the build-time path assigns.
                let canonical = variants
                    .iter()
                    .find(|v| v.ancestors == tc.default_branch)
                    .map(|v| v.id)
                    .unwrap_or_else(|| variants[0].id);
                tc.collection.sections.push(SectionSchema {
                    id: canonical,
                    name: name.clone(),
                    content,
                    priority,
                    depends_on: None,
                    depends_on_absent: None,
                    is_template: false,
                    template_tokens: None,
                });
                tc.variants.push(variants);
                canonical
            }
        };
        self.name_maps
            .section_names
            .insert((LayerId::system_prompt(), name), new_id);
        Ok(new_id)
    }

    /// Associate a runtime-sealed summary section with a collection. Once set,
    /// projection emits that section's K/V just before the collection's selected
    /// members whenever the selection is a proper subset (top-k / threshold
    /// dropped at least one member) and the section is sealed. See
    /// [`SectionCollection::summary_section`].
    pub fn set_collection_summary_section(
        &mut self,
        collection: CollectionId,
        section: SectionId,
    ) -> Result<(), ConstructionError> {
        let items = &mut self.schema.system_prompt.items;
        match locate_collection(items, |c| c.id == collection) {
            Some(CollLoc::TopLevel(ii)) => {
                if let SystemPromptItem::Collection(coll) = &mut items[ii] {
                    coll.summary_section = Some(section);
                }
                Ok(())
            }
            Some(CollLoc::Tree { item, node }) => {
                if let SystemPromptItem::SectionTree(t) = &mut items[item] {
                    if let Some(tc) = t.nodes[node].collection.as_mut() {
                        tc.collection.summary_section = Some(section);
                    }
                }
                Ok(())
            }
            None => Err(ConstructionError::UnknownCollection(format!(
                "CollectionId({collection:?})"
            ))),
        }
    }

    /// Replace a mandatory tree section's content BEFORE prefill, so its sealed
    /// variants carry runtime-generated text (e.g. the tool-catalog summary,
    /// resolved deterministically once the catalog is installed).  The variant
    /// ids and in-tree prefixes were fixed at build time; only the content the
    /// prefill seals changes — so nodes below still anchor on this section's K/V,
    /// keeping the chain intact.  The node must already exist with non-empty
    /// authored content (so its per-branch variants were allocated).  Errors if
    /// no such single-option tree section is found in the layer.
    pub fn set_tree_section_content(
        &mut self,
        name: &str,
        content: impl Into<String>,
    ) -> Result<(), ConstructionError> {
        let content = content.into();
        for item in self.schema.system_prompt.items.iter_mut() {
            if let SystemPromptItem::SectionTree(t) = item {
                for node in t.nodes.iter_mut() {
                    // A glue node also has `collection: None` + one option, but its
                    // emitted tokens come from `glue.tokens` (tokenised separately) —
                    // rewriting its content here would desync label vs output, so
                    // reject it.
                    if node.name == name
                        && node.collection.is_none()
                        && node.glue.is_none()
                        && node.options.len() == 1
                    {
                        node.options[0].content = content;
                        return Ok(());
                    }
                }
            }
        }
        Err(ConstructionError::UnknownSection(name.to_string()))
    }

    /// Reject a duplicate section name in the shared system prompt (top-level OR
    /// nested in a collection or section-tree).
    fn assert_section_name_free(&self, name: &str) -> Result<(), ConstructionError> {
        for it in &self.schema.system_prompt.items {
            match it {
                SystemPromptItem::Section(s) if s.name == name => {
                    return Err(ConstructionError::DuplicateSectionName(name.to_string()));
                }
                SystemPromptItem::Section(_) => {}
                SystemPromptItem::Collection(c) => {
                    if c.sections.iter().any(|s| s.name == name) {
                        return Err(ConstructionError::DuplicateSectionName(name.to_string()));
                    }
                }
                SystemPromptItem::SectionTree(t) => {
                    let clash = t.nodes.iter().any(|n| {
                        n.name == name
                            || n.collection.as_ref().is_some_and(|tc| {
                                tc.collection.sections.iter().any(|s| s.name == name)
                            })
                    });
                    if clash {
                        return Err(ConstructionError::DuplicateSectionName(name.to_string()));
                    }
                }
            }
        }
        Ok(())
    }

    /// Highest currently-allocated section-id across the whole schema, + 1.
    fn next_section_id_raw(&self) -> u32 {
        // Must cover EVERY allocated section id, including the compression-prompt
        // sections that live in `layer.summary` / collection summaries rather
        // than `system_prompt.items` — otherwise a runtime-added section aliases
        // a compression-prompt id and corrupts the compression frame.
        let max_id = self.schema.all_section_ids().into_iter().max().unwrap_or(0);
        max_id + 1
    }

    /// Highest currently-allocated collection-id across the whole
    /// schema, + 1.
    fn next_collection_id_raw(&self) -> u32 {
        let max_id = self
            .schema
            .system_prompt
            .items
            .iter()
            .filter_map(|it| match it {
                SystemPromptItem::Collection(c) => Some(c.id.raw()),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        max_id + 1
    }

    // ── Projection ────────────────────────────────────────────────────────────

    /// Run the full projection pipeline for the given `target`.
    ///
    /// Uses [`ProjectionMode::Decode`] — the steady-state mode for continuous
    /// reprojection during decode.  Call [`Self::project_with_mode`] with
    /// [`ProjectionMode::Prefill`] for the pre-decode initial-guess pass.
    ///
    /// See [`super::project`] for the 12-step pipeline. Pure given a valid
    /// schema and a working resolver — never errors.
    pub fn project<R: ContentResolver>(
        &self,
        target: ProjectionTarget,
        resolver: &R,
    ) -> Projection {
        run(
            &self.schema,
            target,
            resolver,
            ProjectionMode::Decode,
            &SelectionState::default(),
        )
    }

    /// Projection with an explicit section-tree [`SelectionState`] — the entry
    /// the runtime uses to choose selector options (e.g. the thinking-effort /
    /// response-length dials).  An empty state reproduces the authored defaults.
    pub fn project_with_selection<R: ContentResolver>(
        &self,
        target: ProjectionTarget,
        resolver: &R,
        mode: ProjectionMode,
        selection: &SelectionState,
    ) -> Projection {
        run(&self.schema, target, resolver, mode, selection)
    }

    /// Run the projection pipeline for an explicit [`ProjectionMode`].
    ///
    /// `Prefill` mode scores collections with the calibrated prefill profile
    /// (`Max` formula, semantic depth, no score threshold) so a useful
    /// initial section guess can be made before any decode probe runs.
    pub fn project_with_mode<R: ContentResolver>(
        &self,
        target: ProjectionTarget,
        resolver: &R,
        mode: ProjectionMode,
        selection: &SelectionState,
    ) -> Projection {
        run(&self.schema, target, resolver, mode, selection)
    }

    /// Variant of [`Self::project_with_mode`] that delivers score-density
    /// [`SelectionDiagnostics`] to a caller-supplied sink.  The scheduler
    /// uses this to ferry the diagnostic into the substrate's
    /// last-selection side-channel without polluting [`Projection`] with
    /// a test-only field.  When the projection used the rule-based path
    /// (no summary tree for the target timeline), the sink is never
    /// invoked.
    pub fn project_with_mode_and_sink<R: ContentResolver>(
        &self,
        target: ProjectionTarget,
        resolver: &R,
        mode: ProjectionMode,
        selection: &SelectionState,
        prior: &PriorBelief,
        decode_pos: Option<usize>,
        sink: &mut dyn FnMut(SelectionDiagnostics),
    ) -> Projection {
        run_with_sink(
            &self.schema,
            target,
            resolver,
            mode,
            selection,
            prior,
            decode_pos,
            sink,
        )
    }

    /// Token-window budget configured for `layer`.  Used by the
    /// scheduler to populate `SelectionDiagnostics.budget`.  Returns
    /// `None` when the layer id isn't in the schema.
    pub fn schema_layer_window(&self, layer: LayerId) -> Option<usize> {
        self.schema
            .layers
            .iter()
            .find(|l| l.id == layer)
            .map(|l| l.window)
    }

    // ── Synthetic construction ────────────────────────────────────────────────

    /// Build a minimal single-layer schema around a pre-rendered system-prompt
    /// string, for legacy callers that provide a plain `&str` rather than a
    /// YAML schema.
    ///
    /// The resulting builder has one `dialogue` layer (32 768-token window),
    /// one `primary_conversation` group with `AlwaysVisible` selection (all
    /// turns pass through — budget trimming is left to the existing sliding-
    /// window config downstream), and the system-prompt text as a single
    /// `frame` section.
    ///
    /// IDs are fixed at 1 each; there is no name map.
    ///
    pub fn for_plain_prompt(system_prompt_text: &str) -> Self {
        Self::synthetic_single_section(
            system_prompt_text,
            LayerId::new(1),
            GroupId::new(1),
            SectionId::new(1),
        )
    }

    /// Same as [`Self::for_plain_prompt`] but allocates the synthetic
    /// schema's ids from the [`Reserved`] kind's slot at the top of the
    /// u32 range. Use this for engine-internal conversations (the
    /// daemon's titler) that must coexist in the same substrate as a
    /// YAML schema without colliding.
    ///
    /// [`Reserved`]: super::Reserved
    pub fn for_plain_prompt_reserved(system_prompt_text: &str, kind: Reserved) -> Self {
        Self::synthetic_single_section(
            system_prompt_text,
            LayerId::reserved(kind),
            GroupId::reserved(kind),
            SectionId::reserved(kind),
        )
    }

    /// Like [`Self::for_plain_prompt_reserved`], but bases the synthetic schema's
    /// **section** ids at `section_base` instead of the single reserved-slot id.
    /// The layer and group still occupy the reserved slot — the conversation's
    /// turns never enter a user projection — but the frame plus any sections later
    /// added via [`Self::add_collection`] / [`Self::add_section_to_collection`]
    /// allocate upward from `section_base`, occupying a dedicated id partition.
    ///
    /// This is for engine-internal conversations that hold a whole *corpus* of
    /// sections (more than the one-per-kind the [`Reserved`] band at the top of
    /// u32 allows). The partition keeps every section id clear of the user
    /// schema's low `1..n` ids, the tool ids allocated above them, and the
    /// reserved band — however much the user schema grows — since the substrate
    /// keys section state globally by id.
    ///
    /// [`Reserved`]: super::Reserved
    pub fn for_reserved_corpus(
        system_prompt_text: &str,
        kind: Reserved,
        section_base: u32,
    ) -> Self {
        Self::synthetic_single_section(
            system_prompt_text,
            LayerId::reserved(kind),
            GroupId::reserved(kind),
            SectionId::new(section_base),
        )
    }

    /// Shared body of [`Self::for_plain_prompt`] /
    /// [`Self::for_plain_prompt_reserved`] — a single-layer / single-group
    /// schema with one `"frame"` section in the system prompt, using
    /// caller-supplied ids.
    fn synthetic_single_section(
        system_prompt_text: &str,
        layer_id: LayerId,
        group_id: GroupId,
        section_id: SectionId,
    ) -> Self {
        // Id for the summary's answer framing section (never emitted; sealed
        // lazily only if this conversation is summarised). Offset from the frame,
        // special-cased so a reserved frame (top of the u32 range) keeps the id in
        // the reserved band. There is no question-half section: a summary's user
        // half is derived, never decoded, so it has no prompt to frame.
        let summary_a_id = if section_id.raw() == u32::MAX {
            SectionId::new(u32::MAX - 1)
        } else {
            SectionId::new(section_id.raw() + 1)
        };
        let schema = Schema {
            system_prompt: SystemPromptSchema {
                items: vec![SystemPromptItem::Section(SectionSchema {
                    id: section_id,
                    name: "frame".to_string(),
                    content: system_prompt_text.to_string(),
                    priority: 50.0,
                    depends_on: None,
                    depends_on_absent: None,
                    is_template: false,
                    template_tokens: None,
                })],
            },
            layers: vec![LayerSchema {
                id: layer_id,
                name: "dialogue".to_string(),
                description: String::new(),
                score_threshold: 0.0,
                window: 32_768,
                budget: Budget {
                    priority: 100.0,
                    min_percent: None,
                    max_percent: None,
                },
                dials: LayerDials::default(),
                // Template-less fallback summary so a plain-prompt conversation
                // is still summarisable (the production path supplies its own
                // tailored summary via the YAML template). Generic faithful
                // compression — not a product prompt. Only `turns` is provided;
                // SummaryOfSummaries nodes reuse it.
                summary: LayerSummary {
                    turns: TurnSummary {
                        scope: Scope::Union,
                        assistant: CompressionPrompt {
                            system_prompt: SectionSchema {
                                id: summary_a_id,
                                name: "__summary_turns_assistant__".to_string(),
                                content: "You are a faithful compressor. Compress the assistant's \
                                          reply below into a much shorter version that preserves \
                                          the specific facts, numbers, decisions, and advice, in \
                                          the original voice. No commentary, headings, or lists."
                                    .to_string(),
                                priority: 50.0,
                                depends_on: None,
                                depends_on_absent: None,
                                is_template: false,
                                template_tokens: None,
                            },
                            user_prompt:
                                "Compress the assistant reply above into a faithful, much \
                                          shorter version that preserves the specific facts, \
                                          numbers, decisions, and advice, in the original voice."
                                    .to_string(),
                        },
                        max_tokens: 384,
                        content: Content::Decode,
                    },
                    summaries: None,
                },
                groups: vec![GroupSchema {
                    id: group_id,
                    name: "primary_conversation".to_string(),
                    selection: SelectionRule::AlwaysVisible,
                    score_threshold: 0.0,
                    policy: SelectionPolicy::default_policy(),
                    budget: Budget {
                        priority: 100.0,
                        min_percent: None,
                        max_percent: None,
                    },
                    default: None,
                }],
                policy: SelectionPolicy::default_policy(),
                gather_scope: GatherScope::default(),
                // This synthetic fallback layer IS the dialogue layer, so it takes
                // the interactive decode priority the production dialogue layer does.
                decode_priority: DecodePriority::High,
            }],
        };
        validate(&schema).expect("synthetic schema must always be valid");

        // Populate name maps so `id_for_layer` / `id_for_group` /
        // `id_for_section_in` work the same as for YAML-derived builders.
        let mut name_maps = NameMaps::default();
        name_maps
            .layer_names
            .insert("dialogue".to_string(), layer_id);
        name_maps
            .group_names
            .insert("primary_conversation".to_string(), group_id);
        name_maps
            .section_names
            .insert((LayerId::system_prompt(), "frame".to_string()), section_id);

        // A synthetic single-section schema, not parsed from YAML.
        Self {
            schema,
            name_maps,
            source_yaml: None,
        }
    }
}

/// Where a collection lives in a layer's system prompt — a top-level item or a
/// node inside a section tree.  Returned by [`locate_collection`] so the runtime
/// mutators re-borrow the located item mutably by index without fighting the
/// borrow checker.
enum CollLoc {
    TopLevel(usize),
    Tree { item: usize, node: usize },
}

/// Find the first collection in `items` matching `matches`, descending into
/// section-tree collection nodes.  Pure (immutable) index lookup — the single
/// place the runtime mutators resolve a collection, so a tool catalog installed
/// into a tree-embedded `tools` collection is found exactly like a top-level one.
fn locate_collection(
    items: &[SystemPromptItem],
    mut matches: impl FnMut(&SectionCollection) -> bool,
) -> Option<CollLoc> {
    for (ii, item) in items.iter().enumerate() {
        match item {
            SystemPromptItem::Collection(c) if matches(c) => return Some(CollLoc::TopLevel(ii)),
            SystemPromptItem::SectionTree(t) => {
                for (ni, node) in t.nodes.iter().enumerate() {
                    if let Some(tc) = &node.collection {
                        if matches(&tc.collection) {
                            return Some(CollLoc::Tree { item: ii, node: ni });
                        }
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Parallel-retain a tree collection's members (canonical schemas + per-branch
/// variant lists stay index-aligned) by member name.
fn retain_tree_members(tc: &mut TreeCollection, keep: &std::collections::HashSet<String>) {
    let keepers: Vec<bool> = tc
        .collection
        .sections
        .iter()
        .map(|s| keep.contains(&s.name))
        .collect();
    let mut i = 0;
    tc.collection.sections.retain(|_| {
        let k = keepers[i];
        i += 1;
        k
    });
    let mut i = 0;
    tc.variants.retain(|_| {
        let k = keepers[i];
        i += 1;
        k
    });
}
