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

use candle_transformers::models::dialect::Dialect;

use super::error::ConstructionError;
use super::ids::{GroupId, LayerId, SectionId};
use super::project::{run, Projection, ProjectionMode, ProjectionTarget};
use super::schema::{GroupSchema, LayerSchema, Schema, SectionSchema};
use super::yaml::{from_yaml, NameMaps};
use crate::substrate::ContentResolver;

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

    for layer in &schema.layers {
        validate_budget_bounds(&layer.name, &layer.budget)?;

        // Every layer must declare at least one system-prompt item
        // (top-level section or collection).  Collections with zero
        // sections are tolerated only if the layer also has a real
        // top-level section — otherwise the layer can't render any
        // system prompt at all.
        if layer.system_prompt.items.is_empty()
            || layer.system_prompt.all_sections().next().is_none()
        {
            return Err(ConstructionError::EmptyLayerSystemPrompt {
                layer: layer.name.clone(),
            });
        }

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

fn validate_budget_bounds(
    name: &str,
    budget: &super::schema::Budget,
) -> Result<(), ConstructionError> {
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

fn validate_selection(
    name: &str,
    rule: &super::schema::SelectionRule,
) -> Result<(), ConstructionError> {
    use super::schema::SelectionRule;
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
        dialect: Option<&Dialect>,
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

    /// Look up a system-prompt section by id.
    ///
    /// Sections are scoped to a layer; this walks every layer's
    /// system-prompt items (top-level sections + sections nested in
    /// collections) to find the matching id.
    pub fn section(&self, id: SectionId) -> Option<&SectionSchema> {
        self.schema
            .layers
            .iter()
            .flat_map(|l| l.system_prompt.all_sections())
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

    /// Resolve a section's YAML `id:` string to its id, within a specific
    /// layer. Section names are layer-scoped — the same name may appear in
    /// multiple layers.
    pub fn id_for_section_in(&self, layer: LayerId, name: &str) -> Option<SectionId> {
        self.name_maps
            .section_names
            .get(&(layer, name.to_string()))
            .copied()
    }

    /// Resolve a collection's name within a layer to its id.
    /// Collections are layer-scoped (analogous to sections).
    pub fn id_for_collection_in(
        &self,
        layer: LayerId,
        name: &str,
    ) -> Option<super::ids::CollectionId> {
        self.name_maps
            .collection_names
            .get(&(layer, name.to_string()))
            .copied()
    }

    /// The span α used by [`super::project::FIXED_FORMULA`].
    ///
    /// The BDP scanner in the reprojection path must use the same alpha so
    /// scores accumulated during live decode are consistent with the scores
    /// the projection engine reads at group-scoring time (step 5 of the
    /// pipeline).  Returns `DEFAULT_SPAN_ALPHA` for non-Span formulas as a
    /// safe fallback.
    pub fn span_alpha(&self) -> f32 {
        match super::project::FIXED_FORMULA {
            super::schema::ScoreFormula::Span { alpha } => alpha,
            _ => crate::provenance::DEFAULT_SPAN_ALPHA,
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
        layer: LayerId,
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
        let layer_idx = self.layer_idx(layer)?;
        self.assert_section_name_free(layer_idx, &name)?;
        let new_id = SectionId::new(self.next_section_id_raw());
        self.schema.layers[layer_idx].system_prompt.items.push(
            super::schema::SystemPromptItem::Section(SectionSchema {
                id: new_id,
                name: name.clone(),
                content,
                priority,
                depends_on: None,
                is_template: false,
            }),
        );
        self.name_maps.section_names.insert((layer, name), new_id);
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
        layer: LayerId,
        name: impl Into<String>,
        selection: super::schema::SelectionRule,
        score_threshold: f32,
    ) -> Result<super::ids::CollectionId, ConstructionError> {
        let name: String = name.into();
        let layer_idx = self.layer_idx(layer)?;
        if self
            .name_maps
            .collection_names
            .contains_key(&(layer, name.clone()))
        {
            return Err(ConstructionError::DuplicateCollectionName(name));
        }
        if score_threshold < 0.0 {
            return Err(ConstructionError::NegativeScoreThreshold {
                name: name.clone(),
                value: score_threshold,
            });
        }
        let new_id = super::ids::CollectionId::new(self.next_collection_id_raw());
        self.schema.layers[layer_idx].system_prompt.items.push(
            super::schema::SystemPromptItem::Collection(super::schema::SectionCollection {
                id: new_id,
                name: name.clone(),
                sections: Vec::new(),
                selection,
                score_threshold,
                depth_weights: None,
            }),
        );
        self.name_maps
            .collection_names
            .insert((layer, name), new_id);
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
        layer: LayerId,
        collection: super::ids::CollectionId,
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
        let layer_idx = self.layer_idx(layer)?;
        self.assert_section_name_free(layer_idx, &name)?;
        // Validate the collection exists before any mutation.
        let coll_exists = self.schema.layers[layer_idx]
            .system_prompt
            .items
            .iter()
            .any(|it| {
                matches!(it,
                super::schema::SystemPromptItem::Collection(c) if c.id == collection)
            });
        if !coll_exists {
            return Err(ConstructionError::UnknownCollection(format!(
                "CollectionId({:?})",
                collection
            )));
        }
        // Allocate id while only holding an immutable borrow.
        let new_id = SectionId::new(self.next_section_id_raw());
        // Now take a fresh mutable borrow to push.
        let coll = self.schema.layers[layer_idx]
            .system_prompt
            .items
            .iter_mut()
            .find_map(|it| match it {
                super::schema::SystemPromptItem::Collection(c) if c.id == collection => Some(c),
                _ => None,
            })
            .expect("existence checked above");
        coll.sections.push(SectionSchema {
            id: new_id,
            name: name.clone(),
            content,
            priority,
            depends_on: None,
            is_template: false,
        });
        self.name_maps.section_names.insert((layer, name), new_id);
        Ok(new_id)
    }

    /// Install the dialect-specific structural markers — the bytes that
    /// open and close the system block — as two synthetic sections on
    /// `layer`. They live outside `system_prompt.items` (so they are not
    /// subject to selection rules), get fresh [`SectionId`]s, and register
    /// in `name_maps` under reserved names (`__system_start`,
    /// `__system_end`).
    ///
    /// Called by the dialect-aware caller (e.g. zend) after the schema
    /// has been built from YAML but before the conversation is created.
    /// Idempotent per layer — calling twice is an error.
    pub fn set_system_markers(
        &mut self,
        layer: LayerId,
        start_content: impl Into<String>,
        end_content: impl Into<String>,
    ) -> Result<(SectionId, SectionId), ConstructionError> {
        let layer_idx = self.layer_idx(layer)?;
        if self.schema.layers[layer_idx].system_start_section.is_some()
            || self.schema.layers[layer_idx].system_end_section.is_some()
        {
            return Err(ConstructionError::DuplicateSectionName(
                "__system_start/__system_end already set on this layer".to_string(),
            ));
        }
        const START_NAME: &str = "__system_start";
        const END_NAME: &str = "__system_end";
        self.assert_section_name_free(layer_idx, START_NAME)?;
        self.assert_section_name_free(layer_idx, END_NAME)?;

        let start_id = SectionId::new(self.next_section_id_raw());
        let end_id = SectionId::new(self.next_section_id_raw() + 1);
        self.schema.layers[layer_idx].system_start_section = Some(SectionSchema {
            id: start_id,
            name: START_NAME.to_string(),
            content: start_content.into(),
            priority: 100.0,
            depends_on: None,
            is_template: false,
        });
        self.schema.layers[layer_idx].system_end_section = Some(SectionSchema {
            id: end_id,
            name: END_NAME.to_string(),
            content: end_content.into(),
            priority: 100.0,
            depends_on: None,
            is_template: false,
        });
        self.name_maps
            .section_names
            .insert((layer, START_NAME.to_string()), start_id);
        self.name_maps
            .section_names
            .insert((layer, END_NAME.to_string()), end_id);
        Ok((start_id, end_id))
    }

    /// Look up a layer's index in `self.schema.layers` by id, returning
    /// an error if unknown.
    fn layer_idx(&self, layer: LayerId) -> Result<usize, ConstructionError> {
        self.schema
            .layers
            .iter()
            .position(|l| l.id == layer)
            .ok_or_else(|| ConstructionError::UnknownLayer(format!("LayerId({:?})", layer)))
    }

    /// Reject duplicate section names within a layer (top-level OR nested).
    fn assert_section_name_free(
        &self,
        layer_idx: usize,
        name: &str,
    ) -> Result<(), ConstructionError> {
        let layer = &self.schema.layers[layer_idx];
        for it in &layer.system_prompt.items {
            match it {
                super::schema::SystemPromptItem::Section(s) if s.name == name => {
                    return Err(ConstructionError::DuplicateSectionName(name.to_string()));
                }
                super::schema::SystemPromptItem::Collection(c) => {
                    if c.sections.iter().any(|s| s.name == name) {
                        return Err(ConstructionError::DuplicateSectionName(name.to_string()));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Highest currently-allocated section-id across the whole schema,
    /// + 1.  Sections are globally unique even though names are per-
    /// layer scoped.
    fn next_section_id_raw(&self) -> u32 {
        let max_id = self
            .schema
            .layers
            .iter()
            .flat_map(|l| {
                l.system_prompt
                    .all_sections()
                    .chain(l.system_start_section.iter())
                    .chain(l.system_end_section.iter())
            })
            .map(|s| s.id.raw())
            .max()
            .unwrap_or(0);
        max_id + 1
    }

    /// Highest currently-allocated collection-id across the whole
    /// schema, + 1.
    fn next_collection_id_raw(&self) -> u32 {
        let max_id = self
            .schema
            .layers
            .iter()
            .flat_map(|l| l.system_prompt.items.iter())
            .filter_map(|it| match it {
                super::schema::SystemPromptItem::Collection(c) => Some(c.id.raw()),
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
        run(&self.schema, target, resolver, ProjectionMode::Decode)
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
    ) -> Projection {
        run(&self.schema, target, resolver, mode)
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
            super::ids::LayerId::new(1),
            super::ids::GroupId::new(1),
            super::ids::SectionId::new(1),
        )
    }

    /// Same as [`Self::for_plain_prompt`] but allocates the synthetic
    /// schema's ids from the [`Reserved`] kind's slot at the top of the
    /// u32 range. Use this for engine-internal conversations (the
    /// daemon's titler) that must coexist in the same substrate as a
    /// YAML schema without colliding.
    ///
    /// [`Reserved`]: super::Reserved
    pub fn for_plain_prompt_reserved(system_prompt_text: &str, kind: super::ids::Reserved) -> Self {
        Self::synthetic_single_section(
            system_prompt_text,
            super::ids::LayerId::reserved(kind),
            super::ids::GroupId::reserved(kind),
            super::ids::SectionId::reserved(kind),
        )
    }

    /// Shared body of [`Self::for_plain_prompt`] /
    /// [`Self::for_plain_prompt_reserved`] — a single-layer / single-group
    /// schema with one `"frame"` section in the system prompt, using
    /// caller-supplied ids.
    fn synthetic_single_section(
        system_prompt_text: &str,
        layer_id: super::ids::LayerId,
        group_id: super::ids::GroupId,
        section_id: super::ids::SectionId,
    ) -> Self {
        use super::schema::{
            Budget, DepthWeights, GroupSchema, LayerSchema, Schema, SectionSchema, SelectionRule,
            SystemPromptSchema,
        };

        let schema = Schema {
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
                system_prompt: SystemPromptSchema {
                    items: vec![super::schema::SystemPromptItem::Section(SectionSchema {
                        id: section_id,
                        name: "frame".to_string(),
                        content: system_prompt_text.to_string(),
                        priority: 50.0,
                        depends_on: None,
                        is_template: false,
                    })],
                },
                groups: vec![GroupSchema {
                    id: group_id,
                    name: "primary_conversation".to_string(),
                    selection: SelectionRule::AlwaysVisible,
                    score_threshold: 0.0,
                    budget: Budget {
                        priority: 100.0,
                        min_percent: None,
                        max_percent: None,
                    },
                }],
                depth_weights: DepthWeights::default(),
                system_start_section: None,
                system_end_section: None,
            }],
        };
        validate(&schema).expect("synthetic schema must always be valid");

        // Populate name maps so `id_for_layer` / `id_for_group` /
        // `id_for_section_in` work the same as for YAML-derived builders.
        let mut name_maps = super::yaml::NameMaps::default();
        name_maps
            .layer_names
            .insert("dialogue".to_string(), layer_id);
        name_maps
            .group_names
            .insert("primary_conversation".to_string(), group_id);
        name_maps
            .section_names
            .insert((layer_id, "frame".to_string()), section_id);

        // A synthetic single-section schema, not parsed from YAML.
        Self {
            schema,
            name_maps,
            source_yaml: None,
        }
    }
}
