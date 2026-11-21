//! Builds a `Builder` from YAML or a schema, and checks it before any projection
//! runs.
//!
//! Parsing substitutes `{name}` placeholders exactly once, so the resulting schema
//! is immutable and its prefilled prompt stays stable across every projection made
//! from it. Construction validation checks budget shares, sibling minimums and
//! selection ranges, so a schema that passes validation projects without error.
//! Template sections are tokenised by a caller-supplied closure, which keeps this
//! crate free of a tokenizer dependency.

use super::*;

/// Run all construction-time validation checks against a parsed schema.
///
/// Called by both [`Builder::from_yaml`] and [`Builder::from_schema`] so
/// programmatically-constructed schemas get the same guarantees as YAML
/// ones.
pub(super) fn validate(schema: &Schema) -> Result<(), ConstructionError> {
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

            if group.score_threshold.unwrap_or(0.0) < 0.0 {
                return Err(ConstructionError::NegativeScoreThreshold {
                    name: group.name.clone(),
                    value: group.score_threshold.unwrap_or(0.0),
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

pub(super) fn validate_budget_bounds(name: &str, budget: &Budget) -> Result<(), ConstructionError> {
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
pub(super) fn substitute_template(
    template: &str,
    vars: &[(&str, &str)],
) -> Result<String, ConstructionError> {
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

pub(super) fn validate_selection(
    name: &str,
    rule: &SelectionRule,
) -> Result<(), ConstructionError> {
    use SelectionRule;
    match rule {
        SelectionRule::TopK { k } | SelectionRule::Offered { k, .. } if *k == 0 => {
            Err(ConstructionError::InvalidTopK {
                name: name.to_string(),
            })
        }
        SelectionRule::Offered { selector, .. } if selector.trim().is_empty() => {
            Err(ConstructionError::EmptyNamedSelector {
                name: name.to_string(),
            })
        }
        SelectionRule::Sequence {
            recent,
            historical_top_k,
            ..
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
    /// let yaml = include_str!("../projection.yaml"); // contains `{workspace}`
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
            script: None,
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
            script: None,
        })
    }
}
