//! Builds the projection schema from the parsed YAML: the root policy, the shared
//! system prompt, the compression prompts and the turn summaries.
//!
//! The schema's default policy is the root of the inheritance chain, which layers
//! and collections inherit from. The one shared system prompt registers its
//! sections and collections under the reserved system-prompt id, not under any
//! layer. A collection with no explicit policy derives its belief budget from its
//! top-k rule. A band is declared once any level of the chain names a minimum
//! score. Compression framings and turn summaries are validated here, so an empty
//! prompt or a zero token ceiling is refused at build time.

use super::*;

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

pub(super) fn build(
    raw: YamlSchema,
    dialect: Option<&Dialect>,
) -> Result<(Schema, NameMaps), ConstructionError> {
    let mut maps = NameMaps::default();
    let mut section_alloc = SectionIdAlloc(0);
    let mut collection_alloc = CollectionIdAlloc(0);

    let mut layers = Vec::with_capacity(raw.layers.len());
    let layer_ids = persisted_ids::assign(
        &raw.layers
            .iter()
            .map(|l| l.persisted_id)
            .collect::<Vec<_>>(),
        "layer",
    )?;
    let group_ids = persisted_ids::assign(
        &raw.layers
            .iter()
            .flat_map(|l| l.groups.iter().map(|g| g.persisted_id))
            .collect::<Vec<_>>(),
        "group",
    )?;
    let mut group_ids = group_ids.into_iter();

    // Root of the policy inheritance chain: the schema `default_policy:` resolved
    // against the built-in default. Layers inherit this; collections/groups the layer.
    let schema_default = parse_policy(
        "default_policy",
        raw.default_policy.as_ref(),
        &SelectionPolicy::default_policy(),
    )?;

    // The single shared system prompt — built once, framed by every layer's
    // dials at projection time. Its sections/collections register under the
    // reserved system-prompt id, not any layer.
    let system_prompt = build_system_prompt(
        &raw.system_prompt,
        dialect,
        &schema_default,
        &mut section_alloc,
        &mut collection_alloc,
        &mut maps,
    )?;

    for (li, yl) in raw.layers.iter().enumerate() {
        let lid = LayerId::new(layer_ids[li]);
        maps.layer_names.insert(yl.name.clone(), lid);

        let layer_budget = parse_budget(&yl.name, &yl.budget)?;
        let layer_policy = parse_policy(&yl.name, yl.policy.as_ref(), &schema_default)?;
        // A band is "declared" once any level in the chain names a `min_score`.
        let layer_band_declared = yl.policy.as_ref().and_then(|p| p.min_score).is_some();

        // Per-layer dial overrides for the shared system prompt (applied when
        // this layer is the projection target). Empty inherits the section-tree
        // defaults.
        let dials = LayerDials::from_pairs(
            yl.dials
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        );

        // ── this layer's groups ──────────────────────────────────────────────
        let mut groups = Vec::with_capacity(yl.groups.len());
        for yg in &yl.groups {
            if maps.group_names.contains_key(&yg.id) {
                return Err(ConstructionError::DuplicateGroupName(yg.id.clone()));
            }
            let gid = GroupId::new(
                group_ids
                    .next()
                    .expect("persisted_ids::assign numbers every group"),
            );
            maps.group_names.insert(yg.id.clone(), gid);

            if let Some(t) = yg.score_threshold {
                if t < 0.0 {
                    return Err(ConstructionError::NegativeScoreThreshold {
                        name: yg.id.clone(),
                        value: t,
                    });
                }
            }

            let selection = parse_selection(&yg.id, &yg.selection)?;
            let group_budget = parse_budget(&yg.id, &yg.budget)?;
            let group_policy = parse_policy(&yg.id, yg.policy.as_ref(), &layer_policy)?;
            let policy_band_declared =
                layer_band_declared || yg.policy.as_ref().and_then(|p| p.min_score).is_some();
            let draws = parse_draws(&yg.id, yg.draws.as_ref(), &selection, &groups)?;
            let recency = parse_recency(&yg.id, &yg.selection, &selection)?;

            groups.push(GroupSchema {
                id: gid,
                name: yg.id.clone(),
                selection,
                score_threshold: yg.score_threshold,
                policy: group_policy,
                policy_band_declared,
                budget: group_budget,
                budget_adaptive: parse_member_budget_adaptive(&yg.id, yg.budget_adaptive.as_ref())?,
                locality: parse_locality(&yg.id, yg.locality.as_ref())?,
                anchor: parse_anchor(&yg.id, yg.anchor.as_ref())?,
                max_swaps: match yg.max_swaps {
                    None => DEFAULT_MAX_SWAPS,
                    Some(n) => {
                        NonZeroUsize::new(n).ok_or_else(|| ConstructionError::InvalidPolicy {
                            name: yg.id.clone(),
                            reason: "max_swaps must be >= 1: a cap of 0 admits no conversation, \
                                     ever"
                                .to_string(),
                        })?
                    }
                },
                draws,
                recency,
            });
        }

        if yl.score_threshold < 0.0 {
            return Err(ConstructionError::NegativeScoreThreshold {
                name: yl.name.clone(),
                value: yl.score_threshold,
            });
        }

        let layer_summary = build_layer_summary(&yl.name, &yl.summary, &mut section_alloc)?;

        layers.push(LayerSchema {
            id: lid,
            name: yl.name.clone(),
            description: yl.description.clone(),
            score_threshold: yl.score_threshold,
            window: yl.window,
            working_set: parse_working_set(&yl.name, yl.working_set.as_ref())?,
            budget: layer_budget,
            dials,
            summary: layer_summary,
            groups,
            policy: layer_policy,
            gather_scope: yl.gather_scope.into(),
            // Declared layers are gathered; `--disable-layer` clears this at
            // runtime via `Builder::set_layer_gathered`, never from YAML.
            gathered: true,
            decode_priority: yl.decode_priority.into(),
            on_corrupt_turn: yl.on_corrupt_turn.into(),
            ingest_unit: yl.ingest_unit.clone(),
            in_tool_rounds: yl.in_tool_rounds.unwrap_or(true),
            // Its declaration index unless it says otherwise — the stack every
            // schema had before a layer could declare its place in it.
            rank: yl.rank.unwrap_or(layers.len() as i32),
            ingest: yl.ingest.clone(),
            calibrate: yl.calibrate.clone(),
        });
    }

    let schema = Schema {
        layers,
        system_prompt,
        free_tool_calls_from_penalties: raw.free_tool_calls_from_penalties,
        hits: match raw.hits {
            Some(h) => h.resolve()?,
            None => HitsConfig::default(),
        },
        focus: false,
    };
    Ok((schema, maps))
}

/// Build the single shared [`SystemPromptSchema`] from the top-level
/// `system_prompt:` block. Section and collection names register under
/// [`LayerId::system_prompt`] (not any real layer, since the prompt is shared).
/// Two input forms are accepted, exactly as the former per-layer prompt: a legacy
/// flat `sections:` list (always-emit, head of the stream) and the modern
/// interleaved `items:` list (sections / templates / collections / section-trees,
/// where ordering matters). Collections with no explicit `policy:` inherit
/// `base_policy` (the schema default).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_system_prompt(
    raw_sp: &YamlSystemPrompt,
    dialect: Option<&Dialect>,
    base_policy: &SelectionPolicy,
    section_alloc: &mut SectionIdAlloc,
    collection_alloc: &mut CollectionIdAlloc,
    maps: &mut NameMaps,
) -> Result<SystemPromptSchema, ConstructionError> {
    let lid = LayerId::system_prompt();
    let owner = "system_prompt";
    let mut items: Vec<SystemPromptItem> = Vec::new();
    let mut section_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut collection_names: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Legacy `sections:` shortcut — emit before any items.
    for s in &raw_sp.sections {
        if !section_names.insert(s.id.clone()) {
            return Err(ConstructionError::DuplicateSectionName(s.id.clone()));
        }
        let sid = section_alloc.next();
        maps.section_names.insert((lid, s.id.clone()), sid);
        let priority = s.priority.unwrap_or(50.0);
        validate_priority(&format!("{}/{}", owner, s.id), priority)?;
        items.push(SystemPromptItem::Section(SectionSchema {
            id: sid,
            name: s.id.clone(),
            content: s.content.clone(),
            priority,
            depends_on: None,
            depends_on_absent: None,
            depends_on_configured: None,
            depends_on_unconfigured: None,
            is_template: false,
            template_tokens: None,
        }));
    }

    // First pass — pre-allocate every Collection's id (including tree-embedded
    // ones) so a Section's `depends_on: <collection>` can reference a collection
    // declared later.
    let mut collections: std::collections::HashMap<String, super::super::ids::CollectionId> =
        std::collections::HashMap::new();
    for entry in &raw_sp.items {
        match entry {
            YamlSystemPromptItem::Collection { name, .. } => {
                register_collection_id(
                    name,
                    lid,
                    &mut collection_names,
                    &mut collections,
                    maps,
                    collection_alloc,
                )?;
            }
            YamlSystemPromptItem::SectionTree { nodes } => {
                register_tree_collection_ids(
                    nodes,
                    lid,
                    &mut collection_names,
                    &mut collections,
                    maps,
                    collection_alloc,
                )?;
            }
            _ => {}
        }
    }

    // Second pass — build the schema items, resolving `depends_on` against the
    // pre-allocated collection-name map.
    for entry in &raw_sp.items {
        match entry {
            YamlSystemPromptItem::Section {
                id,
                pin: _,
                content,
                priority,
                depends_on,
                depends_on_absent,
                depends_on_configured,
                depends_on_unconfigured,
            } => {
                if !section_names.insert(id.clone()) {
                    return Err(ConstructionError::DuplicateSectionName(id.clone()));
                }
                let sid = section_alloc.next();
                maps.section_names.insert((lid, id.clone()), sid);
                let pri = priority.unwrap_or(50.0);
                validate_priority(&format!("{}/{}", owner, id), pri)?;
                let resolve_dep = |name: &Option<String>| -> Result<_, ConstructionError> {
                    match name {
                        Some(name) => Ok(Some(*collections.get(name).ok_or_else(|| {
                            ConstructionError::UnknownCollection(format!(
                                "{}: section '{}' depends_on unknown collection '{}'",
                                owner, id, name,
                            ))
                        })?)),
                        None => Ok(None),
                    }
                };
                let depends_on_cid = resolve_dep(depends_on)?;
                let depends_on_absent_cid = resolve_dep(depends_on_absent)?;
                let depends_on_configured_cid = resolve_dep(depends_on_configured)?;
                let depends_on_unconfigured_cid = resolve_dep(depends_on_unconfigured)?;
                items.push(SystemPromptItem::Section(SectionSchema {
                    id: sid,
                    name: id.clone(),
                    content: content.clone(),
                    priority: pri,
                    depends_on: depends_on_cid,
                    depends_on_absent: depends_on_absent_cid,
                    depends_on_configured: depends_on_configured_cid,
                    depends_on_unconfigured: depends_on_unconfigured_cid,
                    is_template: false,
                    template_tokens: None,
                }));
            }
            YamlSystemPromptItem::Template {
                id,
                dialect: dialect_name,
                depends_on,
            } => {
                let dlct = dialect.ok_or_else(|| ConstructionError::DialectRequired {
                    item: format!("{}/{}", owner, id),
                })?;
                let template = DialectTemplate::from_yaml_name(dialect_name).ok_or_else(|| {
                    ConstructionError::UnknownDialectTemplate {
                        item: format!("{}/{}", owner, id),
                        name: dialect_name.clone(),
                    }
                })?;
                let content = dlct.template(template);
                // Empty-string templates (e.g. NoThinkPrefix for a dialect that
                // doesn't suppress thinking) are dropped at build time.
                if content.is_empty() {
                    continue;
                }
                if !section_names.insert(id.clone()) {
                    return Err(ConstructionError::DuplicateSectionName(id.clone()));
                }
                let sid = section_alloc.next();
                maps.section_names.insert((lid, id.clone()), sid);
                let depends_on_cid = match depends_on {
                    Some(name) => Some(*collections.get(name).ok_or_else(|| {
                        ConstructionError::UnknownCollection(format!(
                            "{}: template '{}' depends_on unknown collection '{}'",
                            owner, id, name,
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
                    depends_on_configured: None,
                    depends_on_unconfigured: None,
                    is_template: true,
                    template_tokens: None,
                }));
            }
            YamlSystemPromptItem::Collection {
                name,
                pin: _,
                selection,
                score_threshold,
                policy: coll_policy_yaml,
                sections,
                member_glue,
                default: coll_default,
                budget_adaptive: coll_budget_adaptive,
            } => {
                let cid = *collections
                    .get(name)
                    .expect("first-pass pre-allocated every collection id");
                let label = format!("{}/{}", owner, name);
                let coll_selection = parse_collection_selection(&label, selection)?;
                if *score_threshold < 0.0 {
                    return Err(ConstructionError::NegativeScoreThreshold {
                        name: label.clone(),
                        value: *score_threshold,
                    });
                }

                let mut sec_schemas = Vec::with_capacity(sections.len());
                for s in sections {
                    if !section_names.insert(s.id.clone()) {
                        return Err(ConstructionError::DuplicateSectionName(s.id.clone()));
                    }
                    let sid = section_alloc.next();
                    maps.section_names.insert((lid, s.id.clone()), sid);
                    let pri = s.priority.unwrap_or(50.0);
                    validate_priority(&format!("{}/{}", owner, s.id), pri)?;
                    sec_schemas.push(SectionSchema {
                        id: sid,
                        name: s.id.clone(),
                        content: s.content.clone(),
                        priority: pri,
                        depends_on: None,
                        depends_on_absent: None,
                        depends_on_configured: None,
                        depends_on_unconfigured: None,
                        is_template: false,
                        template_tokens: None,
                    });
                }

                // A collection with no explicit `policy:` but a `top_k` selection
                // derives its belief budget from that rule (k → budget max,
                // score_threshold → min/evict floor). An explicit `policy:` wins.
                let coll_policy = match (coll_policy_yaml.as_ref(), &coll_selection) {
                    (None, SelectionRule::TopK { k }) => {
                        let mut config = base_policy.config;
                        config.min_score = *score_threshold;
                        config.evict_score = *score_threshold;
                        config.budget_min = 0;
                        config.budget_max = *k;
                        SelectionPolicy {
                            config,
                            tags: base_policy.tags.clone(),
                            layer_weights: base_policy.layer_weights.clone(),
                            calibration_layer_weights: base_policy
                                .calibration_layer_weights
                                .clone(),
                            scan: base_policy.scan,
                        }
                    }
                    _ => parse_policy(&label, coll_policy_yaml.as_ref(), base_policy)?,
                };
                items.push(SystemPromptItem::Collection(SectionCollection {
                    id: cid,
                    name: name.clone(),
                    sections: sec_schemas,
                    selection: coll_selection,
                    score_threshold: *score_threshold,
                    policy: coll_policy,
                    summary_section: None,
                    member_glue: member_glue.clone().unwrap_or_default(),
                    member_glue_tokens: None,
                    default: parse_default(coll_default.as_ref()),
                    budget_adaptive: parse_member_budget_adaptive(
                        &label,
                        coll_budget_adaptive.as_ref(),
                    )?,
                    mandatory: Vec::new(),
                }));
            }
            YamlSystemPromptItem::SectionTree { nodes } => {
                items.push(build_section_tree(
                    owner,
                    lid,
                    nodes,
                    dialect,
                    section_alloc,
                    maps,
                    &mut section_names,
                    &collections,
                )?);
            }
        }
    }

    let mut sp = SystemPromptSchema {
        items,
        pins: PinSet::default(),
    };
    sp.pins = PinSet::resolve(&pinned_names(&raw_sp.items), raw_sp.pin_tokens, &sp)
        .map_err(ConstructionError::InvalidPins)?;
    Ok(sp)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a [`CompressionPrompt`] from a YAML framing/instruction pair. Allocates
/// a `SectionId` for its system-prompt framing — that section is never emitted in
/// a normal projection (it is reached only by id during compression), so it is
/// deliberately NOT registered in `NameMaps` or `system_prompt.items`. Validates
/// both prompts are non-empty.
pub(super) fn build_compression_prompt(
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
            depends_on_configured: None,
            depends_on_unconfigured: None,
            is_template: false,
            template_tokens: None,
        },
        user_prompt: yc.user_prompt.clone(),
    })
}

/// Build one tree-level [`TurnSummary`] (scope derivation + answer half + decode
/// cap). Validates `max_tokens >= 1`; the assistant half allocates its framing
/// section. The user half carries no prompt — it is derived, never decoded.
pub(super) fn build_turn_summary(
    owner_label: &str,
    yt: &YamlTurnSummary,
    a_name: String,
    section_alloc: &mut SectionIdAlloc,
) -> Result<TurnSummary, ConstructionError> {
    if yt.max_tokens == 0 {
        return Err(ConstructionError::InvalidSummary {
            owner: owner_label.to_string(),
        });
    }
    let scope = match yt.scope.as_str() {
        "" | "union" => Scope::Union,
        "line_spans" => Scope::LineSpans,
        other => {
            return Err(ConstructionError::InvalidSummaryScope {
                owner: owner_label.to_string(),
                scope: other.to_string(),
            })
        }
    };
    let content = match yt.content.as_str() {
        "" | "decode" => Content::Decode,
        "structural" => Content::Structural,
        other => {
            return Err(ConstructionError::InvalidSummaryContent {
                owner: owner_label.to_string(),
                content: other.to_string(),
            })
        }
    };
    // A structural level derives both halves from its children's skeletons, so a
    // `scope:` here would be silently ignored. Reject it rather than let the
    // config claim a derivation that never runs.
    if content == Content::Structural && !yt.scope.is_empty() {
        return Err(ConstructionError::ScopeOnStructuralSummary {
            owner: owner_label.to_string(),
        });
    }
    Ok(TurnSummary {
        scope,
        assistant: build_compression_prompt(owner_label, &yt.assistant, a_name, section_alloc)?,
        max_tokens: yt.max_tokens,
        content,
    })
}

/// Build a layer's [`LayerSummary`] — the mandatory `turns` level plus the
/// optional `summaries` level.
pub(super) fn build_layer_summary(
    owner_label: &str,
    yl: &YamlLayerSummary,
    section_alloc: &mut SectionIdAlloc,
) -> Result<LayerSummary, ConstructionError> {
    let turns = build_turn_summary(
        owner_label,
        &yl.turns,
        "__summary_turns_assistant__".to_string(),
        section_alloc,
    )?;
    let summaries = match &yl.summaries {
        Some(ys) => Some(build_turn_summary(
            &format!("{owner_label} (summaries)"),
            ys,
            "__summary_sums_assistant__".to_string(),
            section_alloc,
        )?),
        None => None,
    };
    Ok(LayerSummary { turns, summaries })
}
