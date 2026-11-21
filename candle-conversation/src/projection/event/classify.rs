//! Turns a real projection into the projection event the record stores, by
//! classifying each selected segment against the schema.
//!
//! Each selected segment is placed as a system section, a named collection or a
//! turn group, and its token count is read from the resolver, so the event shows
//! the composition the model actually attended to. Structural framing is not
//! content and is skipped. The origin-aware form also records why each selected
//! turn was chosen, so the panel can show provenance against recency and coverage.
//! Sections, collections and the glue between their members appear in the order
//! the prompt materialized them, and glue is shown only where its tokens were
//! emitted.

use super::*;

/// Build a [`ProjectionEvent`] from a real projection: classify each selected
/// segment against the schema (system vs named collection vs turn group), read
/// its token count from the resolver, then [`aggregate`].
///
/// Structural [`ProjectionSegment::Generated`] runs (role markers, block
/// envelopes) are framing, not content, and are skipped.
pub fn from_projection(
    segments: &[ProjectionSegment],
    schema: &Schema,
    resolver: &dyn ContentResolver,
    scores: &SelectionScores,
    substrate_tokens: u32,
    start_token: u32,
    seconds: f64,
) -> ProjectionEvent {
    from_projection_with_origins(
        segments,
        &HashMap::new(),
        schema,
        resolver,
        scores,
        substrate_tokens,
        start_token,
        seconds,
    )
}

/// As [`from_projection`], but also stamps each selected turn with *why* it was
/// chosen — the `(group, turn) → SelectionOrigin` map carried on
/// [`super::Projection::selection_origins`]. Production callers pass it so the
/// persisted record / GUI can show provenance vs recency vs coverage; tests and
/// the no-origin wrapper above pass an empty map (every `reason` is then `None`).
pub fn from_projection_with_origins(
    segments: &[ProjectionSegment],
    origins: &HashMap<TurnKey, SelectionOrigin>,
    schema: &Schema,
    resolver: &dyn ContentResolver,
    scores: &SelectionScores,
    substrate_tokens: u32,
    start_token: u32,
    seconds: f64,
) -> ProjectionEvent {
    let classified: Vec<SegmentTokens> = segments
        .iter()
        .filter_map(|seg| match seg {
            ProjectionSegment::Sealed(SealedKind::Section(s)) => {
                let tokens = resolver.section_token_count(s.id) as u32;
                let (label, kind) = match collection_name_of(schema, s.id)
                    .or_else(|| collection_of_summary(schema, s.id))
                {
                    Some(name) => (name.to_string(), BucketKind::Section),
                    None => ("system".to_string(), BucketKind::System),
                };
                Some(SegmentTokens {
                    kind,
                    label,
                    tokens,
                })
            }
            ProjectionSegment::Sealed(SealedKind::Turn(t, _role)) => {
                let tokens = t.key().map_or(0, |k| resolver.turn_token_count(k)) as u32;
                let label = group_name_of(schema, t.group())
                    .unwrap_or("conversation")
                    .to_string();
                Some(SegmentTokens {
                    kind: BucketKind::Turns,
                    label,
                    tokens,
                })
            }
            ProjectionSegment::NewUserMessage { tokens } => Some(SegmentTokens {
                kind: BucketKind::Turns,
                label: "current message".to_string(),
                tokens: tokens.len() as u32,
            }),
            // Structural framing — not materialized content.
            ProjectionSegment::Generated { .. } => None,
        })
        .collect();

    let mut event = aggregate(&classified, substrate_tokens, start_token, seconds);
    event.selection = build_selection(segments, origins, schema, resolver, scores);
    event.groups = schema
        .layers
        .iter()
        .flat_map(|l| l.groups.iter())
        .filter_map(|g| {
            resolver.group_scan(g.id).map(|scan| GroupScanRecord {
                group: g.name.clone(),
                scan,
            })
        })
        .collect();
    event
}

/// Build the structured [`ProjectionSelection`] from the selected segments and
/// the schema. A bare system section is listed when selected; a collection is
/// listed (with all its members flagged) when ≥1 member was selected — which
/// scopes the result to the contributing layer. Turns are the selected
/// conversation segments plus the current message.
pub(super) fn build_selection(
    segments: &[ProjectionSegment],
    origins: &HashMap<TurnKey, SelectionOrigin>,
    schema: &Schema,
    resolver: &dyn ContentResolver,
    scores: &SelectionScores,
) -> ProjectionSelection {
    let mut selected: std::collections::HashSet<SectionId> = std::collections::HashSet::new();
    let mut turns: Vec<SelectedTurn> = Vec::new();
    // Structural template glue (`Generated`) actually emitted this projection,
    // keyed by name → its token count. `depends_on`-gated templates (e.g.
    // `tools_close`) only appear here when they really fired, so the view shows
    // exactly the framing the decode prefilled.
    let mut emitted_glue: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    for seg in segments {
        match seg {
            ProjectionSegment::Sealed(SealedKind::Section(s)) => {
                selected.insert(s.id);
            }
            ProjectionSegment::Sealed(SealedKind::Turn(t, role)) => {
                // A face placed for a unit's opening records the opening: that
                // is what was selected, scored and given a reason, and what the
                // next reprojection's belief is seeded from. Its tokens are the
                // face's, which is what the slot holds.
                let placed = t.key();
                let selected = placed.map(|k| resolver.unit_of(k).unwrap_or(k));
                turns.push(SelectedTurn {
                    layer: layer_name_of_group(schema, t.group())
                        .unwrap_or("")
                        .to_string(),
                    group: group_name_of(schema, t.group())
                        .unwrap_or("conversation")
                        .to_string(),
                    index: selected.map_or(t.index().0, |k| k.index.0),
                    role: role_str(*role).to_string(),
                    tokens: placed.map_or(0, |k| resolver.turn_token_count(k)) as u32,
                    kind: selected.map_or(TurnKind::Normal, |k| resolver.turn_kind(k)),
                    reason: selected.and_then(|k| origins.get(&k)).copied(),
                    timeline: selected.map(|k| k.timeline.raw()),
                    // A turn in the segments was selected; stamp its belief score
                    // so the next reprojection can seed its turn-group carry.
                    selected: true,
                    score: selected.map_or(0.0, |k| scores.turn(k)),
                    qualified: selected.is_some_and(|k| scores.turn_qualified(k)),
                });
            }
            ProjectionSegment::NewUserMessage { tokens } => {
                turns.push(SelectedTurn {
                    layer: "dialogue".to_string(),
                    group: "current message".to_string(),
                    index: u32::MAX,
                    role: "user".to_string(),
                    tokens: tokens.len() as u32,
                    // The live message is the probe, not a selected memory — it
                    // is always a raw turn and has no selection origin.
                    kind: TurnKind::Normal,
                    reason: None,
                    timeline: None,
                    selected: false,
                    score: 0.0,
                    qualified: false,
                });
            }
            ProjectionSegment::Generated { tokens, identity } => {
                emitted_glue.insert(identity.name.as_str(), tokens.len() as u32);
            }
        }
    }

    // Walk the schema in declaration order — which IS the system prompt's
    // materialized order — emitting one [`SystemItem`] per item that fired:
    // template glue (when it emitted), a selected bare section, or a collection
    // with a selected member. Template glue interleaves with content exactly as
    // the prompt is assembled, so the closing markers appear in place.
    let mut system: Vec<SystemItem> = Vec::new();
    // A collection's catalog summary shows just before its opening structural
    // marker (OUTSIDE it), matching the projected order in
    // `emit_system_prompt_items`. Captured up front, drained as it emits.
    //
    // The `selected.contains(&sum)` gate is *inherited* from the projection: a
    // summary section only lands in `selected` if `project.rs` already emitted it
    // (its own `partial && token_count > 0` gate), so the panel can't show a
    // summary the prompt didn't — no need to re-derive that gate here.
    let mut pending_summaries: std::collections::HashMap<CollectionId, (String, u32)> =
        std::collections::HashMap::new();
    for item in &schema.system_prompt.items {
        if let SystemPromptItem::Collection(c) = item {
            if let Some(sum) = c.summary_section {
                if selected.contains(&sum) {
                    pending_summaries.insert(
                        c.id,
                        (
                            format!("{} summary", c.name),
                            resolver.section_token_count(sum) as u32,
                        ),
                    );
                }
            }
        }
    }
    for item in &schema.system_prompt.items {
        match item {
            SystemPromptItem::Section(s) if s.is_template => {
                if let Some(&tokens) = emitted_glue.get(s.name.as_str()) {
                    // Emit the collection's summary just before its opening
                    // marker (e.g. `<tools>`), so it sits OUTSIDE the block —
                    // mirroring the projection.
                    if let Some(cid) = s.depends_on {
                        if let Some((name, sum_tokens)) = pending_summaries.remove(&cid) {
                            system.push(SystemItem::Section {
                                name,
                                tokens: sum_tokens,
                            });
                        }
                    }
                    system.push(SystemItem::Glue {
                        name: s.name.clone(),
                        content: s.content.clone(),
                        tokens,
                    });
                }
            }
            SystemPromptItem::Section(s) if selected.contains(&s.id) => {
                system.push(SystemItem::Section {
                    name: s.name.clone(),
                    tokens: resolver.section_token_count(s.id) as u32,
                });
            }
            SystemPromptItem::Section(_) => {}
            SystemPromptItem::Collection(c) => {
                // The runtime summary section (a reserved section, not a schema
                // item) shows just before the collection's opening marker, so
                // it was already drained above. This is the fallback for a
                // collection with no opening marker — show it before members.
                // Named `"<collection> summary"`; the daemon serves its text
                // under the same key.
                if let Some((name, tokens)) = pending_summaries.remove(&c.id) {
                    system.push(SystemItem::Section { name, tokens });
                }
                if c.sections.iter().any(|s| selected.contains(&s.id)) {
                    // Mirror the projection exactly: it interleaves member glue
                    // only when the tokens exist, so report it on the same
                    // condition rather than on the string being non-empty.
                    let (member_glue, member_glue_tokens) = match &c.member_glue_tokens {
                        Some(t) => (c.member_glue.clone(), t.len() as u32),
                        None => (String::new(), 0),
                    };
                    system.push(SystemItem::Collection {
                        name: c.name.clone(),
                        member_glue,
                        member_glue_tokens,
                        sections: c
                            .sections
                            .iter()
                            .map(|s| SelectedSection {
                                name: s.name.clone(),
                                tokens: resolver.section_token_count(s.id) as u32,
                                selected: selected.contains(&s.id),
                                score: scores.section(s.id),
                                qualified: scores.section_qualified(s.id),
                            })
                            .collect(),
                    });
                }
            }
            SystemPromptItem::SectionTree(t) => {
                // A node emitted exactly one option's branch variant; show
                // the node (and chosen option, for selectors) with its tokens.
                for n in &t.nodes {
                    // A live-prefilled structural marker (`<tools>` etc.): if it
                    // fired this projection (its Generated run is in
                    // `emitted_glue`), surface it as a real glue row.
                    if n.glue.is_some() {
                        if let Some(&tokens) = emitted_glue.get(n.name.as_str()) {
                            system.push(SystemItem::Glue {
                                name: n.name.clone(),
                                content: n
                                    .options
                                    .first()
                                    .map_or(String::new(), |o| o.content.clone()),
                                tokens,
                            });
                        }
                        continue;
                    }
                    for o in &n.options {
                        if let Some(v) = o.variants.iter().find(|v| selected.contains(&v.id)) {
                            let name = if n.options.len() > 1 {
                                format!("{}:{}", n.name, o.id)
                            } else {
                                n.name.clone()
                            };
                            system.push(SystemItem::Section {
                                name,
                                tokens: resolver.section_token_count(v.id) as u32,
                            });
                        }
                    }
                    // An embedded collection node: show each selected member
                    // (its active-branch variant landed in `selected`),
                    // interleaving the collection's `member_glue` as a real
                    // structural glue row BETWEEN consecutive members —
                    // mirroring the `Generated` glue token the projection emits
                    // (and NOT baked into any member's seal), so the panel and
                    // its copy-all reproduce the exact materialized bytes.
                    if let Some(tc) = &n.collection {
                        let glue = &tc.collection.member_glue;
                        // Mirror the projection EXACTLY: it interleaves member
                        // glue only when the tokens exist (`member_glue_tokens`),
                        // so gate the panel row on the same condition (not just
                        // the non-empty string) or the panel would show a 0-token
                        // glue row the materialized prompt never contained.
                        let glue_tokens = tc
                            .collection
                            .member_glue_tokens
                            .as_ref()
                            .map(|t| t.len() as u32);
                        let mut emitted_member = false;
                        for (s, member) in tc.collection.sections.iter().zip(tc.variants.iter()) {
                            if let Some(v) = member.iter().find(|v| selected.contains(&v.id)) {
                                if let (true, Some(toks)) = (emitted_member, glue_tokens) {
                                    system.push(SystemItem::Glue {
                                        name: format!("{}__member_glue", tc.collection.name),
                                        content: glue.clone(),
                                        tokens: toks,
                                    });
                                }
                                emitted_member = true;
                                system.push(SystemItem::Section {
                                    name: s.name.clone(),
                                    tokens: resolver.section_token_count(v.id) as u32,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    ProjectionSelection { system, turns }
}

/// Role → wire string for [`SelectedTurn`].
pub(crate) fn role_str(role: crate::Role) -> &'static str {
    match role {
        crate::Role::User => "user",
        crate::Role::Assistant => "assistant",
        crate::Role::System => "system",
    }
}

/// The name of the system-prompt collection a section belongs to, or `None`
/// when the section is a bare top-level item (i.e. part of the system prompt
/// proper, not a named group).
pub(super) fn collection_name_of(schema: &Schema, id: SectionId) -> Option<&str> {
    schema
        .system_prompt
        .items
        .iter()
        .find_map(|item| match item {
            SystemPromptItem::Collection(c) if c.sections.iter().any(|s| s.id == id) => {
                Some(c.name.as_str())
            }
            _ => None,
        })
}

/// The name of the collection whose *runtime summary section* is `id`. Unlike
/// [`collection_name_of`] (which matches declared members), this matches the
/// reserved summary section a collection injects before its members on partial
/// selection — that section is not a schema item, so it is classified here.
pub(super) fn collection_of_summary(schema: &Schema, id: SectionId) -> Option<&str> {
    schema
        .system_prompt
        .items
        .iter()
        .find_map(|item| match item {
            SystemPromptItem::Collection(c) if c.summary_section == Some(id) => {
                Some(c.name.as_str())
            }
            _ => None,
        })
}

/// The YAML name of a group by its globally-unique id.
pub(crate) fn group_name_of(schema: &Schema, id: GroupId) -> Option<&str> {
    schema
        .layers
        .iter()
        .flat_map(|l| l.groups.iter())
        .find(|g| g.id == id)
        .map(|g| g.name.as_str())
}

/// The YAML name of the layer that owns the group with this id.
pub(crate) fn layer_name_of_group(schema: &Schema, id: GroupId) -> Option<&str> {
    schema
        .layers
        .iter()
        .find(|l| l.groups.iter().any(|g| g.id == id))
        .map(|l| l.name.as_str())
}
