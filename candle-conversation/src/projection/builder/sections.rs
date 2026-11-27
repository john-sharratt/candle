//! Runtime mutators that add sections and collections to a schema before its first
//! projection.
//!
//! Dynamic catalogues such as tool definitions are installed this way instead of
//! being written into YAML. Every mutator must run before the first projection,
//! because the prefilled prompt is keyed by section id. Section names must be
//! unique across the layer, and a collection's selection rule decides which of its
//! members survive a projection. Lookups reach into section trees as well, so a
//! catalogue installed into a tree-embedded collection is found the same way as a
//! top-level one.

use super::*;

impl Builder {
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
                depends_on_configured: None,
                depends_on_unconfigured: None,
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
                    calibration_layer_weights: Vec::new(),
                    scan: ScanPolicy::default(),
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
                summary_section: None,
                member_glue: String::new(),
                member_glue_tokens: None,
                default: None,
                budget_adaptive: None,
                mandatory: Vec::new(),
                range: None,
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
                    depends_on_configured: None,
                    depends_on_unconfigured: None,
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
                    depends_on_configured: None,
                    depends_on_unconfigured: None,
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

    /// Append a section with a caller-chosen id to a top-level collection.
    ///
    /// Unlike [`Self::add_section_to_collection`] this does not draw from the
    /// builder's own id counter. Sections owned by one conversation are merged
    /// into a clone of its builder this way, with ids from the substrate's
    /// owned partition; because later builder-allocated ids start above the
    /// highest id present, nothing further is added to such a clone. Errors for
    /// a section-tree collection, whose members seal once per branch and need
    /// builder-allocated ids.
    pub fn add_section_to_collection_with_id(
        &mut self,
        collection: CollectionId,
        id: SectionId,
        name: impl Into<String>,
        content: impl Into<String>,
        priority: f32,
    ) -> Result<(), ConstructionError> {
        let name: String = name.into();
        if priority <= 0.0 {
            return Err(ConstructionError::InvalidPriority {
                name: name.clone(),
                value: priority,
            });
        }
        self.assert_section_name_free(&name)?;
        let items = &mut self.schema.system_prompt.items;
        let Some(CollLoc::TopLevel(ii)) = locate_collection(items, |c| c.id == collection) else {
            return Err(ConstructionError::UnknownCollection(format!(
                "CollectionId({collection:?}) is not a top-level collection"
            )));
        };
        let SystemPromptItem::Collection(coll) = &mut items[ii] else {
            unreachable!("located a top-level collection")
        };
        coll.sections.push(SectionSchema {
            id,
            name: name.clone(),
            content: content.into(),
            priority,
            depends_on: None,
            depends_on_absent: None,
            depends_on_configured: None,
            depends_on_unconfigured: None,
            is_template: false,
            template_tokens: None,
        });
        self.name_maps
            .section_names
            .insert((LayerId::system_prompt(), name), id);
        Ok(())
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

    /// Mark `section` a mandatory member of the top-level `collection`: it is
    /// emitted on every belief-driven projection, outside the policy's member
    /// budget (see [`SectionCollection::mandatory`]). Errors when the collection
    /// is unknown, when it is embedded in a section tree (those select by
    /// provenance top-k, which has no budget to sit outside), or when it does
    /// not hold `section`.
    pub fn set_collection_member_mandatory(
        &mut self,
        collection: CollectionId,
        section: SectionId,
    ) -> Result<(), ConstructionError> {
        let items = &mut self.schema.system_prompt.items;
        match locate_collection(items, |c| c.id == collection) {
            Some(CollLoc::TopLevel(ii)) => {
                let SystemPromptItem::Collection(coll) = &mut items[ii] else {
                    unreachable!("located a top-level collection")
                };
                if !coll.sections.iter().any(|s| s.id == section) {
                    return Err(ConstructionError::MandatoryMember(format!(
                        "{section:?} is not a member of collection {:?}",
                        coll.name
                    )));
                }
                if !coll.mandatory.contains(&section) {
                    coll.mandatory.push(section);
                }
                Ok(())
            }
            Some(CollLoc::Tree { .. }) => Err(ConstructionError::MandatoryMember(format!(
                "CollectionId({collection:?}) is embedded in a section tree, which selects \
                 by provenance top-k and has no member budget to sit outside"
            ))),
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

    /// Replace the content of option `option` of the tree node `name` BEFORE
    /// prefill, as [`Self::set_tree_section_content`] does for a node of one
    /// option: a node of several — an `optional`'s present side, a selector's
    /// option — whose text is generated at runtime, such as the seed the
    /// knowledge base opens on. The option must already exist with non-empty
    /// authored content, so its per-branch variants were allocated. Errors when
    /// no such node and option is found.
    pub fn set_tree_option_content(
        &mut self,
        name: &str,
        option: &str,
        content: impl Into<String>,
    ) -> Result<(), ConstructionError> {
        let content = content.into();
        for item in self.schema.system_prompt.items.iter_mut() {
            if let SystemPromptItem::SectionTree(t) = item {
                for node in t.nodes.iter_mut() {
                    if node.name != name || node.collection.is_some() || node.glue.is_some() {
                        continue;
                    }
                    if let Some(opt) = node.options.iter_mut().find(|o| o.id == option) {
                        opt.content = content;
                        return Ok(());
                    }
                }
            }
        }
        Err(ConstructionError::UnknownSection(format!(
            "{name}/{option}"
        )))
    }

    /// Reject a duplicate section name in the shared system prompt (top-level OR
    /// nested in a collection or section-tree).
    pub(super) fn assert_section_name_free(&self, name: &str) -> Result<(), ConstructionError> {
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
    pub(super) fn next_section_id_raw(&self) -> u32 {
        // Must cover EVERY allocated section id, including the compression-prompt
        // sections that live in `layer.summary` / collection summaries rather
        // than `system_prompt.items` — otherwise a runtime-added section aliases
        // a compression-prompt id and corrupts the compression frame.
        let max_id = self.schema.all_section_ids().into_iter().max().unwrap_or(0);
        max_id + 1
    }

    /// Highest currently-allocated collection-id across the whole
    /// schema, + 1.
    pub(super) fn next_collection_id_raw(&self) -> u32 {
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
}

/// Where a collection lives in a layer's system prompt — a top-level item or a
/// node inside a section tree.  Returned by [`locate_collection`] so the runtime
/// mutators re-borrow the located item mutably by index without fighting the
/// borrow checker.
pub(super) enum CollLoc {
    TopLevel(usize),
    Tree { item: usize, node: usize },
}

/// Find the first collection in `items` matching `matches`, descending into
/// section-tree collection nodes.  Pure (immutable) index lookup — the single
/// place the runtime mutators resolve a collection, so a tool catalog installed
/// into a tree-embedded `tools` collection is found exactly like a top-level one.
pub(super) fn locate_collection(
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
pub(super) fn retain_tree_members(
    tc: &mut TreeCollection,
    keep: &std::collections::HashSet<String>,
) {
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
