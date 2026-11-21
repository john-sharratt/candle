//! Adjusts a built schema's layers, groups and collections by name before a
//! conversation uses it.
//!
//! A layer taken out of the provenance gather, or given a different window, must
//! be set before the first projection, because the belief scan and normalisation
//! read those settings. Rule and tag overrides let one shared schema serve
//! conversations that want the same corpus at different depths. An empty tag list
//! means no scope at all, which is deliberately not the collection rule's meaning
//! of an empty filter, and a non-empty list makes a group private to its owners.

use super::*;

impl Builder {
    // ── Name-based lookups (YAML string → id) ─────────────────────────────────

    /// Take a layer OUT of the provenance gather, by schema name. Returns
    /// whether the layer was found.
    ///
    /// The runtime half of [`LayerSchema::gathered`] — the only writer. An
    /// operator's `--disable-layer` decision belongs to one boot, so it is
    /// applied to the built schema rather than declared in YAML, and it must be
    /// applied BEFORE the first projection: the flag is read on every belief
    /// scan and by both normalization warm-ups, so a layer cleared afterwards
    /// would already have taught backgrounds and returned candidates.
    ///
    /// Distinct from suppressing a layer's ingest. A layer can be populated and
    /// not gathered (pointless), or gathered and not populated (`--skip-layer`,
    /// the useful case: the substrate already holds its turns and they should
    /// keep answering queries while nothing new is read from disk).
    pub fn set_layer_gathered(&mut self, name: &str, gathered: bool) -> bool {
        match self.schema.layers.iter_mut().find(|l| l.name == name) {
            Some(l) => {
                l.gathered = gathered;
                true
            }
            None => false,
        }
    }

    /// Set a layer's token window, by schema name: the most of its turns one
    /// projection carries. Returns whether the layer was found.
    ///
    /// A YAML schema states its windows; a programmatic one
    /// ([`Self::for_plain_prompt`]) takes a fixed one, and this is how a caller
    /// asks for another — a conversation held whole to a depth the fixed window
    /// would project, for one. Like [`Self::set_layer_gathered`], it is applied
    /// before the first projection.
    pub fn set_layer_window(&mut self, name: &str, window: usize) -> bool {
        match self.schema.layers.iter_mut().find(|l| l.name == name) {
            Some(l) => {
                l.window = window;
                true
            }
            None => false,
        }
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

    /// Override a turn group's selection rule after construction.
    ///
    /// The group-level counterpart of [`Self::set_collection_selection`], and
    /// for the same kind of caller: one schema serving conversations that want
    /// the same corpus at different depths. A conversation that reflects reaches
    /// further into it than one that acts, and the difference is `k`, not a
    /// second schema. A belief-driven group's budget follows its rule (see
    /// [`GroupSchema::belief_config`]), so this moves both.
    pub fn set_group_selection(
        &mut self,
        name: &str,
        selection: SelectionRule,
    ) -> Result<(), ConstructionError> {
        let group = self.group_named(name)?;
        group.selection = selection;
        Ok(())
    }

    /// Scope a turn group to turns carrying one of `tags`, after construction.
    ///
    /// **An empty list is no scope at all** — every turn in the group is a
    /// candidate, which is what a group has always meant. That is deliberately
    /// not the collection rule, where an empty filter admits only untagged
    /// turns: a turn group's candidates are the conversations written to it,
    /// and untagged is how nearly all of them are written.
    ///
    /// A non-empty list is what makes one group private to one owner: the
    /// group holds every owner's conversations, and each conversation reading
    /// it names the tag that is its own.
    pub fn set_group_tags(
        &mut self,
        name: &str,
        tags: Vec<String>,
    ) -> Result<(), ConstructionError> {
        let group = self.group_named(name)?;
        group.policy.tags = tags;
        Ok(())
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
                    // A member dropped here is dropped even if mandatory: the
                    // caller's filter (a tools mode) decides what may project.
                    let kept: Vec<SectionId> = coll.sections.iter().map(|s| s.id).collect();
                    coll.mandatory.retain(|id| kept.contains(id));
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
}
