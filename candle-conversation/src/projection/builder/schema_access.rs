//! Read access to a builder's schema, and the name-to-id lookups it offers.
//!
//! A builder keeps the YAML it was parsed from, so the substrate can persist it as
//! the template; programmatically built schemas have none. The lookups turn YAML
//! layer, group and section names into the opaque ids the rest of the pipeline
//! uses, and return `None` for an unknown name or for a builder made without name
//! maps. `focused` gives the view that a question about the conversation itself is
//! projected under.

use super::*;

impl Builder {
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

    /// Lift every repetition penalty while a tool call is emitted — see
    /// [`Schema::free_tool_calls_from_penalties`]. For a conversation whose
    /// tool arguments are quotations: a path written twice in one call is only
    /// correct if it repeats.
    #[must_use]
    pub fn free_tool_calls_from_penalties(mut self) -> Self {
        self.schema.free_tool_calls_from_penalties = true;
        self
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

    /// This builder, held to the target's own conversation ([`Schema::focus`]):
    /// what a question about the conversation itself is projected under.
    pub fn focused(&self) -> Builder {
        let mut focused = self.clone();
        focused.schema.focus = true;
        focused
    }

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

    pub(super) fn group_named(
        &mut self,
        name: &str,
    ) -> Result<&mut GroupSchema, ConstructionError> {
        self.schema
            .layers
            .iter_mut()
            .flat_map(|l| l.groups.iter_mut())
            .find(|g| g.name == name)
            .ok_or_else(|| ConstructionError::UnknownGroup(name.to_string()))
    }
}
