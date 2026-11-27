//! Builds the projection each chat turn runs under, for its tools mode and
//! its identity.
//!
//! A turn projects through a builder assembled once at startup and handed out
//! as a cheap `Arc` clone, so no turn re-clones the tool schema. Each tools
//! mode keeps only the tools it offers and points the tools collection at its
//! sealed catalog summary, so the full listing appears just before the
//! selected subset. Identity scoping narrows the builder to one persona's
//! anchor and facets. An unresolved identity scopes to nothing rather than to
//! the unscoped builder, which would emit every persona's anchor at once.

use super::*;

use crate::model::MODEL;

/// The projection a chat turn runs under: the tools mode's prebuilt view — a
/// cheap `Arc` clone, no per-turn schema clone — scoped to the conversation's
/// identity's anchor and facets when it names one (`IdentityBuilders`).
pub(super) fn turn_projection(
    state: &InferenceState,
    conv_id: &str,
    identity: Option<&str>,
    tools_mode: ToolMode,
) -> Arc<Builder> {
    let scope = state.identity_builders.resolve(identity);
    let (proj, applied) = match &scope {
        IdentityScope::Named(name) => (
            state
                .identity_builders
                .get(&state.mode_builders, name, tools_mode),
            Some(name.as_str()),
        ),
        IdentityScope::Empty => {
            tracing::warn!(conv_id = %conv_id,
                "conversation names no identity and mind.yaml sets no default — \
                 scoping identity collections to empty (no anchor emitted)");
            (
                state
                    .identity_builders
                    .get_empty(&state.mode_builders, tools_mode),
                None,
            )
        }
        IdentityScope::Unscoped => (state.mode_builders.get(tools_mode), None),
    };
    tracing::info!(conv_id = %conv_id, ?tools_mode, identity = ?applied, "projection applied");
    proj
}

/// What the `tools` collection is sealed behind: the prompt the creation walk
/// lays down before it (`prefix_before_collection`) — bare sections, the
/// default branch of each section tree, and each earlier reserved range as a
/// hole of its size on an engine trained to `native` positions. Sealing the
/// tool-summary section behind the *same* prefix puts its KV exactly where
/// "just before the tools" is. Empty when the schema declares no `tools`
/// collection.
pub(super) fn pre_tools_prefix(builder: &Builder, native: Option<usize>) -> Vec<PrefixItem> {
    builder
        .id_for_system_collection("tools")
        .and_then(|tools| {
            prefix_before_collection(&builder.schema().system_prompt.items, tools, native)
        })
        .unwrap_or_default()
}

/// The reserved section holding `mode`'s tool-catalog summary; `None` projects
/// no tools and has none.
pub(super) fn tool_summary_section(mode: ToolMode) -> Option<Reserved> {
    match mode {
        ToolMode::None => None,
        ToolMode::Restricted => Some(Reserved::ToolSummaryRestricted),
        ToolMode::Comprehensive => Some(Reserved::ToolSummary),
    }
}

/// Build the projection for one tools mode by cloning `base` and filtering the
/// `tools` collection MEMBERS to the tools the mode offers
/// ([`crate::tools::offered_tool_names`]). These builders control only WHICH
/// members project — the WHOLE tool block (markers, catalog, summary) is gated
/// separately by the `tools_enabled` optional_group, which chat.rs sets
/// `absent` for `None`. Called once per mode at startup by
/// [`ModeBuilders::build`]; the results are cached and handed out as cheap
/// `Arc` clones each turn, since both projection and reprojection just read the
/// swapped builder.
pub(super) fn build_mode_builder(base: &Builder, mode: ToolMode) -> anyhow::Result<Arc<Builder>> {
    let mut b = base.clone();
    // Keep exactly the tools this mode offers — none in None, everything in
    // Comprehensive; the summary association below points the mode at its listing.
    b.retain_collection_sections("tools", &crate::tools::offered_tool_names(mode))
        .map_err(|e| anyhow::anyhow!("{} tools projection: {e}", mode.id()))?;
    // Associate the sealed tool-catalog summary (built by `build_tool_summary`,
    // prefilled into its reserved section at startup) with the `tools` collection,
    // so projection emits the FULL tool-name listing just before the selected
    // subset. Without this the catalog K/V is sealed but orphaned — the model only
    // ever sees the 1-3 provenance-selected tools and concludes a tool it can't see
    // (e.g. `datetime` on a "what is the time?" turn) doesn't exist. The projection
    // emits it whenever the selection is a proper subset (§ `record` in
    // `emit_system_prompt_items`), which for a 93-tool catalog is every turn.
    // `None` mode has no tools block, so no summary.
    if let Some(reserved) = tool_summary_section(mode) {
        let tools = b
            .id_for_system_collection("tools")
            .ok_or_else(|| anyhow::anyhow!("projection schema missing 'tools' collection"))?;
        b.set_collection_summary_section(tools, SectionId::reserved(reserved))
            .map_err(|e| anyhow::anyhow!("tool summary association ({mode:?}): {e}"))?;
    }
    Ok(Arc::new(b))
}

/// The per-tools-mode projection builders, built once at startup so each turn
/// hands out a cheap `Arc` clone instead of re-cloning the ~93-section schema.
pub(super) struct ModeBuilders {
    /// Indexed by [`ToolMode::level`].
    pub(super) builders: [Arc<Builder>; ToolMode::ALL.len()],
}

impl ModeBuilders {
    /// Every mode's builder, or the first failure. A failure is only reachable
    /// through a missing dialogue layer or `tools` collection — a broken schema
    /// the daemon cannot serve — and falling back to another mode's builder
    /// would offer a mode tools it does not grant.
    pub(super) fn build(base: &Builder) -> anyhow::Result<Self> {
        let [none, restricted, comprehensive] =
            ToolMode::ALL.map(|mode| build_mode_builder(base, mode));
        Ok(Self {
            builders: [none?, restricted?, comprehensive?],
        })
    }

    /// The prebuilt projection for `mode` (a cheap `Arc` clone).
    pub(super) fn get(&self, mode: ToolMode) -> Arc<Builder> {
        Arc::clone(&self.builders[mode.level() as usize])
    }
}

/// The `identity_anchor` + `identity` members that belong to one identity
/// (namespaced `<name>::…`). The keep-sets a `retain_collection_sections` scopes
/// a conversation down to.
#[derive(Default)]
pub(super) struct IdentityKeep {
    pub(super) anchor: HashSet<String>,
    pub(super) facets: HashSet<String>,
}

/// Which identity scope a turn's projection uses — the three genuine states the
/// resolver distinguishes so an unresolved identity never falls through to the
/// unscoped builder (which would emit every persona's always-visible anchor).
pub(super) enum IdentityScope {
    /// No identities are installed at all (e.g. the coding-assistant schema whose
    /// projection has no `identity`/`identity_anchor` collections). Use the plain
    /// tools-mode builder; nothing identity-shaped exists to emit.
    Unscoped,
    /// Scope both identity collections to this identity's members.
    Named(String),
    /// Identities ARE installed but none resolved (the conversation set none and
    /// `mind.yaml` declares no default). Scope both collections to empty so no
    /// anchor emits — a mind schema with identities should set a default.
    Empty,
}

/// Lazy per-`(identity, ToolMode)` projection builders. Each is a [`ModeBuilders`]
/// view further scoped — via `retain_collection_sections`, which drops members
/// from projection but never re-seals KV — to a single identity's anchor + facets,
/// so a conversation projects only its own identity. Built on first use and
/// cached; bounded by identities-in-use × tool modes. KV is shared across all of
/// them (every identity's sections seal once on the base conv).
pub(super) struct IdentityBuilders {
    /// Per-identity keep-sets, computed once at load from the installed members.
    pub(super) keeps: HashMap<String, IdentityKeep>,
    /// The `mind.yaml` default identity, used when a conversation names none.
    pub(super) default_identity: Option<String>,
    pub(super) cache: Mutex<HashMap<(String, ToolMode), Arc<Builder>>>,
}

impl IdentityBuilders {
    /// Build the keep-sets from the loaded identity sections (member ids
    /// `<name>::<stem>`). `default_identity` comes from `mind.yaml`.
    pub(super) fn new(
        sections: &crate::response_section::IdentitySections,
        default_identity: Option<String>,
    ) -> Self {
        let mut keeps: HashMap<String, IdentityKeep> = HashMap::new();
        for s in &sections.anchors {
            if let Some((name, _)) = s.id.split_once("::") {
                keeps
                    .entry(name.to_string())
                    .or_default()
                    .anchor
                    .insert(s.id.clone());
            }
        }
        for s in &sections.facets {
            if let Some((name, _)) = s.id.split_once("::") {
                keeps
                    .entry(name.to_string())
                    .or_default()
                    .facets
                    .insert(s.id.clone());
            }
        }
        Self {
            keeps,
            default_identity,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Resolve which identity scope a turn's projection uses: the conversation's
    /// identity, else the `mind.yaml` default. Three genuine states — collapsing
    /// them onto `Option` is what leaked every anchor: an unresolved identity when
    /// identities ARE installed must scope to empty, NOT fall through to the
    /// unscoped builder (whose AlwaysVisible `identity_anchor` collection emits
    /// every persona's anchor at once).
    pub(super) fn resolve(&self, conv_identity: Option<&str>) -> IdentityScope {
        if self.keeps.is_empty() {
            return IdentityScope::Unscoped;
        }
        match conv_identity
            .map(str::to_string)
            .or_else(|| self.default_identity.clone())
        {
            Some(name) => IdentityScope::Named(name),
            None => IdentityScope::Empty,
        }
    }

    /// Retain both identity collections on `b` to `name`'s installed members. An
    /// unknown name — including the reserved empty sentinel — scopes to empty. The
    /// single home of the identity retain, shared by `get`, `get_empty`, and the
    /// hires capture path so scoping can never diverge between them.
    pub(super) fn scope_to(&self, b: &mut Builder, name: &str) {
        let empty = HashSet::new();
        let (anchor_keep, facet_keep) = match self.keeps.get(name) {
            Some(k) => (&k.anchor, &k.facets),
            None => (&empty, &empty),
        };
        if let Err(e) = b.retain_collection_sections("identity_anchor", anchor_keep) {
            tracing::warn!("scope identity_anchor to {name}: {e}");
        }
        if let Err(e) = b.retain_collection_sections("identity", facet_keep) {
            tracing::warn!("scope identity to {name}: {e}");
        }
    }

    /// Scope BOTH identity collections to empty — no anchor, no facets. Used when
    /// identities are installed but none resolved (see [`IdentityScope::Empty`]).
    /// Cached under a reserved key that no real identity name can collide with.
    pub(super) fn get_empty(&self, mode_builders: &ModeBuilders, mode: ToolMode) -> Arc<Builder> {
        let key = ("\0empty".to_string(), mode);
        if let Some(b) = self.cache.lock().unwrap().get(&key) {
            return Arc::clone(b);
        }
        let mut b = (*mode_builders.get(mode)).clone();
        self.scope_to(&mut b, "\0empty"); // sentinel is not a real identity → empty
        let arc = Arc::new(b);
        self.cache.lock().unwrap().insert(key, Arc::clone(&arc));
        arc
    }

    /// Apply the conversation's identity scope to an arbitrary builder. The hires
    /// capture path clones the UNSCOPED base, so without this a capture would emit
    /// every persona's always-visible anchor. No-op when no identities are
    /// installed (the coding-assistant schema).
    pub(super) fn scope_hires(&self, b: &mut Builder, conv_identity: Option<&str>) {
        match self.resolve(conv_identity) {
            IdentityScope::Named(name) => self.scope_to(b, &name),
            IdentityScope::Empty => self.scope_to(b, "\0empty"),
            IdentityScope::Unscoped => {}
        }
    }

    /// The projection scoped to `name` for `mode` — cloned from the tools-mode
    /// builder and retained on first use, then cached. An unknown `name` scopes
    /// both identity collections to EMPTY (no identity surfaces) with a warning,
    /// never the unscoped builder — which would leak every identity's
    /// always-visible anchor at once.
    pub(super) fn get(
        &self,
        mode_builders: &ModeBuilders,
        name: &str,
        mode: ToolMode,
    ) -> Arc<Builder> {
        let key = (name.to_string(), mode);
        if let Some(b) = self.cache.lock().unwrap().get(&key) {
            return Arc::clone(b);
        }
        if !self.keeps.contains_key(name) {
            tracing::warn!(identity = %name, "unknown identity — scoping to empty (no identity surfaced)");
        }
        let mut b = (*mode_builders.get(mode)).clone();
        self.scope_to(&mut b, name);
        let arc = Arc::new(b);
        self.cache.lock().unwrap().insert(key, Arc::clone(&arc));
        arc
    }
}

/// Clone the live projection builder with a high-resolution capture override.
///
/// `spec` selects what to force:
/// - `"tools"` — force the whole `tools` collection to AllVisible (all sections);
/// - `"tools/datetime"` — emit ONLY the `datetime` section of `tools`.
///
/// The clone shares all section ids with the base, so a forked sequence's
/// already-sealed section KV is reused unchanged.
pub(super) fn build_hires_projection(
    state: &InferenceState,
    spec: &str,
    conv_identity: Option<&str>,
) -> anyhow::Result<Arc<Builder>> {
    let mut b = state.refresh_builder.clone();
    if let Some((collection, section)) = spec.split_once('/') {
        b.set_collection_single_section(collection, section)
            .map_err(|e| anyhow::anyhow!("force-hires '{spec}': {e}"))?;
    } else {
        b.set_collection_selection(spec, SelectionRule::AlwaysVisible)
            .map_err(|e| anyhow::anyhow!("force-hires '{spec}': {e}"))?;
    }
    // Scope to the conversation's identity so a capture reflects its real
    // projection, not every persona's always-visible anchor — the base
    // `refresh_builder` this clones is unscoped. Independent of the forced
    // collection above, so order does not matter.
    state.identity_builders.scope_hires(&mut b, conv_identity);
    Ok(Arc::new(b))
}

impl ZendSession {
    /// The projection schema this session runs — its layers, as parsed.
    pub fn projection_builder(&self) -> &Builder {
        &self.projection_builder
    }
}

/// The workspace's projection schema — its `projection.yaml` or the embedded
/// default. A schema that does not parse stops zend at its load: there is no
/// daemon to run without one.
pub(super) fn build_projection_builder(workspace: &Path) -> Builder {
    try_projection_builder(workspace).unwrap_or_else(|e| panic!("{e}"))
}

/// [`build_projection_builder`], with a schema that does not parse as an
/// error — for a host reading zend's settings without running zend.
pub(crate) fn try_projection_builder(workspace: &Path) -> Result<Builder, String> {
    let name = workspace
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("this project");
    // Qwen3-30B-A3B is a ChatML-family model; the dialect is required
    // so the schema's `kind: template` items (system_open / system_close,
    // tool_block_open/close, no_think_prefix) resolve to the right
    // structural-token strings at parse time.
    let dialect = Dialect::chat_ml();
    // A `projection.yaml` in the workspace OVERRIDES the embedded schema, so a
    // dedicated (uncommitted) mind directory — e.g. `--working-dir ../mind` — can
    // tune its own cognitive substrate without touching the checked-in default.
    let override_path = workspace.join("projection.yaml");
    let overridden = std::fs::read_to_string(&override_path).ok();
    let (yaml, source): (&str, String) = match &overridden {
        Some(s) => {
            tracing::info!(path = %override_path.display(), "projection schema: workspace override");
            (s.as_str(), override_path.display().to_string())
        }
        None => (PROJECTION_SCHEMA_TEMPLATE, "embedded default".to_string()),
    };
    let model = ModelView::of_model(&MODEL);
    Builder::from_yaml_with_vars_and_dialect(
        yaml,
        &[("workspace", name), ("model", model.release())],
        Some(&dialect),
    )
    .map_err(|e| format!("projection.yaml ({source}) failed to parse — check YAML syntax and its {{workspace}} and {{model}} placeholders: {e:#}"))
}

/// Concatenate the content of every top-level Section that appears
/// **before** the dialogue layer's first Collection, returning the
/// joined text.  Used as the engine's `system_prompt:` argument — it
/// gets ChatML-wrapped by the model builder, then handed to the
/// conversation as the pending-prefill prelude.  Everything from the
/// first Collection onward is expanded inside
/// [`Sequence::preemptive_prefill`] (collection sections via
/// fork-and-merge, post-collection sections via per-section prefill).
pub(super) fn pre_collection_prelude(builder: &Builder) -> String {
    let mut out = String::new();
    for item in &builder.schema().system_prompt.items {
        match item {
            SystemPromptItem::Section(s) => out.push_str(&s.content),
            SystemPromptItem::SectionTree(t) => {
                // Default selection: each node's default option content, in order,
                // STOPPING at the embedded `tools` collection node (the prelude is
                // the text that precedes the tools; a collection node has no
                // options to render and everything below it is post-tools).
                for n in &t.nodes {
                    if n.collection.is_some() {
                        return out;
                    }
                    // Glue markers (`<tools>` etc.) are live-prefilled at projection,
                    // not part of the static system-prompt prelude — skip them here.
                    if n.glue.is_some() {
                        continue;
                    }
                    out.push_str(&n.options[n.chosen(&t.default_selection)].content);
                }
            }
            SystemPromptItem::Collection(_) => break,
        }
    }
    out
}
