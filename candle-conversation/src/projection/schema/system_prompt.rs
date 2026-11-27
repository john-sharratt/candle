//! The schema of the one shared system prompt: its sections, the collections
//! among them, and the selector trees that branch it.
//!
//! The prompt is a list of items emitted in declaration order. A single section
//! always emits, and a collection emits only the members its selection rule
//! keeps, which is how a dynamic tool catalog sits inside static framing. A
//! `SectionTree` seals the full cross-product of its selector options ahead of
//! time, so changing a selection picks a variant that is already prefilled rather
//! than re-prefilling the nodes beneath it. Collection members are kept out of the
//! content-address chain, because projection picks a subset at runtime and every
//! section below a collection is already conditioned on a prefix that ignores
//! which members were picked.

use super::*;

/// Schema for one layer's system-prompt content.
///
/// Holds an ordered list of [`SystemPromptItem`]s — each item is either a
/// single [`SectionSchema`] (always emits) or a [`SectionCollection`] (a
/// named bucket with its own selection rule, only the surviving subset
/// emits).  Items emit in declaration order; sections inside a collection
/// also emit in their declaration order, after the collection's selection
/// rule has filtered them.
///
/// This shape lets a single layer's system prompt mix static framing
/// (always-emit) with dynamic catalogs (top-k filtered) at well-defined
/// positions — typical use is a chat dialect prompt with a `<tools>`
/// envelope: static intro section, then a `tools` collection that picks
/// the K most relevant tool definitions, then a static outro section.
///
/// Lives on [`LayerSchema::system_prompt`]. When projection runs for a
/// target `(layer, group)`, the emitted system prompt is **the target
/// layer's** items in declaration order.
#[derive(Debug, Clone, Default)]
pub struct SystemPromptSchema {
    /// In declaration order — interleaves single sections with
    /// collections.
    pub items: Vec<SystemPromptItem>,
    /// The items every selecting query attends whole, and how many positions
    /// they may cover (`pin: true` on the items, `pin_tokens:`).
    pub pins: PinSet,
}

impl SystemPromptSchema {
    /// Iterate every section in this system_prompt, regardless of
    /// whether it's a top-level item or nested inside a collection.
    /// Yields in declaration order.  Used for diagnostic walks; the
    /// projection emitter walks `items` directly so it can apply each
    /// collection's selection rule.
    pub fn all_sections(&self) -> impl Iterator<Item = &SectionSchema> {
        self.items.iter().flat_map(|it| match it {
            SystemPromptItem::Section(s) => std::slice::from_ref(s).iter(),
            SystemPromptItem::Collection(c) => c.sections.iter(),
            // Tree nodes are not `SectionSchema`s (they carry per-branch
            // variants); address them via [`Self::all_section_ids`] instead.
            SystemPromptItem::SectionTree(_) => [].iter(),
        })
    }

    /// Every substrate [`super::SectionId`] this system prompt owns — bare
    /// sections, collection members, **and** every section-tree variant.  Used
    /// for id-space accounting (max id, emptiness) where tree variants must be
    /// counted even though they aren't `SectionSchema`s.
    pub fn all_section_ids(&self) -> impl Iterator<Item = SectionId> + '_ {
        self.items
            .iter()
            .flat_map(|it| -> Box<dyn Iterator<Item = SectionId>> {
                match it {
                    SystemPromptItem::Section(s) => Box::new(std::iter::once(s.id)),
                    SystemPromptItem::Collection(c) => Box::new(c.sections.iter().map(|s| s.id)),
                    SystemPromptItem::SectionTree(t) => Box::new(t.nodes.iter().flat_map(|n| {
                        n.options
                            .iter()
                            .flat_map(|o| o.variants.iter().map(|v| v.id))
                            .chain(
                                n.collection
                                    .iter()
                                    .flat_map(|tc| tc.variants.iter().flatten().map(|v| v.id)),
                            )
                    })),
                }
            })
    }

    /// Find a [`SectionCollection`] by name, walking only top-level
    /// items.  Returns `None` for unknown names.
    pub fn collection_named(&self, name: &str) -> Option<&SectionCollection> {
        self.items.iter().find_map(|it| match it {
            SystemPromptItem::Collection(c) if c.name == name => Some(c),
            _ => None,
        })
    }

    /// True when `section_id` belongs to any `SectionCollection` in
    /// this layer's prompt.  Used by the content-address chain in
    /// `Sequence::insert_section_collection` to avoid hashing
    /// collection-member tokens into the prefix — without this
    /// filter, every change to a collection member (e.g. installing
    /// or removing a single tool) would cascade into a new
    /// `prefix_hash` for every downstream section, force-invalidating
    /// the manifest entries that would otherwise have cold-loaded.
    /// Collection members are an approximation-rich prefix anyway —
    /// projection picks a subset at runtime, so the section's K/V
    /// already isn't a strict function of which specific members
    /// ingested.  Treating them as outside the content chain matches
    /// that existing approximation.
    pub fn is_collection_member(&self, section_id: super::super::ids::SectionId) -> bool {
        self.items.iter().any(|it| match it {
            SystemPromptItem::Collection(c) => c.sections.iter().any(|s| s.id == section_id),
            // Members of a collection embedded as a tree node are collection
            // members too — every per-branch variant id is one.
            SystemPromptItem::SectionTree(t) => t.nodes.iter().any(|n| {
                n.collection
                    .as_ref()
                    .is_some_and(|tc| tc.variants.iter().flatten().any(|v| v.id == section_id))
            }),
            _ => false,
        })
    }

    /// Every [`SectionTree`] the schema declares, in item order.
    ///
    /// A schema may declare **more than one** — zend's bundled
    /// `projection.yaml` declares two, and a branch entry naming a node in the
    /// second is invisible to anything that reads only the first. That is not
    /// hypothetical: this returned `Option<&SectionTree>` on the premise that
    /// "a schema has at most one", and a branch validation that read only the
    /// first tree silently skipped the option it existed to pin, because that
    /// option lived in the second.
    pub fn section_trees(&self) -> impl Iterator<Item = &SectionTree> {
        self.items.iter().filter_map(|it| match it {
            SystemPromptItem::SectionTree(t) => Some(t),
            _ => None,
        })
    }
}

/// One entry in a layer's system-prompt list.  Either a single
/// always-emit section or a named collection with its own selection
/// rule.
#[derive(Debug, Clone)]
pub enum SystemPromptItem {
    /// Authored static framing.  Always emits in declaration order
    /// regardless of any resolver-supplied scores.
    Section(SectionSchema),
    /// A named bucket of sections with its own selection rule.  Only
    /// the surviving subset emits, but in declaration order.
    Collection(SectionCollection),
    /// An ordered, individually-toggleable tree of sealed sections.  See
    /// [`SectionTree`].
    SectionTree(SectionTree),
}

/// An ordered tree of system-prompt sections where some nodes are SELECTORS —
/// each emits one of N mutually-exclusive options.  The binary `optional` node
/// is the 2-option special case (`present` content vs an empty `absent`); a
/// `mandatory` node is the 1-option case.
///
/// Because a sealed section's K/V is conditioned on everything injected before
/// it, every selector multiplies the possible prefixes for the nodes below it.
/// The tree resolves this by **pre-sealing the full cross-product**: each option
/// carries one [`TreeVariant`] per assignment of the selector dimensions
/// declared above its node, each sealed with that branch's exact prefix.
/// Projection resolves the active selection (one option per selector id) and
/// emits, for each node, the chosen option's variant for that branch — so
/// changing a selector picks a pre-prefilled variant instead of re-prefilling
/// the nodes beneath it.
///
/// Lives as one [`SystemPromptItem::SectionTree`] at its declared position in
/// the layer's system-prompt stream.
#[derive(Debug, Clone, Default)]
pub struct SectionTree {
    /// Root-first declaration order.
    pub nodes: Vec<TreeNode>,
    /// The selector dimensions, in declaration order — one per node with >1
    /// option.  A node's variant key is a mixed-radix pack of the dims declared
    /// before it (see [`Self::pack`]).
    pub dims: Vec<TreeDim>,
    /// Default option index per dim — the selection used when the runtime
    /// supplies no override.
    pub default_selection: Vec<u8>,
    /// The default selection's emitted (non-empty) variant ids, in tree order.
    /// Sections declared **after** the tree (and priming) attend to this branch.
    pub default_present_ids: Vec<SectionId>,
}

impl SectionTree {
    /// Resolve the active selection — option index per dim.  `resolve(selector_id)`
    /// returns the chosen option id, or `None` for the authored default; an
    /// unknown option id also falls back to the default.
    pub fn selection<'a>(&self, mut resolve: impl FnMut(&str) -> Option<&'a str>) -> Vec<u8> {
        self.dims
            .iter()
            .map(|d| {
                let node = &self.nodes[d.node_index];
                match resolve(&d.selector_id) {
                    Some(opt) => node
                        .options
                        .iter()
                        .position(|o| o.id == opt)
                        .map(|i| i as u8)
                        .unwrap_or(d.default_option),
                    None => d.default_option,
                }
            })
            .collect()
    }

    /// Pack the sub-assignment over the first `dim_count` dims into a mixed-radix
    /// key — the [`TreeVariant::ancestors`] key for a node with that many
    /// ancestor dims.
    ///
    /// A gated dim ([`TreeDim::gate`]) that is currently OUT OF SCOPE (its
    /// enclosing group is on the other branch) is masked to its `default_option`,
    /// because the sealed variant for that branch was keyed with the default — the
    /// out-of-scope dim never multiplied it.  Trees with no gated dims pack
    /// byte-identically to the raw `selection`.
    pub fn pack(&self, selection: &[u8], dim_count: usize) -> u32 {
        let mut key = 0u32;
        let mut mult = 1u32;
        for d in 0..dim_count {
            let val = if self.dim_active(selection, d) {
                selection[d]
            } else {
                self.dims[d].default_option
            };
            key += val as u32 * mult;
            mult *= self.dims[d].option_count as u32;
        }
        key
    }

    /// Whether dim `d` is in scope for `selection` — i.e. every gate in its chain
    /// (the enclosing `optional_group` toggles) holds its active value.  A gate
    /// dim `g` is always declared before `d`, so the recursion is well-founded.
    pub(super) fn dim_active(&self, selection: &[u8], d: usize) -> bool {
        match self.dims[d].gate {
            None => true,
            Some((g, v)) => selection[g] == v && self.dim_active(selection, g),
        }
    }

    /// The ordered sealed sections a branch's K/V is built from, for `selection`.
    ///
    /// This is the **state's** view of the prompt, and it differs from what
    /// projection emits in exactly one place: a placeholder node
    /// ([`TreeNode::inject_collection`]) contributes its own sealed anchor
    /// content rather than the collection's runtime top-k. That is deliberate
    /// and inherited, not a new decision — `SystemPrompt::is_collection_member`
    /// already excludes collection members from the content-address prefix
    /// chain because projection picks a subset at runtime, so every sealed K/V
    /// below a collection is *already* conditioned on a prefix that ignores
    /// which members are present. Computing the recurrent state over the same
    /// placeholder-substituted prefix makes the state and the K/V approximate
    /// the collection identically, instead of the state carrying an
    /// approximation of its own.
    ///
    /// Prefix-transparent nodes (embedded collections, glue markers) contribute
    /// nothing, exactly as they contribute nothing to the in-tree prefix the
    /// variants below them were sealed against.
    pub fn branch_prefix_ids(&self, selection: &[u8]) -> Vec<SectionId> {
        let mut out = Vec::new();
        for node in &self.nodes {
            if node.glue.is_some() || node.collection.is_some() {
                continue;
            }
            let Some(option) = node.options.get(node.chosen(selection)) else {
                continue;
            };
            if let Some(v) = option.variant_for(self.pack(selection, node.ancestor_dims)) {
                out.push(v.id);
            }
        }
        out
    }
}

/// One selector dimension of a [`SectionTree`] — a node with more than one
/// option.
#[derive(Debug, Clone)]
pub struct TreeDim {
    /// The selector id — the stable string the runtime sets (e.g.
    /// `thinking_effort`, or a binary node's own name).
    pub selector_id: String,
    /// Which node owns this dimension.
    pub node_index: usize,
    /// Number of options (the radix).
    pub option_count: u8,
    /// Default option index when the runtime supplies no selection.
    pub default_option: u8,
    /// `Some((g, v))` when this dim lives INSIDE an `optional_group` branch — it
    /// is "active" only while gate-dim `g` (an enclosing group's toggle) holds
    /// value `v` (the active side).  When the gate is off the dim is OUT OF SCOPE:
    /// the other branch never multiplied by it, and was sealed at this dim's
    /// `default_option`, so [`SectionTree::pack`] masks the runtime value back to
    /// the default to land on that sealed variant.  `None` for top-level dims
    /// (always active — the common case, masking is a no-op).
    pub gate: Option<(usize, u8)>,
}

/// One node in a [`SectionTree`] — a section that emits one of its options, or
/// a prefix-transparent embedded collection ([`Self::collection`]).
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Declared id (its YAML `id:`).
    pub name: String,
    /// The options this node can emit.  `len() == 1` ⇒ mandatory (always that
    /// content).  `len() > 1` ⇒ a selector dimension (see [`Self::dim`]).
    /// Empty when this node is a collection node ([`Self::collection`]).
    pub options: Vec<TreeOption>,
    /// `Some` when this node is an embedded collection: its members are scored
    /// + top-k selected at projection, sealed ×(ancestor dims), and the node
    /// adds NOTHING to the in-tree prefix of nodes below it.
    pub collection: Option<TreeCollection>,
    /// `Some(cid)` when this node is a PLACEHOLDER whose *projection* is replaced
    /// by collection `cid`'s top-k selection.  The node still seals + emits its
    /// own content into the K/V prefix (so nodes below it anchor on a stable
    /// placeholder, e.g. a `noop` tool), but at projection it is swapped for the
    /// provenance-selected real members.  The referenced collection sets
    /// [`TreeCollection::deferred_projection`] so it does not also emit.
    pub inject_collection: Option<CollectionId>,
    /// `Some` when this node is a live-prefilled structural marker (a `dialect:`
    /// section, e.g. `<tools>` / `</tools>`) rather than sealed content: it
    /// allocates NO sealed variants, is PREFIX-TRANSPARENT (nodes below seal as if
    /// it were not there), and at projection emits a [`ProjectionSegment::Generated`]
    /// run — but only in the branches it lives in (gated, see [`TreeGlue`]).  Its
    /// marker text stays in `options[0].content` (for tokenisation + the GUI label).
    pub glue: Option<TreeGlue>,
    /// Index into [`SectionTree::dims`] when this node is a selector, else `None`.
    pub dim: Option<usize>,
    /// How many selector dims are declared **before** this node — the width of
    /// its ancestor assignment (the `dim_count` for [`SectionTree::pack`]).
    pub ancestor_dims: usize,
}

/// A live-prefilled structural marker in a [`SectionTree`] ([`TreeNode::glue`]).
#[derive(Debug, Clone)]
pub struct TreeGlue {
    /// The marker's pre-tokenised tokens (populated by
    /// [`super::Builder::tokenize_templates`]); the projection emits these as a
    /// [`ProjectionSegment::Generated`] run, re-derived each projection.
    pub tokens: Option<std::sync::Arc<Vec<u32>>>,
    /// The packed ancestor keys (over the node's `ancestor_dims`) of the branches
    /// this glue lives in — it emits only when the active selection's key is one
    /// of these (e.g. a `<tools>` marker inside the `tools_enabled` present branch
    /// emits only when tools are on).
    pub active_keys: Vec<u32>,
}

impl TreeNode {
    /// The selector id this node exposes (its name when it is a dimension), or
    /// `None` for a mandatory or collection node.
    pub fn selector_id(&self) -> Option<&str> {
        (self.options.len() > 1).then_some(self.name.as_str())
    }

    /// The option index this node emits under `selection`.
    pub fn chosen(&self, selection: &[u8]) -> usize {
        self.dim.map_or(0, |d| selection[d] as usize)
    }
}

/// A collection embedded as a [`TreeNode`] — scored + top-k selected at
/// projection exactly like a top-level [`SectionCollection`], but
/// PREFIX-TRANSPARENT (it contributes nothing to the in-tree prefix of nodes
/// below it) and sealed ×(ancestor dims): each member carries one sealed
/// variant per ancestor-branch assignment, so the members fan out across every
/// outer selector declared above this node while the cheap selectors *below*
/// it never multiply the members.
#[derive(Debug, Clone)]
pub struct TreeCollection {
    /// Scoring + selection config and the canonical member schemas (name,
    /// content, priority, [`CollectionId`] for `depends_on` gating).  The
    /// `SectionSchema::id`s here are the default-branch variant ids; the full
    /// per-branch sealed ids live in [`Self::variants`].
    pub collection: SectionCollection,
    /// Per member (declaration order) → one sealed [`TreeVariant`] per ancestor
    /// branch.  `variants[member_index]` mirrors the member's per-branch seals.
    pub variants: Vec<Vec<TreeVariant>>,
    /// The ancestor branches this node fans out over — `(packed ancestor key,
    /// in-tree prefix)` per outer-selector assignment, captured at build time.
    /// Runtime member addition (`add_section_to_collection`) allocates one
    /// sealed variant per branch from these templates.
    pub branches: Vec<(u32, Vec<SectionId>)>,
    /// The packed ancestor key of the default branch — the canonical variant
    /// (the one that backs scoring, `depends_on` gating, and name resolution).
    pub default_branch: u32,
    /// When `true`, this collection does NOT emit its selection at its own tree
    /// position — a placeholder node ([`TreeNode::inject_collection`]) below it
    /// emits the top-k members instead.  Sealing is unchanged; only projection
    /// position moves.
    pub deferred_projection: bool,
}

impl TreeCollection {
    /// The active-branch sealed variant for member `member_index`.
    pub fn member_variant(&self, member_index: usize, ancestors: u32) -> Option<&TreeVariant> {
        self.variants
            .get(member_index)?
            .iter()
            .find(|v| v.ancestors == ancestors)
    }
}

/// One mutually-exclusive option of a [`TreeNode`].
#[derive(Debug, Clone)]
pub struct TreeOption {
    /// Option id (e.g. `present`/`absent` for a binary node, `off`..`exhaustive`
    /// for a selector).
    pub id: String,
    /// Authored/resolved content.  May be empty (e.g. a binary node's `absent`),
    /// in which case the option emits nothing and seals no variants.
    pub content: String,
    /// One sealed variant per ancestor-dim assignment (packed key).  Empty when
    /// `content` is empty.
    pub variants: Vec<TreeVariant>,
}

impl TreeOption {
    /// The sealed variant for a packed ancestor assignment, if this option was
    /// sealed (non-empty content).
    pub fn variant_for(&self, ancestors: u32) -> Option<&TreeVariant> {
        self.variants.iter().find(|v| v.ancestors == ancestors)
    }
}

/// One pre-sealed branch of a [`TreeOption`] — its substrate [`SectionId`] plus
/// the in-tree prefix it was sealed against.
#[derive(Debug, Clone)]
pub struct TreeVariant {
    /// Packed mixed-radix ancestor assignment this variant is sealed for.
    pub ancestors: u32,
    /// This variant's substrate section id — its prefix-conditioned K/V.
    pub id: SectionId,
    /// The present-ancestor variant ids forming this branch's in-tree prefix, in
    /// order.  Ingest prepends the pre-tree content prefix before sealing.
    pub in_tree_prefix: Vec<SectionId>,
}

/// A named bucket of system-prompt sections with its own selection rule.
///
/// Sections inside a collection are individually scored (typically via
/// per-section provenance sigs in the substrate) and filtered by
/// [`Self::selection`].  The surviving subset emits in declaration order
/// at the position of the collection within the system_prompt's items.
///
/// Typical use is a `tools` collection embedded in a dialogue layer's
/// system prompt: 93 tool-definition sections, `selection: TopK { k: 3 }`,
/// driven by provenance scoring against the user's recent intent.
#[derive(Debug, Clone)]
pub struct SectionCollection {
    /// Crate-assigned id.
    pub id: CollectionId,
    /// Original declared name (from YAML or builder injection).
    /// Layer-scoped — collections in different layers may share a name.
    pub name: String,
    /// Sections that belong to this collection.  Each retains its own
    /// [`SectionId`] — the collection is purely a selection-time
    /// grouping.
    pub sections: Vec<SectionSchema>,
    /// Which sections survive selection.  Default
    /// [`SelectionRule::AlwaysVisible`] (collection acts as a label
    /// only); typical TopK to surface the most relevant subset.
    pub selection: SelectionRule,
    /// Sections below this score are filtered before selection.
    /// Default `0.0`.
    pub score_threshold: f32,
    /// Selection policy for this collection. Resolved at build time from the
    /// collection's `policy:` or inherited from the enclosing layer. Its
    /// budget/thresholds subsume the collection's `selection`/`score_threshold`
    /// for belief-driven selection.
    pub policy: SelectionPolicy,
    /// The runtime-sealed summary section ([`Reserved::ToolSummary`](super::Reserved::ToolSummary)),
    /// set once the catalog summary is generated. When `Some` and the collection's
    /// selection is a *proper subset* of its members (top-k dropped at least one),
    /// projection emits this section's sealed K/V just before the selected members —
    /// a compact overview of everything, including what wasn't selected. Omitted when
    /// all members are selected (nothing was dropped) or the section isn't sealed yet.
    pub summary_section: Option<SectionId>,
    /// Glue emitted BETWEEN consecutive selected members at projection — a real
    /// structural token (e.g. a newline) that is NOT baked into any member's
    /// seal, so it is independent of which members provenance selects.  `member_glue`
    /// is the raw text; [`Self::member_glue_tokens`] is its build-time tokenisation
    /// (populated by [`super::Builder::tokenize_templates`]).  Empty ⇒ no glue.
    pub member_glue: String,
    /// Tokenised [`Self::member_glue`], live-prefilled between members like a
    /// structural template.  `None` until tokenised (or when `member_glue` is empty).
    pub member_glue_tokens: Option<std::sync::Arc<Vec<u32>>>,
    /// Fallback section (by name) emitted when belief/scores select no member,
    /// so the collection always contributes at least one section. `None` =
    /// today's behaviour.
    pub default: Option<SelectionDefault>,
    /// Concept B: mass-driven member-budget extension (`k` grows with the
    /// collection's attention mass, capped at `absolute_max`). `None` = static.
    pub budget_adaptive: Option<MemberBudgetAdaptive>,
    /// Members emitted on every belief-driven projection, outside the budget:
    /// the policy selects its top-k from the OTHER members, so a mandatory
    /// member adds to the selection instead of taking one of its slots. Set per
    /// member at runtime (`Builder::set_collection_member_mandatory`), e.g. from
    /// a tool definition's `mandatory: true`.
    pub mandatory: Vec<SectionId>,
    /// The positions the collection always takes, its members at the top of
    /// them and a hole ahead (`docs/placed_section_layout.md` §3.1). A
    /// collection with none takes exactly what its members do.
    pub range: Option<ReservedRange>,
}

impl Default for SectionCollection {
    fn default() -> Self {
        Self {
            id: CollectionId::new(1),
            name: String::new(),
            sections: Vec::new(),
            selection: SelectionRule::AlwaysVisible,
            score_threshold: 0.0,
            policy: SelectionPolicy::default_policy(),
            summary_section: None,
            member_glue: String::new(),
            member_glue_tokens: None,
            default: None,
            budget_adaptive: None,
            mandatory: Vec::new(),
            range: None,
        }
    }
}

/// A single authored section of the system prompt.
///
/// Always emits in declaration order when used as a top-level
/// `SystemPromptItem::Section`.  When nested inside a
/// [`SectionCollection`], emission is gated by the collection's
/// selection rule; the section's `priority` then breaks score ties.
#[derive(Debug, Clone)]
pub struct SectionSchema {
    /// Crate-assigned id.
    pub id: SectionId,
    /// Original declared `id:` string. Kept for diagnostics and name lookup.
    pub name: String,
    /// Authored text. The crate does not tokenize; the caller resolves
    /// content from `id` after projection emits a [`super::ResolvedSection`].
    pub content: String,
    /// Static fallback priority used as a score-tie breaker inside a
    /// collection's selection.  Higher = preferred.  Default `50.0`.
    pub priority: f32,
    /// Conditional emission gate: when `Some(cid)`, this section only
    /// emits at projection time if the named [`SectionCollection`] in
    /// the same layer materialised ≥ 1 of its members. Ingested
    /// unconditionally — the substrate always has its bytes — so the
    /// emission check is purely a projection-time predicate.
    ///
    /// Used by the YAML schema to wrap collections in structural
    /// markers (e.g. `<tools>` / `</tools>`) that should only appear
    /// when the collection itself emits anything.
    pub depends_on: Option<CollectionId>,
    /// Inverse of [`Self::depends_on`]: when `Some(cid)`, this section emits only
    /// when the named collection materialised **zero** members. Lets a layer
    /// carry two variants of a section — one gated `depends_on` a collection,
    /// the other `depends_on_absent` the same collection — so exactly one shows
    /// (e.g. a tools-aware vs a no-tools grounding paragraph).
    pub depends_on_absent: Option<CollectionId>,
    /// Emission gate on whether the named collection is **configured** — has ≥ 1
    /// MEMBER — rather than on what it selected this projection.
    ///
    /// [`Self::depends_on`] asks "did provenance pick something *this instant*",
    /// which is the right question for the structural markers that wrap the
    /// picks (an empty `<tools></tools>` is pointless) and the WRONG question for
    /// the prose that introduces the facility. Gated on selection, a tool
    /// overview vanishes on exactly the turn provenance happens to score
    /// nothing — and the model, told nothing about tools, answers from memory
    /// instead of calling one. A collection with zero members is genuinely
    /// disabled (the tools dial set to `None` filters every member out), and
    /// that is what this asks.
    pub depends_on_configured: Option<CollectionId>,
    /// Inverse of [`Self::depends_on_configured`]: emits only when the named
    /// collection has **no** members, i.e. the facility is switched off.
    pub depends_on_unconfigured: Option<CollectionId>,
    /// Marks this section as resolved from a dialect template (a
    /// `kind: template` YAML item that referenced a `DialectTemplate`
    /// catalog entry, e.g. `system_start`).  The scheduler's
    /// projection assembler routes template-kind items through live
    /// prefill against the current runtime left context rather than
    /// the substrate-backed sealed path, so structural envelope K/V
    /// stays attention-correct under whatever prefix the projection
    /// selected this turn.
    pub is_template: bool,
    /// Pre-tokenised template content, populated by
    /// [`super::Builder::tokenize_templates`] before the first
    /// projection.  `Some` only when `is_template == true`; the
    /// projection engine emits a [`super::ProjectionSegment::Generated`]
    /// carrying these tokens for the assembler to inject as a live-
    /// prefilled run.  `None` for `is_template == false` sections —
    /// their K/V comes from the substrate-pinned sealed path.
    pub template_tokens: Option<std::sync::Arc<Vec<u32>>>,
}
