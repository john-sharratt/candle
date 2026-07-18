//! Static schema types — declared once, immutable thereafter.
//!
//! A schema declares **what content can exist where**. After construction it
//! never changes; content is appended into groups over time, but the schema
//! itself is frozen. Everything dynamic (token counts, scores) flows through
//! the [`super::ContentResolver`] at projection time.
//!
//! # Hierarchy
//!
//! ```text
//!  Schema
//!  └── layers: Vec<LayerSchema>
//!      └── LayerSchema { name, window, score_threshold,
//!                        budget, system_prompt, groups: Vec<GroupSchema> }
//!          ├── system_prompt: SystemPromptSchema  { sections: [SectionSchema] }
//!          │   └── (used when THIS layer is the projection target)
//!          └── GroupSchema { name, selection, score_threshold, budget }
//! ```
//!
//! Each layer carries its own system prompt because each cognitive layer
//! (dialogue, bug analysis, dream exploration, daily convergence …) is a
//! genuinely different conversation with its own framing. The system
//! prompt emitted by [`super::Builder::project`] is the **target** layer's
//! sections.
//!
//! # Budget model
//!
//! Every layer and group carries a [`Budget`] declaring its **priority**
//! (relative weight for proportional allocation), an optional **min** and an
//! optional **max** (as percentages of the parent budget). The reconciler in
//! [`super::reconcile`] treats this CSS-flexbox-style: priorities determine
//! ideal share, mins and maxes are bounds.
//!
//! Sections have no budget — they always emit in declaration order in full.
//!
//! Each [`LayerSchema`] also carries a `window` field — the total
//! turn-budget that flexbox slices when **this layer is the projection
//! target**. Different targets get different pies.
//!
//! # Defaults
//!
//! When the YAML omits a field, the crate fills in a default:
//!
//! | Field                 | Default                 |
//! |-----------------------|-------------------------|
//! | `Budget.priority`     | `50.0` (any positive — all defaults equal = even split) |
//! | `Budget.min_percent`  | `None`                  |
//! | `Budget.max_percent`  | `None`                  |
//! | `score_threshold`     | `0.0`                   |
//! | selection rule        | [`SelectionRule::AlwaysVisible`] |
//!
//! `LayerSchema.window` has no default — it must be declared.

use super::ids::{CollectionId, GroupId, LayerId, SectionId};
use super::policy::{PolicyConfig, SelectionPolicy};
use crate::summary_tree::scope::Scope;

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
    pub fn is_collection_member(&self, section_id: super::ids::SectionId) -> bool {
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
    fn dim_active(&self, selection: &[u8], d: usize) -> bool {
        match self.dims[d].gate {
            None => true,
            Some((g, v)) => selection[g] == v && self.dim_active(selection, g),
        }
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
    /// How this section group is compressed. Parsed, stored, and validated; no
    /// compression path reads a group summary yet (only layer summaries drive
    /// the live compression).
    pub summary: GroupSummary,
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
            summary: GroupSummary::default(),
            summary_section: None,
            member_glue: String::new(),
            member_glue_tokens: None,
            default: None,
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

/// A single compression framing + instruction pair — the unit the compression
/// path consumes.
///
/// The `system_prompt` is sealed once as a section (its [`SectionId`] is
/// allocated from the schema's section pool but never added to
/// `system_prompt.items`, so it only ever materialises via the compression seal,
/// never in a normal projection) and injected zero-copy at the head of a
/// compression pass. The `user_prompt` is prefilled as text after the content —
/// the "user turn" of the compression frame.
#[derive(Debug, Clone)]
pub struct CompressionPrompt {
    pub system_prompt: SectionSchema,
    pub user_prompt: String,
}

impl CompressionPrompt {
    /// A placeholder with an empty prompt and a stand-in section id, used by the
    /// `Default` impls below. Real schemas overwrite both via the YAML/builder
    /// path, which allocates a unique [`SectionId`] per prompt.
    fn placeholder(name: &str) -> Self {
        Self {
            system_prompt: SectionSchema {
                id: SectionId::new(1),
                name: name.to_string(),
                content: String::new(),
                priority: 50.0,
                depends_on: None,
                depends_on_absent: None,
                is_template: false,
                template_tokens: None,
            },
            user_prompt: String::new(),
        }
    }
}

/// How a compression level produces a summary's **assistant half** (its
/// content — the answer body).
///
/// Orthogonal to [`Scope`], which derives the *user* half. Any level may pair
/// any `Content` with any `Scope`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Content {
    /// The model decodes the content from the children under this level's
    /// [`CompressionPrompt`]. The default, and right for prose: an assistant
    /// half is exactly what a model decode is good at producing.
    #[default]
    Decode,
    /// The content is built deterministically from the children, with no model
    /// decode at all (`summary_tree::structural`). For content whose structure
    /// is fully determined by its input — directory trees — where a decode is
    /// pure cost and invites fabrication, and a faithful *merge* just unions the
    /// children so the result grows toward the root instead of compressing.
    Structural,
}

/// How one tree-level of a layer's turns is compressed.
///
/// A summary turn is a compressed *exchange* and keeps both bodies of a real
/// turn — the user-message half (`Role::User`, `[user_start, user_end)`) and the
/// assistant-response half (`Role::Assistant`, `[asst_start, total)`) — so it
/// stays role-coherent when re-injected, and carries a scope for retrieval to
/// match against. The two halves are produced by different means:
///
/// - the **user half** is always *derived* from the children's scopes
///   ([`Scope`]) — never decoded, because a decode always speaks as the
///   assistant and would be inventing a question that was never asked;
/// - the **assistant half** is produced per [`Content`]: decoded under
///   [`Self::assistant`], or built deterministically.
///
/// There is deliberately no user-half prompt: nothing would ever run it.
#[derive(Debug, Clone)]
pub struct TurnSummary {
    /// Derives the user-message half (the scope) from the children's scopes.
    pub scope: Scope,
    /// Compresses the assistant-response half of a turn (`Role::Assistant`).
    /// Unused when `content` is [`Content::Structural`].
    pub assistant: CompressionPrompt,
    /// Hard decode-token ceiling for the compressed assistant half.
    pub max_tokens: usize,
    /// How the assistant half is produced: decoded (default) or structural.
    pub content: Content,
}

impl Default for TurnSummary {
    fn default() -> Self {
        Self {
            scope: Scope::default(),
            assistant: CompressionPrompt::placeholder("__summary_assistant__"),
            max_tokens: 0,
            content: Content::Decode,
        }
    }
}

/// How a layer's turns are compressed across the summary tree.
///
/// `turns` drives `SummaryOfTurns` nodes (compressing raw turns); `summaries`
/// drives `SummaryOfSummaries` nodes (compressing already-compressed children).
/// When `summaries` is `None`, those nodes reuse `turns`.
#[derive(Debug, Clone, Default)]
pub struct LayerSummary {
    pub turns: TurnSummary,
    pub summaries: Option<TurnSummary>,
}

impl LayerSummary {
    /// The [`TurnSummary`] driving compression for a node: `summaries` for a
    /// `SummaryOfSummaries` node (falling back to `turns`), else `turns`.
    pub fn for_kind(&self, is_summary_of_summaries: bool) -> &TurnSummary {
        if is_summary_of_summaries {
            self.summaries.as_ref().unwrap_or(&self.turns)
        } else {
            &self.turns
        }
    }

    /// Push every compression-prompt [`SectionId`] (raw) this layer summary owns
    /// into `out` — the `turns` halves and, if present, the `summaries` halves.
    fn push_section_ids(&self, out: &mut Vec<u32>) {
        self.turns.push_section_ids(out);
        if let Some(s) = &self.summaries {
            s.push_section_ids(out);
        }
    }
}

impl TurnSummary {
    fn push_section_ids(&self, out: &mut Vec<u32>) {
        out.push(self.assistant.system_prompt.id.raw());
    }
}

/// How a section group (collection) is compressed.
///
/// A group is a catalog of sections, so it is summarised by a two-stage
/// **categorize → assign** workflow rather than a single compression pass (a
/// model cannot faithfully reproduce dozens of section names in one shot — see
/// `zend/examples/compress_tools.rs`): stage 1 the model proposes the category
/// labels; stage 2 it assigns each section to one by number, over chunks. The
/// numbers map back to the real section names in code, with a deterministic
/// name-token fallback, so no name can be invented. Parsed, stored, and
/// validated; the execution path that reads it is a follow-up.
#[derive(Debug, Clone)]
pub struct GroupSummary {
    /// Stage 1 — the model proposes the categories.
    pub categorize: GroupSummaryStage,
    /// Stage 2 — assign each section to one of the fixed categories, over chunks
    /// of [`Self::chunk`] sections.
    pub assign: GroupSummaryStage,
    /// Sections per stage-2 assignment chunk.
    pub chunk: usize,
}

/// One stage of a group's categorize→assign workflow: a compression prompt plus
/// its decode-token ceiling.
#[derive(Debug, Clone)]
pub struct GroupSummaryStage {
    pub prompt: CompressionPrompt,
    pub max_tokens: usize,
}

impl Default for GroupSummary {
    fn default() -> Self {
        Self {
            categorize: GroupSummaryStage {
                prompt: CompressionPrompt::placeholder("__categorize__"),
                max_tokens: 0,
            },
            assign: GroupSummaryStage {
                prompt: CompressionPrompt::placeholder("__assign__"),
                max_tokens: 0,
            },
            chunk: 0,
        }
    }
}

/// How a layer's turns are scoped into gather (provenance) trees.
///
/// - [`GatherScope::Shared`] (default): one tree per layer, shared across all
///   conversations — institutional memory / repo map / motivation.
/// - [`GatherScope::Conversation`]: one tree per conversation (keyed by its
///   `TimelineId`) — the live dialogue layer, private to each conversation.
///
/// Section groups are always per-collection regardless of this flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GatherScope {
    /// One shared tree across all conversations.
    #[default]
    Shared,
    /// One private tree per conversation.
    Conversation,
}

/// Schema for one cognitive layer.
///
/// A layer aggregates multiple [`GroupSchema`]s. Turn scores aggregate
/// into a per-group score via the fixed [`super::project::FIXED_FORMULA`],
/// used both for the layer-level threshold gate and for emission ordering
/// within the layer.
///
/// # Per-target turn budget
///
/// `window` declares the **total turn-budget** that flexbox distributes
/// across all visible layers **when this layer is the projection target**.
/// Different targets get different budgets — projecting for a thin "lower"
/// layer at 6K behaves identically to projecting for a 16K dialogue layer,
/// just with a smaller pie to slice. The flex priorities on layers and
/// groups still control the relative cuts; only the absolute size of the
/// pie changes.
#[derive(Debug, Clone)]
pub struct LayerSchema {
    /// Crate-assigned id, used by [`super::ProjectionTarget::layer`].
    pub id: LayerId,
    /// Original YAML `name:` string.
    pub name: String,
    /// Free-form description. Not used by the engine.
    pub description: String,
    /// Groups whose derived score is below this threshold are filtered from
    /// the layer before reconciliation. Default `0.0` (no gate).
    pub score_threshold: f32,
    /// Total turn-budget (in tokens) distributed across all visible layers
    /// when this layer is the projection target.
    pub window: usize,
    /// Flex weight when *some other layer* is the projection target and
    /// this layer is visible (lower than the target). Determines how much
    /// of the target's `window` this layer receives.
    pub budget: Budget,
    /// System-prompt sections framing the cognitive activity for which
    /// this layer is the projection target. **Required** at construction
    /// — every layer must declare at least one section so the layer is
    /// always usable as a projection target.
    pub system_prompt: SystemPromptSchema,
    /// How this layer's turns are compressed across the summary tree — the
    /// `turns`/`summaries` tree-levels, each splitting the question and answer
    /// halves into their own framing + instruction, with a per-level decode cap.
    pub summary: LayerSummary,
    /// Groups in declaration order. At projection time they are sorted by
    /// derived group score for emission.
    pub groups: Vec<GroupSchema>,
    /// Selection policy (belief-update + budget + tag scope) for this layer's
    /// turn groups, resolved from the layer's `policy:` or inherited from the
    /// schema default. Collections and groups may override it.
    pub policy: SelectionPolicy,
    /// Gather-tree scoping for this layer's turns (`gather_scope:` in YAML).
    /// Default [`GatherScope::Shared`].
    pub gather_scope: GatherScope,
}

impl LayerSchema {
    /// Every `SectionId` (as raw `u32`) this layer allocates: the system-prompt
    /// sections and collection member sections (via
    /// [`SystemPromptSchema::all_sections`]), every collection's compression
    /// summary, and the layer's own `turns`/`summaries` × `question`/`answer`
    /// compression prompts.
    ///
    /// Runtime section-id allocation ([`super::Builder::add_section_to_collection`])
    /// must stay disjoint from all of these. The compression-prompt sections
    /// never appear in `system_prompt.items`, so a max over only the *visible*
    /// sections would alias them — a runtime-added section would reuse a
    /// compression-prompt id, and `ensure_summary_section` would then inject that
    /// section's content (e.g. a tool's JSON) as the compression prompt.
    pub fn all_section_ids(&self) -> Vec<u32> {
        // `system_prompt.all_section_ids()` (not `all_sections()`) so section-tree
        // variant ids are counted too — `all_sections()` skips them.
        let mut ids: Vec<u32> = self
            .system_prompt
            .all_section_ids()
            .map(|id| id.raw())
            .collect();
        let push_summary = |c: &SectionCollection, ids: &mut Vec<u32>| {
            ids.push(c.summary.categorize.prompt.system_prompt.id.raw());
            ids.push(c.summary.assign.prompt.system_prompt.id.raw());
        };
        for item in &self.system_prompt.items {
            match item {
                SystemPromptItem::Collection(c) => push_summary(c, &mut ids),
                // Tree-embedded collections carry their own summary prompts.
                SystemPromptItem::SectionTree(t) => {
                    for n in &t.nodes {
                        if let Some(tc) = &n.collection {
                            push_summary(&tc.collection, &mut ids);
                        }
                    }
                }
                SystemPromptItem::Section(_) => {}
            }
        }
        self.summary.push_section_ids(&mut ids);
        ids
    }
}

/// Schema for one group within a layer.
#[derive(Debug, Clone)]
pub struct GroupSchema {
    /// Crate-assigned id, **globally unique** across all layers in the schema.
    pub id: GroupId,
    /// Original YAML `id:` string.
    pub name: String,
    /// Which turns survive into the projection.
    pub selection: SelectionRule,
    /// Turns whose score is below this threshold are invisible to selection.
    /// Default `0.0` (no gate).
    pub score_threshold: f32,
    /// Selection policy (belief-update + budget + tag scope) for this group,
    /// resolved from the group's `policy:` or inherited from the enclosing layer.
    pub policy: SelectionPolicy,
    pub budget: Budget,
    /// Fallback turn (by decl tag) brought in when belief/scores select nothing,
    /// so the group — and its layer — never drops out of the projection. `None`
    /// = no fallback (today's behaviour).
    pub default: Option<SelectionDefault>,
}

impl GroupSchema {
    /// Whether this group's turns are selected by the provenance belief
    /// mechanism (RelLeak scores + budget) rather than pure recency.
    /// `Sequence` is intrinsically a recency rule (recent-N inviolate); every
    /// other rule ranks by score and so becomes belief-driven once turns carry
    /// a fresh wide-Q score.
    pub fn is_belief_driven(&self) -> bool {
        !matches!(self.selection, SelectionRule::Sequence { .. })
    }

    /// The belief [`PolicyConfig`] for a belief-driven group: the group's policy
    /// (β leak rate, inherited) with the **budget** and **score gates** taken
    /// from the selection *rule* and `score_threshold`, so the belief step
    /// selects exactly the rule's cap — `top_k(k)` → at most `k`, `single` → 1,
    /// `always_visible` → all `n_candidates`. This keeps the belief path's
    /// surviving set identical to what `apply_selection` would rank, just
    /// carried across reprojections. `n_candidates` bounds the unbounded
    /// `always_visible` case.
    pub fn belief_config(&self, n_candidates: usize) -> PolicyConfig {
        let mut cfg = self.policy.config;
        cfg.budget_min = 0;
        cfg.budget_max = match &self.selection {
            SelectionRule::TopK { k } => *k,
            SelectionRule::Single => 1,
            SelectionRule::AlwaysVisible => n_candidates.max(1),
            // Named/Sequence aren't belief-driven; fall back to the policy budget.
            SelectionRule::Named { .. } | SelectionRule::Sequence { .. } => {
                self.policy.config.budget_max
            }
        };
        cfg.min_score = self.score_threshold;
        cfg.evict_score = self.score_threshold;
        cfg
    }
}

/// Flexbox-style token budget descriptor.
///
/// All percentages are **of the parent's resolved token budget** at
/// projection time, not of the global window. A min_percent of 30 on a group
/// inside a layer that received 1000 tokens means "at least 300 tokens for
/// this group."
///
/// # Reconciliation rules
///
/// - **`priority`** — relative weight when distributing the parent's budget.
///   Higher = larger share of remainder. Must be > 0.
/// - **`min_percent`** — floor. The flexbox distributor reserves at least
///   this fraction (subject to dynamic-shortfall proportional shrink if
///   sibling mins exceed 100%).
/// - **`max_percent`** — ceiling. Once a node hits its max, it's saturated
///   and excess budget redistributes to unsaturated siblings.
#[derive(Debug, Clone)]
pub struct Budget {
    /// Relative weight for proportional allocation. Must be > 0.
    pub priority: f32,
    /// Floor as a percent of parent budget (0–100).
    pub min_percent: Option<f32>,
    /// Ceiling as a percent of parent budget (0–100).
    pub max_percent: Option<f32>,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            priority: 50.0,
            min_percent: None,
            max_percent: None,
        }
    }
}

/// Which turns from a group survive into the projection.
///
/// # Decision tree
///
/// ```text
///   Selection rule ─┬─ AlwaysVisible        → all turns above threshold
///                   ├─ TopK { k }           → k highest-scored above threshold
///                   ├─ Single               → 1 highest-scored above threshold
///                   ├─ Named { selector }   → the one member named by a runtime selector
///                   └─ Sequence         → recent-N (inviolate) + top-K historical
/// ```
///
/// All rules **emit in insertion order** regardless of selection order
/// (selection picks by relevance; emission preserves dialogue coherence).
///
/// See [`super::selection`] for the implementation.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionRule {
    /// Every turn in the group survives selection (subject to score threshold
    /// and budget). Used for sections where ordering is structural, not
    /// salience-driven.
    AlwaysVisible,

    /// The `k` highest-scored turns survive. Ties broken by lower
    /// `TurnIndex` (earlier wins).
    TopK { k: usize },

    /// The single highest-scored turn survives. Used for groups where only
    /// one entry is ever relevant at a time (a single goal pressure, a
    /// single active threat).
    Single,

    /// Collection-only: the single member whose `name` equals the runtime
    /// value of the named selector (resolved from the projection's
    /// `SelectionState`, e.g. set per turn via `TurnOptions::selection`).
    /// This is an **explicit, score-independent** pick — it ignores provenance
    /// relevance and the score threshold entirely, selecting exactly the
    /// member the caller names (or nothing, if the selector is unset or
    /// names no member). Used to force one section out of a catalog by name —
    /// e.g. calibration pinning a single tool from the `tools` collection.
    ///
    /// On a turn group (which has no member names) this selects nothing.
    Named { selector: String },

    /// Composite for the natural shape of an ongoing conversation: the most
    /// recent `recent` turns survive **unconditionally** (no score threshold,
    /// no budget eviction), plus the top `historical_top_k` from the rest of
    /// the group by score.
    ///
    /// ```text
    ///   group.turns = [t0  t1  t2  t3  t4  t5  t6  t7]   (insertion order)
    ///                  ──────────────  ────────────────
    ///                   historical          recent
    ///                  (top-K by score)   (inviolate, last `recent`)
    /// ```
    Sequence {
        recent: usize,
        historical_top_k: usize,
    },
}

/// A fallback member injected when a group's or collection's normal selection
/// (belief + challenger + rule) yields nothing. Identified by a single string:
/// a turn's gather-scope decl tag for groups (e.g. `"."` for the repo_map
/// workspace-root cluster), or a section name for collections. Guarantees the
/// group/collection contributes at least one member so its layer never vanishes
/// from the projection when scores are cold.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionDefault {
    /// The turn-decl tag (groups) or section name (collections) that identifies
    /// the fallback member. Must uniquely name one member.
    pub tag: String,
}

/// How turn scores are aggregated into a single group score.
///
/// The group score is used for:
/// 1. The **layer-level threshold gate** — groups whose derived score falls
///    below the layer's threshold are dropped entirely.
/// 2. **Emission ordering** — within a layer, groups are sorted by ascending
///    score so higher-scored groups appear LAST (closer to the model's
///    recency bias).
///
/// | Variant         | Behaviour                                              |
/// |-----------------|--------------------------------------------------------|
/// | `Max`           | Maximum turn score. Default. Robust to noise from low-scoring tail content. |
/// | `Sum`           | Sum of all turn scores. Larger groups dominate.        |
/// | `Mean`          | Arithmetic mean. Penalises noisy groups.               |
/// | `TopKMean { k }` | Mean of the top-`k` scores. Smoothed peak.            |
/// | `Count`         | Number of eligible turns. Score-independent salience.  |
/// | `Span { alpha }` | Σ L^α over consecutive runs of above-threshold probe positions. Rewards sustained relevance. |
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScoreFormula {
    Max,
    Sum,
    Mean,
    TopKMean {
        k: usize,
    },
    Count,
    /// Power-law span scoring: consecutive runs of probe tokens that each find
    /// an above-threshold corpus match score L^α (default α=2.0).  Isolated
    /// hits score 1.0; a run of 3 scores 9.0.  The group-level aggregate
    /// (turn scores → group score) uses Max of per-turn span scores.
    Span {
        alpha: f32,
    },
    /// Per-token excess: Σ over probe tokens of `max(0, best_agreement − 64)`.
    /// Recentered on the random XOR-popcount baseline and reduced per probe
    /// token, with no hit threshold.  Calibrated as the strongest
    /// **prefill-phase** section-scoring formula — it recovers the weak,
    /// sub-threshold signal that `Span` (run-based) and `Max` (extreme-value)
    /// miss when the model is *reading* a query rather than generating.
    PerTokenExcess,
}

/// The complete parsed, validated schema. Immutable after construction.
///
/// All structural state lives on individual layers — there is no top-level
/// system prompt and no top-level token budget. Each layer carries its own
/// `window` (per-target turn budget) and `system_prompt` (framing for when
/// it is the target).
///
/// The schema does **not** carry dialect-specific structural tokens
/// (turn-boundary markers, role openers/closers).  Those are runtime
/// concerns owned by the scheduler / projection assembler — see
/// `crate::scheduler::projection_assembler::BoundaryMarkers`.
#[derive(Debug, Clone)]
pub struct Schema {
    /// Ordered: layer 0 first. Order is meaningful for masking and emission.
    pub layers: Vec<LayerSchema>,
}
