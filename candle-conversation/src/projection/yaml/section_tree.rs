//! Expands a `section_tree` into every combination of its selectors and toggles.
//!
//! A node with several options is a dimension. A breadth-first walk multiplies the
//! live branches by each option, sealing one variant per option and branch. A
//! toggle is an optional group that splits each branch into a present side and an
//! absent side. A dimension declared inside a group pads the other side with its
//! default, so every branch carries the same assignment width. A live-prefilled
//! marker takes no variant and does not extend the prefix.

use super::*;

/// Mixed-radix pack of `selection[0..radices.len()]` — the variant key for a
/// node whose ancestor dims have the given option counts.
pub(super) fn pack_assignment(selection: &[u8], radices: &[u8]) -> u32 {
    let mut key = 0u32;
    let mut mult = 1u32;
    for (i, &r) in radices.iter().enumerate() {
        key += selection[i] as u32 * mult;
        mult *= r as u32;
    }
    key
}

/// Append `value` to every suspended branch's assignment.  Called when a dim is
/// declared inside an open `optional_group`: the suspended (other-side) branches
/// never multiplied by that dim, so they must still carry a slot for it to keep
/// every branch's assignment the same width as `radices`.  The slot is the dim's
/// DEFAULT, and the dim is recorded as gated so [`super::schema::SectionTree::pack`]
/// masks the runtime value back to that same default when the group is on the
/// other side — landing the projection on exactly this sealed variant.
pub(super) fn pad_suspended(suspended: &mut [Vec<(Vec<u8>, Vec<SectionId>)>], value: u8) {
    for branch_set in suspended.iter_mut() {
        for (asm, _) in branch_set.iter_mut() {
            asm.push(value);
        }
    }
}

/// One resolved tree node in the flat spec stream the cross-product walk
/// consumes.  Nested `section_tree` nodes are already expanded; a `Collection`
/// node borrows its YAML so Pass 2 can allocate per-branch member ids.
pub(super) enum TreeSpec<'a> {
    /// A selector / optional / mandatory section node.
    Node {
        name: String,
        options: Vec<(String, String)>,
        default_option: u8,
        /// Placeholder node: its projection is replaced by this collection's
        /// top-k (resolved to a `CollectionId` in Pass 2).
        inject_collection: Option<&'a str>,
        /// `true` for a `dialect:` mandatory section (a structural marker like
        /// `<tools>`): it becomes live-prefilled glue — no sealed variants,
        /// prefix-transparent, emitted as `Generated` (see [`TreeNode::glue`]).
        is_glue: bool,
        /// The least the node's slot takes ([`TreeNode::min_tokens`]).
        min_tokens: usize,
    },
    /// A prefix-transparent embedded collection.
    Collection {
        name: &'a str,
        selection: &'a YamlSelection,
        score_threshold: f32,
        sections: &'a [YamlSection],
        member_glue: Option<&'a str>,
    },
    /// Opens a togglable sub-tree ([`YamlTreeNode::OptionalGroup`]) — a binary
    /// dim whose present branch (option 0) carries the specs up to the matching
    /// [`Self::GroupElse`] (or [`Self::GroupEnd`]), and whose absent branch
    /// (option 1) carries the specs from `GroupElse` to `GroupEnd` (none if there
    /// is no `GroupElse`).
    GroupStart {
        name: String,
        /// `0` = present (default), `1` = absent.
        default_option: u8,
    },
    /// Switches sealing from the present branch to the absent branch of the
    /// innermost open group — the specs after it seal on the `absent` sub-tree.
    GroupElse,
    /// Closes the innermost open [`Self::GroupStart`].
    GroupEnd,
}

/// Resolve a tree node/option's content from an explicit `content:` or a
/// dialect template; either way the section is SEALED (never a live template).
pub(super) fn resolve_tree_content(
    item: &str,
    content: &str,
    dlct_name: &Option<String>,
    dialect: Option<&Dialect>,
) -> Result<String, ConstructionError> {
    match dlct_name {
        Some(dname) => {
            let dlct = dialect.ok_or_else(|| ConstructionError::DialectRequired {
                item: item.to_string(),
            })?;
            let template = DialectTemplate::from_yaml_name(dname).ok_or_else(|| {
                ConstructionError::UnknownDialectTemplate {
                    item: item.to_string(),
                    name: dname.clone(),
                }
            })?;
            Ok(dlct.template(template).to_string())
        }
        None => Ok(content.to_string()),
    }
}

/// Flatten a tree-node list into a flat [`TreeSpec`] stream, expanding nested
/// `section_tree` nodes inline.  A nested tree's nodes are appended in place —
/// so its selectors become dims declared *after* everything above the nested
/// node, and each inner node seals against the full outer prefix (inheriting
/// every outer selector).  Recursion is bounded by the YAML's nesting depth.
pub(super) fn flatten_tree_specs<'a>(
    layer_name: &str,
    nodes: &'a [YamlTreeNode],
    dialect: Option<&Dialect>,
    layer_section_names: &mut std::collections::HashSet<String>,
    specs: &mut Vec<TreeSpec<'a>>,
) -> Result<(), ConstructionError> {
    for n in nodes {
        let (id, options, default_option) = match n {
            YamlTreeNode::Section {
                id,
                pin: _,
                content,
                dialect: d,
                inject_collection: _,
                min_tokens: _,
            } => {
                let item = format!("{layer_name}/{id}");
                (
                    id.clone(),
                    vec![(
                        "content".to_string(),
                        resolve_tree_content(&item, content, d, dialect)?,
                    )],
                    0u8,
                )
            }
            YamlTreeNode::Optional {
                id,
                pin: _,
                content,
                dialect: d,
                default,
                min_tokens: _,
            } => {
                let item = format!("{layer_name}/{id}");
                // `present` is option 0, `absent` option 1 — an unspecified
                // `default:` falls back to present.
                let default_option = match default.as_deref() {
                    None => 0u8,
                    Some(s) => match OptionalState::from_id(s) {
                        Some(OptionalState::Present) => 0u8,
                        Some(OptionalState::Absent) => 1u8,
                        None => {
                            return Err(ConstructionError::InvalidToggleDefault {
                                item,
                                value: s.to_string(),
                            })
                        }
                    },
                };
                (
                    id.clone(),
                    vec![
                        (
                            OptionalState::Present.as_id().to_string(),
                            resolve_tree_content(&item, content, d, dialect)?,
                        ),
                        (OptionalState::Absent.as_id().to_string(), String::new()),
                    ],
                    default_option,
                )
            }
            YamlTreeNode::Selector {
                id,
                pin: _,
                default,
                options,
                min_tokens: _,
            } => {
                let item = format!("{layer_name}/{id}");
                let mut opts = Vec::with_capacity(options.len());
                for o in options {
                    opts.push((
                        o.id.clone(),
                        resolve_tree_content(
                            &format!("{item}/{}", o.id),
                            &o.content,
                            &o.dialect,
                            dialect,
                        )?,
                    ));
                }
                let default_option =
                    opts.iter()
                        .position(|(oid, _)| oid == default)
                        .ok_or_else(|| ConstructionError::UnknownTreeOption {
                            item,
                            option: default.clone(),
                        })? as u8;
                (id.clone(), opts, default_option)
            }
            YamlTreeNode::SectionTree { nodes } => {
                flatten_tree_specs(layer_name, nodes, dialect, layer_section_names, specs)?;
                continue;
            }
            YamlTreeNode::Collection {
                name,
                pin: _,
                selection,
                score_threshold,
                sections,
                member_glue,
            } => {
                specs.push(TreeSpec::Collection {
                    name,
                    selection,
                    score_threshold: *score_threshold,
                    sections,
                    member_glue: member_glue.as_deref(),
                });
                continue;
            }
            YamlTreeNode::OptionalGroup {
                id,
                default,
                nodes,
                absent,
            } => {
                let item = format!("{layer_name}/{id}");
                // present = option 0, absent = option 1 (mirrors `optional`).
                let default_option = match default.as_deref() {
                    None => 0u8,
                    Some(s) => match OptionalState::from_id(s) {
                        Some(OptionalState::Present) => 0u8,
                        Some(OptionalState::Absent) => 1u8,
                        None => {
                            return Err(ConstructionError::InvalidToggleDefault {
                                item,
                                value: s.to_string(),
                            })
                        }
                    },
                };
                if !layer_section_names.insert(id.clone()) {
                    return Err(ConstructionError::DuplicateSectionName(id.clone()));
                }
                specs.push(TreeSpec::GroupStart {
                    name: id.clone(),
                    default_option,
                });
                flatten_tree_specs(layer_name, nodes, dialect, layer_section_names, specs)?;
                if !absent.is_empty() {
                    specs.push(TreeSpec::GroupElse);
                    flatten_tree_specs(layer_name, absent, dialect, layer_section_names, specs)?;
                }
                specs.push(TreeSpec::GroupEnd);
                continue;
            }
        };
        if !layer_section_names.insert(id.clone()) {
            return Err(ConstructionError::DuplicateSectionName(id));
        }
        let inject_collection = match n {
            YamlTreeNode::Section {
                inject_collection, ..
            } => inject_collection.as_deref(),
            _ => None,
        };
        let min_tokens = match n {
            YamlTreeNode::Section { min_tokens, .. }
            | YamlTreeNode::Optional { min_tokens, .. }
            | YamlTreeNode::Selector { min_tokens, .. } => *min_tokens,
            _ => 0,
        };
        // A `dialect:` mandatory section is a structural marker → live glue.
        // (An `optional`/`selector` with a dialect is a real dimension, not glue.)
        let is_glue = matches!(
            n,
            YamlTreeNode::Section {
                dialect: Some(_),
                ..
            }
        );
        specs.push(TreeSpec::Node {
            name: id,
            options,
            default_option,
            inject_collection,
            is_glue,
            min_tokens,
        });
    }
    Ok(())
}

/// Build one `kind: section_tree` item — the full cross-product fan-out.
///
/// Each node carries a list of options; nodes with >1 option are SELECTOR
/// dimensions.  A breadth-first walk maintains the live branch set —
/// `(option-assignment-of-dims-so-far, in-tree prefix of emitted variant ids)`.
/// At each node it seals one [`TreeVariant`] per (non-empty option × current
/// branch); then a mandatory node appends its single option's variant to every
/// branch, while a selector multiplies the branch set by its option count (each
/// non-empty option's variant joining that sub-branch's prefix).  The node's
/// declared name maps to its default-selection variant id.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_section_tree<'a>(
    layer_name: &str,
    lid: LayerId,
    nodes: &'a [YamlTreeNode],
    dialect: Option<&Dialect>,
    section_alloc: &mut SectionIdAlloc,
    maps: &mut NameMaps,
    layer_section_names: &mut std::collections::HashSet<String>,
    layer_collections: &std::collections::HashMap<String, CollectionId>,
) -> Result<SystemPromptItem, ConstructionError> {
    // ── Pass 1: resolve each node into a flat spec stream, expanding nested
    // `section_tree` nodes inline (their selectors inherit the outer dims). ──
    let mut specs: Vec<TreeSpec<'a>> = Vec::with_capacity(nodes.len());
    flatten_tree_specs(layer_name, nodes, dialect, layer_section_names, &mut specs)?;

    // ── Pass 2: cross-product fan-out ────────────────────────────────────────
    // A branch = (option assignment over the dims seen so far, in-tree prefix).
    let mut branches: Vec<(Vec<u8>, Vec<SectionId>)> = vec![(Vec::new(), Vec::new())];
    let mut radices: Vec<u8> = Vec::new(); // option_count of each dim, in order
    let mut dims: Vec<TreeDim> = Vec::new();
    let mut default_selection: Vec<u8> = Vec::new();
    let mut tree_nodes: Vec<TreeNode> = Vec::with_capacity(specs.len());
    // Open `optional_group` scopes: each holds the ABSENT sub-branches set aside
    // while the group's children seal/extend only the PRESENT sub-branches. Popped
    // and merged back at the matching `GroupEnd`.
    let mut suspended: Vec<Vec<(Vec<u8>, Vec<SectionId>)>> = Vec::new();
    // Parallel to `suspended`: the (dim-index, active-side) of each open group, so
    // a dim declared inside it is recorded as gated by that group's current side
    // (`0` present / `1` absent — flipped at `GroupElse`).
    let mut group_ctx: Vec<(usize, u8)> = Vec::new();

    for spec in &specs {
        let ancestor_dims = radices.len();
        match spec {
            TreeSpec::Node {
                name,
                options: spec_options,
                default_option,
                inject_collection,
                is_glue,
                min_tokens,
            } => {
                let min_tokens = *min_tokens;
                // A live-prefilled structural marker (`<tools>` etc.): allocate NO
                // sealed variant, do NOT extend the prefix (nodes below seal as if
                // it weren't here), and record the branches it lives in so it emits
                // its `Generated` run only there.  Its text stays in `options[0]`
                // for tokenisation + the GUI glue label.
                if *is_glue {
                    let active_keys: Vec<u32> = branches
                        .iter()
                        .map(|(asm, _)| pack_assignment(asm, &radices))
                        .collect();
                    tree_nodes.push(TreeNode {
                        name: name.clone(),
                        options: vec![TreeOption {
                            id: spec_options[0].0.clone(),
                            content: spec_options[0].1.clone(),
                            variants: Vec::new(),
                        }],
                        collection: None,
                        inject_collection: None,
                        glue: Some(super::super::schema::TreeGlue {
                            tokens: None,
                            active_keys,
                        }),
                        dim: None,
                        ancestor_dims,
                        min_tokens,
                    });
                    continue;
                }
                let is_dim = spec_options.len() > 1;
                // Resolve the placeholder's target collection name → id (if any).
                let inject_cid = match inject_collection {
                    Some(cname) => Some(*layer_collections.get(*cname).ok_or_else(|| {
                        ConstructionError::UnknownCollection(format!(
                            "{layer_name}: node '{name}' inject_collection references unknown \
                             collection '{cname}'",
                        ))
                    })?),
                    None => None,
                };

                // Seal one variant per (non-empty option × current branch).
                let mut options: Vec<TreeOption> = Vec::with_capacity(spec_options.len());
                for (oid, content) in spec_options {
                    let mut variants = Vec::new();
                    if !content.is_empty() {
                        for (asm, prefix) in &branches {
                            variants.push(TreeVariant {
                                ancestors: pack_assignment(asm, &radices),
                                id: section_alloc.next(),
                                in_tree_prefix: prefix.clone(),
                            });
                        }
                    }
                    options.push(TreeOption {
                        id: oid.clone(),
                        content: content.clone(),
                        variants,
                    });
                }

                // The declared name resolves to the default selection's variant.
                let default_opt = &options[*default_option as usize];
                if let Some(v) =
                    default_opt.variant_for(pack_assignment(&default_selection, &radices))
                {
                    maps.section_names.insert((lid, name.clone()), v.id);
                }

                // Advance the live branch set + dims.
                if is_dim {
                    let dim_index = dims.len();
                    dims.push(TreeDim {
                        selector_id: name.clone(),
                        node_index: tree_nodes.len(),
                        option_count: spec_options.len() as u8,
                        default_option: *default_option,
                        // Gated by the innermost open group's current side (if any).
                        gate: group_ctx.last().copied(),
                    });
                    let mut next = Vec::with_capacity(branches.len() * spec_options.len());
                    for (asm, prefix) in &branches {
                        let key = pack_assignment(asm, &radices);
                        for (opt_i, option) in options.iter().enumerate() {
                            let mut sub_asm = asm.clone();
                            sub_asm.push(opt_i as u8);
                            let mut sub_prefix = prefix.clone();
                            if let Some(v) = option.variant_for(key) {
                                sub_prefix.push(v.id); // non-empty option joins the prefix
                            }
                            next.push((sub_asm, sub_prefix));
                        }
                    }
                    branches = next;
                    radices.push(spec_options.len() as u8);
                    default_selection.push(*default_option);
                    // Inside a group, the suspended (other-side) branches never saw
                    // this selector — pad them with its default to keep widths even;
                    // the gate above makes `pack` mask to that same default.
                    pad_suspended(&mut suspended, *default_option);
                    tree_nodes.push(TreeNode {
                        name: name.clone(),
                        options,
                        collection: None,
                        inject_collection: inject_cid,
                        glue: None,
                        dim: Some(dim_index),
                        ancestor_dims,
                        min_tokens,
                    });
                } else {
                    // Mandatory: append its single option's variant to every branch.
                    for (asm, prefix) in branches.iter_mut() {
                        let key = pack_assignment(asm, &radices);
                        if let Some(v) = options[0].variant_for(key) {
                            prefix.push(v.id);
                        }
                    }
                    tree_nodes.push(TreeNode {
                        name: name.clone(),
                        options,
                        collection: None,
                        inject_collection: inject_cid,
                        glue: None,
                        dim: None,
                        ancestor_dims,
                        min_tokens,
                    });
                }
            }
            TreeSpec::Collection {
                name,
                selection,
                score_threshold,
                sections,
                member_glue,
            } => {
                // A prefix-transparent embedded collection: seal each member
                // ONCE PER ancestor branch (so members fan out ×outer-selectors),
                // but DO NOT touch `branches`/`radices` — nodes below seal as if
                // the variable members were not here (they anchor on the next
                // mandatory node instead).
                let label = format!("{layer_name}/{name}");
                let cid = *layer_collections.get(*name).ok_or_else(|| {
                    ConstructionError::UnknownCollection(format!(
                        "{layer_name}: embedded collection '{name}' has no pre-allocated id",
                    ))
                })?;
                let coll_selection = parse_collection_selection(&label, selection)?;
                if *score_threshold < 0.0 {
                    return Err(ConstructionError::NegativeScoreThreshold {
                        name: label.clone(),
                        value: *score_threshold,
                    });
                }

                let default_key = pack_assignment(&default_selection, &radices);
                let mut member_variants: Vec<Vec<TreeVariant>> = Vec::with_capacity(sections.len());
                let mut sec_schemas: Vec<SectionSchema> = Vec::with_capacity(sections.len());
                for s in sections.iter() {
                    if !layer_section_names.insert(s.id.clone()) {
                        return Err(ConstructionError::DuplicateSectionName(s.id.clone()));
                    }
                    let pri = s.priority.unwrap_or(50.0);
                    validate_priority(&format!("{layer_name}/{}", s.id), pri)?;
                    let mut variants = Vec::with_capacity(branches.len());
                    for (asm, prefix) in &branches {
                        variants.push(TreeVariant {
                            ancestors: pack_assignment(asm, &radices),
                            id: section_alloc.next(),
                            in_tree_prefix: prefix.clone(),
                        });
                    }
                    // The default-branch variant id is canonical: it drives
                    // `depends_on` gating, BDP scoring, and `id_for_section`.
                    let canonical = variants
                        .iter()
                        .find(|v| v.ancestors == default_key)
                        .map(|v| v.id)
                        .unwrap_or(variants[0].id);
                    maps.section_names.insert((lid, s.id.clone()), canonical);
                    sec_schemas.push(SectionSchema {
                        id: canonical,
                        name: s.id.clone(),
                        content: s.content.clone(),
                        priority: pri,
                        depends_on: None,
                        depends_on_absent: None,
                        depends_on_configured: None,
                        depends_on_unconfigured: None,
                        is_template: false,
                        min_tokens: 0,
                        template_tokens: None,
                    });
                    member_variants.push(variants);
                }
                let collection = SectionCollection {
                    id: cid,
                    name: name.to_string(),
                    sections: sec_schemas,
                    selection: coll_selection,
                    score_threshold: *score_threshold,
                    // Tree-node collections select by top-k provenance score
                    // (`section_score`), not the belief loop, so their policy is
                    // inert — a default satisfies the schema.
                    policy: SelectionPolicy::default_policy(),
                    summary_section: None,
                    member_glue: member_glue.map(str::to_string).unwrap_or_default(),
                    member_glue_tokens: None,
                    budget_adaptive: None,
                    default: None,
                    mandatory: Vec::new(),
                    // A prefix-transparent tree collection takes no positions
                    // of its own in the prompt below it.
                    range: None,
                    summary_range: None,
                    kv_level: MEMBER_KV_LEVEL,
                };
                // Capture the branch templates so runtime member additions
                // (the tool catalog) can seal ×branch without re-deriving them.
                let branch_templates: Vec<(u32, Vec<SectionId>)> = branches
                    .iter()
                    .map(|(asm, prefix)| (pack_assignment(asm, &radices), prefix.clone()))
                    .collect();
                tree_nodes.push(TreeNode {
                    name: name.to_string(),
                    options: Vec::new(),
                    collection: Some(TreeCollection {
                        collection,
                        variants: member_variants,
                        branches: branch_templates,
                        default_branch: default_key,
                        // Set below if a placeholder node injects this collection.
                        deferred_projection: false,
                    }),
                    inject_collection: None,
                    glue: None,
                    dim: None,
                    ancestor_dims,
                    min_tokens: 0,
                });
            }
            TreeSpec::GroupStart {
                name,
                default_option,
            } => {
                // A binary dim whose options carry NO content of their own (the
                // group's child nodes are what emit).  Split every live branch into
                // a present (option 0) sub-branch — which the children below will
                // seal/extend — and an absent (option 1) sub-branch, set aside until
                // `GroupEnd`.  A synthetic toggle node anchors the dim so the runtime
                // can drive presence by `name`; it emits nothing itself.
                let dim_index = dims.len();
                dims.push(TreeDim {
                    selector_id: name.clone(),
                    node_index: tree_nodes.len(),
                    option_count: 2,
                    default_option: *default_option,
                    // A nested group is itself gated by its ENCLOSING group's side.
                    gate: group_ctx.last().copied(),
                });
                tree_nodes.push(TreeNode {
                    name: name.clone(),
                    options: vec![
                        TreeOption {
                            id: OptionalState::Present.as_id().to_string(),
                            content: String::new(),
                            variants: Vec::new(),
                        },
                        TreeOption {
                            id: OptionalState::Absent.as_id().to_string(),
                            content: String::new(),
                            variants: Vec::new(),
                        },
                    ],
                    collection: None,
                    inject_collection: None,
                    glue: None,
                    dim: Some(dim_index),
                    ancestor_dims,
                    min_tokens: 0,
                });
                let mut present = Vec::with_capacity(branches.len());
                let mut absent = Vec::with_capacity(branches.len());
                for (asm, prefix) in &branches {
                    let mut p_asm = asm.clone();
                    p_asm.push(0); // present
                    present.push((p_asm, prefix.clone()));
                    let mut a_asm = asm.clone();
                    a_asm.push(1); // absent
                    absent.push((a_asm, prefix.clone()));
                }
                // An ENCLOSING group's suspended branches never saw this nested
                // group's dim — pad them with its default (this group's own `absent`
                // already carries the dim, so pad before pushing it).
                pad_suspended(&mut suspended, *default_option);
                suspended.push(absent);
                branches = present;
                radices.push(2);
                default_selection.push(*default_option);
                group_ctx.push((dim_index, 0)); // sealing the PRESENT side first
            }
            TreeSpec::GroupElse => {
                // Switch sealing to the ABSENT sub-branches: stash the extended
                // present branches and restore the (untouched) absent ones, so the
                // following specs seal on the absent branch.
                let absent = suspended
                    .pop()
                    .expect("GroupElse without a matching GroupStart");
                suspended.push(std::mem::take(&mut branches)); // extended present
                branches = absent;
                group_ctx
                    .last_mut()
                    .expect("GroupElse without GroupStart")
                    .1 = 1; // absent side
            }
            TreeSpec::GroupEnd => {
                // Fold the OTHER branch (the absent sub-branches when there was no
                // `GroupElse`, or the extended present sub-branches when there was)
                // back in.  Nodes below this point seal across BOTH (×present/absent).
                let other = suspended
                    .pop()
                    .expect("GroupEnd without a matching GroupStart");
                branches.extend(other);
                group_ctx.pop();
            }
        }
    }

    // A placeholder node (`inject_collection`) takes over its target collection's
    // projection, so mark that collection deferred — it must NOT also emit at its
    // own tree position.
    let injected: std::collections::HashSet<CollectionId> = tree_nodes
        .iter()
        .filter_map(|n| n.inject_collection)
        .collect();
    for node in tree_nodes.iter_mut() {
        if let Some(tc) = node.collection.as_mut() {
            if injected.contains(&tc.collection.id) {
                tc.deferred_projection = true;
            }
        }
    }

    let mut tree = SectionTree {
        nodes: tree_nodes,
        dims,
        default_selection,
        default_present_ids: Vec::new(),
    };
    // The default selection's emitted variant ids — what sections after the tree
    // (and priming) attend to.  Collection nodes are prefix-transparent and emit
    // a runtime-selected subset, so they contribute nothing here.
    tree.default_present_ids = tree
        .nodes
        .iter()
        .filter(|n| n.collection.is_none())
        .filter_map(|n| {
            let opt = &n.options[n.chosen(&tree.default_selection)];
            opt.variant_for(tree.pack(&tree.default_selection, n.ancestor_dims))
                .map(|v| v.id)
        })
        .collect();

    Ok(SystemPromptItem::SectionTree(tree))
}
