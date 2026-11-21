//! Assigns the identifiers that schema items use, and maps their author-given
//! names back to them.
//!
//! Section and collection identifiers are allocated in sequence across the whole
//! schema, while names are scoped to their layer. Collections are registered
//! before the items that reference them, so a dependency on a collection declared
//! later still resolves. A duplicate name is an error.

use super::*;

/// Sequential SectionId allocator. Ids are globally unique across the whole
/// schema even though names are per-layer scoped.
pub(super) struct SectionIdAlloc(pub(super) u32);

impl SectionIdAlloc {
    pub(super) fn next(&mut self) -> SectionId {
        self.0 += 1;
        SectionId::new(self.0)
    }
}

/// Sequential CollectionId allocator.  Ids are globally unique across the
/// whole schema; collection names are layer-scoped (analogous to
/// sections, not groups).
pub(super) struct CollectionIdAlloc(pub(super) u32);

impl CollectionIdAlloc {
    pub(super) fn next(&mut self) -> CollectionId {
        self.0 += 1;
        CollectionId::new(self.0)
    }
}

// ── Name lookup maps produced alongside the Schema ───────────────────────────

/// Reverse maps from YAML string names to crate-assigned ids.
///
/// Built by [`from_yaml`] alongside the [`Schema`], stashed inside the
/// [`super::Builder`], and queried by `id_for_layer` / `id_for_group` /
/// `id_for_section_in`. Empty when the builder is constructed via
/// `from_schema` (no YAML names available).
///
/// Section names are **per-layer scoped** — the same section name may be
/// declared in multiple layers. The lookup map is keyed by `(LayerId,
/// name)`.
#[derive(Debug, Default, Clone)]
pub struct NameMaps {
    pub layer_names: HashMap<String, LayerId>,
    pub group_names: HashMap<String, GroupId>,
    pub section_names: HashMap<(LayerId, String), SectionId>,
    pub collection_names: HashMap<(LayerId, String), CollectionId>,
}

/// Pre-allocate one collection's [`CollectionId`] and register its name so
/// `depends_on` forward references resolve.  Errors on a duplicate name.
pub(super) fn register_collection_id(
    name: &str,
    lid: LayerId,
    layer_collection_names: &mut std::collections::HashSet<String>,
    layer_collections: &mut std::collections::HashMap<String, CollectionId>,
    maps: &mut NameMaps,
    collection_alloc: &mut CollectionIdAlloc,
) -> Result<(), ConstructionError> {
    if !layer_collection_names.insert(name.to_string()) {
        return Err(ConstructionError::DuplicateCollectionName(name.to_string()));
    }
    let cid = collection_alloc.next();
    layer_collections.insert(name.to_string(), cid);
    maps.collection_names.insert((lid, name.to_string()), cid);
    Ok(())
}

/// Recurse a section-tree node list, pre-allocating ids for every embedded
/// `Collection` node (including those inside nested `section_tree` nodes).
pub(super) fn register_tree_collection_ids(
    nodes: &[YamlTreeNode],
    lid: LayerId,
    layer_collection_names: &mut std::collections::HashSet<String>,
    layer_collections: &mut std::collections::HashMap<String, CollectionId>,
    maps: &mut NameMaps,
    collection_alloc: &mut CollectionIdAlloc,
) -> Result<(), ConstructionError> {
    for n in nodes {
        match n {
            YamlTreeNode::Collection { name, .. } => register_collection_id(
                name,
                lid,
                layer_collection_names,
                layer_collections,
                maps,
                collection_alloc,
            )?,
            YamlTreeNode::SectionTree { nodes } => register_tree_collection_ids(
                nodes,
                lid,
                layer_collection_names,
                layer_collections,
                maps,
                collection_alloc,
            )?,
            YamlTreeNode::OptionalGroup { nodes, absent, .. } => {
                for branch in [nodes, absent] {
                    register_tree_collection_ids(
                        branch,
                        lid,
                        layer_collection_names,
                        layer_collections,
                        maps,
                        collection_alloc,
                    )?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}
