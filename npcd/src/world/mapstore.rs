//! Keeping a reshaped map in the authored YAML it came from (effector design
//! Appendix F.4).
//!
//! A [`World::reshape`](npc_map::World::reshape) changes the map in memory; this
//! is what makes the change outlast the process. The map is a folder of
//! one-file-per-area YAML under `<mind>/map/<world>/`, each file named for the
//! area it holds (`vault-casting.yaml` holds the area `vault-casting`), so
//! keeping it is a single-file change: the area a reshape touched
//! ([`MapEdit::area`]) is the file to rewrite, or — for a drowned area
//! ([`MapEdit::removes_area`]) — the file to remove.
//!
//! Where it is kept is a [`Shelf`]'s: the drafts of the account the reshape is
//! made for, over a mind under git — published with the rest of what they
//! wrote (`docs/replicated_substrate.md` §10.3) — or the folder itself.
//!
//! # It rewrites one file, not the whole map
//!
//! Every edit changes exactly one area, so only that area's file is touched. The
//! rest are left byte-for-byte as their authors wrote them — comments and all.
//! The one changed file *is* normalised: `serde_yaml` has no way to edit YAML in
//! place while preserving comments, so the touched area is re-serialised from the
//! in-memory structure. That is the honest cost of an authored topology edit, and
//! it falls only on the area actually edited. The derived fields (`exits`,
//! `visible`) are `#[serde(skip)]`, so what is kept is the authored form a fresh
//! load re-weaves — the round-trip is stable.

use anyhow::{Context, Result};
use npc_map::{MapEdit, MapSet};

use crate::sim::shelf::Shelf;

/// Where a world's reshapes are kept: the map's folder as `shelf` names it,
/// and whether what is kept there is a draft.
pub struct Keep {
    pub shelf: Box<dyn Shelf>,
    /// The map's folder on `shelf`; empty when the shelf is the folder.
    pub folder: String,
    pub drafts: bool,
}

impl Keep {
    /// Where area `id`'s file is on the shelf.
    fn path_of(&self, id: &str) -> String {
        match self.folder.is_empty() {
            true => format!("{id}.yaml"),
            false => format!("{}/{id}.yaml", self.folder.trim_end_matches('/')),
        }
    }
}

/// Keep the one area a reshape touched.
///
/// For a removal, the area's file is removed (an already-absent file is fine —
/// the end state is the same). For every other edit, the touched area is
/// re-serialised from `map`. `map` is the map *after* the reshape, so the area
/// read back here is the reshaped one.
pub fn persist(keep: &Keep, edit: &MapEdit, map: &MapSet) -> Result<()> {
    let area_id = edit.area();
    let path = keep.path_of(area_id);
    if edit.removes_area() {
        return keep
            .shelf
            .commit(&[(path, None)])
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("removing area `{area_id}`"));
    }
    let area = map.get(area_id).with_context(|| {
        format!("area `{area_id}` is gone after a reshape that was not a removal")
    })?;
    let yaml = serde_yaml::to_string(area).context("serialising the reshaped area")?;
    keep.shelf
        .commit(&[(path, Some(yaml))])
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("keeping area `{area_id}`"))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::drafts::disk::Disk;
    use npc_map::schema::{Area, AreaKind, Node, NodeKind};
    use npc_map::MapSet;

    fn node(id: &str, kind: NodeKind, off: &[&str]) -> Node {
        Node {
            id: id.into(),
            kind,
            name: id.into(),
            plural: false,
            stand: None,
            off: off.iter().map(|s| s.to_string()).collect(),
            character: None,
            parts: vec![],
            ground: vec![],
            habit: None,
            sees: vec![],
            exits: vec![],
            visible: vec![],
        }
    }

    fn area(id: &str, nodes: Vec<Node>) -> Area {
        Area {
            id: id.into(),
            kind: AreaKind::Level,
            name: id.into(),
            within: None,
            ordinal: None,
            summary: "a place".into(),
            character: None,
            lacks: vec![],
            announcements: vec![],
            contains: vec![],
            portals: vec![],
            arrival: None,
            teleport_to: None,
            spine: None,
            nodes,
        }
    }

    /// The map's own folder, written as it stands.
    fn on_disk(dir: &Path) -> Keep {
        Keep {
            shelf: Box::new(Disk::new(dir)),
            folder: String::new(),
            drafts: false,
        }
    }

    /// The base one-area set, and its authored file on disk.
    fn base() -> (tempfile::TempDir, MapSet) {
        let dir = tempfile::tempdir().unwrap();
        let set = MapSet::from_areas([area(
            "a",
            vec![
                node("core", NodeKind::Core, &["hall"]),
                node("hall", NodeKind::Passage, &[]),
            ],
        )])
        .unwrap();
        std::fs::write(
            dir.path().join("a.yaml"),
            serde_yaml::to_string(set.get("a").unwrap()).unwrap(),
        )
        .unwrap();
        (dir, set)
    }

    /// **A kept reshape reloads to the same map.** After keeping the area an
    /// `AddNode` touched, loading the folder afresh finds the new room with its
    /// door woven — the whole point of keeping it.
    #[test]
    fn a_kept_reshape_survives_a_reload() {
        let (dir, set) = base();
        let edit = MapEdit::AddNode {
            area: "a".into(),
            node: Box::new(node("green", NodeKind::Social, &["hall"])),
        };
        let reshaped = set.apply(&edit).unwrap();
        persist(&on_disk(dir.path()), &edit, &reshaped).expect("keeping succeeds");

        let reloaded = MapSet::load_dir(dir.path()).expect("the written map reloads");
        let hall = reloaded.get("a").unwrap().node("hall").unwrap();
        assert!(
            hall.exits.contains(&"green".to_string()),
            "the reloaded map did not have the new room woven in: {:?}",
            hall.exits
        );
    }

    /// **Drowning an area removes its file.**
    #[test]
    fn drowning_an_area_removes_its_file() {
        let dir = tempfile::tempdir().unwrap();
        // Two independent one-node areas, each its own file.
        let set = MapSet::from_areas([
            area("keep", vec![node("core", NodeKind::Core, &[])]),
            area("drown", vec![node("core", NodeKind::Core, &[])]),
        ])
        .unwrap();
        for id in ["keep", "drown"] {
            std::fs::write(
                dir.path().join(format!("{id}.yaml")),
                serde_yaml::to_string(set.get(id).unwrap()).unwrap(),
            )
            .unwrap();
        }
        let edit = MapEdit::RemoveArea("drown".into());
        let reshaped = set.apply(&edit).unwrap();
        persist(&on_disk(dir.path()), &edit, &reshaped).expect("removal succeeds");

        assert!(
            !dir.path().join("drown.yaml").exists(),
            "the file was not removed"
        );
        assert!(
            dir.path().join("keep.yaml").exists(),
            "the wrong file was removed"
        );
        let reloaded = MapSet::load_dir(dir.path()).unwrap();
        assert!(reloaded.get("drown").is_none());
        assert!(reloaded.get("keep").is_some());
    }

    /// **Only the touched file is rewritten.** A reshape of one area leaves every
    /// other authored file untouched, byte for byte — comments and all — and the
    /// map's folder on a shelf is where the file goes.
    #[test]
    fn an_untouched_area_file_is_left_alone() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("map/w");
        std::fs::create_dir_all(&dir).unwrap();
        let set = MapSet::from_areas([
            area(
                "a",
                vec![
                    node("core", NodeKind::Core, &["hall"]),
                    node("hall", NodeKind::Passage, &[]),
                ],
            ),
            area("b", vec![node("core", NodeKind::Core, &[])]),
        ])
        .unwrap();
        std::fs::write(
            dir.join("a.yaml"),
            serde_yaml::to_string(set.get("a").unwrap()).unwrap(),
        )
        .unwrap();
        // `b`'s file carries a comment a hand author left; it must survive.
        let authored_b = format!(
            "# the far level, do not touch\n{}",
            serde_yaml::to_string(set.get("b").unwrap()).unwrap()
        );
        std::fs::write(dir.join("b.yaml"), &authored_b).unwrap();

        let edit = MapEdit::AddNode {
            area: "a".into(),
            node: Box::new(node("green", NodeKind::Social, &["hall"])),
        };
        let keep = Keep {
            shelf: Box::new(Disk::new(root.path())),
            folder: "map/w".into(),
            drafts: false,
        };
        persist(&keep, &edit, &set.apply(&edit).unwrap()).unwrap();

        let after_b = std::fs::read_to_string(dir.join("b.yaml")).unwrap();
        assert_eq!(after_b, authored_b, "an untouched file was rewritten");
        assert_eq!(
            std::fs::read_to_string(dir.join("a.yaml")).unwrap(),
            serde_yaml::to_string(set.apply(&edit).unwrap().get("a").unwrap()).unwrap()
        );
    }
}
