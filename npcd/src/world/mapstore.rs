//! Writing a reshaped map back to the authored YAML it came from (effector
//! design Appendix F.4).
//!
//! A [`World::reshape`](npc_map::World::reshape) changes the map in memory; this
//! is what makes the change outlast the process. The map is a directory of
//! one-file-per-area YAML under `<mind>/map/<world>/`, each file named for the
//! area it holds (`vault-casting.yaml` holds the area `vault-casting`), so
//! writeback is a single-file operation: the area a reshape touched
//! ([`MapEdit::area`]) is the file to rewrite, or — for a drowned area
//! ([`MapEdit::removes_area`]) — the file to remove.
//!
//! # It rewrites one file, not the whole map
//!
//! Every edit changes exactly one area, so only that area's file is touched. The
//! rest are left byte-for-byte as their authors wrote them — comments and all.
//! The one changed file *is* normalised: `serde_yaml` has no way to edit YAML in
//! place while preserving comments, so the touched area is re-serialised from the
//! in-memory structure. That is the honest cost of an authored topology edit, and
//! it falls only on the area actually edited. The derived fields (`exits`,
//! `visible`) are `#[serde(skip)]`, so what lands on disk is the authored form a
//! fresh load re-weaves — the round-trip is stable.
//!
//! # The write is atomic
//!
//! A crash mid-write must not leave a half-written area file that fails the next
//! load and takes the world down with it. So the new contents go to a temporary
//! sibling and are `rename`d over the target — atomic on the same volume — and a
//! failure at any step leaves the original file exactly as it was.

use std::path::Path;

use anyhow::{Context, Result};
use npc_map::{MapEdit, MapSet};

/// Persist the one area a reshape touched.
///
/// For a removal, the area's file is deleted (an already-absent file is fine —
/// the end state is the same). For every other edit, the touched area is
/// re-serialised from `map` and written atomically. `map` is the map *after* the
/// reshape, so the area read back here is the reshaped one.
pub fn persist(dir: &Path, edit: &MapEdit, map: &MapSet) -> Result<()> {
    let area_id = edit.area();
    let path = dir.join(format!("{area_id}.yaml"));
    if edit.removes_area() {
        return match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            // Already gone is the state we wanted.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e).with_context(|| format!("removing {}", path.display())),
        };
    }
    let area = map.get(area_id).with_context(|| {
        format!("area `{area_id}` is gone after a reshape that was not a removal")
    })?;
    let yaml = serde_yaml::to_string(area).context("serialising the reshaped area")?;
    write_atomic(&path, &yaml)
}

/// Write `contents` to `path` atomically: a temporary sibling, then a rename.
fn write_atomic(path: &Path, contents: &str) -> Result<()> {
    let tmp = path.with_extension("yaml.tmp");
    std::fs::write(&tmp, contents).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("renaming {} into place", tmp.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn tmp() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "npcd-mapstore-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// The base one-area set, and its authored file on disk.
    fn base() -> (std::path::PathBuf, MapSet) {
        let dir = tmp();
        let set = MapSet::from_areas([area(
            "a",
            vec![
                node("core", NodeKind::Core, &["hall"]),
                node("hall", NodeKind::Passage, &[]),
            ],
        )])
        .unwrap();
        std::fs::write(
            dir.join("a.yaml"),
            serde_yaml::to_string(set.get("a").unwrap()).unwrap(),
        )
        .unwrap();
        (dir, set)
    }

    /// **A persisted reshape reloads to the same map.** After writing back the
    /// area an `AddNode` touched, loading the directory afresh finds the new room
    /// with its door woven — the whole point of writeback.
    #[test]
    fn a_persisted_reshape_survives_a_reload() {
        let (dir, set) = base();
        let edit = MapEdit::AddNode {
            area: "a".into(),
            node: Box::new(node("green", NodeKind::Social, &["hall"])),
        };
        let reshaped = set.apply(&edit).unwrap();
        persist(&dir, &edit, &reshaped).expect("writeback succeeds");

        let reloaded = MapSet::load_dir(&dir).expect("the written map reloads");
        let hall = reloaded.get("a").unwrap().node("hall").unwrap();
        assert!(
            hall.exits.contains(&"green".to_string()),
            "the reloaded map did not have the new room woven in: {:?}",
            hall.exits
        );
    }

    /// **Drowning an area removes its file.** After a `RemoveArea` writeback, the
    /// file is gone, so a reload no longer holds it.
    #[test]
    fn drowning_an_area_removes_its_file() {
        let dir = tmp();
        // Two independent one-node areas, each its own file.
        let set = MapSet::from_areas([
            area("keep", vec![node("core", NodeKind::Core, &[])]),
            area("drown", vec![node("core", NodeKind::Core, &[])]),
        ])
        .unwrap();
        for id in ["keep", "drown"] {
            std::fs::write(
                dir.join(format!("{id}.yaml")),
                serde_yaml::to_string(set.get(id).unwrap()).unwrap(),
            )
            .unwrap();
        }
        let edit = MapEdit::RemoveArea("drown".into());
        let reshaped = set.apply(&edit).unwrap();
        persist(&dir, &edit, &reshaped).expect("removal succeeds");

        assert!(!dir.join("drown.yaml").exists(), "the file was not removed");
        assert!(dir.join("keep.yaml").exists(), "the wrong file was removed");
        let reloaded = MapSet::load_dir(&dir).unwrap();
        assert!(reloaded.get("drown").is_none());
        assert!(reloaded.get("keep").is_some());
    }

    /// **Only the touched file is rewritten.** A reshape of one area leaves every
    /// other authored file untouched, byte for byte — comments and all.
    #[test]
    fn an_untouched_area_file_is_left_alone() {
        let dir = tmp();
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
        persist(&dir, &edit, &set.apply(&edit).unwrap()).unwrap();

        let after_b = std::fs::read_to_string(dir.join("b.yaml")).unwrap();
        assert_eq!(after_b, authored_b, "an untouched file was rewritten");
        // And no stray temp file was left behind.
        assert!(!dir.join("a.yaml.tmp").exists(), "a temp file leaked");
    }
}
