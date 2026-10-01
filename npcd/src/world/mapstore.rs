//! Writing a reshaped map back to the authored files.
//!
//! A map is a directory of `<area>.yaml` files, one [`Area`] each. After a
//! runtime edit has landed in memory ([`npc_map::world::World::reshape`]), this
//! brings the directory into agreement with the new map by rewriting **only the
//! files whose area changed**: every other authored file is left byte for byte,
//! comments and all. The derived `exits`/`visible` are `#[serde(skip)]`, so what
//! is written is the authored form a fresh load re-weaves.
//!
//! An area is matched to its file by the `id` inside the file, not by the file's
//! name — a map is free to keep `creators-vault.yaml` holding the area `vault` —
//! and a new area gets `<id>.yaml`. Every write goes to a sibling temporary file
//! that is then renamed over the target, so a crash mid-write leaves the old
//! file, never half of a new one.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use npc_map::schema::Area;
use npc_map::{MapEdit, MapSet};

/// The authored `.yaml` files directly in `dir`, with the area id each holds.
fn files(dir: &Path) -> Result<Vec<(PathBuf, Area)>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        let named_yaml = path.extension().and_then(|e| e.to_str()) == Some("yaml")
            && path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| !n.starts_with('.'));
        if !named_yaml {
            continue;
        }
        let text =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let area: Area = serde_yaml::from_str(&text)
            .with_context(|| format!("parsing {}", path.display()))?;
        out.push((path, area));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Write `text` to `path` through a temporary sibling, renamed into place.
fn write_atomic(path: &Path, text: &str) -> Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))
}

/// Bring `dir` into agreement with `map`, the result of applying `edit`.
///
/// An area the edit removed has its file removed. Every other area is written
/// when its file is missing or holds something other than the area now in the
/// map — which covers the area the edit named and the parent whose `contains`
/// it adjusted, and nothing else.
pub fn persist(dir: &Path, edit: &MapEdit, map: &MapSet) -> Result<()> {
    let on_disk = files(dir)?;

    if let MapEdit::RemoveArea(id) = edit {
        for (path, _) in on_disk.iter().filter(|(_, a)| &a.id == id) {
            fs::remove_file(path).with_context(|| format!("removing {}", path.display()))?;
        }
    }

    for area in map.areas() {
        let wanted = serde_yaml::to_string(area)
            .with_context(|| format!("serialising area `{}`", area.id))?;
        match on_disk.iter().find(|(_, a)| a.id == area.id) {
            Some((path, held)) => {
                let held = serde_yaml::to_string(held)
                    .with_context(|| format!("serialising {}", path.display()))?;
                if held != wanted {
                    write_atomic(path, &wanted)?;
                }
            }
            None => write_atomic(&dir.join(format!("{}.yaml", area.id)), &wanted)?,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use npc_map::schema::{AreaKind, Node, Where};

    const PARTS: &str = "id: desk\nkind: station\nname: desk\nlong: A desk.\n";

    const HALL: &str = "\
# the hall, hand-written
id: hall
kind: level
name: Hall
summary: s
nodes:
  - id: core
    kind: core
    name: the core
";

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("npcd-mapstore-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(d.join("parts")).unwrap();
        fs::write(d.join("parts").join("desk.yaml"), PARTS).unwrap();
        d
    }

    fn node(id: &str) -> Node {
        serde_yaml::from_str(&format!("id: {id}\nkind: social\nname: {id}\noff: [core]\n"))
            .unwrap()
    }

    fn yard() -> Area {
        serde_yaml::from_str(
            "id: yard\nkind: level\nname: Yard\nsummary: s\nnodes:\n  - id: gate\n    kind: core\n    name: the gate\n",
        )
        .unwrap()
    }

    #[test]
    fn only_the_touched_file_is_rewritten() {
        let d = dir("touched");
        fs::write(d.join("hall.yaml"), HALL).unwrap();
        let other = "# untouched\nid: yard\nkind: level\nname: Yard\nsummary: s\nnodes:\n  - id: gate\n    kind: core\n    name: the gate\n";
        fs::write(d.join("yard.yaml"), other).unwrap();

        let map = MapSet::load_dir(&d).unwrap();
        let edit = MapEdit::AddNode {
            area: "hall".into(),
            node: Box::new(node("annex")),
        };
        let map = map.apply(&edit).unwrap();
        persist(&d, &edit, &map).unwrap();

        assert_eq!(fs::read_to_string(d.join("yard.yaml")).unwrap(), other);
        let hall = fs::read_to_string(d.join("hall.yaml")).unwrap();
        assert!(hall.contains("annex"), "{hall}");
        let reloaded = MapSet::load_dir(&d).unwrap();
        let at = Where::new("hall", "annex");
        assert_eq!(reloaded.node_at(&at).unwrap().exits, vec!["core"]);
        assert!(!d.join("hall.yaml.tmp").exists());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn a_new_area_gets_its_own_file_and_a_removed_one_loses_it() {
        let d = dir("area");
        fs::write(d.join("hall.yaml"), HALL).unwrap();
        let map = MapSet::load_dir(&d).unwrap();

        let add = MapEdit::AddArea(Box::new(yard()));
        let map = map.apply(&add).unwrap();
        persist(&d, &add, &map).unwrap();
        assert!(d.join("yard.yaml").exists());
        assert!(MapSet::load_dir(&d).unwrap().get("yard").is_some());

        let drop = MapEdit::RemoveArea("yard".into());
        let map = map.apply(&drop).unwrap();
        persist(&d, &drop, &map).unwrap();
        assert!(!d.join("yard.yaml").exists());
        assert!(MapSet::load_dir(&d).unwrap().get("yard").is_none());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn an_area_is_found_by_its_id_not_its_file_name() {
        let d = dir("named");
        fs::write(d.join("great-hall.yaml"), HALL).unwrap();
        let map = MapSet::load_dir(&d).unwrap();
        let edit = MapEdit::Retitle {
            at: Where::new("hall", "core"),
            name: "the heart".into(),
        };
        let map = map.apply(&edit).unwrap();
        persist(&d, &edit, &map).unwrap();

        assert!(!d.join("hall.yaml").exists(), "no duplicate was made");
        let text = fs::read_to_string(d.join("great-hall.yaml")).unwrap();
        assert!(text.contains("the heart"), "{text}");
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn a_child_area_added_updates_its_parent_file_too() {
        let d = dir("parent");
        fs::write(d.join("hall.yaml"), HALL).unwrap();
        let map = MapSet::load_dir(&d).unwrap();
        let mut child = yard();
        child.within = Some("hall".into());
        child.kind = AreaKind::Region;
        let edit = MapEdit::AddArea(Box::new(child));
        let map = map.apply(&edit).unwrap();
        persist(&d, &edit, &map).unwrap();

        let hall = fs::read_to_string(d.join("hall.yaml")).unwrap();
        assert!(hall.contains("yard"), "parent now lists its child: {hall}");
        let _ = fs::remove_dir_all(&d);
    }
}
