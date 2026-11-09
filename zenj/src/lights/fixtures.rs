//! The fixture map (§9.1): `zenj/lights.yaml`, every bulb once — its id, its
//! reserved address, its hardware address, where it is along the track, what
//! it points at, and the groups a scene reaches it through. Every bulb is
//! addressed by the address written here, never found by broadcast on the
//! night. A map is refused at load when it is ambiguous: an id or an address
//! twice.
//!
//! ```yaml
//! bulbs:
//!   - {id: t1, ip: 192.168.50.11, mac: "a8:bb:50:00:00:01", pos: 1, aim: floor}
//!   - {id: t2, ip: 192.168.50.12, mac: "a8:bb:50:00:00:02", pos: 2, aim: wall, tags: [left]}
//! ```

use std::collections::BTreeSet;
use std::fs;
use std::net::Ipv4Addr;
use std::path::Path;

use serde::Deserialize;

/// What a bulb points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Aim {
    /// The dancers: where a hit lands.
    Floor,
    /// A wall: a wash that stays off faces.
    Wall,
    /// The hands at the decks.
    Decks,
    /// Everything, softly.
    Wash,
}

impl Aim {
    pub fn name(self) -> &'static str {
        match self {
            Aim::Floor => "floor",
            Aim::Wall => "wall",
            Aim::Decks => "decks",
            Aim::Wash => "wash",
        }
    }
}

/// One bulb.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fixture {
    pub id: String,
    pub ip: Ipv4Addr,
    pub mac: String,
    /// Its place along the track, from one end: what a chase walks.
    pub pos: u32,
    /// Which run of track, when there is more than one.
    #[serde(default)]
    pub run: u32,
    pub aim: Aim,
    #[serde(default)]
    pub tags: Vec<String>,
}

impl Fixture {
    /// Whether `group` names it: one of its tags, or its aim.
    pub fn in_group(&self, group: &str) -> bool {
        self.aim.name() == group || self.tags.iter().any(|t| t == group)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    #[serde(default)]
    bulbs: Vec<Fixture>,
}

/// The bulbs, in the order a chase walks them: by run, then position.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Fixtures {
    pub bulbs: Vec<Fixture>,
}

impl Fixtures {
    pub fn parse(file: &str, yaml: &str) -> Result<Fixtures, String> {
        let f: File = serde_yaml::from_str(yaml).map_err(|e| format!("{file}: {e}"))?;
        let (mut ids, mut ips) = (BTreeSet::new(), BTreeSet::new());
        for b in &f.bulbs {
            if b.id.trim().is_empty() {
                return Err(format!("{file}: a bulb has no id"));
            }
            if !ids.insert(b.id.clone()) {
                return Err(format!("{file}: the id `{}` is given twice", b.id));
            }
            if !ips.insert(b.ip) {
                return Err(format!(
                    "{file}: {} is the address of two bulbs, `{}` among them",
                    b.ip, b.id
                ));
            }
        }
        let mut bulbs = f.bulbs;
        bulbs.sort_by_key(|b| (b.run, b.pos));
        Ok(Fixtures { bulbs })
    }

    /// The map at `path`; none at all when the file is not there — the bulbs
    /// are not fitted yet.
    pub fn load(path: &Path) -> Result<Fixtures, String> {
        match fs::read_to_string(path) {
            Ok(text) => Fixtures::parse(&path.display().to_string(), &text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Fixtures::default()),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.bulbs.is_empty()
    }

    pub fn len(&self) -> usize {
        self.bulbs.len()
    }

    /// The bulbs `group` names, by index; every bulb with no group.
    pub fn in_group(&self, group: Option<&str>) -> Vec<usize> {
        (0..self.bulbs.len())
            .filter(|&i| group.is_none_or(|g| self.bulbs[i].in_group(g)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAP: &str = "\
bulbs:
  - {id: t2, ip: 192.168.50.12, mac: 'a8:bb:50:00:00:02', pos: 2, aim: wall, tags: [left]}
  - {id: t1, ip: 192.168.50.11, mac: 'a8:bb:50:00:00:01', pos: 1, aim: floor}
  - {id: d1, ip: 192.168.50.21, mac: 'a8:bb:50:00:00:03', pos: 1, run: 1, aim: decks}
";

    /// **The map reads in a chase's order**, and a group is a tag or an aim.
    #[test]
    fn the_map_reads_in_a_chases_order() {
        let f = Fixtures::parse("lights.yaml", MAP).unwrap();
        let ids: Vec<&str> = f.bulbs.iter().map(|b| b.id.as_str()).collect();
        assert_eq!(ids, ["t1", "t2", "d1"]);
        assert_eq!(f.in_group(Some("left")), [1]);
        assert_eq!(f.in_group(Some("floor")), [0]);
        assert_eq!(f.in_group(None), [0, 1, 2]);
        assert_eq!(f.bulbs[2].ip, "192.168.50.21".parse::<Ipv4Addr>().unwrap());
    }

    /// **An ambiguous map is refused**, naming what is given twice.
    #[test]
    fn an_ambiguous_map_is_refused() {
        let twice_id = MAP.replace("id: d1", "id: t1");
        assert_eq!(
            Fixtures::parse("f", &twice_id).unwrap_err(),
            "f: the id `t1` is given twice"
        );
        let twice_ip = MAP.replace("192.168.50.21", "192.168.50.11");
        assert_eq!(
            Fixtures::parse("f", &twice_ip).unwrap_err(),
            "f: 192.168.50.11 is the address of two bulbs, `d1` among them"
        );
        let unknown_aim = "bulbs:\n  - {id: x, ip: 1.2.3.4, mac: m, pos: 1, aim: ceiling}\n";
        assert!(Fixtures::parse("f", unknown_aim).is_err());
    }

    /// **No file is no bulbs**: the map is written once they are fitted.
    #[test]
    fn no_file_is_no_bulbs() {
        let dir = tempfile::tempdir().unwrap();
        let none = Fixtures::load(&dir.path().join("lights.yaml")).unwrap();
        assert!(none.is_empty());
    }
}
