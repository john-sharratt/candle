//! The machine table (`docs/static_placement.md` §2): what each machine of the
//! deployment takes — conversations by rank, pictures by preference, a share of
//! the cast, and the host RAM its expert cache may page-lock. One file,
//! `host/machines.yaml`, identical on every machine: the daemons compile it in
//! and the gateway reads it, so the two can never disagree.
//!
//! A machine is named as its members' `HELLO` names their host; the name is
//! matched in any case, as `standby_agents` are.

use std::collections::BTreeMap;

use serde::Deserialize;

/// The machine table: each machine's entry, by name.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(transparent)]
pub struct MachineTable {
    machines: BTreeMap<String, Machine>,
}

/// What one machine takes. Every key is optional: a machine named with
/// nothing (`AlienWorld: {}`) takes no conversation, no picture and no
/// character.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Machine {
    /// Where conversations go (§3). Absent: none.
    #[serde(default)]
    pub conversations: Option<Conversations>,
    /// Its picture preference (§4), lower first. Absent: it draws nothing.
    #[serde(default)]
    pub pictures: Option<u32>,
    /// Its share of the cast while it is free of conversations (§5). Zero:
    /// it hosts no character.
    #[serde(default)]
    pub characters: u32,
    /// How much host RAM its expert cache may page-lock, in GiB (§7). Absent:
    /// derived from the machine.
    #[serde(default)]
    pub pinned_experts_gib: Option<u32>,
    /// Its build preference, lower first: a build or test job goes to the
    /// agent on the preferred machine with a free slot, whatever else shares it
    /// (§2). Absent: after every machine that has one.
    #[serde(default)]
    pub builds: Option<u32>,
}

/// A machine's conversations (§3.2): its place in the deterministic fill and
/// the most it takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Conversations {
    /// The order the fill visits machines in, lower first.
    pub rank: u32,
    /// How many live units it is filled to before any round robin.
    #[serde(default)]
    pub min: u32,
    /// The most it takes; equal to `min` when absent.
    #[serde(default)]
    max: Option<u32>,
}

impl Conversations {
    /// The most it takes.
    pub fn max(&self) -> u32 {
        self.max.unwrap_or(self.min)
    }
}

/// Why a machine table was refused.
#[derive(Debug, thiserror::Error)]
pub enum TableError {
    #[error("the machine table does not parse: {0}")]
    Parse(#[from] serde_yaml::Error),
    #[error("{machine}: conversations max {max} is below its min {min}")]
    MaxBelowMin { machine: String, min: u32, max: u32 },
    #[error("{0} and {1} name the same machine")]
    Duplicate(String, String),
}

impl MachineTable {
    /// The table `yaml` states.
    pub fn from_yaml(yaml: &str) -> Result<MachineTable, TableError> {
        let table: MachineTable = serde_yaml::from_str(yaml)?;
        table.check()?;
        Ok(table)
    }

    /// A table of `machines`, for a test or a caller that builds one.
    pub fn of(machines: impl IntoIterator<Item = (String, Machine)>) -> MachineTable {
        MachineTable {
            machines: machines.into_iter().collect(),
        }
    }

    fn check(&self) -> Result<(), TableError> {
        let mut seen: BTreeMap<String, &str> = BTreeMap::new();
        for (name, m) in &self.machines {
            if let Some(other) = seen.insert(name.to_ascii_lowercase(), name) {
                return Err(TableError::Duplicate(other.to_string(), name.clone()));
            }
            if let Some(c) = m.conversations {
                if c.max() < c.min {
                    return Err(TableError::MaxBelowMin {
                        machine: name.clone(),
                        min: c.min,
                        max: c.max(),
                    });
                }
            }
        }
        Ok(())
    }

    /// The entry for the machine `host` names, in any case.
    pub fn get(&self, host: &str) -> Option<&Machine> {
        self.machines
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(host))
            .map(|(_, m)| m)
    }

    /// `host`'s conversations, when it takes any.
    pub fn conversations(&self, host: &str) -> Option<Conversations> {
        self.get(host).and_then(|m| m.conversations)
    }

    /// `host`'s picture preference, when it draws.
    pub fn pictures(&self, host: &str) -> Option<u32> {
        self.get(host).and_then(|m| m.pictures)
    }

    /// Whether any machine of the table takes a unit: one that draws, for a
    /// draw, and one that takes conversations for any other.
    pub fn any_takes(&self, draw: bool) -> bool {
        self.machines.values().any(|m| match draw {
            true => m.pictures.is_some(),
            false => m.conversations.is_some(),
        })
    }

    /// `host`'s build preference, when the table gives one.
    pub fn builds(&self, host: &str) -> Option<u32> {
        self.get(host).and_then(|m| m.builds)
    }

    /// `host`'s share of the cast; zero for a machine the table does not name.
    pub fn characters(&self, host: &str) -> u32 {
        self.get(host).map_or(0, |m| m.characters)
    }

    /// The bytes `host`'s expert cache may page-lock, when the table says.
    pub fn pinned_experts_bytes(&self, host: &str) -> Option<u64> {
        self.get(host)
            .and_then(|m| m.pinned_experts_gib)
            .map(|gib| u64::from(gib) << 30)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLE: &str = "\
BlackWorld:
  conversations: {rank: 1, min: 4, max: 8}
  pictures: 2
  characters: 3
  pinned_experts_gib: 120
  builds: 1
WhiteWorld:
  conversations: {rank: 2, min: 1}
  pictures: 1
  characters: 1
AlienWorld: {}
";

    /// Every key reads back, a `max` left out is the `min`, and a machine
    /// named with nothing takes nothing.
    #[test]
    fn a_table_reads_back_each_machine() {
        let t = MachineTable::from_yaml(TABLE).unwrap();
        let bw = t.conversations("BlackWorld").unwrap();
        assert_eq!((bw.rank, bw.min, bw.max()), (1, 4, 8));
        let ww = t.conversations("WhiteWorld").unwrap();
        assert_eq!((ww.rank, ww.min, ww.max()), (2, 1, 1));
        assert_eq!(
            (t.pictures("BlackWorld"), t.pictures("WhiteWorld")),
            (Some(2), Some(1))
        );
        assert_eq!(
            (t.characters("BlackWorld"), t.characters("WhiteWorld")),
            (3, 1)
        );
        assert_eq!(t.pinned_experts_bytes("BlackWorld"), Some(120 << 30));
        assert_eq!(t.pinned_experts_bytes("WhiteWorld"), None);
        assert_eq!(t.get("AlienWorld"), Some(&Machine::default()));
        assert_eq!(t.conversations("AlienWorld"), None);
        assert_eq!(t.characters("AlienWorld"), 0);
        assert_eq!(
            (
                t.builds("BlackWorld"),
                t.builds("WhiteWorld"),
                t.builds("nowhere")
            ),
            (Some(1), None, None)
        );
    }

    /// Whether the table takes a unit at all: a conversation or a draw.
    #[test]
    fn a_table_says_whether_any_machine_takes_a_unit() {
        let t = MachineTable::from_yaml(TABLE).unwrap();
        assert!(t.any_takes(false) && t.any_takes(true));
        let cast_only = MachineTable::from_yaml("AlienWorld:\n  characters: 1\n").unwrap();
        assert!(!cast_only.any_takes(false) && !cast_only.any_takes(true));
        let talks = MachineTable::from_yaml("WhiteWorld:\n  conversations: {rank: 1}\n").unwrap();
        assert!(talks.any_takes(false) && !talks.any_takes(true));
        assert!(!MachineTable::default().any_takes(false));
    }

    /// A machine is matched in any case, and one the table does not name
    /// takes nothing.
    #[test]
    fn a_machine_is_matched_in_any_case() {
        let t = MachineTable::from_yaml(TABLE).unwrap();
        assert_eq!(t.characters("blackworld"), 3);
        assert_eq!(t.conversations("WHITEWORLD").map(|c| c.rank), Some(2));
        assert_eq!(t.get("GreenWorld"), None);
        assert_eq!(t.characters("GreenWorld"), 0);
    }

    /// A `max` below its `min`, a key no machine has, and two spellings of
    /// one machine are refused.
    #[test]
    fn a_table_that_contradicts_itself_is_refused() {
        let below = "M:\n  conversations: {rank: 1, min: 4, max: 2}\n";
        assert!(matches!(
            MachineTable::from_yaml(below),
            Err(TableError::MaxBelowMin { min: 4, max: 2, .. })
        ));
        assert!(matches!(
            MachineTable::from_yaml("M:\n  power: 3\n"),
            Err(TableError::Parse(_))
        ));
        assert!(matches!(
            MachineTable::from_yaml("M: {}\nm: {}\n"),
            Err(TableError::Duplicate(..))
        ));
    }

    /// The shipped table parses, and names the machines the cluster runs on.
    #[test]
    fn the_shipped_table_parses() {
        let t = MachineTable::from_yaml(include_str!("../../host/machines.yaml")).unwrap();
        assert!(t.conversations("BlackWorld").is_some());
        assert!(t.conversations("WhiteWorld").is_some());
    }
}
