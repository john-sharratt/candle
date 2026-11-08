//! The house's table (`docs/house_modes.md` §3): one file of the mind,
//! `house/<world>.yaml`, naming the room the house's devices stand in and
//! what each mode does with them.
//!
//! ```yaml
//! room: sharratt-apartment/front-room
//! modes:
//!   watching:
//!     - {device: stereo, act: on}
//!     - {device: stereo, act: input, source: tv}
//!   away:
//!     - {device: stereo, act: off}
//! ```
//!
//! Each step is one act of the device's bridge, as a part linked to it offers
//! it — `on`, `input`, `volume` for an AV receiver; `stop` for a Cast device —
//! with the act's own arguments beside it. A mode the file does not list has
//! no device steps: it still opens or closes the decks. The file is refused
//! whole when one thing in it is wrong.

use std::collections::{BTreeMap, BTreeSet};

use npc_map::schema::Where;
use serde::Deserialize;
use serde_json::{Map, Value};

use super::mode::Mode;
use crate::mind::files::MindFiles;

/// Where world `world`'s table stands in the mind.
pub fn path(world: &str) -> String {
    format!("house/{world}.yaml")
}

/// One thing a mode does to one device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    /// The part standing for the device, by the name the room calls it.
    pub device: String,
    /// Its bridge's act: `on`, `input`, `stop`.
    pub act: String,
    /// The act's arguments.
    pub args: Map<String, Value>,
}

/// The house as its table says.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Table {
    /// The room the devices stand in, `area/node`: where the mode is set from.
    pub room: String,
    pub modes: BTreeMap<Mode, Vec<Step>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    room: String,
    #[serde(default)]
    modes: BTreeMap<String, Vec<StepFile>>,
}

#[derive(Deserialize)]
struct StepFile {
    device: String,
    act: String,
    #[serde(flatten)]
    args: Map<String, Value>,
}

/// Parse and check a table, whole.
pub fn parse(file: &str, yaml: &str) -> Result<Table, String> {
    let f: File = serde_yaml::from_str(yaml).map_err(|e| format!("{file}: {e}"))?;
    if Where::parse(f.room.trim()).is_none() {
        return Err(format!(
            "{file}: the room `{}` is not `area/node`",
            f.room
        ));
    }
    let mut modes = BTreeMap::new();
    for (name, steps) in f.modes {
        let mode = Mode::parse(&name).ok_or_else(|| {
            format!(
                "{file}: `{name}` is no mode: one of {}",
                Mode::ALL.map(Mode::name).join(", ")
            )
        })?;
        let mut out = Vec::new();
        for s in steps {
            let (device, act) = (s.device.trim(), s.act.trim());
            if device.is_empty() || act.is_empty() {
                return Err(format!("{file}: a step of `{name}` names no device or no act"));
            }
            out.push(Step {
                device: device.to_string(),
                act: act.to_string(),
                args: s.args,
            });
        }
        modes.insert(mode, out);
    }
    Ok(Table {
        room: f.room.trim().to_string(),
        modes,
    })
}

/// World `world`'s table — `None` for a world with no house.
pub fn load(mind: &MindFiles, world: &str) -> Result<Option<Table>, String> {
    let file = path(world);
    match mind.read(&file) {
        Some(text) => parse(&file, &text).map(Some),
        None if mind.is_file(&file) => Err(format!("{file} is not UTF-8")),
        None => Ok(None),
    }
}

impl Table {
    /// What `mode` does to the devices, in order.
    pub fn steps(&self, mode: Mode) -> &[Step] {
        self.modes.get(&mode).map_or(&[], Vec::as_slice)
    }

    /// Every device a mode names.
    pub fn devices(&self) -> BTreeSet<&str> {
        self.modes
            .values()
            .flatten()
            .map(|s| s.device.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const TABLE: &str = "\
room: sharratt-apartment/front-room
modes:
  watching:
    - {device: stereo, act: on}
    - {device: stereo, act: input, source: tv}
  away:
    - {device: stereo, act: off}
    - {device: Chromecast, act: stop}
";

    /// **A table reads as its room and each mode's steps in order**, an
    /// act's arguments beside it; a mode it does not list has none.
    #[test]
    fn a_table_reads_as_its_room_and_its_steps() {
        let t = parse("house/moon.yaml", TABLE).unwrap();
        assert_eq!(t.room, "sharratt-apartment/front-room");
        let watching = t.steps(Mode::Watching);
        assert_eq!(watching.len(), 2);
        assert_eq!(
            (watching[1].device.as_str(), watching[1].act.as_str()),
            ("stereo", "input")
        );
        assert_eq!(Value::Object(watching[1].args.clone()), json!({"source": "tv"}));
        assert!(watching[0].args.is_empty());
        assert!(t.steps(Mode::Party).is_empty());
        assert_eq!(
            t.devices().into_iter().collect::<Vec<_>>(),
            ["Chromecast", "stereo"]
        );
    }

    /// **A table is refused whole**, naming the fault.
    #[test]
    fn a_wrong_table_is_refused() {
        assert_eq!(
            parse("f", "room: nowhere\n").unwrap_err(),
            "f: the room `nowhere` is not `area/node`"
        );
        assert_eq!(
            parse("f", "room: a/b\nmodes:\n  dinner: []\n").unwrap_err(),
            "f: `dinner` is no mode: one of away, asleep, home, listening, watching, party"
        );
        assert_eq!(
            parse("f", "room: a/b\nmodes:\n  home:\n    - {device: '', act: on}\n").unwrap_err(),
            "f: a step of `home` names no device or no act"
        );
        assert!(parse("f", "room: a/b\nrooms: []\n").is_err(), "unknown field");
    }
}
