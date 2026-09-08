//! Things the world runs: doors, turrets, fabricators, vehicles, terminals.
//!
//! # One tool, because they are one kind of object
//!
//! A blast door and a wall railgun look nothing alike and are the same thing to
//! the catalogue: something standing in a room, in a state, that a body can put
//! into a different state. `operate { what, mode }` covers both because the
//! *modes* differ and the act does not.
//!
//! This is what stops the catalogue growing a verb per machine. A device added
//! to the world later arrives with its own mode list and the grammar picks it
//! up, with no new tool, no new dispatch arm and no new test file.
//!
//! # The mode set is the device's, which makes the binding dependent
//!
//! Every other live set in the engine answers a question about the *body* —
//! who is here, where can you walk, what do you carry. This one answers a
//! question about another argument: the modes offered for `mode` are the modes
//! of whatever was chosen for `what`. The stencil takes it without strain,
//! because a trie branch can carry its own sub-branch, but it is the one place
//! the binding is `(tool, param, chosen) → values` rather than
//! `(tool, param) → values`.
//!
//! # A turret's targeting policy is a mode
//!
//! *Air only*, *nearest first*, *conserve* are not guidance and not prose. They
//! are the enumerated form of what a sentence would have tried to say, and they
//! are values the firing code branches on. The simulator fires the weapon; this
//! says under what policy.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// What sort of machine it is. Decides nothing mechanical — the modes do that
/// — and everything about how it is described and grouped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Opens, closes, locks. A door, a hatch, a gate, a bulkhead.
    Portal,
    /// Fires under a policy. Wall guns, towers, emplacements.
    Turret,
    /// Makes things. Fabricators, foundries, nano-assemblers.
    Fabricator,
    /// Carries bodies. Vehicles.
    Vehicle,
    /// Reads and writes the record. The vault's terminals and benches.
    Terminal,
    /// Reports on the place itself. Panels, sensors, gauges.
    Panel,
}

/// One machine, where it stands, and what state it is in.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Device {
    pub id: String,
    /// What a character calls it: "the south blast door".
    pub name: String,
    pub kind: Kind,
    /// `area/node` — the room it stands in, matching `npc_map`'s `Where`.
    pub at: String,
    /// The state it is in now. Always one of [`Device::modes`].
    pub mode: String,
    /// Every state it can be put into, in the order they are offered.
    pub modes: Vec<String>,
    /// Whether it answers at all. A broken device is offered and refuses, which
    /// is how a character finds out — *Care of the place* has "report a station
    /// that is not working properly", and it needs something to be wrong with.
    pub working: bool,
    /// Whether working at it takes it, so nobody else can while you are there.
    ///
    /// True for a station and false for a fixture, straight off the map. A
    /// terminal holding an era is claimed; a blast door is not, because a door
    /// somebody has claimed is a door nobody else can shut behind them.
    pub claimable: bool,
}

impl Device {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        kind: Kind,
        at: impl Into<String>,
        modes: &[&str],
    ) -> Device {
        let modes: Vec<String> = modes.iter().map(|m| (*m).to_string()).collect();
        let mode = modes.first().cloned().unwrap_or_default();
        Device {
            id: id.into(),
            name: name.into(),
            kind,
            at: at.into(),
            mode,
            modes,
            working: true,
            claimable: false,
        }
    }

    /// Whether this device admits that state.
    pub fn admits(&self, mode: &str) -> bool {
        let want = mode.trim().to_lowercase();
        self.modes.iter().any(|m| m.to_lowercase() == want)
    }

    /// Put it into a state. `false` when the state is not one of its own — the
    /// grammar should already have made that unreachable, and this is the
    /// backstop for the API and the harness, which are not grammar-constrained.
    pub fn set(&mut self, mode: &str) -> bool {
        let want = mode.trim().to_lowercase();
        match self.modes.iter().find(|m| m.to_lowercase() == want) {
            Some(m) => {
                self.mode = m.clone();
                true
            }
            None => false,
        }
    }
}

/// Every machine in a world.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Devices {
    devices: BTreeMap<String, Device>,
}

impl Devices {
    pub fn new() -> Devices {
        Devices::default()
    }

    pub fn install(&mut self, device: Device) {
        self.devices.insert(device.id.clone(), device);
    }

    pub fn get(&self, id: &str) -> Option<&Device> {
        self.devices.get(id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut Device> {
        self.devices.get_mut(id)
    }

    pub fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }

    pub fn len(&self) -> usize {
        self.devices.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Device> {
        self.devices.values()
    }

    /// What stands in one room, in a stable order.
    pub fn at(&self, place: &str) -> Vec<&Device> {
        self.devices.values().filter(|d| d.at == place).collect()
    }

    /// The names `operate`'s `what` offers to a body standing there.
    pub fn operable_at(&self, place: &str) -> Vec<String> {
        self.at(place).into_iter().map(|d| d.name.clone()).collect()
    }

    /// The modes `operate`'s `mode` offers once `what` is chosen — the
    /// dependent half of the binding.
    ///
    /// Empty for a device that is not there, which drops the branch rather than
    /// offering an arm that leads nowhere.
    pub fn modes_of(&self, place: &str, what: &str) -> Vec<String> {
        self.by_name_at(place, what)
            .map(|d| d.modes.clone())
            .unwrap_or_default()
    }

    /// Find a device in a room by the name a character used for it.
    pub fn by_name_at(&self, place: &str, name: &str) -> Option<&Device> {
        let want = name.trim().to_lowercase();
        self.devices
            .values()
            .find(|d| d.at == place && (d.name.to_lowercase() == want || d.id == want))
    }

    /// The mutable half of [`Self::by_name_at`], for the act that changes it.
    pub fn by_name_at_mut(&mut self, place: &str, name: &str) -> Option<&mut Device> {
        let want = name.trim().to_lowercase();
        self.devices
            .values_mut()
            .find(|d| d.at == place && (d.name.to_lowercase() == want || d.id == want))
    }
}

/// The mode lists the world ships, so a seeded world and a test agree.
pub mod modes {
    /// A door, hatch or gate.
    pub const PORTAL: &[&str] = &["open", "closed", "locked"];
    /// A turret's firing policy. The order is the order they are offered.
    pub const TURRET: &[&str] = &[
        "hold fire",
        "free fire",
        "nearest first",
        "air only",
        "ground only",
        "conserve ammunition",
    ];
    /// A fabricator's running state.
    pub const FABRICATOR: &[&str] = &["idle", "running", "paused", "purging"];
    /// A vehicle.
    pub const VEHICLE: &[&str] = &["parked", "driving", "disembarking"];
    /// An editing terminal — the working states from the tool surface audit.
    pub const TERMINAL: &[&str] = &["reading", "working", "offered"];
    /// A panel is read, not set, so it has exactly one state.
    pub const PANEL: &[&str] = &["reporting"];
}

#[cfg(test)]
mod tests {
    use super::*;

    fn door() -> Device {
        Device::new(
            "south_door",
            "the south blast door",
            Kind::Portal,
            "tower/gatehouse",
            modes::PORTAL,
        )
    }

    fn turret() -> Device {
        Device::new(
            "wall_railgun",
            "the wall railgun",
            Kind::Turret,
            "tower/rampart",
            modes::TURRET,
        )
    }

    #[test]
    fn a_device_starts_in_the_first_mode_it_declares() {
        // Deterministic, so a seeded world reads the same on every run.
        assert_eq!(door().mode, "open");
        assert_eq!(turret().mode, "hold fire");
    }

    #[test]
    fn setting_a_mode_it_has_works_and_one_it_does_not_changes_nothing() {
        let mut d = door();
        assert!(d.set("locked"));
        assert_eq!(d.mode, "locked");
        assert!(!d.set("free fire"), "a door took a turret's mode");
        assert_eq!(d.mode, "locked", "a refused set still moved it");
    }

    #[test]
    fn a_mode_is_matched_however_the_model_cased_it() {
        let mut d = turret();
        assert!(d.set("  Free Fire "));
        assert_eq!(d.mode, "free fire", "the world's own spelling is kept");
    }

    #[test]
    fn the_modes_offered_are_the_modes_of_the_thing_chosen() {
        // The dependent binding, which is the whole reason `operate` is one tool.
        let mut all = Devices::new();
        all.install(door());
        all.install(turret());

        assert_eq!(
            all.modes_of("tower/gatehouse", "the south blast door"),
            vec!["open", "closed", "locked"]
        );
        assert_eq!(all.modes_of("tower/rampart", "the wall railgun")[0], "hold fire");
    }

    #[test]
    fn a_device_in_another_room_is_not_offered_and_has_no_modes_here() {
        let mut all = Devices::new();
        all.install(turret());
        assert!(all.operable_at("tower/gatehouse").is_empty());
        assert!(
            all.modes_of("tower/gatehouse", "the wall railgun").is_empty(),
            "a device two rooms away offered its modes"
        );
    }

    #[test]
    fn what_is_operable_here_is_only_what_stands_here() {
        let mut all = Devices::new();
        all.install(door());
        all.install(turret());
        assert_eq!(
            all.operable_at("tower/rampart"),
            vec!["the wall railgun".to_string()]
        );
    }

    #[test]
    fn a_broken_device_is_still_present_because_something_has_to_be_reported() {
        let mut all = Devices::new();
        let mut d = door();
        d.working = false;
        all.install(d);
        assert_eq!(all.operable_at("tower/gatehouse").len(), 1);
        assert!(!all.get("south_door").unwrap().working);
    }

    #[test]
    fn devices_are_listed_in_a_stable_order() {
        let mut all = Devices::new();
        for id in ["zulu", "alpha", "mike"] {
            all.install(Device::new(id, id, Kind::Panel, "here", modes::PANEL));
        }
        assert_eq!(all.operable_at("here"), vec!["alpha", "mike", "zulu"]);
    }
}
