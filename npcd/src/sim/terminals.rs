//! What a mind with no body reaches real things through
//! (`docs/zen_everywhere.md` §5): the devices the house network has shown,
//! each by the name it goes by, with what the world has earned to work it; and
//! which terminal each such mind is connected to.
//!
//! **A terminal is not a place on the map.** The devices a room's parts stand
//! for (`findings`, `PartLink`) are that room's; these are the house's, known
//! by their hardware address wherever they are, and worked by whoever is
//! connected to them from nowhere at all.
//!
//! World state, replicated with the rest of it, so a device named or paired
//! on the world's authority is the one every member reads.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::bridge::finding::{Finding, WIZ};

/// How many surveys in a row must miss a watched handset before it is said to
/// have left: one missed survey is a phone asleep, not a person gone out.
pub const MISSES_TO_LEAVE: u32 = 2;

/// One device the house network has shown.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Device {
    /// What the last survey that saw it said of it.
    pub finding: Finding,
    /// The name it goes by: what it is listed and worked under.
    pub name: String,
    /// What a pairing earned, for the world to work it with. A key beginning
    /// `_` is the world's own bookkeeping and is never handed to the device.
    #[serde(default)]
    pub creds: BTreeMap<String, String>,
    /// Whose handset it is, when its coming and going is watched.
    #[serde(default)]
    pub person: Option<String>,
    /// Whether it is on the network, as the surveys have it.
    #[serde(default)]
    pub present: bool,
    /// The surveys in a row that did not see it.
    #[serde(default)]
    pub missed: u32,
}

/// A watched handset that joined the house network or left it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Presence {
    /// Whose it is.
    pub person: String,
    /// The handset, by the name it goes by.
    pub device: String,
    /// Whether it joined; `false` when it left.
    pub joined: bool,
}

impl Presence {
    /// How it reads to whoever is told.
    pub fn line(&self) -> String {
        match self.joined {
            true => format!(
                "{}'s handset ({}) has joined the house network: {} is home.",
                self.person, self.device, self.person
            ),
            false => format!(
                "{}'s handset ({}) has left the house network: {} has gone out.",
                self.person, self.device, self.person
            ),
        }
    }
}

/// The house's devices and who is connected to what.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Terminals {
    /// Every device a survey has shown, by its hardware address.
    #[serde(default)]
    devices: BTreeMap<String, Device>,
    /// The terminal each mind is connected to, by its body: at most one.
    #[serde(default)]
    connected: BTreeMap<String, String>,
}

impl Terminals {
    /// Take in a survey of the house network: each device found is kept under
    /// its hardware address — a new one named for what it calls itself, a
    /// known one followed to its new address with its name and what was earned
    /// for it kept — and each known one not found is counted missed. Returns
    /// every watched handset whose standing changed.
    pub fn surveyed(&mut self, found: Vec<Finding>) -> Vec<Presence> {
        let mut seen: Vec<String> = Vec::new();
        let mut changes = Vec::new();
        for finding in found {
            let key = key_of(&finding);
            if key.is_empty() {
                continue;
            }
            seen.push(key.clone());
            match self.devices.get_mut(&key) {
                Some(known) => {
                    let was_here = known.present;
                    known.finding = finding;
                    known.present = true;
                    known.missed = 0;
                    if let (false, Some(person)) = (was_here, &known.person) {
                        changes.push(Presence {
                            person: person.clone(),
                            device: known.name.clone(),
                            joined: true,
                        });
                    }
                }
                None => {
                    let name = self.free_name(&first_name(&finding));
                    self.devices.insert(
                        key,
                        Device {
                            finding,
                            name,
                            present: true,
                            ..Device::default()
                        },
                    );
                }
            }
        }
        for (key, known) in self.devices.iter_mut() {
            if seen.contains(key) || !known.present {
                continue;
            }
            known.missed += 1;
            if known.missed < MISSES_TO_LEAVE {
                continue;
            }
            known.present = false;
            if let Some(person) = &known.person {
                changes.push(Presence {
                    person: person.clone(),
                    device: known.name.clone(),
                    joined: false,
                });
            }
        }
        changes
    }

    /// `wanted`, or — when another device already goes by it — `wanted` with
    /// the first number after it that is free.
    fn free_name(&self, wanted: &str) -> String {
        if !self.is_named(wanted) {
            return wanted.to_string();
        }
        (2..)
            .map(|n| format!("{wanted} {n}"))
            .find(|n| !self.is_named(n))
            .expect("some number is free")
    }

    fn is_named(&self, name: &str) -> bool {
        self.devices
            .values()
            .any(|d| d.name.eq_ignore_ascii_case(name))
    }

    /// Every device, in the order of the names they go by.
    pub fn devices(&self) -> Vec<&Device> {
        let mut all: Vec<&Device> = self.devices.values().collect();
        all.sort_by_key(|d| d.name.to_lowercase());
        all
    }

    /// The device that goes by `name`.
    pub fn named(&self, name: &str) -> Option<&Device> {
        self.devices
            .values()
            .find(|d| d.name.eq_ignore_ascii_case(name.trim()))
    }

    /// The bulbs, in the order of their names: what the one `lights` terminal
    /// works.
    pub fn lights(&self) -> Vec<&Device> {
        self.devices()
            .into_iter()
            .filter(|d| d.finding.profile == WIZ)
            .collect()
    }

    /// Call the device that goes by `name` `new` instead. Refused, with why,
    /// when there is no such device, the new name is empty or too long, or
    /// another device already goes by it.
    pub fn rename(&mut self, name: &str, new: &str) -> Result<(), String> {
        let new = new.trim();
        if new.is_empty() || new.chars().count() > 40 {
            return Err("A device's name is one to forty characters.".into());
        }
        let Some(key) = self.key_named(name) else {
            return Err(format!("There is no device called {name}."));
        };
        let taken = self
            .devices
            .iter()
            .any(|(k, d)| *k != key && d.name.eq_ignore_ascii_case(new));
        if taken {
            return Err(format!("Another device is already called {new}."));
        }
        self.devices.get_mut(&key).expect("just found").name = new.to_string();
        Ok(())
    }

    /// Watch the device that goes by `name` as `person`'s handset, or — with
    /// `None` — stop watching it. `false` when there is no such device.
    pub fn watch(&mut self, name: &str, person: Option<&str>) -> bool {
        let Some(key) = self.key_named(name) else {
            return false;
        };
        let device = self.devices.get_mut(&key).expect("just found");
        device.person = person
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(str::to_string);
        true
    }

    /// Whether any handset is watched: what a survey is worth making for with
    /// nobody asking.
    pub fn watches(&self) -> bool {
        self.devices.values().any(|d| d.person.is_some())
    }

    /// Keep what a pairing earned for the device with hardware address
    /// `hardware`: `set` merged into what it holds, the keys in `drop` removed.
    pub fn keep_creds(&mut self, hardware: &str, set: &BTreeMap<String, String>, drop: &[String]) {
        let Some(device) = self.devices.get_mut(hardware) else {
            return;
        };
        device.creds.extend(set.clone());
        for key in drop {
            device.creds.remove(key);
        }
    }

    fn key_named(&self, name: &str) -> Option<String> {
        self.devices
            .iter()
            .find(|(_, d)| d.name.eq_ignore_ascii_case(name.trim()))
            .map(|(k, _)| k.clone())
    }

    /// Connect `body` to `terminal`, in place of whatever it was connected to.
    pub fn connect(&mut self, body: &str, terminal: &str) {
        self.connected
            .insert(body.to_string(), terminal.to_string());
    }

    /// Let `body`'s connection go. Returns what it was connected to.
    pub fn disconnect(&mut self, body: &str) -> Option<String> {
        self.connected.remove(body)
    }

    /// What `body` is connected to.
    pub fn connected(&self, body: &str) -> Option<&str> {
        self.connected.get(body).map(String::as_str)
    }
}

/// What a device is kept under: its hardware address, lowercase with colons —
/// or, for one that showed none, its network address.
fn key_of(f: &Finding) -> String {
    match f.hardware.trim().is_empty() {
        false => f.hardware.trim().to_ascii_lowercase().replace('-', ":"),
        true => f.address.trim().to_string(),
    }
}

/// What a newly found device is called: what it calls itself, or what it is
/// and where.
fn first_name(f: &Finding) -> String {
    match f.names.first().map(|n| n.trim()).filter(|n| !n.is_empty()) {
        Some(name) => name.to_string(),
        None => format!("{} at {}", f.profile, f.address),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::finding::{DEVICE, SONOS};

    fn finding(hardware: &str, address: &str, name: &str, profile: &str) -> Finding {
        Finding {
            address: address.into(),
            hardware: hardware.into(),
            names: vec![name.into()],
            profile: profile.into(),
            ..Finding::default()
        }
    }

    fn house() -> Vec<Finding> {
        vec![
            finding("A8:BB:50:00:00:01", "192.168.1.21", "wiz_a1", WIZ),
            finding("a8:bb:50:00:00:02", "192.168.1.22", "wiz_a1", WIZ),
            finding("f0:f6:c1:a0:79:b4", "192.168.1.89", "SonosZP", SONOS),
            finding("3c:22:fb:00:00:09", "192.168.1.50", "Pixel-8", DEVICE),
        ]
    }

    /// **A device is the same device across a change of address**, keeping
    /// the name it was given and what was earned for it; two that call
    /// themselves the same are told apart.
    #[test]
    fn a_device_is_followed_by_its_hardware_address() {
        let mut t = Terminals::default();
        assert!(t.surveyed(house()).is_empty(), "nobody is watched yet");
        let names: Vec<&str> = t.devices().iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, ["Pixel-8", "SonosZP", "wiz_a1", "wiz_a1 2"]);
        assert_eq!(t.lights().len(), 2);

        t.rename("wiz_a1", "kitchen light").unwrap();
        t.keep_creds(
            "f0:f6:c1:a0:79:b4",
            &BTreeMap::from([("key".to_string(), "K".to_string())]),
            &[],
        );
        let mut moved = house();
        moved[0].address = "192.168.1.77".into();
        moved[0].names = vec!["wiz_renamed_by_router".into()];
        t.surveyed(moved);
        let kitchen = t.named("Kitchen Light").expect("kept its name");
        assert_eq!(kitchen.finding.address, "192.168.1.77");
        assert_eq!(t.named("SonosZP").unwrap().creds["key"], "K");
        assert_eq!(
            t.devices().len(),
            4,
            "no second device for the one that moved"
        );

        assert_eq!(
            t.rename("SonosZP", "kitchen light").unwrap_err(),
            "Another device is already called kitchen light."
        );
        assert_eq!(
            t.rename("the toaster", "x").unwrap_err(),
            "There is no device called the toaster."
        );
        assert!(t.rename("SonosZP", "  ").is_err());
    }

    /// **A watched handset is said to have left only once it is missed twice
    /// running, and to have joined the moment it is seen again.** One missed
    /// survey is a phone asleep.
    #[test]
    fn a_watched_handset_leaves_after_two_misses_and_joins_at_once() {
        let mut t = Terminals::default();
        t.surveyed(house());
        assert!(!t.watches());
        assert!(t.watch("pixel-8", Some("Kate")));
        assert!(!t.watch("the toaster", Some("Kate")));
        assert!(t.watches());

        let without: Vec<Finding> = house().into_iter().take(3).collect();
        assert!(t.surveyed(without.clone()).is_empty(), "one miss is asleep");
        let left = t.surveyed(without.clone());
        assert_eq!(
            left,
            [Presence {
                person: "Kate".into(),
                device: "Pixel-8".into(),
                joined: false
            }]
        );
        assert_eq!(
            left[0].line(),
            "Kate's handset (Pixel-8) has left the house network: Kate has gone out."
        );
        assert!(t.surveyed(without).is_empty(), "told once");
        let back = t.surveyed(house());
        assert_eq!(
            back[0].line(),
            "Kate's handset (Pixel-8) has joined the house network: Kate is home."
        );
        assert!(t.surveyed(house()).is_empty(), "told once");

        assert!(t.watch("Pixel-8", None));
        assert!(!t.watches());
    }

    /// **A mind is connected to one terminal at a time**: connecting elsewhere
    /// replaces the connection, and letting go says what it was.
    #[test]
    fn a_mind_is_connected_to_one_terminal_at_a_time() {
        let mut t = Terminals::default();
        assert_eq!(t.connected("zen"), None);
        t.connect("zen", "lights");
        t.connect("zen", "SonosZP");
        assert_eq!(t.connected("zen"), Some("SonosZP"));
        assert_eq!(t.disconnect("zen").as_deref(), Some("SonosZP"));
        assert_eq!(t.disconnect("zen"), None);
    }
}
