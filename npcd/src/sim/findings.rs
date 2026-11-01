//! What the last survey of each room found (`docs/zen_moon_mind.md` §13): the
//! devices on the house network, kept with the room they were surveyed from,
//! for the world-building acts to be grounded in.
//!
//! World state, replicated with the rest of it, so a survey made on the
//! world's authority is the one every member reads.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::bridge::finding::Finding;

/// Each room's last survey, by `area/node`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Findings {
    by_room: BTreeMap<String, Vec<Finding>>,
}

impl Findings {
    /// What a survey of `room` found, in place of what the last one did.
    pub fn surveyed(&mut self, room: &str, found: Vec<Finding>) {
        self.by_room.insert(room.to_string(), found);
    }

    /// What the last survey of `room` found.
    pub fn of(&self, room: &str) -> &[Finding] {
        self.by_room
            .get(room)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    /// The finding in `room` labelled `label`.
    pub fn labelled(&self, room: &str, label: &str) -> Option<&Finding> {
        self.of(room).iter().find(|f| f.answers_to(label))
    }

    /// Whether `room` has been surveyed.
    pub fn surveyed_in(&self, room: &str) -> bool {
        self.by_room.contains_key(room)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finding(address: &str, name: &str) -> Finding {
        Finding {
            address: address.into(),
            hardware: "f0:f6:c1:a0:79:b4".into(),
            names: vec![name.into()],
            profile: "sonos".into(),
            ..Finding::default()
        }
    }

    /// **A room's survey replaces its last, and is found by its labels.**
    #[test]
    fn a_rooms_survey_is_kept_and_found_by_label() {
        let mut f = Findings::default();
        let room = "sharratt-apartment/front-room";
        assert!(!f.surveyed_in(room));
        f.surveyed(room, vec![finding("192.168.0.89", "SonosZP")]);
        assert!(f.surveyed_in(room));
        assert_eq!(
            f.labelled(room, " sonoszp at 192.168.0.89 ")
                .map(|x| x.address.as_str()),
            Some("192.168.0.89")
        );
        assert!(f
            .labelled("moon-command/watch-room", "SonosZP at 192.168.0.89")
            .is_none());
        f.surveyed(room, Vec::new());
        assert!(f.of(room).is_empty(), "the last survey stands");
    }
}
