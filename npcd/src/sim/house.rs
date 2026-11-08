//! The house's mode as the world holds it (`docs/house_modes.md` §3): the
//! mode the mirrored home was last set to, when, and by whom — kept and
//! replicated with the rest of the world — and its table, the mind's, read
//! afresh rather than saved.

use serde::{Deserialize, Serialize};

use crate::house::{Mode, Table};

/// A mode, as it was set.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Set {
    pub mode: Mode,
    pub at_ms: u64,
    /// Who set it, by the name the world writes.
    pub by: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct House {
    /// The mode it is in; `None` until it is first set.
    now: Option<Set>,
    /// The mind's `house/<world>.yaml` — `None` for a world with no house.
    #[serde(skip)]
    table: Option<Table>,
}

impl House {
    /// The world's table, as the mind last said.
    pub fn set_table(&mut self, table: Option<Table>) {
        self.table = table;
    }

    pub fn table(&self) -> Option<&Table> {
        self.table.as_ref()
    }

    /// The mode it is in, as set.
    pub fn now(&self) -> Option<&Set> {
        self.now.as_ref()
    }

    /// The mode it is in.
    pub fn mode(&self) -> Option<Mode> {
        self.now.as_ref().map(|s| s.mode)
    }

    /// The house is now in `mode`, set by `by` at `at_ms`.
    pub fn set(&mut self, mode: Mode, by: &str, at_ms: u64) {
        self.now = Some(Set {
            mode,
            at_ms,
            by: by.to_string(),
        });
    }

    /// The modes offered to a body standing in `place`: every mode, in the
    /// room the house's devices stand in, and none anywhere else.
    pub fn modes_at(&self, place: &str) -> Vec<String> {
        match &self.table {
            Some(t) if t.room == place => Mode::ALL.iter().map(|m| m.name().to_string()).collect(),
            _ => Vec::new(),
        }
    }

    /// The house's line in the reading of `place`, for the body that sets its
    /// mode: what it is in, and how to change it. `None` outside its room.
    pub fn line(&self, place: &str) -> Option<String> {
        let room = &self.table.as_ref()?.room;
        if room != place {
            return None;
        }
        let modes = Mode::ALL.map(Mode::name).join(", ");
        Some(match &self.now {
            Some(s) => format!(
                "The house, in {}: {}, set by {}. Change it with `house_mode`, giving `mode` as \
                 one of {modes}.",
                s.mode.name(),
                s.mode.what(),
                s.by
            ),
            None => format!(
                "The house, its mode not yet set. Set it with `house_mode`, giving `mode` as one \
                 of {modes}."
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> Table {
        Table {
            room: "home/front-room".into(),
            ..Table::default()
        }
    }

    /// **The modes are offered in the house's room alone**, and only in a
    /// world whose mind has a table.
    #[test]
    fn the_modes_are_offered_in_the_houses_room_alone() {
        let mut h = House::default();
        assert!(h.modes_at("home/front-room").is_empty(), "no table");
        h.set_table(Some(table()));
        assert_eq!(h.modes_at("home/front-room").len(), Mode::ALL.len());
        assert!(h.modes_at("home/back-hall").is_empty());
    }

    /// **The room's reading says what the house is in**, and only in its room.
    #[test]
    fn the_reading_says_what_the_house_is_in() {
        let mut h = House::default();
        assert_eq!(h.line("home/front-room"), None, "no table");
        h.set_table(Some(table()));
        assert_eq!(
            h.line("home/front-room").as_deref(),
            Some(
                "The house, its mode not yet set. Set it with `house_mode`, giving `mode` as one \
                 of away, asleep, home, listening, watching, party."
            )
        );
        h.set(Mode::Watching, "Zen", 1);
        assert_eq!(
            h.line("home/front-room").as_deref(),
            Some(
                "The house, in watching: the television is on, set by Zen. Change it with \
                 `house_mode`, giving `mode` as one of away, asleep, home, listening, watching, \
                 party."
            )
        );
        assert_eq!(h.line("home/back-hall"), None);
    }

    /// **A mode set is kept with who set it; the table is not saved**, being
    /// the mind's.
    #[test]
    fn a_mode_is_kept_and_the_table_is_not() {
        let mut h = House::default();
        h.set_table(Some(table()));
        h.set(Mode::Party, "Zen", 7);
        assert_eq!(h.mode(), Some(Mode::Party));
        let saved: House = serde_json::from_str(&serde_json::to_string(&h).unwrap()).unwrap();
        assert_eq!(saved.now(), h.now());
        assert!(saved.table().is_none());
    }
}
