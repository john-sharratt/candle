//! The world's day rolling over, as its journal hears of it
//! (`docs/npc_conversations.md` §5.1.1): one line in `bots/world` when a
//! world the authority writes enters a new day of world time.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::clock::{day_of, WorldTime};

/// The day each world was last seen in.
#[derive(Default)]
pub struct Days {
    seen: Mutex<HashMap<String, u64>>,
}

impl Days {
    pub fn new() -> Days {
        Days::default()
    }

    /// The line `world` is told at `world_ms`, when that is a day later than
    /// the one it was last seen in. The first sight of a world only marks its
    /// day: a member that starts, or becomes the authority, mid-day has seen
    /// no roll-over.
    pub fn rolled(&self, world: &str, world_ms: u64) -> Option<String> {
        let day = day_of(world_ms);
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let before = seen.insert(world.to_string(), day);
        match before {
            Some(was) if day > was => Some(day_line(world_ms)),
            _ => None,
        }
    }
}

/// What the world's journal is told when a day begins.
pub fn day_line(world_ms: u64) -> String {
    format!("A new day began: {}.", WorldTime::of(world_ms).date())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::DAY_MS;

    #[test]
    fn a_day_rolls_over_once_and_a_first_sight_is_no_roll_over() {
        let days = Days::new();
        assert_eq!(days.rolled("moon", DAY_MS * 3 + 10), None, "first sight");
        assert_eq!(days.rolled("moon", DAY_MS * 3 + 500), None, "same day");
        assert_eq!(
            days.rolled("moon", DAY_MS * 4 + 1),
            Some(day_line(DAY_MS * 4 + 1))
        );
        assert_eq!(days.rolled("moon", DAY_MS * 4 + 2), None, "rolled already");
        assert_eq!(days.rolled("battle-cities", DAY_MS * 9), None, "its own first sight");
        assert_eq!(days.rolled("moon", DAY_MS * 2), None, "world time set back");
    }

    #[test]
    fn the_line_names_the_date() {
        assert_eq!(day_line(0), "A new day began: 1 Jan 1970.");
    }
}
