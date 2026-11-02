//! What a world's round has done (`docs/zen_moon_mind.md` §17): the item last
//! set, when each item was last done or last failed, and when each body's
//! round mission was set — the clock the round's issuing reads
//! (`engine::mission_gen::round`).
//!
//! A doing is recorded as it closes and stamped with the time on the table's
//! next beat, the way standing missions are (`Missions::stamp_standing`): the
//! close happens inside an act, which has no clock of its own.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::engine::mission::Outcome;

/// The round's bookkeeping, kept with the world.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundState {
    /// The id of the item set last — where the round goes on from.
    #[serde(default)]
    last: Option<String>,
    /// When each item was last done: `None` done and not yet stamped.
    #[serde(default)]
    done: BTreeMap<String, Option<u64>>,
    /// When each item last failed, as [`Self::done`] — until it is done.
    #[serde(default)]
    failed: BTreeMap<String, Option<u64>>,
    /// When each body's round mission was set, while it carries one.
    #[serde(default)]
    set_at: BTreeMap<String, u64>,
}

impl RoundState {
    /// The id of the item set last.
    pub fn last(&self) -> Option<&str> {
        self.last.as_deref()
    }

    /// When item `id` was last done: `None` never, `Some(None)` done and not
    /// yet stamped.
    pub fn done_at(&self, id: &str) -> Option<Option<u64>> {
        self.done.get(id).copied()
    }

    /// When item `id` last failed, as [`Self::done_at`] — `None` when it has
    /// not failed since it was last done.
    pub fn failed_at(&self, id: &str) -> Option<Option<u64>> {
        self.failed.get(id).copied()
    }

    /// When `body`'s round mission was set, while it carries one.
    pub fn set_at(&self, body: &str) -> Option<u64> {
        self.set_at.get(body).copied()
    }

    /// Item `id` was set for `body` at `now_ms`.
    pub fn set(&mut self, body: &str, id: &str, now_ms: u64) {
        self.last = Some(id.to_string());
        self.set_at.insert(body.to_string(), now_ms);
    }

    /// `body`'s round mission, item `id`, closed with `outcome`. Done, it is
    /// no longer failing; failed, it rests from its failure.
    pub fn closed(&mut self, body: &str, id: &str, outcome: Outcome) {
        self.set_at.remove(body);
        match outcome {
            Outcome::Pass => {
                self.failed.remove(id);
                self.done.insert(id.to_string(), None);
            }
            Outcome::Fail => {
                self.failed.insert(id.to_string(), None);
            }
        }
    }

    /// `body` carries its round mission no more, closed other than by the
    /// round: called off, or gone.
    pub fn dropped(&mut self, body: &str) {
        self.set_at.remove(body);
    }

    /// Stamp every doing and failure since the last stamp at `now_ms`.
    pub fn stamp(&mut self, now_ms: u64) {
        for at in self.done.values_mut().chain(self.failed.values_mut()) {
            at.get_or_insert(now_ms);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A doing is recorded at its close and stamped on the next beat**; a
    /// failure rests until the item is done again, and setting an item names
    /// it as the one the round goes on from.
    #[test]
    fn a_doing_is_recorded_then_stamped() {
        let mut r = RoundState::default();
        r.set("zen", "watch-earth", 10);
        assert_eq!((r.last(), r.set_at("zen")), (Some("watch-earth"), Some(10)));

        r.closed("zen", "watch-earth", Outcome::Fail);
        assert_eq!(r.set_at("zen"), None);
        assert_eq!(r.failed_at("watch-earth"), Some(None));
        r.stamp(20);
        assert_eq!(r.failed_at("watch-earth"), Some(Some(20)));

        r.set("zen", "watch-earth", 30);
        r.closed("zen", "watch-earth", Outcome::Pass);
        assert_eq!(r.failed_at("watch-earth"), None, "done, no longer failing");
        assert_eq!(r.done_at("watch-earth"), Some(None));
        r.stamp(40);
        r.stamp(50);
        assert_eq!(r.done_at("watch-earth"), Some(Some(40)), "stamped once");

        r.set("zen", "read-the-reports", 60);
        r.dropped("zen");
        assert_eq!(r.set_at("zen"), None);
        assert_eq!(r.done_at("read-the-reports"), None);
    }
}
