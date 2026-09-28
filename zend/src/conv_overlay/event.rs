//! What one of a conversation's file events says — the `body` the substrate
//! keeps and never reads (`docs/zend_vfs_events.md` §4).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use zend_vfs::{SavedBase, TimedDelta};

/// The key a repository's base is kept under. No path is empty, so it names
/// nothing else.
pub const BASE_KEY: &str = "";

/// One file event.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FileEvent {
    /// The store's base — under [`BASE_KEY`]. The latest wins.
    Base { base: SavedBase },
    /// One delta, appended to its path's chain — or, with `start`, beginning
    /// it again: a chain written whole starts here, and whatever of the path
    /// came before is superseded, whether or not its tombstone landed.
    Delta {
        delta: TimedDelta,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        start: bool,
    },
    /// The path's chain after its last delta: the size it leaves the file at
    /// (`None` once deleted) and whether it is in conflict. With `start`, a
    /// chain written whole with no delta at all — a conflict kept as the
    /// base holds it.
    State {
        size: Option<usize>,
        conflict: bool,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        start: bool,
    },
}

impl FileEvent {
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("a file event always serialises to JSON")
    }

    pub fn from_value(value: &Value) -> Result<Self, String> {
        serde_json::from_value(value.clone()).map_err(|e| format!("not a file event: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use zend_vfs::FileDelta;

    /// **Each event is exactly this JSON**, and reads back as itself.
    #[test]
    fn events_are_exactly_this_json() {
        let state = FileEvent::State {
            size: Some(12),
            conflict: true,
            start: false,
        };
        assert_eq!(
            state.to_value(),
            json!({"kind": "state", "size": 12, "conflict": true})
        );
        let deleted = FileEvent::State {
            size: None,
            conflict: false,
            start: false,
        };
        assert_eq!(
            deleted.to_value(),
            json!({"kind": "state", "size": null, "conflict": false})
        );
        let replace = TimedDelta {
            at_ns: 5,
            delta: FileDelta::Replace {
                content: "x\n".into(),
            },
        };
        let delta = FileEvent::Delta {
            delta: replace.clone(),
            start: false,
        };
        assert_eq!(
            delta.to_value(),
            json!({"kind": "delta", "delta": {"at_ns": 5, "kind": "replace", "content": "x\n"}})
        );
        let start = FileEvent::Delta {
            delta: replace,
            start: true,
        };
        assert_eq!(
            start.to_value(),
            json!({"kind": "delta", "delta": {"at_ns": 5, "kind": "replace", "content": "x\n"}, "start": true})
        );
        for event in [state, deleted, delta, start] {
            assert_eq!(FileEvent::from_value(&event.to_value()).unwrap(), event);
        }
    }

    #[test]
    fn a_body_that_is_no_event_is_refused() {
        assert!(FileEvent::from_value(&json!({"kind": "rename"})).is_err());
        assert!(FileEvent::from_value(&json!({"kind": "state", "size": 1})).is_err());
    }
}
