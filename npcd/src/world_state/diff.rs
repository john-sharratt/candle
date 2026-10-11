//! What a beat changed (`docs/replicated_substrate.md` §9.1): the entity
//! records that differ from what was last written, to be written again, and
//! the ones that are gone, to be tombstoned. At most once per entity per beat,
//! since a beat compares two snapshots, not every change between them.

use std::collections::BTreeMap;

use serde_json::Value;

/// A beat's writes.
#[derive(Debug, Default, PartialEq)]
pub struct Changes {
    /// Entities whose record changed or is new, with their value.
    pub put: Vec<(String, Value)>,
    /// Entities that ceased.
    pub ceased: Vec<String>,
}

impl Changes {
    pub fn is_empty(&self) -> bool {
        self.put.is_empty() && self.ceased.is_empty()
    }
}

/// From `before`, what was last written, to `after`, the world now.
pub fn diff(before: &BTreeMap<String, Value>, after: &BTreeMap<String, Value>) -> Changes {
    Changes {
        put: after
            .iter()
            .filter(|(k, v)| before.get(*k) != Some(*v))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        ceased: before
            .keys()
            .filter(|k| !after.contains_key(*k))
            .cloned()
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn map(entries: &[(&str, Value)]) -> BTreeMap<String, Value> {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn only_what_changed_is_written_and_what_left_is_ceased() {
        let before = map(&[
            ("body/a", json!({"at": "hall/door"})),
            ("body/b", json!({"at": "hall/desk"})),
            ("table", json!({"open": true})),
        ]);
        let after = map(&[
            ("body/a", json!({"at": "hall/window"})),
            ("table", json!({"open": true})),
            ("lift", json!({"floor": 1})),
        ]);
        assert_eq!(
            diff(&before, &after),
            Changes {
                put: vec![
                    ("body/a".into(), json!({"at": "hall/window"})),
                    ("lift".into(), json!({"floor": 1})),
                ],
                ceased: vec!["body/b".into()],
            }
        );
        assert!(diff(&after, &after).is_empty());
    }
}
