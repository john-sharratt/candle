//! A facet of the world as entity records (`docs/replicated_substrate.md`
//! §9.1): its JSON taken apart at the maps that hold one entry per entity —
//! each mission, each operation, each body's working set — into one record
//! per entry, and the rest of it into one `meta` record; and put back
//! together the same way.
//!
//! A record's name is `<map path>/<escaped entry key>`, or `meta`. A facet
//! changes entry by entry, so writing only the records that differ writes only
//! the entities that changed.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use super::keys::{escape, unescape};

/// The record holding everything outside the facet's entity maps.
pub const META: &str = "meta";

/// `value` taken apart at `maps`, each a `/`-separated path to an object
/// whose entries are entities. A map that is absent or not an object is left
/// in `meta`.
pub fn split(value: &Value, maps: &[&str]) -> BTreeMap<String, Value> {
    let mut rest = value.clone();
    let mut out = BTreeMap::new();
    for path in maps {
        let Some(Value::Object(entries)) = take(&mut rest, path) else {
            continue;
        };
        for (k, v) in entries {
            out.insert(format!("{path}/{}", escape(&k)), v);
        }
    }
    out.insert(META.to_string(), rest);
    out
}

/// The facet `records` hold, put back together: `meta` with each map path's
/// entries set under it.
pub fn join(records: &BTreeMap<String, Value>, maps: &[&str]) -> Value {
    let mut value = records
        .get(META)
        .cloned()
        .unwrap_or(Value::Object(Map::new()));
    for path in maps {
        let prefix = format!("{path}/");
        let entries: Map<String, Value> = records
            .iter()
            .filter_map(|(name, v)| {
                let k = name.strip_prefix(&prefix)?;
                (!k.contains('/')).then(|| (unescape(k), v.clone()))
            })
            .collect();
        put(&mut value, path, Value::Object(entries));
    }
    value
}

/// Remove and return what `path` names in `value`.
fn take(value: &mut Value, path: &str) -> Option<Value> {
    let (parent, last) = match path.rsplit_once('/') {
        Some((parent, last)) => (descend(value, parent)?, last),
        None => (value, path),
    };
    parent.as_object_mut()?.remove(last)
}

/// Set what `path` names in `value`, making the objects on the way.
fn put(value: &mut Value, path: &str, v: Value) {
    let mut at = value;
    let mut segments = path.split('/').peekable();
    while let Some(segment) = segments.next() {
        if !at.is_object() {
            *at = Value::Object(Map::new());
        }
        let object = at.as_object_mut().expect("made an object above");
        if segments.peek().is_none() {
            object.insert(segment.to_string(), v);
            return;
        }
        at = object
            .entry(segment.to_string())
            .or_insert_with(|| Value::Object(Map::new()));
    }
}

fn descend<'a>(value: &'a mut Value, path: &str) -> Option<&'a mut Value> {
    let mut at = value;
    for segment in path.split('/') {
        at = at.as_object_mut()?.get_mut(segment)?;
    }
    Some(at)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const MAPS: &[&str] = &["active", "operations/ops"];

    fn facet() -> Value {
        json!({
            "active": {"maker-1": {"step": 2}, "maker/2": {"step": 0}},
            "seed": 7,
            "operations": {"next": 4, "ops": {"3": {"name": "Iron Lantern"}}},
        })
    }

    /// Each entity is a record of its own, under its map's path, and what is
    /// left is `meta`.
    #[test]
    fn a_facet_splits_into_one_record_per_entity() {
        let records = split(&facet(), MAPS);
        let want: BTreeMap<String, Value> = [
            ("active/maker%2F2", json!({"step": 0})),
            ("active/maker-1", json!({"step": 2})),
            ("meta", json!({"seed": 7, "operations": {"next": 4}})),
            ("operations/ops/3", json!({"name": "Iron Lantern"})),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        assert_eq!(records, want);
    }

    /// Joined, the records are the facet again — an emptied map included.
    #[test]
    fn records_join_back_into_the_facet() {
        assert_eq!(join(&split(&facet(), MAPS), MAPS), facet());
        let mut emptied = facet();
        emptied["active"] = json!({});
        let mut records = split(&facet(), MAPS);
        records.retain(|k, _| !k.starts_with("active/"));
        assert_eq!(join(&records, MAPS), emptied);
    }

    /// A map the facet does not have is not made up when splitting, and is
    /// empty when joining.
    #[test]
    fn an_absent_map_stays_in_meta() {
        let records = split(&json!({"seed": 1}), MAPS);
        assert_eq!(records.len(), 1);
        assert_eq!(
            join(&records, MAPS),
            json!({"seed": 1, "active": {}, "operations": {"ops": {}}})
        );
    }
}
