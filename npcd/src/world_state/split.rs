//! A facet of the world as entity records (`docs/replicated_substrate.md`
//! §9.1): its JSON taken apart at the maps that hold one entry per entity —
//! each mission, each operation, each body's working set — into one record
//! per entry, and the rest of it into one `meta` record; and put back
//! together the same way.
//!
//! A record's name is `<map path>/<escaped entry key>`, `<facet path>`, or
//! `meta`. A facet changes entry by entry, so writing only the records that
//! differ writes only the entities that changed.
//!
//! A *facet* is one field that changes far more often than the object holding
//! it — the tower's clock, which moves every beat — taken out whole into a
//! record of its own, so the rest of `meta` is not rewritten with it.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use super::keys::{escape, unescape};

/// The record holding everything outside the facet's entity maps.
pub const META: &str = "meta";

/// `value` taken apart at `maps`, each a `/`-separated path to an object
/// whose entries are entities, and at `facets`, each a path to one field
/// taken out whole. A map that is absent or not an object is left in `meta`;
/// a facet is taken out — `null` included — only when its parent is an
/// object holding it, so a parent that is absent or `null` makes no record.
pub fn split(value: Value, maps: &[&str], facets: &[&str]) -> BTreeMap<String, Value> {
    let mut rest = value;
    let mut out = BTreeMap::new();
    for path in maps {
        let Some(Value::Object(entries)) = take(&mut rest, path) else {
            continue;
        };
        for (k, v) in entries {
            out.insert(format!("{path}/{}", escape(&k)), v);
        }
    }
    for path in facets {
        if let Some(v) = take(&mut rest, path) {
            out.insert(path.to_string(), v);
        }
    }
    out.insert(META.to_string(), rest);
    out
}

/// The facet `records` hold, put back together: `meta` with each map path's
/// entries set under it, and each facet that has a record set into its
/// parent — only when the parent is an object, so a `null` parent stays
/// `null`. A facet with no record is left as `meta` holds it.
pub fn join(records: &BTreeMap<String, Value>, maps: &[&str], facets: &[&str]) -> Value {
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
    for path in facets {
        let Some(v) = records.get(*path) else {
            continue;
        };
        let (parent, last) = match path.rsplit_once('/') {
            Some((parent, last)) => (descend(&mut value, parent), last),
            None => (Some(&mut value), *path),
        };
        if let Some(parent) = parent.and_then(Value::as_object_mut) {
            parent.insert(last.to_string(), v.clone());
        }
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
        let records = split(facet(), MAPS, &[]);
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
        assert_eq!(join(&split(facet(), MAPS, &[]), MAPS, &[]), facet());
        let mut emptied = facet();
        emptied["active"] = json!({});
        let mut records = split(facet(), MAPS, &[]);
        records.retain(|k, _| !k.starts_with("active/"));
        assert_eq!(join(&records, MAPS, &[]), emptied);
    }

    /// A map the facet does not have is not made up when splitting, and is
    /// empty when joining.
    #[test]
    fn an_absent_map_stays_in_meta() {
        let records = split(json!({"seed": 1}), MAPS, &[]);
        assert_eq!(records.len(), 1);
        assert_eq!(
            join(&records, MAPS, &[]),
            json!({"seed": 1, "active": {}, "operations": {"ops": {}}})
        );
    }

    const FACETS: &[&str] = &["tower/clock", "tower/contact"];

    fn records(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    /// **A facet is taken out whole when its parent holds it** — a `null`
    /// one included — and makes no record when the parent is `null` or
    /// absent.
    #[test]
    fn a_facet_is_its_own_record_only_under_a_parent() {
        let present = json!({"tower": {"name": "T", "clock": {"carry": 3}, "contact": null}});
        assert_eq!(
            split(present, &[], FACETS),
            records(&[
                ("meta", json!({"tower": {"name": "T"}})),
                ("tower/clock", json!({"carry": 3})),
                ("tower/contact", Value::Null),
            ])
        );
        assert_eq!(
            split(json!({"tower": null}), &[], FACETS),
            records(&[("meta", json!({"tower": null}))])
        );
        assert_eq!(
            split(json!({"seed": 1}), &[], FACETS),
            records(&[("meta", json!({"seed": 1}))])
        );
    }

    /// Joined, a facet goes back into its parent — a `null` one round-trips
    /// as `null` — and a `null` parent is never made into an object by a
    /// facet record that outlived it.
    #[test]
    fn a_facet_joins_back_and_never_makes_its_parent() {
        let value = json!({"tower": {"name": "T", "clock": {"carry": 3}, "contact": null}});
        assert_eq!(join(&split(value.clone(), &[], FACETS), &[], FACETS), value);
        let orphan = records(&[
            ("meta", json!({"tower": null})),
            ("tower/clock", json!({"carry": 3})),
        ]);
        assert_eq!(join(&orphan, &[], FACETS), json!({"tower": null}));
    }

    /// **A meta still holding the facet, with no record of it, keeps it** —
    /// what a build that writes the tower whole leaves.
    #[test]
    fn a_facet_with_no_record_is_left_as_meta_holds_it() {
        let whole = records(&[(
            "meta",
            json!({"tower": {"name": "T", "clock": {"carry": 9}}}),
        )]);
        assert_eq!(
            join(&whole, &[], FACETS),
            json!({"tower": {"name": "T", "clock": {"carry": 9}}})
        );
    }
}
