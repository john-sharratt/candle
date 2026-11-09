//! Members' listings as one (`Aggregate(Union)`): the first body's fields,
//! each top-level array the union of every body's by item `id` in first-seen
//! order, a duplicate replaced by the copy [`further_along`].
//!
//! Each array keeps an index of its items' ids, so an item is matched in one
//! lookup rather than a search of everything merged before it. An id is keyed
//! by its JSON text when it is a string or a number; an id of any other type
//! is matched by comparing it with each item's, as equal JSON values.

use std::collections::HashMap;

use serde_json::Value;

/// `bodies` as one. An item with no `id`, or a `null` one, is kept as it
/// comes.
pub fn merge_by_id(bodies: Vec<Value>) -> Value {
    let mut bodies = bodies.into_iter();
    let Some(mut out) = bodies.next() else {
        return Value::Null;
    };
    let mut indexes: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for body in bodies {
        let (Value::Object(into), Value::Object(from)) = (&mut out, body) else {
            continue;
        };
        for (field, value) in from {
            match (into.get_mut(&field), value) {
                (Some(Value::Array(have)), Value::Array(more)) => {
                    let index = indexes.entry(field).or_insert_with(|| index_of(have));
                    for item in more {
                        merge_item(have, index, item);
                    }
                }
                (None, value) => {
                    into.insert(field, value);
                }
                _ => {}
            }
        }
    }
    out
}

/// `item` merged into `have`, whose keyed ids `index` holds.
fn merge_item(have: &mut Vec<Value>, index: &mut HashMap<String, usize>, item: Value) {
    let id = item.get("id").filter(|v| !v.is_null());
    let key = id.and_then(key_of);
    let at = match (id, &key) {
        (None, _) => None,
        (Some(_), Some(k)) => index.get(k).copied(),
        (Some(id), None) => have.iter().position(|h| h.get("id") == Some(id)),
    };
    match at {
        Some(i) if further_along(&item, &have[i]) => have[i] = item,
        Some(_) => {}
        None => {
            if let Some(k) = key {
                index.insert(k, have.len());
            }
            have.push(item);
        }
    }
}

/// The keyed ids of `items`, each at its first occurrence.
fn index_of(items: &[Value]) -> HashMap<String, usize> {
    let mut index = HashMap::new();
    for (i, item) in items.iter().enumerate() {
        if let Some(k) = item.get("id").and_then(key_of) {
            index.entry(k).or_insert(i);
        }
    }
    index
}

/// The key of an id that is a string or a number: its JSON text, so `"1"`
/// and `1` stay apart as equal JSON would keep them.
fn key_of(id: &Value) -> Option<String> {
    match id {
        Value::String(_) | Value::Number(_) => Some(id.to_string()),
        _ => None,
    }
}

/// Whether `a` is further along than `b`: more turns, then a later place in
/// the listing.
fn further_along(a: &Value, b: &Value) -> bool {
    let key = |v: &Value| {
        (
            v.get("turn_count").and_then(Value::as_u64).unwrap_or(0),
            v.get("updated_ms").and_then(Value::as_u64).unwrap_or(0),
        )
    };
    key(a) > key(b)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// **Every member's conversations, once each.** A member rejoining lists
    /// one no other member has yet; a conversation two members list is kept
    /// as the copy further along.
    #[test]
    fn lists_merge_by_id_keeping_the_copy_further_along() {
        let in_sync = json!({ "conversations": [
            { "id": "a", "turn_count": 4, "updated_ms": 9 },
            { "id": "b", "turn_count": 2, "updated_ms": 5 },
        ]});
        let rejoining = json!({ "conversations": [
            { "id": "b", "turn_count": 3, "updated_ms": 6 },
            { "id": "c", "turn_count": 1, "updated_ms": 7 },
            { "id": "a", "turn_count": 1, "updated_ms": 1 },
        ]});
        assert_eq!(
            merge_by_id(vec![in_sync, rejoining]),
            json!({ "conversations": [
                { "id": "a", "turn_count": 4, "updated_ms": 9 },
                { "id": "b", "turn_count": 3, "updated_ms": 6 },
                { "id": "c", "turn_count": 1, "updated_ms": 7 },
            ]})
        );
    }

    /// An id twice in one later body is one item, the copy further along; an
    /// id twice in the first body keeps its first; `"1"` and `1` are two ids;
    /// an item with no id, or a null one, is kept as it comes; an object id
    /// is matched as equal JSON.
    #[test]
    fn duplicates_and_items_without_ids_merge_exactly() {
        let first = json!({
            "items": [
                { "id": 1, "turn_count": 1 },
                { "id": 1, "turn_count": 9 },
                { "id": {"k": 2}, "turn_count": 1 },
                { "note": "no id" },
            ],
            "only_first": 5,
        });
        let second = json!({
            "items": [
                { "id": "1", "turn_count": 1 },
                { "id": "x", "turn_count": 1 },
                { "id": "x", "turn_count": 2 },
                { "id": 1, "turn_count": 2 },
                { "id": {"k": 2}, "turn_count": 3 },
                { "id": null, "turn_count": 4 },
                { "note": "no id" },
            ],
            "only_second": [1],
        });
        assert_eq!(
            merge_by_id(vec![first, second]),
            json!({
                "items": [
                    { "id": 1, "turn_count": 2 },
                    { "id": 1, "turn_count": 9 },
                    { "id": {"k": 2}, "turn_count": 3 },
                    { "note": "no id" },
                    { "id": "1", "turn_count": 1 },
                    { "id": "x", "turn_count": 2 },
                    { "id": null, "turn_count": 4 },
                    { "note": "no id" },
                ],
                "only_first": 5,
                "only_second": [1],
            })
        );
    }

    #[test]
    fn no_bodies_merge_to_null() {
        assert_eq!(merge_by_id(Vec::new()), Value::Null);
    }
}
