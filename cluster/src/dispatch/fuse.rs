//! A cluster's view of a performance answer, fused from every member's
//! (`docs/performance_series.md` §3). Each member says in its answer's `fuse`
//! block how its fields add up, and the gateway's envelope (§5.5 of
//! `docs/replicated_substrate.md`) carries the fused answer beside the
//! members' as `cluster` — so a console draws the cluster as it draws one
//! member, and never adds anything up itself.
//!
//! ```json
//! "fuse": { "series": { "waves": ["ws", "wave_ms"] }, "sum": ["host"], "union": ["mix_formats"] }
//! ```
//!
//! - **`series`**: arrays of rows on the members' shared wall-clock grid, each
//!   row stamped `at` (unix ms, its slot's start). Rows with the same `at` and
//!   the same string fields — a phase's `kind` — are one row of the cluster's.
//!   Numbers add, but those the series names take the largest: a duration the
//!   cards spent side by side lasted as long as the longest of them. A `null`
//!   is a reading that row lacks, and adds nothing.
//! - **`sum`**: objects whose numbers add; a reading one member lacks (`null`)
//!   makes the cluster's absent rather than low.
//! - **`union`**: string arrays, every member's entries in first-seen order.
//!
//! Only the members whose answer declares `fuse` are fused, by the first such
//! answer's declaration; the fused body carries the declared fields alone.

use std::collections::BTreeMap;

use serde_json::{Map, Number, Value};

/// The field a member's answer declares its fusion in.
pub const FUSE_FIELD: &str = "fuse";

/// `bodies` — the members' answers — as the cluster's one, or `None` when no
/// answer declares how it fuses.
pub fn fuse(bodies: &[Value]) -> Option<Value> {
    let spec = bodies.iter().find_map(|b| b.get(FUSE_FIELD))?;
    let declaring: Vec<&Value> = bodies
        .iter()
        .filter(|b| b.get(FUSE_FIELD).is_some())
        .collect();
    let mut out = Map::new();
    if let Some(series) = spec.get("series").and_then(Value::as_object) {
        for (field, longest) in series {
            let longest = strings(Some(longest));
            let rows = declaring
                .iter()
                .filter_map(|b| b.get(field).and_then(Value::as_array))
                .flatten();
            out.insert(field.clone(), Value::Array(fuse_rows(rows, &longest)));
        }
    }
    for field in strings(spec.get("sum")) {
        let mut into = Map::new();
        for b in &declaring {
            if let Some(Value::Object(from)) = b.get(field) {
                sum_into(&mut into, from);
            }
        }
        out.insert(field.to_owned(), Value::Object(into));
    }
    for field in strings(spec.get("union")) {
        let mut all: Vec<Value> = Vec::new();
        for b in &declaring {
            for v in b.get(field).and_then(Value::as_array).into_iter().flatten() {
                if !all.contains(v) {
                    all.push(v.clone());
                }
            }
        }
        out.insert(field.to_owned(), Value::Array(all));
    }
    Some(Value::Object(out))
}

/// The strings of a JSON array, or none.
fn strings(v: Option<&Value>) -> Vec<&str> {
    v.and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}

/// One row per slot and string fields, oldest first.
fn fuse_rows<'a>(rows: impl Iterator<Item = &'a Value>, longest: &[&str]) -> Vec<Value> {
    let mut slots: BTreeMap<(u64, String), Map<String, Value>> = BTreeMap::new();
    for row in rows {
        let Value::Object(row) = row else {
            continue;
        };
        let at = row.get("at").and_then(Value::as_u64).unwrap_or(0);
        let named: Vec<String> = row
            .iter()
            .filter_map(|(k, v)| v.as_str().map(|s| format!("{k}={s}")))
            .collect();
        let into = slots.entry((at, named.join("\u{1f}"))).or_default();
        fold_into(into, row, longest);
    }
    slots.into_values().map(Value::Object).collect()
}

/// `from` added into `into`: numbers summed, or the larger for a field named
/// in `longest`; objects field by field; `null` only where nothing is known;
/// anything else as first seen. `at` is the row's slot, not a quantity.
fn fold_into(into: &mut Map<String, Value>, from: &Map<String, Value>, longest: &[&str]) {
    for (k, v) in from {
        match v {
            Value::Number(n) if k != "at" => {
                let folded = match into.get(k) {
                    Some(Value::Number(have)) => combine(have, n, longest.contains(&k.as_str())),
                    _ => Value::Number(n.clone()),
                };
                into.insert(k.clone(), folded);
            }
            Value::Object(o) => {
                let entry = into
                    .entry(k.clone())
                    .or_insert_with(|| Value::Object(Map::new()));
                if let Value::Object(e) = entry {
                    fold_into(e, o, longest);
                }
            }
            _ => {
                into.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
    }
}

/// `from` added into `into`, where a reading one side lacks (`null`) leaves
/// the sum absent.
fn sum_into(into: &mut Map<String, Value>, from: &Map<String, Value>) {
    for (k, v) in from {
        match (into.get_mut(k), v) {
            (Some(Value::Null), _) => {}
            (_, Value::Null) => {
                into.insert(k.clone(), Value::Null);
            }
            (Some(Value::Number(have)), Value::Number(n)) => {
                let sum = combine(have, n, false);
                into.insert(k.clone(), sum);
            }
            (Some(Value::Object(have)), Value::Object(o)) => sum_into(have, o),
            (None, v) => {
                into.insert(k.clone(), v.clone());
            }
            _ => {}
        }
    }
}

/// Two numbers as one: their sum, or the larger. Whole numbers stay whole.
fn combine(a: &Number, b: &Number, larger: bool) -> Value {
    if let (Some(x), Some(y)) = (a.as_u64(), b.as_u64()) {
        return Value::from(if larger {
            x.max(y)
        } else {
            x.saturating_add(y)
        });
    }
    let (x, y) = (a.as_f64().unwrap_or(0.0), b.as_f64().unwrap_or(0.0));
    Number::from_f64(if larger { x.max(y) } else { x + y }).map_or(Value::Null, Value::Number)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// **Two members' rows in one slot are one row of the cluster's**: tokens
    /// and every VRAM band add, the wave's wall-clock is the longer, a slot
    /// one member ran in is its own, and the rows come oldest first.
    #[test]
    fn series_rows_fuse_by_slot() {
        let a = json!({
            "fuse": { "series": { "waves": ["ws", "wave_ms"] } },
            "waves": [
                { "at": 10000, "tok": 300, "ws": 5000, "wave_ms": 40.5, "vram": { "kv": 100, "free": 50 } },
                { "at": 15000, "tok": 10, "ws": 4000, "wave_ms": null, "vram": { "kv": 1, "free": 1 } }
            ]
        });
        let b = json!({
            "fuse": { "series": { "waves": ["ws", "wave_ms"] } },
            "waves": [
                { "at": 10000, "tok": 700, "ws": 4500, "wave_ms": 52.0, "vram": { "kv": 200, "free": 25 } },
                { "at": 5000, "tok": 1, "ws": 100, "wave_ms": 3.0, "vram": { "kv": 2, "free": 2 } }
            ]
        });
        assert_eq!(
            fuse(&[a, b]).unwrap(),
            json!({ "waves": [
                { "at": 5000, "tok": 1, "ws": 100, "wave_ms": 3.0, "vram": { "kv": 2, "free": 2 } },
                { "at": 10000, "tok": 1000, "ws": 5000, "wave_ms": 52.0, "vram": { "kv": 300, "free": 75 } },
                { "at": 15000, "tok": 10, "ws": 4000, "wave_ms": null, "vram": { "kv": 1, "free": 1 } }
            ]})
        );
    }

    /// **A phase's `kind` keeps its own row**: decode and prefill in one slot
    /// stay apart, and each kind adds across members — the window and the
    /// phase's time the longest member's, its peak rate their sum.
    #[test]
    fn rows_with_different_kinds_stay_apart() {
        let spec = json!({ "series": { "points": ["dur", "win"] } });
        let a = json!({ "fuse": spec, "points": [
            { "at": 5000, "kind": "decode", "dur": 3000, "win": 4800, "tok": 500, "peak": 300.0 },
            { "at": 5000, "kind": "prefill", "dur": 1000, "win": 4800, "tok": 9000, "peak": 2500.0 }
        ]});
        let b = json!({ "fuse": spec, "points": [
            { "at": 5000, "kind": "decode", "dur": 4000, "win": 4900, "tok": 700, "peak": 450.5 }
        ]});
        assert_eq!(
            fuse(&[a, b]).unwrap(),
            json!({ "points": [
                { "at": 5000, "kind": "decode", "dur": 4000, "win": 4900, "tok": 1200, "peak": 750.5 },
                { "at": 5000, "kind": "prefill", "dur": 1000, "win": 4800, "tok": 9000, "peak": 2500.0 }
            ]})
        );
    }

    /// Host memory adds up, and a reading one member lacks stays absent rather
    /// than reading low; the arena formats are every member's, once each.
    #[test]
    fn sums_poison_on_an_absent_reading_and_unions_keep_first_seen_order() {
        let spec = json!({ "sum": ["host"], "union": ["mix_formats"] });
        let a = json!({ "fuse": spec, "host": { "total_mib": 64000, "free_mib": 20000, "swap_mib": 4 },
                        "mix_formats": ["c4", "c6"] });
        let b = json!({ "fuse": spec, "host": { "total_mib": 32000, "free_mib": null, "swap_mib": 2 },
                        "mix_formats": ["c6", "c8"] });
        assert_eq!(
            fuse(&[a, b]).unwrap(),
            json!({ "host": { "total_mib": 96000, "free_mib": null, "swap_mib": 6 },
                    "mix_formats": ["c4", "c6", "c8"] })
        );
    }

    /// An answer that declares nothing is not fused, and with none declaring
    /// there is no cluster body at all.
    #[test]
    fn only_declaring_answers_fuse() {
        let a = json!({ "fuse": { "series": { "waves": [] } }, "waves": [{ "at": 0, "tok": 5 }] });
        let old = json!({ "waves": [{ "at": 0, "tok": 99 }] });
        assert_eq!(
            fuse(&[old.clone(), a]).unwrap(),
            json!({ "waves": [{ "at": 0, "tok": 5 }] })
        );
        assert_eq!(fuse(&[old]), None);
    }
}
