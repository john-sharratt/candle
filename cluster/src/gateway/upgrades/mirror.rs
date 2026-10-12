//! Comparing a mirrored call (§3.8): the baseline answered the client, and the
//! candidate answered a copy of the same request.
//!
//! - `Exact` — the statuses agree and the JSON bodies are equal once every
//!   field the route names as volatile is dropped, at any depth. A body that is
//!   not JSON is compared byte for byte.
//! - `Shape` — the statuses agree and the bodies have the same structure and
//!   types: the same keys, each holding the same kind of value; an array's
//!   shape is the set of its elements' shapes. Values themselves may differ.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

use crate::dispatch::Mirror;

/// One side of a mirrored call.
#[derive(Debug, Clone, Copy)]
pub struct Answer<'a> {
    pub status: u16,
    pub body: &'a [u8],
}

/// How two answers compared: equal, or where they first differ.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Compared {
    Same,
    Differs(String),
}

/// Compare under `mode`; a route declaring `None` is never mirrored.
pub fn compare(mode: &Mirror, baseline: Answer<'_>, candidate: Answer<'_>) -> Compared {
    if baseline.status != candidate.status {
        return Compared::Differs(format!(
            "status {} against {}",
            baseline.status, candidate.status
        ));
    }
    let json = |b: &[u8]| serde_json::from_slice::<Value>(b).ok();
    match mode {
        Mirror::None => Compared::Same,
        Mirror::Exact(volatile) => match (json(baseline.body), json(candidate.body)) {
            (Some(a), Some(b)) => {
                let (a, b) = (without(&a, volatile), without(&b, volatile));
                first_difference(&a, &b, "").map_or(Compared::Same, Compared::Differs)
            }
            _ if baseline.body == candidate.body => Compared::Same,
            _ => Compared::Differs("the bodies differ".into()),
        },
        Mirror::Shape => match (json(baseline.body), json(candidate.body)) {
            (Some(a), Some(b)) if shape(&a) == shape(&b) => Compared::Same,
            (Some(a), Some(b)) => {
                Compared::Differs(format!("shape {} against {}", shape(&a), shape(&b)))
            }
            (None, None) => Compared::Same,
            _ => Compared::Differs("one body is JSON and the other is not".into()),
        },
    }
}

/// `v` without any object field named in `volatile`, at any depth.
fn without(v: &Value, volatile: &[String]) -> Value {
    match v {
        Value::Object(m) => Value::Object(
            m.iter()
                .filter(|(k, _)| !volatile.iter().any(|x| x == *k))
                .map(|(k, v)| (k.clone(), without(v, volatile)))
                .collect::<Map<String, Value>>(),
        ),
        Value::Array(a) => Value::Array(a.iter().map(|v| without(v, volatile)).collect()),
        other => other.clone(),
    }
}

/// Where `a` and `b` first differ, as a JSON path, or `None` when equal.
fn first_difference(a: &Value, b: &Value, at: &str) -> Option<String> {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let keys: BTreeSet<&String> = x.keys().chain(y.keys()).collect();
            keys.into_iter().find_map(|k| match (x.get(k), y.get(k)) {
                (Some(p), Some(q)) => first_difference(p, q, &format!("{at}.{k}")),
                _ => Some(format!("{at}.{k} is on one side only")),
            })
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => x
            .iter()
            .zip(y)
            .enumerate()
            .find_map(|(i, (p, q))| first_difference(p, q, &format!("{at}[{i}]"))),
        (Value::Array(x), Value::Array(y)) => Some(format!(
            "{} has {} items against {}",
            if at.is_empty() { "the body" } else { at },
            x.len(),
            y.len()
        )),
        _ if a == b => None,
        _ => Some(format!(
            "{} is {a} against {b}",
            if at.is_empty() { "the body" } else { at }
        )),
    }
}

/// The structure and types of `v`, as a value of its own.
pub fn shape(v: &Value) -> Value {
    match v {
        Value::Null => Value::from("null"),
        Value::Bool(_) => Value::from("bool"),
        Value::Number(_) => Value::from("number"),
        Value::String(_) => Value::from("string"),
        Value::Array(a) => {
            let items: BTreeSet<String> = a.iter().map(|v| shape(v).to_string()).collect();
            Value::Array(
                items
                    .into_iter()
                    .map(|s| serde_json::from_str(&s).expect("a shape is JSON"))
                    .collect(),
            )
        }
        Value::Object(m) => Value::Object(m.iter().map(|(k, v)| (k.clone(), shape(v))).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(body: &str) -> Answer<'_> {
        Answer {
            status: 200,
            body: body.as_bytes(),
        }
    }

    /// `Exact` ignores the volatile fields at any depth and catches any other
    /// difference, naming where.
    #[test]
    fn exact_ignores_declared_volatile_fields_only() {
        let mode = Mirror::Exact(vec!["at_ms".into(), "id".into()]);
        assert_eq!(
            compare(
                &mode,
                ok(r#"{"items":[{"id":1,"name":"a","at_ms":5}],"at_ms":9}"#),
                ok(r#"{"items":[{"id":2,"name":"a","at_ms":6}],"at_ms":10}"#),
            ),
            Compared::Same
        );
        assert_eq!(
            compare(
                &mode,
                ok(r#"{"items":[{"id":1,"name":"a"}]}"#),
                ok(r#"{"items":[{"id":1,"name":"b"}]}"#),
            ),
            Compared::Differs(r#".items[0].name is "a" against "b""#.into())
        );
        assert_eq!(
            compare(&mode, ok(r#"{"a":1}"#), ok(r#"{"a":1,"b":2}"#)),
            Compared::Differs(".b is on one side only".into())
        );
        assert_eq!(compare(&mode, ok("plain"), ok("plain")), Compared::Same);
    }

    /// `Shape` catches a changed structure, not a changed value.
    #[test]
    fn shape_catches_structure_not_values() {
        assert_eq!(
            compare(
                &Mirror::Shape,
                ok(r#"{"load":0.5,"members":[{"node":"a"},{"node":"b"}]}"#),
                ok(r#"{"load":0.9,"members":[{"node":"c"}]}"#),
            ),
            Compared::Same
        );
        assert!(matches!(
            compare(
                &Mirror::Shape,
                ok(r#"{"load":0.5}"#),
                ok(r#"{"load":"high"}"#)
            ),
            Compared::Differs(_)
        ));
        assert!(matches!(
            compare(&Mirror::Shape, ok(r#"{"load":0.5}"#), ok(r#"{"lode":0.5}"#)),
            Compared::Differs(_)
        ));
    }

    #[test]
    fn a_different_status_differs_in_every_mode() {
        let failed = Answer {
            status: 500,
            body: b"{}",
        };
        assert_eq!(
            compare(&Mirror::Shape, ok("{}"), failed),
            Compared::Differs("status 200 against 500".into())
        );
    }
}
