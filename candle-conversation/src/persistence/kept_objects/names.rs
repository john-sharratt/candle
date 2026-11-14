//! The blobs an object names: every string in its metadata or its JSON value
//! that spells a SHA-256 in lowercase hex — the spelling every object that
//! names a blob writes it in (`blob::hex`). A string that only looks like one
//! names nothing unless a blob of that name exists, which the export checks.

use serde_json::Value;

use crate::persistence::blob::parse_hex;
use crate::persistence::record::CustomObjectPayload;

/// The SHA-256s `obj` spells, each once, in the order met.
pub fn named_blobs(obj: &CustomObjectPayload) -> Vec<[u8; 32]> {
    let mut out: Vec<[u8; 32]> = Vec::new();
    let mut take = |s: &str| {
        if let Some(h) = lower_hex(s) {
            if !out.contains(&h) {
                out.push(h);
            }
        }
    };
    for v in obj.metadata.values() {
        take(v);
    }
    walk(&obj.blob, &mut take);
    out
}

fn lower_hex(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64
        || !s
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return None;
    }
    parse_hex(s)
}

fn walk(v: &Value, take: &mut impl FnMut(&str)) {
    match v {
        Value::String(s) => take(s),
        Value::Array(a) => a.iter().for_each(|x| walk(x, take)),
        Value::Object(m) => m.values().for_each(|x| walk(x, take)),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A name in a nested value or in the metadata is found, each once; an
    /// uppercase spelling, a short one and a non-hex one are not names.
    #[test]
    fn every_lowercase_sha256_spelled_anywhere_is_named() {
        let a = "a".repeat(64);
        let b = "0123456789abcdef".repeat(4);
        let mut obj = CustomObjectPayload::new(
            "images/x",
            json!({"portrait": {"sha256": a}, "list": [b, a, "A".repeat(64), "abc", "g".repeat(64)]}),
        );
        obj.metadata.insert("pack".into(), b.clone());
        let names = named_blobs(&obj);
        assert_eq!(names.len(), 2);
        assert!(names.contains(&parse_hex(&a).unwrap()));
        assert!(names.contains(&parse_hex(&b).unwrap()));
    }
}
