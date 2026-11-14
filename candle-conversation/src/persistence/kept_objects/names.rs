//! The blobs an object names: every segment of its key, and every string in
//! its metadata or its JSON value, that spells a SHA-256 in lowercase hex —
//! the spelling every object that names a blob writes it in (`blob::hex`). A
//! string that only looks like one names nothing unless a blob of that name
//! exists, which the export checks.
//!
//! **The key names a blob too.** A repository's pack record is keyed
//! `git/pack/<repo>/<sha256>` and its value names only the pack's index; the
//! pack itself is named by the key alone. Read from the value only, an export
//! carried every pack's index and none of its packs — BlackWorld's, 3,659 of
//! them (2026-11-14) — and the substrate it was imported into listed packs no
//! member held, so its repositories never finished loading.

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
    for segment in obj.key.split('/') {
        take(segment);
    }
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

    /// A pack record names its pack by its key and its index by its value:
    /// both are named, the pack first.
    #[test]
    fn a_pack_record_names_the_pack_its_key_spells() {
        let pack = "6607902263ab8793e9074eae07ec8e7900e8099ea4ca2100a20a6c47ee5dfdc9";
        let index = "9c36793efe47853a4807ff842f472fc1ac2302ef735531574fd3354e7bd73d39";
        let obj = CustomObjectPayload::new(
            format!("git/pack/candle/{pack}"),
            json!({"checksum": "e5f8b2f459ff3ed17d851aab1d36faff69f9fae6", "objects": 1480, "index": index}),
        );
        assert_eq!(
            named_blobs(&obj),
            vec![parse_hex(pack).unwrap(), parse_hex(index).unwrap()]
        );
    }
}
