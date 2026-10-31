//! Keyed custom objects read by prefix: a range of the key index, in key
//! order, the live ones only.

use serde_json::json;

use super::Substrate;
use crate::persistence::identity::RecordIdentity;
use crate::persistence::record::CustomObjectPayload;
use crate::persistence::tombstone_expiry::Expired;

fn stamp(hlc: u64) -> RecordIdentity {
    RecordIdentity {
        origin: 7,
        oseq: hlc,
        gen: 0,
        epoch: 0,
        hlc,
    }
}

fn keys(s: &Substrate, prefix: &str) -> Vec<String> {
    s.custom_objects_under(prefix)
        .into_iter()
        .map(|(obj, _)| obj.key)
        .collect()
}

/// **A prefix reads exactly its own keys, in key order**: not a neighbour
/// that sorts beside it, nor one ceased, and a rewrite is one key, not two.
#[test]
fn a_prefix_reads_its_own_live_keys_in_order() {
    let mut s = Substrate::new();
    for (i, key) in [
        "mcp/t/b", "mcp/t/a", "mcp/u/a", "mcp/t", "mcp/t/c", "mcp/s/z",
    ]
    .into_iter()
    .enumerate()
    {
        s.apply_custom_object(CustomObjectPayload::new(key, json!(i)), stamp(i as u64 + 1));
    }
    s.apply_custom_object(
        CustomObjectPayload::new("mcp/t/a", json!("again")),
        stamp(10),
    );
    s.apply_custom_object(CustomObjectPayload::ceased("mcp/t/c"), stamp(11));
    assert_eq!(keys(&s, "mcp/t/"), ["mcp/t/a", "mcp/t/b"]);
    assert_eq!(s.custom_objects_under("mcp/t/")[0].0.blob, json!("again"));
    assert_eq!(keys(&s, "mcp/t"), ["mcp/t", "mcp/t/a", "mcp/t/b"]);
    assert_eq!(
        keys(&s, "mcp/"),
        ["mcp/s/z", "mcp/t", "mcp/t/a", "mcp/t/b", "mcp/u/a"]
    );
    assert!(keys(&s, "npc/").is_empty());
    assert_eq!(s.custom_objects_count_under("mcp/t/"), 2);
    assert_eq!(s.custom_objects_count_under("npc/"), 0);
    assert_eq!(
        s.custom_object_last_under("mcp/t/").map(|o| o.key),
        Some("mcp/t/b".to_string()),
        "the ceased mcp/t/c sorts after it and is passed over"
    );
    assert_eq!(s.custom_object_last_under("npc/"), None);
}

/// **A ceased key a compaction let go leaves the index**, and a later write
/// of it is read again.
#[test]
fn a_forgotten_key_leaves_the_index_and_returns_when_written() {
    let mut s = Substrate::new();
    s.apply_custom_object(CustomObjectPayload::new("mcp/t/a", json!(1)), stamp(1));
    s.apply_custom_object(CustomObjectPayload::ceased("mcp/t/a"), stamp(2));
    let mut expired = Expired::default();
    expired
        .keys
        .insert(CustomObjectPayload::stream_id_for("mcp/t/a"));
    s.forget_expired_tombstones(&expired);
    assert!(s.custom_object_keys.is_empty());
    s.apply_custom_object(CustomObjectPayload::new("mcp/t/a", json!(3)), stamp(3));
    assert_eq!(keys(&s, "mcp/t/"), ["mcp/t/a"]);
}
