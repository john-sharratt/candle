//! What a substrate holds in memory, structure by structure, for the memory
//! census (`docs/memory_census.md` §5): streams and their chunk locations, the
//! records waiting for their timeline, the residence slab, and the custom
//! objects by key prefix.
//!
//! Every figure is a count over the substrate's maps, so a gauge is taken on
//! the persistence thread once a minute and never on a request.

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;

use super::pending::PendingGauge;
use super::residence::ResidenceGauge;
use super::Substrate;

/// The custom objects under one key prefix.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
pub struct PrefixGauge {
    /// The key up to its second `/`: `mcp/note`, `npc/world`.
    pub prefix: String,
    pub live: u64,
    /// Tombstones kept until they expire.
    pub ceased: u64,
    /// Keys and blobs, estimated from the JSON they hold.
    pub bytes: u64,
}

/// What a substrate's in-memory structures hold.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
pub struct SubstrateGauge {
    pub streams: u64,
    pub chunk_locations: u64,
    pub pending: PendingGauge,
    pub residences: ResidenceGauge,
    /// Largest first.
    pub custom_objects: Vec<PrefixGauge>,
}

/// The key up to its second `/`, or the whole key when it has fewer.
pub fn prefix_of(key: &str) -> &str {
    match key.match_indices('/').nth(1) {
        Some((i, _)) => &key[..i],
        None => key,
    }
}

/// The bytes a JSON value holds, estimated without serialising it: strings and
/// keys by length, every scalar as eight bytes.
pub fn value_bytes(v: &Value) -> u64 {
    match v {
        Value::Null | Value::Bool(_) | Value::Number(_) => 8,
        Value::String(s) => s.len() as u64,
        Value::Array(a) => a.iter().map(value_bytes).sum::<u64>() + 8,
        Value::Object(o) => {
            o.iter()
                .map(|(k, v)| k.len() as u64 + value_bytes(v))
                .sum::<u64>()
                + 8
        }
    }
}

impl Substrate {
    /// What this substrate's in-memory structures hold now. O(entries).
    pub fn gauge(&self) -> SubstrateGauge {
        let (mut streams, mut chunk_locations) = (0u64, 0u64);
        for (_, s) in self.streams.iter_physical() {
            streams += 1;
            chunk_locations += s.chunks.len() as u64;
        }
        let mut by_prefix: BTreeMap<&str, PrefixGauge> = BTreeMap::new();
        for obj in self.custom_objects.values() {
            let prefix = prefix_of(&obj.key);
            let g = by_prefix.entry(prefix).or_insert_with(|| PrefixGauge {
                prefix: prefix.to_string(),
                ..PrefixGauge::default()
            });
            if obj.ceased {
                g.ceased += 1;
            } else {
                g.live += 1;
            }
            g.bytes += obj.key.len() as u64 + value_bytes(&obj.blob);
        }
        let mut custom_objects: Vec<PrefixGauge> = by_prefix.into_values().collect();
        custom_objects.sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.prefix.cmp(&b.prefix)));
        SubstrateGauge {
            streams,
            chunk_locations,
            pending: self.pending.gauge(),
            residences: self.residences.gauge(),
            custom_objects,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::persistence::identity::RecordIdentity;
    use crate::persistence::record::CustomObjectPayload;

    #[test]
    fn a_prefix_is_the_key_up_to_its_second_slash() {
        assert_eq!(prefix_of("mcp/note/owner/thread/1"), "mcp/note");
        assert_eq!(prefix_of("mcp/chat-said"), "mcp/chat-said");
        assert_eq!(prefix_of("solo"), "solo");
    }

    #[test]
    fn a_value_is_sized_by_its_strings_and_scalars() {
        assert_eq!(value_bytes(&json!("hello")), 5);
        assert_eq!(value_bytes(&json!(7)), 8);
        assert_eq!(
            value_bytes(&json!({"ab": "xyz", "c": [1, null]})),
            2 + 3 + 1 + (8 + 8 + 8) + 8
        );
    }

    fn id(oseq: u64) -> RecordIdentity {
        RecordIdentity {
            origin: 5,
            oseq,
            gen: 0,
            epoch: 0,
            hlc: oseq << 16,
        }
    }

    /// **Custom objects are counted by prefix, live and ceased, largest
    /// first.**
    #[test]
    fn custom_objects_are_counted_by_prefix() {
        let mut s = Substrate::default();
        s.apply_custom_object(
            CustomObjectPayload::new("mcp/note/o/t/1", json!("checked the build")),
            id(1),
        );
        s.apply_custom_object(
            CustomObjectPayload::new("mcp/note/o/t/2", json!("ran the tests")),
            id(2),
        );
        s.apply_custom_object(
            CustomObjectPayload::ceased("mcp/chat-said/o/c/1".to_string()),
            id(3),
        );
        let g = s.gauge();
        assert_eq!(g.custom_objects.len(), 2);
        let note = &g.custom_objects[0];
        assert_eq!(
            (note.prefix.as_str(), note.live, note.ceased),
            ("mcp/note", 2, 0)
        );
        assert_eq!(note.bytes, (14 + 17) + (14 + 13));
        let said = &g.custom_objects[1];
        assert_eq!(
            (said.prefix.as_str(), said.live, said.ceased),
            ("mcp/chat-said", 0, 1)
        );
    }
}
