//! The small durable records `/mcp` keeps — threads, workers, OAuth codes and
//! tokens — as custom objects in zend's substrate, so they replicate and
//! compact like the rest of it (`docs/claude_zend_gateway.md` §3.2, §12).
//!
//! Every key starts with [`PREFIX`]. A record is one JSON value under one key;
//! the latest write wins, and ceasing a key ends it.

use std::collections::BTreeMap;
use std::sync::Mutex;

use candle_conversation::persistence::record::CustomObjectPayload;
use candle_conversation::persistence::SharedSubstrate;
use serde_json::Value;

/// The prefix every `/mcp` record's key starts with.
pub const PREFIX: &str = "mcp/";

/// Where the records live: the substrate in the daemon, a map in the tests.
pub trait Records: Send + Sync {
    /// The live record under `key`.
    fn get(&self, key: &str) -> Option<Value>;
    /// Keep `value` under `key`, replacing what was there.
    fn put(&self, key: &str, value: Value) -> anyhow::Result<()>;
    /// Every live record whose key starts with `prefix`, in key order.
    fn under(&self, prefix: &str) -> Vec<(String, Value)>;
    /// How many live records' keys start with `prefix`.
    fn count_under(&self, prefix: &str) -> usize;
    /// The live record under `prefix` whose key sorts last.
    fn last_under(&self, prefix: &str) -> Option<(String, Value)>;
    /// End the record under `key`.
    fn cease(&self, key: &str) -> anyhow::Result<()>;
}

impl Records for SharedSubstrate {
    fn get(&self, key: &str) -> Option<Value> {
        self.custom_object(key).map(|obj| obj.blob)
    }

    fn put(&self, key: &str, value: Value) -> anyhow::Result<()> {
        self.put_custom_object(CustomObjectPayload::new(key, value))
            .map_err(|e| anyhow::anyhow!("the record `{key}` could not be kept: {e}"))
    }

    fn under(&self, prefix: &str) -> Vec<(String, Value)> {
        self.custom_objects_under(prefix)
            .into_iter()
            .map(|(obj, _)| (obj.key, obj.blob))
            .collect()
    }

    fn count_under(&self, prefix: &str) -> usize {
        self.custom_objects_count_under(prefix)
    }

    fn last_under(&self, prefix: &str) -> Option<(String, Value)> {
        self.custom_object_last_under(prefix)
            .map(|obj| (obj.key, obj.blob))
    }

    fn cease(&self, key: &str) -> anyhow::Result<()> {
        self.put_custom_object(CustomObjectPayload::ceased(key))
            .map_err(|e| anyhow::anyhow!("the record `{key}` could not be ceased: {e}"))
    }
}

/// Records held in memory: what the tests run against.
#[derive(Default)]
pub struct MemRecords(Mutex<BTreeMap<String, Value>>);

impl Records for MemRecords {
    fn get(&self, key: &str) -> Option<Value> {
        self.0.lock().unwrap().get(key).cloned()
    }

    fn put(&self, key: &str, value: Value) -> anyhow::Result<()> {
        self.0.lock().unwrap().insert(key.to_string(), value);
        Ok(())
    }

    fn under(&self, prefix: &str) -> Vec<(String, Value)> {
        self.0
            .lock()
            .unwrap()
            .range(prefix.to_string()..)
            .take_while(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    fn count_under(&self, prefix: &str) -> usize {
        self.0
            .lock()
            .unwrap()
            .range(prefix.to_string()..)
            .take_while(|(k, _)| k.starts_with(prefix))
            .count()
    }

    fn last_under(&self, prefix: &str) -> Option<(String, Value)> {
        self.0
            .lock()
            .unwrap()
            .range(prefix.to_string()..)
            .take_while(|(k, _)| k.starts_with(prefix))
            .last()
            .map(|(k, v)| (k.clone(), v.clone()))
    }

    fn cease(&self, key: &str) -> anyhow::Result<()> {
        self.0.lock().unwrap().remove(key);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_prefix_reads_its_own_keys_in_order_and_a_ceased_key_is_gone() {
        let r = MemRecords::default();
        r.put("mcp/t/b", json!(2)).unwrap();
        r.put("mcp/t/a", json!(1)).unwrap();
        r.put("mcp/u/a", json!(3)).unwrap();
        assert_eq!(
            r.under("mcp/t/"),
            vec![
                ("mcp/t/a".to_string(), json!(1)),
                ("mcp/t/b".to_string(), json!(2))
            ]
        );
        assert_eq!(r.count_under("mcp/t/"), 2);
        assert_eq!(
            r.last_under("mcp/t/"),
            Some(("mcp/t/b".to_string(), json!(2)))
        );
        r.cease("mcp/t/a").unwrap();
        assert_eq!(r.get("mcp/t/a"), None);
        assert_eq!(r.under("mcp/t/").len(), 1);
        assert_eq!(r.count_under("mcp/v/"), 0);
        assert_eq!(r.last_under("mcp/v/"), None);
    }
}
