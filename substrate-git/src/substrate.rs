//! [`Records`] over a daemon's substrate: packs as `Blob` records, entries as
//! keyed custom objects. Both replicate like every other record.

use candle_conversation::persistence::record::CustomObjectPayload;
use candle_conversation::persistence::SharedSubstrate;

use crate::records::Records;

impl Records for SharedSubstrate {
    fn put_blob(&self, bytes: Vec<u8>) -> anyhow::Result<[u8; 32]> {
        Ok(SharedSubstrate::put_blob(self, bytes)?)
    }

    fn blob(&self, sha256: &[u8; 32]) -> anyhow::Result<Option<Vec<u8>>> {
        Ok(SharedSubstrate::blob(self, sha256)?)
    }

    fn put_object(&self, key: &str, value: serde_json::Value) -> anyhow::Result<()> {
        Ok(self.put_custom_object(CustomObjectPayload::new(key, value))?)
    }

    fn objects_under(&self, prefix: &str) -> Vec<(String, serde_json::Value)> {
        self.custom_objects_under(prefix)
            .into_iter()
            .map(|(obj, _)| (obj.key, obj.blob))
            .collect()
    }
}
