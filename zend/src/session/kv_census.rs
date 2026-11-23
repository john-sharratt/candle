//! The K/V census of this member's substrate, read while it serves
//! (`candle_conversation::substrate::kv_census`).

use candle_conversation::substrate::kv_census::KvCensus;

use super::ZendSession;

impl ZendSession {
    /// The census of every live chunk this member's substrate holds, with a
    /// head read for about `sample` of them under `seed`. Reads disk for the
    /// sample: call it off the async runtime. `None` while the model is not
    /// loaded.
    pub fn kv_census(&self, sample: usize, seed: u64) -> Option<KvCensus> {
        let state = self.inference.load_full()?;
        let census = state.engine.conversation().read().kv_census(sample, seed);
        Some(census)
    }
}
