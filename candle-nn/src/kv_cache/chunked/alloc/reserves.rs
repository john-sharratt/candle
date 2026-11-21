//! Which size classes keep a reserve of free arena slots, so a card full of
//! native K/V still has somewhere to convert into.
//!
//! Each reserve class keeps one arena's worth of free slots that the empty sweep
//! never releases, and the gap restores any reserve that has fallen short.
//! Formats that share a class share its reserve. The reserve is the guard the
//! between-forward conversion depends on: its first conversion needs a
//! destination to write into.

use super::*;

pub(super) fn push_unique_key(
    keys: &mut Vec<super::super::arena::ArenaKey>,
    key: super::super::arena::ArenaKey,
) {
    if !keys.iter().any(|k| k == &key) {
        keys.push(key);
    }
}

impl ChunkedKvBacking {
    /// Keep a **reserve** for the baseline formats and every quant candidate
    /// the policy can produce (`docs/kv_quantize_in_the_gap.md` §3.2): one
    /// arena's worth of free slots per class, which the empty sweep never
    /// releases, created here and restored between forwards
    /// ([`BackingInner::create_deferred_arenas`]). It is the guard against a
    /// card full of native K/V: the gap's first conversion finds a
    /// destination, and every block it installs gives back more native ground
    /// than it took. Past it the gap's lease buys what a table needs.
    pub(in super::super) fn keep_reserves(
        &self,
        compression: Option<&super::super::CompressionPolicy>,
    ) -> Result<()> {
        let location = self.inner.storage.default_location();
        let mut keys = Vec::new();
        let push = |keys: &mut Vec<ArenaKey>, fmt: KvFormat| -> Result<()> {
            push_unique_key(keys, self.inner.arena_key_for(fmt, location)?);
            Ok(())
        };

        // Baseline warm set requested for runtime stability: F16 and R16.
        push(&mut keys, KvFormat::Float(DType::F16))?;
        push(&mut keys, KvFormat::Quantized(QuantFormat::R16))?;

        // Include the backing's default target formats.
        push(&mut keys, self.inner.storage.k_format())?;
        push(&mut keys, self.inner.storage.v_format())?;

        // Include quantized candidates used by the shared adaptive profile.
        //
        // Under size classes several of these collapse onto one key — the whole
        // sub-320 B tail is a single class — so the warm set is a handful of
        // slabs rather than one per candidate format. That collapse is the
        // point: a slot freed by any of them is allocatable by all of them.
        if let Some(compression) = compression {
            let (k_candidates, v_candidates) =
                super::super::compression_policy::production_adaptive_candidates(
                    compression.compression_level,
                );
            for fmt in k_candidates.iter().chain(v_candidates.iter()) {
                if matches!(fmt, KvFormat::Quantized(_)) {
                    push(&mut keys, *fmt)?;
                }
            }
        }

        // Every quantized chunk the migrate makes takes a record slot too, and
        // records are device-only.
        if location == ArenaLocation::Gpu {
            push_unique_key(&mut keys, self.record_layout()?.key);
        }

        for key in keys {
            self.inner.pool.keep_reserve(key);
            let arena_idx = self.inner.pool.register_arena(key);
            self.ensure_arena_exists(arena_idx, key)?;
            self.inner.pool.finish_creation(key, arena_idx);
        }
        Ok(())
    }

    /// Ensure the backing can hold at least `required_max_blocks` blocks per sequence.
    pub(in super::super) fn ensure_max_blocks(&self, required_max_blocks: usize) -> Result<()> {
        if required_max_blocks <= 1 {
            return Ok(());
        }
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        if required_max_blocks <= state.max_blocks {
            return Ok(());
        }

        let mut new_max_blocks = state.max_blocks;
        while new_max_blocks < required_max_blocks {
            new_max_blocks = cmp::max(required_max_blocks, new_max_blocks.saturating_mul(2));
            if new_max_blocks == 0 {
                new_max_blocks = required_max_blocks;
                break;
            }
        }

        state.max_blocks = new_max_blocks;

        Ok(())
    }

    pub(in super::super) fn create_arena(&self, key: ArenaKey, index: usize) -> Result<Arena> {
        self.inner.create_arena(key, index)
    }
}
