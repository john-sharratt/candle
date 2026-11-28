//! A deferred block between gaps (`docs/kv_quantize_in_the_gap.md` §3.2,
//! §3.5): where it stands — its layer, sequence, index and valid window — and
//! nothing it holds. Its bands are read again from the block table when it
//! leads the next table, so between gaps the cycle names no slot, and a
//! compaction pass run then has no holder in the cycle to sweep: the block
//! table it relocates through is the one that names the block, and a block
//! whose sequence let it go is gone with it rather than kept alive here.

use candle::Result;

use super::super::backing::ChunkedKvBacking;
use super::install::requeue_at;
use super::launch::GapJob;

/// A deferred block, by where it stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Deferred {
    pub layer: usize,
    pub seq: usize,
    pub block: usize,
    /// The block's live token window, `(offset << 8) | len`, as its job had it.
    pub valid_range: i32,
}

impl Deferred {
    /// Where `job`'s block stands, without its bands.
    pub fn of(job: &GapJob) -> Self {
        Self {
            layer: job.layer,
            seq: job.seq,
            block: job.block,
            valid_range: job.valid_range,
        }
    }

    /// The job that converts it now, with the bands its index holds now;
    /// `None` when that is no longer a native block of its window behind its
    /// sequence's cursor — its sequence freed, truncated or quantized past it
    /// ([`requeue_at`]).
    pub fn job(self, backings: &[&ChunkedKvBacking]) -> Result<Option<GapJob>> {
        Ok(requeue_at(backings, &self)?.map(|bands| GapJob {
            layer: self.layer,
            seq: self.seq,
            block: self.block,
            bands,
            valid_range: self.valid_range,
        }))
    }
}
