//! The K/V census of a live substrate (`docs/unbounded_agents.md` §9.6.9):
//! what the store's K/V is sealed in, and what it would take at BF16, read
//! from the running daemon while it serves.
//!
//! Two readings, one exact and one sampled:
//!
//! - **Every live chunk, from memory.** The substrate's chunk index holds each
//!   chunk record's location, payload length, token count and the format of
//!   its first K sub-band (the record header's representative format). Summed
//!   over every chunk of every stream not tombstoned, these are exact and read
//!   no disk.
//! - **A sample of chunk heads, from disk.** The full per-sub-band formats live
//!   in each payload's metadata prefix ([`decode_chunk_head`]), so a uniform
//!   sample of chunks — each kept by a seeded hash of its stream and index — is
//!   read one head ([`HEAD_BYTES`]) each, and tallied ([`Tally`]). Its
//!   BF16-to-stored ratio scales the exact stored bytes to the BF16 estimate.

mod tally;

pub use tally::{ChunkShape, FormatShare, Tally, BF16_BLOCK_BYTES};

use std::collections::BTreeMap;

use candle_nn::kv_cache::KvFormat;
use serde::Serialize;

use crate::persistence::manifest::RecordLoc;
use crate::persistence::record::decode_chunk_head;
use crate::persistence::streams::StreamDecl;

use super::Substrate;

/// How much of a sampled chunk's payload is read: one sector, which holds
/// the metadata prefix of every geometry the engine seals.
pub const HEAD_BYTES: usize = 4096;

/// The chunks, tokens and bytes of one representative format.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LeadShare {
    pub chunks: u64,
    pub tokens: u64,
    pub bytes: u64,
}

/// The sampled reading.
#[derive(Debug, Default, Clone, PartialEq, Serialize)]
pub struct SampledCensus {
    /// The sample size asked for.
    pub asked: usize,
    /// Heads that could not be read or decoded.
    pub unreadable: u64,
    pub tally: Tally,
}

/// The census.
#[derive(Debug, Default, Clone, PartialEq, Serialize)]
pub struct KvCensus {
    /// Live chunk records.
    pub chunks: u64,
    /// The tokens they hold, every layer's counted.
    pub tokens: u64,
    /// Their payload bytes.
    pub stored_bytes: u64,
    /// By the format of each chunk's first K sub-band.
    pub by_lead_format: BTreeMap<String, LeadShare>,
    pub sample: SampledCensus,
    /// `stored_bytes` at the sample's BF16-to-stored ratio.
    pub bf16_bytes_estimate: u64,
}

/// Whether chunk `index` of `stream` is in the sample that keeps `keep` of
/// every `of` chunks under `seed`: a SplitMix64 finalizer over the three, so
/// the sample is uniform and the same for the same seed.
pub fn sampled(stream: u64, index: u64, seed: u64, keep: u64, of: u64) -> bool {
    if of == 0 || keep >= of {
        return true;
    }
    let mut z = stream ^ index.rotate_left(29) ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    ((z as u128 * of as u128) >> 64) < keep as u128
}

/// A format tag's name, or the tag itself when no `KvFormat` carries it.
fn lead_name(tag: u8) -> String {
    KvFormat::from_tag(tag)
        .map(|f| tally::name(&f))
        .unwrap_or_else(|| format!("tag{tag}"))
}

impl Substrate {
    /// The census of every live chunk, with a head read for about `sample`
    /// of them under `seed`.
    pub fn kv_census(&self, sample: usize, seed: u64) -> KvCensus {
        let live = |decl: &Option<StreamDecl>| match decl {
            Some(StreamDecl::Turn(t)) => !self.is_timeline_dead(t.timeline_id),
            Some(StreamDecl::PromptSection(_)) | None => true,
        };
        let mut c = KvCensus::default();
        for (_, stream) in self.all_streams().filter(|(_, s)| live(&s.decl)) {
            for loc in stream.chunks.values() {
                c.chunks += 1;
                c.tokens += loc.token_count;
                c.stored_bytes += loc.payload_len;
                let lead = c.by_lead_format.entry(lead_name(loc.format)).or_default();
                lead.chunks += 1;
                lead.tokens += loc.token_count;
                lead.bytes += loc.payload_len;
            }
        }
        c.sample.asked = sample;
        for (id, stream) in self.all_streams().filter(|(_, s)| live(&s.decl)) {
            for (&index, loc) in stream.chunks.iter() {
                if !sampled(id.0, index, seed, sample as u64, c.chunks) {
                    continue;
                }
                let record = stream.ids.chunks.get(&index).copied().unwrap_or_default();
                let at = RecordLoc {
                    segment: loc.segment,
                    offset: loc.offset,
                    payload_len: loc.payload_len,
                    record_size: loc.record_size,
                };
                let head = self
                    .payloads()
                    .read_head(record, at, HEAD_BYTES.min(loc.payload_len as usize))
                    .ok()
                    .and_then(|bytes| decode_chunk_head(&bytes).ok());
                match head {
                    Some(h) => c.sample.tally.add(&ChunkShape {
                        tokens: loc.token_count,
                        k_formats: &h.k_formats,
                        v_formats: &h.v_formats,
                        kv_len: h.kv_len as u64,
                        side_len: h.side_len as u64,
                    }),
                    None => c.sample.unreadable += 1,
                }
            }
        }
        c.bf16_bytes_estimate = (c.stored_bytes as f64 * c.sample.tally.ratio()) as u64;
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The sample keeps about its share, the same each time**: 1,000 of
    /// 100,000 chunks kept within a few percent, every one kept when the
    /// sample asks for all.
    #[test]
    fn the_sample_is_uniform_and_seeded() {
        let kept = (0..100_000u64)
            .filter(|&i| sampled(i / 64, i % 64, 7, 1_000, 100_000))
            .count();
        assert!((900..=1_100).contains(&kept), "{kept}");
        let again = (0..100_000u64)
            .filter(|&i| sampled(i / 64, i % 64, 7, 1_000, 100_000))
            .count();
        assert_eq!(kept, again);
        assert!(sampled(1, 2, 3, 10, 10));
        assert!(sampled(1, 2, 3, 5, 0));
    }

    #[test]
    fn an_unknown_tag_is_named_by_its_number() {
        assert_eq!(lead_name(250), "tag250");
    }
}
