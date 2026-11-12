//! Bringing turns from the log into the warm tier without placing them hot
//! (`docs/kv_quantize_in_the_gap.md` §5).
//!
//! The hit-driven pool (`docs/hit_driven_projection.md` §7.2) keeps far more
//! of a dialogue's candidates than the slot holds, and a round the slot later
//! chooses should lift at the link's speed, not the disk's. So the pool asks
//! for its pieces' turns to be warmed: each one that holds only a cold copy is
//! read from the log and rebuilt in host RAM, one layer at a time, and
//! installed as the residence's warm copy. Nothing touches the device.
//!
//! Best-effort by design: a turn already warm or hot is skipped, a turn the
//! log cannot read is passed over with a warning, and once `room` bytes are
//! spent the rest of the list is dropped. The caller asks again later.

use candle_nn::kv_cache::{ChunkedKvBacking, ColdChunk, SealedSequence};

use super::ids::TurnKey;
use crate::persistence::content_hash::turn_stream_id;
use crate::persistence::resume::{recover_turn_grid, ChunkImage};
use crate::persistence::streams::{StreamDecl, TurnDecl};
use crate::projection::Conversation;

/// What one warming pass did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Warmed {
    /// Turns brought from cold into warm.
    pub turns: usize,
    /// Their warm bytes.
    pub bytes: u64,
    /// Turns already warm or hot, or not cold-backed: nothing to do.
    pub skipped: usize,
    /// Turns left for want of room.
    pub dropped: usize,
    /// Turns the log could not give back.
    pub failed: usize,
}

/// A chunk image as the warm rebuild reads it, borrowed.
fn cold_view(image: &ChunkImage) -> ColdChunk<'_> {
    let p = &image.payload;
    ColdChunk {
        offset: p.offset,
        token_count: image.token_count,
        k_fmt: &p.k_formats,
        v_fmt: &p.v_formats,
        k_pal: &p.k_pal,
        v_pal: &p.v_pal,
        k_scale: &p.k_scale,
        v_scale: &p.v_scale,
        kv_bytes: &p.kv_bytes,
    }
}

impl Conversation {
    /// Warm every turn of `turns` that holds only a cold copy, spending at
    /// most `room` warm bytes, in the order given. `backings` are the engine's
    /// per-layer backings, in layer order. See the module docs.
    pub fn warm_turns_from_cold(
        &self,
        backings: &[ChunkedKvBacking],
        turns: &[TurnKey],
        room: u64,
    ) -> Warmed {
        let mut out = Warmed::default();
        let n_layers = backings.len();
        let mut spent = 0u64;
        for (i, &key) in turns.iter().enumerate() {
            let substrate = self.read();
            let cold_only = substrate
                .turn_tier_state(key.timeline, key.index)
                .is_some_and(|t| t.cold && !t.warm && !t.hot);
            if !cold_only {
                out.skipped += 1;
                continue;
            }
            let Some(residence) = substrate.turn_residence(key.timeline, key.index) else {
                out.skipped += 1;
                continue;
            };
            // The turn's decl by its stream id, as the cold-to-hot load finds it:
            // one map lookup, never a walk of every stream.
            let Some(stream) = substrate.stream_of(turn_stream_id(key.timeline.raw(), key.index.0))
            else {
                out.failed += 1;
                continue;
            };
            let Some(StreamDecl::Turn(decl)) = &stream.decl else {
                out.failed += 1;
                continue;
            };
            let decl = TurnDecl {
                turn_index: key.index.0,
                ..decl.clone()
            };
            let need: u64 = stream.chunks.values().map(|loc| loc.payload_len).sum();
            if spent + need > room {
                out.dropped += turns.len() - i;
                break;
            }
            let grid = match self
                .persistence
                .call(|p| recover_turn_grid(p, &substrate, &decl, n_layers))
            {
                Ok(grid) if grid.n_layers() == n_layers && !grid.is_empty() => grid,
                Ok(_) => {
                    tracing::warn!(
                        target: "candle_conversation::persistence::tier",
                        ?key,
                        "cold → warm: the log holds no full chunk grid for this turn"
                    );
                    out.failed += 1;
                    continue;
                }
                Err(e) => {
                    tracing::warn!(
                        target: "candle_conversation::persistence::tier",
                        ?key,
                        "cold → warm: reading the turn from the log failed: {e}"
                    );
                    out.failed += 1;
                    continue;
                }
            };
            let warm: candle::Result<Vec<SealedSequence>> = backings
                .iter()
                .zip(grid.iter_layers())
                .map(|(backing, images)| {
                    let views: Vec<ColdChunk<'_>> = images.iter().map(cold_view).collect();
                    backing.warm_from_cold(&views)
                })
                .collect();
            let warm = match warm {
                Ok(warm) => warm,
                Err(e) => {
                    tracing::warn!(
                        target: "candle_conversation::persistence::tier",
                        ?key,
                        "cold → warm: rebuilding the turn in host RAM failed: {e}"
                    );
                    out.failed += 1;
                    continue;
                }
            };
            let bytes: u64 = warm
                .iter()
                .flat_map(|s| s.chunks.iter())
                .map(|c| c.byte_size)
                .sum();
            substrate.install_warm(residence, warm);
            spent += bytes;
            out.turns += 1;
            out.bytes += bytes;
        }
        out
    }
}
