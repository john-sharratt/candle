//! Relocation, driven from the **owner** of the chunks rather than from the
//! arena holding them.
//!
//! # Why the loop runs this way round
//!
//! The pass this replaces started at an arena and had to discover everyone
//! pointing into it. That discovery is the whole problem: a chunk's gid is its
//! physical location, so moving bytes changes identity, and every holder of that
//! identity has to be found and rewritten. There is no index from a gid back to
//! its holders, so the old pass walked block tables and proved completeness by
//! comparing a deduplicated count against a refcount — a proof that quietly
//! fails for any holder that shares an `Arc` rather than cloning a gid, because
//! such a holder adds nothing to the count it is checked against.
//!
//! Sealing and hot→warm migration move KV between arenas thousands of times per
//! run and have never corrupted anything, because they run the other way round:
//! `quantize_sealed_in_place` and `migrate_sealed_to_cpu_batch_async` both take
//! `&[SealedSequence]` and **return replacements**. They never edit a holder, so
//! they never need to find one. This module borrows that shape exactly and
//! spends it on a plain device copy instead of a quantize or a PCIe transfer.
//!
//! # What that buys
//!
//! - **No alias hunt.** The caller owns the sequences it hands in and installs
//!   what comes back. A chunk shared with some other holder simply is not
//!   relocated for that holder, which costs reclaim and nothing else: the old
//!   chunk stays allocated, its bytes untouched, its refcount honestly held.
//! - **No record rewritten in place.** Every relocated chunk gets a *fresh*
//!   [`MetaGid`] at a fresh device address, the way sealing does. A reader sees
//!   either the old chunk — old record, old slot, both still live — or the new
//!   one. There is no intermediate state, so there is no window to lose a race
//!   in.
//! - **No early free.** The old gids die when the caller drops the sequences
//!   they came from, which is after it has installed the replacements.
//!
//! # What it does not do
//!
//! It does not promise to empty any particular arena. Emptiness is emergent:
//! chunks move down the span, and an arena reclaims when its last owner has
//! moved. That is a deliberate trade against the old pass, which refused a donor
//! it could not take *whole* and so let one unreachable chunk waste the arena.
//!
//! Placement is not decided here either. [`relocate_plan`](super::relocate_plan)
//! says which arenas to drain; the pool's own allocator picks the destination,
//! and it already iterates arenas in span-address order so an ordinary claim is
//! the lowest-fitting one.

use std::collections::HashMap;

use candle::Device;

use super::backing::ChunkedKvBacking;
use super::head_gids::HeadGids;
use super::meta_pool::ChunkRecordSrc;
use super::relocate_plan::RelocationPlan;
use super::types::{SealedChunk, SealedSequence};
use crate::kv_cache::ArenaLocation;

/// What one relocation call did, for the caller's log line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RelocationOutcome {
    /// Bands physically copied into a lower arena.
    pub bands_moved: usize,
    /// Chunks that had at least one band move, and so were rebuilt.
    pub chunks_rebuilt: usize,
}

impl RelocationOutcome {
    /// Nothing moved, so the caller should keep the sequences it already has
    /// rather than install an identical copy.
    pub fn is_empty(&self) -> bool {
        self.bands_moved == 0
    }
}

impl ChunkedKvBacking {
    /// Relocate whatever of `sequences` sits in an arena the plans want drained,
    /// returning replacements.
    ///
    /// `plans` is keyed by slot stride, because a band only fits a slot of its
    /// own size class and each class is planned separately. Room is spent as
    /// claims are made, so one set of plans serves a whole pass across many
    /// owners and cannot over-commit the space below the frontier.
    ///
    /// Answers `None` when nothing moved — the caller then keeps what it has,
    /// which avoids installing an identical sequence and invalidating a device
    /// block table for no reason.
    ///
    /// Between forwards only: it takes the same arena window as arena creation,
    /// which refuses while a forward owns the partition.
    /// Take the arena window for a whole relocation pass.
    ///
    /// **Once per pass, not once per call.** The window refuses while a forward
    /// owns the partition, and [`Self::relocate_sealed`] is invoked per residence
    /// *per layer* — 48 layers × 32 residences is ~1,536 acquisitions for one
    /// pass. Measured on run 58: 199,888 `wave in flight` refusals against 788
    /// passes that got through, a 250:1 ratio, because any forward starting
    /// mid-pass killed every remaining call. Holding one window for the pass
    /// makes the decision once and gives the whole pass the same answer.
    #[cfg(feature = "cuda")]
    pub fn begin_relocation_pass(&self) -> candle::Result<Option<super::bump_arena::ArenaWindow>> {
        use super::bump_arena::enter_arena_window;

        let Device::Cuda(cuda) = self.device() else {
            return Ok(None);
        };
        let window = enter_arena_window(&cuda.cuda_stream())?;
        // **Drain every stream before any ground moves.**
        //
        // The window stops a *new* forward from starting; it says nothing about
        // work already in flight. Relocation copies bands to new slots and then
        // frees the old ones host-side, and a freed slot is immediately
        // re-claimable — so without this, ground can be handed to another
        // sequence while the GPU still has queued work reading it, and that read
        // returns whatever the new owner has since written.
        //
        // It has to be the **context**, not `cuda_stream()`. The expert cache
        // holds a `CU_STREAM_NON_BLOCKING` copy stream for DMA overlap, and a
        // non-blocking stream is by definition not ordered against the null
        // stream everything else runs on. Synchronising one stream would leave
        // exactly the transfer least ordered with this pass still in flight —
        // and after a weight-side concession its destination may be ground the
        // KV side now owns.
        //
        // Affordable **because of where this sits**: between forwards, on the
        // persistence thread, once per residence — never inside a wave. The
        // measured cost of getting that granularity wrong is in
        // `persistence::thread`'s note above the caller: widening the hold to a
        // whole pass took dirs/min from 2.12 to 1.41. A drain at the start of a
        // residence, when the queue is shortest, is the cheap end of that trade.
        cuda.cuda_context()
            .synchronize()
            .map_err(candle::Error::wrap)?;
        Ok(Some(window))
    }

    #[cfg(feature = "cuda")]
    pub fn relocate_sealed(
        &self,
        sequences: &[SealedSequence],
        plans: &mut HashMap<usize, RelocationPlan>,
    ) -> candle::Result<Option<(Vec<SealedSequence>, RelocationOutcome)>> {
        use candle::cuda_backend::cudarc::driver::result::memcpy_dtod_async;

        let Device::Cuda(cuda) = self.device() else {
            return Ok(None);
        };
        if sequences.iter().all(|s| s.location != ArenaLocation::Gpu) {
            return Ok(None);
        }

        // The window belongs to the pass — see `begin_relocation_pass`. Taking
        // it here would ask the same question 1,536 times and get a different
        // answer each time a forward started.
        let stream = cuda.cuda_stream();

        let mut out: Vec<SealedSequence> = Vec::with_capacity(sequences.len());
        let mut stats = RelocationOutcome::default();
        // Chunks whose bands moved, paired with the position they must be
        // written back to, so their records can be built in one batched upload
        // after every copy has been issued.
        let mut rebuilt: Vec<(usize, usize)> = Vec::new();
        // **Resolve the arenas this call can touch, not every arena there is.**
        //
        // The dense resolve is O(num_arenas) of pointer lookups, and this
        // function runs once per residence *per layer* — ~1,536 times a pass, on
        // the thread that also owns the hot→warm drain. The set it can actually
        // address is knowable up front and small: the arenas this sequence's
        // bands already live in, plus the recipients the plans may hand out.
        let needed: std::collections::HashSet<usize> = sequences
            .iter()
            .flat_map(|s| s.chunks.iter())
            .flat_map(|c| c.gids.as_slice())
            .filter(|g| !g.is_empty())
            .map(|g| g.arena_idx())
            .chain(
                plans
                    .values()
                    .flat_map(|p| p.recipients.iter().map(|(idx, _)| *idx)),
            )
            .collect();
        let mut info = self.resolve_arena_info_for(&needed)?;

        for (seq_pos, seq) in sequences.iter().enumerate() {
            let mut chunks: Vec<SealedChunk> = Vec::with_capacity(seq.chunks.len());
            for (chunk_pos, chunk) in seq.chunks.iter().enumerate() {
                let old = chunk.gids.as_slice();
                let mut next: Vec<super::gid_pool::ChunkGid> = Vec::with_capacity(old.len());
                let mut touched = false;

                for gid in old {
                    // Sentinels (absent palettes) carry no arena and never move.
                    let Some(key) = gid.route_key().copied() else {
                        next.push(gid.clone());
                        continue;
                    };
                    if key.location != ArenaLocation::Gpu {
                        next.push(gid.clone());
                        continue;
                    }
                    let stride = key.slot_stride();
                    let wanted = plans.get(&stride).is_some_and(|p| p.wants(gid.arena_idx()));
                    if !wanted {
                        next.push(gid.clone());
                        continue;
                    }
                    // **The plan names the destination; the allocator is not
                    // asked to choose.** Left to itself the pool hands back the
                    // leftmost arena with room — and draining an arena is
                    // precisely what makes it the arena with the most room, so
                    // the better this pass worked the more likely each next
                    // chunk would land back in a donor. The plan excludes donors
                    // from its recipients, which is the only thing that stops
                    // the pass chasing its own tail.
                    //
                    // `None` means every recipient is full: leave the band where
                    // it is rather than allocate, because an allocation here
                    // would create a fresh arena and undo the pass's purpose.
                    let Some(target) = plans.get_mut(&stride).and_then(|p| p.claim()) else {
                        next.push(gid.clone());
                        continue;
                    };
                    let Some(new_gid) = self.pool_allocate_from_arena(key, target) else {
                        // Its room went elsewhere between the census and the
                        // claim. Give the budget back before moving on — spent
                        // room that bought nothing makes the plan believe the
                        // recipients are fuller than they are, and the donors it
                        // proved affordable then run out part-drained, which
                        // frees no arena at all.
                        if let Some(p) = plans.get_mut(&stride) {
                            p.release(target);
                        }
                        next.push(gid.clone());
                        continue;
                    };
                    if info
                        .get(new_gid.arena_idx())
                        .is_none_or(|a| a.base_ptr == 0)
                    {
                        info = self.resolve_arena_info()?;
                    }
                    let (Some(src), Some(dst)) = (
                        super::backing::slot_addr(&info, gid.arena_idx(), gid.chunk_idx(), stride),
                        super::backing::slot_addr(
                            &info,
                            new_gid.arena_idx(),
                            new_gid.chunk_idx(),
                            stride,
                        ),
                    ) else {
                        tracing::warn!(
                            from = gid.arena_idx(),
                            to = new_gid.arena_idx(),
                            "relocate: an end of the copy resolves to no base — \
                             leaving the band where it is",
                        );
                        next.push(gid.clone());
                        continue;
                    };
                    // SAFETY: both name one slot of the reservation, `stride`
                    // bytes long, in arenas of the same class. The destination
                    // is a slot only this claim holds; the source is read-only
                    // here and stays live until the caller drops the sequence it
                    // came from.
                    unsafe {
                        memcpy_dtod_async(dst, src, stride, stream.cu_stream())
                            .map_err(|e| candle::Error::Msg(format!("relocate copy: {e}")))?;
                    }
                    next.push(new_gid);
                    touched = true;
                    stats.bands_moved += 1;
                }

                if touched {
                    stats.chunks_rebuilt += 1;
                    rebuilt.push((seq_pos, chunk_pos));
                }
                // A fresh vector either way, but only a touched chunk gets a
                // fresh record — see the batched build below. `meta: None` is
                // the same placeholder `quantize_sealed_in_place` uses between
                // building a chunk and giving it its record.
                chunks.push(SealedChunk {
                    gids: HeadGids::from_vec(next),
                    offset: chunk.offset,
                    token_count: chunk.token_count,
                    k_pal: chunk.k_pal.clone(),
                    v_pal: chunk.v_pal.clone(),
                    k_scale: chunk.k_scale.clone(),
                    v_scale: chunk.v_scale.clone(),
                    k_fmt: chunk.k_fmt.clone(),
                    v_fmt: chunk.v_fmt.clone(),
                    byte_size: chunk.byte_size,
                    meta: if touched { None } else { chunk.meta.clone() },
                });
            }
            out.push(SealedSequence {
                chunks,
                token_count: seq.token_count,
                chunk_size: seq.chunk_size,
                location: seq.location,
            });
        }

        if stats.is_empty() {
            return Ok(None);
        }

        // **Bytes before pointers.** The records built below name the
        // destinations; nothing may read through them until the copies that
        // filled those slots have run. One barrier for the whole call, not one
        // per band.
        stream.synchronize().map_err(candle::Error::wrap)?;

        // **A fresh record per rebuilt chunk, never a rewritten one.** This is
        // the property that removes the race rather than narrowing it: the old
        // chunk keeps its own record, at its own address, naming its own slot,
        // and stays wholly valid until the caller drops it. Nothing is ever
        // half-updated because nothing is updated at all.
        // The destinations the bands actually landed in — the same bounded set
        // reasoning as the resolve above, now that the outcome is known.
        let landed: std::collections::HashSet<usize> = rebuilt
            .iter()
            .flat_map(|&(s, c)| out[s].chunks[c].gids.as_slice())
            .filter(|g| !g.is_empty())
            .map(|g| g.arena_idx())
            .collect();
        let fresh = self.resolve_arena_info_for(&landed)?;
        let srcs: Vec<ChunkRecordSrc<'_>> = rebuilt
            .iter()
            .map(|&(s, c)| {
                let ch = &out[s].chunks[c];
                ChunkRecordSrc {
                    gids: &ch.gids,
                    k_pal: ch.k_pal.as_slice(),
                    v_pal: ch.v_pal.as_slice(),
                    k_scale: ch.k_scale.as_slice(),
                    v_scale: ch.v_scale.as_slice(),
                    k_fmt: ch.k_fmt.as_slice(),
                    v_fmt: ch.v_fmt.as_slice(),
                }
            })
            .collect();
        let metas = self.build_meta_records(&srcs, &fresh)?;
        for (&(s, c), meta) in rebuilt.iter().zip(metas) {
            out[s].chunks[c].meta = meta;
        }
        // The record uploads are asynchronous, so they must land before the
        // caller installs these sequences — installing is what publishes them to
        // readers.
        //
        // Today `MetaPool::cuda_stream` returns the device default stream, the
        // same one synchronised here, so this is a genuine barrier. It is not
        // *stated* as "the meta pool's own stream" on purpose: an earlier comment
        // in this pass claimed exactly that about the copy stream, was wrong, and
        // sent a whole investigation after a race that could not exist. If the
        // meta pool ever takes a private stream — the elevate path already has
        // one — this synchronise stops covering it and the ordering has to be
        // re-established against that stream instead.
        stream.synchronize().map_err(candle::Error::wrap)?;

        Ok(Some((out, stats)))
    }
}
