//! Checks that a compaction sweep rewrites each shared allocation once, passes
//! untouched payloads by, and keeps every retired record alive.
//!
//! Liveness is asserted on refcounts rather than on address reuse, since an address
//! collision would pass or fail by luck. A second visit to a rewritten holder must
//! match nothing, even after its original is dropped, because the scheduler sweeps
//! a shared substrate once per slot. A minted record is never swapped for a
//! relocated copy, since its raw id may be a freed slot that once held another
//! chunk's record.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::kv_cache::chunked::meta_pool::MetaGid;
    use crate::kv_cache::chunked::types::GID_STRIDE;

    fn raw(arena: usize, chunk: usize) -> i64 {
        (arena * GID_STRIDE + chunk) as i64
    }

    /// **An unmoved allocation is scanned once a sweep.** Its later visits —
    /// through any holder sharing it — are answered from the first, and say the
    /// same: nothing moved. A moved allocation visited between them is rewritten
    /// as ever, and an equal but distinct allocation is scanned for itself.
    #[test]
    fn an_unmoved_allocation_is_scanned_once_a_sweep() {
        let mut map = CompactionMap::new();
        map.insert(raw(3, 1), ChunkGid::detached(raw(0, 4)), 0x2000);
        let still = HeadGids::uniform(ChunkGid::detached(raw(3, 2)), 4);
        let shared = still.clone();
        let moved = HeadGids::uniform(ChunkGid::detached(raw(3, 1)), 4);
        let equal = HeadGids::uniform(ChunkGid::detached(raw(3, 2)), 4);
        let mut sweep = Sweep::new(&map);
        assert!(sweep.rewrite_gids(&still).unwrap().is_none());
        assert_eq!(sweep.clean_reused(), 0);
        let next = sweep.rewrite_gids(&moved).unwrap().expect("moved");
        assert_eq!(next.as_slice()[0].raw(), raw(0, 4));
        assert!(sweep.rewrite_gids(&shared).unwrap().is_none());
        assert!(sweep.rewrite_gids(&still).unwrap().is_none());
        assert_eq!(sweep.clean_reused(), 2);
        assert!(sweep.rewrite_gids(&equal).unwrap().is_none());
        assert_eq!(sweep.clean_reused(), 2, "another allocation is its own");
        assert_eq!(sweep.allocations_rewritten(), 1);
    }

    /// A gid from an untouched arena is rejected without consulting the hash map
    /// — the fast path a whole-process sweep depends on.
    #[test]
    fn an_untouched_arena_is_rejected_by_the_presence_byte() {
        let mut map = CompactionMap::new();
        map.insert(raw(9, 3), ChunkGid::detached(raw(0, 0)), 0x1000);
        assert!(map.get(&ChunkGid::detached(raw(4, 3))).is_none());
        assert!(
            map.get(&ChunkGid::detached(raw(9, 7))).is_none(),
            "same arena, different chunk"
        );
        assert!(map.get(&ChunkGid::detached(raw(9, 3))).is_some());
    }

    /// **A run of equal ids is asked once, and a moved id after a run is still
    /// found.** The palettes' repeats of an unmoved K then a moved V say yes;
    /// repeats of unmoved ids, in touched and untouched arenas alike, say no.
    #[test]
    fn names_moved_answers_once_per_run_and_finds_a_moved_id_after_one() {
        let mut map = CompactionMap::new();
        map.insert(raw(9, 3), ChunkGid::detached(raw(0, 0)), 0x1000);
        let g = |arena, chunk| ChunkGid::detached(raw(arena, chunk));
        let k = g(9, 7);
        let v = g(9, 3);
        assert!(map.names_moved(&[k.clone(), k.clone(), k.clone(), v.clone(), v]));
        assert!(!map.names_moved(&[k.clone(), k.clone(), k, g(4, 3), g(4, 3)]));
        assert!(!map.names_moved(&[]));
    }

    /// A sentinel never moves and never hashes.
    #[test]
    fn a_sentinel_gid_is_never_translated() {
        let mut map = CompactionMap::new();
        map.insert(raw(1, 1), ChunkGid::detached(raw(0, 0)), 0x20);
        assert!(map.get(&ChunkGid::detached(-1)).is_none());
    }

    /// An empty map rewrites nothing, and says so rather than handing back an
    /// identical copy.
    #[test]
    fn an_empty_map_rewrites_nothing() {
        let map = CompactionMap::new();
        let mut sweep = Sweep::new(&map);
        assert!(rewrite_sealed(&[], &mut sweep).unwrap().is_none());
    }

    /// **One allocation, one rewrite, installed twice.**
    ///
    /// Two holders sharing a `HeadGids` allocation — what every view, injected
    /// chunk and adopted turn produces — must come back sharing ONE replacement
    /// allocation. If the sweep rewrote per holder they would hold equal but
    /// distinct gids, and the refcounts would then disagree with the sharing the
    /// cache believes exists.
    #[test]
    fn a_shared_allocation_is_rewritten_once_and_shared_again() {
        let shared = HeadGids::uniform(ChunkGid::detached(raw(5, 2)), 1);
        let seq_a = SealedSequence {
            chunks: vec![chunk_with(shared.clone())],
            token_count: 32,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        };
        let seq_b = SealedSequence {
            chunks: vec![chunk_with(shared.clone())],
            token_count: 32,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        };

        let mut map = CompactionMap::new();
        map.insert(raw(5, 2), ChunkGid::detached(raw(0, 1)), 0x4000);
        let mut sweep = Sweep::new(&map);

        let a = rewrite_sealed(&[seq_a], &mut sweep)
            .unwrap()
            .expect("moved");
        let b = rewrite_sealed(&[seq_b], &mut sweep)
            .unwrap()
            .expect("moved");

        assert_eq!(
            sweep.allocations_rewritten(),
            1,
            "one allocation, one rewrite"
        );
        assert!(
            a[0].chunks[0].gids.is_same_alloc(&b[0].chunks[0].gids),
            "both holders must end up on ONE replacement allocation",
        );
        assert_eq!(a[0].chunks[0].gids.as_slice()[0].raw(), raw(0, 1));
    }

    /// **A second visit to an already-rewritten holder must match nothing — even after the
    /// original allocation has been dropped.**
    ///
    /// The scheduler sweeps the same substrate once per batch slot that shares it, so a
    /// second visit is ordinary and the pass relies on it being a no-op. The memo is keyed
    /// on `HeadGids::alloc_id`, an `Arc` address, so that only holds while the original is
    /// alive — and once a compaction mints records, installing a fresh one drops the old
    /// record's clone of the old gids and originals start dying *mid-sweep*. A replacement
    /// allocated at a dead original's address then collides with its memo key, and because
    /// `rewrite_gids` consults the memo before asking whether anything moved, the holder is
    /// handed **another chunk's gids**.
    ///
    /// Asserted on the **refcount**, not on an address collision. A test that dropped an
    /// original and hoped the allocator reused its address would pass or fail by luck; what
    /// makes the memo key safe is that the original is still *alive*, and a band's strong
    /// count says so deterministically. The second-visit checks ride along.
    #[test]
    fn a_second_visit_matches_nothing_after_the_original_is_dropped() {
        let mut map = CompactionMap::new();
        map.insert(raw(5, 2), ChunkGid::detached(raw(0, 1)), 0x4000);
        map.insert(raw(6, 4), ChunkGid::detached(raw(0, 9)), 0x8000);

        let band = ChunkGid::detached(raw(5, 2));
        let unheld = band.strong_count();
        let mut sweep = Sweep::new(&map);

        // Rewrite, install, and drop the original — the holder's own lifecycle.
        let first = {
            let original = HeadGids::uniform(band.clone(), 1);
            let held = band.strong_count();
            assert!(held > unheld, "the original gid vector holds the band");
            let next = sweep.rewrite_gids(&original).unwrap().expect("moved");
            drop(original);
            assert_eq!(
                band.strong_count(),
                held,
                "the SWEEP still holds the original, so its `alloc_id` cannot be recycled \
                 under the memo key while the memo entry lives",
            );
            next
        };
        assert_eq!(first.as_slice()[0].raw(), raw(0, 1));

        let second = {
            let original = HeadGids::uniform(ChunkGid::detached(raw(6, 4)), 1);
            let next = sweep.rewrite_gids(&original).unwrap().expect("moved");
            drop(original);
            next
        };
        assert_eq!(second.as_slice()[0].raw(), raw(0, 9));

        // Both holders now hold replacements, which name destinations the map has no key
        // for, so a second visit must match nothing rather than answer from the memo.
        assert!(
            sweep.rewrite_gids(&first).unwrap().is_none(),
            "a second visit to the first holder must match nothing",
        );
        assert!(
            sweep.rewrite_gids(&second).unwrap().is_none(),
            "and a replacement must never be answered with another chunk's gids",
        );
        assert_eq!(
            sweep.allocations_rewritten(),
            2,
            "two allocations rewritten, and no third invented by a recycled memo key",
        );

        // The sweep is what was holding it; when it goes, so does the original.
        drop(sweep);
        assert_eq!(band.strong_count(), unheld);
    }

    /// A chunk none of whose gids moved is passed through by value, and the
    /// sequence reports no change at all when that is true of every chunk.
    #[test]
    fn an_untouched_sequence_is_not_replaced() {
        let seq = SealedSequence {
            chunks: vec![chunk_with(HeadGids::uniform(
                ChunkGid::detached(raw(7, 7)),
                1,
            ))],
            token_count: 32,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        };
        let mut map = CompactionMap::new();
        map.insert(raw(2, 0), ChunkGid::detached(raw(0, 0)), 0x10);
        let mut sweep = Sweep::new(&map);
        assert!(rewrite_sealed(&[seq], &mut sweep).unwrap().is_none());
        assert_eq!(sweep.allocations_rewritten(), 0);
    }

    /// **A host sequence is not walked, and a GPU one beside it still is.** The two
    /// name the same raw id, so the only thing telling them apart is `location`: the
    /// warm copy is carried through as it is, the hot one is rewritten.
    #[test]
    fn a_host_sequence_is_carried_through_unwalked() {
        let seq = |location| SealedSequence {
            chunks: vec![chunk_with(HeadGids::uniform(
                ChunkGid::detached(raw(5, 2)),
                1,
            ))],
            token_count: 32,
            chunk_size: 32,
            location,
        };
        let mut map = CompactionMap::new();
        map.insert(raw(5, 2), ChunkGid::detached(raw(0, 1)), 0x4000);
        let mut sweep = Sweep::new(&map);
        assert!(rewrite_sealed(&[seq(ArenaLocation::Cpu)], &mut sweep)
            .unwrap()
            .is_none());
        assert_eq!(sweep.allocations_rewritten(), 0, "nothing walked");

        let out = rewrite_sealed(
            &[seq(ArenaLocation::Cpu), seq(ArenaLocation::Gpu)],
            &mut sweep,
        )
        .unwrap()
        .expect("the hot copy moved");
        assert_eq!(out[0].chunks[0].gids.as_slice()[0].raw(), raw(5, 2));
        assert_eq!(out[0].location, ArenaLocation::Cpu);
        assert_eq!(out[1].chunks[0].gids.as_slice()[0].raw(), raw(0, 1));
    }

    /// **Copy on write keeps every untouched chunk and sequence, in order.** The sweep
    /// clones nothing until the first change, so the chunks and sequences before it are
    /// copied in afterwards, and everything after it is carried through — the replacement
    /// must be the input with exactly the moved chunk swapped.
    #[test]
    fn a_change_mid_walk_keeps_every_untouched_chunk_in_place() {
        let gids = |arena, chunk| HeadGids::uniform(ChunkGid::detached(raw(arena, chunk)), 1);
        let seq = |chunks: Vec<SealedChunk>| SealedSequence {
            token_count: 32 * chunks.len(),
            chunks,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        };
        let before = seq(vec![chunk_with(gids(7, 0)), chunk_with(gids(7, 1))]);
        let moved = seq(vec![
            chunk_with(gids(7, 2)),
            chunk_with(gids(5, 2)),
            chunk_with(gids(7, 3)),
        ]);
        let after = seq(vec![chunk_with(gids(7, 4))]);
        let input = [before, moved, after];

        let mut map = CompactionMap::new();
        map.insert(raw(5, 2), ChunkGid::detached(raw(0, 1)), 0x4000);
        let mut sweep = Sweep::new(&map);
        let out = rewrite_sealed(&input, &mut sweep).unwrap().expect("moved");

        let names: Vec<Vec<i64>> = out
            .iter()
            .map(|s| {
                s.chunks
                    .iter()
                    .map(|c| c.gids.as_slice()[0].raw())
                    .collect()
            })
            .collect();
        assert_eq!(
            names,
            vec![
                vec![raw(7, 0), raw(7, 1)],
                vec![raw(7, 2), raw(0, 1), raw(7, 3)],
                vec![raw(7, 4)],
            ],
        );
        for (s, (o, i)) in out.iter().zip(input.iter()).enumerate() {
            assert_eq!(o.token_count, i.token_count, "sequence {s}");
            for (c, (oc, ic)) in o.chunks.iter().zip(i.chunks.iter()).enumerate() {
                if (s, c) != (1, 1) {
                    assert!(
                        oc.gids.is_same_alloc(&ic.gids),
                        "untouched chunk {s}.{c} is carried through, not rebuilt",
                    );
                }
            }
        }
    }

    /// **A destination is witnessed where a holder installs it, not where it is
    /// computed.** A chunk the sweep leaves on its source for want of a record computes
    /// its replacement and installs none, and must not read as a holder of the copy — that
    /// is how the completeness count tells an unreached holder from a declined mint.
    #[test]
    fn a_destination_is_witnessed_only_when_a_holder_installs_it() {
        let mut map = CompactionMap::new();
        map.insert(raw(5, 2), ChunkGid::detached(raw(0, 1)), 0x4000);
        let mut sweep = Sweep::new(&map);
        let gids = HeadGids::uniform(ChunkGid::detached(raw(5, 2)), 1);

        sweep.rewrite_gids(&gids).unwrap().expect("moved");
        assert!(
            sweep.witnessed.is_empty(),
            "computing a replacement names nothing",
        );

        let seq = SealedSequence {
            chunks: vec![chunk_with(gids)],
            token_count: 32,
            chunk_size: 32,
            location: crate::kv_cache::ArenaLocation::Gpu,
        };
        rewrite_sealed(&[seq], &mut sweep).unwrap().expect("moved");
        assert_eq!(
            sweep.witnessed.iter().copied().collect::<Vec<_>>(),
            [raw(0, 1)],
            "the holder that installed the replacement names its destination",
        );
    }

    /// **With no minter, a rewritten chunk keeps its record — and the record keeps the
    /// bands it named, so it is still describing live ground.**
    ///
    /// A `Sweep::new` sweep mints nothing: that is the shape off a device, where there
    /// are no `KvHead` records to mint, and it must still leave the pool consistent. The
    /// rewrite replaces `gids`, so the chunk names the destination while `meta` names the
    /// source — which is safe precisely because the record holds a clone of the
    /// `HeadGids` it was serialized from, so that source keeps a refcount and keeps the
    /// bytes the copy read out of it.
    ///
    /// The device path replaces the record instead ([`super::compact_mint`]); what is
    /// pinned here is the *lifetime* claim underneath both, and the thing that would break
    /// it is somebody clearing `meta` in `rewrite_sealed` — which reads as tidying up and
    /// is the whole corruption.
    #[test]
    fn without_a_minter_a_rewritten_chunk_keeps_a_record_that_still_owns_its_bands() {
        let source = ChunkGid::detached(raw(5, 2));
        let bands = HeadGids::uniform(source.clone(), 1);
        let record = MetaGid::from_slot(ChunkGid::detached(raw(9, 9)), bands.clone(), 0xFEED);
        let seq = SealedSequence {
            chunks: vec![SealedChunk {
                meta: Some(record),
                ..chunk_with(bands)
            }],
            token_count: 32,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        };

        let mut map = CompactionMap::new();
        map.insert(raw(5, 2), ChunkGid::detached(raw(0, 1)), 0x4000);
        let mut sweep = Sweep::new(&map);
        let out = rewrite_sealed(&[seq], &mut sweep).unwrap().expect("moved");
        let chunk = &out[0].chunks[0];

        assert_eq!(
            chunk.gids.as_slice()[0].raw(),
            raw(0, 1),
            "the chunk must name the destination"
        );
        let meta = chunk.meta.as_ref().expect("the record is carried through");
        assert_eq!(meta.device_addr(), 0xFEED, "and it is the SAME record");
        assert_eq!(
            meta.bands().expect("a record names bands").as_slice()[0].raw(),
            raw(5, 2),
            "which still holds — and so still refcounts — the SOURCE band",
        );
    }

    fn sealed(chunk: SealedChunk) -> SealedSequence {
        SealedSequence {
            chunks: vec![chunk],
            token_count: 32,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        }
    }

    /// **A holder of a relocated record is moved onto the copy, and nothing else about
    /// the chunk changes.** The copy's handle carries the very `bands` allocation the old
    /// one held, because a copy of a record's bytes names the same bands.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_relocated_record_is_followed_with_its_bands() {
        let bands = HeadGids::uniform(ChunkGid::detached(raw(5, 2)), 1);
        let record = MetaGid::from_slot(ChunkGid::detached(raw(9, 9)), bands.clone(), 0xFEED);
        let seq = sealed(SealedChunk {
            meta: Some(record),
            ..chunk_with(bands.clone())
        });

        let mut map = CompactionMap::new();
        map.insert_record(raw(9, 9), ChunkGid::detached(raw(1, 4)), 0xBEEF);
        let mut sweep = Sweep::new(&map);
        let out = rewrite_sealed(&[seq], &mut sweep)
            .unwrap()
            .expect("a followed record is a change the caller must install");

        let chunk = &out[0].chunks[0];
        assert!(chunk.gids.is_same_alloc(&bands), "the bands did not move");
        let meta = chunk.meta.as_ref().unwrap();
        assert_eq!(meta.raw(), raw(1, 4));
        assert_eq!(meta.device_addr(), 0xBEEF);
        assert!(meta.bands().unwrap().is_same_alloc(&bands));
        assert_eq!(sweep.records_unfollowed(), 0);
    }

    /// **The record a holder gave up stays alive until the sweep ends.** Records are
    /// followed by raw id; a retired record that freed its slot mid-sweep could have that
    /// slot handed to a mint, and a later visit would then follow the mint's raw id to a
    /// copy of another chunk's record. Asserted on the refcount, as the band-side memo test
    /// is, because liveness is what makes the raw id safe.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_retired_record_is_held_until_the_sweep_ends() {
        let bands = HeadGids::uniform(ChunkGid::detached(raw(5, 2)), 1);
        let slot = ChunkGid::detached(raw(9, 9));
        let unheld = slot.strong_count();
        let seq = sealed(SealedChunk {
            meta: Some(MetaGid::from_slot(slot.clone(), bands.clone(), 0xFEED)),
            ..chunk_with(bands)
        });
        let held = slot.strong_count();
        assert!(held > unheld, "the chunk's record holds its slot");

        let mut map = CompactionMap::new();
        map.insert_record(raw(9, 9), ChunkGid::detached(raw(1, 4)), 0xBEEF);
        let mut sweep = Sweep::new(&map);
        let out = rewrite_sealed(&[seq], &mut sweep)
            .unwrap()
            .expect("followed");
        drop(out);
        assert_eq!(
            slot.strong_count(),
            held,
            "with every holder gone, the sweep is still holding the old record",
        );

        // A second visit to a holder of the copy matches nothing.
        let copy = MetaGid::from_slot(
            ChunkGid::detached(raw(1, 4)),
            HeadGids::uniform(ChunkGid::detached(raw(5, 2)), 1),
            0xBEEF,
        );
        let mut again = Some(copy);
        assert!(!sweep.follow_record(&mut again));

        drop(sweep);
        assert_eq!(slot.strong_count(), unheld, "and lets go with the sweep");
    }

    /// **A record this sweep minted is never swapped for a relocated copy**, even when
    /// its raw id is one the record map knows — which is what a mint landing in a planned
    /// source's freed slot looks like. The copy would name the old chunk's bands.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_minted_record_is_never_followed() {
        let mut map = CompactionMap::new();
        map.insert_record(raw(9, 9), ChunkGid::detached(raw(1, 4)), 0xBEEF);
        let mut sweep = Sweep::new(&map);
        sweep.note_minted(raw(9, 9));

        let fresh = MetaGid::from_slot(
            ChunkGid::detached(raw(9, 9)),
            HeadGids::uniform(ChunkGid::detached(raw(0, 1)), 1),
            0xF00D,
        );
        let mut meta = Some(fresh);
        assert!(!sweep.follow_record(&mut meta));
        assert_eq!(meta.unwrap().device_addr(), 0xF00D);
        assert_eq!(
            sweep.records_unfollowed(),
            1,
            "the copy nobody took is reported, not hidden",
        );
    }

    /// Only the record map moves: a map holding record moves and no band moves is not
    /// empty, so the holder walk does not short-circuit past the records.
    #[test]
    fn a_map_with_only_record_moves_is_not_empty() {
        let mut map = CompactionMap::new();
        assert!(map.is_empty());
        map.insert_record(raw(9, 9), ChunkGid::detached(raw(1, 4)), 0xBEEF);
        assert!(!map.is_empty());
        assert_eq!(map.len(), 0, "no band moved");
        assert_eq!(map.records_len(), 1);
    }

    /// **A payload in arenas a pass touched nothing of is one the sweep finds
    /// nothing in.** Its set names the arenas of its device chunks' gids and
    /// records and none of a host sequence's; a pass that moved a chunk in
    /// another arena — even another chunk of one of its arenas — may touch it
    /// only when an arena it names gave something up, and whenever the answer
    /// is `no`, [`rewrite_sealed`] returns `None` for it.
    #[test]
    fn a_payload_in_untouched_arenas_is_passed_by() {
        let gpu = sealed(chunk_with(HeadGids::uniform(
            ChunkGid::detached(raw(3, 1)),
            4,
        )));
        let host = SealedSequence {
            location: ArenaLocation::Cpu,
            ..sealed(chunk_with(HeadGids::uniform(
                ChunkGid::detached(raw(8, 1)),
                4,
            )))
        };
        let payload = [gpu, host];
        let set = ArenaSet::of(&payload);
        assert_eq!(set.chunk_arenas(), &[3]);
        assert!(set.record_arenas().is_empty());
        let check = |map: &CompactionMap, touches: bool| {
            let mut sweep = Sweep::new(map);
            assert_eq!(sweep.may_touch(&set), touches);
            let found = rewrite_sealed(&payload, &mut sweep).unwrap().is_some();
            assert!(touches || !found, "passed by, yet the sweep found a change");
        };
        let mut elsewhere = CompactionMap::new();
        elsewhere.insert(raw(4, 1), ChunkGid::detached(raw(0, 1)), 0x100);
        check(&elsewhere, false);
        let mut host_only = CompactionMap::new();
        host_only.insert(raw(8, 1), ChunkGid::detached(raw(0, 2)), 0x200);
        check(&host_only, false);
        let mut same_arena = CompactionMap::new();
        same_arena.insert(raw(3, 7), ChunkGid::detached(raw(0, 3)), 0x300);
        check(&same_arena, true);
        let mut moved = CompactionMap::new();
        moved.insert(raw(3, 1), ChunkGid::detached(raw(0, 4)), 0x400);
        check(&moved, true);
    }

    /// **A replacement witnessed once stands for every holder that installs
    /// it**: a second install of the same allocation, and a run of one id
    /// across its palettes, leave the witnessed set as one read of it would —
    /// and a destination no holder names is still the gap it was.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_shared_replacement_is_witnessed_once() {
        let mut map = CompactionMap::new();
        map.insert(raw(3, 1), ChunkGid::detached(raw(0, 4)), 0x100);
        map.insert(raw(3, 2), ChunkGid::detached(raw(0, 5)), 0x200);
        let moved = HeadGids::from_vec(vec![
            ChunkGid::detached(raw(3, 1)),
            ChunkGid::detached(raw(3, 1)),
            ChunkGid::detached(raw(7, 7)),
        ]);
        let mut sweep = Sweep::new(&map);
        let next = sweep.rewrite_gids(&moved).unwrap().expect("moved");
        sweep.witness(&next);
        sweep.witness(&next);
        assert_eq!(sweep.unwitnessed(&map), vec![raw(0, 5)]);
    }

    /// **A record in a touched arena is reason enough to read the payload.**
    #[cfg(feature = "cuda")]
    #[test]
    fn a_payload_whose_record_arena_gave_one_up_is_read() {
        let bands = HeadGids::uniform(ChunkGid::detached(raw(5, 2)), 1);
        let record = MetaGid::from_slot(ChunkGid::detached(raw(9, 9)), bands.clone(), 0xFEED);
        let payload = [sealed(SealedChunk {
            meta: Some(record),
            ..chunk_with(bands)
        })];
        let set = ArenaSet::of(&payload);
        assert_eq!(set.record_arenas(), &[9]);
        let mut map = CompactionMap::new();
        map.insert_record(raw(9, 9), ChunkGid::detached(raw(1, 4)), 0xBEEF);
        assert!(Sweep::new(&map).may_touch(&set));
        let mut other = CompactionMap::new();
        other.insert_record(raw(10, 1), ChunkGid::detached(raw(1, 5)), 0xBEEF);
        assert!(!Sweep::new(&other).may_touch(&set));
    }

    fn chunk_with(gids: HeadGids) -> SealedChunk {
        SealedChunk {
            gids,
            offset: 0,
            token_count: 32,
            k_pal: std::sync::Arc::new(Vec::new()),
            v_pal: std::sync::Arc::new(Vec::new()),
            k_scale: std::sync::Arc::new(Vec::new()),
            v_scale: std::sync::Arc::new(Vec::new()),
            k_fmt: std::sync::Arc::new(Vec::new()),
            v_fmt: std::sync::Arc::new(Vec::new()),
            byte_size: 0,
            meta: None,
            q_sign: None,
        }
    }
}
