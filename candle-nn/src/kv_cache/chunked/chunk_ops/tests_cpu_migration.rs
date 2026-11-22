//! Tests the staging groups, and the write-then-read identity of chunk storage on
//! the CPU.
//!
//! Staging groups never exceed the span, and a band wider than the cap forms its
//! own group rather than being refused or looping. Write-then-read is the identity
//! at every head dimension and across chunk boundaries, and a window ring restored
//! from a snapshot keeps its absolute positions. A relocation into a destination
//! of a different class is refused.

use super::*;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::kv_cache::chunked::ChunkedKvBacking;

    use candle::{DType, Device, Tensor};

    use crate::kv_cache::chunked::size_class::SizeClass;

    // ── Batched-async migrate tests (GPU-only) ──────────────────────────────
    //
    // These exercise `migrate_sealed_to_cpu_batch_async` and the warm lift
    // (`warm_lift::lift_warm_to_gpu`) — the production hot↔warm path. They
    // require a CUDA device; on a CPU-only build each test silently returns
    // early.

    use super::super::tests_support::tests::{cpu_backing, seed_and_seal};

    /// The staging batcher must never hand a group larger than the persistence
    /// domain's span — that span is a hard allocation limit, and exceeding it
    /// fails the migrate outright rather than degrading.
    #[test]
    fn staging_groups_respect_the_cap() {
        // Exact fit, then a boundary crossing.
        let g = staging_groups(&[4, 4, 4, 4], 8);
        assert_eq!(g, vec![(0, 2, 8), (2, 4, 8)]);

        // Uneven: a group stops before it would exceed, never at exactly the
        // element that breaks it.
        let g = staging_groups(&[5, 5, 1], 8);
        assert_eq!(g, vec![(0, 1, 5), (1, 3, 6)]);

        // Empty input yields no groups (the callers skip the loop entirely).
        assert!(staging_groups(&[], 8).is_empty());

        // **A single element wider than the cap forms its own group.** Without
        // the `j > i` escape this loops forever, and the alternative — refusing
        // it — would strand a band that has nowhere smaller to go.
        let g = staging_groups(&[99, 3], 8);
        assert_eq!(g, vec![(0, 1, 99), (1, 2, 3)]);

        // Every group is contiguous, covers the whole input exactly once, and
        // its reported byte total matches its slice.
        let lens: Vec<usize> = (1..=40).map(|i| (i * 7) % 13 + 1).collect();
        let groups = staging_groups(&lens, 16);
        let mut expect_start = 0usize;
        for &(i, j, bytes) in &groups {
            assert_eq!(i, expect_start, "groups must be contiguous");
            assert!(j > i, "a group must make progress");
            assert_eq!(bytes, lens[i..j].iter().sum::<usize>());
            if j - i > 1 {
                assert!(bytes <= 16, "multi-element group {i}..{j} exceeded the cap");
            }
            expect_start = j;
        }
        assert_eq!(expect_start, lens.len(), "groups must cover every element");
    }

    /// A layer wider than the cap is split across groups, a group carries the
    /// tail of one layer and the head of the next, an empty layer contributes no
    /// run — and no multi-band group exceeds the cap. The shape of a 128K turn's
    /// migrate, where one layer is wider than the whole span.
    #[test]
    fn cross_layer_groups_split_a_layer_wider_than_the_cap() {
        let run = |layer, start, end| LayerRun { layer, start, end };
        let groups = cross_layer_staging_groups(
            &[
                vec![30, 30, 30],
                vec![10],
                vec![],
                vec![70],
                vec![20, 20, 20, 20],
            ],
            64,
        );
        assert_eq!(
            groups,
            vec![
                StagingGroup {
                    bytes: 60,
                    runs: vec![run(0, 0, 2)],
                },
                StagingGroup {
                    bytes: 40,
                    runs: vec![run(0, 2, 3), run(1, 0, 1)],
                },
                // A single band wider than the cap still forms its own group.
                StagingGroup {
                    bytes: 70,
                    runs: vec![run(3, 0, 1)],
                },
                StagingGroup {
                    bytes: 60,
                    runs: vec![run(4, 0, 3)],
                },
                StagingGroup {
                    bytes: 20,
                    runs: vec![run(4, 3, 4)],
                },
            ]
        );
        assert!(cross_layer_staging_groups(&[vec![], vec![]], 64).is_empty());
    }

    /// **Write-then-read must be the identity, at every head dim the kernels
    /// accept and across chunk boundaries.**
    ///
    /// The paged prefill path for head dims the int8 kernel does not
    /// instantiate writes unrotated K/V through this backing and immediately
    /// reads the whole prefix back to attend it. Every value it attends is
    /// therefore whatever this round-trip returns, and a layout that is right
    /// for `head_dim 128` but scrambles at 256 would surface as a model that
    /// answers fluently and wrongly — not as a crash.
    ///
    /// Values are distinct per `(token, head, dim)` rather than constant, so a
    /// transposition inside a chunk is caught and not just a dropped one; the
    /// lengths straddle `CHUNK_SIZE` so the partial tail, the exact boundary
    /// and the multi-chunk case are all covered.
    #[test]
    fn contiguous_round_trip_is_exact_at_every_head_dim() {
        let n_kv_head = 2usize;
        for head_dim in [64usize, 128, 256] {
            for n_tokens in [1usize, 31, 32, 33, 64, 100] {
                let backing =
                    ChunkedKvBacking::new(4, n_kv_head, head_dim, DType::F32, &Device::Cpu, 256)
                        .unwrap();
                let slot = backing.alloc_sequence().unwrap();
                backing.ensure_for_offset(slot, 0, n_tokens).unwrap();

                let n = n_kv_head * n_tokens * head_dim;
                // Distinct, exactly representable in F32, and ordered so the
                // failure message names the first wrong element usefully.
                let kv: Vec<f32> = (0..n).map(|i| i as f32).collect();
                let k =
                    Tensor::from_vec(kv.clone(), (1, n_kv_head, n_tokens, head_dim), &Device::Cpu)
                        .unwrap();
                let v = Tensor::from_vec(
                    kv.iter().map(|x| -x).collect::<Vec<f32>>(),
                    (1, n_kv_head, n_tokens, head_dim),
                    &Device::Cpu,
                )
                .unwrap();
                backing.write_contiguous(slot, 0, &k, &v).unwrap();
                backing.set_len(slot, n_tokens);

                let (k_out, v_out) = backing.read_contiguous(slot, 0, n_tokens).unwrap();
                let same = |a: &Tensor, b: &Tensor| -> f32 {
                    a.sub(b)
                        .unwrap()
                        .abs()
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .max(0)
                        .unwrap()
                        .to_scalar::<f32>()
                        .unwrap()
                };
                assert_eq!(
                    k_out.dims(),
                    k.dims(),
                    "K shape changed at head_dim {head_dim}, {n_tokens} tokens"
                );
                assert_eq!(
                    same(&k_out, &k),
                    0.0,
                    "K round-trip differs at head_dim {head_dim}, {n_tokens} tokens"
                );
                assert_eq!(
                    same(&v_out, &v),
                    0.0,
                    "V round-trip differs at head_dim {head_dim}, {n_tokens} tokens"
                );
            }
        }
    }

    /// Turn-seal window-ring snapshot → restore (Artifact A of
    /// docs/deepseek/deepseek_turn_seal_persistence.md): after the sliding-window ring
    /// evicts its front chunk, the resident window + `base_pos` — captured via
    /// the snapshot primitives (`window_base_pos` / `resident_len` /
    /// `read_contiguous`) — restore bit-exactly into a fresh backing
    /// (`ensure_for_offset` / `write_contiguous` / `set_len` /
    /// `set_window_base_pos`), preserving both the window content AND the
    /// absolute-position frame (each restored row still equals its absolute
    /// position `base_pos + i`).
    #[test]
    fn window_ring_snapshot_restore_round_trip() {
        let (n_kv_head, head_dim) = (1usize, 32usize);
        let backing = cpu_backing(n_kv_head, head_dim);
        let slot = backing.alloc_sequence().unwrap();
        let n = 80usize;
        backing.ensure_for_offset(slot, 0, n).unwrap();
        // Token t's whole row is the scalar t (bf16-exact for t < 256), so a
        // restored row can be checked against its ABSOLUTE position.
        let vals: Vec<f32> = (0..n)
            .flat_map(|t| std::iter::repeat_n(t as f32, head_dim))
            .collect();
        let k = Tensor::from_vec(vals, (1, n_kv_head, n, head_dim), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let v = k.clone();
        backing.write_contiguous(slot, 0, &k, &v).unwrap();
        backing.set_len(slot, n);
        // Eviction only slides SEALED front chunks (below `writer_start_idx`);
        // seal chunks 0 and 1 so the front is evictable (in the live model the
        // writer advances past chunks whose tokens are folded into the corpus).
        backing.test_set_writer_start(slot, 2).unwrap();

        // Slide the ring: window 32, query at 79 → front chunk [0,32) fully
        // exits (lowest in-window abs = 79 − 32 + 1 = 48 ≥ 32).
        let base_pos = backing.evict_window_front(slot, 32, n - 1).unwrap();
        assert_eq!(base_pos, 32, "front chunk [0,32) must evict");
        let resident_len = backing.resident_len(slot).unwrap();
        assert_eq!(resident_len, n - 32, "48 resident tokens (abs 32..79)");
        assert_eq!(backing.window_base_pos(slot).unwrap(), 32);

        // Snapshot the resident window.
        let (snap_k, snap_v) = backing.read_contiguous(slot, 0, resident_len).unwrap();

        // Restore into a fresh backing.
        let restored = cpu_backing(n_kv_head, head_dim);
        let rslot = restored.alloc_sequence().unwrap();
        restored.ensure_for_offset(rslot, 0, resident_len).unwrap();
        restored
            .write_contiguous(rslot, 0, &snap_k, &snap_v)
            .unwrap();
        restored.set_len(rslot, resident_len);
        restored.set_window_base_pos(rslot, base_pos).unwrap();

        assert_eq!(restored.window_base_pos(rslot).unwrap(), base_pos);
        assert_eq!(restored.resident_len(rslot).unwrap(), resident_len);

        // Content bit-exact (integers → exact in F32), and each row equals its
        // absolute position base_pos + i.
        let (rk, rv) = restored.read_contiguous(rslot, 0, resident_len).unwrap();
        let f32v = |t: &Tensor| {
            t.flatten_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        assert_eq!(f32v(&snap_k), f32v(&rk), "restored window K bit-exact");
        assert_eq!(f32v(&snap_v), f32v(&rv), "restored window V bit-exact");
        let rows = rk
            .reshape((resident_len, head_dim))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for (i, row) in rows.iter().enumerate() {
            let want = (base_pos as usize + i) as f32;
            assert!(
                row.iter().all(|&x| (x - want).abs() < 1e-3),
                "restored row {i} must equal absolute pos {want}"
            );
        }
    }

    /// `migrate_sealed_to_cpu` on a CPU-backed sequence copies chunks into new
    /// CPU arena slots.  Structural invariants (token count, chunk count, location)
    /// must be preserved.
    #[test]
    fn migrate_to_cpu_preserves_structure() {
        let n_kv_head = 4;
        let head_dim = 32;
        let n_tokens = 40; // 1 full chunk (32) + 1 partial (8)
        let backing = cpu_backing(n_kv_head, head_dim);
        let sealed = seed_and_seal(&backing, n_kv_head, head_dim, n_tokens);

        assert_eq!(sealed.token_count, n_tokens);
        assert_eq!(sealed.chunks.len(), 2);

        let cpu = backing.migrate_sealed_to_cpu(&sealed).unwrap();

        assert_eq!(cpu.location, ArenaLocation::Cpu);
        assert_eq!(cpu.token_count, n_tokens);
        assert_eq!(cpu.chunk_size, sealed.chunk_size);
        assert_eq!(cpu.chunks.len(), sealed.chunks.len());

        // Token counts per chunk must be preserved.
        for (orig, mig) in sealed.chunks.iter().zip(cpu.chunks.iter()) {
            assert_eq!(mig.token_count, orig.token_count);
            assert_eq!(mig.offset, orig.offset);
        }
    }

    /// After migration, the migrated GIDs must be distinct from the originals
    /// (new physical chunks were allocated, not just re-wrapped).
    #[test]
    fn migrate_to_cpu_produces_distinct_gids() {
        let n_kv_head = 2;
        let head_dim = 16;
        let n_tokens = 32; // exactly one full chunk
        let backing = cpu_backing(n_kv_head, head_dim);
        let sealed = seed_and_seal(&backing, n_kv_head, head_dim, n_tokens);

        let cpu = backing.migrate_sealed_to_cpu(&sealed).unwrap();

        assert_eq!(cpu.chunks.len(), 1);
        // At least one GID slot must differ from the original.
        let orig_gids: Vec<i64> = sealed.chunks[0].gids.iter().map(|g| g.raw()).collect();
        let mig_gids: Vec<i64> = cpu.chunks[0].gids.iter().map(|g| g.raw()).collect();
        assert_ne!(
            orig_gids, mig_gids,
            "migrated chunks must be new physical allocations"
        );
    }

    /// Migration preserves the GID-sharing structure of the source HeadGids.
    /// Slots that share a raw id in the source share a raw id in the output
    /// (same degree of uniqueness), which means shared physical chunks are
    /// migrated once, not once-per-slot.
    #[test]
    fn migrate_preserves_gid_sharing_structure() {
        let n_kv_head = 1;
        let head_dim = 16;
        let n_tokens = 32;
        let backing = cpu_backing(n_kv_head, head_dim);
        let sealed = seed_and_seal(&backing, n_kv_head, head_dim, n_tokens);

        let unique_src: std::collections::HashSet<i64> =
            sealed.chunks[0].gids.iter().map(|g| g.raw()).collect();

        let cpu = backing.migrate_sealed_to_cpu(&sealed).unwrap();

        let unique_dst: std::collections::HashSet<i64> =
            cpu.chunks[0].gids.iter().map(|g| g.raw()).collect();

        assert_eq!(
            unique_src.len(),
            unique_dst.len(),
            "sharing structure must be preserved across migration"
        );
    }

    /// Round-trip: CPU → GPU migration (simulated as CPU→CPU on a CPU-only backing)
    /// preserves structure the same way as the forward direction.
    #[test]
    fn migrate_to_gpu_preserves_structure() {
        let n_kv_head = 2;
        let head_dim = 32;
        let n_tokens = 64; // exactly two full chunks
        let backing = cpu_backing(n_kv_head, head_dim);
        let sealed = seed_and_seal(&backing, n_kv_head, head_dim, n_tokens);

        // On a CPU backing, "to_gpu" migrates to the same CPU location
        // (no CUDA device present).  This tests the code path without CUDA.
        let cpu_to_gpu = backing.migrate_sealed_to_gpu(&sealed).unwrap();

        assert_eq!(cpu_to_gpu.token_count, n_tokens);
        assert_eq!(cpu_to_gpu.chunks.len(), 2);
        for (orig, mig) in sealed.chunks.iter().zip(cpu_to_gpu.chunks.iter()) {
            assert_eq!(mig.token_count, orig.token_count);
        }
    }

    /// Empty sequence migrates without error.
    #[test]
    fn migrate_empty_sequence() {
        let backing = cpu_backing(2, 16);
        let empty = SealedSequence {
            chunks: vec![],
            token_count: 0,
            chunk_size: CHUNK_SIZE,
            location: ArenaLocation::Cpu,
        };
        let cpu = backing.migrate_sealed_to_cpu(&empty).unwrap();
        assert_eq!(cpu.chunks.len(), 0);
        assert_eq!(cpu.token_count, 0);

        let gpu = backing.migrate_sealed_to_gpu(&empty).unwrap();
        assert_eq!(gpu.chunks.len(), 0);
    }

    // ── Slot relocation ──────────────────────────────────────────────────────
    //
    // `migrate_chunk` moves a slot between tiers; it does not convert formats.
    // Every production caller (hot->warm demote, warm->hot elevate, fork) is
    // format-preserving, and an arena has no format to convert *to* — it is a
    // run of untyped byte slots.

    /// A relocation is byte-verbatim, including a slot whose class stride is
    /// wider than the band that occupies it.
    #[test]
    fn migrate_chunk_copies_a_slot_verbatim() {
        let n_kv_head = 2;
        let head_dim = 128;
        let backing = cpu_backing(n_kv_head, head_dim);
        let sealed = seed_and_seal(&backing, n_kv_head, head_dim, 32);
        let src = sealed.chunks[0].gids.k_gid(0);

        let key = backing
            .inner
            .storage
            .read(|s| s.arena_key(src.arena_idx()))
            .unwrap()
            .expect("source arena exists");
        let stride = key.slot_stride();

        let before = backing
            .inner
            .storage
            .read(|s| {
                s.arena(src.arena_idx())
                    .unwrap()
                    .slot_bytes(src.chunk_idx(), stride)
                    .unwrap()
                    .to_vec1::<u8>()
                    .unwrap()
            })
            .unwrap();

        let dst = backing.migrate_chunk(src.raw(), key).expect("relocation");
        let after = backing
            .inner
            .storage
            .read(|s| {
                s.arena(dst.arena_idx())
                    .unwrap()
                    .slot_bytes(dst.chunk_idx(), stride)
                    .unwrap()
                    .to_vec1::<u8>()
                    .unwrap()
            })
            .unwrap();

        assert_eq!(before, after, "a relocation must not alter a single byte");
    }

    /// **The guard that makes the verbatim copy safe.** Relocating into a
    /// different class would either truncate the band or spill it into a
    /// neighbour, so it is refused rather than attempted.
    #[test]
    fn migrate_chunk_refuses_a_class_change() {
        let n_kv_head = 2;
        let head_dim = 128;
        let backing = cpu_backing(n_kv_head, head_dim);
        let sealed = seed_and_seal(&backing, n_kv_head, head_dim, 32);
        let src = sealed.chunks[0].gids.k_gid(0);

        let key = backing
            .inner
            .storage
            .read(|s| s.arena_key(src.arena_idx()))
            .unwrap()
            .unwrap();
        // Any other rung of the ladder.
        let other = if key.class.index() == 0 {
            SizeClass::at(1)
        } else {
            SizeClass::at(0)
        };
        let wrong = ArenaKey::new(other, key.location);

        let err = backing.migrate_chunk(src.raw(), wrong);
        assert!(
            err.is_err(),
            "a slot move across classes must be refused, not silently truncated"
        );
    }

    #[test]
    fn test_migrate_chunk_invalid_id() {
        let backing = cpu_backing(2, 128);
        let key = ArenaKey::new(SizeClass::at(0), ArenaLocation::Cpu);
        assert!(backing.migrate_chunk(-1, key).is_err());
    }
}
