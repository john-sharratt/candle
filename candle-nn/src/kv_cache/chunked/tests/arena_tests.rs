//! Tests for arena types: Arena, ArenaKey, ArenaStorage, StoragePolicy.

use candle::{DType, Device, Tensor};

use crate::kv_cache::arena_table::ArenaLocation;
use crate::kv_cache::{KvFormat, QuantFormat};

use crate::kv_cache::chunked::size_class::{class_for_format, SizeClass};
use crate::kv_cache::chunked::{Arena, ArenaKey, ArenaStorage, StoragePolicy};

#[cfg(test)]
mod tests {
    use super::*;

    /// The production palette-4 geometry: `head_dim 128 / N_PALETTE 4 = 32`,
    /// so a slot is `CHUNK_SIZE(32) * 32 = 1024` elements.
    const ELEMS: usize = 1024;

    fn key(format: KvFormat, location: ArenaLocation) -> ArenaKey {
        ArenaKey::for_format(format, ELEMS, location).expect("format is covered by the ladder")
    }

    // ==================== ArenaKey Tests ====================

    mod arena_key_tests {
        use super::*;

        /// **The property the whole initiative exists to obtain.** Two chunks
        /// of *different formats* whose payloads fit the same rung allocate
        /// from the same key — so they share an arena, share a free list, and
        /// a slot freed by one is immediately usable by the other.
        ///
        /// Under per-format arenas every one of these assertions was false.
        #[test]
        fn formats_sharing_a_class_share_a_key() {
            let same = |a: KvFormat, b: KvFormat| {
                assert_eq!(
                    key(a, ArenaLocation::Gpu),
                    key(b, ArenaLocation::Gpu),
                    "{a:?} and {b:?} share a size class and must share a key"
                );
            };
            // The whole sub-320 B tail collapses onto one rung.
            same(
                KvFormat::Quantized(QuantFormat::Q0),
                KvFormat::Quantized(QuantFormat::Q2_0),
            );
            // The dominant sealed pair. (Q8_0 is NOT here: at 1088 B it has
            // its own rung, so it shares with nothing — see
            // `size_class::tests::every_format_above_the_catch_all_lands_exactly`.)
            same(
                KvFormat::Quantized(QuantFormat::Q8_1),
                KvFormat::Quantized(QuantFormat::Q8_KS),
            );
            same(
                KvFormat::Quantized(QuantFormat::Q4_1),
                KvFormat::Quantized(QuantFormat::Q4_KS),
            );
            // Active-K raw capture and F32 share the top rung.
            same(
                KvFormat::Quantized(QuantFormat::R16),
                KvFormat::Float(DType::F32),
            );
        }

        /// Formats in *different* classes still get different keys — the
        /// collapse above must not have flattened everything into one pool.
        #[test]
        fn formats_in_different_classes_get_different_keys() {
            assert_ne!(
                key(KvFormat::Quantized(QuantFormat::Q0), ArenaLocation::Gpu),
                key(KvFormat::Quantized(QuantFormat::Q8_0), ArenaLocation::Gpu),
            );
            assert_ne!(
                key(KvFormat::Float(DType::F16), ArenaLocation::Gpu),
                key(KvFormat::Float(DType::F32), ArenaLocation::Gpu),
            );
        }

        /// Location is still part of the key: hot and warm never share a pool
        /// however identical their strides.
        #[test]
        fn location_still_separates_keys() {
            let f = KvFormat::Float(DType::BF16);
            let gpu = key(f, ArenaLocation::Gpu);
            let cpu = key(f, ArenaLocation::Cpu);
            assert_ne!(gpu, cpu);
            assert!(gpu.is_gpu());
            assert!(!cpu.is_gpu());
            assert_eq!(gpu.class, cpu.class, "same format, so same stride");
        }

        /// A key's stride and slot count come from its class alone — nothing
        /// about the format that happened to create it survives.
        #[test]
        fn key_reports_its_class_geometry() {
            let k = key(KvFormat::Quantized(QuantFormat::Q8_0), ArenaLocation::Gpu);
            let class = class_for_format(KvFormat::Quantized(QuantFormat::Q8_0), ELEMS).unwrap();
            assert_eq!(k.class, class);
            assert_eq!(k.slot_stride(), class.bytes());
            assert_eq!(k.chunks(), class.chunks_per_region());
            // Q8_0's 1088 B payload has a rung of its own, so the stride is
            // the payload: no bytes are read that carry nothing.
            assert_eq!(k.slot_stride(), 1088);
        }

        #[test]
        fn keys_hash_by_class_and_location() {
            use std::collections::HashSet;

            let mut set = HashSet::new();
            set.insert(key(KvFormat::Float(DType::BF16), ArenaLocation::Gpu));
            set.insert(key(KvFormat::Float(DType::BF16), ArenaLocation::Cpu));
            set.insert(key(
                KvFormat::Quantized(QuantFormat::Q0),
                ArenaLocation::Gpu,
            ));
            // BF16 shares F16's class, so the CPU/GPU BF16 pair above already
            // covers it. Q8_KS (1152 B) and Q8_0 (1088 B) are distinct rungs
            // and each adds one.
            set.insert(key(
                KvFormat::Quantized(QuantFormat::Q8_KS),
                ArenaLocation::Gpu,
            ));
            set.insert(key(
                KvFormat::Quantized(QuantFormat::Q8_0),
                ArenaLocation::Gpu,
            ));

            assert_eq!(set.len(), 5);
            assert!(set.contains(&key(KvFormat::Float(DType::F16), ArenaLocation::Gpu)));
        }

        /// Every record stride's slot count must stay **below** `GID_STRIDE`.
        ///
        /// A gid packs `arena_idx * GID_STRIDE + chunk_idx`, so a pool whose arenas hold
        /// `GID_STRIDE` slots or more lets the last chunk index collide with the next
        /// arena's namespace — the gid decodes to a different arena entirely, which is a
        /// silent wrong-address read, not a fault. At 16 MiB regions and `1 << 16` that
        /// makes 256 B the first stride that breaks, which is why `RECORD_STRIDES`
        /// starts at 512. Asserted rather than commented because the two constants that
        /// decide it live in different modules.
        #[test]
        fn record_strides_fit_the_gid_namespace() {
            use crate::kv_cache::chunked::arena::RECORD_STRIDES;
            use crate::kv_cache::chunked::size_class::GID_STRIDE;

            assert!(!RECORD_STRIDES.is_empty());
            for stride in RECORD_STRIDES {
                assert!(
                    stride.is_multiple_of(8),
                    "stride {stride} is not 8-aligned, so a slot base need not be either \
                     — and the kernel stores band pointers as uint64_t",
                );
                assert!(
                    stride >= RECORD_STRIDES[0],
                    "stride {stride} is below the floor the gid namespace allows",
                );
                let key = ArenaKey::for_record_stride(ArenaLocation::Gpu, stride);
                assert_eq!(key.slot_stride(), stride);
                assert!(
                    key.chunks() > 0,
                    "stride {stride} leaves no slots in a region",
                );
                assert!(
                    key.chunks() < GID_STRIDE,
                    "stride {stride} gives {} slots per arena, which reaches GID_STRIDE \
                     ({GID_STRIDE}) — chunk indices would collide with the next arena",
                    key.chunks(),
                );
            }
        }

        /// A key built from a record *size* and one built from its *stride* must be the
        /// same key, or a caller looks up a pool the eager table never created — which
        /// is exactly the `register_arena: missing preallocated pool for key` panic.
        #[test]
        fn a_record_size_and_its_stride_agree_on_the_key() {
            // 1344 is the production GQA record (8 heads x HD128 x 4 bands); it lands on
            // the 1,536 rung with 12.5% pad, where a power-of-two ladder gave 2,048 and 34%.
            for (bytes, want) in [
                (1usize, 512usize),
                (512, 512),
                (513, 768),
                (1344, 1536),
                (1537, 2048),
            ] {
                let by_size = ArenaKey::for_records(ArenaLocation::Gpu, bytes);
                let by_stride = ArenaKey::for_record_stride(ArenaLocation::Gpu, want);
                assert_eq!(
                    by_size, by_stride,
                    "a {bytes} B record must resolve to the {want} B rung",
                );
                assert!(
                    by_size.slot_stride() >= bytes,
                    "the slot must hold the record",
                );
            }
        }

        /// A record key and a band key never collide, however their classes compare —
        /// which is what keeps record arenas out of every walk that enumerates
        /// `SizeClass::all()` against a band key, the compaction census included.
        #[test]
        fn a_record_key_is_never_a_band_key() {
            use std::collections::HashSet;

            let rec = ArenaKey::for_records(ArenaLocation::Gpu, 1344);
            let mut band_keys = HashSet::new();
            for class in SizeClass::all() {
                band_keys.insert(ArenaKey::new(class, ArenaLocation::Gpu));
            }
            assert!(
                !band_keys.contains(&rec),
                "the record key collides with a band pool, so the census would walk it",
            );
        }

        /// A format the ladder does not cover is a configuration error, and it
        /// is reported as one rather than silently landing in the top class.
        #[test]
        fn an_uncovered_geometry_is_an_error() {
            // 64 KiB of F32 is 256 KiB per slot — far past the 4096 B top rung.
            let err = ArenaKey::for_format(KvFormat::Float(DType::F32), 65_536, ArenaLocation::Gpu);
            assert!(err.is_err(), "an uncovered format must not resolve");
        }
    }

    // ==================== StoragePolicy Tests ====================

    mod storage_policy_tests {
        use super::*;

        #[test]
        fn policy_reports_its_target_format_and_location() {
            let cases = [
                (
                    StoragePolicy::GpuFloat(DType::BF16),
                    KvFormat::Float(DType::BF16),
                    ArenaLocation::Gpu,
                    false,
                    DType::BF16,
                ),
                (
                    StoragePolicy::CpuFloat(DType::F32),
                    KvFormat::Float(DType::F32),
                    ArenaLocation::Cpu,
                    false,
                    DType::F32,
                ),
                (
                    StoragePolicy::GpuQuant(QuantFormat::Q4_0),
                    KvFormat::Quantized(QuantFormat::Q4_0),
                    ArenaLocation::Gpu,
                    true,
                    // Quant policies keep active chunks in BF16.
                    DType::BF16,
                ),
                (
                    StoragePolicy::CpuQuant(QuantFormat::Q8_0),
                    KvFormat::Quantized(QuantFormat::Q8_0),
                    ArenaLocation::Cpu,
                    true,
                    DType::BF16,
                ),
            ];
            for (policy, format, location, quantized, active) in cases {
                assert_eq!(policy.target_format(), format, "{policy:?}");
                assert_eq!(policy.location(), location, "{policy:?}");
                assert_eq!(policy.is_quantized(), quantized, "{policy:?}");
                assert_eq!(policy.active_dtype(), active, "{policy:?}");
            }
        }

        #[test]
        fn test_storage_policy_default() {
            let policy = StoragePolicy::default();

            match policy {
                StoragePolicy::GpuFloat(DType::BF16) => {}
                _ => panic!("default should be GpuFloat(BF16)"),
            }
        }
    }

    // ==================== Arena Tests ====================

    mod arena_tests {
        use super::*;

        /// A CPU slab of `class`, sized exactly as the allocator would make it.
        fn slab(class: SizeClass, location: ArenaLocation, index: usize) -> Arena {
            let bytes = class.chunks_per_region() * class.bytes();
            let data = Tensor::zeros(bytes, DType::U8, &Device::Cpu).unwrap();
            Arena::new(data, ArenaKey::new(class, location), index)
        }

        fn small() -> SizeClass {
            SizeClass::at(0)
        }

        #[test]
        fn arena_reports_its_class_geometry() {
            let class = small();
            let arena = slab(class, ArenaLocation::Cpu, 7);

            assert_eq!(arena.index(), 7);
            assert_eq!(arena.class(), class);
            assert_eq!(arena.slot_stride(), class.bytes());
            assert_eq!(arena.chunks(), class.chunks_per_region());
            assert_eq!(arena.location(), ArenaLocation::Cpu);
            assert_eq!(arena.arena_key(), ArenaKey::new(class, ArenaLocation::Cpu));
        }

        /// **Every arena is allocatable.** Under the old Float/Quantized
        /// duality only float arenas were, because a quantized arena's slots
        /// were written by the convert kernel rather than claimed by the
        /// allocator. A slot is a slot now.
        #[test]
        fn every_arena_is_allocatable() {
            assert!(slab(small(), ArenaLocation::Cpu, 0).is_allocatable());
            assert!(slab(SizeClass::at(6), ArenaLocation::Gpu, 1).is_allocatable());
        }

        /// A CPU arena has no device pointer, and asking for one is `None`
        /// rather than a bogus zero-based address.
        #[test]
        fn a_cpu_arena_has_no_device_pointer() {
            let arena = slab(small(), ArenaLocation::Cpu, 0);
            assert!(arena.base_ptr().is_none());
            assert!(arena.slot_ptr(3).is_none());
            assert!(arena.chunk_copy_span(3).is_none());
            assert_eq!(arena.gpu_memory_bytes(), 0, "CPU arenas hold no VRAM");
        }

        /// **Payload and stride are different numbers, and the bounds check
        /// knows it.** A read of at most the stride is fine; anything past it
        /// would run into the next tenant and is refused
        /// (`docs/archived/arena_unification.md` invariant 8).
        #[test]
        fn a_slot_read_cannot_escape_its_slot() {
            let class = small();
            let arena = slab(class, ArenaLocation::Cpu, 0);

            assert!(arena.slot_bytes(0, class.bytes()).is_ok());
            assert!(arena.slot_bytes(0, 32).is_ok(), "a short payload is normal");
            assert!(
                arena.slot_bytes(0, class.bytes() + 1).is_err(),
                "a read past the stride must be refused"
            );
            assert!(
                arena.slot_bytes(class.chunks_per_region(), 32).is_err(),
                "a chunk index past the arena must be refused"
            );
        }

        /// Slots are addressed by the **class stride**, so writing one slot
        /// leaves its neighbours untouched — the property a payload-derived
        /// offset would break the moment a stride exceeded a format's bytes.
        #[test]
        fn slots_do_not_overlap() {
            let class = small();
            let mut arena = slab(class, ArenaLocation::Cpu, 0);
            let stride = class.bytes();

            let ones = Tensor::ones(stride, DType::U8, &Device::Cpu).unwrap();
            arena.write_slot_bytes(1, &ones).unwrap();

            let read = |i: usize| {
                arena
                    .slot_bytes(i, stride)
                    .unwrap()
                    .to_vec1::<u8>()
                    .unwrap()
            };
            assert!(read(0).iter().all(|&b| b == 0), "slot 0 must be untouched");
            assert!(read(1).iter().all(|&b| b == 1), "slot 1 must be written");
            assert!(read(2).iter().all(|&b| b == 0), "slot 2 must be untouched");
        }

        /// **A host slot copies out its own payload bytes, read in place.** Slot
        /// 1 holds a known pattern between two other tenants; the copy of its
        /// first 6 bytes is exactly those bytes, and a copy past the stride is
        /// refused as `slot_bytes` refuses it.
        #[test]
        fn a_host_slot_copies_its_payload_in_place() {
            let class = small();
            let mut arena = slab(class, ArenaLocation::Cpu, 0);
            let stride = class.bytes();
            let fill = |v: u8| Tensor::full(v, stride, &Device::Cpu).unwrap();
            arena.write_slot_bytes(0, &fill(0xAA)).unwrap();
            arena.write_slot_bytes(2, &fill(0xCC)).unwrap();
            let pattern = Tensor::new(&[1u8, 2, 3, 4, 5, 6], &Device::Cpu).unwrap();
            arena.write_slot_bytes(1, &pattern).unwrap();

            let mut dst = [0u8; 6];
            arena.copy_slot_bytes(1, &mut dst).unwrap();
            assert_eq!(dst, [1, 2, 3, 4, 5, 6]);

            let mut too_long = vec![0u8; stride + 1];
            assert!(arena.copy_slot_bytes(1, &mut too_long).is_err());
        }

        /// **A host slot takes a byte write in place**, at the slot's own offset,
        /// leaving its neighbours and the rest of its stride alone — and a write
        /// past the stride is refused as `write_slot_bytes` refuses it.
        #[test]
        fn a_host_slot_takes_bytes_in_place() {
            let class = small();
            let mut arena = slab(class, ArenaLocation::Cpu, 0);
            let stride = class.bytes();
            let fill = |v: u8| Tensor::full(v, stride, &Device::Cpu).unwrap();
            arena.write_slot_bytes(0, &fill(0xAA)).unwrap();
            arena.write_slot_bytes(1, &fill(0xBB)).unwrap();
            arena.write_slot_bytes(2, &fill(0xCC)).unwrap();

            arena.write_slot_from_bytes(1, &[1, 2, 3]).unwrap();

            let mut slot = vec![0u8; stride];
            arena.copy_slot_bytes(1, &mut slot).unwrap();
            assert_eq!(&slot[..3], &[1, 2, 3]);
            assert!(slot[3..].iter().all(|&b| b == 0xBB));
            arena.copy_slot_bytes(0, &mut slot).unwrap();
            assert!(slot.iter().all(|&b| b == 0xAA));
            arena.copy_slot_bytes(2, &mut slot).unwrap();
            assert!(slot.iter().all(|&b| b == 0xCC));

            assert!(arena
                .write_slot_from_bytes(1, &vec![0u8; stride + 1])
                .is_err());
        }

        /// **Zero-on-recycle covers the whole stride, not the payload.** The
        /// next tenant may be any format that fits, so a partial wipe would
        /// leave the previous tenant's bytes readable past the new one's
        /// payload (invariant 4).
        #[test]
        fn recycling_a_slot_zeroes_its_full_stride() {
            let class = small();
            let mut arena = slab(class, ArenaLocation::Cpu, 0);
            let stride = class.bytes();

            let ones = Tensor::ones(stride, DType::U8, &Device::Cpu).unwrap();
            arena.write_slot_bytes(0, &ones).unwrap();
            arena.zero_chunk_at(0).unwrap();

            let bytes = arena
                .slot_bytes(0, stride)
                .unwrap()
                .to_vec1::<u8>()
                .unwrap();
            assert!(
                bytes.iter().all(|&b| b == 0),
                "every byte of the slot must be zero, including the pad"
            );
        }

        /// A typed write and a typed read of the same slot round-trip, and the
        /// dtype comes from the caller (the band's tag) rather than the arena.
        #[test]
        fn a_slot_round_trips_a_typed_band() {
            // 4096 B holds 2048 F16 elements; write 8 of them at offset 4.
            let class = SizeClass::at(6);
            let mut arena = slab(class, ArenaLocation::Cpu, 0);

            let vals: Vec<half::f16> = (0..8)
                .map(|i| half::f16::from_f32(i as f32 + 0.5))
                .collect();
            let band = Tensor::from_vec(vals.clone(), 8, &Device::Cpu).unwrap();
            arena.write_slot_typed(2, 4, &band).unwrap();

            let back = arena.read_slot_typed(2, DType::F16, 12).unwrap();
            let got = back.to_vec1::<half::f16>().unwrap();
            assert_eq!(&got[4..12], &vals[..], "the band must read back exactly");
            assert!(
                got[..4].iter().all(|v| *v == half::f16::from_f32(0.0)),
                "bytes before the offset must be untouched"
            );
        }

        /// The label is the class, because that is all an arena is. A format
        /// label would be a claim the arena cannot support.
        #[test]
        fn the_label_names_the_class() {
            let arena = slab(SizeClass::at(0), ArenaLocation::Cpu, 0);
            assert_eq!(arena.format_label(), "class320");
        }
    }

    // ==================== ArenaStorage Tests ====================

    mod arena_storage_tests {
        use super::*;

        #[test]
        fn test_arena_storage_new_float() {
            let storage = ArenaStorage::new(
                KvFormat::Float(DType::BF16),
                KvFormat::Float(DType::BF16),
                ArenaLocation::Gpu,
            );

            assert_eq!(storage.k_format(), KvFormat::Float(DType::BF16));
            assert!(!storage.is_quantized());
            assert_eq!(storage.default_location(), ArenaLocation::Gpu);
            assert_eq!(storage.dtype(), Some(DType::BF16));
        }

        #[test]
        fn test_arena_storage_new_quantized() {
            let storage = ArenaStorage::new(
                KvFormat::Quantized(QuantFormat::Q4_0),
                KvFormat::Quantized(QuantFormat::Q4_0),
                ArenaLocation::Cpu,
            );

            assert_eq!(storage.k_format(), KvFormat::Quantized(QuantFormat::Q4_0));
            assert!(storage.is_quantized());
            assert_eq!(storage.default_location(), ArenaLocation::Cpu);
            assert_eq!(storage.dtype(), None);
        }

        #[test]
        fn test_arena_storage_arena_count() {
            let storage = ArenaStorage::new(
                KvFormat::Float(DType::BF16),
                KvFormat::Float(DType::BF16),
                ArenaLocation::Gpu,
            );

            assert_eq!(storage.arena_count().unwrap(), 0);
        }

        #[test]
        fn test_arena_storage_truncate() {
            let storage = ArenaStorage::new(
                KvFormat::Float(DType::BF16),
                KvFormat::Float(DType::BF16),
                ArenaLocation::Cpu,
            );
            let class = SizeClass::at(0);
            let bytes = class.chunks_per_region() * class.bytes();

            storage
                .write(|s| {
                    for idx in 0..2 {
                        let data = Tensor::zeros(bytes, DType::U8, &Device::Cpu).unwrap();
                        s.arenas_mut().insert(
                            idx,
                            Arena::new(data, ArenaKey::new(class, ArenaLocation::Cpu), idx),
                        );
                    }
                })
                .unwrap();

            assert_eq!(storage.arena_count().unwrap(), 2);

            storage.truncate_arenas(1).unwrap();
            assert_eq!(storage.arena_count().unwrap(), 1);

            storage.truncate_arenas(0).unwrap();
            assert_eq!(storage.arena_count().unwrap(), 0);
        }
    }

    /// **Widening a key by size class always produces a BAND key — which is why a record
    /// claim must never be promoted.**
    ///
    /// `alloc::stamp_region_promoting` answers a claim it cannot get a region for by
    /// widening `key.class` and rebuilding the key with `ArenaKey::new`. That constructor
    /// is `ArenaKind::Band`, so a *record* claim promoted this way comes back holding a
    /// slot in a band arena — and the record's bytes are then written on top of live K/V,
    /// which raises nothing: the caller's own check is that the slot is at least
    /// `record_bytes`, and a 1,344 B record fits a 2 KiB band slot perfectly.
    ///
    /// A record's size is model geometry, not a rung of the ladder, so there is no wider
    /// class it belongs in. This pins the reason the allocator refuses rather than
    /// promotes; it cost one to three sessions of eight on the Flash-Next engine probe,
    /// intermittently, once a compaction started minting a record per relocated chunk.
    #[test]
    fn widening_a_key_by_class_always_yields_a_band_key() {
        use crate::kv_cache::chunked::arena::ArenaKind;

        let record = ArenaKey::for_records(ArenaLocation::Gpu, 1344);
        assert!(
            matches!(record.kind, ArenaKind::Record { .. }),
            "a record key is not a band key",
        );

        // Exactly what the promotion loop would build for it.
        let wider = record.class.promote().unwrap_or(record.class);
        let promoted = ArenaKey::new(wider, record.location);
        assert!(
            matches!(promoted.kind, ArenaKind::Band),
            "promotion cannot preserve the record kind, so the allocator must refuse it",
        );
        assert_ne!(
            record, promoted,
            "and the promoted key names a different pool entirely",
        );
    }
}
