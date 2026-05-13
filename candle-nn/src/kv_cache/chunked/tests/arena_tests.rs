//! Tests for arena types: Arena, ArenaKey, ArenaStorage, StoragePolicy.

use candle::{DType, Device, Tensor};

use crate::kv_cache::arena_table::ArenaLocation;
use crate::kv_cache::{KvFormat, QuantFormat};

use crate::kv_cache::chunked::{Arena, ArenaKey, ArenaStorage, StoragePolicy};

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== ArenaKey Tests ====================

    mod arena_key_tests {
        use super::*;

        #[test]
        fn test_arena_key_gpu_float() {
            let key = ArenaKey::gpu_float(DType::BF16);

            assert!(key.is_gpu());
            assert!(!key.is_quantized());
            assert_eq!(key.format, KvFormat::Float(DType::BF16));
            assert_eq!(key.location, ArenaLocation::Gpu);
        }

        #[test]
        fn test_arena_key_cpu_float() {
            let key = ArenaKey::cpu_float(DType::F32);

            assert!(!key.is_gpu());
            assert!(!key.is_quantized());
            assert_eq!(key.format, KvFormat::Float(DType::F32));
            assert_eq!(key.location, ArenaLocation::Cpu);
        }

        #[test]
        fn test_arena_key_gpu_quant() {
            let key = ArenaKey::gpu_quant(QuantFormat::Q4_0);

            assert!(key.is_gpu());
            assert!(key.is_quantized());
            assert_eq!(key.format, KvFormat::Quantized(QuantFormat::Q4_0));
            assert_eq!(key.location, ArenaLocation::Gpu);
        }

        #[test]
        fn test_arena_key_new() {
            let key = ArenaKey::uniform(KvFormat::Float(DType::F16), ArenaLocation::Gpu);

            assert_eq!(key.format, KvFormat::Float(DType::F16));
            assert_eq!(key.location, ArenaLocation::Gpu);
        }

        #[test]
        fn test_arena_key_equality() {
            let key1 = ArenaKey::gpu_float(DType::BF16);
            let key2 = ArenaKey::gpu_float(DType::BF16);
            let key3 = ArenaKey::cpu_float(DType::BF16);
            let key4 = ArenaKey::gpu_float(DType::F32);

            assert_eq!(key1, key2);
            assert_ne!(key1, key3); // Different location
            assert_ne!(key1, key4); // Different dtype
        }

        #[test]
        fn test_arena_key_hash() {
            use std::collections::HashSet;

            let mut set = HashSet::new();
            set.insert(ArenaKey::gpu_float(DType::BF16));
            set.insert(ArenaKey::cpu_float(DType::BF16));
            set.insert(ArenaKey::gpu_quant(QuantFormat::Q4_0));

            assert_eq!(set.len(), 3);
            assert!(set.contains(&ArenaKey::gpu_float(DType::BF16)));
        }
    }

    // ==================== StoragePolicy Tests ====================

    mod storage_policy_tests {
        use super::*;

        #[test]
        fn test_storage_policy_gpu_float() {
            let policy = StoragePolicy::GpuFloat(DType::BF16);
            let key = policy.to_arena_key();

            assert_eq!(key.format, KvFormat::Float(DType::BF16));
            assert_eq!(key.location, ArenaLocation::Gpu);
            assert_eq!(policy.active_dtype(), DType::BF16);
        }

        #[test]
        fn test_storage_policy_cpu_float() {
            let policy = StoragePolicy::CpuFloat(DType::F32);
            let key = policy.to_arena_key();

            assert_eq!(key.format, KvFormat::Float(DType::F32));
            assert_eq!(key.location, ArenaLocation::Cpu);
            assert_eq!(policy.active_dtype(), DType::F32);
        }

        #[test]
        fn test_storage_policy_gpu_quant() {
            let policy = StoragePolicy::GpuQuant(QuantFormat::Q4_0);
            let key = policy.to_arena_key();

            assert_eq!(key.format, KvFormat::Quantized(QuantFormat::Q4_0));
            assert_eq!(key.location, ArenaLocation::Gpu);
            // Quant policies use BF16 for active chunks
            assert_eq!(policy.active_dtype(), DType::BF16);
        }

        #[test]
        fn test_storage_policy_cpu_quant() {
            let policy = StoragePolicy::CpuQuant(QuantFormat::Q8_0);
            let key = policy.to_arena_key();

            assert_eq!(key.format, KvFormat::Quantized(QuantFormat::Q8_0));
            assert_eq!(key.location, ArenaLocation::Cpu);
            assert_eq!(policy.active_dtype(), DType::BF16);
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

        fn create_float_arena(dtype: DType, location: ArenaLocation, index: usize) -> Arena {
            let device = Device::Cpu;
            let data = Tensor::zeros((64, 8, 16, 64), dtype, &device).unwrap();
            Arena::Float {
                data,
                dtype,
                location,
                index,
            }
        }

        #[test]
        fn test_arena_float_creation() {
            let arena = create_float_arena(DType::BF16, ArenaLocation::Cpu, 0);

            assert_eq!(arena.index(), 0);
            assert!(matches!(arena.format(), KvFormat::Float(DType::BF16)));
        }

        #[test]
        fn test_arena_kv_format_float() {
            let arena = create_float_arena(DType::F32, ArenaLocation::Cpu, 0);
            let format = arena.format();

            assert_eq!(format, KvFormat::Float(DType::F32));
        }

        #[test]
        fn test_arena_location() {
            let arena_cpu = create_float_arena(DType::BF16, ArenaLocation::Cpu, 0);
            let arena_gpu = create_float_arena(DType::BF16, ArenaLocation::Gpu, 1);

            assert_eq!(arena_cpu.location(), ArenaLocation::Cpu);
            assert_eq!(arena_gpu.location(), ArenaLocation::Gpu);
        }

        #[test]
        fn test_arena_float_kv_access() {
            let arena = create_float_arena(DType::BF16, ArenaLocation::Cpu, 0);

            let result = arena.float_data();
            assert!(result.is_ok());
            let data = result.unwrap();
            assert_eq!(data.dims(), &[64, 8, 16, 64]);
        }

        #[test]
        fn test_arena_as_float_k_v() {
            let arena = create_float_arena(DType::BF16, ArenaLocation::Cpu, 0);

            assert!(arena.as_float_data().is_some());
            assert!(arena.as_quantized_data().is_none());
        }

        #[test]
        fn test_arena_key_from_arena() {
            let arena = create_float_arena(DType::F16, ArenaLocation::Gpu, 0);
            let key = arena.arena_key();

            assert_eq!(key.format, KvFormat::Float(DType::F16));
            assert_eq!(key.location, ArenaLocation::Gpu);
        }

        #[test]
        fn test_arena_to_arena_entry_cpu() {
            let arena = create_float_arena(DType::BF16, ArenaLocation::Cpu, 0);
            let entry = arena.to_arena_entry();

            // CPU arenas have zero pointers
            assert_eq!(entry.k_ptr, 0);
            assert_eq!(entry.v_ptr, 0);
        }
    }

    // ==================== ArenaStorage Tests ====================

    mod arena_storage_tests {
        use super::*;

        #[test]
        fn test_arena_storage_new_float() {
            let storage = ArenaStorage::new(KvFormat::Float(DType::BF16), KvFormat::Float(DType::BF16), ArenaLocation::Gpu);

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
            let storage = ArenaStorage::new(KvFormat::Float(DType::BF16), KvFormat::Float(DType::BF16), ArenaLocation::Gpu);

            assert_eq!(storage.arena_count().unwrap(), 0);
        }

        #[test]
        fn test_arena_storage_truncate() {
            let storage = ArenaStorage::new(KvFormat::Float(DType::BF16), KvFormat::Float(DType::BF16), ArenaLocation::Cpu);

            // Add some arenas manually for test using write closure
            storage
                .write(|s| {
                    let data = Tensor::zeros((64, 8, 16, 64), DType::BF16, &Device::Cpu).unwrap();
                    s.arenas_mut().insert(0, Arena::Float {
                        data: data.clone(),
                        dtype: DType::BF16,
                        location: ArenaLocation::Cpu,
                        index: 0,
                    });
                    s.arenas_mut().insert(1, Arena::Float {
                        data,
                        dtype: DType::BF16,
                        location: ArenaLocation::Cpu,
                        index: 1,
                    });
                })
                .unwrap();

            assert_eq!(storage.arena_count().unwrap(), 2);

            storage.truncate_arenas(1).unwrap();
            assert_eq!(storage.arena_count().unwrap(), 1);

            storage.truncate_arenas(0).unwrap();
            assert_eq!(storage.arena_count().unwrap(), 0);
        }
    }
}
