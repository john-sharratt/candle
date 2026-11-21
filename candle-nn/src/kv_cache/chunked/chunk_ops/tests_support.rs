//! Provides the backing constructors and byte readers the migration tests share.
//!
//! CPU and CUDA backings are built with a single float arena, and the sealing
//! helpers seed known patterns. A device test holds the crate's GPU lock, because
//! backings draw on one shared pool and an overlapping test would fault the
//! context for whichever test ran next. The byte reader takes each band's length
//! from its tag rather than from the arena stride.

use super::*;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::kv_cache::chunked::ChunkedKvBacking;

    use candle::{DType, Device, Tensor};

    // ── Batched-async migrate tests (GPU-only) ──────────────────────────────
    //
    // These exercise `migrate_sealed_to_cpu_batch_async` and the warm lift
    // (`warm_lift::lift_warm_to_gpu`) — the production hot↔warm path. They
    // require a CUDA device; on a CPU-only build each test silently returns
    // early.

    /// Build a CPU-backed `ChunkedKvBacking` with a single float BF16 arena.
    pub(in super::super) fn cpu_backing(n_kv_head: usize, head_dim: usize) -> ChunkedKvBacking {
        ChunkedKvBacking::new(4, n_kv_head, head_dim, DType::BF16, &Device::Cpu, 256).unwrap()
    }

    /// Seed `n_tokens` BF16 ones into slot 0, return a sealed snapshot.
    pub(in super::super) fn seed_and_seal(
        backing: &ChunkedKvBacking,
        n_kv_head: usize,
        head_dim: usize,
        n_tokens: usize,
    ) -> SealedSequence {
        let slot = backing.alloc_sequence().unwrap();
        backing.ensure_for_offset(slot, 0, n_tokens).unwrap();
        let k = Tensor::ones(
            (1, n_kv_head, n_tokens, head_dim),
            DType::BF16,
            &Device::Cpu,
        )
        .unwrap();
        let v = k.clone();
        backing.write_contiguous(slot, 0, &k, &v).unwrap();
        backing.set_len(slot, n_tokens);
        backing.record_turn(slot).unwrap()
    }

    //
    // Byte-identity is checked end-to-end: a known pattern is seeded on
    // GPU, migrated to RAM, then back to VRAM, and the round-tripped
    // bytes must equal the original.

    /// A CUDA device **and** the crate-wide GPU serialisation guard, or `None`
    /// when there is no device.
    ///
    /// The guard is returned alongside the device rather than taken separately
    /// so it cannot be forgotten. These tests build `ChunkedKvBacking`s, and a
    /// backing is not an independent value: it draws on one process-global
    /// region pool carved from a single reservation
    /// (`crate::kv_cache::chunked::gpu_test_lock`). A test holding a device
    /// without the lock races every other GPU test in the crate, and the way
    /// that surfaces is `CUDA_ERROR_ILLEGAL_ADDRESS` from whichever one loses —
    /// which then poisons the context for every test scheduled after it, so the
    /// reported failure is rarely the guilty one.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn cuda_device_or_skip(
    ) -> Option<(Device, crate::kv_cache::chunked::gpu_test_lock::GpuGuard)> {
        let guard = crate::kv_cache::chunked::gpu_test_lock::gpu_serial();
        match Device::cuda_if_available(0) {
            Ok(d @ Device::Cuda(_)) => Some((d, guard)),
            _ => None,
        }
    }

    /// Build a CUDA-backed `ChunkedKvBacking` with a BF16 float arena.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn cuda_backing(
        device: &Device,
        n_kv_head: usize,
        head_dim: usize,
    ) -> ChunkedKvBacking {
        ChunkedKvBacking::new(4, n_kv_head, head_dim, DType::BF16, device, 256).unwrap()
    }

    /// Seed `n_tokens` of a deterministic BF16 pattern starting at
    /// `pattern_base` and return a sealed snapshot. The pattern is
    /// `bf16::from_f32((pattern_base + i) as f32 * 0.001)` per element,
    /// which gives each element a distinct value so byte-identity
    /// checks catch any chunk-swap or offset bug.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn seed_and_seal_pattern(
        backing: &ChunkedKvBacking,
        device: &Device,
        n_kv_head: usize,
        head_dim: usize,
        n_tokens: usize,
        pattern_base: u32,
    ) -> SealedSequence {
        use half::bf16;
        let slot = backing.alloc_sequence().unwrap();
        backing.ensure_for_offset(slot, 0, n_tokens).unwrap();
        let total = n_kv_head * n_tokens * head_dim;
        let data: Vec<bf16> = (0..total)
            .map(|i| bf16::from_f32(((pattern_base as usize + i) as f32) * 0.001))
            .collect();
        let k = Tensor::from_vec(
            data.clone(),
            (1, n_kv_head, n_tokens, head_dim),
            &Device::Cpu,
        )
        .unwrap()
        .to_device(device)
        .unwrap();
        let v = k.clone();
        backing.write_contiguous(slot, 0, &k, &v).unwrap();
        backing.set_len(slot, n_tokens);
        backing.record_turn(slot).unwrap()
    }

    /// Walk every unique GID in `sealed` (which must be CPU-resident),
    /// extract the chunk slot's bytes via `read_chunk_into_pinned_bytes`,
    /// and concatenate in iteration order. Used to compare bytes
    /// before/after a round-trip.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn bytes_of_cpu_sealed(
        backing: &ChunkedKvBacking,
        sealed: &SealedSequence,
    ) -> Vec<u8> {
        let arena_chunks = GID_STRIDE;
        let elems = backing.inner.elems_per_chunk();
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        backing
            .inner
            .storage
            .read(|s| -> candle::Result<()> {
                for chunk in &sealed.chunks {
                    // The band's own tag gives the payload length. Reading the
                    // arena's slot stride instead would compare pad bytes too,
                    // which is exactly the symmetric error a round-trip test
                    // cannot otherwise detect (audit A7).
                    for (gid, tag) in chunk.bands() {
                        let raw = gid.raw();
                        if !seen.insert(raw) {
                            continue;
                        }
                        let arena_idx = (raw as usize) / arena_chunks;
                        let chunk_idx = (raw as usize) % arena_chunks;
                        let arena = s.arenas().get(&arena_idx).unwrap();
                        let len = payload_bytes_for_tag(tag, elems).unwrap();
                        let mut buf = vec![0u8; len];
                        read_chunk_into_pinned_bytes(arena, chunk_idx, &mut buf).unwrap();
                        out.extend(buf);
                    }
                }
                Ok(())
            })
            .unwrap()
            .unwrap();
        out
    }
}
