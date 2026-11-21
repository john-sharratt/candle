//! How many bytes of KV the reservation can still hold, counted exactly in free
//! regions.
//!
//! The count is free regions times the region size, and a device with no
//! reservation reports unknown rather than zero. Ground under a standing wave
//! tier is counted, since it returns before the next forward's claims. Ground
//! above the weight floor is not, because it returns only if the weight side
//! concedes it, and counting it would promise memory admission cannot get back.

use super::*;

/// How many bytes of KV the reservation can still hold: free regions × the
/// region size.
///
/// An **exact count**, not an estimate. It replaced three overlapping
/// approximations — driver headroom, the pool's reserved-but-free gap, and
/// `init_free − pool_used` — that disagreed with each other precisely when it
/// mattered: the pool's reuse gap read as available while a contiguous arena
/// could not fit in any single free block, so admission kept widening into a
/// wall. Under the reservation the question has one answer, and asking it costs
/// a mutex and an integer.
///
/// `None` when this device has no reservation yet (non-CUDA, or before the
/// first KV cache exists) — callers treat that as "unknown", never as zero.
///
/// Counts the **tier's** blocked ground and not the **weight side's**, which is
/// the whole of the difference between an admission that fits and one that OOMs.
///
/// Both halves of `blocked` are unowned ground above the region ceiling, and
/// they behave nothing alike. Ground the current wave's transient tier stands on
/// comes back in phase 0 of the next forward, before phase 1 claims against this
/// number — so counting it is not merely safe but necessary, or every standing
/// tier reads as KV pressure between forwards and admission starves itself
/// against ground it is guaranteed to get back. Ground **above the weight
/// floor** belongs to the weight zone and comes back only if the weight side is
/// asked to concede it, which is legal only between forwards and never happens
/// on its own.
///
/// Counting both was reachable-OOM. Measured on the 27B at 20 contexts with the
/// layer zone grown to full residency: `free 24 | blocked 58`, every one of the
/// 58 the zone's, so this promised 1,312 MiB against 384 MiB claimable. The wave
/// was admitted on that promise and a KV slot-state upload inside it took
/// `CUDA_ERROR_OUT_OF_MEMORY` — at which point nothing could help, because
/// `set_weight_floor` refuses to move the boundary while a wave generation is
/// open. Admission is the last place the answer can still be "narrower".
#[cfg(feature = "cuda")]
pub fn vram_budget_available(device: &Device) -> Option<usize> {
    let candle::DeviceLocation::Cuda { gpu_id } = device.location() else {
        return None;
    };
    region_pool::region_stats(gpu_id).map(|s| (s.free + s.blocked) * region_pool::REGION_BYTES)
}

#[cfg(not(feature = "cuda"))]
pub fn vram_budget_available(_device: &Device) -> Option<usize> {
    None
}
