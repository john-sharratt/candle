//! Manages the GPU KV arenas inside the device reservation: releasing empty
//! ones, compacting the pools, and reporting how much ground is fragmented.
//!
//! Under the reservation an arena's region stays mapped at its address, and only
//! the free list that names it changes. Releasing an arena is therefore a
//! free-list push, and the one wait still needed is paid when a region is
//! claimed again. Compaction packs every backing, not one, because block tables
//! are per layer, and a pass that moved a chunk in only one of them would give
//! wrong attention on the others. The sparse-arena figure is the cheap gate for
//! a compaction pass, since the exact census is the expensive walk it decides on.

use super::*;

impl BatchedInferenceSession {
    /// Sweep fully-empty KV arenas across all backings, returning their regions
    /// to the free list. Returns arenas freed.
    ///
    /// Runs per-wave, and it is cheap: an arena release is a free-list push and
    /// nothing more. It used to be neither. Releasing an arena unmapped its
    /// slab, so this took the process-global arena-topology write lock — the
    /// migrate's per-head table captured raw base pointers of every arena and
    /// dereferenced them from a kernel with no lock held — and then paid a full
    /// `device.synchronize()` to retire trailing kernels before the unmap.
    /// A whole-device sync, every wave, on the sweep path.
    ///
    /// Under the reservation nothing unmaps: the region stays mapped at the
    /// same address forever and only changes which list names it. The wait that
    /// genuinely is needed moved to where re-tenanting happens, in
    /// `region_pool::claim_region`, where it is paid once per claim instead of
    /// once per wave.
    pub fn release_empty_arenas(&self) -> Result<usize> {
        let mut total_freed = 0;
        for backing in &self.backings {
            total_freed += backing.release_empty_arenas()?;
        }
        Ok(total_freed)
    }

    /// Pack the KV pools toward the lowest addresses so the arena frontier falls.
    ///
    /// Passes **every** backing, because the pool is shared but the block tables are
    /// per-layer: a pass that rewrote one layer's view of a relocated chunk and not
    /// the rest would be wrong attention on every other layer.
    ///
    /// `sweep` is where a caller with its own gid holders — a substrate residence, a
    /// projection cache — applies the same relocation map. A caller whose only block
    /// tables are these backings' passes a closure that does nothing.
    ///
    /// Only legal between forwards; refuses otherwise.
    #[cfg(feature = "cuda")]
    pub fn compact_kv(
        &self,
        budget: std::time::Duration,
        sweep: &mut dyn FnMut(&mut candle_nn::kv_cache::Sweep<'_>) -> Result<()>,
    ) -> std::result::Result<
        candle_nn::kv_cache::CompactionReport,
        candle_nn::kv_cache::CompactionRefused,
    > {
        candle_nn::kv_cache::compact_backings(&self.backings, budget, sweep)
    }

    /// Create the arenas that mid-wave refusals recorded, and answer with how
    /// many were made.
    ///
    /// **Call this from the wave loop, between forwards.** The sealing thread is
    /// what gets refused, but it is the wrong party to satisfy the demand: it has
    /// to *find* a gap, and on this engine the gap between one wave and the next
    /// is narrower than a sealing pass, so it never won one — the `1088 B` class
    /// stayed frozen at 49 arenas with 75 regions free while the hot→warm drain
    /// sat at 634 MiB and never moved.
    ///
    /// The wave loop does not have to find the gap; it *is* the gap. Running here
    /// makes the demand converge in one wave instead of never.
    #[cfg(feature = "cuda")]
    pub fn create_deferred_arenas(&self) -> Result<usize> {
        let mut made = 0;
        for backing in &self.backings {
            made += backing.create_deferred_arenas()?;
        }
        Ok(made)
    }

    /// Device VRAM `(free, total)` in bytes, or `None` on non-CUDA devices /
    /// query failure. Drives the scheduler's VRAM-pressure admission gate.
    pub fn vram_free_total(&self) -> Option<(usize, usize)> {
        #[cfg(feature = "cuda")]
        {
            if let Device::Cuda(d) = &self.device {
                return d.mem_get_info().ok();
            }
        }
        None
    }

    /// Bytes of KV the reservation can still hold — free regions × the region
    /// size, an exact count. `None` on non-CUDA, or before this device has a
    /// reservation. See [`candle_nn::kv_cache::vram_budget_available`].
    pub fn vram_budget_available(&self) -> Option<usize> {
        #[cfg(feature = "cuda")]
        return candle_nn::kv_cache::vram_budget_available(&self.device);
        #[cfg(not(feature = "cuda"))]
        return None;
    }

    /// The VRAM Governor for this session's device, if installed. Lets the
    /// scheduler drive admission/eviction off the governor's honest measurement,
    /// forecast, and thresholds.
    pub fn vram_governor(&self) -> Option<std::sync::Arc<candle::vram::VramGovernor>> {
        #[cfg(feature = "cuda")]
        if let candle::DeviceLocation::Cuda { gpu_id } = self.device.location() {
            return candle::vram::get(gpu_id);
        }
        None
    }

    /// Our CUDA memory pool's `(used, reserved)` bytes — what our allocations
    /// actually occupy (model weights + KV + activations) vs the high-water
    /// bytes the pool has reserved from the OS. `reserved - used` is held but
    /// reusable; `reserved` is why the driver's `free` reads near zero. The
    /// real diagnostic for "what's using VRAM". `None` on non-CUDA / failure.
    /// The formats live (unsealed) KV chunks occupy — see
    /// [`candle_nn::kv_cache::active_kv_formats`]. Admission prices candidates
    /// in THESE, not the configured sealed formats.
    pub fn active_kv_formats(
        &self,
    ) -> (candle_nn::kv_cache::KvFormat, candle_nn::kv_cache::KvFormat) {
        candle_nn::kv_cache::active_kv_formats(
            self.config.k_format,
            matches!(self.device, Device::Cuda(_)),
        )
    }

    pub fn vram_pool_stats(&self) -> Option<(usize, usize)> {
        #[cfg(feature = "cuda")]
        {
            if let Device::Cuda(d) = &self.device {
                if let (Ok(used), Ok(reserved)) = (d.pool_used_bytes(), d.pool_reserved_bytes()) {
                    return Some((used, reserved));
                }
            }
        }
        None
    }

    /// GPU KV arena occupancy per size class. Reads the shared GID pool
    /// via layer 0's backing (arenas pool globally across same-config layers, so
    /// one backing is the whole model). `None` when there are no backings.
    pub fn kv_gpu_class_stats(&self) -> Option<GpuArenaClassStats> {
        self.backings.first().map(|b| b.gpu_arena_class_stats())
    }

    /// How fragmented every GPU KV pool is — the arenas a perfect pack would
    /// empty, per size class.
    ///
    /// Reads layer 0's backing for the same reason [`Self::kv_gpu_class_stats`]
    /// does: arenas pool globally across same-config layers, so one backing's view
    /// is the whole model's.
    ///
    /// Returns `(key, fragmentation)` for every pool holding at least one arena,
    /// so a caller can report the ladder without a row per empty rung.
    pub fn kv_fragmentation(&self) -> Vec<(ArenaKey, Fragmentation)> {
        let Some(b) = self.backings.first() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for class in SizeClass::all() {
            let key = ArenaKey::new(class, ArenaLocation::Gpu);
            let Ok(census) = b.compaction_census(key) else {
                continue;
            };
            if census.is_empty() {
                continue;
            }
            let f = fragmentation(&census, key);
            if f.arenas > 0 {
                out.push((key, f));
            }
        }
        out
    }

    /// Chunk moves a perfect pack would cost, across every GPU pool.
    ///
    /// The other half of the fragmentation figure: [`Self::kv_fragmentation`] says
    /// how much ground a pack would return, this says what it would cost to return
    /// it. Both are wanted before running one — a pass that frees four arenas for
    /// forty thousand copies is not worth its bandwidth, and the ratio is the only
    /// thing that says so.
    ///
    /// Unbounded (`max_moves = 0`): this is the *whole* cost of reaching a gapless
    /// prefix, which is the figure to compare a per-pass budget against.
    pub fn kv_planned_moves(&self) -> usize {
        let Some(b) = self.backings.first() else {
            return 0;
        };
        let mut moves = 0;
        for class in SizeClass::all() {
            let key = ArenaKey::new(class, ArenaLocation::Gpu);
            let Ok(census) = b.compaction_census(key) else {
                continue;
            };
            if let Some(plan) = plan_pool(&census, key, 0) {
                moves += plan.moves.len();
            }
        }
        moves
    }

    /// Everything fragmentation denies the weight side, in regions.
    ///
    /// **The highest live arena is the marker.** The wave transient tier must stand
    /// above it and `weight_floor` is measured from there, so the weight side's
    /// ground — and therefore expert residency, and therefore decode — is set by
    /// where the topmost live arena sits. Neither the live arena count nor the
    /// occupancy ratio matters except through that one number.
    ///
    /// Sums the per-pool packed floor across the ladder and reads the frontier from
    /// the region pool, which is device-global: regions are shared by every pool, so
    /// the frontier is not a per-class quantity and cannot be derived from
    /// [`Self::kv_fragmentation`] alone.
    /// Regions a perfect pack would free, **cheaply** — the gate that decides
    /// whether a compaction pass is worth its census.
    ///
    /// Sums `(arenas_held - arenas_if_packed)` over the GPU pools from the refcount
    /// tables' live counters, with no occupancy bitmap walked. That distinction is
    /// the whole point: the exact figure is [`Self::kv_ground_lost`], which *is* the
    /// census, so it cannot be what decides whether to pay for one. The cheap
    /// counterpart to the region pool's hole count, which
    /// [`Self::kv_region_stats`] already gives for free.
    ///
    /// Holes alone are not a usable gate. They are self-correcting — the region free
    /// list is lowest-index-first, so the next claim takes the lowest hole — and a
    /// steadily-loaded pool sits at zero holes with tens of sparse arenas underneath.
    /// Measured: 16 passes over 110 s of churn, because the gate read holes and holes
    /// were zero on all but three samples while 50 arenas of air sat in the pools.
    pub fn kv_sparse_arenas(&self) -> usize {
        let Some(b) = self.backings.first() else {
            return 0;
        };
        SizeClass::all()
            .map(|class| {
                let (held, packed) = b.pool_sparsity(ArenaKey::new(class, ArenaLocation::Gpu));
                held.saturating_sub(packed)
            })
            .sum()
    }

    /// The highest region any KV arena of this session's device holds, band and
    /// record pools together — see [`ChunkedKvBacking::kv_top_rank`]. Read beside
    /// [`Self::kv_region_stats`]'s frontier, it says whether a KV arena is what
    /// stands at the frontier, and so whether a KV compaction could lower it.
    pub fn kv_top_rank(&self) -> Option<usize> {
        self.backings.first().and_then(|b| b.kv_top_rank())
    }

    /// The region pool's own counters for this session's device.
    ///
    /// The **cheap** half of the fragmentation picture, and the reason it is exposed
    /// separately from [`Self::kv_ground_lost`]: this is a handful of field reads
    /// behind one lock, whereas `kv_ground_lost` sums `packed_arenas` across the
    /// ladder, which means walking every arena's occupancy bitmap. A caller deciding
    /// *whether* to pay for that walk cannot use the walk to decide.
    pub fn kv_region_stats(&self) -> Option<candle_nn::kv_cache::RegionStats> {
        let candle::DeviceLocation::Cuda { gpu_id } = self.device.location() else {
            return None;
        };
        candle_nn::kv_cache::region_stats(gpu_id)
    }

    pub fn kv_ground_lost(&self) -> Option<GroundLost> {
        // This session's own device ordinal, not a hardcoded 0: a second engine on
        // a second card would otherwise report the first card's frontier.
        let candle::DeviceLocation::Cuda { gpu_id } = self.device.location() else {
            return None;
        };
        let stats = candle_nn::kv_cache::region_stats(gpu_id)?;
        let packed_arenas = self
            .kv_fragmentation()
            .iter()
            .map(|(_, f)| f.packed_arenas)
            .sum();
        Some(GroundLost {
            watermark: stats.live_watermark,
            live_arenas: stats.live,
            packed_arenas,
            span_regions: stats.span_tenant,
            // In-use ground that appears in no size-class row, so a consumer summing those
            // rows would charge it as waste. See `GroundLost::record_regions`.
            record_regions: self
                .backings
                .first()
                .map(|b| b.record_arena_regions())
                .unwrap_or(0),
        })
    }
}
