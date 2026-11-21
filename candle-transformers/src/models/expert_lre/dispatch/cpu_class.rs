//! The device side of the CPU expert class: the buffers and counters that one
//! dispatch shares with the CPU pool.
//!
//! The pool's service owns the mapped rings. The device buffers hold the routing
//! tables, the export's block count, and the carry that promotes CPU experts behind
//! the pool. One set serves every layer, because each invocation's use of it ends in
//! stream order before the next invocation writes it. An exploration counter lets
//! the dispatch try the pool again once the pool has served nothing for a set
//! number of invocations.

use super::*;

/// The CPU class's device side for one dispatch (`docs/cpu_expert_compute.md`
/// §5.1–§5.5): the pool's service, which owns the mapped rings, and the two
/// device buffers bucketize, the export kernel and the down launch's CPU item
/// share. One set serves every layer: each invocation's use ends — the down
/// launch has pulled its results — before the next invocation's bucketize
/// writes them, in stream order.
pub(crate) struct CpuPath {
    pub(crate) service: CpuService,
    /// The warm tier's pageable copies, which the pool computes from directly.
    pub(super) paged: Arc<PagedImages>,
    /// `u32[2 · MAX_PREFILL_TOKENS · k]`: ring row → expert-grouped row.
    pub(super) dst_row: CudaSlice<u32>,
    /// The export kernel's finished-block count, keyed by the ticket it counts
    /// (`moe_cpu_export.cu`): the ticket's low word above the count.
    pub(super) blocks_done: CudaSlice<u64>,
    /// `u32[CPU_CARRY_WORDS]`: the CPU experts one launch hands the next to
    /// promote behind the pool (`moe_bucketize_cpu.cuh`, THE CARRY). Zeroed:
    /// sequence word 0 names no launch, so the first launch finds nothing.
    pub(super) carry: CudaSlice<u32>,
    /// The inputs (`moe_bucketize_cpu.cuh`, THE INPUTS): `u32[n_tokens]`
    /// scratch, and `u32[cap_rows]` input row → token, which the export reads.
    pub(super) tok_first: CudaSlice<u32>,
    pub(super) in_tok: CudaSlice<u32>,
    /// Exploration (§5.3.2): the pool's served count when last seen to move,
    /// and the invocations since.
    pub(super) explore: Mutex<Quiet>,
}

#[derive(Default)]
pub(super) struct Quiet {
    pub(super) served: u64,
    pub(super) invocations: u32,
}

impl CpuPath {
    pub(crate) fn new(
        device: &CudaDevice,
        service: CpuService,
        paged: Arc<PagedImages>,
        k: usize,
    ) -> Result<Self> {
        let cap_rows = service.rings().cap_rows;
        Ok(Self {
            paged,
            // SAFETY (all three): bucketize writes every word a reader reads —
            // `tok_first` it sets for every token before it reads one.
            dst_row: unsafe { device.alloc::<u32>(2 * MAX_PREFILL_TOKENS * k)? },
            tok_first: unsafe { device.alloc::<u32>(2 * MAX_PREFILL_TOKENS)? },
            in_tok: unsafe { device.alloc::<u32>(cap_rows.max(1))? },
            service,
            blocks_done: device.memcpy_stod(&[0u64])?,
            carry: device.memcpy_stod(&[0u32; CPU_CARRY_WORDS])?,
            explore: Mutex::new(Quiet::default()),
        })
    }

    /// Whether this invocation explores: the pool has served nothing for
    /// `period` invocations. Exploring restarts the count.
    pub(super) fn explores(&self, period: u32) -> bool {
        let mut q = self.explore.lock().unwrap_or_else(|e| e.into_inner());
        let served = self.service.served();
        if served != q.served {
            q.served = served;
            q.invocations = 0;
            return false;
        }
        q.invocations += 1;
        if period == 0 || q.invocations < period {
            return false;
        }
        q.invocations = 0;
        true
    }
}
