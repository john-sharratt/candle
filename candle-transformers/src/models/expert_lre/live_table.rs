//! The live expert table — where the device-side expert forward finds a weight.
//!
//! One `[3][rows][n_experts]` u64 table — gate, up, down planes — in mapped
//! pinned host memory (`cuMemHostAlloc(DEVICEMAP)`). An entry is the address of
//! the expert's projection in one of three places:
//!
//! - **VRAM** — a weight-zone slot (a hit);
//! - **pinned host memory** — a warm-tier slot or a pad slot, both slot images
//!   at the pack's stride, readable by the device at their host address;
//! - **0** — cold: on the NVMe pack only, or in a pageable warm slot.
//!
//! `moe_bucketize` reads a row's entries for the routed experts, classifies
//! each expert by them, and snapshots them into VRAM; the layer's GEMMs read
//! the snapshot. A cold expert's worker blocks read the live entries, waiting
//! on the gate entry until the stager publishes the expert.
//!
//! **Every write is a host store** — no copy, no kernel, no driver call. That is
//! what lets a waiting worker be released while a thread is blocked in the
//! driver (`docs/moe_live_dispatch_design.md` §0.1). The order is fixed:
//!
//! - **publish**: up, down, full fence, gate — a reader that sees the gate entry
//!   sees the other two;
//! - **clear**: gate, full fence, up, down — a reader that saw a non-zero gate
//!   entry and then a 0 up or down entry classifies the expert cold, which is
//!   what it is.
//!
//! Who may write which value when is `residency` and `reclaim`'s business; this
//! type only stores.

use super::pinned::LayerGeometry;
use super::slot_image::slot_offsets;
use candle::cuda_backend::CudaDevice;
use candle::quantized::cuda::{MOE_MAX_EXPERTS, MOE_MAX_TOPK};
use candle::quantized::GgmlDType;
use candle::Result;
use cudarc::driver::sys;
use std::sync::atomic::{fence, Ordering};

/// The projections, in plane order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Proj {
    Gate = 0,
    Up = 1,
    Down = 2,
}

/// The table's memory: mapped pinned host memory, or — for host-only tests of
/// the policy above it — an ordinary allocation with no device address.
enum TableMemory {
    Mapped { host: *mut u64, dev: u64 },
    #[cfg(test)]
    Host(Box<[u64]>),
}

pub(crate) struct LiveTable {
    memory: TableMemory,
    n_experts: usize,
    n_rows: usize,
    gate_nrows: usize,
    down_nrows: usize,
    gate_dtype: Vec<GgmlDType>,
    down_dtype: Vec<GgmlDType>,
    /// `{gate, up, down}` byte offsets inside a slot image, per row.
    offsets: Vec<[u64; 3]>,
}

// SAFETY: the table is written only through `publish` / `clear` (plain stores
// the callers serialise under the residency lock) and read by the device and by
// `entry`.
unsafe impl Send for LiveTable {}
unsafe impl Sync for LiveTable {}

impl LiveTable {
    /// The table for every row the cache serves, every entry 0.
    ///
    /// Every condition the device-side expert forward depends on is checked here
    /// and refused with its reason: there is no other expert path, so a model
    /// that does not meet them cannot load.
    pub(crate) fn new(
        device: &CudaDevice,
        geoms: &[LayerGeometry],
        n_experts: usize,
        k: usize,
    ) -> Result<Self> {
        let n_rows = geoms.len();
        if n_rows == 0 || n_experts == 0 {
            candle::bail!("live table: no expert grid ({n_rows} rows × {n_experts} experts)");
        }
        if n_experts > MOE_MAX_EXPERTS {
            candle::bail!(
                "live table: {n_experts} experts per layer exceeds the bucketize kernel's \
                 {MOE_MAX_EXPERTS}"
            );
        }
        // `moe_route` caps top-k at 16 and bucketize at MOE_MAX_TOPK; the
        // tighter of the two is the one a routed model can use.
        if k == 0 || k > 16.min(MOE_MAX_TOPK) {
            candle::bail!("live table: top-{k} routing is outside the router's 1..=16");
        }
        // The grouped kernels launch on the legacy null stream while bucketize
        // launches on the compute-stream handle; they are ordered only because
        // that handle IS the null stream.
        if !device.cuda_stream().cu_stream().is_null() {
            candle::bail!(
                "live table: the compute stream is not the legacy null stream, so the expert \
                 chain's launches would be unordered"
            );
        }
        let (gate_nrows, down_nrows, gate_dtype, down_dtype, offsets) = Self::check_rows(geoms)?;
        let len = 3 * n_rows * n_experts;
        let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: a page-locked, device-mapped allocation, freed in `drop`.
        let r = unsafe { sys::cuMemHostAlloc(&mut raw, len * 8, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
        if r != sys::CUresult::CUDA_SUCCESS {
            candle::bail!("live table: mapped allocation of {len} entries failed: {r:?}");
        }
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: `raw` was allocated with DEVICEMAP just above.
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, raw, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: allocated just above and never handed out.
            unsafe {
                sys::cuMemFreeHost(raw);
            }
            candle::bail!("live table: the mapped table has no device address: {r:?}");
        }
        // SAFETY: `len` u64s just allocated, not yet visible to any reader.
        unsafe { std::ptr::write_bytes(raw as *mut u64, 0, len) };
        Ok(Self {
            memory: TableMemory::Mapped {
                host: raw as *mut u64,
                dev,
            },
            n_experts,
            n_rows,
            gate_nrows,
            down_nrows,
            gate_dtype,
            down_dtype,
            offsets,
        })
    }

    /// A table in ordinary host memory, for host-only tests: `offsets` per row.
    #[cfg(test)]
    pub(crate) fn host_only(n_rows: usize, n_experts: usize, offsets: [u64; 3]) -> Self {
        Self {
            memory: TableMemory::Host(vec![0u64; 3 * n_rows * n_experts].into_boxed_slice()),
            n_experts,
            n_rows,
            gate_nrows: 32,
            down_nrows: 32,
            gate_dtype: vec![GgmlDType::Q6_KO; n_rows],
            down_dtype: vec![GgmlDType::Q6_KO; n_rows],
            offsets: vec![offsets; n_rows],
        }
    }

    #[allow(clippy::type_complexity)]
    fn check_rows(
        geoms: &[LayerGeometry],
    ) -> Result<(usize, usize, Vec<GgmlDType>, Vec<GgmlDType>, Vec<[u64; 3]>)> {
        let dims = |s: &[usize], what: &str, row: usize| -> Result<(usize, usize)> {
            match s {
                &[n, kk] => Ok((n, kk)),
                other => candle::bail!("live table: row {row} {what} shape {other:?} is not 2-D"),
            }
        };
        let (gate_nrows, gate_k) = dims(&geoms[0].gate_shape, "gate", 0)?;
        let (down_nrows, down_k) = dims(&geoms[0].down_shape, "down", 0)?;
        let mut gate_dtype = Vec::with_capacity(geoms.len());
        let mut down_dtype = Vec::with_capacity(geoms.len());
        let mut offsets = Vec::with_capacity(geoms.len());
        for (row, g) in geoms.iter().enumerate() {
            let shapes = (
                dims(&g.gate_shape, "gate", row)?,
                dims(&g.up_shape, "up", row)?,
                dims(&g.down_shape, "down", row)?,
            );
            if shapes != ((gate_nrows, gate_k), (gate_nrows, gate_k), (down_nrows, down_k)) {
                candle::bail!(
                    "live table: row {row} projections {shapes:?} differ from row 0's — every \
                     row feeds the same tensors, so they must share one shape"
                );
            }
            if g.up_dtype != g.gate_dtype {
                candle::bail!(
                    "live table: row {row} gate is {:?} but up is {:?} — they share one GEMM \
                     dtype argument",
                    g.gate_dtype,
                    g.up_dtype
                );
            }
            if !g.gate_dtype.is_ko() || !g.down_dtype.is_ko() {
                candle::bail!(
                    "live table: row {row} has non-KO expert weights ({:?}/{:?}) — the device \
                     GEMM is the int8 q8a128 × KO kernel",
                    g.gate_dtype,
                    g.down_dtype
                );
            }
            gate_dtype.push(g.gate_dtype);
            down_dtype.push(g.down_dtype);
            let (gate, up, down, _) = slot_offsets(g);
            offsets.push([gate as u64, up as u64, down as u64]);
        }
        // The grouped GEMM tiles N by 32 and K by 128 for both projections.
        if !gate_nrows.is_multiple_of(32)
            || !down_nrows.is_multiple_of(32)
            || !gate_k.is_multiple_of(128)
            || !down_k.is_multiple_of(128)
        {
            candle::bail!(
                "live table: expert dims outside the grouped GEMM's tiling (gate {gate_nrows}×\
                 {gate_k}, down {down_nrows}×{down_k}; N must be a multiple of 32 and K of 128)"
            );
        }
        Ok((gate_nrows, down_nrows, gate_dtype, down_dtype, offsets))
    }

    fn host(&self) -> *mut u64 {
        match &self.memory {
            TableMemory::Mapped { host, .. } => *host,
            #[cfg(test)]
            TableMemory::Host(b) => b.as_ptr() as *mut u64,
        }
    }

    fn index(&self, proj: Proj, row: usize, expert: usize) -> usize {
        assert!(
            row < self.n_rows && expert < self.n_experts,
            "live table: ({row}, {expert}) outside {} × {}",
            self.n_rows,
            self.n_experts
        );
        (proj as usize * self.n_rows + row) * self.n_experts + expert
    }

    fn store(&self, proj: Proj, row: usize, expert: usize, v: u64) {
        let at = self.index(proj, row, expert);
        // SAFETY: in bounds by `index`; the allocation lives as long as `self`.
        unsafe { std::ptr::write_volatile(self.host().add(at), v) };
    }

    /// Publish `(row, expert)` as the slot image at `base` (a VRAM, warm or pad
    /// slot's first byte): up, down, fence, gate.
    pub(crate) fn publish(&self, row: usize, expert: usize, base: u64) {
        let [g, u, d] = self.offsets[row];
        self.store(Proj::Up, row, expert, base + u);
        self.store(Proj::Down, row, expert, base + d);
        fence(Ordering::SeqCst);
        self.store(Proj::Gate, row, expert, base + g);
        fence(Ordering::SeqCst);
    }

    /// Withdraw `(row, expert)`: gate, fence, up, down.
    pub(crate) fn clear(&self, row: usize, expert: usize) {
        self.store(Proj::Gate, row, expert, 0);
        fence(Ordering::SeqCst);
        self.store(Proj::Up, row, expert, 0);
        self.store(Proj::Down, row, expert, 0);
        fence(Ordering::SeqCst);
    }

    /// The current value of one entry.
    #[cfg(test)]
    pub(crate) fn entry(&self, proj: Proj, row: usize, expert: usize) -> u64 {
        let at = self.index(proj, row, expert);
        // SAFETY: as `store`.
        unsafe { std::ptr::read_volatile(self.host().add(at)) }
    }

    /// Device address of `row`'s entries of `proj`.
    pub(crate) fn row_ptr(&self, proj: Proj, row: usize) -> u64 {
        let dev = match &self.memory {
            TableMemory::Mapped { dev, .. } => *dev,
            #[cfg(test)]
            TableMemory::Host(_) => panic!("live table: a host-only table has no device address"),
        };
        dev + (self.index(proj, row, 0) * 8) as u64
    }

    /// Byte offset of `proj` inside `row`'s slot images.
    pub(crate) fn offset(&self, proj: Proj, row: usize) -> u64 {
        self.offsets[row][proj as usize]
    }

    /// Entries between one projection's row and the next's, what bucketize
    /// steps by to reach a row's up and down entries.
    pub(crate) fn plane(&self) -> i64 {
        (self.n_rows * self.n_experts) as i64
    }

    pub(crate) fn n_experts(&self) -> usize {
        self.n_experts
    }

    pub(crate) fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub(crate) fn gate_nrows(&self) -> usize {
        self.gate_nrows
    }

    pub(crate) fn down_nrows(&self) -> usize {
        self.down_nrows
    }

    /// `row`'s gate/up weight dtype.
    pub(crate) fn gate_dtype(&self, row: usize) -> GgmlDType {
        self.gate_dtype[row]
    }

    /// `row`'s down weight dtype.
    pub(crate) fn down_dtype(&self, row: usize) -> GgmlDType {
        self.down_dtype[row]
    }
}

impl Drop for LiveTable {
    fn drop(&mut self) {
        match self.memory {
            // SAFETY: allocated by `cuMemHostAlloc` in `new`; every reader — the
            // pipeline and stager threads, the device — is joined or drained
            // before the cache drops.
            TableMemory::Mapped { host, .. } => unsafe {
                sys::cuMemFreeHost(host as *mut std::ffi::c_void);
            },
            #[cfg(test)]
            TableMemory::Host(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A publish writes the slot image's three projection addresses; a clear
    /// writes 0 to all three; other entries are untouched.
    #[test]
    fn publish_and_clear_write_the_three_planes() {
        let t = LiveTable::host_only(2, 4, [0, 0x100, 0x300]);
        t.publish(1, 2, 0x7000_0000);
        assert_eq!(t.entry(Proj::Gate, 1, 2), 0x7000_0000);
        assert_eq!(t.entry(Proj::Up, 1, 2), 0x7000_0100);
        assert_eq!(t.entry(Proj::Down, 1, 2), 0x7000_0300);
        assert_eq!(t.entry(Proj::Gate, 0, 2), 0);
        assert_eq!(t.entry(Proj::Gate, 1, 3), 0);
        t.clear(1, 2);
        for p in [Proj::Gate, Proj::Up, Proj::Down] {
            assert_eq!(t.entry(p, 1, 2), 0);
        }
        assert_eq!(t.plane(), 8);
    }
}
