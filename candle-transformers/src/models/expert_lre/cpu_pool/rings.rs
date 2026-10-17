//! The mapped memory the device and the pool hand a layer's CPU work through
//! (`docs/cpu_expert_compute.md` §5.2).
//!
//! | Buffer | Written by | Holds |
//! |---|---|---|
//! | work slot | bucketize | `{n_experts, n_rows, row, seq}`, then per CPU expert `{expert, ring_start, count, _, gate, up, down}` — its ring rows, one per assignment |
//! | input map | bucketize | the input rows' count, then each ring row's input row |
//! | row ring | the export kernel | the input rows: each distinct token the CPU experts route, once, its q8a128 input in the q8a1024 layout |
//! | doorbell | the export kernel | the invocation's sequence, once its rows are in the ring |
//! | result ring | the pool | one F32 output row per ring row, ring order |
//! | done | the pool | the sequence whose results have landed |
//! | alive | the pool | non-zero while the pool lives; a dropped pool clears it |
//! | wait | the down launch's CPU items | the longest a CPU item waited for `done`, ns; the pool reads and clears it at the next doorbell |
//!
//! **One slot of each is enough.** The device's down launch for invocation S
//! waits for `done == S` before anything after it runs, and S + 1's
//! bucketize and export come after it in stream order; by then the pool has
//! read S's work and rows and written S's results. Nothing on the host needs
//! to hold the device back.

use std::ffi::c_void;
use std::ptr::{self, null_mut};
use std::sync::atomic::{fence, Ordering};

use candle::{bail, Result};
use candle_kernels::simple::moe_bucketize::CPU_WORK_SLOT_BYTES;
use cudarc::driver::sys;

use super::kernel::act::TILE;

/// One mapped allocation: its host pointer and its device address.
pub struct Mapped {
    host: *mut u8,
    dev: u64,
    bytes: usize,
}

// SAFETY: plain pinned host memory; who writes which part when is the
// protocol above, and every access goes through raw pointers.
unsafe impl Send for Mapped {}
unsafe impl Sync for Mapped {}

impl Mapped {
    /// `bytes` of zeroed, page-locked, device-mapped host memory.
    pub fn new(bytes: usize) -> Result<Self> {
        let mut raw: *mut c_void = null_mut();
        // SAFETY: a page-locked, device-mapped allocation, freed in `drop`.
        let r = unsafe { sys::cuMemHostAlloc(&mut raw, bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
        if r != sys::CUresult::CUDA_SUCCESS {
            bail!("cpu pool: {bytes} mapped bytes refused: {r:?}");
        }
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: `raw` was allocated with DEVICEMAP just above.
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, raw, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: allocated just above and never handed out.
            unsafe { sys::cuMemFreeHost(raw) };
            bail!("cpu pool: mapped buffer has no device address: {r:?}");
        }
        // SAFETY: `bytes` just allocated.
        unsafe { ptr::write_bytes(raw as *mut u8, 0, bytes) };
        Ok(Self {
            host: raw as *mut u8,
            dev,
            bytes,
        })
    }

    pub fn host(&self) -> *mut u8 {
        self.host
    }

    pub fn dev(&self) -> u64 {
        self.dev
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: allocated in `new`; the owner outlives every launch that
        // addressed it (the dispatch drops it after its stream drains).
        unsafe { sys::cuMemFreeHost(self.host as *mut c_void) };
    }
}

/// The work slot's header, as the pool reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkHeader {
    pub n_experts: u32,
    pub n_rows: u32,
    pub row: i32,
    pub seq: u32,
}

/// One CPU expert of the work slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkExpert {
    pub expert: u32,
    pub ring_start: u32,
    pub count: u32,
    pub gate: u64,
    pub up: u64,
    pub down: u64,
}

/// Bytes of a q8a1024 operand holding `rows` rows of `tiles` tiles.
pub fn q8a1024_bytes(rows: usize, tiles: usize) -> usize {
    (rows * tiles).div_ceil(8) * 1152
}

/// The CPU class's mapped buffers for one dispatch.
pub struct CpuRings {
    /// Most CPU rows one invocation may hand over (`SplitParams::cap_rows`).
    pub cap_rows: usize,
    /// The output width: floats per result row.
    pub hidden: usize,
    pub work: Mapped,
    /// The input rows: each distinct token the layer's CPU experts route, once.
    pub rows: Mapped,
    /// `u32[1 + cap_rows]`: the input rows' count, then ring row → input row.
    pub inputs: Mapped,
    /// One result row per ring row — per assignment.
    pub results: Mapped,
    /// `u64`.
    pub doorbell: Mapped,
    /// `u64`.
    pub done: Mapped,
    /// `u32`.
    pub alive: Mapped,
    /// `u64`.
    pub wait: Mapped,
}

impl CpuRings {
    pub fn new(cap_rows: usize, hidden: usize) -> Result<Self> {
        assert!(hidden.is_multiple_of(TILE));
        Ok(Self {
            cap_rows,
            hidden,
            work: Mapped::new(CPU_WORK_SLOT_BYTES)?,
            rows: Mapped::new(q8a1024_bytes(cap_rows, hidden / TILE))?,
            inputs: Mapped::new(4 * (1 + cap_rows))?,
            results: Mapped::new(cap_rows * hidden * 4)?,
            doorbell: Mapped::new(64)?,
            done: Mapped::new(64)?,
            alive: Mapped::new(64)?,
            wait: Mapped::new(64)?,
        })
    }

    /// The device's wait on the last layer the pool served, ns, cleared for
    /// the next. Read at a doorbell: that layer's down launch, which wrote it,
    /// finished before this doorbell's export ran, and this layer's CPU items
    /// write only after the `done` this thread has yet to publish.
    pub fn take_wait(&self) -> u64 {
        // SAFETY: the mapped wait word; see above for who writes it when.
        unsafe {
            let p = self.wait.host() as *mut u64;
            let v = ptr::read_volatile(p);
            ptr::write_volatile(p, 0);
            v
        }
    }

    /// The work slot's header.
    pub fn header(&self) -> WorkHeader {
        // SAFETY: the slot's first 16 bytes; bucketize finished writing them
        // before the export kernel rang the doorbell this was read behind.
        unsafe {
            let p = self.work.host() as *const u32;
            WorkHeader {
                n_experts: ptr::read_volatile(p),
                n_rows: ptr::read_volatile(p.add(1)),
                row: ptr::read_volatile(p.add(2)) as i32,
                seq: ptr::read_volatile(p.add(3)),
            }
        }
    }

    /// CPU expert `j` of the work slot.
    pub fn expert(&self, j: usize) -> WorkExpert {
        // SAFETY: inside the slot (`j` < the header's `n_experts` ≤ MAX_EXPERTS).
        unsafe {
            let p = self.work.host().add(16 + 40 * j);
            let w = |o: usize| ptr::read_volatile(p.add(o) as *const u32);
            let d = |o: usize| ptr::read_volatile(p.add(o) as *const u64);
            WorkExpert {
                expert: w(0),
                ring_start: w(4),
                count: w(8),
                gate: d(16),
                up: d(24),
                down: d(32),
            }
        }
    }

    /// The input rows' count, and each of the first `n_rows` ring rows' input
    /// row, both at most `cap_rows`.
    pub fn input_map(&self, n_rows: usize) -> (usize, Vec<usize>) {
        // SAFETY: `1 + cap_rows` mapped words, which bucketize finished
        // writing before the export kernel rang the doorbell this is read
        // behind.
        unsafe {
            let p = self.inputs.host() as *const u32;
            let n_in = (ptr::read_volatile(p) as usize).min(self.cap_rows);
            let rows = (0..n_rows.min(self.cap_rows))
                .map(|j| ptr::read_volatile(p.add(1 + j)) as usize)
                .collect();
            (n_in, rows)
        }
    }

    pub fn doorbell(&self) -> u64 {
        // SAFETY: the mapped doorbell word.
        unsafe { ptr::read_volatile(self.doorbell.host() as *const u64) }
    }

    pub fn doorbell_ptr(&self) -> *const u64 {
        self.doorbell.host() as *const u64
    }

    /// Publishes `seq`'s results: every result row is written before this.
    pub fn publish_done(&self, seq: u64) {
        fence(Ordering::SeqCst);
        // SAFETY: the mapped done word.
        unsafe { ptr::write_volatile(self.done.host() as *mut u64, seq) };
    }

    pub fn set_alive(&self, alive: bool) {
        // SAFETY: the mapped liveness word.
        unsafe { ptr::write_volatile(self.alive.host() as *mut u32, u32::from(alive)) };
    }
}
