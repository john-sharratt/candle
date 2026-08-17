//! Zero-copy staging ring for grouped-GEMM launch tables.
//!
//! Every grouped (MoE) matmul launch needs a small host-built descriptor blob —
//! the expert weight-pointer array plus the three tile tables. Uploading that
//! blob per launch (one tiny `memcpy_stod` each) was the last per-launch H2D
//! copy on the hot path: trivial GPU time, but a real WDDM submission tax at
//! tens of thousands of launches per sweep, plus a device allocation per call.
//!
//! The ring removes both. Tables are written into a device-mapped, write-
//! combined pinned slab (`cuMemHostAlloc(DEVICEMAP | WRITECOMBINED)`) and the
//! kernel reads them **in place** over PCIe — no `memcpy`, no device
//! allocation, no per-call driver work beyond the launch itself. Table reads
//! are a few dozen bytes once per block, so the PCIe latency hides under
//! block-level parallelism.
//!
//! # Reuse fencing
//!
//! Launches are asynchronous, so a slab slot must not be rewritten while a
//! launch that reads it may still be pending. The slab is split into two
//! halves used alternately: crossing into the other half records an event on
//! the launch stream (fencing every launch that read the half being left) and
//! host-waits the event previously recorded for the half being entered. With
//! megabytes of runway per half, the entered half's fence covers launches
//! enqueued thousands of calls ago — the wait is a formality except under
//! pathological queue depth.
//!
//! The write **and** the launch happen inside the ring lock ([`Self::with_table`]
//! takes the launch as a closure), which is what makes the fence airtight
//! across threads: an event recorded at half-swap is ordered after every
//! launch that allocated from that half.

use crate::cuda_backend::WrapErr;
use crate::{CudaDevice, Result};
use cudarc::driver::sys;
use cudarc::driver::{CudaEvent, CudaStream};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Total slab bytes (two halves). A launch table is ≤ ~64 KB (256 experts × 8 B
/// + 3 × num_tiles × 4 B), so each 2 MB half holds dozens-to-thousands of
/// launches of runway before its fence is consulted.
const RING_BYTES: usize = 4 * 1024 * 1024;

struct RingState {
    /// Bump offset within the current half.
    offset: usize,
    /// Which half `offset` lives in (0 or 1).
    half: usize,
    /// Fence for each half: recorded when the half was last LEFT, i.e. after
    /// every launch that read it. `None` until the half has been left once.
    fences: [Option<CudaEvent>; 2],
}

/// Device-mapped pinned staging ring for per-launch descriptor tables.
pub struct TableRing {
    host_base: *mut u8,
    dev_base: u64,
    stream: Arc<CudaStream>,
    state: Mutex<RingState>,
}

// SAFETY: the raw pointers address host pinned memory owned by this struct;
// all writes go through `with_table`, which holds the state mutex.
unsafe impl Send for TableRing {}
unsafe impl Sync for TableRing {}

impl TableRing {
    fn new(dev: &CudaDevice) -> Result<Self> {
        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            // CU_MEMHOSTALLOC_DEVICEMAP = 0x02, CU_MEMHOSTALLOC_WRITECOMBINED = 0x04
            sys::cuMemHostAlloc(&mut host_ptr, RING_BYTES, 0x02 | 0x04)
                .result()
                .map_err(|e| {
                    crate::Error::Msg(format!("cuMemHostAlloc for table ring failed: {e:?}"))
                })?;
        }
        let mut dev_ptr: sys::CUdeviceptr = 0;
        unsafe {
            sys::cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0)
                .result()
                .map_err(|e| {
                    crate::Error::Msg(format!(
                        "cuMemHostGetDevicePointer for table ring failed: {e:?}"
                    ))
                })?;
        }
        crate::vram::note_host_pinned_alloc(RING_BYTES as u64);
        Ok(Self {
            host_base: host_ptr as *mut u8,
            dev_base: dev_ptr,
            stream: dev.cuda_stream(),
            state: Mutex::new(RingState {
                offset: 0,
                half: 0,
                fences: [None, None],
            }),
        })
    }

    /// Stage `table` into the ring and run `launch` with the table's
    /// device-visible base pointer. The closure must enqueue every launch that
    /// reads the table before returning — the half-swap fence is recorded
    /// after it, and that ordering is the reuse-safety argument.
    pub fn with_table<R>(&self, table: &[u8], launch: impl FnOnce(u64) -> R) -> Result<R> {
        const HALF: usize = RING_BYTES / 2;
        // 16-align each table so the kernel's 8-byte pointer reads stay aligned.
        let len = table.len().div_ceil(16) * 16;
        if len > HALF {
            crate::bail!(
                "TableRing: launch table of {} bytes exceeds the {HALF}-byte half-slab",
                table.len()
            );
        }
        let mut st = self.state.lock().unwrap();
        if st.offset + len > HALF {
            // Leaving this half: fence every launch that read it, then enter
            // the other half once its own last fence has drained.
            let ev = self.stream.record_event(None).w()?;
            let leaving = st.half;
            st.fences[leaving] = Some(ev);
            st.half ^= 1;
            st.offset = 0;
            if let Some(prev) = &st.fences[st.half] {
                prev.synchronize().w()?;
            }
        }
        let off = st.half * HALF + st.offset;
        st.offset += len;
        unsafe {
            std::ptr::copy_nonoverlapping(table.as_ptr(), self.host_base.add(off), table.len());
        }
        Ok(launch(self.dev_base + off as u64))
    }
}

impl Drop for TableRing {
    fn drop(&mut self) {
        // Launches referencing the slab must drain before the memory goes away.
        let _ = self.stream.synchronize();
        unsafe {
            let _ = sys::cuMemFreeHost(self.host_base as *mut std::ffi::c_void);
        }
        crate::vram::note_host_pinned_free(RING_BYTES as u64);
    }
}

/// The per-device table ring, created on first use.
pub fn table_ring(dev: &CudaDevice) -> Result<Arc<TableRing>> {
    static RINGS: OnceLock<Mutex<HashMap<crate::cuda_backend::DeviceId, Arc<TableRing>>>> =
        OnceLock::new();
    let rings = RINGS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = rings.lock().unwrap();
    if let Some(r) = map.get(&dev.id()) {
        return Ok(r.clone());
    }
    let r = Arc::new(TableRing::new(dev)?);
    map.insert(dev.id(), r.clone());
    Ok(r)
}
