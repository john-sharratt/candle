//! Which invocation of each MoE row the device has begun reading.
//!
//! One word per row. Bucketize stores its invocation's ticket into its row's
//! word, behind a system fence, before it reads any of the row's live entries
//! (`moe_bucketize.cu`, phase 1b); the host reads the word after retargeting an
//! entry, behind its own fence (`ReclaimClock::retire_key`). That store/load
//! pair on each side means either the host sees the device's ticket — and holds
//! the slot until that invocation is done — or the device's bucketize sees the
//! retargeted entry and never reads the slot. So the key a slot waits on is the
//! invocation the GPU is actually inside, however far the forward thread has
//! enqueued past it.

use candle::cuda_backend::cudarc::driver::sys;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

/// The per-row started words: mapped pinned memory the device writes through
/// [`Self::dev_ptr`], or plain host memory for a host-only test.
pub(crate) struct StartedRows {
    words: *const AtomicU64,
    rows: usize,
    dev: u64,
    backing: Backing,
}

enum Backing {
    /// Held only to own the words `StartedRows::words` points into.
    #[cfg(test)]
    Host {
        _words: Box<[AtomicU64]>,
    },
    Mapped,
}

// SAFETY: the words are atomics, written by the device or by `store` and read
// by `load`; the pointer is to memory this value owns for its whole life.
unsafe impl Send for StartedRows {}
unsafe impl Sync for StartedRows {}

impl StartedRows {
    /// `rows` words in mapped pinned memory, each 0 — no invocation started.
    pub(crate) fn mapped(rows: usize) -> candle::Result<Self> {
        let bytes = rows.max(1) * std::mem::size_of::<u64>();
        let mut raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: a page-locked, device-mapped allocation, freed in `drop`.
        let r = unsafe { sys::cuMemHostAlloc(&mut raw, bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
        if r != sys::CUresult::CUDA_SUCCESS {
            candle::bail!("expert cache: mapped started-row words allocation failed: {r:?}");
        }
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: `raw` was allocated with DEVICEMAP just above.
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, raw, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: allocated just above and never handed out.
            unsafe { sys::cuMemFreeHost(raw) };
            candle::bail!("expert cache: started-row words have no device address: {r:?}");
        }
        // SAFETY: `bytes` just allocated; page-aligned, so every word is.
        unsafe { std::ptr::write_bytes(raw as *mut u8, 0, bytes) };
        Ok(Self {
            words: raw as *const AtomicU64,
            rows,
            dev,
            backing: Backing::Mapped,
        })
    }

    /// `rows` words in host memory, for a test standing in for the device.
    #[cfg(test)]
    pub(crate) fn host(rows: usize) -> Self {
        let words: Box<[AtomicU64]> = (0..rows.max(1)).map(|_| AtomicU64::new(0)).collect();
        Self {
            words: words.as_ptr(),
            rows,
            dev: 0,
            backing: Backing::Host { _words: words },
        }
    }

    /// The words' device address — bucketize's `started_rows`.
    pub(crate) fn dev_ptr(&self) -> u64 {
        self.dev
    }

    /// The ticket of the latest invocation of `row` the device has begun; 0 for
    /// none.
    pub(crate) fn load(&self, row: usize) -> u64 {
        assert!(row < self.rows, "started word {row} of {}", self.rows);
        // SAFETY: in bounds by the assertion; the words live as long as `self`.
        unsafe { &*self.words.add(row) }.load(Ordering::SeqCst)
    }

    /// How many rows there are words for.
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// The latest invocation the device has begun on any row; 0 for none.
    pub(crate) fn latest(&self) -> u64 {
        (0..self.rows).map(|r| self.load(r)).max().unwrap_or(0)
    }

    /// What the device's bucketize does, for a test standing in for it.
    #[cfg(test)]
    pub(crate) fn store(&self, row: usize, ticket: u64) {
        assert!(row < self.rows, "started word {row} of {}", self.rows);
        // SAFETY: in bounds by the assertion.
        unsafe { &*self.words.add(row) }.store(ticket, Ordering::SeqCst);
    }
}

impl Drop for StartedRows {
    fn drop(&mut self) {
        match &self.backing {
            #[cfg(test)]
            Backing::Host { .. } => {}
            Backing::Mapped => {
                // SAFETY: allocated by `cuMemHostAlloc` in `mapped`; the device
                // work that writes it is done before the cache drops.
                unsafe { sys::cuMemFreeHost(self.words as *mut c_void) };
            }
        }
    }
}
