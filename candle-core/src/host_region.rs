//! One contiguous block of host memory, page-locked or pageable, held for the
//! life of its owner and freed on drop.
//!
//! A cache that keeps bytes on the host and uploads them to the device carves
//! its entries out of regions like this one, taken once at startup, rather than
//! allocating per entry. A pinned region is one the copy engine reads directly,
//! so an upload from it is a DMA rather than a copy the driver stages through
//! its own buffer. A pageable region holds the same bytes when the driver
//! grants no more pinned memory. Both are aligned to [`REGION_ALIGN`].

use std::alloc::{alloc, dealloc, Layout};

use crate::vram::{note_host_pinned_alloc, note_host_pinned_free, PinnedUse};
use crate::{Error, Result};

#[cfg(feature = "cuda")]
use crate::cuda_backend::graph::outside_capture;
#[cfg(feature = "cuda")]
use cudarc::driver::sys;

/// A region's base alignment, and the granule its users carve in: one disk
/// sector, so a region's carvings are valid direct-I/O destinations.
pub const REGION_ALIGN: usize = 4096;

/// What backs a [`HostRegion`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    /// Page-locked by the driver (`cuMemHostAlloc`, no write-combining, so the
    /// CPU reads it at full speed too).
    Pinned,
    /// Ordinary heap memory.
    Pageable,
}

/// See the module documentation.
pub struct HostRegion {
    ptr: *mut u8,
    len: usize,
    kind: RegionKind,
    use_: PinnedUse,
}

// SAFETY: the region is plain host memory owned by this value; it is not tied
// to the thread that allocated it, and every access goes through a pointer the
// owner hands out.
unsafe impl Send for HostRegion {}
// SAFETY: the region hands out only its base pointer and length; what is
// written through them is its owner's to order.
unsafe impl Sync for HostRegion {}

impl HostRegion {
    /// `len` bytes of page-locked memory, accounted to `use_`. Refused, not
    /// aborted, when the driver will not pin them — and always without CUDA.
    pub fn pinned(len: usize, use_: PinnedUse) -> Result<Self> {
        let len = round_up(len)?;
        let ptr = pin(len)?;
        note_host_pinned_alloc(use_, len as u64);
        Ok(Self {
            ptr,
            len,
            kind: RegionKind::Pinned,
            use_,
        })
    }

    /// `len` bytes of pageable memory, aligned as a pinned region is. Refused,
    /// not aborted, when the heap cannot supply them.
    pub fn pageable(len: usize) -> Result<Self> {
        let len = round_up(len)?;
        let layout = layout(len)?;
        // SAFETY: `layout` has a non-zero size (`round_up` refuses zero).
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(Error::Msg(format!(
                "host region: the heap refused {len} bytes"
            )));
        }
        Ok(Self {
            ptr,
            len,
            kind: RegionKind::Pageable,
            use_: PinnedUse::IndexPages,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn kind(&self) -> RegionKind {
        self.kind
    }

    /// The region's first byte. Every byte up to [`Self::len`] belongs to the
    /// region for as long as it lives.
    pub fn base(&self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for HostRegion {
    fn drop(&mut self) {
        match self.kind {
            RegionKind::Pinned => {
                unpin(self.ptr);
                note_host_pinned_free(self.use_, self.len as u64);
            }
            RegionKind::Pageable => {
                if let Ok(layout) = layout(self.len) {
                    // SAFETY: allocated by `alloc` with this same layout.
                    unsafe { dealloc(self.ptr, layout) };
                }
            }
        }
    }
}

/// `len` rounded up to [`REGION_ALIGN`]; zero is refused.
fn round_up(len: usize) -> Result<usize> {
    if len == 0 {
        return Err(Error::Msg("host region: zero bytes requested".into()));
    }
    len.checked_next_multiple_of(REGION_ALIGN)
        .ok_or_else(|| Error::Msg(format!("host region: {len} bytes overflows")))
}

fn layout(len: usize) -> Result<Layout> {
    Layout::from_size_align(len, REGION_ALIGN)
        .map_err(|e| Error::Msg(format!("host region layout: {e}")))
}

#[cfg(feature = "cuda")]
fn pin(len: usize) -> Result<*mut u8> {
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    // Pinning is refused on a recording thread, and the refusal ends its
    // capture: the thread's wave is paused around it.
    // SAFETY: `ptr` receives the allocation; flags 0 is plain pinned memory.
    outside_capture(|| unsafe { sys::cuMemHostAlloc(&mut ptr, len, 0) })?
        .result()
        .map_err(|e| Error::Msg(format!("host region: cuMemHostAlloc of {len} bytes: {e:?}")))?;
    Ok(ptr as *mut u8)
}

#[cfg(not(feature = "cuda"))]
fn pin(len: usize) -> Result<*mut u8> {
    Err(Error::Msg(format!(
        "host region: {len} pinned bytes need CUDA"
    )))
}

#[cfg(feature = "cuda")]
fn unpin(ptr: *mut u8) {
    // SAFETY: `ptr` came from `cuMemHostAlloc` and is freed once, here.
    let free = || unsafe { sys::cuMemFreeHost(ptr as *mut std::ffi::c_void) };
    // A wave that cannot be paused is already broken; the pages are freed
    // rather than leaked.
    if outside_capture(free).is_err() {
        let _ = free();
    }
}

#[cfg(not(feature = "cuda"))]
fn unpin(_ptr: *mut u8) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pageable region is aligned, rounded up to whole granules, and its
    /// every byte is the region's to write and read back.
    #[test]
    fn a_pageable_region_is_aligned_whole_and_writable() {
        let r = HostRegion::pageable(10_000).unwrap();
        assert_eq!(r.len(), 12_288);
        assert_eq!(r.kind(), RegionKind::Pageable);
        assert_eq!(r.base() as usize % REGION_ALIGN, 0);
        // SAFETY: every byte up to `len` belongs to the region.
        let bytes = unsafe { std::slice::from_raw_parts_mut(r.base(), r.len()) };
        bytes.fill(0xA5);
        assert!(bytes.iter().all(|&b| b == 0xA5));
    }

    /// Zero bytes is no region.
    #[test]
    fn a_zero_byte_region_is_refused() {
        assert!(HostRegion::pageable(0).is_err());
        assert!(HostRegion::pinned(0, PinnedUse::IndexPages).is_err());
    }
}
