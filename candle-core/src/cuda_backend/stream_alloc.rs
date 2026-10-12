//! Device allocations made on a bare [`CudaStream`] rather than through a
//! [`super::CudaDevice`].
//!
//! Some code holds a stream and no device handle — a copy stream the persistence
//! thread owns, a stream passed down into a kernel wrapper — and allocates on it
//! directly. cudarc's `CudaStream::{alloc, alloc_zeros, memcpy_stod}` reach the
//! driver without passing through anything this crate instruments, so every such
//! call was invisible to [`crate::forbidden_alloc`]: a capture of the daemon
//! counted 3,755 driver allocations in thirty seconds where the detector had
//! named a few hundred.
//!
//! These are the same three calls with the record in front. A site that must
//! allocate on a bare stream goes through here, so a report names it like any
//! other.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, HostSlice, ValidAsZeroBits};

use super::WrapErr;
use crate::forbidden_alloc;
use crate::Result;

/// `len` uninitialised elements on `stream`.
///
/// # Safety
///
/// As `CudaStream::alloc`: every element must be written before it is read.
pub unsafe fn stream_alloc<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<CudaSlice<T>> {
    forbidden_alloc::record("CudaStream::alloc", len * std::mem::size_of::<T>());
    stream.alloc::<T>(len).w()
}

/// `len` zeroed elements on `stream`.
pub fn stream_alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<CudaSlice<T>> {
    forbidden_alloc::record("CudaStream::alloc_zeros", len * std::mem::size_of::<T>());
    stream.alloc_zeros::<T>(len).w()
}

/// A fresh device copy of `src` on `stream`.
pub fn stream_memcpy_stod<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
    stream: &Arc<CudaStream>,
    src: &Src,
) -> Result<CudaSlice<T>> {
    forbidden_alloc::record(
        "CudaStream::memcpy_stod",
        src.len() * std::mem::size_of::<T>(),
    );
    stream.memcpy_stod(src).w()
}
