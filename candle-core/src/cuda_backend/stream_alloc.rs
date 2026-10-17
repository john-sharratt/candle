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

//!
//! **Never on a stream that is capturing.** The device's one stream is the one a
//! wave records on, and an allocation issued there mid-segment would be taken
//! into the graph — which the segment's audit then refuses, losing the wave.
//! The caller pauses the capture first ([`super::CudaDevice::pause_capture`]);
//! one that did not is refused here, by name, before anything is issued.

use std::panic::Location;
use std::sync::Arc;

use cudarc::driver::{sys, CudaSlice, CudaStream, DeviceRepr, HostSlice, ValidAsZeroBits};

use super::WrapErr;
use crate::forbidden_alloc;
use crate::Result;

/// Refuse an allocation on `stream` while it captures.
#[track_caller]
fn refuse_while_capturing(stream: &Arc<CudaStream>, what: &str) -> Result<()> {
    let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    // SAFETY: a status query on a live stream, which the driver answers during
    // a capture.
    unsafe { sys::cuStreamIsCapturing(stream.cu_stream(), &mut status) }
        .result()
        .w()?;
    if status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
        crate::bail!(
            "{what} at {} on a stream that is recording a wave segment: the allocation would \
             be taken into the graph. Pause the capture around it (`pause_capture`).",
            Location::caller()
        );
    }
    Ok(())
}

/// `len` uninitialised elements on `stream`.
///
/// # Safety
///
/// As `CudaStream::alloc`: every element must be written before it is read.
#[track_caller]
pub unsafe fn stream_alloc<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<CudaSlice<T>> {
    refuse_while_capturing(stream, "CudaStream::alloc")?;
    forbidden_alloc::record("CudaStream::alloc", len * std::mem::size_of::<T>());
    stream.alloc::<T>(len).w()
}

/// `len` zeroed elements on `stream`.
#[track_caller]
pub fn stream_alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<CudaSlice<T>> {
    refuse_while_capturing(stream, "CudaStream::alloc_zeros")?;
    forbidden_alloc::record("CudaStream::alloc_zeros", len * std::mem::size_of::<T>());
    stream.alloc_zeros::<T>(len).w()
}

/// A fresh device copy of `src` on `stream`.
#[track_caller]
pub fn stream_memcpy_stod<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
    stream: &Arc<CudaStream>,
    src: &Src,
) -> Result<CudaSlice<T>> {
    refuse_while_capturing(stream, "CudaStream::memcpy_stod")?;
    forbidden_alloc::record(
        "CudaStream::memcpy_stod",
        src.len() * std::mem::size_of::<T>(),
    );
    stream.memcpy_stod(src).w()
}
