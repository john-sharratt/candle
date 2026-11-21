//! Passes a storage's device address to a kernel, or a table the device's
//! cache owns, as a launch argument.
//!
//! `SlicePtrOrNull` lets a launch argument be either a buffer pointer or a
//! table owned by the device's info-table cache. The cached table is shared by
//! every launch that uses it and has nothing to free. Layout parameters for
//! strided kernels come from that cache, so a layout that repeats is a cache
//! hit with no upload and no allocation.

use super::*;

pub enum SlicePtrOrNull<T> {
    Ptr(Uploaded<T>),
    /// A cache-shared table ([`CudaDevice::info_table`]) — same launch-arg shape as
    /// `Ptr`, but the buffer is owned by the device's info-table cache and shared
    /// across launches, so there is nothing to free here.
    Shared(Arc<Uploaded<T>>),
    Null,
}

impl<T: DeviceRepr> SlicePtrOrNull<T> {
    pub fn builder_arg<'a, 'b: 'a>(&'b self, builder: &mut cudarc::driver::LaunchArgs<'a>) {
        match self {
            SlicePtrOrNull::Ptr(slice) => builder.arg(&**slice),
            SlicePtrOrNull::Shared(slice) => builder.arg(&***slice),
            SlicePtrOrNull::Null => builder.arg(&0usize),
        };
    }
}

impl crate::scalar::Scalar {
    pub fn builder_arg<'a, 'b: 'a>(&'b self, builder: &mut cudarc::driver::LaunchArgs<'a>) {
        use crate::scalar::Scalar;
        match self {
            Scalar::U8(v) => builder.arg(v),
            Scalar::U32(v) => builder.arg(v),
            Scalar::I64(v) => builder.arg(v),
            Scalar::F32(v) => builder.arg(v),
            Scalar::F64(v) => builder.arg(v),
            Scalar::F16(v) => builder.arg(v),
            Scalar::BF16(v) => builder.arg(v),
            Scalar::F8E4M3(v) => builder.arg(v),
        };
    }
}

impl SlicePtrOrNull<usize> {
    /// The dims/stride blob a strided kernel reads, served from the device's
    /// memoized info-table cache ([`CudaDevice::info_table`]): layouts repeat
    /// across launches, so the steady state is a cache hit with no upload and
    /// no allocation at all.
    pub fn params_from_layout(dev: &CudaDevice, l: &Layout) -> Result<Self> {
        let ds = if l.is_contiguous() {
            SlicePtrOrNull::Null
        } else {
            SlicePtrOrNull::Shared(dev.info_table(&[l.dims(), l.stride()].concat())?)
        };
        Ok(ds)
    }
}
