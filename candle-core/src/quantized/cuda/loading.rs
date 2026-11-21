//! Wraps quantized weights in device storage, uploading them on a copy stream or
//! into slots the caller owns.
//!
//! Weights load either into memory the storage owns, or into a fixed slot of the
//! weight zone. A storage over a slot releases only its view and never the bytes,
//! so an eviction can hand the slot back to the zone. Uploads on a secondary stream
//! can overlap compute, and the caller must order the first use of the result
//! behind an event. Repacked data is loaded at its repacked size, with no padding
//! tail, because the expert kernels read exactly that layout. A view over bytes
//! already on the device covers a slot that was relocated.

use super::*;

pub fn load_quantized<T: super::super::GgmlType + Send + Sync + 'static>(
    device: &CudaDevice,
    data: &[T],
) -> Result<super::super::QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let dtype = T::DTYPE;
    let padded_len = data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    let mut inner = unsafe { device.alloc::<u8>(padded_len)? };
    device.memcpy_htod(data, &mut inner.slice_mut(..data.len()))?;
    zero_padding_tail(&device.cuda_stream(), &mut inner, data.len())?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
            inner,
            len: data.len(),
        }),
        device: device.clone(),
        dtype,
        backing: Backing::Owned,
    }))
}

/// Like [`load_quantized`], but issues the alloc + H2D copy on a secondary
/// [`CudaStream`] so it can overlap with compute on the device's main stream.
///
/// The caller is responsible for synchronisation: record a
/// [`CudaEvent`](cudarc::driver::CudaEvent) on the copy stream *after* calling
/// this, then make the compute stream [`wait`](cudarc::driver::CudaStream::wait)
/// on that event before first use.
///
/// # Safety
///
/// The returned `QStorage` must not be accessed on the device's main stream
/// until the copy-stream work has been synchronised via an event.
pub fn load_quantized_on_stream<T: super::super::GgmlType + Send + Sync + 'static>(
    device: &CudaDevice,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    data: &[T],
) -> Result<super::super::QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let dtype = T::DTYPE;
    let padded_len = data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    let mut inner = unsafe { stream_alloc::<u8>(stream, padded_len)? };
    stream
        .memcpy_htod(data, &mut inner.slice_mut(..data.len()))
        .map_err(crate::Error::wrap)?;
    zero_padding_tail(stream, &mut inner, data.len())?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
            inner,
            len: data.len(),
        }),
        device: device.clone(),
        dtype,
        backing: Backing::Owned,
    }))
}

// Repacked-format helpers (swap file support)

/// Load pre-repacked K/128 GEMX data from host bytes into a new CUDA buffer.
///
/// Unlike [`load_quantized_on_stream`], this does **not** add
/// [`MATRIX_ROW_PADDING`] — the repacked data is already correctly sized
/// by `get_repacked_size_bytes()`.
///
/// Used when loading experts from a swap file where weights are already in
/// K/128 format.  No GPU repack kernel is needed.
pub fn load_repacked_on_stream(
    device: &CudaDevice,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    repacked_data: &[u8],
    dtype: GgmlDType,
) -> Result<super::super::QStorage> {
    let len = repacked_data.len();
    let mut inner = unsafe { stream_alloc::<u8>(stream, len)? };
    stream
        .memcpy_htod(repacked_data, &mut inner.slice_mut(..len))
        .map_err(crate::Error::wrap)?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: std::mem::ManuallyDrop::new(PaddedCudaSlice { inner, len }),
        device: device.clone(),
        dtype,
        backing: Backing::Owned,
    }))
}

/// Upload `repacked_data` to `dst_ptr` — device memory owned by somebody else —
/// and wrap it as a storage that will not free those bytes.
///
/// The weight-zone counterpart of [`load_repacked_on_stream`]. An expert slot is
/// a fixed-size range of the device reservation, handed out by
/// `candle_nn::kv_cache::WeightZone`, so the bytes outlive any one `QTensor`
/// built over them: a miss overwrites the slot in place and an eviction returns
/// it to the zone's free list. Dropping the storage must therefore release the
/// *view* and nothing else, which is exactly what [`Backing::Lease`] means here.
///
/// No `MATRIX_ROW_PADDING` tail, matching what the pool path allocated: the
/// expert kernels read the repacked layout exactly and never over-read it.
///
/// # Safety
///
/// `dst_ptr` must point to at least `repacked_data.len()` bytes of device memory
/// that stays live, and un-aliased for writes, for the storage's lifetime.
pub unsafe fn load_repacked_into(
    device: &CudaDevice,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    dst_ptr: u64,
    repacked_data: &[u8],
    dtype: GgmlDType,
) -> Result<super::super::QStorage> {
    let len = repacked_data.len();
    // Upgraded once and moved into the storage: a second upgrade would need its
    // own `leak` on every path out of here, and forgetting one strands a pair of
    // `CudaEvent`s per expert load.
    let mut inner = stream.upgrade_device_ptr::<u8>(dst_ptr, len);
    stream
        .memcpy_htod(repacked_data, &mut inner.slice_mut(..len))
        .map_err(crate::Error::wrap)?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: std::mem::ManuallyDrop::new(PaddedCudaSlice { inner, len }),
        device: device.clone(),
        dtype,
        backing: Backing::Lease(LeaseOrigin::Foreign),
    }))
}

/// Wrap `bytes` of already-populated device memory at `ptr` as a storage that
/// will not free it — [`load_repacked_into`] without the upload.
///
/// For a slot that has been **relocated**: the bytes were copied device-to-device
/// and are already correct, but the three `QMatMul`s hold device pointers, so
/// their storages have to be rebuilt over the new address.
///
/// # Safety
///
/// `ptr` must point to at least `extent` of live device memory whose first
/// `bytes` hold a valid `dtype` payload, un-aliased for writes for the storage's
/// lifetime.
pub unsafe fn view_repacked(
    device: &CudaDevice,
    ptr: u64,
    bytes: usize,
    extent: usize,
    dtype: GgmlDType,
) -> Result<super::super::QStorage> {
    if extent < bytes {
        crate::bail!("view_repacked: extent {extent} is shorter than the {bytes}-byte payload");
    }
    let inner = device
        .stream_unleased()
        .upgrade_device_ptr::<u8>(ptr, extent);
    Ok(QStorage::Cuda(QCudaStorage {
        data: std::mem::ManuallyDrop::new(PaddedCudaSlice { inner, len: bytes }),
        device: device.clone(),
        dtype,
        backing: Backing::Lease(LeaseOrigin::Foreign),
    }))
}

/// Bytes an owned allocation reserves for an `el_count` payload of `dtype`.
///
/// Payload plus the [`MATRIX_ROW_PADDING`] tail the GGML matmul kernels address
/// unconditionally. A caller that wants to run those kernels over a *leased*
/// view must reserve this much in the memory it leases from — see
/// [`QCudaStorage::fwd`], which checks it.
pub fn padded_storage_bytes(el_count: usize, dtype: GgmlDType) -> usize {
    ceil_div(el_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size()
}

/// Like [`load_repacked_on_stream`], but uses the device's default stream.
pub fn load_repacked(
    device: &CudaDevice,
    repacked_data: &[u8],
    dtype: GgmlDType,
) -> Result<super::super::QStorage> {
    let len = repacked_data.len();
    let mut inner = unsafe { device.alloc::<u8>(len)? };
    device.memcpy_htod(repacked_data, &mut inner.slice_mut(..len))?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: std::mem::ManuallyDrop::new(PaddedCudaSlice { inner, len }),
        device: device.clone(),
        dtype,
        backing: Backing::Owned,
    }))
}
