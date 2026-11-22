//! Implements the CUDA storage backend: the tensor operations a CUDA device runs
//! and the storage type they act on.
//!
//! `CudaStorage` wraps a typed device slice and its device. Generic operations
//! are small unit structs that launch the AOT-compiled PTX kernels, which are
//! embedded at compile time, so no NVCC is needed at runtime. Quantised matmul
//! and the paged KV kernels are dispatched elsewhere and do not go through this
//! generic path. The backing types record who owns each allocation.
//!
//! - `argmax_rows`: row-wise argmax launches.
//! - `breadcrumbs`: per-thread record of recent kernel launches.
//! - `casting`: dtype conversion, in place where possible.
//! - `convolution`: convolution, pooling and upsampling launches.
//! - `copy`: strided copies between storages.
//! - `cudnn`: cuDNN-backed convolution.
//! - `device`: the CUDA device and its streams.
//! - `driver_alloc`: allocation made directly through the driver.
//! - `error`: CUDA error values and their wrapping.
//! - `ffi_map`: dtype and op codes for the FFI dispatchers.
//! - `gemm`: cuBLAS matrix multiplies.
//! - `held`: ground held for work in flight.
//! - `host_transfer`: device-to-host copies.
//! - `index_ops`: gather, scatter and index-add.
//! - `info_ring`: ring of launch information.
//! - `kernel_scratch`: scratch space for kernel arguments.
//! - `link_probe`: probe of the device link.
//! - `load_window`: window over a load.
//! - `local_memory`: local memory handling.
//! - `matmul_into`: matmul into a given output.
//! - `module_loading`: loading of PTX modules.
//! - `single_operand`: single-operand kernels.
//! - `slice_ptr`: kernel arguments that carry addresses.
//! - `sparse_update`: sparse in-place updates.
//! - `staged`: staged transfers.
//! - `storage`: the storage type and its backings.
//! - `stream_alloc`: stream-ordered allocation.
//! - `stream_lease`: stream leases.
//! - `stream_owner`: stream ownership.
//! - `sync_watch`: synchronisation watch.
//! - `thread_blas`: per-thread cuBLAS handles.
//! - `two_operand`: two-operand kernels.
//! - `utils`: shared helpers.

use crate::backend::{BackendDevice, BackendStorage};
use crate::forbidden_alloc;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
pub use crate::wave_provenance;
use crate::wave_provenance::{LeaseOrigin, WaveTicket};
use crate::{CpuStorage, DType, Layout, Result, WithDType};
pub use candle_kernels as kernels;
pub use cudarc;

/// Set the thread-local kernel breadcrumb.  Captures `file!()` / `line!()` at
/// the macro call site so the recorded location points to the wrapper, not to
/// this utility module.
macro_rules! cuda_breadcrumb {
    ($name:expr) => {
        set_kernel_breadcrumb($name, file!(), line!())
    };
}
// ─────────────────────────────────────────────────────────────────────────────
use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::result::memcpy_dtoh_async;
use cudarc::driver::{
    CudaSlice, CudaStream, DevicePtr, DeviceRepr, PushKernelArg, ValidAsZeroBits,
};
use float8::F8E4M3;
use half::{bf16, f16};
use std::any::Any;
use std::fmt;
use std::sync::{Arc, Mutex, Weak};

/// The greedy pick, many blocks to a row.
pub mod argmax_rows;
#[cfg(feature = "cudnn")]
pub mod cudnn;
mod device;
mod driver_alloc;
mod error;
pub mod graph;
mod held;
mod info_ring;
mod kernel_scratch;
pub mod link_probe;
pub mod load_scratch;
mod load_window;
mod local_memory;
mod matmul_into;
mod module_loading;
pub mod staged;
mod stream_alloc;
mod stream_lease;
mod stream_owner;
mod sync_watch;
mod thread_blas;
mod utils;
pub use device::{CudaDevice, DeviceId, Uploaded};
pub use error::{CudaError, WrapErr};
pub use held::{ask_holders_to_move, install_held_ground, vacate_epoch, Held, HeldGround};
pub use kernel_scratch::{
    close_scratch_scope, install_scope_carve, open_scratch_scope, scratch_reserve, ScopeCarve,
};
pub use load_window::LoadWindow;
pub use local_memory::local_memory_bytes;
pub use matmul_into::matmul_f32_into;
pub use stream_alloc::{stream_alloc, stream_alloc_zeros, stream_memcpy_stod};
pub use stream_lease::StreamLease;
pub use thread_blas::BlasHandle;
pub use utils::{Map1, Map1Any, Map2, Map2Any, Map2InPlace, Out, OutS, S};

#[derive(Debug)]
pub enum CudaStorageSlice {
    U8(CudaSlice<u8>),
    U32(CudaSlice<u32>),
    I64(CudaSlice<i64>),
    BF16(CudaSlice<bf16>),
    F16(CudaSlice<f16>),
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
    F8E4M3(CudaSlice<F8E4M3>),
    /// Tombstone left behind when a leased storage's drop moves its slice out.
    ///
    /// `CudaSlice::leak` takes `self` by value, so it is unreachable from
    /// `Drop::drop(&mut self)` without moving the slice out first — and moving
    /// out of a type that implements `Drop` requires putting something back.
    /// This variant is that something. It is never observable: it exists only
    /// between the `mem::replace` and the end of `drop`.
    ///
    /// **Named for the hole, not for ownership.** It was `Empty`, which reads as
    /// "a slice of zero elements" and invites the guess that it means "borrowed"
    /// or "not owned" — neither is true. Non-ownership is [`Backing::Lease`]'s
    /// job and is orthogonal to this. A `Moved` slice is the absence left by the
    /// move, and the only correct thing to do with one is diverge.
    ///
    /// The tombstone survives rather than becoming `ManuallyDrop<CudaStorageSlice>`
    /// (the shape `QCudaStorage` uses) because `slice` is a **public** field with
    /// ~330 uses in this crate and ~130 more outside it; the wrapper would put a
    /// deref at every one of them to delete a variant that is unreachable by
    /// construction. `QCudaStorage`'s field is private with a handful of uses,
    /// which is why the same fix was proportionate there and is not here.
    Moved,
}

/// Who owns the device memory behind a [`CudaStorage`].
///
/// A pool allocation is freed on drop, as always. A **leased** one is an offset
/// into memory this process claimed once and never returns — letting
/// `CudaSlice::drop` reach `cuMemFreeAsync` on it would be an error, not a
/// leak.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backing {
    /// Allocated from the stream-ordered pool; freed on drop.
    Owned,
    /// A view over memory owned elsewhere; drop releases the view, never the
    /// memory. The [`LeaseOrigin`] says *whose* memory, which is what lets an
    /// op allocate its output from the same arena its operand came from.
    Lease(LeaseOrigin),
}

impl Backing {
    /// The wave arena an op reading this storage should allocate its output
    /// from, or `None` to allocate from the pool.
    ///
    /// `None` for owned storage and for foreign leases alike: neither names an
    /// arena that is a scratch space to carve from.
    pub fn inherit_ticket(&self) -> Option<WaveTicket> {
        match self {
            Self::Owned => None,
            Self::Lease(origin) => origin.ticket(),
        }
    }

    /// The backing an allocation seeded by a bare ticket should inherit.
    ///
    /// For the call sites that hold a coordinate rather than an operand — a host
    /// upload, or work that reached another thread over a channel. Absence is
    /// real state: no ticket means no arena, and the allocation is an ordinary
    /// owned one.
    pub fn from_ticket(ticket: Option<WaveTicket>) -> Self {
        match ticket {
            Some(t) => Self::Lease(LeaseOrigin::Wave(t)),
            None => Self::Owned,
        }
    }
}

#[derive(Debug)]
pub struct CudaStorage {
    pub slice: CudaStorageSlice,
    pub device: CudaDevice,
    /// Whether dropping this storage may free its device memory.
    /// [`Backing::Owned`] for everything the pool allocated — which is
    /// everything except the arena leases built by
    /// [`CudaStorage::from_leased_device_ptr`].
    pub backing: Backing,
    /// The holder of the leased memory, kept alive by this storage — see
    /// [`LeaseAnchor`]. `None` for owned storage and for a lease whose memory
    /// outlives the process's use of it (a KV arena slot, a wave range).
    pub anchor: Option<LeaseAnchor>,
}

/// A share in whatever owns the memory behind a lease, carried by the storage so
/// the memory cannot be handed to anyone else while any view of it exists.
///
/// **Why the storage carries it and not the caller.** A lease is a raw address,
/// and a tensor built on one can be cloned, reshaped, narrowed or re-leased
/// (`Tensor::as_foreign_lease`) by code that never sees the owner. Every one of
/// those shares or copies this storage's `Arc`, so a share held *here* travels with
/// all of them and the memory stays reserved until the last view is dropped. Held
/// anywhere else it is a promise that every such copy was found — the promise a
/// stale `kvheads_ptr` word broke for `KvHead` records.
#[derive(Clone)]
pub struct LeaseAnchor(Arc<dyn Any + Send + Sync>);

impl LeaseAnchor {
    /// Anchor a lease to `owner`: the lease's storage, and every view of it, keeps a
    /// share of it.
    pub fn new<T: Any + Send + Sync>(owner: Arc<T>) -> Self {
        Self(owner)
    }
}

impl fmt::Debug for LeaseAnchor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("LeaseAnchor")
            .field(&Arc::as_ptr(&self.0))
            .finish()
    }
}

impl Drop for CudaStorage {
    fn drop(&mut self) {
        // Exhaustive rather than `!= Lease`: a future `Backing` variant must
        // state which side it falls on here, instead of silently inheriting the
        // free path and releasing memory the storage does not own.
        match self.backing {
            Backing::Owned => {
                // A free is refused while this thread records a wave, and the
                // memory may still be read by a recorded launch: it is freed
                // once that launch has been issued.
                let slice = std::mem::replace(&mut self.slice, CudaStorageSlice::Moved);
                self.device.retire(slice);
            }
            Backing::Lease(origin) => {
                let ptr = std::mem::replace(&mut self.slice, CudaStorageSlice::Moved).leak_view();
                // An arena that frees as its storages die takes the range back;
                // every other arena ignores this.
                if let LeaseOrigin::Wave(ticket) = origin {
                    wave_provenance::wave_release(ticket, ptr);
                }
            }
        }
    }
}

pub trait CudaDType: Sized {
    fn as_cuda_slice(s: &CudaStorage) -> Result<&CudaSlice<Self>>;
    fn as_cuda_slice_mut(s: &mut CudaStorage) -> Result<&mut CudaSlice<Self>>;
    fn wrap_cuda_slice(s: CudaSlice<Self>, dev: CudaDevice) -> CudaStorage;

    /// Wrap `len` elements at a device address the caller does not own.
    ///
    /// The typed counterpart of [`CudaStorage::from_leased_device_ptr`], for
    /// generic kernel wrappers that know their output type as a parameter
    /// rather than as a runtime [`DType`]. Same lease semantics: dropping the
    /// storage releases the handle without freeing the memory.
    ///
    /// # Safety
    /// `ptr` must point to at least `len` elements of `Self`, be aligned for
    /// it, and stay live and un-aliased-for-writes for the storage's lifetime.
    unsafe fn wrap_leased_ptr(
        ptr: u64,
        len: usize,
        dev: CudaDevice,
        origin: LeaseOrigin,
    ) -> CudaStorage;
}

macro_rules! cuda_dtype {
    ($ty:ty, $dtype:ident) => {
        impl CudaDType for $ty {
            fn as_cuda_slice(s: &CudaStorage) -> Result<&CudaSlice<Self>> {
                match &s.slice {
                    CudaStorageSlice::$dtype(data) => Ok(&data),
                    _ => Err(crate::Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }

            fn as_cuda_slice_mut(s: &mut CudaStorage) -> Result<&mut CudaSlice<Self>> {
                match s.slice {
                    CudaStorageSlice::$dtype(ref mut data) => Ok(data),
                    _ => Err(crate::Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }

            fn wrap_cuda_slice(slice: CudaSlice<Self>, device: CudaDevice) -> CudaStorage {
                let slice = CudaStorageSlice::$dtype(slice);
                CudaStorage {
                    slice,
                    device,
                    backing: Backing::Owned,
                    anchor: None,
                }
            }

            unsafe fn wrap_leased_ptr(
                ptr: u64,
                len: usize,
                device: CudaDevice,
                origin: LeaseOrigin,
            ) -> CudaStorage {
                let slice = device
                    .stream_unleased()
                    .upgrade_device_ptr::<Self>(ptr, len);
                CudaStorage {
                    slice: CudaStorageSlice::$dtype(slice),
                    device,
                    backing: Backing::Lease(origin),
                    anchor: None,
                }
            }
        }
    };
}

cuda_dtype!(u8, U8);

cuda_dtype!(u32, U32);

cuda_dtype!(i64, I64);

cuda_dtype!(f16, F16);

cuda_dtype!(bf16, BF16);

cuda_dtype!(f32, F32);

cuda_dtype!(f64, F64);

cuda_dtype!(F8E4M3, F8E4M3);

mod breadcrumbs;
mod casting;
mod convolution;
mod copy;
mod ffi_map;
mod gemm;
mod host_transfer;
mod index_ops;
mod single_operand;
mod slice_ptr;
mod sparse_update;
mod storage;
mod two_operand;

pub use breadcrumbs::*;
use convolution::*;
use copy::*;
pub use ffi_map::*;
pub use gemm::*;
use index_ops::*;
use single_operand::*;
pub use slice_ptr::*;
pub use storage::*;
use two_operand::*;
