//! Host memory the GPU may still address.
//!
//! Page-locked host memory is not only a host resource. Under WDDM every byte
//! the driver locks for the device is charged against the adapter's
//! **NON_LOCAL** segment budget — host memory the GPU can address — and that
//! budget is smaller than what the driver will lock: on the 31.5 GiB RTX 4090
//! Laptop box it is 17.22 GiB, and usage tracked pinned bytes exactly from
//! 2 GiB to 16.07 GiB (`dxgi_budget_probe::print_pinned_dma_under_pinning`).
//!
//! The driver keeps granting locks while the budget lasts, so a process can pin
//! almost all of it and only then discover that the device needs non-local room
//! of its own: a staging buffer for a pageable upload, the paging of its own
//! allocations when VRAM is full. Measured on the same box: the expert warm
//! tier pinned 16.02 GiB, leaving 1.2 GiB of budget, and the startup fill's
//! first upload failed with `CUDA_ERROR_OUT_OF_MEMORY` on run after run — with
//! between 0.9 and 2.7 GiB of host RAM still free, so not a RAM shortage.
//!
//! [`gpu_addressable_room`] is what is left of that budget. A pinned tier sizes
//! its locked part against it; `None` means the platform has no such budget to
//! read (Linux, a TCC driver) and only the driver's own refusal bounds a lock.

use crate::cuda_backend::CudaDevice;

/// Bytes of the NON_LOCAL segment budget not yet in use by this process, or
/// `None` where the platform reports no such budget.
pub fn gpu_addressable_room(device: &CudaDevice) -> Option<u64> {
    room(device)
}

#[cfg(windows)]
fn room(device: &CudaDevice) -> Option<u64> {
    let probe = super::DxgiProbe::for_cuda_device(device).ok()?;
    let (budget, usage) = probe.non_local().ok()?;
    Some(budget.saturating_sub(usage))
}

#[cfg(not(windows))]
fn room(_device: &CudaDevice) -> Option<u64> {
    None
}
