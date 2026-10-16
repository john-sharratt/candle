//! The allocator the CUDA launchers' own grow-only scratch is taken from.
//!
//! Some launchers keep a scratch they size from the launch shape — the INT8
//! decode, glue and prefill attention's split-KV partials, the provenance
//! scans' accumulators — and grow it inside the launcher, where no host code
//! sees it (`candle-kernels`' `grow_scratch.cuh`). It grew with `cudaMalloc`:
//! memory outside the reservation and outside every report.
//!
//! The launchers now call back here for it and get a [`Held`] block: held
//! ground when the device has some, a recorded pool allocation when not. The
//! block lives until its launcher replaces it with a larger one and hands it
//! back, and is then retired through the device ([`CudaDevice::retire`]): a
//! launch already queued, or recorded into the segment being captured, may still
//! read it, so its range goes back only once that segment has been issued.
//!
//! Keeping every replaced block instead — the earlier rule, on the reasoning
//! that a recorded graph might replay one — stranded each step of the growth for
//! the life of the process. A recorded wave's executables are refreshed from its
//! recapture every wave (`docs/decode_graphs.md` §0.1), so no replay reads an
//! address the launcher has moved off.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use candle_kernels::paged_decode::candle_set_kernel_scratch;
use cudarc::driver::DevicePtr;

use super::held::Held;
use super::CudaDevice;

/// Each device's handle, and the blocks its launchers hold by address.
#[derive(Default)]
struct Scratch {
    devices: HashMap<usize, CudaDevice>,
    blocks: HashMap<u64, Held<u8>>,
}

fn scratch() -> &'static Mutex<Scratch> {
    static SCRATCH: OnceLock<Mutex<Scratch>> = OnceLock::new();
    SCRATCH.get_or_init(|| Mutex::new(Scratch::default()))
}

/// Serve `bytes` for a launcher's scratch on device `device`.
unsafe extern "C" fn take_kernel_scratch(device: i32, bytes: usize) -> *mut c_void {
    let mut s = scratch().lock().unwrap_or_else(|e| e.into_inner());
    let Some(dev) = s.devices.get(&(device as usize)) else {
        return std::ptr::null_mut();
    };
    let stream = dev.compute_stream();
    match Held::<u8>::alloc_on(&stream, bytes) {
        Ok(held) => {
            let ptr = held.device_ptr(&stream).0;
            s.blocks.insert(ptr, held);
            ptr as *mut c_void
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Take back a block a launcher has replaced, and free it behind every launch
/// that may still read it.
unsafe extern "C" fn release_kernel_scratch(device: i32, block: *mut c_void) {
    let (dev, held) = {
        let mut s = scratch().lock().unwrap_or_else(|e| e.into_inner());
        let held = s.blocks.remove(&(block as u64));
        (s.devices.get(&(device as usize)).cloned(), held)
    };
    // Retired outside the lock: dropping a block gives its range back to the
    // ground, which takes the ground's own lock.
    if let (Some(dev), Some(held)) = (dev, held) {
        dev.retire(held);
    }
}

/// Register `dev` as its ordinal's handle and install the allocator. The first
/// handle registered for an ordinal is kept.
pub(crate) fn install(dev: &CudaDevice) {
    scratch()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .devices
        .entry(dev.compute_stream().context().ordinal())
        .or_insert_with(|| dev.clone());
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        // SAFETY: plain function pointers with the signatures the launchers call.
        unsafe { candle_set_kernel_scratch(take_kernel_scratch, release_kernel_scratch) }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendDevice;

    /// A block a launcher replaces is given back, and only that one: the
    /// launcher's current block stays held.
    #[test]
    fn a_replaced_block_is_released_and_the_current_one_kept() {
        let dev = CudaDevice::new(0).unwrap();
        let ordinal = dev.compute_stream().context().ordinal() as i32;
        let held = |p: *mut c_void| scratch().lock().unwrap().blocks.contains_key(&(p as u64));
        // SAFETY: the allocator's own entry points, as a launcher calls them.
        unsafe {
            let first = take_kernel_scratch(ordinal, 1 << 20);
            assert!(!first.is_null());
            let second = take_kernel_scratch(ordinal, 2 << 20);
            assert!(!second.is_null());
            assert!(held(first) && held(second));
            release_kernel_scratch(ordinal, first);
            assert!(!held(first), "the replaced block went back");
            assert!(held(second), "the current block is still the launcher's");
            release_kernel_scratch(ordinal, second);
        }
        dev.synchronize().unwrap();
    }
}
