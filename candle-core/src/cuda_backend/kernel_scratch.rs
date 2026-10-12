//! The allocator the CUDA launchers' own grow-only scratch is taken from.
//!
//! Some launchers keep a scratch they size from the launch shape — the INT8
//! decode, glue and prefill attention's split-KV partials — and grow it inside
//! the launcher, where no host code sees it (`candle-kernels`'
//! `grow_scratch.cuh`). It grew with `cudaMalloc`: memory outside the
//! reservation and outside every report, never given back because a recorded
//! graph may still read a replaced block.
//!
//! The launchers now call back here for it, and get a [`Held`] block: held
//! ground when the device has some, a recorded pool allocation when not. The
//! block is kept for the life of the process, exactly as the launcher keeps it.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use candle_kernels::paged_decode::candle_set_kernel_scratch;
use cudarc::driver::{CudaStream, DevicePtr};

use super::held::Held;

fn streams() -> &'static Mutex<HashMap<usize, Arc<CudaStream>>> {
    static STREAMS: OnceLock<Mutex<HashMap<usize, Arc<CudaStream>>>> = OnceLock::new();
    STREAMS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Serve `bytes` for a launcher's scratch on device `device`.
unsafe extern "C" fn take_kernel_scratch(device: i32, bytes: usize) -> *mut c_void {
    let Some(stream) = streams()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&(device as usize))
        .cloned()
    else {
        return std::ptr::null_mut();
    };
    match Held::<u8>::alloc_on(&stream, bytes) {
        Ok(held) => {
            let ptr = held.device_ptr(&stream).0;
            // Kept for the life of the process: the launcher never gives a block
            // back, since a recorded graph may still read one it has replaced.
            std::mem::forget(held);
            ptr as *mut c_void
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Register device `ordinal`'s compute stream and install the allocator.
/// Idempotent; the first stream registered for an ordinal is kept.
pub(crate) fn install(ordinal: usize, stream: &Arc<CudaStream>) {
    streams()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry(ordinal)
        .or_insert_with(|| stream.clone());
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        // SAFETY: a plain function pointer with the signature the launchers call.
        unsafe { candle_set_kernel_scratch(take_kernel_scratch) }
    });
}
