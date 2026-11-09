//! The local memory a context holds, measured.
//!
//! Every thread a kernel runs has a stack in local memory, and the driver keeps
//! one per thread the device can hold resident at once — sized to the per-thread
//! stack limit, which it raises to whatever the most demanding kernel launched so
//! far needed (a kernel that spills raises it). The memory is the driver's own,
//! taken after the span is claimed and never released by a later, lighter
//! kernel, so it is part of what a span report reads as "outside the span" and no
//! allocation of the engine's names.

use cudarc::driver::sys::{self, CUdevice_attribute, CUlimit};

use super::{CudaDevice, WrapErr};
use crate::Result;

/// Bytes of local memory the driver holds for `dev`'s context now: the
/// per-thread stack limit across every thread the device can hold resident.
pub fn local_memory_bytes(dev: &CudaDevice) -> Result<usize> {
    let ctx = dev.cuda_stream().context().clone();
    ctx.bind_to_thread().w()?;
    let mut stack: usize = 0;
    // SAFETY: a valid out-pointer, on the context bound just above.
    unsafe { sys::cuCtxGetLimit(&mut stack, CUlimit::CU_LIMIT_STACK_SIZE) }
        .result()
        .w()?;
    let sms = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .w()?;
    let threads = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
        .w()?;
    Ok(resident_stack_bytes(stack, sms as usize, threads as usize))
}

/// One `stack`-byte stack for each of `threads` resident threads on each of
/// `sms` multiprocessors.
fn resident_stack_bytes(stack: usize, sms: usize, threads: usize) -> usize {
    stack * sms * threads
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendDevice;

    #[test]
    fn every_resident_thread_holds_one_stack() {
        assert_eq!(resident_stack_bytes(1024, 110, 1536), 173_015_040);
        assert_eq!(resident_stack_bytes(0, 110, 1536), 0);
    }

    /// A context has a stack for every resident thread from its first launch.
    #[test]
    fn a_context_holds_local_memory() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        assert!(local_memory_bytes(&dev)? > 0);
        Ok(())
    }
}
