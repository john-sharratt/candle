//! Every kernel module is loaded when the context is created, never on a
//! kernel's first launch.
//!
//! Since CUDA 12.2 the driver loads a module lazily, the first time one of its
//! kernels launches, and that load can synchronise the context. The driver
//! refuses a synchronise while any stream captures, and the refusal invalidates
//! the capture — on whichever thread is recording, whichever thread launched.
//! The scheduler thread records every wave; the persistence thread, the expert
//! workers and npcd's tenants launch kernels of their own. The first time one of
//! them launched a kernel nothing had used yet, the wave beside it lost its
//! segment: measured on `.5`, 2026-10-17, a `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`
//! two seconds into serving as the persistence thread's first hot→warm pass ran
//! its quantize kernels, and `candle-core`'s graph tests failing whenever the
//! MoE live-launch tests ran beside them.
//!
//! The driver reads its loading mode once, at initialisation, from
//! `CUDA_MODULE_LOADING`. So the mode is set here, before this crate creates its
//! first context, and **verified** after: a process whose driver was initialised
//! some other way first runs lazily, and that is refused at the first device
//! rather than found as a lost wave.

use std::env::set_var;
use std::sync::Once;

use cudarc::driver::sys::{cuModuleGetLoadingMode, CUmoduleLoadingMode, CUresult};

use crate::Result;

/// What the driver reads its loading mode from, and the mode.
const MODE_VAR: (&str, &str) = ("CUDA_MODULE_LOADING", "EAGER");

/// Ask the driver for eager loading. Before the first context of the process;
/// once.
pub(crate) fn request_eager() {
    static ASKED: Once = Once::new();
    ASKED.call_once(|| {
        let (key, value) = MODE_VAR;
        set_var(key, value);
    });
}

/// Refuse a driver that loads lazily — see the module docs.
pub(crate) fn verify_eager() -> Result<()> {
    let mut mode = CUmoduleLoadingMode::CU_MODULE_LAZY_LOADING;
    // SAFETY: a plain query writing one enum through a valid pointer.
    let status = unsafe { cuModuleGetLoadingMode(&mut mode) };
    if status != CUresult::CUDA_SUCCESS {
        crate::bail!("the CUDA driver could not report its module loading mode: {status:?}");
    }
    check_mode(mode)
}

fn check_mode(mode: CUmoduleLoadingMode) -> Result<()> {
    match mode {
        CUmoduleLoadingMode::CU_MODULE_EAGER_LOADING => Ok(()),
        CUmoduleLoadingMode::CU_MODULE_LAZY_LOADING => crate::bail!(
            "the CUDA driver loads kernel modules lazily: it was initialised before this \
             crate could ask for eager loading (CUDA_MODULE_LOADING=EAGER). A lazy load on \
             one thread invalidates a graph capture on another, so this process cannot run \
             waves safely; initialise CUDA through candle's device first"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eager_loading_passes_and_lazy_is_refused() {
        assert!(check_mode(CUmoduleLoadingMode::CU_MODULE_EAGER_LOADING).is_ok());
        assert!(check_mode(CUmoduleLoadingMode::CU_MODULE_LAZY_LOADING).is_err());
    }

    /// A process that made a device runs eagerly — the property the module
    /// exists for, checked against the driver itself.
    #[test]
    fn a_device_leaves_the_driver_loading_eagerly() {
        use crate::backend::BackendDevice;
        let _dev = crate::CudaDevice::new(0).unwrap();
        verify_eager().unwrap();
    }
}
