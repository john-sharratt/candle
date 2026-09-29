use crate::{DType, Layout};

/// cudarc related errors
#[derive(thiserror::Error, Debug)]
pub enum CudaError {
    #[error(transparent)]
    Cuda(#[from] cudarc::driver::DriverError),

    #[error(transparent)]
    Compiler(#[from] cudarc::nvrtc::CompileError),

    #[error(transparent)]
    Cublas(#[from] cudarc::cublas::result::CublasError),

    #[error(transparent)]
    Curand(#[from] cudarc::curand::result::CurandError),

    #[error("missing kernel '{module_name}'")]
    MissingKernel { module_name: String },

    #[error("unsupported dtype {dtype:?} for {op}")]
    UnsupportedDtype { dtype: DType, op: &'static str },

    #[error("internal error '{0}'")]
    InternalError(String),

    #[error("matmul is only supported for contiguous tensors lstride: {lhs_stride:?} rstride: {rhs_stride:?} mnk: {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Layout,
        rhs_stride: Layout,
        mnk: (usize, usize, usize),
    },

    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },

    #[error("{cuda} when loading {module_name}")]
    Load {
        cuda: cudarc::driver::DriverError,
        module_name: String,
    },
}

impl From<CudaError> for crate::Error {
    fn from(val: CudaError) -> Self {
        note_sticky_cuda_fault(&val);
        crate::Error::Cuda(Box::new(val)).bt()
    }
}

pub trait WrapErr<O> {
    fn w(self) -> std::result::Result<O, crate::Error>;
}

impl<O, E: Into<CudaError>> WrapErr<O> for std::result::Result<O, E> {
    fn w(self) -> std::result::Result<O, crate::Error> {
        // Route through `From<CudaError>` so the poison detection below runs on
        // every CUDA error path, not just the direct `?`/`From` ones.
        self.map_err(|e| {
            let cuda: CudaError = e.into();
            crate::Error::from(cuda)
        })
    }
}

/// A sticky CUDA fault leaves the context permanently dead (see
/// [`crate::gpu_poison`]). Flag the context poisoned on the FIRST such error so
/// the daemon can exit cleanly for a restart instead of spewing the same error
/// forever. `OUT_OF_MEMORY` is not sticky on its own — a single occurrence is
/// often just a request too large for the moment — but it feeds the same
/// module's OOM streak, which poisons in its own right once nothing but OOM
/// has been seen for a sustained stretch (`gpu_poison::OOM_STICKY_AFTER`).
fn note_sticky_cuda_fault(err: &CudaError) {
    let driver = match err {
        CudaError::Cuda(e) => e,
        CudaError::Load { cuda, .. } => cuda,
        _ => return,
    };
    let root = || format!("{driver:?}\n{}", super::last_cuda_kernel_launch());
    if is_out_of_memory_driver_error(driver) {
        crate::gpu_poison::note_oom(root);
        return;
    }
    if is_sticky_driver_error(driver) {
        crate::gpu_poison::poison_gpu(root);
    }
}

/// Whether a `DriverError` is `CUDA_ERROR_OUT_OF_MEMORY`. Matched the same way
/// as [`is_sticky_driver_error`] — a stable error-name substring off the cold
/// error path.
fn is_out_of_memory_driver_error(e: &cudarc::driver::DriverError) -> bool {
    format!("{e:?}").contains("OUT_OF_MEMORY")
}

/// Whether a `DriverError` names an unrecoverable, context-killing fault.
/// Matched by the stable CUDA error-name substrings off the cold error path
/// (no dependence on a particular cudarc numeric layout).
fn is_sticky_driver_error(e: &cudarc::driver::DriverError) -> bool {
    // Deliberately excludes OUT_OF_MEMORY (handled separately, as a streak —
    // see `is_out_of_memory_driver_error`) and ordinary API-misuse errors
    // (INVALID_VALUE, NOT_READY, …) which do not kill the context.
    const STICKY: &[&str] = &[
        "ILLEGAL_ADDRESS",
        "LAUNCH_FAILED",
        "LAUNCH_TIMEOUT",
        "MISALIGNED_ADDRESS",
        "ILLEGAL_INSTRUCTION",
        "INVALID_ADDRESS_SPACE",
        "INVALID_PC",
        "HARDWARE_STACK_ERROR",
        "ECC_UNCORRECTABLE",
        "NVLINK_UNCORRECTABLE",
        "ASSERT",
    ];
    let s = format!("{e:?}");
    STICKY.iter().any(|k| s.contains(k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::sys::CUresult;
    use cudarc::driver::DriverError;

    #[test]
    fn out_of_memory_is_recognised_and_not_sticky() {
        let oom = DriverError(CUresult::CUDA_ERROR_OUT_OF_MEMORY);
        assert!(is_out_of_memory_driver_error(&oom));
        assert!(
            !is_sticky_driver_error(&oom),
            "OOM is handled as its own streak, never the immediate-poison path"
        );
    }

    #[test]
    fn illegal_address_is_sticky_and_not_out_of_memory() {
        let fault = DriverError(CUresult::CUDA_ERROR_ILLEGAL_ADDRESS);
        assert!(is_sticky_driver_error(&fault));
        assert!(!is_out_of_memory_driver_error(&fault));
    }

    #[test]
    fn ordinary_api_misuse_is_neither() {
        let misuse = DriverError(CUresult::CUDA_ERROR_INVALID_VALUE);
        assert!(!is_sticky_driver_error(&misuse));
        assert!(!is_out_of_memory_driver_error(&misuse));
    }
}
