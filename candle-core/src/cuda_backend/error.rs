#[cfg(feature = "tensor-assert")]
use std::sync::atomic::{AtomicUsize, Ordering};

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
        traced(crate::Error::Cuda(Box::new(val)))
    }
}

/// How many of a `tensor-assert` build's CUDA errors carry their stack.
///
/// **The first failing call names itself.** A diagnostic motherd on WhiteWorld
/// failed its first wave on `CUDA_ERROR_INVALID_VALUE` and every wave after on
/// the capture it left open, and the log said neither which call refused nor
/// from where: services run without `RUST_BACKTRACE`. The first few errors of
/// the process are the ones worth a stack; once a stream is wedged, thousands
/// follow a second, and a stack on each would be the whole log.
#[cfg(feature = "tensor-assert")]
const TRACED_CUDA_ERRORS: usize = 32;

/// `e` with its stack: forced for a `tensor-assert` build's first
/// [`TRACED_CUDA_ERRORS`], as the environment says otherwise.
fn traced(e: crate::Error) -> crate::Error {
    #[cfg(feature = "tensor-assert")]
    {
        static TRACED: AtomicUsize = AtomicUsize::new(0);
        traced_by(e, &TRACED)
    }
    #[cfg(not(feature = "tensor-assert"))]
    e.bt()
}

/// [`traced`] against `count`, the errors traced so far.
#[cfg(feature = "tensor-assert")]
fn traced_by(e: crate::Error, count: &AtomicUsize) -> crate::Error {
    match count.fetch_add(1, Ordering::Relaxed) < TRACED_CUDA_ERRORS {
        true => e.bt_forced(),
        false => e.bt(),
    }
}

#[cfg(all(test, feature = "tensor-assert"))]
mod traced_tests {
    use super::*;

    /// **A diagnostic build's first CUDA errors carry their stack** with no
    /// `RUST_BACKTRACE` set, and once [`TRACED_CUDA_ERRORS`] have, the rest
    /// carry what the environment asks for.
    #[test]
    fn the_first_cuda_errors_carry_their_stack() {
        let count = AtomicUsize::new(0);
        let error = || crate::Error::Cuda(Box::new(CudaError::InternalError("x".into())));
        for _ in 0..TRACED_CUDA_ERRORS {
            assert!(matches!(
                traced_by(error(), &count),
                crate::Error::WithBacktrace { .. }
            ));
        }
        assert_eq!(
            std::mem::discriminant(&traced_by(error(), &count)),
            std::mem::discriminant(&error().bt()),
            "past the first, what the environment asks for"
        );
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
    let root = || format!("{driver:?}\n{}", super::every_thread_kernel_launches());
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

    /// A sticky fault's record names the launches of threads other than the one
    /// that noticed it — the thread that synchronises first is usually not the
    /// one that launched the faulting kernel.
    #[test]
    fn the_fault_record_shows_every_live_threads_launches() {
        use crate::cuda_backend::{every_thread_kernel_launches, set_kernel_breadcrumb};
        use std::sync::mpsc;
        let (marked_tx, marked_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let worker = std::thread::Builder::new()
            .name("breadcrumb-probe".into())
            .spawn(move || {
                set_kernel_breadcrumb("probe_kernel", "probe.rs", 7);
                marked_tx.send(()).expect("the test is waiting");
                let _ = done_rx.recv();
            })
            .expect("spawn the probe thread");
        marked_rx.recv().expect("the probe marked its launch");
        let record = every_thread_kernel_launches();
        done_tx.send(()).expect("the probe is waiting");
        worker.join().expect("the probe thread panicked");
        assert!(
            record.contains("thread 'breadcrumb-probe':")
                && record.contains("'probe_kernel' (probe.rs:7)"),
            "another thread's launch is missing from the fault record:\n{record}"
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
