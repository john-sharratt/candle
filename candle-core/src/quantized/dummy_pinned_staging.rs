//! No-op stubs for [`Generation`] and [`PinnedStager`] on non-CUDA builds.
//!
//! The real implementation lives in `pinned_staging.rs` and is only compiled
//! with `feature = "cuda"`.  This module provides the same types
//! so that call-sites can use them unconditionally.

use crate::Device;

/// No-op generation guard for non-CUDA builds.
///
/// On CUDA this is an RAII guard that keeps the pinned arena alive.
/// Without CUDA it carries no state and all methods are no-ops.
pub struct Generation {
    _private: (),
}

impl Generation {
    /// Create a no-op generation (non-CUDA builds).
    pub fn noop() -> Self {
        Self { _private: () }
    }
}

/// No-op pinned-memory stager for non-CUDA builds.
#[derive(Clone, Debug)]
pub struct PinnedStager {
    _private: (),
}

impl PinnedStager {
    /// Create a no-op stager. Ignores the device.
    pub fn new_from_device(_device: &Device) -> Self {
        Self { _private: () }
    }

    /// Begin a no-op generation.
    pub fn begin_generation(&self) -> Generation {
        Generation::noop()
    }
}

/// No-op GPU buffer for non-CUDA builds.
///
/// On CUDA this holds a device pointer into the pinned-staging arena.
/// Without CUDA it is never constructed; `Option<GpuBuf>` is always `None`.
#[derive(Clone)]
pub struct GpuBuf {
    _private: (),
}

impl GpuBuf {
    /// Raw device pointer — always 0 on non-CUDA builds.
    pub fn dev_ptr(&self) -> u64 {
        0
    }

    /// Buffer length in bytes — always 0 on non-CUDA builds.
    pub fn len(&self) -> usize {
        0
    }

    /// Always empty on non-CUDA builds; mirrors the CUDA `GpuBuf`'s API so
    /// callers need no `cfg` of their own.
    pub fn is_empty(&self) -> bool {
        true
    }

    /// No-op constructor for non-CUDA builds.
    pub fn from_borrowed(_dev_ptr: u64, _len: usize) -> Self {
        Self { _private: () }
    }
}
