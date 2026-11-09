//! A watch on every device allocation cudarc makes.
//!
//! This copy of cudarc is vendored for this hook alone. A caller installs an
//! [AllocWatch] once; from then on [CudaStream::alloc] (and everything built on
//! it — `alloc_zeros`, `memcpy_stod`, `clone_htod`, [CudaStream::null]) reports
//! each request to it before the driver allocates. The watch observes; it does
//! not serve. Without one installed every path is the upstream one.

use std::sync::{Arc, OnceLock};

use super::core::CudaStream;

/// Told of every device allocation before the driver makes it.
pub trait AllocWatch: Send + Sync {
    /// `bytes` of device memory are about to be allocated for `stream`.
    fn alloc(&self, stream: &Arc<CudaStream>, bytes: usize);
}

static WATCH: OnceLock<Arc<dyn AllocWatch>> = OnceLock::new();

/// Install the process's allocation watch. Once: `false` when one is already
/// installed, which is left in place.
pub fn install_alloc_watch(watch: Arc<dyn AllocWatch>) -> bool {
    WATCH.set(watch).is_ok()
}

/// The installed watch, if any.
pub(crate) fn alloc_watch() -> Option<&'static Arc<dyn AllocWatch>> {
    WATCH.get()
}
