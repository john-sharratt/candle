//! An aside without CUDA: there is no held ground and no driver to keep work
//! off, so the closure runs as it is.

use candle::{Device, Result};

/// `f()`: see the CUDA build's `aside`.
pub fn aside<R>(_device: &Device, _capacity: usize, f: impl FnOnce() -> Result<R>) -> Result<R> {
    f()
}
