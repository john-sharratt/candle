// FFI binding for the NVTX3 range shim. See `simple/nvtx.cu` for why the
// implementation is a compiled shim rather than a Rust NVTX binding.

use std::ffi::c_char;

extern "C" {
    /// Open an NVTX range and return its id.
    ///
    /// `message` must be a NUL-terminated string that stays valid for the
    /// duration of the call; NVTX copies it.
    pub fn candle_nvtx_range_start(message: *const c_char) -> u64;

    /// Close the range opened by [`candle_nvtx_range_start`].
    ///
    /// Ranges use the START/END form, so they may overlap and may close out of
    /// order — unlike push/pop, which requires strict nesting.
    pub fn candle_nvtx_range_end(id: u64);
}
