// FFI binding for the warm lift's scatter kernel (docs/warm_lift.md).
//
// `run_warm_lift_scatter` copies every band one pinned ring slot holds into its
// destination arena slot. `slot` and `table` are the ring slot's mapped pinned
// memory, at their host addresses: the kernel reads the payloads and the
// descriptor table straight from host memory, so nothing is uploaded.
//
//   slot:      the slot's payload base, mapped pinned host memory
//   table:     `n_entries` [`WarmLiftEntry`]s, mapped pinned host memory
//   stream:    cudaStream_t

use std::ffi::c_void;

/// One band of a ring slot: `bytes` of payload at `slot + src_off`, copied to
/// the device address `dst`. The layout the kernel reads (`warm_lift.cu`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarmLiftEntry {
    pub src_off: u64,
    pub dst: u64,
    pub bytes: u64,
}

extern "C" {
    pub fn run_warm_lift_scatter(
        slot: *const u8,
        table: *const WarmLiftEntry,
        n_entries: i32,
        stream: *mut c_void,
    );
}
