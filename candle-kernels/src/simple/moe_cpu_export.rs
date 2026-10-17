// FFI binding for the CPU class's export kernel.
//
// `run_moe_cpu_export` copies the input rows bucketize chose for the host pool
// — each distinct token its CPU experts route, once (`in_tok[r]` is input row
// r's token) — from the token-major q8a1024 MoE input into the mapped row
// ring, then — from the last block, behind a system fence — stores `seq` into
// the mapped doorbell. With no input rows (`header[7] == 0`) it copies, fences
// and rings nothing. `blocks_done` is a device `u32`, zero between launches.
// See `simple/moe_cpu_export.cu`.

use std::ffi::c_void;

/// Warps per block, mirrored from `EXPORT_WARPS`: one 128-wide tile each.
pub const EXPORT_WARPS: usize = 8;

/// The most blocks an export launch needs: the warps stride over the rows, and
/// a layer's few input rows (a decode step's one, a prompt's hundreds) are a
/// copy of at most a few MB the link takes far longer to move than these warps
/// take to issue.
pub const EXPORT_MAX_BLOCKS: usize = 64;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn run_moe_cpu_export(
        src: *const c_void,
        tiles: i32,
        in_tok: *const c_void,
        header: *const c_void,
        ring: *mut c_void,
        cap_rows: u32,
        doorbell: *mut c_void,
        seq: u64,
        blocks_done: *mut c_void,
        grid: i32,
        stream: *mut c_void,
    ) -> i32;
}
