// FFI binding for the CPU expert pool's handoff measurement kernel.
//
// `run_handoff_ping` launches one thread that rings `doorbell` with 1..n and,
// after each ring, spins on `done` until the host answers that value. Both are
// device addresses of mapped host words on separate cache lines. A wait past
// `limit_ns` writes its iteration to `gave_up` (also mapped) and ends the
// kernel. See `simple/handoff_ping.cu`.

use std::ffi::c_void;

extern "C" {
    pub fn run_handoff_ping(
        doorbell: *mut u32,
        done: *const u32,
        n: u32,
        limit_ns: u64,
        gave_up: *mut u32,
        stream: *mut c_void,
    );
}
