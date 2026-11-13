//! FFI bindings for the gap's destination claim (`simple/kv_gap_claim.cu`,
//! `docs/kv_quantize_in_the_gap.md` §3.2–§3.4).

use std::ffi::c_void;

/// The most size classes the claim kernel counts; the host refuses a ladder
/// longer than this. Mirrors `MAX_CLASS` in the `.cu`.
pub const MAX_CLASS: usize = 16;

/// Length of the code tables (`code_class`, `code_fmt`): every select code is
/// below this. Mirrors `N_CODES` in the `.cu`.
pub const N_CODES: usize = 64;

/// Bands per head: four palettes, K and V. Mirrors `BANDS_PER_HEAD`.
pub const BANDS_PER_HEAD: usize = 8;

/// Bytes of one head's summary row at `head_dim`: its destination KvHead
/// record (`head_dim / 2 + 104`) followed by its eight bands' lease indices.
pub const fn summary_head_bytes(head_dim: usize) -> usize {
    head_dim / 2 + 104 + BANDS_PER_HEAD * 4
}

/// The claim's "no override" value for a side's uniform format.
pub const NO_OVERRIDE: i32 = -1;

extern "C" {
    /// Give every band of `n_jobs × n_kv_head` heads its index in its size
    /// class, continuing from `taken`, and mark each job deferred when any band
    /// falls past its class's lease. One block.
    ///
    /// # Safety
    ///
    /// Device pointers: `k_tags`/`v_tags` hold `n_jobs × n_kv_head × 4` select
    /// codes; `code_class` holds [`N_CODES`] entries; `lease_len` and `taken`
    /// hold `n_class` entries, `n_class <= MAX_CLASS`; `band_index` holds
    /// `n_jobs × n_kv_head × 8`; `job_deferred` holds `n_jobs`; `error_word`
    /// is one word, device or mapped host.
    pub fn run_kv_gap_claim(
        k_tags: *const i32,
        v_tags: *const i32,
        k_override: i32,
        v_override: i32,
        code_class: *const i8,
        lease_len: *const u32,
        taken: *mut u32,
        band_index: *mut u32,
        job_deferred: *mut u32,
        error_word: *mut u32,
        n_jobs: i32,
        n_kv_head: i32,
        n_class: i32,
        stream: *mut c_void,
    );

    /// Write every head's destination KvHead record into `dst_records` (the
    /// convert buffer's destination half) and into `summary_heads`, with each
    /// band's lease index after the record and each job's deferred flag in
    /// `summary_flags`. One block per head.
    ///
    /// # Safety
    ///
    /// Device pointers sized for `n_heads` heads at `head_dim`; `stack` holds
    /// every leased slot address, `stack_off` each class's first entry;
    /// `summary_heads` and `summary_flags` may be mapped host memory.
    pub fn run_kv_gap_fill(
        k_tags: *const i32,
        v_tags: *const i32,
        k_scale: *const f32,
        v_scale: *const f32,
        k_map: *const i32,
        v_map: *const i32,
        k_override: i32,
        v_override: i32,
        code_class: *const i8,
        code_fmt: *const u8,
        stack: *const u64,
        stack_off: *const u32,
        band_index: *const u32,
        job_deferred: *const u32,
        sink: u64,
        dst_records: *mut u8,
        summary_heads: *mut u8,
        summary_flags: *mut u32,
        n_heads: i32,
        head_dim: i32,
        n_kv_head: i32,
        stream: *mut c_void,
    );

    /// Store `seq` in `word` after everything enqueued before it on `stream`,
    /// behind a system-scope fence: the host's signal that the gap is done.
    ///
    /// # Safety
    ///
    /// `word` is a device-visible address of one `u32`, normally mapped host
    /// memory.
    pub fn run_kv_gap_complete(word: *mut u32, seq: u32, stream: *mut c_void);
}
