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

/// The most KV heads a job may have: a block's tile holds whole jobs. Mirrors
/// `MAX_KV_HEAD` in the `.cu`.
pub const MAX_KV_HEAD: usize = 64;

/// Heads per block's tile (whole jobs only). Mirrors `TILE_HEADS` in the `.cu`.
pub const TILE_HEADS: usize = 64;

/// Status words (`u64`) a claim launch over up to `n_heads` heads needs: one
/// per (tile, class). At any `n_kv_head <= MAX_KV_HEAD <= TILE_HEADS` a tile
/// holds more than half of `TILE_HEADS` heads.
///
/// Zero them once when they are allocated and hand them to nothing else, then
/// pass a different `epoch` with every launch on them (a counter beside the
/// buffer): a word an earlier launch left reads as not ready by its epoch, so
/// nothing is cleared between launches. Launches sharing one status buffer
/// must be ordered (one stream).
pub const fn claim_status_words(n_heads: usize) -> usize {
    n_heads.div_ceil(TILE_HEADS / 2) * MAX_CLASS
}

extern "C" {
    /// One policy group's claim: give every band of `n_jobs × n_kv_head` heads
    /// its index in its size class, continuing from `taken` (which it leaves
    /// advanced past the group), defer each job any of whose bands falls past
    /// its class's lease, and write every head's destination KvHead record into
    /// `dst_records` (the convert buffer's destination half) and into
    /// `summary_heads`, each band's lease index after the record and each job's
    /// deferred flag in `summary_flags`. Returns 0, or nonzero when it refused
    /// the arguments (`head_dim` other than 128 or 256, the convert's; `n_kv_head`
    /// past [`MAX_KV_HEAD`]; `n_class` past [`MAX_CLASS`]) or the launch failed.
    ///
    /// # Safety
    ///
    /// Device pointers: tags and scales hold `n_heads × 4`, maps
    /// `n_heads × head_dim`; `code_class`/`code_fmt` hold [`N_CODES`];
    /// `lease_len`, `stack_off` and `taken` hold `n_class <= MAX_CLASS`;
    /// `stack` every leased slot address; `dst_records`, `summary_heads` and
    /// `summary_flags` are sized for the group (the summary may be mapped host
    /// memory); `error_word` is one word; `status` holds
    /// [`claim_status_words`]`(n_heads)`, and `epoch` differs from every
    /// earlier launch's on it.
    pub fn run_kv_gap_claim(
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
        lease_len: *const u32,
        stack: *const u64,
        stack_off: *const u32,
        taken: *mut u32,
        sink: u64,
        dst_records: *mut u8,
        summary_heads: *mut u8,
        summary_flags: *mut u32,
        error_word: *mut u32,
        status: *mut u64,
        epoch: u32,
        n_jobs: i32,
        n_kv_head: i32,
        head_dim: i32,
        n_class: i32,
        stream: *mut c_void,
    ) -> i32;

    /// Store `seq` in `word` after everything enqueued before it on `stream`,
    /// behind a system-scope fence: the host's signal that the gap is done.
    ///
    /// # Safety
    ///
    /// `word` is a device-visible address of one `u32`, normally mapped host
    /// memory.
    ///
    /// Returns the launch's own status: 0 when queued, else the `cudaError_t`
    /// the launch returned.
    pub fn run_kv_gap_complete(word: *mut u32, seq: u32, stream: *mut c_void) -> i32;
}
