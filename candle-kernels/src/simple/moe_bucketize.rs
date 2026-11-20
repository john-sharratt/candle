// FFI binding for the GPU MoE expert-bucketize kernel.
//
// See `simple/moe_bucketize.cu` for the authoritative contract: outputs,
// stability guarantees, and the padding conventions. All array arguments are
// device pointers on `stream`; every output/scratch buffer is sized to the
// `n_tokens × k` launch bound; `header` is `i32[HEADER_WORDS]` and
// `token_starts` is `i32[n_tokens + 1]`.

use std::ffi::c_void;

/// Kernel bound on experts (`n_experts`), mirrored from `MAX_EXPERTS` in the
/// `.cu`. The Rust wrapper validates against THESE constants so its checks can
/// never drift from the launcher's silent-return guards (a drifted wrapper
/// would skip the launch and leave the workspace holding the PREVIOUS layer's
/// tables).
pub const MAX_EXPERTS: usize = 512;
/// Kernel bound on top-k width (`k`), mirrored from `MAX_TOPK` in the `.cu`.
pub const MAX_TOPK: usize = 32;
/// Kernel bound on the decode-scored token ranges, mirrored from
/// `MAX_DECODE_RANGES` in the `.cu`.
pub const MAX_DECODE_RANGES: usize = 32;

/// A promotion offer with no resident expert behind it, mirrored from
/// `PROMO_EMPTY` in the `.cu`.
pub const PROMO_EMPTY: u64 = u64::MAX;
/// The expert a skipped victim's log entry names, mirrored from `PROMO_SKIP`.
pub const PROMO_SKIP: u32 = 0xffff;

/// Read-ahead items one bucketize may write, mirrored from `AHEAD_MAX` in
/// `moe_read_ahead.cuh` — which documents the item layout.
pub const AHEAD_MAX: usize = 64;
/// `u64` words per read-ahead item, mirrored from `AHEAD_ITEM_WORDS`.
pub const AHEAD_ITEM_WORDS: usize = 11;
/// Pieces each read-ahead item is copied in, mirrored from `AHEAD_CHUNKS`.
pub const AHEAD_CHUNKS: usize = 16;
/// Set in a promotion log entry's expert field for a read-ahead claim,
/// mirrored from `AHEAD_FLAG`.
pub const AHEAD_FLAG: u32 = 0x4000;
/// Set with [`AHEAD_FLAG`] for a claim that promotes an expert the previous
/// launch's CPU pool served (the carry), mirrored from `CARRY_FLAG`.
pub const CARRY_FLAG: u32 = 0x2000;

/// Words in the device header: `{n_active, total_valid, num_tiles, remote
/// experts, remote tiles, CPU rows, CPU experts, CPU input rows}`.
pub const HEADER_WORDS: usize = 8;

/// Routed counts the CPU split's cost tables cover, mirrored from `CPU_MAX_N`
/// in `moe_bucketize_cpu.cuh`.
pub const CPU_MAX_N: usize = 32;
/// The summary bit a CPU-class expert carries, mirrored from `SUMMARY_CPU_BIT`;
/// the routed count is the 28 bits below it.
pub const SUMMARY_CPU_BIT: u32 = 28;
/// The summary word's count field.
pub const SUMMARY_COUNT_MASK: u32 = (1 << SUMMARY_CPU_BIT) - 1;
/// Bytes of one CPU work slot, mirrored from `CPU_WORK_SLOT_BYTES`: a 16-byte
/// header, then `MAX_EXPERTS` 40-byte expert entries.
pub const CPU_WORK_SLOT_BYTES: usize = 16 + MAX_EXPERTS * 40;
/// `u32` words of the carry, mirrored from `CPU_CARRY_WORDS`: `{seq, n}`, then
/// `AHEAD_MAX` `(row, expert)` pairs.
pub const CPU_CARRY_WORDS: usize = 2 + 2 * AHEAD_MAX;

/// [`CpuSplit::promote`], mirrored from `CPU_PROMOTE_*`: which experts the CPU
/// pool served the next launch copies into VRAM behind it — none, the cold
/// ones, or every one.
pub const CPU_PROMOTE_NONE: u32 = 0;
pub const CPU_PROMOTE_COLD: u32 = 1;
pub const CPU_PROMOTE_ALL: u32 = 2;

/// The CPU split's costs, by value in the launch parameters
/// (`moe_bucketize_cpu.cuh` documents each). `cap_rows == 0` turns the CPU
/// class off.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuSplit {
    pub cap_rows: u32,
    pub max_n: u32,
    pub copy_ns: f32,
    pub read_shared_ns: f32,
    pub read_alone_ns: f32,
    pub handoff_ns: f32,
    pub row_ns: f32,
    pub min_gain_ns: f32,
    pub explore_ns: f32,
    pub promote: u32,
    /// Most CPU experts a launch takes: what the pool's performance cores
    /// carry alone. The rest of its misses stay on the GPU.
    pub max_experts: u32,
    pub cpu_n_ns: [f32; CPU_MAX_N + 1],
    pub gpu_n_ns: [f32; CPU_MAX_N + 1],
}

impl CpuSplit {
    /// The CPU class off.
    pub const OFF: Self = Self {
        cap_rows: 0,
        max_n: 0,
        copy_ns: 0.0,
        read_shared_ns: 0.0,
        read_alone_ns: 0.0,
        handoff_ns: 0.0,
        row_ns: 0.0,
        min_gain_ns: 0.0,
        explore_ns: 0.0,
        promote: CPU_PROMOTE_NONE,
        max_experts: 0,
        cpu_n_ns: [0.0; CPU_MAX_N + 1],
        gpu_n_ns: [0.0; CPU_MAX_N + 1],
    };
}

/// [`run_moe_bucketize`] launched the kernel.
pub const BUCKETIZE_LAUNCHED: i32 = 0;
/// [`run_moe_bucketize`]'s argument guards refused the call; nothing was written.
pub const BUCKETIZE_REFUSED: i32 = 1;
/// The launch itself returned an error.
pub const BUCKETIZE_LAUNCH_FAILED: i32 = 2;
/// An earlier launch on the calling thread had left an error pending; nothing
/// was launched.
pub const BUCKETIZE_EARLIER_FAILURE: i32 = 3;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn run_moe_bucketize(
        topk_ids: *const c_void,
        n_tokens: i32,
        k: i32,
        n_experts: i32,
        tile_w: i32,
        tok_ids: *mut c_void,
        weight_ids: *mut c_void,
        tile_expert: *mut c_void,
        tile_b_start: *mut c_void,
        tile_b_cnt: *mut c_void,
        perm: *mut c_void,
        rw_ids: *mut c_void,
        token_starts: *mut c_void,
        header: *mut c_void,
        inv: *mut c_void,
        scan: *mut c_void,
        // This layer's row of the live gate table, or null when every expert is
        // in VRAM. The up and down rows are `table_plane` and `2·table_plane`
        // entries after it.
        gate_row: *const c_void,
        table_plane: i64,
        // `u64[3][n_experts]` snapshot of the routed experts' entries (gate,
        // up, down); required when `gate_row` is set.
        snap: *mut c_void,
        // The two pinned host ranges `[lo, hi)` a remote (non-VRAM) entry lies in.
        pinned0_lo: u64,
        pinned0_hi: u64,
        pinned1_lo: u64,
        pinned1_hi: u64,
        // Tokens inside one of the `decode_ranges` ranges `[decode_lo[i],
        // decode_hi[i])` are decode-scored (the summary's decode bit). Host
        // pointers, read by the launcher into the launch parameters; at most
        // `MAX_DECODE_RANGES`.
        decode_lo: *const u32,
        decode_hi: *const u32,
        decode_ranges: i32,
        // `u32[n_experts + 1]` routing summary, or null; `summary_seq` is
        // stored in its last word after every count is visible system-wide.
        summary: *mut c_void,
        summary_seq: u32,
        // `i32[n_experts][4]` remote list, or null.
        remote: *mut c_void,
        // `i32[3]` launch work counters to zero, or null.
        counters: *mut c_void,
        // The promotion ring in mapped host memory, or null: `u64[cap]` slot
        // images, `u64[cap]` log (`summary_seq << 32 | row << 16 | expert`),
        // and the `u32` head (this kernel's) and tail (the host's) counters.
        promo_slots: *const c_void,
        promo_log: *mut c_void,
        promo_head: *mut c_void,
        promo_tail: *const c_void,
        promo_cap: u32,
        // `u32[rows][n_experts]` in-flight promotion marks (mapped), with the
        // ring.
        promo_marks: *mut c_void,
        // Mapped `u32`, with the ring: a prompt-only expert takes a slot only
        // while more than this is stocked; null = never.
        promo_reserve: *const c_void,
        // Mapped `u32`, required with the ring: a launch with more claiming
        // experts than this is a sweep and claims nothing.
        promo_sweep: *const c_void,
        // Mapped, required with the ring: `u64[cap]` the victim behind each
        // offer (`row · n_experts + expert` in the gate plane, or
        // [`PROMO_EMPTY`]), and `u64[cap][3]` the entries (gate, up, down) a
        // claimed victim is retargeted to.
        promo_victims: *const c_void,
        promo_retarget: *const c_void,
        // Mapped `u32[zone_slots]` slot tags `(row + 1) << 16 | expert`, or null:
        // every VRAM entry snapshotted is checked against its slot's tag, and a
        // mismatch traps. Slot `s` spans `[zone_end - (s + 1) · zone_slot_bytes,
        // zone_end - s · zone_slot_bytes)`.
        slot_owner: *const c_void,
        zone_end: u64,
        zone_slot_bytes: u64,
        zone_slots: u32,
        // This layer's row, recorded in the log.
        row: i32,
        // `u64[n_experts]` promotion slot per remote expert (0 = none), or null.
        remote_dst: *mut c_void,
        started_rows: *mut c_void,
        ticket: u64,
        // Read-ahead, or a null `ahead_items` (which requires the ring): the
        // mapped `u32` window (slot images the link moves in one layer) and
        // depth (targets `row + 2 ..= row + depth`), the mapped prediction
        // lists `u32[rows]` counts, `u32[rows][ahead_cap]` experts and
        // `u64[rows][ahead_cap]` vetted source images — an expert is read
        // ahead only while its entries still point at that image; the device
        // `u64[rows][4]` row layout (gate, up, down offset, image bytes), item
        // buffer `u64[1 + AHEAD_MAX · AHEAD_ITEM_WORDS]` and `u32[AHEAD_MAX]`
        // piece counters.
        ahead_window: *const c_void,
        ahead_depth: *const c_void,
        ahead_n: *const c_void,
        ahead_list: *const c_void,
        ahead_src: *const c_void,
        ahead_cap: u32,
        rows: i32,
        row_layout: *const c_void,
        ahead_items: *mut c_void,
        ahead_done: *mut c_void,
        // Non-zero: read the predictions ahead. Zero with `ahead_items` set:
        // the items carry only the CPU pool's promotions.
        ahead_predict: i32,
        // The CPU class: the split's costs (host pointer, copied into the
        // launch; null = off), this invocation's mapped work slot
        // (`CPU_WORK_SLOT_BYTES`), the device `u32[n_tokens · k]` ring row →
        // expert-grouped row map, and the device `u32[CPU_CARRY_WORDS]` carry
        // (null = nothing promoted behind the pool). With the class on,
        // `gate_row` and `cpu_dst_row` are required.
        cpu: *const CpuSplit,
        cpu_work: *mut c_void,
        cpu_dst_row: *mut c_void,
        cpu_carry: *mut c_void,
        // The CPU experts' inputs, each distinct token once: the device
        // `u32[n_tokens]` scratch, the device `u32[cap]` input row → token,
        // and the mapped `u32[1 + cap]` input-row count, then ring row →
        // input row. Null `cpu_input_map` writes none.
        cpu_tok_first: *mut c_void,
        cpu_in_tok: *mut c_void,
        cpu_input_map: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// The Rust struct is the CUDA struct byte for byte.
    #[test]
    fn cpu_split_layout() {
        assert_eq!(size_of::<CpuSplit>(), 11 * 4 + 2 * (CPU_MAX_N + 1) * 4);
        assert_eq!(offset_of!(CpuSplit, copy_ns), 8);
        assert_eq!(offset_of!(CpuSplit, min_gain_ns), 28);
        assert_eq!(offset_of!(CpuSplit, explore_ns), 32);
        assert_eq!(offset_of!(CpuSplit, promote), 36);
        assert_eq!(offset_of!(CpuSplit, max_experts), 40);
        assert_eq!(offset_of!(CpuSplit, cpu_n_ns), 44);
        assert_eq!(offset_of!(CpuSplit, gpu_n_ns), 44 + (CPU_MAX_N + 1) * 4);
    }
}
