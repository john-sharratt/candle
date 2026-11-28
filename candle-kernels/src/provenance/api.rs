//! FFI binding for the batched Binary Directional Provenance (BDP) flat scan.
//!
//! One launch scores a wave of probes against a resident gallery, emitting the
//! per-(query-token, layer-group, SEGMENT) `(leading_case, z*margin)` pairs. The
//! needle gate + per-case tally run host-side (identical to the CPU reference).
//! A segment is a code-read file / timeline: the gallery tokens are sorted by
//! segment and each segment owns a contiguous token AND case range, so z/margin
//! are per-segment (the belief scan's per-file normalization) in ONE launch. The
//! non-segmented global-z scan is `n_segments == 1`.

use core::ffi::c_void;

/// One probe token's statistics in one segment, as the scan's statistics mode
/// writes them — the C `BdpStats` of `bdp_vote.cuh`, field for field: the
/// leading and runner-up cases' best agreements and their GLOBAL case ids
/// (`-1` when fewer cases scored), and the agreement sum and sum of squares
/// over the segment's scanned tokens.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BdpStats {
    pub a1: u32,
    pub c1: i32,
    pub a2: u32,
    pub c2: i32,
    pub sum: u64,
    pub sumsq: u64,
}

extern "C" {
    /// Launch the segmented batched BDP scan on `stream`.
    ///
    /// - `gallery_words`: group-major `[group][token][gw]` sign words, tokens
    ///   sorted by segment, device ptr. Total tokens = `seg_tok_start[n_segments]`.
    /// - `gallery_case`: GLOBAL case ids, device ptr — one per token
    ///   (contiguous), one per page entry (paged).
    /// - `probe_words`: `n_probe_tokens * wpt` sign words (token-major), device ptr.
    /// - `seg_tok_start` / `seg_case_start`: `n_segments+1` prefix boundaries —
    ///   segment `s` owns tokens (paged: slots) `[seg_tok_start[s],
    ///   seg_tok_start[s+1])` and cases `[seg_case_start[s], seg_case_start[s+1])`,
    ///   device ptrs.
    /// - `seg_tok_count`: `n_segments+1` prefix of the tokens each segment
    ///   actually scans — its vote's mean and spread are over them. Contiguous:
    ///   `seg_tok_start` itself.
    /// - `page_ptr` / `entry_page` / `entry_slots`: the PAGED gallery (resident
    ///   arena). `page_ptr` is `n_pages * n_groups` absolute device addresses,
    ///   `page_ptr[page * n_groups + g]` group `g`'s strip in that page. Slot `j`
    ///   of the scan is slot `j & 31` of page `entry_page[j >> 5]`, scanned when
    ///   bit `j & 31` of `entry_slots[j >> 5]` is set. All null ⇒ the CONTIGUOUS
    ///   layout in `gallery_words` (with `gallery_words` null in the paged case).
    /// - `seg_top_k`: `n_segments` (device) — how many cases a probe token votes
    ///   for in each segment, `1..=BDP_MAX_TOP_K`; null ⇒ 1 for every segment.
    ///   At 1 a token votes `z × margin` for its leading case; at `k` for each
    ///   of its best `k`, case `r` by `z_r × (a_r − a_k)`.
    /// - `n_groups` / `gw` / `wpt`: folded-signature geometry (`wpt = n_groups*gw`).
    /// - `max_seg_cases`: the largest segment's case count (shared-mem stride).
    /// - `max_k`: votes per output row, at least every segment's `seg_top_k`.
    /// - `out_case` / `out_vote`: `n_probe_tokens * n_groups * n_segments *
    ///   max_k` outputs (device) — per (query token, group, segment) row, its
    ///   voted GLOBAL cases best first (`-1` past the last) and their votes.
    /// - `out_stats`: null for votes; otherwise the **statistics mode** — one
    ///   [`BdpStats`] per (query token, group, segment) row (device), in the
    ///   rows' order, and `out_case` / `out_vote` are not written.
    /// - `stream`: the CUDA stream handle so the kernel orders with the copies.
    pub fn run_batched_bdp_scan(
        gallery_words: *const u64,
        gallery_case: *const u32,
        probe_words: *const u64,
        seg_tok_start: *const i32,
        seg_tok_count: *const i32,
        seg_case_start: *const i32,
        page_ptr: *const u64,
        entry_page: *const u32,
        entry_slots: *const u32,
        seg_top_k: *const i32,
        n_probe_tokens: i32,
        n_groups: i32,
        n_segments: i32,
        max_seg_cases: i32,
        gw: i32,
        wpt: i32,
        max_k: i32,
        out_case: *mut i32,
        out_vote: *mut f32,
        out_stats: *mut BdpStats,
        stream: *mut c_void,
    );

    /// Whether the BMMA backend can run on the current device: b1 tensor-core
    /// hardware (sm_75..sm_89) AND a loadable kernel image in the embedded
    /// fatbin (probed via `cudaFuncGetAttributes` — SASS is minor-version
    /// specific, so capability alone over-promises). The caller falls down its
    /// backend ladder when this reports 0.
    pub fn bdp_bmma_supported() -> i32;

    /// Launch the tensor-core (b1 BMMA) BDP scan on `stream` — the PAGED gallery
    /// only (`page_ptr`, `entry_page` and `entry_slots` required, `gw == 8`),
    /// over `n_slots` slots (`entries × 32`). Emits the same `(out_case,
    /// out_vote)` layout as [`run_batched_bdp_scan`]; the integer per-case
    /// statistics are identical to the scalar kernel's, and the float finalize
    /// is the shared `bdp_vote`, its `n` each segment's `seg_tok_count`.
    /// Requires each segment's gallery cases to be non-decreasing over the scan
    /// order (the index builder sorts its windows by case). Returns 0 on
    /// success, 1 when this device/geometry cannot run the BMMA path, or a
    /// negative cudaError code.
    pub fn run_bmma_bdp_scan(
        gallery_case: *const u32,
        probe_words: *const u64,
        seg_tok_count: *const i32,
        seg_case_start: *const i32,
        page_ptr: *const u64,
        entry_page: *const u32,
        entry_slots: *const u32,
        seg_top_k: *const i32,
        n_slots: i32,
        n_probe_tokens: i32,
        n_groups: i32,
        n_segments: i32,
        n_cases: i32,
        gw: i32,
        wpt: i32,
        max_k: i32,
        out_case: *mut i32,
        out_vote: *mut f32,
        out_stats: *mut BdpStats,
        stream: *mut c_void,
    ) -> i32;

    /// Whether the IMMA backend can run on the current device: INT8 MMA
    /// hardware (sm_80+) AND a loadable kernel image in the embedded fatbin
    /// (the build ships sm_89 and sm_120 SASS + compute_120 PTX — probed via
    /// `cudaFuncGetAttributes`, so the answer tracks the build's arch set).
    pub fn bdp_imma_supported() -> i32;

    /// Launch the INT8 tensor-core (IMMA) BDP scan — the Blackwell-portable
    /// twin of [`run_bmma_bdp_scan`]: identical inputs, contract, and integer
    /// statistics (0/1-encoded operands accumulate `m11 = popc(q AND t)`, and
    /// `agreement = 512 - popc(q) - popc(t) + 2*m11` exactly), with the shared
    /// finalize emitting bit-matching votes.
    pub fn run_imma_bdp_scan(
        gallery_case: *const u32,
        probe_words: *const u64,
        seg_tok_count: *const i32,
        seg_case_start: *const i32,
        page_ptr: *const u64,
        entry_page: *const u32,
        entry_slots: *const u32,
        seg_top_k: *const i32,
        n_slots: i32,
        n_probe_tokens: i32,
        n_groups: i32,
        n_segments: i32,
        n_cases: i32,
        gw: i32,
        wpt: i32,
        max_k: i32,
        out_case: *mut i32,
        out_vote: *mut f32,
        out_stats: *mut BdpStats,
        stream: *mut c_void,
    ) -> i32;
}
