#pragma once

// READ-AHEAD — the contract between `moe_bucketize` (simple/moe_bucketize.cu),
// which claims promotion-ring slots for experts predicted for later rows and
// writes one ITEM per claim, and the gate launch's worker blocks
// (quantized/kernel.cuh, `moe_live_ahead_piece`), which copy each item's slot image
// and publish it. Mirrored in Rust by `candle_kernels::simple::moe_bucketize`.
//
// Items live in a device buffer `u64[1 + AHEAD_MAX * AHEAD_ITEM_WORDS]`: word 0
// is the item count, then each item's words:
//
//   [0] src        the expert's pinned warm-slot image (host address the
//                  device reads over the link)
//   [1] dst        the claimed VRAM slot image
//   [2] bytes      bytes of the image to copy, a multiple of 16
//   [3] gate_at    device address of the expert's live gate entry
//   [4] up_at      … its up entry
//   [5] down_at    … its down entry
//   [6] gate_val   the value each entry is published with: `dst` plus the
//   [7] up_val     projection's offset in the row's slot image
//   [8] down_val
//   [9] owner_at   device address of the slot's owner tag, or 0
//  [10] owner_tag  `(row + 1) << 16 | expert`
//
// Every word is computed by bucketize, so the worker needs no table geometry:
// it copies `bytes` from `src` to `dst` in AHEAD_CHUNKS pieces, and the worker
// that finishes an item's last piece publishes it — owner tag, up, down, a
// system fence, then gate (the host's publishing order, `live_table`).

// Items one bucketize may write: a read-ahead budget never exceeds this.
#define AHEAD_MAX 64
#define AHEAD_ITEM_WORDS 11
// Pieces per item: an image (~1.3 MB on Qwen3.8-Flash-Next) split this many
// ways spreads one expert over the launch's workers.
#define AHEAD_CHUNKS 16
// Set in a promotion log entry's expert field for a read-ahead claim; the
// expert is the low bits, the row the target row, not the launch's.
#define AHEAD_FLAG 0x4000u

// Bytes of piece `c` of an item of `bytes`: each piece a whole number of
// 16-byte units, the last piece whatever remains.
__host__ __device__ __forceinline__ unsigned long long ahead_piece_bytes(unsigned long long bytes) {
    const unsigned long long units = (bytes / 16ull + AHEAD_CHUNKS - 1ull) / AHEAD_CHUNKS;
    return units * 16ull;
}
