# QSA index keys in Q8_0

**Status:** design, agreed 2026-11-15. Built in the order of §7.

The QSA index cache holds, per sequence and per full-attention layer, one pooled,
RMS-normed, un-rotated 128-wide key per block of `ratio` tokens (4 on the
released Flash-Next). Every key is stored `F32` today — on the card in the
span's QSA-index arenas, and on disk in a turn's carried state and in its
positional index pages. This document moves every completed key to **Q8_0**.

## 1. Why

On WhiteWorld (RTX 3090, 2026-11-15, 8 characters) the QSA index cache held
**3.87 GB** of the 22.8 GB span — more than the 1.4 GB of live K/V it selects
from, and more than the expert zone's 5.1 GB beside it. Experts hit 10–20%, and
each decode step waited on ~10,000 expert misses a wave: about one second a step.
At `F32` the index costs 12 layers × 128 × 4 B ÷ 4 tokens = **1.5 KB per token
per live sequence**; the characters' contexts run 50–150k tokens.

The reference stores the compressed index key in BF16; DeepSeek's indexer, the
same design, stores 8-bit keys. The key only feeds `Σ_h ReLU(⟨q_h, k̄⟩)` and a
top-512 pick, which tolerates 8-bit keys.

## 2. The format

- **Q8_0, blocks of 32 elements along the key**, a 128-wide key being four
  blocks. Each block is `int8[32]` and one `f16` scale `= max|x| / 127`.
  Symmetric: the keys are RMS-normed with `k_norm`'s per-channel weight, not
  mean-centred, and a symmetric scale represents an off-centre block exactly,
  losing at most one bit of range. Q8_1's block sum serves dot products against
  an offset-quantized operand; the query is never one, so it is not used.
- **Layout on the card: two planes per page**, not interleaved: `int8 [256, d]`
  then `f16 [256, d / 32]`. The scorer reads the int8 plane in whole rows (the
  IMMA k32 tile is exactly one block), and the scales once per tile.
- **Size:** `d + 2·d/32` = 136 B a key at `d = 128`, against 512 B — **3.76×**.
  WhiteWorld's 3.87 GB becomes about 1.03 GB.
- **The open block stays `F32`**: at most `ratio − 1` raw rows and its rewind
  copies, pooled into a key only when the block completes.

## 3. Where a key is quantized

**In the append kernel, as it is made** (`qsa_index_append`). A key is complete
the moment its block closes: pooled, normed, 128 values, needing nothing beyond
itself. The kernel quantizes in registers and stores int8 and scales; no `F32`
key is ever written (hot-path invariant 1: kernels emit the final type).

It is **not** done in the KV gap (`docs/kv_quantize_in_the_gap.md`): the gap is
for formats chosen by per-block statistics and a policy, off the forward thread.
An index key needs neither, and routing it there would add a store, a read and
a stage for nothing.

## 4. The scorer

`qsa_score_paged` reads Q8_0 pages: per row, four int8 blocks, each dequantized
by its scale and rotated at the row's position, as the `F32` row is today. First
built with `dp4a`; then (§7 step 5) an int8 tensor-core path — the query rows
of one sequence on a decode step are 4 heads × its MTP positions, about 16, the
m16 of `mma.sync m16n8k32` (sm_80+) and the IMMA paths on Blackwell, with the
query quantized per row and `ReLU` and the head sum on the dequantized score.

`qsa_page_place` copies Q8_0 pages as they are; nothing on the placement path
converts.

## 5. On disk

Two persisted forms hold completed keys, both opaque blobs the model owns:

- **The turn's carried state** (`paged_index::encode_aux`). `AUX_VERSION`
  becomes **4**: each layer's completed rows as Q8_0 (the two planes), its open
  block as `F32`.
- **The unit's positional index pages** (`seal_positional_tail_span`), which a
  slot places when it borrows a turn: the same Q8_0 page encoding, with a
  version word.

**A build reads Q8_0 only.** A version-3 aux blob, or an `F32` page, is refused
by name — "written before the index was Q8_0: run `substrate_inspect
migrate-index-q8`" — never read as before. There is no second reader.

## 6. Migration

A one-time program, `substrate_inspect migrate-index-q8 --log <dir>`, run on a
stopped substrate:

1. `--dry-run` counts every record holding an `F32` index (carried state and
   positional pages) and their bytes.
2. The run decodes each, quantizes its completed rows to Q8_0 with the same
   quantizer the append kernel uses (bit-identical: a Rust reference of it is
   the kernel's test oracle), and appends the re-encoded record in its place,
   as compaction re-emits one; the superseded records are dead weight a later
   compaction reclaims.
3. It ends with a count of `F32` records left, which must be zero.

Run on WhiteWorld's npcd and BlackWorld's zend after the build reaches main,
each with its daemon stopped, then the new build started.

## 7. Order of work

1. **Q8_0 on the card**: the quantizer (Rust reference + device), the append
   kernel storing Q8_0, the scorer and placement reading it (`dp4a`).
2. **The gate**: on recorded contexts, `F32` and Q8_0 select the same top-512
   blocks (≥99% overlap) and the attention outputs agree within the KV gate's
   tolerance; tested by `#[test]`s against a Rust reference scorer.
3. **Q8_0 on disk**: aux version 4, page version word, refusal of the old ones.
4. **The migration program**, with its dry run, against a copy of a real store.
5. **The int8 tensor-core scorer**, validated against the `dp4a` one.
