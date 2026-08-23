# DeepSeek-V4 turn-seal persistence — overnight execution report

**Date:** 2026-08-09 (overnight, autonomous). **Branch:** `deepseek-flash`.
**Design doc:** [deepseek_turn_seal_persistence.md](deepseek_turn_seal_persistence.md)
(updated — §8 decisions resolved, §9 implementation status added).
**Nothing committed** — per standing instruction, all changes are staged in the working
tree for your review and commit.

> **Post-review refinement (same session):** on your feedback, neither the corpus nor the
> window ring stores absolute positions. Positions are **reconstructed from the layout**
> at inject time — the corpus at `base + i·ratio`, the window as the tail ending at the
> resume position — matching the fork-wide invariant that RoPE positions are always
> `base_pos + Σ preceding chunk.usage`, never persisted per token (confirmed against
> `SequenceState::rope_pos`, `projection_assembler`, and the batched decode path). Added
> `BatchedEngine::resume_sequence` as the composed resume path. Codec bumped `DSC1→DSC2`.

---

## TL;DR

The **model-side seal/resume primitives for all four artifacts are implemented and
tested green**. This is the hard, novel, correctness-critical half — the on-device
math and the byte-exact serialization that resume depends on. The **conversation-side
record + scheduler seal wiring is specified to the exact seam** (doc §9.2) rather than
half-built, because it must land atomically with a scheduler that actually drives
DeepSeek (which does not exist yet) — half-wiring it would be a stub on the persistence-GC
surface that the memory bank repeatedly flags for silent data-loss, and would add
always-on hot-path capture with no consumer. That trade is explained under
"Why the boundary is here."

**Every test below is green.** The full 284B model still answers **"Paris"** through the
wave path (regression gate), confirming the additions are purely additive.

---

## What was asked

Execute `docs/deepseek_turn_seal_persistence.md` end-to-end: at each DeepSeek turn seal,
produce four artifacts — (A) a transient raw-window-ring snapshot, (B) closed partial
HCA/CSA compressor groups, (C) the durable native corpus, (D) Indexer `sign(Q)` wide-Q
signatures — and the matching resume path, with the doc's staged green checkpoints.

---

## What is implemented and tested (green)

### Stage 1 — Artifact B: close the partial compressor groups
`candle-transformers/src/models/latent_moe/compressor.rs`, `kernel_attention.rs`

- `IncrementalCompressor::close() -> Result<Option<(Tensor, u32)>>` — finalizes the
  trailing `< ratio` buffered rows into a pre-RoPE corpus entry + group-start position.
  Overlapping (`ratio == 4`) compressor includes the previous complete group's first-half
  rows; group-0-partial masks the absent prev via `-inf`. Terminal (clears buffer,
  advances `group_idx`).
- `KernelLayerSeqState::seal_close()` — closes `comp`+`icomp` in lockstep, appends the
  closed entry+key to the gallery.
- **Tests:** `close_pools_trailing_partial_nonoverlap`, `close_pools_trailing_partial_overlap`
  — assert against a hand-computed scalar softmax reference (byte-level, per the TDD rule),
  including the empty-buffer → `None` boundary. Existing 6 compressor tests still pass.

### Stage 2 — Artifact C: corpus persist/restore
`candle-transformers/src/models/latent_moe/gallery.rs`, `wave.rs`

- `CorpusSnapshot` + `encode()`/`decode()` — self-describing little-endian codec
  (magic `DSC1` + 5×u32 geometry + the native durable set: `nope_i8`/`nope_scale`/
  `rope_bf`/`keys`/`pos`). Signs are rebuilt from keys on restore; archival f32 `attn`
  is not persisted (reference-only, unused post-resume).
- `FloatGallery::snapshot()` / `from_snapshot(device, &snap)` — native durable form, no
  re-quant.
- `BatchedEngine::{seal_sequence, corpus_snapshot, corpus_restore}` — per-layer compose.
- **Tests:** `corpus_snapshot_codec_round_trip` (CPU, asserts **exact header bytes** +
  total length + identity round-trip + rejects foreign magic / geometry mismatch /
  truncation); `gallery_snapshot_restore_round_trip` (**GPU**, asserts the restored gallery
  is **bit-exact** on `nope_i8`/`nope_scale`/`rope_bf`/`keys`/`pos`/`signs` across a
  growth boundary).

### Stage 3b — Artifact A: window-ring snapshot/restore (model side)
`candle-nn/src/kv_cache/chunked/{sequence_ops.rs,types.rs,chunk_ops.rs}`,
`candle-transformers/src/models/latent_moe/wave.rs`

- `SequenceState::{base_pos, set_base_pos}` + `ChunkedKvBacking::{window_base_pos,
  resident_len, set_window_base_pos}` — resident-window + absolute-frame accessors.
- `BatchedEngine::{window_ring_snapshot, window_ring_restore}` + `WindowRingSnapshot`/
  `WindowRingLayer` — snapshot reads the resident window via `read_contiguous`
  (FP8/BF16 ⇄ F32 is lossless for the writer formats); restore writes it back at resident
  offset 0 and seeds `base_pos`.
- **Test:** `window_ring_snapshot_restore_round_trip` (candle-nn backing level, after a
  **real front-chunk eviction**) — proves the restored window is **content-bit-exact** AND
  each restored row still equals its **ABSOLUTE** position (`base_pos + i`). This validates
  the whole ring mechanic without needing the 284B model.

### Stage 4 — Artifact D: Indexer wide-Q source (model side)
`candle-transformers/src/models/latent_moe/indexer.rs`

- `Indexer::query_band(x, qr, rope, pos) -> Result<Vec<f32>>` — the roped per-head Indexer
  query flattened `(head, dim)`, the exact input to `WideQSig::from_band`. This is the
  model's **learned** significance space (the Indexer), replacing the R16 cross-layer
  sign-fold other models use.
- **Test:** `query_band_matches_query_space_pack_order` — locks length + `(head, dim)`
  ordering against `query_space`, and demonstrates the `from_band` pack rule over it.

### Validation summary

| Suite | Result |
|-------|--------|
| `latent_moe::*` non-ignored (incl. all new codec/close/query_band tests) | **37 passed, 0 failed** |
| `gallery_snapshot_restore_round_trip` (GPU) | **pass** |
| `window_ring_snapshot_restore_round_trip` (candle-nn) | **pass** |
| `wave_paris` isolated — full 284B model, `continuation="Paris<｜EOS｜>"` | **pass (regression gate)** |

The concurrent-`wave_paris*` OOM you may see in a combined run is the documented
environmental pinned-expert-pool contention (3 × 284B loads request ~100 GB pinned host
RAM); run the DeepSeek model tests with `--test-threads 1` and an `--exact` filter. Run in
isolation each passes.

---

## Why the boundary is here (implemented vs. specified)

Reconnaissance of `candle-conversation` (two deep passes, mapped in the doc) established:

1. **The scheduler seals model-agnostically.** `gather_wide_sigs` reads only
   `self.session` (the KV backings), never `self.model`; `ManagedBatchedModel` has **no
   seal hook**. DeepSeek's corpus/compressor/gallery state lives entirely inside
   `BatchedEngine`, invisible to the scheduler.
2. **DeepSeek is not scheduler-driven yet.** Its tests call `forward_wave` directly; no
   `Scheduler` runs it. So "the scheduler seals a DeepSeek turn" is not an existing path.
3. **The conversation-side record has ~11 GC touch points** across `record.rs`,
   `substrate.rs`, `mod.rs`, `accounting.rs`, `manifest.rs`, `compaction.rs`,
   `maintenance.rs`, `resume.rs`. This is the exact surface the memory bank repeatedly
   flags for **silent data loss**.

Given the project's own rules — *no stubs, don't land what isn't correct yet, no
env-flag/optional dual paths* — the correct move was to complete and test the model-side
primitives (which have real callers: the tests and the eventual hook), and to **specify**
the conversation-side record + capture + scheduler override precisely (doc §9.2) so they
land **atomically** with the seal hook that gives them a real writer. Landing a
written-but-ignored-on-replay record, or an always-on per-token capture with no consumer,
would have been a stub and a hot-path cost with no payoff — and unverifiable tonight
without the scheduler integration and a fresh `.substrate`.

The specification in §9.2 is exact (record tag `20`, the `ConvState`-family per-timeline
LWW mechanism with the precise touch points, the `Chunk`-reuse for corpus, the
`gather_wide_sigs` DeepSeek override at `scheduler/mod.rs:6476`, the capture-ring
placement), and each piece names the existing test it mirrors for CPU verification
(`wide_q_sigs_persist_and_recover`, `wide_q_sigs_record_survives_compaction`).

---

## Files changed (all uncommitted)

**candle-transformers/src/models/latent_moe/**
- `compressor.rs` — `IncrementalCompressor::close()` + 2 tests
- `kernel_attention.rs` — `KernelLayerSeqState::{seal_close, snapshot_gallery, restore_gallery}`
- `gallery.rs` — `CorpusSnapshot` + codec + `FloatGallery::{snapshot, from_snapshot}` + 2 tests
- `indexer.rs` — `Indexer::{query_band, head_dim, n_heads}` + 1 test
- `wave.rs` — `BatchedEngine::{seal_sequence, corpus_snapshot, corpus_restore,
  window_ring_snapshot, window_ring_restore}` + `WindowRing{Snapshot,Layer}`
- `mod.rs` — exports (`CorpusSnapshot`, `WindowRingSnapshot`, `WindowRingLayer`)

**candle-nn/src/kv_cache/chunked/**
- `types.rs` — `SequenceState::{base_pos, set_base_pos}`
- `sequence_ops.rs` — `ChunkedKvBacking::{window_base_pos, resident_len, set_window_base_pos}`
- `chunk_ops.rs` — `window_ring_snapshot_restore_round_trip` test

**docs/**
- `deepseek_turn_seal_persistence.md` — §8 decisions resolved, §9 status + integration spec, §A supersede wording corrected
- `deepseek_turn_seal_overnight_report.md` — this report

---

## Suggested next session (the atomic integration)

1. Add `ManagedBatchedModel::seal_artifacts` (default empty) + `BatchedEngine` override
   composing the §9.1 primitives; add the per-token `query_band` capture ring.
2. Land `RecordType::WindowRing = 20` (§9.2 item 3) mirroring the `ConvState` family;
   verify with the two mirrored CPU tests.
3. Route corpus blobs through the per-turn `Chunk` path; DeepSeek `gather_wide_sigs`
   override.
4. End-to-end gate: decode N tokens → seal → drop session → resume → continue; assert the
   post-resume stream matches the un-interrupted run (doc Stage 3 checkpoint), on the
   284B model with `--test-threads 1`.

All four primitives are in place and green, so this is composition + the persistence-GC
wiring, not new model math.
