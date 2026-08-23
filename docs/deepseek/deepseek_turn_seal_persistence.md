# DeepSeek-V4 turn seal & resume: window-ring snapshot + closed corpus + Indexer wide-Q

> **Status:** the **model-side seal/resume primitives are implemented and tested**
> (Artifacts A/B/C/D, see §9); the **conversation-side record + scheduler seal
> wiring is specified** (§9.2) and lands atomically with the scheduler path that
> drives DeepSeek. Builds directly on the sliding-window KV ring (`base_pos` /
> `evict_window_front`, committed `67309fc0`). Authoritative for the DeepSeek
> seal/resume path; where it disagrees with code, the code is wrong.

## 1. Motivation

The sliding-window ring bounds the *live* window to `O(window_size)`. What it does
**not** yet do is persist a turn so a resumed conversation reconstructs (a) the raw
ring buffer and (b) the turn's compressed KV, and so provenance can *select* the turn.
DeepSeek currently has **no substrate seal wiring at all** — `BatchedEngine` is only a
`ManagedBatchedModel` (`wave.rs:132`), driven by tests; nothing calls `record_turn` /
emits `WideQSig` / persists the gallery. This document specifies that seal, and the
matching resume.

The design rests on one structural fact established earlier: **`window_size ≥
compress_ratio`** (128 ≥ 128 HCA, 128 ≥ 4 CSA). Every token that leaves the window has
already been folded into the compressed corpus, so the corpus is the durable
conversation KV and the raw window is only ever a *transient tail* needed to warm-start
the ring.

## 2. The four seal artifacts

At each turn seal, DeepSeek emits **four** artifacts. Three are durable conversation
state; one is a transient tail snapshot.

| # | Artifact | Durability | Substrate form | Consumed on resume by |
|---|----------|-----------|----------------|------------------------|
| A | **Raw window ring** (last `window_size` FP8 latents, per layer) | **Transient** — tombstoned when superseded | New `RecordType::WindowRing` (standalone, per-timeline) | Ring-buffer warm-start (chronological tail) |
| B | **Closed** partial HCA/CSA groups | folded into C | (becomes corpus entries) | (see C) |
| C | **Corpus entries** (all complete + the just-closed partials), native form | Durable | `RecordType::Chunk` per turn stream | Provenance turn injection → gallery |
| D | **Indexer sign-128 wide-Q** (possibly several per turn) | Durable | `RecordType::WideQSig` (existing) | Provenance selection scan (unchanged) |

The corpus (C) is the conversation. The window (A) is a warm-start convenience. The
wide-Q (D) is how the corpus becomes *findable*.

---

### Artifact A — raw window ring snapshot (transient, tombstoned)

**What.** At seal, snapshot the ring's resident window: per layer, the FP8 latents of
the last `≤ window_size` tokens, plus `base_pos` and the resident positions. This is the
exact live-ring state produced by `evict_window_front` — the arena's resident chunks.

**Why standalone + transient.** The ring only ever needs the **latest** `window_size`
tokens of the whole conversation (the sliding window is chronological, independent of
provenance). So turn `N+1`'s snapshot fully supersedes turn `N`'s. Keeping every seal's
window would accumulate dead FP8 forever, so each new `WindowRing` **tombstones the
prior one** for that timeline. Because it is decoupled from the turn's corpus/tokens
records (which are permanent), it must be its **own record type**, not a field on
`TurnDecl`.

**New `RecordType::WindowRing = 20`** (next free after `TurnCoupling = 19`).
Payload (versioned):
- `timeline_id`, seal `turn_index` (keying + provenance/debug)
- per layer: the dense resident-window KV bytes (K≡V single latent → one band set).

**No absolute position is stored.** RoPE attention depends only on the relative offset
`q_pos − key_pos`, so the absolute frame is a free gauge — the only requirement is that
the window shares ONE frame with the corpus and the continuing query. The raw window is
always the **contiguous tail** of the turn (sinks are separate per-head logits, not window
KVs), so its base is exactly `resume_position − resident_len`, and both are known at
resume: `resume_position` is the turn's seal point (recovered from turn metadata) and
`resident_len` is the KV blob length. Storing `base_pos` / per-chunk `rope_base` would be
a redundant, drift-prone second source of truth for a value that must equal the live
decode position. (Contrast the corpus, Artifact C: its entries are *scattered*
group-starts that provenance *reorders*, so each carries its own `pos` — not derivable
like the window tail.) The restore primitive `window_ring_restore` still takes the base as
a parameter (frame-agnostic write); the resume path supplies `resume_position −
resident_len`.

**Emit + supersede.** On seal, write the `WindowRing` keyed to the timeline. It is
**per-timeline last-writer-wins metadata** (the `ConvState` / `TreeMetadata` family):
the newest record for a timeline supersedes the prior one *positionally* — the
compaction/maintenance liveness pass re-emits only the current (latest) copy from live
in-RAM state, so the superseded copy is absent from the resident set and reads as dead.
There is **no per-record `Tombstone`**: `RecordType::Tombstone` (record 14) is
timeline/turn-scoped death, not a metadata-supersession marker; the supersede-and-keep-
latest is the positional LWW mechanism `ConvState` uses (`persistence/mod.rs`
`is_tracked_metadata` / `metadata_locs`; `maintenance.rs` `gather_resident_set`;
`compaction.rs` `collect_live_records`). Distillation drops the ring like the KV chunks
(it is transient window state, not the durable belief signature `WideQSig`).

**Resume.** Load the timeline's latest `WindowRing` → rebuild the ring arena directly
(allocate the chunks, stamp `offset`/`usage`/`base_pos`, upload the FP8 bytes) so the
first post-resume decode continues the sliding window bit-exactly. Never
provenance-selected; loaded once at conversation open.

---

### Artifact B — close the partial HCA/CSA at seal

**What.** When the turn seals, **finalize** (close) the two open compressor groups
(`comp` = attention, `icomp` = indexer) instead of carrying an open accumulator across
the boundary. Closing pools the group's *buffered* `< ratio` rows with the same softmax
`emit_group_raw` uses for a full group (for the overlapping `ratio == 4` compressor,
including the retained prev-half), yielding a normal corpus entry of a partial group.

**Why it's correct.** A partial-group entry is a softmax-weighted latent like any other;
the attention kernel merges it in the combined softmax **exactly** as it merges a
complete entry (fewer pooled rows, same math) — no kernel change, no special case. This
is what the user means by "a partially sealed turn gets softmax and merged in the
attention kernel anyway."

**Supersedes the monoid-persistence idea.** We do **not** persist an open `(m,l,acc)`
tail across the seam. `GroupPartial` (`compressor.rs`) stays **reference-only algebra**
(its seam-fold proof remains a correctness test, not a runtime path). The next turn
starts a fresh group; `group_idx` continues so corpus positions stay monotone.

**API.** Add `IncrementalCompressor::close(&mut self) -> Result<Option<(Tensor, u32)>>`
(pre-RoPE entry + group-start position, `None` if the buffer is empty), mirroring
`emit_group_raw` but not requiring a full `ratio` of rows. Called at seal for both
`comp` and `icomp`; the returned entries append to the gallery (Artifact C) before the
snapshot.

---

### Artifact C — persist the corpus (complete + closed), native form

**What.** After closing (B), the gallery holds every corpus entry for the turn:
- attention two-region: `nope_i8 [g,448]` + `nope_scale [g,14]` + `rope_bf [g,64]`
- indexer scoring keys: `[g,128]`

Persist them in **native form** (no re-quant at seal — they already are the QAT storage
precision) as the turn's KV, tied to the turn stream, alongside the `TurnDecl`. Reuse
`RecordType::Chunk` (record 4) — the same records dialogue KV already persists — with a
DeepSeek corpus-entry layout.

**No positions are stored.** A group-start position is `g · ratio` (turn-relative,
`g` = entry order), so it is *reconstructed* at inject time — exactly as the chunked KV
cache derives every RoPE position from cumulative layout (`SequenceState::rope_pos =
base_pos + Σ preceding usage`) rather than persisting an absolute per token. Persisting
the original absolute `pos` would be a stale second source of truth that provenance
re-layout (which reorders turns and thus reassigns their base offsets) invalidates.

**Resume.** Provenance selects turns (via D); each selected turn's corpus entries are
**injected into the gallery** at their **reconstructed** positions — `corpus_base + i ·
ratio`, where `corpus_base` is the position the turn's tokens land at in the reconstructed
context (the running slot offset at injection, per `projection_assembler`'s layout-order
convention) and `ratio` is the layer's compression ratio. `gather_corpus` then re-heats
the bounded selection per query as it already does. This is the corpus analogue of the
existing per-turn KV-chunk injection — the turn-injection plumbing (which likewise
re-derives positions from layout, never stores them) is reused; only the payload shape
(corpus entry vs window chunk) differs.

**Note on the seam.** The *closed partial* of turn `N` and the leading tokens of turn
`N+1` belong to different groups now (we closed at the boundary), so no cross-turn group
straddles the seam — injection is turn-local and order-independent, which is what makes
provenance reordering safe.

---

### Artifact D — Indexer sign-128 wide-Q signatures

**What.** At seal, extract the **Indexer's** per-head roped query `sign(Q)` for the
turn and store it in the existing `WideQSig` records. `Indexer::query_space`
(`indexer.rs:60`) already yields the roped per-head `q [n_heads, head_dim=128]` — sign
its 128 dims per head → pack via `WideQSig::from_band(band, head_dim=128)`
(`provenance/wide_sig.rs`). "It might be multiple extracted from it": a turn yields
several rows (per captured token / per indexer head); store all via `encode_wide_sigs`
and `enqueue_wide_q_sigs(turn_stream_id, blob)` — the same off-thread writer other
models use.

**Why this is the *right* source.** DeepSeek's Indexer **is** the learned significance /
recall space ("BDP but learned/float/≤1M"). For every other model the wide-Q is a
*heuristic* fold of R16 `sign(Q)` across all layers (`gather_wide_sigs`,
`scheduler/mod.rs:6476`, `fold_provenance`). DeepSeek instead reads the model's **real**
significance directly from the Indexer — no layer-fold heuristic. The output format is
identical (`WideQSig`, 128-bit heads), so it **plugs straight into** the unchanged
selection pipeline: `score_belief_*` / `gather_wide_sigs`-consumers read
`wide_q_sigs_blob` exactly as today.

**Integration.** Give `BatchedEngine` a seal-time `indexer_wide_sigs(seq, range) ->
Vec<WideQSig>` and route the scheduler's wide-Q capture through the model when it is
DeepSeek (override the R16 `gather_wide_sigs` path), rather than reading R16 (which
DeepSeek's latent attention does not even hold in the same shape).

---

## 3. Seal lifecycle (end to end)

On turn seal, in order:

1. **Close** the open `comp` + `icomp` groups → append the partial entries to the
   gallery (Artifact B → C).
2. **Persist corpus**: serialize the turn's gallery entries (native two-region + keys +
   pos) as `Chunk` records tied to the turn stream; write the `TurnDecl` (Artifact C).
3. **Extract wide-Q**: Indexer `sign(Q)` for the turn → `WideQSig` record on the turn
   stream (Artifact D).
4. **Snapshot the ring**: write `WindowRing` for the timeline; tombstone the previous
   one (Artifact A).

Steps 2–4 are independent given step 1; step 1 must precede all (it changes the gallery
and clears the compressor buffers so the next turn starts a fresh group).

## 4. Resume lifecycle

On conversation open:

1. **Warm-start the ring**: load the timeline's latest `WindowRing` → rebuild the FP8
   ring arena (chunks + `base_pos` + absolute positions). Decode continues the sliding
   window bit-exactly.
2. **Provenance selection** runs over the persisted `WideQSig` (Artifact D) exactly as
   for other models → selects turns.
3. **Inject corpus**: each selected turn's persisted corpus entries (Artifact C) load
   into the gallery at their absolute `pos`. Live decode then attends window (from 1) +
   selected corpus (from 3), identical to a never-persisted session.

## 5. Integration points

**candle-transformers / deepseek4**
- `compressor.rs`: `IncrementalCompressor::close()` (+ byte-exact test).
- `kernel_attention.rs` / `gallery.rs`: seal snapshot of the gallery's corpus entries
  (serialize) + inject (deserialize into `append_batch`/`gather_corpus` shape).
- `indexer.rs`: `sign(Q)` extraction → `Vec<WideQSig>`.
- `wave.rs` (`BatchedEngine`): seal hook exposing (window bytes, corpus entries,
  indexer wide-Q) to the scheduler.
- Ring snapshot/restore reads/writes the `ChunkedKvBacking` window arena (the
  `base_pos`/resident chunks from the committed ring work).

**candle-conversation**
- `persistence/record.rs`: `RecordType::WindowRing = 20` + payload codec.
- Seal path (`scheduler/mod.rs::perform_seal_and_write` / `projection/resolver.rs::
  record_turn`): accept + persist DeepSeek's window/corpus/wide-Q; DeepSeek override of
  `gather_wide_sigs` (Indexer source).
- `persistence/maintenance.rs` / `compaction.rs`: `WindowRing` = supersede-and-tombstone
  per-timeline (mirror `ConvState`); keep only the latest live.
- Resume (`persistence/resume.rs`): load latest `WindowRing` → ring rebuild; corpus
  inject on turn selection.

**Reused unchanged**: `WideQSig` + `encode_wide_sigs`/`decode_wide_sigs` +
`enqueue_wide_q_sigs`; `RecordType::Chunk`, `RecordType::Tombstone` + `TombstonePayload`;
`FloatGallery::append_batch` / `gather_corpus`; the whole provenance selection scan.

## 6. Stages (each a green checkpoint)

1. **Close partials.** `IncrementalCompressor::close`; test the partial pool equals the
   softmax over the buffered rows (overlap prev-half included). Gallery gains the closed
   entry; `wave_paris` / StoryRewrite still green.
2. **Corpus persist ↔ resume.** Serialize gallery entries at seal, inject on resume;
   byte-exact round-trip test (`seal → Chunk → load → inject == live gallery`).
3. **WindowRing event.** New record type + payload + emit/tombstone/restore; a
   resume-continues-decode test (decode N, seal, drop session, resume from `WindowRing`,
   continue — output matches the un-interrupted run).
4. **Indexer wide-Q.** `sign(Q)` extraction + `WideQSig` record + DeepSeek
   `gather_wide_sigs` override; test that provenance selects a DeepSeek turn from its
   Indexer signature (extend the existing selection fixtures).

## 7. Non-goals / invariants

- **No open `(m,l,acc)` persistence.** Closing at seal supersedes it; `GroupPartial`
  stays reference-only.
- **`window_size ≥ compress_ratio`** (asserted at model construction) — guarantees a
  token leaving the window is already in the corpus, so closing/eviction never drops an
  uncompressed token.
- **Native corpus form** — no re-quant on seal (entries are already QAT-precision).
- **Wide-Q format identical** to other models — DeepSeek only changes the *source*
  (Indexer, not R16 fold), never the record or the selection.
- **Positions stay absolute** through seal/resume (windows and corpus share the absolute
  frame — the property the ring already relies on).

## 8. Decisions (resolved)

- **WindowRing granularity**: **one record for all layers** (atomic supersede; the whole
  ring is one warm-start unit). `WindowRingSnapshot { layers: Vec<WindowRingLayer> }`.
- **Wide-Q sampling**: **per-token** Indexer `sign(Q)` (matches other models). The
  extraction primitive is `Indexer::query_band`; the capture ring and any later
  subsampling are §9.2 integration concerns.
- **Corpus record keying**: **reuse the per-turn `Chunk` stream** so injection and
  tombstoning inherit the dialogue machinery. The native corpus blob is
  `CorpusSnapshot::encode()` (one per compression layer), carried as the turn's KV.

---

## 9. Implementation status

### 9.1 Model side — IMPLEMENTED + TESTED (candle-transformers / candle-nn)

All four artifacts' model-side primitives are landed and green. Each is self-contained
(no dependency on `candle-conversation`, honoring the crate direction), so the scheduler
integration composes them.

**Artifact B — close partials** (`latent_moe/compressor.rs`, `kernel_attention.rs`)
- `IncrementalCompressor::close(&mut self) -> Result<Option<(Tensor, u32)>>` — pools the
  trailing `< ratio` buffered rows (overlap: previous group's first-half included) into a
  pre-RoPE corpus entry + group-start position. Terminal (clears the buffer, advances
  `group_idx`). Tests `close_pools_trailing_partial_{nonoverlap,overlap}` assert against a
  hand-computed scalar softmax reference (byte-level, not tolerance).
- `KernelLayerSeqState::seal_close(&mut self)` — closes `comp`+`icomp` in lockstep and
  appends the closed entry+key to the gallery.

**Artifact C — corpus persist/restore** (`latent_moe/gallery.rs`, `wave.rs`)
- `CorpusSnapshot { index_head_dim, len, nope_i8, nope_scale, rope_bf, keys }` with
  `encode()`/`decode()` — self-describing little-endian codec (magic `DSC2` + 5×u32
  geometry). **No positions stored** (reconstructed at inject — see below). Test
  `corpus_snapshot_codec_round_trip` asserts the exact header bytes + total length +
  identity round-trip + rejects foreign magic/geometry/truncation.
- `FloatGallery::snapshot() -> CorpusSnapshot` / `from_snapshot(device, &snap, positions)`
  — native durable form (two-region + keys); `positions` are the RECONSTRUCTED
  group-starts the caller supplies (`base + i·ratio`), not from the snapshot; signs
  repacked from keys; archival `attn` is a CPU zero placeholder (reference
  `gather_selected` only, unused post-resume). GPU test
  `gallery_snapshot_restore_round_trip` restores at a fresh base (10 000) and asserts
  bit-exact `nope_i8`/`nope_scale`/`rope_bf`/`keys`/`signs` + the reconstructed `pos`.
- `BatchedEngine::{seal_sequence, corpus_snapshot, corpus_restore}` — per-layer seal
  (B), snapshot (`Vec<Option<CorpusSnapshot>>`, `None` on SWA), and fresh-state restore
  that reconstructs each layer's positions as `corpus_base + i · compress_ratio(layer)`.

**Artifact A — window ring snapshot/restore** (`kv_cache/chunked/sequence_ops.rs`,
`chunk_ops.rs`; `latent_moe/wave.rs`)
- `ChunkedKvBacking::{window_base_pos, resident_len, set_window_base_pos}` +
  `SequenceState::{base_pos, set_base_pos}` — the resident-window + absolute-frame
  accessors. Snapshot reads via `read_contiguous(seq, 0, resident_len)` (FP8/BF16 ⇄ F32 is
  lossless for the writer formats); restore writes via `ensure_for_offset` +
  `write_contiguous` + `set_len` + `set_window_base_pos`. Backing test
  `window_ring_snapshot_restore_round_trip` proves content bit-exact AND each restored row
  still equals its absolute position after a real front-chunk eviction.
- `WindowRingSnapshot`/`WindowRingLayer { resident_len, kv }` — **no stored position**.
  `BatchedEngine::window_ring_snapshot` captures the dense resident tail;
  `window_ring_restore(session, seq, snap, decode_pos)` places it as the contiguous tail
  ending at `decode_pos`, so `base_pos = decode_pos − resident_len` is RECONSTRUCTED from
  the resume frame.

**The resume path** (`latent_moe/wave.rs`)
- `BatchedEngine::resume_sequence(session, seq, corpus_snaps, window, base, total_tokens)`
  — composes the artifacts at a (possibly new) absolute frame: corpus at `base + i·ratio`,
  the raw window as the tail ending at `decode_pos = base + total_tokens`
  (`base_pos = decode_pos − resident_len`), and `session.set_sequence_offset(seq,
  decode_pos)` so decode continues there. Positions everywhere follow the reconstruction
  layout, never a stored value — the same convention the chunked cache and
  `projection_assembler` already enforce.

**Artifact D — Indexer wide-Q source** (`latent_moe/indexer.rs`)
- `Indexer::query_band(x, qr, rope, pos) -> Result<Vec<f32>>` — the roped per-head query
  flattened `(head, dim)`, the exact input to `WideQSig::from_band`. Test
  `query_band_matches_query_space_pack_order` locks length + ordering + the `from_band`
  pack rule. This is the model's LEARNED significance space (the Indexer), replacing the
  R16 cross-layer sign-fold other models use.

### 9.2 Conversation side — integration spec (lands atomically with the scheduler seal path)

DeepSeek is **not yet driven by the `Scheduler`** (tests exercise `forward_wave`
directly), and the scheduler's seal path is model-agnostic — `gather_wide_sigs` reads only
`self.session` (the KV backings), never `self.model`, and `ManagedBatchedModel` has no
seal hook. So the conversation-side wiring below must land together with (a) a
`ManagedBatchedModel` seal hook and (b) the scheduler actually running DeepSeek; landing
it earlier would be a record nothing writes / an always-on hot-path capture with no
consumer. The seams are exact (mapped against the current tree):

1. **`ManagedBatchedModel` seal hook** (`candle-transformers/.../batched_inference.rs`):
   add `fn seal_artifacts(&self, session, seq) -> Result<SealArtifacts>` (default: empty),
   overridden by `BatchedEngine` to call `seal_sequence` → `corpus_snapshot` →
   `window_ring_snapshot` + drain the per-token `query_band` capture. `SealArtifacts`
   carries the encoded corpus blobs, the encoded ring blob, and the wide-Q bands.

2. **Per-token wide-Q capture** (`BatchedEngine`): a per-`(seq)` ring appended in
   `forward_wave` for CSA layers (each token's `Indexer::query_band`, concatenated across
   CSA layers), drained at seal. Always-on (no env flag), so it lands only with the
   scheduler consumer that pays for it.

3. **`RecordType::WindowRing = 20`** (`persistence/record.rs`): next free tag. Add the
   enum variant, the `from_tag` arm, a binary `WindowRingRecordPayload {timeline_id,
   turn_index, ring: Vec<u8>}` (per-layer dense window-KV blob; the model's
   `WindowRingSnapshot` bytes). **No absolute position in the record** — the resume path
   derives the base as `resume_position − resident_len` (see Artifact A) and passes it to
   `window_ring_restore`; `resident_len` is implicit in the blob. Per-**timeline** LWW
   metadata (the `ConvState` family): `apply_walker_entry` arm stores it on the timeline
   (latest wins);
   `is_tracked_metadata` / `accounting.rs` / `manifest.rs` skip-list classify it;
   `compaction.rs` `collect_live_records` + `maintenance.rs` `gather_resident_set`
   re-emit only the live copy; **dropped on distill** (transient, unlike `WideQSig`).
   `write_window_ring` helper + a `WriteJob::WindowRing` on the off-thread writer.
   Resume reads the timeline's latest ring → `window_ring_restore`. Verifiable on CPU by
   mirroring `wide_q_sigs_persist_and_recover` (emit → reopen → readback) and
   `wide_q_sigs_record_survives_compaction`.

4. **Corpus persist as `Chunk`** (`persistence/resume.rs` `persist_turn_chunks*`): the
   turn's `CorpusSnapshot::encode()` blobs ride the existing per-turn `Chunk` records
   (one per compression layer, `chunk_index = flat_chunk_index(layer, 0, ...)`), injected
   on resume via `corpus_restore`.

5. **DeepSeek `gather_wide_sigs` override** (`scheduler/mod.rs:6476`): when the model is
   DeepSeek, source the turn's wide-Q from the captured Indexer bands
   (`WideQSig::from_band(band, 128)` per token) instead of the R16 fold — same
   `WideQSig` record + `enqueue_wide_q_sigs` + selection scan, only the *source* changes.
