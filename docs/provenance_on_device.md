# Provenance on the device

The provenance pass a projection boundary runs — the turn's live probe, the
belief scan over the gallery, the tally, the normalization — moves onto the
device and runs between waves on the device's one stream, with nothing on the
host waiting for it. Its result is also the hit signal for every model: the
hit-driven projection (`docs/hit_driven_projection.md`) reads what attention
is locked onto from the scan's scores, and Flash-Next adds its QSA hits as the
one input only it has.

The work is in four phases, ordered by what each buys. Each phase replaces the
mechanism before it; none leaves the old path beside the new one.

## 1. What runs at a boundary now

`Scheduler::reproject_view_prepare` (`scheduler/reproject_prepare.rs`), on the
scheduler thread, between waves:

| Step | Where | Device ↔ host |
|---|---|---|
| **Probe** — the turn's folded `sign(Q)`: its head chunks, its last chunks, the question window (`turn_probe.rs`) | `gather_wide_sigs` (`provenance_capture.rs:238`): the sign-pack kernel reads the R16 Q band across the stream layers (`batched_inference/provenance.rs:190`); the host XOR-folds the packed bits per token (`assemble_folded_prov_sigs`) | upload of the Q pointers; **a blocking copy back of the packed bits** |
| **Scan** — every belief collection, every layer's belief groups, the knowledge base's files (`Conversation::score_beliefs`, `projection/resolver/belief_scoring.rs:159`) | the paged BDP kernel over the resident gallery arena (`gallery_arena/scan/launch.rs`), one launch per scope | upload of the probe; `out_case`/`out_vote` copied back, then **`stream.synchronize()` on the device's stream** (`launch.rs:551`) — the wave in flight drains |
| **Tally** — each segment's needle gate (its top quarter of query tokens by vote magnitude) and the weighed vote laws summed into cases | host, `needle_tally_segments` (`provenance/gpu.rs:412`), per token, in parallel | — |
| **Normalize** — each score as tail surprise against its scope's background, the question-pin probe max-fused | host, `NormalizationCache::normalize` (`normalization/cache.rs:57`) | — |
| **Decide** — the belief step, `project()`, working-set and files-set momentum, the fill, the metadata-only rebuild | host | — |

The first four rows are per-token work, two copies back and a drain of the
stream at every boundary of every sequence — the one GPU→CPU path on the
forward's critical path that hot-path invariant 3 does not sanction. The last
row is control over a few thousand per-case numbers, and stays on the host.

A model with no QSA index reads none of this as hits: `take_hits` answers
`None`, and its conversations' trained window is never set
(`scheduler/construct.rs:343`, gated on `carries_positional_state()`), so a
Qwen3.6 dialogue is filled by its layers' windows, not as a slot.

## 2. Where the pass runs, and CUDA graphs

A wave records its sweep as a chain of graphs (`docs/decode_graphs.md`): the
wave driver opens the capture, the model records, the capture's `finish`
launches the last segment (`wave_driver/forward_core.rs:207–231`), and the
scheduler's read of the wave's tokens is the wave's one host wait.

**The provenance pass is issued eagerly, after that wait, before the next
wave.** On the one stream it runs ahead of the next forward; the next wave's
host wait therefore covers it, and the host reads its result then with no wait
of its own and no publish protocol — stream order is the guarantee. The pass
is never inside a recorded segment, so:

- no wave changes topology by carrying it or not, and no new shape reaches the
  executable library (`decode_graphs.md` §2.1.1);
- its grids need no ladder to fold in place, though they are sized from bounds
  anyway, so its launch cost does not track the probe's length;
- it follows `decode_graphs.md` §2.4 as eager work: launched on
  `dev.cuda_stream()` under the scheduler thread's own lease, its uploads
  through the staging ring (`upload_raw`) — never `with_synced_upload`, which
  synchronizes — and its scratch on held ground or `grow_scratch`;
- a launch outside a recording costs about 10 µs of host time (§0.1 there); the
  pass is a handful of launches, so it costs tens of microseconds a boundary,
  against the stall it removes.

Its result is one wave late: the decisions a boundary takes after wave N use
the probe of the turn as it stood at wave N − 1. A decode wave is one token.

## 3. Phase 1 — every model hit-driven

*Benefit: Qwen3.6, Qwen3.5, the GQA models and DeepSeek-V4 get the hit-driven
slot fill, round momentum and knowledge-base momentum Flash-Next has. No
change to throughput.*

- **The density is the scan's own.** At a boundary, each declared round's
  density is its share of the scan's fresh scores over the declared rounds:

      d(r) = s(r) / Σ_{r'} s(r')

  where `s(r)` is the highest fresh score of the round's turns (a section's
  score for a knowledge-base round). Pooled rounds are scored by the same scan,
  so loaded and pooled rounds are read alike. The density folds into the round
  momentum unchanged — `h ← (1 − β)·h + d`, a lift of `1 + gain·h`
  (`projection/hits.rs`).
- **Flash-Next adds its QSA hits.** Its `take_hits` read joins the scan's
  density as one more part, as an assembly's read does today
  (`hit_density::hit_density`); every other model's read is empty.
- **The trained window is every model's.** `construct.rs` sets each
  conversation's trained window from `rope_select().ceilings()` for every
  model; the `carries_positional_state()` gate goes.
- **The prose goes**: "a model whose attention reads every position takes none
  of it" in `managed_model.rs`, `hit_layers.rs`, the model builder,
  `hit_driven_projection.md` §1 and `npc_mind_layout.md` §3.

Tests: the density over a fixed score map, raw values; a round with no scored
turn takes none; pooled rounds read on the loaded rounds' scale; a Qwen3.6
conversation's trained window is its RoPE ceiling.

## 4. Phase 2 — no wait at the boundary

*Benefit: the boundary's two copies back and the drain of the stream go. A
decode step that lands on a boundary no longer stops the device for the scan;
the scheduler thread no longer blocks on it.*

- **The probe is folded on the device.** The sign-pack kernel folds each
  token's bits under the model's fold (`FoldParams`: layer groups, heads
  separate) as it packs them, and writes the folded probe — the head chunks,
  the last chunks, and the question window — straight into the scan's probe
  buffer, which stays on the device. The host uploads only the Q band pointers,
  a descriptor table (invariant 2b). `assemble_folded_prov_sigs` and the CPU
  gather go.
- **The scan reads that buffer.** Every scope of the boundary — collections,
  belief groups, files — is one launch over the device's probe, with the
  indices the gallery arena already keeps resident (`DeviceIndex`). An index
  rebuilt because the arena changed uploads through the staging ring, not
  `with_synced_upload`.
- **Its output lands in pinned, mapped memory.** `out_case` and `out_vote` are
  written into a per-sequence block of pinned host memory mapped into the
  device, so nothing is copied back.
- **The host reads it after the next wave's wait.** The scheduler issues the
  pass after wave N's tokens are read, marks the boundary as pending on that
  sequence, and after wave N + 1's tokens are read runs the tally, the
  normalization and the decision over the block, as today.

Tests: the device fold against the host fold, bit for bit, for every fold
geometry the models declare; the scan's mapped output against the copied-back
output of the current path on the same probe and arena (kept only as the test's
reference); a boundary that issues the pass and reads it a wave later decides
exactly what the synchronous path decided on the same probe.

Measured: `reproject_ms` and its `probe_ms`/`readback_us` split before and
after, and the decode step time of a wave that lands on a boundary against one
that does not, per machine, in `docs/performance.md`.

## 5. Phase 3 — the tally and the normalization on the device

*Benefit: the per-token host work of a boundary goes — the needle gate and
tally over every probe token, and the normalization of every case — and the
mapped block shrinks from every probe token's votes to one score per case.*

- **The tally kernel.** Per segment: each query token's magnitude (the weighed
  votes over the fold groups), the segment's `NEEDLE_KEEP_FRAC` quantile by
  selection, and each kept token's votes, raised by the segment's vote law and
  weighed by its group weights, summed into its cases. The accumulation order
  is the host's, so the result is the host tally's bit for bit.
- **The backgrounds on the device.** Each scope's background
  (`normalization/scope.rs`: counts in log-spaced bins, the pooled background,
  the scoring-sample counts) is mirrored to a device table. A scope changes only
  when a seal or a warm-up teaches it, so its table is uploaded then, through the
  staging ring, beside the observe that changed it.
- **The normalize kernel.** Each case's tail surprise against its background
  — the pooled background below `min_background`, the probe's own scores for a
  scope never taught, zero for no evidence, the slope past the top bin — and the
  question-pin probe's max-fuse.
- **The density on the device.** Phase 1's `d(r)` from the normalized case
  scores through a case→round map declared with the rounds.
- **The block holds the result.** Normalized and raw scores per case and
  density per declared round; the raw scores are what a seal teaches from.

Tests: the tally and the normalization kernels against `needle_tally_segments`
and `NormalizationCache::normalize`, which remain as the tests' oracles only —
raw expected values over a hand-built competition: a hub, a candidate with no
background, a scope never taught, a score past every bin.

## 6. Phase 4 — the seal and the first reading on the same pass

*Benefit: the last provenance copies back go, and with them the CPU fallbacks;
invariant 3 holds for provenance.*

- **The seal's signatures.** A sealed turn's per-token signatures — what the
  gallery and the substrate record keep — are folded on the device by the same
  kernel and written into the mapped block; the persistence thread reads them
  once the next wave's wait has covered the pass, and the seal's `observe`
  learns from the block's raw scores.
- **The first reading.** A request's first reading (`first_reading.rs`), which
  orders the knowledge base by the question before the reply decodes, is issued
  after the opening's last prefill wave and read after the reply's first decode
  wave: the reply's first token decodes over the order the opening had, and the
  reading's order lands at its second.
- **What goes**: the CPU gather (`gather_r16_kv_provenance_layers` and the
  host sign pass), the contiguous CPU scan path where a GPU arena exists, and
  `with_synced_upload` on the provenance path.

Tests: a sealed turn's stored signatures from the device path against the
current path, bit for bit; a first reading read a wave late orders the range as
the synchronous reading did.

## 7. Open choices

- **The first reading's wave.** Phase 4 decodes the reply's first token over
  the opening's order. Running the pass ahead of that wave instead would hold
  the first token for the scan's device time, with still no host wait.
- **The per-step knowledge-base order.** Flash-Next reorders the knowledge
  base's range every decode step from its QSA hits (`section_step.rs`). The
  scan is several times the cost of the hit count it would replace there, so
  every other model reorders at boundaries.
