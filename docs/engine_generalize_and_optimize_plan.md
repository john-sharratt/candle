# Engine Generalization & Optimization Plan

Status: **DRAFT for review** · Owner: overnight autonomous run · Supersedes nothing (adds to
`docs/deepseek_batched_paged_attention_plan.md`, which remains the canonical attention design).

This plan takes the working DeepSeek-V4-Flash paged latent-attention engine (crash/OOM/garbage on
long prefill now fixed — see `[[wave-prefill-snapshot-dangling]]`) and drives it to a **generic,
fast, correct** state. The phases are executed **strictly in the order below** — each gate must be
green before the next phase starts.

---

## Guardrails (non-negotiable, apply to every phase)

- **Never weaken a test to make it pass.** The 8-session StoryRewrite failure (Phase 6) is the test
  doing its job — it found a real *model-quality* weakness. Fix the engine/kernels, never the
  assertion, the tolerance, the prompt, or the batch size. No cheating, no special-casing the
  fixture, no "close enough" thresholds.
- **Stay true to the CPU/reference baseline.** Every kernel change is validated bit-exact (or within
  the documented O(1)-error envelope) against the reference forward. For codec/serialization/quant
  code, assert against **raw expected bytes**, never error tolerances.
- **No `TODO`/stub/deferral.** A phase is done when *every* part of it is done, tests included.
- **Do not `git commit`.** Leave changes staged/working; the user commits. Update the progress
  memory at every gate.
- **Imports not fully-qualified paths; one concern per file; TDD alongside the code.**
- **Windows/PowerShell:** never rewrite files through PowerShell (CP1252 mojibake) — use Edit/Write.

## Working protocol

- Each phase has a **gate** (objective, measurable). Record the measurement (timing/accuracy/ncu
  metric) in the phase's "Result" line before advancing.
- Keep a running note in memory (`deepseek-implementation-progress`) at each gate.
- The full end-to-end check is `test_parallel_batched_forwarding` (StoryRewrite n=1/4/8, thinking
  stripped). The cheap per-kernel checks are the `paged.rs` mirror-oracle tests and the
  wave-vs-decode state audit. Load model once; the run is ~11 min for all three configs.

## Baselines, references & grounding (read before starting)

- **Read the canonical design doc first:** `docs/deepseek_batched_paged_attention_plan.md`. It is
  authoritative for the attention design (compressor monoid, two-stage BDP→Indexer corpus select,
  single-latent K≡V, glue, the C/E/G/K sections referenced throughout this plan). Ground every
  kernel change in it; if the doc is itself wrong, fix the doc in the same change.
- **Two distinct oracles — use the right one for the job:**
  - **GPU per-token decode reference** (`kernel_attn_decode_step` looped one token at a time, as in
    `wave_prefill_state_matches_decode_steps`). Fast enough to run full prompts and every batch size.
    This is the **full-prompt / whole-test oracle**: the batched path (wave prefill + batched decode)
    must match it bit-for-bit, or within the O(1) envelope.
  - **CPU DeepSeek reference model** (the pure reference forward). Bit-exact ground truth, but far too
    slow to run the full StoryRewrite suite (especially n=8). Use it for **component-level**
    validation and iteration: one layer, one attention step, a short prompt, a codec/quant round-trip
    — comparing a single GPU kernel's output against the CPU computation of the same input. Do **not**
    attempt the full n=8 test on it.
  - Rule of thumb: **CPU model → is this kernel/component correct?** **GPU decode reference → does the
    batched path reproduce the reference over the whole prompt/batch?**
- **HuggingFace reference implementation** of DeepSeek-V4-Flash is available online and may be
  consulted to confirm reference math (RoPE/YaRN, MLA latent projection, MoE routing, the indexer)
  when the CPU port or a kernel's intent is ambiguous. It is a reference for *correctness intent*,
  not a performance target.

---

## Phase 1 — De-naming: make model-specific modules generic

**Goal.** Nothing that is not intrinsically DeepSeek should carry the name. Names describe *what the
code operates on*, not *which model first needed it*.

**Current state (offenders).**
- `candle-kernels/src/simple/deepseek_bdp.{cu,rs}` — this is Binary Directional Provenance
  sign-packing + XNOR/popcount recall over packed signs. It is **not** DeepSeek-specific (BDP is the
  same primitive used by `candle-conversation/src/provenance/gallery_arena`).
- `candle-kernels/src/paged-deepseek/` (`deepseek_decode_kernel.cuh`, `paged_deepseek_api_bf16.cu`,
  `api.rs`) — this is **single-latent (K≡V) MLA-style paged attention** with a sliding window +
  compressed CSA/HCA selection. The *latent* geometry is the distinguishing feature, not the model.
- FFI/wrappers: `run_paged_deepseek_decode_bf16`, `run_paged_deepseek_prefill_bf16`,
  `paged_deepseek_decode(_raw)`, `paged_deepseek_glue_scatter`, `launch_deepseek_decode/prefill`.

**Steps.**
1. Rename `simple/deepseek_bdp.{cu,rs}` → `simple/bdp.{cu,rs}` (module `bdp`); functions
   `run_deepseek_sign_pack`/`run_deepseek_bdp_recall` → `run_sign_pack`/`run_bdp_recall`. Update the
   archive-group registration in `candle-kernels/build.rs` (SHA256), the FFI in `lib.rs`, and the
   two Rust call sites (`gallery.rs`).
2. Rename `paged-deepseek/` → `paged-latent/` (kernel = "paged latent attention"). Files:
   `latent_decode_kernel.cuh`, `latent_prefill_kernel.cuh` (Phase 3 splits it out),
   `paged_latent_api_bf16.cu`, `api.rs`. FFI `run_paged_latent_{decode,prefill}_bf16`; launchers
   `launch_latent_{decode,prefill}`. Update `build.rs`, `lib.rs`, `candle-core/src/cuda_backend`
   bindings, and every Rust caller (`deepseek4/paged.rs`, `kernel_attention.rs`).
3. Rust side: `deepseek4/paged.rs::paged_deepseek_*` → `paged_latent_*`. Keep the `deepseek4` model
   module name (it *is* the DeepSeek model impl) but its kernel calls point at the generic names.
4. Grep sweep: no `deepseek` string survives in `candle-kernels/` except genuinely
   model-parameter-derived constants (there should be none — HEAD_DIM=512 etc. are latent geometry,
   pass them as parameters/consts named for the geometry, not the model).

**Gate.** `cargo build --features cuda` clean; `rg -i deepseek candle-kernels/` returns nothing;
all mirror-oracle + wave tests still pass (pure rename, zero behavior change). **Result:** _(fill)_

**Risk.** Rename churn touches `build.rs` SHA groups → forces a full nvcc recompile; verify PTX
still embeds. Low logical risk (mechanical).

---

## Phase 2 — Load-time optimization (137 s → target < 15 s)

**Goal.** Model load is dominated by work that should not happen at load time.

**Current state** (`expert_lre/pipeline.rs:277` stage log): total ~137 s =
`repack(read+reorder) ≈ 78–121 s` + `place(h2d/pinned) ≈ 40–58 s`.
- **Repack** re-orders/re-quantizes expert weights into the int8-KO GEMM layout *at load*. This is
  deterministic and should be **persisted to disk once** by the offline preparer
  (`candle-core/src/quantized/prepare.rs`, which already emits the `MXFP4_KO` file) so load is a
  straight mmap + placement.
- **H2D** moves ~57 GB to VRAM + pins ~94 GB. At 64 GB/s PCIe this is ~1–2 s of actual transfer, so
  40–58 s means we are **not** saturating the bus — likely per-expert copies with per-copy syncs,
  non-pinned staging, or serialized H2D behind the repack.

**Steps.**
1. **Move repack offline.** Extend `prepare.rs` to emit the *final* int8-KO expert layout (the exact
   bytes `ExpertCache` uploads) into the prepared file, plus a header/version tag. At load,
   `ExpertCache` mmaps and uploads verbatim — zero reorder, zero re-quant. Assert byte-identity: the
   offline-repacked bytes must equal what the current load-time repack produces (raw-bytes test on a
   couple of experts). If the prepared file lacks the layout (old file), fail loudly with a
   "re-run prepare" message (no silent fallback repack — that's the slow path we're killing).
2. **Saturate H2D.** Profile the placement loop (`pipeline.rs` place phase). Fixes to apply as the
   profile dictates: (a) upload in a few large contiguous `memcpy_htod_async` chunks, not per-expert;
   (b) use pinned staging arenas for the transfer source so DMA runs at full bandwidth; (c) overlap
   H2D with the next chunk's read via a copy stream + events; (d) one final sync, not per-expert.
   Measure achieved GB/s (`bytes / place_seconds`) — target ≥ 40 GB/s effective.
3. Re-measure and report the split.

**Gate.** Cold load < 15 s on the dev box (repack ≈ 0 at load, place near PCIe roofline); prepared
file round-trips byte-identical to the old load-time repack. **Result:** _(fill)_

**Risk.** Prepared-file format change is a data-format break (fine per repo policy — no back-compat).
Must regen the on-disk model; document the new prepare invocation. Pinned budget already ~94 GB;
watch host-pinned headroom when adding transfer staging (relates to `[[wave-prefill-snapshot-dangling]]`
OOM sensitivity).

---

## Phase 3 — Dedicated prefill kernel (stop reusing the decode kernel)

**Goal.** Prefill and decode are different workloads and must have different kernels. Today the wave
prefill absorbs the prompt **one token at a time through `kernel_attn_decode_step`** (the decode
kernel) — correct but O(prompt·layers) launches and no batched-prefill parallelism. A real batched
prefill kernel `paged_latent_prefill` (ex-`paged_deepseek_prefill`, `launch_..._prefill`) **already
exists** but is only exercised by `paged.rs` tests, not the wave — it was shelved on the
bf16-diagonal-source issue (canonical plan §G.5).

**Reference design to copy.** `candle-kernels/src/paged-prefill/paged_prefill_int8_kernel.cuh`
(+ `kv_store.cuh`, `pal_rank.cuh`, `api.rs`): the production multi-query-over-settled-slot prefill
with cu_seqlens ragged batching, tiled K/V loads, and the pal_map-aware loaders. That is the shape
the latent prefill should take.

**Steps.**
1. Study the paged-prefill kernel (tiling, ragged cu_seqlens, split-KV, sink fold, the settled-slot
   read path) and the latent prefill's existing skeleton. Identify the diagonal-source discrepancy:
   decode stages the new token's latent as bf16; the batched prefill must feed each query its own
   just-written diagonal latent at the *right* per-query position, matching decode bit-for-bit.
2. Implement `latent_prefill_kernel.cuh`: many queries over a **settled** slot (all latents written +
   committed before launch), per-query positions + per-query CSA/HCA selection, bf16 diagonal source
   sized to match `kernel_attn_decode_step`'s per-token result exactly. Sink fold, RoPE, de-rotation
   identical to decode.
3. Wire it into `wave.rs`: replace the per-token `kernel_attn_decode_step` loop for prefill rows with
   a single `paged_latent_prefill` launch per layer over the ragged prefill batch (the decode/glue
   rows keep their paths). Keep the host-side write-len patch / snapshot metadata invariant intact
   (Phase-fixed in `[[wave-prefill-snapshot-dangling]]`); a settled-slot prefill means the arena is
   fully written first, so revisit whether the per-token snapshot is even needed for prefill (a
   settled slot has one stable header → likely one snapshot, not N).
4. Keep the per-token path available behind the reference/audit tests as the oracle.

**Gate.** `wave_prefill_state_matches_decode_steps` extended to a **>64-token** prompt (crosses ≥2
chunk boundaries) shows the batched-prefill arena state + next-step logits bit-match the per-token
decode reference. Prefill wall-clock drops materially (measure t/s). StoryRewrite n=1/4 stay 100%.
**Result:** _(fill)_

**Risk.** This is the highest-value correctness-sensitive change. The bf16-diagonal parity is the
known trap — gate on bit-exact state audit, not just output coherence.

---

## Phase 4 — Full pal_map + quant-format support in decode & prefill

**Goal.** The latent decode and prefill kernels must handle **pal_map** (palette-mapped per-band
formats) and **multiple quant formats** for K/V, not just fixed FP8. This unblocks Phase 7 (Q8
experiments) and aligns the latent kernels with the production paged-decode/paged-prefill loaders.

**Current state.** The latent kernels read one fixed FP8 (E4M3) band layout with an outer scale.
`paged-decode/slot_types.cuh` + `paged-prefill/kv_store.cuh`/`pal_rank.cuh` already implement the
general pal_map + per-band format/scale machinery (`k_pal`/`v_pal`, `k_fmt`/`v_fmt`, `k_scale`/
`v_scale`), which the latent path currently ignores (single-latent chunks store `meta = None`, fixed
scale).

**Steps.**
1. Route the latent kernels' band reads through the shared `slot_types.cuh`/`kv_store.cuh` accessors
   (`kvhead_k_pal_map`, `kvhead_k_ptr`, `kvhead_k_scale`, `k_fmt`) instead of the hard-coded FP8
   path, so per-band format/scale/palette are honored.
2. Support the format set the paged loaders support (at minimum FP8-E4M3, Q8_0, Q8_1; the dequant
   helpers already exist in the loaders — reuse, don't reimplement). Single-latent K≡V means one
   read path feeds both K and V; keep the alias.
3. Extend the compression/format selection so single-latent chunks can carry a real per-band
   format/scale record (drop the `meta = None` shortcut where it blocks pal_map).
4. Bit-exact tests per format against the CPU dequant baseline (raw bytes), mirroring the existing
   `mxfp4-ko` bit-exact fixtures.

**Gate.** Latent decode + prefill produce bit-exact output for FP8 / Q8_0 / Q8_1 K/V against the CPU
baseline; pal_map with mixed per-band formats reads correctly (extend the offset-window kernel tests
— see `[[offset-window-kv-read-is-correct]]`). **Result:** _(fill)_

**Risk.** Mixed-band strides are the classic OOB source (`[[select-family-band0-extrapolation]]`);
stage per-band {ptr,fmt,scale,outer} and never extrapolate band-0 format across bands. Trust the
sanitizer, not WDDM timing.

---

## Phase 5 — Attention-kernel profiling & optimization

**Goal.** The profile shows the attention kernels are a top cost. Drive them to a strong fraction of
achievable throughput while staying bit-true to the baseline.

**Tooling.** Nsight Compute (`ncu`) for per-kernel metrics (occupancy, warp stall reasons, memory
throughput, bank conflicts, achieved vs peak DRAM/L2), Nsight Systems (`nsys`) for the launch
timeline + gaps (the `[[wddm-forward-floor]]` launch floor is real — CUDA graphs / batching launches
is on the table). The existing `--features profile` pipeline marks give the coarse split; ncu gives
the why.

**Steps.**
1. `nsys` the wave to quantify launch overhead vs kernel time (confirm/deny the WDDM floor as the
   dominant term at low occupancy).
2. `ncu` the latent decode + (Phase-3) prefill kernels. Rank bottlenecks: smem bank conflicts on the
   `sQ/sK/kv_f` tiles, uncoalesced band reads, low occupancy from smem pressure (8 KB sQ + 8 KB kv_f
   + …), split-KV factor mis-sizing, the argsort/gather in two-stage selection.
3. Fix top-N in priority order; after each fix re-run the **bit-exact** state audit + StoryRewrite
   n=1 (never trade correctness for speed). Candidate levers: vectorized/coalesced band loads,
   conflict-free smem layout, better split-KV heuristic, fusing the combine, CUDA graphs for the
   per-token launches if the floor dominates.

**Gate.** Each optimization keeps the state audit bit-exact; measured decode + prefill kernel time
drops with a recorded ncu before/after per lever. **Result:** _(fill)_

**Risk.** Easy to introduce a numeric drift. Every change gates on the bit-exact audit, not eyeballed
output.

---

## Phase 6 — Fix the 8-session concurrent attention failure (do NOT touch the test)

**Goal.** At n=8, one or two sessions fail StoryRewrite by **not rewriting the protagonist's name**
("Marcus" — the *original* name in the source story — survives instead of the session's assigned
name), while the rest of the story reproduces verbatim. Keeping the original token where a new one
was instructed is a **classic attention failure**: the batched path, at some batch position, fails to
attend to the rewrite instruction / the new name. n=1 and n=4 are 100%, so it is batch-count- or
batch-position-dependent.

**Step 0 — quick quant probe (fast diagnostic, run first).** Before the full bisect, cheaply test
whether K/V dynamic range is implicated: with Phase-4's multi-format support in hand, re-run the n=8
StoryRewrite with K/V as **Q8_0** and then **Q8_1** (both carry a per-block scale — Q8_1 also a min —
and cover the latent's dynamic range better than FP8-E4M3). This is a *diagnostic*, not the fix:
- If a higher-dynamic-range format makes the failing session(s) pass (or shifts which position
  fails), we have a strong lead that the weakness is **quantization/precision** in the K/V path →
  pursue it here and carry the finding into Phase 7's fuller evaluation.
- If it does **not** move the failure, quantization is likely not the cause → set it aside, proceed to
  the bisect below, and defer the full quant A/B to Phase 7.
Keep FP8 as the control; record the n=8 pass rate per format. (This is the one place quant work jumps
the queue — deliberately, because it's a 10-minute discriminator, not a rebuild of Phase 7.)

**Hypotheses to discriminate (in order).**
1. **Batched attention divergence** — glue/compressed-selection/window at high batch differs from the
   reference for a specific slot. Most likely, given it's batch-position-specific and the story body
   is otherwise perfect.
2. **Per-token snapshot metadata at high batch** — the wave builds N snapshots × 8 seqs; a
   slot-indexing or write-len edge at position 6/7 (now guarded, but verify at n=8).
3. **Selection/gallery at n=8** — two-stage CSA selection picking a slightly different top-k for one
   slot (the records cache is per-sequence, so cross-contamination is excluded — see
   `[[wave-prefill-snapshot-dangling]]`).
4. **Genuine model-quality softmax degradation** exposed by batch-order float differences (user's
   leading suspicion) — a near-decision the batched reduction order tips.

**Steps.**
1. Reproduce deterministically: fixed seed/prompt, dump per-session generated token ids at n=8,
   confirm which position(s) fail and whether it's deterministic across runs (rules 4 in/out).
2. **Bisect with the right oracle** (see "Baselines & references"). Extend the state audit to n=8
   with the StoryRewrite prompt and diff the failing session against the **GPU per-token decode
   reference** for that exact prompt — every layer's arena window + compressed selection + logits at
   the token where the name is emitted; the first divergent artifact localizes the bug to a
   component. Then confirm that component against the **CPU reference model** at the component level
   (single layer / single step / short input — the CPU model can't run the whole n=8 prompt) to
   decide whether the GPU kernel or the reference math is wrong. Consult the HuggingFace reference if
   the intended math is ambiguous.
3. Fix the identified component (kernel or metadata), re-validate bit-exact vs baseline, then re-run
   the full n=8 StoryRewrite. **The fix lives in the engine; the test is untouched.**
4. If (and only if) the audit proves the batched path is bit-identical to the reference and the
   reference *also* keeps "Marcus," then it is genuine model quality → carry into Phase 7 (quant) and
   the decode-quality work; document the evidence. Still do not weaken the test.

**Gate.** n=8 StoryRewrite 8/8 with the batched path proven bit-faithful (or within the O(1) envelope)
to the reference at the failing token. **Result:** _(fill)_

**Risk.** Tempting to blame float noise and move on — resist. Prove it with the audit before calling
it model quality.

---

## Phase 7 — K/V quant & compression (7a formats · 7b levels)

### 7a — Format evaluation (Q8_0 / Q8_1 vs FP8)

**Goal.** Once the kernels support multiple formats (Phase 4) and attention is correct (Phase 6),
evaluate whether a higher-dynamic-range 8-bit format improves decode quality — the fuller version of
the Phase-6 Step-0 probe. FP8-E4M3 (current) has coarse mantissa; **Q8_0/Q8_1** carry a per-block
scale (Q8_1 also a min) and better represent the latent's dynamic range.

**Steps.**
1. Add Q8_0 and Q8_1 as selectable K/V (single-latent) formats via the Phase-4 machinery.
2. A/B the StoryRewrite suite + a quality probe (NIAH / longer context) across FP8 vs Q8_0 vs Q8_1;
   record accuracy and t/s (Q8 is heavier per read).
3. Keep the winner as the default only if it does not regress the bit-exact envelope elsewhere.

**Gate (7a).** Measured accuracy/throughput table across the three formats; a defensible default.
**Expectation (user):** likely *not* the fix for the n=8 failure (Phase 6 owns that; the Phase-6
Step-0 probe already screened it) — this is a quality/robustness lever, run it anyway. **Result:** _(fill)_

### 7b — Adaptive compression levels (C0–C9)

**Goal.** The adaptive per-block KV compression (C0 near-lossless → C9 max, the `CompressionPolicy`
in `candle-nn/.../chunked/compression_policy.rs`) is the engine's headline capability. With the
latent kernels now format-aware (Phase 4), enable it on the single-latent path and characterize how
far it can be pushed before quality breaks.

**Steps.**
1. Enable compression-mode selection for single-latent chunks (per-32-token block format selection
   via the cosine-distance thresholds) — reusing the existing policy, not a new one. The K/V
   asymmetry (`PRODUCTION_K_*` / `PRODUCTION_V_*`) must be re-derived by measurement for
   DeepSeek-V4-Flash (they are model-specific — see the CLAUDE.md note).
2. Sweep compression levels against the StoryRewrite suite + a longer-context quality probe: for each
   level C0..C9, record whether the tests hold and the memory/throughput won. Find the highest level
   that keeps n=1/4/8 passing.
3. Verify the attention-sink protection (first 4 tokens' dedicated fine scale) holds under
   compression on the latent path.

**Gate (7b).** A compression-level vs quality/footprint table; the highest level that keeps the
StoryRewrite suite green identified and set as the aggressive default; sink protection verified.
**Result:** _(fill)_

**Risk.** Q8 read cost may hurt the Phase-8 throughput target; treat as a quality/speed trade to
measure, not assume. Thresholds are model-specific — do not reuse Qwen3's `PRODUCTION_*` constants.

---

## Phase 8 — Decode throughput to target (40–50 t/s single, 1000s t/s prefill)

**Goal.** With fast load (Phase 2), a real prefill kernel (Phase 3), correct multi-format attention
(Phases 4/6), and profiled kernels (Phase 5), iterate on decode throughput.

**Steps.**
1. Establish the current single-session decode t/s and prefill t/s baseline post-Phases 1–7.
2. Profiling-driven iteration cycles (nsys/ncu): attack the largest remaining term each cycle — WDDM
   launch floor (CUDA graphs / TCC — see `[[wddm-forward-floor]]`), expert dispatch/thrash
   (`[[gpu-native-moe-dispatch]]`), attention kernel occupancy, the prefill batch shape.
3. Each cycle: measure → fix largest term → re-validate correctness (StoryRewrite + audit) → measure.
   Stop when single-session ≥ 40 t/s and prefill ≥ ~1000 t/s (or diminishing returns, documented).

**Gate.** Single-session decode ≥ 40–50 t/s and prefill ≥ ~1000 t/s on the dev box, StoryRewrite
still passing at n=1/4/8. **Result:** _(fill)_

**Risk.** Throughput work is where correctness silently rots — every cycle re-runs the correctness
gate before accepting a speedup.

---

## Dependency order (why this sequence)

1 (naming) is prerequisite hygiene so all later kernel work lands in correctly-named files. 2 (load)
is independent but makes every subsequent iteration cycle ~2 min faster (huge multiplier over dozens
of runs). 3 (prefill kernel) must precede 5/8 (can't optimize a kernel that's still the wrong shape).
4 (formats) unblocks both the Phase-6 Step-0 quant probe and Phase 7. 6 (n=8 correctness) precedes
7/8 because we optimize/quantize only a *correct* engine — but its **Step 0 borrows Phase-4's formats
for a fast Q8 discriminator** before the bisect, since it's a 10-minute test that could hand us the
cause. 7 splits into 7a (formats, the full version of that probe) and 7b (adaptive C0–C9 compression
levels). 8 is last — throughput tuning on a correct, fast-loading, generic engine.

**One caveat on strict ordering:** the Phase-6 Step-0 quant probe is the single intentional
exception — a cheap diagnostic that reaches forward to Phase-4 format support. It does not reorder
the work (Phase 4 is already done by the time Phase 6 runs); it just front-loads a 10-minute test
whose result steers Phase 6. Everything else runs strictly 1→8.
