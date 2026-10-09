# Decode CUDA graphs — the wave chain

Scope: the **hybrid wave** — the one forward that carries a wave's decode rows, its
speculative verify windows and its prefill rows together — run as **multiple graphs chained
together with no host sync**. A wave is never split: its decode and prefill rows run through
the same graphs.

The mechanism is model-agnostic and lives in `candle-core` (`cuda_backend/graph/`). A model
takes part by marking where its recorded launches begin (`Device::record_launches`); until it
does, its forward runs as it always has. Status per model is in §3.

The design is measured, not assumed. §0 holds the measurements that chose it; §6 records
what the earlier, capture-once design specified and why each part of it was dropped.

---

## 0. Measured facts (RTX 3090 24 GB, WDDM, 2026-10-05)

The performance bar is `docs/results/baseline_rtx_3090_24gb_2026-10-05_decode_graphs.md`:
the Qwen3.6-35B-A3B and Qwen3.8-Flash-Next gates at 256 generated tokens (C9/C10 at 64),
every row validated, no graph code.

### 0.1 What a launch costs, and what a graph costs instead

Qwen3.6 gate, whole run under Nsight Systems CUDA tracing: 1,637,166 `cudaLaunchKernel`
calls, median 6.0 µs host time each, about 1,250 launches per forward — 7–12 ms of
submission against a 19.4 ms single-session decode step. Flash-Next: about 2,000 launches
and ~14 ms of submission per forward against steps of 70–170 ms, of which the streamed
expert GEMM is the bulk.

`graph::tests::per_wave_capture_cost` (ignored benchmark) issues 1,250 small launches per
round, 20 rounds:

| Path | Host cost per launch |
|---|---:|
| eager launch on the null stream, to completion | 10.83 µs |
| record on the capture stream, fold into the executable with `cuGraphExecUpdate`, launch, to completion | 2.36 µs |
| of which recording alone | 0.59 µs |
| replay of an up-to-date executable | 1.27 µs |

The executable was updated in place on 20 of 21 rounds. **Recording every wave and updating
its executable is about 4.5× cheaper than launching eagerly**, so a graph never has to be
reused unchanged to pay. That one number is why this design records every wave instead of
capturing once and keying replays (§6).

### 0.2 What the capturing thread may not do

`graph::tests::null_stream_calls_during_capture` (ignored probe) issues each call on the
legacy null stream from the thread that is recording, `ThreadLocal` mode:

| Call | Result | Capture afterwards |
|---|---|---|
| `synchronize` | refused (`STREAM_CAPTURE_UNSUPPORTED`) | invalidated |
| pageable host → device copy | refused | invalidated |
| pool allocation | refused | invalidated |
| pool free (`cuMemFreeAsync`) | the drop swallows the error | refused at end |
| device → host readback | **succeeds** | intact |
| kernel launch on the null stream | **succeeds, runs at once** | intact |

The first four fail loudly; the last two are silent and wrong — they run immediately,
ahead of recorded work that should precede them. §2.3 is how each is kept out.

---

## 1. Prior art: Strata

Strata (`strata`, v0.1.26) captures a verify window or a decode token as one graph per
row count, keyed by slot combination, with raw runtime calls (`cudaStreamBeginCapture`
`ThreadLocal`, instantiate, explicit upload, launch), every per-run variation in device
tables, and no `cudaGraphExecUpdate`. Its batched graphs bake slot pointers, so it needs
one graph per slot combination. This engine reads slot addressing from device tables
(hot-path invariant 2b) and, per §0.1, can afford to refresh every parameter every wave, so
it needs neither the key nor the combination.

---

## 2. Design

### 2.1 The pieces

| Piece | Where | Does |
|---|---|---|
| `CaptureHub` | `cuda_backend/graph/hub.rs` | One per device, shared by every clone of the handle. Holds the capture stream (created, non-blocking), the open wave, the executables of each segment ordinal kept across waves, the graveyard, the two staging rings and the two hand-off events between the capture and compute streams (§2.3). |
| `ExecSlot` | `cuda_backend/graph/slot.rs` | One segment ordinal's executables — up to four, one per wave shape, most recently used first. A recapture is folded into the first executable of its node count that accepts it, and instantiated only when none does. |
| `WaveCapture` | `cuda_backend/graph/hub.rs` | RAII for one forward. `CudaDevice::begin_wave_capture` opens it **held**; `finish` launches the last segment; dropping it unfinished discards what is recording (the forward has already failed). Not `Send`, like the `Paused` guard: the driver ends a thread-local capture only on the thread that began it. |
| `Device::record_launches` | `eager.rs` | The model's mark: recording starts here. |
| `CudaDevice::cuda_stream` | `device.rs` | The **launch stream**: the capture stream on the thread that is recording, the compute stream otherwise. Every launcher takes it. |
| `CudaDevice::compute_stream` | same | The legacy null stream. Anything that outlives the call — a slice's home stream, an event, a readback, a registry key — uses this. |
| `Device::eager` / `CudaDevice::pause_capture` | `eager.rs`, `device.rs` | Suspend recording until the guard drops; see §2.2. |
| `CudaDevice::flush_launches` | `device.rs` | End the segment, launch it, free what it retired, `cuStreamQuery` the compute stream, resume — with no cross-stream hand-off: a flush runs no eager section, so the event pairs a pause records order nothing there. |
| `CudaDevice::recording_segment` | `device.rs` | The `(wave, segment ordinal)` this thread is recording into, or `None` when its launches run eagerly — what the MoE dispatch's flush schedule keys on. |
| `CudaDevice::retire` | `device.rs`, `hub.rs` | Free device memory after the segment that may still read it has been launched — whichever thread releases it. |
| `CudaDevice::upload_raw` | `device.rs`, `staging.rs` | A host upload that, while recording, is staged and recorded instead of ending the segment. |
| `GraphExec::{instantiate_audited, try_fold}` | `exec.rs` | Instantiate and upload an audited recapture; or fold one in place with `cuGraphExecUpdate_v2`, reporting when the driver refuses it as a different topology. |
| audit | `exec.rs` | Every node must be a kernel, a memset, a device-to-device copy or a profiling span's external event record; anything else is refused with the graph dumped as DOT (`cuGraphDebugDotPrint`) and named in the error. |
| `grow_scratch` | `candle-kernels/src/grow_scratch.cuh` | Grow-only launcher scratch that is safe to grow while recording. |

### 2.2 One forward

```
drive_wave
  begin_wave_capture()            held: nothing is recorded
  model.sweep(..)
    admission, tier placement,    eager, exactly as before
    tables, embedding
    dev.record_launches()         recording starts
    for each layer:
      launches ............................ recorded into segment k
      eager section (host protocol) ....... segment k ends and is launched;
                                            the section runs in issue order behind it;
                                            recording resumes as segment k+1
      MoE: before() / record() / after()    after() = flush_launches once the
                                            segment holds its share of MoE
                                            invocations (1, 1, 2, 4, 8, 16, then
                                            32): the segment holding bucketize is
                                            launched, the compute stream queried
                                            for WDDM
    head ................................ recorded
  finish()                        last segment launched
  logits readback                 the wave's one host wait, after the chain
```

A **segment** is everything recorded between two host interactions. Segment `i` of a wave
folds into the executable ordinal `i` keeps for that node count: the parameters a wave
changes — tier addresses, row counts, the dispatch sequence number, grid sizes — are
rewritten in place, and a segment whose topology differs is instantiated afresh. An ordinal
keeps one executable per wave shape (a prompt prefill, a verify step and a draft step are
different graphs at the same ordinal): with one per ordinal, Flash-Next's first decode wave
after every prefill re-instantiated ~50 segments. The node count is a cheap filter, not a
correctness key: two shapes can share a count, so each executable of that count is offered
the recapture, most recently used first, and the driver accepts only the one of the same
topology; when none does, it is instantiated beside them. Nothing is replayed stale, and
there is no epoch to check.

**Order is the invariant.** Recording changes when launches reach the driver — a segment at a
time instead of one at a time — never their order: every eager call runs after everything
issued before it, because the segment before it is launched first, and before everything
issued after it, because recording resumes only when it returns.

**Thread scoping without thread-local state.** The hub records the owning `ThreadId` in its
mutex, and beside it in an atomic a tag folded from that id; `cuda_stream()` checks an atomic
flag (one load when no wave is open), then the tag — a different tag is a different thread,
answered without the lock — and confirms an equal one against the owner under the lock.
Another thread asking for the launch stream during a wave — the persistence thread, the
expert pipeline, the stager — gets the compute stream and runs exactly as before. The hub is
`Send + Sync`; there is no `thread_local!`.

### 2.3 What ends a segment, and the silent cases

Every device-level host interaction pauses recording on its own, so a model does not have to
know which of its calls are host interactions:

- `CudaDevice::{alloc, alloc_zeros, memcpy_htod, memcpy_dtod, memcpy_dtov, memcpy_stod,
  with_staging, with_synced_upload, synchronize}`, an info-ring or permutation-table miss,
  `to_cpu_storage` / `to_cpu_scalar` (the silent readback of §0.2), the pinned stager's
  owned copies, fences and syncs, the table ring's half switch, the bucketize workspace's
  regrowth, KV format selection.
- Device calls a recorded upload cannot stand in for, made by the KV slot-state buffers
  (`GpuChunks`): a pinned-staging copy when the wave's ring is full or the thread is not
  recording, the fence before a slot changes hands, a slot-state slab claim. Each runs
  inside `pause_capture` itself. The slot-state upload and the paged slot headers built on
  it are otherwise **recorded** — the bytes into the ring, the copy into the segment — so a
  verify's attention layer and a draft-walk step build their headers without a cut.
- The streamed layer store (`layer_stream::LayerCache`), when a layer must be loaded or
  joined: its copies are ordered behind a compute-stream event and its joins wait on the
  compute stream, so the recorded GEMMs still reading an evicted slot are launched first. A
  resident layer, or a plan with nothing to load, touches nothing.

What does **not** end a segment:

- **Frees.** A `CudaStorage` dropped while recording, or a staged `GpuBuf`, goes to the
  graveyard and is freed right after its segment is launched, in stream order behind it —
  dropped on any thread, since the last reference to memory a recorded launch reads may be
  released elsewhere. The graveyard is emptied outside the hub's lock.
- **Uploads to existing or freshly carved device memory** (`upload_raw`, `upload_into_slice`,
  `wave_from_vec_ticketed`, `upload_into`, `record_upload`, the selection and QSA index
  tables, the KV slot-state buffers and the paged slot headers). The bytes are copied into
  the wave's pinned, device-mapped ring and a `copy_bytes` from the ring is recorded. Its
  grid comes from a power-of-two ladder rather than from the size, so a table that grows a
  little from one wave to the next — a slot-state buffer gaining a chunk — keeps its
  segment's shape and folds in place. A slot-state buffer gaining a chunk uploads only its
  tail: its records sit past room for its slot class's capacity, so an append moves no
  entry already written and the sync serialises from the previous writer — or from the
  first chunk a pending commit filled — onward (`GpuChunksGuard::extend_decode`); only an
  append past the class rebuilds. Two rings alternate by wave, each fenced by an event
  recorded when its wave finishes, so staged bytes outlive every replay that reads them. A
  full ring falls back to an eager upload.
- **cuBLAS.** `CudaDevice::cublas()` rebinds the handle to the launch stream on every call and
  re-applies the handle's fixed 4 MiB workspace — `cublasSetStream` resets it to cuBLAS's own
  pool, which allocates on the stream, and a graph allocation is refused by the audit.
- **Launcher scratch growth** (split-KV partial pools in the paged decode and prefill
  kernels): `grow_scratch` allocates in relaxed capture mode, grows by doubling, and never
  frees the block it replaces, since a queued or recorded launch may still read it.

The silent cases of §0.2 are closed structurally:

- **Default-stream launches.** Every launcher in `candle-kernels` takes its stream from the
  caller (the remaining `<<<g, b>>>` sites are the convolution dispatcher, the GEMX repack and
  the L2-flush benchmark, none on an inference path). A missed one runs at once, out of
  order; finding the last of them is what restored Qwen3.6's draft acceptance (§3).
- **Readbacks** go through device methods that pause.
- **Host fences that cover "everything issued"** — an event recorded on the compute stream to
  fence a reuse — are recorded after a pause, so they cover the recorded work too
  (`TableRing`, the pinned stager's generation fence, KV selection).
- **Raw slices replaced while recording** (a workspace swapped for a larger one) are replaced
  inside an eager section, so the old buffers are freed behind their last reader.
- **A launch-stream handle taken while recording and used in an eager section.** The handle
  is the capture stream, which is not capturing during the section, so its launch ran at
  once — unordered against the segments still executing on the compute stream. This was
  Flash-Next's intermittent `ILLEGAL_ADDRESS` (§3.2): the indexer's staged uploads took the
  stream before entering `with_staged_upload`. Two device-side event hand-offs close it for
  every such launch, not only the ones found: pausing (and finishing a wave) records an event
  on the compute stream that the capture stream waits on, so a stray launch runs after the
  launched segments; resuming (and opening recording) records one on the capture stream that
  the compute stream waits on, so it completes before anything issued after the section.
  The indexer now also takes its stream inside the closure (rule 6).
- **Profiling spans.** `gpu_span` records its events with `CU_EVENT_RECORD_EXTERNAL` while the
  stream is capturing, so they become event-record nodes (which the audit admits) instead of
  invalidating the capture, and the span ring's high-water drain — it queries events — runs
  in an eager section.

### 2.4 Rules for code that runs while recording

1. Launch on `dev.cuda_stream()`; never on a stored stream or `slice.stream()`.
2. Anything that must still name a stream after the call — a slice built with
   `upgrade_device_ptr`, an event, a registry key, a fence — uses `dev.compute_stream()`.
3. Host protocol that allocates, frees raw slices, reads back or fences runs inside
   `dev.eager()?`.
4. A host upload goes through `upload_raw` (recorded) or a device method (eager).
5. A struct owning raw `CudaSlice`s is replaced only inside an eager section.
6. Code inside an eager section takes the launch stream inside it, never before it — the
   handle taken outside is the capture stream (the hand-offs of §2.3 keep a miss ordered,
   but it is still a miss).

---

## 3. Status and results

### 3.1 Qwen3.6-35B-A3B — recording, gate green

`qwen35::forward::sweep_layers`, the layer loop of `HybridBatched`'s `WaveSweep`, marks
recording just before its layer loop. Every row of
`quantized_qwen36_moe::tests::test_parallel_batched_forwarding_36_35b` validates.

| mode | ctx | prefill t/s (base → graphs) | decode t/s (base → graphs) | decode Δ |
|---|---:|---:|---:|---:|
| BF16 | 1 | 1001.8 → 996.9 | 119.7 → 134.2 | +12% |
| BF16 | 4 | 4187.3 → 4144.4 | 513.3 → 554.6 | +8% |
| Q8_0 | 1 | 2692.6 → 2728.4 | 153.8 → 172.5 | +12% |
| C0 | 1 | 2889.9 → 3021.1 | 153.4 → 170.7 | +11% |
| C1 | 1 | 3006.8 → 3120.4 | 150.4 → 170.5 | +13% |
| C2 | 1 | 3009.5 → 3116.3 | 153.4 → 169.8 | +11% |
| C3 | 1 | 2983.6 → 3110.1 | 153.1 → 170.1 | +11% |
| C4 | 1 | 3018.3 → 3093.6 | 151.6 → 167.1 | +10% |
| C5 | 1 | 2992.8 → 3092.5 | 152.4 → 168.5 | +11% |
| C5 | 8 | 4424.3 → 4438.4 | 666.3 → 689.8 | +4% |
| C6 | 1 | 3035.5 → 3035.8 | 148.4 → 164.3 | +11% |
| C7 | 1 | 3363.0 → 3466.4 | 150.1 → 165.6 | +10% |
| C8 | 5 | 4858.7 → 4943.2 | 570.8 → 593.9 | +4% |
| C9 | 2 | 4001.7 → 4103.1 | 258.3 → 280.9 | +9% |
| C10 | 8 | 4595.0 → 4657.4 | 627.0 → 638.9 | +2% |
| C10 | 16 | 4249.9 → 4443.2 | 753.4 → 759.8 | +1% |

Final code (with the expert-pipeline changes of §3.2, which this gate shares), from the
2026-10-06 sweep, every row validated. Prefill runs between 1% under (BF16×4) and 4.5% over
(C0×1, C10×16) its pre-graphs rate, since prompt rows stopped speculating (§3.2 item 8). A wave runs as about 52 segments of about 25 launches, almost all updated in place.

### 3.2 Qwen3.8-Flash-Next — recording, gate green

`Qwen4ExpBatched::sweep_layers` marks recording just before its layer loop, QSA selection
included. Every row of `quantized_qwen38_moe::tests::test_parallel_batched_forwarding`
validates (RTX 3090, Q2_KO experts; the working set is about three times the expert zone, so
the expert cache streams throughout).

| mode | ctx | prefill t/s (base → graphs) | decode t/s (base → graphs) | decode Δ |
|---|---:|---:|---:|---:|
| BF16 | 1 (cold) | 413.6 → 417.5 | 47.1 → 62.8 | +33% |
| BF16 | 4 | 1262.4 → 1248.0 | 157.8 → 227.8 | +44% |
| BF16 | 8 | 1510.9 → 1548.6 | 224.9 → 299.4 | +33% |
| BF16 | 16 | 1577.8 → 1565.5 | 240.6 → 362.9 | +51% |
| BF16 | 1 (warm) | 491.7 → 482.5 | 71.1 → 85.6 | +20% |
| C0 | 2 | 894.1 → 925.4 | 138.2 → 162.8 | +18% |
| C5 | 2 | 898.1 → 928.6 | 137.2 → 159.3 | +16% |
| C5 | 8 | 1499.1 → 1534.6 | 228.3 → 297.4 | +30% |
| C8 | 2 | 893.4 → 861.5 | 133.4 → 147.6 | +11% |
| C10 | 2 | 892.0 → 917.3 | 130.1 → 133.5 | +3% |
| C10 | 8 | 1521.4 → 1551.4 | 209.2 → 242.7 | +16% |

Final code, 2026-10-06, run alone on a warm page cache (the sweep's run of the same gate, made
right after Qwen3.6's 22–42 GB checkpoints had evicted Flash-Next's files, read the cold row's
prefill at 264 t/s and every other row within the ~3% run-to-run spread of these). Prefill runs
between 1.9% under (warm BF16×1) and 3.5% over (C0×2) its pre-graphs rate on every row but
C8×2, 3.6% under (§5). About 75 segments a
wave, nearly all updated in place.

Recording exposed two problems, a race and a pipeline that could not keep up.

**The race.** With recording on, a wide prefill faulted (`ILLEGAL_ADDRESS`) intermittently.
The cause was a stale launch-stream handle (§2.3): the indexer's staged uploads took the
capture stream before entering their eager sections and launched on it there, unordered
against the MoE segments still executing. The hand-offs close it structurally.

**The expert pipeline.** Recording makes the forward cheap to issue, so the forward thread
runs a wave ahead of the GPU — and the expert pipeline was built on two timing assumptions
that this broke. Measured on the ×1 cold row before the fixes: 39 t/s, with three quarters of
decode's misses finding no promotion-ring slot and crossing the link a second time by the copy
engine (12,700 copy-engine promotions, 16 GiB). In order of discovery:

1. **Reclaim keyed on what the device has begun** (`reclaim.rs`, `started.rs`). A slot was
   reusable once the row's latest *enqueued* invocation completed; a wave ahead, every row
   always had one, and every eviction waited a pass on the retire list. Bucketize now stores
   its ticket in its row's mapped *started word*, behind a system fence, before reading the
   live table, and the host retargets, fences and reads that word — the same store/load
   pairing the enqueued key had, on the invocation the GPU is actually inside. Reuse also
   follows the device's latest started ticket rather than only the summaries a host thread has
   read (an in-order stream makes them equivalent bounds).
2. **Copy-engine promotions issued off the pipeline thread** (`copier.rs`). The real cause of
   the dry ring: on WDDM, issuing a `cuMemcpyHtoDAsync` from pinned memory stalled the pipeline
   thread for 157–189 ms once per step — the rest of the forward the GPU was running — and the
   ring it restocks every layer ran dry behind it. A copier thread then issued those copies and
   reported completions, and unslotted misses went from ~12,300 to ~0. The copy-engine prefetch
   is gone since: the GEMM workers read ahead into ring slots and publish the entries
   themselves (`moe_live_dispatch_design.md` §0.7.3), so the pipeline thread issues no copy at
   all and its per-layer work is host memory and mapped words only.
3. **Verify rows scored as decode** (`DecodeRows`, `models/residency_rows.rs`). Residency
   scoring weights a decode row's routing above a prompt row's, and bucketize's decode bit
   covered only the leading decode rows. Flash-Next decodes entirely through verify waves,
   which are laid out as prefill segments, so every decode miss arrived scored as a prompt
   miss — the zone's cheapest victim — and was evicted again at the next refill. The decode
   rows are now a set of token ranges passed by value to bucketize: the decode rows, every
   verify segment, and each prompt's last row (the token its decode continues from). This one
   change took ×4 decode from 95 to 223 t/s at the time.
4. **Rows about to run held back as victims** (`take_slots`). The started-word key made every
   row ahead of the GPU quiet, so a wide prefill evicted the experts of the rows it was about
   to read; on Qwen3.6 that doubled a prompt's misses and cost up to 24% of prefill. Rows with
   an invocation enqueued and not begun (`ReclaimClock::upcoming`) give up victims only for a
   shortfall. Holding back only a window of rows just past the GPU's front measured the same
   as holding back every enqueued row once item 6 was in; not holding back at all cost
   Flash-Next's wide decode most of its gain.
5. **Executables per wave shape** (§2.2), which took the re-instantiations after each prefill
   to near zero.
6. **Decode misses credited, with a long memory beside the short one** (`cache.rs`). A miss
   was scored as a fresh elevation (−0.1), so the expert the next decode step would route
   again was the zone's first victim: Flash-Next's wide rows thrashed (×16 225 t/s). Crediting
   the miss as a hit (+1) fixed them (×16 368) but cost the short rows: with only the ×0.85
   per-pass decay, an expert decode last routed 50 steps ago scores ~0, so a reply that
   returns to it pays for it again (C10×2, 13 steps: 150 → 120 t/s, its first step 57 →
   107 ms). One slower decay could not serve both (×0.99: ×2 rows +8%, ×16 −16%). The score
   is now the recency term plus `DECODE_REUSE_WEIGHT` (0.1) × a decode-reuse term earned only
   by decode **hits** and decayed ×0.99 a pass: a miss holds its slot to the next step, and
   only reuse makes it outrank what decode keeps returning to. Weights 0.02–0.1 each raised
   every row a little; 0.3 cost ×16 7%. Scores age only on passes that carry a decode row.
7. **Prompt promotion only with room** (`PROMPT_ROOM`, the ring's reserve). Where the zone
   holds under three quarters of the experts, a prompt's misses are left to the workers'
   copies: promoted, a two-sequence Flash-Next prompt took ~12,600 promotions into a
   ~12,000-slot zone, cycling decode's working set out, and its prefill rows ran up to 5%
   slower. The ring's reserve (a kernel parameter: a prompt-only expert takes a slot only
   while the stock exceeds it) is decode's ring target there, and 0 with room. Promoting a
   prompt only into victims no decode step held, or into stock past decode's, bought nothing.
8. **Speculation only where nothing else moves the expert.** A launch carrying prompt rows
   speculates only without room: with room the ring already promotes the prompt's misses
   and a speculative copy competes with them (Qwen3.6's one-sequence prompts 4–9% slower);
   without room it is the only way a prompt's experts reach VRAM ahead of the workers
   (Flash-Next's warm one-sequence prompt 4% faster). Decode-only launches speculate as
   before.

The pipeline statistics table now reports the ring's health: slots bucketize took, misses it
found the ring empty for, and how many invocations the device had begun past the one the
pipeline thread was serving.

**The draft walk records too.** A speculative step's walk (`draft_walk`, shared by every
NextN drafter) opens a wave capture of its own between the verify forwards: each drafted
position is one head block over a row per sequence — dozens of small launches — and issued
eagerly the walk was launch-bound, ~3.9 ms of every 30 ms single-session Flash-Next step
idle between its back-to-back launches. Recorded, with its per-step slot headers built
inside the recording, the step fell from 29.4 to 28.2 ms (254 → 265 tok/s).

### 3.3 Other models

The Qwen3.5 lineage (0.8B, 9B, Qwen3.8-27B, 3.5-35B and the three Qwen3.6 gates) records
through `qwen35::forward::sweep_layers`. Qwen2, both Llamas, Qwen3-8B and Qwen3-30B-A3B share
one uniform sweep, `BatchedInference::forward_wave_contexts`, which marks recording before its
layer loop in the same place. Every gate of both groups passes with every validated session
green.

**A segment per layer.** That sweep's layers meet the host nowhere, so recorded whole it ran
as one segment a wave, and the GPU sat idle until the host had recorded every layer where eager
launches had it start on the first: decode fell 5–9% under eager. It hands each layer to the
device as it is recorded (`Device::flush_launches`), the MoE layers of Qwen3-30B-A3B and the
Qwen3.5/3.6 MoE checkpoints included: the expert dispatch flushes only on
`expert_lre::flush_schedule`'s doubling cadence (§5), measured for Flash-Next's sweep, so these
sweeps keep their own per-layer cut. A layer whose dispatch has just flushed adds an empty cut,
which launches nothing.

**An in-place cast that read ahead of itself.** `to_dtype_mut` — the MoE's BF16→F16 narrowing
on Qwen3-30B-A3B's F16 and quantized-KV rows — copied its result into the retyped buffer with a
synchronous legacy-stream copy, which ran at once, before the recorded cast; and freed the
buffer it replaced at once, before the segment reading it. Every such row failed validation.
The copy is queued on the launch stream and the old buffer retired
(`an_in_place_cast_while_recording_reads_back_what_it_reads_eagerly`).

RTX 3090, decode t/s, eager (2026-10-06 sweep) → recorded:

| model | row | eager → graphs | Δ |
|---|---|---:|---:|
| Qwen2-0.5B | F32 ×1 | 257.2 → 485.2 | +89% |
| | F16 ×4 | 1,098.8 → 1,482.3 | +35% |
| | BF16 ×60 | 6,268.0 → 6,468.1 | +3% |
| Llama-3.2-3B | F16 ×1 | 165.2 → 178.5 | +8% |
| | BF16 ×4 | 514.2 → 545.9 | +6% |
| | C8 ×10 | 750.5 → 775.6 | +3% |
| Llama-2-7B | BF16 ×1 | 110.1 → 112.5 | +2% |
| | BF16 ×48 | 945.6 → 938.1 | −1% |
| | Q8_0 ×32 | 621.8 → 599.5 | −4% |
| Qwen3-8B | BF16 ×1 | 78.4 → 81.7 | +4% |
| | C8 ×10 | 361.4 → 380.8 | +5% |
| Qwen3-30B-A3B | Q8_0 ×20 | 344.7 → 388.8 | +13% |
| | C0 ×2 | 93.5 → 105.3 | +13% |
| | BF16 ×10 | 351.3 → 363.9 | +4% |

Prefill is within run-to-run spread of eager on every row but Qwen2's single-sequence rows
(+18–23%). Llama-2-7B's widest rows are the one place recording does not pay: 32–48 sequences
make a layer long enough on the GPU that recording it ahead buys little, and the per-segment
cost is left over.

---

## 4. Tests

`cargo test --release --features cuda -p candle-core --lib cuda_backend::graph`:

| Test | Proves |
|---|---|
| `replay_on_the_null_stream_orders_against_null_stream_work` | A graph captured on the side stream and launched into the null stream orders against null-stream work on both sides. |
| `a_capture_survives_null_stream_work_from_another_thread` | Another thread's null-stream work neither breaks nor joins a capture (`ThreadLocal`, non-blocking capture stream). |
| `a_default_stream_launch_is_not_recorded_and_the_capture_is_refused` | A default-stream launch runs at once and the short graph is refused. |
| `a_dropped_session_leaves_the_stream_usable` | An unfinished capture ends on drop. |
| `migrated_launchers_record_on_the_capture_stream` | The repository's launchers record and replay bit-identical. |
| `a_recapture_with_new_scalars_and_addresses_updates_in_place` | `update` folds new scalars and addresses in place; a new topology is re-instantiated. |
| `a_host_copy_inside_a_capture_is_refused_and_a_device_copy_is_kept` | The audit. |
| `a_wave_capture_is_bit_identical_to_eager_execution` | Element-wise ops, cuBLAS, reductions, a mid-chain readback, allocations, frees and a flush give bit-identical results recorded and eager, wave after wave, with in-place updates. |
| `an_upload_while_recording_lands_in_issue_order_without_a_cut` | A recorded upload is ordered between the launches around it and does not end the segment. |
| `only_the_capturing_thread_is_redirected` | A held wave redirects nothing; a recording one redirects only its own thread. |
| `a_stale_capture_stream_launch_in_an_eager_section_stays_in_order` | A launch on a capture-stream handle taken while recording, issued inside an eager section, runs after the recorded work before it and before the recorded work after it (the hand-offs). |
| `alternating_wave_shapes_each_fold_into_their_own_executable` | Two wave shapes taking turns at one ordinal each fold in place once both have run. |
| `two_shapes_of_one_node_count_each_keep_their_executable` | The same, for two shapes of equal node count and different topology. |
| `a_release_from_another_thread_waits_for_the_recording_segment` | Memory another thread releases while a segment records is freed only once that segment is launched. |
| `a_retired_item_may_retire_from_its_own_drop` | The graveyard is emptied outside the hub's lock. |
| `a_pause_guard_outliving_its_wave_leaves_the_next_alone` | A pause guard acts only on the wave it paused. |
| `layer_stream::cache::tests::a_recording_wave_reads_every_layer_it_streams` (ignored, GPU) | A streamed layer store loads and joins layers inside a recording wave, and every recorded read sees its own layer's bytes. |
| `per_wave_capture_cost`, `null_stream_calls_during_capture` (ignored) | §0.1 and §0.2. |

The expert pipeline's pieces, `-p candle-core --lib` and `-p candle-transformers --lib`:

| Test | Proves |
|---|---|
| `cuda_moe_bucketize_stores_its_ticket_in_its_rows_started_word` | Bucketize writes its ticket to its row's started word, and leaves it untouched when it reads no live table. |
| `cuda_moe_bucketize_live_table_orders_remote_first` (extended) | The decode bit follows arbitrary token ranges, not only a prefix. |
| `decode_rows::tests::*` | Ranges merge when they touch, a range past the kernel bound is left out, the bound mirrors the kernel's. |
| `reclaim::tests::*` | Reuse keys on the started word; a queued, unbegun invocation holds no slot; a later start on any row completes every earlier invocation; an enqueued, unbegun row is upcoming. |
| `residency_rows::tests::*` | A wave's decode rows, verify segments and prompts' last rows, as ranges. |
| `cuda_moe_bucketize_promotes_remote_experts_from_the_ring` (extended) | A prompt-only expert takes no slot without room for it, none while the stock is at the reserve, and none when there is no reserve word; a decode expert takes the reserved slot. |
| `cache::tests::a_decode_miss_earns_the_decode_credit_but_no_reuse`, `decode_reuse_decays_on_its_own_slower_clock`, `a_one_off_decode_miss_is_evicted_before_a_reused_expert`, `a_prefill_elevation_keeps_the_experts_decode_reuse` | §3.2 item 6's two-term score. |

`batch_test::graph_report` prints each gate row's line — waves, segments per wave, launches
per segment, in-place updates, instantiations — and is unit-tested on its text.

---

## 5. Next

1. **Flash-Next's C8×2 prefill.** It runs 3.6% under its pre-graphs rate (861.5 against
   893.4 t/s) under every policy that keeps the wide decode gains; only scoring decode misses
   as elevations (§3.2 item 6) restores it, and that costs ×16 decode a third. The prompt
   hits more of the zone when the zone holds old residents rather than decode's recent
   experts; a prompt-side long memory (prompt hits earning reuse) did not move it and cost
   Qwen3.6's prefill 3–5%.
2. **Qwen3.6's verify rows.** `DecodeRows` now reaches its hybrid sweep through
   `BatchedAttentionLayer::ffn_residual`, but it passes the decode prefix only: scoring its
   verify segments and prompts' last rows as decode, as Flash-Next does, measured lower
   prefill on its one-sequence rows.
3. **Fewer segments — done for the MoE flushes.** The dispatch's host readers poll the
   summary word bucketize writes *when it executes*, so launching each MoE invocation's
   segment at once bought nothing past the start of a wave, where the GPU must not idle
   while the host records. `expert_lre::flush_schedule` now launches a wave's segments at
   1, 1, 2, 4, 8, 16, then 32 MoE invocations (keyed on the hub's segment ordinal, so it
   restarts every wave and a segment some other cut began counts from its own start), and a
   flush no longer records the pause's two cross-stream hand-offs. On Flash-Next's
   single-session Strata benchmark segments per wave fell 19.5 → 4.4 and, with the
   bucketize and restock changes of the same round, decode rose 148.9 → 156.1 t/s at 4K and
   121.9 → 134.2 at 128K (RTX PRO 5000, 2026-10-09). What remains of the attention layers'
   header builds cuts only when its slot-state buffer needs a device call a recording cannot
   make: a full ring, a fence, a slab claim.
4. **Fuse MoE layers whose experts are all resident.** With the flush schedule a segment
   already spans up to 32 MoE layers; what is left is the host protocol's per-invocation
   bookkeeping (`before`), not a segment cut.

---

## 6. The earlier design, and why it changed

The previous version of this document specified graphs **captured once and replayed**: a
lazily filled cache keyed on `(layer or span, region kind, decode rows ≤ 4, prefill bucket,
tier placement epoch, variant)`, a typed region API (`Region`, `Recorder<M: Mode>` with
`Eager` / `Chained` / `Capturing`, operand roles `Act` / `Resid` / `Table` / `Weight`, a
`NotInGraph` anti-trait of banned calls with `compile_fail` tests), a fence-free region mark
rewinding the wave tier at every link, fixed-address staged tables, a device-derived MoE
sequence number, a `Chained` fallback for refused slots and for waves above four rows, and a
composition layer (`Fuse`, `Known`, variants, fork/join, beacons, pending waves).

Every one of those exists to make a **stale** replay correct: a graph replayed with the
addresses and scalars of the wave that captured it. Measured, recording every wave and
updating its executable costs 2.4 µs a launch against 10.8 µs eager (§0.1), and a recapture
carries the current wave's addresses and scalars by construction. So:

- the key, the epoch (`tier_epoch.rs` was built, measured — a layout held for 10–30 decode
  steps — and removed), the four-row cap, the prefill bucket and the variants are unnecessary:
  every wave is recorded at its own shape;
- the region mark and fixed-address tables are unnecessary: tier addresses may differ wave to
  wave;
- the device-derived MoE sequence number is unnecessary: `summary_seq` and the ring slot are
  recorded fresh every wave;
- the `Chained` fallback is unnecessary: there is no slot to refuse, and a failed recording
  fails the forward, which the wave driver already rolls back;
- the typed API's guarantees are replaced by the ordering argument of §2.2 and the runtime
  backstops of §2.3 (the audit refuses host copies and graph allocations; the driver refuses
  syncs, allocations and frees on the capturing thread) — at the cost of the compile-time
  bans, which remain a possible later hardening of the rules in §2.4.

What carried over unchanged: capture on a created non-blocking stream in `ThreadLocal` mode
and launch into the null stream (§4.2 of the old design, proven by the first two tests), the
MoE dispatch split into `before` / `record` / `after` with the flush in `after`, every launcher
taking its stream, launchers reporting failure instead of skipping silently, and the node
audit.

Non-goals, unchanged: splitting a wave, row padding, a graph per slot combination, moving the
MoE host protocol into a graph, device-data control flow, a per-model opt-out flag or an
environment toggle, and moving the compute stream off the null stream.
