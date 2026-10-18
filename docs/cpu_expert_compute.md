# CPU Expert Compute

**Status:** design, being built. Reviewed 2026-10-16: an adversarial review
against the code, and an inventory of candle's CPU side. Both are folded in
below.

Decisions taken by the author, 2026-10-16:
- **CPU results are not byte-identical to the GPU's.** Output may depend on
  placement (§6).
- **The invariant amendments in §4 are approved as written.**
- **CPU-served misses claim no promotion slot of their own launch.** Revised
  2026-10-17: the next launch may promote them behind the pool, copying from
  their pinned image off the result's path (`promote`, §5.11) — cold ones
  first, then every one, to measure which helps.
- **The build runs straight through.** The measurements in §8 run alongside it
  and tune it; they do not gate it.

A routed expert that misses VRAM can be computed on the host CPU, where it
already sits in RAM, instead of having its weights copied to the GPU over
PCIe. This document covers:

- why the CPU can be faster;
- where it is not;
- how the CPU path fits the live MoE dispatch, for decode, prefill and mixed
  waves alike, without breaking the hot-path invariants.

Two invariants are amended (§4). Strata is the reference implementation (§2).

Related designs:
- `docs/moe_live_dispatch_design.md`: the dispatch this extends.
- `docs/expert_cache_design.md`: the tiers.
- `CLAUDE.md`: hot-path invariants 3, 4 and 7.

---

## 1. Why

A routed expert that misses VRAM is served today in one of two ways:

- **Warm.** The expert sits in pinned host RAM. The grouped GEMM's worker
  blocks copy each 32-row slice into VRAM scratch over PCIe, then run the int8
  GEMM on it (`kernel.cuh:1625-1680`, `moe_live_worker` at `:2918`).
- **Cold.** The stager reads the expert from the pack on NVMe into a pad slot,
  and the GPU then copies it the same way (`expert_lre/stager.rs`).

Either way, every byte of a missed expert crosses PCIe once per projection per
layer. At decode an expert is a GEMV: one or a few tokens against about 2 MB of
weights. The GPU's compute on it is negligible; the cost is the copy.

A CPU reading the same expert from the same RAM is limited by DRAM bandwidth,
not by the link, and on every machine in the fleet DRAM bandwidth exceeds PCIe
bandwidth (§3). Splitting the misses between the two serves them at up to the
full DRAM rate (§5.3).

Strata reports about 75–94 tok/s single-stream (self-reported) from a 12 GB RTX
5070 by doing this on the same model family (§2.4).

### 1.1 What it does not fix

- **Cold experts.** An expert that is only on NVMe is read at the drive's rate
  whoever computes it. On the 16 GB laptop, 61–68% of expert loads read the
  pack (`expert_cache_design.md:814-829`). For those, CPU compute saves one PCIe
  crossing and nothing else.
  - The laptop is bound first by RAM *capacity*. Flash-Next at Q3_KO is 25,088
    experts × 1,996,800 B = **50.1 GB**, against 31.5 GiB of RAM.
  - Strata's reference box holds its whole expert set (34 GB of 1.38 MB Q2_0
    experts) in 64 GB.
  - What this design does for the cold share is make residency price it (§5.11).
- **Cards that already hold everything.** The 72 GB RTX PRO 5000 hits VRAM
  98.8–100% of the time on Flash-Next
  (`results/flash_next_single_session_rtx_pro_5000_72gb_2026-10-06.md:97-98`).
- **Heavy experts at prefill.** Past a few tokens per expert the CPU is
  compute-bound and the GPU wins. The split (§5.3) keeps those on the GPU, and
  the CPU takes the low-count tail plus, when the GPU is the bottleneck, the
  lowest-count resident experts.

---

## 2. Strata, the reference

Repository: <https://github.com/Niko1221/Strata> (MIT), read at `296b59d`,
v0.1.44.1, 2026-10-14. It is its own C++/CUDA engine, not a llama.cpp fork, and
links ggml-cpu only for the i-quant CPU dot products. All performance figures
below are the author's own runs; none has been reproduced independently.

### 2.1 Placement

Strata holds the following on the GPU (`docs/DETAILS.md:1909-1911`):
- attention, the DeltaNet mixers and the routers;
- the shared expert, the head and MTP;
- an expert cache filling the rest of VRAM.

All 24,576 routed experts (48 × 512, top-10) are pinned in RAM. Per layer, per
token, each routed expert gets one of three classes
(`src/core/expert_source.cpp:3720-3853`):

| Class | Meaning |
|---|---|
| 0 | VRAM hit |
| 1 | PCIe share: copied to a 16-slot staging buffer and computed by the GPU |
| −1 | Computed by the CPU in place |

The **PCIe share** is the last `pcie_num/256` of the distinct misses in routing
order. Since v0.1.42 it is measured: `share = L/(L + 1.85·R)`, where L is the
probed link GB/s and R the CPU pool's GB/s per expert. It is clamped to
0.10–0.60 and changes at most three times (`DETAILS.md:1698-1709`). On an RTX
3060 that took decode from 47.7 to 62.1 tok/s.

Residency starts from a static frequency profile (`profile.bin`). An adaptive
tier then swaps up to 96 experts every 4 windows by decayed counts
(`src/program/generate.cpp:596-628, 8833-8859`).

### 2.2 The CPU kernel

- **Format.** Q2_0: 64 weights per fp16 scale, 18 B per block, 1,382,400 B per
  expert at H=2560, FF=640 (`include/strata/kernels/cpu/expert.hpp:36-49`).
- **AVX-512 (F/BW/VL/VNNI/VBMI).**
  - `vpmultishiftqb` unpacks the 2-bit codes and `vpdpbusd` does the dot, one
    each per 64-weight block (`src/kernels/cpu/expert.cpp:113-190`).
  - The accumulator is reduced once per row.
  - Recorded: 15.03 ms for one token's routed set on 6 cores, at
    **44.14 GB/s (DRAM-bound)**.
- **Other tiers.** AVX-VNNI, AVX2 and AVX1. No AMX, no AVX-512 BF16.
- **Multi-token.** A multi-token kernel runs when ≥ 2 tokens share an expert
  (`MAXT = 8`). Single tokens use ggml's `vec_dot` (`native_expert.cpp:143-231`).
- **Pool.** One worker per *physical* core, P-cores first, pinned via
  `SetThreadSelectedCpuSets`.
  - The host thread drains work too: 33.7 vs 36.8 GB/s ideal
    (`src/kernels/cpu/pool.cpp:97-378, 867-891`).
  - Work is claimed from one atomic `(epoch, n, i)` word.
  - Two phases with a barrier between them, about 3 tasks per thread
    (`pool.cpp:596-605, 778-846`).

### 2.3 The handshake

Decode is one CUDA graph per verify window. Per layer
(`src/core/verify.cpp:323, 1453-1686`):

1. The GPU routes. `doorbell_publish` writes the hidden rows (T × 2560 f32)
   and ids into `cudaHostAllocMapped` memory.
2. The host thread, spinning, plans the split. It publishes the GPU plan (flag
   A), starts the PCIe-share DMA (flag B), then runs the pool.
3. The GPU waits for flag A inside a kernel (`wait_flag_ge`) and runs the VRAM
   hits, then waits for flag B and runs the PCIe share. The shared expert runs
   on a side stream.
4. The GPU waits for the final flag and reads the CPU rows back: up to
   T × 10 × 2560 f32.

The residual chain is serial, so CPU/GPU overlap is only *within* a layer.
Strata's own note says its first pipeline hid 1.055 ms of 19.035 ms of CPU work
(`include/strata/core/expert_cache.hpp:3-7`).

### 2.4 Strata's numbers, and what they say about ours

| Box | Quant | Decode tok/s | Prefill tok/s |
|---|---|---:|---:|
| RTX 5070 12 GB, Ryzen 5 7600 (6C, AVX-512), 64 GB DDR5-5200 | Q2_0 | 93.5 | 2,653 |
| same | IQ3_XXS | 61.6 | 1,745 |
| RX 9070 XT 16 GB, Ryzen 9 3900X, 47 GB | Q2_0 | 60 | 1,160 |

Our 16 GB laptop decoded 34.9 t/s at ×1 warm in the Q2_KO era
(`docs/performance.md:1514-1525`). The differences:

| | Strata box | Our laptop |
|---|---|---|
| Expert set fits in RAM | yes | no |
| Missed experts computed in place | yes | no (this design) |
| Expert size | 1.38 MB | 2.00 MB (Q3_KO) |
| Lossy route tail skip on by default (§2.5) | yes | no |

### 2.5 What we take and what we leave

| Strata | Here | Why |
|---|---|---|
| Fourth execution site: the CPU computes misses in place | **Take** | §1 |
| Split by measured CPU-vs-link rate | **Take**, on the device, per layer, per wave (§5.3) | Routing never leaves the device |
| A GPU kernel waits on a flag in mapped memory | **Take**, as a worker item of the down launch (§5.5) | The sanctioned wait shape; `cudaLaunchHostFunc` stalls under WDDM |
| A host thread *plans* the split after reading routing | **Leave** | A readback the device waits on (invariant 3). Planning is in `moe_bucketize`. |
| Multi-token CPU kernel per expert | **Take** | Verify windows, ×N waves and prefill share experts |
| Physical cores, pinned | **Take** (§5.6) | |
| Static `profile.bin` + adaptive swaps | **Leave** | The live residency scoring and promotion already do this, now priced by backing (§5.11) |
| `STRATA_ROUTE_TAIL_SKIP` (drops rank-≥7 misses, no renormalise) | **Leave** | Lossy: mean KL 0.003–0.006 |
| Expert Deferral (KTransformers) | **Leave** | Lossy |
| CPU and GPU round differently, so output depends on placement | **Take** | The CPU kernel folds however is fastest, and the pool pre-sums each token's CPU experts. Exact replays run all-GPU (§6). |
| `STRATA_*` env switches | **Leave** | No env feature flags |

---

## 3. The numbers

Figures marked ~ are estimates derived from datasheets and published
efficiencies. §8 replaces each with a measurement before anything is built.

### 3.1 The expert

Flash-Next (`host::MODEL = Qwen38_FlashNext`, `host/src/model.rs:18`):

| Item | Value |
|---|---|
| Layers | 48: 36 DeltaNet + 12 QSA, plus the MTP block |
| Routed experts | 512 per layer, top-10 |
| Shared expert | one, FF 640 |
| Hidden size | H = 2560 |
| Expert FF | 640 |
| Expert weights | `gate`, `up` [640, 2560], `down` [2560, 640] |

Experts are Q3_KO under 64 GiB of VRAM and Q4_KO at or above it
(`quant_ladder.rs:36-41`):

| | Q3_KO | Q4_KO |
|---|---:|---:|
| Bytes per chunk (8 rows × 128 K) | 416 | 544 |
| Bits per weight | 3.25 | 4.25 |
| Projection, 1,600 chunks | 665,600 B | 870,400 B |
| **Expert, 3 projections** | **1,996,800 B** | **2,611,200 B** |

- Per token-expert work: 3 × 2560 × 640 = **4.92 M int8 MACs**.
- Per decoded token: 480 routed experts, **958 MB** at Q3_KO; the MTP head adds
  10 more when it drafts.
- Per token, the expert input is one q8a128 row: 20 blocks × 144 B = **2,880 B**.
- The pool returns one F32 row per *token*: **10,240 B**. That row is the
  routing-weighted sum of the token's CPU experts (§5.5). `down_out` is F32,
  and the scatter reads F32 only (`cuda.rs:7449-7455`).

### 3.2 Bandwidth per machine

| Machine | CPU (SIMD) | DRAM, peak → realistic | Link, realistic | DRAM / link |
|---|---|---|---|---:|
| 16 GB laptop | Core Ultra 9 185H, 6P+8E+2LP (AVX2 + AVX-VNNI) | LPDDR5X-7467, 128-bit (8 × 4 GB): 119.5 peak → **81 GB/s measured** on all 16 physical cores, 69 on the 6 P-cores alone (§8, measurement 1) | ~12 GB/s (Gen4 ×8, read 2026-10-16) | ~6.8 |
| RTX 3090 box | i7-10700K, 8C (AVX2 only) | DDR4-2933 ×2: 45.8 → ~33 GB/s | ~12 GB/s (PCIe 3.0) | ~2.75 |
| RTX PRO 5000 | Ryzen 9 9950X3D, 16C (AVX-512 VNNI) | 189 GB fitted, so likely 2 DIMMs per channel at a reduced speed: ~60–70 GB/s | ~50 GB/s (PCIe 5.0) | ~1.3 |
| Workstation (planned) | Threadripper 7970X, 32C (AVX-512 VNNI) | DDR5-5200 ×4: 166 → ~120 GB/s | ~50 GB/s (PCIe 5.0) | ~2.4 |

- **"Realistic" DRAM** applies the ~72% sustained efficiency measured by
  KTransformers (220 of 307 GB/s) and MoE-Lens (150 of 205 GB/s). Untuned code
  reaches far less: 28% for PyTorch in Huang et al.
- **The laptop's link is Gen4 ×8.** `nvidia-smi` read on 2026-10-16:
  `pcie.link.width.current = 8` against a maximum of 16. The generation reads
  1 at idle and trains up to 4 under load. `dispatch.rs:112` and the
  CPU-throttle memory note were right; `CLAUDE.md` and
  `docs/performance.md:229` (×16, ~25 GB/s) are corrected in this change.
- **The laptop's memory is LPDDR5X-7467** on Meteor Lake's 128-bit bus (SMBIOS
  type 35, 8 × 4 GB packages), for a 119.5 GB/s peak. Measurement 1 on
  2026-10-16, GPU idle, read **81.2 GB/s** with one worker on each of the 16
  physical cores, which is 68% of peak.
  - The P-cores alone read 21.6, 37.3, 40.6, 52.7, 58.6 and 69.2 GB/s at 1–6
    workers.
  - SMT siblings add almost nothing: 83.6 GB/s on all 22 logical processors.
  - The E-cores add ~12 GB/s over the six P-cores. The pool's core choice
    (§5.6) therefore includes E-cores for streaming, rather than skipping them
    on principle.
- **DRAM is shared, but on the laptop barely contended.** A warm expert copied
  over PCIe is a DMA read of the same DRAM the pool reads. Measurement 2
  streamed 64 MiB pinned→VRAM copies while the pool read (2026-10-16, laptop,
  GB/s):

  | Pool | B_d alone | B_c while copying | Link under load | Sum |
  |---|---:|---:|---:|---:|
  | 4 P | 51.6 | 49.0 | 12.37 | 61.3 |
  | 6 P | 70.7 | 65.0 | 12.37 | 77.3 |
  | 16 physical | 83.6 | 74.6 | 12.27 | 86.8 |

  - The link alone ran 11.90 GB/s at Gen4 ×8, and stayed at Gen4 ×8 under
    load.
  - **The design's "B_c = B_d − B_l" is too pessimistic.** Up to six P-cores
    the pool is bound by its cores, not by DRAM, so the copies cost it only
    2.6–5.7 GB/s.
  - Only with every core does the sum meet the DRAM ceiling, ~87 GB/s.
  - The split therefore uses the measured `cpu_read_shared_ns`, never the
    subtraction (§5.3.1).

### 3.3 Where the CPU stops winning

For one expert with b tokens:

```
T_cpu(b) = max(S / B_d,  c(b))       memory-bound until compute catches up
T_gpu    = S / B_l + c_g(b)           one copy, whatever b is
the CPU is cheaper in isolation while T_cpu(b) < T_gpu(b)
```

c(b) is per expert at b tokens, and sub-linear in b because the kernel reuses
each unpacked weight across tokens (§5.10).

Estimates, assuming the kernel reaches half its int8 MAC peak (32 MAC per
cycle per core on AVX2, 32–64 on AVX-VNNI ymm, 64 on Zen 4 zmm VNNI, 128 on
Zen 5):

| Machine | Pool | c(1), µs | S/B_d, µs | S/B_l, µs | b at which CPU = copy |
|---|---|---:|---:|---:|---:|
| Laptop | 4 P-cores, ~3.5 GHz sustained | ~22 | ~38 (53 GB/s measured on 4 P) | ~166 | ~8 |
| 3090 box | 6 cores | ~11 | ~60 | ~166 | ~15 |
| PRO 5000 | 12 cores | ~1.3 | ~31 | ~40 | ~31 |
| 7970X | 28 cores, all-core AVX-512 clock unconfirmed | ~1.1 | ~17 | ~40 | ~36 |

What the counts look like:

- **×1 decode** routes 1 token per expert, up to T ≤ 4 in a verify window. In
  isolation the CPU is cheaper for every warm miss on every machine. The split
  still gives the GPU a share, so both resources run (§5.3).
- **×16 decode** has 160 assignments over 512 experts per layer. About 138
  distinct experts are touched, almost all with counts of 1–3.
- **Prefill at 512 tokens** has a mean of 10 per expert.
  - On the 3090 box most experts are *below* the crossover: P(n ≥ 15) ≈ 0.08
    at λ = 10. The split, not the cut, decides there, and the CPU's compute
    rate is the limit.
  - On the laptop more of them pass it.
  - Strata's v0.1.41 CPU prefill share reported −20 to −35% prompt time at
    512–1,000 tokens.
- **Laptop thermals.** The CPU throttles to 25–90% under sustained load, so
  c(b) there is what §8 measures under GPU load, not an idle number.

### 3.4 What crosses the link

| Item | Size | Notes |
|---|---|---|
| Activation rows out | 2,880 B per token | Written before routing at decode (§5.9). Exported per CPU token at prefill, and charged to the CPU side (§5.3). |
| Results back | 10,240 B per CPU token, pre-summed over its CPU experts | Pulled in bulk into VRAM by the down launch (§5.5) |
| Signalling | one doorbell line out, one `done` word back | §5.9 |

The fixed cost per layer with CPU work is about 3–5 µs: a measured 2.2 µs
signalling round trip (§5.9) plus the result pull. That is 0.15–0.25 ms per
token over 48 layers.
- It matters when the CPU has one or two experts (~31–60 µs of work).
- It is noise when the CPU has several.
- A layer that gives the CPU nothing pays only the decode row writes (§5.9).

---

## 4. Invariants this touches

The amendments below land in `CLAUDE.md` and
`docs/deepseek/deepseek_hot_path_invariants.md` in the same change as the code.
**The author approved both as written on 2026-10-16.**

1. **Invariant 3, no unnecessary GPU→CPU transfers.** Add a second sanctioned
   transfer:

   > the q8a128 rows of tokens the CPU expert pool may compute, written by the
   > device into mapped host memory, and the pool's results, pulled by the
   > device from mapped host memory. No host CUDA call reads or waits on
   > either.

   **Nothing on the device waits for the host to read anything.** The device
   waits only for the *results* of work it handed out, through a mapped word,
   as it waits for a cold expert's publish today.
2. **Invariant 4, host code does not compute over per-token data.** Add an
   exception: the CPU expert pool, for the experts the device assigned to it.
   The decision of what the pool gets stays on the GPU (§5.3).
3. **"Once a gate block can wait, nothing it waits for may need the driver"**
   (`moe_live_dispatch_design.md:60-62`). This is unchanged. The pool makes no
   CUDA calls and talks through mapped words only, like the stager
   (`stager.rs:17-21`).
4. **Only worker blocks spin; no block ever waits on another block.** The CPU
   wait is a worker item of the down launch (§5.5), honouring `SPIN_LIMIT_NS`,
   the abort word and the fault word (`kernel.cuh:1665-1674`). No new
   block-to-block dependency is created.
5. **Decode graphs** (`decode_graphs.md:96`). Every new device action is a
   kernel that reads or writes mapped memory. Recorded segments already do
   this: bucketize writes the summary ring, and read-ahead publishes
   live-table words (`kernel.cuh:2892-2899`).
6. **Reclaim rules R/R′** (`moe_live_dispatch_design.md:184-229`).
   - The device waits for the pool's `done` inside invocation T's down launch,
     before T+1's bucketize in stream order. Any started word at T+1 or above
     therefore already implies the pool is finished with T's slots, so Rule R
     covers pool reads as it stands.
   - Rule R′'s premise ("the GPU is inside T's gate launch", `stager.rs:23-27`)
     is reworded to "inside T".
   - The pool only ever receives addresses bucketize classified as pinned warm
     or pad.
7. **Determinism.** With CPU work in a layer, output depends on placement
   (§6). An all-GPU run (`max_cpu_assignments = 0`) is exactly as reproducible
   as today's.
8. **Pinned and mapped budgets.**
   - Pinned memory counts against WDDM's non-local budget (17.22 GiB on the
     laptop).
   - The 64 MiB / 256 MiB observation (`cuda.rs:2446-2450`) is a *time* limit:
     one launch writing host memory past the 2 s TDR. It is not a size cap. The
     rings are sized against the non-local budget, and no single launch writes
     more than the TDR allows.
   - Pageable warm slots need no pinning at all (§5.7).
9. **The existing refusals** (`expert_cache_design.md:898-902`,
   `expert_lre/handle.rs:11-12, 612`) refuse a CPU *device* build, which stays
   true. They are amended to point here, not replaced.

---

## 5. Design

### 5.1 One more class in `moe_bucketize`

`moe_bucketize` sorts every routed expert into VRAM, pinned or cold
(`dispatch.rs:905-937`). It gains a fourth class, **CPU**, decided on the
device per layer from:
- the expert's count;
- its live-table entry;
- `SplitParams` (§5.3.1).

The summary word is `count | pinned<<29 | cold<<30 | decode<<31`
(`dispatch.rs:249-250`). Bit 28 becomes the CPU class, which leaves the count
28 bits. A CPU-class cold expert keeps bit 30, so the stager still stages it
(§5.4).

The forward thread's sequence for a MoE layer:

```
router + shared gate_up      (unchanged: one stacked launch)
moe_bucketize                (classes, split, work slot, input map, carry)
moe_cpu_export               (each distinct CPU token's input row → row ring;
                              doorbell — ahead of the gather, from the MoE input)
gather                       (unchanged: every assignment's q8a128 row, grouped)
grouped gate → up → SwiGLU → down   (CPU-class experts own no tiles; down
                              workers take "CPU result" items after their
                              demand items: each waits for `done` and pulls
                              16 result rows into `down_out` at their grouped
                              rows; §5.5)
scatter                      (unchanged)
```

**Inputs once per token, results once per assignment.** A token's activation
is the same row for every expert it routes, so it crosses the link once
however many CPU experts it routes: bucketize ranks the distinct tokens of the
layer's CPU ring rows (`cpu_inputs`, in order of each token's first ring row)
into **input rows**, and publishes each ring row's input row. A decode token
routed to three CPU experts is one 2,880 B row out, not three. The export
copies the input rows straight from the token-major MoE input, so it runs
right after bucketize, before the gather, and the pool's doorbell rings as
early as the layer allows. A layer with no CPU expert exports, fences and
rings nothing; the export's few blocks only count themselves out.

The result ring holds one unweighted F32 down row per CPU *assignment*, ring
order, each landing in `down_out` exactly where the GPU's grouped down would
have written it, so the scatter, the permutation and the routing weights are
untouched: the scatter cannot tell which device computed a row. Pre-summing
per token would return fewer bytes but needs a per-token bitmap and a second
scatter input, and the bytes it saves are small next to the experts' (a 10 KB
row against a 2 MB expert).

- **The stacked launch stays.** The shared SwiGLU and down run after the
  stack (`quantized_moe.rs:173-175`, before the routed path at `:202-204`).
- **These launches are serial.** Everything is on one stream
  (`live_table.rs:103-110`), so the overlap is between the CPU pool and the GPU
  stream, not among GPU branches. The GPU work behind bucketize is what hides
  the pool's time:
  - the GPU-class experts' copies and grouped gate/up GEMMs;
  - the down launch's demand items, which every worker drains before it takes
    a CPU result item.

### 5.2 The rings

All buffers live in mapped pinned memory (`cpu_pool/rings.rs`), one of each
per dispatch.

| Buffer | Written by | Holds |
|---|---|---|
| **Work slot** | bucketize, only when the split takes something | A 16-byte header `{n_experts, n_rows, row, seq}`, then per CPU expert `{expert, ring_start, count, gate, up, down}` (40 B): its ring rows — one per assignment — and the host addresses of its slot image's three projections (0 for a cold expert, §5.4) |
| **Input map** | bucketize | The input rows' count, then each ring row's input row |
| **Row ring** | `moe_cpu_export` | The input rows: one q8a128 row per distinct token the CPU experts route, 2,880 B, in the operand's own q8a1024 layout |
| **Doorbell** (one cache line) | `moe_cpu_export`'s last block, only when it exported rows | A monotone u64: the invocation's sequence. The single line the pool monitor-waits on |
| **Result ring** | the pool | One unweighted F32 down row per CPU assignment, ring order |
| **Done** (one cache line) | the pool | The sequence whose results have landed |
| **Alive** (one word) | the pool | Non-zero while the service runs; a drop guard clears it on any exit (§5.13) |

Bucketize also writes `dst_row`, a device array mapping ring row → grouped
row, which the down launch's CPU item writes through, and `in_tok`, input row
→ token, which the export reads through.

**One slot of each is enough.** The down launch of invocation S waits for
`done == S` before anything after it runs, and S + 1's bucketize and export
come after it in stream order. By then the pool has read S's work and rows and
written S's results, so nothing on the host has to hold the device back, and
`hold_for_ring` is unchanged.

**Sizing.** `cap_rows` = `max_cpu_assignments` rows, from the knob at load.

| Buffer | Per row | At the default cap (1,024) |
|---|---|---|
| Row ring | 2,880 B | 2.9 MB |
| Result ring | 10,240 B | 10.5 MB |

`max_cpu_assignments` is a **constraint inside the split** (§5.3), counted in
assignments, not experts. Below the crossover the CPU could otherwise take up
to 512 × (b_x − 1) assignments, about 15k on the PRO 5000. The rings, and so
the cap, are sized against the pinned budget at load. On the laptop that
budget is tight.

### 5.3 The split, decided on the device

The split runs **per layer, per wave, over whatever the wave holds**. The
unified wave engine puts decode tokens, prefill chunks and glue in one
forward, so n_i is the number of tokens of *any* kind routed to expert *i* in
this layer of this wave. There is no decode rule and no prefill rule, only
counts.

**Costs.** One quant covers all routed experts, so every one has the same size
S:

```
g        = S / B_l                    GPU: one copy over the link
c_g(n)   GPU grouped-GEMM time for one expert at n tokens          (measured table)
c(n)     CPU compute for one expert at n tokens                    (measured table)
B_d      DRAM read bandwidth the pool gets alone
B_c      the same while the link's DMA streams                     (measured, ≤ B_d)
t_h      fixed handoff per layer (§5.9)
ρ        per CPU assignment: its 2,880 B row out and 10,240 B result back (row_ns)
```

**The pool's time.** It reads its k experts at S/B_c each while the GPU still
has copies to make, and at S/B_d once it has taken every miss:

```
read(k) = S / B_c   if k < misses,   S / B_d   otherwise
T_cpu   = t_h + max( k · read(k), Σ_{i∈C} c(n_i) ) + ρ · Σ_{i∈C} n_i
```

A token's input goes out once however many CPU experts it routes, and a
result row comes back per assignment (§5.1), so ρ, priced per assignment from
the link's measured rate at load, over-counts the inputs of a token routed to
several: an upper bound, by at most the 2,880 B input against the 10,240 B
result.

**The GPU's time** is the misses it keeps, each a copy then its grouped GEMM:

```
T_gpu   = Σ_{i∈M_g} ( g + c_g(n_i) )
layer expert time = max(T_cpu, T_gpu)
```

- This charges the copy and the GEMM in series. Inside one launch they overlap
  across experts, so it overstates the GPU's time; `gpu_scale` and the online
  loop (§5.3.2) absorb the difference, and the overlap is measured, not
  assumed (§8).
- The GPU's work that doesn't depend on the split (the shared expert, the
  gather, SwiGLU, launch tails) is the same on both sides and cancels.

**The rule.**

1. **Cut.** An expert with c(n_i) > g + c_g(n_i) stays on the GPU.
2. **Order.** Missed experts ascending by n, cold before pinned at the same n,
   then resident experts ascending by n. For a fixed number of CPU experts, the
   lowest counts are the cheapest set for the CPU and cost the GPU the same.
   Cold goes first because both sides wait on its pack read, but only the GPU
   waits in its gate launch, where the whole layer stalls; the pool waits
   beside the GPU's own work. On the laptop this took the GPU's cold waits from
   ~20 a decode step to under one (`cpu_order_key`).
3. **Pick the split.** For k misses and j residents given to the CPU, T_cpu
   rises and T_gpu falls. A two-pointer scan finds the (k, j) minimising
   max(T_cpu, T_gpu) subject to `max_cpu_assignments` and to k + j ≤
   `max_experts` — two per performance core of the pool (the laptop's ten;
   `attach::EXPERTS_PER_FAST_CORE`). Every run takes all the performance cores
   whatever its expert count (§5.6), so the cap is not a core count: under it
   the price decides, and it stops only the rare launch the price would load
   past what the device can wait for. On the laptop one a core bound the
   split at ~4.3 experts a layer (split decode 17.8–18.4 t/s); two let the
   price settle at ~5.2 (19.4 t/s); four took no more on average and ran
   17.6–18.6, the device waiting longer on the rare large layer. Past the cap
   a layer's misses stay on the GPU, at every width: a prompt's layer gains
   little from the pool
   (prefill measured ~5% slower with it uncapped on the laptop) and has as
   much reason to keep off the slow cores as decode. It is exact under this
   model and linear in the layer's expert count.
   - Counts at decode are ≤ T, so the order comes from a **histogram over
     counts**, not a sort. Bucketize's single block pays a pass, not an
     O(E log E) sort, on its critical path.
   - j > 0 only when the GPU is still the bottleneck after every eligible
     miss: a prefill wave whose GPU GEMMs dominate. **This is how the CPU
     accelerates prefill even when nothing misses.**
4. **Never slower.** The split is taken only if it beats all-GPU (k = j = 0) by
   `min_gain_ns`. Otherwise the layer is all-GPU and writes no work slot and no
   doorbell.
5. **Promotion follows the split.** Bucketize gives promotion slots only to
   the misses that stayed on the GPU (§5.11).

**Example: the 3090 box, M = 5 warm misses at ×1** (one token). All
estimates: B_d = 33, B_c = 21, B_l = 12 GB/s; g = 166 µs; t_h = 6 µs; rows
already out; Y_back is one 10 KB row (~1 µs).

| k | T_copy (µs) | CPU time (µs) | Layer miss time (µs) |
|---:|---:|---:|---:|
| 0 | 830 | — | 830 |
| 3 | 333 | 6 + 285 + 1 = 292 | 333 |
| 4 | 167 | 6 + 167 + 4.5 MB at 33 GB/s (137) + 1 = 311 | 311 |
| 5 | 1 | 6 + 5 × 61 + 1 = 312 | 312 |

k = 4 is the optimum, 2.7× on the layer's misses. k = 5 is within a
microsecond of it.

**Closed form at decode**, in the fully overlapped case: the balanced miss time
is M·S / B_d. Misses are served at full DRAM speed, a speedup of B_d / B_l over
all-GPU:

| Machine | Miss speedup (estimate) |
|---|---:|
| Laptop | ~7.1× measured with all 16 cores (86.8 ÷ 12.3); ~4.9× with a 4-P pool (61.3 ÷ 12.4) |
| 3090 box | ~2.75× |
| PRO 5000 | ~1.3× |
| 7970X | ~2.4× |

These are upper bounds. Mixed CPU+DMA traffic reaching B_d is measurement 1.

**A lone miss.** It goes to the CPU when t_h + S/B_d + y/B_l < S/B_l:
- 3090 box: 67 vs 166 µs.
- PRO 5000: ~38 vs 40 µs, a coin-flip that t_h decides.

**Prefill and mixed waves.** Heavy experts stay on the GPU by the cut. The CPU
takes the low-count misses, paying their row export and result return, and,
while the GPU is the bottleneck, the lowest resident experts. A mixed wave is a
count vector like any other.

**Residency of the candidates.**
- **Pageable warm:** CPU only (§5.7).
- **Cold:** the stager reads it into a pad slot either way, which adds the same
  amount to both sides (§5.4).
- **VRAM-resident:** a candidate only in step 3's j-phase.

**The model can be wrong; the loop corrects it** (§5.3.2). "Never slower" is
only as true as the costs.

#### 5.3.1 `SplitParams`: everything tunable is data

Every input to the split and to the CPU wait is a field of one `#[repr(C)]`
struct. It has a CUDA twin, and a `static_assert` plus a Rust layout test pin
the offsets byte for byte.

- **How it reaches the kernels.** The forward thread passes it **by value**,
  under 4 KB of kernel parameters, to bucketize and the down launch at every
  invocation.
  - Decode graphs rewrite a changed parameter in place at each wave's
    recapture (`decode_graphs.md:124-134`), the same way they rewrite tier
    addresses and the dispatch sequence number.
  - A value written between waves therefore takes effect on the next wave,
    with no rebuild, no restart and no extra kernel.
  - The kernel never reads it from host memory, so no PCIe read lands on the
    per-layer path.
- **The authoritative copy** is host memory. Writers store into it and the
  forward thread reads it at each invocation. A half-updated copy is still
  valid, because every field is independently meaningful.

**Measured fields** (f32 ns, `cpu_pool/params.rs` `Measured`): calibration
fills them at load (`cpu_pool/calibrate.rs`), and the online loop (§5.3.2)
refines the CPU ones.

| Field | Meaning |
|---|---|
| `copy_ns` | g, from the link rate measured at load |
| `cpu_read_alone_ns` | S / B_d |
| `cpu_read_shared_ns` | S / B_c |
| `cpu_n_ns[10]` | c(n) at n ∈ {1, 2, 3, 4, 6, 8, 12, 16, 24, 32}; the device table holds every n to 32, interpolated, and is linear beyond. Timed on experts read from DRAM, as a layer's are: the timed layers rotate through 256 MiB of synthetic experts, past any last-level cache here. Timed over one set of eight (16 MB, inside the laptop's 24 MB L3) c(1) came out at 48 µs against the ~90 µs the pool then took per expert in decode; from DRAM it measures 69 µs plus 56 µs per layer, which predicts decode's layers within 10% |
| `gpu_n_ns[10]` | c_g(n), same buckets |
| `handoff_ns` | t_h |
| `row_ns` | ρ: one assignment's row out and result back |

**Knobs** (`Knobs`). Defaults are neutral, meaning "trust the measurements".
Each exists because the model leaves out something the measurement can't
capture.

| Knob | Default | Effect |
|---|---|---|
| `cpu_scale` | 1.0 | Multiplies every CPU cost. Above 1 biases to the GPU. |
| `gpu_scale` | 1.0 | Multiplies `copy_ns`. |
| `handoff_extra_ns` | 0 | Added to t_h. |
| `cut_scale` | 1.0 | Step 1's cut: GPU when c(n) > `cut_scale` · (g + c_g(n)), for n ≥ 2 — a lone token always passes and the layer model decides it (`SplitParams::split`). |
| `min_gain_ns` | 0 | Hysteresis against near-ties. |
| `max_cpu_assignments` | 1,024 | The cap, and the rings' rows at load. **0 sends everything to the GPU.** A deployment value, not a feature flag: the same arithmetic with a cap of zero. A live raise is held to the rings' rows. |
| `wait_tight_ns` | 200,000 | How long the down launch's CPU item polls tightly before it backs off (§5.5). |
| `explore_period` | 3,072 | Invocations without CPU work before one explores (§5.3.2). 0 never explores. |
| `explore_ns` | 20,000 | How much an exploring invocation's single CPU expert may lose by. |
| `online_tau_ms` | 500 | The online loop's time constant. |
| `cold_bias` | 1.0 | Residency, not the split: the weight of the pack read in a pack-only expert's refill cost (§5.11). Applies whether or not the split runs. |
| `promote` | `cold` | Which experts the pool served the next launch copies into VRAM behind it: `none`, `cold`, `all` (§5.11). |

**Writers:**
1. calibration at load;
2. the daemon config: an `expert_split:` section under `host:` in
   `motherd.yaml`, and the same settings as `--expert-split*` flags on `zend`
   and `npcd` (`host/src/expert_split.rs`). Deployment config, not an env flag:
   `enabled` decides whether the pool starts with the engine. It is on by
   default (2026-10-18); `enabled: false` or `--expert-split false` keeps every
   expert on the GPU;
3. `POST /v1/expert_split` (admin role) for live tuning. Its body names any
   subset of the knobs; an unknown knob refuses the whole patch. It returns the
   knobs, the measured costs, the cut and the layers served;
4. the online loop;
5. tests and benches.

#### 5.3.2 The online loop

`cpu_pool/online.rs`, run by the service thread after each layer's `done` is
published, so nothing on the device waits on it.

**Sources.** Both exist in the production build, not only the profile build:

| What | Measured by | Gives |
|---|---|---|
| Pool time | The service, doorbell seen → `done` published, per layer | the observed cost of exactly the experts the device handed over |
| Device lateness | The down launch's CPU items: each that waited raises the mapped `wait` word to its wait (`atomicMax`), and the service reads and clears it at the next doorbell | how long the GPU sat on the pool |

The two clocks are never compared: each source is a duration in its own
domain.

**Correction.** Each layer's pool time is set against what the measured tables
predict for the same experts: the longer of k · S/B_d and Σ c(n_i). The ratio,
clamped to [¼, 4] so a preemption or a page fault is not taken for the pool's
speed, moves every CPU cost (`cpu_n_ns`, both read rates) through an EWMA whose
weight follows wall time, `1 − e^(−Δt/τ)` with τ = `online_tau_ms`. A thermal
swing therefore reaches the split within a few waves without the loop chasing
one layer's noise. The ratio scales the whole CPU table, not one bucket: a
layer mixes counts, and throttling and contention slow every count alike.
Because the pool time includes the service's own overhead (reading the ring,
writing results), the correction prices that overhead into the CPU side too,
which is where the device's decision needs it.

**No absorbing state.** When the pool has served nothing for `explore_period`
invocations, the next invocation carries `explore_ns` in its split. If that
invocation's split takes nothing, but giving the CPU the single cheapest
eligible expert loses to all-GPU by at most `explore_ns`, it takes that one
expert (`moe_bucketize_cpu.cuh`). That refreshes the correction. It costs at
most one expert's mis-placement per period, and keeps a throttle spell from
locking the CPU out for good.

**Visibility.** A device wait past 50 µs on 64 consecutive served layers is
logged once per streak with its likelier cause:
- the pool running slower than its tables (its mean ratio over the streak
  above 1.25);
- otherwise the handoff.

A CPU path that is losing time is never silent. A layer's wait on a cold
expert's pack read is the drive's, not the pool's: the service takes it out of
both the pool's time and the device's wait before the loop sees them, and
counts it apart (`stats`, `cold_wait`).

### 5.4 Cold experts

A cold expert is read by the stager into a pad slot either way: the CPU class
keeps summary bit 30, and its snapshot entries are 0. When the split gives it
to the CPU (`cpu_pool/cold.rs`):
- the pool waits on the expert's live gate entry — the word a GPU worker would
  wait on, which the stager stores last, after up and down — in `umwait` on its
  line, which the store wakes;
- it computes from the pad, so the expert never crosses the link.

The pool runs the experts already in memory first, then each cold one as its
publish lands. The pad slot cannot move while the pool reads it: the
invocation is not done until its down launch has the pool's results, and the
stager's demand rule evicts only slots the current invocation does not route.

**A pageable copy is computed at once.** An expert the warm tier holds in
pageable memory is cold to the device, which cannot read that memory, and the
stager stages it into the pad with a `memcpy` for the GPU. The pool can read
it, though, so it does not wait for that copy: it computes from the pageable
slot (`PagedImages`), whose image has the pad slot's layout byte for byte. The
copies are written once at startup and never move, and the cache holds the
tier for as long as the pool runs. Bucketize's cold-first order hands these
to the pool before any pinned miss, so the pageable tier is what runs on the
CPU up to the cap. Past the cap, those experts stream to the GPU through the
pad. A machine whose RAM the pinned part fills (the 16 GB laptop: 0 pageable
slots) never takes this path.

**Only unmarked.** A cold expert with a promotion in flight stays on the GPU:
the promotion lands by publishing a VRAM address into the entry the pool
would be waiting on, which the GPU's wait reads and the pool cannot. The
pipeline thread publishes a landing's VRAM entry before it clears the mark,
so an unmarked cold expert never sees one.

**Giving up.** A wait ends without the expert on the abort word, another
wait's fault, or past the device's spin limit; an expert published where the
pool cannot read is given up at once. The pool claims the fault word naming
the expert, exactly as a GPU worker would, and publishes no `done`: the down
launch's wait on the pool sees the word and gives up, and the host fails the
forward (§5.13).

**The device's cold waits are measured** (`cold_wait`): the worker holding a
cold expert's first row tile in the gate launch adds its wait for the publish
to a mapped `{ns, count}` pair, one sample per cold expert per layer — on the
laptop all-GPU, ~25–30 a decode step at ~1 ms each.

### 5.5 The device side of the wait

**The wait is a worker item of the down launch** (`kernel.cuh`, the live
worker loop). The down launch's `MoeLive` carries the pool's `done` word, the
sequence it must read, the result ring, `dst_row`, the liveness word and
`cpu_tight_ns`. Its worker queue is:

```
[ remote-expert items | ceil(CPU rows / 16) CPU items | read-ahead pieces ]
```

so a worker reaches a CPU item only once every remote expert's copy has been
claimed, and waiting on the pool never delays a GPU expert. The worker that
claims a CPU item:

1. **Polls `done`** (`moe_live_cpu_wait`). It polls tightly for
   `cpu_tight_ns` (the `wait_tight_ns` knob), then backs off. The cold wait's
   back-off is tuned for an NVMe read and would add tens of microseconds to a
   CPU result ready in 30.
   - It honours `SPIN_LIMIT_NS`, the abort word, the fault word and the pool
     liveness word. A wait that ends without results claims the fault word
     naming expert `0xffff` (§5.13).
2. **Pulls its 16 result rows** from the mapped result ring into `down_out` at
   `dst_row[r]`, with the wide 16-byte load loop the workers use for weight
   slices: many loads in flight, so the pull is bandwidth-bound, not a chain of
   round trips. The loads are `ld.global.cv`, not the weight copy's `.cg`.
   The host rewrites the ring at the same addresses every layer, and
   consecutive layers' down launches run back to back in one captured graph
   with nothing between them to invalidate L2, so a `.cg` pull can return the
   previous layer's rows.
3. **Exits.** The down launch completes in stream order.

This is the existing wait shape: a worker spinning on a host store. No block
waits on another block, so there is no co-residency assumption at any grid
size.

**Nothing else changes.** CPU-class experts get `n_tiles = 0` in bucketize, so
gate, up and down emit no tile for them; their grouped rows in `down_out` are
written only by the CPU items. The scatter sums every assignment's row from
`down_out`, weighted, in assignment order, as it always has.
- The rows of CPU experts in `gate_out` and `up_out` are never written. Nothing
  reads them, but a `tensor-assert` check over the whole of either buffer sees
  stale bytes there; the checks on those operands cover GPU rows only.
- The CPU item writes F32 rows, which is what the grouped int8 down emits; the
  dispatch asserts the output's type (`expect_dtype`) whenever the layer has a
  CPU class.

### 5.6 The pool

`expert_lre/cpu_pool/` holds one concern per file: pool, kernel, rings,
params, timing.

**Cores.**
- **Workers sit on physical cores, one each, never on SMT siblings.** They take
  P-cores first, then E-cores, because the laptop's E-cores add ~12 GB/s of
  read bandwidth over its six P-cores (§3.2). They are pinned with
  `SetThreadSelectedCpuSets`. `windows-sys`'s `Win32_System_Threading` is
  already enabled (`candle-core/Cargo.toml:59-73`). Elsewhere it uses the
  affinity mask.
- **Reserved cores.** The pool takes the remainder after reserving cores for
  the threads that must not be starved:
  - the forward thread;
  - the pipeline thread;
  - the stager readers (`NVME_QD = 8`);
  - the persistence thread (hot→warm copies, compaction);
  - HTTP, ingest and calibration;
  - in motherd, the agent's build, git and mind jobs, which run in the same
    process at low priority (`motherd.yaml:23-24`).
- **Starting sizes.** The size is derived from the topology at load, and the
  §5.10 core sweep replaces these starting points:

  | Machine | Starting pool |
  |---|---|
  | Laptop | 4 of 6 P-cores |
  | 3090 box | 6 of 8 cores |
  | PRO 5000 | 12 of 16 cores |
  | 7970X | 28 of 32 cores |

**Stragglers.** CPU sets are *soft* affinity, and the layer waits for its last
down task, so one preempted worker or a DPC costs the whole layer, ×48 per
token.
- Tasks are fine-grained, a few rows each.
- An idle worker takes a late worker's unclaimed rows.
- Measurements report p99, not only p50 (§8).

**Participants.** A run takes every performance core of the pool
(`layer::participants_for`, `Pool::fast`) — the caller and the workers before
the first slower core — whatever its expert count. The matvec is
compute-bound on a core (§5.10: Q3 one token unpacks at ~16 GB/s, against
~22 one core reads), so each performance core adds to a layer's rate until
together they meet DRAM's: one DRAM-cold one-token expert on the laptop takes
~62 µs on four, ~60 on five, and five beat four at every shape measured. A
cold expert the pool waited on runs alone on the critical path, which is
where this matters most. No run takes a slower core, so inference never
runs on an E-core, whatever the split's cap (`CpuSplit::max_experts`, §5.3);
the E-core workers take only calibration's jobs. The workers a run does not
take are not woken. Inside a run, a participant with nothing to claim waits
in `tpause` (C0.1), not a `pause` loop.

**Warm through inference, parked outside it.** The forward calls
`CpuService::wake` ahead of every invocation, and each call holds the pool
warm for a 100 ms cool-down (`service::COOL_DOWN`). Inside that window the
service thread waits on the doorbell, and the performance-core workers on the
job word, in `umwait` (C0.1, woken by the store at no latency cost), and never
park; the first call after an idle spell unparks them, ahead of the doorbell
that follows. Once inference has been quiet for the cool-down, the service
waits 5 ms after a layer and a worker 200 µs after a job (`attach::SPIN`,
`WORKER_SPIN`), then park. A parked worker costs 50–100 µs to wake on
Windows; fourteen workers kept waiting held the laptop's CPU at ~75% of base
clock, which is why only the performance cores stay warm.

`has_waitpkg` is asked once: every wait loop asks it each pass, and `cpuid`
serializes the core and, under Windows' virtualization-based security, traps
to the hypervisor.

**Schedule.** Dataflow per expert, with no pool-wide barrier:
1. **Gate + up.** All CPU experts' gate and up rows are split into tasks and
   claimed from one atomic counter. Each task covers rows of one expert × all
   of that expert's tokens, reading each weight byte once for all of them.
2. **The last gate/up task of an expert** applies SwiGLU and emits q8a128 for
   that expert (§6). Each expert has its own countdown.
3. **Down.** That expert's down rows become claimable at once. Down tasks are
   preferred over gate/up tasks, so finished experts drain first.
4. **Reduce.** A down task owns one expert × one output row range. It writes
   that expert's unweighted rows into per-assignment scratch, which stays in L2
   (10 KB per assignment), and decrements its row range's countdown.
   - The task that takes a row range's countdown to zero sums weight × scratch
     over each row's experts into the row's result.
   - Each result element therefore has exactly one writer, with no atomics.
   - The service hands `run_layer` one row per *assignment* (§5.1), each with
     one expert at weight 1, so the reduce is the identity: each result row is
     one assignment's unweighted down output, which the device places and the
     scatter weights.
5. **After the last reduction** the service copies the results into the mapped
   result ring and stores `done` after a release fence.

**FP environment.** The workers run with MXCSR FTZ | DAZ: denormals are
flushed, the same as the device's fast-math code.

### 5.7 Pageable warm slots

The pool reads host RAM directly. A warm slot only the CPU computes therefore
needs no pinning, no mapping and no share of the WDDM non-local budget.

- **Today** pageable warm slots are treated as cold: a pageable expert is
  published to the live table as 0 (`live_table.rs:9`,
  `moe_live_dispatch_design.md:134-137`), and a hit costs a `memcpy` into the
  pad (`warm_tier.rs:20-24`).
- **The GPU planes stay as they are.** A pageable address is never written into
  a live-table plane the GPU reads. Every GPU reader treats a non-zero entry as
  device-readable: bucketize's classifier (`pinned.rs:611-616`) and the cold
  workers (`kernel.cuh:1750-1753`). A host address there would be dereferenced
  by the GPU and lose the context.
- **A separate host-only plane** in the work ring's metadata holds pageable
  addresses by (layer, expert). Bucketize reads only a *bit* from a
  device-visible bitmap ("RAM-backed, pageable"). It may put such an expert in
  the CPU class, or leave it cold for the stager. The address is resolved by
  the pool, on the host.

The warm tier can then grow to whatever RAM is free, past the pinned budget.
On the laptop that is the gain this design can offer against the capacity gap
of §1.1. On the 3090 box it decides how much of the 50.1 GB expert set is RAM
at all: the pinned part is capped by WDDM (`warm_tier.rs:45-56`), and that size
is measured, not assumed (§8).

### 5.8 The VRAM partition

The elastic boundary's concession cost prices an evicted expert at its PCIe
refill. With a CPU class, that refill is:

```
min(copy_ns + gpu_n_ns[1],  handoff_ns + cpu_read_shared_ns + cpu_n_ns[1])
```

It comes from the same `SplitParams`. KV wins more of the span on machines with
a fast pool, so CPU compute buys more sessions as well as faster ones. The
cold-backed case is §5.11's.

### 5.9 The handoff, made fast

The residual chain is serial, so whatever the CPU's part of a layer doesn't
overlap with GPU work is on the critical path. These are the legs, in order.
Each target is an estimate until measurement 0 replaces it.

| # | Leg | How it is kept short | Target |
|---|---|---|---:|
| 1 | Rows reach host RAM | Decode: written by `hc_mix`'s epilogue before routing exists, so off the path. Prefill: a wide export kernel, charged to the CPU side (§5.3). | 0 at decode |
| 2 | Work reaches host RAM | Bucketize writes the slot, then `__threadfence_system()`, then the doorbell. These are posted writes. | 2–3 and 5 together: **2.2 µs p50, 2.8 µs p99, measured** (below) |
| 3 | The pool notices | Monitor-wait on the doorbell line | (in the above) |
| 4 | Gate/up → down | Per-expert countdown (§5.6) | ~0 |
| 5 | `done` reaches the GPU | Release store; the waiter polls tightly until the predicted finish | (in the above) |
| 6 | Results reach VRAM | The bulk pull in the down launch (§5.5) | y·assignments / B_l + one round trip |

**Rows before routing (decode).**
- `hc_mix_with_operand`'s epilogue already writes the q8a128 operand
  (`wave.rs:4162`). At decode widths it also stores the rows into the row-ring
  slot the forward thread reserved for this invocation.
- **Where it applies.** It applies wherever the operand is emitted, including
  the MTP head's FFN (`draft.rs:527`).
- **Where the operand comes from `to_dynamic`** (`wave.rs:4195-4204`,
  `h2_q8 = None`), there is no epilogue to extend. A small row-export kernel
  runs there instead.
- **Cost.** These writes happen on every decode layer, whether or not the split
  uses the CPU: about 2.9 KB of posted writes per token per layer.
- **If they slow the kernel.** Measurement 0 times them, including any
  store-drain delay before `hc_mix` can retire. If they cost more than they
  save, the rows move to an export kernel after bucketize that exits at once
  when `n_cpu = 0`.

**Rows at prefill.** A prefill wave's operand is up to 8,192 rows, about 23 MB
a layer. Sending all of it would cost ~2 ms per layer on a 12 GB/s link. A
wide export kernel after bucketize sends only the rows of tokens routed to a
CPU expert. Its time is in the split's T_cpu and T_copy (§5.3), so the CPU is
given prefill work only where the export pays for itself.

**Monitor-wait, not spin and not park.**
- Parking costs tens of µs to wake on Windows.
- Spinning wastes power, and on the laptop that power comes out of the GPU's
  shared budget.
- **Where it works.** The pool waits on the doorbell line with
  **UMONITOR/UMWAIT** (WAITPKG; the laptop) or **MONITORX/MWAITX** (the 9950X3D
  and 7970X).
  - The OS caps UMWAIT's wait time (`IA32_UMWAIT_CONTROL`), so the pool
    re-arms it in a loop.
  - Whether a device's PCIe write to the line wakes the wait is measured per
    part (measurement 0). Where it doesn't, the pool spins.
- **Where it doesn't.** The 10700K has neither, so it spins on `_mm_pause`.
- **Between steps** the pool parks. The forward thread writes a step-start word
  before it launches a step, so the wake overlaps the first layers' attention.

**Layers with nothing for the CPU.** Bucketize writes no work slot and rings no
doorbell, so the down launch has no CPU item and the pool sees nothing. The
only cost is leg 1's decode row writes.

**At ×N: half-waves.** A ×N decode wave can be split into two halves that
alternate: A's CPU experts run under B's attention and GPU experts, then they
swap. This hides the pool behind real GPU work. The cost is a second set of
narrower launches per layer.
- Strata measured the single-stream form as 7% slower, so it is not used at ×1.
- At ×N it is kept only where A/B binaries show it wins (measurement 5).

**Measured: the handoff round trip** (measurement 0, laptop, 2026-10-16;
`cpu_pool::handoff`). A full device→host→device round trip through mapped
words, 5,000 iterations:

| Device waits by | Host waits by | p50 | p90 | p99 | min |
|---|---|---:|---:|---:|---:|
| Kernel poll (`handoff_ping`) | spin | **2.20 µs** | 2.40 | 2.80 | 1.10 |
| Kernel poll | UMWAIT, re-armed every ~10 µs | **2.20 µs** | 2.30 | 2.70 | 1.30 |
| Kernel poll | UMWAIT, re-armed every ~1 ms | **2.20 µs** | 2.30 | 2.70 | 1.30 |
| `cuStreamWaitValue32` | any | 130 µs | 148 | 237–250 | 31 |

- **The kernel poll is the device's wait.** Its 2.2 µs round trip covers legs
  2, 3 and 5, against the ≤ 3.5 µs the table above budgeted.
- **`cuStreamWaitValue32` is ruled out under WDDM.** It waits ~60× longer,
  because the front end polls host memory slowly.
- **A device write over PCIe wakes UMWAIT on Meteor Lake**, at no latency cost
  against spinning. With a 1 ms re-arm deadline, the round trip stays 2.2 µs.
  The pool therefore monitor-waits on WAITPKG parts instead of spinning.
- **Still to measure:** MONITORX/MWAITX on the AMD machines, and the 10700K
  (spin only), on those machines.

**Not available.** gdrcopy, a CPU writing straight into VRAM over BAR1, needs a
Linux kernel module, and all three dev machines run Windows.

### 5.10 The CPU kernel: optimisation runs

The split is only as good as the kernel it prices. A kernel at 40% of DRAM
bandwidth hands the CPU half the misses it could take. Getting the quantised
matmul as fast as the hardware allows is a deliverable of its own, run as a
measured campaign. A change stays only if a measurement says so.

**Nothing to inherit from candle's CPU side** (inventory, 2026-10-16):
- `k_quants::matmul` quantises and walks one token at a time, re-reading the
  whole weight matrix per token (`k_quants.rs:6082-6129`), on the global rayon
  pool.
- SIMD is chosen at compile time (`cpu/mod.rs:1-9`, `utils.rs:31-33`) under
  `target-cpu=native`.
- gemm's AVX-512 path needs its `nightly` feature, which is off.
- There is no AVX-512, VNNI, runtime dispatch, pinning, monitor-wait or MXCSR
  code anywhere.
- Q3_KO has no CPU form (`quantized/mod.rs:1333-1341`).

**What carries over is specification:**
- the KO plane layout (`ko_quant.rs:306-389`, `dequant_ko` at `:641-699`);
- the affine fold's arithmetic (`ko_affine_fold`, `kernel.cuh:2173-2183`): the
  scale and min terms and the sum convention, though not its rounding order
  (§6);
- the q8a128 byte layout (`cuda_tests.rs:2875-2960`) and the Σx conventions
  (`SumScale::Raw` for the SwiGLU operand, `dispatch.rs:1063-1069`);
- the u8×s8 `maddubs` idioms in `quantized/avx.rs:21-44`;
- crossbeam's `CachePadded` for ring words;
- `raw-cpuid` (already in the lockfile) for WAITPKG, MONITORX and topology.

**The bench.** `cpu_expert_kernel_bench` (`#[ignore]`):
- runs one Flash-Next expert from a real pack slot, at every `cpu_n_ns` bucket,
  on the production pool layout;
- reports time, **GB/s against measured streaming read bandwidth**
  (measurement 1), and int8 MAC/s against cores × ISA peak;
- checks the §6 per-expert accuracy gate in the same run. A faster kernel
  that fails the gate is not faster.

Its output is the `cpu_n_ns` table, and calibration at load runs a short form
of it.

**Targets:**
- ≥ 80% of measured streaming bandwidth while memory-bound. KTransformers
  sustains ~72% of DRAM peak, and Strata 44 of ~60 GB/s.
- ≥ 50% of int8 MAC peak while compute-bound.

**Levers, each measured on its own:**

1. **Unpack once, dot many.** Q3_KO's crumb + hi planes unpack to u8 0..7,
   which is `vpdpbusd`'s unsigned operand (AVX-VNNI, AVX-512 VNNI) and
   `vpmaddubsw`'s on plain AVX2, against s8 activations. One unpacked weight
   vector is dotted against every token held in registers (up to 8), which
   makes c(n) sub-linear. VBMI's `vpmultishiftqb` unpacks crumbs in one
   instruction.
2. **Permute the activation, not the weights.** The KO planes are lane-major
   and permuted. The int32 sum over a 128-K block doesn't depend on order, so
   each token's 128-byte activation block is permuted once per block into the
   weights' byte order, and the weights stream untouched.
3. **Fold 8 rows in one register.** A chunk is 8 output rows, so the
   per-block float fold for 8 rows is one ymm, or two chunks per zmm on
   AVX-512, with FMAs placed wherever they are fastest (§6).
4. **Fused gate + up.** One walk, one activation read.
5. **Task shape.** Rows of one expert across workers, or whole experts per
   worker, chosen by count. Three experts on 4 workers must split rows; twenty
   must not.
6. **Prefetch distance.** Swept; the hardware streamer may already saturate.
7. **Pages.** A 2 MB expert spans 488 4 KB pages. Large pages apply to pageable
   slots (`SeLockMemoryPrivilege` on Windows); `cuMemAllocHost` doesn't take
   them. They stay only if the TLB cost measures.
8. **Which cores.** P only, P+E, and SMT on or off, swept per machine. This
   replaces §5.6's starting sizes.
9. **ISA tiers.** AVX2 (10700K), AVX-VNNI ymm (laptop), AVX-512 VNNI zmm
   (9950X3D, 7970X; `kernel::simd512`, built 2026-10-17).
   - Detection takes AVX-512 VNNI first. A Zen 4 part reports it without
     the VEX `avxvnni` flag, so the 256-bit tiers alone would leave it on
     AVX2.
   - Every tier runs one loop (`simd`'s `matvec_tier`), so the tiers this
     machine has test all of it but the 512-bit unpack, load and dot. The
     512-bit unpack is also checked on any CPU by a model of its steps on
     plain bytes, with the real tables
     (`the_modelled_512_unpack_is_the_scalar_unpack`). On an AVX-512
     machine, the instruction-level unpack test and the byte-equality tests
     of every tier run; elsewhere they report themselves skipped or pass
     over the tier.
   - Its unpack is ~11 vector ops a row against ~25: one `vpsrlvw` makes two
     bit pairs, and one `vptestmb` with a masked add makes the hi bit.
     Unmeasured until it runs on an AMD box (`kernel_profile`).
   - They are dispatched at runtime with `is_x86_feature_detected!` +
     `#[target_feature(enable)]` (stable on rustc 1.97), so a test can force
     every tier on one machine.
   - `target-cpu=native` stays for the rest of the build.
10. **Down input stays hot.** The 640-wide SwiGLU output (5 q8a128 blocks)
    stays in L1/L2 if the expert's down rows run on the cores that produced it.

**Method** (per the memory notes on laptop throttling and GPU clock caps):
- alternating A/B binaries, never back-to-back runs;
- clocks sampled inside the run;
- on the laptop, with the GPU loaded as in production;
- results per machine and ISA tier in `docs/performance.md`.

**First runs, laptop** (2026-10-16, AVX-VNNI, GPU idle; `cpu_pool::kernel_bench`).
Built so far:
- `cpu_pool::kernel` (scalar reference + AVX2 + AVX-VNNI, byte-equal to each
  other);
- `cpu_pool::pool` (pinned workers; UMWAIT spin, then park);
- `cpu_pool::layer` (the §5.6 dataflow schedule).

Q3_KO experts at Flash-Next geometry. The pool is the 5 non-reserved P-cores
plus the 8 E-cores on their cache (`topology::pool_cores`). The caller, standing
in for the forward thread, is on the reserved P-core.

| Experts × tokens | µs / layer | µs / expert | Weight GB/s | GMAC/s |
|---|---:|---:|---:|---:|
| 1 × 1 | 57–72 (p50 ~63) | ~63 | ~32 | ~78 |
| 5 × 1 | 191 | 38 | 52 | 129 |
| 10 × 1 | 366 | 37 | 55 | 134 |
| 5 × 4 | 434 | 87 | 23 | 227 |
| 1 × 32 | 840 | 840 | 2.4 | 187 |

Two optimisation levers are in:
- **Eight-row chunk blocking.** A chunk's rows are unpacked once into L1, each
  token's eight dots reduce in one transpose, and the fold is vectorised over
  rows.
- **Task sizes scaled to experts × participants.** Together with the blocking,
  this took 10 × 1 from 46 to 37 µs per expert and lifted multi-token
  throughput ~1.5×.

Against the targets:
- **Memory-bound:** 55 GB/s is ~70% of the ~80 GB/s this core set reads, short
  of 80%.
- **Compute-bound:** ~230 GMAC/s is well short of 50% of peak.

The remaining levers are 2-token register blocking, fused gate + up, prefetch
distance and pages, and AVX-512 on the AMD machines. Even now, a decode-shaped
expert costs 37–63 µs on the CPU against ~166 µs to copy it over this link.

**Second round, laptop** (2026-10-17, AVX-VNNI, the production pool: five
P-cores, the caller on the first, held warm). The harness is
`cpu_pool::kernel_profile` (`#[ignore]`, ~20 s, `cargo test --release -p
candle-transformers --features cuda --lib cpu_kernel_profile -- --ignored
--nocapture`). Run it on each machine; it prints:
- the DRAM roof of one core and of the performance cores;
- the hot (L2) matvec in **cycles per chunk** at a measured clock, so
  throttling cancels, per ISA, format and token count;
- one core DRAM-cold;
- the serial per-expert steps;
- decode and prefill shapes on the pool, DRAM-cold and L3-hot, with the
  pool's own join and busy share.

The laptop's thermal state moves the roof 41–69 GB/s between runs, so pool
rows compare as **% of roof**.

What the first profile showed:
- **The kernel is compute-bound.** One core runs the hot matvec at 12 GB/s
  against the 23 GB/s it reads.
- **The pool reached only 31–47% of roof.**
- **A row group's chunks sit 33–133 KB apart**, so the hardware streamers never
  saw them coming.

Kept:

| Change | Effect |
|---|---|
| One-token path dots each row from its unpack, the 8 rows' sums in a register across K blocks; raw chunk pointers | hot 12.1 → 14.5 GB/s a core |
| Software prefetch 4 chunks ahead **in the loop's own order**, carrying into the next row group, the run's first chunks in flight at once | cold 1×1 90 → 60 µs; 5×1 339 → 198 µs |
| Q3 hi-bit unpack: `min_epu8` against 4 replaces cmpeq + and | 3 of ~25 ops a row |
| SwiGLU applied by each gate/up task to its own rows | serial step gone |
| Activation permute and the down input's quantize in AVX2 | serial per expert 3.1 → 1.1 µs (1 token), 37 → 12 µs (16) |
| Every layer takes all performance cores (`participants_for`) | five beat four at every shape |
| Each projection's groups split into near-equal tasks (`task_range`) | busy 68–91% → 86–93% |

Rejected, measured:
- **Prefetch distance 2 or 8:** both worse than 4.
- **Six tasks per participant:** worse at 4 tokens.
- **Prefetching the down weights during gate/up:** worse everywhere. The
  burst fills the line-fill buffers ahead of the demand loads.
- **Staging decode's predictions down to a 25% cell probability** (listing
  kept at 50%), to start the CPU's cold reads a layer early: pack reads
  rose ~47 → ~75 a step for ~6 more staged hits. The cold experts the pool
  waits on stayed at 0.4–0.6 a layer and decode was unchanged within noise
  (split 19.4–20.5 vs 19.3–21.1 t/s). Those cold misses are the long tail of
  rarely routed experts, and their rarity is both why they are cold and why
  routing history cannot predict them.

All kept changes are byte-equal to the scalar reference: same int sums, and
the fold in the same order.

| Pool, DRAM-cold, 5 P-cores | before | after |
|---|---:|---:|
| 1 × 1 | 92 µs (33% of roof) | 54–61 µs (51–56%) |
| 3 × 1 | 221 µs | 125–135 µs (69–71%) |
| 5 × 1 | 320 µs (47%) | 196–210 µs (74–76%) |
| 5 × 4 | 640 µs | 351–550 µs |
| 1 × 1, L3-hot | 54 µs | 36–39 µs |

Remaining bounds:
- **Q3 compute.** Q3 at one token is ~105–115 cycles per chunk at 4.1 GHz,
  ~11 a row, almost all of it the 3-bit unpack (Q4 is 56–65). Each extra token
  costs ~29 cycles, the 32 `vpdpbusd` a chunk needs.
- **The single-miss barrier.** A single cold miss still waits on the gate/up →
  down barrier (~11% idle). A per-tile dataflow, with down starting on each
  128-row tile of the SwiGLU output as it completes, is the next lever.

In the engine (`tests/cpu_expert_speed.rs`, the split with promote none):
- decode 15.1–15.4 → **17.8–18.4 t/s**;
- the CPU's experts per decode layer 3.5 → 4.3, against a cap of 5;
- pool compute per layer 503 → 260–290 µs;
- the last worker's join 23–40 → 3–4 µs.

Calibration prices one expert at 42 µs (was 79), so the split then ran
against `max_experts`, one a core. Raised to two a core (§5.3), the price
settles at ~5.2 experts a layer and split decode reaches 19.4 t/s.

**Tests.** Byte-level checks run fast in `cargo test`:
- `simd::tiers_match_scalar_bytes`;
- `a_nan_weight_scale_poisons_only_its_row`, which checks a NaN's row is NaN
  and every other row is bit-equal;
- `permute_tile_is_the_scalar_permutation`;
- `act::tile_codes_are_the_scalar_definition`, covering halves, NaN, ±∞, −0
  and a dead tile;
- `prefetch_lookahead_carries_into_the_next_group`;
- the task split;
- `layer::pooled_layer_is_the_per_expert_sum`.

The measurements are the `#[ignore]` benches.

### 5.11 Residency: priced by what a miss costs

The CPU path changes what a miss costs, and the cost now depends on what the
expert falls back to:

| A non-resident expert is served from | Cost after this design |
|---|---|
| RAM (pinned or pageable warm) | min(g + c_g(n), t_h + p + c(n)): cheap, and cheaper still where the CPU takes it |
| NVMe only | `cold_read_ns` (~0.5–1 ms for 1.9 MiB on the laptop) + the above |

A warm miss shrinks by up to B_d/B_l and a cold one does not. **The gap widens,
and residency must price it.**

**Eviction.** The ranking already orders its passes by eviction class (whether
the row routed the victim on its last visit, and whether a warm copy backs it),
each class costed at its measured regret rate × what restoring one costs
(`regret::eviction_costs`, set per pass by `pipeline.rs`'s
`set_eviction_order`). The CPU class changes the restore costs, which
`SplitParams::refill_secs` now supplies:

```
warm refill = copy                                                  split off
            = min(copy, handoff + cpu_read_shared + c(1) − c_g(1))  split on
pack refill = cold_bias · pack_read + copy
```

- A resident expert pays its GEMM too, so a refill is priced by what it adds.
  The pool's path replaces the GPU's GEMM, which is why c_g(1) comes off.
- A pack refill's read lands ahead of whichever side computes the expert, the
  GPU's workers or the pool (§5.4), so it is priced the same on both.
- Between two equally hot experts, the one with a RAM copy is evicted first,
  and more so as the CPU makes warm misses cheaper.
- `cold_bias` (default 1.0, on the config section and the admin route) is the
  tuning knob. On the laptop it is the most important one in this design.

**Promotion.** A decode miss's GPU copy also installs the expert into a
promotion slot: the workers store each slice into `remote_dst` as they copy it
(`kernel.cuh:1650-1653`, `dispatch.rs:16-23`). A CPU-served miss has no GPU
copy to ride on, so its own launch claims nothing for it.

**Promotion behind the pool** (`promote`: `none`, `cold`, `all`). The launch
that hands the pool its experts leaves the ones `promote` names in a device
**carry** (`moe_bucketize_cpu.cuh`, THE CARRY), and the next launch — any row
— promotes them: each takes an offer as a miss does, is logged with
`CARRY_FLAG` and marked, and its gate launch's workers copy its pinned image
into the slot and publish it, as a read-ahead item. By then the pool has the
expert's result and, for a cold one, the stager has published the pad slot
the copy reads. Nothing waits on the copy: it is off the path the pool's
result is on, and lands for the row's next visit. A carry is the next launch's
only — it names the writer's sequence word — and the stager pins the pad
slots of every CPU-served expert until that launch is done, since its demand
rule could otherwise overwrite one under the copy. The pipeline thread books a
carry claim as it books a read-ahead one (the device published it) but judges
it with the demand claims, and counts it apart (`carry_claims`).

`cold` was the first setting tried — a cold expert the pool served without
promoting comes back from the pack at its next miss — then `all`. Measured on
the laptop (§8), alternating in one run: `cold` 9.9 / 9.0 t/s, `none` 9.3 /
8.0, `all` 8.3 / 8.1 — `cold` ahead by 7–12%, `all` behind, its ~135 copies a
step costing more link than its promotions save. `cold` is the default. Every
split arm was measured on a CPU the pool's load throttled to ~75% of base
(§8), so the ranking stands and the absolute figures do not.

Without a carry, an expert the CPU served and that recurs is brought into VRAM
by prediction and read-ahead ahead of its next use. The gate launch's
read-ahead workers fill spare link time with plain byte copies into claimed
slots (`kernel.cuh:1654-1663`). Three things keep that path working:

1. **A CPU-served miss is still recorded as a miss.** Its summary word carries
   the CPU bit, and the pipeline thread scores it like any miss
   (`record_miss`). Eviction, the look-ahead votes and read-ahead claims all
   see it.
2. **Read-ahead's gate counts GPU-class copies only; its window counts the
   CPU's experts too.** The gate prices link time per invocation
   (`read_ahead_gate.rs:12-20, 39-44`, hysteresis `ON_US = 80` /
   `OFF_US = 40`). `link_us` counts only GPU-class copy time, so a CPU-heavy
   split keeps read-ahead *on* instead of switching it off. But each launch's
   budget — the layer's window less what it already spends — charges a CPU
   expert as a copy (`promotion_walk`). Uncharged, the window the split freed
   went to read-ahead copies of predicted experts, ~40% of them wrong, whose
   victims came back as pack reads: on the laptop (Flash-Next, 1K prompt,
   128 greedy steps) read-ahead claims went from 27 to 110 a step and decode
   from 13.0 to 11.2 t/s. Charged, claims stay near all-GPU's and decode
   rose to 12.3–12.6 t/s.
3. **Promotion is priced by backing.** Promoting an NVMe-only expert saves more
   than promoting an equally hot RAM-backed one, so bucketize's promotion walk
   offers slots to the routed **cold** experts first, then the pinned ones
   (`moe_bucketize_live.cuh`, `promotion_walk`). When offers run short they go
   where the next miss would wait on the pack. Promotion is tuned before
   eviction, which is the standing order for this cache.

Measurement 6 (§8) confirms the expectation: steady-state VRAM hit rate with
the split on versus off.

**The warm tier.** It is filled once at startup and never changes
(`pinned.rs:1-15`), and it now decides which misses are cheap.
- It should hold the most misses *not already absorbed by VRAM*. A copy of an
  expert that is nearly always resident buys little, and the next expert VRAM
  can't hold is a better use of the RAM.
- It ships with the better static choice. Whether it should adapt at run time
  is measured with the eviction counters' "routed again at next visit" share
  (`cache.rs:301-309`).

**Sweeps.** Run `cold_bias` on the laptop at ×1, ×8 and C5×8, alternating, as
the `FADE_FLOOR` sweep was. Record the pack-read share, the hit rate and t/s.

### 5.12 Two tenants, and two GPUs

**motherd's two tenants** share one engine, so one dispatch and one pool serve
both. Nothing changes.

**Several GPUs** (the planned workstation, 2× RTX 5090):
- Each device has its own dispatch, rings, doorbell, `SplitParams` and online
  loop.
- The pool is partitioned by core sets, one per device. On the 7970X that is
  CCD-aligned: 4 CCDs of 8 cores, two per device.
- **DRAM is one resource.** Both devices' pool partitions and both links' DMA
  read the same four channels, so B_d and B_c are measured **with both devices
  running**, not per device in isolation.
- The 7970X is one socket, so KTransformers' NUMA tensor parallelism doesn't
  apply. Whether per-CCD L3 locality matters is part of the core sweep (§5.10,
  lever 8).

### 5.13 Faults

- **Pool liveness.** The service sets the mapped `alive` word before it
  calibrates, and a drop guard clears it however the thread exits, panic
  included.
- **Waiter outcomes.** The down launch's CPU item checks liveness while it
  polls:
  - `done` lands: normal path.
  - `alive` falls, the abort word rises, another fault is claimed, or
    `SPIN_LIMIT_NS` passes: the worker claims the fault word with expert
    `0xffff` (`MOE_FAULT_CPU_EXPERT`), the row and the wait, and gives the item
    up. The error names the pool, not a cold expert (`fault.rs`).
  - The host fails the forward through `take_segment_fault`, so no result
    computed from the layer is returned. A cache with a CPU class always takes
    the check, even with every expert resident or pinned, because the pool can
    fault where no cold expert could.
- **No drain is needed.** There is one work slot, and the device waits for the
  pool's `done` before anything after the layer runs, so the pool never holds
  a slot the device has moved past. A pool still finishing a failed forward's
  layer writes results nobody reads, then serves the next doorbell; it reads
  the work entries once, at the start, and only from pinned memory that
  outlives it.
- **Persistent failure.** A pool that fails three forwards in a row
  (`CPU_FAULT_LIMIT`) has its measured costs withdrawn, so every later split
  is all-GPU, and the log says so. The split stops using it until a restart.

---

## 6. Accuracy: the device's arithmetic, but for one division

**Decision (author, 2026-10-16): CPU results are not required to be
byte-identical to the GPU's**, and the GPU kernels are not touched for the
CPU's sake: no change to `ko_affine_fold`, `fast_exp` or the sigmoid. The
author confirmed both on 2026-10-17, with the measurements below in hand.

**What was measured.** A CPU kernel free to fold, round and quantize its own
way, with a mathematically exact SiLU and f32 tile scales, matched each expert
to ~1e-3 relative and was far below the 3-bit weight noise (both sides 2.7e-2
from the f32 chain). It still moved the model a long way: one decode step's
CPU layers moved that position's logits by KL 3e-3, and a run with the split
on gave mean KL 3e-2 to 4e-2 and top-1 93–97% against all-GPU. This model
amplifies sub-noise differences in its expert outputs; matching the noise
level is not enough. So the CPU kernel reproduces the device's arithmetic:

- **The down projection's input is the device's tile**
  (`kernel/act.rs`, `ActTiles::quantize`): codes `rint(x · 127/amax)`, the
  scale as f16(amax/127), and Σx summed in the warp's order (each lane's four
  in turn, then the xor butterfly) and rounded to f16. These are the stored
  values the int8 GEMM reads.
- **The SiLU is the device's** (`kernel/expert.rs`, `device_silu`): the same
  cubic polynomial for 2^x in the same three FMAs, the same clamp and the same
  branch. Only the division differs. The device divides approximately
  (`rcp.approx`, and `e · rcp(1 + e)` for x < 0), and no CPU instruction
  reproduces that reciprocal, so here it is exact. Over 4.2M gate values, 7%
  of results sit an ulp from the device's, and every one of them is that
  division (`gpu_parity`).
- **The int8 dots and the per-128 fold** are exact integer arithmetic folded
  in f32; they agree bit for bit.

With that, the pool's experts are the GPU's bit for bit on the pack's own
experts at every count tested (1–9, 40, 130) and every tile mode. The one
exception is the rare SwiGLU value the device's division puts an ulp across an
int8 code boundary: one flipped code measured 2.3e-4 over 130 rows. End to
end:

| Arm (Flash-Next, laptop, 2,048-token prompt + 256 decode steps) | CPU rows | Mean KL | Top-1 |
|---|---|---|---|
| All-GPU again (control) | 0 | 0 | 100% |
| CPU in decode only | 755 | 1.3e-2 | 97.7% |
| CPU in prefill only | 20,267 | 2.5e-2 | 94.2% |
| CPU in both | 21,499 | 4.1e-2 | 92.2% |
| GPU's SiLU dividing exactly (`__fdiv_rn`), CPU in prefill | 21,059 | 0 (bit-exact) | 100% |

A run is bit-exact until the first flipped code; decode arms that served only
23–57 rows measured exactly 0, and one with 755 did not.

The last row is an experiment, not the build. It shows the flipped codes are
the whole prefill difference. Making the device divide exactly would make the
CPU path bit-identical end to end, but the author chose to leave the GPU
kernels as they are.

**What the CPU kernel must keep:**
- the same weights and the same activation bytes in (§5.2);
- the device's arithmetic, as above;
- finite outputs.

It is still free to order its work: tasks, threads and SIMD width do not
change a value, because every reduction it does is exact integer arithmetic
or the device's own order.

**The gates:**

1. **Per expert, against a common reference.** Both sides are measured against
   the f32 dequantised reference (`dequant_ko` over the f32 activation). The
   CPU's error, as relative L2, must be **no worse than 1.1 × the GPU's own
   error** against that reference (`gpu_parity`; it measures equal).
2. **Per expert, against the GPU, on the pack's own experts**: within the
   flipped-code bound above (`gpu_parity`, `#[ignore]`d — it reads the pack).
3. **End to end** (`tests/cpu_expert_gate.rs`): over a 2,048-token prompt and
   256 teacher-forced decode steps, with the split on in both phases, against
   all-GPU:
   - mean KL ≤ 0.1;
   - top-1 agreement ≥ 90%.

   These were set from the measured prefill flips with the GPU's kernels
   unchanged (the author's decision, 2026-10-17). A second all-GPU arm must be
   exactly 0, so whatever the CPU arm shows is the CPU path's.
3. **KV compression.** Re-run the C0–C10 gate once with the split forced
   all-CPU, to confirm the per-model `*_KV_FACTORS` thresholds still hold under
   CPU-computed experts.

**Codec pieces are exact.** `CLAUDE.md`'s rule for codec code applies to the
parts that are codecs, and those tests assert raw bytes:
- unpacking Q3_KO/Q4_KO codes;
- permuting the activation block;
- emitting q8a128 (the device's tile, including the f16 scale and the warp's
  Σx order).

**Reproducibility where it is needed.**
- **Replays.** Any run that needs exact replay sets `max_cpu_assignments = 0`
  (all-GPU). Placement then cannot vary, and output is exactly as
  reproducible as today. This covers every corruption hunt, every NaN capture,
  and every byte-equality test above a MoE layer.
- **Test engines.** Engines built by test fixtures default to a cap of 0. A
  test that exercises the CPU path opts in through the fixture's constructor.
  That is a parameter, not an env flag.
- **Logging.** The split's per-layer class counts are logged with the wave, so
  a production divergence can be attributed to placement after the fact.

---

## 7. Tests

Written alongside each piece. Codec pieces assert raw bytes. The float GEMV
is judged by the §6 accuracy gates.

| Test | Asserts |
|---|---|
| `q3_ko_cpu_unpack_bytes` / `q4_ko_cpu_unpack_bytes` | Each SIMD tier's code unpack equals `dequant_ko`'s codes, byte for byte |
| `cpu_activation_permute_bytes` | The permuted activation block equals the weight byte order, byte for byte |
| `cpu_q8a128_emit_bytes` | The pool's SwiGLU → q8a128 emission against a scalar reference, byte for byte |
| `cpu_expert_within_gpu_error` (cuda) | At every `cpu_n_ns` bucket and on each ISA tier (forced by the test): the CPU's rel-L2 against the f32 reference is ≤ 1.1 × the GPU's |
| `cpu_presum_matches_weighted_rows` | The pool's per-token row equals the weighted sum of its per-assignment rows, within f32 reassociation |
| `split_params_layout` | Rust and CUDA layouts byte-equal |
| `bucketize_split_is_optimal` | Classes for given counts, residency and `SplitParams`: the two-pointer result equals a brute-force min over every (k, j), under the piecewise model; the cut and `max_cpu_assignments` hold; promotion slots go only to GPU-class misses; each knob moves the result as §5.3.1 says |
| `split_never_slower_than_all_gpu` | Random decode, prefill and mixed count vectors and random `SplitParams`: modelled time ≤ all-GPU − `min_gain_ns`, or all-GPU |
| `split_takes_resident_when_gpu_bound` | A GPU-bound prefill vector with no misses moves the lowest resident experts; `resident_to_cpu = 0` moves none |
| `split_charges_prefill_rows` | The same vector with a slower link sends fewer experts to the CPU, by the row-export and result terms |
| `max_cpu_assignments_zero_sends_all_to_gpu` / `split_forces_cpu` | Both extremes |
| `split_params_take_effect_next_wave` (cuda) | A knob written between two recorded decode waves changes the second wave's classes |
| `exploration_reopens_cpu` | After `explore_period` all-GPU waves near the threshold, one lowest-count miss goes to the CPU |
| `cpu_wait_is_a_down_worker_item` (cuda) | A prefill-sized down launch (8k tokens) with a delayed `done` completes with no hang, and the scatter adds exactly the flagged tokens' CPU rows |
| `all_gpu_cap_is_bit_identical_to_today` (cuda) | With `max_cpu_assignments = 0`, a MoE layer's output is byte-identical to the dispatch without the CPU path |
| `placement_kl_gate` (cuda, `#[ignore]`) | The §6 end-to-end gate (KL, top-1) for all-CPU against all-GPU on Flash-Next, on each machine |
| `cpu_served_miss_records_a_miss` | A CPU-class miss is recorded as a miss, claims no promotion slot, and is excluded from `link_us` |
| `keep_value_prefers_evicting_ram_backed` | With equal scores, the RAM-backed expert is the victim; `cold_bias = 0` makes them tie |
| `pageable_never_reaches_a_gpu_plane` | The device classifier never sees a pageable address; the pool resolves it |
| `pool_dead_faults_promptly` / `pool_drains_on_abort` | A dropped pool faults the wait with the CPU encoding before `SPIN_LIMIT_NS`; an abort leaves every slot marked done or abandoned |

---

## 8. Measurements

The build runs straight through (author's decision). These measurements run
alongside it. They fill `SplitParams`' measured fields and tune the knobs; they
do not gate the work. They are `#[ignore]`d benches beside the code they
measure, not scripts. Each machine's figures go into `docs/performance.md`.

0. **The handoff**, on each machine under WDDM, as p50 and p99:
   - doorbell → pool wake → `done` → GPU observe, with no expert work;
   - the pool spinning versus UMWAIT/MWAITX, which also answers whether a PCIe
     write wakes it;
   - the GPU waiting with a worker poll versus `cuStreamWaitValue32`;
   - the decode row writes' cost on `hc_mix` when unused (§5.9).

   This gives t_h and settles §5.9's open choices.
1. **DRAM read bandwidth with the pool's layout:** alone (B_d), with the copy
   engine streaming (B_c), and on the workstation with both devices running.
2. **The link:** the live width and generation under load on the laptop, and a
   timed 2 MB warm copy (B_l).
3. **The kernel** across the `cpu_n_ns` buckets, per ISA tier. On the laptop,
   run alternating A/B under GPU load with clocks sampled.
4. **The miss profile under Q3_KO** from the per-layer counters, on the laptop
   and the 3090 box:
   - warm and cold miss shares;
   - copy time per step;
   - the 3090 box's actual pinned warm-tier size under WDDM.

   The published Flash-Next tables predate Q3_KO.
5. **Half-waves at ×N** (§5.9): A/B binaries at ×8 and ×16.
6. **Hit rate with the split on versus off**, steady state, at ×1, ×8 and
   prefill. This checks the expectation behind the promotion decision (§5.11):
   that prediction and read-ahead keep the miss rate small without CPU-served
   misses promoting.
7. **End to end.** Tokens per second at ×1, ×8, C5×8 and prefill, A/B,
   alternating.
   - On the laptop, pool load can lower the GPU's SM clock, which slows the
     non-MoE part of the step where `SplitParams` can't see it. The split can
     improve layer miss time and still lose t/s.
   - **The decision is the end-to-end number, not layer timing.**
8. **The EWMA time constant** against the laptop's measured throttle period.
9. **Accuracy** (§6): the per-expert error ratio, and the end-to-end KL and
   top-1, per machine and ISA tier.

**The laptop, measured** (`tests/cpu_expert_speed.rs`: Flash-Next, a 1,024-token
prompt then 128 greedy steps, all-GPU and split runs alternating, online loop
held still). Per decode step of 480 routed experts:

| | all-GPU | split |
|---|---:|---:|
| decode t/s | 13.0–13.4 | 12.4–12.7 |
| prefill t/s | 305–311 | 293–299 |
| VRAM hits | ~308 | 238–257 |
| CPU-served | — | ~140 (3.3 a layer, 290–350 µs; the device waits 20–35 µs) |
| pinned copies over the link | ~144 | ~45 |
| cold demand reads (the layer waits on the pack) | ~28 | ~47 |
| pack reads, demand and staged ahead | 175–178 | 171–181 |

Measurements 4, 6 and 7 together say why it doesn't pay here yet:
- **Decode is bound by the drive, not the link.** ~178 pack reads a step at
  13 t/s is ~4.7 GB/s against the drive's ~6.3 GB/s ceiling. The split
  removes ~100 link copies a step, and that link time was overlapping waits on
  the drive anyway.
- **The split moves drive waits onto the critical path.** It reads no more
  from the pack, but its staging lands late: 50% of predicted stagings
  are routed in time against 70%. A layer that sheds its copies reaches its
  successor's bucketize sooner, and the predictions lose the lead time they
  were made with.
- **The pageable warm tier has no room here.** The warm tier was bound by
  available RAM (12.9 GB free at launch of 33.8 GB, 4,829 pinned slots of
  24,064 wanted, 2.5 GB free once loaded), so §5.7 can't grow the RAM-backed
  share on this machine as loaded. That share is what would turn pack reads
  into CPU work.

**The pool's own load throttles the laptop's CPU.** Logged alongside the
alternating bench (`typeperf`, `% Processor Performance`): idle 185% of base,
all-GPU arms 120–155% at 30–36% CPU load, split arms 71–78% at 72–83% load —
the pool's fourteen raised-priority threads hold the package busy between
layers whether they compute or wait. The pool then computes a decode layer in
0.8–1.15 ms against calibration's ~0.3 ms, and every split arm lands below
all-GPU (8–10 t/s against 11.0–11.4 in the same run). Decode hands the pool
~3 one-row experts a layer, which a few cores read as fast as fourteen
(measured: a P-core-only pool matched the full one), so the pool's size and
its idle policy are what to cut next on this machine.

**After the pool was cut to its participants, cold-first and the decode
lookahead** (same bench, 2026-10-17, arms rotated each run, warm tier
4,300–5,600 slots as free RAM allowed):

| | all-GPU | split, promote none |
|---|---:|---:|
| decode t/s | 11.3–12.8 | 12.2–13.4 |
| the GPU's cold waits a step | 36–46, ~0.8–1.0 ms each | ~1 |
| CPU-served a layer | — | 3.0–3.7 experts, 1.0–1.3 of them cold |
| pool per layer | — | ~600 µs computing, ~500 µs waiting on pack reads |
| the device's wait on the pool, a layer | — | 700–800 µs |

- **The decode step is now a chain of pack reads.** Two thirds of the layers
  hand the pool a cold expert. Its read (~0.9 ms, against the drive's ~0.6 ms
  for a lone 2 MB record) and then its compute (~300 µs) sit between the
  layer's bucketize and its down launch, and the GPU's own work in between is
  ~350 µs. `moe:down` is ~38 ms of a profiled step's ~95.
- **The pool computes 2–3× slower in the run than alone, and the clock is why.**
  Each participant's matvecs take 190–330 µs a run, against ~55 µs alone for
  the same share; the last participant joins 50–100 µs in. `% Processor
  Performance` falls from ~180% to 80–110% over a run. DRAM contention with
  the link was ruled out earlier (6%), and pinned memory reads as fast as
  pageable (`decode_layer_latency`).
- **Measured and rejected:**
  - workers spinning through the layer gap (3 ms): no faster;
  - helper threads (readers, stager, pipeline) kept off the pool's P-cores:
    their matvecs got ~15% faster, but the stager's publish got slower;
  - only cold experts on the CPU: 6–12% slower — the GPU then copies every
    pinned miss;
  - a cold expert read in two parts, gate+up landing first so the pool starts
    on it: the two requests made each read ~15% slower, and the pool's
    participants spun through the rest of it, so it was 8–15% slower;
  - staging the router look-ahead's whole voted set for the next two rows:
    no change, because the predicted cold experts were already staged — the
    cold misses left are the ones it does not predict;
  - one speculative read in flight, or four: slower than two;
  - a pad of four layers, out of the warm tier's budget: prefill ~15%
    slower, and no fewer cold misses.

**Where it is expected to pay** (to be confirmed by 4 and 7):
- **The 3090 box:** PCIe 3.0, with as much of the expert set warm as WDDM
  allows.
- **The laptop:** on its warm share, on the pageable warm tier (§5.7), and
  through residency priced by backing (§5.11).
- **The workstation:** on whatever 2 × 32 GB doesn't hold.
- **The 72 GB card:** almost nothing.

---

## 9. Build order

Straight through, with the §8 measurements alongside. Each step lands with its
tests and leaves the tree working. No GPU kernel's bytes change: with
`max_cpu_assignments = 0`, every step leaves output byte-identical to today.

1. **The CPU kernel.**
   - The scalar Q3_KO/Q4_KO reference and the codec pieces: unpack, activation
     permute and q8a128 emission, with raw-byte tests.
   - The SIMD tiers: AVX2, AVX-VNNI and AVX-512 VNNI.
   - **The optimisation runs** (§5.10), until each machine meets the targets or
     the bench shows where it stops. The `cpu_n_ns` tables go into
     `docs/performance.md`.
   - The handoff, bandwidth and link benches (§8: 0–2) are built alongside.
2. **The pool, the rings, liveness and drain, and `SplitParams`**, with its
   config section and admin route. Driven by tests that write work slots
   directly.
3. **The integration.**
   - Bucketize's CPU class, the split and the online loop.
   - The down launch's CPU item and the scatter's CPU-row add.
   - The row writes and export.
   - CPU-class misses recorded as misses with no promotion slot, and `link_us`
     counting GPU copies only (§5.11).
   - The invariant amendments (§4, approved) and the doc amendments, in the
     same change.
   - The §6 accuracy gates run here.
4. **Pageable warm slots** (§5.7) and **the partition's refill price** (§5.8).
5. **Residency priced by backing** (§5.11): the promotion offer order first,
   then eviction's `keep_value`, then the warm tier's fill rule. Each is swept
   with `cold_bias` on the laptop.
6. **Half-waves at ×N**, kept only if measurement 5 shows a win.
7. **Several GPUs** (§5.12), on the workstation.

---

## 10. Sources

**Strata**
- <https://github.com/Niko1221/Strata> at `296b59d`. File and line references
  in §2 are from that checkout.
- Releases: <https://api.github.com/repos/Niko1221/Strata/releases>
  (v0.1.39, v0.1.42).
- <https://www.besthub.dev/articles/how-strata-runs-a-125b-model-on-a-12gb-gaming-gpu-47ec066aabfc>

**KTransformers (SOSP '25)**
- Paper: <https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf>
- kt-kernel: <https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md>
- What it contributes here:
  - AMX/AVX-512 kernels, switching at more than 4 tokens per expert;
  - NUMA tensor parallelism;
  - `cudaLaunchHostFunc` graphs;
  - Expert Deferral (up to 1.45×, lossy);
  - 220 of 307 GB/s DRAM.

**llama.cpp**
- `-ot exps=CPU`, `--n-cpu-moe`, and op offload at
  `GGML_OP_OFFLOAD_MIN_BATCH = 32`:
  <https://github.com/ggml-org/llama.cpp/pull/15077>,
  `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-cpu/ggml-cpu.c`
  (`mul_mat_id`).

**ik_llama.cpp**
- `-fmoe`: <https://github.com/ikawrakow/ik_llama.cpp/pull/229>
- `-rtr`: <https://github.com/ikawrakow/ik_llama.cpp/discussions/258>
- The (E/k) × 32 per-expert threshold:
  <https://github.com/ikawrakow/ik_llama.cpp/pull/520>

**Papers**
- Fiddler: <https://arxiv.org/html/2402.07033v3>. Per-expert cost model.
- HybriMoE: <https://arxiv.org/html/2504.05897>. Three-queue intra-layer split;
  the CPU also takes cached experts when idle (§5.3's j-phase).
- MoE-Lens: <https://arxiv.org/html/2504.09345>.
- MoE-Gen: <https://arxiv.org/html/2503.09716>.
- DAOP: <https://arxiv.org/html/2501.10375>.
- Huang, Lin, Lee (ASP-DAC '26): <https://arxiv.org/html/2512.16473>.
- CoX-MoE: <https://arxiv.org/html/2605.17889v1>.
- Wang et al. (OSDI '26): <https://arxiv.org/html/2606.10493>.
  - AVX-512 FP8 GEMV at 947 GB/s on 2× EPYC 9355.
  - Per-expert barriers.
  - KTransformers at ~50% of DRAM peak.

**Synchronisation**
- `cudaLaunchHostFunc` stalls and WDDM ordering:
  <https://forums.developer.nvidia.com/t/culaunchhostfunc-overhead-latency-usage-cpu-gpu-signaling/327066>
- Stream memory operations:
  <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html>

**Instruction throughput**
- <https://uops.info/html-instr/VPDPBUSD_ZMM_ZMM_ZMM.html>
- <https://uops.info/html-instr/VPMADDUBSW_YMM_YMM_YMM.html>
