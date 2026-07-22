# Continuous Fair Waves

**Thesis.** Stop *time-sharing* decode and prefill. Run them **concurrently** by
decoupling the rate at which each advances through the model's layers: fast
decode sweeps every layer on every wave (a token per sweep) while a large prefill
**creeps** through the layers in the background at a throttled rate. Because
decode re-touches every layer's experts continuously, its hot expert working set
is never evicted by the background prefill — so the "cold decode after a big
prefill" penalty disappears, and decode/prefill overlap becomes continuous rather
than an either/or flip.

This supersedes the small⇄large **flip + recovery cooldown** of
[`unified_wave_inference_engine.md`](unified_wave_inference_engine.md). That design
still ran one homogeneous mode at a time and then spent a cooldown re-warming
decode after each large prefill smashed the LRU. Continuous fair waves removes the
smash at its source, so there is nothing to cool down from.

---

## 1. The problem it solves

The current loop dispatches one forward at a time and flips between a small
(decode) forward and a large (prefill/glue) forward. A large prefill forward runs
**all layers, all 128 experts/layer** in one shot. On the streaming box (weights
don't fit; experts DMA from pinned RAM), that one forward loads every expert of
every layer into the LRU and **evicts the tail — which is exactly the working set
the live decode conversations were riding.** The next several decodes are cold:
they re-stream their experts from RAM and run far below steady state.

The flip design papers over this with a *recovery cooldown* — after a large wave,
run decode-only waves until decode tps recovers before allowing the next large
wave. That is a scheduling band-aid over a structural problem: **decode and the
big prefill are fighting over the same resident-expert LRU because they never
share the layer traversal.**

## 2. Core idea — decouple the layer wave from the inference wave

Two distinct notions of "wave" are conflated today:

- **Layer wave** — the traversal `layer 0 → 1 → … → N-1`. This is where the
  expensive work is: per layer, load/prefetch this layer's experts (Markov
  predictor, `transition.rs`) and run one grouped GEMM over whatever tokens are
  present.
- **Inference wave** — the logical unit of *output*: one decode token, or one
  completed prefill forward.

Today these are locked together: an inference wave *is* one full layer traversal.
Continuous fair waves **decouples them**. The layer wave runs continuously; two
independent cursors move through it at different speeds:

- **Decode cursor** advances one layer per step and completes a full sweep
  `0→N` every wave — one token per wave, exactly as today.
- **Prefill / glue cursor** advances through the layers **slowly**, governed by a
  priority ratio (§3). Its inter-layer hidden state (the residual stream between
  layers) is **persisted across waves** rather than living only inside a single
  forward call — so a prefill can pause at layer `L`, let many decode sweeps
  overtake it, and resume at `L` later.

Wherever the two cursors coincide at a layer, their tokens **co-batch through that
layer's experts in one grouped GEMM** — the expert load is shared. Where they
don't, decode runs that layer alone (it was going to pay that layer's decode-set
load anyway).

The result: decode touches every layer every wave, so its top-8 experts/layer stay
at the head of the LRU and never age out. The background prefill still activates
all 128 experts/layer, but it does so **one layer at a time, spread across many
waves**, and each of those loads lands in the LRU *behind* decode's continuously
re-touched set. The prefill's expert churn no longer evicts the decode working
set — it fills the cold slots around it.

## 3. Decode priority — the throttle

New per-layer metadata in `projection.yaml`:

- Each **layer** declares a `decode_priority`: `Low` | `Normal` | `High`.
- A conversation inherits its layer's priority when it is created and attached.
- **Default is `Low`** for every layer; the **dialogue** layer is `High`.

The priority is the **decode-to-prefill airtime ratio** `R` — how many decode
tokens are produced for each prefill forward that completes:

| priority | R (decode tokens per completed prefill) | effect |
|---|---:|---|
| Low    | 1  | prefill drains at full speed; least decode protection (bulk ingest) |
| Normal | 16 | prefill stretched ~16× while 16 decode tokens land |
| High   | 64 | prefill heavily throttled; decode latency maximally protected (dialogue) |

Concretely, with `N` layers the prefill cursor is given a budget of `~N/R` layers
of forward progress per wave (fractional, accumulated):

- **Low (R=1):** `N/1 = N` — the prefill clears all layers in one wave, i.e. it
  fully overlaps a single decode token. Maximum ingest throughput, minimum decode
  shielding.
- **Normal (R=16):** `N/16 ≈ 3` layers/wave — the prefill completes over ~16
  decode sweeps.
- **High (R=64):** `N/64 < 1` — the prefill advances roughly one layer every one
  or two waves, creeping in the deep background while ~64 decode tokens land.

`R` is the single knob that trades ingest throughput against decode latency, set
per layer by how interactive that layer's conversations are.

## 4. Glue rides with prefill

Glue (boundary gap-fill, `paged-glue`) is treated as **prefill-class** by the
wave loop: it inherits the prefill ratio and its cursor advances on the same
throttle. It differs only in the attention kernel that runs at each layer
(`paged_glue_attn` vs `paged_prefill_batched` vs `paged_decode`) — the layer
traversal, expert load, and grouped GEMM are identical. Glue *is* part of getting
a reprojected context prefilled, so it belongs on the prefill cursor.

## 5. What stays exactly the same

- **Slot allocation and the single background scheduler thread** — unchanged.
- **Concurrency levels** (active decode / prefill / section counts) — unchanged.
- **Attention kernels** — `paged_decode`, `paged_prefill`, `paged_glue` are
  untouched; each still fills its own disjoint rows.
- **The layer traversal is still single-threaded, start → end**, and still uses
  the **Markov expert prediction + per-layer expert streaming** exactly as today
  (`quantized_qwen3_moe.rs`, `expert_lre/`).
- **Decode state** — the sampler, DRY window, sampling-state-per-sequence — is
  unchanged; it is simply applied to the decode cursor's rows within the unified
  waves.

The novelty is entirely in **(a)** lifting the prefill/glue inter-layer buffers
out of the per-forward scope so they persist across waves, and **(b)** the
per-cursor layer-advance throttle driven by `decode_priority`.

## 6. Why it works

- **Expert cache pressure is fixed at the source.** Continuous decode sweeps keep
  the interactive experts resident *through* a large background prefill. No
  wholesale eviction, so no cold-restart penalty — and therefore **no recovery
  cooldown** to schedule around.
- **Continuous decode/prefill overlap.** Decode and prefill are no longer mutually
  exclusive modes; a large ingest prefill runs *underneath* live dialogue decode,
  cutting the latency of mixed decode+prefill workloads instead of serialising
  them.
- **One knob, per-layer, matched to interactivity.** `High` on dialogue protects
  first-token and inter-token latency; `Low` on bulk-ingest layers lets prefill
  run at full speed when nothing interactive is in flight. The same loop serves
  both without a mode switch.

## 7. Open questions

- **Per-cursor layer budget arithmetic.** `N/R` is the clean statement of intent;
  the exact fractional-accumulator rule (and whether prefill advances in
  contiguous layer runs or single layers per wave) is the primary tuning knob and
  should be validated against realised decode tps and ingest throughput.
- **Co-batch dtype at the shared layer.** Where decode (int8 context) and prefill
  (float context) coincide at a layer, their o_proj inputs must reconcile to one
  hidden dtype before the shared FFN — the same reconciliation the unified-wave
  design calls out (cheap; o_proj output is common hidden).
- **Persisted prefill activation VRAM.** Holding a large prefill's residual stream
  between waves costs `[prefill_tokens, hidden]` of resident activation for the
  duration of its (now stretched) traversal. On the 16 GB box this competes with
  KV; the throttle must fall back toward `Low` (drain fast, hold briefly) under
  VRAM pressure.
- **Bounded ingest context (prerequisite, unchanged from the unified-wave design
  §4.7).** Each prefilled scope must attend a bounded window, not the whole
  growing file, or the persisted prefill activation and its swept KV grow without
  limit.
