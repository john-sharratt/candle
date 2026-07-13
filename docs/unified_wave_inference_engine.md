# Unified Wave Inference Engine — Design

**Status:** Design draft (pre-implementation). Supersedes the current three-phase
scheduler quantum for the MoE decode/prefill/glue path.

**Target model / hardware:** Qwen3-30B-A3B (Q4_K_M, 18.56 GB) on a 16 GB
RTX 4090 Mobile — i.e. the weights do **not** fit resident, experts stream from
pinned RAM over PCIe. The design generalises to the 2×5090 production box (where
it becomes a pure throughput win rather than a survival mechanism).

---

## 1. Problem statement

### 1.1 What we measured

Single-session upload ingest sustained **~70 t/s prefill**. The GPU prefill
*forward* did ≤512 tokens in ~1068 ms — a **~950 ms fixed cost per forward that
is independent of token count** (339 tok → 1068 ms; 1907 tok → 1714 ms; marginal
~0.4 ms/tok ≈ 2400 t/s). The fixed cost is **streaming experts from pinned RAM**:
the 18.56 GB model does not fit in 16 GB, so the expert-cache LRU holds only a
fraction of the 128 experts/layer resident (`num_slots < total_experts`,
`quantized_qwen3_moe.rs:1468-1496`), and each forward evicts + DMAs the experts it
misses.

### 1.2 Why ≤512-token forwards are near the worst case

MoE routing is top-8 over 128 experts. By the coupon-collector bound, a chunk of
`C` tokens activates `≈ 128·(1 − (127/128)^(8C))` distinct experts:

| C tokens | assignments 8C | distinct experts hit |
|---:|---:|---:|
| 32 | 256 | ~111 / 128 |
| 64 | 512 | ~126 / 128 |
| **128** | 1024 | **~128 (all)** |
| **512** | 4096 | **all 128** |

So **any forward ≥ ~64–128 tokens already touches essentially all 128 experts**
— it pays the full all-expert weight-load. Processing only ≤512 tokens per such
forward is the worst amortisation: maximum PCIe traffic for minimum tokens. The
cost model (confirmed in `compute.rs` / `pipeline.rs`) is:

```
forward_time(C) ≈ expert_residency_churn(fixed, ~all experts) + C · compute_per_token
```

Once the active experts are resident, per-layer compute is `∝ C` and the load
cost is amortised to zero. **The lever is therefore C: drive tokens-per-forward
as high as possible** so the one-time all-expert residency is spread over an
unbounded token count. The grouped GEMM already applies each loaded expert weight
once to its whole token-group regardless of group size (`compute.rs:108`,
"~60 kernel launches → 3").

### 1.3 Why the current architecture can't drive C up

The scheduler runs **three separate, homogeneous forward passes** — it never
mixes token types into one forward, and each type's forward is small:

- **Scheduling quantum** = one iteration of `Scheduler::run` (`run.rs:94`):
  `drain → promote → {decode quantum ≤32 steps, prefill 1 pass, section 1 chunk}`,
  widest-phase-first (`run.rs:136-151`). ("Wave" in the logs is only a **2 s
  logging window**, `mod.rs:1091` — not a scheduling unit.)
- **Decode**, **prefill**, and **glue** are dispatched to **different attention
  kernels and different forward shapes**, forked purely on query length at
  `batched_inference.rs:2702` (`max_input_len == 1` → decode; `>1` → prefill;
  glue rides the prefill route). A forward is all-decode `[b,1,h]` or
  all-prefill `[1,Σq,h]` — never both.
- Per-forward prefill is capped at **512 tokens/seq** (`max_prefill_pass_tokens`
  ← `config.scheduler.large_prefill_max_tokens`, `engine.rs:386`; the model also
  re-slices any batch over `MAX_PREFILL_TOKENS = 4096`, `batched_inference.rs:2590`).
- There is **no time budget, no cooldown, no adaptive sizing, and no
  queue-depth signal** — sizing is fixed constants (`DECODE_BUDGET=32`,
  `PREFILL_BUDGET=1`, `MAX_ACTIVE_PREFILLS=16`); the only backpressure is a
  VRAM-band eviction throttle (`prefill.rs:46,151`). `SchedulerConfig`'s
  `small_prefill_threshold` is dead (`config.rs:1275`, "reserved for Phase 2").

So during a single-session ingest each quantum spends most of its wall-clock on
32 memory-bound decode steps (the whole-file summary) **plus** one ≤512-token
memory-bound prefill pass — every one of them re-streaming experts. Nothing
amortises the expert load across the decode + prefill + glue tokens that are
in flight *at the same time*.

---

## 2. Core insight

**The expensive path is already unified; only the cheap plumbing forks.**

From the four subsystem studies:

1. **Post-attention is fully type-agnostic and token-flat.** There is exactly
   one FFN call site per layer (`batched_layer.rs:390-393`): `ffn_norm →
   ffn_forward → SparseMoeBlock::forward_dynamic`, which immediately flattens to
   `num_tokens = b_size·seq_len` (`quantized_qwen3_moe.rs:253`) and routes/gathers
   per flat row. The router and experts never see sequence structure or token
   type. All projections (q/k/v/o) and the FFN qmatmul are shared `QMatMul`
   weights invoked through one `forward_dynamic` dispatch.

2. **One `process_request` per layer serves the whole co-batched set**
   (`quantized_qwen3_moe.rs:529` → `handle.rs:459`). All sessions' tokens for a
   layer are counting-sorted into one `assignments` array and one grouped GEMM
   (`compute.rs:108`); an expert loaded to VRAM serves **every** token from
   **every** session routed to it. This *is* the "wave-batched grouped GEMM
   across 64 sessions" — the amortisation is the session dimension folded into
   `num_tokens` **before** the expert dispatch. It is created upstream by the
   batch assembler, not inside the pipeline.

3. **The three attention flavors already read the identical chunked KV arena**
   via `SlotHeader`, with the same `rope_base` positional convention. Glue is
   literally the decode kernel widened from 1 query token to G
   (`paged-glue/api.rs:1-8`) and already co-travels with prefill in one forward
   (`meta.glue`), carrying **per-row** masks (`fwd_ahead[t]` bridge window,
   `cu_seqlens_q`). Heterogeneous per-token attention semantics in one forward is
   already proven.

4. **The only fork points are (a) the `max_input_len==1` header build
   (`batched_inference.rs:2702`) that produces a different metadata struct for
   decode, and (b) the attention-kernel dispatch (`batched_layer.rs:413`).**
   Everything downstream of attention already unifies.

**Therefore the unified wave is a batch-assembler + attention-dispatch reshape.
The expert/qmatmul compute — the expensive, amortisation-critical part — is
untouched.**

---

## 3. Architecture — the unified wave

### 3.1 One mixed forward

Replace the three homogeneous forwards with **one forward over a single packed
activation buffer** whose rows are the union of all in-flight decode, prefill,
and glue tokens for this step:

```
   ┌──────────────── one unified wave (one forward, all layers) ────────────────┐
   │                                                                            │
   │  build packed batch:  rows = [ decode(q=1)… | prefill(q=N)… | glue(q=G)… ] │
   │  one cu_seqlens spans all three, decode rows are q_len==1 segments         │
   │                                                                            │
   │  for each layer L:                                                         │
   │    ── attention (per-type kernels, each fills its DISJOINT rows) ──        │
   │        paged_decode  → decode rows      (SlotHeader[decode slots])         │
   │        paged_prefill → prefill rows     (cu_seqlens_q / kv_lens)           │
   │        paged_glue    → glue rows        (+ write_slice / fwd_ahead)        │
   │        → all write into ONE [total_tokens, hidden] attention-out buffer    │
   │    ── o_proj (shared QMatMul, reconcile int8/float context → common hid) ──│
   │    ── residual + ffn_norm ──                                               │
   │    ── ONE MoE/FFN pass over ALL rows (one process_request, grouped GEMM) ──│
   │        experts loaded once this layer, amortised across decode+prefill+glue│
   │    ── residual ──                                                          │
   └────────────────────────────────────────────────────────────────────────────┘
```

Key properties:

- **Attention stays per-type** — three kernel launches, each over its row-subset,
  because their masking/position/KV-write differ. This is *exactly* the user's
  "the only difference is the attention kernels that run one after the other for
  decode, prefill and glue to prepare before it goes through the experts."
- **Everything after attention is one pass.** o_proj, ffn_norm, and the MoE
  grouped GEMM run once over `total_tokens` rows. The experts loaded for this
  layer serve decode + prefill + glue tokens together — that is the new
  amortisation the current architecture cannot get.
- **Experts and qmatmul are reused** across all row-types and (via the existing
  session concatenation) all sessions in the wave.

### 3.2 What must change (the two fork points)

1. **Unified batch header (`batched_inference.rs:2702`, `batched_layer.rs:39`).**
   Stop hard-forking on `max_input_len==1`. Build one `WaveBatch` header that
   carries, per row-group:
   - `cu_seqlens_q` / `q_lens` / `kv_lens` spanning **all** rows (decode rows are
     `q_len == 1` segments — the varlen prefill substrate already supports this,
     `BatchedPrefillMeta::new_ragged`).
   - The decode `SlotHeader[…]` metadata (currently `build_decode_metadata`) for
     the decode row-subset.
   - The glue `GlueMeta` (`write_slice`, `in_blk`, `fwd_ahead`) for the glue
     row-subset.
   These are additive per-subset descriptors, not a merged kernel contract.

2. **Attention dispatch runs all three sub-kernels into one buffer
   (`batched_layer.rs:413`).** Instead of `if seq_len==1 { decode } else {
   prefill/glue }`, the layer runs whichever sub-kernels have rows this wave,
   each writing its slice of the `[total_tokens, hidden]` attention-out buffer.
   Row offsets come from the unified `cu_seqlens`. Reconcile the decode int8
   context path (`want_q8`, `batched_layer.rs:566`) with the prefill/glue float
   context at o_proj so all rows land in a common hidden-space dtype before the
   shared FFN (trivial — o_proj output is already the common FP/BF16 hidden).

### 3.3 What does not change

- `SparseMoeBlock::forward_dynamic`, `submit_moe_work`, `process_request`,
  `classify_and_load`, the grouped GEMM (`compute.rs`), the Markov prefetch
  (`transition.rs`), and every `QMatMul` — **untouched**. They already consume a
  flat `[num_tokens, hidden]` buffer.
- KV arena layout, `SlotHeader` indexing, `rope_base` positions — untouched; all
  three attention paths already share them.
- Cross-session concatenation (`batched_model.rs:288`) — reused; a wave is just a
  richer `contexts` set that now spans token *types* as well as sessions.

---

## 4. Batch-sizing policy — small ⇄ large flip

The unified wave still has to choose **how many tokens** to pack. §1.2 says the
danger zone is the middle (128–512 tokens: all experts, no amortisation). The
policy explicitly avoids the middle by living at one of two operating points and
time-sharing between them.

### 4.1 Two operating points

- **Small wave (latency mode).** Adaptive **8–64 tokens** (start ~16),
  self-correcting on measured throughput within the range each iteration. Small
  waves keep interactive decode responsive and, crucially, exploit **temporal
  expert locality**: back-to-back small waves for the same conversation(s) re-hit
  overlapping experts that the LRU keeps resident → mostly cache hits, cheap.
  They ride along whatever decode + glue + a little prefill is pending.

- **Large wave (throughput mode).** Triggered only when the pending-token backlog
  exceeds a threshold (**~8192 tokens**, tunable) — enough that streaming all 128
  experts is worth it. Packs as many backlogged **prefill + glue** tokens as VRAM
  allows (see §4.4) into one forward. Amortises the all-expert load over thousands
  of tokens (~2000+ t/s territory, §6). **A large wave carries NO decode.**

The middle band (roughly 65–8191 tokens of *prefill* backlog) is never run as a
dedicated prefill wave; that backlog either rides inside small waves (a few
prefill rows at a time) or waits until it crosses the large-wave threshold.

**Why large waves exclude decode but include glue — the tokens-per-KV-prefix
rule.** The discriminator is *how many query tokens each admitted work item
amortises its resident KV prefix over*:

- A **decode** row is **one** query token, but to attend it needs the
  sequence's entire provenance-selected KV context (~thousands of tokens)
  resident in VRAM. Ratio ≈ **1 token / thousands of KV**. In a large wave —
  which is already VRAM-tight because its big activation buffer competes with KV
  for the ~3 GB headroom (§4.4) — admitting decode spends precious VRAM on
  resident context to produce a single token of output. Terrible trade. So
  decode is served **exclusively by small waves** (§4.6), where its latency also
  matters most.
- **Prefill** and **glue** rows come in *runs* — G or N query tokens sharing one
  resident KV prefix (a prefill chunk over its growing context; a glue gap-fill
  run over its conversation's sealed prefix). Ratio ≈ **many tokens / one
  prefix**. They amortise both the resident KV *and* the all-expert load, which
  is exactly what a large wave is for. Glue also *is* part of getting a
  reprojected context prefilled, so it belongs with the prefill backlog.

Rule: **a large wave admits only work with a high tokens-per-KV-prefix ratio
(prefill, glue); one-token-per-prefix work (decode) is excluded.**

### 4.2 Cooldown — recovery-aware, not a fixed time slice

A large wave doesn't just consume wall-clock — it **smashes the hot expert
set**. Because a large prefill wave activates all 128 experts/layer, it loads
them into the LRU and evicts the tail, which includes the experts the active
**decode** conversations were riding. So the *first several decodes after a large
wave are cold* — they re-stream their working set from RAM and run much slower
than steady state. A naive "equal wall-clock" cooldown under-protects decode:
half the decode airtime is spent just re-warming.

Two refinements:

**(a) Skip the cooldown entirely when there is nothing to decode.** If
`active_decodes` is empty (and no in-flight prefill is about to promote into a
decode), there is no interactive work to protect — run large waves back-to-back
at full throughput. This is the bulk-ingest / offline case (code-read, upload
storms), where we *want* 100% large-wave time. The cooldown exists only to shield
live decode.

**(b) When decode IS pending, cool down until decode has RECOVERED, not for a
fixed time.** After a large wave, run small (decode) waves until BOTH:

- a floor of `K · t_large` wall-clock has been spent on small waves, with
  `K > 1` (e.g. 2–3×) — the cooldown is *longer* than the large wave precisely
  because the large wave inflicted the cold-restart penalty on decode; and
- **decode throughput has recovered** to ~its pre-large-wave steady state
  (self-measured tps for the active decode set is back within a band of the
  baseline captured just before the large wave fired) — i.e. the decode working
  set is hot again.

Only then may the next large wave fire (and only if the backlog still exceeds the
threshold). Condition (b) is the smart part: the cooldown ends when decode has
*actually* recovered, adapting to how badly the large wave hurt it, rather than
after an arbitrary interval.

**Mitigation — pin the decode working set (optional, VRAM-permitting).** To blunt
the smash at its source, the large-wave admission can **pin the resident experts
of the active decode conversations** so the LRU eviction band skips them
(`evict_cold_tail`, `prefill.rs:178`, would exclude pinned keys). The large wave
then loads its experts into the *un*-pinned slots only, so decode stays warm
across the wave — at the cost of a smaller expert budget for the large wave
(fewer resident slots → more of its own streaming). This is a direct
VRAM/throughput trade; enable it when decode latency SLAs bind, disable it for
pure offline ingest. Requires an expert-cache pin API (does not exist today).

### 4.3 Adaptive small-wave sizing (self-correcting controller)

Small-wave size is a closed loop over measured throughput, bounded to `[8, 64]`:

```
each small wave:
  measure  tps = tokens_this_wave / forward_ms
  if tps improved vs last wave and size < 64:  size = min(64, size + step)
  else if tps regressed and size > 8:          size = max(8,  size - step)
  # hill-climb with a small step (e.g. ±8), decaying step near the optimum
```

The controller learns the per-situation sweet spot (which shifts with how much
of the expert working set is resident, how many sessions are co-batched, and KV
depth) instead of a fixed constant. It is deliberately conservative (narrow
range, small step) so it never wanders into the middle danger zone.

### 4.4 Large-wave ceiling (VRAM-bounded)

The large-wave size is bounded by **activation VRAM**, not by the token count.
The current VRAM reality (corrected from the batching study):

- `used` on the wave line (`~11.7 GB`) is the **whole resident pool** — non-expert
  weights + resident expert-LRU slots + KV arenas + activation workspaces
  (`vram_pool_stats`, `batched_inference.rs:1282`) — **not** KV alone.
- `budget` (`~3 GB`) is **growth headroom** = `init_free − pool_used − reserve`
  (`alloc.rs:141-154`, reserve default 384 MiB). KV competes for this headroom
  with the large-wave activation buffer.

So the large-wave ceiling is `available_headroom / bytes_per_token_activation`,
recomputed each trigger from `vram_budget_available()`. A 4096–8192-token wave
needs a correspondingly larger `[total_tokens, hidden]` activation + grouped-GEMM
scratch; the admission controller must size the wave to fit the *current*
headroom (which shrinks as KV grows), and fall back toward small waves under
pressure (reuse the existing `vram_under_pressure` machinery, `prefill.rs:151`).

### 4.5 New signal required: pending-token backlog

**This does not exist today and must be built.** The prefill FIFO
(`prefill_queue: VecDeque<PrefillWork>`, `mod.rs:1220`) is only ever tested with
`.is_empty()`; its depth is never computed, and there is no aggregate
pending-token accounting anywhere (no `>8192` trigger, no `queue.len()` read).
Add:

- A running **`pending_prefill_tokens`** counter (sum of un-started + un-advanced
  prefill/section token counts, plus optionally queued decode demand).
- Surface it on the wave log line alongside the executed-forward counters (today
  the line reports only *executed* work, never *backlog*).
- The large-wave trigger reads this counter; the adaptive controller reads
  realised throughput.

### 4.6 Small-wave selection — fair, expert-reusing, KV-reusing

Small waves carry **decode** (plus ride-along glue). Each wave picks a set `S` of
ready decode sequences (≤ the §4.3 latency cap `C_max ≈ 64`) to run in one
forward. Selection must be **(a) fair** (nothing starved), **(b) expert-reusing**
(hit resident/hot experts so the wave is near-all-cache-hit), **(c) KV-reusing**
(share KV prefixes; don't fault a cold multi-thousand-token context in just to
emit one token).

> This section states the *principled* form. An earlier draft used a hand-weighted
> linear score; it worked directionally but had two ranking bugs (it normalised the
> expert term, and it double-counted prefix sharing). The form below fixes those and
> is derived straight from the objective — same cost to compute, with a real latency
> guarantee and a clean stopping rule.

**Objective.** Maximise long-run committed-decode throughput
`Σ tokens / Σ cost` **subject to** a per-sequence latency bound `W_max`. Decompose
it: latency becomes a **deadline constraint** (EDF); throughput becomes a per-wave
**cost objective**.

**Cost model.** A small wave's wall-clock is
`≈ c_load·|E_new(S)| + c_attn·KV_read(S) + c_compute·|S|` where
`c_compute·|S|` is negligible (`|S| ≤ 64`) and the two real costs are:

- `E_new(S) = ⋃_{s∈S} Ê(s) \ R` — the distinct experts the wave activates that are
  **not** resident. This is a **union**: an expert loaded once serves every
  selected sequence routed to it. So the marginal of adding `s` is the *union
  growth* `|Ê(s) \ (R ∪ E(S))|` — **absolute new experts, not a per-sequence
  fraction** (a sequence hitting 100 experts with 90 resident still costs 10 loads;
  one hitting 10 with 8 resident costs 2 — the second is cheaper to add even though
  its reuse *fraction* is lower).
- `KV_read(S)` — total KV swept, with any prefix **shared** across selected
  sequences read once. So the marginal KV of adding `s` is its *incremental,
  non-shared* context `new_kv(s|S)` — sharing is captured **inside this term**, not
  as a separate bonus (a separate prefix bonus would double-count the saving).

`E_new(S)` (coverage) and the prefix-sharing in `KV_read(S)` are both
**submodular**, so **greedy is near-optimal** — no exact solver on the hot path.

**Predicting a sequence's experts.** Decode has strong temporal locality —
consecutive tokens of a sequence hit nearly the same experts — so
`Ê(s) = EMA/Bloom of the experts hit by s's last W decoded tokens` is a cheap,
accurate predictor, refined by the Markov transition matrix (`transition.rs`,
~69% next-layer hit). `R` = currently-resident expert set (LRU); `E(S)` = experts
already claimed by the partial selection.

**The formula (bang-per-buck greedy).** Because every decode row is one token,
`maximise tokens/cost` ⟺ **`minimise marginal cost`**:

```
Δcost(s | S) = c_load · |Ê(s) \ (R ∪ E(S))|     # absolute new (non-resident, unclaimed) experts
             + c_attn · new_kv(s | S)            # incremental non-shared KV (prefix sharing folded in)
```

Build `S`:

1. **Fairness = deadline constraint (hard).** Each ready sequence has a deadline
   `t_ready + W_max`; **force-admit any past its deadline first**. EDF is optimal
   for meeting deadlines → worst-case wait ≤ `W_max` → no starvation, independent
   of the cost objective. (An optional small urgency term `+ε·wait/W_max` on the
   score smooths latency so demand isn't bursty — a *smoother*, not the guarantee.)
2. **Seed** `S` with the oldest ready sequence.
3. **Greedy fill.** Repeatedly add `argmin_s Δcost(s|S)` (cheapest-to-add).
4. **Stopping rule (marginal vs average).** Adding `s` raises the wave's overall
   `tokens/cost` **iff `Δcost(s) < cost(S)/|S|`** — the cheapest remaining
   sequence's marginal cost is below the running average cost-per-token. **Stop
   when that fails**, or the latency cap `C_max` binds, or the VRAM/KV budget is
   hit — whichever comes first. (Do *not* pad the wave to `C_max` with an
   expensive cold sequence; a short all-hot wave beats a padded cold one.)

`c_load ≫ c_attn` on the streaming box (expert load ~10–20 ms/expert vs a cheap
per-token KV read); both can feed the §4.3 self-correcting loop.

**Why this is (near-)optimal — and hardware-adaptive.** The stopping rule is the
exact condition that maximises the `tokens/cost` ratio, and greedy on a submodular
union is within `(1−1/e)` of the best subset. Crucially, the resident-set term
makes the *same* policy right on both boxes: as residency → full (`R` = all
experts), `|Ê(s) \ R| → 0` for every `s`, so `Δcost` collapses to KV only and the
wave keeps admitting until the KV/latency cap — i.e. it **auto-degrades to
"batch-all-ready"** on the 2×5090 box (where batch-all *is* optimal), and stays
selective on the 16 GB streaming box (where a diverse batch would inflate the
expert union and blow KV). One formula, both regimes.

**How it meets the three goals:**

- **Fair:** the EDF deadline (step 1) bounds every sequence's wait to `W_max`; the
  optional urgency smoother avoids bursts.
- **Expert-reusing:** step 3 accretes sequences whose experts are already resident
  or already claimed → `E_new → 0` → near-all-cache-hit. Reinforced by *not*
  interleaving unrelated conversations gratuitously (drain a hot conversation's
  decode for a bounded run before switching, bounded by `W_max`, so the LRU keeps
  its working set hot) and by the Markov prefetch keeping predicted experts warm.
- **KV-reusing:** `new_kv(s|S)` rewards fork siblings that share a prefix (the Zen
  Code shared-prefix-across-forks case) — the shared context is read **once** for
  all their tokens — and penalises faulting a cold context in for a single token.
  Provenance selection already bounds each sequence's swept KV; **cache the
  selected subset across a sequence's consecutive decodes** (re-select only when
  its context changes) so it isn't recomputed per token.

**Alternatives considered (and why this wins).**

| Alternative | Verdict |
|---|---|
| FIFO / round-robin (fairness only) | Dominated — ignores reuse → cold, load-bound waves. |
| Greedy-reuse, no fairness | Violates the latency bound (starves never-hot sequences); the EDF floor is the fix. |
| Batch-all-ready | Optimal *only* when experts are resident; on 16 GB it inflates the expert union + resident KV. **Subsumed** — the formula auto-degrades to it when `R` = all. |
| Cluster-major / gang-by-expert | Clean special case (one wave = one hot cluster = minimal union). Great on crisp clusters, degrades on fuzzy overlap. **Recommended first cut** (below). |
| Exact ILP per wave | NP-hard, unnecessary — submodular greedy suffices. |
| MDP / RL lookahead | Accounts for the future value of keeping a set hot; complex, needs an arrival model. The `R`-overlap term already captures most of it. Future work. |

**Simplest correct first cut** (migration step 5): the **cluster-major**
approximation — bucket ready decode sequences by `(fork_root, dominant
resident-expert cluster)` and drain the bucket containing the oldest sequence up
to `C_max`. It captures most of the expert- and KV-reuse win with trivial
bookkeeping (O(clusters), no per-add `E(S)` recompute); the full bang-per-buck
greedy replaces it once the coarse version is validated.

### 4.7 Bounded ingest context — the rolling window (PREREQUISITE)

Code investigation confirmed a real defect the wave engine must **not** inherit:
**the `code_read` (and any append-only bulk) ingest prefills each scope against
the ENTIRE accumulated linear context of that file's conversation — system prompt
+ every prior scope — with no window, no top-k, no eviction. Context grows
unboundedly with scope count.**

Mechanism (why it's unbounded today):

- `code_read_config` sets `disable_reprojection = true` (`repo_scan/mod.rs:48`).
- That makes `skip_projection = disable_reprojection && block_count > 0` true from
  the first scope (`scheduler/mod.rs:1699`), so `apply_projection` is skipped and
  the cumulative slot is left intact (`:1975-1987`); the prefill view borrows the
  **whole parent** — `BlockRange::new(0, parent_block_count)` (`:2009`).
- The `scopes` group *does* declare `selection: {top_k, k:8}` + `window: 6000`
  (`projection.yaml:244`), but that only runs under `apply_projection`
  (retrieval / summary), **never during the append-only ingest**.
- So `context_depth = sequence_offset(view)` climbs monotonically
  (`prefill.rs:643`). The ~6200 seen in logs is just a small file's cumulative
  total (~61 scopes × ~100 tok), **not a bound** — a genuinely large file grows
  past the model's context window and blows up KV VRAM.
- Neither provenance/BDP selection **nor** any window is active for these turns —
  full linear KV, prefill and summary decode alike. (Provenance selection is the
  *projected* path's machinery; the ingest opts out via `disable_reprojection`.)

Why it's a prerequisite for the wave engine: the cost model (§1.2, §4.6) assumes
each sequence's `kv_swept` is **bounded** (a relevant subset, not the whole
growing file). Unbounded ingest context (a) makes every prefill forward's
`T_attn` grow without limit, (b) consumes the exact VRAM headroom the large wave
needs for its activation buffer (§4.4), and (c) can exceed the model's positional
limit outright. A large wave batches prefill backlog; if each backlogged scope
carries an ever-growing context, the wave's KV footprint is unbounded and the §6
throughput math breaks.

**Fix — a rolling window of N turns.** At the point where `skip_projection`
currently borrows the whole parent (`scheduler/mod.rs:1975-2013`), borrow instead
**only the system-prompt blocks + the last N turns' (scopes') blocks**, so each
scope prefills against `system_prompt ⊕ last-N-scopes` — bounded, constant-cost.

- `N` is the design knob (≈4–8 scopes of local context is plenty for the model to
  read a file coherently; whole-file understanding comes from the *summary* layer,
  not from holding every scope in one context).
- Alternative/compat: stop disabling reprojection for ingest so the `scopes`
  top_k=8 selection trims the context to the k most-relevant prior scopes — reuses
  existing machinery but pays the per-scope selection cost; the rolling window is
  simpler and deterministic.
- The dormant `context_window_turns` field (`config.rs:1345`, default 0, inert —
  superseded by projection) describes exactly this "system prompt + last N turns"
  rebuild but is not wired into `insert_turn`/prefill; the rolling window can
  revive it for this path or be a fresh ingest-local mechanism.

With the window, each scope's context is `O(N · scope_size)` — constant — so the
large wave's KV is predictable. **This lands early in the migration (§7) because
the whole large-wave throughput argument assumes bounded per-item KV.**

---

## 5. Scheduler restructure

Replace the three-phase quantum with a **single wave builder + dispatcher**:

```
loop {                                        // one wave per iteration
  drain_submissions();                        // unchanged (run.rs:102)
  promote_new_prefills();                     // unchanged admission (prefill.rs:36)
  update_backlog_counter();                   // NEW (§4.5)

  // choose_mode: Large iff backlog > THRESH AND cooldown allows. Cooldown is
  // SKIPPED when active_decodes is empty (nothing to protect → back-to-back
  // large waves); otherwise it holds until decode has recovered (§4.2).
  let mode = choose_mode(backlog, active_decodes, cooldown_state);

  let wave = match mode {
      // Small: decode (+ ride-along glue), sized 8..64 (§4.3), sequences chosen
      // by the fair/expert-reuse/KV-reuse selection (§4.6). NO prefill backlog
      // dump here — only a trickle if decode is light.
      Small => build_small_wave(adaptive_small_size(), select_decode_seqs()),
      // Large: prefill + glue ONLY (no decode, §4.1 tokens-per-KV-prefix rule),
      // sized to the current VRAM headroom ceiling (§4.4), optionally pinning the
      // decode working set (§4.2 mitigation).
      Large => build_large_wave(vram_bounded_ceiling()),
  };
  //   both pack their rows into ONE WaveBatch header (§3.2): decode = q_len==1
  //   segments, prefill = q_len==N, glue = q_len==G + write-slices/fwd_ahead.

  let out = model.forward_unified(wave);      // one forward, all layers (§3.1)

  sample_and_commit(out);                     // decode rows → sampler; prefill rows
  //   → advance offsets; glue rows → scatter into reserved gap chunks
  record_wave_stats(mode, wave, out);         // extend WaveStats with mode + backlog
  enforce_cooldown_accounting(mode, elapsed); // NEW (§4.2)
}
```

Notes:

- `active_decodes`, `active_prefills`, `active_section_ingests`,
  `pending_reprojections`, `sampling_states` stay as the in-flight holding
  structures; `build_unified_wave` co-drains them into one batch instead of three
  separate phase loops.
- The per-sequence sampler state (`sampling_states`, DRY window spanning turns)
  is applied only to the **decode row-subset** of the wave output — same logic as
  today's `sample_batch`, just selecting the decode rows out of the unified
  output.
- Glue commit (scatter into pre-reserved gap chunks via `write_slice/in_blk`) and
  prefill offset advance are unchanged, applied to their row-subsets.
- `PREFILL_BUDGET`/`DECODE_BUDGET` count-quanta are **retired** in favour of the
  token-budget wave; `DECODE_BUDGET=32` was really "32 decode steps back-to-back"
  — under the unified wave, decode is one row-group per wave, so its cadence is
  the wave cadence.

---

## 6. Expected gains (order-of-magnitude)

Using the measured decomposition (~950 ms fixed all-expert residency churn,
~0.4 ms/token marginal on this box) and folding decode+prefill+glue into one
amortised forward:

| wave tokens C | forward ms (est.) | throughput (est.) |
|---:|---:|---:|
| 512 (today) | ~1068 | ~480 t/s |
| 2048 | ~1420 | ~1440 t/s |
| 4096 | ~1900 | ~2160 t/s |
| 8192 | ~3300 | ~2480 t/s |

Plus a second-order win the table doesn't show: today the ingest's **decode**
(summaries) and **prefill** (scopes) each separately re-stream experts every
quantum. Unified, they share one per-layer expert load — so the effective
amortisation base is decode+prefill+glue tokens combined, and the LRU thrash
between the decode working set and the prefill working set disappears.

On the 2×5090 box (weights + full expert residency + KV all fit), the fixed
residency churn goes to ~0 and the same design becomes a pure batching win: large
waves push toward compute-bound throughput; small waves keep latency low.

---

## 7. Migration plan (incremental, each step shippable)

0. **Rolling-window ingest context (PREREQUISITE, §4.7).** Bound the code_read /
   bulk-ingest prefill to `system_prompt ⊕ last-N-scopes` instead of the whole
   growing parent (`scheduler/mod.rs:1975-2013`). Independent of everything below,
   fixes a live unbounded-context/VRAM defect, and makes per-item KV constant so
   the large-wave math holds. Do first.
1. **Backlog instrumentation (no behaviour change).** Add
   `pending_prefill_tokens` accounting and surface it on the wave line + a new
   `WaveStats` field. Validates the signal before anything depends on it.
2. **Raise the per-forward cap, measure.** Make `large_prefill_max_tokens` a
   real, VRAM-bounded knob (not fixed 512); confirm the §6 amortisation curve on
   the 4090. Cheap early win *within* the current 3-phase architecture; de-risks
   the cost model.
3. **Unified batch header (compute unchanged).** Introduce `WaveBatch` /
   `forward_unified` that packs **prefill + glue only** (they already co-batch)
   with decode still separate — proving the packed-buffer + per-type-attention
   path end-to-end with the smallest change.
4. **Fold decode into the unified wave.** Promote decode rows into `q_len==1`
   segments of the unified `cu_seqlens`; run `paged_decode` into its row-subset of
   the shared buffer; reconcile the int8 context at o_proj. Core change — one
   forward for decode+prefill+glue. (Even after this, the *policy* keeps decode
   out of large waves per §4.1; the mechanism just makes mixing possible.)
5. **Small/large flip + recovery cooldown.** Add `choose_mode` with the
   backlog>THRESH large-wave trigger, the **decode-excluding** large-wave
   composition (§4.1), the **recovery-aware cooldown** with skip-if-no-decode
   (§4.2), and the fair/expert-reuse/KV-reuse **small-wave selection** (§4.6,
   affinity-batching first cut).
6. **Adaptive small-wave controller + expert pin.** The self-correcting `[8,64]`
   hill-climb (§4.3), and the optional decode-working-set pin API (§4.2) if
   decode latency during large waves needs it.
7. **Retire the three-phase quantum** (`DECODE_BUDGET`/`PREFILL_BUDGET`) and the
   dormant `SchedulerConfig` Phase-2 fields; make the wave the single scheduling
   unit.

Each step is independently testable (TDD): golden-token determinism for the
packed-forward equivalence (steps 3–4 must produce bit-identical logits to the
current split forwards for the same inputs — and step 0 must produce identical
ingest summaries for files ≤ N scopes, where the window is a no-op), and
throughput/latency probes for steps 5–6.

---

## 8. Risks & open questions

- **Unified attention masking correctness.** Decode ("attend all prior"), prefill
  (intra-chunk causal + prefix), and glue (`cpos ≤ row_pos + fwd_ahead[t]`) are
  per-row today; packing them into one `cu_seqlens` must preserve each row's
  mask + `rope_base`. The golden-token equivalence test (step 3–4) is the guard.
- **Per-layer KV capacity.** `ensure_chunked_capacity_batch` runs per layer
  (`batched_inference.rs:2679`); a large wave writes far more new K/V per layer —
  must confirm capacity growth stays within the VRAM ceiling and doesn't trip
  eviction mid-forward.
- **int8 context reconciliation.** Decode can emit a q8a1024 context feeding
  o_proj int8 (B2); prefill/glue feed float. Merging into one o_proj input needs a
  single dtype — cheap (o_proj output is common hidden), but the B2 fast path may
  need a float fallback for mixed waves, or per-subset o_proj then concat.
- **Sampler / decode-state selection.** The sampler must pick exactly the decode
  rows out of the unified output; off-by-one in row mapping corrupts sampling.
- **Large-wave latency spikes.** An 8192-token wave is ~3.3 s of forward, during
  which no decode runs (§4.1). The recovery cooldown (§4.2) bounds *how often*
  large waves fire, but a single large wave still blocks interactive first-token
  latency for its duration — may need to cap large-wave size lower on the 16 GB
  box, or preempt at layer granularity (finish the current layer, yield to a small
  wave, resume) if per-turn latency SLAs are tight.
- **Unbounded ingest context (see §4.7).** Without the rolling-window prerequisite
  (migration step 0), code_read prefills the full growing linear context per scope
  — the large wave then batches ever-larger per-item KV and the throughput math
  breaks. This is a live defect independent of the wave engine; the wave engine
  just makes its VRAM cost acute.
- **Recovery-cooldown measurement.** The "decode has recovered" signal (§4.2)
  needs a stable per-conversation baseline tps captured just before the large wave
  and a recovery band; noisy tps (variable context depth, few decode sequences)
  could make the cooldown oscillate. Needs a smoothed estimator + a hard `K·t_large`
  floor and ceiling so it can't hang.
- **Expert-pin vs large-wave budget.** Pinning the decode working set (§4.2) to
  survive a large wave shrinks the large wave's own resident expert budget → it
  streams more. Quantify the crossover; likely only worth it when decode latency
  SLAs bind. Requires an expert-cache pin API (does not exist).
- **VRAM headroom is small and shared.** With ~3 GB headroom, a large activation
  buffer competes directly with KV; the ceiling controller must be conservative
  and fall back to small waves under pressure. Fundamentally a 16 GB constraint
  the 2×5090 box removes.
- **Expert prediction accuracy for selection (§4.6).** The `Ê(s)` EMA predictor
  drives expert-reuse batching; if it mispredicts, a small wave meant to be
  all-cache-hit faults experts and stalls. The Markov matrix (~69%) helps; measure
  realised hit-rate under the selection and fall back to pure-fairness ordering if
  it degrades.

---

## 9. Key code anchors

| Concern | Location |
|---|---|
| Scheduler loop / quanta | `candle-conversation/src/scheduler/run.rs:94`, budgets `:5-7`, phase order `:136-151` |
| Wave stats / log line | `scheduler/mod.rs:1101` (`flush`), fields `:1115-1138`, vram tuple `run.rs:163-167` |
| Forward dispatch fork | `candle-transformers/src/models/batched_inference.rs:2702` (`max_input_len==1`) |
| Attention dispatch | `batched_layer.rs:413` (`forward_attn_batched`), decode `:474/:899`, prefill/glue `:609/:711/:736` |
| Attn↔FFN boundary (type-agnostic) | `batched_layer.rs:390-393` |
| MoE token-flat | `quantized_qwen3_moe.rs:243` (`forward_dynamic`), flatten `:253`, submit `:529` |
| One process_request/layer | `expert_lre/handle.rs:459`, `pipeline.rs:887` (`process_request`) |
| Grouped GEMM (3 launches) | `expert_lre/compute.rs:108` |
| Expert VRAM budget | `quantized_qwen3_moe.rs:1468-1496`; pinned pool `handle.rs:287-294` |
| VRAM budget/used semantics | `candle-nn/src/kv_cache/chunked/alloc.rs:141-154`; `batched_inference.rs:1282` |
| Per-forward token cap (512) | `engine.rs:386`, `prefill.rs:368/379`; model cap 4096 `batched_inference.rs:2590` |
| Prefill queue (no depth signal) | `scheduler/mod.rs:1220`; backpressure `prefill.rs:46/151` |
| Glue assembly / bridge window | `scheduler/projection_assembler.rs` (`GapFillPlan`, `fire_gap_fill_batch`), `paged-glue/api.rs:44-54` |
| Cross-session concat | `batched_model.rs:288` (`forward_batch`) |
| Dormant Phase-2 config | `config.rs:1273-1292` |
| Ingest skips projection (unbounded ctx) | `repo_scan/mod.rs:48` (`disable_reprojection`), `scheduler/mod.rs:1699` (`skip_projection`), whole-parent borrow `:1975-2013`/`:2009` |
| `scopes` top_k / window (inactive in ingest) | `prompts/projection.yaml:244` (`top_k k:8`), layer window `:176` |
| Dormant rolling-window field | `config.rs:1345` (`context_window_turns`, default 0), `conversation.rs:2232` (`window_state`) |
| Expert affinity / prefetch (for §4.6) | `expert_lre/transition.rs` (`predict_prefetch`, `observe`) |
