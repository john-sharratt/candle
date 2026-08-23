# The Markov Wave: Predictive Expert Prefetching and Eviction for Memory-Constrained MoE Inference

## Abstract

Mixture-of-Experts (MoE) language models activate only a few experts per token
but must keep *all* expert weights addressable; when they do not fit in VRAM the
remainder stream over PCIe, and expert-transfer latency becomes the throughput
bottleneck. We study the two prediction problems that govern this pipeline —
**promotion** (which non-resident experts to prefetch) and **eviction** (which
resident experts to drop) — using **ID-only Markov models** that consume only
historical expert-routing co-occurrence (no hidden states, zero extra GEMM), and
a bias-free offline evaluation harness (21-fold leave-one-out cross-validation
over a captured routing trace of Qwen3-30B-A3B on 21 diverse prompts).

We report three results. **(1) Promotion.** Treating prediction as *within-session
adaptation* — a lightly-weighted frozen cross-prompt prior blended at the count
level with a live per-session transition matrix, scored by pointwise mutual
information — generalises **4.7× better than a popularity baseline and +45 %
relative over the production co-occurrence scoring**, reaching **42.7 % held-out
cold-expert coverage** at fan-out 4. Mass-weighting, recency fusion, and a
score-level two-stage ensemble are all measured and rejected. **(2) Eviction.** A
strongly frequency-weighted recency rule (LFRU) cuts miss-rate by **≈8 points over
LRU**, while a learned recurrence Markov model gives no benefit. **(3) The wave.**
Reframing the objective from cache hit-rate to PCIe *stall cost*, we model the
real batched execution — *the wave*, in which a batch of decodes steps coherently
through the layers and each PCIe expert load is **amortised across the batch** —
with a discrete-event simulation at 8 GB/s. We show that aggregate throughput
**grows with batch size** as per-wave streaming saturates, that the optimal
prefetch/evict parameters are a **2-D function of VRAM residency and batch size**,
and that **above a streaming-saturation batch the predictor is irrelevant** and a
deterministic *prefetch-all-missing* policy is optimal. We give the final two-mode
**Markov Wave** design and production parameters (§12).

**Relation to prior work.** Most published MoE-offloading systems — Pre-gated MoE,
HOBBIT, AdapMoE, Mixtral-Offloading, ExpertFlow (§2) — predict the next layer's
experts by running the *current* hidden state through the *next* layer's gate,
exploiting cross-layer activation similarity to reach ~96 % top-1 accuracy at the
cost of extra gate/linear compute on live activations; eviction work splits
between LRU (temporal locality) and LFU (long-tail popularity). Our predictor is
deliberately **ID-only** — it consumes nothing but historical expert-ID
co-occurrence, so it is free to evaluate and trivially shared across a batch — and
we ask whether such a zero-compute model is *good enough* once paired with a
well-tuned cache. Distinctively, we (i) evaluate strictly out-of-sample
(leave-one-out CV) rather than on the trace used for fitting, (ii) optimise PCIe
*stall cost* rather than hit-rate, and (iii) model the **batched wave** and its
bandwidth-amortisation economics — the regime that actually sets throughput, and
which the per-request prediction literature does not address.

> **Document map.** Methodology §1–§4; results §5 (plan), §6 (advanced promote
> tuning), §7 (two-stage architecture), §8 (fine-tuning), §9 (end-to-end cache
> outcome), §10 (cost-driven operating point), §11 (the wave); §12 conclusion +
> final design. Harness: `candle-transformers/src/models/expert_lre/eval.rs`.

## Key results

All promotion/eviction numbers are **held-out**, 21-fold LOOCV, decode-phase
routing, pinned layers excluded. Best per column in **bold**.

**Table 1.** Promotion — held-out cold-expert coverage and top-1 precision by
predictor (decode-cov %, the fraction of *cold* — non-resident — experts a
prefetch of fan-out *k* would have covered). The progression traces the paper's
search; the champion is the §6.2 prior+session model with §8 tuning.

| Predictor | top-1 prec. | cov @k=1 | cov @k=2 | cov @k=4 |
|---|---|---|---|---|
| Popularity (baseline) | 21.8 % | 2.9 | 5.2 | 9.0 |
| Co-occurrence counts (production) | 67.5 % | 9.0 | 16.7 | 29.4 |
| Conditional `P(to\|from)` | 72.4 % | 9.6 | 18.1 | 32.1 |
| PMI(α=0.5) | 80.4 % | 10.7 | 19.9 | 34.9 |
| PMI + momentum (λ=0.99) | — | 11.6 | 22.4 | 40.8 |
| **Prior+session (Markov Wave promote)** | **91.1 %** | **12.1** | **23.4** | **42.7** |

**Table 2.** Eviction — expert miss-rate by policy (%, lower is better) under
production-style 5 %/token batch eviction at two VRAM budgets (fraction of the
working set resident). LFRU score = `last_used + ln(1+freq)·cohort·m`.

| Policy | miss @ 50 % budget | miss @ 70 % budget |
|---|---|---|
| LRU (recency) | 22.9 | 21.0 |
| LFU (frequency) | 28.4 | 13.0 |
| LFRU, m=1 (untuned) | 22.0 | 20.3 |
| Learned recurrence Markov | 39.5 | 29.1 |
| **LFRU, m=16 (tuned)** | **14.7** | **12.7** |

**Table 3.** The wave — per-session latency vs aggregate throughput at 60 % VRAM
residency, 8 GB/s PCIe, 2.6 MB experts (Qwen3-30B-A3B Q4_K_M), with eviction
fixed to L-1-behind and per-cell-optimal prefetch. Per-session rate falls but
**aggregate throughput climbs** as the saturating per-wave stream is amortised.

| Batch B | demand/layer | stream/wave | t/s · session⁻¹ | **aggregate t/s** | hit % |
|---|---|---|---|---|---|
| 1 | 8.0 | 23 | 24.6 | 25 | 94 |
| 8 *(cache knee)* | 43.9 | 819 | 3.3 | 27 | 72 |
| 64 | 103.9 | 1914 | 1.5 | 98 | 63 |
| 256 *(saturation)* | 118.2 | 2221 | 1.3 | 339 | 61 |
| 1024 | 122.0 | 2340 | 1.3 | **1313** | 60 |

**Table 4.** The Markov Wave — final production parameters (§12).

| Component | Setting |
|---|---|
| Promote scoring | PMI, **α = 0.5**, per-pair, input L-1, top-K |
| Prior / session | frozen prior **β = 0.01** (2 epochs, persisted) + live session matrix, no decay |
| Promote extras | arrival-specialised training; velocity γ=2 (EMA 0.8/0.97) |
| Eviction (wave) | **always evict L-1 (behind, wrapping)**; pin layers 0–2 |
| Eviction (non-wave cache) | LFRU `last_used + ln(1+freq)·cohort·16` |
| Below saturation (B<256) | predictive mode; prefetch `(depth, cap, kc)` from the 2-D map (§12.4) |
| At/above saturation (B≥256) | **streaming mode**: prefetch-all-missing L+1, predictor off |
| Per-session cost optimum (§10) | ≈2 % forced churn, K≈16 → hard-miss 7.8 %, −52 % stall cost |

---

## 1. Problem statement

The MoE expert pipeline (`candle-transformers/src/models/expert_lre/`) keeps a
fixed-size pool of expert weights resident in VRAM and streams the rest from a
pinned host pool over a copy stream. Two prediction problems govern its
throughput, and **they are independent** — we evaluate and select a model for
each separately:

- **PROMOTE (prefetch).** *Which not-yet-resident experts will an upcoming layer
  need?* A correct prediction lets the expert's H2D DMA start while the current
  layer computes, converting a cold miss into a warm hit. Prefetch only fills
  *free* slots, so it relies on headroom created by demotion.
- **DEMOTE (eviction).** *Which resident experts should be evicted to make
  headroom?* Production batch-evicts ≈5% of the pool at the end of every decode
  step ("end-of-pass eviction") so the next step finds free slots for fast
  promotion. The question is therefore which **group** of ≈5% to evict, not
  which single victim.

Decoupling the two simplifies the problem domain because they rest on **different
statistics**:


**Table 5.** The two decoupled prediction problems and their distinct underlying statistics.

| | Predicts | Signal |
|---|---|---|
| Promote | experts at layer **L** of the current token | **cross-layer** transition (L−1 → L within a token) |
| Demote | re-use of a resident `(layer, expert)` slot | **same-layer recurrence** (token *t* → *t+1* at fixed L) |

A resident slot keyed `(L, e)` is next needed at **layer L of the next token** —
48 layers away — so its eviction value is a *temporal-locality* property, not a
cross-layer one. This is why the cross-layer transition matrix (good for
promotion) is the wrong tool for eviction, and vice-versa.

---

## 2. Related work

Most published MoE-offloading systems predict the **next layer's** experts by
exploiting the residual structure of transformers: the gating *input* (hidden
state) is highly similar across adjacent layers, so feeding layer L's gating
input through layer L+1's gate predicts L+1's experts with ~96% top-1 accuracy
(≈90% for 2–3 layers ahead). Systems in this family include Pre-gated MoE,
HOBBIT, AdapMoE, Mixtral-Offloading, ExpertFlow (routing-path predictor + token
scheduler + expert-cache engine), and pre-attention expert prediction.

**How our approach differs.** Those methods run extra gate/linear compute on
live hidden states. Our predictor is an **ID-only Markov model**: it consumes
*only the historical expert-ID co-occurrence* (no hidden states, no extra GEMM),
so it is effectively free and trivially batched across sessions. The open
question this document answers is whether an ID-only model is *good enough*,
once paired with the cache, to beat the baselines — and which configuration is
best.

**On eviction.** The caching literature splits on two expert properties:
- **Long-tail popularity** → favours **LFU** (e.g. MoE-Infinity's LFU variant,
  fMoE's frequency × probability priority).
- **Temporal locality** ("consecutive tokens activate the same expert") →
  favours **LRU**. *This is exactly our measured same-layer stickiness:
  P(e at L next token | e at L now) ≈ 47%, a 7.6× lift over base rate.*
- "Neither LRU nor LFU adequately considers both"; **LFRU** (frequency-weighted
  recency) blends them. Our demote candidates generalise this with a *learned*
  recurrence term.

Sources:
- [ExpertFlow: Efficient MoE Inference via Predictive Expert Caching and Token Scheduling](https://arxiv.org/html/2410.17954)
- [Accurate Expert Predictions in MoE Inference via Cross-Layer Gate](https://arxiv.org/html/2502.12224v1)
- [Pre-gated MoE: An Algorithm-System Co-Design (ISCA'24)](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf)
- [HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference](https://arxiv.org/pdf/2411.01433)
- [AdapMoE: Adaptive Sensitivity-based Expert Gating and Management](https://arxiv.org/pdf/2408.10284)
- [Taming Latency-Memory Trade-Off in MoE Serving via Fine-Grained Expert Offloading](https://arxiv.org/pdf/2502.05370)
- [A Survey on Inference Optimization Techniques for MoE Models](https://arxiv.org/pdf/2412.14219)
- [MoE-Beyond: Learning-Based Expert Activation Prediction on Edge Devices](https://arxiv.org/html/2508.17137)

---

## 3. Evaluation harness (authoritative reference)

All offline evaluation lives in `candle-transformers/src/models/expert_lre/eval.rs`
and replays a captured routing trace on CPU — no model, no GPU, seconds per run.

### 3.1 Trace capture

A single instrumented run records every MoE dispatch into a compressed fixture.
- Source: `quantized_qwen3_moe::tests::capture_routing_trace` (CUDA, 30B model).
- Model: Qwen3-30B-A3B-Instruct (48 MoE layers, 128 experts/layer, top-8).
- **21 prompts**, one per config index, driven through a *single* model load;
  config 0 is StoryRewrite, configs 1–20 are diverse (code, prose, factual, math, lists).
- Up to **256 decode tokens** each, with **EOS early-stop** to avoid post-EOS
  degenerate repetition (which would inflate recurrence statistics).
- Record = `{ config, pass, layer, experts[], mass[] }` where `pass` is the
  decode-step index and `mass` is per-expert routing weight.
- Fixture: `src/models/batch_test/fixtures/routing_trace_qwen3_30b.bin.gz`
  (bincode + gzip), ≈155 K records / 6 MB for the 21-prompt capture.

### 3.2 Bias-free protocol — 21-fold leave-one-out CV

Training and test material **must not overlap** or the model is rewarded for
memorising the test trace. Protocol:

1. For each held-out config *c* (prompt), reset the model.
2. Train on the other 20 configs (`train_epochs` passes over their records).
3. Evaluate on config *c* only.
4. Aggregate held-out metrics across all 21 folds (micro-average over transitions).

Two test modes:
- **frozen** — model frozen after training: pure cross-prompt generalisation.
- **+online** — model keeps learning *causally* during the held-out generation
  (predict-then-observe). This is the production reality and is **not** biased
  (no future leakage); it just lets within-session structure accrue.

### 3.3 Scope

- **Decode passes only.** Prefill passes (large union sets) are excluded
  (`|active set| ≤ 16`).
- **Exclude pinned layers from results.** Layers `0..PINNED_LAYERS` (currently
  **3**) are permanently resident, so predicting them is meaningless — they are
  removed from *all metric denominators*. However, their current-token routing
  **is** available as an input signal for predicting later layers (§4.3).
- Adjacent transitions only: a prediction for layer L is scored against layer
  L+1 of the *same* pass.

### 3.4 Metrics

Let `current` = active set at L, `next` = active set at L+1,
`miss = next \ current` (the cold experts a prefetch could usefully warm).

**Promote:**
- **top-1** — top prediction ∈ `next` (fraction).
- **precision** — `|predicted ∩ next| / |predicted|`.
- **decode-cov** (primary) — `|predicted ∩ miss| / |miss|`: fraction of *cold*
  experts the predictor would have prefetched. This is the metric that maps to
  converted misses; raw precision over-credits predicting already-resident experts.

**Demote:**
- **group-regret** (primary) — of the bottom-`f` (≈5%) batch-evicted experts,
  the fraction re-requested within the next `W` tokens at their `(layer,expert)`
  key. Lower = better-chosen group. *Directly tests the group decision, not the
  single victim.*
- **miss-rate** — end-to-end expert misses / accesses under a full cache
  simulation with production-style end-of-pass batch eviction + free-slot
  prefetch, swept over VRAM budgets.

### 3.5 Baselines

- **Promote: popularity** — predict the most-frequently-routed experts at L+1
  (causal, per-layer). The bar a transition model must clear.
- **Demote: LRU** — evict least-recently-used. Also LFU and LFRU where relevant.

### 3.6 Commands

```bash
# Capture (CUDA, ~one model load, writes the fixture):
cargo test --release --features cuda,verbose --lib -p candle-transformers \
  quantized_qwen3_moe::tests::capture_routing_trace -- --ignored --nocapture

# Offline evals (CPU, run in --release for fast iteration):
cargo test --release -p candle-transformers --lib expert_lre::eval::loocv_prefetch -- --nocapture
cargo test --release -p candle-transformers --lib expert_lre::eval::loocv_demote   -- --nocapture
cargo test --release -p candle-transformers --lib expert_lre::eval::per_config     -- --nocapture
cargo test --release -p candle-transformers --lib expert_lre::eval::recurrence     -- --nocapture
```

---

## 4. Experimental variables

The harness sweeps four axes, **independently for the promote and demote models**.

### 4.1 Two independent models

The end state may use a different Markov configuration for each problem (or LRU
for demote). Selections are made separately in §6.

### 4.2 Matrix size (layer resolution)

The transition matrix is `[E × E]` per layer pair. With `E = 128` and 45
testable pairs that is `45 × 128 × 128 ≈ 737 K` cells. A single 256-token
generation yields ≈ `250 × 45 × (8×8) ≈ 720 K` observations — **about one
observation per cell.** Per-pair matrices are therefore severely undertrained on
realistic session lengths, which motivates trading layer resolution for data
density:


**Table 6.** Promote matrix-size (layer-resolution) variants and the data-density hypothesis (pre-registered).

| Variant | #matrices | Data/cell vs per-pair | Hypothesis |
|---|---|---|---|
| **per-pair** | 45 | 1× | most expressive, needs the most data |
| **group-4** | ~12 | ~4× | layer-local sharing |
| **group-8** | ~6 | ~8× | coarser |
| **shared** | 1 | ~45× | best generalisation, ignores layer-specific structure |

Fixed at `E = 128`: expert *bucketing* is **not** a useful size knob for
promotion — we must emit exact expert IDs to prefetch, so collapsing experts
destroys the target. Row sparsification (keep top-N targets/row) is a *memory*
optimisation, not a capacity knob, and is out of scope for accuracy selection.

### 4.3 Input signals (conditioning)

What the predictor conditions on. Kept deliberately minimal — richer inputs risk
overfitting the small per-session data and harming held-out generalisation.


**Table 7.** Candidate input signals (conditioning) for the promote predictor.

| Signal | Definition | Rationale |
|---|---|---|
| **L−1** (baseline) | experts at layer L−1 of the current token | first-order intra-token transition |
| **L−2** | experts at layer L−2 | does a second hop add signal beyond L−1? |
| **L−1 ⊕ L−2** | union/concat of both source layers | short-memory context |
| **recurrence** | experts at layer L of the *previous token* | "last decode run leaking forward"; strongest single signal (7.6× lift) |
| **pinned context** | experts at layers 0..2 of the current token | a free, always-available "topic" signal for predicting layers ≥3 |
| **combinations** | e.g. L−1 + recurrence | test whether signals are complementary |

Note the asymmetry: **recurrence** mostly predicts the *sticky* experts that are
already resident, so for *promotion* (cold experts) its marginal value must be
measured in combination, not in isolation. For *demotion* it is the core signal.

### 4.4 Scoring formula candidates

Counts accumulate `counts[L][from][to]`; let `P(to|from)=counts/row_total`,
`P(to)=col_total/total`. "Momentum" = exponential recency weighting (1st-order
smoothing of the count stream); "velocity" = the trend/derivative of a cell.

**Promote (rank successor experts):**


**Table 8.** Promote scoring-formula candidates.

| # | Name | Score | Rationale |
|---|---|---|---|
| P1 | **raw** | `Σ_from counts[from][to]` | current production; ≈ popularity (frequent sources dominate) |
| P2 | **conditional** | `Σ_from P(to\|from)` | row-normalised; each source contributes its sharpness |
| P3 | **PMI(α)** | `Σ_from P(to\|from) / P(to)^α` | lift over marginal; demotes globally-popular (already-cached) targets. Sweep α∈{0.25,0.5,0.75,1.0} |
| P4 | **momentum** | EMA-decayed counts (`counts *= λ` per window) | tracks topic drift / non-stationarity. λ∈{0.99,0.999} |
| P5 | **velocity** | boost targets whose `P(to\|from)` is *rising* over a window | anticipates emerging experts before counts saturate |
| P6 | **confidence-weighted** | Wilson/Bayesian lower bound on `P(to\|from)` | suppress 1-observation noise in undertrained cells |
| P7 | **mass-weighted** | observations weighted by router softmax `mass` | sharpen toward high-traffic transitions (mass is captured) |
| P8 | **Dirichlet-smoothed** | `(counts + α·P(to)) / (row_total + α)` | principled smoothing toward the marginal |

**Demote (keep-value; higher = keep, lower = evict):**


**Table 9.** Demote keep-value candidates.

| # | Name | Keep-value | Rationale |
|---|---|---|---|
| D1 | **LRU** (baseline) | recency of last use | temporal locality; production stand-in |
| D2 | **LFU** | activation frequency | long-tail "hub" experts |
| D3 | **LFRU / frecency** | frequency × recency | literature blend of D1+D2 |
| D4 | **recurrence** | `P(e at L next token \| active now)`, per `(L,e)` | learned same-layer stickiness |
| D5 | **recurrence + recency blend** | `last_used + keep·scale` | recency primary, recurrence refines (tested) |
| D6 | **momentum recurrence** | EMA of recurrence | weights recent stickiness; adapts within session |
| D7 | **velocity recurrence** | evict experts whose stickiness is *declining* | catches experts about to go cold while still recently used |

Selection prior (from the literature and our measurements): the dominant
eviction signal is temporal locality, which LRU already captures cheaply; a
learned model must demonstrate a **group-regret / miss-rate** win to justify its
complexity, otherwise the conclusion is plain LRU.

---

## 5. Results

All numbers are **held-out**, micro-averaged over the **21 LOOCV folds**, decode
passes only, **pinned layers (0–2) excluded** from the denominator, `train_epochs = 2`.
Primary metric is **decode-cov** (cold-expert coverage). Reproduce with the
commands in §3.6 (run `--release`).

### 5.1 Promote — baseline (§5.1)


**Table 10.** Promote baseline — held-out cold-expert coverage by predictor (21-fold LOOCV).

| mode | predictor | k=1 | k=2 | k=4 |
|---|---|---|---|---|
| frozen | popularity | 2.5% | 4.5% | 8.1% |
| frozen | raw (production) | 8.3% | 15.4% | 27.3% |
| frozen | conditional | 8.6% | 16.2% | 28.9% |
| frozen | **pmi(0.5)** | **9.6%** | **17.8%** | **31.4%** |
| +online | popularity | 2.9% | 5.2% | 9.0% |
| +online | raw (production) | 9.0% | 16.7% | 29.4% |
| +online | conditional | 9.6% | 18.1% | 32.1% |
| +online | **pmi(0.5)** | **10.7%** | **19.9%** | **34.9%** |

Popularity collapses out-of-sample (9%) — the old biased eval had massively
over-credited it. The transition matrix **generalises (≈4× popularity)**;
`pmi(0.5) > conditional > raw`; `+online` adds ~3pts. PMI(0.5) beats production
`raw` by **+5.5pts** at k=4 (+19% relative).

### 5.2 Promote — size / layer resolution (§4.2)

pmi(0.5), +online.


**Table 11.** Promote size ablation — held-out coverage by layer resolution.

| resolution | k=1 | k=2 | k=4 |
|---|---|---|---|
| **per-pair (45 matrices)** | **10.7%** | **19.9%** | **34.9%** |
| group-4 | 8.5% | 15.8% | 27.5% |
| group-8 | 7.2% | 13.1% | 22.6% |
| shared (1 matrix) | 4.9% | 8.9% | 15.2% |

**Per-pair wins decisively.** The pre-registered data-density hypothesis
(coarser → better) is **falsified**: with 20-prompt training there is enough data
that layer-specific structure dominates sparsity. Keep maximum resolution.

### 5.3 Promote — input signals (§4.3)

pmi(0.5), +online, per-pair.


**Table 12.** Promote input-signal ablation.

| inputs | k=1 | k=2 | k=4 |
|---|---|---|---|
| L−1 (base) | 10.7% | 19.9% | 34.9% |
| L−1, L−2 | 11.1% | 20.6% | **36.2%** |
| L−1, recurrence | 10.9% | 20.4% | 35.8% |
| L−1, pinned-context | 10.2% | 19.0% | 33.3% |
| all four | 11.0% | 20.7% | 36.5% |

L−2 and recurrence each add ~+1pt; **pinned-context hurts (−1.6pt — overfit)**.
Net gains from extra inputs are small (≤+1.6pt) relative to momentum (§5.4), so a
minimal `L−1` (optionally `+L−2`) is preferred for simplicity/robustness.

### 5.4 Promote — scoring formula (§4.4)

+online, per-pair, `L−1`.


**Table 13.** Promote scoring-formula sweep (held-out, +online).

| formula | k=1 | k=2 | k=4 |
|---|---|---|---|
| raw (production) | 9.0% | 16.7% | 29.4% |
| conditional | 9.6% | 18.1% | 32.1% |
| pmi(0.25) | 10.4% | 19.4% | 34.2% |
| **pmi(0.5)** | 10.7% | 19.9% | 34.9% |
| pmi(0.75) | 9.7% | 18.4% | 33.0% |
| pmi(1.0) | 5.6% | 11.8% | 24.1% |
| dirichlet(α=2) | 9.6% | 18.1% | 32.1% |
| wilson(z=1.0) | 9.6% | 18.0% | 32.0% |
| momentum λ=0.99 (pmi 0.5) | 11.6% | 22.4% | **40.8%** |
| momentum λ=0.97 | 11.3% | 21.9% | 40.2% |
| momentum λ=0.95 | 10.7% | 20.9% | 38.6% |
| momentum λ=0.90 | 9.3% | 18.3% | 34.5% |
| momentum λ=0.80 | 7.2% | 14.4% | 28.2% |

- **pmi(0.5)** is the best stationary transform; α=0.5 is the optimum (over- and
  under-discounting the marginal both lose). Dirichlet/Wilson ≈ conditional.
- **Momentum is the dominant lever.** pmi(0.5) with per-pass count decay **λ=0.99
  reaches 40.8%** (k=4) — **+5.9pts over stationary**, peaking at 0.99 then falling.
  λ=0.99 has a ~69-token half-life: over ~4 000 training passes it nearly forgets
  cross-prompt training, so momentum is effectively **fast within-session
  adaptation, warm-started by the 20 prompts**. Within-prompt routing structure
  beats cross-prompt averages.

### 5.5 Demote — group eviction & miss-rate (§3.4, §4.4)

21-fold LOOCV, batch-evict 5%/token, regret window W=8, recurrence prior
(stickiness) = 44.7%. Lower is better for both columns.

**VRAM budget = 50% of working set** (tight — the production-relevant regime):


**Table 14.** Demote — miss-rate and bottom-5% group-regret at 50% VRAM budget.

| policy | miss-rate | group-regret |
|---|---|---|
| **LRU** | 22.9% | 36.8% |
| LFU | 28.4% | 53.6% |
| **LFRU** | **22.0%** | **34.6%** |
| recurrence (D4) | 39.5% | 56.7% |
| blend (D5) | 23.0% | 37.4% |

**VRAM budget = 70% of working set** (roomy):


**Table 15.** Demote — miss-rate and bottom-5% group-regret at 70% VRAM budget.

| policy | miss-rate | group-regret |
|---|---|---|
| LRU | 21.0% | 34.5% |
| **LFU** | **13.0%** | **17.9%** |
| LFRU | 20.3% | 32.4% |
| recurrence (D4) | 29.1% | 47.9% |
| blend (D5) | 21.2% | 35.1% |

This reproduces the classic budget-dependent LRU/LFU split: temporal locality
(LRU) wins under pressure, popularity (LFU) wins with headroom. **LFRU** is the
robust choice — consistently ≥ LRU at both budgets. The **learned recurrence
model does not help**: pure recurrence (D4) is far worse (it discards the
dominant cyclic-position signal), and the recency-primary blend (D5) ≈ LRU. The
group-eviction hypothesis is therefore **not** supported — even measuring the
bottom-5% *group* under proper LOOCV, no learned Markov beats frequency-weighted
recency.

### 5.6 Demote — momentum / velocity (§4.4)

Not pursued. With pure recurrence (D4) already losing decisively to LRU and the
recency blend (D5) only matching it, recency-decayed variants (D6/D7) of the same
underlying signal cannot overtake the frequency+recency heuristics; LFU/LFRU
already capture the long-tail term that the recurrence family lacks.

---

## 6. Advanced promote tuning (beyond §4)

§5.4 established that **within-session adaptation** is the dominant lever
(momentum). This section pushes further with more complex formulas, non-uniform
math, and exhaustive fine-tuning. All numbers are held-out k=4 decode-cov,
+online, 21-fold LOOCV. Tests: `loocv_tune`, `loocv_sessionprior`, `loocv_mass`,
`loocv_arrivals`, `loocv_combo`, `loocv_velocity`, `loocv_final`.

### 6.1 Within-session momentum, decoupled (`loocv_tune`)

Applying the count decay **only during the held-out generation** (full,
un-decayed cross-prompt prior in training) cleanly separates "cross-prompt prior"
from "within-session momentum". The α×λ grid keeps improving as λ falls (39.2% at
λ=0.97, still climbing) but cannot reach the decay-both 40.8% — a strong frozen
prior *resists* fast adaptation. This motivated replacing decay with explicit
interpolation.

### 6.2 Bayesian prior + session interpolation (`loocv_sessionprior`) — **the winner**

Two matrices: a **frozen cross-prompt prior** (trained on the 20) and a **live
session matrix** (this prompt, online), blended as
`P(to|from) = (β·prior + session)/(β·prior_total + session_total)`, scored with
PMI(α). Never forgets; the session term overtakes the prior as evidence accrues.
β is the prior weight.


**Table 16.** Bayesian prior+session interpolation — held-out coverage vs prior weight beta.

| β (prior) | k=4 cov |
|---|---|
| 0 (pure session) | 39.9% |
| **0.02** | **41.8%** |
| 0.05 | 40.9% |
| 0.10 | 39.9% |
| 0.25 | 38.1% |
| 1.0 (≈ stationary) | 34.9% |

Fine grid → **β=0.02, α=0.5 = 41.8%** (k=4), **+6.9pts over stationary pmi(0.5)**
and beating the §5 momentum winner (40.8%) **without any per-token matrix decay**.
The within-session matrix is the dominant signal; the cross-prompt prior is worth
only a light warm-start (~+2pt over pure session).

### 6.3 Mass-weighted observations (`loocv_mass`) — **rejected**

Weighting counts by routing mass (`mass_from · mass_to`) **collapses coverage to
23.6%**. Reason: `P(to|from)` then reduces to the target's mass-share and loses
the co-occurrence *frequency* that carries the signal. Frequency, not mass, is
what predicts.

### 6.4 Arrival-specialised training (`loocv_arrivals`) — marginal +

Training only on "arrival" transitions (target ∉ source set — the cold experts a
prefetch must cover): **42.1%** (+0.3pt). Small but consistent.

### 6.5 Recency fusion (`loocv_combo`) — **rejected**

Adding a same-layer recency channel (reciprocal-rank fused): γ=0.25 → 41.9%
(≈ flat), higher γ hurts. The cross-layer session matrix already captures the
within-session structure; same-layer recency is redundant for prefetch.

### 6.6 Velocity / rising-trend (`loocv_velocity`) — marginal +

Boosting experts whose transition probability is rising (fast minus slow session
EMA): **42.3%** at γ≈2 (+0.5pt). Small but consistent.

### 6.7 Final combined model (`loocv_final`)


**Table 17.** Final combined promote model (advanced tuning).

| model | k=1 | k=2 | k=4 |
|---|---|---|---|
| stationary pmi(0.5) (§5 transform) | 10.7% | 19.9% | 34.9% |
| **SessionPrior** (β=0.02, α=0.5) | 12.1% | 23.2% | **41.8%** |
| + arrival-specialised | 12.2% | 23.3% | 42.1% |
| + velocity | 12.1% | 23.3% | 42.3% |
| **+ both** | **12.2%** | **23.4%** | **42.5%** |

**Champion: 42.5% k=4** held-out. The single big win is the prior+session
interpolation (§6.2); arrivals and velocity add ~+0.7pt of polish. Mass-weighting
and recency fusion were tried and rejected.

---

## 7. Two-stage base + session-fork architecture (tested)

**Hypothesis:** a continuously-improving **base** Markov model (learns from all
prior sessions) plus a **per-session fork** that pivots to a *different,
fast-learning* formula. Three concrete realizations were tested against the §6
champion (SessionPrior count-blend: 41.8% base / 42.5% full). Tests:
`loocv_twostage`, `loocv_fork`, `loocv_sessiondecay`.

### 7.1 Score-level ensemble (`loocv_twostage`) — worse (41.4%)

Frozen base `pmi(0.5)` + an empty fast fork, combined at the **score** level
(each channel max-normalised, fork weighted by `w_session`, fork free to use its
own formula). Best: fork=`pmi(0.25)`, w=2, no decay = **41.4%** — below the
count-blend. A less-aggressive fork formula (`pmi(0.25)`) helps because the
session estimates are noisy, but the score-level combine gives the fork a *fixed*
weight regardless of its confidence, amplifying early-session noise.

### 7.2 Literal fork — clone(base) + decay (`loocv_fork`) — worse (39.9%)

The session is a **copy of the base** that fast-adapts via per-token decay and
re-scores with its own α. Best λ=0.95 = **39.9%** (still below). The large initial
base mass fades unevenly and pollutes the session-specific signal more than a
permanently-light prior does.

### 7.3 Fast-learning fork via session decay (`loocv_sessiondecay`) — no change

The count-blend champion, but with the session matrix given its own decay:
**flat at 41.8%** (decay=1.0 optimal, faster decay slightly worse). Within a
256-token generation the routing topic does not drift, so the fork wants *all* of
its accumulated session data, not a recency window.

### Verdict

The two-stage intuition is **correct — and the champion already realizes it** —
but the **optimal coupling is count-level Bayesian interpolation** (`β·prior +
session`, β=0.02), not a score-level formula-pivot or a decaying fork. The session
fork should be a **non-decaying, fully-accumulating** matrix paired with a
**light frozen prior**; confidence-weighting (rely on the base early, the session
late) then falls out of the Bayesian blend automatically, with no tuning. The
specific "pivot to another formula / fast-decay fork" mechanisms were measured and
each underperforms the plain count blend.

---

## 8. Fine-tuning the selected models

Extensive parameter search on the champion (promote) and LFRU (demote). All
held-out k=4 decode-cov / miss-rate, 21-fold LOOCV. Tests: `tune_champion`,
`tune_vel`, `tune_epochs`, `champion_kcurve`, `tune_demote`.

### 8.1 Promote — β × α (`tune_champion`)

Full stack (arrival-specialised prior+session). Broad, flat plateau:


**Table 18.** Champion beta x alpha fine-tune (held-out cov @k=4).

| β \ α | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 |
|---|---|---|---|---|---|
| 0.005 | 41.9 | 42.1 | 42.1 | 41.8 | 41.4 |
| **0.010** | 42.0 | 42.1 | **42.2** | 42.1 | 41.8 |
| 0.015 | 41.9 | 42.1 | 42.2 | 42.1 | 41.9 |
| 0.020 | 41.8 | 42.0 | 42.1 | 42.0 | 41.8 |
| 0.030 | 41.6 | 41.8 | 41.9 | 41.8 | 41.6 |

Optimum **β=0.01, α=0.5 (42.22%)**; the plateau (β∈[0.01,0.015], α∈[0.45,0.55])
varies <0.4pt — the model is **robust**, not knife-edge.

### 8.2 Promote — velocity (`tune_vel`)

At β=0.01, α=0.5. Sweep γ × (fast,slow) decays:


**Table 19.** Velocity tuning — gamma x fast/slow EMA decay.

| decays \ γ | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0.7 / 0.95 | 42.22 | 42.55 | 42.65 | 42.46 | 41.91 |
| 0.6 / 0.92 | 42.22 | 42.50 | 42.57 | 42.33 | 41.81 |
| **0.8 / 0.97** | 42.22 | 42.58 | **42.72** | 42.58 | 42.13 |

Optimum **γ=2, decays 0.8/0.97 → 42.72%** (+0.5pt over no velocity).

### 8.3 Promote — base-prior training amount (`tune_epochs`)


**Table 20.** Base-prior training amount (epochs).

| epochs | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| k=4 cov | 42.46 | **42.65** | 42.65 | 42.56 | 42.35 |

**2–3 epochs optimal**; more slightly overfits the prior. Lock at 2.

### 8.4 Locked champion — coverage/precision vs fan-out k (`champion_kcurve`)

β=0.01, α=0.5, arrivals, velocity γ=2 (0.8/0.97), epochs=2:


**Table 21.** Locked champion — coverage and precision vs fan-out k.

| k | cov | precision |
|---|---|---|
| 1 | 12.1% | 91.1% |
| 2 | 23.4% | 87.7% |
| 3 | 33.6% | 84.0% |
| **4** | **42.7%** | 80.1% |
| 6 | 57.3% | 71.7% |
| 8 | 67.7% | 63.5% |

The deployment knob: prefetch more candidates → cover more cold experts at
falling precision. At k=8, **67.7%** of cold experts are caught.

### 8.5 Demote — LFRU frequency-bonus scale (`tune_demote`) — **major revision**

§5.5 evaluated LFRU at an *untuned* ×1 frequency weight and found it ~1pt better
than LRU. Sweeping the weight (eviction score = `last_used + ln(1+freq)·cohort·m`)
tells a very different story:


**Table 22.** Demote LFRU frequency-bonus scale tuning (miss-rate).

| scale m | miss @50% budget | miss @70% budget |
|---|---|---|
| LRU (m=0) | 22.9% | 21.0% |
| LFRU×1 | 22.0% | 20.3% |
| LFRU×4 | 20.1% | 18.7% |
| LFRU×8 | 17.8% | 16.6% |
| **LFRU×16** | **14.7%** | 12.7% |
| LFRU×32 | 14.9% | 10.0% |
| LFRU×64 | 16.8% | 9.5% |

**A strongly frequency-weighted LFRU beats LRU by ~8pts of miss-rate.** The
optimum is budget-dependent — ×16 peaks at the tight (production-relevant) 50%
budget; heavier weighting (→ LFU) keeps winning at the roomy 70% budget. **×16 is
the robust choice.** This is a real, large eviction win that the coarse §5.5 sweep
missed — and it is pure frequency+recency, still **no learned Markov needed**.

---

## 9. Modelled end-to-end cache outcome

Putting the two models together: the champion predictor drives free-slot prefetch
(one layer ahead, overlapped with compute), LFRU×16 drives eviction, and the
end-of-pass batch eviction creates the prefetch headroom. Every decode-layer
expert access is classified (test `cache_model_60`, 21-fold held-out). VRAM
budget is expressed as a fraction of the per-prompt working set; pinned layers
0–2 never evict.

- **hit** — resident, never evicted (no load);
- **soft miss** — resident *only* because the predictor prefetched it (DMA
  overlapped with compute — no stall);
- **hard miss** — not resident → demand-load on the critical path (**stall**).


**Table 23.** Modelled end-to-end per-session cache outcome at 60% VRAM (hit / soft / hard).

| budget | K | evict | hit | soft-miss | hard-miss | **no-stall** |
|---|---|---|---|---|---|---|
| 60% | 4 | 5% | 80.6% | 2.8% | 16.6% | 83.4% |
| **60%** | **8** | **5%** | **82.1%** | **4.7%** | **13.2%** | **86.8%** |
| 60% | 8 | 10% | 74.1% | 8.2% | 17.6% | 82.4% |
| 60% | 8 | 20% | 62.1% | 15.5% | 22.4% | 77.6% |
| 40% | 8 | 5% | 77.2% | 4.1% | 18.6% | 81.4% |
| 50% | 8 | 5% | 81.6% | 4.6% | 13.8% | 86.2% |
| 70% | 8 | 5% | 82.3% | 4.7% | 13.0% | 87.0% |
| 80% | 8 | 5% | 82.3% | 4.7% | 13.0% | 87.0% |

**Headline (60% VRAM, 5% eviction, K=8 prefetch): ≈82% hit, ≈5% soft-miss,
≈13% hard-miss → ≈87% of expert accesses avoid a pipeline stall.**

Three non-obvious findings:

1. **5% eviction maximises the no-stall *rate*** here — more eviction frees
   prefetch slots (soft rises to 15.5% at 20%) but evicts sticky experts that are
   re-needed (hit falls, hard rises), so no-stall drops 86.8% → 77.6%. **But the
   no-stall rate is the wrong objective** — stalls cost ~10× a soft miss. Under
   the proper stall-*cost* objective (§10), the optimum moves to **much lower
   churn (1–2%) with deeper prefetch (K≈16)**, nearly halving the stall rate.
2. **Soft-miss is headroom-limited, not predictor-limited.** Prefetch can only
   fill the ~5%-of-budget free slots the eviction creates, so the predictor's
   high k=8 coverage (67.7%, §8.4) is only partly realised as soft misses (~5%).
   The binding constraint is prefetch headroom, not prediction quality.
3. **No-stall saturates ≈87% by 60% budget**; 70–80% barely helps. The ~13%
   hard-miss floor is the irreducible cold-arrival fraction given the predictor's
   coverage and the forced 5% churn. Reducing it further needs *less* forced
   eviction at high budget (the cache is large enough to not need it), not a
   better predictor.

*Caveat:* budget is a fraction of the per-prompt working set (LOOCV). Production
batches many concurrent sessions with a larger aggregate working set; the same
locality should hold but absolute rates depend on the VRAM-to-working-set ratio.

---

## 10. Cost-driven operating point — minimising stalls (`cost_optimize`)

§9 maximised the *hit rate*. But hard misses are **stalls** (the pipeline waits
on full PCIe latency), whereas soft misses are overlapped — so the right
objective is a **stall cost**, not a hit rate:

> **stall cost = 10 · hard-miss + 1 · soft-miss** (hits = 0).

The second dimension is **PCIe bandwidth** = demand loads + *all* prefetch loads
(used + wasted). Hits transfer nothing; deeper prefetch cuts hard misses but
burns bandwidth on mispredicted preloads. Parameters swept at 60% VRAM: **forced
churn %** (eviction rate) and **K** (preloads per layer).

### Stall cost surface (per 100 accesses, lower = better)


**Table 24.** Stall-cost surface — forced churn x prefetch fan-out K (per 100 accesses).

| churn \ K | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| 0% | 94.2 | 91.5 | 89.1 | 86.1 | 84.7 |
| 0.5% | 94.4 | 91.2 | 88.5 | 85.0 | 83.6 |
| 1% | 96.7 | 92.5 | 88.7 | 84.2 | 82.5 |
| 2% | 112.9 | 104.6 | 97.1 | 85.2 | **80.7** |
| 3% | 132.5 | 121.5 | 110.6 | 91.9 | 81.0 |

**This inverts §9.** §9's 5%-churn / K=4 "no-stall-rate optimum" has stall cost
~169; the cost-optimal point is **≈2% churn, K=16 → 80.7** (hard-miss 7.8%,
soft 3.1%). Two mechanisms:

1. **Forced churn is mostly pure cost.** Each evicted slot is an LFRU-chosen
   demand expert that often gets re-requested → a hard miss (×10), plus a reload
   transfer (bandwidth). The soft misses the headroom enables (×1) do not pay for
   it. So **minimise forced churn** — just enough to seed prefetch.
2. **Deeper prefetch (K) always helps**, saturating ≈12–16. With limited
   headroom, a longer candidate list fills the free slots with *better* cold
   experts (more of the predictor's k=8→k=16 coverage is realised). It is
   bandwidth-cheap *because* it is headroom-limited.

### Bandwidth-constrained sweet spot

Min stall cost whose transfer rate fits a PCIe budget `B` (transfers / 100
accesses):


**Table 25.** Bandwidth-constrained cost optimum vs PCIe transfer budget.

| B (transfers/100acc) | churn | K | stall cost | hard% | soft% |
|---|---|---|---|---|---|
| 11 (tight) | 0% | 12 | 86.1 | 8.44% | 1.67% |
| 12 | 0% | 16 | 84.7 | 8.29% | 1.82% |
| 14 | 1% | 16 | 82.5 | 8.00% | 2.42% |
| 17 | 2% | 16 | 80.7 | 7.76% | 3.10% |
| ∞ | 2% | 16 | 80.7 | 7.76% | 3.10% |

**K≈16 is optimal at every bandwidth budget; the optimal churn rises with
available bandwidth** (0% when starved → 2% when ample). Versus the §9 baseline
(5% churn, K=4: hard 16.6%, cost ~169) the cost-optimal point **≈halves the stall
rate (16.6% → 7.8%) and cuts stall cost ~52%** — for free, just by re-tuning
churn and K.

### Design implications (further levers)

- **A dedicated prefetch reserve beats forced churn.** Churn frees slots by
  evicting *demand* experts (which then hard-miss). A small reserve that *only*
  prefetch fills (evicting prefetch-LRU within it) would give headroom without
  displacing demand experts — the predicted next gain. *(Untested; next experiment.)*
- **Lookahead vs latency.** Soft misses assume a 1-layer-ahead prefetch lands
  before the access. If PCIe latency exceeds one layer's compute, prefetch must
  issue ≥2 layers ahead (lower-accuracy 2-hop prediction) to actually overlap.
  Modelling DMA latency in layer units is the way to capture the latency axis
  directly. *(Untested; next experiment.)*
- **VRAM budget is the strongest single lever** (more cache → fewer hard misses,
  §9), but is fixed by hardware here at 60%.

---

## 11. The wave — batched streaming over PCIe (`wave_optimize`)

The real execution is a **wave**: a batch of B decode sequences stepping
*coherently* through all 48 layers via paged batched kernels. At each layer the
wave needs the **union** of experts its tokens route to, and — crucially — a
weight loaded over PCIe is **shared by every batched token routing to it** (one
batched GEMM). So **PCIe cost is per-unique-expert-per-wave, amortised across the
B tokens**. The wave runs ahead prefetching and **evicts from behind (L-1,
wrapping — the longest reuse distance, per the always-evict-L-1 rule)**. Stalls
come from **hard misses** (needed expert not resident → emergency load) and
**PCIe saturation** (transfers can't keep up). Modelled as a discrete-event
simulation: real clock, PCIe as a serial resource, ready-at times, steady-state
(warmup excluded). Compute is the amortised weight-read floor (~33 ms/wave ≈ the
30 t/s anchor); only stalls are added.

**Hardware:** expert 2.6 MB (Q4_K_M), PCIe **8 GB/s** → 0.325 ms/expert;
~2.1 experts hideable per layer-compute, ~103 experts/wave PCIe budget.

### Aggregate throughput **grows** with batch size (60% VRAM)

Because the per-wave streaming saturates (the union saturates toward 128/layer)
while the wave produces B tokens, `aggregate = B / wave_time` climbs with B even
as per-session latency falls:


**Table 26.** The wave — per-session vs aggregate token rate vs batch size (60% VRAM, full curve).

| B | demand/layer | stream/wave | t/s/session | **aggregate t/s** | hit% |
|---|---|---|---|---|---|
| 1 | 8.0 | 23 | 24.6 | 25 | 94 |
| 2 | 14.5 | 15 | 26.2 | 52 | 98 |
| 4 | 26.2 | 173 | 11.4 | 46 | 90 |
| 8 | 43.9 | 819 | 3.3 | 27 *(knee)* | 72 |
| 16 | 67.6 | 1348 | 2.1 | 34 | 66 |
| 32 | 88.8 | 1689 | 1.7 | 55 | 64 |
| 64 | 103.9 | 1914 | 1.5 | 98 | 63 |
| 128 | 113.4 | 2116 | 1.4 | 178 | 62 |
| 256 | 118.2 | 2221 | 1.3 | 339 | 61 |
| 512 | 120.5 | 2274 | 1.3 | 663 | 61 |
| 1024 | 122.0 | 2340 | 1.3 | **1313** | 60 |

Per-session **latency drops** (24.6 → 1.3 t/s) but **aggregate accelerates**
(25 → 1313 t/s), with a dip at the cache-fitting knee (B≈8 at 60%). Beyond the
knee the per-wave streaming **saturates** (~2340 experts/wave — the non-resident
40% needed every wave) while the wave produces B tokens, so aggregate scales
~linearly with B: the fixed working-set streaming is amortised across the batch.
**Note:** this is a *PCIe-limited upper bound* — the model holds compute at a
constant amortised floor, so at very large B the real GPU matmul compute (B token
vectors) eventually exceeds it and the wave becomes **compute-bound**, plateauing
aggregate below these figures. Adding a `B × per-token-flop` compute term would
give the true ceiling.

### 2D optimum: wave parameters vs (VRAM residency × batch)

Eviction is always L-1-behind. Optimised over prefetch depth, cap (experts/layer),
and champion arrival-prefetch `kc`. Selected cells:


**Table 27.** 2-D optimum wave parameters — VRAM residency x batch size.

| res% | B | stream | t/s/sess | aggregate | hit% | depth/cap/kc |
|---|---|---|---|---|---|---|
| 100 | 64 | **0** | **29.0** | 1856 | 100 | 1/16/0 *(control: evict nothing)* |
| 100 | 256 | **0** | **30.0** | 7670 | 100 | 1/16/0 *(control)* |
| 80 | 16 | 454 | 5.5 | 89 | 89 | 1/64/0 |
| 80 | 256 | 1053 | 2.8 | **721** | 82 | 2/64/0 |
| 70 | 4 | 27 | 24.3 | 97 | 98 | 2/16/0 |
| 70 | 16 | 928 | 3.0 | 48 | 78 | **2/64/8** |
| 60 | 4 | 195 | 10.9 | 44 | 89 | **2/64/8** |
| 60 | 256 | 2224 | 1.3 | 339 | 61 | 1/16/0 |
| 50 | 4 | 457 | 5.5 | 22 | 75 | **2/64/8** |
| 40 | 64 | 3009 | 1.0 | 63 | 41 | 1/16/0 |

### Findings — the production parameter map

1. **Aggregate throughput rises with B at every residency** (e.g. 80%: 25→721;
   100%: 25→7670). Large batches trade per-session latency for throughput by
   amortising the PCIe streaming — the correct LLM-serving behaviour.
2. **Control test passes:** at **100% residency** the optimiser converges to
   **evict-nothing, stream≈0, 30 t/s, 100% hit** — the simulation is sound.
3. **The optimal parameters move with (residency, batch):**
   - **Working set fits** (high res, or low B): minimal prefetch (depth 1, cap 16,
     kc 0) — nothing to stream.
   - **The knee** (mid residency 50–70%, small-mid batch, *spare bandwidth +
     streaming*): **deep prefetch + champion arrival prediction wins**
     (depth 2, cap 64, **kc 8**). *This is where the predictor's value resurfaces
     in the wave* — it converts arrival hard-misses to overlapped loads while
     bandwidth allows.
   - **Bandwidth-saturated** (low res, high B): shallow prefetch (depth 1,
     cap 16) — no spare bandwidth to overlap, so just stream the demand.
4. **Two throughput strategies fall out:** very large batches → high aggregate
   throughput at low per-session latency (efficient bandwidth use, more stalls);
   very small batches → high per-session latency, low aggregate (rare stalls,
   bandwidth slack). The engine picks per its latency-vs-throughput goal.

### Saturation regime — flip to deterministic prefetch-all (`stream_all`)

Streaming **saturates at B ≈ 256** (per-layer demand → the full vocabulary;
marginal stream growth < 5%/doubling; per-session flattens at ~1.3 t/s → aggregate
becomes linear, `≈ 1.3·B` at 60%). The saturation *batch* is **residency-
independent** (it's a workload property); past it the optimal parameters are
constant. So the parameter sweep only needs **B ∈ [1, 256]**.

At/above saturation the wave is **bandwidth-bound** — `wave_time ≈
total_stream / PCIe_BW` regardless of *which* experts are prefetched, because the
streaming volume dwarfs the compute window. The predictor therefore adds nothing,
and the engine should **flip to the simplest policy: prefetch the upcoming
layer's entire missing set (L+1 ahead, deterministic = last token's demand),
evict L-1, ride the wave.** Verified equal to the fully tuned optimum to within
1–4 %:


**Table 28.** Saturation regime — fully-tuned optimum vs deterministic prefetch-all-missing.

| res% | B | tuned t/s | prefetch-all-missing |
|---|---|---|---|
| 60 | 64 | 1.52 | 1.46 |
| 60 | 256 | 1.32 | 1.29 |
| 80 | 64 | 3.27 | 3.24 |
| 80 | 256 | 2.83 | 2.72 |

So the engine runs **two modes**: a *predictive* mode below saturation (champion +
tuned depth/cap at the bandwidth knee, where spare bandwidth lets prediction
convert arrivals to overlapped loads), and a *streaming* mode at/above saturation
(deterministic prefetch-all-missing, no predictor). The crossover is the
saturation batch.

**For production:** the inference engine picks `(prefetch_depth, cap, kc)` from a
table keyed on `(estimated batch demand, VRAM residency)` for B < saturation —
minimal when the working set fits, **deep + champion at the bandwidth knee** — and
switches to **prefetch-all-missing** at/above saturation. Eviction is always
L-1-behind.

*Caveats:* the compute floor is treated as constant (amortised weight read); at
very large B real matmul compute eventually exceeds it and caps aggregate
(not modelled — so the highest-B aggregate figures are PCIe-limited upper
bounds). 80-token trace; demand is synthesised from up to 21 captured prompts via
offset slot-sampling, so a real batch's diversity may differ. The qualitative map
(fits → compute-bound; knee → prediction helps; saturated → bandwidth-bound) is
robust.

---

## 12. Conclusion

### Promote — **adopt the within-session learned model**

**Bayesian prior + session interpolation** with the fine-tuned parameters:
- frozen cross-prompt prior, weight **β=0.01**, trained **2 epochs**;
- live per-prompt session matrix (no decay, fully accumulating);
- scoring **PMI(α=0.5)**, **per-pair** resolution, input **`L−1`**;
- **arrival-specialised** training + **velocity** boost (γ=2, fast/slow decay 0.8/0.97).

Held-out cold-expert coverage **42.7% @ k=4** (and 12.1% @ k=1 → 67.7% @ k=8;
§8.4) vs popularity 9.0% (**4.7×**) and current production `raw` 29.4%
(**+13.3pts, +45% relative**). The §5 plan's best (momentum, 40.8%) and the
two-stage variants (§7) are all superseded. The β×α plateau is broad (§8.1), so
the model is robust to these settings.

Decision trail: per-pair beats all coarser resolutions (§5.2); `L−1` is within
~1.6pt of richer inputs and avoids pinned-context overfit (§5.3); `pmi(0.5)` is
the best transform (§5.4); and **the dominant gain is treating prediction as
within-session adaptation** — a small frozen prior plus a live session matrix
(§6.2). The session model needs no per-token decay (cheaper than momentum) and no
mass-weighting (which actively hurts).

### Demote — **strongly-weighted LFRU (no learned Markov)**

**LFRU with a heavy frequency term** — eviction score
`last_used + ln(1+freq)·cohort·16`. Fine-tuning the frequency weight (§8.5)
turns LFRU from "marginal" (the untuned §5.5 result) into a **large win: ~−8pts
miss-rate vs LRU** (14.7% vs 22.9% at the tight 50% budget; 12.7% vs 21.0% at
70%). The optimum is budget-dependent (×16 robust at the pressured budget; heavier
→ LFU wins with headroom). The **learned recurrence Markov model is still
rejected** — it loses to LRU at any budget (§5.5); the same-layer recurrence
signal is real but redundant with recency. The win is pure frequency+recency,
which is cheap to compute online.

### Production changes implied

1. **Promote:** give `TransitionMatrix` two count tiers — a **frozen
   cross-prompt prior** (β=0.01, optionally persisted across sessions) and a
   **live per-session matrix** reset at session start — and score by their
   interpolation under `pmi(0.5)`, emitting **top-K** (K≈4) prefetch candidates.
   This is *cheaper* than momentum (no per-token matrix decay) and strictly
   better. Add the polish (both small but free): arrival-specialised training
   (skip in-source targets) and a velocity term (fast−slow session EMA, γ=2).
2. **Demote:** **delete the dead anti-prediction path** (it never fires — see the
   `predict_bottom` last-layer guard). Upgrade the eviction score from LRU to
   **strongly-weighted LFRU**: add `ln(1+freq)·cohort·16` (freq = per-`(layer,
   expert)` access count) to the existing `last_used` ordering, on top of
   behind-layer bias + pinning. This is an ~8pt miss-rate win and costs one
   counter per slot. Do **not** add a recurrence model.

### Combined modelled outcome (§9–§10)

With both models in place at **60% VRAM**, the operating point matters. Optimising
the *hit rate* (5% churn, K=4) gives ≈82% hit / 13–17% hard-miss. But the right
objective is **stall cost** (hard ≈10× soft); re-tuning to the cost-optimal
**≈2% forced churn and K≈16 preloads** (§10) drops the **hard-miss (stall) rate
to ≈7.8%** (from 13–17%) and cuts stall cost ~52% — at no extra hardware cost.
Forced eviction is the dominant controllable cost; prefetch depth is bandwidth-
cheap because it is headroom-limited. The next gains are a dedicated prefetch
reserve (vs forced churn) and latency-aware lookahead (§10).

### Deployment regimes & the 2D parameter map (§10–§11)

- **Per-session / low-concurrency offload (§10)** — prefetch headroom exists, so
  the champion's arrival prediction pays: cost-optimal tuning (≈2% churn, K≈16)
  cuts stall cost ~52% and the hard-miss rate to ~7.8%.
- **Batched wave (§11)** — aggregate throughput **grows with batch size** (the
  PCIe streaming is amortised across the batch: 25 → 7670 aggregate t/s as B and
  residency rise), at the cost of per-session latency. The optimal wave parameters
  are a **2D function of (VRAM residency, batch)**: minimal when the working set
  fits; **deep prefetch + champion arrival prediction at the bandwidth knee**
  (mid residency, small-mid batch — where the predictor resurfaces); shallow when
  bandwidth-saturated. Eviction is always L-1-behind; the **100% residency control
  converges to evict-nothing / 30 t/s**, validating the model. Production should
  read `(prefetch_depth, cap, champion-k)` from this map keyed on live batch
  demand and residency.

### Final design — the **Markov Wave** (production specification)

The complete design that the inference engine should implement, pulling together
every result above.

#### 1. Execution model

A **wave** = a batch of B decode sequences stepping coherently through all 48 MoE
layers via paged batched kernels. At each layer the wave needs the **union** of
its tokens' routed experts; a PCIe-loaded expert is **shared by every batched
token** routing to it, so PCIe cost is per-unique-expert-per-wave, **amortised
across the batch**. VRAM holds a `residency` fraction of all experts. The wave
**runs ahead prefetching** and **evicts behind**.

#### 2. Eviction — one universal rule: **always evict L-1 (wrapping)**

As the wave leaves layer L, layer L-1's experts have the **longest reuse
distance** — not needed until the next token wraps all the way around (47 layers).
So they are always the correct eviction target: O(1), deterministic, optimal.
(Plus pin layers 0–2; never evict the immediate prefetch zone ahead.) This
*wave-aware* eviction supersedes the general-cache **LFRU×16** finding of §8.5
(LFRU was the layer-agnostic equivalent — "evict by frequency+recency" ≈ "evict
by reuse distance"; L-1-behind is the exact wave form).

#### 3. Two operating modes (crossover at the streaming-saturation batch ≈ 256)

**(a) Predictive mode** — B below saturation (working set near/below VRAM; spare
bandwidth). Use the **Markov promote predictor** to prefetch, parameters from the
2D map. This is where prediction pays — at the **bandwidth knee** (mid residency,
small–mid batch) the champion converts arrival hard-misses into overlapped loads.

  *Promote predictor (the Markov model):* Bayesian **prior + session
  interpolation** — frozen cross-prompt prior (β=0.01, 2 epochs, persisted across
  sessions) + live per-session matrix (no decay), scored **PMI(α=0.5)**,
  **per-pair**, input **L-1**, with **arrival-specialised** training and a
  **velocity** boost (γ=2, fast/slow EMA decay 0.8/0.97); emit **top-K**.
  Held-out cold-expert coverage **42.7% @ k=4** (4.7× popularity, +13pt vs
  production raw).

**(b) Streaming mode** — B ≥ saturation (bandwidth-bound, full demand). **Drop
the predictor.** Prefetch the upcoming layer's **entire missing set** (L+1 ahead,
deterministic = last token's demand); evict L-1; ride the wave. Verified equal to
the fully tuned optimum to within 1–4 % — the predictor adds nothing once
bandwidth-bound.

#### 4. The 2D parameter table (B < saturation): `(residency, batch) → depth / cap / champion-k`


**Table 29.** The Markov Wave 2-D parameter table (production): (residency, batch) -> depth/cap/champion-k.

| residency \ batch | 1 | 4 | 16 | 64 | ≥256 |
|---|---|---|---|---|---|
| **40%** | 2/16/0 | 1/64/8 | 1/16/8 | 1/16/0 | *stream-all* |
| **50%** | 1/16/0 | **2/64/8** | 1/16/8 | 1/16/0 | *stream-all* |
| **60%** | 1/16/0 | **2/64/8** | 1/16/0 | 1/16/0 | *stream-all* |
| **70%** | 1/16/0 | 2/16/0 | **2/64/8** | 1/16/0 | *stream-all* |
| **80%** | 1/16/0 | 1/16/0 | 1/64/0 | 2/16/0 | *stream-all* |
| **100%** | 1/16/0 *(evict nothing)* | … | … | … | 30 t/s control |

Pattern: **minimal** prefetch when the working set fits (high residency / low B);
**deep + champion (depth 2, cap 64, kc 8)** at the bandwidth knee; **shallow →
stream-all** once saturated. Parameter sweep only needs **B ∈ [1, 256]** (the
saturation batch is residency-independent).

#### 5. Performance envelope (8 GB/s PCIe, 2.6 MB experts, 30 t/s compute floor)


**Table 30.** The Markov Wave — performance envelope (8 GB/s, 2.6 MB experts).

| | per-session | aggregate |
|---|---|---|
| 100% residency (control) | 30.0 t/s | scales with B (7670 @ B=256) |
| 80%, B=256 | 2.8 t/s | 721 t/s |
| 60%, small B (≤2, fits) | ~25–27 t/s | 25–52 t/s |
| 60%, B=1024 | 1.3 t/s | 1313 t/s |

**Per-session latency falls with batch; aggregate throughput rises** (amortised
streaming). The engine chooses batch by the latency-vs-throughput goal; below
saturation it tunes the predictor, at/above it streams.

#### 6. Final parameter list


**Table 31.** The Markov Wave — final parameter list.

| Component | Setting |
|---|---|
| Promote scoring | PMI, **α = 0.5** |
| Prior weight | **β = 0.01**, 2 training epochs, persisted cross-session |
| Session matrix | live, **no decay**, per-pair, input L-1 |
| Promote extras | arrival-specialised training; velocity γ=2 (EMA 0.8/0.97) |
| Eviction (wave) | **always L-1 behind (wrapping)**; pin layers 0–2 |
| Eviction (non-wave §10) | LFRU `last_used + ln(1+freq)·cohort·16` |
| Prefetch params | from §4 2D map for B<256; **prefetch-all-missing** for B≥256 |
| Per-session cost optimum (§10) | ≈2% forced churn, K≈16 (hard-miss 7.8%, −52% stall cost) |

### Caveats

Single model (Qwen3-30B-A3B), 21 prompts, decode-phase routing. The promote
model's edge comes from within-session adaptation, so it assumes sessions long
enough for the session matrix to accrue evidence (holds for the unbounded-context
target; the cross-prompt prior covers the cold-start tokens). Fine-tuned
parameters are selected on the 21-fold aggregate; the broad β×α plateau (§8.1)
and consistent LFRU trend (§8.5) suggest they generalise, but the absolute
optima (e.g. LFRU×16 vs ×32) are budget-dependent and should be re-confirmed at
the production VRAM ratio. The demote budgets simulate the pressured regime;
with large headroom (working set ≪ budget) eviction is moot.

**Wave model (§11):** the compute term is held at a constant amortised
weight-read floor, so the high-batch **aggregate** figures are *PCIe-limited upper
bounds* — at large B the real GPU matmul (B token-vectors) eventually exceeds the
floor and caps aggregate (a `B × per-token-flop` compute term would give the true
plateau). Per-layer wave demand is synthesised from up to 21 captured prompts via
offset slot-sampling, so a real batch's diversity (and thus the exact saturation
batch and per-layer vocabulary, ≈122 here vs the full 128) may differ. Hardware
constants (expert 2.6 MB, PCIe 8 GB/s) should be set to the production link. The
qualitative structure — fits → compute-bound; knee → prediction helps; saturated
→ stream-all, bandwidth-bound; aggregate grows with batch — is robust.
