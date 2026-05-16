# Attentional Provenance Strategy Comparison

**Model:** Qwen3-30B-A3B  
**Date:** 2026-05-15  
**Harness:** `projection_harness` — 8 tools × 6 positive scenarios = 48 probes  
**Raw data:** `raw_kvq.prov` (1.24 GB), `RAW_MANIFEST.json`

---

## Setup

The test captures raw f32 K and Q vectors from three provenance bands during decode
(syntactic band centre layer 7, semantic centre layer 24, pragmatic centre layer 40,
each ±4 layers = 9 layers per band). For Qwen3-30B-A3B: n_kv_heads=4, head_dim=128.

Each **SignatureStrategy** defines how to binarise those vectors into a 128-bit
`TokenSignature`. The harness then runs BDP (Binary Directional Provenance —
XNOR+popcount agreement in [0,128]) between probe signatures and corpus signatures
to score each tool section.

Two scoring modes are evaluated:

- **Count**: every BDP hit above threshold contributes +1
- **Span(α)**: consecutive runs of probe tokens hitting the same section score L^α
  instead of L (run-length boosting)

The **discrimination ratio** is `intra_score / inter_mean_score`. Ratio > 1 means
the correct tool section outscores all others on average. **min_ratio > 1.0** means
it does so on *every* probe — the reliability bar.

---

## § 1 — Full Strategy Sweep (Count, equal depth weights, 48 probes)

Only strategies with min_ratio > 1.0 discriminate reliably across all probes.

| strategy | min_ratio | mean_ratio | max_ratio |
|---|---|---|---|
| **MH_XOR_QQ_l4** | **1.0652** | **1.2740** | 1.5330 |
| **MH_XOR_QQ_l8** | **1.0196** | **1.2087** | 1.4271 |
| MH_XOR_QQ_l0 | 0.9952 | 1.1973 | 1.4547 |
| QQ_l4_h1 | 0.6605 | 0.9547 | 1.8325 |
| QQ_l0_h1 | 0.6367 | 0.9411 | 1.7747 |
| QK_l4_h1 | 0.0000 | 2.4301 | 17.5000 |
| SimHash_l4_h1 | 0.0746 | 1.0417 | 3.1652 |

Selected others for comparison:

| strategy | min_ratio | mean_ratio | notes |
|---|---|---|---|
| QQ (all single-head) | 0.55–0.67 | 0.86–0.95 | below 1.0 mean — unreliable |
| KK (all single-head) | 0.53–0.57 | 0.87–0.92 | same |
| MH_Mean_QQ | 0.59–0.63 | 0.89–0.91 | mean worse than XOR |
| QK (per-head) | 0.00 | 0.65–2.43 | min=0 — fails on some probes |
| MH_XOR_QK | 0.00 | 0.25–1.10 | zero signal, 0/0 false-inf |
| BandMeanQQ | 0.55–0.58 | 0.86–0.90 | averaging kills signal |

**Top-5 by (min_ratio DESC, mean_ratio DESC):**  
`MH_XOR_QQ_l4 | MH_XOR_QQ_l8 | MH_XOR_QQ_l0 | QQ_l4_h1 | QQ_l0_h1`

> **Note on `inf` values:** Several `QK` variants show `inf` mean_ratio. This is an
> artifact of the ratio function when both intra and inter are zero — the strategy
> produces no BDP signal at all above threshold. Corrected ratio: 0/0 → 1.0 (no
> information). The `MH_XOR_QK` strategies are not discriminative.

---

## § 2 — WindowMeanQ Sweep

WindowMeanQ computes sign(mean(Q_{t-w}..Q_{t+w})) for the probe, single-token K for
the corpus. All window variants have min_ratio=0 across the 48-probe set — they fail
completely on at least one probe regardless of window size or head. Not recommended
for production.

---

## § 3 — Span Scoring Comparison (top-5 strategies from § 1)

Span scoring strongly boosts the MH_XOR_QQ family. Single-head QQ sees almost no lift.

| strategy | count_mean | span1.5_mean | span2.0_mean |
|---|---|---|---|
| **MH_XOR_QQ_l0** | 1.1973 | 2.1778 | **3.0688** |
| **MH_XOR_QQ_l4** | 1.2740 | 2.1035 | 2.8537 |
| MH_XOR_QQ_l8 | 1.2087 | 2.0196 | 2.7640 |
| QQ_l4_h1 | 0.9547 | 1.0625 | 1.1039 |
| QQ_l0_h1 | 0.9411 | 1.0382 | 1.0598 |

The layer ordering **reverses** between count and span:
- **Count** winner: `l4` (band centre, model layer 7) — most selective per token
- **Span** winner: `l0` (band start, model layer 3) — produces longer consecutive runs

Layer 0 Q vectors are smoother and more sequential (early-layer local syntax
processing). When the model is focused on a tool decision, consecutive decode tokens
at layer 3 all point in a correlated direction, forming runs. Layer 7 is more
token-selective — higher peak but shorter runs, making it better for count but worse
for span.

Span with α=2.0 gives **~2.5× better discrimination** than count alone
(3.07 vs 1.27 mean_ratio for MH_XOR_QQ).

---

## § 4 — Per-Tool Breakdown: MH_XOR_QQ_l4 (best count strategy, 48 probes)

| tool | mean_intra | mean_inter | mean_r | min_r |
|---|---|---|---|---|
| weather | 225.8 | 190.4 | 1.1859 | 1.1571 |
| web_search | 290.9 | 253.9 | 1.1457 | 1.0699 |
| **file_write** | 263.6 | 255.2 | 1.0327 | **0.9838** |
| **file_read** | 200.7 | 199.3 | 1.0070 | **0.9461** |
| code_run | 286.9 | 255.5 | 1.1229 | 1.0692 |
| datetime | 213.0 | 203.7 | 1.0458 | 1.0406 |
| calculator | 199.0 | 194.1 | 1.0252 | 0.9966 |
| random | 229.7 | 204.8 | 1.1212 | 1.0988 |

`file_read` and `file_write` are the hardest pair — naturally similar KV patterns,
their Q vectors partly overlap in sign space. Count scoring alone cannot reliably
separate them (min_ratio < 1.0 on some probes).

---

## § 5 — QK Layer × Head Sensitivity

QK mean_ratio across layers and heads — highly variable, no reliable head/layer:

| layer | h0 | h1 | h2 | h3 |
|---|---|---|---|---|
| 0 | 0.95 | 0.85 | **1.14** | 1.03 |
| 4 | inf* | **2.43** | 0.91 | 1.09 |
| 8 | 0.98 | 0.70 | 0.99 | 1.00 |

*inf at l4_h0 = genuine perfect discrimination on mean but min_ratio=0 (fragile).

FloatSimHash (random ±1 projection then sign) is consistently sub-1.1, slightly better
than single-head QQ but far below MH_XOR_QQ. The random projection adds no value over
the raw sign — the dimensions are already approximately independent.

---

## § 6 — BandMean Head Sensitivity

Averaging Q/K across all 9 layers within the band (`BandMeanQQ`) consistently scores
below 0.90 mean_ratio — worse than single-layer. Averaging destroys the sharp
layer-specific signal. Not recommended.

---

## § 7 — Span Per-Tool Breakdown: MH_XOR_QQ_l0 with α=2.0 (48 probes)

| tool | cnt_mean | sp1.5_mean | sp2.0_mean | cnt_min | sp2_min |
|---|---|---|---|---|---|
| weather | 1.2483 | 2.7288 | 4.5875 | 1.2408 | 3.5740 |
| web_search | 1.3283 | 2.7427 | 4.1776 | 1.2981 | 3.8980 |
| file_write | 1.3452 | 2.1273 | 2.6879 | 1.2669 | 2.4850 |
| **file_read** | 1.0253 | 1.6076 | 1.9099 | **0.9952** | **1.6127** |
| code_run | 1.3489 | 2.2956 | 3.2009 | 1.3069 | 2.8756 |
| datetime | 1.0674 | 1.9181 | 2.5238 | 1.0353 | 2.2337 |
| calculator | 1.0589 | 1.8590 | 2.4638 | 1.0272 | 2.1191 |
| random | 1.1564 | 2.1434 | 2.9987 | 1.1370 | 2.0639 |

With span α=2.0, **every tool exceeds ratio=1.0 on every probe** including
`file_read` (sp2_min=1.61). Count alone cannot achieve this for `file_read`
(cnt_min=0.9952 — just below the reliability threshold). At this stage the
recommended strategy was MH_XOR_QQ_l0 + span α=2.0; § 8 supersedes this.

---

## § 8 — Dual-Layer Combinations: l0 × l4 (48 probes)

§ 1–7 identified a structural tension: l0 (model layer 3) produces smooth,
sequentially-correlated Q vectors that form long runs (good for span), while l4
(model layer 7) produces sharper, more token-selective Q vectors (good for count).
No single layer dominates both modes. This section tests three algorithms for
combining both layers simultaneously.

### Algorithms

#### Algorithm A — Dual-Layer XOR Fold (MH_XOR_QQ_l0xl4)

Extend the single-layer 4-head XOR to an 8-head XOR across both layers. For each
token t at syntactic band layer b:

```
sig_hi(l) = sign(Q[t, band=b, layer=l, head=i])  ∈ {±1}^128

TokenSignature = sig_h0(l0) XOR sig_h1(l0) XOR sig_h2(l0) XOR sig_h3(l0)
               XOR sig_h0(l4) XOR sig_h1(l4) XOR sig_h2(l4) XOR sig_h3(l4)
```

The resulting 128-bit fingerprint encodes directional agreement across 8 independent
(head, layer) subspaces. Both probe and corpus tokens are binarised identically. BDP
scoring and span scoring proceed unchanged — the only difference is a richer input
fingerprint.

**Properties:** The 8-head fold is strictly harder to spuriously match than the 4-head
fold. A false positive requires sign-agreement across heads from two structurally
different transformer depths simultaneously. The l0 component preserves the
sequential correlation that creates span runs; the l4 component injects sharper
per-token discrimination. Both properties are captured in a single fingerprint,
so no scoring mechanism changes are needed.

#### Algorithm B — Normalised Additive Fusion

Run two independent BDP passes — l0 with span scoring, l4 with count scoring — and
combine the resulting per-section scores with per-probe normalisation:

```
span_score(section, l0)    = Σ L^α  over runs in hit_log(probe, section, l0)
count_score(section, l4)   = BDP mean agreement(probe, section, l4)

μ_span  = mean over all sections of span_score(·, l0)
μ_count = mean over all sections of count_score(·, l4)

combined(section) = span_score(section, l0) / μ_span
                  + count_score(section, l4) / μ_count

ratio = combined(target_section) / mean_{j ≠ target} combined(section_j)
```

Dividing by the per-probe mean normalises the two components to equal average
contribution, removing the scale mismatch between absolute span scores (~1–10) and
absolute BDP count scores (~150–300).

**Properties:** Independent scoring passes; no single fingerprint required. The
l4 count component penalises spurious per-token hits even when l0 span is high.
However, because the two passes operate on independent hit sets, the combination
misses the joint constraint that makes Algorithm A strong.

#### Algorithm C — Gated Span

Use l4 as a binary gate: a probe token at depth d contributes to the span score
for a section only if it produced a BDP hit in *both* the l0 and l4 passes:

```
gate_toks(section, depth) = {t : t ∈ hits(probe, section, l0, depth)}
                           ∩ {t : t ∈ hits(probe, section, l4, depth)}

score(section) = (1/3) Σ_{d ∈ {syn,sem,prag}}  Σ_{runs in gate_toks(·,d)} L^α

ratio = score(target_section) / mean_{j ≠ target} score(section_j)
```

**Properties:** The intersection strictly reduces the set of contributing probe
tokens relative to l0 alone. This suppresses spurious span starts that l0 allows
but l4 would reject. The cost is that genuine runs lose tokens where l4 happens
to miss — a particular risk for the hardest tool pairs (file_read/file_write) where
l4 count is already near-threshold.

---

### Results

| strategy | min_ratio | mean_ratio | max_ratio |
|---|---|---|---|
| MH_XOR_QQ_l0 count (baseline) | 0.9952 | 1.1973 | 1.4547 |
| MH_XOR_QQ_l4 count (baseline) | 1.0652 | 1.2740 | 1.5330 |
| MH_XOR_QQ_l0 span α=2.0 (baseline) | 1.6127 | 3.0688 | 5.7599 |
| **A: MH_XOR_QQ_l0xl4 count** | **1.4191** | **1.7465** | **2.1909** |
| **A: MH_XOR_QQ_l0xl4 span α=2.0** | **2.5284** | **5.3141** | **8.1998** |
| B: norm span α=2.0(l0) + count(l4) | 1.3541 | 2.0681 | 3.1075 |
| C: gated span α=2.0(l0, gate=l4) | 1.5343 | 3.3281 | 6.2253 |

All three dual-layer strategies exceed the single-layer baselines on mean_ratio.
Only **Algorithm A** improves the reliability floor (min_ratio) over the previous
best single-layer result (l0 span α=2.0: 1.61 → A span α=2.0: **2.53**).

### Per-Tool Breakdown: MH_XOR_QQ_l0xl4 with α=2.0

| tool | cnt_mean | sp2.0_mean | cnt_min | sp2_min |
|---|---|---|---|---|
| weather | 2.0006 | 5.4709 | 1.9399 | 4.6667 |
| web_search | 2.0824 | 7.5432 | 1.9924 | 6.8795 |
| file_write | 1.6976 | 5.6010 | 1.6022 | 5.0244 |
| **file_read** | 1.5112 | 3.0607 | **1.4191** | **2.5284** |
| code_run | 1.8714 | 6.8053 | 1.6984 | 5.5319 |
| datetime | 1.5895 | 4.6472 | 1.5086 | 3.5000 |
| calculator | 1.5403 | 3.5247 | 1.4455 | 2.5298 |
| random | 1.6789 | 5.8595 | 1.6473 | 4.6964 |

**Every tool, every probe: min_ratio > 1.0 under both count and span α=2.0.**

This is a qualitatively stronger result than the § 7 baseline: the previous champion
(l0 span α=2.0) needed span scoring to rescue file_read from a sub-threshold count
floor (cnt_min=0.9952). The dual-layer strategy clears the bar under count alone
(file_read cnt_min=1.42), making span a performance amplifier rather than a safety net.

### Why Algorithm A Dominates

**The joint-subspace argument.** The 4-head single-layer XOR fingerprint encodes
directional agreement across 4 independent attention heads from a single transformer
depth. Adding 4 more heads from a structurally distinct depth doubles the number of
independent constraints a false match must satisfy. Because the two layers specialise
in different computational roles — l0 in local syntactic patterning, l4 in mid-level
semantic composition — their sign patterns are statistically near-independent. A token
pair that spuriously agrees in l0's subspace is unlikely to also agree in l4's; the
joint probability of a false match falls roughly as the product of the two marginal
probabilities.

**Why C falls short of A.** Algorithm C's gating is logically equivalent to requiring
agreement in both l0 and l4 separately, but it applies this constraint only at the
token-hit level after independent thresholding. The intersection loses tokens where
one layer's BDP value is just below threshold — a high-variance process for the
hardest tool pairs. Algorithm A's XOR fold applies the joint constraint at the
fingerprint level, before thresholding, so partial agreement in both subspaces
compounds into a stronger combined signal rather than being discarded. The result
is that A's min_ratio (2.53) exceeds C's (1.53) despite both encoding the same
logical requirement.

**Why B falls short.** Algorithm B's additive fusion never achieves a joint
discriminative fingerprint — it always reasons about l0 and l4 as separate evidence
sources and sums their normalised ratios. This works better than either source alone
(B mean=2.07 vs baselines of 1.20–1.27) but misses the multiplicative suppression
of false positives that XOR-folding achieves. The per-probe normalisation also dilutes
the signal when one layer has very high inter-tool scores for a particular probe,
pulling the combined ratio toward the stronger layer's ratio rather than amplifying
their joint constraint.

---

## Conclusions

### Winner: MH_XOR_QQ_l0xl4 + Span α=2.0

The recommended production strategy is **MH_XOR_QQ at syntactic band layers 0 and 4
(model layers 3 and 7), 8-head XOR fold, with span scoring α=2.0**.

It is the only strategy that achieves min_ratio > 1.0 across all 48 probes and all
8 tools under **both** count and span scoring, including the hardest file_read/file_write
pair. The prior champion (MH_XOR_QQ_l0 + span α=2.0) achieved this under span only;
under count its file_read floor was 0.9952 — just below the reliability threshold.

### Signature Computation

**Per token, at syntactic band, layers 0 and 4:**

1. Read Q vectors for all 4 KV heads at layer 0: Q^0_h0..Q^0_h3 ∈ ℝ^128
2. Read Q vectors for all 4 KV heads at layer 4: Q^4_h0..Q^4_h3 ∈ ℝ^128
3. Binarise each: s^l_i = sign(Q^l_hi) ∈ {±1}^128
4. XOR-fold all 8 vectors:

```
TokenSignature = s^0_0 ⊕ s^0_1 ⊕ s^0_2 ⊕ s^0_3
               ⊕ s^4_0 ⊕ s^4_1 ⊕ s^4_2 ⊕ s^4_3
```

**Span scoring (α=2.0):** For each consecutive run of L probe tokens all hitting
the same section at a given depth, contribute L² to that section's score
(rather than L for count). Mean across the three depth bands gives the final score.

### Why This Works

**8-head XOR fold:** Each attention head specialises in a different feature subspace;
consecutive heads are approximately statistically independent in their directional
preferences. XOR-folding h heads from a single layer creates a fingerprint that is
stable when all heads agree (strong semantic activation) and cancels noise when they
disagree. Extending to two layers doubles the independence constraints: l0 encodes
local syntactic patterns, l4 encodes mid-level semantic composition. A random
token pair must simultaneously agree across 8 head-subspaces spanning two distinct
computational depths — the probability of a spurious match is suppressed to the
product of the two layers' individual false-positive rates.

**Q→Q not Q→K:** Comparing probe Q against corpus Q asks "was the model attending
for the same reason?" — directional intent similarity. Q→K would ask "does this
query match what this key offers?" which is what attention computes at runtime, but
the BDP binarisation of Q→K is too variable to be discriminative (min_ratio=0 for
all QK strategies tested).

**Layer 0 for runs, layer 4 for precision:** Early transformer layers (l0, model
layer 3) produce smoothly evolving Q patterns that drift gradually across consecutive
decode tokens during sustained tool focus, creating long span runs. Mid-level layers
(l4, model layer 7) are more token-selective, providing sharper individual-hit
discrimination. The 8-head XOR captures both properties simultaneously: the l0
component ensures the fingerprint remains correlated across consecutive tokens
(good for span), while the l4 component sharpens per-token distinctiveness (good
for count). Neither is sacrificed.

**Span scoring α=2.0:** A run of L consecutive probe tokens all pointing to the
same corpus section scores L² rather than L. The quadratic reward strongly
distinguishes sustained directional focus (the signature of genuine tool intent)
from isolated hits (noise or coincidental similarity):

| run length L | count score | span α=2.0 score | ratio |
|---|---|---|---|
| 1 | 1 | 1 | 1× |
| 3 | 3 | 9 | 3× |
| 5 | 5 | 25 | 5× |
| 10 | 10 | 100 | 10× |

The file_read/file_write pair produces similar per-token hit distributions under count
but different run-length distributions — the two tools sustain directional focus for
different durations. Span scoring exposes this difference; the dual-layer fingerprint
amplifies it by making each token's contribution more discriminative.

### Final Summary Table

| strategy | min_ratio (count) | min_ratio (span α=2.0) | mean_ratio (count) | mean_ratio (span α=2.0) |
|---|---|---|---|---|
| **MH_XOR_QQ_l0xl4** | **1.419** | **2.528** | **1.747** | **5.314** |
| MH_XOR_QQ_l0 | 0.995 | 1.613 | 1.197 | 3.069 |
| MH_XOR_QQ_l4 | 1.065 | — | 1.274 | — |
| MH_XOR_QQ_l8 | 1.020 | — | 1.209 | — |
| C: gated span(l0, gate=l4) | — | 1.534 | — | 3.328 |
| B: norm span(l0)+count(l4) | 1.354 | — | 2.068 | — |
| QQ single-head (best) | 0.66 | ~1.06 | 0.95 | ~1.10 |
| QK per-head (best) | 0.00 | — | 2.43* | — |
| BandMeanQQ | 0.55 | — | 0.89 | — |

*QK mean_ratio inflated by 0/0 → inf artifacts on zero-signal probes.

### Open Questions

1. **Storage cost:** MH_XOR_QQ_l0xl4 requires 8 Q-vector captures per token per
   syntactic band (4 heads × 2 layers) versus 4 for the single-layer strategy.
   The raw KVQ file is already 1.24 GB for 48 scenarios; production storage impact
   on `signatures.prov` should be quantified before adoption. The signature itself
   remains 128 bits — only the capture overhead increases.

2. **α sensitivity:** Only α=1.5 and α=2.0 were tested. Values in [2.0, 3.0] may
   give further lift, especially for the harder tool pairs (file_read, calculator).
   Given A span α=2.0 already passes with large margin, this is low priority unless
   production constraints require a tighter threshold.

3. **Boundary and negative scenarios:** All 48 probes are positive scenarios. The
   full test set includes boundary (4/tool) and negative (4/tool) cases. False-positive
   rate under boundary/negative conditions must be measured before deployment —
   a higher-discrimination fingerprint could in principle also generate stronger false
   positives if boundary scenarios accidentally resemble tool-focus patterns.

4. **Cross-band:** Results are for the syntactic band only. Semantic (layer 24) and
   pragmatic (layer 40) bands may provide complementary signal — especially for
   longer-context tool invocations where pragmatic-band Q vectors capture intent that
   formed many tokens earlier. A dual-layer sweep within each band, and potentially
   a cross-band combination, are natural extensions.

5. **Dual-layer choice generalisation:** The l0×l4 pair was chosen based on the
   structural argument (early smooth vs mid selective). The optimal pair may differ
   for other models. For each new model variant the single-layer sweep (§ 1) should
   identify the best count layer and best span layer before constructing the dual
   combination.

6. **Model portability:** All thresholds and layer rankings are specific to
   Qwen3-30B-A3B (n_kv_heads=4, head_dim=128, syntactic band centre=layer 7). The
   full measurement pipeline must be re-run for each new model variant before
   deployment. The structural argument for dual-layer combination is model-agnostic;
   the specific layer indices are not.
