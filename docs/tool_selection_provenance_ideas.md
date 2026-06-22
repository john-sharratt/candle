# Tool-selection provenance — signal-extraction game plan

Goal: from the persisted KV/provenance data, select the correct tool (1 of 93)
for a captured tool-call turn, model-free. Harness: `zend/examples/tool_select_from_substrate`.

## Where we are (measured, 2026-06-22)

- **Signal is real, not absent.** Per-token, individual clean cases rank the
  correct tool #1 (e.g. `weather`, sem, consecutive-2). Aggregated naively it
  looked like chance because the discriminative tokens were averaged in with
  structural/argument tokens.
- Narrowed + consecutive n-gram over the assistant name-region: **Top-1 ~3.9%,
  Top-5 ~11.4%** (L=3, window=12) vs chance 1.1% / 5.4% — i.e. **~3.5×**.
- **Why it's hard with what I used (1-bit sign agreement):**
  - Pairwise sign-agreement sits at a **~80/128 correlation floor** (chance 64);
    tool-identity is only **~2–4 bits** on top. Per-bit masking is a no-op (all
    128 bits are individually balanced — the common-mode is *joint* correlation,
    not constant bits).
  - **Q is contextualised**: the literal `weather` token in the call (after
    `<tool_call>{"name":"`) only agrees ~83/128 with `weather` in the definition
    (different absolute position/context), not the ~120 a "same token" would
    suggest.
  - `max over section tokens` is **length-biased** — long definitions win a
    spurious single-token max by chance.
  - The production scanner's `hit_threshold = 90` is **above** the entire
    agreement regime (~80–88) → it counts zero hits → flat zero. Calibrated for
    turn-vs-turn, not call-vs-definition.

## What data we actually have (to confirm, then exploit)

1. **Sign signatures** `TokenSignature` (128-bit), 3 depths (syn/sem/prag =
   layer-group pools), per token, for tool sections **and** captured turns.
   — This is all I used so far. 1 bit/dim is a hard SNR cap.
2. **Full-resolution K** in the persisted KV chunk records (`substrate_inspect
   chunks <stream-id>`), per stream (tool sections + turns). **This is the big
   unused lever.** K is the actual attention key — real-valued, multi-head.
3. **Q** — captured as sign-sigs; need to check whether float/R16 Q is persisted
   (`extract_raw_kvq`, R16 dumps) or only live. Magnitudes may be recoverable.
4. **Token ids** per stream (`substrate_inspect tokens <stream-id>`) — lets us
   locate the literal tool-name token on both sides.
5. **Multiple captures per tool** (~2–6 driver runs in the substrate) — lets us
   average a per-tool query/key centroid to cut variance.
6. **Labels**: `test_config.json` maps each turn → expected tool. We can *train*
   light models (we have ~186–387 labelled examples).

---

## Idea catalog

### A. Use full-resolution K/Q instead of sign bits (biggest lever)

1. **Real Q·K dot product.** Replace popcount(sign agreement) with the actual
   inner product of probe-Q and definition-K. This is literally the attention
   score the model computes; sign agreement is a 1-bit-quantised proxy. Expect a
   large SNR jump.
2. **Scaled dot-product attention readout.** `softmax(Q·K / √d)` over the 93
   definitions' name-token K — replicate the exact op the model uses to *route*.
   The def with max post-softmax weight is the selected tool.
3. **Cosine similarity.** `Q·K / (|Q||K|)` — direction-only; removes magnitude
   bias if magnitudes turn out to be common-mode.
4. **Magnitude-weighted sign agreement** (cheap upgrade to current scan): weight
   each sign-bit match by `|Q_d|·|K_d|`. The dims that dominate the dot product
   get more vote. Uses the user's "magnitude as signal enhancer" with only the
   sign sigs + a magnitude sidecar.
5. **Per-dimension energy weighting.** Down-weight low-variance dims, up-weight
   the high-energy dims that carry attention. (IDF over dimensions.)
6. **K-vs-K matching.** Both the call name token and the def name token have
   *cached K* (full-res). Match K(call-name) · K(def-name): same token, and K is
   the representation attention actually keys on.
7. **Multi-head, not pooled.** The 3 depths are pooled layer-groups. Use
   per-head K/Q (full-res chunks are multi-head) — a single "routing head" may
   carry the tool signal that pooling washes out.

### B. Whitening / common-mode removal (kill the ~80/128 floor)

8. **Mean-vector subtraction.** Subtract the corpus-mean K (and Q) before
   dotting — removes the shared "everybody" direction responsible for the floor.
9. **PCA / SVD residual.** Project out the top-`k` principal components of the
   corpus K matrix (shared JSON/tool-format structure) and score in the residual
   subspace (tool identity). Sweep `k`.
10. **ZCA whitening** of the K space; score by Mahalanobis-style inner product.
11. **Per-tool baseline subtraction.** Subtract each tool's mean agreement to all
    probe tokens (its self-similarity baseline) → removes per-tool length/common
    bias. (Note: per-*probe-token* row-centering does NOT change ranking — must
    center per *tool*/column.)
12. **Per-tool z-score.** `(score − tool_mean) / tool_std` across probe tokens.
13. **Whitened cosine (≈ correlation of residuals).**

### C. Token localisation (match the right tokens)

14. **Literal name-token isolation.** Read token ids; find the tool-name
    token(s) in the call and in each definition; match name-vs-name only. The
    cleanest possible probe — no structure, no args.
15. **Q-norm saliency selection.** Weight/select probe tokens by `|Q|`: the token
    where the model "commits" (likely the name) has a high-norm, decisive Q. The
    model's own attention concentrates there.
16. **Drop structural tokens** (`{ } " : ,` and the shared `{"name":` /
    `"arguments":` scaffold) — pure common-mode shared by all tools.
17. **TF-IDF token weighting.** Down-weight tokens frequent across all
    calls/defs (structure), up-weight rare tokens (the name). Classic IR.
18. **Position prior.** The name is always right after `{"name":"`; weight tokens
    by a learned/observed positional prior of where signal lives.
19. **Section-side name isolation.** On the corpus side, match against the def's
    *name* token, not max over the whole (description-/schema-dominated) def.

### D. Sequence / alignment — "humps and patterns"

20. **Consecutive n-gram** (done, best so far, +3.5×) — extend to variable L and
    **gapped** n-grams (tokenisation differences between call and def).
21. **Smith–Waterman local alignment.** Align the call's name region to the def's
    name region with gaps; score the best local alignment of sign/float vectors.
    Robust to tokenisation drift; rewards a *run* of aligned tokens (a "hump").
22. **Agreement-matrix diagonal detection.** Treat `A[i,j] = agree(call_i,def_j)`
    as an image; a true match shows a bright **diagonal band** (the name aligning
    in order). Convolve with a diagonal kernel / Radon transform to detect it —
    distractors have no coherent diagonal.
23. **Dynamic time warping** of the call vs def token-vector sequences.
24. **Peak/hump statistics on the per-token agreement profile.** Correct tool →
    sharp peak at the name token; distractors → flat. Score by peak prominence /
    kurtosis / max-minus-median, not by mean.
25. **Cross-correlation / FFT** of the agreement profile to detect structured
    humps and their alignment offset.
26. **Curve-shape matching.** Compare the *shape* of the agreement-vs-position
    curve to a template, not just its height.

### E. Multi-layer / multi-depth fusion

27. **Concatenate depths** (syn‖sem‖prag = 384-bit, or full-res per layer) for one
    higher-resolution comparison.
28. **Per-tool best depth** (max over depths) vs **learned depth weights** vs
    **product/AND across depths** (a real match should agree in *all* depths).
29. **Use all layers, not 3 pools.** Full-res K per layer — find the
    "routing layer(s)" where tool identity is most separable. (Cf. the Markov
    expert-prediction finding: routing lives in specific layers.)
30. **Layer-consensus.** Require the same tool to win across multiple layers
    (intersection) — kills per-layer flukes.

### F. Replicate the model's own routing (most principled)

31. **Attention readout at the name-decode step.** When the model emitted the
    tool-name token, its attention `softmax(Q_name · K_context)` over the cached
    tool-definition K's *is* the selection. The def it attends to most is the
    tool it used. This is the ground-truth mechanism, not a proxy.
32. **Pre-name commit token.** The token *just before* the name is emitted is
    where routing happens; use its Q.
33. **Expert-routing signature** (MoE). The expert-selection pattern at the name
    token may itself index the tool (ties to the Markov expert work).

### G. Calibration / scoring formula

34. **Recalibrate `hit_threshold`** to the real regime (~80–88, not 90) — the
    current scan is blind. Sweep on the labelled set.
35. **Recalibrate `span_alpha`, `score_threshold`, `scan_top_k`** for this regime.
36. **Rank/margin scoring.** Top-1−Top-2 margin, rank stats, softmax confidence —
    scale-free, robust to the correlation floor.
37. **Per-query normalisation.** Softmax over the 93 tool scores → calibrated.

### H. Token-score aggregation (less length-biased than max)

38. **Top-k mean** over (probe×def) pairs instead of max.
39. **Length normalisation.** Divide section score by √(#def tokens) (BM25-style)
    to kill the long-definition advantage.
40. **Voting / Borda.** Each probe token ranks the tools; aggregate ranks across
    tokens (rank-aggregation is robust to per-token scale).
41. **Name-window mean** only (ignore args entirely).

### I. Statistical / learned (we have labels)

42. **Linear probe / logistic regression** on agreement features (per-depth,
    per-statistic, name-token) — let supervision find the weights. ~186 labelled
    examples.
43. **LDA / Fisher discriminant** to find the projection separating tools.
44. **Nearest-centroid in whitened space**, per-tool centroid from multiple
    captures.
45. **Learned per-dimension weights** (which K dims encode tool identity).

### J. Variance reduction

46. **Average Q (or K) across the multiple captures** of the same tool → a
    cleaner per-tool query/key centroid; cuts per-sample noise.
47. **Average over the holdout+train calls** of a tool to denoise its query.

### K. Sanity / upper-bound checks

48. **Self-match upper bound.** Score a def's *own* tokens against the corpus —
    confirms the metric can find an identical match (sanity that the pipeline +
    metric are sound before blaming the data).
49. **Name-token-id exact baseline.** How often do call and def even share the
    name token id? Bounds what any Q/K method can do.
50. **Confusion structure.** When wrong, *which* tools win? If they're
    semantically related, the signal is real but coarse; if random, it's noise.

---

## Game plan (priority order)

**Tier 0 — unlock the data (do first; gates everything else):**
- Confirm what floats are persisted: read KV **chunks** for a tool section and a
  turn (`substrate_inspect`), establish full-res **K** access (#2), and check for
  float/R16 **Q** (#3). This decides whether we get real Q·K or K-vs-K.
- Read **token ids** to locate name tokens (#14, #49).

**Tier 1 — highest expected payoff:**
- **Real Q·K (or K·K) with magnitudes** on the **name token**, name-vs-name
  (#1, #6, #14) — the user's core hint. The big SNR jump.
- **Mean-subtraction + PCA-residual whitening** of K (#8, #9) — remove the
  correlation floor.
- **Recalibrate `hit_threshold`** to ~85 and re-sweep the existing scan (#34) —
  cheap, may already lift the current pipeline off zero.

**Tier 2 — pattern/structure:**
- **Smith–Waterman / diagonal-band detection** on the agreement matrix (#21,
  #22) — exploit the "humps."
- **Q-norm saliency** token weighting + **structural-token drop** (#15, #16).
- **Per-head / per-layer K** to find the routing layer (#7, #29, #30).

**Tier 3 — learned + principled:**
- **Logistic/LDA probe** on the best features from Tiers 1–2 (#42, #43).
- **Attention readout** at the name-decode step (#31) — the ground-truth router.

Each idea is measured the same way: Top-1 / Top-5 / MRR over all captured turns
via a new `--score*` mode in `tool_select_from_substrate`, against chance
(1.1% / 5.4%) and the current best (~4% / 11%).

---

## Idea catalog — wave 2 (deeper math + attention internals)

### L. Position is NOT a confound — values are pre-RoPE (corrected)

**Correction:** the stored Q/K are **pre-RoPE** — RoPE is applied inside the
attention kernel, not baked into the persisted values. So there is no rotation
to undo, and absolute position is *not* the reason same-token agreement is only
~83. The residual same-token gap is genuine **contextual** difference (the
upstream residual stream feeding the attention projection differs between the
call context `<tool_call>{"name":"` and the def context) plus the shared
common-mode. Implications:

51. **Content matching is already position-invariant.** Pre-RoPE K is the clean
    *content* key — good for tool-identity matching; we don't fight position.
    The lever for the residual gap is **common-mode/context removal** (family B
    whitening) and **name-token isolation** (family C), not de-rotation.
52. **To replicate the model's *real* attention, ADD RoPE** (opposite direction):
    family F#31's attention readout needs post-RoPE Q/K, so apply RoPE at the
    correct *relative* offset between the name token and each def's key before
    `softmax(Q·K/√d)`. (RoPE matters here as something to *add*, not remove.)
53. **Pre-RoPE dot = pure content similarity.** `Q·K` on pre-RoPE vectors is the
    position-free content score — arguably the *cleanest* signal we have for
    identity. Prioritise it (family A#1/#6) without any positional correction.

### M. Head specialisation (use the right heads, not pooled depths)

56. **Induction heads.** Heads that match "previous occurrence of X" would fire
    when the call name matches the def name. Identify induction heads (prefix-match
    behaviour) and use only their Q·K.
57. **Name-mover heads.** The heads that copy the tool name into the residual at
    the decode step; their Q·K is the routing signal.
58. **MI-selected heads.** With labels, pick the heads whose per-head Q·K most
    separates tools (max mutual information with the label).
59. **Per-head vote / max.** Score per head; aggregate by max or rank-vote — a
    single routing head beats the pooled average.
60. **Attention-pattern fingerprint.** The *set* of heads that attend strongly to
    a def (a binary/real head-mask) is itself a tool signature; compare masks.

### N. Value vectors and the OV circuit (entirely unused so far)

61. **V·V on the name token.** We've only used K. The V is what attention *reads
    out*; the def name's V is the content delivered. Match call-name V to def V.
62. **OV-circuit readout.** Project V through the output matrix (`W_O`) — the
    actual contribution to the residual stream. Compare in that space.
63. **Reconstructed read-out.** `Σ softmax(QK)·V` over the def tokens = what the
    model would pull from that def; compare to what it actually pulled.

### O. Information theory / discriminability (supervised, we have labels)

64. **Per-dimension mutual information** with the tool label → keep MI-max dims.
65. **Fisher discriminant ratio** per dim (between-tool / within-tool variance);
    weight dims by it.
66. **Bhattacharyya / KL** between a tool's agreement distribution and the
    background; rank by divergence.
67. **Score entropy / sharpness** as a confidence gate (peaky = trustworthy).

### P. Optimal transport / assignment (principled fix for greedy-max length bias)

68. **Hungarian (max-weight bipartite matching)** between call tokens and def
    tokens on the agreement matrix — one-to-one, no token reused, kills the
    "long def wins a spurious max" bias.
69. **Optimal transport / Wasserstein** between the call's token-vector
    distribution and each def's — earth-mover distance as the score.
70. **Sinkhorn divergence** (entropic-regularised OT) — fast, differentiable.
71. **Soft-DTW** — differentiable sequence alignment for the "humps."

### Q. Kernel / nonlinear similarity

72. **RBF kernel** `exp(−‖Q−K‖²/2σ²)` instead of linear dot — sharpens near
    matches, suppresses the correlation floor; sweep σ.
73. **Polynomial kernel** (degree 2–3) — captures higher-order coincidences.
74. **Random Fourier features** to approximate RBF cheaply at scale.
75. **Kernel target alignment** (supervised) to pick the kernel.

### R. Robust statistics (outlier-resistant aggregation)

76. **Median / trimmed-mean** agreement instead of max/mean.
77. **RANSAC** — find the largest consistent set of token matches (inliers) per
    def; score by inlier count.
78. **Winsorised / Huber** aggregation.

### S. Multi-scale signal processing on the agreement profile

79. **Matched filter.** Correlate the per-position agreement profile with the
    expected name-template shape (a localised bump) — classic detection-theory
    optimum for a known signal in noise.
80. **Wavelet / scale-space** peak detection (humps at multiple widths).
81. **Hough / Radon transform** for the diagonal band in `A[i,j]`.
82. **Cepstrum / autocorrelation** for periodic JSON structure (so we can
    *subtract* it and keep the aperiodic name signal).

### T. Whitening++ and subspace learning

83. **CCA** between call-Q space and def-K space — the maximally-correlated
    directions are the shared tool-identity axes.
84. **Procrustes** alignment mapping call space → def space, then compare.
85. **LDA subspace** (supervised) projecting to the ≤92-dim tool-separating space.
86. **Sparse autoencoder features** (if available) — interpretable tool-identity
    units; match in feature space.
87. **Top-PC removal swept jointly with the metric** (B#9 × A#1 grid).

### U. Decode-trajectory features

88. **Q-trajectory through the name.** The sequence of Q vectors across the name
    tokens traces a path; match trajectories (correct def's K-path aligns).
89. **Attention lock-on.** The decode step where attention entropy collapses
    (the model "decides") — use that step's Q.
90. **Q-velocity** (token-to-token Q delta) as a content-change feature.

### V. Re-hashing / richer signatures (if full Q/K is on disk)

91. **Re-hash to more bits.** If we have float Q/K, project onto 256/512/1024
    random hyperplanes for finer Hamming resolution than the stored 128.
92. **Learned hyperplanes (LSH).** Choose hyperplanes that separate tools
    (supervised), not random ones.
93. **Multi-bit / magnitude-bucketed sigs** — keep a few magnitude bits per dim.

### W. Calibration / null models

94. **Per-tool null distribution.** Score each tool against random/irrelevant
    probes to get its null; report the call's score as a z-score vs that null —
    removes per-tool "agrees with everything" bias rigorously.
95. **PMI scoring.** `score(call,def) − E[score(·,def)]` (pointwise mutual
    information) — down-weights promiscuous defs.

### X. Ensemble / fusion

96. **Reciprocal rank fusion** across all scorers (depths × statistics ×
    metrics) — robust, parameter-light, often beats any single scorer.
97. **Stacked meta-model** (logistic) over the scorers' ranks/scores.
98. **AND-consensus** — require a tool to rank top-k under several independent
    metrics (precision-boosting).

### Y. Set / cross-example structure

99. **Global assignment over 93×93.** Score every (call_i, def_j); solve the
    assignment problem — the optimal permutation should be the identity, and its
    accuracy upper-bounds any per-query method.
100. **Consensus query** from a tool's train+holdout calls (and multiple runs) —
     average to denoise the query before matching.
101. **Leave-one-out retrieval** sanity: does a call retrieve its own def over
     the *other tools' calls* (call-vs-call), isolating identity from format.

### Z. Attention-sink & structure handling

102. **Strip attention-sink tokens** (positions 0–3) — they carry global-scale
     junk, not identity (cf. the sink-protection design).
103. **Subtract the shared scaffold.** Explicitly null the `{"name": "`,
     `"arguments":`, `}` token vectors (identical across all calls/defs) before
     scoring — they are pure common-mode by construction.

---

## Revised top priorities (with wave 2)

(RoPE de-confounding is **removed** — values are pre-RoPE, so position isn't the
confound; see family L. The residual same-token gap is contextual common-mode,
handled by whitening + name isolation.)

- **#1 Real Q·K / K·K / V on the name token, magnitude-weighted** (#1, #6, #14,
  #61) — the data-resolution unlock. Pre-RoPE `Q·K` is *pure content similarity*
  (position-free), the cleanest identity signal we have — do this first.
- **#2 Whitening: mean-sub + PCA-residual** (#8, #9) and **per-tool null z-score**
  (#94) — remove the correlation floor and promiscuous-def bias.
- **#3 Principled aggregation: Hungarian / OT** (#68, #69) and **matched-filter /
  diagonal-band** (#79, #81) — kill greedy-max length bias and exploit humps.
- **#4 Right heads/layers** (#56–59, E#29) — routing lives in specific
  heads/layers, not pooled depths.
- **#5 Cheap-but-overdue:** recalibrate `hit_threshold` 90→~85 (#34), strip
  attention sinks (#102) and scaffold tokens (#103).
- **#6 Supervised, on the best raw features:** MI/LDA dim selection (#64, #85),
  logistic probe (#42), reciprocal-rank-fusion ensemble (#96).
- **Upper bounds to keep us honest:** self-match (#48), global 93×93 assignment
  (#99), call-vs-call identity check (#101).

Order of operations: **Tier 0 data unlock → pre-RoPE Q·K/K·K/V on the name token
(magnitude-weighted) → whiten (mean-sub + PCA + null z-score) → align/aggregate
(Hungarian + matched-filter) → heads/layers → supervised fusion.** Re-measure
after each; let the numbers, not the priors, drive the next step.

---

## Experimentation methodology — sweeping & rigour

The effects are small (single-digit % over chance) and the knob space is huge, so
**how** we experiment matters as much as the ideas. Every idea above is one point
in a shared configuration space; we search it, not eyeball it.

### Unified scoring config (everything is a config point)

A single `ScoreConfig` captures every knob, and one `score(config, turn, corpus)
→ ranking` function realises any idea family. This makes the whole catalogue a
search space, not 100 bespoke scripts:

| Knob | Values to sweep |
|---|---|
| `metric` | sign-agree · Q·K · K·K · V·V · cosine · RBF(σ) · poly(d) |
| `magnitude_weight` | none · \|Q\| · \|K\| · \|Q\|\|K\| · per-dim energy |
| `whiten` | none · mean-sub · PCA-residual(k=1..32) · ZCA · per-tool z-score |
| `tokens` | whole-half · name-window(W=2..24) · name-token-only · Q-norm-top-m · drop-structural · drop-sink |
| `section_tokens` | whole-def · name-only · description-only |
| `aggregate` | max · top-k-mean(k=1..8) · mean · n-gram(L=1..5) · Hungarian · OT/Sinkhorn · matched-filter · diagonal-band |
| `depth` | syn · sem · prag · concat · max-over · AND · learned-weights |
| `heads` | pooled · per-head-max · MI-selected · induction-only |
| `threshold` | hit∈[60,95] · score_threshold · span_alpha∈[0,4] · scan_top_k |
| `normalise` | none · per-query softmax · per-tool null z-score · length-norm(√n, BM25) |
| `ensemble` | none · reciprocal-rank-fusion · stacked-logistic |

Full cross-product is ~10¹⁰ — infeasible to grid. Hence the search strategy below.

### Search strategy (coarse-to-fine, not brute grid)

1. **Coordinate ascent.** Tune one knob at a time holding the rest at the current
   best; cycle until stable. Cheap, interpretable, good first pass.
2. **Coarse-to-fine grid.** Wide/low-res grid → zoom into the promising region at
   higher res. For continuous knobs (σ, α, PCA-k, thresholds).
3. **Random search** over the joint space — beats grid in high dimensions; good
   for discovering interactions.
4. **Successive halving / Hyperband.** Launch many configs on a *subset* of
   turns, keep the top fraction, re-run survivors on more — kills losers cheaply.
5. **Bayesian optimisation (TPE/GP)** over the continuous knobs once the discrete
   structure is fixed.
6. **Fractional-factorial DOE.** Estimate main effects + key interactions of the
   discrete knobs with O(knobs) runs instead of the full product.
7. **Greedy forward component selection.** Start from the simplest scorer; add the
   one component that most improves the *holdout*; repeat — natural ablation.

### Splits & overfitting control (non-negotiable for small effects)

- The dataset already has **train + holdout** per tool. **Tune every sweep on
  train; report only on holdout.** A swept number on its own tuning data is
  meaningless here.
- Multiple driver runs in the substrate = **natural CV folds**; k-fold across
  runs for stability.
- **Nested CV** for any learned component (logistic/LDA/learned-weights/MI-dims):
  inner loop tunes, outer loop reports — no leakage.
- **Lock the holdout**: touch it only for the final reported number of a phase.

### Statistics (the effects are small — quantify uncertainty)

- Report **95% bootstrap CIs** on Top-1/Top-5/MRR (resample turns) — 3.9% on
  n=387 has a wide CI; we need to know if a gain is real.
- **Significance vs chance:** binomial test, Top-1 hits ~ Binomial(n, 1/93).
- **Paired comparison vs current best:** paired bootstrap on the *same* turns
  (config A vs B), not independent runs.
- **Multiple-comparison discipline:** sweeping N configs inflates false positives;
  validate the chosen config on the **locked holdout** (one shot), and/or
  Bonferroni/Benjamini–Hochberg on the sweep.

### Ablations & attribution

- **On/off matrix:** from the best config, disable each component → measure the
  Top-1 delta → attribute the gain to components (and find dead weight).
- **Component-replace-with-oracle:** swap a component for its perfect version to
  see its ceiling (e.g. oracle token selection).

### Oracles & upper bounds (are we search-limited or signal-limited?)

- **Best-token / best-depth / best-head / best-L oracle:** if we could pick the
  best per case, what's Top-1? Bounds what any tuning of that knob can yield.
- **Self-match** (def vs corpus) and **call-vs-call identity** (does a call
  retrieve its own def over *other tools' calls*) — isolates identity from format.
- **Global 93×93 assignment** (Hungarian over all calls × all defs): the optimal
  permutation should approach the identity; its accuracy upper-bounds per-query.
- If oracles are high but our scorer is low → **search-limited** (keep tuning).
  If oracles are also low → **signal-limited** (need richer data, e.g. V/heads).

### Negative controls (validate the harness, not the wish)

- **Label shuffle** → must collapse to chance (proves no leakage/bug).
- **Random-signature corpus** → chance.
- **Probe/corpus swap** sanity.

### Per-segment diagnostics (where the signal lives)

- **Per-tool and per-category** Top-1 (file ops · net sessions · crypto · code ·
  …) — some families may be solved while others aren't; report the breakdown.
- **Confusion structure:** when wrong, which tool wins — semantically adjacent
  (coarse-but-real signal) vs random (noise)?
- **Score-separation plots:** correct-vs-distractor score distributions; overlap
  is the real figure of merit beneath Top-1.
- **Sensitivity curves:** metric vs each continuous knob (find plateaus/cliffs).
- **Learning curve:** accuracy vs #training examples (for learned components).

### Harness/approach enhancements to build

- **Load once, score many.** Substrate open + per-turn sig/K reads are the
  expensive part; cache them in RAM and run *all* configs against the cached
  arrays. (Today each run re-opens the 6.7 GB substrate — fix this first.)
- **`--sweep <spec.json>`** mode: enumerates grid/random/coordinate configs, runs
  each over cached turns, computes metrics + bootstrap CI on train *and* holdout,
  writes a ranked `sweep_results.json` (config → metrics) for offline analysis.
- **Parallelise** across configs (rayon) since scoring is CPU-bound and per-config
  independent once data is cached.
- **Deterministic + logged:** every run records its full `ScoreConfig` + git SHA +
  metrics so results are reproducible and comparable across sessions.

---

## Compounding signals — pipelines & combined formulas

The single methods each attack *one* noise source. The big wins should come from
**stacking methods that attack different sources**, because their gains multiply.
The `ScoreConfig` is deliberately a *pipeline* of orthogonal stages so the sweep
explores compounds, not just singletons.

### The pipeline (orthogonal stages — each kills a different noise source)

```
 select(tokens)  →  denoise(whiten)  →  similarity(metric)  →  channel-fuse(depth/head/K,V)
        ↓                  ↓                    ↓                          ↓
 kill arg/structure   kill common-mode    resolution (vs 1-bit)     independent evidence
        →  aggregate(match)  →  calibrate(null/length)  →  ensemble(fuse scorers)
                  ↓                     ↓                          ↓
          kill length/greedy bias   kill promiscuous-def     fuse uncorrelated signals
```

Because these are (mostly) orthogonal, a compound that does *select + whiten +
real-metric + match + calibrate* can in principle multiply five independent
gains. That is the central bet.

### Orthogonality / redundancy map (what to combine vs not)

- **Strongly orthogonal (combine — multiply gains):** whiten ⟂ token-select ⟂
  magnitude/real-metric ⟂ aggregation-vs-length ⟂ per-tool calibration ⟂
  head/layer selection. Each removes a distinct nuisance.
- **Same axis (pick one, don't stack):** max vs top-k-mean vs mean (aggregation);
  RBF vs polynomial (kernel); n-gram vs consecutive (sequence); cosine ≈
  mean-sub + dot (don't do both — redundant normalisation).
- **Sequencing matters:** whiten *before* the metric; drop-structural *before*
  aggregation; calibrate *after* scoring; ensemble *last*. Order is itself a knob.

### Notation

`q_i` = probe token *i* vector (depth *d*, head *h*); `k_j`, `v_j` = def token
vectors; `W` = whitening map (project out top-*r* corpus-K PCs); `m_i = ‖q_i‖`
(Q-norm saliency); `A[i,j] = sim(W q_i, W k_j)`; `n` = #def tokens.

### Compound recipes (2–3 methods each, with formula + why it stacks)

**C1 — Whitened magnitude-weighted name dot** (denoise × real-metric × salience)
`score = m_name · ⟨W q_name , W k_name^def⟩`.
The single cleanest content pair, with common-mode removed and weighted by the
token the model committed on. Three orthogonal fixes in one number.

**C2 — Whitened Hungarian, length-normalised** (denoise × optimal-match × length)
`score = (1/√n) · max_π Σ_i A[i, π(i)]` over one-to-one assignments π.
Removes common-mode, the greedy-max spurious hit, *and* the long-definition bias
simultaneously.

**C3 — Drop-structural → consecutive n-gram → whiten** (mask × sequence × denoise)
Strip `{ } " : ,` + the `{"name":` scaffold, then best consecutive L-gram on
whitened content vectors. The name run is then the *only* structured signal left.

**C4 — GCC-PHAT alignment** (cross-correlation × phase-whitening) — *the DSP one.*
Treat the call-token and def-token vector sequences as signals; cross-correlate;
**phase-transform** weight (divide the cross-spectrum by its magnitude) to whiten
and sharpen the alignment peak; `score = max_lag |GCC-PHAT|`. Borrowed from
acoustic time-delay estimation — exactly the "find where two patterned signals
align" problem, and PHAT is the optimal sharpener for a single dominant peak.

**C5 — Whitened matched filter on the agreement profile** (denoise × detection)
`p_i = max_j A[i,j]`; `score = Σ_i t_i · p_i` with `t` a name-region bump
template (learned/observed). Detection-theory optimum for a known-shape signal in
noise — the optimal way to read the "hump."

**C6 — Cepstral / homomorphic deconvolution** (periodic-structure removal × detect)
`C = IFFT(log|FFT(p)|)`; null the quefrency bin of the periodic JSON scaffold;
score the aperiodic residual. Subtracts the repeating `","..."` structure and
keeps the one-off name bump.

**C7 — Depth-AND consensus on the name token** (multi-channel × conjunction)
`score = min_d ⟨W q_name^d , W k_name^d⟩`. A *real* match agrees in syn **and**
sem **and** prag; the min punishes per-depth flukes — precision booster.

**C8 — MI-head dot + per-tool null-z** (head-select × calibrate)
`score = max_{h∈H*} [ s_h(name) − μ_h^null(tool) ] / σ_h^null(tool)`, where `H*`
= MI-selected routing heads. Uses only informative heads, and z-scores out each
tool's promiscuity. (Supervised head pick + per-tool calibration.)

**C9 — K⊕V channel fusion** (two independent representations)
`score = α · ⟨W k_name^call, W k_name^def⟩ + (1−α) · ⟨W v_name^call, W v_name^def⟩`.
K = what attention keys on; V = what it reads out — partly independent evidence;
sweep α.

**C10 — Reciprocal-rank fusion of uncorrelated scorers** (ensemble)
`RRF(tool) = Σ_{scorer} 1 / (c + rank_scorer(tool))` over a *diverse* set: C1
(point), C2 (assignment), C4/C5 (sequence/DSP), C8 (heads). Fusion helps most
when the members are uncorrelated — pick scorers from *different* families.

**C11 — Supervised stack (grand-unified)** (learned compound)
`score = σ( w · φ )`, `φ = [C1, C2, C4, C5, C7, C8, C9, n-gram-L, peak-prominence,
score-margin, per-tool-z]`. Logistic/GBM over the compound features; nested-CV.
The endpoint that lets the data weight the compounds — only meaningful once the
component features individually clear chance.

**C12 — Whitened cosine of consensus query** (variance-reduction × denoise × norm)
Average a tool's call vectors across train+holdout+multiple runs → consensus
query `q̄`; `score = cos(W q̄_name, W k_name^def)`. Cuts per-sample Q noise before
matching.

### Suggested "core compound" to build toward

`select=name-window+drop-structural+drop-sink` → `whiten=PCA-residual(r)` →
`metric=magnitude-weighted Q·K` → `channels=max over MI-heads, K⊕V` →
`aggregate=Hungarian(length-norm)` → `calibrate=per-tool null-z + per-query
softmax` → `ensemble=RRF with C4 (GCC-PHAT) and C5 (matched filter)`.
Every arrow removes a different nuisance; the sweep tunes each stage's parameter
with the others held at current-best (coordinate ascent), then a final
joint coarse-to-fine pass to catch interactions.

### Compounding-aware search (don't tune stages in isolation)

Stages **interact** (e.g. whitening changes the best threshold; head-selection
changes the best aggregator), so pure independent coordinate-ascent under-shoots.
Mitigations: (1) re-cycle coordinate ascent to convergence; (2) a final
**fractional-factorial** pass over the discrete stage choices to estimate
two-way interactions; (3) keep the **RRF ensemble** as a safety net — it captures
cross-stage signal even when no single pipeline is globally optimal.
