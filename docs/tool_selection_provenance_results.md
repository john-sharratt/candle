# Tool-selection provenance — results

**Status:** shipped, 2026-07-06. Live harness: the `belief-*` subcommands of
`candle-conversation/examples/substrate_inspect.rs` (`belief-eval`, `belief-probe`,
`belief-dissect`, `belief-sweep`, `belief-decay`) — load-once, model-free, run on CPU against
the workspace `.substrate/substrate.log`. Companion to the idea catalogue in
[`tool_selection_provenance_ideas.md`](tool_selection_provenance_ideas.md). §§1–22 are the
research record that led here; the shipped design is §23–§24.8, summarised next.

---

## ★ Shipped production design (§23–§24.8)

**The mechanism in the daemon today.** Blind (no labels, no per-request tuning); the query is a
window the model generates itself; runs on the sign/popcount path.

1. **Signature** — each token's `sign(Q)` over all heads/layers is folded to a compact
   [`WideQSig`]: 3 fold-groups `[46, 1, 1]` (L0–45 absorbs noise; L46, L47 carry tool identity)
   × 4 KV-heads = 1536 bits. Captured at seal, stored per turn on the substrate (§23).
2. **Probe** — at each reprojection the **last 256 tokens** of the decode window
   (`reproject_max_probe_tokens`), folded the same way.
3. **Gallery** — past turns whose tags intersect the collection's `tags` (the `tools` collection
   scopes to the tagged calibration corpus); cross-conversation and self-reinforcing.
4. **Scorer** (`score_provenance_late_fusion`) — per query token, per fold-group, a
   **`z × margin`** vote for the leading tool: `z` (outlier vs the group's agreement
   distribution) mutes noisy *groups*, `margin` (leader − runner-up tool) mutes generic *tokens*
   (§24.7). Then the **needle gate** keeps only the top **25 %** of tokens by vote magnitude —
   position-independently isolating the signal from the diffuse haystack (§24.8).
5. **Belief** — `RelLeak` (`acc[t] = max(0, acc[t] − β·max(acc)) + s[t]`, β = 0.40) carries and
   decays belief across a turn's reprojections (§24.5).
6. **Selection** — `SectionSelector` under `committed_tool_scope`: β 0.40, **min 1000 / evict 750**,
   **budget 1..3**, `fix_per_turn` (§24.6). Persisted per turn by `projection_event`, which scores
   the just-sealed turn's stored signature so the record and the self-reinforcing gallery agree.

**Numbers** (leave-one-out over 372 tagged tool turns, 93 tools; chance ≈ Top-1 1.1 %):

| | Tool-1 | Tool-3 | Tool-5 | MRR |
|---|--:|--:|--:|--:|
| ranking | **97.8 %** | **100 %** | **100 %** | 0.989 |

At the shipped policy: **100 % recall** (the true tool is always in scope), **mean 0.37 false
positives**, **58 % exact-1** (only the right tool). The lone Top-5 miss under the plain z-fusion
(`tcp_session_list`, a `*_session_list` family collision) is closed by the margin + needle gate.

**Honest limit:** all numbers are the **tool** corpus (labelled calibration turns). The design is
domain-agnostic by construction — the needle gate keys on signal quality, not recency or domain, so
it should carry to content/code collections — but that case is **not yet measured**; it needs a
labelled code-section corpus.

---

## ★ Best candidate — per-token Q·Q consensus voting (§74)

> **Superseded.** This was the research-era locked mechanism (pooled/consensus `Q·Q` on the raw
> per-token sign path). The shipped system generalises it: the folded [`WideQSig`] + `z × margin`
> late-fusion + needle gate + `RelLeak` belief above (§23–§24.8). Kept for the record.

**The locked, production-validated mechanism.** Measured 2026-06-28, harness
`zend/examples/calibrate_alignment.rs` (`S21_ONLY=1 S74=1`). Full detail in [§22](#22-the-pivot--decodedecode-qq-retrieval-the-product-path).

**What it answers:** at the projection boundary, *which past decoded context should be
pulled into scope?* — blind (no labels, no tuning), from a query the model generates
itself, on the existing sign/popcount path.

**The mechanism:**

1. **Greedy-draft** the next few tokens at the projection boundary — let the model *say*
   what it is about to talk about instead of guessing what it will attend to.
2. **Each draft token is a query** — its stored per-layer/per-head `Q` (pre-RoPE content).
3. **Score each draft token against every stored decode token** — per-head cosine of `Q`
   summed over all 48 layers. This is **`Q·Q`** (query-to-query), *same-domain* — the
   regime that works, unlike the call→def `Q·K` wall.
4. **Each draft token votes** for the source decode of its single best match.
5. **Tally the votes** → pull the winning source's context into scope; decode for real.

**The numbers** (185 decodes, ~93 tools, chance ≈ Top-1 0.5 % / Top-5 2.7 %):

| uniform all-48, consensus `vote` | Tool-1 | Tool-5 | Fam-1 | Fam-5 |
|---|--:|--:|--:|--:|
| cosine | 51.9 | 67.0 | 67.6 | 78.9 |
| **SIGN / BDP** | 50.3 | 68.1 | 65.9 | 80.0 |

**≈ 50 % exact-tool / ≈ 80 % family at Top-5 (~70× chance at Top-1), and binarizing to the
1-bit popcount scan costs ~1 point** — the cheap scan is the real scan.

**Why it works:** (1) **same-domain** decode→decode, not the cross-domain call→def gap;
(2) **consensus ≫ pooling** — independent per-token votes (50.8) crush an averaged query
(`meanpair` 22.7), because pooling blurs the distinctive name token into the boilerplate;
(3) **sign-robust** — `head_dim = 128` = one machine word/head, so similarity is
`sign(Qa) XOR sign(Qb)` + popcount, ~free on existing hardware.

**The one ceiling:** exact-tool tops out at ~52 % on the **family wall** (`telnet_` vs
`tcp_session_list` differ only by transport, which the *call* does not carry — proven a
genuine information limit in §22.6). Family-level (~80 %) is the metric that matters for
"pull the right neighbourhood into scope."

> The §16 CCA path (probe→definition, Top-1 90 / Top-5 95) remains the strongest *cold-start*
> mechanism with zero history; §74 is the strongest *same-domain memory* mechanism and the
> one the greedy-draft projection loop runs in. They are complementary, not competing.

> **One-line answer.** Tool identity is **strongly** present in the provenance data —
> but it lives in *matching a call against the same distribution* (past tool-call
> probes / their keys), not against the static **definitions** in sign space (~10 %
> Top-5, marginal — a call-context vs definition-context **domain gap**). Three layers
> of result, each stronger: (1) **sign-proxy memory** (probe→past-probes) is ~20×
> chance (Top-1 18–23 %, Top-5 32–36 %), robust to ordering, and a **stateful lock-on**
> prior lifts it to **Top-1 35 % / Top-5 48 %** on bursty workloads; (2) **real Q·K**
> (the actual attention score, with float Q+K reconstructed from the R16 chunks —
> validated bit-exact) — best config **mean-subtracted K·K with top-8-PC whitening, all
> 48 layers — reaches Top-1 64.5 % / Top-5 78.9 %** (~60× chance, ~2× the sign proxy),
> with **K·K > Q·K > V·V ≫ Q·Q**, a single **routing head (head 3)**, and **whitening as
> the biggest lever**; (3) the literal probe→**definition** test was run with correctly
> dequantised def-K (self-match 100 %) and is **a dead end — Top-5 11 %, the same regime
> as the 1-bit scan** — proving the bottleneck is a **representational domain gap, not
> resolution**: a tool *call* and its *definition* live in different activation regions
> that no amount of resolution/whitening bridges **in a shared space**. **But that gap is
> *linearly bridgeable*: a learned CCA alignment (§16 — per-token × per-copy pairs, d=160,
> r=32 canonical dims) lifts probe→definition to Top-1 90 % / Top-5 95 % on holdout (**per-tool
> eval, 2 def copies; the per-probe cold-start figure is ~76 / 88 — see the §16 correction note**), *beating*
> the memory path and enabling cold-start gather with no history.** The corrected product mechanism is
> therefore **either** probe→gather-from-memory (whitened K·K + lock-on) **or** probe→
> definition via the CCA map (stronger, cold-start). **§16 is the headline result; the
> "do not match against definitions" guidance in this paragraph predates the alignment and
> is superseded by §16.** Finally (§17), the canonical-space routing is made **sublinear** by
> an **IVF tree** (cluster into √N groups, scan the top ~5): full 90 / 95 quality at a 2.3×
> compute cut (→ O(√N)), and **needle-safe** — the most-isolated tools recall 100 % *earliest*,
> because isolation gives a needle a sharp centroid the probe locks onto.

---

## 1. Setup, data, methodology

### Data (all read once from `.substrate/substrate.log`)

| Asset | Count | What it is |
|---|---|---|
| Probes (tool-call turns) | **269** full-grid captures | user prompt + assistant `<tool_call>{…}`, sealed **lossless** (R16 K / F16 V), with per-token Q sign-signatures (3 depths) + token ids + content bounds |
| Tool definitions (corpus) | **93** distinct tools (186 section copies, folded by max) | the per-tool definition section, Q sign-signatures + token ids |
| Labels + splits | 186 `(tool, split)` rows | `test_config.json`; verbatim prompt → tool + train/holdout |

After dropping 101 tail-only captures (no clean user/assistant boundary), the set is
**138 train / 131 holdout** probes; two def-copies per tool exist (two daemon
sessions) and are folded to one per tool by **max over copies**. Chance for 1-of-93 is
**Top-1 1.08 % / Top-5 5.38 %**.

### What is actually on disk (this gates every method)

- **The R16 block stores BOTH K and Q.** Each `BlockR16` is `{ f16 d[32] = K,
  f16 q[32] = Q }` ([`backing.rs:1696`](../candle-nn/src/kv_cache/chunked/backing.rs#L1696)).
  So the probe's lossless R16 chunks carry the full-resolution **query *and* key** per
  token — "real Q·K" is fully available on the probe side (see §9, **run**). The earlier
  belief that "only sign(Q) is persisted" was wrong: sign(Q) is the *signature*, but the
  float Q sits in the R16 `q[]` field. Reconstructing it from the persisted bytes and
  re-deriving the syn signature (L3⊕L7, head-folded) reproduces the stored signature at
  **128.0/128 agreement** — a bit-exact correctness gate.
- **Signatures are `sign(Q)`** — 128-bit per token, 3 depths. For 48 layers the depth
  layer-pairs are **syn=L3⊕L7, sem=L20⊕L24, prag=L36⊕L40**. The production scan is Q-vs-Q.
- The corpus **definitions** are sealed near-lossless quantised (Q8_KS K), so their float
  K needs a dequant pass — the only reason probe→def real Q·K is not yet run (§9).
- Every one of the 128 sign bits is individually balanced across the corpus
  (population in (0.2, 0.8) for **128/128** bits, all 3 depths) — so per-bit masking is
  a no-op, confirming the catalogue's note that the common mode is *joint*, not
  constant bits.

### Methodology (rigour knobs)

- **Unified pipeline.** Every idea is one point in
  `select(tokens) → whiten → metric → aggregate → depth-fuse → calibrate`. One
  `score()` realises all of them; the sweep searches the space rather than running 100
  bespoke scripts.
- **Train/holdout discipline.** Tuning (coordinate ascent, NullZ baselines) fits on
  **train**; numbers reported on **holdout** or on a **fixed a-priori config** over all
  probes (no selection on the reported data).
- **Expected metrics under ties.** A block of `e` tools tied with the correct one is
  scored by averaging the metric over its rank positions — so an all-tie config scores
  exactly at chance, not a spurious 100 %. (This bug bit the first run; fixing it is
  what makes the numbers honest.)
- **Bootstrap 95 % CIs** (500 resamples of the probe set, deterministic seed).
- **Oracles** (search-limited vs signal-limited), **negative controls** (label
  shuffle), and a **confusion** breakdown keep it honest.

---

## 2. Headline numbers

| Measurement | Top-1 | Top-5 | MRR | Note |
|---|--:|--:|--:|---|
| **chance** | 1.1 % | 5.4 % | ~0.02 | 1-of-93 |
| Best **fixed** sign-space config (drop-struct + n-gram + bitmask + null-z) | 3.3 % | **10.0 %** | 0.088 | CI [6.7–13.8]; ~2× chance |
| Coordinate-ascent best, **holdout** | 1.5 % | 3.1 % | 0.058 | overfit — train was 12.3 % |
| **Oracle:** best-per-probe over 13 scorers | 9.9 % | **38.1 %** | 0.248 | headroom exists |
| **Oracle:** self-match (def→corpus) | 100 % | 100 % | 1.000 | pipeline sanity ✓ |
| **call-vs-call** (probe→other probes) | **32.7 %** | **58.9 %** | 0.453 | identity is *real and strong* |
| label-shuffle control (best cfg) | 1.9 % | 7.6 % | 0.066 | **promiscuity floor** |
| **probe→gather memory** (sign proxy, online) | 18–23 % | 32–36 % | — | robust to order |
| **memory + lock-on** (bursty) | **35.5 %** | **48.0 %** | — | session workloads |
| **real Q·K** (float, all 48 layers summed) | 48.8 % | 77.3 % | 0.621 | beats sign proxy +16/+18pt |
| **whitened K·K** (mean-sub + top-8 PC, all layers) | **64.5 %** | **78.9 %** | **0.712** | best overall, ~60× chance |

The three rows that matter: **self-match 100 %** (the metric works), **call-vs-call
58.9 %** (Q identity is strong), **probe→def 10 %** (cross-domain matching is the weak
part). Everything below follows from that triangle.

---

## 3. Family-by-family sweep (everything tested)

All fixed a-priori, all 269 probes, Top-5 with 95 % CI. The **promiscuity floor**
(label-shuffle ≈ 7.6 % Top-5) is the *real* baseline to beat, not raw chance 5.4 %.

| Family / config | Top-1 | Top-5 | MRR | Top-5 CI | verdict |
|---|--:|--:|--:|:--:|---|
| **D** consecutive n-gram (L2, W12) | 1.7 % | 6.9 % | 0.066 | [4.2–9.9] | weak |
| **D** consecutive n-gram (L3, W12) | 0.4 % | 5.6 % | 0.051 | [3.2–8.2] | chance |
| **H** max-pair (W12) | 0.6 % | 6.0 % | 0.053 | [3.6–8.7] | chance |
| **H** top-k mean (k8, W12) | 1.1 % | 5.0 % | 0.058 | [2.6–7.6] | chance |
| **C** name-token only | 0.3 % | 4.2 % | 0.047 | [2.2–6.5] | chance |
| **C** name-vs-name only | 1.1 % | 5.1 % | 0.059 | [2.7–7.6] | chance |
| **C** drop-structural (W12, L2) | 3.3 % | 9.3 % | 0.084 | [6.2–12.5] | **real** |
| **B** bitmask whiten (W12, L2) | 1.7 % | 6.9 % | 0.066 | [4.2–9.9] | no-op vs none |
| **E** depth = pragmatic | 0.7 % | 5.2 % | 0.049 | [3.1–7.9] | chance |
| **E** depth = AND (consensus) | 1.2 % | 5.4 % | 0.055 | [2.9–8.2] | chance |
| **G** length-norm | 1.9 % | 6.3 % | 0.060 | [3.3–9.3] | weak |
| **W** per-tool null-z | 1.9 % | 6.7 % | 0.068 | [3.7–9.7] | weak+ |
| **C3** drop-struct + n-gram + bitmask + null-z | 3.3 % | **10.0 %** | 0.088 | [6.7–13.8] | **best fixed** |

**Reading the table.**

- **Dropping structural tokens is the single most useful sign-space stage.** Nulling
  the JSON scaffold (`{ } " : ,`, `name`, `arguments`, `<tool_call>`, `</think>`) lifts
  Top-5 from 6.9 % → 9.3 % — those tokens are pure common-mode shared by every call and
  every def, exactly as the catalogue predicted.
- **Bitmask whitening is a no-op** (identical to "none") because all 128 bits are
  balanced — the corpus has no constant bits to drop. The common mode is joint
  correlation, which a per-bit mask cannot touch. (PCA-residual whitening — projecting
  out the top joint components — is the unrun variant that *could*; see §9.)
- **Aggregation** matters modestly: consecutive **L2 n-gram > L3 > max-pair ≈ mean ≈
  top-k**. A short run of aligned tokens beats both a single spurious max and a diluted
  mean — but the ceiling is low.
- **Depth fusion** (best-over / sum / AND / single) barely moves the needle; pragmatic
  alone and AND-consensus are at chance. No single depth-pool is the "routing pool".
- **Calibration** (length-norm, per-tool null-z) each adds ~0.5–1 pt by removing the
  promiscuous-def bias — small but real, and they **stack** with drop-structural.
- **The best compound (C3)** stacks the orthogonal winners (drop-structural ⟂ n-gram ⟂
  null-z) for **Top-5 10.0 %, CI [6.7–13.8]** — the lower CI bound clears raw chance
  (5.4 %) but only *just* clears the promiscuity floor (7.6 %). Honest verdict:
  **a real but marginal effect, ~2× chance.**

### Composites tried (catalogue C-recipes, sign-space-feasible)

`C3` (drop-structural → n-gram → whiten → null-z) is the strongest and is in the table.
`C1` (name-vs-name) and `C7` (depth-AND consensus) underperform — name-token isolation
*hurts* here because a single contextualised call-name token agrees only ~83/128 with
its definition counterpart (the domain gap, §4), so isolating it discards the weak
corroborating signal in the surrounding window. The float-resolution composites
(`C2` Hungarian, `C4` GCC-PHAT, `C5` matched filter, `C9` K⊕V) are deferred with §9 —
they need float K, and §4–§6 show the sign-space ceiling is set by the domain gap, not
by aggregation, so they are unlikely to change the verdict on their own.

---

## 4. The decisive diagnosis: a domain gap, not a resolution gap

Put the three oracle/identity numbers side by side:

```
self-match     (def name  → defs )   Top-5 100.0%   ← the metric finds identity perfectly
call-vs-call   (probe     → probes)  Top-5  58.9%   ← Q-sigs carry strong tool identity
probe → def    (probe     → defs )   Top-5  10.0%   ← cross-domain match is marginal
```

The metric is sound (100 %). The Q-signature *does* encode which tool — a call retrieves
**other calls of the same tool** at 58.9 % Top-5 / 32.7 % Top-1, ~30× chance. What
collapses is **call → definition**: the Q vector of a tool *call* (emitted in the
context `<tool_call>{"name":"…`) lives in a different region of sign-space than the Q
vector of the same tool's *definition* (emitted in a catalogue-listing context). Same
token, different context ⇒ ~83/128 agreement instead of the ~120 a "same token" would
give. **This is the catalogue's "Q is contextualised" point, now quantified as the
dominant bottleneck.** Higher bit-resolution does not fix a domain mismatch; matching
against the right domain does.

---

## 5. Search-limited or signal-limited?

The catalogue's own decision rule: if oracles are high but the scorer is low →
search-limited (keep tuning); if oracles are also low → signal-limited.

- **best-per-probe oracle = 38.1 % Top-5.** If we could pick the best of 13 scorers per
  probe, we'd quadruple the best single config. So the *probe→def* task is **partly
  search-limited** — there is recoverable signal — **but no single sign-space config
  reaches it**, and coordinate-ascent tuning **overfits**: train MRR 0.104 → holdout
  0.058 (≈ chance), holdout Top-5 collapses to 3.1 %. The effect is at the noise floor
  *for formula tuning on 138 examples*.
- **call-vs-call = 58.9 %** shows the *identity itself* is not signal-limited at all —
  it is abundant. The limitation is purely the cross-domain projection.

**Verdict:** for `probe → static-definition`, signal is present but at the tuning noise
floor (marginal, ~2× chance, won't survive formula search). For `probe → past-probes`,
signal is strong and robust. **Switch the gather target, not the resolution.**

---

## 6. Negative controls & the promiscuity floor

- **Label shuffle** (permute tool labels, keep signatures, re-score best config):
  Top-5 **7.6 %**, MRR 0.066. This is **above** raw chance (5.4 %) — a real *promiscuity
  floor*: some tools score high against everything (length / structure bias), so they
  land in the top-5 regardless of the true label. **Any honest "is it real" test must
  clear ~7.6 %, not 5.4 %.** Only the drop-structural / null-z configs do. (Per-tool
  null-z calibration is precisely the tool that erases this floor, and it is in the best
  compound.)
- The shuffle does **not** reproduce the real config's signal (best fixed config 10.0 %
  > 7.6 %), so there is no harness leakage — the residual ~2.4 pt is genuine.

---

## 7. Confusion structure

Best fixed config, when wrong: of 261 errors, **4.2 %** pick a **same name-family** tool
(e.g. `ssh_session_exec` ↔ `ssh_session_exec_async`) vs **3.6 %** expected at random.
Barely above random ⇒ the sign-space probe→def signal, where it exists, is **coarse**:
it is *not* reliably resolving within a family, only occasionally landing in the right
neighbourhood. (Contrast with memory-matching in §8, which resolves the exact tool.)

---

## 8. Probe-and-gather + stateful lock-on (the product mechanism)

This reframes the task the way the system actually runs: a **stream** of turns, and for
each turn the system must *gather* the provenance (which tool to inject), from three
combinable sources, each min-max normalised per query so their scales don't swamp:

- **def-match** — current probe vs the 93 static definitions (sign n-gram, drop-struct),
- **memory** — current probe vs **past probes already seen** (call-vs-call, online/causal),
- **lock-on** — a per-tool **activation** boosted when a tool is used and **decayed**
  every step (`a ← 0.6·a` each turn, `+1` on use), so a tool in active use stays hot and
  idle tools fall away. This is the "boost attended tools, iterate away decaying ones"
  ask, as an attractor over the candidate set.

After each prediction the true tool is revealed, appended to memory, and activations
update. Online Top-1 / Top-5, on a **bursty** ordering (a tool's calls grouped — a
session that reuses a tool) and a **shuffled** ordering (no locality):

| Variant | bursty Top-1 | bursty Top-5 | shuffled Top-1 | shuffled Top-5 |
|---|--:|--:|--:|--:|
| def-match only (stateless) | 3.3 % | 9.3 % | 3.3 % | 9.3 % |
| **memory only** (kNN past probes) | 22.5 % | 35.8 % | 20.6 % | 32.3 % |
| def + memory | 15.4 % | 28.6 % | 13.1 % | 27.1 % |
| **memory + lock-on** (no def) | **35.5 %** | **48.0 %** | 16.7 % | 30.9 % |
| def + memory + lock-on | 33.1 % | 46.9 % | 10.8 % | 27.1 % |
| chance | 1.1 % | 5.4 % | 1.1 % | 5.4 % |

**What this says.**

1. **Memory is the robust core.** Matching the current call against remembered past
   calls gives **Top-1 ~21 % / Top-5 ~33 %, independent of stream order** — ~19× chance
   Top-1. This is the single most important practical result: the gather should retrieve
   from a **memory of past (probe → tool)** associations.
2. **Lock-on is a session-coherence prior.** On bursty/session streams (the realistic
   case — a coding session hammers a tool) it lifts memory to **Top-1 35.5 % / Top-5
   48.0 %**. On a uniformly-shuffled stream it is roughly neutral (slightly helpful for
   memory-only, a small cost when stacked with noisy def-match). It is an *attractor*,
   not a standalone signal: it amplifies temporal locality and must not be the only
   term, or non-repetitive streams regress.
3. **Static def-match is the weak link** — adding it to memory *lowers* accuracy
   (memory-only 35.8 % → def+memory 28.6 % Top-5). Definitions inject the domain-gap
   noise of §4. **Drop def-matching from the gather; keep memory + lock-on.**

**Recommended gather policy:** `score(tool) = memory_kNN(probe, past_calls_of_tool)`
`+ λ · activation(tool)`, with `activation ← decay·activation` each step and a boost on
use; **no static-definition term**. Seed memory from the labelled captures (or a short
warm-up), and the system "locks on" to the tools a session is actually using.

---

## 9. Real Q·K (float) — run, and it beats the sign proxy

The catalogue's #1 priority was "real Q·K on the name token". It is now **run** on the
probe side. The R16 block carries float Q (`q[]`) and float K (`d[]`); extraction is
validated bit-exact (128/128 syn-sig agreement, §1). We compute the **actual attention
score** `Q_probe · K / √d`, summed over the 4 KV heads, max over the name-window token
pairs — replacing the 1-bit sign agreement — in the **memory setting** (probe Q → other
probes' K, the §8 winner; both sides are R16):

**All 48 layers swept** (m = 242 probes), single-layer and combined:

| config | Top-1 | Top-5 | MRR |
|---|--:|--:|--:|
| sign proxy (1-bit) call-vs-call | 32.7 % | 58.9 % | 0.453 |
| real Q·K, best single layer **L36** | 37.2 % | 58.7 % | 0.473 |
| real Q·K, best Top-5 single **L26** | 31.0 % | 61.2 % | 0.446 |
| real Q·K, 6 provenance layers (3,7,20,24,36,40) | 41.3 % | 62.8 % | 0.509 |
| **real Q·K, all 48 layers summed** (no selection) | **48.8 %** | **77.3 %** | 0.621 |
| real Q·K, top-8 layers (selected on data) | **51.2 %** | 74.8 % | **0.620** |

**Three findings.**

1. **Real Q·K beats the 1-bit sign proxy by a wide margin** — all-48 summed is
   **Top-1 48.8 % / Top-5 77.3 % vs 32.7 % / 58.9 %** (+16 pt Top-1, +18 pt Top-5). The
   sign quantisation was discarding roughly a third of the recoverable signal.
2. **Layer choice matters enormously, and the syn/sem/prag layers were the wrong ones.**
   The per-layer Top-1 profile: layers **0–11 are near-noise** (≤2 %, except a couple),
   the signal rises through the mid-stack and **peaks across L24–L39** — best single
   **L36 (37 % Top-1)**, **L26 (61 % Top-5)** — then tails. The production provenance
   depths (syn=L3⊕7, sem=L20⊕24, prag=L36⊕40) include two near-dead layers (L3, L7) and
   miss the strong band L26–L33; that is a large part of why the pooled-depth sign scan
   read at chance. **Match on the L24–L39 band, or just sum all layers.**
3. **Combining layers compounds.** No single layer exceeds 37 % Top-1, but **summing all
   48 reaches 48.8 % Top-1 / 77.3 % Top-5** with *no* layer selection (the honest,
   leak-free number); selecting the top-8 by Top-1 reaches 51 % Top-1 (mild
   selection-on-data — validate on a held-out split before trusting the extra ~2 pt).
   Independent per-layer evidence stacks, exactly the catalogue's "channel-fuse" bet.

### Full memory-setting battery (all complete)

With the validated float Q/K/V extraction, the entire catalogue's float ideas were run in
the memory setting (probe → other probes), all-48-layers summed, m = 242 probes:

| family | config | Top-1 | Top-5 | MRR |
|---|---|--:|--:|--:|
| **A** channel | Q·K | 48.8 % | 77.3 % | 0.621 |
| **A** channel | **K·K** (#6) | 52.9 % | 73.1 % | 0.625 |
| **N** channel | V·V (#61) | 41.7 % | 60.7 % | 0.510 |
| baseline | Q·Q (float) | 25.6 % | 49.6 % | 0.370 |
| **M** head | per-head Q·K, **head 3** (routing head) | 57.0 % | 74.0 % | 0.650 |
| **M** head | head 0 / 1 / 2 | 42.6 / 33.9 / 32.2 % | — | — |
| **H/F** agg | max-pair | 48.8 % | 77.3 % | 0.621 |
| **H/F** agg | **logsumexp** (soft attention readout) | 50.4 % | 76.9 % | 0.628 |
| **H** agg | mean-pair | 31.4 % | 65.7 % | 0.468 |
| **A** metric | cosine | 37.6 % | 74.8 % | 0.539 |
| **B** whiten | mean-sub Q·K | 58.7 % | 75.6 % | 0.664 |
| **B** whiten | **mean-sub K·K** | 62.4 % | 76.0 % | 0.689 |
| **B** whiten | mean-sub V·V | 42.6 % | 61.2 % | 0.522 |
| **B** whiten | **mean-sub K·K + remove top-8 PCs** (#9) | **64.5 %** | **78.9 %** | **0.712** |
| **B** whiten | mean-sub K·K + remove top-3 PCs | 64.0 % | 75.6 % | 0.699 |
| **J** consensus | per-tool centroid K (nearest-centroid) | 17.8 % | 21.9 % | 0.217 |

**What the battery says (each an orthogonal lever, and they stack):**

1. **K·K is the best channel** (52.9 % Top-1 raw) — the *key* is a cleaner content
   representation than the query-key cross or the value. **Q·Q is weakest even in float**
   (25.6 %), confirming the production Q-vs-Q scan was the wrong operation at the root,
   not just under-resolved. V carries identity too but less.
2. **There is a single routing head** — head 3 alone (57 % Top-1) beats the 4-head sum
   (48.8 %); the other heads dilute it. (Family M / Markov-expert intuition confirmed at
   the head level, not just the layer level.)
3. **Whitening is the biggest lever.** Subtracting the per-layer corpus mean (the joint
   common-mode) lifts K·K 52.9 → 62.4 % Top-1; removing the **top-8 principal components**
   lifts it further to **64.5 % Top-1 / 78.9 % Top-5**. This is the family-B hypothesis
   vindicated in float space — what the *sign*-space bit-mask could not do (all bits
   balanced) the float PCA does cleanly.
4. **logsumexp ≥ max ≫ mean** for window aggregation; **cosine < dot**; **nearest-centroid
   consensus collapses** (averaging keys destroys the per-instance signal — keep the
   instance memory, do not summarise it).

**Best overall (memory setting): mean-sub + top-8-PC-residual K·K, all 48 layers ⇒
Top-1 64.5 % / Top-5 78.9 %** — ~60× chance Top-1, ~2× the sign proxy. The product gather
should use *this*, not sign agreement.

### Signal composition — fusing Q·K with K·K (confirmed: > either alone)

Q·K (query-key cross) and K·K (key content) are partly independent views, so fusing
them should help. Per-query **z-normalise** each (so their scales match), then linearly
combine (mean-sub, all-48 summed):

| fusion | Top-1 | Top-5 | MRR |
|---|--:|--:|--:|
| Q·K only | 58.7 % | 75.6 % | 0.664 |
| K·K only | 62.4 % | 76.0 % | 0.689 |
| **0.4·Q·K + 0.6·K·K** | **65.3 %** | 76.4 % | **0.706** |
| 0.5·Q·K + 0.5·K·K | 64.0 % | 76.9 % | 0.701 |
| Q·K + K·K + V·V (equal) | 59.5 % | 70.2 % | 0.646 |
| RRF(Q·K, K·K) | 60.7 % | 76.4 % | 0.682 |

**The fusion beats both singletons across the whole α ∈ [0.25, 0.6] range** (not a
single-point artefact), peaking at **~0.4·Q·K + 0.6·K·K → Top-1 65.3 % / MRR 0.706**
(+3 pt Top-1, +0.017 MRR over K·K alone). Two caveats: **adding V·V hurts** (V is a
noisier channel — §battery — and drags the fusion down), and **z-normalised linear
fusion beats reciprocal-rank fusion** here (RRF underperforms because it discards the
score margins that the K·K vs Q·K disagreement encodes). Net: the gather should fuse a
**K·K-weighted blend of Q·K and K·K**, not V.

#### Dynamic (per-query) weighting — limited headroom

Could a per-query α (gated on some indicator) beat the fixed blend?

| approach | Top-1 | Top-5 | MRR |
|---|--:|--:|--:|
| fixed 0.4·Q·K + 0.6·K·K | 65.3 % | 76.4 % | 0.706 |
| **ORACLE** (per-query pick the better metric) | **67.4 %** | 77.3 % | 0.726 |
| margin-gated fusion (unsupervised confidence) | 64.5 % | 76.4 % | 0.703 |

Two findings: (1) **the oracle ceiling is only ~2 pt above the fixed blend** — Q·K and
K·K mostly agree on *which* queries they get right, so there is little per-query
complementarity left for a gate to exploit; the fixed K·K-weighted blend is already near
the perfect-gate ceiling. (2) **A confidence-margin gate slightly *underperforms* the
fixed blend** — a metric's own sharpness is *not* a reliable indicator of its
correctness (a distractor tool can yield a confident-but-wrong peak — the promiscuity
effect), so "trust the sharper metric" misfires. **Conclusion: keep the fixed blend.**
Realising the last ~2 pt would need a *supervised* gate (learn when Q·K beats K·K from
[margin_q, margin_k, entropy, Q-norm, layer-agreement] features), and the upside is
bounded at ~2 pt — not worth the added model/overfitting risk for the product.

### probe → **definition** real K·K — RUN, and the domain gap survives full resolution

The literal cross-domain test. The definition K is sealed quantised (Q8_KS); it is
dequantised with the **existing** kernel — `BlockQ8_KS::to_float` per `(head,palette)`
sub-band (the same block kernel `QTensor::dequantize` dispatches to), walking the
interleaved K/V gid layout. Correctness is gated by **def-K self-match = 100 % Top-1**
(a def's own dequantised K retrieves its own tool), so the float K_def is trustworthy.
Then probe K (R16) is matched against def K (mean-pooled per layer, all-48 summed):

| probe → definition, real K·K | Top-1 | Top-5 | MRR |
|---|--:|--:|--:|
| raw K·K | 1.2 % | 4.1 % | 0.054 |
| **mean-sub K·K** | **5.4 %** | **11.2 %** | 0.103 |
| best single layer (mean-sub) | 3.3 % | 10.3 % | 0.088 |
| — for contrast — | | | |
| sign-space probe→def (best fixed) | 3.3 % | 10.0 % | 0.088 |
| **probe→probe memory** whitened K·K | **64.5 %** | **78.9 %** | 0.712 |

**The decisive result: full-resolution K does NOT bridge the call↔definition domain
gap.** With provably-correct float def-K (self-match 100 %), probe→def real K·K reaches
only **Top-1 5.4 % / Top-5 11.2 %** — statistically the *same regime as the 1-bit
sign-space scan* (~3 % / 10 %), ~2× chance, and an order of magnitude below the
same-domain memory match (64 % / 79 %). This confirms §4 conclusively: **the bottleneck
was never bit-resolution — it is a genuine representational domain gap.** The Q/K of a
tool *call* (emitted in `<tool_call>{"name":"…`) and the Q/K of the same tool's
*definition* (catalogue-listing context) occupy different regions of activation space,
and neither full resolution, whitening, nor layer/head selection closes that gap **with a
same-space match**.

> **⚠ SUPERSEDED — see §16.** This "dead end" conclusion held only for *same-space*
> matching (probe and definition compared in one shared representation). A **learned
> cross-domain alignment** (CCA, §16) bridges the gap almost completely: probe→definition
> jumps from ~11 % to **Top-5 95 % / Top-1 90 %** on holdout. The gap is real but
> **linearly bridgeable** — it was the wrong *transform*, not missing signal. Definition
> matching is **not** a dead end after all; with alignment it is the strongest path and it
> enables cold-start gather. Read §16 as the corrected conclusion.

---

## 9b. A scan-friendly production formula — LSH of whitened K

The winning metric (whitened K·K cosine) is a float dot over thousands of dims per token
— too heavy to scan a 10 M-token substrate directly. But the *current* production formula
already shows the way: `sign(Q)` + XOR + popcount is just **1-bit-per-dim LSH of Q** whose
Hamming distance approximates `Q·Q`. It is at chance only because `Q·Q` is the wrong
signal. Point the same trick at the right signal:

**Persist a `b`-bit sign-random-projection LSH of the mean-subtracted K, concatenated over
the routing band (L24–L39).** Hamming distance over the code ≈ `cosine(whitened K)` — the
metric that wins. Scan is the identical XOR+popcount kernel; whitening + projection happen
once at ingest (seal time), not per scan.

| representation (routing band L24–39, dim 8192) | Top-1 | Top-5 | bytes/token |
|---|--:|--:|--:|
| float whitened-K cosine (ceiling) | 72.3 % | 80.6 % | — |
| LSH 128 bits (= today's budget) | 48.0 % | 67.4 % | 16 |
| LSH 256 bits | 54.7 % | 75.9 % | 32 |
| **LSH 512 bits** | 63.8 % | **79.3 %** | 64 |
| LSH 1024 bits | **68.6 %** | 79.3 % | 128 |
| *today's `sign(Q)`* | ~1 % | ~5 % | 16 |

- **Same scan kernel, same regime as the current BDP.** At 512 bits = 8 `u64`
  popcounts/comparison; over 10 M tokens ≈ 80 M popcounts/query-token ≈ low tens of ms.
  `b` is the single speed/accuracy dial.
- **Even at today's 128-bit budget**, whitened-K LSH reaches Top-5 67 % vs ~5 % for
  `sign(Q)` — swapping only the *signal* (not the kernel or bit width) moves it off chance.
- **For a top-k gather, 512 bits is effectively lossless** (Top-5 79.3 % ≈ the 80.6 %
  ceiling). 1024 bits closes most of the Top-1 gap.
- **Better float formula found en route:** routing band L24–39 + **cosine** beats all-48
  dot (72.3 % vs 65.7 % Top-1) and halves the concat dim (8192 vs 24576) → cheaper ingest.
- **Storage:** 64 B/token × 10 M = 640 MB (4× today's 16 B). 256 bits (32 B, 320 MB) still
  gives Top-5 76 %; dropping to the top-4 layers shrinks the projection further.

**Net production recipe:** at seal time, for each token compute mean-subtracted K over
L24–39, project onto `b≈512` fixed random hyperplanes, store the sign bits. At scan time,
XOR+popcount against the corpus codes, aggregate per-token (max / span) and fold to tools
— the exact shape of today's scanner, now carrying the K·K signal.

### Better hits at the same bit budget — fix the input, not the hash

Can we get more accuracy out of the *same* `b`? Tested at 256 bits (routing band):

| technique | Top-1 | Top-5 | MRR | verdict |
|---|--:|--:|--:|---|
| random hyperplanes | 51.6 % | 74.2 % | 0.613 | baseline |
| super-bit (orthonormalised) | 51.3 % | 73.0 % | 0.608 | no help |
| variance-top coords | 52.7 % | 74.6 % | 0.624 | marginal |
| variance-WEIGHTED (×std) | 52.8 % | 72.5 % | 0.622 | trades Top-5 |
| **variance-NORMALISED (÷std)** | **56.1 %** | 72.7 % | 0.638 | best Top-1 |
| **variance-norm p=0.5 (÷√std)** | 55.6 % | **75.4 %** | **0.647** | **best balance** |
| XOR-fold parity bits | ~52–58 % | ~74 % | — | ≈ random (seed noise) |
| supervised top-256 of 2048 pool (holdout) | 45.7 % | 59.8 % | 0.533 | **overfits — hurts** |

**Findings.** (1) **Cleverer projections don't beat random** — super-bit/orthogonalisation
is a no-op because random hyperplanes are *already* near-orthogonal in 8192-dim; XOR-folding
just wanders within the ±3 pt single-draw seed noise. (2) **Greedy supervised bit-selection
overfits and hurts** — 46 % on holdout vs 55 % for random; single-bit tool-discriminability
doesn't generalise across 93 tools (a learned hash like ITQ might, greedy selection is a
trap). (3) **The one principled win is whitening the *vectors* before projecting**: dividing
each dim by its std (diagonal whitening, p=0.5) gives **+4 pt Top-1 and +1 pt Top-5 at the
same bits, free at scan time** (an ingest-side rescale). It is the cheap diagonal analogue of
the PCA-residual whitening that won the float battery — it suppresses the common-mode
high-variance directions that random projection would otherwise waste most bits encoding.

**Lesson: "fix the input, not the hash."** Spend the bit budget via **mean-sub +
variance-normalise, then random-project** — not rotations, folding, or selection.

#### PCA-sign — the big bit-efficiency win (and ITQ)

The clearest improvement comes from choosing the **hyperplanes to be the data's own
principal axes**. Sign of the projection onto the top-`b` PCs (one bit per principal
component) destroys the redundancy that random hyperplanes suffer — each bit encodes one
orthogonal high-variance direction. At 256 bits (same tokens):

| method | Top-1 | Top-5 | MRR |
|---|--:|--:|--:|
| random hyperplanes | 54.1 % | 73.0 % | 0.622 |
| variance-norm p=0.5 | 55.0 % | 72.3 % | 0.622 |
| **PCA-sign (top-256 PCs)** | **66.5 %** | **78.1 %** | **0.718** |
| PCA-residual (drop top 32 PCs) | 66.3 % | 77.2 % | 0.712 |
| *float cosine ceiling* | 72.3 % | 80.6 % | 0.759 |

**PCA-sign gives +12 pt Top-1 over random at the same 256 bits**, closing most of the
random→float gap (Top-5 78.1 % is within 2.5 pt of the 80.6 % ceiling). The mechanism is
confirmed by PCA-residual: dropping the top-32 common-mode PCs barely changes the result
(66.3 %), because PCA-sign gives each PC exactly one bit — the common-mode never dominates
the code, unlike in a random *dot* where high-variance dims swamp the sum.

**ITQ** (PCA + a learned rotation that minimises quantisation error) is the textbook
method, but a hand-rolled version here was **numerically unstable** (read chance↔35 %
across runs — FP non-determinism in the parallel reductions perturbs the PCA basis and the
polar-decomposition rotation amplifies it). It is also, per the literature, only a small
gain over PCA-sign — and PCA-sign already captures the win. **Recommendation: use PCA-sign,
not ITQ.**

**Updated production recipe:** learn the top-`b` PC basis once (offline, from a token
sample) over the routing-band mean-subtracted K; store the `b×D` basis; at seal time
project each token onto it and store the `b` sign bits. Scan is the identical XOR+popcount.
This is the same cost shape as random LSH but +12 pt Top-1 for free — the only added
ingest work is the (one-time-learned) basis instead of a fixed random matrix.

#### Locking the bit budget — 256 bits (inductive holdout sweep)

PCA basis learned on **train** tokens, projected onto all, evaluated on **held-out**
queries (the honest generalisation test):

| bits | holdout Top-1 / Top-5 | MRR | random @ same bits (3-seed) |
|---|--:|--:|--:|
| **256** (32 B) | **70.0 % / 83.6 %** | 0.753 | 55.8 % / 77.3 % |
| 384 (48 B) | 68.7 % / 82.0 % | 0.747 | 56.9 % / 78.3 % |
| 512 (64 B) | 67.0 % / 82.6 % | 0.737 | 60.0 % / 79.5 % |

**Decision: lock 256 bits (32 B/token).** Three reasons. (1) **256 is optimal — more bits
hurt.** Top-1 *declines* (70.0 → 68.7 → 67.0) and Top-5 is flat as `b` grows: the
discriminative signal lives in the top ~256 PCs, and PCs 256–512 are lower-variance noise
that dilutes the popcount. (2) **It generalises** — the held-out number (70.0 % / 83.6 %)
matches or exceeds the transductive 66.5 %, so PCA-sign is not overfit. (3) **PCA-sign @
256 ≈ the float ceiling** (70 % / 84 % vs ~72 % / 81 % float) and **beats random @ 512**
(60 % / 79.5 %) at half the storage — binarisation is nearly free once the bits are
PCA-aligned. Storage: 32 B/token × 10 M = 320 MB.

## 10. Recommendations

> **Superseded by §23–§24.8.** These recommendations are from the float-`K·K` / definition-matching
> exploration, before the pivot to decode→decode `Q·Q` (§22) and the shipped folded-signature belief
> system. They record what that line of work concluded; the design that actually shipped is the
> production summary at the top of this document. Kept for the record.

1. **Ship the gather as memory + lock-on, not definition matching.** Build a per-tool
   memory of past tool-call probes; gather by matching the current probe's name-window
   against that memory with **whitened K·K** — mean-subtract the per-layer corpus key,
   remove the top ~8 principal components, dot keys, sum all 48 layers, max over the
   window. That reaches **Top-1 64.5 % / Top-5 78.9 %** (vs ~33 % / 59 % for the sign
   proxy and ~3 % / 9 % for definition matching). Add a decaying per-tool activation that
   boosts on use (lock-on) for session-coherent gains.
2. **Match on keys, whitened, over all layers — not Q-vs-Q on the syn/sem/prag pools.**
   Three independent wins stack: **K·K** (key is the cleaner content channel; Q·Q is the
   worst even in float), **whitening** (subtract the corpus mean + top PCs — the biggest
   single lever, +12 pt), and **all-layer fusion** (no single layer beats ~37 % Top-1;
   the sum reaches 49 %, whitened 64 %). A single **routing head (head 3)** carries most
   of the raw signal if you want a cheaper match.
3. **Drop the static-definition gather term** (it injects domain-gap noise) except as a
   **cold-start fallback** when a tool has never been seen.
4. **If you must keep sign-space def-matching:** drop structural tokens + per-tool
   null-z, lower `hit_threshold` to the ~80–88 band. These are the only sign-space
   stages that produced real lift. Do **not** formula-tune on the small labelled set —
   it overfits (holdout → chance).
5. **Definition matching, via a learned cross-domain alignment, is the *strongest* path — see §16.**
   ~~A dead end~~ — that held only for *same-space* matching. A CCA alignment (per-token ×
   per-copy pairs, d=160, r=32 canonical dims) lifts probe→definition from Top-5 11 % to
   **Top-5 95 % / Top-1 90 %** on holdout, *beating* the same-domain memory match (79 %).
   Recommendation: **learn the section→probe CCA map once, project sections into the canonical
   space, and route there** (mean-pooled probe) — this delivers cold-start gather (no history
   needed) and is the best path measured.
   Same-space def-matching (sign / raw K·K) remains a dead end; the alignment is what unlocks it.
6. **Route the canonical space with an IVF tree, not a flat scan or an exact metric tree — see §17.**
   Cluster the section vectors into k₁ ≈ √N groups; at query, score the k₁ centroids and
   flat-scan only the top **p ≈ 5** clusters. This holds the full **90 % / 95 %** quality at a
   **2.3× compute cut** (→ **O(√N)**: ~17× at 10 k tools, ~170× at 1 M), and is **needle-safe** —
   the most-isolated tools recall **100 %** *earlier* than average, because isolation gives a
   needle a sharp cluster centroid the probe locks onto. Do **not** use an exact cone-bound
   metric tree for the speed-up — at 32-d its bounds are too loose to prune (it visits *more*
   than the flat scan); keep it only as a lossless O(N) fallback.

---

## 11. Reproduction

**Shipped design (§23–§24.8)** — the `belief-*` subcommands of `substrate_inspect` reproduce every
current number against a workspace `.substrate/substrate.log`:

```bash
cd <workspace>   # holds .substrate/substrate.log
E="cargo run -q --release -p candle-conversation --example substrate_inspect --"

$E belief-eval                              # §80: leave-one-out Tool-1/3/5, MRR, per-tool, hardest
$E belief-probe   <turn-stream-id>          # one probe's full ranking vs the gallery
$E belief-dissect <turn-stream-id>          # §81: per fold-group + per-token breakdown
$E belief-sweep   --probe-tokens 256        # §80.2: min_score × budget threshold sweep
$E belief-decay   --chunk 64                # §82: recency decay vs quality weight vs needle gate
```

**Research record (§1–§22)** was produced by `zend/examples/tool_provenance_research.rs` and
`calibrate_alignment.rs`. The former (a coordinate-ascent sweep over the `ScoreConfig` space below)
was retired when the design locked; the harness loaded the substrate once (~seconds for 8.5 GB),
cached all probes + defs in RAM, and ran the reference scorers, family table, oracles, controls,
confusion, and lock-on simulation in one rayon-parallel pass with a deterministic xorshift seed.

### Knobs realised in `ScoreConfig`

`ProbeSel` {AsstWindow(W), NameOnly, WholeAsst, UserWindow(W)} · `DefSel` {Whole,
NameOnly} · `Whiten` {None, BitMask} · `Depth` {Syn, Sem, Prag, BestOver, Sum, And} ·
`Agg` {Max, TopK(k), Mean, Ngram(L)} · `Calib` {None, LengthNorm, NullZ} · drop-structural
{on, off}. Coordinate ascent searches this space on train; the family table pins
representatives a priori. Float-metric stages (K·K, V·V, Hungarian, matched-filter,
GCC-PHAT) are the documented §9 extension.

---

## 16. Cross-domain alignment — the section path revived (the headline result)

> **⚠ MEASUREMENT CORRECTION (2026-06-25).** The §16 headline **90 % / 95 %** is a
> **per-tool** number — the eval (`phase16_cca`, `ho_mean` per tool) averaged *every*
> held-out call of a tool into a single query before routing (n = 76 tools), and was
> measured on the earlier substrate that carried **two def copies** per tool. Routing
> **per-probe** — each call on its own, which is the cold-start production path —
> using §16's *own* code on a single-copy catalog gives **Top-1 ~76 % / Top-5 ~88 %**
> (d=160 r=32: 75.9 / 87.9, n = 116). This is independently reproduced by
> `zend/examples/calibrate_alignment.rs` (k-fold per-probe **74 / 84**). The two
> levers between the figures: **per-tool averaging** buys ~7–14 pt by smoothing
> call-noise across a tool's calls, and the **second def copy** buys ~5–7 pt of Top-1
> (item-side CCA rank 186 vs 98). **Read every "90 / 95" in §16 as the per-tool /
> 2-copy figure; the honest production cold-start number is ~76 / 88.** The recipe and
> conclusions below are unchanged — only the metric is corrected.

§4/§9 concluded probe→definition was a dead end because every *same-space* match (sign,
PCA-sign, raw K·K, Q·K, V·V, K⊙V, all 48 layers, every aggregation, with and without
discriminative-token selection) sat at chance-to-~11 % Top-5. That conclusion was wrong in
scope: it showed the call and definition representations are *not directly comparable*, not
that the signal is absent. A **learned linear alignment** between the two domains closes the
gap almost entirely.

### 16.1 Section-only token-importance formula (the tree's "ignore vs branch" rule)

First, a probe-free formula over the tool sections' K alone, scoring each token's
distinctiveness = `− max dot (global mean-sub K, layers ~L25–L42) against tokens in OTHER
sections` (low cross-section similarity = distinctive → branch; high = common → ignore).
Validated by name-vs-rest AUC (does it rank tool-name tokens above scaffold):

| section-only formula | AUC |
|---|--:|
| K dot (mean-sub), band L24–39 | 0.66 |
| **K dot, best layers (L25–L42), best-16 concat** | **0.74** |
| V dot / K⊙V / cosine / per-position / PCA-residual | ≤ 0.62 (all worse) |

Lesson: **K, global mean-sub, dot, max, late-middle layers** — the section signal is
fragile; PCA-residual and per-position whitening *remove* it (the section's top PCs and
positional structure *carry* tool identity), opposite to the probe side.

### 16.2 The alignment: ridge → CCA, mean-pooled → per-token

Learn a map from section space → probe space on **train (probe, section) pairs**, in a
PCA-reduced space, then route **held-out** probes (different probes, same tools) to sections.

| method | pairs | dim | **Top-1** | **Top-5** |
|---|---|---|--:|--:|
| same-space K·K (no alignment) | — | — | 5 % | 11 % |
| ridge regression, mean-pooled | 87 | d=64 | 50 % | 77.5 % |
| CCA, per-token (avg copies) | 715 | d=80, r=16 | 86.2 % | 95.0 % |
| **CCA, per-token × per-copy (tightened)** | **1430** | **d=160, r=32** | **90.0 %** | **95.0 %** |
| CCA, per-token × per-copy | 1430 | d=192, r=32 | 91.2 %\* | 95.0 % |

\* d=192 is past the section-side rank (186 copies) — Top-1 still nudges up but it's the
overfit edge; **d≈160 is the robust lock.**

**probe→definition routing: ~11 % → Top-1 90 % / Top-5 95 % on holdout — now *clearly better*
than the memory path (70 % / 79 %).** Three compounding levers: **per-token pairs** (probe
token-level variation) × **both def copies** (1430 pairs, 186 section exemplars → higher
canonical rank), **CCA** canonical whitening (far tighter than ridge), and **dimensionality**
(Top-1 climbs 80→128→160 = 86→89→90 %). The Top-5 signal saturates at just **r ≈ 16 canonical
directions**; Top-1 wants **r ≈ 32**. Holdout-validated (alignment learned on train pairs,
evaluated on different probes); Top-1 plateaus at ~90 % as `d` approaches the 186-copy
section-side rank — the honest saturation signature. **Negative result: per-token *eval*
voting (max over the holdout probe's tokens) does *not* beat mean-pooling the probe — mean is
as good or better everywhere, so route on the mean-pooled probe vector.** Locked recipe:
**d=160 PCA-reduce, r=32 canonical dims, per-token×per-copy training pairs, mean-pooled probe
at query, corr-weighted cosine.**

### 16.3 What this overturns and unblocks

- **The domain gap is real but *linearly bridgeable*.** It was the wrong transform, not
  missing signal — exactly as a low-rank CCA map demonstrates. §4/§9's "dead end" is corrected.
- **Cold-start gather is solved.** A tool call routes to the correct tool *definition* at
  **Top-1 90 % / Top-5 95 %** with **zero memory** — just the catalog + a one-time-learned CCA
  map. This is the prize the section path always promised, and it beats memory.
- **The tree-over-sections is unblocked and worth building.** Routing now works at 95 %, so:
  §16.1's formula picks each node's branch/ignore tokens, the CCA map bridges the domain, and
  the tree gives sublinear scan. Build it in the **r=32 canonical space** — the vectors are
  tiny (32-d) and the match is fast.

### 16.4 Locked recipe

Offline (once): **PCA-reduce to d=160**, then learn CCA directions `Wx` (probe), `Wy`
(section) on **per-token × per-copy** pairs (each train-probe token paired with each def copy
of its tool — 1430 pairs), keeping **r=32** canonical dims. At ingest: project each section
copy through `Wy` into the 32-d canonical space. At query: **mean-pool** the probe's tokens,
project through `Wx`, and match by **correlation-weighted cosine** (fold-by-tool, max over the
tool's copies) — directly, or via a tree built over the canonical section vectors.

Two measured negatives that simplify the recipe: **per-token eval voting does not beat
mean-pooling** the probe (so mean-pool — cheaper), and **dimensionality past d≈160 overfits**
(it exceeds the 186-copy section-side rank; Top-1 plateaus at ~90 %).

**This is the corrected headline of the whole study: the gather can route a call to its tool
either by same-domain memory (~79 % Top-5) or by CCA-aligned definition matching (Top-1 90 % /
Top-5 95 %, cold-start) — and the latter, once the alignment is learned, is both stronger and
needs no history.**

---

## 17. The gather tree — sublinear routing that keeps quality *and* finds needles

§16 made probe→definition routing work (Top-1 90 % / Top-5 95 %) but as a **flat** scan
(compare the probe against every section). At production scale (thousands of tools, a
10M-token substrate) that O(N) scan is the cost we must cut — **without losing the rare,
isolated "needle" tools**, which is exactly what naive pruning destroys. Phase 17 builds the
tree in the locked **r=32 canonical space** and measures all three axes against the flat
baseline.

### 17.1 Exact metric-tree pruning fails at 32-d (a clean negative)

A cone-bounded branch-and-bound tree (recursive 2-means; each node carries an **admissible**
upper bound — the cosine cone from centroid + angular radius — so a branch is pruned only when
its *best possible* member can't beat the current 5th-best):

| | Top-1 | Top-5 | comparisons |
|---|--:|--:|--:|
| flat | 90.0 % | 95.0 % | 186 |
| **B&B exact** | **90.0 %** | **95.0 %** | **218** |
| B&B ≤60 / ≤40 / ≤24 | 56 / 39 / 15 % | — | 60 / 41 / 25 |

Exact B&B reproduces the flat result *exactly* (the bound is lossless) and — see §17.3 —
protects needles perfectly. **But it gives no compute win**: at 32 dimensions the cone bounds
are too loose to prune, so it visits *more* than the flat scan (218 > 186), and capping the
budget collapses recall. Lesson: **exact tree-pruning is the wrong tool in 32-d** (curse of
dimensionality). The win comes from *approximate coarse routing*, not exact bounds.

### 17.2 IVF coarse-routing — the compute win

Cluster the section vectors into **k₁ ≈ √N** groups (spherical k-means); at query, score the
k₁ centroids and flat-scan only the **top-p** clusters' members:

| mode | Top-1 | Top-5 | comparisons | vs flat |
|---|--:|--:|--:|--:|
| flat | 90.0 % | 95.0 % | 186 | 1× |
| **IVF p=5** | **90.0 %** | **95.0 %** | **81** | **2.3×** |
| IVF p=3 | 88.8 % | 92.5 % | 56 | 3.3× |
| IVF p=2 | 87.5 % | 92.5 % | 42 | 4.4× |
| IVF p=1 | 82.5 % | 85.0 % | 30 | 6.2× |

**p=5 reproduces the full flat quality (90 / 95) at 81 comparisons — a 2.3× cut even at this
tiny N**, and the saving grows as √N: `cmp ≈ k₁ + p·(N/k₁) = O(√N)`. Projected: **~17× at
10 k tools, ~170× at 1 M** (multi-level clustering → O(log N) if needed). Quality is a smooth
knob in p, so the recall/compute trade is tunable per deployment.

### 17.3 Needle-in-haystack — solved, and needles are *easier*

Ranking tools by **isolation** (max cosine from a tool's copies to any *other* tool's copy;
low = outlier needle) and evaluating recall on the 12 most-isolated (cross-tool cos 0.26–0.44):

| | exact / p=3 | p=2 | p=1 |
|---|--:|--:|--:|
| **needle Top-1 (IVF)** | **100 %** | 91.7 % | 75.0 % |

**Needles hit 100 % at p=3 — before the *average* tool peaks (88.8 %).** The intuition we set
out to guarantee holds and then some: the very isolation that defines a needle gives it a
**sharp, distinctive cluster centroid**, so the probe that matches the needle matches its
centroid strongly and routes there *early*. Coarse routing finds needles *sooner* than common
tools, the opposite of the pruning failure mode. (Exact B&B also gives needles 100 % via the
admissible bound — so both modes are needle-safe; IVF is the one that's also cheap.)

### 17.4 Recipe and verdict

Offline: project sections into the r=32 canonical space (§16.4), cluster into k₁ ≈ √N groups,
store centroids + member lists. Query: mean-pool the probe → Wx → unit; score the k₁
centroids; flat-scan the top **p≈5** clusters; fold-by-tool (max over copies). All three goals
met: **quality held at 90 / 95, compute cut 2.3× (→ √N asymptotically), needles at 100 %.** The
exact cone-bound B&B stays available as a lossless, needle-safe fallback when a deployment wants
zero approximation and can pay the O(N) cost.

---

## 18. The mixture-of-rotations — making the gather map relation-general

**Status:** measured, 2026-06-25. Harness: `zend/examples/calibrate_alignment.rs`
(load-once, model-free, CPU, against the rebuilt 15 GB substrate carrying the 200-case
calibration corpus, 99 tool defs, and 217 tool-call probes). This section supersedes the
"cold-start from generic content" premise of §16.3 and replaces it with a **bank of
per-relation rotations** governed by a two-level combine law.

> **One-line answer.** A single CCA rotation is **relation-specific**, not a universal
> model-geometry bridge — a map fit on one query→target relation routes a *different*
> relation at chance, even with identical preprocessing. But that is not a dead end: keep
> a **bank of rotations** and fuse their scores. The fuse law is two-level and measured:
> **across relations, MAX** (diverse experts self-select — the wrong rotation scores low,
> out of distribution, and loses with no false positives); **within a relation, SUM**
> (redundant per-head/per-phase experts vote, and consensus beats any single view). With
> that, the calibration is **a learned *library*, not one map** — extend it per gather
> relation without retraining the others.

### 18.1 Measurement correction to §16 (per-tool → per-probe)

See the correction box at the head of §16. In brief: §16's **90 / 95** is a **per-tool**
figure (the eval averaged *all* of a tool's held-out calls into one query) on a **2-copy**
catalog. Re-run **per-probe** (route each call individually — the cold-start production
path) on §16's own code with a single-copy catalog, the honest number is **Top-1 ~76 % /
Top-5 ~88 %** (d=160 r=32: 75.9 / 87.9), independently reproduced by the calibrate harness's
k-fold per-probe **74.7 / 84.3**. The per-tool averaging buys ~7–14 pt; the second def copy
buys ~5–7 pt of Top-1. Everything below uses **per-probe holdout** and treats ~76 / 88 as the
real tool-routing bar.

### 18.2 A CCA rotation is relation-specific (the fair 2×2)

Two relations: **tools** (a tool *call* → its *definition*, name↔description) and **corpus**
(a generic case's final-turn *response* → its *previous turn*, topic↔topic). Fit a map on
each, with **one shared preprocessing** (mean-pooled, structural-filtered, whole spans, K
over L24–40), and cross-test:

| | corpus self-retr. (chance 0.6/2.8) | call→def (chance 1.0/5.1) |
|---|--:|--:|
| **corpus-trained map** | 11.1 / 31.1  *(in-domain ✓)* | 0.5 / **6.0**  *(transfer = chance)* |
| **tool-trained map** | 1.1 / **3.3**  *(transfer = chance)* | 99 / 100  *(in-sample)* |

Both off-diagonal (transfer) cells sit at chance. The corpus map is *demonstrably valid*
(31 % in-domain), so its chance on tools is a genuine transfer failure, not a broken fit.
**A linear CCA is a rotation between two *specific* token-content distributions; applied to
a third, inputs land out-of-distribution and route at chance.** (An earlier, broken-fit
version of this experiment wrongly concluded "generic content doesn't transfer" — the fix
that validated the corpus fit was mean-pooling the *whole* structural-filtered span instead
of per-token over a 12-token window where `\n\n` and a single topic word made the CCA
memorize noise. With the fit valid, the transfer failure is the real, narrower finding.)

### 18.3 What changes between relations — layers? heads? directions?

Per-layer and per-head in-domain Top-5, corpus vs tools:

```
per-LAYER          per-HEAD
        corpus tools           corpus tools
L24-30   5-12  16-27   head 0   22.2  23.1
L31      25.6  23.1    head 1   17.8  21.3
L32      27.8  26.9    head 2   18.9  20.4
L33-39   3-14  19-28   head 3   17.8  23.1
```

- **Heads — not the differentiator.** Both relations spread evenly across all four KV
  heads (head 0 marginally best for both). (Aside: §9's "head-3 routing head" was a
  per-token name-window artifact; under mean-whole pooling the signal is even.)
- **Layers — partial.** Corpus topic-matching is **sharply concentrated at L31–L32**; tool
  routing is **distributed across L24–L39** — but L31–L32 are strong for *both*.
- **The K directions — primary.** At the *shared* best layers/heads, the full-band maps
  still cross-route at chance. The relation lives in the **canonical directions** (the
  rotation), not in which layers/heads carry signal. The two relations are different
  rotations of a **shared subspace** — which is exactly why a bank can work.

### 18.4 The bank + the two-level combine law

**Across relations → MAX, no false positives.** Bank = {corpus rotation, tool rotation};
score each item under both, take the max. Result (per-probe / per-case holdout):

| | own-rotation | bank-max |
|---|--:|--:|
| tools holdout | 13.0 / 23.1 | 13.0 / 22.2 |
| corpus holdout | 11.1 / 31.1 | 11.1 / 31.1 |

bank-max ≈ own-rotation: the **wrong** rotation in the bank contributes *low* scores
(out-of-distribution) and loses the max. This is a *consequence* of §18.2's
relation-specificity, and scores are comparable for free because every signature is unit
(score is a cosine in [−1,1] regardless of rotation). So a bank of per-relation rotations is
relation-agnostic and extensible.

**Within a relation → SUM, not max.** Redundant experts (same relation, different
phase/head views) must be fused by **consensus**, not by letting the loudest single view
decide. Scanned 6 budget distributions × 9 fuse operators on corpus self-retrieval (Top-5):

```
split         sum   max   min   med   p75   p90  top2  top3   lse
[ 0, 0,32]  63.3  40.0  41.1  50.0  50.0  40.0  48.9  55.6  61.1   ← best
[ 0,16,16]  57.8  36.7  41.1  50.0  50.0  44.4  41.1  50.0  56.7
[11,11,10]  57.8  35.6  35.6  50.0  55.6  43.3  40.0  53.3  60.0
[16, 8, 8]  55.6  38.9  24.4  46.7  51.1  46.7  50.0  53.3  56.7
```

**Sum wins every distribution.** The operator ordering is a clean consensus→single-vote
spectrum: `sum (= mean) > lse (smooth-sum) > top3 > median/p75 > top2 > min > max/p90`.
Max is the *worst* within-relation — the mirror image of it being *best* across-relations.

### 18.5 Per-head consensus + concentrate the budget on the strongest phase

Two levers fall out of the budget scan (32 canonical dims/head, distributed across the three
decode phases user / user+think / think+resp, sum-combined; corpus self-retrieval Top-5):

1. **Per-head decomposition + sum is the big lever (+21 pt).** `think+resp` as one
   full-band rotation is 42.2; decomposed into **4 per-head rotations and summed** it is
   **63.3**. Each head is a semi-independent vote; summing them beats the entangled joint
   rotation.
2. **Concentrate, don't spread.** The best distribution is **[0,0,32]** — the whole budget
   on the strongest phase. Adding the weaker phases *hurts* even under sum
   (`[0,0,32]`=63.3 > `[11,11,10]`=57.8 > full 3-phase 12-bank=48.9); `[32,0,0]` (user only)
   is near-chance. Phase ordering is everything: **think+resp ≫ user+think > user**.

Corpus-relation progression: chance 2.8 → single full-band response 31 → single full-band
think+resp 42.2 → **per-head sum on think+resp 63.3** Top-5 (~22× chance Top-1).
(`d≈24` is the overfit ceiling on 90 cases, so the 32-dim budget caps at r=24/head.)

**Granularity is a sweet spot at per-head — locked.** Slicing the band into rotations at
every granularity (each fit on the think+resp phase, sum-combined; `concat = #rotations × r`
is the per-token signature length that gets stored and scanned):

| granularity | rotations | concat dims (cost) | Top-5 |
|---|--:|--:|--:|
| full-band | 1 | 24 | 42.2 |
| **per-head** | **4** | **96** | **63.3** |
| per-layer | 16 | 384 | 48.9 |
| per-(layer,head) | 64 | 1536 | 10.0 |
| per-32-block | 256 | 6144 | 0.0 |
| per-K-value | 8192 | 8192 | 63.3 |

Accuracy is **non-monotonic with a peak at per-head**. full→head helps (specialisation
without losing the cross-dimension mixing a rotation needs to bridge a domain gap);
head→finer hurts — the mid-range collapse is partly a denoise artifact (a fixed PCA `d=24`
barely reduces a 128- or 32-dim expert, so it over-fits), but even regularised, finer can at
best *match* per-head while costing more. The **per-K-value extreme is the surprise**: a 1-d
"rotation" does **not** become a plain sign scan — it learns the **per-dim correlation sign**
(`sign(corr_i)·sign(q_i)·sign(t_i)`, summed), a learned per-dim sign code that *ties* per-head
at 63.3. But the cost column is decisive: it needs an **8192-dim concat signature vs
per-head's 96 — the same accuracy at 85× the storage and scan compute.** This is §9b
restated: a learned *dense* per-head rotation is radically more bit-efficient than per-dim
sign. **Decision: lock granularity at per-head** — best accuracy *and* the cheapest concat
code (96 dims). Going finer is strictly worse: it overfits (mid-range) or pays 85× for no
accuracy gain (per-K-value).

### 18.6 The calibration architecture and recipe

The calibration is **a library of rotations**, not one map:

- **Per gather relation** (tool routing, conversational-context recall, memory, …): a
  sub-bank of **(strongest-phase × per-head)** rotations, **SUM-combined** into one
  consensus score. Granularity is **locked at per-head** (§18.5): the per-token signature is
  the concatenation of the 4 per-head canonical sigs (≈ 4 × r = 96 dims) — the cheapest code
  that reaches peak accuracy; coarser (full-band) entangles the heads, finer overfits or pays
  ~85× for no gain. Each relation keeps its *own* best span recipe (per-token name-window for
  tools per §16; mean-whole-filtered for corpus) — the unit-cosine signatures make mixed
  recipes composable.
- **Across relations:** **MAX** over the sub-banks, so the active relation self-selects with
  no false positives and no need to know the relation a priori.
- Every signature is a unit vector, so the §9b PCA-sign / XOR scan and the §17 IVF tree
  apply unchanged, **per rotation**.

**What this overturns.** §16.3's "learn one cross-domain map from generic content, transfer
to tools" is false — a single rotation is relation-specific. The corpus of 200 generic
conversations does **not** serve tool-gather; it serves a *different* gather relation
(retrieving conversational context), at which it is genuinely strong (63 % Top-5). The win
is the **mixture**: calibrate a rotation per relation, fuse by sum-within / max-across.

**Caveats.** (1) Numbers are single-copy / small-holdout (90 corpus cases, 217 probes) — the
*ordering* (sum≫max within, max-no-false-positives across, per-head≫full-band,
concentrate≫spread) is the robust result, not the absolute points. (2) The cross-relation
max property is shown for 2 relations; verify it holds as the library grows to N. (3) The
tool relation's per-head-sum recipe should be confirmed to clear its own 76/88 bar (the
tool numbers here use the weaker mean-whole recipe for an apples-to-apples 2×2).

---

## 19. Two retrieval modes — the semantic bridge and the copy lane

**Status:** measured, 2026-06-25. Harness: `zend/examples/calibrate_alignment.rs`
(windowed-probe lanes; production `WhitenedK` for the memory lane). §16–18 built the
**cross-domain** map (call → definition). This section adds the **second model** the
codebase already ships — [`WhitenedK`](../candle-conversation/src/provenance/gather/memory.rs),
a *same-domain* match — and shows the two are **orthogonal retrieval modes**, how to fuse
them without bleeding the calibrated signal, and why the same-domain lane is essential.

> **One-line answer.** Provenance gather has **two lanes, not one**. The **catalog lane**
> (cross-domain CCA) is a *semantic bridge*: it maps a call onto its definition. The
> **memory lane** (`WhitenedK` = mean-sub + PCA of the routing-band K, L2-normalised) is the
> *copy / induction* match: it matches the current K against a **prior literal occurrence**
> of the same thing. They are specialised to two different attention circuits — each is at
> **chance** on the other's job — so a gather needs both. Fuse them by **union of each lane's
> top-K** (monotonic), never by merging into one ranking (which bleeds the calibrated lane).

### 19.1 Protect the signal — union, not merge

The naive idea is to drop the same-domain match into the same signature and take a max
(score-max or reciprocal-rank-max) over a shared item list. **It bleeds**, measured on the
windowed tool task (108-probe holdout, chance 1.0 / 5.1):

| fuse | catalog Top-1/5 | after fuse |
|---|--:|--:|
| score-max(raw-cross-domain ⊕ rotated) on shared DEFS | 39.8 / **48.1** | 26.9 / **35.2** |
| **UNION**(catalog top-5 ∪ memory top-5) | 39.8 / **48.1** | — / **48.1** |

Score-max **cut Top-5 from 48.1 to 35.2**: a chance lane still has real *maxima over wrong
items*, and `max(rotated_low, chance_high)` lets those win. Worse, you cannot gate it by
confidence — **89 % of windowed calls have memory cosine ≥ 0.8** (every `<tool_call>{…}`
shares the same JSON scaffold), so absolute confidence can't tell a reachable tool from a
confident-wrong one. The fix is to **not merge rankings at all**: gather the **union of each
lane's top-K**. Union is monotonic — adding a lane can only add hits — so the calibrated
catalog signal **cannot bleed by construction** (48.1 → 48.1). The catalog is the protected
floor; the memory lane only ever adds candidates.

### 19.2 The lanes are orthogonal — the copy-head 2×2

The two `SignatureModel`s × two target item-sets. Each lane wins on its **own** mode and
sits at **chance** on the other (Top-1 / Top-5, chance 1.0 / 5.1):

| | catalog lane (CCA) | memory lane (WhitenedK) |
|---|--:|--:|
| **target = DEFS** | **39.8 / 48.1** | 2.8 / 19.4 |
| **target = PAST CALLS** | 1.9 / 7.4 | **38.0 / 40.7** |

- **catalog × defs (48.1)** — the *semantic bridge*: call → its definition.
- **memory × past-calls (40.7)** — the *copy / induction recall*: call → a prior occurrence
  of the same call. (40.7 is the **reachability ceiling** of the 50/50 split — ~59 % of
  tools have no same-tool training call — not a recipe limit; on reachable tools it is
  near-perfect.)
- **catalog × past-calls (7.4 ≈ chance)** — the headline. With **no definition to bridge
  to**, the cross-domain map has nothing to say; only the memory lane works there.
- **memory × defs (19.4)** — above chance only because a call and its def share the literal
  tool-name tokens (a weak copy overlap), and ≪ the catalog's 48.1. The copy lane can't do
  the bridge.

### 19.3 Why the copy lane is essential

The memory lane is the retrieval-time counterpart of the model's own **induction / copy
heads** — "I've seen this exact K before → attend to it → copy the continuation." We build it
on the routing-band **K** precisely because that is where induction heads do their prefix
matching. The cross-domain CCA is a *different* circuit (a learned semantic map), and **for
literal recall there is nothing to bridge to**: a variable defined 50 k tokens ago, a
repeated identifier, an entity named earlier, a fact stated upstream, a tool already
called — none have a "definition", so the catalog is at chance (the 7.4 cell) and the copy
lane is the **only** signal. On the narrow tool→def benchmark the two coincide (a called tool
usually also has a def), which is why the memory lane looked like a subset in §18 — that is
the *task* hiding the distinction, not the lane being redundant. Pull the target off
"definition" and onto "prior occurrence" and the catalog drops to chance while memory holds.

### 19.4 Recipe and verdict

- **Two lanes per gather.** Catalog = the §16–18 cross-domain CCA (per-head, sum-combined).
  Memory = `WhitenedK` over the routing-band K (mean-sub + PCA, L2-normalised; common-mode
  `drop` optional — marginal once the span is the §16 name window, not the washed
  mean-whole call: windowing alone lifts the memory lane **14/20 → 40.7**).
- **Item-set separation.** The memory lane scans the **prior-occurrence store**; the catalog
  lane scans the **definition catalog**. The chance cross-domain raw match never scores a
  definition, so it cannot pollute the calibrated lane.
- **Union, not merge.** Gather each lane's top-K and **union** the candidate sets — monotonic,
  so the catalog signal is an untouchable floor; merging rankings (score- or RR-max) bleeds
  and cannot be confidence-gated (all calls look alike).
- **Verdict.** Provenance gather is two orthogonal modes — a *semantic bridge* and a *copy
  lane* — unioned. The copy lane carries general unbounded recall; the catalog carries
  cold-start routing to a stable catalog. Neither covers the other (7.4 vs 40.7 on copy;
  19.4 vs 48.1 on bridge).

**Caveats.** (1) On this sparse 50/50 holdout the union equals the catalog (memory's hits are
a subset there) — the *additive* gain needs data where the lanes diverge (a distinctive call
pattern with an ambiguous def, or a novel/uncalibrated def); the 2×2 proves the modes are
distinct, not that they add on *this* task. (2) Numbers use cosine signatures; the
storage-cheap **sign / XOR** form (§9b) is the production code and is ≈-equal at enough bits.
(3) The 40.7 memory ceiling is the holdout's reachability limit; a production store (many
calls per tool) reaches far more.

---

## 20. The calibration model through the tool test — 3-phase parse of the 186 tool conversations

**Status:** measured, 2026-06-26. Harness: `zend/examples/calibrate_alignment.rs`
(`§20` block). This runs the calibration model through the *tool-routing* eval — route each
held-out tool **conversation** to its tool's **definition** (call→def, the
`tool_provenance_research` final test) — but with the tool conversations parsed into the
**same three query phases as the corpus calibration** (user / user+think / think+resp), each
phase scored separately. tpr's own tool-side calibration is skipped; the rotation is supplied
externally.

> **One-line answer.** The 3-phase cross-domain pipeline is **correct and works** — but only
> when the rotation is fit on a relation that *matches* the test. A rotation fit on the
> generic conversation corpus (response→prior-turn) routes the tool conversations at
> **chance**; the identical pipeline fit on **conversation→def** routes them at **65.6 Top-5**
> on the **user-prompt phase**. The phase that carries the tool signal is the **user prompt**
> (the task description), not the call itself.

### 20.1 Parsing the 186 tool conversations like the corpus

Each tool case is a 2-turn conversation `[user prompt][assistant: <think> + <tool_call>]`,
but the two turns are **separate streams in one timeline**, and the prompt is stored **without
an `<|im_start|>user` marker** (so `parse_messages` drops it). Reconstruction: group turn
streams by `timeline_id`, concatenate tokens + bands in `turn_index` order, locate the
assistant turn by the **first `<|im_start|>`**, and take everything before it as the user
prompt. Then the identical phase recipe as the corpus: `user` / `user+think` / `think+call`,
each mean-pooled over structural-filtered tokens. All **186** conversations reconstruct.

### 20.2 Result — training relation decides everything

Route each phase → tool def (cross-domain CCA, per-head sum; chance 1.0 / 5.1):

| rotation fit on … | user | user+think | think+resp |
|---|--:|--:|--:|
| **generic corpus** (phase → prior-turn), full / no holdout | 1.1 / 4.8 | 1.1 / 4.3 | 0.5 / 3.2 |
| **the tool conversations** (phase → its def), 50/50 holdout | 23.7 / **65.6** | 25.8 / 63.4 | 9.7 / 21.5 |

The generic-corpus rotation is at **chance on every phase** — confirming §18.2's
relation-specificity holds even with the full 3-phase parse and the matched preprocessing:
"response→prior-turn" does not bridge "tool-conversation→def."

The **wiring is proven correct** by the second row: the *same* reconstruction, the *same*
3-phase parse, the *same* per-head pipeline — only the training pairs change — and it routes
at **65.6**. So the chance result is the training relation, not a bug.

### 20.3 The user-prompt phase carries the tool signal

The phase ordering **inverts** the corpus self-retrieval finding. For the corpus relation
(§18.5), `think+resp` was strongest (the response carries the most context about the prior
turn). For tool routing, **`user` and `user+think` are strongest (65.6 / 63.4) and
`think+resp` is weakest (21.5)** — the **task description** ("Open an HTTP session with base
URL …") is what maps to the tool definition, while by the time the model is emitting the
`<tool_call>` JSON it has committed to a specific call and carries less def-matching signal.
This is the actionable lever: route on the **user-prompt phase** of the in-flight turn, before
the call is formed — which is also the moment a gather is most useful (cold, pre-call).

### 20.4 Verdict and open fork

- **The pipeline works** — 3-phase parse, cross-domain per-head rotation, call→def routing —
  reaching **65.6 Top-5** on the user-prompt phase (50/50 holdout over the 186 tool
  conversations).
- **The generic corpus does not transfer** (chance), no matter the phase or parse — the
  rotation must be fit on a relation whose target *is* a definition-like context.
- **Open fork (next path):** the training source. Either fit on **conversation→def** directly
  (the 65.6 path, but train/test must be separated to avoid in-sample leak), or find a
  corpus whose "contextual input" is definition-like so a single generic rotation transfers.
  The 50/50 split above is a holdout estimate, not the production fit.

---

## 21. Back to basics — does raw attention select the definition? The weighted sign-XOR-pop diagnostic

**Status:** measured, **frozen 2026-06-27**. Harness: `zend/examples/calibrate_alignment.rs`
(`§21`–`§31` blocks; run with `S21_ONLY=1 S31=1`, against the 15 GB substrate — 186 tool
conversations, 93 tool defs, with **live-captured per-token Q** from the decode). **This is a
single-token diagnostic checkpoint, not a generalization result** — read §21.5 before quoting
any number here.

> **One-line answer.** Strip the §16–§20 CCA rotation away and ask the irrefutable question:
> can *plain* attention — `sign(Q)·sign(K)` XOR+popcount, the production BDP kernel — pick a
> tool call's definition out of the catalog? The naive answer (§21–§29) is **no, ~chance** —
> the same call↔def domain gap as §4, reconfirmed across every readout (per-token MAX/SUM,
> real float Q·K, noise-subtracted, peak-both-sides). But the test had **two bugs, not a
> missing signal**: (1) the definition candidate list was **polluted with system-prompt
> sections** — the attention-sink `frame` outscored every real tool by raw magnitude, so the
> scan was ranking the *system frame*, not the definitions; and (2) the readout **threw away
> the magnitude**, counting a near-zero-`|Q|` dim's random sign as a full bit. Fix both — clean
> defs + a per-head_dim **integer importance weight** `w[d]=f(|Q[d]|)` under a 3-level
> percentile roll-up — and one fixed call token separates its correct def at **+9.6σ, rank 1 of
> 93**, with **layer-normalized squared weighting** the decisive lever (uniform 4.4σ → head-norm
> sq 6.8σ → layer-norm sq 9.6σ). The machinery demonstrably separates the def *for a single
> token*; whether the **one** winning formula generalizes across cases is the next test, not yet
> run.

### 21.1 What went wrong first (§21–§29) — raw-sign call→def is at chance, every which way

Same-space (no rotation) call→def, the four decode phases × 93 defs (chance 1.0 / 5.2):

| readout (harness §) | best Top-1 | best Top-5 | verdict |
|---|--:|--:|---|
| per-token **MAX** + 0.4·Q·K+0.6·K·K fusion (§21.2) | 2.2 % | 7.5 % | ~chance |
| per-token **SUM** + same fusion (§22) | 0.0 % | 6.5 % | chance |
| per-head K·K match distribution, cohen's **d** (§23) | — | — | d ≈ 0.01–0.05 (faint) |
| noise-subtracted routing (§25) | — | — | **identical to §22** — znorm removes the constant |
| three-level aggregation scan, k-fold CV (§26) | — | — | chance (raw-sign aggregation has no signal) |
| real **float Q·K** mean-pool (§27) | 0.0 % | ~5 % | chance |
| per-token max vs sum, real Q·K (§28) | 0.0 % | 5.9 % | chance |
| true peak `max over (call tok × def tok)` Q·K (§29) | 0.0 % | 8.1 % | ~chance |

Every honest readout sat at the promiscuity floor. The §25 noise-subtraction being **bit-identical**
to §22 is the tell that closed off that avenue: per-query z-normalisation already removes any
constant common-mode offset, so subtracting a calibrated noise vector before znorm is a no-op (as
predicted). §26's aggregation-formula scan, read under k-fold CV (select on train, score
held-out), collapsed to chance — confirming there is no magic roll-up of *raw* signs.

### 21.2 The bug (§30) — the dot product was being run against the system frame

The single-token diagnostic (§30: fix one call token, rank the correct def against all others by
its Q·K) exposed why every readout above read at chance: **the candidate "definition" list was
contaminated with system-prompt sections.** The projection that prefills a tool case injects the
system sections (`frame`, `reasoning_stance`, `grounding`, `history_stance`, `tools_overview`, …)
*and* the one relevant tool def. The def-enumeration filter was picking those system sections up as
candidates, and `frame` — the **attention sink** (positions 0–3, the largest-magnitude block in the
whole context) — scored **~8700 vs ~1600** for any real tool. The scan's "winner" was the system
frame on essentially every probe. We were dot-producting the call against the system scaffold, not
the tool definitions. Fix: exclude the named system sections from the def candidate set (match only
real tool sections). This is the §4/§9 "drop-structural" lesson resurfacing one level up — the
attention sink is common-mode that no readout can see past until it is removed from the *candidates*,
not just the tokens.

### 21.3 The rebuild (§31) — clean defs, a weighted sign-XOR-pop atom, a 3-level formula scan

With clean defs, §31 rebuilds the score from scratch on **one fixed call token** (mid-`<tool_call>`),
to make the mechanism legible before trusting any aggregate:

- **Atom** — per head, the **weighted** sign disagreement `score(head) = Σ_d w[d]·[sign(Q[d]) ≠
  sign(K[d])]` over the head's 128 dims, against **only** the def's own tokens. `w[d]=1` recovers the
  plain 128-bit XOR-popcount (the production BDP atom).
- **Importance weight** — `w[d]` is a bounded integer from the per-dim importance `|Q[d]|` (the
  query magnitude *is* the importance of a key dim to the attention, since `Q·K` weights each `K[d]`
  by `Q[d]`; it also kills the random sign-noise of near-zero-`|Q|` dims). Normalised **within a
  group** (head's 128 dims, or layer's 512) and mapped to `[0,8]` — so the atom can never overflow
  (worst-case all-`sum` roll-up ≈ 9.8 M < 2²⁴, exact in f32). Seven formula shapes: `uniform`, `rank`,
  `top` (top-⅛ mask), `lin`, `sq`, `sqrt`, `log`, each under **head-** and **layer-**normalisation
  (`L:` prefix) = 13 weights.
- **3-level roll-up** — def-tokens→head, heads→layer, layers→def, each reduced by one of 16
  percentile/sum/mean formulas (a fine grid `sum, mean, p10…p100`).
- **Scan** — all `13 weights × 16³` combos (≈ 53 k), rank the correct def among 93 by separation
  **z** (`(score−mean)/std`), parallel over the combo space.

### 21.4 The frozen result — weight × formula scan, one token (`telnet_session_list`, tok 75, 93 defs)

| z | rank | weight | head | layer | def | reading |
|--:|--:|---|---|---|---|---|
| **−9.59** | **1** | **L:sq** | p75 | p10 | p10 | **frozen best** — layer-norm squared |
| −6.75 | 1 | sq | p100 | p10 | p25 | head-norm squared (prev best) |
| −5.48 | 1 | L:sq | p98 | p10 | p25 | |
| +4.41 | 1 | uniform | p25 | p50 | p98 | best **without** any weight |
| −1.45 | 1 | uniform | sum | sum | sum | the plain production atom (baseline) |

Three findings, each a lever and they compound:

1. **Clean defs + a high-tail roll-up already separate the def** — `uniform` reaches +4.4σ (rank 1)
   with `head=p25, def=p98`, vs the plain `sum/sum/sum` atom at −1.45σ. The roll-up wants the *tail*
   (low/high percentiles), never the mean — `sum/sum/sum` buries the signal.
2. **The magnitude weight sharpens it** — reintroducing `|Q[d]|` as the per-dim multiplier lifts
   |z| 4.4 → 6.8 (`sq`). The steep `sq` shape wins (concentrate the score on the few high-`|Q|`
   channels the probe actually attends to); `sqrt`/`rank`/`top` also beat `uniform`.
3. **Layer-normalisation is the decisive grouping** — normalising `|Q|` across the whole 512-dim
   layer (not per-head) lifts |z| 6.8 → **9.6**. Per-head normalisation forced every head's top dim
   to weight 8, flattening the very thing that discriminates; layer-norm lets the **loud heads
   outweigh the quiet ones**, and the routing signal lives in a *few dominant heads per layer*. The
   `L:*` variants own the top of the table.

The negative z is correct and meaningful: the right def has a **low** weighted-mismatch (it *matches*
the probe at its important dims), so it lands at the bottom of the disagreement distribution, rank 1.

### 21.5 Caveats — what this is and is not

- **One token, one case, best-of-53 k.** Every knob here (clean defs · weighted atom · `sq` ·
  layer-norm · the percentile grid) was tuned against a **single probe token of a single case**
  (`telnet_session_list`). With 53 k combos the pure-noise max-|z| floor over 93 defs is ≈ 4.7. The
  `uniform` 4.4σ sits **at** that floor (plausibly selection luck); **L:sq's 9.6σ is ~2× the floor**,
  so it is real signal *on this token* — but a single token cannot tell signal from a formula fit to
  it. **Do not read 9.6σ as an accuracy.**
- **Case order is now deterministic** — the tool cases were being reconstructed in `HashMap`
  iteration order, so "case 0" was a different tool each run; it is now sorted by timeline id
  (`case 0 = telnet_session_list`, stable), so this checkpoint is reproducible.
- **The honest next test (not yet run): cross-case k-fold.** Score **every** case (one probe token
  each) under each combo, **pick the combo on a train split, evaluate Top-1/Top-5 on held-out
  cases**. That is the only thing that converts "9.6σ on telnet" into a number we can trust, and it is
  the immediate next step. Until then, §21 stands as a **mechanism proof** (the weighted sign-XOR-pop
  atom *can* separate a def, and which knobs matter), **not** a routing result. The §16–§20 CCA path
  (Top-5 65.6, §20) remains the measured cross-domain bar this raw-sign line is trying to reach
  without a learned rotation.

---

## 22. The pivot — decode→decode Q·Q retrieval (the product path)

**Status:** measured 2026-06-28. Harness: `zend/examples/calibrate_alignment.rs`
(`S21_ONLY=1 S73=1` / `S74=1`). 185 tool-call decodes, ~93 tools (~2 instances each),
chance ≈ **Top-1 0.5 % / Top-5 2.7 %**.

**Why this supersedes §1–§21 for the product.** Everything above (and the §22–§72
exploration in the harness) matched a decode **call** against a tool **definition** —
`Q·K`, *cross-domain*. That is the representational wall: ~19 % Top-1, bridged only by a
learned CCA (§16). The product loop does not need it. The loop is:

```
project substrate every N tokens → greedy-draft the next step →
the draft's Q is the query → scan stored decode Qs → bring matched context into scope →
decode N tokens for real
```

The draft's query reaches for **past decoded content of the same kind** — `Q·Q`,
*same-domain*. No domain gap. This is the regime the early memory note already saw
(call-vs-call ~59 % Top-5); §73–§74 measure it cleanly and find the aggregation that
makes it work.

> A separate dead end recorded so it is not re-tried: the **RoPE un-rotation** sweep
> (harness `S72`) came up flat (target stuck at rank ~90 for every offset). Reason: the
> KV cache stores K/Q **pre-RoPE** (content) and ropes in-kernel via per-chunk
> `rope_pos` metadata (`types.rs:107`, `get_chunk_refs_with_rope`). There is no RoPE in
> the persisted Q/K to remove — so the §73/§74 cosines are already clean content matches.

### 22.1 §73 — pooled Q·Q holdout retrieval

Pool each decode's content-token Q into one per-head query; hold each out; rank all other
decodes by per-head cosine; does a same-tool / same-family decode land Top-1/5.

| config | Tool-1 | Tool-5 | Fam-1 | Fam-5 |
|---|--:|--:|--:|--:|
| raw cosine, all-band | 39.1 | 51.6 | 53.3 | 66.3 |
| mean-sub, all-band | 38.0 | 52.7 | 51.6 | 67.9 |
| mean-sub, mid-band (L24–39) | 40.2 | 52.2 | 52.7 | 67.4 |
| mean-sub, **SIGN**, mid | 39.7 | 51.1 | 54.3 | 66.8 |

- **It works** — ~39 % Tool Top-1 / ~52 % Top-5 (~70× chance), blind query geometry (the
  tool label only *scores* a match, never makes it).
- **Mean-subtraction is a wash** (38–40 %) — the predicted "remove the common query
  component" lever did **not** materialise once you pool over content tokens. *Do not
  re-try it.*
- **SIGN ≈ cosine** — the 1-bit XOR-popcount form is within a point of full precision.

### 22.2 §74 — per-token Q·Q with cross-token aggregation

Pooling blurs the distinctive name token. Instead score **every** probe (draft) token
against **every** stored decode token, aggregate to a per-case score five ways:

| algo (mid, cosine) | Tool-1 | Tool-5 | Fam-1 | Fam-5 |
|---|--:|--:|--:|--:|
| maxpair (max signal) | 43.2 | 53.5 | 55.7 | 68.1 |
| meanpair | 22.7 | 44.9 | 36.2 | 57.3 |
| bestprobe (Σ per-probe best) | 44.3 | 55.7 | 58.9 | 69.7 |
| **vote (consensus argmax)** | **50.8** | **68.6** | **66.5** | **81.1** |
| consec (longest run) | 47.6 | 67.6 | 61.1 | 79.5 |

**Consensus voting wins big** — Tool Top-1 **39 → 51**, Tool Top-5 **52 → 69**, Fam Top-5
**66 → 81** over pooled §73. The ordering is the lesson: **meanpair is *worst* (22.7)** —
averaging destroys it (and explains §73's mediocrity) — while the two best, `vote` and
`consec`, are the ones that let individual tokens speak. The right model is **"how many
draft tokens *independently* point at the same stored decode,"** not "is the mean query
similar." It is robust to the boilerplate tokens being noise — the distinctive tokens
out-vote them.

**Binarization is free.** SIGN/BDP preserves the consensus lift:

| vote | Tool-1 | Tool-5 | Fam-1 | Fam-5 |
|---|--:|--:|--:|--:|
| mid, cosine | 50.8 | 68.6 | 66.5 | 81.1 |
| mid, **SIGN** | 49.7 | 66.5 | 65.9 | 80.5 |
| all, **SIGN** | 50.3 | 68.1 | 65.9 | 80.0 |

Sign costs ~1 point; band choice (mid vs all) is in the noise.

### 22.3 The validated design

1. Store **per-token Q** (the substrate already does).
2. **Greedy-draft** N tokens at the projection boundary.
3. **Per-token sign/BDP scan** — each draft token's Q vs stored decode Qs, XOR-sign +
   128-bit popcount per head (head_dim = 128 = one word/head), aggregated over the band.
4. **Consensus vote** — each draft token points at its best-matching stored decode; tally
   per source.
5. Bring the winning context into scope; decode N tokens for real.

Delivered blind, from a model-generated query: **~50 % exact-tool / ~80 % family at
Top-5**, robust to binarization — and it ships on the existing BDP/popcount path.

### 22.4 Caveats

- **185 decodes, ~2 instances per tool.** A typical holdout has ~1 correct match among
  184 distractors; "Top-1" = that one correct decode is ranked #1. This is
  leave-one-decode-out within the corpus, not yet k-fold across a held-out *tool* split.
- **Family ≫ exact** persists (81 vs 68 Top-5) — the `_session_list` transport cluster.
  For "pull the right neighbourhood into scope," family-level may suffice.
- **Untried headroom:** PCA-before-sign (production-exact BDP), vote-thresholding,
  head-weighting. Not needed for the result; potential lift on exact-tool.

### 22.5 §75 — band sweep + blind per-layer weighting (weak lever)

Two questions left over from §74's mid-band default: which layers carry the
decode→decode signal, and can a blind per-layer weight beat uniform all-48.

**Band sweep (`vote`, cosine, `S74_BAND=lo-hi`).** Unlike call→def K·K (which peaked in
the *late-middle* L24–39 routing band), the decode→decode Q·Q signal is **broad across
early-and-mid layers, and the late band HURTS**:

| band | Tool-1 | Tool-5 | Fam-1 | Fam-5 |
|---|--:|--:|--:|--:|
| 0–48 (all) | **51.9** | 67.0 | **67.6** | 78.9 |
| 0–24 (early) | 50.8 | 68.6 | 64.9 | 81.6 |
| 12–36 | 48.6 | 70.3 | 64.3 | 82.2 |
| 24–40 (mid, inherited) | 50.8 | 68.6 | 66.5 | 81.1 |
| 32–48 (late) | 48.6 | 64.3 | 65.4 | 77.3 |
| 36–48 | 47.0 | 62.7 | 63.8 | 75.7 |

Mechanistic read: Q·Q "what am I querying for" intent is established **early/mid**; late
layers specialise toward the *exact next token*, which is less about topic similarity.
**Decision: all-48** (best Tool-1, no band to defend) — *do not* use a late band.

**Blind per-layer weighting (`S75`).** Weight each layer per draft-token by how decisively
it separates the candidates (from the layer's own similarity distribution, no label):
`std`, `zmax=(max−mean)/std`, `maxmean`, `topgap=max−2nd`. `vote`, all-48:

| weight | cosine T1/T5/F1/F5 | SIGN T1/T5/F1/F5 |
|---|---|---|
| uniform | 51.9 / 67.0 / 67.6 / 78.9 | 50.3 / 68.1 / 65.9 / 80.0 |
| std / maxmean / zmax | = uniform (±0.5) | = uniform |
| **topgap** | 51.4 / **70.8** / 67.0 / **82.2** | 50.3 / 68.6 / 65.9 / 81.1 |

- **Per-layer weighting is a weak lever** — `std`/`maxmean`/`zmax` land on `uniform`. Once
  at all-48 the layers are ~equally informative per query; this matches the earlier
  call→def layer-weighting goes (§21/§36) all being marginal.
- **`topgap` is the only mover, and it's float-only.** Under cosine it lifts Top-5 +3.8 /
  Fam-5 +3.3 (a blind, untuned formula — not an overfit). But under **SIGN it shrinks to
  +0.5 / +1.1 ≈ a wash** — the best-vs-2nd margin it keys on is quantised away by the
  1/128 Hamming steps. **The production BDP/sign path does not get the `topgap` gain.**
- **Tool-1 is unmoved (~50–52 %) under every weight** — re-confirming the exact-tool
  ceiling is the **family confusion**, not a layer-quality problem.

**Net:** the §74 recipe stands unchanged — **uniform all-48, per-token consensus `vote`,
sign/BDP**. `topgap`-weighting is an optional **float-only** refinement (~+3pt Top-5) for
a full-precision rescore; it is not worth carrying on the sign path.

### 22.6 §76 — family / common-hit cross-scoring (negative — proves the wall is real)

The §47–§50 idea, ported to the Q·Q vote: strip the *shared* part of each hit so only the
*distinctive* residual decides the exact tool. Two **blind** levers on each draft token's
per-case score (families detected by pooled-query cosine, **not** the stem label):
`null` (subtract candidate promiscuity = mean blind case-case sim to everything),
`famres` (subtract candidate's blind-family mean = top-m most-similar cases).

| variant (all-48 cosine, m=6) | Tool-1 | Tool-5 | Fam-1 | Fam-5 |
|---|--:|--:|--:|--:|
| **vote** (baseline, do nothing) | **51.9** | **67.0** | **67.6** | **78.9** |
| null (promiscuity subtract) | 47.6 | 65.4 | 64.3 | 77.8 |
| famres (family-residual) | 39.5 | 61.6 | 52.4 | 76.8 |
| both | 42.2 | 62.2 | 53.0 | 77.3 |

**All variants hurt; `famres` collapses Tool-1 by 12 and Fam-1 by 15.** This is a clean
diagnostic of *why*: subtracting the family-shared component **removes the answer**,
because the true tool sits *at* its family's level, not above it — there is no distinctive
residual to expose. Worse, `famres` depresses the whole family relative to unrelated
tools, pushing votes *out* of the family (the Fam-1 67→52 drop). `null` hurts more mildly
(−4) because a candidate's generic attractiveness is **correlated** with its genuine
attractiveness to the right query, so subtracting it removes real signal too.

**Conclusion: the ~52 % Tool-1 ceiling is a genuine information limit, not a fixable
aggregation problem.** The §47–50 cross-scoring helped in the call→def Q·K regime (there
*was* a shared JSON-structure component to strip); it does **not** transfer to same-domain
Q·Q, where the family overlap is the signal itself, not separable noise. Within
`telnet_session_list` vs `tcp_session_list` the *call* genuinely does not carry which
transport it is — no rescoring recovers what isn't there. **Plain §74 `vote` (uniform
all-48) remains the best.** The path to exact-tool is not a smarter scorer; it is either
accepting family-level retrieval (~80 % Top-5 — enough to pull the right neighbourhood
into scope) or adding a signal the call lacks (the arguments / surrounding context).

## 23. The compact folded provenance signature — from wide-Q to a 1536-bit product signature

§22 established decode→decode `Q·Q` as the product path. This section takes it to a
shippable signature: it runs on the **substrate-native, aligned wide-Q**, evaluates it the
way production actually queries (per-projection rolling window, conversation-disjoint
gallery), and distils the 24576-bit per-token `sign(Q)` down to a **1536-bit folded
signature** that is *more* accurate than the full stack. Tool `provenance_probe` (fast,
standalone) is the harness; every number here is k=4-fold, probe/gallery disjoint by
conversation.

### 23.0 Prerequisite — the wide-Q alignment fix

The `WideQSig` capture and the diagnostic gathers computed token position as
`block_idx × 32 + t`, treating every chunk as a full 32-token block. Interior **partial
chunks** (section / glue / reprojection boundaries end a chunk early) made that mislabel
their zero *padding* as "dead tokens" — proven by bytes: the sum of non-zero-K slots equals
the token count exactly, and the dead positions are precisely the padding tails. The model
was never degraded; the gather was misaligned. `gather_wide_sigs` now walks each chunk's
real `[offset, offset+len)` window (via `provenance_chunk_layout`, the same derivation
attention reads), so the record is 1:1 with the tokens. This alone lifted the rolling-window
retrieval **Tool-1 57.5 → 74.6** — the misaligned windows had been indexing wrong tokens.

### 23.1 Setup — per-projection rolling window, conversation-disjoint

- **Probe** = one projection event; its lookup signature is the last **64 tokens** of the
  turn's wide-Q history ending at the projection point (`assistant_content_start +
  end_token`) — exactly the query a production reprojection issues.
- **Label** = the `tools` section that projection locked (substrate-native ground truth).
- **Gallery** = all *other* conversations' probes; k-fold assigns folds per conversation so
  probe and gallery are disjoint by conversation. k=4 is the plateau/operating point.
- **Score** = per-token consensus (each of the 64 query tokens votes for its best-matching
  gallery token by sign-agreement = popcount of XNOR; tally per case, rank).

Baseline (full 48-layer, uniform concat): **75.8 Tool-1 / 94.0 Tool-5**.

### 23.2 Where the signal lives — the top layers, and a single peak (L46)

Per-layer probes (one layer's 4 heads = 512 bits) map it cleanly:

| layers | Tool-1 | note |
|---|--:|---|
| top-2 (46–47) | 77.3 | **beats the full 48-layer stack** |
| **L46 alone** | **78.9** | single best layer — beats every combination |
| L47 | 75.2 | |
| L8, L20 | ~75 | strong middle layers (signal is *not* a clean top gradient) |
| bottom-2 (0–1) | 67.4 | weak |
| L0 | 1.3 | dead (pure lexical/positional) |

The tool-identity signal concentrates in the **late layers**, peaking at **L46**. Early
layers are non-discriminative and, mixed in, *dilute* (top-2 77.3 > all-48 75.8). Critically,
**naive concatenation averages layers rather than stacking them** — the single best layer
beats every concat combination.

### 23.3 Layer folding + the 32-bit decorrelating shift

Dim-aligned XOR of correlated adjacent layers **cancels the shared (discriminative)
component**. Rotating each layer by `position × 32` bits inside its 128-bit head *before*
the XOR staggers them out of phase, combining decorrelated dims:

| stride-4 fold (6144 bits) | Tool-1 | Tool-5 |
|---|--:|--:|
| dim-aligned XOR | 75.7 | 94.0 |
| **32-bit rotate per layer** | **76.4** | **94.5** |

Lesson: **when XOR-combining correlated layers, rotate them out of phase first.** Folding
is only lossy when a group XORs ~≥10 layers (parity of many independent signs → coin flip);
≤~5 layers/group is near-lossless.

### 23.4 Group distributions — spend resolution on the top

Fine-at-top beats fine-at-bottom by ~4–5 points at equal budget. The winning move is to keep
**L46 (the peak) in its own ≤2-layer group** and fold the dead middle/bottom into big noise
groups:

| dist | groups | bits | Tool-1 | Tool-5 |
|---|--:|--:|--:|--:|
| edges `1,2,20,2,20,2,1` | 7 | 3584 | 76.0 | 94.5 |
| `46,1,1` (L0–45 fold · L46 · L47) | 3 | 1536 | **80.2** | **94.9** |

`46,1,1` is the locked distribution — essentially "L46 + L47 vote and reinforce."

### 23.5 Late fusion + z-score self-weighting — breaking the concat ceiling

Every *concat* scheme (weighted or not) tops out at ≈ the best single group, because it
**averages**. The fix is **late fusion**: each group casts its *own* vote (its own argmax),
tallied across groups — so agreeing groups **reinforce**. Late fusion is fragile to noise
groups (uniform late fusion collapses to 70), so votes are self-weighted by **confidence =
z-score of the best match**, `(best − mean)/std` of that group's agreement distribution for
the query token. A discriminative group's true match is a many-σ outlier → strong vote; a
noise group's best is only the ~3–4σ expected-max of random → self-muted. **No external
weights** — it adapts per-query and transfers across tasks.

| method | Tool-1 | weights? |
|---|--:|---|
| full concat | 75.8 | — |
| best single group (L46) | 78.9 | — |
| early-fusion weighted | 76.7 | learned |
| **late fusion, z-score, uniform** | **80.0** | **none** |

The empirical variance is load-bearing — an analytic closed form (assuming bit
independence) degenerates the std and collapses accuracy to ~60. Fewer, well-placed groups
help both speed and accuracy: `46,1,1` (3 groups) beats `edges` (7 groups).

### 23.6 Heads are independent — do NOT fold them

The XOR-shift trick that *helps* on layers *hurts* on heads:

| `46,1,1` | Tool-1 | Tool-5 |
|---|--:|--:|
| heads separate | **80.0** | **95.2** |
| head-fold (shift=32) | 77.9 | 92.6 |
| head-fold (aligned) | 76.4 | 92.0 |

Layers are **redundant** (smooth `Q` evolution → fold-friendly); the 4 KV-head
Q-projections are **independent/complementary** → folding merges distinct information and
destroys it. Keep the 4 heads at full resolution.

### 23.7 The locked design + performance

```
signature : DIST 46,1,1 · 32-bit layer stagger · heads separate
            3 groups × 4 heads × 128 bits = 1536 bits/token  (16× smaller than wide-Q)
scoring   : z-score late-fusion vote, uniform (weight-free), k=4 conversation-disjoint
result    : Tool-1 80.0   Tool-5 95.2   Fam-1 86.8   Fam-5 96.3
```

**Performance (CPU flat scan, 128 k-token gallery, 32 cores):** 4.4 s / **2.2 ms per
probe** — well within the decode-time budget (reprojections fire ~2 s apart). Getting there
from an initial 21.7 s needed two things: (1) the per-group inner popcount must be a
**compile-time-constant width** (`0..8`, not `0..gw`) so it vectorizes (2.4× alone), and (2)
**fewer groups** cut the per-pair bookkeeping (argmax + sum/sumsq) that dominates once the
popcount is cheap. Next order of magnitude: GPU or LSH/kNN indexing. This 1536-bit folded
signature **replaces the wide-Q record** in the substrate.

## 24. From per-projection scoring to a per-conversation online tool belief

§23 scores one probe against a gallery. A live decode is not one probe — it is a **stream**
of reprojections firing on cadence through the user prompt, thinking block, and response.
This section builds the retrieval that a running agent actually uses: an **online belief**
over which tool is in scope, updated at every projection, that both (a) pools weak evidence
into a confident selection and (b) forgets a stale topic when the conversation moves on.

The corpus (unchanged from §23): **93 tools × 4 conversations each = 372 conversations**,
each firing 2–18 projection events → **1950 `ProbeCase`s**. A `ProbeCase` is one projection:
the 64-token rolling wide-Q window ending at the projection point, tagged with the `tools`
section it locked and its decode position. The three-level tree is
`tool → conversation (4) → projection (2–18)`.

### 24.1 Confidence is a predictor, not a selector — two negatives

The per-token confidence of §23 (a token's z-sum for its winning case) is a strong *predictor*
of correctness, so the obvious move is to select with it. Both attempts **lose**:

| selection | Tool-1 | Tool-5 |
|---|---|---|
| sum-of-z consensus (baseline) | **81.0** | 96.5 |
| pick highest-confidence token | 69.1 | 95.4 |
| confidence-gated consensus (drop tokens below gate) | ≤ 81.2 (monotone worse as gate rises) | — |

The baseline sum-of-z **is already a confidence-weighted vote** (each token's contribution is
scaled by its z). Trusting one token throws away the aggregation; gating out weak tokens
removes weak-but-real signal that the wrong-case noise can't overcome in aggregate. There is
no nonlinear reweighting of these z's that beats their plain sum — **the aggregation is the
accuracy**. Confidence's real job is per-selection *trust*, not intra-selection picking.

### 24.2 Per-projection trust + per-link confidence

The actionable confidence attaches to the *projection* and to each *turn it links to*. For a
projection, aggregate the sum-of-z vote per candidate turn (`votes[turn]` — the same
accumulator the baseline ranks by); the per-link confidence is `votes[turn] / n_tokens`
(per-token-normalised so it is comparable across projections). The winner's value is the
projection's **trust**; `(winner − runner-up)/winner` is its **margin**. Both predict
correctness monotonically:

| trust (avg per-token z) | count | Tool-1% |
|---|---|---|
| 1 | 682 | 71.4 |
| 2 | 604 | 83.6 |
| 3 | 286 | 89.2 |
| 5 | 91 | 91.2 |
| 7+ | 63 | ~100 |

Hits average trust 2.51 vs misses 1.64. The per-link confidence decays cleanly from the
selected family into siblings (e.g. a `tls_session_open` pick at conf 4.38 trailed by more
`tls_*` links, then `tcp_*` siblings at conf 0.4) — a downstream layer can treat it as a link
weight and sum a family's links into a family score.

### 24.3 The online walk — decay hurts single-intent data (and why)

Holdout = one **conversation** (leave-one-conversation-out; the gallery is every projection of
the other 371, so each tool keeps 3 exemplars). Walk the holdout's projections in decode order;
each scans the full gallery → a per-tool sum-of-z score. Maintain a running `tool → confidence`
belief and predict its argmax at the end. Per-conversation results:

| policy | Tool-1 | Tool-5 |
|---|---|---|
| last projection only | 87.4 | 99.7 |
| uniform pool (sum, no decay) | 92.7 | 99.7 |
| **max-merge (keep each tool's peak, no decay)** | **94.1** | **100.0** |
| any multiplicative decay λ<1 | monotone worse | — |

Two findings: **pooling the stream is a big win** (87.4 → 94.1, Tool-5 perfect), and **decay
strictly hurts**. The cause is structural — the header reports **0 mixed-tool conversations**:
every projection in a conversation observes the *same* target tool, so there is no stale wrong
belief to forget and decay can only discard good early observations. This corpus cannot, on its
own, justify decay.

### 24.4 The topic-switch stress test + the mechanism sweep (§80.1)

To exercise decay we synthesise topic-switch trials **for free**: because each precomputed
per-projection score already excludes only its *own* conversation, we can concatenate a
different-tool conversation **A** before **B** and require the walk to predict **B**. 1471 A→B
pairs at spread offsets. A *generic* mechanism must score high on **both** regimes; we rank by
`min(single-T1, switch-T1)`.

The sweep covered pooling (sum/max), multiplicative EWMA, leaky-max, position-ramp, simplex,
two-timescale, surprise-gated reset, and **RelLeak** — the additive-with-delay rule
(`acc[t] = max(0, acc[t] − β·max(acc)) + s[t]`). Representative rows:

| mechanism | single-T1 | switch-T1 | min |
|---|---|---|---|
| Sum (pool) | 92.7 | 47.2 | 47.2 |
| Max (pool) | 93.8–94.4 | 48.0 | 48.0 |
| Mult λ=0.70 | 90.1 | 88.9 | 88.9 |
| LeakyMax λ=0.70 | 89.5 | 89.3 | 89.3 |
| surprise-gated (best) | 89.2 | 88.0 | 88.0 |
| **RelLeak β=0.40** | **92.5** | **90.3** | 90.3 |
| **RelLeak β=0.50** | **91.7** | **90.8** | **90.8** |

The switch test does its job: **pooling is not generic** — Sum/Max collapse to ~47% when a
stale topic precedes the target, because they never forget A. Every decay mechanism recovers
the switch, but only **RelLeak** does so at almost no single-intent cost. Two mechanisms that
looked promising failed: **surprise-gated reset** (decay only when the fresh top tool disagrees
with the belief) misfires because a single projection's argmax is only ~70–80% reliable, so it
decays inside stable conversations; **two-timescale** never forgets its slow pool hard enough.

**Why RelLeak wins.** The leak is proportional to the *current leader*, so it self-scales
(no magnitude tuning — scores span p50 8 to max 971). Within a topic, the leader is the correct
tool: subtracting β·leader from itself is a mild `(1−β)` shrink it immediately re-earns with
fresh support, while followers lose β·leader — more than their own small mass — and are pinned
near zero, sharpening the selection. On a switch, the incumbent A receives β·A_mass of leak per
step with **no** fresh support and bleeds out over B's few projections while B accumulates.

### 24.5 The locked belief-update

```
per projection step, fresh per-tool score s, running belief acc:
    m       = max(acc)
    acc[t]  = max(0, acc[t] − β·m) + s[t]     for all tools t
    predict = argmax(acc)          (top-k for the in-scope set)

β = 0.40   (default; single-intent 92.5 / topic-switch 90.3, Tool-5 ~100 on both)
```

β=0.40 is the default over the 0.50 min-optimum because the synthetic switch (A at full
strength immediately before B) is harsher than a real decode trajectory, so preserving
single-intent accuracy is the safer bet; β rises toward 0.50 if abrupt topic drift proves
common. Tool-5 stays ~100 on both regimes throughout — only Tool-1 is contested, and RelLeak
is the frontier for a scale-free, single-parameter, additive belief-update. It is implemented
as `ToolBelief` in `candle-conversation/src/provenance/`, fed by the per-tool aggregation of
`score_provenance_late_fusion` on the live reprojection path.

### 24.6 The selection policy — thresholds, budgets, fix (§80.2)

The belief produces a ranked confidence; the **selection policy** turns it into the set of
sections actually in projection. `SectionSelector` (in `provenance/selection.rs`) wraps the
belief with a per-section [`SectionPolicy`] and per-group [`GroupBudget`]:

- **per-section β** — each section decays at its own rate (a stable system prompt slow, a
  volatile tool fast).
- **`min_score` / `evict_score`** — a hysteresis band: a section is selected once it reaches
  `min_score` and held until it drops below `evict_score` (`< min_score`), so it does not flap.
- **`fix_per_turn`** — a section selected while in the assistant block is *pinned*: immune to
  eviction and budget displacement until `turn_end()`. Pinning is **bounded by the budget** —
  a section can only pin if the group's projected set has room, so pins never exceed `max`.
- **`GroupBudget { min, max }`** per collection / substrate layer — `max` keeps the strongest
  members (pinned first, then by score); `min` force-fills from the top even below `min_score`.

Swept on the same regimes (β=0.40, all tools in one budget group, `min`=1). The **top-1 of the
selected set is invariant to every knob** — it is always the belief argmax — so the policy
does not trade accuracy; it shapes the *set* (recall vs size). Representative rows:

| policy (min/evict, max, fix) | single T1 | single Rec | single sz | switch T1 | switch Rec |
|---|---|---|---|---|---|
| max 5, no threshold | 91.7 | 99.7 | 5.0 | 89.7 | 99.6 |
| min40 / ev20, max 5 | 91.7 | 99.7 | 3.9 | 89.7 | 99.6 |
| min25 / ev12, max 3 | 91.7 | 98.9 | 3.0 | 89.7 | 98.4 |
| **min25 / ev12, max 3, fix** | **93.5** | **100.0** | 3.0 | **92.5** | 99.4 |
| min25 / ev12, max 1, fix | 96.5 | 96.5 | 1.0 | **59.7** | 59.7 |

Findings that set the template values:

1. **β owns accuracy, the budget owns the recall/size frontier.** `max=5` reaches 99.7% recall;
   a `min40/ev20` threshold prunes the weak tail (belief leader ~600, 5th-place ~33) to ~3.9
   sections at the *same* recall — free efficiency.
2. **`fix_per_turn` earns its keep only with budget room.** At `max≥3` it *raises* both
   single-intent (91.7 → 93.5) and topic-switch (89.7 → 92.5) Tool-1 by locking the
   early-committed correct tool against late-projection noise, while the new topic still climbs
   into the remaining slots. At `max=1` it **starves** the switch (59.7): the single slot
   re-pins the stale topic on the first projection of the new turn before it can be displaced.
   **Never pin a single-slot budget.**

**Recommended template presets:**

| policy | β | min / evict | budget | fix | use |
|---|---|---|---|---|---|
| committed tool scope | 0.40 | 25 / 12 | min 1, max 3 | yes | the `tools` collection — stable, correct, small |
| high-recall scope | 0.40 | 40 / 20 | min 1, max 5 | no | when recall matters more than set size |
| single pick | 0.40 | 0 / 0 | min 1, max 1 | no | one tool, no fix (fix would starve switches) |

The **committed tool scope** is the default for the `tools` collection. The thresholds in the
table above were the pre-hybrid z-fusion values; the shipped preset now runs on the **`z × margin`
hybrid scorer** (§24.7) with a **256-token probe window**, whose scores live on a different scale.
On that scale the §80.2 sweep derives **min 1000 / evict 750 / max 3 / fix** for **100% recall**:
the true tool is always Top-3 at the 256-token width, `min 1000` sits ~0.7× the weakest true-tool
score (a wide margin below the recall floor) so live probes noticeably weaker than the corpus
worst-case still keep the truth, trimming the set to ~1 false positive (exact-1 ~50%), and
`evict 750` sits below that floor so a correct pick is never evicted.

### 24.7 The `z × margin` hybrid scorer (§80, §81)

The z-fusion self-mutes non-discriminative **groups** (a layer-group whose best match isn't an
outlier vs its own agreement distribution). It does **not** mute non-discriminative **tokens** —
and the residual errors are exactly there: a query like *"how many TCP connections do I have"* is
mostly generic *"list … sessions"* tokens shared by the whole `*_session_list` family, with the
tool identity carried by a sparse few. Under winner-take-all argmax those generic tokens split
their votes across the family and drown the identity tokens (`tcp_session_list` fell to rank #6,
beaten by `udp`/`tls`/`sql`/…).

The fix adds a second, orthogonal gate: **margin**. Per query token per group, score the best
agreement *per tool*, and weight the leader's vote by its lead over the runner-up tool
(`best − second_best`). A token where one tool sharply wins dominates; a generic token where the
family ties votes ~0. The vote becomes `z × margin`. This lifts the leave-one-out corpus from
**96.0 → 97.6 Tool-1, 99.7 → 100.0 Tool-3/Tool-5** with no family regression, closing the lone
Top-5 miss. The margin also auto-mutes the noise fold-group L0–45 (near-zero margin there), so no
group selection is needed. Implemented in `score_provenance_late_fusion`
(`candle-conversation/src/provenance/scan.rs`).

### 24.8 The needle gate (§82)

A wide probe window (256 tokens) risks a **haystack** problem: most tokens are generic/boilerplate
(shared across the family, or — for content collections — stale surrounding context), and only a
sparse few carry the identity **needle**. Summing all of them lets the diffuse majority overwhelm
the needle, and (for content) lets a diffuse tail pull in wrong sections and exhaust the window.

The first instinct — a **recency decay** that walks the window backward and aborts once confident —
was swept in §82 and **fails to generalize**: it assumes the needle is recent, but on the tool
corpus the discriminative tokens are scattered (§81), so any decay/abort costs recall (Tool-3
100 → 98–99%). Chunk-level quality weighting is neutral (too coarse — it dilutes a needle inside a
mostly-generic chunk).

The mechanism that works is a **per-token magnitude gate**: keep only the top `NEEDLE_KEEP_FRAC`
(0.25) of query tokens by vote magnitude, drop the rest. It is **position-independent** — it finds
the needle wherever it sits — which is exactly why it generalizes where recency can't. On the tool
corpus it *holds* Tool-3/Tool-5 at 100%, *sharpens* Tool-1 (96.8 → 97.8 gated at min 1000), and
*improves* the set (mean FP 0.56 → 0.37, exact-1 49 → 58%), all on ~75% fewer effective tokens.
The decisive comparison, same ~53-token budget: **top-25%-by-magnitude 97.0 / 100.0** vs
**last-64-by-recency 89.2 / 98.1** — magnitude *finds* the needle, recency *guesses* at it.

The gated scores keep essentially the same scale (the kept needles dominate the magnitude sum), so
the §80.2 thresholds (min 1000 / evict 750 / max 3) are unchanged — the gate is a strict-improvement
drop-in. Because it keys on signal quality, not domain, it is left **on for all collections**; for a
content collection (code) the same gate keeps the sharp relevant reference and mutes the boilerplate.
Validated only on tools so far — the content case awaits a labelled code-section corpus.
