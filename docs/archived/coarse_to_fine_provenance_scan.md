# Lossless Coarse-to-Fine Provenance Scan — Archived (Negative Result)

## Status

**Archived. Investigated, measured, and abandoned.** The flat single-stage
`BdpScanner` (committed `fb03ca05`: AVX-512 `VPOPCNTDQ` kernel + SIMD
aggregation + per-item rayon parallelism, **86 → 7 ms** on a ~345 K-token turn
corpus) is the right architecture for exact provenance retrieval on this data.
No lossless — and no *usefully* lossy — sublinear method beat it.

The design below (single-ball bound, two-stage branch-and-bound, formula
capability/fallback) is preserved as the record of what was explored. The
sections from *Motivation* onward describe that design in the present tense as
originally written; read them as the hypothesis, not the conclusion. **The
conclusion is this section.**

## What we learned (why this was abandoned)

The whole approach rests on one empirical assumption — stated honestly in *Risks
and the gating experiment* below — that the per-chunk medoid radius `R` is small,
i.e. that the 128-bit BDP signatures within a chunk (and across the corpus)
**cluster**. They do not. BDP signatures are `sign(Q_PCA^T · K)`, and that sign
projection is **designed to decorrelate** — it spreads mass roughly uniformly
over the hypercube. We confirmed the diffuseness three independent ways on a real
sealed substrate:

1. **Chunk medoid radius `R`: p50 = 65** (of 128 bits). A random pair of
   128-bit vectors agrees on ~64 bits in expectation; a median radius of 65 means
   the signatures inside a single 32-token chunk are **indistinguishable from
   random**. The ball bound `UB_chunk(p) = agreement(p, r) + R` with `R ≈ 65`
   adds the whole dimension back — it prunes **nothing**. Gating the bound on
   real data dropped zero turns.
2. **Probe OR-compression: 1.00×.** ORing similar probe signatures together
   (the two-sided OR/AND clustering idea) found no similar signatures to merge —
   measured compression ratio `1.00×` at threshold τ = 24.
3. **Corpus OR-compression: 1.02×.** Same experiment on the corpus side —
   `1.02×`. There is essentially no clusterable structure to exploit on either
   side.

The hit test compounds the problem. `DEFAULT_HIT_THRESHOLD = 90` ⇒ hit radius
Hamming **38** of 128 — a *large* ball. The set of tokens within radius 38 of any
probe is dense, so even a tight bound (which we don't have) would admit most of
the corpus as "potential hits." Diffuse data + large radius is the textbook
curse-of-dimensionality regime where **no exact sublinear NN method exists**:
the single-ball bound, OR/AND two-sided masks, VP-trees, BSP, and MIH all prune
nothing here, for the same root reason.

### Approximate (HNSW) — also not worth it

Dropping the losslessness requirement, we built and tuned an HNSW index over the
corpus signatures (diversity heuristic / Algorithm 4, `ef` swept 256 → 8192,
parallel build, pre-indexed at daemon load so the cost is off the reproject
path). Evaluated by the *correct* metric — **does the projection select the same
substrate window?**, not bit-identical scores:

- It recovers the **identical substrate window ~2/3 of reprojects**, and is a
  **superset (never-missing)** the rest of the time — it never drops a turn the
  flat scan selected.
- But to reach even that, `ef` has to be large enough that HNSW is **slower than
  the warm 7 ms flat scan at 345 K corpus**. It only wins asymptotically (the
  10 M-token target), and even there it is *not bit-perfect* — it cannot
  guarantee identical selection, which was the hard requirement.

So HNSW buys nothing at today's scale and trades correctness for speed at the
target scale. "Superset, never-missing" is noted as a *possible* future
relaxation if the selection contract is ever loosened to allow extra turns — but
that is a different product decision, not this optimization.

### Bottom line

On diffuse BDP signatures with a Hamming-38 hit test, the expensive linear term
in the flat scan is **irreducible** without changing the signatures themselves
or relaxing the exact-selection contract. The flat SIMD + parallel scan
(`fb03ca05`) stays. If 10 M tokens ever makes the ~300 ms flat scan a real stall,
the productive levers are upstream of this doc: shrink the work (fewer
depths / probe tokens), change the signature so it *does* cluster (a different
provenance projection), or accept superset selection and use HNSW — not an exact
coarse-to-fine bound, which the data rules out.

---

*Original design follows, unchanged, for the record.*

## Motivation

The flat scan is `O(corpus_tokens × probe_tokens × depths)` — every reproject
compares the probe against **every token signature in the entire turn history**.
This is linear in corpus size:

| Turn corpus | Flat scan (parallel + AVX-512) | Signature bytes streamed/scan |
|------------:|-------------------------------:|------------------------------:|
| 230 K (today) | 7 ms | ~13 MB |
| 2.3 M | ~70 ms | ~130 MB |
| **10 M (target)** | **~300 ms** | **~480 MB** |
| 100 M | ~3 s | ~4.8 GB |

At the **10 M-token target**, a flat scan is a **~300 ms decode stall on every
reproject** — roughly 7 tokens of dead time at the current ~40 ms/token decode
rate, and 30–60× over the design's 3–10 ms scan budget. The flat scan does not
survive the jump to 10 M.

Coarse-to-fine replaces the single flat pass with a cheap **prune** pass over
per-chunk summaries followed by an exact **fine** pass over only the survivors.
Modelled at 10 M: **~300 ms → ~12 ms (~25×)**, and ~25× less memory traffic.

## Hard requirement: losslessness

The optimization **must not change the functional output**. "Functional output"
is defined precisely as:

> the ordered list of projected turn / section IDs
> (`Vec<(LayerId, GroupId, TurnIndex)>` and the section equivalents) that the
> projection emits into the KV cache.

Pruning may skip computing the intermediate `TurnScores` for turns that are
provably never selected — but the **selected set, its order, and the scores of
the selected items must be bit-identical** to the flat scan. This rules out the
approximate "top-K nearest chunks" pruning used in ANN search; we use
**admissible branch-and-bound** instead (the A\* discipline: only prune
candidates that *provably* cannot win).

## What the selection actually depends on (and why pruning is feasible)

Traced through the projection (`projection/project.rs`, `score.rs`,
`selection.rs`, `substrate.rs`):

1. The BDP scanner produces, per turn and per section, a `PerDepthScores` —
   three depths (`syn`/`sem`/`prag`), each a `TurnScores` carrying seven
   statistics (`max`, `sum`, `mean`, `top_k_mean`, `count`, `span`,
   `pertok_excess`). [`substrate.rs:463-485`]

2. Selection uses a **single** statistic, fixed at
   `FIXED_FORMULA = ScoreFormula::Span { alpha: 2.0 }`
   [`project.rs:112`]. The other six statistics are computed but **unused** by
   the current projection.

3. Per turn: the chosen statistic is read from each depth and combined with
   `DepthWeights` (default `syn:1 / sem:1 / prag:4`, normalised) into one scalar:
   `combine_per_depth(syn.span, sem.span, prag.span, weights)`
   [`substrate.rs:3287-3293`, `schema.rs:269-281`].

4. Per group: turns are gated by `score_threshold`, ranked, and selected by a
   `SelectionRule` — `TopK { k }`, `Single`, `AlwaysVisible`, or
   `Sequence { recent, historical_top_k }` — then trimmed to a token `budget`
   [`selection.rs:84-239`]. The group score used for layer gating is the
   **`MAX`** of the selected turns' scores [`score.rs:29-61`].

### Why `span` admits an upper bound

`span` for one (turn, depth) is computed from `probe_hits`: a boolean per probe
position, `true` iff **any** corpus token in the turn reached agreement
≥ `hit_threshold` (default 90) with that probe token. Then

```
span = Σ over maximal runs of consecutive hit positions of (run_length)^α   (α ≥ 1)
```

**Monotonicity (key lemma).** `span` is monotonic non-decreasing in the hit set:
adding any probe position to the hit set never decreases `span` — it starts a
new run (`+1`), extends a run (`(L+1)^α − L^α > 0`), or bridges two runs
(`(L₁+L₂+1)^α − L₁^α − L₂^α > 0` for α ≥ 1). Therefore **an upper bound on the
hit set yields an upper bound on `span`.**

The depth combination is a non-negative weighted sum, and the group score is a
`MAX` — both preserve "upper bound in ⇒ upper bound out." So an admissible
upper bound on each turn's `span` gives an admissible upper bound on its
selection score, which is exactly what branch-and-bound needs.

> **Scope note.** This holds for max-like formulas. Additive formulas fall back
> to the exact flat scan. The dispatch is explicit and fail-safe — see
> *Formula capability and fallback*. The default and current production formula
> is `Span`, so coarse-to-fine is on the live path today, but **nothing in the
> design assumes the formula is fixed.**

## The admissible bound

We need, per chunk and per depth, a cheap upper bound on the best agreement any
token in that chunk can reach with a probe token `p`:

```
UB_chunk(p)  ≥  max_{t ∈ chunk} agreement(p, t)            (admissible)
```

### Ball bound (triangle inequality)

Hamming distance is a metric, so for any reference signature `r`:

```
hamming(p, t) ≥ hamming(p, r) − hamming(r, t)
```

Store per chunk a **center** `r` and a **radius** `R = max_{t ∈ chunk} hamming(r, t)`.
Then for every token `t` in the chunk:

```
hamming(p, t) ≥ hamming(p, r) − R
⇒ agreement(p, t) = 128 − hamming(p, t) ≤ (128 − hamming(p, r)) + R = agreement(p, r) + R
```

So `UB_chunk(p) = min(128, agreement(p, r) + R)`. Computing it costs **one**
signature comparison per (probe position, chunk) — a `32×` reduction versus the
per-token scan, and it reuses the existing AVX-512 agreement kernel verbatim
(the "corpus" is now the array of chunk centers).

### From chunk bound to turn span bound

For a turn spanning chunks `C`:

```
UB_hit[p]    = ( max_{c ∈ C} UB_chunk_c(p) ) ≥ hit_threshold       # position p *might* hit
UB_span(turn) = span( UB_hit )                                      # span of the upper-bound hit set
```

By the monotonicity lemma, `UB_span ≥ true span`. Combine across depths with the
same weights ⇒ `UB_score(turn) ≥ true selection score(turn)`. Admissible.

### Index structure and cost

Per chunk, per depth: `center` (16 B) + `radius` (1 B) = 17 B; three depths =
**51 B/chunk**. At 10 M tokens (`CHUNK_SIZE = 32` ⇒ ~312 K chunks): **~16 MB**
resident index. Built **once at seal time** (signatures are immutable), so it is
reused across every reproject; only the probe changes.

The index is **formula-agnostic** — `(center, radius)` summarizes the signatures,
not any scoring choice — so it is built unconditionally and stays valid no matter
how the schema's formula changes. Whether a given reproject *uses* it for lossless
pruning is decided by formula capability (next section); the index is also usable
for tier-prefetch regardless of formula, since prefetch is a heuristic, not a
correctness-bearing decision.

- **Center choice.** The medoid (token minimising max in-chunk distance)
  minimises `R` and gives the tightest bound; it is `O(32²)` per chunk at seal
  time (one-time, cheap). A per-bit-majority centroid is `O(32)` and a
  reasonable cheaper alternative. Tightness is decided by measurement (below).
- **Persistence.** The index is derivable from the signatures in the provenance
  file; it is persisted as a sidecar (rebuildable on load), consistent with the
  "persistence is mandatory" rule for the substrate.

## The two-stage scan

```
Stage 0 (once, at seal time):  build per-chunk (center, radius) per depth.

Dispatch (every reproject):  if any active formula lacks a coarse bound,
                             run the flat scan instead (exact) and stop here.

Stage 1 — COARSE (every reproject, only when all formulas are bound-capable):
    For each chunk c, each depth d:
        UB_chunk_{c,d}(p) = min(128, agreement(p, center_{c,d}) + R_{c,d})   ∀ probe p
    Aggregate per turn:  UB_hit, UB_span, UB_score(turn)        # admissible upper bound
    # reuses the AVX-512 kernel over the ~32× smaller center array, rayon per-turn

Stage 2 — FINE (branch-and-bound, exact):
    Always fine-scan turns the schema forces in (Sequence `recent`).
    Drop turns with UB_score < group score_threshold        # provably gated out
    Sort remaining turns by UB_score descending.
    maintain running exact top-K (min-heap of exact scores).
    for turn in UB-descending order:
        if UB_score(turn) < kth_exact_score: break           # all remaining provably lose
        compute EXACT TurnScores(turn)   # full per-token scan of this turn only
        update running top-K
    Run the unchanged selection/budget logic on the exact scores.
```

**Why it is lossless.** A turn is skipped only when (a) its admissible
`UB_score` is below the group threshold (so its exact score is too — it fails
the gate), or (b) its `UB_score` is below the `k`-th *exact* score already found
(so its exact score `≤ UB_score < kth_exact` — it cannot enter the top-K), and,
because turns are visited in `UB_score`-descending order, every later turn has an
even smaller bound. Forced turns (`recent`) are always scored exactly. Budget
trimming operates on the exact selected set. Therefore the selected set, order,
and selected-item scores are identical to the flat scan. ∎

The fine pass touches only turns whose bound clears the running cutoff — a count
governed by `k` and the bound tightness, not by corpus size. The expensive term
stops growing with history.

## Bonus: tier-prefetch hook

At 10 M tokens the KV lives across the three-tier cache (GPU → RAM → NVMe; not
yet built — see `docs/kv_tier_migration.md`). The coarse pass already computes,
cheaply, the set of turns that *could* be relevant (those whose `UB_score`
clears the threshold). That candidate set is exactly what the tiering layer must
prefetch from warm/cold while the GPU decodes. Coarse-to-fine is therefore not
only a scan speedup but the **retrieval primitive the tiering depends on**.

## Correctness and testing

The flat `BdpScanner` becomes the **reference oracle** (same discipline as the
scalar↔SIMD cross-check that guards the current kernel):

1. **Identical-selection property test.** Over many randomized substrates
   (varying corpus size, turn sizes, chunk-radius distributions, probes,
   thresholds, and selection rules), assert the coarse-to-fine projection emits
   a **bit-identical** `Vec<(LayerId, GroupId, TurnIndex)>` to the flat-scan
   projection. This is the losslessness gate.
2. **Bound-admissibility test.** For every chunk, assert
   `UB_chunk(p) ≥ max_t agreement(p, t)` directly (no false negatives) across
   random probes.
3. **Branch-and-bound-stop test.** Assert the running cutoff never prunes a turn
   whose exact score would have entered the top-K.
4. **Index round-trip test.** Seal-time index rebuilt from persisted signatures
   matches the in-memory index byte-for-byte.

## Formula capability and fallback

The decode/reproject path currently scores with a single hardcoded constant,
`FIXED_FORMULA = ScoreFormula::Span { alpha: 2.0 }` ([`project.rs:112`]) — it is
*not* read from the schema today. But the `ScoreFormula` machinery
(`pick` / `aggregate`) is fully general, and that constant can be changed — or
made schema-driven — at any time. The design must therefore **never assume the
formula is fixed**, and a future formula must be unable to silently break
losslessness. The discipline:

> Coarse pruning is **opt-in per formula** via an explicit admissible-bound
> function. A formula that does not provide one **defaults to the exact flat
> scan**. Therefore adding a new formula can only ever make the scan *slow*
> (it falls back), **never wrong.**

This is encoded as a capability on the formula itself:

```rust
impl ScoreFormula {
    /// Admissible per-turn upper bound on this formula's score, computed from
    /// the formula-agnostic coarse stats and the probe — or `None` when this
    /// formula has no coarse bound, which forces the exact flat scan.
    ///
    /// `None` is the safe default: a formula added without an implementation
    /// here falls back to flat, preserving losslessness by construction.
    fn coarse_upper_bound(
        &self,
        coarse: &CoarseTurnStats,   // UB_hit / UB_best derived from (center, radius)
        weights: &DepthWeights,
    ) -> Option<f32>;
}
```

Capability of the existing formulas:

| Formula | Aggregation | Coarse bound | Path |
|---|---|---|---|
| `Span { α }` | max-like (monotone in hit set) | `span(UB_hit)` — admissible, tight | **coarse-to-fine** |
| `PerTokenExcess` | max-like (per-probe-token best) | `Σ max(0, UB_best_p − 64)` — admissible | **coarse-to-fine** |
| `Max` | max-like | `UB_max` — admissible, tight | **coarse-to-fine** |
| `TopKMean { k }` | max-like | `≤ UB_max` — admissible but loose | **coarse-to-fine** (looser prune) |
| `Sum` | additive over all pairs | only `≤ 32·Σ UB_chunk` — uselessly loose | **flat fallback** |
| `Mean` | additive ratio | needs UB(sum) ∧ LB(count) — messy | **flat fallback** |
| `Count` | additive threshold count | valid but very loose (assumes all 32 hit) | **flat fallback** |

### Dispatch (fail-safe)

At the start of each reproject the scan inspects the **active formula(s)** in the
projection schema:

```
if every formula required by the active schema returns Some(bound):
        run coarse-to-fine (Stage 1 + Stage 2)
else:
        run the flat scan (exact, unchanged)
# the coarse index is built either way; only its *use* is gated
```

Because formulas can in principle differ per group/layer, the gate is over **all**
formulas the schema actually uses; any single unsupported formula drops that
reproject (or, as a refinement, just the affected groups) to the flat path. The
flat `BdpScanner` is retained permanently as both the fallback and the
correctness oracle — it is never deleted.

This keeps two invariants:

1. **Losslessness is unconditional.** Unsupported or unimplemented formula ⇒ flat
   scan ⇒ identical output, by definition.
2. **No silent regressions.** A formula change that loses the fast path is a
   visible latency change (and shows up in `scan_ms`), never a wrong result.

An additive bound (per-chunk partial-sum bounds for `Sum`/`Mean`/`Count`) is a
possible future refinement, but stays out of scope until an additive formula is
actually used in production — until then those formulas simply take the flat
path.

## Risks and the gating experiment

The bound's value is **entirely governed by the chunk radius `R`**:
`UB_chunk(p) = agreement(p, r) + R`. A probe position is a *potential* hit when
`agreement(p, r) ≥ hit_threshold − R`. With random 128-bit signatures, agreement
is ~Binomial(128, ½) (mean 64, σ ≈ 5.7), so a large `R` floods the upper-bound
hit set with false potentials, inflates `UB_span`, and prunes little.

**This is the make-or-break quantity, and it is empirical.** Plausibly `R` is
small — 32 *consecutive* tokens within one turn are contextually related, so
their signatures should cluster — but that must be measured, not assumed.

**Gating experiment (do this first, before any implementation):** instrument the
existing sealed substrate to compute, per chunk per depth, the medoid radius `R`,
and report the distribution. Decision rule:

- **Median `R` small** (tight clusters): the single-ball bound is viable; build
  as specified.
- **Median `R` large** (diffuse chunks): tighten before committing — options are
  sub-chunk centers (e.g. one ball per 8 tokens → 4× index, smaller `R`),
  *k*-medoids per chunk (2–4 centers), or adding the AND/OR-mask bound
  (`UB = popcount(p & AND) + popcount(¬p & ¬OR) + free_bits`, 32 B/chunk) and
  taking the tighter of the two. All remain admissible.

No coarse-to-fine code lands until the radius distribution says the bound will
actually prune.

## Scope

- **One level is sufficient for the 10 M-token target** (~9 ms coarse + ~1–2 ms
  fine ≈ ~12 ms). The coarse pass is still `O(N/32)` — linear with a 32× smaller
  slope, not sub-linear.
- The next wall is ~50–100 M tokens, where the coarse pass itself
  (~50–100 ms) needs a **second level** (chunks-of-chunks, `N/1024`) or a metric
  tree (VP-tree / ball-tree over chunk centers) for genuinely sub-linear
  traversal. Out of scope here; the single-level index is forward-compatible
  with adding a coarse-over-coarse layer above it.

## Implementation outline

1. **Gating experiment** — measure chunk-radius distribution on a real
   sealed substrate; decide bound form. *(blocking)*
2. **Seal-time index** — compute `(center, radius)` per chunk per depth as
   chunks seal; persist as a sidecar; rebuild-on-load + round-trip test.
3. **Coarse pass** — `UB_chunk` scan over centers (reuse the AVX-512 kernel),
   per-turn `UB_hit` / `UB_span` aggregation; admissibility test.
4. **Formula capability + fail-safe dispatch** — add
   `ScoreFormula::coarse_upper_bound`; inspect the active schema's formulas and
   route bound-capable formulas to coarse-to-fine and everything else to the
   retained flat scan. `None`-by-default guarantees a new formula falls back.
5. **Branch-and-bound fine pass** — UB-descending visitation, running exact
   top-K cutoff, forced `recent` turns; wire into `reproject_view_prepare`
   behind the flat scan as oracle.
6. **Losslessness gate** — randomized identical-selection property test must be
   green before the coarse path replaces the flat path on the live route;
   include a case that exercises an unsupported formula and asserts it takes the
   flat path with identical output.
7. **Tier-prefetch hook** — expose the coarse candidate set to the (future)
   tiering layer (usable regardless of formula).
