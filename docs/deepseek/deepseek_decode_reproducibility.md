# DeepSeek-V4 decode reproducibility

**Status: FIXED.** Greedy decode of a fixed prompt now produces identical token
streams run to run, end to end, with the expert cache churning (10,049
evictions across the three passes):

```
[repro:deepseek] pass1v2 = None   pass1v3 = None   pass2v3 = None
[repro:deepseek] REPRODUCIBLE
```

Two independent sources, one in decode and one in prefill. Both were confirmed
by an instrument that isolates them, not inferred from token streams.

## Why it mattered

Greedy argmax decoding of a fixed prompt from a fresh session is a pure
function of the weights. It was not: every ablation, threshold derivation and
"config X is 3% better than Y" measured against this engine carried an
unquantified amount of run-to-run noise. For a codebase heading to a paper
submission that is a load-bearing property to lack.

**Why it went unnoticed:** every existing model gate asserts *validity*, not
reproducibility. StoryRewrite checks each session reproduces its own
name-substituted story; `wave_prefill_then_decode_end_to_end` checks the
continuation is the expected one. Both pass
happily on an engine that returns different text every run.

## The amplifier

**MoE top-k routing is discontinuous.** A one-ULP difference in a router score
flips which expert a token is dispatched to, and the logits then differ by
order 1. A large observed gap does *not* imply a large underlying fault — both
root causes here were sub-ULP effects. Hunt for ULPs, not for garbage.

## Source 1 — residency-dependent accumulation order (decode)

`expert_lre/pipeline.rs`. `compute_experts_grouped` was called **twice per
layer** — once for `classified.hits`, once for `classified.loaded` — both
accumulating into the same `ys`. Inside each call the scatter builds its
permutation with

```rust
perm.sort_by_key(|&i| (all_token_ids[i as usize], i));
```

where `i` is the position in **that call's** expert-major array. So a token's k
contributions were reduced in an order determined by which experts happened to
be resident — and residency is measurably variable run to run (two identical
warm passes differed ~20% in misses, ~60% in evictions). Float addition is not
associative, so the same k terms in a different grouping give a different sum.

The CPU arm had the identical defect (it accumulates one expert at a time in
`hits` then `loaded` order) and takes the same canonical order now.

**Fix:** one call per layer over `hits ++ loaded`, ordered by **expert id** — a
key that is a function of routing alone, so the accumulation order no longer
depends on residency.

**This mechanism was previously recorded as "tested and refuted".** That
experiment measured *prefill* `max|Δ|`, which source 2 below keeps dirty
regardless, so it could not have detected a decode-side fix. A refutation is
only as good as the instrument's ability to see the thing being refuted.

**Cost:** the merged call runs after the copy fence, giving up the overlap of
hit-expert compute against miss DMA. Measured: bulk throughput at or above the
previous best (cfg16 971 vs 920–929, cfg8 787 vs 770), trailing single-session
decode 19.0 t/s against a recorded 21.3. If that gap proves real rather than
run variance, the overlap-preserving form is to keep both GEMM phases writing
into disjoint row ranges of one contribution buffer and run a single canonical
scatter after the fence.

## Source 2 — non-associative band-collapse (prefill)

`paged-latent/latent_prefill_kernel.cuh`. The QK phase assigns **warp = band**,
and each warp `atomicAdd`s its band's contribution into a shared
`scores[head][key]`. With `NPAL = 16` every score was a sixteen-term float sum
accumulated across sixteen warps in an order the hardware does not define —
**non-determinism by construction**, not a race and nothing corrupted.

`latent_decode_kernel.cuh` contains zero `atomicAdd`s, which is the structural
asymmetry that made this findable.

**Fix: int32 fixed-point accumulation.** Integer addition is associative, so the
sum is order-independent. The scale is **derived in-kernel from a bound**
rather than picked as a constant:

```
|Σ_p c_p·scaleQ[h][p]·scaleK[k][p]| ≤ NKS·32·127² · Σ_p |scaleQ[h][p]|·max_k|scaleK[k][p]|
```

`|c_p| ≤ NKS·32·127²` because it is an int8 m16n8k32 accumulator, and every
other term is already in shared memory. Scaling by `2³⁰/bound` puts the largest
representable sum just inside int32, so **overflow is impossible by
construction** — not contingent on the activation statistics of the day, which
is what made the originally-designed constant-scale version unsafe to land. It
costs one thread per `(head, band)` — exactly the 512-thread block — plus one
barrier.

All 24 kernel gates stayed green, **including the bit-exact mirror oracles**, so
the re-baselining the earlier design feared was not needed.

### Two traps inside this fix

* **Placement.** The bound reads `scaleK` and `key_valid`, which *other warps*
  write during staging. Computed before the staging barrier it races them and
  reintroduces exactly the non-determinism it exists to remove. It must run
  after both staging barriers.
* **Uninitialised shared memory (hardening, not a fix).** The per-dim V scale
  scanned all 32 key slots including unstaged ones, whose `sK`/`scaleK` hold
  whatever the previous launch left. That is wrong — the residue is only stable
  while the surrounding schedule is — but it was **ablated and is not required
  for bitwise repeatability**. It is masked now as hardening; do not credit it
  with the fix.

## Also fixed: silently dropped experts

Both call sites built their expert list with
`filter_map(… slots[slot_idx].as_ref()?)`, which **discards** an expert whose
slot is empty at compute time. Its contribution simply vanishes and the layer
returns an answer computed from fewer than k experts, indistinguishable
downstream from a correct one. Now a hard error at both sites
(`pipeline.rs`, `handle.rs`).

Measured: it never fires on these workloads, so it was **not** a cause of the
non-determinism. It is a latent correctness hazard, closed.

## The gates

| Test | Where | Asks |
|---|---|---|
| `prefill_kernel_is_bitwise_repeatable` | `latent_moe/paged.rs` | the prefill kernel returns the same bits twice — **0.16 s, no model** |
| `prefill_replay_probe` | `batch_test/utils.rs` | re-prefilling a prompt is bitwise stable |
| `decode_replay_probe` | `batch_test/utils.rs` | a decode step replayed from identical KV state is bitwise stable |
| `wave_decode_step_bitwise_probe` | `latent_moe/wave.rs` | both of the above, on the real model |
| `wave_decode_is_reproducible` | `latent_moe/wave.rs` | end-to-end token streams |
| `qwen3_moe_decode_replay_is_bitwise_repeatable` | `quantized_qwen3_moe.rs` | **the harness control** (see below) |

Prefill and decode are gated **separately and on purpose**: end-to-end token
streams cannot attribute a divergence to one or the other, so a fix to either
gets judged by a test the other also fails. That is precisely why the decode fix
looked ineffective until the two were split.

## Method notes — three instruments that lied

Worth more than the fixes themselves.

**1. Whole token streams.** Argmax hides sub-ULP differences until one crosses a
routing boundary, and each verdict costs a 50 s three-pass run whose result is
intermittent. Every framing built on this was wrong.

**2. The depth sweep — a false-positive generator.** Sweeping `layer_end` to
name the layer that introduces a fault looks compelling and is invalid: a
partial-depth `forward_wave` returns a *paused wave's* residual, meant to be
resumed on a later forward, and replaying that path does not restore identical
state. Run against Qwen3-MoE — whose full-depth replays here are bit-identical —
it reported **every layer** as dirty. It had already produced two confident,
wrong attributions ("first dirty depth = 16", "= 3") before the control caught
it. Deleted, along with the layer-isolation probe built on the same mechanism.

**Any per-layer instrument must clear that control before its output means
anything.**

**3. The `PINNED_LAYERS` coincidence.** An earlier depth bisect found decode
clean through layers 0–2 and first failing at layer 3 — exactly the boundary
past which experts become evictable. That was read as a smoking gun for the
expert load/evict path. It was a coincidence of where the fault became visible.
The expert cache is exonerated by direct measurement: decode replay is
bit-identical through 5278 misses and 5949 evictions.

**4. A cross-model control that controlled less than it looked like.** The
"Qwen3-MoE reproduces and shares DeepSeek's `expert_lre` cache, so the cache is
exonerated" argument was **wrong**: Qwen3-30B fits in VRAM, `all_resident`
short-circuits the pipeline, and its counters read `hits=0 misses=0
evictions=0`. It never touches that code. The exoneration is sound, but it rests
entirely on the DeepSeek replay measurement above, not on Qwen3. This is why the
probe now prints the expert counters — a green verdict from a run that never
exercised the path is not evidence about that path.

## Known-open, unrelated: Qwen3-MoE intermittent non-reproducibility

`qwen3_moe_decode_is_reproducible` fails roughly 1 run in 7 (observed once in
seven; six clean). Not caused by anything here: Qwen3's expert cache is inert
(`hits=0`), the prefill kernel change is DeepSeek-only, and it passed three
consecutive runs immediately after these edits.

Localised as far as: its prefill replay is clean (0/5) and its decode replay is
clean (0/7), yet the token-stream test — which interleaves 32 decode steps
between prefills — diverges. So the suspect is state that survives
`truncate_sequence_to_tokens` **after decoding** rather than arithmetic
non-determinism (Qwen3 compacts a rope cache during decode, for instance). Its
own investigation; DeepSeek passes the same gate.

**What worked, every time:** the *differential* check — same code, one variable
changed. Kernel against itself (found source 2 in 0.16 s). Model against a
second model (killed the expert-cache lead, and caught the bad instrument).
Prefill isolated from decode (made the decode fix visible). Build the localising
check first; theorise after.
