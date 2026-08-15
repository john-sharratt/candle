# DeepSeek-V4-Flash — DSpark Speculative Decode

**Status:** in progress. Branch `speculative-decode` (forked from `deepseek-flash`).
**Goal:** draft a block of up to 5 tokens per decode cycle with the DSpark `dflash` drafter —
**as far as the drafter's own confidence justifies** (0 drafts ⇒ a normal decode) — verify
them in one main-model forward, and commit the prefix that converges. Turns our launch-bound
decode floor into ~2.4+ tokens per forward with no change to the output distribution.

**The entire drafter is implemented and GPU-validated on the real 10.9 GB weights**
(`deepseek4/dspark.rs`, `deepseek4/attention.rs`, `deepseek4/transformer.rs`,
`zend/src/download.rs`): automatic drafter+model download (`zend --download-deepseek`); the
`dflash` config; the complete loader (every tensor); the context encoder `encode_context`
(Eq. 2); the bit-exact Markov bias + confidence math (Eqs. 5, 7); the **backbone**
(`Attention::forward_injected` — non-causal latent MLA with `Hctx` injected via `wkv`;
`Block::forward_injected`; `DsparkDrafter::backbone` — mHC expand → 3 injected blocks →
`output_hc` head-reduce → `output_norm`); the **semi-autoregressive sampler** `sample_block`
(Eq. 4 + Alg. 1); and `DsparkDrafter::draft` tying it all together. 10 tests (6 CPU bit-exact,
4 download) + an on-GPU real-weight test that loads the drafter and runs the 3 injected blocks
→ finite hidden `[5,4096]`. **Remaining = engine plumbing only** (§4.3-4.5): capture the
target's hidden states at layers [41,42,43] to feed `draft()`, hand it the target's shared
`token_embd`+`output.weight`, and run the verify/accept/rollback loop against the running 284 B
target — that's where real acceptance (and the residency go/no-go, §7) is measured. The drafter
is DSpark verified against the real GGUF — **not** the V3-MTP `eh_proj` recurrence an earlier
draft of this doc assumed.

---

## 1. Why this pays off *here* specifically

Our single-session decode is **not compute-bound** — it is bound by the WDDM per-launch
overhead (`~74 ms/token` floor; see `docs`/memory `wddm-forward-floor`). Every token pays
one full forward's worth of kernel-launch + submission latency regardless of how little
compute the forward actually does.

Speculative decode attacks exactly that floor. One main-model forward (the *verify* pass)
covers `k+1` token positions at once. If the drafter's tokens are accepted at the measured
rate, that single launch-heavy forward yields **~2.4 committed tokens** instead of 1:

| draft tokens k | per-token accept α (DSpark, measured on this model) | expected committed / cycle = 1+α+α²+…+αᵏ |
|---|---|---|
| 3 | 0.66 | **2.39** |
| 4 | 0.59 | 2.48 |
| 5 | 0.51 | 2.37 |

k=3 is the sweet spot the vLLM recipe ships (`{"method":"mtp","num_speculative_tokens":3}`).
The table is the **static** yield; §4.2 makes the draft length **adaptive** on the drafter's
confidence, which is measured to add ~1.4× on top of static speculation (vLLM DSL) by not
wasting draft launches on tokens that will be rejected. Effective decode floor drops from
~74 ms/token to **~74/2.39 ≈ 31 ms/token** *before* the adaptive gain — and the draft is one
transformer block, not 43.

**The catch, stated up front:** the drafter is a full decoder block with MoE routing, and
routing readback is the WDDM wall we already fought (memory `deepseek-moe-wddm-readback-wall`,
`gpu-native-moe-dispatch`). Three sequential recurrent draft steps add three routing
readbacks per cycle on top of the verify forward's 43 — but the confidence-gated draft length
(§4.2) only launches those steps while the drafter is sure, cutting the readbacks it doesn't
earn. The design keeps the drafter on the resident GPU-native MoE dispatch (`moe_bucketize` +
`grouped_qmatmul_dev_q8a128`) so those readbacks are the on-device histogram path, not a host
counting-sort. §6 makes measuring this the first gate — if draft launch overhead still eats
the win, we raise `τ` / lower `k_max` or drop to the n-gram drafter (§7), but the arithmetic
above says we have comfortable margin.

Output is **bit-for-bit the base model's greedy output** (verify uses the true model; drafts
only propose). This is a pure throughput change, not a quality trade — the same property
DSpark advertises ("served output distribution is identical … they affect throughput only").

---

## 2. What the drafter is — DSpark (`dflash`), verified against the real GGUF

The drafter DeepSeek ships is **not** a V3-style MTP `eh_proj` recurrence. It is **DSpark**
("Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation", Cheng et
al. 2026, arXiv:2607.05147) — a small standalone `dflash` model in a **separate** GGUF
(`dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf`), confirmed by reading the file's tensor
directory + `dflash.*` metadata directly:

- **`dflash.block_count` = 3 backbone blocks**, each *structurally identical to a main
  DeepSeek-V4 block* (MLA single-latent + 256-expert MoE + hyper-connections; same
  `blk.N.attn_*` / `ffn_*` / `hc_*` tensor set) — so they load through the existing
  `load_block`.
- **Shares the target's frozen `token_embd` + output head (`output.weight`)** — neither is
  present in the drafter file; only `output_norm.weight` is local.
- **Draft-specific tensors:** `markov_w1` / `markov_w2` (the low-rank Markov head) and an
  optional confidence projection.
- **`dflash.block_size` = 5** — the trained max draft-block length (`--spec-draft-n-max`
  clamps to it); `dflash.target_layers` — the target layers whose hidden states are injected.

**How it drafts (paper equations):**

- **Semi-autoregressive block (Eq. 4).** The 3-block backbone emits `γ ≤ block_size` base
  logits `U₁…U_γ` for the whole block in **one** forward pass, then each position is sampled
  left-to-right with a transition bias:
  `pₖ(v) = softmax( Uₖ(v) + Bₖ(x_{<k}, v) )`.
- **Low-rank Markov head (Eq. 5).** `B(x_{k-1}) = W₁[x_{k-1}] · W₂`, with `W₁ ∈ ℝ^{V×r}`
  (a lookup by the previously sampled token), `W₂ ∈ ℝ^{r×V}`, rank `r = 256`. This is the
  left-to-right signal a pure block-parallel draft loses.
- **Target conditioning / KV injection (Eq. 2-3).**
  `Hctx = RMSNorm(Wc · [H^{l₁};…;H^{lₘ}])` from the target's `dflash.target_layers`, then the
  draft's attention runs over injected keys/values `K = [Wᵏ·Hctx ; Wᵏ·Hd]`,
  `V = [Wᵛ·Hctx ; Wᵛ·Hd]` — **bidirectional** over (target context features ‖ draft tokens).
  This is what lifts a 3-block draft to the measured acceptance; it is also the piece most
  coupled to our attention stack (§4).
- **Confidence head (Eq. 7).** `cₖ = σ(wᵀ[hₖ ; W₁[x_{k-1}]])` — a calibrated survival
  probability per block position; the scheduler picks the draft length as the longest prefix
  whose **cumulative product `∏ cᵢ`** clears a throughput-optimal threshold (§4.2). Confidence
  scheduling *is* DSpark's native mechanism — the adaptive-length feature is built in, not
  bolted on.

**Measured (DSpark card / paper):** ~66% @ 3, 59% @ 4, 51% @ 5 draft tokens; 1.84–1.91×
end-to-end on B200 at n=3 — **with the target fully GPU-resident** (see the residency caveat
in §7; our streaming-expert config is the at-risk case).

---

## 3. Where it plugs into our code

There are three decode implementations; only the batched wave is production and only it
matters here (the reference `IncrementalTransformer::step` and single-seq engine paths stay
as correctness oracles):

- **Production:** `DeepSeekBatched::forward_wave` (`wave.rs:470`) — one call = one wave =
  one forward step for every session in the batch.
- Per-wave the head block already gathers scored rows and runs **one batched `lm_head` GEMM**
  over `[1, R, dim]` (`wave.rs:1494-1516`), returning per-row logits. Sampling (argmax) is
  the caller's host responsibility (`wave.rs:1695,1726,1830`) — there is no sampler in the
  module.
- **The verify shape already exists.** `paged_latent_prefill_raw` (`paged.rs:548`,
  driven at `wave.rs:1344-1363`) feeds K new tokens for one sequence through **one** kernel
  launch: packed `kv_new`, a `seq_of` query→slot map, per-token `q_pos`, and per-seq diagonal
  metadata `new_meta[n_seq,4] = {rows, base, start, -}`. Each query attends its arena slot +
  its own new-token diagonal slice with causal masking, and is **argmax-equal to per-token
  decode** (guaranteed by the bf16 diagonal staging, `wave.rs:650-664`). This is exactly a
  speculative verify of K proposed tokens — we reach it today only through `prefill_seqs`.

So verify is "run the drafts as a K-token prefill row-group". Drafting is the genuinely new
compute; rollback plumbing is the genuinely new bookkeeping.

---

## 4. Design

### 4.1 The drafter: DSpark `dflash` backbone + Markov head (see §2 for the math)

Module `deepseek4/dspark.rs` (**implemented** — config, loader, and the pure Markov/confidence
math; §5). It holds:

- `DsparkDrafter` — the 3 `dflash` backbone `Block`s (loaded via the existing `load_block`),
  the `MarkovHead` (`markov_w1/w2`), the `output_norm`, and the optional `ConfidenceHead`.
  It borrows the shared `token_embd` + output head from the target (no copy).
- `MarkovHead::bias(prev) = W₁[prev] · W₂` and `ConfidenceHead::confidence(h, W₁[prev])`
  are the paper's Eq. 5 / Eq. 7, unit-tested bit-exact.

**Drafting a block** (`DsparkDrafter::draft` — the GPU-coupled remainder, §4.5): unlike an
EAGLE-style recurrence, DSpark is **semi-autoregressive** —

```
Hctx = rmsnorm( Wc · [ target_hidden[l] for l in target_layers ] )   # Eq. 2-3 (from verify pass)
U[1..γ] = dflash_backbone( draft_tokens, kv = [Hctx ‖ draft_kv] )    # γ = block_size base logits, ONE pass
t = committed_token
for k in 1..=γ:                                                       # sequential sampling, bias-only
    logits_k = U[k] + markov.bias(t)                                  # Eq. 5 low-rank Markov bias
    t = argmax(logits_k)                                              # d_k
    c_k = confidence(h_k, markov.embed(t))                            # Eq. 7 survival prob
    if cumulative_product(c_1..c_k) < TAU: break                     # confidence-scheduled length (§4.2)
    drafts.push(t)
```

The **backbone runs once per block** (all γ base logits in parallel), then the γ positions are
a cheap sequential bias+argmax — so the WDDM-costly part (the 3-block forward with MoE routing
readbacks) is paid **once per cycle, not once per token**. This is structurally better on our
launch-bound box than a per-token recurrence. Keep the backbone on GPU-native resident MoE
dispatch (`forward_moe_gpu`) so its routing readback is the on-device histogram.

**Target conditioning (the hard part).** `Hctx` needs the target's hidden states at
`dflash.target_layers`, captured during the verify/prefill forward, and injected into the
draft attention's K/V (bidirectional over `[Hctx ‖ draft_kv]`). Our attention is the bespoke
causal paged-latent kernel, so this injection is a new draft-attention path — the main GPU
integration cost, and the reason acceptance can't be validated without a GPU run (§7).

**Seeding.** The first block seeds from the prompt's last prefill hidden; later blocks seed
from the verify pass (§4.3), which already produces the residual hidden per position.

### 4.2 Adaptive draft length (confidence-gated)

The drafter decides *how far to speculate* per cycle from its own confidence, instead of
always drafting a fixed `k`. This is vLLM's Dynamic Speculation Length with confidence-threshold
early exit, and it directly matches the requirement "speculate up to 4; if it's not confident,
draft fewer — at zero, do a normal decode."

- **Signal:** DSpark ships a **trained confidence head** (§2 Eq. 7),
  `c_k = σ(wᵀ[h_k ; W₁[x_{k-1}]])`, supervised to predict `1 − ½‖p_draft − p_target‖₁` — a
  directly calibrated acceptance estimate, stronger than the generic max-softmax proxy. (If a
  quantised checkpoint omits the head, fall back to `c_k = maxᵥ softmax(U_k + markov.bias)`,
  the well-calibrated EAGLE/DSL proxy.)
- **Rule:** DSpark's native scheduler keeps the longest prefix whose **cumulative product
  `∏_{i≤k} c_i`** clears a throughput-optimal threshold (paper Alg. 1); operationally, stop the
  block the first time the running product drops below `TAU`. The draft length `d` this cycle
  is therefore data-dependent, `d ∈ [0, block_size]`:
  - `d = 0` (first draft already below τ) ⇒ **the verify pass degenerates to a 1-token decode**
    — i.e. exactly a normal decode step. Speculation and plain decode are the *same code path*
    with `d = 0`; there is no separate "spec on/off" branch. (This is why there is no env flag:
    the cycle always runs; τ and the drafter's confidence decide its depth.)
  - `d = k_max` in confident, low-entropy regions (boilerplate, repeated structure) ⇒ full
    4-token draft, most of it accepted.
- **τ selection:** default range **0.4–0.6** (vLLM DSL); model- and workload-dependent, so we
  calibrate it on our fixtures against effective ms/token (§6, measurement 5). A single scalar,
  read from deployment config like a model path — **not** a per-path feature flag.
- **Why it compounds on WDDM specifically:** each avoided draft step is a full avoided kernel
  launch *and* an avoided MoE routing readback (the §7 wall). Confidence-gating removes draft
  launches exactly where they have negative expected value (low accept probability), so the
  per-launch WDDM cost we're most exposed to is spent only when it pays. For Zen Code the
  local predictability is strongly bimodal, so adaptive depth is a large lever, not a marginal
  one.
- **Variant (noted, not v1):** stop on the *cumulative* product `∏ c_j < τ_chain` (probability
  the whole remaining chain survives) rather than per-step `c_j`. Stricter; worth an ablation
  once per-step is measured. EAGLE-2's confidence-driven *draft tree* (multi-branch) is a
  further step but needs a tree-causal-mask verify kernel we don't have — out of scope; the
  linear chain here is the "keep 1–4 that converge" shape.

### 4.3 The verify step: K-token prefill row-group + hidden readout

Extend the wave with a fourth row-group, `verify_seqs`/`verify_inputs`, that reuses the
existing multi-slot prefill machinery rather than adding a kernel:

1. **Input row:** for a verifying session, the `d+1` tokens `[t_{p+1}, d_1, …, d_d]` (`d` =
   this cycle's adaptive draft count, `∈ [0, k_max]`) at positions `p+1 … p+1+d`, fed through
   the same `new_meta`/`seq_of`/`q_pos` path prefill uses (`wave.rs:1344-1363`). Causal masking
   makes draft position i attend to corpus + arena window + drafts `< i` — exactly verify
   semantics. When `d = 0` this is a single-row decode — the normal path (§4.2).
2. **Corpus / provenance selection:** select **once** using the base query at position `p`
   and **share** that corpus set across all `d+1` verify rows. The drafts are ≤`k_max` positions
   in the future and share locality; per-draft reselection would cost `d` extra BDP scans for
   negligible accept-rate change. This is an explicit approximation — it affects *which
   corpus is visible during verify*, never correctness of the accept test (that compares the
   main model's own argmax to the drafts). Flagged for measurement (§6).
3. **Logits:** add all `d+1` rows to `sel_rows` (`wave.rs:1503`) so the existing single
   batched `lm_head` GEMM scores them together — no new head path.
4. **Hidden readout (new):** also surface the post-`head_reduce`, post-final-norm residual
   hidden for each verify row (the tensor already computed at `wave.rs:1494-1511` just before
   `lm_head`). The accepted position's hidden seeds the next cycle's draft. Return it in
   `WaveStep` alongside `logits`.

`verify_seqs` is guarded like prefill (skip when empty) so pure-decode waves are unchanged.

**Kernel choice — reuse prefill now, extend decode later.** The prefill kernel is *correct*
for verify (gate test `prefill_rows_equal_decode_steps`, `paged.rs:2861`, proves prefill row i
== decode step i) but it is **not** flash-style query-tiled: it launches **one thread block per
query token** (`latent_prefill_kernel.cuh:986`, `grid(total_q,1,num_splits)`; the MMA M
dimension is *heads*, not tokens), so the `d+1` verify tokens **each re-stream the whole
window+corpus KV** — ~`(d+1)×` the attention memory traffic of one decode query — and it runs a
**per-layer baked-RoPE corpus pre-pass** each forward (`latent_prefill_kernel.cuh:962-973`;
~2 launches × 43 layers) that decode avoids via its **position-free persistent corpus cache**
(`latent_decode_kernel.cuh:327-357`). At `d ≤ 4` with `num_splits=1` that is also only ~5
thread blocks — badly under-occupied on a ~100-SM Blackwell part.

- **Phases 2–3 reuse the prefill kernel as-is.** Even paying `~5×` attention + the pre-pass
  launches, the verify forward stays well under the `~2.4× decode-forward` budget that makes
  speculation win (attention is a fraction of a 43-layer forward; the win survives an
  unoptimized verify). This lets us prove correctness and *measure* before spending kernel
  effort — measurement 5 breaks out verify:kernel vs verify:prepass launch time.
- **The optimized verify kernel is a small extension of the *decode* kernel, not a tuned
  prefill** (§7, its own phase in §8). Pack the `d+1` verify tokens into the MMA so **one
  KV-tile load serves all `d+1` queries** (true ~1× amortization), keep decode's position-free
  corpus cache so verify **reuses** the base position's already-cached corpus and **skips the
  per-forward RoPE pre-pass** (the dominant WDDM cost here — ~86 fewer launches/forward), and
  add the cheap `(d+1)²` causal diagonal. Clean because verify shares the base corpus selection
  (step 2) and the cache is position-free; real work because decode's M dimension is currently
  heads, not tokens.

### 4.4 Accept / reject

Greedy first (matches our current argmax decode and gives bit-exact base-model output).
`d` = this cycle's adaptive draft count (§4.2):

```
verify logits give a_i = argmax at position p+i, for i in 1..=d+1
accepted = [t_{p+1}]                      # committed last cycle, free
for i in 1..=d:
    if a_i == d_i:  accepted.push(d_i)     # draft confirmed
    else:           accepted.push(a_i); break_with_correction   # first miss: take model's token, stop
if all d confirmed:
    accepted.push(a_{d+1})                 # bonus token, free
```

- Commit length ∈ `[1, d+1]` new tokens this cycle (the `t_{p+1}` slot rolls in from the
  previous cycle's correction/bonus, so steady-state net commit is 1..d+1). With `d = 0` this
  reduces to committing one token — a normal decode.
- The corrected/bonus token and **its** verify hidden become `(t_{p+1}, h_p)` for the next
  cycle — the verify pass doubles as the next cycle's seed forward. No separate main decode
  step is ever run once speculation is active.
- **Sampling (temperature/top-p) variant** is a later extension: replace the equality test
  with the standard modified-rejection sampling (accept `d_i` w.p. `min(1, p_main/p_draft)`,
  else resample from the residual). It preserves the *sampled* distribution exactly. v1 ships
  greedy; the accept function is written to take a `Verdict` so the sampling arm slots in
  without touching the wave.

### 4.5 KV & corpus consistency (the real new bookkeeping)

Per cycle we speculatively advance state for `d+1` tokens, then keep only the accepted prefix.
Four pieces of state, three already truncatable:

1. **Raw SWA arena (main model)** — the verify kernel writes the `d+1` latents and self-bumps
   write-len (`commit_write_len`). On accept length `m ≤ d+1`, roll back with
   `truncate_sequence_to_tokens(seq, p+m)` (`sequence_ops.rs:2071`) + `set_sequence_offset`
   (`batched_inference.rs:2455`). **Present.** Needs a session-level token-granular wrapper
   (only a block-granular `truncate_sequence_to_blocks` wrapper exists at
   `batched_inference.rs:1216`).
2. **Drafter SWA KV** — same truncate for `MtpKvState`, plus the drafter writes `d`
   speculative positions; trim to the accepted count. New but small (single layer).
3. **Compressed corpus (`FloatGallery`) + streaming compressor** — **snapshot → absorb →
   restore + replay** (implemented; an earlier draft of this section proposed deferring
   absorption until after the accept decision, which is WRONG: corpus selection explicitly
   includes groups completed inside the current block — `kernel_attn_prefill_assemble`'s
   `n_visible[t] = base + (l0+t+1)/ratio` — mirroring decode's push-then-select, so deferral
   would diverge verify from decode at every group-boundary token). The implemented design:
   - `verify_block` takes a per-layer pre-verify snapshot (`VerifySnapshot` in `wave.rs`):
     each `IncrementalCompressor`'s streaming state (`CompressorState` — partial-group
     buffers, overlap prev-group, `group_idx`; `Arc`-clone cheap) + the gallery length.
     During the verify forward, prefill pass 1 stashes the block's projected compressor
     rows (comp + icomp per layer) into the snapshot — the replay source.
   - The generic driver routes its truncation through a `ManagedBatchedModel::
     truncate_sequence` hook (default = session KV truncate); DeepSeek's override adds
     `rollback_verify_state`: full accept discards the snapshot (the absorbed block IS the
     accepted text); partial accept restores the compressors, truncates the gallery
     (`FloatGallery::truncate` — every reader narrows to `len`, appends overwrite `len..`,
     so a length rollback suffices), and replays exactly the `m` accepted rows via
     `emit_groups_projected` — bit-identical to having absorbed only those tokens
     (`snapshot_restore_replay_matches_clean_absorb`, compressor.rs).
   - Without this, a partial accept leaves rejected draft rows in the partial-group buffer,
     a group pooled over draft content in the gallery, and `group_idx` advanced — the
     re-decoded positions get absorbed AGAIN as duplicate, shifted groups, and the model
     re-attends duplicated context (the observed repeated-sentence corruption).
4. **mHC residual checkpoint** — nothing to roll back: the residual is recomputed each wave
   from the arena/corpus, which we already truncated. The only cross-wave carried state is
   per-`(layer,seq)` compressor buffers, handled by (3).

**No `window_ring_snapshot`/`corpus_snapshot` full checkpoints** (`wave.rs:168-299`) on the
hot path — they copy full corpus/window state and are far heavier than truncate. They remain
for cold resume only.

### 4.6 Batching speculation across sessions

Both draft count `d` and accept length vary per session, so a batched spec wave is ragged.
Phasing:

- **v1 — single-session spec.** One session speculates per wave; other sessions decode
  normally in the same wave (they're just decode rows). Proves correctness and measures the
  single-session win, which is the stated target ("decode speedup"). This is the regime where
  the WDDM floor hurts most and the win is largest.
- **v2 — batched spec.** All sessions draft and verify together; the verify row-group packs
  `Σ (d_s+1)` rows across sessions (the multi-slot kernel already handles arbitrary per-seq row
  counts via `new_meta`/`seq_of`). Per-session `d_s` and accept lengths differ → per-session
  truncate. The wave-batching machinery (`decode|prefill|glue` layout, `wave.rs:632-649`)
  extends to a fourth `verify` group cleanly. **Adaptive draft length across a batch:** either
  keep it per-session (fully ragged, what the packed kernel already supports) or, like vLLM
  DSL, gate on the **batch-mean** confidence to keep `d` uniform across the batch (rectangular,
  simpler bookkeeping) — decide from the v2 measurement. Defer until v1 is measured.

---

## 5. Acquisition + loading the DSpark drafter — **IMPLEMENTED**

The drafter is a **separate GGUF** (`dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf`, 10.9 GB, at
the root of `bartowski/DeepSeek-V4-Flash-0731-GGUF`), not part of the main quant. Both the
acquisition and the load are done and tested:

- **Automatic download** (`zend/src/download.rs`, `zend --download-deepseek [--model-dir DIR]`).
  `ensure_deepseek` resolves the 4 main MXFP4 splits (mapping the repo's
  `DeepSeek-V4-Flash-0731-MXFP4/` subfolder → flat local names) **and** the DSpark drafter,
  downloading only what's missing (an existing main-model install pulls just the drafter). Same
  automatic HF path as the model download — no manual `curl`. Pure URL/path logic is unit-tested
  against the verified resolve endpoints.
- **Config + loader** (`deepseek4/dspark.rs`). `DsparkConfig::from_gguf` parses the `dflash.*`
  metadata through the existing `config_from_gguf` (arch = `dflash` ⇒ `n_layers` =
  `dflash.block_count` = 3) plus `block_size` and `target_layers`. `DsparkDrafter::load` loads
  the 3 backbone blocks via `load_block`, the `markov_w1/w2` head, `output_norm`, and the
  optional confidence head. The shared `token_embd` + `output.weight` are **not** in the file —
  the drafter reuses the target's. No env flag, no dual path: the drafter loads iff the file is
  present.
- **Pure math, unit-tested bit-exact** (`dspark.rs` tests): `MarkovHead::bias` = `W₁[prev]·W₂`
  (Eq. 5) against a hand-computed reference, orientation-robust to either GGUF storage layout;
  `ConfidenceHead::confidence` = `σ(wᵀ[h; W₁[prev]])` (Eq. 7).

**Remaining (GPU-coupled, needs a device to validate):** the backbone `draft()` forward with
target-feature extraction + KV injection (§4.1), and the verify/rollback/accept wiring in the
wave (§4.3-4.5). These are specified above; they are integration + kernel work, not open
research — the architecture is pinned by §2 and the loaded weights.

---

## 6. Tests & correctness gates (TDD, bit-exact where it applies)

Following repo convention — build tests with the code, assert raw bytes for codec/kernel
pieces, not tolerances.

1. **Markov / confidence math** (`dspark.rs`, **done**): `MarkovHead::bias` = `W₁[prev]·W₂`
   (Eq. 5) bit-exact against a hand-computed reference, orientation-robust; `ConfidenceHead`
   = `σ(wᵀ[h; W₁[prev]])` (Eq. 7). **Loader** (skip-when-absent, like the existing real-GGUF
   tests): `DsparkConfig::from_gguf` yields `block_count = 3`, `block_size = 5`, non-empty
   `target_layers`; `DsparkDrafter::load` loads 3 blocks + `markov_w1/w2` + `output_norm`.
2. **Verify == per-token decode**: a K-token verify row-group over `[t, d_1, d_2]` must be
   argmax-equal, row-for-row, to running those three tokens as three separate decode steps
   (inherits the existing prefill mirror gate `wave.rs:650-664`). Raw-token equality.
3. **Rollback exactness**: append `d+1` tokens, truncate to `m`, and assert the arena +
   position + compressor state are **byte-identical** to having only ever decoded `m` tokens
   (this is the property that makes speculation invisible to the model). Covers
   `truncate_sequence_to_tokens`, the drafter KV truncate, and `absorb_accepted`.
4. **End-to-end determinism**: greedy generation with speculation on must produce the **exact
   same token stream** as greedy decode with speculation off, on the StoryRewrite fixture, all
   configs. This is the headline gate — spec is a no-op on output by construction.
5. **Acceptance, draft-length & throughput measurement** (not a pass/fail gate, a report):
   log per-cycle draft length `d`, accept length distribution, and effective ms/token vs the
   `d=0` (normal-decode) baseline, swept over `TAU ∈ {0, 0.3, 0.4, 0.5, 0.6}`. Expect mean
   accept ≈ 2.4 and effective ≈ 31 ms/token at the best τ if draft launch overhead is
   contained, with adaptive length beating any fixed `k`. Also **check draft-confidence
   calibration** directly: bucket drafts by `c_j` and plot realized acceptance per bucket — the
   whole scheme rests on `c_j` tracking acceptance (EAGLE/DSL show it does; confirm on our
   model). Break out draft vs verify launch time to size the WDDM-readback concern from §1.

---

## 7. Risks & fallbacks

- **Target residency (DSpark's own caveat — the headline risk).** The paper/model card note
  DSpark pays off *when the target is GPU-resident*; when most of the target is offloaded to
  CPU, the drafter competes for VRAM and can be **slower than no speculation**. Our DeepSeek
  runs with **streaming experts** (not fully resident) — the at-risk config. Counterweight: our
  bottleneck is WDDM *launch* overhead, which amortization still attacks. Net effect is
  genuinely unknown until measured (test 5) — this must be an empirical go/no-go, not assumed.
- **Draft backbone overhead (WDDM).** The 3-block backbone forward carries MoE routing
  readbacks — but it runs **once per block** (all γ base logits in one pass), not once per
  token, so the semi-AR structure already amortizes it. Confidence scheduling (§4.2) further
  caps `block_size`. If test 5 still shows it eating the win: raise `τ` / lower `block_size`,
  keep the backbone on GPU-native resident MoE dispatch (`forward_moe_gpu`), or fall back to a
  cheap n-gram drafter behind the same accept/verify machinery (drafter-agnostic — a swap).
- **Target KV-injection integration (the main build risk).** `Hctx` from `dflash.target_layers`
  injected into a *bidirectional* draft attention (§4.1) is a new attention path on top of our
  causal paged-latent kernel. Getting it bit-faithful to the trained drafter is the crux of
  acceptance; it's the piece that must be GPU-validated (the CPU reference pins the Markov +
  confidence math, but not the injected-KV attention).
- **Verify-kernel efficiency (prefill reuse).** The reused prefill kernel re-streams KV per
  verify token and re-bakes the corpus RoPE per layer (§4.3) — fine for the first cut; if test 5
  shows verify:kernel/prepass launch time material, build the extended-decode verify kernel
  (Phase 5). On WDDM the per-layer pre-pass launches dominate the `~5×` L2-hot KV re-stream.
- **Confidence calibration.** DSpark's confidence head is trained to predict acceptance
  directly (§2 Eq. 8), so it should be well-calibrated — confirm on our model (test 5); if not,
  τ still bounds *wasted* drafts, or fall back to the max-softmax proxy.

---

## 8. Phasing

1. **Phase 1 — acquisition + loader + math. ✅ DONE.** Automatic download
   (`zend --download-deepseek`), `dflash` config + `DsparkDrafter::load`, and the bit-exact
   Markov/confidence math with gate test 1 (§5, §6.1). Nothing wired into decode yet.
2. **Phase 2 — verify path.** Fourth `verify` row-group reusing the prefill kernel, hidden
   readout (incl. the `target_layers` capture for `Hctx`), batched logits over `d+1` rows;
   gate test 2. No drafter yet — feed *known* tokens to prove verify+accept+rollback (tests 2, 3).
3. **Phase 3 — drafter forward + single-session spec.** `DsparkDrafter::draft` (semi-AR
   backbone with target-KV injection + Markov bias + confidence scheduling, §4.1–4.2), wire the
   full cycle for one session (§4.6 v1); calibrate `τ`; gates 3, 4; measurement 5 — **including
   the residency go/no-go (§7).** This is the milestone that delivers (or refutes) the speedup.
4. **Phase 4 — batched spec + sampling.** Ragged multi-session verify (§4.6 v2) and the
   modified-rejection sampling arm (§4.4).
5. **Phase 5 — extended-decode verify kernel (optimization, measurement-gated).** Only if
   Phase 3's measurement shows the reused prefill kernel's per-token KV re-stream or per-layer
   RoPE pre-pass is material: a decode-derived kernel that packs the `d+1` verify tokens (one
   KV-tile load for all), reuses the position-free corpus cache (no per-forward pre-pass), and
   adds the `(d+1)²` causal diagonal (§4.3, §7). Gated by the same `prefill_rows_equal_decode_steps`
   equality it replaces. Do **not** build speculatively — the Phase 3 numbers decide whether it
   pays.

Commit boundaries follow the phases; each lands with its gates green and no half-wired path.
