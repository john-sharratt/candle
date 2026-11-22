# DeepSeek-V4-Flash — DSpark Speculative Decode

**Status:** built end to end. The DSpark drafter, the batched verify wave, the accept loop
and the corpus rollback are in `latent_moe/`, and the generic driver
(`ManagedBatchedModel::speculative_decode_step`, `batched_inference/managed_model.rs`) runs
them. Gates: `wave_speculative_step_is_lossless`, `wave_speculative_step_with_drafter_is_lossless`,
`wave_speculative_speedup_with_drafter`, `wave_speculative_realtext_acceptance`
(`latent_moe/wave/tests_speculative.rs`).
**Goal:** draft a block of up to 5 tokens per decode cycle with the DSpark `dflash` drafter —
**as far as the drafter's own confidence justifies** (0 drafts ⇒ a normal decode) — verify
them in one main-model forward, and commit the prefix that converges. Turns our launch-bound
decode floor into ~2.4+ tokens per forward with no change to the output distribution.

**The entire drafter is implemented and GPU-validated on the real 10.9 GB weights**
(`latent_moe/dspark.rs`, `latent_moe/attention.rs`, `latent_moe/transformer.rs`): the
`dflash` config; the complete loader (every tensor); the context encoder `encode_context`
(Eq. 2); the Markov bias + confidence math (Eqs. 5, 7); the **backbone**
(`Attention::forward_injected` — non-causal latent MLA with `Hctx` injected via `wkv`;
`Block::forward_injected`; `DsparkDrafter::backbone` — mHC expand → 3 injected blocks →
`output_hc` head-reduce → `output_norm`); the **semi-autoregressive sampler** `sample_block`
(Eq. 4 + Alg. 1); and `DsparkDrafter::draft` tying it all together. `dspark.rs` carries 7 unit tests
that need no weights (the Markov bias against a hand-computed vector — exact equality — its
orientation robustness and shape refusal, the sequential sampler, the confidence schedule, the device
sampler against the host reference, and the confidence head, compared to 1e-6) and 5 `#[ignore]`d tests:
two CPU benches (`bench_cpu_expert_matmul_floor`, `bench_cpu_router`) and three that read the real GGUF
(`inspect_real_dspark_sizes`, `compare_dspark_vs_target_experts`, and `real_dspark_loads_and_runs_on_gpu`,
which loads the drafter and runs the 3 injected blocks → finite hidden `[5,4096]`). The engine plumbing (§4.3-4.5) is in place: the wave stashes the
target's hidden states at `dflash.target_layers` per sequence (`target_feat`) to feed `draft()`,
`draft()` takes the target's shared `token_embd`+`output.weight`, and `begin_verify` /
`end_verify` / `truncate_sequence` (`latent_moe/wave/managed.rs`) run the verify/accept/rollback
loop against the target — where real acceptance (and the residency go/no-go, §7) is measured. The drafter
is DSpark verified against the real GGUF — **not** the V3-MTP `eh_proj` recurrence an earlier
draft of this doc assumed.

**Reach.** The drafter is attached with `Engine::load_with_drafter` + `BatchedEngine::with_drafter`
(the drafter loads into the dense tier before the span reservation, so it is a permanent resident the
reservation measures around). The scheduler's load path (`load_arch.rs`) calls `Engine::load`, which
attaches none, and `DsparkDrafter::load` refuses a device of 64 GiB or less; see Open items.

---

## 1. Why this pays off *here* specifically

Our single-session decode is **not compute-bound** — it is bound by the WDDM per-launch
overhead (a `~74 ms/token` floor measured early in the port; the release-build plain wave in
`wave/speculative.rs`'s cost-ratio comment is ~112 ms single-session on the dev box). Every token pays
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
~74 ms/token to **~74/2.39 ≈ 31 ms/token** *before* the adaptive gain — and the draft is three
transformer blocks, not 43.

**The catch, stated up front:** the drafter is a full decoder block with MoE routing, so
each draft pass adds MoE launches on top of the verify forward's 43 layers. Neither side
reads routing back to the host: the verify forward's experts run through the live MoE
dispatch (`ExpertCache::forward_routed`, `docs/moe_live_dispatch_design.md`), and the
drafter's all-resident MoE (`dspark_experts.rs`: `moe_bucketize` + a device-resident pointer
table into `grouped_qmatmul_dev_q8a128`) never leaves the GPU. What remains is launch
overhead, and the confidence-gated draft length (§4.2) only spends it while the drafter is
sure. §6 makes measuring this the first gate — if draft launch overhead still eats the win,
the levers are `τ` (a code constant, §4.2), `k_max` or the n-gram drafter (§7).

**Measured (release build, single session, from the cost-ratio comments in `wave/speculative.rs`):** a
drafted step costs about 238 ms against about 112 ms for a plain wave — a ratio of about 2.1 — so
drafting wins only when the accepted tokens per step exceed it; the run counted 20.7 tok/s at 4.92
accepts, 2.32× greedy. The break-even grows with cohort width (about 5.4 accepts at width 8), because a
plain wave amortizes its launch floor across sessions while each drafted step pays a serial drafter
forward per session. This is why the built driver gates drafting on a rolling acceptance EMA (§4.2)
rather than always speculating.

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

- **Production:** `BatchedEngine::forward_wave` (`latent_moe/wave/managed.rs`, which calls
  `drive_wave`; the layer loop is `latent_moe/wave/sweep_wave.rs`) — one call = one wave =
  one forward step for every session in the batch.
- Per-wave the head block already gathers scored rows and runs **one batched `lm_head` GEMM**
  over `[1, R, dim]` (`sweep_wave.rs`, `e.lm_head().forward(&scored_hidden)`), returning per-row logits. Sampling (argmax) is
  the caller's host responsibility — there is no sampler in the module.
- **A K-token prefill launch exists** — `paged_latent_prefill_raw`
  (`latent_moe/paged/prefill.rs`) feeds K new tokens for a sequence through **one** kernel
  launch: packed `kv_new`, a `seq_of` query→slot map, per-token `q_pos`, and per-seq diagonal
  metadata `new_meta[n_seq,4] = {rows, base, start, -}`, and it is gated by
  `prefill_rows_equal_decode_steps`. The built verify does not use it: a verify block rides the
  wave as a multi-token member of the prefill slot and the sweep expands it into one
  **virtual decode slot per position** that runs the decode kernel (§4.3), so verify rows have
  decode numerics by construction.

So verify is "run the block as per-position decode rows in one wave". Drafting is the genuinely
new compute; rollback plumbing is the genuinely new bookkeeping.

---

## 4. Design

### 4.1 The drafter: DSpark `dflash` backbone + Markov head (see §2 for the math)

Module `latent_moe/dspark.rs` (**implemented** — config, loader, and the pure Markov/confidence
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
a cheap sequential bias+argmax — so the WDDM-costly part (the 3-block forward's launches) is
paid **once per cycle, not once per token**. This is structurally better on our launch-bound
box than a per-token recurrence. The backbone's MoE is the drafter's all-resident GPU-native
dispatch (`dspark_experts.rs`), so its routing never leaves the device.

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
- **τ selection:** vLLM DSL's default range is **0.4–0.6**; the value here is
  `DSPARK_TAU = 0.3`, a function-local `const` in `latent_moe/wave/speculative.rs::draft_one` that is
  passed to `DsparkDrafter::draft(…, tau)`. It is hard-coded: no config, environment variable or
  per-session setting changes it, and it is a single scalar, **not** a per-path feature flag. The
  sweep over `TAU` named in §6 measurement 5 is therefore done by editing the constant, not at runtime.
- **Cost control around τ (built):** `draft_one` also gates on a per-sequence rolling
  acceptance EMA (`SpecStats`, `SPEC_EMA_ALPHA = 0.25`): below `SPEC_MIN_ACCEPT = 2.3`
  accepted tokens per drafted step (plus 0.45 per extra session in the cohort) the drafter is
  skipped and the step is a 1-token verify, with a drafted probe every `SPEC_PROBE_INTERVAL =
  32` skipped steps; and it caps the block at `ceil(EMA + 1)` minus a shave when the verify
  wave's expert miss rate passes 50 %. All of these shape cost and acceptance, never the output.
- **Why it compounds on WDDM specifically:** each avoided draft step is a full avoided set of
  kernel launches (the §7 overhead). Confidence-gating removes draft
  launches exactly where they have negative expected value (low accept probability), so the
  per-launch WDDM cost we're most exposed to is spent only when it pays. For Zen Code the
  local predictability is strongly bimodal, so adaptive depth is a large lever, not a marginal
  one.
- **Cumulative, not per-step:** the built schedule stops on the *cumulative* product `∏ c_j < τ`
  (the probability the whole chain so far survives), not on the per-step `c_j`
  (`confidence_schedule_picks_prefix_by_cumulative_product`). EAGLE-2's confidence-driven
  *draft tree* (multi-branch) is a further step but needs a tree-causal-mask verify kernel we don't
  have — **Not built**; the linear chain here is the "keep 1–4 that converge" shape.

### 4.3 The verify step: decode-shaped block rows + hidden readout

A verify block rides the wave as a multi-token member of the **prefill** slot
(`begin_verify` in `latent_moe/wave/managed.rs` builds the `VerifyPlan`: the plain cohort's
committed tokens as decode rows, each verifying sequence's block as a `verify_inputs` row),
and inside the sweep (`latent_moe/wave/sweep_wave.rs`) the block is expanded back to **one
virtual decode slot per position**, so verify runs the **decode kernel** — decode numerics by
construction (FP PV, read-time rope, decode softmax structure) — rather than the prefill
kernel's int8-PV envelope, which flips narrow-margin argmaxes:

1. **Input row:** for a verifying session, the `d+1` tokens `[t_{p+1}, d_1, …, d_d]` (`d` =
   this cycle's adaptive draft count, `∈ [0, k_max]`) at positions `p+1 … p+1+d`. All of a
   block's virtual slots share identical slot headers serialized with the write length committed
   over the whole block (the rows are host-written before each layer's launch, and
   `mark_decode_writer_stale` makes the header build re-serialise the writer region); per-slot
   causal visibility comes from the kernel's `key_pos <= q_pos` bound alone, so draft position i
   attends to corpus + arena window + drafts `< i`. When `d = 0` this is a single-row decode — the
   normal path (§4.2).
2. **Corpus / provenance selection:** per row. Each row captures its selection intent against
   its OWN corpus size and the selection batches over every row of every session (pass 1c of
   the decode block in `sweep_wave.rs`), exactly as for plain decode rows. The earlier
   plan to select once at the base position and share the set across the block is not built:
   per-row selection is what makes verify equal per-token decode.
3. **Logits:** all `d+1` rows of a verifying sequence are scored (`verify_all_rows` marks the
   sequences for full-row head scoring) so the existing single batched `lm_head` GEMM scores
   them together — no new head path. `end_verify` splits the scored rows back per cohort.
4. **Hidden readout:** the post-`head_reduce`, post-final-norm residual hidden at
   `dflash.target_layers` is stashed per sequence and position (`target_feat`) as the wave
   scores each row; `draft_one` gathers the longest consecutive run of stashed positions ending at
   the block's start (up to `window_size`) as the drafter's `Hctx` source.

`verify_seqs` is empty for pure-decode waves, so they are unchanged.

**Kernel choice — the decode kernel over virtual slots.** The prefill kernel is correct for a
K-token verify (`prefill_rows_equal_decode_steps`) but launches **one thread block per query
token** and runs a **per-layer baked-RoPE corpus pre-pass**, while decode reads its position-free
corpus cache. Verify therefore uses the decode kernel; the one remaining cost is that each
virtual slot re-streams the window and corpus (the MMA M dimension is *heads*, not tokens), a
`~(d+1)×` attention memory traffic that stays well under the `~2.4× decode-forward` budget that
makes speculation win. Packing the `d+1` tokens into one MMA so a single KV-tile load serves all
of them is the open kernel optimisation (§7, §8 Phase 5).

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

1. **Raw SWA arena (main model)** — the verify wave writes the `d+1` latents from the host
   before each layer's launch, commits their length on the host (`set_len`), and marks the
   cached decode slot buffer with `mark_decode_writer_stale`, so the header build's sync
   re-serialises it. The reason: the decode kernel is the only writer that advances the serialised
   buffer's write length (on the device, one token per step); a block committed by the host advances
   only the host's count, so without the mark the next decode step writes its token over a committed slot
   and leaves the counted slot unwritten, reading whatever it last held (another sequence's KV once slots
   recycle). History: `../archived/deepseek/deepseek_perf_optimization_report.md` ("the live buffer's
   one invariant"). On accept length `m ≤ d+1`, roll back with
   `truncate_sequence_to_tokens(seq, p+m)` (`candle-nn/src/kv_cache/chunked/sequence_ops/tail.rs`, with the session-level
   token-granular wrapper in `batched_inference/block_tail.rs`) + `set_sequence_offset`
   (`batched_inference/stats.rs`). **Present.**
2. **Drafter SWA KV** — the drafter attends over the injected `Hctx` window rebuilt per block from
   the stashed target features (`draft_one`), so it carries no KV that needs trimming.
3. **Compressed corpus (`FloatGallery`) + streaming compressor** — **snapshot → absorb →
   restore + replay** (implemented; an earlier draft of this section proposed deferring
   absorption until after the accept decision, which is WRONG: corpus selection explicitly
   includes groups completed inside the current block — `kernel_attn_prefill_assemble`'s
   `n_visible[t] = base + (l0+t+1)/ratio` — mirroring decode's push-then-select, so deferral
   would diverge verify from decode at every group-boundary token). The implemented design:
   - `begin_verify` takes a per-layer pre-verify snapshot (`VerifySnapshot`/`LayerVerifySnap` in
     `latent_moe/wave/speculative.rs`, `snapshot_verify_state` in `wave/mod.rs`):
     each `IncrementalCompressor`'s streaming state (`CompressorState` — partial-group
     buffers, overlap prev-group, `group_idx`; `Arc`-clone cheap) + the gallery length.
     During the verify forward, the sweep stashes the streaming state after absorbing each
     block row (`row_states`: comp, icomp and gallery length per row).
   - The generic driver routes its truncation through a `ManagedBatchedModel::
     truncate_sequence` hook (default = session KV truncate); DeepSeek's override adds
     `rollback_verify_state`: full accept discards the snapshot (the absorbed block IS the
     accepted text); partial accept **installs** the stashed state after the last accepted row
     (the pre-block state when none is accepted) and truncates the gallery to the recorded length
     (`FloatGallery::truncate` — every reader narrows to `len`, appends overwrite `len..`,
     so a length rollback suffices). No replay and no per-layer launches: the per-row states
     are the wave's own decode-shaped capture, bit-identical to having absorbed only those tokens.
     (`snapshot_restore_replay_matches_clean_absorb`, `latent_moe/compressor/tests_incremental.rs`,
     gates the compressor state round trip.)
   - Without this, a partial accept leaves rejected draft rows in the partial-group buffer,
     a group pooled over draft content in the gallery, and `group_idx` advanced — the
     re-decoded positions get absorbed AGAIN as duplicate, shifted groups, and the model
     re-attends duplicated context (the observed repeated-sentence corruption).
4. **mHC residual checkpoint** — nothing to roll back: the residual is recomputed each wave
   from the arena/corpus, which we already truncated. The only cross-wave carried state is
   per-`(layer,seq)` compressor buffers, handled by (3).

**No `window_ring_snapshot`/`corpus_snapshot` full checkpoints** (`latent_moe/wave/mod.rs`) on the
hot path — they copy full corpus/window state and are far heavier than truncate. They remain
for cold resume only.

### 4.6 Batching speculation across sessions

Both draft count `d` and accept length vary per session, so a batched spec wave is ragged.
Phasing:

- **v1 — single-session spec.** One session speculates per wave; other sessions decode
  normally in the same wave (they're just decode rows). Proves correctness and measures the
  single-session win, which is the stated target ("decode speedup"). This is the regime where
  the WDDM floor hurts most and the win is largest.
- **v2 — batched spec (built).** All sessions draft and verify together: `begin_verify` packs
  every verifying sequence's block into one wave (`Σ (d_s+1)` virtual decode rows) beside the
  plain cohort's decode rows. Per-session `d_s` and accept lengths differ → per-session
  truncate. **Adaptive draft length across a batch** is per-session (fully ragged): each
  sequence's `draft_one` runs on its own stashed features and acceptance EMA, with the gate's
  threshold widened by 0.45 accepted tokens per extra session in the cohort (§4.2), so a wide
  cohort rides the plain batched wave unless the text is very predictable. Drafting itself is
  one drafter forward per session (`speculative_draft` maps over the cohort); batching the
  drafters is not built.

---

## 5. Acquisition + loading the DSpark drafter — **IMPLEMENTED**

The drafter is a **separate GGUF** (`dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf`, 10.9 GB, at
the root of `bartowski/DeepSeek-V4-Flash-0731-GGUF`), not part of the main quant. It is
placed by hand beside the main quant (`D:\models\deepseek-v4-flash-mxfp4\` on the dev
machines, where the gated tests look): no daemon runs DeepSeek, so the hosts' model download
(`host::download`) fetches only the prescribed model. The load is done and tested:

- **Config + loader** (`latent_moe/dspark.rs`). `DsparkConfig::from_gguf` parses the `dflash.*`
  metadata through the existing `config_from_gguf` (arch = `dflash` ⇒ `n_layers` =
  `dflash.block_count` = 3) plus `block_size` and `target_layers`. `DsparkDrafter::load` loads
  the 3 backbone blocks (attention + norms, `load_dspark_block`), the drafter's routed experts, the
  `markov_w1/w2` head, `output_norm`, and the optional confidence head. The experts are a different
  shape from the target's: all 3 × 256 are requantized at load (MXFP4 → f32 → `Q2_KO`, about 5 GiB) into a
  permanently VRAM-resident set (`DsparkStreamingMoe`, `dspark_experts.rs`), and
  `resident_slots_for_vram` returns a slot count only above 64 GiB of VRAM — below that `load` errors and
  speculative decode is disabled. The shared `token_embd` + `output.weight` are **not** in the file —
  the drafter reuses the target's. No env flag, no dual path: the drafter loads iff the file is
  present and the device clears the VRAM gate.
- **Pure math, unit-tested** (`dspark.rs` tests): `MarkovHead::bias` = `W₁[prev]·W₂`
  (Eq. 5) against a hand-computed reference by exact equality, orientation-robust to either GGUF storage
  layout; `ConfidenceHead::confidence` = `σ(wᵀ[h; W₁[prev]])` (Eq. 7), compared to 1e-6.

The backbone `draft()` forward with target-feature extraction + KV injection (§4.1), and the
verify/rollback/accept wiring in the wave (§4.3-4.5), are built on top of this load
(`latent_moe/wave/speculative.rs`, `latent_moe/wave/managed.rs`).

---

## 6. Tests & correctness gates (TDD, bit-exact where it applies)

Following repo convention — build tests with the code, assert raw bytes for codec/kernel
pieces, not tolerances.

_Where each is gated: 1 by the `dspark.rs` unit tests and its real-weight load test (§0); 2–3 by
`wave_speculative_step_is_lossless` (verify + accept + rollback with known tokens) and
`snapshot_restore_replay_matches_clean_absorb` (`latent_moe/compressor/tests_incremental.rs`, the
compressor state round trip); 4 by `wave_speculative_step_with_drafter_is_lossless`; 5 is the report of
`wave_speculative_speedup_with_drafter` and `wave_speculative_realtext_acceptance`. All of
`tests_speculative.rs` needs the real weights and is `#[ignore]`d._

1. **Markov / confidence math** (`dspark.rs`, **done**): `MarkovHead::bias` = `W₁[prev]·W₂`
   (Eq. 5) exactly equal to a hand-computed reference, orientation-robust; `ConfidenceHead`
   = `σ(wᵀ[h; W₁[prev]])` (Eq. 7), compared to 1e-6. **Loader** (skip-when-absent, like the existing real-GGUF
   tests): `DsparkConfig::from_gguf` yields `block_count = 3`, `block_size = 5`, non-empty
   `target_layers`; `DsparkDrafter::load` loads 3 blocks + `markov_w1/w2` + `output_norm`.
2. **Verify == per-token decode**: a K-token verify row-group over `[t, d_1, d_2]` must be
   argmax-equal, row-for-row, to running those three tokens as three separate decode steps
   (the verify rows run the decode kernel, so this is decode-step equality; the prefill-kernel
   analogue is `prefill_rows_equal_decode_steps`). Raw-token equality.
3. **Rollback exactness**: append `d+1` tokens, truncate to `m`, and assert the arena +
   position + compressor state are **byte-identical** to having only ever decoded `m` tokens
   (this is the property that makes speculation invisible to the model). Covers
   `truncate_sequence_to_tokens` and `rollback_verify_state`
   (`snapshot_restore_replay_matches_clean_absorb`).
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
   model). Break out draft vs verify launch time to size the WDDM launch-overhead concern from §1.

---

## 7. Risks & fallbacks

- **Target residency (DSpark's own caveat — the headline risk).** The paper/model card note
  DSpark pays off *when the target is GPU-resident*; when most of the target is offloaded to
  CPU, the drafter competes for VRAM and can be **slower than no speculation**. Our DeepSeek
  runs with **streaming experts** (not fully resident) — the at-risk config. Counterweight: our
  bottleneck is WDDM *launch* overhead, which amortization still attacks. Net effect is
  genuinely unknown until measured (test 5) — this must be an empirical go/no-go, not assumed.
- **Draft backbone overhead (WDDM).** The 3-block backbone forward carries its own MoE
  launches (all on the GPU — no routing readback, `dspark_experts.rs`) — but it runs **once
  per block** (all γ base logits in one pass), not once per token, so the semi-AR structure
  already amortizes it. Confidence scheduling (§4.2) further caps `block_size`. If test 5
  still shows it eating the win: raise `τ` / lower `block_size`, or fall back to a cheap
  n-gram drafter behind the same accept/verify machinery (drafter-agnostic — a swap).
- **Target KV-injection integration (the main build risk).** `Hctx` from `dflash.target_layers`
  injected into a *bidirectional* draft attention (§4.1) is a new attention path on top of our
  causal paged-latent kernel. Getting it bit-faithful to the trained drafter is the crux of
  acceptance; it's the piece that must be GPU-validated (the CPU reference pins the Markov +
  confidence math, but not the injected-KV attention).
- **Verify-kernel efficiency.** Each virtual verify slot runs the decode kernel and re-streams
  the window and corpus KV (§4.3); if test 5 shows the verify kernel's launch time material,
  build the token-packed extended-decode verify kernel (Phase 5).
- **Confidence calibration.** DSpark's confidence head is trained to predict acceptance
  directly (§2 Eq. 8), so it should be well-calibrated — confirm on our model (test 5); if not,
  τ still bounds *wasted* drafts, or fall back to the max-softmax proxy.

---

## 8. Phasing

1. **Phase 1 — loader + math. ✅ DONE.** `dflash` config + `DsparkDrafter::load`, and the bit-exact
   Markov/confidence math with gate test 1 (§5, §6.1).
2. **Phase 2 — verify path. ✅ DONE.** The verify block as decode-shaped rows in one wave (§4.3),
   hidden readout (incl. the `target_layers` capture for `Hctx`), batched logits over `d+1` rows;
   `wave_speculative_step_is_lossless` proves verify+accept+rollback with known tokens.
3. **Phase 3 — drafter forward + speculation. ✅ DONE.** `DsparkDrafter::draft` (semi-AR
   backbone with target-KV injection + Markov bias + confidence scheduling, §4.1–4.2) wired into
   the full cycle; `wave_speculative_step_with_drafter_is_lossless` is the end-to-end gate,
   `wave_speculative_speedup_with_drafter` and `wave_speculative_realtext_acceptance` the measurement.
4. **Phase 4 — batched spec + sampling.** Ragged multi-session verify (§4.6 v2) is built; the
   modified-rejection sampling arm (§4.4) is not (greedy only).
5. **Phase 5 — extended-decode verify kernel (optimization, measurement-gated; not built).** Only if
   the measurement shows the per-virtual-slot KV re-stream is material: a decode-derived kernel
   that packs the `d+1` verify tokens (one KV-tile load for all) and adds the `(d+1)²` causal
   diagonal (§4.3, §7). Do **not** build speculatively — the measurement decides whether it pays.

---

## Open items

1. **The conversation engine never attaches the drafter.** `load_arch.rs` calls `Engine::load` and
   `BatchedEngine::new`; only the gates call `Engine::load_with_drafter` and `BatchedEngine::with_drafter`.
   Speculative decode is reachable through the generic driver and the tests, not through the scheduler.
2. **The VRAM gate excludes two of the three dev machines.** `resident_slots_for_vram` returns a slot
   count only above 64 GiB, so the 16 GB and 24 GB cards never load the drafter; the 72 GB machine does.
   Whether a smaller resident fraction pays on those cards is unmeasured.
3. **Not built:** the modified-rejection sampling arm (§4.4, greedy only), batching the per-session
   drafter forwards (§4.6), the token-packed extended-decode verify kernel (Phase 5), and the
   confidence-driven draft tree (§4.2).
4. **No speculative-specific results row.** The measurement gates exist
   (`wave_speculative_speedup_with_drafter`, `wave_speculative_realtext_acceptance`); the 2.4-token and
   ~31 ms/token figures in §1 are the paper's static yield and a projection, not measurements on this
   engine. The engine's DeepSeek throughput on the RTX PRO 5000 (72 GB) is in `docs/performance.md`
   (row I12, §3.7: ×1 warm 319.0 prefill / 18.9 decode t/s, ×16 1,177.1 prefill / 114.4 aggregate decode,
   build `21c36efbd`, 2026-10-10), which does not say whether the drafter was attached. The measured
   drafted-versus-plain ratios (238 ms vs 112 ms, break-even ≈ 2.1 at width 1 and ≈ 5.4 at width 8) live
   only in the `speculative.rs` constant comments.
