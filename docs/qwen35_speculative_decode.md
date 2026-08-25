# Speculative decode for the Qwen3.5/3.6 hybrid

**Status:** built and measured end to end on the dense 9B. The drafter is the
checkpoint's own NextN/MTP head; the verify wave and the recurrent rewind are
model-side; the driver is shared with DeepSeek-V4
(`docs/deepseek/deepseek_v4_speculative_decode.md`).

---

## 1. What was already generic, and what was not

`ManagedBatchedModel` already carried the whole lossless loop
(`batched_inference.rs`):

| hook | job | default |
|---|---|---|
| `speculative_draft` | propose the next `k` tokens | none → plain decode |
| `verify_blocks` | run every block in one forward, score every row | sequential decode steps |
| `truncate_sequence` | roll a sequence back to the accepted length | session KV truncate |
| `speculative_decode_step_batch` | draft → verify → argmax → accept → emit → rewind | — |

Accepted tokens are always **the target model's own argmaxes**, so output is
bit-identical to greedy decoding whatever the drafter proposes. Everything below
is therefore free to be a heuristic about *speed*; none of it can change a token.

Two things stopped the hybrid using it:

* **No drafter.** DeepSeek overrides `speculative_draft` with DSpark. The
  `qwen35` lineage needed its own — see §2.
* **`truncate_sequence` refused any non-zero rewind**, and said why: `S` is a
  running sum over every token of the sequence, so there is no suffix to remove,
  and KV rewound under a state that still holds the un-truncated history is
  silent corruption. See §4.

There is deliberately **no generic fallback proposer**. One was built and
removed: an n-gram/prompt-lookup drafter over a per-sequence token log. It can
only re-propose text the sequence already contains, so it is worth nothing on
the reasoning and first-draft tokens a decode loop actually spends its time on,
and it charged a `Vec<u32>` per sequence growing with context to buy that. Its
measured acceptance was identical to two decimal places across a 0.8B and three
35B-class models — the tell that it was measuring the fixture, not the model.

## 2. The drafter: the NextN/MTP head the checkpoint ships

Qwen3.5/3.6 are trained with multi-token prediction and ship the head — one
block past the trunk, which SGLang serves as the NEXTN speculative algorithm.

**Pin the `-MTP-GGUF` repo.** The head is not a separate draft checkpoint (so
"no drafter file" is true and irrelevant), and the **plain GGUF conversion drops
it**. unsloth publishes both:

| | plain | MTP-preserving |
|---|---|---|
| repo | `unsloth/Qwen3.5-9B-GGUF` | `unsloth/Qwen3.5-9B-**MTP**-GGUF` |
| file | `Qwen3.5-9B-Q6_K.gguf` | *same name, same quant* |
| `nextn_predict_layers` | absent | **present** |
| MTP tensors | absent | **`blk.32.nextn.*`** |

Siblings exist for `Qwen3.5-35B-A3B`, `Qwen3.6-35B-A3B` and `Qwen3.6-27B`. The
0.8B has no MTP head in any conversion (none in its upstream config or tensor
index) and so has no drafter at all.

> Pinning the plain repo costs the drafter and shows up **nowhere else** — the
> model loads and answers identically. That is exactly how an earlier pass here
> concluded the architecture had no MTP at all, from evidence that was only ever
> about one conversion. `qwen35::mtp`'s checkpoint test exists to make a pin
> moved back to a plain repo fail loudly.

### What the head is

**Structurally a trunk layer.** `blk.N` (N = trunk depth) carries exactly the
tensor set of a trunk attention layer — `attn_q`, `attn_k`, `attn_v`,
`attn_output`, the Q/K norms, `attn_norm`, `post_attention_norm`, and the FFN.
Four tensors are its own:

```text
  blk.N.nextn.enorm.weight             RMSNorm over the token embedding
  blk.N.nextn.hnorm.weight             RMSNorm over the carried hidden
  blk.N.nextn.eh_proj.weight           [hidden, 2·hidden] over their concat
  blk.N.nextn.shared_head_norm.weight  final norm before the SHARED lm_head
```

It **shares** `token_embd` and `output.weight` with the target, so it costs one
block of weights, not a second model.

Because it is structurally a trunk layer, it *is* one: the loader builds it with
the same two helpers (`Loader::attention`, `Loader::dense_ffn`) every trunk block
uses, `MtpHead::block` is a plain `QuantLayer`, and it runs the production path —
`Qwen35AttentionLayer` → `forward_layer_batched_mixed` → the paged batched decode
kernel. Nothing about it is head-specific.

> **This was got wrong first, and the wrong version shipped.** The head
> originally carried its own `MtpAttention` weight struct — the same tensors, the
> same `[q|gate]` interleave, differing only in holding the Q/K gains as bare
> tensors — so that it could run `attention::gated_attention_core`, which is the
> F32 *reference* stack's attention. A reference oracle was running in
> production: ~45 tensor-op launches per sequence per drafted token, flat in KV
> length (so launch-bound, not history-bound), measured at 0.42 ms **per
> sequence** against the paged kernel's 0.166 ms **per call for four sequences at
> variable KV** — 25% of the whole decode step. The weight representation had
> been bent to fit the oracle's signature, which is what made it look
> deliberate. If a model path cannot be expressed with the production types, that
> is the finding, not an inconvenience to route around.

### The recurrence

Given the target's post-final-norm hidden `h` at the position whose argmax
produced token `t`:

```text
  x      = eh_proj( [ enorm(embed(t)) ; hnorm(h) ] )
  h'     = block(x)                       // attention + FFN, both residuals
  logits = lm_head( head_norm(h') )
  t'     = argmax(logits)
```

and the next step feeds `(t', h')` back in. A `k`-token draft is `k` one-block
forwards seeded once from the trunk — which is why §3's verify wave has to hand
back the target's hidden per scored row, not just its logits.

**Post-final-norm on both ends.** llama.cpp takes `t_h_nextn` after
`output_norm` on the trunk ("post-norm hidden state feeds both the LM head and
the MTP seed") and after `shared_head_norm` on the head, so the two ends of the
recurrence speak the same normalised space.

**The concat order is `[embedding ; hidden]`, and it is not guessable.**
`eh_proj` is a single `[hidden, 2·hidden]` weight spanning both halves, so
swapping them multiplies each input by the other's block. The head still returns
a finite, plausible token — and speculation is lossless, so nothing downstream
ever disagrees. A mis-wiring would surface only as acceptance that never beats
chance. The order is llama.cpp `src/models/qwen35moe.cpp`:
`ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0)` — ggml's dim 0 is the embedding
axis. `MtpInput`'s test pins it by which input the output responds to.

### The head's KV is a layer of the paged cache

The head attends over its own history, and that history is **the last KV layer
of the session** — one past every trunk attention layer, allocated by
`engine::session_kv_layers` and named by `engine::mtp_kv_layer`. Not shared with
any trunk layer (its history is its own), but sitting beside them and paged,
compressed, forked, sealed and truncated by exactly the same machinery.

**It prefills, decodes and glues at the same length, over the same tokens.**
That uniformity is the design, not a convenience. Every session-wide operation —
fork, view, prefix injection, turn sealing, truncation — assumes a sequence's
layers describe one stream at one length. A layer that stood at a different
length would have needed a range parameter threaded through all of them plus an
invariant test to keep it honest; a layer that stands at the same length needs
none of them to know it exists. `KvLayerMap` only ever yields
`0..num_kv_layers()`, so the sweep cannot name the head's layer and cannot
disturb it, while the wave driver's post-decode `set_current_seq_len` walks
*every* cache a sequence has and so advances it without being told.

`draft::head_wave_pass` is what fills it: after the trunk sweep, one attention
pass over the same rows at the same positions against `caches[head_kv]`. It runs
after the sweep rather than inside it because its input is the trunk's **output**
— `eh_proj([enorm(embed(t)) ; hnorm(h(t-1))])`, where `h` is the trunk's
post-`final_norm` hidden shifted one position right. Only the attention half
runs: all the pass owes is K/V, both projections of `attn_norm(x)`, and the FFN
would be a second full read of the head's weights for a value nothing reads.

Position 0 of a sequence has no `h(t-1)` and takes zeros. That costs nothing
real — a sequence always begins with a prefill, so position 0 is inside the
prompt and is never a draft seed; the zeros keep head row `i` aligned with trunk
row `i` so RoPE agrees, and nothing more.

**A rejected draft needs no rollback because the draft rolls itself back.**
`draft::draft_cohort` appends one position per sequence per step to the head's
layer — exactly as a decode row does, with slot headers built for that layer
**alone** (see below) — and truncates the whole run away before it returns. What
the target then accepts is written again, properly, by the next wave.

> **Slot headers for the head's layer alone.** During drafting the head's layer
> is legitimately one or two positions longer than the trunk's — that is what a
> speculative position *is*. A whole-stack `build_decode_metadata_at` would read
> that as divergence and "heal" it by truncating the proposal away, so the draft
> asks for `head_kv..head_kv + 1` and the head's headers are the group's only
> entry, at index 0. The wave is uniform; drafting is not, and that is the one
> place the difference shows.

The head ropes on **absolute** sequence positions, like every trunk layer,
because its history is the sequence's — one row per token, including the prompt.
The earlier private-KV drafter roped *relatively* over a window that began at the
first generated token, so it never saw the prompt at all; giving it back is worth
0.42 accepted tokens per step (§5).

## 3. The verify wave

`HybridBatched::verify_blocks` runs the plain cohort and every drafted block in
**one** forward.

Each block is a **prefill span**, not a run of decode rows — the recurrence is
sequential within a sequence, so two rows of one sequence cannot decode in
parallel against a single carried state, and the prefill scan is the form that
walks them in order. Plain (undrafted) sequences ride the same wave as ordinary
decode rows, so the step pays one launch floor rather than two.

Naming the verifying sequences does three things inside the sweep:

1. **The head scores every one of their rows** — a block position's prediction
   is what the proposal at that position is checked against.
2. **Each DeltaNet layer stashes their recurrence operands**, for §4.
3. **The post-final-norm hidden is captured per scored row** — the MTP seed.
   Armed for *both* cohorts: a plain row's hidden is what lets a sequence that
   drafted nothing this step draft on the next, and it is the only way the first
   step after prefill ever acquires a seed.

## 4. The rewind: replay, don't subtract

After the accept the sequence must stand at `pos + m` with **every** piece of
state consistent with exactly those tokens. Paged KV is append-only, so its
truncation is exact. The head's KV never held the rejected tokens (§2). The
DeltaNet recurrence is the hard one, and the way back is forward:

* The store's ping-pong means a wave writes the buffer it is **not** reading, so
  immediately after `commit_wave` the non-live half still holds the state the
  block was entered with — untouched, at no cost (`layer_state_rewind`).
* The mixer's arithmetic for row `i` depends on row `i` and the rows before it,
  and on nothing after. Re-running it over the block's first `m` rows from the
  entering state produces exactly the state the model would have had if only
  those `m` tokens had ever been decoded.

So `truncate_sequence` replays the block's **operands**, which the wave arena
reclaims when the forward ends — hence the stash (`SpanOperands`: `qkv`, `z`,
`beta_lin`, `alpha_lin` rows, one set per DeltaNet layer, `Some` only for a span
that will have to rewind, so an ordinary wave copies nothing).

Post-projection deliberately: re-running the projections would rest bit-identity
on a GEMM's reduction order not depending on its row count. The replay runs
through `delta_net_mix_spans` — the same function the wave ran — so it takes
whichever path the wave took and matches it by construction.

**Both the stash and the hidden buffers are sized before the forward opens.** A
wave's storage is claimed by `admit_wave_kv` and the transient tier is placed
against the arena frontier as it stands when the forward begins, so a device
allocation from inside the sweep is refused outright.

## 5. Measured — Qwen3.5-9B, dense, Q6_K

RTX 4090 Mobile 16 GB, release, StoryRewrite fixture, 256 generated tokens, one
context, BF16 KV. Every row validated 100% against the same fixture the plain
run validates against — the text is identical to greedy decode.

| draft budget | t/s | vs plain | accepted/step | of a possible |
|---|---|---|---|---|
| 0 (plain) | 52.5 | — | — | — |
| 1 | 80.2 | 1.53× | 1.99 | 2 |
| **2** | **104.4** | **1.99×** | **2.97** | 3 |

and at four concurrent contexts, same run:

| draft budget | t/s (cohort) | vs plain | accepted/step (per session) |
|---|---|---|---|
| 0 (plain) | 163.3 | — | — |
| 1 | 224.9 | 1.38× | 1.99 |
| **2** | **301.3** | **1.85×** | **2.97** |

Ratios are **within-run**, and only within-run: this laptop's absolute t/s has a
wide band under heat soak (§6), so a number here is comparable to the plain row
beside it and to nothing else. An earlier, warmer run of the same code read
1.75× and 1.39× on these two rows with identical acceptance — the acceptance
figure is the stable one, the t/s ratio moves with the card.

**1.99× at budget 2**, and the shape of the curve is the interesting part.

> **Budgets 3 and 4 are no longer swept.** They were, and they turned the curve
> over — 74.7 and 74.0 t/s against budget 2's 83.8, at 2.52 and 2.80
> accepted/step. That measurement is what set `MTP_MAX_DRAFT = 2`; past the cap
> the drafter clamps, so sweeping further would report the same configuration
> twice. The binding reason for the cap is the expert cache rather than the head:
> a verify wave carries `d + 1` rows per session where a decode wave carries one,
> and every extra row routes independently, so draft width multiplies the routed
> expert union each MoE layer must have resident.

*Acceptance rose when the head got the prompt.* 2.55 → 2.97 per step at budget 2
came entirely from making the head's KV a paged layer: it now attends over the
prompt and every accepted token from position 0, where the private-KV drafter's
window began at the first *generated* token. Solving `1 + p₁ + p₁p₂` for the two
budgets gives a first proposal landing ~99% of the time (was ~96%) and a second
~99% (was ~61%).

**And that second number is where this fixture stops being evidence.** A rewrite
task is copy-heavy, and a drafter that can now see the prompt can copy from it —
which is precisely the mechanism that took p₂ from 0.61 to 0.99. The gain is
real and it is lossless either way, but it is the *fixture's* ceiling being
measured, not the head's. A non-copy fixture was already the honest next
measurement before this change; it is more so now, because the change moved the
number most sensitive to copying.

*Per-token acceptance decays fast.* At budget 1 the head is right 1.96 times out
of 2 — it predicts the immediate next token ~96% of the time, which is the
strongest available evidence the recurrence is wired correctly (a mis-wired head
would sit near chance). The second draft lands ~59% of the time, the third and
fourth almost never. A one-block NextN head has short reach; SGLang's own
recommended `num_speculative_tokens` for NEXTN is in the same range.

*Draft cost is sequential and is what turns the curve over.* Every proposed
token is another one-block forward whether or not it is accepted, so budget 3
buys no more acceptance than budget 2 and pays an extra forward for it. The
optimum is where marginal acceptance stops covering a draft step, and here that
is 2.

*Draft cost is now the production path.* The head runs
`forward_layer_batched_mixed` over one decode group — the same paged batched
kernel every trunk attention layer runs, one launch for the whole cohort, with
per-sequence KV lengths carried by the slot descriptor table. What it replaced
was the reference attention at ~45 tensor-op launches per sequence per drafted
token (see §2). Since draft cost is exactly what caps the budget, that was the
win worth taking.

The plain gates are unmoved: `test_parallel_batched_forwarding_9b` runs 16/16
valid, C10 at 6.31× on the MTP checkpoint — the ratio shifts slightly from the
8-layer calibration because the head's KV is now a ninth paged layer and
compresses with the rest.

## 6. Correctness gates

* `qwen35::mtp::tests::the_first_half_of_the_concat_is_the_embedding` — the
  ordering, by which input the output responds to. Cannot pass on the wrong
  order, which the end-to-end number could.
* `qwen35::mtp::tests::the_pinned_checkpoint_carries_a_draft_head` — the pinned
  GGUF really has the head, at the right block, with `eh_proj` spanning both
  halves. A pin moved back to a plain repo fails here.
* `qwen35::draft::tests::a_draft_leaves_the_head_at_the_trunk_s_length` — the
  two invariants the design rests on, through a real session: after the wave the
  head's KV layer holds exactly what every trunk attention layer holds, and after
  drafting it holds that again. A leak in the second is not visible as a wrong
  answer — it is a length skew the next wave silently "heals" by truncating, i.e.
  a dropped token discovered much later and somewhere else.
* `qwen35::engine::tests::an_mtp_checkpoint_pages_one_more_kv_layer_and_it_is_last`
  and `…::a_full_sweep_claims_the_head_s_kv_layer_and_a_partial_window_does_not`
  — the head's layer is past everything `KvLayerMap` can name, and the one range
  that admission, the failure rollback and the sweep all share covers it exactly
  when the head's pass runs.
* `delta_net::mix::tests::replaying_a_prefix_equals_running_only_that_prefix` —
  a 5-row block replayed at 2 rows must equal a clean 2-row run, and the block
  and prefix must have ended in *different* states so it cannot pass by
  comparing a rewind to a no-op.
* `qwen35::spec::tests` — the wave's scored rows split back to their blocks.
* `quantized_qwen35::tests::speculative_decode_9b` — §5, each budget validating
  its output at 100%.

## 7. What is not built

* **MoE checkpoints.** The 35B/3.6 MTP block is routed (256 experts + shared),
  and the expert cache is sized and indexed over the trunk's MoE layers only.
  The loader refuses that case loudly rather than reading experts nothing
  staged, and those models stay pinned to their plain repos until the cache
  carries the head's layer.
* **Sampling (temperature / top-p).** The accept test is greedy equality. The
  standard modified-rejection arm preserves the sampled distribution exactly and
  slots into the same driver.
* **A non-copy measurement.** See §5 — and it matters more now than it did.
* **The head's KV is compressed like any other layer.** It participates in the
  C0–C10 policy, which is consistent with it being a layer, and its errors cost
  draft *quality* rather than correctness (verify keeps only the target's own
  argmaxes). Whether the head would rather spend the bytes than the acceptance
  is unmeasured; the ladder shows it costs nothing at C10 on this fixture.
