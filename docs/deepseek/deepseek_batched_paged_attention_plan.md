# DeepSeek-V4 — batched/paged hybrid attention + kernelization plan

## Abstract

DeepSeek-V4-Flash replaces dense attention with a **natively sparse, learned** form: one 512-d
latent per position serves as both K and V, and each query attends a **fixed ~640-key budget** — a
sliding window plus an Indexer-selected top-k over a pooled compressed corpus — regardless of
context depth. This plan ports that model from its working single-session prototype (host-tensor KV,
green on the "Paris" golden, fully int8) onto the engine's production **batched/paged substrate**:
the window moves into the arena KV cache, the compressed corpus into a tiered float gallery, and the
attention into a thin `paged-deepseek` fork of the tuned INT8 paged kernels, batching ~64 sessions
with exactly one host readback per decoded token.

Two ideas carry the design past the model's native 1M-token limit. First, **retrieval becomes
two-stage**: the engine's training-free binary BDP scan runs as a cheap recall filter *inside the
Indexer's own learned space*, and the Indexer float-rescores only the shortlist — so the learned
selector never sees a corpus larger than it was trained on, removing both its cost bound and its
training bound. Second, **persistence exploits that the compressor is a monoid**: KV is stored
pre-RoPE and position-free, partial compression groups carry an associative `(m,l,acc)` accumulator
across turn boundaries, and the fold is completed in the glue phase — the same seam where
reorder-invalidated boundary markers are already re-prefilled — so a conversation persists, resumes,
and reassembles exactly. Everything static stays baked (sealed turn interiors, intra-turn markers,
static system-prompt tree glue); only the few dynamic seams regenerate. The implementation advances
along a strict test ladder — frozen CPU reference → kernels on synthetic data → the "Paris" engine
golden → a temp-substrate conversation engine — with each step gated on the previous rung staying
green, and the reported compression re-baselined against what a traditional FP16 dense-attention
model would hold at the same depth.

## Key principles (what drove every outcome below)

1. **Attend a relevant subset, never the full history.** Qwen does it externally (BDP elevates
   what dense attention sees); DeepSeek does it natively (the Indexer). Same idea, different layer —
   so wrap the native selector with the external one instead of choosing between them (§A, §D).
2. **Coarse-to-fine: training-free recall wraps learned precision.** Binary sign-space BDP handles
   unbounded recall cheaply; the learned float Indexer supplies exact top-k on a shortlist it can
   trust. Each stage does only what it is good at (§B, §D).
3. **The attended budget is fixed; only the addressable corpus grows.** ~640 keys per query at any
   depth — that bound *is* the O(1)-error property. Scaling to 1M+ means growing RoPE extent and
   corpus tiers, never the window or top-k (§B, §L).
4. **The conversation is the KV corpus.** Token ids are ground truth; per-layer KV (window ring +
   compressed pair + partial tails) is the persisted cache that makes resume O(1). The residual
   stream is per-token scratch and is never persisted (§C).
5. **KV is position-free and relocatable.** Store pre-RoPE; derive position and apply RoPE in the
   kernel at read time. Bit-identical KV at position 100 or 900,000 is what enables prefix sharing,
   tier migration, and reassembly (§C).
6. **Streams fold as monoids; boundaries are cuts in the fold.** The compressor pool is associative
   under the LSE merge — the same primitive as split-KV combine — so turn-aligned persistence and
   ratio-aligned compression coexist by carrying `(m,l,acc)` across the cut (§C, §E, §G.6).
7. **Embed everything static; regenerate only what reordering invalidates.** Sealed interiors,
   intra-turn glue, and static tree glue are baked and Arc-cloned; only inter-turn seams and
   collection seams re-prefill — and the compression fold rides those same seams for free (§E, §F).
8. **Reuse the substrate; fork only what attention control flow forces.** pal_map routing, INT8 MMU,
   32-token paging, split-KV combine transfer verbatim; the fork exists only for K≡V single-latent,
   the index-driven walk, and partial RoPE (§G, Part III).
9. **Everything stays on the GPU; one readback per token.** The sampler sync is the only
   host round-trip decode is allowed. Weight preparation happens offline, never at load (§1, §2).
10. **Every step is gated on a frozen golden.** The CPU reference is never edited to make a kernel
    pass; kernels validate on synthetic data before touching live state; the substrate enters last;
    "Paris" must survive every rung (Part IV).

---

Synthesis of a code investigation across the paged attention kernels, the arena KV system, the
wide-Q per-conversation state machinery, and the DeepSeek decode path. This is the plan for turning
the current **single-session prototype** (`latent_moe/engine.rs`, host `Vec<Tensor>` KV) into a
**batched/paged engine on the production substrate**, plus the smaller fixes.

---

## 0. Where we are

- `latent_moe/engine.rs` (`latent_moe::Engine`) is a **single-session reference** — one conversation, KV as
  host `Vec<Tensor>`, dense attention in Rust tensor ops. Correct + green ("Paris"), fully int8.
- The production batched path already exists for Qwen3-MoE: `latent_moe/batched.rs` implements
  `BatchedModelCore` + `BatchedAttentionLayer`, drives the **paged INT8 attention kernels**
  (`paged-decode/`, `paged-prefill/`) over an **arena KV cache** (`candle-nn/kv_cache/chunked/`),
  batching ~64 sessions spatially across the grid. **DeepSeek must join this path.** The int8 weight
  work done already carries over unchanged.

_Reuse audit (built components): int8-KO forward reused verbatim (`forward_via_int8` mod.rs:2361, `dense_qmatmul` cuda.rs:5255). `QLinear` (latent_moe/linear.rs:10) Quant/Int8 arms overlap the transformers `QMatMul` wrapper (quantized_matmul.rs:18) — only its `Dense` reference arm is additive (a fold-able cleanup). MXFP4_KO (ko_quant.rs:25), Sinkhorn (sinkhorn.cu:27), `prepare_ko_gguf` (reuses `repack_to_host` cuda.rs:3886), and the Compressor/Indexer/mHC math are NOVEL — no prior art._

---

# Part I — Architecture foundations (why the design is what it is)

## A. How DeepSeek attention differs from Qwen3 (the premise)

**Qwen3-30B** (what the paged kernels were built for): standard GQA, **dense over the full causal
history**, separate K/V per KV head, full-vector RoPE. The model has no notion of relevance — it
attends whatever is resident. "Unbounded context" for Qwen is achieved **outside the model**: the
BDP/provenance layer scans the corpus with `sign(Q)·sign(K)`, ranks past chunks/turns, and
**elevates the winners into the hot KV tier** so the dense kernel only *sees* a relevant subset.
Coarse (chunk/turn), residency-level, external, binary, training-free → demonstrated to 25M.

**DeepSeek-V4:** selection is **inside the model, and trained.**
- **Latent single-KV (K≡V):** one 512-d latent per position serves as **both K and V**, shared
  across all query heads (MQA, `n_kv_head=1`). The "value" is the (partially-roped) latent itself.
- **Hybrid two-source attention:** each query attends `[ sliding window (last window_size raw
  tokens) ‖ indexer-selected top-k compressed entries ]`. A **Compressor** pools every `ratio` raw
  tokens into one compressed entry; the **Indexer** (learned `q·k`) picks the top-k.
- **Layer kinds:** SWA (window only), HCA (window + all compressed), CSA (window + indexer top-k).
- Per-head learned **sinks**; **partial-range RoPE** (nope‖rope); grouped low-rank output proj.

Both systems implement the same idea — *attend a relevant subset, not all of history* — at different
layers: **Qwen = dense attention + external BDP retrieval; DeepSeek = native learned sparse
attention (the Indexer).** The Indexer is "**BDP, but learned, float, per-query, trained-for-1M.**"
That correspondence is the whole design.

## B. What bounds DeepSeek to 1M

The **attention is O(1) in context** — each query attends a fixed budget (`window_size + index_topk
≈ 640` keys) regardless of depth. That is the O(1)-error property; it is **not** the limit.

The limit is the **Indexer**, two ways:
1. **Learned → trained for ≤1M.** The Indexer (and the YaRN RoPE on the long-context layers) is
   calibrated to the context distribution seen in training. Past ~1M its top-k stops being the
   actually-relevant entries. *Quality bound.*
2. **O(N) float scan.** The Indexer scores **all** `N/ratio` compressed entries every step to find
   the top-k — a linear float `q·k` scan. *Cost bound.*

BDP has neither: training-free (position never goes OOD) and cheap-binary (XNOR+popcount, 3–10 ms
over the full corpus). That contrast is the entire lever for §D.

## C. Persistence — the conversation *is* the KV corpus

**Two clocks.** The residual/mHC stream flows layer→layer but is **rebuilt from scratch every token**
and discarded — it is one token's computation, not conversation state. The conversation state is the
**KV cache** each layer accumulates; the inter-layer flow is never persisted, the KV is.

**The persisted unit** (per attention layer, per conversation):
```
window ring     [window_size, 512]   — pre-RoPE
attn corpus     [G, 512]             — pre-RoPE, post-RMSNorm   (attended values)
indexer corpus  [G, 128]             — pre-RoPE, post-RMSNorm   (scoring keys)
trailing partial group (per compressor: attn AND indexer)      — pre-RoPE
```
Ground truth is the **token ids** (the KV is a cache, replayable in O(N)); we persist the KV to make
resume O(1) (no re-prefill). The corpus is a **pair** (512 attended + 128 scoring) — persist both, or
the Indexer has nothing to score against on resume.

**Position + RoPE are NOT persisted** — same as the paged kernels: store KV **pre-RoPE**, and the
decode/prefill kernel applies RoPE at read time from a position **derived from the arena layout**
(window: token seq-pos from `TokenSlice.rope`+offset; compressed: group-start = `f(corpus index)`).
This drops the position counter entirely and makes the persisted KV **position-free / relocatable**
(bit-identical at pos 100 or 900,000) — the property that already lets the paged cache prefix-share
and tier-migrate.

**Turn/group misalignment — the compressor is a monoid.** Persistence is **turn-aligned**;
compression is **`ratio`-aligned**; so groups straddle turn boundaries, leaving partials. The pool
is a per-channel softmax-weighted sum:
```
s_t      = score_t + ape[within_group_offset(t)]
entry[c] = Σ_t softmax_t(s_t[c]) · kv_t[c]          # pre-RoPE
```
which is **associative under the online-softmax (LSE) merge** — the *same* primitive as the attention
split-KV combine. Represent a partial as per-channel `(m, l, acc) = (max s, Σ e^{s−m}, Σ e^{s−m}·kv)`;
merge two partials of the same group by the LSE rule; finalize `acc/l` → RMSNorm when the group
completes. So a turn persists its complete groups **+ a trailing `(m,l,acc)`**; on resume the next
turn's leading tokens fold in and the group finalizes — **exact, no re-prefill**. The fold is
position-free; the only thing riding along is each fragment's bounded **within-group offset**
(`0..ratio−1`), because `ape` is *relative*. (Overlapping `ratio=4`: same fold, the partial also
carries the retained prev-half; dependency window is `2·ratio`.)

The compressor being a monoid `⟨(m,l,acc), LSE-merge, identity (−∞,0,0)⟩` is **why** turn-aligned
persistence and ratio-aligned compression coexist: you are folding a stream, and folds do not care
where you cut them, as long as you carry the accumulator across the cut. **The turn boundary is just
a cut in the fold.**

> **Where the fold fires (see §E):** the monoid is the merge *operator*; it does not run at
> persistence time. On a **reassembly** (BDP reorders the prefix) the boundary group is
> *reorder-dependent* — its tokens' hidden states change — so it must be **re-folded against the new
> order in the glue phase**, using the boundary tokens' freshly re-prefilled compressor projections.
> Only the trailing partial `(m,l,acc)` is persisted; the finalized boundary entry is regenerated at
> the seam, never stored baked. On a plain resume with no reorder this degenerates to the pure
> arithmetic replay above.

## D. Corpus selection — BDP recall (in Indexer space) → Indexer precision

**Reuse the wide-Q BDP scan as the *recall* stage, not as a replacement for the Indexer.** Two
refinements make it right:

1. **Run BDP in the Indexer's own learned 128-d space** — sign the *Indexer's* query and the
   *Indexer's* keys, XNOR+popcount agreement over that space (not the raw attention Q). The model
   *trained* that space to make relevance ≈ similarity, so the domain gap is small (query and key are
   the same learned relevance representation — unlike the tool-selection call-vs-def gap that runs at
   chance). Theory: sign-agreement is SimHash — Hamming agreement estimates the cosine **angle**,
   which is the direction signal the Indexer's `Σ_h relu(q·k)·w_h` is dominated by.
2. **Binary is lossy on precision** — it drops the Indexer's per-head gating (`weights_proj`), the
   relu, and magnitude. So BDP is a faithful **recall** filter but a lossy **precision** selector,
   and the model was QAT-trained on the Indexer's *exact* top-k.

**Two-stage, coarse-to-fine:**
- **BDP recall** (cheap binary, training-free, O(N) but ms over millions): scan the full corpus —
  *including past 1M* — and shortlist a few thousand candidates in the sign space.
- **Indexer precision** (learned float `q·k`, now **O(shortlist)** not O(corpus)): re-score the
  shortlist → the top-k the model expects.

This **kills both 1M bounds at once**: the **cost bound** (Indexer's O(N) float scan → cheap BDP
O(N) + float O(shortlist)); the **training bound** (BDP handles the >1M recall; the Indexer only ever
re-scores a shortlist that fits inside its ≤1M competence — it *never sees a corpus bigger than it
was trained on*). And it is **one retrieval hierarchy**: the engine's BDP is the outer recall stage,
DeepSeek's Indexer is the inner precision stage — the same coarse-to-fine that turns 32K into 25M,
now wrapping the model's native selector instead of a dense attention.

**Cadence (what keeps the one-readback rule intact):** the two stages run on different clocks.
**BDP recall is asynchronous shortlist maintenance** — per-turn / periodic, off the decode hot path
(the paged GPU BDP scan, `gallery_arena/scan.rs::run_batched_bdp_scan`), refreshing which corpus
entries sit in the `FloatGalleryArena` hot tier. **Indexer precision is per-token and fully
on-device** — the float re-score runs over that resident shortlist and yields a device index tensor
the decode kernel walks directly (§G.4). Neither stage reads back to the host; the current per-token
`to_vec1` + host sort (indexer.rs:184) is exactly what this replaces (§2's −C syncs).

**Must be measured, not assumed:** BDP **recall** — does the sign-space top-M contain the Indexer's
float top-k? Sweep M vs recovered-fraction, **per layer**, on real traces (the
`provenance-experiment-findings` harness is the instrument). Shortlist size trades recall for Indexer
cost. Pure-BDP replacement (drop the Indexer) is possible but bets the model tolerates OOD selection;
the two-stage version does not require that bet.

_Measured (first real-trace sweep, `indexer_recall_sweep_real_traces`, 416-token generation, G=96
entries/CSA layer, probe k=8): recall@M is **moderate, not strong** at this depth — per-layer
recall@M=32 spans 0.00–1.00 (median ≈ 0.5), recall@M=64 spans 0.38–1.00. The relu + per-head gate
weights discard information plain sign-agreement can't see — exactly the precision loss anticipated
above. Consequences: (1) the two-stage structure stands (any M-superset is valid; the Indexer
re-ranks), but the shortlist must be sized generously and re-measured at depth — the step-6
long-context runs re-sweep at G in the tens of thousands, the regime that actually matters; (2) if
depth confirms weak recall, the recall kernel has headroom: gate-weight-aware agreement (weight each
head's popcount by its `w_h`) is a small kernel change. Machinery validated exactly: recall@G ≡ 1
per layer, two-stage ≡ full Indexer when the shortlist covers the corpus._

**Storage:** the compressed corpus is unbounded + per-layer (tens of GB at 1M), so it lives in a
tiered store. The provenance corpus's *tiering skeleton* (`StreamId` residency, fingerprint upload,
free-list pool, LRU + rebuild-from-warm) transfers, but its leaf is sign-bits; the compressed entries
are floats. So build a **`FloatGalleryArena`** — the same lifecycle skeleton, **float leaves**
(`[G,512]` attended + `[G,128]` scoring) — with **BDP sign-signatures stored alongside** for the
recall scan. We reuse the corpus's *tiering* + add a *sign-index for recall*; we do **not** reuse its
BDP *selector* verbatim (it returns case-votes, not per-entry top-k — see §"Decisions").

## E. The glue phase — the seam, and where the compression merge lives

**What glue is (the seam problem).** On every projection, BDP reselects and **reorders** which past
turns/sections sit in the slot. Sealed KV Arc-clones for free (immutable, position-relative via slice
`rope_base`). But the **inter-turn boundary markers** (`user_start` opening a turn, `assistant_end`
closing it, the `/no_think` switch) have hidden states **dominated by whatever causal prefix now
precedes them** — which changed under the reorder. So their KV is invalid after reassembly and must
be **re-prefilled** against the freshly-assembled order. That re-prefill is the **glue phase**: a
co-batched forward (the `[decode | prefill | glue]` wave) running a decode-derivative **`paged-glue`
kernel** that scatters recomputed marker KV into gap chunks reserved *in place* at their true logical
positions (`reserve_glue_gap_chunk`: `offset = 32 − n`, "full by construction", so
`rope_base == logical position`) and attends each marker **backward-unbounded + forward-windowed**
(`cpos ≤ row_pos + fwd_ahead`, the bridge into the following section). Sealed content is preserved;
only the reorder-dependent glue is recomputed. The codebase already calls the deferred version
"**ingest / compression gap-fill**" (`scheduler/prefill.rs:1842`).

**DECISION — DeepSeek glue is causal-only; drop the forward bridge.** The `fwd_ahead > 0` forward
window (a glue token peeking into the first `TURN_BRIDGE_FWD_AHEAD (16)` tokens of the turn it
introduces) is an asymmetric, semantically fuzzy special case whose quality effect is unmeasured. For
DeepSeek we set **`fwd_ahead ≡ 0`** — glue attends **backward-unbounded, strictly causal**. This is a
single-parameter change (`glue_bridge_window` → `0`, `projection_assembler.rs:690`); the kernel
already treats `fwd_ahead==0` as pure causal, so nothing else moves. Benefits: (a) removes the
forward "unknown"; (b) removes the dependency on the *following* turn being resident-in-place for the
bridge — glue needs only its backward (sealed-before + earlier-glue) context; (c) **cleaner
compression seam** — the boundary group's tokens are then projected under standard causal attention,
consistent with how every interior group was computed, so the §C fold has no asymmetric-context
caveat. Scope: the DeepSeek glue fork; the existing Qwen glue is left as-is unless we choose to drop
its bridge too.

**DECISION — thinking markers: DeepSeek always thinks, so emit no think/no-think markers.** This is
pure per-model **`Dialect`** config (`candle-transformers/src/models/dialect.rs`) — the main-engine
machinery is untouched; DeepSeek's dialect simply leaves the fields empty:
- **`no_think`** = the `/no_think` soft-switch text, emitted as a glue island on *suppressed*
  (prefilled, never-decoded) turns. `"/no_think\n"` for Qwen3/ChatML; **`""` for DeepSeek** → the
  `no_think` node tokenises to an empty run → **the `/no_think` glue island is empty, nothing is
  emitted.** That is the flag: an empty dialect `no_think` suppresses the marker *and* its glue with
  zero code change.
- **`think_block`** (`<think>`) is **no longer force-prefilled** — a thinking model emits its own
  `<think>` as its first *decoded* token, so for DeepSeek it is just part of the normally-decoded,
  normally-sealed assistant turn. The field is kept only as a BDP structural-noise seed (set it to
  DeepSeek's `<think>` string).
- **`no_think_block`** (the closed `<think>\n\n</think>` block) is descriptive-only / BDP-seed, never
  prefilled → **`""` for DeepSeek** (it never runs no-think).

Net effect on the glue set: DeepSeek drops the `/no_think` island entirely; only the **turn-framing
boundary glue** (`user_start`/`assistant_end`) remains as the seam (§E), and even that is now
causal-only. Implementation: add a DeepSeek `DialectType` (or reuse one) with `no_think=""`,
`no_think_block=""`, `think_block="<think>…"`, and the model's own role markers — no change to the
`no_think` node / glue machinery, which stays for Qwen3.

_Prior art: EXTEND — add a `DialectType` variant + `Dialect::deepseek()` preset (dialect.rs:12); pure data (Llama2/3 already ship the empty no-think fields, dialect.rs:194-224)._

**The identical problem exists for the compressed corpus — and it is *why* the merge belongs here.**
A compressed entry is `Compressor(x_t…)` — a pool of the group's tokens' hidden states `x`, and `x`
is **prefix-dependent** (it came from attention over the prefix). So:
- **Compressed groups wholly inside a sealed turn pair** → treated **invariant** (Arc-cloned, like
  sealed KV). This includes groups that span the turn's own **intra-turn** glue
  (`user_end`/`assistant_start`, `<think>`), because that glue is **left embedded in the turn's KV +
  substrate** (content-dominated, never regenerated). So these groups are **continuous and inserted
  as-is.**
- **Only the group that STRADDLES a reprojection slice/recombine point** (an **inter-turn** seam,
  where `user_start`/`assistant_end` glue regenerates) → **reorder-dependent, exactly like the
  boundary markers.** Its tokens' `x` changes when the prefix reorders, so its compressed entry is
  invalid after reassembly and **must be regenerated, not restored.**

**Scope optimization (leave intra-turn glue embedded).** The regeneration set is *not* every
group boundary — it is exactly the handful of **slice/recombine points** the reprojection cuts
between selected turns, the same points where inter-turn glue is already re-rendered. Everything
inside a turn pair (content + intra-turn glue) stays baked in the sealed KV/substrate, so its
compression groups are **maximally continuous and insert as-is** with no fold. This mirrors the
existing intra-turn(baked) / inter-turn(regenerated) marker split — the compression-merge scope
rides on top of it for free: `merge only where glue regenerates`.

This **corrects §C**: the monoid LSE fold is not a pure *persistence-time* arithmetic replay — on a
**reassembly** (reorder) the boundary group must be **re-folded against the new prefix**, and the
boundary tokens' fresh compressor projections come from the **glue re-prefill that is already
recomputing them**. So the fold **runs in the glue phase**. (On a plain resume with *no* reorder it
degenerates to the pure §C replay — same LSE operation, no re-prefill needed. So §C's monoid is still
the merge *operator*; §E is *where and when* it fires when the prefix changed.)

**One seam, done once, for both.** At the start of a turn the glue phase recombines, at the same
boundary:
1. the **dense seam** — re-prefill the boundary marker KV (existing behavior), and
2. the **compressed seam** — fold the last turn's persisted **partial tail `(m,l,acc)`** with the
   boundary tokens' freshly-recomputed compressor contributions (§C LSE), finalize `acc/l` →
   RMSNorm, and write the completed compressed entry into the corpus via a reserved gap.

Both write into in-place gaps at true logical positions, so the compressed-corpus seam is coherent
with the dense seam — the "excellent seam between the two."

**Insertion — all at the existing glue hooks (no new pipeline):**
- **Assembly-time** (`projection_assembler.rs::apply_segments_build` / `SegmentWalker`,
  `reserve_glue_island` / `glue_bridge_window`) — the one point that sees the reassembled adjacency:
  attach a **compression-merge descriptor** to the `GapFillPlan` alongside `glue_write_slice`/
  `fwd_ahead` (which boundary group, its persisted partial `(m,l,acc)`, and the boundary tokens that
  complete it).
- **Gap reservation** (`reserve_glue_gap_chunk` / `reserve_glue_gap`) — reserve the finalized
  compressed entry's slot the same full-by-construction way (preserving `rope_base == logical pos`,
  which the `apply_segments_build` convention assert enforces).
- **Descriptor staging** (`PendingGlue` / `GlueMeta`) — a per-boundary merge channel threaded flat to
  the kernel; zero new plumbing.
- **Kernel** (the DeepSeek **fork** of `paged_glue_kernel.cuh`): during the boundary re-prefill, run
  the compressor on the boundary tokens, fold with the incoming `(m,l,acc)` (LSE), finalize, scatter
  the compressed entry into its gap — piggybacking the kernel's dequant-once stream.
- **Wave fire** (`fire_gap_fill_batch` / the `[decode|prefill|glue]` wave) — co-batch the compression
  merge in the same launch as the glue, so it amortizes one forward.

**Consequence for persistence (ties back to §C):** you persist only the small trailing partial
`(m,l,acc)` per boundary group; you do **not** persist a "finalized" boundary entry, because it is
reorder-dependent and gets regenerated at the next reassembly. Complete interior groups persist as
invariant entries. The seam is never stored "baked" — it is always re-folded in glue against the
current order, exactly as the boundary markers are.

**Note:** the glue kernel must be **forked for DeepSeek** (single-latent K≡V + the compressor fold),
same as the decode/prefill forks — a `paged-deepseek` glue variant sharing the fork's substrate.
(Minor cleanup: `paged-glue/api.rs:22` has a stale `col_actual_pos` doc-comment; the live signature
derives position from `slice_rope`.)

## F. Embed the static system-prompt glue too (same principle, applied to the prompt)

The system prompt is a **`SectionTree`** — a **bounded** decision tree (`∏ option_count` variants
over its selector `dims`, a mixed-radix cross-product enumerable at build time). Its sealed sections
(`TreeOption.variants`) and collection members are **pre-sealed per ancestor-branch and Arc-injected
by GID reference** — the shared static prefix, reused across every projection target/layer, cached
hot/warm/cold in the substrate. A selector *switch* just picks a different pre-sealed variant; it
never re-prefills the nodes below.

**But `TreeGlue` is the one part not baked.** Every structural marker in the tree is emitted
`ProjectionSegment::Generated` (live-prefilled via a gap-fill forward) on **every** projection —
even glue that sits between two purely-static sealed nodes (`project.rs:1816-1830`;
`assemble_pieces` routes it into a `Glue` island → `reserve_glue_island` → gap-fill). Only the
`options`/collection-member variants are baked.

**Fix — classify tree glue into static-interior vs seam** (build-time, from node adjacency):
- **Static-interior glue** (between sealed nodes, no adjacent collection) → **bake it**: give it a
  `Vec<TreeVariant>`, one sealed per ancestor-branch (its prefix is the *same* bounded cross-product
  as the sections around it), and emit **`Sealed`** (Arc-injected) instead of `Generated`. Nodes
  below fold it into their `in_tree_prefix`. Still `∏ option_count` variants — **no explosion**. It
  joins the shared static prefix reused across all conversations/layers.
- **Seam glue** (the `<tools>`/`</tools>` markers wrapping a collection, and the collection's
  `member_glue` between selected members) → **stays `Generated`/live.** Its attention prefix depends
  on the belief-selected member subset (up to `2^N`, non-enumerable), so it cannot be baked. But
  there are only **O(#collection nodes)** such seams — a handful, one per `tools`-style block — **not
  per-permutation.**

Result: per-projection gap-fill fires only for the few **tree↔collection seams**; every other
structural marker in the system prompt drops in as-is with the Arc-injected sealed KV. This is the
exact system-prompt analogue of the turn-pair rule (§E): **embed static glue, regenerate only at the
dynamic seams** — and it stacks with DeepSeek being causal-only (§E) and carrying no thinking markers
(§E), so the system-prompt glue work per projection collapses to just the collection seams.

**Why collections themselves can't be baked (the permutation wall you named):** a collection's
surviving subset is one of up to `2^N` combinations that shift turn-to-turn with belief scores; the
codebase deliberately keeps collection members **out of the prefix chain**
(`SystemPromptSchema::is_collection_member`) so their variance never cascades into downstream sealed
KV. A `SectionTree` is cacheable precisely because its dims are a **finite enumerable radix set**;
its design *contains* the combinatorial collection inside a **prefix-transparent** node so the
variance can't leak into the sealed nodes below.

**Touch-points + caveat.** Add a build-time static-interior/seam flag on `TreeGlue`; seal
static-interior glue per branch (the build-time cross-product sealing AND the runtime
`add_section_to_collection` per-branch allocation, `builder.rs:753-791`, must stay in lockstep); emit
`Sealed` for it in the tree arm of `emit_system_prompt_items` (`project.rs:1803-1879`); keep seam
glue on the `Generated` path (`push_member_glue`). **Caveat:** the new sealed glue ids must be
counted by the `pack`/`ancestors` accounting and the `all_section_ids`/`is_collection_member` walks,
and nodes-below prefixes shift — so the two sealing sites move together. This is a **shared-engine**
improvement (it helps Qwen too), gated behind the same dialect/config; it is not DeepSeek-specific,
but DeepSeek benefits from it directly.

_Prior art: EXTEND — reuse the per-branch seal + `TreeVariant` (schema.rs:423, builder.rs:762, `TreeOption.variants` :409); give `TreeGlue` (:331) a `variants` vec routed through it. A sealed glue drops prefix-transparency (the intended static-interior semantic) — a new node mode over existing seal/inject primitives, not a rewrite._

---

# Part II — Implementation plan

## 1. `qtensor_from_ggml`: build the real GPU-repack path (not the byte-copy)

**Finding.** The GPU repack primitive already exists: `QCudaStorage::repack_ko` (`cuda.rs:3713`)
does dequantize-on-device → `run_quantize_ko`, and is used by `repack_to_host` (`cuda.rs:3916`, the
expert-staging path) and by `QMatMul::repack_for_optimization` (the loader's int8 path). The
`qtensor_from_ggml` KO arm I added only handles the case where the on-disk bytes are **already** KO
(offline-prepared, via `load_repacked` — a byte copy borrowed from the swap-file helpers). It cannot
take a standard `Q8_0` weight and produce KO.

**Plan.** Give `qtensor_from_ggml` a first-class GPU-repack path: for a compact quant source
(`Q8_0`, later the affine K-quants) whose target is a KO twin, upload the compact bytes and call
`repack_ko` on-device — the same code `repack_to_host` already runs, but building the `QTensor`
directly instead of round-tripping to host. Keep the byte-copy path only for genuinely pre-KO
tensors. Net effect: the engine can load a **standard `Q8_0` GGUF** and get int8-KO weights with no
offline CPU-quantize step and no dependency on a custom `Q8_KO` file — the "workaround" is gone and
the GPU kernel becomes the single source of truth for the KO layout.

**DECIDED — DeepSeek is offline-repack only; do NOT build the GPU-repack path for this model.**
GPU-repack-at-load reintroduces the per-tensor F32 transient (dequant → requant) that we moved
*offline* precisely because the CUDA pool retains freed transients and inflated resident to 11.2 GB.
The offline-prepare (Q8_KO in the GGUF) + byte-copy `load_repacked` path stays as the sole path for
DeepSeek. (The general GPU-repack capability could still be built for *other* models later, with a
reusable scratch buffer to bound the transient — but it is out of scope here.)

---

## 2. Drive host round-trips to zero (audit result)

Measured host readbacks per decode token ≈ **N + C + 1** (N = layers ≈ 43, C = CSA layers ≈ 20 →
~64 syncs/token). Every one except the sampler is eliminable, and all fold into the batched refactor:

| Op | Now | Fix |
|---|---|---|
| **MoE route dispatch** (`engine.rs` `to_vec2` + host counting-sort, ×N) | GPU→CPU readback of route indices then host sort | The indices are **already on-device** (`arg_sort_last_dim` in `Gate::route`). The batched path (`batched.rs`) already builds the dispatch on-device via **`moe_bucketize`** + `GpuDispatchTables`. Route through that — **−N syncs**. |
| **Indexer top-k** (`indexer.rs:183` `to_vec1` + host sort, ×C) | readback + host partial sort | Prefill's `build_mask` (`attention.rs:211`) **already does this top-k fully on-device** (`arg_sort_last_dim` + `narrow` + `scatter_add`). Reuse it in decode → return a **device index tensor** — **−C syncs**. |
| **KV gather** (`attention.rs:362` `Vec<Tensor>` + `stack`, ×N) + window ring `remove(0)` | host-orchestrated per-step gather | Disappears entirely once KV lives in arenas + the attention kernel reads it (§3–5). |
| **Sinkhorn** | ✅ already fused (shipped) | one kernel launch |
| **Sampler argmax→`to_scalar`** | 1/token | **UNAVOIDABLE** — the autoregressive sync point. |
| Embedding host lookup | 1/token, tiny | intentional (embed in RAM); not a readback. |

**Target: exactly 1 unavoidable readback per token** (the sampler).

---

## 3–9. The batched/paged hybrid attention — the big piece

### The two kinds of per-conversation state (and their correct homes)

DeepSeek decode keeps two buffers, both currently naive host `Vec<Tensor>` in `IncrementalAttention`:

1. **Linear sliding-window buffer** — the last `window_size` raw KV vectors (K≡V, one 512-d latent
   per token). Grows, then rings. **→ Arena KV cache**, exactly like the normal engine:
   `ChunkedKvBacking` per layer (shared pool via `new_layer`), one **slot per conversation**
   (`KvCache::set_chunked_backing(backing, batch_idx, policy)`), append per token, seal every 32
   tokens in the fixed **FP8 E4M3** window format (§H — no adaptive C-level; the C0–C9 candidate
   tables assume per-head separate K/V and do not apply to the single latent). DeepSeek is
   single-latent-KV → `n_kv_head
   = 1`, `head_dim = 512`, which splits cleanly into the **4 palette sub-bands** (128-d each) the
   arena/kernel already use. The sliding-window bound = a bounded slot (old chunks' `ChunkGid`s
   `Drop` → recycle to the pool) or the `RotatingKvCache` variant.

2. **Stateful compressed-entry corpus** — the pair `[G,512]` attended + `[G,128]` scoring, growing
   one row per completed group, attended via a **top-k selected subset** (§D). **→ a
   `FloatGalleryArena`** (Part I §D): the provenance corpus's tiering *skeleton*
   (`StreamId`-keyed residency, fingerprint-gated upload, free-list pool, LRU + rebuild-from-warm
   substrate blob backed by the redo log) with **float leaves**, plus **BDP sign-signatures stored
   alongside** for the recall scan. Selection is **two-stage**: BDP recall in the Indexer's sign
   space → Indexer float precision on the shortlist. The lifecycle is the wide-Q one (staged in-slot
   on `elevate_to_hot`, appended during `step`, drained at seal via a `gather_*`→`enqueue_*` pass),
   and the persisted unit is the position-free, monoid-mergeable form from §C. We reuse the corpus's
   *tiering* and *sign-scan*, NOT its case-vote *selector*.

   _Prior art: EXTEND — reuse `GalleryArena`'s tiering skeleton (free-list pool, StreamId residency, LRU, fingerprint upload; gallery_arena/mod.rs:216-416) as a sibling `FloatGalleryArena`; only the leaves are sign-bit (storage.rs, scan.rs) and a float warm/cold tier is new. Two-stage recall→rescore = NOVEL composition of existing halves (BDP scan.rs:82 + Indexer top-k indexer.rs:62); no pluggable selector interface exists today._

### The attention math, and how it splits (your point 9 — and it's the clean win)

The final softmax normalizes over `[window ‖ selected-compressed]` **jointly**, with per-head
learned **sinks** (a virtual zero-value key adding to each denominator). This decomposes exactly onto
the **split-KV combine** the paged kernels already have: compute two partial attentions, each
emitting `(Σw·V, running-max m, denom l)`, and merge with the existing **log-sum-exp combine
kernel**. So:

- **Window attention** = a paged-decode-style kernel over the arena KV: sequential slice-scan +
  **sliding-window bound** + **per-head sink**, single-latent K≡V.
- **Compressed attention** = a second kernel over the staged compressed entries: an
  **index-driven walk** over the indexer's top-k device index list, single-latent K≡V.
- **Combine** = the existing split-KV LSE combine → identical output to today's joint softmax.

This is the split you hoped for: **two attention functions sharing the combine**, not one monolith.
The window function is close to the stock paged kernel; only the compressed function is genuinely new.

### Fork vs flag (your points 6 & 7)

**A fork is required** — and there's precedent (`paged-glue/` is already a decode-derivative fork
with its own archive group). But it's a *thin* fork: the entire quantization/compute substrate is
**orthogonal to attention control flow and transfers verbatim**, which is what gets you "pal_map
compression + int8-MMU compatible + already tuned" for free:

- **Inherited unchanged:** `pal_map` per-head palette routing (`pal_iter.cuh`), `ArenaAccessor` +
  the 22-format dequant-in-kernel + the **INT8 read-through MMU** (`convert_all.cuh` —
  `is_int8_readthrough_format`, V read straight as int8, no FP round-trip), `mma_int8_m16n8k32`
  wrappers, the `SlotHeader`/`TokenSlice` 32-token paging + on-device `len` commit, the split-KV
  combine, and the launcher/grid scaffolding.
- **Free flags (no SMEM cost):** per-head **sinks** (fold into the `(m,l)` init), **sliding-window**
  bound (clamp the tile range).
- **What forces the fork (not flaggable):**
  1. **Single-latent K≡V** — the stock kernel assumes *separate* K and V arenas per head with the
     `SUB_HEAD_DIM==32` MMA geometry; DeepSeek stores one latent used as both K and V.
  2. **Top-k selected subset** — the stock tile-walk is a monotonic sequential slice-scan; a
     selected subset needs an **index-driven walk** over a GID list passed as a new kernel param.
  3. **Partial-range RoPE** — nope/rope head-dim split vs the stock full-vector RoPE.

SMEM is already at the wall (hd256 forced single-stage; prefill has a hard 25.6 KB arena assert), so
bolting the index-walk + latent-KV onto the production kernels behind flags would be a net loss —
a clean fork is correct.

### Batched engine (your point 3)

The attention kernels are inherently batched (grid.x = active slots; each block reads its own
`SlotHeader` from GPU memory, so ragged-depth sessions co-batch with no host raggedness). To use
them, `deepseek4` must implement **`BatchedModelCore` + `BatchedAttentionLayer`** (as `batched.rs`
does for Qwen3) rather than the single-session `latent_moe::Engine`. The MoE side already has the batched,
GPU-native dispatch (`ExpertCache` + `moe_bucketize` + `GpuDispatchTables`).

---

## Decisions (locked)
- **§1 repack policy:** DeepSeek is **offline-repack only — no GPU-repack at load.** The current
  offline-prepare (Q8_KO in the GGUF) + byte-copy load path stays. Do NOT wire `repack_ko` into the
  GGUF load path for this model.
- **Sequencing:** **Build toward the batched core directly.** Start with arena-KV integration + the
  forked decode kernel; the sync-elimination fixes land inside the batched core, not on the
  throwaway single-session prototype. `engine.rs` (`latent_moe::Engine`) is kept only as the **numeric
  golden reference** to validate the kernel against.
- **Compressed-entry store + selection (§C, §D):** a **`FloatGalleryArena`** — reuse the provenance
  corpus's *tiering skeleton* (StreamId residency, fingerprint upload, free-list pool, LRU +
  rebuild-from-warm) with **float leaves** (`[G,512]`+`[G,128]`), plus **BDP sign-signatures for
  recall**. Selection is **two-stage: BDP recall in the Indexer's sign space → Indexer float
  precision on the shortlist** — this removes both 1M bounds (cost + training) and unifies the
  engine's BDP (outer recall) with DeepSeek's Indexer (inner precision) into one hierarchy. We do
  **not** reuse the BDP *case-vote selector* (it returns label votes, not per-entry top-k). Persist
  the position-free, monoid-mergeable form (pre-RoPE window + corpus pair + trailing `(m,l,acc)`);
  RoPE + position live in the kernel. **To validate:** BDP recall (sign top-M ⊇ Indexer top-k),
  swept per layer.
- **Latent KV:** confirmed — one **512-d latent used as both K and V** (K≡V, `n_kv_head=1`),
  split into 4 palette sub-bands of **128-d** each (not the stock 32).
- **First kernel:** **full-hybrid decode in one shot** (window + compressed top-k + sink), developed
  against a standalone correctness test vs `engine.rs::sink_attend` before wiring live data.

## Build order (revised)
1. **`paged-deepseek/` decode kernel** (the long pole): fork `int8_decode_kernel.cuh`, single-latent
   K≡V `KvHead` variant, hybrid loop (sequential window + index-driven compressed top-k), per-head
   sink, windowed RoPE, reusing arena_table/convert/mma/slot_types + split-KV combine. Rust op
   wrapper mirrors `prefill_utils.rs`. Develop against a **standalone test** with synthetic arena
   KV + selection, checked vs the CPU reference.
2. **Arena-KV window integration** — `ChunkedKvBacking` per layer, single-latent 512-d/4-palette
   layout, slot per conversation.
   _Prior art: REUSE (zero new cache code) — `KvCache::set_chunked_backing` (cache.rs:1077), `append` (:1139), `chunked_write_kv` (:1094) via `BatchedInferenceSession`._
3. **`FloatGalleryArena` + two-stage selection (§C, §D)** — float-leaf tiering skeleton for the
   `[G,512]`+`[G,128]` corpus; BDP sign-index for recall; Indexer float re-score on the shortlist.
   Persist the position-free, monoid-mergeable unit (§C: pre-RoPE window + corpus pair + trailing
   `(m,l,acc)`; RoPE/position in the kernel; turn-boundary partials fold via LSE). Validate BDP
   recall (sign top-M ⊇ Indexer top-k) per layer before trusting it as the primary recall stage.
4. **`paged-deepseek/` prefill kernel** — one shot.
5. **`paged-deepseek/` GLUE kernel + compression-seam merge (§E)** — fork `paged_glue_kernel.cuh`
   for single-latent K≡V, **causal-only** (`fwd_ahead ≡ 0` via `glue_bridge_window → 0`, so the fork
   drops the forward-window path entirely), and add the boundary compressed-group fold: attach a
   compression-merge descriptor on the `GapFillPlan` (`projection_assembler.rs`), reserve the
   finalized entry's gap, thread it via `PendingGlue`/`GlueMeta`, and in the forked glue kernel run
   the compressor on the boundary tokens + LSE-fold with the persisted `(m,l,acc)` + scatter the
   entry. Co-batched in the `[decode|prefill|glue]` wave. (Also: drop the stale `col_actual_pos`
   doc-comment in `paged-glue/api.rs:22`.)
6. **`BatchedModelCore` for DeepSeek** — assemble the batched engine.
   _Prior art: REUSE — `BatchedModelCore` (batched_model.rs:98) + `BatchedAttentionLayer` (batched_layer.rs:232); `latent_moe/batched.rs:761` (Qwen3) is the template._

---

# Part III — `paged-deepseek` kernel design

Grounded in the measured budgets of the stock INT8 kernels (48 KiB decode static cap; 25.6 KB
prefill union arena at 4 blocks/SM; 64-reg `__launch_bounds__` → 67% occupancy on Ada; `-maxrregcount`
96 decode/glue, 128 prefill; `mma_int8_m16n8k32` M=16/N=8/K=32; split-KV "~2 waves at 3 blocks/SM").
Shape assumptions: **HEAD_DIM=512, K≡V single latent, `n_kv_head=1` (MQA), `H` query heads, per-query
attended keys `K ≈ window_size + index_topk ≈ 640`** (bounded — the O(1) budget).

## G.1 Reused verbatim (the substrate — the reason a fork is cheap)
`arena_table.cuh` (`ArenaFormat`, `ArenaAccessor`), `convert/convert_all.cuh`
(`load_head_scaled`, `is_int8_readthrough_format`, `load_head_int8_readthrough`), `mma/mma_wrappers.cuh`
(`mma_int8_m16n8k32`), `paged-decode/slot_types.cuh` (`SlotHeader`/`TokenSlice`/`KvHead`, on-device
`len` commit), `fast_exp` (base-2 online softmax), the split-KV **combine** kernels, the persistent
partial pool, and the launcher grid/split heuristics. This buys "pal_map compression + INT8-MMU +
already-tuned" for free.

_Prior art: int8-KO forward + MoE dispatch reused verbatim (`forward_via_int8` mod.rs:2361; ExpertCache/`moe_bucketize` batched.rs:397). Partial-range RoPE + per-head sinks exist HOST-side (attention.rs:152 `rope_last`, :229 `sink_attend`) but NOT as kernels — no paged kernel has a nope/rope window or a sink scalar (the `Q4_KS`/`Q8_KS` "attention-sink formats" are an unrelated block-quant naming collision); both are new kernel work here._

## G.2 The four structural blockers at HEAD_DIM=512 / SUB=128 (fork's new constants)
Hard (compile-fail or overflow), not tuning:
1. `static_assert(VEC <= 8, "HEAD_DIM <= 256")` (`int8_decode_kernel.cuh:62`) → VEC=16 at HD=512. The
   fork templates `VEC` up to 16 (or head-dim-tiles), auditing every `float regs[VEC]`.
2. `static_assert(SUB >= 16 && SUB <= 64)` prefill rank pack `(palette<<6)|rank` — SUB=128 rank
   reaches 127 → needs **7 bits**. Repack `(palette<<7)|rank` (u16 table) OR tile the palette to 32.
3. `static_assert(SUB_HEAD_DIM == 32)` batched-M MMA — SUB=128 = **4× m16n8k32 K-steps** (K=32 each,
   accumulate the 4 into one C). The tensor-core math is unchanged; only the fragment loop grows ×4.
4. `PalIter::scatter` is `uint8_t[VEC]` (`pal_iter.cuh:33`) — index `3*128+127=511` overflows. Widen
   to `uint16_t`.
Plus: the 25.6 KB **prefill arena only closes at HD=128** (§G.5).

## G.3 The K≡V single-latent leverage (the design's spine)
K and V are the **same** (partially-roped) 512-d latent. Per key: **load once → RoPE once
(partial nope‖rope) → requant int8 for QK → reuse the SAME staged FP latent for the PV accumulate.**
This halves KV memory traffic vs the stock separate-K/V (which cp.async-loads K then V again), and
collapses two smem staging regions into one — the headroom that makes HD=512 fit. (Exactly the
`skt`-reuse the bmma PV pass and the glue dequant-once already do.)

## G.4 Decode kernel — grid, SMEM, occupancy
**Grid** = `dim3(num_slots, head_tiles, num_splits)`; `head_tiles = ceil(H / HEADS_PER_BLOCK)`,
`HEADS_PER_BLOCK = 16` (one MMA M-tile). Block = `dim3(256)` = 8 warps, owning
`(slot, 16-head M-tile, KV-split)`; it reads the slot's bounded KV (window ∪ selected-compressed),
redundant across head-tiles but **L2-resident** (~640×512×1 B ≈ 320 KB/slot). Split-KV over total `K`
reuses the "~2 waves at 3 blocks/SM" factor; the stock LSE combine merges. (For small `H` where
`head_tiles==1` this degenerates to the batched-M layout.)

**SMEM (per block, single-stage; K≡V halves the need for double-buffer):**
```
Q-tile int8   [16][512]      =  8192 B   (this M-tile's heads, quantized once)
scaleQ        [16][4] f32    =   256 B
latent FP     [8][512] T     =  8192 B   (8-key tile; RoPE + PV reuse — the K=V share)
latent int8   [8][512]       =  4096 B   (requant for QK)
scaleK        [8][4] f32     =   128 B
scores        [16][8] f32    =   512 B
                              ---------
                    total  ~= 21.4 KB  -> fits 48 KiB; <=25.5 KB => 4 blocks/SM (67%)
```
**out accumulator** `out_reg[16 heads × 512]` = 8192 floats / 256 threads = **32 F32/thread** —
identical to the prefill dim-half `o_acc` budget that already closes at 64 regs. Target
`__launch_bounds__(256, 4)` → 64 regs → **67% occupancy**, matching stock decode. (Optional
double-buffer of the 8-key latent = +8 KB → 3 blocks/SM; skip it — occupancy hides the single load.)

**QK:** M=16 heads, N=8 keys, K=512 = **4 palettes × 4 k-steps** of `mma_int8_m16n8k32`; scaled-acc
`acc += c · scaleQ[h][p] · scaleK[key][p]`. **PV:** FP `fmaf` over the reused staged latent (or int8
read-through when all palettes are read-through formats — one arena, so the gate is per-slot cheap).
**Sink:** fold per head at the end — `m'=max(m,sink[h]); l = l·e^{m−m'} + e^{sink−m'}`.
**RoPE:** read from a **factored cos/sin table** (`rope_lookup`, `latent_common.cuh`) — neither the
flat position-indexed `rope_cs` table (`int8_decode_kernel.cuh:56`; hundreds of MB growing with N,
§L) nor per-key in-kernel trig (f64 runs at 1/64 rate on consumer Blackwell, and every key needs
`rope_head_dim/2` angles per tile). The position splits at bit 10 — `pos = hi·2¹⁰ + lo` — and the
angle-addition identity `sin θ = s_hi c_lo + c_hi s_lo`, `cos θ = c_hi c_lo − s_hi s_lo` recombines
two tiny blocks: (sin, cos) of `(hi·2¹⁰)·f` and of `lo·f`, `(2048+1024)×NF` float2 ≈ **768 KB per
frequency set** — L2-resident, covering every position below 2M (context hard-cap 1M) with size
independent of N. The table is built **once per frequency set at model load**
(`latent_rope_table_kernel` → `build_rope_table`) using the fork's exact trig primitives: the angle
`pos·freq` reduced to quadrant + residual **in f64** (an f32 product is unusable at depth:
ulp(10⁶ rad) ≈ 0.06 rad, and `__sincosf` degrades outside the principal range), sin/cos from
plain-arithmetic minimax polynomials (`rope_angle`/`ds_sincos`, `_rn` intrinsics so the compiler
cannot contract them). The runtime combination is 2 float2 loads + 6 exact-rounded f32 ops per
(key, pair) — accurate at any position AND reproduced bit-for-bit by the CPU mirror oracle
(`table_sincos`), gated by `rope_table_device_matches_mirror` + the bit-exact mirror suite.

**Hybrid loop (one online-softmax, two key sources):**
- **Window pass** — the stock gap-aware sequential slice-scan over the slot's window chunks, bounded
  to the last `window_size` (a `tile_lo` clamp), strictly causal (§E: no forward bridge).
- **Compressed pass** — an **index-driven walk** over the Indexer's top-k selected GIDs (device index
  tensor from the two-stage BDP-recall→Indexer-precision, §D), gathering each selected 512-d latent
  from the `FloatGalleryArena`, into the *same* `(m_i, l_i, out_reg)`. The index tensor is produced
  **on-device each token** by the Indexer re-score over the resident shortlist; BDP recall refreshes
  that shortlist asynchronously (§D cadence) — never on the decode hot path.
Both are key streams feeding one accumulator; the split-KV `num_splits` spans their union.

## G.5 Prefill kernel — per-query decode geometry over a settled slot
_Revised at implementation (the palette-tiled-arena plan below it replaced assumed the stock
adaptive-format staging; the fixed-FP8 direct-load fork obviates it)._ The implemented prefill is
the **decode step itself, run once per prompt token inside the wave**. The wave's real batching
win is the MoE (one grouped call + one routing readback per layer per wave, amortized over every
row); attention absorbs the prompt through `kernel_attn_decode_step` — the SAME launch the decode
rows use: in-kernel FP8 writer scatter, push→select→attend corpus order, auto split factor.
Mechanism: `build_decode_metadata_at(seqs, generation, offset_overrides)` pre-builds one header
snapshot per prompt token at wave entry (token `t` serialized at offset `base+t`, all layers;
non-prefill slots are frozen for the wave, so snapshot 0 serves the decode and glue rows). This is
**bit-identical to per-token stepping by construction** — same kernel, same launch geometry, same
accumulation order. The multi-query `deepseek_prefill_kernel` remains the glue-row path (arbitrary per-query positions
over a settled slot) but it is **NOT the prompt-absorption path** — and the reason is a genuine
semantic fork, not a reassociation. The decode kernel **bf16-stages the current token's PV** (the
query's own key contributes to the value sum at bf16 precision), while a settled-slot prefill reads
every key — its own diagonal included — from the FP8 arena. On real bf16 activations the FP8
diagonal differs from decode's bf16 diagonal; early SWA layers weight self-attention heavily
(≈0.9), so FP8's ~12% mantissa error on the diagonal value shifts the layer output ~11%, which
flips a downstream CSA top-k selection — a discrete cliff, not smooth drift — and 43 layers compound
it into an argmax flip (measured: garbage output, `" The\n * GNU General Public License…"` instead
of `Paris`). **No single-launch batched prefill can bit-match per-token decode**, because decode
transitions each key from bf16 (diagonal at step _t_) to FP8 (window key at step _t+1_), while a
batched pass has one representation per key. The correct fast batched prefill is therefore a
**kernel change, not a host reshuffle**: give the prefill kernel a `[s, HEAD_DIM]` bf16 `diag_src`
and, when a window key's `key_pos == q_pos[qi]` (the diagonal), read it from `diag_src[qi]`
bf16-staged (int8-from-roped-f32 for QK, matching decode) instead of the arena FP8 — window keys
`[t-w, t-1]` still come from the arena. That reproduces the decode walk exactly and is gated by
`prefill_rows_equal_decode_steps` extended to compare against a per-token decode reference on
**real** (not fp8-exact synthetic) activations. Until that kernel lands, prompt absorption uses the
per-token path above (correct, ~5 tok/s), which is sufficient for the step-6 recall gate at
multi-thousand-token depth. **Residual batched-float variance** (independent of the above): the
wave's row-parallel non-attention ops (mHC projections, MoE gate/experts) reduce in shape-dependent
order, so a multi-row wave's hidden states carry a few-ulp drift vs strict per-token stepping — the
same variance multi-session co-batched decode waves already carry by design. With attention
absorption order-exact (per-token), the residual is ≤1 fp8-ulp window drift and a ~2.7 max logit gap
with IDENTICAL argmax. The audit gate (`wave_prefill_state_matches_decode_steps`) therefore asserts
**next-step argmax equality** — the property the product depends on — not bitwise state equality.

## G.6 Glue-recombine kernel — attention seam + compression fold in one launch
Fork `paged_glue_kernel.cuh` (decode-derivative, **causal-only** — `fwd_ahead≡0`, drop the
forward-window path), single-latent K≡V. Register-resident flash-state (glue is occupancy-bound); at
HD=512 the `q_reg`+`o_reg` per glue token is large, so **`GLUE_G_TILE=1–2`**, O distributed across
threads (32 F32/thread), glue runs fanned across `gridDim.z`.

**+ compression fold (§C/§E) — the LSE monoid operator is IMPLEMENTED host-side**
(`compressor.rs::GroupPartial`: `identity` / `fold(scores,kvs)` / `merge` / `finalize`, proven by
`group_partial_seam_fold_matches_whole` — a group's pooling rows cut at every interior seam and
re-merged equal the single-shot softmax pool, and the identity law holds). The glue kernel transcribes
this same per-channel recurrence in-register. The glue forward already recomputes the boundary
tokens' hidden states `x`. Add a pass that runs the compressor projections (`wkv`/`wgate` →
per-channel `kv`, `s = score + ape[within-group offset]`) on those tokens and **LSE-folds them per
channel** with the incoming persisted partial `(m,l,acc)`:
```
per channel c:  m' = max(m, s_c);  l = l*e^(m-m') + e^(s_c-m');  acc = acc*e^(m-m') + e^(s_c-m')*kv_c
finalize on group completion:  entry_c = acc_c / l_c  -> RMSNorm   (pre-RoPE; RoPE at read)
```
A per-channel reduction over `<2·ratio` boundary tokens + the partial — a handful of registers (the
512-ch `(m,l,acc)` triple distributes 2 f32/thread over 256 threads). Scatter the finalized 512-d
entry into its reserved gap (like the glue KV scatter). Descriptors ride `PendingGlue`/`GlueMeta`;
co-batched in the `[decode|prefill|glue]` wave — one launch does both seams.

_Prior art: the `(m,l,acc)` fold reuses the split-KV combine LSE monoid — `int8_decode_combine_kernel` (int8_decode_kernel.cuh:1522-1530), tile-fold at :663-671. No host-side LSE merge exists; transcribe :663-671 for the glue/host fold._

## G.7 Occupancy / budget summary
| kernel | grid | block | SMEM | regs / launch_bounds | blocks/SM | occ |
|---|---|---|---|---|---|---|
| decode | (slots, head_tiles, splits) | 256 | ~21 KB single-stage | 64 / (256,4) | 4 | 67% |
| prefill (A) | (q_tiles, batch·splits) palette-loop | 256 | ≤25.6 KB palette-tiled | 64 / (256,4) | 4 | 67% |
| glue+fold | (slots, kv_head=1, glue_tiles) | 256 | ~12 KB + reg flash-state | 96 max, no LB | occ-bound | high |

**Net:** the tensor-core math + the whole quant/paging substrate are reused; the fork's real work is
(1) the 512/128 constant re-derivation (§G.2), (2) the K≡V single-load restructuring (§G.3), (3) the
hybrid two-source loop (§G.4), (4) the palette-tiled prefill arena (§G.5), (5) the glue compression
fold (§G.6). Decode first (validate vs `engine.rs::sink_attend` on synthetic KV+selection), then
prefill, then glue — each one-shot to amortize the slow compiles.

---

# Part IV — Test-driven implementation & migration plan

The whole port advances behind **one invariant: the model still answers "Paris."** Every step below
is gated on that answer surviving, so a regression is caught at the step that caused it, not three
kernels later. This part defines the golden ladder, reconciles the compression harness (which no
longer matches this model), and gives each build-order step its concrete test + gate.

## H. The compression harness no longer matches this model (the C0–C10 mismatch)

The Qwen harness measures KV compression with `InferenceMode` C0–C10 (batched_inference.rs:57) and
`compression_ratio_by_sequences` → `16.0 / bpe` (batched_inference.rs:2611). Both assume Qwen's KV
shape: **per-head 128-d K and V, format-selected per 32-token block** against cosine thresholds. None
of that is true for DeepSeek:

- DeepSeek KV is **one 512-d latent, K≡V, MQA `n_kv_head=1`** (§A, §G.3), split into 4×128-d palette
  sub-bands. There is no per-32 K/V format pair to select, so the C0–C10 candidate tables are
  meaningless here.
- The *real* memory win is not per-block bit savings — it is that each query attends to a **bounded
  `window + top-k ≈ 640` keys** (§B, Part III) instead of the full O(N) history, plus a **pooled
  compressed corpus** (§C) instead of one KV entry per token.

**Decision — window-KV format (REVISED): FP8 E4M3 writer + adaptive per-BAND compression on sealed
chunks.** The WRITER chunk stores the latent as FP8 E4M3 (`F8E4M3 = 34`; the fused/glue scatters
write FP8 only). At seal time, the **latent band compressor**
(`candle-kernels/src/quantize/latent_band_quant.cuh` + `quantize_sealed_latent` in
`kv_cache/chunked/compress.rs`) selects ONE format per (chunk, 128-dim band) from the level's K
candidate ladder — whole-band round-trip rel-L2 against the level threshold, first passing wins,
no-pass preserves FP8 — and re-encodes the band as token-oriented GGML blocks. The paged-latent
kernels dispatch per band on the KvHead format tags (`load_band_elem`), so mixed FP8/quant chunks
read directly. This SUPERSEDES the earlier "one fixed format, no adaptive C-level" decision: the
original objection (the C0–C10 tables assume Qwen's per-head K/V shape) was answered by selecting at
the latent's own granularity — the four per-band KvHead slots — rather than retrofitting the
palette4 machinery (whose per-32-dim selection is structurally wrong for this shape, 16 sub-palette
picks vs 4 band slots). Q8_KS's fine-scaled sub-block (elements 0–3) maps to the chunk's first four
TOKENS in the token-oriented layout — the attention-sink protection carries over intact. Byte-wise,
Q8 ≈ FP8 (quality rung, not size); the size wins start at Q4 (≈1.78×) and grow down-ladder, selected
only where the threshold proves them. Gates: `latent_band_select_convert_matches_codec` (GPU/CPU
codec byte contract), `mirror_bit_exact_mixed_band_formats` (decode dispatch, bit-exact),
`prefill_mixed_rows_equal_decode_steps` (prefill dispatch), `latent_compressed_arena_roundtrip_decode`
(the full production seal→compress→decode chain). The in-session seal
(`quantize_and_seal_sequences`) drives compression; the persistence thread's hot→warm drain moves
single-latent chunks format-preserving (the latent path is immediate-only — no deferred-descs
contract). The compressed-corpus entries stay **float** (`FloatGalleryArena` `[G,512]`+`[G,128]`,
§C/§D) — their job is retrieval, not compression.

**Decision — the ratio we report:** re-baseline against *traditional FP16 linear attention*, as
requested. Add `fp16_linear_baseline_bytes(n_tokens) = n_tokens · head_dim · 2 (K and V) · 2 B
(FP16) · n_layers` (what a traditional dense-attention FP16 model would hold — separate full-history
K and V, every token, every layer) and `deepseek_kv_footprint()` = FP8 window + float corpus pair +
`(m,l,acc)` tails, summed **per layer kind** (SWA layers hold only the window; HCA/CSA add the
corpus pair — a flat per-layer count over-states the footprint by the SWA fraction). Report
`baseline / actual`. This is a
**system-level** ratio (bounded attention × FP8 × corpus pooling), not the per-chunk `16/bpe`, and it
is where the number becomes dramatic: at 1M tokens dense FP16 is hundreds of GB; DeepSeek holds a
bounded window + a pooled corpus.

- **Test:** raw-byte unit test on both accounting functions for a synthetic `N` (assert exact byte
  counts, not a tolerance — CLAUDE.md TDD rule for codec/accounting code), plus a `ratio vs FP16
  linear` line printed alongside the Paris/RULER stats.
- **Gate:** host forward is untouched, so the streaming/engine Paris goldens are unaffected — this
  step lands independent of the kernel.

## I. The golden ladder (four rungs, each gates the next)

1. **CPU host reference** — `attention.rs::sink_attend` / `rope_last`, `streaming.rs::forward`.
   Already green (17 codec/forward tests + Paris via block-streaming). **Frozen** — never edited to
   make a kernel pass; the kernel conforms to it.
2. **Kernel vs CPU reference** — standalone, **synthetic arena KV + selection, no substrate, no
   model**, with **two oracles** (the `moe_bucketize` precedent): **(a) arithmetic mirror** — a CPU
   reference that applies the kernel's exact numerics (the FP8 round-trip on window latents, the
   int8 requant for QK), gated **bit-exact**; **(b) model quality** — `sink_attend` in float, gated
   at tolerance. (The window is the FP8-stored path and the corpus stays float; naive bit-exactness
   against float `sink_attend` is impossible by construction once QK requants to int8 — a sloppy
   oracle here costs days.)
   _Achieved (decode): 9/9 green, bit-exact **including live RoPE**. Three hard-won mirror-contract
   facts, load-bearing for the prefill/glue forks too: the SM80+ `fast_exp` runs its PTX-asm
   variant with different FMA-tuned coefficients (mirror THAT); all fork trig is own-polynomial
   `_rn`-intrinsic code (`rope_angle`/`ds_sincos` — `__sincosf` is neither mirrorable nor accurate
   at depth; the attention path reads them through the factored table + `_rn` angle-addition,
   mirrored by `table_sincos`); and the kernel quantizes int8 from roped f32 registers while PV reads the bf16-staged
   latent — the mirror keeps both (zero-rope tests cannot catch this: fp8 ⊂ bf16). Device-vs-mirror
   bit probes for `ds_exp`/`ds_sincos` + a nullable kernel stage-dump are permanent regression
   infrastructure (`latent_moe/paged.rs`)._
3. **Engine vs CPU reference** — `engine_generate_paris_fast` (engine.rs:532) still answers Paris
   after each kernel is swapped into `latent_moe::Engine`. The engine is kept solely as this rung's harness
   (Decisions-locked: numeric golden, not a product path).
4. **Conversation-engine vs engine** — same Paris after migration into `BatchedModelCore` + a
   (temp-file) substrate, plus persist→reboot→resume exactness.

**Standalone-first rule (your ask):** build every attention function against synthetic inputs at
rung 2 *before* any live data. Substrate enters only at rung 4. This is why the temp-substrate helper
(§J) is the **last** piece of test infra, not the first.

## J. Substrate hosting for tests (temp-file, Zend-less)

`ConversationBuilder::workspace_path(dir)` roots persistence at `dir/.substrate/` (builder.rs:71).
For tests, point it at a `TempDir` under the scratchpad: a `deepseek_test_conversation(tmp)` helper
builds a no-daemon `Conversation` with a real-or-tiny model (gated on file presence, like the existing
`#[ignore]` goldens) and a throwaway substrate. Persistence stays mandatory (CLAUDE.md) — it is simply
rooted in a directory dropped at test end, so we get full persist/reboot/resume coverage without Zend.
Only rung 4 uses it; rungs 1–3 never construct a `Conversation` at all (they are kernel/engine tests
below candle-conversation). **Never pass `workspace_path = None` in a test:** `None` does not disable
persistence — the engine falls back to the *current directory* (engine.rs:262-265) and would silently
write `.substrate/` into the repo checkout. Rung 4 always passes an explicit `TempDir`.

## K. Per-step plan — implement / test / gate

Each step keeps the prior rung green as its gate; that green rung is the **next step's baseline**.

| # | Implement (build order) | Test (rung) | Gate |
|---|---|---|---|
| 0 | FP8 window wiring (`Float(F8E4M3)`, §H) + `fp16_linear_baseline` / `deepseek_kv_footprint`; one-line comment fixes (`arena_table.cuh:16` header says "3=F8E4M3" but the constant is 34; `paged-glue/api.rs:22` stale `col_actual_pos`) | raw-byte footprint units + ratio line | streaming Paris unaffected |
| 1 | `paged-deepseek` **decode** kernel (BO §1) | rung 2, both §I oracles: arithmetic-mirror **bit-exact**; `sink_attend` at tolerance | rung 2 green **before any wiring** |
| 2 | Arena-KV window: single-latent 512-d/4-palette `ChunkedKvBacking` (BO §2) | write→read round-trip **raw bytes**; kernel over arena == kernel over synthetic | rung 2 still green through the arena |
| 3 | `FloatGalleryArena` + two-stage select (BO §3) | (a) **BDP recall ⊇ Indexer top-k**, swept per layer (§D validation); (b) corpus round-trip raw bytes; (c) two-stage == full Indexer top-k on synthetic gallery | rung 2 corpus path matches `sink_attend` (oracle b) over the same selection |
| 4 | `paged-deepseek` **prefill** kernel (BO §4) | rung 2: prefill vs `streaming.rs::forward` on synthetic prompt → then **rung 3** | **`engine_generate_paris_fast` green on the real kernel path** (host attention gone) |
| 5 | `BatchedModelCore` + temp-substrate migration (BO §6), incl. the §2 sync eliminations (`moe_bucketize` routing −N, on-device Indexer top-k −C) | **rung 4**: persist→reboot→resume == pre-reboot KV (monoid boundary-merge, §C); same Paris; instrumented **readbacks/token == 1** (the sampler) | rung 4 Paris + resume-exact + one-readback assert |
|   | _Implemented as `ManagedBatchedModel` directly (`latent_moe/wave.rs` — the scheduler binds `Box<dyn ManagedBatchedModel>`, so the mHC loop stays private; zero scheduler changes). Readback accounting, measured and asserted (`wave_paris`): the Indexer top-k and prefill selections are fully on-device (−C ✓); MoE routing is ONE amortized readback per layer per **wave** — intrinsic to the **streaming** `ExpertCache` (host must see expert ids to schedule pinned→VRAM uploads; §2's zero-readback dispatch engages exactly when the expert set is fully resident, as on Qwen). Budget = 1 sampler readback/token + that documented routing set, nothing else._ | | |
| 6 | Dynamic corpus/RoPE budgets to 1M (§L) | NIAH needle recall via `ruler_gen` (generic over `ManagedBatchedModel` — needs the step-5 core) + footprint **flat** as N grows | Paris green; footprint-flat + needle recall hold |
| 7 | `paged-deepseek` **glue** kernel + compression-seam merge (BO §5, §E) | rung 2: glue-recombine vs host LSE fold (raw `(m,l,acc)`) → two-turn seam-straddling group reconstructs == single-shot forward of the concatenation | rung 4 Paris across a 2-turn seam |
| 8 | **Intra-turn glue embedding (LAST)** (§E optimization) | equivalence: embedded-intra-glue reconstruction == fully-regenerated-glue reconstruction | no Paris/coherence regression |

Three ordering notes: the **batched core (step 5) precedes the 1M work (step 6)** because the NIAH
harness (`ruler_gen::sweep_parallelism`) is generic over `ManagedBatchedModel` — there is nothing to
drive it with until the core exists; the **glue kernel (step 7) lands after** the batched core +
substrate exist, because its payoff (seam reconstruction) is only observable across a persisted
multi-turn conversation; and the **intra-turn glue optimization (step 8) is dead last**, an
equivalence-gated change over an already-green stack so it cannot destabilize an earlier gate. Steps
0–4 need no substrate at all — the first product-level milestone is **step 4: Paris on kernels with
zero host attention.**

## L. Window budget — dynamic sizing toward 1M

Today `window_size = 128` (config.rs:82) and `max_seq = 512` (engine.rs:309, a prototype RoPE-cache
constant). Scaling to 1M does **not** grow the per-query attended set — that stays bounded at
`window + top-k ≈ 640` (the O(1) budget, and the entire premise). What must become dynamic:

- **RoPE to 1M — a factored table with N-independent size (IMPLEMENTED).** The inherited kernels
  index a position-sized `rope_cs` table (`int8_decode_kernel.cuh:56`); at 1M positions that table
  is hundreds of MB of device memory growing with N — a quiet violation of the fixed-budget
  principle. The `paged-latent` fork splits the position at bit 10 and recombines two tiny cos/sin
  blocks via the angle-addition identity (`rope_lookup`, §G.4): ≈768 KB per frequency set covering
  all positions below 2M, size **independent of N** — the fixed-budget principle holds because the
  factorization is logarithmic in extent, not because the table was banished. Entries are built once
  at load by `latent_rope_table_kernel` with the exact f64-reduction + minimax-polynomial trig, so
  the per-key cost drops from an f64 reduction (1/64 rate on consumer Blackwell) to 2 cache-hot
  float2 loads + 6 f32 ops, bit-mirrorable as before. The host `RotaryCache` (`latent_moe/rope.rs`)
  is **table-free**: it holds only the `rope_head_dim/2` `yarn_freqs` and computes `cos`/`sin` per
  call for exactly the positions requested (`apply(start, seq)` / `apply_positions(&[u32])`), same
  `f64`-angle math (the `max_seq` constructor argument is gone — 15 call sites updated). A query at
  position 10⁶ costs the same memory as one at position 10; the indexer/compressor projections and
  reference forwards are all position-unbounded.
- **Corpus tier budget (IMPLEMENTED).** The `FloatGallery` splits its storage by access pattern:
  the sign/pos **index** (`signs` [G, sign_words] u32 + `pos` [G] u32, `sign_words·4 + 4` B/entry)
  stays GPU-resident at any depth — the BDP recall scan reads all of it, and it is tiny (≈105 MB at
  1M tokens across the compression layers). The **float pair** (`attn` [G, head_dim] + `keys` [G,
  index_head_dim], the bulk) spills to CPU RAM past `HOT_ENTRY_CAP` entries; per query, only the
  bounded shortlist (`gather_keys`, ≤`top_m`≤1024 rows) and the selection (`gather_selected`,
  `top_k` rows) are gathered back to the GPU, and the kernel walks that compacted pair densely. The
  resident GPU footprint is therefore the index alone — bounded — while the corpus grows in RAM;
  `spilled_corpus_selects_exactly` proves selection stays bit-exact vs the full-residency oracle
  across the spill boundary. Below the threshold a gallery is fully hot, so short conversations pay
  nothing.
- **Window itself stays 128.** "Extend the window beyond current budgets" resolves to *extend the
  addressable context and corpus*, while the attended window and top-k remain fixed — that bound is
  what keeps error O(1) at 1M.

The **step-6 footprint-flat assertion is exactly this test**: drive N large, assert the attended-set
and hot footprint stay bounded while corpus tiers spill, and a needle planted at depth is still
recalled (BDP→Indexer). If footprint tracks N, the bound is broken; if it plateaus, 1M holds.
</content>
