# Progressive YaRN — a per-slot RoPE schedule for every model

**Status:** design, not built, with no open questions. §11 is the build order.
§12 records how the last one was settled: the QSA indexer takes YaRN and `m`
exactly as attention does.

The engine stores K un-rotated and rotates it inside the attention kernel,
using a position it derives from each chunk's `rope_base`. That one convention
means a model's RoPE frequencies are a property of the **read**, not of the
stored cache. So a slot can switch to a different frequency set without
recomputing a single cached byte.

Every other serving stack stores K already rotated. They must therefore fix a
YaRN factor for the whole server. Qwen warns that this fixed factor hurts short
contexts. Qwen's own advice is to turn YaRN on only when a long context is
actually needed.

We can follow that advice per slot, which no other stack can:

- A slot inside the trained window runs exactly the RoPE the model was trained
  with.
- A slot that outgrows the window moves up a rung, to the YaRN factor Qwen
  publishes for that length.

This document specifies five things:

1. The rungs for every model we ship (§2).
2. What the audits of the existing code found: fixed scaling, `mscale`, every
   late-RoPE kernel, and the QSA index (§3).
3. **One RoPE table format for the whole engine** (§5). It is two small tables
   joined by the angle-addition identity. That makes a table of any reach about
   1,300 times smaller than today's and lets every rung of a model sit in L2.
4. How a slot's rung reaches every kernel **without bleeding into another
   session** (§6), and how it is chosen (§8). The rung is a pure function of the
   slot's length, so there is no rung state to keep.
5. **Un-rotated storage for the QSA index, rotated on load by the scorer**
   (§7). The index keeps one stored form, like K/V, and nothing rotated
   anywhere. The scorer comes with its own test harness and a measured
   optimisation plan.

---

## 1. The invariants

> **I1 — Nothing stored depends on a rung.** K/V, sealed chunks, the cold tier,
> DeltaNet and PLE state, and every QSA index row are stored un-rotated. RoPE
> exists only at the moment of a read, in the kernel that reads.
>
> **I2 — A rung never crosses a session.** Every kernel takes a row's rung from
> that sequence's own state: its `SlotHeader`, or, for a kernel launched on one
> sequence, the launch argument. No table pointer, scale or staged rotation is
> shared by rows of two sequences.
>
> **I3 — The attention temperature is applied exactly once**, to Q's rotary
> pairs, and never to K, the table or `softmax_scale`.

**Why hidden-state drift does not matter.** K and V from layers after the
first attention layer do carry a trace of the rung that was active when they
were produced: their residual stream passed through earlier attention layers.
That is the same approximation splicing already makes, in a far smaller
amount.

- **Splicing is worse.** A spliced chunk's hidden states came from a
  different context altogether.
- **YaRN barely changes attention at short distances.** It leaves the
  high-frequency pairs alone, and those carry local attention.

---

## 2. The schedules

A model's RoPE is one of four schedules: plain, static linear, static Llama3,
or YaRN. A **progressive** schedule is simply one with more than one rung. The
published parameters are properties of the model, like its tokenizer revision.
The GGUFs for these models carry no YaRN keys, because YaRN is opt-in upstream,
so the parameters live in the preset.

| Model (arch) | Schedule | Rotary pairs | θ | Trained window L₀ | Rungs: ceiling → factor | Supported max |
|---|---|---|---|---|---|---|
| Qwen3-8B / 14B (`Qwen3`), Qwen3-30B-A3B (`Qwen3Moe`) | **progressive YaRN** | 64 of 64 (hd 128) | 1e6 | 32,768 | 32,768 → 1 · 65,536 → 2 · 131,072 → 4 | 131,072 |
| Qwen3.5-0.8B, Qwen3.5-9B (`Qwen35Dense`), Qwen3.6-35B-A3B + AntiLoop (`Qwen35Hybrid`) | **progressive YaRN** | 32 of 128 (hd 256) | 1e7 | 262,144 | 262,144 → 1 · 524,288 → 2 · 1,010,000 → 4 | 1,010,000 |
| Qwen3.8-Flash-Next (`Qwen4Exp`) | **progressive YaRN** | attention 32 of 128 (hd 256); QSA indexer 32 of 64 (hd 128) | 1e7 | 262,144 | 262,144 → 1 · 524,288 → 2 · 1,000,000 → 4 | 1,000,000 |
| Qwen2-0.5B (`Qwen2`) | plain | 32 of 32 (hd 64) | 1e6 | 32,768 | — | 32,768 |
| Hermes-3 3B / 70B (`Llama`) | **static Llama3** (factor 32 / 8, L₀ 8,192, low 1, high 4) | 64 of 64 (hd 128) | 5e5 | 131,072 | one rung, at every length | 131,072 |
| DeepSeek-V4-Flash (`DeepSeekV4`) | **static YaRN** on compressing layers (factor 16, L₀ 65,536, β 32/1, θ 160,000); plain on SWA layers | 32 | per layer | 1,048,576 | one rung per layer kind | 1,048,576 |

The preset for Qwen3-30B-A3B is `unsloth/Qwen3-30B-A3B-GGUF`, the original
release with a 32K native window. It is not the 2507 release, whose native
window is 262K.

**Sources.**

- Qwen model cards: Qwen3-30B-A3B, Qwen3.5-9B, Qwen3.6-35B-A3B, Qwen3.8-Flash-Next.
- vLLM's Qwen3.8-Flash-Next recipe.
- The `config.json` of Hermes-3-Llama-3.2-3B, Hermes-3-Llama-3.1-70B, DeepSeek-V4-Flash and Qwen2-0.5B-Instruct.

**Why the rung list stops at 2 and 4.** Those are the two factors Qwen gives
guidance for: 2.0 "if the typical context length … is 524,288", and 4.0 for
the full extension. The schedule type takes any list, so adding a rung is a
data change plus the needle gate in §10. Nothing else changes.

**Why Qwen3.5-0.8B is progressive.** It is the same lineage with the same RoPE.
Its rung 1 is the trained model exactly, so turning the schedule on changes
nothing below 262,144. Past 262,144 the alternative is plain RoPE running
outside anything it was trained on. The lineage's rungs are the better bet, and
§10's needle gate measures them for the 0.8B like any other model. Its 1,010,000
supported max comes from the lineage, not from the 0.8B's own card.

**Why AntiLoop is progressive.** It is a fine-tune of Qwen3.6-35B-A3B with the
base's RoPE unchanged, so it inherits the base schedule.

**Why static schedules are in scope.** Llama3 and DeepSeek-V4 were trained
with their scaling switched on, so a crossover would be wrong for them: they
run their one rung at every length. They still move onto the §5 table format and
the ceiling guard.

**Why Qwen2-0.5B stays plain.** Qwen documents YaRN only for Qwen2's 7B and
72B Instruct models, and this checkpoint's `config.json` has no
`rope_scaling`.

**GGUF-declared schedules** (for `Model::Custom`) are taken at their word:

- A declared `"yarn"` gets a **static** YaRN schedule at the declared factor.
- A declared `"linear"` gets a static linear schedule.
- A file with a `rope_freqs.weight` tensor gets a static Llama3 schedule from
  that tensor (§3.1).
- Anything else gets plain RoPE.

Progressive rungs are a property of the presets. Nothing invents rungs for a
file.

---

## 3. What the audits found

### 3.1 The fixed-scaling models: one already scaled, one silently not

**DeepSeek-V4 already applies its static YaRN, and correctly.**

- `latent_moe::rope::yarn_freqs` (`latent_moe/rope.rs:32`) feeds the kernel
  table through `build_rope_table` (`kernel_attention.rs:117-129`,
  `paged.rs:190`).
- Per layer, compressing layers use YaRN with θ 160,000 and SWA layers use plain
  θ 10,000 (`latent_moe/config.rs:118-124`).
- The constants live in `deepseek4.rs:129-136`. The GGUF does not carry them.

**Hermes-3 does *not* apply its Llama3 scaling.**

- llama.cpp's converter writes Llama3 scaling only as a `rope_freqs.weight`
  tensor, never as metadata keys (`conversion/llama.py`,
  `generate_extra_tensors`).
- Our loader looks only for four `llama.rope.scaling.*` keys
  (`quantized_llama.rs:1091-1128`). It never reads that tensor, so it builds
  plain RoPE at θ 5e5.
- Llama3 scaling divides the low-frequency pairs **at every position**. So
  Hermes is slightly wrong even at short contexts, not only past 8K.
- The note at `quantized_llama.rs:2188-2198` concluded that the Nidum file
  "ships no rope scaling". It checked only the keys.

**Qwen3-8B/14B run with an unintended linear factor.**

- `infer_rope_scaling_factor` (`quantized_qwen3.rs:51-62`) turns any
  `qwen3.context_length` above 32,768 into a linear factor `ctx / 32768`.
- Our Qwen3-8B GGUF declares 40,960 (`quantized_qwen3.rs:1608-1610`).
- So every Qwen3 dense position is compressed 1.25× today, at every length.

**The two neighbours are not affected.** Qwen3Moe scales only on a declared
factor, and Qwen2's copy of the inference is inert for our file, which declares
exactly 32,768.

### 3.2 `mscale`: not applied anywhere, so there is nothing to double

- **No path applies an attention temperature today.** Every GQA softmax scale
  is exactly `1/√head_dim`: `prefill_utils.rs:667, 1686`,
  `batched_layer.rs:1290, 1511`, `decode_utils.rs:447`, and the reference paths
  in the `quantized_*` loaders.
- **The one `mscale` in the tree** is `deepseek2.rs:375-494`, which is not on any
  path we serve.
- **DeepSeek-V4 applies none, matching its reference.** Its design doc records
  "No mscale/attention-scale correction is applied (unlike deepseek2)"
  (`docs/deepseek/deepseek_v4_flash.md:282`). The RoPE module is parity-tested
  against a scalar transcription of `model.py` (same doc, line 63).
- **Every rope table stores unit-magnitude `(cos, sin)`.**

### 3.3 Every late-RoPE kernel: no CTA spans two sessions

Every kernel that rotates at read time was audited for any thread, warp, CTA,
shared buffer, register cache or per-wave array holding RoPE data for rows of
more than one sequence. None does.

| Kernel | Work → sequence | Q rotated at | K rotated at | Shared across sequences? |
|---|---|---|---|---|
| `int8_decode_kernel`, split-KV (`int8_decode_kernel.cuh:71, 829`) | CTA = (slot, kv_head, split), `slot = blockIdx.x` | `ws_rope + ws_len` (`:224-231`) | `slice_rope(sl) + within − off` (`:485-495`) | no |
| `int8_decode_stripe_kernel` (`:1109, 1462`) | same; warp pairs stripe one slot's tokens | `:1236-1250` into per-CTA `shared_q` | `:1049-1053` | no |
| `int8_decode_bmma_kernel`, batched-M (`:1492, 1900`) | same; the MMA M rows are **one slot's** heads, not several sequences | `:1611-1623` | `:1711, 1773-1777` | no |
| `int8_decode_tile_kernel`, HD 256 (`int8_decode_tile_kernel.cuh:1433`) | same; a tile's quads may come from different slices **of the same slot** | `i8_apply_rope(x, ws_rope + ws_len)` (`:2150-2168`) | row `rope0`, then **table row 1 as the unit step** walked by angle addition (`:1254-1268, 1359-1369`) | no; rope values live in per-lane registers |
| combine and commit kernels (`int8_decode_kernel.cuh:1936`, `decode_helpers.cuh:203`) | per (slot, head) row / per slot | — | — | no |
| `paged_prefill_int8_kernel` (`paged_prefill_int8_kernel.cuh:165`) | `batch_idx = blockIdx.z / splits`; grid x = `ceil(max_q_len / block_m)` measured from the sequence's own `q_start`, so **a tile cannot cross `cu_seqlens`** (`:215-235, 940`) | `prefix_len + tok + rope_offsets[b]` (`:365-366`) | `pos + rope_offsets[b]` (`:606`) | no; every shared buffer holds one sequence |
| `paged_glue_kernel` (`paged_glue_kernel.cuh:66`) | `slot = blockIdx.x` | `slice_rope(gap) + in_blk − off` (`:248-259`) | `:295, 338-341` | no |
| `latent_decode_kernel` / `latent_prefill_kernel` (DeepSeek) | one slot per CTA; one query per CTA | `latent_decode_kernel.cuh:211-220`, `latent_prefill_kernel.cuh:468-520` | `:380-389`, `stage_key` `:407-418` | no |
| QSA `append_kernel` (`qsa_index_append.cu:125-187`) | **one launch spans every sequence in the wave**; one CUDA block = one key of one sequence | — | `job.pos` | the table pointer is launch-wide |
| QSA `flush_kernel`, `place_kernel` | one sequence per launch in practice | — | `tail_base + n·ratio`; one `delta` per page | the table pointer is launch-wide |
| QSA query rope, host (`indexer.rs:1187-1201`) | one call over **all rows of the wave** | each row's own position | — | one table for every row |
| float fallback, host (`batched_layer.rs:1119-1258`) | per-sequence loop | per-sequence slice | per-sequence slice | one table split once per wave |

So bleed is possible only through **launch-wide parameters**. §6.2 lists every
one of them with its fix.

Positions are already per-sequence everywhere, and K is stored un-rotated on
every path (the decode scatters at `int8_decode_kernel.cuh:145-195` and friends
write raw K). So a sealed chunk shared by two sessions is rotated by each reader
with that reader's own table, and I1 already holds for K/V.

### 3.4 The QSA index stores rotated keys

- **The live tail is rotated at absolute positions.** The append and flush
  kernels rotate each pooled key at `tail_base + n·ratio` into
  `IndexCache.keys` (`indexer.rs:62-64, 313-341, 1365`).
- **The live tail is scored by cuBLAS reading those rows directly**
  (`indexer.rs:875-905`), not by the paged scorer.
- **Pages are rotated relative to their own start.**
  - A seal turns rows back by `−frame` (`seal_page`, `wave.rs:81-89`).
  - `IndexPage` carries `roped_base` (`paged_index.rs:47-66`).
  - Placement rotates the whole page forward by one signed `delta` into a
    channel-blocked staging (`place.rs:47-60`, `qsa_page_place.cu:132-139`).
  - The paged scorer reads only that staging (`qsa_score_paged.cu:150-205`).
- **The index tables are a second, separate family:** `RopeTables`
  `[max_pos, 32]` for cos and for sin, with **f32 angles**
  (`qwen35/attention.rs:73-87`). The error in those angles grows with position:
  about 0.01 rad of phase at 200K on the fastest pair.

**The frame arithmetic is consistent on every path read.** These cover the live
seal, the range seal, tail-span pages, `close_tail_into_page`, push, and the
snapshot export and restore. So the rotated storage is not a misplacement bug.
It is fragile by construction:

- every sealed row is rotated two or three times, each through an f32-angle
  table;
- a sealed page carries its sealing table's frequencies inside its rows, so it
  cannot be read under another rung;
- a wrong frame anywhere shows up only as a wrong selection above 2,051 tokens.

**The aux-blob index is live.** A new conversation runs the priming projection
and then installs its prompt-branch checkpoint. `restore_aux_state` replaces the
primed index with the checkpoint's, and the first turn then skips
`apply_projection` (`conversation.rs:1041-1073`). The checkpoint's rows were
prefilled from position 0 on the same prompt, so they sit where the primed K/V
sits. §7 keeps this path; it simply stops rotating it.

§7 removes the stored rotation entirely.

---

## 4. The frequency source: `models/rope_schedule/`

This is one module, one concern per file, and the only place in the engine that
computes a RoPE frequency.

| File | Holds |
|---|---|
| `schedule.rs` | `RopeSchedule { Plain, Linear{factor}, Llama3{…}, Yarn{l0, beta_fast, beta_slow, rungs} }`, `Rung { ceiling, factor }`, `RungFreqs { inv_freq: Vec<f32>, q_rot_scale: f32 }`, `supported_max()` |
| `yarn.rs` | The YaRN transform and `mscale`, moved from `latent_moe/rope.rs` |
| `llama3.rs` | `rope_freqs.weight` → frequencies, and the published-parameter formula (moved from `llama_rope.rs`) that the tests check the tensor against |
| `select.rs` | `rung_for(schedule, reach)` and the ceiling refusal (§8) |
| `table.rs` | The factored table (§5): CPU builder, CPU lookup mirror, upload, plus the QSA lane tables (§7.4) |

### 4.1 The YaRN transform

The reference is vLLM's `YaRNScalingRotaryEmbedding`, because the Qwen cards
point at vLLM and SGLang. The formula below is written over the rotary width
`d` (64 for the hybrid lineage, 128 for Qwen3), with factor `s` and trained
window `L₀`:

```text
base_i   = θ^(2i/d),                       i = 0 .. d/2
extrap_i = 1 / base_i
interp_i = 1 / (s · base_i)
cdim(r)  = d · ln(L₀ / (2π r)) / (2 ln θ)
low      = max(floor(cdim(β_fast)), 0)
high     = min(ceil (cdim(β_slow)), d − 1)          (high += 0.001 if low == high)
ramp_i   = clamp((i − low) / (high − low), 0, 1)
inv_i    = interp_i · ramp_i + extrap_i · (1 − ramp_i)
m        = 1 if s ≤ 1, else 0.1 · ln(s) + 1
```

`latent_moe::rope::yarn_freqs` already computes `inv_i` exactly this way.

Worked by hand from the formula, with β = 32/1; the unit tests pin the full
vectors:

| Model | d | θ | L₀ | low | high | pairs kept · blended · interpolated | m at s=2 / s=4 |
|---|---|---|---|---|---|---|---|
| Qwen3 | 128 | 1e6 | 32,768 | 23 | 40 | 24 · 16 · 24 | 1.0693 / 1.1386 |
| Qwen3.5 / 3.6 / 3.8 | 64 | 1e7 | 262,144 | 14 | 22 | 15 · 7 · 10 | 1.0693 / 1.1386 |

### 4.2 `mscale`: `m²` on Q's rotary pairs

vLLM multiplies its cos/sin cache by `m`, which rotates Q and K each by `m`.
The logits over the rotary dims then gain `m²`, and the pass-through dims are
untouched.

We apply `m²` to **Q's rotary pairs only**, at the one place each path rotates
Q. The kernel reads it by rung from the rung buffer (§6.1), never from a
per-slot copy. Each alternative fails:

- **Not in the table.** The tile kernel steps a group of keys by repeated
  multiplication with the unit-step entry, so a scaled table would compound to
  `mⁿ`. The factored lookup (§5) multiplies two entries, so a scaled table would
  also give `m²` from every lookup.
- **Not in `softmax_scale`.** It is one scalar per launch, which bleeds across
  sessions, and it also covers the pass-through dims.
- **Not on K.** K's int8 window scales must never depend on the rung.

The Q sites:

| Path | The single site |
|---|---|
| Decode split-KV, stripe, batched-M | `apply_rope_*_f32(q, …, q_rope_pos, …)` (`int8_decode_kernel.cuh:224-231, 1236-1250, 1611-1623`) |
| Tile decode | `i8_apply_rope(x, q_pos, …)` for the Q rows (`int8_decode_tile_kernel.cuh:2168`), before the Q window scale is taken |
| Int8 prefill | `i8_apply_rope` for Q (`paged_prefill_int8_kernel.cuh:365-366`), before int8 quantization |
| Glue | `paged_glue_kernel.cuh:255-259` |
| Float fallback, CPU oracles | the one Q-rotation helper `rope_schedule` exports |
| QSA indexer query | the scorer's query-tile load (§7.4). The reference rotates the indexer with the attention's own YaRN cache, `m` included (§12) |

---

## 5. One table format: two small tables and the angle-addition identity

### 5.1 The mathematics

For rotary pair `i` with frequency `ω_i`, RoPE rotates by the angle
`α(p) = p·ω_i`. The angle is **linear in position** for every schedule here:
plain, linear, Llama3 and YaRN all change `ω_i` and nothing else. Split the
position as

```text
p = h·B + l,     B = 2^b,   0 ≤ l < B,   0 ≤ h < H
α(p) = h·B·ω_i + l·ω_i
```

The rotation by a sum of angles is the product of the two rotations:

```text
cos α(p) = cos(hBω_i)·cos(lω_i) − sin(hBω_i)·sin(lω_i)
sin α(p) = sin(hBω_i)·cos(lω_i) + cos(hBω_i)·sin(lω_i)
```

So two tables, `HI[h][i] = (sin, cos)(h·B·ω_i)` and
`LO[l][i] = (sin, cos)(l·ω_i)`, reproduce every position below `H·B` from
`H + B` rows instead of `H·B`.

- **The row count is smallest at `H = B = √N`.**
- **Reach extends at almost no size.** Doubling `H` doubles the reach and adds
  `H·P·8` bytes.
- **A third level extends it further.** `p = a·B² + h·B + l` covers `N` with
  `3·N^(1/3)` rows, at one more combine.
- **The identity holds for any frequency.** So every rung, and every schedule,
  is the same format with a different `ω`. A YaRN rung is one more table pair,
  built from `yarn.rs`'s `ω`.
- **The tile kernel's unit step is `LO[1]`,** which already exists in the format.

### 5.2 The format, and why it is DeepSeek's

DeepSeek-V4's latent kernels already run this format. It is built, bit-exact
and tested, so it becomes the engine's one RoPE table:

- **Layout** (`latent_common.cuh:216-237`):
  - `float2 (sin, cos)`;
  - a **hi block** `[2048][P]`, where row `h` is position `h·2¹⁰`;
  - then a **lo block** `[1024][P]`;
  - frequency innermost;
  - `rope_table_len = (2048 + 1024)·P·2` f32s (`geometry.rs:119-121`).
- **Reach:** 2²¹ = 2,097,152 positions. Every supported max in §2 is at most
  1,048,576.
- **Builder** (`latent_rope_table_kernel`, `latent_common.cuh:257-272`):
  - the angle is reduced in **f64** with `__dmul_rn/__dsub_rn/__dadd_rn`;
  - `pos·ω` is exact in f64, because a 21-bit integer times a 24-bit mantissa
    fits in 53 bits;
  - then a deterministic minimax `sincosf` on [−π/4, π/4], all `_rn`.
- **Lookup** (`rope_lookup<NF>`, `latent_common.cuh:222-237`): two `float2`
  loads, then `_rn` multiply and add. There is **no contraction**, so the CPU
  mirror can be bit-exact.
- **Tests:** a CPU mirror of builder and lookup (`paged.rs:2444-2492`), with
  `rope_table_device_matches_mirror` (`paged.rs:4195-4246`) checking every entry
  and a spot-checked lookup.

It moves out of `paged-latent/` into a shared
`candle-kernels/src/rope/rope_table.cuh`. **`P` is a template parameter only**:
a kernel is instantiated per rotary width, and nothing carries `P` at run time.

**Only rotary pairs are stored.** For partial rotary (the hybrid lineage, 32 of
128 kernel pairs after the `RotaryLayout` permutation), the helper returns
`(sin 0, cos 1)` for a pair index ≥ `P`. That is a select, not a table row.

### 5.3 Sizes

| Table | Today | Factored, per rung |
|---|---|---|
| Hybrid attention (`[max_pos, 256]` F32) at 262,144 | 256 MiB | **768 KiB** (P = 32) |
| Hybrid attention at 1,010,000 | ≈ 986 MiB | 768 KiB |
| Qwen3 attention (`[max_pos, 128]`) at 131,072 | 64 MiB | **1.5 MiB** (P = 64) |
| Qwen4Exp indexer (`RopeTables`, cos + sin `[max_pos, 32]`) at 262,144 | 64 MiB | shares the attention rung table |
| DeepSeek-V4 | 43 layers × 768 KiB = 32 MiB (one table per layer, `wave.rs:370-381`) | **2 × 768 KiB**, one per frequency set |

**All rungs of a model fit in one allocation of at most 4.5 MiB.** That is three
rungs of 1.5 MiB, well inside even the 3090's 6 MiB L2.

So every table exists from load:

- no lazy build and no `max_blocks` rebuild (today's caches compare with `==`,
  `batched_model.rs:899-915`, `qwen4exp/wave.rs:1436`);
- no host loop over hundreds of millions of entries;
- no upload of hundreds of MB.

**The indexer and attention share one rung table.** The QSA indexer rotates with
the model's own `θ` and 64-dim rotary width (`config.rs:229-236`), so its
frequencies are the attention's.

### 5.4 Cost and accuracy

- **Traffic.** Today every K token's rotation reads a `head_dim`-wide row of a
  table that at depth lives in DRAM: 1 KiB per position for HD 256. The factored
  lookup reads two `float2`s from a table that stays in L2.
- **Arithmetic.** Six `_rn` operations per pair, where today there are none.
- **Precision.**
  - The angle is exact and reduced in f64.
  - `sincosf` rounds once, and the combine adds four roundings. The error is a
    few ulp and **does not grow with position**.
  - **Today it depends on the path.**
    - `compute_rope_cs` (Qwen2, Qwen3, Llama) takes an f64 angle and rounds once
      (`prefill_utils.rs:1764`), so it is slightly tighter than the factored
      table.
    - `RotaryLayout::rope_table` (the whole hybrid lineage and Flash-Next)
      computes `pos as f32 * inv` (`rotary_layout.rs:185`).
    - The QSA `RopeTables` computes its angles in f32 as well.
    - So does the HF reference.
    - In all of those f32-angle paths, the error grows with position.
- **Nobody has measured this format's accuracy** (DeepSeek's tests prove only
  device = mirror). §10 adds the measurement and pins a bound.

This is a numerical change for every GQA model, so rung 1 is **no longer
bit-identical to the pre-change build**. It is instead:

- bit-exact to its own CPU mirror;
- within a pinned bound of f64 truth at every position;
- gated by the twelve model forward gates, perplexity, and a throughput gate
  that must show no loss.

**If the tile kernel regresses:** its per-group cost is one row `rope0` plus the
unit step, and it already walks the rest by angle addition. So the factored
lookup adds one extra `float2` load per pair per group, not per token.

---

## 6. Rungs without bleed

### 6.1 The rung buffer

A model's rungs live in one device buffer: a short metadata block, then the
tables.

```text
struct RungMeta { float q_rot_scale; uint32_t _pad; };   // m²; exactly 1.0 on rung 1
RopeRungs {
    const RungMeta* meta;      // [n_rungs]
    const float2*   tables;    // [n_rungs][(2048 + 1024) · P]
    uint32_t        n_rungs;
}
```

- **Addressing.** Every rung's table has the same size, so rung `r` is at
  `tables + r·(3072·P)`, one integer multiply-add with `P` a compile-time
  constant. Its `m²` is `__ldg(&meta[r].q_rot_scale)`, which is L1-hot after the
  first warp.
- **It replaces `rope_cs`** in every paged-kernel signature.
- **Bounds.** A rung ≥ `n_rungs` traps, rather than reading the next table.
  Positions are bounded as `latent_common.cuh:222-237` already bounds them:
  `hi` is clamped.

### 6.2 Every launch-wide parameter, and its fix

| # | Launch-wide today | Fix |
|---|---|---|
| 1 | The single `rope_cs` pointer passed to every paged kernel (`paged-decode/api.rs:34, 57, 90, 113`; prefill and glue likewise) | `RopeRungs`, indexed by the sequence's `SlotHeader.rope_rung` (below) |
| 2 | The single `softmax_scale` (prefill `:731`, glue `:382`, decode `:702/725`, `:1062`, `:1829/1882`, tile `:2154`) | unchanged, and it carries nothing rung-dependent: `m²` goes on Q (§4.2) |
| 3 | Table row 1 in the tile kernel, hard-coded as `rope_cs + HEAD_DIM` (`int8_decode_tile_kernel.cuh:1261`) | the unit step is `LO[1]` **of the slot's rung table** |
| 4 | The QSA append launch's one table, covering every sequence in the wave | append no longer rotates (§7), so it has no table |
| 5 | The host QSA query rope's one table, covering every row in the wave | deleted: queries are rotated in the scorer, which is launched per sequence (§7.4) |
| 6 | The float fallback's one per-wave split, and the per-group host cos/sin behind `BatchedAttentionParams::rope_cos/rope_sin` (non-paged path) | the fallback picks the sequence's rung inside its per-sequence loop; the non-paged cos/sin are built per row from the row's rung |

`SlotHeader` (`slot_types.cuh:56-64`) grows from 24 to 32 bytes, with the rung
as an explicit field:

```text
struct SlotHeader {
    uint32_t n_slices;
    uint32_t write_slice;
    uint64_t slices_ptr;
    uint64_t position_map_ptr;
    uint32_t rope_rung;       // index into RopeRungs
    uint32_t _pad;
};
```

Every attention kernel in §3.3 already loads its sequence's `SlotHeader`
before it rotates anything:

- decode: `int8_decode_kernel.cuh:122-125, 1166-1169, 1543-1546`;
- tile: `int8_decode_tile_kernel.cuh:1590-1593`;
- prefill: `paged_prefill_int8_kernel.cuh:220`;
- glue: `paged_glue_kernel.cuh:150-152`.

So the rung comes from the same record as the sequence's positions. A 32-byte
header is exactly one sector, where the 24-byte stride straddles sectors today.

**Every reader of the header changes stride, not only the ones that rotate:**

- `get_slot_header`'s `slot_idx * 24` (`slot_types.cuh:67-69`);
- the per-layer header stride on the Rust side (`DecodeHeaders`, the slot
  serialiser in `slot_state.rs`);
- the commit kernel (`decode_helpers.cuh:184`);
- **DeepSeek's latent kernels,** which read the same `SlotHeader`:
  - `latent_decode_kernel.cuh:125`;
  - `latent_prefill_kernel.cuh:174`;
  - `latent_common.cuh:318`.

  They never read `rope_rung`. A latent layer's frequency set is chosen by layer
  kind (compressing or SWA, `latent_moe/config.rs:118-124`) and passed per
  launch, as today, from the two shared tables in §5.3. DeepSeek's serialiser
  writes `rope_rung = 0`.

The stride change is one constant on each side. It is checked by a layout test
that asserts `sizeof(SlotHeader) == 32` and the field offsets on both sides, the
same way `slot_types.cuh:64` asserts 24 today.

**Why an explicit field rather than packed bits.** Each candidate the audit
found means something else somewhere:

- `position_map_ptr` is 0 in decode but a live pointer in prefill and glue;
- `n_slices` already traps at 2²³ in the tile kernel;
- `rope_offsets` is added straight into positions.

Masking a packed field at every consumer is exactly the kind of site a future
kernel forgets.

### 6.3 Why this cannot bleed

- Every attention kernel maps a CTA to one sequence (§3.3). Within a CTA, the
  table base and `m²` come from that sequence's header, loaded once, in
  registers.
- The only launch that spans sequences, QSA append, no longer rotates.
- The QSA scorer is launched per sequence span, so its rung argument belongs to
  that sequence alone.

**Nothing rung-dependent is a launch parameter.**

---

## 7. The QSA index stores un-rotated keys, rotated on load

### 7.1 The rule

> **Every stored index row — live tail, page, turn record, snapshot — is the
> pooled, normed, un-rotated block key. Rotation happens only inside the
> scorer, as it reads the key, at the key's position and the slot's rung.**

This is also the reference's own structure: it caches raw keys and ropes them
when read (`docs/qwen38_flash_next.md` §12.5). The CPU oracle already works this
way (`qsa.rs:127-136`), so the device path becomes structurally identical to it.

**Nothing rotated is ever written:** not a page view, not a tail view, not a
placement staging.

- The paged scorer reads stored, un-rotated rows, plus a per-page base position
  and the slot's rung.
- The live tail stops being scored by cuBLAS (`indexer.rs:875-905`) and becomes
  the last page of the same scorer.
- The indexer's queries are rotated in the scorer too.

**What that buys:**

- One stored form and one read path.
- A rung change costs nothing: the next launch reads a different table.
- The host rope ops on the query path are gone.

### 7.2 What it costs

- **The scorer is L2-bound today:** 87.8% of peak L2 against 40% compute at 128K
  and 64 rows (`qsa_score_paged.cu:301-306`).
- **Rotation needs 32 `(sin, cos)` pairs per candidate.** Read naively, that is
  256 B beside a 512-B key, on the kernel's own bound. §7.4 avoids the read.
- **Rotation adds arithmetic.** Per rotary pair, forming the rotation (warp term
  × lane term) and applying it to the key is about 12 operations. That is about
  384 per candidate.
  - At `(TILE_R, CPT) = (4, 2)` with H = 4, the candidate's dot products are
    4 × 4 × 128 = 2,048 multiply-adds, so the overhead is about **19%**.
  - At the `(1, 1)` arm they are 4 × 128 = 512, so it is about **75%**. That arm
    is chosen because the grid cannot be filled and it is latency-bound
    (`:194-197`), which may hide the extra arithmetic. The harness decides.
- **It holds a second `float4` live,** the rotary partner, on a kernel already at
  its register edge. It uses 64 registers, and at 51 it spilled (`:99-104`).
- **The live tail moves off cuBLAS, including at prefill width.** Today the
  tail is scored by a cuBLAS matmul over **every** query row of the span
  (`indexer.rs:875-905`, tiled by `SCORE_TILE_BYTES`). A prefill chunk is
  thousands of rows.
  - The paged scorer puts rows on the grid's y axis and re-reads every key once
    per row tile (`:301-306`).
  - At 8,192 rows and 128K depth, that is about 2,000 reads of each key, where a
    GEMM stages a key tile once and reuses it across a large row tile.
  - The decode-shaped arms are not built for this. §7.4 adds an arm that is.

### 7.3 What changes

| Piece | Today | Under §7 |
|---|---|---|
| `append_kernel` / `flush_kernel` | write the rotated row, row-major, into `IndexCache.keys` | write the **un-rotated** row, channel-blocked (`[D/4, capacity, 4]`, pitch = capacity), into `keys`. No table and no rung: these kernels stop touching RoPE at all. Growing the capacity re-lays the rows at the new pitch |
| Live-tail scoring | cuBLAS on `keys` plus `indexer_score_reduce`, at every row count | the tail is the last page of the paged scorer: `{keys, pitch = capacity, base = tail_base}`. Prefill-width spans take the wide-row arm (§7.4) |
| Indexer query rope | host eager ops over every row of the wave (`indexer.rs:1187-1201`) | deleted; the scorer rotates its query tile (§7.4). `rms_norm_last(q_norm)` stays on the host |
| `IndexPage` | rows rotated relative to the page start; `roped_base`; `at_frame` | **un-rotated rows**, channel-blocked. `roped_base` and `at_frame` are deleted |
| `seal_page`, `rotate_rows(±frame)` | rotate out of or into a frame | deleted. A seal copies stored rows |
| `Placement`, `place_kernel`, `place_pending`, `IndexCache.placed` | rotate each page into a staging | deleted |
| Record format (`paged_index.rs:503-628`) | row-major `[rows, D]` F32 | **channel-blocked `[D/4, rows, 4]` F32, the scorer's layout.** Disk and device share one layout, so `push_page` does not transpose and a seal does not un-block |
| Scorer page descriptor | `{keys}` + `page_first` | `{keys, pitch, base}` + `page_first`; the launch adds the slot's rung and the span's first query position |
| `export_aux_state` / `restore_aux_state` | normalise the tail by `−tail_base`; restore at `tail_base = 0` | export stored rows as they are; restore at `tail_base = 0`, which is where a branch checkpoint's rows sit (§3.4) |
| `RopeTables` | index tables with f32 angles | deleted; the attention rung table serves the index |
| Oracles (`PagedIndex::score_reference`, `qsa.rs`) | rotate each page by `base − roped_base` | rotate stored rows at `base + j·ratio`, which is what `qsa.rs` already does |
| Scorer generic arm (`qsa_score_paged.cu:262-297`: runtime H, `D` not a multiple of 4) | exists for head_dim-16 test geometries | deleted; the tests run at the production geometry |

**Memory falls.** Today a page holds a record and a rotated staging, and the
live tail's rows are rotated in place. Under §7 each row is held once, in one
layout.

### 7.4 The scorer's rotation: a warp term and a lane table

**Key positions.** Rows are ordered by position, and page `p`'s row `j` sits at
`base_p + j·ratio`. The scorer's candidate axis is the global row index
`g = page_first[p] + j`, so

```text
pos(g) = δ_p + g·ratio,         δ_p = base_p − page_first[p]·ratio   (signed, per page)
```

A thread carries `CPT` candidates, `256` apart. Thread `t` of block `b` holds,
in slot `s`, candidate `g = b·span + t + 256·s`, plus the grid stride
(`qsa_score_paged.cu:146, 165`). A warp's lanes are 32 consecutive threads, so
in every slot they hold 32 consecutive candidates: `g = G + L`, where `G` is the
slot's warp-first candidate, uniform across the warp, and `L` is the lane. So

```text
pos(g) = [δ_p + G·ratio]  +  [L·ratio]
            warp term W       lane term
```

By the angle-addition identity (§5.1), the rotation at `pos(g)` is the rotation
at `W` composed with the rotation at `L·ratio`.

- **The warp term is uniform** when the slot's 32 candidates lie in one page.
  That is every warp except one straddling a page boundary, and a page is a turn,
  tens to thousands of rows long. `W` is computed **per CPT slot**, because slot
  `s` may sit in a different page than slot 0, and it comes from one factored
  lookup per slot per warp, as a broadcast.
- **The lane term depends only on `L`, `ratio` and the rung.** It is a
  `[32 pairs][32 lanes]` table of `float2`, 8 KiB, one per (rung, ratio) pair.
  - `ratio` is per layer (`compress_ratios[li]`). Layers with ratio 0 attend
    densely and never score, so they need no table.
  - The tables are built at load (`rope_schedule/table.rs`), and each block
    copies its launch's table into shared memory beside the query tile.
  - Stored pair-major and lane-minor, a warp reading pair `i` touches 32
    consecutive `float2`s, so there are no bank conflicts.
- **Straddling warps** fall back to a per-lane factored lookup. The harness
  reports their share.
- **No L2 traffic is added** on the fast path: the lane term is in shared memory
  and the warp term is one broadcast.

**The rotary pairs.** The indexer rotates NeoX half-split within its 64-dim
rotary width (`attention.rs:57-62`). In the channel-blocked layout, pair
`(i, i + 32)` for `i < 32` is channel group `c = i/4` with its partner group
`c + 8`. So the rotary width is channel groups 0–15, **half of `D = 128`**, and
the channel loop becomes two loops:

- **`c < 8` (rotary):** load `k[c]` and `k[c + 8]`, rotate four pairs, and
  accumulate both `float4`s.
- **`c ≥ 16` (pass-through):** today's loop, unchanged.

**Queries.** The query tile is already loaded into shared memory once per block
(`qsa_score_paged.cu:136-141`). Row `r` of a launch sits at `q_pos0 + r`,
because a launch is one sequence's span of consecutive positions. The load
rotates each row's rotary pairs at that position with the launch's rung, and
applies the rung's `m²`, as the reference does (§12). That is `TILE_R × H × 32` pair rotations per
block, once, instead of a host op over the whole wave.

**The wide-row arm, for prefill spans.** The arms above hold the rows of one
tile (`TILE_R ≤ 4`) in registers and put row tiles on the grid, which is right
when rows are few and candidates are many. A prefill span inverts that. So a
third arm is selected when a span's row count passes a threshold the harness
measures:

- **A block owns a candidate tile.** For example, 64 candidates × 128 dims × 4 B
  = 32 KiB. The block stages it into shared memory **rotated once**, using the
  same warp term and lane table as above.
- **The block loops over the span's rows** in tiles, instead of the grid doing
  so. Each key is read from L2 once per block rather than once per row tile, and
  each rotation is paid once per key, not once per row tile.
- **The inner product is a shared-memory GEMM tile:** rows × candidates × `D`,
  written out through the same ReLU fold over heads and the same prefix mask.
  Tensor-core MMA over TF32 (or BF16, under §7.6 lever 5) is a lever for this
  arm, measured like any other.

This is the kernel that replaces cuBLAS for the tail at prefill width, and it
scores injected pages at that width too. Today they go through the paged scorer
at the same row counts.

**Precision.** A key rotation is an f64-exact angle rounded once in the table,
then two combines: warp term × lane term, then applied to the key. The harness
measures the error against f64 truth and pins it.

### 7.5 The test harness — built before the kernel changes

The harness comes first: every optimisation below is a claim about a number it
produces. It lives in `candle-transformers/tests/qsa_score_rot_harness.rs`,
beside `qsa_paged_index_tests.rs` and `qsa_page_place_bench.rs`.

**Correctness.** These are plain GPU tests, not `#[ignore]`, so `/fast-test`'s
GPU pass runs them. They use the production geometry: `D = 128`, rotary 64,
`ratio = 4`.

| Test | Asserts |
|---|---|
| `rot_scorer_matches_the_f64_oracle` | Scores against an f64 host oracle that rotates each stored key at `base_p + j·ratio` (and each query at its position) and takes the ReLU-summed dot product. The error is within the §5.4 table bound propagated through the dot product. The bound is derived in the test, not a tuned tolerance |
| `rot_scorer_agrees_with_todays_scorer` | Both scorers see keys rotated from **the same f64 source**. Today's scorer reads a staging the test builds by rotating on the host in f64. The new scorer reads the un-rotated rows. Scores agree within the propagated bound. **Top-k agrees except at near-ties**: any cell that differs has a score within the bound of the k-th score |
| `pages_straddle_warps_and_blocks` | Page boundaries at lane 0, lane 31, mid-warp and at block edges; pages of 1, 31, 32, 33 and 257 rows; a short last row (`last_cells < ratio`); CPT slots landing in different pages |
| `signed_page_offsets` | `δ_p < 0` (a short-rowed page ahead), `δ_p = 0`, and large `δ_p` near 2²¹ |
| `positions_cross_hi_rows` | Candidates whose positions cross `hi` boundaries (`pos mod 1024` wrapping) inside one warp |
| `every_tile_arm` | `(TILE_R, CPT)` ∈ {(4,2), (2,2), (2,1), (1,1)} at H = 4, and H ∈ {1, 2, 8} at one arm |
| `wide_row_arm` | Spans of 511, 512, 2,048 and 8,193 rows (either side of the arm's threshold and a ragged last row tile) match the oracle, and match the decode arms on the same inputs |
| `rungs_1_2_4` | The same keys scored at each rung equal the oracle at that rung. A launch never reads another rung's table: rung 1's table is poisoned while rung 2 runs |
| `tail_page_with_capacity_pitch` | A tail page with `pitch = capacity > rows` scores the same as a tight page |
| `mask_prefix` | `g ≥ cnt[r]` gives exactly `−1e30`, bit-identical to today's padding |

**Benchmark** (`bench_rot_scorer`, `#[ignore]`, run by hand and by `/sweep`):

- **Sweep:** depth {8K, 32K, 128K, 512K, 1M tokens} × query rows {1, 8, 64, 512,
  2,048, 8,192} × pages {1, 16, 256}. The three widest row counts are prefill
  chunks, where today's tail goes through cuBLAS.
- **The prefill cells** are the ones that choose the wide-row arm's threshold:
  the row count at which it overtakes the decode arms.
- **Arms compared:**
  - (a) today's scorer on a pre-built staging;
  - (b) today's full path: host query rope, placement, scorer, and cuBLAS plus
    reduce for the tail;
  - (c) the new scorer.
- **Timing:** CUDA events, 20 warm-ups and 200 timed launches; median and p90.
  The daemon is stopped and the card checked idle.
- **Printed alongside:** achieved L2 and DRAM GB/s from bytes moved, the
  straddling-warp share, and the registers and spills of each arm.
- **Results are recorded** in the §7.6 results table, with the commit and the
  card. A number with no commit is not evidence.

### 7.6 Optimisation: method, levers, gates

**Method.** This follows the repository's kernel rule: attribute first, from
ncu and SASS, then make **one** change and rebuild **once**. Nothing is
recompiled per hypothesis.

Each round:

1. ncu on the worst sweep cell: L2 %, DRAM %, issue-stall reasons, registers, spills.
2. One lever chosen from that evidence.
3. Rebuild, run the harness (correctness, then the three-run benchmark), and
   record the round in the results table.

**Levers, in the order the evidence is expected to call for them:**

1. **Split the channel loop** into rotary and pass-through halves (§7.4), so the
   pass-through half of `D` keeps today's body exactly.
2. **Re-tune the tile arms.** A second live `float4` and eight rotation values
   move the register budget. Re-measure `(TILE_R, CPT)` and the grid-fill rule
   (`QSA_PAGED_LAUNCH`, `:374-381`) rather than inherit them.
3. **Warp-uniform fast path.** Test once per slot per warp whether all 32
   candidates share a page (`__all_sync`). If they do, take the broadcast warp
   term; if not, the per-lane path.
4. **Hoist the rotation factors out of the channel loop.** Form each candidate's
   32 rotation factors once before the loop, rather than per channel group, if
   the register budget allows it.
5. **BF16 stored keys.** The reference caches its index keys in BF16
   (`docs/qwen38_flash_next.md` §3.1). Halving the key's bytes halves the
   dominant L2 stream. This is a separate win, since rotation adds no L2 traffic
   on the fast path.
   - It is a numerical change to the index, so it is gated separately: top-k
     agreement against the F32 path over the harness sweep, plus the Flash-Next
     needle gate with a dense control.
   - It is adopted only if both hold.

**Gates:**

- **Decode shape** (rows ≤ 8, any depth): the new scorer (c) is no slower than
  today's scorer on staging (a), within run-to-run variance.
- **Selection shape** (64 rows, 128K and above): (c) is at most 10% slower than
  (a) **and** faster than today's full path (b). Today also pays the query rope,
  the placement pass and the cuBLAS tail; the new path pays none of them.
- **Prefill shape** (512 to 8,192 rows, every depth): (c) is no slower than (b),
  within run-to-run variance. At this width (b)'s tail is cuBLAS, so this is the
  gate the wide-row arm answers to.
- **Every correctness test green,** and `device_selection_matches_the_cpu_oracle`
  (`indexer.rs:2779`) unchanged.

**If the rounds do not reach the gates,** the fallback runs in this order. Each
step is taken only on the evidence that the previous one fell short:

1. **BF16 stored keys** (lever 5), under its own numerical gate. It halves the
   bytes of the dominant stream on every shape.
2. **A transient rotated tile for the prefill shape only.** Rotate the span's
   candidate keys once into a scratch buffer for the duration of the launch, and
   hand it to cuBLAS as today. It is freed when the launch ends.
   - This keeps I1: nothing rotated is stored, and the rotation exists only for
     one read.
   - Decode and selection shapes keep rotate-on-load.
3. **If neither holds,** the result goes back to the design with its measurements
   before anything is built on it. The gates are not loosened to fit.

**Results** (filled in as rounds land):

| Round | Commit | Card | Change | 128K·8,192 rows (ms) | 128K·64 rows | 128K·8 rows | 8K·1 row | Regs / spills | Notes |
|---|---|---|---|---|---|---|---|---|---|
| baseline (a) | — | — | today's scorer on staging | | | | | 64 / 0 | |
| baseline (b) | — | — | query rope + placement + scorer + cuBLAS tail | | | | | | |

### 7.7 Records already on disk

The turn-record and snapshot blobs (`paged_index.rs:503-628`, `AUX_VERSION = 2`)
hold row-major rows rotated by `j·ratio` relative to each page's start.
`decode_page` always gives `roped_base = 0`.

- **`AUX_VERSION` goes to 3:** un-rotated and channel-blocked (§7.3).
- **The migration needs each layer's `ratio`.** The page header stores only
  rows, dim, `last_cells` and `n_open`, so the ratio is taken from the model
  config, in blob layer order. The pass therefore runs with the model loaded.
- **It runs once, blocking, at daemon start, before the first request is
  served.** It rotates row `j` of every v2 page back by `j·ratio` with the old
  f32-angle frequencies, re-lays the rows channel-blocked, and rewrites the blob
  as v3. A background pass would leave a window in which the v3-only reader
  meets v2 blobs.
- **The reader accepts only v3.** No dual read path is kept.
- **Section pages need nothing.** `section_positional` is process-local
  (`scheduler/mod.rs:2850-2855`).

### 7.8 Stale comments this work corrects

- `qsa_index_append.cu:79` cites a gate test `qsa_index_append_matches_host`
  that does not exist.
- `qsa_page_place.cu:46-52` says one placement covers every page of every
  layer. Every caller places one cache or one window. That file is deleted
  under §7.
- `latent_moe/paged.rs:233-238` says the CorpusCache is roped once. It holds
  pre-rotation `rope_bf` plus `comp_pos`.
- `latent_moe/gallery.rs:531` says indexer keys are pre-RoPE. They are stored
  rotated (`wave.rs:2139, 2595` via `Compressor::pool_and_norm`).
  - DeepSeek's schedule is static, so this cannot bleed.
  - It is outside §7, which covers the Qwen4Exp QSA index; DeepSeek's indexer
    keys remain rotated.
  - The comment is corrected to say so.

---

## 8. Choosing a slot's rung

### 8.1 The rung is a function of the slot's length

Promotion costs nothing: every table exists from load (§5.3), and the QSA index
is rotated on load (§7). So there is no rung state to keep and no promotion
step. The rung is computed each time the session serialises the slot's headers
for a wave:

```text
rung = rung_for(schedule, offset + input_len − base_pos)
     = the lowest rung whose ceiling ≥ that reach
```

- `offset + input_len` is the deepest position this wave writes.
- `base_pos` is 0 except on a sliding-window ring (`sequence_ops.rs:199-214`),
  whose evicted front shifts every position but no distance. RoPE depends only
  on distance.

**Consequences:**

- **Monotone without a rule.** A slot's offset only grows between rebuilds, so
  its rung only rises. A projection rebuild re-derives it from the rebuilt
  length.
- **Rungs rise only when needed.** A slot switches rung exactly when it crosses
  a ceiling, never on a budget it might not use. A Qwen3 slot at 17K keeps rung
  1, even though `max_response_tokens` (16,384 by default) could carry it past
  32K.
- **One rung per wave.** The header is serialised once per wave, so speculative
  verify and the MTP draft read the same rung as the forward.

### 8.2 The ceiling

A reach above `schedule.supported_max()` is **refused** when the wave is
admitted, with an error naming the model, the reach and the maximum. Nothing
enforces a window today (`builder.rs:976, 999, 1037` discard it), so this is new
for every schedule.

### 8.3 Why discrete rungs, not a continuous factor

A continuous `s = reach / L₀` gives every slot its own frequency set. That means
a table per slot, built on the fly, and factors Qwen has not published. Discrete
rungs keep the tables prebuilt and the factors vendor-backed.

---

## 9. Host builders and the defects

These all take a `&RungFreqs` or the rung buffer, instead of a θ:

- `RotaryLayout::rope_cos_sin`;
- the per-row host cos/sin behind `BatchedAttentionParams::rope_cos/rope_sin`;
- the float fallback;
- the oracles: `qwen35/attention.rs`, `qwen4exp/loader.rs:344` (capped at
  65,536 positions) and `latent_moe::rope::RotaryCache`.

`BatchedInference::new_with_inv_freq` becomes `new_with_schedule`.

**Deleted:**

- `compute_rope_cs` and `RotaryLayout::rope_table`;
- the four per-arch `*_inv_freq` helpers;
- `infer_rope_scaling_factor` and its Qwen2 twin;
- `RopeTables` and the host indexer query rope;
- the θ loops at `qwen35/batched.rs:151-156`, `qwen4exp/wave.rs:1143-1147` and
  `qwen35/attention.rs:78`;
- the per-layer DeepSeek tables;
- the placement kernel and its staging;
- the scorer's generic arm.

**Llama3 frequencies.** The source is decided once, at load, in this order:

1. **`rope_freqs.weight`**, when the file has it (one divisor per pair,
   llama.cpp's form). It is the file's own statement of its scaling.
2. **Otherwise, the preset's published parameters** (factor, low and high
   frequency factors, L₀). The preset's schedule is the model's property
   (§2), and a conversion that dropped the tensor has not changed the model it
   converted.
3. **A custom model with neither** gets plain RoPE, with a warning naming the
   missing scaling. Nothing supplies its parameters.

Whether our Hermes files carry the tensor is checked by a test (§10), which
also checks that the tensor agrees with the published parameters. It is not
checked again at load.

---

## 10. Tests and gates

The unit tests follow the repository's TDD rule: raw expected values next to
the code.

| What | Where | Asserts |
|---|---|---|
| YaRN frequencies and `m` | `rope_schedule/yarn.rs` | The full `inv_freq` vectors for §4.1's two rows at s = 2 and 4, as raw values transcribed from vLLM's `_compute_inv_freq`; `low`/`high` as tabled; `m` = 1.0693…, 1.1386… |
| DeepSeek unchanged | `latent_moe` | The moved `yarn_freqs` returns identical vectors for `(160000, 65536, 16, 32, 1)`; the shared table is byte-equal to today's per-layer table; the DeepSeek forward gate passes |
| Llama3 | `rope_schedule/llama3.rs` | Records whether both Hermes presets' files (and the Nidum file) carry `rope_freqs.weight`; where one does, it equals the published-parameter formula; a Llama3 preset without the tensor builds exactly the published-parameter frequencies |
| `SlotHeader` layout | `slot_types.cuh` + `slot_state.rs` | `sizeof == 32` and every field offset agree between CUDA and Rust; decode, prefill, glue and latent kernels all read a 32-byte stride |
| Qwen3 linear gone | `quantized_qwen3.rs` | Our 40,960-declaring Qwen3-8B GGUF builds rung-1 `ω` exactly `1/θ^(2i/d)` |
| Factored table exactness | `rope_schedule/table.rs` + GPU | Device builder = CPU builder for every entry; device `rope_lookup` = CPU lookup mirror bit-for-bit at positions {0, 1, 1023, 1024, 1025, 2¹⁰·k ± 1 for k up to 2047, 2²¹ − 1} for P = 32 and 64 |
| **Factored table accuracy** | `rope_schedule/table.rs` | Max \|lookup − f64 (sin, cos)\| over a sweep of every `hi` row × sampled `lo` rows, for each model's rung-1 and rung-4 `ω`. The bound is measured, then pinned as a raw constant |
| Pass-through pairs | `table.rs` | Pair ≥ P returns exactly `(0, 1)` |
| Lane tables | `table.rs` | Each (rung, ratio) lane table equals the factored lookup at `L·ratio` for every lane and pair |
| Tile kernel unit step | `int8_decode_tile_kernel` test | A group walked from `rope0` by the slot's `LO[1]` equals per-token lookups at the slot's rung. Also run with the slot on rung 2 **and the rung-1 table deliberately placed first**, which catches a step taken from the wrong rung |
| **No bleed, per kernel** | every attention kernel's test module | One launch with sequences on rungs 1, 2 and 4 **interleaved in the wave's order** matches each sequence run alone at its own rung, bit for bit. This covers decode (split-KV, stripe, bmma, tile), int8 prefill with sequences of unequal `q_len`, and glue. It mirrors `a_row_s_ban_stays_in_its_row_in_a_shared_launch` |
| No bleed, QSA | `tests/qsa_score_rot_harness.rs` | Consecutive scorer launches for sequences on different rungs each match their own oracle (`rungs_1_2_4`, with the other rungs' tables poisoned) |
| One `m²`, on Q's rotary dims | every attention path | At rung 2: rotary-dim logits scale by exactly `m²` (not `m`, not `m⁴`); pass-through logits unchanged; stored K bytes identical across rungs; `softmax_scale` unchanged |
| Rung bounds | kernels | A header rung ≥ `n_rungs` traps |
| Selection | `rope_schedule/select.rs` | reach → rung at each ceiling and one past it; refusal past `supported_max` for every schedule |
| QSA stored rows | `qwen4exp/indexer.rs` | Rows after append equal the reference's pooled normed un-rotated key; a seal is a byte copy of stored rows; `device_selection_matches_the_cpu_oracle` (`indexer.rs:2779`) passes unchanged, since the oracle already ropes at read time |
| QSA rotate-on-load scorer | `tests/qsa_score_rot_harness.rs` | The full §7.5 correctness set |
| v2 → v3 migration | `persistence` | A v2 page migrated to v3 and scored by the new scorer at base b selects the same blocks, near-ties aside, as the v2 page placed and scored by today's path |

**GPU gates.** These run one `cargo` process per model, with the daemon
stopped, as `--ignored` on the named test only (never on the zend suite):

1. **Numerical.**
   - All twelve model forward gates pass.
   - Perplexity at 8K and 32K is within the pre-change build's measured run-to-run noise for every GQA model.
   - Qwen3-8B and Hermes are excluded, because §3.1's fixes change them on purpose. They are re-baselined.
2. **Throughput.** Decode and prefill t/s at 8K, 32K and 128K on the 4090 Mobile
   and on this card, three runs each. Every rate must be within the pre-change
   build's measured variance or better. At 128K the factored table is expected
   to gain, because today's rows there come from DRAM.
3. **Long context, with a dense control.**
   - Qwen3-30B-A3B at 48K, 64K and 112K.
   - Qwen3.5-0.8B at 384K and 512K. This is the gate that backs its unpublished rungs.
   - Qwen3.8-Flash-Next at 384K and 512K.
   - The gate: progressive beats rung-1 extrapolation at every length past L₀,
     and matches today below it.
4. **Mixed waves.** Two conversations in one wave, one on rung 1 at 8K and one
   on rung 2 at 64K (Qwen3-30B-A3B). Each produces the same tokens as when it
   runs alone.
5. **Crossing a ceiling.** A Qwen3-30B-A3B conversation at 30K that grows to
   40K through tool results switches rung once and still retrieves a needle
   placed before the switch.

---

## 11. Build order

Each step lands green on its own gates before the next starts.

1. **`rope_schedule/`**: `yarn.rs`, `llama3.rs`, `schedule.rs`. DeepSeek moves
   onto it. Frequencies only, with no behaviour change.
2. **The §3.1 defects.** Hermes reads `rope_freqs.weight`; the Qwen3 and Qwen2
   linear inference is deleted. These are independent correctness fixes, in
   their own commit.
3. **The factored table, engine-wide** (§5):
   - the shared `rope_table.cuh`;
   - every paged kernel and the host paths read it;
   - DeepSeek's per-layer tables collapse to two;
   - every slot is on rung 1.

   This is a numerical change, gated by §10 gates 1 and 2.
4. **QSA rotate-on-load** (§7), in this order:
   - (a) the §7.5 harness, with baselines (a) and (b) recorded against the shipped path;
   - (b) the rotating scorer, **both** its decode-shaped arms and the wide-row
     arm, correct against the harness before any tuning;
   - (c) the §7.6 optimisation rounds until the gates hold, or the §7.6
     fallback;
   - (d) un-rotated channel-blocked storage, deletion of placement, the cuBLAS
     tail and the host query rope, and the v2 → v3 migration.

   This stands on its own: it is simpler and more precise even with no rung
   above 1.
5. **Rungs:** `RopeRungs`, `SlotHeader.rope_rung`, `rung_for` at header
   serialisation, the ceiling refusal and the presets' schedules. Then the bleed
   and `m²` tests.
6. **Gates 3, 4 and 5.**

---

## 12. Settled: the QSA indexer takes YaRN **and** `m`, exactly as attention does

**Question.** Does Qwen's reference apply YaRN, and `m`, to the QSA indexer's
rotation? It matters for ranking, not just scale: only 64 of the indexer's 128
dims rotate before the ReLU.

**Answer: yes, both.** The indexer has no rotary embedding of its own. It rotates
with the attention layer's, including the YaRN frequencies and the `m` baked
into its cache. The chain, from vLLM's `main` (`vllm/models/qwen4_exp/nvidia/`):

1. **The attention layer builds the only rotary embedding** from the model's
   rope parameters, which is where a YaRN override lands (`qsa.py`):

   ```python
   self.rotary_emb = get_rope(
       head_size=self.head_dim,
       max_position=config.max_position_embeddings,
       rope_parameters=config.rope_parameters,
   )
   ```

2. **It hands that object to the indexer** (`qsa.py`):
   `QSAIndexer(..., rotary_emb=self.rotary_emb, ...)`.
3. **The indexer rotates its queries and its pooled keys with it, cache as-is**
   (`indexer_qsa.py`, `apply_qsa_rope`, documented as "Apply the main
   attention's exact 1D/MRoPE composition to QSA heads"):
   - `cos_sin = rotary_emb._match_cos_sin_cache_dtype(tensor)[positions]`, then
     rotation over `rotary_emb.rotary_dim`;
   - queries at their positions (`q = apply_qsa_rope(self.rotary_emb, positions, q)`);
   - pooled keys at their block's first position
     (`apply_qsa_rope(self.rotary_emb, first_positions, compressed_keys)`);
   - the fused path passes `self.rotary_emb.cos_sin_cache` to `qsa_pre_indexer`
     directly.

   Nothing divides out a scale or recomputes cos and sin.
4. **That cache carries `m`.**
   - With `mrope_section` in the parameters, `get_rope` builds `MRotaryEmbedding`.
   - With a YaRN `scaling_factor`, `MRotaryEmbedding` sets
     `self.mscale = yarn_get_mscale(scaling_factor)` (or `attention_factor`, if
     given). It then delegates `_compute_cos_sin_cache` to
     `YaRNScalingRotaryEmbedding._compute_cos_sin_cache`, which writes
     `cos = freqs.cos() * self.mscale` and `sin = freqs.sin() * self.mscale`
     (`rotary_embedding/mrope.py`, `rotary_embedding/yarn_scaling_rope.py`).
5. **The main attention's softmax scale stays `head_dim ** -0.5`** (`qsa.py`).
   `m` reaches the logits only through the rotated query and key, exactly as
   §4.2 assumes.

**Consequences for this design:**

- **The indexer shares the attention's rung table**, as §5.3 already specifies:
  the same YaRN `ω`.
- **The indexer applies `m²` to its query's rotary pairs.** The reference scales
  the rotated query and the rotated key by `m` each, so their rotary dot product
  gains `m²` and the pass-through half is untouched. Putting `m²` on the query
  alone, with keys stored and rotated at unit magnitude, gives the same dot
  product (§4.2). The site is the scorer's query-tile load (§7.4).
- **The QSA parity test pins it.** At rung 2 the indexer's rotary-dim
  contribution scales by exactly `m²` and the pass-through contribution does
  not. The oracle rotates both query and key through a cos/sin scaled by `m`, as
  vLLM does.
- **The Qwen card's YaRN block sets no `attention_factor`,** so
  `m = 0.1·ln(s) + 1`, the value §4.1 tabulates.
