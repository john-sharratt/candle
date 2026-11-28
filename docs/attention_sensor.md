# The attention sensor

Every model counts what its attention reads. At a few sampled attention
layers, each query's softmax mass over the slot is measured per chunk, the
chunks holding most of that mass are the query's **hits**, and the counts per
round feed the hit-driven projection (`docs/hit_driven_projection.md` §6–§7)
exactly as QSA's selection did. A model with no QSA index — Qwen3.6, the GQA
models, DeepSeek-V4's latent engine — gets the hit-driven slot fill, the
knowledge base's per-step ordering (`docs/placed_section_layout.md` §3.2,
§3.4) and the hit momentum, and the code that ran those paths only "when the
model selects" runs them always.

## 1. Why

The hit-driven projection reads one input from the model: per declared round,
how often the model read it (`PieceHits`, through
`ManagedBatchedModel::take_hits`). Only QSA produced it: each block a QSA
selection chose at a sampled layer was one hit (`qsa_hits.cu`), and a shadow
scan scored unloaded rounds' index rows (`qsa_shadow.cu`). Every other model
answered `None`, so:

- `take_hits` is `Option`, and every reader has an arm for a model with
  nothing to read (`scheduler/section_step.rs`, `reproject_prepare.rs`,
  `first_reading.rs`, `projection_assembler/apply_build.rs`).
- A conversation's trained window is set only for a model that
  `carries_positional_state()` (`scheduler/construct.rs`), so on Qwen3.6 the
  dialogue is filled by its layers' windows, not as a slot.
- The knowledge base's resident set admits a section without index pages only
  on a model that cannot rate it at a step (`scheduler/kb_residency.rs`).

The QSA indexer is a trained predictor of where attention goes. The softmax
is where it went. Measuring it is available to every model, and on a QSA model
it is ground truth for what the selection let attention read.

## 2. The design in one page

| Part | What it is |
|---|---|
| **Sampled layers** | `hits.layers` percentiles over the model's **attention** layers (Qwen3.6: one layer in four attends; Flash-Next: its selecting layers, as before). |
| **Sensor rows** | Every decode row each step, and the last `KEEP_SLOTS` (64) rows of each prefill span — the queries nearest the reply. |
| **Mass** | Per sensor row, per query head: the softmax of `q·k / √d` over every position attention read — the slot for a dense model, the selected entries for a QSA model — summed per 32-token chunk and averaged over the row's query heads. |
| **Hits** | The row's **nucleus**: the fewest round-owned chunks holding `NUCLEUS` (0.9) of the mass the row gave to declared rounds. Each is one hit for its round. The lowest mass in the nucleus is the row's **cut**. |
| **Shadow** | Resident sections out of the range, whose K is hot (§3.5 residency), scored with the same rows: a chunk whose mass — over the row's own softmax denominator — reaches the row's cut is one hit for its round. Flash-Next keeps its QSA pool shadow beside it (§6). |
| **Counting** | One counter module for every model: the block→round map, per-round counters, the publish into a mapped summary block the host reads without waiting. Moved out of `qwen4exp/hits.rs` and `qsa_hits.cu`. |
| **Interface** | `set_hit_layers`, `set_hit_rounds` and `take_hits` are required of every model; `take_hits` answers `PieceHits`. |

## 3. The sensor

### 3.1 Where it runs

In `batched_layer`, after the attention of a sampled layer, in the decode path
(`paged_decode_attention`) and the prefill path (`prefill_attn`) alike: one
launch per sampled layer per wave, over every sequence's sensor rows (hot-path
invariant 5), each sequence named by a span table entry — its rows, its
slot's K descriptor table, its chunk→round map, its counters (invariant 2b).
Qwen3.6, Qwen3.5 and the GQA models attend through these functions, and so
does Qwen4Exp, whose selection arrives there as `QsaSelection`; DeepSeek's
latent engine runs its own kernels and its own launch of the same sensor (§7).

The attention kernels are not changed. The sensor reads what they read — Q
as the layer passes it (unrotated, rotated in the sensor at each slot's rung,
as the attention kernel rotates it), and K through the same per-slot arena
descriptor tables and K tile loaders (`paged-decode/int8_decode_kernel`,
`k_v_decode.cuh`), so every arena format is read alike — and writes only its
counters. Attention output bytes are what they were.

### 3.2 Mass and the nucleus

For row `r`, query head `h`, position `p` attention read:

    s(h, p) = q_h · k_{g(h)}(p) / √d          (g: the head's KV group)
    m(h)    = log Σ_p exp s(h, p)             (the row's own denominator)
    a(h, p) = exp(s(h, p) − m(h))
    A(c)    = mean_h Σ_{p ∈ c} a(h, p)        (chunk c's mass)

A chunk belongs to a round through the map; a chunk no declared round owns —
the system prompt, the query's own tail, pinned ranges — is no round's and
takes no hit, as `QSA_NO_PIECE` blocks take none today. The nucleus is taken
over round-owned chunks only: sorted by `A(c)` descending, the fewest whose
sum reaches `NUCLEUS · Σ_{c owned} A(c)`. A row whose round-owned mass is
below `MASS_FLOOR` (0.02) reads nothing declared and counts no hits, though
it counts as a query.

`NUCLEUS` and `MASS_FLOOR` are constants of the sensor, measured against
Flash-Next's QSA hits (§8), not schema settings.

### 3.3 Units

A hit is per chunk (32 positions), where QSA's is per block (four cells). The
consumers divide hits by the blocks a round owns and the queries counted
(`hit_density`), so `PieceHits.blocks` carries chunks for a sensor count and
the densities are fractions on one scale.

### 3.4 The shadow

A resident section out of the range has its K hot on the device and no
position in the slot. Its chunks are scored with the same rows as if appended
behind the query's context: `a(h, p) = exp(s(h, p) − m(h))` over the row's own
denominator, so its mass is on the slot's scale without entering it. A chunk
whose `A(c)` reaches the row's cut is one hit for its round — it would have
displaced the weakest chunk the row did read, which is QSA's rule (§6.3)
stated in mass. Shadow rows are counted into `shadow_queries`.

Rounds whose K is cold — the pool — have no shadow from the sensor and are
judged by provenance at boundaries, as they are on npcd today.

## 4. Counting

`models/attention_hits/` holds what `qwen4exp/hits.rs` and `qsa_hits.cu` held
that is not QSA's: the declaration (`set_hit_rounds` → chunk→round map,
counters, generation), the views that count into a parent's counters, the
publish launch and the mapped summary block (`hit_summary.rs`), and `take`
(the difference since the last read). The sensor kernel adds into its
counters; Qwen4Exp's QSA launch keeps the stride's indexer queries for its
shadow scan and no longer counts loaded hits.

## 5. What goes

- The no-op defaults of `set_hit_layers`, `set_hit_rounds` and `take_hits` on
  `ManagedBatchedModel`; `take_hits` returns `Result<PieceHits>`, and the
  `Ok(None)` arms of its four readers.
- The `carries_positional_state()` gate on the trained window
  (`construct.rs`): every conversation's trained window is the model's first
  RoPE ceiling.
- The `rated` gate in `kb_residency.rs`: every model rates its resident
  sections at a step.
- "A model whose attention reads every position takes none of it" — in
  `managed_model.rs`, `hit_layers.rs`, the model builder, this repository's
  docs (`hit_driven_projection.md` §1, `npc_mind_layout.md` §3).
- `HitPercentiles` names percentiles over attention layers.

What stays: every other `carries_positional_state` use — QSA's index pages
travel with K/V and that is a capability, not a branch — and the selection
prompt and pins, which a dense attention has nothing to do with.

## 6. Flash-Next

Flash-Next's loaded hits come from the sensor over the entries its selection
chose. Its pool shadow — the pooled rounds' index rows at the sampled layers,
scored at each boundary by `qsa_shadow.cu` — stays, as the one input only a
QSA model has; its counts go into the same counters beside the sensor's
resident shadow.

## 7. DeepSeek-V4

The latent engine attends through its own paged latent kernels (K≡V, the
latent geometry of `latent_moe/geometry.rs`). Its sensor is the same
computation over its latent K and its own descriptor tables, launched from its
wave at its sampled layers, counting into the same module.

## 8. Tests and the gate

- **CPU oracle, raw values**: a sensor launch over a hand-built slot — two
  rounds, the prompt, a tail — against the counters a CPU reference computes
  from the same dequantized K: exact hit counts, cuts and query counts, for
  every arena format the slot holds, for decode rows and prefill rows.
- **Units**: the nucleus over a fixed mass vector; a chunk no round owns takes
  no hit; a row under `MASS_FLOOR` counts a query and no hit; a shadow chunk
  at, above and below the cut.
- **Agreement on Flash-Next** (BlackWorld): the same strides counted by QSA's
  selection and by the sensor, per-round densities compared; `NUCLEUS` is set
  where they agree best, and the agreement is recorded in
  `docs/performance.md`.
- **Cost**: the sensor's share of a decode step at npcd's prompt and at 262K,
  per machine, in `docs/performance.md`.
- **Live**: npcd on Qwen3.6, a reprojection's `hit_summary` reporting loaded
  hits, and the knowledge base reordering at a step.
