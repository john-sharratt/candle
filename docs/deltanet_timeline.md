# DeltaNet timeline: one state buffer, and a partial accept that picks a row

**Status:** implemented (2026-10-18, branch `more_tuning`), after a full audit
against the code (2026-10-17, §11) with its open questions decided (§12). §13 records
what the implementation settled that the design left open, and what it measured.
Where the code and this document disagree, this document is the authority until it
is amended.

**Scope:** the Gated DeltaNet recurrent state of every hybrid model —
`qwen35`/`qwen36`/`qwen38` (`models/qwen35/`) and Qwen3.8-Flash-Next
(`models/qwen4exp/`). The loaders `quantized_qwen35.rs`, `quantized_qwen36_moe.rs`,
`quantized_qwen38.rs` and `quantized_qwen38_moe.rs` build these two stacks and hold
no state of their own. The attention half is unchanged; the PLE window, the QSA
index cache and the draft seed keep their rewinds but lose the row layout they
borrow from the stash (§6.6).

## 1. Summary

Every DeltaNet layer of every sequence holds its recurrent state twice today: the
live `S` and conv tail, and a second copy the next wave writes into
(`delta_net/state_store.rs`). The second copy exists for two reasons — a failed
wave must leave no trace, and a speculative block accepted at `m` of its `k + 1`
rows must rewind — and the rewind adds a third structure, a per-row operand stash
(`VerifyStash`, `SlotTenant::RewindStash`) that a replay (`replay_accepted_prefixes`,
`delta_net/replay_stack.rs`) re-runs the conv and the mixer over.

This design replaces all three with **one state buffer per layer and a short
timeline reserved at the end of it**:

- A short span (at most `R = 8` rows: a decode row, a verify block, a short prompt)
  never writes `S` or the conv tail. It writes one **timeline row** per token: the
  token's contribution to the state, in the factored form the kernels already
  compute.
- Once a span's kept length `m` is known, a batched **fold** applies the first `m`
  rows of its timeline to its state, in place. A partial accept is a choice of `m`;
  a full accept and any span that is not being verified fold `m = len`; a failed
  span folds nothing.
- Long spans (prefill chunks) keep a second state buffer, but only while the
  sequence is prefilling (§7).

What it removes: the resident second state copy of every decoding sequence, the
rewind stash and its slot tenant, the replay (conv, solve and mixer over the
accepted rows), and `layer_state_rewind`. On Flash-Next the `state` band of the
batched gate is predicted to fall from 2,048 MiB to about 1,216 MiB at ×8 (§5).

## 2. What one token does to the state

Per V head, the state is a `d_v × d_k` matrix (`[n_v_heads, d_v, d_k]`, F32). The
decode kernel (`delta_net_decode_kernel.cuh`) advances it by one token as:

```
decay  = exp(g)                       g = a · softplus(alpha + dt_bias) ≤ 0
err[i] = β · (v[i] − Σ_j S[i][j] · decay · k[j])
S[i][j] ← S[i][j] · decay + err[i] · k[j]
o[i]   = Σ_j S[i][j] · q[j]           post-update read
```

**Every element of `S` changes, and the change carries almost no information.**
The whole update of a head is one scalar (`decay`), one `d_v` vector (`err`) and one
`d_k` vector (`k`) — and `k` belongs to the K head, shared by every V head of its
GQA group. At `d_k = d_v = 128` that is 257 numbers against a 16,384-element head:
a token's effect is a scalar decay plus a rank-1 matrix.

Two consequences decide the representation:

- **Diffing, deduplicating or compressing `S` itself does not work.** The decay
  scales every element, so an element-wise difference between consecutive states
  is a dense matrix of unrelated floats — nothing repeats, nothing is sparse.
- **Storing the generators does.** The update is fully described by
  `(decay, err, k)`, and `k` dedupes across the GQA group. Lossy compression of the
  generators is excluded: `S` is a running sum, so a rounded generator becomes
  permanent error in every later token.

## 3. A span is already a timeline

A multi-row span of at most 8 rows runs through the short-span kernel
(`delta_net_short_span_kernel.cuh`), which uses the chunked (WY) form. Its state
phase writes:

```
S_len = e^{G[len−1]} · S_0 + Σ_{t<len} v_new[t] ⊗ (k[t] · e^{G[len−1] − G[t]})
```

where `S_0` is the entering state, `G` the within-span inclusive scan of the
log-decays, `k[t]` the normed key, and `v_new[t] = u[t] − w[t] · S_0ᵀ` from the
forward-substitution solve. Every term is anchored at `S_0`.

### 3.1 Prefix stability

The row quantities of row `t` — `G[t]`, `k[t]`, `v_new[t]` — depend only on rows
`0..=t`, bit for bit, whatever the span's length (audited, §11.1):

- **`G`:** the Hillis–Steele scan adds slot `t − off` into slot `t` at each
  doubling; lanes at or past `len` start at zero and only feed higher lanes; at
  `off = 32` every lane adds `+0` (`short_span:230-247`; the three-kernel scan,
  `prefill_kernel.cuh:476-482`, does the same).
- **Conv and norm:** each row's window is a fixed `j`-ascending chain over its own
  inputs, and the l2-norm tree is per row (`short_span:152-159, 206-273`).
- **`A` and `kq`:** entry `(i, j ≤ i)` is one d-ascending FMA chain over rows `i`
  and `j`, times `e^{G[i] − G[j]}` (`short_span:304-326`).
- **The solve:** `x[i] = b[i] − Σ_{j<i} A[i][j] · x[j]`, reading only rows `j < i`;
  `len` only stops the loop (`short_span:343-351`).
- **`v_new[t]`:** `u[t] − w[t] · S_0ᵀ` — `len` is only a guard (`short_span:366-374`).

Only the state phase's final factors depend on `len` — `e^{G[len−1]}` and
`e^{G[len−1] − G[t]}` (`short_span:287-293`) — and both are computed from the
stored `G`. So **the state for the first `m` rows is the state phase with
`len = m` applied to the stored rows.** The conv tail follows the same way: after
`m` rows it is the last `K − 1` raw conv inputs of `[tail_0 | x[0] .. x[m−1]]`,
exactly what the kernel's tail writer computes for `len = m` (`short_span:190-201`).

### 3.2 The reference this is exact against

Today's replay launches the **three-kernel prefill path** (conv, intra, state;
`replay_stack.rs:280-330`), not the short-span kernel. The two are equal bit for
bit by test, not by construction:
`a_stacked_replay_is_each_layers_wave_launch_bit_for_bit` (`replay_stack.rs:411-581`)
asserts the short-span state and tail against the stack. The fold is therefore held
to the three-kernel state pass over `m` rows, which keeps its state-out (it runs
long spans, §7) and is the arithmetic today's replay is locked to. That is the
oracle of §10, test 1.

### 3.3 Rounding is pinned with intrinsics

The DeltaNet kernels compile with `--use_fast_math` (`candle-kernels/build_utils.rs:261`,
the `delta_net` group at `:586-594`): FMA contraction is free (`-fmad=true`),
denormals flush, and `expf` is `__expf`. The state phase's bits today depend on
contraction the compiler chose (`acc += vv·kg` → `fma(vv, kg, acc)`,
`s·dec + acc` → `fma(s, dec, acc)`), which a shared helper does not guarantee
across two kernels. So:

- the fold and the three-kernel state pass both write the phase as explicit
  `__fmaf_rn` / `__fmul_rn` and call `__expf` for `e^{G[len−1]}` and
  `e^{G[len−1] − G[t]}`;
- the fold is compiled in `delta_net_api_f32.cu`, under the same flag group;
- the first accumulation is written as the kernels do it today (`acc = 0`, then
  `__fmaf_rn(v, kg, acc)`), so the `+0`/`−0` of an all-zero entering state comes out
  the same everywhere.

## 4. The slot

A layer's slot holds, in order, each on a `SLOT_ALIGN` boundary:

| Region | Shape (F32) | Held |
|---|---|---|
| `S` | `[h_v, d_v, d_k]` | always |
| conv tail | `[conv_dim, K − 1]`, channel-major | always |
| timeline | `R` rows | always; meaningful only between a span's forward and its fold |

`R = DNS_ROWS = DELTA_NET_SHORT_SPAN_ROWS = 8` (`api.rs:33`): the longest span the
short-span kernel runs. Every span that kernel takes keeps a timeline and every
longer span is a long span (§7), so the short-span kernel loses its state-out and
tail-out entirely. Production verify blocks are at most 5 rows (a draft ceiling of
4 in every shipped ladder, `draft_ladder.rs:222, 268, 331, 420`; a ceiling of 5
measured slower, `docs/possible-performance-boost.md`). The other rows serve short
prompts and leave room for a ceiling up to 7. Draft depth is bounded at `R − 1 = 7`
everywhere a block can be formed, and a deeper block is refused (§6.2, §12 D1).

One timeline row, at row `t`:

| Field | Shape | Written by | Read by the fold for |
|---|---|---|---|
| `G[t]` | `[h_v]` | short-span: block of V head `h`, tile 0 (every block holds `sg`, `short_span:243-246`); decode: the decode-step kernel | the decays |
| `k[t]` | `[h_k, d_k]`, normed, unscaled | short-span: block of V head `h = kh < h_k`, tile 0 (every block holds its K head's 128 channels, `:282-284`); decode: the decode-step kernel | the rank-1 terms |
| `v_new[t]` | `[h_v, d_v]` | short-span: each (V head, d_v tile) block, its own 32 `d_v` rows (`vnew`, `:372`); decode: `err` from the decode-step kernel | the rank-1 terms |
| `x[t]` | `[conv_dim]`, raw conv input | short-span: spread over the span's blocks, one writer per element (the tail writer's pattern, `:190-201`); decode: the conv-decode kernel, which is the only decode kernel that reads raw `x` (`decode_kernel.cuh:203`) | the conv tail |

`k` is stored once per K head and `x` once per row — the two dedupes the structure
allows. `x` is the widest field (10,240 of 18,480 floats per row on Flash-Next); it
cannot be read back from the wave instead, because the fold of a verify block runs
after the forward, at the reconcile, and the wave's per-layer QKV buffers are gone
by then (§11.1, item 17).

The slot is one `SlotTenant::RecurrentState` claim per layer, as today; only its
stride grows. Compaction, fork, export, eviction and reset move or read the slot
whole. `relocate` (`state_store.rs:977`) copies only `S` and the tail: nothing is
pending between steps (§6.4), so the timeline region holds nothing to keep.

## 5. Size

Per layer, `D = d_k = d_v`:

```
S        = h_v · D²
tail     = (K − 1) · conv_dim,          conv_dim = (2·h_k + h_v) · D
row      = h_v + h_k·D + h_v·D + conv_dim
```

**Today** a sequence holds `2 · (S + tail)` per layer, and each verifying **cohort**
— not each sequence — holds one rewind stash of
`(Σ cohort verify rows) · (conv_dim + h_v·D + 2·h_v)` per layer, grow-only and reused
across steps (`qwen4exp/spec.rs:169-195`, `qwen35/spec.rs:194-215`). **With the
timeline** a decoding sequence holds `S + tail + R · row` per layer and there is no
stash.

Flash-Next's geometry (`qwen4exp/config.rs:410, 441-445, 482`): `D = 128`, `h_k = 16`,
`h_v = 48`, `K = 4`, 36 DeltaNet layers of 48. Per layer: `S` = 3,145,728 B, tail =
122,880 B, one row = 73,920 B, so today's block is 3,268,608 B and the timeline
block 3,859,968 B.

**Region packing decides the real figure.** State slots come from 16 MiB regions.
Today five blocks fit a region (2.6% unused); with an 8-row timeline four fit (8.0%
unused). The float saving is 44%; the region saving is 41%:

| Flash-Next, decoding sequences | today | timeline | saving |
|---|---:|---:|---:|
| per sequence, region-effective | 230.4 MiB + stash share | 144.0 MiB | |
| `state` band ×1 | 352 MiB (22 regions) | 208 MiB (13) | 41% |
| `state` band ×4 | 1,072 MiB (67) | 640 MiB (40) | 40% |
| `state` band ×8 | 2,048 MiB (128) | **1,216 MiB (76)** | **40.6%** |

The `state` band is every slot tenant (`vram_bands.rs:75`, `region_pool.rs:3206`):
at ×8 it is 1,856 MiB of recurrent state (91%), 128 MiB of rewind stash (6%) and
64 MiB of PLE histories, draft seeds, QSA pages and gallery (3%). Each step the
band takes in the CSV — 304 → 352 MiB at ×1, 1,920 → 2,048 at ×8 — is the stash
claimed at the first verify.

Other models (`docs/archived/qwen35_qwen38_models.md`):

| Model | DN layers | blocks a region today → timeline |
|---|---:|---:|
| 27B (48 V / 16 K) | 48 | 5 → 4 |
| 35B-A3B (32 V / 16 K) | 30 | 7 → 6 |
| 0.8B (16 V / 16 K) | 18 | 14 → 11 |

On Flash-Next any `R` from 2 to 12 packs four to a region; `R` stays at 8 (§12 D1).

## 6. The step

### 6.1 Forward

A **short span** reads `S` and the conv tail and **writes neither**. It writes its
output rows and its timeline rows.

- **The kernel is chosen per span.** Today it is chosen per wave: the short-span
  kernel runs only when *every* multi-row span of the wave fits it
  (`mix.rs:1723-1726`), and a creep segment routinely packs verify blocks beside long
  creep spans (`wave_driver.rs:1213-1224`). The span table is split per layer into a
  short table (short-span kernel, timeline out) and a long table (conv + prefill
  scan, state out to the scratch, §7): one more launch on a layer whose wave carries
  both.
- **The short-span kernel** writes `G`, `k`, `v_new` and `x` per row (§4), computes
  the state phase's reads (`inter`, `v_new` → `o`), and stores no state and no tail.
- **The decode path is two kernels, and both change.** The conv-decode kernel
  writes `x[0]` into the timeline in place of its tail shift (`decode_kernel.cuh:213-219`).
  The decode-step kernel writes `G[0] = g`, `k`, `v_new[0] = err`, and computes its
  in-register update in the fold's exact operations at `m = 1` — `kg = k · __expf(g − g)`,
  `acc = __fmaf_rn(err, kg, 0)`, `S' = __fmaf_rn(S, decay, acc)` — so the state `o` is
  read from is the state the fold stores (§11.1, items 6 and 10). `g` already
  matches `dn_decay_gate` bit for bit (`decode_kernel.cuh:95`, `common.cuh:45-47`).
  Every `len == 1` span takes this path, a one-token prefill included
  (`mix.rs:1705-1709`).
- **Tables.** A decode row's and a short span's rows are `[tail, state, timeline]`;
  a long span's keep four rows, `[tail in, tail out, state in, state out]`, the outs
  pointing at the scratch. The short and long tables of one layer are views of **one
  pointer upload and one extent upload** (`build_span_tables`): the tier prices four
  pointers and two extents per multi-row span as one carve each
  (`WaveBuffer::DeltaNetLiveSpanPtrs`/`…Extents`), and two tables uploaded apart
  would be four carves whose alignment overruns that price. The refusal of a shared
  entering/advanced tail applies to long spans only. Tables are rebuilt every
  forward and decode graphs are re-recorded and updated in place each wave
  (`docs/decode_graphs.md:153-163`), so no slot address has to outlive a forward.

### 6.2 Kept length

A short span's kept length `m` is `len` unless the span is a verify block of the
**armed verify cohort** — the sequences `begin_verify` captured this step. For those,
`m` is decided on the host after the device-side sampling: the accept walk
(`candle-conversation/src/scheduler/decode.rs:1045-1124`, standalone
`batched_inference.rs:5649-5679`), cut shorter by the commit loop for EOS, stencil
exit or a page cut (`decode.rs:1205-1283`), handed over as `truncate_sequences`
targets (`decode.rs:1310-1325`).

Every path that forms a verify block bounds it at `R` rows: the scheduler's ladder
(ceilings ≤ 4 today, `decode.rs:762-774`), qwen35's `MTP_MAX_DRAFT` (`mtp.rs:116`,
applied at `qwen35/forward.rs:242`), and — new — qwen4exp's `begin_verify` /
`mtp_draft` (`qwen4exp/wave.rs:2462`, `qwen4exp/draft.rs:627-640`) and
`speculative_decode_step_batch` (`batched_inference.rs:5449, 5507`), which take any
depth today. A block longer than `R` rows — more than 7 drafts — is refused where it
is formed; it is never silently run as a long span, which could not partially
accept. The Flash-Next profiling ladders at depths 8, 12 and 16
(`draft_ladder.rs:388-405`) go with it.

### 6.3 Fold

One launch over every (sequence, recurrent layer, V head, `d_v` tile) to fold, from
a descriptor table `[4, n]` i64 — each (sequence, layer)'s conv tail, state and
timeline addresses, then its `m` — uploaded once (`delta_net_fold`):

```
S[i][j] ← S[i][j] · e^{G[m−1]} + Σ_{t<m} v_new[t][i] · (k[t][j] · e^{G[m−1] − G[t]})
tail    ← last K − 1 of [tail | x[0] .. x[m−1]]
```

- **`S`:** the operations are §3.3's; which thread computes an element does not
  enter its arithmetic, so the fold takes its own map — a lane owns four
  consecutive columns and a warp four rows, every access a 128-bit load, and the
  tile is loaded before the timeline is read (§13).
- **Tail:** one thread owns a whole channel's `K − 1` elements, reads them, then
  writes them. An in-place fold with `m < K − 1` reads `tail[c][jj + m]` while
  writing `tail[c][jj]`, so the tail writer's per-element spread would race
  (§11.1, item 14).
- **The table** is a few entries a sequence; it is taken from held ground, where
  every other launch table lives, so a fold issued between a step's forward and its
  reconcile needs no span.
- **Where it is issued:**
  - **at each forward's `commit_wave`**, for every short span not in the armed
    verify cohort (`m = len`) — decode rows of a step with no block, plain rows
    beside a block, short prompts, short creep spans, heal prefills, stencil
    injection, `verify_block`'s default loop, every batch-test forward;
  - **at the reconcile**, for the armed cohort, with the walk's `m`: qwen4exp's
    `rewind_cohort` → `rewind_recorded` (`qwen4exp/spec.rs:262-428`), inside its
    recorded capture beside the PLE and QSA rewinds; qwen35's `truncate_sequences`
    → `replay_recurrent` (`qwen35/forward.rs:443-559`, `qwen35/batched.rs:1249-1317`),
    which is eager today and is recorded as one wave capture the same way (§12 D4).

  Today a full accept and a plain row skip the replay (`qwen4exp/spec.rs:367-371`);
  every one of them now folds.

### 6.4 Pending, per layer

A layer is **pending** from the moment a forward hands it a timeline until its fold
or its discard. Pending is tracked per layer, beside today's per-layer `advanced`
flag (`state_store.rs:705-721`), because a step opens and commits the store's wave
once **per segment**, not once per step (`qwen35/forward.rs:896-916`,
`qwen4exp/wave.rs:3327-3382`): a verify row rides every segment
(`wave_driver.rs:1220-1227`), each segment writes a disjoint layer range, and the
armed cohort's layers stay pending across the segments' commits until the reconcile.

The store's rules:

- A forward refuses to hand a layer a timeline while that layer is pending — the
  overlapping-wave refusal, moved from the store to the layer
  (`RecurrentStateStore::layer_span_mut`).
- The sweep's recurrent bracket closes a successful wave with `commit_stores`: it
  first folds every pending layer of every store **not** in the armed cohort, in
  one launch over all of them, with the waves still open, then runs each store's
  `commit_wave`, which exchanges every layer a long span advanced and leaves the
  armed stores' layers pending. The fold is the one step of the close that can
  fail, and a refused fold writes nothing, so a failure is rolled back like any
  failed wave; the commits after it are infallible. A rollback drops only the
  pending rows its own wave wrote; rows an earlier, committed segment left stay for
  the reconcile.
- `fold_stores` with `Some(m)` and `discard_pending` end pending; a discard writes
  nothing — `S` and the tail are as the step entered.
- `export`, `import`, `fork_from`, `relocate` and `compact_stores` refuse a store
  with a pending layer, as they refuse an open wave today (`state_store.rs:1123, 1304,
  967, 350, 917`).

**Every exit from an armed step ends pending**, by `truncate_sequences` (fold) or
`abort_verify` (discard). Paths that skip both today, and must not:

- rows that committed nothing are failed and never truncated (`rollback_targets`,
  `decode.rs:2362-2377`, `:1310-1318`) → discard;
- a truncate failure fails every decode (`decode.rs:1325-1328`) → discard;
- the standalone walk's `?` exits on `chooser.choose` / `walk.commit`
  (`batched_inference.rs:5641-5677`) → `abort_verify`;
- `describe_recurrent_state` exports mid-step, between the commit and the truncate
  (`decode.rs:1152`) → moves after the truncate.

**What a failed step leaves.** An armed sequence's `S` is untouched until its fold,
so a verify step that fails anywhere — any segment, the readback, the walk — leaves
its DeltaNet state exactly as it entered, which today it does not. A non-armed span
folds at its own segment's commit, so a later segment's failure leaves the earlier
segments' layers advanced, as it does today; those turns are failed
(`fail_all_decodes`, `decode.rs:2350-2356`), and the K/V of the earlier segments is
not truncated, also as today. A fault inside the fold itself leaves `S` unrecoverable;
a device fault ends the process, so this is accepted.

### 6.5 Creep windows

A creep commits its window mid-step, before the head (`wave_driver.rs:1339-1347`), and
can advance while the head fails (`:1314-1320`). A short creep span is never armed,
so it folds at its own segment's commit like any non-armed span; a long creep span
commits through the scratch (§7). The per-layer `advanced` and pending flags keep
layers outside `[cursor, win_end)` untouched, as today.

### 6.6 PLE, QSA and the draft seed

qwen4exp's three other rewound states keep their mechanisms and their failure
bracket (snapshot at every forward's entry, restore on failure,
`qwen4exp/wave.rs:3277-3326, 3383-3420`); GDN needs no snapshot. What they lose is
the row layout they borrow from the stash: `SpecCapture::new` takes each sequence's
row from `delta.span_of` (`qwen4exp/spec.rs:99-119`), `CaptureRows` is "laid out like
the rewind stash" (`capture_rows.rs:8-9`), `head_wave_pass` writes seed rows at
`stash.row` (`qwen4exp/draft.rs:326-343`), and the `verify_stash` mutex holds
`(VerifyStash, CaptureRows)` together (`qwen4exp/wave.rs:186`). The cohort's row map
— sequence → `(row, len)` — moves into `CaptureRows`, which keeps every buffer it has
today; only the GDN operand buffers go. The `spec.rs` header table and its
"snapshotted anyway" paragraph (`qwen4exp/spec.rs:10-23`) are rewritten to match.

## 7. Long spans: the prefill scratch

A span longer than `R` rows cannot keep a timeline. It runs the conv + prefill scan
and writes a final state, and for that alone the sequence holds a **scratch slot**:

- **Claimed** in the prologue that already claims recurrent state before the tier
  stands and runs in the prepare pass of a step: qwen35's `ensure_recurrent`
  (`qwen35/forward.rs:888-895, 924-956`), which gains the span lengths it does not
  receive today, and qwen4exp's `ensure_seq_state` (`qwen4exp/wave.rs:3107-3111`),
  which already has `off + q`.
- **Used** as today's ping-pong, per layer: the scan reads the live slot and writes
  the scratch; `commit_wave` exchanges them for every layer that advanced; a failed
  forward exchanges nothing. Mid-creep, live and scratch roles differ by layer, so
  the store tracks them per layer.
- **Released** when the sequence stops prefilling: at the commit of the first
  forward that gives it no long span, and at seal, park and idle demotion, which can
  end a sequence's prefill without another commit. The store records whether a
  layer's last commit carried a long span; nothing records it today.
- **Priced** separately: admission charges one figure per sequence today
  (`reserved_bytes_for`, `state_store.rs:1084`; `recurrent_store_bytes`,
  `qwen35/batched.rs:588`). A prefilling sequence and a decoding one are priced
  apart.

Length decides: a span of at most `R` rows takes the timeline, whatever its kind;
a longer one takes the scratch. No sequence carries a short and a long span in one
forward — the driver refuses a sequence in two groups (`wave_driver.rs:724-742`) —
and glue never reaches a DeltaNet layer: both hybrid stacks refuse glue rows
(`qwen35/forward.rs:1003-1009`, `qwen4exp/wave.rs:3020-3024`).

**A prefilling sequence costs more than today, and that is accepted (§12 D2).** The
scratch has the home slot's stride, so commit is an exchange and never a copy: two
slots per layer at the timeline stride are 18 regions (288 MiB) on Flash-Next against
today's 15 (240 MiB), +20%. Prefilling sequences are few and the gate's prefill peak
is set by the transient tier (1,616 MiB at ×8); decoding sequences, which outnumber
them, carry the saving.

## 7.1 The non-fused path

Every store user takes the same contract whatever kernel runs it (§12 D3). The
non-fused path — `mix_spans_by_ops` and the tensor-op reference it calls, used on
CPU devices, for non-F32 state, `head_dim ≠ 128` and `conv_kernel ≠ 4`
(`mix.rs:1678-1694, 1723-1726`) — gains the timeline and the fold in tensor ops:

- **A short span** computes its rows' `G`, `k`, `v_new` and `x` with the reference
  chunk algebra (`delta_chunked` over one chunk) and writes them to the slot's
  timeline; it writes neither `S` nor the tail. Its outputs are what it produces
  today.
- **The fold** is the same formula as §6.3 in tensor ops: `S · e^{G[m−1]}` plus one
  matmul of `v_new[0..m]ᵀ` against `k[0..m] · e^{G[m−1] − G[0..m]}`, and the tail
  slice. The store calls whichever fold its device and geometry run.
- **A long span** writes the scratch, as on the fused path, and stops advancing the
  live `S` in place: the mirror into the write half (`mix.rs:1860-1883`) goes.

The non-fused path is held to the fused one by the same parity tolerance the
reference is held to today, not bit for bit. What it gains is the atomicity it lacks
today: a failed wave on the CPU leaves `S` as it entered, and §10's store and failure
tests run on the CPU.

The single-state entry points — `delta_net_mix`'s solo path, `absorb_solo`,
`quantized_delta_net_layer_forward` — and the reference models
(`qwen35/model.rs`, `qwen4exp/model.rs`) own a plain `DeltaNetState` outside any
store. They are parity oracles, not serving paths, and keep advancing it in place.

## 8. What is deleted or changed

Deleted:

- `LayerSlot::backup` for every sequence not prefilling; `claim_layer_states` claims
  one slot per layer, and a second only as the scratch.
- `RecurrentStateStore::layer_state_rewind`.
- `qwen35/spec.rs`: `VerifyStash`, `StashSpan`, `stash_in_slots`,
  `compact_verify_stash`, `replay_accepted_prefixes`, `replay_stacked`.
- `qwen35/batched.rs`: `begin_verify_stash`, `verify_stash_width`,
  `take/put/drop_verify_stashes` (`:278-380`); `replay_recurrent`'s replay body
  becomes the fold.
- `qwen35/forward.rs`: the sweep's stash take, `StashSlot`, `filled`, the put-back
  (`:1344-1361, 1502-1573, 1645-1656`) and the replay pricing (`:1107-1158`).
- `qwen4exp`: `verify_stash_for`, `park_verify_stash` (`spec.rs:169-210`), the stash
  capture in the sweep (`wave.rs:3963-3993, 4033`), its pricing (`wave.rs:3151-3191`),
  its compaction accounting (`wave.rs:1541-1573`) and release (`wave.rs:2730-2736`).
- `delta_net/mix.rs`: `SpanOperands` (`:1074`), `StashSlot` (`:983`),
  `DeltaNetSeq.stash` (`:971`), `delta_net_advance_spans` (`:1932-2028`).
- `delta_net/replay_stack.rs`, after its bit-for-bit test is carried over as the
  fold's oracle test (§10, test 1).
- candle-nn `wave_plan.rs`: `WaveWidth::{staged_rows, staged_spans, replay}`,
  `DeltaNetWidths::{replay_stack, replay_layer_bytes}` (`:192-287, 390-402, 2049-2078`),
  and the reference at `batched_model.rs:950`.
- `SlotTenant::RewindStash`: its label, compaction, `arena_reserved_bytes` term
  (`state_store.rs:1039-1041`) and region accounting.
- Both `affordable_draft_budget`s (`qwen35/batched.rs:879-939`,
  `qwen4exp/spec.rs:229-253`), which price draft depth from stash rows: the timeline
  is fixed per slot, so depth is bounded by `R − 1` and by K/V alone.

Changed:

- The short-span kernel: timeline out, no state or tail out (§6.1).
- Both decode kernels (§6.1).
- The three-kernel state pass: explicit intrinsics (§3.3).
- `decode_table_addresses` and both table builders (§6.1).
- The wave contract of `RecurrentStateStore`: per-layer pending, `fold`, `discard`,
  the scratch (§6.4, §7). Store tests that bake in two halves are rewritten:
  `fork_buffers_are_distinct_allocations_on_both_halves`,
  `fork_reads_the_committed_buffer_not_the_write_buffer`,
  `decode_addresses_are_the_halves_and_follow_a_commit`,
  `a_store_is_priced_from_geometry_not_from_what_is_resident`.
- The non-fused path: `mix_spans_by_ops` writes a timeline for short spans and the
  scratch for long ones, its in-place advance and mirror (`mix.rs:1860-1883`) go, and
  a tensor-op fold sits beside the CUDA one (§7.1).
- The draft-depth bound at every block-forming path, and the Flash-Next profiling
  ladders at depths 8, 12 and 16 removed (`draft_ladder.rs:388-405`) (§6.2).
- qwen35's reconcile recorded as one wave capture (§6.3).
- Comments that state today's sizes or stash: `tenant_arena.rs:33-34` ("five
  recurrent states on Flash-Next"), `vram_bands.rs:32` (the `state` band omits the
  stash), `draft_ladder.rs:317-331, 355-379` (stash OOM as the width limit).

Unchanged: the PLE, QSA and seed rewinds (§6.6); compaction's timing (between
forwards, skipped while any store is open, `state_store.rs:329-352`); drafting, which
never touches DeltaNet state (qwen35's MTP head and qwen4exp's draft head are
attention blocks, `qwen35/mtp.rs:16-18`, `qwen4exp/draft.rs:208-219`).

## 9. Cost

| | today | timeline |
|---|---|---|
| `S` traffic, full accept | forward R + W | forward R; fold R + W |
| `S` traffic, partial accept | forward R + W; replay R + W | forward R; fold R + W |
| per-row work, partial accept | stash write; replay conv, solve and mixer over `m` rows | timeline write; fold over `m` rows |
| launches, short spans beside a long span | one multi-row launch | one short + one long |
| launches per armed step | replay triple per layer stack on a partial accept | one fold |
| launches per forward, non-armed | none | one fold at commit |
| resident, decoding sequence | two `S` + two tails, plus the cohort stash | one `S` + one tail + 8 rows |
| resident, prefilling sequence | two `S` + two tails | two slots at the timeline stride, +20% (§12 D2) |

The fold adds one read of `S` on a full accept and saves one write on a partial
accept. The batched gate accepts almost every block whole (4.90 of 5 at Flash-Next's
ceiling of 4); free continuation accepts about 2.1 and the essay task 1.68–2.22
(`draft_ladder.rs:342-407`). The decode rate is measured both ways (§10).

## 10. Tests and measurements

Codec-style, raw bytes, written with the code. The fused path requires `D = 128`
(`mix.rs:1682`); the small geometry is `h_k = 2`, `h_v = 4`, `D = 128`, as
`replay_stack.rs`' tests use, beside Flash-Next's 16 K / 48 V. (`test_arch` is a
latent-MoE architecture and has no DeltaNet.)

1. **The fold is the three-kernel state pass at every prefix.** From one entering
   state: run the three-kernel scan over the first `m` rows into a state-out; run
   the short-span kernel over the whole span into a timeline and fold `m` rows. `S`
   and tail bytes equal, for every span length in `1..=R` and every `m` up to it.
2. **A decode row's `o` reads the state the fold stores.** One decode step, then the
   fold at `m = 1`: the folded `S` equals the decode kernel's in-register update,
   written out through a test-only state-out in the test build. A decode row and the
   same token at `len = 1` through the short-span kernel are not compared: their
   conv chains differ (`decode_kernel.cuh:206-209`), by design.
3. **A failed armed step leaves no trace.** A verify forward over several segments
   that fails in the last leaves the armed sequence's `S` and tail bytes as they
   entered; a discard does the same after a successful forward. Run on the fused
   path and on the CPU (§7.1).
3b. **The tensor-op fold matches the fused fold** within the reference's parity
   tolerance, for every span length in `1..=R` and every `m` up to it.
4. **The store's rules.** A pending layer refuses a second timeline; `export`,
   `fork_from`, `relocate` and `compact_stores` refuse a pending store;
   `commit_wave` leaves pending and the caller's fold lands it; a rollback drops its
   own wave's rows and keeps an earlier segment's; fold and discard clear it.
5. **Exits.** Every path in §6.4's list ends pending: a zero-commit row, a truncate
   failure, a standalone-walk error.
6. **Per-span kernel choice.** A wave carrying a verify block and a long creep span
   sends the block through the short-span kernel and the span through the scan.
7. **The scratch.** Claimed with the first long span in the prepare pass, exchanged
   per layer at commit, released one forward after the last long span and at seal;
   a decoding sequence holds one slot per layer; admission prices the two apart.
8. **The bound.** Every block-forming path — the scheduler ladder, qwen35's
   `speculative_draft`, qwen4exp's `begin_verify` / `mtp_draft`,
   `speculative_decode_step_batch` — refuses 8 drafts and accepts 7.
9. **End to end.** The hybrid wave-equivalence tests and the speculative ladders
   (`test_speculative_ladder`) stay exact.

Measured on the 16 GB card, alternating A/B builds per the laptop's throttling
(`laptop_cpu_throttle_benchmarks`): the Flash-Next batched gate's `state` band
(predicted 1,216 MiB at ×8, §5) and expert zone at ×1–×8, and decode t/s on the gate
(full accepts) and on a zend dialogue (partial accepts).

## 11. Audit notes (2026-10-17)

The design was audited against the code in four parts — the kernels, the state
store and its lifecycle, the speculative path, and the sizing. What the audit
changed, with the section that now carries it:

### 11.1 Confirmed

- Prefix stability of `G`, conv, norm, `A`/`kq`, the solve and `v_new` (§3.1).
- The fold needs nothing beyond `G`, `k`, `v_new` and `x`: `β` is inside `u` and `w`,
  hence `v_new`; `eps` and `q_scale` act before them (§4).
- Every timeline field has a natural single writer in the short-span grid
  `(n_v_heads, 4, n_spans)` (§4).
- The decode gate is `dn_decay_gate` bit for bit; the decode `k` is the same normed,
  unscaled `k`; `o` read from the in-register state is sound once that state is the
  fold's arithmetic (§6.1).
- The tail is channel-major and `last K − 1 of [tail | x[0..m)]` is exactly the tail
  writer at `len = m` (§3.1).
- Drafting never advances DeltaNet state (§8).
- No sequence carries a short and a long span in one forward; glue never reaches a
  DeltaNet layer (§7).
- Graphs and tables are rebuilt or updated per forward, so slot addresses need not
  be stable across forwards; compaction cannot meet a captured graph (§6.1).
- Persistence readers run between steps: seal export (`scheduler/mod.rs:9692`),
  checkpoints, forks (`mod.rs:4099, 10234, 10634`), restores, moves, and compaction
  (`prefill.rs:400-431`) — except one diagnostic export (§6.4).
- The CSV's `state` band, region for region (§5).

### 11.2 Corrected in this revision

| Was | Is | Where |
|---|---|---|
| today's replay runs "the same kernels" | it runs the three-kernel path; equality to the short-span kernel is a test | §3.2 |
| a shared helper keeps the fold's bits | fast math makes contraction free; the phase is written in intrinsics | §3.3 |
| "32-column tile" | 32-row (`d_v`) tile | §4, §6.3 |
| the decode kernel writes `x[0]` | only the conv-decode kernel sees raw `x`; both decode kernels change | §6.1 |
| length decides the kernel | the wave decides today; the span table is split | §6.1 |
| verify blocks are ≤ `MTP_MAX_DRAFT + 1` | qwen4exp and the standalone driver take any depth; every path is bounded at `R` | §6.2 |
| the fold is issued "at the call every step ends with" | non-armed spans fold at every commit; many short spans never reconcile | §6.3 |
| qwen35 reconciles in a "capture consume" | it reconciles in `truncate_sequences` → `replay_recurrent`, eagerly | §6.3 |
| `begin_wave` refuses a pending store | a step commits per segment; pending is per layer | §6.4 |
| a failed later segment discards earlier rows | true for the armed cohort only; non-armed spans keep today's semantics | §6.4 |
| the fold reads and stores "as the short-span kernel does" | the tail store is per-channel, or it races | §6.3 |
| today's stash is per sequence | it is per cohort | §5 |
| 44% saving | 41% in regions (5 → 4 blocks a region) | §5 |
| memory during prefill is today's | +20% per prefilling sequence | §7 |
| test 1's oracle is the short-span kernel | it has no state-out any more; the oracle is the three-kernel scan | §10 |
| `test_arch` geometry | DeltaNet has none; the small geometry is 2 K / 4 V / D 128 | §10 |
| the PLE, QSA and seed rewinds are unchanged | their row layout comes from the stash and moves to `CaptureRows` | §6.6 |

### 11.3 Already broken today, and fixed by this design or kept out of it

- **The non-fused path has no rollback for `S`.** `mix_spans_by_ops` advances the
  live `S` in place and mirrors it into the write half (`mix.rs:1860-1883`); only the
  tail ping-pongs. Every non-fused caller — CPU devices, non-F32, `head_dim ≠ 128`,
  `conv_kernel ≠ 4` (`mix.rs:1678-1694, 1723-1726`), the single-state entry points
  (`delta_net_mix`, `absorb_solo`, `quantized_delta_net_layer_forward`) and the
  reference models — already lacks wave atomicity. The design fixes it for every
  store user (§7.1, §12 D3); the oracles that own a plain state keep it.
- **The store's `seeded` flag** is read only by qwen35 (`qwen35/batched.rs:513`);
  qwen4exp keeps its own (`qwen4exp/wave.rs:1667, 1791`). Untouched by this design.

## 12. Decisions

The audit's open questions, decided 2026-10-17.

**D1. `R = 8 = DNS_ROWS`, and deeper blocks are refused.** Production ladders draft at
most 4, so 8 rows cover every production block with three to spare. Every path that
forms a block refuses more than 7 drafts (§6.2); the Flash-Next profiling ladders at
depths 8, 12 and 16 are removed. Not chosen: `R = 12` — free in regions on
Flash-Next but one block a region fewer on the 35B-A3B (6 → 5) and the 0.8B (11 → 10),
and a re-budget of the short-span kernel's registers and shared memory, which are
sized by `DNS_ROWS` (`short_span:92-105, 135, 150, 206`); and a per-model `R`.

**D2. The prefill scratch has the home slot's stride, and commit exchanges.** A
prefilling sequence costs 288 MiB on Flash-Next against today's 240 (+20%), accepted:
prefilling sequences are few and the prefill peak is set by the transient tier. Not
chosen: a scratch at today's smaller stride with commit copying it home — 272 MiB
(+13%) for one state copy per layer per long-span commit.

**D3. The non-fused path gets a tensor-op timeline and fold** (§7.1). One store
contract on every device and geometry; the CPU and fallback geometries gain the
atomicity they lack today. Not chosen: atomicity on the fused path only.

**D4. qwen35's fold is recorded as one wave capture** at the reconcile, as qwen4exp's
rewind is: eager issue measured ~3.6 ms of wall for ~0.8 ms of kernels
(`qwen4exp/spec.rs:328-347`).

## 13. Implementation notes (2026-10-18)

Where each part landed, and what building it settled.

**The store** (`delta_net/state_store.rs`). A layer is `{live: Half, scratch:
Option<Half>, pending: Option<Pending>, advanced}`; a `Half` is a state, its conv
tail and its timeline, views into one `SlotTenant::RecurrentState` slot laid out
`S | tail | timeline`, each on a 256-byte boundary. `layer_span_mut(li, len)` hands
a span the live state and its write — `SpanWrite::Timeline` for `len ≤ 8`,
`SpanWrite::State` (the scratch) beyond — and refuses a pending layer or a long span
with no scratch. `Pending` carries `this_wave`, so a rollback drops only its own
wave's rows. The scratch is claimed in the prepare pass (`ensure_scratch`: qwen35's
`ensure_recurrent`, which now takes the span lengths, and qwen4exp's
`ensure_seq_state`), and released by a commit that ran recurrent layers with no long
span, by `export` (the seal), and with the store; a commit that ran no recurrent
layer — a window of attention layers — keeps it. `fold_stores` is the one fold entry
point, CUDA (one launch over every store) or tensor ops.

**The armed cohort.** qwen35 records it as `armed: seq → block length`, set by
`begin_verify` (`arm_verify`, which refuses a block past 8 rows), folded by
`truncate_sequences` (`fold_armed`) and discarded by `abort_verify`
(`disarm_verify`). qwen4exp's is its `SpecCapture`'s sequences; `rewind_cohort`
folds the targets it reconciles and `release_capture` discards every armed sequence
the step did not, so a capture never outlives its step with rows pending.
`CaptureRows` keeps the cohort's row layout itself (`SpecCapture::new` packs the
blocks in wave order).

**Exits.** Beyond §6.4's list, `verify_blocks`' own `begin_verify` and `end_verify`
failures abort, and the scheduler's two "wrong row count" returns after
`end_verify` abort. `describe_recurrent_state` runs after the rollback on a
speculative step, and at once on a plain one, whose rows folded at commit.

**The fold runs eagerly** (`Device::eager`). It is issued inside a wave capture at
both sites — the forward's commit and the reconcile — and its table is held ground,
whose give-back fences the range on the stream; a recording stream cannot take that.
So D4's capture records nothing at qwen35's reconcile: the DeltaNet half of a rewind
is one launch, not the replay's chain of conv, scan and table uploads that D4 was
measured against, and `fold_armed` issues it directly. qwen4exp's rewind keeps its
capture for the PLE, QSA and seed work around the fold.

**Kernels.** The short-span kernel writes its timeline and no state; the decode step
and the conv step write row 0 of it; the fold is `delta_net_fold_kernel.cuh`. The
prefill kernels' layer-stack path (`DnLayerOps`, the replay's) is gone. Micro-
optimisation, held by the bit-for-bit tests (§10, test 1) throughout:

- the fold: a lane owns four columns and a warp four rows, so every S, k and v
  access is a 128-bit load; the S tile is loaded before the timeline is read; the
  conv tail is rebuilt with static-index selects rather than a runtime-indexed array
  (which spilled to local memory); the table is one `[4, n]` upload rather than a
  pointer table and a row-count table;
- the decode step stages each row's `err` and output in shared memory and writes
  both as coalesced rows, where lane 0 of each warp stored one float per row.

**Pricing.** `WaveWidth::{staged_rows, staged_spans}`, `WaveWidth::replay`,
`Chain::DeltaNetReplay`, its eight `WaveBuffer`s and `DeltaNetWidths::replay_*` are
gone; the decode table prices three addresses per row per layer, not four.
Admission prices an arriving sequence (`recurrent_store_bytes`) at its store plus
its scratch, since a sequence arrives prefilling; a decoding sequence holds the
store alone.

**Measured** (RTX 4090 Mobile, 16 GiB, Flash-Next Q3_KO, the batched gate
`test_parallel_batched_forwarding`, all 10 rows valid), the `state` band at decode
end against the same tree without this change — §5's predictions to the region:

| row | before | timeline | decode t/s before → after |
|---|---:|---:|---:|
| BF16 ×1 | 352 MiB | **208 MiB** | 15.1 → 15.3 |
| BF16 ×4 | 1,072 MiB | **640 MiB** | 49.2 → 49.0 |
| BF16 ×8 | 2,048 MiB | **1,216 MiB** | 66.5 → 67.4 |
| C0/C5/C8/C10 ×2 | 576 MiB | **352 MiB** | 29.8/27.7/26.9/25.9 → 29.1/27.6/27.1/26.3 |
| C5 ×8 / C10 ×8 | 2,048 MiB | **1,216 MiB** | 62.4/59.8 → 62.8/59.9 |

Acceptance per step is identical in every row. Both columns were run with main's
`6e60c910a` (the graph library's learn/trial/lock) reverted: with it, the ×8 rows'
sliced prefill comes back NaN on this card with or without this change.
