# Flash-Next vision: images in a turn

Qwen3.8-Flash-Next is a vision-language model: its release carries a ViT
that turns an image into rows the text trunk reads in place of placeholder
tokens. The engine brought the trunk up text-only
(`docs/archived/qwen38_flash_next.md` §6.4) and nothing in the repo loads the
tower, takes an image, or keeps one in a turn. This document is the design
for all three, chosen for the least new machinery: every piece hangs off a
mechanism the engine already has.

Scope is **still images in user turns**. Video (token 248057, grids with
`t > 1`) uses the same tower with time stepping and is not part of this
design.

**The shape of it, in five sentences.** An image in a prompt is a run of
*virtual token ids* above the vocabulary, swapped in for its
`<|image_pad|>` ids at the one place the scheduler turns a prompt into the
engine's input, so nothing recorded ever holds one. The engine, seeing
virtual ids in a wave, runs the tower over those images on the wave's own
arena and hands the rows to the embedding gather as a second table, so the
rows exist only inside the wave that uses them. Each image row's query and
key are rotated once, in place, by the difference between its multimodal
position and its slot, so every attention kernel's read-time rotation
composes to the right position and no kernel or KV structure changes. The
image itself is a replicated blob in zend's substrate, named from the turn;
every path that rebuilds a turn from its ids re-derives the virtual ids from
it. The tower's weights are one more source in the model pack, loaded by
name like the MTP head.

Sources, in order of authority:

- HF transformers `models/qwen4_exp/modeling_qwen4_exp.py` /
  `modular_qwen4_exp.py` (main), `vision_utils.py`,
  `qwen2_vl/image_processing_qwen2_vl.py`, `qwen3_vl/processing_qwen3_vl.py`.
- llama.cpp `conversion/qwen4exp.py` (`class Qwen4ExpVisionModel(Qwen3VLVisionModel):
  """The vision tower is an unmodified Qwen3-VL ViT."""`), `conversion/qwen3vl.py`,
  `tools/mtmd/models/qwen3vl.cpp`, `tools/mtmd/mtmd.cpp`,
  `ggml/src/ggml-cpu/ops.cpp` (`ggml_mrope_cache_init`).
- vLLM `models/qwen4_exp/nvidia/model.py`, which builds
  `Qwen3_VisionTransformer` directly.
- The checkpoint's `config.json` and `preprocessor_config.json`, and the
  header of `unsloth/Qwen3.8-Flash-Next-GGUF` `mmproj-F16.gguf` at commit
  `f5d175de2cf7f720d45d6064019a81fcf056e986`, parsed directly.

All three implementations agree the tower is Qwen3-VL's, unchanged.

---

## 1. The tower, exactly

### 1.1 Configuration

| Key | Value |
|---|---|
| `depth` | 27 |
| `hidden_size` | 1152 (16 heads × **head dim 72**) |
| `intermediate_size` | 4304 |
| `hidden_act` | `gelu_pytorch_tanh` (blocks) |
| `patch_size` / `temporal_patch_size` / `spatial_merge_size` | 16 / 2 / 2 |
| `num_position_embeddings` | 2304 = 48 × 48 |
| `out_hidden_size` | 2560 = the trunk's `hidden_size` |
| `deepstack_visual_indexes` | `[]` — no deepstack mergers |
| vision RoPE theta | 10 000 (the default; no `rope_parameters`) |
| norm eps | 1e-6 |
| token ids | `<|vision_start|>` 248053, `<|vision_end|>` 248054, `<|image_pad|>` 248056 |

### 1.2 The mmproj file

The trunk's doc names AtomicChat's `mmproj-Qwen3.8-Flash-Next-F16.gguf`; the
source here is unsloth's — the repo the trunk's pinned checkpoint comes from
(`quantized_qwen38_moe.rs:42`) — where the files are **`mmproj-F16.gguf`**
and `mmproj-BF16.gguf`. GGUF v3, 334 tensors, `general.architecture = clip`,
`clip.projector_type = qwen3vl_merger`, no min/max-pixel keys.

| Tensor | Dims (ggml order) | HF name |
|---|---|---|
| `v.patch_embd.weight`, `v.patch_embd.weight.1` | [16,16,3,1152] each | `patch_embed.proj.weight[:,:,0]`, `[:,:,1]` |
| `v.patch_embd.bias` | [1152] | `patch_embed.proj.bias` |
| `v.position_embd.weight` | [1152, 2304] | `pos_embed.weight` |
| `v.blk.N.ln1.*`, `v.blk.N.ln2.*` | [1152] | `blocks.N.norm1`, `norm2` |
| `v.blk.N.attn_qkv.*` | [1152, 3456] | `blocks.N.attn.qkv` |
| `v.blk.N.attn_out.*` | [1152, 1152] | `blocks.N.attn.proj` |
| `v.blk.N.ffn_up.*` / `ffn_down.*` | [1152, 4304] / [4304, 1152] | `blocks.N.mlp.linear_fc1` / `fc2` |
| `v.post_ln.*` | [1152] | `merger.norm` |
| `mm.0.*` / `mm.2.*` | [4608, 4608] / [4608, 2560] | `merger.linear_fc1` / `fc2` |

Weights F16, biases and norms F32. ≈ 0.45 B parameters: ≈ 0.9 GB at F16,
≈ 0.48 GB at Q8_0.

### 1.3 Forward

`N = grid_h · grid_w` patches, in **merge-block order** (the four patches of
each 2 × 2 block consecutive, `(dy,dx) = (0,0),(0,1),(1,0),(1,1)`).

1. **Patch embed.** HF's `Conv3d(kernel = stride = [2,16,16])` over a still
   image (the frame twice) is exactly `(W₀ + W₁)·p + b`: one GEMM with the
   two temporal slices summed **once, at pack build**, over 768-value patch
   rows.
2. **Position embedding.** The 48 × 48 table resampled to the grid,
   bilinear, `align_corners = True`, borders clamped: `src = i·47/max(size−1, 1)`,
   taps `floor`, `floor + 1`, weights `1 − |d|`, the outer product of row and
   column taps (`vision_utils.py::get_vision_interpolation_indices_and_weights`).
   Added to the patch rows. No class token, no pre-LN.
3. **27 pre-norm blocks**, all with bias:
   `x += proj(attn(LN₁(x)))`, `x += fc2(gelu_tanh(fc1(LN₂(x))))`, LayerNorm
   with weight and bias.
4. **Attention**: `qkv` split Q | K | V, `[N, 16, 72]` each; **full,
   non-causal, per image** (`cu_seqlens`); scale `72^-½`.
5. **2D RoPE** before attention: `inv_freq = 1/10000^(arange(0, 36, 2)/36)`
   (18 frequencies); pair `(j, j + 36)` of the 72 dims rotates by the patch's
   **row** for `j < 18` and its **column** for `18 ≤ j < 36`, NeoX style, F32.
6. **Merger**: `LN` per patch (`v.post_ln`), view `[N/4, 4608]`, `fc1`,
   **exact-erf GELU**, `fc2` → `[N/4, 2560]`, one row per `<|image_pad|>`.
   (llama.cpp uses ggml's tanh GELU here; we follow HF.)

---

## 2. Preprocessing

All on the host at **submission**, once per image, in
`candle-conversation` (which already depends on the `image` crate,
`Cargo.toml:52`).

1. Decode (JPEG, PNG, WebP, GIF first frame, BMP, TIFF) to RGB8.
2. `smart_resize`, factor 32: refuse aspect > 200; `h̄ = round(h/32)·32`,
   `w̄ = round(w/32)·32` (half to even); above `MAX_PIXELS` scale by
   `β = √(hw/MAX)` with `floor`, below `MIN_PIXELS` by `β = √(MIN/hw)` with
   `ceil`. **`MIN_PIXELS = 65 536`** (HF), **`MAX_PIXELS = 4 194 304`**
   (4 096 image tokens; HF allows 16 384). Constants, not settings.
3. Resize bicubic (`image::imageops::FilterType::CatmullRom`; HF's is
   torchvision's bicubic with antialias — close, not bit-identical; parity
   tests feed pre-sized images, §9).

The result, a `PreparedImage { rgb: Vec<u8>, grid: [h, w] }` (patch units),
is what travels to the engine. Normalization `(x/255 − 0.5)/0.5` and the
patch permutation happen on the device (§4.2). Reference sizes:

| Input | Resized | grid | `<|image_pad|>` | advance `max(h/2, w/2)` |
|---|---|---|---|---|
| 224×224 | 256×256 | 16×16 | 64 | 8 |
| 640×480 | 640×480 | 30×40 | 300 | 20 |
| 800×600 | 800×608 | 38×50 | 475 | 25 |
| 1280×720 | 1280×704 | 44×80 | 880 | 40 |
| 1920×1080 | 1920×1088 | 68×120 | 2040 | 60 |
| 100×50 | 384×192 | 12×24 | 72 | 12 |

---

## 3. Weights: one more pack source, like the MTP head

The MTP head is the template, end to end:

- the recipe lists a source by role — `SourceRole::{Trunk, DraftHead,
  ExpertImport}` (`qwen4exp/prepare/recipe.rs:34-51`), the Flash-Next recipe
  adds `DraftHead` (`quantized_qwen38_moe/engine_artifact.rs:47-55`);
- `prepare_engine` gathers sources by role and merges their tensors into the
  one pack (`prepare/build.rs:97-138`);
- the engine reads them by name from the same `GgufModel` as resident
  `QMatMul`s (`MtpDense::load`, `qwen4exp/mtp.rs:175-233`,
  `WeightResidency::Span`), before `dense_span::close_load`
  (`engine.rs:481-518`), and attaches the module (`engine.rs:583-605`).

The tower is a fourth role, **`SourceRole::Vision`**: repo
`unsloth/Qwen3.8-Flash-Next-GGUF`, the pinned revision above,
`mmproj-F16.gguf`. At build its GEMM weights are requantized to Q8_0, the
patch weight is folded (`W₀ + W₁`) and kept F16, biases, norms and the
position table stay F32, and the `v.*`/`mm.*` names are kept. At load a
`VisionTower` is read the way `MtpDense` is and attached as `model.vision`.
0.48 GB of dense weights, resident — an image arrives any time, and loading
it per image would be ~40 ms over the PCIe-limited links of the 4090 Mobile
and the 3090 for memory the expert cache would only lend back.

The image-generation guest (`candle-conversation/src/guest/`) is the other
existing way to run a network on the card, and the wrong one here: it loads
per drain, stops every wave while it runs, and hands only host data back
(`guest/mod.rs:10-22`, `work.rs:386-402`). The tower's rows are wanted on
the device, inside a prefill wave.

---

## 4. In the wave

### 4.1 Virtual ids

An image's `h'·w'` placeholder positions (`h' = grid_h/2`, `w' = grid_w/2`)
are recorded everywhere as `<|image_pad|>` 248056. In the **engine's input
only**, they are `VOCAB + base + i` for `i` in `0..h'·w'`, where `base` is
the image's range in the engine's **image registry**:

```rust
/// Host-side: images waiting to be read, by the virtual id range each holds.
pub struct ImageRegistry { … }                  // in the engine, Arc-shared
impl ImageRegistry {
    pub fn register(&self, image: Arc<PreparedImage>) -> ImageHandle;
}
/// RAII: holds the id range `VOCAB + base ..` until dropped.
pub struct ImageHandle { base: u32, len: u32, … }
```

Registering is host-only (no device work), so it can be done from any
thread. A range is returned to the registry when its handle drops.

### 4.2 Reading the images

`sweep_layers` already reads every input id to the host once, for the PLE
hash (`host_token_ids`, `wave/layers.rs:202`). Right after it:

1. Collect the distinct images named by ids ≥ `VOCAB` in this wave.
   None → everything below is skipped, and a text wave is the wave it is
   today.
2. Run the tower over **all of them in one call** on the wave's forward
   arena (`fwd_ticket`, the arena the residual lives on): upload each
   image's RGB8, then `vision_patchify` (normalize + merge-block permute into
   768-wide rows), the patch GEMM, the position-embedding add, 27 blocks, the
   merger — into one `[Σ h'w', 2560]` F32 buffer, images back to back.
3. Rewrite the wave's ids: a virtual id becomes `VOCAB + offset_of(image) + i`,
   its row in that buffer; and for the PLE hash, `seq_tokens` gets 248056 in
   its place (`ple.rs:184-199` hashes the id, and HF hashes `input_ids`,
   which hold `<|image_pad|>` there; `PleState.prev` must never carry a
   virtual id, `ple.rs:231`).
4. `gather_into` (`layers.rs:212-213`) takes the buffer as a **second
   table**: in `embed_gather_dequant.cu`, an id in `[n_src_rows, n_src_rows + n_extra)`
   copies that F32 row into every destination — the residual's `hc` copies
   and the head's `row_embeds` — where today any id past the table writes
   zeros (`:94-104`). One branch in the kernel, two arguments in the
   launcher. The MTP head therefore sees image rows as HF's does, through
   `inputs_embeds`.

The rows live only for this wave. A wave holding part of an image runs the
tower over the whole image and uses the part it holds; an image cut across
two waves (`pack_prefill_slabs`, `wave_driver.rs:479-527`; prefill caps at
8 192 rows) is encoded twice. That is milliseconds against a prefill, and it
is what makes the rows need no lifetime, no cache and no eviction: nothing
about an image outlives a wave on the device.

The tower's buffers are priced in the wave's transient plan like the
residual is (`WaveBuffer::HyperResidual`, `HeadRowEmbeds` in
`layers.rs:173-174`): a `WaveBuffer::VisionRows` for the output and the
largest block's activations for the blocks, from the wave's patch count —
so `tier_fits` refuses a wave the tier cannot hold, as it refuses any other
(invariant 7). The scheduler's admission sees the cost through the same
plan.

### 4.3 What the tower needs that does not exist

- **`vision_patchify`** — normalize and permute RGB8 into patch rows, emitted
  in the GEMM's input type (invariants 1 and 4).
- **ViT attention** — non-causal, segmented by image, head dim 72 (padded to
  80 in shared memory, not in tensors), online softmax, never materialising
  `N × N` scores: a 4 096-token image is 16 384 patches. Nothing in the
  fork fits (`sdpa` is Metal-only, `ops.rs:1630`; `z_image`'s int8 kernel is
  head-dim 128, `dit-attn/api.rs:24`; `candle-flash-attn` is an sm80-only
  optional feature candle-conversation does not enable). Check the
  `perf-boost` branch first (`feedback_port_perf_boost_kernels`).
- **`rope_axes_inplace`** — §4.4; it also serves the ViT's 2D RoPE, with a
  different axis rule.

Everything else is what the trunk and the MTP head already call:
`QMatMul::forward_live`, the fused LayerNorm with a ticket, `gelu` /
`gelu_erf`, `broadcast_add` for biases. The position-embedding resample is
four gathers and a weighted sum of the resident 2304-row table, from a tap
table built per image on the host (grid only).

### 4.4 Positions: rotate each image row once

**The rule** (`get_rope_index`, Qwen3.5 variant; llama.cpp `mtmd.cpp`
agrees). Text tokens share one position on all three axes. The image token
at merged `(r, c)` of an image starting at `s` sits at `(t, h, w) =
(s, s + r, s + c)`, and text after the image continues at `s + max(h', w')`.
With `mrope_interleaved`, sections `[11, 11, 10]` over the 32 rotating
pairs: pair `j % 3 == 1` → H, `j % 3 == 2` → W, else T
(`recomposition_frequencies`; ggml's `is_imrope`). For text all three axes
are equal, which is why text-only IMRoPE is plain NeoX.

**What the engine does.** A key's rotation is its slot: K is stored
un-rotated and every attention kernel rotates it at read time by
`rope_base + offset` (`candle-nn/src/kv_cache/chunked/types.rs:132-149`),
which is what lets a sealed chunk be injected anywhere.

**The change.** RoPE composes per pair, `R(a)·R(b) = R(a + b)`. So each image
row's **query and key are rotated once, in place, by `δ = target − slot`**
per axis, before anything reads them. Every reader then rotates by the
slot as it does today, and the composition is the target. `δ` depends only
on the row's place in its image — not on where the image sits — so the K in
the cache stays position-agnostic and injectable anywhere.

The hook is the one point where the prefill wave holds `q`
`[rows, n_head, 256]` and `k` `[rows, n_kv, 256]` between projection
(+ q/k norm + `RotaryLayout` gather) and their first reader:
`batched_layer/prefill_attn.rs`, after line 71, before
`paged_prefill_batched` (`:222`). Nothing is fused across it: the int8
prefill kernel rotates Q in registers at `prefix_len + tok`
(`prefill_kernel.cuh:259`) and writes K un-rotated (`:282-304`,
`kv_store.cuh:63-81`); prestage, tiles, the float fallback, decode
(`decode_attn.cuh:444-445`) and glue all rotate what is stored by the slot.
In kernel slot order rotary pair `f` is dims `(f, f + 128)`
(`rotary_layout.rs:15-23, 105-107`), so the axis of a pair is `f % 3`.

`rope_axes_inplace(x, runs, half, axis_rule, rung)` rotates pair
`(f, f + half)` of each row in `runs` by the row's position on axis
`axis_rule(f)`. For the trunk: `half = 128`, 32 pairs, `axis = f % 3`, the
sequence's rung (`rope_schedule/rungs.rs`), and the per-row `δ` computed in
the kernel from a run descriptor per image span in the wave
`(wave_row, len, i0, n, m, w')`, so the host builds one entry per image, not
one per row. It reaches `forward_attn_batched` (`batched_layer/mod.rs:288`)
as an `Option` on `BatchedAttentionParams`: `None` for every wave without an
image. The MTP head runs the same `forward_attn_batched` over the wave's
prefill rows (`draft.rs:228, 295`) and gets the same runs, shifted by its
one-row offset.

**Where the image is anchored.** The exact rule shifts every token after an
image down by `n − max(h', w')`, which is not local to the image: decode Q
(`decode_attn.cuh:179`) and every later chunk would have to carry it. So
the image is anchored at its **end**: its rows take positions
`(e − m) + (0, r, c)` where `e` is the slot of `<|vision_end|>` and
`m = max(h', w')`. Then

- relative positions within the image are exact;
- relative positions between the image and everything **after** it — the
  question about it, the answer — are exact;
- relative positions between text **before** the image and everything after
  it are `n − m` longer than training saw (a 1 024-token image: 32 → 1 024).

With `i = r·w' + c` the row's index in the image and its slot `e − n + i`,
`δ = (n − m − i, n − m + r − i, n − m + c − i)`. Positions before the image
are untouched, so nothing else in the conversation moves. The stretch is of
the same kind provenance projection already makes on every retrieved chunk,
which is placed at a slot unrelated to where it was written; the gates (§9)
measure what it costs.

**Under projection, an image is one unit.** RoPE is applied at read time
because projection re-places stored K/V at slots unrelated to where it was
written. δ depends only on a row's place in its image, so it survives any
placement: `R(slot_new)·R(δ) = R(slot_new + δ)`, and wherever the image
lands its geometry is exact. That holds only for the image **whole**. If
part of an image were placed without the rest, the gap would close up
uniformly on all three axes, and on H and W a uniform shift reads as "those
rows moved up and left": the image would come back with its pieces out of
place.

So projection never places part of an image, and today it holds by
construction. The carving unit is one whole turn
(`docs/hit_driven_projection.md` §8.1, "Whole turns, never records"; every
unit of `dialogue_fill.rs` is whole turns), and an image lies inside its
turn's user segment. The one place a turn is split, the reasoning split
(`turn_layout/layout.rs`, which windows reasoning out on index-page
boundaries), splits the assistant segment only. To keep it true:

- `TurnDecl.images` spans are checked at seal to lie within the turn's one
  real `User` segment, beside `validate_tiling`; a turn that breaks it is
  refused at seal, not projected wrong later.
- Anything that comes to carve finer than a turn (paging a long user
  message, windowing a user segment) treats every image span as atomic: it
  places the span whole or not at all. This is a rule of the turn layout,
  stated with `TurnSegment::User`, not a convention of each carver.
- QSA selection inside a slot is not a placement. It picks which cells a
  query attends to without moving any, so attending to part of an image
  sees that part at its exact positions.

**What stays one-dimensional, and why.**

- **The QSA indexer.** Its keys are pooled over four cells, normalised by a
  per-channel gain, quantized and stored by one kernel
  (`qsa_index_append.cu:169-207`), and rotated at read by the block's first
  position (`qsa_score_paged.cu:129`); a row's δ cannot be applied before the
  pooling and the gain. The indexer only chooses which cells attend once a
  context passes 2 051 positions (`qwen38_flash_next.md` §12.5), and it keeps
  choosing by slot positions, consistently for queries and keys — attention
  over what it chooses is exact.
- **Rung changes.** Past 262 144 positions a sequence moves to the next
  rope rung and every key is re-rotated with its frequencies at read; a δ
  applied at the earlier rung stays at that rung's frequencies. YaRN changes
  only the low frequencies, where `δ·freq` is tiny.
- **Provenance capture.** The prefill kernel captures the first head's Q
  for seal-time selection and the BDP signatures (`prefill_kernel.cuh:291, 298`);
  for image rows that is the δ-rotated Q. Image positions' signatures carry
  their in-image offset on 64 of 256 dims, which is position information the
  signatures of text do not have; retrieval of image chunks is measured in
  §9.3 before anything is done about it.

---

## 5. The scheduler: where virtual ids enter

A live turn's prompt reaches the engine through one copy:
`scheduler/prefill/wave_run.rs:93` (`work.tokens[off..end].to_vec()` in
`build_wave_group_inputs`), and nothing recorded reads that copy — the
substrate's token record (`seal_turn.rs:297`), the penalty and DRY context
(`first_samples.rs:370`, which asserts ids below vocab,
`penalty_counts.rs:47`), the context dump, the page cut and the replay
check all read `work.tokens` itself. So the swap goes **there**: each
position inside an image span becomes `VOCAB + handle.base + (pos − start)`.

The spans travel exactly as `recurrent_spans` do — built in
`turn_submitter.rs:220-236`, shifted by the turn head at `:487-493`, carried
in `SubmitTurn` (`requests.rs:238`), copied into `PrefillWork` at
`submit_turn.rs:588` — as `image_spans: Vec<(u32, ImageHandle)>`. The
handles drop with the `PrefillWork`, which returns the ranges.

The other places that turn ids into engine input take spans the same way,
through one helper, `engine_ids(&tokens, &spans)`:

| Path | Engine input | Spans from |
|---|---|---|
| live turn, compression turn, zend replay | `wave_run.rs:93` | `PrefillWork.image_spans` |
| recurrent-memory catch-up, branch checkpoint pass | `recurrent.rs:375` (`prefill_tokens_on`) | the turn's `images` (§6.2) |
| section ingest / restore | `section_ingest.rs:87`, `wave_run.rs:120` | the section's turns' `images` |

Cold reload re-prefills nothing — it restores sealed K/V
(`persistence/integrity.rs`) — and cold-to-warm, projection injection and
replication move K/V; the image is in it already. Glue, stencil, heal and
tail prefills (`wave_run.rs:407`, `gap_fill.rs:57`, `stencil_wave.rs:112`,
`decode/commit.rs:902`, `cleanup.rs:513-520`) prefill text the engine wrote
itself and need nothing.

---

## 6. The conversation layer

### 6.1 Requests

- **`POST /v1/chat/completions`** (`zend/src/api/chat.rs:70`). Today
  `text_content` keeps only the `text` of array content and drops every
  image part (`zend/src/types.rs:66-87`, pinned by a test at `:275-292`).
  `ChatMessage.content` becomes a list of parts; an image part is the
  OpenAI shape `{"type": "image_url", "image_url": {"url": …}}`, the url a
  `data:` URL or a substrate image id. Nothing on the request path fetches
  from the network; a client sends the bytes.
- **The routing key and the passthrough transcript** read text
  (`cluster/src/dispatch/derived.rs:92`, `session.rs:8384`) and keep doing
  so, with an image rendered as its marker (§6.4).
- **`zen_chat_send`** gains `images`: ids of images uploaded through a zend
  `image_upload` (a signed `PUT`, as npcd's, `npcd/src/mcp/image_upload.rs`),
  or of images npcd drew (`npcd/src/images.rs`), which the host — both
  tenants in one process — copies into zend's substrate as zend's own blob.

### 6.2 Turns

- **Tokens.** An image is a third kind of `TurnText` piece (`turn_text.rs`),
  `Image { id, grid }`. Its ids — `248053, h'·w' × 248056, 248054` — are
  spliced into `user_tokens` right after `encode_text`
  (`turn_submitter.rs:217`), where the scaffold's raw ids are the precedent
  (`:262-269`); `user_content_end` (`:307`) stays right. Never as markup
  text: `user_text` is stored verbatim and the byte spans at `:224-229`
  would stop lining up.
- **The image.** A `Blob` in **zend's** substrate, named by SHA-256
  (`RecordType::Blob`, `persistence/record.rs:300-309`; `put_blob`,
  `persistence/mod.rs:561-580`) — content-addressed, replicated. The
  original bytes.
- **The turn.** `TurnDecl` (`persistence/streams.rs:65-113`) gains
  `#[serde(default)] images: Vec<TurnImage>`, `TurnImage { id: sha256,
  start: u32, grid: [u32; 2] }` — the turn-relative position of the first
  `<|image_pad|>` and the grid, fixed at submission so the placeholder count
  never changes under a turn. Not a new `TurnSegment` variant: that enum has
  no `#[serde(other)]`, and a member that does not know a variant drops the
  whole turn (`substrate.rs:6388-6392`).
- **Replay paths** (§5) read `images`, load and prepare the blobs, register
  them, and pass the handles as spans.
- **Lifetime.** Blobs are not retired by a timeline tombstone; deleting a
  conversation retires its images' blobs with a replicating tombstone, as
  npcd's kept images do (`npcd/src/images/kept.rs`), once no turn names
  them.

### 6.3 Capabilities

`Capabilities { tokens, dialect }` (`engine_slot/capabilities.rs:11-50`)
gains `vision: bool`, checked at attach (`engine_slot/mod.rs:186-195`)
against the model spec's `vision` flag (beside `supports_thinking`,
`models/qwen38_flash_next.rs:115`); zend declares it and the three tokens
(`zend/src/tenant.rs:41-46`). A request with an image to a model without
vision is a 400.

### 6.4 Text views and hashes

- `user_text_of` and the history and hydrate paths
  (`substrate.rs:6886-6951`) render an image as `[image <8 hex>, W×H]`.
- Section content hashes (`persistence/content_hash.rs:153`,
  `projection/plain_prompt.rs:127`) take the image ids with the tokens.
- Repetition and DRY counts skip `<|image_pad|>`: a run of thousands of one
  id would dominate them.

---

## 7. What changes, file by file

| Where | Change |
|---|---|
| `qwen4exp/prepare/recipe.rs`, `build.rs`, `engine_artifact.rs` | `SourceRole::Vision`; fold, requant, merge |
| `qwen4exp/vision/` (new: `tower.rs`, `patchify.rs`, `pos_embed.rs`, `registry.rs`) | `VisionTower` load + forward; `ImageRegistry`, `ImageHandle` |
| `qwen4exp/engine.rs` | load the tower beside `MtpDense`; own the registry |
| `qwen4exp/wave/layers.rs` | after `host_token_ids`: images → tower → id rewrite, PLE ids; second table to `gather_into`; runs to attention |
| `models/device_embedding.rs`, `simple/embed_gather_dequant.cu` | the second F32 table |
| `batched_layer/prefill_attn.rs`, `mod.rs` | `BatchedAttentionParams.image_runs`; `rope_axes_inplace` on `q`, `k` |
| `candle-kernels` (new) | `vision_patchify`, ViT attention, `rope_axes_inplace` |
| wave transient plan | `WaveBuffer::VisionRows` and the tower's blocks |
| `qwen4exp/config.rs` | parse and assert `ple.image_token_id` |
| `scheduler/prefill/wave_run.rs`, `recurrent.rs`, `section_ingest.rs` | `engine_ids` with spans |
| `PrefillWork`, `SubmitTurn` | `image_spans` |
| `turn_text.rs`, `turn_submitter.rs` | the image piece and its ids |
| `persistence/streams.rs` | `TurnDecl.images` |
| `turn_layout/` | image spans lie within the `User` segment, checked at seal beside `validate_tiling` |
| preprocessing (new, `candle-conversation/src/vision/`) | decode, `smart_resize`, `PreparedImage` |
| `zend/src/types.rs`, `api/chat.rs`, MCP | image parts, `image_upload`, `zen_chat_send.images` |
| `engine_slot/capabilities.rs` | `vision` |

No change to: any paged attention kernel, `ChunkMeta`, the KV formats, the
QSA indexer, the substrate's chunk records, replication, or the decode path.

---

## 8. Hot-path invariants

| Invariant | How it holds |
|---|---|
| 1, 1b | patchify emits the GEMM's input type; the merger writes F32, the gather's second table's type and the residual's |
| 2, 2b | the merger's `[N/4, 4608]` is a view; images are written back to back by patchify, never concatenated after; the gather reads the second table in place |
| 3 | no readback: the virtual ids are read in the readback PLE already makes |
| 4 | host work is per image (decode, resize, a tap table, one run descriptor per span); normalize, permute and every layer are kernels |
| 5 | one tower call per wave over all its images; one `rope_axes_inplace` per attention layer over all image rows |
| 6 | every tower buffer and the rows are uninitialised; every byte written |
| 7 | the tower's weights are budgeted dense weights; its activations are priced in the wave's transient plan |

---

## 9. Testing

TDD; each piece lands with its tests.

### 9.1 Exact values

- `smart_resize`: §2's table, both bounds, half-to-even, the aspect refusal.
- `vision_patchify`: a 32 × 32 image whose bytes encode `(c, y, x)`, the
  patch rows as raw expected values, merge-block order included.
- Position-embedding taps and weights for 1-, 2-, 16- and 48-wide grids.
- `rope_axes_inplace`: rows whose δ-rotation then slot-rotation is compared
  with a direct rotation at the target position, per axis rule; the δ table
  for a 3 × 5 image, by hand.
- The gather's second table: virtual ids among real ones, every destination,
  bit patterns.
- `engine_ids`: spans in, virtual ids at exactly the span positions, the
  rest untouched; the PLE ids carry 248056.
- `TurnDecl` with and without `images`; the image piece's ids.

### 9.2 Tower parity

A reference dump from HF transformers at the pinned revision for two images
given at their grid size: patch rows, the output after blocks 0, 13 and 26,
the merger's output — a safetensors fixture under
`candle-transformers/tests/fixtures/` with its generating command beside it,
as `prefill_fixture.bin` has (`tests/prefill_replay.rs:115`). A CPU reference
tower in plain candle ops against the fixture; the CUDA tower against the
reference, at the trunk gates' tolerances.

### 9.3 End to end

- **Reading**: an image of an eight-character random code; the model must
  answer it (no text carries it — `feedback_tool_oracles`).
- **Where**: a 5 × 5 grid with one filled cell; the model names its row and
  column. Wrong h/w positions read the content and misplace it.
- **Prefill parity**, two ways. Against HF given our end-anchored positions
  as explicit `position_ids` (its forward takes them): this must match at the
  trunk's tolerances, and is the test of the implementation. Against HF with
  its own positions: the difference is the end anchor's cost (§4.4), measured
  as last-position logit agreement and on the two tests above.
- **Survival**: an image turn, a recurrent catch-up over it, a replicated
  member answering a follow-up about the image.
- **Retrieval**: an image turn deep in a long conversation, projected back
  and asked the grid question — the image re-placed whole (§4.4), its
  geometry exact at its new slot. Plus a seal-time test: a turn whose image
  span crosses out of its user segment is refused.
- **Text unchanged**: the existing gates and `/sweep`, bit-identical: a wave
  without virtual ids takes no new branch.

---

## 10. Order of work

Each step lands complete, tested and committed before the next.

1. The pack source, the tower's load, and its CPU reference; preprocessing;
   §9.2's first test.
2. The CUDA tower: patchify, attention, the plan's prices; §9.2's second
   test.
3. The registry, the second gather table, the id rewrite in
   `sweep_layers`, PLE ids; `rope_axes_inplace` at `prefill_attn.rs`.
4. The scheduler's spans and `engine_ids` at every site of §5.
5. The conversation layer: the image piece, blobs, `TurnDecl.images`, text
   views, hashes, penalties, capabilities, blob retirement.
6. Clients: chat-completions parts, `image_upload`, `zen_chat_send.images`.
7. §9.3 on each machine.

---

## 11. Decisions for the author

- **The end anchor** (§4.4) — exact within the image and after it, stretched
  across it. The exact rule needs a per-sequence shift carried by every
  later token, decode included.
- **The QSA indexer stays one-dimensional** (§4.4).
- **4 096 image tokens per image** (§2); HF allows 16 384.
- **The tower resident** (§3), 0.48 GB.
- **Re-encode per wave** (§4.2) rather than keep rows across waves.
- **Erf GELU in the merger** (§1.3), following HF over llama.cpp.
