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

**The shape of it, in five sentences.** The tower runs as a **guest** — the
mechanism the image-generation model already uses — between waves, once per
image, and its rows come back to the host, where the image's original bytes
and its rows are kept in the substrate under the image's id. An image in a
prompt is a run of *virtual token ids* above the vocabulary, naming its rows
in a process-wide registry, swapped in for its `<|image_pad|>` ids at the one
place the scheduler turns a prompt into the engine's input, so nothing
recorded ever holds one. The engine resolves the virtual ids of a wave after
the id readback it already makes and uploads their rows as the embedding
gather's second table. Each image row's query and key are rotated once, in
place, by the difference between its multimodal position and its slot, so
every attention kernel's read-time rotation composes to the right position
and no kernel or KV structure changes. A turn's text names each image by a
marker, which is how every path that runs a stored turn again finds its
rows.

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

The result, a `PreparedImage` (`candle-conversation/src/vision/prepare.rs`) —
the resized RGB8 as a `VisionImage`, the size it arrived at, and its id, the
first 16 hex digits of the original bytes' SHA-256 — is what travels to the
tower. Normalization `(x/255 − 0.5)/0.5` and the patch permutation are the
tower's first ops (§3). Reference sizes:

| Input | Resized | grid | `<|image_pad|>` | advance `max(h/2, w/2)` |
|---|---|---|---|---|
| 224×224 | 256×256 | 16×16 | 64 | 8 |
| 640×480 | 640×480 | 30×40 | 300 | 20 |
| 800×600 | 800×608 | 38×50 | 475 | 25 |
| 1280×720 | 1280×704 | 44×80 | 880 | 40 |
| 1920×1080 | 1920×1088 | 68×120 | 2040 | 60 |
| 100×50 | 384×192 | 12×24 | 72 | 12 |

---

## 3. The tower: a guest, its rows kept

**The weights.** A model pack of its own
(`docs/self_contained_model_packs.md`), built from `mmproj-F16.gguf` at the
pinned revision (`quantized_qwen38_moe/checkpoint_pins.rs`,
`QWEN4EXP_MMPROJ_FILE` with its length and SHA-256, checked before the build,
named as `FLASH_NEXT_VISION` — a `VisionTowerSource`, `docs/vision_models.md`
§2.1) by the caller that is about to need it (`VisionTowerSource::pack`) — never by the
drain that reads it, which runs with every wave stopped. The pack is
*twinned* (`PackFamily::Twinned`, `model_pack/twins.rs`): every projection the
int8 matmul tiles is stored as its `Q8_KO` twin, converted once on the host
at build, and the position table — a matrix the tower gathers rows from,
never multiplies by — as its file's F16. Each pair's fingerprint is recorded,
so a pack written by another converter is rebuilt. One pack,
`mmproj-F16.precision.pack.gguf`, serves every card with the int8 matmul (an
F16 projection's twin is `Q8_KO` in either int8 mode); the CPU reference
reads the `off` pack, which twins nothing. The source is released once its
pack is built. It is separate from the engine's own pack: the engine recipe,
and the 190 GB pack built from it, stay as they are.

**The guest.** `Guest::Vision` (`candle-conversation/src/guest/vision.rs`)
serves a `VisionRequest` — images at their grid — with each image's rows on
the host. Between waves it places its activation arena on the guest ground's
widest run, places the pack's tensors in the ground beside it (`place_gguf`
at `Int8Mode::Off`: every tensor, twins included, is one DMA of its stored
bytes from the pinned staging buffers — no source upload, no repack kernel,
no scratch off the ground), loads `VisionTower` (`qwen_vision/tower.rs`:
plain candle ops, one forward that is both the CPU reference and the guest
arena's run), runs each image, and gives the ground back. The arena is opened
before the tower loads, so the load's own conversions — the patch embedding,
the position table, the norms and biases into bf16, with the f32 each passes
through, 32.7 MiB (`LOAD_BYTES`) — carve from the ground and none from the
pool the span leaves. Its footprint is the weights, the load's conversions and
the largest image's activations, stated per patch
(`ACTIVATION_BYTES_PER_PATCH`; 88 KiB measured at 768 × 512 against 128
priced). The arena is a bump, so what bounds it is the stages: the embedding,
each block's attention half and MLP half, and each 512-row attention band are
guest stages of their own, so a 16 384-patch image never holds `N × N`
scores, nor more than one band's.
Flash-Next's spec names its tower (`ModelSpec::vision`), so its engine
registers the guest itself (`models/builder/model_guests.rs`): the tower ships
with the model, so it is not named in a deployment's `guests.yaml`.

**Why a guest, not resident weights in the wave.** An image is read once:
its rows are a function of its bytes alone, so they are kept (§6.2) and no
later turn, replay, or member runs the tower for it again. Resident, the
tower would hold 0.48 GB of the card for the whole run, against the expert
cache, for work done once per image. As a guest it costs nothing between
images, and the engine's own load is untouched.

**Keeping the rows.** `vision::turn_images` prepares each upload, takes its
rows from the substrate when an earlier turn already read the same image,
and otherwise runs the guest and keeps them: the original bytes and the rows
(`rows_codec.rs`: `ZVR1`, height, width, row width as `u32`, then bf16
values) as two content-addressed blobs, and a keyed record
`vision/image/<id>` naming both (`vision/store.rs`). Blobs replicate with the
substrate, so a member that adopts the turn finds the image too.

---

## 4. In the wave

### 4.1 Virtual ids

An image's `h'·w'` placeholder positions (`h' = grid_h/2`, `w' = grid_w/2`)
are recorded everywhere as `<|image_pad|>` 248056. In the **engine's input
only**, they are `VIRTUAL_BASE + base + i` for `i` in `0..h'·w'`
(`VIRTUAL_BASE = 0x8000_0000`, far above any vocabulary), where `base` is the
image's range in the process-wide **image registry**
(`qwen_vision/registry.rs`):

```rust
/// RAII: holds `rows` under the id range `VIRTUAL_BASE + base ..` until dropped.
pub struct ImageHandle { base: u32, len: u32, rows: Arc<ImageRows>, … }
impl ImageHandle {
    pub fn register(rows: ImageRows) -> Result<Arc<Self>, String>;
    pub fn id(&self, i: usize) -> u32;
}
pub fn lookup(id: u32) -> Option<(Arc<ImageRows>, usize)>;
```

Registering is host-only, so it is done from any thread. A range goes back
to the registry when its last handle drops; the scheduler's work holds the
handles for as long as it can be prefilled.

### 4.2 Reading the rows in a wave

`sweep_layers` already reads every input id to the host once, for the PLE
hash (`host_token_ids`). Right after it, `take_wave_images`
(`qwen4exp/wave/images.rs`):

1. Finds the virtual ids in the wave. None → nothing below runs, and a text
   wave is the wave it was.
2. Lays the rows they name back to back in one F32 `[Σ rows, 2560]` upload
   on the wave's forward ticket, priced in the transient plan as
   `WaveBuffer::ImageRows` from `WaveWidth::image_rows` — so `tier_fits`
   refuses a wave the tier cannot hold, as it refuses any other
   (invariant 7).
3. Rewrites the gather's ids: a virtual id becomes `n_vocab + row`, its row
   in that upload; and for the PLE hash the wave's ids get 248056 in its
   place (HF hashes `input_ids`, which hold `<|image_pad|>` there; the PLE
   state never carries a virtual id).
4. `gather_into` takes the upload as a **second table**: in
   `embed_gather_dequant.cu`, an id in `[n_src_rows, n_src_rows + n_extra)`
   copies that F32 row into every destination — the residual's copies and
   the head's `row_embeds` — and an id past both writes zeros. The MTP head
   therefore sees image rows as HF's does, through `inputs_embeds`.

The kernel moves image rows in 16-byte stores where the row width and every
pointer allow, the quantized rows as before. `embed_gather_bench` is its
micro harness: a correctness gate (text, text with a second table bound,
image rows), then µs a call and GB/s for each. On the RTX PRO 5000 at 8 192
rows, binding the second table costs a text gather nothing (472.6 µs both
ways, 935 GB/s).

### 4.3 The tower's own ops

`VisionTower::encode` is plain candle ops over tensors in the guest arena:
patchify on the device (normalize, merge-block permute into 768-wide rows),
the folded patch GEMM, the position-embedding resample as four row gathers
of the 2304-row table weighted from a tap table built per image on the host,
27 blocks (`guest_chain`), and the merger with exact-erf GELU. Attention is
`z_image::attention::banded` over 512-row bands at head dim 72; the 2D RoPE
is NeoX `rope` with per-patch angle tables (`vit_rope_angles`: row angles on
the first 18 pairs, column angles on the next 18). It runs BF16 on CUDA and
F32 on the CPU, where the CPU run is the reference.

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

`ImageRotation` (`qwen_vision/rotation.rs`) is the turn. Once per forward,
`rotation_tables` lists the prefill group's image rows and each one's
`(cos, sin)` of `δ · inv_freq` at the 32 rotary pairs (`axis = f % 3`,
`trunk_axis`), F32, uploaded on the forward ticket — about 260 bytes an image
row, so an 8 192-row wave of image is 2.1 MB of the forward span's 16 MiB.
Then each attention layer turns Q and K in place with one launch each,
`image_rope_rows` (`candle-kernels/src/simple/image_rope_rows.cu`): pair
`(f, f + 128)` of every head of every listed row, in f32, each product
rounded before the sum so the result is the host's arithmetic to the bit,
rounded once to BF16. It reaches `forward_attn_batched_multi` as an `Option`
on `BatchedAttentionParams` (`with_image_rotation`): `None` for every wave
without an image, which takes no new branch. `image_rope_bench` is the
kernel's micro harness: Q and K gated bit for bit against the host, then
µs a call and GB/s.

The turn is at the slot's base frequencies. Past 262 144 positions a
sequence moves to the next rope rung (below); `δ·freq` at the low
frequencies YaRN changes is tiny.

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

So projection never places part of an image, and it holds by construction.
The carving unit is one whole turn (`docs/hit_driven_projection.md` §8.1,
"Whole turns, never records"; every unit of `dialogue_fill.rs` is whole
turns), and an image lies inside its turn's user segment: its block is put at
the head of the user tokens by the submitter itself
(`TurnSubmitter::image_blocks`), and no other path writes one. The one place
a turn is split, the reasoning split (`turn_layout/layout.rs`, which windows
reasoning out on index-page boundaries), splits the assistant segment only.
To keep it true:

- Anything that comes to carve finer than a turn (paging a long user
  message, windowing a user segment) treats every image block as atomic: it
  places the block whole or not at all. This is a rule of the turn layout,
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

A live turn's prompt reaches the engine through one copy, in
`build_wave_group_inputs` (`scheduler/prefill/wave_run.rs`), and nothing
recorded reads that copy — the substrate's token record, the penalty and DRY
context, the context dump, the page cut and the replay check all read
`work.tokens` itself. So the swap goes **there**:
`engine_ids(&work.tokens[off..end], off, &work.image_spans)`
(`vision/spans.rs`) names each position inside an image span by its row's
virtual id.

The spans travel exactly as `recurrent_spans` do — built by
`TurnSubmitter::image_blocks`, shifted by the turn head in
`submit_prefill_unit`, carried in `SubmitTurn.image_spans`, copied into
`PrefillWork.image_spans` — as `Vec<ImageSpan { start, handle }>`. The
handles drop with the `PrefillWork`, which returns the ranges.

The other places that turn ids into engine input:

| Path | Engine input | Spans from |
|---|---|---|
| live turn, compression turn, zend replay | `build_wave_group_inputs` | `PrefillWork.image_spans` |
| recurrent-memory catch-up over spliced turns | `prefill_tokens_on` | `vision::stored_spans`: each turn's user half, its markers, its rows from the substrate |
| branch checkpoint pass, section ingest | — | sections hold ingested text and catalogs; no path writes an image block into one |

Cold reload re-prefills nothing — it restores sealed K/V
(`persistence/integrity.rs`) — and cold-to-warm, projection injection and
replication move K/V; the image is in it already. Glue, stencil, heal and
tail prefills prefill text the engine wrote itself and need nothing.

---

## 6. The conversation layer

### 6.1 Requests

- **`POST /v1/chat/completions`** (`zend/src/api/chat.rs`). A message's
  `content` is a string, `null`, or OpenAI parts; its text parts join in
  order and each `{"type": "image_url", "image_url": {"url": …}}` part's url
  is kept apart as `ChatMessage.images` (`zend/src/types.rs`). A url is a
  `data:` URL carrying the file, or the id `POST /v1/images` answered with
  (`zend/src/images.rs`). Nothing on the request path fetches from the
  network; a client sends the bytes. A url that names nothing readable is a
  `400 invalid_image` before a turn starts, and so is an image sent to the
  `passthrough` model, which runs the client's transcript as text.
- **`POST /v1/images`** keeps an image ahead of the turn that names it: the
  body is the file, read by the tower once on the member that takes it, kept
  and replicated, answered with its id, size and row count.
  **`GET /v1/images/{id}`** serves the bytes it arrived as, typed by their
  format, and **`GET /v1/images/{id}/thumb`** its thumbnail
  (`zend/src/api/images.rs`): a JPEG at most 480 px on its longest side
  (`vision/thumb.rs`), made once when the image is prepared and kept as a
  third blob of its record beside the original and the rows, so a
  conversation of photographs loads in kilobytes. Both are cached by the
  browser (`private`, a day): an id is its content's digest, so its bytes
  never change.
- **Whose an image is, and how long it is kept** (`vision/tenancy.rs`). An
  image is its sender's: a file sent — a `data:` part, `POST /v1/images`, a
  `zen_chat_send` picture — records its sender as a holder
  (`vision/held/<id>/<holder>`, the caller's account key), and the GET, or a
  turn naming the image by id, answers a holder or an admin and no one else:
  a stranger's GET is the `404` an unkept id gets, and their turn the `400`
  (`images::claim`). An id learned is not an image shared. A turn records
  that its timeline reads each image (`vision/used/<id>/<timeline>`, and
  `vision/timeline/<timeline>/<id>` to find them from the timeline) *before*
  it reads them, so a delete elsewhere meanwhile never finds them unused;
  deleting a timeline by any path — the user's delete, a chat closed,
  `DELETE /v1/substrate/zend/timeline/{tl}` — forgets its uses and retires
  every image no other timeline reads: its blobs released (each retired once
  no image names it, `vision/blob_refs.rs`), the record and its holders
  ceased. An archived conversation keeps its images, as it keeps its text. An
  image no timeline ever read is retired a day after it was kept or last
  sent (`kept_ms` on its record, renewed by `vision::touch` on every
  `POST /v1/images`; the sweep runs hourly on its own clock,
  `zend/src/session/image_sweep.rs`, never on a request's path). Each fact is
  a key of its own, so two members adding a holder or a use at once never
  overwrite each other. Between members, a use noted on one reaches another
  with replication; a delete on the second inside that window can still
  retire the image, and the first turn's history then shows it as gone.
- **The turn.** The session parses the last user message's images into
  `TurnStart.images`, and the turn reads them before its first round
  (`images::resolve` on the blocking pool, under a "Reading the images…"
  status): from the substrate when an earlier turn read the same image, by
  the tower otherwise. They go to that round's `TurnOptions.images`; a tool
  round's message is the results and carries none.
- **`zen_chat_send`** takes `images`: the ids (or links) the existing
  `image_upload` tool answered once the caller's sandbox sent the picture
  (`npcd/src/mcp/image_upload.rs`). Each is fetched up the cluster as the
  caller, `GET /v1/image/{id}`, as their browser would fetch it
  (`zend/src/mcp/chats/pictures.rs`). On a conversation of the web UI's the
  turn runs where the gateway places it, so the pictures go with it as
  `data:` URLs; on a private chat the worker's first turn runs on this
  member, so they are kept here first and named by their kept ids
  (`First::Message(text, images)`).
- **The web UI.** An image dropped on or picked into the composer waits
  there as a thumbnail instead of uploading as a file, and goes with the
  next message as an image part (`zend/web/index.html`,
  `zend-api.live.js`); a stored user turn draws its images from the markers
  in its text, each its thumbnail (`GET /v1/images/{id}/thumb`), and a click
  opens the original (`GET /v1/images/{id}`) over the page, closed by a
  click or Escape.

### 6.2 Turns

- **Tokens.** Each image's ids — `<|vision_start|>`, `h'·w'` ×
  `<|image_pad|>`, `<|vision_end|>` (`VisionTokens::block`) — are put at the
  head of the turn's user tokens by `TurnSubmitter::image_blocks`, before the
  text, and recorded like every other id.
- **The text.** The user text the turn records starts with one marker per
  image, `[image <16 hex> W×H]` and a newline (`vision/marker.rs`), so the
  turn's own text names its images; the `k`-th image block in the ids is the
  `k`-th marker. `TurnDecl` does not change.
- **The image.** Its original bytes and its rows are kept under
  `vision/image/<id>` (§3), content-addressed and replicated.
- **Replay paths** (§5) read a stored turn's user half (`Substrate::user_half`),
  pair its blocks with its markers (`turn_spans`), read the rows from the
  substrate, register them, and pass the handles as spans. A turn whose
  blocks and markers do not pair, or whose rows the substrate does not hold,
  is refused rather than prefilled as pads.
- **Lifetime.** An image is content-addressed and shared by every turn that
  shows it, in any conversation, so deleting one conversation leaves it kept;
  it goes with the substrate.

### 6.3 Which engines read images

An engine whose spec names a vision tower registers the guest that runs it
(§3) — Flash-Next's and Qwen3.6-35B-A3B's (`docs/vision_models.md`). An engine
whose spec names none refuses an image turn before any guest is asked. An
image whose rows its tower already kept needs no guest.

### 6.4 Text views and penalties

- The history shows a user turn's markers as recorded; the web UI draws them
  as the images (§6.1).
- Repetition and DRY counts see `<|image_pad|>` like any id: it is never a
  token the model should write, so counting it only keeps the model from
  writing it.

---

## 7. What changes, file by file

| Where | Change |
|---|---|
| `qwen_vision/` (new: `geometry.rs`, `registry.rs`, `tower.rs`, `rotation.rs`) | grid and positions; `ImageHandle` and virtual ids; `VisionTower`; `ImageRotation` |
| `quantized_qwen38_moe/checkpoint_pins.rs`, `qwen_vision/source.rs` | the pinned `mmproj` (`FLASH_NEXT_VISION`), fetched, checked and packed |
| `qwen_vision/wave_images.rs`, `qwen4exp/wave/layers.rs`, `wave_sweep.rs` | after `host_token_ids`: rows uploaded, ids rewritten, PLE ids; second table to `gather_into`; the turn to attention; `WaveWidth::image_rows` |
| `models/device_embedding.rs`, `simple/embed_gather_dequant.cu` | the second F32 table |
| `simple/image_rope_rows.cu` (new) | the in-place turn |
| `batched_layer/prefill_attn.rs`, `mod.rs` | `BatchedAttentionParams.image_rotation`, applied to `q`, `k` |
| `candle-nn` wave plan | `WaveBuffer::ImageRows` |
| `qwen4exp/config.rs` | `ple.image_token_id` |
| `guest/vision.rs`, `guest/work.rs`, `models/builder/model_guests.rs` | `Guest::Vision`, registered for Flash-Next |
| `candle-conversation/src/vision/` (new) | prepare, rows codec, store, markers, spans, stored spans, `turn_images` |
| `turn_submitter.rs`, `turn.rs` | markers in the user text, image blocks in the user tokens, `TurnOptions.images` |
| `scheduler/` | `image_spans` on `SubmitTurn`, `PrefillWork`, `MemoryCatchUp`; `engine_ids` at `build_wave_group_inputs` and `prefill_tokens_on` |
| `substrate/user_half.rs`, `turn_access.rs` | a stored turn's user half in place |
| `zend/src/types.rs`, `images.rs`, `api/chat.rs`, `api/images.rs`, `session/` | image parts, `/v1/images`, the turn's images |
| `zend/src/mcp/chats/pictures.rs`, `zen_chat_send.rs`, `workers/` | `zen_chat_send.images` |
| `zend/web/` | the composer's images, the bubble's images |

No change to: any paged attention kernel, `ChunkMeta`, the KV formats, the
QSA indexer, the engine recipe and its pack, `TurnDecl`, the substrate's
chunk records, replication, or the decode path.

---

## 8. Hot-path invariants

| Invariant | How it holds |
|---|---|
| 1, 1b | the rows are F32, the gather's second table's type; the turn reads Q and K in their own type and writes it back |
| 2, 2b | the rows are uploaded back to back once; the gather reads them in place; the turn rotates in place, no copy |
| 3 | no readback: the virtual ids are read in the readback PLE already makes; the rows go host → device once a wave |
| 4 | host work is per image (decode, resize) and per forward (the row list and its tables); the rows come from the guest, which runs between waves |
| 5 | one upload per wave for all its images; one turn per attention layer per tensor over all image rows |
| 6 | the rows' upload and the tables are written whole |
| 7 | the rows are priced in the wave's transient plan (`WaveBuffer::ImageRows`); the tables sit in the forward span; the tower runs in guest ground between waves |

---

## 9. Testing

TDD; each piece lands with its tests.

### 9.1 Exact values

- `smart_resize`: §2's table, both bounds, half-to-even, the aspect refusal.
- Patch order (merge-block), position-embedding taps and weights, the ViT
  and trunk rotary positions, by hand.
- The turn: rows whose δ-rotation then slot-rotation is compared with a
  direct rotation at the target position, on the kernel, the rows outside
  the image and the pairs past the rotary width bit for bit as they were;
  `image_rope_bench` against the host's arithmetic bit for bit at the
  trunk's geometry.
- The gather's second table: virtual ids among real ones, every destination,
  bit patterns (`device_embedding`, `embed_gather_bench`).
- `engine_ids`: spans in, virtual ids at exactly the span positions, the
  rest untouched; the PLE ids carry 248056 (`wave::images`).
- The rows codec's exact bytes; markers and image blocks; a stored turn's
  spans from its ids and text, refused when they do not pair.
- The chat request's image parts; `/v1/images` urls; `zen_chat_send`'s
  pictures as parts, ids and links.

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

1. The tower, its geometry and its CPU reference; preprocessing.
2. The registry, the second gather table, the id rewrite in
   `sweep_layers`, PLE ids; the turn at `prefill_attn.rs`.
3. The tower as a guest; the rows kept.
4. The scheduler's spans and `engine_ids` at every site of §5.
5. The conversation layer: markers, image blocks, stored spans.
6. Clients: chat-completions parts, `/v1/images`, `zen_chat_send.images`,
   the web UI.
7. §9.3 on each machine.

---

## 11. Decisions

- **The end anchor** (§4.4) — exact within the image and after it, stretched
  across it. The exact rule needs a per-sequence shift carried by every
  later token, decode included.
- **The QSA indexer stays one-dimensional** (§4.4).
- **4 096 image tokens per image** (§2); HF allows 16 384.
- **The tower a guest, its rows kept** (§3): read once per image, not
  resident, not run inside a wave.
- **Erf GELU in the merger** (§1.3), following HF over llama.cpp.
