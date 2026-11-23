# Flash-Next vision: images in a turn

Qwen3.8-Flash-Next is a vision-language model: its release carries a ViT
that turns an image into rows the text trunk reads in place of placeholder
tokens. The engine brought the trunk up text-only
(`docs/archived/qwen38_flash_next.md` §6.4) and nothing anywhere in the repo
loads the tower, takes an image, or keeps one in a turn. This document is
the design for doing all three, end to end: from an image in a request to
rows in the wave, positions the kernels rotate correctly, and a turn that
survives reload, re-prefill and replication.

Scope is **still images in user turns**. Video (token 248057, `T > 1`
grids) uses the same tower and the same position rule with time stepping
and is not part of this design.

Sources, in order of authority (as `qwen38_flash_next.md` §1 set them for
the trunk):

- HF transformers `models/qwen4_exp/modeling_qwen4_exp.py` and
  `modular_qwen4_exp.py` (main), `vision_utils.py`,
  `qwen2_vl/image_processing_qwen2_vl.py`,
  `qwen3_vl/processing_qwen3_vl.py`.
- llama.cpp `conversion/qwen4exp.py` (`class Qwen4ExpVisionModel(Qwen3VLVisionModel):
  """The vision tower is an unmodified Qwen3-VL ViT."""`),
  `conversion/qwen3vl.py`, `tools/mtmd/models/qwen3vl.cpp`,
  `tools/mtmd/mtmd.cpp`, `ggml/src/ggml-cpu/ops.cpp` (`ggml_mrope_cache_init`).
- vLLM `models/qwen4_exp/nvidia/model.py`, which builds
  `Qwen3_VisionTransformer` directly.
- The checkpoint's own `config.json` and `preprocessor_config.json`, and the
  header of `unsloth/Qwen3.8-Flash-Next-GGUF` `mmproj-F16.gguf` at commit
  `f5d175de2cf7f720d45d6064019a81fcf056e986`, parsed directly.

All three implementations agree that the tower is Qwen3-VL's, unchanged.
Where they differ from each other it is said below.

---

## 1. The tower, exactly

### 1.1 Configuration

From `config.json` `vision_config`, with the values the code falls back to:

| Key | Value |
|---|---|
| `depth` | 27 |
| `hidden_size` | 1152 (16 heads × **head dim 72**) |
| `intermediate_size` | 4304 |
| `hidden_act` | `gelu_pytorch_tanh` (blocks) |
| `in_channels` | 3 |
| `patch_size` / `temporal_patch_size` / `spatial_merge_size` | 16 / 2 / 2 |
| `num_position_embeddings` | 2304 = 48 × 48 |
| `out_hidden_size` | 2560 = the trunk's `hidden_size` |
| `deepstack_visual_indexes` | `[]` — no deepstack mergers exist |
| vision RoPE theta | 10 000 (no `rope_parameters`; the default) |
| norm eps | 1e-6 |
| token ids | `<|vision_start|>` 248053, `<|vision_end|>` 248054, `<|image_pad|>` 248056, `<|video_pad|>` 248057 |

### 1.2 The mmproj file

The trunk's doc names AtomicChat's `mmproj-Qwen3.8-Flash-Next-F16.gguf`;
the source here is unsloth's, the repo the trunk's pinned checkpoint comes
from, where the files are **`mmproj-F16.gguf`** and **`mmproj-BF16.gguf`**.
GGUF v3, 334 tensors (27 × 12 + 10). Metadata:

| Key | Value |
|---|---|
| `general.architecture` / `general.type` | `clip` / `mmproj` |
| `clip.projector_type` | `qwen3vl_merger` |
| `clip.has_vision_encoder` / `clip.use_gelu` | true / true |
| `clip.vision.embedding_length` / `feed_forward_length` / `block_count` / `attention.head_count` | 1152 / 4304 / 27 / 16 |
| `clip.vision.projection_dim` | 2560 |
| `clip.vision.patch_size` / `image_size` / `spatial_merge_size` | 16 / 768 (= 48 × 16) / 2 |
| `clip.vision.image_mean` / `image_std` | `[0.5, 0.5, 0.5]` each |
| `clip.vision.attention.layer_norm_epsilon` | 1e-6 |
| `clip.vision.is_deepstack_layers` | 27 × false |

No min/max-pixel keys: the preprocessing limits are ours to set (§2.2).

Tensors (ggml dimension order; weights F16, biases and norms F32):

| Tensor | Dims | HF name |
|---|---|---|
| `v.patch_embd.weight`, `v.patch_embd.weight.1` | [16,16,3,1152] each | `patch_embed.proj.weight[:,:,0]`, `[:,:,1]` (the Conv3d split at t = 0 and t = 1) |
| `v.patch_embd.bias` | [1152] | `patch_embed.proj.bias` |
| `v.position_embd.weight` | [1152, 2304] | `pos_embed.weight` |
| `v.blk.N.ln1.{weight,bias}`, `v.blk.N.ln2.{weight,bias}` | [1152] | `blocks.N.norm1`, `norm2` |
| `v.blk.N.attn_qkv.{weight,bias}` | [1152, 3456], [3456] | `blocks.N.attn.qkv` |
| `v.blk.N.attn_out.{weight,bias}` | [1152, 1152] | `blocks.N.attn.proj` |
| `v.blk.N.ffn_up.{weight,bias}` | [1152, 4304] | `blocks.N.mlp.linear_fc1` |
| `v.blk.N.ffn_down.{weight,bias}` | [4304, 1152] | `blocks.N.mlp.linear_fc2` |
| `v.post_ln.{weight,bias}` | [1152] | `merger.norm` |
| `mm.0.{weight,bias}` | [4608, 4608] | `merger.linear_fc1` |
| `mm.2.{weight,bias}` | [4608, 2560] | `merger.linear_fc2` |

Size: ≈ 0.45 B parameters — 15.2 M per block × 27 plus 33 M in the merger —
so ≈ 0.9 GB at F16, ≈ 0.48 GB at Q8_0.

### 1.3 Forward

`N = grid_h · grid_w` patches for an image (`grid_t = 1`). Patch rows are in
**merge-block order** throughout (§2.3): the four patches of each 2 × 2 merge
block are consecutive, `(dy,dx) = (0,0),(0,1),(1,0),(1,1)`.

1. **Patch embed.** HF: `Conv3d(3 → 1152, kernel = stride = [2,16,16], bias)`
   over rows `[N, 1536]` viewed as `[N, 3, 2, 16, 16]`. A still image is the
   same frame twice, so this is exactly `W₀·p + W₁·p + b` with `W₀`, `W₁` the
   two temporal slices — which is how the GGUF stores it and how llama.cpp
   computes it (`build_inp_temporal_merge`). Here it is one GEMM: the two
   slices are summed **once at load** into one `[1152, 768]` weight, and each
   patch row is the single frame's 768 values (`3 × 16 × 16`). Stride equals
   kernel, so no convolution is needed at all.
2. **Position embedding.** `pos_embed [2304, 1152]` is a 48 × 48 grid,
   resampled to `(grid_h, grid_w)` bilinearly with `align_corners = True`
   and clamped borders: source coordinate `src = i · 47 / max(size − 1, 1)`,
   taps `floor(src)` and `floor(src) + 1`, weights `1 − |d|`; four taps per
   patch, the outer product of the row and column taps
   (`vision_utils.py::get_vision_interpolation_indices_and_weights`). The
   taps and weights depend only on the grid, so they are built on the host
   per image as `(idx[4], w[4])` per patch — a table, not per-token compute
   — and one kernel adds `Σ w·pos_embed[idx]` into the patch rows. No class
   token, no pre-LN.
3. **27 blocks**, pre-norm, all with bias:
   `x += proj(attn(LN₁(x)))`, `x += fc2(gelu_tanh(fc1(LN₂(x))))`.
   LayerNorm with weight and bias, eps 1e-6 — not RMSNorm.
4. **Attention.** `qkv [N, 3456]` split as Q = cols 0..1152, K = 1152..2304,
   V = 2304..3456, each `[N, 16, 72]`. **Full, non-causal** attention within
   each image — no windowing; several images in one call are separate
   segments (`cu_seqlens` = cumulative `h·w`). Scale `72^-½`.
5. **2D RoPE in the ViT** (axial). `inv_freq = 1 / 10000^(arange(0, 36, 2)/36)`
   — 18 frequencies. Each patch has `(row, col)` in un-merged patch units.
   The 72-wide cos/sin is `[row·f(18) | col·f(18) | row·f(18) | col·f(18)]`,
   applied NeoX-style (`rotate_half`, pairs `(j, j + 36)`) over all 72 dims,
   in F32. llama.cpp's `ggml_rope_multi(n_dims 36, sections {18,18,18,18},
   GGML_ROPE_TYPE_VISION)` with positions `[y, x, y, x]` is the same thing.
6. **Merger.** `LN(1152)` per patch (`v.post_ln`, *before* the shuffle),
   view `[N/4, 4608]` (the four consecutive rows of a merge block), then
   `fc1 4608 → 4608`, **exact-erf GELU** (`nn.GELU()`), `fc2 4608 → 2560`.
   Output: `[N/4, 2560]`, one row per `<|image_pad|>`.
   llama.cpp's graph uses `FFN_GELU` here, ggml's tanh approximation — a
   small, real divergence. We follow HF.

---

## 2. From bytes to patch rows

### 2.1 Decoding

The `image` crate (workspace `Cargo.toml:137-145`, already a dependency of
`candle-conversation`) decodes JPEG, PNG, WebP, GIF (first frame), BMP and
TIFF. Converted to RGB8; alpha is dropped, as PIL's `convert("RGB")` does.

### 2.2 Size: `smart_resize`

`factor = patch × merge = 32`. In order:

1. Refuse an aspect ratio over 200.
2. `h̄ = round(h/32)·32`, `w̄ = round(w/32)·32` — round half to even, as
   Python's `round`.
3. If `h̄·w̄ > MAX_PIXELS`: `β = √(h·w / MAX_PIXELS)`,
   `h̄ = max(32, floor(h/β/32)·32)`, likewise `w̄`.
4. Else if `h̄·w̄ < MIN_PIXELS`: `β = √(MIN_PIXELS / (h·w))`,
   `h̄ = ceil(h·β/32)·32`, likewise `w̄`.

The checkpoint's processor says `MIN_PIXELS = 65 536` (64 image tokens) and
`MAX_PIXELS = 16 777 216` (16 384 tokens); llama.cpp defaults to 8 192 and
4 194 304. **Ours: `MIN_PIXELS = 65 536` (HF), `MAX_PIXELS = 4 194 304`
(4 096 image tokens).** The upper bound is set by the tower's attention (a
4 096-token image is 16 384 patches) and by what one image costs a turn:
4 096 positions of KV across 12 attention layers is already a large
fraction of a turn. Images within the bound resize exactly as HF does, so
the reference stays HF's; an image above it is scaled down further than HF
would. These are constants in the preprocessing module, not settings.

Resampling is bicubic (HF's fast processor: torchvision bicubic with
antialias). The `image` crate's `CatmullRom` filter is bicubic but not
bit-identical to torchvision's; parity tests therefore feed the encoder an
image already at its grid size (§9.2), so resampling never enters a
comparison.

Reference sizes (HF bounds; ours agree below 4 194 304 pixels):

| Input W×H | Resized W×H | `grid_thw` | `<|image_pad|>` count | position advance |
|---|---|---|---|---|
| 224×224 | 256×256 | [1,16,16] | 64 | 8 |
| 640×480 | 640×480 | [1,30,40] | 300 | 20 |
| 800×600 | 800×608 | [1,38,50] | 475 | 25 |
| 1024×1024 | 1024×1024 | [1,64,64] | 1024 | 32 |
| 1280×720 | 1280×704 | [1,44,80] | 880 | 40 |
| 1920×1080 | 1920×1088 | [1,68,120] | 2040 | 60 |
| 100×50 | 384×192 | [1,12,24] | 72 | 12 |

(The grid is `[t, H/16, W/16]`; the pad count is `t·h·w/4`; the advance is
`max(h/2, w/2)`, §4.)

### 2.3 Patch rows

Normalize `(x/255 − 0.5)/0.5`. With the resized image as `[C, H, W]`,
`gh = H/16`, `gw = W/16`:

```
[C, H, W] → reshape (C, gh/2, 2, 16, gw/2, 2, 16)
          → permute (gh/2, gw/2, 2, 2, C, 16, 16)
          → reshape [gh·gw, C·16·16]          one row per patch, 768 values
```

Row order: merge block (row-major over the `gh/2 × gw/2` grid), then the
patch within the block (`dy`, then `dx`); within a row, channel, then pixel
row, then pixel column. HF inserts the duplicated time axis between `C` and
the 16 × 16 (giving 1536 values); folding the two temporal weight slices at
load (§1.3 step 1) removes it.

The host uploads the resized RGB8 bytes (`3·H·W`, a fraction of the F16
rows) and **one kernel** normalizes and permutes them into the
`[N, 768]` patch rows in the encoder's input type — hot-path invariant 4:
no host compute a kernel can do.

---

## 3. The encoder on the device

### 3.1 Weights and where they live

The tower is part of the model, so it is part of the model's pack
(`self_contained_model_packs`: every model loads from one `*.pack.gguf`).
The pack request gains a source with role **`mmproj`**, exactly as the MTP
sidecar is added (`quantized_qwen38.rs:82-86`, `.with_source("mtp", …)` at
`:329`; `qwen35/pack_build.rs:101,173` take its tensors by role): repo
`unsloth/Qwen3.8-Flash-Next-GGUF`, revision `f5d175de2cf7f720d45d6064019a81fcf056e986`,
file `mmproj-F16.gguf`, its SHA-256 recorded as pack provenance
(`model_pack/provenance.rs`). The pack keeps the `v.*`/`mm.*` names.

At pack build the tower is requantized to **Q8_0** for its GEMMs (the
trunk's own precision, `host::MODEL`), with the patch weight folded
(`W₀ + W₁`) and kept at F16, and biases, norms and the position table kept
F32. 0.48 GB.

The tower's weights are **resident** for the life of the engine, in the
span's persist tenant beside the trunk's dense weights — they are dense
weights, and an image arrives at any time. On the 16 GB card that is
0.48 GB the expert cache does not get; `docs/vram_span_partition.md`'s
budget gains the line. Streaming the tower per image (load, run, release)
was considered and rejected: 0.48 GB over a ~12 GB/s link is ~40 ms per
image on the 4090 Mobile and the 3090, every image, for memory the expert
cache would only lend back.

### 3.2 Kernels

One encoder call takes **every image of the wave at once** (hot-path
invariant 5): patch rows of all images concatenated, `cu_seqlens` per
image. Per call:

| Step | Implementation |
|---|---|
| bytes → patch rows | new kernel `vision_patchify` (normalize + permute, §2.3), emits the GEMM's input type |
| patch embed | the existing Q8_0/F16 GEMM path, bias fused |
| position embedding | new kernel `vision_pos_embed_add`: per patch, 4 taps from a per-call `(idx, w)` table |
| LayerNorm | existing fused `layer_norm` (`candle-nn/src/ops.rs:1311`), emitting the next GEMM's input type |
| qkv / proj / fc1 / fc2 | existing GEMMs with fused bias; `gelu_tanh` fused into fc1's epilogue |
| 2D RoPE | applied inside the attention kernel from a per-call `(row, col)` table — no rotated copy of Q/K |
| attention | **new kernel**: non-causal, segmented (`cu_seqlens`), head dim 72, online softmax, never materialising `N × N` scores |
| merger | `post_ln`, then the view `[N/4, 4608]` (a reinterpretation of the contiguous rows, not a copy), fc1, erf-GELU, fc2 into F32 rows |

Attention is the one real new kernel. What exists does not fit:
`candle_nn::ops::sdpa` is Metal-only; `z_image/attention.rs`'s banded path
materialises scores and its int8 kernel is head-dim 128 only
(`candle-kernels/src/dit-attn/api.rs:24`); `candle-flash-attn` has
non-causal varlen kernels that accept head dim 72, but it is an optional
feature candle-conversation does not enable, is built for sm80, and would
be a second attention stack beside ours. Before writing it, check the
`perf-boost` branch for a non-causal prefill kernel to port
(`feedback_port_perf_boost_kernels`). Head dim 72 is not a multiple of 16:
the kernel pads the head to 80 in shared memory, not in the tensors.

The paged attention kernels are not reused for this: the tower's attention
has no KV cache, no paging, no causal mask and its own RoPE, and bending
the trunk's kernels to it would put a vision branch in every one of them.

### 3.3 Output

The merger writes `[Σ N/4, 2560]` F32 rows — the residual's type
(`DeviceEmbedding::gather_into` writes F32 rows, `device_embedding.rs:130-240`)
— into a buffer that lives until the wave that consumes them has run
(`alloc_uninit`; invariant 6). The tower runs on the forward stream ahead
of the wave's layer sweep, so the wave's scatter (§5) needs no extra
synchronisation.

---

## 4. Positions: multimodal RoPE

### 4.1 The rule

Text tokens carry one position on all three axes. An image block starts at
`s`, the position after `<|vision_start|>`, and over the **merged** grid
`h' = grid_h/2`, `w' = grid_w/2`, in raster order, the image token at
merged `(r, c)` has

```
t = s        h = s + r        w = s + c
```

After the image, text continues at `s + max(h', w')` (`get_rope_index`,
Qwen3.5 variant; llama.cpp `mtmd.cpp` computes the same, `n_pos = max(nx, ny)`).
So an image of `h'·w'` tokens advances the position by only `max(h', w')`.

Which axis a rotary pair uses — `mrope_interleaved`, sections `[11, 11, 10]`
over the 32 rotating pairs (`Qwen4ExpTextRotaryEmbedding.recomposition_frequencies`;
ggml's `is_imrope` branch):

| Pair `j` (of 32) | Axis |
|---|---|
| `j % 3 == 1` (1, 4, …, 31) | H |
| `j % 3 == 2` (2, 5, …, 29) | W |
| otherwise (0, 3, …, 30) | T |

For text all three axes are equal, which is why text-only IMRoPE is plain
NeoX and why the engine has never needed this. `RotaryLayout` already
permutes the head so kernel pair `f < 32` is model pair `f`
(`rotary_layout.rs:92-134`), so the axis of a pair is `f % 3` directly.

### 4.2 What the engine assumes today

A key's rotation position **is its slot index**. K is stored un-rotated
(`candle-nn/src/kv_cache/chunked/types.rs:132-149`) and rotated at read
time by `rope_base + offset`, where a chunk's `rope_base` is the running
sum of the preceding chunks' token counts in the slot's current layout
(`slot_state.rs:756-781`; `ChunkMeta.rope_pos_u32`, `types.rs:102-118`).
That is what lets a sealed chunk be injected anywhere — provenance
projection depends on it. `rope_sections` is parsed and asserted at load
(`qwen4exp/config.rs:116, 384, 486`) and read by nothing after.

Both halves of the rule break that: image tokens have three different
positions, and every token after an image sits *below* its slot index.

### 4.3 The design: a chunk advances, and an image chunk carries its own offsets

Masking stays on slot order — the causal mask, QSA block order and every
"which cells exist" question are unchanged. Only **rotation** changes, in
two parts, both relative to a chunk so a chunk stays injectable anywhere:

1. **A chunk's advance.** Each chunk records how far it moves the rotation
   position: its token count for a text chunk; for a chunk holding image
   tokens, its text tokens plus `max(h', w')` for each image that **ends**
   in it, and nothing for image tokens otherwise. `rope_base` becomes the
   running sum of **advances** instead of token counts — the same running
   sum, over a different per-chunk number. A layout of text chunks is
   bit-identical to today's.
2. **An image chunk's offset table.** A chunk that holds any image token
   carries a table of `32 × 3` signed 16-bit offsets: for each token of the
   chunk, its `(t, h, w)` rotation position minus the chunk's `rope_base`.
   Text tokens of that chunk get `(k, k, k)` for their place `k` in the
   chunk's advance order; image tokens get `(a, a + r, a + c)` with `a` the
   image's start relative to `rope_base`. 192 bytes per image chunk,
   nothing for any other chunk.

`ChunkMeta` has two spare `u32`s (`_pad0`, `_pad1`, `types.rs:113-117`):
`_pad0` becomes the chunk's index into the forward's **offset-table
buffer**, `u32::MAX` for a chunk with none. The buffer is built with the
chunk metadata each sync, from the tables the chunks own. A kernel reading
a chunk's key at in-chunk index `j` for pair `f` rotates at

```
pos = rope_base + (table == NONE ? j : table[j][axis(f)])
```

— one predictable branch per chunk, nothing for a text chunk. Queries do
the same: a wave's prefill rows get their offsets from the same tables the
chunks being written will carry; decode rows are text, at
`rope_base + j` as now.

An image's first chunk and its last chunk usually hold text too (the
`<|vision_start|>` / `<|vision_end|>` markers and whatever follows); the
table handles any mix, and a chunk can hold the end of one image and the
start of the next.

The tables are **content** of the chunk: written with it at prefill, sealed
with it, persisted with it, and replicated with it (§7.3). A sealed window
(`SealedChunk.offset/token_count`) reads the table from its window's
start; the advance of a window is computed from its table the same way.

### 4.4 Every site that rotates

Each of these takes `pos` from the formula above instead of `rope_base + j`:

- paged prefill Q (`paged_prefill_int8_kernel.cuh:311, 484`) and K
  (`column_stage.cuh:387`; `resolve_pos`, `paged_prefill_int8_kernel.cuh:511`);
- the prefill float fallback (`batched_layer.rs:1156-1300`);
- paged decode K (`int8_decode_kernel.cuh:494-504, 1076, 1871`;
  `int8_decode_tile_kernel.cuh:1884-1954`) — decode Q is text and unchanged;
- the glue kernel (`paged_glue_kernel.cuh:248, 294, 364`);
- the non-paged tables (`rotary_layout.rs:200-233`, applied at
  `batched_layer.rs:774-810, 952-980`);
- the QSA indexer: query rotation (`rotate_rows`, `indexer.rs:1661-1760`),
  the scorer's rotation of stored compressed keys (`qsa_score_paged.cu:129`)
  and the shadow scan (`qsa_shadow.cu:171`). A compressed key is rotated at
  its block's **first** cell's position (`qwen38_flash_next.md` §12.5); for
  a block holding image cells that is the first cell's `(t, h, w)`, read
  from the table;
- the MTP head, which reuses the wave's parameters (`wave.rs:5494-5508`,
  `draft.rs:227-234`);
- the oracle (`RopeTables`, `qsa.rs:118-136`), so oracle parity covers it.

`rope_cs_at(view, pos, f)` (`rope/rope_table.cuh`) takes the per-axis
position; the table itself (`RopeRungs`/`FactoredRope`, `rope.rs:24-48`) is
unchanged — it is indexed by position, whatever axis the position came from.

Two shortcuts walk positions by a constant step and cannot inside an image
span: `rope_cs_step` and the affine step composition in
`qsa_score_paged.cu:141-347`. Both stay the path for chunks without a
table and take the direct lookup for chunks with one. The rung is chosen
from the largest rotation position reached (`indexer.rs:2788`,
`batched_layer.rs:1709`), which is now the advance sum, not the slot count.

---

## 5. Image rows in the wave

The placeholder ids stay in the wave's inputs: the PLE hash reads them
(§6), the verify capture stores them (`spec.rs:65`), and the token grid
must stay 1:1 with the KV grid (`scheduler/mod.rs:8695-8717`). Each image
contributes `h'·w'` positions of id 248056.

What changes is their embedding. Today one kernel gathers every input's
rows from the device embedding table into the four-stream residual and the
draft head's bare rows (`gather_into`, `wave.rs:4738`; table
`device_embedding.rs:130-240`). Immediately after it, **one**
`rows_scatter` (`latent_moe/scatter.rs:46, 148`, used the same way at
`draft.rs:179-201`) writes the encoder's rows over the placeholder rows —
into all four streams of `entry` and into `row_embeds` — from a run table
`(wave_row, encoder_row, len)` built from each sequence's `pre_off`/`q_lens`
(`wave.rs:4771-4778`). HF does the same with `masked_scatter` over
`input_ids == image_token_id`.

The rows reach the wave as a new field on the prefill work —
`PrefillWork` (`scheduler/mod.rs:2276`) to `StepRows`
(`wave_driver.rs:1010-1013`), `WaveGroups` (`wave_driver.rs:77-154`) and
`forward_core` — carrying, per sequence, the device rows and the absolute
position of each image span. Absolute, because a wave or slab can cut an
image span anywhere (`pack_prefill_slabs`, `wave_driver.rs:479-527`;
`MAX_PREFILL_TOKENS` 8192): a slab takes the part of a span it covers.

Ordering on the stream: the encoder for a turn's images runs once, before
the first wave that holds any of them, and its output buffer is released
after the last. Decode never sees an image.

---

## 6. PLE

The n-gram per-layer embedding hashes **input ids** (`ple_row_ids`,
`ple.rs:206`); HF computes it from `input_ids` too (`modular_qwen4_exp.py`,
`ple_input_ids`), so image positions hash the `<|image_pad|>` id, and a run
of them hashes as a run of identical tokens. That is today's behaviour
unchanged: no code changes here. `ple.image_token_id` (248056) is in the
GGUF metadata and `PleConfig` does not parse it
(`qwen4exp/config.rs:56-78`); it is parsed and asserted equal to the
tokenizer's `<|image_pad|>` id, so a checkpoint that ever disagrees is
refused at load rather than hashing the wrong token. The parity gate (§9.3)
runs a prompt with an image through PLE against the reference, which is
what settles this.

GDN layers and their conv state are position-free
(`delta_net/` has no position input) and need nothing.

---

## 7. The conversation layer

### 7.1 Requests

- **zend's `POST /v1/chat/completions`** (`zend/src/api/chat.rs:70`). The
  message parser keeps only the `text` of array content and drops every
  image part (`text_content`, `zend/src/types.rs:66-87`, pinned by a test at
  `:275-292`). It keeps them: `ChatMessage.content` becomes a list of parts,
  text or image, and an image part is the OpenAI shape
  `{"type": "image_url", "image_url": {"url": …}}` with the url a `data:`
  URL (base64 PNG/JPEG/WebP) or a substrate image id (§7.3). The request
  path fetches nothing from the network — an `http` URL is refused — so a
  slow or hostile host never holds a turn; a client sends the bytes.
- **The routing key and the passthrough transcript** read text only
  (`cluster/src/dispatch/derived.rs:92`, `zend/src/session.rs:8384`). They
  keep reading text, with an image rendered as its marker (§7.5).
- **MCP.** `zen_chat_send` gains `images`: a list of image ids, from a new
  zend `image_upload` (a signed `PUT`, as npcd's at
  `npcd/src/mcp/image_upload.rs:1-10, 43-50`) or from an image Zen drew.
  npcd's images live in npcd's substrate (`host/src/tenant.rs:40-42`); a
  chat that names one reads it through the host, which holds both tenants
  in one process, and writes it into zend's substrate as its own blob
  before the turn is submitted.

### 7.2 Turns and tokens

The user half of a turn is a `TurnText` of markup and literal pieces
(`turn_text.rs:1-18, 41`), and literal text can never become a control
token. An image is a third kind of piece, `Image { id, grid }`, which
`submit_turn_on` (`conversation.rs:2119-2355`) emits **as ids** —
`248053, h'·w' × 248056, 248054` — the way the grammar scaffold is
appended as ids (`conversation.rs:2226-2243`). The chat template places an
image where it occurs in the message, before or between text.

The grid is computed once, at submission, from the image's size and §2.2's
bounds, and stored with the turn. The placeholder count is fixed from then
on: a later change to the bounds does not change how an existing turn's
tokens line up with its KV.

### 7.3 Storage

- **The image.** A substrate `Blob` record in **zend's** substrate, named by
  its SHA-256 (`RecordType::Blob`, `persistence/record.rs:300-309`;
  `put_blob`/`blob`, `persistence/mod.rs:561-580`) — content-addressed,
  never superseded, replicated. The original bytes, not the resized ones,
  so a better preprocessor can re-read them.
- **The turn.** `TurnDecl` (`persistence/streams.rs:65-113`) gains
  `images: Vec<TurnImage>` with `#[serde(default)]`, each
  `{ id: sha256, start: u32, grid: [u32; 3] }` — the image, the turn-relative
  position of its first placeholder, and its grid. A new `TurnSegment`
  variant was the alternative and is refused: the enum has no
  `#[serde(other)]`, so a member that does not know the variant fails
  `StreamDecl::decode` and drops the whole turn silently
  (`substrate.rs:6388-6392`). A defaulted field is read as empty by such a
  member.
- **The chunk offset tables** (§4.3) are part of each chunk's record
  (`record.rs:758-776`), so they persist and replicate with the K/V they
  describe.
- **Lifetime.** A blob is not retired by a timeline tombstone. Deleting a
  conversation retires its turns' image blobs with a replicating tombstone,
  as npcd's kept-image notes do (`npcd/src/images/kept.rs:1-24, 84-119`).
  Two turns naming the same image hold one blob; it is retired when the last
  turn naming it goes.

### 7.4 Re-prefill

Every path that rebuilds K/V or recurrent state from token ids alone would
rebuild an image turn as `h'·w'` copies of the `<|image_pad|>` embedding.
Each one reads the turn's `images`, loads the blobs, runs the encoder, and
passes the rows with the tokens — the same `PrefillWork` field as a live
turn:

- cold-reload re-prefill of a turn with tokens but no chunks
  (`persistence/integrity.rs:52-58, 153-161`);
- recurrent-memory catch-up (`MemoryCatchUp`, `conversation.rs:3618-3650`);
- the seal-time clean re-prefill (`SealAction::Turn`, `scheduler/mod.rs:8442-8457`);
- section restore (`scheduler/mod.rs:8815-8830`);
- the compression re-prefill (`scheduler/mod.rs:6530-6545`);
- the replay harness (`zend/src/session/replay.rs:1-13`).

The encoder's output is not stored: rerunning the tower is milliseconds
against a prefill that already recomputes every layer, and storing
`[n, 2560]` F32 per image would be 4–40 MB per image in a log that keeps
everything. Paths that move K/V rather than rebuild it — cold-to-warm
(`projection/cold_to_warm.rs`), projection injection, replication
(`persistence/remote/completeness.rs`) — carry the image's effect already,
in the chunks and their offset tables, and need nothing.

### 7.5 Text views and hashes

- **Rendering.** `user_text_of` and the history/hydrate paths
  (`substrate.rs:6886-6951`; `session.rs:6947, 7097`) render an image as
  `[image <first 8 hex of its id>, W×H]`, so a transcript, a summariser or a
  routing key sees that an image was there without its tokens.
- **Content hashes** of turns, sections and plain prompts
  (`persistence/content_hash.rs:154`, `conversation.rs:1671`,
  `projection/plain_prompt.rs:127`) take each image's id with the tokens:
  two prompts with the same placeholders and different images must not
  collide.
- **Penalties.** Repetition and DRY counts over prompt ids
  (`penalty_counts.rs`, `dry_reference.rs`) skip the image span; a run of
  thousands of `<|image_pad|>` would otherwise dominate them.
- **Projection windows.** Sub-turn windowing can project part of a turn
  (`turn_layout.rs:22-24`). It may cut an image span: the chunks carry their
  own offsets, so a projected part of an image rotates correctly. What it
  carries is partial image content, by design of provenance-selected
  attention.

### 7.6 Capabilities

`Capabilities { tokens, dialect }` (`engine_slot/capabilities.rs:11-50`)
gains `vision: bool`, checked at engine attach (`engine_slot/mod.rs:186-195`)
against the model spec's new `vision` flag (beside `supports_thinking`,
`models/qwen38_flash_next.rs:115`). zend declares it and the three vision
tokens (`zend/src/tenant.rs:41-46`); npcd does not need it. A request with
an image to a model without vision is refused with a 400, not stripped.

---

## 8. Hot-path invariants, checked

| Invariant | How it holds |
|---|---|
| 1, 1b — no `to_dtype` | patchify emits the GEMM's input type; the merger writes F32, the residual's type; norms emit their consumer's type |
| 2, 2b — no layout copies | the merger's `[N/4, 4608]` is a view of contiguous rows; the scatter writes in place from a run table; several images go in by `cu_seqlens`, never concatenated after the fact (their patch rows are written contiguously by patchify) |
| 3 — no readbacks | nothing returns to the host; grids, tap tables and run tables are host-built from sizes the host already has |
| 4 — no host compute | the host decodes and resizes the image file (not per-token data: one call per image, before any wave) and builds index tables; normalize, permute and everything after are kernels |
| 5 — batched | one encoder call per wave for all its images; one scatter per wave |
| 6 — no zeroing | encoder buffers and the output rows are `alloc_uninit`; every byte is written |
| 7 — span boundaries | the tower's weights are persist-tenant weights, budgeted; its activations are wave-transient tier allocations, sized by the wave's patch count and refused by `tier_fits` like any other |

---

## 9. Testing

TDD: each piece below lands with its tests.

### 9.1 Unit tests, exact values

- `smart_resize`: every row of §2.2's table, both bounds, half-to-even
  rounding, the 200 aspect refusal.
- Patchify: a synthetic 32 × 32 image whose pixel values encode
  `(c, y, x)`; the patch rows asserted as raw expected values, merge-block
  order included.
- Position-embedding taps: indices and weights for 1-, 2-, 16- and
  48-wide grids.
- ViT RoPE positions: `(row, col)` per patch in merge-block order.
- MRoPE axis of each pair: the table of §4.1.
- Advances and offset tables: text-only chunks give today's `rope_base`
  exactly; an image spanning three chunks with text on both sides gives
  the hand-computed offsets; two images in one chunk.
- Placeholder ids for an image piece; the turn's `images` serde round-trip,
  and a `TurnDecl` without the field decoding to none.
- Content hash: same placeholders, different image id, different hash.

### 9.2 Encoder parity

A reference dump from HF transformers at the pinned revision for two fixed
images, given at exactly their grid size (so resampling is not compared):
the patch rows, the output after blocks 0, 13 and 26, and the merger's
output. Stored as a safetensors fixture under
`candle-transformers/tests/fixtures/` with the command that produced it
written beside it, as `prefill_fixture.bin` is (`tests/prefill_replay.rs:115`).
Two tests in Rust:

1. a CPU reference forward of the tower, in plain candle ops, against the
   fixture;
2. the CUDA path against the CPU reference.

Tolerances are the trunk gates' (Q8_0 GEMMs against an F16 reference).

### 9.3 End to end

- **Reading.** A generated image holding an eight-character random code
  in large type. The model must answer the code. No text in the prompt
  carries it, so a correct answer is the image being read
  (`feedback_tool_oracles`).
- **Where.** A 5 × 5 grid image with one cell filled; the model must name
  its row and column. A model given wrong h/w positions reads the content
  and misplaces it, so this is the test of §4.
- **Prefill parity.** An image prompt's last-position logits against HF's,
  the bar the trunk was brought up to (`qwen38_flash_next.md` §11).
- **Survival.** The same image turn, then a reload that drops the chunks
  and re-prefills, then a replicated member answering a follow-up about
  the image.
- **Text unchanged.** The existing gates and `/sweep` on text, before and
  after, bit-identical: a text-only layout must not change one rotation.

---

## 10. Order of work

Each step lands complete, tested and committed before the next.

1. **Pack and loader.** The `mmproj` source in the pack request, the
   build-time Q8_0 requant and weight fold, the tower's tensors read by
   name, its VRAM line in the span budget. `PleConfig` parses
   `ple.image_token_id`.
2. **Preprocessing and the CPU reference tower**, with §9.1's preprocessing
   tests and §9.2's first parity test.
3. **The CUDA tower**: patchify, position-embedding add, the attention
   kernel, the fused epilogues; §9.2's second test.
4. **Rows in the wave**: the `PrefillWork` field through to the scatter
   after `gather_into`.
5. **MRoPE**: chunk advances and offset tables, `ChunkMeta._pad0`, every
   site of §4.4, the oracle. The text gates bit-identical.
6. **The conversation layer**: request parts, the image piece and its ids,
   blobs and `TurnDecl.images`, the re-prefill paths, text views, hashes,
   penalties, capabilities, blob retirement.
7. **Clients**: chat-completions image parts, zend `image_upload`,
   `zen_chat_send` images.
8. **§9.3's gates** on each machine.

---

## 11. Decisions for the author

Each has a default above; each is the author's to change.

- **The image-token ceiling** (§2.2): 4 096 per image. HF allows 16 384.
- **Resident tower** (§3.1): 0.48 GB of VRAM permanently, against ~40 ms
  per image on the PCIe-limited cards if it streams.
- **Re-encode on re-prefill** (§7.4) rather than storing encoder output.
- **Erf GELU in the merger** (§1.3), following HF over llama.cpp.
