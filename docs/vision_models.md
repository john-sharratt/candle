# Images for any model with a Qwen3-VL tower

`docs/flash_next_vision.md` brought images to Qwen3.8-Flash-Next. This document
makes that work a surface any model plugs into, and plugs in the second one,
Qwen3.6-35B-A3B. Where the two documents differ, this one is the authority for
what is shared and the other for what Flash-Next does inside it.

## 1. Why one surface fits both

Both releases carry the same tower — an unmodified Qwen3-VL ViT — and the
same multimodal rotary scheme in the trunk. Read off each checkpoint's
`config.json` and `mmproj` header:

| | Flash-Next | Qwen3.6-35B-A3B |
|---|---|---|
| tower `depth` / `hidden_size` / `intermediate_size` / heads | 27 / 1152 / 4304 / 16 | 27 / 1152 / 4304 / 16 |
| `num_position_embeddings`, `patch_size`, `spatial_merge_size` | 2304, 16, 2 | 2304, 16, 2 |
| `out_hidden_size` (the trunk's width) | **2560** | **2048** |
| image / vision-start / vision-end ids | 248056 / 248053 / 248054 | 248056 / 248053 / 248054 |
| trunk head dim, rotary share | 256, ¼ → 64 dims | 256, ¼ → 64 dims |
| `mrope_section`, interleaved | `[11, 11, 10]`, yes | `[11, 11, 10]`, yes |
| `rope_theta` | the checkpoint's, read at load | 10 000 000, read at load |
| preprocessing | `Qwen2VLImageProcessorFast`, mean/std 0.5 | the same |

So the tower, the preprocessing, the grid and positions, the turn of each
image row by its place in the image (`qwen_vision/rotation.rs`), the virtual
ids and their registry, the scheduler's spans, the conversation layer, the
guest that runs the tower and everything kept in the substrate are one
implementation. The tower's output width is the only number that differs, and
the tower reads it from its own `mm.2` projection.

A tower whose geometry is not this one — another depth, another patch size —
is refused at load by the tensor shapes `VisionTower::load` asks for, not
mis-read.

## 2. What a model plugs in

Three things, each in the model's own files.

### 2.1 Its tower: a `VisionTowerSource`

`qwen_vision/source.rs`. A constant in the model's file naming where its
tower comes from and what it is called:

```rust
pub struct VisionTowerSource {
    /// Names the tower wherever its output is kept: an image's rows are the
    /// tower's, so two towers' rows for one image are two records.
    pub name: &'static str,
    pub repo: &'static str,
    pub rev: &'static str,
    /// The `mmproj` GGUF: (file, bytes, SHA-256).
    pub file: (&'static str, u64, &'static str),
    /// The text model's tokenizer — the one whose `<|image_pad|>`s its rows
    /// stand for, which its pack records.
    pub tokenizer: (&'static str, &'static str),
}
```

`source.pack(mode)` fetches the `mmproj`, holds it to its length and digest,
and builds its twinned model pack (`model_pack/twins.rs`); `source.cached_pack(mode)`
answers from the machine alone, each remembered once a process so the drain
never re-reads a pack's header.

| Model | Constant | `mmproj` |
|---|---|---|
| Flash-Next | `quantized_qwen38_moe::FLASH_NEXT_VISION`, `name = "qwen3.8-flash-next"` | `unsloth/Qwen3.8-Flash-Next-GGUF` @ `f5d175de…`, `mmproj-F16.gguf`, 904 004 000 B |
| Qwen3.6-35B-A3B | `quantized_qwen36_moe::QWEN36_VISION`, `name = "qwen3.6-35b-a3b"` | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` @ `5bc3e238…` (the checkpoint's own pinned commit), `mmproj-F16.gguf`, 899 283 584 B, `71f3cbc1…` |

### 2.2 Its spec names the tower

`ModelSpec::vision: Option<&'static VisionTowerSource>`. A spec that names
one gets the vision guest registered with its engine
(`models/builder/model_guests.rs`), and its engine answers
`ConversationEngine::vision()` with it, so a turn's images are read by that
tower and kept under its name. A spec that names none takes no images: its
turn is refused before any guest is asked.

The 35B's two presets both name `QWEN36_VISION`: the stock checkpoint, whose
release it is, and the AntiLoop/StyleTune hybrid, both of whose trunks are
fine-tunes of that checkpoint's and keep its embedding geometry.

### 2.3 Its engine takes image rows

A prompt reaches an engine with each image row named by a **virtual id**
(`flash_next_vision.md` §4.1). Every engine does the same three things with
them, from shared code (`qwen_vision/wave_images.rs`), and differs only in how
its embedding is written:

1. **Resolve.** `take_wave_images` over the wave's host ids: each virtual id's
   row values laid back to back, each id rewritten to `<|image_pad|>`, and the
   runs of one image's consecutive rows listed. `count_image_rows` prices the
   wave first (`WaveWidth::image_rows`), so the tier refuses a wave it cannot
   hold.
2. **Embed.** The engine's own:
   - Flash-Next's table is on the card (`DeviceEmbedding`): the rows are the
     gather's second table, one launch.
   - The 35B's table is host-mapped (`EmbeddingTable::HostMapped`): the gather
     reads a placeholder row (id 0) at each image position, and one
     `rows_scatter` launch (`latent_moe/scatter.rs`, a tuned kernel — nothing
     new is written) copies each run's rows over the placeholders
     (`qwen35/forward/images.rs`). The rows are uploaded in the residual's own
     type, so the copy converts nothing. The draft head's embedding of the same
     ids gets the same copy: HF's head reads `inputs_embeds`, which hold the
     image there.
3. **Turn.** `image_rotation(runs, row_base, rope_dim, theta, …)` builds the
   prefill group's `ImageRotation`, handed to the shared prefill attention
   with `BatchedAttentionParams::with_image_rotation` — the trunk's attention
   layers and the draft head's alike.

A wave with no virtual id takes none of this: no upload, no launch.

## 3. What is kept, per tower

The image's record is one per image: `vision/image/<id>` names its original,
its thumbnail and **each tower's rows** — `"rows": {<tower>: <sha256>}` — and
says when it was kept (`vision/store.rs`); who sent it and which timelines use
it are `vision::tenancy`'s.

The rows are a tower's output. A turn reads them only when its engine's tower
wrote them (`load_kept(…, tower)`); an image only another tower read is read
again by this one, from its kept original, and its rows are kept beside the
other tower's. A tower reading an image again replaces only its own rows. The
recurrent catch-up replaying a stored turn asks for its engine's tower too: a
`Sequence` carries the engine's `VisionTowerSource`, and `stored_spans` reads
that tower's rows, refusing — for that sequence alone — a turn whose image its
tower never read, rather than handing a wave rows of another width.

Every blob a record names is named back by a key of its own,
`vision/blob/<sha256>/<id>` (`vision/blob_refs.rs`): blobs are
content-addressed, so two files of the same pixels share their rows and
thumbnail, and a file kept elsewhere — an npcd picture sent to Zen — is the
same blob. An image releases its names when it is retired, and a blob is
retired only once nothing names it; a blob already held before any image
named it is named `outside` as well, which is never released.

## 4. Tests

- **Exact values.** The source's pack name and request; the tower's width read
  from `mm.2`; `take_wave_images` and the rotation tables as before, from their
  new home, and another width's rows refused; the row scatter placing each run
  over its placeholders and leaving every other row bit for bit; the
  placeholders in the 35B's host inputs; each tower's rows kept beside the
  other's; a blob two images name going with the second; the tenancy keys by
  timeline.
- **Each tower on the card, from its pack**, and as a guest from its own
  ground: one finite row per `<|image_pad|>`, `out_hidden` wide — 2560 and
  2048.
- **The 35B end to end** (`zend/tests/vision_qwen36.rs`, GPU, `#[ignore]`d):
  an engine on the stock checkpoint and two image turns, each a 5 × 5 grid
  with one filled cell, the answer naming the cell; then a third turn naming
  the first image by its kept id, whose rows come from the substrate. Two
  cells right is a one-in-625 chance for a model not seeing them. The
  Flash-Next gates run unchanged, text waves bit-identical.
- The tests that read a pinned tower are `#[ignore]`d: their first run fetches
  both `mmproj`s (~1.8 GB) and builds their packs.

## 5. Adding a third model

A model whose release carries this tower:

1. Pin its `mmproj` as a `VisionTowerSource` in its model file, and confirm
   the table in §1 against its `config.json`.
2. Name it in its presets' `ModelSpec::vision`.
3. In its engine's wave: `take_wave_images`, its embedding's way of placing
   the rows (§2.3), `with_image_rotation` on its prefill attention, and
   `image_rows` in its wave price.
4. Its tests per §4.

A model whose tower is not this one needs its own tower type behind the same
source and guest — the rest of the surface is unchanged.
