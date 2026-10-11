# Self-Contained Model Packs

> **Status — Built (2026-10-10).** `candle-transformers/src/models/model_pack/`
> (container, request, resolution, per-family build), the per-pair fingerprint
> in `models/repack_fingerprint.rs`, and every loader and call site moved over
> (§6, §10). Makes the repacked pack the model's only file at run time, so a
> checkpoint's GGUF is needed once, to build the pack, and is then released.
> Two changes, which are worth having together:
>
> 1. **The pack holds the whole model** — every tensor, the metadata, and the
>    tokenizer — not only the routed experts or streamed layers (§3).
> 2. **The repack fingerprint is per format**, so a change to one quantisation's
>    repack invalidates the packs that contain that format and no others (§4).
>
> Supersedes `docs/expert_cache_design.md` §5.2 (where the pack lives) and
> refines its §5.6 (the fingerprint). Every disk figure below is measured on the
> RTX 4090 Mobile 16 GB dev machine on 2026-10-10.

---

## 1. Why

Every model is stored at least twice, and a MoE model nearly three times over.

A pack is a cache derived from a GGUF, and it can be neither validated nor
completed without that GGUF:

- **Validation reads the source.** `PackIdentity::of` hashes the first 4 MiB of
  the GGUF mapping and takes its length (`expert_lre/pack/mod.rs:77`,
  `layer_stream/pack/mod.rs:93`), on every boot.
- **The pack is partial.** The routed experts (or, for the layer-streamed dense
  models, the streamed projections) are in it; nothing else is (§2).

So the GGUF stays on disk for the life of the model, and the experts — most of a
MoE checkpoint — sit there twice: once in the GGUF, once repacked. Qwen3-30B-A3B
is 17.3 GB of GGUF plus 15.9 GB of pack.

Measured on 2026-10-10:

| Where | GB | What |
|---|---:|---|
| `~/.cache/huggingface/hub` | 438 | GGUFs ~253, packs 185 |
| `~/.cache/zend/models` | 170 | Flash-Next artifact 88 + its pack 31; 51 stale (§9) |
| **Model storage** | **608** | |

Two further costs follow from the pack living beside the GGUF:

- **Two cache roots.** A GGUF downloaded by the gates (`hf-hub`) lands in the HF
  cache; one downloaded by zend lands in `~/.cache/zend/models`; a prepared
  artifact (Flash-Next) can only live in the latter. The pack follows its GGUF
  into whichever it is.
- **Stale packs accumulate.** The pack's name omits the repack fingerprint by
  design (`expert_lre/pack/mod.rs:698`), but it carries the int8 mode and the
  narrowing, and the layer-pack layout has changed under the same checkpoint —
  41 GB of packs on this machine were not read by today's full sweep.

## 2. What the GGUF is still needed for today

| Need | Where | Who |
|---|---|---|
| Identity check (length + 4 MiB fletcher32) | `expert_lre/handle.rs:648`, `qwen35/quantized_loader.rs:316` | every packed model |
| Pinned MoE layers — **repacked from the GGUF on every boot**; the pack holds no records for them | `expert_lre/startup.rs:264-320`, `pack/header.rs:25-29` | every MoE model |
| Pinned layer head (`PINNED_LAYERS = 2`), per-layer residues | `qwen35/layer_loader.rs:209-213`, `quantized_weights.rs:1738` | layer-streamed dense models |
| Metadata: architecture, shapes, rope, expert counts, chat template, `general.*` | e.g. `quantized_qwen3_moe.rs:1002-1039`, `candle-conversation/src/models/builder.rs:1587-1620` | every model |
| Attention, norms, router, shared expert, dense FFN, embedding, output head | e.g. `quantized_qwen3_moe.rs:1346-1447`, `qwen4exp/engine.rs:225-419` | every model |
| Tokenizer vocabulary check (`tokenizer.ggml.tokens` vs `tokenizer.json`) | `builder.rs:1493-1576` | every model |
| Flash-Next n-gram / PLE table, **read at run time** through a 2 GiB row cache | `qwen4exp/loader.rs:151-208` | Flash-Next |
| `config.json` beside the GGUF | `quantized_qwen3_moe.rs:744-776` | Qwen3-30B |

`tokenizer.json` itself comes from a separate repo for every model
(`builder.rs:1493`), so even a model whose weights were self-contained would
still reach for the HF cache for its tokenizer.

## 3. The model pack

### 3.1 One file per model and numeric mode

A **model pack** is one file holding everything a load reads:

```text
<cache>/<repo-with-dashes>/<stem>.<mode>[.n<N>].pack.gguf
```

- `<mode>` is the int8 mode the records target (`off`, `performance`,
  `precision`) — the records differ per mode, so the file does.
- `.n<N>` is the layer-stream narrowing, present only when the tight-VRAM
  schedule is active (`qwen35/layer_loader.rs:144`), because it changes record
  lengths.

**The container is a GGUF part followed by the record sections.**

```text
GGUF part   every tensor but the experts and the streamed projections,
            the checkpoint's metadata, the tokenizer, the provenance,
            and where each section starts            (general.alignment 4096)
experts     the expert section, every layer's records       (routed models)
layers      the layer section, every layer's records        (streamed dense)
```

- Every dense tensor — attention, norms, router, embedding, head, residues, the
  PLE table — is stored **verbatim** in the GGUF part, under its original name
  and quantisation. The loaders' existing tensor lookups read it unchanged, and
  a GGUF reader opens the file as an ordinary GGUF that ends where the part
  does.
- Each record section is the section format the expert and layer caches
  already read — `CNDLXPK2` (`expert_lre/pack/header.rs`) and `CNDLLYR2`
  (`layer_stream/pack/header.rs`), header then records then checksum trailer —
  appended at a 4 KiB-aligned offset. `zen.pack.experts.offset`/`.len` and
  `zen.pack.layers.offset`/`.len` record where (`model_pack/keys.rs`). The
  alignment is the one `direct_io` requires, so a cold read still lands directly
  in a pinned slot with no bounce buffer, and `PackHeader::encode`/`decode` and
  their raw-byte tests are reused, not re-specified.
- `tokenizer.json` is a metadata string, `zen.tokenizer.json`, with its repo and
  revision beside it. `zen.pack.version`, `zen.pack.int8_mode`,
  `zen.pack.narrow`, `zen.pack.checkpoint_bytes` and `zen.pack.gguf_len` say what
  the pack is.

Sections after the part rather than records as `U8` tensors inside it, because
the sections are written by streaming writers that never seek (each appends its
checksum trailer last), and a GGUF tensor table must be complete before the
first tensor's bytes. Appending keeps both writers sequential and leaves the
GGUF part an ordinary GGUF. Writer: `candle-core/src/quantized/gguf_writer.rs`;
reader: `model_pack/open.rs`.

### 3.2 Every layer, including the pinned ones

The pack today omits the leading `PINNED_LAYERS` MoE layers on the grounds that
they are never reloaded (`pack/header.rs:25-29`), so they are repacked from the
GGUF on every boot (`startup.rs:264`). In a self-contained pack there is nowhere
else to get them, so **every layer has records**. Two consequences:

- `startup_pinned_prefix` reads records from the pack instead of repacking from
  the mapping — the same read `startup_from_pack` already does for the rest.
- The pinned count stops being part of the pack's identity. A change to
  `PINNED_LAYERS` no longer invalidates any pack.

The same holds for the layer packs' pinned head.

### 3.3 Dense models with nothing to repack

Qwen2-0.5B, Llama-3.2-3B, Llama-2-7B and Qwen3-8B load straight from the GGUF
(`quantized_qwen2.rs:543`, `quantized_llama.rs:1078`, `quantized_qwen3.rs:806`).
Their model pack is the same container with no record tensors: the GGUF's tensors
verbatim, plus the tokenizer and the provenance block. One resolution path for
every model is worth the copy; it costs nothing at steady state, because the
source is deleted after the build (§5.3).

### 3.4 Extra source files are folded in

Several models load more than one file: the MTP sidecar, the gate donor and the
tensor-override files (`qwen35/quantized_loader.rs:157-240`), and the DSpark
drafter (`latent_moe/engine.rs:360`). The build folds their tensors in verbatim,
and the provenance block (§4.1) records each source. The drafter stays a separate
model pack — it is a separate model with its own consumers.

## 4. Identity

### 4.1 Provenance replaces the source

With no source on disk, "which checkpoint is this" has to be recorded at build
time rather than re-derived at load. The provenance block, as metadata:

| Key | Value |
|---|---|
| `zen.source.count` | how many sources follow |
| `zen.source.<i>.role` | `checkpoint`, `mtp`, `gate-donor` or `override:<tensor>` |
| `zen.source.<i>.repo` / `.rev` / `.file` | one entry per source file (§3.4). A caller's local file is filed under a `local/<dir>-<digest>` label, its `rev` the file's length and modification time (`model_pack::local_rev`). Flash-Next's prepared artifact is named by its recipe digest in `rev` (`qwen4exp/prepare/fetch.rs`) |
| `zen.source.<i>.len` | the source's length |
| `zen.source.<i>.sha256` | **SHA-256 over the whole source file** |
| `zen.tokenizer.repo` / `.rev` | where the tokenizer came from — a local `tokenizer.json` pinned like a local source |

A whole-file hash is affordable because it is computed **once, at build time**,
on its own thread per source, beside the build (`model_pack/digest.rs`). It
cannot ride on the build's own reads — those follow the composition and the
repack, not the file's order — so it is a second sequential read, overlapped
with the repack rather than ahead of it. The metadata is laid out with a
digest-length placeholder per source, and the hashes are written over them in
the temp file before the rename publishes the pack. The 4 MiB sample was only ever a compromise against hashing at
every boot. It also had an unmeasured exposure that this removes: on a
large-vocabulary GGUF, the `tokenizer.ggml.*` arrays at the front of the file
may push the tensor table past the sampled window.

**The load check** is then against the *request*, not a file: the caller names
`(repo, rev, file)` for each source and `(repo, rev)` for the tokenizer (the
gates' pins, zend's `ModelSpec`), and the pack is accepted when its provenance
names the same — the revision only where the request pins one. The recorded
hash is provenance, not a check: by the time a pack is opened its sources are
gone. Content is pinned by revision instead — a hub commit, a local file's
length and mtime, a prepared artifact's recipe digest. Where a source table
carries SHA-256 pins — Flash-Next's Q8_0 split (`quantized_qwen38_moe.rs`) — the
prepare step checks every file against them before the merge
(`qwen4exp/prepare/store.rs`), and the artifact it produces is what the pack
names.

### 4.2 The fingerprint is per format

The expert cache's original `repack_fp` was one FNV-1a over every
`(source dtype → target dtype)` pair the engine supports, swept regardless of
the model, and deliberately so: "a pack written by a binary that repacks Q5_K
differently is stale whether or not today's model contains a Q5_K tensor".

That rule is right when invalidation is cheap — a stale pack costs one local
repack. It is wrong once the source is gone, because every repack change then
costs re-downloading **every model**. So:

- **Each pair is hashed on its own.** `pair_fp(src, tgt)` = FNV-1a over the
  sweep's version tag, the reference geometry, the pair, and the repacked bytes
  (or the refusal), exactly as now but not folded together.
- **The pack records the pairs it contains**, with their hashes: a table of
  `(src u32, tgt u32, fp u64)` in the header. The set is read off the pack's own
  record geometry — every projection's source and repacked dtype — so it is a
  fact about the file, not an inventory someone maintains.
- **Open recomputes only those pairs.** Any mismatch marks the pack stale.
  Both sections share the hashing and the check (`models/repack_fingerprint.rs`),
  which build without CUDA so the section formats' tests run anywhere; only
  producing a fingerprint from this build's repack needs the device.

**Why this does not reopen the hole §5.6 of the expert-cache design closed.** That
hole was a repack change that leaves sizes and offsets alone but changes bytes.
Per-pair hashing still catches it for every pair the file contains. The original
rule protected against the *inventory* being wrong — a pack containing a format
the sweep did not cover. Here the inventory is read from the pack itself, so it
cannot be missing a format the pack holds. And a change to code shared by every
pair (a common permutation, the slot writer) moves every pair's hash, so every
pack that uses any of them is invalidated — as it should be.

**What stays global.** A change to the container or record layout bumps the
header `VERSION` (`pack/header.rs:228`) and invalidates every pack. That is
untargeted by nature and is rare; it is the one event that re-downloads
everything.

The layer packs use the same fingerprint today, XOR'd with the gate donor's
length (`qwen35/quantized_loader.rs:316-322`). The donor moves into provenance
(§4.1), and the layer records take the per-pair table like the expert records.

## 5. Where it lives, how it is found, how it is rebuilt

### 5.1 One cache root

`~/.cache/zend/models/<repo-with-dashes>/`, the root prepared artifacts already
use (`builder.rs:2001`). The HF cache is used only as a **transient download
area** for sources during a build.

`zend::download::cache_dir` and `builder::model_cache_dir` disagreed on how the
root is found — zend tried `XDG_CACHE_HOME`, `HOME`, then `USERPROFILE`; the
builder tried `USERPROFILE` then `HOME`. There is now one function,
`model_pack::cache_root` (`USERPROFILE`, then `HOME`), in the crate the gates,
the prepare step and the loaders share; `candle-conversation` re-exports it, and
zend calls it through that.

### 5.2 Resolution

One entry point resolves every model, for zend and the gates alike:

```text
model_pack(request, root, device, fetch) -> path        (model_pack/resolve.rs)
  1. a pack under <root>/<repo>/ whose name the request owns, whose provenance
     names the request's sources, whose mode and narrowing are this card's, and
     whose sections pass this build's VERSION and pair_fp checks → return it
  2. otherwise → build (§5.3), then return it
```

A `PackRequest` names the family (which sections the build writes), the
sources in order — the checkpoint, then any `mtp`, `gate-donor` or
`override:<tensor>` file — the tokenizer's repo and revision, and the int8 mode
or `None` for the one `Int8Mode::auto_sized` picks on the loading card. A pack
of another narrowing, or of another mode picked for another card, is left for
that card; any other mismatch is stale and is removed once its replacement is
published. `existing_pack` is step 1 alone, for a caller choosing between
requests by what is already built.

`fetch` is a `SourceFetch`: where a source comes from, where a tokenizer comes
from, and how a fetched source is released.

- **The gates** use `gate_pack` (`batch_test/test_helpers.rs`) over the same `HubFetch`,
  at every site that used to `hf_get` a checkpoint.
- **The engine builder** resolves through `ModelBuilder::resolve_model` (hub,
  cache first, through `model_pack::HubFetch` and `models::hub_download` — one
  resumable, timeout-protected download, never hf-hub's own, which has no read
  timeout) or
  `resolve_model_with` (`models/pack_source.rs`). A caller's local checkpoint is
  packed under a `local/…` label through `LocalFetch`, which never releases it;
  a path that is already a pack is used as it is. A gate donor is added to the
  request only when the checkpoint stores its DeltaNet gates below F32, judged
  from its header — or when a pack built with one is already there.
- **zend** passes `ZendFetch` (`zend/src/download.rs`): its own download step
  survives as the source fetcher, because it reports progress to the status
  pane.
- **Flash-Next** resolves through `prepared_engine_pack`, whose fetch is the
  prepared artifact (`qwen4exp/prepare/fetch.rs`) and which never prepares one —
  that is the forward gate's job.

### 5.3 Build

1. **Fetch the sources** into the HF cache — `hf-hub` for the gates, zend's
   streaming fetch for zend.
2. **Hash them** (SHA-256, §4.1), each on its own thread beside step 3, and
   record the digests as provenance before the pack is published.
   A source table's own SHA-256 pins are checked where that source is prepared
   (Flash-Next's prepare step, §4.1).
3. **Write the pack**: copy the dense tensors verbatim, repack the records for
   the requested mode and narrowing, write the headers, the per-pair table, the
   tokenizer and the provenance. Write to `*.partial`, fsync, rename — as the
   layer pack does now (`layer_stream/pack/mod.rs:540-546`).
4. **Open it** through the §5.2 check, so a pack that does not validate is never
   reported built.
5. **Delete the sources this build fetched.** Never a file the caller supplied by
   path (`--model`): the user's file is not ours to delete. Flash-Next's prepare
   already deletes its sources after the merge (`qwen4exp/prepare/build.rs:180`).

A build is per mode. A model used in two int8 modes — the AntiLoop hybrid's
Precision and Performance gates — fetches its source a second time for the
second mode. That is once per mode per machine, and it is the price of not
building, and storing, modes nobody asked for.

### 5.4 This supersedes "beside the checkpoint"

`docs/expert_cache_design.md` §5.2 puts the pack beside its GGUF so that it is
shared by every workspace and is deleted by the same act that deletes the model.
Both properties hold here, for a stronger reason: the pack *is* the model. That
section's other rule — that an embedder, an example or a test must never have a
large file appear beside its model without asking — becomes a rule about the
cache root. Nothing is written beside a caller-supplied file, and there is no
ephemeral pack any more: the builder's `expert_pack_dir` setting is gone, along
with the temp-file pack it chose between.

## 6. Load path, per family

| Family | Loader | Load path |
|---|---|---|
| Qwen3-30B-A3B | `ModelWeights::from_pack` (`quantized_qwen3_moe.rs`) | tensors and metadata from the pack's GGUF part, experts from its expert section; the mapping stays for the host-resident embedding table. No `config.json` beside a pack: the GGUF states context and RoPE base, and `norm_topk_prob` is the lineage's own `true` |
| Qwen3.5/3.6-35B, hybrids | `load_hybrid_pack` (`qwen35/quantized_loader.rs`), wrapped by each model file's `from_pack` | sidecar, donor and overrides folded in at build (`qwen35/pack_build.rs`); one mapping |
| Qwen3.5-0.8B/9B, Qwen3.8-27B | same, layer-stream path (`qwen35/layer_loader.rs`) | pinned head filled from the layer section (`LayerCache::fill_pinned`), residues from the GGUF part |
| Flash-Next | `Qwen4ExpGpu::load` (`qwen4exp/engine.rs`) | the PLE table is a dense tensor of the pack, read through the same row cache (`loader::open_cached_ple`) |
| DeepSeek-V4-Flash | `Engine::load` / `load_with_drafter` (`latent_moe/engine.rs`) | as the 30B; the drafter is its own plain model pack |
| Qwen2, Llama, Qwen3-8B | their `from_gguf` | the pack is the GGUF plus tokenizer and provenance, opened as a GGUF |

Every loader keeps reading metadata and dense tensors through `gguf_file::Content`
over the pack. The expert and layer caches have no `mmap` input for the records:
`ExpertCacheSetup` carries the open expert section (`expert_lre/section.rs`), and
the repack from a checkpoint lives in the build (`model_pack/build.rs`,
`model_pack/family.rs`).

## 7. What it costs

- **A repack change re-downloads the models in that format.** One or two models
  for a typical change to one quantisation. A container or layout change
  re-downloads all of them.
- **A second int8 mode re-fetches its source once** (§5.3).
- **The first build is slower than today's repack** by the dense-tensor copy.
  The SHA-256 is a second read of each source, but beside the repack rather
  than ahead of it, so it costs wall time only where it outlasts the build.
- **A boot is faster.** No identity sample read, no pinned-layer repack, and no
  GGUF mapping beside the pack for MoE models.

## 8. What it saves

Projected from the 2026-10-10 inventory, with stale packs removed, for the models
today's sweep and probes used:

| | GB now | GB after |
|---|---:|---:|
| MoE and layer-streamed models (packs in use + their GGUFs) | ~400 | ~165 |
| Small dense models | ~15 | ~15 |
| Flash-Next | 119 | ~90 |
| Stale files (§9) | ~92 | 0 |
| **Total** | **~608** | **~270** |

The "after" column is the in-use packs plus their non-expert tensors, which are a
small fraction of a MoE checkpoint. It is an estimate until the first builds are
measured; §10 step 5 measures it.

## 9. Stale files on this machine

Independent of this design, and safe to delete now:

- `~/.cache/zend/models/Qwen3-30B-A3B{,-Instruct-2507}-Q4_K_M.gguf` and the
  `Qwen3-30B-A3B-Q4_K_M.86cd….experts.pack` beside them (51 GB). They sit at the
  cache root, from before the cache was keyed by repo
  (`zend/src/download.rs:122-135`); nothing resolves there now.
- Packs not read by the 2026-10-10 sweep (41 GB): the 27B `layers.1.pack`
  (superseded by `layers.1.n64.pack`), the 9B `layers.0`/`layers.1` packs, the
  0.8B `layers.1` pack, and the non-MTP Qwen3.6-35B pack.

## 10. Build order

1. **Per-pair fingerprint** (§4.2), on the current packs. Self-contained work
   that is useful alone: a repack change stops invalidating unrelated packs.
   Unit test: the pair hashes are computed over a repack function supplied by
   the test, so a changed output for one pair moves that pair's hash and no
   other's — asserted against raw expected values.
2. **Records for every layer** (§3.2), on the current packs. Removes the
   pinned-prefix repack.
3. **The container** (§3.1, §4.1): writer and reader, with raw-byte tests of the
   provenance block and the per-pair table, and a round trip of a small
   synthetic model (`latent_moe::arch::test_arch` style) through
   build → open → tensor reads.
4. **Resolution and build** (§5): `model_pack`, the single cache root, source
   deletion. Test with a temp cache root: a hit, a provenance mismatch, a pair
   mismatch, and a caller-supplied path that is never deleted.
5. **Loaders and call sites** (§6), one family at a time, each held by its
   forward gate: the gate's ladder from the pack must match the same gate's
   ladder from the GGUF on the commit before (`/sweep`). Then measure §8.

Steps 1 and 2 change today's packs, so every pack on the machine rebuilds once
(locally, from the GGUFs still present). Step 5 deletes the GGUFs as each model
moves over.

## 11. Corrections this design carried

All made with the build:

- `docs/expert_cache_design.md` §5.2 and §5.6 point at this document.
- `qwen35/quantized_loader.rs` said the expert cache "keeps its own `Arc` and
  streams from it for the life of the model". It never did, and the loader that
  said it is rewritten over the pack (`load_hybrid_pack`).
- Comments that cited `docs/qwen38_layer_streaming.md` cite
  `docs/archived/qwen38_layer_streaming.md`, where it lives.
