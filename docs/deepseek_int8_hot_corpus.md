# DeepSeek-V4 Latent: int8-hot corpus + BF16 rope bands (review items B + D)

## Motivation

Two coupled problems in the compressed-corpus path, from the kernel review:

- **B — the f32 gallery is the hot artifact.** `attn` f32 (2 KB/entry) is what the
  decode `CorpusCache` builds from every wave and what the prefill pre-pass reads.
  At substrate scale it's the single largest resident thing (≈51 GB at 25M entries
  vs 12.8 GB for the int8 cache). And decode + prefill independently derive int8
  from it, so the two int8 representations of the same entry differ (a documented
  but avoidable divergence).
- **D — the compressed rope bands are ~½ bit behind the window tier.** Bands 14–15
  (the 64 RoPE dims the reference protects with BF16) are stored int8 with the
  rotation-invariant pair-magnitude scale, a √2-looser bound than component amax.
  The window ring stores those same dims in BF16 — so the compressed tier is
  behind the window tier on exactly the protected dims.

## Target format — the position-free int8-hot corpus cache

Per gallery entry, the HOT artifact is the position-free cache (built once on
append, rotated at read):

| Region | Dims | Format | Bytes | Notes |
|---|---|---|---|---|
| NOPE bands 0–13 | `[0, 448)` | int8, per-band **component amax** scale | 448 | position-free (never rotate); 7-bit, beats the window's FP8 nope |
| ROPE bands 14–15 | `[448, 512)` | **BF16** (pre-rotation) | 128 | float, 8-bit mantissa, matches the window tier; no √2 margin, no clip |

Total **576 B/entry** (vs 2048 f32 → 3.6× smaller hot; vs 512 all-int8 → +12.5%
to buy back the rope precision). The canonical `attn` f32 stays for merge/rebuild
but becomes ARCHIVAL — always CPU-resident, off the hot substrate.

Gallery buffers:
- `attn_i8`   `[cap, NOPE_DIM=448]` u8      — nope int8
- `attn_scale``[cap, NOPE_BANDS=14]` f32    — per-nope-band amax scale
- `attn_rope` `[cap, ROPE_DIM=64]` bf16     — rope bands, pre-rotation
- `attn`      `[cap, HEAD_DIM]` f32          — canonical, now ARCHIVAL (always CPU)

## Read contract (decode + prefill comp path)

For entry `gid`, dim `d`:
- `d < NOPE_DIM`: `attn_i8[gid][d] * attn_scale[gid][d/32]`
- `d >= NOPE_DIM`: `bf16→f32 attn_rope[gid][d-NOPE_DIM]`
Then rotate the rope dims at the entry's position and re-quantize into the int8 QK
operand (decode: per-slot pos; prefill: baked at `comp_pos`). Both paths now derive
from the SAME position-free bytes → the divergence closes (only `comp_v8`'s
corpus-global PV scale remains, already documented).

## Stages (each a green checkpoint)

1. **Build kernel + mirror.** `latent_build_corpus_cache_kernel`: f32 → int8-nope
   (component amax) + bf16-rope. CPU mirror `build_corpus_cache`. Byte-exact test.
2. **Gallery cache.** Add the three buffers + build-on-append; `gather_corpus`
   returns `(attn_i8, attn_scale, attn_rope, pos)` compacted, tier-aware.
3. **Decode** reads the two-region cache (kernel gains `comp_rope`; `comp_i8` is
   nope-only, `comp_scale` is `[G, NOPE_BANDS]`). `CorpusCache` holds the gathered
   buffers — no per-wave build. Validate paged gate + wave_paris.
4. **Prefill** pre-pass reads the two-region cache, dequant → rope → requant into
   the baked `comp_i8`/`comp_v8`. Validate prefill parity + wave_paris.
5. **Tier flip.** `attn` f32 is CPU-archival at all depths; the int8 two-region
   cache + Indexer `keys` are the HOT tier — GPU-resident below `HOT_ENTRY_CAP`,
   spilling together to CPU RAM past it (re-heated per query by `gather_corpus`,
   ~576 B/entry — 3.5× cheaper than the old f32 pair). `signs`/`pos` stay GPU at
   any depth so the BDP scan never touches RAM. This keeps the resident footprint
   bounded at unbounded corpus depth. Validate memory + a large-ingest run.

## Non-goals / preserved invariants
- Pair-magnitude stays the rotation-invariant scale wherever int8 rope bands remain
  (none, after D — rope is BF16). Nope int8 uses component amax (never rotates).
- `NOPE_DIM % SUB == 0` (band boundary) still required.
- Causality/window bounds stay ABSOLUTE; RoPE window ≤ 1M (no rebasing).

---

## STATUS: COMPLETE (all 5 stages + cleanup)

- **Stage 1** — `latent_build_corpus_cache_kernel` + FFI + CPU mirror `build_corpus_cache` + byte-exact test `corpus_cache_two_region_bytes`. ✅
- **Stage 2** — decode reads the two-region cache (`comp_i8`→`nope_i8`+`nope_scale`+`comp_rope`); `CorpusCache` holds the three buffers + `comp_pos`. ✅
- **Stage 3** — prefill closes the divergence: the nope bands already agree (both int8 of f32); the pre-pass now sources the two-region cache directly (or, transitionally, rounds the rope tail through BF16), so both paths derive from ONE representation. ✅
- **Stage 4** — the `FloatGallery` builds the two-region cache on append (`build_corpus_cache_into`) and serves it via `gather_corpus`; decode and prefill are unified on `&CorpusCache` (no per-assembly rebuild). ✅
- **Stage 5** — tier flip: `attn` f32 is CPU-archival (built from the incoming GPU rows, then archived — never in VRAM); the two-region cache + Indexer `keys` are the HOT tier, GPU below `HOT_ENTRY_CAP` and spilling together to CPU RAM past it (`gather_corpus` re-heats the bounded selection); `signs`/`pos` stay GPU at any depth. Hot VRAM is bounded at unbounded depth. ✅
- **Cleanup** — removed the dead uniform builder (`latent_quant_corpus_range_kernel`), its FFI (`run_latent_corpus_cache_build`), and `quant_bands_corpus`; repurposed `corpus_cache_is_position_free` to the two-region format.

**Net:** the hot retrieval artifact is the position-free int8+BF16 cache (~576 B/entry vs 2 KB f32); the compressed rope bands match the window tier's BF16 (no √2 int8 margin); decode and prefill share one representation (divergence closed); the canonical f32 is archival.

**Validation:** `deepseek4::paged::tests` (22/0, incl. `corpus_cache_two_region_bytes` + prefill↔decode parity), `corpus_cache_is_position_free` (CPU), gallery spill/gather tests, `wave_paris` (3/3) — all green.
