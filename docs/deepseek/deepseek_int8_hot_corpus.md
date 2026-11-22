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

Total **576 B/entry** of int8 and BF16 payload (vs 2048 f32 → 3.6× smaller hot; vs 512 all-int8 → +12.5%
to buy back the rope precision), plus the 14 per-band f32 scales (56 B) = **632 B/entry** stored, and the
f32 Indexer key (512 B) beside it. No f32 copy of the latent
is kept: the int8+BF16 two-region cache is the canonical stored form.

Gallery buffers (`FloatGallery`, `latent_moe/gallery/mod.rs`):
- `nope_i8`    `[cap, NOPE_DIM=448]` u8      — nope int8
- `nope_scale` `[cap, NOPE_BANDS=14]` f32    — per-nope-band amax scale
- `rope_bf`    `[cap, ROPE_DIM=64]` bf16     — rope bands, pre-rotation

There is no f32 `attn` buffer: the two-region cache IS the stored latent. Reference and
bench callers that want dense f32 rows reconstruct them from it
(`FloatGallery::two_region_rows_f32`).

## Read contract (decode + prefill comp path)

For entry `gid`, dim `d`:
- `d < NOPE_DIM`: `nope_i8[gid][d] * nope_scale[gid][d/32]`
- `d >= NOPE_DIM`: `bf16→f32 rope_bf[gid][d-NOPE_DIM]`
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
   buffers — no per-wave build. Validate the paged gates + `wave_prefill_then_decode_end_to_end`.
4. **Prefill** pre-pass reads the two-region cache, dequant → rope → requant into
   the baked `comp_i8`/`comp_v8`. Validate prefill parity + `wave_prefill_then_decode_end_to_end`.
5. **Tier flip.** The int8 two-region
   cache + Indexer `keys` are the HOT tier — GPU-resident below `HOT_ENTRY_CAP`,
   spilling together to CPU RAM past `HOT_ENTRY_CAP` (8192 entries; re-heated per query by `gather_corpus`,
   ~576 B/entry of latent payload — 3.5× cheaper than the old f32 latent). `signs`/`pos` stay GPU at
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
- **Stage 3** — prefill closes the divergence: the nope bands already agree (both int8 of f32); the pre-pass sources the two-region cache directly, so both paths derive from ONE representation. ✅
- **Stage 4** — the `FloatGallery` builds the two-region cache on append (`build_corpus_cache_into`) and serves it via `gather_corpus`; decode and prefill are unified on `&CorpusCache` (no per-assembly rebuild). ✅
- **Stage 5** — tier flip: the two-region cache + Indexer `keys` are the HOT tier, GPU below `HOT_ENTRY_CAP` and spilling together to CPU RAM past it (`gather_corpus` re-heats the bounded selection); `signs`/`pos` stay GPU at any depth. The f32 latent is not kept at all (the two-region cache is canonical). Hot VRAM is bounded at unbounded depth. ✅
- **Cleanup** — removed the dead uniform builder (`latent_quant_corpus_range_kernel`), its FFI (`run_latent_corpus_cache_build`), and `quant_bands_corpus`; repurposed `corpus_cache_is_position_free` to the two-region format.

**Net:** the hot retrieval artifact is the position-free int8+BF16 cache (~576 B/entry vs 2 KB f32); the compressed rope bands match the window tier's BF16 (no √2 int8 margin); decode and prefill share one representation (divergence closed); there is no separate f32 archive.

**Validation:** the `latent_moe::paged` tests (`tests_oracle.rs`: `corpus_cache_two_region_bytes`, `corpus_cache_is_position_free`; `tests_prefill.rs`: prefill↔decode parity), the gallery spill/gather tests (`gallery/tests_selection.rs`, `gallery/tests_gather.rs`), and the wave gates in `latent_moe/wave/tests_prefill.rs` (`wave_prefill_then_decode_end_to_end`, `wave_prefill_state_matches_decode_steps`).

## Open items

- **`latent_moe/footprint.rs` accounts the corpus as an f32 pair** (`CORPUS_DTYPE = F32`, a
  `(head_dim + index_head_dim) · 4` = 2560 B row) while the gallery stores this two-region cache (632 B)
  plus f32 keys (512 B) and a sign/position index. `deepseek_kv_footprint` and `ratio_vs_fp16_linear`
  therefore overstate the resident corpus (`deepseek_batched_paged_attention_plan.md` Open items).
