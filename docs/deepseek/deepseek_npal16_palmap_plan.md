# DeepSeek-V4 Latent: NPAL=16 + 4-bit PalMap — Implementation Plan

Status: **PLAN / pre-implementation.** Supersedes the "4 palette sub-bands (128-d)"
geometry in `deepseek_batched_paged_attention_plan.md` (that doc is updated as a
deliverable of this work — see Phase 6).

## Goal

Move the single-latent (K≡V, MQA, HEAD_DIM=512) attention from **8 bands** to
**16 bands** (`SUB_DIM` 64→32), and change the per-dim palette map from a
**2-bit** field (names 4 palettes) to a **4-bit** field (names 16). This:

1. Gives per-band int8 scales at 32-dim granularity — finer error control, and
   re-aligns bands with the stock 32-dim quant-block size (`SUB=32` = exactly one
   `m16n8k32` MMA tile, and satisfies the prefill rank-pack `static_assert(SUB≤64)`
   that `SUB=128` violated).
2. Isolates the 64 RoPE dims `[448,512)` into a clean **2-band tail (14,15)**,
   so they can be protected (BF16/fine) independently of the compressible nope
   region `[0,448)`.
3. Makes the palette map faithful (4 bits names all 16 ids), re-enabling the
   **PalMap magnitude-sort regroup** — sort the nope dims by magnitude into
   bands 0–13 so each band quantizes tightly, while the rope tail stays pinned.
   This is the efficiency mechanism that was dropped at 8 bands because rope
   (a single band there) could not be excluded from the sort.

## Invariants / geometry (the load-bearing numbers)

| Quantity | 8 bands | 16 bands |
|---|---|---|
| `NPAL` / `N_BANDS` / `LATENT_N_BANDS` | 8 | **16** |
| `SUB_DIM = HEAD_DIM/NPAL` | 64 | **32** |
| RoPE bands (`ROPE_DIM=64` @ `[448,512)`) | band 7 | **bands 14,15** |
| pal_map density | 2-bit (4 dims/byte) | **4-bit (2 dims/byte)** |
| pal_map region | `[0,HD/2)` = k_pal[HD/4]+v_pal[HD/4] | **`[0,HD/2)` = one 4-bit map** (v_pal dropped, K≡V) |
| `kv_head_byte_size = HD/2 + NP*26` | 256+208 = 464 | 256+**416** = **672** |
| GIDs/head = `NPAL*2` | 16 | **32** |
| decode QK k-steps `SUB/32` | 2 | **1** |
| `KLB = SUB/DPT` (K stage) | 4 | **2** |
| `Q_THREADS_PER_BAND = SUB/32` | 2 | **1** |

**Record byte layout is offset-stable.** The 4-bit map (4 bits × 512 = 256 B)
exactly fills the old k_pal+v_pal region `[0,HD/2)`; the per-band block still
starts at `HD/2`. Only `NP*26` grows (208→416). No block offset moves. (The
`v_ptr/v_fmt/v_scale` 12 B/band are dead under K≡V; dropping them → `NP*14` is a
*separate optional* record-shrink, NOT part of this plan — keep `*26` for layout
stability.)

**4-bit map encoding.** `id = (m[d>>1] >> ((d&1)*4)) & 0xF`, 2 dims/byte, over
bytes `[0,256)`. Device u32 word holds **8 dims** (was 16), so `HD/8 = 64` words.

**GQA / Qwen path is frozen at N_PALETTE=4 and 2-bit — untouched.** Everything
below is gated on `single_latent` / `n_palette()`; the module `N_PALETTE=4`
const and the palette4 record (`candle-core/.../cuda.rs`, HD=128, 168-byte
record) are a *different* subsystem and stay 2-bit/4-palette.

---

## Staging strategy (why two encoding/geometry phases, not one)

The 4-bit-map change and the 8→16 geometry change are **independent axes**. We
land them in sequence so each is validated in isolation (TDD, per CLAUDE.md):

- **Phase 1** flips the map encoding to 4-bit **at NPAL=8** (map faithful but
  still identity/dead — `MAPPED=false`). This isolates the record-serializer /
  device-decoder / host-packer change and proves byte-parity before any geometry
  moves. All 20 currently-green gates must stay green; the 4 seal/regroup gates
  stay red (regroup not yet re-enabled).
- **Phase 2** flips geometry to NPAL=16 (identity layout, map still dead). Core
  attention must stay bit-exact; the seal path now runs at 16 identity bands.
- **Phase 3** re-enables the magnitude-sort regroup (map goes live, `MAPPED=true`)
  with rope pinned to bands 14,15. This is the efficiency payoff.
- **Phase 4** persistence, **Phase 5** validate + measure, **Phase 6** doc.

If regroup (Phase 3) is descoped, Phases 1–2 still deliver 16 identity bands
(finer per-band scales + 2 rope bands) with the map correct-but-dead.

---

## Phase 1 — pal_map: 2-bit → 4-bit (NPAL still 8, map dead)

### Host — `candle-transformers/src/models/deepseek4/paged.rs`
- `pal_of` (:1468-1471): `((m[d>>1] >> ((d&1)*4)) & 0xF)`.
- `identity_pal_map` (:1481-1487): return `[u8; HEAD_DIM/2]`; body
  `m[d>>1] |= ((d/SUB_DIM) as u8) << ((d&1)*4)`; drop the `&3` mask + the "DEAD /
  cannot name ≥4" doc (map is now faithful).
- `pack_map` test helper (:5003-5016): `[u8; HEAD_DIM/2]`, `m[d>>1] |= p<<((d&1)*4)`.
- Record writer `build_mapped` (:1697-1704): `chunk_maps: Vec<[u8; HEAD_DIM/2]>`;
  **single** `rec[..HEAD_DIM/2].copy_from_slice(&chunk_maps[chunk])` — delete the
  duplicate v_pal copy at :1704. Update comment :1702.
- Type-site sweep `[u8; HEAD_DIM/4]` → `[u8; HEAD_DIM/2]`: :1501 (`mapped_chunk_roundtrip`),
  :1582 (`build_mixed` maps), :1597 (`build_mapped`), :2179/:2470 (`win_maps`),
  :4455 (`map_out = zeros(HEAD_DIM/2)`).

### Host serializer — `candle-nn/src/kv_cache/chunked/meta_pool.rs`
- `kv_head_record_bytes` (:67-69): keep `(head_dim/4)*2 + n_palette*26` — the
  `(head_dim/4)*2 = head_dim/2` map region is unchanged. Update the :63-66 comment
  ("2-bit packing density … per side") to "one 4-bit map over head_dim/2".
- `serialize_kv_heads` (:86-153): `pal_bytes = head_dim/2` (single map, not two
  `head_dim/4` sides); identity pack `dst[pos + d/2] |= pal_idx << ((d&1)*4)`;
  clamp `min(n_palette-1)` — but see note below, identity uses `d/sub_hd` and
  `sub_hd` must be `head_dim/n_palette` not `head_dim/N_PALETTE` (fix :112).
  Update `debug_assert(head_dim>=4, "…2-bit…")` (:98-101) text.
- `gpu_chunks.rs` `kv_head_serialized_size` (:638-641) — mirror; keep byte-identical.

### Host builders — `candle-transformers/src/models/slot_state.rs`
- `fill_identity_pal_map` (:44-60): 4-bit pack; `pal_map packs 2 dims/byte` doc
  (:425-426); `debug_assert` text (:337). `KvHeadRecord` fixed `[_; N_PALETTE]`
  arrays (:93-105) stay 4 here **only if** this builder is GQA-only — verify;
  if it serves the latent, its arrays must key on the band count (Phase 2).

### Device — `candle-kernels/src/paged-decode/slot_types.cuh`
- Doc block (:31-43): one 4-bit `pal_map[HD/2]`, drop the `v_pal[HD/4]` line.
- `kvhead_v_pal_map` (:170-173): **remove** (v_pal region absorbed).
- `kvhead_k_pal_map` (:164-166): now spans `[0,HD/2)`.
- `pal_map_equal` (:178-188): `N_U32 = (HD/2)/4 = 64` words.
- `kv_head_byte_size` (:89-92): **unchanged** (`HD/2 + NP*26`).

### Device decoders — `candle-kernels/src/paged-latent/latent_common.cuh`
- `pal_map_identity_warp` (:82-88): 4-bit map = `HD/8 = 64` words → **2 words per
  lane** (not 1); replace `static_assert(HD/16==32)` with `HD/8`; replication
  const `0x55555555 → 0x11111111`; recompute the expected id per word
  (`word wi covers dims [wi*8,+8) → band wi*8/SUB`); `__all_sync` over both words.
- `pal_rank_of` (:94-114): `wi=d>>3` (8 dims/word); `sh=(d&7)*4`; `p=(w>>sh)&0xF`;
  `pat = 0x11111111u*p`; the zero-nibble popcount becomes
  `(~(x|(x>>1)|(x>>2)|(x>>3))) & 0x11111111u`; word loop spans 8 dims/word.
- `load_mapped_key_dims` (:173-195): functionally OK once `pal_rank_of` is 4-bit;
  `bp4/bf4/bo4[NPAL]` stay (grow with NPAL in Phase 2).

### Device assigner — `candle-kernels/src/quantize/latent_band_quant.cuh`
- `latent_assign_palettes` (:207-255): the 2-bit writer. Even at NPAL=8, move to
  4-bit here so the encoding is consistent: `s_map[HD/8]` words, `atomicOr(&s_map[d>>3],
  p<<((d&7)*4))`, `pal_map_out + chunk*(HD/2)`, init loop `d<HD/8`. **Also add an
  `n_palette` param** (`HD = sub_dim*n_palette`, not `*4`) — needed for Phase 2/3;
  thread it through the launcher (:468-485) and the Rust extern
  (`candle-kernels/src/simple/quantized.rs:1633-1642`) and the caller
  (`paged.rs:4458`). Update the `[n_chunks*HD/4]` doc → `HD/2`.

### Tests to update to 4-bit (Phase 1)
- `candle-kernels/tests/pal_rank_prefill_test.rs`, `paliter_test.rs`: 4-bit unpack.
- `paged.rs` assign gate (:4432-4499): `map_out` HD/2; `exp_map = pack_map(...)`
  4-bit (`exp_pal[d]=pos/SUB_DIM` unchanged).
- Mirror/mapped tests compile against the new `[u8; HEAD_DIM/2]` type.

### Phase 1 exit gate
Full `deepseek4::paged` at NPAL=8: the 20 core-attention gates stay green; the
mirror/pal tests pass with 4-bit; the seal/regroup gates remain red (regroup off).
`pal_rank_prefill_test` / `paliter_test` green.

---

## Phase 2 — geometry 8 → 16 (SUB=32, identity, map still dead)

### Constants (lockstep — all three, one commit)
- `paged.rs:32` `N_BANDS: 8→16`; `candle-nn/.../arena_table.rs:367`
  `LATENT_N_BANDS: 8→16`; `candle-kernels/.../latent_common.cuh:56` `NPAL: 8→16`.
- Rewrite the doc comments at each (8×64 → 16×32; rope = last two bands).

### Decode kernel — `latent_decode_kernel.cuh` (WARPS=8, 16 bands)
- **QK warp-budget (the crux, :399-417):** replace `if (warp < NPAL)` single-band
  with a **band-stride loop** `for (int p = warp; p < NPAL; p += WARPS)` — each of
  the 8 warps covers bands `{warp, warp+8}`; MMA count unchanged (2 bands × 1
  k-step = the old 1 band × 2 k-steps). `scores_p[NPAL][16][8]` → 16-wide (smem
  4KB→8KB, still <48KB). The `for(p<NPAL) lg+=scores_p[p]` combine (:443) and the
  dump combine (:425) are already NPAL-general — verify they now see all 16
  initialized.
- Q/K per-band staging (`Q_THREADS_PER_BAND`, `KLB`) already generalized — verify
  (QTB=1, KLB=2).
- `__launch_bounds__(WARPS*32, 4)` (:33): stays 256 threads; re-check the `,4`
  min-blocks against the +4KB smem (may settle to 3 blocks/SM — acceptable).

### Prefill kernel — `latent_prefill_kernel.cuh` (PF_WARPS=16, 16 bands)
- **QK warp-budget (:493-502, :591-617):** `warp<NPAL*PF_ROW_TILES` = 32 > 16
  warps, and `hgroup=warp/NPAL` collapses to 0 → head-group 1 never computed.
  Remap to **warp = band (0..15), loop the 2 row-tiles inside**: `if(warp<NPAL){
  int p=warp; for(int rt=0; rt<PF_ROW_TILES; ++rt){ hbase=rt*16; …qa_frag[rt]…;
  MMA; atomicAdd scores[hbase+…] }}`. `qa_frag` becomes `[PF_ROW_TILES][NKS][4]`
  (NKS=1). MMA count unchanged.
- smem scaleQ/scaleK grow (`PF_PASS_HEADS*NPAL`, `PF_KEYS*NPAL`): ~+2KB; verify
  2 blocks/SM still fit under the opt-in budget (`prefill_smem_bytes` auto-tracks).

### Corpus rope+quant kernels — the DPL=1 RoPE break
- `latent_common.cuh:371-417` (`latent_rope_quant_corpus_range_kernel`) and
  `latent_prefill_kernel.cuh:742-800` (`latent_rope_quant_corpus_kernel`):
  `DPL = SUB/32 = 1`, so `float v[DPL]` is size 1 and `rope_pair(v[j], v[j+1])`
  reads `v[1]` OOB, and the interleaved pair is no longer register-local.
  **Fix (partner-shuffle):** each lane owns 1 dim `d = band*SUB + lane`; for rope
  dims, fetch the pair partner via `__shfl_xor_sync(mask, x, 1)`, apply the
  rotation on `(even_lane_val, odd_lane_val)`, keep the lane's component. Freq
  index `(d-NOPE_DIM)>>1`. Per-band amax reduction stays a full-warp reduce (32
  lanes = 32 dims = one band) — already correct. `dmax[DPL]` → size 1.

### Host / tests
- Widen hand-written 8-element band-literal arrays → 16: `paged.rs:4053-4054,
  4173-4179, 5104-5105, 5186-5187`.
- `latent_reference_error_margins` (:1908): the per-64 block loop
  `for … in SUB_DIM/64` (:1956) degenerates to 0 iterations at SUB_DIM=32 →
  `eps_nope` stays 0 → assert trips. Change to `min(64, SUB_DIM)` or per-`SUB_DIM`.
- `band_thr` rope protection — `compress.rs:624-628`: set **both**
  `band_thr[N_PALETTE-1]` and `band_thr[N_PALETTE-2]` to `rope_thr` (rope now = 2
  bands). Fix the :624-625 comment. Update the seal-test rope gates
  `paged.rs:3408, 3436-3443` (last band → last two).
- Hardening: assert `n_kv_head == 1` where the single-latent path uses the const
  `GIDS_PER_HEAD`/`HeadGids::n_kv_head()` (`head_gids.rs:79-148`) — it only works
  today because n_kv_head==1; make the invariant explicit (candle-nn audit).
- `backing.rs identity_scale` (:876): size by `n_palette()` not `N_PALETTE`
  (currently relies on `unwrap_or(1.0)` for bands 4..15).
- Stale golden tests `meta_pool.rs:638, 716-717, 731, 809` (`104=4*26`,
  old signatures) — update to the current 3-arg/11-arg API and 16-band goldens.

### Phase 2 exit gate
Core attention bit-exact at NPAL=16 identity: decode/prefill/rope/window/SWA/
splits + all `mirror_bit_exact_*` green. `wave_paris` → "Paris". C3 StoryRewrite
(identity seal) passes; measure the compression ratio (identity 16-band baseline).

---

## Phase 3 — re-enable PalMap regroup (nope-only sort, pinned rope)

This reverses the compress.rs:673-680 "regroup dropped" decision, made valid by
the 2-band rope tail. **Design:** magnitude-sort the **448 nope dims `[0,448)`**
into bands 0–13 (each 32-dim band gets dims of similar magnitude → tight int8
scale); **pin the 64 rope dims `[448,512)` to bands 14,15** (excluded from the
sort, protected by the rope thresholds). The pal_map thus permutes only the nope
region; rope dims map identically.

- `latent_assign_palettes` (kernel): partition **only `pos ∈ [0, NOPE_DIM)`** by
  descending amax into 14 bands (`p = (pos * 14) / NOPE_DIM` over the sorted
  order, or quantile as today but over 448 not 512); emit rope dims to bands
  14,15 by natural position. `dim_order` and 4-bit `pal_map` reflect this.
- `quantize_sealed_latent_impl` (`compress.rs:572`): call `run_latent_assign_palettes`
  to get `(pal_map, dim_order)`; pass `dim_order` (not `ident_order`) into
  `run_latent_band_select`/`convert`; store `pal_map` into `SealedChunk.k_pal`;
  call `set_mapped_sealed()`. Keep the rope bands (14,15) on the rope threshold,
  the 14 nope bands on the (now per-magnitude) nope thresholds.
- Decode/prefill launch `MAPPED=true` when `has_mapped_sealed()` (already wired
  via `kernel_attention.rs:206,306`, `wave.rs:529,629,688`). This makes
  `pal_rank_of` / `load_mapped_key_dims` load-bearing for the first time —
  exercise them hard (they were cold).
- CPU mirror `mapped_chunk_roundtrip` (`paged.rs:1499`): **restore** the
  map-honoring gather/scatter (the identity-only collapse from the NPAL=8 stabilize
  pass is removed) — now correct because `pal_of` is 4-bit and names 16.

### Phase 3 exit gate
Mapped gates green (`mirror_bit_exact_mapped_mixed_formats`,
`palmap_output_invariant_across_maps`, assign gate). Seal gates green
(`arena_backed_matches_synthetic`, `latent_compressed_arena_roundtrip_decode`,
`latent_c9_seal_decode_robustness`). C3 StoryRewrite with **regroup on**; measure
compression ratio vs identity-16 and vs the 8-band 2.12×/2.7× baselines.

---

## Phase 4 — persistence

- `candle-conversation/src/persistence/transfer.rs` `reassemble_block`
  (:667-690): 4-bit unpack (`(byte >> ((d&1)*4)) & 0xF`), `pal_bytes=head_dim/2`,
  loop `0..n_palette` (16), `expect_bands = n_kv_head*n_palette()` (:655).
- On-disk format is length-prefixed (`record.rs:404-559`) so it is structurally
  agnostic; **old 8-band/2-bit images are incompatible** — bump the persistence
  version and reject/skip stale latent chunks (this is a research codebase; no
  back-compat shim — fresh `.substrate`). Document in the redo-log version note.

---

## Phase 5 — validate + measure

- 26 `deepseek4::paged` gates green (run isolated if any illegal-address risk).
- `wave_paris` ×3 → "Paris"; StoryRewrite (uncompressed) 100%; C3 StoryRewrite
  (regroup) 100%.
- Report: compression ratio (target > 2.7× — regroup + finer bands should beat
  the 8-band identity 2.7×); decode/prefill t/s + occupancy (smem grew;
  confirm blocks/SM).

## Phase 6 — docs

- Update `docs/deepseek_batched_paged_attention_plan.md`: "4 palette sub-bands
  (128-d)" → "16 sub-bands (32-d)"; "4 palettes × 4 k-steps" → "16 × 1 k-step";
  the `(palette<<6)|rank` note (SUB≤64 now satisfied); pal_map 2-bit → 4-bit.
- Update `docs/deepseek_v4_flash.md` and the memory notes
  (`deepseek-asymmetric-corpus-compression`, a new npal16-palmap note).

---

## Highest-risk items (must-not-miss)
1. **One-sided host/device map change is silent** — record byte-size is unchanged,
   so no assert catches a mismatch. `meta_pool.rs` serializer, `slot_types.cuh`
   accessors, `paged.rs` writer, and the device decoders must flip to 4-bit *in the
   same change* and be proven byte-parity by the mirror before geometry moves.
2. **Warp-budget** (decode band-stride loop; prefill warp=band + row-tile loop) —
   silent corruption of half the bands/heads if missed.
3. **Corpus DPL=1 RoPE OOB** — both rope-quant kernels.
4. **Regroup ⇄ rope reconciliation** — sort nope-only, pin rope to bands 14,15;
   without this the rope dims get scrambled and the asymmetric protection breaks
   (the original reason regroup was dropped).
5. **`pal_map_identity_warp` "one word per lane"** — 64 words > 32 lanes at 4-bit;
   a design change (2 words/lane), not a constant tweak.
