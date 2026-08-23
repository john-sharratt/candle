# Qwen3.6-35B-A3B performance plan — closing the gap to Qwen3-30B-A3B

Status: **open**. Quality calibration is done (`docs/qwen35_qwen38_models.md`
§7.22); this document is the ordered work list for the throughput gap, with
the measured evidence each item rests on. Reference numbers, same card
(RTX 4090 Mobile 16 GB), 2026-08-23:

| | prefill (bulk) | decode (aggregate) |
|---|---|---|
| Qwen3-30B-A3B, Q8_0×20 | 2,710 t/s | 189.7 t/s |
| Qwen3-30B-A3B, Q4_0×20 | 2,729 t/s | 200.4 t/s |
| Qwen3.6-35B, C8×20 | 605 t/s | 22–90 t/s (unstable run-to-run) |
| Qwen3.6-35B, BF16×20 | 189 t/s (first-wide cost) | 90.7 t/s |

The structural budget says the two models should be close: both are ~3B
active, per-wave expert bytes are within ~20% of each other, and the hybrid
attends on only ¼ of its layers. The gap is therefore engine-side, and it
splits into the items below, ordered by expected leverage.

## T1 — Decode at width falls off the fused DeltaNet path (largest, decode)

**Evidence.** Aggregate decode at ×20 swings between 22 and 90 t/s across
runs; per-session decode at width ≥5 sits far below the ×4 rate (C9×5:
~2–3 t/s/session vs ~16 at ×4). The `--features profile` span table
(`/tmp/gate_36_prof.log`, 2026-08-23) puts the cost in the mixer: `dn:mix`
at C9×5 is 2,290 ms/300 calls ≈ 7.6 ms/call vs 0.09 ms at ×1 — **~89× per
call** — with `dn:proj`/`dn:out_proj` inflated alongside, and the
attention-side `prefill:qkv_proj`/`out_proj` showing the same ~10–60×
per-call inflation in the same configs.

**Suspects (in order).**
1. The fused CUDA path's dispatch guard in `delta_net/mix.rs` (~line 915):
   it requires `qkv`/`alpha_lin`/`beta_lin`/`z`/`dt_bias`/`a`/`norm` all
   F32, `d == DELTA_NET_PREFILL_DIM`, and every span state on CUDA. Any
   miss silently drops the whole wave to the tensor-op fallback — the exact
   shape of a cliff that appears at some widths/modes and not others.
2. The per-sequence `delta_net_prefill_scan` loop (glue / `len > 1` spans
   run one launch per sequence, serialized).
3. `build_layer_table` construction conditions (decode spans only batch
   through the pointer table).

**Attack.** Instrument *which path ran* (a per-wave counter, not timings —
the profile feature's sync overhead distorts width rows ~2×). Then make the
fused path cover the shapes the C-mode waves actually produce, and batch
the per-sequence scan loop. Success: per-session decode flat from ×1 to
×20; aggregate decode at ×20 ≥ 200 t/s.

## T2 — Wide prefill (605 vs 2,710 t/s)

**Evidence.** Same profile table: at C9×5 prefill, `dn:proj` 6,080 ms,
`dn:mix` 4,418 ms, `dn:out_proj` 1,680 ms (×210 calls) vs 943/432/26 ms at
BF16×4 (×180 calls) — per-call inflation ~6–60× with call counts nearly
flat. This is the same fallback signature as T1, on the prefill side; fix
T1's dispatch and re-measure before assuming a second cause. Whatever
remains after T1 lands is genuine wide-prefill work (expert cycling per
layer — see T6).

## T3 — First-wide-config pays ~30 s of establishment

**Evidence.** The first ×20 row of a process prefills at 150–210 t/s;
every later ×20 row at 590–605 (measured three ways; the BF16×20 float
control row currently absorbs it in the gates). Something in first-wide
wave assembly — arena establishment for the width, warm elevation, expert
zone renegotiation churn — is a one-time ~30 s cost.

**Attack.** Time-slice the first wide forward (the spans exist), name the
cost, then either do it at model load (the geometry is known) or make it
proportional to growth rather than all-at-once.

## T4 — Warm-tier coverage: 57% of experts vs the 30B's 85%

**Evidence.** The 3.6's pack is 19.8 GB against ~11.3 GB pinnable
(`available − 3 GiB` headroom is the binding ceiling on the 32 GB box);
the 30B's 17.8 GB pack fits 85%. C8×20 still issues ~20K cold NVMe loads.

**Attack.** The warm fill is a stratified *uniform* draw
(`expert_lre/handle.rs`), but MoE routing is Zipf-distributed — fill warm
by measured routing frequency (the routing-capture facility already tags
per-config records; an online counter also works) so ~11 GB covers most of
the *traffic* rather than 57% of the population. The tier stays static;
only the draw changes. Re-examine `WARM_TIER_HEADROOM` on this box only
after that (the 3 GiB → 1 GiB experiment on the 30B was flat).

## T5 — VRAM expert-zone growth never reaches its limit

**Evidence.** Post-governor the zone opens at 3,193 slots with limit 4,998,
but hit rate plateaus at 33–40% (30B: 44–54%) and evictions on C8×20 are
~115K — churn suggests the boundary isn't growing to the limit during a
run, or growth is immediately eaten by KV claims.

**Attack.** Log the boundary position per pass; verify
`reclaim_spare_ground` cadence actually moves it between forwards under
gate-shaped load; consider per-layer pinning of the highest-frequency
experts (the pinned-set arithmetic exists — `affordable_pinned_layers`).

## T6 — Cold reads still synchronous where prediction already knows better

**Evidence.** Prediction precision at width is 93%, but prefetch volume is
~9K against ~115K DMA loads on C8×20 — the predictor is right and barely
used. Cold misses are synchronous NVMe reads on the pipeline thread.

**Attack.** Raise prefetch depth/queue so predicted next-layer experts are
in flight during the current layer's GEMM; overlap cold reads behind the
fence. Fence stalls (~1,400/config) are the observable to drive down.

## T7 — Single-session decode: 13–17 t/s vs the 30B's 25–26

**Evidence.** BF16×1 decode 12.9–16.9 vs 25.0–26.6. Both are
expert-bandwidth-bound at width 1; the hybrid loads 320 experts/token
(40 layers × top-8) at 1.93 MB vs the 30B's 384 × 2.9 MB — *fewer bytes*,
yet slower. Per-load fixed overhead (launch + fence per smaller load) is
the likely term; measure a per-step layer profile (a MoE-capable
`profile_layers` — the dense helper exists in `quantized_qwen35.rs`) before
attacking. Lower priority: T1/T4/T6 likely move this too.

## T8 — Price the wave-plan norms by session encoding (batch with a re-derivation)

**Evidence.** `WaveBuffer::AttnNorm`/`FfnNorm` price dense for every session
even though an int8 session's norm output is q8a128 — a deliberate,
documented over-bound ("the two encodings are alternatives and the plan has
to bound both"), worth 1,792 B/row ≈ 3 points of the attention union margin,
which directly narrows admissible wave width on the production int8 mode.

**Attack.** Thread the encoding into `ModelGeometry` (an
`int8_projections` flag = `is_int8 && hidden % 128 == 0 && hidden ≤ 8192`,
set by each model's `wave_geometry` — the same pattern as
`projection_accum_roundtrip`/`gated_qkv`), price q8 when set, re-pin the
margin tests.

**Sequencing constraint — do NOT land this alone:** changing transient
pricing changes admitted wave widths, which changes accumulation order and
moves every marginal KV-factor calibration session (measured when the
VRAM-governor fix did exactly this). Land it in the same change set as the
next factor re-derivation, never between derivations.

## Tooling debt discovered while measuring

- The pipeline-profile printer caps at 6 config columns and silently drops
  the rest (the C2–C8 columns vanished from the §T1 table's run).
- Profile-feature builds suppress decode-at-width absolutes ~2× (per-span
  syncs); only within-build comparisons are valid.
- Path-taken counters (T1) beat span timings for dispatch-cliff hunting.
