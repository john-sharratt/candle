# Gates — RTX PRO 5000 Blackwell 72 GB, 2026-10-05 (zero-allocation forward)

Two width gates, one `cargo` process per model with the card to itself. The build is
`86c91146` on `main`, which is `16c3bff8` ("Zero device allocations in the forward pass;
join turns whose rows do not pay") merged with `origin/main`. Since the live-dispatch sweep
(`sweep_rtx_pro_5000_72gb_2026-10-05.md`), this build carries the following.

**Allocation-free forward pass**
- Layout tables are carved from a per-stream info ring.
- Uploads are staged through segmented and synced uploads.
- The persistence thread's migration plans go through `with_synced_upload`.
- Window residuals come from a session-owned pool.
- Caller↔internal token reordering copies contiguous runs.
- `ForwardResidual` is on the forward span for every single-stream forward.
- The draft head pass (`Chain::HeadPass`) is priced and rooted on the span.

**Admission**
- A turn the rate model refuses is offered again as a join instead of latching the wave.
- A rate width clamps the pass budget until a pass's rows all pay.

**Qwen3.8-Flash-Next**
- The prefill width cap scales with the card's placeable tier.
- The parked verify stash is released with the sequence.
- `QWEN4EXP_KV_FACTORS` V moved from 2.8 to 2.6. The wider prefill slabs and the fused PLE kernel's FMA reorder flipped one first-token near-tie at C10×8. That is edge sensitivity, not a precision loss (see the row's comment in `params.rs`).

These are single runs, so a gap under about 5% is noise. Every row validated, every session.

## Qwen3.6-35B-A3B — `quantized_qwen36_moe::…::test_parallel_batched_forwarding_36_35b`

All 18 configurations passed. Wall clock 42.9 s.

| mode | ctx | prefill t/s | decode t/s | compression | pass |
|---|---:|---:|---:|---:|---|
| BF16 | 1 | 6,930.8 | 129.7 | — | 1/1 |
| BF16 | 4 | 9,192.7 | 522.8 | — | 4/4 |
| Q8_0 | 1 | 6,938.8 | 134.5 | 1.88× | 1/1 |
| C0 | 1 | 6,944.5 | 140.4 | 2.21× | 1/1 |
| C1 | 1 | 6,929.2 | 138.3 | 2.68× | 1/1 |
| C2 | 1 | 6,933.4 | 137.1 | 3.13× | 1/1 |
| C3 | 1 | 6,894.7 | 138.0 | 3.35× | 1/1 |
| C4 | 1 | 6,932.0 | 137.0 | 3.58× | 1/1 |
| C5 | 1 | 6,937.8 | 136.8 | 3.91× | 1/1 |
| C5 | 8 | 9,622.0 | 896.6 | 3.91× | 8/8 |
| C6 | 1 | 6,955.6 | 138.9 | 4.40× | 1/1 |
| C7 | 1 | 6,937.2 | 138.1 | 4.63× | 1/1 |
| C8 | 5 | 9,675.9 | 602.5 | 5.24× | 5/5 |
| C9 | 2 | 8,765.7 | 272.6 | 5.84× | 2/2 |
| C10 | 8 | 9,598.3 | 875.9 | 6.45× | 8/8 |
| C10 | 16 | 9,733.5 | 1,119.6 | 6.43× | 16/16 |
| C10 | 32 | 9,647.0 | 1,764.4 | 6.43× | 32/32 |
| C10 | 64 | 9,646.0 | 2,258.4 | 6.44× | 64/64 |

Every expert is resident: 21,976 MiB hot, 100% hit rate, no promotions.

## Qwen3.8-Flash-Next — `quantized_qwen38_moe::…::test_parallel_batched_forwarding`

All 11 configurations passed. Wall clock 65.0 s.

| mode | ctx | prefill t/s | decode t/s | compression | pass |
|---|---:|---:|---:|---:|---|
| BF16 | 1 (cold) | 2,879.0 | 130.7 | — | 1/1 |
| BF16 | 4 | 4,130.9 | 484.0 | — | 4/4 |
| BF16 | 8 | 4,215.5 | 738.2 | — | 8/8 |
| BF16 | 16 | 4,150.1 | 792.8 | — | 16/16 |
| BF16 | 1 (warm) | 3,282.0 | **147.5** | — | 1/1 |
| C0 | 2 | 3,917.5 | 263.7 | 2.21× | 2/2 |
| C5 | 2 | 3,919.3 | 264.9 | 4.15× | 2/2 |
| C5 | 8 | 4,194.9 | 697.5 | 4.15× | 8/8 |
| C8 | 2 | 3,914.9 | 264.3 | 5.52× | 2/2 |
| C10 | 2 | 3,913.5 | 263.3 | 7.14× | 2/2 |
| C10 | 8 | 4,184.9 | 681.5 | 7.13× | 8/8 |

Expert hit rate was 98.0–100%. The 24,064 experts in the warm tier are all pinned, and there were no cold misses.

## Against earlier runs

Decode and prefill in t/s. "Live-dispatch" is `sweep_rtx_pro_5000_72gb_2026-10-05.md`. "Kernel-size" is `sweep_rtx_pro_5000_72gb_2026-10-04_kernel_size.md`.

| model | row | live-dispatch | now | change | kernel-size |
|---|---|---:|---:|---:|---:|
| Qwen3.6-35B | ×1 decode | 129.8 | 129.7 | −0.1% | 140.2 |
| Qwen3.6-35B | ×1 prefill | 6,881.7 | 6,930.8 | +0.7% | — |
| Qwen3.6-35B | C10×64 decode | 2,182.6 | 2,258.4 | **+3.5%** | 2,341.3 |
| Qwen3.6-35B | C10×64 prefill | 9,587.0 | 9,646.0 | +0.6% | — |
| Qwen3.6-35B | C10 compression | 6.44× | 6.44× | — | — |
| Flash-Next | ×1 warm decode | 143.1 | **147.5** | +3.1% | 116.7 |
| Flash-Next | ×1 warm prefill | 3,274.3 | 3,282.0 | +0.2% | 2,473.7 |
| Flash-Next | ×16 decode | 807.9 | 792.8 | −1.9% | 685.9 |
| Flash-Next | ×16 prefill | 3,841.4 | 4,150.1 | **+8.0%** | 3,273.9 |
| Flash-Next | C10×8 compression | 7.33× | 7.13× | −2.7% | — |

- **Flash-Next** is level or faster on every row. Single-stream warm decode, at 147.5 t/s, is the best recorded on this card. Its wider prefill (+8% at ×16) comes from the card-scaled prefill cap. The one loss is C10 compression (7.33× → 7.13×), which is the V threshold move above.
- **Qwen3.6-35B** has recovered about half of the live-dispatch regression at ×64 (2,182.6 → 2,258.4, now 3.5% under the kernel-size sweep's 2,341.3). Its single-stream decode is unchanged at 130 t/s, against 140 before live dispatch. That regression predates this build and is still not attributed.
