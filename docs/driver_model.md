# The driver model: WDDM, TCC, Linux

A CUDA device on Windows runs under one of two NVIDIA driver models, and the
engine's memory arithmetic depends on which:

| | WDDM | TCC | Linux |
|---|---|---|---|
| Who manages VRAM | Windows' video memory manager | the NVIDIA driver | the NVIDIA driver |
| VRAM past capacity | **demoted** to system RAM, silently | refused: `CUDA_ERROR_OUT_OF_MEMORY` | refused |
| `cuMemGetInfo` free | polluted by what the OS manages | honest | honest |
| Residency budget | DXGI per-process `Budget − CurrentUsage` | none | none |
| Commit charged for VRAM | every allocation is backed in system commit | none | none |
| GPU-addressable host budget | DXGI NON_LOCAL segment | none: the driver's refusal | none |
| Launch submission | batched; a `cuStreamQuery` forces it | immediate | immediate |
| Display watchdog (TDR, 2 s) | yes | no | no (headless) |
| Visible to DXGI | yes | no | — |

TCC needs a card that supports it — the RTX PRO 5000 does, GeForce cards do not
— and a card under TCC drives no display. BlackWorld runs its console from the
CPU's integrated GPU and is reached over RDP, so its card can run TCC.

## 1. Measured, not assumed

`candle::vram::DriverModel` is read once per device from
`CU_DEVICE_ATTRIBUTE_TCC_DRIVER` (`vram/driver_model.rs`) and cached by device
ordinal. Windows with the attribute 0 is WDDM; with it 1, TCC; any other OS is
Linux. Every branch below asks it; nothing infers WDDM from `cfg(windows)`.

The startup line `vram: driver model` names it, and `/v1/memory` reports it
with the probe that produced the governor's readings
(`report.vram.governor.driver_model`, `.probe`).

## 2. The VRAM probe follows the model

`VramGovernor::pick_probe`:

- **WDDM** — the DXGI probe, required. DXGI is the residency signal the
  balloon's live-headroom stop and the wobble margin are built on
  (`balloon.rs`); a WDDM device whose DXGI adapter cannot be opened gets no
  governor, and `ensure_vram_governor` logs why, instead of a governor sized
  from a `cuMemGetInfo` reading that would claim into demotion.
- **TCC, Linux** — the `cuMemGetInfo` probe. Its `free` is honest there, the
  balloon's refusals are real, and the wobble margin is the configured reserve.

`DxgiProbe::for_cuda_device` itself refuses a device that is not WDDM: DXGI
does not enumerate a TCC adapter, and "no adapter matched the LUID" would hide
why.

## 3. The GPU-addressable host budget follows the model

`gpu_addressable_budget` answers `Ok(None)` under TCC and Linux — there is no
NON_LOCAL budget, and only the driver's refusal and the warm tier's
`DRIVER_LOCK_MARGIN` bound a lock — and on WDDM reads DXGI, an error there
being an error. It no longer turns every failure into "no budget", which on a
WDDM machine would have pinned past what the device can address.

## 4. Overcommit is an error under TCC

The engine's long-lived device memory is inside the reservation — KV arenas,
span tenants, held ground — and a granule there is mapped, touched and counted
before it is used, so it cannot overcommit under either model. What lives
outside it does: the CUDA pool's activations and transients, a held-ground take
that found no room (`held_ground::refuse`, which hands the buffer to the pool),
the graph library's budget, cuBLAS workspaces and the driver's own allocations.

Under WDDM an overrun there is a demotion — slow, and silent. Under TCC it is
`CUDA_ERROR_OUT_OF_MEMORY`. The room outside the reservation is the same under
both: `scratch_margin` (512 MiB, the pool's cushion) plus `capacity_reserve`
(256 MiB, never claimed), as it already is on Linux. BlackWorld's pool reserved
436 MB at peak on build 4779 (2026-11-09), inside the cushion.

**The verification belongs to the switch** (§7): with the card on TCC, a sweep
under `--features forbidden_allocations` names what allocates outside the
reservation at peak, and the cushion is re-derived from that figure, per model,
in `budget.rs`.

## 5. The reservation's refusal

`Reservation::map_granule` treats `CUDA_ERROR_OUT_OF_MEMORY` from `cuMemCreate`
or `cuMemMap` as the capacity, and a failed zero-fill likewise. Under WDDM the
create succeeds past capacity and only the first write fails; under TCC the
create is expected to refuse. Both are already handled; the switch confirms
which one TCC gives.

## 6. Submission nudges

WDDM batches submissions in the driver until something flushes them, and the
graph hub ends a segment with a `cuStreamQuery` for that purpose alone
(`graph/hub.rs`, `query_submits`) — the host threads that wait on a summary
word would otherwise wait for WDDM to submit the bucketize writing it. Under
TCC and Linux a launch is submitted as it is issued, so `query_submits` returns
without the call.

The limits sized under WDDM's 2 s display watchdog stay as they are. The
expert workers' spin limit (`moe_live.cuh`, `dispatch.rs`) is a deadlock
backstop under any driver, and the banded launches in `quantized/cuda.rs` cost
nothing under TCC.

## 7. Switching a machine

1. `nvidia-smi -i 0 -dm TCC` from an elevated shell; it applies at the next
   reboot (`nvidia-smi --query-gpu=driver_model.pending`).
2. Reboot. keep's boot task starts `web` and `motherd`; the member rejoins and
   catches up.
3. Confirm the startup line `vram: driver model … TCC`, and in `/v1/memory`:
   `driver_model: tcc`, `probe: cuda`, and the ~32 MB private allocations that
   back VRAM under WDDM gone from `process.ram.private_buckets`.
4. Run the sweep under `forbidden_allocations` (§4) and record decode, prefill
   and the expert hit rate against the WDDM baseline in `docs/performance.md`,
   naming the driver model.
5. Back: `nvidia-smi -i 0 -dm WDDM` and reboot.

WhiteWorld's RTX 3090 is a GeForce card, which the driver does not let run TCC;
it stays WDDM, so WDDM stays a fully supported model.
