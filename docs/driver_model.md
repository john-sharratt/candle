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
| Virtual memory management (`cuMemAddressReserve`/`cuMemCreate`/`cuMemMap`) | yes | **no** | yes |
| Stream-ordered memory pools (`cuMemAllocAsync`) | yes | **no** | yes |

The last two rows were measured on the RTX PRO 5000 under TCC, driver 596.59
(2026-11-09): `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 0`,
`CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 0`. The span was built on the
first, which §4 replaces; cudarc falls back from the second on its own.

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

## 4. One allocation, and the driver's ground beside it

Under every model the span is **one `cuMemAlloc`**. Nothing needs virtual
memory management, so the same code runs under WDDM, TCC and Linux.

- **The span** (`kv_cache/chunked/reservation.rs`) is allocated once at
  `target − load_headroom`, capped at the free memory the driver reports less
  256 MiB for the driver's own allocations, stepping down by a thirty-second on
  a refusal. A guard granule sits in front of it, so no tenant ever holds the
  allocation's base. It is zero-filled as it is claimed. Under WDDM an
  allocation past the residency budget is granted and demoted rather than
  refused, so the cap, and the governor sizing the target from DXGI's budget,
  keeps it resident; the eviction unit there is the whole span.
- **It never grows.** The load headroom (`peak_load_pool_bytes`: the largest
  source tensor and the repack's two bands) stays outside the span for the life
  of the process. It is the driver's ground: the load's scratch, the layer
  pack's build after `close_load`, and every allocation not yet carried by an
  arena of the span land there.
- **Everything else cudarc allocates is the driver's**, as upstream: a
  stream-ordered pool allocation where the card has memory pools (WDDM, Linux),
  a synchronous `cuMemAlloc` — and a stream synchronise plus `cuMemFree` on its
  drop — where it has none (TCC). The held ground (`held_ground.rs`) serves
  `Held` buffers, owned copies and asides (below); one it cannot hold goes to
  the driver, reported.

### Moving every allocation onto the span

The target is that nothing but the span reaches the driver — and, before it,
the VRAM governor's balloon (`vram/balloon.rs`), which measures what the driver
will grant by asking it directly (`cuMemAlloc`, a memset, `cuMemFree`), since
that measurement is what sizes the span: the dense weights
bumped from the span's start and kept for good, the wave's transients in the
transient tier, long-lived buffers in a tenant's slot arenas, where compaction
can move them. The `driver-alloc-panic` feature is the instrument. cudarc is
vendored (`third_party/cudarc`) with one hook, an allocation watch that
`CudaStream::alloc` (and `alloc_zeros`, `memcpy_stod`, `clone_htod`, `null`)
reports to before the driver allocates; with the feature on, candle's watch
(`candle-core/src/cuda_backend/driver_alloc.rs`) panics there, naming the
`#[track_caller]` site of the entry point that asked, the size, whether the span
existed yet, and the stack. Off — the default — the watch is not installed and
`at_site` compiles to its closure.

**A model load is exempt.** A load runs inside a `LoadWindow`
(`cuda_backend/load_window.rs`), opened by `dense_span::open_for_load` and by
a batched wrapper's one-time construction (`dense_span::open_window`), and the
watch lets a device's allocations through while any window on it stands. A
load's driver allocations are the source tensors and the repack's bands — what
the load headroom exists for — and they are freed when it ends. The feature
polices the runtime: every allocation after the model is loaded.

The runtime's homes, each the cheapest one that is not the driver:

- **Inside a forward** — the forward's tier. An op's output inherits its
  operand's wave arena (`wave_provenance`).
- **An owned copy** — a value taken off the tier to outlive its forward
  (`Tensor::to_owned_tensor`, `Tensor::cat_owned`, so `logits_owned`): one
  device-to-device copy into a `Held` range of held ground, anchored to the
  storage and given back when it drops (`CudaDevice::alloc_owned`). It never
  crosses the bus.
- **A kernel's output that follows its input** — the batched sampler's picks:
  into the input's wave arena while it is open, an owned held range otherwise
  (`CudaDevice::alloc_following`).
- **Work between forwards** — an *aside* (`kv_cache/chunked/aside.rs`): one
  held range taken for the closure, every op on a ticketless operand inside it
  a bump of that range (`with_ambient` seeds the aside's ticket), the range
  given back as one when the closure is done — or, if a value carved there
  escapes it, when the last such value drops. A harness's picks and checks over
  owned logits run in one.
- **Host data** — prompt ids are built on the host, as the scheduler hands them
  over; the forward uploads a wave's ids once, onto its span.

A run with the feature stops at the next site to move. Once none is left, the
load headroom is no longer the driver's for anything and the span takes it.

## 5. The reservation's refusal

`Reservation::reserve` treats `CUDA_ERROR_OUT_OF_MEMORY` from `cuMemAlloc` as a
step down, and the size that succeeds is the span. Under TCC and Linux the
driver refuses honestly; under WDDM the cap on free memory is what stops it.

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
4. Run the Flash-Next gate
   (`quantized_qwen38_moe::tests::test_parallel_batched_forwarding`) with the
   daemon stopped, and record decode, prefill and the expert hit rate against
   the WDDM baseline in `docs/performance.md`, naming the driver model.
5. Back: `nvidia-smi -i 0 -dm WDDM` and reboot.

WhiteWorld's RTX 3090 is a GeForce card, which the driver does not let run TCC;
it stays WDDM, so WDDM stays a fully supported model.
