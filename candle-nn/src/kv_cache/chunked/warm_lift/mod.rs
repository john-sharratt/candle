//! The warm lift: RAM → VRAM for every layer and item of an elevate, batched
//! and sync-free (`docs/warm_lift.md`).
//!
//! One plan over every layer's bands ([`layer`]), cut into ring-slot batches
//! ([`plan`]). For each batch the host copiers fill a slot of the pinned ring
//! from the pageable warm arenas ([`copiers`], [`ring`]) and one scatter
//! launch on the device's stream moves it, read in place, into the
//! destination slots. The launch for batch `n` runs while the copiers fill
//! batch `n + 1`.
//!
//! The lift is enqueued on the device's own stream, so what it needs ordered
//! is ordered by the stream: a destination slot claimed here is written only
//! after every queued read of its previous tenant, and every later launch
//! reads the lifted bytes after they land. The host waits only to refill a
//! ring slot whose last scatter has not run yet.

mod copiers;
mod layer;
mod plan;
mod ring;
mod split;

pub use ring::WarmLiftRing;
pub use split::LiftSplit;

use std::time::Instant;

use candle::cuda_backend::kernels::simple::warm_lift::{run_warm_lift_scatter, WarmLiftEntry};
use candle::{Device, Result};

/// Blocks of the scatter resident per SM — `SCATTER_BLOCKS_PER_SM` in
/// `candle-kernels/src/simple/scatter_copy.cuh`, which the kernel's launch
/// bounds hold its registers to: six blocks of 256 threads, 48 warps, fill an
/// SM of 1,536 threads.
const SCATTER_BLOCKS_PER_SM: usize = 6;

use self::copiers::fill_slot;
use self::plan::{batches, LiftBand};
use crate::kv_cache::chunked::types::SealedSequence;
use crate::kv_cache::chunked::ChunkedKvBacking;

/// A lift's result: per layer, the items' sequences at their new GPU
/// placement, in the order given, and where the host's time went.
pub struct WarmLifted {
    pub layers: Vec<Vec<SealedSequence>>,
    pub split: LiftSplit,
}

/// Lift `layers[l]` — every item's warm sequence at layer `l` — into VRAM
/// through `backings[l]`.
pub fn lift_warm_to_gpu(
    backings: &[ChunkedKvBacking],
    ring: &mut WarmLiftRing,
    device: &Device,
    layers: &[Vec<&SealedSequence>],
) -> Result<WarmLifted> {
    let Device::Cuda(cuda) = device else {
        candle::bail!("warm lift: requires a CUDA device");
    };
    if layers.len() > backings.len() {
        candle::bail!(
            "warm lift: {} layers for {} backings",
            layers.len(),
            backings.len()
        );
    }
    // **The lift meets the device eagerly.** On a thread recording a wave,
    // launches are captured into the wave's segment rather than issued, and a
    // query of an event invalidates the capture. A lift's launches read ring
    // slots the host refills as soon as their events say the scatter ran, and
    // `acquire` asks exactly that — so the lift suspends the capture until it
    // has issued everything: the recorded segment goes first, and the lift's
    // launches follow it, in order, on the one stream. Nothing waits for the
    // device here; only the recording is set aside.
    let _paused = cuda.pause_capture()?;
    let mut split = LiftSplit::default();
    let t = Instant::now();
    let lifts = layers
        .iter()
        .enumerate()
        .map(|(l, seqs)| backings[l].plan_warm_lift(l, seqs))
        .collect::<Result<Vec<_>>>()?;
    let bands: Vec<LiftBand> = lifts.iter().flat_map(|l| l.bands.iter().copied()).collect();
    let stream = cuda.cuda_stream();
    // A few blocks per SM: a warp a run, and enough of them resident to keep
    // the reads' round trips covered (`scatter_copy.cuh`).
    let blocks = (cuda.multiprocessor_count()? * SCATTER_BLOCKS_PER_SM) as i32;
    let cut = batches(&bands, ring.slot_bytes(), ring.max_entries())?;
    split.plan_us = micros(t);
    for batch in cut {
        let t = Instant::now();
        let mut slot = ring.acquire()?;
        split.wait_us += micros(t);
        let t = Instant::now();
        fill_slot(backings, &bands, &batch, slot.payload)?;
        split.fill_us += micros(t);
        let t = Instant::now();
        let entries: Vec<WarmLiftEntry> = bands[batch.bands.clone()]
            .iter()
            .zip(&batch.offsets)
            .map(|(band, &off)| WarmLiftEntry {
                src_off: off as u64,
                dst: band.dst,
                bytes: band.bytes as u64,
            })
            .collect();
        let table = slot.write_table(&entries)?;
        // Every destination this launch writes, checked against memory
        // declared immutable after load: the plan is resolved from arena
        // bases, and a wrong one writes wherever it lands.
        #[cfg(feature = "tensor-assert")]
        for e in &entries {
            candle::readonly_regions::forbid_write(
                "warm_lift_scatter destination",
                e.dst,
                e.bytes as usize,
            );
        }
        // SAFETY: `slot.payload` and `table` are the slot's mapped pinned
        // memory, held by the ring and not refilled until the event recorded
        // below has completed; every destination is a slot claimed for this
        // lift and resolved from its arena's base above.
        unsafe {
            candle::set_kernel_breadcrumb("run_warm_lift_scatter", file!(), line!());
            run_warm_lift_scatter(
                slot.payload.as_ptr(),
                table,
                entries.len() as i32,
                blocks,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
        slot.scattered(&stream)?;
        split.launch_us += micros(t);
        split.batches += 1;
        split.entries += entries.len() as u64;
        split.bytes += entries.iter().map(|e| e.bytes).sum::<u64>();
    }
    let t = Instant::now();
    let layers = layers
        .iter()
        .zip(&lifts)
        .enumerate()
        .map(|(l, (seqs, lift))| backings[l].finish_warm_lift(seqs, lift))
        .collect::<Result<Vec<_>>>()?;
    split.finish_us = micros(t);
    Ok(WarmLifted { layers, split })
}

fn micros(since: Instant) -> u64 {
    since.elapsed().as_micros() as u64
}

/// One layer's lift through a ring of its own — what a test of one backing
/// asks for.
#[cfg(test)]
pub(crate) fn lift_one(
    backing: &ChunkedKvBacking,
    device: &Device,
    sequences: &[&SealedSequence],
) -> Result<Vec<SealedSequence>> {
    let mut ring = WarmLiftRing::new()?;
    let mut lifted = lift_warm_to_gpu(
        std::slice::from_ref(backing),
        &mut ring,
        device,
        &[sequences.to_vec()],
    )?;
    Ok(lifted.layers.remove(0))
}
