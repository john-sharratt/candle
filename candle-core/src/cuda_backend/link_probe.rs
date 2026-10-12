//! The host→device link rate, measured once per device — before the model
//! loads, when the card is empty and the bus idle.
//!
//! The scheduler's wave planner seeds its copy rate from this figure, so it is
//! this machine's own rather than a datasheet's: the 4090 Mobile, the 3090 box
//! behind PCIe 3.0 and the Blackwell workstation each read their own bus.
//!
//! # Why before the load
//!
//! The probe used to run at scheduler start, which is the worst moment for both
//! things it touches:
//!
//! - **The bus is busy.** The expert cache is still staging its resident set
//!   over the same link, so a reading measured the contention it landed in: the
//!   4090 Mobile read 2.61 GB/s against a ~25 GB/s PCIe 4.0 ×16 link, and the
//!   planner seeded a copy rate that made a prefill forward look ten seconds
//!   long.
//! - **The card is full.** Its 256 MiB landing buffer asked held ground for
//!   room the decode slot-state slabs had already taken, and went to the
//!   driver: 256 MiB outside the reservation at the moment the headroom
//!   matters most.
//!
//! Before the load neither is true. The model builder calls
//! [`host_to_device_bytes_per_s`] ahead of claiming the reservation; every later
//! call — the planner's — reads the cached figure.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::held::Held;
use super::{CudaDevice, WrapErr};
use crate::quantized::pinned_staging::PinnedBuf;
use crate::Result;

/// Bytes each probe run copies. Large enough that launch latency is noise
/// against the transfer; small enough to be a moment at load.
pub const PROBE_BYTES: usize = 256 << 20;

/// Timed runs, keeping the fastest. Contention can only ever make a transfer
/// slower, so the quickest of a few is the closest to the link.
const PROBE_RUNS: usize = 3;

/// The device window the probe's bytes stream through, a window's worth per
/// copy, back to back: the transfer measured is the whole [`PROBE_BYTES`], and
/// the device holds only this much of it — so a probe that does run on a full
/// card asks for a quarter of what one buffer would.
const PROBE_WINDOW_BYTES: usize = 64 << 20;

fn measured() -> &'static Mutex<HashMap<usize, f64>> {
    static MEASURED: OnceLock<Mutex<HashMap<usize, f64>>> = OnceLock::new();
    MEASURED.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `dev`'s pinned host→device copy rate, in bytes a second: measured the first
/// time it is asked for, the cached reading every time after.
pub fn host_to_device_bytes_per_s(dev: &CudaDevice) -> Result<f64> {
    let ordinal = dev.cuda_stream().context().ordinal();
    if let Some(&bps) = measured()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&ordinal)
    {
        return Ok(bps);
    }
    let bps = probe(dev)?;
    measured()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(ordinal, bps);
    Ok(bps)
}

/// [`PROBE_RUNS`] pinned copies of [`PROBE_BYTES`] through one window on the
/// device's stream, each timed to completion; the fastest, in bytes a second.
fn probe(dev: &CudaDevice) -> Result<f64> {
    let stream = dev.cuda_stream();
    let host = PinnedBuf::alloc_owned(PROBE_BYTES)?;
    // SAFETY: written by every copy below before anything could read it.
    let mut window = unsafe { Held::<u8>::alloc_on(&stream, PROBE_WINDOW_BYTES)? };
    let mut stream_through = || -> Result<()> {
        for chunk in host.as_slice().chunks(PROBE_WINDOW_BYTES) {
            let mut landing = window.slice_mut(..chunk.len());
            stream.memcpy_htod(chunk, &mut landing).w()?;
        }
        Ok(())
    };
    // Once untimed, so the first-touch page mapping is not in the measurement.
    stream_through()?;
    stream.synchronize().w()?;
    let mut best = 0.0f64;
    for _ in 0..PROBE_RUNS {
        let start = std::time::Instant::now();
        stream_through()?;
        stream.synchronize().w()?;
        let secs = start.elapsed().as_secs_f64();
        if secs > 0.0 && !secs.is_nan() {
            best = best.max(PROBE_BYTES as f64 / secs);
        }
    }
    if best <= 0.0 {
        crate::bail!("link probe: the transfer took no time");
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendDevice;

    /// The probe reads a plausible PCIe rate on the device, and a second ask is
    /// the cached reading — not a second measurement, which would differ.
    #[test]
    fn the_link_is_measured_once_and_cached() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let first = host_to_device_bytes_per_s(&dev)?;
        // PCIe 3.0 ×4 at the slowest to 5.0 ×16 at the fastest.
        assert!((1e9..80e9).contains(&first), "{first} B/s is no PCIe link");
        assert_eq!(host_to_device_bytes_per_s(&dev)?, first);
        Ok(())
    }
}
