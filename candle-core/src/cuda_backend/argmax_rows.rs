//! Greedy pick over many blocks per row (`candle-kernels/src/sampling/argmax_rows.cu`).
//!
//! The batched sampler's greedy pick runs one block per row; a vocabulary row
//! of a quarter of a million logits then streams through a single SM. This
//! launch spreads every row over many blocks, each folding its slice's best key
//! into the row's slot with one `atomicMax`, and returns exactly the sampler's
//! pick — the same order, ties included (see the kernel's header).
//!
//! The slots and arrival counters are zero between launches — every launch
//! returns them to zero itself — so one pair serves every launch on a stream and
//! a recorded launch replays with no reset node. Launches on one stream run in
//! order, so they never share a pair while one is running; launches on two
//! streams could, so each stream gets its own pair. They are allocated on a
//! stream's first launch, sized for [`MAX_ROWS`] rows.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use candle_kernels::sampling::run_argmax_rows_f32;
use cudarc::driver::{CudaSlice, DevicePtr};

use super::{CudaDevice, DeviceId};
use crate::Result;

/// The most rows one launch picks over: the grid's y-extent.
pub const MAX_ROWS: usize = 65_535;

/// One stream's slots and arrival counters.
struct Scratch {
    slots: CudaSlice<u64>,
    arrived: CudaSlice<u32>,
}

/// Keyed by device and by the stream's handle: a pair is only ever reused by
/// launches that run one after another.
fn scratch() -> &'static Mutex<HashMap<(DeviceId, usize), Scratch>> {
    static SCRATCH: OnceLock<Mutex<HashMap<(DeviceId, usize), Scratch>>> = OnceLock::new();
    SCRATCH.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Write the sampler's greedy pick of each of `rows` rows of `logits` — F32,
/// `row_stride` elements apart, the first `live` of each eligible — to `out`
/// (`rows` `u32`s), on the device's launch stream.
pub fn argmax_rows_f32(
    device: &CudaDevice,
    logits: u64,
    rows: usize,
    row_stride: usize,
    live: usize,
    out: u64,
) -> Result<()> {
    if rows > MAX_ROWS {
        crate::bail!("argmax_rows: {rows} rows, past the grid's {MAX_ROWS}");
    }
    let stream = device.cuda_stream();
    let mut map = scratch().lock().unwrap_or_else(|e| e.into_inner());
    let s = match map.entry((device.id(), stream.cu_stream() as usize)) {
        Entry::Occupied(e) => e.into_mut(),
        // Zeroed once: every launch leaves the slots and counters it used at zero.
        Entry::Vacant(e) => e.insert(Scratch {
            slots: device.alloc_zeros::<u64>(MAX_ROWS)?,
            arrived: device.alloc_zeros::<u32>(MAX_ROWS)?,
        }),
    };
    let (slots, _g1) = s.slots.device_ptr(&stream);
    let (arrived, _g2) = s.arrived.device_ptr(&stream);
    // SAFETY: `logits` holds `rows` rows `row_stride` apart and `out` holds
    // `rows` u32s (the caller's contract); the scratch holds `MAX_ROWS` ≥ `rows`
    // of each; everything runs on this device's launch stream.
    let status = unsafe {
        run_argmax_rows_f32(
            logits as *const f32,
            rows as i32,
            row_stride as i32,
            live as i32,
            out as *mut u32,
            slots as *mut u64,
            arrived as *mut u32,
            stream.cu_stream() as *mut c_void,
        )
    };
    match status {
        0 => Ok(()),
        1 => crate::bail!(
            "argmax_rows: refused rows={rows} row_stride={row_stride} live={live} and wrote nothing"
        ),
        _ => crate::bail!("argmax_rows: the launch of rows={rows} failed and wrote nothing"),
    }
}
