//! Device-time measurement for the int8 matmul benches: launches captured into a CUDA graph and
//! replayed between two events, so the figure is the kernels' own back-to-back cost and never the
//! host's (a host-timed call carries ~9 µs of launch overhead the device never sees).

use cudarc::driver::result;
use cudarc::driver::result::memcpy_dtod_async;
use cudarc::driver::sys;
use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
use cudarc::driver::sys::CUstream;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::CudaDevice;
use crate::backend::BackendDevice;
use crate::Result;

/// Bytes of distinct weight copies a rotation spans: past the RTX PRO 5000's 96 MiB L2 by enough
/// that a loop over them reads every launch's weight from DRAM.
const ROTATION_BYTES: usize = 320 << 20;

/// Distinct copies of one weight's bytes, enough that a pass over them reads each launch's weight
/// from DRAM rather than the L2.
pub(crate) struct Rotation {
    copies: Vec<CudaSlice<u8>>,
}

impl Rotation {
    /// Copies of the `len` bytes at device address `ptr`, at least 4 and at most 256 of them.
    pub(crate) fn of_device_bytes(dev: &CudaDevice, ptr: u64, len: usize) -> Result<Self> {
        let count = ROTATION_BYTES.div_ceil(len).clamp(4, 256);
        let stream = dev.cuda_stream();
        let mut copies = Vec::with_capacity(count);
        for _ in 0..count {
            // SAFETY: every byte is written by the copy below before any launch reads it.
            let mut c = unsafe { dev.alloc::<u8>(len)? };
            {
                let (dst, _g) = c.device_ptr_mut(&stream);
                // SAFETY: `ptr` is a live allocation of `len` bytes; `dst` was just allocated at
                // `len` bytes; both on this device, ordered on its stream.
                unsafe {
                    memcpy_dtod_async(dst, ptr, len, stream.cu_stream())
                        .map_err(|e| crate::Error::Msg(format!("weight copy: {e:?}")))?;
                }
            }
            copies.push(c);
        }
        dev.synchronize()?;
        Ok(Self { copies })
    }

    /// Every copy's device address, read once — the timed loop is captured into a graph, where
    /// reading a slice's address (which records its sync event) has no place.
    pub(crate) fn ptrs(&self, dev: &CudaDevice) -> Vec<u64> {
        let stream = dev.cuda_stream();
        self.copies
            .iter()
            .map(|c| {
                let (p, _g) = c.device_ptr(&stream);
                p
            })
            .collect()
    }
}

/// Device time the replays of one measurement aim to span; the replay count is sized from an eager
/// pass so a millisecond-scale prefill launch and a microsecond-scale decode launch both measure
/// over enough work to be stable, and neither takes long.
const TARGET_MS: f64 = 150.0;

/// Device µs per launch of `launch` — called with each of `items` and the stream — measured the
/// way a wave issues it: one eager pass (warm-up, and every launcher's one-time setup, timed to
/// size the replays), then the same pass captured into a CUDA graph and replayed between two
/// events. A graph replay issues its launches with no host work between them, so the figure is the
/// kernels' own back-to-back cost (programmatic launches overlapping as they do in a wave).
pub(crate) fn time_graph<T: Copy>(
    dev: &CudaDevice,
    items: &[T],
    mut launch: impl FnMut(T, CUstream) -> Result<()>,
) -> Result<f64> {
    let err = |what: &'static str| move |e| crate::Error::Msg(format!("{what}: {e:?}"));
    // A stream of its own: the device's may be the legacy stream, which cannot capture.
    dev.synchronize()?;
    let stream = dev.cuda_context().new_stream().map_err(err("new stream"))?;
    let cs = stream.cu_stream();
    let t0 = stream
        .record_event(Some(CU_EVENT_DEFAULT))
        .map_err(err("event"))?;
    for &it in items {
        launch(it, cs)?;
    }
    let t1 = stream
        .record_event(Some(CU_EVENT_DEFAULT))
        .map_err(err("event"))?;
    stream.synchronize().map_err(err("sync"))?;
    let eager_ms = t0.elapsed_ms(&t1).map_err(err("elapsed"))? as f64;
    let replays = ((TARGET_MS / eager_ms.max(1e-3)).ceil() as usize).clamp(2, 50);
    // SAFETY: this test's stream, idle, not capturing; ended below.
    unsafe {
        result::stream::begin_capture(
            cs,
            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
        .map_err(err("begin capture"))?;
    }
    let issued: Result<()> = items.iter().try_for_each(|&it| launch(it, cs));
    // SAFETY: the capture begun above, ended whatever the launches returned.
    let graph = unsafe { result::stream::end_capture(cs) }.map_err(err("end capture"))?;
    issued?;
    let mut exec: sys::CUgraphExec = std::ptr::null_mut();
    // SAFETY: a valid graph; default flags.
    unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0).result() }
        .map_err(err("instantiate"))?;
    // SAFETY: an instantiated graph on this stream.
    unsafe { sys::cuGraphLaunch(exec, cs).result() }.map_err(err("warm replay"))?;
    stream.synchronize().map_err(err("sync"))?;
    let start = stream
        .record_event(Some(CU_EVENT_DEFAULT))
        .map_err(err("event"))?;
    for _ in 0..replays {
        // SAFETY: as above.
        unsafe { sys::cuGraphLaunch(exec, cs).result() }.map_err(err("replay"))?;
    }
    let stop = stream
        .record_event(Some(CU_EVENT_DEFAULT))
        .map_err(err("event"))?;
    let ms = start.elapsed_ms(&stop).map_err(err("elapsed"))?;
    stream.synchronize().map_err(err("sync"))?;
    // SAFETY: the executable and graph made above, each destroyed once, the stream drained.
    unsafe {
        sys::cuGraphExecDestroy(exec)
            .result()
            .map_err(err("destroy exec"))?;
        result::graph::destroy(graph).map_err(err("destroy graph"))?;
    }
    Ok(ms as f64 * 1e3 / (replays * items.len()) as f64)
}
