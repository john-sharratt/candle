//! GPU time of a stretch of work, by CUDA events on the device's stream.
//!
//! Every kernel the harness times launches on the device's persistent stream
//! (candle's `cuda_stream()` returns the stored one), so a pair of events
//! around the call measures pure device time — none of the host-side launch
//! or synchronisation cost that `Instant` + `synchronize` would add.

use std::time::Duration;

use candle::cuda_backend::cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
use candle::{Device, Result};

/// Run `f` and return its result with the device time it took. The stream is
/// drained first, so the measurement starts from an idle device.
pub fn gpu_timed<T>(device: &Device, f: impl FnOnce() -> Result<T>) -> Result<(T, Duration)> {
    let stream = match device {
        Device::Cuda(d) => d.cuda_stream(),
        _ => candle::bail!("GPU timing requires a CUDA device"),
    };
    let ev_err = |e| candle::Error::Msg(format!("cuda event: {e:?}"));
    device.synchronize()?;
    let start = stream
        .record_event(Some(CU_EVENT_DEFAULT))
        .map_err(ev_err)?;
    let out = f()?;
    let stop = stream
        .record_event(Some(CU_EVENT_DEFAULT))
        .map_err(ev_err)?;
    let ms = start.elapsed_ms(&stop).map_err(ev_err)?;
    Ok((out, Duration::from_secs_f64(ms as f64 / 1000.0)))
}

/// The median of a set of timings (the upper one of an even count).
pub fn median(mut ts: Vec<Duration>) -> Duration {
    ts.sort();
    ts[ts.len() / 2]
}
