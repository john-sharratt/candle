//! Which device a stream is the one stream of.
//!
//! A buffer that knows only its stream — [`super::Held`], [`super::Uploaded`] —
//! releases its memory on that stream when it drops. On a device's stream that
//! release has to go through the device ([`CudaDevice::retire`]): issued while a
//! segment records there, it would be taken into the capture. A private stream
//! (a copy stream of its own) never captures, and its releases run at once.

use std::sync::{Arc, Mutex};

use cudarc::driver::CudaStream;

use super::CudaDevice;

/// Every device's stream, by its driver handle.
static DEVICES: Mutex<Vec<(usize, CudaDevice)>> = Mutex::new(Vec::new());

/// Record `dev` as the owner of its stream. The first handle for a stream is
/// kept.
pub(crate) fn register(dev: &CudaDevice) {
    let key = dev.stream_unleased().cu_stream() as usize;
    let mut devices = DEVICES.lock().unwrap_or_else(|p| p.into_inner());
    if !devices.iter().any(|(k, _)| *k == key) {
        devices.push((key, dev.clone()));
    }
}

/// The device whose one stream `stream` is, if it is one.
pub(crate) fn device_of(stream: &Arc<CudaStream>) -> Option<CudaDevice> {
    let key = stream.cu_stream() as usize;
    DEVICES
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, d)| d.clone())
}
