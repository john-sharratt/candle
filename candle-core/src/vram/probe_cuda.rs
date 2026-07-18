//! The CUDA measurement backend: `cuMemGetInfo` free/total.
//!
//! Primary on Linux (where the NVIDIA driver owns the framebuffer directly and
//! `free` is honest) and the universal fallback on Windows when the DXGI probe
//! can't initialise. On WDDM this `free` is polluted, so the governor prefers
//! [`super::probe_dxgi::DxgiProbe`] there.

use super::reading::{ProbeKind, VramProbe, VramReading};
use crate::{Device, Error, Result};

/// Reads VRAM via the CUDA driver's `cuMemGetInfo`.
pub struct CudaProbe {
    device: Device,
}

impl CudaProbe {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

impl VramProbe for CudaProbe {
    fn read(&self) -> Result<VramReading> {
        match &self.device {
            Device::Cuda(d) => {
                let (free, total) = d.mem_get_info()?;
                Ok(VramReading::new(free as u64, total as u64, ProbeKind::Cuda))
            }
            _ => Err(Error::Msg(
                "CudaProbe constructed on a non-CUDA device".into(),
            )),
        }
    }
}
