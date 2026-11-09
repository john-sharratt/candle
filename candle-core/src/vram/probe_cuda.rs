//! The CUDA measurement backend: `cuMemGetInfo` free/total.
//!
//! The probe for a device the NVIDIA driver owns — under TCC or on Linux —
//! where `free` is honest and an allocation past it is refused
//! (`docs/driver_model.md` §2). Under WDDM this `free` is polluted by what the
//! OS manages, and the governor reads the DXGI probe instead.

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
