//! Which NVIDIA driver model a device runs under (`docs/driver_model.md`).
//!
//! On Windows a CUDA device runs under WDDM — Windows' video memory manager owns
//! VRAM, demotes it to system RAM past capacity, charges system commit for every
//! allocation and batches submissions — or under TCC, the compute-only model in
//! which the NVIDIA driver owns the card as it does on Linux. The memory
//! arithmetic differs between them, so it is read from the device, once, never
//! inferred from the operating system.

#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(feature = "cuda")]
use cudarc::driver::result::device::get_attribute;
#[cfg(feature = "cuda")]
use cudarc::driver::sys::{CUdevice, CUdevice_attribute_enum};

#[cfg(feature = "cuda")]
use crate::{Error, Result};

/// The driver model a device runs under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriverModel {
    /// Windows' display driver model: the OS manages VRAM residency.
    Wddm,
    /// Windows' compute-only driver model: the NVIDIA driver owns the card.
    Tcc,
    /// Any other operating system: the NVIDIA driver owns the card.
    Linux,
}

impl DriverModel {
    /// The model a device reporting `tcc_attribute`
    /// (`CU_DEVICE_ATTRIBUTE_TCC_DRIVER`) runs under, on Windows or not.
    pub fn from_attribute(windows: bool, tcc_attribute: i32) -> Self {
        match (windows, tcc_attribute) {
            (false, _) => Self::Linux,
            (true, 0) => Self::Wddm,
            (true, _) => Self::Tcc,
        }
    }

    /// Its name, as the log and `/v1/memory` give it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Wddm => "wddm",
            Self::Tcc => "tcc",
            Self::Linux => "linux",
        }
    }

    /// Whether the OS manages VRAM residency: an allocation past capacity is
    /// demoted rather than refused, `cuMemGetInfo`'s free is not the residency
    /// signal, and the DXGI per-process budget is.
    pub fn os_manages_residency(self) -> bool {
        matches!(self, Self::Wddm)
    }

    /// Whether launches wait in the driver until something forces their
    /// submission.
    pub fn batches_submissions(self) -> bool {
        matches!(self, Self::Wddm)
    }

    #[cfg(any(feature = "cuda", test))]
    fn code(self) -> u8 {
        match self {
            Self::Wddm => 1,
            Self::Tcc => 2,
            Self::Linux => 3,
        }
    }

    #[cfg(any(feature = "cuda", test))]
    fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Wddm),
            2 => Some(Self::Tcc),
            3 => Some(Self::Linux),
            _ => None,
        }
    }
}

/// Devices whose model is cached; one past it is read every time.
#[cfg(feature = "cuda")]
const CACHED: usize = 16;

/// Each device ordinal's model once read, `0` before.
#[cfg(feature = "cuda")]
static MODELS: [AtomicU8; CACHED] = [const { AtomicU8::new(0) }; CACHED];

/// The model of the device with ordinal `device`, read from the driver the
/// first time and cached: the attribute is fixed for the life of the process,
/// since a change of model applies only at the next boot.
#[cfg(feature = "cuda")]
pub fn driver_model(device: CUdevice) -> Result<DriverModel> {
    let slot = usize::try_from(device).ok().filter(|&i| i < CACHED);
    if let Some(m) = slot.and_then(|i| DriverModel::from_code(MODELS[i].load(Ordering::Relaxed))) {
        return Ok(m);
    }
    // SAFETY: a plain attribute query on a device the caller obtained from a
    // live context.
    let tcc = unsafe {
        get_attribute(
            device,
            CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_TCC_DRIVER,
        )
    }
    .map_err(|e| Error::Msg(format!("reading the driver model of device {device}: {e}")))?;
    let model = DriverModel::from_attribute(cfg!(windows), tcc);
    if let Some(i) = slot {
        MODELS[i].store(model.code(), Ordering::Relaxed);
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The model is the attribute on Windows, and Linux everywhere else.**
    #[test]
    fn the_model_follows_the_attribute_and_the_os() {
        assert_eq!(DriverModel::from_attribute(true, 0), DriverModel::Wddm);
        assert_eq!(DriverModel::from_attribute(true, 1), DriverModel::Tcc);
        assert_eq!(DriverModel::from_attribute(false, 0), DriverModel::Linux);
        assert_eq!(DriverModel::from_attribute(false, 1), DriverModel::Linux);
    }

    /// Only WDDM manages residency and batches submissions; each model has
    /// its own name and its own cache code.
    #[test]
    fn only_wddm_manages_residency_and_batches() {
        let all = [DriverModel::Wddm, DriverModel::Tcc, DriverModel::Linux];
        assert_eq!(
            all.map(DriverModel::os_manages_residency),
            [true, false, false]
        );
        assert_eq!(
            all.map(DriverModel::batches_submissions),
            [true, false, false]
        );
        assert_eq!(all.map(DriverModel::name), ["wddm", "tcc", "linux"]);
        for m in all {
            assert_eq!(DriverModel::from_code(m.code()), Some(m));
        }
        assert_eq!(DriverModel::from_code(0), None);
    }
}
