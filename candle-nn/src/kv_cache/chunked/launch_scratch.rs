//! The launchers' scratch reserve a forward on a device carries in its
//! forward-wide span, read for pricing.
//!
//! The placement widens the forward span by this reserve
//! ([`super::wave_plan::launch_plan`]); a price of the same wave
//! ([`super::wave_plan::WavePlan::tier_bytes`]) must carry the same figure, or
//! admission fits a wave the placement refuses by the reserve.

#[cfg(feature = "cuda")]
use candle::cuda_backend::scratch_reserve;
use candle::Device;
#[cfg(feature = "cuda")]
use candle::DeviceLocation;

/// Bytes the next forward on `device` will carry for its launchers' scratch:
/// what recent forwards' launchers asked for (`scratch_reserve`).
#[cfg(feature = "cuda")]
pub fn launch_scratch_reserve(device: &Device) -> usize {
    match device.location() {
        DeviceLocation::Cuda { gpu_id } => scratch_reserve(gpu_id),
        _ => 0,
    }
}

/// Nothing: without CUDA no launcher carves scratch from a span.
#[cfg(not(feature = "cuda"))]
pub fn launch_scratch_reserve(_device: &Device) -> usize {
    0
}
