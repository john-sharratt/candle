//! Prints the DXGI per-process VRAM budget for the CUDA device — the
//! OS-authoritative number the WDDM residency work is calibrated against.
//! Ignored by default: it needs a CUDA device and a Windows/WDDM host, and its
//! value is the printout, not an assertion.
#![cfg(all(feature = "cuda", windows))]

use candle_core::vram::{DxgiProbe, VramProbe};
use candle_core::Device;

#[test]
#[ignore]
fn print_dxgi_budget() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let Device::Cuda(cuda) = &device else {
        unreachable!()
    };
    let probe = DxgiProbe::for_cuda_device(cuda)?;
    let r = probe.read()?;
    println!(
        "dxgi: total={:.2}GB ({}MiB) headroom(Budget-CurrentUsage)={:.2}GB ({}MiB)",
        r.total as f64 / 1e9,
        r.total / (1024 * 1024),
        r.headroom as f64 / 1e9,
        r.headroom / (1024 * 1024),
    );
    Ok(())
}
