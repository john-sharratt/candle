//! Shared Criterion benchmark plumbing plus one submodule per benchmarked op
//! (`affine`, `conv_transpose2d`, `copy`, `matmul`, `qmatmul`, `random`,
//! `reduce`, `unary`, `where_cond`).
//!
//! `BenchDevice` (implemented for `Device`) supplies `sync()` — a per-backend
//! barrier (`cudaDeviceSynchronize`/`waitUntilCompleted`/no-op on CPU) so
//! timings measure real device work — and `bench_name()`, which prefixes
//! results with the active backend (`cuda_`/`metal_`/`accelerate_`/`mkl_`/
//! `cpu_`). `BenchDeviceHandler::new()` picks the single device to benchmark
//! by feature flag: `metal` > `cuda` > CPU fallback.

pub(crate) mod affine;
pub(crate) mod conv_transpose2d;
pub(crate) mod copy;
pub(crate) mod matmul;
pub(crate) mod qmatmul;
pub(crate) mod random;
pub(crate) mod reduce;
pub(crate) mod unary;
pub(crate) mod where_cond;

use candle_core::{Device, Result};

pub(crate) trait BenchDevice {
    fn sync(&self) -> Result<()>;

    fn bench_name<S: Into<String>>(&self, name: S) -> String;
}

impl BenchDevice for Device {
    fn sync(&self) -> Result<()> {
        match self {
            Device::Cpu => Ok(()),
            Device::Cuda(device) => {
                #[cfg(feature = "cuda")]
                {
                    use candle_core::backend::BackendDevice;
                    device.synchronize()
                }
                #[cfg(not(feature = "cuda"))]
                panic!("Cuda device without cuda feature enabled: {:?}", device)
            }
            Device::Metal(device) => {
                #[cfg(feature = "metal")]
                return Ok(device.wait_until_completed()?);
                #[cfg(not(feature = "metal"))]
                panic!("Metal device without metal feature enabled: {:?}", device)
            }
        }
    }

    fn bench_name<S: Into<String>>(&self, name: S) -> String {
        match self {
            Device::Cpu => {
                let cpu_type = if cfg!(feature = "accelerate") {
                    "accelerate"
                } else if cfg!(feature = "mkl") {
                    "mkl"
                } else {
                    "cpu"
                };
                format!("{}_{}", cpu_type, name.into())
            }
            Device::Cuda(_) => format!("cuda_{}", name.into()),
            Device::Metal(_) => format!("metal_{}", name.into()),
        }
    }
}

struct BenchDeviceHandler {
    devices: Vec<Device>,
}

impl BenchDeviceHandler {
    pub fn new() -> Result<Self> {
        let mut devices = Vec::new();
        if cfg!(feature = "metal") {
            devices.push(Device::new_metal(0)?);
        } else if cfg!(feature = "cuda") {
            devices.push(Device::new_cuda(0)?);
        } else {
            devices.push(Device::Cpu);
        }
        Ok(Self { devices })
    }
}
