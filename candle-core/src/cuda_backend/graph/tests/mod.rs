//! The driver layer's tests on a real device. Each launch names its stream, so a test shows which stream a launch
//! lands on.
//!
//! The kernel is compiled here rather than taken from the kernel crate, so every launch names its stream explicitly.
//! The wave tests run one at a time under a gate that holds other tests' stream leases off, and each test takes the
//! device's waves with a fresh library, since the capture hub is shared by the whole process. A compiled kernel stays
//! loaded for the life of the process, because executables in the hub may still name it and a later fold into one of
//! them must find its function.
//!
//! - `tests_capture`: tests of what capture records, replays and survives.
//! - `tests_library`: tests of the executable library's learning, trial and lock.
//! - `tests_waves`: tests of wave recording and issue order across threads.

use super::record_gate::{admit_this_thread, exclusive, Exclusive};
use super::{CaptureSession, CaptureStream, ComputeStream, GraphPhase};
use crate::backend::BackendDevice;
use crate::cuda_backend::{CudaDevice, WrapErr};
use crate::quantized::pinned_staging::PinnedBuf;
use crate::Result;
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const N: usize = 4096;

/// Take the device's waves to this test alone, with its executable library
/// fresh: empty, learning, at the card's budget.
///
/// `CudaDevice::new` hands back the one cached device per ordinal, so every
/// test shares its context, its one stream and its capture hub. The gate held
/// to this test ([`exclusive`]) runs the wave tests one at a time and holds
/// every other test's stream lease off, so the stat deltas they assert are
/// their own and no other test's lease turns a segment away. A thread a test
/// spawns joins it with [`admit_this_thread`]. The library's lifecycle
/// (`regime`) is the hub's, so a test that left it locked would otherwise leave
/// every later test running eagerly.
fn wave_lock() -> Exclusive {
    use super::budget::graph_budget_bytes;
    let guard = exclusive();
    let dev = CudaDevice::new(0).expect("a CUDA device");
    let budget = graph_budget_bytes().expect("the card's graph budget");
    dev.capture_hub().reset_library(budget);
    guard
}

const SCALE_ADD: &str = r#"
extern "C" __global__ void scale_add(const float* x, float* y, float mul, float add, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * mul + add;
}
"#;

/// `scale_add`'s signature with another body: `y = x * mul - sub`.
const SCALE_SUB: &str = r#"
extern "C" __global__ void scale_sub(const float* x, float* y, float mul, float sub, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * mul - sub;
}
"#;

fn scale_add(dev: &CudaDevice) -> Result<CudaFunction> {
    compile(dev, "scale_add", SCALE_ADD)
}

fn scale_sub(dev: &CudaDevice) -> Result<CudaFunction> {
    compile(dev, "scale_sub", SCALE_SUB)
}

/// `name` compiled once for the process and kept loaded, as every kernel the
/// engine launches is.
///
/// **Kept, not reloaded per test.** The capture hub is the device's, so its
/// executables outlive any one test, and a later test's segment of the same
/// node kinds folds into them, rewriting their function — the old one must
/// still be loaded for that. A test that unloaded its module at its end left
/// executables naming a function that no longer existed, and the next fold
/// into one of them crashed the driver.
fn compile(dev: &CudaDevice, name: &'static str, src: &str) -> Result<CudaFunction> {
    static LOADED: Mutex<Vec<(&'static str, CudaFunction)>> = Mutex::new(Vec::new());
    let cached = |loaded: &Vec<(&'static str, CudaFunction)>| {
        loaded
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, f)| f.clone())
    };
    if let Some(f) = cached(&LOADED.lock().unwrap_or_else(|p| p.into_inner())) {
        return Ok(f);
    }
    // Loaded with the cache unlocked: the load waits for every recording to
    // end (`load_ptx`), and a test that is recording may be reading the cache.
    let ptx = cudarc::nvrtc::compile_ptx(src).map_err(crate::Error::wrap)?;
    let module = dev.load_ptx(name, &ptx)?;
    let f = module.load_function(name).w()?;
    let mut loaded = LOADED.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(first) = cached(&loaded) {
        return Ok(first);
    }
    loaded.push((name, f.clone()));
    Ok(f)
}

/// `y = x * mul + add` over `N` elements, launched on `stream`.
fn launch(
    f: &CudaFunction,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    mul: f32,
    add: f32,
) -> Result<()> {
    let n = N as i32;
    let mut b = stream.launch_builder(f);
    b.arg(x).arg(y).arg(&mul).arg(&add).arg(&n);
    // SAFETY: the kernel's signature is the five arguments pushed above, and
    // both buffers hold `N` elements.
    unsafe { b.launch(LaunchConfig::for_num_elems(N as u32)) }.w()?;
    Ok(())
}

/// `y = x * mul + add` over the first `n` elements, on a grid sized for `n` —
/// the same kernel at a different shape when `n` differs.
#[allow(clippy::too_many_arguments)]
fn launch_n(
    f: &CudaFunction,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    mul: f32,
    add: f32,
    n: usize,
) -> Result<()> {
    let count = n as i32;
    let mut b = stream.launch_builder(f);
    b.arg(x).arg(y).arg(&mul).arg(&add).arg(&count);
    // SAFETY: the kernel's signature is the five arguments pushed above, and
    // both buffers hold `N ≥ n` elements.
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.w()?;
    Ok(())
}

fn ramp(scale: f32) -> Vec<f32> {
    (0..N).map(|i| i as f32 * scale).collect()
}

/// A run of ordinary tensor ops — element-wise kernels, a cuBLAS matmul, a
/// reduction — with a readback, fresh allocations, frees and a flush in the
/// middle. Returns every value it read back.
fn tensor_chain(dev: &crate::Device, seed: f32) -> Result<Vec<Vec<f32>>> {
    use crate::{DType, Tensor};
    let a = (Tensor::arange(0f32, 64.0, dev)?.reshape((8, 8))? * seed as f64)?;
    let b = (a.sqr()? + 1.0)?;
    let c = a.matmul(&b)?;
    // A readback in the middle of the chain must see everything before it.
    let mid = c.sum_all()?.to_vec0::<f32>()?;
    let d = (c.exp()?.clamp(0f32, 1e6f32)? / 7.0)?;
    let fresh = Tensor::ones((8, 8), DType::F32, dev)?;
    let e = (d + &fresh)?.relu()?;
    drop(fresh);
    if let crate::Device::Cuda(cuda) = dev {
        cuda.flush_launches()?;
    }
    let f = e.matmul(&a.t()?.contiguous()?)?.sum_keepdim(1)?;
    Ok(vec![
        vec![mid],
        e.flatten_all()?.to_vec1()?,
        f.flatten_all()?.to_vec1()?,
    ])
}

/// Sets its flag when dropped — a stand-in for device memory being freed.
struct Freed(Arc<AtomicBool>);

impl Drop for Freed {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// One wave of `launches` launches of `y = x + add`: its topology is the
/// launch count. Returns what `y` holds after it.
fn add_wave(
    dev: &CudaDevice,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    launches: usize,
    add: f32,
) -> Result<Vec<f32>> {
    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    for _ in 0..launches {
        launch(f, &dev.cuda_stream(), x, y, 1.0, add)?;
    }
    capture.finish()?;
    dev.memcpy_dtov(y)
}

/// One `y = x + add` over the first `n` elements, captured on `stream` and
/// audited: a template the slot tests fold.
fn captured_n(
    f: &CudaFunction,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    n: usize,
    add: f32,
) -> Result<(super::exec::Template, super::exec::Audited)> {
    use super::exec::{audit, Template};
    // SAFETY: a created stream the caller owns, not capturing.
    unsafe { super::capture::begin(stream)? };
    launch_n(f, stream, x, y, 1.0, add, n)?;
    // SAFETY: the capture begun above, on this thread.
    let template = Template(unsafe { super::capture::end(stream)? });
    let audited = audit(template.0, 1)?;
    Ok((template, audited))
}

mod tests_capture;
mod tests_library;
mod tests_waves;
