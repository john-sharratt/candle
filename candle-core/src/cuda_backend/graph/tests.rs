//! The driver layer's proofs, on a real device.
//!
//! The kernel is compiled here rather than taken from `candle-kernels` so that
//! every launch names its stream explicitly: the point of these tests is which
//! stream a launch lands on.

use super::{CaptureSession, CaptureStream, ComputeStream, GraphPhase};
use crate::backend::BackendDevice;
use crate::cuda_backend::{CudaDevice, WrapErr};
use crate::quantized::pinned_staging::PinnedBuf;
use crate::Result;
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

const N: usize = 4096;

/// `CudaDevice::new` hands back the one cached device per ordinal, so every
/// test here shares its context and its capture hub: they take this first and
/// run one at a time, so the stat deltas they assert are their own and no
/// test's driver work lands inside another's capture.
static WAVE: Mutex<()> = Mutex::new(());

/// Take the device's waves to this test alone, with its executable library
/// fresh: empty, learning, at the card's budget. The library's lifecycle
/// (`regime`) is the hub's, so a test that left it locked would otherwise
/// leave every later test running eagerly.
fn wave_lock() -> MutexGuard<'static, ()> {
    use super::budget::graph_budget_bytes;
    let guard = WAVE.lock().unwrap_or_else(|p| p.into_inner());
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

/// A captured graph replayed into the null stream reads what null-stream work
/// wrote before it, and what it writes is what null-stream work after it reads
/// — the ordering §4.2 rests on. Per-wave data written to the same addresses
/// between replays is picked up, because the graph bakes addresses, not
/// contents.
#[test]
fn replay_on_the_null_stream_orders_against_null_stream_work() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let null = dev.cuda_stream();
    let mut x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let mut z = dev.alloc_zeros::<f32>(N)?;
    let compute = ComputeStream::of(&dev);
    let mut cap = CaptureStream::new(&dev)?;

    let exec = {
        let s = CaptureSession::begin(&mut cap)?;
        launch(&f, s.stream(), &x, &mut y, 2.0, 1.0)?;
        launch(&f, s.stream(), &y, &mut z, 3.0, 0.0)?;
        s.finish(&compute, 2)?
    };
    assert_eq!(exec.node_count(), 2);
    // A capture executes nothing.
    assert!(dev.memcpy_dtov(&z)?.iter().all(|&v| v == 0.0));

    for scale in [2.0f32, 5.0] {
        null.memcpy_htod(&ramp(scale), &mut x).w()?;
        exec.launch(&compute)?;
        let got = dev.memcpy_dtov(&z)?;
        let want: Vec<f32> = ramp(scale).iter().map(|v| v * 6.0 + 3.0).collect();
        assert_eq!(got, want, "replay at scale {scale}");
    }
    Ok(())
}

/// Another thread keeps submitting to the null stream — allocating, copying,
/// launching and synchronizing, as the persistence thread does — for the whole
/// time a capture is open. The capture neither fails nor records that work,
/// and its replay is exact. This is the non-blocking capture stream plus the
/// `ThreadLocal` mode doing what §4.2 needs of them.
#[test]
fn a_capture_survives_null_stream_work_from_another_thread() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let stop = Arc::new(AtomicBool::new(false));
    let rounds = Arc::new(AtomicUsize::new(0));
    let worker = {
        let (dev, f, stop, rounds) = (dev.clone(), f.clone(), stop.clone(), rounds.clone());
        std::thread::spawn(move || -> Result<()> {
            dev.cuda_context().bind_to_thread().w()?;
            let null = dev.cuda_stream();
            while !stop.load(Ordering::Acquire) {
                let a = dev.memcpy_stod(&ramp(1.0))?;
                let mut b = dev.alloc_zeros::<f32>(N)?;
                launch(&f, &null, &a, &mut b, 1.0, 1.0)?;
                let back = dev.memcpy_dtov(&b)?;
                assert_eq!(back[7], 8.0);
                rounds.fetch_add(1, Ordering::Release);
            }
            Ok(())
        })
    };
    while rounds.load(Ordering::Acquire) < 4 {
        std::thread::yield_now();
    }

    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let mut z = dev.alloc_zeros::<f32>(N)?;
    let compute = ComputeStream::of(&dev);
    let mut cap = CaptureStream::new(&dev)?;
    let before = rounds.load(Ordering::Acquire);
    let exec = {
        let s = CaptureSession::begin(&mut cap)?;
        for _ in 0..8 {
            launch(&f, s.stream(), &x, &mut y, 2.0, 0.0)?;
            launch(&f, s.stream(), &y, &mut z, 1.0, 1.0)?;
            // Hold the capture open while the other thread works.
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        s.finish(&compute, 16)?
    };
    let during = rounds.load(Ordering::Acquire) - before;
    stop.store(true, Ordering::Release);
    worker.join().expect("worker thread panicked")?;
    assert!(
        during > 0,
        "the other thread made no progress while the capture was open"
    );

    exec.launch(&compute)?;
    let want: Vec<f32> = ramp(1.0).iter().map(|v| v * 2.0 + 1.0).collect();
    assert_eq!(dev.memcpy_dtov(&z)?, want);
    Ok(())
}

/// A launch on the null stream while a capture is open is not recorded: it
/// runs at once, and the graph comes up short. This is why only launchers that
/// take their stream from the caller can sit inside a captured region, and why
/// the node count is checked rather than trusted.
#[test]
fn a_default_stream_launch_is_not_recorded_and_the_capture_is_refused() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let compute = ComputeStream::of(&dev);
    let mut cap = CaptureStream::new(&dev)?;
    let s = CaptureSession::begin(&mut cap)?;
    launch(&f, &dev.cuda_stream(), &x, &mut y, 4.0, 0.0)?;
    let err = match s.finish(&compute, 1) {
        Ok(_) => panic!("a capture that recorded nothing was instantiated"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("recorded 0 nodes where 1"),
        "unexpected refusal: {err}"
    );
    // It ran eagerly instead.
    assert_eq!(dev.memcpy_dtov(&y)?[3], 12.0);
    Ok(())
}

/// The repository's own launchers, given the session's stream, record into a
/// graph: an RMSNorm, a sigmoid, an in-place residual add and the MoE router —
/// one per migrated family — captured as four nodes and replayed bit-identical
/// to the same four launched eagerly on the compute stream.
#[test]
fn migrated_launchers_record_on_the_capture_stream() -> Result<()> {
    let _wave = wave_lock();
    use candle_kernels::simple::binary::{run_binary_inplace_op, BinaryDType, BinaryInplaceOp};
    use candle_kernels::simple::moe_scatter::{run_moe_route, MoeScatterDType};
    use candle_kernels::simple::reduce::run_rmsnorm_op;
    use candle_kernels::simple::unary::{run_unary_op, UnaryDType, UnaryOp};
    use cudarc::driver::DevicePtr;
    use std::ffi::c_void;

    const ROWS: usize = 4;
    const COLS: usize = 1024;
    const EXPERTS: usize = 64;
    const TOP_K: usize = 8;
    let dev = CudaDevice::new(0)?;
    let null = dev.cuda_stream();
    let x_host: Vec<f32> = (0..ROWS * COLS)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let alpha_host: Vec<f32> = (0..COLS).map(|i| 1.0 + (i % 7) as f32 / 16.0).collect();
    let logits_host: Vec<f32> = (0..ROWS * EXPERTS)
        .map(|i| ((i * 53 % 97) as f32) / 13.0)
        .collect();
    let x = dev.memcpy_stod(&x_host)?;
    let alpha = dev.memcpy_stod(&alpha_host)?;
    let logits = dev.memcpy_stod(&logits_host)?;
    let mut y = dev.alloc_zeros::<f32>(ROWS * COLS)?;
    let mut z = dev.alloc_zeros::<f32>(ROWS * COLS)?;
    let mut idx = dev.alloc_zeros::<u32>(ROWS * TOP_K)?;
    let mut w = dev.alloc_zeros::<f32>(ROWS * TOP_K)?;

    // The four launches, on whatever stream `s` names.
    let launch_all = |s: *mut c_void,
                      y: &mut CudaSlice<f32>,
                      z: &mut CudaSlice<f32>,
                      idx: &mut CudaSlice<u32>,
                      w: &mut CudaSlice<f32>| {
        let (xp, _g0) = x.device_ptr(&null);
        let (ap, _g1) = alpha.device_ptr(&null);
        let (lp, _g2) = logits.device_ptr(&null);
        let (yp, _g3) = y.device_ptr(&null);
        let (zp, _g4) = z.device_ptr(&null);
        let (ip, _g5) = idx.device_ptr(&null);
        let (wp, _g6) = w.device_ptr(&null);
        // SAFETY: every buffer holds the element count its launch reads or
        // writes, and all four kernels are launched on `s`.
        unsafe {
            run_rmsnorm_op(
                0, // F32
                xp as *const c_void,
                yp as *mut c_void,
                ap as *const c_void,
                ROWS as i32,
                COLS as i32,
                1e-6,
                s,
            );
            run_unary_op(
                UnaryOp::Sigmoid as i32,
                UnaryDType::F32 as i32,
                ROWS * COLS,
                2,
                std::ptr::null(),
                yp as *const c_void,
                zp as *mut c_void,
                s,
            );
            run_binary_inplace_op(
                BinaryInplaceOp::Add as i32,
                BinaryDType::F32 as i32,
                ROWS * COLS,
                2,
                std::ptr::null(),
                zp as *mut c_void,
                xp as *const c_void,
                s,
            );
            run_moe_route(
                MoeScatterDType::F32 as i32,
                lp as *const c_void,
                ip as *mut u32,
                wp as *mut f32,
                ROWS as i32,
                EXPERTS as i32,
                EXPERTS as i32,
                TOP_K as i32,
                1,
                s,
            );
        }
    };

    launch_all(
        null.cu_stream() as *mut c_void,
        &mut y,
        &mut z,
        &mut idx,
        &mut w,
    );
    let (y_eager, z_eager) = (dev.memcpy_dtov(&y)?, dev.memcpy_dtov(&z)?);
    let (idx_eager, w_eager) = (dev.memcpy_dtov(&idx)?, dev.memcpy_dtov(&w)?);
    assert!(
        y_eager.iter().any(|&v| v != 0.0),
        "the eager launches wrote nothing"
    );

    null.memcpy_htod(&vec![0f32; ROWS * COLS], &mut y).w()?;
    null.memcpy_htod(&vec![0f32; ROWS * COLS], &mut z).w()?;
    null.memcpy_htod(&vec![0u32; ROWS * TOP_K], &mut idx).w()?;
    null.memcpy_htod(&vec![0f32; ROWS * TOP_K], &mut w).w()?;
    let compute = ComputeStream::of(&dev);
    let mut cap = CaptureStream::new(&dev)?;
    let exec = {
        let s = CaptureSession::begin(&mut cap)?;
        launch_all(
            s.stream().cu_stream() as *mut c_void,
            &mut y,
            &mut z,
            &mut idx,
            &mut w,
        );
        s.finish(&compute, 4)?
    };
    assert_eq!(exec.node_count(), 4);
    assert!(
        dev.memcpy_dtov(&y)?.iter().all(|&v| v == 0.0),
        "a capture executed a kernel"
    );

    exec.launch(&compute)?;
    assert_eq!(dev.memcpy_dtov(&y)?, y_eager, "rmsnorm");
    assert_eq!(dev.memcpy_dtov(&z)?, z_eager, "sigmoid + residual add");
    assert_eq!(dev.memcpy_dtov(&idx)?, idx_eager, "router indices");
    assert_eq!(dev.memcpy_dtov(&w)?, w_eager, "router weights");
    Ok(())
}

/// A recapture of the same launches with different scalars and different
/// buffers folds into the existing executable in place, and its replay runs
/// the new scalars on the new buffers — what a per-wave capture rests on.
#[test]
fn a_recapture_with_new_scalars_and_addresses_updates_in_place() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y0 = dev.alloc_zeros::<f32>(N)?;
    let mut y1 = dev.alloc_zeros::<f32>(N)?;
    let compute = ComputeStream::of(&dev);
    let mut cap = CaptureStream::new(&dev)?;
    let mut slot = None;

    let s = CaptureSession::begin(&mut cap)?;
    launch(&f, s.stream(), &x, &mut y0, 2.0, 1.0)?;
    assert!(!s.finish_into(&mut slot, &compute, 1)?, "first capture");
    slot.as_ref().expect("instantiated").launch(&compute)?;
    let want0: Vec<f32> = ramp(1.0).iter().map(|v| v * 2.0 + 1.0).collect();
    assert_eq!(dev.memcpy_dtov(&y0)?, want0);

    let s = CaptureSession::begin(&mut cap)?;
    launch(&f, s.stream(), &x, &mut y1, 3.0, -1.0)?;
    assert!(
        s.finish_into(&mut slot, &compute, 1)?,
        "same topology must update in place"
    );
    slot.as_ref().expect("updated").launch(&compute)?;
    let want1: Vec<f32> = ramp(1.0).iter().map(|v| v * 3.0 - 1.0).collect();
    assert_eq!(dev.memcpy_dtov(&y1)?, want1);
    assert_eq!(
        dev.memcpy_dtov(&y0)?,
        want0,
        "the old target was not rewritten"
    );

    // A different topology cannot be folded and is instantiated afresh.
    let s = CaptureSession::begin(&mut cap)?;
    launch(&f, s.stream(), &x, &mut y0, 1.0, 0.0)?;
    launch(&f, s.stream(), &y0, &mut y1, 1.0, 5.0)?;
    assert!(!s.finish_into(&mut slot, &compute, 2)?, "new topology");
    let exec = slot.as_ref().expect("re-instantiated");
    assert_eq!(exec.node_count(), 2);
    exec.launch(&compute)?;
    let want2: Vec<f32> = ramp(1.0).iter().map(|v| v + 5.0).collect();
    assert_eq!(dev.memcpy_dtov(&y1)?, want2);
    Ok(())
}

/// A recapture with another kernel at a node — what a kernel variant chosen by
/// the wave's width does to a segment — folds into the existing executable,
/// and the replay runs the new kernel. This is why the library finds a segment
/// by its node kinds and not by its kernels (`library`).
#[test]
fn a_recapture_with_another_kernel_at_a_node_folds_in_place() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let (add, sub) = (scale_add(&dev)?, scale_sub(&dev)?);
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let compute = ComputeStream::of(&dev);
    let mut cap = CaptureStream::new(&dev)?;
    let mut slot = None;

    let s = CaptureSession::begin(&mut cap)?;
    launch(&add, s.stream(), &x, &mut y, 2.0, 1.0)?;
    assert!(!s.finish_into(&mut slot, &compute, 1)?, "first capture");

    let s = CaptureSession::begin(&mut cap)?;
    launch(&sub, s.stream(), &x, &mut y, 2.0, 1.0)?;
    assert!(
        s.finish_into(&mut slot, &compute, 1)?,
        "another kernel at the node must fold in place"
    );
    slot.as_ref().expect("folded").launch(&compute)?;
    let want: Vec<f32> = ramp(1.0).iter().map(|v| v * 2.0 - 1.0).collect();
    assert_eq!(dev.memcpy_dtov(&y)?, want);
    Ok(())
}

/// A host upload issued inside a region is recorded as a memcpy node reading
/// the host buffer at replay time, and the audit refuses the graph: the buffer
/// is gone by then. A device-to-device copy reads nothing the host owns, and
/// is recorded and replayed in order like a kernel.
#[test]
fn a_host_copy_inside_a_capture_is_refused_and_a_device_copy_is_kept() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let mut z = dev.alloc_zeros::<f32>(N)?;
    let compute = ComputeStream::of(&dev);
    let mut cap = CaptureStream::new(&dev)?;

    let exec = {
        let s = CaptureSession::begin(&mut cap)?;
        launch(&f, s.stream(), &x, &mut y, 1.0, 3.0)?;
        s.stream().memcpy_dtod(&y, &mut z).w()?;
        s.finish(&compute, 2)?
    };
    exec.launch(&compute)?;
    let want: Vec<f32> = ramp(1.0).iter().map(|v| v + 3.0).collect();
    assert_eq!(dev.memcpy_dtov(&z)?, want);

    let s = CaptureSession::begin(&mut cap)?;
    launch(&f, s.stream(), &x, &mut y, 1.0, 0.0)?;
    s.stream().memcpy_htod(&ramp(2.0), &mut z).w()?;
    let err = match s.finish(&compute, 2) {
        Ok(_) => panic!("a capture holding a host upload was instantiated"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("copy CU_MEMORYTYPE_HOST -> CU_MEMORYTYPE_DEVICE"),
        "unexpected refusal: {err}"
    );
    Ok(())
}

/// What a per-wave capture costs against launching eagerly, on this driver.
///
/// `LAUNCHES` small kernels — about one forward's worth on Qwen3.6 — are
/// issued (a) eagerly on the null stream, (b) captured, folded into the
/// executable with an update and replayed, and (c) replayed from an executable
/// already up to date. Each is timed host-side to completion. Prints the three
/// per-launch costs; asserts only that the replays computed the same thing.
#[test]
#[ignore = "benchmark: cargo test -p candle-core --features cuda --release graph::tests::per_wave_capture_cost -- --ignored --nocapture"]
fn per_wave_capture_cost() -> Result<()> {
    const LAUNCHES: usize = 1250;
    const ROUNDS: usize = 20;
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let null = dev.cuda_stream();
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let compute = ComputeStream::of(&dev);
    let mut cap = CaptureStream::new(&dev)?;
    let mut slot = None;

    let mut eager = std::time::Duration::ZERO;
    let mut captured = std::time::Duration::ZERO;
    let mut capture_only = std::time::Duration::ZERO;
    let mut replay = std::time::Duration::ZERO;
    let mut updated = 0usize;
    for round in 0..=ROUNDS {
        let add = round as f32;
        let t = std::time::Instant::now();
        for _ in 0..LAUNCHES {
            launch(&f, &null, &x, &mut y, 1.0, add)?;
        }
        null.synchronize().w()?;
        let e = t.elapsed();

        let t = std::time::Instant::now();
        let s = CaptureSession::begin(&mut cap)?;
        for _ in 0..LAUNCHES {
            launch(&f, s.stream(), &x, &mut y, 1.0, add)?;
        }
        let c0 = t.elapsed();
        updated += s.finish_into(&mut slot, &compute, LAUNCHES)? as usize;
        slot.as_ref().expect("captured").launch(&compute)?;
        null.synchronize().w()?;
        let c = t.elapsed();

        let t = std::time::Instant::now();
        slot.as_ref().expect("captured").launch(&compute)?;
        null.synchronize().w()?;
        let r = t.elapsed();
        if round > 0 {
            eager += e;
            captured += c;
            capture_only += c0;
            replay += r;
        }
        assert_eq!(dev.memcpy_dtov(&y)?[9], 9.0 + add);
    }
    let per = |d: std::time::Duration| d.as_secs_f64() * 1e6 / (ROUNDS * LAUNCHES) as f64;
    println!(
        "per launch over {LAUNCHES} x {ROUNDS}: eager {:.2} us | capture+update+replay {:.2} us \
         (recording alone {:.2} us) | replay {:.2} us | updated in place {updated}/{}",
        per(eager),
        per(captured),
        per(capture_only),
        per(replay),
        ROUNDS + 1,
    );
    Ok(())
}

/// What `cuGraphExecUpdate` costs on a segment-sized graph by how much of it
/// changed — nothing, one node's scalar, every node's scalar — against rewriting
/// one changed node alone with `cuGraphExecKernelNodeSetParams`. Prints µs per
/// call for each; asserts nothing but that every path took effect.
#[test]
#[ignore = "benchmark: cargo test -p candle-core --features cuda --release graph::tests::exec_update_cost_by_change -- --ignored --nocapture"]
fn exec_update_cost_by_change() -> Result<()> {
    use cudarc::driver::{result, sys};
    const NODES: usize = 220;
    const ROUNDS: usize = 50;
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let stream = dev.cuda_context().new_stream().w()?;

    // One capture of NODES launches; node `i`'s `add` is `adds[i]`.
    let capture = |adds: &[f32], y: &mut CudaSlice<f32>| -> Result<sys::CUgraph> {
        // SAFETY: a created stream this test owns, not capturing.
        unsafe { super::capture::begin(&stream)? };
        for &add in adds {
            launch(&f, &stream, &x, y, 1.0, add)?;
        }
        // SAFETY: the capture begun above, on this thread.
        unsafe { super::capture::end(&stream) }
    };
    let base: Vec<f32> = vec![0.0; NODES];
    let template = capture(&base, &mut y)?;
    let mut exec: sys::CUgraphExec = std::ptr::null_mut();
    // SAFETY: a valid graph; default flags.
    unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, template, 0).result() }.w()?;

    let update = |graph: sys::CUgraph| -> Result<f64> {
        let mut info = sys::CUgraphExecUpdateResultInfo {
            result: sys::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS,
            errorNode: std::ptr::null_mut(),
            errorFromNode: std::ptr::null_mut(),
        };
        let t = std::time::Instant::now();
        // SAFETY: a valid executable and a graph of the same topology.
        unsafe { sys::cuGraphExecUpdate_v2(exec, graph, &mut info).result() }.w()?;
        let us = t.elapsed().as_secs_f64() * 1e6;
        assert_eq!(
            info.result,
            sys::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS
        );
        // SAFETY: a graph this test captured, destroyed once.
        unsafe { result::graph::destroy(graph) }.w()?;
        Ok(us)
    };
    let (mut same, mut one, mut all, mut set_one) = (0.0, 0.0, 0.0, 0.0);
    for round in 0..ROUNDS {
        let r = round as f32 + 1.0;
        same += update(capture(&base, &mut y)?)?;
        let mut adds = base.clone();
        adds[NODES / 2] = r;
        one += update(capture(&adds, &mut y)?)?;
        let adds: Vec<f32> = (0..NODES).map(|i| r + i as f32).collect();
        all += update(capture(&adds, &mut y)?)?;

        // One node's parameters, rewritten in the executable directly: the
        // fresh capture supplies them, the template supplies the node handle.
        let mut adds = base.clone();
        adds[NODES / 2] = r;
        let fresh = capture(&adds, &mut y)?;
        let nodes_of = |g: sys::CUgraph| -> Result<Vec<sys::CUgraphNode>> {
            let mut n = NODES;
            let mut v = vec![std::ptr::null_mut(); NODES];
            // SAFETY: `v` holds `n` slots.
            unsafe { sys::cuGraphGetNodes(g, v.as_mut_ptr(), &mut n).result() }.w()?;
            Ok(v)
        };
        let (old, new) = (nodes_of(template)?, nodes_of(fresh)?);
        let mut p = std::mem::MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::uninit();
        // SAFETY: a kernel node of `fresh`; the driver fills the parameters.
        let p = unsafe {
            sys::cuGraphKernelNodeGetParams_v2(new[NODES / 2], p.as_mut_ptr())
                .result()
                .w()?;
            p.assume_init()
        };
        let t = std::time::Instant::now();
        // SAFETY: the executable's own node, parameters of the same function.
        unsafe { sys::cuGraphExecKernelNodeSetParams_v2(exec, old[NODES / 2], &p).result() }.w()?;
        set_one += t.elapsed().as_secs_f64() * 1e6;
        // SAFETY: a graph this test captured, destroyed once.
        unsafe { result::graph::destroy(fresh) }.w()?;
    }
    // SAFETY: the executable and template this test made, launched and destroyed once.
    unsafe {
        sys::cuGraphLaunch(exec, stream.cu_stream()).result().w()?;
        stream.synchronize().w()?;
        sys::cuGraphExecDestroy(exec).result().w()?;
        result::graph::destroy(template).w()?;
    }
    let r = ROUNDS as f64;
    println!(
        "{NODES}-node exec update, µs per call: nothing changed {:.1} | one node {:.1} | \
         every node {:.1} || one node via SetParams {:.2}",
        same / r,
        one / r,
        all / r,
        set_one / r
    );
    Ok(())
}

/// What the capturing thread's own null-stream calls do while a capture is
/// open: a synchronise, a readback, a host upload and a pool allocation.
/// Prints each outcome and whether the capture survived it.
#[test]
#[ignore = "probe: cargo test -p candle-core --features cuda --release graph::tests::null_stream_calls_during_capture -- --ignored --nocapture"]
fn null_stream_calls_during_capture() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let null = dev.cuda_stream();
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let compute = ComputeStream::of(&dev);
    let mut cap = CaptureStream::new(&dev)?;
    type Probe<'a> = Box<dyn Fn() -> Result<()> + 'a>;
    let probes: Vec<(&str, Probe)> = vec![
        ("synchronize", Box::new(|| null.synchronize().w())),
        ("memcpy_dtov", Box::new(|| dev.memcpy_dtov(&x).map(|_| ()))),
        (
            "memcpy_htod",
            Box::new(|| {
                let mut d = dev.alloc_zeros::<f32>(N)?;
                dev.memcpy_htod(&ramp(1.0), &mut d)
            }),
        ),
        (
            "alloc",
            Box::new(|| unsafe { dev.alloc::<f32>(N) }.map(|_| ())),
        ),
    ];
    let doomed = std::cell::RefCell::new(Some(dev.alloc_zeros::<f32>(N)?));
    let free: Probe = Box::new(|| {
        drop(doomed.borrow_mut().take());
        Ok(())
    });
    let probes = probes.into_iter().chain([("free", free)]);
    for (name, probe) in probes {
        let mut y = dev.alloc_zeros::<f32>(N)?;
        let s = CaptureSession::begin(&mut cap)?;
        launch(&f, s.stream(), &x, &mut y, 1.0, 0.0)?;
        let outcome = probe();
        let finished = s.finish(&compute, 1);
        println!(
            "{name}: call {:?} | capture {:?}",
            outcome.as_ref().map_err(|e| e.to_string()),
            finished.as_ref().map(|_| ()).map_err(|e| e.to_string())
        );
        null.synchronize().w()?;
    }
    Ok(())
}

/// What ANOTHER thread's calls do to a capture open on this one: each probe
/// runs on a second thread, start to finish, while the capture records. Prints
/// each call's outcome and whether the capture survived it — the facts the
/// record gate's list of refused operations (`device_wide`, `build_lints.rs`,
/// `clippy.toml`) rests on.
#[test]
#[ignore = "probe: cargo test -p candle-core --features cuda --release graph::tests::other_threads_calls_during_capture -- --ignored --nocapture --test-threads=1"]
fn other_threads_calls_during_capture() -> Result<()> {
    use cudarc::driver::{result, sys};
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let null = dev.cuda_stream();
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let compute = ComputeStream::of(&dev);
    let mut cap = CaptureStream::new(&dev)?;
    type Probe = Box<dyn Fn(&CudaDevice) -> Result<()> + Send + Sync>;
    let probes: Vec<(&str, Probe)> = vec![
        (
            "null stream synchronize",
            Box::new(|d| d.cuda_stream().synchronize().w()),
        ),
        (
            "memcpy_dtov",
            Box::new(|d| d.memcpy_dtov(&d.memcpy_stod(&[1f32; 16])?).map(|_| ())),
        ),
        (
            "pool alloc + free",
            Box::new(|d| unsafe { d.alloc::<f32>(1 << 20) }.map(drop)),
        ),
        (
            "cuMemAlloc + cuMemFree",
            Box::new(|d| unsafe {
                d.cuda_stream().context().bind_to_thread().w()?;
                let p = result::malloc_sync(1 << 20).w()?;
                result::free_sync(p).w()
            }),
        ),
        (
            "cuMemHostAlloc + cuMemFreeHost",
            Box::new(|d| unsafe {
                d.cuda_stream().context().bind_to_thread().w()?;
                let mut p = std::ptr::null_mut();
                sys::cuMemHostAlloc(&mut p, 1 << 20, 0).result().w()?;
                sys::cuMemFreeHost(p).result().w()
            }),
        ),
        (
            "kernel launch on the null stream",
            Box::new(move |d| {
                let mut y = d.alloc_zeros::<f32>(N)?;
                let g = scale_add(d)?;
                let x = d.memcpy_stod(&ramp(1.0))?;
                launch(&g, &d.cuda_stream(), &x, &mut y, 1.0, 0.0)
            }),
        ),
        (
            "stream create + destroy",
            Box::new(|d| d.cuda_context().new_stream().w().map(drop)),
        ),
        (
            "device handle (its own cuBLAS + curand)",
            Box::new(|_| CudaDevice::new(0).map(drop)),
        ),
        (
            "cuBLAS handle (ungated)",
            #[allow(clippy::disallowed_methods)]
            Box::new(|d| {
                cudarc::cublas::CudaBlas::new(d.cuda_stream())
                    .map(drop)
                    .map_err(crate::Error::wrap)
            }),
        ),
        (
            "curand generator",
            Box::new(|d| {
                cudarc::curand::CudaRng::new(0, d.cuda_stream())
                    .map(drop)
                    .map_err(crate::Error::wrap)
            }),
        ),
        // The two the gate exists for, made raw here to show what it prevents.
        (
            "context synchronize (ungated)",
            #[allow(clippy::disallowed_methods)]
            Box::new(|d| d.cuda_context().synchronize().w()),
        ),
        (
            "module load (ungated)",
            #[allow(clippy::disallowed_methods)]
            Box::new(|d| {
                let ptx = cudarc::nvrtc::compile_ptx(
                    "extern \"C\" __global__ void probe_k(float* p) { p[0] = 1.0f; }",
                )
                .map_err(crate::Error::wrap)?;
                d.cuda_context().load_module(ptx).w().map(drop)
            }),
        ),
    ];
    for (name, probe) in &probes {
        let mut y = dev.alloc_zeros::<f32>(N)?;
        let s = CaptureSession::begin(&mut cap)?;
        launch(&f, s.stream(), &x, &mut y, 1.0, 0.0)?;
        let other = dev.clone();
        let outcome = std::thread::scope(|sc| {
            sc.spawn(|| probe(&other).map_err(|e| e.to_string()))
                .join()
                .expect("the probe thread")
        });
        let finished = s.finish(&compute, 1);
        println!(
            "{name}: call {outcome:?} | capture {:?}",
            finished.as_ref().map(|_| ()).map_err(|e| e.to_string())
        );
        null.synchronize().w()?;
    }
    Ok(())
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

/// The chain run under a wave capture reads back exactly what it reads back
/// eagerly, wave after wave with changing scalars — the capture changes when
/// launches reach the driver, never what they compute.
#[test]
fn a_wave_capture_is_bit_identical_to_eager_execution() -> Result<()> {
    let _wave = wave_lock();
    let device = crate::Device::new_cuda(0)?;
    let crate::Device::Cuda(dev) = &device else {
        unreachable!("asked for a CUDA device")
    };
    let before = dev.capture_stats();
    for (i, seed) in [0.01f32, 0.02, 0.015, 0.02].into_iter().enumerate() {
        let eager = tensor_chain(&device, seed)?;
        let capture = dev.begin_wave_capture()?;
        dev.record_launches()?;
        let captured = tensor_chain(&device, seed);
        capture.finish()?;
        assert_eq!(captured?, eager, "wave {i} at seed {seed}");
    }
    let after = dev.capture_stats();
    assert_eq!(after.waves - before.waves, 4);
    assert!(
        after.segments - before.segments >= 8,
        "the readback, the allocation and the flush each end a segment: {after:?}"
    );
    assert!(
        after.updated > before.updated,
        "later waves fold into the first wave's executables: {after:?}"
    );
    Ok(())
}

/// Waves of two shapes taking turns each keep their own executable for the
/// same segment ordinal: once each shape has run, every later wave folds in
/// place — and each still computes its own result. (The first two waves may
/// instantiate or fold, depending on what earlier tests left on the device.)
#[test]
fn alternating_wave_shapes_each_fold_into_their_own_executable() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.alloc_zeros::<f32>(N)?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    dev.memcpy_htod(&ramp(1.0), &mut y)?;
    let mut before = dev.capture_stats();
    for wave in 0..6u32 {
        if wave == 2 {
            before = dev.capture_stats();
        }
        let capture = dev.begin_wave_capture()?;
        dev.record_launches()?;
        // Shape A is one launch, shape B two: y = 0·x + (wave + launch).
        for k in 0..(1 + wave % 2) {
            launch(&f, &dev.cuda_stream(), &x, &mut y, 0.0, (wave + k) as f32)?;
        }
        capture.finish()?;
        let want = (wave + wave % 2) as f32;
        assert!(
            dev.memcpy_dtov(&y)?.iter().all(|&v| v == want),
            "wave {wave}"
        );
    }
    let after = dev.capture_stats();
    assert_eq!(after.instantiated - before.instantiated, 0, "{after:?}");
    assert_eq!(after.updated - before.updated, 4, "{after:?}");
    Ok(())
}

/// One kernel at two grid sizes taking turns — the same topology, so either
/// capture could be folded over the other — each keep an executable at their
/// own shape once both have recurred: every later wave folds in place without
/// rewriting another shape's executable, and each computes its own result.
#[test]
fn recurring_shapes_of_one_topology_each_keep_their_executable() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let mut before = dev.capture_stats();
    for wave in 0..10u32 {
        if wave == 4 {
            before = dev.capture_stats();
        }
        let n = if wave % 2 == 0 { N } else { N / 2 };
        dev.memcpy_htod(&vec![-1.0f32; N], &mut y)?;
        let capture = dev.begin_wave_capture()?;
        dev.record_launches()?;
        launch_n(&f, &dev.cuda_stream(), &x, &mut y, 1.0, wave as f32, n)?;
        capture.finish()?;
        let want: Vec<f32> = (0..N)
            .map(|i| if i < n { i as f32 + wave as f32 } else { -1.0 })
            .collect();
        assert_eq!(dev.memcpy_dtov(&y)?, want, "wave {wave}");
    }
    let after = dev.capture_stats();
    assert_eq!(after.instantiated - before.instantiated, 0, "{after:?}");
    assert_eq!(after.reshaped - before.reshaped, 0, "{after:?}");
    assert_eq!(after.updated - before.updated, 6, "{after:?}");
    Ok(())
}

/// An upload issued while recording is staged and recorded rather than run:
/// it reaches the device in issue order — after the launch before it read the
/// old contents, before the launch after it reads the new — although the
/// caller's host buffer is overwritten the moment the call returns, and the
/// segment is not cut for it.
#[test]
fn an_upload_while_recording_lands_in_issue_order_without_a_cut() -> Result<()> {
    use cudarc::driver::DevicePtr;
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let compute = dev.compute_stream();
    let mut table = dev.alloc_zeros::<f32>(N)?;
    let mut before = dev.alloc_zeros::<f32>(N)?;
    let mut after = dev.alloc_zeros::<f32>(N)?;
    dev.memcpy_htod(&ramp(1.0), &mut table)?;
    let at = table.device_ptr(&compute).0;

    let segments = dev.capture_stats().segments;
    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    launch(&f, &dev.cuda_stream(), &table, &mut before, 1.0, 0.0)?;
    let mut host = ramp(3.0);
    // SAFETY: `host` is `N` f32s.
    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, N * 4) };
    dev.upload_raw(at, bytes)?;
    host.iter_mut().for_each(|v| *v = -1.0);
    launch(&f, &dev.cuda_stream(), &table, &mut after, 1.0, 0.0)?;
    capture.finish()?;

    assert_eq!(
        dev.memcpy_dtov(&before)?,
        ramp(1.0),
        "read before the upload"
    );
    assert_eq!(dev.memcpy_dtov(&after)?, ramp(3.0), "read after the upload");
    assert_eq!(
        dev.capture_stats().segments - segments,
        1,
        "the upload did not end the segment"
    );
    Ok(())
}

/// Two staged-upload tables in one recording share the staging scratch, and
/// each launch that reads it still sees its own: the second copy is recorded
/// after the first table's reader, so it cannot land under it. Neither call
/// ends the segment.
#[test]
fn staged_uploads_while_recording_keep_their_order_without_a_cut() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let mut first = dev.alloc_zeros::<f32>(N)?;
    let mut second = dev.alloc_zeros::<f32>(N)?;
    // Grow the scratch eagerly first: a recorded call never allocates.
    dev.with_staged_upload(&ramp(0.0), |_| Ok(()))?;
    dev.synchronize()?;

    // Launch `scale_add` reading the staged table at `at` into `out`.
    let read = |at: u64, out: &mut CudaSlice<f32>| -> Result<()> {
        let stream = dev.cuda_stream();
        let n = N as i32;
        let (mul, add) = (1.0f32, 0.0f32);
        let mut b = stream.launch_builder(&f);
        b.arg(&at).arg(out).arg(&mul).arg(&add).arg(&n);
        // SAFETY: `at` holds `N` f32s staged by the caller, `out` holds `N`.
        unsafe { b.launch(LaunchConfig::for_num_elems(N as u32)) }.w()?;
        Ok(())
    };

    let segments = dev.capture_stats().segments;
    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    dev.with_staged_upload(&ramp(2.0), |at| read(at, &mut first))?;
    dev.with_staged_upload(&ramp(5.0), |at| read(at, &mut second))?;
    capture.finish()?;

    assert_eq!(
        dev.memcpy_dtov(&first)?,
        ramp(2.0),
        "the first table's reader"
    );
    assert_eq!(
        dev.memcpy_dtov(&second)?,
        ramp(5.0),
        "the second table's reader"
    );
    assert_eq!(
        dev.capture_stats().segments - segments,
        1,
        "a staged upload ended the segment"
    );
    Ok(())
}

/// Two uses of the staging scratch in one recording — each a kernel writing it
/// and a kernel reading it back, as a quantized matmul's quantize and product
/// do — each read their own write, and neither ends the segment.
#[test]
fn staging_scratch_while_recording_keeps_its_order_without_a_cut() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let a = dev.memcpy_stod(&ramp(2.0))?;
    let b = dev.memcpy_stod(&ramp(5.0))?;
    let mut first = dev.alloc_zeros::<f32>(N)?;
    let mut second = dev.alloc_zeros::<f32>(N)?;
    // Grow the scratch eagerly first: a recorded call never allocates.
    dev.with_staging(N * 4, |_| Ok(()))?;
    dev.synchronize()?;

    // Copy `src` into the scratch with one launch, then the scratch into `out`
    // with another.
    let through = |src: &CudaSlice<f32>, out: &mut CudaSlice<f32>| -> Result<()> {
        dev.with_staging(N * 4, |scratch| {
            let stream = dev.cuda_stream();
            let n = N as i32;
            let (mul, add) = (1.0f32, 0.0f32);
            let (s, _gs) = scratch.device_ptr(&stream);
            let mut w = stream.launch_builder(&f);
            w.arg(src).arg(&s).arg(&mul).arg(&add).arg(&n);
            // SAFETY: the scratch holds `N` f32s, `src` holds `N`.
            unsafe { w.launch(LaunchConfig::for_num_elems(N as u32)) }.w()?;
            let mut r = stream.launch_builder(&f);
            r.arg(&s).arg(out).arg(&mul).arg(&add).arg(&n);
            // SAFETY: as above, `out` holds `N`.
            unsafe { r.launch(LaunchConfig::for_num_elems(N as u32)) }.w()?;
            Ok(())
        })
    };

    let segments = dev.capture_stats().segments;
    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    through(&a, &mut first)?;
    through(&b, &mut second)?;
    capture.finish()?;

    assert_eq!(dev.memcpy_dtov(&first)?, ramp(2.0), "the first use's read");
    assert_eq!(
        dev.memcpy_dtov(&second)?,
        ramp(5.0),
        "the second use's read"
    );
    assert_eq!(
        dev.capture_stats().segments - segments,
        1,
        "a use of the staging scratch ended the segment"
    );
    Ok(())
}

/// A launcher that took the launch stream while recording and launches inside
/// a later eager section holds the capture stream, which is no longer
/// capturing, so its launch runs at once. It must still run after everything
/// recorded before it and before everything issued after it.
#[test]
fn a_stale_capture_stream_launch_in_an_eager_section_stays_in_order() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let mut z = dev.alloc_zeros::<f32>(N)?;
    let mut w = dev.alloc_zeros::<f32>(N)?;
    for round in 0..8 {
        let capture = dev.begin_wave_capture()?;
        dev.record_launches()?;
        let stale = dev.cuda_stream();
        // A long recorded chain ahead of the stale launch.
        for _ in 0..64 {
            launch(&f, &dev.cuda_stream(), &x, &mut y, 2.0, round as f32)?;
        }
        {
            let _eager = dev.pause_capture()?;
            launch(&f, &stale, &y, &mut z, 1.0, 1.0)?;
        }
        launch(&f, &dev.cuda_stream(), &z, &mut w, 3.0, 0.0)?;
        capture.finish()?;
        let want_z: Vec<f32> = ramp(1.0)
            .iter()
            .map(|v| v * 2.0 + round as f32 + 1.0)
            .collect();
        let want_w: Vec<f32> = want_z.iter().map(|v| v * 3.0).collect();
        assert_eq!(
            dev.memcpy_dtov(&z)?,
            want_z,
            "round {round}: after the recorded work"
        );
        assert_eq!(
            dev.memcpy_dtov(&w)?,
            want_w,
            "round {round}: before the work after it"
        );
    }
    Ok(())
}

/// Other threads keep the compute stream while one thread records: a launch
/// from another thread during the capture executes at once, in its own order.
#[test]
fn only_the_capturing_thread_is_redirected() -> Result<()> {
    let _wave = wave_lock();
    let device = crate::Device::new_cuda(0)?;
    let crate::Device::Cuda(dev) = &device else {
        unreachable!("asked for a CUDA device")
    };
    let capture = dev.begin_wave_capture()?;
    // Held: nothing is redirected until the wave starts recording.
    assert_eq!(
        dev.cuda_stream().cu_stream(),
        dev.compute_stream().cu_stream()
    );
    dev.record_launches()?;
    let here = dev.cuda_stream();
    let there = {
        let dev = dev.clone();
        std::thread::spawn(move || dev.cuda_stream())
            .join()
            .expect("the other thread panicked")
    };
    capture.finish()?;
    assert_ne!(here.cu_stream(), dev.compute_stream().cu_stream());
    assert_eq!(there.cu_stream(), dev.compute_stream().cu_stream());
    assert_eq!(
        dev.cuda_stream().cu_stream(),
        dev.compute_stream().cu_stream()
    );
    Ok(())
}

/// Dropping a session unfinished ends the capture and leaves the stream able
/// to record the next one.
#[test]
fn a_dropped_session_leaves_the_stream_usable() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let compute = ComputeStream::of(&dev);
    let mut cap = CaptureStream::new(&dev)?;
    {
        let s = CaptureSession::begin(&mut cap)?;
        launch(&f, s.stream(), &x, &mut y, 9.0, 9.0)?;
    }
    let exec = {
        let s = CaptureSession::begin(&mut cap)?;
        launch(&f, s.stream(), &x, &mut y, 1.0, 2.0)?;
        s.finish(&compute, 1)?
    };
    exec.launch(&compute)?;
    let want: Vec<f32> = ramp(1.0).iter().map(|v| v + 2.0).collect();
    assert_eq!(dev.memcpy_dtov(&y)?, want);
    Ok(())
}

/// An in-place cast inside a recording wave keeps the buffer it replaces until
/// the segment that reads it has been launched: memory allocated after it, and
/// written in the same segment, must not land on the bytes the cast still has
/// to read.
#[test]
fn an_in_place_cast_while_recording_reads_back_what_it_reads_eagerly() -> Result<()> {
    use crate::{DType, Tensor};
    let _wave = wave_lock();
    let device = crate::Device::new_cuda(0)?;
    let crate::Device::Cuda(dev) = &device else {
        unreachable!("asked for a CUDA device")
    };
    let cast = |device: &crate::Device| -> Result<Vec<f32>> {
        let mut t = (Tensor::arange(0f32, N as f32, device)? * 0.25)?;
        t.to_dtype_mut(DType::F16)?;
        // Same size as the buffer the cast gave up, written before the cast's
        // result is read.
        let other = (Tensor::ones(N, DType::F32, device)? * 7.0)?;
        let back = t.to_dtype(DType::F32)?;
        drop(other);
        back.to_vec1::<f32>()
    };
    let eager = cast(&device)?;
    for wave in 0..4 {
        let capture = dev.begin_wave_capture()?;
        dev.record_launches()?;
        let recorded = cast(&device);
        capture.finish()?;
        assert_eq!(recorded?, eager, "wave {wave}");
    }
    Ok(())
}

/// Sets its flag when dropped — a stand-in for device memory being freed.
struct Freed(Arc<AtomicBool>);

impl Drop for Freed {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Memory another thread releases while a segment records is held until that
/// segment has been launched: the last reference to something a recorded
/// launch reads may be dropped anywhere, and a free issued at once would
/// queue on the compute stream ahead of the segment.
#[test]
fn a_release_from_another_thread_waits_for_the_recording_segment() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let freed = Arc::new(AtomicBool::new(false));
    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    launch(&f, &dev.cuda_stream(), &x, &mut y, 1.0, 0.0)?;
    {
        let dev = dev.clone();
        let item = Freed(freed.clone());
        std::thread::spawn(move || dev.retire(item))
            .join()
            .expect("the releasing thread panicked");
    }
    assert!(
        !freed.load(Ordering::SeqCst),
        "released while the segment that may read it was recording"
    );
    {
        let _eager = dev.pause_capture()?;
        assert!(
            freed.load(Ordering::SeqCst),
            "released once the segment was launched"
        );
    }
    capture.finish()?;
    Ok(())
}

/// The segment this thread is recording, by `(wave, ordinal)`: the ordinal
/// counts the segments the wave has launched, a flush moves to the next, a
/// paused wave records none, and another wave carries another serial.
#[test]
fn the_recording_segment_advances_with_each_launch() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    assert_eq!(dev.recording_segment(), None, "no wave open");
    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    launch(&f, &dev.cuda_stream(), &x, &mut y, 1.0, 0.0)?;
    let (wave, first) = dev.recording_segment().expect("recording");
    assert_eq!(first, 0);
    dev.flush_launches()?;
    launch(&f, &dev.cuda_stream(), &x, &mut y, 1.0, 1.0)?;
    assert_eq!(
        dev.recording_segment(),
        Some((wave, 1)),
        "a flush launched segment 0"
    );
    {
        let _eager = dev.pause_capture()?;
        assert_eq!(
            dev.recording_segment(),
            None,
            "a paused wave records nothing"
        );
    }
    launch(&f, &dev.cuda_stream(), &x, &mut y, 1.0, 2.0)?;
    assert_eq!(
        dev.recording_segment(),
        Some((wave, 2)),
        "the pause launched segment 1"
    );
    capture.finish()?;
    assert_eq!(dev.recording_segment(), None, "the wave closed");
    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    launch(&f, &dev.cuda_stream(), &x, &mut y, 1.0, 3.0)?;
    let (next, ordinal) = dev.recording_segment().expect("recording");
    assert_ne!(next, wave, "a new wave has its own serial");
    assert_eq!(ordinal, 0);
    capture.finish()?;
    dev.synchronize()?;
    let want: Vec<f32> = ramp(1.0).iter().map(|v| v + 3.0).collect();
    assert_eq!(dev.memcpy_dtov(&y)?, want);
    Ok(())
}

/// **An abandoned wave still runs what it recorded.** A forward that fails
/// mid-wave drops its capture unfinished, and parties outside the wave may
/// already wait on launches the open segment holds — an MoE invocation's
/// summary word is announced to the stager before its bucketize is recorded.
/// So the drop launches the segment rather than discarding it.
#[test]
fn an_abandoned_wave_launches_its_recording_segment() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    launch(&f, &dev.cuda_stream(), &x, &mut y, 2.0, 1.0)?;
    drop(capture);
    dev.synchronize()?;
    let want: Vec<f32> = ramp(1.0).iter().map(|v| 2.0 * v + 1.0).collect();
    assert_eq!(dev.memcpy_dtov(&y)?, want);
    // The hub is free for the next wave.
    let capture = dev.begin_wave_capture()?;
    capture.finish()?;
    Ok(())
}

/// **Another thread's eager staging never shares the wave's scratch.** A
/// recording thread's segment writes the scratch through its recorded uploads
/// and is launched whenever that thread next ends a segment — which can fall
/// between a second thread's eager copy into the scratch and the kernel that
/// reads it. The kernel then reads the segment's bytes as its own: measured in
/// zend as the persistence thread's record fill taking a quantized activation
/// for its descriptor table and writing a KV record to an unmapped address.
#[test]
fn another_threads_eager_staging_is_not_the_waves_scratch() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    let (wave_scratch, _) = dev.with_staged_upload(&[1u8; 16], |at| Ok((at, ())))?;
    let foreign_scratch = {
        let dev = dev.clone();
        std::thread::spawn(move || -> Result<u64> {
            dev.cuda_context().bind_to_thread().w()?;
            dev.with_staged_upload(&[2u8; 16], Ok)
        })
        .join()
        .expect("the eager thread panicked")?
    };
    capture.finish()?;
    assert_ne!(
        wave_scratch, foreign_scratch,
        "an eager copy from another thread landed in the scratch a recorded segment writes"
    );
    Ok(())
}

/// **A full info ring does not rewind under another thread's recording
/// segment.** That segment's launches read layout tables from the ring and are
/// not issued yet; a rewind from a second thread would overwrite those words
/// before they run. The ring moves to a fresh buffer instead and retires the
/// old one until the segment has launched.
#[test]
fn an_info_ring_wrap_from_another_thread_retires_instead_of_rewinding() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    launch(&f, &dev.cuda_stream(), &x, &mut y, 1.0, 0.0)?;
    let before = dev.capture_hub().retired_len();
    {
        let dev = dev.clone();
        // Distinct tables a sixteenth of the ring wide, enough to wrap it at
        // least once from wherever earlier tests left it.
        std::thread::spawn(move || -> Result<()> {
            dev.cuda_context().bind_to_thread().w()?;
            let words = crate::cuda_backend::info_ring::RING_WORDS / 16;
            for i in 0..17usize {
                let table: Vec<usize> = (0..words).map(|w| w ^ (i << 32)).collect();
                dev.info_table(&table)?;
            }
            Ok(())
        })
        .join()
        .expect("the uploading thread panicked")?;
    }
    assert!(
        dev.capture_hub().retired_len() > before,
        "the ring rewound or freed its buffer under a recording segment"
    );
    {
        let _eager = dev.pause_capture()?;
        assert_eq!(dev.capture_hub().retired_len(), 0, "released once launched");
    }
    capture.finish()?;
    Ok(())
}

/// Retired memory is dropped outside the hub's lock, so an item whose own drop
/// retires more does not deadlock the wave.
#[test]
fn a_retired_item_may_retire_from_its_own_drop() -> Result<()> {
    struct Nested(CudaDevice, Arc<AtomicBool>);
    impl Drop for Nested {
        fn drop(&mut self) {
            self.0.retire(Freed(self.1.clone()));
        }
    }
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let freed = Arc::new(AtomicBool::new(false));
    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    launch(&f, &dev.cuda_stream(), &x, &mut y, 1.0, 0.0)?;
    dev.retire(Nested(dev.clone(), freed.clone()));
    capture.finish()?;
    assert!(freed.load(Ordering::SeqCst), "the nested release ran");
    Ok(())
}

/// **Pinned staging taken and freed mid-segment pauses the wave around it.**
/// The driver refuses to pin or unpin host pages on a recording thread, and the
/// refusal ends the capture — measured in motherd, a prefill wave that took a
/// staging buffer mid-segment lost the segment and faulted the next wave step.
/// The staging calls hold no device, so they pause through the wave this
/// thread has open: the segment recorded before each runs first, recording
/// resumes, and every launch lands in issue order.
#[test]
fn pinned_staging_inside_a_recording_pauses_the_wave() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    launch(&f, &dev.cuda_stream(), &x, &mut y, 2.0, 0.0)?;
    let (wave, first) = dev.recording_segment().expect("recording");
    assert_eq!(first, 0);
    let staging = PinnedBuf::alloc_owned(4096)?;
    assert_eq!(
        dev.recording_segment(),
        Some((wave, 1)),
        "the allocation launched segment 0 and recording resumed"
    );
    launch(&f, &dev.cuda_stream(), &x, &mut y, 1.0, 1.0)?;
    drop(staging);
    assert_eq!(
        dev.recording_segment(),
        Some((wave, 2)),
        "the free launched segment 1 and recording resumed"
    );
    launch(&f, &dev.cuda_stream(), &x, &mut y, 1.0, 3.0)?;
    capture.finish()?;
    dev.synchronize()?;
    let want: Vec<f32> = ramp(1.0).iter().map(|v| v + 3.0).collect();
    assert_eq!(dev.memcpy_dtov(&y)?, want);
    // With no wave open the staging calls go straight to the driver.
    assert_eq!(PinnedBuf::alloc_owned(4096)?.len(), 4096);
    assert_eq!(dev.recording_segment(), None);
    Ok(())
}

/// A pause guard that outlives its wave leaves the next wave alone: that wave
/// still records, and still pauses and resumes in step.
#[test]
fn a_pause_guard_outliving_its_wave_leaves_the_next_alone() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let first = dev.begin_wave_capture()?;
    dev.record_launches()?;
    launch(&f, &dev.cuda_stream(), &x, &mut y, 1.0, 0.0)?;
    let stale = dev.pause_capture()?;
    first.finish()?;

    let second = dev.begin_wave_capture()?;
    dev.record_launches()?;
    drop(stale);
    let recording =
        |dev: &CudaDevice| dev.cuda_stream().cu_stream() != dev.compute_stream().cu_stream();
    assert!(
        recording(&dev),
        "the second wave records past the stale guard"
    );
    {
        let _eager = dev.pause_capture()?;
        assert!(!recording(&dev), "a pause of the second wave pauses it");
    }
    assert!(
        recording(&dev),
        "the second wave resumes after its own pause"
    );
    launch(&f, &dev.cuda_stream(), &x, &mut y, 2.0, 1.0)?;
    second.finish()?;
    let want: Vec<f32> = ramp(1.0).iter().map(|v| v * 2.0 + 1.0).collect();
    assert_eq!(dev.memcpy_dtov(&y)?, want);
    Ok(())
}

/// Two wave shapes whose segment holds the same number of nodes — one kernel,
/// one memset — each keep their own executable: once both have run, every
/// later wave folds in place rather than re-instantiating over the other.
#[test]
fn two_shapes_of_one_node_count_each_keep_their_executable() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let mut before = dev.capture_stats();
    for wave in 0..6u32 {
        if wave == 2 {
            before = dev.capture_stats();
        }
        let capture = dev.begin_wave_capture()?;
        dev.record_launches()?;
        if wave % 2 == 0 {
            launch(&f, &dev.cuda_stream(), &x, &mut y, 1.0, wave as f32)?;
        } else {
            dev.cuda_stream().memset_zeros(&mut y).w()?;
        }
        capture.finish()?;
        let want: Vec<f32> = if wave % 2 == 0 {
            ramp(1.0).iter().map(|v| v + wave as f32).collect()
        } else {
            vec![0.0; N]
        };
        assert_eq!(dev.memcpy_dtov(&y)?, want, "wave {wave}");
    }
    let after = dev.capture_stats();
    assert_eq!(after.instantiated - before.instantiated, 0, "{after:?}");
    assert_eq!(after.updated - before.updated, 4, "{after:?}");
    Ok(())
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

/// **A budget nothing fits locks the library at once, and a locked library
/// runs what it does not hold as issued.** The first admission does not fit:
/// nothing is evicted to make room, the executable is dropped once launched,
/// and with nothing admitted there is nothing to trial. The next wave's segment
/// is not recorded at all. Every wave's result is what the eager forward
/// would have produced.
#[test]
fn a_budget_nothing_fits_locks_and_runs_eagerly() -> Result<()> {
    use super::budget::graph_budget_bytes;
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    dev.capture_hub().reset_library(1);
    let want = |add: f32| -> Vec<f32> { ramp(1.0).iter().map(|v| v + add).collect() };

    let s0 = dev.capture_stats();
    assert_eq!(add_wave(&dev, &f, &x, &mut y, 3, 1.0)?, want(1.0));
    let s1 = dev.capture_stats();
    assert_eq!(
        (s1.instantiated - s0.instantiated, s1.execs_held, s1.phase),
        (1, 0, GraphPhase::Lock),
        "the first admission does not fit: nothing is kept, and the library locks"
    );

    assert_eq!(add_wave(&dev, &f, &x, &mut y, 3, 2.0)?, want(2.0));
    let s2 = dev.capture_stats();
    assert_eq!(
        (
            s2.eager_segments - s1.eager_segments,
            s2.segments - s1.segments,
            s2.instantiated - s1.instantiated,
        ),
        (1, 0, 0),
        "locked, a shape it does not hold runs unrecorded"
    );

    // Its ordinal rechecks once it has run eagerly `RECHECK_EVERY` times in a
    // row: recorded, not held, so run once and not kept.
    for wave in 0..super::hub::RECHECK_EVERY {
        let add = 3.0 + wave as f32;
        assert_eq!(add_wave(&dev, &f, &x, &mut y, 3, add)?, want(add));
    }
    let s3 = dev.capture_stats();
    assert_eq!(
        (
            s3.eager_segments - s1.eager_segments,
            s3.transient - s1.transient,
            s3.execs_held,
        ),
        (u64::from(super::hub::RECHECK_EVERY), 1, 0),
        "{} eager runs, then one recorded to recheck its shape",
        super::hub::RECHECK_EVERY
    );

    dev.capture_hub().reset_library(graph_budget_bytes()?);
    Ok(())
}

/// **A lock with room reopens for traffic it never learned.** The library
/// locks holding nothing; waves then cycle four topologies it does not hold.
/// The ordinal's rechecks meet each of them again and again, and once the lock
/// has stood its minimum, it reopens. Every wave's result is still what the
/// eager forward would have produced.
#[test]
fn a_lock_reopens_for_shapes_it_keeps_meeting() -> Result<()> {
    use super::budget::graph_budget_bytes;
    use super::regime::MIN_LOCK_WAVES;
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    dev.capture_hub().reset_library(1);
    let want = |add: f32| -> Vec<f32> { ramp(1.0).iter().map(|v| v + add).collect() };
    assert_eq!(add_wave(&dev, &f, &x, &mut y, 3, 1.0)?, want(1.0));
    assert_eq!(dev.capture_stats().phase, GraphPhase::Lock);

    let limit = MIN_LOCK_WAVES as u32 + 40 * super::hub::RECHECK_EVERY;
    let mut waves = 0u32;
    while dev.capture_stats().reopened == 0 {
        assert!(waves < limit, "never reopened: {:?}", dev.capture_stats());
        let add = (waves % 5) as f32;
        let launches = 3 + (waves % 4) as usize;
        assert_eq!(
            add_wave(&dev, &f, &x, &mut y, launches, add)?,
            want(add),
            "wave {waves}"
        );
        waves += 1;
    }
    assert!(
        u64::from(waves) >= MIN_LOCK_WAVES,
        "reopened after {waves} waves, before the lock's minimum"
    );
    dev.capture_hub().reset_library(graph_budget_bytes()?);
    Ok(())
}

/// **The library learns, trials and locks within a couple of hundred waves.**
/// One shape: the learn admits it and goes quiet, the trial runs it graphed and
/// eager by turns, and the verdict either proves it (the next learn finds
/// nothing new, and locks) or bans it (and locks). Whichever, the library ends
/// locked holding exactly what was proven, the trial ran eager halves, and every
/// wave's result is what the eager forward would have produced.
#[test]
fn the_library_learns_trials_and_locks_quickly() -> Result<()> {
    use super::budget::graph_budget_bytes;
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    dev.capture_hub().reset_library(graph_budget_bytes()?);
    let s0 = dev.capture_stats();
    let mut waves = 0u32;
    while dev.capture_stats().phase != GraphPhase::Lock {
        assert!(
            waves < 300,
            "still {:?} after {waves} waves",
            dev.capture_stats()
        );
        let add = (waves % 7) as f32;
        let want: Vec<f32> = ramp(1.0).iter().map(|v| v + add).collect();
        assert_eq!(
            add_wave(&dev, &f, &x, &mut y, 3, add)?,
            want,
            "wave {waves}"
        );
        waves += 1;
    }
    let s = dev.capture_stats();
    assert!(
        s.eager_segments > s0.eager_segments,
        "the trial ran eager halves"
    );
    assert_eq!(s.banned + s.proven, 1, "one shape, one verdict: {s:?}");
    assert_eq!(
        s.execs_held, s.proven,
        "what is held is what was proven: {s:?}"
    );
    dev.capture_hub().reset_library(graph_budget_bytes()?);
    Ok(())
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

/// **Past the learn, a held topology at a shape it does not hold runs once,
/// and the held executable keeps its own shape.** The same launch at another
/// grid is one topology. Under `Never` — the trial's and the lock's mode — it
/// is instantiated for its launch and computes its own extent, nothing in the
/// slot is rewritten, and the held shape still folds in place after it.
#[test]
fn past_the_learn_a_new_shape_runs_once_and_reshapes_nothing() -> Result<()> {
    use super::slot::{Admit, ExecSlot, Folded};
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let stream = dev.cuda_context().new_stream().w()?;
    let compute = ComputeStream::of(&dev);
    let mut slot = ExecSlot::default();
    let half = N / 2;
    // `y = x + head` over the first `n` elements, `x + tail` past them.
    let want = |n: usize, head: f32, tail: f32| -> Vec<f32> {
        ramp(1.0)
            .iter()
            .enumerate()
            .map(|(i, v)| v + if i < n { head } else { tail })
            .collect()
    };

    let (t, a) = captured_n(&f, &stream, &x, &mut y, N, 1.0)?;
    let (once, how, _) = slot.fold(t, a, &compute, 1, Admit::Keep)?;
    assert!(once.is_none());
    assert_eq!(how, Folded::Instantiated);
    slot.front().expect("kept").launch(&compute)?;
    assert_eq!(dev.memcpy_dtov(&y)?, want(N, 1.0, 1.0));

    let (t, a) = captured_n(&f, &stream, &x, &mut y, half, 2.0)?;
    let (once, how, _) = slot.fold(t, a, &compute, 1, Admit::Never)?;
    assert_eq!(how, Folded::Transient, "never: run once, kept nothing");
    once.expect("instantiated for its launch")
        .launch(&compute)?;
    assert_eq!(dev.memcpy_dtov(&y)?, want(half, 2.0, 1.0));

    let (t, a) = captured_n(&f, &stream, &x, &mut y, N, 3.0)?;
    let (once, how, _) = slot.fold(t, a, &compute, 1, Admit::Never)?;
    assert!(once.is_none());
    assert_eq!(how, Folded::InPlace, "the held shape was not rewritten");
    slot.front().expect("held").launch(&compute)?;
    assert_eq!(dev.memcpy_dtov(&y)?, want(N, 3.0, 3.0));
    assert_eq!(slot.held().0, 1);
    Ok(())
}

/// **A bracketed segment yields its device time.** A start and an end recorded
/// around launches on the compute stream drain, once the end has fired, as one
/// sample of the key and mode they were recorded for. The trial is judged on
/// nothing else: events made without timing drained as no sample at all.
#[test]
fn a_timed_segment_yields_one_sample() -> Result<()> {
    use super::regime::Mode;
    use super::timing::SegmentTimer;
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let stream = dev.cuda_stream();
    let mut timer = SegmentTimer::default();
    let start = timer.start(&stream)?;
    for _ in 0..8 {
        launch(&f, &stream, &x, &mut y, 1.0, 2.0)?;
    }
    timer.end(&stream, start, 7, Mode::Eager)?;
    stream.synchronize().w()?;
    let mut samples = Vec::new();
    timer.drain(|key, mode, took| samples.push((key, mode, took)))?;
    assert_eq!(samples.len(), 1, "one pair, one sample: {samples:?}");
    let (key, mode, _took) = samples[0];
    assert_eq!((key, mode), (7, Mode::Eager));
    Ok(())
}

/// **A stream taken while recording and launched in a later eager segment stays
/// in issue order.** An eager segment runs on the capture stream — the stream a
/// recording segment hands out — so a launcher holding it from before a flush
/// lands in order with everything the eager segment issues around it. On the
/// compute stream instead, the held stream ran at once, unordered against the
/// eager launches ahead of it and the ones behind it: on the Flash-Next gate a
/// trial or a locked library, mixing recorded and eager segments in one wave,
/// read half-written activations and sampled from NaN logits.
///
/// The library is given a budget nothing fits, so it locks on the first wave
/// and every segment after runs eagerly — but each ordinal is recorded once
/// every `RECHECK_EVERY + 1` runs, so across the rounds a recorded first
/// segment hands its stream over a flush to an eager second one, and eager
/// segments hand theirs to eager ones.
#[test]
fn a_held_stream_in_an_eager_segment_stays_in_order() -> Result<()> {
    use super::budget::graph_budget_bytes;
    use super::hub::RECHECK_EVERY;
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let mut z = dev.alloc_zeros::<f32>(N)?;
    let mut w = dev.alloc_zeros::<f32>(N)?;
    let mut v = dev.alloc_zeros::<f32>(N)?;
    dev.capture_hub().reset_library(1);
    let rounds = 3 * (RECHECK_EVERY + 1);
    let mut s0 = None;
    for round in 0..rounds {
        if round == 1 {
            assert_eq!(dev.capture_stats().phase, GraphPhase::Lock);
            s0 = Some(dev.capture_stats());
        }
        let add = round as f32;
        let capture = dev.begin_wave_capture()?;
        dev.record_launches()?;
        let held = dev.cuda_stream();
        for _ in 0..3 {
            launch(&f, &dev.cuda_stream(), &x, &mut y, 1.0, add)?;
        }
        dev.flush_launches()?;
        assert!(
            Arc::ptr_eq(&held, &dev.cuda_stream()),
            "round {round}: the eager segment launches on the stream the wave handed out"
        );
        // A long eager run ahead of the held launch, and one behind it.
        for _ in 0..64 {
            launch(&f, &dev.cuda_stream(), &y, &mut z, 1.0, 1.0)?;
        }
        launch(&f, &held, &z, &mut w, 3.0, 0.0)?;
        launch(&f, &dev.cuda_stream(), &w, &mut v, 1.0, 5.0)?;
        capture.finish()?;
        let want: Vec<f32> = ramp(1.0)
            .iter()
            .map(|x| (x + add + 1.0) * 3.0 + 5.0)
            .collect();
        assert_eq!(dev.memcpy_dtov(&v)?, want, "round {round}");
    }
    let (s0, s) = (s0.expect("past round 1"), dev.capture_stats());
    assert!(
        s.segments > s0.segments && s.eager_segments >= s0.eager_segments + u64::from(rounds),
        "rechecks recorded some segments, and most ran eagerly: {s:?}"
    );
    dev.capture_hub().reset_library(graph_budget_bytes()?);
    Ok(())
}

/// What one instantiated executable costs the device, by node count: the
/// driver's free memory read across `EXECS` instantiations of one captured
/// graph, at each size. Every byte of it is outside the reservation, and the
/// library keeps up to `MAX_TOPOLOGIES × EXECS_PER_SLOT` of them.
#[test]
#[ignore = "probe: cargo test -p candle-core --features cuda --lib graph::tests::device_bytes_per_executable -- --ignored --nocapture --test-threads=1"]
fn device_bytes_per_executable() -> Result<()> {
    use cudarc::driver::{result, sys};
    const EXECS: usize = 64;
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let stream = dev.cuda_context().new_stream().w()?;
    let free_now = || -> Result<u64> {
        dev.cuda_stream().synchronize().w()?;
        let (free, _total) = result::mem_get_info().w()?;
        Ok(free as u64)
    };
    for nodes in [1usize, 100, 400, 800] {
        // SAFETY: a created stream this test owns, not capturing.
        unsafe { super::capture::begin(&stream)? };
        for i in 0..nodes {
            launch(&f, &stream, &x, &mut y, 1.0, i as f32)?;
        }
        // SAFETY: the capture begun above, on this thread.
        let template = unsafe { super::capture::end(&stream)? };
        let before = free_now()?;
        let mut execs = Vec::with_capacity(EXECS);
        for _ in 0..EXECS {
            let mut exec: sys::CUgraphExec = std::ptr::null_mut();
            // SAFETY: a valid graph; default flags.
            unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, template, 0).result() }.w()?;
            // One launch, as the hub's executables have all been launched.
            // SAFETY: an executable instantiated above, on a stream this test owns.
            unsafe { sys::cuGraphLaunch(exec, stream.cu_stream()).result() }.w()?;
            execs.push(exec);
        }
        stream.synchronize().w()?;
        let after = free_now()?;
        let per_exec = before.saturating_sub(after) / EXECS as u64;
        println!(
            "{nodes:>4} nodes: {per_exec:>9} B per executable ({:.0} B per node), \
             {EXECS} took {} KiB",
            per_exec as f64 / nodes as f64,
            before.saturating_sub(after) >> 10
        );
        for exec in execs {
            // SAFETY: each executable instantiated above, destroyed once.
            unsafe { sys::cuGraphExecDestroy(exec).result() }.w()?;
        }
        // SAFETY: the template captured above, destroyed once.
        unsafe { result::graph::destroy(template) }.w()?;
    }
    Ok(())
}
