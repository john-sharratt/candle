//! Tests of what CUDA graph capture records and replays, and of what the calls around it do to a capture
//! open on another thread.
//!
//! A capture executes nothing, and a launch on the default stream is not recorded, so only launchers that take
//! their stream from the caller can sit inside a captured region. A recorded graph replays bit-identical to the
//! same launches run eagerly. A recapture with new scalars or addresses folds into the existing executable in
//! place, while a different topology is instantiated afresh. A host copy inside a capture is refused, since its
//! buffer is gone by replay, while a device copy is recorded in order. The cost probes print their timings and
//! assert only that the replays agree.

use super::*;

/// A captured graph replayed into the null stream reads what null-stream work
/// wrote before it, and what it writes is what null-stream work after it reads
/// — the ordering §4.2 rests on. Per-wave data written to the same addresses
/// between replays is picked up, because the graph bakes addresses, not
/// contents.
#[test]
pub(super) fn replay_on_the_null_stream_orders_against_null_stream_work() -> Result<()> {
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
pub(super) fn a_capture_survives_null_stream_work_from_another_thread() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let stop = Arc::new(AtomicBool::new(false));
    let rounds = Arc::new(AtomicUsize::new(0));
    let worker = {
        let (dev, f, stop, rounds) = (dev.clone(), f.clone(), stop.clone(), rounds.clone());
        std::thread::spawn(move || -> Result<()> {
            admit_this_thread();
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
pub(super) fn a_default_stream_launch_is_not_recorded_and_the_capture_is_refused() -> Result<()> {
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
pub(super) fn migrated_launchers_record_on_the_capture_stream() -> Result<()> {
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
pub(super) fn a_recapture_with_new_scalars_and_addresses_updates_in_place() -> Result<()> {
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
pub(super) fn a_recapture_with_another_kernel_at_a_node_folds_in_place() -> Result<()> {
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
pub(super) fn a_host_copy_inside_a_capture_is_refused_and_a_device_copy_is_kept() -> Result<()> {
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
pub(super) fn per_wave_capture_cost() -> Result<()> {
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
pub(super) fn exec_update_cost_by_change() -> Result<()> {
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
        unsafe { super::super::capture::begin(&stream)? };
        for &add in adds {
            launch(&f, &stream, &x, y, 1.0, add)?;
        }
        // SAFETY: the capture begun above, on this thread.
        unsafe { super::super::capture::end(&stream) }
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
pub(super) fn null_stream_calls_during_capture() -> Result<()> {
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
pub(super) fn other_threads_calls_during_capture() -> Result<()> {
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
                cudarc::cublas::CudaBlas::new((*d.cuda_stream()).clone())
                    .map(drop)
                    .map_err(crate::Error::wrap)
            }),
        ),
        (
            "curand generator",
            Box::new(|d| {
                cudarc::curand::CudaRng::new(0, (*d.cuda_stream()).clone())
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
            sc.spawn(|| {
                admit_this_thread();
                probe(&other).map_err(|e| e.to_string())
            })
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

/// Dropping a session unfinished ends the capture and leaves the stream able
/// to record the next one.
#[test]
pub(super) fn a_dropped_session_leaves_the_stream_usable() -> Result<()> {
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
pub(super) fn an_in_place_cast_while_recording_reads_back_what_it_reads_eagerly() -> Result<()> {
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
pub(super) fn a_held_stream_in_an_eager_segment_stays_in_order() -> Result<()> {
    use super::super::budget::graph_budget_bytes;
    use super::super::hub::RECHECK_EVERY;
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
pub(super) fn device_bytes_per_executable() -> Result<()> {
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
        unsafe { super::super::capture::begin(&stream)? };
        for i in 0..nodes {
            launch(&f, &stream, &x, &mut y, 1.0, i as f32)?;
        }
        // SAFETY: the capture begun above, on this thread.
        let template = unsafe { super::super::capture::end(&stream)? };
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

/// **What a fence wait does to a capture open on the waiting thread.** Held
/// ground reads its fences — events recorded on the compute stream outside any
/// capture — from inside a forward: a query on every take, a wait when the
/// take finds no room otherwise. Each is tried here inside a `ThreadLocal`
/// capture on a stream of the test's own, and the capture's status read after.
#[test]
pub(super) fn a_fence_wait_inside_a_capture() -> Result<()> {
    use cudarc::driver::sys;
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let stream = dev.cuda_context().new_stream().w()?;
    let fence = dev.cuda_stream().record_event(None).w()?;
    dev.cuda_stream().synchronize().w()?;
    let status = |s: &CudaStream| -> Result<sys::CUstreamCaptureStatus> {
        let mut st = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        // SAFETY: a live stream and a status to write.
        unsafe { sys::cuStreamIsCapturing(s.cu_stream(), &mut st) }
            .result()
            .w()?;
        Ok(st)
    };
    let mut outcome = Vec::new();
    for relaxed in [false, true] {
        for wait in [false, true] {
            // SAFETY: a created stream this test owns, not capturing.
            unsafe { super::super::capture::begin(&stream)? };
            launch(&f, &stream, &x, &mut y, 1.0, 1.0)?;
            let call = || {
                // SAFETY: an event recorded above, outside any capture.
                unsafe {
                    if wait {
                        sys::cuEventSynchronize(fence.cu_event())
                    } else {
                        sys::cuEventQuery(fence.cu_event())
                    }
                }
            };
            let called = if relaxed {
                super::super::capture::relaxed(call)
            } else {
                call()
            };
            let after = status(&stream)?;
            // SAFETY: the capture begun above, on this thread.
            let ended = unsafe { super::super::capture::end(&stream) };
            if let Ok(graph) = ended {
                // SAFETY: a graph this test captured, destroyed once.
                unsafe { cudarc::driver::result::graph::destroy(graph) }.w()?;
            }
            outcome.push((relaxed, wait, called, after));
        }
    }
    use sys::CUresult::{CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED as REFUSED, CUDA_SUCCESS as FIRED};
    use sys::CUstreamCaptureStatus::{
        CU_STREAM_CAPTURE_STATUS_ACTIVE as ACTIVE, CU_STREAM_CAPTURE_STATUS_INVALIDATED as LOST,
    };
    assert_eq!(
        outcome,
        vec![
            (false, false, REFUSED, LOST),
            (false, true, REFUSED, LOST),
            (true, false, FIRED, ACTIVE),
            (true, true, FIRED, ACTIVE),
        ]
    );
    Ok(())
}
