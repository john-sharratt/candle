//! Tests of wave-level recording: that a recorded wave computes what eager execution does, and that work from other
//! threads stays in issue order with it.
//!
//! A wave captured with changing scalars reads back exactly what it reads back eagerly, and alternating shapes each
//! keep their own executable. Uploads, scratch use and held streams recorded during a wave stay in issue order with no
//! cut to the segment. Another thread's lease waits for a recording segment and is granted behind its launch, so the
//! other thread reads what the segment wrote. Pinned staging inside a recording pauses the wave around it, since the
//! driver refuses to pin pages on a recording thread.

use super::*;

/// The chain run under a wave capture reads back exactly what it reads back
/// eagerly, wave after wave with changing scalars — the capture changes when
/// launches reach the driver, never what they compute.
#[test]
pub(super) fn a_wave_capture_is_bit_identical_to_eager_execution() -> Result<()> {
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
pub(super) fn alternating_wave_shapes_each_fold_into_their_own_executable() -> Result<()> {
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
pub(super) fn recurring_shapes_of_one_topology_each_keep_their_executable() -> Result<()> {
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
pub(super) fn an_upload_while_recording_lands_in_issue_order_without_a_cut() -> Result<()> {
    use cudarc::driver::DevicePtr;
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let compute = dev.stream_unleased().clone();
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
pub(super) fn staged_uploads_while_recording_keep_their_order_without_a_cut() -> Result<()> {
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
pub(super) fn staging_scratch_while_recording_keeps_its_order_without_a_cut() -> Result<()> {
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

/// A launcher that took the stream while recording and launches inside a later
/// eager section runs after everything recorded before it and before
/// everything issued after it: there is one stream, so the stream it holds is
/// the one everything else is on.
#[test]
pub(super) fn a_stream_held_into_an_eager_section_stays_in_order() -> Result<()> {
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

/// **Another thread's lease keeps the wave from recording, and its work lands
/// in issue order.** A segment that would begin while another thread holds the
/// stream runs as issued — the capture would take that thread's launches in —
/// and the next one, once the lease is back, records. The other thread's launch
/// is ordered against the wave's by the one stream alone.
#[test]
pub(super) fn another_threads_lease_runs_the_segment_as_issued() -> Result<()> {
    use std::sync::mpsc;
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let mut z = dev.alloc_zeros::<f32>(N)?;
    let before = dev.capture_stats();

    let (leased, leased_rx) = mpsc::channel();
    let (give_back, give_back_rx) = mpsc::channel::<()>();
    let holder = {
        let (dev, f) = (dev.clone(), f.clone());
        let x = dev.memcpy_stod(&ramp(1.0))?;
        std::thread::spawn(move || -> Result<CudaSlice<f32>> {
            admit_this_thread();
            let stream = dev.cuda_stream();
            let mut w = dev.alloc_zeros::<f32>(N)?;
            launch(&f, &stream, &x, &mut w, 4.0, 0.0)?;
            leased.send(()).expect("the test is waiting");
            give_back_rx.recv().expect("the test gives the word");
            drop(stream);
            Ok(w)
        })
    };
    leased_rx.recv().expect("the holder took its lease");

    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    assert!(
        dev.recording_segment().is_none(),
        "another thread holds the stream: the segment runs as issued"
    );
    launch(&f, &dev.cuda_stream(), &x, &mut y, 2.0, 0.0)?;
    give_back.send(()).expect("the holder is waiting");
    let w = holder.join().expect("the holder panicked")?;
    dev.flush_launches()?;
    assert!(
        dev.recording_segment().is_some(),
        "the lease is back: the next segment records"
    );
    launch(&f, &dev.cuda_stream(), &y, &mut z, 3.0, 1.0)?;
    capture.finish()?;

    let after = dev.capture_stats();
    assert_eq!(after.leased_eager - before.leased_eager, 1);
    assert_eq!(after.segments - before.segments, 1, "one segment recorded");
    let want: Vec<f32> = ramp(1.0).iter().map(|v| v * 6.0 + 1.0).collect();
    assert_eq!(dev.memcpy_dtov(&z)?, want);
    let want_w: Vec<f32> = ramp(1.0).iter().map(|v| v * 4.0).collect();
    assert_eq!(dev.memcpy_dtov(&w)?, want_w);
    Ok(())
}

/// **Another thread's work waits while a segment records.** Its lease is
/// granted once the segment ends, so nothing it issues is taken into the graph,
/// and the segment's node count is exactly the wave's own launches.
#[test]
pub(super) fn another_threads_lease_waits_for_the_recording_segment() -> Result<()> {
    use std::sync::mpsc;
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let mut z = dev.alloc_zeros::<f32>(N)?;
    let a = dev.memcpy_stod(&ramp(1.0))?;
    let mut b = dev.alloc_zeros::<f32>(N)?;
    let before = dev.capture_stats();

    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    assert!(dev.recording_segment().is_some(), "the segment records");
    launch(&f, &dev.cuda_stream(), &x, &mut y, 2.0, 0.0)?;
    let (started, started_rx) = mpsc::channel();
    let other = {
        let (dev, f) = (dev.clone(), f.clone());
        std::thread::spawn(move || -> Result<Vec<f32>> {
            admit_this_thread();
            started.send(()).expect("the test is waiting");
            launch(&f, &dev.cuda_stream(), &a, &mut b, 5.0, 0.0)?;
            dev.memcpy_dtov(&b)
        })
    };
    started_rx.recv().expect("the other thread started");
    std::thread::sleep(std::time::Duration::from_millis(50));
    launch(&f, &dev.cuda_stream(), &y, &mut z, 1.0, 1.0)?;
    capture.finish()?;
    let theirs = other.join().expect("the other thread panicked")?;

    let after = dev.capture_stats();
    assert_eq!(after.segments - before.segments, 1);
    assert_eq!(
        after.nodes - before.nodes,
        2,
        "the other thread's launch is not in the graph"
    );
    let want: Vec<f32> = ramp(1.0).iter().map(|v| v * 2.0 + 1.0).collect();
    assert_eq!(dev.memcpy_dtov(&z)?, want);
    let want_theirs: Vec<f32> = ramp(1.0).iter().map(|v| v * 5.0).collect();
    assert_eq!(theirs, want_theirs);
    Ok(())
}

/// Memory another thread releases while a segment records is held until that
/// segment has been launched: the last reference to something a recorded
/// launch reads may be dropped anywhere, and a free issued at once would
/// queue on the compute stream ahead of the segment.
#[test]
pub(super) fn a_release_from_another_thread_waits_for_the_recording_segment() -> Result<()> {
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
        std::thread::spawn(move || {
            admit_this_thread();
            dev.retire(item)
        })
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
pub(super) fn the_recording_segment_advances_with_each_launch() -> Result<()> {
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
pub(super) fn an_abandoned_wave_launches_its_recording_segment() -> Result<()> {
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
///
/// The other thread's call waits for its lease while the segment records, so
/// it is joined only once the wave has finished.
#[test]
pub(super) fn another_threads_eager_staging_is_not_the_waves_scratch() -> Result<()> {
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    let (wave_scratch, _) = dev.with_staged_upload(&[1u8; 16], |at| Ok((at, ())))?;
    let foreign = {
        let dev = dev.clone();
        std::thread::spawn(move || -> Result<u64> {
            admit_this_thread();
            dev.cuda_context().bind_to_thread().w()?;
            dev.with_staged_upload(&[2u8; 16], Ok)
        })
    };
    capture.finish()?;
    let foreign_scratch = foreign.join().expect("the eager thread panicked")?;
    assert_ne!(
        wave_scratch, foreign_scratch,
        "an eager copy from another thread landed in the scratch a recorded segment writes"
    );
    Ok(())
}

/// **A lease waiting on a recording segment is granted behind that segment's
/// launch.** The other thread reads what the segment writes: granted when the
/// capture ended but before its graph was launched, its read would run first
/// and see the zeros the buffer started as.
#[test]
pub(super) fn a_waiting_lease_is_granted_behind_the_segments_launch() -> Result<()> {
    use std::sync::mpsc;
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let mut b = dev.alloc_zeros::<f32>(N)?;
    let y_addr = y.device_ptr(dev.stream_unleased()).0;

    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    assert!(dev.recording_segment().is_some(), "the segment records");
    for _ in 0..64 {
        launch(&f, &dev.cuda_stream(), &x, &mut y, 2.0, 0.0)?;
    }
    let (started, started_rx) = mpsc::channel();
    let reader = {
        let (dev, f) = (dev.clone(), f.clone());
        std::thread::spawn(move || -> Result<Vec<f32>> {
            admit_this_thread();
            started.send(()).expect("the test is waiting");
            let stream = dev.cuda_stream();
            // SAFETY: a view of `y`'s `N` f32s, which outlive this thread.
            let y = unsafe { stream.upgrade_device_ptr::<f32>(y_addr, N) };
            launch(&f, &stream, &y, &mut b, 5.0, 0.0)?;
            y.leak();
            drop(stream);
            dev.memcpy_dtov(&b)
        })
    };
    started_rx.recv().expect("the reader started");
    std::thread::sleep(std::time::Duration::from_millis(50));
    capture.finish()?;
    let read = reader.join().expect("the reader panicked")?;
    let want: Vec<f32> = ramp(1.0).iter().map(|v| v * 10.0).collect();
    assert_eq!(read, want, "the read ran ahead of the segment it waited on");
    Ok(())
}

/// Retired memory is dropped outside the hub's lock, so an item whose own drop
/// retires more does not deadlock the wave.
#[test]
pub(super) fn a_retired_item_may_retire_from_its_own_drop() -> Result<()> {
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
pub(super) fn pinned_staging_inside_a_recording_pauses_the_wave() -> Result<()> {
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
pub(super) fn a_pause_guard_outliving_its_wave_leaves_the_next_alone() -> Result<()> {
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
    let recording = |dev: &CudaDevice| dev.recording_segment().is_some();
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
pub(super) fn two_shapes_of_one_node_count_each_keep_their_executable() -> Result<()> {
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
