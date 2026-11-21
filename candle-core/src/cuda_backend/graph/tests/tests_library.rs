//! Tests of the executable library's lifecycle: which shapes it learns, trials and locks, and what it does with a
//! shape it does not hold.
//!
//! A budget nothing fits locks the library at once, and every wave then runs as issued. A locked library with room
//! reopens for traffic it did not learn. A proven topology keeps its executable when a one-off shape arrives, while
//! an unproven one is reshaped by it. The library learns, trials and locks within a couple of hundred waves, and
//! every wave's result is what the eager forward would have produced. A finiteness assert recorded inside a wave
//! must find a NaN, finish the wave, and leave the stream usable by other threads.

use super::*;

/// **A budget nothing fits locks the library at once, and a locked library
/// runs what it does not hold as issued.** The first admission does not fit:
/// nothing is evicted to make room, the executable is dropped once launched,
/// and with nothing admitted there is nothing to trial. The next wave's segment
/// is not recorded at all. Every wave's result is what the eager forward
/// would have produced.
#[test]
pub(super) fn a_budget_nothing_fits_locks_and_runs_eagerly() -> Result<()> {
    use super::super::budget::graph_budget_bytes;
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
    for wave in 0..super::super::hub::RECHECK_EVERY {
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
        (u64::from(super::super::hub::RECHECK_EVERY), 1, 0),
        "{} eager runs, then one recorded to recheck its shape",
        super::super::hub::RECHECK_EVERY
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
pub(super) fn a_lock_reopens_for_shapes_it_keeps_meeting() -> Result<()> {
    use super::super::budget::graph_budget_bytes;
    use super::super::regime::MIN_LOCK_WAVES;
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    dev.capture_hub().reset_library(1);
    let want = |add: f32| -> Vec<f32> { ramp(1.0).iter().map(|v| v + add).collect() };
    assert_eq!(add_wave(&dev, &f, &x, &mut y, 3, 1.0)?, want(1.0));
    assert_eq!(dev.capture_stats().phase, GraphPhase::Lock);

    let limit = MIN_LOCK_WAVES as u32 + 40 * super::super::hub::RECHECK_EVERY;
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
pub(super) fn the_library_learns_trials_and_locks_quickly() -> Result<()> {
    use super::super::budget::graph_budget_bytes;
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

/// **Past the learn, a held topology at a shape it does not hold runs once,
/// and the held executable keeps its own shape.** The same launch at another
/// grid is one topology. Under `Never` — the trial's and the lock's mode — it
/// is instantiated for its launch and computes its own extent, nothing in the
/// slot is rewritten, and the held shape still folds in place after it.
#[test]
pub(super) fn past_the_learn_a_new_shape_runs_once_and_reshapes_nothing() -> Result<()> {
    use super::super::slot::{Admit, ExecSlot, Folded};
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
    let (once, how, _) = slot.fold(t, a, &compute, 1, Admit::Keep, false)?;
    assert!(once.is_none());
    assert_eq!(how, Folded::Instantiated);
    slot.front().expect("kept").launch(&compute)?;
    assert_eq!(dev.memcpy_dtov(&y)?, want(N, 1.0, 1.0));

    let (t, a) = captured_n(&f, &stream, &x, &mut y, half, 2.0)?;
    let (once, how, _) = slot.fold(t, a, &compute, 1, Admit::Never, false)?;
    assert_eq!(how, Folded::Transient, "never: run once, kept nothing");
    once.expect("instantiated for its launch")
        .launch(&compute)?;
    assert_eq!(dev.memcpy_dtov(&y)?, want(half, 2.0, 1.0));

    let (t, a) = captured_n(&f, &stream, &x, &mut y, N, 3.0)?;
    let (once, how, _) = slot.fold(t, a, &compute, 1, Admit::Never, false)?;
    assert!(once.is_none());
    assert_eq!(how, Folded::InPlace, "the held shape was not rewritten");
    slot.front().expect("held").launch(&compute)?;
    assert_eq!(dev.memcpy_dtov(&y)?, want(N, 3.0, 3.0));
    assert_eq!(slot.held().0, 1);
    Ok(())
}

/// **A learn's one-off never reshapes a proven topology's executable.** In a
/// learn, a shape seen once folds over another executable of its topology —
/// unless the topology is proven: then it runs once, and the shape the trial
/// proved still folds in place after it. Unproven, the same one-off reshapes.
#[test]
pub(super) fn a_learns_one_off_leaves_a_proven_topology_alone() -> Result<()> {
    use super::super::slot::{Admit, ExecSlot, Folded};
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let f = scale_add(&dev)?;
    let x = dev.memcpy_stod(&ramp(1.0))?;
    let mut y = dev.alloc_zeros::<f32>(N)?;
    let stream = dev.cuda_context().new_stream().w()?;
    let compute = ComputeStream::of(&dev);
    let half = N / 2;
    for proven in [true, false] {
        let mut slot = ExecSlot::default();
        let (t, a) = captured_n(&f, &stream, &x, &mut y, N, 1.0)?;
        let (_, how, _) = slot.fold(t, a, &compute, 1, Admit::Keep, false)?;
        assert_eq!(how, Folded::Instantiated);

        let (t, a) = captured_n(&f, &stream, &x, &mut y, half, 2.0)?;
        let (once, how, _) = slot.fold(t, a, &compute, 1, Admit::Keep, proven)?;
        if proven {
            assert_eq!(how, Folded::Transient, "proven: the one-off runs once");
            once.expect("instantiated for its launch")
                .launch(&compute)?;
            let (t, a) = captured_n(&f, &stream, &x, &mut y, N, 3.0)?;
            let (_, how, _) = slot.fold(t, a, &compute, 1, Admit::Never, proven)?;
            assert_eq!(how, Folded::InPlace, "the proven shape is still held");
        } else {
            assert!(once.is_none());
            assert_eq!(how, Folded::Reshaped, "unproven: the one-off reshapes");
        }
        assert_eq!(slot.held().0, 1, "proven = {proven}");
    }
    Ok(())
}

/// **A bracketed segment yields its device time.** A start and an end recorded
/// around launches on the compute stream drain, once the end has fired, as one
/// sample of the key and mode they were recorded for. The trial is judged on
/// nothing else: events made without timing drained as no sample at all.
#[test]
pub(super) fn a_timed_segment_yields_one_sample() -> Result<()> {
    use super::super::regime::Mode;
    use super::super::timing::SegmentTimer;
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

/// **The finiteness instrument inside a wave.** An assert recorded into a
/// wave's segment and a drain at the wave's end — where the wave driver reads
/// its report — find the NaN, finish the wave, and leave the device's stream
/// to every other thread. A tensor-assert motherd on WhiteWorld left the stream
/// capturing at its first wave: every other thread's synchronise, cold load
/// and arena recycle was refused "operation not permitted when stream is
/// capturing" from then on, and no character decoded.
#[cfg(feature = "tensor-assert")]
#[test]
pub(super) fn a_drain_inside_a_wave_finds_the_nan_and_leaves_the_stream_usable() -> Result<()> {
    use crate::{Device, Tensor};
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let device = Device::Cuda(dev.clone());
    let t = Tensor::new(&[1f32, f32::NAN, 3.0], &device)?;
    // The slot table is made eagerly, as at load: a recorded call never
    // allocates.
    crate::tensor_assert::epoch(&device)?;
    dev.synchronize()?;

    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    t.assert("graph::tests::drain_in_a_wave");
    let bad = crate::tensor_assert::report(&device)?;
    capture.finish()?;

    assert_eq!(
        bad.first().map(|f| (f.name.as_str(), f.nan)),
        Some(("graph::tests::drain_in_a_wave", 1)),
        "the drain inside the wave found the NaN"
    );
    let other = std::thread::scope(|sc| {
        sc.spawn(|| {
            admit_this_thread();
            BackendDevice::synchronize(&dev).map_err(|e| e.to_string())
        })
        .join()
        .expect("the other thread")
    });
    assert_eq!(other, Ok(()), "another thread's synchronise after the wave");
    crate::tensor_assert::epoch(&device)?;
    Ok(())
}

/// **A quantized assert's first use, inside a wave.** The MoE dispatch checks
/// its int8 activations (`QTYPE_Q8A128V`) on every layer of every wave, so the
/// dequant scratch those asserts stage through is first made — and later grown
/// — while a wave records. It must count, finish the wave, and leave the stream
/// to other threads.
#[cfg(feature = "tensor-assert")]
#[test]
pub(super) fn a_quantized_assert_inside_a_wave_leaves_the_stream_usable() -> Result<()> {
    use crate::tensor_assert::QTYPE_Q8A128V;
    use crate::Device;
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let device = Device::Cuda(dev.clone());
    // Zeroed q8a128 blocks dequantize to zeros: finite, so the site is clean.
    let elems = 128 * 64;
    let q = dev.alloc_zeros::<u8>(elems * 2)?;
    let ptr = q.device_ptr(&dev.cuda_stream()).0;
    crate::tensor_assert::epoch(&device)?;
    dev.synchronize()?;

    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    // SAFETY: `q` holds `elems` q8a128 elements' worth of zeroed blocks and
    // outlives the wave.
    unsafe {
        crate::tensor_assert::assert_device_quant(
            "graph::tests::quant_in_a_wave",
            ptr,
            QTYPE_Q8A128V,
            elems,
            &dev,
        )
    };
    let report = crate::tensor_assert::report(&device);
    let finished = capture.finish();

    assert!(report.is_ok(), "the drain inside the wave: {report:?}");
    assert!(finished.is_ok(), "the wave finished: {finished:?}");
    let other = std::thread::scope(|sc| {
        sc.spawn(|| {
            admit_this_thread();
            BackendDevice::synchronize(&dev).map_err(|e| e.to_string())
        })
        .join()
        .expect("the other thread")
    });
    assert_eq!(other, Ok(()), "another thread's synchronise after the wave");
    crate::tensor_assert::epoch(&device)?;
    drop(q);
    Ok(())
}

/// **The process's first assert, issued inside a wave.** Nothing in a motherd
/// asserts at load, so the slot table is made by the first assert of the first
/// wave — while the wave records. Run alone (the table is per process), this
/// is that first assert: it must find the NaN and leave the stream as the
/// eager-table case does.
#[cfg(feature = "tensor-assert")]
#[test]
pub(super) fn a_first_assert_inside_a_wave_finds_the_nan_and_leaves_the_stream_usable() -> Result<()>
{
    use crate::{Device, Tensor};
    let _wave = wave_lock();
    let dev = CudaDevice::new(0)?;
    let device = Device::Cuda(dev.clone());
    let t = Tensor::new(&[f32::NAN, 2f32, 3.0], &device)?;
    dev.synchronize()?;

    let capture = dev.begin_wave_capture()?;
    dev.record_launches()?;
    t.assert("graph::tests::first_assert_in_a_wave");
    let bad = crate::tensor_assert::report(&device)?;
    capture.finish()?;

    assert_eq!(
        bad.first().map(|f| (f.name.as_str(), f.nan)),
        Some(("graph::tests::first_assert_in_a_wave", 1)),
        "the first assert, made inside the wave, was counted"
    );
    let other = std::thread::scope(|sc| {
        sc.spawn(|| {
            admit_this_thread();
            BackendDevice::synchronize(&dev).map_err(|e| e.to_string())
        })
        .join()
        .expect("the other thread")
    });
    assert_eq!(other, Ok(()), "another thread's synchronise after the wave");
    crate::tensor_assert::epoch(&device)?;
    Ok(())
}
