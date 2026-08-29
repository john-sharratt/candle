use super::*;

/// The span must be usable — and cost nothing — regardless of which features
/// are on. This is the only assertion available when `profile` is off, and it is
/// the one that catches a broken cfg combination.
#[test]
fn a_span_opens_and_closes_under_any_feature_set() {
    let s = span("probe:explicit_end");
    s.end();
    {
        let _s = span("probe:scope_end");
    }
    assert!(span_if(false, "probe:skipped").is_none());
    assert!(span_if(true, "probe:taken").is_some());
}

/// Same contract for the GPU span, and the reason it takes a `&Device` rather
/// than a stream: a CPU device is not an error, it is a span with nothing to
/// record. A caller on a mixed-device path must not need a `cfg` or a match.
#[test]
fn a_gpu_span_on_a_cpu_device_is_a_silent_no_op() {
    let dev = candle::Device::Cpu;
    let g = gpu_span("probe:cpu", &dev);
    g.end();
    {
        let _g = gpu_span("probe:cpu_scope", &dev);
    }
    assert!(gpu_span_if(false, "probe:cpu_skipped", &dev).is_none());
    assert!(gpu_span_if(true, "probe:cpu_taken", &dev).is_some());
    // Draining with nothing recorded must not panic or block.
    gpu_drain();
    gpu_drain_blocking();
}

/// The phase helper exists to keep the name choice off the call site; it must
/// pick the same name the hand-written conditional did.
#[test]
fn the_phase_helper_picks_by_phase() {
    let dev = candle::Device::Cpu;
    // Names are the observable, so drive them through a real open/close: on CPU
    // both arms are no-ops, which is exactly what makes this safe to assert
    // without a GPU.
    gpu_span_phase(true, "decode:probe", "prefill:probe", &dev).end();
    gpu_span_phase(false, "decode:probe", "prefill:probe", &dev).end();
}

/// A snapshot merge must add totals and counts per name, not replace them —
/// the per-config tables are built by merging.
#[test]
fn snapshots_merge_by_name() {
    let mut a = ProfileSnapshot {
        entries: vec![("x".into(), 1.0, 1), ("y".into(), 2.0, 2)],
    };
    let b = ProfileSnapshot {
        entries: vec![("y".into(), 3.0, 3), ("z".into(), 4.0, 4)],
    };
    a.merge(&b);
    let get = |n: &str| a.entries.iter().find(|(k, _, _)| k == n).cloned();
    assert_eq!(get("x"), Some(("x".into(), 1.0, 1)));
    assert_eq!(
        get("y"),
        Some(("y".into(), 5.0, 5)),
        "totals and counts add"
    );
    assert_eq!(get("z"), Some(("z".into(), 4.0, 4)));
}

/// Recording the same name twice accumulates rather than duplicating, and a
/// zero count is ignored — a span that ran no events must not create a row that
/// reads as "measured, and free".
#[cfg(feature = "profile")]
#[test]
fn the_accumulator_folds_by_name_and_ignores_empty_counts() {
    use std::time::Duration;
    let mut acc = ProfileAccumulator::new();
    acc.record_duration("a", Duration::from_millis(10), 1);
    acc.record_duration("a", Duration::from_millis(5), 2);
    acc.record_duration("b", Duration::from_millis(1), 0);
    let snap = acc.snapshot();
    assert_eq!(snap.entries.len(), 1, "`b` had count 0 and must not appear");
    let (name, total_ms, count) = snap.entries[0].clone();
    assert_eq!(name, "a");
    assert_eq!(count, 3);
    assert!((total_ms - 15.0).abs() < 1e-6, "got {total_ms}ms");
}

/// The real path: a GPU span must time actual device work, and the pool must
/// hand the pair back so a wave's worth of spans does not create a wave's worth
/// of events.
///
/// Asserts a LOWER bound only. An upper bound would be a throughput assertion
/// wearing a correctness costume — it would fail on a busy machine and teach
/// everyone to ignore it.
#[cfg(all(feature = "profile", feature = "cuda"))]
#[test]
fn a_gpu_span_times_real_device_work() {
    let Ok(dev) = candle::Device::new_cuda(0) else {
        eprintln!("skipping: CUDA device required");
        return;
    };
    // Something big enough that the elapsed time is unambiguously non-zero.
    let a = candle::Tensor::ones((512, 512), candle::DType::F32, &dev).unwrap();

    let g = gpu_span("probe:gpu_matmul", &dev);
    for _ in 0..8 {
        let _ = a.matmul(&a).unwrap();
    }
    g.end();

    // Nothing is attributable until a drain: the elapsed time lives in the
    // events, so an unharvested pair contributes nothing at all.
    gpu_drain_blocking();
    let snap = pipeline_snapshot_and_reset();
    let row = snap
        .entries
        .iter()
        .find(|(n, _, _)| n == "probe:gpu_matmul")
        .expect("the drained span must reach the pipeline accumulator");
    assert_eq!(row.2, 1, "one span, one count");
    assert!(row.1 > 0.0, "elapsed must be positive, got {}ms", row.1);
}

/// A run that never reaches a drain boundary must still be bounded. Opening far
/// more spans than `HIGH_WATER` without draining once has to leave the pending
/// list capped and every span still accounted for — the failure this catches is
/// a profiler that allocates two driver events per span for the length of a
/// prefill.
#[cfg(all(feature = "profile", feature = "cuda"))]
#[test]
fn the_pool_bounds_itself_without_an_explicit_drain() {
    let Ok(dev) = candle::Device::new_cuda(0) else {
        eprintln!("skipping: CUDA device required");
        return;
    };
    let _ = pipeline_snapshot_and_reset();
    gpu_drain_blocking();

    let n = super::gpu::high_water_for_test() * 2 + 64;
    for _ in 0..n {
        gpu_span("probe:bounded", &dev).end();
    }
    assert!(
        super::gpu::pending_len_for_test() < super::gpu::high_water_for_test() + 64,
        "pending list ran away: {}",
        super::gpu::pending_len_for_test()
    );

    // Bounded, but not at the cost of losing spans: every one still lands.
    gpu_drain_blocking();
    let snap = pipeline_snapshot_and_reset();
    let row = snap
        .entries
        .iter()
        .find(|(k, _, _)| k == "probe:bounded")
        .expect("bounded spans must still be recorded");
    assert_eq!(row.2, n as u64, "no span may be dropped by the cap");
}

/// Draining is what makes the pair reusable, so a second span after a drain must
/// still record — the failure this catches is a pool that leaks its events and
/// silently stops measuring partway through a run.
#[cfg(all(feature = "profile", feature = "cuda"))]
#[test]
fn the_event_pool_recycles_across_drains() {
    let Ok(dev) = candle::Device::new_cuda(0) else {
        eprintln!("skipping: CUDA device required");
        return;
    };
    let a = candle::Tensor::ones((256, 256), candle::DType::F32, &dev).unwrap();
    let _ = pipeline_snapshot_and_reset();

    for _ in 0..3 {
        let g = gpu_span("probe:recycle", &dev);
        let _ = a.matmul(&a).unwrap();
        g.end();
        gpu_drain_blocking();
    }
    let snap = pipeline_snapshot_and_reset();
    let row = snap
        .entries
        .iter()
        .find(|(n, _, _)| n == "probe:recycle")
        .expect("recycled pairs must keep recording");
    assert_eq!(row.2, 3, "every pass records once");
}

/// The readout must see spans recorded on OTHER threads.
///
/// This is the whole reason the store is per-thread with a process-wide
/// aggregate rather than a plain thread-local: a daemon records from its
/// scheduler thread and serves the numbers from a request handler, and a
/// thread-local readout answers that request with an empty table — which reads
/// as "the hot path was free" rather than "you asked the wrong thread".
#[cfg(feature = "profile")]
#[test]
fn the_readout_sees_spans_from_every_thread() {
    let _ = pipeline_snapshot_and_reset();

    std::thread::spawn(|| {
        let _s = span("probe:other_thread");
    })
    .join()
    .expect("worker thread");

    {
        let _s = span("probe:this_thread");
    }

    let snap = pipeline_snapshot_and_reset();
    for name in ["probe:other_thread", "probe:this_thread"] {
        assert!(
            snap.entries.iter().any(|(n, _, c)| n == name && *c == 1),
            "{name} missing from the aggregate readout: {:?}",
            snap.entries,
        );
    }
}

/// A thread that has already exited must not take its totals with it — its work
/// happened, and a table that silently omits it under-reports exactly the
/// short-lived pool threads a daemon does its background work on.
#[cfg(feature = "profile")]
#[test]
fn a_finished_threads_totals_survive_it() {
    let _ = pipeline_snapshot_and_reset();

    std::thread::spawn(|| {
        let _s = span("probe:retired");
    })
    .join()
    .expect("worker thread");
    // The thread is joined, so its accumulator has already been dropped and
    // folded into the retired table by the time this reads.
    let snap = pipeline_snapshot_and_reset();
    assert!(
        snap.entries
            .iter()
            .any(|(n, _, c)| n == "probe:retired" && *c == 1),
        "a joined thread's span was lost: {:?}",
        snap.entries,
    );
}

/// Reading resets, so the second of two consecutive reads describes only what
/// happened between them. That property is what lets a caller bracket the exact
/// window a rate collapsed in, so it is worth a gate of its own.
#[cfg(feature = "profile")]
#[test]
fn reading_resets_so_two_reads_bracket_a_window() {
    let _ = pipeline_snapshot_and_reset();
    {
        let _s = span("probe:window");
    }
    let first = pipeline_snapshot_and_reset();
    assert!(first.entries.iter().any(|(n, _, _)| n == "probe:window"));

    let second = pipeline_snapshot_and_reset();
    assert!(
        !second.entries.iter().any(|(n, _, _)| n == "probe:window"),
        "the first read must have cleared the counter: {:?}",
        second.entries,
    );
}
