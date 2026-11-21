//! Measures the clean baseline row through the engine before any churn, and
//! judges its answers the way the fragmented pool's are judged.
//!
//! A warm-up batch of the same shape runs first, so first-use costs such as
//! planner setup, first arena claims and first kernel launches are not charged
//! to the measured row. A baseline that answers wrongly is reported with the
//! story failures rather than aborting, so the later phases still run and are
//! judged. The pipeline profile is cleared before the batch, so a profiling
//! build reports this batch alone.

use super::*;

/// The clean baseline: the gate's `C<level> ×width` row through `engine`, and the
/// story-gate failures, if any.
///
/// **One unmeasured batch of the same shape runs first.** The gate's row sits in the
/// middle of its ladder, so every path it takes has run before its clock starts; an
/// engine measured on its very first turn would charge first-use costs — the wave
/// planner arming, the first arena claims, each kernel's first launch — to a row the
/// gate measures warm. The warm-up is evicted like the measured batch, so neither leaves
/// KV behind.
pub(super) fn clean_baseline(
    engine: &ConversationEngine,
    probe: &Probe,
) -> anyhow::Result<(BaselineRow, Vec<String>)> {
    let builder = probe.builder();
    let batch = |label: &str| -> anyhow::Result<_> {
        println!(
            "baseline ({label}) — the gate's C{}×{} row through the engine, clean pool\n",
            probe.compression_level, probe.baseline_width,
        );
        let b = run_story_batch(
            engine,
            &builder,
            probe.baseline_width,
            probe.baseline_decode,
            probe.story_gate(),
        )?;
        print_timing(&b.timing, probe.baseline_width);
        print_submit_blocked(&b.submit_blocked);
        for c in b.conversations.iter() {
            let _ = engine.evict_ingest_timeline(c.timeline_id());
        }
        // A turn that errored, or whose stream differs from what it generated, is a
        // failure of this row — reported with the story failures rather than
        // aborting the probe, so the phases after it still run and are judged.
        let fails = b
            .story_fail
            .into_iter()
            .chain(b.errors)
            .map(|line| format!("baseline ({label}): {line}"))
            .collect::<Vec<_>>();
        Ok((b.timing, b.story_pass, fails))
    };
    let (_, _, warm_fails) = batch("warm-up, unmeasured")?;
    // The pipeline profile accumulates process-wide, so the warm-up, the load and
    // every conversation opened so far would blend into the measured batch's spans.
    // Cleared here, a `profile` build's breakdown is this batch's alone; without the
    // feature it is an empty snapshot and costs nothing.
    let _ = pipeline_snapshot_and_reset();
    let (timing, story_pass, measured_fails) = batch("measured")?;
    Ok((
        BaselineRow {
            level: probe.compression_level,
            width: probe.baseline_width,
            timing,
            story_pass,
        },
        warm_fails.into_iter().chain(measured_fails).collect(),
    ))
}

/// Load the probe's model and run only the clean baseline — the iteration loop for
/// optimising the engine against the gate's row, without the minutes of churn and
/// drain the full probe spends answering a different question.
pub fn run_baseline(probe: &Probe) -> anyhow::Result<(BaselineRow, Vec<String>)> {
    let device = Device::new_cuda(probe.device)?;
    let builder = probe.builder();
    let resolved = builder.resolve_model(&device)?;
    let tokenizer = resolved.tokenizer()?;
    println!("Loading {:?} …", resolved.pack);
    let model = builder
        .load_model(&builder.weight_files(&resolved.pack)?, &device, None)?
        .model;
    let (_scratch, engine) = start_engine(probe, &device, &resolved.pack, &tokenizer, model)?;
    clean_baseline(&engine, probe)
}
