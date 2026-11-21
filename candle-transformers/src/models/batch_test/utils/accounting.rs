//! Prices a prefill pass against the ground a forward can reach, and takes the
//! VRAM baseline a model load is measured from.
//!
//! A prefill is sliced only when the wave it would form cannot be placed. The
//! wave is priced by the model's own placement arithmetic rather than by the
//! width cap, since the transient tier grows with sequences as well as rows and
//! the cap prices only a one-sequence wave. A load's weight footprint is a
//! free-VRAM delta across the load, so the baseline is taken before it runs;
//! measured after the load, the weights read as zero. A model that cannot price
//! its wave runs it as composed, and the placement decides.

use super::*;

/// Whether a prefill of `takes[i]` rows on each of `seqs` fits the ground a
/// forward can reach, priced as the wave it actually is: the model's own tier
/// for that many sequences at their current depths
/// ([`ManagedBatchedModel::wave_tier_bytes`], the function the placement uses)
/// plus the KV those rows write, in whole regions, against
/// [`ManagedBatchedModel::prefill_ground_bytes`].
///
/// The width cap alone cannot answer this: it prices a one-sequence wave, and
/// the tier grows with sequences as well as rows. Measured on
/// Qwen3.8-Flash-Next at BF16 ×8 on a 16 GB card: a cap of 4,354 rows, a
/// 4,160-row eight-sequence wave, and a tier that did not fit.
///
/// `true` when the model cannot price the wave — it then runs as composed, and
/// the placement is the check.
pub(super) fn prefill_pass_fits<M: ManagedBatchedModel + ?Sized>(
    model: &M,
    session: &BatchedInferenceSession,
    seqs: &[usize],
    takes: &[usize],
) -> bool {
    prefill_pass_cost(model, session, seqs, takes).is_none_or(|c| c.tier + c.kv <= c.ground)
}

/// What a prefill pass costs against the ground it can reach, in bytes.
pub(super) struct PassCost {
    pub(super) tier: usize,
    pub(super) kv: usize,
    pub(super) ground: usize,
}

/// [`prefill_pass_fits`]'s three figures, or `None` when the model cannot price
/// the wave.
pub(super) fn prefill_pass_cost<M: ManagedBatchedModel + ?Sized>(
    model: &M,
    session: &BatchedInferenceSession,
    seqs: &[usize],
    takes: &[usize],
) -> Option<PassCost> {
    let act = session.activation_dtype();
    let ground = model.prefill_ground_bytes()?;
    let offsets: Vec<usize> = seqs
        .iter()
        .map(|&s| session.sequence_offset(s).unwrap_or(0))
        .collect();
    let rows: usize = takes.iter().sum();
    let tier = model.wave_tier_bytes(rows, seqs.len(), takes, &offsets, seqs, &[], act)? as usize;
    let kv = model.kv_bytes_per_row(act).map_or(0, |per_row| {
        rows.saturating_mul(per_row).div_ceil(REGION_BYTES) * REGION_BYTES
    });
    Some(PassCost { tier, kv, ground })
}

/// The expert pipeline's tallies for the interval just ended, with its
/// residency ceiling filled in — taken where the tallies are, before the reset
/// that starts the next interval.
pub(super) fn expert_stats_with_ceiling<M: ManagedBatchedModel + ?Sized>(
    model: &M,
) -> Option<PipelineStats> {
    let mut stats = model.expert_stats()?;
    if let Some((lru, min)) = model.expert_hit_references() {
        stats.hit_lru = lru;
        stats.hit_ceiling = min;
    }
    Some(stats)
}

/// Calculate statistics from timing measurements, dropping the first (warmup) and worst outlier
pub fn calculate_stats(times: &[Duration]) -> (f64, f64, f64, usize) {
    if times.is_empty() {
        return (0.0, 0.0, 0.0, 0);
    }

    let mut sorted: Vec<f64> = times.iter().map(|d| d.as_secs_f64()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Skip first (warmup) and last (worst outlier)
    let used = if sorted.len() > 2 {
        &sorted[1..sorted.len() - 1]
    } else {
        &sorted[..]
    };

    if used.is_empty() {
        return (0.0, 0.0, 0.0, 0);
    }

    let avg = used.iter().sum::<f64>() / used.len() as f64;
    let min = *used
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max = *used
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    (avg, min, max, used.len())
}

/// Take the VRAM baseline, run `load`, and register what the weights cost.
///
/// The weights' footprint is a free-VRAM delta **across** the load, so the
/// baseline has to be taken before they come in. That is why this wraps the
/// load rather than sitting inside [`TestParams::run_loaded`], which by
/// construction runs with the model already resident: a baseline taken there
/// measures the weights as zero and folds the whole checkpoint into the
/// `cuda context` row, which is the number the VRAM table's sizing decisions
/// are read off.
///
/// One definition, two consumers: [`TestParams::run`] wraps its `load_model`,
/// and a caller that loads once for several [`TestParams::run_loaded`] sweeps
/// wraps its own load so its table is populated the same way.
pub fn account_model_load<M>(device: &Device, load: impl FnOnce() -> Result<M>) -> Result<M> {
    candle::gpu_memory::clear();
    let (free_before, total) = device.mem_get_info().unwrap_or((0, 0));
    let cuda_ctx_bytes = total.saturating_sub(free_before);
    if cuda_ctx_bytes > 0 {
        candle::gpu_memory::register("cuda context", cuda_ctx_bytes);
    }
    let _ = candle::gpu_memory::snapshot("before_model_load", device);
    let model = load()?;
    let _ = candle::gpu_memory::snapshot("after_model_load", device);
    print_host_ram_line("after model load");
    let free_after = device.mem_get_info().map(|(f, _)| f).unwrap_or(0);
    candle::gpu_memory::register("model weights", free_before.saturating_sub(free_after));
    Ok(model)
}
