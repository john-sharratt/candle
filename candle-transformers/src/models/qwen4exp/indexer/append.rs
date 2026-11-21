//! Appends every sequence's index keys for a wave in one pair of launches.
//!
//! A per-sequence loop cost around thirty launches per span, per layer, per
//! wave, and the arithmetic was never the expensive part. The kernel reads each
//! span's rows in place through a descriptor table, so no copy is needed. The
//! two launches run in stream order: the append reads the open-block rows the
//! previous wave carried, and only then does the carry overwrite them. Admission
//! sizes the pages for the wave, so an append claims no new ground.

use super::*;

/// One span of the wave: whose cache it appends to, and which rows of the
/// wave's key projection are its own.
pub struct AppendSpan<'a> {
    pub cache: &'a mut IndexCache,
    /// First row of `k_all` belonging to this span.
    pub start: usize,
    pub rows: usize,
}

/// Append every span's index keys — pool, RMS-norm, store un-rotated — in ONE
/// pair of launches for the whole wave.
///
/// This replaces a per-sequence loop that cost ~30 launches a span, per layer,
/// per wave: `nsys` over `tests/qsa_index_bench.rs` measured ~21,500 launches
/// for 20.5 ms of GPU time against ~208 ms of wall clock, the GPU busy a tenth
/// of the time. The arithmetic was never the cost, so the fix is not faster
/// arithmetic but issuing it once (hot-path invariant 5). The kernel reads each
/// span's rows in place through a descriptor table rather than requiring one
/// dense block, which is what removes the copies as well (invariant 2b).
///
/// The two launches are ordered: the append reads the open-block rows the
/// PREVIOUS wave carried, and the carry then overwrites them with this wave's
/// trailing rows. Same stream, append first.
pub fn append_wave(
    work: &mut [AppendSpan<'_>],
    k_all: &Tensor,
    w: &IndexerWeights,
    ratio: usize,
    rms_eps: f64,
) -> Result<()> {
    use candle_kernels::simple::qsa_index_append::{
        run_qsa_index_append, run_qsa_index_carry, CARRY_WORDS, JOB_WORDS, MAX_D,
    };

    if work.is_empty() || work.iter().all(|s| s.rows == 0) {
        return Ok(());
    }
    if ratio == 0 || ratio > MAX_RATIO {
        candle::bail!("qsa append: ratio {ratio} outside 1..={MAX_RATIO}");
    }
    let d = work[0].cache.keys.head_dim();
    if d == 0 || d > MAX_D {
        candle::bail!(
            "qsa append: head_dim {d} outside 1..={MAX_D} — the kernel gives one thread to a \
             channel, so the channel count is the block width"
        );
    }
    let device = k_all.device().clone();
    let k_base = tensor_ptr(k_all)?;
    let elem = std::mem::size_of::<f32>() as u64;
    let row = d as u64 * elem;

    let mut jobs: Vec<i64> = Vec::new();
    let mut carries: Vec<i64> = Vec::new();
    // (n_blocks, n_open) each span ends at, applied only once the launches that
    // depend on the entering values have been issued.
    let mut commits: Vec<(usize, usize)> = Vec::with_capacity(work.len());

    for span in work.iter_mut() {
        let (n_new, left) = span.cache.plan(span.rows, ratio);
        let n_blocks = span.cache.n_blocks;
        let n_open = span.cache.n_open;
        // Admission sized the pages for this wave, so this claims nothing; a
        // page it did have to claim inside the forward is refused by the arena
        // window rather than carved out of ground the wave stands on.
        span.cache.keys.ensure(n_blocks + n_new)?;
        let pages = span.cache.keys.page_ptrs()?;
        let raw = span.cache.raw_ptr()?;

        for i in 0..n_new {
            // Only the first block of a span can straddle the carried rows;
            // every later one is contiguous in the wave's projection.
            let n0 = if i == 0 { n_open.min(ratio) } else { 0 };
            let src1 = span.start + (i * ratio).saturating_sub(n_open);
            let (codes, scales) = key_addr(&pages, n_blocks + i, d);
            jobs.push(codes as i64);
            jobs.push(if n0 > 0 { raw as i64 } else { 0 });
            jobs.push(n0 as i64);
            jobs.push((k_base + src1 as u64 * row) as i64);
            jobs.push(scales as i64);
        }

        if left > 0 {
            // No block completed, so the carried rows stand and this span's
            // rows extend them; otherwise the block consumed them and the
            // trailing rows start the next one.
            let (dst, src, rows) = if n_new == 0 {
                (
                    raw + n_open as u64 * row,
                    k_base + span.start as u64 * row,
                    span.rows,
                )
            } else {
                (
                    raw,
                    k_base + (span.start + span.rows - left) as u64 * row,
                    left,
                )
            };
            carries.push(dst as i64);
            carries.push(src as i64);
            carries.push(rows as i64);
        }
        commits.push((n_blocks + n_new, left));
    }

    let candle::Device::Cuda(dev) = &device else {
        candle::bail!("qsa append runs on CUDA");
    };
    let n_jobs = jobs.len() / JOB_WORDS;
    let n_carry = carries.len() / CARRY_WORDS;
    let grids = AppendGrids::for_spans(work.iter().map(|s| s.rows), ratio);
    if n_jobs > grids.jobs || n_carry > grids.carries {
        candle::bail!(
            "qsa append: {n_jobs} jobs and {n_carry} carries past the grids' bound {grids:?}"
        );
    }
    // Both tables in one upload through the device's staging scratch — the
    // jobs, then the carries — so an append allocates nothing, whether it runs
    // inside a forward or in a rewind between two. Every span with rows
    // completes a block or carries rows, so the table is never empty.
    let carries_at = jobs.len();
    let mut table = jobs;
    table.extend_from_slice(&carries);
    let k_norm = tensor_ptr(&w.k_norm)?;
    // The stream is taken inside, where the launches are — the staged upload
    // runs eagerly, so they belong on the stream it names there.
    dev.with_staged_upload(&table, |base| {
        let stream = dev.cuda_stream();
        let raw_stream = stream.cu_stream() as *mut c_void;
        // Both launch every wave, over the bound's grids, whatever this wave
        // completes or carries ([`AppendGrids`]).
        candle::set_kernel_breadcrumb("run_qsa_index_append", file!(), line!());
        unsafe {
            run_qsa_index_append(
                base as *const i64,
                k_norm as *const f32,
                d as i32,
                ratio as i32,
                rms_eps as f32,
                n_jobs as i32,
                grids.jobs as i32,
                raw_stream,
            );
        }
        candle::set_kernel_breadcrumb("run_qsa_index_carry", file!(), line!());
        unsafe {
            run_qsa_index_carry(
                (base as *const i64).add(carries_at),
                d as i32,
                n_carry as i32,
                grids.carries as i32,
                raw_stream,
            );
        }
        Ok(())
    })?;

    for (span, (n_blocks, n_open)) in work.iter_mut().zip(commits) {
        span.cache.n_blocks = n_blocks;
        span.cache.n_open = n_open;
    }
    Ok(())
}

/// The most candidate blocks any row of the wave sees on KV layer `kv` — at
/// least one. What the score buffer and the selection table are sized from;
/// [`select_layer`] and the admission pricing (`select_bytes`) both ask it here
/// so they cannot size the same carve two ways.
pub fn widest_candidates(
    spans: &[SeqSpan],
    offsets: &[usize],
    idx_map: &HashMap<usize, Vec<IndexCache>>,
    kv: usize,
    ratio: usize,
) -> usize {
    spans
        .iter()
        .zip(offsets)
        .map(|(span, &off)| {
            let last = off + span.len;
            match idx_map.get(&span.seq).and_then(|c| c.get(kv)) {
                Some(cache) => cache.candidates_at(last.saturating_sub(1), ratio),
                None => last.div_ceil(ratio),
            }
        })
        .max()
        .unwrap_or(0)
        .max(1)
}

/// The wave's spans, by index, cut into the runs that are scored together: a run
/// of prompt chunks (at least [`PREFILL_KV_STAGE_MIN_Q_LEN`] rows each) or a run
/// of narrow spans — decode rows and verify windows. Each run's score buffer is
/// as wide as its own deepest row ([`select_layer`], `select_bytes`).
///
/// **One width for the whole wave priced every row at the deepest one's.** A
/// decode row of a deep dialogue riding beside an ingest chunk made the chunk's
/// thousands of rows as wide as the dialogue's candidates: measured on `.5`,
/// 2026-10-17, an 8,202-row wave carved its score buffer at 78,939 columns —
/// 2.6 GB on every attention layer's phase, bought from the expert zone — where
/// the chunk's own rows ranked under 16K. Runs are contiguous in the wave's
/// packed order, so each one's rows are a row range of the selection table.
pub fn score_runs(spans: &[SeqSpan]) -> Vec<Range<usize>> {
    let bulk = |s: &SeqSpan| s.len >= PREFILL_KV_STAGE_MIN_Q_LEN;
    let mut runs: Vec<Range<usize>> = Vec::new();
    for (i, s) in spans.iter().enumerate() {
        match runs.last_mut() {
            Some(r) if bulk(&spans[r.start]) == bulk(s) => r.end = i + 1,
            _ => runs.push(i..i + 1),
        }
    }
    runs
}

/// A bulk selection's shape, which is what its scoring and top-k cost follow:
/// the rows scored, the candidate blocks the widest row ranks, and the page
/// layout the attention walks through.
pub(super) fn log_bulk_selection(rows: usize, widest: usize, pages: usize) {
    if rows >= PREFILL_KV_STAGE_MIN_Q_LEN {
        note("bulk qsa selection", || {
            format!("rows={rows} widest_candidates={widest} pages={pages}")
        });
    }
}
