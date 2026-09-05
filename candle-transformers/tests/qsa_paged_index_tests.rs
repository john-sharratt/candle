//! The ragged, paged QSA index scorer: correctness against a CPU oracle, and a
//! benchmark shaped so an optimisation cycle is seconds rather than minutes.
//!
//! # Why a separate scorer exists
//!
//! The dense path hands cuBLAS one contiguous key buffer. A reconstructed index
//! is not one buffer — it is the sealed pieces of the turns a projection
//! selected, separately allocated, and **ragged**: a turn of `T` tokens seals
//! `ceil(T / ratio)` rows whose last one covers `T mod ratio` tokens, because a
//! turn boundary does not land on a block boundary.
//!
//! # What the oracle is, and why it is not the dense kernel
//!
//! [`PagedIndex::score_reference`] is the naive expression evaluated eagerly on
//! the host — one dot product at a time, in row order. Comparing against the
//! dense GPU path instead would only prove the two agree, which is exactly the
//! thing a shared bug would also produce. The tolerance is not laxity: the
//! kernel accumulates in a different order (a `float4` body, heads folded after
//! a per-head ReLU), so bit-equality is not available and asking for it would
//! pin an implementation rather than a result.

#![cfg(feature = "cuda")]

use candle::{DType, Device, Result, Tensor};
use candle_transformers::models::qwen4exp::paged_index::{
    decode_page, encode_page, IndexPage, PagedIndex, SealedIndex,
};

/// The released checkpoint's indexer geometry.
const HEAD_DIM: usize = 128;
const N_HEADS: usize = 4;
const RATIO: usize = 4;

/// Every GPU test in this file shares one device and one process-global lock:
/// the arenas draw from a process-wide pool, so two tests decoding at once
/// corrupt each other's regions rather than merely running slowly.
fn gpu() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

fn dev() -> Result<Device> {
    Device::new_cuda(0)
}

/// A deterministic LCG — reproducibility matters more than statistical quality,
/// and a failing case has to be re-runnable exactly.
struct Lcg(u64);

impl Lcg {
    fn f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }
}

/// Build a window whose pages have the given row counts, with the last page
/// ragged by `last_cells`.
fn window(rows_per_page: &[usize], last_cells: usize, seed: u64) -> Result<PagedIndex> {
    let d = dev()?;
    let mut rng = Lcg(seed);
    let mut pages = Vec::new();
    let mut pos = 0usize;
    let n = rows_per_page.len();
    for (i, &r) in rows_per_page.iter().enumerate() {
        let cells = if i + 1 == n { last_cells } else { RATIO };
        let keys = Tensor::from_vec(rng.vec(r * HEAD_DIM), (r, HEAD_DIM), &d)?;
        pages.push(IndexPage::new(keys, pos, cells));
        pos += if r == 0 { 0 } else { (r - 1) * RATIO + cells };
    }
    PagedIndex::new(pages, RATIO, &d)
}

fn queries(t: usize, seed: u64) -> Result<Tensor> {
    let d = dev()?;
    let mut rng = Lcg(seed);
    Tensor::from_vec(rng.vec(t * N_HEADS * HEAD_DIM), (t, N_HEADS, HEAD_DIM), &d)
}

/// Compare kernel against oracle, returning the worst relative error over the
/// unmasked cells and asserting the mask lands on exactly the same columns.
fn compare(idx: &mut PagedIndex, q: &Tensor, qpos: &[usize]) -> Result<f32> {
    let d = dev()?;
    let n = idx.total_rows();
    let t = qpos.len();
    let out = Tensor::zeros((t, n.max(1)), DType::F32, &d)?;
    idx.score_rows(q, qpos, N_HEADS, HEAD_DIM, &out, n.max(1), 0)?;
    let got = out.flatten_all()?.to_vec1::<f32>()?;
    let want = idx.score_reference(q, qpos, N_HEADS, HEAD_DIM)?;

    let mut worst = 0f32;
    for r in 0..t {
        for g in 0..n {
            let (a, b) = (got[r * n + g], want[r * n + g]);
            let masked_a = a <= -1e29;
            let masked_b = b <= -1e29;
            assert_eq!(
                masked_a, masked_b,
                "row {r} col {g}: mask disagrees (kernel {a}, oracle {b}) — the visible \
                 prefix is the one thing the ragged widths change, so a disagreement here \
                 is the widths being folded into `cnt` wrongly, not a numerics issue"
            );
            if !masked_a {
                let denom = b.abs().max(1e-3);
                worst = worst.max((a - b).abs() / denom);
            }
        }
    }
    Ok(worst)
}

/// One page, every row full: the paged scorer must agree with the oracle on the
/// case the dense path already covers, or nothing below means anything.
#[test]
fn uniform_single_page_matches_the_oracle() -> Result<()> {
    let _g = gpu().lock().unwrap();
    let mut idx = window(&[64], RATIO, 11)?;
    let qpos: Vec<usize> = vec![10, 63, 128, 255];
    let q = queries(qpos.len(), 12)?;
    let worst = compare(&mut idx, &q, &qpos)?;
    assert!(worst < 2e-3, "worst relative error {worst}");
    Ok(())
}

/// Many pages, every one still block-aligned. Isolates the paging from the
/// raggedness: if this fails the descriptor walk is wrong, not the widths.
#[test]
fn many_aligned_pages_match_the_oracle() -> Result<()> {
    let _g = gpu().lock().unwrap();
    let mut idx = window(&[7, 3, 19, 1, 44, 12], RATIO, 21)?;
    let qpos: Vec<usize> = (0..8).map(|i| i * 37 + 5).collect();
    let q = queries(qpos.len(), 22)?;
    let worst = compare(&mut idx, &q, &qpos)?;
    assert!(worst < 2e-3, "worst relative error {worst}");
    Ok(())
}

/// **The ragged case.** The last row covers fewer than `ratio` tokens, so the
/// candidate prefix is no longer `(pos + 1) / ratio` for any position inside the
/// final page. Swept across every width so an off-by-one in the width walk
/// cannot hide in one of them.
#[test]
fn a_ragged_last_row_matches_the_oracle_at_every_width() -> Result<()> {
    let _g = gpu().lock().unwrap();
    for cells in 1..=RATIO {
        let mut idx = window(&[9, 5, 17], cells, 30 + cells as u64)?;
        let total = idx.total_tokens();
        // Positions across the whole span, and every position in the final
        // page's neighbourhood, where the short row decides visibility.
        let mut qpos: Vec<usize> = (0..total).step_by(7).collect();
        qpos.extend(total.saturating_sub(2 * RATIO)..total);
        let q = queries(qpos.len(), 40 + cells as u64)?;
        let worst = compare(&mut idx, &q, &qpos)?;
        assert!(worst < 2e-3, "cells {cells}: worst relative error {worst}");
    }
    Ok(())
}

/// Every page ragged, which is what a real reconstruction looks like: each turn
/// ends wherever its tokens ended.
#[test]
fn every_page_ragged_matches_the_oracle() -> Result<()> {
    let _g = gpu().lock().unwrap();
    let d = dev()?;
    let mut rng = Lcg(77);
    let mut pages = Vec::new();
    let mut pos = 0usize;
    // Turn lengths that are deliberately NOT multiples of the ratio.
    for (i, &tokens) in [13usize, 7, 26, 5, 41, 2, 18].iter().enumerate() {
        let rows = tokens.div_ceil(RATIO);
        let last = tokens - (rows - 1) * RATIO;
        let keys = Tensor::from_vec(rng.vec(rows * HEAD_DIM), (rows, HEAD_DIM), &d)?;
        pages.push(IndexPage::new(keys, pos, last));
        pos += tokens;
        let _ = i;
    }
    let mut idx = PagedIndex::new(pages, RATIO, &d)?;
    let total = idx.total_tokens();
    let qpos: Vec<usize> = (0..total).collect();
    let q = queries(qpos.len(), 78)?;
    let worst = compare(&mut idx, &q, &qpos)?;
    assert!(worst < 2e-3, "worst relative error {worst}");
    Ok(())
}

/// A window must refuse a gap: the candidate prefix comes from the running
/// token total, so a page that does not continue the previous one would shift
/// every later row's visibility — silently, and in the retrieval rather than in
/// a crash.
#[test]
fn a_gap_between_pages_is_refused() -> Result<()> {
    let _g = gpu().lock().unwrap();
    let d = dev()?;
    let mut rng = Lcg(5);
    let mk = |rng: &mut Lcg, r: usize, first_pos: usize| -> Result<IndexPage> {
        Ok(IndexPage::new(
            Tensor::from_vec(rng.vec(r * HEAD_DIM), (r, HEAD_DIM), &d)?,
            first_pos,
            RATIO,
        ))
    };
    let a = mk(&mut rng, 4, 0)?;
    let b = mk(&mut rng, 4, 999)?; // should be 16
    assert!(
        PagedIndex::new(vec![a, b], RATIO, &d).is_err(),
        "a page starting past the previous page's end was accepted"
    );
    Ok(())
}

/// The turn record's round trip. Raw bytes, not a tolerance: this is a
/// serialization contract, and a drifted layout must fail here rather than
/// produce a plausible index on resume.
#[test]
fn a_page_round_trips_through_its_record_bytes() -> Result<()> {
    let _g = gpu().lock().unwrap();
    let d = dev()?;
    let mut rng = Lcg(91);
    let rows = 11usize;
    let vals = rng.vec(rows * HEAD_DIM);
    let keys = Tensor::from_vec(vals.clone(), (rows, HEAD_DIM), &d)?;
    let open_vals = rng.vec(2 * HEAD_DIM);
    let open = Tensor::from_vec(open_vals.clone(), (2, HEAD_DIM), &d)?;
    let blob = encode_page(&keys, 3, &open)?;
    let back = decode_page(&blob, &d)?;
    assert_eq!(back.page.rows()?, rows);
    assert_eq!(back.page.last_cells, 3);
    let got = back.page.keys.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(got, vals, "the page's keys did not survive the round trip");
    assert_eq!(
        back.open.flatten_all()?.to_vec1::<f32>()?,
        open_vals,
        "the open block did not survive — a resume would stand behind its own K/V"
    );
    Ok(())
}

/// The open block at **every** width it can hold, 0 through `MAX_RATIO`.
///
/// A turn ends where its text ends, so the remainder is uniformly distributed
/// over `0..ratio` and no single width is the representative case. Width 0 in
/// particular is the one a boundary-aligned fixture would test by accident
/// while proving nothing about the other three.
#[test]
fn an_open_block_of_every_width_round_trips() -> Result<()> {
    let _g = gpu().lock().unwrap();
    let d = dev()?;
    let mut rng = Lcg(93);
    for n_open in 0..=RATIO {
        let keys = Tensor::from_vec(rng.vec(5 * HEAD_DIM), (5, HEAD_DIM), &d)?;
        let open_vals = rng.vec(n_open * HEAD_DIM);
        let open = Tensor::from_vec(open_vals.clone(), (n_open, HEAD_DIM), &d)?;
        let back = decode_page(&encode_page(&keys, RATIO, &open)?, &d)?;
        assert_eq!(
            back.open.dim(0)?,
            n_open,
            "an open block of {n_open} rows came back as {} — the cache's \
             `n_blocks · ratio + n_open == tokens` no longer holds",
            back.open.dim(0)?
        );
        assert_eq!(back.open.flatten_all()?.to_vec1::<f32>()?, open_vals);
        assert_eq!(
            back.tokens(RATIO)?,
            5 * RATIO + n_open,
            "the record describes the wrong token count"
        );
    }
    Ok(())
}

/// A truncated blob is refused rather than reshaped into a plausible index.
#[test]
fn a_truncated_page_blob_is_refused() -> Result<()> {
    let _g = gpu().lock().unwrap();
    let d = dev()?;
    let mut rng = Lcg(92);
    let keys = Tensor::from_vec(rng.vec(8 * HEAD_DIM), (8, HEAD_DIM), &d)?;
    let open = Tensor::from_vec(rng.vec(3 * HEAD_DIM), (3, HEAD_DIM), &d)?;
    let blob = encode_page(&keys, RATIO, &open)?;
    assert!(decode_page(&blob[..blob.len() / 2], &d).is_err());
    // Losing only the open block is the interesting truncation: the completed
    // rows still decode, so a reader that did not check the declared open count
    // would hand back a page that looks whole and stands short.
    assert!(decode_page(&blob[..blob.len() - HEAD_DIM * 4], &d).is_err());
    Ok(())
}

/// **The composition property the whole design rests on.**
///
/// A window built from per-turn pages must see the same candidate prefix as one
/// built from a single page covering the same tokens — that is what makes
/// "logically reconstruct from the turns that were selected" mean anything. Only
/// the prefix is asserted here, not the key values: a flushed short block is
/// deliberately a summary of fewer tokens, so its key is not the key a
/// continuous run would have produced.
#[test]
fn per_turn_pages_expose_the_same_candidate_prefix_as_one_page() -> Result<()> {
    let _g = gpu().lock().unwrap();
    let d = dev()?;
    let mut rng = Lcg(303);
    // Three turns whose lengths are not multiples of the ratio, so every
    // boundary is ragged.
    let turns = [13usize, 7, 26];
    let total: usize = turns.iter().sum();

    let mut pages = Vec::new();
    let mut pos = 0usize;
    for &t in &turns {
        let rows = t.div_ceil(RATIO);
        pages.push(IndexPage::new(
            Tensor::from_vec(rng.vec(rows * HEAD_DIM), (rows, HEAD_DIM), &d)?,
            pos,
            t - (rows - 1) * RATIO,
        ));
        pos += t;
    }
    let paged = PagedIndex::new(pages, RATIO, &d)?;
    assert_eq!(paged.total_tokens(), total);

    // Every position: the number of rows visible must never exceed the rows
    // that exist, must be monotone, and must reach exactly the total at the end.
    let mut last = 0usize;
    for p in 0..total {
        let c = paged.candidates_at(p);
        assert!(
            c >= last,
            "candidate prefix went backwards at position {p}: {last} -> {c}"
        );
        assert!(
            c <= paged.total_rows(),
            "position {p} sees {c} rows but the window holds {}",
            paged.total_rows()
        );
        last = c;
    }
    assert_eq!(
        paged.candidates_at(total - 1),
        paged.total_rows(),
        "the last token must see every row — a page's short final row is still \
         wholly below it"
    );
    Ok(())
}

/// The carried-state container: PLE bytes and every layer's page, out exactly
/// as they went in. A serialization contract, so raw bytes rather than a
/// tolerance — a drifted layout must fail here rather than resume a
/// plausible-looking index.
#[test]
fn the_aux_container_round_trips_ple_and_every_page() -> Result<()> {
    let _g = gpu().lock().unwrap();
    let d = dev()?;
    let mut rng = Lcg(555);
    let ple: Vec<u8> = (0..37u8).collect();
    let layers = 12usize;
    let mut sealed = Vec::new();
    let mut expect = Vec::new();
    let mut expect_open = Vec::new();
    for i in 0..layers {
        let rows = 3 + i;
        let vals = rng.vec(rows * HEAD_DIM);
        expect.push(vals.clone());
        // A different open width per layer: the layers seal at one instant, but
        // they run at different ratios, so their remainders genuinely differ and
        // a container that stored one width for all of them would pass a
        // uniform fixture.
        let n_open = i % (RATIO + 1);
        let open_vals = rng.vec(n_open * HEAD_DIM);
        expect_open.push(open_vals.clone());
        sealed.push(SealedIndex {
            page: IndexPage::new(
                Tensor::from_vec(vals, (rows, HEAD_DIM), &d)?,
                0,
                1 + (i % RATIO),
            ),
            open: Tensor::from_vec(open_vals, (n_open, HEAD_DIM), &d)?,
        });
    }
    let blob = candle_transformers::models::qwen4exp::paged_index::encode_aux(&ple, &sealed)?;
    let (ple_back, back) =
        candle_transformers::models::qwen4exp::paged_index::decode_aux(&blob, &d)?;
    assert_eq!(ple_back, ple, "the PLE section did not survive");
    assert_eq!(back.len(), layers, "page count changed");
    for (i, s) in back.iter().enumerate() {
        assert_eq!(s.page.rows()?, 3 + i, "page {i} row count");
        assert_eq!(s.page.last_cells, 1 + (i % RATIO), "page {i} last_cells");
        assert_eq!(
            s.page.keys.flatten_all()?.to_vec1::<f32>()?,
            expect[i],
            "page {i} keys did not survive the round trip"
        );
        assert_eq!(
            s.open.flatten_all()?.to_vec1::<f32>()?,
            expect_open[i],
            "layer {i}'s open block did not survive the round trip"
        );
    }
    Ok(())
}

/// A container truncated anywhere is refused rather than yielding a short page
/// list — a missing page is an attention layer with no candidates at all.
#[test]
fn a_truncated_aux_container_is_refused() -> Result<()> {
    let _g = gpu().lock().unwrap();
    let d = dev()?;
    let mut rng = Lcg(556);
    let sealed = vec![SealedIndex {
        page: IndexPage::new(
            Tensor::from_vec(rng.vec(4 * HEAD_DIM), (4, HEAD_DIM), &d)?,
            0,
            RATIO,
        ),
        open: Tensor::from_vec(rng.vec(2 * HEAD_DIM), (2, HEAD_DIM), &d)?,
    }];
    let blob = candle_transformers::models::qwen4exp::paged_index::encode_aux(&[1, 2, 3], &sealed)?;
    for cut in [4usize, 16, blob.len() / 2, blob.len() - 1] {
        assert!(
            candle_transformers::models::qwen4exp::paged_index::decode_aux(&blob[..cut], &d)
                .is_err(),
            "a container truncated at {cut} bytes decoded without error"
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

/// Median of `iters` timed launches, CUDA-event timed so host launch overhead
/// and the final sync stay out of the number.
#[cfg(feature = "cuda")]
fn bench_once(idx: &mut PagedIndex, q: &Tensor, qpos: &[usize], iters: usize) -> Result<f64> {
    use candle::cuda_backend::cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
    let d = dev()?;
    let n = idx.total_rows();
    let out = Tensor::zeros((qpos.len(), n), DType::F32, &d)?;
    let Device::Cuda(cuda) = &d else {
        candle::bail!("bench runs on CUDA")
    };
    let stream = cuda.cuda_stream();
    // Warm: first launch pays module load and the descriptor upload.
    for _ in 0..5 {
        idx.score_rows(q, qpos, N_HEADS, HEAD_DIM, &out, n, 0)?;
    }
    d.synchronize()?;
    let mut ms: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = stream
            .record_event(Some(CU_EVENT_DEFAULT))
            .map_err(|e| candle::Error::Msg(format!("event: {e}")))?;
        idx.score_rows(q, qpos, N_HEADS, HEAD_DIM, &out, n, 0)?;
        let stop = stream
            .record_event(Some(CU_EVENT_DEFAULT))
            .map_err(|e| candle::Error::Msg(format!("event: {e}")))?;
        d.synchronize()?;
        ms.push(
            start
                .elapsed_ms(&stop)
                .map_err(|e| candle::Error::Msg(format!("elapsed: {e}")))? as f64,
        );
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(ms[ms.len() / 2])
}

/// The optimisation loop's instrument.
///
/// Shapes span the two regimes the scorer actually meets: **decode**, where the
/// row count is the handful of sessions that selected this step and every bit of
/// parallelism has to come from the candidate axis, and **prefill**, where rows
/// are wide. Depth is swept to 128K tokens, where the candidate axis is tens of
/// thousands of rows and the page count is a real conversation's worth of turns.
///
/// Run with:
/// `cargo test --release --features cuda -p candle-transformers --test
///  qsa_paged_index_tests bench_paged_scorer -- --ignored --nocapture`
#[test]
#[ignore = "benchmark; run with --ignored --nocapture"]
fn bench_paged_scorer() -> Result<()> {
    let _g = gpu().lock().unwrap();
    println!(
        "\n{:>8} {:>8} {:>7} {:>10} {:>12} {:>12}",
        "tokens", "rows", "pages", "queries", "median ms", "Gcell/s"
    );
    // (tokens, pages, query rows)
    let shapes = [
        (8_192usize, 16usize, 1usize),
        (8_192, 16, 8),
        (32_768, 64, 1),
        (32_768, 64, 8),
        (131_072, 256, 1),
        (131_072, 256, 8),
        (131_072, 256, 64),
    ];
    for (tokens, pages, t) in shapes {
        let per = tokens / pages;
        let rows_each = per.div_ceil(RATIO);
        let counts: Vec<usize> = vec![rows_each; pages];
        let mut idx = window(&counts, RATIO, 1234)?;
        let n = idx.total_rows();
        let total = idx.total_tokens();
        let qpos: Vec<usize> = (0..t).map(|i| total.saturating_sub(1 + i)).collect();
        let q = queries(t, 4321)?;
        let ms = bench_once(&mut idx, &q, &qpos, 50)?;
        let cells = (t * n) as f64;
        println!(
            "{tokens:>8} {n:>8} {pages:>7} {t:>10} {ms:>12.3} {:>12.2}",
            cells / (ms * 1.0e6)
        );
    }
    println!();
    Ok(())
}
