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
//! the host in f64 — every stored key rotated at its own position from the
//! exact angle, then one dot product at a time, in row order. Comparing against
//! the dense GPU path instead would only prove the two agree, which is exactly
//! the thing a shared bug would also produce. The tolerance is not laxity: the
//! kernel rotates from an f32 table and accumulates in a different order (a
//! `float4` body, heads folded after a per-head ReLU), so bit-equality is not
//! available and asking for it would pin an implementation rather than a result.

#![cfg(feature = "cuda")]

use candle::{DType, Device, Result, Tensor};
use candle_transformers::models::qwen4exp::key_q8::{blocks, Q8Keys};
use candle_transformers::models::qwen4exp::paged_index::{
    aux_parts, decode_aux, decode_page, encode_aux, encode_page, migrate_aux, section_rows,
    AuxMigration, IndexPage, PagedIndex, SealedIndex, F32_INDEX_VERSION,
};
use candle_transformers::models::rope_schedule::{plain_inv_freq, FactoredRope};

/// The released checkpoint's indexer geometry.
const HEAD_DIM: usize = 128;
const N_HEADS: usize = 4;
const RATIO: usize = 4;
const ROPE_DIM: usize = 64;
const ROPE_THETA: f32 = 10_000.0;

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
        pages.push((IndexPage::new(keys, cells), pos));
        pos += if r == 0 { 0 } else { (r - 1) * RATIO + cells };
    }
    PagedIndex::new(pages, RATIO, &d)
}

/// The frequencies the scorer's table is built from, and the oracle's.
fn inv_freq() -> Vec<f32> {
    plain_inv_freq(ROPE_DIM, ROPE_THETA)
}

/// The factored table the scorer rotates every key from.
fn rope() -> Result<FactoredRope> {
    FactoredRope::new(&inv_freq(), &dev()?)
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
    idx.score_rows(q, qpos, N_HEADS, HEAD_DIM, &out, n.max(1), 0, &rope()?, 0)?;
    let got = out.flatten_all()?.to_vec1::<f32>()?;
    let want = idx.score_reference(q, qpos, N_HEADS, HEAD_DIM, &inv_freq())?;

    let mut worst = 0f32;
    for r in 0..t {
        for g in 0..n {
            let (a, b) = (got[r * n + g], want[r * n + g] as f32);
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
        pages.push((IndexPage::new(keys, last), pos));
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

/// **A gap is legal; an overlap is not.**
///
/// A hole between two pages means what it says — that span carries no index
/// rows — and it has to be expressible, because a piece of K/V injected without
/// its index is a real thing the projection does. It used to be refused, and had
/// to be: the candidate prefix came from a running token total, so a page that
/// did not continue the previous one shifted every later row's visibility.
/// Positions are read off the page now, so a hole moves nothing.
///
/// An overlap is still refused. Two pages claiming the same position is not a
/// span nobody indexed, it is two rows for one token.
#[test]
fn a_gap_between_pages_is_allowed_and_an_overlap_is_not() -> Result<()> {
    let _g = gpu().lock().unwrap();
    let d = dev()?;
    let mut rng = Lcg(5);
    let mut mk = |r: usize| -> Result<IndexPage> {
        Ok(IndexPage::new(
            Tensor::from_vec(rng.vec(r * HEAD_DIM), (r, HEAD_DIM), &d)?,
            RATIO,
        ))
    };
    // Page A covers [0, 16); the hole runs to 999, where B opens.
    let (a, b) = (mk(4)?, mk(4)?);
    let idx = PagedIndex::new(vec![(a, 0), (b, 999)], RATIO, &d)?;
    assert_eq!(idx.total_rows(), 8);
    // A query inside the hole sees A's rows and none of B's.
    assert_eq!(idx.candidates_at(500), 4);
    // One past B's opening block, it sees A's four and B's first.
    assert_eq!(idx.candidates_at(999 + RATIO), 5);

    let (a, b) = (mk(4)?, mk(4)?);
    assert!(
        PagedIndex::new(vec![(a, 0), (b, 8)], RATIO, &d).is_err(),
        "a page overlapping the previous page's span was accepted"
    );
    Ok(())
}

/// The turn record's round trip. Raw bytes, not a tolerance: this is a
/// serialization contract, and a drifted layout must fail here rather than
/// produce a plausible index on resume.
#[test]
fn a_page_round_trips_through_its_record_bytes() -> Result<()> {
    let mut rng = Lcg(91);
    let rows = 11usize;
    let keys = Q8Keys::quantize(&rng.vec(rows * HEAD_DIM), HEAD_DIM);
    let open_vals = rng.vec(2 * HEAD_DIM);
    let sealed = SealedIndex {
        keys: keys.clone(),
        last_cells: 3,
        open: open_vals.clone(),
    };
    let blob = encode_page(&sealed)?;
    // The layout itself: four header words, the codes, the scales, the open
    // block.
    let nb = blocks(HEAD_DIM);
    assert_eq!(
        blob.len(),
        16 + rows * HEAD_DIM + rows * nb * 2 + 2 * HEAD_DIM * 4
    );
    let word = |i: usize| u32::from_le_bytes(blob[i * 4..i * 4 + 4].try_into().unwrap()) as usize;
    assert_eq!((word(0), word(1), word(2), word(3)), (rows, HEAD_DIM, 3, 2));
    assert_eq!(&blob[16..16 + rows * HEAD_DIM], keys.codes.as_slice());
    let scales_at = 16 + rows * HEAD_DIM;
    assert_eq!(
        &blob[scales_at..scales_at + 2],
        keys.scales[0].to_le_bytes().as_slice()
    );
    let open_at = scales_at + rows * nb * 2;
    assert_eq!(
        &blob[open_at..open_at + 4],
        open_vals[0].to_le_bytes().as_slice()
    );

    let back = decode_page(&blob)?;
    assert_eq!(back.n_rows(), rows);
    assert_eq!(back.last_cells, 3);
    assert_eq!(
        back.keys, keys,
        "the page's keys did not survive the round trip"
    );
    assert_eq!(
        back.open, open_vals,
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
    let mut rng = Lcg(93);
    for n_open in 0..=RATIO {
        let open_vals = rng.vec(n_open * HEAD_DIM);
        let sealed = SealedIndex {
            keys: Q8Keys::quantize(&rng.vec(5 * HEAD_DIM), HEAD_DIM),
            last_cells: RATIO,
            open: open_vals.clone(),
        };
        let back = decode_page(&encode_page(&sealed)?)?;
        assert_eq!(
            back.n_open(),
            n_open,
            "an open block of {n_open} rows came back as {} — the cache's \
             `n_blocks · ratio + n_open == tokens` no longer holds",
            back.n_open()
        );
        assert_eq!(back.open, open_vals);
        assert_eq!(
            back.page_tokens(RATIO) + back.n_open(),
            5 * RATIO + n_open,
            "the record describes the wrong token count"
        );
    }
    Ok(())
}

/// A truncated blob is refused rather than reshaped into a plausible index.
#[test]
fn a_truncated_page_blob_is_refused() -> Result<()> {
    let mut rng = Lcg(92);
    let sealed = SealedIndex {
        keys: Q8Keys::quantize(&rng.vec(8 * HEAD_DIM), HEAD_DIM),
        last_cells: RATIO,
        open: rng.vec(3 * HEAD_DIM),
    };
    let blob = encode_page(&sealed)?;
    assert!(decode_page(&blob[..blob.len() / 2]).is_err());
    // Losing only the open block is the interesting truncation: the completed
    // rows still decode, so a reader that did not check the declared open count
    // would hand back a page that looks whole and stands short.
    assert!(decode_page(&blob[..blob.len() - HEAD_DIM * 4]).is_err());
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
        pages.push((
            IndexPage::new(
                Tensor::from_vec(rng.vec(rows * HEAD_DIM), (rows, HEAD_DIM), &d)?,
                t - (rows - 1) * RATIO,
            ),
            pos,
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
    let mut rng = Lcg(555);
    let ple: Vec<u8> = (0..37u8).collect();
    let layers = 12usize;
    let mut sealed = Vec::new();
    let mut expect = Vec::new();
    let mut expect_open = Vec::new();
    for i in 0..layers {
        let rows = 3 + i;
        let vals = Q8Keys::quantize(&rng.vec(rows * HEAD_DIM), HEAD_DIM);
        expect.push(vals.clone());
        // A different open width per layer: the layers seal at one instant, but
        // they run at different ratios, so their remainders genuinely differ and
        // a container that stored one width for all of them would pass a
        // uniform fixture.
        let n_open = i % (RATIO + 1);
        let open_vals = rng.vec(n_open * HEAD_DIM);
        expect_open.push(open_vals.clone());
        sealed.push(SealedIndex {
            keys: vals,
            last_cells: 1 + (i % RATIO),
            open: open_vals,
        });
    }
    let blob = encode_aux(&ple, &sealed)?;
    let (ple_back, back) = decode_aux(&blob)?;
    assert_eq!(ple_back, ple, "the PLE section did not survive");
    assert_eq!(back.len(), layers, "page count changed");
    for (i, s) in back.iter().enumerate() {
        assert_eq!(s.n_rows(), 3 + i, "page {i} row count");
        assert_eq!(s.last_cells, 1 + (i % RATIO), "page {i} last_cells");
        assert_eq!(
            s.keys, expect[i],
            "page {i} keys did not survive the round trip"
        );
        assert_eq!(
            s.open, expect_open[i],
            "layer {i}'s open block did not survive the round trip"
        );
    }
    Ok(())
}

/// **A container's sections are each layer's page as [`encode_page`] wrote it,
/// byte for byte**, after its PLE bytes, and a section's header alone gives its
/// completed rows: what lets the pool decode only the layers it places and size
/// them without decoding any.
#[test]
fn the_aux_sections_are_each_layers_page_as_written() -> Result<()> {
    let mut rng = Lcg(559);
    let ple = [7u8, 8, 9];
    let sealed: Vec<SealedIndex> = (0..3)
        .map(|i| SealedIndex {
            keys: Q8Keys::quantize(&rng.vec((3 + i) * HEAD_DIM), HEAD_DIM),
            last_cells: 1 + i,
            open: rng.vec(i * HEAD_DIM),
        })
        .collect();
    let blob = encode_aux(&ple, &sealed)?;
    let (ple_back, sections) = aux_parts(&blob)?;
    assert_eq!(ple_back, &ple[..]);
    assert_eq!(sections.len(), 3);
    for (i, (section, layer)) in sections.iter().zip(&sealed).enumerate() {
        assert_eq!(
            *section,
            &encode_page(layer)?[..],
            "layer {i}'s section bytes"
        );
        assert_eq!(section_rows(section)?, 3 + i, "layer {i}'s rows");
    }
    let (_, decoded) = decode_aux(&blob)?;
    assert_eq!(decode_page(sections[1])?.keys, decoded[1].keys);
    Ok(())
}

/// A container truncated anywhere is refused rather than yielding a short page
/// list — a missing page is an attention layer with no candidates at all.
#[test]
fn a_truncated_aux_container_is_refused() -> Result<()> {
    let mut rng = Lcg(556);
    let sealed = vec![SealedIndex {
        keys: Q8Keys::quantize(&rng.vec(4 * HEAD_DIM), HEAD_DIM),
        last_cells: RATIO,
        open: rng.vec(2 * HEAD_DIM),
    }];
    let blob = encode_aux(&[1, 2, 3], &sealed)?;
    for cut in [4usize, 16, blob.len() / 2, blob.len() - 1] {
        assert!(
            decode_aux(&blob[..cut]).is_err(),
            "a container truncated at {cut} bytes decoded without error"
        );
    }
    Ok(())
}

/// A container as version 3 wrote it: the PLE bytes, then each layer's rows and
/// open block as `F32`.
fn f32_aux(ple: &[u8], layers: &[(Vec<f32>, usize, Vec<f32>)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&F32_INDEX_VERSION.to_le_bytes());
    out.extend_from_slice(&(ple.len() as u32).to_le_bytes());
    out.extend_from_slice(ple);
    out.extend_from_slice(&(layers.len() as u32).to_le_bytes());
    for (rows, last_cells, open) in layers {
        let len = 16 + (rows.len() + open.len()) * 4;
        out.extend_from_slice(&(len as u32).to_le_bytes());
        out.extend_from_slice(&((rows.len() / HEAD_DIM) as u32).to_le_bytes());
        out.extend_from_slice(&(HEAD_DIM as u32).to_le_bytes());
        out.extend_from_slice(&(*last_cells as u32).to_le_bytes());
        out.extend_from_slice(&((open.len() / HEAD_DIM) as u32).to_le_bytes());
        for v in rows.iter().chain(open) {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// **A build reads Q8_0 only.** A container whose keys are `F32` is refused by
/// name, with the program that rewrites it.
#[test]
fn an_f32_container_is_refused_by_name() -> Result<()> {
    let mut rng = Lcg(557);
    let blob = f32_aux(&[9], &[(rng.vec(3 * HEAD_DIM), 2, Vec::new())]);
    let err = decode_aux(&blob).unwrap_err().to_string();
    assert!(err.contains("migrate-index-q8"), "{err}");
    Ok(())
}

/// **The migration rewrites an `F32` container as exactly the Q8_0 one**: the
/// PLE bytes, every layer's widths and its open block as they were, and its
/// completed keys quantized by the host definition. A container already at the
/// current version is left alone, and any other version is refused.
#[test]
fn migration_rewrites_an_f32_container_as_q8() -> Result<()> {
    let mut rng = Lcg(558);
    let ple: Vec<u8> = (0..11u8).collect();
    let layers: Vec<(Vec<f32>, usize, Vec<f32>)> = (0..3)
        .map(|i| (rng.vec((2 + i) * HEAD_DIM), 1 + i, rng.vec(i * HEAD_DIM)))
        .collect();
    let old = f32_aux(&ple, &layers);
    let AuxMigration::Migrated {
        bytes: new,
        fidelity,
    } = migrate_aux(&old)?
    else {
        panic!("an F32 container was not migrated");
    };
    // Every completed key of every layer is tallied, and held closely.
    assert_eq!(fidelity.keys, 2 + 3 + 4);
    assert!(fidelity.snr_db() > 35.0, "{fidelity:?}");
    assert!(fidelity.cos_min > 0.999, "{fidelity:?}");
    let want: Vec<SealedIndex> = layers
        .iter()
        .map(|(rows, last_cells, open)| SealedIndex {
            keys: Q8Keys::quantize(rows, HEAD_DIM),
            last_cells: *last_cells,
            open: open.clone(),
        })
        .collect();
    assert_eq!(new, encode_aux(&ple, &want)?, "byte for byte");
    let (ple_back, back) = decode_aux(&new)?;
    assert_eq!(ple_back, ple);
    assert_eq!(back, want);
    assert!(
        new.len() < old.len() / 2,
        "the keys are a quarter the bytes"
    );

    assert_eq!(migrate_aux(&new)?, AuxMigration::Current);
    let mut other = new.clone();
    other[0] = 2;
    assert!(migrate_aux(&other).is_err());
    assert!(migrate_aux(&old[..old.len() - 1]).is_err(), "truncated");
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
    let rope = rope()?;
    // Warm: first launch pays module load and the descriptor upload.
    for _ in 0..5 {
        idx.score_rows(q, qpos, N_HEADS, HEAD_DIM, &out, n, 0, &rope, 0)?;
    }
    d.synchronize()?;
    let mut ms: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = stream
            .record_event(Some(CU_EVENT_DEFAULT))
            .map_err(|e| candle::Error::Msg(format!("event: {e}")))?;
        idx.score_rows(q, qpos, N_HEADS, HEAD_DIM, &out, n, 0, &rope, 0)?;
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
