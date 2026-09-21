//! The QSA scorer's rotate-on-load: correctness against an f64 oracle, and the
//! benchmark its optimisation rounds are measured with
//! (`docs/progressive_yarn.md` §7.5–§7.6).
//!
//! The index stores its keys un-rotated and the scorer rotates each one as it
//! loads it, at the key's own position, from the factored RoPE table. These
//! tests drive the kernel through its FFI directly, so every descriptor — page
//! layout, strides, signed offset — is the test's to choose, and the oracle
//! shares nothing with the kernel but the frequencies.
//!
//! # The bound is derived, not tuned
//!
//! Each cell is `Σ_h relu(q_h · rot(k))`. A rotated key element carries the
//! table's lookup error on each of its two terms plus the rotation's own
//! roundings, and a `D`-term f32 dot product carries at most `γ_D ≈ D·u` of
//! its absolute sum, `u = 2⁻²⁴`. So a cell may differ from the f64 truth by at
//! most `(D + 16)·u · Σ_h Σ_c |q_hc|·|k_c|`, and ReLU never widens a gap. A
//! mis-rotated key — wrong position, wrong pair, wrong frequency — moves a cell
//! by order one, many thousands of times that.
//!
//! # Running it
//!
//! ```text
//! cargo test --release --features cuda -p candle-transformers \
//!     --test qsa_score_rot_harness -- --test-threads=1
//! cargo test --release --features cuda -p candle-transformers \
//!     --test qsa_score_rot_harness -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use std::sync::{Mutex, MutexGuard, OnceLock};

use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::{DType, Device, Result, Tensor};
use candle_kernels::simple::qsa_score_paged::{run_qsa_score_paged, PAGE_WORDS};
use candle_transformers::models::qwen4exp::config::IndexerConfig;
use candle_transformers::models::qwen4exp::indexer::{
    append_wave, AppendSpan, IndexCache, TailRoute,
};
use candle_transformers::models::qwen4exp::paged_index::IndexPage;
use candle_transformers::models::qwen4exp::qsa::IndexerWeights;
use candle_transformers::models::rope_schedule::{
    plain_inv_freq, FactoredRope, RopeRungs, RopeSchedule, Rung, ROPE_REACH,
};

/// The released indexer geometry.
const D: usize = 128;
const H: usize = 4;
const RATIO: usize = 4;
const ROPE_DIM: usize = 64;
const THETA: f32 = 1e7;

/// One kernel launch at a time: the deep cases allocate hundreds of MB.
fn gpu_serial() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Deterministic values in `[-1, 1)` — a hash, not an RNG, so a failing case
/// reruns exactly.
fn seeded(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            x ^= x >> 33;
            x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            x ^= x >> 33;
            ((x >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn dev_ptr_f32(t: &Tensor) -> Result<u64> {
    let Device::Cuda(dev) = t.device() else {
        candle::bail!("harness tensors live on CUDA")
    };
    let stream = dev.cuda_stream();
    let (s, l) = t.storage_and_layout();
    let slice = match &*s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("expected CUDA f32"),
    }
    .slice(l.start_offset()..);
    let (p, _g) = slice.device_ptr(&stream);
    Ok(p)
}

fn dev_ptr_u32(t: &Tensor) -> Result<u64> {
    let Device::Cuda(dev) = t.device() else {
        candle::bail!("harness tensors live on CUDA")
    };
    let stream = dev.cuda_stream();
    let (s, l) = t.storage_and_layout();
    let slice = match &*s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("expected CUDA u32"),
    }
    .slice(l.start_offset()..);
    let (p, _g) = slice.device_ptr(&stream);
    Ok(p)
}

fn dev_ptr_i64(t: &Tensor) -> Result<u64> {
    let Device::Cuda(dev) = t.device() else {
        candle::bail!("harness tensors live on CUDA")
    };
    let stream = dev.cuda_stream();
    let (s, l) = t.storage_and_layout();
    let slice = match &*s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<i64>()?,
        _ => candle::bail!("expected CUDA i64"),
    }
    .slice(l.start_offset()..);
    let (p, _g) = slice.device_ptr(&stream);
    Ok(p)
}

/// How a page's keys sit in device memory.
#[derive(Clone, Copy, Debug)]
enum Layout {
    /// `[d/4, rows, 4]` — a placed page.
    Blocked,
    /// `[cap, d]`, rows beyond `rows` poisoned — the live tail, whose buffer
    /// is larger than what it holds.
    RowMajor { cap: usize },
}

/// One page as the test sees it: un-rotated keys, where it sits, how it is
/// laid out.
#[derive(Clone)]
struct Page {
    base: usize,
    rows: usize,
    last_cells: usize,
    /// `[rows, d]`, un-rotated.
    keys: Vec<f32>,
    layout: Layout,
}

impl Page {
    fn new(base: usize, rows: usize, last_cells: usize, seed: u64, d: usize) -> Self {
        Self {
            base,
            rows,
            last_cells,
            keys: seeded(rows * d, seed),
            layout: Layout::Blocked,
        }
    }

    fn tokens(&self) -> usize {
        (self.rows - 1) * RATIO + self.last_cells
    }
}

/// Pages laid out back to back from `start`, each `rows[i]` rows with the
/// given last-row width — ragged widths make later pages' offsets negative.
fn contiguous_pages(
    start: usize,
    rows: &[usize],
    last: &[usize],
    seed: u64,
    d: usize,
) -> Vec<Page> {
    let mut at = start;
    rows.iter()
        .zip(last)
        .enumerate()
        .map(|(i, (&r, &l))| {
            let p = Page::new(at, r, l, seed + i as u64, d);
            at += p.tokens();
            p
        })
        .collect()
}

/// Candidate rows wholly at or below `pos` — a row covers `[start, start +
/// width)`, and a row is visible once its whole span is.
fn candidates(pages: &[Page], pos: usize) -> u32 {
    let limit = pos + 1;
    let mut c = 0u32;
    for p in pages {
        for j in 0..p.rows {
            let width = if j + 1 == p.rows { p.last_cells } else { RATIO };
            if p.base + j * RATIO + width <= limit {
                c += 1;
            } else {
                return c;
            }
        }
    }
    c
}

/// The pages uploaded with their descriptor table.
struct Uploaded {
    _bufs: Vec<Tensor>,
    desc: Tensor,
    first: Tensor,
    n_rows: usize,
}

fn upload(pages: &[Page], d: usize, dev: &Device) -> Result<Uploaded> {
    let mut bufs = Vec::with_capacity(pages.len());
    let mut desc: Vec<i64> = Vec::with_capacity(pages.len() * PAGE_WORDS);
    let mut first: Vec<u32> = vec![0];
    let mut acc = 0usize;
    let d4 = d / 4;
    for p in pages {
        let (host, cstride, rstride) = match p.layout {
            Layout::Blocked => {
                let mut v = vec![0f32; p.rows * d];
                for j in 0..p.rows {
                    for c in 0..d4 {
                        for e in 0..4 {
                            v[(c * p.rows + j) * 4 + e] = p.keys[j * d + c * 4 + e];
                        }
                    }
                }
                (v, p.rows as i64, 1i64)
            }
            Layout::RowMajor { cap } => {
                let mut v = vec![f32::NAN; cap * d];
                v[..p.rows * d].copy_from_slice(&p.keys);
                (v, 1i64, d4 as i64)
            }
        };
        let n = host.len();
        let buf = Tensor::from_vec(host, n, dev)?;
        desc.push(dev_ptr_f32(&buf)? as i64);
        desc.push(cstride);
        desc.push(rstride);
        desc.push(p.base as i64 - (acc * RATIO) as i64);
        bufs.push(buf);
        acc += p.rows;
        first.push(acc as u32);
    }
    let n_desc = desc.len();
    let n_first = first.len();
    Ok(Uploaded {
        _bufs: bufs,
        desc: Tensor::from_vec(desc, n_desc, dev)?,
        first: Tensor::from_vec(first, n_first, dev)?,
        n_rows: acc,
    })
}

/// One launch of the scorer at rung `rung`. `pairs = 0` runs the same kernel
/// with no rotary width — the scorer as it was before keys were rotated on
/// load.
#[allow(clippy::too_many_arguments)]
fn launch(
    up: &Uploaded,
    q: &Tensor,
    cnt: &Tensor,
    rope: &FactoredRope,
    rung: u32,
    out: &Tensor,
    t: usize,
    h: usize,
    d: usize,
    pairs: usize,
    n_pages: usize,
) -> Result<()> {
    let Device::Cuda(dev) = q.device() else {
        candle::bail!("harness runs on CUDA")
    };
    let stream = dev.cuda_stream();
    let table = rope.table(rung)?;
    unsafe {
        run_qsa_score_paged(
            dev_ptr_f32(q)? as *const f32,
            dev_ptr_i64(&up.desc)? as *const i64,
            dev_ptr_u32(&up.first)? as *const u32,
            dev_ptr_u32(cnt)? as *const u32,
            dev_ptr_f32(&table)? as *const f32,
            dev_ptr_f32(rope.steps(rung, RATIO)?)? as *const f32,
            dev_ptr_f32(out)? as *mut f32,
            t as i32,
            h as i32,
            d as i32,
            up.n_rows as i32,
            n_pages as i32,
            pairs as i32,
            RATIO as i32,
            up.n_rows as i64,
            0,
            stream.cu_stream() as *mut std::ffi::c_void,
        );
    }
    Ok(())
}

/// Score `q` (`[t, h, d]`, already rotated) against `pages` with the kernel,
/// at rung `rung`.
fn score(
    pages: &[Page],
    q: &[f32],
    qpos: &[usize],
    rope: &FactoredRope,
    rung: u32,
    h: usize,
    d: usize,
) -> Result<Vec<f32>> {
    let dev = rope.table(rung)?.device().clone();
    let up = upload(pages, d, &dev)?;
    let t = qpos.len();
    let cnt: Vec<u32> = qpos.iter().map(|&p| candidates(pages, p)).collect();
    let cnt = Tensor::from_vec(cnt, t, &dev)?;
    let q = Tensor::from_vec(q.to_vec(), (t * h, d), &dev)?;
    let out = Tensor::zeros((t, up.n_rows.max(1)), DType::F32, &dev)?;
    launch(
        &up,
        &q,
        &cnt,
        rope,
        rung,
        &out,
        t,
        h,
        d,
        rope.pairs(),
        pages.len(),
    )?;
    out.flatten_all()?.to_vec1::<f32>()
}

/// A key rotated at `pos` in f64, from the exact angle.
fn rotate_f64(key: &[f32], pos: usize, inv_freq: &[f32]) -> Vec<f64> {
    let pairs = inv_freq.len();
    let mut out: Vec<f64> = key.iter().map(|&x| x as f64).collect();
    for (i, &w) in inv_freq.iter().enumerate() {
        let (s, c) = (pos as f64 * w as f64).sin_cos();
        let (lo, hi) = (key[i] as f64, key[i + pairs] as f64);
        out[i] = lo * c - hi * s;
        out[i + pairs] = hi * c + lo * s;
    }
    out
}

/// The oracle and its per-cell bound: `(scores, bounds)`, `[t, n]` each.
fn oracle(
    pages: &[Page],
    q: &[f32],
    qpos: &[usize],
    inv_freq: &[f32],
    h: usize,
    d: usize,
) -> (Vec<f64>, Vec<f64>) {
    let u = f32::EPSILON as f64 / 2.0;
    let mut keys: Vec<Vec<f64>> = Vec::new();
    for p in pages {
        for j in 0..p.rows {
            keys.push(rotate_f64(
                &p.keys[j * d..(j + 1) * d],
                p.base + j * RATIO,
                inv_freq,
            ));
        }
    }
    let n = keys.len();
    let t = qpos.len();
    let mut s = vec![-1e30f64; t * n];
    let mut b = vec![0f64; t * n];
    for (r, &pos) in qpos.iter().enumerate() {
        let valid = candidates(pages, pos) as usize;
        for (g, k) in keys.iter().enumerate().take(valid) {
            let mut sum = 0f64;
            let mut mag = 0f64;
            for hh in 0..h {
                let qb = (r * h + hh) * d;
                let mut dot = 0f64;
                for c in 0..d {
                    dot += q[qb + c] as f64 * k[c];
                    mag += (q[qb + c] as f64 * k[c]).abs();
                }
                sum += dot.max(0.0);
            }
            s[r * n + g] = sum;
            b[r * n + g] = (d as f64 + 16.0) * u * mag + 1e-12;
        }
    }
    (s, b)
}

/// Assert the kernel's scores are the oracle's within each cell's bound, and
/// that masked cells are exactly the mask value.
fn assert_matches(got: &[f32], want: &[f64], bound: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for i in 0..got.len() {
        if want[i] <= -1e29 {
            assert_eq!(
                got[i].to_bits(),
                (-1e30f32).to_bits(),
                "{what}: cell {i} is masked in the oracle but the kernel wrote {}",
                got[i]
            );
            continue;
        }
        let e = (got[i] as f64 - want[i]).abs();
        assert!(
            e <= bound[i],
            "{what}: cell {i} off by {e:e} (kernel {} against {}) — bound {:e}",
            got[i],
            want[i],
            bound[i]
        );
    }
}

fn rope(theta: f32, rope_dim: usize, dev: &Device) -> Result<(FactoredRope, Vec<f32>)> {
    let inv = plain_inv_freq(rope_dim, theta);
    Ok((FactoredRope::new(&inv, dev)?, inv))
}

fn check(
    pages: &[Page],
    qpos: &[usize],
    h: usize,
    d: usize,
    theta: f32,
    rope_dim: usize,
    what: &str,
) -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let (table, inv) = rope(theta, rope_dim, &dev)?;
    let q = seeded(qpos.len() * h * d, 0x51 ^ qpos.len() as u64);
    let got = score(pages, &q, qpos, &table, 0, h, d)?;
    let (want, bound) = oracle(pages, &q, qpos, &inv, h, d);
    assert_matches(&got, &want, &bound, what);
    Ok(())
}

// ── Correctness ──────────────────────────────────────────────────────────────

/// The production geometry against the f64 oracle: ragged pages, holes between
/// them, and a page deep in the table's reach.
#[test]
fn rot_scorer_matches_the_f64_oracle() -> Result<()> {
    let _g = gpu_serial();
    let mut pages = contiguous_pages(0, &[40, 7, 129], &[3, 1, 4], 0x10, D);
    // A hole, then a page near the far end of the reach.
    let deep = ROPE_REACH - 64 * RATIO - 8;
    pages.push(Page::new(deep, 64, RATIO, 0x20, D));
    let end = deep + 64 * RATIO;
    let qpos = [5usize, 200, 700, deep + 30, end - 1, end + 5];
    for &t in &[1usize, 4, 8, 64] {
        let q: Vec<usize> = qpos.iter().copied().cycle().take(t).collect();
        check(&pages, &q, H, D, THETA, ROPE_DIM, &format!("t {t}"))?;
    }
    Ok(())
}

/// Page boundaries at lane 0, lane 31, mid-warp and at block edges, pages of
/// 1, 31, 32, 33 and 257 rows, so warps and CPT slots straddle pages.
#[test]
fn pages_straddle_warps_and_blocks() -> Result<()> {
    let _g = gpu_serial();
    let rows = [1usize, 31, 32, 33, 257, 1, 255, 256, 17];
    let last = [RATIO, 2, RATIO, 1, RATIO, 3, RATIO, RATIO, 2];
    let pages = contiguous_pages(3, &rows, &last, 0x30, D);
    let end = pages.last().unwrap().base + pages.last().unwrap().tokens();
    let qpos: Vec<usize> = (0..16).map(|i| end * (i + 1) / 16).collect();
    check(&pages, &qpos, H, D, THETA, ROPE_DIM, "straddling pages")
}

/// Ragged pages make later offsets `delta = base − first·ratio` negative; a
/// page placed with a gap makes it positive. Both, in one launch.
#[test]
fn signed_page_offsets() -> Result<()> {
    let _g = gpu_serial();
    let mut pages = contiguous_pages(0, &[9, 9, 9], &[1, 1, 1], 0x40, D);
    let mut acc = 0usize;
    let mut negative = false;
    for p in &pages {
        negative |= (p.base as i64) < (acc * RATIO) as i64;
        acc += p.rows;
    }
    assert!(negative, "the fixture must produce a negative page offset");
    pages.push(Page::new(50_000, 12, RATIO, 0x41, D));
    check(
        &pages,
        &[3, 20, 60, 50_020, 60_000],
        H,
        D,
        THETA,
        ROPE_DIM,
        "signed offsets",
    )
}

/// Candidates whose positions cross `hi` boundaries (`pos mod 1024` wrapping)
/// inside one warp.
#[test]
fn positions_cross_hi_rows() -> Result<()> {
    let _g = gpu_serial();
    let pages = vec![
        Page::new(1000, 40, RATIO, 0x50, D),
        Page::new(2040, 16, RATIO, 0x51, D),
        Page::new(1_048_560, 12, RATIO, 0x52, D),
    ];
    check(
        &pages,
        &[1100, 2100, 1_048_700],
        H,
        D,
        THETA,
        ROPE_DIM,
        "hi-row crossings",
    )
}

/// Every tile arm, at the production width: small windows drive the grid-fill
/// rule to the narrow arms, wide ones to `(4, 2)`.
#[test]
fn every_tile_arm() -> Result<()> {
    let _g = gpu_serial();
    for &n_pages in &[1usize, 64] {
        let rows: Vec<usize> = vec![250; n_pages];
        let last: Vec<usize> = vec![RATIO; n_pages];
        let pages = contiguous_pages(0, &rows, &last, 0x60 + n_pages as u64, D);
        let end = n_pages * 250 * RATIO;
        for &t in &[1usize, 2, 3, 4, 5, 8] {
            let qpos: Vec<usize> = (0..t).map(|i| end - 1 - i * 7).collect();
            check(
                &pages,
                &qpos,
                H,
                D,
                THETA,
                ROPE_DIM,
                &format!("pages {n_pages}, t {t}"),
            )?;
        }
    }
    // Every compiled head count.
    let pages = contiguous_pages(0, &[300, 200], &[RATIO, 2], 0x70, D);
    for &h in &[1usize, 2, 8] {
        check(
            &pages,
            &[1500, 1998],
            h,
            D,
            THETA,
            ROPE_DIM,
            &format!("h {h}"),
        )?;
    }
    Ok(())
}

/// A geometry the templated arms do not take — a rotary width that is not a
/// whole `float4` group on each side, and an odd head count — goes to the
/// generic arm and still matches.
#[test]
fn the_generic_arm_rotates_too() -> Result<()> {
    let _g = gpu_serial();
    let (d, rope_dim) = (16usize, 4usize);
    let pages = contiguous_pages(0, &[20, 13], &[2, RATIO], 0x80, d);
    check(&pages, &[10, 60, 200], 3, d, 1e6, rope_dim, "generic arm")
}

/// The same keys scored with two frequency sets each match their own oracle —
/// a launch rotates from the table it is given and nothing else.
#[test]
fn frequencies_are_the_launchs_own() -> Result<()> {
    let _g = gpu_serial();
    let pages = contiguous_pages(0, &[64, 64], &[RATIO, RATIO], 0x90, D);
    check(&pages, &[100, 511], H, D, 1e7, ROPE_DIM, "theta 1e7")?;
    check(&pages, &[100, 511], H, D, 1e4, ROPE_DIM, "theta 1e4")
}

/// **The rungs.** The same keys scored at each rung of a progressive schedule
/// each match the oracle at that rung's frequencies: a launch rotates from its
/// sequence's rung table and step table, not a neighbour's — and the rungs
/// differ enough that a wrong one would miss the bound by orders of magnitude.
#[test]
fn rungs_1_2_4() -> Result<()> {
    let _g = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let schedule = RopeSchedule::yarn(
        ROPE_DIM,
        THETA,
        262_144,
        vec![
            Rung {
                ceiling: 262_144,
                factor: 1.0,
            },
            Rung {
                ceiling: 524_288,
                factor: 2.0,
            },
            Rung {
                ceiling: 1_010_000,
                factor: 4.0,
            },
        ],
        true,
    )?;
    let rungs = RopeRungs::new(&schedule, &dev)?;
    let table = FactoredRope::over(&rungs, &dev)?;
    let freqs = schedule.rungs();
    // Ragged pages spanning a warp's worth of runs, one deep past rung 1's
    // ceiling, where the rungs' rotations are furthest apart.
    let mut pages = contiguous_pages(0, &[40, 7, 33], &[3, 1, RATIO], 0xC0, D);
    pages.push(Page::new(900_000, 64, RATIO, 0xC1, D));
    let qpos = [100usize, 250, 900_100, 900_255];
    let q = seeded(qpos.len() * H * D, 0xC2);
    for (r, f) in freqs.iter().enumerate() {
        let got = score(&pages, &q, &qpos, &table, r as u32, H, D)?;
        let (want, bound) = oracle(&pages, &q, &qpos, &f.inv_freq, H, D);
        assert_matches(&got, &want, &bound, &format!("rung {r}"));
    }
    // A wrong rung is a miss, not a near miss: rung 2's scores against rung 0's
    // oracle break the bound.
    let got = score(&pages, &q, &qpos, &table, 2, H, D)?;
    let (want, bound) = oracle(&pages, &q, &qpos, &freqs[0].inv_freq, H, D);
    assert!(
        got.iter()
            .zip(want.iter().zip(&bound))
            .any(|(&g, (&w, &b))| w > -1e29 && (g as f64 - w).abs() > b),
        "rung 2's rotation is indistinguishable from rung 0's — the test proves nothing"
    );
    Ok(())
}

/// A live tail in its row-major buffer, larger than the rows it holds (the
/// rows beyond are NaN), scores the same as the same rows placed.
#[test]
fn tail_page_with_capacity_pitch() -> Result<()> {
    let _g = gpu_serial();
    let mut pages = contiguous_pages(0, &[48, 30], &[RATIO, 3], 0xA0, D);
    let mut tail = Page::new(pages[1].base + pages[1].tokens(), 70, RATIO, 0xA1, D);
    tail.layout = Layout::RowMajor { cap: 128 };
    pages.push(tail);
    let end = pages[2].base + pages[2].tokens();
    check(
        &pages,
        &[40, 250, end - 1, end + 3],
        H,
        D,
        THETA,
        ROPE_DIM,
        "row-major tail",
    )
}

/// The production path end to end, both tail routes: pages pushed and placed,
/// a live tail appended by the kernel, scored through `IndexCache` by the
/// paged scorer and by the GEMM, each against the f64 oracle over the same
/// stored rows.
#[test]
fn both_tail_routes_match_the_oracle() -> Result<()> {
    let _g = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let (table, inv) = rope(THETA, ROPE_DIM, &dev)?;
    let cfg = IndexerConfig {
        n_heads: H,
        head_dim: D,
        top_k: 2048,
    };
    let w = IndexerWeights {
        q_proj: Tensor::zeros((H * D, 8), DType::F32, &dev)?,
        k_proj: Tensor::zeros((D, 8), DType::F32, &dev)?,
        q_norm: Tensor::ones(D, DType::F32, &dev)?,
        k_norm: Tensor::from_vec(seeded(D, 0xB0).iter().map(|v| v + 1.5).collect(), D, &dev)?,
    };

    // Two pages, then a live tail of 300 appended tokens.
    let pages = contiguous_pages(0, &[40, 23], &[RATIO, 2], 0xB1, D);
    let mut cache = IndexCache::new(D, &dev)?;
    for p in &pages {
        let keys = Tensor::from_vec(p.keys.clone(), (p.rows, D), &dev)?;
        cache.push_page(IndexPage::new(keys, p.last_cells), p.base, RATIO)?;
    }
    cache.place_pending()?;
    let tail_tokens = 300usize;
    let raw = Tensor::from_vec(seeded(tail_tokens * D, 0xB2), (tail_tokens, D), &dev)?;
    let mut work = [AppendSpan {
        cache: &mut cache,
        start: 0,
        rows: tail_tokens,
    }];
    append_wave(&mut work, &raw, &w, RATIO, 1e-6)?;

    // The oracle's view: the pages, then the tail's stored rows where they sit.
    let tail_rows = cache.live_rows()?;
    let n_tail = tail_rows.dim(0)?;
    let mut all = pages.clone();
    all.push(Page {
        base: cache.page_token_span(),
        rows: n_tail,
        last_cells: RATIO,
        keys: tail_rows.flatten_all()?.to_vec1::<f32>()?,
        layout: Layout::Blocked,
    });

    let end = cache.page_token_span() + n_tail * RATIO;
    for &t in &[1usize, 8, 64, 65, 300] {
        let qpos: Vec<usize> = (0..t).map(|i| end - 1 - (i * 13) % end).collect();
        let qv = seeded(t * H * D, 0xB3 ^ t as u64);
        let q = Tensor::from_vec(qv.clone(), (t, H, D), &dev)?;
        let (want, bound) = oracle(&all, &qv, &qpos, &inv, H, D);
        let n = all.iter().map(|p| p.rows).sum::<usize>();
        for route in [TailRoute::Paged, TailRoute::Gemm] {
            let out = Tensor::full(-7.0f32, (t, n), &dev)?;
            let cand = cache.score_rows_routed(
                &q,
                &qpos,
                &cfg,
                RATIO,
                &table,
                0,
                &out,
                n,
                0,
                |_, _| route,
            )?;
            let got = out.flatten_all()?.to_vec1::<f32>()?;
            // Columns past a row's candidates are the scorer's to leave alone on
            // the GEMM route, so compare only what each row can see.
            for r in 0..t {
                let valid = cand[r] as usize;
                assert_eq!(
                    valid as u32,
                    candidates(&all, qpos[r]),
                    "row {r} candidates"
                );
                let lo = r * n;
                assert_matches(
                    &got[lo..lo + valid],
                    &want[lo..lo + valid],
                    &bound[lo..lo + valid],
                    &format!("{route:?}, t {t}, row {r}"),
                );
            }
        }
    }
    Ok(())
}

// ── Benchmark ────────────────────────────────────────────────────────────────

/// Median and p90 of `iters` CUDA-event-timed runs of `f`, in ms.
fn time_ms<F: FnMut() -> Result<()>>(dev: &Device, iters: usize, mut f: F) -> Result<(f64, f64)> {
    use candle::cuda_backend::cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
    let Device::Cuda(cuda) = dev else {
        candle::bail!("bench runs on CUDA")
    };
    let stream = cuda.cuda_stream();
    for _ in 0..20 {
        f()?;
    }
    dev.synchronize()?;
    let mut ms = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = stream
            .record_event(Some(CU_EVENT_DEFAULT))
            .map_err(|e| candle::Error::Msg(format!("event: {e}")))?;
        f()?;
        let stop = stream
            .record_event(Some(CU_EVENT_DEFAULT))
            .map_err(|e| candle::Error::Msg(format!("event: {e}")))?;
        dev.synchronize()?;
        ms.push(
            start
                .elapsed_ms(&stop)
                .map_err(|e| candle::Error::Msg(format!("elapsed: {e}")))? as f64,
        );
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((ms[ms.len() / 2], ms[ms.len() * 9 / 10]))
}

/// The sweep the optimisation rounds are measured with: depth × query rows ×
/// pages, the rotating scorer against the same kernel with no rotary width.
#[test]
#[ignore = "benchmark; run with --ignored --nocapture"]
fn bench_rot_scorer() -> Result<()> {
    let _g = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let (table, _) = rope(THETA, ROPE_DIM, &dev)?;
    println!(
        "\n{:>9} {:>7} {:>6} {:>9} {:>11} {:>11} {:>11} {:>8} {:>10}",
        "tokens", "pages", "rows", "keys MB", "rot ms", "rot p90", "norot ms", "rot/no", "Gcell/s"
    );
    for &tokens in &[8_192usize, 32_768, 131_072, 524_288, 1_048_576] {
        for &n_pages in &[16usize, 256] {
            let rows_each = (tokens / n_pages).div_ceil(RATIO);
            let rows: Vec<usize> = vec![rows_each; n_pages];
            let last: Vec<usize> = vec![RATIO; n_pages];
            let pages = contiguous_pages(0, &rows, &last, 0x1234, D);
            let up = upload(&pages, D, &dev)?;
            let n = up.n_rows;
            let end = n * RATIO;
            for &t in &[1usize, 8, 64, 512, 2048, 8192] {
                // The score buffer is `t × n` floats; past 2 GB the sweep skips.
                if t * n * 4 > 2 << 30 {
                    continue;
                }
                let qpos: Vec<usize> = (0..t).map(|i| end - 1 - i).collect();
                let cnt: Vec<u32> = qpos.iter().map(|&p| candidates(&pages, p)).collect();
                let cnt = Tensor::from_vec(cnt, t, &dev)?;
                let q = Tensor::from_vec(seeded(t * H * D, 7), (t * H, D), &dev)?;
                let out = Tensor::zeros((t, n), DType::F32, &dev)?;
                let (rot, rot90) = time_ms(&dev, 50, || {
                    launch(
                        &up,
                        &q,
                        &cnt,
                        &table,
                        0,
                        &out,
                        t,
                        H,
                        D,
                        table.pairs(),
                        n_pages,
                    )
                })?;
                let (norot, _) = time_ms(&dev, 50, || {
                    launch(&up, &q, &cnt, &table, 0, &out, t, H, D, 0, n_pages)
                })?;
                println!(
                    "{tokens:>9} {n_pages:>7} {t:>6} {:>9.1} {rot:>11.4} {rot90:>11.4} {norot:>11.4} {:>8.3} {:>10.2}",
                    (n * D * 4) as f64 / 1e6,
                    rot / norot,
                    (t * n) as f64 / (rot * 1e6),
                );
            }
        }
    }
    println!();
    Ok(())
}

/// The live tail's two routes at the same row counts — where the paged scorer
/// stops beating the GEMM is [`GEMM_TAIL_MIN_CELLS`], rows × live-tail blocks.
#[test]
#[ignore = "benchmark; run with --ignored --nocapture"]
fn bench_tail_routes() -> Result<()> {
    let _g = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let (table, _) = rope(THETA, ROPE_DIM, &dev)?;
    let cfg = IndexerConfig {
        n_heads: H,
        head_dim: D,
        top_k: 2048,
    };
    let w = IndexerWeights {
        q_proj: Tensor::zeros((H * D, 8), DType::F32, &dev)?,
        k_proj: Tensor::zeros((D, 8), DType::F32, &dev)?,
        q_norm: Tensor::ones(D, DType::F32, &dev)?,
        k_norm: Tensor::ones(D, DType::F32, &dev)?,
    };
    println!(
        "\n{:>9} {:>6} {:>11} {:>11} {:>8}",
        "tokens", "rows", "paged ms", "gemm ms", "winner"
    );
    for &tokens in &[8_192usize, 32_768, 65_536, 131_072, 262_144] {
        let mut cache = IndexCache::new(D, &dev)?;
        let raw = Tensor::from_vec(seeded(tokens * D, 0xC0), (tokens, D), &dev)?;
        let mut work = [AppendSpan {
            cache: &mut cache,
            start: 0,
            rows: tokens,
        }];
        append_wave(&mut work, &raw, &w, RATIO, 1e-6)?;
        let n = cache.live_blocks();
        for &t in &[1usize, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
            if t * n * 4 > 2 << 30 {
                continue;
            }
            let qpos: Vec<usize> = (0..t).map(|i| tokens - 1 - i).collect();
            let q = Tensor::from_vec(seeded(t * H * D, 9), (t, H, D), &dev)?;
            let out = Tensor::zeros((t, n), DType::F32, &dev)?;
            let run = |route| {
                cache
                    .score_rows_routed(&q, &qpos, &cfg, RATIO, &table, 0, &out, n, 0, |_, _| route)
                    .map(|_| ())
            };
            let (paged, _) = time_ms(&dev, 30, || run(TailRoute::Paged))?;
            let (gemm, _) = time_ms(&dev, 30, || run(TailRoute::Gemm))?;
            println!(
                "{tokens:>9} {t:>6} {paged:>11.4} {gemm:>11.4} {:>8}",
                if paged <= gemm { "paged" } else { "gemm" }
            );
        }
    }
    println!();
    Ok(())
}

/// One launch of one shape — the `ncu` target for an optimisation round.
///
/// ```text
/// ncu --set full --kernel-name regex:qsa_score_paged_kernel --launch-count 1 \
///     <this binary> --ignored --exact bench_rot_scorer_profile_one
/// ```
#[test]
#[ignore = "profiling target; run under ncu"]
fn bench_rot_scorer_profile_one() -> Result<()> {
    let _g = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let (table, _) = rope(THETA, ROPE_DIM, &dev)?;
    let tokens = 131_072usize;
    let n_pages = 256usize;
    let t = 64usize;
    let rows_each = (tokens / n_pages).div_ceil(RATIO);
    let pages = contiguous_pages(
        0,
        &vec![rows_each; n_pages],
        &vec![RATIO; n_pages],
        0x1234,
        D,
    );
    let up = upload(&pages, D, &dev)?;
    let n = up.n_rows;
    let end = n * RATIO;
    let qpos: Vec<usize> = (0..t).map(|i| end - 1 - i).collect();
    let cnt: Vec<u32> = qpos.iter().map(|&p| candidates(&pages, p)).collect();
    let cnt = Tensor::from_vec(cnt, t, &dev)?;
    let q = Tensor::from_vec(seeded(t * H * D, 7), (t * H, D), &dev)?;
    let out = Tensor::zeros((t, n), DType::F32, &dev)?;
    launch(
        &up,
        &q,
        &cnt,
        &table,
        0,
        &out,
        t,
        H,
        D,
        table.pairs(),
        n_pages,
    )?;
    dev.synchronize()?;
    Ok(())
}
