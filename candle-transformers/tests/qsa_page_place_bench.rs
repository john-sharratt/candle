//! Correctness gates + tuning harness for `qsa_page_place`.
//!
//! The kernel gives a position-free index page the position it is being placed
//! at: one constant RoPE rotation by `placement_base − roped_base`, fused into
//! the pass that builds the scorer's channel-blocked `[head_dim/4, rows, 4]`
//! staging. It replaces a `reshape → transpose → contiguous` chain that could
//! not carry a rotation at all.
//!
//! # What is gated, and why in this file
//!
//! The kernel is pure bandwidth — `rows · head_dim` floats in, the same out —
//! so it is tuned, and a tuned kernel needs its correctness pinned against a
//! reference that does not move while the tuning does. Both live here so one
//! command answers both questions:
//!
//! - **`delta = 0` is a pure transpose.** Bit-identical to the chain it
//!   replaces. This is the live decode path (a page closed in the cache it is
//!   placed in), so it is the one that must not drift.
//! - **`delta = Δ` is the rotation.** Bit-comparable to
//!   `RopeTables::apply_at_positions` at `Δ` followed by the same transpose.
//! - **Rotations compose.** Placing at `a` then at `b` equals placing at
//!   `a + b`, which is the identity the whole placement design rests on.
//! - **Every tile is the same kernel.** Each `tile_r` arm must agree with the
//!   default bit for bit; a tuning change cannot become a numerics change.
//!
//! # Running it
//!
//! ```text
//! cargo test --release --features cuda -p candle-transformers \
//!     --test qsa_page_place_bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `bench_qsa_page_place_tile_sweep` says which shape is slow;
//! `bench_qsa_page_place_every_shape` launches each one exactly once so `ncu`
//! can say why, and `bench_qsa_page_place_plan_cost` reports the half of a
//! placement that is not the kernel. The sweep prints the `ncu` line.
//!
//! Where it stands, measured on an RTX PRO 5000 Blackwell: the two shapes that
//! actually stream run at **87–88% of peak DRAM** with 95–97% warp occupancy and
//! 40 registers a thread. That is the memory floor, and it is why the row tile
//! only separates the shapes small enough to sit in L2.

#![cfg(feature = "cuda")]

use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use candle::{Device, Result, Tensor};
use candle_transformers::models::qwen35::attention::RopeTables;
use candle_transformers::models::qwen4exp::place::{
    place_pages, PlacePage, Placement, PLACE_TILE_R,
};

/// The released indexer geometry (`qsa_index_bench`): 128-wide keys, a 64-wide
/// rotary within them, one block key per 4 tokens.
const HEAD_DIM: usize = 128;
const ROPE_DIM: usize = 64;
const ROPE_THETA: f32 = 10_000.0;
const RATIO: usize = 4;

/// Full-attention layers in the released checkpoint — a placement rebuilds
/// every page on every one of them, so this is the natural job multiplier.
const ATTN_LAYERS: usize = 12;

/// The tiles the kernel is compiled for. Every one must be numerically
/// identical; only their speed differs.
const TILES: [usize; 4] = [8, 16, 32, 64];

/// One GPU at a time: the deep cases allocate tens of MB per page per layer.
fn gpu_serial() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Deterministic values in a stable range — a hash, not an RNG, so a rerun on
/// another machine compares against the same numbers.
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

fn page_keys(rows: usize, seed: u64, dev: &Device) -> Result<Tensor> {
    Tensor::from_vec(seeded(rows * HEAD_DIM, seed), (rows, HEAD_DIM), dev)
}

fn rope_for(max_pos: usize, dev: &Device) -> Result<RopeTables> {
    RopeTables::new(ROPE_DIM, ROPE_THETA, max_pos.max(4096), dev)
}

/// The staging the kernel replaces: `[rows, d] → [d/4, rows, 4]`, no rotation.
///
/// Verbatim the chain `IndexPage::blocked` used to run, so "bit-identical to
/// what it replaces" is a comparison against the actual former expression.
fn reference_blocked(keys: &Tensor) -> Result<Tensor> {
    let (rows, dim) = keys.dims2()?;
    keys.reshape((rows, dim / 4, 4))?
        .transpose(0, 1)?
        .contiguous()
}

/// The reference placement: rotate every row by `delta`, then stage it.
///
/// Positive only — the host rope indexes its tables by position and has no way
/// to spell a rotation the other way, which is exactly why the kernel carries
/// the sign itself. The negative direction is gated by
/// [`a_placement_and_its_inverse_return_the_page`] instead.
fn reference_placed(keys: &Tensor, delta: usize, rope: &RopeTables) -> Result<Tensor> {
    let (rows, dim) = keys.dims2()?;
    let positions = vec![delta; rows];
    let rotated = rope
        .apply_at_positions(&keys.reshape((rows, 1, dim))?, &positions)?
        .reshape((rows, dim))?;
    reference_blocked(&rotated)
}

fn to_vec(t: &Tensor) -> Result<Vec<f32>> {
    t.flatten_all()?.to_vec1::<f32>()
}

/// Bit-for-bit. Used where the two sides really do the same arithmetic in the
/// same order — a zero-delta placement against the transpose it replaces, and
/// one row tile against another.
fn assert_bits(got: &Tensor, want: &Tensor, what: &str) -> Result<()> {
    let g = to_vec(got)?;
    let w = to_vec(want)?;
    assert_eq!(g.len(), w.len(), "{what}: length");
    for (i, (a, b)) in g.iter().zip(&w).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what}: element {i} — kernel {a} against reference {b}",
        );
    }
    Ok(())
}

/// Absolute error a rotation may carry against the host reference.
///
/// **Absolute, not relative, and that is the point.** The rotation is
/// `lo·cos − hi·sin` over keys in `[-1, 1]` and table entries in `[-1, 1]`, so
/// every output is a difference of two products of order one: when it lands near
/// zero the cancellation is catastrophic in the relative sense and says nothing
/// about the arithmetic. A few ULP of ONE is the honest bound, and it is still
/// tight — a mis-paired channel or a mis-indexed table row moves an output by
/// order one, five million times this.
///
/// The kernel is not bit-identical to the host chain and cannot be: the archive
/// compiles `--use_fast_math`, so nvcc contracts `lo·co − hi·si` into an FMA
/// while the host rounds the product out first. Measured worst case across these
/// gates is 6e-8.
const ROTATION_EPS: f32 = 2e-7;

/// Same shape as [`assert_bits`], to the rotation's error bound.
fn assert_close(got: &Tensor, want: &Tensor, what: &str) -> Result<()> {
    let g = to_vec(got)?;
    let w = to_vec(want)?;
    assert_eq!(g.len(), w.len(), "{what}: length");
    let mut worst = (0f32, 0usize);
    for (i, (a, b)) in g.iter().zip(&w).enumerate() {
        let e = (a - b).abs();
        if e > worst.0 {
            worst = (e, i);
        }
    }
    assert!(
        worst.0 <= ROTATION_EPS,
        "{what}: element {} off by {} (kernel {} against reference {}) — bound is {ROTATION_EPS}",
        worst.1,
        worst.0,
        g[worst.1],
        w[worst.1],
    );
    Ok(())
}

// ── Correctness ──────────────────────────────────────────────────────────────

/// **The live decode path.** A page closed in the cache it is placed in has
/// delta 0, and must come out as the transpose it always was — the rotation is
/// `x·cos 0 − y·sin 0`, which is `x` exactly.
#[test]
fn a_zero_delta_placement_is_the_transpose_it_replaces() -> Result<()> {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let rope = rope_for(4096, &dev)?;
    for &rows in &[1usize, 7, 64, 501] {
        let keys = page_keys(rows, 0xA11CE ^ rows as u64, &dev)?;
        let got = place_pages(
            &[PlacePage {
                keys: &keys,
                delta: 0,
            }],
            &rope,
            PLACE_TILE_R,
        )?;
        assert_bits(&got[0], &reference_blocked(&keys)?, &format!("rows {rows}"))?;
    }
    Ok(())
}

/// The rotation itself, against the host rope the reference implementation
/// uses. Deltas that are not multiples of `ratio` are included deliberately: a
/// placement base is a token position, and nothing rounds it to a block.
#[test]
fn a_placement_rotates_every_row_by_the_delta() -> Result<()> {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let rope = rope_for(65_536, &dev)?;
    for &delta in &[1usize, 4, 37, 2048, 32_771] {
        for &rows in &[3usize, 128] {
            let keys = page_keys(rows, 0xBEEF ^ delta as u64, &dev)?;
            let got = place_pages(
                &[PlacePage {
                    keys: &keys,
                    delta: delta as isize,
                }],
                &rope,
                PLACE_TILE_R,
            )?;
            assert_close(
                &got[0],
                &reference_placed(&keys, delta, &rope)?,
                &format!("delta {delta}, rows {rows}"),
            )?;
        }
    }
    Ok(())
}

/// **The seal's direction, and the round trip that pins it.**
///
/// A live cache ropes its blocks at their absolute positions, so turning a page
/// into a record means rotating it *back* to zero — a negative delta, which the
/// host rope cannot express and the tables cannot index. The kernel takes the
/// sign itself (`cos(-x) = cos(x)`, `sin(-x) = -sin(x)`), and the property that
/// says it did so correctly is that `+Δ` then `−Δ` is the identity.
///
/// Without this the seal would be normalising with a rotation nothing checks,
/// and every record on disk would carry the error.
///
/// **The bound is the tables', measured, not a tolerance.** A round trip is two
/// rotations, and it returns the page exactly only insofar as `cos²Δ + sin²Δ`
/// is one — which in `f32` tables it is not. That residual, times the key
/// magnitude, is the floor, and the slack above it is the kernel's own.
#[test]
fn a_placement_and_its_inverse_return_the_page() -> Result<()> {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let rope = rope_for(65_536, &dev)?;
    let (cos_t, sin_t) = rope.tables();
    for &delta in &[1isize, 4, 37, 2048, 32_771] {
        let rows = 96;
        let keys = page_keys(rows, 0xFEED ^ delta as u64, &dev)?;
        let unblock = |t: &Tensor| -> Result<Tensor> {
            t.reshape((HEAD_DIM / 4, rows, 4))?
                .transpose(0, 1)?
                .reshape((rows, HEAD_DIM))?
                .contiguous()
        };
        let fwd = place_pages(&[PlacePage { keys: &keys, delta }], &rope, PLACE_TILE_R)?;
        let at_delta = unblock(&fwd[0])?;
        let back = place_pages(
            &[PlacePage {
                keys: &at_delta,
                delta: -delta,
            }],
            &rope,
            PLACE_TILE_R,
        )?;

        // How far this delta's row is from a unit rotation, per rotary pair.
        let row = |t: &Tensor| -> Result<Vec<f32>> {
            t.narrow(0, delta as usize, 1)?
                .flatten_all()?
                .to_vec1::<f32>()
        };
        let (c, s) = (row(cos_t)?, row(sin_t)?);
        let unit_resid = (0..c.len())
            .map(|m| (c[m] * c[m] + s[m] * s[m] - 1.0).abs())
            .fold(0f32, f32::max);
        // Each output is a two-term combination of keys bounded by one, so the
        // residual can enter it twice; the rest is the two rotations' rounding.
        let bound = 2.0 * unit_resid + 2.0 * ROTATION_EPS;
        let g = to_vec(&unblock(&back[0])?)?;
        let w = to_vec(&keys)?;
        let worst = g
            .iter()
            .zip(&w)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            worst <= bound,
            "round trip through ±{delta} left {worst}, past the {bound} the tables allow \
             (unit residual {unit_resid})",
        );
    }
    Ok(())
}

/// **The identity the design rests on.** RoPE rotations compose additively, so
/// a page placed at `a` and then re-placed by a further `b` is the page placed
/// at `a + b`. This is what lets a sealed page be moved anywhere with one
/// rotation instead of being re-derived from the hidden states it came from.
///
/// **The bound is the TABLES', not the kernel's, and the test measures it
/// rather than guessing at a tolerance.** `cos a · cos b − sin a · sin b` equals
/// `cos(a + b)` in exact arithmetic; in the stored `f32` tables it does not,
/// because each entry is `cos(pos · inv_freq)` rounded, and at `pos = 710` with
/// `inv_freq = 1` the angle alone carries ~4e-5 radians of representation error.
/// So the identity's floor is set by the three table rows involved, and the
/// gate asserts the kernel does not exceed it by more than the key magnitude
/// allows. A tolerance picked by hand here would be a number chosen to make the
/// test pass; this one is derived from the data the test is about.
#[test]
fn two_placements_compose_into_one() -> Result<()> {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let rope = rope_for(16_384, &dev)?;
    let (a, b, rows) = (97usize, 613usize, 96usize);
    let keys = page_keys(rows, 0xC0FFEE, &dev)?;

    // What the tables themselves say the identity is worth, per rotary pair.
    let row = |p: usize, t: &Tensor| -> Result<Vec<f32>> {
        t.narrow(0, p, 1)?.flatten_all()?.to_vec1::<f32>()
    };
    let (cos_t, sin_t) = rope.tables();
    let (ca, sa) = (row(a, cos_t)?, row(a, sin_t)?);
    let (cb, sb) = (row(b, cos_t)?, row(b, sin_t)?);
    let (cab, sab) = (row(a + b, cos_t)?, row(a + b, sin_t)?);
    let table_resid = (0..ca.len())
        .map(|m| {
            let dc = (ca[m] * cb[m] - sa[m] * sb[m] - cab[m]).abs();
            let ds = (sa[m] * cb[m] + ca[m] * sb[m] - sab[m]).abs();
            dc.max(ds)
        })
        .fold(0f32, f32::max);

    // Place at `a`, read the rows back out of the staging, place again at `b`.
    let once = place_pages(
        &[PlacePage {
            keys: &keys,
            delta: a as isize,
        }],
        &rope,
        PLACE_TILE_R,
    )?;
    let rows_at_a = once[0]
        .reshape((HEAD_DIM / 4, rows, 4))?
        .transpose(0, 1)?
        .reshape((rows, HEAD_DIM))?
        .contiguous()?;
    let twice = place_pages(
        &[PlacePage {
            keys: &rows_at_a,
            delta: b as isize,
        }],
        &rope,
        PLACE_TILE_R,
    )?;

    let direct = place_pages(
        &[PlacePage {
            keys: &keys,
            delta: (a + b) as isize,
        }],
        &rope,
        PLACE_TILE_R,
    )?;

    let g = to_vec(&twice[0])?;
    let w = to_vec(&direct[0])?;
    let worst = g
        .iter()
        .zip(&w)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    // Each output is a two-term combination of keys bounded by 1, so the
    // table's per-pair residual can enter it twice; the slack past that is the
    // kernel's own, and it is a few ULP.
    let bound = 2.0 * table_resid + ROTATION_EPS;
    assert!(
        worst <= bound,
        "composing {a} then {b} diverged from {} by {worst}, past the {bound} the tables \
         themselves allow (per-pair residual {table_resid})",
        a + b,
    );
    Ok(())
}

/// A placement is batched over every (page, layer) job, and the pages are not
/// all the same width. The narrow jobs' spare row-tiles must retire without
/// writing anything, and every job must land in its own buffer.
#[test]
fn a_batch_of_unequal_pages_each_get_their_own_placement() -> Result<()> {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let rope = rope_for(8192, &dev)?;
    let widths = [1usize, 33, 512, 7, 129];
    // Mixed directions, because a real batch carries both: a projection placing
    // pages rotates forward, a seal normalising them rotates back.
    let deltas = [0isize, 64, -1000, 3, -4095];
    let keys: Vec<Tensor> = widths
        .iter()
        .enumerate()
        .map(|(i, &r)| page_keys(r, 0x5EED + i as u64, &dev))
        .collect::<Result<_>>()?;
    let pages: Vec<PlacePage<'_>> = keys
        .iter()
        .zip(&deltas)
        .map(|(k, &delta)| PlacePage { keys: k, delta })
        .collect();

    let got = place_pages(&pages, &rope, PLACE_TILE_R)?;
    assert_eq!(got.len(), widths.len());
    // Against the SAME page placed on its own, bit for bit. What is in question
    // here is the batching — that a narrow job's spare row tiles write nothing,
    // that each job lands in its own slice of the arena, and that a job reads
    // its own delta and not a neighbour's. The arithmetic itself is gated
    // against the host above, and in the negative direction by the round trip;
    // comparing to a single-job run is what isolates the batching from both.
    for (i, ((k, &delta), out)) in keys.iter().zip(&deltas).zip(&got).enumerate() {
        let alone = place_pages(&[PlacePage { keys: k, delta }], &rope, PLACE_TILE_R)?;
        assert_bits(out, &alone[0], &format!("job {i} (delta {delta})"))?;
    }
    Ok(())
}

/// **Tuning may not become a numerics change.**
///
/// Every compiled tile runs the identical expression — the tile is a work
/// partition, not an arithmetic choice — so they must agree with the default
/// **bit for bit**, and the sweep below is then free to pick any of them. This
/// is compared tile-against-tile rather than against the host precisely because
/// it is the stronger claim: the host comparison would only say each tile is
/// within the rotation's error bound, which two genuinely different kernels
/// could both satisfy.
#[test]
fn every_row_tile_computes_the_same_placement() -> Result<()> {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let rope = rope_for(8192, &dev)?;
    let rows = 777;
    let keys = page_keys(rows, 0x0D1A_1CE5, &dev)?;
    let page = |delta| PlacePage { keys: &keys, delta };
    let want = place_pages(&[page(1234)], &rope, PLACE_TILE_R)?;
    for &tile in &TILES {
        let got = place_pages(&[page(1234)], &rope, tile)?;
        assert_bits(&got[0], &want[0], &format!("tile_r {tile}"))?;
    }
    // And the default is one of the compiled arms, not a fifth path.
    assert!(
        TILES.contains(&PLACE_TILE_R),
        "PLACE_TILE_R {PLACE_TILE_R} is not one of the compiled tiles {TILES:?}",
    );
    Ok(())
}

// ── Tuning ───────────────────────────────────────────────────────────────────

/// One case's shape: how many pages of how many rows, across every attention
/// layer.
struct Case {
    label: &'static str,
    /// Pages the placement installs, per layer.
    pages: usize,
    /// Rows in each — `tokens / ratio`, so a 2,048-token turn is 512.
    rows: usize,
}

const CASES: [Case; 6] = [
    Case {
        label: "one short turn",
        pages: 1,
        rows: 128,
    },
    Case {
        label: "one long turn",
        pages: 1,
        rows: 2048,
    },
    Case {
        label: "a projection (8 turns)",
        pages: 8,
        rows: 512,
    },
    Case {
        label: "a projection (32 turns)",
        pages: 32,
        rows: 512,
    },
    Case {
        label: "many tiny pages",
        pages: 128,
        rows: 24,
    },
    Case {
        label: "one deep section",
        pages: 1,
        rows: 32_768,
    },
];

/// Time `iters` launches of one planned placement, returning ms per launch.
///
/// The plan is hoisted deliberately. A placement allocates one staging buffer
/// per page **per attention layer** and the kernel runs in tens of microseconds,
/// so timing plan-plus-launch measures the allocator: the first version of this
/// sweep reported 31 GB/s on the many-tiny-pages case and moved not at all
/// across the tiles, because 1,536 `Tensor::empty` calls were the whole figure.
/// That cost is real and [`bench_qsa_page_place_plan_cost`] reports it — but it
/// is not the thing a row tile changes.
fn time_launch(plan: &Placement, rope: &RopeTables, tile: usize, dev: &Device) -> Result<f64> {
    for _ in 0..3 {
        plan.run(rope, tile)?;
    }
    dev.synchronize()?;
    let iters = 50;
    let t = Instant::now();
    for _ in 0..iters {
        plan.run(rope, tile)?;
    }
    dev.synchronize()?;
    Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

/// **The sweep.** Every case at every tile, reported as achieved bandwidth —
/// the kernel reads `rows · head_dim` floats and writes the same, so GB/s
/// against the card's peak is the only figure that says whether it is done.
///
/// The winner per case is marked. If one tile wins everywhere, that is
/// [`PLACE_TILE_R`]; if the wide and narrow cases disagree, the note beside the
/// constant has to say which one it is chosen for.
#[test]
#[ignore = "tuning harness; run with --ignored --nocapture"]
fn bench_qsa_page_place_tile_sweep() -> Result<()> {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let rope = rope_for(262_144, &dev)?;

    println!("\n  qsa_page_place — row-tile sweep (ms per launch, GB/s achieved)");
    println!("  head_dim {HEAD_DIM}, rope_dim {ROPE_DIM}, ratio {RATIO}, {ATTN_LAYERS} layers");
    print!("  {:>24} {:>7} {:>7} {:>9}", "case", "pages", "rows", "MB");
    for t in TILES {
        print!(" {:>15}", format!("tile {t}"));
    }
    println!();

    for case in &CASES {
        let n_jobs = case.pages * ATTN_LAYERS;
        let keys: Vec<Tensor> = (0..n_jobs)
            .map(|i| page_keys(case.rows, 0x1234 + i as u64, &dev))
            .collect::<Result<_>>()?;
        // A real placement's deltas differ per page — the base each turn sits
        // at — so the cos/sin row a block stages differs per job too.
        let pages: Vec<PlacePage<'_>> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| PlacePage {
                keys: k,
                delta: ((i * case.rows * RATIO) % 200_000) as isize,
            })
            .collect();

        // Read plus write, which is all this kernel does.
        let bytes = 2.0 * (n_jobs * case.rows * HEAD_DIM * 4) as f64;
        print!(
            "  {:>24} {:>7} {:>7} {:>9.1}",
            case.label,
            case.pages,
            case.rows,
            bytes / 2.0 / 1e6,
        );
        let plan = Placement::plan(&pages)?;
        let mut best = (f64::MAX, 0usize);
        let mut ms_of = Vec::with_capacity(TILES.len());
        for &tile in &TILES {
            let ms = time_launch(&plan, &rope, tile, &dev)?;
            if ms < best.0 {
                best = (ms, tile);
            }
            ms_of.push(ms);
        }
        for (&tile, &ms) in TILES.iter().zip(&ms_of) {
            let gbs = bytes / (ms * 1e-3) / 1e9;
            let mark = if tile == best.1 { '*' } else { ' ' };
            print!(" {:>13}{}", format!("{ms:.3}/{gbs:.0}"), mark);
        }
        println!();
    }

    println!(
        "\n  profile one shape:\n    ncu --set full --kernel-name regex:place_kernel \\\n\
         \x20     --launch-skip 8 --launch-count 1 \\\n\
         \x20     <this binary> --ignored --exact \\\n\
         \x20     bench_qsa_page_place_every_shape\n"
    );
    Ok(())
}

/// **The other half of a placement, and on the small shapes the larger half.**
///
/// Planning allocates one staging buffer per page per attention layer and
/// resolves two device pointers for each. The sweep above hoists that so it can
/// see the kernel; this reports it, because a projection pays both and the
/// balance between them decides which one is worth attention next.
#[test]
#[ignore = "tuning harness; run with --ignored --nocapture"]
fn bench_qsa_page_place_plan_cost() -> Result<()> {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let rope = rope_for(262_144, &dev)?;

    println!("\n  qsa_page_place — plan against launch (ms)");
    println!(
        "  {:>24} {:>7} {:>9} {:>9} {:>9} {:>7}",
        "case", "jobs", "plan", "launch", "total", "plan %"
    );
    for case in &CASES {
        let n_jobs = case.pages * ATTN_LAYERS;
        let keys: Vec<Tensor> = (0..n_jobs)
            .map(|i| page_keys(case.rows, 0x1234 + i as u64, &dev))
            .collect::<Result<_>>()?;
        let pages: Vec<PlacePage<'_>> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| PlacePage {
                keys: k,
                delta: ((i * case.rows * RATIO) % 200_000) as isize,
            })
            .collect();

        // Planning allocates, so it is timed on its own with the previous
        // plan already dropped — otherwise the pool grows across iterations
        // and the later ones measure a different allocator state.
        let iters = 20;
        for _ in 0..3 {
            let _ = Placement::plan(&pages)?;
        }
        dev.synchronize()?;
        let t = Instant::now();
        for _ in 0..iters {
            let _ = Placement::plan(&pages)?;
        }
        dev.synchronize()?;
        let plan_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let plan = Placement::plan(&pages)?;
        let launch_ms = time_launch(&plan, &rope, PLACE_TILE_R, &dev)?;
        let total = plan_ms + launch_ms;
        println!(
            "  {:>24} {n_jobs:>7} {plan_ms:>9.3} {launch_ms:>9.3} {total:>9.3} {:>6.0}%",
            case.label,
            100.0 * plan_ms / total,
        );
    }
    Ok(())
}

/// **Every case, launched exactly once, in order** — the profiling target.
///
/// One launch per case is what makes this usable under `ncu`: there is no
/// warm-up to skip and no repetition to swamp the report, so
/// `--launch-count 6 --launch-skip 0` profiles all six and labels each by its
/// grid, `(row_tiles, jobs, 1)`. The sweep above says which one is slow; this
/// says why.
///
/// ```text
/// ncu --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
/// sm__warps_active.avg.pct_of_peak_sustained_active,\
/// launch__registers_per_thread,launch__waves_per_multiprocessor,\
/// l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
///     --kernel-name regex:place_kernel --launch-count 6 \
///     <this binary> --ignored --exact bench_qsa_page_place_every_shape
/// ```
#[test]
#[ignore = "profiling target; run under ncu"]
fn bench_qsa_page_place_every_shape() -> Result<()> {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let rope = rope_for(262_144, &dev)?;
    for case in &CASES {
        let n_jobs = case.pages * ATTN_LAYERS;
        let keys: Vec<Tensor> = (0..n_jobs)
            .map(|i| page_keys(case.rows, 0x1234 + i as u64, &dev))
            .collect::<Result<_>>()?;
        let pages: Vec<PlacePage<'_>> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| PlacePage {
                keys: k,
                delta: ((i * case.rows * RATIO) % 200_000) as isize,
            })
            .collect();
        let plan = Placement::plan(&pages)?;
        plan.run(&rope, PLACE_TILE_R)?;
        dev.synchronize()?;
        println!(
            "  launched: {:>24}  grid ({}, {n_jobs})",
            case.label,
            case.rows.div_ceil(PLACE_TILE_R),
        );
    }
    Ok(())
}
