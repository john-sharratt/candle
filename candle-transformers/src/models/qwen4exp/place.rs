//! Placing a QSA index page — the rotation that gives a position-free page one.
//!
//! An index page is a position-free artifact: its rows are one sealed unit's
//! pooled, normed, roped block keys, in that unit's own frame. Placing it in a
//! projection is what gives it a position, and RoPE rotations compose
//! additively — so the page enters the placement's frame through ONE further
//! rotation by the difference between the two.
//!
//! That difference is a constant over the whole page, so it is applied **once
//! per placement**, not once per score: the scorer's economy is that a key
//! `float4` is read once and reused across every query row of its tile, and a
//! per-candidate rotation would add a cos/sin row against a 512-byte key on
//! exactly the traffic that bounds it.
//!
//! The rotation rides inside the pass the placement had to make anyway. The
//! scorer reads keys channel-blocked as `[head_dim/4, rows, 4]`; building that
//! staging is one pass over the page, and this is that pass.

use candle::{DType, Result, Tensor};

use super::indexer::{i64_ptr, tensor_ptr};
use crate::models::qwen35::attention::RopeTables;

/// Rows per block in [`Placement::run`].
///
/// The kernel's only real dial, and it pulls two ways: it is the length of the
/// store phase's contiguous run and it multiplies the shared-memory footprint.
///
/// **Measured** (`tests/qsa_page_place_bench.rs`, RTX PRO 5000 Blackwell, three
/// runs, GB/s of read+write):
///
/// | case | tile 8 | tile 16 | tile 32 | tile 64 |
/// |---|---|---|---|---|
/// | 8 turns × 512 rows (25 MB, L2-resident) | 2145–2701 | 2864–3268 | 3216–3343 | 2712–2743 |
/// | 128 pages × 24 rows (19 MB) | 2429–2519 | 2955–3007 | 2880–3025 | 1980–2030 |
/// | 32 turns × 512 rows (101 MB) | 1114–1116 | 1104–1105 | 1101–1102 | 1105–1106 |
/// | 1 page × 32,768 rows (201 MB) | 1110–1112 | 1074–1105 | 1089–1105 | 1108–1109 |
///
/// The two DRAM-bound cases are flat across every tile — `ncu` puts them at
/// 87–88% of peak DRAM with 95–97% warp occupancy and 40 registers a thread, so
/// there is nothing there for a tile to move. The dial only shows up where the
/// working set fits L2, and there **8 is consistently worst and 64 loses a
/// third on many small pages**. Sixteen is never worst and asks half the shared
/// memory of 32 (8.4 KB against 16.9 KB), which is what holds the occupancy.
pub const PLACE_TILE_R: usize = 16;

/// One page to place.
pub struct PlacePage<'a> {
    /// `[rows, head_dim]` F32 — the page's prepared block keys, in the frame
    /// they were roped in.
    pub keys: &'a Tensor,
    /// The rotation to apply, `placement_base − roped_base`, in tokens.
    ///
    /// **Signed.** Positive gives a position-free page the position it is being
    /// placed at; negative takes a page that carries one back to zero, which is
    /// how a seal turns a live cache's rows into an artifact that can be
    /// injected anywhere. Zero is the live decode path — a page closed in the
    /// cache it already sits in — and the kernel then writes a pure transpose.
    pub delta: isize,
}

/// One placement's staging buffers and the descriptor table addressing them.
///
/// Split from the launch because the two have different lifetimes and very
/// different costs. A projection places its pages once and then scores against
/// them for every decode step until it changes, so the plan is amortised over
/// the whole projection while the launch is paid once.
///
/// **The staging is ONE allocation, sliced.** A placement covers every page on
/// every attention layer, and a projection carrying a hundred short pages is
/// over a thousand jobs — measured, per-page allocation was 1.21 ms of
/// `Tensor::empty` against 0.013 ms of kernel, 99% of a placement spent in the
/// allocator. The descriptor table already gives every job its own base pointer
/// (hot-path invariant 2b), so separate allocations bought nothing: one flat
/// buffer and a narrow per page addresses exactly the same bytes.
#[derive(Debug)]
pub struct Placement {
    staged: Vec<Tensor>,
    jobs: Tensor,
    n_jobs: usize,
    d: usize,
    max_rows: usize,
    /// The largest rotation magnitude any job asks for — checked against the
    /// rope tables at launch.
    max_delta: usize,
    /// The one buffer every entry of `staged` is a view into. Held so the views
    /// cannot outlive it.
    _arena: Option<Tensor>,
}

impl Placement {
    /// Allocate the staging and build the descriptor table that addresses it.
    #[cfg(feature = "cuda")]
    pub fn plan(pages: &[PlacePage<'_>]) -> Result<Self> {
        use candle_kernels::simple::qsa_page_place::PLACE_JOB_WORDS;

        if pages.is_empty() {
            return Ok(Self {
                staged: Vec::new(),
                jobs: Tensor::zeros((0,), DType::I64, &candle::Device::Cpu)?,
                n_jobs: 0,
                d: 0,
                max_rows: 0,
                max_delta: 0,
                _arena: None,
            });
        }
        let device = pages[0].keys.device().clone();
        let (_, d) = pages[0].keys.dims2()?;
        if d % 4 != 0 {
            candle::bail!(
                "index page placement: head_dim {d} is not a multiple of four — the scorer's \
                 channel-blocked staging is built in `float4` groups"
            );
        }

        // One pass for the geometry, so the arena can be sized before anything
        // is allocated.
        let mut rows_of = Vec::with_capacity(pages.len());
        let mut total_rows = 0usize;
        let mut max_rows = 0usize;
        for p in pages {
            let (rows, dim) = p.keys.dims2()?;
            if dim != d {
                candle::bail!(
                    "index page placement: page widths disagree — {dim} against {d}; every \
                     layer of a placement indexes one stream and shares its head_dim"
                );
            }
            rows_of.push(rows);
            total_rows += rows;
            max_rows = max_rows.max(rows);
        }

        // Fully overwritten by the kernel, so never zeroed (invariant 6).
        let arena = Tensor::empty((total_rows * d,), DType::F32, &device)?;
        let arena_base = tensor_ptr(&arena)?;
        let elem = std::mem::size_of::<f32>() as u64;

        let mut staged = Vec::with_capacity(pages.len());
        let mut jobs: Vec<i64> = Vec::with_capacity(pages.len() * PLACE_JOB_WORDS);
        let mut at = 0usize;
        let mut max_delta = 0usize;
        for (p, &rows) in pages.iter().zip(&rows_of) {
            // The destination address is arithmetic on the arena's base rather
            // than a second `tensor_ptr` per job: resolving a pointer walks the
            // storage and the layout, and at a thousand jobs that walk was
            // itself a measurable share of the plan.
            jobs.push(tensor_ptr(p.keys)? as i64);
            jobs.push((arena_base + (at * d) as u64 * elem) as i64);
            jobs.push(rows as i64);
            jobs.push(p.delta as i64);
            max_delta = max_delta.max(p.delta.unsigned_abs());
            // A contiguous narrow reshaped is a view, so this costs no copy.
            staged.push(
                arena
                    .narrow(0, at * d, rows * d)?
                    .reshape((d / 4, rows, 4))?,
            );
            at += rows;
        }
        let n_jobs = pages.len();
        let jobs = Tensor::from_vec(jobs, (n_jobs * PLACE_JOB_WORDS,), &device)?;
        Ok(Self {
            staged,
            jobs,
            n_jobs,
            d,
            max_rows,
            max_delta,
            _arena: Some(arena),
        })
    }

    /// Rotate every page into its placement frame — **one launch over all of
    /// them** (hot-path invariant 5).
    ///
    /// Idempotent: the kernel fully rewrites every staging buffer from the
    /// pages, so running twice is running once. `tile_r` selects the row tile;
    /// production passes [`PLACE_TILE_R`] and the bench sweeps the rest.
    #[cfg(feature = "cuda")]
    pub fn run(&self, rope: &RopeTables, tile_r: usize) -> Result<()> {
        use candle_kernels::simple::qsa_page_place::run_qsa_page_place;

        if self.n_jobs == 0 || self.max_rows == 0 {
            return Ok(());
        }
        // The tables are indexed by |delta|, and a row past their end is a read
        // off the end of the allocation — silent, and wrong by whatever happened
        // to be there. Checked here rather than at plan time because the plan
        // does not know which tables it will be run against.
        if self.max_delta >= rope.max_pos() {
            candle::bail!(
                "index page placement: a rotation of {} token(s) is past the {}-entry rope \
                 tables — the tables must span the deepest placement, not just the tail",
                self.max_delta,
                rope.max_pos(),
            );
        }
        let candle::Device::Cuda(cuda) = self.jobs.device() else {
            candle::bail!("index page placement runs on CUDA");
        };
        let stream = cuda.cuda_stream();
        let (cos, sin) = rope.table_ptrs()?;
        candle::set_kernel_breadcrumb("run_qsa_page_place", file!(), line!());
        unsafe {
            run_qsa_page_place(
                i64_ptr(&self.jobs)? as *const i64,
                cos as *const f32,
                sin as *const f32,
                self.d as i32,
                rope.rope_dim() as i32,
                self.n_jobs as i32,
                self.max_rows as i32,
                tile_r as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
        Ok(())
    }

    /// The staging buffers, in the order the pages were given.
    pub fn staged(&self) -> &[Tensor] {
        &self.staged
    }

    /// Take the staging buffers.
    pub fn into_staged(self) -> Vec<Tensor> {
        self.staged
    }
}

/// Plan and run — what a projection calls.
///
/// Returns one `[head_dim/4, rows, 4]` F32 tensor per input page, in order.
#[cfg(feature = "cuda")]
pub fn place_pages(
    pages: &[PlacePage<'_>],
    rope: &RopeTables,
    tile_r: usize,
) -> Result<Vec<Tensor>> {
    let placement = Placement::plan(pages)?;
    placement.run(rope, tile_r)?;
    Ok(placement.into_staged())
}

/// Rotate `keys` by `delta` and hand them back **row-major** — `[rows,
/// head_dim]`, the layout a record stores.
///
/// **The seal's direction.** A live cache ropes its blocks at their absolute
/// positions, so lifting rows out of one gives an artifact that carries a
/// position; passing `-(frame)` here takes it back to zero, which is what makes
/// the record injectable anywhere. [`Placement`] writes the scorer's
/// channel-blocked staging because that is what the scorer reads, so this
/// un-blocks it again — one extra pass over one page, on a path that runs once
/// per turn.
#[cfg(feature = "cuda")]
pub fn rotate_rows(keys: &Tensor, delta: isize, rope: &RopeTables) -> Result<Tensor> {
    let (rows, d) = keys.dims2()?;
    if rows == 0 || delta == 0 {
        return keys.to_owned_tensor();
    }
    let staged = place_pages(&[PlacePage { keys, delta }], rope, PLACE_TILE_R)?;
    staged[0]
        .reshape((d / 4, rows, 4))?
        .transpose(0, 1)?
        .reshape((rows, d))?
        .contiguous()
}
