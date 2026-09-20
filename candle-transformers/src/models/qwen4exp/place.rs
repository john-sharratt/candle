//! Placing a QSA index page — laying its rows out the way the scorer reads them.
//!
//! An index page's rows are one sealed unit's pooled, normed, **un-rotated**
//! block keys, `[rows, head_dim]` as a record holds them. The paged scorer reads
//! keys channel-blocked as `[head_dim/4, rows, 4]`, so a warp's key read is one
//! contiguous run; placing a page writes that layout once, rather than once per
//! score.
//!
//! Placing carries no rotation and no position. The scorer rotates each key as
//! it loads it, at the key's own position, so a page's placement is the same
//! wherever it sits and whatever RoPE schedule reads it.

use candle::{DType, Result, Tensor};

use super::indexer::{i64_ptr, tensor_ptr};

/// Rows per block in [`Placement::run`].
///
/// The kernel's only real dial, and it pulls two ways: it is the length of the
/// store phase's contiguous run and it multiplies the shared-memory footprint.
///
/// **Measured** (`tests/qsa_page_place_bench.rs`, RTX PRO 5000 Blackwell, three
/// runs, GB/s of read+write, with the rotation the pass carried at the time):
///
/// | case | tile 8 | tile 16 | tile 32 | tile 64 |
/// |---|---|---|---|---|
/// | 8 turns × 512 rows (25 MB, L2-resident) | 2145–2701 | 2864–3268 | 3216–3343 | 2712–2743 |
/// | 128 pages × 24 rows (19 MB) | 2429–2519 | 2955–3007 | 2880–3025 | 1980–2030 |
/// | 32 turns × 512 rows (101 MB) | 1114–1116 | 1104–1105 | 1101–1102 | 1105–1106 |
/// | 1 page × 32,768 rows (201 MB) | 1110–1112 | 1074–1105 | 1089–1105 | 1108–1109 |
///
/// The two DRAM-bound cases are flat across every tile. The dial only shows up
/// where the working set fits L2, and there **8 is consistently worst and 64
/// loses a third on many small pages**. Sixteen is never worst and asks half
/// the shared memory of 32, which is what holds the occupancy.
pub const PLACE_TILE_R: usize = 16;

/// One page to place: `[rows, head_dim]` F32.
pub struct PlacePage<'a> {
    pub keys: &'a Tensor,
}

/// One placement's buffers and the descriptor table addressing them.
///
/// Split from the launch because a projection places its pages once and then
/// scores against them for every decode step until it changes.
///
/// **The placements are ONE allocation, sliced.** A projection carrying a
/// hundred short pages is over a thousand jobs across its layers — measured,
/// per-page allocation was 1.21 ms of `Tensor::empty` against 0.013 ms of
/// kernel. The descriptor table already gives every job its own base pointer
/// (hot-path invariant 2b), so one flat buffer and a narrow per page addresses
/// exactly the same bytes.
#[derive(Debug)]
pub struct Placement {
    staged: Vec<Tensor>,
    jobs: Tensor,
    n_jobs: usize,
    d: usize,
    max_rows: usize,
    /// The one buffer every entry of `staged` is a view into. Held so the views
    /// cannot outlive it.
    _arena: Option<Tensor>,
}

impl Placement {
    /// Allocate the placements and build the descriptor table that addresses
    /// them.
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
                _arena: None,
            });
        }
        let device = pages[0].keys.device().clone();
        let (_, d) = pages[0].keys.dims2()?;
        if d % 4 != 0 {
            candle::bail!(
                "index page placement: head_dim {d} is not a multiple of four — the scorer's \
                 channel-blocked layout is built in `float4` groups"
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
        for (p, &rows) in pages.iter().zip(&rows_of) {
            // The destination address is arithmetic on the arena's base rather
            // than a second `tensor_ptr` per job: resolving a pointer walks the
            // storage and the layout, and at a thousand jobs that walk was
            // itself a measurable share of the plan.
            jobs.push(tensor_ptr(p.keys)? as i64);
            jobs.push((arena_base + (at * d) as u64 * elem) as i64);
            jobs.push(rows as i64);
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
            _arena: Some(arena),
        })
    }

    /// Lay every page out for the scorer — **one launch over all of them**
    /// (hot-path invariant 5).
    ///
    /// Idempotent: the kernel fully rewrites every placement from the pages,
    /// so running twice is running once. `tile_r` selects the row tile;
    /// production passes [`PLACE_TILE_R`] and the bench sweeps the rest.
    #[cfg(feature = "cuda")]
    pub fn run(&self, tile_r: usize) -> Result<()> {
        use candle_kernels::simple::qsa_page_place::run_qsa_page_place;

        if self.n_jobs == 0 || self.max_rows == 0 {
            return Ok(());
        }
        let candle::Device::Cuda(cuda) = self.jobs.device() else {
            candle::bail!("index page placement runs on CUDA");
        };
        let stream = cuda.cuda_stream();
        candle::set_kernel_breadcrumb("run_qsa_page_place", file!(), line!());
        unsafe {
            run_qsa_page_place(
                i64_ptr(&self.jobs)? as *const i64,
                self.d as i32,
                self.n_jobs as i32,
                self.max_rows as i32,
                tile_r as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
        Ok(())
    }

    /// The placements, `[head_dim/4, rows, 4]` each, in the order the pages
    /// were given.
    pub fn staged(&self) -> &[Tensor] {
        &self.staged
    }

    /// Take the placements.
    pub fn into_staged(self) -> Vec<Tensor> {
        self.staged
    }
}

/// Plan and run — what a projection calls.
///
/// Returns one `[head_dim/4, rows, 4]` F32 tensor per input page, in order.
#[cfg(feature = "cuda")]
pub fn place_pages(pages: &[PlacePage<'_>], tile_r: usize) -> Result<Vec<Tensor>> {
    let placement = Placement::plan(pages)?;
    placement.run(tile_r)?;
    Ok(placement.into_staged())
}
