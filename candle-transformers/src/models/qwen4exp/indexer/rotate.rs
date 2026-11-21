//! Rotates rows of queries or keys at their own positions and rungs.
//!
//! Each row is rotated at its own position and rung. A query's `m²` scale applies
//! on the query side only, since keys take none. A launch refuses a zero group
//! size before its destination is allocated, because the kernel would otherwise
//! return without writing it and leave it uninitialised. The source may be a
//! dense tensor, a dense tensor normed first, or the pages of a live tail.

use super::*;

/// Where each group of rows sits, for [`rotate_rows`].
pub enum RowPositions<'a> {
    /// One position per group.
    PerGroup(&'a [usize]),
    /// Group `k` at `base + k·step`.
    Affine { base: usize, step: usize },
}

/// Which rung each group of rows rotates at, for [`rotate_rows`].
#[derive(Clone, Copy)]
pub enum RowRungs<'a> {
    /// One rung per group — a wave's queries, one sequence's rung per row.
    PerGroup(&'a [u32]),
    /// Every group at one rung — one sequence's keys.
    Uniform(u32),
}

/// What is being rotated, which decides the scale on the rotated channels.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RotSide {
    /// Queries take the rung's `m²` (`docs/progressive_yarn.md` §4.2).
    Query,
    /// Keys take none.
    Key,
}

/// The rows [`rotate_rows`] reads.
pub enum RowSource<'a> {
    /// One dense `[n, d]` tensor.
    Dense(&'a Tensor),
    /// One dense `[n, d]` tensor RMS-normed over each row with `weight` and
    /// `eps` before it rotates — a wave's queries, normed and rotated in one
    /// launch with `qsa::rms_norm_last`'s arithmetic.
    Normed {
        rows: &'a Tensor,
        weight: &'a Tensor,
        eps: f64,
    },
    /// `rows` rows of width `d` in pages of `rows_per_page`, row `r` at
    /// `pages[r / rows_per_page] + (r % rows_per_page)·d` floats — a live tail's
    /// keys, read in place.
    Paged {
        pages: &'a [u64],
        rows_per_page: usize,
        rows: usize,
        d: usize,
        device: &'a Device,
    },
}

/// Rotate the rows of `src` at their positions and rungs from `rope`, into a new
/// dense `[n, d]` tensor.
///
/// Consecutive runs of `rows_per_pos` rows share a position and a rung — a
/// query's heads all sit at the query's position, on its sequence's rung.
#[cfg(feature = "cuda")]
pub fn rotate_rows(
    src: RowSource<'_>,
    rope: &FactoredRope,
    rows_per_pos: usize,
    positions: RowPositions<'_>,
    rungs: RowRungs<'_>,
    side: RotSide,
    // The open layer phase, for the position and rung tables below.
    ticket: Option<WaveTicket>,
) -> Result<Tensor> {
    use candle_kernels::simple::qsa_rope_rows::{
        run_qsa_rope_rows, run_qsa_rope_rows_norm, QSA_ROPE_NORM_MAX_D,
    };

    let (n, d, device) = match &src {
        RowSource::Dense(t) => {
            expect_dense(t, "qsa rope rows")?;
            let (n, d) = t.dims2()?;
            (n, d, t.device())
        }
        RowSource::Normed { rows, weight, .. } => {
            expect_dense(rows, "qsa rope rows")?;
            expect_dense_dtype(weight, DType::F32, "qsa rope rows norm weight")?;
            let (n, d) = rows.dims2()?;
            if weight.dims() != [d] || d > QSA_ROPE_NORM_MAX_D {
                candle::bail!(
                    "qsa rope rows: a {:?} norm weight on {d}-wide rows (at most {})",
                    weight.dims(),
                    QSA_ROPE_NORM_MAX_D
                );
            }
            (n, d, rows.device())
        }
        RowSource::Paged {
            pages,
            rows_per_page,
            rows,
            d,
            device,
        } => {
            if *rows_per_page == 0 || rows.div_ceil(*rows_per_page) > pages.len() {
                candle::bail!(
                    "qsa rope rows: {rows} rows in pages of {rows_per_page} against a \
                     {}-page table",
                    pages.len()
                );
            }
            (*rows, *d, *device)
        }
    };
    if rope.rope_dim() > d {
        candle::bail!(
            "qsa rope rows: a {}-wide rotary width on {d}-wide rows",
            rope.rope_dim()
        );
    }
    // The kernel's own guard refuses to launch when `rows_per_pos == 0`
    // (`row / rows_per_pos` would divide by zero) and returns without writing
    // a single byte of `dst` — silently leaving it uninitialised rather than
    // erroring. Catch it here, before `dst` is even allocated.
    if rows_per_pos == 0 {
        candle::bail!("qsa rope rows: rows_per_pos must be at least 1");
    }
    let dst = wave_empty_ticketed((n, d), DType::F32, device, ticket)?;
    if n == 0 {
        return Ok(dst);
    }
    let groups = n / rows_per_pos;
    let (pos_t, base, step) = match positions {
        RowPositions::PerGroup(p) => {
            if p.len() * rows_per_pos != n {
                candle::bail!(
                    "qsa rope rows: {} positions for {n} rows in groups of {rows_per_pos}",
                    p.len()
                );
            }
            let v: Vec<u32> = p.iter().map(|&x| x as u32).collect();
            (
                Some(wave_from_vec_ticketed(v, (p.len(),), device, ticket)?),
                0,
                0,
            )
        }
        RowPositions::Affine { base, step } => (None, base, step),
    };
    let (rung_t, rung) = match rungs {
        RowRungs::PerGroup(r) => {
            if r.len() != groups {
                candle::bail!(
                    "qsa rope rows: {} rungs for {groups} groups of {rows_per_pos}",
                    r.len()
                );
            }
            (
                Some(wave_from_vec_ticketed(
                    r.to_vec(),
                    (r.len(),),
                    device,
                    ticket,
                )?),
                0,
            )
        }
        RowRungs::Uniform(r) => (None, r),
    };
    let n_rungs = rope.rungs().n_rungs() as u32;
    let top = match rungs {
        RowRungs::PerGroup(r) => r.iter().copied().max().unwrap_or(0),
        RowRungs::Uniform(r) => r,
    };
    if top >= n_rungs {
        candle::bail!("qsa rope rows: rung {top} of a {n_rungs}-rung set");
    }
    let pos_ptr = match &pos_t {
        Some(t) => u32_ptr(t)? as *const u32,
        None => std::ptr::null(),
    };
    let rung_ptr = match &rung_t {
        Some(t) => u32_ptr(t)? as *const u32,
        None => std::ptr::null(),
    };
    // The page table, uploaded for the launch; the dense source needs none.
    let (src_ptr, pages_t, rows_per_src_page) = match &src {
        RowSource::Dense(t) => (tensor_ptr(t)?, None, 0usize),
        RowSource::Normed { rows, .. } => (tensor_ptr(rows)?, None, 0usize),
        RowSource::Paged {
            pages,
            rows_per_page,
            rows,
            ..
        } => {
            let used: Vec<i64> = pages[..rows.div_ceil(*rows_per_page)]
                .iter()
                .map(|&p| p as i64)
                .collect();
            let len = used.len();
            (
                0,
                Some(wave_from_vec_ticketed(used, (len,), device, ticket)?),
                *rows_per_page,
            )
        }
    };
    let pages_ptr = match &pages_t {
        Some(t) => i64_ptr(t)? as *const i64,
        None => std::ptr::null(),
    };
    let candle::Device::Cuda(cuda) = device else {
        candle::bail!("qsa rope rows runs on CUDA");
    };
    let stream = cuda.cuda_stream();
    if let RowSource::Normed { weight, eps, .. } = &src {
        candle::set_kernel_breadcrumb("run_qsa_rope_rows_norm", file!(), line!());
        // SAFETY: `src_ptr` is `[n, d]` dense F32, `weight` `[d]` F32 and `dst`
        // `[n, d]`, checked above; the position and rung tables are sized to the
        // groups.
        unsafe {
            run_qsa_rope_rows_norm(
                src_ptr as *const f32,
                tensor_ptr(&dst)? as *mut f32,
                n as i32,
                d as i32,
                rows_per_pos as i32,
                pos_ptr,
                base as i64,
                step as i32,
                rope.rungs().ffi()?,
                rung_ptr,
                rung,
                i32::from(side == RotSide::Query),
                tensor_ptr(weight)? as *const f32,
                *eps as f32,
                stream.cu_stream() as *mut c_void,
            );
        }
        return Ok(dst);
    }
    candle::set_kernel_breadcrumb("run_qsa_rope_rows", file!(), line!());
    unsafe {
        run_qsa_rope_rows(
            src_ptr as *const f32,
            pages_ptr,
            rows_per_src_page as i32,
            tensor_ptr(&dst)? as *mut f32,
            n as i32,
            d as i32,
            rows_per_pos as i32,
            pos_ptr,
            base as i64,
            step as i32,
            rope.rungs().ffi()?,
            rung_ptr,
            rung,
            i32::from(side == RotSide::Query),
            stream.cu_stream() as *mut c_void,
        );
    }
    Ok(dst)
}
