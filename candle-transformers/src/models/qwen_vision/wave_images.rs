//! A wave's image rows: the ids past the vocabulary that name rows of an
//! image registered with the vision registry, and what any engine does with
//! them (`docs/vision_models.md` §2.3).
//!
//! The wave's host ids are read once; there every virtual id is resolved
//! against the registry. Its row's values are laid back to back, the id is
//! rewritten to `<|image_pad|>` — what the reference holds there — and each
//! maximal run of one image's consecutive rows is kept for the attention
//! layers, which turn those rows' queries and keys by their place in the image
//! (`docs/flash_next_vision.md` §4.4).
//!
//! How the rows reach the residual is the engine's: a table on the card takes
//! them as its gather's second table, and a host-mapped one gathers the
//! placeholders and has them overwritten by [`place_image_rows`].

use std::sync::Arc;

use candle::wave_provenance::WaveTicket;
use candle::{DType, Device, Result, Tensor};
use half::{bf16, f16};

use super::registry::{is_virtual, lookup};
use super::rotation::{rotation_tables, ImageRotation, ImageRun};
use crate::models::latent_moe::desc;
use crate::models::latent_moe::scatter::{rows_scatter, rows_scatter_inline, RowRun};
use crate::models::wave_buffers::wave_from_vec_ticketed;
use candle_kernels::simple::rows_scatter::ROWS_SCATTER_INLINE_MAX;

/// What a wave with images feeds its embedding and its attention. Each run's
/// `row` is its first row in the wave.
pub struct WaveImages {
    /// Every row's id, with each virtual id replaced by its row of the second
    /// table (`n_vocab + k`) — for an embedding that reads one.
    pub gather_ids: Vec<u32>,
    /// The rows the virtual ids name, `[k, width]` row-major, in wave order.
    pub values: Vec<f32>,
    pub runs: Vec<ImageRun>,
}

/// Virtual ids among `inputs` that are on the host — what the wave is priced
/// for before the forward ([`WaveWidth::image_rows`]). A device input is a
/// sampled token, never an image's.
///
/// [`WaveWidth::image_rows`]: candle_nn::kv_cache::WaveWidth
pub fn count_image_rows(inputs: &[&Tensor]) -> Result<usize> {
    let mut n = 0;
    for t in inputs {
        if t.device().is_cuda() || t.dtype() != DType::U32 {
            continue;
        }
        n += t
            .flatten_all()?
            .to_vec1::<u32>()?
            .into_iter()
            .filter(|&id| is_virtual(id))
            .count();
    }
    Ok(n)
}

/// Resolve the virtual ids in `seq_tokens` (each input's ids, in wave order),
/// or `None` for a wave with none. Rewrites each virtual id to `image_pad` in
/// place.
pub fn take_wave_images(
    seq_tokens: &mut [Vec<u32>],
    n_vocab: usize,
    image_pad: u32,
    width: usize,
) -> Result<Option<WaveImages>> {
    if !seq_tokens.iter().flatten().any(|&id| is_virtual(id)) {
        return Ok(None);
    }
    let mut gather_ids = Vec::with_capacity(seq_tokens.iter().map(Vec::len).sum());
    let mut values: Vec<f32> = Vec::new();
    let mut runs: Vec<ImageRun> = Vec::new();
    let mut row = 0usize;
    for toks in seq_tokens.iter_mut() {
        // A run never crosses from one sequence into the next.
        let mut open: Option<(usize, usize, usize)> = None; // (run index, image ptr, next index)
        for id in toks.iter_mut() {
            if !is_virtual(*id) {
                gather_ids.push(*id);
                open = None;
                row += 1;
                continue;
            }
            let (rows, i) = lookup(*id).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "wave images: virtual id {id:#x} names no registered image — its handle \
                     was dropped before the prefill that reads it"
                ))
            })?;
            if rows.width != width {
                candle::bail!(
                    "wave images: an image's rows are {} wide, the trunk {width} — read by \
                     another model's tower",
                    rows.width
                );
            }
            let k = values.len() / width;
            gather_ids.push((n_vocab + k) as u32);
            values.extend_from_slice(rows.row(i));
            let ptr = Arc::as_ptr(&rows) as usize;
            match open {
                Some((r, p, next)) if p == ptr && next == i => {
                    runs[r].len += 1;
                    open = Some((r, p, i + 1));
                }
                _ => {
                    runs.push(ImageRun {
                        row,
                        len: 1,
                        grid: rows.grid,
                        first: i,
                    });
                    open = Some((runs.len() - 1, ptr, i + 1));
                }
            }
            *id = image_pad;
            row += 1;
        }
    }
    Ok(Some(WaveImages {
        gather_ids,
        values,
        runs,
    }))
}

/// The turn of every image row of a group whose first row is wave row
/// `row_base` — the prefill group, after the decode rows — for a trunk rotating
/// `rope_dim` dims at base `theta`, its tables on the forward's `ticket`.
pub fn image_rotation(
    images: &WaveImages,
    row_base: usize,
    rope_dim: usize,
    theta: f64,
    dev: &Device,
    ticket: Option<WaveTicket>,
) -> Result<ImageRotation> {
    let runs: Vec<ImageRun> = images
        .runs
        .iter()
        .map(|r| ImageRun {
            row: r.row - row_base,
            ..*r
        })
        .collect();
    let t = rotation_tables(&runs, rope_dim, theta);
    let n = t.rows.len();
    ImageRotation::from_tables(
        wave_from_vec_ticketed(t.rows, n, dev, ticket)?,
        t.rows_end,
        wave_from_vec_ticketed(t.cos, (n, t.pairs), dev, ticket)?,
        wave_from_vec_ticketed(t.sin, (n, t.pairs), dev, ticket)?,
    )
}

/// The rows the wave's virtual ids name, `[k, width]` in `dtype` — the
/// residual's own type, so [`place_image_rows`] copies them as they are —
/// uploaded on the forward's `ticket`. Each value is rounded once, from the
/// tower's F32, as the residual's own rows are.
pub fn upload_image_rows(
    images: &WaveImages,
    width: usize,
    dtype: DType,
    dev: &Device,
    ticket: Option<WaveTicket>,
) -> Result<Tensor> {
    let k = images.values.len() / width;
    match dtype {
        DType::F32 => wave_from_vec_ticketed(images.values.clone(), (k, width), dev, ticket),
        DType::BF16 => {
            let v: Vec<bf16> = images.values.iter().map(|&x| bf16::from_f32(x)).collect();
            wave_from_vec_ticketed(v, (k, width), dev, ticket)
        }
        DType::F16 => {
            let v: Vec<f16> = images.values.iter().map(|&x| f16::from_f32(x)).collect();
            wave_from_vec_ticketed(v, (k, width), dev, ticket)
        }
        other => candle::bail!("wave images: no image rows in a {other:?} residual"),
    }
}

/// Copy `rows` — what [`upload_image_rows`] uploaded — over each run's rows of
/// `dst`, `[n, width]`, whose row 0 is wave row `row_base`: one launch for every
/// run (`rows_scatter`, a descriptor per run), every other row of `dst` left as
/// it was.
pub fn place_image_rows(
    dst: &Tensor,
    rows: &Tensor,
    images: &WaveImages,
    row_base: usize,
) -> Result<()> {
    let mut k = 0usize;
    let mut runs = Vec::with_capacity(images.runs.len());
    for run in &images.runs {
        runs.push(RowRun::new(
            rows.narrow(0, k, run.len)?,
            dst,
            run.row - row_base,
        ));
        k += run.len;
    }
    if runs.len() <= ROWS_SCATTER_INLINE_MAX {
        rows_scatter_inline(&runs)
    } else {
        rows_scatter(&runs, &desc::scope(dst.device())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::gpu_test_lock::gpu_serial;
    use crate::models::qwen_vision::geometry::FACTOR;
    use crate::models::qwen_vision::registry::{ImageHandle, ImageRows};
    use crate::models::qwen_vision::Grid;

    fn image(merged_rows: usize, merged_cols: usize, seed: f32) -> ImageRows {
        let grid = Grid {
            height: merged_rows * FACTOR,
            width: merged_cols * FACTOR,
        };
        ImageRows {
            grid,
            width: 2,
            values: (0..grid.tokens() * 2).map(|i| seed + i as f32).collect(),
        }
    }

    /// Two sequences: one with text around part of an image, one with a whole
    /// second image. Ids, placeholders, the rows and the runs, exactly.
    #[test]
    fn virtual_ids_become_the_second_tables_rows_and_runs() {
        let a = ImageHandle::register(image(1, 3, 10.0)).unwrap();
        let b = ImageHandle::register(image(1, 2, 50.0)).unwrap();
        let mut seq = vec![vec![7, a.id(1), a.id(2), 8], vec![b.id(0), b.id(1)]];
        let got = take_wave_images(&mut seq, 100, 9, 2).unwrap().unwrap();
        assert_eq!(got.gather_ids, [7, 100, 101, 8, 102, 103]);
        assert_eq!(seq, [vec![7, 9, 9, 8], vec![9, 9]]);
        assert_eq!(got.values, [12.0, 13.0, 14.0, 15.0, 50.0, 51.0, 52.0, 53.0]);
        assert_eq!(got.runs.len(), 2);
        assert_eq!(
            (got.runs[0].row, got.runs[0].len, got.runs[0].first),
            (1, 2, 1)
        );
        assert_eq!(
            (got.runs[1].row, got.runs[1].len, got.runs[1].first),
            (4, 2, 0)
        );
        assert_eq!(got.runs[1].grid, b.rows().grid);
    }

    #[test]
    fn a_wave_without_images_takes_nothing() {
        let mut seq = vec![vec![1, 2, 3]];
        assert!(take_wave_images(&mut seq, 100, 9, 2).unwrap().is_none());
        assert_eq!(seq, [vec![1, 2, 3]]);
    }

    /// An id whose handle has gone is refused, never gathered as zeros.
    #[test]
    fn a_dropped_image_is_refused() {
        let a = ImageHandle::register(image(1, 1, 0.0)).unwrap();
        let id = a.id(0);
        drop(a);
        let mut seq = vec![vec![id]];
        assert!(take_wave_images(&mut seq, 100, 9, 2).is_err());
    }

    /// Rows another model's tower wrote — another width — are refused, never
    /// read at this trunk's.
    #[test]
    fn another_towers_rows_are_refused() {
        let a = ImageHandle::register(image(1, 1, 0.0)).unwrap();
        let mut seq = vec![vec![a.id(0)]];
        let e = take_wave_images(&mut seq, 100, 9, 4)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("2 wide, the trunk 4"), "{e}");
    }

    #[test]
    fn host_virtual_ids_are_counted() {
        let a = ImageHandle::register(image(1, 2, 0.0)).unwrap();
        let t = Tensor::new(&[[1u32, a.id(0), a.id(1), 4]], &Device::Cpu).unwrap();
        assert_eq!(count_image_rows(&[&t]).unwrap(), 2);
    }

    /// **Each run's rows land over its placeholders, in the residual's type**,
    /// and every other row is left bit for bit — on the card, both runs in one
    /// launch, BF16 rounded once from the tower's F32.
    #[test]
    fn image_rows_are_placed_over_their_placeholders() {
        let _gpu = gpu_serial();
        let Ok(dev) = Device::new_cuda(0) else { return };
        let a = ImageHandle::register(image(1, 3, 10.25)).unwrap();
        let b = ImageHandle::register(image(1, 2, 50.5)).unwrap();
        // Decode rows 0 and 1, then [7, a1, a2, 8] and [b0, b1].
        let mut seq = vec![
            vec![3],
            vec![4],
            vec![7, a.id(1), a.id(2), 8],
            vec![b.id(0), b.id(1)],
        ];
        let got = take_wave_images(&mut seq, 100, 9, 2).unwrap().unwrap();
        let n = 8;
        let before: Vec<bf16> = (0..n * 2).map(|i| bf16::from_f32(-(i as f32))).collect();
        let dst = Tensor::from_vec(before.clone(), (n, 2), &dev).unwrap();
        let rows = upload_image_rows(&got, 2, DType::BF16, &dev, None).unwrap();
        place_image_rows(&dst, &rows, &got, 0).unwrap();

        let mut want = before;
        for (row, values) in [
            (3, [12.25, 13.25]),
            (4, [14.25, 15.25]),
            (6, [50.5, 51.5]),
            (7, [52.5, 53.5]),
        ] {
            want[row * 2] = bf16::from_f32(values[0]);
            want[row * 2 + 1] = bf16::from_f32(values[1]);
        }
        assert_eq!(dst.flatten_all().unwrap().to_vec1::<bf16>().unwrap(), want);
    }

    /// The prefill group's rotation lists each image row by its row in the
    /// group, not the wave.
    #[test]
    fn the_rotation_counts_rows_from_the_group() {
        let _gpu = gpu_serial();
        let Ok(dev) = Device::new_cuda(0) else { return };
        let a = ImageHandle::register(image(1, 2, 0.0)).unwrap();
        let mut seq = vec![vec![3], vec![7, a.id(0), a.id(1)]];
        let got = take_wave_images(&mut seq, 100, 9, 2).unwrap().unwrap();
        let rot = image_rotation(&got, 1, 4, 10.0, &dev, None).unwrap();
        assert!(!rot.is_empty());
        let q = Tensor::zeros((2, 1, 8), DType::F32, &dev).unwrap();
        // Rows 1 and 2 of the group: a two-row tensor is one short.
        assert!(rot.apply(&q, &q.clone()).is_err());
        let q = Tensor::zeros((3, 1, 8), DType::F32, &dev).unwrap();
        let k = Tensor::zeros((3, 1, 8), DType::F32, &dev).unwrap();
        rot.apply(&q, &k).unwrap();
    }
}
