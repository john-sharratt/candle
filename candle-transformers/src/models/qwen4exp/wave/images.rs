//! A wave's image rows: the ids past the vocabulary that name rows of an
//! image registered with the vision registry, and what the wave does with them.
//!
//! The wave reads its ids to the host once, for the PLE hash; there, every
//! virtual id is resolved against the registry. Its row's values go into one
//! upload — the embedding gather's second table — and the id becomes that
//! table's row, `n_vocab + k`; the PLE hash sees `<|image_pad|>` in its place,
//! as the reference hashes its input ids. Each maximal run of one image's
//! consecutive rows is kept for the attention layers, which turn those rows'
//! queries and keys by their place in the image (`docs/flash_next_vision.md`
//! §4.4).

use candle::cuda_backend::wave_provenance::WaveTicket;
use candle::{DType, Device, Result, Tensor};

use crate::models::qwen_vision::registry::{is_virtual, lookup};
use crate::models::qwen_vision::ImageRun;
use crate::models::wave_buffers::wave_from_vec_ticketed;

/// What a wave with images feeds its gather and its attention. Each run's
/// `row` is its first row in the wave.
pub(super) struct WaveImages {
    /// Every row's id, with each virtual id replaced by its row of `extra`
    /// (`n_vocab + k`).
    pub gather_ids: Vec<u32>,
    /// `[k, width]` F32 on the forward span: the rows the virtual ids name.
    pub extra: Tensor,
    pub runs: Vec<ImageRun>,
}

/// Virtual ids among `inputs` that are on the host — what the forward span is
/// priced for before the forward ([`WaveWidth::image_rows`]). A device input is
/// a sampled token, never an image's.
///
/// [`WaveWidth::image_rows`]: candle_nn::kv_cache::WaveWidth
pub(super) fn count_image_rows(inputs: &[Tensor]) -> Result<usize> {
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
/// place, for the PLE hash.
pub(super) fn take_wave_images(
    seq_tokens: &mut [Vec<u32>],
    n_vocab: usize,
    image_pad: u32,
    width: usize,
    dev: &Device,
    ticket: Option<WaveTicket>,
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
                    "qwen4exp wave: virtual id {id:#x} names no registered image — its handle \
                     was dropped before the prefill that reads it"
                ))
            })?;
            if rows.width != width {
                candle::bail!(
                    "qwen4exp wave: an image's rows are {} wide, the trunk {width}",
                    rows.width
                );
            }
            let k = values.len() / width;
            gather_ids.push((n_vocab + k) as u32);
            values.extend_from_slice(rows.row(i));
            let ptr = std::sync::Arc::as_ptr(&rows) as usize;
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
    let k = values.len() / width;
    let extra = wave_from_vec_ticketed(values, (k, width), dev, ticket)?;
    Ok(Some(WaveImages {
        gather_ids,
        extra,
        runs,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
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
    /// second image. Ids, PLE tokens, the second table and the runs, exactly.
    #[test]
    fn virtual_ids_become_the_second_tables_rows_and_runs() {
        let a = ImageHandle::register(image(1, 3, 10.0)).unwrap();
        let b = ImageHandle::register(image(1, 2, 50.0)).unwrap();
        let mut seq = vec![vec![7, a.id(1), a.id(2), 8], vec![b.id(0), b.id(1)]];
        let got = take_wave_images(&mut seq, 100, 9, 2, &Device::Cpu, None)
            .unwrap()
            .unwrap();
        assert_eq!(got.gather_ids, [7, 100, 101, 8, 102, 103]);
        assert_eq!(seq, [vec![7, 9, 9, 8], vec![9, 9]]);
        assert_eq!(
            got.extra.to_vec2::<f32>().unwrap(),
            [[12.0, 13.0], [14.0, 15.0], [50.0, 51.0], [52.0, 53.0]]
        );
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
        assert!(take_wave_images(&mut seq, 100, 9, 2, &Device::Cpu, None)
            .unwrap()
            .is_none());
        assert_eq!(seq, [vec![1, 2, 3]]);
    }

    /// An id whose handle has gone is refused, never gathered as zeros.
    #[test]
    fn a_dropped_image_is_refused() {
        let a = ImageHandle::register(image(1, 1, 0.0)).unwrap();
        let id = a.id(0);
        drop(a);
        let mut seq = vec![vec![id]];
        assert!(take_wave_images(&mut seq, 100, 9, 2, &Device::Cpu, None).is_err());
    }

    #[test]
    fn host_virtual_ids_are_counted() {
        let a = ImageHandle::register(image(1, 2, 0.0)).unwrap();
        let t = Tensor::new(&[[1u32, a.id(0), a.id(1), 4]], &Device::Cpu).unwrap();
        assert_eq!(count_image_rows(&[t]).unwrap(), 2);
    }
}
