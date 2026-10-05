//! The rows of a wave's residual that the LM head scores.
//!
//! A wave scores every decode row and one row (or, for a verify block, every
//! row) of each prefill span. When those rows form one contiguous run — every
//! decode-only wave, where the run is the whole residual — they are a view of
//! the residual, and gathering them would allocate and copy a tensor the
//! residual already is (hot-path invariant 2) after uploading an index the
//! layout already implies. Only a genuinely scattered selection gathers.

use candle::wave_provenance::WaveTicket;
use candle::{Result, Tensor};

#[cfg(feature = "cuda")]
use crate::models::wave_buffers::wave_from_vec_ticketed;

/// The first row and length of `rows` when they are consecutive, ascending.
pub fn contiguous_run(rows: &[u32]) -> Option<(usize, usize)> {
    let first = *rows.first()?;
    rows.windows(2)
        .all(|w| w[1] == w[0] + 1)
        .then_some((first as usize, rows.len()))
}

/// `x` restricted to `rows` along `dim`: a `narrow` view when the rows are one
/// consecutive run, otherwise an `index_select` gather.
///
/// The gather lands beside `x`; its index is uploaded onto `index_ticket`'s
/// arena when one is given, and onto the pool otherwise.
pub fn select_head_rows(
    x: &Tensor,
    rows: Vec<u32>,
    dim: usize,
    index_ticket: Option<WaveTicket>,
) -> Result<Tensor> {
    match contiguous_run(&rows) {
        Some((start, len)) if start == 0 && len == x.dim(dim)? => Ok(x.clone()),
        Some((start, len)) => x.narrow(dim, start, len),
        None => {
            let n = rows.len();
            #[cfg(feature = "cuda")]
            let idx = wave_from_vec_ticketed(rows, (n,), x.device(), index_ticket)?;
            #[cfg(not(feature = "cuda"))]
            let idx = {
                let _ = index_ticket;
                Tensor::from_vec(rows, n, x.device())?
            };
            x.index_select(&idx, dim)
        }
    }
}

#[cfg(test)]
mod tests {
    use candle::{Device, Tensor};

    use super::{contiguous_run, select_head_rows};

    fn residual() -> Tensor {
        Tensor::arange(0f32, 12.0, &Device::Cpu)
            .unwrap()
            .reshape((6, 2))
            .unwrap()
    }

    #[test]
    fn a_consecutive_selection_is_a_run() {
        assert_eq!(contiguous_run(&[0, 1, 2, 3]), Some((0, 4)));
        assert_eq!(contiguous_run(&[3, 4]), Some((3, 2)));
        assert_eq!(contiguous_run(&[5]), Some((5, 1)));
    }

    #[test]
    fn a_gap_or_reorder_is_not_a_run() {
        assert_eq!(contiguous_run(&[]), None);
        assert_eq!(contiguous_run(&[0, 1, 3]), None);
        assert_eq!(contiguous_run(&[1, 0]), None);
        assert_eq!(contiguous_run(&[2, 2]), None);
    }

    #[test]
    fn scoring_every_row_returns_the_residual_itself() {
        let x = residual();
        let got = select_head_rows(&x, (0..6).collect(), 0, None).unwrap();
        assert!(got.same_storage(&x));
        assert_eq!(got.dims(), &[6, 2]);
        assert_eq!(got.to_vec2::<f32>().unwrap(), x.to_vec2::<f32>().unwrap());
    }

    #[test]
    fn a_run_is_a_view_of_the_residual() {
        let x = residual();
        let got = select_head_rows(&x, vec![2, 3, 4], 0, None).unwrap();
        assert!(got.same_storage(&x));
        assert!(got.is_contiguous());
        assert_eq!(
            got.to_vec2::<f32>().unwrap(),
            vec![vec![4.0, 5.0], vec![6.0, 7.0], vec![8.0, 9.0]]
        );
    }

    #[test]
    fn a_scattered_selection_gathers_exactly_those_rows() {
        let x = residual();
        let got = select_head_rows(&x, vec![0, 1, 5], 0, None).unwrap();
        assert!(!got.same_storage(&x));
        assert_eq!(
            got.to_vec2::<f32>().unwrap(),
            vec![vec![0.0, 1.0], vec![2.0, 3.0], vec![10.0, 11.0]]
        );
    }

    #[test]
    fn rows_select_along_the_requested_dim() {
        let x = residual().reshape((1, 6, 2)).unwrap();
        let run = select_head_rows(&x, vec![4, 5], 1, None).unwrap();
        assert_eq!(
            run.to_vec3::<f32>().unwrap(),
            vec![vec![vec![8.0, 9.0], vec![10.0, 11.0]]]
        );
        let gathered = select_head_rows(&x, vec![5, 0], 1, None).unwrap();
        assert_eq!(
            gathered.to_vec3::<f32>().unwrap(),
            vec![vec![vec![10.0, 11.0], vec![0.0, 1.0]]]
        );
    }
}
