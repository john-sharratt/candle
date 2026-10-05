//! The residual's token order where it crosses the forward API.
//!
//! The sweep packs its rows in **internal** order
//! `[decode | single-token prefills | multi-token prefills | glue]`: a
//! one-token prefill is a decode and rides the decode group. The residual
//! crosses the API in **caller** order `[decode | prefill (caller order) |
//! glue]`, so a co-batched caller can split it by contiguous group and hold a
//! creeping cohort whole while the full-sweep members continue. The two orders
//! differ only by where the single-token prefills sit, so the mapping between
//! them is a handful of contiguous runs — each multi-token prefill moves as
//! one block — and reordering is one copy per run into a held buffer
//! ([`WindowResiduals`]): no table to upload, nothing allocated.

use candle::{Result, Tensor};

use crate::models::window_residuals::WindowResiduals;

/// `len` tokens at `caller` in caller order and at `internal` in internal
/// order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    pub caller: usize,
    pub internal: usize,
    pub len: usize,
}

/// The runs between the two orders for `n_decode` decode rows, prefills of
/// `pre_lens` tokens each (caller order) and `glue_tok` glue rows — merged
/// wherever two neighbours are adjacent in both orders.
pub fn caller_runs(n_decode: usize, pre_lens: &[usize], glue_tok: usize) -> Vec<Run> {
    fn push(runs: &mut Vec<Run>, caller: usize, internal: usize, len: usize) {
        if len == 0 {
            return;
        }
        if let Some(last) = runs.last_mut() {
            if last.caller + last.len == caller && last.internal + last.len == internal {
                last.len += len;
                return;
            }
        }
        runs.push(Run {
            caller,
            internal,
            len,
        });
    }
    let singles = pre_lens.iter().filter(|&&l| l == 1).count();
    let mut runs = Vec::new();
    push(&mut runs, 0, 0, n_decode);
    let mut caller = n_decode;
    let mut next_single = n_decode;
    let mut next_multi = n_decode + singles;
    for &l in pre_lens {
        if l == 1 {
            push(&mut runs, caller, next_single, 1);
            next_single += 1;
        } else {
            push(&mut runs, caller, next_multi, l);
            next_multi += l;
        }
        caller += l;
    }
    push(&mut runs, caller, next_multi, glue_tok);
    runs
}

/// `src` — tokens on dim 1 — reordered into a buffer from `pool`: caller to
/// internal order when `to_internal`, internal to caller otherwise.
pub fn reorder(
    src: &Tensor,
    runs: &[Run],
    to_internal: bool,
    pool: &WindowResiduals,
) -> Result<Tensor> {
    let dst = pool.take(src.dims(), src.dtype(), src.device())?;
    for r in runs {
        let (from, to) = if to_internal {
            (r.caller, r.internal)
        } else {
            (r.internal, r.caller)
        };
        dst.slice_set(&src.narrow(1, from, r.len)?, 1, to)?;
    }
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    #[test]
    fn single_prefills_move_up_beside_the_decode_rows() {
        let runs = caller_runs(2, &[1, 3, 1, 2], 2);
        let run = |caller, internal, len| Run {
            caller,
            internal,
            len,
        };
        assert_eq!(
            runs,
            vec![run(0, 0, 3), run(3, 4, 3), run(6, 3, 1), run(7, 7, 4)]
        );
    }

    #[test]
    fn with_no_single_prefill_the_orders_are_one_run() {
        assert_eq!(
            caller_runs(3, &[4, 2], 1),
            vec![Run {
                caller: 0,
                internal: 0,
                len: 10
            }]
        );
    }

    #[test]
    fn reorder_places_each_token_and_round_trips() -> Result<()> {
        let pool = WindowResiduals::default();
        let runs = caller_runs(2, &[1, 3, 1, 2], 2);
        let caller = Tensor::arange(0f32, 11., &Device::Cpu)?.reshape((1, 11, 1))?;
        let internal = reorder(&caller, &runs, true, &pool)?;
        assert_eq!(
            internal.flatten_all()?.to_vec1::<f32>()?,
            vec![0., 1., 2., 6., 3., 4., 5., 7., 8., 9., 10.]
        );
        let back = reorder(&internal, &runs, false, &pool)?;
        assert_eq!(
            back.flatten_all()?.to_vec1::<f32>()?,
            caller.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(back.dtype(), DType::F32);
        Ok(())
    }
}
