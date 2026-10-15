//! A finished prefill's first token, sampled in the wave that finished it.
//!
//! The head writes its logits on the forward's span, and the span is reclaimed
//! by the next forward — so a prefill that finished in one wave and was
//! promoted to decode after the loop's other quanta had to copy its row off
//! the span to survive that long: a vocabulary-wide device allocation per
//! finished prefill, inside the wave. The sample is taken instead while the
//! wave still holds its logits, and what survives to promotion is the token.

use candle::{DType, Tensor};

use crate::batched_sampler::SequenceSamplingState;
use crate::config::SamplingConfig;
use crate::error::ConversationError;
use crate::stencil::StencilDriver;

/// Everything a turn's first sample needs, readied before the sample.
pub(crate) struct FirstSample {
    pub(super) context_depth: usize,
    pub(super) sampling_state: SequenceSamplingState,
    /// The turn's config, with a turn grammar's opening mask applied.
    pub(super) sampling: SamplingConfig,
    pub(super) turn_driver: Option<StencilDriver>,
}

/// A turn's first sample as taken: what it was readied with, and the token
/// it drew — or why it drew none.
pub(crate) struct FirstSampled {
    pub(super) first: FirstSample,
    pub(super) token: Result<u32, ConversationError>,
}

/// How a prefill's last chunk ended, recorded in the wave that ran it.
pub(crate) enum PrefillEnd {
    /// A turn that decodes: its first token, sampled from the finishing row.
    Sampled(FirstSampled),
    /// A compression re-prefill: it decodes nothing, so its row is not read.
    Sealed,
}

/// `rows` — each `[1, vocab]` — split into maximal runs that are already one
/// block in memory, as `(start, end)` index pairs in order. Each run is named
/// whole by [`Tensor::cat_view`], without a copy.
pub(super) fn contiguous_runs(rows: &[Tensor]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start = 0;
    for end in 1..=rows.len() {
        let breaks = end == rows.len() || Tensor::cat_view(&rows[start..=end], 0).is_none();
        if breaks {
            runs.push((start, end));
            start = end;
        }
    }
    runs
}

/// Whether a logits row is free of NaN — `Some(false)` when any entry is
/// NaN, `None` when the row could not be read back.
///
/// Read only on a failed sample's path, where one row of a block has made the
/// whole dispatch fail and the question is which. NaN, not "non-finite": a row
/// may carry `-inf` entries legitimately (vocabulary padding), and a forward
/// gone bad shows as NaN throughout.
pub(super) fn row_is_nan_free(row: &Tensor) -> Option<bool> {
    let values = row
        .flatten_all()
        .and_then(|r| r.to_dtype(DType::F32))
        .and_then(|r| r.to_vec1::<f32>())
        .ok()?;
    Some(!values.iter().any(|v| v.is_nan()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Result};

    /// Adjacent rows of one block are one run; a row out of order, or from
    /// another block, starts a new one.
    #[test]
    fn rows_split_where_the_block_breaks() -> Result<()> {
        let block = Tensor::arange(0f32, 12., &Device::Cpu)?.reshape((4, 3))?;
        let other = Tensor::zeros((2, 3), candle::DType::F32, &Device::Cpu)?;
        let row = |t: &Tensor, i: usize| t.narrow(0, i, 1);
        let rows = vec![
            row(&block, 0)?,
            row(&block, 1)?,
            row(&block, 3)?,
            row(&other, 0)?,
            row(&other, 1)?,
        ];
        assert_eq!(contiguous_runs(&rows), vec![(0, 2), (2, 3), (3, 5)]);
        let first = Tensor::cat_view(&rows[0..2], 0).expect("one run");
        assert_eq!(
            first.to_vec2::<f32>()?,
            vec![vec![0., 1., 2.], vec![3., 4., 5.]]
        );
        assert!(contiguous_runs(&[]).is_empty());
        assert_eq!(contiguous_runs(&rows[2..3]), vec![(0, 1)]);
        Ok(())
    }

    /// A NaN anywhere marks the row; `-inf` padding does not.
    #[test]
    fn a_row_is_nan_free_unless_it_holds_nan() -> Result<()> {
        let dev = Device::Cpu;
        let ok = Tensor::new(&[[1f32, -2., f32::NEG_INFINITY]], &dev)?;
        let bad = Tensor::new(&[[1f32, f32::NAN, 3.]], &dev)?;
        assert_eq!(row_is_nan_free(&ok), Some(true));
        assert_eq!(row_is_nan_free(&bad), Some(false));
        Ok(())
    }
}
