//! Greedy token picks for the harness, on the fused batched sampler.
//!
//! Every pick runs the sampler's greedy path — the kernel the scheduler runs
//! at temperature zero — rather than the generic `argmax` reduction, whose
//! half-precision path addresses every element of a vocabulary-wide row
//! through the strided-index walk. A batch of rows is one launch and one
//! read-back.
//!
//! `greedy_rows` and `greedy_token` pick over the whole row. Their callers —
//! the reproducibility and replay probes, the long-context reads — compare a
//! model's picks against its own, so a padded column is no more able to win
//! one side than the other, and none of them holds a tokenizer. The
//! speculative decode path bounds its picks with [`live_vocab`].

use candle::{Result, Tensor, D};
use tokenizers::Tokenizer;

/// The tokens `tokenizer` can name — its largest id + 1 — clamped to a logits
/// row of `row_width` columns. A checkpoint pads its output projection past
/// the tokenizer, and the padded tail of a row is not a token.
pub fn live_vocab(tokenizer: &Tokenizer, row_width: usize) -> usize {
    tokenizer
        .get_vocab(true)
        .values()
        .max()
        .map_or(row_width, |&id| (id as usize + 1).min(row_width))
}

/// The greedy token of each row of `rows`, every row a `[1, vocab]` (or
/// `[vocab]`) logits row: one launch over all of them and one read-back.
pub fn greedy_rows(rows: &[&Tensor]) -> Result<Vec<u32>> {
    let rows = rows
        .iter()
        .map(|r| r.reshape((1, r.dim(D::Minus1)?)))
        .collect::<Result<Vec<_>>>()?;
    let stacked = Tensor::cat(&rows, 0)?;
    let width = stacked.dim(1)?;
    stacked.batched_sample_argmax(width)?.to_vec1::<u32>()
}

/// The greedy token of one `[1, vocab]` (or `[vocab]`) logits row.
pub fn greedy_token(row: &Tensor) -> Result<u32> {
    let vocab = row.dim(D::Minus1)?;
    Ok(row
        .reshape((1, vocab))?
        .batched_sample_argmax(vocab)?
        .to_vec1::<u32>()?[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    /// Each row's own maximum, wherever it sits, for a batch and for one row;
    /// the half-precision rows the generic reduction was slow on included.
    #[test]
    fn picks_each_row_s_maximum() -> Result<()> {
        let dev = Device::Cpu;
        let a = Tensor::new(&[[0.0f32, 3.0, 1.0, 2.0]], &dev)?;
        let b = Tensor::new(&[1.0f32, 0.0, 0.5, 4.0], &dev)?;
        assert_eq!(greedy_rows(&[&a, &b])?, vec![1, 3]);
        assert_eq!(greedy_token(&a.to_dtype(DType::BF16)?)?, 1);
        assert_eq!(greedy_token(&b.to_dtype(DType::F16)?)?, 3);
        Ok(())
    }
}
