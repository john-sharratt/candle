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

/// The ids `tokenizer` can name — its largest id + 1 — or `None` for an empty
/// vocabulary. It walks the whole vocabulary, so it is read once per tokenizer,
/// never inside a timed phase.
pub fn token_ids(tokenizer: &Tokenizer) -> Option<usize> {
    tokenizer
        .get_vocab(true)
        .values()
        .max()
        .map(|&id| id as usize + 1)
}

/// The columns of a `row_width` logits row that are tokens: [`token_ids`]
/// clamped to the row. A checkpoint pads its output projection past the
/// tokenizer, and the padded tail of a row is not a token.
pub fn live_vocab(token_ids: Option<usize>, row_width: usize) -> usize {
    token_ids.map_or(row_width, |n| n.min(row_width))
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

    /// The bound is the largest id + 1, not the entry count: ids 0, 3 and 7
    /// name 8 columns.
    #[test]
    fn token_ids_is_the_largest_id_plus_one() {
        let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{"a":0,"c":3,"b":7},"unk_token":"a"}}"#;
        let tokenizer = Tokenizer::from_bytes(json.as_bytes()).unwrap();
        assert_eq!(token_ids(&tokenizer), Some(8));
    }

    /// A padded row is clamped to the tokenizer; a row narrower than the
    /// tokenizer, or no tokenizer bound at all, keeps the whole row.
    #[test]
    fn live_vocab_clamps_to_the_row() {
        assert_eq!(live_vocab(Some(248_077), 248_320), 248_077);
        assert_eq!(live_vocab(Some(10), 8), 8);
        assert_eq!(live_vocab(None, 8), 8);
    }
}
