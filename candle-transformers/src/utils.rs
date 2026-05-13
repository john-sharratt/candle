//! Apply penalty and repeat_kv

use candle::{Result, Tensor};

pub fn apply_repeat_penalty(logits: &Tensor, penalty: f32, context: &[u32]) -> Result<Tensor> {
    let device = logits.device();
    let mut logits = logits.to_dtype(candle::DType::F32)?.to_vec1::<f32>()?;
    let mut already_seen = std::collections::HashSet::new();
    for token_id in context {
        if already_seen.contains(token_id) {
            continue;
        }
        already_seen.insert(token_id);
        if let Some(logit) = logits.get_mut(*token_id as usize) {
            if *logit >= 0. {
                *logit /= penalty
            } else {
                *logit *= penalty
            }
        }
    }
    let logits_len = logits.len();
    Tensor::from_vec(logits, logits_len, device)
}

/// Repeats a key or value tensor for grouped query attention
/// The input tensor should have a shape `(batch, num_kv_heads, seq_len, head_dim)`,
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        // Using cat is faster than a broadcast as it avoids going through a potentially
        // strided copy.
        // https://github.com/huggingface/candle/pull/2043
        // Optimize common cases to avoid vec allocation overhead
        let repeated = match n_rep {
            2 => Tensor::cat(&[&xs, &xs], 2)?,
            4 => {
                let doubled = Tensor::cat(&[&xs, &xs], 2)?;
                Tensor::cat(&[&doubled, &doubled], 2)?
            }
            8 => {
                let doubled = Tensor::cat(&[&xs, &xs], 2)?;
                let quad = Tensor::cat(&[&doubled, &doubled], 2)?;
                Tensor::cat(&[&quad, &quad], 2)?
            }
            _ => Tensor::cat(&vec![&xs; n_rep], 2)?,
        };
        repeated.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}
