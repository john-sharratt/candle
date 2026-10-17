//! Speculative verify for the hybrid: rewinding a recurrence that has no
//! suffix to remove (`docs/deltanet_timeline.md`).
//!
//! A speculative step runs a block of proposed tokens through one forward and
//! then learns how many of them the model actually agrees with. For the
//! attention half that is free — paged KV is append-only, so a truncation to
//! the accepted length erases exactly the rejected tokens. For the DeltaNet
//! half it is not: `S` is a running sum over every token of the sequence, with
//! no per-token decomposition.
//!
//! So the block never writes `S`. Each DeltaNet layer reads its state and
//! writes the block's **timeline** — per row, the decay scan `G`, the key `k`,
//! the delta-rule value `v_new` and the raw conv input `x` — and the layer stays
//! **pending** past the forward's commit. Every one of those row quantities
//! depends only on the rows before it, so once the accept walk knows the block
//! keeps `m` rows, the **fold** applies the first `m` and nothing has to be
//! replayed:
//!
//! ```text
//!   S    ← S·e^{G[m−1]} + Σ_{t<m} v_new[t] ⊗ k[t]·e^{G[m−1]−G[t]}
//!   tail ← the last K−1 of [tail | x[0] .. x[m−1]]
//! ```
//!
//! The fold is the chunked scan's own state carry over those `m` rows, written
//! in the same rounding-pinned operations, so the state is bit for bit what the
//! model would have reached had only those `m` tokens ever been decoded. A full
//! accept folds every row; a step that keeps none discards the timelines and
//! the state stands as the step entered it.
//!
//! **Which spans wait.** Only the armed cohort — the sequences `begin_verify`
//! named this step — is folded at the reconcile. Every other short span (a
//! plain decode row, a short prompt) folds at its forward's commit, because
//! nothing will ever ask it to keep fewer rows.

use candle::{Result, Tensor};

/// The rows of `logits` a verify block scored, split off a wave's output.
///
/// The head emits one row per scored position in wave order, and a verify
/// wave's order is `[plain decode rows | each block's rows]` — this is the
/// split, kept beside the fold because the two are the same bookkeeping seen
/// from either end.
pub fn split_block_rows(
    logits: &[Tensor],
    n_plain: usize,
    block_lens: &[usize],
) -> Result<(Vec<Tensor>, Vec<Vec<Tensor>>)> {
    let want: usize = n_plain + block_lens.iter().sum::<usize>();
    if logits.len() != want {
        candle::bail!(
            "qwen35 verify: wave scored {} rows, expected {want} ({n_plain} plain + \
             blocks {block_lens:?}) — the head did not score every verify row",
            logits.len()
        );
    }
    let plain = logits[..n_plain].to_vec();
    let mut blocks = Vec::with_capacity(block_lens.len());
    let mut off = n_plain;
    for &l in block_lens {
        blocks.push(logits[off..off + l].to_vec());
        off += l;
    }
    Ok((plain, blocks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::delta_net::{
        fold_stores, DeltaNetDims, LayerKind, RecurrentStateStore, SpanWrite, TimelineLayout,
    };
    use candle::{DType, Device};

    fn row(v: f32) -> Tensor {
        Tensor::full(v, (1, 4), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
    }

    #[test]
    fn block_rows_split_after_the_plain_prefix() {
        let rows: Vec<Tensor> = (0..6).map(|i| row(i as f32)).collect();
        let (plain, blocks) = split_block_rows(&rows, 2, &[3, 1]).unwrap();
        assert_eq!(plain.len(), 2);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].len(), 3);
        assert_eq!(blocks[1].len(), 1);
        let first = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert_eq!(first(&plain[0]), 0.0);
        assert_eq!(first(&blocks[0][0]), 2.0);
        assert_eq!(first(&blocks[1][0]), 5.0);
    }

    /// A short count is the symptom of the head scoring only each prefill
    /// span's LAST row, which is what it does on an ordinary wave — so it must
    /// be an error here, not a silent misalignment of every block's argmaxes.
    #[test]
    fn a_short_row_count_is_refused() {
        let rows: Vec<Tensor> = (0..3).map(|i| row(i as f32)).collect();
        let err = split_block_rows(&rows, 1, &[3]).unwrap_err().to_string();
        assert!(err.contains("score every verify row"), "{err}");
    }

    /// **The cohort fold keeps each store's own rows.** Two stores, each with a
    /// three-row block pending over one layer (rows `v_new = 1, 2, 4`, no
    /// decay): one keeps two rows, one keeps none. The first moves by exactly
    /// `1 + 2`, the second not at all, and neither is left pending.
    #[test]
    fn the_cohort_fold_keeps_each_stores_own_rows() -> Result<()> {
        let d = DeltaNetDims {
            head_dim: 4,
            n_k_heads: 1,
            n_v_heads: 2,
            conv_kernel: 3,
        };
        let kinds = [LayerKind::DeltaNet];
        let dev = Device::Cpu;
        let block = |store: &mut RecurrentStateStore| -> Result<()> {
            store.begin_wave()?;
            let (_, write) = store.layer_span_mut(0, 3)?;
            let SpanWrite::Timeline(tl) = write else {
                candle::bail!("a three-row span writes a timeline");
            };
            let v = Tensor::new(&[1f32, 2., 4.], &dev)?
                .reshape((3, 1, 1))?
                .broadcast_as((3, 2, 4))?
                .contiguous()?;
            TimelineLayout::of(&d).write_rows(
                &tl,
                &Tensor::zeros((3, 2), DType::F32, &dev)?,
                &Tensor::ones((3, 1, 4), DType::F32, &dev)?,
                &v,
                &Tensor::ones((3, d.conv_dim()), DType::F32, &dev)?,
            )?;
            store.commit_wave();
            Ok(())
        };
        let mut a = RecurrentStateStore::new(&kinds, &d, &dev)?;
        let mut b = RecurrentStateStore::new(&kinds, &d, &dev)?;
        block(&mut a)?;
        block(&mut b)?;
        fold_stores(&mut [(&mut a, Some(2)), (&mut b, Some(0))])?;
        assert!(!a.has_pending() && !b.has_pending());
        let s = |st: &RecurrentStateStore| -> Result<Vec<f32>> {
            st.layer_state(0)?.s.flatten_all()?.to_vec1()
        };
        assert!(s(&a)?.iter().all(|&x| x == 3.0), "{:?}", s(&a)?);
        assert!(s(&b)?.iter().all(|&x| x == 0.0), "{:?}", s(&b)?);
        Ok(())
    }
}
