//! The factored RoPE tables a latent model's layers rotate from: one per
//! frequency set, shared by every layer of that kind
//! (`docs/progressive_yarn.md` §5.3).
//!
//! DeepSeek-V4 has two sets — the compressing layers' YaRN and the SWA layers'
//! plain θ — so a model of forty-odd layers holds two tables, not forty-odd.

use candle::{Device, Result, Tensor};

use super::paged::{build_rope_table, ROPE_DIM};
use crate::models::rope_schedule::yarn_freqs;

/// The tables built so far, keyed by the frequency set.
#[derive(Default)]
pub struct LatentRopeTables {
    sets: Vec<((u64, usize), Tensor)>,
}

impl LatentRopeTables {
    /// The table for a layer rotating at `theta` over trained window
    /// `original_seq_len` (0: YaRN off), built the first time a set is asked
    /// for and shared after.
    ///
    /// `rope_factor`/`beta_fast`/`beta_slow` are the model's YaRN knobs, the
    /// same for every set of one checkpoint; grouping them would add a type that
    /// exists only to satisfy an argument count.
    pub fn for_layer(
        &mut self,
        theta: f64,
        original_seq_len: usize,
        rope_factor: f64,
        beta_fast: f64,
        beta_slow: f64,
        device: &Device,
    ) -> Result<Tensor> {
        let key = (theta.to_bits(), original_seq_len);
        if let Some((_, t)) = self.sets.iter().find(|(k, _)| *k == key) {
            return Ok(t.clone());
        }
        let freqs_v: Vec<f32> = yarn_freqs(
            ROPE_DIM,
            theta,
            original_seq_len,
            rope_factor,
            beta_fast,
            beta_slow,
        )
        .into_iter()
        .map(|f| f as f32)
        .collect();
        let freqs = Tensor::from_vec(freqs_v, ROPE_DIM / 2, device)?;
        let tab = build_rope_table(&freqs)?;
        self.sets.push((key, tab.clone()));
        Ok(tab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DeepSeek-V4's two kinds over forty layers build two tables: every layer
    /// of a kind gets the one its kind built first — the same storage, not an
    /// equal copy — and the two kinds' tables differ.
    #[test]
    fn layers_of_a_kind_share_one_table() -> Result<()> {
        let Ok(dev) = Device::new_cuda(0) else {
            eprintln!("skipping: CUDA device required");
            return Ok(());
        };
        let mut tables = LatentRopeTables::default();
        // (θ, window): the compressing layers' YaRN set, the SWA layers' plain one.
        let kinds = [(160_000.0, 65_536), (10_000.0, 0)];
        let mut first: Vec<Option<Tensor>> = vec![None, None];
        for layer in 0..40 {
            let k = layer % 2;
            let (theta, window) = kinds[k];
            let t = tables.for_layer(theta, window, 16.0, 32.0, 1.0, &dev)?;
            match &first[k] {
                None => first[k] = Some(t),
                Some(f) => assert_eq!(t.id(), f.id(), "layer {layer} built its own table"),
            }
        }
        assert_eq!(tables.sets.len(), 2);
        let (a, b) = (first[0].as_ref().unwrap(), first[1].as_ref().unwrap());
        assert_ne!(
            a.to_vec1::<f32>()?,
            b.to_vec1::<f32>()?,
            "the two kinds' frequency sets built the same table"
        );
        Ok(())
    }
}
