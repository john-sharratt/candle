//! The batched sampler: the handle that drives the CUDA sampling kernel over
//! sequences whose state the caller owns.
//!
//! The sampler holds what is shared across a dispatch: the device count tables,
//! the argument buffer, and the DRY reference flags once they are uploaded. The
//! logits row is `vocab_size` wide, which a checkpoint may pad, but a row may
//! produce only the tokenizer's ids, so the padded tail carries no probability.
//! `sample_batch` is the plain case of scoring rows and committing them, which a
//! speculative step drives directly.

use super::*;

/// Stateless batched sampler that invokes the CUDA kernel.
///
/// This sampler does not own per-sequence state. Instead, callers pass
/// `SequenceSamplingState` references which are updated in place.
pub struct BatchedSampler {
    /// Device the sampler operates on.
    #[allow(dead_code)]
    pub(super) device: Device,

    /// Vocabulary size — the logits row width, which a checkpoint may pad.
    pub(super) vocab_size: usize,

    /// Tokens a row may produce: the tokenizer's last id + 1. The padded tail
    /// of each row past it carries no probability.
    pub(super) live_vocab: usize,

    /// Maximum recent token history length.
    pub(super) max_recent_len: usize,

    /// EOS token ID.
    pub(super) eos_tokens: TokenBuffer,

    /// Optional path to write penalty state during decoding.
    pub(super) penalty_log_path: Option<std::path::PathBuf>,

    /// The kernel's count tables, kept on the device between dispatches.
    pub(super) penalty_tables: Mutex<PenaltyTables>,

    /// The device half of a dispatch's argument pack: every small per-dispatch
    /// array, uploaded into this buffer in one copy and read back from it in
    /// one copy. Held and grown to the widest dispatch seen, so a dispatch
    /// allocates nothing; the readback that ends each dispatch is what lets the
    /// next one overwrite it.
    pub(super) args: Mutex<Option<Held<u32>>>,

    /// One byte per token naming what DRY leaves alone — see
    /// [`crate::dry_reference`]. Empty when no tokenizer was given, and then
    /// no token is exempt.
    pub(super) reference_flags: Vec<u8>,

    /// [`Self::reference_flags`] on the device, uploaded at the first dispatch
    /// and read by every one after.
    pub(super) reference_flags_dev: Mutex<Option<Held<u8>>>,
}

impl BatchedSampler {
    /// Create a new batched sampler. `live_vocab` is clamped to `vocab_size`.
    pub fn new(
        device: Device,
        vocab_size: usize,
        live_vocab: usize,
        max_recent_len: usize,
        eos_tokens: TokenBuffer,
        penalty_log_path: Option<std::path::PathBuf>,
    ) -> Self {
        Self {
            device,
            vocab_size,
            live_vocab: live_vocab.min(vocab_size),
            max_recent_len,
            eos_tokens,
            penalty_log_path,
            penalty_tables: Mutex::new(PenaltyTables::new(vocab_size)),
            args: Mutex::new(None),
            reference_flags: Vec::new(),
            reference_flags_dev: Mutex::new(None),
        }
    }

    /// The per-token flags DRY reads to leave a repeated reference alone,
    /// built from the tokenizer with [`crate::dry_reference::reference_flags`]
    /// at `vocab_size` entries.
    pub fn with_reference_flags(mut self, flags: Vec<u8>) -> Self {
        self.reference_flags = flags;
        self
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the maximum recent token history length.
    pub fn max_recent_len(&self) -> usize {
        self.max_recent_len
    }

    /// Sample tokens for a batch of sequences.
    ///
    /// # Arguments
    /// - `logits`: Batched logits tensor, shape `[batch_size, vocab_size]` or similar.
    /// - `states`: Mutable references to per-sequence sampling states.
    /// - `configs`: Per-sequence sampling configs.
    ///
    /// # Returns
    /// Sampled token IDs for each sequence. States are updated in place.
    ///
    /// One row per sequence, scored in one dispatch and committed — the plain
    /// case of [`Self::score_rows`] and [`Self::commit_row`], which a
    /// speculative step drives directly.
    pub fn sample_batch(
        &self,
        logits: &Tensor,
        states: &mut [&mut SequenceSamplingState],
        configs: &[&SamplingConfig],
    ) -> candle::Result<Vec<u32>> {
        let batch_size = states.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        let logits2d = self.flatten_to_2d(logits)?;
        let rows: Vec<RowSpec<'_>> = (0..batch_size)
            .map(|i| RowSpec {
                seq: i,
                logits_row: i,
                draft: None,
                ahead: &[],
            })
            .collect();
        let scored = self.score_rows(&logits2d, states, configs, &rows, None)?;
        let mut results = Vec::with_capacity(batch_size);
        for (i, scored) in scored.into_iter().enumerate() {
            let scored = scored.ok_or_else(|| {
                candle::Error::Msg(format!(
                    "sample_batch: row {i} has nothing ahead of it and was still scored unreachable"
                ))
            })?;
            results.push(self.commit_row(&logits2d, i, &mut *states[i], configs[i], scored)?);
        }
        Ok(results)
    }

    /// Flatten logits of rank 1/2/3 to `[batch, vocab]` (taking the last
    /// position for rank-3).
    pub(super) fn flatten_to_2d(&self, logits: &Tensor) -> candle::Result<Tensor> {
        match logits.dims().len() {
            1 => logits.unsqueeze(0),
            2 => Ok(logits.clone()),
            3 => {
                let seq_len = logits.dim(1)?;
                logits.i((.., seq_len - 1, ..))
            }
            n => Err(candle::Error::Msg(format!("unexpected logits rank: {n}"))),
        }
    }
}

/// Map a [`SamplingConfig`] to candle's `Sampling` strategy.
pub(super) fn config_to_sampling(
    config: &SamplingConfig,
) -> candle_transformers::generation::Sampling {
    use candle_transformers::generation::Sampling;
    if config.temperature <= 0.0 {
        Sampling::ArgMax
    } else if config.top_k > 0 && config.top_p < 1.0 {
        Sampling::TopKThenTopP {
            k: config.top_k as usize,
            p: config.top_p as f64,
            temperature: config.temperature as f64,
        }
    } else if config.top_k > 0 {
        Sampling::TopK {
            k: config.top_k as usize,
            temperature: config.temperature as f64,
        }
    } else if config.top_p < 1.0 {
        Sampling::TopP {
            p: config.top_p as f64,
            temperature: config.temperature as f64,
        }
    } else {
        Sampling::All {
            temperature: config.temperature as f64,
        }
    }
}
