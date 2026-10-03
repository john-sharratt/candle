//! Logit Processing and Sampling
//!
//! Functionality for modeling sampling strategies and logits processing in text generation
//! with support for temperature-based sampling, top-k filtering, nucleus sampling (top-p),
//! and combinations thereof.
use candle::{Context, DType, Error, Result, Tensor};
use rand::{distr::Distribution, SeedableRng};

#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,
    All { temperature: f64 },
    TopK { k: usize, temperature: f64 },
    TopP { p: f64, temperature: f64 },
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
    // Note that the rng is not used for the Gumbel-Softmax sampling.
    GumbelSoftmax { temperature: f64 },
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling: Sampling,
}

/// A sample's device work, enqueued and not yet read back
/// ([`LogitsProcessor::sample_issue`]).
///
/// Holding it lets a caller enqueue many rows' device work before reading any
/// of them, so the reads find their results already computed — one pipeline
/// drain for the batch rather than one per row.
pub enum PendingSample {
    /// The token itself, computed on the device (argmax, Gumbel-softmax).
    Token(Tensor),
    /// The probability vector the host-side sampling draws from.
    Probs(Tensor),
}

impl LogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self { rng, sampling }
    }

    pub fn new(
        seed: u64,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
    ) -> Self {
        let temperature = temperature.filter(|&v| v >= 1e-7);
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match top_p {
                None => match top_k {
                    None => Sampling::All { temperature },
                    Some(k) => Sampling::TopK { k, temperature },
                },
                Some(p) => match top_k {
                    None => Sampling::TopP { p, temperature },
                    Some(k) => Sampling::TopKThenTopP { k, p, temperature },
                },
            },
        };
        Self::from_sampling(seed, sampling)
    }

    /// Read back a device-computed token: an argmax index (rank 0 or 1) or a
    /// Gumbel-softmax draw.
    fn read_token(idx: &Tensor) -> Result<u32> {
        match idx.rank() {
            0 => idx.to_vec0::<u32>(),
            1 => idx
                .to_vec1::<u32>()?
                .first()
                .copied()
                .context("empty logits"),
            r => candle::bail!("unexpected argmax rank {r} for logits"),
        }
    }

    fn sample_multinomial(&mut self, prs: &Vec<f32>) -> Result<u32> {
        let distr = rand::distr::weighted::WeightedIndex::new(prs).map_err(Error::wrap)?;
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    /// top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed
    /// probability top_p. This way we never sample tokens that have very low probabilities and are
    /// less likely to go "off the rails".
    fn sample_topp(&mut self, prs: &mut Vec<f32>, top_p: f32) -> Result<u32> {
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                prs[*index] = 0.0;
            } else {
                cumsum += prs[*index];
            }
        }
        // Sample with clamped probabilities.
        self.sample_multinomial(prs)
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    fn sample_topk(&mut self, prs: &mut Vec<f32>, top_k: usize) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_multinomial(prs)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let index = self.sample_multinomial(&prs)?;
            Ok(indices[index as usize] as u32)
        }
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    // then top-p sampling.
    fn sample_topk_topp(&mut self, prs: &mut Vec<f32>, top_k: usize, top_p: f32) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_topp(prs, top_p)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let mut prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let sum_p = prs.iter().sum::<f32>();
            let index = if top_p <= 0.0 || top_p >= sum_p {
                self.sample_multinomial(&prs)?
            } else {
                self.sample_topp(&mut prs, top_p)?
            };
            Ok(indices[index as usize] as u32)
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        self.sample_f(logits, |_| {})
    }

    pub fn sample_f(&mut self, logits: &Tensor, f: impl FnOnce(&mut [f32])) -> Result<u32> {
        let pending = self.sample_issue(logits)?;
        self.sample_finish_f(pending, f)
    }

    /// The device half of a sample: every device op the strategy runs, enqueued
    /// and not read back. Argmax stays on the device (Candle's reduction, no
    /// full-vocab cast or transfer); the Gumbel draw and the probabilities are
    /// taken in F32 — doing them in bf16/f16 can be unstable.
    pub fn sample_issue(&self, logits: &Tensor) -> Result<PendingSample> {
        Ok(match &self.sampling {
            Sampling::ArgMax => PendingSample::Token(logits.argmax(candle::D::Minus1)?),
            Sampling::GumbelSoftmax { temperature } => {
                let logits = logits.to_dtype(DType::F32)?;
                PendingSample::Token(candle_nn::sampling::gumbel_softmax(
                    &logits,
                    *temperature,
                    candle::D::Minus1,
                )?)
            }
            Sampling::All { temperature }
            | Sampling::TopP { temperature, .. }
            | Sampling::TopK { temperature, .. }
            | Sampling::TopKThenTopP { temperature, .. } => {
                let logits = (logits.to_dtype(DType::F32)? / *temperature)?;
                PendingSample::Probs(candle_nn::ops::softmax_last_dim(&logits)?)
            }
        })
    }

    /// The host half of a sample: read back what [`Self::sample_issue`]
    /// enqueued and draw the token.
    pub fn sample_finish(&mut self, pending: PendingSample) -> Result<u32> {
        self.sample_finish_f(pending, |_| {})
    }

    fn sample_finish_f(
        &mut self,
        pending: PendingSample,
        f: impl FnOnce(&mut [f32]),
    ) -> Result<u32> {
        let prs = match pending {
            PendingSample::Token(t) => return Self::read_token(&t),
            PendingSample::Probs(p) => p,
        };
        let mut prs: Vec<f32> = prs.to_vec1()?;
        f(&mut prs);
        match &self.sampling {
            Sampling::All { .. } => self.sample_multinomial(&prs),
            Sampling::TopP { p, .. } => {
                if *p <= 0.0 || *p >= 1.0 {
                    // simply sample from the predicted probability distribution
                    self.sample_multinomial(&prs)
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    self.sample_topp(&mut prs, *p as f32)
                }
            }
            Sampling::TopK { k, .. } => self.sample_topk(&mut prs, *k),
            Sampling::TopKThenTopP { k, p, .. } => self.sample_topk_topp(&mut prs, *k, *p as f32),
            Sampling::ArgMax | Sampling::GumbelSoftmax { .. } => {
                candle::bail!("logits processor: a probability vector for a strategy that draws on the device")
            }
        }
    }
}
