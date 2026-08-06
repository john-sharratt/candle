//! Full-model **streaming** reference forward: loads one transformer block at a time from
//! the merged GGUF, runs it, and drops it before loading the next — so the 145 GB model
//! never has to be resident. This is the correctness-complete real-weight forward (not the
//! performance path; the batched/streaming engine integration is separate). One block
//! (~3.4 GB of MXFP4 experts) is resident at a time.

use candle::{DType, Device, IndexOp, Result, Tensor, D};

use super::config::Config;
use super::hyper::{HyperConnection, HyperParams};
use super::linear::QLinear;
use super::loader::{self, GgufModel};
use super::rope::RotaryCache;

/// A full DeepSeek-V4-Flash model that streams its blocks from disk on each forward.
pub struct StreamingModel {
    gguf: GgufModel,
    cfg: Config,
    device: Device,
    embed: Tensor,       // [vocab, dim] f32
    output_norm: Tensor, // [dim]
    lm_head: QLinear,
    hc: HyperConnection,
    hc_head: HyperParams,
    rope_compress: RotaryCache,
    rope_swa: RotaryCache,
}

impl StreamingModel {
    /// Open the merged GGUF and load the persistent (non-block) weights. Blocks are loaded
    /// lazily per forward.
    pub fn open(merged_path: &std::path::Path, device: &Device) -> Result<Self> {
        let mut gguf = GgufModel::open(std::slice::from_ref(&merged_path.to_path_buf()))?;
        let cfg = loader::config_from_gguf(&gguf)?;

        let embed = loader::dequant_f32(&mut gguf, "token_embd.weight", device)?;
        let output_norm = loader::dequant_f32(&mut gguf, "output_norm.weight", device)?;
        let lm_head = loader::qlinear(&mut gguf, "output.weight", device)?;
        let hc = HyperConnection::new(cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps);
        let hc_head = HyperParams {
            fn_w: loader::dequant_f32(&mut gguf, "output_hc_fn.weight", device)?.into(),
            base: loader::dequant_f32(&mut gguf, "output_hc_base.weight", device)?,
            scale: loader::dequant_f32(&mut gguf, "output_hc_scale.weight", device)?,
        };
        let rope_compress = RotaryCache::new(
            cfg.rope_head_dim,
            cfg.compress_rope_theta,
            cfg.original_seq_len,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
            device,
        )?;
        let rope_swa = RotaryCache::new(
            cfg.rope_head_dim,
            cfg.rope_theta,
            0,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
            device,
        )?;
        Ok(Self {
            gguf,
            cfg,
            device: device.clone(),
            embed,
            output_norm,
            lm_head,
            hc,
            hc_head,
            rope_compress,
            rope_swa,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Full-sequence forward, streaming blocks. `input_ids` `[1, s]` u32 → logits
    /// `[1, s, vocab]`. Loads/drops each block; slow but exact on the full model.
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let ids = input_ids.to_dtype(DType::U32)?;
        let flat = ids.reshape(b * s)?;
        let emb = self
            .embed
            .index_select(&flat, 0)?
            .reshape((b, s, self.cfg.dim))?;
        let mut h = self.hc.expand(&emb.to_dtype(DType::F32)?)?;

        for layer in 0..self.cfg.n_layers {
            let block = loader::load_block(&mut self.gguf, &self.cfg, layer, &self.device)?;
            let rope = if self.cfg.layer_kind(layer).compresses() {
                &self.rope_compress
            } else {
                &self.rope_swa
            };
            h = block.forward(&h, &ids, rope)?;
            // `block` (with its ~3.4 GB of resident MXFP4 experts) drops here.
        }

        let h = self.hc.head_reduce(&h, &self.hc_head)?;
        let h = rms_norm(&h, &self.output_norm, self.cfg.norm_eps)?;
        self.lm_head.forward(&h)
    }

    /// Greedy-decode `max_new` tokens after `prompt` (recompute-full-prefix each step).
    pub fn generate(&mut self, prompt: &[u32], max_new: usize) -> Result<Vec<u32>> {
        let mut ids = prompt.to_vec();
        let mut out = Vec::new();
        for _ in 0..max_new {
            let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &self.device)?;
            let logits = self.forward(&input)?;
            let next = logits
                .i((0, ids.len() - 1))?
                .argmax(D::Minus1)?
                .to_scalar::<u32>()?;
            ids.push(next);
            out.push(next);
        }
        Ok(out)
    }
}

fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = x.broadcast_div(&(ms + eps)?.sqrt()?)?;
    normed.broadcast_mul(&w.to_dtype(DType::F32)?)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    fn merged() -> std::path::PathBuf {
        std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4-merged.gguf")
    }

    /// Coherence gate: tokenize a real prompt, greedy-generate a few tokens through the
    /// full streaming model, and decode. Prints the continuation for inspection — coherent
    /// output is the ultimate proof the implementation is numerically correct on real
    /// weights. Ignored (needs merged file + CUDA + internet; ~minutes per token).
    #[test]
    #[ignore]
    fn full_model_generate_coherent() -> Result<()> {
        let path = merged();
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let mut model = StreamingModel::open(&path, &device)?;
        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        // DeepSeek-V4 is an instruct/reasoning model — use its chat markers, not a raw
        // base-LM completion. The special tokens are encoded as their registered ids.
        let prompt = "<｜begin▁of▁sentence｜><｜User｜>What is the capital of France? \
             Reply with only the city name.<｜Assistant｜>";
        let enc = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?;
        let ids: Vec<u32> = enc.get_ids().to_vec();
        eprintln!("[gen] prompt={prompt:?} ids={ids:?}");
        let gen = model.generate(&ids, 12)?;
        let text = tokenizer
            .decode(&gen, false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        eprintln!("[gen] generated ids={gen:?}");
        eprintln!("[gen] continuation={text:?}");
        assert!(!gen.is_empty());
        assert!(gen
            .iter()
            .all(|&t| (t as usize) < model.config().vocab_size));
        Ok(())
    }

    /// The full 43-layer DeepSeek-V4-Flash model runs end-to-end on real weights via block
    /// streaming and produces finite logits with a valid argmax token. Ignored (needs the
    /// merged file + CUDA; minutes per forward).
    #[test]
    #[ignore]
    fn full_model_streaming_forward() -> Result<()> {
        let path = merged();
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let mut model = StreamingModel::open(&path, &device)?;
        let vocab = model.config().vocab_size;
        // A short arbitrary token prompt (BOS=0 + a few ids).
        let ids = Tensor::from_vec(vec![0u32, 100, 200, 300, 400, 500], (1, 6), &device)?;
        let logits = model.forward(&ids)?;
        assert_eq!(logits.dims(), &[1, 6, vocab]);
        let last = logits.i((0, 5))?;
        let v = last.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        assert!(v.iter().all(|x| x.is_finite()), "non-finite logits");
        let next = last.argmax(D::Minus1)?.to_scalar::<u32>()?;
        assert!((next as usize) < vocab);
        let maxabs = v.iter().fold(0f32, |a, &x| a.max(x.abs()));
        eprintln!(
            "[ok] FULL MODEL streaming forward: logits {:?}, argmax={next}, max|logit|={maxabs:.2}",
            logits.dims()
        );
        Ok(())
    }
}
