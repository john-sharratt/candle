//! GGUF loading for the reference model: arch detection, the tensor-name
//! schema, and dequantized-to-F32 weight assembly.
//!
//! The name schema is the llama.cpp `qwen35`/`qwen35moe` arch (design doc
//! §7.1, frozen from `gguf-py/gguf/constants.py` + `qwen35.cpp`). Everything
//! here dequantizes to F32 — this is the parity/reference path; the
//! production loader (KO twins, expert mmap refs, int8) is Phase 4.

use candle::quantized::gguf_file::{Content, Value};
use candle::{Device, Result, Tensor};
use std::io::{Read, Seek};

use super::attention::{AttentionWeights, RopeTables};
use super::config::{LayerKind, Qwen35Config};
use crate::models::delta_net::DeltaNetWeights;
use super::model::{LayerFfn, LayerMix, Qwen35Layer, Qwen35Model};
use super::moe::{FfnWeights, MoeWeights};

/// The architecture prefix, from `general.architecture` with a probe
/// fallback over the known names (mirroring `quantized_qwen3_moe`'s loader
/// posture: trust the declaration, survive its absence).
pub fn detect_arch(content: &Content) -> String {
    if let Some(Value::String(a)) = content.metadata.get("general.architecture") {
        return a.clone();
    }
    for candidate in ["qwen35moe", "qwen35"] {
        if content
            .metadata
            .keys()
            .any(|k| k.starts_with(&format!("{candidate}.")))
        {
            return candidate.to_string();
        }
    }
    "qwen35".to_string()
}

/// Every tensor name a layer of `kind` owns (MoE-ness decided separately, by
/// tensor presence, exactly like the qwen3 MoE loader).
pub fn layer_tensor_names(layer: usize, kind: LayerKind) -> Vec<String> {
    let p = format!("blk.{layer}");
    let mut names = vec![
        format!("{p}.attn_norm.weight"),
        format!("{p}.post_attention_norm.weight"),
    ];
    match kind {
        LayerKind::Attention => names.extend([
            format!("{p}.attn_q.weight"),
            format!("{p}.attn_k.weight"),
            format!("{p}.attn_v.weight"),
            format!("{p}.attn_output.weight"),
            format!("{p}.attn_q_norm.weight"),
            format!("{p}.attn_k_norm.weight"),
        ]),
        LayerKind::DeltaNet => names.extend([
            format!("{p}.attn_qkv.weight"),
            format!("{p}.attn_gate.weight"),
            format!("{p}.ssm_conv1d.weight"),
            format!("{p}.ssm_dt.bias"),
            format!("{p}.ssm_a"),
            format!("{p}.ssm_beta.weight"),
            format!("{p}.ssm_alpha.weight"),
            format!("{p}.ssm_norm.weight"),
            format!("{p}.ssm_out.weight"),
        ]),
    }
    names
}

/// Dense-FFN tensor names for a layer.
pub fn dense_ffn_names(layer: usize) -> [String; 3] {
    let p = format!("blk.{layer}");
    [
        format!("{p}.ffn_gate.weight"),
        format!("{p}.ffn_up.weight"),
        format!("{p}.ffn_down.weight"),
    ]
}

/// MoE tensor names for a layer (merged-3D experts + shared expert).
pub fn moe_ffn_names(layer: usize) -> [String; 8] {
    let p = format!("blk.{layer}");
    [
        format!("{p}.ffn_gate_inp.weight"),
        format!("{p}.ffn_gate_exps.weight"),
        format!("{p}.ffn_up_exps.weight"),
        format!("{p}.ffn_down_exps.weight"),
        format!("{p}.ffn_gate_inp_shexp.weight"),
        format!("{p}.ffn_gate_shexp.weight"),
        format!("{p}.ffn_up_shexp.weight"),
        format!("{p}.ffn_down_shexp.weight"),
    ]
}

struct Gguf<'a, R: Read + Seek> {
    content: &'a Content,
    reader: &'a mut R,
    device: Device,
}

impl<R: Read + Seek> Gguf<'_, R> {
    fn f32_tensor(&mut self, name: &str) -> Result<Tensor> {
        let qt = self.content.tensor(self.reader, name, &self.device)?;
        qt.dequantize(&self.device)
    }

    fn has(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }
}

/// Load the reference model from a GGUF (any quantization; dequantized to
/// F32 on `device`). Handles dense `qwen35` and `qwen35moe`, tied LM heads,
/// mixed dense/MoE stacks, and skips trailing MTP blocks.
pub fn load_reference_model<R: Read + Seek>(
    content: &Content,
    reader: &mut R,
    device: &Device,
) -> Result<Qwen35Model> {
    let arch = detect_arch(content);
    let cfg = Qwen35Config::from_gguf_metadata(&arch, &content.metadata)?;
    let mut g = Gguf {
        content,
        reader,
        device: device.clone(),
    };

    let embed = g.f32_tensor("token_embd.weight")?;
    let final_norm = g.f32_tensor("output_norm.weight")?;
    let lm_head = if g.has("output.weight") {
        g.f32_tensor("output.weight")?
    } else {
        embed.clone()
    };

    let mut layers = Vec::with_capacity(cfg.num_layers);
    for li in 0..cfg.num_layers {
        let p = format!("blk.{li}");
        let kind = cfg.layer_kinds[li];

        let mix = match kind {
            LayerKind::Attention => LayerMix::Attention(AttentionWeights {
                wq: g.f32_tensor(&format!("{p}.attn_q.weight"))?,
                wk: g.f32_tensor(&format!("{p}.attn_k.weight"))?,
                wv: g.f32_tensor(&format!("{p}.attn_v.weight"))?,
                wo: g.f32_tensor(&format!("{p}.attn_output.weight"))?,
                q_norm: g.f32_tensor(&format!("{p}.attn_q_norm.weight"))?,
                k_norm: g.f32_tensor(&format!("{p}.attn_k_norm.weight"))?,
            }),
            LayerKind::DeltaNet => LayerMix::DeltaNet(DeltaNetWeights {
                wqkv: g.f32_tensor(&format!("{p}.attn_qkv.weight"))?,
                wz: g.f32_tensor(&format!("{p}.attn_gate.weight"))?,
                w_beta: g.f32_tensor(&format!("{p}.ssm_beta.weight"))?,
                w_alpha: g.f32_tensor(&format!("{p}.ssm_alpha.weight"))?,
                dt_bias: g.f32_tensor(&format!("{p}.ssm_dt.bias"))?,
                a: g.f32_tensor(&format!("{p}.ssm_a"))?,
                conv: {
                    // GGUF ne = {d_conv, conv_dim} → candle [conv_dim, d_conv],
                    // already the [C, K] layout causal_conv1d takes.
                    g.f32_tensor(&format!("{p}.ssm_conv1d.weight"))?
                },
                norm: g.f32_tensor(&format!("{p}.ssm_norm.weight"))?,
                w_out: g.f32_tensor(&format!("{p}.ssm_out.weight"))?,
            }),
        };

        // MoE vs dense by tensor presence, per layer.
        let ffn = if g.has(&format!("{p}.ffn_gate_inp.weight")) {
            let moe_cfg = cfg.moe.ok_or_else(|| {
                candle::Error::Msg(format!(
                    "blk.{li} has a router but the metadata declares no experts"
                ))
            })?;
            let router = g.f32_tensor(&format!("{p}.ffn_gate_inp.weight"))?;
            let gate3 = g.f32_tensor(&format!("{p}.ffn_gate_exps.weight"))?;
            let up3 = g.f32_tensor(&format!("{p}.ffn_up_exps.weight"))?;
            let down3 = g.f32_tensor(&format!("{p}.ffn_down_exps.weight"))?;
            let n_e = moe_cfg.n_experts;
            let mut experts = Vec::with_capacity(n_e);
            for e in 0..n_e {
                experts.push(FfnWeights {
                    gate: gate3.get(e)?.contiguous()?,
                    up: up3.get(e)?.contiguous()?,
                    down: down3.get(e)?.contiguous()?,
                });
            }
            LayerFfn::Moe(MoeWeights {
                router,
                experts,
                shared: FfnWeights {
                    gate: g.f32_tensor(&format!("{p}.ffn_gate_shexp.weight"))?,
                    up: g.f32_tensor(&format!("{p}.ffn_up_shexp.weight"))?,
                    down: g.f32_tensor(&format!("{p}.ffn_down_shexp.weight"))?,
                },
                shared_gate: g
                    .f32_tensor(&format!("{p}.ffn_gate_inp_shexp.weight"))?
                    .reshape((1, cfg.hidden_size))?,
                n_experts_used: moe_cfg.n_experts_used,
                norm_topk_prob: moe_cfg.norm_topk_prob,
                weights_scale: 1.0,
            })
        } else {
            LayerFfn::Dense(FfnWeights {
                gate: g.f32_tensor(&format!("{p}.ffn_gate.weight"))?,
                up: g.f32_tensor(&format!("{p}.ffn_up.weight"))?,
                down: g.f32_tensor(&format!("{p}.ffn_down.weight"))?,
            })
        };

        layers.push(Qwen35Layer {
            attn_norm: g.f32_tensor(&format!("{p}.attn_norm.weight"))?,
            post_attn_norm: g.f32_tensor(&format!("{p}.post_attention_norm.weight"))?,
            mix,
            ffn,
        });
    }

    let rope = RopeTables::new(
        cfg.rope_dim,
        cfg.rope_theta,
        cfg.max_position_embeddings.min(65_536),
        device,
    )?;
    Ok(Qwen35Model {
        cfg,
        embed,
        layers,
        final_norm,
        lm_head,
        rope,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real-weights validation on the pinned Qwen3.5-0.8B BF16 GGUF (the
    /// smallest hybrid — full-precision reference fits CPU RAM). Four
    /// checks:
    /// 1. the loader + forward produce finite logits over a real prompt;
    /// 2. segmented forward ≡ one-shot forward **on real weights** — the
    ///    contract every unit test proves on synthetic weights, now proven
    ///    against the actual checkpoint;
    /// 3. greedy continuation is deterministic and non-degenerate (no
    ///    single-token collapse);
    /// 4. the continuation is *semantically* right — the model completes a
    ///    factual prompt correctly. This is the check that catches geometry
    ///    that is self-consistent but wrong (a mis-parsed rotary width
    ///    passes 1–3 and fails here).
    ///
    /// Both revisions pinned (the C10 lesson: upstream re-uploads must never
    /// silently shift a validation).
    #[test]
    #[ignore = "reads the pinned Qwen3.5-0.8B GGUF from the HF cache (large download)"]
    fn qwen35_0_8b_real_weights_reference() -> Result<()> {
        use crate::models::batch_test::test_helpers::hf_get;
        use candle::quantized::gguf_file::Content;
        use hf_hub::RepoType;
        use std::io::{BufReader, Seek, SeekFrom};

        // `hf_get` rather than `hf_hub` directly: cache first, then a
        // resumable IPv4-only download. Plain hf-hub gets its connection
        // dropped on the networks this is developed over.
        let path = hf_get(
            "unsloth/Qwen3.5-0.8B-GGUF",
            RepoType::Model,
            "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
            "Qwen3.5-0.8B-BF16.gguf",
        )?;

        let file = std::fs::File::open(&path)?;
        let mut reader = BufReader::new(file);
        let content = Content::read(&mut reader)?;
        let arch = detect_arch(&content);
        println!("arch = {arch}");
        reader.seek(SeekFrom::Start(0))?;

        let device = Device::Cpu;
        let model = load_reference_model(&content, &mut reader, &device)?;
        println!(
            "loaded: {} layers ({} attention / {} deltanet), hidden {}, vocab {}, \
             rotary {}/{}",
            model.cfg.num_layers,
            model.cfg.n_attention_layers(),
            model.cfg.n_delta_net_layers(),
            model.cfg.hidden_size,
            model.cfg.vocab_size,
            model.cfg.rope_dim,
            model.cfg.attn_head_dim,
        );

        // The tokenizer comes from the source repo the GGUF was converted
        // from — the GGUF itself ships no `tokenizer.json`. Prove the two
        // agree before trusting any id: same vocabulary size, and the same
        // string at sampled ids.
        let tok_path = hf_get(
            "Qwen/Qwen3.5-0.8B",
            RepoType::Model,
            "2fc06364715b967f1860aea9cf38778875588b17",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::Msg(format!("load tokenizer: {e}")))?;

        let gguf_tokens = match content.metadata.get("tokenizer.ggml.tokens") {
            Some(Value::Array(a)) => a,
            _ => candle::bail!("gguf carries no tokenizer.ggml.tokens"),
        };
        assert_eq!(
            gguf_tokens.len(),
            model.cfg.vocab_size,
            "gguf token table disagrees with the parsed vocab size"
        );
        // The tokenizer covers a *prefix* of the embedding rows: this repo's
        // `tokenizer.json` stops at 248070, the GGUF table continues with the
        // audio/TTS specials the shared vocabulary reserves (248070–248076)
        // and then `[PAD…]` filler out to a multiple of 128. Every id the
        // tokenizer does know must agree; the tail is unreachable rows, not
        // an off-by-something.
        let tok_vocab = tokenizer.get_vocab_size(true);
        assert!(
            tok_vocab <= model.cfg.vocab_size,
            "tokenizer has {tok_vocab} ids but the checkpoint only embeds {} — \
             wrong source repo",
            model.cfg.vocab_size
        );
        println!(
            "tokenizer {tok_vocab} ids over {} embedded rows",
            model.cfg.vocab_size
        );
        for id in [1u32, 100, 1_000, 10_000, 100_000, 248_000] {
            assert!(id < tok_vocab as u32, "sample id {id} is past the tokenizer");
            let from_gguf = match &gguf_tokens[id as usize] {
                Value::String(s) => s.clone(),
                other => candle::bail!("gguf token {id} is not a string: {other:?}"),
            };
            let from_json = tokenizer
                .id_to_token(id)
                .ok_or_else(|| candle::Error::Msg(format!("tokenizer has no id {id}")))?;
            assert_eq!(
                from_gguf, from_json,
                "tokenizer/checkpoint disagree at id {id}"
            );
        }

        let prompt_text = "The capital of France is";
        let prompt: Vec<u32> = tokenizer
            .encode(prompt_text, false)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        println!("prompt {prompt_text:?} → {prompt:?}");
        assert!(!prompt.is_empty());

        // (1) + (2): one-shot vs segmented logits on real weights.
        let mut s_full = model.new_session()?;
        let logits_full = model.forward(&prompt, &mut s_full)?;
        let last_full = logits_full.narrow(0, prompt.len() - 1, 1)?;
        let max_abs = last_full
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(max_abs.is_finite(), "non-finite logits: {max_abs}");

        let mut s_seg = model.new_session()?;
        let l1 = model.forward(&prompt[..2], &mut s_seg)?;
        let l2 = model.forward(&prompt[2..], &mut s_seg)?;
        let seg = Tensor::cat(&[l1, l2], 0)?;
        let diff = logits_full
            .sub(&seg)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            diff < 2e-3,
            "segmented forward diverged from one-shot on real weights: {diff}"
        );

        // (3): deterministic, non-degenerate greedy continuation.
        // 8 tokens, not more: every assertion below — determinism across two
        // sessions, ≥ 4 distinct ids, "Paris" in the decode (the argmax after
        // the prompt is already " Paris") — discriminates just as well at 8,
        // and each extra token is a full single-token CPU forward, twice.
        let greedy = |state: &mut super::super::model::SessionState,
                      first: u32|
         -> Result<Vec<u32>> {
            let mut ids = Vec::with_capacity(8);
            let mut tok = first;
            for _ in 0..8 {
                let l = model.forward(&[tok], state)?;
                tok = l
                    .get(0)?
                    .argmax(0)?
                    .to_scalar::<u32>()? ;
                ids.push(tok);
            }
            Ok(ids)
        };
        let first = last_full.get(0)?.argmax(0)?.to_scalar::<u32>()?;
        let cont_a = greedy(&mut s_full, first)?;
        let mut s_b = model.new_session()?;
        model.forward(&prompt, &mut s_b)?;
        let cont_b = greedy(&mut s_b, first)?;
        assert_eq!(cont_a, cont_b, "greedy continuation must be deterministic");
        let distinct: std::collections::HashSet<u32> = cont_a.iter().copied().collect();
        assert!(
            distinct.len() >= 4,
            "degenerate continuation (only {} distinct tokens): {cont_a:?}",
            distinct.len()
        );

        // (4): the semantic check. A stack whose geometry is subtly wrong —
        // a rotary width read as the full head, a swapped SSM head count —
        // still produces finite, deterministic, varied tokens. Only the text
        // tells them apart.
        let mut full_ids = vec![first];
        full_ids.extend_from_slice(&cont_a);
        let completion = tokenizer
            .decode(&full_ids, false)
            .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
        println!("greedy continuation ids: {full_ids:?}");
        println!("{prompt_text}|{completion}");
        assert!(
            completion.contains("Paris"),
            "the model did not complete {prompt_text:?} with Paris — got {completion:?}"
        );
        Ok(())
    }

    #[test]
    fn delta_net_layer_names_match_the_frozen_schema() {
        let names = layer_tensor_names(2, LayerKind::DeltaNet);
        for expect in [
            "blk.2.attn_norm.weight",
            "blk.2.post_attention_norm.weight",
            "blk.2.attn_qkv.weight",
            "blk.2.attn_gate.weight",
            "blk.2.ssm_conv1d.weight",
            "blk.2.ssm_dt.bias",
            "blk.2.ssm_a",
            "blk.2.ssm_beta.weight",
            "blk.2.ssm_alpha.weight",
            "blk.2.ssm_norm.weight",
            "blk.2.ssm_out.weight",
        ] {
            assert!(names.iter().any(|n| n == expect), "missing {expect}");
        }
        assert_eq!(names.len(), 11);
    }

    #[test]
    fn attention_layer_names_match_the_frozen_schema() {
        let names = layer_tensor_names(3, LayerKind::Attention);
        for expect in [
            "blk.3.attn_q.weight",
            "blk.3.attn_k.weight",
            "blk.3.attn_v.weight",
            "blk.3.attn_output.weight",
            "blk.3.attn_q_norm.weight",
            "blk.3.attn_k_norm.weight",
        ] {
            assert!(names.iter().any(|n| n == expect), "missing {expect}");
        }
        assert_eq!(names.len(), 8);
    }

    #[test]
    fn moe_names_include_router_experts_and_gated_shared_expert() {
        let names = moe_ffn_names(7);
        assert!(names.contains(&"blk.7.ffn_gate_inp.weight".to_string()));
        assert!(names.contains(&"blk.7.ffn_gate_exps.weight".to_string()));
        assert!(names.contains(&"blk.7.ffn_gate_inp_shexp.weight".to_string()));
        assert!(names.contains(&"blk.7.ffn_down_shexp.weight".to_string()));
        assert_eq!(dense_ffn_names(7).len(), 3);
    }
}
