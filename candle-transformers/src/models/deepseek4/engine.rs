//! CUDA batched-inference engine for DeepSeek-V4-Flash.
//!
//! This is the **fast** path (vs the block-streaming `StreamingModel` reference): the
//! non-expert weights (embedding, attention, mHC params, norms, router, shared experts —
//! ~8 GB) are resident in VRAM, and the 147 GB of routed MXFP4 experts live in the shared
//! [`ExpertCache`] (VRAM hot pool + pinned host RAM, reordered to the 4-bit MXFP4_KO
//! exponent-collapse int8 twin at stage-in — an exact byte-permute, no requant). The forward reuses the **validated** decode math (`Attention` /
//! `HyperConnection` / `Gate`, all locked against `model.py`), but each layer's routed MoE
//! runs against resident experts instead of re-reading them from disk every forward — the
//! streaming reference's bottleneck.
//!
//! The reference forward (`streaming.rs` / `transformer.rs`) remains the correctness oracle:
//! the engine output is validated against it (same "Paris"), then measured for speed.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use candle::quantized::{get_vram_info, gguf_file, GgmlDType, Int8Mode, MmapRegistration};
use candle::{DType, Device, Result, Tensor, D};
use memmap2::MmapOptions;

use crate::models::expert_lre::{ExpertCache, MmapExpertRef, MoeInput};

use super::config::Config;
use super::hyper::{HyperConnection, HyperParams};
use super::linear::QLinear;
use super::loader::{self, GgufModel};
use super::moe::{Expert, Gate, ScoreFunc};
use super::rope::RotaryCache;

/// One transformer layer's resident (non-routed-expert) weights. The routed experts for this
/// layer live in the shared [`ExpertCache`], indexed by `moe_layer_idx`.
struct EngineLayer {
    attn: super::attention::Attention,
    hc_attn: HyperParams,
    hc_ffn: HyperParams,
    attn_norm: Tensor,
    ffn_norm: Tensor,
    /// MoE router (`sqrtsoftplus`/`noaux_tc`/hash — reused from the validated base). Produces
    /// `(weights, indices)`; the routed experts execute via the `ExpertCache`.
    gate: Gate,
    /// Always-on shared expert (resident, not routed).
    shared: Expert,
    /// This layer's index into the `ExpertCache` (0-based over MoE layers).
    moe_layer_idx: usize,
}

/// The resident DeepSeek-V4-Flash model for the CUDA batched engine.
pub struct Dsv4Engine {
    cfg: Config,
    embed: Tensor,
    layers: Vec<EngineLayer>,
    hc: HyperConnection,
    hc_head: HyperParams,
    output_norm: Tensor,
    lm_head: QLinear,
    /// RoPE tables: `rope_compress` (long-context theta + YaRN) for CSA/HCA layers,
    /// `rope_swa` (base theta, YaRN off) for sliding-window layers.
    rope_compress: RotaryCache,
    rope_swa: RotaryCache,
    /// Resident-expert pool: streams the routed MXFP4 experts pinned↔VRAM per wave.
    experts: Arc<ExpertCache>,
    _mmap: Arc<memmap2::Mmap>,
    _reg: Option<MmapRegistration>,
    device: Device,
    int8mode: Int8Mode,
}

impl Dsv4Engine {
    /// Load the merged single-file GGUF into the resident engine model.
    ///
    /// Non-expert weights load to `device` (VRAM); the routed experts are registered as
    /// byte-ranges into the page-locked mmap and streamed by the `ExpertCache` (int8-KO when
    /// `int8mode` is enabled — the 2×-FP16 grouped GEMM path).
    pub fn load(merged_path: &Path, device: &Device, int8mode: Int8Mode) -> Result<Self> {
        // GgufModel handles config + convenient loading of the resident (non-expert) tensors.
        let mut gguf = GgufModel::open(std::slice::from_ref(&merged_path.to_path_buf()))?;
        let cfg = loader::config_from_gguf(&gguf)?;

        // A raw mmap + its Content give the expert byte-offsets for the ExpertCache. We deliberately
        // do NOT `register_mmap_cuda` here: that page-locks the ENTIRE model file (~156 GB), and the
        // CUDA path already repacks every expert into the page-locked `PinnedPool` (~100 GB) and
        // loads from THERE (`load_from_pinned`), never DMAing native bytes from the mmap. Registering
        // the mmap on top double-locks the experts (156 + 100 GB > host RAM) → `cuMemAllocHost` OOM.
        // Staging reads the mmap via ordinary CPU access (pageable page cache), which needs no pin.
        let file = File::open(merged_path)?;
        let mmap = Arc::new(unsafe { MmapOptions::new().map(&file)? });
        let reg: Option<MmapRegistration> = None;
        let ct = gguf_file::Content::read(&mut std::io::Cursor::new(&mmap[..]))?;

        // ── Build the ExpertCache from the 3D-merged MXFP4 routed experts ──
        let n_expert = cfg.n_routed_experts;
        let moe_layers: Vec<usize> = (0..cfg.n_layers)
            .filter(|&i| {
                ct.tensor_infos
                    .contains_key(&format!("blk.{i}.ffn_gate_exps.weight"))
            })
            .collect();
        let mut all_host_refs: Vec<Vec<MmapExpertRef>> = Vec::with_capacity(moe_layers.len());
        for &i in &moe_layers {
            let p = format!("blk.{i}");
            let get = |suffix: &str| -> Result<&gguf_file::TensorInfo> {
                ct.tensor_infos
                    .get(&format!("{p}.{suffix}"))
                    .ok_or_else(|| candle::Error::msg(format!("missing {p}.{suffix}")))
            };
            let (gi, ui, di) = (
                get("ffn_gate_exps.weight")?,
                get("ffn_up_exps.weight")?,
                get("ffn_down_exps.weight")?,
            );
            // Per-expert byte length (product of dims after the expert axis), per-projection dtype.
            let ebytes = |info: &gguf_file::TensorInfo| {
                info.shape.dims()[1..].iter().product::<usize>() / info.ggml_dtype.block_size()
                    * info.ggml_dtype.type_size()
            };
            let (gb, ub, db) = (ebytes(gi), ebytes(ui), ebytes(di));
            let base =
                |info: &gguf_file::TensorInfo| (ct.tensor_data_offset + info.offset) as usize;
            let (gbase, ubase, dbase) = (base(gi), base(ui), base(di));
            let refs = (0..n_expert)
                .map(|j| MmapExpertRef {
                    gate_offset: gbase + j * gb,
                    gate_len: gb,
                    up_offset: ubase + j * ub,
                    up_len: ub,
                    down_offset: dbase + j * db,
                    down_len: db,
                    gate_shape: gi.shape.dims()[1..].to_vec(),
                    up_shape: ui.shape.dims()[1..].to_vec(),
                    down_shape: di.shape.dims()[1..].to_vec(),
                    gate_dtype: gi.ggml_dtype,
                    up_dtype: ui.ggml_dtype,
                    down_dtype: di.ggml_dtype,
                })
                .collect();
            all_host_refs.push(refs);
        }

        // Size the VRAM hot pool: budget = min(free-headroom, all-experts), slots = budget/max.
        // A slot holds the REPACKED expert (the MXFP4_KO twin the pool actually stores), not the
        // native GGUF source — sizing off `*_len` (native) would under-count the slot and let the
        // pool overshoot free VRAM. Ask `repacked_size_bytes` for the KO twin per projection.
        let ko_bytes = |shape: &[usize], dt: GgmlDType| -> usize {
            let kod = dt.to_ko(int8mode).unwrap_or(dt);
            candle::quantized::repacked_size_bytes(shape[0], shape[1], kod).unwrap_or(0)
        };
        let max_expert_size = all_host_refs
            .iter()
            .flatten()
            .map(|r| {
                ko_bytes(&r.gate_shape, r.gate_dtype)
                    + ko_bytes(&r.up_shape, r.up_dtype)
                    + ko_bytes(&r.down_shape, r.down_dtype)
            })
            .max()
            .unwrap_or(0);
        let total_experts = moe_layers.len() * n_expert;
        let total_expert_bytes = total_experts * max_expert_size;
        let (free_vram, _total_vram) = get_vram_info()?;
        // Reserve for everything that lives OUTSIDE the expert pool and is allocated after it:
        // the resident base weights (embedding/attention/lm_head/mHC/norms), the KV cache, and
        // per-forward activations. This is the razor's-edge knob between a device OOM (too small —
        // base overruns it) and a pinned-host OOM (too large — the shrunken VRAM pool pushes the
        // 152 GB of experts past the ~95 GB page-lock ceiling). ~152 GB experts vs VRAM+pinned is
        // near-saturated. 10 GiB proved too small — the resident base OOM'd with only 9.5 GB left;
        // 13 GiB gives base + working set room while keeping the VRAM pool large enough that the
        // pinned remainder stays under the page-lock ceiling. Cheap to retune now that load is fast.
        let headroom = 13usize << 30;
        let expert_budget = free_vram
            .saturating_sub(headroom)
            .max(free_vram / 2)
            .min(total_expert_bytes);
        let num_slots = if max_expert_size > 0 {
            (expert_budget / max_expert_size).min(total_experts)
        } else {
            0
        };
        let gb = |b: usize| b as f64 / (1usize << 30) as f64;
        let num_pinned_est = total_experts.saturating_sub(num_slots) + num_slots / 10;
        eprintln!(
            "[mem] free_vram={:.1}GB total_vram={:.1}GB | experts: n={} slot={:.1}MB total={:.1}GB \
             | budget={:.1}GB headroom={:.1}GB → vram_slots={} (~{:.1}GB), pinned~{} (~{:.1}GB)",
            gb(free_vram),
            gb(_total_vram),
            total_experts,
            max_expert_size as f64 / 1e6,
            gb(total_expert_bytes),
            gb(expert_budget),
            gb(headroom),
            num_slots,
            gb(num_slots * max_expert_size),
            num_pinned_est,
            gb(num_pinned_est * max_expert_size),
        );
        let free_before_experts = free_vram;
        let experts = Arc::new(ExpertCache::new(
            mmap.clone(),
            all_host_refs,
            num_slots,
            device,
            n_expert,
            Some(merged_path),
            None,
            int8mode,
        )?);
        let (free_after_experts, _) = get_vram_info()?;
        eprintln!(
            "[mem] after ExpertCache: free_vram={:.1}GB (expert VRAM pool consumed {:.1}GB); \
             resident_vram_gauge={:.1}GB",
            gb(free_after_experts),
            gb(free_before_experts.saturating_sub(free_after_experts)),
            gb(experts.resident_vram_bytes()),
        );

        // ── Resident non-expert weights ──
        // Token embedding stays in HOST RAM: it's a lookup table, read one row per token
        // (~5 KB), so gathering the row on the host and uploading just that row costs nothing
        // against a ~50-70 ms/token decode — and it frees ~1.3 GB of VRAM. See `step`.
        let embed = loader::dequant_f32(&mut gguf, "token_embd.weight", &Device::Cpu)?;
        let mut layers = Vec::with_capacity(moe_layers.len());
        // DeepSeek-V4-Flash is MoE on every layer, so the MoE index equals the layer index; the
        // enumerate index is the ExpertCache slot-group. (A dense leading block would need a
        // separate arm here — none exist in this model.)
        for (moe_layer_idx, &i) in moe_layers.iter().enumerate() {
            let p = format!("blk.{i}");
            let attn = loader::load_attention(&mut gguf, &cfg, i, device, int8mode)?;
            let hc_attn =
                loader::load_hc_params(&mut gguf, &format!("{p}.hc_attn"), device, int8mode)?;
            let hc_ffn =
                loader::load_hc_params(&mut gguf, &format!("{p}.hc_ffn"), device, int8mode)?;
            let attn_norm =
                loader::dequant_f32(&mut gguf, &format!("{p}.attn_norm.weight"), device)?;
            let ffn_norm = loader::dequant_f32(&mut gguf, &format!("{p}.ffn_norm.weight"), device)?;

            let gate_w = loader::qlinear_int8(
                &mut gguf,
                &format!("{p}.ffn_gate_inp.weight"),
                device,
                int8mode,
            )?;
            let (bias, tid2eid) = if cfg.is_hash_layer(i) {
                let t =
                    gguf.read_int_tensor_u32(&format!("{p}.ffn_gate_tid2eid.weight"), device)?;
                (None, Some(t))
            } else {
                let b = loader::dequant_f32(&mut gguf, &format!("{p}.exp_probs_b.bias"), device)?;
                (Some(b), None)
            };
            let gate = Gate::new(
                gate_w,
                bias,
                tid2eid,
                cfg.n_activated_experts,
                n_expert,
                ScoreFunc::parse(&cfg.score_func),
                cfg.route_scale,
            );
            let shared = Expert::new(
                loader::qlinear_int8(
                    &mut gguf,
                    &format!("{p}.ffn_gate_shexp.weight"),
                    device,
                    int8mode,
                )?,
                loader::qlinear_int8(
                    &mut gguf,
                    &format!("{p}.ffn_down_shexp.weight"),
                    device,
                    int8mode,
                )?,
                loader::qlinear_int8(
                    &mut gguf,
                    &format!("{p}.ffn_up_shexp.weight"),
                    device,
                    int8mode,
                )?,
                cfg.swiglu_limit,
            );
            layers.push(EngineLayer {
                attn,
                hc_attn,
                hc_ffn,
                attn_norm,
                ffn_norm,
                gate,
                shared,
                moe_layer_idx,
            });
        }

        let hc = HyperConnection::new(cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps);
        let hc_head = HyperParams {
            fn_w: loader::qlinear_int8(&mut gguf, "output_hc_fn.weight", device, int8mode)?,
            base: loader::dequant_f32(&mut gguf, "output_hc_base.weight", device)?,
            scale: loader::dequant_f32(&mut gguf, "output_hc_scale.weight", device)?,
        };
        let output_norm = loader::dequant_f32(&mut gguf, "output_norm.weight", device)?;
        let lm_head = loader::qlinear_int8(&mut gguf, "output.weight", device, int8mode)?;
        let (free_after_resident, _) = get_vram_info()?;
        eprintln!(
            "[mem] after resident base: free_vram={:.1}GB (resident consumed {:.1}GB); \
             ready for KV + activations",
            gb(free_after_resident),
            gb(free_after_experts.saturating_sub(free_after_resident)),
        );

        let max_seq = 512;
        let rope_compress = RotaryCache::new(
            cfg.rope_head_dim,
            max_seq,
            cfg.compress_rope_theta,
            cfg.original_seq_len,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
            device,
        )?;
        let rope_swa = RotaryCache::new(
            cfg.rope_head_dim,
            max_seq,
            cfg.rope_theta,
            0,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
            device,
        )?;

        Ok(Self {
            cfg,
            embed,
            layers,
            hc,
            hc_head,
            output_norm,
            lm_head,
            rope_compress,
            rope_swa,
            experts,
            _mmap: mmap,
            _reg: reg,
            device: device.clone(),
            int8mode,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// RoPE table for `layer`: long-context (compressed layers) vs base (sliding-window).
    fn rope_for(&self, layer: usize) -> &RotaryCache {
        if self.cfg.layer_kind(layer).compresses() {
            &self.rope_compress
        } else {
            &self.rope_swa
        }
    }

    /// The MoE sub-block for `layer` over the mHC block input `x` `[1, 1, dim]`: route
    /// (`sqrtsoftplus`/`noaux`/hash), run the routed experts through the resident `ExpertCache`
    /// on the int8-KO path (`MoeInput::Q8`), add the always-on shared expert. Returns
    /// `[1, 1, dim]`. `token_id` drives the hash-layer `tid2eid` routing.
    fn moe_forward(&self, layer: &EngineLayer, x: &Tensor, token_id: u32) -> Result<Tensor> {
        let dim = self.cfg.dim;
        let x2 = x.reshape((1, dim))?; // [nt=1, dim]
                                       // Float-normalized input for routing + the shared expert.
        let normed = rms_norm(&x2, &layer.ffn_norm, self.cfg.norm_eps)?;
        let ids = Tensor::from_vec(vec![token_id], 1, &self.device)?;
        let (weights, indices) = layer.gate.route(&normed, &ids)?; // [1,k], [1,k] u32

        // q8a128 activation for the int8-KO grouped expert GEMM (fused RMSNorm→quant, same
        // normalization as `normed` — quantization noise only, within the QAT tolerance).
        let cuda_dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => candle::bail!("Dsv4Engine::moe_forward requires a CUDA device"),
        };
        let q8 = candle::quantized::cuda::rms_norm_q8a128(
            &x2,
            &layer.ffn_norm,
            self.cfg.norm_eps as f32,
            &cuda_dev,
        )?;

        // Counting-sort the (token, expert) assignments by ascending expert id (O(A+E)),
        // matching the grouped-GEMM dispatch contract (see `SparseMoeBlock::forward_with_indices`).
        let idx_cpu: Vec<Vec<u32>> = indices.to_vec2::<u32>()?;
        let weights_flat = weights.flatten_all()?; // [nt*k]
        let (k, ne) = (self.cfg.n_activated_experts, self.cfg.n_routed_experts);
        let mut counts = vec![0u32; ne];
        for row in &idx_cpu {
            for &eid in row {
                if (eid as usize) < ne {
                    counts[eid as usize] += 1;
                }
            }
        }
        let mut cursor = vec![0u32; ne];
        let mut expert_ids: Vec<usize> = Vec::new();
        let mut running = 0u32;
        for (e, &c) in counts.iter().enumerate() {
            cursor[e] = running;
            running += c;
            if c > 0 {
                expert_ids.push(e);
            }
        }
        let mut assignments: Vec<(u32, u32, u32)> = vec![(0, 0, 0); running as usize];
        for (tok, row) in idx_cpu.iter().enumerate() {
            for (slot_k, &eid) in row.iter().enumerate() {
                if (eid as usize) >= ne {
                    continue;
                }
                let pos = cursor[eid as usize] as usize;
                assignments[pos] = (eid, tok as u32, tok as u32 * k as u32 + slot_k as u32);
                cursor[eid as usize] += 1;
            }
        }

        let routed = self.experts.submit_moe_work(
            layer.moe_layer_idx,
            expert_ids,
            MoeInput::Q8(q8),
            DType::F32,
            &weights_flat,
            assignments,
        )?; // [nt, dim] F32
        let shared = layer.shared.forward(&normed)?; // [nt, dim] F32
        (routed + shared)?.reshape((1, 1, dim))
    }

    /// Open a decode session (per-layer streaming attention KV) over this resident model.
    pub fn session(&self) -> Result<EngineSession<'_>> {
        let attn = self
            .layers
            .iter()
            .map(|l| l.attn.decoder())
            .collect::<Result<Vec<_>>>()?;
        Ok(EngineSession { engine: self, attn })
    }

    /// Greedy-decode `max_new` tokens after `prompt` — the engine analogue of the streaming
    /// reference's `generate`, used to validate identical output (e.g. "Paris") at engine speed.
    pub fn generate(&self, prompt: &[u32], max_new: usize) -> Result<Vec<u32>> {
        let mut sess = self.session()?;
        let mut logits: Option<Tensor> = None;
        for &t in prompt {
            logits = Some(sess.step(t)?);
        }
        let mut logits = logits.ok_or_else(|| candle::Error::msg("empty prompt"))?;
        let mut out = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            let next = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
            out.push(next);
            logits = sess.step(next)?;
        }
        Ok(out)
    }
}

/// A decode session: one streaming [`IncrementalAttention`] per layer (the session-owned KV) plus
/// the running position. Feeds tokens one at a time through the resident model.
pub struct EngineSession<'a> {
    engine: &'a Dsv4Engine,
    attn: Vec<super::attention::IncrementalAttention<'a>>,
}

impl EngineSession<'_> {
    /// One decode step: `token_id` → logits `[vocab]` at this position. mHC-wrapped incremental
    /// attention + resident-`ExpertCache` MoE, all in the validated math — same result as the
    /// streaming reference row, with resident (not per-forward-reloaded) experts.
    pub fn step(&mut self, token_id: u32) -> Result<Tensor> {
        let e = self.engine;
        let dim = e.cfg.dim;
        // Embedding is host-resident: gather the row on the host, upload just that row (~5 KB).
        let idt = Tensor::from_vec(vec![token_id], 1, &Device::Cpu)?;
        let row = e
            .embed
            .index_select(&idt, 0)?
            .reshape((1, 1, dim))?
            .to_dtype(DType::F32)?
            .to_device(&e.device)?;
        let mut h = e.hc.expand(&row)?; // [1,1,hc,dim]

        for (l, layer) in e.layers.iter().enumerate() {
            // Attention sub-block: mHC pre → norm → incremental attention → mHC post.
            let (x, post, comb) = e.hc.pre(&h, &layer.hc_attn)?;
            let x = rms_norm(&x, &layer.attn_norm, e.cfg.norm_eps)?;
            let x = self.attn[l].step(&x, e.rope_for(l))?;
            let h1 = e.hc.post(&x, &h, &post, &comb)?;

            // MoE sub-block: mHC pre → routed(ExpertCache)+shared → mHC post.
            let (x, post, comb) = e.hc.pre(&h1, &layer.hc_ffn)?;
            let moe = e.moe_forward(layer, &x, token_id)?;
            h = e.hc.post(&moe, &h1, &post, &comb)?;
        }

        let h = e.hc.head_reduce(&h, &e.hc_head)?;
        let h = rms_norm(&h, &e.output_norm, e.cfg.norm_eps)?;
        let logits = e.lm_head.forward(&h)?; // [1,1,vocab]
        logits.reshape((e.cfg.vocab_size,))
    }
}

/// RMSNorm with a learned weight, in F32 — matches the reference (`transformer.rs::rms_norm`).
fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = x.broadcast_div(&(ms + eps)?.sqrt()?)?;
    normed.broadcast_mul(&w.to_dtype(DType::F32)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn merged() -> std::path::PathBuf {
        std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            // Pre-repacked MXFP4_KO file (offline `prepare_ko_gguf`): experts are already the
            // lane-major KO twin, so staging skips the runtime repack entirely (fast load).
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf")
    }

    /// The resident-expert engine greedy-generates the SAME "Paris" answer as the block-streaming
    /// reference, but with experts resident (int8-KO `ExpertCache`) instead of re-read from disk
    /// every forward — the speedup. Prints load time + decode tok/s. Ignored (needs the merged
    /// file + CUDA + ~220 GB Q6_KO expert footprint across VRAM + pinned RAM).
    #[test]
    #[ignore]
    fn engine_generate_paris_fast() -> Result<()> {
        let path = merged();
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let t0 = std::time::Instant::now();
        // MXFP4 experts repack to the exponent-collapse int8 twin MXFP4_KO: an exact 4-bit
        // byte-reorder (no F32 requant), so the 147 GB stays ~156 GB and fits VRAM(72)+RAM(189)
        // where the old Q6_KO/Q8_KO requant blew past RAM. The int8 grouped GEMM reads it directly.
        let engine = Dsv4Engine::load(&path, &device, Int8Mode::Performance)?;
        eprintln!("[engine] load {:.1}s", t0.elapsed().as_secs_f32());

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        let prompt = "<｜begin▁of▁sentence｜><｜User｜>What is the capital of France? \
             Reply with only the city name.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();

        let t1 = std::time::Instant::now();
        let gen = engine.generate(&ids, 12)?;
        let dt = t1.elapsed().as_secs_f32();
        let text = tokenizer
            .decode(&gen, false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        eprintln!("[engine] generated ids={gen:?}");
        eprintln!("[engine] continuation={text:?}");
        eprintln!(
            "[engine] {} prompt + 12 gen in {:.1}s = {:.2} tok/s decode",
            ids.len(),
            dt,
            12.0 / dt
        );
        assert!(
            text.contains("Paris"),
            "engine did not answer Paris: {text:?}"
        );
        Ok(())
    }
}
