//! GGUF loading for DeepSeek-V4-Flash (llama.cpp `deepseek4` architecture).
//!
//! The model ships as a multi-file split GGUF: global metadata + a subset of tensors in
//! each file. This module opens every split, merges the tensor directories, and reads
//! `deepseek4.*` metadata into a [`Config`].

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use candle::quantized::gguf_file::{Content, TensorInfo, Value};
use candle::quantized::{GgmlDType, Int8Mode, QMatMul, QTensor};
use candle::{DType, Device, Result, Tensor};

use super::attention::{Attention, AttentionParams};
use super::compressor::Compressor;
use super::config::Config;
use super::indexer::Indexer;
use super::linear::QLinear;

/// A merged view over the split GGUF files: one [`Content`] + open file per split, with a
/// name→split index for tensor lookups.
pub struct GgufModel {
    contents: Vec<Content>,
    files: Vec<File>,
    paths: Vec<PathBuf>,
    tensor_split: HashMap<String, usize>,
    pub metadata: HashMap<String, Value>,
}

impl GgufModel {
    /// Open all split files (in order). Global metadata is taken from the first split.
    pub fn open(paths: &[PathBuf]) -> Result<Self> {
        let mut contents = Vec::new();
        let mut files = Vec::new();
        let mut tensor_split = HashMap::new();
        let mut metadata = HashMap::new();
        for (i, p) in paths.iter().enumerate() {
            let mut f = File::open(p)?;
            let content = Content::read(&mut f)?;
            if i == 0 {
                metadata = content.metadata.clone();
            }
            for name in content.tensor_infos.keys() {
                tensor_split.insert(name.clone(), i);
            }
            contents.push(content);
            files.push(f);
        }
        Ok(Self {
            contents,
            files,
            paths: paths.to_vec(),
            tensor_split,
            metadata,
        })
    }

    /// Discover the ordered split paths from any one member, using the
    /// `NAME-00001-of-000NN.gguf` convention.
    pub fn discover_splits(one: &Path) -> Result<Vec<PathBuf>> {
        let name = one
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| candle::Error::msg("bad gguf path"))?;
        let dir = one.parent().unwrap_or_else(|| Path::new("."));
        // Match "...-<idx>-of-<count>.gguf".
        if let Some(pos) = name.find("-of-") {
            let count: usize = name[pos + 4..]
                .trim_end_matches(".gguf")
                .parse()
                .map_err(|_| candle::Error::msg("bad split count"))?;
            let prefix_end = name[..pos].rfind('-').unwrap_or(0);
            let prefix = &name[..prefix_end + 1];
            let mut out = Vec::new();
            for i in 1..=count {
                out.push(dir.join(format!("{prefix}{i:05}-of-{count:05}.gguf")));
            }
            return Ok(out);
        }
        Ok(vec![one.to_path_buf()])
    }

    pub fn tensor_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.tensor_split.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        let &i = self.tensor_split.get(name)?;
        self.contents[i].tensor_infos.get(name)
    }

    /// Read a tensor as a `QTensor` from whichever split holds it.
    pub fn qtensor(&mut self, name: &str, device: &Device) -> Result<candle::quantized::QTensor> {
        let &i = self
            .tensor_split
            .get(name)
            .ok_or_else(|| candle::Error::msg(format!("missing tensor {name}")))?;
        self.contents[i].tensor(&mut self.files[i], name, device)
    }

    /// Read an integer tensor (e.g. the I32 hash-routing `tid2eid`) directly from its
    /// source split as a `U32` [`Tensor`]. The generic GGUF tensor loader has no path for
    /// raw integer dtypes, so we read the bytes and reinterpret.
    pub fn read_int_tensor_u32(&mut self, name: &str, device: &Device) -> Result<Tensor> {
        use std::io::{Read, Seek, SeekFrom};
        let &i = self
            .tensor_split
            .get(name)
            .ok_or_else(|| candle::Error::msg(format!("missing tensor {name}")))?;
        let info = self.contents[i]
            .tensor_infos
            .get(name)
            .ok_or_else(|| candle::Error::msg(format!("missing info {name}")))?;
        let dims = info.shape.dims().to_vec();
        let elems: usize = dims.iter().product();
        let type_size = info.ggml_dtype.type_size(); // 4 for I32
        let off = self.contents[i].tensor_data_offset + info.offset;
        let mut buf = vec![0u8; elems * type_size];
        self.files[i].seek(SeekFrom::Start(off))?;
        self.files[i].read_exact(&mut buf)?;
        let vals: Vec<u32> = match type_size {
            4 => buf
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
                .collect(),
            8 => buf
                .chunks_exact(8)
                .map(|c| {
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as u32
                })
                .collect(),
            other => candle::bail!("read_int_tensor_u32: unexpected type size {other}"),
        };
        Tensor::from_vec(vals, dims, device)
    }

    pub fn metadata_u32(&self, key: &str) -> Result<u32> {
        self.metadata
            .get(key)
            .ok_or_else(|| candle::Error::msg(format!("missing metadata {key}")))?
            .to_u32()
    }

    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }
}

/// Dequantize a tensor to an F32 `Tensor` (used for norms, sinks, ape, and the
/// compressor/indexer projections that `model.py` keeps in float).
pub fn dequant_f32(m: &mut GgufModel, name: &str, device: &Device) -> Result<Tensor> {
    m.qtensor(name, device)?.dequantize(device)
}

/// Load a float weight at its **native** width (BF16/F16 stay half-size; F32 stays F32) rather
/// than always inflating to F32 — halves VRAM for the BF16-on-disk matmul weights (lm_head,
/// compressor/indexer/router/mHC projections), losslessly. Callers convert to F32 at use anyway
/// (the forwards `.to_dtype(F32)` the weight), so this is a pure storage win. The dequant to F32
/// is transient (freed after the down-cast); only the native-width tensor is retained.
pub fn dequant_native(m: &mut GgufModel, name: &str, device: &Device) -> Result<Tensor> {
    use candle::quantized::GgmlDType;
    let dt = m.info(name).map(|i| i.ggml_dtype);
    let t = m.qtensor(name, device)?.dequantize(device)?;
    match dt {
        Some(GgmlDType::BF16) => t.to_dtype(DType::BF16),
        Some(GgmlDType::F16) => t.to_dtype(DType::F16),
        _ => Ok(t),
    }
}

/// Load a linear weight: quantized formats (Q8_0, MXFP4, …) as a `QMatMul` (stays quantized);
/// float formats as a dense `Tensor` at **native** width (BF16 stays BF16 — not inflated to F32).
pub fn qlinear(m: &mut GgufModel, name: &str, device: &Device) -> Result<QLinear> {
    let is_float = matches!(
        m.info(name).map(|i| i.ggml_dtype),
        Some(candle::quantized::GgmlDType::F32)
            | Some(candle::quantized::GgmlDType::F16)
            | Some(candle::quantized::GgmlDType::BF16)
    );
    if is_float {
        Ok(QLinear::from_weight(dequant_native(m, name, device)?))
    } else {
        let qt = m.qtensor(name, device)?;
        Ok(QLinear::from_qmatmul(QMatMul::from_qtensor(qt)?))
    }
}

/// Load a **quantized** linear weight for the int8 tensor-core path: repack its KO twin once at
/// load (`repack_for_optimization`) so the forward runs q8a128×KO int8 (no F32 dequant). Float
/// weights (no KO twin) fall back to native dense; a weight whose shape can't tile to KO
/// (`nrows%8`/`ncols%128`) falls back to the standard `Quant` path. `mode` picks the KO twin.
pub fn qlinear_int8(
    m: &mut GgufModel,
    name: &str,
    device: &Device,
    mode: Int8Mode,
) -> Result<QLinear> {
    // Off → the standard path (F32 reference / non-int8 callers): dense float or dequant Quant.
    if !mode.is_int8() {
        return qlinear(m, name, device);
    }
    // Already a KO twin (offline-prepared by `prepare_ko_gguf`) → wrap straight for the int8
    // path, no at-load repack (no F32 transient — the whole reason we moved it offline).
    if matches!(m.info(name).map(|i| i.ggml_dtype), Some(d) if d.is_ko()) {
        let qt = m.qtensor(name, device)?;
        return Ok(QLinear::from_int8(QMatMul::from_qtensor(qt)?));
    }
    let is_float = matches!(
        m.info(name).map(|i| i.ggml_dtype),
        Some(candle::quantized::GgmlDType::F32)
            | Some(candle::quantized::GgmlDType::F16)
            | Some(candle::quantized::GgmlDType::BF16)
    );
    if is_float {
        // int8 mode + a float matmul weight (BF16 lm_head / compressor / indexer / router / mHC
        // fn): requant to the KO twin (dequant → Q8_0 → KO). Bounded transient — the big weights
        // are already offline-KO, so only these smaller ones hit the at-load path. Falls back to
        // native-width dense if the shape isn't KO-tileable (odd dim).
        let f32 = dequant_f32(m, name, device)?;
        let ko = candle::quantized::QTensor::quantize(&f32, candle::quantized::GgmlDType::Q8_0)
            .and_then(QMatMul::from_qtensor)
            .and_then(|q| q.repack_for_optimization(mode));
        return match ko {
            Ok(ko) => Ok(QLinear::from_int8(ko)),
            Err(_) => Ok(QLinear::from_weight(dequant_native(m, name, device)?)),
        };
    }
    let qmm = QMatMul::from_qtensor(m.qtensor(name, device)?)?;
    match qmm.repack_for_optimization(mode) {
        Ok(ko) => Ok(QLinear::from_int8(ko)),
        Err(_) => Ok(QLinear::from_qmatmul(qmm)), // shape not KO-tileable → dequant path
    }
}

/// Load a `Compressor` from the `{prefix}_kv/_gate/_ape/_norm.weight` tensors.
#[allow(clippy::too_many_arguments)]
fn load_compressor(
    m: &mut GgufModel,
    prefix: &str,
    ratio: usize,
    head_dim: usize,
    rope_head_dim: usize,
    eps: f64,
    device: &Device,
    mode: Int8Mode,
) -> Result<Compressor> {
    // wkv/wgate are matmul projections → int8-KO. `ape` is an additive positional bias (not a
    // matmul weight) and stays dense.
    let wkv = qlinear_int8(m, &format!("{prefix}_kv.weight"), device, mode)?;
    let wgate = qlinear_int8(m, &format!("{prefix}_gate.weight"), device, mode)?;
    // ape/norm stored F32 so the compressor's per-call `to_dtype(F32)` on these
    // constants is a proven no-op (no in-loop copy). Widening ape via the same
    // native→F32 path the per-call cast used keeps it bit-identical.
    let ape = dequant_native(m, &format!("{prefix}_ape.weight"), device)?.to_dtype(DType::F32)?;
    let norm = dequant_f32(m, &format!("{prefix}_norm.weight"), device)?;
    Ok(Compressor::new(
        wkv,
        wgate,
        ape,
        norm,
        ratio,
        head_dim,
        rope_head_dim,
        eps,
    ))
}

/// Split the single `attn_output_a` weight `[ng·olr, per_group]` (Q8_0 on disk) into `ng` per-group
/// int8-KO linears `[olr, per_group]` for the per-group output projection `o_g[g] @ wo_a[g]ᵀ`. The
/// KO tiling of one `[ng·olr, per_group]` weight wouldn't match the per-group matmul, so each group
/// is requantized to its own Q8_KO twin (float dense under `Int8Mode::Off`). Dequantized once; the
/// F32 materialization is a per-layer transient (~one weight's worth), freed after requant.
fn load_wo_a_groups(
    m: &mut GgufModel,
    name: &str,
    ng: usize,
    device: &Device,
    mode: Int8Mode,
) -> Result<Vec<QLinear>> {
    let full = m.qtensor(name, device)?.dequantize(device)?; // [ng·olr, per_group] f32
    let rows = full.dim(0)?;
    let olr = rows / ng;
    let mut out = Vec::with_capacity(ng);
    for g in 0..ng {
        let slice = full.narrow(0, g * olr, olr)?.contiguous()?; // [olr, per_group]
        if !mode.is_int8() {
            out.push(QLinear::from_weight(slice));
            continue;
        }
        let q = QMatMul::from_qtensor(QTensor::quantize(&slice, GgmlDType::Q8_0)?)?;
        out.push(match q.repack_for_optimization(mode) {
            Ok(ko) => QLinear::from_int8(ko),
            Err(_) => QLinear::from_weight(slice), // not KO-tileable → dense
        });
    }
    Ok(out)
}

/// Load the full attention module for `layer` from the real GGUF tensors.
pub fn load_attention(
    m: &mut GgufModel,
    cfg: &Config,
    layer: usize,
    device: &Device,
    mode: Int8Mode,
) -> Result<Attention> {
    let b = format!("blk.{layer}.");
    let kind = cfg.layer_kind(layer);
    let ratio = cfg.compress_ratio(layer);

    let (compressor, indexer) = if kind.compresses() {
        let comp = load_compressor(
            m,
            &format!("{b}attn_compressor"),
            ratio,
            cfg.head_dim,
            cfg.rope_head_dim,
            cfg.norm_eps,
            device,
            mode,
        )?;
        let indexer = if kind.has_indexer() {
            let icomp = load_compressor(
                m,
                &format!("{b}indexer_compressor"),
                ratio,
                cfg.index_head_dim,
                cfg.rope_head_dim,
                cfg.norm_eps,
                device,
                mode,
            )?;
            Some(Indexer::new(
                qlinear_int8(m, &format!("{b}indexer.attn_q_b.weight"), device, mode)?,
                qlinear_int8(m, &format!("{b}indexer.proj.weight"), device, mode)?,
                icomp,
                cfg.index_n_heads,
                cfg.index_head_dim,
                cfg.rope_head_dim,
                cfg.index_topk,
            ))
        } else {
            None
        };
        (Some(comp), indexer)
    } else {
        (None, None)
    };

    let p = AttentionParams {
        wq_a: qlinear_int8(m, &format!("{b}attn_q_a.weight"), device, mode)?,
        q_norm: dequant_f32(m, &format!("{b}attn_q_a_norm.weight"), device)?,
        wq_b: qlinear_int8(m, &format!("{b}attn_q_b.weight"), device, mode)?,
        wkv: qlinear_int8(m, &format!("{b}attn_kv.weight"), device, mode)?,
        kv_norm: dequant_f32(m, &format!("{b}attn_kv_a_norm.weight"), device)?,
        // Keep wo_a quantized (Q8_0) in VRAM; `output_proj` dequantizes it per layer on use.
        wo_a: load_wo_a_groups(
            m,
            &format!("{b}attn_output_a.weight"),
            cfg.o_groups,
            device,
            mode,
        )?,
        wo_b: qlinear_int8(m, &format!("{b}attn_output_b.weight"), device, mode)?,
        attn_sink: dequant_f32(m, &format!("{b}attn_sinks.weight"), device)?,
        compressor,
        indexer,
    };
    Attention::new(cfg, layer, p)
}

/// Slice a 3D expert tensor `[n_experts, out, inn]` into per-expert `QMatMul`s. Each
/// expert's raw block bytes are read from the (CPU-loaded) 3D tensor and rebuilt as an
/// owned device `QTensor` — cheap for MXFP4 (4-bit) and reused by the streaming path.
fn load_experts_3d(
    m: &mut GgufModel,
    name: &str,
    n_experts: usize,
    out: usize,
    inn: usize,
    device: &Device,
) -> Result<Vec<QMatMul>> {
    let dtype = m
        .info(name)
        .map(|i| i.ggml_dtype)
        .ok_or_else(|| candle::Error::msg(format!("missing {name}")))?;
    let q3d = m.qtensor(name, &Device::Cpu)?;
    let per_bytes = out * inn / dtype.block_size() * dtype.type_size();
    let mut experts = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let bytes = q3d.data_range(e * per_bytes..(e + 1) * per_bytes)?;
        let qt =
            candle::quantized::ggml_file::qtensor_from_ggml(dtype, &bytes, vec![out, inn], device)?;
        experts.push(QMatMul::from_qtensor(qt)?);
    }
    Ok(experts)
}

/// Load the full MoE block for `layer` from the real GGUF tensors.
pub fn load_moe(
    m: &mut GgufModel,
    cfg: &Config,
    layer: usize,
    device: &Device,
) -> Result<super::moe::MoE> {
    use super::moe::{Expert, Gate, MoE, ScoreFunc};
    let b = format!("blk.{layer}.");
    let (dim, inter, ne) = (cfg.dim, cfg.moe_inter_dim, cfg.n_routed_experts);

    let gate_w = dequant_native(m, &format!("{b}ffn_gate_inp.weight"), device)?;
    let (bias, tid2eid) = if cfg.is_hash_layer(layer) {
        let t = m.read_int_tensor_u32(&format!("{b}ffn_gate_tid2eid.weight"), device)?;
        (None, Some(t))
    } else {
        (
            Some(dequant_f32(m, &format!("{b}exp_probs_b.bias"), device)?),
            None,
        )
    };
    let gate = Gate::new(
        gate_w,
        bias,
        tid2eid,
        cfg.n_activated_experts,
        ne,
        ScoreFunc::parse(&cfg.score_func),
        cfg.route_scale,
    );

    let gate_e = load_experts_3d(
        m,
        &format!("{b}ffn_gate_exps.weight"),
        ne,
        inter,
        dim,
        device,
    )?;
    let up_e = load_experts_3d(m, &format!("{b}ffn_up_exps.weight"), ne, inter, dim, device)?;
    let down_e = load_experts_3d(
        m,
        &format!("{b}ffn_down_exps.weight"),
        ne,
        dim,
        inter,
        device,
    )?;
    let experts: Vec<Expert> = gate_e
        .into_iter()
        .zip(up_e)
        .zip(down_e)
        .map(|((g, u), d)| {
            Expert::new(
                QLinear::from_qmatmul(g),
                QLinear::from_qmatmul(d),
                QLinear::from_qmatmul(u),
                cfg.swiglu_limit,
            )
        })
        .collect();

    let shared = Expert::new(
        qlinear(m, &format!("{b}ffn_gate_shexp.weight"), device)?,
        qlinear(m, &format!("{b}ffn_down_shexp.weight"), device)?,
        qlinear(m, &format!("{b}ffn_up_shexp.weight"), device)?,
        cfg.swiglu_limit,
    );
    Ok(MoE::new(gate, experts, shared, dim))
}

/// Load the `{prefix}_fn/_base/_scale.weight` hyper-connection parameters.
pub fn load_hc_params(
    m: &mut GgufModel,
    prefix: &str,
    device: &Device,
    mode: Int8Mode,
) -> Result<super::hyper::HyperParams> {
    Ok(super::hyper::HyperParams {
        // `fn_w` is a mixing matmul → int8-KO (falls back to dense when `hc_mult < 8` makes the
        // head variant non-KO-tileable). `base`/`scale` are small per-copy vectors, stay dense.
        fn_w: qlinear_int8(m, &format!("{prefix}_fn.weight"), device, mode)?,
        base: dequant_f32(m, &format!("{prefix}_base.weight"), device)?,
        scale: dequant_f32(m, &format!("{prefix}_scale.weight"), device)?,
    })
}

/// Load a full transformer block (mHC-wrapped attention + MoE) for `layer`.
pub fn load_block(
    m: &mut GgufModel,
    cfg: &Config,
    layer: usize,
    device: &Device,
) -> Result<super::transformer::Block> {
    use super::hyper::HyperConnection;
    let b = format!("blk.{layer}.");
    let hc = HyperConnection::new(cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps);
    let hc_attn = load_hc_params(m, &format!("{b}hc_attn"), device, Int8Mode::Off)?;
    let hc_ffn = load_hc_params(m, &format!("{b}hc_ffn"), device, Int8Mode::Off)?;
    let attn_norm = dequant_f32(m, &format!("{b}attn_norm.weight"), device)?;
    let ffn_norm = dequant_f32(m, &format!("{b}ffn_norm.weight"), device)?;
    let attn = load_attention(m, cfg, layer, device, Int8Mode::Off)?;
    let moe = load_moe(m, cfg, layer, device)?;
    Ok(super::transformer::Block::new(
        hc,
        hc_attn,
        hc_ffn,
        attn_norm,
        ffn_norm,
        attn,
        moe,
        cfg.norm_eps,
    ))
}

/// Load a DSpark drafter block: the mHC-wrapped attention sub-block + its norms, but **no** eager
/// MoE. The drafter's routed experts are far too large to keep VRAM-resident, so its MoE lives in
/// the shared [`super::dspark_experts::DsparkStreamingMoe`] (host RAM + a small hot slot set) and is
/// spliced into the FFN sub-block at forward time. This reads exactly the block's attention/norm/hc
/// tensors — the `ffn_gate_inp`/`ffn_*_exps`/`ffn_*_shexp` weights are read by the streaming MoE.
pub fn load_dspark_block(
    m: &mut GgufModel,
    cfg: &Config,
    layer: usize,
    device: &Device,
) -> Result<super::dspark::DsparkBlock> {
    use super::hyper::HyperConnection;
    let b = format!("blk.{layer}.");
    let hc = HyperConnection::new(cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps);
    // Load the backbone attention + mHC mixes as int8 (the target engine's expert/perf mode), not
    // full precision: the drafter is lossless (every draft is verified against the target), so int8
    // drafter weights only affect acceptance, never output — and they halve the drafter's VRAM
    // footprint, which matters on a box where the target already saturates the pinned pool.
    let hc_attn = load_hc_params(m, &format!("{b}hc_attn"), device, Int8Mode::Performance)?;
    let hc_ffn = load_hc_params(m, &format!("{b}hc_ffn"), device, Int8Mode::Performance)?;
    let attn_norm = dequant_f32(m, &format!("{b}attn_norm.weight"), device)?;
    let ffn_norm = dequant_f32(m, &format!("{b}ffn_norm.weight"), device)?;
    let attn = load_attention(m, cfg, layer, device, Int8Mode::Performance)?;
    Ok(super::dspark::DsparkBlock::new(
        hc,
        hc_attn,
        hc_ffn,
        attn_norm,
        ffn_norm,
        attn,
        cfg.norm_eps,
    ))
}

/// Read the `deepseek4.*` metadata into a [`Config`], falling back to the model defaults
/// for anything the file omits.
pub fn config_from_gguf(m: &GgufModel) -> Result<Config> {
    let arch = m
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .cloned()
        .unwrap_or_else(|| "deepseek4".to_string());
    let p = format!("{arch}.");
    let g = |k: &str| m.metadata.get(&format!("{p}{k}"));

    // Metadata keys verified against the real bartowski GGUF. YaRN parameters
    // (factor/beta/original context) are NOT stored by llama.cpp's deepseek4 arch — it
    // bakes them in — so they come from the model-known defaults here.
    let cfg = Config {
        vocab_size: uget(g("vocab_size"), 129280),
        dim: uget(g("embedding_length"), 4096),
        moe_inter_dim: uget(g("expert_feed_forward_length"), 2048),
        n_layers: uget(g("block_count"), 43),
        n_hash_layers: uget(g("hash_layer_count"), 3),
        n_heads: uget(g("attention.head_count"), 64),
        n_routed_experts: uget(g("expert_count"), 256),
        n_shared_experts: uget(g("expert_shared_count"), 1),
        n_activated_experts: uget(g("expert_used_count"), 6),
        score_func: "sqrtsoftplus".to_string(),
        route_scale: fget(g("expert_weights_scale"), 1.5),
        swiglu_limit: array_first_f32(g("swiglu_clamp_exp"), 10.0),
        q_lora_rank: uget(g("attention.q_lora_rank"), 1024),
        head_dim: uget(g("attention.key_length"), 512),
        rope_head_dim: uget(g("rope.dimension_count"), 64),
        norm_eps: fget(g("attention.layer_norm_rms_epsilon"), 1e-6),
        o_groups: uget(g("attention.output_group_count"), 8),
        o_lora_rank: uget(g("attention.output_lora_rank"), 1024),
        window_size: uget(g("attention.sliding_window"), 128),
        compress_ratios: compress_ratios(m, &p),
        compress_rope_theta: fget(g("attention.compress_rope_freq_base"), 160000.0),
        original_seq_len: 65536,
        rope_theta: fget(g("rope.freq_base"), 10000.0),
        rope_factor: 16.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        index_n_heads: uget(g("attention.indexer.head_count"), 64),
        index_head_dim: uget(g("attention.indexer.key_length"), 128),
        index_topk: uget(g("attention.indexer.top_k"), 512),
        hc_mult: uget(g("hyper_connection.count"), 4),
        hc_sinkhorn_iters: uget(g("hyper_connection.sinkhorn_iterations"), 20),
        hc_eps: fget(g("hyper_connection.epsilon"), 1e-6),
    };
    Ok(cfg)
}

fn uget(v: Option<&Value>, default: usize) -> usize {
    v.and_then(|v| v.to_u32().ok())
        .map(|x| x as usize)
        .or_else(|| v.and_then(|v| v.to_i32().ok()).map(|x| x as usize))
        .unwrap_or(default)
}

fn fget(v: Option<&Value>, default: f64) -> f64 {
    v.and_then(|v| v.to_f32().ok())
        .map(|x| x as f64)
        .unwrap_or(default)
}

/// First element of an F32 array metadata value (the per-layer `swiglu_clamp_exp` is a
/// 43-long array of identical limits); falls back to `default`.
fn array_first_f32(v: Option<&Value>, default: f64) -> f64 {
    v.and_then(|v| v.to_vec().ok())
        .and_then(|a| a.first())
        .and_then(|x| x.to_f32().ok())
        .map(|x| x as f64)
        .unwrap_or(default)
}

/// `deepseek4.attention.compress_ratios` — a per-layer i32 array. Falls back to the known
/// 43-layer schedule if the key is absent.
fn compress_ratios(m: &GgufModel, prefix: &str) -> Vec<usize> {
    if let Some(v) = m
        .metadata
        .get(&format!("{prefix}attention.compress_ratios"))
    {
        if let Ok(arr) = v.to_vec() {
            return arr
                .iter()
                .map(|x| {
                    x.to_u32()
                        .or_else(|_| x.to_i32().map(|i| i as u32))
                        .unwrap_or(0) as usize
                })
                .collect();
        }
    }
    // Known schedule: L0,L1 SWA; L2..=L42 alternate CSA(4)/HCA(128); tail SWA.
    let mut r = vec![0usize; 43];
    for (i, item) in r.iter_mut().enumerate().take(43).skip(2) {
        if i >= 40 {
            *item = 0;
        } else if i % 2 == 0 {
            *item = 4;
        } else {
            *item = 128;
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::DType;

    fn model_dir() -> PathBuf {
        PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
    }

    fn first_split() -> PathBuf {
        model_dir().join("DeepSeek-V4-Flash-0731-MXFP4-00001-of-00004.gguf")
    }

    /// Inspection harness: when the real GGUF is present, dump every metadata key and a
    /// categorized sample of tensor names + dtypes. Skips cleanly when the file is absent
    /// (so CI without the 156 GB model still passes). Run with `--nocapture`.
    #[test]
    fn dump_real_gguf() -> Result<()> {
        let first = first_split();
        if !first.exists() {
            eprintln!("[skip] real GGUF not present at {}", first.display());
            return Ok(());
        }
        let splits = GgufModel::discover_splits(&first)?;
        let present: Vec<PathBuf> = splits.into_iter().filter(|p| p.exists()).collect();
        eprintln!("[dump] {} splits present", present.len());
        let m = GgufModel::open(&present)?;

        eprintln!("=== METADATA ({}) ===", m.metadata.len());
        let mut keys: Vec<&String> = m.metadata.keys().collect();
        keys.sort();
        for k in keys {
            if k.starts_with("tokenizer.ggml.tokens")
                || k.starts_with("tokenizer.ggml.merges")
                || k.starts_with("tokenizer.ggml.token_type")
            {
                continue; // huge arrays
            }
            let v = &m.metadata[k];
            let vs = match v {
                Value::String(s) => format!("\"{s}\""),
                Value::U32(x) => x.to_string(),
                Value::I32(x) => x.to_string(),
                Value::F32(x) => x.to_string(),
                Value::Bool(x) => x.to_string(),
                Value::Array(a) => format!("[array len {}]", a.len()),
                other => format!("{other:?}"),
            };
            eprintln!("  {k} = {vs}");
        }

        // Tensors for block 0 + the globals + block 2 (a CSA layer) + block 3 (HCA).
        let names = m.tensor_names();
        eprintln!("=== TENSORS ({} total) ===", names.len());
        for n in &names {
            let is_global = !n.starts_with("blk.");
            let b0 = n.starts_with("blk.0.");
            let b2 = n.starts_with("blk.2.");
            let b3 = n.starts_with("blk.3.");
            if is_global || b0 || b2 || b3 {
                if let Some(info) = m.info(n) {
                    eprintln!("  {n}  {:?}  {:?}", info.ggml_dtype, info.shape.dims());
                }
            }
        }
        Ok(())
    }

    /// Merge the 4 bartowski splits into one standalone GGUF (§6.5) so the model loads
    /// via the engine's single-mmap path. Ignored (156 GB, minutes of I/O); run explicitly.
    #[test]
    #[ignore]
    fn merge_splits_to_single_file() -> Result<()> {
        let first = first_split();
        if !first.exists() {
            eprintln!("[skip] real GGUF not present");
            return Ok(());
        }
        let splits: Vec<PathBuf> = GgufModel::discover_splits(&first)?
            .into_iter()
            .filter(|p| p.exists())
            .collect();
        assert_eq!(splits.len(), 4, "expected 4 splits");
        let out = model_dir().join("DeepSeek-V4-Flash-0731-MXFP4-merged.gguf");
        eprintln!("[merge] {} splits -> {}", splits.len(), out.display());
        let cb = |i: usize, n: usize| {
            if i % 200 == 0 || i == n {
                eprintln!("[merge] tensor {i}/{n}");
            }
        };
        candle::quantized::gguf_file::merge_split_ggufs(&splits, &out, Some(&cb))?;
        // Sanity: the merged file opens and has the right tensor count + config.
        let m = GgufModel::open(std::slice::from_ref(&out))?;
        let cfg = config_from_gguf(&m)?;
        assert_eq!(cfg.n_layers, 43);
        assert_eq!(cfg.n_routed_experts, 256);
        assert!(m.info("blk.0.ffn_gate_exps.weight").is_some());
        assert!(m.info("output.weight").is_some());
        eprintln!(
            "[merge] OK — {} tensors in merged file",
            m.tensor_names().len()
        );
        Ok(())
    }

    /// Load a full real transformer block (mHC-wrapped attention + MoE) and run it on the
    /// GPU — the last real-weight composition check (hyper-connections + Sinkhorn over the
    /// 4-copy residual stream around real attention + MoE). Ignored (needs merged file).
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn real_block_runs() -> Result<()> {
        let merged = model_dir().join("DeepSeek-V4-Flash-0731-MXFP4-merged.gguf");
        if !merged.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let mut m = GgufModel::open(std::slice::from_ref(&merged))?;
        let cfg = config_from_gguf(&m)?;
        let device = Device::new_cuda(0)?;
        let layer = 3usize;
        let block = super::load_block(&mut m, &cfg, layer, &device)?;
        let (theta, orig) = cfg.rope_params(layer);
        let rope = super::super::rope::RotaryCache::new(
            cfg.rope_head_dim,
            theta,
            orig,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
            &device,
        )?;
        let hc = super::super::hyper::HyperConnection::new(
            cfg.hc_mult,
            cfg.hc_sinkhorn_iters,
            cfg.hc_eps,
        );
        let x = Tensor::randn(0f32, 1.0, (1, 8, cfg.dim), &device)?;
        let h = hc.expand(&x)?; // [1, 8, hc_mult, dim]
        let ids = Tensor::zeros((1, 8), DType::U32, &device)?;
        let out = block.forward(&h, &ids, &rope)?;
        assert_eq!(out.dims(), &[1, 8, cfg.hc_mult, cfg.dim]);
        let v = out.flatten_all()?.to_vec1::<f32>()?;
        assert!(v.iter().all(|x| x.is_finite()), "non-finite block output");
        eprintln!(
            "[ok] real full block {layer} (attn+MoE+mHC) ran: out {:?}",
            out.dims()
        );
        Ok(())
    }

    /// Load a real MoE layer (non-hash, `sqrtsoftplus`/`noaux_tc` router + 256 MXFP4
    /// experts + shared expert) from the merged GGUF and run it on the GPU. Validates the
    /// real router and MXFP4 expert compute on actual weights. Ignored (needs merged file).
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn real_moe_layer_runs() -> Result<()> {
        let merged = model_dir().join("DeepSeek-V4-Flash-0731-MXFP4-merged.gguf");
        if !merged.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let mut m = GgufModel::open(std::slice::from_ref(&merged))?;
        let cfg = config_from_gguf(&m)?;
        let device = Device::new_cuda(0)?;
        let layer = 3usize; // non-hash layer: exp_probs_b bias router
        let moe = super::load_moe(&mut m, &cfg, layer, &device)?;
        let x = Tensor::randn(0f32, 1.0, (1, 4, cfg.dim), &device)?;
        let ids = Tensor::zeros((1, 4), DType::U32, &device)?;
        let out = moe.forward(&x, &ids)?;
        assert_eq!(out.dims(), &[1, 4, cfg.dim]);
        let v = out.flatten_all()?.to_vec1::<f32>()?;
        assert!(v.iter().all(|x| x.is_finite()), "non-finite MoE output");
        let maxabs = v.iter().fold(0f32, |a, &x| a.max(x.abs()));
        eprintln!("[ok] real MoE layer {layer} (256 MXFP4 experts) ran: max|.|={maxabs:.3}");
        Ok(())
    }

    /// Load a real CSA attention layer from the merged GGUF and run it on the GPU — proves
    /// the loader + real Q8_0/BF16/F32 weights + the attention math work on actual weights.
    /// Ignored (needs the merged file + CUDA).
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn real_attention_layer_runs() -> Result<()> {
        let merged = model_dir().join("DeepSeek-V4-Flash-0731-MXFP4-merged.gguf");
        if !merged.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let mut m = GgufModel::open(std::slice::from_ref(&merged))?;
        let cfg = config_from_gguf(&m)?;
        let device = Device::new_cuda(0)?;
        let layer = 2usize; // CSA layer
        let att = super::load_attention(&mut m, &cfg, layer, &device, Int8Mode::Off)?;
        let (theta, orig) = cfg.rope_params(layer);
        let rope = super::super::rope::RotaryCache::new(
            cfg.rope_head_dim,
            theta,
            orig,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
            &device,
        )?;
        let x = Tensor::randn(0f32, 1.0, (1, 16, cfg.dim), &device)?;
        let out = att.forward(&x, &rope)?;
        assert_eq!(out.dims(), &[1, 16, cfg.dim]);
        let v = out.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            v.iter().all(|x| x.is_finite()),
            "non-finite attention output"
        );
        let maxabs = v.iter().fold(0f32, |a, &x| a.max(x.abs()));
        eprintln!(
            "[ok] real CSA attention layer {layer} ran: out {:?}, max|.|={maxabs:.3}",
            out.dims()
        );
        Ok(())
    }

    /// The config parsed from the real GGUF matches the known DeepSeek-V4-Flash-0731
    /// hyperparameters. Skips when the model is absent.
    #[test]
    fn config_from_real_gguf() -> Result<()> {
        let first = first_split();
        if !first.exists() {
            eprintln!("[skip] real GGUF not present");
            return Ok(());
        }
        let splits: Vec<PathBuf> = GgufModel::discover_splits(&first)?
            .into_iter()
            .filter(|p| p.exists())
            .collect();
        let m = GgufModel::open(&splits)?;
        let cfg = config_from_gguf(&m)?;
        assert_eq!(cfg.n_layers, 43);
        assert_eq!(cfg.dim, 4096);
        assert_eq!(cfg.vocab_size, 129280);
        assert_eq!(cfg.n_heads, 64);
        assert_eq!(cfg.head_dim, 512);
        assert_eq!(cfg.rope_head_dim, 64);
        assert_eq!(cfg.q_lora_rank, 1024);
        assert_eq!(cfg.o_groups, 8);
        assert_eq!(cfg.o_lora_rank, 1024);
        assert_eq!(cfg.n_routed_experts, 256);
        assert_eq!(cfg.n_activated_experts, 6);
        assert_eq!(cfg.n_shared_experts, 1);
        assert_eq!(cfg.moe_inter_dim, 2048);
        assert_eq!(cfg.n_hash_layers, 3);
        assert_eq!(cfg.window_size, 128);
        assert_eq!(cfg.index_n_heads, 64);
        assert_eq!(cfg.index_head_dim, 128);
        assert_eq!(cfg.index_topk, 512);
        assert_eq!(cfg.hc_mult, 4);
        assert_eq!(cfg.hc_sinkhorn_iters, 20);
        assert_eq!((cfg.route_scale * 10.0).round(), 15.0);
        assert_eq!(cfg.swiglu_limit, 10.0);
        // compress_ratios: L0,L1 = 0; L2 = 4 (CSA); L3 = 128 (HCA).
        assert_eq!(cfg.compress_ratio(0), 0);
        assert_eq!(cfg.compress_ratio(1), 0);
        assert_eq!(cfg.compress_ratio(2), 4);
        assert_eq!(cfg.compress_ratio(3), 128);
        assert_eq!(cfg.layer_kind(2), crate::models::deepseek4::LayerKind::Csa);
        eprintln!("[ok] real config: {cfg:?}");
        Ok(())
    }

    /// Decode a real MXFP4 routed-expert tensor on the GPU and verify the values are
    /// sane FP4 reconstructions (finite, bounded by the E2M1 range × block scale). This
    /// exercises the production GPU FP4 path on the actual trained weights. Ignored by
    /// default (needs the 156 GB model + a CUDA device); run explicitly.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn real_mxfp4_expert_decodes_on_gpu() -> Result<()> {
        let first = first_split();
        if !first.exists() {
            eprintln!("[skip] real GGUF not present");
            return Ok(());
        }
        let splits: Vec<PathBuf> = GgufModel::discover_splits(&first)?
            .into_iter()
            .filter(|p| p.exists())
            .collect();
        let mut m = GgufModel::open(&splits)?;
        let device = Device::new_cuda(0)?;

        let name = "blk.0.ffn_gate_exps.weight";
        let info = m.info(name).expect("expert tensor present");
        assert_eq!(info.ggml_dtype, candle::quantized::GgmlDType::MXFP4);
        assert_eq!(info.shape.dims(), &[256, 2048, 4096]);
        let (inter, dim) = (2048usize, 4096usize);

        // Load the full MXFP4 tensor (CPU), then slice out just expert 0's raw bytes and
        // decode that [2048, 4096] slice on the GPU. Dequantizing all 256 experts at once
        // would be 2^31 elements — exactly one past the i32 `elem_count` the dequant FFI
        // accepts — so real code (and this test) decodes per expert.
        let q = m.qtensor(name, &candle::Device::Cpu)?;
        let expert0_bytes = inter * dim / 32 * 17; // 32 elems/block, 17 bytes/block
        let bytes = q.data_range(0..expert0_bytes)?;
        let (qg, _guard) = candle::quantized::QTensor::from_host_mapped_ggml(
            candle::quantized::GgmlDType::MXFP4,
            &bytes,
            vec![inter, dim],
            &device,
        )?;
        let deq = qg.dequantize(&device)?; // [2048, 4096] f32
        let vals = deq.flatten_all()?.to_vec1::<f32>()?;

        assert!(vals.iter().all(|v| v.is_finite()), "non-finite FP4 decode");
        let maxabs = vals.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let nonzero = vals.iter().filter(|&&v| v != 0.0).count();
        // FP4 magnitudes are E2M1 (max 6) × a per-block power-of-two scale; trained MoE
        // weights are small but a whole expert is not all-zero.
        assert!(
            nonzero > vals.len() / 2,
            "expert mostly zeros: {nonzero}/{}",
            vals.len()
        );
        assert!(maxabs < 100.0, "implausible FP4 magnitude {maxabs}");
        eprintln!(
            "[ok] real MXFP4 expert 0 [{inter},{dim}] on GPU: max|w|={maxabs:.4}, nonzero={nonzero}/{}",
            vals.len()
        );
        Ok(())
    }
}
