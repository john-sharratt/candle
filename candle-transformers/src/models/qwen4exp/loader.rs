//! GGUF loading for the qwen4exp oracle: split discovery, F32 assembly of
//! everything that fits memory, and on-demand disk sources for what does not.
//!
//! The weight split (`docs/qwen38_flash_next.md` §12.8):
//!
//! - **Dequantized F32 at load** (~19 GB on the real checkpoint): embeddings,
//!   the LM head, every HC module, the GDN and attention mixers, the indexer,
//!   the PLE projections, routers and shared experts.
//! - **Routed experts stay on disk**: each expert's rows are contiguous in
//!   the `[n_experts, …]` expert tensors, so one expert is one positioned
//!   read, requantized into a small `QTensor` and dequantized per use.
//! - **The PLE table stays on disk**: a row is a handful of quant blocks
//!   (170 bytes at Q8_0), gathered per token.

use std::path::Path;
use std::sync::Arc;

use candle::quantized::ggml_file::qtensor_from_ggml;
use candle::quantized::gguf_file::Value;
use candle::quantized::ko_quant::dequant_ko;
use candle::quantized::GgmlDType;
use candle::{Device, Result, Tensor};

use super::config::Qwen4ExpConfig;
use super::hyper::HcWeights;
use super::model::{ExpertSource, LayerMix, PleSource, Qwen4ExpLayer, Qwen4ExpModel};
use super::ple::PleWeights;
use super::ple_cache::{PleCacheStats, PleRowCache, PleRowFetch};
use super::qsa::IndexerWeights;
use crate::models::delta_net::{DeltaNetWeights, LayerKind};
use crate::models::latent_moe::GgufModel;
use crate::models::qwen35::attention::{AttentionWeights, RopeTables};
use crate::models::qwen35::moe::FfnWeights;

/// A quantized tensor left on disk, addressed by ranges of its outermost
/// dimension. `dims` is the full candle shape (outermost first); a range of
/// `count` outer indices reads `count × inner/block × type_size` contiguous
/// bytes — the layout guarantee that makes an expert (or a table row) one
/// positioned read.
#[derive(Debug, Clone)]
struct DiskSlab {
    split: usize,
    /// Absolute file offset of the tensor's first byte.
    offset: u64,
    dtype: GgmlDType,
    dims: Vec<usize>,
    /// Bytes per outermost index.
    outer_bytes: usize,
}

impl DiskSlab {
    fn locate(gguf: &GgufModel, name: &str) -> Result<Self> {
        let (split, offset, info) = gguf
            .raw_location(name)
            .ok_or_else(|| candle::Error::Msg(format!("qwen4exp: missing tensor {name}")))?;
        let dims = info.shape.dims().to_vec();
        if dims.len() < 2 {
            candle::bail!("qwen4exp: {name} is not sliceable by outer index: {dims:?}");
        }
        let inner: usize = dims[1..].iter().product();
        let dtype = info.ggml_dtype;
        if !inner.is_multiple_of(dtype.block_size()) {
            candle::bail!("qwen4exp: {name} inner extent {inner} does not tile {dtype:?} blocks");
        }
        Ok(Self {
            split,
            offset,
            dtype,
            dims: dims.clone(),
            outer_bytes: inner / dtype.block_size() * dtype.type_size(),
        })
    }

    /// Read outer indices `[start, start + count)` and dequantize to F32.
    fn read_f32(
        &self,
        gguf: &GgufModel,
        start: usize,
        count: usize,
        device: &Device,
    ) -> Result<Tensor> {
        if start + count > self.dims[0] {
            candle::bail!(
                "qwen4exp: outer range {start}+{count} exceeds {} rows",
                self.dims[0]
            );
        }
        let mut buf = vec![0u8; count * self.outer_bytes];
        gguf.read_at(
            self.split,
            self.offset + (start * self.outer_bytes) as u64,
            &mut buf,
        )?;
        let mut dims = self.dims.clone();
        dims[0] = count;
        if self.dtype.is_ko() {
            // KO bytes are the lane-major GPU chunk layout; `qtensor_from_ggml`
            // refuses them on the CPU (there is no CPU block codec for that
            // layout), so the oracle decodes through `dequant_ko` — each outer
            // index is one whole 2-D KO image.
            let inner: Vec<usize> = dims[1..].to_vec();
            let (rows, cols) = match inner.len() {
                2 => (inner[0], inner[1]),
                other => {
                    candle::bail!("qwen4exp: a KO slab must be outer × 2-D, got {other}+1 dims")
                }
            };
            let per = self.outer_bytes;
            let mut vals = Vec::with_capacity(count * rows * cols);
            for c in 0..count {
                vals.extend(dequant_ko(
                    &buf[c * per..(c + 1) * per],
                    rows,
                    cols,
                    self.dtype,
                ));
            }
            return Tensor::from_vec(vals, dims, device);
        }
        qtensor_from_ggml(self.dtype, &buf, dims, device)?.dequantize(device)
    }
}

/// The routed experts of every layer, on disk. Layer `li`'s three slabs are
/// indexed by expert: `gate`/`up` are `[E, ffn, hidden]`, `down` is
/// `[E, hidden, ffn]`, so expert `e` is outer index `e` of each.
struct DiskExperts {
    gguf: Arc<GgufModel>,
    device: Device,
    /// `[layer] → (gate, up, down)`.
    slabs: Vec<(DiskSlab, DiskSlab, DiskSlab)>,
}

impl ExpertSource for DiskExperts {
    fn expert(&self, layer: usize, e: usize) -> Result<FfnWeights> {
        let (gate, up, down) = &self.slabs[layer];
        Ok(FfnWeights {
            gate: gate.read_f32(&self.gguf, e, 1, &self.device)?.squeeze(0)?,
            up: up.read_f32(&self.gguf, e, 1, &self.device)?.squeeze(0)?,
            down: down.read_f32(&self.gguf, e, 1, &self.device)?.squeeze(0)?,
        })
    }
}

/// The PLE table's disk half: a positioned read of one row's quantized
/// record — what the row cache calls on a miss.
struct DiskPleFetch {
    gguf: Arc<GgufModel>,
    slab: DiskSlab,
}

impl PleRowFetch for DiskPleFetch {
    fn fetch(&self, id: u32, dst: &mut [u8]) -> Result<()> {
        if id as usize >= self.slab.dims[0] {
            candle::bail!("ple: row {id} past the {}-row table", self.slab.dims[0]);
        }
        self.gguf.read_at(
            self.slab.split,
            self.slab.offset + id as u64 * self.slab.outer_bytes as u64,
            dst,
        )
    }
}

/// The PLE table behind the §0.1 cache: NVMe records, a bounded non-pinned
/// RAM row cache, rows held and gathered **quantized** (the engine uploads
/// the gathered records and dequantizes on the card; the oracle widens the
/// same bytes host-side, so the two consume identical records).
struct CachedPle {
    cache: PleRowCache<DiskPleFetch>,
    dtype: GgmlDType,
    head_dim: usize,
    device: Device,
}

impl PleSource for CachedPle {
    fn rows(&self, ids: &[u32]) -> Result<Tensor> {
        let mut buf = Vec::new();
        self.cache.gather(ids, &mut buf)?;
        qtensor_from_ggml(
            self.dtype,
            &buf,
            vec![ids.len(), self.head_dim],
            &self.device,
        )?
        .dequantize(&self.device)
    }

    fn cache_stats(&self) -> Option<PleCacheStats> {
        Some(self.cache.stats())
    }
}

/// §0.1's figure: the row cache's arena budget.
const PLE_CACHE_BYTES: usize = 2 << 30;

/// Dequantize one whole tensor to F32.
fn f32t(gguf: &mut GgufModel, name: &str, device: &Device) -> Result<Tensor> {
    gguf.qtensor(name, device)?.dequantize(device)
}

/// One HC module's four (or three, at the head) tensors.
fn hc_weights(
    gguf: &mut GgufModel,
    prefix: &str,
    with_inject: bool,
    device: &Device,
) -> Result<HcWeights> {
    // The inject rows are stacked under the down projection, exactly as the
    // engine loader does it — `HcWeights::down` says why. The oracle has to
    // build the same layout or it would not be reading the same weight.
    let down = f32t(gguf, &format!("{prefix}_down.weight"), device)?;
    let down = if with_inject {
        let inject = f32t(gguf, &format!("{prefix}_inject.weight"), device)?;
        Tensor::cat(&[&down, &inject], 0)?.contiguous()?
    } else {
        down
    };
    Ok(HcWeights {
        norm: f32t(gguf, &format!("{prefix}_norm.weight"), device)?,
        down,
        up: f32t(gguf, &format!("{prefix}_up.weight"), device)?,
    })
}

/// Load the oracle model from any member path of the split GGUF. Everything
/// lands on `device` (the oracle's home is the CPU).
pub fn load_oracle_model(one_split: &Path, device: &Device) -> Result<Qwen4ExpModel> {
    let paths = GgufModel::discover_splits(one_split)?;
    let mut gguf = GgufModel::open(&paths)?;

    match gguf.metadata.get("general.architecture") {
        Some(Value::String(a)) if a == "qwen4exp" => {}
        other => candle::bail!(
            "qwen4exp: general.architecture is {other:?}, not \"qwen4exp\" — wrong checkpoint"
        ),
    }
    let cfg = Qwen4ExpConfig::from_gguf_metadata(&gguf.metadata)?;

    let embed = f32t(&mut gguf, "token_embd.weight", device)?;
    let lm_head = if gguf.info("output.weight").is_some() {
        f32t(&mut gguf, "output.weight", device)?
    } else {
        embed.clone()
    };
    let out_hc = hc_weights(&mut gguf, "output_hc", false, device)?;

    let mut layers = Vec::with_capacity(cfg.num_layers);
    let mut expert_slabs = Vec::with_capacity(cfg.num_layers);
    let mut ple_w: Option<PleWeights> = None;
    for li in 0..cfg.num_layers {
        let p = format!("blk.{li}");
        let g = &mut gguf;
        let hc_attn = hc_weights(g, &format!("{p}.hc_attn"), true, device)?;
        let hc_ffn = hc_weights(g, &format!("{p}.hc_ffn"), true, device)?;

        let mix = match cfg.layer_kinds[li] {
            LayerKind::DeltaNet => LayerMix::DeltaNet(DeltaNetWeights {
                wqkv: f32t(g, &format!("{p}.attn_qkv.weight"), device)?,
                wz: f32t(g, &format!("{p}.attn_gate.weight"), device)?,
                w_beta: f32t(g, &format!("{p}.ssm_beta.weight"), device)?,
                w_alpha: f32t(g, &format!("{p}.ssm_alpha.weight"), device)?,
                dt_bias: f32t(g, &format!("{p}.ssm_dt.bias"), device)?,
                a: f32t(g, &format!("{p}.ssm_a"), device)?,
                conv: f32t(g, &format!("{p}.ssm_conv1d.weight"), device)?,
                norm: f32t(g, &format!("{p}.ssm_norm.weight"), device)?,
                w_out: f32t(g, &format!("{p}.ssm_out.weight"), device)?,
            }),
            LayerKind::Attention => LayerMix::Attention {
                attn: AttentionWeights {
                    wq: f32t(g, &format!("{p}.attn_q.weight"), device)?,
                    wk: f32t(g, &format!("{p}.attn_k.weight"), device)?,
                    wv: f32t(g, &format!("{p}.attn_v.weight"), device)?,
                    wo: f32t(g, &format!("{p}.attn_output.weight"), device)?,
                    q_norm: f32t(g, &format!("{p}.attn_q_norm.weight"), device)?,
                    k_norm: f32t(g, &format!("{p}.attn_k_norm.weight"), device)?,
                },
                indexer: IndexerWeights {
                    q_proj: f32t(g, &format!("{p}.indexer.q_proj.weight"), device)?,
                    k_proj: f32t(g, &format!("{p}.indexer.k_proj.weight"), device)?,
                    q_norm: f32t(g, &format!("{p}.indexer.q_norm.weight"), device)?,
                    k_norm: f32t(g, &format!("{p}.indexer.k_norm.weight"), device)?,
                },
                compress_ratio: cfg.compress_ratios[li],
            },
        };

        if li == cfg.ple.layer {
            ple_w = Some(PleWeights {
                key: f32t(g, &format!("{p}.ple_key.weight"), device)?,
                value: f32t(g, &format!("{p}.ple_value.weight"), device)?,
                norm_key: f32t(g, &format!("{p}.ple_norm_key.weight"), device)?,
                norm_query: f32t(g, &format!("{p}.ple_norm_query.weight"), device)?,
                norm_conv: f32t(g, &format!("{p}.ple_norm_conv.weight"), device)?,
                conv: f32t(g, &format!("{p}.ple_conv1d.weight"), device)?,
            });
        }

        let router = f32t(g, &format!("{p}.ffn_gate_inp.weight"), device)?;
        let shared = FfnWeights {
            gate: f32t(g, &format!("{p}.ffn_gate_shexp.weight"), device)?,
            up: f32t(g, &format!("{p}.ffn_up_shexp.weight"), device)?,
            down: f32t(g, &format!("{p}.ffn_down_shexp.weight"), device)?,
        };
        let shared_gate = f32t(g, &format!("{p}.ffn_gate_inp_shexp.weight"), device)?
            .reshape((1, cfg.hidden_size))?;

        expert_slabs.push((
            DiskSlab::locate(&gguf, &format!("{p}.ffn_gate_exps.weight"))?,
            DiskSlab::locate(&gguf, &format!("{p}.ffn_up_exps.weight"))?,
            DiskSlab::locate(&gguf, &format!("{p}.ffn_down_exps.weight"))?,
        ));
        for (name, slab) in [
            ("gate", &expert_slabs[li].0),
            ("up", &expert_slabs[li].1),
            ("down", &expert_slabs[li].2),
        ] {
            if slab.dims[0] != cfg.moe.n_experts {
                candle::bail!(
                    "qwen4exp: blk.{li} ffn_{name}_exps has {} experts, metadata says {}",
                    slab.dims[0],
                    cfg.moe.n_experts
                );
            }
        }

        layers.push(Qwen4ExpLayer {
            hc_attn,
            hc_ffn,
            mix,
            router,
            shared,
            shared_gate,
        });
    }

    let ple_w = ple_w.ok_or_else(|| {
        candle::Error::Msg(format!(
            "qwen4exp: PLE layer {} produced no weights — schedule/tensor mismatch",
            cfg.ple.layer
        ))
    })?;

    let rope = RopeTables::new(
        cfg.rope_dim,
        cfg.rope_theta,
        cfg.max_position_embeddings.min(65_536),
        device,
    )?;

    let gguf = Arc::new(gguf);
    let ple_table = open_cached_ple(gguf.clone(), &cfg, device)?;
    Ok(Qwen4ExpModel {
        embed,
        layers,
        ple_w,
        out_hc,
        lm_head,
        rope,
        experts: Box::new(DiskExperts {
            gguf,
            device: device.clone(),
            slabs: expert_slabs,
        }),
        ple_table,
        cfg,
    })
}

/// Open the §0.1 PLE source — the 320M-row table behind the bounded RAM row
/// cache — over an already-open GGUF. Shared by the oracle above and the GPU
/// engine, so both consume identical quantized records.
pub(crate) fn open_cached_ple(
    gguf: Arc<GgufModel>,
    cfg: &Qwen4ExpConfig,
    device: &Device,
) -> Result<Box<dyn PleSource>> {
    let ple_slab = DiskSlab::locate(&gguf, "per_layer_token_embd.weight")?;
    if ple_slab.dims[1] != cfg.ple.head_dim {
        candle::bail!(
            "qwen4exp: PLE table width {} != embedding_length_per_layer_input {}",
            ple_slab.dims[1],
            cfg.ple.head_dim
        );
    }
    let table_rows = ple_slab.dims[0] as u64;
    for (h, (&off, &vocab)) in cfg
        .ple
        .head_offsets
        .iter()
        .zip(&cfg.ple.head_vocab_sizes)
        .enumerate()
    {
        if off + vocab > table_rows {
            candle::bail!(
                "qwen4exp: PLE head {h} range {off}+{vocab} exceeds the {table_rows}-row table"
            );
        }
    }
    let ple_record_bytes = ple_slab.outer_bytes;
    Ok(Box::new(CachedPle {
        dtype: ple_slab.dtype,
        head_dim: cfg.ple.head_dim,
        device: device.clone(),
        cache: PleRowCache::new(
            DiskPleFetch {
                gguf,
                slab: ple_slab,
            },
            ple_record_bytes,
            PLE_CACHE_BYTES,
        )?,
    }))
}
