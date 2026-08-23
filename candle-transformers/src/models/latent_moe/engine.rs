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

use candle::quantized::cuda::{to_dynamic, DynamicActs};
use candle::quantized::{get_vram_info, gguf_file, Int8Mode, MmapRegistration};
use candle::{DType, Device, Result, Tensor, D};
use memmap2::MmapOptions;

use crate::models::expert_lre::{
    layer_geometries, minimum_resident_slots, slot_bytes_for, ExpertCache, ExpertCacheSetup,
    MmapExpertRef, MoeInput,
};
use crate::models::profile::span;
use candle_nn::kv_cache::WeightZone;

use super::arch::{Arch, Ffn, Global, Hyper as HyperSite, Weight};
use super::config::Config;
use super::hyper::{HyperConnection, HyperParams};
use super::linear::QLinear;
use super::loader::{self, GgufModel};
use super::moe::{Expert, Gate, ScoreFunc};
use super::paged;
use super::rope::RotaryCache;

/// One transformer layer's resident (non-routed-expert) weights. The routed experts for this
/// layer live in the shared [`ExpertCache`], indexed by `moe_layer_idx`.
pub(super) struct EngineLayer {
    pub(super) attn: super::attention::Attention,
    pub(super) hc_attn: HyperParams,
    pub(super) hc_ffn: HyperParams,
    pub(super) attn_norm: Tensor,
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
pub struct Engine {
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
}

impl Engine {
    /// Load the merged single-file GGUF into the resident engine model.
    ///
    /// Non-expert weights load to `device` (VRAM); the routed experts are registered as
    /// byte-ranges into the mmap and served by the `ExpertCache`'s three tiers (VRAM slots
    /// leased from the span's weight zone / pinned warm bank / the repacked `.experts.pack`
    /// on NVMe; int8-KO when `int8mode` is enabled — the 2×-FP16 grouped GEMM path).
    /// Load a model of architecture `arch` from its merged GGUF.
    ///
    /// `arch` is what makes the engine model-agnostic: it supplies the config
    /// defaults, metadata namespace, and tensor names. Pass
    /// [`deepseek4::DEEPSEEK_V4`](crate::models::deepseek4::DEEPSEEK_V4) for
    /// DeepSeek-V4-Flash.
    pub fn load(
        merged_path: &Path,
        arch: &'static dyn Arch,
        device: &Device,
        int8mode: Int8Mode,
    ) -> Result<Self> {
        Ok(Self::load_with_drafter(merged_path, None, arch, device, int8mode)?.0)
    }

    /// [`Self::load`], with the DSpark speculative-decode drafter loaded as part of the
    /// engine's **dense tier**: the drafter's int8 backbone and its all-resident Q2_KO
    /// expert set land in VRAM *before* the span reservation is taken, so they are
    /// permanent residents the reservation measures around — exactly like the attention
    /// weights. The elastic KV↔expert boundary then balances the *target's* experts
    /// against KV in whatever the card has left; nothing about the drafter competes with
    /// the span at runtime. (Loading a drafter after the span would have to live in the
    /// pool cushion, which the activation peak already owns.)
    ///
    /// Returns the drafter alongside the engine; attach it with
    /// [`super::wave::BatchedEngine::with_drafter`].
    pub fn load_with_drafter(
        merged_path: &Path,
        dspark_path: Option<&Path>,
        arch: &'static dyn Arch,
        device: &Device,
        int8mode: Int8Mode,
    ) -> Result<(Self, Option<super::dspark::DsparkDrafter>)> {
        // GgufModel handles config + convenient loading of the resident (non-expert) tensors.
        let mut gguf = GgufModel::open(std::slice::from_ref(&merged_path.to_path_buf()))?;
        let cfg = loader::config_from_gguf(&gguf, arch)?;

        // Refuse a model/host/kernel geometry disagreement before any arena is laid
        // out: the band offsets in every KvHead record written from here on assume
        // one latent shape, and a divergence reads as wrong attention, not a fault.
        // Runs after the config parse because the checkpoint's own dims are one of
        // the four declarations that have to agree.
        paged::assert_geometry(&cfg)?;

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
                    .contains_key(&cfg.arch.weight(i, Weight::RoutedExperts(Ffn::Gate)))
            })
            .collect();
        let mut all_host_refs: Vec<Vec<MmapExpertRef>> = Vec::with_capacity(moe_layers.len());
        for &i in &moe_layers {
            let get = |f: Ffn| -> Result<&gguf_file::TensorInfo> {
                let name = cfg.arch.weight(i, Weight::RoutedExperts(f));
                ct.tensor_infos
                    .get(&name)
                    .ok_or_else(|| candle::Error::msg(format!("missing {name}")))
            };
            let (gi, ui, di) = (get(Ffn::Gate)?, get(Ffn::Up)?, get(Ffn::Down)?);
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

        // ── VRAM governor ──
        // Balloon to measure the real resident capacity `C`, then install it so
        // the span reservation, KV admission, and the weight zone all coordinate
        // through one authority. The reservation (`region_pool`) sizes itself
        // from `governor.usable()` at first touch — which happens BELOW, after
        // every dense tensor is resident, so the span takes exactly what is
        // genuinely left (`docs/elastic_vram_partition.md` §4).
        let total_experts = moe_layers.len() * n_expert;
        let gb = |b: usize| b as f64 / (1usize << 30) as f64;
        #[cfg(feature = "cuda")]
        let gpu_id = match device.location() {
            candle::DeviceLocation::Cuda { gpu_id } => gpu_id,
            _ => 0,
        };
        #[cfg(feature = "cuda")]
        if matches!(device, Device::Cuda(_)) && candle::vram::get(gpu_id).is_none() {
            // DeepSeek's forward still allocates its per-wave transients from
            // the ordinary CUDA pool (it has not adopted the span's transient
            // tier), so the cushion the reservation leaves OUTSIDE the span
            // must cover the activation peak — a wide batched prefill
            // materialises several GiB. 6 GiB matches the engine's measured
            // reserve on this card; when the forward moves onto the wave
            // arenas this drops back to the 512 MiB pool default.
            let config = candle::vram::GovernorConfig {
                scratch_margin: 6 << 30,
                ..Default::default()
            };
            match candle::vram::VramGovernor::from_device_with_config(device, gpu_id, config) {
                Ok(gov) => {
                    let mut balloon =
                        candle::vram::balloon::DeviceBalloonAllocator::new(device.clone());
                    match gov.run_balloon(&mut balloon) {
                        Ok(c) => eprintln!(
                            "[mem] VRAM governor installed: capacity C={:.1}GB",
                            c as f64 / 1e9
                        ),
                        Err(e) => eprintln!("[mem] VRAM governor balloon failed: {e}"),
                    }
                    candle::vram::install(gov);
                }
                Err(e) => eprintln!("[mem] VRAM governor init failed: {e}"),
            }
        }
        let (free_before_dense, _total_vram) = get_vram_info()?;

        // ── Resident non-expert weights ──
        // Token embedding stays in HOST RAM: it's a lookup table, read one row per token
        // (~5 KB), so gathering the row on the host and uploading just that row costs nothing
        // against a ~50-70 ms/token decode — and it frees ~1.3 GB of VRAM. See `step`.
        let embed =
            loader::dequant_f32(&mut gguf, cfg.arch.global(Global::Embedding), &Device::Cpu)?;
        let mut layers = Vec::with_capacity(moe_layers.len());
        // This model is MoE on every layer, so the MoE index equals the layer index; the
        // enumerate index is the ExpertCache slot-group. (A dense leading block would need a
        // separate arm here — none exist in this model.)
        for (moe_layer_idx, &i) in moe_layers.iter().enumerate() {
            let w = |x| cfg.arch.weight(i, x);
            let attn = loader::load_attention(&mut gguf, &cfg, i, device, int8mode)?;
            let hc_attn = loader::load_hc_params(
                &mut gguf,
                loader::hc_block(&cfg, HyperSite::Attn, i),
                device,
                int8mode,
            )?;
            let hc_ffn = loader::load_hc_params(
                &mut gguf,
                loader::hc_block(&cfg, HyperSite::Ffn, i),
                device,
                int8mode,
            )?;
            let attn_norm = loader::dequant_f32(&mut gguf, &w(Weight::AttnNorm), device)?;
            let ffn_norm = loader::dequant_f32(&mut gguf, &w(Weight::FfnNorm), device)?;

            let gate_w = loader::qlinear_int8(&mut gguf, &w(Weight::FfnGateInp), device, int8mode)?;
            let (bias, tid2eid) = if cfg.is_hash_layer(i) {
                let t = gguf.read_int_tensor_u32(&w(Weight::FfnGateTid2Eid), device)?;
                (None, Some(t))
            } else {
                let b = loader::dequant_f32(&mut gguf, &w(Weight::ExpProbsBias), device)?;
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
                    &w(Weight::SharedExpert(Ffn::Gate)),
                    device,
                    int8mode,
                )?,
                loader::qlinear_int8(
                    &mut gguf,
                    &w(Weight::SharedExpert(Ffn::Down)),
                    device,
                    int8mode,
                )?,
                loader::qlinear_int8(
                    &mut gguf,
                    &w(Weight::SharedExpert(Ffn::Up)),
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
             span reservation next",
            gb(free_after_resident),
            gb(free_before_dense.saturating_sub(free_after_resident)),
        );

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

        // ── Drafter (dense tier) ──
        //
        // Loaded HERE — after the target's dense weights, before the span
        // reservation — so its backbone + all-resident Q2_KO expert set are
        // permanent residents the reservation measures around.
        let drafter = match dspark_path {
            Some(p) => {
                let drafter_arch = arch.drafter().ok_or_else(|| {
                    candle::Error::msg(format!(
                        "a DSpark drafter was supplied but {} declares none — its GGUF metadata \
                         would be read under the wrong namespace",
                        arch.id()
                    ))
                })?;
                let (free_b, _) = get_vram_info()?;
                let d = super::dspark::DsparkDrafter::load(p, drafter_arch, device)?;
                let (free_a, _) = get_vram_info()?;
                eprintln!(
                    "[mem] drafter loaded into the dense tier: {:.2} GB (free {:.1}→{:.1} GB)",
                    gb(free_b.saturating_sub(free_a)),
                    gb(free_b),
                    gb(free_a),
                );
                Some(d)
            }
            None => None,
        };

        // ── Reserve the span, then build the expert cache into it ──
        //
        // Every dense tensor (and the drafter, when speculative decode is on)
        // is resident at this point, so the governor's `usable()` reports what
        // is genuinely left and the reservation (created lazily by the first
        // `span_end` call) takes it. The weight zone opens at the span's right
        // edge; its capacity in slots IS the resident-expert count — no byte
        // budget, no headroom constant (`docs/elastic_vram_partition.md` §4,
        // `docs/expert_cache_design.md`).
        #[cfg(feature = "cuda")]
        let zone = if let Device::Cuda(cuda_dev) = device {
            use candle_nn::kv_cache::{
                initial_weight_bytes, set_weight_floor, span_end, weight_capacity_bytes,
            };
            let stream = cuda_dev.cuda_stream();
            let geoms = layer_geometries(&all_host_refs, int8mode)?;
            let slot_bytes = slot_bytes_for(&geoms);
            let limit_bytes = weight_capacity_bytes(&stream)?;
            let initial_bytes = initial_weight_bytes(&stream)?;
            let slots_in = |bytes: usize| {
                bytes
                    .checked_div(slot_bytes)
                    .map_or(0, |n| n.min(total_experts))
            };
            let capacity = slots_in(initial_bytes);
            let limit = slots_in(limit_bytes);
            let floor = minimum_resident_slots(n_expert);
            let zone = WeightZone::new(span_end(&stream)?, slot_bytes, capacity, limit, floor);
            let regions = set_weight_floor(&stream, zone.frontier_for_capacity())?;
            eprintln!(
                "[mem] expert cache opened against the span: slots={capacity} (max {limit}, \
                 floor {floor}) of {total_experts} experts, slot={:.1}MB → resident {:.1}GB, \
                 kv_regions={regions}",
                slot_bytes as f64 / 1e6,
                gb(capacity * slot_bytes),
            );
            zone
        } else {
            WeightZone::new(0, 0, 0, 0, 0)
        };
        #[cfg(not(feature = "cuda"))]
        let zone = WeightZone::new(0, 0, total_experts, total_experts, 0);
        let zone_capacity = zone.capacity();
        let zone_slot_bytes = zone.slot_bytes();

        let experts = Arc::new(ExpertCache::new(ExpertCacheSetup {
            mmap: mmap.clone(),
            host_refs: all_host_refs,
            zone,
            device,
            experts_per_layer: n_expert,
            gguf_path: merged_path,
            // Persistent pack beside the GGUF: written once on first boot
            // (repacked kernel-layout records), authoritative cold tier after.
            expert_pack_dir: merged_path.parent(),
            progress: None,
            int8mode,
        })?);
        #[cfg(feature = "cuda")]
        {
            if let Some(g) = candle::vram::get(gpu_id) {
                g.set_class(
                    candle::vram::AllocClass::Expert,
                    (zone_capacity * zone_slot_bytes) as u64,
                );
            }
            // Open the shop: a KV arena claim that runs out of ground can buy
            // more at the price of expert residency (eviction = drop, the pack
            // re-supplies) instead of refusing.
            let seller = Arc::downgrade(&experts);
            candle_nn::kv_cache::set_ground_broker(gpu_id, move |regions| {
                seller
                    .upgrade()
                    .map_or(0, |cache| cache.request_kv_ground(regions))
            });
        }
        #[cfg(not(feature = "cuda"))]
        let _ = (zone_capacity, zone_slot_bytes);

        Ok((
            Self {
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
            },
            drafter,
        ))
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// RoPE table for `layer`: long-context (compressed layers) vs base (sliding-window).
    pub(super) fn rope_for(&self, layer: usize) -> &RotaryCache {
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
        self.moe_forward_batch(layer, x, &[token_id])
    }

    /// Batched MoE over `nt` rows: ONE routing readback per call (the
    /// counting-sort's expert ids must be host-visible to schedule the
    /// streaming cache's pinned→VRAM uploads — intrinsic to a non-resident
    /// expert set, and amortized across every row of the wave).
    pub(super) fn moe_forward_batch(
        &self,
        layer: &EngineLayer,
        x: &Tensor,
        token_ids: &[u32],
    ) -> Result<Tensor> {
        let dim = self.cfg.dim;
        let nt = token_ids.len();
        let x2 = x.reshape((nt, dim))?;
        let s_route = span("moe:route");
        // Float-normalized input for routing + the shared expert.
        let normed = rms_norm(&x2, &layer.ffn_norm, self.cfg.norm_eps)?;
        let ids = Tensor::from_vec(token_ids.to_vec(), nt, &self.device)?;
        let (weights, indices) = layer.gate.route(&normed, &ids)?; // [nt,k], [nt,k] u32

        // q8a128 activation for the int8-KO grouped expert GEMM: quantize the SAME
        // `normed` the router already saw (a quantize-only launch), rather than
        // re-normalizing `x2` a second time. One RMSNorm reduction over [nt, dim] per
        // layer instead of two, and the expert input is now exactly the routed
        // normalization (a single source of truth); quantization noise only, within QAT.
        let cuda_dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => candle::bail!("Engine::moe_forward requires a CUDA device"),
        };
        let q8 = match to_dynamic(&normed, Int8Mode::Performance, &cuda_dev)? {
            DynamicActs::Int8(op) => op,
            DynamicActs::Float(_) => {
                candle::bail!("q8a128 activation quantize returned a non-int8 operand")
            }
        };
        s_route.end();

        // Counting-sort the (token, expert) assignments by ascending expert id (O(A+E)),
        // matching the grouped-GEMM dispatch contract (see `SparseMoeBlock::forward_with_indices`).
        // The ONE intrinsic wave-path readback: the paged expert cache schedules
        // pinned→VRAM uploads by expert id, so the routing indices must be
        // host-visible (amortized across every row of the wave). A dedicated-routing-stream async
        // DtoH into a pinned buffer was measured here and REVERTED: neutral on single-session
        // speculative but −8% on the cfg8 batched gate (589.9→541.0), because the per-layer
        // event/side-stream/sync overhead over 44 layers × N sessions outweighs the pinned-copy
        // saving — the readback is a genuine per-layer GPU-catch-up wait, not a hideable flush.
        let s_sort = span("moe:sort");
        super::readback::note_readback();
        // Split out the readback itself: it is a synchronous D2H, so it does not
        // just transfer 4 bytes per routed token — it blocks until every kernel
        // issued this layer has retired. Timing it apart from the counting sort
        // is what separates "the host sort is slow" (fixable by moving it to the
        // GPU) from "the pipeline drains 43 times per token" (fixable only by
        // removing the sync). The sort below is O(A+E) over ~128 assignments, so
        // any large number here is the drain.
        let s_rb = span("moe:sort_readback");
        let idx_cpu: Vec<Vec<u32>> = indices.to_vec2::<u32>()?;
        s_rb.end();
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

        s_sort.end();
        let s_submit = span("moe:submit");
        let routed = self.experts.submit_moe_work(
            layer.moe_layer_idx,
            expert_ids,
            MoeInput::Q8(q8),
            DType::F32,
            &weights_flat,
            assignments,
            None,
        )?; // [nt, dim] F32
        s_submit.end();
        let s_shared = span("moe:shared");
        let shared = layer.shared.forward(&normed)?; // [nt, dim] F32
        let out = (routed + shared)?.reshape((1, nt, dim));
        s_shared.end();
        out
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

    // ── Accessors for the batched wave model (`deepseek4/wave.rs`) — the
    // resident engine IS the loaded model; the wave impl drives its weights,
    // experts, and hyper-connections over many sequences. ──
    pub(super) fn cfg(&self) -> &Config {
        &self.cfg
    }
    pub(super) fn hc(&self) -> &HyperConnection {
        &self.hc
    }
    pub(super) fn hc_head(&self) -> &HyperParams {
        &self.hc_head
    }
    pub(super) fn embed(&self) -> &Tensor {
        &self.embed
    }
    pub(super) fn output_norm(&self) -> &Tensor {
        &self.output_norm
    }
    pub(super) fn lm_head(&self) -> &QLinear {
        &self.lm_head
    }
    pub(super) fn engine_device(&self) -> &Device {
        &self.device
    }
    pub(super) fn layer_count(&self) -> usize {
        self.layers.len()
    }
    pub(super) fn engine_layer(&self, l: usize) -> &EngineLayer {
        &self.layers[l]
    }
    /// The shared routed-expert cache — used to surface its pipeline telemetry
    /// (hit rate, DMA/fence stalls) and worker-thread profile (upload-wait vs
    /// GEMM) up through the `ManagedBatchedModel` profiling hooks.
    pub(super) fn experts(&self) -> &ExpertCache {
        &self.experts
    }
    /// The paged-kernel decode session: attention runs in the
    /// `paged-latent` kernel over a production chunked-arena slot
    /// (single-latent FP8 window) + `FloatGallery` corpus; the host keeps the
    /// int8 projections and MoE. One sequence, one slot per layer group.
    pub fn kernel_session(&self) -> Result<KernelSession<'_>> {
        use crate::models::batched_inference::{BatchedConfig, BatchedInferenceSession};
        use candle_nn::kv_cache::{ChunkedKvBacking, KvFormat};

        let cfg = BatchedConfig {
            k_format: KvFormat::Float(DType::F8E4M3),
            v_format: KvFormat::Float(DType::F8E4M3),
            initial_seq_len: 4096,
            ..Default::default()
        };
        let first = ChunkedKvBacking::new_with_format_adaptive(
            1,
            1,
            self.cfg.head_dim,
            cfg.k_format,
            cfg.v_format,
            &self.device,
            cfg.initial_seq_len,
            None,
        )?;
        first.set_single_latent(true);
        let mut backings = Vec::with_capacity(self.cfg.n_layers);
        backings.push(first.clone());
        for layer_idx in 1..self.cfg.n_layers {
            backings.push(first.new_layer(layer_idx, 1, cfg.initial_seq_len));
        }
        let mut kv =
            BatchedInferenceSession::new_with_backings(backings.clone(), cfg, &self.device);
        let seq = kv.create_sequence()?;

        let mut layers = Vec::with_capacity(self.cfg.n_layers);
        let ws = std::sync::Arc::new(super::paged::LatentWorkspace::build(&self.device)?);
        for (l, layer) in self.layers.iter().enumerate() {
            let (theta, orig) = self.cfg.rope_params(l);
            layers.push(super::kernel_attention::KernelAttnLayer::new(
                &layer.attn,
                theta,
                orig,
                self.cfg.rope_factor,
                self.cfg.beta_fast,
                self.cfg.beta_slow,
                self.cfg.index_head_dim,
                ws.clone(),
                &self.device,
            )?);
        }
        Ok(KernelSession {
            engine: self,
            kv,
            backings,
            seq,
            layers,
            pos: 0,
        })
    }

    /// Greedy decode on the kernel path — the rung-3 vehicle
    /// (`generate`'s host-attention twin with the attention in the kernel).
    pub fn generate_kernel(&self, prompt: &[u32], max_new: usize) -> Result<Vec<u32>> {
        let mut sess = self.kernel_session()?;
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

/// The paged-kernel decode session: per-layer [`KernelAttnLayer`] state over
/// a shared `BatchedInferenceSession` slot group.
pub struct KernelSession<'a> {
    engine: &'a Engine,
    kv: crate::models::batched_inference::BatchedInferenceSession,
    /// Per-layer backings (Arc-shared with `kv`) — the per-step CPU usage
    /// commit (`set_len`) runs on each before the slot headers serialize.
    backings: Vec<candle_nn::kv_cache::ChunkedKvBacking>,
    seq: usize,
    layers: Vec<super::kernel_attention::KernelAttnLayer>,
    pos: usize,
}

impl KernelSession<'_> {
    /// One decode step on the kernel path: `token_id` → logits `[vocab]`.
    pub fn step(&mut self, token_id: u32) -> Result<Tensor> {
        let e = self.engine;
        let dim = e.cfg.dim;
        let idt = Tensor::from_vec(vec![token_id], 1, &Device::Cpu)?;
        let row = e
            .embed
            .index_select(&idt, 0)?
            .reshape((1, 1, dim))?
            .to_dtype(DType::F32)?
            .to_device(&e.device)?;
        let mut h = e.hc.expand(&row)?;

        // Slide the sliding-window ring BEFORE building this step's metadata:
        // free every front chunk that has fully exited the `window_size`-token
        // window ending at the query's absolute position (`self.pos`). Without
        // this the FP8 window arena grows one chunk per 32 tokens forever and
        // the decode kernel walks every one (masking all but the last
        // `window_size`) — O(N) memory and per-step work. Evicting bounds both
        // to O(window_size); positions stay ABSOLUTE (the freed count folds
        // into each backing's `base_pos`), so the attention is unchanged.
        // Every layer's slot has the identical chunk layout (they decode in
        // lockstep), so the evicted count is uniform across backings.
        let window = e.cfg.window_size;
        let mut evicted = 0u32;
        for b in &self.backings {
            evicted = b.evict_window_front(self.seq, window, self.pos)?;
        }
        let resident = self.pos - evicted as usize;

        // Per-step slot metadata for ALL layers (24-byte SlotHeader each,
        // one pinned upload). Kept alive through the layer loop. The CPU-side
        // chunk usage must mirror the RESIDENT tokens the GPU commits have
        // written (absolute `self.pos` minus the evicted front) — `set_len`
        // distributes that across each layer's writer chunks before the
        // headers serialize. The kernel still ropes at absolute positions
        // (q_pos = `self.pos`, chunk `rope_base` = `base_pos` + cum-usage).
        self.kv.set_sequence_offset(self.seq, resident)?;
        for b in &self.backings {
            b.set_len(self.seq, resident);
        }
        let generation = self.kv.begin_stager_generation();
        let (_pm, headers, stride) = self.kv.build_decode_metadata(&[self.seq], &generation)?;
        let headers = headers.ok_or_else(|| candle::Error::msg("no decode metadata"))?;
        let base = headers.dev_ptr();

        for (l, layer) in e.layers.iter().enumerate() {
            let (x, post, comb) = e.hc.pre(&h, &layer.hc_attn)?;
            let x = rms_norm(&x, &layer.attn_norm, e.cfg.norm_eps)?;
            let x = self.layers[l].step(
                &layer.attn,
                &x,
                e.rope_for(l),
                self.pos,
                base + (l as u64) * stride,
            )?;
            let h1 = e.hc.post(&x, &h, &post, &comb)?;

            let (x, post, comb) = e.hc.pre(&h1, &layer.hc_ffn)?;
            let moe = e.moe_forward(layer, &x, token_id)?;
            h = e.hc.post(&moe, &h1, &post, &comb)?;
        }
        drop(generation);
        self.pos += 1;

        let h = e.hc.head_reduce(&h, &e.hc_head)?;
        let h = rms_norm(&h, &e.output_norm, e.cfg.norm_eps)?;
        let logits = e.lm_head.forward(&h)?;
        logits.reshape((e.cfg.vocab_size,))
    }
}

/// A decode session: one streaming [`IncrementalAttention`] per layer (the session-owned KV) plus
/// the running position. Feeds tokens one at a time through the resident model.
pub struct EngineSession<'a> {
    engine: &'a Engine,
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
    // End-to-end gates against the real DeepSeek-V4-Flash checkpoint.
    use crate::models::deepseek4::DEEPSEEK_V4;

    fn merged() -> std::path::PathBuf {
        std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            // Pre-repacked MXFP4_KO file (offline `prepare_ko_gguf`): experts are already the
            // lane-major KO twin, so staging skips the runtime repack entirely (fast load).
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf")
    }

    /// RUNG 3 — the step-4 milestone: the engine answers "Paris" with the
    /// attention running entirely in the `paged-latent` kernel (FP8 arena
    /// window + FloatGallery corpus + two-stage selection); host attention is
    /// gone from this path. Ignored (needs the merged file + CUDA).
    #[test]
    #[ignore]
    fn engine_generate_paris_kernel() -> Result<()> {
        let path = merged();
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let t0 = std::time::Instant::now();
        let engine = Engine::load(&path, &DEEPSEEK_V4, &device, Int8Mode::Performance)?;
        eprintln!("[kernel] load {:.1}s", t0.elapsed().as_secs_f32());

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
        let gen = engine.generate_kernel(&ids, 12)?;
        let dt = t1.elapsed().as_secs_f32();
        let text = tokenizer
            .decode(&gen, false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        eprintln!("[kernel] generated ids={gen:?}");
        eprintln!("[kernel] continuation={text:?}");
        eprintln!(
            "[kernel] {} prompt + 12 gen in {:.1}s = {:.2} tok/s decode",
            ids.len(),
            dt,
            12.0 / dt
        );
        assert!(
            text.contains("Paris"),
            "kernel-path engine did not answer Paris: {text:?}"
        );
        Ok(())
    }

    /// Template-distribution discriminator for the rung-4 conversation gate:
    /// the conversation-engine prompt (BOS + system text + chat markers) run
    /// through BOTH the reference per-token attention (`generate`) and the
    /// paged-kernel path (`generate_kernel`) on the same loaded weights.
    /// Agreement (both crisp or both rambling) attributes the rung-4 output
    /// to the model's template distribution; divergence attributes it to the
    /// kernel path. Ignored (needs the merged file + CUDA).
    #[test]
    #[ignore]
    fn engine_conversation_prompt_ab() -> Result<()> {
        let path = merged();
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let engine = Engine::load(&path, &DEEPSEEK_V4, &device, Int8Mode::Performance)?;

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        let prompt = "<｜begin▁of▁sentence｜>You are a concise, factual assistant.\
             <｜User｜>What is the capital of France? \
             Reply with only the city name.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();

        let gen_ref = engine.generate(&ids, 16)?;
        let text_ref = tokenizer
            .decode(&gen_ref, false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        eprintln!("[ab] reference ids={gen_ref:?}");
        eprintln!("[ab] reference continuation={text_ref:?}");

        let gen_k = engine.generate_kernel(&ids, 16)?;
        let text_k = tokenizer
            .decode(&gen_k, false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        eprintln!("[ab] kernel    ids={gen_k:?}");
        eprintln!("[ab] kernel    continuation={text_k:?}");
        eprintln!("[ab] identical={}", gen_ref == gen_k);
        Ok(())
    }

    /// Step-3 gate (a): per-CSA-layer recall sweep on REAL Indexer spaces —
    /// does the training-free sign top-M contain the learned float top-k?
    /// Runs a real multi-hundred-token generation so every CSA layer
    /// accumulates a meaningful compressed-entry count, captures each layer's
    /// (query, weights, entry keys) at the final step, and sweeps recall@M.
    /// The probe k is a strict subset of the entry count at this depth (the
    /// production `index_topk = 512` exceeds it — the full-depth re-sweep
    /// rides the step-6 long-context runs). Ignored (needs the merged file +
    /// CUDA; several minutes).
    #[test]
    #[ignore]
    fn indexer_recall_sweep_real_traces() -> Result<()> {
        use super::super::gallery::{bdp_recall, sign_pack, topm_select};

        let path = merged();
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let engine = Engine::load(&path, &DEEPSEEK_V4, &device, Int8Mode::Performance)?;
        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer load: {e}")))?;
        let prompt = "<｜begin▁of▁sentence｜><｜User｜>Write a detailed, factual essay about \
             the history of navigation at sea: dead reckoning, the astrolabe, the marine \
             chronometer and the longitude problem, radio beacons, and satellite \
             positioning. Cover each era with specific dates, names, and instruments, and \
             explain how each advance changed trade routes and naval strategy.<｜Assistant｜>";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .map_err(|e| candle::Error::msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();

        let mut sess = engine.session()?;
        let mut logits = None;
        for &t in &ids {
            logits = Some(sess.step(t)?);
        }
        let mut logits = logits.unwrap();
        // Generate enough real text that CSA layers hold ~100 entries each.
        let gen_len = 320usize;
        let mut last = 0u32;
        for i in 0..gen_len {
            let next = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
            last = next;
            if i + 2 == gen_len {
                for a in sess.attn.iter_mut() {
                    a.set_capture_indexer_space(true);
                }
            }
            logits = sess.step(next)?;
        }
        let _ = last;

        let probe_k = 8usize;
        let mut worst_recall_at_4k = 1.0f32;
        for (l, a) in sess.attn.iter().enumerate() {
            let Some((q, w, kv)) = a.captured_space.as_ref() else {
                continue;
            };
            let n = kv.dim(0)?;
            if n < 4 * probe_k {
                continue;
            }
            // Float reference top-k (host).
            let scores = q
                .matmul(&kv.t()?.contiguous()?)?
                .relu()?
                .broadcast_mul(&w.reshape(((), 1))?)?
                .sum(0)?
                .to_vec1::<f32>()?;
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&x, &y| scores[y].partial_cmp(&scores[x]).unwrap());
            let full: std::collections::HashSet<usize> = order[..probe_k].iter().copied().collect();

            // Sign recall on-device.
            let q_signs = sign_pack(q)?;
            let kv_signs = sign_pack(kv)?;
            let ih = kv.dim(1)?;
            let counts = bdp_recall(&q_signs, &kv_signs, ih)?;
            let bins = q.dim(0)? * ih + 1;
            let mut line = format!("[sweep] layer {l:2} G={n:4}:");
            for mult in [1usize, 2, 4, 8] {
                let m = (probe_k * mult).min(n);
                let ids = topm_select(&counts, m, bins)?.to_vec1::<u32>()?;
                let short: std::collections::HashSet<usize> =
                    ids.into_iter().map(|v| v as usize).collect();
                let hit = full.iter().filter(|g| short.contains(g)).count();
                let recall = hit as f32 / probe_k as f32;
                if mult == 4 {
                    worst_recall_at_4k = worst_recall_at_4k.min(recall);
                }
                line.push_str(&format!("  R@{mult}k={recall:.2}"));
            }
            // Machinery sanity: shortlist == everything ⇒ recall 1.
            let ids = topm_select(&counts, n, bins)?.to_vec1::<u32>()?;
            let all: std::collections::HashSet<usize> =
                ids.into_iter().map(|v| v as usize).collect();
            assert_eq!(
                full.iter().filter(|g| all.contains(g)).count(),
                probe_k,
                "layer {l}: recall@G must be 1"
            );
            eprintln!("{line}");
        }
        eprintln!("[sweep] worst recall@4k across CSA layers = {worst_recall_at_4k:.2}");
        Ok(())
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
        let engine = Engine::load(&path, &DEEPSEEK_V4, &device, Int8Mode::Performance)?;
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
        let n_ctx = ids.len() + 12;
        use super::super::footprint;
        eprintln!(
            "[engine] kv ratio vs FP16-linear @ {n_ctx} tokens: {:.1}x ({} B vs {} B)",
            footprint::ratio_vs_fp16_linear(n_ctx, engine.config()),
            footprint::deepseek_kv_footprint(n_ctx, engine.config()).total(),
            footprint::fp16_linear_baseline_bytes(n_ctx, engine.config()),
        );
        assert!(
            text.contains("Paris"),
            "engine did not answer Paris: {text:?}"
        );
        Ok(())
    }
}
