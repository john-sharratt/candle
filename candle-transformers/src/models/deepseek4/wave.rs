//! DeepSeek-V4-Flash as a `ManagedBatchedModel`: the conversation engine's
//! wave-forward over the resident [`Dsv4Engine`], with attention in the
//! `paged-latent` kernels and the mHC hyper-connection loop private to this
//! implementation (the scheduler sees only the trait surface — `WaveStep`
//! residuals are opaque, so the multi-stream mHC state rides them directly).
//!
//! Per-sequence corpus state (galleries + streaming compressors, one per
//! compression layer) lives behind interior mutability keyed by sequence
//! index: the trait has no sequence-free hook, so a prefill arriving at
//! offset 0 for a known index resets that sequence's state, and `prune`
//! clears everything.

use std::collections::HashMap;
use std::sync::RwLock;

use candle::{DType, Device, Result, Tensor};

use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ManagedBatchedModel, WaveStep,
};

use super::attention::{rms_norm, rms_scale};
use super::engine::Dsv4Engine;
use super::kernel_attention::{KernelLayerSeqState, KernelLayerStatic};
use super::paged::HEAD_DIM;
use crate::models::profile::{pipeline_record, profile_now};

/// One sequence's per-layer corpus state plus the offset bookkeeping used to
/// detect slot reuse.
struct SeqEntry {
    /// Tokens this state has absorbed (compressor pushes). A prefill starting
    /// below this (or at 0) means the slot was freed and reused — reset.
    absorbed: usize,
    layers: Vec<KernelLayerSeqState>,
}

/// DeepSeek's batched wave model. See the module docs.
pub struct DeepSeekBatched {
    engine: Dsv4Engine,
    layer_static: Vec<KernelLayerStatic>,
    seq_state: RwLock<HashMap<usize, SeqEntry>>,
}

impl DeepSeekBatched {
    pub fn new(engine: Dsv4Engine) -> Result<Self> {
        let cfg = engine.cfg();
        let mut layer_static = Vec::with_capacity(cfg.n_layers);
        let ws = std::sync::Arc::new(super::paged::LatentWorkspace::build(
            engine.engine_device(),
        )?);
        for l in 0..cfg.n_layers {
            let (theta, orig) = cfg.rope_params(l);
            layer_static.push(KernelLayerStatic::new(
                &engine.engine_layer(l).attn,
                theta,
                orig,
                cfg.rope_factor,
                cfg.beta_fast,
                cfg.beta_slow,
                ws.clone(),
                engine.engine_device(),
            )?);
        }
        Ok(Self {
            engine,
            layer_static,
            seq_state: RwLock::new(HashMap::new()),
        })
    }

    pub fn engine(&self) -> &Dsv4Engine {
        &self.engine
    }

    fn fresh_seq_entry(&self) -> Result<SeqEntry> {
        let cfg = self.engine.cfg();
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            layers.push(KernelLayerSeqState::new(
                &self.engine.engine_layer(l).attn,
                cfg.index_head_dim,
                self.engine.engine_device(),
            )?);
        }
        Ok(SeqEntry {
            absorbed: 0,
            layers,
        })
    }

    /// Ensure per-seq state exists and matches the incoming positions; reset
    /// on slot reuse (prefill starting at/below a position we've already
    /// absorbed past, at offset 0).
    fn ensure_seq_state(&self, seq: usize, start_pos: usize) -> Result<()> {
        let mut map = self
            .seq_state
            .write()
            .map_err(|_| candle::Error::Msg("seq_state lock poisoned".into()))?;
        let reset = match map.get(&seq) {
            None => true,
            Some(e) => start_pos == 0 && e.absorbed > 0,
        };
        if reset {
            map.insert(seq, self.fresh_seq_entry()?);
        }
        Ok(())
    }

    /// Extract host token ids from an input tensor (`[1, s]` or `[s]`).
    /// Inputs are scheduler-built host tensors; when one arrives on the GPU
    /// this is a transfer — counted by the readback instrumentation.
    fn token_ids(t: &Tensor) -> Result<Vec<u32>> {
        if t.device().is_cuda() {
            super::readback::note_readback();
        }
        Ok(t.flatten_all()?.to_dtype(DType::U32)?.to_vec1::<u32>()?)
    }

    /// Embed a list of token ids: host-resident gather, one upload.
    fn embed_rows(&self, ids: &[u32]) -> Result<Tensor> {
        let dim = self.engine.cfg().dim;
        let idt = Tensor::from_vec(ids.to_vec(), ids.len(), &Device::Cpu)?;
        self.engine
            .embed()
            .index_select(&idt, 0)?
            .reshape((1, ids.len(), dim))?
            .to_dtype(DType::F32)?
            .to_device(self.engine.engine_device())
    }
}

impl ManagedBatchedModel for DeepSeekBatched {
    fn num_layers(&self) -> usize {
        self.engine.layer_count()
    }

    fn n_kv_head(&self) -> usize {
        1
    }

    fn head_dim(&self) -> usize {
        HEAD_DIM
    }

    fn device(&self) -> &Device {
        self.engine.engine_device()
    }

    fn create_batched_session(&self, config: BatchedConfig) -> Result<BatchedInferenceSession> {
        use candle_nn::kv_cache::{ChunkedKvBacking, KvFormat};
        // DeepSeek pins the WRITER format to the reference two-region latent
        // window: the non-RoPE dims `[0,448)` are FP8 E4M3, the 64 RoPE dims
        // `[448,512)` stay BF16 (the rope tail must not lose precision). The
        // single-latent backing splits the two regions per head automatically
        // (`alloc_block_chunks`): the nope bands take this writer format tag,
        // the rope bands are pinned BF16. The fused scatter + fresh-diagonal
        // quant dispatch on the per-band format tag. The window stores FP8 at
        // unit scale (a per-64 ue8m0 scale would need per-token record scales —
        // the window ring is written a token at a time into a per-chunk scalar
        // record, so a non-unit scale is not representable here).
        let cfg = BatchedConfig {
            k_format: KvFormat::Float(DType::F8E4M3),
            v_format: KvFormat::Float(DType::F8E4M3),
            ..config
        };
        let first = ChunkedKvBacking::new_with_format_adaptive(
            1,
            1,
            HEAD_DIM,
            cfg.k_format,
            cfg.v_format,
            self.engine.engine_device(),
            cfg.initial_seq_len,
            None,
        )?;
        first.set_single_latent(true);
        let mut backings = Vec::with_capacity(self.num_layers());
        backings.push(first.clone());
        for layer_idx in 1..self.num_layers() {
            backings.push(first.new_layer(layer_idx, 1, cfg.initial_seq_len));
        }
        Ok(BatchedInferenceSession::new_with_backings(
            backings,
            cfg,
            self.engine.engine_device(),
        ))
    }

    fn prune(&self) -> Result<()> {
        if let Ok(mut map) = self.seq_state.write() {
            map.clear();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_wave(
        &self,
        session: &mut BatchedInferenceSession,
        decode_seqs: &[usize],
        decode_inputs: &[Tensor],
        prefill_seqs: &[usize],
        prefill_inputs: &[Tensor],
        glue_seqs: &[usize],
        glue_inputs: &[Tensor],
        layer_start: usize,
        layer_end: usize,
        residual_in: Option<Tensor>,
    ) -> Result<WaveStep> {
        if decode_inputs.len() != decode_seqs.len()
            || prefill_inputs.len() != prefill_seqs.len()
            || glue_inputs.len() != glue_seqs.len()
        {
            candle::bail!("forward_wave: input/seq length mismatch");
        }
        let e = &self.engine;
        let cfg = e.cfg();
        let n_layers = self.num_layers();
        let hc = e.hc();

        // Token ids (host) + per-seq geometry.
        let decode_ids: Vec<u32> = decode_inputs
            .iter()
            .map(|t| Ok(Self::token_ids(t)?[0]))
            .collect::<Result<Vec<_>>>()?;
        let prefill_ids: Vec<Vec<u32>> = prefill_inputs
            .iter()
            .map(Self::token_ids)
            .collect::<Result<Vec<_>>>()?;
        let prefill_lens: Vec<usize> = prefill_ids.iter().map(|v| v.len()).collect();
        let glue_ids: Vec<Vec<u32>> = glue_inputs
            .iter()
            .map(Self::token_ids)
            .collect::<Result<Vec<_>>>()?;
        let glue_lens: Vec<usize> = glue_ids.iter().map(|v| v.len()).collect();

        // Glue descriptors: staged on the session before this call, one per
        // glue sequence in order. Positions derive from the reserved gap
        // chunks: block logical start + (in-block offset − chunk offset).
        // DeepSeek glue is causal-only (locked §E) — fwd_ahead must be 0.
        let glue_desc = if glue_seqs.is_empty() {
            Vec::new()
        } else {
            let desc = session.take_pending_glue().ok_or_else(|| {
                candle::Error::Msg("glue rows without staged PendingGlue descriptors".into())
            })?;
            if desc.len() != glue_seqs.len() {
                candle::bail!(
                    "PendingGlue count {} != glue seq count {}",
                    desc.len(),
                    glue_seqs.len()
                );
            }
            desc
        };
        let mut glue_pos: Vec<Vec<u32>> = Vec::with_capacity(glue_seqs.len());
        for (gi, &seq) in glue_seqs.iter().enumerate() {
            let d = &glue_desc[gi];
            if d.fwd_ahead.iter().any(|&f| f != 0) {
                candle::bail!("deepseek glue is causal-only (fwd_ahead must be 0)");
            }
            let chunks = session.backings()[0]
                .live_chunks_as_sealed(seq, &[])
                .unwrap_or_default();
            let mut block_start = Vec::with_capacity(chunks.len());
            let mut block_off = Vec::with_capacity(chunks.len());
            let mut cum = 0usize;
            for c in &chunks {
                block_start.push(cum);
                block_off.push(c.offset as usize);
                cum += c.token_count as usize;
            }
            let mut pos = Vec::with_capacity(d.write_slice.len());
            for (s_idx, &blk) in d.write_slice.iter().enumerate() {
                let b = blk as usize;
                if b >= block_start.len() {
                    candle::bail!("glue descriptor block {b} out of table range");
                }
                let within = d.write_in_blk[s_idx] as usize;
                pos.push((block_start[b] + within.saturating_sub(block_off[b])) as u32);
            }
            if pos.len() != glue_lens[gi] {
                candle::bail!(
                    "glue descriptor tokens {} != glue input tokens {}",
                    pos.len(),
                    glue_lens[gi]
                );
            }
            glue_pos.push(pos);
        }

        // Sequence offsets (position of the first new token per sequence).
        let decode_pos: Vec<usize> = decode_seqs
            .iter()
            .map(|&s| session.sequence_offset(s).unwrap_or(0))
            .collect();
        let prefill_base: Vec<usize> = prefill_seqs
            .iter()
            .map(|&s| session.sequence_offset(s).unwrap_or(0))
            .collect();

        // Per-seq corpus state (reset on slot reuse).
        for (&s, &base) in prefill_seqs.iter().zip(&prefill_base) {
            self.ensure_seq_state(s, base)?;
        }
        for &s in decode_seqs.iter().chain(glue_seqs) {
            self.ensure_seq_state(s, 1)?; // decode/glue never reset
        }

        // Residual: fresh waves embed all rows [decode… | prefill… | glue…]
        // flat and expand into the mHC stream; resumed waves reuse it.
        let total_rows: usize = decode_seqs.len()
            + prefill_lens.iter().sum::<usize>()
            + glue_lens.iter().sum::<usize>();
        let mut h = match residual_in {
            Some(r) => {
                // Persisted as [1, rows, hc·dim]; restore the mHC axes.
                r.reshape((1, total_rows, cfg.hc_mult, cfg.dim))?
            }
            None => {
                let mut flat_ids: Vec<u32> = decode_ids.clone();
                for ids in prefill_ids.iter().chain(&glue_ids) {
                    flat_ids.extend_from_slice(ids);
                }
                let rows = self.embed_rows(&flat_ids)?;
                hc.expand(&rows)?
            }
        };

        // Metadata: commit each layer's CPU chunk usage to the committed
        // prefix, then serialize slot headers for decode ++ prefill ++ glue
        // sequences (glue tables are already the scheduler-reserved state —
        // no usage commit). Prefill sequences get ONE HEADER SNAPSHOT PER
        // TOKEN: prompt token `t` absorbs through the SAME per-token decode
        // step the decode rows use, so its launch needs headers serialized at
        // offset `base + t`. This is the ONLY batched absorption that
        // bit-matches per-token decode: the decode kernel bf16-STAGES the
        // current token's PV (diagonal), while a settled-slot prefill would
        // read the diagonal as FP8 from the arena — a lossy difference that
        // heavy early-layer self-attention amplifies into CSA selection flips
        // (measured: garbage output). A single-launch batched prefill that
        // reproduces the decode diagonal needs a kernel-level bf16 diagonal
        // source (§L optimization note); until then, per-token is correct.
        // Snapshot `t` covers every layer; non-prefill slots serialize
        // identically in each snapshot, so snapshot 0 serves decode + glue.
        let all_seqs: Vec<usize> = decode_seqs
            .iter()
            .chain(prefill_seqs)
            .chain(glue_seqs)
            .copied()
            .collect();
        for backing in session.backings() {
            for (&s, &pos) in decode_seqs.iter().zip(&decode_pos) {
                backing.set_len(s, pos);
            }
        }
        let t_meta = profile_now();
        let generation = session.begin_stager_generation();
        // Glue-only sequences never decode-write (their latents scatter into
        // reserved gap chunks) — excluding them from the metadata build's
        // write-chunk ensure keeps their block tables untouched, so the
        // scheduler's turn-start block accounting stays exact.
        let non_writer: Vec<usize> = glue_seqs
            .iter()
            .copied()
            .filter(|s| !decode_seqs.contains(s) && !prefill_seqs.contains(s))
            .collect();
        // Prefill absorbs each prompt in ONE `paged_latent_prefill` launch per
        // layer (the per-token decode-step loop it replaces was the dominant
        // wall-clock cost — see the wave profile). So each prefill seq needs a
        // SINGLE header snapshot at its committed-prefix length `base` (the arena
        // walk covers `[0,base)`; the fresh bf16 latents cover `[base,base+s)`),
        // with its whole write-back range ensured up-front. Decode/glue seqs
        // serialize their live offsets in the same snapshot.
        let mut overrides: Vec<(usize, usize)> = Vec::with_capacity(prefill_seqs.len());
        for (pi, &s) in prefill_seqs.iter().enumerate() {
            let base = prefill_base[pi];
            // Ensure the FULL prompt range so the post-launch arena write-back
            // has its chunks; the header stays at `base` (the committed prefix).
            let ensure_to = base + prefill_lens[pi].saturating_sub(1);
            for backing in session.backings() {
                backing.ensure_for_batch_entries(&[(s, ensure_to)], 1)?;
                backing.set_len(s, base);
            }
            overrides.push((s, base));
        }
        let (pm, headers, stride) = session.build_decode_metadata_at(
            &all_seqs,
            &generation,
            &overrides,
            &non_writer,
            true,
        )?;
        let headers = headers.ok_or_else(|| candle::Error::Msg("no decode metadata".into()))?;
        let snaps = [(pm, headers, stride)];
        let hdr_of = |layer: usize, seq_slot: usize| -> u64 {
            let (_, headers, stride) = &snaps[0];
            headers.dev_ptr() + (layer as u64) * stride + (seq_slot as u64) * 24
        };
        pipeline_record("deepseek:wave_metadata", t_meta);

        let mut state = self
            .seq_state
            .write()
            .map_err(|_| candle::Error::Msg("seq_state lock poisoned".into()))?;

        for l in layer_start..layer_end {
            let layer = e.engine_layer(l);
            let a = &layer.attn;
            let st = &self.layer_static[l];
            let rope = e.rope_for(l);

            // Attention sub-block.
            let t_attn_pre = profile_now();
            let (x, post, comb) = hc.pre(&h, &layer.hc_attn)?;
            let x = rms_norm(&x, &layer.attn_norm, cfg.norm_eps)?;
            pipeline_record("deepseek:hc_pre_norm", t_attn_pre);

            // Phase A — glue scatter FIRST: every glue row's latent lands in
            // its reserved gap chunk before ANY attention pass of this layer
            // reads the arena (stream-ordered; glue keys then read like any
            // window key — no double-source, no garbage gaps).
            let glue_row_base = decode_seqs.len() + prefill_lens.iter().sum::<usize>();
            let mut glue_proj: Vec<(Tensor, Tensor)> = Vec::with_capacity(glue_seqs.len());
            let t_glue_scatter = profile_now();
            {
                let mut cursor = glue_row_base;
                for (gi, _seq) in glue_seqs.iter().enumerate() {
                    let g_len = glue_lens[gi];
                    let xs = x.narrow(1, cursor, g_len)?.to_dtype(DType::F32)?;
                    let qr = rms_norm(&a.wq_a().forward(&xs)?, a.q_norm(), a.eps())?;
                    let kv = rms_norm(&a.wkv().forward(&xs)?, a.kv_norm(), a.eps())?;
                    let kv_bf = kv.reshape((g_len, a.head_dim()))?.to_dtype(DType::BF16)?;
                    let d = &glue_desc[gi];
                    let dev = xs.device();
                    let sl = Tensor::from_vec(d.write_slice.clone(), g_len, dev)?;
                    let ib = Tensor::from_vec(d.write_in_blk.clone(), g_len, dev)?;
                    super::paged::paged_latent_glue_scatter(
                        &kv_bf,
                        hdr_of(l, decode_seqs.len() + prefill_seqs.len() + gi),
                        &sl,
                        &ib,
                    )?;
                    glue_proj.push((xs, qr));
                    cursor += g_len;
                }
            }
            if !glue_seqs.is_empty() {
                pipeline_record("deepseek:glue_scatter", t_glue_scatter);
            }

            let mut attn_rows: Vec<Tensor> = Vec::with_capacity(total_rows);
            // Decode rows: one kernel step per sequence.
            // Decode rows: all sessions attend in ONE `paged_latent_decode`
            // launch over every decode slot (grid.x = n_decode), instead of a
            // per-session launch loop. Per session we run the same host
            // projections + on-device corpus select/gather, then concatenate the
            // gathered compressed blocks — each slot's selection is the dense
            // range `[offset, offset+k)` into the concat, so per-session galleries
            // stay isolated with no cross-bleed and no GID readback.
            let t_decode = profile_now();
            if !decode_seqs.is_empty() {
                let n_dec = decode_seqs.len();
                let mut q_rows = Vec::with_capacity(n_dec);
                let mut kv_rows = Vec::with_capacity(n_dec);
                // Per-slot HOT two-region cache blocks, concatenated across slots
                // into one launch (each slot's selection is a dense offset range).
                let mut nope_blocks: Vec<Tensor> = Vec::new();
                let mut scale_blocks: Vec<Tensor> = Vec::new();
                let mut rope_blocks: Vec<Tensor> = Vec::new();
                let mut pos_blocks: Vec<Tensor> = Vec::new();
                let mut cnts: Vec<u32> = Vec::with_capacity(n_dec);
                let mut offsets: Vec<u32> = Vec::with_capacity(n_dec);
                let mut off = 0u32;
                for (i, &seq) in decode_seqs.iter().enumerate() {
                    let xi = x.narrow(1, i, 1)?;
                    let entry = state.get_mut(&seq).expect("ensured above");
                    let (q_bf, kv_bf, block, k) =
                        super::kernel_attention::kernel_attn_decode_prepare(
                            a,
                            &mut entry.layers[l],
                            &xi,
                            rope,
                            decode_pos[i],
                        )?;
                    q_rows.push(q_bf);
                    kv_rows.push(kv_bf);
                    if let Some((ni8, nsc, rbf, cpos)) = block {
                        nope_blocks.push(ni8);
                        scale_blocks.push(nsc);
                        rope_blocks.push(rbf);
                        pos_blocks.push(cpos);
                    }
                    offsets.push(off);
                    cnts.push(k as u32);
                    off += k as u32;
                    if l + 1 == n_layers {
                        entry.absorbed = decode_pos[i] + 1;
                    }
                }
                let q_all = Tensor::cat(&q_rows, 0)?; // [n_dec, H, hd]
                let kv_all = Tensor::cat(&kv_rows, 0)?; // [n_dec, hd]
                let dev = q_all.device();
                let cache = if nope_blocks.is_empty() {
                    st.empty_corpus_cache()?
                } else {
                    let nope = Tensor::cat(&nope_blocks, 0)?;
                    let clen = nope.dim(0)?;
                    super::paged::CorpusCache::from_gathered(
                        nope,
                        Tensor::cat(&scale_blocks, 0)?,
                        Tensor::cat(&rope_blocks, 0)?,
                        Tensor::cat(&pos_blocks, 0)?,
                        clen,
                    )?
                };
                let max_sel = cnts.iter().map(|&c| c as usize).max().unwrap_or(0).max(1);
                // Each slot's selection is the dense range [offset, offset+k) —
                // strictly ascending, the compressed-index contract the kernel
                // documents (attention is order-independent, but the contract
                // holds by construction here).
                let mut idx_flat = vec![u32::MAX; n_dec * max_sel];
                for i in 0..n_dec {
                    for k in 0..cnts[i] as usize {
                        idx_flat[i * max_sel + k] = offsets[i] + k as u32;
                    }
                }
                let comp_idx = Tensor::from_vec(idx_flat, (n_dec, max_sel), dev)?;
                let comp_cnt = Tensor::from_vec(cnts, n_dec, dev)?;
                // Explicit per-slot query position (the decode kernel no longer
                // derives it from the writer slice, so the windowless slot works
                // and the compressed causal guard has a reference).
                let q_pos_dec = Tensor::from_vec(
                    (0..n_dec).map(|i| decode_pos[i] as u32).collect::<Vec<u32>>(),
                    n_dec,
                    dev,
                )?;
                // `cache` is the gathered two-region hot cache (built above from
                // the gallery's pre-built int8 — no per-wave rebuild).
                let out = super::paged::paged_latent_decode_raw(
                    &q_all,
                    hdr_of(l, 0),
                    &kv_all,
                    &cache,
                    &comp_idx,
                    &comp_cnt,
                    &q_pos_dec,
                    st.sinks(),
                    st.rope_tab(),
                    a.softmax_scale() as f32,
                    a.window_size(),
                    0,
                    false,
                    st.ws(),
                    None,
                )?;
                for i in 0..n_dec {
                    let oi = out.narrow(0, i, 1)?.to_dtype(DType::F32)?.reshape((
                        1,
                        a.n_heads(),
                        1,
                        a.head_dim(),
                    ))?;
                    attn_rows.push(a.output_proj(&oi, 1, 1)?);
                }
                pipeline_record("deepseek:decode_attn", t_decode);
            }
            // Prefill rows: each prompt is absorbed in ONE batched
            // `paged_latent_prefill` launch per layer (argmax-equal to per-token
            // decode absorption, validated by `wave_prefill_state_matches_decode_steps`),
            // which replaced the per-token launch loop that dominated the profile.
            let t_prefill = profile_now();
            let mut row_cursor = decode_seqs.len();
            for (pi, &seq) in prefill_seqs.iter().enumerate() {
                let s_len = prefill_lens[pi];
                let base = prefill_base[pi];
                let entry = state.get_mut(&seq).expect("ensured above");
                // One launch over the whole prompt. Per token: host projections +
                // causal corpus-select (identical to the decode step's selection),
                // then a single `paged_latent_prefill` reads the committed prefix
                // `[0,base)` from the arena and the prompt `[base,base+s)` from the
                // fresh bf16 latents (`kv_fresh`) — matching the per-token diagonal.
                let mut q_rows = Vec::with_capacity(s_len);
                let mut kv_rows = Vec::with_capacity(s_len);
                let mut idx_rows: Vec<Vec<u32>> = Vec::with_capacity(s_len);
                for t in 0..s_len {
                    let xi = x.narrow(1, row_cursor + t, 1)?;
                    let (q_bf, kv_bf, gids) = super::kernel_attention::kernel_attn_prefill_prepare(
                        a,
                        &mut entry.layers[l],
                        &xi,
                        rope,
                        base + t,
                    )?;
                    q_rows.push(q_bf);
                    kv_rows.push(kv_bf);
                    idx_rows.push(gids);
                }
                let q_all = Tensor::cat(&q_rows, 0)?; // [s, H, hd]
                let kv_all = Tensor::cat(&kv_rows, 0)?; // [s, hd]
                let dev = q_all.device();
                // Gather ONLY the union of selected entries onto the GPU (tier-
                // aware — works when the gallery has spilled to CPU RAM past
                // HOT_ENTRY_CAP), then remap each query's absolute GIDs to their
                // dense index in that compacted set. `comp_idx` indexes the small
                // union, so the launch never needs the whole (possibly-spilled)
                // corpus resident.
                let mut union: Vec<u32> = idx_rows.iter().flatten().copied().collect();
                union.sort_unstable();
                union.dedup();
                let cache = match entry.layers[l].gallery.as_ref() {
                    Some(g) if !union.is_empty() => {
                        let ids = Tensor::from_vec(union.clone(), union.len(), dev)?;
                        let (ni8, nsc, rbf, cpos) = g.gather_corpus(&ids)?;
                        super::paged::CorpusCache::from_gathered(ni8, nsc, rbf, cpos, union.len())?
                    }
                    _ => st.empty_corpus_cache()?,
                };
                let remap: HashMap<u32, u32> = union
                    .iter()
                    .enumerate()
                    .map(|(i, &g)| (g, i as u32))
                    .collect();
                let max_sel = idx_rows.iter().map(|v| v.len()).max().unwrap_or(0).max(1);
                let mut idx_flat = vec![u32::MAX; s_len * max_sel];
                let mut cnt_v = vec![0u32; s_len];
                for (t, gids) in idx_rows.iter().enumerate() {
                    // Contract: each row's compressed indices are strictly
                    // ascending (the gallery returns ascending GIDs and the union
                    // is sorted, so the remap is monotonic). The attention is
                    // order-independent, but callers must uphold this.
                    let mut prev: i64 = -1;
                    for (j, &g) in gids.iter().enumerate() {
                        let mapped = remap[&g];
                        debug_assert!(
                            (mapped as i64) > prev,
                            "comp_idx row {t} not strictly ascending: {mapped} after {prev}"
                        );
                        prev = mapped as i64;
                        idx_flat[t * max_sel + j] = mapped;
                    }
                    cnt_v[t] = gids.len() as u32;
                }
                let comp_idx = Tensor::from_vec(idx_flat, (s_len, max_sel), dev)?;
                let comp_cnt = Tensor::from_vec(cnt_v, s_len, dev)?;
                let q_pos = Tensor::from_vec(
                    (base as u32..(base + s_len) as u32).collect::<Vec<_>>(),
                    s_len,
                    dev,
                )?;
                let out = super::paged::paged_latent_prefill_raw(
                    &q_all,
                    hdr_of(l, decode_seqs.len() + pi),
                    &q_pos,
                    Some((&kv_all, base)),
                    &cache,
                    &comp_idx,
                    &comp_cnt,
                    st.sinks(),
                    st.rope_tab(),
                    a.softmax_scale() as f32,
                    a.window_size(),
                    0,
                    session.backings()[l].k_format().to_tag(),
                    st.ws(),
                )?;
                // Write the prompt latents into the arena so FUTURE decode waves
                // read them (this launch read the fresh bf16 diagonal, not the
                // arena). K≡V single latent → k = v.
                let kv_4d = kv_all.reshape((1, 1, s_len, a.head_dim()))?;
                session.backings()[l].write_contiguous(seq, base, &kv_4d, &kv_4d)?;
                session.backings()[l].set_len(seq, base + s_len);
                let o = out
                    .to_dtype(DType::F32)?
                    .reshape((1, s_len, a.n_heads(), a.head_dim()))?
                    .transpose(1, 2)?
                    .contiguous()?;
                attn_rows.push(a.output_proj(&o, 1, s_len)?);
                if l + 1 == n_layers {
                    entry.absorbed = base + s_len;
                }
                row_cursor += s_len;
            }
            if !prefill_seqs.is_empty() {
                pipeline_record("deepseek:prefill_attn", t_prefill);
            }
            // Phase D — glue attention: each glue row attends its causal
            // window at its TRUE position, keys read from the arena (its own
            // island included — scattered in phase A). Compressed selection
            // and the compression-seam fold are step-7 scope; the corpus
            // state does not absorb glue tokens.
            let t_glue_attn = profile_now();
            for (gi, _seq) in glue_seqs.iter().enumerate() {
                let g_len = glue_lens[gi];
                let (xs, qr) = &glue_proj[gi];
                let q = a
                    .wq_b()
                    .forward(qr)?
                    .reshape((1, g_len, a.n_heads(), a.head_dim()))?;
                let q = rms_scale(&q, a.eps())?;
                let q_bf = q
                    .reshape((g_len, a.n_heads(), a.head_dim()))?
                    .to_dtype(DType::BF16)?;
                let dev = xs.device();
                let q_pos_t = Tensor::from_vec(glue_pos[gi].clone(), g_len, dev)?;
                let empty_idx = Tensor::full(u32::MAX, (g_len, 1), dev)?;
                let empty_cnt = Tensor::zeros(g_len, DType::U32, dev)?;
                let out = super::paged::paged_latent_prefill_raw(
                    &q_bf,
                    hdr_of(l, decode_seqs.len() + prefill_seqs.len() + gi),
                    &q_pos_t,
                    None,
                    &st.empty_corpus_cache()?,
                    &empty_idx,
                    &empty_cnt,
                    st.sinks(),
                    st.rope_tab(),
                    a.softmax_scale() as f32,
                    a.window_size(),
                    1,
                    session.backings()[l].k_format().to_tag(),
                    st.ws(),
                )?;
                let o = out
                    .to_dtype(DType::F32)?
                    .reshape((1, g_len, a.n_heads(), a.head_dim()))?
                    .transpose(1, 2)?
                    .contiguous()?;
                attn_rows.push(a.output_proj(&o, 1, g_len)?);
            }
            if !glue_seqs.is_empty() {
                pipeline_record("deepseek:glue_attn", t_glue_attn);
            }
            let t_hc_post = profile_now();
            let x = Tensor::cat(&attn_rows, 1)?; // [1, rows, dim]
            let h1 = hc.post(&x, &h, &post, &comb)?;
            pipeline_record("deepseek:hc_post_attn", t_hc_post);

            // MoE sub-block: one batched call — a single routing readback per
            // layer per wave, amortized over every row (the expert ids must be
            // host-visible to schedule the streaming cache's pinned→VRAM
            // uploads; it reaches zero only under full residency).
            let t_moe = profile_now();
            let (x, post, comb) = hc.pre(&h1, &layer.hc_ffn)?;
            let mut flat_ids: Vec<u32> = decode_ids.clone();
            for ids in prefill_ids.iter().chain(&glue_ids) {
                flat_ids.extend_from_slice(ids);
            }
            let moe = e.moe_forward_batch(layer, &x, &flat_ids)?;
            h = hc.post(&moe, &h1, &post, &comb)?;
            pipeline_record("deepseek:moe", t_moe);
        }
        drop(state);
        drop(generation);

        if layer_end < n_layers {
            // Pause: persist the mHC stream flattened to a plain 3-D hidden
            // shape (opaque to the scheduler).
            let flat = h.reshape((1, total_rows, cfg.hc_mult * cfg.dim))?;
            return Ok(WaveStep {
                residual: Some(flat),
                logits: None,
            });
        }

        // Head: decode rows + each prefill sequence's LAST row.
        let t_head = profile_now();
        let reduced = hc.head_reduce(&h, e.hc_head())?; // [1, rows, dim]
        let normed = rms_norm(&reduced, e.output_norm(), cfg.norm_eps)?;
        let mut logits_rows: Vec<Tensor> = Vec::with_capacity(all_seqs.len());
        for i in 0..decode_seqs.len() {
            let row = normed.narrow(1, i, 1)?;
            logits_rows.push(e.lm_head().forward(&row)?.reshape((1, cfg.vocab_size))?);
        }
        let mut cursor = decode_seqs.len();
        for &s_len in &prefill_lens {
            let row = normed.narrow(1, cursor + s_len - 1, 1)?;
            logits_rows.push(e.lm_head().forward(&row)?.reshape((1, cfg.vocab_size))?);
            cursor += s_len;
        }
        pipeline_record("deepseek:head_lm", t_head);
        Ok(WaveStep {
            residual: None,
            logits: Some(logits_rows),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::Int8Mode;
    use candle::IndexOp;

    /// Multi-session concurrent batched forwarding — the throughput + coherence
    /// gate the wave architecture exists to serve, on the SAME shared
    /// `TestParams::run` / StoryRewrite harness the Qwen/Llama batched models
    /// use (`quantized_qwen3_moe::test_parallel_batched_forwarding`). N sessions
    /// are given the same story with a per-session name to substitute; they are
    /// ragged-batch-prefilled and decoded **batched, one wave per step**, and
    /// each session must reproduce ITS OWN name-substituted story (the harness's
    /// common-prefix reproduction check + adjacent-session distinctness — the
    /// strongest cross-session-bleed / GID-collision / decode-desync detector).
    ///
    /// DeepSeek ALWAYS thinks, so every reply opens with a `<think>…</think>`
    /// block; `with_suppress_thinking(true)` makes the harness strip it
    /// (`strip_thinking_blocks`) before the reproduction match — the model's
    /// dialect can't suppress the block itself, but the validator ignores it.
    /// The per-phase `forward_wave` profile (this file's `pipeline_record`
    /// marks) rides the harness's profile snapshot under `--features profile`.
    ///
    /// Run ONLY this one (the bare name matches every model's copy → cargo
    /// would run all six concurrently and OOM the pinned pools):
    ///   cargo test --release --features cuda,profile --lib \
    ///     deepseek4::wave::tests::test_parallel_batched_forwarding \
    ///     -- --ignored --nocapture --test-threads 1
    #[test]
    #[ignore]
    fn test_parallel_batched_forwarding() -> Result<()> {
        use crate::models::batch_test::utils::{TestConfig, TestMode, TestParams};
        use crate::models::batched_inference::InferenceMode;
        use crate::models::dialect::Dialect;

        let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer_json = std::fs::read_to_string(&tok_path)
            .map_err(|e| candle::Error::msg(format!("read tokenizer.json: {e}")))?;
        let eos = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer: {e}")))?
            .token_to_id("<｜end▁of▁sentence｜>")
            .expect("deepseek eos id");

        // The story is long, DeepSeek thinks before reproducing it, and it is a
        // 284B model over per-token prefill — so a couple of modest batch widths
        // keep the run inside the harness timeout while still exercising genuine
        // concurrency. The `InferenceMode` is cosmetic: `create_batched_session`
        // forces DeepSeek's single-latent FP8 arena regardless.
        let params = TestParams::new(64, &tokenizer_json, Dialect::deepseek())
            .map_err(|e| candle::Error::msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true) // strip <think>…</think> before validation
            .with_stop_on_eos(vec![eos])
            .with_print_outputs(true)
            .with_timeout_secs(1800);

        let configs = [1usize, 4, 8]
            .into_iter()
            .map(|n| TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: n,
                num_repeats: 1,
                generate_max_len: 64,
                test_mode: Some(TestMode::StoryRewrite),
            })
            .collect::<Vec<_>>();

        let load_model = || {
            let engine = Dsv4Engine::load(&path, &device, Int8Mode::Performance)?;
            DeepSeekBatched::new(engine)
        };
        params.run(configs, load_model)
    }

    /// Step-5 pre-integration gate: the wave path (batched prefill + decode
    /// through `ManagedBatchedModel::forward_wave`) answers "Paris" — the same
    /// rung-3 semantic gate the per-token kernel engine passed, now through
    /// the scheduler-facing trait surface. Ignored (merged file + CUDA).
    #[test]
    #[ignore]
    fn wave_paris() -> Result<()> {
        let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let engine = Dsv4Engine::load(&path, &device, Int8Mode::Performance)?;
        let model = DeepSeekBatched::new(engine)?;

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

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let n_layers = model.num_layers();
        super::super::readback::reset_readbacks();

        // Prefill the whole prompt in one wave.
        let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
        let t0 = std::time::Instant::now();
        let step = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            std::slice::from_ref(&prompt_t),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        eprintln!(
            "[wave] prefill {} tokens in {:.1}s",
            ids.len(),
            t0.elapsed().as_secs_f32()
        );
        session.advance_sequence(seq, ids.len())?;
        let logits = step
            .logits
            .ok_or_else(|| candle::Error::msg("prefill wave produced no logits"))?;
        let mut next = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;

        // Greedy decode until EOS (bounded) through decode waves. STRICT gate:
        // the answer must be exactly "Paris" followed by EOS — the same stream
        // the per-token kernel engine produces; anything longer is wave-path
        // degradation, not a style choice.
        let eos = tokenizer
            .token_to_id("<｜end▁of▁sentence｜>")
            .expect("eos id");
        let mut gen = vec![next];
        let mut decode_waves = 0usize;
        let t1 = std::time::Instant::now();
        while gen.len() < 12 && next != eos {
            let tok = Tensor::from_vec(vec![next], (1, 1), &Device::Cpu)?;
            let step = model.forward_wave(
                &mut session,
                &[seq],
                std::slice::from_ref(&tok),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, 1)?;
            decode_waves += 1;
            let logits = step
                .logits
                .ok_or_else(|| candle::Error::msg("decode wave produced no logits"))?;
            next = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;
            gen.push(next);
        }
        let dt = t1.elapsed().as_secs_f32();
        let text = tokenizer
            .decode(&gen, false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        eprintln!("[wave] generated ids={gen:?}");
        eprintln!("[wave] continuation={text:?}");
        eprintln!(
            "[wave] {decode_waves} decode waves in {dt:.1}s = {:.2} tok/s",
            decode_waves as f32 / dt
        );
        assert_eq!(
            *gen.last().unwrap(),
            eos,
            "wave path did not stop on EOS within 12 tokens: {text:?}"
        );
        let answer = tokenizer
            .decode(&gen[..gen.len() - 1], false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        assert_eq!(
            answer.trim(),
            "Paris",
            "wave path must answer exactly \"Paris\": {text:?}"
        );

        // Readback budget: the wave path's ONLY device→host transfers are the
        // per-layer MoE routing reads (one per wave per layer — intrinsic to
        // the streaming expert cache, amortized across the whole wave).
        // Sampling (`to_scalar` above) is the one-per-token the budget allows
        // and belongs to the caller.
        let expected = (1 + decode_waves) * n_layers;
        let got = super::super::readback::readback_count();
        assert_eq!(
            got, expected,
            "wave-path readbacks beyond the documented MoE-routing set: \
             {got} vs {expected}"
        );
        Ok(())
    }

    /// Wave-path bisect: same conversation prompt, but absorbed through the
    /// wave DECODE path one token at a time (no batched prefill) — the exact
    /// per-token regime the KernelSession engine proved crisp ("Paris"+EOS)
    /// on the same KO weights. Crisp here → the batched prefill
    /// (`prefill_layer_rows`/`write_contiguous`/`set_len`) corrupts state;
    /// junk here → the wave decode state containers themselves diverge.
    #[test]
    #[ignore]
    fn wave_paris_decode_only_prefill() -> Result<()> {
        let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let engine = Dsv4Engine::load(&path, &device, Int8Mode::Performance)?;
        let model = DeepSeekBatched::new(engine)?;

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
        let eos = tokenizer
            .token_to_id("<｜end▁of▁sentence｜>")
            .expect("eos id");

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let n_layers = model.num_layers();

        let mut step_row = |tok: u32| -> Result<u32> {
            let t = Tensor::from_vec(vec![tok], (1, 1), &Device::Cpu)?;
            let step = model.forward_wave(
                &mut session,
                &[seq],
                std::slice::from_ref(&t),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, 1)?;
            let logits = step
                .logits
                .ok_or_else(|| candle::Error::msg("decode wave produced no logits"))?;
            logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()
        };

        // Absorb the prompt through decode steps (per-token regime).
        let mut next = 0u32;
        for &t in &ids {
            next = step_row(t)?;
        }
        let mut gen = vec![next];
        while gen.len() < 16 && next != eos {
            next = step_row(next)?;
            gen.push(next);
        }
        let text = tokenizer
            .decode(&gen, false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        eprintln!("[wave-bisect] generated ids={gen:?}");
        eprintln!("[wave-bisect] continuation={text:?}");
        eprintln!("[wave-bisect] crisp={}", gen.len() >= 2 && gen[1] == eos);
        Ok(())
    }

    /// Layer-resolved divergence probe: run BOTH absorption paths segmented
    /// one layer at a time (residual pause/resume), capturing the residual at
    /// every layer boundary. The first layer whose residual diverges between
    /// batched prefill and per-token decode names the defective computation.
    /// Also prints each layer's kind + gallery size for the interpretation.
    #[test]
    #[ignore]
    fn wave_prefill_residual_divergence() -> Result<()> {
        let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let engine = Dsv4Engine::load(&path, &device, Int8Mode::Performance)?;
        let model = DeepSeekBatched::new(engine)?;

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
        let n = ids.len();
        let n_layers = model.num_layers();

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq_a = session.create_sequence()?; // per-token decode absorb
        let seq_b = session.create_sequence()?; // batched prefill

        // A: per-token, layer-segmented; keep residual after each layer for
        // every token: res_a[l] = [n] rows.
        let mut res_a: Vec<Vec<Tensor>> = vec![Vec::with_capacity(n); n_layers];
        for &t in &ids {
            let tt = Tensor::from_vec(vec![t], (1, 1), &Device::Cpu)?;
            let mut resid: Option<Tensor> = None;
            for l in 0..n_layers {
                let step = model.forward_wave(
                    &mut session,
                    &[seq_a],
                    std::slice::from_ref(&tt),
                    &[],
                    &[],
                    &[],
                    &[],
                    l,
                    l + 1,
                    resid.take(),
                )?;
                match step.residual {
                    Some(r) => {
                        res_a[l].push(r.clone());
                        resid = Some(r);
                    }
                    None => {
                        // final layer returns logits; no residual to store
                    }
                }
            }
            session.advance_sequence(seq_a, 1)?;
        }

        // B: batched prefill, layer-segmented; res_b[l] = [1, n, hc*dim].
        let pt = Tensor::from_vec(ids.clone(), (1, n), &Device::Cpu)?;
        let mut res_b: Vec<Tensor> = Vec::with_capacity(n_layers);
        let mut resid: Option<Tensor> = None;
        for l in 0..n_layers {
            let step = model.forward_wave(
                &mut session,
                &[],
                &[],
                &[seq_b],
                std::slice::from_ref(&pt),
                &[],
                &[],
                l,
                l + 1,
                resid.take(),
            )?;
            if let Some(r) = step.residual {
                res_b.push(r.clone());
                resid = Some(r);
            }
        }
        session.advance_sequence(seq_b, n)?;

        // Compare per layer: prefill row t vs decode step t residual.
        let state = model.seq_state.read().unwrap();
        let ea = state.get(&seq_a).expect("seq_a state");
        for l in 0..res_b.len() {
            let rb = &res_b[l]; // [1, n, hcdim]
            let hcdim = rb.dim(2)?;
            let mut worst = 0f32;
            let mut worst_t = 0usize;
            for t in 0..n {
                let a_row = res_a[l][t].reshape(hcdim)?.to_dtype(DType::F32)?;
                let b_row = rb.i((0, t))?.reshape(hcdim)?.to_dtype(DType::F32)?;
                let d = (a_row - b_row)?.abs()?.max_all()?.to_scalar::<f32>()?;
                if d > worst {
                    worst = d;
                    worst_t = t;
                }
            }
            let glen = ea.layers[l].gallery.as_ref().map_or(0, |g| g.len());
            let kind = {
                let e = model.engine.engine_layer(l);
                format!("{:?}", e.attn.kind())
            };
            eprintln!(
                "[resid-div] layer {l:2} ({kind:>13}, gallery {glen:2}): max|Δresidual| = {worst:.6} (worst token {worst_t})"
            );
        }
        Ok(())
    }

    /// Prefill-vs-decode state audit: absorb the SAME prompt into two
    /// sequences — one through per-token decode waves (proven crisp), one
    /// through the batched prefill — then diff every layer's arena window
    /// bytes, corpus state, and finally the next decode step's logits. The
    /// first divergent artifact is the batched-prefill bug.
    #[test]
    #[ignore]
    fn wave_prefill_state_matches_decode_steps() -> Result<()> {
        let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let engine = Dsv4Engine::load(&path, &device, Int8Mode::Performance)?;
        let model = DeepSeekBatched::new(engine)?;

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
        let n = ids.len();

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq_a = session.create_sequence()?; // decode-step absorb (reference)
        let seq_b = session.create_sequence()?; // batched prefill (suspect)
        let n_layers = model.num_layers();

        // A: per-token decode waves.
        for &t in &ids {
            let tt = Tensor::from_vec(vec![t], (1, 1), &Device::Cpu)?;
            model.forward_wave(
                &mut session,
                &[seq_a],
                std::slice::from_ref(&tt),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq_a, 1)?;
        }
        // B: one batched prefill wave.
        let pt = Tensor::from_vec(ids.clone(), (1, n), &Device::Cpu)?;
        let step_b = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq_b],
            std::slice::from_ref(&pt),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        session.advance_sequence(seq_b, n)?;
        let logits_b = step_b.logits.expect("prefill logits")[0].clone();
        let next_b = logits_b.i(0)?.argmax(0)?.to_scalar::<u32>()?;
        eprintln!("[state-diff] prefill argmax token={next_b}");

        // Window bytes per layer. The decode-absorbed slot's HOST population
        // count lags one token (the final token was written in-kernel with no
        // host set_len after it), so compare the fully-covered [0, n-1).
        let cmp_len = n - 1;
        for l in 0..n_layers {
            let b = &session.backings()[l];
            let (ka, _) = b.read_contiguous(seq_a, 0, cmp_len)?;
            let (kb, _) = b.read_contiguous(seq_b, 0, cmp_len)?;
            let d = (ka.to_dtype(DType::F32)? - kb.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            if d != 0.0 {
                eprintln!("[state-diff] layer {l}: window max|Δ| = {d}");
            }
        }
        // Corpus state per layer.
        {
            let state = model.seq_state.read().unwrap();
            let ea = state.get(&seq_a).expect("seq_a state");
            let eb = state.get(&seq_b).expect("seq_b state");
            for l in 0..n_layers {
                let ga = ea.layers[l].gallery.as_ref();
                let gb = eb.layers[l].gallery.as_ref();
                let (la, lb) = (ga.map_or(0, |g| g.len()), gb.map_or(0, |g| g.len()));
                if la != lb {
                    eprintln!("[state-diff] layer {l}: gallery len {la} vs {lb}");
                    continue;
                }
                if let (Some(ga), Some(gb)) = (ga, gb) {
                    if la > 0 {
                        let ea_t = ga.attn_entries()?.to_dtype(DType::F32)?;
                        let eb_t = gb.attn_entries()?.to_dtype(DType::F32)?;
                        let d = (&ea_t - &eb_t)?.abs()?.max_all()?.to_scalar::<f32>()?;
                        if d != 0.0 {
                            let na = ea_t.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                            let nb = eb_t.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                            let va: Vec<f32> = ea_t.i((0, 0..4))?.to_vec1::<f32>()?;
                            let vb: Vec<f32> = eb_t.i((0, 0..4))?.to_vec1::<f32>()?;
                            eprintln!(
                                "[state-diff] layer {l}: gallery entries max|Δ| = {d} \
                                 |A|={na:.4} |B|={nb:.4} A[0][..4]={va:?} B[0][..4]={vb:?}"
                            );
                        }
                        let pa = ga.positions()?.to_vec1::<u32>()?;
                        let pb = gb.positions()?.to_vec1::<u32>()?;
                        if pa != pb {
                            eprintln!("[state-diff] layer {l}: gallery positions {pa:?} vs {pb:?}");
                        }
                    }
                }
            }
        }

        // Next decode step on both with the same token.
        let tok = Tensor::from_vec(vec![next_b], (1, 1), &Device::Cpu)?;
        let toks = [tok.clone(), tok];
        let step = model.forward_wave(
            &mut session,
            &[seq_a, seq_b],
            &toks,
            &[],
            &[],
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        let logits = step.logits.expect("decode logits");
        let arg_a = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;
        let arg_b = logits[1].i(0)?.argmax(0)?.to_scalar::<u32>()?;
        let dl = (logits[0].to_dtype(DType::F32)? - logits[1].to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        eprintln!("[state-diff] next-step argmax: decode-absorbed={arg_a} prefill-absorbed={arg_b} max|Δlogits|={dl}");
        assert_eq!(
            arg_a, arg_b,
            "decode step after prefill diverges from decode step after per-token absorb"
        );
        Ok(())
    }

    /// Template-distribution oracle for the rung-4 conversation gate: the
    /// EXACT token stream the conversation engine assembles (BOS + system
    /// text + `<｜User｜>` + question + `<｜Assistant｜>`), fed as ONE plain
    /// contiguous prefill through the reference wave path, greedy-decoded to
    /// EOS. What this prints is what the conversation engine MUST reproduce:
    /// if this answers a crisp "Paris" while the conversation engine rolls
    /// on, the divergence is in the glue/position wiring, not the prompt.
    #[test]
    #[ignore]
    fn wave_paris_conversation_prompt() -> Result<()> {
        let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let engine = Dsv4Engine::load(&path, &device, Int8Mode::Performance)?;
        let model = DeepSeekBatched::new(engine)?;

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
        eprintln!("[wave-conv] prompt ids={ids:?}");
        let eos = tokenizer
            .token_to_id("<｜end▁of▁sentence｜>")
            .expect("eos id");

        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let seq = session.create_sequence()?;
        let n_layers = model.num_layers();

        let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
        let step = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            std::slice::from_ref(&prompt_t),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        session.advance_sequence(seq, ids.len())?;
        let logits = step
            .logits
            .ok_or_else(|| candle::Error::msg("prefill wave produced no logits"))?;
        let mut next = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;

        let mut gen = vec![next];
        while gen.len() < 24 && next != eos {
            let tok = Tensor::from_vec(vec![next], (1, 1), &Device::Cpu)?;
            let step = model.forward_wave(
                &mut session,
                &[seq],
                std::slice::from_ref(&tok),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, 1)?;
            let logits = step
                .logits
                .ok_or_else(|| candle::Error::msg("decode wave produced no logits"))?;
            next = logits[0].i(0)?.argmax(0)?.to_scalar::<u32>()?;
            gen.push(next);
        }
        let text = tokenizer
            .decode(&gen, false)
            .map_err(|e| candle::Error::msg(format!("decode: {e}")))?;
        eprintln!("[wave-conv] generated ids={gen:?}");
        eprintln!("[wave-conv] continuation={text:?}");
        eprintln!(
            "[wave-conv] stopped_on_eos={} after {} tokens",
            next == eos,
            gen.len()
        );
        Ok(())
    }
}
