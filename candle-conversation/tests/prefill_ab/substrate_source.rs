//! Real-substrate prefix source for the prefill kernel tests.
//!
//! Loads sealed turns straight out of a substrate redo log — the *actual
//! production KV bytes*, with their real per-block format selections, real
//! palette maps, real partial-chunk layouts, and real R16 Q-capture blocks —
//! and injects one layer's chunks into a test backing as the prefill prefix.
//! No model, no scheduler, no projection: `SubstratePersistence` +
//! `load_turn_into_hot` only.
//!
//! There is no CPU golden for these cases (the pre-quantization source
//! values no longer exist); their role is determinism + sanity over
//! kernel backends over byte-identical arenas, plus structural sanity
//! (finite output, determinism across a reset-and-rerun).
//!
//! Gated behind the `ZEN_PREFILL_AB_SUBSTRATE` env var — the **workspace**
//! directory (the parent that CONTAINS `.substrate/`; `open_in_with_
//! substrate` appends `.substrate/substrate.log` itself, and will silently
//! create an empty log if handed the `.substrate` dir directly). The tests
//! are `#[ignore]`d so CI without a substrate skips them:
//!
//! ```text
//! ZEN_PREFILL_AB_SUBSTRATE=path/to/workspace \
//!   cargo test -p candle-conversation --test prefill_ab -- --ignored substrate --nocapture
//! ```

use crate::harness::{
    host_to_flat, BuiltCase, Rng, Scenario, Segment, SeqSpec, HEAD_DIM, N_HEAD, N_KV_HEAD,
};
use candle::{DType, Device, Result, Tensor};
use candle_conversation::persistence::cold_load::ColdLoadStager;
use candle_conversation::persistence::resume::recovered_turn_decls;
use candle_conversation::persistence::transfer::load_turn_into_hot;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::substrate::Substrate;
use candle_nn::kv_cache::ChunkedKvBacking;
use candle_transformers::models::prefill_utils::compute_rope_cs;

/// Directory containing `substrate.log`, or `None` to skip.
pub fn substrate_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("ZEN_PREFILL_AB_SUBSTRATE").map(std::path::PathBuf::from)
}

/// A turn recovered into per-layer backings: `backings[l]` holds layer `l`'s
/// sealed chunks attached to slot 0, `prefix_len` tokens each.
pub struct RecoveredPrefix {
    pub backings: Vec<ChunkedKvBacking>,
    pub prefix_len: usize,
    pub n_layers: usize,
}

/// Open the log, pick the turn with the largest per-layer chunk span (the
/// longest real prefix available), and load every layer hot.
pub fn load_largest_turn(dir: &std::path::Path, device: &Device) -> Result<RecoveredPrefix> {
    let mut substrate = Substrate::new();
    let mut persistence = SubstratePersistence::open_in_with_substrate(dir, &mut substrate)
        .map_err(|e| candle::Error::Msg(format!("open substrate: {e}")))?;

    let decls = recovered_turn_decls(&substrate);
    let decl = decls
        .iter()
        .max_by_key(|d| d.block_end.saturating_sub(d.block_start))
        .ok_or_else(|| candle::Error::Msg("substrate has no recovered turns".into()))?
        .clone();
    let chunks_per_layer = (decl.block_end - decl.block_start) as usize;
    if chunks_per_layer == 0 {
        candle::bail!("largest recovered turn has no chunks");
    }

    let stream_id = candle_conversation::persistence::content_hash::turn_stream_id(
        decl.timeline_id,
        decl.turn_index,
    );
    let total_chunks = substrate
        .stream_of(stream_id)
        .map(|s| s.chunks.len())
        .unwrap_or(0);
    let n_layers = total_chunks / chunks_per_layer.max(1);
    if n_layers == 0 {
        candle::bail!("turn stream {stream_id:?} has no chunk records");
    }

    // One small backing per layer: the turn's chunks + headroom for the
    // fresh q tokens the prefill writes. Slot 0 = the sequence, slot 1
    // spare (unused; keeps parity with the synthetic harness layout).
    let max_blocks = chunks_per_layer + 8;
    let mut backings = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        backings.push(ChunkedKvBacking::new(
            2,
            N_KV_HEAD,
            HEAD_DIM,
            DType::F16,
            device,
            max_blocks,
        )?);
    }

    let mut stager = ColdLoadStager::new();
    let sealed = load_turn_into_hot(
        &backings,
        device,
        &mut persistence,
        &substrate,
        &decl,
        &mut stager,
    )
    .map_err(|e| candle::Error::Msg(format!("load_turn_into_hot: {e}")))?;

    let mut prefix_len = 0usize;
    for (li, seq) in sealed.iter().enumerate() {
        let tokens: usize = seq.chunks.iter().map(|c| c.token_count as usize).sum();
        if li == 0 {
            prefix_len = tokens;
        } else if tokens != prefix_len {
            candle::bail!("layer {li} token count {tokens} != layer 0 {prefix_len}");
        }
        // Allocating slot 0 (what a cache bind does in production) must
        // precede the inject — `load_stream_into_hot` freed its scratch slot.
        let slot = backings[li].alloc_sequence()?;
        if slot != 0 {
            candle::bail!("expected fresh backing to allocate slot 0, got {slot}");
        }
        backings[li].inject_sealed_at_tail(0, seq)?;
    }
    if prefix_len == 0 {
        candle::bail!("recovered turn has zero tokens");
    }

    Ok(RecoveredPrefix {
        backings,
        prefix_len,
        n_layers,
    })
}

/// Wrap one recovered layer as a harness `BuiltCase` with seeded synthetic
/// Q/K/V for the fresh tokens. `prefix_k`/`prefix_v` stay EMPTY — there is
/// no pre-quantization source for real chunks, so `golden::golden` must not
/// be called on these cases (determinism over real bytes is the check).
pub fn case_for_layer(
    rec: &RecoveredPrefix,
    layer: usize,
    q_len: usize,
    seed: u64,
    device: &Device,
) -> Result<BuiltCase> {
    let backing = rec.backings[layer].clone();
    let mut cache = crate::harness::bind_cache(&backing, 0)?;
    cache.set_current_seq_len(rec.prefix_len)?;
    let prefix_chunks = cache
        .k_cache()
        .chunked_live_chunks_as_sealed()
        .map(|c| c.len())
        .unwrap_or(0);

    // Production-like RoPE table sized to cover prefix + q.
    let theta = 1e6f64;
    let inv_freq: Vec<f32> = (0..HEAD_DIM / 2)
        .map(|i| (1.0 / theta.powf(2.0 * i as f64 / HEAD_DIM as f64)) as f32)
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, HEAD_DIM / 2, device)?;
    let blocks = (rec.prefix_len + q_len).div_ceil(32) + 2;
    let rope_cs = compute_rope_cs(&inv_freq, blocks, HEAD_DIM, device)?;
    let rope_cs_host = rope_cs
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let rope_offsets = Tensor::zeros(1, DType::U32, device)?;

    let mut rng = Rng::new(seed);
    let spec = Scenario {
        name: "substrate_layer",
        seqs: vec![SeqSpec {
            segments: vec![Segment {
                len: rec.prefix_len,
                level: None, // real chunks: level is a property of the log, not the spec
            }],
            q_len,
        }],
        theta,
        seed,
        golden_band: f32::INFINITY, // golden is not applicable to real chunks
        min_cos: -1.0,
        structured_dims: false,
    };

    let stager = match device {
        Device::Cuda(d) => candle::quantized::pinned_staging::PinnedStager::new(d),
        _ => candle::bail!("prefill_ab requires a CUDA device"),
    };
    let new_q = vec![rng.fill(q_len * N_HEAD * HEAD_DIM)];
    let new_k = vec![rng.fill(q_len * N_KV_HEAD * HEAD_DIM)];
    let new_v = vec![rng.fill(q_len * N_KV_HEAD * HEAD_DIM)];
    let q_dev = host_to_flat(&new_q[0], N_HEAD, q_len, device)?;
    let k_dev = host_to_flat(&new_k[0], N_KV_HEAD, q_len, device)?;
    let v_dev = host_to_flat(&new_v[0], N_KV_HEAD, q_len, device)?;
    let cu_seqlens_q = Tensor::from_vec(vec![0u32, q_len as u32], 2, device)?;
    let q_lens_dev = Tensor::from_vec(vec![q_len as u32], 1, device)?;
    let kv_lens_dev = Tensor::from_vec(vec![(rec.prefix_len + q_len) as u32], 1, device)?;
    Ok(BuiltCase {
        backing,
        caches: vec![cache],
        prefix_chunks: vec![prefix_chunks],
        stager,
        prefix_k: vec![Vec::new()],
        prefix_v: vec![Vec::new()],
        new_q,
        new_k,
        new_v,
        rope_cs_host,
        rope_cs,
        rope_offsets,
        q_dev,
        k_dev,
        v_dev,
        cu_seqlens_q,
        q_lens_dev,
        kv_lens_dev,
        spec,
    })
}
