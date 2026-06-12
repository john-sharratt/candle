//! Quantize-on-evict — adaptive per-`(head, palette)` compression at the
//! hot→warm tier boundary.
//!
//! The substrate's persistence thread calls [`quantize_sealed_in_place`]
//! when a `CompressionPolicy` is configured. The flow within this module is:
//!
//! ```text
//!   GPU Float / R16  ──select_palette4_formats_fused──▶  per-(h, p) format choice
//!                    ──quantize_palette4_convert_buffered──▶  GPU Quant arenas
//! ```
//!
//! The selection kernel reads the source float bytes and picks a
//! quantization format per `(chunk, head, palette)` from the policy's
//! candidate list using the production K/V error thresholds. The
//! conversion kernel writes the quantized bytes into GPU-quant arena
//! slots. The returned `SealedSequence`s still live in VRAM — the
//! persist thread does a separate format-preserving DtoH copy via
//! `migrate_sealed_to_cpu_batch_async` to produce the warm-tier image,
//! and then `install_warm_and_hot` atomically replaces hot with the new
//! Q-format GPU chunks and installs warm.
//!
//! Source-side rules:
//! - Eligible chunks must have all their (h, p) GIDs in GPU `Float` or
//!   `Quantized(R16)` arenas — the formats the selection + convert
//!   kernels read directly. Ineligible chunks (typically cold-loaded
//!   ones already in mixed Q-formats) are routed to a pass-through
//!   "preserve" bucket and merged back unchanged at the end.
//! - `head_dim` must be 128 — the palette-4 kernel is compiled for
//!   that shape only.

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::CudaStream;
#[cfg(feature = "cuda")]
use candle::quantized::cuda::{dtype_to_ggml_float, identity_pal_map_128, PalHeadDesc};
#[cfg(feature = "cuda")]
use candle::quantized::pinned_staging::PinnedBuf;
#[cfg(feature = "cuda")]
use candle::quantized::GgmlDType;
#[cfg(feature = "cuda")]
use candle::{Device, Result};

#[cfg(feature = "cuda")]
use super::arena::ArenaKey;
#[cfg(feature = "cuda")]
use super::backing::ChunkedKvBacking;
#[cfg(feature = "cuda")]
use super::compression_policy::{
    CompressionPolicy, PRODUCTION_K_QREL_HIGH_THRESHOLDS, PRODUCTION_K_QREL_LOW_THRESHOLDS,
    PRODUCTION_V_QREL_HIGH_THRESHOLDS, PRODUCTION_V_QREL_LOW_THRESHOLDS,
};
#[cfg(feature = "cuda")]
use super::gid_pool::ChunkGid;
#[cfg(feature = "cuda")]
use super::head_gids::{HeadGids, GIDS_PER_HEAD};
#[cfg(feature = "cuda")]
use super::sampled_selection::{PagedSelectionGpuInputs, SampleFormat};
#[cfg(feature = "cuda")]
use super::types::{SealedChunk, SealedSequence, CHUNK_SIZE};
#[cfg(feature = "cuda")]
use crate::kv_cache::arena_table::{ArenaLocation, N_PALETTE};
#[cfg(feature = "cuda")]
use crate::kv_cache::{KvFormat, QuantFormat};

/// Pack per-dim palette assignments (one byte per dim, value in `{0..3}`)
/// into the 32-byte packed map the decode/prefill kernels read.
///
/// Layout in `assignments`: `[chunk][head][dim]`, row-major.
/// Layout in output `head_palette_maps`: one `Vec<u8>` per chunk, each of
/// length `n_head * (head_dim/4)` bytes. Within each head's packed slice,
/// dim `d` occupies bits `2*(d%4) .. 2*(d%4)+2` of byte `d/4`.
///
/// Enforces the palette-4 invariant: each of the 4 palette slots must
/// receive exactly `head_dim / N_PALETTE` dimensions; the kernel's
/// per-palette quantization assumes that strict partition.
#[cfg(feature = "cuda")]
fn pack_head_palette_maps(
    assignments: &[u8],
    n_chunks: usize,
    n_head: usize,
    head_dim: usize,
) -> Result<Vec<Vec<u8>>> {
    let packed_bytes = head_dim / 4;
    let expected_packed_bytes = identity_pal_map_128().len();
    if head_dim % 4 != 0 || head_dim % N_PALETTE != 0 || packed_bytes != expected_packed_bytes {
        candle::bail!(
            "quantize_sealed_to_cpu: palette4 conversion requires head_dim={}, got {}",
            expected_packed_bytes * 4,
            head_dim,
        );
    }

    let expected = n_chunks
        .checked_mul(n_head)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| candle::Error::Msg("palette map shape overflow".into()))?;
    if assignments.len() != expected {
        candle::bail!(
            "palette map length mismatch: got {}, expected {} ({} dims/head × {} chunks × {} heads)",
            assignments.len(),
            expected,
            head_dim,
            n_chunks,
            n_head,
        );
    }

    let target_per_palette = head_dim / N_PALETTE;
    let mut out = Vec::with_capacity(n_chunks);
    for chunk_i in 0..n_chunks {
        let mut block_map = vec![0u8; n_head * packed_bytes];
        for h in 0..n_head {
            let base = (chunk_i * n_head + h) * head_dim;
            let mut counts = [0usize; N_PALETTE];
            for (dim_offset, &raw_slot) in assignments[base..base + head_dim].iter().enumerate() {
                let slot = (raw_slot & 0x3) as usize;
                counts[slot] += 1;
                let byte_idx = h * packed_bytes + (dim_offset / 4);
                block_map[byte_idx] |= (slot as u8) << (2 * (dim_offset % 4));
            }
            if counts.iter().any(|&count| count != target_per_palette) {
                candle::bail!(
                    "invalid palette4 routing for chunk {} head {}: {:?} (each palette must \
                     receive exactly {} dims)",
                    chunk_i,
                    h,
                    counts,
                    target_per_palette,
                );
            }
        }
        out.push(block_map);
    }
    Ok(out)
}

/// Adaptive per-(head, palette) quantization that runs entirely on GPU
/// and returns `SealedSequence`s whose chunks still live in VRAM.
///
/// Eligible chunks land in fresh GPU Q-format arenas chosen by the
/// palette-4 selection kernel. Ineligible chunks (those whose source
/// GIDs are not in GPU `Float` / `Quantized(R16)` arenas — typically
/// cold-loaded chunks already in mixed Q-formats) flow through a
/// pass-through "preserve" bucket and get merged back unchanged in
/// the output sequence, preserving original chunk order.
///
/// **No DMA to CPU.** Callers that need warm-tier storage feed the
/// returned GPU sequences to [`ChunkedKvBacking::migrate_sealed_to_cpu_batch_async`]
/// for a separate, format-preserving DtoH copy. The split keeps the
/// in-session reconcile purely on-GPU (decode reads the convert
/// kernel's output bytes directly, no intervening `kv_migrate` scatter).
#[cfg(feature = "cuda")]
pub fn quantize_sealed_in_place(
    backing: &ChunkedKvBacking,
    sequences: &[&SealedSequence],
    policy: &CompressionPolicy,
    device: &Device,
    copy_stream: &Arc<CudaStream>,
    pinned_scratch: &mut Option<PinnedBuf>,
) -> Result<Vec<SealedSequence>> {
    let _ = pinned_scratch; // kept for signature compatibility; no DtoH performed.
    if sequences.is_empty() {
        return Ok(Vec::new());
    }

    let cuda_dev = match device {
        Device::Cuda(d) => d,
        _ => candle::bail!("quantize_sealed_in_place: requires a CUDA device"),
    };

    let n_kv_head = backing.n_kv_head();
    let head_dim = backing.head_dim();
    if head_dim != identity_pal_map_128().len() * 4 {
        candle::bail!(
            "quantize_sealed_in_place: palette4 conversion requires head_dim=128, got {head_dim}"
        );
    }
    let sub_head_dim = head_dim / N_PALETTE;
    let elems_per_head = CHUNK_SIZE * sub_head_dim;

    // ── Bucket each source chunk: quantize vs preserve ──────────────
    //
    // A chunk is eligible for the fused palette4 kernel when every
    // (h, p) source GID lives in a GPU `Float` or `Quantized(R16)`
    // arena — the formats the selection + convert kernels read
    // directly. Full and partial chunks both qualify; see the inner
    // loop for the partial-tail correctness argument.
    //
    // Ineligible chunks go into the `preserve` bucket and get merged
    // back into the output sequence unchanged at the end. The case
    // that hits production: a turn's view borrows older chunks that
    // came back from cold storage in mixed Q-formats. Those GIDs
    // already carry the format/pal_map/scale state an earlier persist
    // pass selected; re-quantizing would be redundant and the
    // source-format check would reject them anyway.
    //
    // `chunk_jobs[i]` corresponds to one SealedChunk that will go
    // through the kernel. `seq_chunk_map[i] = (seq_idx, chunk_idx)`
    // lets us reassemble per-sequence outputs in the original chunk
    // order at the end.
    let mut chunk_jobs: Vec<HeadGids> = Vec::new();
    let mut seq_chunk_map: Vec<(usize, usize)> = Vec::new();
    let mut source_seq_chunks: Vec<&SealedChunk> = Vec::new();
    // Per-sequence preserve list, recorded as `(chunk_idx_within_seq,
    // chunk_clone)` so we can rebuild the original chunk order at merge
    // time. Preserve-bucket chunks are passed through unchanged — same
    // GIDs, same arenas, no re-quantization.
    let mut preserve_per_seq: Vec<Vec<(usize, SealedChunk)>> =
        sequences.iter().map(|_| Vec::new()).collect();
    backing.inner.storage.read(|storage| {
        for (seq_idx, seq) in sequences.iter().enumerate() {
            for (chunk_idx, chunk) in seq.chunks.iter().enumerate() {
                // Partial trailing chunks (token_count < CHUNK_SIZE) are
                // never quantized. They stay in their source float format and
                // pass through the preserve bucket unchanged: the active
                // writer chunk is GPU-float while it fills, and the persist
                // path writes the partial tail as a plain float `Chunk`
                // record (see docs/kv_tier_migration.md §3 / §5.5).
                //
                // Quantizing a partial is unsafe: the palette4 selection
                // (`select_palette4_formats_fused`) is token_count-blind — it
                // computes per-(h, p) amax over the full 32-token block. The
                // active writer chunk is allocated via `alloc_block_chunks`,
                // whose pool path does NOT zero recycled slots, so positions
                // [token_count..32] hold stale bytes from the prior tenant.
                // Feeding that garbage into amax inflates the outer scale and
                // corrupts the precision of the real tokens [0..token_count].
                //
                // A full chunk goes through the fused palette4 kernel when
                // every (h, p) source GID lives in a GPU `Float` or
                // `Quantized(R16)` arena — the formats the selection + convert
                // kernels read directly. Otherwise (e.g. a view borrowing
                // older chunks that came back from cold storage in mixed
                // Q-formats) it goes to the preserve bucket: those GIDs
                // already carry the format/pal_map/scale an earlier persist
                // pass selected, and re-quantizing would be redundant.
                let is_full = usize::from(chunk.token_count) >= CHUNK_SIZE;
                let mut all_gids_kernel_eligible = is_full;
                if all_gids_kernel_eligible {
                    for gid in chunk.gids.as_slice() {
                        let ok = matches!(
                            storage.arena_key(gid.arena_idx()),
                            Some(k) if k.location == ArenaLocation::Gpu
                                && super::chunk_ops::needs_reconcile_source_format(k.format)
                        );
                        if !ok {
                            all_gids_kernel_eligible = false;
                            break;
                        }
                    }
                }
                if all_gids_kernel_eligible {
                    chunk_jobs.push(chunk.gids.clone());
                    seq_chunk_map.push((seq_idx, chunk_idx));
                    source_seq_chunks.push(chunk);
                } else {
                    preserve_per_seq[seq_idx].push((chunk_idx, chunk.clone()));
                }
            }
        }
        Ok::<(), candle::Error>(())
    })??;
    if chunk_jobs.is_empty() {
        // Nothing eligible for the kernel — every chunk is either
        // partial or already in a non-R16 quant arena. Pass through
        // verbatim (chunks are already on GPU; we just clone the
        // SealedSequence wrappers).
        return Ok(sequences.iter().map(|s| (*s).clone()).collect());
    }

    // ── Per-(chunk, head, palette) format selection ───────────────────
    // The selection kernel runs across all chunk jobs in one launch
    // and returns:
    //   - per-(chunk, head) array of 4 palette format choices,
    //   - per-(chunk, head) array of 4 palette outer scales,
    //   - per-(chunk, head, dim) palette assignment (which slot
    //     each dim belongs to — used to derive the packed palette map).
    let (k_candidates, v_candidates) = {
        let (k_profile, v_profile) =
            super::compression_policy::production_adaptive_candidates(policy.compression_level);
        let k: Vec<SampleFormat> = k_profile
            .into_iter()
            .filter_map(SampleFormat::from_kv_format)
            .filter(|fmt| !fmt.is_float())
            .collect();
        let v: Vec<SampleFormat> = v_profile
            .into_iter()
            .filter_map(SampleFormat::from_kv_format)
            .filter(|fmt| !fmt.is_float())
            .collect();
        if k.is_empty() || v.is_empty() {
            candle::bail!(
                "quantize_sealed_in_place: missing K/V production candidates for level {}",
                policy.compression_level
            );
        }
        (k, v)
    };

    let threshold_idx = policy.compression_level.min(10) as usize;
    let k_threshold_hi =
        PRODUCTION_K_QREL_HIGH_THRESHOLDS[threshold_idx] * policy.k_hi_error_threshold_factor;
    let k_threshold_lo =
        PRODUCTION_K_QREL_LOW_THRESHOLDS[threshold_idx] * policy.k_low_error_threshold_factor;
    let v_threshold_hi =
        PRODUCTION_V_QREL_HIGH_THRESHOLDS[threshold_idx] * policy.v_hi_error_threshold_factor;
    let v_threshold_lo =
        PRODUCTION_V_QREL_LOW_THRESHOLDS[threshold_idx] * policy.v_low_error_threshold_factor;

    let stager_generation = backing.begin_stager_generation_required();
    let gpu_inputs = PagedSelectionGpuInputs::from_head_gids(
        backing.inner.clone(),
        &chunk_jobs,
        Some(&stager_generation),
        cuda_dev,
    )?;

    let (
        k_palette4_rows,
        v_palette4_rows,
        k_palette_scale_rows,
        v_palette_scale_rows,
        k_assignments,
        v_assignments,
        _k_head_amax,
        _v_head_amax,
    ) = gpu_inputs.select_palette4_formats_fused(
        &k_candidates,
        &v_candidates,
        k_threshold_hi,
        k_threshold_lo,
        v_threshold_hi,
        v_threshold_lo,
        Some(&stager_generation),
    )?;

    let to_quant = |fmt: SampleFormat| -> Result<QuantFormat> {
        fmt.to_quant_format().ok_or_else(|| {
            candle::Error::Msg(format!(
                "quantize_sealed_in_place: selection produced non-quant format {fmt}"
            ))
        })
    };

    let n_chunks = chunk_jobs.len();
    let k_palette_formats: Vec<Vec<QuantFormat>> = (0..n_chunks)
        .map(|c| {
            let start = c * n_kv_head;
            let end = start + n_kv_head;
            k_palette4_rows[start..end]
                .iter()
                .flat_map(|row| row.iter().copied())
                .map(to_quant)
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let v_palette_formats: Vec<Vec<QuantFormat>> = (0..n_chunks)
        .map(|c| {
            let start = c * n_kv_head;
            let end = start + n_kv_head;
            v_palette4_rows[start..end]
                .iter()
                .flat_map(|row| row.iter().copied())
                .map(to_quant)
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;

    let k_palette_scales: Vec<Vec<f32>> = (0..n_chunks)
        .map(|c| {
            let start = c * n_kv_head;
            let end = start + n_kv_head;
            k_palette_scale_rows[start..end]
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<f32>>()
        })
        .collect();
    let v_palette_scales: Vec<Vec<f32>> = (0..n_chunks)
        .map(|c| {
            let start = c * n_kv_head;
            let end = start + n_kv_head;
            v_palette_scale_rows[start..end]
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<f32>>()
        })
        .collect();
    let k_palette_maps = pack_head_palette_maps(&k_assignments, n_chunks, n_kv_head, head_dim)?;
    let v_palette_maps = pack_head_palette_maps(&v_assignments, n_chunks, n_kv_head, head_dim)?;

    // ── Allocate GPU-quant destination GIDs ───────────────────────────
    // One destination GID per (chunk, head, palette, K/V). The
    // candidate arenas are warm-protected at backing creation when
    // the policy is set, so `alloc_chunk_for_key` reuses them.
    //
    // When `policy.override_k_quant` is set, K dst is uniform `fmt`
    // regardless of selection's K format pick. V always uses selection's
    // per-(c, h, p) format.
    let mut new_gids_per_chunk: Vec<Vec<ChunkGid>> = Vec::with_capacity(n_chunks);
    for chunk_i in 0..n_chunks {
        let mut chunk_gids: Vec<ChunkGid> = Vec::with_capacity(n_kv_head * GIDS_PER_HEAD);
        for h in 0..n_kv_head {
            for p in 0..N_PALETTE {
                let slot = h * N_PALETTE + p;
                let k_fmt = policy
                    .override_k_quant
                    .unwrap_or(k_palette_formats[chunk_i][slot]);
                let v_fmt = policy
                    .override_v_quant
                    .unwrap_or(v_palette_formats[chunk_i][slot]);
                let k_gid = backing.alloc_chunk_for_key(ArenaKey::gpu_quant(k_fmt))?;
                let v_gid = backing.alloc_chunk_for_key(ArenaKey::gpu_quant(v_fmt))?;
                chunk_gids.push(k_gid);
                chunk_gids.push(v_gid);
            }
        }
        new_gids_per_chunk.push(chunk_gids);
    }

    // ── Build PalHeadDesc structs + launch kernel ─────────────────────
    let r16_bytes_per_head = (elems_per_head / 32) * 128; // R16 block = 32 elems × 4 bytes

    let mut descs: Vec<PalHeadDesc> = Vec::with_capacity(n_chunks * n_kv_head);

    backing.inner.storage.try_write(|storage| {
        for (chunk_i, src_gids) in chunk_jobs.iter().enumerate() {
            let ident = identity_pal_map_128();
            for h in 0..n_kv_head {
                let mut k_src_ptrs = [0u64; N_PALETTE];
                let mut k_dst_ptrs = [0u64; N_PALETTE];
                let mut k_src_fmts = [GgmlDType::F16; N_PALETTE];
                let mut k_dst_fmts = [GgmlDType::F16; N_PALETTE];
                let mut v_src_ptrs = [0u64; N_PALETTE];
                let mut v_dst_ptrs = [0u64; N_PALETTE];
                let mut v_src_fmts = [GgmlDType::F16; N_PALETTE];
                let mut v_dst_fmts = [GgmlDType::F16; N_PALETTE];

                for p in 0..N_PALETTE {
                    let gid_slot = h * GIDS_PER_HEAD + p * 2;

                    // K source.
                    let k_src = src_gids.k_gid_pal(h, p);
                    let k_src_ai = k_src.arena_idx();
                    let k_src_arena = storage.arenas().get(&k_src_ai).ok_or_else(|| {
                        candle::Error::Msg(format!("k src arena {k_src_ai} not found"))
                    })?;
                    let k_is_r16 =
                        matches!(k_src_arena.format(), KvFormat::Quantized(QuantFormat::R16));
                    k_src_fmts[p] = if k_is_r16 {
                        GgmlDType::R16
                    } else {
                        dtype_to_ggml_float(k_src_arena.float_data()?.dtype())?
                    };
                    k_src_ptrs[p] = if k_is_r16 {
                        let qt = k_src_arena.quantized_data()?;
                        ChunkedKvBacking::qtensor_ptr_at_byte_offset(
                            qt,
                            k_src.chunk_idx() * r16_bytes_per_head,
                        )?
                    } else {
                        ChunkedKvBacking::tensor_ptr_at_offset(
                            k_src_arena.float_data()?,
                            k_src.chunk_idx() * elems_per_head,
                        )?
                    };

                    // K destination — `policy.override_k_quant` decides
                    // between uniform override format and selection's
                    // per-(chunk, head, palette) pick. Must match the K
                    // arena alloc above.
                    let k_dst = &new_gids_per_chunk[chunk_i][gid_slot];
                    let k_qfmt = policy
                        .override_k_quant
                        .unwrap_or(k_palette_formats[chunk_i][h * N_PALETTE + p]);
                    let k_qdtype = k_qfmt.to_ggml_dtype();
                    let k_dst_bpc = (elems_per_head / k_qdtype.block_size()) * k_qdtype.type_size();
                    let k_dst_arena =
                        storage.arenas().get(&k_dst.arena_idx()).ok_or_else(|| {
                            candle::Error::Msg(format!(
                                "k dst arena {} not found",
                                k_dst.arena_idx()
                            ))
                        })?;
                    let k_dst_qt = k_dst_arena.quantized_data()?;
                    k_dst_fmts[p] = k_qdtype;
                    k_dst_ptrs[p] = ChunkedKvBacking::qtensor_ptr_at_byte_offset(
                        k_dst_qt,
                        k_dst.chunk_idx() * k_dst_bpc,
                    )?;

                    // V source.
                    let v_src = src_gids.v_gid_pal(h, p);
                    let v_src_ai = v_src.arena_idx();
                    let v_src_arena = storage.arenas().get(&v_src_ai).ok_or_else(|| {
                        candle::Error::Msg(format!("v src arena {v_src_ai} not found"))
                    })?;
                    let v_is_r16 =
                        matches!(v_src_arena.format(), KvFormat::Quantized(QuantFormat::R16));
                    v_src_fmts[p] = if v_is_r16 {
                        GgmlDType::R16
                    } else {
                        dtype_to_ggml_float(v_src_arena.float_data()?.dtype())?
                    };
                    v_src_ptrs[p] = if v_is_r16 {
                        let qt = v_src_arena.quantized_data()?;
                        ChunkedKvBacking::qtensor_ptr_at_byte_offset(
                            qt,
                            v_src.chunk_idx() * r16_bytes_per_head,
                        )?
                    } else {
                        ChunkedKvBacking::tensor_ptr_at_offset(
                            v_src_arena.float_data()?,
                            v_src.chunk_idx() * elems_per_head,
                        )?
                    };

                    // V destination — `policy.override_v_quant` decides
                    // between uniform override format and selection's
                    // per-(chunk, head, palette) pick. Must match the V
                    // arena alloc above.
                    let v_dst = &new_gids_per_chunk[chunk_i][gid_slot + 1];
                    let v_qfmt = policy
                        .override_v_quant
                        .unwrap_or(v_palette_formats[chunk_i][h * N_PALETTE + p]);
                    let v_qdtype = v_qfmt.to_ggml_dtype();
                    let v_dst_bpc = (elems_per_head / v_qdtype.block_size()) * v_qdtype.type_size();
                    let v_dst_arena =
                        storage.arenas().get(&v_dst.arena_idx()).ok_or_else(|| {
                            candle::Error::Msg(format!(
                                "v dst arena {} not found",
                                v_dst.arena_idx()
                            ))
                        })?;
                    let v_dst_qt = v_dst_arena.quantized_data()?;
                    v_dst_fmts[p] = v_qdtype;
                    v_dst_ptrs[p] = ChunkedKvBacking::qtensor_ptr_at_byte_offset(
                        v_dst_qt,
                        v_dst.chunk_idx() * v_dst_bpc,
                    )?;
                }

                let pal_bytes = ident.len();
                let pal_start = h * pal_bytes;
                let pal_end = pal_start + pal_bytes;
                // K and V dst pal_maps: identity when the corresponding
                // override is active, else selection's per-(chunk, head).
                let k_dst_pal_map = if policy.override_k_quant.is_some() {
                    ident
                } else {
                    let mut m = ident;
                    m.copy_from_slice(&k_palette_maps[chunk_i][pal_start..pal_end]);
                    m
                };
                let v_dst_pal_map = if policy.override_v_quant.is_some() {
                    ident
                } else {
                    let mut m = ident;
                    m.copy_from_slice(&v_palette_maps[chunk_i][pal_start..pal_end]);
                    m
                };

                let scale_start = h * N_PALETTE;
                let scale_end = scale_start + N_PALETTE;
                // K and V dst scales: unit (1.0) when the corresponding
                // override is active, else selection's per-palette.
                let k_dst_scales = if policy.override_k_quant.is_some() {
                    [1.0f32; N_PALETTE]
                } else {
                    let mut s = [1.0f32; N_PALETTE];
                    s.copy_from_slice(&k_palette_scales[chunk_i][scale_start..scale_end]);
                    s
                };
                let v_dst_scales = if policy.override_v_quant.is_some() {
                    [1.0f32; N_PALETTE]
                } else {
                    let mut s = [1.0f32; N_PALETTE];
                    s.copy_from_slice(&v_palette_scales[chunk_i][scale_start..scale_end]);
                    s
                };

                descs.push(PalHeadDesc {
                    k_src_arena_ptrs: k_src_ptrs,
                    v_src_arena_ptrs: v_src_ptrs,
                    k_src_fmts,
                    v_src_fmts,
                    k_src_pal_map: ident,
                    v_src_pal_map: ident,
                    // Source is restricted to GPU Float (F16/F32/BF16) or
                    // Quantized(R16) by the eligibility check upstream; the
                    // palette-4 kernel API takes 1.0 for any float source
                    // because those formats have no outer-scale concept.
                    k_src_scales: [1.0f32; N_PALETTE],
                    v_src_scales: [1.0f32; N_PALETTE],
                    k_dst_arena_ptrs: k_dst_ptrs,
                    v_dst_arena_ptrs: v_dst_ptrs,
                    k_dst_fmts,
                    v_dst_fmts,
                    k_dst_pal_map,
                    v_dst_pal_map,
                    k_dst_scales,
                    v_dst_scales,
                });
            }
        }
        Ok(())
    })?;

    // All kernel work runs on the primary stream — selection, alloc, and
    // the convert kernel are FIFO on a single stream, so no cross-stream
    // fence is needed. The caller is responsible for syncing the primary
    // stream before the produced GPU bytes are consumed (the persist
    // thread does this once after the per-layer loop completes; that
    // single sync also covers any subsequent DtoH on `copy_stream`,
    // which records a wait on the primary stream's event).
    let primary_stream = cuda_dev.cuda_stream();
    // One launch per chunk. The convert kernel mishandles per-chunk
    // dst-side state variance (pal_map + outer scales) when multiple
    // chunks share a launch; selection produces those per (chunk, head),
    // so we must keep each chunk in its own launch to preserve the
    // variance correctly. Layer-level FIFO ordering on the primary
    // stream means these queue cleanly behind each other.
    for chunk_i in 0..n_chunks {
        let start = chunk_i * n_kv_head;
        let end = start + n_kv_head;
        candle::quantized::cuda::quantize_palette4_convert_buffered(
            &descs[start..end],
            n_kv_head,
            1, // one chunk per launch
            1,
            &stager_generation,
            &primary_stream,
        )?;
    }
    let _ = copy_stream; // intentionally unused: convert runs on primary.
    drop(stager_generation);

    // Convert-2 (the legacy "K → uniform Q8_0 + identity layout + unit scale"
    // re-quantize pass) has been removed. Its job is now done by
    // `policy.override_k_quant` at convert-1 time when the caller opts in,
    // and `None` lets selection's full K state propagate to storage directly.

    // ── Reassemble per-sequence GPU-quant SealedSequences ────────────
    let arena_infos = backing.resolve_arena_info()?;
    // For each (seq_idx, chunk_idx_within_seq), build the new GPU
    // Q-format SealedChunk. The kernel just wrote the per-(h, p)
    // palette4 bytes into the GIDs in `new_gids_per_chunk`; we wrap
    // those plus the policy-selected palette maps + outer scales (or
    // identity + 1.0 fallbacks when the corresponding override is set).
    let mut full_quant_chunks: std::collections::HashMap<(usize, usize), SealedChunk> =
        std::collections::HashMap::with_capacity(seq_chunk_map.len());
    for (job_idx, &(seq_idx, chunk_idx)) in seq_chunk_map.iter().enumerate() {
        let new_gids = HeadGids::from_vec(new_gids_per_chunk[job_idx].clone());
        let byte_size = new_gids.arena_byte_size(&arena_infos);
        let src = source_seq_chunks[job_idx];
        // K and V pal_map / scale match what convert-1 wrote per side:
        // identity layout + empty Arc (1.0 fallback) when the corresponding
        // override is set; selection's full state otherwise. Identity bytes
        // are written explicitly per head because `gpu_chunks::rebuild_decode`
        // indexes them directly without a length-fallback check (unlike
        // `slot_state::from_sealed_chunk`).
        let pal_bytes_per_head = head_dim / 4;
        let identity_head_bytes: Option<Vec<u8>> =
            if policy.override_k_quant.is_some() || policy.override_v_quant.is_some() {
                let identity_head = identity_pal_map_128();
                let mut buf = vec![0u8; n_kv_head * pal_bytes_per_head];
                for h in 0..n_kv_head {
                    buf[h * pal_bytes_per_head..(h + 1) * pal_bytes_per_head]
                        .copy_from_slice(&identity_head);
                }
                Some(buf)
            } else {
                None
            };
        let (k_pal, k_scale): (Arc<Vec<u8>>, Arc<Vec<f32>>) = if policy.override_k_quant.is_some() {
            (
                Arc::new(identity_head_bytes.clone().unwrap()),
                Arc::new(Vec::new()),
            )
        } else {
            (
                Arc::new(k_palette_maps[job_idx].clone()),
                Arc::new(k_palette_scales[job_idx].clone()),
            )
        };
        let (v_pal, v_scale): (Arc<Vec<u8>>, Arc<Vec<f32>>) = if policy.override_v_quant.is_some() {
            (Arc::new(identity_head_bytes.unwrap()), Arc::new(Vec::new()))
        } else {
            (
                Arc::new(v_palette_maps[job_idx].clone()),
                Arc::new(v_palette_scales[job_idx].clone()),
            )
        };
        full_quant_chunks.insert(
            (seq_idx, chunk_idx),
            SealedChunk {
                gids: new_gids,
                offset: src.offset,
                token_count: src.token_count,
                k_pal,
                v_pal,
                k_scale,
                v_scale,
                byte_size,
            },
        );
    }

    // ── Merge: kernel-quantized chunks + preserve-bucket chunks ──────
    // Preserve-bucket chunks (typically cold-loaded mixed-Q chunks)
    // keep their original GIDs, arena indices, and format/pal_map/scale
    // state — no DMA, no copy. The substrate residence ends up with
    // mixed Q-format chunks per the convert kernel's output plus
    // preserve-bucket chunks per their original persist pass. Decode
    // reads them all directly via the per-(h, p) format dispatch in
    // `serialize_chunk_window_with_len`.
    let merged: Vec<SealedSequence> = sequences
        .iter()
        .enumerate()
        .map(|(seq_idx, orig)| {
            // Build partial lookup: chunk_idx → position in preserve list.
            let partial_pos_lookup: std::collections::HashMap<usize, usize> = preserve_per_seq
                [seq_idx]
                .iter()
                .enumerate()
                .map(|(pos, (chunk_idx, _))| (*chunk_idx, pos))
                .collect();
            let mut chunks: Vec<SealedChunk> = Vec::with_capacity(orig.chunks.len());
            for chunk_idx in 0..orig.chunks.len() {
                if let Some(new_chunk) = full_quant_chunks.remove(&(seq_idx, chunk_idx)) {
                    chunks.push(new_chunk);
                } else if let Some(&pos) = partial_pos_lookup.get(&chunk_idx) {
                    chunks.push(preserve_per_seq[seq_idx][pos].1.clone());
                } else {
                    return Err(candle::Error::Msg(format!(
                        "quantize_sealed_in_place: seq {seq_idx} chunk {chunk_idx} \
                         missing from both full-quant and partial-preserve buckets"
                    )));
                }
            }
            Ok(SealedSequence {
                chunks,
                token_count: orig.token_count,
                chunk_size: orig.chunk_size,
                location: ArenaLocation::Gpu,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(merged)
}
