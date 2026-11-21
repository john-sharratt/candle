//! The device addresses and GGML layouts of a chunk's bands, read from the
//! arena and from each band's own tag.
//!
//! A slot is one `(head, palette, side)` band, so its device address is the
//! arena's base plus the chunk index times the class stride, and nothing else.
//! The GGML layout of a band comes only from its tag, because an arena holds
//! whatever fits its slots and cannot say what a band is. The palette maps
//! pack each dimension's palette into the packed map the decode and prefill
//! kernels read, and enforce that each of the four palettes receives exactly
//! `head_dim / N_PALETTE` dimensions, since the kernel's quantization assumes
//! that strict partition.

use super::*;

/// Device pointer of one band's chunk slot.
///
/// Every pointer the convert kernels take is this: a slot *is* one
/// `(head, palette, side)` band, so its address is the arena's base plus
/// `chunk_idx * class_stride` and nothing else. Under per-format arenas this
/// had to be spelled two ways — `tensor_ptr_at_offset` scaled an element
/// offset by a dtype width, `qtensor_ptr_at_byte_offset` took a byte offset
/// computed from a ggml block layout — and both were reconstructing an address
/// the arena can hand over directly.
#[cfg(feature = "cuda")]
pub(in super::super) fn band_ptr(
    storage: &ArenaStorageState,
    gid: &ChunkGid,
    what: &str,
) -> Result<u64> {
    let ai = gid.arena_idx();
    let arena = storage
        .arenas()
        .get(&ai)
        .ok_or_else(|| candle::Error::Msg(format!("{what} arena {ai} not found")))?;
    arena
        .slot_ptr(gid.chunk_idx())
        .ok_or_else(|| candle::Error::Msg(format!("{what} arena {ai} is not GPU-resident")))
}

/// The GGML layout a band's bytes are in, from the band's own tag.
///
/// The arena cannot answer this — it holds whatever fits its slots
/// (`docs/archived/arena_unification.md` principle 8) — so the tag is the only source,
/// and it is the same byte the substrate persisted.
#[cfg(feature = "cuda")]
pub(in super::super) fn band_ggml_dtype(tag: u8) -> Result<GgmlDType> {
    let format = KvFormat::from_tag(tag).ok_or_else(|| {
        candle::Error::Msg(format!("chunk band carries unrecognised format tag {tag}"))
    })?;
    match format {
        KvFormat::Quantized(qf) => Ok(qf.to_ggml_dtype()),
        KvFormat::Float(dt) => dtype_to_ggml_float(dt),
    }
}

/// One head's native source bands as the convert reads them: per palette, each
/// band's address and the GGML layout its tag names.
#[cfg(feature = "cuda")]
pub(in super::super) struct HeadSource {
    pub k_ptrs: [u64; N_PALETTE],
    pub v_ptrs: [u64; N_PALETTE],
    pub k_fmts: [GgmlDType; N_PALETTE],
    pub v_fmts: [GgmlDType; N_PALETTE],
}

/// Head `h`'s source bands of `bands`: the layout from each band's own tag,
/// the address from its arena, through `addrs` — one per walk, so a table's
/// worth of heads looks each arena up once.
#[cfg(feature = "cuda")]
pub(in super::super) fn head_source(
    addrs: &mut BandAddrs<'_>,
    bands: &ChunkBands,
    h: usize,
) -> Result<HeadSource> {
    let mut src = HeadSource {
        k_ptrs: [0; N_PALETTE],
        v_ptrs: [0; N_PALETTE],
        k_fmts: [GgmlDType::F16; N_PALETTE],
        v_fmts: [GgmlDType::F16; N_PALETTE],
    };
    for p in 0..N_PALETTE {
        let (kt, vt) = bands.band_tags(h, p);
        src.k_fmts[p] = band_ggml_dtype(kt)?;
        src.v_fmts[p] = band_ggml_dtype(vt)?;
        src.k_ptrs[p] = addrs.addr(bands.gids.k_gid_pal(h, p), "k src")?;
        src.v_ptrs[p] = addrs.addr(bands.gids.v_gid_pal(h, p), "v src")?;
    }
    Ok(src)
}

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
pub(super) fn pack_head_palette_maps(
    assignments: &[u8],
    n_chunks: usize,
    n_head: usize,
    head_dim: usize,
) -> Result<Vec<Vec<u8>>> {
    let packed_bytes = head_dim / 4;
    if !head_dim.is_multiple_of(4)
        || !head_dim.is_multiple_of(N_PALETTE)
        || !candle::quantized::cuda::kvhead_supported_head_dim(head_dim)
    {
        candle::bail!(
            "quantize_sealed_to_cpu: palette4 conversion requires head_dim 128 or 256, got {}",
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
