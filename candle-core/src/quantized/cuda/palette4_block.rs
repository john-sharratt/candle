//! One block of device memory for a fused palette-4 format selection: its two
//! candidate lists and valid ranges in, its thirteen result arrays out.
//!
//! The selection used to take each of those sixteen arrays from the driver
//! separately — three uploads and seventeen zeroed allocations per call — and
//! read the eight it used back one synchronising copy at a time. It runs on the
//! persistence thread for every sealed group, so that was twenty allocations
//! outside the reservation and eight host round trips per seal.
//!
//! Laid out as one block, the caller carves it from a transient span it already
//! holds, the inputs go up in one copy, the outputs are cleared in one memset,
//! and everything the kernel wrote comes back in one copy.

use std::sync::Arc;

use cudarc::driver::result::{memcpy_dtoh_async, memcpy_htod_async, memset_d8_async};
use cudarc::driver::sys::CUstream;
use cudarc::driver::{CudaStream, DevicePtr};

use super::{ggml_to_select_qtype, select_kv_format_palette4_paged, select_qtype_bpe_x4};
use crate::cuda_backend::{CudaDevice, WrapErr};
use crate::quantized::GgmlDType;
use crate::Result;

/// Every section starts on this boundary — what a fresh allocation guarantees
/// and what the kernel's vectorised loads assume of a base pointer.
const SECTION_ALIGN_WORDS: usize = 256 / 4;

/// Where each array of one selection sits in its block, in 4-byte words from
/// the block's base. Every element the kernel reads or writes is four bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Palette4Layout {
    pub total_heads: usize,
    pub total_blocks: usize,
    pub k_cand: usize,
    pub v_cand: usize,
    pub valid_ranges: usize,
    /// First word of the output section; everything from here to
    /// [`Self::words`] is cleared before the launch and read back after it.
    pub outputs: usize,
    pub q_rel_median: usize,
    pub q_rel_spread: usize,
    pub k_head_amax: usize,
    pub v_head_amax: usize,
    pub k_head_p95: usize,
    pub v_head_p95: usize,
    pub k_palette_tags: usize,
    pub v_palette_tags: usize,
    pub k_pal_scale: usize,
    pub v_pal_scale: usize,
    pub k_palette_map: usize,
    pub v_palette_map: usize,
    pub k_eff_block_tags: usize,
    pub v_eff_block_tags: usize,
    pub k_head_tags: usize,
    pub v_head_tags: usize,
    pub q_rel: usize,
    /// Words the whole block spans.
    pub words: usize,
}

impl Palette4Layout {
    pub fn new(
        n_chunks: usize,
        n_kv_head: usize,
        blocks_per_head: usize,
        n_k_cand: usize,
        n_v_cand: usize,
    ) -> Self {
        let total_heads = n_chunks * n_kv_head;
        let total_blocks = total_heads * blocks_per_head;
        let mut at = 0usize;
        let mut section = |len: usize| {
            let start = at;
            at = (start + len).next_multiple_of(SECTION_ALIGN_WORDS);
            start
        };
        let k_cand = section(n_k_cand);
        let v_cand = section(n_v_cand);
        let valid_ranges = section(n_chunks);
        let q_rel_median = section(total_heads);
        let q_rel_spread = section(total_heads);
        let k_head_amax = section(total_heads);
        let v_head_amax = section(total_heads);
        let k_head_p95 = section(total_heads);
        let v_head_p95 = section(total_heads);
        let k_palette_tags = section(total_heads * 4);
        let v_palette_tags = section(total_heads * 4);
        let k_pal_scale = section(total_heads * 4);
        let v_pal_scale = section(total_heads * 4);
        let k_palette_map = section(total_blocks);
        let v_palette_map = section(total_blocks);
        let k_eff_block_tags = section(total_blocks);
        let v_eff_block_tags = section(total_blocks);
        let k_head_tags = section(total_heads);
        let v_head_tags = section(total_heads);
        let q_rel = section(total_blocks);
        Self {
            total_heads,
            total_blocks,
            k_cand,
            v_cand,
            valid_ranges,
            outputs: q_rel_median,
            q_rel_median,
            q_rel_spread,
            k_head_amax,
            v_head_amax,
            k_head_p95,
            v_head_p95,
            k_palette_tags,
            v_palette_tags,
            k_pal_scale,
            v_pal_scale,
            k_palette_map,
            v_palette_map,
            k_eff_block_tags,
            v_eff_block_tags,
            k_head_tags,
            v_head_tags,
            q_rel,
            words: at,
        }
    }

    /// Bytes the block spans — what the caller carves.
    pub fn bytes(&self) -> usize {
        self.words * 4
    }
}

/// What one selection wrote, read back to the host.
#[derive(Debug, Clone, PartialEq)]
pub struct Palette4Result {
    /// `[total_heads * 4]`.
    pub k_palette_tags: Vec<i32>,
    pub v_palette_tags: Vec<i32>,
    /// `[total_heads * 4]` — the outer scale per palette slot.
    pub k_pal_scale: Vec<f32>,
    pub v_pal_scale: Vec<f32>,
    /// `[total_blocks]` — each block's palette slot.
    pub k_palette_map: Vec<i32>,
    pub v_palette_map: Vec<i32>,
    /// `[total_heads]`.
    pub k_head_amax: Vec<f32>,
    pub v_head_amax: Vec<f32>,
    /// `[total_blocks]` — each block's tag after palette expansion.
    pub k_eff_block_tags: Vec<i32>,
    pub v_eff_block_tags: Vec<i32>,
    /// `[total_heads]` — the most conservative format of each head.
    pub k_head_tags: Vec<i32>,
    pub v_head_tags: Vec<i32>,
    /// `[total_blocks]`.
    pub q_rel: Vec<f32>,
}

impl Palette4Result {
    /// Decode the output section — `words` read back from `layout.outputs` on.
    pub fn decode(layout: &Palette4Layout, words: &[u32]) -> Self {
        let base = layout.outputs;
        let i32s = |at: usize, len: usize| -> Vec<i32> {
            words[at - base..at - base + len]
                .iter()
                .map(|&w| w as i32)
                .collect()
        };
        let f32s = |at: usize, len: usize| -> Vec<f32> {
            words[at - base..at - base + len]
                .iter()
                .map(|&w| f32::from_bits(w))
                .collect()
        };
        let (th, tb) = (layout.total_heads, layout.total_blocks);
        Self {
            k_palette_tags: i32s(layout.k_palette_tags, th * 4),
            v_palette_tags: i32s(layout.v_palette_tags, th * 4),
            k_pal_scale: f32s(layout.k_pal_scale, th * 4),
            v_pal_scale: f32s(layout.v_pal_scale, th * 4),
            k_palette_map: i32s(layout.k_palette_map, tb),
            v_palette_map: i32s(layout.v_palette_map, tb),
            k_head_amax: f32s(layout.k_head_amax, th),
            v_head_amax: f32s(layout.v_head_amax, th),
            k_eff_block_tags: i32s(layout.k_eff_block_tags, tb),
            v_eff_block_tags: i32s(layout.v_eff_block_tags, tb),
            k_head_tags: i32s(layout.k_head_tags, th),
            v_head_tags: i32s(layout.v_head_tags, th),
            q_rel: f32s(layout.q_rel, tb),
        }
    }

    fn empty() -> Self {
        Self {
            k_palette_tags: Vec::new(),
            v_palette_tags: Vec::new(),
            k_pal_scale: Vec::new(),
            v_pal_scale: Vec::new(),
            k_palette_map: Vec::new(),
            v_palette_map: Vec::new(),
            k_head_amax: Vec::new(),
            v_head_amax: Vec::new(),
            k_eff_block_tags: Vec::new(),
            v_eff_block_tags: Vec::new(),
            k_head_tags: Vec::new(),
            v_head_tags: Vec::new(),
            q_rel: Vec::new(),
        }
    }
}

/// The candidate codes in the order the kernel walks them: ascending bytes per
/// element, so the first format to fill the 32-block quota is the most
/// aggressive one. A **stable** sort — formats of equal cost keep the caller's
/// order, which is how a caller states priority within a tier.
fn sorted_codes(candidates: &[GgmlDType]) -> Result<Vec<i32>> {
    let mut codes = candidates
        .iter()
        .map(|d| ggml_to_select_qtype(*d))
        .collect::<Result<Vec<_>>>()?;
    codes.sort_by_key(|c| select_qtype_bpe_x4(*c));
    Ok(codes)
}

/// The inputs as the words the input section holds, padded to `layout.outputs`.
fn input_words(layout: &Palette4Layout, k: &[i32], v: &[i32], valid: &[i32]) -> Vec<u32> {
    let mut words = vec![0u32; layout.outputs];
    for (dst, src) in [
        (layout.k_cand, k),
        (layout.v_cand, v),
        (layout.valid_ranges, valid),
    ] {
        for (i, &x) in src.iter().enumerate() {
            words[dst + i] = x as u32;
        }
    }
    words
}

/// Run the fused palette-4 selection in `block`, a device range of at least
/// [`Palette4Layout::bytes`] the caller holds for the call, and read back what
/// it wrote. Synchronises `stream` once, for the readback.
///
/// # Safety
///
/// `per_head_table_ptr` and `head_gids_ptr` are raw device addresses, laid out
/// as the kernel expects — `PerHeadEntry` rows and the matching `HeadGids`
/// block — and live on `stream` for the call. `block` names device memory of at
/// least `layout.bytes()` that nothing else touches until this returns.
#[allow(clippy::too_many_arguments)]
pub unsafe fn select_palette4_in(
    per_head_table_ptr: u64,
    head_gids_ptr: u64,
    n_chunks: usize,
    k_candidates: &[GgmlDType],
    v_candidates: &[GgmlDType],
    thresholds: [f32; 4],
    blocks_per_head: usize,
    n_kv_head: usize,
    arena_chunks: usize,
    valid_ranges: &[i32],
    block: u64,
    stream: &Arc<CudaStream>,
) -> Result<Palette4Result> {
    if valid_ranges.len() != n_chunks {
        crate::bail!(
            "valid_ranges length {} != n_chunks {}",
            valid_ranges.len(),
            n_chunks
        );
    }
    let k_codes = sorted_codes(k_candidates)?;
    let v_codes = sorted_codes(v_candidates)?;
    let layout = Palette4Layout::new(
        n_chunks,
        n_kv_head,
        blocks_per_head,
        k_codes.len(),
        v_codes.len(),
    );
    if layout.total_heads == 0 {
        return Ok(Palette4Result::empty());
    }
    let cu: CUstream = stream.cu_stream();
    let word = |at: usize| block + (at * 4) as u64;
    let inputs = input_words(&layout, &k_codes, &v_codes, valid_ranges);
    // SAFETY: `inputs` is `layout.outputs` words and the block holds at least
    // that many before its output section; pageable, so staged before return.
    memcpy_htod_async(block, &inputs, cu).w()?;
    let out_bytes = (layout.words - layout.outputs) * 4;
    // SAFETY: the output section lies inside the block.
    memset_d8_async(word(layout.outputs), 0, out_bytes, cu).w()?;
    let [k_hi, k_lo, v_hi, v_lo] = thresholds;
    select_kv_format_palette4_paged(
        per_head_table_ptr,
        head_gids_ptr,
        word(layout.q_rel_median),
        word(layout.q_rel_spread),
        word(layout.k_head_amax),
        word(layout.v_head_amax),
        word(layout.k_head_p95),
        word(layout.v_head_p95),
        word(layout.k_cand),
        word(layout.v_cand),
        k_codes.len() as i32,
        v_codes.len() as i32,
        k_hi,
        k_lo,
        v_hi,
        v_lo,
        layout.total_heads as i32,
        blocks_per_head as i32,
        n_kv_head as i32,
        arena_chunks as i32,
        word(layout.valid_ranges),
        word(layout.k_palette_tags),
        word(layout.v_palette_tags),
        word(layout.k_pal_scale),
        word(layout.v_pal_scale),
        word(layout.k_palette_map),
        word(layout.v_palette_map),
        word(layout.k_eff_block_tags),
        word(layout.v_eff_block_tags),
        word(layout.k_head_tags),
        word(layout.v_head_tags),
        word(layout.q_rel),
        cu as *mut std::ffi::c_void,
    );
    let mut out = vec![0u32; layout.words - layout.outputs];
    // SAFETY: the output section lies inside the block; the copy is ordered
    // behind the launch and waited for below.
    memcpy_dtoh_async(&mut out, word(layout.outputs), cu).w()?;
    stream.synchronize().w()?;
    Ok(Palette4Result::decode(&layout, &out))
}

/// [`select_palette4_in`] with its block taken from the driver — for the
/// test wrappers, which hold no transient span to carve from.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn select_palette4_owned(
    per_head_table_ptr: u64,
    head_gids_ptr: u64,
    n_chunks: usize,
    k_candidates: &[GgmlDType],
    v_candidates: &[GgmlDType],
    thresholds: [f32; 4],
    blocks_per_head: usize,
    n_kv_head: usize,
    arena_chunks: usize,
    valid_ranges: &[i32],
    dev: &CudaDevice,
) -> Result<Palette4Result> {
    let layout = Palette4Layout::new(
        n_chunks,
        n_kv_head,
        blocks_per_head,
        k_candidates.len(),
        v_candidates.len(),
    );
    let stream = dev.cuda_stream();
    let block = dev.alloc::<u8>(layout.bytes().max(4))?;
    let (base, _guard) = block.device_ptr(&stream);
    select_palette4_in(
        per_head_table_ptr,
        head_gids_ptr,
        n_chunks,
        k_candidates,
        v_candidates,
        thresholds,
        blocks_per_head,
        n_kv_head,
        arena_chunks,
        valid_ranges,
        base,
        &stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sections follow each other on 64-word boundaries in the order the
    /// kernel's arguments name them, inputs first; the output section is
    /// everything after the inputs.
    #[test]
    fn sections_are_aligned_and_in_order() {
        // 2 chunks × 2 heads = 4 heads, 128 blocks each = 512 blocks.
        let l = Palette4Layout::new(2, 2, 128, 5, 3);
        assert_eq!((l.total_heads, l.total_blocks), (4, 512));
        assert_eq!((l.k_cand, l.v_cand, l.valid_ranges), (0, 64, 128));
        assert_eq!(l.outputs, 192);
        assert_eq!(l.q_rel_median, 192);
        assert_eq!(l.q_rel_spread, 256);
        assert_eq!(l.k_head_amax, 320);
        assert_eq!(l.v_head_amax, 384);
        assert_eq!(l.k_head_p95, 448);
        assert_eq!(l.v_head_p95, 512);
        assert_eq!(l.k_palette_tags, 576);
        assert_eq!(l.v_palette_tags, 640);
        assert_eq!(l.k_pal_scale, 704);
        assert_eq!(l.v_pal_scale, 768);
        assert_eq!(l.k_palette_map, 832);
        assert_eq!(l.v_palette_map, 1344);
        assert_eq!(l.k_eff_block_tags, 1856);
        assert_eq!(l.v_eff_block_tags, 2368);
        assert_eq!(l.k_head_tags, 2880);
        assert_eq!(l.v_head_tags, 2944);
        assert_eq!(l.q_rel, 3008);
        assert_eq!(l.words, 3520);
        assert_eq!(l.bytes(), 14080);
    }

    /// The input section holds each list at its offset, zero elsewhere.
    #[test]
    fn inputs_land_at_their_offsets() {
        let l = Palette4Layout::new(1, 1, 128, 2, 1);
        let w = input_words(&l, &[7, -1], &[9], &[0x0120]);
        assert_eq!(w.len(), l.outputs);
        assert_eq!(&w[0..3], &[7, u32::MAX, 0]);
        assert_eq!(w[l.v_cand], 9);
        assert_eq!(w[l.valid_ranges], 0x0120);
        assert_eq!(w.iter().filter(|&&x| x != 0).count(), 4);
    }

    /// Each result array is read from its own section of the output words, as
    /// the bit pattern the kernel stored.
    #[test]
    fn results_decode_from_their_sections() {
        let l = Palette4Layout::new(1, 1, 128, 1, 1);
        let mut words = vec![0u32; l.words - l.outputs];
        let at = |x: usize| x - l.outputs;
        words[at(l.k_palette_tags)] = 3;
        words[at(l.v_palette_tags) + 3] = (-2i32) as u32;
        words[at(l.k_pal_scale) + 1] = 0.5f32.to_bits();
        words[at(l.v_palette_map) + 127] = 2;
        words[at(l.k_head_amax)] = 1.5f32.to_bits();
        words[at(l.v_head_tags)] = 11;
        words[at(l.q_rel) + 5] = 0.25f32.to_bits();
        let r = Palette4Result::decode(&l, &words);
        assert_eq!(r.k_palette_tags, vec![3, 0, 0, 0]);
        assert_eq!(r.v_palette_tags, vec![0, 0, 0, -2]);
        assert_eq!(r.k_pal_scale, vec![0.0, 0.5, 0.0, 0.0]);
        assert_eq!(r.v_palette_map.len(), 128);
        assert_eq!(r.v_palette_map[127], 2);
        assert_eq!(r.k_head_amax, vec![1.5]);
        assert_eq!(r.v_head_tags, vec![11]);
        assert_eq!(r.q_rel[5], 0.25);
        assert_eq!(r.k_eff_block_tags, vec![0; 128]);
    }
}
