//! QSA block-sparse selection, at the kernel boundary.
//!
//! The paged decode and prefill kernels take a per-query selection
//! (`candle-kernels/src/qsa_select.cuh`) built from the packing in
//! `models::qwen4exp::qsa_select`. Two properties pin it:
//!
//! 1. **Selecting everything is the old behaviour.** A selection whose rows
//!    are marked dense, and one that names every block explicitly (so the
//!    kernel's binary search actually runs), must both reproduce the
//!    unselected kernel's output. This catches an off-by-one in the position
//!    arithmetic, a mis-indexed row, and any perturbation of the softmax the
//!    mask was not supposed to cause. It is checked to a couple of BF16 ulps
//!    rather than bit for bit: the launcher sizes the split factor from the
//!    selection's row pitch, so a selected run walks the same tokens in a
//!    different number of splits and merges its fp32 partials in a different
//!    order. A mask bug moves the output by orders of magnitude more.
//!
//! 2. **An unselected token cannot influence the answer.** Rebuild the history
//!    with *different values* at every unselected position and the output must
//!    not move by one bit — the two runs do identical work in identical order,
//!    so this one needs no tolerance. This is the property QSA actually claims,
//!    and it is checkable without reimplementing the kernel's int8 arithmetic:
//!    nothing about the reference has to be computed, only that two runs agree.
//!    The same comparison run without the selection must DISAGREE — otherwise
//!    the test would pass on a kernel that ignores the selection entirely.
//!
//! The arenas are float BF16, which is what the engine runs (`k_format`
//! `Float(BF16)`): per-element storage, so changing an unselected token's
//! value cannot move a selected token's stored bits through a shared
//! quantization scale.
//!
//! Geometries cover all three decode routes the launcher dispatches between —
//! `heads_per_group` 12 at head_dim 256 (the warp=head kernel, which is
//! Qwen3.8-Flash-Next's own shape), 2 at 128 (batched-M MMA), and 2 at 64
//! (warp-stripe) — and the prefill kernel at 256 and 128.

#![cfg(feature = "cuda")]

use candle::quantized::pinned_staging::{Generation, GpuBuf, PinnedBuf, PinnedStager};
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{
    quantize_sealed_in_place, ArenaFormatTag, ChunkedKvBacking, CompressionPolicy, KvCache,
    CHUNK_SIZE, N_PALETTE,
};
use candle_transformers::models::prefill_utils::{
    compute_rope_cs, paged_decode_attn, paged_prefill_batched,
};
use candle_transformers::models::qsa_selection::QsaSelection;
use candle_transformers::models::qwen4exp::qsa_select::{entry_block, pack_entry, DENSE_ROW};
use std::sync::{Mutex, MutexGuard};

// One GPU, one process-global quantized arena table — same serialization rule
// as the other kernel test binaries.
static GPU_SERIAL: Mutex<()> = Mutex::new(());

fn gpu_serial() -> MutexGuard<'static, ()> {
    GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

const MAX_BLOCKS: usize = 64;
const RATIO: usize = 4;

#[derive(Clone, Copy)]
struct Geom {
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
}

fn pseudo(i: usize, j: usize, k: usize, seed: u64) -> f32 {
    let mut x = (i as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add((k as u64).wrapping_mul(0x1656_67B1_9E37_79F9))
        .wrapping_add(seed.wrapping_mul(0x94D0_49BB_1331_11EB));
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x as i64 as f32) / (i64::MAX as f32) * 0.5
}

/// Flat `[n, n_head, hd]` Q and `[n, n_kv_head, hd]` K/V, BF16.
///
/// `alt` names the positions whose K/V are **negated** — the history a run is
/// not allowed to look at.
///
/// Negation, not a different seed, because the int8 kernels quantize some
/// operands over a span wider than one token: the prefill kernel's FP-fallback
/// V requant takes its scale as a max-abs per (dim, 32-token tile). A perturbed
/// magnitude anywhere in a tile therefore moves the *quantization* of the
/// selected tokens beside it — a numerical shift the mask cannot prevent, and
/// one the kernel already has for causally-dead columns. Flipping signs leaves
/// every magnitude, and so every scale, exactly where it was, while changing
/// the value any read of that token would return.
fn make_qkv(
    g: Geom,
    n_tokens: usize,
    seed: u64,
    alt: &[bool],
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    make_qkv_at(g, 0, n_tokens, seed, alt, device)
}

/// [`make_qkv`] for a segment starting at absolute token `base`.
///
/// The values are a function of the absolute token index, so a history built in
/// segments is bit-identical to the same history built in one call — which is
/// what lets the depth cases avoid materialising a multi-gigabyte query tensor.
fn make_qkv_at(
    g: Geom,
    base: usize,
    n_tokens: usize,
    seed: u64,
    alt: &[bool],
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut q = Vec::with_capacity(n_tokens * g.n_head * g.head_dim);
    let mut k = Vec::with_capacity(n_tokens * g.n_kv_head * g.head_dim);
    let mut v = Vec::with_capacity(n_tokens * g.n_kv_head * g.head_dim);
    for local in 0..n_tokens {
        let t = base + local;
        let sign = if alt.get(local).copied().unwrap_or(false) {
            -1.0
        } else {
            1.0
        };
        for h in 0..g.n_head {
            for d in 0..g.head_dim {
                q.push(pseudo(t, h, d, seed ^ 0x111));
            }
        }
        for h in 0..g.n_kv_head {
            for d in 0..g.head_dim {
                k.push(sign * pseudo(t, h, d, seed ^ 0x222));
                v.push(sign * pseudo(t, h, d, seed ^ 0x333));
            }
        }
    }
    Ok((
        Tensor::from_vec(q, (n_tokens, g.n_head, g.head_dim), device)?.to_dtype(DType::BF16)?,
        Tensor::from_vec(k, (n_tokens, g.n_kv_head, g.head_dim), device)?.to_dtype(DType::BF16)?,
        Tensor::from_vec(v, (n_tokens, g.n_kv_head, g.head_dim), device)?.to_dtype(DType::BF16)?,
    ))
}

/// A prefilled slot of `history` tokens, with the K/V at `alt` positions drawn
/// from a different seed.
fn build_history_slot(
    g: Geom,
    history: usize,
    seed: u64,
    alt: &[bool],
    rope_cs: &Tensor,
    stager: &PinnedStager,
    device: &Device,
) -> Result<(ChunkedKvBacking, KvCache)> {
    // Capacity from the history, not a fixed constant: [`MAX_BLOCKS`] is sized
    // for the 100-token cases and holds 2,048 tokens, so a depth case silently
    // ran off the end of both the backing and the rope table — reading garbage
    // at 8K and faulting at 32K.
    let blocks = MAX_BLOCKS.max(history.div_ceil(CHUNK_SIZE) + 2);
    let backing = ChunkedKvBacking::new(4, g.n_kv_head, g.head_dim, DType::BF16, device, blocks)?;
    let mut cache = KvCache::new(2, 64);
    cache.force_dtype(DType::BF16);
    cache.set_chunked_backing(&backing, 0, None)?;

    backing.ensure_for_batch_entries(&[(0, 0)], history)?;
    let rope_offsets = Tensor::zeros(1, DType::U32, device)?;

    // One call, which caps the reachable depth: the query side of a prefill is
    // `history × n_head × head_dim`, so at 256K and this geometry it is a 2 GiB
    // tensor — past what 32-bit element offsets address, and the fault it
    // produces (`CUDA_ERROR_ILLEGAL_ADDRESS`) reads exactly like a kernel bug.
    // 128K fits; 256K needs this filled a segment at a time, and a first attempt
    // at that changed the history the depth case builds (the write position and
    // rope base both come off the cache's own length, and getting that wrong is
    // silent), so it is left as the harness's known ceiling rather than a
    // half-verified builder.
    let (q, k, v) = make_qkv(g, history, seed, alt, device)?;
    backing.ensure_for_batch_entries(&[(0, 0)], history)?;
    let generation = stager.begin_generation();
    {
        let mut caches_arr: [&mut KvCache; 1] = [&mut cache];
        let _ = paged_prefill_batched(
            None,
            &mut caches_arr[..],
            &[0],
            &q,
            &k,
            &v,
            1,
            &[history],
            g.n_head,
            g.n_kv_head,
            g.head_dim,
            None,
            &rope_offsets,
            rope_cs,
            false,
            &generation,
            &std::cell::RefCell::new(None),
            None,
        )?;
    }
    cache.set_current_seq_len(history)?;
    Ok((backing, cache))
}

/// Slot 0's history re-sealed through the compression policy at `level`, on
/// `slot` of the same backing: the committed chunks recorded as a turn,
/// quantized in place (the policy picks each `(head, palette)` band's format
/// against its own error threshold), injected at the tail of `slot` behind a
/// primed writer chunk. This is what every chunk behind the live one looks
/// like once the seal path has run over it, so a decode against it exercises
/// the kernel's quant staging and readthrough rather than the BF16 element
/// decode the float slot measures.
///
/// Returns the bound cache and a histogram of the formats the policy chose,
/// `(k_fmt, v_fmt) → bands`, so the column names what it measured.
/// One `(k_fmt, v_fmt)` pair and how many bands the seal gave it.
type FormatBandCount = ((ArenaFormatTag, ArenaFormatTag), usize);

fn seal_history(
    backing: &ChunkedKvBacking,
    history: usize,
    level: u8,
    slot: usize,
    device: &Device,
) -> Result<(KvCache, Vec<FormatBandCount>)> {
    let policy = CompressionPolicy::new(level);
    // The history alone: launches against the float slot have appended past
    // it, and a history off the chunk grid seals as a partial chunk — the
    // gap shape every turn boundary leaves behind.
    backing.truncate_sequence_to_tokens(0, history)?;
    let src = backing.record_turn(0)?;
    let copy_stream = match device {
        Device::Cuda(d) => d.cuda_stream(),
        _ => unreachable!("gated on a CUDA device"),
    };
    let mut pinned: Option<PinnedBuf> = None;
    let warm =
        quantize_sealed_in_place(backing, &[&src], &policy, device, &copy_stream, &mut pinned)?;
    copy_stream
        .synchronize()
        .map_err(|e| candle::Error::Msg(format!("seal sync: {e}")))?;
    let warm = warm.into_iter().next().expect("one sealed in → one out");

    let mut formats: Vec<((ArenaFormatTag, ArenaFormatTag), usize)> = Vec::new();
    for c in &warm.chunks {
        for (k, v) in c.k_fmt.iter().zip(c.v_fmt.iter()) {
            let key = (ArenaFormatTag::from_u8(*k), ArenaFormatTag::from_u8(*v));
            match formats.iter_mut().find(|(f, _)| *f == key) {
                Some((_, n)) => *n += 1,
                None => formats.push((key, 1)),
            }
        }
    }
    // Descending by band count, so the dominant format leads the report.
    formats.sort_by_key(|f| std::cmp::Reverse(f.1));
    let bands = warm.chunks.len() * warm.chunks[0].k_fmt.len();
    assert_eq!(
        formats.iter().map(|(_, n)| n).sum::<usize>(),
        bands,
        "every (chunk, head, palette) band carries a recorded format"
    );
    assert_eq!(
        warm.chunks[0].k_fmt.len() % N_PALETTE,
        0,
        "formats are recorded per (head, palette)"
    );

    let mut cache = KvCache::new(2, 64);
    cache.force_dtype(DType::BF16);
    cache.set_chunked_backing(backing, slot, None)?;
    backing.inject_sealed_at_tail(slot, &warm)?;
    backing.push_empty_writer_chunk(slot)?;
    cache.set_current_seq_len(warm.token_count)?;
    Ok((cache, formats))
}

/// One slot's device-side decode metadata — slot header, slice table and
/// kvhead records — exactly as `build_decode_metadata` builds it. The tensors
/// and the staging generation stay alive for as long as the header is
/// launched against.
struct SlotHeaders {
    headers: GpuBuf,
    _headers_dev: Tensor,
    _generation: Generation,
    _records: Tensor,
    _slices: Tensor,
}

impl SlotHeaders {
    fn build(
        g: Geom,
        backing: &ChunkedKvBacking,
        cache: &KvCache,
        stager: &PinnedStager,
        device: &Device,
    ) -> Result<Self> {
        use candle_transformers::models::slot_state::{
            tensor_u8_device_ptr, SlotStateHost, TokenSliceHost,
        };

        let seq_offset = cache.current_seq_len();
        let slot = cache
            .k_cache()
            .chunked_slot()
            .expect("cache bound to a chunked backing");
        let arena_info = backing.resolve_arena_info()?;
        backing.ensure_for_batch_entries(&[(slot, seq_offset)], 1)?;

        let chunks = cache
            .k_cache()
            .chunked_live_chunks_as_sealed()
            .unwrap_or_default();
        let writer_start = cache.k_cache().chunked_writer_start_idx().unwrap_or(0);

        let mut slot = SlotStateHost::from_sealed_chunks(
            &chunks,
            g.n_kv_head,
            g.head_dim,
            &arena_info,
            writer_start,
            false,
        );
        slot.extend_for_write_region(1, CHUNK_SIZE);

        let mut records_buf: Vec<u8> = Vec::new();
        let mut rec_offset: Vec<Option<usize>> = Vec::with_capacity(slot.slices.len());
        for s in &slot.slices {
            if s.meta.is_some() {
                rec_offset.push(None);
            } else {
                rec_offset.push(Some(records_buf.len()));
                s.serialize_record(&mut records_buf, None);
            }
        }
        let records = if records_buf.is_empty() {
            Tensor::zeros(1, DType::U8, device)?
        } else {
            Tensor::from_slice(&records_buf, records_buf.len(), device)?
        };
        let records_base = tensor_u8_device_ptr(&records)?;

        let mut slice_buf =
            Vec::with_capacity(slot.slices.len() * TokenSliceHost::SLICE_HEADER_SIZE);
        for (s, off) in slot.slices.iter().zip(&rec_offset) {
            let kvheads_ptr = match s.meta.as_ref() {
                Some(m) => m.device_addr(),
                None => records_base + off.expect("non-meta slice has a record offset") as u64,
            };
            s.serialize_slice_header(&mut slice_buf, kvheads_ptr);
        }
        let slices = Tensor::from_slice(&slice_buf, slice_buf.len(), device)?;
        let slices_base_ptr = tensor_u8_device_ptr(&slices)?;

        // Decode headers carry no position map (no decode kernel reads one),
        // exactly as `build_decode_metadata` serialises them.
        let mut hdr = Vec::with_capacity(24);
        hdr.extend_from_slice(&(slot.slices.len() as u32).to_le_bytes());
        hdr.extend_from_slice(&slot.write_slice.to_le_bytes());
        hdr.extend_from_slice(&slices_base_ptr.to_le_bytes());
        hdr.extend_from_slice(&0u64.to_le_bytes());

        let generation = stager.begin_generation();
        let headers_dev = Tensor::from_slice(&hdr, hdr.len(), device)?;
        let headers = GpuBuf::from_borrowed(tensor_u8_device_ptr(&headers_dev)?, hdr.len());
        Ok(Self {
            headers,
            _headers_dev: headers_dev,
            _generation: generation,
            _records: records,
            _slices: slices,
        })
    }

    /// One decode step through the production wrapper. The kernel commits
    /// the step's token into the write slice on the device copy of the slice
    /// table, so consecutive launches against one header append one token
    /// each — the decode pattern the wrapper runs in production.
    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        g: Geom,
        q: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        rope_cs: &Tensor,
        qsa: Option<&QsaSelection>,
    ) -> Result<Tensor> {
        paged_decode_attn(
            None,
            q,
            self.headers.dev_ptr(),
            DType::BF16,
            g.n_head,
            g.n_kv_head,
            g.head_dim,
            1.0f32 / (g.head_dim as f32).sqrt(),
            k_new,
            v_new,
            rope_cs,
            false,
            qsa,
        )
    }
}

/// Slot header built fresh from the host cache state, then one decode step.
#[allow(clippy::too_many_arguments)]
fn decode_one_slot(
    g: Geom,
    backing: &ChunkedKvBacking,
    cache: &KvCache,
    q: &Tensor,
    k_new: &Tensor,
    v_new: &Tensor,
    rope_cs: &Tensor,
    qsa: Option<&QsaSelection>,
    stager: &PinnedStager,
    device: &Device,
) -> Result<Tensor> {
    SlotHeaders::build(g, backing, cache, stager, device)?.decode(g, q, k_new, v_new, rope_cs, qsa)
}

// ──────────────────────────────────────────────────────────────────────
// Selections
// ──────────────────────────────────────────────────────────────────────

/// Build a selection from one entry list per row.
fn selection(rows: &[Vec<u32>], device: &Device) -> Result<QsaSelection> {
    let stride = rows.iter().map(|r| r.len()).max().unwrap_or(1).max(1);
    let mut flat = vec![0u32; rows.len() * stride];
    let mut cnt = Vec::with_capacity(rows.len());
    for (i, r) in rows.iter().enumerate() {
        flat[i * stride..i * stride + r.len()].copy_from_slice(r);
        cnt.push(r.len() as u32);
    }
    QsaSelection::new(
        Tensor::from_vec(flat, (rows.len(), stride), device)?,
        Tensor::from_vec(cnt, (rows.len(),), device)?,
        RATIO,
    )
}

/// A selection whose rows are all marked dense.
fn dense_selection(rows: usize, device: &Device) -> Result<QsaSelection> {
    QsaSelection::new(
        Tensor::from_vec(vec![0u32; rows], (rows, 1), device)?,
        Tensor::from_vec(vec![DENSE_ROW; rows], (rows,), device)?,
        RATIO,
    )
}

/// `rows` with the ones `dense` flags marked dense instead — their entry
/// lists stay in the table, unread behind the marker.
fn mixed_selection(rows: &[Vec<u32>], dense: &[bool], device: &Device) -> Result<QsaSelection> {
    assert_eq!(rows.len(), dense.len());
    let stride = rows.iter().map(|r| r.len()).max().unwrap_or(1).max(1);
    let mut flat = vec![0u32; rows.len() * stride];
    let mut cnt = Vec::with_capacity(rows.len());
    for (i, r) in rows.iter().enumerate() {
        flat[i * stride..i * stride + r.len()].copy_from_slice(r);
        cnt.push(if dense[i] { DENSE_ROW } else { r.len() as u32 });
    }
    QsaSelection::new(
        Tensor::from_vec(flat, (rows.len(), stride), device)?,
        Tensor::from_vec(cnt, (rows.len(),), device)?,
        RATIO,
    )
}

/// Every cell of `[0, visible)` named explicitly, so the kernel's binary
/// search runs over a full list rather than short-circuiting on the marker.
fn explicit_full_row(visible: usize) -> Vec<u32> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < visible {
        let cells = RATIO.min(visible - pos);
        out.push(pack_entry(pos / RATIO, cells));
        pos += RATIO;
    }
    out
}

/// A pseudo-random subset of the blocks below `visible`, always including the
/// block that holds position `visible - 1` (the query's own tail, which QSA
/// forces in) so no row can select nothing.
fn subset_row(visible: usize, seed: u64) -> (Vec<u32>, Vec<bool>) {
    let n_blocks = visible.div_ceil(RATIO);
    let tail_block = (visible - 1) / RATIO;
    let mut entries = Vec::new();
    let mut selected = vec![false; visible];
    for b in 0..n_blocks {
        let keep = b == tail_block || pseudo(b, 7, 3, seed) > 0.0;
        if !keep {
            continue;
        }
        let cells = RATIO.min(visible - b * RATIO);
        entries.push(pack_entry(b, cells));
        for c in 0..cells {
            selected[b * RATIO + c] = true;
        }
    }
    (entries, selected)
}

/// A **production-shaped** selection: a fixed budget of positions however deep
/// the cache is.
///
/// [`subset_row`] keeps roughly half the blocks, so its selection grows with
/// depth — fine for checking that the kernel honours a selection, useless for
/// asking what the kernel costs, because the thing QSA promises is exactly that
/// the attended set stops growing. This keeps `top_k` positions: the tail block
/// (which the selection always forces in) plus evenly-spread earlier blocks.
///
/// `phase` slides the spread by half its stride. The indexer scores every
/// query token on its own, so two neighbouring tokens select sets that overlap
/// in a few blocks and differ in most; a prefill block packing both walks the
/// UNION of the two. Rows built with `phase = 0` for one token and `phase = 1`
/// for its neighbour have that shape — interior blocks disjoint, tails shared —
/// which is what makes a block's tile stream twice its budget, the way the
/// model's does. With every row at phase `0` the neighbours' sets coincide and
/// the block walks one budget's worth: the case a real selection never hands
/// the kernel.
fn budget_row(visible: usize, top_k: usize, phase: usize) -> (Vec<u32>, Vec<bool>) {
    let n_blocks = visible.div_ceil(RATIO);
    let tail_block = (visible - 1) / RATIO;
    let want = (top_k / RATIO).min(n_blocks).max(1);
    let mut keep: Vec<usize> = Vec::with_capacity(want);
    // Spread the budget over the history rather than clustering it, so the
    // slice cursor advances the way a real selection makes it advance.
    let shift = phase * (tail_block / want / 2);
    for i in 0..want.saturating_sub(1) {
        let b = (i * tail_block.max(1) / want.max(1) + shift).min(tail_block.max(1) - 1);
        if keep.last() != Some(&b) {
            keep.push(b);
        }
    }
    if keep.last() != Some(&tail_block) {
        keep.push(tail_block);
    }
    let mut entries = Vec::with_capacity(keep.len());
    let mut selected = vec![false; visible];
    for &b in &keep {
        let cells = RATIO.min(visible - b * RATIO);
        entries.push(pack_entry(b, cells));
        for c in 0..cells {
            selected[b * RATIO + c] = true;
        }
    }
    (entries, selected)
}

fn bits(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

/// Property 1's comparison: the two outputs walked the same tokens through a
/// different number of splits, so they agree to the fp32 merge order — a
/// couple of BF16 ulps at the output's own magnitude. `2^-7` is one BF16 ulp
/// of the largest value; the bound is two of them.
fn assert_same_to_bf16_ulps(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    let mag = a.iter().fold(0f32, |m, x| m.max(x.abs()));
    let tol = 2.0 * mag * (1.0 / 128.0);
    let (worst_i, worst) = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .enumerate()
        .fold((0, 0f32), |w, (i, d)| if d > w.1 { (i, d) } else { w });
    assert!(
        worst <= tol,
        "{what}: element {worst_i} moved by {worst:.3e} ({} vs {}), more than {tol:.3e} — \
         two BF16 ulps of the output's magnitude {mag:.3e}",
        a[worst_i],
        b[worst_i],
    );
}

// ──────────────────────────────────────────────────────────────────────
// Decode
// ──────────────────────────────────────────────────────────────────────

/// `skip` drops that many tokens from the head of the first chunk after the
/// prefill (its window becomes `offset = skip, usage = 32 − skip`), the shape
/// a splice or tombstone leaves behind. Every later chunk's rope base then
/// sits `skip` below a multiple of 32, so a selected 4-cell block straddles
/// two aligned token quads of the chunk and the resolve has to split it into
/// two groups with partial masks; the dense walk reads the same chunks by
/// their own window, so the two paths cross-check each other.
fn decode_case(g: Geom, history: usize, seed: u64, skip: usize) -> Result<()> {
    let _guard = gpu_serial();
    assert!(skip < CHUNK_SIZE, "skip stays inside the first chunk");
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping: CUDA device required");
            return Ok(());
        }
    };
    let stager = PinnedStager::new_from_device(&device);
    let inv_freq = Tensor::from_vec(
        (0..g.head_dim / 2)
            .map(|i| 1f32 / 10000f32.powf(2.0 * i as f32 / g.head_dim as f32))
            .collect::<Vec<f32>>(),
        (g.head_dim / 2,),
        &device,
    )?;
    // Sized from this case's own history — see `build_history_slot`.
    let rope_cs = compute_rope_cs(
        &inv_freq,
        MAX_BLOCKS.max(history.div_ceil(CHUNK_SIZE) + 2),
        g.head_dim,
        &device,
    )?;

    // The decode step's own token; `visible` counts it.
    let (q1, k1, v1) = make_qkv(g, 1, seed ^ 0x55, &[], &device)?;
    let q = q1.reshape((1, g.n_head, 1, g.head_dim))?;
    let k_new = k1.reshape((1, g.n_kv_head, 1, g.head_dim))?;
    let v_new = v1.reshape((1, g.n_kv_head, 1, g.head_dim))?;
    let visible = history - skip + 1;

    // Re-window the first chunk past its first `skip` tokens and shorten the
    // sequence to match, so the slot's cum positions start at prefill token
    // `skip`.
    let build = |alt: &[bool]| -> Result<(ChunkedKvBacking, KvCache)> {
        let (backing, mut cache) =
            build_history_slot(g, history, seed, alt, &rope_cs, &stager, &device)?;
        if skip > 0 {
            backing.set_block_window(0, 0, skip as u16, (CHUNK_SIZE - skip) as u32)?;
            backing.invalidate_decode_slot(0);
            cache.set_current_seq_len(history - skip)?;
            let chunks = backing
                .live_chunks_as_sealed(0)
                .expect("slot 0 holds the history");
            assert_eq!(
                (chunks[0].offset as usize, chunks[0].token_count as usize),
                (skip, CHUNK_SIZE - skip),
                "the first chunk's window is what the header is built from"
            );
            assert!(
                chunks.len() > 1 && chunks[1].token_count as usize == CHUNK_SIZE,
                "a full chunk follows, whose rope base is now off the cell grid"
            );
        }
        Ok((backing, cache))
    };

    let none_alt = vec![false; history];
    let (backing, cache) = build(&none_alt)?;

    let run = |sel: Option<&QsaSelection>, b: &ChunkedKvBacking, c: &KvCache| -> Result<Vec<f32>> {
        bits(&decode_one_slot(
            g, b, c, &q, &k_new, &v_new, &rope_cs, sel, &stager, &device,
        )?)
    };

    // 1. Dense marker and an explicit full list both reproduce no-selection.
    let base = run(None, &backing, &cache)?;
    let marked = run(Some(&dense_selection(1, &device)?), &backing, &cache)?;
    assert_same_to_bf16_ulps(&base, &marked, "dense-marked selection changed the output");
    let full = selection(&[explicit_full_row(visible)], &device)?;
    let listed = run(Some(&full), &backing, &cache)?;
    assert_same_to_bf16_ulps(&base, &listed, "select-everything changed the output");

    // 2. An unselected token cannot influence the answer.
    let (entries, selected) = subset_row(visible, seed);
    let sel = selection(&[entries], &device)?;
    // `selected` covers the decode token too; prefill token `t` is cum
    // position `t − skip` (the first `skip` tokens are outside the window and
    // never read), and an unselected position gets different values.
    let alt: Vec<bool> = (0..history)
        .map(|t| t >= skip && !selected[t - skip])
        .collect();
    assert!(alt.iter().any(|&a| a), "the subset selected everything");
    let (backing_b, cache_b) = build(&alt)?;

    let a = run(Some(&sel), &backing, &cache)?;
    let b = run(Some(&sel), &backing_b, &cache_b)?;
    assert_eq!(
        a, b,
        "an unselected token moved the answer — the selection is not being honoured"
    );

    // The comparison is discriminating: without the selection, the same two
    // histories DO differ, so the agreement above is the mask's doing.
    let a_dense = run(None, &backing, &cache)?;
    let b_dense = run(None, &backing_b, &cache_b)?;
    assert_ne!(
        a_dense, b_dense,
        "the two histories are indistinguishable even unmasked — the test proves nothing"
    );
    assert_ne!(
        a, a_dense,
        "the selection did not change the answer at all — it was ignored"
    );
    Ok(())
}

#[test]
fn decode_honours_selection_at_hd256_hpg12() -> Result<()> {
    decode_case(
        Geom {
            n_head: 24,
            n_kv_head: 2,
            head_dim: 256,
        },
        100,
        0xA1,
        0,
    )
}

/// The first chunk windowed past one token, so every chunk boundary behind it
/// is off the 4-cell grid and a selected block splits across two token quads.
#[test]
fn decode_honours_selection_off_grid_hd256() -> Result<()> {
    decode_case(FLASH_NEXT, 100, 0xA1, 1)
}

/// The same, three tokens in: the split lands at the other end of the quad.
#[test]
fn decode_honours_selection_off_grid_hd256_skip3() -> Result<()> {
    decode_case(FLASH_NEXT, 100, 0xA7, 3)
}

#[test]
fn decode_honours_selection_at_hd128_hpg2() -> Result<()> {
    decode_case(
        Geom {
            n_head: 4,
            n_kv_head: 2,
            head_dim: 128,
        },
        100,
        0xB2,
        0,
    )
}

#[test]
fn decode_honours_selection_at_hd64_hpg2() -> Result<()> {
    decode_case(
        Geom {
            n_head: 4,
            n_kv_head: 2,
            head_dim: 64,
        },
        100,
        0xC3,
        0,
    )
}

// ──────────────────────────────────────────────────────────────────────
// Prefill
// ──────────────────────────────────────────────────────────────────────

/// The same two properties over the prefill kernel, where the selection is
/// per PACKED QUERY rather than per slot.
fn prefill_case(g: Geom, history: usize, q_len: usize, seed: u64) -> Result<()> {
    let _guard = gpu_serial();
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping: CUDA device required");
            return Ok(());
        }
    };
    let stager = PinnedStager::new_from_device(&device);
    let inv_freq = Tensor::from_vec(
        (0..g.head_dim / 2)
            .map(|i| 1f32 / 10000f32.powf(2.0 * i as f32 / g.head_dim as f32))
            .collect::<Vec<f32>>(),
        (g.head_dim / 2,),
        &device,
    )?;
    let rope_cs = compute_rope_cs(&inv_freq, MAX_BLOCKS, g.head_dim, &device)?;
    let (q, k, v) = make_qkv(g, q_len, seed ^ 0x77, &[], &device)?;

    // Per-query selections over each query's own visible range.
    let mut rows_full = Vec::with_capacity(q_len);
    let mut rows_subset = Vec::with_capacity(q_len);
    // A history position is looked at if ANY query selects it.
    let mut looked_at = vec![false; history];
    for t in 0..q_len {
        let visible = history + t + 1;
        rows_full.push(explicit_full_row(visible));
        // One seed for every query, so the rows agree on which blocks they
        // ignore — otherwise 24 independent coin-flips per block leave no
        // history position unselected by ALL of them, and there is nothing to
        // vary. Real QSA rows correlate for the same reason: the indexer
        // scores the same blocks.
        let (entries, selected) = subset_row(visible, seed);
        for (p, s) in selected[..history].iter().enumerate() {
            looked_at[p] |= *s;
        }
        rows_subset.push(entries);
    }
    let alt: Vec<bool> = looked_at.iter().map(|s| !s).collect();
    assert!(
        alt.iter().any(|&a| a),
        "every history position was selected"
    );

    let sel_full = selection(&rows_full, &device)?;
    let sel_subset = selection(&rows_subset, &device)?;
    let sel_dense = dense_selection(q_len, &device)?;

    let run = |sel: Option<&QsaSelection>, alt: &[bool]| -> Result<Vec<f32>> {
        let (backing, mut cache) =
            build_history_slot(g, history, seed, alt, &rope_cs, &stager, &device)?;
        backing.ensure_for_batch_entries(&[(0, history)], q_len)?;
        let rope_offsets = Tensor::zeros(1, DType::U32, &device)?;
        let generation = stager.begin_generation();
        let out = {
            let mut caches_arr: [&mut KvCache; 1] = [&mut cache];
            paged_prefill_batched(
                None,
                &mut caches_arr[..],
                &[history],
                &q,
                &k,
                &v,
                1,
                &[q_len],
                g.n_head,
                g.n_kv_head,
                g.head_dim,
                None,
                &rope_offsets,
                &rope_cs,
                false,
                &generation,
                &std::cell::RefCell::new(None),
                sel,
            )?
        };
        bits(&out.to_owned_tensor()?)
    };

    let base = run(None, &none_alt(history))?;
    assert_eq!(
        base,
        run(Some(&sel_dense), &none_alt(history))?,
        "dense-marked selection changed the prefill output"
    );
    // Naming every visible cell attends the same set as no selection, but
    // not in the same tiles: a row's explicit list ends at its own horizon,
    // so the packed walk's last tile holds only the blocks below the block's
    // rows' horizons where the dense walk stages the full 32 columns. The
    // per-(dim, tile) int8 V scale is a max-abs over the staged columns, so
    // the two runs round V differently — by less than a step of 1/127 of a
    // ±0.5 value, which this bound sits an order above.
    const FULL_TOL: f32 = 1e-2;
    let full = run(Some(&sel_full), &none_alt(history))?;
    let (worst, at) = base
        .iter()
        .zip(&full)
        .map(|(a, b)| (a - b).abs())
        .enumerate()
        .fold(
            (0f32, 0usize),
            |(w, wi), (i, d)| if d > w { (d, i) } else { (w, wi) },
        );
    eprintln!("select-everything vs no selection: worst |diff| {worst:.3e} at element {at}");
    assert!(
        worst <= FULL_TOL,
        "select-everything vs no selection: worst |diff| {worst:.3e} at element {at}"
    );

    let a = run(Some(&sel_subset), &none_alt(history))?;
    let b = run(Some(&sel_subset), &alt)?;
    assert_eq!(
        a, b,
        "an unselected history token moved the prefill answer — the selection is not honoured"
    );
    assert_ne!(
        run(None, &none_alt(history))?,
        run(None, &alt)?,
        "the two histories are indistinguishable even unmasked — the test proves nothing"
    );
    assert_ne!(a, base, "the selection did not change the prefill answer");
    Ok(())
}

fn none_alt(history: usize) -> Vec<bool> {
    vec![false; history]
}

#[test]
fn prefill_honours_per_query_selection_at_hd256() -> Result<()> {
    prefill_case(
        Geom {
            n_head: 24,
            n_kv_head: 2,
            head_dim: 256,
        },
        100,
        24,
        0xD4,
    )
}

#[test]
fn prefill_honours_per_query_selection_at_hd128() -> Result<()> {
    prefill_case(
        Geom {
            n_head: 4,
            n_kv_head: 2,
            head_dim: 128,
        },
        100,
        24,
        0xE5,
    )
}

/// The query tokens one prefill block packs: the kernel's M rows over the
/// heads of a group (`i8_m_rows` in `paged_prefill_int8_kernel.cuh`).
fn prefill_block_tokens(g: Geom) -> usize {
    let m_rows = if g.head_dim >= 256 { 32 } else { 64 };
    m_rows / (g.n_head / g.n_kv_head)
}

/// **A block's tiles hold exactly the union of its rows' selections, and each
/// row sees exactly its own.**
///
/// The prefill kernel packs the positions its block's rows select into tiles
/// in position order — a tile is 32 selected positions, not 32 consecutive
/// ones — and masks each row to the columns it selected. Neither property
/// above can see a tile the walk left out or a column it let through for the
/// wrong row: an unselected token still cannot move the answer, and the
/// answer still differs from the dense one. Two exact oracles pin both, and
/// neither needs a reference:
///
/// * **Union-preserving edit.** Give one row of a block the entries another
///   row of the same block already holds. The union is unchanged, so the
///   tiles are unchanged; every other row's output must not move by a bit
///   (its columns, its mask, its probabilities are the same numbers in the
///   same order), and the edited row's must.
/// * **Union-changing edit.** Take a block that only some rows of a block
///   select out of every row that holds it. Each of those rows loses
///   columns, so each must move; rows of OTHER blocks share no tile and must
///   not move at all. A walk that never visited that block hands the edited
///   rows the same tiles either way and fails the first half.
///
/// Both run on the float arena the prefill writes and on the sealed quant
/// chunks production walks at depth. Bit for bit, because the grid covers
/// the SMs, so the launcher does not split the walk and there is no partial
/// merge whose order could differ.
///
/// The dense-marked twin is no longer an exact twin: a block holding a dense
/// row steps every position in consecutive tiles, and the int8 grouping of P
/// and V is per tile, so the same probabilities and values round differently
/// than they do in packed tiles. It stays as a tolerance check, at a query
/// gain that makes the softmax peaked enough for a column carrying the wrong
/// position's rope or value to move the argmax — the fault the exact oracles
/// are blind to, since they compare the kernel with itself.
///
/// Deep history with the released budget so most positions are skipped and
/// the packing crosses slices; a prefix off the tile grid so the fresh tiles
/// start mid-slice. Rows at alternating phases, so neighbouring rows of one
/// block disagree on most of their columns.
fn prefill_walk_case(g: Geom, history: usize, q_len: usize, seed: u64) -> Result<()> {
    let _guard = gpu_serial();
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping: CUDA device required");
            return Ok(());
        }
    };
    let stager = PinnedStager::new_from_device(&device);
    let inv_freq = Tensor::from_vec(
        (0..g.head_dim / 2)
            .map(|i| 1f32 / 10000f32.powf(2.0 * i as f32 / g.head_dim as f32))
            .collect::<Vec<f32>>(),
        (g.head_dim / 2,),
        &device,
    )?;
    let rope_blocks = MAX_BLOCKS.max((history + q_len).div_ceil(CHUNK_SIZE) + 2);
    let rope_cs = compute_rope_cs(&inv_freq, rope_blocks, g.head_dim, &device)?;
    let rope_offsets = Tensor::zeros(1, DType::U32, &device)?;

    // The released checkpoint's indexer budget.
    const TOP_K: usize = 2048;
    // The level the session seals at (`session.rs`).
    const SEAL_LEVEL: u8 = 5;
    // The tolerance check's query gain. `pseudo` draws Q and K in ±0.5, so a
    // raw score spreads by ~0.08 after the softmax scale and every selected
    // position weighs about the same: a column carrying the wrong position
    // moves the answer by a part in two thousand, under the int8 rounding. At
    // ×16 the spread is ~1.3, a few dozen positions carry each row, and a
    // wrong column moves the argmax and the answer with it. A power of two,
    // so the scaled BF16 query is exact and the kernel's int8 Q is the same
    // codes under a scaled scale.
    const PEAK_GAIN: f64 = 16.0;
    // The dense-marked twin's allowance. The twin and the packed walk
    // quantise the same probabilities and values in different 32-column
    // groups; the per-row int8 P and the per-group int8 V each round to half
    // a step of 1/127, and the two runs' roundings are independent. Measured
    // ~1e-3 at this gain; the bound sits an order above that and an order
    // below what a misplaced column costs.
    const TWIN_TOL: f32 = 1e-2;
    let block_tok = prefill_block_tokens(g);
    assert!(block_tok >= 2, "a block must hold two rows to disagree");
    // Split-KV engages only when the grid leaves SMs idle; this many blocks
    // covers any card, so every run walks unsplit and stores O directly.
    assert!(
        q_len.div_ceil(block_tok) * g.n_kv_head >= 256,
        "the grid must cover the SMs or the launcher splits the walk"
    );
    let row_len = g.n_head * g.head_dim;
    // The last token of each block is the one marked dense in the twin.
    let dense_at = |t: usize| t % block_tok == block_tok - 1;
    let dense: Vec<bool> = (0..q_len).map(dense_at).collect();
    // Neighbouring tokens at opposite phases: the rows of a block disagree on
    // their interior blocks and share their tails.
    let (rows, selected): (Vec<Vec<u32>>, Vec<Vec<bool>>) = (0..q_len)
        .map(|t| budget_row(history + t + 1, TOP_K, t & 1))
        .unzip();
    let block_of = entry_block;

    // The edited block: the middle one of the launch, rows `t0` (even phase)
    // and `t1 = t0 + 1` (odd phase).
    let kb = q_len.div_ceil(block_tok) / 2;
    let t0 = kb * block_tok;
    let t1 = t0 + 1;
    let in_block = |t: usize| t / block_tok == kb;

    // Union-preserving: `t1` takes every block `t0` holds. `t0` sees `t1`'s
    // positions; for a block both hold, `t1`'s own entry (its cells reach at
    // least as far) stays.
    let mut rows_add = rows.clone();
    for &e in &rows[t0] {
        if !rows[t1].iter().any(|&o| block_of(o) == block_of(e)) {
            rows_add[t1].push(e);
        }
    }
    rows_add[t1].sort_by_key(|&e| block_of(e));
    assert!(
        rows_add[t1].len() > rows[t1].len(),
        "the neighbours already select the same blocks — the phases coincide"
    );

    // Union-changing: an interior history block `t1` holds and `t0` does not,
    // taken out of every row of the block that holds it.
    let interior: Vec<u32> = rows[t1][1..rows[t1].len() - 1]
        .iter()
        .copied()
        .filter(|&e| {
            (block_of(e) + 1) * RATIO <= history
                && !rows[t0].iter().any(|&o| block_of(o) == block_of(e))
        })
        .collect();
    assert!(
        !interior.is_empty(),
        "no interior block sets the neighbours apart"
    );
    let cut = block_of(interior[interior.len() / 2]);
    let holders: Vec<usize> = (t0..t0 + block_tok)
        .filter(|&t| rows[t].iter().any(|&e| block_of(e) == cut))
        .collect();
    assert!(holders.contains(&t1) && !holders.contains(&t0));
    let mut rows_del = rows.clone();
    for &t in &holders {
        rows_del[t].retain(|&e| block_of(e) != cut);
    }

    let padded = mixed_selection(&rows, &dense, &device)?;
    let sparse = selection(&rows, &device)?;
    let sel_add = selection(&rows_add, &device)?;
    let sel_del = selection(&rows_del, &device)?;

    let (q_flat, k, v) = make_qkv_at(g, history, q_len, seed, &[], &device)?;
    let q_peak = (&q_flat * PEAK_GAIN)?;
    let (backing, mut cache) = build_history_slot(
        g,
        history,
        seed,
        &none_alt(history),
        &rope_cs,
        &stager,
        &device,
    )?;

    let run = |backing: &ChunkedKvBacking,
               cache: &mut KvCache,
               slot: usize,
               sel: &QsaSelection,
               q: &Tensor|
     -> Result<Vec<f32>> {
        // Each launch appends the chunk's tokens to the slot; cut the previous
        // launch's tail so every run prefills at the same offset.
        backing.truncate_sequence_to_tokens(slot, history)?;
        backing.ensure_for_batch_entries(&[(slot, history)], q_len)?;
        let generation = stager.begin_generation();
        let out = {
            let mut caches_arr: [&mut KvCache; 1] = [cache];
            paged_prefill_batched(
                None,
                &mut caches_arr[..],
                &[history],
                q,
                &k,
                &v,
                1,
                &[q_len],
                g.n_head,
                g.n_kv_head,
                g.head_dim,
                None,
                &rope_offsets,
                &rope_cs,
                false,
                &generation,
                &std::cell::RefCell::new(None),
                Some(sel),
            )?
        };
        let out = bits(&out.to_owned_tensor()?)?;
        assert_eq!(out.len(), q_len * row_len, "output shape");
        Ok(out)
    };
    let token = |o: &[f32], t: usize| o[t * row_len..(t + 1) * row_len].to_vec();
    let same = |a: &[f32], b: &[f32], t: usize| token(a, t) == token(b, t);

    let check_arena = |backing: &ChunkedKvBacking,
                       cache: &mut KvCache,
                       slot: usize,
                       what: &str|
     -> Result<Vec<f32>> {
        let base = run(backing, cache, slot, &sparse, &q_flat)?;
        for t in 0..q_len {
            let o = token(&base, t);
            assert!(
                o.iter().all(|x| x.is_finite()) && o.iter().any(|x| *x != 0.0),
                "{what}: query token {t} came out degenerate"
            );
        }

        let added = run(backing, cache, slot, &sel_add, &q_flat)?;
        for t in 0..q_len {
            if t == t1 {
                assert!(
                    !same(&base, &added, t),
                    "{what}: token {t1} gained its neighbour's blocks and did not move"
                );
            } else {
                assert!(
                    same(&base, &added, t),
                    "{what}: token {t} moved when token {t1} gained blocks the block's union \
                     already held — a row's mask is not exactly its own selection"
                );
            }
        }

        let removed = run(backing, cache, slot, &sel_del, &q_flat)?;
        for t in 0..q_len {
            if holders.contains(&t) {
                assert!(
                    !same(&base, &removed, t),
                    "{what}: token {t} lost block {cut} and did not move — the walk never \
                     visited it"
                );
            } else if !in_block(t) {
                assert!(
                    same(&base, &removed, t),
                    "{what}: token {t} moved when a block it does not share a tile with \
                     lost block {cut}"
                );
            }
        }

        let twin = run(backing, cache, slot, &padded, &q_peak)?;
        let packed = run(backing, cache, slot, &sparse, &q_peak)?;
        let mut worst = (0usize, 0usize, 0f32);
        for t in (0..q_len).filter(|&t| !dense_at(t)) {
            for (d, (a, b)) in token(&twin, t).iter().zip(&token(&packed, t)).enumerate() {
                let diff = (a - b).abs();
                if diff > worst.2 {
                    worst = (t, d, diff);
                }
            }
        }
        eprintln!(
            "{what}: packed walk vs dense-marked twin, worst |diff| {:.3e} at token {} dim {}",
            worst.2, worst.0, worst.1
        );
        assert!(
            worst.2 <= TWIN_TOL,
            "{what}: token {} dim {} differs by {:.3e} between the packed walk and its \
             dense-marked twin — a packed column carries the wrong position's rope or value",
            worst.0,
            worst.1,
            worst.2
        );
        Ok(base)
    };

    let base_float = check_arena(&backing, &mut cache, 0, "float arena")?;
    let (mut sealed, _formats) = seal_history(&backing, history, SEAL_LEVEL, 1, &device)?;
    check_arena(&backing, &mut sealed, 1, "sealed arena")?;

    // Positions no row selects are never staged: flip every one of them and
    // nothing may move. On the float arena only — a sealed chunk's quant
    // blocks run along its tokens, so flipping an unselected token legitimately
    // re-rounds the selected ones beside it.
    let mut looked_at = vec![false; history];
    for s in &selected {
        for (p, &x) in s[..history].iter().enumerate() {
            looked_at[p] |= x;
        }
    }
    let alt: Vec<bool> = looked_at.iter().map(|s| !s).collect();
    assert!(
        alt.iter().any(|&a| a),
        "every history position was selected"
    );
    let (backing_alt, mut cache_alt) =
        build_history_slot(g, history, seed, &alt, &rope_cs, &stager, &device)?;
    let flipped = run(&backing_alt, &mut cache_alt, 0, &sparse, &q_flat)?;
    for t in 0..q_len {
        assert!(
            same(&base_float, &flipped, t),
            "float arena: token {t} moved when unselected history flipped — the walk \
             staged a position nobody selected"
        );
    }
    Ok(())
}

/// Qwen3.8-Flash-Next's prefill geometry: two query tokens per block, so a
/// block's walk is the merge of one sparse row and its neighbour.
#[test]
fn prefill_selected_walk_matches_masked_full_walk_at_hd256() -> Result<()> {
    prefill_walk_case(
        Geom {
            n_head: 24,
            n_kv_head: 2,
            head_dim: 256,
        },
        32_768 + 100,
        256,
        0xA11,
    )
}

/// Thirty-two query tokens per block: the widest merge the walk runs, with a
/// lane's two row slots both bound.
#[test]
fn prefill_selected_walk_matches_masked_full_walk_at_hd128() -> Result<()> {
    prefill_walk_case(
        Geom {
            n_head: 4,
            n_kv_head: 2,
            head_dim: 128,
        },
        8_192 + 100,
        4_096,
        0xA12,
    )
}

// ──────────────────────────────────────────────────────────────────────
// Depth
// ──────────────────────────────────────────────────────────────────────

/// Qwen3.8-Flash-Next's own attention geometry.
///
/// `n_head / n_kv_head = 8`, which is what routes this model to the **stripe**
/// kernel — the other decode kernels take a different traversal. None of the
/// cases above lands there at `head_dim = 256` (the `hd256` case is hpg 12), so
/// the path the released checkpoint actually decodes through had no coverage of
/// its own.
const FLASH_NEXT: Geom = Geom {
    n_head: 16,
    n_kv_head: 2,
    head_dim: 256,
};

/// The selection must still be honoured **at conversational depth**.
///
/// The shallow cases above run 100 tokens of history, which is one slice-cursor
/// step and a handful of blocks. The kernel's sparse walk maps a work index to a
/// rope position and then to a slice, and every one of those mappings is
/// identity-ish at 100 tokens — a cursor bug, an off-by-one in the prefix scan,
/// or a wrong slice for a position all need real depth and many slices before
/// they can show. This runs the same sign-only property at 32K.
#[test]
fn decode_honours_selection_at_depth() -> Result<()> {
    decode_case(FLASH_NEXT, 32_768, 0xD0E9, 0)
}

/// **Decode cost must not grow with the cache.**
///
/// This is the property QSA exists to provide and the one the engine was not
/// getting: the selection keeps a fixed budget of positions however deep the
/// cache is, so a decode step should cost the same at 128K as at 8K. Measured
/// end to end on the real checkpoint it did not — `decode:kernel` went
/// 112 ms → 28,111 ms across 8K→128K, 251× for 16× depth, while the span that
/// *computes* the selection stayed flat.
///
/// Timing the kernel directly is the short iteration: a full-model run pays
/// minutes of checkpoint load and prefill to reach one data point, and the
/// question is entirely about one launch.
///
/// Prints ms per call against depth. Flat is the pass condition; the assertion
/// is deliberately loose because this is a benchmark on a shared device, and
/// what it is guarding against is a return to super-linear growth, not a few
/// percent of noise.
#[test]
#[ignore = "microbenchmark; run with: cargo test --release --features cuda \
            -p candle-transformers --test qsa_kernel_tests \
            bench_decode_cost_vs_depth -- --ignored --nocapture"]
fn bench_decode_cost_vs_depth() -> Result<()> {
    use std::time::Instant;

    let _guard = gpu_serial();
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping: CUDA device required");
            return Ok(());
        }
    };
    let g = FLASH_NEXT;
    let stager = PinnedStager::new_from_device(&device);
    let inv_freq = Tensor::from_vec(
        (0..g.head_dim / 2)
            .map(|i| 1f32 / 10000f32.powf(2.0 * i as f32 / g.head_dim as f32))
            .collect::<Vec<f32>>(),
        (g.head_dim / 2,),
        &device,
    )?;
    // Rope table sized for the deepest case below, for the same reason the
    // backing is: the 100-token constant covers 2,048 positions.
    let rope_blocks = MAX_BLOCKS.max(262_144usize.div_ceil(CHUNK_SIZE) + 2);
    let rope_cs = compute_rope_cs(&inv_freq, rope_blocks, g.head_dim, &device)?;
    let (q1, k1, v1) = make_qkv(g, 1, 0x9001, &[], &device)?;
    let q = q1.reshape((1, g.n_head, 1, g.head_dim))?;
    let k_new = k1.reshape((1, g.n_kv_head, 1, g.head_dim))?;
    let v_new = v1.reshape((1, g.n_kv_head, 1, g.head_dim))?;

    // The released checkpoint's indexer budget.
    const TOP_K: usize = 2048;
    const WARM: usize = 3;
    const ITERS: usize = 20;
    // Every launch against one header appends the step's token to the write
    // slice; a column's launches must stay inside it.
    const _: () = assert!(WARM + ITERS < CHUNK_SIZE);
    // The level the session seals at (`session.rs`); the cache behind the live
    // chunk is quantized at this level for the whole of a conversation.
    const SEAL_LEVEL: u8 = 5;
    let mut first: Option<f64> = None;
    let mut worst = 0f64;
    let mut deepest_frac = 1f64;
    let mut sealed_first: Option<f64> = None;
    let mut sealed_worst = 0f64;
    // The dense column is the control, measured in the same run on the same
    // device: the launcher's unselected route — its own split heuristic over
    // the whole cache, the shape a short-context decode runs. A selection that
    // is genuinely driving the iteration is both far cheaper than it and far
    // flatter across depth; one that is merely *masking* a dense walk tracks it.
    //
    // The BF16 columns measure the kernel's float path — the arena format the
    // prefill writes and the live chunk keeps. The sealed column measures what
    // production decode actually walks at depth: the same history quantized
    // through the session's compression policy, staged as quant block runs and
    // read through as int8, which is a different kernel path with its own cost.
    //
    // The slot header is built once per column and the timed loop launches
    // the wrapper alone, so the columns are the decode path — kernel, split
    // combine, commit — and not the harness's host-side slice-table build,
    // which grows with depth (4,096 kvhead records at 128K) and would swamp a
    // kernel that takes a tenth of a millisecond.
    println!(
        "\n  depth      selected   sparse ms   dense ms   sealed-C{SEAL_LEVEL} ms   \
         sparse vs 8K   sparse/dense   sealed vs 8K"
    );
    let mut sealed_formats = Vec::new();
    for &history in &[8_192usize, 32_768, 131_072] {
        let visible = history + 1;
        let none_alt = vec![false; history];
        let (backing, cache) =
            build_history_slot(g, history, 0x9001, &none_alt, &rope_cs, &stager, &device)?;
        let (entries, selected) = budget_row(visible, TOP_K, 0);
        let n_sel = selected.iter().filter(|&&s| s).count();
        let sel = selection(&[entries], &device)?;

        let time_it = |c: &KvCache, s: Option<&QsaSelection>| -> Result<f64> {
            let headers = SlotHeaders::build(g, &backing, c, &stager, &device)?;
            // Warm: the first launch pays module load.
            for _ in 0..WARM {
                headers.decode(g, &q, &k_new, &v_new, &rope_cs, s)?;
            }
            device.synchronize()?;
            let t0 = Instant::now();
            for _ in 0..ITERS {
                headers.decode(g, &q, &k_new, &v_new, &rope_cs, s)?;
            }
            device.synchronize()?;
            Ok(t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64)
        };
        let ms = time_it(&cache, Some(&sel))?;
        let ms_dense = time_it(&cache, None)?;
        // Sealing records slot 0's chunks, so it runs after the float columns
        // have finished with them.
        let (sealed, formats) = seal_history(&backing, history, SEAL_LEVEL, 1, &device)?;
        let ms_sealed = time_it(&sealed, Some(&sel))?;
        sealed_formats = formats;

        let base = *first.get_or_insert(ms);
        let rel = ms / base;
        worst = worst.max(rel);
        let frac = ms / ms_dense;
        deepest_frac = frac;
        let sealed_rel = ms_sealed / *sealed_first.get_or_insert(ms_sealed);
        sealed_worst = sealed_worst.max(sealed_rel);
        println!(
            "  {history:>7}  {n_sel:>9}  {ms:>9.3}  {ms_dense:>9.3}  {ms_sealed:>12.3}  \
             {rel:>12.2}×  {frac:>13.3}  {sealed_rel:>12.2}×"
        );
    }
    println!("  sealed formats (K, V) → bands: {sealed_formats:?}");
    // The selected walk is O(selection): the kernel enumerates the row's
    // entries, resolves each run to its 32-token tile, and while a tile's K/V
    // quads are read straight into registers the staging warp resolves the
    // next tile's descriptors and prefetches its spans into L2. What depth
    // still costs is the working set: 2,048 tiles scattered over 16 MB sit in
    // L2, over 268 MB they do not, and each tile's first touch is a DRAM round
    // trip — under ncu that is 11 µs warm against 19 µs cold at every rung,
    // and this harness reads 0.023–0.031 ms across 8K→128K (kernel + split
    // combine + commit, pipelined). The guards hold the two properties that
    // make the selection worth having — cost bounded across depth, and well
    // under the unselected walk at the deepest rung — with headroom for the
    // machine's own variance (a single run on the same build has swung by a
    // quarter).
    assert!(
        worst < 2.5,
        "decode cost grew {worst:.2}× across 16× depth at a fixed selection budget — \
         the sparse walk has regressed to scanning the cache"
    );
    assert!(
        deepest_frac < 0.9,
        "at the deepest rung the selection cost {deepest_frac:.3} of a dense walk — \
         a selection that does not beat walking everything is not being used to \
         drive the iteration"
    );
    assert!(
        sealed_worst < 2.5,
        "sealed-chunk decode cost grew {sealed_worst:.2}× across 16× depth at a fixed \
         selection budget — the quant staging path has regressed to scanning the cache"
    );
    Ok(())
}

/// **Prefill cost must not grow with the cache either.**
///
/// [`bench_decode_cost_vs_depth`] holds for one query. A prompt chunk is the
/// same question asked `q_len` times at once: every packed query keeps a fixed
/// budget of positions, so one prefill launch over a chunk against 128K of
/// history should cost what it costs against 8K. Measured end to end on the
/// real checkpoint it did not — `prefill:kernel` went 2,370 → 9,821 →
/// 40,201 ms across 8K → 16K → 32K of prompt, 4.1× per doubling — and the
/// speculative verify, a multi-token step through the same kernel, carried the
/// only depth term left in decode once the decode kernel was flat.
///
/// The chunk is the engine's own prefill width; the selection is the released
/// checkpoint's budget, spread over the history the way [`budget_row`] spreads
/// it, so a query tile's selected positions land in scattered 32-token tiles
/// rather than clustered ones — the least favourable case for a walk that
/// visits only the tiles a tile of queries selects. The dense column is the
/// same launch with no selection: a kernel that applies the selection as a
/// mask over a full causal walk tracks it; one that walks the selection is far
/// cheaper and flat.
///
/// The timed call is the prefill wrapper — slot headers, kernel, split
/// combine, fresh K/V writeback — which is what a prefill layer launches.
#[test]
#[ignore = "microbenchmark; run with: cargo test --release --features cuda \
            -p candle-transformers --test qsa_kernel_tests \
            bench_prefill_cost_vs_depth -- --ignored --nocapture"]
fn bench_prefill_cost_vs_depth() -> Result<()> {
    use std::time::Instant;

    let _guard = gpu_serial();
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping: CUDA device required");
            return Ok(());
        }
    };
    let g = FLASH_NEXT;
    let stager = PinnedStager::new_from_device(&device);
    let inv_freq = Tensor::from_vec(
        (0..g.head_dim / 2)
            .map(|i| 1f32 / 10000f32.powf(2.0 * i as f32 / g.head_dim as f32))
            .collect::<Vec<f32>>(),
        (g.head_dim / 2,),
        &device,
    )?;
    // One prompt chunk, at the width the engine prefills in.
    const Q_LEN: usize = 2048;
    // The released checkpoint's indexer budget.
    const TOP_K: usize = 2048;
    const WARM: usize = 2;
    const ITERS: usize = 5;
    let deepest = 131_072usize;
    let rope_blocks = MAX_BLOCKS.max((deepest + Q_LEN).div_ceil(CHUNK_SIZE) + 2);
    let rope_cs = compute_rope_cs(&inv_freq, rope_blocks, g.head_dim, &device)?;
    let rope_offsets = Tensor::zeros(1, DType::U32, &device)?;

    let mut first: Option<f64> = None;
    let mut worst = 0f64;
    let mut deepest_frac = 1f64;
    println!(
        "\n  history   chunk   selected/query   sparse ms   dense ms   sparse vs 8K   sparse/dense"
    );
    for &history in &[8_192usize, 32_768, deepest] {
        let (backing, mut cache) = build_history_slot(
            g,
            history,
            0x9002,
            &none_alt(history),
            &rope_cs,
            &stager,
            &device,
        )?;
        // The chunk's own tokens follow the history, so the values are the
        // ones a single-call build of `history + Q_LEN` would have produced.
        let (q, k, v) = make_qkv_at(g, history, Q_LEN, 0x9002, &[], &device)?;
        // Neighbouring tokens at opposite phases: the block walks two budgets'
        // worth of blocks, the way it does under the indexer.
        let mut rows = Vec::with_capacity(Q_LEN);
        let mut n_sel = 0;
        for t in 0..Q_LEN {
            let (entries, selected) = budget_row(history + t + 1, TOP_K, t & 1);
            n_sel = selected.iter().filter(|&&s| s).count();
            rows.push(entries);
        }
        let sel = selection(&rows, &device)?;

        let mut time_it = |s: Option<&QsaSelection>| -> Result<f64> {
            let mut launch = |s: Option<&QsaSelection>| -> Result<()> {
                // Each launch appends the chunk's tokens to the slot; cut the
                // previous launch's tail so every run prefills at the same
                // offset. Both walks pay it, in the timed loop.
                backing.truncate_sequence_to_tokens(0, history)?;
                backing.ensure_for_batch_entries(&[(0, history)], Q_LEN)?;
                let generation = stager.begin_generation();
                let mut caches_arr: [&mut KvCache; 1] = [&mut cache];
                paged_prefill_batched(
                    None,
                    &mut caches_arr[..],
                    &[history],
                    &q,
                    &k,
                    &v,
                    1,
                    &[Q_LEN],
                    g.n_head,
                    g.n_kv_head,
                    g.head_dim,
                    None,
                    &rope_offsets,
                    &rope_cs,
                    false,
                    &generation,
                    &std::cell::RefCell::new(None),
                    s,
                )?;
                Ok(())
            };
            for _ in 0..WARM {
                launch(s)?;
            }
            device.synchronize()?;
            let t0 = Instant::now();
            for _ in 0..ITERS {
                launch(s)?;
            }
            device.synchronize()?;
            Ok(t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64)
        };
        let ms = time_it(Some(&sel))?;
        let ms_dense = time_it(None)?;

        let base = *first.get_or_insert(ms);
        let rel = ms / base;
        worst = worst.max(rel);
        let frac = ms / ms_dense;
        deepest_frac = frac;
        println!(
            "  {history:>7}  {Q_LEN:>6}  {n_sel:>14}  {ms:>9.1}  {ms_dense:>9.1}  \
             {rel:>12.2}×  {frac:>13.3}"
        );
    }
    assert!(
        worst < 2.5,
        "prefill cost grew {worst:.2}× across 16× depth at a fixed selection budget — \
         the prefill walk is scanning the cache"
    );
    assert!(
        deepest_frac < 0.9,
        "at the deepest rung a selected prefill cost {deepest_frac:.3} of a dense one — \
         the selection is a mask over a full walk, not the walk"
    );
    Ok(())
}
