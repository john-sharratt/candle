//! CUDA wrappers for the Gated DeltaNet kernels, parity-locked to the
//! sequential reference in [`super::delta_net`].
//!
//! A short span — a decode row, a verify block, a short prompt — reads its
//! state and writes a timeline ([`super::timeline`]); [`delta_net_fold`]
//! applies the rows a step keeps, in place. A long span writes its advanced
//! state to a separate buffer. All tensors are F32 and contiguous; dims are
//! validated here, and every launcher reports a refused shape.

use candle::cuda_backend::cudarc::cublas::sys as cublas;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::quantized::cuda::{produce_q8a128, Q8a128Operand};
use candle::quantized::SumScale;
use candle::{DType, Device, Error, LiveTensor, Result, Storage, Tensor};
pub use candle_kernels::delta_net::DELTA_NET_PREFILL_DIM;
use candle_kernels::delta_net::{
    run_delta_net_batch_ptrs, run_delta_net_conv_decode_f32, run_delta_net_conv_prefill_f32,
    run_delta_net_decode_step_f32, run_delta_net_fold_f32, run_delta_net_norm_gate_f32,
    run_delta_net_prefill_intra_f32, run_delta_net_prefill_state_f32, run_delta_net_short_span_f32,
    DELTA_NET_MAX_SPANS, DELTA_NET_PREFILL_CHUNK, DELTA_NET_PREFILL_CONV_MAX,
};
pub use candle_kernels::delta_net::{DELTA_NET_SHORT_SPAN_CONV, DELTA_NET_SHORT_SPAN_ROWS};

use super::mix::{DeltaNetLayerTable, DeltaNetSeq, DeltaNetSpanTable, DeltaNetState, SeqSpan};
use super::state_store::RecurrentStateStore;
use super::timeline::{TimelineLayout, TIMELINE_ROWS};
use super::types::DeltaNetDims;
use super::ZGate;
use crate::models::held_tensor::held_empty;
use crate::models::wave_buffers::{upload_into, wave_from_vec_ticketed};
use candle::wave_provenance::WaveTicket;
use std::ffi::c_void;

/// The wave tensors every fused DeltaNet kernel reads through strides, plus
/// the geometry derived from them — validated once per layer, not once per
/// span.
///
/// `conved` is the conv kernels' output `[T, conv_dim]` — post-SiLU, with
/// the leading `2·h_k·d` Q|K columns already l2-normed by the conv epilogue.
/// It is the *only* operand buffer: q, k and v are strided views of it (V
/// head `h` reads K head `h % h_k`; q is scaled by `q_scale` inside the
/// kernels, so no scaled copy of q ever exists). `alpha`/`blin` are the raw
/// gate projections `[T, h_v]` — the gates are computed in-kernel with the
/// reference softplus/sigmoid. `o` is the whole-wave output `[T, h_v·d]`
/// each span's kernels write their own rows of, so a multi-sequence wave
/// needs no concatenation.
pub struct DeltaNetFused<'a, 'w> {
    pub conved: &'a LiveTensor<'w>,
    pub alpha: &'a LiveTensor<'w>,
    pub blin: &'a LiveTensor<'w>,
    pub dt_bias: &'a Tensor,
    pub a: &'a Tensor,
    pub o: &'a LiveTensor<'w>,
    pub q_scale: f32,
}

/// The per-launch pointer/geometry bundle [`DeltaNetFused::resolve`] produces.
struct FusedPtrs {
    qk_p: u64,
    v_p: u64,
    alpha_p: u64,
    blin_p: u64,
    dt_p: u64,
    a_p: u64,
    o_p: u64,
    h_v: usize,
    h_k: usize,
    d: usize,
    conv_dim: usize,
}

impl<'a, 'w> DeltaNetFused<'a, 'w> {
    /// Validate the wave tensors against the state's geometry and resolve the
    /// base addresses for a span starting at `start` — an offset into every
    /// stride, never a copy.
    ///
    /// `h_k` is derived, not passed: the conv row is `[Q|K|V]` with Q and K
    /// `h_k·d` each and V `h_v·d`, so `h_k = (conv_dim/d − h_v) / 2`.
    fn resolve_wave(&self) -> Result<FusedPtrs> {
        let (t, conv_dim) = self.conved.dims2()?;
        // Derived from the WAVE tensors rather than from a carried state: the
        // spans that share this launch each bring their own state, so no single
        // one of them can be the source of truth for the geometry they all use.
        let (ta, h_v) = self.alpha.dims2()?;
        if ta != t || h_v == 0 {
            candle::bail!(
                "delta_net cuda: alpha must be [{t}, h_v], got {:?}",
                self.alpha.dims()
            );
        }
        let (to, o_cols) = self.o.dims2()?;
        if to != t || !o_cols.is_multiple_of(h_v) {
            candle::bail!(
                "delta_net cuda: o must be [{t}, h_v·d], got {:?}",
                self.o.dims()
            );
        }
        let d = o_cols / h_v;
        if !conv_dim.is_multiple_of(d) || conv_dim <= h_v * d {
            candle::bail!(
                "delta_net cuda: conv row of {conv_dim} cannot be [Q|K|V] with \
                 h_v = {h_v}, d = {d}"
            );
        }
        let qk_heads = conv_dim / d - h_v;
        if !qk_heads.is_multiple_of(2) {
            candle::bail!(
                "delta_net cuda: conv row of {conv_dim} leaves {qk_heads} Q|K heads, \
                 which do not split into equal Q and K"
            );
        }
        let h_k = qk_heads / 2;
        if !h_v.is_multiple_of(h_k) {
            candle::bail!("delta_net cuda: h_v {h_v} must be a multiple of h_k {h_k}");
        }
        if self.blin.dims2()? != (t, h_v) {
            candle::bail!("delta_net cuda: beta_lin must be [{t}, {h_v}]");
        }
        if self.dt_bias.dims1()? != h_v || self.a.dims1()? != h_v {
            candle::bail!("delta_net cuda: dt_bias/a must be [{h_v}]");
        }
        // Bases of the packed wave buffers. q/k/v are all views of the conv
        // output — Q|K at column 0, V at the trailing block — and each kernel
        // adds its own span's row offset from the table.
        let conved_p = f32_ptr(self.conved, "conved")?;
        let v_off = conv_dim - h_v * d;
        Ok(FusedPtrs {
            qk_p: conved_p,
            v_p: conved_p + v_off as u64 * std::mem::size_of::<f32>() as u64,
            alpha_p: f32_ptr(self.alpha, "alpha")?,
            blin_p: f32_ptr(self.blin, "beta_lin")?,
            dt_p: f32_ptr(self.dt_bias, "dt_bias")?,
            a_p: f32_ptr(self.a, "a")?,
            o_p: f32_ptr(self.o, "o")?,
            h_v,
            h_k,
            d,
            conv_dim,
        })
    }
}

const MAX_HEAD_DIM: usize = 256;

/// Rows of a decode or short-span pointer table: conv tail, state, timeline.
/// The kernels read the first two and write only the third, so a step that
/// fails leaves every state as it entered.
const TABLE_ROWS: usize = 3;

/// Rows of a long-span pointer table: conv tail in, conv tail out, state in,
/// state out — the entering buffers and the prefill scratch the advance is
/// written to, which `commit_wave` exchanges in.
const LONG_TABLE_ROWS: usize = 4;

/// Rows of a fold table: conv tail, state, timeline, and the rows to fold.
const FOLD_TABLE_ROWS: usize = 4;

/// Device address of `t`'s first element, checking dtype and contiguity.
///
/// Public because it is generic tensor plumbing rather than anything about the
/// mixer: the MTP head's kernel wrapper builds the same pointer tables from the
/// same checks, and a second transcription of "resolve, verify, hand the
/// address to a kernel" is a second place for the verification to be forgotten.
pub fn typed_ptr(t: &LiveTensor<'_>, dtype: DType, what: &str) -> Result<u64> {
    if t.dtype() != dtype {
        candle::bail!(
            "delta_net cuda: {what} must be {dtype:?}, got {:?}",
            t.dtype()
        );
    }
    if !t.is_contiguous() {
        candle::bail!("delta_net cuda: {what} must be contiguous");
    }
    let (storage, layout) = t.storage_and_layout();
    let cuda = match &*storage {
        Storage::Cuda(c) => c,
        _ => candle::bail!("delta_net cuda: {what} must be a CUDA tensor"),
    };
    let dev = match t.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("delta_net cuda: {what} must live on a CUDA device"),
    };
    let stream = dev.cuda_stream();
    let offset = layout.start_offset();
    // The guard is dropped before returning, as in every other launcher here:
    // the address stays valid because the storage outlives the launch, and the
    // stream ordering the guard would record is already implied by every op on
    // this device sharing one stream.
    let ptr = match dtype {
        DType::F32 => {
            let view = cuda.as_cuda_slice::<f32>()?.slice(offset..);
            let (ptr, _guard) = view.device_ptr(&stream);
            ptr
        }
        DType::I64 => {
            let view = cuda.as_cuda_slice::<i64>()?.slice(offset..);
            let (ptr, _guard) = view.device_ptr(&stream);
            ptr
        }
        DType::U32 => {
            let view = cuda.as_cuda_slice::<u32>()?.slice(offset..);
            let (ptr, _guard) = view.device_ptr(&stream);
            ptr
        }
        _ => candle::bail!("delta_net cuda: no pointer accessor for {dtype:?}"),
    };
    Ok(ptr)
}

pub fn f32_ptr(t: &LiveTensor<'_>, what: &str) -> Result<u64> {
    typed_ptr(t, DType::F32, what)
}

/// `ptr`, checked 16-byte aligned: the prefill passes read the state, the
/// Q|K rows and the scan transients four floats at a time, and a misaligned
/// base would fault inside a kernel that cannot say which operand it was.
pub fn aligned16(ptr: u64, what: &str) -> Result<u64> {
    if !ptr.is_multiple_of(16) {
        candle::bail!("delta_net cuda: {what} at {ptr:#x} is not 16-byte aligned");
    }
    Ok(ptr)
}

/// The whole forward's decode pointer table: for every DeltaNet layer, every
/// decode sequence's conv-tail and state base pointers, plus the wave rows —
/// ONE host upload, built before the layer sweep. Mid-sweep uploads are the
/// thing this exists to avoid: a host→device copy syncs the stream, and 24 of
/// them per token would serialise the launch pipeline.
///
/// **Valid for this forward only, and rebuilt every forward.** Compaction moves
/// state slots between forwards, so a table kept into the next forward could
/// name a slot another sequence now holds. The buffers themselves do not move
/// during a forward: the store holds them, and each tensor's anchor keeps its
/// state-arena slot held while anything can read it. Built once per forward and
/// read by **every** layer, which is what [`LayerPhase::Forward`] exists for: a
/// buffer carved from the attention span is reclaimed when layer 0's guard
/// drops, and layer 1 would overwrite the table it is still reading. The `'w`
/// ties it to that guard.
pub struct DeltaNetWaveTable<'w> {
    /// `[n_dn_layers, 3, n_decode]` I64 — per layer, each decode sequence's
    /// conv tail, state and timeline.
    ptrs: LiveTensor<'w>,
    /// `[n_decode]` U32 wave rows, shared by every layer.
    rows: LiveTensor<'w>,
    /// Trunk layer index per table row, in slot order.
    layers: Vec<usize>,
}

impl DeltaNetWaveTable<'_> {
    /// This layer's slice, as the view type the mixer consumes.
    pub fn layer_slice(&self, layer_idx: usize) -> Result<DeltaNetLayerTable<'_>> {
        let Some(ord) = self.layers.iter().position(|&l| l == layer_idx) else {
            candle::bail!("delta_net cuda: layer {layer_idx} has no slot in the decode table");
        };
        let n = self.rows.dim(0)?;
        Ok(DeltaNetLayerTable {
            // A narrow on the leading axis of a contiguous tensor is itself
            // contiguous, so the reshape is a view — no copy per layer.
            ptrs: self.ptrs.narrow(0, ord, 1)?.reshape((TABLE_ROWS, n))?,
            rows: self.rows.clone(),
        })
    }
}

/// Build the forward's decode table from the wave's spans and state stores —
/// `None` when the wave carries no decode span. Called once per forward,
/// before any launch, so its two small uploads land on an empty queue.
/// `forward` names the [`LayerPhase::Forward`] generation the two uploads land
/// on — held by the caller across the whole layer sweep, which is exactly as
/// long as the table is read for (`WaveBuffer::DeltaNetDecodeTable`).
pub fn build_wave_table<'w>(
    spans: &[SeqSpan],
    stores: &[&mut RecurrentStateStore],
    forward: Option<WaveTicket>,
) -> Result<Option<DeltaNetWaveTable<'w>>> {
    let decode: Vec<usize> = (0..spans.len()).filter(|&i| spans[i].len == 1).collect();
    if decode.is_empty() {
        return Ok(None);
    }
    let Some(first) = stores.first() else {
        candle::bail!("delta_net cuda: spans without stores");
    };
    let layers: Vec<usize> = first.recurrent_layer_indices().collect();
    let n = decode.len();
    // Three rows per layer: the conv tail and the state the step reads, and the
    // timeline it writes.
    //
    // Read off each store's slots (`decode_table_addresses`), which marks
    // nothing: this table covers every recurrent layer, but a sweep may run
    // only part of the stack, and a layer that never ran must not be left
    // holding a pending timeline. Each layer records itself when it actually
    // runs.
    //
    // A store answers its layers in slot order, so every store must carry the
    // same schedule as the first for its rows to land under the right layer.
    let schedule = first.schedule_hash();
    let mut ptrs: Vec<i64> = vec![0; layers.len() * TABLE_ROWS * n];
    for (k, &i) in decode.iter().enumerate() {
        if stores[i].schedule_hash() != schedule {
            candle::bail!(
                "delta_net cuda: span {i}'s recurrent store has schedule {:#x}, the \
                 wave's first has {schedule:#x} — its rows would land under other layers",
                stores[i].schedule_hash()
            );
        }
        for (ord, addrs) in stores[i].decode_table_addresses()?.enumerate() {
            for (row, addr) in addrs.into_iter().enumerate() {
                ptrs[(ord * TABLE_ROWS + row) * n + k] = addr as i64;
            }
        }
    }
    let rows: Vec<u32> = decode.iter().map(|&i| spans[i].start as u32).collect();
    // Onto the forward-phase span. There is no device operand to inherit from —
    // both tables are built on the host — so the generation is named directly.
    //
    // The recurrent state cannot serve as an anchor: it is arena memory tagged
    // `LeaseOrigin::Foreign`, which carries no ticket by design, and anchoring
    // on it yields a driver allocation while implying otherwise.
    let dev = stores[decode[0]].layer_state(layers[0])?.s.device().clone();
    Ok(Some(DeltaNetWaveTable {
        ptrs: wave_from_vec_ticketed(ptrs, (layers.len(), TABLE_ROWS, n), &dev, forward)?,
        rows: wave_from_vec_ticketed(rows, n, &dev, forward)?,
        layers,
    }))
}

/// A single layer's table built straight from the mixer's spans — the form
/// the reference path and unit tests use when no wave table was supplied.
/// Same layout, same kernels; the upload merely happens closer to the launch.
/// `ticket` is the open layer phase: the table is rebuilt per layer and dead
/// once its launches are issued, so a per-layer span is exactly its lifetime.
pub fn build_layer_table(
    seqs: &[DeltaNetSeq<'_>],
    ticket: Option<WaveTicket>,
) -> Result<DeltaNetLayerTable<'static>> {
    let decode: Vec<&DeltaNetSeq<'_>> = seqs.iter().filter(|s| s.len == 1).collect();
    if decode.is_empty() {
        candle::bail!("delta_net cuda: no decode spans to table");
    }
    let n = decode.len();
    // Same three rows as the wave table, taken from each span's own state and
    // timeline.
    let mut ptrs: Vec<i64> = Vec::with_capacity(TABLE_ROWS * n);
    for s in &decode {
        ptrs.push(f32_ptr(&s.state.conv_tail, "conv tail")? as i64);
    }
    for s in &decode {
        ptrs.push(f32_ptr(&s.state.s, "state")? as i64);
    }
    for s in &decode {
        ptrs.push(timeline_ptr(s)? as i64);
    }
    let rows: Vec<u32> = decode.iter().map(|s| s.start as u32).collect();
    let dev = decode[0].state.s.device().clone();
    Ok(DeltaNetLayerTable {
        ptrs: wave_from_vec_ticketed(ptrs, (TABLE_ROWS, n), &dev, ticket)?,
        rows: wave_from_vec_ticketed(rows, n, &dev, ticket)?,
    })
}

/// The address of a short span's timeline, 16-byte aligned (the kernels write
/// its rows four floats at a time).
fn timeline_ptr(s: &DeltaNetSeq<'_>) -> Result<u64> {
    let Some(tl) = s.write.timeline() else {
        candle::bail!(
            "delta_net cuda: a {}-row span was handed a state to write, where every span \
             of at most {TIMELINE_ROWS} rows writes a timeline",
            s.len
        );
    };
    aligned16(f32_ptr(tl, "timeline")?, "timeline")
}

/// The multi-row spans of a wave, tabled for one launch per kernel, split by
/// length: `(short, long)` — spans of at most [`TIMELINE_ROWS`] rows for the
/// short-span scan, longer ones for the conv + scan pair. Either is `None` when
/// the wave carries no such span, so the caller issues no empty launch.
///
/// The pointer rows are built as [`build_layer_table`] builds them, and for the
/// same reason: a span's conv tail and state live in that sequence's own
/// allocation, so batching needs their addresses where the kernel can read
/// them. `anchor` names the wave whose transient tier the uploads land on — the
/// packed activation the spans index into, which is carved from that same tier.
/// It is read for its provenance only; its shape and dtype are irrelevant.
///
/// **One upload of pointers and one of extents for both tables**, each table a
/// view into them — the carves the tier prices for a wave's span tables
/// (`WaveBuffer::DeltaNetLiveSpanPtrs` / `…Extents`, four pointers and two
/// extents a span, which covers a short span's three pointers too). Two tables
/// uploaded apart would be four carves against two priced, and the cursor's
/// alignment of each would overrun the price.
#[allow(clippy::type_complexity)]
pub fn build_span_tables<'w>(
    seqs: &[DeltaNetSeq<'_>],
    anchor: &LiveTensor<'w>,
) -> Result<(Option<DeltaNetSpanTable<'w>>, Option<DeltaNetSpanTable<'w>>)> {
    let short: Vec<&DeltaNetSeq<'_>> = seqs
        .iter()
        .filter(|s| s.len > 1 && s.len <= TIMELINE_ROWS)
        .collect();
    let long: Vec<&DeltaNetSeq<'_>> = seqs.iter().filter(|s| s.len > TIMELINE_ROWS).collect();
    if short.is_empty() && long.is_empty() {
        return Ok((None, None));
    }
    let mut ptrs = short_rows(&short)?;
    let short_ptrs = ptrs.len();
    ptrs.extend(long_rows(&long)?);
    let mut extents = span_extents(&short);
    extents.extend(span_extents(&long));
    // Uploaded onto the wave's transient tier, named by `anchor`. These tables
    // are rebuilt every wave — the pointers and extents are what changes — so
    // they are transients, and `Tensor::from_vec` would put them in driver
    // memory the reservation does not cover.
    //
    // **Not `spans[0].state.s`**, which was tried: the recurrent state is arena
    // memory tagged `LeaseOrigin::Foreign`, and `Foreign` carries no ticket by
    // design — a KV arena slot is not a scratch space, and its `'static` handle
    // must not be able to hand out wave-scoped memory.
    let n_ptrs = ptrs.len();
    let n_ext = extents.len();
    let ptrs = anchor.from_vec_beside(ptrs, n_ptrs)?;
    let extents = anchor.from_vec_beside(extents, n_ext)?;
    let table = |group: &[&DeltaNetSeq<'_>],
                 ptr_at: usize,
                 rows: usize,
                 ext_at: usize|
     -> Result<Option<DeltaNetSpanTable<'w>>> {
        let n = group.len();
        if n == 0 {
            return Ok(None);
        }
        Ok(Some(DeltaNetSpanTable {
            ptrs: ptrs.narrow(0, ptr_at, rows * n)?.reshape((rows, n))?,
            spans: extents.narrow(0, ext_at, 2 * n)?.reshape((2, n))?,
            n,
            max_len: group.iter().map(|s| s.len).max().unwrap_or(0),
        }))
    };
    Ok((
        table(&short, 0, TABLE_ROWS, 0)?,
        table(&long, short_ptrs, LONG_TABLE_ROWS, 2 * short.len())?,
    ))
}

/// The state's shape, checked square per head: the geometry a launch shares is
/// derived from the wave tensors (`DeltaNetFused::resolve_wave`), so each span's
/// own state is checked here — a span carrying a differently shaped state would
/// otherwise be read with the cohort's strides.
fn checked_state_dims(s: &Tensor) -> Result<(usize, usize, usize)> {
    let (h_v, d_v, d_k) = s.dims3()?;
    if d_v != d_k {
        candle::bail!("delta_net cuda: state must be square per head, got ({d_v}, {d_k})");
    }
    Ok((h_v, d_v, d_k))
}

/// The short-span table's pointer rows: `[tail, state, timeline]` per span.
fn short_rows(spans: &[&DeltaNetSeq<'_>]) -> Result<Vec<i64>> {
    let n = spans.len();
    let mut ptrs: Vec<i64> = Vec::with_capacity(TABLE_ROWS * n);
    for s in spans {
        ptrs.push(f32_ptr(&s.state.conv_tail, "conv tail")? as i64);
    }
    for s in spans {
        checked_state_dims(&s.state.s)?;
        ptrs.push(aligned16(f32_ptr(&s.state.s, "state")?, "state")? as i64);
    }
    for s in spans {
        ptrs.push(timeline_ptr(s)? as i64);
    }
    Ok(ptrs)
}

/// The long-span table's pointer rows: `[tail in, tail out, state in, state
/// out]` per span, the outs in the prefill scratch.
fn long_rows(spans: &[&DeltaNetSeq<'_>]) -> Result<Vec<i64>> {
    let n = spans.len();
    let outs = spans
        .iter()
        .map(|s| {
            s.write.state().ok_or_else(|| {
                Error::Msg(format!(
                    "delta_net cuda: a {}-row span was handed a timeline, where every span \
                     longer than {TIMELINE_ROWS} rows writes a state",
                    s.len
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut ptrs: Vec<i64> = Vec::with_capacity(LONG_TABLE_ROWS * n);
    for s in spans {
        ptrs.push(f32_ptr(&s.state.conv_tail, "conv tail in")? as i64);
    }
    for (s, out) in spans.iter().zip(&outs) {
        // The prefill conv's blocks that read the entering tail run alongside
        // the one that writes the advance, so the two must be distinct buffers.
        // Checked here, where the table is assembled and both are in hand,
        // rather than inside a kernel that cannot report it.
        if s.state.conv_tail.same_storage(&out.conv_tail) {
            candle::bail!(
                "delta_net cuda: a prefill span's entering and advanced conv tails \
                 must be separate buffers"
            );
        }
        ptrs.push(f32_ptr(&out.conv_tail, "conv tail out")? as i64);
    }
    for (s, out) in spans.iter().zip(&outs) {
        let dims = checked_state_dims(&s.state.s)?;
        if out.s.dims3()? != dims {
            candle::bail!(
                "delta_net cuda: a span's entering and advanced states differ in shape: \
                 {:?} against {:?}",
                s.state.s.dims(),
                out.s.dims()
            );
        }
        ptrs.push(aligned16(f32_ptr(&s.state.s, "state in")?, "state in")? as i64);
    }
    for out in &outs {
        ptrs.push(aligned16(f32_ptr(&out.s, "state out")?, "state out")? as i64);
    }
    Ok(ptrs)
}

/// Each span's `(start, len)`, as the table's two extent rows.
///
/// `start` is the span's row in the PACKED wave buffer; the transients the scan
/// hands between its two passes are indexed by the same number, which is what
/// lets them be one wave-wide allocation instead of one per span.
fn span_extents(spans: &[&DeltaNetSeq<'_>]) -> Vec<u32> {
    let mut extents: Vec<u32> = Vec::with_capacity(2 * spans.len());
    extents.extend(spans.iter().map(|s| s.start as u32));
    extents.extend(spans.iter().map(|s| s.len as u32));
    extents
}

/// One fused gated-delta-rule token step for EVERY decode sequence in the
/// wave — one launch however many sessions it carries. States are read through
/// the pointer table and never written: each token's generators go to row 0 of
/// its sequence's timeline, for the fold. Each sequence's output lands in its
/// own row of the wave's `o`. Gates are computed in-kernel from the raw
/// projections; q/k come from the conv output with the tiled GQA mapping.
pub fn delta_net_decode_batch(
    fused: &DeltaNetFused<'_, '_>,
    table: &DeltaNetLayerTable,
) -> Result<()> {
    let (t, conv_dim) = fused.conved.dims2()?;
    let (ta, h_v) = fused.alpha.dims2()?;
    let (to, cols) = fused.o.dims2()?;
    if ta != t || to != t || !cols.is_multiple_of(h_v) {
        candle::bail!(
            "delta_net cuda: decode batch geometry mismatch (conved {t}, alpha {ta}, o {to})"
        );
    }
    let d = cols / h_v;
    // The timeline's fields are laid out for the fused kernels' head width.
    if d != DELTA_NET_PREFILL_DIM {
        candle::bail!(
            "delta_net cuda: the decode step writes a timeline at d == \
             {DELTA_NET_PREFILL_DIM}, got {d}"
        );
    }
    if !conv_dim.is_multiple_of(d) || conv_dim <= h_v * d {
        candle::bail!("delta_net cuda: conv row {conv_dim} is not [Q|K|V] at d = {d}");
    }
    let qk_heads = conv_dim / d - h_v;
    if !qk_heads.is_multiple_of(2) {
        candle::bail!("delta_net cuda: conv row {conv_dim} does not split into Q and K");
    }
    let h_k = qk_heads / 2;
    let (table_rows, n_decode) = table.ptrs.dims2()?;
    if table_rows != TABLE_ROWS || table.rows.dims1()? != n_decode {
        candle::bail!("delta_net cuda: malformed decode table");
    }
    let dev = match fused.conved.device() {
        Device::Cuda(dv) => dv.clone(),
        _ => candle::bail!("delta_net cuda: conved must live on a CUDA device"),
    };
    let stream = dev.cuda_stream();
    // Row 1 holds the state pointers, row 2 the timelines.
    let states_p = typed_ptr(&table.ptrs.narrow(0, 1, 1)?, DType::I64, "state table")?;
    let timelines_p = typed_ptr(&table.ptrs.narrow(0, 2, 1)?, DType::I64, "timeline table")?;
    let rows_p = typed_ptr(&table.rows, DType::U32, "row table")?;
    let conved_p = f32_ptr(fused.conved, "conved")?;
    let alpha_p = f32_ptr(fused.alpha, "alpha")?;
    let blin_p = f32_ptr(fused.blin, "beta_lin")?;
    let dt_p = f32_ptr(fused.dt_bias, "dt_bias")?;
    let a_p = f32_ptr(fused.a, "a")?;
    let o_p = f32_ptr(fused.o, "o")?;
    let status = unsafe {
        run_delta_net_decode_step_f32(
            states_p as *const i64,
            timelines_p as *const i64,
            conved_p as *const f32,
            rows_p as *const u32,
            alpha_p as *const f32,
            blin_p as *const f32,
            dt_p as *const f32,
            a_p as *const f32,
            o_p as *mut f32,
            n_decode as i32,
            h_v as i32,
            h_k as i32,
            d as i32,
            d as i32,
            conv_dim as i32,
            fused.q_scale,
            stream.cu_stream() as *mut core::ffi::c_void,
        )
    };
    if status != 0 {
        candle::bail!(
            "delta_net cuda: the decode step refused {n_decode} rows over h_v {h_v} / h_k {h_k} \
             at d {d} and wrote nothing"
        );
    }
    Ok(())
}

/// The batched decode conv: one launch over every decode sequence in the
/// wave, reading each sequence's raw QKV row of `qkv` and its conv tail, and
/// writing its post-epilogue row of `conved` and its raw input into row 0 of
/// its timeline. The tail is read and never written.
pub fn delta_net_conv_decode(
    qkv: &LiveTensor<'_>,
    kernel: &Tensor,
    table: &DeltaNetLayerTable,
    conved: &LiveTensor<'_>,
    qk_channels: usize,
    eps: f32,
) -> Result<()> {
    let (t, channels) = qkv.dims2()?;
    let (kc, kwidth) = kernel.dims2()?;
    if kc != channels {
        candle::bail!("delta_net cuda: kernel channels {kc} != x channels {channels}");
    }
    if conved.dims2()? != (t, channels) {
        candle::bail!(
            "delta_net cuda: conved must match qkv, got {:?}",
            conved.dims()
        );
    }
    check_qk_channels(qk_channels, channels)?;
    let (table_rows, n_decode) = table.ptrs.dims2()?;
    if table_rows != TABLE_ROWS || table.rows.dims1()? != n_decode {
        candle::bail!("delta_net cuda: malformed decode table");
    }
    let dev = match qkv.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("delta_net cuda: qkv must live on a CUDA device"),
    };
    let stream = dev.cuda_stream();
    // Row 0 holds the tail pointers, row 2 the timelines.
    let tails_p = typed_ptr(&table.ptrs.narrow(0, 0, 1)?, DType::I64, "tail table")?;
    let timelines_p = typed_ptr(&table.ptrs.narrow(0, 2, 1)?, DType::I64, "timeline table")?;
    let rows_p = typed_ptr(&table.rows, DType::U32, "row table")?;
    let x_p = f32_ptr(qkv, "qkv")?;
    let k_p = f32_ptr(kernel, "kernel")?;
    let y_p = f32_ptr(conved, "conved")?;
    let status = unsafe {
        run_delta_net_conv_decode_f32(
            x_p as *const f32,
            k_p as *const f32,
            tails_p as *const i64,
            timelines_p as *const i64,
            rows_p as *const u32,
            y_p as *mut f32,
            n_decode as i32,
            channels as i32,
            kwidth as i32,
            qk_channels as i32,
            eps,
            stream.cu_stream() as *mut core::ffi::c_void,
        )
    };
    if status != 0 {
        candle::bail!(
            "delta_net cuda: the decode conv refused {n_decode} rows over {channels} channels \
             ({qk_channels} Q|K), window {kwidth}, and wrote nothing"
        );
    }
    Ok(())
}

/// Whether the fold kernel serves `dims`: the fused head width, and a conv
/// window the tail's registers hold.
pub fn fold_runs_fused(dims: &DeltaNetDims) -> bool {
    dims.head_dim == DELTA_NET_PREFILL_DIM
        && (2..=DELTA_NET_PREFILL_CONV_MAX).contains(&dims.conv_kernel)
}

/// One sequence's layer to fold: its conv tail, state and timeline addresses,
/// and the rows to keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldEntry {
    pub tail: u64,
    pub state: u64,
    pub timeline: u64,
    pub m: usize,
}

/// Apply every entry's first `m` timeline rows to its state and conv tail, in
/// place, in ONE launch however many sequences and layers it covers — up to the
/// grid's z extent, [`DELTA_NET_MAX_SPANS`] entries, a launch each beyond it
/// (`delta_net_fold_kernel.cuh`).
///
/// The table — three pointers and a count per entry, one `[4, n]` upload — goes
/// on held ground: a fold is issued after a forward, where no wave span stands,
/// and its table is dead once the launch is behind it.
pub fn delta_net_fold(entries: &[FoldEntry], dims: &DeltaNetDims, device: &Device) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }
    if !fold_runs_fused(dims) {
        candle::bail!(
            "delta_net cuda: the fold kernel runs at d == {DELTA_NET_PREFILL_DIM} with a conv \
             window of 2..={DELTA_NET_PREFILL_CONV_MAX}, got d {} and window {}",
            dims.head_dim,
            dims.conv_kernel
        );
    }
    let Device::Cuda(dev) = device else {
        candle::bail!("delta_net cuda: the fold kernel runs on a CUDA device");
    };
    // **Eager, in issue order behind whatever this thread has recorded.** A fold
    // is issued at a forward's commit and at a step's reconcile, both inside a
    // wave capture, and its table is held ground: giving the range back fences
    // it on the stream, which a recording stream cannot take. Declared first so
    // it is dropped last — the table goes back while the stream is still eager.
    let _eager = device.eager()?;
    // Every entry checked before anything launches, so a refusal writes nothing.
    for e in entries {
        if e.m == 0 || e.m > TIMELINE_ROWS {
            candle::bail!(
                "delta_net cuda: a fold of {} rows from a {TIMELINE_ROWS}-row timeline",
                e.m
            );
        }
        if !e.state.is_multiple_of(16) || !e.timeline.is_multiple_of(16) {
            candle::bail!(
                "delta_net cuda: a fold entry's state or timeline is not 16-byte aligned"
            );
        }
    }
    // One launch per grid's worth: an entry is a block column of the grid's z
    // extent. The launcher refuses only by geometry, which every chunk shares,
    // so a refusal comes on the first launch, before anything is written.
    let stream = dev.cuda_stream();
    for chunk in entries.chunks(DELTA_NET_MAX_SPANS) {
        let n = chunk.len();
        let mut table: Vec<i64> = Vec::with_capacity(FOLD_TABLE_ROWS * n);
        table.extend(chunk.iter().map(|e| e.tail as i64));
        table.extend(chunk.iter().map(|e| e.state as i64));
        table.extend(chunk.iter().map(|e| e.timeline as i64));
        table.extend(chunk.iter().map(|e| e.m as i64));
        let table_t = held_empty((FOLD_TABLE_ROWS, n), DType::I64, device)?;
        upload_into(&table_t, &table)?;
        let status = unsafe {
            run_delta_net_fold_f32(
                typed_ptr(&table_t, DType::I64, "fold table")? as *const i64,
                n as i32,
                dims.n_v_heads as i32,
                dims.n_k_heads as i32,
                dims.conv_dim() as i32,
                dims.conv_kernel as i32,
                stream.cu_stream() as *mut c_void,
            )
        };
        if status != 0 {
            candle::bail!(
                "delta_net cuda: the fold refused {n} entries over h_v {} / h_k {} and wrote \
                 nothing",
                dims.n_v_heads,
                dims.n_k_heads
            );
        }
    }
    Ok(())
}

/// [`delta_net_fold`] over states and timelines held as tensors — the
/// single-sequence entry points and the kernel tests.
pub fn delta_net_fold_states(
    jobs: &[(&DeltaNetState, &Tensor, usize)],
    dims: &DeltaNetDims,
) -> Result<()> {
    let Some((first, _, _)) = jobs.first() else {
        return Ok(());
    };
    let lay = TimelineLayout::of(dims);
    let entries = jobs
        .iter()
        .map(|(st, tl, m)| {
            if tl.dims1()? != lay.elems() {
                candle::bail!(
                    "delta_net cuda: a {:?} timeline for a geometry of {} elements",
                    tl.dims(),
                    lay.elems()
                );
            }
            Ok(FoldEntry {
                tail: f32_ptr(&st.conv_tail, "conv tail")?,
                state: f32_ptr(&st.s, "state")?,
                timeline: f32_ptr(tl, "timeline")?,
                m: *m,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let device = first.s.device().clone();
    delta_net_fold(&entries, dims, &device)
}

/// Validate the epilogue geometry both conv wrappers share: `qk_channels`
/// must cover whole 256-channel blocks so the in-kernel norm reduction never
/// straddles a block (`qk_channels = 2·h_k·128 = h_k·256` always does).
fn check_qk_channels(qk_channels: usize, channels: usize) -> Result<()> {
    if qk_channels > channels || !qk_channels.is_multiple_of(256) {
        candle::bail!(
            "delta_net cuda: qk_channels {qk_channels} must be a multiple of 256 \
             within {channels} conv channels"
        );
    }
    Ok(())
}

/// Solve `(I + A) X = B` for every head, where `a [H, c, c]` is **strictly**
/// lower triangular (its diagonal is ignored — the identity supplies it) and
/// `rhs [H, c, n]` holds the right-hand sides. Returns `X [H, c, n]`.
///
/// # Why a solve and not an inverse
///
/// The chunked scan needs `T·(βv)` and `T·(βk⊙e^G)` where `T = (I + A)⁻¹`.
/// Forming `T` and multiplying is the obvious reading, and it is what the
/// reference [`super::mix::unit_lower_inverse`] does — by a recursive 2×2
/// block inversion that costs `log₂ c` levels of a handful of launches each,
/// about eighty host-issued ops per chunk, all of them serially dependent. Both
/// right-hand sides share a left side, so concatenating them into one solve
/// replaces the whole of that with a single `trsm`, which is also the more
/// accurate answer: substitution is backward-stable and never forms an inverse.
///
/// # Row-major against a column-major library
///
/// cuBLAS reads column-major, and every tensor here is row-major, so the
/// buffers are *reinterpreted* rather than transposed — a row-major `[c, c]`
/// with leading dimension `c` is exactly a column-major `Aᵀ`, and `Aᵀ` of a
/// lower triangle is an upper one. Transposing the equation to match,
/// `A X = B` becomes `Xᵀ Aᵀ = Bᵀ`, which is `trsm`'s right-side form over the
/// reinterpreted buffers. Hence `SIDE_RIGHT` and `FILL_MODE_UPPER` on data that
/// is neither: they describe the column-major view, not the tensor.
///
/// `trsm` overwrites its right-hand side with the solution, so `rhs` must be a
/// buffer the caller owns; the result aliases it.
pub fn solve_unit_lower<'w>(a: &LiveTensor<'w>, rhs: LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    let (h, c, c2) = a.dims3()?;
    if c != c2 {
        candle::bail!("delta_net cuda: solve matrix must be square, got [{h}, {c}, {c2}]");
    }
    let (hr, cr, n) = rhs.dims3()?;
    if hr != h || cr != c {
        candle::bail!(
            "delta_net cuda: rhs [{hr}, {cr}, {n}] does not match matrix [{h}, {c}, {c}]"
        );
    }
    let dev = match a.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("delta_net cuda: solve_unit_lower requires a CUDA tensor"),
    };
    if h == 0 || c == 0 || n == 0 {
        return Ok(rhs);
    }
    let stream = dev.cuda_stream();
    let a_p = f32_ptr(a, "solve matrix")?;
    let b_p = f32_ptr(&rhs, "solve rhs")?;

    // Two `[h]` address arrays, written on the device. `I64` because that is
    // the eight-byte dtype available; the bytes are pointers.
    let ptrs = rhs.empty_beside((2, h), DType::I64)?;
    let ptr_base = typed_ptr(&ptrs, DType::I64, "solve pointer table")?;
    let a_ptrs = ptr_base as *mut *const f32;
    // SAFETY: `ptrs` is `2 × h` eight-byte slots; the second row starts at `h`.
    let b_ptrs = unsafe { (ptr_base as *mut *mut f32).add(h) };
    unsafe {
        run_delta_net_batch_ptrs(
            a_p as *const f32,
            (c * c) as i64,
            b_p as *mut f32,
            (c * n) as i64,
            a_ptrs,
            b_ptrs,
            h as i32,
            stream.cu_stream() as *mut core::ffi::c_void,
        );
    }

    let alpha = 1f32;
    let status = unsafe {
        cublas::cublasStrsmBatched(
            *dev.cublas()?.handle(),
            cublas::cublasSideMode_t::CUBLAS_SIDE_RIGHT,
            cublas::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
            cublas::cublasOperation_t::CUBLAS_OP_N,
            cublas::cublasDiagType_t::CUBLAS_DIAG_UNIT,
            n as i32,
            c as i32,
            &alpha as *const f32,
            a_ptrs as *const *const f32,
            c as i32,
            b_ptrs as *const *mut f32,
            n as i32,
            h as i32,
        )
    };
    if status != cublas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        candle::bail!("delta_net cuda: cublasStrsmBatched failed: {status:?}");
    }
    Ok(rhs)
}

/// Token-parallel causal conv over every prefill span of a wave, with the SiLU
/// and Q|K-norm epilogue, written directly into each span's rows of the
/// whole-wave `conved` buffer — the operand buffer the scan kernels read q/k/v
/// from through strides.
///
/// `qkv [T_wave, channels]` is **token-major** — the projection's own rows, so
/// nothing is transposed to channel-major. Each span's advanced tail (raw
/// values) goes to its tail-out, a **different** allocation from its entering
/// tail: blocks computing the first `kwidth−1` outputs read the entering tail
/// concurrently, so writing over it would race. The tail-out is the sequence's
/// prefill scratch, so the advance lands where `commit_wave` installs it and
/// nothing is copied.
// Eight: the kernel's own operands. Boxing them into a struct would put a
// shape-checked argument list behind a constructor that checks nothing.
#[allow(clippy::too_many_arguments)]
pub fn delta_net_conv_prefill(
    qkv: &LiveTensor<'_>,
    kernel: &Tensor,
    table: &DeltaNetSpanTable,
    qk_channels: usize,
    eps: f32,
    conved: &LiveTensor<'_>,
) -> Result<()> {
    let (t_wave, channels) = qkv.dims2()?;
    let (kc, kwidth) = kernel.dims2()?;
    if kc != channels {
        candle::bail!("delta_net cuda: kernel channels {kc} != x channels {channels}");
    }
    if !(2..=DELTA_NET_PREFILL_CONV_MAX).contains(&kwidth) {
        candle::bail!(
            "delta_net cuda: the prefill conv takes windows of 2..={DELTA_NET_PREFILL_CONV_MAX}, \
             got {kwidth}"
        );
    }
    let (tc, cc) = conved.dims2()?;
    if cc != channels || tc != t_wave {
        candle::bail!(
            "delta_net cuda: conv output {:?} does not match its operand {:?}",
            conved.dims(),
            qkv.dims()
        );
    }
    check_qk_channels(qk_channels, channels)?;
    let dev = match qkv.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("delta_net cuda: x must live on a CUDA device"),
    };
    let stream = dev.cuda_stream();
    check_table_rows(table, LONG_TABLE_ROWS, "the prefill conv")?;
    let x_p = f32_ptr(qkv, "qkv")?;
    let k_p = f32_ptr(kernel, "kernel")?;
    let y_p = f32_ptr(conved, "conved")?;
    let ptrs_p = typed_ptr(&table.ptrs, DType::I64, "span ptrs")?;
    let spans_p = typed_ptr(&table.spans, DType::U32, "span extents")?;
    let status = unsafe {
        run_delta_net_conv_prefill_f32(
            x_p as *const f32,
            k_p as *const f32,
            y_p as *mut f32,
            ptrs_p as *const i64,
            spans_p as *const u32,
            table.n as i32,
            table.max_len as i32,
            channels as i32,
            kwidth as i32,
            qk_channels as i32,
            eps,
            stream.cu_stream() as *mut c_void,
        )
    };
    if status != 0 {
        candle::bail!(
            "delta_net cuda: the prefill conv refused {} spans of up to {} rows, {channels} \
             channels, window {kwidth} and wrote nothing",
            table.n,
            table.max_len
        );
    }
    Ok(())
}

/// The fused prefill scan for every prefill span of a wave: the chunked
/// parallel form of the gated delta rule in two launches, replacing
/// `delta_chunked`'s ~65 ops per chunk.
///
/// Each span's entering state `[h_v, 128, 128]` is read and its advanced state
/// written to its state-out — the prefill scratch — in the stored orientation
/// (no `s_fla` transpose exists anywhere on this path); everything else is
/// read from — and written into — the wave tensors in `fused`, through
/// strides. The intra-chunk kernel (parallel over chunks × heads) hands
/// `u`/`w`/`kq` to the sequential state walk through wave-backed transients;
/// the state kernel keeps its S tile in shared memory across the whole sequence.
pub fn delta_net_prefill_scan(
    fused: &DeltaNetFused<'_, '_>,
    table: &DeltaNetSpanTable,
) -> Result<()> {
    // Resolved over the WHOLE wave, not a span: the kernels rebase to their own
    // span from the table, so the pointers handed to them are the packed
    // buffers' bases.
    let t_wave = fused.conved.dim(0)?;
    let p = fused.resolve_wave()?;
    if p.d != DELTA_NET_PREFILL_DIM {
        candle::bail!(
            "delta_net cuda: the prefill scan is compiled for d == {DELTA_NET_PREFILL_DIM}, \
             got {}",
            p.d
        );
    }
    let dev = match fused.conved.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("delta_net cuda: the wave must live on a CUDA device"),
    };
    // Transients: fully kernel-written, allocated in the wave's arena by
    // operand provenance (pool fallback when full). `kq` rows are valid for
    // `s ≤ t` only; the state kernel never reads past that, so the rest stays
    // uninitialised.
    //
    // ONE allocation for the whole wave rather than one per span: a head's rows
    // are `t_wave` apart and each span writes the slice at its own `start`, so
    // the spans never overlap and no span reads another's. Sized by the wave and
    // not by the spans' total, so the row index is the same number the packed
    // buffers use and no second mapping exists to disagree.
    let (h_v, d) = (p.h_v, p.d);
    let u = fused.conved.empty_beside((h_v, t_wave, d), DType::F32)?;
    let w = fused.conved.empty_beside((h_v, t_wave, d), DType::F32)?;
    let kq = fused
        .conved
        .empty_beside((h_v, t_wave, DELTA_NET_PREFILL_CHUNK), DType::F32)?;
    let g_cs = fused.conved.empty_beside((h_v, t_wave), DType::F32)?;
    {
        let stream = dev.cuda_stream();
        aligned16(p.qk_p, "the conv output")?;
        let u_p = aligned16(f32_ptr(&u, "u")?, "u")?;
        let w_p = aligned16(f32_ptr(&w, "w")?, "w")?;
        let kq_p = aligned16(f32_ptr(&kq, "kq")?, "kq")?;
        let gcs_p = f32_ptr(&g_cs, "g_cs")?;
        check_table_rows(table, LONG_TABLE_ROWS, "the prefill scan")?;
        let ptrs_p = typed_ptr(&table.ptrs, DType::I64, "span ptrs")?;
        let spans_p = typed_ptr(&table.spans, DType::U32, "span extents")?;
        let raw = stream.cu_stream() as *mut core::ffi::c_void;
        unsafe {
            run_delta_net_prefill_intra_f32(
                p.qk_p as *const f32,
                p.v_p as *const f32,
                p.alpha_p as *const f32,
                p.blin_p as *const f32,
                p.dt_p as *const f32,
                p.a_p as *const f32,
                u_p as *mut f32,
                w_p as *mut f32,
                kq_p as *mut f32,
                gcs_p as *mut f32,
                spans_p as *const u32,
                table.n as i32,
                table.max_len as i32,
                t_wave as i32,
                p.h_v as i32,
                p.h_k as i32,
                p.conv_dim as i32,
                fused.q_scale,
                raw,
            );
            run_delta_net_prefill_state_f32(
                p.qk_p as *const f32,
                u_p as *const f32,
                w_p as *const f32,
                kq_p as *const f32,
                gcs_p as *const f32,
                p.o_p as *mut f32,
                ptrs_p as *const i64,
                spans_p as *const u32,
                table.n as i32,
                t_wave as i32,
                p.h_v as i32,
                p.h_k as i32,
                p.conv_dim as i32,
                fused.q_scale,
                raw,
            );
        }
    }
    Ok(())
}

/// A span table's pointer rows, checked: a short table carries
/// [`TABLE_ROWS`], a long one [`LONG_TABLE_ROWS`], one column per span.
fn check_table_rows(table: &DeltaNetSpanTable, rows: usize, what: &str) -> Result<()> {
    if table.ptrs.dims2()? != (rows, table.n) {
        candle::bail!(
            "delta_net cuda: {what} takes a [{rows}, {}] span table, got {:?}",
            table.n,
            table.ptrs.dims()
        );
    }
    Ok(())
}

/// The conv, the intra-chunk solve and the state pass's reads of the fused
/// scan for the wave's short spans — each at most [`DELTA_NET_SHORT_SPAN_ROWS`]
/// rows — in ONE launch, where [`delta_net_conv_prefill`] +
/// [`delta_net_prefill_scan`] take three chained through global memory.
///
/// The same bits as that pair in `fused.o`'s span rows. It writes each span's
/// timeline (G, k, v_new and the raw conv rows) and leaves the span's state and
/// conv tail as they were — the fold applies the rows the step keeps. It reads
/// the raw `qkv` rows and convolves them itself, so the span rows of
/// `fused.conved` are neither read nor written — that buffer supplies the
/// wave's geometry here, and holds only the decode rows the batched decode conv
/// writes beside this.
///
/// Refused with an error, nothing written, when a span is longer than
/// [`DELTA_NET_SHORT_SPAN_ROWS`] or the conv is not
/// [`DELTA_NET_SHORT_SPAN_CONV`] wide — the shapes the kernel serves.
pub fn delta_net_short_span_scan(
    fused: &DeltaNetFused<'_, '_>,
    qkv: &LiveTensor<'_>,
    kernel: &Tensor,
    table: &DeltaNetSpanTable,
    eps: f32,
) -> Result<()> {
    check_table_rows(table, TABLE_ROWS, "the short-span scan")?;
    if table.max_len > DELTA_NET_SHORT_SPAN_ROWS {
        candle::bail!(
            "delta_net cuda: the short-span scan takes spans of at most \
             {DELTA_NET_SHORT_SPAN_ROWS} rows, got one of {}",
            table.max_len
        );
    }
    let p = fused.resolve_wave()?;
    if p.d != DELTA_NET_PREFILL_DIM {
        candle::bail!(
            "delta_net cuda: the short-span scan is compiled for d == \
             {DELTA_NET_PREFILL_DIM}, got {}",
            p.d
        );
    }
    let t_wave = fused.conved.dim(0)?;
    if qkv.dims2()? != (t_wave, p.conv_dim) {
        candle::bail!(
            "delta_net cuda: qkv {:?} does not match the wave's conv rows [{t_wave}, {}]",
            qkv.dims(),
            p.conv_dim
        );
    }
    if kernel.dims2()? != (p.conv_dim, DELTA_NET_SHORT_SPAN_CONV) {
        candle::bail!(
            "delta_net cuda: the short-span scan convolves {DELTA_NET_SHORT_SPAN_CONV} wide over \
             {} channels, got a {:?} kernel",
            p.conv_dim,
            kernel.dims()
        );
    }
    let dev = match qkv.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("delta_net cuda: qkv must live on a CUDA device"),
    };
    let stream = dev.cuda_stream();
    let x_p = f32_ptr(qkv, "qkv")?;
    let k_p = f32_ptr(kernel, "kernel")?;
    let ptrs_p = typed_ptr(&table.ptrs, DType::I64, "span ptrs")?;
    let spans_p = typed_ptr(&table.spans, DType::U32, "span extents")?;
    let status = unsafe {
        run_delta_net_short_span_f32(
            x_p as *const f32,
            k_p as *const f32,
            p.alpha_p as *const f32,
            p.blin_p as *const f32,
            p.dt_p as *const f32,
            p.a_p as *const f32,
            p.o_p as *mut f32,
            ptrs_p as *const i64,
            spans_p as *const u32,
            table.n as i32,
            table.max_len as i32,
            p.h_v as i32,
            p.h_k as i32,
            p.conv_dim as i32,
            DELTA_NET_SHORT_SPAN_CONV as i32,
            eps,
            fused.q_scale,
            stream.cu_stream() as *mut core::ffi::c_void,
        )
    };
    if status != 0 {
        candle::bail!(
            "delta_net cuda: the short-span scan refused {} spans of up to {} rows over \
             h_v {} / h_k {} / {} channels and wrote nothing",
            table.n,
            table.max_len,
            p.h_v,
            p.h_k,
            p.conv_dim
        );
    }
    Ok(())
}

/// The mixer epilogue over the whole wave in one launch: per `(token, V head)`
/// row, `out = (o / sqrt(mean(o²) + eps)) ⊙ gain ⊙ zgate(z)` — the per-head
/// RMS norm and the z-gate that were ~6 ops and three full-width
/// intermediates. The gate is a kernel template instantiation selected by
/// `zgate` (SiLU for the Qwen3.5 lineage, sigmoid for qwen4exp).
pub fn delta_net_norm_gate<'w>(
    o: &LiveTensor<'w>,
    z: &LiveTensor<'w>,
    gain: &Tensor,
    d: usize,
    eps: f32,
    zgate: ZGate,
) -> Result<LiveTensor<'w>> {
    let (t, cols) = norm_gate_shape(o, z, gain, d)?;
    let out = o.empty_beside((t, cols), DType::F32)?;
    norm_gate_launch(o, z, gain, d, eps, zgate, f32_ptr(&out, "out")?, 0, 0)?;
    Ok(out)
}

/// [`delta_net_norm_gate`]'s output as the q8a128 operand under `sum_scale`
/// alone — the bytes a standalone quantize of that output would write, emitted
/// by the epilogue from the values it computes — for an out-projection that
/// runs int8 and reads nothing else. No F32 output is stored. One tile per row,
/// so `d` must be 128.
pub fn delta_net_norm_gate_q8<'w>(
    o: &LiveTensor<'w>,
    z: &LiveTensor<'w>,
    gain: &Tensor,
    d: usize,
    eps: f32,
    zgate: ZGate,
    sum_scale: SumScale,
) -> Result<Q8a128Operand<'w>> {
    let (t, cols) = norm_gate_shape(o, z, gain, d)?;
    if d != 128 {
        candle::bail!("delta_net cuda: the norm-gate emits its operand one 128-wide tile per row, got d = {d}");
    }
    // The operand is placed beside `o`, the way the F32 output would have been.
    produce_q8a128(o, t, cols, sum_scale, |q8| {
        norm_gate_launch(o, z, gain, d, eps, zgate, 0, q8, sum_scale.as_code())
    })
}

/// `(T, cols)` of a norm-gate over `o`, after checking `z` and `gain` against it.
fn norm_gate_shape(
    o: &LiveTensor<'_>,
    z: &LiveTensor<'_>,
    gain: &Tensor,
    d: usize,
) -> Result<(usize, usize)> {
    let (t, cols) = o.dims2()?;
    if z.dims2()? != (t, cols) {
        candle::bail!(
            "delta_net cuda: z must match o, got {:?} vs {:?}",
            z.dims(),
            o.dims()
        );
    }
    if gain.dims1()? != d || !cols.is_multiple_of(d) || d > MAX_HEAD_DIM {
        candle::bail!(
            "delta_net cuda: gain [{d}] must divide {cols} rows-wise (cap {MAX_HEAD_DIM})"
        );
    }
    Ok((t, cols))
}

/// One norm-gate launch writing the F32 output at `out_p`, the q8a128 operand
/// at `q8`, or both; a `0` address writes nothing there.
#[allow(clippy::too_many_arguments)]
fn norm_gate_launch(
    o: &LiveTensor<'_>,
    z: &LiveTensor<'_>,
    gain: &Tensor,
    d: usize,
    eps: f32,
    zgate: ZGate,
    out_p: u64,
    q8: u64,
    sum_norm: i32,
) -> Result<()> {
    let (t, cols) = o.dims2()?;
    let dev = match o.device() {
        Device::Cuda(dv) => dv.clone(),
        _ => candle::bail!("delta_net cuda: o must live on a CUDA device"),
    };
    let o_p = f32_ptr(o, "o")?;
    let z_p = f32_ptr(z, "z")?;
    let g_p = f32_ptr(gain, "gain")?;
    let rows = t * (cols / d);
    let stream = dev.cuda_stream();
    // SAFETY: `o`, `z` and `gain` are live F32 buffers of the checked shapes,
    // and `out_p` / `q8` are either 0 or sized by the caller for `rows` rows
    // of `d` (one q8a128 tile per row); the launch is on the device's stream.
    let status = unsafe {
        run_delta_net_norm_gate_f32(
            o_p as *const f32,
            z_p as *const f32,
            g_p as *const f32,
            out_p as *mut f32,
            rows as i32,
            d as i32,
            eps,
            matches!(zgate, ZGate::Sigmoid) as i32,
            q8 as *mut u8,
            sum_norm,
            stream.cu_stream() as *mut c_void,
        )
    };
    if status != 0 {
        candle::bail!("delta_net cuda: the norm-gate refused rows={rows} d={d} and wrote nothing");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::mix::{causal_conv1d, delta_recurrence, l2_norm};
    use super::*;
    use candle::{DType, Device};

    mod prefill_golden;

    /// One span as the kernel tests hold it: they drive the launches directly,
    /// without the mixer's [`DeltaNetSeq`] bookkeeping around them.
    struct TestSpan<'a> {
        /// The table is shared by the conv and the scan, and each reads only the
        /// rows it needs — the conv the tails, the scan the states. A case that
        /// drives one kernel names any live tensor in the rows that kernel does
        /// not dereference.
        tail: &'a Tensor,
        tail_out: &'a Tensor,
        state: &'a Tensor,
        state_out: &'a Tensor,
        start: usize,
        len: usize,
    }

    /// The table [`build_span_table`] produces, from tensors rather than from
    /// `DeltaNetSeq`s. Same layout, so a divergence between them is a
    /// compile-time break rather than a silent one.
    fn span_table(rows: &[TestSpan<'_>]) -> DeltaNetSpanTable<'static> {
        let n = rows.len();
        let mut ptrs: Vec<i64> = Vec::with_capacity(4 * n);
        for r in rows {
            ptrs.push(f32_ptr(r.tail, "tail").unwrap() as i64);
        }
        for r in rows {
            ptrs.push(f32_ptr(r.tail_out, "tail out").unwrap() as i64);
        }
        for r in rows {
            ptrs.push(f32_ptr(r.state, "state").unwrap() as i64);
        }
        for r in rows {
            ptrs.push(f32_ptr(r.state_out, "state out").unwrap() as i64);
        }
        let mut ext: Vec<u32> = Vec::with_capacity(2 * n);
        ext.extend(rows.iter().map(|r| r.start as u32));
        ext.extend(rows.iter().map(|r| r.len as u32));
        let max_len = rows.iter().map(|r| r.len).max().unwrap();
        let dev = rows[0].tail.device().clone();
        DeltaNetSpanTable {
            ptrs: Tensor::from_vec(ptrs, (4, n), &dev).unwrap(),
            spans: Tensor::from_vec(ext, (2, n), &dev).unwrap(),
            n,
            max_len,
        }
    }

    fn lcg_tensor(shape: &[usize], seed: u64, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let vals: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect();
        Tensor::from_vec(vals, shape, dev).unwrap()
    }

    fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        a.sub(b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// One synthetic mixer wave in the fused kernels' own layout — a single
    /// conv-output buffer whose Q|K columns are l2-normed and whose trailing
    /// columns are V, plus the raw gate projections — and the
    /// tiled/scaled/gated operands the sequential reference consumes.
    /// Building both from the same randomness is what pins the in-kernel GQA
    /// mapping, read scale, and gate arithmetic to the tensor-op forms.
    struct FusedCase {
        conved: Tensor,
        alpha: Tensor,
        blin: Tensor,
        dt_bias: Tensor,
        a: Tensor,
        q_scale: f32,
        // Reference operands (CPU): tiled to h_v, q scaled, gates applied.
        q_ref: Tensor,
        k_ref: Tensor,
        v_ref: Tensor,
        g_ref: Tensor,
        b_ref: Tensor,
    }

    impl FusedCase {
        fn build(t: usize, h_k: usize, h_v: usize, seed: u64, cpu: &Device, gpu: &Device) -> Self {
            use super::super::mix::softplus;
            let d = 128usize;
            let conv_dim = (2 * h_k + h_v) * d; // Q | K | V column layout

            // The scan kernels read a post-epilogue buffer: Q|K columns
            // l2-normed, V columns arbitrary (post-SiLU values — random works,
            // the reference reads the same buffer).
            let qk = l2_norm(&lcg_tensor(&[t, 2 * h_k, d], seed, cpu), 1e-6).unwrap();
            let v_cols = lcg_tensor(&[t, h_v * d], seed + 1, cpu);
            let conved =
                Tensor::cat(&[&qk.reshape((t, 2 * h_k * d)).unwrap(), &v_cols], 1).unwrap();
            let alpha = lcg_tensor(&[t, h_v], seed + 2, cpu);
            let blin = lcg_tensor(&[t, h_v], seed + 3, cpu);
            let dt_bias = lcg_tensor(&[h_v], seed + 4, cpu);
            // Negative per-head decay coefficients, as `−exp(A_log)` is.
            let a = (lcg_tensor(&[h_v], seed + 5, cpu).abs().unwrap() + 0.1)
                .unwrap()
                .neg()
                .unwrap();
            let q_scale = 1.0f32 / (d as f32).sqrt();
            assert_eq!(conved.dims2().unwrap(), (t, conv_dim));

            // The reference operands, by the tensor-op definitions.
            let tile = |x: &Tensor| -> Tensor {
                x.unsqueeze(1)
                    .unwrap()
                    .expand((t, h_v / h_k, h_k, d))
                    .unwrap()
                    .reshape((t, h_v, d))
                    .unwrap()
            };
            let q_ref = (tile(&qk.narrow(1, 0, h_k).unwrap().contiguous().unwrap())
                * q_scale as f64)
                .unwrap();
            let k_ref = tile(&qk.narrow(1, h_k, h_k).unwrap().contiguous().unwrap());
            let v_ref = v_cols.reshape((t, h_v, d)).unwrap();
            let g_ref = softplus(&alpha.broadcast_add(&dt_bias).unwrap())
                .unwrap()
                .broadcast_mul(&a)
                .unwrap();
            let b_ref = candle_nn::ops::sigmoid(&blin).unwrap();

            let to = |x: &Tensor| x.to_device(gpu).unwrap().contiguous().unwrap();
            Self {
                conved: to(&conved),
                alpha: to(&alpha),
                blin: to(&blin),
                dt_bias: to(&dt_bias),
                a: to(&a),
                q_scale,
                q_ref,
                k_ref,
                v_ref,
                g_ref,
                b_ref,
            }
        }

        fn fused<'a>(&'a self, o: &'a Tensor) -> DeltaNetFused<'a, 'static> {
            DeltaNetFused {
                conved: &self.conved,
                alpha: &self.alpha,
                blin: &self.blin,
                dt_bias: &self.dt_bias,
                a: &self.a,
                o,
                q_scale: self.q_scale,
            }
        }
    }

    /// The kernel must match the sequential CPU reference token by token,
    /// including the carried state, at realistic dims (d = 128, the published
    /// family's DeltaNet head width).
    /// **The solve must equal the explicit inverse it replaced.**
    ///
    /// `solve_unit_lower` is the chunked scan's hot path and it swaps a
    /// recursive block inversion followed by two matmuls for one `trsm`. The two
    /// are the same mathematics and *not* the same code, and the swap crosses a
    /// row-major/column-major boundary where a wrong `side` or `uplo` produces a
    /// plausible matrix rather than an error — so it is pinned against the
    /// reference form directly.
    ///
    /// A width that is not a multiple of anything convenient (37) and a
    /// right-hand side wider than the matrix, so a transposed dimension cannot
    /// go unnoticed.
    #[test]
    fn solve_matches_the_explicit_inverse() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        use super::super::mix::unit_lower_inverse_for_test;

        let (h, c, n) = (3usize, 37usize, 53usize);
        // Strictly lower triangular, with entries of a realistic size — the
        // scan's `A` is `(βk·k)·D` with `β ≤ 1` and `D ≤ 1`.
        let mut vals = vec![0f32; h * c * c];
        let raw: Vec<f32> = lcg_tensor(&[h, c, c], 21, &Device::Cpu)
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for head in 0..h {
            for i in 0..c {
                for j in 0..i {
                    vals[(head * c + i) * c + j] = raw[(head * c + i) * c + j];
                }
            }
        }
        let a = Tensor::from_vec(vals, (h, c, c), &gpu).unwrap();
        let rhs = lcg_tensor(&[h, c, n], 22, &gpu);

        let want = unit_lower_inverse_for_test(&a)
            .unwrap()
            .matmul(&rhs)
            .unwrap();
        let got = solve_unit_lower(&a, rhs.clone()).unwrap();

        let d = max_diff(&got, &want);
        let scale = want
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        println!("solve vs explicit inverse: max abs {d:.3e} over scale {scale:.3e}");
        assert!(
            d / scale.max(1e-6) < 1e-4,
            "triangular solve disagrees with the explicit inverse: {d} over {scale}"
        );
    }

    /// **The chunked scan, on the device, against the sequential rule.**
    ///
    /// `delta_net`'s own scan tests all run on the CPU, where
    /// [`super::super::mix::solve_pseudo_values`] takes the explicit-inverse
    /// fallback — so none of them touches the `trsm` path, and the first thing
    /// that path got wrong (a strided `narrow` handed to `matmul`) reached a
    /// gate run instead of a unit test. This is that coverage: the same
    /// property those tests assert, evaluated where the solve actually runs.
    #[test]
    fn chunked_scan_matches_sequential_on_cuda() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        use super::super::mix::{delta_chunked, delta_recurrence};

        // Two chunks plus a ragged tail at width 8, so the tail path and the
        // chunk-to-chunk state carry both run.
        let (t, h, d) = (21usize, 3usize, 8usize);
        let q = l2_norm(&lcg_tensor(&[t, h, d], 31, &gpu), 1e-6).unwrap();
        let k = l2_norm(&lcg_tensor(&[t, h, d], 32, &gpu), 1e-6).unwrap();
        let v = lcg_tensor(&[t, h, d], 33, &gpu);
        let g = lcg_tensor(&[t, h], 34, &gpu).abs().unwrap().neg().unwrap();
        let beta = candle_nn::ops::sigmoid(&lcg_tensor(&[t, h], 35, &gpu)).unwrap();
        let s0 = Tensor::zeros((h, d, d), DType::F32, &gpu).unwrap();

        let (o_ref, s_ref) = delta_recurrence(s0.copy().unwrap(), &q, &k, &v, &g, &beta).unwrap();
        for chunk in [8usize, 21, 64] {
            let mut s = s0.copy().unwrap();
            let o = delta_chunked(&mut s, &q, &k, &v, &g, &beta, chunk).unwrap();
            assert!(
                max_diff(&o, &o_ref) < 3e-4,
                "chunk={chunk} outputs diverged: {}",
                max_diff(&o, &o_ref)
            );
            assert!(
                max_diff(&s, &s_ref) < 3e-4,
                "chunk={chunk} state diverged: {}",
                max_diff(&s, &s_ref)
            );
        }
    }

    #[test]
    fn decode_step_matches_reference_across_a_sequence() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        // TWO decode sequences stepped together through the batched kernel —
        // interleaved wave rows (seq A on even rows, B on odd), separate
        // in-place states — against two independent sequential references.
        // h_k < h_v so the in-kernel tiled GQA mapping is exercised too.
        let (steps, h_k, h_v, d) = (5usize, 2usize, 4usize, 128usize);
        let t_wave = 2 * steps;
        let case = FusedCase::build(t_wave, h_k, h_v, 61, &cpu, &gpu);

        // CPU oracles: each sequence's rows of the shared wave buffers.
        let seq_rows = |first: usize| -> Vec<usize> { (0..steps).map(|i| 2 * i + first).collect() };
        let gather = |x: &Tensor, rows: &[usize]| -> Tensor {
            let idx = Tensor::from_vec(
                rows.iter().map(|&r| r as u32).collect::<Vec<_>>(),
                rows.len(),
                &cpu,
            )
            .unwrap();
            x.index_select(&idx, 0).unwrap()
        };
        let mut o_refs = Vec::new();
        let mut s_refs = Vec::new();
        for first in [0usize, 1] {
            let rows = seq_rows(first);
            let s0 = Tensor::zeros((h_v, d, d), DType::F32, &cpu).unwrap();
            let (o_r, s_r) = delta_recurrence(
                s0,
                &gather(&case.q_ref, &rows),
                &gather(&case.k_ref, &rows),
                &gather(&case.v_ref, &rows),
                &gather(&case.g_ref, &rows),
                &gather(&case.b_ref, &rows),
            )
            .unwrap();
            o_refs.push(o_r);
            s_refs.push(s_r);
        }

        // GPU: one batched launch per step over BOTH sequences, then one fold
        // of each sequence's one-row timeline — the decode step reads its state
        // and writes only the timeline.
        let dims = DeltaNetDims {
            head_dim: d,
            n_k_heads: h_k,
            n_v_heads: h_v,
            conv_kernel: 4,
        };
        let lay = TimelineLayout::of(&dims);
        let states = [
            DeltaNetState::zeros(&dims, &gpu).unwrap(),
            DeltaNetState::zeros(&dims, &gpu).unwrap(),
        ];
        let timelines = [lay.alloc(&gpu).unwrap(), lay.alloc(&gpu).unwrap()];
        let o = Tensor::zeros((t_wave, h_v * d), DType::F32, &gpu).unwrap();
        let fused = case.fused(&o);
        for i in 0..steps {
            let ptrs: Vec<i64> = vec![
                f32_ptr(&states[0].conv_tail, "tail").unwrap() as i64,
                f32_ptr(&states[1].conv_tail, "tail").unwrap() as i64,
                f32_ptr(&states[0].s, "state").unwrap() as i64,
                f32_ptr(&states[1].s, "state").unwrap() as i64,
                f32_ptr(&timelines[0], "timeline").unwrap() as i64,
                f32_ptr(&timelines[1], "timeline").unwrap() as i64,
            ];
            let table = DeltaNetLayerTable {
                ptrs: Tensor::from_vec(ptrs, (3, 2), &gpu).unwrap(),
                rows: Tensor::from_vec(vec![2 * i as u32, 2 * i as u32 + 1], 2, &gpu).unwrap(),
            };
            delta_net_decode_batch(&fused, &table).unwrap();
            delta_net_fold_states(
                &[
                    (&states[0], &timelines[0], 1),
                    (&states[1], &timelines[1], 1),
                ],
                &dims,
            )
            .unwrap();
        }

        let o_cpu = o
            .reshape((t_wave, h_v, d))
            .unwrap()
            .to_device(&cpu)
            .unwrap();
        for (first, (o_ref, s_ref)) in o_refs.iter().zip(s_refs.iter()).enumerate() {
            let o_gpu = gather(&o_cpu, &seq_rows(first));
            let s_gpu = states[first].s.to_device(&cpu).unwrap();
            let od = max_diff(o_ref, &o_gpu);
            let sd = max_diff(s_ref, &s_gpu);
            assert!(
                od <= 2e-5,
                "seq {first}: outputs diverged from reference: {od}"
            );
            assert!(
                sd <= 2e-5,
                "seq {first}: state diverged from reference: {sd}"
            );
        }
    }

    /// The op-form conv epilogue: SiLU everywhere, then l2_norm over each
    /// 128-wide head of the leading `qk_channels` — exactly what the fallback
    /// computes with `silu` and `l2_norm`. `y` is channel-major `[C, T]`.
    fn conv_epilogue_ref(y: &Tensor, qk_channels: usize, eps: f64) -> Tensor {
        use super::super::mix::{l2_norm, silu};
        let (c, t) = y.dims2().unwrap();
        let s = silu(y).unwrap();
        let qk = s
            .narrow(0, 0, qk_channels)
            .unwrap()
            .t()
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((t, qk_channels / 128, 128))
            .unwrap();
        let qk = l2_norm(&qk, eps)
            .unwrap()
            .reshape((t, qk_channels))
            .unwrap()
            .t()
            .unwrap()
            .contiguous()
            .unwrap();
        Tensor::cat(
            &[&qk, &s.narrow(0, qk_channels, c - qk_channels).unwrap()],
            0,
        )
        .unwrap()
    }

    /// The batched decode conv vs the CPU conv + epilogue, column by column,
    /// with the carried tail shifted in place across a chained decode. Two
    /// sequences stepped together on interleaved wave rows, so the pointer
    /// table's scattering is exercised; 512 channels = 2 normed Q|K heads +
    /// 2 V heads, so both epilogue regimes and the block alignment are too.
    #[test]
    fn conv_step_matches_reference() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        let (c, kw, steps, qkc) = (512usize, 4usize, 6usize, 256usize);
        let eps = 1e-6f64;
        let kern = lcg_tensor(&[c, kw], 12, &cpu);
        let kern_g = kern.to_device(&gpu).unwrap().contiguous().unwrap();

        // Two sequences with different token streams.
        let xa = lcg_tensor(&[c, steps], 11, &cpu);
        let xb = lcg_tensor(&[c, steps], 13, &cpu);

        // CPU oracles: whole segments in one call each, then the epilogue.
        let mut y_refs = Vec::new();
        let mut tail_refs = Vec::new();
        for x in [&xa, &xb] {
            let tail0 = Tensor::zeros((c, kw - 1), DType::F32, &cpu).unwrap();
            let (y_raw, tail_ref) = causal_conv1d(x, &kern, &tail0).unwrap();
            y_refs.push(conv_epilogue_ref(&y_raw, qkc, eps));
            tail_refs.push(tail_ref);
        }

        // GPU: one batched launch per step over both sequences, interleaved
        // wave rows (A on even, B on odd). The conv reads each tail and writes
        // the row's raw input into its timeline; the fold shifts it in. 512
        // channels with 256 of them Q|K is h_k 1, h_v 2 at d 128. The timeline's
        // S fields stay zero (no decode step runs), so the fold leaves S as it is.
        let dims = DeltaNetDims {
            head_dim: 128,
            n_k_heads: 1,
            n_v_heads: 2,
            conv_kernel: kw,
        };
        assert_eq!(dims.conv_dim(), c);
        let lay = TimelineLayout::of(&dims);
        let states = [
            DeltaNetState::zeros(&dims, &gpu).unwrap(),
            DeltaNetState::zeros(&dims, &gpu).unwrap(),
        ];
        let timelines = [lay.alloc(&gpu).unwrap(), lay.alloc(&gpu).unwrap()];
        let t_wave = 2 * steps;
        // The wave's raw QKV rows: row 2i = A's token i, row 2i+1 = B's.
        let mut xw = Vec::with_capacity(t_wave * c);
        let (va, vb) = (
            xa.t()
                .unwrap()
                .contiguous()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            xb.t()
                .unwrap()
                .contiguous()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
        );
        for i in 0..steps {
            xw.extend_from_slice(&va[i * c..(i + 1) * c]);
            xw.extend_from_slice(&vb[i * c..(i + 1) * c]);
        }
        let qkv = Tensor::from_vec(xw, (t_wave, c), &gpu).unwrap();
        let conved = Tensor::zeros((t_wave, c), DType::F32, &gpu).unwrap();
        for i in 0..steps {
            // The state row is unused by the conv kernel.
            let ptrs: Vec<i64> = vec![
                f32_ptr(&states[0].conv_tail, "tail").unwrap() as i64,
                f32_ptr(&states[1].conv_tail, "tail").unwrap() as i64,
                f32_ptr(&states[0].s, "state").unwrap() as i64,
                f32_ptr(&states[1].s, "state").unwrap() as i64,
                f32_ptr(&timelines[0], "timeline").unwrap() as i64,
                f32_ptr(&timelines[1], "timeline").unwrap() as i64,
            ];
            let table = DeltaNetLayerTable {
                ptrs: Tensor::from_vec(ptrs, (3, 2), &gpu).unwrap(),
                rows: Tensor::from_vec(vec![2 * i as u32, 2 * i as u32 + 1], 2, &gpu).unwrap(),
            };
            delta_net_conv_decode(&qkv, &kern_g, &table, &conved, qkc, eps as f32).unwrap();
            delta_net_fold_states(
                &[
                    (&states[0], &timelines[0], 1),
                    (&states[1], &timelines[1], 1),
                ],
                &dims,
            )
            .unwrap();
        }

        let y_cpu = conved.to_device(&cpu).unwrap();
        for (s, (y_ref, tail_ref)) in y_refs.iter().zip(tail_refs.iter()).enumerate() {
            // Gather this sequence's rows back to channel-major [C, steps].
            let idx = Tensor::from_vec(
                (0..steps).map(|i| (2 * i + s) as u32).collect::<Vec<_>>(),
                steps,
                &cpu,
            )
            .unwrap();
            let y_gpu = y_cpu
                .index_select(&idx, 0)
                .unwrap()
                .t()
                .unwrap()
                .contiguous()
                .unwrap();
            let tail_gpu = states[s].conv_tail.to_device(&cpu).unwrap();
            let yd = max_diff(y_ref, &y_gpu);
            let td = max_diff(tail_ref, &tail_gpu);
            assert!(yd <= 1e-5, "seq {s}: conv outputs diverged: {yd}");
            assert_eq!(
                td, 0.0,
                "seq {s}: the folded conv tail is not the raw inputs"
            );
        }
    }

    /// The token-parallel prefill conv against [`causal_conv1d`] + the
    /// op-form epilogue, including a carried tail from a previous segment, a
    /// segment shorter than the tail, and the advanced tail after each call.
    #[test]
    fn conv_prefill_matches_the_segment_reference() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        let (c, kw, qkc) = (512usize, 4usize, 256usize);
        let eps = 1e-6f64;
        let kern = lcg_tensor(&[c, kw], 41, &cpu);
        let kern_g = kern.to_device(&gpu).unwrap().contiguous().unwrap();

        // Segments 9, then 2 (shorter than kw−1, so the new tail mixes old
        // tail and fresh tokens), then 5 — one running CPU oracle throughout.
        let mut tail_cpu = Tensor::zeros((c, kw - 1), DType::F32, &cpu).unwrap();
        // Two buffers, exchanged after each segment — the store's live state and
        // prefill scratch, in the smallest form that still exercises the
        // kernel's contract that the entering and advanced tails are separate
        // allocations.
        let mut tail_gpu = Tensor::zeros((c, kw - 1), DType::F32, &gpu).unwrap();
        let mut tail_gpu_out = Tensor::zeros((c, kw - 1), DType::F32, &gpu).unwrap();
        for (seg, seed) in [(9usize, 42u64), (2, 43), (5, 44)] {
            let x_cm = lcg_tensor(&[c, seg], seed, &cpu); // [C, seg]
            let (y_raw, tail_next) = causal_conv1d(&x_cm, &kern, &tail_cpu).unwrap();
            let y_ref = conv_epilogue_ref(&y_raw, qkc, eps);
            tail_cpu = tail_next;

            // Kernel input is token-major; the output lands in the caller's
            // whole-wave buffer at the span's rows (here: the whole of it).
            let x_tok = x_cm
                .t()
                .unwrap()
                .contiguous()
                .unwrap()
                .to_device(&gpu)
                .unwrap();
            let conved = Tensor::zeros((seg, c), DType::F32, &gpu).unwrap();
            let tbl = span_table(&[TestSpan {
                tail: &tail_gpu,
                tail_out: &tail_gpu_out,
                state: &tail_gpu,
                state_out: &tail_gpu_out,
                start: 0,
                len: seg,
            }]);
            delta_net_conv_prefill(&x_tok, &kern_g, &tbl, qkc, eps as f32, &conved).unwrap();
            std::mem::swap(&mut tail_gpu, &mut tail_gpu_out);

            let y_cm = conved
                .t()
                .unwrap()
                .contiguous()
                .unwrap()
                .to_device(&cpu)
                .unwrap();
            let yd = max_diff(&y_ref, &y_cm);
            let td = max_diff(&tail_cpu, &tail_gpu.to_device(&cpu).unwrap());
            assert!(yd <= 1e-5, "segment {seg}: conv outputs diverged: {yd}");
            assert!(td <= 1e-6, "segment {seg}: advanced tail diverged: {td}");
        }
    }

    /// **The fused prefill scan against the sequential rule**, at the real
    /// head width (d = 128) with several chunks, a ragged tail, and h_k < h_v
    /// so the in-kernel GQA tiling is exercised — output and carried state
    /// both, everything (gates, scale, tiling) derived in-kernel.
    #[test]
    fn prefill_scan_matches_the_sequential_reference() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        // 2 full chunks + a 22-token tail at the kernel's width of 64.
        let (t, h_k, h_v, d) = (150usize, 2usize, 4usize, 128usize);
        let case = FusedCase::build(t, h_k, h_v, 51, &cpu, &gpu);
        let s0 = lcg_tensor(&[h_v, d, d], 56, &cpu);

        let (o_ref, s_ref) = delta_recurrence(
            s0.copy().unwrap(),
            &case.q_ref,
            &case.k_ref,
            &case.v_ref,
            &case.g_ref,
            &case.b_ref,
        )
        .unwrap();

        let s_gpu = s0.to_device(&gpu).unwrap().contiguous().unwrap();
        let o = Tensor::zeros((t, h_v * d), DType::F32, &gpu).unwrap();
        let tbl = span_table(&[TestSpan {
            tail: &s_gpu,
            tail_out: &s_gpu,
            state: &s_gpu,
            state_out: &s_gpu,
            start: 0,
            len: t,
        }]);
        delta_net_prefill_scan(&case.fused(&o), &tbl).unwrap();

        let o_gpu = o.reshape((t, h_v, d)).unwrap().to_device(&cpu).unwrap();
        let od = max_diff(&o_gpu, &o_ref);
        let sd = max_diff(&s_gpu.to_device(&cpu).unwrap(), &s_ref);
        println!("prefill scan vs sequential: o {od:.3e}, state {sd:.3e}");
        assert!(od < 3e-4, "outputs diverged from the sequential rule: {od}");
        assert!(sd < 3e-4, "state diverged from the sequential rule: {sd}");
    }

    /// A verify wave's spans — one to five tokens each, in one launch — match the
    /// sequential rule, output and state. These are the lengths where the intra
    /// pass sizes its row buffers to the span rather than the chunk, and where the
    /// A/kq grid's 4-wide tiles read rows past `c_len`: one- and two-token spans
    /// are the ones a buffer cut to the bare length would overrun.
    #[test]
    fn verify_width_spans_match_the_sequential_reference() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        let (h_k, h_v, d) = (2usize, 4usize, 128usize);
        for lens in [&[1usize][..], &[2, 1], &[3, 5, 1, 2]] {
            let t: usize = lens.iter().sum();
            let case = FusedCase::build(t, h_k, h_v, 61 + t as u64, &cpu, &gpu);
            let states: Vec<Tensor> = (0..lens.len())
                .map(|i| lcg_tensor(&[h_v, d, d], 700 + i as u64, &cpu))
                .collect();
            let gpu_states: Vec<Tensor> = states
                .iter()
                .map(|s| s.to_device(&gpu).unwrap().contiguous().unwrap())
                .collect();
            let mut start = 0usize;
            let rows: Vec<TestSpan<'_>> = lens
                .iter()
                .zip(&gpu_states)
                .map(|(&len, s)| {
                    let row = TestSpan {
                        tail: s,
                        tail_out: s,
                        state: s,
                        state_out: s,
                        start,
                        len,
                    };
                    start += len;
                    row
                })
                .collect();
            let o = Tensor::zeros((t, h_v * d), DType::F32, &gpu).unwrap();
            delta_net_prefill_scan(&case.fused(&o), &span_table(&rows)).unwrap();
            let o_gpu = o.reshape((t, h_v, d)).unwrap().to_device(&cpu).unwrap();

            let mut start = 0usize;
            for (i, &len) in lens.iter().enumerate() {
                let span = |x: &Tensor| x.narrow(0, start, len).unwrap().contiguous().unwrap();
                let (o_ref, s_ref) = delta_recurrence(
                    states[i].copy().unwrap(),
                    &span(&case.q_ref),
                    &span(&case.k_ref),
                    &span(&case.v_ref),
                    &span(&case.g_ref),
                    &span(&case.b_ref),
                )
                .unwrap();
                let od = max_diff(&span(&o_gpu), &o_ref);
                let sd = max_diff(&gpu_states[i].to_device(&cpu).unwrap(), &s_ref);
                assert!(od < 3e-4, "{lens:?} span {i}: output diverged by {od}");
                assert!(sd < 3e-4, "{lens:?} span {i}: state diverged by {sd}");
                start += len;
            }
        }
    }

    /// **Several sequences in one launch must equal each of them alone.**
    ///
    /// This is what the span table exists for, and the property it can break:
    /// the spans share a launch but not a state, so an index that reaches past
    /// its own span — the transients are one wave-wide allocation, and the
    /// packed buffers are rebased per span — silently mixes one sequence's rows
    /// into another's recurrence. Bit-identity is the assertion, not closeness:
    /// batching changes which blocks run together, never the arithmetic any one
    /// of them does, so a span's rows and its advanced state must come back the
    /// same to the bit as when it ran by itself.
    ///
    /// The spans are deliberately UNEQUAL in length (and neither a multiple of
    /// the 64-token chunk), because the launch is a rectangle over the widest of
    /// them: an equal-length case would never exercise the surplus-block guard.
    #[test]
    fn spans_batched_equal_spans_alone() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        let (h_k, h_v, d) = (2usize, 4usize, 128usize);
        let lens = [70usize, 33, 129];
        let t: usize = lens.iter().sum();
        let case = FusedCase::build(t, h_k, h_v, 77, &cpu, &gpu);

        let starts: Vec<usize> = lens
            .iter()
            .scan(0usize, |acc, &l| {
                let s = *acc;
                *acc += l;
                Some(s)
            })
            .collect();

        // Entering states that differ per span, so a span reading the wrong
        // one cannot coincidentally agree.
        let states: Vec<Tensor> = (0..lens.len())
            .map(|i| {
                lcg_tensor(&[h_v, d, d], 900 + i as u64, &cpu)
                    .to_device(&gpu)
                    .unwrap()
            })
            .collect();

        // Together: one table, one launch pair.
        let batched_states: Vec<Tensor> = states.iter().map(|s| s.copy().unwrap()).collect();
        let o_batched = Tensor::zeros((t, h_v * d), DType::F32, &gpu).unwrap();
        let rows: Vec<TestSpan<'_>> = (0..lens.len())
            .map(|i| TestSpan {
                tail: &batched_states[i],
                tail_out: &batched_states[i],
                state: &batched_states[i],
                state_out: &batched_states[i],
                start: starts[i],
                len: lens[i],
            })
            .collect();
        delta_net_prefill_scan(&case.fused(&o_batched), &span_table(&rows)).unwrap();

        // Alone: one table per span, into a separate output buffer.
        let o_alone = Tensor::zeros((t, h_v * d), DType::F32, &gpu).unwrap();
        let alone_states: Vec<Tensor> = states.iter().map(|s| s.copy().unwrap()).collect();
        for i in 0..lens.len() {
            let tbl = span_table(&[TestSpan {
                tail: &alone_states[i],
                tail_out: &alone_states[i],
                state: &alone_states[i],
                state_out: &alone_states[i],
                start: starts[i],
                len: lens[i],
            }]);
            delta_net_prefill_scan(&case.fused(&o_alone), &tbl).unwrap();
        }

        let b = o_batched.to_device(&cpu).unwrap().to_vec2::<f32>().unwrap();
        let a = o_alone.to_device(&cpu).unwrap().to_vec2::<f32>().unwrap();
        for (r, (br, ar)) in b.iter().zip(a.iter()).enumerate() {
            assert_eq!(
                br, ar,
                "row {r} differs between the batched and lone launches"
            );
        }
        for i in 0..lens.len() {
            let bs = batched_states[i]
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap();
            let as_ = alone_states[i]
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap();
            assert_eq!(
                bs.to_vec1::<f32>().unwrap(),
                as_.to_vec1::<f32>().unwrap(),
                "span {i}'s advanced state differs between the batched and lone launches"
            );
        }
    }

    fn f32_bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// One mixer layer's raw wave operands for the short-span tests: the QKV
    /// projection rows the conv reads, the conv weights, and the gate
    /// projections and constants. GQA 3:1, Flash-Next's ratio (48 V heads over
    /// 16 K heads), at a width a test can afford.
    struct ShortCase {
        qkv: Tensor,
        kern: Tensor,
        alpha: Tensor,
        blin: Tensor,
        dt_bias: Tensor,
        a: Tensor,
        h_k: usize,
        h_v: usize,
        conv_dim: usize,
    }

    impl ShortCase {
        const D: usize = 128;

        fn build(t: usize, seed: u64, gpu: &Device) -> Self {
            let cpu = Device::Cpu;
            let (h_k, h_v) = (2usize, 6usize);
            let conv_dim = (2 * h_k + h_v) * Self::D;
            let to = |x: Tensor| x.to_device(gpu).unwrap().contiguous().unwrap();
            Self {
                qkv: to(lcg_tensor(&[t, conv_dim], seed, &cpu)
                    .affine(4.0, 0.0)
                    .unwrap()),
                kern: to(lcg_tensor(
                    &[conv_dim, DELTA_NET_SHORT_SPAN_CONV],
                    seed + 1,
                    &cpu,
                )),
                alpha: to(lcg_tensor(&[t, h_v], seed + 2, &cpu)
                    .affine(4.0, 0.0)
                    .unwrap()),
                blin: to(lcg_tensor(&[t, h_v], seed + 3, &cpu)
                    .affine(4.0, 0.0)
                    .unwrap()),
                dt_bias: to(lcg_tensor(&[h_v], seed + 4, &cpu)),
                a: to((lcg_tensor(&[h_v], seed + 5, &cpu).abs().unwrap() + 0.1)
                    .unwrap()
                    .neg()
                    .unwrap()),
                h_k,
                h_v,
                conv_dim,
            }
        }

        fn fused<'a>(&'a self, conved: &'a Tensor, o: &'a Tensor) -> DeltaNetFused<'a, 'static> {
            DeltaNetFused {
                conved,
                alpha: &self.alpha,
                blin: &self.blin,
                dt_bias: &self.dt_bias,
                a: &self.a,
                o,
                q_scale: 1.0 / (Self::D as f32).sqrt(),
            }
        }
    }

    /// A short-span table from tensors: each span's conv tail, state and
    /// timeline, then its extents — the layout [`build_span_tables`] writes for
    /// short spans.
    fn short_table(
        rows: &[(&Tensor, &Tensor, &Tensor, usize, usize)],
    ) -> DeltaNetSpanTable<'static> {
        let n = rows.len();
        let mut ptrs: Vec<i64> = Vec::with_capacity(3 * n);
        for field in 0..3 {
            for r in rows {
                let t = [r.0, r.1, r.2][field];
                ptrs.push(f32_ptr(t, "short span").unwrap() as i64);
            }
        }
        let mut ext: Vec<u32> = rows.iter().map(|r| r.3 as u32).collect();
        ext.extend(rows.iter().map(|r| r.4 as u32));
        let dev = rows[0].0.device().clone();
        DeltaNetSpanTable {
            ptrs: Tensor::from_vec(ptrs, (3, n), &dev).unwrap(),
            spans: Tensor::from_vec(ext, (2, n), &dev).unwrap(),
            n,
            max_len: rows.iter().map(|r| r.4).max().unwrap(),
        }
    }

    /// **The fold of a short span's timeline is the three-launch scan over that
    /// many rows, to the bit, at every prefix** (`docs/deltanet_timeline.md`
    /// §10, test 1).
    ///
    /// A short span reads its state and writes a timeline; the fold of its
    /// first `m` rows must be exactly the state and conv tail the conv + scan
    /// pair writes for a span of those `m` rows alone — the arithmetic a
    /// speculative rewind is locked to. One wave carries a span of every short
    /// length, 2..=8, from random entering states and tails; for every span and
    /// every `m`, the pair runs over its first `m` rows into poisoned buffers,
    /// and the fold runs over a copy of the entering state. `S`, the conv tail
    /// and the `o` rows of the prefix must match bit for bit — the `o` rows
    /// because a row's output reads only the rows before it, whatever the span's
    /// length.
    #[test]
    fn a_short_span_timeline_folds_to_the_three_launches_bit_for_bit() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let d = ShortCase::D;
        let eps = 1e-6f32;
        let lens: Vec<usize> = (2..=DELTA_NET_SHORT_SPAN_ROWS).collect();
        let t: usize = lens.iter().sum();
        let seed = 300u64;
        let c = ShortCase::build(t, seed, &gpu);
        let (h_v, conv_dim) = (c.h_v, c.conv_dim);
        let dims = DeltaNetDims {
            head_dim: d,
            n_k_heads: c.h_k,
            n_v_heads: h_v,
            conv_kernel: DELTA_NET_SHORT_SPAN_CONV,
        };
        let lay = TimelineLayout::of(&dims);
        let starts: Vec<usize> = lens
            .iter()
            .scan(0usize, |acc, &l| {
                let s = *acc;
                *acc += l;
                Some(s)
            })
            .collect();
        let entering: Vec<Tensor> = (0..lens.len())
            .map(|i| lcg_tensor(&[h_v, d, d], seed + 10 + i as u64, &gpu))
            .collect();
        let tails: Vec<Tensor> = (0..lens.len())
            .map(|i| lcg_tensor(&[conv_dim, 3], seed + 30 + i as u64, &gpu))
            .collect();
        let poison = |shape: &[usize], s: u64| lcg_tensor(shape, s, &gpu).affine(1e3, 0.0).unwrap();

        // The short path: every span in one launch, each into its own timeline.
        // The states and tails it reads are the entering ones — it must leave
        // them as they were.
        let states: Vec<Tensor> = entering.iter().map(|s| s.copy().unwrap()).collect();
        let timelines: Vec<Tensor> = (0..lens.len())
            .map(|i| poison(&[lay.elems()], seed + 50 + i as u64))
            .collect();
        let o_short = poison(&[t, h_v * d], seed + 70);
        let geometry = Tensor::zeros((t, conv_dim), DType::F32, &gpu).unwrap();
        let rows: Vec<_> = (0..lens.len())
            .map(|i| (&tails[i], &states[i], &timelines[i], starts[i], lens[i]))
            .collect();
        delta_net_short_span_scan(
            &c.fused(&geometry, &o_short),
            &c.qkv,
            &c.kern,
            &short_table(&rows),
            eps,
        )
        .unwrap();
        for i in 0..lens.len() {
            assert_eq!(
                f32_bits(&states[i]),
                f32_bits(&entering[i]),
                "span {i}: S was written"
            );
        }
        let o_short = o_short.to_device(&Device::Cpu).unwrap();

        for (i, &len) in lens.iter().enumerate() {
            for m in 1..=len {
                // The oracle: the pair over the span's first `m` rows alone.
                let state_out = poison(&[h_v, d, d], seed + 1000 + m as u64);
                let tail_out = poison(&[conv_dim, 3], seed + 2000 + m as u64);
                let o_three = poison(&[t, h_v * d], seed + 3000 + m as u64);
                let conved = Tensor::zeros((t, conv_dim), DType::F32, &gpu).unwrap();
                let tbl = span_table(&[TestSpan {
                    tail: &tails[i],
                    tail_out: &tail_out,
                    state: &entering[i],
                    state_out: &state_out,
                    start: starts[i],
                    len: m,
                }]);
                delta_net_conv_prefill(&c.qkv, &c.kern, &tbl, 2 * c.h_k * d, eps, &conved).unwrap();
                delta_net_prefill_scan(&c.fused(&conved, &o_three), &tbl).unwrap();

                // The fold of the first `m` rows into a copy of the entering state.
                let folded = DeltaNetState {
                    s: entering[i].copy().unwrap(),
                    conv_tail: tails[i].copy().unwrap(),
                };
                delta_net_fold_states(&[(&folded, &timelines[i], m)], &dims).unwrap();

                assert_eq!(
                    f32_bits(&folded.s),
                    f32_bits(&state_out),
                    "span of {len}, m = {m}: S"
                );
                assert_eq!(
                    f32_bits(&folded.conv_tail),
                    f32_bits(&tail_out),
                    "span of {len}, m = {m}: conv tail"
                );
                let o_three = o_three.to_device(&Device::Cpu).unwrap();
                assert_eq!(
                    f32_bits(&o_three.narrow(0, starts[i], m).unwrap()),
                    f32_bits(&o_short.narrow(0, starts[i], m).unwrap()),
                    "span of {len}, m = {m}: o"
                );
            }
        }
    }

    /// **The tensor-op fold is the kernel fold** (`docs/deltanet_timeline.md`
    /// §10, test 3b): from the timelines the short-span kernel writes, for every
    /// span length 2..=8 and every kept prefix, `fold_by_ops` lands the state the
    /// fold kernel lands within the reference's parity tolerance, and the conv
    /// tail exactly — the tail is a copy of raw inputs, with no arithmetic to
    /// differ in.
    #[test]
    fn the_tensor_op_fold_matches_the_kernel_fold() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        use super::super::timeline::fold_by_ops;
        let d = ShortCase::D;
        let lens: Vec<usize> = (2..=DELTA_NET_SHORT_SPAN_ROWS).collect();
        let t: usize = lens.iter().sum();
        let seed = 500u64;
        let c = ShortCase::build(t, seed, &gpu);
        let (h_v, conv_dim) = (c.h_v, c.conv_dim);
        let dims = DeltaNetDims {
            head_dim: d,
            n_k_heads: c.h_k,
            n_v_heads: h_v,
            conv_kernel: DELTA_NET_SHORT_SPAN_CONV,
        };
        let lay = TimelineLayout::of(&dims);
        let starts: Vec<usize> = lens
            .iter()
            .scan(0usize, |acc, &l| {
                let s = *acc;
                *acc += l;
                Some(s)
            })
            .collect();
        let entering: Vec<Tensor> = (0..lens.len())
            .map(|i| lcg_tensor(&[h_v, d, d], seed + 10 + i as u64, &gpu))
            .collect();
        let tails: Vec<Tensor> = (0..lens.len())
            .map(|i| lcg_tensor(&[conv_dim, 3], seed + 30 + i as u64, &gpu))
            .collect();
        let timelines: Vec<Tensor> = (0..lens.len()).map(|_| lay.alloc(&gpu).unwrap()).collect();
        let o = Tensor::zeros((t, h_v * d), DType::F32, &gpu).unwrap();
        let geometry = Tensor::zeros((t, conv_dim), DType::F32, &gpu).unwrap();
        let rows: Vec<_> = (0..lens.len())
            .map(|i| (&tails[i], &entering[i], &timelines[i], starts[i], lens[i]))
            .collect();
        delta_net_short_span_scan(
            &c.fused(&geometry, &o),
            &c.qkv,
            &c.kern,
            &short_table(&rows),
            1e-6,
        )
        .unwrap();

        for (i, &len) in lens.iter().enumerate() {
            for m in 1..=len {
                let kernel = DeltaNetState {
                    s: entering[i].copy().unwrap(),
                    conv_tail: tails[i].copy().unwrap(),
                };
                delta_net_fold_states(&[(&kernel, &timelines[i], m)], &dims).unwrap();
                let mut ops = DeltaNetState {
                    s: entering[i].copy().unwrap(),
                    conv_tail: tails[i].copy().unwrap(),
                };
                fold_by_ops(&mut ops, &timelines[i], m, &dims).unwrap();
                let sd = max_diff(&kernel.s, &ops.s);
                assert!(sd < 1e-4, "span of {len}, m = {m}: S differs by {sd}");
                assert_eq!(
                    f32_bits(&kernel.conv_tail),
                    f32_bits(&ops.conv_tail),
                    "span of {len}, m = {m}: conv tail"
                );
            }
        }
    }

    /// **Each span of a mixed wave takes its own kernel** (§10, test 6): a wave
    /// carrying two verify blocks beside a prefill chunk runs the blocks through
    /// the short-span kernel into their timelines and the chunk through the
    /// conv + scan pair into its scratch — and every span's output rows, state
    /// and conv tail are, bit for bit, what it produces running alone.
    ///
    /// The short spans' states are checked untouched before their fold, which
    /// is what says they took the timeline path at all; the long span's live
    /// state likewise, because its advance belongs in the scratch.
    #[test]
    fn a_mixed_wave_runs_each_span_through_its_own_kernel() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        use super::super::mix::{
            delta_net_mix_spans, finish_solo, solo_write, DeltaNetConstants, DeltaNetOut,
            DeltaNetProjections, SpanWrite,
        };
        let dims = DeltaNetDims {
            head_dim: 128,
            n_k_heads: 1,
            n_v_heads: 2,
            conv_kernel: 4,
        };
        let lens = [5usize, TIMELINE_ROWS + 12, 3];
        let total: usize = lens.iter().sum();
        let starts = [0usize, lens[0], lens[0] + lens[1]];
        let p = DeltaNetProjections {
            qkv: lcg_tensor(&[total, dims.conv_dim()], 11, &gpu),
            z: lcg_tensor(&[total, dims.value_dim()], 12, &gpu),
            beta_lin: lcg_tensor(&[total, dims.n_v_heads], 13, &gpu),
            alpha_lin: lcg_tensor(&[total, dims.n_v_heads], 14, &gpu),
        };
        let dt_bias = lcg_tensor(&[dims.n_v_heads], 15, &gpu);
        let a = lcg_tensor(&[dims.n_v_heads], 16, &gpu)
            .abs()
            .unwrap()
            .affine(1.0, 0.1)
            .unwrap()
            .neg()
            .unwrap();
        let conv = lcg_tensor(&[dims.conv_dim(), dims.conv_kernel], 17, &gpu);
        let norm = lcg_tensor(&[dims.head_dim], 18, &gpu).abs().unwrap();
        let c = DeltaNetConstants {
            dt_bias: &dt_bias,
            a: &a,
            conv: &conv,
            norm: &norm,
        };
        let entering = |i: usize| DeltaNetState {
            s: lcg_tensor(&[2, 128, 128], 40 + i as u64, &gpu),
            conv_tail: lcg_tensor(&[dims.conv_dim(), 3], 60 + i as u64, &gpu),
        };

        // Each span alone, over its own rows of the same projections.
        let mut alone_o = Vec::new();
        let mut alone_states = Vec::new();
        for i in 0..lens.len() {
            let rows = |x: &Tensor| {
                x.narrow(0, starts[i], lens[i])
                    .unwrap()
                    .contiguous()
                    .unwrap()
            };
            let ps = DeltaNetProjections {
                qkv: rows(&p.qkv),
                z: rows(&p.z),
                beta_lin: rows(&p.beta_lin),
                alpha_lin: rows(&p.alpha_lin),
            };
            let mut st = entering(i);
            let write = solo_write(&st, lens[i], &dims).unwrap();
            let mut seqs = [DeltaNetSeq {
                start: 0,
                len: lens[i],
                state: &mut st,
                write,
            }];
            let o = delta_net_mix_spans(&ps, &c, &dims, &mut seqs, 1e-6, None, ZGate::Silu, None)
                .unwrap();
            let [s] = seqs;
            finish_solo(s.state, &s.write, s.len, &dims).unwrap();
            alone_o.push(o.to_device(&Device::Cpu).unwrap());
            alone_states.push(st);
        }

        // Together: one call, a timeline for each block and — as a store hands
        // a prefilling sequence — a scratch apart from the live state for the
        // chunk.
        let mut states: Vec<DeltaNetState> = (0..lens.len()).map(entering).collect();
        let scratch = DeltaNetOut {
            s: Tensor::zeros((2, 128, 128), DType::F32, &gpu).unwrap(),
            conv_tail: Tensor::zeros((dims.conv_dim(), 3), DType::F32, &gpu).unwrap(),
        };
        let writes: Vec<SpanWrite> = states
            .iter()
            .zip(lens)
            .map(|(s, l)| match l > TIMELINE_ROWS {
                true => SpanWrite::State(scratch.clone()),
                false => solo_write(s, l, &dims).unwrap(),
            })
            .collect();
        assert!(writes[0].timeline().is_some() && writes[2].timeline().is_some());
        assert!(writes[1].state().is_some());
        let mixed = {
            let mut seqs: Vec<DeltaNetSeq<'_>> = states
                .iter_mut()
                .zip(writes.iter())
                .enumerate()
                .map(|(i, (state, write))| DeltaNetSeq {
                    start: starts[i],
                    len: lens[i],
                    state,
                    write: write.clone(),
                })
                .collect();
            delta_net_mix_spans(&p, &c, &dims, &mut seqs, 1e-6, None, ZGate::Silu, None).unwrap()
        };
        for (i, st) in states.iter().enumerate() {
            assert_eq!(
                f32_bits(&st.s),
                f32_bits(&entering(i).s),
                "span {i}: the wave wrote the live state it read"
            );
        }
        // The blocks fold whole; the chunk's advance is the scratch, as a
        // commit would make it.
        for i in [0, 2] {
            finish_solo(&mut states[i], &writes[i], lens[i], &dims).unwrap();
        }
        states[1] = DeltaNetState {
            s: scratch.s.clone(),
            conv_tail: scratch.conv_tail.clone(),
        };
        let mixed = mixed.to_device(&Device::Cpu).unwrap();
        for i in 0..lens.len() {
            assert_eq!(
                f32_bits(&mixed.narrow(0, starts[i], lens[i]).unwrap()),
                f32_bits(&alone_o[i]),
                "span {i}: output rows"
            );
            assert_eq!(
                f32_bits(&states[i].s),
                f32_bits(&alone_states[i].s),
                "span {i}: state"
            );
            assert_eq!(
                f32_bits(&states[i].conv_tail),
                f32_bits(&alone_states[i].conv_tail),
                "span {i}: conv tail"
            );
        }
    }

    /// A span past [`DELTA_NET_SHORT_SPAN_ROWS`] is refused by name, before
    /// anything launches — the pair serves it.
    #[test]
    fn a_nine_row_span_is_refused_by_the_short_span_launch() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let t = DELTA_NET_SHORT_SPAN_ROWS + 1;
        let c = ShortCase::build(t, 77, &gpu);
        let d = ShortCase::D;
        let state = Tensor::zeros((c.h_v, d, d), DType::F32, &gpu).unwrap();
        let tail = Tensor::zeros((c.conv_dim, 3), DType::F32, &gpu).unwrap();
        let timeline = Tensor::zeros(16, DType::F32, &gpu).unwrap();
        let o = Tensor::zeros((t, c.h_v * d), DType::F32, &gpu).unwrap();
        let conved = Tensor::zeros((t, c.conv_dim), DType::F32, &gpu).unwrap();
        let tbl = short_table(&[(&tail, &state, &timeline, 0, t)]);
        let err = delta_net_short_span_scan(&c.fused(&conved, &o), &c.qkv, &c.kern, &tbl, 1e-6)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("the short-span scan takes spans of at most 8 rows, got one of 9"),
            "{err}"
        );
        assert_eq!(
            f32_bits(&o).iter().filter(|&&b| b != 0).count(),
            0,
            "a refused launch wrote output"
        );
    }

    /// The conv half of the same property: every span's tail advanced in one
    /// launch must equal each advanced alone.
    #[test]
    fn conv_spans_batched_equal_spans_alone() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        let (c, kw, qkc, eps) = (768usize, 4usize, 512usize, 1e-6f64);
        let lens = [5usize, 17, 2];
        let t: usize = lens.iter().sum();
        let starts: Vec<usize> = lens
            .iter()
            .scan(0usize, |acc, &l| {
                let s = *acc;
                *acc += l;
                Some(s)
            })
            .collect();

        let kern = lcg_tensor(&[c, kw], 51, &cpu).to_device(&gpu).unwrap();
        let qkv = lcg_tensor(&[t, c], 52, &cpu).to_device(&gpu).unwrap();
        let tails: Vec<Tensor> = (0..lens.len())
            .map(|i| {
                lcg_tensor(&[c, kw - 1], 60 + i as u64, &cpu)
                    .to_device(&gpu)
                    .unwrap()
            })
            .collect();

        let run = |outs: &[Tensor], batched: bool| -> Tensor {
            let conved = Tensor::zeros((t, c), DType::F32, &gpu).unwrap();
            let mk = |i: usize| TestSpan {
                tail: &tails[i],
                tail_out: &outs[i],
                state: &tails[i],
                state_out: &outs[i],
                start: starts[i],
                len: lens[i],
            };
            if batched {
                let rows: Vec<TestSpan<'_>> = (0..lens.len()).map(mk).collect();
                delta_net_conv_prefill(&qkv, &kern, &span_table(&rows), qkc, eps as f32, &conved)
                    .unwrap();
            } else {
                for i in 0..lens.len() {
                    delta_net_conv_prefill(
                        &qkv,
                        &kern,
                        &span_table(&[mk(i)]),
                        qkc,
                        eps as f32,
                        &conved,
                    )
                    .unwrap();
                }
            }
            conved
        };

        let outs_b: Vec<Tensor> = (0..lens.len())
            .map(|_| Tensor::zeros((c, kw - 1), DType::F32, &gpu).unwrap())
            .collect();
        let outs_a: Vec<Tensor> = (0..lens.len())
            .map(|_| Tensor::zeros((c, kw - 1), DType::F32, &gpu).unwrap())
            .collect();
        let cb = run(&outs_b, true).to_device(&cpu).unwrap();
        let ca = run(&outs_a, false).to_device(&cpu).unwrap();
        assert_eq!(
            cb.to_vec2::<f32>().unwrap(),
            ca.to_vec2::<f32>().unwrap(),
            "conv output differs between the batched and lone launches"
        );
        for i in 0..lens.len() {
            assert_eq!(
                outs_b[i].to_device(&cpu).unwrap().to_vec2::<f32>().unwrap(),
                outs_a[i].to_device(&cpu).unwrap().to_vec2::<f32>().unwrap(),
                "span {i}'s advanced conv tail differs"
            );
        }
    }

    /// Segmenting a sequence across calls must equal one call over the whole
    /// of it — the property turn sealing, forks and resume rest on, evaluated
    /// through the kernel (segment boundaries fall mid-chunk on purpose, and
    /// the segments are spans of one wave, exactly as the mixer issues them).
    #[test]
    fn prefill_scan_carries_state_across_segments() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        let (t, h_k, h_v, d) = (130usize, 2usize, 4usize, 128usize);
        let case = FusedCase::build(t, h_k, h_v, 61, &cpu, &gpu);

        let s_one = Tensor::zeros((h_v, d, d), DType::F32, &gpu).unwrap();
        let o_one = Tensor::zeros((t, h_v * d), DType::F32, &gpu).unwrap();
        let one = |s: &Tensor, start: usize, len: usize| {
            span_table(&[TestSpan {
                tail: s,
                tail_out: s,
                state: s,
                state_out: s,
                start,
                len,
            }])
        };
        delta_net_prefill_scan(&case.fused(&o_one), &one(&s_one, 0, t)).unwrap();

        // 70 + 60: neither boundary is a multiple of the 64-token chunk. Two
        // CALLS, not two spans of one table: the segments are the same sequence
        // and the second must enter with the state the first left, where spans
        // sharing a launch are independent and all enter with their own.
        let s_seg = Tensor::zeros((h_v, d, d), DType::F32, &gpu).unwrap();
        let o_seg = Tensor::zeros((t, h_v * d), DType::F32, &gpu).unwrap();
        let fused = case.fused(&o_seg);
        delta_net_prefill_scan(&fused, &one(&s_seg, 0, 70)).unwrap();
        delta_net_prefill_scan(&fused, &one(&s_seg, 70, 60)).unwrap();

        // The chunk grouping differs (the second call re-chunks from its own
        // origin), so this is numerical closeness, not bitwise identity.
        let od = max_diff(&o_one, &o_seg);
        let sd = max_diff(&s_one, &s_seg);
        println!("prefill scan segmented vs one-shot: o {od:.3e}, state {sd:.3e}");
        assert!(od < 3e-4, "segmented outputs diverged: {od}");
        assert!(sd < 3e-4, "segmented state diverged: {sd}");
    }

    /// Component-level timing of the fused mixer kernels at the 9B's real
    /// geometry — the harness the end-to-end `dn:mix` span cannot be: it
    /// splits the per-layer cost into conv / intra / state / norm_gate (and
    /// the decode step), with no `profile_sync` overhead in the numbers.
    ///
    /// `#[ignore]` because it is a benchmark, not an assertion — it prints.
    /// Run alone on an idle GPU; this machine's run-to-run band is ~2×, so
    /// compare components within one run, not absolutes across runs.
    #[test]
    #[ignore = "GPU timing benchmark — prints per-kernel µs, asserts nothing"]
    fn fused_mixer_kernel_timing() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        // The 9B mixer geometry: 16 K heads, 32 V heads, d = 128,
        // conv_dim = (2·16 + 32)·128 = 8192, a 649-token gate prompt.
        let (t, h_k, h_v, d, kw) = (649usize, 16usize, 32usize, 128usize, 4usize);
        let conv_dim = (2 * h_k + h_v) * d;
        let case = FusedCase::build(t, h_k, h_v, 81, &cpu, &gpu);
        let qkv = lcg_tensor(&[t, conv_dim], 82, &gpu);
        let kern = lcg_tensor(&[conv_dim, kw], 83, &gpu);
        let tail = Tensor::zeros((conv_dim, kw - 1), DType::F32, &gpu).unwrap();
        let state = Tensor::zeros((h_v, d, d), DType::F32, &gpu).unwrap();
        let o = Tensor::zeros((t, h_v * d), DType::F32, &gpu).unwrap();
        let z = lcg_tensor(&[t, h_v * d], 84, &gpu);
        let gain = lcg_tensor(&[d], 85, &gpu);
        let fused = case.fused(&o);
        let qk_channels = 2 * h_k * d;

        let time = |label: &str, f: &mut dyn FnMut()| {
            for _ in 0..5 {
                f();
            }
            gpu.synchronize().unwrap();
            let reps = 50;
            let start = std::time::Instant::now();
            for _ in 0..reps {
                f();
            }
            gpu.synchronize().unwrap();
            let us = start.elapsed().as_secs_f64() * 1e6 / reps as f64;
            eprintln!("  {label:<26} {us:9.1} µs/call");
            us
        };

        eprintln!("--- fused mixer kernels, 9B geometry, T={t} ---");
        let conved = Tensor::zeros((t, conv_dim), DType::F32, &gpu).unwrap();
        let tail_out = Tensor::zeros((conv_dim, kw - 1), DType::F32, &gpu).unwrap();
        let whole = span_table(&[TestSpan {
            tail: &tail,
            tail_out: &tail_out,
            state: &state,
            state_out: &state,
            start: 0,
            len: t,
        }]);
        let c_us = time("conv_prefill (+epilogue)", &mut || {
            delta_net_conv_prefill(&qkv, &kern, &whole, qk_channels, 1e-6, &conved).unwrap();
        });
        let s_us = time("scan (intra + state)", &mut || {
            delta_net_prefill_scan(&fused, &whole).unwrap();
        });
        let g_us = time("norm_gate", &mut || {
            let _ = delta_net_norm_gate(&o, &z, &gain, d, 1e-6, super::super::ZGate::Silu).unwrap();
        });
        eprintln!(
            "  {:<26} {:9.1} µs/layer (T={t} prefill)",
            "TOTAL",
            c_us + s_us + g_us
        );
        // Decode: batched over 4 sequences (the gate's multi-context shape),
        // one launch pair regardless of the count.
        let n_dec = 4usize;
        let dims = DeltaNetDims {
            head_dim: d,
            n_k_heads: h_k,
            n_v_heads: h_v,
            conv_kernel: kw,
        };
        let lay = TimelineLayout::of(&dims);
        let dec_states: Vec<DeltaNetState> = (0..n_dec)
            .map(|_| DeltaNetState::zeros(&dims, &gpu).unwrap())
            .collect();
        let dec_timelines: Vec<Tensor> = (0..n_dec).map(|_| lay.alloc(&gpu).unwrap()).collect();
        let mut ptrs: Vec<i64> = Vec::new();
        for st in &dec_states {
            ptrs.push(f32_ptr(&st.conv_tail, "tail").unwrap() as i64);
        }
        for st in &dec_states {
            ptrs.push(f32_ptr(&st.s, "state").unwrap() as i64);
        }
        for tl in &dec_timelines {
            ptrs.push(f32_ptr(tl, "timeline").unwrap() as i64);
        }
        let dec_table = DeltaNetLayerTable {
            ptrs: Tensor::from_vec(ptrs, (3, n_dec), &gpu).unwrap(),
            rows: Tensor::from_vec((0..n_dec as u32).collect::<Vec<_>>(), n_dec, &gpu).unwrap(),
        };
        let d_us = time("decode step (batch of 4)", &mut || {
            delta_net_decode_batch(&fused, &dec_table).unwrap();
        });
        let dc = time("conv_decode (batch of 4)", &mut || {
            delta_net_conv_decode(&qkv, &kern, &dec_table, &conved, qk_channels, 1e-6).unwrap();
        });
        let jobs: Vec<(&DeltaNetState, &Tensor, usize)> = dec_states
            .iter()
            .zip(&dec_timelines)
            .map(|(s, tl)| (s, tl, 1))
            .collect();
        let df = time("fold (4 × 1 row)", &mut || {
            delta_net_fold_states(&jobs, &dims).unwrap();
        });
        let jobs8: Vec<(&DeltaNetState, &Tensor, usize)> = dec_states
            .iter()
            .zip(&dec_timelines)
            .map(|(s, tl)| (s, tl, TIMELINE_ROWS))
            .collect();
        time("fold (4 × 8 rows)", &mut || {
            delta_net_fold_states(&jobs8, &dims).unwrap();
        });
        eprintln!(
            "  {:<26} {:9.1} µs/layer (4-seq decode)",
            "TOTAL",
            d_us + dc + df
        );
    }

    /// The fused epilogue against the op forms it replaced: per-head RMS norm
    /// with the gain, gated by the z-gate — BOTH instantiations (SiLU for the
    /// Qwen3.5 lineage, sigmoid for qwen4exp), because a kernel reused at a
    /// new configuration extends its harness to cover it (§0.4 rule 4).
    #[test]
    fn norm_gate_matches_the_op_epilogue() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        let (t, h_v, d) = (37usize, 4usize, 128usize);
        let eps = 1e-6f64;
        let o = lcg_tensor(&[t, h_v * d], 71, &cpu);
        let z = lcg_tensor(&[t, h_v * d], 72, &cpu);
        let gain = lcg_tensor(&[d], 73, &cpu);

        // The op-form reference, exactly as the fallback computes it.
        let o3 = o.reshape((t, h_v, d)).unwrap();
        let z3 = z.reshape((t, h_v, d)).unwrap();
        let ms = o3.sqr().unwrap().mean_keepdim(candle::D::Minus1).unwrap();
        let denom = (ms + eps).unwrap().sqrt().unwrap();
        let normed = o3
            .broadcast_div(&denom)
            .unwrap()
            .broadcast_mul(&gain)
            .unwrap();
        let to = |x: &Tensor| x.to_device(&gpu).unwrap().contiguous().unwrap();

        for zgate in [super::super::ZGate::Silu, super::super::ZGate::Sigmoid] {
            let sig_z = candle_nn::ops::sigmoid(&z3).unwrap();
            let gate = match zgate {
                super::super::ZGate::Silu => z3.broadcast_mul(&sig_z).unwrap(),
                super::super::ZGate::Sigmoid => sig_z,
            };
            let want = normed.mul(&gate).unwrap().reshape((t, h_v * d)).unwrap();
            let got = delta_net_norm_gate(&to(&o), &to(&z), &to(&gain), d, eps as f32, zgate)
                .unwrap()
                .to_device(&cpu)
                .unwrap();
            let diff = max_diff(&got, &want);
            assert!(
                diff <= 2e-5,
                "epilogue diverged from the op forms under {zgate:?}: {diff}"
            );
        }
    }

    /// Each q8a128 tile of `op` as its 128 int8 codes then its 4-byte
    /// `{scale, Σx}` header, in tile order — the operand's bytes without the
    /// super-block padding, which no reader touches.
    fn tile_bytes(op: &Q8a128Operand<'_>, dev: &candle::cuda_backend::CudaDevice) -> Vec<u8> {
        use candle::quantized::cuda::Q8a128Data;
        const BLK: usize = 1152;
        const META: usize = 1024;
        let raw = match &op.data {
            Q8a128Data::Tensor(t) => t.flatten_all().unwrap().to_vec1::<u8>().unwrap(),
            Q8a128Data::Owned(s) => dev.memcpy_dtov(s).unwrap(),
        };
        let tiles = op.rows * op.cols / 128;
        let mut out = Vec::with_capacity(tiles * 132);
        for tile in 0..tiles {
            let base = (tile / 8) * BLK;
            let qs = base + (tile % 8) * 128;
            let ds = base + META + (tile % 8) * 16;
            out.extend_from_slice(&raw[qs..qs + 128]);
            out.extend_from_slice(&raw[ds..ds + 4]);
        }
        out
    }

    /// The operand the epilogue emits is the standalone quantize of the output
    /// the F32 epilogue stores, byte for byte — under both gates and both Σx
    /// conventions, at a decode width, a verify width and a prefill-like one,
    /// at the released 48 heads (48 tiles a row, whole super-blocks of 8) and at
    /// 5 heads, whose 5-tile rows leave every super-block but the first partial.
    #[test]
    fn the_norm_gate_emits_the_quantize_bytes() {
        use candle::quantized::cuda::{to_dynamic, DynamicTensor};
        use candle::quantized::Int8Mode;
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let Device::Cuda(dev) = &gpu else {
            unreachable!("built as CUDA")
        };
        let cpu = Device::Cpu;
        let d = 128usize;
        let mode = Int8Mode::auto(&gpu);
        for h_v in [48usize, 5] {
            for t in [1usize, 5, 37] {
                let seed = 81 + (h_v * 100 + t) as u64;
                let o = lcg_tensor(&[t, h_v * d], seed, &cpu)
                    .to_device(&gpu)
                    .unwrap();
                let z = lcg_tensor(&[t, h_v * d], seed + 1, &cpu)
                    .to_device(&gpu)
                    .unwrap();
                let gain = lcg_tensor(&[d], 83, &cpu).to_device(&gpu).unwrap();
                for zgate in [ZGate::Silu, ZGate::Sigmoid] {
                    for ss in [SumScale::Raw, SumScale::ByAmax] {
                        let plain = delta_net_norm_gate(&o, &z, &gain, d, 1e-6, zgate).unwrap();
                        let op = delta_net_norm_gate_q8(&o, &z, &gain, d, 1e-6, zgate, ss).unwrap();
                        let acts = to_dynamic(&plain, mode, dev, ss).unwrap();
                        let DynamicTensor::Int8(want) = acts.as_dynamic() else {
                            panic!("an int8 mode quantizes")
                        };
                        assert_eq!(
                            tile_bytes(&op, dev),
                            tile_bytes(want, dev),
                            "h_v={h_v} t={t} {zgate:?} {ss:?}: operand bytes"
                        );
                    }
                }
            }
        }
    }
}
