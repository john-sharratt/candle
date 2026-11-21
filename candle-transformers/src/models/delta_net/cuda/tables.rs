//! Builds the pointer and extent tables a DeltaNet launch reads, once per forward
//! or per wave.
//!
//! A forward's decode table is built once before the layer sweep, because a
//! mid-sweep upload would synchronise the stream. The span tables for a wave are
//! uploaded in one pass for both kernel families, onto the wave's transient tier
//! so that the reservation covers them. A table is valid only for the forward or
//! wave it was built in, since compaction moves state slots between forwards. A
//! recurrent state cannot anchor a transient, because it carries no ticket.

use super::*;

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
    pub(super) ptrs: LiveTensor<'w>,
    /// `[n_decode]` U32 wave rows, shared by every layer.
    pub(super) rows: LiveTensor<'w>,
    /// Trunk layer index per table row, in slot order.
    pub(super) layers: Vec<usize>,
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

/// The short-span table's pointer rows: `[tail, state, timeline]` per span.
pub(super) fn short_rows(spans: &[&DeltaNetSeq<'_>]) -> Result<Vec<i64>> {
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
pub(super) fn long_rows(spans: &[&DeltaNetSeq<'_>]) -> Result<Vec<i64>> {
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
pub(super) fn span_extents(spans: &[&DeltaNetSeq<'_>]) -> Vec<u32> {
    let mut extents: Vec<u32> = Vec::with_capacity(2 * spans.len());
    extents.extend(spans.iter().map(|s| s.start as u32));
    extents.extend(spans.iter().map(|s| s.len as u32));
    extents
}

/// A span table's pointer rows, checked: a short table carries
/// [`TABLE_ROWS`], a long one [`LONG_TABLE_ROWS`], one column per span.
pub(super) fn check_table_rows(table: &DeltaNetSpanTable, rows: usize, what: &str) -> Result<()> {
    if table.ptrs.dims2()? != (rows, table.n) {
        candle::bail!(
            "delta_net cuda: {what} takes a [{rows}, {}] span table, got {:?}",
            table.n,
            table.ptrs.dims()
        );
    }
    Ok(())
}
