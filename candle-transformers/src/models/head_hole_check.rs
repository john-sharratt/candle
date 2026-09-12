//! Does a wave leave its last KV layer holding a position it never wrote?
//!
//! The paged decode kernel reads every position its layer's block table
//! commits. A position that is committed but was never written reads whatever
//! the slot held before — under `tensor-assert`, the claim poison, which is
//! NaN. The draft head's KV layer (the last one) is where this has surfaced: a
//! stale decode slot buffer had a draft step write over a committed position
//! and leave the one the host counted unwritten (see
//! `KvCache::commit_written_tokens`).
//!
//! Two places ask:
//!
//! * [`check_committed_rows`], after a wave whose head ran: each decode and
//!   prefill row has just committed `offset .. offset + input_len` to every
//!   layer its sweep reached, the last included, and its tail is read back. A
//!   hole is reported beside the first KV layer's verdict over the same
//!   positions and both layers' writer indices, which says whether the write
//!   and its commit disagreed on the chunk or on the layer.
//! * [`check_history`], after each draft step's commit: the position that step
//!   just wrote, with the device's own slice lengths beside the host's.
//!
//! The bands are read raw, through the host block table
//! (`ChunkedKvBacking::band_map`): the active K format is `R16`, which the
//! contiguous read path cannot decode. No fence of its own at the wave end —
//! it runs after the assert drain, which has already synchronised — and it
//! never fails a wave: a row it cannot read is logged and skipped, and only a
//! real hole stops the process.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use candle::cuda_backend::CudaDevice;
use candle::{Device, Result};
use candle_nn::kv_cache::{BlockBands, KvCache};

use super::batched_inference::BatchedInferenceSession;
use super::decode_kv_walk::{
    band_payload_bytes, parse_slices, parse_slot_header, read_device, scan_band, RECORD_BYTES,
};
use super::kv_cache_utils::SequenceContext;

/// Positions at the tail of each row's new range that are read back. The hole
/// has been the last slot a row's write reached, so the tail is where to look;
/// reading a whole prefill span every wave would cost a copy per band per chunk.
const TAIL_CHECKED: usize = 64;

/// Uncommitted-row reports logged before the instrument goes quiet.
const UNCOMMITTED_REPORTED: usize = 32;

/// Uncommitted rows seen after waves whose head ran.
static UNCOMMITTED: AtomicUsize = AtomicUsize::new(0);

/// The chunk and physical slot holding logical position `pos`, given each
/// chunk's `(offset, usage)` window in order.
fn slot_of(windows: &[(u16, u32)], pos: usize) -> Option<(usize, usize)> {
    let mut cum = 0usize;
    for (i, &(offset, usage)) in windows.iter().enumerate() {
        let usage = usage as usize;
        if pos < cum + usage {
            return Some((i, offset as usize + (pos - cum)));
        }
        cum += usage;
    }
    None
}

/// A committed position no write reached.
struct Hole {
    /// Every unwritten position among those read.
    positions: Vec<usize>,
    /// Tokens the layer holds.
    held: usize,
    /// Positions read.
    read: usize,
    /// The first unwritten position's chunk and slot.
    chunk: usize,
    slot: usize,
    /// The hole's chunk and the two after it, each with the slots holding
    /// finite values in its first K band — where the writes that did land went.
    written: Vec<(usize, Vec<usize>)>,
    /// The last few chunks' `(offset, usage)` windows.
    tail: Vec<(u16, u32)>,
}

/// What reading a range back found.
enum RowCheck {
    /// Every position read holds finite values.
    Clean,
    /// The layer does not hold the whole range: it commits `held` tokens.
    NotCommitted { held: usize },
    Hole(Hole),
}

/// Read `cache`'s positions `offset .. offset + len` (at most the last
/// [`TAIL_CHECKED`] of them) back raw and report any that hold a non-finite
/// value.
fn check_row(cache: &KvCache, dev: &CudaDevice, offset: usize, len: usize) -> Result<RowCheck> {
    let kc = cache.k_cache();
    let (Some(backing), Some(batch)) = (kc.chunked_backing(), kc.chunked_batch_idx()) else {
        return Ok(RowCheck::Clean);
    };
    let blocks: Vec<BlockBands> = backing.band_map(batch)?;
    let windows: Vec<(u16, u32)> = blocks.iter().map(|b| (b.offset, b.usage)).collect();
    let held: usize = windows.iter().map(|&(_, u)| u as usize).sum();
    let end = offset + len;
    if held < end {
        return Ok(RowCheck::NotCommitted { held });
    }
    let start = offset.max(end.saturating_sub(TAIL_CHECKED));
    let n_kv_head = backing.n_kv_head().max(1);
    let n_palette = blocks.first().map_or(1, |b| (b.k.len() / n_kv_head).max(1));
    let sub = backing.head_dim() / n_palette;

    // One read per band, shared by every position in its chunk.
    let mut bytes_of: HashMap<u64, Vec<u8>> = HashMap::new();
    let mut positions = Vec::new();
    for pos in start..end {
        let Some((blk, slot)) = slot_of(&windows, pos) else {
            continue;
        };
        let b = &blocks[blk];
        let mut unwritten = false;
        for band in b.k.iter().chain(b.v.iter()) {
            if band.ptr == 0 {
                continue;
            }
            let Some(payload) = band_payload_bytes(band.fmt, sub) else {
                continue;
            };
            let bytes = match bytes_of.get(&band.ptr) {
                Some(b) => b,
                None => {
                    // SAFETY: `band.ptr` is a live slot the host block table
                    // names for this chunk, and `payload` is its format's band
                    // size, which fits inside the slot's class stride.
                    let read = unsafe { read_device(dev, band.ptr, payload)? };
                    bytes_of.entry(band.ptr).or_insert(read)
                }
            };
            if scan_band(band.fmt, bytes, sub, slot..slot + 1)?.nonfinite > 0 {
                unwritten = true;
                break;
            }
        }
        if unwritten {
            positions.push(pos);
        }
    }
    let Some(&first) = positions.first() else {
        return Ok(RowCheck::Clean);
    };
    let (chunk, slot) = slot_of(&windows, first).unwrap_or((usize::MAX, usize::MAX));
    // Every slot of the hole's chunk and the two after it, judged by their
    // first K band: a write that went to the wrong slot, or to the wrong
    // chunk, shows up as a written slot no window counts.
    let mut written = Vec::new();
    for (c, block) in blocks.iter().enumerate().skip(chunk).take(3) {
        let Some(band) = block.k.iter().find(|x| x.ptr != 0).copied() else {
            continue;
        };
        let Some(payload) = band_payload_bytes(band.fmt, sub) else {
            continue;
        };
        let bytes = match bytes_of.get(&band.ptr) {
            Some(b) => b.clone(),
            // SAFETY: as above — a live slot the host block table names.
            None => unsafe { read_device(dev, band.ptr, payload)? },
        };
        let mut slots = Vec::new();
        for s in 0..32 {
            if scan_band(band.fmt, &bytes, sub, s..s + 1)?.nonfinite == 0 {
                slots.push(s);
            }
        }
        written.push((c, slots));
    }
    Ok(RowCheck::Hole(Hole {
        positions,
        held,
        read: end - start,
        chunk,
        slot,
        written,
        tail: windows.iter().rev().take(6).rev().copied().collect(),
    }))
}

/// A layer's writer boundary and writer chunk, as the host holds them.
fn writer_of(cache: &KvCache) -> String {
    let kc = cache.k_cache();
    match (kc.chunked_backing(), kc.chunked_batch_idx()) {
        (Some(backing), Some(batch)) => match backing.writer_indices(batch) {
            Ok(w) => format!("{w:?}"),
            Err(e) => format!("unreadable ({e})"),
        },
        _ => "not chunked".to_string(),
    }
}

/// One line on what reading a range back found.
fn verdict(check: Result<RowCheck>) -> String {
    match check {
        Ok(RowCheck::Clean) => "every position written".to_string(),
        Ok(RowCheck::NotCommitted { held }) => format!("not committed (the layer holds {held})"),
        Ok(RowCheck::Hole(h)) => format!(
            "unwritten {:?}, written slots by chunk {:?}",
            h.positions, h.written
        ),
        Err(e) => format!("unreadable ({e})"),
    }
}

/// Check every decode and prefill row of a wave whose head ran. `seqs` is the
/// wave's sequence ids in the same `[decode | prefill | glue]` order as
/// `contexts`.
pub(crate) fn check_committed_rows(
    contexts: &[SequenceContext<'_>],
    seqs: &[usize],
    n_decode: usize,
    n_prefill: usize,
) {
    for (i, ctx) in contexts.iter().enumerate().take(n_decode + n_prefill) {
        let Device::Cuda(dev) = ctx.input_ids.device() else {
            continue;
        };
        let Some(cache) = ctx.kv_caches.caches.last() else {
            continue;
        };
        let kind = if i < n_decode { "decode" } else { "prefill" };
        match check_row(cache, dev, ctx.offset, ctx.input_len) {
            Ok(RowCheck::Clean) => {}
            // A creep row whose layer window has not reached the last layer
            // yet commits it in a later wave. Said out loud rather than
            // skipped in silence: a decode row, or a verify block, that lands
            // here is a row this check could not vouch for.
            Ok(RowCheck::NotCommitted { held }) => {
                let n = UNCOMMITTED.fetch_add(1, Ordering::Relaxed) + 1;
                if n <= UNCOMMITTED_REPORTED {
                    tracing::warn!(
                        target: "candle_transformers::head_hole_check",
                        seq = seqs[i],
                        kind,
                        offset = ctx.offset,
                        len = ctx.input_len,
                        held,
                        occurrence = n,
                        "a row's range is not committed to the last KV layer after a wave \
                         whose head ran — not checked"
                    );
                }
            }
            Ok(RowCheck::Hole(h)) => {
                let first = ctx.kv_caches.caches.first();
                let first_verdict = first.map_or_else(
                    || "no first layer".to_string(),
                    |c| verdict(check_row(c, dev, ctx.offset, ctx.input_len)),
                );
                panic!(
                    "HEAD KV HOLE: this wave committed position {} of sequence {} to the last KV \
                     layer without writing it ({kind} row, offset {}, {} new tokens; the layer \
                     holds {}). Unwritten among the {} positions read: {:?}. Position {} is \
                     chunk {} slot {}. Written slots by chunk, from the hole's: {:?}. Last chunk \
                     windows (offset, tokens): {:?}. Head layer writer: {}. First KV layer \
                     writer: {}; over the same positions it reads: {first_verdict}. Wave: \
                     {n_decode} decode rows, {n_prefill} prefill rows.",
                    h.positions[0],
                    seqs[i],
                    ctx.offset,
                    ctx.input_len,
                    h.held,
                    h.read,
                    h.positions,
                    h.positions[0],
                    h.chunk,
                    h.slot,
                    h.written,
                    h.tail,
                    writer_of(cache),
                    first.map_or_else(|| "none".to_string(), writer_of),
                )
            }
            // An instrument that cannot read its data must not take the wave
            // down with it.
            Err(e) => tracing::warn!(
                target: "candle_transformers::head_hole_check",
                seq = seqs[i],
                kind,
                offset = ctx.offset,
                "head hole check could not read this row: {e}"
            ),
        }
    }
}

/// Check the head layer's last `window` positions below each sequence's
/// `end` — the history a draft step is about to attend, or the position a
/// draft step just wrote.
///
/// Called after each draft step's commit with a one-position window, it names
/// the step whose own write did not land.
///
/// `headers`, when given, is the device header table the step's kernel read —
/// one `SlotHeader` per entry of `seqs`, in order — and a hole then reports the
/// device's own slice lengths beside the host's windows.
pub(crate) fn check_history(
    session: &BatchedInferenceSession,
    seqs: &[usize],
    ends: &[usize],
    window: usize,
    kv_layer: usize,
    headers: Option<u64>,
    stage: &'static str,
) {
    let Device::Cuda(dev) = session.device() else {
        return;
    };
    for (row, (&seq, &base)) in seqs.iter().zip(ends).enumerate() {
        let Some(cache) = session.sequence_caches(seq).and_then(|c| c.caches.get(kv_layer)) else {
            continue;
        };
        let len = base.min(window);
        match check_row(cache, dev, base - len, len) {
            Ok(RowCheck::Clean) => {}
            Ok(RowCheck::NotCommitted { held }) => tracing::error!(
                target: "candle_transformers::head_hole_check",
                seq,
                stage,
                base,
                held,
                "the head layer holds fewer tokens than the draft's base — the draft will \
                 attend positions this layer never committed"
            ),
            Ok(RowCheck::Hole(h)) => {
                let device = match headers {
                    Some(table) => device_slices(dev, table, row),
                    None => "not given".to_string(),
                };
                panic!(
                    "HEAD KV HOLE at {stage}: sequence {seq}'s head layer holds position {} \
                     unwritten, below {base} (the layer holds {}). Unwritten among the {} \
                     positions read: {:?}. Position {} is chunk {} slot {}. Written slots by \
                     chunk, from the hole's: {:?}. Host windows (offset, tokens), last six: \
                     {:?}. Device header and slices: {device}.",
                    h.positions[0],
                    h.held,
                    h.read,
                    h.positions,
                    h.positions[0],
                    h.chunk,
                    h.slot,
                    h.written,
                    h.tail,
                )
            }
            Err(e) => tracing::warn!(
                target: "candle_transformers::head_hole_check",
                seq,
                stage,
                "draft hole check could not read: {e}"
            ),
        }
    }
}

/// The device's own view of `row`: its `SlotHeader` in the header table at
/// `table`, and the last six slices it points at as `(offset, len, rope)` —
/// what the kernel read, beside what the host believes.
fn device_slices(dev: &CudaDevice, table: u64, row: usize) -> String {
    let read = || -> Result<String> {
        // SAFETY: `table` is the header table the step's kernel read, one
        // 16-byte header per row.
        let hdr = parse_slot_header(&unsafe {
            read_device(dev, table + (row * RECORD_BYTES) as u64, RECORD_BYTES)?
        })?;
        // SAFETY: the header names `n_slices` 16-byte slices at `slices_ptr`.
        let slices = parse_slices(&unsafe {
            read_device(dev, hdr.slices_ptr, hdr.n_slices as usize * RECORD_BYTES)?
        })?;
        let tail: Vec<(u16, u16, u32)> =
            slices.iter().rev().take(6).rev().map(|s| (s.offset, s.len, s.rope)).collect();
        Ok(format!(
            "n_slices {} write_slice {}, last six (offset, len, rope) {tail:?}",
            hdr.n_slices, hdr.write_slice
        ))
    };
    read().unwrap_or_else(|e| format!("unreadable ({e})"))
}

#[cfg(test)]
mod tests {
    use super::slot_of;

    /// The captured layout: a partial chunk, an empty pad, then full chunks.
    /// Logical position 41 is the last slot of the chunk after the pad.
    #[test]
    fn a_position_maps_to_its_chunk_and_slot_past_a_pad() {
        let w = [(0u16, 10u32), (0, 0), (0, 32), (0, 1)];
        assert_eq!(slot_of(&w, 0), Some((0, 0)));
        assert_eq!(slot_of(&w, 9), Some((0, 9)));
        assert_eq!(slot_of(&w, 10), Some((2, 0)));
        assert_eq!(slot_of(&w, 41), Some((2, 31)));
        assert_eq!(slot_of(&w, 42), Some((3, 0)));
        assert_eq!(slot_of(&w, 43), None);
    }

    /// A window that starts mid-chunk counts its slots from its own offset.
    #[test]
    fn a_windowed_chunk_counts_slots_from_its_offset() {
        assert_eq!(slot_of(&[(4u16, 6u32)], 2), Some((0, 6)));
    }
}
