//! Per-piece hit counters on the device (`docs/hit_driven_projection.md` §6.4).
//!
//! A projection declares the pieces it placed, as position ranges; each
//! becomes a run of blocks in a block→piece map, built once through the
//! sequence's index layout. At every sampled layer, the selection's chosen
//! blocks — and the rows that chose them — are counted into the sequence's
//! counters by one batched launch (`qsa_hits.cu`). After the wave's last
//! sampled layer a second launch publishes the counters into a mapped block
//! ([`super::hit_summary`]), which the projection reads at its next boundary
//! without waiting on the device (AGENTS hot-path invariant 3).
//!
//! The counters, the map and the launch table live on the reservation's held
//! ground, never in loose allocations.

#[cfg(feature = "cuda")]
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard};

#[cfg(feature = "cuda")]
use candle::Storage;
use candle::{DType, Device, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_kernels::simple::qsa_hits::{HitSpan, HIT_SPAN_WORDS};

#[cfg(feature = "cuda")]
use crate::models::delta_net::mix::SeqSpan;
#[cfg(feature = "cuda")]
use crate::models::held_tensor::{held_empty, held_zeros};
use crate::models::hit_layers::PieceHits;
#[cfg(feature = "cuda")]
use crate::models::qsa_selection::QsaSelection;
#[cfg(feature = "cuda")]
use crate::models::wave_buffers::upload_into;

#[cfg(feature = "cuda")]
use super::hit_summary::{self, MappedBlock, HEADER_WORDS};

/// A block no placed piece owns (`QSA_NO_PIECE` in `qsa_hits.cu`).
pub const NO_PIECE: u32 = u32::MAX;

/// Words a counter set holds past its hits: the queries counted, then the
/// publish sequence word (`COUNTER_TAIL_WORDS` in the kernel's binding).
const COUNTER_TAIL: usize = 2;

/// One projection's declared pieces and their counters.
#[derive(Debug)]
struct Counts {
    /// `[map_len]` U32: block → piece, [`NO_PIECE`] for none.
    map: Tensor,
    map_len: usize,
    /// `[n_pieces + COUNTER_TAIL]` U32: hits per piece, the queries counted,
    /// and the publish sequence word, cumulative over the declaration. Atomic
    /// accumulators, so they start at zero; never reset while the
    /// declaration stands — a read takes the difference from the last.
    counters: Tensor,
    n_pieces: usize,
    /// Blocks each piece owns in the map, host-side.
    blocks: Vec<u32>,
    /// Which declaration this is: each redeclaration counts one more.
    generation: u64,
    /// Which holder the summary block belongs to.
    token: u32,
    /// Where the device publishes the counters. Off CUDA there is none, and a
    /// read takes the counters themselves.
    #[cfg(feature = "cuda")]
    summary: Option<MappedBlock>,
    /// The publish the last read took, and what it held: a read returns
    /// what has landed since.
    read_seq: u32,
    read_hits: Vec<u32>,
    read_queries: u32,
}

/// A sequence's hit counters, shared by every view carved from it.
///
/// A projection rebuilds the PARENT slot and declares its pieces there, while
/// the turn decodes on a view forked from it: the view's rows are the reads
/// that count. So a fork holds the parent's handle, every holder counts into
/// one set of counters, a redeclaration replaces them in place — reaching
/// views carved before it — and a read through any holder sees them all.
#[derive(Debug, Clone)]
pub struct SeqHits(Arc<Mutex<Counts>>);

/// One sequence's hold on hit counters: its own, which its projection
/// declared, or the parent's it was carved from, which its rows count into.
#[derive(Debug, Clone)]
pub struct HitHolder {
    pub hits: SeqHits,
    /// Whether this sequence's projection declared these counters. Only an
    /// owner redeclares them in place or reads them; a sequence holding its
    /// parent's that declares pieces of its own gets counters of its own.
    pub owned: bool,
    /// The declaration a view was carved under. A rebuild of the parent
    /// redeclares the pieces over a new layout the view does not read, so a
    /// view counts only while this is still the counters' generation.
    pub generation: u64,
}

/// The kernel's span table as last uploaded: the words and the device copy.
/// A wave's spans and counter pointers are the same at every sampled layer,
/// and across waves until a declaration changes, so the table goes up only
/// when its words do.
pub type HitTable = Option<(Vec<u32>, Tensor)>;

/// The blocks position range `r` covers under a layout whose
/// `candidates_at(pos)` is the blocks wholly at or below `pos`: from the block
/// its first position falls in to the last block wholly inside it. The one rule
/// pinned ranges and hit pieces both go by.
pub fn block_range(r: &Range<usize>, candidates_at: impl Fn(usize) -> usize) -> Range<usize> {
    if r.is_empty() {
        return 0..0;
    }
    let lo = match r.start {
        0 => 0,
        s => candidates_at(s - 1),
    };
    lo..candidates_at(r.end - 1).max(lo)
}

/// The block→piece map of `pieces` (position ranges, in declaration order)
/// under a layout whose `to_blocks` turns a position range into the blocks it
/// covers ([`block_range`], the rule pins use too). Where two pieces
/// would share a block the later one owns it. Blocks no piece owns are
/// [`NO_PIECE`].
pub fn block_map(
    pieces: &[Range<usize>],
    to_blocks: impl Fn(&Range<usize>) -> Range<usize>,
) -> Vec<u32> {
    let blocks: Vec<Range<usize>> = pieces.iter().map(to_blocks).collect();
    let len = blocks.iter().map(|b| b.end).max().unwrap_or(0);
    let mut map = vec![NO_PIECE; len];
    for (piece, b) in blocks.iter().enumerate() {
        for slot in &mut map[b.clone()] {
            *slot = piece as u32;
        }
    }
    map
}

/// The hits and queries in `now` past `then`, both cumulative over one
/// declaration. The counters are u32 and wrap, so the difference wraps too.
fn since(now: &[u32], then: &[u32], q_now: u32, q_then: u32) -> (Vec<u32>, u64) {
    let hits = now
        .iter()
        .enumerate()
        .map(|(i, &n)| n.wrapping_sub(then.get(i).copied().unwrap_or(0)))
        .collect();
    (hits, q_now.wrapping_sub(q_then) as u64)
}

impl SeqHits {
    /// Counters for `n_pieces` pieces over `map`.
    pub fn new(map: Vec<u32>, n_pieces: usize, device: &Device) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let token = hit_summary::next_token();
        #[cfg(not(feature = "cuda"))]
        let token = 0;
        #[cfg(feature = "cuda")]
        let c = {
            let mut c = Counts::new(map, n_pieces, device, token, 0)?;
            c.adopt(None, device)?;
            c
        };
        #[cfg(not(feature = "cuda"))]
        let c = Counts::new(map, n_pieces, device, token, 0)?;
        Ok(Self(Arc::new(Mutex::new(c))))
    }

    /// Replace the declared pieces and their counters, for every holder. The
    /// counts since the last read are dropped: a caller reads them first.
    /// Returns the new declaration's generation.
    pub fn redeclare(&self, map: Vec<u32>, n_pieces: usize, device: &Device) -> Result<u64> {
        let mut c = self.lock()?;
        let generation = c.generation + 1;
        let fresh = Counts::new(map, n_pieces, device, c.token, generation)?;
        #[cfg(feature = "cuda")]
        let carried = c.summary.take();
        *c = fresh;
        #[cfg(feature = "cuda")]
        c.adopt(carried, device)?;
        Ok(generation)
    }

    /// The current declaration's generation.
    pub fn generation(&self) -> Result<u64> {
        Ok(self.lock()?.generation)
    }

    /// What has been counted since the last read, as far as the device has
    /// published it.
    ///
    /// Reads the mapped summary the device last finished writing: no copy,
    /// no wait. Counts still in flight are in the next read.
    pub fn take(&self) -> Result<PieceHits> {
        self.lock()?.take()
    }

    fn lock(&self) -> Result<MutexGuard<'_, Counts>> {
        self.0
            .lock()
            .map_err(|_| candle::Error::Msg("qsa hits: counter lock poisoned".into()))
    }
}

impl Counts {
    fn new(
        map: Vec<u32>,
        n_pieces: usize,
        device: &Device,
        token: u32,
        generation: u64,
    ) -> Result<Self> {
        let mut blocks = vec![0u32; n_pieces];
        for &p in &map {
            if let Some(b) = blocks.get_mut(p as usize) {
                *b += 1;
            }
        }
        let map_len = map.len();
        // An empty map still holds one slot, so the tensor is never zero-sized;
        // `map_len` says none of it is read.
        let map = match map_len {
            0 => vec![NO_PIECE],
            _ => map,
        };
        let (map, counters) = Self::buffers(map, n_pieces + COUNTER_TAIL, device)?;
        // The summary block is the caller's to give it ([`Self::adopt`]): a
        // redeclaration hands on the one it already holds.
        Ok(Self {
            map,
            map_len,
            counters,
            n_pieces,
            blocks,
            generation,
            token,
            #[cfg(feature = "cuda")]
            summary: None,
            read_seq: 0,
            read_hits: vec![0; n_pieces],
            read_queries: 0,
        })
    }

    /// The map, uploaded, and the counters, zeroed: on held ground on CUDA.
    fn buffers(map: Vec<u32>, counters: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        if device.is_cuda() {
            let m = held_empty(map.len(), DType::U32, device)?;
            upload_into(&m, &map)?;
            return Ok((m, held_zeros(counters, DType::U32, device)?));
        }
        let n = map.len();
        Ok((
            Tensor::from_vec(map, n, device)?,
            Tensor::zeros(counters, DType::U32, device)?,
        ))
    }

    /// Take `carried` — the previous declaration's summary block — when it
    /// holds this one's pieces, and a block that does otherwise. Off CUDA
    /// there is no block.
    #[cfg(feature = "cuda")]
    fn adopt(&mut self, carried: Option<MappedBlock>, device: &Device) -> Result<()> {
        let Device::Cuda(dev) = device else {
            if let Some(b) = carried {
                hit_summary::release(b);
            }
            return Ok(());
        };
        let words = HEADER_WORDS + self.n_pieces;
        let block = match carried {
            Some(b) if b.words() >= words => b,
            other => {
                if let Some(b) = other {
                    hit_summary::release(b);
                }
                hit_summary::acquire(dev, words)?
            }
        };
        self.summary = Some(block);
        Ok(())
    }

    /// The newest cumulative counts: the summary's last finished publish on
    /// CUDA, the counters themselves elsewhere. `None` while nothing new has
    /// landed since the last read.
    fn newest(&self) -> Result<Option<(u32, u32, Vec<u32>)>> {
        #[cfg(feature = "cuda")]
        if let Some(block) = &self.summary {
            return Ok(block
                .read(self.token, self.generation as u32, self.n_pieces)
                .filter(|s| s.seq != self.read_seq)
                .map(|s| (s.seq, s.queries, s.hits)));
        }
        let all = self.counters.to_vec1::<u32>()?;
        let queries = all[self.n_pieces];
        Ok(Some((0, queries, all[..self.n_pieces].to_vec())))
    }

    fn take(&mut self) -> Result<PieceHits> {
        let Some((seq, queries, now)) = self.newest()? else {
            return Ok(PieceHits {
                hits: vec![0; self.n_pieces],
                blocks: self.blocks.clone(),
                queries: 0,
            });
        };
        let (hits, q) = since(&now, &self.read_hits, queries, self.read_queries);
        self.read_seq = seq;
        self.read_hits = now;
        self.read_queries = queries;
        Ok(PieceHits {
            hits,
            blocks: self.blocks.clone(),
            queries: q,
        })
    }
}

#[cfg(feature = "cuda")]
impl Drop for Counts {
    fn drop(&mut self) {
        if let Some(b) = self.summary.take() {
            hit_summary::release(b);
        }
    }
}

/// Count the hits of `sel` — one sampled layer's selection over a wave whose
/// rows `spans` describe — into each counted sequence's counters, in one
/// launch, and when `publish` (the wave's last sampled layer) publish every
/// counted set into its summary block, in a second. A sequence with no
/// declared pieces is not counted, nor a view carved under an older
/// declaration. A dense row counts no hit and no query. The span table is
/// `table`'s when its words match and is uploaded otherwise.
#[cfg(feature = "cuda")]
pub fn count_hits(
    sel: &QsaSelection,
    spans: &[SeqSpan],
    hits: &HashMap<usize, HitHolder>,
    table: &mut HitTable,
    publish: bool,
    device: &Device,
) -> Result<()> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_kernels::simple::qsa_hits::{run_qsa_count_hits, run_qsa_publish_hits};
    use std::ffi::c_void;

    let Device::Cuda(dev) = device else {
        candle::bail!("qsa hits are counted on CUDA");
    };
    let stream = dev.cuda_stream();
    // Each counted span as the kernel's `HitSpan`, in the words its layout
    // test pins (`HitSpan::words`).
    let mut words: Vec<u32> = Vec::new();
    let mut guards = Vec::new();
    let mut published: Vec<u64> = Vec::new();
    let mut max_rows = 0usize;
    for span in spans {
        let Some(holder) = hits.get(&span.seq) else {
            continue;
        };
        let h = holder.hits.lock()?;
        if h.n_pieces == 0 || h.map_len == 0 {
            continue;
        }
        if !holder.owned && holder.generation != h.generation {
            continue;
        }
        let Some(summary) = h.summary.as_ref() else {
            continue;
        };
        // Taken on the launch stream, which is the only stream the buffers
        // are ever touched on, so the access event the guard records orders
        // nothing the launch below does not already follow.
        let ptr_of = |t: &Tensor| -> Result<u64> {
            let (s, l) = t.storage_and_layout();
            let slice = match &*s {
                Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                _ => candle::bail!("qsa hits: counters must be CUDA"),
            }
            .slice(l.start_offset()..);
            let (p, _g) = slice.device_ptr(&stream);
            Ok(p)
        };
        let map_ptr = ptr_of(&h.map)?;
        let cnt_ptr = ptr_of(&h.counters)?;
        // A view counts into its parent's counters: the set is published once.
        let first = !published.contains(&cnt_ptr);
        if first {
            published.push(cnt_ptr);
        }
        let entry = HitSpan {
            row_lo: span.start as u32,
            row_hi: (span.start + span.len) as u32,
            map_len: h.map_len as u32,
            n_pieces: h.n_pieces as u32,
            map: map_ptr as *const u32,
            counters: cnt_ptr as *mut u32,
            summary: summary.dev_ptr() as *mut u32,
            token: h.token,
            generation: h.generation as u32,
            publish: first as u32,
            pad: 0,
        };
        words.extend_from_slice(&entry.words());
        guards.push((h.map.clone(), h.counters.clone()));
        max_rows = max_rows.max(span.len);
    }
    let n_spans = words.len() / HIT_SPAN_WORDS;
    if n_spans == 0 {
        return Ok(());
    }
    // Words name the counters' addresses, so a table built for other
    // counters — a redeclaration — never matches.
    let fresh = match table {
        Some((cached, _)) => *cached != words,
        None => true,
    };
    if fresh {
        let t = held_empty((n_spans, HIT_SPAN_WORDS), DType::U32, device)?;
        upload_into(&t, &words)?;
        *table = Some((words, t));
    }
    let Some((_, table)) = table.as_ref() else {
        candle::bail!("qsa hits: span table missing after upload");
    };
    let (t_s, t_l) = table.storage_and_layout();
    let t_slice = match &*t_s {
        Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("qsa hits: span table must be CUDA"),
    }
    .slice(t_l.start_offset()..);
    let (e_s, e_l) = sel.entries().storage_and_layout();
    let e_slice = match &*e_s {
        Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("qsa hits: entries must be CUDA"),
    }
    .slice(e_l.start_offset()..);
    let (c_s, c_l) = sel.cnt().storage_and_layout();
    let c_slice = match &*c_s {
        Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("qsa hits: cnt must be CUDA"),
    }
    .slice(c_l.start_offset()..);
    let (t_ptr, _tg) = t_slice.device_ptr(&stream);
    let (e_ptr, _eg) = e_slice.device_ptr(&stream);
    let (c_ptr, _cg) = c_slice.device_ptr(&stream);
    candle::set_kernel_breadcrumb("run_qsa_count_hits", file!(), line!());
    unsafe {
        run_qsa_count_hits(
            e_ptr as *const u32,
            sel.stride() as i32,
            c_ptr as *const u32,
            t_ptr as *const c_void,
            n_spans as i32,
            max_rows as i32,
            stream.cu_stream() as *mut c_void,
        );
    }
    if publish {
        candle::set_kernel_breadcrumb("run_qsa_publish_hits", file!(), line!());
        unsafe {
            run_qsa_publish_hits(
                t_ptr as *const c_void,
                n_spans as i32,
                stream.cu_stream() as *mut c_void,
            );
        }
    }
    // The map and counter buffers stay owned by their `SeqHits` across the
    // launches; the clones above only pin them for their duration.
    drop(guards);
    Ok(())
}

// Piece ranges are lists of ranges, and a list of one is the common case.
#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;

    /// Uniform blocks of 4: each piece takes the blocks its positions cover,
    /// a block a piece starts inside goes to it, a block it ends inside does
    /// not, and the gaps are no piece's.
    #[test]
    fn a_block_map_follows_the_positions() {
        let at = |r: &Range<usize>| block_range(r, |pos| (pos + 1) / 4);
        // Piece 0: [0, 20) → blocks 0..5; piece 1: [22, 40) → blocks 5..10
        // (block 5 holds 20..24, where piece 1 starts); piece 2: [60, 64).
        let map = block_map(&[0..20, 22..40, 60..64], at);
        let mut want = vec![NO_PIECE; 16];
        want[..5].fill(0);
        want[5..10].fill(1);
        want[15] = 2;
        assert_eq!(map, want);
        assert!(block_map(&[], at).is_empty());
        assert!(block_map(&[5..5], at).is_empty());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn no_piece_is_the_kernels() {
        assert_eq!(NO_PIECE, candle_kernels::simple::qsa_hits::NO_PIECE);
        assert_eq!(
            COUNTER_TAIL,
            candle_kernels::simple::qsa_hits::COUNTER_TAIL_WORDS
        );
        assert_eq!(
            HEADER_WORDS,
            candle_kernels::simple::qsa_hits::SUMMARY_HEADER_WORDS
        );
    }

    /// A read takes the difference from the last, wrapping as the counters do.
    #[test]
    fn a_read_takes_what_landed_since_the_last() {
        assert_eq!(since(&[5, 9], &[2, 9], 30, 10), (vec![3, 0], 20));
        assert_eq!(since(&[1], &[u32::MAX], 2, u32::MAX), (vec![2], 3));
        assert_eq!(since(&[4, 4], &[], 4, 0), (vec![4, 4], 4));
    }

    /// **The device counts what the entries name, per sequence, per piece,
    /// and publishes it where the host reads it**: two sequences in one
    /// launch, a dense row that counts nothing, a tail entry past the map,
    /// and a view whose rows count into the counters of the parent it was
    /// carved from — checked against the count taken on the host from the
    /// same entries.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_device_counts_each_sequences_selected_blocks() -> Result<()> {
        use crate::models::gpu_test_lock::gpu_serial;
        use crate::models::qwen4exp::qsa_select::{pack_entry, DENSE_ROW};
        let _gpu = gpu_serial();
        let Ok(device) = Device::cuda_if_available(0) else {
            return Ok(());
        };
        if !device.is_cuda() {
            return Ok(());
        }
        let stride = 6usize;
        // Rows 0–1: seq 1. Rows 2–3: seq 2. Row 4: seq 3, a view of seq 1.
        let rows: Vec<Vec<u32>> = vec![
            vec![
                pack_entry(0, 4),
                pack_entry(3, 4),
                pack_entry(5, 2),
                pack_entry(40, 1),
            ],
            vec![pack_entry(3, 4), pack_entry(4, 4)],
            vec![pack_entry(1, 4), pack_entry(2, 1)],
            vec![pack_entry(0, 4)],
        ];
        let mut flat = vec![0u32; rows.len() * stride];
        let mut cnt = Vec::new();
        for (r, e) in rows.iter().enumerate() {
            flat[r * stride..r * stride + e.len()].copy_from_slice(e);
            cnt.push(e.len() as u32);
        }
        // Seq 2's second row is dense: nothing selected, nothing counted.
        let mut rows_cnt = cnt.clone();
        rows_cnt.insert(3, DENSE_ROW);
        flat.splice(3 * stride..3 * stride, vec![0u32; stride]);
        let n_rows = rows_cnt.len();
        let entries = Tensor::from_vec(flat, (n_rows, stride), &device)?;
        let cnt_t = Tensor::from_vec(rows_cnt, n_rows, &device)?;
        let sel = QsaSelection::new(entries, cnt_t, 4)?;
        let spans = [
            SeqSpan {
                seq: 1,
                start: 0,
                len: 2,
            },
            SeqSpan {
                seq: 2,
                start: 2,
                len: 2,
            },
            SeqSpan {
                seq: 3,
                start: 4,
                len: 1,
            },
        ];
        let mut hits = HashMap::new();
        // Seq 1: piece 0 owns blocks 0..3, piece 1 owns 3..6.
        let parent = SeqHits::new(vec![0, 0, 0, 1, 1, 1], 2, &device)?;
        let own = |hits: SeqHits| HitHolder {
            hits,
            owned: true,
            generation: 0,
        };
        hits.insert(1, own(parent.clone()));
        // Seq 2: piece 0 owns block 1 only.
        hits.insert(
            2,
            own(SeqHits::new(vec![NO_PIECE, 0, NO_PIECE], 1, &device)?),
        );
        // Seq 3 is a view carved from seq 1: it holds the same counters.
        hits.insert(
            3,
            HitHolder {
                hits: parent.clone(),
                owned: false,
                generation: parent.generation()?,
            },
        );
        // Nothing published yet: a read finds nothing.
        assert_eq!(hits[&1].hits.take()?.queries, 0);
        let mut table: HitTable = None;
        // Two sampled layers of one wave: the second reuses the table and
        // publishes.
        count_hits(&sel, &spans, &hits, &mut table, false, &device)?;
        let words = table.as_ref().map(|(w, _)| w.clone());
        count_hits(&sel, &spans, &hits, &mut table, true, &device)?;
        assert_eq!(table.as_ref().map(|(w, _)| w.clone()), words);
        device.synchronize()?;
        let one = hits[&1].hits.take()?;
        // Per layer — counted twice here — row 0: 0 → p0, 3 → p1, 5 → p1, 40 →
        // past the map. Row 1: 3, 4 → p1. Row 4, the view's: 0 → p0.
        assert_eq!(one.hits, vec![4, 8]);
        assert_eq!(one.blocks, vec![3, 3]);
        assert_eq!(one.queries, 6);
        assert_eq!(
            hits[&3].hits.take()?.hits,
            vec![0, 0],
            "one read, through either"
        );
        let two = hits[&2].hits.take()?;
        assert_eq!(
            two.hits,
            vec![2],
            "block 1 once a layer; the dense row counts nothing"
        );
        assert_eq!(two.queries, 2, "the dense row is no query");

        // A second wave: a read takes only what it added.
        count_hits(&sel, &spans, &hits, &mut table, true, &device)?;
        device.synchronize()?;
        let again = hits[&1].hits.take()?;
        assert_eq!(again.hits, vec![2, 4]);
        assert_eq!(again.queries, 3);

        // The parent is rebuilt and redeclares: the view, carved under the
        // old layout, stops counting; the parent's own rows go on.
        parent.redeclare(vec![0, 0, 0, 1, 1, 1], 2, &device)?;
        count_hits(&sel, &spans, &hits, &mut table, true, &device)?;
        device.synchronize()?;
        let after = hits[&1].hits.take()?;
        assert_eq!(after.hits, vec![1, 4], "the parent's rows alone");
        assert_eq!(after.queries, 2);
        Ok(())
    }

    /// Off CUDA a read takes the counters themselves, the difference from the
    /// last read; a redeclaration reaches every holder.
    #[test]
    fn a_read_takes_the_counts_since_the_last() -> Result<()> {
        let h = SeqHits::new(vec![0, 0, 1, NO_PIECE], 2, &Device::Cpu)?;
        assert_eq!(
            h.take()?,
            PieceHits {
                hits: vec![0, 0],
                blocks: vec![2, 1],
                queries: 0
            }
        );
        {
            let mut c = h.lock()?;
            c.counters = Tensor::from_vec(vec![3u32, 9, 7, 0], 4, &Device::Cpu)?;
        }
        assert_eq!(
            h.take()?,
            PieceHits {
                hits: vec![3, 9],
                blocks: vec![2, 1],
                queries: 7
            }
        );
        assert_eq!(h.take()?.hits, vec![0, 0]);
        let view = h.clone();
        assert_eq!(h.redeclare(vec![0], 1, &Device::Cpu)?, 1);
        assert_eq!(view.generation()?, 1, "every holder sees the redeclaration");
        assert_eq!(view.take()?.blocks, vec![1]);
        Ok(())
    }
}
