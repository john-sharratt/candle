//! Per-piece hit counters on the device (`docs/hit_driven_projection.md` §6.4).
//!
//! A projection declares the pieces it placed, as position ranges; each
//! becomes a run of blocks in a block→piece map, built once through the
//! sequence's index layout. At every sampled layer, the selection's chosen
//! blocks are counted into their pieces' counters by one batched launch
//! (`qsa_hits.cu`), and the counters are read — and zeroed — when the
//! projection next asks.

#[cfg(feature = "cuda")]
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard};

#[cfg(feature = "cuda")]
use candle::Storage;
use candle::{DType, Device, Result, Tensor};

#[cfg(feature = "cuda")]
use crate::models::delta_net::mix::SeqSpan;
use crate::models::hit_layers::PieceHits;
#[cfg(feature = "cuda")]
use crate::models::qsa_selection::QsaSelection;

/// A block no placed piece owns (`QSA_NO_PIECE` in `qsa_hits.cu`).
pub const NO_PIECE: u32 = u32::MAX;

/// One projection's declared pieces and their counters.
#[derive(Debug)]
struct Counts {
    /// `[map_len]` U32 on the device: block → piece, [`NO_PIECE`] for none.
    map: Tensor,
    map_len: usize,
    /// `[n_pieces]` U32 on the device: hits since the last read. An atomic
    /// accumulator, so it starts — and is reset — at zero.
    counters: Tensor,
    n_pieces: usize,
    /// Blocks each piece owns in the map, host-side.
    blocks: Vec<u32>,
    /// Query rows counted since the last read, summed over sampled layers.
    queries: u64,
    /// Which declaration this is: each redeclaration counts one more.
    generation: u64,
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
/// so the table goes up once per wave and is reused while its words match.
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

impl SeqHits {
    /// Counters for `n_pieces` pieces over `map`.
    pub fn new(map: Vec<u32>, n_pieces: usize, device: &Device) -> Result<Self> {
        Ok(Self(Arc::new(Mutex::new(Counts::new(
            map, n_pieces, device,
        )?))))
    }

    /// Replace the declared pieces and their counters, for every holder. The
    /// counts since the last read are dropped: a caller reads them first.
    /// Returns the new declaration's generation.
    pub fn redeclare(&self, map: Vec<u32>, n_pieces: usize, device: &Device) -> Result<u64> {
        let mut c = self.lock()?;
        let generation = c.generation + 1;
        *c = Counts::new(map, n_pieces, device)?;
        c.generation = generation;
        Ok(generation)
    }

    /// The current declaration's generation.
    pub fn generation(&self) -> Result<u64> {
        Ok(self.lock()?.generation)
    }

    /// The counts since the last read, and zero them.
    ///
    /// A device→host copy, made where the projection reads it — at a
    /// boundary, between forwards — never inside one.
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
    fn new(map: Vec<u32>, n_pieces: usize, device: &Device) -> Result<Self> {
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
            0 => Tensor::from_vec(vec![NO_PIECE], 1, device)?,
            n => Tensor::from_vec(map, n, device)?,
        };
        let counters = Tensor::zeros(n_pieces.max(1), DType::U32, device)?;
        Ok(Self {
            map,
            map_len,
            counters,
            n_pieces,
            blocks,
            queries: 0,
            generation: 0,
        })
    }

    fn take(&mut self) -> Result<PieceHits> {
        // Nothing counted since the last read: the counters are still zero,
        // so there is nothing to copy back and nothing to reset — and the
        // cached span table, which names them, stays valid.
        if self.queries == 0 {
            return Ok(PieceHits {
                hits: vec![0; self.n_pieces],
                blocks: self.blocks.clone(),
                queries: 0,
            });
        }
        let all = self.counters.to_vec1::<u32>()?;
        let hits = all[..self.n_pieces].to_vec();
        self.counters = self.counters.zeros_like()?;
        let queries = std::mem::take(&mut self.queries);
        Ok(PieceHits {
            hits,
            blocks: self.blocks.clone(),
            queries,
        })
    }
}

/// Count the hits of `sel` — one sampled layer's selection over a wave whose
/// rows `spans` describe, `selecting[i]` of span `i`'s rows selecting — into
/// each counted sequence's counters, in one launch. A sequence with no
/// declared pieces is not counted, nor a view carved under an older
/// declaration. A dense row counts no hit and no query. The span table is
/// `table`'s when its words match — every sampled layer of one wave — and is
/// uploaded once otherwise.
#[cfg(feature = "cuda")]
pub fn count_hits(
    sel: &QsaSelection,
    spans: &[SeqSpan],
    selecting: &[usize],
    hits: &HashMap<usize, HitHolder>,
    table: &mut HitTable,
    device: &Device,
) -> Result<()> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_kernels::simple::qsa_hits::{run_qsa_count_hits, HitSpan};
    use std::ffi::c_void;

    let Device::Cuda(dev) = device else {
        candle::bail!("qsa hits are counted on CUDA");
    };
    let stream = dev.cuda_stream();
    // Each counted span as the kernel's `HitSpan`, in the words its layout
    // test pins (`HitSpan::words`).
    let mut words: Vec<u32> = Vec::new();
    let mut guards = Vec::new();
    let mut max_rows = 0usize;
    for (i, span) in spans.iter().enumerate() {
        let Some(holder) = hits.get(&span.seq) else {
            continue;
        };
        let mut h = holder.hits.lock()?;
        if h.n_pieces == 0 || h.map_len == 0 {
            continue;
        }
        if !holder.owned && holder.generation != h.generation {
            continue;
        }
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
        let entry = HitSpan {
            row_lo: span.start as u32,
            row_hi: (span.start + span.len) as u32,
            map_len: h.map_len as u32,
            n_pieces: h.n_pieces as u32,
            map: map_ptr as *const u32,
            counters: cnt_ptr as *mut u32,
        };
        words.extend_from_slice(&entry.words());
        guards.push((h.map.clone(), h.counters.clone()));
        h.queries += selecting.get(i).copied().unwrap_or(span.len) as u64;
        max_rows = max_rows.max(span.len);
    }
    let n_spans = words.len() / 8;
    if n_spans == 0 {
        return Ok(());
    }
    // Words name the counters' addresses, so a table built for other
    // counters — a redeclaration, a read that zeroed them — never matches.
    let fresh = match table {
        Some((cached, _)) => *cached != words,
        None => true,
    };
    if fresh {
        let t = Tensor::from_vec(words.clone(), (n_spans, 8), device)?;
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
    // The map and counter buffers stay owned by their `SeqHits` across the
    // launch; the clones above only pin them for its duration.
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
    }

    /// **The device counts what the entries name, per sequence, per piece**:
    /// two sequences in one launch, a dense row that counts nothing, a tail
    /// entry past the map, and a view whose rows count into the counters of
    /// the parent it was carved from — checked against the count taken on the
    /// host from the same entries.
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
        // Seq 2's second row is dense, so one of its two rows selects.
        let selecting = [2usize, 1, 1];
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
        let mut table: HitTable = None;
        count_hits(&sel, &spans, &selecting, &hits, &mut table, &device)?;
        let words = table.as_ref().map(|(w, _)| w.clone());
        // A second sampled layer of the same wave reuses the table.
        count_hits(&sel, &spans, &selecting, &hits, &mut table, &device)?;
        assert_eq!(table.as_ref().map(|(w, _)| w.clone()), words);
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

        // The parent is rebuilt and redeclares: the view, carved under the
        // old layout, stops counting; the parent's own rows go on.
        parent.redeclare(vec![0, 0, 0, 1, 1, 1], 2, &device)?;
        count_hits(&sel, &spans, &selecting, &hits, &mut table, &device)?;
        let after = hits[&1].hits.take()?;
        assert_eq!(after.hits, vec![1, 4], "the parent's rows alone");
        assert_eq!(after.queries, 2);
        Ok(())
    }

    /// The counters start at zero and a read zeroes them; a redeclaration
    /// reaches every holder.
    #[test]
    fn a_read_takes_the_counts_and_zeroes_them() -> Result<()> {
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
            c.counters = Tensor::from_vec(vec![3u32, 9], 2, &Device::Cpu)?;
            c.queries = 7;
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
