//! Per-round hit counters on the device (`docs/hit_driven_projection.md` §6.4).
//!
//! A projection declares the rounds it placed, as position ranges; each
//! becomes a run of blocks in a block→round map, built once through the
//! sequence's index layout. It declares, after them, the pooled rounds it did
//! not place: their unloaded rows at each sampled layer, which the shadow scan
//! reads ([`super::shadow`]). At every sampled layer, the selection's chosen
//! blocks — and the rows that chose them — are counted into the sequence's
//! counters by one batched launch (`qsa_hits.cu`), which also keeps the
//! stride's last queries for the scan. The first wave after a boundary scores
//! those kept queries against the pooled rows before it counts its own. After
//! the wave's last sampled layer a second launch publishes the counters into a
//! mapped block ([`super::hit_summary`]), which the projection reads at its
//! next boundary without waiting on the device (AGENTS hot-path invariant 3).
//!
//! The counters, the map, the kept queries and the launch table live on the
//! reservation's held ground, never in loose allocations.

#[cfg(feature = "cuda")]
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard};

#[cfg(feature = "cuda")]
use candle::Storage;
use candle::{DType, Device, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_kernels::simple::qsa_hits::{HitSpan, HIT_SPAN_WORDS, KEEP_SLOTS};
#[cfg(feature = "cuda")]
use candle_kernels::simple::qsa_topk::CUT_WORDS;

#[cfg(feature = "cuda")]
use crate::models::delta_net::mix::SeqSpan;
#[cfg(feature = "cuda")]
use crate::models::held_tensor::{held_empty, held_zeros};
use crate::models::hit_layers::PieceHits;
#[cfg(feature = "cuda")]
use crate::models::qsa_selection::QsaSelection;
#[cfg(feature = "cuda")]
use crate::models::rope_schedule::FactoredRope;
#[cfg(feature = "cuda")]
use crate::models::wave_buffers::upload_into;

#[cfg(feature = "cuda")]
use super::hit_summary::{self, MappedBlock, HEADER_WORDS};
#[cfg(feature = "cuda")]
use super::indexer::{tensor_ptr, SelectedRows};
#[cfg(feature = "cuda")]
use super::shadow::{self, ShadowGeometry, ShadowJob, ShadowRows};

/// A block no declared round owns (`QSA_NO_PIECE` in `qsa_hits.cu`).
pub const NO_PIECE: u32 = u32::MAX;

/// Words a counter set holds past its hits: the queries counted, the shadow
/// queries counted, then the publish sequence word (`COUNTER_TAIL_WORDS` in the
/// kernel's binding).
const COUNTER_TAIL: usize = 3;

/// The pooled rounds a declaration names after its loaded ones, in their
/// order: the blocks each holds and — on CUDA — their unloaded rows at each
/// sampled layer, one [`ShadowRows`] per layer, for the shadow scan.
#[derive(Debug, Default)]
pub struct PooledRounds {
    pub blocks: Vec<u32>,
    #[cfg(feature = "cuda")]
    pub layers: Vec<Arc<ShadowRows>>,
}

/// One publish's cumulative counts: its sequence word, the queries, the
/// shadow queries and the hits per round.
type Published = (u32, u32, u32, Vec<u32>);

/// The kept-query buffer's shape: one set of [`KEEP_SLOTS`] slots per sampled
/// layer, each a rotated query of `q_words` and its cut record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeepShape {
    pub layers: usize,
    pub q_words: usize,
}

/// One projection's declared rounds and their counters.
#[derive(Debug)]
struct Counts {
    /// `[map_len]` U32: block → round, [`NO_PIECE`] for none.
    map: Tensor,
    map_len: usize,
    /// `[n_pieces + COUNTER_TAIL]` U32: hits per round, the queries counted,
    /// the shadow queries counted and the publish sequence word, cumulative
    /// over the declaration. Atomic accumulators, so they start at zero; never
    /// reset while the declaration stands — a read takes the difference from
    /// the last.
    counters: Tensor,
    /// Every declared round, loaded and pooled.
    n_pieces: usize,
    /// The loaded rounds, which come first.
    n_loaded: usize,
    /// Blocks each round owns, host-side.
    blocks: Vec<u32>,
    /// The pooled rounds' unloaded rows at each sampled layer.
    #[cfg(feature = "cuda")]
    shadow: Vec<Arc<ShadowRows>>,
    /// Which declaration this is: each redeclaration counts one more.
    generation: u64,
    /// The publish the last read took, and what it held: a read returns
    /// what has landed since.
    read_seq: u32,
    read_hits: Vec<u32>,
    read_queries: u32,
    read_shadow: u32,
}

/// The stride's kept queries: written by every wave's count, scanned by the
/// first wave after a boundary. Outlives a redeclaration — the boundary that
/// asks for the scan is the one that declares the pool it scans.
#[cfg(feature = "cuda")]
#[derive(Debug)]
struct Stride {
    /// `[layers · KEEP_SLOTS · (q_words + CUT_WORDS)]` U32 on held ground.
    /// Uninitialised: a slot is read only once the stride has written it.
    keep: Tensor,
    shape: KeepShape,
    /// Per sampled layer, the slots this stride wrote.
    kept: Vec<u64>,
    /// A boundary read the counters: the next wave scans.
    asked: bool,
    /// The wave under way scans: latched from `asked` at its first sampled
    /// layer, so every layer of one wave scans or none does.
    due: bool,
}

#[cfg(feature = "cuda")]
impl Stride {
    fn new(shape: KeepShape, device: &Device) -> Result<Self> {
        let words = (shape.layers * KEEP_SLOTS * (shape.q_words + CUT_WORDS)).max(1);
        let keep = match device {
            Device::Cuda(_) => held_empty(words, DType::U32, device)?,
            _ => Tensor::zeros(words, DType::U32, device)?,
        };
        Ok(Self {
            keep,
            shape,
            kept: vec![0; shape.layers],
            asked: false,
            due: false,
        })
    }

    /// The device address of `layer`'s slots.
    fn layer_ptr(&self, layer: usize) -> Result<u64> {
        let rec = (self.shape.q_words + CUT_WORDS) as u64;
        Ok(u32_device_ptr(&self.keep)? + (layer * KEEP_SLOTS) as u64 * rec * 4)
    }
}

/// A sequence's counters and what goes with them across declarations.
#[derive(Debug)]
struct Held {
    counts: Counts,
    /// Which holder the summary block belongs to.
    #[cfg(feature = "cuda")]
    token: u32,
    /// Where the device publishes the counters. Off CUDA there is none, and a
    /// read takes the counters themselves.
    #[cfg(feature = "cuda")]
    summary: Option<MappedBlock>,
    #[cfg(feature = "cuda")]
    stride: Stride,
}

/// A sequence's hit counters, shared by every view carved from it.
///
/// A projection rebuilds the PARENT slot and declares its rounds there, while
/// the turn decodes on a view forked from it: the view's rows are the reads
/// that count. So a fork holds the parent's handle, every holder counts into
/// one set of counters, a redeclaration replaces them in place — reaching
/// views carved before it — and a read through any holder sees them all.
#[derive(Debug, Clone)]
pub struct SeqHits(Arc<Mutex<Held>>);

/// One sequence's hold on hit counters: its own, which its projection
/// declared, or the parent's it was carved from, which its rows count into.
#[derive(Debug, Clone)]
pub struct HitHolder {
    pub hits: SeqHits,
    /// Whether this sequence's projection declared these counters. Only an
    /// owner redeclares them in place or reads them; a sequence holding its
    /// parent's that declares rounds of its own gets counters of its own.
    pub owned: bool,
    /// The declaration a view was carved under. A rebuild of the parent
    /// redeclares the rounds over a new layout the view does not read, so a
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
/// pinned ranges and hit rounds both go by.
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

/// The block→round map of `pieces` (position ranges, in declaration order)
/// under a layout whose `to_blocks` turns a position range into the blocks it
/// covers ([`block_range`], the rule pins use too). Where two rounds
/// would share a block the later one owns it. Blocks no round owns are
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

/// The kept slots the last [`KEEP_SLOTS`] positions of a span ending before
/// `end` write, `len` positions long: slot `position mod KEEP_SLOTS`.
pub fn kept_slots(end: usize, len: usize) -> u64 {
    let n = len.min(64);
    (end - n..end).fold(0u64, |m, p| m | (1u64 << (p % 64)))
}

/// The counts in `now` past `then`, both cumulative over one declaration. The
/// counters are u32 and wrap, so the difference wraps too.
fn since(now: &[u32], then: &[u32]) -> Vec<u32> {
    now.iter()
        .enumerate()
        .map(|(i, &n)| n.wrapping_sub(then.get(i).copied().unwrap_or(0)))
        .collect()
}

impl SeqHits {
    /// Counters for the loaded rounds `map` names and the `pooled` rounds
    /// after them, with a kept-query buffer of `keep`'s shape.
    pub fn new(
        map: Vec<u32>,
        n_loaded: usize,
        pooled: PooledRounds,
        keep: KeepShape,
        device: &Device,
    ) -> Result<Self> {
        let counts = Counts::new(map, n_loaded, pooled, device, 0)?;
        #[cfg(feature = "cuda")]
        let held = {
            let mut h = Held {
                counts,
                token: hit_summary::next_token(),
                summary: None,
                stride: Stride::new(keep, device)?,
            };
            h.adopt(None, device)?;
            h
        };
        #[cfg(not(feature = "cuda"))]
        let held = {
            let _ = keep;
            Held { counts }
        };
        Ok(Self(Arc::new(Mutex::new(held))))
    }

    /// Replace the declared rounds and their counters, for every holder. The
    /// counts since the last read are dropped: a caller reads them first. The
    /// summary block and the stride's kept queries carry over. Returns the new
    /// declaration's generation.
    pub fn redeclare(
        &self,
        map: Vec<u32>,
        n_loaded: usize,
        pooled: PooledRounds,
        device: &Device,
    ) -> Result<u64> {
        let mut h = self.lock()?;
        let generation = h.counts.generation + 1;
        h.counts = Counts::new(map, n_loaded, pooled, device, generation)?;
        #[cfg(feature = "cuda")]
        {
            let carried = h.summary.take();
            h.adopt(carried, device)?;
        }
        Ok(generation)
    }

    /// The current declaration's generation.
    pub fn generation(&self) -> Result<u64> {
        Ok(self.lock()?.counts.generation)
    }

    /// What has been counted since the last read, as far as the device has
    /// published it, and ask the next wave to scan the stride's kept queries.
    ///
    /// Reads the mapped summary the device last finished writing: no copy,
    /// no wait. Counts still in flight are in the next read.
    pub fn take(&self) -> Result<PieceHits> {
        let mut h = self.lock()?;
        #[cfg(feature = "cuda")]
        {
            h.stride.asked = true;
        }
        h.take()
    }

    fn lock(&self) -> Result<MutexGuard<'_, Held>> {
        self.0
            .lock()
            .map_err(|_| candle::Error::Msg("qsa hits: counter lock poisoned".into()))
    }
}

impl Counts {
    fn new(
        map: Vec<u32>,
        n_loaded: usize,
        pooled: PooledRounds,
        device: &Device,
        generation: u64,
    ) -> Result<Self> {
        let mut blocks = vec![0u32; n_loaded];
        for &p in &map {
            if let Some(b) = blocks.get_mut(p as usize) {
                *b += 1;
            }
        }
        blocks.extend_from_slice(&pooled.blocks);
        let n_pieces = blocks.len();
        let map_len = map.len();
        // An empty map still holds one slot, so the tensor is never zero-sized;
        // `map_len` says none of it is read.
        let map = match map_len {
            0 => vec![NO_PIECE],
            _ => map,
        };
        let (map, counters) = Self::buffers(map, n_pieces + COUNTER_TAIL, device)?;
        Ok(Self {
            map,
            map_len,
            counters,
            n_pieces,
            n_loaded,
            blocks,
            #[cfg(feature = "cuda")]
            shadow: pooled.layers,
            generation,
            read_seq: 0,
            read_hits: vec![0; n_pieces],
            read_queries: 0,
            read_shadow: 0,
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
}

impl Held {
    /// Take `carried` — the previous declaration's summary block — when it
    /// holds this one's rounds, and a block that does otherwise. Off CUDA
    /// there is no block.
    #[cfg(feature = "cuda")]
    fn adopt(&mut self, carried: Option<MappedBlock>, device: &Device) -> Result<()> {
        let Device::Cuda(dev) = device else {
            if let Some(b) = carried {
                hit_summary::release(b);
            }
            return Ok(());
        };
        let words = HEADER_WORDS + self.counts.n_pieces;
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

    /// The newest cumulative counts — sequence word, queries, shadow queries,
    /// hits: the summary's last finished publish on CUDA, the counters
    /// themselves elsewhere. `None` while nothing new has landed since the
    /// last read.
    fn newest(&self) -> Result<Option<Published>> {
        let c = &self.counts;
        #[cfg(feature = "cuda")]
        if let Some(block) = &self.summary {
            return Ok(block
                .read(self.token, c.generation as u32, c.n_pieces)
                .filter(|s| s.seq != c.read_seq)
                .map(|s| (s.seq, s.queries, s.shadow_queries, s.hits)));
        }
        let all = c.counters.to_vec1::<u32>()?;
        Ok(Some((
            0,
            all[c.n_pieces],
            all[c.n_pieces + 1],
            all[..c.n_pieces].to_vec(),
        )))
    }

    fn take(&mut self) -> Result<PieceHits> {
        let Some((seq, queries, shadow, now)) = self.newest()? else {
            let c = &self.counts;
            return Ok(PieceHits {
                hits: vec![0; c.n_pieces],
                blocks: c.blocks.clone(),
                loaded: c.n_loaded,
                queries: 0,
                shadow_queries: 0,
            });
        };
        let c = &mut self.counts;
        let hits = since(&now, &c.read_hits);
        let q = queries.wrapping_sub(c.read_queries) as u64;
        let sq = shadow.wrapping_sub(c.read_shadow) as u64;
        c.read_seq = seq;
        c.read_hits = now;
        c.read_queries = queries;
        c.read_shadow = shadow;
        Ok(PieceHits {
            hits,
            blocks: c.blocks.clone(),
            loaded: c.n_loaded,
            queries: q,
            shadow_queries: sq,
        })
    }
}

#[cfg(feature = "cuda")]
impl Drop for Held {
    fn drop(&mut self) {
        if let Some(b) = self.summary.take() {
            hit_summary::release(b);
        }
    }
}

/// A U32 tensor's device address.
#[cfg(feature = "cuda")]
fn u32_device_ptr(t: &Tensor) -> Result<u64> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    let Device::Cuda(dev) = t.device() else {
        candle::bail!("qsa hits: buffers must be CUDA");
    };
    let stream = dev.cuda_stream();
    let (s, l) = t.storage_and_layout();
    let slice = match &*s {
        Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("qsa hits: buffers must be CUDA u32"),
    }
    .slice(l.start_offset()..);
    let (p, _g) = slice.device_ptr(&stream);
    Ok(p)
}

/// What the scan needs of the engine at a sampled layer: the kernel's
/// geometry and the RoPE each sequence's rung is read from.
#[cfg(feature = "cuda")]
pub struct ShadowScan<'a> {
    pub geom: ShadowGeometry,
    pub rope: &'a FactoredRope,
}

/// Count the hits of `sel` — sampled layer `layer`'s selection over a wave
/// whose rows `spans` describe, `offsets[i]` the first position of `spans[i]`
/// — into each counted sequence's counters, keeping the last rows' queries
/// (`rows`) for the shadow scan, in one launch. A sequence a boundary asked
/// has its kept queries scanned against its pooled rows first, in one launch
/// over every such sequence. When `publish` (the wave's last sampled layer)
/// every counted set is published into its summary block, in a third.
///
/// A sequence with no declared rounds is not counted, nor a view carved under
/// an older declaration. A dense row counts no hit and no query. The span
/// table is `table`'s when its words match and is uploaded otherwise.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn count_hits(
    sel: &QsaSelection,
    rows: &SelectedRows,
    spans: &[SeqSpan],
    offsets: &[usize],
    hits: &HashMap<usize, HitHolder>,
    table: &mut HitTable,
    layer: usize,
    publish: bool,
    scan: &ShadowScan<'_>,
    device: &Device,
) -> Result<()> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_kernels::simple::qsa_hits::{run_qsa_count_hits, run_qsa_publish_hits};
    use std::ffi::c_void;

    let Device::Cuda(dev) = device else {
        candle::bail!("qsa hits are counted on CUDA");
    };
    let stream = dev.cuda_stream();
    let q_words = rows.q.elem_count() / sel.rows().max(1);
    // Each counted span as the kernel's `HitSpan`, in the words its layout
    // test pins (`HitSpan::words`); each counter set's shadow job when a
    // boundary asked for one; and every counted span's kept slots.
    let mut words: Vec<u32> = Vec::new();
    let mut guards = Vec::new();
    let mut sets: Vec<u64> = Vec::new();
    let mut jobs: Vec<(u64, u64, Arc<ShadowRows>, u64, u32, Tensor)> = Vec::new();
    let mut slots: Vec<(usize, u64)> = Vec::new();
    let mut max_rows = 0usize;
    for (i, span) in spans.iter().enumerate() {
        let Some(holder) = hits.get(&span.seq) else {
            continue;
        };
        let mut h = holder.hits.lock()?;
        if h.counts.n_pieces == 0 {
            continue;
        }
        if !holder.owned && holder.generation != h.counts.generation {
            continue;
        }
        if h.stride.shape.q_words != q_words || layer >= h.stride.shape.layers {
            candle::bail!(
                "qsa hits: a kept-query buffer of {:?} against layer {layer} of {q_words}-word \
                 queries",
                h.stride.shape
            );
        }
        let Some(summary) = h.summary.as_ref() else {
            continue;
        };
        let summary_ptr = summary.dev_ptr();
        let map_ptr = u32_device_ptr(&h.counts.map)?;
        let cnt_ptr = u32_device_ptr(&h.counts.counters)?;
        let keep_ptr = u32_device_ptr(&h.stride.keep)?;
        // A view counts into its parent's counters: the set is scanned and
        // published once.
        let first = !sets.contains(&cnt_ptr);
        if first {
            sets.push(cnt_ptr);
            if layer == 0 {
                h.stride.due = h.stride.asked;
                h.stride.asked = false;
            }
            let mask = h.stride.kept[layer];
            if h.stride.due {
                if let Some(rows) = h.counts.shadow.get(layer).filter(|_| mask != 0) {
                    let rung = scan.rope.rung_for(offsets[i] + span.len)?;
                    let tab = scan.rope.table(rung)?;
                    jobs.push((
                        h.stride.layer_ptr(layer)?,
                        mask,
                        Arc::clone(rows),
                        cnt_ptr,
                        h.counts.n_pieces as u32,
                        tab,
                    ));
                }
                h.stride.kept[layer] = 0;
            }
        }
        slots.push((i, cnt_ptr));
        let entry = HitSpan {
            row_lo: span.start as u32,
            row_hi: (span.start + span.len) as u32,
            map_len: h.counts.map_len as u32,
            n_pieces: h.counts.n_pieces as u32,
            map: map_ptr as *const u32,
            counters: cnt_ptr as *mut u32,
            summary: summary_ptr as *mut u32,
            keep: keep_ptr as *mut u32,
            token: h.token,
            generation: h.counts.generation as u32,
            publish: first as u32,
            pad: 0,
        };
        words.extend_from_slice(&entry.words());
        guards.push((
            h.counts.map.clone(),
            h.counts.counters.clone(),
            h.stride.keep.clone(),
        ));
        max_rows = max_rows.max(span.len);
    }
    let n_spans = words.len() / HIT_SPAN_WORDS;
    if n_spans == 0 {
        return Ok(());
    }

    // The scan reads the stride the count below is about to overwrite, so it
    // goes first, on the same stream.
    if !jobs.is_empty() {
        let tabs: Vec<u64> = jobs
            .iter()
            .map(|j| tensor_ptr(&j.5))
            .collect::<Result<_>>()?;
        let shadow_jobs: Vec<ShadowJob<'_>> = jobs
            .iter()
            .zip(&tabs)
            .map(
                |((keep, mask, rows, counters, n_pieces, _), &tab)| ShadowJob {
                    keep: *keep,
                    mask: *mask,
                    rows: rows.as_ref(),
                    counters: *counters,
                    n_pieces: *n_pieces,
                    tab,
                },
            )
            .collect();
        shadow::launch(&shadow_jobs, &scan.geom, device)?;
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
    let t_ptr = u32_device_ptr(table)?;
    let (e_s, e_l) = sel.entries().storage_and_layout();
    let e_slice = match &*e_s {
        Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("qsa hits: entries must be CUDA"),
    }
    .slice(e_l.start_offset()..);
    let (e_ptr, _eg) = e_slice.device_ptr(&stream);
    let c_ptr = u32_device_ptr(sel.cnt())?;
    let q_ptr = tensor_ptr(&rows.q)?;
    let u_ptr = u32_device_ptr(&rows.cuts)?;
    candle::set_kernel_breadcrumb("run_qsa_count_hits", file!(), line!());
    unsafe {
        run_qsa_count_hits(
            e_ptr as *const u32,
            sel.stride() as i32,
            c_ptr as *const u32,
            q_ptr as *const f32,
            q_words as i32,
            u_ptr as *const u32,
            layer as i32,
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

    // What the count kept, for the next scan; and a wave's last sampled layer
    // ends its scan.
    for (i, cnt_ptr) in slots {
        let span = &spans[i];
        let Some(holder) = hits.get(&span.seq) else {
            continue;
        };
        let mut h = holder.hits.lock()?;
        if u32_device_ptr(&h.counts.counters)? != cnt_ptr {
            continue;
        }
        h.stride.kept[layer] |= kept_slots(offsets[i] + span.len, span.len);
        if publish {
            h.stride.due = false;
        }
    }
    // The buffers stay owned by their `SeqHits` across the launches; the
    // clones above only pin them for their duration.
    drop(guards);
    drop(jobs);
    Ok(())
}

// Round ranges are lists of ranges, and a list of one is the common case.
#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;

    /// Uniform blocks of 4: each round takes the blocks its positions cover,
    /// a block a round starts inside goes to it, a block it ends inside does
    /// not, and the gaps are no round's.
    #[test]
    fn a_block_map_follows_the_positions() {
        let at = |r: &Range<usize>| block_range(r, |pos| (pos + 1) / 4);
        // Round 0: [0, 20) → blocks 0..5; round 1: [22, 40) → blocks 5..10
        // (block 5 holds 20..24, where round 1 starts); round 2: [60, 64).
        let map = block_map(&[0..20, 22..40, 60..64], at);
        let mut want = vec![NO_PIECE; 16];
        want[..5].fill(0);
        want[5..10].fill(1);
        want[15] = 2;
        assert_eq!(map, want);
        assert!(block_map(&[], at).is_empty());
        assert!(block_map(&[5..5], at).is_empty());
    }

    /// The last 64 positions of a span write their slots, wrapping; a short
    /// span writes only its own.
    #[test]
    fn a_span_keeps_its_last_positions_slots() {
        assert_eq!(kept_slots(5, 2), 0b11000);
        assert_eq!(kept_slots(66, 3), (1 << 63) | 0b11);
        assert_eq!(kept_slots(1000, 900), u64::MAX);
        assert_eq!(kept_slots(1000, 64), u64::MAX);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_constants_are_the_kernels() {
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
        assert_eq!(since(&[5, 9, 30], &[2, 9, 10]), vec![3, 0, 20]);
        assert_eq!(since(&[1], &[u32::MAX]), vec![2]);
        assert_eq!(since(&[4, 4], &[]), vec![4, 4]);
    }

    fn keep_shape() -> KeepShape {
        KeepShape {
            layers: 2,
            q_words: 4,
        }
    }

    /// **The device counts what the entries name, per sequence, per round,
    /// keeps the last rows' queries, and publishes it where the host reads
    /// it**: two sequences in one launch, a dense row that counts nothing, a
    /// tail entry past the map, and a view whose rows count into the counters
    /// of the parent it was carved from — checked against the count taken on
    /// the host from the same entries.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_device_counts_each_sequences_selected_blocks() -> Result<()> {
        use crate::models::gpu_test_lock::gpu_serial;
        use crate::models::qwen4exp::qsa_select::{pack_entry, DENSE_ROW};
        use crate::models::rope_schedule::plain_inv_freq;
        use candle_kernels::simple::qsa_topk::NO_CUT;
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
        let offsets = [70usize, 9, 72];
        // Each row's query is its row number four times; its cut record its
        // position.
        let q: Vec<f32> = (0..n_rows).flat_map(|r| [r as f32; 4]).collect();
        let positions = [70u32, 71, 9, 10, 72];
        let cuts: Vec<u32> = positions
            .iter()
            .enumerate()
            .flat_map(|(r, &p)| {
                let cut = if r == 3 { NO_CUT } else { 1.5f32.to_bits() };
                [cut, p, 20, 1]
            })
            .collect();
        let selected = SelectedRows {
            q: Tensor::from_vec(q, (n_rows, 1, 4), &device)?,
            cuts: Tensor::from_vec(cuts, (n_rows, CUT_WORDS), &device)?,
        };
        let rope = FactoredRope::new(&plain_inv_freq(4, 1e4), &device)?;
        let scan = ShadowScan {
            geom: ShadowGeometry {
                heads: 1,
                head_dim: 4,
                pairs: 2,
                ratio: 4,
                top_k: 8,
            },
            rope: &rope,
        };
        let none = || PooledRounds::default();
        let mut hits = HashMap::new();
        // Seq 1: round 0 owns blocks 0..3, round 1 owns 3..6.
        let parent = SeqHits::new(vec![0, 0, 0, 1, 1, 1], 2, none(), keep_shape(), &device)?;
        let own = |hits: SeqHits| HitHolder {
            hits,
            owned: true,
            generation: 0,
        };
        hits.insert(1, own(parent.clone()));
        // Seq 2: round 0 owns block 1 only.
        let two_hits = SeqHits::new(
            vec![NO_PIECE, 0, NO_PIECE],
            1,
            none(),
            keep_shape(),
            &device,
        )?;
        hits.insert(2, own(two_hits.clone()));
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
        count_hits(
            &sel, &selected, &spans, &offsets, &hits, &mut table, 0, false, &scan, &device,
        )?;
        let words = table.as_ref().map(|(w, _)| w.clone());
        count_hits(
            &sel, &selected, &spans, &offsets, &hits, &mut table, 1, true, &scan, &device,
        )?;
        assert_eq!(table.as_ref().map(|(w, _)| w.clone()), words);
        device.synchronize()?;
        let one = hits[&1].hits.take()?;
        // Per layer — counted twice here — row 0: 0 → r0, 3 → r1, 5 → r1, 40 →
        // past the map. Row 1: 3, 4 → r1. Row 4, the view's: 0 → r0.
        assert_eq!(one.hits, vec![4, 8]);
        assert_eq!(one.blocks, vec![3, 3]);
        assert_eq!((one.loaded, one.queries, one.shadow_queries), (2, 6, 0));
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

        // The kept queries: seq 1's rows (positions 70, 71) and its view's (72)
        // in one buffer, at both layers; seq 2's two rows, the dense one with
        // no cut.
        {
            let h = parent.lock()?;
            let keep = h.stride.keep.to_vec1::<u32>()?;
            let rec = 4 + CUT_WORDS;
            for layer in 0..2 {
                for (row, slot) in [(0usize, 70 % 64), (1, 71 % 64), (4, 72 % 64)] {
                    let at = (layer * KEEP_SLOTS + slot) * rec;
                    assert_eq!(keep[at..at + 4], [(row as f32).to_bits(); 4]);
                    assert_eq!(
                        keep[at + 4..at + 8],
                        [1.5f32.to_bits(), positions[row], 20, 1]
                    );
                }
                assert_eq!(h.stride.kept[layer], kept_slots(72, 2) | kept_slots(73, 1));
            }
        }
        {
            let h = two_hits.lock()?;
            let keep = h.stride.keep.to_vec1::<u32>()?;
            let at = 10 * (4 + CUT_WORDS);
            assert_eq!(keep[at + 4], NO_CUT, "a dense row is kept with no cut");
        }

        // A second wave: a read takes only what it added. The reads above
        // asked for a scan; with no pooled rounds it scans nothing, and the
        // stride starts over.
        count_hits(
            &sel, &selected, &spans, &offsets, &hits, &mut table, 0, false, &scan, &device,
        )?;
        count_hits(
            &sel, &selected, &spans, &offsets, &hits, &mut table, 1, true, &scan, &device,
        )?;
        device.synchronize()?;
        let again = hits[&1].hits.take()?;
        assert_eq!(again.hits, vec![4, 8]);
        assert_eq!((again.queries, again.shadow_queries), (6, 0));
        assert_eq!(
            parent.lock()?.stride.kept[0],
            kept_slots(72, 2) | kept_slots(73, 1),
            "the scanned stride starts over with this wave's"
        );

        // The parent is rebuilt and redeclares: the view, carved under the
        // old layout, stops counting; the parent's own rows go on.
        parent.redeclare(vec![0, 0, 0, 1, 1, 1], 2, none(), &device)?;
        count_hits(
            &sel, &selected, &spans, &offsets, &hits, &mut table, 1, true, &scan, &device,
        )?;
        device.synchronize()?;
        let after = hits[&1].hits.take()?;
        assert_eq!(after.hits, vec![1, 4], "the parent's rows alone");
        assert_eq!(after.queries, 2);
        Ok(())
    }

    /// Off CUDA a read takes the counters themselves, the difference from the
    /// last read; a redeclaration reaches every holder; pooled rounds follow
    /// the loaded ones, with their own blocks and their own denominator.
    #[test]
    fn a_read_takes_the_counts_since_the_last() -> Result<()> {
        let pooled = PooledRounds {
            blocks: vec![6],
            ..Default::default()
        };
        let h = SeqHits::new(
            vec![0, 0, 1, NO_PIECE],
            2,
            pooled,
            keep_shape(),
            &Device::Cpu,
        )?;
        assert_eq!(
            h.take()?,
            PieceHits {
                hits: vec![0, 0, 0],
                blocks: vec![2, 1, 6],
                loaded: 2,
                queries: 0,
                shadow_queries: 0,
            }
        );
        {
            let mut c = h.lock()?;
            c.counts.counters = Tensor::from_vec(vec![3u32, 9, 2, 7, 4, 0], 6, &Device::Cpu)?;
        }
        let got = h.take()?;
        assert_eq!(
            got,
            PieceHits {
                hits: vec![3, 9, 2],
                blocks: vec![2, 1, 6],
                loaded: 2,
                queries: 7,
                shadow_queries: 4,
            }
        );
        assert_eq!((got.queries_of(1), got.queries_of(2)), (7, 4));
        assert_eq!(h.take()?.hits, vec![0, 0, 0]);
        let view = h.clone();
        assert_eq!(
            h.redeclare(vec![0], 1, PooledRounds::default(), &Device::Cpu)?,
            1
        );
        assert_eq!(view.generation()?, 1, "every holder sees the redeclaration");
        assert_eq!(view.take()?.blocks, vec![1]);
        Ok(())
    }
}
