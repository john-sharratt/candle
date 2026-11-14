//! One gap on the device (`docs/kv_quantize_in_the_gap.md` §3.2–§3.4): the
//! format selection, the claim and the convert, launched back to back on the
//! gap's own stream — behind the forward that completed the blocks, beside the
//! next — with nothing read back and nothing waited for.
//!
//! Per policy group the unchanged selection kernel leaves its tags, outer
//! scales and per-channel palette slots in a held block, and the claim reads
//! them there: it gives every band a slot from the lease, defers a job whose
//! class ran dry, and writes each head's destination record into the convert
//! buffer and the mapped summary. One convert then runs over the whole table,
//! and the completion word goes last. The host learns the outcome from the
//! summary once the word is in (§3.5).
//!
//! The device keeps, across gaps: its stream, the summary, the claim's
//! look-back status words and their epoch, the sink slot a deferred job's bands
//! are converted into, and held scratch for the selection blocks, the convert
//! buffer and the inputs, sized once for the widest table. Every buffer is
//! reused only in the gap stream's order behind the gap that last read it.

use std::ffi::c_void;
use std::ops::Range;
use std::sync::Arc;

use candle::cuda_backend::cudarc::driver::result::memset_d8_async;
use candle::cuda_backend::cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
use candle::cuda_backend::cudarc::driver::{CudaEvent, CudaStream, DevicePtr};
use candle::cuda_backend::kernels::simple::kv_gap_claim::{
    claim_status_words, run_kv_gap_claim, run_kv_gap_complete, MAX_CLASS, MAX_KV_HEAD, NO_OVERRIDE,
};
use candle::cuda_backend::staged::{segment, SEGMENT_ALIGN};
use candle::cuda_backend::{CudaDevice, Held, WrapErr};
use candle::quantized::cuda::{
    check_token_window, ggml_to_select_qtype, identity_pal_map, kvhead_supported_head_dim,
    palette4_convert_launch, select_palette4_inputs, select_palette4_launch_staged,
    write_kvhead_bytes, Palette4ConvertLayout, Palette4Inputs, Palette4Layout,
};
use candle::quantized::pinned_staging::PinnedBuf;
use candle::quantized::GgmlDType;
use candle::Result;

use super::super::backing::ChunkedKvBacking;
use super::super::band_addrs::BandAddrs;
use super::super::compress::{head_source, policy_selection};
use super::super::compression_policy::CompressionPolicy;
use super::super::head_gids::{ChunkBands, GIDS_PER_HEAD};
use super::super::size_class::{elems_per_chunk, SizeClass, GID_STRIDE};
use super::codes::CodeTables;
use super::lease::StackImage;
use super::summary::{completion_word, Finished, MappedSummary, SummaryLayout, BANDS_PER_HEAD};
use crate::kv_cache::arena_table::{PerHeadEntry, N_PALETTE};
use crate::kv_cache::{KvFormat, QuantFormat};

/// The most jobs one gap's table holds: the convert's single launch
/// (`gridDim.y`).
pub(crate) const MAX_GAP_JOBS: usize = 65_535;

/// Device blocks start on this boundary — what the selection's vectorised
/// loads assume of a base pointer.
const BLOCK_ALIGN: usize = 256;

/// One block of one sequence-layer: the native chunk the gap quantizes.
#[derive(Debug, Clone)]
pub(crate) struct GapJob {
    /// Index into the backings the gap is launched over.
    pub layer: usize,
    /// The sequence's batch index in that layer's block table.
    pub seq: usize,
    /// The block's index in the sequence's chunk list.
    pub block: usize,
    /// The chunk's native bands: their gids and format tags. Held by the job
    /// for the gap's life, it is what keeps those bands from going back to
    /// their pool while the convert may still read them (§3.8 rule 4).
    pub bands: ChunkBands,
    /// The chunk's live token window, `(offset << 8) | len`.
    pub valid_range: i32,
}

/// A run of the table's jobs quantized under one policy: one selection launch
/// and one claim launch.
#[derive(Debug, Clone)]
pub(crate) struct GapGroup<'p> {
    pub policy: &'p CompressionPolicy,
    pub jobs: Range<usize>,
}

/// A held buffer that only grows: kept while it is wide enough, replaced by a
/// wider one when it is not. The old one is released in stream order, behind
/// every launch already queued on the gap's stream that reads it.
struct Grown(Option<Held<u8>>);

impl Grown {
    #[track_caller]
    fn at_least(&mut self, stream: &Arc<CudaStream>, bytes: usize) -> Result<u64> {
        let fits = self
            .0
            .as_mut()
            .is_some_and(|h| h.len() >= bytes && !h.asked_to_move());
        if !fits {
            self.0 = None;
            // SAFETY: every byte a launch reads is written ahead of it on the
            // stream — the uploads, the selection's own clear, or the claim.
            self.0 = Some(unsafe { Held::alloc_on(stream, bytes.max(BLOCK_ALIGN))? });
        }
        let held = self.0.as_ref().expect("allocated above");
        Ok(held.device_ptr(stream).0)
    }
}

/// The host image's segments from here on are the launch inputs, copied as
/// one block; the two before them land in the convert buffer.
const INPUTS_FROM: usize = 2;

/// The most candidates one side of a level's selection names — what the
/// scratch is sized for.
const MAX_CANDIDATES: usize = 16;

/// The policy groups the scratch is sized for in one table: each group's
/// selection block carries its own candidate sections and alignment.
const SCRATCH_GROUPS: usize = 64;

/// Sections in a selection block ([`Palette4Layout`]), each aligned apart.
const SELECT_SECTIONS: usize = 21;

/// What a table of `jobs` jobs takes of the gap's device scratch, at most:
/// the selection blocks, the convert buffer, and the inputs the claim and the
/// selections read. Allocated once, at this size, when the gap's state is
/// made: a buffer grown between forwards waits for the forward ahead of it,
/// because the held ground it comes from reuses ground behind that forward's
/// fence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GapScratch {
    pub select: usize,
    pub convert: usize,
    pub inputs: usize,
}

impl GapScratch {
    pub(crate) fn for_jobs(jobs: usize, n_kv_head: usize, head_dim: usize) -> Self {
        let heads = jobs * n_kv_head;
        let block = |n: usize| {
            Palette4Layout::new(n, n_kv_head, head_dim, MAX_CANDIDATES, MAX_CANDIDATES).bytes()
        };
        // A group's own fixed sections, and the alignment it may add.
        let per_group = block(0) + SELECT_SECTIONS * 4 * SECTION_PAD_WORDS + BLOCK_ALIGN;
        let select = block(jobs).next_multiple_of(BLOCK_ALIGN) + SCRATCH_GROUPS * per_group;
        let convert = Palette4ConvertLayout::new(head_dim, heads).bytes();
        // Every class's lease stack holds at most every band of the table.
        let bands = heads * BANDS_PER_HEAD;
        let lease = SizeClass::COUNT * (bands * 8 + 3 * 4);
        let tables = heads * (PerHeadEntry::COLS + GIDS_PER_HEAD) * 8;
        // Each group's candidates and windows: its sections, and a word a job.
        let words = SCRATCH_GROUPS * 3 * 4 * SECTION_PAD_WORDS + jobs * 4;
        let codes = 2 * 256;
        let segments = (6 + 3 * SCRATCH_GROUPS) * SEGMENT_ALIGN;
        Self {
            select,
            convert,
            inputs: lease + tables + words + codes + segments,
        }
    }

    pub(crate) fn total(&self) -> usize {
        self.select + self.convert + self.inputs
    }
}

/// The words a selection section may be padded by.
const SECTION_PAD_WORDS: usize = 64;

/// The most jobs a table may hold for its scratch to fit `budget` bytes, at
/// `n_kv_head` heads of `head_dim`: the gap's cut (§3.1), from the scratch it
/// holds. At least one job; at most the convert's single launch.
pub fn gap_jobs_within(budget: usize, n_kv_head: usize, head_dim: usize) -> usize {
    let fits = |n: usize| GapScratch::for_jobs(n, n_kv_head, head_dim).total() <= budget;
    let (mut lo, mut hi) = (1usize, MAX_GAP_JOBS + 1);
    if !fits(lo) {
        return 1;
    }
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Where each of `segments` starts when laid end to end on
/// [`SEGMENT_ALIGN`] boundaries, and the length of the whole — the layout
/// [`pack_segments`](candle::cuda_backend::staged::pack_segments) packs.
fn segment_offsets(segments: &[&[u8]]) -> (Vec<usize>, usize) {
    let mut end = 0usize;
    let offsets = segments
        .iter()
        .map(|s| {
            let at = end.next_multiple_of(SEGMENT_ALIGN);
            end = at + s.len();
            at
        })
        .collect();
    (offsets, end)
}

/// The gap's pinned host image, which only grows. Rewritten only once the gap
/// that last copied out of it has completed.
#[derive(Default)]
struct HostImage(Option<PinnedBuf>);

impl HostImage {
    fn at_least(&mut self, bytes: usize) -> Result<&mut [u8]> {
        let held = self.0.as_ref().map_or(0, PinnedBuf::len);
        if held < bytes {
            self.0 = None;
            self.0 = Some(PinnedBuf::alloc_owned(bytes.max(held * 2))?);
        }
        Ok(&mut self.0.as_mut().expect("allocated above")[..bytes])
    }
}

/// The gap's device state, kept across gaps.
pub(crate) struct GapDevice {
    dev: CudaDevice,
    n_kv_head: usize,
    head_dim: usize,
    codes: CodeTables,
    /// The gap's own stream. Each gap waits on the compute stream as it
    /// launches — for the forward that completed its blocks and for the lease's
    /// region cleans — and then runs beside the next forward rather than ahead
    /// of it; the compute stream waits on it once the catch-up has seen the
    /// gap complete. Nothing a forward reads or writes is the gap's while it
    /// runs: its sources are pinned by its jobs, its destinations are the
    /// lease's, and its scratch is its own.
    stream: Arc<CudaStream>,
    summary: MappedSummary,
    /// The claim's look-back status words, zeroed once.
    status: Held<u64>,
    /// The next claim launch's epoch on `status`.
    epoch: u32,
    /// One slot as wide as the widest class: a deferred job's bands are
    /// converted here and nothing reads it.
    sink: Held<u8>,
    select: Grown,
    convert: Grown,
    /// The claim's and the selections' inputs, copied in from `host`.
    inputs: Grown,
    /// Everything a gap uploads, packed.
    host: HostImage,
    /// The completion word of the gap launched and not yet seen complete.
    outstanding: Option<u32>,
    /// The job count of the gap launched and not yet read by a catch-up.
    unread: Option<usize>,
    /// Events on the stream around that gap's work, from its first upload to
    /// its completion word: its device time, read once it has completed.
    timing: Option<(CudaEvent, CudaEvent)>,
}

impl GapDevice {
    /// The gap's device state for tables of up to `max_jobs` jobs of
    /// `n_kv_head` heads at `head_dim`.
    pub(crate) fn new(
        dev: &CudaDevice,
        n_kv_head: usize,
        head_dim: usize,
        max_jobs: usize,
    ) -> Result<Self> {
        if !kvhead_supported_head_dim(head_dim) || !head_dim.is_multiple_of(N_PALETTE) {
            candle::bail!("gap: head_dim {head_dim} has no convert record");
        }
        if n_kv_head == 0 || n_kv_head > MAX_KV_HEAD {
            candle::bail!("gap: {n_kv_head} KV heads, the claim takes 1..={MAX_KV_HEAD}");
        }
        if max_jobs == 0 || max_jobs > MAX_GAP_JOBS {
            candle::bail!("gap: a table of {max_jobs} jobs, the convert takes 1..={MAX_GAP_JOBS}");
        }
        if SizeClass::COUNT > MAX_CLASS {
            candle::bail!(
                "gap: {} size classes, the claim counts {MAX_CLASS}",
                SizeClass::COUNT
            );
        }
        let codes = CodeTables::new(elems_per_chunk(head_dim / N_PALETTE))?;
        let summary = MappedSummary::new(SummaryLayout {
            max_jobs,
            n_kv_head,
            head_dim,
        })?;
        let stream = dev.cuda_context().new_stream().w()?;
        let words = claim_status_words(max_jobs * n_kv_head);
        // SAFETY: cleared just below, before any launch reads it.
        let status = unsafe { Held::<u64>::alloc_on(&stream, words)? };
        let widest = SizeClass::all().map(SizeClass::bytes).max().unwrap_or(0);
        // SAFETY: written by deferred bands' converts and never read.
        let sink = unsafe { Held::<u8>::alloc_on(&stream, widest.max(BLOCK_ALIGN))? };
        // The scratch for the widest table, now: nothing grows between forwards.
        let scratch = GapScratch::for_jobs(max_jobs, n_kv_head, head_dim);
        let (mut select, mut convert, mut inputs) = (Grown(None), Grown(None), Grown(None));
        select.at_least(&stream, scratch.select)?;
        convert.at_least(&stream, scratch.convert)?;
        inputs.at_least(&stream, scratch.inputs)?;
        // The held ground hands bytes out in the compute stream's order: the
        // gap's stream starts behind it.
        {
            let _eager = dev.pause_capture()?;
            stream.join(&dev.cuda_stream()).w()?;
        }
        // SAFETY: the whole status buffer, just allocated on this stream.
        unsafe {
            memset_d8_async(
                status.device_ptr(&stream).0,
                0,
                words * std::mem::size_of::<u64>(),
                stream.cu_stream(),
            )
            .w()?;
        }
        Ok(Self {
            dev: dev.clone(),
            n_kv_head,
            head_dim,
            codes,
            stream,
            summary,
            status,
            epoch: 0,
            sink,
            select,
            convert,
            inputs,
            host: HostImage::default(),
            outstanding: None,
            unread: None,
            timing: None,
        })
    }

    /// The summary the last gap writes, read raw by the tests; everything
    /// else reads it through [`Self::finished`].
    #[cfg(test)]
    pub(crate) fn summary(&self) -> &MappedSummary {
        &self.summary
    }

    /// The gap's stream, for the tests.
    #[cfg(test)]
    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The last launched gap's summary, once — when its completion word is in
    /// and no catch-up has read it yet. `None` while it runs, after it has been
    /// read, and before any gap with a table has been launched. Borrowing the
    /// device state keeps the next gap from being armed over it.
    pub(crate) fn finished(&mut self) -> Option<Finished<'_>> {
        if !self.poll() {
            return None;
        }
        let jobs = self.unread.take()?;
        // Both events are behind the completion word on the stream, so both
        // have fired. A reading that fails reports no time, not a wrong one.
        let device_ms = self
            .timing
            .take()
            .and_then(|(start, end)| start.elapsed_ms(&end).ok())
            .map_or(0.0, f64::from);
        // SAFETY: `poll` saw this gap's completion word, and no later gap has
        // been armed: `launch` refuses while a summary is unread.
        Some(unsafe { Finished::new(&self.summary, jobs, device_ms) })
    }

    /// Order the compute stream after everything the gap's stream has queued:
    /// once a catch-up has seen a gap complete, the forwards that read the
    /// blocks it installs are queued behind its writes, not merely after them.
    pub(crate) fn order_compute_after(&self) -> Result<()> {
        let _eager = self.dev.pause_capture()?;
        self.dev.cuda_stream().join(&self.stream).w()
    }

    /// Wait for everything queued on the gap's stream — for a caller that
    /// must see the gap complete now rather than at a later gap: a settle
    /// before figures are read, and the tests.
    pub(crate) fn wait(&self) -> Result<()> {
        self.stream.synchronize().w()
    }

    /// Whether the gap last launched has completed — a load of one mapped
    /// word. Once it has, its summary may be read and the next gap launched.
    pub(crate) fn poll(&mut self) -> bool {
        match self.outstanding {
            Some(word) if self.summary.complete(word) => {
                self.outstanding = None;
                true
            }
            Some(_) => false,
            None => true,
        }
    }

    /// Launch gap `seq` over `jobs`, quantized group by group, claiming from
    /// the lease laid out as `lease`. Every job's source bands stay the
    /// caller's — pinned, not freed, not rewritten — until the gap completes,
    /// and the lease's slots stay leased until the catch-up takes them.
    pub(crate) fn launch(
        &mut self,
        backings: &[&ChunkedKvBacking],
        jobs: &[GapJob],
        groups: &[GapGroup<'_>],
        lease: &StackImage,
        seq: u64,
    ) -> Result<()> {
        if self.outstanding.is_some() {
            candle::bail!("gap {seq}: the previous gap has not completed");
        }
        if self.unread.is_some() {
            // Arming would clear the summary a catch-up has yet to install.
            candle::bail!("gap {seq}: the previous gap's summary has not been read");
        }
        let (n_kv_head, head_dim) = (self.n_kv_head, self.head_dim);
        let layout = self.summary.layout();
        if jobs.is_empty() {
            return Ok(());
        }
        if jobs.len() > layout.max_jobs {
            candle::bail!(
                "gap {seq}: {} jobs, the gap holds {}",
                jobs.len(),
                layout.max_jobs
            );
        }
        check_groups(groups, jobs.len())?;
        for b in backings {
            if b.n_kv_head() != n_kv_head || b.head_dim() != head_dim {
                candle::bail!("gap {seq}: a layer of another geometry");
            }
        }

        // ── Host images: each group's selection tables, every job's source
        // records and windows. Everything that can refuse the gap does so
        // here, before the first launch ──
        let plans = groups
            .iter()
            .map(|g| GroupPlan::new(g, jobs, backings, &self.codes, n_kv_head, head_dim))
            .collect::<Result<Vec<_>>>()?;
        let n_heads = jobs.len() * n_kv_head;
        let convert = Palette4ConvertLayout::new(head_dim, n_heads);
        let (src_records, ranges) = source_records(jobs, backings, head_dim, n_kv_head)?;

        // ── Device room ──
        let mut select_at = Vec::with_capacity(plans.len());
        let mut select_bytes = 0usize;
        for p in &plans {
            select_at.push(select_bytes);
            select_bytes = (select_bytes + p.inputs.layout.bytes()).next_multiple_of(BLOCK_ALIGN);
        }
        let select_base = self.select.at_least(&self.stream, select_bytes)?;
        let convert_base = self.convert.at_least(&self.stream, convert.bytes())?;

        // ── Every byte the gap reads from the host, in one pinned image ──
        //
        // The convert buffer's source half and windows (the claim writes its
        // destination half), then every input the claim and the selections
        // read. Pinned, so each copy is queued behind the forward and returns at
        // once: from pageable memory the driver stages a copy this size only
        // once the stream reaches it, and the host would wait out the forward
        // ahead of the gap — measured on the RTX 3090 at 250–430 ms for a
        // ×8–×16 prefill gap.
        let lease_len: Vec<u32> = lease.len.to_vec();
        let stack_off: Vec<u32> = lease.off.to_vec();
        let stack: &[u64] = if lease.addrs.is_empty() {
            &[0]
        } else {
            &lease.addrs
        };
        let taken = vec![0u32; SizeClass::COUNT];
        let code_class: Vec<u8> = self.codes.class.iter().map(|&c| c as u8).collect();
        let mut segments: Vec<&[u8]> = vec![
            &src_records,
            segment(&ranges),
            &code_class,
            &self.codes.fmt,
            segment(&lease_len),
            segment(stack),
            segment(&stack_off),
            segment(&taken),
        ];
        for p in &plans {
            segments.push(segment(&p.table));
            segments.push(segment(&p.head_gids));
            segments.push(segment(&p.inputs.words));
        }
        let (offsets, image_len) = segment_offsets(&segments);
        // The inputs start at the third segment, on a segment boundary; the
        // device block is block-aligned, so every segment keeps its alignment.
        let inputs_from = offsets[INPUTS_FROM];
        let inputs_base = self
            .inputs
            .at_least(&self.stream, image_len - inputs_from)?;
        // No copy out of the image is in flight: the last gap's completed
        // before this one may launch, and its copies were behind its word.
        let image = self.host.at_least(image_len)?;
        for (s, &off) in segments.iter().zip(&offsets) {
            image[off..off + s.len()].copy_from_slice(s);
        }
        let image: &[u8] = image;

        let stream = self.stream.clone();
        let _eager = self.dev.pause_capture()?;
        // Behind the forward that completed the table's blocks and the lease's
        // region cleans, both queued on the compute stream.
        stream.join(&self.dev.cuda_stream()).w()?;
        // The gap's device time runs from here — once the forward ahead of it
        // is done, ahead of its first upload — to its completion word.
        let started = stream.record_event(Some(CU_EVENT_DEFAULT)).w()?;
        {
            let span = |i: usize| &image[offsets[i]..offsets[i] + segments[i].len()];
            let held = self.convert.0.as_mut().expect("sized above");
            stream
                .memcpy_htod(span(0), &mut held.slice_mut(..convert.dst_off(0)))
                .w()?;
            stream
                .memcpy_htod(
                    span(1),
                    &mut held.slice_mut(convert.ranges_off()..convert.bytes()),
                )
                .w()?;
            let held = self.inputs.0.as_mut().expect("sized above");
            stream
                .memcpy_htod(
                    &image[inputs_from..image_len],
                    &mut held.slice_mut(..image_len - inputs_from),
                )
                .w()?;
        }
        let at: Vec<u64> = offsets[INPUTS_FROM..]
            .iter()
            .map(|&o| inputs_base + (o - inputs_from) as u64)
            .collect();

        self.summary.arm();
        let status = self.status.device_ptr(&stream).0;
        let sink = self.sink.device_ptr(&stream).0;
        let summary = &self.summary;
        let epoch = &mut self.epoch;
        let launched = (|| -> Result<()> {
            let at = &at;
            let [class_at, fmt_at, lease_at, stack_at, off_at, taken_at] =
                [at[0], at[1], at[2], at[3], at[4], at[5]];
            for (g, (p, group)) in plans.iter().zip(groups).enumerate() {
                let block = select_base + select_at[g] as u64;
                let sel = &p.inputs.layout;
                // SAFETY: the tables and the input words are in the staging
                // scratch, reused only in stream order; `block` holds
                // `sel.bytes()`.
                unsafe {
                    select_palette4_launch_staged(
                        at[6 + 3 * g],
                        at[7 + 3 * g],
                        &p.inputs,
                        at[8 + 3 * g],
                        p.thresholds,
                        head_dim,
                        n_kv_head,
                        GID_STRIDE,
                        block,
                        &stream,
                    )?
                };
                let word = |w: usize| block + (w * 4) as u64;
                let head0 = group.jobs.start * n_kv_head;
                *epoch = epoch.wrapping_add(1);
                // SAFETY: every pointer is sized for the group's jobs as the
                // claim's contract states; the summary and the convert buffer
                // are sized for the whole table, and the group starts at
                // `head0` inside it.
                let rc = unsafe {
                    run_kv_gap_claim(
                        word(sel.k_palette_tags) as *const i32,
                        word(sel.v_palette_tags) as *const i32,
                        word(sel.k_pal_scale) as *const f32,
                        word(sel.v_pal_scale) as *const f32,
                        word(sel.k_palette_map) as *const i32,
                        word(sel.v_palette_map) as *const i32,
                        p.k_override,
                        p.v_override,
                        class_at as *const i8,
                        fmt_at as *const u8,
                        lease_at as *const u32,
                        stack_at as *const u64,
                        off_at as *const u32,
                        taken_at as *mut u32,
                        sink,
                        (convert_base + convert.dst_off(head0) as u64) as *mut u8,
                        summary.dev_at(layout.head_off(head0)) as *mut u8,
                        summary.dev_at(layout.flags_off() + group.jobs.start * 4) as *mut u32,
                        summary.dev_at(layout.error_off()) as *mut u32,
                        status as *mut u64,
                        *epoch,
                        group.jobs.len() as i32,
                        n_kv_head as i32,
                        head_dim as i32,
                        SizeClass::COUNT as i32,
                        stream.cu_stream() as *mut c_void,
                    )
                };
                if rc != 0 {
                    candle::bail!("gap {seq}: the claim refused group {g} ({rc})");
                }
            }
            // SAFETY: the source half and windows were uploaded above and the
            // destination half written by the claims, all ahead on `stream`.
            unsafe {
                palette4_convert_launch(convert_base, head_dim, n_kv_head, jobs.len(), 1, &stream)
                    .map_err(|e| e.context(format!("gap {seq}: the convert")))?;
            }
            // SAFETY: the completion word is in the mapped summary, live
            // while `self` is.
            let rc = unsafe {
                run_kv_gap_complete(
                    summary.dev_at(layout.complete_off()) as *mut u32,
                    completion_word(seq),
                    stream.cu_stream() as *mut c_void,
                )
            };
            if rc != 0 {
                // Never outstanding: a word that was not queued is never stored.
                candle::bail!("gap {seq}: the completion word was refused ({rc})");
            }
            Ok(())
        })();
        if let Err(e) = launched {
            // A launch the driver refused part way leaves the gap's earlier
            // launches queued, still writing the summary and the convert
            // buffer. Wait them out, so the next gap's arm and uploads meet
            // nothing in flight.
            stream.synchronize().w()?;
            return Err(e);
        }
        let ended = stream.record_event(Some(CU_EVENT_DEFAULT)).w()?;
        self.timing = Some((started, ended));
        self.outstanding = Some(completion_word(seq));
        self.unread = Some(jobs.len());
        Ok(())
    }
}

impl Drop for GapDevice {
    fn drop(&mut self) {
        // The summary is freed with `self`, and an outstanding gap's kernels
        // still write it: wait for them first. Nothing to report a failure to,
        // and a failed wait means the context is already gone.
        if self.outstanding.is_some() {
            let _ = self.stream.synchronize();
        }
    }
}

/// Refuse groups that do not cover `0..n_jobs` in order, without overlap.
fn check_groups(groups: &[GapGroup<'_>], n_jobs: usize) -> Result<()> {
    let mut at = 0usize;
    for g in groups {
        if g.jobs.start != at || g.jobs.end <= g.jobs.start {
            candle::bail!("gap: groups must cover the table in order, one run each");
        }
        at = g.jobs.end;
    }
    if at != n_jobs {
        candle::bail!("gap: groups cover {at} of {n_jobs} jobs");
    }
    Ok(())
}

/// One group's host-side inputs.
struct GroupPlan {
    /// The selection's block layout and input words — candidates and windows —
    /// uploaded with the gap's other inputs.
    inputs: Palette4Inputs,
    thresholds: [f32; 4],
    k_override: i32,
    v_override: i32,
    /// The selection's per-(chunk, head) table and band gids, in job order.
    table: Vec<i64>,
    head_gids: Vec<i64>,
}

impl GroupPlan {
    fn new(
        group: &GapGroup<'_>,
        jobs: &[GapJob],
        backings: &[&ChunkedKvBacking],
        codes: &CodeTables,
        n_kv_head: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let policy = group.policy;
        let sel = policy_selection(policy)?;
        let k_candidates: Vec<GgmlDType> = sel.k.iter().map(|f| f.to_ggml_dtype()).collect();
        let v_candidates: Vec<GgmlDType> = sel.v.iter().map(|f| f.to_ggml_dtype()).collect();
        let override_code = |q: Option<QuantFormat>| -> Result<i32> {
            q.map(|q| ggml_to_select_qtype(q.to_ggml_dtype()))
                .transpose()
                .map(|c| c.unwrap_or(NO_OVERRIDE))
        };
        let k_override = override_code(policy.override_k_quant)?;
        let v_override = override_code(policy.override_v_quant)?;
        // Every code the claim can meet is one the tables place.
        let mut reachable = k_candidates
            .iter()
            .chain(&v_candidates)
            .map(|d| ggml_to_select_qtype(*d))
            .collect::<Result<Vec<_>>>()?;
        reachable.extend([k_override, v_override].into_iter().filter(|&c| c >= 0));
        codes.check_covers(&reachable)?;

        let group_jobs = &jobs[group.jobs.clone()];
        let mut table = Vec::new();
        let mut head_gids = Vec::with_capacity(group_jobs.len() * n_kv_head * 2 * N_PALETTE);
        // The table is built a layer's run at a time: one read of that layer's
        // storage per run.
        for run in layer_runs(group_jobs) {
            let layer = group_jobs[run.start].layer;
            let backing = layer_backing(backings, layer)?;
            let bands: Vec<&ChunkBands> = group_jobs[run].iter().map(|j| &j.bands).collect();
            backing
                .inner
                .validate_selection_gids(&bands, &format!("gap layer {layer}"))?;
            table.extend(backing.inner.per_head_table_host(&bands)?);
        }
        for j in group_jobs {
            j.bands
                .gids
                .extend_selection_gids(n_kv_head, &mut head_gids);
        }
        let valid_ranges: Vec<i32> = group_jobs.iter().map(|j| j.valid_range).collect();
        let inputs = select_palette4_inputs(
            group_jobs.len(),
            &k_candidates,
            &v_candidates,
            head_dim,
            n_kv_head,
            &valid_ranges,
        )?;
        Ok(Self {
            inputs,
            thresholds: sel.thresholds,
            k_override,
            v_override,
            table,
            head_gids,
        })
    }
}

/// Whether a band's bytes are a format the selection and convert read
/// directly: any float, or R16.
fn readable_source(tag: u8) -> bool {
    matches!(
        KvFormat::from_tag(tag),
        Some(KvFormat::Float(_)) | Some(KvFormat::Quantized(QuantFormat::R16))
    )
}

/// The runs of consecutive jobs on one layer, as ranges into `jobs`.
fn layer_runs(jobs: &[GapJob]) -> Vec<Range<usize>> {
    runs_by(jobs.len(), |i| jobs[i].layer)
}

/// The maximal runs of `0..n` over which `key` holds one value, in order.
fn runs_by(n: usize, key: impl Fn(usize) -> usize) -> Vec<Range<usize>> {
    let mut runs = Vec::new();
    let mut start = 0;
    for i in 1..=n {
        if i == n || key(i) != key(start) {
            runs.push(start..i);
            start = i;
        }
    }
    runs
}

/// The backing a job's layer names.
fn layer_backing<'b>(
    backings: &[&'b ChunkedKvBacking],
    layer: usize,
) -> Result<&'b ChunkedKvBacking> {
    backings.get(layer).copied().ok_or_else(|| {
        candle::Error::Msg(format!(
            "gap: job layer {layer}, the gap is launched over {} layers",
            backings.len()
        ))
    })
}

/// Every head's source KvHead record — the job's native bands as they lie, at
/// the identity map and unit scales — and its token window, in table order.
/// One read of a layer's storage per run of its jobs, each arena looked up
/// once per run, and every record written into one buffer sized up front.
fn source_records(
    jobs: &[GapJob],
    backings: &[&ChunkedKvBacking],
    head_dim: usize,
    n_kv_head: usize,
) -> Result<(Vec<u8>, Vec<i32>)> {
    let ident = identity_pal_map(head_dim);
    let record_bytes = Palette4ConvertLayout::new(head_dim, 1).record_bytes;
    let mut records = vec![0u8; jobs.len() * n_kv_head * record_bytes];
    let mut ranges = Vec::with_capacity(jobs.len() * n_kv_head);
    let mut out = records.chunks_exact_mut(record_bytes);
    for run in layer_runs(jobs) {
        let layer = jobs[run.start].layer;
        let backing = layer_backing(backings, layer)?;
        backing.inner.storage.read(|storage| -> Result<()> {
            // An arena that is not GPU-resident has no slot address, so the
            // address lookup refuses a band off the device.
            let mut addrs = BandAddrs::new(storage);
            for job in &jobs[run] {
                check_token_window(job.valid_range)
                    .map_err(|e| e.context(format!("gap: a job on layer {layer}")))?;
                for h in 0..n_kv_head {
                    for p in 0..N_PALETTE {
                        let (kt, vt) = job.bands.band_tags(h, p);
                        for tag in [kt, vt] {
                            if !readable_source(tag) {
                                candle::bail!(
                                    "gap: a job's band (tag {tag}) is not a float or R16 band"
                                );
                            }
                        }
                    }
                    let src = head_source(&mut addrs, &job.bands, h)?;
                    write_kvhead_bytes(
                        head_dim,
                        &src.k_ptrs,
                        &src.v_ptrs,
                        &src.k_fmts,
                        &src.v_fmts,
                        &ident,
                        &ident,
                        &[1.0; N_PALETTE],
                        &[1.0; N_PALETTE],
                        out.next().expect("sized for every head"),
                    )?;
                    ranges.push(job.valid_range);
                }
            }
            Ok(())
        })??;
    }
    Ok((records, ranges))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::cuda_backend::staged::pack_segments;

    #[test]
    fn the_scratch_grows_with_the_table_and_covers_its_convert_exactly() {
        let small = GapScratch::for_jobs(10, 4, 128);
        let large = GapScratch::for_jobs(1000, 4, 128);
        assert!(large.select > small.select);
        assert!(large.inputs > small.inputs);
        assert_eq!(
            large.convert,
            Palette4ConvertLayout::new(128, 4000).bytes(),
            "the convert buffer is the convert's own layout"
        );
        // A selection of every job in one group fits the scratch.
        let one_group = Palette4Layout::new(1000, 4, 128, MAX_CANDIDATES, MAX_CANDIDATES).bytes();
        assert!(large.select >= one_group);
    }

    #[test]
    fn the_cut_is_the_most_jobs_whose_scratch_fits_the_budget() {
        let budget = 64 << 20;
        let n = gap_jobs_within(budget, 4, 128);
        // The production budget's cut at the geometries the fleet runs.
        assert_eq!(n, 3866);
        assert_eq!(gap_jobs_within(budget, 2, 256), 4761);
        assert_eq!(gap_jobs_within(budget, 8, 128), 1933);
        assert!(GapScratch::for_jobs(n, 4, 128).total() <= budget);
        assert!(GapScratch::for_jobs(n + 1, 4, 128).total() > budget);
        // Wider heads take more a job, so fewer fit.
        assert!(gap_jobs_within(budget, 4, 256) < n);
        // A budget no job fits still cuts at one; a vast one at the convert's launch.
        assert_eq!(gap_jobs_within(0, 4, 128), 1);
        assert_eq!(gap_jobs_within(usize::MAX / 2, 4, 128), MAX_GAP_JOBS);
    }

    #[test]
    fn the_host_image_is_laid_out_as_the_staged_upload_packs() {
        let a = [1u8, 2, 3];
        let b = [7u64, 8];
        let c: [u8; 0] = [];
        let d = [9u32; 5];
        let segments: [&[u8]; 4] = [&a, segment(&b), &c, segment(&d)];
        let (packed, packed_at) = pack_segments(&segments);
        let (offsets, len) = segment_offsets(&segments);
        assert_eq!(offsets, packed_at);
        assert_eq!(offsets, vec![0, 16, 32, 32]);
        assert_eq!(len, packed.len());
        assert_eq!(len, 52);
        assert_eq!(segment_offsets(&[]), (Vec::new(), 0));
    }

    fn policy() -> CompressionPolicy {
        CompressionPolicy::new(5)
    }

    #[test]
    fn groups_must_cover_the_table_in_order() {
        let p = policy();
        let g = |r: Range<usize>| GapGroup {
            policy: &p,
            jobs: r,
        };
        assert!(check_groups(&[g(0..3), g(3..5)], 5).is_ok());
        assert!(check_groups(&[g(0..3)], 5).is_err());
        assert!(check_groups(&[g(0..3), g(2..5)], 5).is_err());
        assert!(check_groups(&[g(1..5)], 5).is_err());
        assert!(check_groups(&[g(0..0), g(0..5)], 5).is_err());
    }

    #[test]
    fn group_edges() {
        let p = policy();
        let g = |r: Range<usize>| GapGroup {
            policy: &p,
            jobs: r,
        };
        // No groups over no jobs; no groups over some jobs.
        assert!(check_groups(&[], 0).is_ok());
        assert!(check_groups(&[], 3).is_err());
        // One group a job.
        assert!(check_groups(&[g(0..1), g(1..2), g(2..3)], 3).is_ok());
        // Out of order, a gap, and past the table.
        assert!(check_groups(&[g(1..3), g(0..1)], 3).is_err());
        assert!(check_groups(&[g(0..1), g(2..3)], 3).is_err());
        assert!(check_groups(&[g(0..4)], 3).is_err());
    }

    #[test]
    fn an_unrecorded_or_unknown_tag_is_not_a_readable_source() {
        use crate::kv_cache::ArenaFormatTag;
        assert!(!readable_source(ArenaFormatTag::Invalid.as_u8()));
        assert!(!readable_source(u8::MAX));
    }

    #[test]
    fn runs_split_where_the_key_changes() {
        let keys = [0, 0, 1, 1, 1, 0, 2];
        assert_eq!(
            runs_by(keys.len(), |i| keys[i]),
            vec![0..2, 2..5, 5..6, 6..7]
        );
        assert_eq!(runs_by(1, |_| 3), vec![0..1]);
        assert!(runs_by(0, |_| 0).is_empty());
    }

    #[test]
    fn a_layer_past_the_backings_is_refused() {
        assert!(layer_backing(&[], 0).is_err());
    }

    #[test]
    fn floats_and_r16_are_readable_sources_and_other_quants_are_not() {
        use candle::DType;
        assert!(readable_source(KvFormat::Float(DType::F16).to_tag()));
        assert!(readable_source(KvFormat::Float(DType::BF16).to_tag()));
        assert!(readable_source(
            KvFormat::Quantized(QuantFormat::R16).to_tag()
        ));
        assert!(!readable_source(
            KvFormat::Quantized(QuantFormat::Q8_0).to_tag()
        ));
    }
}
