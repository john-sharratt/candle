//! One gap on the device (`docs/kv_quantize_in_the_gap.md` §3.2–§3.4): the
//! format selection, the claim and the convert, launched back to back on the
//! compute stream with nothing read back and nothing waited for.
//!
//! Per policy group the unchanged selection kernel leaves its tags, outer
//! scales and per-channel palette slots in a held block, and the claim reads
//! them there: it gives every band a slot from the lease, defers a job whose
//! class ran dry, and writes each head's destination record into the convert
//! buffer and the mapped summary. One convert then runs over the whole table,
//! and the completion word goes last. The host learns the outcome from the
//! summary once the word is in (§3.5).
//!
//! The device keeps, across gaps: the summary, the claim's look-back status
//! words and their epoch, the sink slot a deferred job's bands are converted
//! into, and held scratch for the selection blocks and the convert buffer that
//! grows to the widest gap. Every buffer is reused only in stream order behind
//! the gap that last read it.

use std::ffi::c_void;
use std::ops::Range;

use candle::cuda_backend::cudarc::driver::result::memset_d8_async;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::cuda_backend::kernels::simple::kv_gap_claim::{
    claim_status_words, run_kv_gap_claim, run_kv_gap_complete, MAX_CLASS, MAX_KV_HEAD, NO_OVERRIDE,
};
use candle::cuda_backend::staged::segment;
use candle::cuda_backend::{CudaDevice, Held, WrapErr};
use candle::quantized::cuda::{
    build_kvhead_bytes_raw, check_token_window, ggml_to_select_qtype, identity_pal_map,
    kvhead_supported_head_dim, palette4_convert_launch, select_palette4_launch,
    Palette4ConvertLayout, Palette4Layout,
};
use candle::quantized::GgmlDType;
use candle::Result;

use super::super::backing::ChunkedKvBacking;
use super::super::compress::{head_source, policy_selection};
use super::super::compression_policy::CompressionPolicy;
use super::super::head_gids::ChunkBands;
use super::super::size_class::{elems_per_chunk, SizeClass, GID_STRIDE};
use super::codes::CodeTables;
use super::lease::StackImage;
use super::summary::{completion_word, MappedSummary, SummaryLayout};
use crate::kv_cache::arena_table::{ArenaLocation, N_PALETTE};
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
    /// The chunk's native bands: their gids and format tags.
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
/// every launch already queued that reads it.
struct Grown(Option<Held<u8>>);

impl Grown {
    #[track_caller]
    fn at_least(&mut self, dev: &CudaDevice, bytes: usize) -> Result<u64> {
        let fits = self
            .0
            .as_mut()
            .is_some_and(|h| h.len() >= bytes && !h.asked_to_move());
        if !fits {
            self.0 = None;
            // SAFETY: every byte a launch reads is written ahead of it on the
            // stream — the uploads, the selection's own clear, or the claim.
            self.0 = Some(unsafe { Held::alloc(dev, bytes.max(BLOCK_ALIGN))? });
        }
        let held = self.0.as_ref().expect("allocated above");
        Ok(held.device_ptr(&dev.cuda_stream()).0)
    }
}

/// The gap's device state, kept across gaps.
pub(crate) struct GapDevice {
    dev: CudaDevice,
    n_kv_head: usize,
    head_dim: usize,
    codes: CodeTables,
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
    /// The completion word of the gap launched and not yet seen complete.
    outstanding: Option<u32>,
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
        let words = claim_status_words(max_jobs * n_kv_head);
        // SAFETY: cleared just below, before any launch reads it.
        let status = unsafe { Held::<u64>::alloc(dev, words)? };
        let widest = SizeClass::all().map(SizeClass::bytes).max().unwrap_or(0);
        // SAFETY: written by deferred bands' converts and never read.
        let sink = unsafe { Held::<u8>::alloc(dev, widest.max(BLOCK_ALIGN))? };
        {
            let stream = dev.cuda_stream();
            let _eager = dev.pause_capture()?;
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
        }
        Ok(Self {
            dev: dev.clone(),
            n_kv_head,
            head_dim,
            codes,
            summary,
            status,
            epoch: 0,
            sink,
            select: Grown(None),
            convert: Grown(None),
            outstanding: None,
        })
    }

    /// The summary the last gap writes.
    pub(crate) fn summary(&self) -> &MappedSummary {
        &self.summary
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
            select_bytes = (select_bytes + p.layout.bytes()).next_multiple_of(BLOCK_ALIGN);
        }
        let select_base = self.select.at_least(&self.dev, select_bytes)?;
        let convert_base = self.convert.at_least(&self.dev, convert.bytes())?;

        let stream = self.dev.cuda_stream();
        let _eager = self.dev.pause_capture()?;
        // The convert buffer's source half and windows; the claim writes its
        // destination half.
        {
            let held = self.convert.0.as_mut().expect("sized above");
            stream
                .memcpy_htod(&src_records, &mut held.slice_mut(..convert.dst_off(0)))
                .w()?;
            stream
                .memcpy_htod(
                    segment(&ranges),
                    &mut held.slice_mut(convert.ranges_off()..convert.bytes()),
                )
                .w()?;
        }

        // ── Every input the claim and the selections read, in one upload ──
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
        }

        self.summary.arm();
        let status = self.status.device_ptr(&stream).0;
        let sink = self.sink.device_ptr(&stream).0;
        let summary = &self.summary;
        let epoch = &mut self.epoch;
        let launched = self.dev.with_staged_segments(&segments, |at| {
            let [class_at, fmt_at, lease_at, stack_at, off_at, taken_at] =
                [at[0], at[1], at[2], at[3], at[4], at[5]];
            for (g, (p, group)) in plans.iter().zip(groups).enumerate() {
                let block = select_base + select_at[g] as u64;
                // SAFETY: both tables are in the staging scratch, reused only
                // in stream order; `block` holds `p.layout.bytes()`.
                let sel = unsafe {
                    select_palette4_launch(
                        at[6 + 2 * g],
                        at[7 + 2 * g],
                        p.jobs.len(),
                        &p.k_candidates,
                        &p.v_candidates,
                        p.thresholds,
                        head_dim,
                        n_kv_head,
                        GID_STRIDE,
                        &p.valid_ranges,
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
                palette4_convert_launch(convert_base, head_dim, n_kv_head, jobs.len(), 1, &stream);
                run_kv_gap_complete(
                    summary.dev_at(layout.complete_off()) as *mut u32,
                    completion_word(seq),
                    stream.cu_stream() as *mut c_void,
                );
            }
            Ok(())
        });
        if let Err(e) = launched {
            // A launch the driver refused part way leaves the gap's earlier
            // launches queued, still writing the summary and the convert
            // buffer. Wait them out, so the next gap's arm and uploads meet
            // nothing in flight.
            stream.synchronize().w()?;
            return Err(e);
        }
        self.outstanding = Some(completion_word(seq));
        Ok(())
    }
}

impl Drop for GapDevice {
    fn drop(&mut self) {
        // The summary is freed with `self`, and an outstanding gap's kernels
        // still write it: wait for them first. Nothing to report a failure to,
        // and a failed wait means the context is already gone.
        if self.outstanding.is_some() {
            let _ = self.dev.cuda_stream().synchronize();
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
    jobs: Range<usize>,
    layout: Palette4Layout,
    k_candidates: Vec<GgmlDType>,
    v_candidates: Vec<GgmlDType>,
    thresholds: [f32; 4],
    k_override: i32,
    v_override: i32,
    valid_ranges: Vec<i32>,
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
        let layout = Palette4Layout::new(
            group_jobs.len(),
            n_kv_head,
            head_dim,
            k_candidates.len(),
            v_candidates.len(),
        );
        Ok(Self {
            jobs: group.jobs.clone(),
            layout,
            k_candidates,
            v_candidates,
            thresholds: sel.thresholds,
            k_override,
            v_override,
            valid_ranges: group_jobs.iter().map(|j| j.valid_range).collect(),
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
/// One read of a layer's storage per run of its jobs.
fn source_records(
    jobs: &[GapJob],
    backings: &[&ChunkedKvBacking],
    head_dim: usize,
    n_kv_head: usize,
) -> Result<(Vec<u8>, Vec<i32>)> {
    let ident = identity_pal_map(head_dim);
    let record_bytes = Palette4ConvertLayout::new(head_dim, 1).record_bytes;
    let mut records = Vec::with_capacity(jobs.len() * n_kv_head * record_bytes);
    let mut ranges = Vec::with_capacity(jobs.len() * n_kv_head);
    for run in layer_runs(jobs) {
        let layer = jobs[run.start].layer;
        let backing = layer_backing(backings, layer)?;
        backing.inner.storage.read(|storage| -> Result<()> {
            for job in &jobs[run] {
                check_token_window(job.valid_range)
                    .map_err(|e| e.context(format!("gap: a job on layer {layer}")))?;
                for h in 0..n_kv_head {
                    for p in 0..N_PALETTE {
                        let (kt, vt) = job.bands.band_tags(h, p);
                        for (gid, tag) in [
                            (job.bands.gids.k_gid_pal(h, p), kt),
                            (job.bands.gids.v_gid_pal(h, p), vt),
                        ] {
                            let on_gpu = storage
                                .arena_key(gid.arena_idx())
                                .is_some_and(|k| k.location == ArenaLocation::Gpu);
                            if !on_gpu || !readable_source(tag) {
                                candle::bail!(
                                    "gap: a job's band (tag {tag}) is not a GPU float or R16 band"
                                );
                            }
                        }
                    }
                    let src = head_source(storage, &job.bands, h)?;
                    records.extend(build_kvhead_bytes_raw(
                        head_dim,
                        &src.k_ptrs,
                        &src.v_ptrs,
                        &src.k_fmts,
                        &src.v_fmts,
                        &ident,
                        &ident,
                        &[1.0; N_PALETTE],
                        &[1.0; N_PALETTE],
                    )?);
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
