//! The paged belief scan: build the tiny per-scan index over the resident arena
//! and launch the paged BDP kernel.
//!
//! The gallery records stay resident (in the arena); a scan uploads only the
//! addressing — `page_ptr` (device address per page), `pos_map` (page + offset
//! per scanned token), `case`, the segment prefixes, and the probe — then runs
//! the same kernel as the contiguous path in its paged mode. The host tally is
//! shared with the contiguous path ([`needle_tally_segments`]). See
//! `docs/archived/paged_gallery_arena.md` §7–8.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use candle::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DriverError};
use candle::{CudaDevice, Device, Result};
use candle_kernels::provenance::{
    bdp_bmma_supported, bdp_imma_supported, bdp_take_pending_error, run_batched_bdp_scan,
    run_bmma_bdp_scan, run_imma_bdp_scan,
};
use core::ffi::c_void;

use crate::persistence::streams::StreamId;
use crate::scheduler::profile;

use super::super::gpu::{needle_tally_segments, BDP_TQ};
use super::super::WideQSig;
use super::pages::PAGE_TOKENS;
use super::probe_rows::ProbeRows;
use super::GalleryArena;

/// A sub-window of a resident turn that contributes to a scan: `[start, end)`
/// tokens of the turn identified by `sid`/`fingerprint`, voting for `case`.
pub struct PagedWindow<'a> {
    pub sid: StreamId,
    pub fingerprint: u64,
    /// The WHOLE turn window (for residency); the scan reads `[start, end)`.
    pub turn: &'a [WideQSig],
    pub start: usize,
    pub end: usize,
    pub case: usize,
}

/// One segment (file): its windows and its exchange (case) count.
pub struct PagedSegment<'a> {
    pub windows: Vec<PagedWindow<'a>>,
    pub n_cases: usize,
}

/// The index's arrays on the device — uploaded once, when the index is built,
/// and read in place by every launch that reuses it.
///
/// They used to be uploaded per launch, from pageable host memory, even when the
/// index itself was reused: `pos_map` and `case` are one `u32` per scanned
/// token, so a 7.4M-token `code_reading` scan moved ~60 MB host-to-device on
/// every launch — twice per reprojection (tail and question probes).
struct DeviceIndex {
    page_ptr: CudaSlice<u64>,
    pos_map: CudaSlice<u32>,
    case: CudaSlice<u32>,
    seg_tok: CudaSlice<i32>,
    seg_case: CudaSlice<i32>,
}

/// The assembled per-scan index, built from the resident arena. `page_ptr`
/// holds absolute device addresses, valid while every referenced turn keeps
/// the run it was built against (see `run_ids`).
pub(super) struct PagedIndex {
    /// `None` when the scan covers no token — nothing to launch.
    device: Option<DeviceIndex>,
    /// Scanned tokens (the length of `pos_map`).
    pub(super) n_tokens: usize,
    /// Host copy of the per-segment case prefixes, which the tally reads.
    seg_case: Vec<i32>,
    n_cases: usize,
    n_segments: usize,
    max_seg_cases: usize,
    /// Turns referenced by this scan; pinned for its duration then unpinned.
    pub(super) pinned_sids: Vec<StreamId>,
    /// Each referenced turn's run id when its addresses were taken, parallel to
    /// `pinned_sids` — what a later reuse checks the arena against.
    pub(super) run_ids: Vec<u64>,
    /// Every probe token's row a launch over this index has produced, so the
    /// next launch scores only the tokens no earlier one did (see
    /// [`ProbeRows`]).
    rows: Mutex<ProbeRows>,
}

impl PagedIndex {
    /// Device bytes the index's arrays hold — what caching it costs the card,
    /// outside the span and the arena's own ceiling.
    pub(super) fn device_bytes(&self) -> u64 {
        self.device.as_ref().map_or(0, |d| {
            (d.page_ptr.len() * 8
                + (d.pos_map.len() + d.case.len()) * 4
                + (d.seg_tok.len() + d.seg_case.len()) * 4) as u64
        })
    }
}

/// A [`PagedIndex`] cached for reuse (keyed by segment fingerprint in the arena's
/// map), valid while every turn it references still holds the run it recorded.
pub(super) struct CachedIndex {
    pub(super) idx: Arc<PagedIndex>,
    /// When a scan last stored or reused it — the cache evicts the oldest first.
    pub(super) used: u64,
}

/// One scan backend. The auto ladder resolves per DEVICE, fastest first: b1
/// BMMA where the hardware has it (sm_75..sm_89), then INT8 IMMA (sm_80+ —
/// Hopper/Blackwell, which dropped b1), then the scalar kernel. The forced
/// variants keep every backend first-class for differential testing and
/// benchmarking — all three produce bit-identical votes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PagedBackend {
    Bmma,
    Imma,
    Scalar,
}

/// A tensor launcher's negative return code, decoded: `(stage, cudaError)`.
///
/// The launchers return `-(stage * 1000 + cudaError)`, where the stage names
/// WHICH of their CUDA calls rejected the work (1 alloc, 2 memset, 3 accumulate
/// launch, 4 finalize launch). A bare `-cudaError` names none of them and reads
/// as `unstaged`.
fn launch_failure(rc: i32) -> (&'static str, i32) {
    let (stage, cuda_err) = if rc <= -1000 {
        ((-rc) / 1000, (-rc) % 1000)
    } else {
        (0, -rc)
    };
    let stage = match stage {
        1 => "alloc",
        2 => "memset",
        3 => "accum_launch",
        4 => "finalize_launch",
        _ => "unstaged",
    };
    (stage, cuda_err)
}

/// Fingerprint the segment structure the built index depends on — file order,
/// each file's case count, and every window's `(sid, fingerprint, turn length,
/// start, end, case)`. Cheap (per-window, not per-token).
///
/// **Trust boundary:** two guarantees back reuse. (1) Each turn's *run id*
/// covers physical page moves — an evict/re-seal/compaction that relocates a
/// turn's pages gives it a new one, so the cache can't serve stale device
/// addresses. (2) This 64-bit
/// SipHash covers logical changes to the scan; a caller MUST vary `w.fingerprint`
/// whenever the turn's content changes (the resolver's `fp_of` folds in the
/// decoded-sig `Arc` len + a content sample), and `turn.len()` is hashed directly
/// so a length change alone (which shifts `pos_map`) also invalidates. A 64-bit
/// hash collision under an unchanged generation could serve a wrong index — a
/// ~2⁻⁶⁴ event we accept, the standard content-address trade.
fn fingerprint_segments(segments: &[PagedSegment]) -> u64 {
    let mut h = DefaultHasher::new();
    segments.len().hash(&mut h);
    for seg in segments {
        seg.n_cases.hash(&mut h);
        seg.windows.len().hash(&mut h);
        for w in &seg.windows {
            w.sid.0.hash(&mut h);
            w.fingerprint.hash(&mut h);
            w.turn.len().hash(&mut h);
            w.start.hash(&mut h);
            w.end.hash(&mut h);
            w.case.hash(&mut h);
        }
    }
    h.finish()
}

impl GalleryArena {
    /// Ensure every referenced turn is resident and assemble the paged index.
    /// A turn is uploaded at most once per scan (deduped by `sid`); its pages'
    /// device addresses are appended to `page_ptr` in first-encounter order and
    /// every window into that turn resolves against the recorded base.
    fn build_index(&self, segments: &[PagedSegment]) -> Result<PagedIndex> {
        let wpt = self.wpt();
        let mut turn_base: HashMap<StreamId, usize> = HashMap::new();
        let mut page_ptr: Vec<u64> = Vec::new();
        let mut pos_map: Vec<u32> = Vec::new();
        let mut case: Vec<u32> = Vec::new();
        let mut seg_tok = vec![0i32];
        let mut seg_case = vec![0i32];
        let mut case_off = 0usize;
        let mut max_seg_cases = 1usize;

        // Turns pinned so far — released on error so a failed build never leaks pins.
        let mut pinned_sids: Vec<StreamId> = Vec::new();
        let mut run_ids: Vec<u64> = Vec::new();
        for seg in segments {
            // Emit each segment's windows in CASE order, so gallery case ids are
            // non-decreasing over the scan order. The scan math is order-independent
            // (per-case max/sum), but the BMMA backend dense-ranks each chunk's
            // cases with a single monotone walk — this ordering is its invariant.
            // The resolver's exchange slots already arrive sorted (stable no-op).
            let mut order: Vec<usize> = (0..seg.windows.len()).collect();
            order.sort_by_key(|&i| seg.windows[i].case);
            for &wi in &order {
                let w = &seg.windows[wi];
                // Mirror `PackedGallery::from_windows`: drop an out-of-range case.
                if w.case >= seg.n_cases {
                    continue;
                }
                let base = match turn_base.get(&w.sid) {
                    Some(&b) => b,
                    None => {
                        // Pin the turn atomically with residency so the governor
                        // can't free its pages before the launch reads them.
                        let (addrs, run_id) = match self.scan_ensure(w.sid, w.turn, w.fingerprint) {
                            Ok(a) => a,
                            Err(e) => {
                                for &s in &pinned_sids {
                                    self.unpin(s);
                                }
                                return Err(e);
                            }
                        };
                        pinned_sids.push(w.sid);
                        run_ids.push(run_id);
                        let b = page_ptr.len();
                        page_ptr.extend_from_slice(&addrs);
                        turn_base.insert(w.sid, b);
                        b
                    }
                };
                for t in w.start..w.end.min(w.turn.len()) {
                    // Uniform width only (mirror the reference's admission guard).
                    if w.turn[t].words.len() != wpt {
                        continue;
                    }
                    let page = base + t / PAGE_TOKENS;
                    let in_pg = t % PAGE_TOKENS;
                    pos_map.push(((page as u32) << 5) | (in_pg as u32));
                    case.push((case_off + w.case) as u32);
                }
            }
            case_off += seg.n_cases;
            seg_tok.push(pos_map.len() as i32);
            seg_case.push(case_off as i32);
            max_seg_cases = max_seg_cases.max(seg.n_cases);
        }

        let n_tokens = pos_map.len();
        let device = if n_tokens == 0 {
            None
        } else {
            match self.upload_index(&page_ptr, &pos_map, &case, &seg_tok, &seg_case) {
                Ok(d) => Some(d),
                Err(e) => {
                    for &s in &pinned_sids {
                        self.unpin(s);
                    }
                    return Err(e);
                }
            }
        };
        Ok(PagedIndex {
            device,
            n_tokens,
            seg_case,
            n_cases: case_off,
            n_segments: segments.len(),
            max_seg_cases,
            pinned_sids,
            run_ids,
            rows: Mutex::new(ProbeRows::new(self.n_groups() * segments.len())),
        })
    }

    /// Upload a built index's arrays once, onto the arena device's stream, for
    /// every launch that reuses the index to read in place.
    fn upload_index(
        &self,
        page_ptr: &[u64],
        pos_map: &[u32],
        case: &[u32],
        seg_tok: &[i32],
        seg_case: &[i32],
    ) -> Result<DeviceIndex> {
        let Device::Cuda(dev) = &self.device else {
            return Err(candle::Error::Msg("paged scan requires CUDA".into()));
        };
        let stream = dev.cuda_stream();
        let up = |what: &str, e: DriverError| {
            candle::Error::Msg(format!("paged scan: HtoD {what}: {e}"))
        };
        Ok(DeviceIndex {
            page_ptr: stream
                .memcpy_stod(page_ptr)
                .map_err(|e| up("page_ptr", e))?,
            pos_map: stream.memcpy_stod(pos_map).map_err(|e| up("pos_map", e))?,
            case: stream.memcpy_stod(case).map_err(|e| up("case", e))?,
            seg_tok: stream.memcpy_stod(seg_tok).map_err(|e| up("seg_tok", e))?,
            seg_case: stream
                .memcpy_stod(seg_case)
                .map_err(|e| up("seg_case", e))?,
        })
    }

    /// Paged belief scan over the resident gallery. Ensures residency, builds the
    /// index, launches the paged kernel, and tallies — one per-case vote vector
    /// per probe. Numerically equivalent to the contiguous
    /// [`BatchedGpuGallery::scan_weighted`](super::super::gpu::BatchedGpuGallery)
    /// on the same windows (only the byte layout differs). Runs on the b1
    /// tensor-core (BMMA) backend when the device has it (sm_75..sm_89) and the
    /// geometry is the locked fold (`gw == 8`); the scalar kernel otherwise.
    pub fn scan_weighted(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        group_weights: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        self.scan_weighted_impl(segments, probes, group_weights, None)
    }

    /// [`Self::scan_weighted`] forced onto the scalar kernel — the universal
    /// fallback backend, kept first-class for differential testing and
    /// benchmarking (all backends' integer statistics are identical).
    pub fn scan_weighted_scalar(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        group_weights: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        self.scan_weighted_impl(segments, probes, group_weights, Some(PagedBackend::Scalar))
    }

    /// [`Self::scan_weighted`] forced onto the b1 tensor-core (BMMA) kernel —
    /// the auto ladder's top rung on sm_75..sm_89. Forcing it surfaces a
    /// backend-specific launch failure as a hard error instead of the ladder's
    /// silent degrade, which is what differential tests and geometry
    /// reproductions need. Errors on devices without b1 BMMA.
    pub fn scan_weighted_bmma(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        group_weights: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        self.scan_weighted_impl(segments, probes, group_weights, Some(PagedBackend::Bmma))
    }

    /// [`Self::scan_weighted`] forced onto the INT8 tensor-core (IMMA) kernel —
    /// the backend the auto ladder selects on devices without b1 BMMA (Hopper/
    /// Blackwell). Forcing it keeps the path benchmarkable and differential-
    /// tested on b1 hardware too. Errors on devices without INT8 MMA (below
    /// sm_80).
    pub fn scan_weighted_imma(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        group_weights: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        self.scan_weighted_impl(segments, probes, group_weights, Some(PagedBackend::Imma))
    }

    /// This arena device's tensor capabilities `(b1 BMMA, INT8 IMMA)`, queried
    /// once per arena. The FFI probes the calling thread's current CUDA device,
    /// which is the arena's device on every scan path (scans run on threads
    /// holding the arena's context).
    fn tensor_caps(&self) -> (bool, bool) {
        *self
            .tensor_caps
            .get_or_init(|| unsafe { (bdp_bmma_supported() != 0, bdp_imma_supported() != 0) })
    }

    fn scan_weighted_impl(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        group_weights: &[f32],
        force: Option<PagedBackend>,
    ) -> Result<Vec<Vec<f32>>> {
        // **Every scan path binds this arena's context first.**
        //
        // Below here are raw driver calls and FFI that read the *calling
        // thread's* current device — `tensor_caps` says so explicitly. That
        // used to be free, because scans only ever ran on threads that already
        // held the context. The normalization warm-ups moved onto their own
        // rayon pool, whose workers never bound it, and the whole path failed
        // with `CUDA_ERROR_INVALID_CONTEXT` — an error that reads like a
        // hardware fault and is really a thread that was never introduced to
        // the device. Binding here makes the assumption true for any caller
        // instead of documenting it and hoping.
        if let Device::Cuda(dev) = &self.device {
            dev.bind_to_thread()?;
        }
        // Reuse the cached index if the same segment set is rescanned and its
        // turns still hold the runs it recorded (the common case, whatever other
        // conversations upload meanwhile) — this pins the turns. Otherwise
        // rebuild (which also pins) and cache it. Each turn's run id is taken
        // with its addresses under the residency lock, so a turn moved after
        // that point carries a new id and the next reuse rejects the entry.
        let t_index = Instant::now();
        let gen_before = self.residency_gen();
        let fp = fingerprint_segments(segments);
        let (idx, reused) = match self.reuse_index(fp) {
            Some(idx) => (idx, true),
            None => {
                let built = Arc::new(self.build_index(segments)?);
                self.store_index(fp, built.clone());
                (built, false)
            }
        };
        let index_us = t_index.elapsed().as_micros() as u64;
        profile::record(
            if reused {
                "arena:index_reused"
            } else {
                "arena:index_built"
            },
            t_index.elapsed(),
        );
        // Residency mutations (uploads, evictions, moves) on the whole device
        // while this index was found or built — this scan's and any concurrent
        // conversation's. With `reused` it separates "my working set churned"
        // from "someone else's did".
        let mutations = self.residency_gen().saturating_sub(gen_before);
        let t_launch = Instant::now();
        let result = self.launch_paged(&idx, probes, group_weights, force);
        let launch_us = t_launch.elapsed().as_micros() as u64;
        profile::record("arena:launch", t_launch.elapsed());
        // Release the scan's pins whether or not the launch succeeded — the pages
        // are no longer being read once the launch has synchronized (or failed).
        for &sid in &idx.pinned_sids {
            self.unpin(sid);
        }
        // Index (reuse or rebuild, which pins pages resident) versus launch
        // (which synchronizes and tallies on the host). They have unrelated
        // costs, so a slow scan is attributed to one or the other.
        tracing::trace!(
            target: "candle_conversation::provenance::gallery_arena",
            probes = probes.len(),
            segments = segments.len(),
            turns = idx.pinned_sids.len(),
            tokens = idx.n_tokens,
            reused,
            mutations,
            held_mib = self.page_bytes() >> 20,
            index_us,
            launch_us,
            "arena scan"
        );
        result
    }

    fn launch_paged(
        &self,
        idx: &PagedIndex,
        probes: &[&[WideQSig]],
        group_weights: &[f32],
        force: Option<PagedBackend>,
    ) -> Result<Vec<Vec<f32>>> {
        let dev = match self.device() {
            Device::Cuda(d) => d,
            _ => return Err(candle::Error::Msg("paged scan requires CUDA".into())),
        };
        let wpt = self.wpt();
        let n_groups = self.n_groups();
        let n_cases = idx.n_cases;
        let n_segments = idx.n_segments;

        if probes.is_empty() {
            return Ok(Vec::new());
        }
        let Some(d_idx) = idx.device.as_ref() else {
            return Ok(vec![vec![0.0; n_cases]; probes.len()]);
        };
        if n_segments == 0 || n_cases == 0 {
            return Ok(vec![vec![0.0; n_cases]; probes.len()]);
        }

        // Batch probes (token-major, full-width only — the reference's filter).
        let mut probe_words: Vec<u64> = Vec::new();
        let mut per_req_tokens: Vec<usize> = Vec::with_capacity(probes.len());
        for probe in probes {
            let mut cnt = 0usize;
            for tok in *probe {
                if tok.words.len() >= wpt {
                    probe_words.extend_from_slice(&tok.words[..wpt]);
                    cnt += 1;
                }
            }
            per_req_tokens.push(cnt);
        }
        let n_probe_tokens = probe_words.len() / wpt;
        if n_probe_tokens == 0 {
            return Ok(vec![vec![0.0; n_cases]; probes.len()]);
        }

        // Only the tokens no earlier launch over this index scored go to the
        // kernel; the rest are read back from the rows it kept (`ProbeRows`).
        // Held to the end of the launch, so a concurrent scan of the same index
        // cannot start the cache over between this plan and the assembly.
        let t_plan = Instant::now();
        let mut rows = idx.rows.lock().unwrap_or_else(|e| e.into_inner());
        let fresh = rows.plan(&probe_words, wpt);
        profile::record("arena:rows_plan", t_plan.elapsed());
        let (upload_us, kernel_us, readback_us) = if fresh.is_empty() {
            (0, 0, 0)
        } else {
            let tokens: Vec<&[u64]> = fresh
                .iter()
                .map(|&t| &probe_words[t * wpt..(t + 1) * wpt])
                .collect();
            let words: Vec<u64> = tokens.concat();
            let (case, vote, timing) =
                self.launch_kernel(dev, d_idx, idx, &words, tokens.len(), force)?;
            let t_insert = Instant::now();
            rows.insert(&tokens, &case, &vote);
            profile::record("arena:rows_insert", t_insert.elapsed());
            timing
        };
        let t_assemble = Instant::now();
        let (out_case, out_vote) = rows
            .assemble(&probe_words, wpt)
            .expect("every probe token has a row once its launch is in");
        drop(rows);
        profile::record("arena:rows_assemble", t_assemble.elapsed());

        let t_tally = Instant::now();
        let votes = needle_tally_segments(
            &out_case,
            &out_vote,
            &per_req_tokens,
            &idx.seg_case,
            n_groups,
            n_segments,
            n_cases,
            group_weights,
        );
        profile::record("arena:launch_upload", Duration::from_micros(upload_us));
        profile::record("arena:launch_kernel", Duration::from_micros(kernel_us));
        profile::record("arena:launch_readback", Duration::from_micros(readback_us));
        profile::record("arena:launch_tally", t_tally.elapsed());
        // Where a launch's time goes: the probe upload, the kernel (to its
        // synchronize), the vote readback, and the host tally — over the
        // `scored` tokens no earlier launch over this index had scored.
        tracing::trace!(
            target: "candle_conversation::provenance::gallery_arena",
            tokens = idx.n_tokens,
            probe_tokens = n_probe_tokens,
            scored = fresh.len(),
            segments = n_segments,
            upload_us,
            kernel_us,
            readback_us,
            tally_us = t_tally.elapsed().as_micros() as u64,
            "arena launch"
        );
        Ok(votes)
    }

    /// One kernel launch over `n_probe_tokens` probe tokens (`probe_words`,
    /// `wpt` words each) against `idx`, on the fastest backend the device and
    /// geometry admit. Returns the raw output — one row per probe token — and
    /// the `(upload, kernel, readback)` microseconds.
    #[allow(clippy::type_complexity)]
    fn launch_kernel(
        &self,
        dev: &CudaDevice,
        d_idx: &DeviceIndex,
        idx: &PagedIndex,
        probe_words: &[u64],
        n_probe_tokens: usize,
        force: Option<PagedBackend>,
    ) -> Result<(Vec<i32>, Vec<f32>, (u64, u64, u64))> {
        let wpt = self.wpt();
        let n_groups = self.n_groups();
        let gw = wpt / n_groups;
        let n_cases = idx.n_cases;
        let n_segments = idx.n_segments;

        // Backend candidates — see [`PagedBackend`]. A forced backend is exactly
        // one candidate; auto lists every rung this arena's device might run,
        // fastest first, so a runtime gate mismatch (launcher rc == 1 — a
        // host-side pre-check, nothing enqueued) degrades to the next rung
        // instead of failing the scan. The scalar rung's largest-segment
        // shared-memory guard is checked when that rung is reached.
        let (has_bmma, has_imma) = self.tensor_caps();
        let mut candidates: Vec<PagedBackend> = Vec::with_capacity(3);
        match force {
            Some(b) => candidates.push(b),
            None => {
                if gw == 8 && has_bmma {
                    candidates.push(PagedBackend::Bmma);
                }
                if gw == 8 && has_imma {
                    candidates.push(PagedBackend::Imma);
                }
                candidates.push(PagedBackend::Scalar);
            }
        }

        let stream = dev.cuda_stream();

        // Only the probe is uploaded per launch: the index arrays are already
        // on the device (`DeviceIndex`), and the records stay resident.
        let t_upload = Instant::now();
        let d_probe = stream
            .memcpy_stod(probe_words)
            .map_err(|e| candle::Error::Msg(format!("paged scan: HtoD probes: {e}")))?;

        // An error already pending on this thread is an EARLIER launch's that
        // nothing checked. Taken and named here, before any backend runs: a
        // launcher reads `cudaGetLastError` after its own work, so left pending
        // it is reported as that backend failing, and the fall to the next rung
        // buries the launch that actually failed.
        let pending = unsafe { bdp_take_pending_error() };
        if pending != 0 {
            tracing::warn!(
                target: "candle_conversation::provenance",
                cuda_err = pending,
                "paged scan: an earlier CUDA launch on this thread left error {pending} \
                 unchecked"
            );
        }

        let n_out = n_probe_tokens * n_groups * n_segments;
        let mut d_out_case = unsafe { stream.alloc::<i32>(n_out) }
            .map_err(|e| candle::Error::Msg(format!("paged scan: alloc out_case: {e}")))?;
        let mut d_out_vote = unsafe { stream.alloc::<f32>(n_out) }
            .map_err(|e| candle::Error::Msg(format!("paged scan: alloc out_vote: {e}")))?;

        let upload_us = t_upload.elapsed().as_micros() as u64;
        let t_kernel = Instant::now();
        {
            let (p_case, _g1) = d_idx.case.device_ptr(&stream);
            let (p_probe, _g2) = d_probe.device_ptr(&stream);
            let (p_seg_tok, _g3) = d_idx.seg_tok.device_ptr(&stream);
            let (p_seg_case, _g4) = d_idx.seg_case.device_ptr(&stream);
            let (p_page_ptr, _g5) = d_idx.page_ptr.device_ptr(&stream);
            let (p_pos_map, _g6) = d_idx.pos_map.device_ptr(&stream);
            let (p_out_case, _g7) = d_out_case.device_ptr_mut(&stream);
            let (p_out_vote, _g8) = d_out_vote.device_ptr_mut(&stream);
            let mut launched = false;
            for &backend in &candidates {
                match backend {
                    PagedBackend::Bmma | PagedBackend::Imma => {
                        // The two tensor launchers share one signature and contract.
                        let launcher = if backend == PagedBackend::Bmma {
                            run_bmma_bdp_scan
                        } else {
                            run_imma_bdp_scan
                        };
                        let rc = unsafe {
                            launcher(
                                p_case as *const u32,
                                p_probe as *const u64,
                                p_seg_tok as *const i32,
                                p_seg_case as *const i32,
                                p_page_ptr as *const u64,
                                p_pos_map as *const u32,
                                idx.n_tokens as i32,
                                n_probe_tokens as i32,
                                n_groups as i32,
                                n_segments as i32,
                                n_cases as i32,
                                gw as i32,
                                wpt as i32,
                                p_out_case as *mut i32,
                                p_out_vote as *mut f32,
                                stream.cu_stream() as *mut c_void,
                            )
                        };
                        if rc == 0 {
                            launched = true;
                            break;
                        }
                        if rc == 1 {
                            // The launcher's host-side gate declined (device or
                            // geometry) — nothing was enqueued; try the next rung.
                            tracing::debug!(
                                target: "candle_conversation::provenance",
                                "paged scan: {backend:?} gate declined, next rung"
                            );
                            continue;
                        }
                        // Negative rc: a CUDA error mid-sequence — work may
                        // already be in flight, so drain the stream BEFORE
                        // continuing (the caller unpins the scanned turns on
                        // return, and the governor must never free pages a
                        // still-running kernel is reading). After the drain the
                        // stream is quiescent, so the NEXT rung can safely
                        // retry the same geometry — a backend-specific launch
                        // failure degrades to the next backend, not to the CPU.
                        let _ = stream.synchronize();
                        let (stage_name, cuda_err) = launch_failure(rc);
                        tracing::warn!(
                            target: "candle_conversation::provenance",
                            backend = ?backend,
                            rc,
                            stage = stage_name,
                            cuda_err,
                            n_tokens = idx.n_tokens,
                            n_probe_tokens,
                            n_groups,
                            n_segments,
                            n_cases,
                            "paged scan: launch failed; draining and trying next rung"
                        );
                        continue;
                    }
                    PagedBackend::Scalar => {
                        // Shared-memory budget guard: the scalar kernel's dynamic
                        // shared scales with the largest segment's case count.
                        let shmem = BDP_TQ * idx.max_seg_cases * std::mem::size_of::<u32>();
                        if shmem > 48 * 1024 {
                            return Err(candle::Error::Msg(format!(
                                "paged scan: largest segment has {} cases \
                                 ({shmem} B shared > 48 KiB)",
                                idx.max_seg_cases
                            )));
                        }
                        unsafe {
                            run_batched_bdp_scan(
                                std::ptr::null(), // gallery_words — paged mode
                                p_case as *const u32,
                                p_probe as *const u64,
                                p_seg_tok as *const i32,
                                p_seg_case as *const i32,
                                p_page_ptr as *const u64,
                                p_pos_map as *const u32,
                                n_probe_tokens as i32,
                                n_groups as i32,
                                n_segments as i32,
                                idx.max_seg_cases as i32,
                                gw as i32,
                                wpt as i32,
                                p_out_case as *mut i32,
                                p_out_vote as *mut f32,
                                stream.cu_stream() as *mut c_void,
                            );
                        }
                        launched = true;
                        break;
                    }
                }
            }
            if !launched {
                return Err(candle::Error::Msg(
                    "paged scan: no backend available for this device/geometry".into(),
                ));
            }
            profile::record("arena:launch_enqueue", t_kernel.elapsed());
            let t_sync = Instant::now();
            stream
                .synchronize()
                .map_err(|e| candle::Error::Msg(format!("paged scan: synchronize: {e}")))?;
            profile::record("arena:launch_sync", t_sync.elapsed());
        }
        let kernel_us = t_kernel.elapsed().as_micros() as u64;

        let t_readback = Instant::now();
        let out_case = stream
            .memcpy_dtov(&d_out_case)
            .map_err(|e| candle::Error::Msg(format!("paged scan: DtoH out_case: {e}")))?;
        let out_vote = stream
            .memcpy_dtov(&d_out_vote)
            .map_err(|e| candle::Error::Msg(format!("paged scan: DtoH out_vote: {e}")))?;
        let readback_us = t_readback.elapsed().as_micros() as u64;
        Ok((out_case, out_vote, (upload_us, kernel_us, readback_us)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::content_hash::turn_stream_id;
    use crate::provenance::gpu::{BatchedGpuGallery, SegmentInput};

    /// The launchers' `-(stage * 1000 + cudaError)` codes, decoded exactly —
    /// and a bare `-cudaError`, which names no stage.
    #[test]
    fn a_launch_failure_names_its_stage() {
        assert_eq!(launch_failure(-1002), ("alloc", 2));
        assert_eq!(launch_failure(-2001), ("memset", 1));
        assert_eq!(launch_failure(-3009), ("accum_launch", 9));
        assert_eq!(launch_failure(-4009), ("finalize_launch", 9));
        assert_eq!(launch_failure(-9), ("unstaged", 9));
    }

    fn sig(fill: u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: (0..24)
                .map(|w| fill.wrapping_mul(0x9E37).wrapping_add(w))
                .collect(),
        }
    }

    /// **Probes scanned in one launch vote exactly as they do scanned alone.**
    /// The kernel writes one result per probe token and the tally splits them
    /// back by each request's token count, so batching a turn's tail and
    /// question windows into one launch — one gallery pass, one sync — must not
    /// move a single bit of either probe's votes.
    #[test]
    fn probes_batched_into_one_launch_vote_as_they_do_alone() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let a: Vec<WideQSig> = (0..40).map(|t| sig(0x5000 + t)).collect();
        let b: Vec<WideQSig> = (0..9).map(|t| sig(0x6000 + t)).collect();
        let segs = vec![
            PagedSegment {
                windows: vec![
                    PagedWindow {
                        sid: turn_stream_id(7, 0),
                        fingerprint: 70,
                        turn: &a,
                        start: 0,
                        end: 40,
                        case: 0,
                    },
                    PagedWindow {
                        sid: turn_stream_id(7, 1),
                        fingerprint: 71,
                        turn: &b,
                        start: 0,
                        end: 9,
                        case: 1,
                    },
                ],
                n_cases: 2,
            },
            PagedSegment {
                windows: vec![PagedWindow {
                    sid: turn_stream_id(8, 0),
                    fingerprint: 80,
                    turn: &a,
                    start: 5,
                    end: 30,
                    case: 0,
                }],
                n_cases: 1,
            },
        ];
        let tail = vec![
            sig(0x5000 + 3),
            sig(0xBEEF),
            sig(0x6000 + 2),
            sig(0x5000 + 31),
        ];
        let question = vec![sig(0x6000 + 7), sig(0xF00D)];
        let weights = [0.5f32, 1.0, 2.0];

        let alone_tail = arena
            .scan_weighted(&segs, &[tail.as_slice()], &weights)
            .unwrap();
        let alone_q = arena
            .scan_weighted(&segs, &[question.as_slice()], &weights)
            .unwrap();
        let both = arena
            .scan_weighted(&segs, &[tail.as_slice(), question.as_slice()], &weights)
            .unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        assert_eq!(both.len(), 2);
        assert_eq!(
            bits(&both[0]),
            bits(&alone_tail[0]),
            "the tail's votes moved"
        );
        assert_eq!(
            bits(&both[1]),
            bits(&alone_q[0]),
            "the question's votes moved"
        );
        assert!(
            both[0].iter().any(|&v| v != 0.0),
            "the tail scored nothing — the comparison proves nothing"
        );
    }

    /// **A rescan that reuses earlier rows votes exactly as a fresh scan.**
    /// The second probe shares most of its tokens with the first — a decode's
    /// next reprojection — so only its new tokens reach the kernel, and every
    /// vote must still match an arena that never saw the first probe.
    #[test]
    fn a_rescan_reusing_rows_votes_as_a_fresh_scan() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let a: Vec<WideQSig> = (0..40).map(|t| sig(0x7000 + t)).collect();
        let b: Vec<WideQSig> = (0..12).map(|t| sig(0x8000 + t)).collect();
        let segs = || {
            vec![
                PagedSegment {
                    windows: vec![PagedWindow {
                        sid: turn_stream_id(9, 0),
                        fingerprint: 90,
                        turn: &a,
                        start: 0,
                        end: 40,
                        case: 0,
                    }],
                    n_cases: 1,
                },
                PagedSegment {
                    windows: vec![
                        PagedWindow {
                            sid: turn_stream_id(10, 0),
                            fingerprint: 100,
                            turn: &b,
                            start: 0,
                            end: 6,
                            case: 0,
                        },
                        PagedWindow {
                            sid: turn_stream_id(10, 0),
                            fingerprint: 100,
                            turn: &b,
                            start: 6,
                            end: 12,
                            case: 1,
                        },
                    ],
                    n_cases: 2,
                },
            ]
        };
        let first: Vec<WideQSig> = [3, 9, 17, 0x8000 + 2, 25]
            .iter()
            .map(|&t| sig(if t < 0x8000 { 0x7000 + t } else { t }))
            .collect();
        let mut second: Vec<WideQSig> = first[1..].to_vec();
        second.extend([
            sig(0x7000 + 33),
            sig(0x8000 + 9),
            sig(0xCAFE),
            sig(0x7000 + 9),
        ]);
        let question = vec![sig(0x8000 + 2), sig(0x7000 + 3)];
        let weights = [1.0f32, 0.5, 2.0];

        let warm = GalleryArena::new(&device, 24, 3).unwrap();
        warm.scan_weighted(&segs(), &[first.as_slice()], &weights)
            .unwrap();
        let reused = warm
            .scan_weighted(&segs(), &[second.as_slice(), question.as_slice()], &weights)
            .unwrap();
        let fresh = GalleryArena::new(&device, 24, 3)
            .unwrap()
            .scan_weighted(&segs(), &[second.as_slice(), question.as_slice()], &weights)
            .unwrap();
        let bits = |v: &[Vec<f32>]| {
            v.iter()
                .map(|p| p.iter().map(|x| x.to_bits()).collect::<Vec<u32>>())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&reused), bits(&fresh));
        assert!(
            reused[0].iter().any(|&v| v != 0.0),
            "the probe scored nothing — the comparison proves nothing"
        );
    }

    /// The paged scan must be **bit-identical** to the contiguous
    /// `BatchedGpuGallery` scan over the same windows: identical bytes, identical
    /// kernel math — only the physical layout differs. Exercises multi-page turns,
    /// seam sub-windows (two windows of one turn into different cases), several
    /// files, and non-uniform group weights. Skips when no CUDA device is present.
    #[test]
    fn paged_matches_contiguous_bit_identical() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let arena = GalleryArena::new(&device, 24, 3).unwrap();

        // Turns (whole windows), owned so both paths can borrow them.
        let a0: Vec<WideQSig> = (0..3).map(|t| sig(0x1000 + t)).collect();
        let a1: Vec<WideQSig> = (0..2).map(|t| sig(0x2000 + t)).collect();
        let b0: Vec<WideQSig> = (0..40).map(|t| sig(0x3000 + t)).collect(); // 2 pages
        let c0: Vec<WideQSig> = (0..33).map(|t| sig(0x4000 + t)).collect(); // partial 2nd page

        // ── Paged segments (reference the WHOLE turn + [start,end)) ──
        let paged_segs = vec![
            PagedSegment {
                windows: vec![
                    PagedWindow {
                        sid: turn_stream_id(1, 0),
                        fingerprint: 10,
                        turn: &a0,
                        start: 0,
                        end: 3,
                        case: 0,
                    },
                    PagedWindow {
                        sid: turn_stream_id(1, 1),
                        fingerprint: 11,
                        turn: &a1,
                        start: 0,
                        end: 2,
                        case: 1,
                    },
                ],
                n_cases: 2,
            },
            PagedSegment {
                // Two sub-windows of ONE multi-page turn → different cases (a seam).
                windows: vec![
                    PagedWindow {
                        sid: turn_stream_id(2, 0),
                        fingerprint: 20,
                        turn: &b0,
                        start: 0,
                        end: 20,
                        case: 0,
                    },
                    PagedWindow {
                        sid: turn_stream_id(2, 0),
                        fingerprint: 20,
                        turn: &b0,
                        start: 20,
                        end: 40,
                        case: 1,
                    },
                ],
                n_cases: 2,
            },
            PagedSegment {
                windows: vec![PagedWindow {
                    sid: turn_stream_id(3, 0),
                    fingerprint: 30,
                    turn: &c0,
                    start: 0,
                    end: 33,
                    case: 0,
                }],
                n_cases: 1,
            },
        ];

        // ── Contiguous segments (sub-slices packed directly) ──
        let a0s = a0.as_slice();
        let a1s = a1.as_slice();
        let contig_segs = vec![
            SegmentInput {
                windows: vec![&a0s[0..3], &a1s[0..2]],
                window_case: vec![0, 1],
                n_cases: 2,
            },
            SegmentInput {
                windows: vec![&b0[0..20], &b0[20..40]],
                window_case: vec![0, 1],
                n_cases: 2,
            },
            SegmentInput {
                windows: vec![&c0[0..33]],
                window_case: vec![0],
                n_cases: 1,
            },
        ];

        let probe = vec![sig(0x1000), sig(0x3000 + 25), sig(0xBEEF), sig(0x4000 + 10)];
        let weights = [0.25f32, 1.0, 3.0];

        // Scalar-paged vs scalar-contiguous: the strict oracle (same kernel, only
        // the physical layout differs — must be bit-for-bit equal).
        let contiguous = BatchedGpuGallery::from_segments(&contig_segs)
            .unwrap()
            .scan_weighted(&device, &[probe.as_slice()], &weights)
            .unwrap();
        let paged = arena
            .scan_weighted_scalar(&paged_segs, &[probe.as_slice()], &weights)
            .unwrap();

        assert_eq!(paged.len(), contiguous.len());
        assert_eq!(paged[0].len(), contiguous[0].len(), "global case count");
        for (i, (p, c)) in paged[0].iter().zip(&contiguous[0]).enumerate() {
            assert_eq!(
                p.to_bits(),
                c.to_bits(),
                "case {i}: paged {p} vs contiguous {c} must be bit-identical"
            );
        }

        // Unweighted must also match (weights = &[] → uniform).
        let contiguous_u = BatchedGpuGallery::from_segments(&contig_segs)
            .unwrap()
            .scan(&device, &[probe.as_slice()])
            .unwrap();
        let paged_u = arena
            .scan_weighted_scalar(&paged_segs, &[probe.as_slice()], &[])
            .unwrap();
        for (i, (p, c)) in paged_u[0].iter().zip(&contiguous_u[0]).enumerate() {
            assert_eq!(
                p.to_bits(),
                c.to_bits(),
                "unweighted case {i} must be bit-identical"
            );
        }

        // The auto backend (BMMA on this hardware) must agree with the scalar
        // oracle on the same fixture — full bit equality (integer statistics are
        // identical by construction; the float finalize is the shared bdp_vote).
        let auto = arena
            .scan_weighted(&paged_segs, &[probe.as_slice()], &weights)
            .unwrap();
        for (i, (a, c)) in auto[0].iter().zip(&contiguous[0]).enumerate() {
            assert_eq!(
                a.to_bits(),
                c.to_bits(),
                "case {i}: auto backend {a} vs scalar {c} must be bit-identical"
            );
        }
    }

    /// Governor relief path: a scan → evict EVERYTHING (as the cheap-rung relief
    /// closure does under VRAM pressure) → the next scan rebuilds residency from
    /// the same sigs and reproduces the result bit-identically. Skips without CUDA.
    #[test]
    fn eviction_then_rebuild_is_stable() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let arena = GalleryArena::new(&device, 24, 3).unwrap();

        let t0: Vec<WideQSig> = (0..50).map(|t| sig(0xA00 + t)).collect(); // 2 pages
        let t1: Vec<WideQSig> = (0..12).map(|t| sig(0xB00 + t)).collect();
        let t2: Vec<WideQSig> = (0..7).map(|t| sig(0xC00 + t)).collect();
        let segs = vec![
            PagedSegment {
                windows: vec![
                    PagedWindow {
                        sid: turn_stream_id(9, 0),
                        fingerprint: 1,
                        turn: &t0,
                        start: 0,
                        end: 50,
                        case: 0,
                    },
                    PagedWindow {
                        sid: turn_stream_id(9, 1),
                        fingerprint: 2,
                        turn: &t1,
                        start: 0,
                        end: 12,
                        case: 1,
                    },
                ],
                n_cases: 2,
            },
            PagedSegment {
                windows: vec![PagedWindow {
                    sid: turn_stream_id(9, 2),
                    fingerprint: 3,
                    turn: &t2,
                    start: 0,
                    end: 7,
                    case: 0,
                }],
                n_cases: 1,
            },
        ];
        let probe = vec![sig(0xA00 + 5), sig(0xC00 + 2), sig(0x999)];
        let weights = [1.0f32, 0.5, 2.0];

        let first = arena
            .scan_weighted(&segs, &[probe.as_slice()], &weights)
            .unwrap();
        let before = arena.resident_turns();
        assert_eq!(before, 3, "three turns resident after the first scan");
        assert!(arena.resident_bytes() > 0);

        // The scan unpinned its working set, so relief can evict everything.
        let freed = arena.evict_lru(u64::MAX);
        assert!(freed > 0, "eviction freed VRAM");
        assert_eq!(arena.resident_turns(), 0, "all turns evicted");

        // Re-scan → residency rebuilds → bit-identical result.
        let second = arena
            .scan_weighted(&segs, &[probe.as_slice()], &weights)
            .unwrap();
        assert_eq!(arena.resident_turns(), before, "rebuilt the same residency");
        assert_eq!(first[0].len(), second[0].len());
        for (i, (a, b)) in first[0].iter().zip(&second[0]).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "case {i}: rebuild after eviction must reproduce the scan bit-identically"
            );
        }
    }

    /// The index cache is fingerprint-KEYED (not a single slot): two distinct
    /// segment sets scanned alternately both stay cached, so neither rebuilds
    /// after its first scan — the multi-belief-group reprojection pattern. Each
    /// still produces the correct result. Skips without CUDA.
    #[test]
    fn index_cache_keyed_by_fingerprint_no_thrash() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let arena = GalleryArena::new(&device, 24, 3).unwrap();

        // Two distinct "groups" — different turns/files → different fingerprints.
        let a: Vec<WideQSig> = (0..20).map(|t| sig(0xA00 + t)).collect();
        let b: Vec<WideQSig> = (0..15).map(|t| sig(0xB00 + t)).collect();
        let seg_a = vec![PagedSegment {
            windows: vec![PagedWindow {
                sid: turn_stream_id(7, 0),
                fingerprint: 1,
                turn: &a,
                start: 0,
                end: 20,
                case: 0,
            }],
            n_cases: 1,
        }];
        let seg_b = vec![PagedSegment {
            windows: vec![PagedWindow {
                sid: turn_stream_id(8, 0),
                fingerprint: 2,
                turn: &b,
                start: 0,
                end: 15,
                case: 0,
            }],
            n_cases: 1,
        }];
        let probe = vec![sig(0xA00 + 3), sig(0xB00 + 3)];

        // Prime both caches.
        let a1 = arena
            .scan_weighted(&seg_a, &[probe.as_slice()], &[])
            .unwrap();
        let b1 = arena
            .scan_weighted(&seg_b, &[probe.as_slice()], &[])
            .unwrap();
        // Interleave: with a single-slot cache these would each rebuild; keyed,
        // they both hit and reproduce their first result bit-identically. (No
        // residency mutation between scans, so the generation holds.)
        for _ in 0..3 {
            let a2 = arena
                .scan_weighted(&seg_a, &[probe.as_slice()], &[])
                .unwrap();
            let b2 = arena
                .scan_weighted(&seg_b, &[probe.as_slice()], &[])
                .unwrap();
            assert_eq!(a1, a2, "group A cached result must be stable");
            assert_eq!(b1, b2, "group B cached result must be stable");
        }
        assert_eq!(arena.resident_turns(), 2, "both turns resident, no churn");
    }

    /// **Another conversation's upload leaves this index valid; a change to
    /// one of its own turns does not.** The index cache used to be validated by
    /// a device-wide generation, so ingest sealing unrelated turns rebuilt a
    /// dialogue's whole index every reprojection. Skips without CUDA.
    #[test]
    fn an_unrelated_upload_keeps_the_index_and_a_reseal_drops_it() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let a: Vec<WideQSig> = (0..20).map(|t| sig(0xA00 + t)).collect();
        let seg_a = vec![PagedSegment {
            windows: vec![PagedWindow {
                sid: turn_stream_id(7, 0),
                fingerprint: 1,
                turn: &a,
                start: 0,
                end: 20,
                case: 0,
            }],
            n_cases: 1,
        }];
        let probe = vec![sig(0xA00 + 3)];
        let fp = fingerprint_segments(&seg_a);
        arena
            .scan_weighted(&seg_a, &[probe.as_slice()], &[])
            .unwrap();

        // Another conversation's turn uploads: the device churns, this index
        // does not.
        let other: Vec<WideQSig> = (0..10).map(|t| sig(0xC00 + t)).collect();
        let gen = arena.residency_gen();
        arena
            .ensure_resident(turn_stream_id(9, 0), &other, 3)
            .unwrap();
        assert!(arena.residency_gen() > gen, "the upload is counted");
        let kept = arena
            .reuse_index(fp)
            .expect("an unrelated upload keeps the index");
        for &sid in &kept.pinned_sids {
            arena.unpin(sid);
        }

        // This index's own turn is re-sealed under a new fingerprint: its pages
        // are replaced, so the cached addresses are gone with them.
        arena.ensure_resident(turn_stream_id(7, 0), &a, 2).unwrap();
        assert!(
            arena.reuse_index(fp).is_none(),
            "a re-sealed turn invalidates every index over it"
        );
    }

    /// **A compaction must be invisible to the scan.** Score the same segments
    /// before and after packing the pages and require the vectors to be
    /// bit-identical.
    ///
    /// This is the test that matters for the pass, because it exercises the real
    /// consumer — the kernel dereferencing the page addresses in `PagedIndex` —
    /// rather than the bookkeeping that produced them. The two ways a move goes
    /// wrong both land here and nowhere else: a page copied to the wrong place
    /// changes the scores, and a page moved without bumping `residency_gen` leaves
    /// the cached index pointing at the vacated address, which the next scan reads
    /// as whatever the arena has since put there. Neither is a fault; both are
    /// quietly wrong retrieval.
    #[test]
    fn a_compaction_leaves_every_score_identical() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        // Several turns of differing length, so the runs are ragged.
        let turns: Vec<Vec<WideQSig>> = (0..5u64)
            .map(|t| (0..(14 + t * 9)).map(|k| sig((t << 32) + k)).collect())
            .collect();
        let windows: Vec<PagedWindow<'_>> = turns
            .iter()
            .enumerate()
            .map(|(i, turn)| PagedWindow {
                sid: turn_stream_id(40 + i as u64, 0),
                fingerprint: 100 + i as u64,
                turn,
                start: 0,
                end: turn.len(),
                case: i,
            })
            .collect();
        let segments = vec![PagedSegment {
            windows,
            n_cases: turns.len(),
        }];
        let probe = vec![sig(1 << 32), sig((3u64 << 32) + 5), sig(0xFEED)];

        let before = arena
            .scan_weighted(&segments, &[probe.as_slice()], &[])
            .unwrap();

        // Scatter: drop two turns from the middle of the corpus, leaving holes
        // under live pages, then pack. `scan_weighted` unpinned on its way out, so
        // every survivor is movable.
        arena.drop_turn(turn_stream_id(41, 0));
        arena.drop_turn(turn_stream_id(43, 0));
        let report = arena.compact(0).unwrap();
        assert!(report.moved > 0, "nothing moved, so nothing is proven");

        // The dropped turns rebuild on demand; the survivors are read at their new
        // addresses. Either way every score must reproduce exactly.
        let after = arena
            .scan_weighted(&segments, &[probe.as_slice()], &[])
            .unwrap();
        assert_eq!(
            before, after,
            "compaction changed the scan's answer — a page moved to the wrong \
             place, or a cached index outlived the move",
        );
    }

    fn xorshift(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    fn rand_sig(state: &mut u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: (0..24).map(|_| xorshift(state)).collect(),
        }
    }

    /// Adversarial BMMA-vs-scalar parity over a large randomized corpus that
    /// hits every structural edge at once: 1-token exchanges, case-id GAPS
    /// (empty cases), windows supplied in DESCENDING case order (exercises the
    /// build_index sort the BMMA dense-rank relies on), out-of-range cases
    /// (dropped), seam sub-windows of one multi-page turn split across cases,
    /// segments that straddle 64-token chunk boundaries, and probe/token counts
    /// that are not multiples of the 8/32/64 tile shapes. The two backends'
    /// integer statistics are identical by construction and the float finalize
    /// is shared, so the votes must be bit-for-bit equal. Skips without CUDA
    /// or on hardware without b1 BMMA.
    #[test]
    fn bmma_matches_scalar_on_adversarial_corpus() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let (has_bmma, has_imma) = arena.tensor_caps();
        if !has_bmma || !has_imma {
            return; // needs both tensor backends for the 3-way comparison
        }

        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        // Turn lengths spanning page boundaries and tile tails.
        let lens = [1usize, 3, 7, 8, 31, 32, 33, 40, 63, 64, 65, 100, 129, 200];
        let turns: Vec<Vec<WideQSig>> = (0..40)
            .map(|i| {
                let n = lens[i % lens.len()];
                (0..n).map(|_| rand_sig(&mut rng)).collect()
            })
            .collect();

        let mut segs: Vec<PagedSegment> = Vec::new();
        let mut next_turn = 0usize;
        for si in 0..14usize {
            let n_cases = 2 + (si % 7); // 2..8 cases per segment
            let mut windows = Vec::new();
            // 2-4 turns per segment.
            for wi in 0..(2 + si % 3) {
                let ti = (next_turn + wi) % turns.len();
                let t = &turns[ti];
                // DESCENDING case ids (build_index must sort), with deliberate
                // gaps (cases that never receive a window stay empty).
                let case = (n_cases - 1).saturating_sub(wi * 2 % n_cases);
                if t.len() > 8 {
                    // Seam split: two sub-windows of one turn, different cases.
                    let mid = t.len() / 2;
                    windows.push(PagedWindow {
                        sid: turn_stream_id(500 + si as u64, ti as u32),
                        fingerprint: (si * 100 + ti) as u64,
                        turn: t,
                        start: 0,
                        end: mid,
                        case,
                    });
                    windows.push(PagedWindow {
                        sid: turn_stream_id(500 + si as u64, ti as u32),
                        fingerprint: (si * 100 + ti) as u64,
                        turn: t,
                        start: mid,
                        end: t.len(),
                        case: case.saturating_sub(1),
                    });
                } else {
                    windows.push(PagedWindow {
                        sid: turn_stream_id(500 + si as u64, ti as u32),
                        fingerprint: (si * 100 + ti) as u64,
                        turn: t,
                        start: 0,
                        end: t.len(),
                        case,
                    });
                }
            }
            // An out-of-range case that must be dropped by both backends.
            let tdrop = &turns[next_turn % turns.len()];
            windows.push(PagedWindow {
                sid: turn_stream_id(500 + si as u64, (next_turn % turns.len()) as u32),
                fingerprint: (si * 100 + next_turn % turns.len()) as u64,
                turn: tdrop,
                start: 0,
                end: tdrop.len().min(4),
                case: n_cases + 3,
            });
            next_turn += 3;
            segs.push(PagedSegment { windows, n_cases });
        }

        // Probes exercising tile tails: 1, 9, and 100 query tokens.
        let probes_owned: Vec<Vec<WideQSig>> = [1usize, 9, 100]
            .iter()
            .map(|&n| (0..n).map(|_| rand_sig(&mut rng)).collect())
            .collect();

        for (pi, probe) in probes_owned.iter().enumerate() {
            for weights in [&[] as &[f32], &[0.25, 1.0, 3.0]] {
                let scalar = arena
                    .scan_weighted_scalar(&segs, &[probe.as_slice()], weights)
                    .unwrap();
                let auto = arena
                    .scan_weighted(&segs, &[probe.as_slice()], weights)
                    .unwrap();
                assert_eq!(scalar[0].len(), auto[0].len(), "probe {pi}: case count");
                for (ci, (s, b)) in scalar[0].iter().zip(&auto[0]).enumerate() {
                    assert_eq!(
                        s.to_bits(),
                        b.to_bits(),
                        "probe {pi} case {ci}: scalar {s} vs auto {b} must be bit-identical"
                    );
                }
                // The INT8 (IMMA) backend — the Blackwell production path,
                // parity-proven here on Ada — must also match bit-for-bit.
                let imma = arena
                    .scan_weighted_imma(&segs, &[probe.as_slice()], weights)
                    .unwrap();
                for (ci, (s, m)) in scalar[0].iter().zip(&imma[0]).enumerate() {
                    assert_eq!(
                        s.to_bits(),
                        m.to_bits(),
                        "probe {pi} case {ci}: scalar {s} vs IMMA {m} must be bit-identical"
                    );
                }
            }
        }
    }
}
