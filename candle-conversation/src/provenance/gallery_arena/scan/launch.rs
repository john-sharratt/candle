//! Launches the paged scan kernel over an index, and reads back the votes or the
//! statistics they are made of.
//!
//! The launcher tries the device's backends fastest first, and a launch the host
//! gate declines passes to the next backend. A CUDA error left pending by an
//! earlier launch is taken and named before any backend runs, so it is not blamed
//! on the one that follows. Only the probe is uploaded for each launch, and the
//! output is drained before the launch returns. A launch that fails part-way
//! drains the stream before the caller unpins the turns it reads.

use super::*;

/// What producing a scan's rows cost: the launch's timing, the probe tokens it
/// scored (the rest were read from [`ProbeRows`]), and the row cache's own
/// time.
#[derive(Default)]
pub(super) struct Launched {
    pub(super) timing: LaunchTiming,
    pub(super) fresh: usize,
    pub(super) rows_us: u64,
}

/// A tensor launcher's negative return code, decoded: `(stage, cudaError)`.
///
/// The launchers return `-(stage * 1000 + cudaError)`, where the stage names
/// WHICH of their CUDA calls rejected the work (1 alloc, 2 memset, 3 accumulate
/// launch, 4 finalize launch). A bare `-cudaError` names none of them and reads
/// as `unstaged`.
pub(super) fn launch_failure(rc: i32) -> (&'static str, i32) {
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

impl GalleryArena {
    /// The statistics launch over `idx` for `probes`, each probe token narrowed
    /// to the groups the index scans.
    pub(super) fn stats_of(
        &self,
        dev: &CudaDevice,
        idx: &PagedIndex,
        probes: &[&[WideQSig]],
        force: Option<PagedBackend>,
    ) -> Result<TokenStats> {
        let wpt = self.wpt();
        let gw = self.gw();
        let mut probe_words: Vec<u64> = Vec::new();
        let mut per_probe: Vec<usize> = Vec::with_capacity(probes.len());
        for probe in probes {
            let mut n = 0usize;
            for tok in *probe {
                if tok.words.len() >= wpt {
                    for g in idx.groups.iter() {
                        probe_words.extend_from_slice(&tok.words[g * gw..(g + 1) * gw]);
                    }
                    n += 1;
                }
            }
            per_probe.push(n);
        }
        let n_tokens: usize = per_probe.iter().sum();
        let Some(d_idx) = idx.device.as_ref() else {
            return Ok(TokenStats::none(per_probe));
        };
        if n_tokens == 0 || idx.n_segments == 0 || idx.n_cases == 0 || idx.groups.is_empty() {
            return Ok(TokenStats::none(per_probe));
        }
        let (out, timing) =
            self.launch_kernel(dev, d_idx, idx, &probe_words, n_tokens, force, true)?;
        let LaunchOut::Stats(rows) = out else {
            return Err(candle::Error::Msg(
                "paged scan: a statistics launch wrote votes".into(),
            ));
        };
        scan_split::add(|s| {
            s.launches += 1;
            s.fresh_tokens += n_tokens as u64;
            s.upload_us += timing.upload_us;
            s.kernel_us += timing.kernel_us;
            s.kernel_gpu_us += timing.kernel_gpu_us;
            s.readback_us += timing.readback_us;
        });
        Ok(TokenStats {
            rows,
            per_probe,
            groups: idx.groups.iter().collect(),
            seg_tokens: idx
                .seg_tok
                .windows(2)
                .map(|w| (w[1] - w[0]) as u64)
                .collect(),
        })
    }

    pub(super) fn launch_paged(
        &self,
        idx: &PagedIndex,
        weighed: &[GroupSet],
        probes: &[&[WideQSig]],
        segment_weights: &[&[f32]],
        force: Option<PagedBackend>,
    ) -> Result<Vec<Vec<f32>>> {
        let (raw, launched) = self.token_votes(idx, weighed, probes, force)?;
        let t_tally = Instant::now();
        let votes = needle_tally_segments(
            &raw.case,
            &raw.vote,
            &raw.per_probe,
            &idx.seg_case,
            raw.n_groups,
            idx.n_segments,
            idx.n_cases,
            idx.max_k,
            segment_weights,
            &idx.seg_law,
        );
        let LaunchTiming {
            upload_us,
            kernel_us,
            kernel_gpu_us,
            readback_us,
        } = launched.timing;
        profile::record("arena:launch_upload", Duration::from_micros(upload_us));
        profile::record("arena:launch_kernel", Duration::from_micros(kernel_us));
        profile::record("arena:kernel_gpu", Duration::from_micros(kernel_gpu_us));
        profile::record("arena:launch_readback", Duration::from_micros(readback_us));
        profile::record("arena:launch_tally", t_tally.elapsed());
        let tally_us = t_tally.elapsed().as_micros() as u64;
        scan_split::add(|s| {
            s.launches += u64::from(launched.fresh > 0);
            s.fresh_tokens += launched.fresh as u64;
            s.upload_us += upload_us;
            s.kernel_gpu_us += kernel_gpu_us;
            s.kernel_us += kernel_us;
            s.readback_us += readback_us;
            s.rows_us += launched.rows_us;
            s.tally_us += tally_us;
            if launched.fresh > 0 {
                s.note_launch(launched.fresh as u64, idx.n_tokens as u64, kernel_gpu_us);
            }
        });
        // Where a launch's time goes: the probe upload, the kernel (to its
        // synchronize), the vote readback, and the host tally — over the
        // `scored` tokens no earlier launch over this index had scored.
        tracing::trace!(
            target: "candle_conversation::provenance::gallery_arena",
            tokens = idx.n_tokens,
            probe_tokens = raw.per_probe.iter().sum::<usize>(),
            scored = launched.fresh,
            segments = idx.n_segments,
            cases = idx.n_cases,
            upload_us,
            kernel_us,
            readback_us,
            tally_us,
            "arena launch"
        );
        Ok(votes)
    }

    /// Every probe token's raw `(case, vote)` row against `idx`, for segments
    /// weighing `weighed`: the tokens no earlier launch over this index scored
    /// go to the kernel, the rest are read back from the rows it kept
    /// ([`ProbeRows`]). Returns the rows and what the launch cost.
    pub(super) fn token_votes(
        &self,
        idx: &PagedIndex,
        weighed: &[GroupSet],
        probes: &[&[WideQSig]],
        force: Option<PagedBackend>,
    ) -> Result<(TokenVotes, Launched)> {
        let dev = match self.device() {
            Device::Cuda(d) => d,
            _ => return Err(candle::Error::Msg("paged scan requires CUDA".into())),
        };
        let wpt = self.wpt();
        let n_groups = self.n_groups();
        let gw = self.gw();
        let n_cases = idx.n_cases;
        let n_segments = idx.n_segments;
        let none = || {
            (
                TokenVotes::none(probes.len(), n_groups, n_segments, idx.max_k),
                Launched::default(),
            )
        };

        let Some(d_idx) = idx.device.as_ref() else {
            return Ok(none());
        };
        if probes.is_empty() || n_segments == 0 || n_cases == 0 || idx.groups.is_empty() {
            return Ok(none());
        }

        // Batch probes (token-major, full-width only — the reference's filter),
        // each token narrowed to the groups the launch scans: the kernel reads
        // `launch × gw` words a token, as the index's pages give them.
        let lwpt = idx.groups.len() * gw;
        let mut probe_words: Vec<u64> = Vec::new();
        let mut per_req_tokens: Vec<usize> = Vec::with_capacity(probes.len());
        for probe in probes {
            let mut cnt = 0usize;
            for tok in *probe {
                if tok.words.len() >= wpt {
                    for g in idx.groups.iter() {
                        probe_words.extend_from_slice(&tok.words[g * gw..(g + 1) * gw]);
                    }
                    cnt += 1;
                }
            }
            per_req_tokens.push(cnt);
        }
        if probe_words.is_empty() {
            return Ok(none());
        }
        let wpt = lwpt;

        // Only the tokens no earlier launch over this index scored go to the
        // kernel; the rest are read back from the rows it kept (`ProbeRows`).
        // Held to the end of the launch, so a concurrent scan of the same index
        // cannot start the cache over between this plan and the assembly.
        let t_plan = Instant::now();
        let mut rows = idx.rows.lock().unwrap_or_else(|e| e.into_inner());
        let fresh = rows.plan(&probe_words, wpt);
        profile::record("arena:rows_plan", t_plan.elapsed());
        let mut rows_us = t_plan.elapsed().as_micros() as u64;
        let timing = if fresh.is_empty() {
            LaunchTiming::default()
        } else {
            let tokens: Vec<&[u64]> = fresh
                .iter()
                .map(|&t| &probe_words[t * wpt..(t + 1) * wpt])
                .collect();
            let words: Vec<u64> = tokens.concat();
            let (out, timing) =
                self.launch_kernel(dev, d_idx, idx, &words, tokens.len(), force, false)?;
            let LaunchOut::Votes { case, vote } = out else {
                return Err(candle::Error::Msg(
                    "paged scan: a votes launch wrote statistics".into(),
                ));
            };
            let t_insert = Instant::now();
            rows.insert(&tokens, &case, &vote);
            profile::record("arena:rows_insert", t_insert.elapsed());
            rows_us += t_insert.elapsed().as_micros() as u64;
            timing
        };
        let t_assemble = Instant::now();
        let (out_case, out_vote) = rows
            .assemble(&probe_words, wpt)
            .expect("every probe token has a row once its launch is in");
        drop(rows);
        profile::record("arena:rows_assemble", t_assemble.elapsed());
        rows_us += t_assemble.elapsed().as_micros() as u64;
        // Rows for every fold group, as every reader of raw votes indexes them:
        // a group a segment does not weigh voted for nothing in it, though the
        // index may scan it for another segment or an earlier reader.
        let (out_case, out_vote) = widen_rows(
            &out_case, &out_vote, idx.groups, weighed, n_groups, idx.max_k,
        );

        Ok((
            TokenVotes {
                case: out_case,
                vote: out_vote,
                per_probe: per_req_tokens,
                n_groups,
                n_segments,
                k: idx.max_k,
            },
            Launched {
                timing,
                fresh: fresh.len(),
                rows_us,
            },
        ))
    }

    /// One kernel launch over `n_probe_tokens` probe tokens (`probe_words`,
    /// `wpt` words each) against `idx`, on the fastest backend the device and
    /// geometry admit. Returns the raw output — one row per probe token, its
    /// votes, or with `stats` the statistics they are made of — and where the
    /// launch's time went.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_kernel(
        &self,
        dev: &CudaDevice,
        d_idx: &DeviceIndex,
        idx: &PagedIndex,
        probe_words: &[u64],
        n_probe_tokens: usize,
        force: Option<PagedBackend>,
        stats: bool,
    ) -> Result<(LaunchOut, LaunchTiming)> {
        // The launch's own geometry: its groups, as the index addresses them
        // and the probe was narrowed to.
        let n_groups = idx.groups.len();
        let gw = self.gw();
        let wpt = n_groups * gw;
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
        // on the device (`DeviceIndex`), and the records stay resident. The
        // probe and the two outputs ride the device's synced scratch, which
        // this launch drains before it returns.
        let t_upload = Instant::now();

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

        let max_k = idx.max_k;
        // Votes: `max_k` a row. Statistics: one a row, and no votes.
        let n_rows = n_probe_tokens * n_groups * n_segments;
        let n_out = if stats { 0 } else { n_rows * max_k };
        let n_stats = if stats { n_rows } else { 0 };
        let upload_us = t_upload.elapsed().as_micros() as u64;
        let t_kernel = Instant::now();
        let mut out_case = vec![0i32; n_out];
        let mut out_vote = vec![0f32; n_out];
        let mut out_stats = vec![BdpStats::default(); n_stats];
        let mut readback_us = 0u64;
        let mut kernel_gpu_us = 0u64;
        // The outputs are written in full by the kernel (one block per tile ×
        // group × segment), so their room is neither zeroed nor uploaded.
        let launch = |at: &[u64], room: u64| -> candle::Result<()> {
            let (p_case, _g1) = d_idx.entry_case.device_ptr(&stream);
            let p_probe = at[0];
            let (p_seg_slot, _g2) = d_idx.seg_slot.device_ptr(&stream);
            let (p_seg_tok, _g3) = d_idx.seg_tok.device_ptr(&stream);
            let (p_seg_case, _g4) = d_idx.seg_case.device_ptr(&stream);
            let (p_page_ptr, _g5) = d_idx.page_ptr.device_ptr(&stream);
            let (p_entry_page, _g6) = d_idx.entry_page.device_ptr(&stream);
            let (p_entry_slots, _g7) = d_idx.entry_slots.device_ptr(&stream);
            let (p_seg_top_k, _g8) = d_idx.seg_top_k.device_ptr(&stream);
            let p_out_case = room;
            let p_out_vote = room + (n_out * std::mem::size_of::<i32>()) as u64;
            let p_out_stats: *mut BdpStats = if stats {
                room as *mut BdpStats
            } else {
                std::ptr::null_mut()
            };
            let (p_votes_case, p_votes_vote) = if stats {
                (std::ptr::null_mut(), std::ptr::null_mut())
            } else {
                (p_out_case as *mut i32, p_out_vote as *mut f32)
            };
            // The kernel's own device time, apart from whatever the shared
            // stream had queued ahead of it — the wall-clock sync below counts
            // both. Timing events: `None` flags would make them untimed.
            let ev_start = stream.record_event(Some(CU_EVENT_DEFAULT)).ok();
            let mut launched = false;
            // The rungs whose host-side gate declined, named if none launched.
            let mut declined: Vec<PagedBackend> = Vec::new();
            // The rungs that failed on the device: backend, stage, CUDA error.
            let mut failed = Vec::new();
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
                                p_entry_page as *const u32,
                                p_entry_slots as *const u32,
                                p_seg_top_k as *const i32,
                                idx.n_slots as i32,
                                n_probe_tokens as i32,
                                n_groups as i32,
                                n_segments as i32,
                                n_cases as i32,
                                gw as i32,
                                wpt as i32,
                                max_k as i32,
                                p_votes_case,
                                p_votes_vote,
                                p_out_stats,
                                stream.cu_stream() as *mut c_void,
                            )
                        };
                        if rc == 0 {
                            profile::record(
                                match backend {
                                    PagedBackend::Bmma => "arena:backend_bmma",
                                    _ => "arena:backend_imma",
                                },
                                Duration::ZERO,
                            );
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
                            declined.push(backend);
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
                        failed.push((backend, stage_name, cuda_err));
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
                                p_seg_slot as *const i32,
                                p_seg_tok as *const i32,
                                p_seg_case as *const i32,
                                p_page_ptr as *const u64,
                                p_entry_page as *const u32,
                                p_entry_slots as *const u32,
                                p_seg_top_k as *const i32,
                                n_probe_tokens as i32,
                                n_groups as i32,
                                n_segments as i32,
                                idx.max_seg_cases as i32,
                                gw as i32,
                                wpt as i32,
                                max_k as i32,
                                p_votes_case,
                                p_votes_vote,
                                p_out_stats,
                                stream.cu_stream() as *mut c_void,
                            );
                        }
                        profile::record("arena:backend_scalar", Duration::ZERO);
                        launched = true;
                        break;
                    }
                }
            }
            if !launched {
                return Err(candle::Error::Msg(format!(
                    "paged scan: no backend available for this device/geometry \
                     (declined: {declined:?}, failed: {failed:?}, of {candidates:?})"
                )));
            }
            profile::record("arena:launch_enqueue", t_kernel.elapsed());
            // The kernel's time is the events', so nothing waits for it alone:
            // the copies follow it in stream order and one wait covers both.
            let ev_stop = stream.record_event(Some(CU_EVENT_DEFAULT)).ok();
            let t_readback = Instant::now();
            // SAFETY: both ranges are this launch's outputs inside the scratch,
            // held until the closure returns; the copies are waited for below.
            unsafe {
                if stats {
                    memcpy_dtoh_async(&mut out_stats, room, stream.cu_stream()).map_err(|e| {
                        candle::Error::Msg(format!("paged scan: DtoH out_stats: {e}"))
                    })?;
                } else {
                    memcpy_dtoh_async(&mut out_case, p_out_case, stream.cu_stream()).map_err(
                        |e| candle::Error::Msg(format!("paged scan: DtoH out_case: {e}")),
                    )?;
                    memcpy_dtoh_async(&mut out_vote, p_out_vote, stream.cu_stream()).map_err(
                        |e| candle::Error::Msg(format!("paged scan: DtoH out_vote: {e}")),
                    )?;
                }
            }
            stream
                .synchronize()
                .map_err(|e| candle::Error::Msg(format!("paged scan: readback: {e}")))?;
            if let (Some(start), Some(stop)) = (&ev_start, &ev_stop) {
                if let Ok(ms) = start.elapsed_ms(stop) {
                    kernel_gpu_us = (f64::from(ms) * 1000.0) as u64;
                }
            }
            // The wait covered the kernel's tail as well as the copies; the
            // readback is what is left once the kernel's own time is taken off.
            readback_us = (t_readback.elapsed().as_micros() as u64).saturating_sub(kernel_gpu_us);
            Ok(())
        };
        dev.with_synced_upload(
            &stream,
            &[segment(probe_words)],
            n_out * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>())
                + n_stats * std::mem::size_of::<BdpStats>(),
            launch,
        )?;
        let kernel_us = (t_kernel.elapsed().as_micros() as u64).saturating_sub(readback_us);
        let out = if stats {
            LaunchOut::Stats(out_stats)
        } else {
            LaunchOut::Votes {
                case: out_case,
                vote: out_vote,
            }
        };
        Ok((
            out,
            LaunchTiming {
                upload_us,
                kernel_us,
                kernel_gpu_us,
                readback_us,
            },
        ))
    }
}

/// What one launch wrote back: its votes, or in the statistics mode the
/// statistics they are made of — one [`BdpStats`] a row.
pub(super) enum LaunchOut {
    Votes { case: Vec<i32>, vote: Vec<f32> },
    Stats(Vec<BdpStats>),
}
